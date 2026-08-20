/* Native Qwen3.8 + DSpark CUDA-graph decode gate.
 *
 * This intentionally uses the device-only controller rather than the legacy
 * host speculative loop.  It pre-fills once on the host, captures/binds the
 * native target graph, then queues exactly 33 gamma=7 replays on one stream.
 * Per-cycle evidence is copied device-to-device into a history ring; the only
 * device-to-host transfer and synchronization happen after every replay has
 * been queued.
 */

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"
#include "axiom/qwen38_dspark_compute.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_speculative.h"

namespace {

constexpr uint32_t kGraphCycles = 33u;
constexpr uint32_t kDefaultMaxContext = 2048u;
constexpr uint32_t kValidationTokens[] = {
        271u, 248068u, 198u, 760u, 1156u, 369u, 40719u, 728u,
};
constexpr uint32_t kValidationTokenCount =
        static_cast<uint32_t>(sizeof(kValidationTokens) / sizeof(kValidationTokens[0]));

static_assert(kValidationTokenCount == AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH,
              "the temporal validation input is width eight");

struct CycleHistory {
    uint32_t proposal[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS];
    uint32_t accepted_prefix;
    uint32_t continuation_token;
    uint32_t async_status;
    uint32_t next_position;
};

static_assert(sizeof(CycleHistory) ==
                      (AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS + 4u) * sizeof(uint32_t),
              "history layout must remain one compact D2H copy");

int fail(const char *stage, int status = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-speculative-graph-gate: %s%s%s\n",
                 stage, status == AXIOM_OK ? "" : ": ",
                 status == AXIOM_OK ? "" : axiom_status_string(status));
    return 1;
}

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

bool read_prompt_file(const char *path, std::string *out) {
    if (!path || !*path || !out) return false;
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) return false;
    try {
        out->assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    } catch (...) {
        return false;
    }
    return input.eof() || !input.bad();
}

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

bool append_device_copy(
        uint32_t *destination,
        const uint32_t *source,
        size_t count,
        cudaStream_t stream,
        int *out_status) {
    if (!destination || !source || !out_status) return false;
    const cudaError_t status = cudaMemcpyAsync(
            destination, source, count * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        *out_status = cuda_status(status);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::fprintf(
                stderr,
                "usage: %s TARGET_DIR DSPARK_DIR [PROMPT_OR_@PROMPT_FILE] [MAX_CONTEXT]\n",
                argv[0]);
        return 2;
    }
    uint32_t max_context = kDefaultMaxContext;
    const char *prompt_argument = nullptr;
    if (argc == 4) {
        uint32_t legacy_max_context = 0u;
        if (parse_u32(argv[3], &legacy_max_context)) {
            max_context = legacy_max_context;
        } else {
            prompt_argument = argv[3];
        }
    } else if (argc == 5) {
        prompt_argument = argv[3];
        if (!parse_u32(argv[4], &max_context)) return fail("invalid MAX_CONTEXT");
    }
    if (max_context == 0u) return fail("invalid MAX_CONTEXT");
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        return fail("disable duplicate generic residency");
    }

    std::string prompt_text;
    bool prompt_from_file = false;
    if (prompt_argument) {
        const char *file_path = prompt_argument[0] == '@' ? prompt_argument + 1 : prompt_argument;
        if (read_prompt_file(file_path, &prompt_text)) {
            prompt_from_file = true;
        } else if (prompt_argument[0] == '@') {
            return fail("prompt file read");
        } else {
            prompt_text.assign(prompt_argument);
        }
    }

    axiom_tokenizer *tokenizer = nullptr;
    axiom_qwen38_model *target = nullptr;
    axiom_runtime *draft_runtime = nullptr;
    axiom_model *draft_checkpoint = nullptr;
    axiom_qwen38_dspark *draft = nullptr;
    axiom_qwen38_dspark_compute *compute = nullptr;
    axiom_qwen38_speculative *speculative = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t replay_begin = nullptr;
    cudaEvent_t replay_end = nullptr;
    uint32_t *seed_anchor_device = nullptr;
    uint32_t *seed_position_device = nullptr;
    CycleHistory *history_device = nullptr;
    CycleHistory *history_host = nullptr;
    bool stream_synchronized = false;
    const bool profiler_capture = std::getenv("AXIOM_QWEN38_NSYS_CAPTURE") != nullptr;
    bool profiler_started = false;
    int rc = AXIOM_OK;
    const char *stage = "backend_available";

    if (axiom_qwen38_speculative_backend_available() != 1) {
        rc = AXIOM_ERR_NOT_IMPLEMENTED;
    }

    std::vector<uint32_t> prompt_ids;
    uint32_t prompt_count = 0u;
    if (rc == AXIOM_OK && prompt_argument) {
        axiom_tokenizer_config tokenizer_config{};
        tokenizer_config.abi_version = AXIOM_ABI_VERSION;
        tokenizer_config.path = argv[1];
        tokenizer_config.name = "qwen3.8";
        tokenizer_config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
        stage = "tokenizer_open";
        rc = axiom_tokenizer_open(&tokenizer, &tokenizer_config);
        if (rc == AXIOM_OK) {
            try {
                prompt_ids.resize(max_context);
            } catch (...) {
                rc = AXIOM_ERR_BUDGET;
            }
        }
        if (rc == AXIOM_OK) stage = "tokenizer_encode";
        if (rc == AXIOM_OK) {
            rc = axiom_tokenizer_encode_text(
                    tokenizer, prompt_text.c_str(), prompt_ids.data(),
                    static_cast<uint32_t>(prompt_ids.size()), &prompt_count);
        }
        if (rc == AXIOM_OK) prompt_ids.resize(prompt_count);
    } else if (rc == AXIOM_OK) {
        try {
            prompt_ids.assign(kValidationTokens, kValidationTokens + kValidationTokenCount);
            prompt_count = kValidationTokenCount;
        } catch (...) {
            rc = AXIOM_ERR_BUDGET;
        }
    }
    if (rc == AXIOM_OK &&
        (prompt_count == 0u ||
         static_cast<uint64_t>(prompt_count) +
                         static_cast<uint64_t>(kGraphCycles) *
                                 AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH >
                 max_context)) {
        stage = "prompt_context_budget";
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) stage = "target_create";
    if (rc == AXIOM_OK) rc = axiom_qwen38_model_create(argv[1], 0, max_context, &target);
    if (rc == AXIOM_OK) stage = "temporal_validate";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                target, kValidationTokens, 0.0f, &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
    }

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0u;
    if (rc == AXIOM_OK) stage = "draft_runtime_create";
    if (rc == AXIOM_OK) rc = axiom_runtime_create(&draft_runtime, &runtime_config);
    axiom_model_config checkpoint_config{};
    checkpoint_config.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.path = argv[2];
    checkpoint_config.name = "qwen3.8-dspark";
    checkpoint_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    checkpoint_config.placement.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    if (rc == AXIOM_OK) stage = "draft_model_open";
    if (rc == AXIOM_OK) rc = axiom_model_open(draft_runtime, &draft_checkpoint, &checkpoint_config);
    if (rc == AXIOM_OK) stage = "draft_load";
    if (rc == AXIOM_OK) rc = axiom_qwen38_dspark_load(draft_checkpoint, draft_runtime, 0, &draft);

    axiom_qwen38_dspark_target_binding target_binding{};
    target_binding.abi_version = AXIOM_ABI_VERSION;
    target_binding.user_data = target;
    target_binding.embed_f32_device = axiom_qwen38_model_dspark_target_embed_f32_device;
    target_binding.lm_head_f32_device = axiom_qwen38_model_dspark_target_lm_head_f32_device;
    axiom_qwen38_dspark_compute_config compute_config{};
    compute_config.abi_version = AXIOM_ABI_VERSION;
    compute_config.max_context = max_context;
    if (rc == AXIOM_OK) stage = "compute_create";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_create(
                draft, draft_runtime, 0, &compute_config, &target_binding, &compute);
    }
    axiom_qwen38_speculative_config speculative_config{};
    speculative_config.abi_version = AXIOM_ABI_VERSION;
    speculative_config.device = 0;
    if (rc == AXIOM_OK) stage = "speculative_create";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_speculative_create(
                target, draft, compute, &speculative_config, &speculative);
    }

    uint32_t initial_anchor = 0u;
    if (rc == AXIOM_OK) stage = "host_prefill";
    for (uint32_t index = 0u; index < prompt_count && rc == AXIOM_OK; ++index) {
        axiom_qwen38_speculative_prefill_request request{};
        request.abi_version = AXIOM_ABI_VERSION;
        request.token_id = prompt_ids[index];
        axiom_qwen38_speculative_prefill_result result{};
        result.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_speculative_prefill_token(speculative, &request, &result);
        if (rc == AXIOM_OK) initial_anchor = result.target_token_id;
    }
    axiom_qwen38_dspark_compute_info prefill_info{};
    prefill_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) stage = "prefill_watermark";
    if (rc == AXIOM_OK) rc = axiom_qwen38_dspark_compute_info_get(compute, &prefill_info);
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(target) != prompt_count ||
         prefill_info.transaction_open != 0u ||
         prefill_info.committed_position != prompt_count ||
         initial_anchor >= AXIOM_QWEN38_DSPARK_VOCAB)) {
        rc = AXIOM_ERR_RUNTIME;
    }

    if (rc == AXIOM_OK) stage = "device_target_bind_qwen38_model";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_speculative_device_target_bind_qwen38_model(speculative);
    }
    if (rc == AXIOM_OK) stage = "decode_stream_create";
    if (rc == AXIOM_OK) rc = cuda_status(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (rc == AXIOM_OK) stage = "replay_events_create";
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventCreate(&replay_begin));
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventCreate(&replay_end));

    if (rc == AXIOM_OK) stage = "device_history_allocate";
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&seed_anchor_device), sizeof(*seed_anchor_device)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&seed_position_device), sizeof(*seed_position_device)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&history_device),
            static_cast<size_t>(kGraphCycles) * sizeof(*history_device)));
    if (rc == AXIOM_OK) stage = "host_history_allocate";
    if (rc == AXIOM_OK) rc = cuda_status(cudaHostAlloc(
            reinterpret_cast<void **>(&history_host),
            static_cast<size_t>(kGraphCycles) * sizeof(*history_host), cudaHostAllocPortable));

    const uint32_t initial_position = prompt_count;
    if (rc == AXIOM_OK) stage = "seed_anchor_position";
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(seed_anchor_device, &initial_anchor, sizeof(initial_anchor),
                                         cudaMemcpyHostToDevice, stream));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(seed_position_device, &initial_position, sizeof(initial_position),
                                         cudaMemcpyHostToDevice, stream));
    }

    const uint32_t *anchor_device = seed_anchor_device;
    const uint32_t *position_device = seed_position_device;
    if (rc == AXIOM_OK) stage = "replay_event_start";
    if (rc == AXIOM_OK && profiler_capture) {
        rc = cuda_status(cudaProfilerStart());
        profiler_started = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(replay_begin, stream));
    for (uint32_t cycle = 0u; cycle < kGraphCycles && rc == AXIOM_OK; ++cycle) {
        axiom_qwen38_speculative_device_step_request request{};
        request.abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
        request.anchor_token_device = anchor_device;
        request.anchor_position_device = position_device;
        request.stream = stream;
        axiom_qwen38_speculative_device_step_result result{};
        result.abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
        stage = "device_graph_replay";
        rc = axiom_qwen38_speculative_device_step_enqueue(speculative, &request, &result);
        if (rc == AXIOM_OK &&
            (result.graph_replayed != 1u ||
             result.proposal_tokens != AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
             result.verify_width != AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
             !result.proposal_tokens_device || !result.accepted_prefix_device ||
             !result.continuation_token_device || !result.async_status_device ||
             !result.next_anchor_token_device || !result.next_anchor_position_device)) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) stage = "device_history_copy";
        if (rc == AXIOM_OK && !append_device_copy(
                history_device[cycle].proposal,
                result.proposal_tokens_device, AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS, stream, &rc)) {
            break;
        }
        if (rc == AXIOM_OK && !append_device_copy(
                &history_device[cycle].accepted_prefix, result.accepted_prefix_device, 1u, stream, &rc)) {
            break;
        }
        if (rc == AXIOM_OK && !append_device_copy(
                &history_device[cycle].continuation_token, result.continuation_token_device, 1u, stream, &rc)) {
            break;
        }
        if (rc == AXIOM_OK && !append_device_copy(
                &history_device[cycle].async_status, result.async_status_device, 1u, stream, &rc)) {
            break;
        }
        if (rc == AXIOM_OK && !append_device_copy(
                &history_device[cycle].next_position, result.next_anchor_position_device, 1u, stream, &rc)) {
            break;
        }
        anchor_device = result.next_anchor_token_device;
        position_device = result.next_anchor_position_device;
    }
    if (rc == AXIOM_OK) stage = "replay_event_stop";
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(replay_end, stream));

    /* The only D2H transfer is queued after all 33 graph replays. Pinned host
     * history makes these asynchronous; the single stream sync below is the
     * only host wait in the device-decode measurement. */
    if (rc == AXIOM_OK) stage = "history_d2h";
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpyAsync(
            history_host, history_device, static_cast<size_t>(kGraphCycles) * sizeof(*history_host),
            cudaMemcpyDeviceToHost, stream));
    if (stream && !stream_synchronized) {
        const cudaError_t sync_status = cudaStreamSynchronize(stream);
        stream_synchronized = true;
        if (rc == AXIOM_OK && sync_status != cudaSuccess) {
            stage = "final_stream_sync";
            rc = cuda_status(sync_status);
        }
    }
    if (profiler_started) {
        const cudaError_t profiler_status = cudaProfilerStop();
        profiler_started = false;
        if (rc == AXIOM_OK && profiler_status != cudaSuccess) {
            stage = "profiler_stop";
            rc = cuda_status(profiler_status);
        }
    }

    float replay_ms = 0.0f;
    if (rc == AXIOM_OK) stage = "replay_event_elapsed";
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventElapsedTime(&replay_ms, replay_begin, replay_end));

    std::vector<uint32_t> output_ids;
    uint64_t accepted_total = 0u;
    uint64_t emitted_total = 0u;
    uint32_t expected_watermark = prompt_count;
    if (rc == AXIOM_OK) {
        try {
            output_ids.reserve(static_cast<size_t>(kGraphCycles) *
                               AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH);
        } catch (...) {
            rc = AXIOM_ERR_BUDGET;
            stage = "output_history_allocate";
        }
    }
    if (rc == AXIOM_OK) stage = "history_validate";
    for (uint32_t cycle = 0u; cycle < kGraphCycles && rc == AXIOM_OK; ++cycle) {
        const CycleHistory &history = history_host[cycle];
        const uint32_t accepted = history.accepted_prefix;
        const uint32_t status = history.async_status;
        const uint32_t continuation = history.continuation_token;
        if (status != static_cast<uint32_t>(AXIOM_OK) ||
            accepted > AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
            continuation >= AXIOM_QWEN38_DSPARK_VOCAB ||
            static_cast<uint64_t>(expected_watermark) + 1u + accepted > max_context) {
            rc = AXIOM_ERR_RUNTIME;
            break;
        }
        for (uint32_t token = 0u; token < accepted; ++token) {
            const uint32_t draft_token = history.proposal[token];
            if (draft_token >= AXIOM_QWEN38_DSPARK_VOCAB) {
                rc = AXIOM_ERR_RUNTIME;
                break;
            }
            output_ids.push_back(draft_token);
        }
        if (rc != AXIOM_OK) break;
        output_ids.push_back(continuation);
        accepted_total += accepted;
        emitted_total += static_cast<uint64_t>(accepted) + 1u;
        expected_watermark += accepted + 1u;
        if (history.next_position != expected_watermark) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK && (emitted_total != output_ids.size() ||
                           expected_watermark > max_context)) {
        rc = AXIOM_ERR_RUNTIME;
        stage = "emission_or_watermark";
    }

    if (rc == AXIOM_OK) {
        const double replay_seconds = static_cast<double>(replay_ms) / 1000.0;
        const double token_per_second = replay_seconds > 0.0
                ? static_cast<double>(emitted_total) / replay_seconds
                : 0.0;
        const double acceptance = static_cast<double>(accepted_total) /
                static_cast<double>(kGraphCycles * AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS);
        std::printf(
                "graph_gate=PASS cycles=%u prefill_tokens=%u prompt_source=%s initial_anchor=%u "
                "emitted=%llu accepted=%llu proposed=%u acceptance=%.6f "
                "replay_ms=%.6f tok_s=%.3f final_watermark=%u max_context=%u\n",
                kGraphCycles, prompt_count,
                prompt_argument ? (prompt_from_file ? "file" : "literal") : "built_in",
                initial_anchor,
                static_cast<unsigned long long>(emitted_total),
                static_cast<unsigned long long>(accepted_total),
                kGraphCycles * AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS,
                acceptance, static_cast<double>(replay_ms), token_per_second,
                expected_watermark, max_context);
        std::fputs("output_ids=", stdout);
        for (size_t index = 0u; index < output_ids.size(); ++index) {
            std::printf("%s%u", index == 0u ? "" : ",", output_ids[index]);
        }
        std::fputc('\n', stdout);
    }

    if (replay_end) (void)cudaEventDestroy(replay_end);
    if (replay_begin) (void)cudaEventDestroy(replay_begin);
    if (history_host) (void)cudaFreeHost(history_host);
    if (history_device) (void)cudaFree(history_device);
    if (seed_anchor_device) (void)cudaFree(seed_anchor_device);
    if (seed_position_device) (void)cudaFree(seed_position_device);
    if (stream) (void)cudaStreamDestroy(stream);
    axiom_qwen38_speculative_destroy(speculative);
    axiom_qwen38_dspark_compute_destroy(compute);
    axiom_qwen38_dspark_destroy(draft);
    axiom_model_close(draft_checkpoint);
    axiom_runtime_destroy(draft_runtime);
    axiom_qwen38_model_destroy(target);
    axiom_tokenizer_close(tokenizer);

    if (rc != AXIOM_OK) return fail(stage, rc);
    return 0;
}
