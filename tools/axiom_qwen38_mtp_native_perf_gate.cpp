/*
 * Bounded native-Qwen3.8 MTP CUDA-graph performance gate.
 *
 * The prompt is the byte-exact 1,002-token fixture used by
 * benchmarks/qwen38_rtx5090_api.py.  One calibration session is deliberately
 * excluded from measurement and determines the fixed replay count used by all
 * three measured repetitions: at least 33 graph cycles and enough cycles to
 * emit at least 256 authoritative tokens, bounded by 128 cycles.
 *
 * Every measured session uses one untimed priming replay to establish device
 * ownership.  One CUDA event pair brackets the uninterrupted replay train and
 * its fixed device-side history snapshots.  A single bulk D2H copy follows the
 * stop event, so elapsed_ms excludes model load, prefill, graph capture,
 * session setup, D2H validation and materialization.  No target-only or scalar
 * fallback is available.
 */

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"
#include "axiom/qwen38_mtp_speculative.h"
#include "axiom/sha256.hpp"

namespace {

constexpr uint32_t kMaxContext = 2048u;
constexpr uint32_t kCanonicalGraphCycles = 33u;
constexpr uint32_t kRequiredMeasuredTokens = 256u;
constexpr uint32_t kMaxMeasuredCycles = 128u;
constexpr uint32_t kMeasuredRepetitions = 3u;
constexpr uint32_t kDraft = AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
constexpr uint32_t kWidth = AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH;
constexpr uint32_t kHistoryCapacity = kMaxMeasuredCycles + 2u;
constexpr double kSustainedFloorTokensPerSecond = 320.0;
constexpr double kStretchTokensPerSecond = 380.0;
constexpr double kMaximumDispersionPercent = 5.0;
constexpr size_t kCanonicalPromptBytes = 1406u;
constexpr uint32_t kCanonicalPromptTokens = 1002u;
constexpr char kPromptFixture[] = "qwen38_rtx5090_api_1002_256";
constexpr char kPromptSha256[] =
        "e9b93103010c01e67e8fe8ac8f07d2e8b03e9dd012bd5dca821625fb891cfd86";
constexpr std::array<uint32_t, 2u> kStopTokens = {151643u, 151645u};
constexpr std::array<uint32_t,
        AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH>
        kTemporalValidationTokens = {
                151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u,
        };

static_assert(kDraft == 7u, "native MTP performance gate requires gamma seven");
static_assert(kWidth == 8u, "native MTP performance gate requires M8 verification");
static_assert(kCanonicalGraphCycles <= kMaxMeasuredCycles,
              "canonical graph cycle floor must fit the bounded gate");
static_assert(kMeasuredRepetitions == 3u,
              "the production performance sample is exactly three runs");
static_assert(kCanonicalPromptTokens + kWidth +
                      kMaxMeasuredCycles * kWidth + 1u <= kMaxContext,
              "prompt, prime, bounded replay and handoff probe must fit context");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_embed_f32_device),
                     axiom_qwen38_mtp_target_embed_f32_device_fn>::value,
        "target embedding callback ABI changed");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_lm_head_f32_device),
                     axiom_qwen38_mtp_target_lm_head_f32_device_fn>::value,
        "target LM-head callback ABI changed");

struct DeviceCycleHistory {
    uint32_t draft[kDraft];
    uint32_t verify[kWidth];
    uint32_t emitted[kWidth];
    uint32_t accepted_prefix;
    uint32_t commit_count;
    uint32_t emitted_count;
    uint32_t continuation;
    uint32_t stop_detected;
    uint32_t matched_stop_token;
    uint32_t next_anchor;
    uint32_t next_position;
    uint32_t async_status;
    float continuation_logit;
};

static_assert(std::is_standard_layout<DeviceCycleHistory>::value,
              "device history must have a stable byte layout");

struct CycleTotals {
    uint64_t cycles = 0u;
    uint64_t accepted = 0u;
    uint64_t proposed = 0u;
    uint64_t emitted = 0u;
};

struct RunEvidence {
    CycleTotals prime{};
    CycleTotals measured{};
    double elapsed_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    double acceptance_percent = 0.0;
    uint32_t first_position = 0u;
    uint32_t measured_final_position = 0u;
    uint32_t handoff_final_position = 0u;
    uint32_t final_anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t handoff_replays = 0u;
    bool exact_session_handoff = false;
    std::vector<uint32_t> output_token_ids;
    std::string output_sha256;
};

struct PhaseResources {
    int device = -1;
    axiom_qwen38_model *target = nullptr;
    axiom_runtime *runtime = nullptr;
    axiom_model *checkpoint = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    axiom_qwen38_mtp_compute *compute = nullptr;
    axiom_qwen38_mtp_speculative *controller = nullptr;
    uint32_t temporal_m8_validated = 0u;
    uint32_t mtp_tensor_count = 0u;

    void release() {
        if (device >= 0) (void)cudaSetDevice(device);
        (void)cudaDeviceSynchronize();
        axiom_qwen38_mtp_speculative_destroy(controller);
        controller = nullptr;
        axiom_qwen38_mtp_compute_destroy(compute);
        compute = nullptr;
        axiom_qwen38_mtp_destroy(mtp);
        mtp = nullptr;
        if (checkpoint) axiom_model_close(checkpoint);
        checkpoint = nullptr;
        if (runtime) axiom_runtime_destroy(runtime);
        runtime = nullptr;
        axiom_qwen38_model_destroy(target);
        target = nullptr;
    }
};

struct DeviceBuffers {
    int device = -1;
    cudaStream_t stream = nullptr;
    uint32_t *seed_anchor = nullptr;
    uint32_t *seed_position = nullptr;
    DeviceCycleHistory *history_device = nullptr;
    DeviceCycleHistory *history_host = nullptr;

    void release() {
        if (device >= 0) (void)cudaSetDevice(device);
        if (stream) (void)cudaStreamSynchronize(stream);
        if (history_host) (void)cudaFreeHost(history_host);
        history_host = nullptr;
        (void)cudaFree(history_device);
        history_device = nullptr;
        (void)cudaFree(seed_position);
        seed_position = nullptr;
        (void)cudaFree(seed_anchor);
        seed_anchor = nullptr;
        if (stream) (void)cudaStreamDestroy(stream);
        stream = nullptr;
    }
};

struct ReplayEvents {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;

    void release() {
        if (end) (void)cudaEventDestroy(end);
        if (begin) (void)cudaEventDestroy(begin);
        end = nullptr;
        begin = nullptr;
    }
};

struct DeviceCursor {
    const uint32_t *anchor_device = nullptr;
    const uint32_t *position_device = nullptr;
    uint32_t expected_anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t expected_position = 0u;
};

int cuda_status(const cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

bool parse_device(const char *text, int *out) {
    if (!text || !out || text[0] == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

bool env_is_zero(const char *name) {
    const char *value = name ? std::getenv(name) : nullptr;
    return value && value[0] == '0' && value[1] == '\0';
}

bool env_is_one(const char *name) {
    const char *value = name ? std::getenv(name) : nullptr;
    return value && value[0] == '1' && value[1] == '\0';
}

bool parse_optional_env_u32(const char *name, uint32_t *out) {
    if (!name || !out) return false;
    *out = 0u;
    const char *value = std::getenv(name);
    if (!value) return true;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *out = static_cast<uint32_t>(parsed);
    return true;
}

std::string canonical_prompt() {
    std::string prompt;
    prompt.reserve(kCanonicalPromptBytes);
    char line[32]{};
    for (uint32_t index = 1u; index <= 101u; ++index) {
        const int bytes = std::snprintf(
                line, sizeof(line), "item_%03u = %u", index, index);
        if (bytes <= 0 || static_cast<size_t>(bytes) >= sizeof(line)) return {};
        if (index != 1u) prompt.push_back('\n');
        prompt.append(line, static_cast<size_t>(bytes));
    }
    return prompt;
}

std::string token_ids_sha256(const std::vector<uint32_t> &tokens) {
    std::string bytes;
    try {
        bytes.resize(tokens.size() * sizeof(uint32_t));
    } catch (...) {
        return {};
    }
    for (size_t index = 0u; index < tokens.size(); ++index) {
        const uint32_t token = tokens[index];
        bytes[index * 4u + 0u] = static_cast<char>(token & 0xffu);
        bytes[index * 4u + 1u] = static_cast<char>((token >> 8u) & 0xffu);
        bytes[index * 4u + 2u] = static_cast<char>((token >> 16u) & 0xffu);
        bytes[index * 4u + 3u] = static_cast<char>((token >> 24u) & 0xffu);
    }
    return axiom::crypto::sha256_string_hex(bytes);
}

int tokenize_canonical_prompt(
        const char *model_path,
        std::vector<uint32_t> *token_ids,
        std::string *prompt_digest,
        const char **stage) {
    if (!model_path || !token_ids || !prompt_digest || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "canonical_prompt_build";
    std::string prompt;
    try {
        prompt = canonical_prompt();
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    if (prompt.size() != kCanonicalPromptBytes) return AXIOM_ERR_RUNTIME;
    *prompt_digest = axiom::crypto::sha256_string_hex(prompt);
    if (*prompt_digest != kPromptSha256) return AXIOM_ERR_RUNTIME;

    axiom_tokenizer *tokenizer = nullptr;
    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = model_path;
    config.name = "qwen38-mtp-native-perf-gate";
    config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    *stage = "tokenizer_open";
    int rc = axiom_tokenizer_open(&tokenizer, &config);
    if (rc == AXIOM_OK) {
        try {
            token_ids->assign(kMaxContext, 0u);
        } catch (...) {
            rc = AXIOM_ERR_BUDGET;
        }
    }
    uint32_t token_count = 0u;
    if (rc == AXIOM_OK) {
        *stage = "tokenizer_encode";
        rc = axiom_tokenizer_encode_text(
                tokenizer, prompt.c_str(), token_ids->data(),
                static_cast<uint32_t>(token_ids->size()), &token_count);
    }
    if (rc == AXIOM_OK) {
        *stage = "canonical_prompt_token_count";
        if (token_count != kCanonicalPromptTokens) rc = AXIOM_ERR_RUNTIME;
        else token_ids->resize(token_count);
    }
    if (tokenizer) axiom_tokenizer_close(tokenizer);
    return rc;
}

int controller_info(
        const axiom_qwen38_mtp_speculative *controller,
        axiom_qwen38_mtp_speculative_info *out) {
    if (!controller || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
    out->struct_size = sizeof(*out);
    return axiom_qwen38_mtp_speculative_info_get(controller, out);
}

int compute_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out) {
    if (!compute || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    return axiom_qwen38_mtp_compute_info_get(compute, out);
}

int device_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_device_info_v1 *out) {
    if (!compute || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION;
    out->struct_size = sizeof(*out);
    return axiom_qwen38_mtp_compute_device_info_get_v1(
            compute, out, sizeof(*out));
}

int load_phase(
        const char *model_path,
        const int device,
        PhaseResources *phase,
        const char **stage) {
    if (!model_path || !phase || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    phase->device = device;
    *stage = "cuda_device";
    int rc = cuda_status(cudaSetDevice(device));
    if (rc == AXIOM_OK) {
        *stage = "target_create";
        rc = axiom_qwen38_model_create(
                model_path, device, kMaxContext, &phase->target);
    }

    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        *stage = "temporal_m8_validate";
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                phase->target, kTemporalValidationTokens.data(), 0.0f,
                &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
        phase->temporal_m8_validated = validation.passed;
    }

    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        *stage = "temporal_capabilities";
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
                phase->target, &capabilities);
        if (rc == AXIOM_OK &&
            (capabilities.temporal_m8_available != 1u ||
             capabilities.temporal_m8_implemented != 1u ||
             capabilities.temporal_m8_validated != 1u ||
             capabilities.temporal_verify_width != kWidth ||
             capabilities.draft_block_size != kDraft ||
             capabilities.requires_gdn_causal_rows != 1u ||
             capabilities.requires_attention_triangular_kv != 1u ||
             capabilities.requires_prefix_state_install != 1u)) {
            rc = AXIOM_ERR_NOT_IMPLEMENTED;
        }
    }

    if (rc == AXIOM_OK) {
        *stage = "runtime_create";
        axiom_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.backend = AXIOM_BACKEND_CUDA;
        config.device = static_cast<uint32_t>(device);
        rc = axiom_runtime_create(&phase->runtime, &config);
    }
    if (rc == AXIOM_OK) {
        *stage = "checkpoint_open";
        axiom_model_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.path = model_path;
        config.name = "qwen38-mtp-native-perf-gate";
        config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        config.placement.abi_version = AXIOM_ABI_VERSION;
        config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        config.placement.device_id = static_cast<uint32_t>(device);
        rc = axiom_model_open(phase->runtime, &phase->checkpoint, &config);
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_load";
        rc = axiom_qwen38_mtp_load(
                phase->checkpoint, phase->runtime, device, &phase->mtp);
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_contract";
        axiom_qwen38_mtp_info info{};
        info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_mtp_info_get(phase->mtp, &info);
        phase->mtp_tensor_count = info.tensor_count;
        if (rc == AXIOM_OK &&
            (info.tensor_count != AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
             info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
             info.device_bytes == 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_compute_create";
        axiom_qwen38_mtp_target_binding binding{};
        binding.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        binding.user_data = phase->target;
        binding.embed_f32_device =
                axiom_qwen38_model_dspark_target_embed_f32_device;
        binding.lm_head_f32_device =
                axiom_qwen38_model_dspark_target_lm_head_f32_device;
        axiom_qwen38_mtp_compute_config_v2 config{};
        config.abi_version = AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.max_context = kMaxContext;
        config.cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
        config.rope_profile = axiom_qwen38_model_rope_profile(phase->target);
        rc = axiom_qwen38_mtp_compute_create_v2(
                phase->mtp, phase->runtime, device, &config, sizeof(config),
                &binding, &phase->compute);
    }
    if (rc == AXIOM_OK) {
        *stage = "controller_create";
        axiom_qwen38_mtp_speculative_config config{};
        config.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.device = static_cast<uint32_t>(device);
        rc = axiom_qwen38_mtp_speculative_create_v1(
                phase->target, phase->compute, &config, sizeof(config),
                &phase->controller);
    }
    return rc;
}

int allocate_device_buffers(
        const int device,
        DeviceBuffers *buffers,
        const char **stage) {
    if (!buffers || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    buffers->device = device;
    *stage = "device_stream_create";
    int rc = cuda_status(cudaStreamCreateWithFlags(
            &buffers->stream, cudaStreamNonBlocking));
    if (rc == AXIOM_OK) {
        *stage = "device_seed_anchor_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&buffers->seed_anchor),
                sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_seed_position_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&buffers->seed_position),
                sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_history_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&buffers->history_device),
                static_cast<size_t>(kHistoryCapacity) *
                        sizeof(DeviceCycleHistory)));
    }
    if (rc == AXIOM_OK) {
        *stage = "pinned_history_allocate";
        rc = cuda_status(cudaHostAlloc(
                reinterpret_cast<void **>(&buffers->history_host),
                static_cast<size_t>(kHistoryCapacity) *
                        sizeof(DeviceCycleHistory),
                cudaHostAllocPortable));
    }
    return rc;
}

int prepare_session(
        PhaseResources *phase,
        const std::vector<uint32_t> &prompt_ids,
        uint32_t *initial_anchor,
        const char **stage) {
    if (!phase || !phase->controller || !initial_anchor || !stage ||
        prompt_ids.empty()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *initial_anchor = AXIOM_TOKEN_ID_INVALID;
    *stage = "controller_reset";
    int rc = axiom_qwen38_mtp_speculative_reset(phase->controller);
    for (uint32_t index = 0u;
         index < prompt_ids.size() && rc == AXIOM_OK; ++index) {
        *stage = "prompt_prefill";
        axiom_qwen38_mtp_speculative_prefill_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_prefill_token(
                phase->controller, prompt_ids[index], &result);
        if (rc == AXIOM_OK &&
            (result.token_position != index ||
             result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
             !std::isfinite(result.target_logit))) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) *initial_anchor = result.target_token_id;
    }
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(phase->target) != prompt_ids.size() ||
         axiom_qwen38_mtp_compute_position(phase->compute) !=
                 prompt_ids.size())) {
        *stage = "prompt_position_invariants";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        *stage = "device_stop_contract";
        rc = axiom_qwen38_mtp_speculative_device_stop_tokens_set(
                phase->controller, kStopTokens.data(),
                static_cast<uint32_t>(kStopTokens.size()));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_graph_prepare";
        rc = axiom_qwen38_mtp_speculative_device_prepare(phase->controller);
    }
    if (rc == AXIOM_OK) {
        *stage = "prepared_capture_invariants";
        axiom_qwen38_mtp_speculative_info controller{};
        axiom_qwen38_mtp_compute_info compute{};
        axiom_qwen38_mtp_compute_device_info_v1 device{};
        rc = controller_info(phase->controller, &controller);
        if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
        if (rc == AXIOM_OK) rc = device_info(phase->compute, &device);
        if (rc == AXIOM_OK &&
            (controller.poisoned != 0u ||
             controller.position != prompt_ids.size() ||
             compute.committed_position != prompt_ids.size() ||
             compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
             device.graph_capture_active != 0u ||
             device.device_session_active != 0u ||
             device.committed_position != prompt_ids.size())) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    return rc;
}

int start_device_cursor(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        const uint32_t anchor,
        const uint32_t position,
        DeviceCursor *cursor,
        const char **stage) {
    if (!phase || !buffers || !buffers->stream || !buffers->seed_anchor ||
        !buffers->seed_position || !cursor || !stage ||
        anchor >= AXIOM_QWEN38_MODEL_VOCAB) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "device_seed_anchor";
    int rc = cuda_status(cudaMemcpyAsync(
            buffers->seed_anchor, &anchor, sizeof(anchor),
            cudaMemcpyHostToDevice, buffers->stream));
    if (rc == AXIOM_OK) {
        *stage = "device_seed_position";
        rc = cuda_status(cudaMemcpyAsync(
                buffers->seed_position, &position, sizeof(position),
                cudaMemcpyHostToDevice, buffers->stream));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_commit_limit";
        rc = axiom_qwen38_mtp_speculative_device_commit_limit_set(
                phase->controller, kWidth,
                reinterpret_cast<void *>(buffers->stream));
    }
    if (rc == AXIOM_OK) {
        cursor->anchor_device = buffers->seed_anchor;
        cursor->position_device = buffers->seed_position;
        cursor->expected_anchor = anchor;
        cursor->expected_position = position;
    }
    return rc;
}

int append_device_to_device_copy(
        void *destination,
        const void *source,
        const size_t bytes,
        const cudaStream_t stream) {
    if (!destination || !source || bytes == 0u || !stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return cuda_status(cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
}

int snapshot_cycle(
        DeviceCycleHistory *history,
        const axiom_qwen38_mtp_speculative_device_step_result &result,
        const cudaStream_t stream) {
    if (!history) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = append_device_to_device_copy(
            history->draft, result.draft_token_ids_device,
            static_cast<size_t>(kDraft) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            history->verify, result.verify_token_ids_device,
            static_cast<size_t>(kWidth) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            history->emitted, result.emitted_token_ids_device,
            static_cast<size_t>(kWidth) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->accepted_prefix, result.accepted_prefix_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->commit_count, result.target_commit_count_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->emitted_count, result.emitted_token_count_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->continuation, result.continuation_token_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->stop_detected, result.stop_detected_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->matched_stop_token, result.matched_stop_token_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->next_anchor, result.next_anchor_token_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->next_position, result.next_anchor_position_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->async_status, result.async_status_device,
            sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_device_copy(
            &history->continuation_logit, result.continuation_logit_device,
            sizeof(float), stream);
    return rc;
}

int copy_history_to_host(
        DeviceBuffers *buffers,
        const uint32_t first,
        const uint32_t count) {
    if (!buffers || !buffers->stream || !buffers->history_device ||
        !buffers->history_host || count == 0u ||
        first > kHistoryCapacity || count > kHistoryCapacity - first) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return cuda_status(cudaMemcpyAsync(
            buffers->history_host + first,
            buffers->history_device + first,
            static_cast<size_t>(count) * sizeof(DeviceCycleHistory),
            cudaMemcpyDeviceToHost, buffers->stream));
}

bool valid_device_result_contract(
        const axiom_qwen38_mtp_speculative_device_step_result &result) {
    return result.abi_version ==
                    AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION &&
            result.struct_size == sizeof(result) && result.graph_replayed == 1u &&
            result.draft_tokens == kDraft && result.verify_width == kWidth &&
            result.draft_token_ids_device && result.verify_token_ids_device &&
            result.accepted_prefix_device && result.target_commit_count_device &&
            result.emitted_token_count_device && result.continuation_token_device &&
            result.continuation_logit_device && result.emitted_token_ids_device &&
            result.stop_detected_device && result.matched_stop_token_device &&
            result.next_anchor_token_device && result.next_anchor_position_device &&
            result.async_status_device;
}

int enqueue_cycle(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        DeviceCursor *cursor,
        DeviceCycleHistory *history,
        const cudaEvent_t begin,
        const cudaEvent_t end,
        const char **stage) {
    if (!phase || !buffers || !cursor || !history || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = AXIOM_OK;
    if (begin) {
        *stage = "replay_event_begin";
        rc = cuda_status(cudaEventRecord(begin, buffers->stream));
    }
    axiom_qwen38_mtp_speculative_device_step_result result{};
    if (rc == AXIOM_OK) {
        *stage = "device_graph_replay";
        axiom_qwen38_mtp_speculative_device_step_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_device = cursor->anchor_device;
        request.anchor_position_device = cursor->position_device;
        request.stream = reinterpret_cast<void *>(buffers->stream);
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_device_step_enqueue(
                phase->controller, &request, &result);
    }
    if (rc == AXIOM_OK && !valid_device_result_contract(result)) {
        *stage = "device_result_contract";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK && end) {
        *stage = "replay_event_end";
        rc = cuda_status(cudaEventRecord(end, buffers->stream));
    }
    if (rc == AXIOM_OK) {
        *stage = "immediate_per_replay_d2h";
        rc = snapshot_cycle(history, result, buffers->stream);
    }
    if (rc == AXIOM_OK) {
        cursor->anchor_device = result.next_anchor_token_device;
        cursor->position_device = result.next_anchor_position_device;
    }
    return rc;
}

int validate_cycle(
        const DeviceCycleHistory &history,
        DeviceCursor *cursor,
        std::vector<uint32_t> *committed_history,
        std::vector<uint32_t> *output_tokens,
        CycleTotals *totals,
        const char **stage) {
    if (!cursor || !committed_history || !output_tokens || !totals || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "cycle_invariants";
    if (history.async_status != static_cast<uint32_t>(AXIOM_OK) ||
        history.accepted_prefix > kDraft ||
        history.commit_count != history.accepted_prefix + 1u ||
        history.emitted_count != history.commit_count ||
        history.commit_count == 0u || history.commit_count > kWidth ||
        history.stop_detected != 0u ||
        history.matched_stop_token != AXIOM_TOKEN_ID_INVALID ||
        history.continuation >= AXIOM_QWEN38_MODEL_VOCAB ||
        history.next_anchor != history.continuation ||
        history.verify[0] != cursor->expected_anchor ||
        history.next_position != cursor->expected_position + history.commit_count ||
        history.next_position > kMaxContext ||
        !std::isfinite(history.continuation_logit)) {
        return AXIOM_ERR_RUNTIME;
    }
    for (uint32_t index = 0u; index < kDraft; ++index) {
        if (history.draft[index] >= AXIOM_QWEN38_MODEL_VOCAB ||
            history.verify[index + 1u] != history.draft[index]) {
            return AXIOM_ERR_RUNTIME;
        }
    }
    for (uint32_t index = 0u; index < history.accepted_prefix; ++index) {
        if (history.emitted[index] != history.draft[index]) {
            return AXIOM_ERR_RUNTIME;
        }
    }
    if (history.emitted[history.accepted_prefix] != history.continuation) {
        return AXIOM_ERR_RUNTIME;
    }
    try {
        for (uint32_t index = 0u; index < history.emitted_count; ++index) {
            if (history.emitted[index] >= AXIOM_QWEN38_MODEL_VOCAB) {
                return AXIOM_ERR_RUNTIME;
            }
            committed_history->push_back(history.emitted[index]);
            output_tokens->push_back(history.emitted[index]);
        }
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    totals->cycles += 1u;
    totals->accepted += history.accepted_prefix;
    totals->proposed += kDraft;
    totals->emitted += history.emitted_count;
    cursor->expected_anchor = history.next_anchor;
    cursor->expected_position = history.next_position;
    return AXIOM_OK;
}

int verify_enqueue_counters(
        const axiom_qwen38_mtp_speculative_info &before,
        const axiom_qwen38_mtp_speculative_info &after,
        const uint64_t expected_replays,
        const char **stage) {
    if (!stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "device_enqueue_counters";
    if (after.poisoned != 0u || after.attempted_steps < before.attempted_steps ||
        after.committed_steps < before.committed_steps ||
        after.proposed_tokens < before.proposed_tokens ||
        after.failed_steps < before.failed_steps ||
        after.attempted_steps - before.attempted_steps != expected_replays ||
        after.committed_steps - before.committed_steps != expected_replays ||
        after.proposed_tokens - before.proposed_tokens != expected_replays * kDraft ||
        after.failed_steps != before.failed_steps) {
        return AXIOM_ERR_RUNTIME;
    }
    return AXIOM_OK;
}

int materialize_and_verify(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        const std::vector<uint32_t> &committed_history,
        const uint32_t expected_position,
        const char **stage) {
    if (!phase || !buffers || !stage ||
        committed_history.size() != expected_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "device_session_materialize";
    int rc = axiom_qwen38_mtp_speculative_device_session_end(
            phase->controller, reinterpret_cast<void *>(buffers->stream),
            committed_history.data(),
            static_cast<uint32_t>(committed_history.size()));
    if (rc == AXIOM_OK) {
        *stage = "exact_session_handoff_invariants";
        axiom_qwen38_mtp_speculative_info controller{};
        axiom_qwen38_mtp_compute_info compute{};
        axiom_qwen38_mtp_compute_device_info_v1 device{};
        rc = controller_info(phase->controller, &controller);
        if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
        if (rc == AXIOM_OK) rc = device_info(phase->compute, &device);
        if (rc == AXIOM_OK &&
            (controller.poisoned != 0u || controller.position != expected_position ||
             axiom_qwen38_model_position(phase->target) != expected_position ||
             axiom_qwen38_mtp_compute_position(phase->compute) != expected_position ||
             compute.committed_position != expected_position ||
             compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
             device.device_session_active != 0u ||
             device.committed_position != expected_position)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    return rc;
}

int run_handoff_probe(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        DeviceCursor *cursor,
        std::vector<uint32_t> *committed_history,
        RunEvidence *evidence,
        const char **stage) {
    if (!phase || !buffers || !cursor || !committed_history || !evidence ||
        !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    DeviceCursor probe{};
    int rc = start_device_cursor(
            phase, buffers, cursor->expected_anchor, cursor->expected_position,
            &probe, stage);
    if (rc == AXIOM_OK) {
        *stage = "handoff_commit_limit_one";
        rc = axiom_qwen38_mtp_speculative_device_commit_limit_set(
                phase->controller, 1u,
                reinterpret_cast<void *>(buffers->stream));
    }
    if (rc == AXIOM_OK) {
        rc = enqueue_cycle(
                phase, buffers, &probe, &buffers->history_device[0], nullptr,
                nullptr, stage);
    }
    if (rc == AXIOM_OK) {
        *stage = "handoff_probe_history_d2h";
        rc = copy_history_to_host(buffers, 0u, 1u);
    }
    if (rc == AXIOM_OK) {
        *stage = "handoff_probe_sync";
        rc = cuda_status(cudaStreamSynchronize(buffers->stream));
    }
    std::vector<uint32_t> probe_output;
    CycleTotals probe_totals{};
    if (rc == AXIOM_OK) rc = validate_cycle(
            buffers->history_host[0], &probe, committed_history, &probe_output,
            &probe_totals, stage);
    if (rc == AXIOM_OK &&
        (probe_totals.cycles != 1u || probe_totals.emitted != 1u ||
         probe_totals.accepted != 0u || probe.expected_position !=
                 cursor->expected_position + 1u)) {
        *stage = "handoff_probe_exact_advance";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) rc = materialize_and_verify(
            phase, buffers, *committed_history, probe.expected_position, stage);
    if (rc == AXIOM_OK) {
        evidence->handoff_replays = 1u;
        evidence->handoff_final_position = probe.expected_position;
        evidence->exact_session_handoff = true;
    }
    return rc;
}

int create_events(
        const uint32_t cycles,
        ReplayEvents *events,
        const char **stage) {
    if (!events || !stage || cycles == 0u || cycles > kMaxMeasuredCycles) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "replay_events_allocate";
    int rc = cuda_status(cudaEventCreateWithFlags(
            &events->begin, cudaEventDefault));
    if (rc == AXIOM_OK) rc = cuda_status(cudaEventCreateWithFlags(
            &events->end, cudaEventDefault));
    return rc;
}

int initialize_run_history(
        const std::vector<uint32_t> &prompt_ids,
        std::vector<uint32_t> *committed_history,
        RunEvidence *evidence) {
    if (!committed_history || !evidence) return AXIOM_ERR_INVALID_ARGUMENT;
    try {
        committed_history->clear();
        committed_history->reserve(kMaxContext);
        committed_history->insert(
                committed_history->end(), prompt_ids.begin(), prompt_ids.end());
        evidence->output_token_ids.clear();
        evidence->output_token_ids.reserve(
                static_cast<size_t>(kWidth) * (kMaxMeasuredCycles + 1u));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    evidence->first_position = static_cast<uint32_t>(prompt_ids.size());
    return AXIOM_OK;
}

int run_calibration(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        const std::vector<uint32_t> &prompt_ids,
        RunEvidence *evidence,
        uint32_t *measured_cycles,
        const char **stage) {
    if (!phase || !buffers || !evidence || !measured_cycles || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t initial_anchor = AXIOM_TOKEN_ID_INVALID;
    int rc = prepare_session(phase, prompt_ids, &initial_anchor, stage);
    std::vector<uint32_t> committed_history;
    if (rc == AXIOM_OK) rc = initialize_run_history(
            prompt_ids, &committed_history, evidence);
    DeviceCursor cursor{};
    if (rc == AXIOM_OK) rc = start_device_cursor(
            phase, buffers, initial_anchor,
            static_cast<uint32_t>(prompt_ids.size()), &cursor, stage);

    axiom_qwen38_mtp_speculative_info counters_before{};
    if (rc == AXIOM_OK) rc = controller_info(
            phase->controller, &counters_before);
    if (rc == AXIOM_OK) rc = enqueue_cycle(
            phase, buffers, &cursor, &buffers->history_device[0], nullptr,
            nullptr, stage);
    if (rc == AXIOM_OK) {
        *stage = "calibration_prime_history_d2h";
        rc = copy_history_to_host(buffers, 0u, 1u);
    }
    if (rc == AXIOM_OK) {
        *stage = "calibration_prime_sync";
        rc = cuda_status(cudaStreamSynchronize(buffers->stream));
    }
    if (rc == AXIOM_OK) rc = validate_cycle(
            buffers->history_host[0], &cursor, &committed_history,
            &evidence->output_token_ids, &evidence->prime, stage);

    uint32_t cycles = 0u;
    while (rc == AXIOM_OK && cycles < kMaxMeasuredCycles &&
           (cycles < kCanonicalGraphCycles ||
            evidence->measured.emitted < kRequiredMeasuredTokens)) {
        rc = enqueue_cycle(
                phase, buffers, &cursor,
                &buffers->history_device[1u + cycles], nullptr, nullptr, stage);
        if (rc == AXIOM_OK) {
            *stage = "calibration_cycle_history_d2h";
            rc = copy_history_to_host(buffers, 1u + cycles, 1u);
        }
        if (rc == AXIOM_OK) {
            *stage = "calibration_cycle_sync";
            rc = cuda_status(cudaStreamSynchronize(buffers->stream));
        }
        if (rc == AXIOM_OK) rc = validate_cycle(
                buffers->history_host[1u + cycles], &cursor,
                &committed_history, &evidence->output_token_ids,
                &evidence->measured, stage);
        ++cycles;
    }
    if (rc == AXIOM_OK &&
        (cycles < kCanonicalGraphCycles ||
         evidence->measured.emitted < kRequiredMeasuredTokens)) {
        *stage = "bounded_calibration_output_floor";
        rc = AXIOM_ERR_BUDGET;
    }
    axiom_qwen38_mtp_speculative_info counters_after{};
    if (rc == AXIOM_OK) rc = controller_info(
            phase->controller, &counters_after);
    if (rc == AXIOM_OK) rc = verify_enqueue_counters(
            counters_before, counters_after, 1u + cycles, stage);
    if (rc == AXIOM_OK) {
        evidence->measured_final_position = cursor.expected_position;
        evidence->final_anchor = cursor.expected_anchor;
        evidence->output_sha256 = token_ids_sha256(evidence->output_token_ids);
        if (evidence->output_sha256.empty()) rc = AXIOM_ERR_BUDGET;
    }
    if (rc == AXIOM_OK) rc = materialize_and_verify(
            phase, buffers, committed_history, cursor.expected_position, stage);
    if (rc == AXIOM_OK) rc = run_handoff_probe(
            phase, buffers, &cursor, &committed_history, evidence, stage);
    if (rc == AXIOM_OK) *measured_cycles = cycles;
    return rc;
}

int run_measured_repetition(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        const std::vector<uint32_t> &prompt_ids,
        const uint32_t measured_cycles,
        RunEvidence *evidence,
        const char **stage) {
    if (!phase || !buffers || !evidence || !stage ||
        measured_cycles < kCanonicalGraphCycles ||
        measured_cycles > kMaxMeasuredCycles) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t initial_anchor = AXIOM_TOKEN_ID_INVALID;
    int rc = prepare_session(phase, prompt_ids, &initial_anchor, stage);
    std::vector<uint32_t> committed_history;
    if (rc == AXIOM_OK) rc = initialize_run_history(
            prompt_ids, &committed_history, evidence);
    DeviceCursor cursor{};
    if (rc == AXIOM_OK) rc = start_device_cursor(
            phase, buffers, initial_anchor,
            static_cast<uint32_t>(prompt_ids.size()), &cursor, stage);

    axiom_qwen38_mtp_speculative_info counters_before{};
    if (rc == AXIOM_OK) rc = controller_info(
            phase->controller, &counters_before);

    /* This replay establishes device ownership and is intentionally excluded
     * from every event-derived performance metric. */
    if (rc == AXIOM_OK) rc = enqueue_cycle(
            phase, buffers, &cursor, &buffers->history_device[0], nullptr,
            nullptr, stage);
    if (rc == AXIOM_OK) {
        *stage = "measured_prime_history_d2h";
        rc = copy_history_to_host(buffers, 0u, 1u);
    }
    if (rc == AXIOM_OK) {
        *stage = "measured_prime_sync";
        rc = cuda_status(cudaStreamSynchronize(buffers->stream));
    }
    if (rc == AXIOM_OK) rc = validate_cycle(
            buffers->history_host[0], &cursor, &committed_history,
            &evidence->output_token_ids, &evidence->prime, stage);

    ReplayEvents events{};
    if (rc == AXIOM_OK) rc = create_events(measured_cycles, &events, stage);
    const bool profiler_capture =
            std::getenv("AXIOM_QWEN38_NSYS_CAPTURE") != nullptr;
    bool profiler_started = false;
    if (rc == AXIOM_OK && profiler_capture) {
        *stage = "profiler_start";
        rc = cuda_status(cudaProfilerStart());
        profiler_started = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        *stage = "replay_event_begin";
        rc = cuda_status(cudaEventRecord(events.begin, buffers->stream));
    }
    for (uint32_t cycle = 0u; cycle < measured_cycles && rc == AXIOM_OK;
         ++cycle) {
        rc = enqueue_cycle(
                phase, buffers, &cursor, &buffers->history_device[1u + cycle],
                nullptr, nullptr, stage);
    }
    if (rc == AXIOM_OK) {
        *stage = "replay_event_end";
        rc = cuda_status(cudaEventRecord(events.end, buffers->stream));
    }
    if (rc == AXIOM_OK) {
        *stage = "measured_history_d2h";
        rc = copy_history_to_host(buffers, 1u, measured_cycles);
    }
    if (rc == AXIOM_OK) {
        *stage = "measured_stream_sync";
        rc = cuda_status(cudaStreamSynchronize(buffers->stream));
    }
    if (profiler_started) {
        const cudaError_t profiler_rc = cudaProfilerStop();
        profiler_started = false;
        if (rc == AXIOM_OK && profiler_rc != cudaSuccess) {
            *stage = "profiler_stop";
            rc = cuda_status(profiler_rc);
        }
    }
    if (rc == AXIOM_OK) {
        *stage = "replay_event_elapsed";
        float replay_ms = 0.0f;
        rc = cuda_status(cudaEventElapsedTime(
                &replay_ms, events.begin, events.end));
        if (rc == AXIOM_OK &&
            (!std::isfinite(replay_ms) || replay_ms <= 0.0f)) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) evidence->elapsed_ms = replay_ms;
    }
    for (uint32_t cycle = 0u; cycle < measured_cycles && rc == AXIOM_OK;
         ++cycle) {
        rc = validate_cycle(
                buffers->history_host[1u + cycle], &cursor,
                &committed_history, &evidence->output_token_ids,
                &evidence->measured, stage);
    }
    events.release();

    if (rc == AXIOM_OK &&
        (evidence->measured.cycles != measured_cycles ||
         evidence->measured.proposed !=
                 static_cast<uint64_t>(measured_cycles) * kDraft ||
         evidence->measured.emitted < kRequiredMeasuredTokens ||
         evidence->measured.accepted == 0u || evidence->elapsed_ms <= 0.0)) {
        *stage = "measured_output_invariants";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        const double elapsed_seconds = evidence->elapsed_ms / 1000.0;
        evidence->decode_tokens_per_second =
                static_cast<double>(evidence->measured.emitted) / elapsed_seconds;
        evidence->acceptance_percent =
                100.0 * static_cast<double>(evidence->measured.accepted) /
                static_cast<double>(evidence->measured.proposed);
        if (!std::isfinite(evidence->decode_tokens_per_second) ||
            evidence->decode_tokens_per_second <= 0.0 ||
            !std::isfinite(evidence->acceptance_percent)) {
            *stage = "measured_metric_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
    }

    axiom_qwen38_mtp_speculative_info counters_after{};
    if (rc == AXIOM_OK) rc = controller_info(
            phase->controller, &counters_after);
    if (rc == AXIOM_OK) rc = verify_enqueue_counters(
            counters_before, counters_after, 1u + measured_cycles, stage);
    if (rc == AXIOM_OK) {
        evidence->measured_final_position = cursor.expected_position;
        evidence->final_anchor = cursor.expected_anchor;
        evidence->output_sha256 = token_ids_sha256(evidence->output_token_ids);
        if (evidence->output_sha256.empty()) rc = AXIOM_ERR_BUDGET;
    }
    if (rc == AXIOM_OK) rc = materialize_and_verify(
            phase, buffers, committed_history, cursor.expected_position, stage);
    if (rc == AXIOM_OK) rc = run_handoff_probe(
            phase, buffers, &cursor, &committed_history, evidence, stage);
    return rc;
}

double median_of_samples(
        std::array<double, kMeasuredRepetitions> samples,
        const uint32_t count) {
    if (count == 0u || count > samples.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(samples.begin(), samples.begin() + count);
    if ((count & 1u) != 0u) return samples[count / 2u];
    return 0.5 * (samples[count / 2u - 1u] + samples[count / 2u]);
}

}  // namespace

int main(int argc, char **argv) {
    const char *stage = "arguments";
    int rc = AXIOM_OK;
    int device = 0;
    uint32_t requested_repetitions = kMeasuredRepetitions;
    if (argc < 3 || argc > 4 || !parse_device(argv[2], &device) ||
        (argc == 4 && std::strcmp(argv[3], "1") != 0 &&
         std::strcmp(argv[3], "3") != 0)) {
        std::fprintf(stderr,
                "usage: %s ABLITERATED_CHECKPOINT_DIR DEVICE [1|3]\n",
                argv[0]);
        return 2;
    }
    if (argc == 4) requested_repetitions =
            static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        stage = "disable_duplicate_generic_residency";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK &&
        setenv("AXIOM_QWEN38_MATMUL_AUTOTUNE", "1", 0) != 0) {
        stage = "enable_production_matmul_autotune";
        rc = AXIOM_ERR_RUNTIME;
    }

    uint32_t fast_graph_min_position = 0u;
    if (rc == AXIOM_OK && !parse_optional_env_u32(
            "AXIOM_QWEN38_MTP_FAST_GRAPH_MIN_POSITION",
            &fast_graph_min_position)) {
        stage = "fast_graph_min_position_environment";
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    const bool dual_graph = fast_graph_min_position > 0u;
    const bool selected_native_splitk = dual_graph &&
            kCanonicalPromptTokens >= fast_graph_min_position;
    const bool mtp_fp8_weight_gemv =
            env_is_one("AXIOM_QWEN38_MTP_FP8_WEIGHT_GEMV");
    const char *selected_temporal_backend =
            selected_native_splitk ? "native_splitk" : "exact";

    std::vector<uint32_t> prompt_ids;
    std::string prompt_digest;
    if (rc == AXIOM_OK) rc = tokenize_canonical_prompt(
            argv[1], &prompt_ids, &prompt_digest, &stage);

    PhaseResources phase{};
    DeviceBuffers buffers{};
    if (rc == AXIOM_OK) rc = load_phase(argv[1], device, &phase, &stage);
    if (rc == AXIOM_OK) rc = allocate_device_buffers(device, &buffers, &stage);

    RunEvidence calibration{};
    uint32_t measured_cycles = 0u;
    if (rc == AXIOM_OK) rc = run_calibration(
            &phase, &buffers, prompt_ids, &calibration, &measured_cycles,
            &stage);

    std::array<RunEvidence, kMeasuredRepetitions> runs{};
    for (uint32_t repetition = 0u;
         repetition < requested_repetitions && rc == AXIOM_OK; ++repetition) {
        rc = run_measured_repetition(
                &phase, &buffers, prompt_ids, measured_cycles,
                &runs[repetition], &stage);
        if (rc == AXIOM_OK &&
            (runs[repetition].output_token_ids != calibration.output_token_ids ||
             runs[repetition].output_sha256 != calibration.output_sha256 ||
             runs[repetition].measured.accepted != calibration.measured.accepted ||
             runs[repetition].measured.emitted != calibration.measured.emitted ||
             runs[repetition].measured.cycles != calibration.measured.cycles)) {
            stage = "calibration_measured_token_parity";
            rc = AXIOM_ERR_RUNTIME;
        }
    }

    std::array<double, kMeasuredRepetitions> throughput{};
    uint64_t accepted_total = 0u;
    uint64_t proposed_total = 0u;
    uint64_t emitted_total = 0u;
    uint64_t graph_replays_total = 0u;
    bool exact_handoff = true;
    bool output_parity = rc == AXIOM_OK;
    for (uint32_t index = 0u; index < requested_repetitions; ++index) {
        throughput[index] = runs[index].decode_tokens_per_second;
        accepted_total += runs[index].measured.accepted;
        proposed_total += runs[index].measured.proposed;
        emitted_total += runs[index].measured.emitted;
        graph_replays_total += runs[index].measured.cycles;
        exact_handoff = exact_handoff && runs[index].exact_session_handoff;
        if (index != 0u) {
            output_parity = output_parity &&
                    runs[index].output_token_ids == runs[0].output_token_ids;
        }
    }

    const double minimum_tps = *std::min_element(
            throughput.begin(), throughput.begin() + requested_repetitions);
    const double maximum_tps = *std::max_element(
            throughput.begin(), throughput.begin() + requested_repetitions);
    const double median_tps = median_of_samples(
            throughput, requested_repetitions);
    const double dispersion_percent = median_tps > 0.0
            ? 100.0 * (maximum_tps - minimum_tps) / median_tps
            : std::numeric_limits<double>::infinity();
    const bool floor_320_pass = minimum_tps >= kSustainedFloorTokensPerSecond;
    const bool target_380_pass = median_tps >= kStretchTokensPerSecond;
    const bool dispersion_pass =
            dispersion_percent <= kMaximumDispersionPercent;

    if (rc == AXIOM_OK && (!output_parity || !exact_handoff)) {
        stage = "cross_repetition_invariants";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK && (!floor_320_pass || !dispersion_pass)) {
        stage = !floor_320_pass ? "sustained_320_floor" : "dispersion_5pct";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) stage = "complete";

    const uint32_t temporal_m8_validated = phase.temporal_m8_validated;
    const uint32_t mtp_tensor_count = phase.mtp_tensor_count;
    buffers.release();
    phase.release();

    const bool passed = rc == AXIOM_OK;
    std::printf(
            "{\"schema\":\"axiom.qwen38.mtp-native-perf.v1\","
            "\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"device\":%d,\"model_loads\":1,"
            "\"execution_profile\":{\"matmul_autotune\":1,"
            "\"device_temporal_exact\":%u,"
            "\"dual_graph\":%u,"
            "\"fast_graph_min_position\":%u,"
            "\"mtp_fp8_weight_gemv\":%u,"
            "\"selected_temporal_backend\":\"%s\","
            "\"mtp_target_kv_replay\":%u,\"mtp_gemv_m1\":%u},"
            "\"prompt_fixture\":\"%s\",\"prompt_sha256\":\"%s\","
            "\"prompt_bytes\":%zu,\"prompt_tokens\":%zu,"
            "\"max_context\":%u,\"required_output_tokens\":%u,"
            "\"canonical_minimum_cycles\":%u,\"measured_cycles\":%u,"
            "\"requested_repetitions\":%u,"
            "\"maximum_bounded_cycles\":%u,"
            "\"warmup_separated\":1,"
            "\"timing_scope\":\"cuda_graph_replay_plus_device_history\","
            "\"bulk_d2h_after_timing\":1,"
            "\"warmup\":{\"prime_replays\":%llu,"
            "\"calibration_replays\":%llu,\"accepted\":%llu,"
            "\"proposed\":%llu,\"emitted\":%llu},"
            "\"repetitions\":[",
            passed ? "pass" : "fail", stage, rc, device,
            selected_native_splitk ? 0u : 1u,
            dual_graph ? 1u : 0u,
            fast_graph_min_position,
            mtp_fp8_weight_gemv ? 1u : 0u,
            selected_temporal_backend,
            env_is_zero("AXIOM_QWEN38_MTP_TARGET_KV_REPLAY") ? 0u : 1u,
            env_is_zero("AXIOM_QWEN38_MTP_GEMV_M1") ? 0u : 1u,
            kPromptFixture,
            prompt_digest.c_str(), kCanonicalPromptBytes, prompt_ids.size(),
            kMaxContext, kRequiredMeasuredTokens, kCanonicalGraphCycles,
            measured_cycles, requested_repetitions, kMaxMeasuredCycles,
            static_cast<unsigned long long>(calibration.prime.cycles),
            static_cast<unsigned long long>(calibration.measured.cycles),
            static_cast<unsigned long long>(calibration.measured.accepted),
            static_cast<unsigned long long>(calibration.measured.proposed),
            static_cast<unsigned long long>(calibration.measured.emitted));
    for (uint32_t index = 0u; index < requested_repetitions; ++index) {
        const RunEvidence &run = runs[index];
        std::printf(
                "%s{\"run\":%u,\"status\":\"%s\","
                "\"accepted\":%llu,\"proposed\":%llu,"
                "\"emitted\":%llu,\"cycles\":%llu,"
                "\"graph_replays\":%llu,\"elapsed_ms\":%.6f,"
                "\"decode_tokens_per_second\":%.6f,"
                "\"acceptance_percent\":%.6f,"
                "\"prime_replays\":%llu,\"handoff_replays\":%u,"
                "\"measured_final_position\":%u,"
                "\"handoff_final_position\":%u,"
                "\"exact_session_handoff\":%u,"
                "\"output_token_ids_sha256_le\":\"%s\","
                "\"no_fallback\":1}",
                index == 0u ? "" : ",", index + 1u,
                run.measured.emitted >= kRequiredMeasuredTokens &&
                                run.exact_session_handoff
                        ? "pass" : "fail",
                static_cast<unsigned long long>(run.measured.accepted),
                static_cast<unsigned long long>(run.measured.proposed),
                static_cast<unsigned long long>(run.measured.emitted),
                static_cast<unsigned long long>(run.measured.cycles),
                static_cast<unsigned long long>(run.measured.cycles),
                run.elapsed_ms, run.decode_tokens_per_second,
                run.acceptance_percent,
                static_cast<unsigned long long>(run.prime.cycles),
                run.handoff_replays, run.measured_final_position,
                run.handoff_final_position,
                run.exact_session_handoff ? 1u : 0u,
                run.output_sha256.c_str());
    }
    std::printf(
            "],\"summary\":{\"accepted\":%llu,\"proposed\":%llu,"
            "\"emitted\":%llu,\"cycles\":%llu,"
            "\"graph_replays\":%llu,\"decode_tokens_per_second\":{"
            "\"min\":%.6f,\"median\":%.6f,\"max\":%.6f,"
            "\"dispersion_percent\":%.6f},"
            "\"sustained_floor_tokens_per_second\":%.1f,"
            "\"floor_320_pass\":%u,\"stretch_tokens_per_second\":%.1f,"
            "\"target_380_pass\":%u,"
            "\"maximum_dispersion_percent\":%.1f,"
            "\"dispersion_pass\":%u},"
            "\"temporal_m8_validated\":%u,\"mtp_tensor_count\":%u,"
            "\"token_output_parity_100pct\":%u,"
            "\"exact_session_handoff\":%u,\"no_fallback\":1}\n",
            static_cast<unsigned long long>(accepted_total),
            static_cast<unsigned long long>(proposed_total),
            static_cast<unsigned long long>(emitted_total),
            static_cast<unsigned long long>(graph_replays_total),
            static_cast<unsigned long long>(graph_replays_total),
            minimum_tps, median_tps, maximum_tps, dispersion_percent,
            kSustainedFloorTokensPerSecond, floor_320_pass ? 1u : 0u,
            kStretchTokensPerSecond, target_380_pass ? 1u : 0u,
            kMaximumDispersionPercent, dispersion_pass ? 1u : 0u,
            temporal_m8_validated, mtp_tensor_count,
            output_parity ? 1u : 0u, exact_handoff ? 1u : 0u);
    return passed ? 0 : 1;
}
