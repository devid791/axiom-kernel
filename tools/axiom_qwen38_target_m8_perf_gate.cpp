/*
 * Target-only temporal-M8 diagnostic for the canonical Qwen3.8 fixture.
 *
 * Model load, temporal validation, the 1,002-token prefill, one warm-up, the
 * target transaction snapshot and its rollback are excluded from timing.
 * Each CUDA event pair brackets only the public device M8 verification call
 * used by the speculative controller.  The diagnostic never loads MTP.
 */

#include <cuda_runtime_api.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"

namespace {

constexpr uint32_t kMaxContext = 2048u;
constexpr uint32_t kCycles = 33u;
constexpr uint32_t kWidth =
        AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH;
constexpr size_t kCanonicalPromptBytes = 1406u;
constexpr uint32_t kCanonicalPromptTokens = 1002u;
constexpr std::array<uint32_t, kWidth> kTemporalValidationTokens = {
        151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u,
};

static_assert(kWidth == 8u, "target M8 diagnostic requires temporal width eight");
static_assert(kCanonicalPromptTokens + kWidth <= kMaxContext,
              "canonical prompt and M8 block must fit the diagnostic context");

int cuda_status(const cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET
                                                : AXIOM_ERR_CUDA;
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

int tokenize_canonical_prompt(
        const char *model_path,
        std::vector<uint32_t> *token_ids,
        const char **stage) {
    if (!model_path || !token_ids || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "canonical_prompt_build";
    std::string prompt;
    try {
        prompt = canonical_prompt();
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    if (prompt.size() != kCanonicalPromptBytes) return AXIOM_ERR_RUNTIME;

    axiom_tokenizer *tokenizer = nullptr;
    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = model_path;
    config.name = "qwen38-target-m8-perf-gate";
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
        if (token_count != kCanonicalPromptTokens) {
            rc = AXIOM_ERR_RUNTIME;
        } else {
            token_ids->resize(token_count);
        }
    }
    if (tokenizer) axiom_tokenizer_close(tokenizer);
    return rc;
}

struct Resources {
    int device = -1;
    axiom_qwen38_model *model = nullptr;
    cudaStream_t stream = nullptr;
    uint32_t *verify_tokens_device = nullptr;
    uint32_t *position_device = nullptr;
    std::array<cudaEvent_t, kCycles> begin_events{};
    std::array<cudaEvent_t, kCycles> end_events{};
    bool transaction_open = false;

    void release() {
        if (device >= 0) (void)cudaSetDevice(device);
        if (transaction_open && model) {
            (void)axiom_qwen38_model_dspark_device_transaction_abort(
                    model, reinterpret_cast<void *>(stream));
            transaction_open = false;
        }
        if (stream) (void)cudaStreamSynchronize(stream);
        for (cudaEvent_t &event : end_events) {
            if (event) (void)cudaEventDestroy(event);
            event = nullptr;
        }
        for (cudaEvent_t &event : begin_events) {
            if (event) (void)cudaEventDestroy(event);
            event = nullptr;
        }
        (void)cudaFree(position_device);
        position_device = nullptr;
        (void)cudaFree(verify_tokens_device);
        verify_tokens_device = nullptr;
        if (stream) (void)cudaStreamDestroy(stream);
        stream = nullptr;
        axiom_qwen38_model_destroy(model);
        model = nullptr;
    }
};

int enqueue_target_m8(
        Resources *resources,
        const cudaEvent_t begin_event,
        const cudaEvent_t end_event,
        const char **stage) {
    if (!resources || !resources->model || !resources->stream ||
        !resources->verify_tokens_device || !resources->position_device ||
        !stage || resources->transaction_open) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    *stage = "target_transaction_begin";
    int rc = axiom_qwen38_model_dspark_device_transaction_begin(
            resources->model, resources->position_device,
            reinterpret_cast<void *>(resources->stream));
    if (rc == AXIOM_OK) resources->transaction_open = true;
    if (rc == AXIOM_OK && begin_event) {
        *stage = "target_event_begin";
        rc = cuda_status(cudaEventRecord(begin_event, resources->stream));
    }

    axiom_qwen38_model_dspark_device_target_verify_view view{};
    view.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        *stage = "target_m8_verify";
        rc = axiom_qwen38_model_dspark_device_transaction_verify(
                resources->model, resources->verify_tokens_device,
                resources->position_device, kWidth,
                reinterpret_cast<void *>(resources->stream), &view);
    }
    if (rc == AXIOM_OK && end_event) {
        *stage = "target_event_end";
        rc = cuda_status(cudaEventRecord(end_event, resources->stream));
    }
    if (rc == AXIOM_OK &&
        (!view.target_taps_device || !view.target_token_ids_device ||
         !view.target_logits_device ||
         view.target_tap_count != AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT ||
         view.target_tap_tokens != kWidth || view.target_tap_hidden_size != 5120u)) {
        *stage = "target_m8_result_contract";
        rc = AXIOM_ERR_RUNTIME;
    }

    if (resources->transaction_open) {
        const int abort_rc =
                axiom_qwen38_model_dspark_device_transaction_abort(
                        resources->model,
                        reinterpret_cast<void *>(resources->stream));
        resources->transaction_open = false;
        if (rc == AXIOM_OK && abort_rc != AXIOM_OK) {
            *stage = "target_transaction_abort";
            rc = abort_rc;
        }
    }
    return rc;
}

void print_result(
        const bool passed,
        const char *stage,
        const int rc,
        const size_t prompt_tokens,
        const uint32_t cycles,
        const double elapsed_ms,
        const double target_steps_per_second) {
    std::printf(
            "{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"prompt_tokens\":%zu,\"cycles\":%u,\"elapsed_ms\":",
            passed ? "pass" : "fail", stage, rc, prompt_tokens, cycles);
    if (std::isfinite(elapsed_ms)) std::printf("%.6f", elapsed_ms);
    else std::fputs("null", stdout);
    std::fputs(",\"target_steps_per_second\":", stdout);
    if (std::isfinite(target_steps_per_second)) {
        std::printf("%.6f", target_steps_per_second);
    } else {
        std::fputs("null", stdout);
    }
    std::fputs("}\n", stdout);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr,
                "usage: %s ABLITERATED_CHECKPOINT_DIR DEVICE\n", argv[0]);
        return 2;
    }

    const char *stage = "arguments";
    int rc = AXIOM_OK;
    int device = 0;
    if (!parse_device(argv[2], &device)) {
        std::fprintf(stderr, "invalid CUDA device\n");
        return 2;
    }
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        stage = "disable_duplicate_generic_residency";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK &&
        setenv("AXIOM_QWEN38_MATMUL_AUTOTUNE", "1", 0) != 0) {
        stage = "enable_production_matmul_autotune";
        rc = AXIOM_ERR_RUNTIME;
    }

    std::vector<uint32_t> prompt_ids;
    if (rc == AXIOM_OK) {
        rc = tokenize_canonical_prompt(argv[1], &prompt_ids, &stage);
    }

    Resources resources{};
    resources.device = device;
    if (rc == AXIOM_OK) {
        stage = "cuda_device";
        rc = cuda_status(cudaSetDevice(device));
    }
    if (rc == AXIOM_OK) {
        stage = "target_create";
        rc = axiom_qwen38_model_create(
                argv[1], device, kMaxContext, &resources.model);
    }

    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        stage = "target_m8_validate";
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                resources.model, kTemporalValidationTokens.data(), 0.0f,
                &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
    }

    uint32_t next_token = AXIOM_TOKEN_ID_INVALID;
    for (uint32_t token_id : prompt_ids) {
        if (rc != AXIOM_OK) break;
        stage = "canonical_prompt_prefill";
        float next_logit = 0.0f;
        rc = axiom_qwen38_model_forward_token(
                resources.model, token_id, &next_token, &next_logit);
        if (rc == AXIOM_OK &&
            (next_token >= AXIOM_QWEN38_MODEL_VOCAB ||
             !std::isfinite(next_logit))) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK &&
        axiom_qwen38_model_position(resources.model) != prompt_ids.size()) {
        stage = "canonical_prompt_position";
        rc = AXIOM_ERR_RUNTIME;
    }

    std::array<uint32_t, kWidth> verify_tokens{};
    if (rc == AXIOM_OK) {
        verify_tokens[0] = next_token;
        for (uint32_t index = 1u; index < kWidth; ++index) {
            verify_tokens[index] =
                    prompt_ids[prompt_ids.size() - (kWidth - index)];
        }
        stage = "target_stream_create";
        rc = cuda_status(cudaStreamCreateWithFlags(
                &resources.stream, cudaStreamNonBlocking));
    }
    if (rc == AXIOM_OK) {
        stage = "target_verify_tokens_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&resources.verify_tokens_device),
                static_cast<size_t>(kWidth) * sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        stage = "target_position_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&resources.position_device),
                sizeof(uint32_t)));
    }
    const uint32_t prompt_position = static_cast<uint32_t>(prompt_ids.size());
    if (rc == AXIOM_OK) {
        stage = "target_inputs_copy";
        rc = cuda_status(cudaMemcpyAsync(
                resources.verify_tokens_device, verify_tokens.data(),
                static_cast<size_t>(kWidth) * sizeof(uint32_t),
                cudaMemcpyHostToDevice, resources.stream));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                resources.position_device, &prompt_position, sizeof(prompt_position),
                cudaMemcpyHostToDevice, resources.stream));
    }
    if (rc == AXIOM_OK) {
        stage = "target_inputs_ready";
        rc = cuda_status(cudaStreamSynchronize(resources.stream));
    }

    if (rc == AXIOM_OK) {
        stage = "target_m8_warmup";
        rc = enqueue_target_m8(&resources, nullptr, nullptr, &stage);
    }
    if (rc == AXIOM_OK) {
        stage = "target_m8_warmup_sync";
        rc = cuda_status(cudaStreamSynchronize(resources.stream));
    }
    for (uint32_t index = 0u; index < kCycles && rc == AXIOM_OK; ++index) {
        stage = "target_event_create";
        rc = cuda_status(cudaEventCreateWithFlags(
                &resources.begin_events[index], cudaEventDefault));
        if (rc == AXIOM_OK) {
            rc = cuda_status(cudaEventCreateWithFlags(
                    &resources.end_events[index], cudaEventDefault));
        }
    }

    uint32_t measured_cycles = 0u;
    for (; measured_cycles < kCycles && rc == AXIOM_OK; ++measured_cycles) {
        rc = enqueue_target_m8(
                &resources, resources.begin_events[measured_cycles],
                resources.end_events[measured_cycles], &stage);
    }
    if (rc == AXIOM_OK) {
        stage = "target_m8_measurement_sync";
        rc = cuda_status(cudaStreamSynchronize(resources.stream));
    }

    double elapsed_ms = 0.0;
    for (uint32_t index = 0u; index < measured_cycles && rc == AXIOM_OK;
         ++index) {
        float cycle_ms = 0.0f;
        stage = "target_m8_elapsed_time";
        rc = cuda_status(cudaEventElapsedTime(
                &cycle_ms, resources.begin_events[index],
                resources.end_events[index]));
        if (rc == AXIOM_OK && (!std::isfinite(cycle_ms) || cycle_ms <= 0.0f)) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) elapsed_ms += static_cast<double>(cycle_ms);
    }
    double target_steps_per_second =
            std::numeric_limits<double>::quiet_NaN();
    if (rc == AXIOM_OK && elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
        target_steps_per_second =
                static_cast<double>(measured_cycles) * 1000.0 / elapsed_ms;
        stage = "complete";
    } else {
        elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    }

    resources.release();
    const bool passed = rc == AXIOM_OK && measured_cycles == kCycles &&
            std::isfinite(target_steps_per_second);
    print_result(
            passed, stage, rc, prompt_ids.size(), measured_cycles, elapsed_ms,
            target_steps_per_second);
    return passed ? 0 : 1;
}
