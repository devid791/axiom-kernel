/* Native Qwen3.8 target + DSpark end-to-end greedy generation gate. */

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"
#include "axiom/qwen38_dspark_compute.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_speculative.h"

namespace {

constexpr uint32_t kDefaultMaxNew = 64u;
constexpr uint32_t kDefaultMaxContext = 256u;
constexpr uint32_t kTextCapacity = 65536u;
constexpr uint32_t kValidationTokens[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH] = {
    271u, 248068u, 198u, 760u, 1156u, 369u, 40719u, 728u,
};

int fail(const char *stage, int status = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-speculative-generate: %s%s%s\n",
                 stage, status == AXIOM_OK ? "" : ": ",
                 status == AXIOM_OK ? "" : axiom_status_string(status));
    return 1;
}

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

bool is_stop(uint32_t token, const axiom_tokenizer_info &info) {
    return token == info.endoftext_token_id || token == info.im_end_token_id;
}

double seconds_since(
        const std::chrono::steady_clock::time_point &begin,
        const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration<double>(end - begin).count();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr,
                     "usage: %s TARGET_DIR DSPARK_DIR PROMPT [MAX_NEW] [MAX_CONTEXT]\n",
                     argv[0]);
        return 2;
    }
    uint32_t max_new = kDefaultMaxNew;
    uint32_t max_context = kDefaultMaxContext;
    if ((argc >= 5 && (!parse_u32(argv[4], &max_new) || max_new == 0u)) ||
        (argc == 6 && (!parse_u32(argv[5], &max_context) || max_context == 0u))) {
        return fail("invalid MAX_NEW or MAX_CONTEXT");
    }
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        return fail("disable duplicate generic residency");
    }

    axiom_tokenizer *tokenizer = nullptr;
    axiom_qwen38_model *target = nullptr;
    axiom_runtime *draft_runtime = nullptr;
    axiom_model *draft_checkpoint = nullptr;
    axiom_qwen38_dspark *draft = nullptr;
    axiom_qwen38_dspark_compute *compute = nullptr;
    axiom_qwen38_speculative *speculative = nullptr;
    const char *stage = "tokenizer_open";
    int rc = AXIOM_OK;

    axiom_tokenizer_config tokenizer_config{};
    tokenizer_config.abi_version = AXIOM_ABI_VERSION;
    tokenizer_config.path = argv[1];
    tokenizer_config.name = "qwen3.8";
    tokenizer_config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    rc = axiom_tokenizer_open(&tokenizer, &tokenizer_config);
    axiom_tokenizer_info tokenizer_info{};
    tokenizer_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) stage = "tokenizer_info";
    if (rc == AXIOM_OK) rc = axiom_tokenizer_info_get(tokenizer, &tokenizer_info);

    std::vector<uint32_t> prompt_ids;
    std::vector<uint32_t> generated;
    std::vector<char> text;
    try {
        prompt_ids.resize(max_context);
        generated.reserve(static_cast<size_t>(max_new) + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH);
        text.resize(kTextCapacity, '\0');
    } catch (...) {
        rc = AXIOM_ERR_BUDGET;
    }
    uint32_t prompt_count = 0u;
    if (rc == AXIOM_OK) stage = "tokenizer_encode";
    if (rc == AXIOM_OK) {
        rc = axiom_tokenizer_encode_text(
                tokenizer, argv[3], prompt_ids.data(), static_cast<uint32_t>(prompt_ids.size()),
                &prompt_count);
    }
    if (rc == AXIOM_OK &&
        (prompt_count == 0u || prompt_count > max_context ||
         static_cast<uint64_t>(prompt_count) + max_new +
                         AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH >
                 max_context)) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }

    const auto target_load_begin = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) stage = "target_create";
    if (rc == AXIOM_OK) rc = axiom_qwen38_model_create(argv[1], 0, max_context, &target);
    const auto target_load_end = std::chrono::steady_clock::now();

    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    const auto validation_begin = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) stage = "temporal_validate";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                target, kValidationTokens, 0.0f, &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
    }
    std::fprintf(
            stderr,
            "temporal_validation status=%d passed=%u position=%u max_logit_error=%.9g "
            "max_tap_error=%.9g max_tap_bf16_materialization_error=%.9g\n",
            rc, validation.passed, validation.tested_position,
            static_cast<double>(validation.max_logit_abs_error),
            static_cast<double>(validation.max_tap_abs_error),
            static_cast<double>(validation.max_tap_bf16_materialization_abs_error));
    const auto validation_end = std::chrono::steady_clock::now();

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
    const auto draft_load_begin = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) stage = "draft_model_open";
    if (rc == AXIOM_OK) {
        rc = axiom_model_open(draft_runtime, &draft_checkpoint, &checkpoint_config);
    }
    if (rc == AXIOM_OK) stage = "draft_load";
    if (rc == AXIOM_OK) rc = axiom_qwen38_dspark_load(draft_checkpoint, draft_runtime, 0, &draft);

    axiom_qwen38_dspark_target_binding binding{};
    binding.abi_version = AXIOM_ABI_VERSION;
    binding.user_data = target;
    binding.embed_f32_device = axiom_qwen38_model_dspark_target_embed_f32_device;
    binding.lm_head_f32_device = axiom_qwen38_model_dspark_target_lm_head_f32_device;
    axiom_qwen38_dspark_compute_config compute_config{};
    compute_config.abi_version = AXIOM_ABI_VERSION;
    compute_config.max_context = max_context;
    if (rc == AXIOM_OK) stage = "compute_create";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_create(
                draft, draft_runtime, 0, &compute_config, &binding, &compute);
    }
    axiom_qwen38_speculative_config speculative_config{};
    speculative_config.abi_version = AXIOM_ABI_VERSION;
    speculative_config.device = 0;
    if (rc == AXIOM_OK) stage = "speculative_create";
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_speculative_create(
                target, draft, compute, &speculative_config, &speculative);
    }
    const auto draft_load_end = std::chrono::steady_clock::now();

    uint32_t anchor = 0u;
    float anchor_logit = 0.0f;
    const auto prefill_begin = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) stage = "speculative_prefill";
    for (uint32_t index = 0u; index < prompt_count && rc == AXIOM_OK; ++index) {
        axiom_qwen38_speculative_prefill_request request{};
        request.abi_version = AXIOM_ABI_VERSION;
        request.token_id = prompt_ids[index];
        axiom_qwen38_speculative_prefill_result result{};
        result.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_speculative_prefill_token(speculative, &request, &result);
        if (rc == AXIOM_OK) {
            anchor = result.target_token_id;
            anchor_logit = result.target_logit;
        }
    }
    const auto prefill_end = std::chrono::steady_clock::now();
    axiom_qwen38_dspark_compute_info compute_info{};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        stage = "prefill_watermark";
        rc = axiom_qwen38_dspark_compute_info_get(compute, &compute_info);
    }
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(target) != prompt_count ||
         compute_info.transaction_open != 0u ||
         compute_info.committed_position != prompt_count)) {
        rc = AXIOM_ERR_RUNTIME;
    }

    const auto decode_begin = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) stage = "speculative_decode";
    if (rc == AXIOM_OK) generated.push_back(anchor);
    bool stopped = rc == AXIOM_OK && is_stop(anchor, tokenizer_info);
    while (rc == AXIOM_OK && !stopped && generated.size() < max_new) {
        axiom_qwen38_speculative_step_request request{};
        request.abi_version = AXIOM_ABI_VERSION;
        request.anchor_token_id = anchor;
        axiom_qwen38_speculative_step_result result{};
        result.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_speculative_step(speculative, &request, &result);
        if (rc != AXIOM_OK) break;
        for (uint32_t index = 0u;
             index < result.emitted_token_count && generated.size() < max_new;
             ++index) {
            generated.push_back(result.emitted_token_ids[index]);
            if (is_stop(result.emitted_token_ids[index], tokenizer_info)) {
                stopped = true;
                break;
            }
        }
        anchor = result.continuation_token_id;
        anchor_logit = result.continuation_logit;
    }
    const auto decode_end = std::chrono::steady_clock::now();

    /* Surface any late asynchronous CUDA error before accepting the final
     * counters and watermarks as an end-to-end pass. */
    if (rc == AXIOM_OK) {
        stage = "decode_stream_sync";
        if (cudaStreamSynchronize(nullptr) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }

    axiom_qwen38_speculative_counters counters{};
    counters.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) stage = "counters_get";
    if (rc == AXIOM_OK) rc = axiom_qwen38_speculative_counters_get(speculative, &counters);
    compute_info = {};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        stage = "decode_watermark";
        rc = axiom_qwen38_dspark_compute_info_get(compute, &compute_info);
    }
    const bool counter_math_valid =
            counters.attempted_steps == counters.committed_steps &&
            counters.aborted_steps == 0u && counters.failed_steps == 0u &&
            counters.proposed_draft_tokens ==
                    counters.committed_steps * AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS &&
            counters.accepted_draft_tokens + counters.rejected_draft_tokens ==
                    counters.proposed_draft_tokens &&
            counters.authoritative_tail_tokens == counters.committed_steps &&
            counters.emitted_tokens ==
                    counters.accepted_draft_tokens + counters.authoritative_tail_tokens &&
            counters.full_accept_steps + counters.correction_steps ==
                    counters.committed_steps &&
            counters.prefill_attempted_tokens == prompt_count &&
            counters.prefill_committed_tokens == prompt_count &&
            counters.prefill_failed_tokens == 0u;
    if (rc == AXIOM_OK &&
        (!counter_math_valid || compute_info.transaction_open != 0u ||
         compute_info.committed_position != prompt_count + counters.emitted_tokens ||
         axiom_qwen38_model_position(target) != prompt_count + counters.emitted_tokens ||
         counters.poisoned != 0u || counters.last_status != AXIOM_OK ||
         counters.prefill_committed_tokens != prompt_count)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    uint32_t text_bytes = 0u;
    if (rc == AXIOM_OK) stage = "tokenizer_decode";
    if (rc == AXIOM_OK) {
        rc = axiom_tokenizer_decode_ids(
                tokenizer, generated.data(), static_cast<uint32_t>(generated.size()),
                text.data(), static_cast<uint32_t>(text.size()), &text_bytes);
    }

    if (rc == AXIOM_OK) {
        const double target_load_seconds = seconds_since(target_load_begin, target_load_end);
        const double validation_seconds = seconds_since(validation_begin, validation_end);
        const double draft_load_seconds = seconds_since(draft_load_begin, draft_load_end);
        const double prefill_seconds = seconds_since(prefill_begin, prefill_end);
        const double decode_seconds = seconds_since(decode_begin, decode_end);
        const double decode_tok_s = decode_seconds > 0.0
                ? static_cast<double>(counters.emitted_tokens) / decode_seconds
                : 0.0;
        const double acceptance = counters.proposed_draft_tokens != 0u
                ? static_cast<double>(counters.accepted_draft_tokens) /
                          static_cast<double>(counters.proposed_draft_tokens)
                : 0.0;
        std::printf(
                "prompt_tokens=%u visible_generated_tokens=%zu computed_decode_tokens=%llu "
                "target_load_seconds=%.3f "
                "validation_seconds=%.3f dspark_load_seconds=%.3f prefill_seconds=%.3f "
                "decode_seconds=%.6f decode_tok_s=%.3f cycles=%llu full_accept=%llu "
                "proposed=%llu accepted=%llu acceptance=%.6f continuation_id=%u "
                "continuation_logit=%.9g propose_ms=%.3f proposal_copy_ms=%.3f "
                "verify_ms=%.3f inject_ms=%.3f commit_ms=%.3f cycle_total_ms=%.3f "
                "prefill_target_ms=%.3f prefill_inject_ms=%.3f prefill_commit_ms=%.3f\n",
                prompt_count, generated.size(),
                static_cast<unsigned long long>(counters.emitted_tokens),
                target_load_seconds, validation_seconds,
                draft_load_seconds, prefill_seconds, decode_seconds, decode_tok_s,
                static_cast<unsigned long long>(counters.committed_steps),
                static_cast<unsigned long long>(counters.full_accept_steps),
                static_cast<unsigned long long>(counters.proposed_draft_tokens),
                static_cast<unsigned long long>(counters.accepted_draft_tokens), acceptance,
                anchor, static_cast<double>(anchor_logit),
                static_cast<double>(counters.propose_ns) / 1.0e6,
                static_cast<double>(counters.proposal_copy_ns) / 1.0e6,
                static_cast<double>(counters.verify_ns) / 1.0e6,
                static_cast<double>(counters.inject_ns) / 1.0e6,
                static_cast<double>(counters.commit_ns) / 1.0e6,
                static_cast<double>(counters.total_ns) / 1.0e6,
                static_cast<double>(counters.prefill_target_ns) / 1.0e6,
                static_cast<double>(counters.prefill_inject_ns) / 1.0e6,
                static_cast<double>(counters.prefill_commit_ns) / 1.0e6);
        std::fputs("text=", stdout);
        if (text_bytes != 0u) std::fwrite(text.data(), 1u, text_bytes, stdout);
        std::fputc('\n', stdout);
        std::fputs("ids=", stdout);
        for (size_t index = 0u; index < generated.size(); ++index) {
            std::printf("%s%u", index == 0u ? "" : ",", generated[index]);
        }
        std::fputc('\n', stdout);
    }

    axiom_qwen38_speculative_destroy(speculative);
    axiom_qwen38_dspark_compute_destroy(compute);
    axiom_qwen38_dspark_destroy(draft);
    axiom_model_close(draft_checkpoint);
    axiom_runtime_destroy(draft_runtime);
    axiom_qwen38_model_destroy(target);
    axiom_tokenizer_close(tokenizer);
    return rc == AXIOM_OK ? 0 : fail(stage, rc);
}
