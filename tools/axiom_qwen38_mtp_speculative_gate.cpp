/* End-to-end correctness gate for the lossless native Qwen3.8 MTP controller. */

#include <cuda_runtime_api.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"
#include "axiom/qwen38_mtp_speculative.h"

namespace {

constexpr uint32_t kMaxContext = 16u;
constexpr std::array<uint32_t, 1u> kPromptTokens = {151643u};
constexpr std::array<uint32_t, AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH>
        kTemporalValidationTokens = {
                151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u,
        };

static_assert(
        AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS == 7u,
        "the gate qualifies a gamma-7 controller");
static_assert(
        AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH ==
                AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH,
        "controller and target M8 widths must match");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_embed_f32_device),
                     axiom_qwen38_mtp_target_embed_f32_device_fn>::value,
        "target embedding callback ABI changed");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_lm_head_f32_device),
                     axiom_qwen38_mtp_target_lm_head_f32_device_fn>::value,
        "target LM-head callback ABI changed");

struct RunSnapshot {
    std::array<axiom_qwen38_mtp_speculative_prefill_result, kPromptTokens.size()>
            prefill{};
    axiom_qwen38_mtp_speculative_step_result step{};
    axiom_qwen38_mtp_speculative_info before{};
    axiom_qwen38_mtp_speculative_info after{};
    axiom_qwen38_mtp_compute_info compute_after{};
    uint32_t target_position = 0u;
};

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

int controller_info(
        const axiom_qwen38_mtp_speculative *controller,
        axiom_qwen38_mtp_speculative_info *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
    out->struct_size = sizeof(*out);
    return axiom_qwen38_mtp_speculative_info_get(controller, out);
}

int compute_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    return axiom_qwen38_mtp_compute_info_get(compute, out);
}

bool valid_step_result(const axiom_qwen38_mtp_speculative_step_result &result) {
    const uint32_t accepted = result.accepted_draft_prefix;
    if (result.abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        result.struct_size != sizeof(result) ||
        accepted > AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ||
        result.emitted_token_count != accepted + 1u ||
        result.position_after_commit !=
                result.snapshot_position + result.emitted_token_count ||
        result.continuation_token_id != result.target_token_ids[accepted] ||
        result.full_block_accept !=
                (accepted == AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ? 1u : 0u)) {
        return false;
    }
    for (uint32_t index = 0u;
         index < AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS; ++index) {
        if (result.draft_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB) return false;
        if (index < accepted &&
            result.target_token_ids[index] != result.draft_token_ids[index]) {
            return false;
        }
        if (index == accepted && accepted < AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS &&
            result.target_token_ids[index] == result.draft_token_ids[index]) {
            return false;
        }
    }
    for (uint32_t index = 0u;
         index < AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH; ++index) {
        if (result.target_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB ||
            !std::isfinite(result.target_logits[index])) {
            return false;
        }
    }
    for (uint32_t index = 0u; index < accepted; ++index) {
        if (result.emitted_token_ids[index] != result.draft_token_ids[index]) return false;
    }
    return result.emitted_token_ids[accepted] == result.continuation_token_id;
}

bool valid_counter_delta(const RunSnapshot &run) {
    const uint32_t accepted = run.step.accepted_draft_prefix;
    const bool full = accepted == AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
    return run.after.attempted_steps == run.before.attempted_steps + 1u &&
            run.after.committed_steps == run.before.committed_steps + 1u &&
            run.after.proposed_tokens ==
                    run.before.proposed_tokens +
                            AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS &&
            run.after.accepted_tokens == run.before.accepted_tokens + accepted &&
            run.after.emitted_tokens ==
                    run.before.emitted_tokens + run.step.emitted_token_count &&
            run.after.full_accept_steps == run.before.full_accept_steps + (full ? 1u : 0u) &&
            run.after.correction_steps == run.before.correction_steps + (full ? 0u : 1u) &&
            run.after.failed_steps == run.before.failed_steps;
}

int run_sequence(
        axiom_qwen38_mtp_speculative *controller,
        axiom_qwen38_model *target,
        axiom_qwen38_mtp_compute *compute,
        RunSnapshot *out,
        const char **stage) {
    if (!controller || !target || !compute || !out || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = {};
    *stage = "sequence_info_before";
    int rc = controller_info(controller, &out->before);
    uint32_t anchor = AXIOM_TOKEN_ID_INVALID;
    for (uint32_t index = 0u; index < kPromptTokens.size() && rc == AXIOM_OK; ++index) {
        *stage = "sequence_prefill";
        auto &result = out->prefill[index];
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_prefill_token(
                controller, kPromptTokens[index], &result);
        if (rc == AXIOM_OK &&
            (result.token_position != index ||
             result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
             !std::isfinite(result.target_logit))) {
            *stage = "sequence_prefill_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) anchor = result.target_token_id;
    }
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(target) != kPromptTokens.size() ||
         axiom_qwen38_mtp_compute_position(compute) != kPromptTokens.size())) {
        *stage = "sequence_prefill_positions";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        *stage = "sequence_gamma7";
        axiom_qwen38_mtp_speculative_step_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_id = anchor;
        out->step.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        out->step.struct_size = sizeof(out->step);
        rc = axiom_qwen38_mtp_speculative_step(controller, &request, &out->step);
    }
    if (rc == AXIOM_OK &&
        (out->step.snapshot_position != kPromptTokens.size() ||
         !valid_step_result(out->step))) {
        *stage = "sequence_gamma7_invariants";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        *stage = "sequence_info_after";
        rc = controller_info(controller, &out->after);
    }
    if (rc == AXIOM_OK) {
        *stage = "sequence_compute_info";
        rc = compute_info(compute, &out->compute_after);
    }
    if (rc == AXIOM_OK) {
        out->target_position = axiom_qwen38_model_position(target);
        const uint32_t expected_position = out->step.position_after_commit;
        if (out->after.poisoned != 0u || out->after.position != expected_position ||
            out->target_position != expected_position ||
            axiom_qwen38_mtp_compute_position(compute) != expected_position ||
            out->compute_after.committed_position != expected_position ||
            out->compute_after.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
            !valid_counter_delta(*out)) {
            *stage = "sequence_state_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    return rc;
}

bool repeatable(const RunSnapshot &first, const RunSnapshot &second) {
    return std::memcmp(first.prefill.data(), second.prefill.data(),
                       sizeof(first.prefill)) == 0 &&
            std::memcmp(&first.step, &second.step, sizeof(first.step)) == 0 &&
            first.target_position == second.target_position &&
            first.compute_after.committed_position ==
                    second.compute_after.committed_position;
}

void print_json_float(float value) {
    if (std::isfinite(value)) std::printf("%.9g", static_cast<double>(value));
    else std::fputs("null", stdout);
}

void print_u32_array(const uint32_t *values, uint32_t count) {
    std::fputc('[', stdout);
    for (uint32_t index = 0u; index < count; ++index) {
        std::printf("%s%u", index == 0u ? "" : ",", values[index]);
    }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char **argv) {
    const char *stage = "arguments";
    int rc = AXIOM_OK;
    int device = 0;
    if (argc < 2 || argc > 3 || (argc == 3 && !parse_device(argv[2], &device))) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_qwen38_model *target = nullptr;
    axiom_runtime *runtime = nullptr;
    axiom_model *checkpoint = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    axiom_qwen38_mtp_compute *compute = nullptr;
    axiom_qwen38_mtp_speculative *controller = nullptr;
    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    axiom_qwen38_mtp_info mtp_info{};
    axiom_qwen38_mtp_compute_info initial_compute_info{};
    axiom_qwen38_mtp_speculative_info initial_controller_info{};
    axiom_qwen38_mtp_speculative_info reset_info{};
    RunSnapshot first{};
    RunSnapshot replay{};
    bool reset_ok = false;
    bool replay_equal = false;

    if (rc == AXIOM_OK) {
        stage = "cuda_device";
        rc = cudaSetDevice(device) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        stage = "target_create";
        rc = axiom_qwen38_model_create(argv[1], device, kMaxContext, &target);
    }
    if (rc == AXIOM_OK) {
        stage = "temporal_m8_validate";
        validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                target, kTemporalValidationTokens.data(), 0.0f, &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        stage = "temporal_capabilities";
        capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(target, &capabilities);
        if (rc == AXIOM_OK &&
            (capabilities.temporal_m8_available != 1u ||
             capabilities.temporal_m8_implemented != 1u ||
             capabilities.temporal_m8_validated != 1u ||
             capabilities.temporal_verify_width !=
                     AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH ||
             capabilities.draft_block_size !=
                     AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ||
             capabilities.requires_gdn_causal_rows != 1u ||
             capabilities.requires_attention_triangular_kv != 1u ||
             capabilities.requires_prefix_state_install != 1u)) {
            rc = AXIOM_ERR_NOT_IMPLEMENTED;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "runtime_create";
        axiom_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.backend = AXIOM_BACKEND_CUDA;
        config.device = static_cast<uint32_t>(device);
        rc = axiom_runtime_create(&runtime, &config);
    }
    if (rc == AXIOM_OK) {
        stage = "checkpoint_open";
        axiom_model_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.path = argv[1];
        config.name = "qwen38-mtp-speculative-gate";
        config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        config.placement.abi_version = AXIOM_ABI_VERSION;
        config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        config.placement.device_id = static_cast<uint32_t>(device);
        rc = axiom_model_open(runtime, &checkpoint, &config);
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_load";
        rc = axiom_qwen38_mtp_load(checkpoint, runtime, device, &mtp);
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_contract";
        mtp_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_mtp_info_get(mtp, &mtp_info);
        if (rc == AXIOM_OK &&
            (mtp_info.tensor_count != AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
             mtp_info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
             mtp_info.device_bytes == 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_compute_create";
        axiom_qwen38_mtp_target_binding binding{};
        binding.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        binding.user_data = target;
        binding.embed_f32_device = axiom_qwen38_model_dspark_target_embed_f32_device;
        binding.lm_head_f32_device = axiom_qwen38_model_dspark_target_lm_head_f32_device;
        axiom_qwen38_mtp_compute_config_v2 config{};
        config.abi_version = AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.max_context = kMaxContext;
        config.cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
        config.rope_profile = axiom_qwen38_model_rope_profile(target);
        rc = axiom_qwen38_mtp_compute_create_v2(
                mtp, runtime, device, &config, sizeof(config), &binding, &compute);
    }
    if (rc == AXIOM_OK) {
        stage = "mtp_compute_contract";
        rc = compute_info(compute, &initial_compute_info);
        if (rc == AXIOM_OK &&
            (initial_compute_info.device != static_cast<uint32_t>(device) ||
             initial_compute_info.max_context != kMaxContext ||
             initial_compute_info.cache_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
             initial_compute_info.max_columns !=
                     AXIOM_QWEN38_MTP_COMPUTE_MAX_COLUMNS ||
             initial_compute_info.transaction_state !=
                     AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
             initial_compute_info.committed_position != 0u ||
             initial_compute_info.kv_cache_device_bytes == 0u ||
             initial_compute_info.device_bytes == 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "controller_create";
        axiom_qwen38_mtp_speculative_config config{};
        config.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.device = static_cast<uint32_t>(device);
        rc = axiom_qwen38_mtp_speculative_create_v1(
                target, compute, &config, sizeof(config), &controller);
    }
    if (rc == AXIOM_OK) {
        stage = "controller_initial_state";
        rc = controller_info(controller, &initial_controller_info);
        if (rc == AXIOM_OK &&
            (initial_controller_info.position != 0u ||
             initial_controller_info.poisoned != 0u ||
             initial_controller_info.attempted_steps != 0u ||
             initial_controller_info.committed_steps != 0u ||
             initial_controller_info.failed_steps != 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) rc = run_sequence(controller, target, compute, &first, &stage);
    if (rc == AXIOM_OK) {
        stage = "controller_reset";
        rc = axiom_qwen38_mtp_speculative_reset(controller);
    }
    if (rc == AXIOM_OK) {
        stage = "controller_reset_state";
        rc = controller_info(controller, &reset_info);
        reset_ok = rc == AXIOM_OK && reset_info.position == 0u &&
                reset_info.poisoned == 0u && axiom_qwen38_model_position(target) == 0u &&
                axiom_qwen38_mtp_compute_position(compute) == 0u &&
                reset_info.attempted_steps == first.after.attempted_steps &&
                reset_info.committed_steps == first.after.committed_steps &&
                reset_info.proposed_tokens == first.after.proposed_tokens &&
                reset_info.accepted_tokens == first.after.accepted_tokens &&
                reset_info.emitted_tokens == first.after.emitted_tokens &&
                reset_info.full_accept_steps == first.after.full_accept_steps &&
                reset_info.correction_steps == first.after.correction_steps &&
                reset_info.failed_steps == first.after.failed_steps;
        if (rc == AXIOM_OK && !reset_ok) rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) rc = run_sequence(controller, target, compute, &replay, &stage);
    if (rc == AXIOM_OK) {
        stage = "deterministic_replay";
        replay_equal = repeatable(first, replay);
        if (!replay_equal) rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        stage = "complete";
        if (cudaDeviceSynchronize() != cudaSuccess) {
            stage = "final_cuda_sync";
            rc = AXIOM_ERR_CUDA;
        }
    }

    axiom_qwen38_mtp_speculative_destroy(controller);
    axiom_qwen38_mtp_compute_destroy(compute);
    axiom_qwen38_mtp_destroy(mtp);
    axiom_model_close(checkpoint);
    axiom_runtime_destroy(runtime);
    axiom_qwen38_model_destroy(target);

    const bool passed = rc == AXIOM_OK && validation.passed == 1u && reset_ok &&
            replay_equal && replay.after.poisoned == 0u && replay.after.failed_steps == 0u;
    std::printf(
            "{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"device\":%d,\"max_context\":%u,\"no_fallback\":%u,"
            "\"temporal_m8_validated\":%u,\"temporal_max_logit_error\":",
            passed ? "pass" : "fail", stage, rc, device, kMaxContext,
            passed ? 1u : 0u, validation.passed);
    print_json_float(validation.max_logit_abs_error);
    std::printf(
            ",\"mtp_tensor_count\":%u,\"mtp_device_bytes\":%llu,"
            "\"compute_device_bytes\":%llu,\"mtp_kv_device_bytes\":%llu,"
            "\"prompt_tokens\":%zu,\"first_anchor\":%u,"
            "\"first_snapshot_position\":%u,\"first_accepted_prefix\":%u,"
            "\"first_emitted_tokens\":%u,\"first_position\":%u,"
            "\"reset_ok\":%u,\"replay_equal\":%u,"
            "\"replay_position\":%u,\"controller_position\":%u,"
            "\"target_position\":%u,\"mtp_position\":%u,"
            "\"poisoned\":%u,\"attempted_steps\":%llu,"
            "\"committed_steps\":%llu,\"proposed_tokens\":%llu,"
            "\"accepted_tokens\":%llu,\"emitted_tokens_total\":%llu,"
            "\"failed_steps\":%llu,\"draft_tokens\":",
            mtp_info.tensor_count,
            static_cast<unsigned long long>(mtp_info.device_bytes),
            static_cast<unsigned long long>(initial_compute_info.device_bytes),
            static_cast<unsigned long long>(initial_compute_info.kv_cache_device_bytes),
            kPromptTokens.size(), first.prefill.back().target_token_id,
            first.step.snapshot_position, first.step.accepted_draft_prefix,
            first.step.emitted_token_count, first.step.position_after_commit,
            reset_ok ? 1u : 0u, replay_equal ? 1u : 0u,
            replay.step.position_after_commit, replay.after.position,
            replay.target_position, replay.compute_after.committed_position,
            replay.after.poisoned,
            static_cast<unsigned long long>(replay.after.attempted_steps),
            static_cast<unsigned long long>(replay.after.committed_steps),
            static_cast<unsigned long long>(replay.after.proposed_tokens),
            static_cast<unsigned long long>(replay.after.accepted_tokens),
            static_cast<unsigned long long>(replay.after.emitted_tokens),
            static_cast<unsigned long long>(replay.after.failed_steps));
    print_u32_array(
            first.step.draft_token_ids,
            AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS);
    std::fputs(",\"target_tokens\":", stdout);
    print_u32_array(
            first.step.target_token_ids,
            AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH);
    std::fputs(",\"emitted_tokens\":", stdout);
    print_u32_array(first.step.emitted_token_ids, first.step.emitted_token_count);
    std::fputs("}\n", stdout);
    return passed ? 0 : 1;
}
