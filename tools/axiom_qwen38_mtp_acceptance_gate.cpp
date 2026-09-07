/*
 * Bounded acceptance diagnostic for the lossless native Qwen3.8 MTP path.
 *
 * The fixed 21-token prompt is the exact tokenizer output for the
 * `greeting_it` case in benchmarks/qwen38_speculative_real_corpus.json:
 * "Ciao. Presentati in italiano in due frasi concise e dimmi come puoi
 * aiutarmi oggi."
 *
 * This tool has no target-only fallback.  Any API or invariant failure stops
 * the run immediately and produces one fail-closed JSON record.
 */

#include <cuda_runtime_api.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"
#include "axiom/qwen38_mtp_speculative.h"

namespace {

constexpr uint32_t kMaxContext = 64u;
constexpr uint32_t kCycles = 4u;
constexpr char kPromptCase[] = "greeting_it";
constexpr std::array<uint32_t, 21u> kPromptTokens = {
        34u, 21817u, 13u, 25790u, 9031u, 303u, 57786u,
        303u, 4016u, 1374u, 9911u, 61446u, 378u, 4941u,
        7898u, 2445u, 175507u, 186986u, 203007u, 85593u, 13u,
};
constexpr std::array<uint32_t, AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH>
        kTemporalValidationTokens = {
                151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u,
        };

static_assert(kPromptTokens.size() <= 32u, "diagnostic prompt must remain bounded");
static_assert(kCycles <= 4u, "diagnostic cycle count must remain bounded");
static_assert(
        AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS == 7u,
        "acceptance gate requires gamma-7 proposals");
static_assert(
        AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH ==
                AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH,
        "controller and target verification widths must match");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_embed_f32_device),
                     axiom_qwen38_mtp_target_embed_f32_device_fn>::value,
        "target embedding callback ABI changed");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_lm_head_f32_device),
                     axiom_qwen38_mtp_target_lm_head_f32_device_fn>::value,
        "target LM-head callback ABI changed");

struct CycleObservation {
    uint32_t index = 0u;
    uint32_t anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t proposed = 0u;
    uint32_t accepted = 0u;
    uint32_t emitted = 0u;
    uint32_t snapshot_position = 0u;
    uint32_t position_after = 0u;
    uint32_t continuation = AXIOM_TOKEN_ID_INVALID;
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

bool valid_step(
        const axiom_qwen38_mtp_speculative_step_result &step,
        uint32_t expected_snapshot) {
    const uint32_t accepted = step.accepted_draft_prefix;
    if (step.abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        step.struct_size != sizeof(step) ||
        step.snapshot_position != expected_snapshot ||
        accepted > AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ||
        step.emitted_token_count != accepted + 1u ||
        step.position_after_commit != expected_snapshot + step.emitted_token_count ||
        step.continuation_token_id != step.target_token_ids[accepted] ||
        step.full_block_accept !=
                (accepted == AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS ? 1u : 0u)) {
        return false;
    }
    for (uint32_t index = 0u;
         index < AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS; ++index) {
        if (step.draft_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB) return false;
        if (index < accepted &&
            step.target_token_ids[index] != step.draft_token_ids[index]) {
            return false;
        }
        if (index == accepted &&
            accepted < AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS &&
            step.target_token_ids[index] == step.draft_token_ids[index]) {
            return false;
        }
    }
    for (uint32_t index = 0u;
         index < AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH; ++index) {
        if (step.target_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB ||
            !std::isfinite(step.target_logits[index])) {
            return false;
        }
    }
    for (uint32_t index = 0u; index < accepted; ++index) {
        if (step.emitted_token_ids[index] != step.draft_token_ids[index]) return false;
    }
    return step.emitted_token_ids[accepted] == step.continuation_token_id;
}

void print_cycles(
        const std::array<CycleObservation, kCycles> &cycles,
        uint32_t completed) {
    std::fputc('[', stdout);
    for (uint32_t index = 0u; index < completed; ++index) {
        const CycleObservation &cycle = cycles[index];
        std::printf(
                "%s{\"cycle\":%u,\"anchor\":%u,\"proposed\":%u,"
                "\"accepted\":%u,\"emitted\":%u,"
                "\"snapshot_position\":%u,\"position_after\":%u,"
                "\"continuation\":%u}",
                index == 0u ? "" : ",", cycle.index, cycle.anchor,
                cycle.proposed, cycle.accepted, cycle.emitted,
                cycle.snapshot_position, cycle.position_after,
                cycle.continuation);
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
    axiom_qwen38_mtp_compute_info final_compute_info{};
    axiom_qwen38_mtp_speculative_info final_controller_info{};
    std::array<CycleObservation, kCycles> cycles{};
    uint32_t completed_cycles = 0u;
    uint32_t aggregate_proposed = 0u;
    uint32_t aggregate_accepted = 0u;
    uint32_t aggregate_emitted = 0u;
    uint32_t expected_position = 0u;
    uint32_t anchor = AXIOM_TOKEN_ID_INVALID;
    bool controller_info_valid = false;
    bool compute_info_valid = false;

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
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
                target, &capabilities);
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
        config.name = "qwen38-mtp-acceptance-gate";
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
        binding.lm_head_f32_device =
                axiom_qwen38_model_dspark_target_lm_head_f32_device;
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
        stage = "controller_create";
        axiom_qwen38_mtp_speculative_config config{};
        config.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.device = static_cast<uint32_t>(device);
        rc = axiom_qwen38_mtp_speculative_create_v1(
                target, compute, &config, sizeof(config), &controller);
    }
    for (uint32_t index = 0u;
         index < kPromptTokens.size() && rc == AXIOM_OK; ++index) {
        stage = "prompt_prefill";
        axiom_qwen38_mtp_speculative_prefill_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_prefill_token(
                controller, kPromptTokens[index], &result);
        if (rc == AXIOM_OK &&
            (result.token_position != index ||
             result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
             !std::isfinite(result.target_logit))) {
            stage = "prompt_prefill_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) anchor = result.target_token_id;
    }
    if (rc == AXIOM_OK) {
        expected_position = static_cast<uint32_t>(kPromptTokens.size());
        if (axiom_qwen38_model_position(target) != expected_position ||
            axiom_qwen38_mtp_compute_position(compute) != expected_position) {
            stage = "prompt_position_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    for (uint32_t index = 0u; index < kCycles && rc == AXIOM_OK; ++index) {
        stage = "gamma7_cycle";
        axiom_qwen38_mtp_speculative_step_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_id = anchor;
        axiom_qwen38_mtp_speculative_step_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_step(controller, &request, &result);
        if (rc == AXIOM_OK && !valid_step(result, expected_position)) {
            stage = "gamma7_cycle_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) {
            CycleObservation &cycle = cycles[index];
            cycle.index = index;
            cycle.anchor = anchor;
            cycle.proposed = AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
            cycle.accepted = result.accepted_draft_prefix;
            cycle.emitted = result.emitted_token_count;
            cycle.snapshot_position = result.snapshot_position;
            cycle.position_after = result.position_after_commit;
            cycle.continuation = result.continuation_token_id;
            aggregate_proposed += cycle.proposed;
            aggregate_accepted += cycle.accepted;
            aggregate_emitted += cycle.emitted;
            expected_position = cycle.position_after;
            anchor = cycle.continuation;
            ++completed_cycles;
        }
    }
    if (controller) {
        const int info_rc = controller_info(controller, &final_controller_info);
        controller_info_valid = info_rc == AXIOM_OK;
        if (rc == AXIOM_OK && info_rc != AXIOM_OK) {
            stage = "controller_info";
            rc = info_rc;
        }
    }
    if (compute) {
        const int info_rc = compute_info(compute, &final_compute_info);
        compute_info_valid = info_rc == AXIOM_OK;
        if (rc == AXIOM_OK && info_rc != AXIOM_OK) {
            stage = "compute_info";
            rc = info_rc;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "final_invariants";
        if (!controller_info_valid || !compute_info_valid ||
            completed_cycles != kCycles ||
            aggregate_accepted == 0u ||
            final_controller_info.poisoned != 0u ||
            final_controller_info.position != expected_position ||
            final_controller_info.attempted_steps != kCycles ||
            final_controller_info.committed_steps != kCycles ||
            final_controller_info.proposed_tokens != aggregate_proposed ||
            final_controller_info.accepted_tokens != aggregate_accepted ||
            final_controller_info.emitted_tokens != aggregate_emitted ||
            final_controller_info.failed_steps != 0u ||
            axiom_qwen38_model_position(target) != expected_position ||
            axiom_qwen38_mtp_compute_position(compute) != expected_position ||
            final_compute_info.committed_position != expected_position ||
            final_compute_info.transaction_state !=
                    AXIOM_QWEN38_MTP_TRANSACTION_IDLE) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "final_cuda_sync";
        rc = cudaDeviceSynchronize() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) stage = "complete";

    const uint32_t target_position = target ? axiom_qwen38_model_position(target) : 0u;
    const uint32_t mtp_position = compute ? axiom_qwen38_mtp_compute_position(compute) : 0u;

    axiom_qwen38_mtp_speculative_destroy(controller);
    axiom_qwen38_mtp_compute_destroy(compute);
    axiom_qwen38_mtp_destroy(mtp);
    axiom_model_close(checkpoint);
    axiom_runtime_destroy(runtime);
    axiom_qwen38_model_destroy(target);

    const bool passed = rc == AXIOM_OK;
    std::printf(
            "{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"device\":%d,\"prompt_case\":\"%s\",\"prompt_tokens\":%zu,"
            "\"requested_cycles\":%u,\"completed_cycles\":%u,"
            "\"gamma\":%u,\"no_fallback\":1,\"poisoned\":",
            passed ? "pass" : "fail", stage, rc, device, kPromptCase,
            kPromptTokens.size(), kCycles, completed_cycles,
            AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS);
    if (controller_info_valid) std::printf("%u", final_controller_info.poisoned);
    else std::fputs("null", stdout);
    std::printf(
            ",\"temporal_m8_validated\":%u,\"mtp_tensor_count\":%u,"
            "\"aggregate_proposed\":%u,\"aggregate_accepted\":%u,"
            "\"aggregate_emitted\":%u,\"aggregate_acceptance\":",
            validation.passed, mtp_info.tensor_count, aggregate_proposed,
            aggregate_accepted, aggregate_emitted);
    if (aggregate_proposed != 0u) {
        std::printf("%.9f", static_cast<double>(aggregate_accepted) /
                                  static_cast<double>(aggregate_proposed));
    } else {
        std::fputs("null", stdout);
    }
    std::printf(
            ",\"prefill_position\":%zu,\"expected_final_position\":%u,"
            "\"controller_position\":",
            kPromptTokens.size(), expected_position);
    if (controller_info_valid) std::printf("%u", final_controller_info.position);
    else std::fputs("null", stdout);
    std::printf(
            ",\"target_position\":%u,\"mtp_position\":%u,\"cycles\":",
            target_position, mtp_position);
    print_cycles(cycles, completed_cycles);
    std::fputs("}\n", stdout);
    return passed ? 0 : 1;
}
