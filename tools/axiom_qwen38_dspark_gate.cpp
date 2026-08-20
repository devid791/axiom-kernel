/* Correctness gate for the Qwen3.8 target-side DSpark temporal-M8 ABI. */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark_compute.h"
#include "axiom/qwen38_model.h"

namespace {

constexpr uint32_t kDefaultMaxContext = 32u;
constexpr uint32_t kDefaultPrefillTokens = 4u;
constexpr float kDefaultTolerance = 1.0e-3f;
constexpr uint32_t kCommitPrefix = 4u;

static_assert(kCommitPrefix > 0u &&
                      kCommitPrefix < AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH,
              "continuity probe must leave one fixed token to consume");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_embed_f32_device),
                     axiom_qwen38_dspark_target_embed_f32_device_fn>::value,
        "target embedding callback must stay ABI-compatible with DSpark compute");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_lm_head_f32_device),
                     axiom_qwen38_dspark_target_lm_head_f32_device_fn>::value,
        "target LM-head callback must stay ABI-compatible with DSpark compute");

/* Fixed valid Qwen ids.  They are deliberately not a greedy continuation:
 * the parity gate must validate causality for arbitrary same-session inputs. */
constexpr std::array<uint32_t, 8u> kPrefillTokens = {
        151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u};
constexpr std::array<uint32_t, AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH>
        kTemporalTokens = {151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u};

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

bool parse_f32(const char *text, float *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value) || value < 0.0f) return false;
    *out = value;
    return true;
}

int consume_tokens(
        axiom_qwen38_model *model,
        const uint32_t *tokens,
        uint32_t count) {
    if (!model || (!tokens && count != 0u)) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t index = 0u; index < count; ++index) {
        uint32_t ignored_token = 0u;
        float ignored_logit = 0.0f;
        const int rc = axiom_qwen38_model_forward_token(
                model, tokens[index], &ignored_token, &ignored_logit);
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

int reset_and_replay(
        axiom_qwen38_model *model,
        uint32_t prefill_count,
        uint32_t committed_temporal_prefix) {
    int rc = axiom_qwen38_model_reset(model);
    if (rc == AXIOM_OK) rc = consume_tokens(model, kPrefillTokens.data(), prefill_count);
    if (rc == AXIOM_OK) {
        rc = consume_tokens(model, kTemporalTokens.data(), committed_temporal_prefix);
    }
    return rc;
}

void print_tokens() {
    std::fputs("\"validation_tokens\":[", stdout);
    for (uint32_t index = 0u; index < kTemporalTokens.size(); ++index) {
        std::printf("%s%u", index == 0u ? "" : ",", kTemporalTokens[index]);
    }
    std::fputs("]", stdout);
}

void print_json_f32(float value) {
    if (std::isfinite(value)) {
        std::printf("%.9g", static_cast<double>(value));
    } else {
        std::fputs("null", stdout);
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 5) {
        std::fprintf(stderr,
                     "usage: %s MODEL_DIR [MAX_CONTEXT] [PREFILL_TOKENS_0_TO_8] [ABS_TOLERANCE]\n",
                     argv[0]);
        return 2;
    }

    uint32_t max_context = kDefaultMaxContext;
    uint32_t prefill_count = kDefaultPrefillTokens;
    float tolerance = kDefaultTolerance;
    if ((argc >= 3 && (!parse_u32(argv[2], &max_context) || max_context == 0u)) ||
        (argc >= 4 && (!parse_u32(argv[3], &prefill_count) ||
                       prefill_count > kPrefillTokens.size())) ||
        (argc == 5 && !parse_f32(argv[4], &tolerance))) {
        std::fprintf(stderr, "axiom-qwen38-dspark-gate: invalid argument\n");
        return 2;
    }
    if (max_context < prefill_count + AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH) {
        std::fprintf(stderr,
                     "axiom-qwen38-dspark-gate: MAX_CONTEXT must fit prefill plus temporal width\n");
        return 2;
    }

    axiom_qwen38_model *model = nullptr;
    axiom_qwen38_model_transaction *transaction = nullptr;
    axiom_qwen38_model_dspark_temporal_capabilities capabilities_before{};
    axiom_qwen38_model_dspark_temporal_capabilities capabilities_after{};
    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    axiom_qwen38_model_dspark_verify_block8_result verified{};
    const char *stage = "model_create";
    int rc = axiom_qwen38_model_create(argv[1], 0, max_context, &model);
    uint32_t position_after_validate = 0u;
    uint32_t position_after_commit = 0u;
    uint32_t position_after_replay = 0u;
    uint32_t committed_next_token = 0u;
    uint32_t replay_next_token = 0u;
    float committed_next_logit = 0.0f;
    float replay_next_logit = 0.0f;
    float continuity_logit_error = std::numeric_limits<float>::infinity();
    bool position_restored = false;
    bool binding_ok = false;
    bool continuity_ok = false;

    if (rc == AXIOM_OK) {
        axiom_qwen38_dspark_target_binding target_binding{};
        target_binding.abi_version = AXIOM_ABI_VERSION;
        target_binding.user_data = model;
        target_binding.embed_f32_device = axiom_qwen38_model_dspark_target_embed_f32_device;
        target_binding.lm_head_f32_device = axiom_qwen38_model_dspark_target_lm_head_f32_device;
        binding_ok = target_binding.abi_version == AXIOM_ABI_VERSION &&
                target_binding.user_data == model && target_binding.embed_f32_device != nullptr &&
                target_binding.lm_head_f32_device != nullptr;
        if (!binding_ok) {
            stage = "compute_binding";
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "prefill";
        rc = consume_tokens(model, kPrefillTokens.data(), prefill_count);
    }
    if (rc == AXIOM_OK && axiom_qwen38_model_position(model) != prefill_count) {
        stage = "prefill_position";
        rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        stage = "capabilities_before";
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(model, &capabilities_before);
    }
    if (rc == AXIOM_OK) {
        stage = "temporal8_validate";
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                model, kTemporalTokens.data(), tolerance, &validation);
    }
    if (rc == AXIOM_OK) {
        position_after_validate = axiom_qwen38_model_position(model);
        position_restored = position_after_validate == prefill_count;
        if (!position_restored || validation.passed != 1u) {
            stage = "validation_result";
            rc = AXIOM_ERR_CUDA;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "capabilities_after";
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(model, &capabilities_after);
    }
    if (rc == AXIOM_OK && (capabilities_after.temporal_m8_available != 1u ||
                           capabilities_after.temporal_m8_validated != 1u ||
                           capabilities_after.temporal_verify_width !=
                                   AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH ||
                           capabilities_after.target_tap_count !=
                                   AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT)) {
        stage = "capability_gate";
        rc = AXIOM_ERR_NOT_IMPLEMENTED;
    }

    /* This is intentionally external to temporal8_validate(): prove that a
     * committed M8 prefix produces the same next scalar result as a reset and
     * replay of exactly that prefix. */
    if (rc == AXIOM_OK) {
        stage = "transaction_begin";
        rc = axiom_qwen38_model_transaction_begin(model, &transaction);
    }
    if (rc == AXIOM_OK) {
        stage = "transaction_verify";
        rc = axiom_qwen38_model_transaction_verify_block8(
                transaction, kTemporalTokens.data(), nullptr, &verified);
    }
    if (rc == AXIOM_OK) {
        stage = "transaction_commit_prefix";
        rc = axiom_qwen38_model_transaction_commit_prefix(transaction, kCommitPrefix);
        transaction = nullptr;  // Commit consumes the handle on success and failure.
    }
    if (rc == AXIOM_OK) {
        position_after_commit = axiom_qwen38_model_position(model);
        if (position_after_commit != prefill_count + kCommitPrefix) {
            stage = "commit_position";
            rc = AXIOM_ERR_CUDA;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "committed_continuation";
        rc = axiom_qwen38_model_forward_token(
                model, kTemporalTokens[kCommitPrefix], &committed_next_token, &committed_next_logit);
    }
    if (rc == AXIOM_OK) {
        stage = "reset_replay";
        rc = reset_and_replay(model, prefill_count, kCommitPrefix);
    }
    if (rc == AXIOM_OK) {
        position_after_replay = axiom_qwen38_model_position(model);
        if (position_after_replay != prefill_count + kCommitPrefix) {
            stage = "replay_position";
            rc = AXIOM_ERR_CUDA;
        }
    }
    if (rc == AXIOM_OK) {
        stage = "replay_continuation";
        rc = axiom_qwen38_model_forward_token(
                model, kTemporalTokens[kCommitPrefix], &replay_next_token, &replay_next_logit);
    }
    if (rc == AXIOM_OK) {
        continuity_logit_error = std::fabs(committed_next_logit - replay_next_logit);
        continuity_ok = committed_next_token == replay_next_token &&
                std::isfinite(continuity_logit_error) && continuity_logit_error <= tolerance &&
                axiom_qwen38_model_position(model) == prefill_count + kCommitPrefix + 1u;
        if (!continuity_ok) {
            stage = "commit_continuity";
            rc = AXIOM_ERR_CUDA;
        }
    }

    if (transaction) {
        const int abort_rc = axiom_qwen38_model_transaction_abort(transaction);
        transaction = nullptr;
        if (rc == AXIOM_OK && abort_rc != AXIOM_OK) {
            stage = "transaction_abort";
            rc = abort_rc;
        }
    }
    const bool passed = rc == AXIOM_OK && validation.passed == 1u && position_restored &&
            binding_ok && continuity_ok;
    std::printf("{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
                "\"max_context\":%u,\"prefill_tokens\":%u,\"tolerance\":%.9g,"
                "\"validation_passed\":%u,\"validation_position\":%u,"
                "\"position_after_validate\":%u,\"position_restored\":%u,"
                "\"max_logit_abs_error\":",
                passed ? "pass" : "fail", stage, rc, max_context, prefill_count,
                static_cast<double>(tolerance), validation.passed, validation.tested_position,
                position_after_validate, position_restored ? 1u : 0u);
    print_json_f32(validation.max_logit_abs_error);
    std::fputs(",\"max_tap_abs_error\":", stdout);
    print_json_f32(validation.max_tap_abs_error);
    std::fputs(",\"max_tap_bf16_materialization_abs_error\":", stdout);
    print_json_f32(validation.max_tap_bf16_materialization_abs_error);
    std::printf(",\"temporal_m8_available_before\":%u,\"temporal_m8_available_after\":%u,"
                "\"temporal_m8_validated_after\":%u,\"compute_binding_ok\":%u,"
                "\"commit_prefix\":%u,\"position_after_commit\":%u,"
                "\"position_after_replay\":%u,\"commit_top1\":%u,\"replay_top1\":%u,"
                "\"commit_continuity_logit_abs_error\":",
                capabilities_before.temporal_m8_available,
                capabilities_after.temporal_m8_available,
                capabilities_after.temporal_m8_validated, binding_ok ? 1u : 0u,
                kCommitPrefix, position_after_commit, position_after_replay,
                committed_next_token, replay_next_token);
    print_json_f32(continuity_logit_error);
    std::printf(",\"commit_continuity\":%u,", continuity_ok ? 1u : 0u);
    print_tokens();
    std::fputs("}\n", stdout);

    axiom_qwen38_model_destroy(model);
    return passed ? 0 : 1;
}
