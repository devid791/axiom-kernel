/* Reproducible dense-vs-NVMe-paged boundary gate for native Qwen3.8.
 *
 * The gate deliberately runs two fresh model instances sequentially so the
 * comparison does not require two copies of the 27B checkpoint in VRAM. The
 * streaming leg crosses a 256-token page boundary, flushes page zero, reads it
 * back through the tier, and compares greedy outputs with the resident leg.
 * It never changes an installed service or removes the tier file supplied by
 * the operator.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_kv_tier.h"
#include "axiom/qwen38_model.h"

namespace {

constexpr uint32_t kDefaultContext = 512u;
constexpr uint32_t kDefaultTokens = 300u;
constexpr float kLogitTolerance = 5.0e-2f;

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !text[0]) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > std::numeric_limits<uint32_t>::max()) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

int run_profile(
        const char *model_path,
        axiom_qwen38_kv_tier *tier,
        const bool streaming,
        const uint32_t max_context,
        const std::vector<uint32_t> &inputs,
        std::vector<uint32_t> *out_tokens,
        std::vector<float> *out_logits) {
    if (!model_path || !out_tokens || !out_logits) return AXIOM_ERR_INVALID_ARGUMENT;
    if (streaming) {
        if (!tier || setenv("AXIOM_QWEN38_KV_STREAMING", "1", 1) != 0) {
            return AXIOM_ERR_IO;
        }
    } else {
        (void)unsetenv("AXIOM_QWEN38_KV_STREAMING");
    }
    if (setenv("AXIOM_QWEN38_YARN", "1", 1) != 0 ||
        setenv("AXIOM_QWEN38_COMPACT_KV", "1", 1) != 0 ||
        setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        return AXIOM_ERR_IO;
    }

    const char *profile = streaming ? "paged" : "resident";
    axiom_qwen38_model *model = nullptr;
    int rc = axiom_qwen38_model_create(model_path, 0, max_context, &model);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "paged-runtime-gate: %s model_create failed: %s\n",
                     profile, axiom_status_string(rc));
        return rc;
    }
    if (streaming) rc = axiom_qwen38_model_kv_tier_bind(model, tier);
    out_tokens->clear();
    out_logits->clear();
    out_tokens->reserve(inputs.size());
    out_logits->reserve(inputs.size());
    for (uint32_t token : inputs) {
        uint32_t output = 0u;
        float logit = 0.0f;
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_model_forward_token(model, token, &output, &logit);
        }
        if (rc != AXIOM_OK) {
            std::fprintf(stderr,
                         "paged-runtime-gate: %s forward failed token=%zu position=%u: %s\n",
                         profile, out_tokens->size(), axiom_qwen38_model_position(model),
                         axiom_status_string(rc));
            break;
        }
        out_tokens->push_back(output);
        out_logits->push_back(logit);
    }
    if (rc == AXIOM_OK && streaming) {
        rc = axiom_qwen38_model_kv_tier_flush(model, axiom_qwen38_model_position(model));
        if (rc != AXIOM_OK) {
            std::fprintf(stderr, "paged-runtime-gate: %s tier_flush failed: %s\n",
                         profile, axiom_status_string(rc));
        }
    }
    const int destroy_rc = [&]() {
        axiom_qwen38_model_destroy(model);
        return AXIOM_OK;
    }();
    if (rc == AXIOM_OK) rc = destroy_rc;
    return rc;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr,
                     "usage: %s TARGET_DIR TIER_PATH [TOKENS]\n", argv[0]);
        return 2;
    }
    uint32_t token_count = kDefaultTokens;
    if (argc == 4 && (!parse_u32(argv[3], &token_count) || token_count < 257u)) {
        std::fprintf(stderr, "TOKENS must be >= 257\n");
        return 2;
    }
    const uint32_t max_context = std::max(kDefaultContext, token_count + 1u);
    std::vector<uint32_t> inputs(token_count);
    for (uint32_t index = 0u; index < token_count; ++index) {
        inputs[index] = (index * 7919u + 17u) % 248320u;
    }

    axiom_qwen38_kv_tier_config config{};
    config.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    config.path = argv[2];
    config.max_context = max_context;
    config.hot_pages = 0u;
    config.queue_depth = 64u;
    config.flags = AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT |
            AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING;
    axiom_qwen38_kv_tier *tier = nullptr;
    int rc = axiom_qwen38_kv_tier_create(&config, &tier);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "paged-runtime-gate: tier_create failed: %s\n",
                     axiom_status_string(rc));
        return 1;
    }
    rc = axiom_qwen38_kv_tier_reset(tier);
    std::vector<uint32_t> resident_tokens;
    std::vector<float> resident_logits;
    if (rc == AXIOM_OK) {
        rc = run_profile(argv[1], tier, false, max_context, inputs,
                         &resident_tokens, &resident_logits);
    }
    std::vector<uint32_t> paged_tokens;
    std::vector<float> paged_logits;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_kv_tier_reset(tier);
    }
    if (rc == AXIOM_OK) {
        rc = run_profile(argv[1], tier, true, max_context, inputs,
                         &paged_tokens, &paged_logits);
    }
    uint32_t token_mismatches = 0u;
    float max_logit_error = 0.0f;
    if (rc == AXIOM_OK && resident_tokens.size() == paged_tokens.size() &&
        resident_logits.size() == paged_logits.size()) {
        for (size_t index = 0u; index < resident_tokens.size(); ++index) {
            if (resident_tokens[index] != paged_tokens[index]) ++token_mismatches;
            max_logit_error = std::max(
                    max_logit_error, std::fabs(resident_logits[index] - paged_logits[index]));
        }
    } else if (rc == AXIOM_OK) {
        rc = AXIOM_ERR_RUNTIME;
    }
    axiom_qwen38_kv_tier_info info{};
    info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    if (rc == AXIOM_OK) rc = axiom_qwen38_kv_tier_info_get(tier, &info);
    std::fprintf(stderr,
                 "paged-runtime-gate: rc=%d tokens=%u boundary=256 token_mismatches=%u "
                 "max_logit_abs_error=%.9g committed_tokens=%u\n",
                 rc, token_count, token_mismatches, static_cast<double>(max_logit_error),
                 info.committed_tokens);
    axiom_qwen38_kv_tier_destroy(tier);
    if (rc != AXIOM_OK || token_mismatches != 0u || max_logit_error > kLogitTolerance) {
        return 1;
    }
    return 0;
}
