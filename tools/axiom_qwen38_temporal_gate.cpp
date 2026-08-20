#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"

namespace {

constexpr uint32_t kMaxContext = 64u;
constexpr uint32_t kInputs[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH] = {
    271u, 248068u, 198u, 760u, 1156u, 369u, 40719u, 728u,
};

int fail(const char *operation, int status) {
    std::fprintf(stderr, "axiom-qwen38-temporal-gate: %s: %s\n",
                 operation, axiom_status_string(status));
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [ABSOLUTE_TOLERANCE]\n", argv[0]);
        return 2;
    }
    float tolerance = 0.0f;
    if (argc == 3) {
        char *end = nullptr;
        tolerance = std::strtof(argv[2], &end);
        if (end == argv[2] || *end != '\0' || !std::isfinite(tolerance) || tolerance < 0.0f) {
            std::fprintf(stderr, "invalid absolute tolerance\n");
            return 2;
        }
    }

    axiom_qwen38_model *model = nullptr;
    int status = axiom_qwen38_model_create(argv[1], 0, kMaxContext, &model);
    if (status != AXIOM_OK) return fail("model create", status);

    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    status = axiom_qwen38_model_dspark_temporal8_validate(
            model, kInputs, tolerance, &validation);
    if (status != AXIOM_OK) {
        axiom_qwen38_model_destroy(model);
        return fail("temporal M8 validation", status);
    }

    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    status = axiom_qwen38_model_dspark_temporal_capabilities_get(model, &capabilities);
    if (status != AXIOM_OK) {
        axiom_qwen38_model_destroy(model);
        return fail("capabilities", status);
    }

    std::printf(
            "passed=%u available=%u implemented=%u validated=%u position=%u "
            "tokens=%u max_logit_abs_error=%.9g max_tap_abs_error=%.9g "
            "max_tap_bf16_materialization_abs_error=%.9g\n",
            validation.passed, capabilities.temporal_m8_available,
            capabilities.temporal_m8_implemented, capabilities.temporal_m8_validated,
            axiom_qwen38_model_position(model), validation.tested_tokens,
            validation.max_logit_abs_error, validation.max_tap_abs_error,
            validation.max_tap_bf16_materialization_abs_error);
    axiom_qwen38_model_destroy(model);
    return validation.passed == 1u && capabilities.temporal_m8_available == 1u ? 0 : 1;
}
