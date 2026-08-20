/* End-to-end native Axiom Qwen3.8 forward gate. */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"

namespace {

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-forward-gate: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

bool parse_u32(const char *text, uint32_t *out) {
    if (!text || !out || !*text) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > 0xfffffffful) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [TOKEN_ID] [MAX_CONTEXT]\n", argv[0]);
        return 2;
    }
    uint32_t token = 151643u;
    uint32_t max_context = 4u;
    if ((argc >= 3 && !parse_u32(argv[2], &token)) ||
        (argc == 4 && (!parse_u32(argv[3], &max_context) || max_context == 0u))) {
        return fail("TOKEN_ID and MAX_CONTEXT must be unsigned integers");
    }

    const auto load_start = std::chrono::steady_clock::now();
    axiom_qwen38_model *model = nullptr;
    int rc = axiom_qwen38_model_create(argv[1], 0, max_context, &model);
    const auto load_end = std::chrono::steady_clock::now();
    if (rc != AXIOM_OK) return fail("model create", rc);

    uint32_t first_token = 0u;
    uint32_t second_token = 0u;
    float first_logit = 0.0f;
    float second_logit = 0.0f;
    const auto first_start = std::chrono::steady_clock::now();
    rc = axiom_qwen38_model_forward_token(model, token, &first_token, &first_logit);
    const auto first_end = std::chrono::steady_clock::now();
    if (rc == AXIOM_OK) rc = axiom_qwen38_model_reset(model);
    if (rc == AXIOM_OK) rc = axiom_qwen38_model_forward_token(model, token, &second_token, &second_logit);
    const auto second_end = std::chrono::steady_clock::now();
    const bool finite = std::isfinite(first_logit) && std::isfinite(second_logit);
    const bool pass = rc == AXIOM_OK && finite && first_token == second_token && first_logit == second_logit &&
            axiom_qwen38_model_position(model) == 1u;
    const double load_seconds = std::chrono::duration<double>(load_end - load_start).count();
    const double first_seconds = std::chrono::duration<double>(first_end - first_start).count();
    const double repeat_seconds = std::chrono::duration<double>(second_end - first_end).count();
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-qwen38-full-forward\",\"input_token\":%u,"
                "\"top1_token\":%u,\"top1_logit\":%.9g,\"max_context\":%u,"
                "\"device_bytes\":%llu,\"load_seconds\":%.9g,"
                "\"first_forward_seconds\":%.9g,\"repeat_forward_seconds\":%.9g}\n",
                pass ? "pass" : "fail", token, first_token, static_cast<double>(first_logit), max_context,
                static_cast<unsigned long long>(axiom_qwen38_model_device_bytes(model)), load_seconds,
                first_seconds, repeat_seconds);
    axiom_qwen38_model_destroy(model);
    return pass ? 0 : 1;
}
