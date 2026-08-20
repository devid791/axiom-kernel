/* Isolated native Qwen3.8 vision gate.
 *
 * It loads only model.visual.* tensors, synthesizes one deterministic 14x14
 * image patch grid, runs the full CUDA tower twice, and checks finite output
 * plus deterministic replay. It never creates the text model or the HTTP
 * service, so it is safe to execute beside the production API when the GPU
 * memory preflight passes.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_vision.h"

namespace {

axiom_model *open_model(axiom_runtime *runtime, const char *path) {
    axiom_model_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = path;
    config.name = "qwen38-vision-gate";
    config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    config.placement.abi_version = AXIOM_ABI_VERSION;
    config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    axiom_model *model = nullptr;
    return axiom_model_open(runtime, &model, &config) == AXIOM_OK ? model : nullptr;
}

bool finite_values(const std::vector<float> &values, float *out_min, float *out_max) {
    if (values.empty()) return false;
    float minimum = values[0];
    float maximum = values[0];
    for (float value : values) {
        if (!std::isfinite(value)) return false;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    if (out_min) *out_min = minimum;
    if (out_max) *out_max = maximum;
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [MAX_PATCH_TOKENS]\n", argv[0]);
        return 2;
    }
    uint32_t max_tokens = 512u;
    if (argc == 3) {
        char *end = nullptr;
        const unsigned long parsed = std::strtoul(argv[2], &end, 10);
        if (!end || *end != '\0' || parsed < 4ul || parsed > 65536ul) {
            std::fprintf(stderr, "invalid MAX_PATCH_TOKENS\n");
            return 2;
        }
        max_tokens = static_cast<uint32_t>(parsed);
    }

    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::fprintf(stderr, "VISION_GATE status=fail stage=cuda_mem_info\n");
        return 1;
    }
    if (free_bytes < (1536ull * 1024ull * 1024ull)) {
        std::fprintf(stderr,
                "VISION_GATE status=skip stage=memory_preflight free_bytes=%zu required_bytes=%llu\n",
                free_bytes, 1536ull * 1024ull * 1024ull);
        return 3;
    }

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0u;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_config);
    axiom_model *model = nullptr;
    axiom_qwen38_vision *vision = nullptr;
    const char *stage = "runtime_create";
    if (rc == AXIOM_OK) {
        stage = "model_open";
        model = open_model(runtime, argv[1]);
        if (!model) rc = AXIOM_ERR_IO;
    }
    if (rc == AXIOM_OK) {
        stage = "vision_create";
        rc = axiom_qwen38_vision_create(model, 0, max_tokens, &vision);
    }

    axiom_qwen38_vision_info info{};
    info.abi_version = AXIOM_QWEN38_VISION_ABI_VERSION;
    if (rc == AXIOM_OK) rc = axiom_qwen38_vision_info_get(vision, &info);

    constexpr uint32_t kHeight = 14u;
    constexpr uint32_t kWidth = 14u;
    constexpr uint32_t kPatches = kHeight * kWidth;
    constexpr uint32_t kOutputTokens = (kHeight / 2u) * (kWidth / 2u);
    std::vector<float> patches(static_cast<size_t>(kPatches) * AXIOM_QWEN38_VISION_PATCH_FEATURES);
    for (uint32_t patch = 0u; patch < kPatches; ++patch) {
        for (uint32_t feature = 0u; feature < AXIOM_QWEN38_VISION_PATCH_FEATURES; ++feature) {
            const float x = static_cast<float>((patch * 131u + feature * 17u) % 997u) / 996.0f;
            patches[static_cast<size_t>(patch) * AXIOM_QWEN38_VISION_PATCH_FEATURES + feature] =
                    (x - 0.5f) * 2.0f;
        }
    }
    const axiom_qwen38_vision_grid grid{1u, kHeight, kWidth};
    std::vector<float> output_a(static_cast<size_t>(kOutputTokens) * AXIOM_QWEN38_VISION_OUTPUT);
    std::vector<float> output_b(output_a.size());
    uint32_t output_tokens_a = 0u;
    if (rc == AXIOM_OK) {
        stage = "forward_first";
        rc = axiom_qwen38_vision_forward_patches(
                vision, patches.data(), patches.size(), &grid, 1u,
                output_a.data(), output_a.size(), &output_tokens_a);
    }
    uint32_t output_tokens_b = 0u;
    if (rc == AXIOM_OK) {
        stage = "forward_replay";
        rc = axiom_qwen38_vision_forward_patches(
                vision, patches.data(), patches.size(), &grid, 1u,
                output_b.data(), output_b.size(), &output_tokens_b);
    }
    float min_value = 0.0f;
    float max_value = 0.0f;
    bool finite = false;
    float max_diff = 0.0f;
    if (rc == AXIOM_OK) {
        stage = "assertions";
        finite = finite_values(output_a, &min_value, &max_value) && finite_values(output_b, nullptr, nullptr);
        for (size_t index = 0u; index < output_a.size(); ++index) {
            max_diff = std::max(max_diff, std::fabs(output_a[index] - output_b[index]));
        }
        if (!finite || output_tokens_a != kOutputTokens || output_tokens_b != kOutputTokens ||
            max_diff != 0.0f || info.depth != AXIOM_QWEN38_VISION_DEPTH ||
            info.loaded_tensor_count != 333u) {
            rc = AXIOM_ERR_CUDA;
        }
    }

    std::printf(
            "VISION_GATE status=%s stage=%s depth=%u loaded_tensors=%llu loaded_bytes=%llu "
            "device_bytes=%llu patch_tokens=%u output_tokens=%u finite=%u max_diff=%g "
            "range=[%g,%g] free_before=%zu total=%zu\n",
            rc == AXIOM_OK ? "pass" : "fail", stage, info.depth,
            static_cast<unsigned long long>(info.loaded_tensor_count),
            static_cast<unsigned long long>(info.loaded_tensor_bytes),
            static_cast<unsigned long long>(info.device_bytes), kPatches, output_tokens_a,
            finite ? 1u : 0u, max_diff, min_value, max_value, free_bytes, total_bytes);

    axiom_qwen38_vision_destroy(vision);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return rc == AXIOM_OK ? 0 : 1;
}
