/* Resident Qwen3.8 full-attention cache/weight gate. */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_attention.h"

namespace {

constexpr uint32_t kHidden = AXIOM_QWEN38_ATTENTION_HIDDEN;
constexpr uint32_t kBatch = AXIOM_QWEN38_ATTENTION_BATCH;

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-attention-probe: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

void fill_input(std::vector<float> *values) {
    for (uint32_t column = 0u; column < kBatch; ++column) {
        for (uint32_t i = 0u; i < kHidden; ++i) {
            (*values)[static_cast<size_t>(column) * kHidden + i] =
                    0.37f * std::sin(0.0023f * static_cast<float>(i) + 0.11f * column) +
                    0.31f * std::cos(0.0059f * static_cast<float>(i) - 0.037f * column);
        }
    }
}

bool all_finite(const std::vector<float> &values) {
    for (float value : values) if (!std::isfinite(value)) return false;
    return true;
}

float max_abs_delta(const std::vector<float> &left, const std::vector<float> &right) {
    float maximum = 0.0f;
    for (size_t i = 0u; i < left.size(); ++i) {
        const float delta = std::fabs(left[i] - right[i]);
        maximum = delta > maximum ? delta : maximum;
    }
    return maximum;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [ATTENTION_LAYER] [MAX_CONTEXT]\n", argv[0]);
        return 2;
    }
    char *end = nullptr;
    const uint32_t layer_index = argc >= 3
            ? static_cast<uint32_t>(std::strtoul(argv[2], &end, 10)) : 3u;
    if ((argc >= 3 && (end == argv[2] || *end != '\0')) || (layer_index % 4u) != 3u) {
        return fail("attention layer must be 3 modulo 4");
    }
    end = nullptr;
    const uint32_t max_context = argc == 4
            ? static_cast<uint32_t>(std::strtoul(argv[3], &end, 10)) : 16u;
    if ((argc == 4 && (end == argv[3] || *end != '\0')) || max_context < 2u) {
        return fail("MAX_CONTEXT must be at least two");
    }

    axiom_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.backend = AXIOM_BACKEND_CUDA;
    config.device = 0;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &config);
    if (rc != AXIOM_OK) return fail("runtime", rc);
    axiom_model_config model_config{};
    model_config.abi_version = AXIOM_ABI_VERSION;
    model_config.path = argv[1];
    model_config.name = "unsloth-qwen3.8-27b-nvfp4";
    model_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_config.placement.abi_version = AXIOM_ABI_VERSION;
    model_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    axiom_model *model = nullptr;
    rc = axiom_model_open(runtime, &model, &model_config);
    if (rc != AXIOM_OK) {
        axiom_runtime_destroy(runtime);
        return fail("model", rc);
    }
    axiom_qwen38_attention_layer *layer = nullptr;
    rc = axiom_qwen38_attention_layer_load(model, runtime, 0, layer_index, max_context, &layer);
    axiom_model_close(model);
    if (rc != AXIOM_OK) {
        axiom_runtime_destroy(runtime);
        return fail("attention load", rc);
    }

    const size_t bytes = static_cast<size_t>(kHidden) * kBatch * sizeof(float);
    std::vector<float> input(bytes / sizeof(float));
    std::vector<float> first(bytes / sizeof(float));
    std::vector<float> second(bytes / sizeof(float));
    std::vector<float> after_reset(bytes / sizeof(float));
    fill_input(&input);
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaError_t status = cudaMalloc(&d_input, bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_output, bytes);
    if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (d_output) (void)cudaFree(d_output);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_attention_layer_destroy(layer);
        axiom_runtime_destroy(runtime);
        return fail("device setup");
    }
    rc = axiom_qwen38_attention_layer_forward_f32_device(layer, d_input, d_output, nullptr);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(first.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = axiom_qwen38_attention_layer_forward_f32_device(layer, d_input, d_output, nullptr);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(second.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    const uint32_t position_after_two = axiom_qwen38_attention_layer_position(layer);
    if (rc == AXIOM_OK) rc = axiom_qwen38_attention_layer_reset(layer);
    if (rc == AXIOM_OK) rc = axiom_qwen38_attention_layer_forward_f32_device(layer, d_input, d_output, nullptr);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(after_reset.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    const bool finite = rc == AXIOM_OK && all_finite(first) && all_finite(second) && all_finite(after_reset);
    const float state_delta = finite ? max_abs_delta(first, second) : 0.0f;
    const float reset_delta = finite ? max_abs_delta(first, after_reset) : INFINITY;
    const bool pass = finite && position_after_two == 2u &&
            axiom_qwen38_attention_layer_position(layer) == 1u && state_delta > 0.0f && reset_delta == 0.0f;
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-qwen38-attention\",\"layer\":%u,\"batch\":8,"
                "\"max_context\":%u,\"device_bytes\":%llu,\"state_delta\":%.9g,"
                "\"reset_delta\":%.9g}\n",
                pass ? "pass" : "fail", layer_index, max_context,
                static_cast<unsigned long long>(axiom_qwen38_attention_layer_device_bytes(layer)),
                static_cast<double>(state_delta), static_cast<double>(reset_delta));
    (void)cudaFree(d_output);
    (void)cudaFree(d_input);
    axiom_qwen38_attention_layer_destroy(layer);
    axiom_runtime_destroy(runtime);
    return pass ? 0 : 1;
}
