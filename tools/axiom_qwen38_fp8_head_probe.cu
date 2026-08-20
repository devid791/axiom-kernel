/* Native FP8 LM-head residency/throughput gate for Unsloth Qwen3.8. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kVocab = 248320u;

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-fp8-head-probe: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_FP8_HEAD_RUNS");
    if (!text || !*text) return 20u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' && value > 0u && value <= 10000u
            ? static_cast<uint32_t>(value) : 0u;
}

void fill_input(std::vector<float> *input) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        for (uint32_t index = 0; index < kHidden; ++index) {
            (*input)[static_cast<size_t>(column) * kHidden + index] =
                    0.53f * std::sin(static_cast<float>(index) * 0.013f + 0.047f * column) +
                    0.29f * std::cos(static_cast<float>(index) * 0.0061f - 0.019f * column);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_FP8_HEAD_RUNS");

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
    model_config.memory_budget_bytes = 0u;
    axiom_model *model = nullptr;
    rc = axiom_model_open(runtime, &model, &model_config);
    if (rc != AXIOM_OK) {
        axiom_runtime_destroy(runtime);
        return fail("model", rc);
    }
    axiom_qwen38_fp8_linear *head = nullptr;
    rc = axiom_qwen38_fp8_linear_load(
            model, 0, "lm_head.weight", "lm_head.weight_scale", &head);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("LM head load", rc);
    if (axiom_qwen38_fp8_linear_rows(head) != kVocab ||
        axiom_qwen38_fp8_linear_cols(head) != kHidden ||
        !axiom_qwen38_fp8_linear_tensor_core_enabled(head)) {
        axiom_qwen38_fp8_linear_destroy(head);
        return fail("LM head geometry or tensor-core heuristic");
    }

    const size_t input_bytes = static_cast<size_t>(kHidden) * AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(kVocab) * AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    std::vector<float> input(input_bytes / sizeof(float));
    std::vector<float> output(output_bytes / sizeof(float));
    fill_input(&input);
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t status = cudaMalloc(&d_input, input_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_output, output_bytes);
    if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (d_output) (void)cudaFree(d_output);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_fp8_linear_destroy(head);
        return fail("device setup");
    }
    rc = axiom_qwen38_fp8_linear_forward_f32_device(head, d_input, d_output, nullptr);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc != AXIOM_OK || cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
        cudaEventRecord(start, nullptr) != cudaSuccess) {
        if (stop) (void)cudaEventDestroy(stop);
        if (start) (void)cudaEventDestroy(start);
        (void)cudaFree(d_output);
        (void)cudaFree(d_input);
        axiom_qwen38_fp8_linear_destroy(head);
        return fail("warm-up or timing setup", rc);
    }
    for (uint32_t run = 0; run < runs && rc == AXIOM_OK; ++run) {
        rc = axiom_qwen38_fp8_linear_forward_f32_device(head, d_input, d_output, nullptr);
    }
    float ms = 0.0f;
    if (rc == AXIOM_OK && cudaEventRecord(stop, nullptr) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventSynchronize(stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    bool finite = rc == AXIOM_OK;
    uint32_t first_argmax = 0u;
    float first_max = -INFINITY;
    for (uint32_t column = 0; finite && column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        for (uint32_t row = 0; row < kVocab; ++row) {
            const float value = output[static_cast<size_t>(column) * kVocab + row];
            finite = finite && std::isfinite(value);
            if (column == 0u && value > first_max) {
                first_max = value;
                first_argmax = row;
            }
        }
    }
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-fp8-lm-head\",\"rows\":%u,\"cols\":%u,"
                "\"batch\":8,\"tensor_core\":true,\"runs\":%u,"
                "\"device_bytes\":%llu,\"ms_per_batch8\":%.9g,"
                "\"first_column_argmax\":%u,\"first_column_max\":%.9g}\n",
                finite ? "pass" : "fail", kVocab, kHidden, runs,
                static_cast<unsigned long long>(axiom_qwen38_fp8_linear_device_bytes(head)),
                static_cast<double>(ms / static_cast<float>(runs)), first_argmax,
                static_cast<double>(first_max));
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    (void)cudaFree(d_output);
    (void)cudaFree(d_input);
    axiom_qwen38_fp8_linear_destroy(head);
    return finite ? 0 : 1;
}
