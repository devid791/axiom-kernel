/* Full 64-layer Qwen3.8 MLP resident gate.  This deliberately invokes each
 * layer independently on the same finite batch: it proves all original MLP
 * encodings are resident and executable without claiming a transformer pass. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_mlp_bank.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-mlp-bank-probe: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_MLP_BANK_RUNS");
    if (!text || !*text) return 10u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' && value > 0u && value <= 10000u
            ? static_cast<uint32_t>(value) : 0u;
}

void fill_input(std::vector<float> *input) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_NVFP4_TC_BATCH; ++column) {
        for (uint32_t index = 0; index < AXIOM_QWEN38_NVFP4_HIDDEN; ++index) {
            (*input)[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + index] =
                    0.58f * std::sin(static_cast<float>(index) * 0.011f + 0.031f * column) +
                    0.26f * std::cos(static_cast<float>(index) * 0.0043f - 0.027f * column);
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
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_MLP_BANK_RUNS");
    if (setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) return fail("set AXIOM_MODEL_NO_RESIDENT");

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
    size_t free_before = 0u;
    size_t free_after = 0u;
    size_t total = 0u;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) {
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return fail("cudaMemGetInfo");
    }
    axiom_qwen38_mlp_bank *bank = nullptr;
    rc = axiom_qwen38_mlp_bank_load(model, 0, &bank);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("MLP bank load", rc);
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) {
        axiom_qwen38_mlp_bank_destroy(bank);
        return fail("cudaMemGetInfo");
    }

    const size_t bytes = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) *
            AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float);
    std::vector<float> input(bytes / sizeof(float));
    std::vector<float> output(bytes / sizeof(float));
    fill_input(&input);
    float *d_input = nullptr;
    float *d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t status = cudaMalloc(&d_input, bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_output, bytes);
    if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (d_output) (void)cudaFree(d_output);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_mlp_bank_destroy(bank);
        return fail("device setup");
    }
    auto run_all = [&]() -> int {
        for (uint32_t layer = 0; layer < AXIOM_QWEN38_MLP_LAYER_COUNT; ++layer) {
            const int step = axiom_qwen38_mlp_bank_forward_f32_device(bank, layer, d_input, d_output, nullptr);
            if (step != AXIOM_OK) return step;
        }
        return AXIOM_OK;
    };
    rc = run_all();
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc != AXIOM_OK || cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
        cudaEventRecord(start, nullptr) != cudaSuccess) {
        if (stop) (void)cudaEventDestroy(stop);
        if (start) (void)cudaEventDestroy(start);
        (void)cudaFree(d_output);
        (void)cudaFree(d_input);
        axiom_qwen38_mlp_bank_destroy(bank);
        return fail("warm-up or timing setup", rc);
    }
    for (uint32_t run = 0; run < runs && rc == AXIOM_OK; ++run) rc = run_all();
    float ms = 0.0f;
    if (rc == AXIOM_OK && cudaEventRecord(stop, nullptr) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventSynchronize(stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(output.data(), d_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    bool finite = rc == AXIOM_OK;
    for (float value : output) finite = finite && std::isfinite(value);
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-full-mlp-bank\",\"mlp_layers\":64,\"batch\":8,"
                "\"runs\":%u,\"device_bytes\":%llu,\"cuda_bytes_consumed\":%llu,"
                "\"cuda_total_bytes\":%llu,\"mlp_bank_ms_per_batch8\":%.9g}\n",
                finite ? "pass" : "fail", runs,
                static_cast<unsigned long long>(axiom_qwen38_mlp_bank_device_bytes(bank)),
                static_cast<unsigned long long>(free_before - free_after),
                static_cast<unsigned long long>(total),
                static_cast<double>(ms / static_cast<float>(runs)));
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    (void)cudaFree(d_output);
    (void)cudaFree(d_input);
    axiom_qwen38_mlp_bank_destroy(bank);
    return finite ? 0 : 1;
}
