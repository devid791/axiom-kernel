/* Full 56-layer NVFP4 MLP residency gate for the Unsloth Qwen3.8 checkpoint. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4_bank.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-resident-probe: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

uint32_t runs_from_env() {
    const char *text = getenv("AXIOM_QWEN38_NVFP4_STACK_RUNS");
    if (!text || !*text) return 20u;
    char *end = nullptr;
    const unsigned long value = strtoul(text, &end, 10);
    return (end != text && *end == '\0' && value > 0u && value <= 10000u)
            ? static_cast<uint32_t>(value) : 0u;
}

void fill_input(std::vector<float> *input) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_NVFP4_TC_BATCH; ++column) {
        for (uint32_t index = 0; index < AXIOM_QWEN38_NVFP4_HIDDEN; ++index) {
            (*input)[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + index] =
                    0.63f * sinf(static_cast<float>(index) * 0.0127f + 0.03f * column) +
                    0.31f * cosf(static_cast<float>(index) * 0.0047f - 0.02f * column);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    /* Full bank load is streamed disk -> one layer host buffer -> device.  It
     * must not retain a second multi-GiB host copy of the checkpoint. */
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
    model_config.memory_budget_bytes = 0;
    axiom_model *model = nullptr;
    rc = axiom_model_open(runtime, &model, &model_config);
    if (rc != AXIOM_OK) {
        axiom_runtime_destroy(runtime);
        return fail("model", rc);
    }
    size_t free_before = 0, total = 0, free_after = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) {
        axiom_model_close(model); axiom_runtime_destroy(runtime); return fail("cudaMemGetInfo");
    }
    axiom_qwen38_nvfp4_bank *bank = nullptr;
    rc = axiom_qwen38_nvfp4_bank_load(model, 0, &bank);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("resident bank", rc);
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess) {
        axiom_qwen38_nvfp4_bank_destroy(bank); return fail("cudaMemGetInfo");
    }
    const size_t bytes = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) *
            AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float);
    const uint32_t runs = runs_from_env();
    if (runs == 0u) {
        axiom_qwen38_nvfp4_bank_destroy(bank);
        return fail("invalid AXIOM_QWEN38_NVFP4_STACK_RUNS");
    }
    std::vector<float> host(bytes / sizeof(float));
    fill_input(&host);
    float *seed = nullptr, *a = nullptr, *b = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    float stack_ms = 0.0f;
    cudaError_t cuda_status = cudaMalloc(&seed, bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&a, bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&b, bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(seed, host.data(), bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        if (b) (void)cudaFree(b);
        if (a) (void)cudaFree(a);
        if (seed) (void)cudaFree(seed);
        axiom_qwen38_nvfp4_bank_destroy(bank);
        return fail("activation allocation");
    }
    auto run_stack = [&](float **out_final) -> int {
        if (cudaMemcpyAsync(a, seed, bytes, cudaMemcpyDeviceToDevice, nullptr) != cudaSuccess) {
            return AXIOM_ERR_CUDA;
        }
        float *src = a;
        float *dst = b;
        for (uint32_t layer = 0; layer < AXIOM_QWEN38_NVFP4_TC_LAYERS; ++layer) {
            const int step = axiom_qwen38_nvfp4_bank_forward_f32_device(bank, layer, src, dst, nullptr);
            if (step != AXIOM_OK) return step;
            float *next = src;
            src = dst;
            dst = next;
        }
        *out_final = src;
        return AXIOM_OK;
    };
    float *final_output = nullptr;
    rc = run_stack(&final_output);  // warm-up each descriptor and tensor-core kernel
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc != AXIOM_OK || cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
        cudaEventRecord(start, nullptr) != cudaSuccess) {
        if (stop) (void)cudaEventDestroy(stop);
        if (start) (void)cudaEventDestroy(start);
        (void)cudaFree(b);
        (void)cudaFree(a);
        (void)cudaFree(seed);
        axiom_qwen38_nvfp4_bank_destroy(bank);
        return fail("warm-up or CUDA timing setup", rc);
    }
    for (uint32_t run = 0; run < runs && rc == AXIOM_OK; ++run) {
        rc = run_stack(&final_output);
    }
    if (rc == AXIOM_OK && cudaEventRecord(stop, nullptr) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventSynchronize(stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaEventElapsedTime(&stack_ms, start, stop) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(host.data(), final_output, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    bool finite = rc == AXIOM_OK;
    for (float value : host) finite = finite && std::isfinite(value);
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-resident-fp4\",\"nvfp4_mlp_layers\":56,"
                "\"batch\":8,\"device_bytes\":%llu,\"cuda_bytes_consumed\":%llu,"
                "\"cuda_total_bytes\":%llu,\"runs\":%u,\"mlp_stack_ms_per_batch8\":%.9g}\n",
                finite ? "pass" : "fail",
                static_cast<unsigned long long>(axiom_qwen38_nvfp4_bank_device_bytes(bank)),
                static_cast<unsigned long long>(free_before - free_after),
                static_cast<unsigned long long>(total), runs,
                static_cast<double>(stack_ms / static_cast<float>(runs)));
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    (void)cudaFree(b);
    (void)cudaFree(a);
    (void)cudaFree(seed);
    axiom_qwen38_nvfp4_bank_destroy(bank);
    return finite ? 0 : fail("resident forward", rc);
}
