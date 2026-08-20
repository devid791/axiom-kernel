/* Resident FP8 MLP composition gate for the eight final Qwen3.8 layers. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"
#include "axiom/qwen38_fp8_mlp.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-fp8-mlp-probe: %s%s%s\n", what,
                 rc == AXIOM_OK ? "" : ": ", rc == AXIOM_OK ? "" : axiom_status_string(rc));
    return 1;
}

bool parse_layer(const char *text, uint32_t *out) {
    if (!text || !*text || !out) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value < 56u || value > 63u) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_FP8_MLP_RUNS");
    if (!text || !*text) return 100u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return (end != text && *end == '\0' && value > 0u && value <= 10000u)
            ? static_cast<uint32_t>(value) : 0u;
}

void fill(std::vector<float> *out, bool identical_columns) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        const float shift = identical_columns ? 0.0f : static_cast<float>(column) * 0.03125f;
        for (uint32_t index = 0; index < AXIOM_QWEN38_NVFP4_HIDDEN; ++index) {
            (*out)[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + index] =
                    0.67f * sinf(static_cast<float>(index) * 0.019f + shift) +
                    0.23f * cosf(static_cast<float>(index) * 0.006f - shift);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [LAYER(56..63)]\n", argv[0]);
        return 2;
    }
    uint32_t layer = 56u;
    if (argc == 3 && !parse_layer(argv[2], &layer)) return fail("invalid FP8 MLP layer");
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_FP8_MLP_RUNS");
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
    axiom_qwen38_fp8_mlp *mlp = nullptr;
    rc = axiom_qwen38_fp8_mlp_load(model, 0, layer, &mlp);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("resident load", rc);

    const size_t bytes = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) *
            AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    std::vector<float> input(bytes / sizeof(float));
    std::vector<float> got(bytes / sizeof(float));
    std::vector<float> repeat(bytes / sizeof(float));
    std::vector<float> check(bytes / sizeof(float));
    fill(&input, false);
    float *d_input = nullptr, *d_out = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaError_t status = cudaMalloc(&d_input, bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_out, bytes);
    if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (d_out) (void)cudaFree(d_out);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_fp8_mlp_destroy(mlp);
        return fail("device setup");
    }
    rc = axiom_qwen38_fp8_mlp_forward_f32_device(mlp, d_input, d_out, nullptr);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() == cudaSuccess &&
        cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) {
        for (float value : got) if (!std::isfinite(value)) rc = AXIOM_ERR_RUNTIME;
    } else if (rc == AXIOM_OK) {
        rc = AXIOM_ERR_CUDA;
    }
    float max_abs = 0.0f;
    bool pass = rc == AXIOM_OK;
    for (uint32_t column = 0; pass && column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        for (uint32_t lane = 0; lane < AXIOM_QWEN38_FP8_BATCH; ++lane) {
            std::memcpy(repeat.data() + static_cast<size_t>(lane) * AXIOM_QWEN38_NVFP4_HIDDEN,
                        input.data() + static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN,
                        AXIOM_QWEN38_NVFP4_HIDDEN * sizeof(float));
        }
        if (cudaMemcpy(d_input, repeat.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            axiom_qwen38_fp8_mlp_forward_f32_device(mlp, d_input, d_out, nullptr) != AXIOM_OK ||
            cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(check.data(), d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            pass = false;
            break;
        }
        for (uint32_t row = 0; row < AXIOM_QWEN38_NVFP4_HIDDEN; ++row) {
            const float actual = got[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + row];
            const float expected = check[row];
            const float error = fabsf(actual - expected);
            max_abs = fmaxf(max_abs, error);
            if (error > 1.0e-4f + 5.0e-3f * fabsf(expected)) pass = false;
        }
    }
    float ms = 0.0f;
    if (pass && cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess &&
        cudaEventRecord(start, nullptr) == cudaSuccess) {
        for (uint32_t run = 0; run < runs && rc == AXIOM_OK; ++run) {
            rc = axiom_qwen38_fp8_mlp_forward_f32_device(mlp, d_input, d_out, nullptr);
        }
        if (rc == AXIOM_OK && cudaEventRecord(stop, nullptr) == cudaSuccess &&
            cudaEventSynchronize(stop) == cudaSuccess && cudaEventElapsedTime(&ms, start, stop) == cudaSuccess) {
            ms /= static_cast<float>(runs);
        } else {
            pass = false;
        }
    } else {
        pass = false;
    }
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"backend\":\"axiom-native-resident-fp8\",\"layer\":%u,\"batch\":8,"
                "\"runs\":%u,\"device_bytes\":%llu,\"mlp_ms_per_batch8\":%.9g,"
                "\"max_abs_cross_column\":%.9g}\n",
                pass ? "pass" : "fail", layer, runs,
                static_cast<unsigned long long>(axiom_qwen38_fp8_mlp_device_bytes(mlp)),
                static_cast<double>(ms), static_cast<double>(max_abs));
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    (void)cudaFree(d_out);
    (void)cudaFree(d_input);
    axiom_qwen38_fp8_mlp_destroy(mlp);
    return pass ? 0 : 1;
}
