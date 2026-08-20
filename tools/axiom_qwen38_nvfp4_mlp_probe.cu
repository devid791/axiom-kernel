/* Resident Axiom MLP gate for the original Unsloth Qwen3.8 NVFP4 checkpoint. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

int fail(const char *what) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-mlp-probe: %s\n", what);
    return 1;
}

bool parse_layer(const char *text, uint32_t *out) {
    if (!text || !*text || !out) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value >= AXIOM_QWEN38_NVFP4_TC_LAYERS) return false;
    *out = static_cast<uint32_t>(value);
    return true;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_NVFP4_MLP_RUNS");
    if (!text || !*text) return 100u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return (end != text && *end == '\0' && value > 0u && value <= 10000u)
            ? static_cast<uint32_t>(value) : 0u;
}

void fill_input(std::vector<float> *input, bool same_column) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_NVFP4_TC_BATCH; ++column) {
        const float column_bias = same_column ? 0.0f : static_cast<float>(column) * 0.03125f;
        for (uint32_t index = 0; index < AXIOM_QWEN38_NVFP4_HIDDEN; ++index) {
            (*input)[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + index] =
                    0.71f * std::sin(static_cast<float>(index) * 0.017f + column_bias) +
                    0.23f * std::cos(static_cast<float>(index) * 0.0061f - column_bias) +
                    static_cast<float>(static_cast<int>(index % 17u) - 8) * 0.0078125f;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [LAYER(0..55)]\n", argv[0]);
        return 2;
    }
    uint32_t layer = 0u;
    if (argc == 3 && !parse_layer(argv[2], &layer)) return fail("invalid NVFP4 layer");
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_NVFP4_MLP_RUNS");

    axiom_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.backend = AXIOM_BACKEND_CUDA;
    config.device = 0;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &config);
    if (rc != AXIOM_OK) return fail(axiom_status_string(rc));

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
        return fail(axiom_status_string(rc));
    }
    axiom_qwen38_nvfp4_mlp *mlp = nullptr;
    rc = axiom_qwen38_nvfp4_mlp_load(model, 0, layer, &mlp);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail(axiom_status_string(rc));

    const size_t input_bytes = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) *
            AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float);
    std::vector<float> input(input_bytes / sizeof(float));
    std::vector<float> got(input.size());
    std::vector<float> one(input.size());
    std::vector<float> ref(input.size());
    float *d_input = nullptr;
    float *d_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    int result = 1;
    fill_input(&input, false);
    cudaError_t cuda_status = cudaMalloc(&d_input, input_bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_out, input_bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        (void)cudaGetLastError();
        goto done;
    }
    rc = axiom_qwen38_nvfp4_mlp_forward_f32_device(mlp, d_input, d_out, nullptr);
    if (rc != AXIOM_OK || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(got.data(), d_out, input_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        goto done;
    }
    for (float value : got) if (!std::isfinite(value)) goto done;

    /* Each column is independent.  Re-running its input in all eight lanes
     * detects a transposed/tiled scale plane or cross-column alias without
     * relying on a second model implementation. */
    {
        float max_abs = 0.0f;
        bool pass = true;
        for (uint32_t column = 0; column < AXIOM_QWEN38_NVFP4_TC_BATCH; ++column) {
            for (uint32_t lane = 0; lane < AXIOM_QWEN38_NVFP4_TC_BATCH; ++lane) {
                std::memcpy(one.data() + static_cast<size_t>(lane) * AXIOM_QWEN38_NVFP4_HIDDEN,
                            input.data() + static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN,
                            AXIOM_QWEN38_NVFP4_HIDDEN * sizeof(float));
            }
            if (cudaMemcpy(d_input, one.data(), input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
                axiom_qwen38_nvfp4_mlp_forward_f32_device(mlp, d_input, d_out, nullptr) != AXIOM_OK ||
                cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(ref.data(), d_out, input_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                goto done;
            }
            for (uint32_t row = 0; row < AXIOM_QWEN38_NVFP4_HIDDEN; ++row) {
                const float expected = ref[row];
                const float actual = got[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + row];
                const float abs_error = std::fabs(actual - expected);
                max_abs = std::fmax(max_abs, abs_error);
                if (abs_error > 5.0e-5f + 5.0e-3f * std::fabs(expected)) pass = false;
            }
        }
        if (!pass) {
            std::fprintf(stderr, "axiom-qwen38-nvfp4-mlp-probe: cross-column numerical mismatch\n");
            goto done;
        }
        if (cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess ||
            cudaEventRecord(start, nullptr) != cudaSuccess) {
            goto done;
        }
        for (uint32_t run = 0; run < runs; ++run) {
            if (axiom_qwen38_nvfp4_mlp_forward_f32_device(mlp, d_input, d_out, nullptr) != AXIOM_OK) goto done;
        }
        if (cudaEventRecord(stop, nullptr) != cudaSuccess || cudaEventSynchronize(stop) != cudaSuccess) goto done;
        float elapsed_ms = 0.0f;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) goto done;
        std::printf("{\"status\":\"pass\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                    "\"backend\":\"axiom-native-resident-fp4\",\"layer\":%u,"
                    "\"batch\":%u,\"runs\":%u,\"device_bytes\":%llu,"
                    "\"mlp_ms_per_batch8\":%.9g,\"max_abs_cross_column\":%.9g}\n",
                    layer, AXIOM_QWEN38_NVFP4_TC_BATCH, runs,
                    static_cast<unsigned long long>(axiom_qwen38_nvfp4_mlp_device_bytes(mlp)),
                    static_cast<double>(elapsed_ms / static_cast<float>(runs)), static_cast<double>(max_abs));
        result = 0;
    }

done:
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    if (d_out) (void)cudaFree(d_out);
    if (d_input) (void)cudaFree(d_input);
    axiom_qwen38_nvfp4_mlp_destroy(mlp);
    return result;
}
