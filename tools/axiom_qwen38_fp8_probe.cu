/* FP8 tensor-core parity and timing gate for actual Unsloth Qwen3.8 tensors. */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kFfn = 17408u;

int fail(const char *what, int rc = AXIOM_OK) {
    std::fprintf(stderr, "axiom-qwen38-fp8-probe: %s%s%s\n", what,
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

bool geometry(const char *projection, uint32_t *rows, uint32_t *cols) {
    if (!projection || !rows || !cols) return false;
    if (std::strcmp(projection, "gate") == 0 || std::strcmp(projection, "up") == 0) {
        *rows = kFfn;
        *cols = kHidden;
        return true;
    }
    if (std::strcmp(projection, "down") == 0) {
        *rows = kHidden;
        *cols = kFfn;
        return true;
    }
    return false;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_FP8_PROBE_RUNS");
    if (!text || !*text) return 100u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return (end != text && *end == '\0' && value > 0u && value <= 10000u)
            ? static_cast<uint32_t>(value) : 0u;
}

void fill_input(std::vector<float> *input, uint32_t cols) {
    for (uint32_t column = 0; column < AXIOM_QWEN38_FP8_BATCH; ++column) {
        for (uint32_t index = 0; index < cols; ++index) {
            (*input)[static_cast<size_t>(column) * cols + index] =
                    0.59f * std::sin(static_cast<float>(index) * 0.014f + 0.037f * column) +
                    0.37f * std::cos(static_cast<float>(index) * 0.005f - 0.021f * column) +
                    static_cast<float>(static_cast<int>(index % 11u) - 5) * 0.011f;
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [LAYER(56..63)] [gate|up|down]\n", argv[0]);
        return 2;
    }
    uint32_t layer = 56u;
    if (argc >= 3 && !parse_layer(argv[2], &layer)) return fail("invalid FP8 layer");
    const char *projection = argc >= 4 ? argv[3] : "gate";
    uint32_t rows = 0u, cols = 0u;
    if (!geometry(projection, &rows, &cols)) return fail("projection must be gate, up, or down");
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_FP8_PROBE_RUNS");

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
    char prefix[192];
    std::snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u.mlp.%s_proj", layer, projection);
    const std::string base(prefix);
    axiom_qwen38_fp8_linear *linear = nullptr;
    rc = axiom_qwen38_fp8_linear_load(model, 0, (base + ".weight").c_str(),
                                       (base + ".weight_scale").c_str(), &linear);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("FP8 resident load", rc);
    if (axiom_qwen38_fp8_linear_rows(linear) != rows || axiom_qwen38_fp8_linear_cols(linear) != cols) {
        axiom_qwen38_fp8_linear_destroy(linear);
        return fail("checkpoint geometry mismatch");
    }

    const size_t input_bytes = static_cast<size_t>(cols) * AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(rows) * AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    std::vector<float> input(input_bytes / sizeof(float));
    std::vector<float> reference(output_bytes / sizeof(float));
    std::vector<float> got(output_bytes / sizeof(float));
    fill_input(&input, cols);
    float *d_input = nullptr, *d_reference = nullptr, *d_got = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaError_t status = cudaMalloc(&d_input, input_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_reference, output_bytes);
    if (status == cudaSuccess) status = cudaMalloc(&d_got, output_bytes);
    if (status == cudaSuccess) status = cudaMemcpy(d_input, input.data(), input_bytes, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (d_got) (void)cudaFree(d_got);
        if (d_reference) (void)cudaFree(d_reference);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_fp8_linear_destroy(linear);
        return fail("device setup");
    }
    rc = axiom_qwen38_fp8_linear_forward_reference_f32_device(linear, d_input, d_reference, nullptr);
    if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(linear, d_input, d_got, nullptr);
    if (rc != AXIOM_OK || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(reference.data(), d_reference, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(got.data(), d_got, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        if (d_got) (void)cudaFree(d_got);
        if (d_reference) (void)cudaFree(d_reference);
        if (d_input) (void)cudaFree(d_input);
        axiom_qwen38_fp8_linear_destroy(linear);
        return fail("FP8 parity launch", rc);
    }
    float max_abs = 0.0f;
    bool pass = true;
    for (size_t index = 0; index < got.size(); ++index) {
        const float abs_error = std::fabs(got[index] - reference[index]);
        max_abs = std::fmax(max_abs, abs_error);
        if (abs_error > 2.0e-3f + 2.0e-3f * std::fabs(reference[index])) pass = false;
    }
    float ms = 0.0f;
    if (pass && cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess &&
        cudaEventRecord(start, nullptr) == cudaSuccess) {
        for (uint32_t run = 0; run < runs && rc == AXIOM_OK; ++run) {
            rc = axiom_qwen38_fp8_linear_forward_f32_device(linear, d_input, d_got, nullptr);
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
                "\"backend\":\"axiom-native-fp8\",\"layer\":%u,\"projection\":\"%s\","
                "\"rows\":%u,\"cols\":%u,\"batch\":8,\"tensor_core\":%s,"
                "\"device_bytes\":%llu,\"runs\":%u,\"ms_per_batch8\":%.9g,\"max_abs\":%.9g}\n",
                pass ? "pass" : "fail", layer, projection, rows, cols,
                axiom_qwen38_fp8_linear_tensor_core_enabled(linear) ? "true" : "false",
                static_cast<unsigned long long>(axiom_qwen38_fp8_linear_device_bytes(linear)),
                runs, static_cast<double>(ms), static_cast<double>(max_abs));
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    (void)cudaFree(d_got);
    (void)cudaFree(d_reference);
    (void)cudaFree(d_input);
    axiom_qwen38_fp8_linear_destroy(linear);
    return pass ? 0 : 1;
}
