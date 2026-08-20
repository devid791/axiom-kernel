/*
 * Component gate for the native Unsloth Qwen3.8 NVFP4 path.
 *
 * It reads a real MLP projection directly from unsloth/Qwen3.8-27B-NVFP4,
 * quantizes a deterministic f32 activation on the RTX 5090 using the
 * checkpoint's input_global_scale, and compares the native device W4A4 result
 * against a host dequantization of those exact packed activation bytes.
 *
 * This is intentionally a component parity gate, not a model benchmark.
 */
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kFfn = 17408u;
constexpr uint32_t kNvfp4Layers = 56u;
constexpr uint32_t kRows = 32u;
constexpr uint32_t kGroup = 16u;
constexpr int kNoStatus = (-2147483647 - 1);

int fail(const char *what, int rc = kNoStatus) {
    if (rc == kNoStatus) {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: %s\n", what);
    } else {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: %s: %s\n", what,
                     axiom_status_string(rc));
    }
    return 1;
}

int cuda_fail(const char *what, cudaError_t err) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: %s: %s\n", what,
                 cudaGetErrorString(err));
    return 1;
}

float e2m1(uint8_t code) {
    static constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = values[code & 7u];
    return (code & 8u) ? -value : value;
}

float e4m3fn(uint8_t code) {
    if ((code & 0x7fu) == 0x7fu) return std::numeric_limits<float>::quiet_NaN();
    const uint32_t exp = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    const float value = exp == 0u
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                         static_cast<int>(exp) - 7);
    return (code & 0x80u) ? -value : value;
}

bool parse_layer(const char *text, uint32_t *out) {
    if (!text || !*text || !out) return false;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed >= kNvfp4Layers) return false;
    *out = static_cast<uint32_t>(parsed);
    return true;
}

bool projection_geometry(const char *projection, uint32_t *rows, uint32_t *cols) {
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

int tensor_read(axiom_model *model, const std::string &name, void *dst, uint64_t bytes) {
    const int rc = axiom_model_tensor_read_slice(model, name.c_str(), 0, dst, bytes);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: read %s: %s\n", name.c_str(),
                     axiom_status_string(rc));
    }
    return rc;
}

void cuda_free(void *ptr) {
    if (ptr) (void)cudaFree(ptr);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [LAYER(0..55)] [gate|up|down]\n", argv[0]);
        return 2;
    }
    uint32_t layer = 0u;
    if (argc >= 3 && !parse_layer(argv[2], &layer)) {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: invalid NVFP4 layer\n");
        return 2;
    }
    const char *projection = argc >= 4 ? argv[3] : "gate";
    uint32_t full_rows = 0u, cols = 0u;
    if (!projection_geometry(projection, &full_rows, &cols)) {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-probe: projection must be gate, up, or down\n");
        return 2;
    }
    const uint32_t rows = full_rows < kRows ? full_rows : kRows;
    const uint32_t groups = cols / kGroup;
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * (cols / 2u);
    const uint64_t weight_scale_bytes = static_cast<uint64_t>(rows) * groups;

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_config);
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
        return fail("model open", rc);
    }

    char prefix[192];
    std::snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u.mlp.%s_proj", layer, projection);
    const std::string base(prefix);
    const std::string packed_name = base + ".weight_packed";
    const std::string scale_name = base + ".weight_scale";
    const std::string input_global_name = base + ".input_global_scale";
    const std::string weight_global_name = base + ".weight_global_scale";

    std::vector<uint8_t> weight(weight_bytes);
    std::vector<uint8_t> weight_scale(weight_scale_bytes);
    float input_global_scale = 0.0f;
    float weight_global_scale = 0.0f;
    rc = tensor_read(model, packed_name, weight.data(), weight.size());
    if (rc == AXIOM_OK) rc = tensor_read(model, scale_name, weight_scale.data(), weight_scale.size());
    if (rc == AXIOM_OK) rc = tensor_read(model, input_global_name, &input_global_scale, sizeof(input_global_scale));
    if (rc == AXIOM_OK) rc = tensor_read(model, weight_global_name, &weight_global_scale, sizeof(weight_global_scale));
    if (rc != AXIOM_OK || !std::isfinite(input_global_scale) || !std::isfinite(weight_global_scale) ||
        input_global_scale <= 0.0f || weight_global_scale <= 0.0f) {
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return fail("invalid Unsloth NVFP4 scale", rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc);
    }
    for (uint8_t scale : weight_scale) {
        if (!std::isfinite(e4m3fn(scale))) {
            axiom_model_close(model);
            axiom_runtime_destroy(runtime);
            return fail("NaN in Unsloth weight_scale", AXIOM_ERR_INVALID_ARGUMENT);
        }
    }

    std::vector<float> input(cols);
    for (uint32_t i = 0; i < cols; ++i) {
        input[i] = 0.73f * std::sin(static_cast<float>(i) * 0.03125f) +
                   0.29f * std::cos(static_cast<float>(i) * 0.0078125f) +
                   static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.015625f;
    }
    std::vector<uint8_t> packed_input(cols / 2u);
    std::vector<uint8_t> input_scale(groups);
    std::vector<float> gpu_out(rows), host_ref(rows);

    uint8_t *d_weight = nullptr;
    uint8_t *d_weight_scale = nullptr;
    float *d_input = nullptr;
    uint8_t *d_packed_input = nullptr;
    uint8_t *d_input_scale = nullptr;
    float *d_out = nullptr;
    cudaError_t cuda_rc = cudaSetDevice(0);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_weight, weight.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_weight_scale, weight_scale.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_input, input.size() * sizeof(float));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_packed_input, packed_input.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_input_scale, input_scale.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_out, gpu_out.size() * sizeof(float));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_weight, weight.data(), weight.size(), cudaMemcpyHostToDevice);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_weight_scale, weight_scale.data(), weight_scale.size(), cudaMemcpyHostToDevice);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_rc != cudaSuccess) {
        cuda_free(d_out); cuda_free(d_input_scale); cuda_free(d_packed_input); cuda_free(d_input);
        cuda_free(d_weight_scale); cuda_free(d_weight);
        axiom_model_close(model); axiom_runtime_destroy(runtime);
        return cuda_fail("device setup", cuda_rc);
    }

    rc = axiom_qwen38_nvfp4_quantize_reference_f32(
            0, d_input, input_global_scale, d_packed_input, d_input_scale, cols, nullptr);
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_nvfp4_w4a4_reference_matvec_f32(
                0, d_weight, d_weight_scale, weight_global_scale,
                d_packed_input, d_input_scale, input_global_scale,
                d_out, rows, cols, nullptr);
    }
    if (rc == AXIOM_OK) cuda_rc = cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (rc == AXIOM_OK && cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(packed_input.data(), d_packed_input, packed_input.size(), cudaMemcpyDeviceToHost);
    if (rc == AXIOM_OK && cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(input_scale.data(), d_input_scale, input_scale.size(), cudaMemcpyDeviceToHost);

    cuda_free(d_out); cuda_free(d_input_scale); cuda_free(d_packed_input); cuda_free(d_input);
    cuda_free(d_weight_scale); cuda_free(d_weight);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) return fail("native W4A4", rc);
    if (cuda_rc != cudaSuccess) return cuda_fail("device download", cuda_rc);

    for (uint8_t scale : input_scale) {
        if (!std::isfinite(e4m3fn(scale))) return fail("native quantizer emitted NaN scale");
    }
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (uint32_t row = 0; row < rows; ++row) {
        float acc = 0.0f;
        for (uint32_t group = 0; group < groups; ++group) {
            const float ws = e4m3fn(weight_scale[static_cast<uint64_t>(row) * groups + group]) /
                             weight_global_scale;
            const float xs = e4m3fn(input_scale[group]) / input_global_scale;
            for (uint32_t pair = 0; pair < kGroup / 2u; ++pair) {
                const uint8_t w = weight[static_cast<uint64_t>(row) * (cols / 2u) + group * 8u + pair];
                const uint8_t x = packed_input[group * 8u + pair];
                acc = std::fma(e2m1(w & 0x0fu) * ws, e2m1(x & 0x0fu) * xs, acc);
                acc = std::fma(e2m1(w >> 4u) * ws, e2m1(x >> 4u) * xs, acc);
            }
        }
        host_ref[row] = acc;
        const float abs_error = std::fabs(gpu_out[row] - host_ref[row]);
        const float rel_error = abs_error / std::fmax(1.0e-6f, std::fabs(host_ref[row]));
        max_abs = std::fmax(max_abs, abs_error);
        max_rel = std::fmax(max_rel, rel_error);
    }
    const bool pass = max_abs <= 2.0e-4f && max_rel <= 2.0e-5f;
    std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                "\"layer\":%u,\"projection\":\"%s\",\"rows\":%u,\"cols\":%u,"
                "\"input_global_scale\":%.9g,\"weight_global_scale\":%.9g,"
                "\"max_abs\":%.9g,\"max_rel\":%.9g}\n",
                pass ? "pass" : "fail", layer, projection, rows, cols,
                static_cast<double>(input_global_scale), static_cast<double>(weight_global_scale),
                static_cast<double>(max_abs), static_cast<double>(max_rel));
    return pass ? 0 : 1;
}
