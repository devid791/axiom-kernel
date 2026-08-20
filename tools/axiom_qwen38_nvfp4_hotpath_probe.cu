/* Real-device parity/timing gate for the Qwen3.8 ModelOpt NVFP4 hot path.
 * Build manually against libaxiom; no production target depends on this file. */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

constexpr uint32_t kVocab = 248320u;

__global__ void reference_silu_mul(
        const float *gate,
        const float *up,
        float *out,
        uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = gate[index];
    out[index] = (value / (1.0f + expf(-value))) * up[index];
}

int fail(const char *what, int status = AXIOM_ERR_RUNTIME) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-hotpath-probe: %s (status=%d)\n",
                 what, status);
    return 1;
}

uint32_t runs_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_NVFP4_HOTPATH_RUNS");
    if (!text || !*text) return 100u;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    return end != text && *end == '\0' && value > 0u && value <= 10000u
            ? static_cast<uint32_t>(value) : 0u;
}

bool parse_layer(const char *text, uint32_t *out) {
    if (!text || !*text || !out) return false;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' ||
        value >= AXIOM_QWEN38_NVFP4_MODEL_LAYERS) {
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

void fill_input(std::vector<float> *input) {
    for (uint32_t column = 0u;
         column < AXIOM_QWEN38_NVFP4_TC_BATCH; ++column) {
        const float bias = static_cast<float>(column) * 0.03125f;
        for (uint32_t row = 0u; row < AXIOM_QWEN38_NVFP4_HIDDEN; ++row) {
            (*input)[static_cast<size_t>(column) * AXIOM_QWEN38_NVFP4_HIDDEN + row] =
                    0.71f * std::sin(static_cast<float>(row) * 0.017f + bias) +
                    0.23f * std::cos(static_cast<float>(row) * 0.0061f - bias) +
                    static_cast<float>(static_cast<int>(row % 17u) - 8) * 0.0078125f;
        }
    }
}

template <typename Enqueue>
int time_calls(uint32_t runs, Enqueue enqueue, float *out_ms) {
    if (!out_ms || runs == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t warmup = 0u; warmup < 5u; ++warmup) {
        const int rc = enqueue();
        if (rc != AXIOM_OK) return rc;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaError_t cuda_status = cudaEventCreate(&start);
    if (cuda_status == cudaSuccess) cuda_status = cudaEventCreate(&stop);
    if (cuda_status == cudaSuccess) cuda_status = cudaEventRecord(start, nullptr);
    int rc = AXIOM_OK;
    for (uint32_t run = 0u; run < runs && rc == AXIOM_OK; ++run) rc = enqueue();
    if (cuda_status == cudaSuccess && rc == AXIOM_OK) {
        cuda_status = cudaEventRecord(stop, nullptr);
    }
    if (cuda_status == cudaSuccess && rc == AXIOM_OK) {
        cuda_status = cudaEventSynchronize(stop);
    }
    float elapsed = 0.0f;
    if (cuda_status == cudaSuccess && rc == AXIOM_OK) {
        cuda_status = cudaEventElapsedTime(&elapsed, start, stop);
    }
    if (stop) (void)cudaEventDestroy(stop);
    if (start) (void)cudaEventDestroy(start);
    if (rc != AXIOM_OK) return rc;
    if (cuda_status != cudaSuccess) return AXIOM_ERR_CUDA;
    *out_ms = elapsed / static_cast<float>(runs);
    return AXIOM_OK;
}

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    uint64_t outside_tolerance = 0u;
};

ErrorStats compare(
        const std::vector<float> &actual,
        const std::vector<float> &reference,
        size_t count,
        float absolute_tolerance,
        float relative_tolerance) {
    ErrorStats stats{};
    for (size_t index = 0u; index < count; ++index) {
        const float expected = reference[index];
        const float got = actual[index];
        const float absolute = std::fabs(got - expected);
        const float relative = absolute / std::max(std::fabs(expected), 1.0e-6f);
        stats.max_abs = std::max(stats.max_abs, absolute);
        stats.max_rel = std::max(stats.max_rel, relative);
        if (!std::isfinite(got) ||
            absolute > absolute_tolerance + relative_tolerance * std::fabs(expected)) {
            ++stats.outside_tolerance;
        }
    }
    return stats;
}

uint32_t top1(const float *values, uint32_t count) {
    uint32_t best = 0u;
    float best_value = values[0];
    for (uint32_t index = 1u; index < count; ++index) {
        if (values[index] > best_value) {
            best = index;
            best_value = values[index];
        }
    }
    return best;
}

std::string layer_base(uint32_t layer, const char *projection) {
    char value[192]{};
    const int written = std::snprintf(
            value, sizeof(value),
            "model.language_model.layers.%u.mlp.%s_proj", layer, projection);
    return written > 0 && static_cast<size_t>(written) < sizeof(value)
            ? std::string(value) : std::string();
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s MODEL_DIR [LAYER]\n", argv[0]);
        return 2;
    }
    uint32_t layer = 0u;
    if (argc == 3 && !parse_layer(argv[2], &layer)) return fail("invalid layer");
    const uint32_t runs = runs_from_env();
    if (runs == 0u) return fail("invalid AXIOM_QWEN38_NVFP4_HOTPATH_RUNS");

    axiom_runtime *runtime = nullptr;
    axiom_model *model = nullptr;
    axiom_qwen38_nvfp4_linear *head = nullptr;
    axiom_qwen38_nvfp4_linear *gate = nullptr;
    axiom_qwen38_nvfp4_linear *up = nullptr;
    axiom_qwen38_nvfp4_linear *down = nullptr;
    axiom_qwen38_nvfp4_mlp *mlp = nullptr;
    float *d_input = nullptr;
    float *d_head_dynamic = nullptr;
    float *d_head_padded = nullptr;
    float *d_gate = nullptr;
    float *d_up = nullptr;
    float *d_mid = nullptr;
    float *d_mlp_fused = nullptr;
    float *d_mlp_reference = nullptr;
    const size_t input_count = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) *
            AXIOM_QWEN38_NVFP4_TC_BATCH;
    const size_t head_count = static_cast<size_t>(kVocab) *
            AXIOM_QWEN38_NVFP4_TC_BATCH;
    const size_t ffn_count = static_cast<size_t>(AXIOM_QWEN38_NVFP4_FFN) *
            AXIOM_QWEN38_NVFP4_TC_BATCH;
    std::vector<float> input(input_count);
    std::vector<float> head_dynamic(head_count);
    std::vector<float> head_padded(head_count);
    std::vector<float> mlp_fused(input_count);
    std::vector<float> mlp_reference(input_count);
    cudaError_t cuda_status = cudaSuccess;
    int result = 1;

    axiom_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.backend = AXIOM_BACKEND_CUDA;
    config.device = 0u;
    int rc = axiom_runtime_create(&runtime, &config);
    if (rc != AXIOM_OK) return fail("runtime create", rc);
    axiom_model_config model_config{};
    model_config.abi_version = AXIOM_ABI_VERSION;
    model_config.path = argv[1];
    model_config.name = "qwen3.8-27b-modelopt-nvfp4";
    model_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_config.placement.abi_version = AXIOM_ABI_VERSION;
    model_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    rc = axiom_model_open(runtime, &model, &model_config);
    if (rc != AXIOM_OK) {
        (void)fail("model open", rc);
        goto done;
    }
    rc = axiom_qwen38_nvfp4_linear_load(
            model, 0, "lm_head", kVocab, AXIOM_QWEN38_NVFP4_HIDDEN, &head);
    if (rc == AXIOM_OK) {
        const std::string base = layer_base(layer, "gate");
        rc = base.empty() ? AXIOM_ERR_INVALID_ARGUMENT
                          : axiom_qwen38_nvfp4_linear_load(
                                    model, 0, base.c_str(), AXIOM_QWEN38_NVFP4_FFN,
                                    AXIOM_QWEN38_NVFP4_HIDDEN, &gate);
    }
    if (rc == AXIOM_OK) {
        const std::string base = layer_base(layer, "up");
        rc = base.empty() ? AXIOM_ERR_INVALID_ARGUMENT
                          : axiom_qwen38_nvfp4_linear_load(
                                    model, 0, base.c_str(), AXIOM_QWEN38_NVFP4_FFN,
                                    AXIOM_QWEN38_NVFP4_HIDDEN, &up);
    }
    if (rc == AXIOM_OK) {
        const std::string base = layer_base(layer, "down");
        rc = base.empty() ? AXIOM_ERR_INVALID_ARGUMENT
                          : axiom_qwen38_nvfp4_linear_load(
                                    model, 0, base.c_str(), AXIOM_QWEN38_NVFP4_HIDDEN,
                                    AXIOM_QWEN38_NVFP4_FFN, &down);
    }
    if (rc == AXIOM_OK) rc = axiom_qwen38_nvfp4_mlp_load(model, 0, layer, &mlp);
    axiom_model_close(model);
    model = nullptr;
    axiom_runtime_destroy(runtime);
    runtime = nullptr;
    if (rc != AXIOM_OK) {
        (void)fail("resident load", rc);
        goto done;
    }
    if (axiom_qwen38_nvfp4_mlp_uses_fused_gate_up(mlp) != 1u) {
        (void)fail("gate/up checkpoint globals do not permit exact fusion");
        goto done;
    }

    fill_input(&input);
    cuda_status = cudaMalloc(&d_input, input_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_head_dynamic, head_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_head_padded, head_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_gate, ffn_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_up, ffn_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_mid, ffn_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_mlp_fused, input_count * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&d_mlp_reference, input_count * sizeof(float));
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(
                d_input, input.data(), input_count * sizeof(float),
                cudaMemcpyHostToDevice);
    }
    if (cuda_status != cudaSuccess) {
        (void)fail("device allocation/upload", AXIOM_ERR_CUDA);
        goto done;
    }

    std::printf("{\"status\":\"pass\",\"layer\":%u,\"runs\":%u,"
                "\"fused_gate_up\":1,\"head\":[", layer, runs);
    {
        const uint32_t widths[] = {1u, 7u, 8u};
        for (size_t width_index = 0u; width_index < 3u; ++width_index) {
            const uint32_t columns = widths[width_index];
            rc = axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                    head, d_input, d_head_dynamic, columns, nullptr);
            if (rc == AXIOM_OK) {
                rc = axiom_qwen38_nvfp4_linear_forward_f32_device(
                        head, d_input, d_head_padded, nullptr);
            }
            if (rc != AXIOM_OK || cudaDeviceSynchronize() != cudaSuccess) {
                (void)fail("head parity enqueue", rc);
                goto done;
            }
            const size_t compared = static_cast<size_t>(kVocab) * columns;
            if (cudaMemcpy(head_dynamic.data(), d_head_dynamic,
                           compared * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(head_padded.data(), d_head_padded,
                           compared * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void)fail("head parity download", AXIOM_ERR_CUDA);
                goto done;
            }
            const ErrorStats stats = compare(
                    head_dynamic, head_padded, compared, 5.0e-5f, 5.0e-4f);
            uint32_t top1_mismatches = 0u;
            for (uint32_t column = 0u; column < columns; ++column) {
                const size_t offset = static_cast<size_t>(column) * kVocab;
                top1_mismatches += top1(head_dynamic.data() + offset, kVocab) !=
                        top1(head_padded.data() + offset, kVocab) ? 1u : 0u;
            }
            if (stats.outside_tolerance != 0u || top1_mismatches != 0u) {
                (void)fail("head dynamic-M parity");
                goto done;
            }
            float dynamic_ms = 0.0f;
            float padded_ms = 0.0f;
            rc = time_calls(runs, [&]() {
                return axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                        head, d_input, d_head_dynamic, columns, nullptr);
            }, &dynamic_ms);
            if (rc == AXIOM_OK) {
                rc = time_calls(runs, [&]() {
                    return axiom_qwen38_nvfp4_linear_forward_f32_device(
                            head, d_input, d_head_padded, nullptr);
                }, &padded_ms);
            }
            if (rc != AXIOM_OK) {
                (void)fail("head timing", rc);
                goto done;
            }
            std::printf("%s{\"m\":%u,\"dynamic_ms\":%.9g,\"padded_m8_ms\":%.9g,"
                        "\"speedup\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g,"
                        "\"top1_mismatches\":%u}",
                        width_index == 0u ? "" : ",", columns,
                        static_cast<double>(dynamic_ms), static_cast<double>(padded_ms),
                        static_cast<double>(padded_ms / dynamic_ms),
                        static_cast<double>(stats.max_abs), static_cast<double>(stats.max_rel),
                        top1_mismatches);
        }
    }
    std::printf("],\"mlp\":[");
    {
        const uint32_t widths[] = {1u, 7u, 8u};
        for (size_t width_index = 0u; width_index < 3u; ++width_index) {
            const uint32_t columns = widths[width_index];
            const auto enqueue_reference = [&]() -> int {
                int local_rc = axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                        gate, d_input, d_gate, columns, nullptr);
                if (local_rc == AXIOM_OK) {
                    local_rc = axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                            up, d_input, d_up, columns, nullptr);
                }
                if (local_rc != AXIOM_OK) return local_rc;
                constexpr uint32_t threads = 256u;
                const uint32_t count = AXIOM_QWEN38_NVFP4_FFN * columns;
                reference_silu_mul<<<(count + threads - 1u) / threads, threads>>>(
                        d_gate, d_up, d_mid, count);
                if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
                return axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                        down, d_mid, d_mlp_reference, columns, nullptr);
            };
            rc = axiom_qwen38_nvfp4_mlp_forward_f32_device_m(
                    mlp, d_input, d_mlp_fused, columns, nullptr);
            if (rc == AXIOM_OK) rc = enqueue_reference();
            if (rc != AXIOM_OK || cudaDeviceSynchronize() != cudaSuccess) {
                (void)fail("MLP parity enqueue", rc);
                goto done;
            }
            const size_t compared = static_cast<size_t>(AXIOM_QWEN38_NVFP4_HIDDEN) * columns;
            if (cudaMemcpy(mlp_fused.data(), d_mlp_fused,
                           compared * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(mlp_reference.data(), d_mlp_reference,
                           compared * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                (void)fail("MLP parity download", AXIOM_ERR_CUDA);
                goto done;
            }
            const ErrorStats stats = compare(
                    mlp_fused, mlp_reference, compared, 5.0e-5f, 5.0e-3f);
            if (stats.outside_tolerance != 0u) {
                (void)fail("fused MLP parity");
                goto done;
            }
            float fused_ms = 0.0f;
            float separate_ms = 0.0f;
            rc = time_calls(runs, [&]() {
                return axiom_qwen38_nvfp4_mlp_forward_f32_device_m(
                        mlp, d_input, d_mlp_fused, columns, nullptr);
            }, &fused_ms);
            if (rc == AXIOM_OK) rc = time_calls(runs, enqueue_reference, &separate_ms);
            if (rc != AXIOM_OK) {
                (void)fail("MLP timing", rc);
                goto done;
            }
            std::printf("%s{\"m\":%u,\"fused_ms\":%.9g,\"separate_ms\":%.9g,"
                        "\"speedup\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g}",
                        width_index == 0u ? "" : ",", columns,
                        static_cast<double>(fused_ms), static_cast<double>(separate_ms),
                        static_cast<double>(separate_ms / fused_ms),
                        static_cast<double>(stats.max_abs), static_cast<double>(stats.max_rel));
        }
    }
    std::printf("]}\n");
    result = 0;

done:
    if (d_mlp_reference) (void)cudaFree(d_mlp_reference);
    if (d_mlp_fused) (void)cudaFree(d_mlp_fused);
    if (d_mid) (void)cudaFree(d_mid);
    if (d_up) (void)cudaFree(d_up);
    if (d_gate) (void)cudaFree(d_gate);
    if (d_head_padded) (void)cudaFree(d_head_padded);
    if (d_head_dynamic) (void)cudaFree(d_head_dynamic);
    if (d_input) (void)cudaFree(d_input);
    axiom_qwen38_nvfp4_mlp_destroy(mlp);
    axiom_qwen38_nvfp4_linear_destroy(down);
    axiom_qwen38_nvfp4_linear_destroy(up);
    axiom_qwen38_nvfp4_linear_destroy(gate);
    axiom_qwen38_nvfp4_linear_destroy(head);
    if (model) axiom_model_close(model);
    if (runtime) axiom_runtime_destroy(runtime);
    return result;
}
