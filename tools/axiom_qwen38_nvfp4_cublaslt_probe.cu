/*
 * Tensor-core capability gate for the native Unsloth Qwen3.8 NVFP4 loader.
 *
 * The matrix operands are supplied exactly in the checkpoint's packed E2M1
 * and E4M3 group-16 layout.  cuBLASLt is used only as the CUDA tensor-core
 * implementation; model loading, checkpoint layout and control flow remain
 * Axiom-native.  A scalar Axiom oracle runs over the same device bytes.
 */
#include <cuda_runtime.h>
#include <cublasLt.h>

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
constexpr uint32_t kBatch = 8u;  // FP4 tensor-core TN kernels require a nontrivial N dimension.

int fail(const char *what) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-cublaslt-probe: %s\n", what);
    return 1;
}

int status_fail(const char *what, int rc) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-cublaslt-probe: %s: %s\n", what,
                 axiom_status_string(rc));
    return 1;
}

int cuda_fail(const char *what, cudaError_t rc) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-cublaslt-probe: %s: %s\n", what,
                 cudaGetErrorString(rc));
    return 1;
}

int lt_fail(const char *what, cublasStatus_t rc) {
    std::fprintf(stderr, "axiom-qwen38-nvfp4-cublaslt-probe: %s: cublasLt status %d\n",
                 what, static_cast<int>(rc));
    return 1;
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

bool env_truthy(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
}

uint32_t env_runs(void) {
    const char *value = std::getenv("AXIOM_QWEN38_NVFP4_TC_RUNS");
    if (!value || !*value) return 0u;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return (end != value && *end == '\0' && parsed > 0u && parsed <= 10000u)
            ? static_cast<uint32_t>(parsed) : 0u;
}

int tensor_read(axiom_model *model, const std::string &name, void *dst, uint64_t bytes) {
    const int rc = axiom_model_tensor_read_slice(model, name.c_str(), 0, dst, bytes);
    if (rc != AXIOM_OK) {
        std::fprintf(stderr, "axiom-qwen38-nvfp4-cublaslt-probe: read %s: %s\n", name.c_str(),
                     axiom_status_string(rc));
    }
    return rc;
}

/* cuBLASLt VEC16_UE4M3 does not consume a row-major [outer][K/16]
 * scale matrix.  It mandates 128x4 tiled scale factors.  The Unsloth file
 * remains canonical row-major; this compact runtime repack is only the
 * tensor-core view and has no second copy of the 23 GiB checkpoint. */
size_t cublaslt_vec16_scale_bytes(uint32_t outer, uint32_t inner) {
    const uint32_t outer_tiles = (outer + 127u) / 128u;
    const uint32_t inner_tiles = (inner + 3u) / 4u;
    return static_cast<size_t>(outer_tiles) * inner_tiles * 512u;
}

std::vector<uint8_t> cublaslt_pack_vec16_scales(
        const uint8_t *source, uint32_t outer, uint32_t inner) {
    const uint32_t inner_tiles = (inner + 3u) / 4u;
    std::vector<uint8_t> packed(cublaslt_vec16_scale_bytes(outer, inner), 0u);
    for (uint32_t o = 0; o < outer; ++o) {
        const uint32_t outer_tile = o / 128u;
        const uint32_t outer_in_tile = o % 128u;
        for (uint32_t i = 0; i < inner; ++i) {
            const uint32_t inner_tile = i / 4u;
            const uint32_t inner_in_tile = i % 4u;
            const size_t base = (static_cast<size_t>(outer_tile) * inner_tiles + inner_tile) * 512u;
            const size_t offset = base + (outer_in_tile % 32u) * 16u +
                                  (outer_in_tile / 32u) * 4u + inner_in_tile;
            packed[offset] = source[static_cast<size_t>(o) * inner + i];
        }
    }
    return packed;
}

void destroy_layout(cublasLtMatrixLayout_t *layout) {
    if (*layout) {
        (void)cublasLtMatrixLayoutDestroy(*layout);
        *layout = nullptr;
    }
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
    if (argc >= 3 && !parse_layer(argv[2], &layer)) return fail("invalid NVFP4 layer");
    const char *projection = argc >= 4 ? argv[3] : "gate";
    uint32_t full_rows = 0u, cols = 0u;
    if (!projection_geometry(projection, &full_rows, &cols)) return fail("projection must be gate, up, or down");
    const bool full = env_truthy("AXIOM_QWEN38_NVFP4_PROBE_FULL");
    const uint32_t rows = full ? full_rows : (full_rows < kRows ? full_rows : kRows);
    const uint32_t benchmark_runs = env_runs();
    const uint32_t groups = cols / kGroup;
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * (cols / 2u);
    const uint64_t weight_scale_bytes = static_cast<uint64_t>(rows) * groups;

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_config);
    if (rc != AXIOM_OK) return status_fail("runtime", rc);

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
        return status_fail("model open", rc);
    }

    char prefix[192];
    std::snprintf(prefix, sizeof(prefix), "model.language_model.layers.%u.mlp.%s_proj", layer, projection);
    const std::string base(prefix);
    std::vector<uint8_t> weight(weight_bytes), weight_scale(weight_scale_bytes);
    float input_global_scale = 0.0f, weight_global_scale = 0.0f;
    rc = tensor_read(model, base + ".weight_packed", weight.data(), weight.size());
    if (rc == AXIOM_OK) rc = tensor_read(model, base + ".weight_scale", weight_scale.data(), weight_scale.size());
    if (rc == AXIOM_OK) rc = tensor_read(model, base + ".input_global_scale", &input_global_scale, sizeof(input_global_scale));
    if (rc == AXIOM_OK) rc = tensor_read(model, base + ".weight_global_scale", &weight_global_scale, sizeof(weight_global_scale));
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK || !std::isfinite(input_global_scale) || !std::isfinite(weight_global_scale) ||
        input_global_scale <= 0.0f || weight_global_scale <= 0.0f) {
        return status_fail("invalid Unsloth NVFP4 tensor", rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc);
    }
    for (uint8_t scale : weight_scale) if (!std::isfinite(e4m3fn(scale))) return fail("NaN weight scale");

    std::vector<float> input(cols);
    for (uint32_t i = 0; i < cols; ++i) {
        input[i] = 0.73f * std::sin(static_cast<float>(i) * 0.03125f) +
                   0.29f * std::cos(static_cast<float>(i) * 0.0078125f) +
                   static_cast<float>(static_cast<int>(i % 13u) - 6) * 0.015625f;
    }
    std::vector<float> ref(rows), got(static_cast<size_t>(rows) * kBatch);
    std::vector<uint8_t> input_scale_raw(groups);
    const std::vector<uint8_t> weight_scale_tc = cublaslt_pack_vec16_scales(weight_scale.data(), rows, groups);
    uint8_t *d_weight = nullptr, *d_weight_scale = nullptr, *d_weight_scale_tc = nullptr;
    uint8_t *d_packed_input = nullptr, *d_input_scale = nullptr, *d_input_scale_tc = nullptr;
    float *d_input = nullptr, *d_ref = nullptr, *d_got = nullptr;
    void *workspace = nullptr;
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, c_layout = nullptr, d_layout = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cudaEvent_t benchmark_start = nullptr, benchmark_stop = nullptr;
    float tc_ms_per_call = 0.0f;
    int result = 1;
    cudaError_t cuda_rc = cudaSetDevice(0);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_weight, weight.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_weight_scale, weight_scale.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_weight_scale_tc, weight_scale_tc.size());
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_input, input.size() * sizeof(float));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_packed_input, static_cast<size_t>(cols / 2u) * kBatch);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_input_scale, static_cast<size_t>(groups) * kBatch);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_input_scale_tc, cublaslt_vec16_scale_bytes(kBatch, groups));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_ref, ref.size() * sizeof(float));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&d_got, got.size() * sizeof(float));
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMalloc(&workspace, 8u * 1024u * 1024u);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_weight, weight.data(), weight.size(), cudaMemcpyHostToDevice);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_weight_scale, weight_scale.data(), weight_scale.size(), cudaMemcpyHostToDevice);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_weight_scale_tc, weight_scale_tc.data(), weight_scale_tc.size(), cudaMemcpyHostToDevice);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_rc != cudaSuccess) {
        (void)cuda_fail("device setup", cuda_rc);
        goto done;
    }

    rc = axiom_qwen38_nvfp4_quantize_reference_f32(
            0, d_input, input_global_scale, d_packed_input, d_input_scale, cols, nullptr);
    if (rc == AXIOM_OK) {
        for (uint32_t column = 1u; column < kBatch && cuda_rc == cudaSuccess; ++column) {
            cuda_rc = cudaMemcpy(d_packed_input + static_cast<size_t>(column) * (cols / 2u),
                                 d_packed_input, cols / 2u, cudaMemcpyDeviceToDevice);
            if (cuda_rc == cudaSuccess) {
                cuda_rc = cudaMemcpy(d_input_scale + static_cast<size_t>(column) * groups,
                                     d_input_scale, groups, cudaMemcpyDeviceToDevice);
            }
        }
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_nvfp4_w4a4_reference_matvec_f32(
                0, d_weight, d_weight_scale, weight_global_scale,
                d_packed_input, d_input_scale, input_global_scale,
                d_ref, rows, cols, nullptr);
    }
    if (rc != AXIOM_OK) {
        (void)status_fail("Axiom reference", rc);
        goto done;
    }
    if (cuda_rc != cudaSuccess) {
        (void)cuda_fail("input batch replication", cuda_rc);
        goto done;
    }
    cuda_rc = cudaMemcpy(input_scale_raw.data(), d_input_scale, input_scale_raw.size(), cudaMemcpyDeviceToHost);
    if (cuda_rc != cudaSuccess) {
        (void)cuda_fail("input scale download", cuda_rc);
        goto done;
    }
    {
        std::vector<uint8_t> input_scale_rows(static_cast<size_t>(groups) * kBatch);
        for (uint32_t column = 0; column < kBatch; ++column) {
            std::memcpy(input_scale_rows.data() + static_cast<size_t>(column) * groups,
                        input_scale_raw.data(), groups);
        }
        const std::vector<uint8_t> input_scale_tc =
                cublaslt_pack_vec16_scales(input_scale_rows.data(), kBatch, groups);
        cuda_rc = cudaMemcpy(d_input_scale_tc, input_scale_tc.data(), input_scale_tc.size(), cudaMemcpyHostToDevice);
        if (cuda_rc != cudaSuccess) {
            (void)cuda_fail("tiled input scale upload", cuda_rc);
            goto done;
        }
    }

    {
        cublasStatus_t lt_rc = cublasLtCreate(&handle);
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("cublasLtCreate", lt_rc);
            goto done;
        }
        lt_rc = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("matmul descriptor", lt_rc);
            goto done;
        }
        /* FP4 tensor cores support the documented TN form.  W is row-major
         * [M,K] on disk; viewing its identical bytes as a column-major [K,M]
         * operand and transposing A produces W without repacking. */
        const cublasOperation_t trans_a = CUBLAS_OP_T;
        const cublasOperation_t trans_b = CUBLAS_OP_N;
        const cublasLtMatmulMatrixScale_t vector16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vector16, sizeof(vector16));
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vector16, sizeof(vector16));
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_weight_scale_tc, sizeof(d_weight_scale_tc));
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_input_scale_tc, sizeof(d_input_scale_tc));
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("scale descriptor", lt_rc);
            goto done;
        }
        /* Both checkpoint global scales are quantization multipliers, so the
         * sole output multiplier is their reciprocal product. */
        const float alpha = 1.0f / (input_global_scale * weight_global_scale);
        const float beta = 0.0f;
        lt_rc = cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_4F_E2M1, cols, rows, cols);
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_4F_E2M1, cols, kBatch, cols);
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, rows, kBatch, rows);
        if (lt_rc == CUBLAS_STATUS_SUCCESS) lt_rc = cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F, rows, kBatch, rows);
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("matrix layout", lt_rc);
            goto done;
        }
        lt_rc = cublasLtMatmulPreferenceCreate(&preference);
        if (lt_rc == CUBLAS_STATUS_SUCCESS) {
            const size_t workspace_bytes = 8u * 1024u * 1024u;
            lt_rc = cublasLtMatmulPreferenceSetAttribute(preference,
                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
        }
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("matmul preference", lt_rc);
            goto done;
        }
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        lt_rc = cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_layout, b_layout, c_layout, d_layout,
                                                preference, 1, &heuristic, &returned);
        if (lt_rc != CUBLAS_STATUS_SUCCESS || returned != 1) {
            (void)lt_fail("FP4 heuristic", lt_rc == CUBLAS_STATUS_SUCCESS ? CUBLAS_STATUS_NOT_SUPPORTED : lt_rc);
            goto done;
        }
        const auto run_matmul = [&]() {
            return cublasLtMatmul(handle, op_desc, &alpha,
                                  d_weight, a_layout, d_packed_input, b_layout, &beta,
                                  d_got, c_layout, d_got, d_layout,
                                  &heuristic.algo, workspace, heuristic.workspaceSize, nullptr);
        };
        lt_rc = run_matmul();
        if (lt_rc != CUBLAS_STATUS_SUCCESS) {
            (void)lt_fail("FP4 matmul", lt_rc);
            goto done;
        }
        cuda_rc = cudaDeviceSynchronize();
        if (cuda_rc != cudaSuccess) {
            (void)cuda_fail("FP4 matmul completion", cuda_rc);
            goto done;
        }
        if (benchmark_runs != 0u) {
            cuda_rc = cudaEventCreate(&benchmark_start);
            if (cuda_rc == cudaSuccess) cuda_rc = cudaEventCreate(&benchmark_stop);
            if (cuda_rc == cudaSuccess) cuda_rc = cudaEventRecord(benchmark_start, nullptr);
            for (uint32_t run = 0; run < benchmark_runs && cuda_rc == cudaSuccess; ++run) {
                lt_rc = run_matmul();
                if (lt_rc != CUBLAS_STATUS_SUCCESS) break;
            }
            if (cuda_rc == cudaSuccess && lt_rc == CUBLAS_STATUS_SUCCESS) {
                cuda_rc = cudaEventRecord(benchmark_stop, nullptr);
            }
            if (cuda_rc == cudaSuccess) cuda_rc = cudaEventSynchronize(benchmark_stop);
            if (cuda_rc == cudaSuccess) cuda_rc = cudaEventElapsedTime(&tc_ms_per_call, benchmark_start, benchmark_stop);
            if (cuda_rc != cudaSuccess || lt_rc != CUBLAS_STATUS_SUCCESS) {
                if (lt_rc != CUBLAS_STATUS_SUCCESS) (void)lt_fail("FP4 benchmark", lt_rc);
                else (void)cuda_fail("FP4 benchmark", cuda_rc);
                goto done;
            }
            tc_ms_per_call /= static_cast<float>(benchmark_runs);
        }
    }
    cuda_rc = cudaMemcpy(ref.data(), d_ref, ref.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_rc == cudaSuccess) cuda_rc = cudaMemcpy(got.data(), d_got, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_rc != cudaSuccess) {
        (void)cuda_fail("output download", cuda_rc);
        goto done;
    }
    {
        float max_abs = 0.0f, max_rel = 0.0f;
        bool pass = true;
        for (uint32_t column = 0; column < kBatch; ++column) {
            for (uint32_t row = 0; row < rows; ++row) {
                const float abs_error = std::fabs(got[static_cast<size_t>(column) * rows + row] - ref[row]);
                max_abs = std::fmax(max_abs, abs_error);
                max_rel = std::fmax(max_rel, abs_error / std::fmax(1.0e-6f, std::fabs(ref[row])));
                if (abs_error > 5.0e-5f + 5.0e-3f * std::fabs(ref[row])) pass = false;
            }
        }
        /* Tensor-core reassociation is expected.  This threshold distinguishes
         * that from an invalid transpose, nibble order, or scale convention. */
        if (!pass) {
            for (uint32_t row = 0; row < 4u && row < rows; ++row) {
                std::fprintf(stderr,
                             "axiom-qwen38-nvfp4-cublaslt-probe: sample row=%u ref=%.9g got=%.9g\n",
                             row, static_cast<double>(ref[row]), static_cast<double>(got[row]));
            }
        }
        std::printf("{\"status\":\"%s\",\"source\":\"unsloth/Qwen3.8-27B-NVFP4\","
                    "\"backend\":\"axiom-cublaslt-fp4\",\"layer\":%u,\"projection\":\"%s\","
                    "\"rows\":%u,\"cols\":%u,\"batch\":%u,\"full\":%s,\"benchmark_runs\":%u,"
                    "\"tc_ms_per_call\":%.9g,\"max_abs\":%.9g,\"max_rel\":%.9g}\n",
                    pass ? "pass" : "fail", layer, projection, rows, cols, kBatch,
                    full ? "true" : "false", benchmark_runs, static_cast<double>(tc_ms_per_call),
                    static_cast<double>(max_abs), static_cast<double>(max_rel));
        result = pass ? 0 : 1;
    }

done:
    if (benchmark_stop) (void)cudaEventDestroy(benchmark_stop);
    if (benchmark_start) (void)cudaEventDestroy(benchmark_start);
    if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
    destroy_layout(&d_layout); destroy_layout(&c_layout); destroy_layout(&b_layout); destroy_layout(&a_layout);
    if (op_desc) (void)cublasLtMatmulDescDestroy(op_desc);
    if (handle) (void)cublasLtDestroy(handle);
    cuda_free(workspace); cuda_free(d_got); cuda_free(d_ref); cuda_free(d_input_scale_tc); cuda_free(d_input_scale);
    cuda_free(d_packed_input); cuda_free(d_input); cuda_free(d_weight_scale_tc); cuda_free(d_weight_scale); cuda_free(d_weight);
    return result;
}
