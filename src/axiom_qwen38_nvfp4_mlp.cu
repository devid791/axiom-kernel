/*
 * Resident Blackwell FP4 MLP for native Qwen3.8 NVFP4 checkpoints.
 *
 * This is Axiom execution code.  It reads compressed-tensors or NVIDIA
 * ModelOpt bytes through axiom_model, retains the packed weights unchanged on
 * the GPU, and gives cuBLASLt the documented scale-plane view for Blackwell
 * tensor cores.  No model conversion and no external runtime are involved.
 */
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4_mlp.h"

namespace {

constexpr uint32_t kGroup = 16u;
constexpr uint8_t kE4m3Epsilon = 0x20u;
constexpr uint8_t kE4m3MaxFinite = 0x7eu;
constexpr size_t kLtWorkspaceBytes = 8u * 1024u * 1024u;
constexpr uint32_t kAutotuneCompareThreads = 256u;

struct MatmulPlan {
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulHeuristicResult_t algo{};
};

struct Projection {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t groups = 0;
    float input_global = 0.0f;
    float weight_global = 0.0f;
    float alpha = 0.0f;
    uint8_t *weight = nullptr;
    uint8_t *weight_scale_tc = nullptr;
    uint8_t *input_packed = nullptr;
    uint8_t *input_scale_tc = nullptr;
    float *output = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    MatmulPlan plans[AXIOM_QWEN38_NVFP4_TC_BATCH]{};
    uint64_t device_bytes = 0;
};

struct ProjectionHostData {
    std::vector<uint8_t> weight;
    std::vector<uint8_t> scale;
    float input_global = 0.0f;
    float weight_global = 0.0f;
    float alpha = 0.0f;
};

bool valid_scale(float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool valid_e4m3fn(uint8_t value) {
    return (value & 0x7fu) != 0x7fu;
}

size_t scale_view_bytes(uint32_t outer, uint32_t inner) {
    const uint32_t outer_tiles = (outer + 127u) / 128u;
    const uint32_t inner_tiles = (inner + 3u) / 4u;
    return static_cast<size_t>(outer_tiles) * inner_tiles * 512u;
}

__host__ __device__ size_t scale_view_offset(uint32_t outer, uint32_t inner, uint32_t inner_tiles) {
    const uint32_t outer_tile = outer / 128u;
    const uint32_t outer_in_tile = outer % 128u;
    const uint32_t inner_tile = inner / 4u;
    const uint32_t inner_in_tile = inner % 4u;
    const size_t base = (static_cast<size_t>(outer_tile) * inner_tiles + inner_tile) * 512u;
    return base + (outer_in_tile % 32u) * 16u + (outer_in_tile / 32u) * 4u + inner_in_tile;
}

std::vector<uint8_t> pack_weight_scales(const uint8_t *canonical, uint32_t rows, uint32_t groups) {
    const uint32_t inner_tiles = (groups + 3u) / 4u;
    std::vector<uint8_t> packed(scale_view_bytes(rows, groups), 0u);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t group = 0; group < groups; ++group) {
            packed[scale_view_offset(row, group, inner_tiles)] =
                    canonical[static_cast<size_t>(row) * groups + group];
        }
    }
    return packed;
}

__device__ __forceinline__ float e4m3fn(uint8_t code) {
    if ((code & 0x7fu) == 0x7fu) return nanf("");
    const uint32_t exp = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    const float value = exp == 0u
            ? ldexpf(static_cast<float>(mantissa), -9)
            : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                     static_cast<int>(exp) - 7);
    return (code & 0x80u) ? -value : value;
}

__device__ __forceinline__ uint8_t encode_e4m3fn_scale(float value) {
    uint8_t code = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
            value, __NV_SATFINITE, __NV_E4M3));
    const uint8_t magnitude = code & 0x7fu;
    if (magnitude == 0u) return kE4m3Epsilon;
    if (magnitude == 0x7fu) return static_cast<uint8_t>((code & 0x80u) | kE4m3MaxFinite);
    return code;
}

/* Input is column-major [cols,batch].  Each thread owns one 16-value group
 * in one column and writes directly in the VEC16_UE4M3 scale-plane layout. */
__global__ void quantize_batch_to_tc_kernel(
        const float *input,
        float input_global,
        uint8_t *out_packed,
        uint8_t *out_scale_tc,
        uint32_t cols,
        uint32_t groups,
        uint32_t inner_tiles,
        uint32_t columns) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (group >= groups || column >= columns) return;
    const uint32_t first = group * kGroup;
    const float *in = input + static_cast<size_t>(column) * cols + first;
    float max_abs = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kGroup; ++i) max_abs = fmaxf(max_abs, fabsf(in[i]));
    const uint8_t scale = encode_e4m3fn_scale(max_abs * (1.0f / 6.0f) * input_global);
    const float dequant_scale = e4m3fn(scale) / input_global;
    const size_t scale_offset = scale_view_offset(column, group, inner_tiles);
    out_scale_tc[scale_offset] = scale;
    uint8_t *packed = out_packed + static_cast<size_t>(column) * (cols / 2u) + group * (kGroup / 2u);
    #pragma unroll
    for (uint32_t pair = 0; pair < kGroup / 2u; ++pair) {
        const uint8_t low = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                in[pair * 2u] / dequant_scale, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        const uint8_t high = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                in[pair * 2u + 1u] / dequant_scale, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        packed[pair] = static_cast<uint8_t>(low | (high << 4u));
    }
}

/* One 16-lane subgroup owns one NVFP4 scale group.  This preserves the
 * per-value conversion and exact max-derived scale while exposing the 16
 * independent conversions to the GPU instead of serializing them in one
 * thread. */
__global__ void quantize_batch_to_tc_warp_kernel(
        const float *input,
        float input_global,
        uint8_t *out_packed,
        uint8_t *out_scale_tc,
        uint32_t cols,
        uint32_t groups,
        uint32_t inner_tiles,
        uint32_t columns) {
    constexpr uint32_t kSubgroup = 16u;
    const uint32_t subgroup = threadIdx.x / kSubgroup;
    const uint32_t lane = threadIdx.x % kSubgroup;
    const uint32_t groups_per_block = blockDim.x / kSubgroup;
    const uint32_t linear_group = blockIdx.x * groups_per_block + subgroup;
    const uint32_t total_groups = groups * columns;
    if (linear_group >= total_groups) return;
    const uint32_t column = linear_group / groups;
    const uint32_t group = linear_group - column * groups;
    const uint32_t feature = group * kGroup + lane;
    const float value = input[static_cast<size_t>(column) * cols + feature];
    float max_abs = fabsf(value);
    #pragma unroll
    for (uint32_t offset = kSubgroup / 2u; offset != 0u; offset >>= 1u) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset, kSubgroup));
    }
    uint32_t scale_code = lane == 0u
            ? encode_e4m3fn_scale(max_abs * (1.0f / 6.0f) * input_global)
            : 0u;
    scale_code = __shfl_sync(0xffffffffu, scale_code, 0, kSubgroup);
    const float dequant_scale = e4m3fn(static_cast<uint8_t>(scale_code)) / input_global;
    if (lane == 0u) {
        out_scale_tc[scale_view_offset(column, group, inner_tiles)] =
                static_cast<uint8_t>(scale_code);
    }
    const uint32_t nibble = static_cast<uint32_t>(__nv_cvt_float_to_fp4(
            value / dequant_scale, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
    const uint32_t next_nibble = __shfl_down_sync(
            0xffffffffu, nibble, 1u, kSubgroup) & 0x0fu;
    if ((lane & 1u) == 0u) {
        uint8_t *packed = out_packed + static_cast<size_t>(column) * (cols / 2u) +
                group * (kGroup / 2u);
        packed[lane / 2u] = static_cast<uint8_t>(nibble | (next_nibble << 4u));
    }
}

/* Fuses the exact serving boundary between the packed gate/up projection and
 * the down projection.  The intermediate remains F32, matching the previous
 * store/load boundary, but never materializes in global memory: every thread
 * owns one complete 16-value NVFP4 scale group. */
__global__ void silu_mul_quantize_batch_to_tc_kernel(
        const float *gate_up,
        float input_global,
        uint8_t *out_packed,
        uint8_t *out_scale_tc,
        uint32_t rows,
        uint32_t groups,
        uint32_t inner_tiles,
        uint32_t columns) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (group >= groups || column >= columns) return;
    const uint32_t first = group * kGroup;
    const float *column_base = gate_up + static_cast<size_t>(column) * rows * 2u;
    const float *gate = column_base + first;
    const float *up = column_base + rows + first;
    float values[kGroup];
    float max_abs = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kGroup; ++i) {
        const float x = gate[i];
        const float value = (x / (1.0f + expf(-x))) * up[i];
        values[i] = value;
        max_abs = fmaxf(max_abs, fabsf(value));
    }
    const uint8_t scale = encode_e4m3fn_scale(
            max_abs * (1.0f / 6.0f) * input_global);
    const float dequant_scale = e4m3fn(scale) / input_global;
    out_scale_tc[scale_view_offset(column, group, inner_tiles)] = scale;
    uint8_t *packed = out_packed +
            static_cast<size_t>(column) * (rows / 2u) + group * (kGroup / 2u);
    #pragma unroll
    for (uint32_t pair = 0; pair < kGroup / 2u; ++pair) {
        const uint8_t low = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                values[pair * 2u] / dequant_scale,
                __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        const uint8_t high = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                values[pair * 2u + 1u] / dequant_scale,
                __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        packed[pair] = static_cast<uint8_t>(low | (high << 4u));
    }
}

__global__ void silu_mul_quantize_batch_to_tc_warp_kernel(
        const float *gate_up,
        float input_global,
        uint8_t *out_packed,
        uint8_t *out_scale_tc,
        uint32_t rows,
        uint32_t groups,
        uint32_t inner_tiles,
        uint32_t columns) {
    constexpr uint32_t kSubgroup = 16u;
    const uint32_t subgroup = threadIdx.x / kSubgroup;
    const uint32_t lane = threadIdx.x % kSubgroup;
    const uint32_t groups_per_block = blockDim.x / kSubgroup;
    const uint32_t linear_group = blockIdx.x * groups_per_block + subgroup;
    const uint32_t total_groups = groups * columns;
    if (linear_group >= total_groups) return;
    const uint32_t column = linear_group / groups;
    const uint32_t group = linear_group - column * groups;
    const uint32_t feature = group * kGroup + lane;
    const float *column_base = gate_up + static_cast<size_t>(column) * rows * 2u;
    const float x = column_base[feature];
    const float value = (x / (1.0f + expf(-x))) * column_base[rows + feature];
    float max_abs = fabsf(value);
    #pragma unroll
    for (uint32_t offset = kSubgroup / 2u; offset != 0u; offset >>= 1u) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffu, max_abs, offset, kSubgroup));
    }
    uint32_t scale_code = lane == 0u
            ? encode_e4m3fn_scale(max_abs * (1.0f / 6.0f) * input_global)
            : 0u;
    scale_code = __shfl_sync(0xffffffffu, scale_code, 0, kSubgroup);
    const float dequant_scale = e4m3fn(static_cast<uint8_t>(scale_code)) / input_global;
    if (lane == 0u) {
        out_scale_tc[scale_view_offset(column, group, inner_tiles)] =
                static_cast<uint8_t>(scale_code);
    }
    const uint32_t nibble = static_cast<uint32_t>(__nv_cvt_float_to_fp4(
            value / dequant_scale, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
    const uint32_t next_nibble = __shfl_down_sync(
            0xffffffffu, nibble, 1u, kSubgroup) & 0x0fu;
    if ((lane & 1u) == 0u) {
        uint8_t *packed = out_packed + static_cast<size_t>(column) * (rows / 2u) +
                group * (kGroup / 2u);
        packed[lane / 2u] = static_cast<uint8_t>(nibble | (next_nibble << 4u));
    }
}

__global__ void silu_mul_batch_kernel(const float *gate, const float *up, float *out, uint32_t count) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = gate[index];
    out[index] = (x / (1.0f + expf(-x))) * up[index];
}

void projection_destroy(Projection *projection) {
    if (!projection) return;
    for (uint32_t index = 0u; index < AXIOM_QWEN38_NVFP4_TC_BATCH; ++index) {
        MatmulPlan &plan = projection->plans[index];
        if (plan.d_layout) (void)cublasLtMatrixLayoutDestroy(plan.d_layout);
        if (plan.c_layout) (void)cublasLtMatrixLayoutDestroy(plan.c_layout);
        if (plan.b_layout) (void)cublasLtMatrixLayoutDestroy(plan.b_layout);
    }
    if (projection->a_layout) (void)cublasLtMatrixLayoutDestroy(projection->a_layout);
    if (projection->desc) (void)cublasLtMatmulDescDestroy(projection->desc);
    if (projection->output) (void)cudaFree(projection->output);
    if (projection->input_scale_tc) (void)cudaFree(projection->input_scale_tc);
    if (projection->input_packed) (void)cudaFree(projection->input_packed);
    if (projection->weight_scale_tc) (void)cudaFree(projection->weight_scale_tc);
    if (projection->weight) (void)cudaFree(projection->weight);
    *projection = Projection{};
}

int read_tensor(axiom_model *model, const std::string &name, void *dst, uint64_t bytes) {
    return axiom_model_tensor_read_slice(model, name.c_str(), 0, dst, bytes);
}

int tensor_info(axiom_model *model, const std::string &name, axiom_tensor_info *out) {
    if (!model || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = AXIOM_ABI_VERSION;
    return axiom_model_tensor_info_get(model, name.c_str(), out);
}

bool exact_matrix(
        const axiom_tensor_info &info,
        axiom_tensor_dtype dtype,
        uint64_t rows,
        uint64_t cols,
        uint64_t bytes) {
    return info.dtype == dtype && info.rank == 2u &&
            info.shape[0] == rows && info.shape[1] == cols &&
            info.byte_count == bytes;
}

bool exact_f32_scalar(const axiom_tensor_info &info) {
    const bool scalar_shape = info.rank == 0u ||
            (info.rank == 1u && info.shape[0] == 1u);
    return info.dtype == AXIOM_TENSOR_DTYPE_F32 && scalar_shape &&
            info.byte_count == sizeof(float);
}

int cublas_to_axiom(cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? AXIOM_OK : AXIOM_ERR_RUNTIME;
}

__global__ void nvfp4_exact_output_compare_kernel(
        const float *__restrict__ reference,
        const float *__restrict__ candidate,
        uint64_t count,
        uint32_t *__restrict__ mismatch) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && __float_as_uint(reference[index]) != __float_as_uint(candidate[index])) {
        atomicExch(mismatch, 1u);
    }
}

__global__ void nvfp4_autotune_packed_pattern_kernel(
        uint8_t *__restrict__ values,
        uint64_t count,
        uint32_t seed) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t mixed = static_cast<uint32_t>(index) * 747796405u + seed * 2891336453u;
    mixed = ((mixed >> ((mixed >> 28u) + 4u)) ^ mixed) * 277803737u;
    mixed = (mixed >> 22u) ^ mixed;
    values[index] = static_cast<uint8_t>(mixed);
}

__global__ void nvfp4_autotune_scale_pattern_kernel(
        uint8_t *__restrict__ values,
        uint64_t count,
        uint32_t seed) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t mixed = static_cast<uint32_t>(index) * 1597334677u + seed * 3812015801u;
    mixed ^= mixed >> 16u;
    values[index] = static_cast<uint8_t>(0x30u + ((mixed >> 5u) & 0x0cu));
}

bool nvfp4_autotune_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_NVFP4_AUTOTUNE");
    if (!value || value[0] == '\0') {
        value = std::getenv("AXIOM_QWEN38_MATMUL_AUTOTUNE");
    }
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
}

bool nvfp4_warp_quant_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_NVFP4_WARP_QUANT");
    return !value || value[0] == '\0' ||
            (std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
             std::strcmp(value, "False") != 0);
}

struct AlgoCacheKey {
    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint32_t columns = 0u;

    bool operator<(const AlgoCacheKey &other) const {
        if (rows != other.rows) return rows < other.rows;
        if (cols != other.cols) return cols < other.cols;
        return columns < other.columns;
    }
};

struct AlgoCacheValue {
    cublasLtMatmulHeuristicResult_t heuristic{};
    uint32_t candidate_index = 0u;
    uint32_t candidate_count = 0u;
    float milliseconds = 0.0f;
};

std::atomic_flag g_algo_cache_lock = ATOMIC_FLAG_INIT;
std::map<AlgoCacheKey, AlgoCacheValue> g_algo_cache;

struct AlgoCacheGuard {
    AlgoCacheGuard() {
        while (g_algo_cache_lock.test_and_set(std::memory_order_acquire)) {}
    }
    ~AlgoCacheGuard() {
        g_algo_cache_lock.clear(std::memory_order_release);
    }
};

bool nvfp4_candidate_is_exact(
        cublasLtHandle_t handle,
        Projection *projection,
        MatmulPlan *plan,
        float *reference,
        float *candidate_output,
        uint32_t *mismatch_device,
        void *workspace,
        size_t input_bytes,
        size_t scale_bytes,
        uint64_t output_elements,
        const cublasLtMatmulHeuristicResult_t &reference_candidate,
        const cublasLtMatmulHeuristicResult_t &candidate,
        cudaStream_t stream) {
    if (!handle || !projection || !plan || !reference || !candidate_output ||
        !mismatch_device || !workspace || !stream || input_bytes == 0u ||
        scale_bytes == 0u || output_elements == 0u) {
        return false;
    }
    constexpr uint32_t kValidationPatterns = 3u;
    constexpr uint32_t kPatternSeeds[kValidationPatterns] = {
            0x243f6a88u, 0x9e3779b9u, 0xb7e15162u};
    const float beta = 0.0f;
    const uint64_t input_grid =
            (input_bytes + kAutotuneCompareThreads - 1u) / kAutotuneCompareThreads;
    const uint64_t scale_grid =
            (scale_bytes + kAutotuneCompareThreads - 1u) / kAutotuneCompareThreads;
    const uint64_t output_grid =
            (output_elements + kAutotuneCompareThreads - 1u) / kAutotuneCompareThreads;
    if (input_grid > std::numeric_limits<uint32_t>::max() ||
        scale_grid > std::numeric_limits<uint32_t>::max() ||
        output_grid > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    for (uint32_t pattern = 0u; pattern < kValidationPatterns; ++pattern) {
        nvfp4_autotune_packed_pattern_kernel<<<
                static_cast<uint32_t>(input_grid), kAutotuneCompareThreads, 0, stream>>>(
                projection->input_packed, input_bytes, kPatternSeeds[pattern]);
        nvfp4_autotune_scale_pattern_kernel<<<
                static_cast<uint32_t>(scale_grid), kAutotuneCompareThreads, 0, stream>>>(
                projection->input_scale_tc, scale_bytes, kPatternSeeds[pattern]);
        if (cudaGetLastError() != cudaSuccess) return false;
        cublasStatus_t status = cublasLtMatmul(
                handle, projection->desc, &projection->alpha,
                projection->weight, projection->a_layout,
                projection->input_packed, plan->b_layout,
                &beta, reference, plan->c_layout, reference, plan->d_layout,
                &reference_candidate.algo, workspace,
                reference_candidate.workspaceSize, stream);
        if (status != CUBLAS_STATUS_SUCCESS) return false;
        status = cublasLtMatmul(
                handle, projection->desc, &projection->alpha,
                projection->weight, projection->a_layout,
                projection->input_packed, plan->b_layout,
                &beta, candidate_output, plan->c_layout, candidate_output, plan->d_layout,
                &candidate.algo, workspace, candidate.workspaceSize, stream);
        if (status != CUBLAS_STATUS_SUCCESS) return false;
        uint32_t mismatch = 1u;
        cudaError_t compare_status = cudaMemsetAsync(
                mismatch_device, 0, sizeof(uint32_t), stream);
        if (compare_status == cudaSuccess) {
            nvfp4_exact_output_compare_kernel<<<
                    static_cast<uint32_t>(output_grid), kAutotuneCompareThreads, 0, stream>>>(
                    reference, candidate_output, output_elements, mismatch_device);
            compare_status = cudaGetLastError();
        }
        if (compare_status == cudaSuccess) compare_status = cudaMemcpyAsync(
                &mismatch, mismatch_device, sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream);
        if (compare_status == cudaSuccess) compare_status = cudaStreamSynchronize(stream);
        if (compare_status != cudaSuccess || mismatch != 0u) return false;
    }
    return true;
}

/* cuBLASLt's heuristic order is a useful default, but it is not a measured
 * ordering for this exact Blackwell GPU/driver/shape.  When explicitly
 * enabled, benchmark the returned M8 candidates once per unique projection
 * geometry and reuse the winning algorithm for every layer.  The operation
 * is load-time only; serving and CUDA graph capture remain allocation-free. */
cublasStatus_t select_projection_algorithm(
        cublasLtHandle_t handle,
        Projection *projection,
        MatmulPlan *plan,
        cublasLtMatmulPreference_t preference,
        uint32_t columns) {
    if (!handle || !projection || !plan || !preference || columns == 0u) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool tune = nvfp4_autotune_enabled() &&
            columns == AXIOM_QWEN38_NVFP4_TC_BATCH;
    const AlgoCacheKey key{projection->rows, projection->cols, columns};
    if (tune) {
        AlgoCacheGuard guard;
        const auto cached = g_algo_cache.find(key);
        if (cached != g_algo_cache.end()) {
            plan->algo = cached->second.heuristic;
            return CUBLAS_STATUS_SUCCESS;
        }
    }

    constexpr int kMaxCandidates = 32;
    cublasLtMatmulHeuristicResult_t candidates[kMaxCandidates]{};
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, projection->desc, projection->a_layout, plan->b_layout,
            plan->c_layout, plan->d_layout, preference,
            tune ? kMaxCandidates : 1, candidates, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned < 1) {
        return status == CUBLAS_STATUS_SUCCESS
                ? CUBLAS_STATUS_NOT_SUPPORTED : status;
    }
    plan->algo = candidates[0];
    if (!tune || returned == 1) {
        if (tune) {
            {
                AlgoCacheGuard guard;
                g_algo_cache.emplace(key, AlgoCacheValue{
                        candidates[0], 0u, static_cast<uint32_t>(returned), 0.0f});
            }
            std::fprintf(stderr,
                         "axiom-qwen38-nvfp4: autotune rows=%u cols=%u columns=%u "
                         "candidates=%d selected=0 measured_ms=0\n",
                         projection->rows, projection->cols, columns, returned);
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    void *workspace = nullptr;
    float *output = nullptr;
    float *reference = nullptr;
    uint32_t *mismatch_device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    const size_t input_bytes =
            static_cast<size_t>(projection->cols / 2u) * columns;
    const size_t scale_bytes = scale_view_bytes(columns, projection->groups);
    const size_t output_bytes =
            static_cast<size_t>(projection->rows) * columns * sizeof(float);
    const uint64_t output_elements =
            static_cast<uint64_t>(projection->rows) * columns;
    cudaError_t cuda_status = cudaMalloc(&workspace, kLtWorkspaceBytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&output, output_bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&reference, output_bytes);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&mismatch_device, sizeof(uint32_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking);
    if (cuda_status == cudaSuccess) cuda_status = cudaEventCreate(&begin);
    if (cuda_status == cudaSuccess) cuda_status = cudaEventCreate(&end);
    constexpr uint32_t kWarmupRuns = 2u;
    constexpr uint32_t kMeasuredRuns = 20u;
    constexpr float kMinimumSpeedup = 1.03f;
    constexpr float kCandidateReplacementSpeedup = 1.03f;
    const float beta = 0.0f;
    uint32_t best_index = 0u;
    uint32_t exact_candidates = 0u;
    float best_ms = std::numeric_limits<float>::infinity();
    float baseline_ms = std::numeric_limits<float>::infinity();
    if (cuda_status == cudaSuccess) {
        for (int candidate = 0; candidate < returned; ++candidate) {
            if (candidates[candidate].state != CUBLAS_STATUS_SUCCESS ||
                candidates[candidate].workspaceSize > kLtWorkspaceBytes) {
                continue;
            }
            const bool exact = candidate == 0 || nvfp4_candidate_is_exact(
                    handle, projection, plan, reference, output, mismatch_device,
                    workspace, input_bytes, scale_bytes, output_elements,
                    candidates[0], candidates[candidate], stream);
            if (!exact) {
                (void)cudaGetLastError();
                continue;
            }
            ++exact_candidates;
            const uint64_t input_grid =
                    (input_bytes + kAutotuneCompareThreads - 1u) /
                    kAutotuneCompareThreads;
            const uint64_t scale_grid =
                    (scale_bytes + kAutotuneCompareThreads - 1u) /
                    kAutotuneCompareThreads;
            nvfp4_autotune_packed_pattern_kernel<<<
                    static_cast<uint32_t>(input_grid), kAutotuneCompareThreads, 0, stream>>>(
                    projection->input_packed, input_bytes, 0xd1b54a35u);
            nvfp4_autotune_scale_pattern_kernel<<<
                    static_cast<uint32_t>(scale_grid), kAutotuneCompareThreads, 0, stream>>>(
                    projection->input_scale_tc, scale_bytes, 0xd1b54a35u);
            if (cudaGetLastError() != cudaSuccess) continue;
            cublasStatus_t run_status = CUBLAS_STATUS_SUCCESS;
            for (uint32_t run = 0u; run < kWarmupRuns &&
                 run_status == CUBLAS_STATUS_SUCCESS; ++run) {
                run_status = cublasLtMatmul(
                        handle, projection->desc, &projection->alpha,
                        projection->weight, projection->a_layout,
                        projection->input_packed, plan->b_layout,
                        &beta, output, plan->c_layout, output, plan->d_layout,
                        &candidates[candidate].algo, workspace,
                        candidates[candidate].workspaceSize, stream);
            }
            if (run_status != CUBLAS_STATUS_SUCCESS ||
                cudaStreamSynchronize(stream) != cudaSuccess ||
                cudaEventRecord(begin, stream) != cudaSuccess) {
                (void)cudaGetLastError();
                continue;
            }
            for (uint32_t run = 0u; run < kMeasuredRuns &&
                 run_status == CUBLAS_STATUS_SUCCESS; ++run) {
                run_status = cublasLtMatmul(
                        handle, projection->desc, &projection->alpha,
                        projection->weight, projection->a_layout,
                        projection->input_packed, plan->b_layout,
                        &beta, output, plan->c_layout, output, plan->d_layout,
                        &candidates[candidate].algo, workspace,
                        candidates[candidate].workspaceSize, stream);
            }
            float elapsed_ms = 0.0f;
            if (run_status != CUBLAS_STATUS_SUCCESS ||
                cudaEventRecord(end, stream) != cudaSuccess ||
                cudaEventSynchronize(end) != cudaSuccess ||
                cudaEventElapsedTime(&elapsed_ms, begin, end) != cudaSuccess) {
                (void)cudaGetLastError();
                continue;
            }
            const float per_call_ms = elapsed_ms / kMeasuredRuns;
            if (candidate == 0) baseline_ms = per_call_ms;
            if (!std::isfinite(best_ms) ||
                per_call_ms * kCandidateReplacementSpeedup < best_ms) {
                best_ms = per_call_ms;
                best_index = static_cast<uint32_t>(candidate);
            }
        }
    }
    if (end) (void)cudaEventDestroy(end);
    if (begin) (void)cudaEventDestroy(begin);
    if (stream) (void)cudaStreamDestroy(stream);
    if (mismatch_device) (void)cudaFree(mismatch_device);
    if (reference) (void)cudaFree(reference);
    if (output) (void)cudaFree(output);
    if (workspace) (void)cudaFree(workspace);

    if (!std::isfinite(best_ms)) {
        best_index = 0u;
        best_ms = 0.0f;
    } else if (best_index != 0u &&
               (!std::isfinite(baseline_ms) || best_ms * kMinimumSpeedup >= baseline_ms)) {
        best_index = 0u;
        best_ms = baseline_ms;
    }
    plan->algo = candidates[best_index];
    {
        AlgoCacheGuard guard;
        g_algo_cache.emplace(
                key, AlgoCacheValue{candidates[best_index], best_index,
                                    static_cast<uint32_t>(returned), best_ms});
    }
    int algo_id = -1;
    int split_k = 0;
    size_t written = 0u;
    (void)cublasLtMatmulAlgoConfigGetAttribute(
            &candidates[best_index].algo, CUBLASLT_ALGO_CONFIG_ID,
            &algo_id, sizeof(algo_id), &written);
    (void)cublasLtMatmulAlgoConfigGetAttribute(
            &candidates[best_index].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &split_k, sizeof(split_k), &written);
    std::fprintf(stderr,
                 "axiom-qwen38-nvfp4: autotune rows=%u cols=%u columns=%u "
                 "candidates=%d exact=%u selected=%u algo=%d split_k=%d baseline_ms=%.6f "
                 "measured_ms=%.6f workspace=%zu\n",
                 projection->rows, projection->cols, columns, returned,
                 exact_candidates, best_index, algo_id, split_k, static_cast<double>(baseline_ms),
                 static_cast<double>(best_ms), candidates[best_index].workspaceSize);
    return CUBLAS_STATUS_SUCCESS;
}

int read_projection_host_data(
        axiom_model *model,
        const char *base_name,
        uint32_t rows,
        uint32_t cols,
        ProjectionHostData *out) {
    if (!model || !base_name || !base_name[0] || !out || rows == 0u || cols == 0u ||
        (cols % kGroup) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t groups = cols / kGroup;
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * (cols / 2u);
    const uint64_t scale_bytes = static_cast<uint64_t>(rows) * groups;
    if (weight_bytes > std::numeric_limits<size_t>::max() ||
        scale_bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_BUDGET;
    }
    const std::string base(base_name);
    const std::string legacy_weight_name = base + ".weight_packed";
    const std::string modelopt_weight_name = base + ".weight";
    const std::string scale_name = base + ".weight_scale";
    const std::string legacy_input_name = base + ".input_global_scale";
    const std::string legacy_global_name = base + ".weight_global_scale";
    const std::string modelopt_input_name = base + ".input_scale";
    const std::string modelopt_global_name = base + ".weight_scale_2";

    axiom_tensor_info weight_info{};
    axiom_tensor_info scale_info{};
    axiom_tensor_info input_info{};
    axiom_tensor_info global_info{};
    int rc = tensor_info(model, legacy_weight_name, &weight_info);
    const bool modelopt = rc != AXIOM_OK;
    if (modelopt) rc = tensor_info(model, modelopt_weight_name, &weight_info);
    if (rc == AXIOM_OK) rc = tensor_info(model, scale_name, &scale_info);
    if (rc == AXIOM_OK) rc = tensor_info(
            model, modelopt ? modelopt_input_name : legacy_input_name, &input_info);
    if (rc == AXIOM_OK) rc = tensor_info(
            model, modelopt ? modelopt_global_name : legacy_global_name, &global_info);
    if (rc != AXIOM_OK ||
        !exact_matrix(weight_info, AXIOM_TENSOR_DTYPE_U8, rows, cols / 2u, weight_bytes) ||
        !exact_matrix(scale_info, AXIOM_TENSOR_DTYPE_F8_E4M3, rows, groups, scale_bytes) ||
        !exact_f32_scalar(input_info) || !exact_f32_scalar(global_info)) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }

    ProjectionHostData data{};
    try {
        data.weight.resize(static_cast<size_t>(weight_bytes));
        data.scale.resize(static_cast<size_t>(scale_bytes));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    rc = read_tensor(
            model, modelopt ? modelopt_weight_name : legacy_weight_name,
            data.weight.data(), weight_bytes);
    if (rc == AXIOM_OK) rc = read_tensor(model, scale_name, data.scale.data(), scale_bytes);
    if (rc == AXIOM_OK) rc = read_tensor(
            model, modelopt ? modelopt_input_name : legacy_input_name,
            &data.input_global, sizeof(float));
    if (rc == AXIOM_OK) rc = read_tensor(
            model, modelopt ? modelopt_global_name : legacy_global_name,
            &data.weight_global, sizeof(float));
    if (rc != AXIOM_OK || !valid_scale(data.input_global) ||
        !valid_scale(data.weight_global)) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    /* ModelOpt serializes dequant multipliers.  The existing Axiom kernels
     * store their reciprocals: block_scale/global and alpha=1/(ig*wg).
     * Therefore this is an exact semantic mapping, not a requantization. */
    if (modelopt) {
        const float input_scale = data.input_global;
        const float weight_scale = data.weight_global;
        data.alpha = input_scale * weight_scale;
        data.input_global = 1.0f / data.input_global;
        data.weight_global = 1.0f / data.weight_global;
        if (!valid_scale(data.alpha) || !valid_scale(data.input_global) ||
            !valid_scale(data.weight_global)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    } else {
        data.alpha = 1.0f / (data.input_global * data.weight_global);
        if (!valid_scale(data.alpha)) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint8_t value : data.scale) {
        if (!valid_e4m3fn(value)) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = std::move(data);
    return AXIOM_OK;
}

bool equal_float_bits(float left, float right) {
    uint32_t left_bits = 0u;
    uint32_t right_bits = 0u;
    std::memcpy(&left_bits, &left, sizeof(left_bits));
    std::memcpy(&right_bits, &right, sizeof(right_bits));
    return left_bits == right_bits;
}

int upload_projection(
        const ProjectionHostData &host,
        int device,
        uint32_t rows,
        uint32_t cols,
        cublasLtHandle_t handle,
        bool allocate_output,
        Projection *out) {
    if (!handle || !out || rows == 0u || cols == 0u || (cols % kGroup) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    Projection projection{};
    projection.rows = rows;
    projection.cols = cols;
    projection.groups = cols / kGroup;
    projection.input_global = host.input_global;
    projection.weight_global = host.weight_global;
    projection.alpha = host.alpha;
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * (cols / 2u);
    const uint64_t canonical_scale_bytes = static_cast<uint64_t>(rows) * projection.groups;
    if (host.weight.size() != weight_bytes || host.scale.size() != canonical_scale_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<uint8_t> tiled_scale;
    try {
        tiled_scale = pack_weight_scales(host.scale.data(), rows, projection.groups);
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    cudaError_t cuda_status = cudaMalloc(&projection.weight, static_cast<size_t>(weight_bytes));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&projection.weight_scale_tc, tiled_scale.size());
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            &projection.input_packed,
            static_cast<size_t>(cols / 2u) * AXIOM_QWEN38_NVFP4_TC_BATCH);
    const size_t input_scale_bytes = scale_view_bytes(AXIOM_QWEN38_NVFP4_TC_BATCH, projection.groups);
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&projection.input_scale_tc, input_scale_bytes);
    if (cuda_status == cudaSuccess && allocate_output) cuda_status = cudaMalloc(
            &projection.output,
            static_cast<size_t>(rows) * AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(
            projection.weight, host.weight.data(), static_cast<size_t>(weight_bytes), cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) cuda_status = cudaMemcpy(
            projection.weight_scale_tc, tiled_scale.data(), tiled_scale.size(), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        projection_destroy(&projection);
        return cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }

    cublasStatus_t lt_status = cublasLtMatmulDescCreate(&projection.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t vector16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vector16, sizeof(vector16));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vector16, sizeof(vector16));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &projection.weight_scale_tc, sizeof(projection.weight_scale_tc));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulDescSetAttribute(
            projection.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &projection.input_scale_tc, sizeof(projection.input_scale_tc));
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatrixLayoutCreate(
            &projection.a_layout, CUDA_R_4F_E2M1, cols, rows, cols);
    cublasLtMatmulPreference_t preference = nullptr;
    if (lt_status == CUBLAS_STATUS_SUCCESS) lt_status = cublasLtMatmulPreferenceCreate(&preference);
    if (lt_status == CUBLAS_STATUS_SUCCESS) {
        lt_status = cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &kLtWorkspaceBytes, sizeof(kLtWorkspaceBytes));
    }
    for (uint32_t columns = 1u;
         columns <= AXIOM_QWEN38_NVFP4_TC_BATCH && lt_status == CUBLAS_STATUS_SUCCESS;
         ++columns) {
        MatmulPlan &plan = projection.plans[columns - 1u];
        lt_status = cublasLtMatrixLayoutCreate(
                &plan.b_layout, CUDA_R_4F_E2M1, cols, columns, cols);
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            lt_status = cublasLtMatrixLayoutCreate(
                    &plan.c_layout, CUDA_R_32F, rows, columns, rows);
        }
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            lt_status = cublasLtMatrixLayoutCreate(
                    &plan.d_layout, CUDA_R_32F, rows, columns, rows);
        }
        if (lt_status == CUBLAS_STATUS_SUCCESS) {
            lt_status = select_projection_algorithm(
                    handle, &projection, &plan, preference, columns);
        }
    }
    if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
    if (lt_status != CUBLAS_STATUS_SUCCESS) {
        projection_destroy(&projection);
        return cublas_to_axiom(lt_status);
    }
    projection.device_bytes = weight_bytes + tiled_scale.size() +
            static_cast<uint64_t>(cols / 2u) * AXIOM_QWEN38_NVFP4_TC_BATCH +
            input_scale_bytes + (allocate_output
                    ? static_cast<uint64_t>(rows) *
                            AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float)
                    : 0u);
    *out = projection;
    return AXIOM_OK;
}

int projection_load_base(
        axiom_model *model,
        int device,
        const char *base_name,
        uint32_t rows,
        uint32_t cols,
        cublasLtHandle_t handle,
        bool allocate_output,
        Projection *out) {
    ProjectionHostData host{};
    const int rc = read_projection_host_data(
            model, base_name, rows, cols, &host);
    if (rc != AXIOM_OK) return rc;
    return upload_projection(
            host, device, rows, cols, handle, allocate_output, out);
}

int projection_load_fused_base(
        axiom_model *model,
        int device,
        const char *first_base,
        const char *second_base,
        uint32_t rows_per_projection,
        uint32_t cols,
        cublasLtHandle_t handle,
        Projection *out) {
    ProjectionHostData first{};
    ProjectionHostData second{};
    int rc = read_projection_host_data(
            model, first_base, rows_per_projection, cols, &first);
    if (rc == AXIOM_OK) {
        rc = read_projection_host_data(
                model, second_base, rows_per_projection, cols, &second);
    }
    if (rc != AXIOM_OK) return rc;
    if (!equal_float_bits(first.input_global, second.input_global) ||
        !equal_float_bits(first.weight_global, second.weight_global) ||
        !equal_float_bits(first.alpha, second.alpha)) {
        /* A single Lt matmul has one activation-global scale and alpha.  Keep
         * exact checkpoint semantics by selecting the separate fallback when
         * the two projections cannot share those scalars bit-for-bit. */
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    ProjectionHostData fused{};
    fused.input_global = first.input_global;
    fused.weight_global = first.weight_global;
    fused.alpha = first.alpha;
    try {
        fused.weight.reserve(first.weight.size() + second.weight.size());
        fused.weight.insert(fused.weight.end(), first.weight.begin(), first.weight.end());
        fused.weight.insert(fused.weight.end(), second.weight.begin(), second.weight.end());
        fused.scale.reserve(first.scale.size() + second.scale.size());
        fused.scale.insert(fused.scale.end(), first.scale.begin(), first.scale.end());
        fused.scale.insert(fused.scale.end(), second.scale.begin(), second.scale.end());
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    if (rows_per_projection > std::numeric_limits<uint32_t>::max() / 2u) {
        return AXIOM_ERR_BUDGET;
    }
    return upload_projection(
            fused, device, rows_per_projection * 2u, cols, handle, true, out);
}

int projection_load(
        axiom_model *model,
        int device,
        uint32_t layer,
        const char *name,
        uint32_t rows,
        uint32_t cols,
        cublasLtHandle_t handle,
        bool allocate_output,
        Projection *out) {
    if (!name) return AXIOM_ERR_INVALID_ARGUMENT;
    char base[192]{};
    const int written = std::snprintf(
            base, sizeof(base), "model.language_model.layers.%u.mlp.%s_proj", layer, name);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(base)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return projection_load_base(
            model, device, base, rows, cols, handle, allocate_output, out);
}

int projection_load_fused_gate_up(
        axiom_model *model,
        int device,
        uint32_t layer,
        cublasLtHandle_t handle,
        Projection *out) {
    char gate_base[192]{};
    char up_base[192]{};
    const int gate_written = std::snprintf(
            gate_base, sizeof(gate_base),
            "model.language_model.layers.%u.mlp.gate_proj", layer);
    const int up_written = std::snprintf(
            up_base, sizeof(up_base),
            "model.language_model.layers.%u.mlp.up_proj", layer);
    if (gate_written <= 0 || up_written <= 0 ||
        static_cast<size_t>(gate_written) >= sizeof(gate_base) ||
        static_cast<size_t>(up_written) >= sizeof(up_base)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return projection_load_fused_base(
            model, device, gate_base, up_base,
            AXIOM_QWEN38_NVFP4_FFN, AXIOM_QWEN38_NVFP4_HIDDEN,
            handle, out);
}

int projection_matmul(
        Projection *projection,
        cublasLtHandle_t handle,
        void *workspace,
        float *output,
        uint32_t columns,
        cudaStream_t stream) {
    if (!projection || !handle || !workspace || !output || columns == 0u ||
        columns > AXIOM_QWEN38_NVFP4_TC_BATCH) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const MatmulPlan &plan = projection->plans[columns - 1u];
    if (!plan.b_layout || !plan.c_layout || !plan.d_layout) {
        return AXIOM_ERR_RUNTIME;
    }
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
            handle, projection->desc, &projection->alpha,
            projection->weight, projection->a_layout,
            projection->input_packed, plan.b_layout,
            &beta, output, plan.c_layout,
            output, plan.d_layout,
            &plan.algo.algo, workspace, plan.algo.workspaceSize, stream);
    return cublas_to_axiom(status);
}

int projection_run(
        Projection *projection,
        cublasLtHandle_t handle,
        void *workspace,
        const float *input,
        float *output,
        uint32_t columns,
        cudaStream_t stream) {
    if (!projection || !handle || !workspace || !input || !output ||
        columns == 0u || columns > AXIOM_QWEN38_NVFP4_TC_BATCH) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    constexpr uint32_t threads = 128u;
    const uint32_t inner_tiles = (projection->groups + 3u) / 4u;
    if (nvfp4_warp_quant_enabled()) {
        constexpr uint32_t warp_threads = 256u;
        constexpr uint32_t groups_per_block = warp_threads / kGroup;
        const uint32_t total_groups = projection->groups * columns;
        quantize_batch_to_tc_warp_kernel<<<
                (total_groups + groups_per_block - 1u) / groups_per_block,
                warp_threads, 0, stream>>>(
                input, projection->input_global, projection->input_packed,
                projection->input_scale_tc, projection->cols, projection->groups,
                inner_tiles, columns);
    } else {
        const uint32_t grid_x = (projection->groups + threads - 1u) / threads;
        quantize_batch_to_tc_kernel<<<dim3(grid_x, columns), threads, 0, stream>>>(
                input, projection->input_global, projection->input_packed,
                projection->input_scale_tc, projection->cols, projection->groups,
                inner_tiles, columns);
    }
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    return projection_matmul(
            projection, handle, workspace, output, columns, stream);
}

}  // namespace

struct axiom_qwen38_nvfp4_mlp {
    int device = -1;
    uint32_t layer = 0;
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    Projection gate_up{};
    Projection gate{};
    Projection up{};
    Projection down{};
    float *mid = nullptr;
    bool fused_gate_up = false;
    uint64_t device_bytes = 0;
};

struct axiom_qwen38_nvfp4_linear {
    int device = -1;
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    Projection projection{};
    uint64_t device_bytes = 0u;
};

extern "C" int axiom_qwen38_nvfp4_mlp_load(
        axiom_model *model,
        int device,
        uint32_t layer,
        axiom_qwen38_nvfp4_mlp **out) {
    if (out) *out = nullptr;
    if (!model || !out || device < 0 || layer >= AXIOM_QWEN38_NVFP4_MODEL_LAYERS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_nvfp4_mlp *mlp = nullptr;
    try { mlp = new axiom_qwen38_nvfp4_mlp(); } catch (...) { return AXIOM_ERR_BUDGET; }
    mlp->device = device;
    mlp->layer = layer;
    cublasStatus_t lt_status = cublasLtCreate(&mlp->handle);
    cudaError_t cuda_status = cudaSuccess;
    if (lt_status == CUBLAS_STATUS_SUCCESS) cuda_status = cudaMalloc(&mlp->workspace, kLtWorkspaceBytes);
    int rc = lt_status == CUBLAS_STATUS_SUCCESS
            ? (cuda_status == cudaSuccess ? AXIOM_OK : (cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA))
            : AXIOM_ERR_RUNTIME;
    if (rc == AXIOM_OK) {
        rc = projection_load_fused_gate_up(
                model, device, layer, mlp->handle, &mlp->gate_up);
        if (rc == AXIOM_OK) {
            mlp->fused_gate_up = true;
        } else if (rc == AXIOM_ERR_NOT_IMPLEMENTED) {
            rc = projection_load(
                    model, device, layer, "gate",
                    AXIOM_QWEN38_NVFP4_FFN, AXIOM_QWEN38_NVFP4_HIDDEN,
                    mlp->handle, true, &mlp->gate);
            if (rc == AXIOM_OK) {
                rc = projection_load(
                        model, device, layer, "up",
                        AXIOM_QWEN38_NVFP4_FFN, AXIOM_QWEN38_NVFP4_HIDDEN,
                        mlp->handle, true, &mlp->up);
            }
        }
    }
    if (rc == AXIOM_OK) {
        rc = projection_load(
                model, device, layer, "down",
                AXIOM_QWEN38_NVFP4_HIDDEN, AXIOM_QWEN38_NVFP4_FFN,
                mlp->handle, false, &mlp->down);
    }
    if (rc == AXIOM_OK && !mlp->fused_gate_up) {
        cuda_status = cudaMalloc(&mlp->mid,
                static_cast<size_t>(AXIOM_QWEN38_NVFP4_FFN) * AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float));
        if (cuda_status != cudaSuccess) rc = cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_nvfp4_mlp_destroy(mlp);
        return rc;
    }
    mlp->device_bytes = static_cast<uint64_t>(kLtWorkspaceBytes) +
            mlp->gate_up.device_bytes + mlp->gate.device_bytes +
            mlp->up.device_bytes + mlp->down.device_bytes +
            (mlp->mid ? static_cast<uint64_t>(AXIOM_QWEN38_NVFP4_FFN) *
                    AXIOM_QWEN38_NVFP4_TC_BATCH * sizeof(float) : 0u);
    *out = mlp;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_nvfp4_mlp_destroy(axiom_qwen38_nvfp4_mlp *mlp) {
    if (!mlp) return;
    if (mlp->device >= 0) (void)cudaSetDevice(mlp->device);
    if (mlp->mid) (void)cudaFree(mlp->mid);
    projection_destroy(&mlp->down);
    projection_destroy(&mlp->up);
    projection_destroy(&mlp->gate);
    projection_destroy(&mlp->gate_up);
    if (mlp->workspace) (void)cudaFree(mlp->workspace);
    if (mlp->handle) (void)cublasLtDestroy(mlp->handle);
    delete mlp;
}

extern "C" uint64_t axiom_qwen38_nvfp4_mlp_device_bytes(const axiom_qwen38_nvfp4_mlp *mlp) {
    return mlp ? mlp->device_bytes : 0u;
}

extern "C" uint32_t axiom_qwen38_nvfp4_mlp_uses_fused_gate_up(
        const axiom_qwen38_nvfp4_mlp *mlp) {
    return mlp && mlp->fused_gate_up ? 1u : 0u;
}

extern "C" int axiom_qwen38_nvfp4_mlp_forward_f32_device(
        axiom_qwen38_nvfp4_mlp *mlp,
        const float *input,
        float *out,
        void *stream) {
    return axiom_qwen38_nvfp4_mlp_forward_f32_device_m(
            mlp, input, out, AXIOM_QWEN38_NVFP4_TC_BATCH, stream);
}

extern "C" int axiom_qwen38_nvfp4_mlp_forward_f32_device_m(
        axiom_qwen38_nvfp4_mlp *mlp,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream) {
    if (!mlp || !input || !out || !mlp->handle || !mlp->workspace ||
        columns == 0u || columns > AXIOM_QWEN38_NVFP4_TC_BATCH ||
        (mlp->fused_gate_up ? !mlp->gate_up.output
                            : (!mlp->gate.output || !mlp->up.output || !mlp->mid))) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(mlp->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (mlp->fused_gate_up) {
        int rc = projection_run(
                &mlp->gate_up, mlp->handle, mlp->workspace,
                input, mlp->gate_up.output, columns, cuda_stream);
        if (rc != AXIOM_OK) return rc;
        constexpr uint32_t threads = 128u;
        const uint32_t inner_tiles = (mlp->down.groups + 3u) / 4u;
        if (nvfp4_warp_quant_enabled()) {
            constexpr uint32_t warp_threads = 256u;
            constexpr uint32_t groups_per_block = warp_threads / kGroup;
            const uint32_t total_groups = mlp->down.groups * columns;
            silu_mul_quantize_batch_to_tc_warp_kernel<<<
                    (total_groups + groups_per_block - 1u) / groups_per_block,
                    warp_threads, 0, cuda_stream>>>(
                    mlp->gate_up.output, mlp->down.input_global,
                    mlp->down.input_packed, mlp->down.input_scale_tc,
                    AXIOM_QWEN38_NVFP4_FFN, mlp->down.groups,
                    inner_tiles, columns);
        } else {
            const uint32_t grid_x = (mlp->down.groups + threads - 1u) / threads;
            silu_mul_quantize_batch_to_tc_kernel<<<
                    dim3(grid_x, columns), threads, 0, cuda_stream>>>(
                    mlp->gate_up.output, mlp->down.input_global,
                    mlp->down.input_packed, mlp->down.input_scale_tc,
                    AXIOM_QWEN38_NVFP4_FFN, mlp->down.groups,
                    inner_tiles, columns);
        }
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        return projection_matmul(
                &mlp->down, mlp->handle, mlp->workspace,
                out, columns, cuda_stream);
    }

    int rc = projection_run(
            &mlp->gate, mlp->handle, mlp->workspace,
            input, mlp->gate.output, columns, cuda_stream);
    if (rc == AXIOM_OK) {
        rc = projection_run(
                &mlp->up, mlp->handle, mlp->workspace,
                input, mlp->up.output, columns, cuda_stream);
    }
    if (rc != AXIOM_OK) return rc;
    constexpr uint32_t threads = 256u;
    const uint32_t count = AXIOM_QWEN38_NVFP4_FFN * columns;
    silu_mul_batch_kernel<<<
            (count + threads - 1u) / threads, threads, 0, cuda_stream>>>(
            mlp->gate.output, mlp->up.output, mlp->mid, count);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    return projection_run(
            &mlp->down, mlp->handle, mlp->workspace,
            mlp->mid, out, columns, cuda_stream);
}

extern "C" int axiom_qwen38_nvfp4_linear_load(
        axiom_model *model,
        int device,
        const char *base,
        uint32_t rows,
        uint32_t cols,
        axiom_qwen38_nvfp4_linear **out) {
    if (out) *out = nullptr;
    if (!model || !base || !base[0] || !out || device < 0 || rows == 0u || cols == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_nvfp4_linear *linear = nullptr;
    try { linear = new axiom_qwen38_nvfp4_linear(); } catch (...) { return AXIOM_ERR_BUDGET; }
    linear->device = device;
    cublasStatus_t lt_status = cublasLtCreate(&linear->handle);
    cudaError_t cuda_status = cudaSuccess;
    if (lt_status == CUBLAS_STATUS_SUCCESS) {
        cuda_status = cudaMalloc(&linear->workspace, kLtWorkspaceBytes);
    }
    int rc = lt_status == CUBLAS_STATUS_SUCCESS
            ? (cuda_status == cudaSuccess ? AXIOM_OK
                                          : (cuda_status == cudaErrorMemoryAllocation
                                                     ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA))
            : AXIOM_ERR_RUNTIME;
    if (rc == AXIOM_OK) {
        rc = projection_load_base(
                model, device, base, rows, cols, linear->handle,
                false, &linear->projection);
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_nvfp4_linear_destroy(linear);
        return rc;
    }
    linear->device_bytes = static_cast<uint64_t>(kLtWorkspaceBytes) +
            linear->projection.device_bytes;
    *out = linear;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_nvfp4_linear_destroy(
        axiom_qwen38_nvfp4_linear *linear) {
    if (!linear) return;
    if (linear->device >= 0) (void)cudaSetDevice(linear->device);
    projection_destroy(&linear->projection);
    if (linear->workspace) (void)cudaFree(linear->workspace);
    if (linear->handle) (void)cublasLtDestroy(linear->handle);
    delete linear;
}

extern "C" uint32_t axiom_qwen38_nvfp4_linear_rows(
        const axiom_qwen38_nvfp4_linear *linear) {
    return linear ? linear->projection.rows : 0u;
}

extern "C" uint32_t axiom_qwen38_nvfp4_linear_cols(
        const axiom_qwen38_nvfp4_linear *linear) {
    return linear ? linear->projection.cols : 0u;
}

extern "C" uint64_t axiom_qwen38_nvfp4_linear_device_bytes(
        const axiom_qwen38_nvfp4_linear *linear) {
    return linear ? linear->device_bytes : 0u;
}

extern "C" int axiom_qwen38_nvfp4_linear_forward_f32_device(
        axiom_qwen38_nvfp4_linear *linear,
        const float *input,
        float *out,
        void *stream) {
    return axiom_qwen38_nvfp4_linear_forward_f32_device_m(
            linear, input, out, AXIOM_QWEN38_NVFP4_TC_BATCH, stream);
}

extern "C" int axiom_qwen38_nvfp4_linear_forward_f32_device_m(
        axiom_qwen38_nvfp4_linear *linear,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream) {
    if (!linear || !input || !out || !linear->handle || !linear->workspace ||
        columns == 0u || columns > AXIOM_QWEN38_NVFP4_TC_BATCH) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    return projection_run(
            &linear->projection, linear->handle, linear->workspace,
            input, out, columns, cuda_stream);
}
