/*
 * Native Qwen3.8 FP8 linear: checkpoint E4M3 weights + BF16 row scales,
 * dynamic per-column E4M3 activation quantization, fixed batch 8.
 *
 * cuBLASLt does not expose the checkpoint's BF16-per-output-row scale as a
 * direct matrix scale mode.  Scaling is separable from the dot product, so the
 * serving path executes a unit-scaled raw FP8 tensor-core matmul and applies
 * weight_scale[row] * input_scale[column] to its F32 result in one epilogue.
 * A CUDA reference dot product is retained for unsupported Lt shapes and for
 * parity gates.  Neither path converts or rewrites the model weight plane.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
#include <new>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_fp8.h"

namespace {

constexpr uint32_t kThreads = 256u;
constexpr float kE4m3MaxFinite = 448.0f;
constexpr size_t kWorkspaceBytes = 8u * 1024u * 1024u;
constexpr uint64_t kUploadChunkBytes = 64ull * 1024ull * 1024ull;

__device__ __forceinline__ float decode_e4m3fn(uint8_t code) {
    const uint32_t magnitude = code & 0x7fu;
    if (magnitude == 0x7fu) return nanf("");
    const uint32_t exponent = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    const float value = exponent == 0u
            ? ldexpf(static_cast<float>(mantissa), -9)
            : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                     static_cast<int>(exponent) - 7);
    return (code & 0x80u) ? -value : value;
}

__host__ __device__ __forceinline__ float decode_bf16(uint16_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
#else
    const uint32_t word = static_cast<uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &word, sizeof(value));
    return value;
#endif
}

__global__ void quantize_columns_e4m3_kernel(
        const float *__restrict__ input,
        uint8_t *__restrict__ quantized,
        float *__restrict__ scales,
        uint32_t cols) {
    const uint32_t column = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    if (column >= AXIOM_QWEN38_FP8_BATCH) return;
    const float *in = input + static_cast<uint64_t>(column) * cols;
    float local_max = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(in[c]));
    }
    __shared__ float reduction[kThreads];
    reduction[thread] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] = fmaxf(reduction[thread], reduction[thread + stride]);
        __syncthreads();
    }
    float scale = reduction[0] == 0.0f ? 1.0f : reduction[0] / kE4m3MaxFinite;
    if (!isfinite(scale) || scale <= 0.0f) scale = 1.0f;
    if (thread == 0u) scales[column] = scale;
    __syncthreads();
    uint8_t *out = quantized + static_cast<uint64_t>(column) * cols;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        const __nv_fp8_storage_t value = __nv_cvt_float_to_fp8(
                in[c] / scale, __NV_SATFINITE, __NV_E4M3);
        out[c] = static_cast<uint8_t>(value);
    }
}

__global__ void quantize_columns_static_e4m3_kernel(
        const float *__restrict__ input,
        uint8_t *__restrict__ quantized,
        float scale,
        uint32_t cols) {
    const uint32_t column = blockIdx.y;
    const uint32_t feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= AXIOM_QWEN38_FP8_BATCH || feature >= cols) return;
    const uint64_t index = static_cast<uint64_t>(column) * cols + feature;
    quantized[index] = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
            input[index] / scale, __NV_SATFINITE, __NV_E4M3));
}

__global__ void fp8_row_scaled_reference_kernel(
        const uint8_t *__restrict__ weight,
        const uint16_t *__restrict__ weight_scale,
        const uint8_t *__restrict__ input,
        const float *__restrict__ input_scale,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_FP8_BATCH) return;
    const uint8_t *w = weight + static_cast<uint64_t>(row) * cols;
    const uint8_t *x = input + static_cast<uint64_t>(column) * cols;
    float partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        partial += decode_e4m3fn(w[c]) * decode_e4m3fn(x[c]);
    }
    __shared__ float reduction[kThreads];
    reduction[thread] = partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    if (thread == 0u) {
        out[static_cast<uint64_t>(column) * rows + row] =
                reduction[0] * decode_bf16(weight_scale[row]) * input_scale[column];
    }
}

__global__ void fp8_row_column_scale_epilogue_kernel(
        float *out,
        const uint16_t *weight_scale,
        const float *input_scale,
        uint32_t rows) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * AXIOM_QWEN38_FP8_BATCH;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index % rows);
    const uint32_t column = static_cast<uint32_t>(index / rows);
    out[index] *= decode_bf16(weight_scale[row]) * input_scale[column];
}

__global__ void fp8_scalar_scaled_reference_kernel(
        const uint8_t *__restrict__ weight,
        const uint8_t *__restrict__ input,
        float weight_scale,
        float input_scale,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_FP8_BATCH) return;
    const uint8_t *w = weight + static_cast<uint64_t>(row) * cols;
    const uint8_t *x = input + static_cast<uint64_t>(column) * cols;
    float partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        partial += decode_e4m3fn(w[c]) * decode_e4m3fn(x[c]);
    }
    __shared__ float reduction[kThreads];
    reduction[thread] = partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    if (thread == 0u) {
        out[static_cast<uint64_t>(column) * rows + row] =
                reduction[0] * weight_scale * input_scale;
    }
}

__device__ __forceinline__ float round_bf16_f32(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void fp8_group_row_scaled_reference_kernel(
        const uint8_t *__restrict__ weight,
        const uint16_t *__restrict__ weight_scale,
        const uint8_t *__restrict__ input,
        const float *__restrict__ input_scale,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_FP8_BATCH) return;
    const uint8_t *w = weight + static_cast<uint64_t>(row) * cols;
    const uint8_t *x = input + static_cast<uint64_t>(column) * cols;
    float partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        partial += decode_e4m3fn(w[c]) * decode_e4m3fn(x[c]);
    }
    __shared__ float reduction[kThreads];
    reduction[thread] = partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    if (thread == 0u) {
        out[static_cast<uint64_t>(column) * rows + row] = round_bf16_f32(
                reduction[0] * decode_bf16(weight_scale[row]) * input_scale[column]);
    }
}

__global__ void fp8_group_scalar_scaled_reference_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ row_weight_scale,
        const uint8_t *__restrict__ input,
        float input_scale,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_FP8_BATCH) return;
    const uint8_t *w = weight + static_cast<uint64_t>(row) * cols;
    const uint8_t *x = input + static_cast<uint64_t>(column) * cols;
    float partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        partial += decode_e4m3fn(w[c]) * decode_e4m3fn(x[c]);
    }
    __shared__ float reduction[kThreads];
    reduction[thread] = partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    if (thread == 0u) {
        out[static_cast<uint64_t>(column) * rows + row] = round_bf16_f32(
                reduction[0] * row_weight_scale[row] * input_scale);
    }
}

__global__ void fp8_group_row_scale_bf16_epilogue_kernel(
        float *out,
        const uint16_t *weight_scale,
        const float *input_scale,
        uint32_t rows) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * AXIOM_QWEN38_FP8_BATCH;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index % rows);
    const uint32_t column = static_cast<uint32_t>(index / rows);
    out[index] = round_bf16_f32(
            out[index] * decode_bf16(weight_scale[row]) * input_scale[column]);
}

__global__ void fp8_group_scalar_scale_bf16_epilogue_kernel(
        float *out,
        const float *row_weight_scale,
        uint32_t rows) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * AXIOM_QWEN38_FP8_BATCH;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index % rows);
    out[index] = round_bf16_f32(out[index] * row_weight_scale[row]);
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool valid_bf16_scales(const std::vector<uint16_t> &scales) {
    for (const uint16_t bits : scales) {
        const float value = decode_bf16(bits);
        if (!std::isfinite(value) || value <= 0.0f) return false;
    }
    return true;
}

bool valid_positive_f32(float value) {
    return std::isfinite(value) && value > 0.0f;
}

bool exact_f32_scalar(const axiom_tensor_info &info) {
    return info.dtype == AXIOM_TENSOR_DTYPE_F32 && info.rank == 0u &&
            info.byte_count == sizeof(float);
}

int upload_tensor_chunked(
        axiom_model *model,
        const char *name,
        uint8_t *device_dst,
        uint64_t bytes) {
    if (!model || !name || !device_dst || bytes == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t capacity = bytes < kUploadChunkBytes ? bytes : kUploadChunkBytes;
    std::vector<uint8_t> chunk;
    try {
        chunk.resize(static_cast<size_t>(capacity));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const uint64_t count = remaining < capacity ? remaining : capacity;
        int rc = axiom_model_tensor_read_slice(model, name, offset, chunk.data(), count);
        if (rc != AXIOM_OK) return rc;
        const cudaError_t status = cudaMemcpy(
                device_dst + offset, chunk.data(), static_cast<size_t>(count), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
        offset += count;
    }
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_fp8_linear {
    int device = -1;
    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint8_t *weight = nullptr;
    uint16_t *weight_scale = nullptr;
    float *modelopt_weight_scale = nullptr;
    uint8_t *input_quantized = nullptr;
    float *input_scale = nullptr;
    float modelopt_weight_scale_host = 0.0f;
    float modelopt_input_scale_host = 0.0f;
    float *unit_scale = nullptr;
    void *workspace = nullptr;
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool modelopt_scalar = false;
    bool tensor_core = false;
    uint64_t device_bytes = 0u;
};

struct axiom_qwen38_fp8_projection_group {
    int device = -1;
    uint32_t projection_count = 0u;
    uint32_t cols = 0u;
    uint32_t rows = 0u;
    uint32_t row_offsets[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX + 1u]{};
    uint8_t *weight = nullptr;
    uint16_t *legacy_weight_scale = nullptr;
    float *modelopt_row_weight_scale = nullptr;
    uint8_t *input_quantized = nullptr;
    float *input_scale = nullptr;
    float input_scale_host = 0.0f;
    float *unit_scale = nullptr;
    void *workspace = nullptr;
    cublasLtHandle_t lt = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    cublasLtMatrixLayout_t d_layout = nullptr;
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool modelopt_scalar = false;
    bool tensor_core = false;
    uint64_t device_bytes = 0u;
};

namespace {

__global__ void fp8_exact_output_compare_kernel(
        const float *__restrict__ reference,
        const float *__restrict__ candidate,
        uint64_t count,
        uint32_t *__restrict__ mismatch) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count && __float_as_uint(reference[index]) != __float_as_uint(candidate[index])) {
        atomicExch(mismatch, 1u);
    }
}

__global__ void fp8_autotune_input_pattern_kernel(
        uint8_t *__restrict__ values,
        uint64_t count,
        uint32_t seed) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t mixed = static_cast<uint32_t>(index) * 747796405u + seed * 2891336453u;
    mixed = ((mixed >> ((mixed >> 28u) + 4u)) ^ mixed) * 277803737u;
    mixed = (mixed >> 22u) ^ mixed;
    const uint8_t magnitude = static_cast<uint8_t>(0x18u + mixed % 0x40u);
    values[index] = static_cast<uint8_t>(magnitude | ((mixed >> 8u) & 0x80u));
}

bool fp8_autotune_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_FP8_AUTOTUNE");
    if (!value || value[0] == '\0') {
        value = std::getenv("AXIOM_QWEN38_MATMUL_AUTOTUNE");
    }
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0;
}

struct Fp8AlgoKey {
    uint32_t rows = 0u;
    uint32_t cols = 0u;

    bool operator<(const Fp8AlgoKey &other) const {
        return rows != other.rows ? rows < other.rows : cols < other.cols;
    }
};

struct Fp8AlgoValue {
    cublasLtMatmulHeuristicResult_t heuristic{};
    uint32_t candidate_index = 0u;
    uint32_t candidate_count = 0u;
    float milliseconds = 0.0f;
};

std::atomic_flag g_fp8_algo_lock = ATOMIC_FLAG_INIT;
std::map<Fp8AlgoKey, Fp8AlgoValue> g_fp8_algo_cache;

struct Fp8AlgoGuard {
    Fp8AlgoGuard() {
        while (g_fp8_algo_lock.test_and_set(std::memory_order_acquire)) {}
    }
    ~Fp8AlgoGuard() {
        g_fp8_algo_lock.clear(std::memory_order_release);
    }
};

bool fp8_candidate_is_exact(
        cublasLtHandle_t handle,
        cublasLtMatmulDesc_t desc,
        cublasLtMatrixLayout_t a_layout,
        cublasLtMatrixLayout_t b_layout,
        cublasLtMatrixLayout_t c_layout,
        cublasLtMatrixLayout_t d_layout,
        const uint8_t *weight,
        uint8_t *input,
        float *reference,
        float *candidate_output,
        uint32_t *mismatch_device,
        void *workspace,
        size_t input_bytes,
        uint64_t output_elements,
        const cublasLtMatmulHeuristicResult_t &reference_candidate,
        const cublasLtMatmulHeuristicResult_t &candidate,
        cudaStream_t stream) {
    if (!handle || !desc || !a_layout || !b_layout || !c_layout || !d_layout ||
        !weight || !input || !reference || !candidate_output || !mismatch_device ||
        !workspace || !stream || input_bytes == 0u || output_elements == 0u) {
        return false;
    }
    constexpr uint32_t kValidationPatterns = 3u;
    constexpr uint32_t kPatternSeeds[kValidationPatterns] = {
            0x243f6a88u, 0x9e3779b9u, 0xb7e15162u};
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const uint64_t input_grid = (input_bytes + kThreads - 1u) / kThreads;
    const uint64_t output_grid = (output_elements + kThreads - 1u) / kThreads;
    if (input_grid > std::numeric_limits<uint32_t>::max() ||
        output_grid > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    for (uint32_t pattern = 0u; pattern < kValidationPatterns; ++pattern) {
        fp8_autotune_input_pattern_kernel<<<
                static_cast<uint32_t>(input_grid), kThreads, 0, stream>>>(
                input, input_bytes, kPatternSeeds[pattern]);
        if (cudaGetLastError() != cudaSuccess) return false;
        cublasStatus_t status = cublasLtMatmul(
                handle, desc, &alpha, weight, a_layout, input, b_layout,
                &beta, reference, c_layout, reference, d_layout,
                &reference_candidate.algo, workspace,
                reference_candidate.workspaceSize, stream);
        if (status != CUBLAS_STATUS_SUCCESS) return false;
        status = cublasLtMatmul(
                handle, desc, &alpha, weight, a_layout, input, b_layout,
                &beta, candidate_output, c_layout, candidate_output, d_layout,
                &candidate.algo, workspace, candidate.workspaceSize, stream);
        if (status != CUBLAS_STATUS_SUCCESS) return false;
        uint32_t mismatch = 1u;
        cudaError_t compare_status = cudaMemsetAsync(
                mismatch_device, 0, sizeof(uint32_t), stream);
        if (compare_status == cudaSuccess) {
            fp8_exact_output_compare_kernel<<<
                    static_cast<uint32_t>(output_grid), kThreads, 0, stream>>>(
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

cublasStatus_t select_fp8_algorithm(
        cublasLtHandle_t handle,
        cublasLtMatmulDesc_t desc,
        cublasLtMatrixLayout_t a_layout,
        cublasLtMatrixLayout_t b_layout,
        cublasLtMatrixLayout_t c_layout,
        cublasLtMatrixLayout_t d_layout,
        cublasLtMatmulPreference_t preference,
        const uint8_t *weight,
        uint8_t *input,
        void *workspace,
        uint32_t rows,
        uint32_t cols,
        cublasLtMatmulHeuristicResult_t *out) {
    if (!handle || !desc || !a_layout || !b_layout || !c_layout || !d_layout ||
        !preference || !weight || !input || !workspace || !out || rows == 0u || cols == 0u) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool tune = fp8_autotune_enabled();
    const Fp8AlgoKey key{rows, cols};
    if (tune) {
        Fp8AlgoGuard guard;
        const auto cached = g_fp8_algo_cache.find(key);
        if (cached != g_fp8_algo_cache.end()) {
            *out = cached->second.heuristic;
            return CUBLAS_STATUS_SUCCESS;
        }
    }

    constexpr int kMaxCandidates = 32;
    cublasLtMatmulHeuristicResult_t candidates[kMaxCandidates]{};
    int returned = 0;
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            handle, desc, a_layout, b_layout, c_layout, d_layout,
            preference, tune ? kMaxCandidates : 1, candidates, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned < 1) {
        return status == CUBLAS_STATUS_SUCCESS
                ? CUBLAS_STATUS_NOT_SUPPORTED : status;
    }
    *out = candidates[0];
    if (!tune || returned == 1) {
        if (tune) {
            {
                Fp8AlgoGuard guard;
                g_fp8_algo_cache.emplace(
                        key, Fp8AlgoValue{candidates[0], 0u,
                                          static_cast<uint32_t>(returned), 0.0f});
            }
            std::fprintf(stderr,
                         "axiom-qwen38-fp8: autotune rows=%u cols=%u candidates=%d "
                         "selected=0 measured_ms=0\n",
                         rows, cols, returned);
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    float *scratch = nullptr;
    float *reference = nullptr;
    uint32_t *mismatch_device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    const size_t input_bytes = static_cast<size_t>(cols) * AXIOM_QWEN38_FP8_BATCH;
    const size_t output_bytes = static_cast<size_t>(rows) *
            AXIOM_QWEN38_FP8_BATCH * sizeof(float);
    const uint64_t output_elements =
            static_cast<uint64_t>(rows) * AXIOM_QWEN38_FP8_BATCH;
    cudaError_t cuda_status = cudaMalloc(&scratch, output_bytes);
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
    const float alpha = 1.0f;
    const float beta = 0.0f;
    uint32_t best_index = 0u;
    uint32_t exact_candidates = 0u;
    float best_ms = std::numeric_limits<float>::infinity();
    float baseline_ms = std::numeric_limits<float>::infinity();
    if (cuda_status == cudaSuccess) {
        for (int candidate = 0; candidate < returned; ++candidate) {
            if (candidates[candidate].state != CUBLAS_STATUS_SUCCESS ||
                candidates[candidate].workspaceSize > kWorkspaceBytes) {
                continue;
            }
            const bool exact = candidate == 0 || fp8_candidate_is_exact(
                    handle, desc, a_layout, b_layout, c_layout, d_layout,
                    weight, input, reference, scratch, mismatch_device, workspace,
                    input_bytes, output_elements, candidates[0], candidates[candidate], stream);
            if (!exact) {
                (void)cudaGetLastError();
                continue;
            }
            ++exact_candidates;
            const uint64_t input_grid = (input_bytes + kThreads - 1u) / kThreads;
            fp8_autotune_input_pattern_kernel<<<
                    static_cast<uint32_t>(input_grid), kThreads, 0, stream>>>(
                    input, input_bytes, 0xd1b54a35u);
            if (cudaGetLastError() != cudaSuccess) continue;
            cublasStatus_t run_status = CUBLAS_STATUS_SUCCESS;
            for (uint32_t run = 0u; run < kWarmupRuns &&
                 run_status == CUBLAS_STATUS_SUCCESS; ++run) {
                run_status = cublasLtMatmul(
                        handle, desc, &alpha, weight, a_layout, input, b_layout,
                        &beta, scratch, c_layout, scratch, d_layout,
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
                        handle, desc, &alpha, weight, a_layout, input, b_layout,
                        &beta, scratch, c_layout, scratch, d_layout,
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
    if (scratch) (void)cudaFree(scratch);
    if (!std::isfinite(best_ms)) {
        best_index = 0u;
        best_ms = 0.0f;
    } else if (best_index != 0u &&
               (!std::isfinite(baseline_ms) || best_ms * kMinimumSpeedup >= baseline_ms)) {
        best_index = 0u;
        best_ms = baseline_ms;
    }
    *out = candidates[best_index];
    {
        Fp8AlgoGuard guard;
        g_fp8_algo_cache.emplace(
                key, Fp8AlgoValue{candidates[best_index], best_index,
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
                 "axiom-qwen38-fp8: autotune rows=%u cols=%u candidates=%d "
                 "exact=%u selected=%u algo=%d split_k=%d baseline_ms=%.6f measured_ms=%.6f "
                 "workspace=%zu\n",
                 rows, cols, returned, exact_candidates, best_index, algo_id, split_k,
                 static_cast<double>(baseline_ms), static_cast<double>(best_ms),
                 candidates[best_index].workspaceSize);
    return CUBLAS_STATUS_SUCCESS;
}

void destroy_lt(axiom_qwen38_fp8_linear *linear) {
    if (!linear) return;
    if (linear->d_layout) (void)cublasLtMatrixLayoutDestroy(linear->d_layout);
    if (linear->c_layout) (void)cublasLtMatrixLayoutDestroy(linear->c_layout);
    if (linear->b_layout) (void)cublasLtMatrixLayoutDestroy(linear->b_layout);
    if (linear->a_layout) (void)cublasLtMatrixLayoutDestroy(linear->a_layout);
    if (linear->desc) (void)cublasLtMatmulDescDestroy(linear->desc);
    if (linear->lt) (void)cublasLtDestroy(linear->lt);
    linear->d_layout = nullptr;
    linear->c_layout = nullptr;
    linear->b_layout = nullptr;
    linear->a_layout = nullptr;
    linear->desc = nullptr;
    linear->lt = nullptr;
    linear->tensor_core = false;
}

bool configure_tensor_core(axiom_qwen38_fp8_linear *linear) {
    if (!linear || !linear->weight || !linear->input_quantized || !linear->unit_scale ||
        !linear->workspace ||
        (linear->modelopt_scalar &&
         (!linear->modelopt_weight_scale || !linear->input_scale))) return false;
    cublasStatus_t status = cublasLtCreate(&linear->lt);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescCreate(&linear->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    }
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    float *a_scale = linear->modelopt_scalar
            ? linear->modelopt_weight_scale : linear->unit_scale;
    float *b_scale = linear->modelopt_scalar
            ? linear->input_scale : linear->unit_scale;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scalar, sizeof(scalar));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scalar, sizeof(scalar));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale, sizeof(a_scale));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            linear->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale, sizeof(b_scale));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &linear->a_layout, CUDA_R_8F_E4M3, linear->cols, linear->rows, linear->cols);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &linear->b_layout, CUDA_R_8F_E4M3, linear->cols,
            AXIOM_QWEN38_FP8_BATCH, linear->cols);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &linear->c_layout, CUDA_R_32F, linear->rows,
            AXIOM_QWEN38_FP8_BATCH, linear->rows);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &linear->d_layout, CUDA_R_32F, linear->rows,
            AXIOM_QWEN38_FP8_BATCH, linear->rows);
    cublasLtMatmulPreference_t preference = nullptr;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceCreate(&preference);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes));
    if (status == CUBLAS_STATUS_SUCCESS) status = select_fp8_algorithm(
            linear->lt, linear->desc,
            linear->a_layout, linear->b_layout, linear->c_layout, linear->d_layout,
            preference, linear->weight, linear->input_quantized, linear->workspace,
            linear->rows, linear->cols, &linear->heuristic);
    if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
        destroy_lt(linear);
        return false;
    }
    linear->tensor_core = true;
    return true;
}

int quantize_input(
        axiom_qwen38_fp8_linear *linear,
        const float *input,
        cudaStream_t stream) {
    if (linear->modelopt_scalar) {
        const dim3 grid(
                (linear->cols + kThreads - 1u) / kThreads,
                AXIOM_QWEN38_FP8_BATCH);
        quantize_columns_static_e4m3_kernel<<<grid, kThreads, 0, stream>>>(
                input, linear->input_quantized,
                linear->modelopt_input_scale_host, linear->cols);
    } else {
        quantize_columns_e4m3_kernel<<<AXIOM_QWEN38_FP8_BATCH, kThreads, 0, stream>>>(
                input, linear->input_quantized, linear->input_scale, linear->cols);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_reference(
        axiom_qwen38_fp8_linear *linear,
        float *out,
        cudaStream_t stream) {
    if (linear->modelopt_scalar) {
        fp8_scalar_scaled_reference_kernel<<<
                dim3(linear->rows, AXIOM_QWEN38_FP8_BATCH), kThreads, 0, stream>>>(
                linear->weight, linear->input_quantized,
                linear->modelopt_weight_scale_host, linear->modelopt_input_scale_host,
                out, linear->rows, linear->cols);
    } else {
        fp8_row_scaled_reference_kernel<<<
                dim3(linear->rows, AXIOM_QWEN38_FP8_BATCH), kThreads, 0, stream>>>(
                linear->weight, linear->weight_scale,
                linear->input_quantized, linear->input_scale,
                out, linear->rows, linear->cols);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_reference_prepared(
        axiom_qwen38_fp8_linear *linear,
        const uint8_t *input_quantized,
        float *out,
        cudaStream_t stream) {
    if (!linear->modelopt_scalar) return AXIOM_ERR_INVALID_ARGUMENT;
    fp8_scalar_scaled_reference_kernel<<<
            dim3(linear->rows, AXIOM_QWEN38_FP8_BATCH), kThreads, 0, stream>>>(
            linear->weight, input_quantized,
            linear->modelopt_weight_scale_host, linear->modelopt_input_scale_host,
            out, linear->rows, linear->cols);
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_tensor_core(
        axiom_qwen38_fp8_linear *linear,
        float *out,
        cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
            linear->lt, linear->desc, &alpha,
            linear->weight, linear->a_layout,
            linear->input_quantized, linear->b_layout,
            &beta, out, linear->c_layout,
            out, linear->d_layout,
            &linear->heuristic.algo, linear->workspace,
            linear->heuristic.workspaceSize, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return AXIOM_ERR_RUNTIME;
    if (!linear->modelopt_scalar) {
        const uint64_t count = static_cast<uint64_t>(linear->rows) * AXIOM_QWEN38_FP8_BATCH;
        fp8_row_column_scale_epilogue_kernel<<<
                static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                out, linear->weight_scale, linear->input_scale, linear->rows);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_tensor_core_prepared(
        axiom_qwen38_fp8_linear *linear,
        const uint8_t *input_quantized,
        float *out,
        cudaStream_t stream) {
    if (!linear->modelopt_scalar) return AXIOM_ERR_INVALID_ARGUMENT;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
            linear->lt, linear->desc, &alpha,
            linear->weight, linear->a_layout,
            input_quantized, linear->b_layout,
            &beta, out, linear->c_layout,
            out, linear->d_layout,
            &linear->heuristic.algo, linear->workspace,
            linear->heuristic.workspaceSize, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return AXIOM_ERR_RUNTIME;
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

void destroy_group_lt(axiom_qwen38_fp8_projection_group *group) {
    if (!group) return;
    if (group->d_layout) (void)cublasLtMatrixLayoutDestroy(group->d_layout);
    if (group->c_layout) (void)cublasLtMatrixLayoutDestroy(group->c_layout);
    if (group->b_layout) (void)cublasLtMatrixLayoutDestroy(group->b_layout);
    if (group->a_layout) (void)cublasLtMatrixLayoutDestroy(group->a_layout);
    if (group->desc) (void)cublasLtMatmulDescDestroy(group->desc);
    if (group->lt) (void)cublasLtDestroy(group->lt);
    group->d_layout = nullptr;
    group->c_layout = nullptr;
    group->b_layout = nullptr;
    group->a_layout = nullptr;
    group->desc = nullptr;
    group->lt = nullptr;
    group->tensor_core = false;
}

bool configure_group_tensor_core(axiom_qwen38_fp8_projection_group *group) {
    if (!group || !group->weight || !group->input_quantized || !group->input_scale ||
        !group->unit_scale || !group->workspace ||
        (group->modelopt_scalar ? !group->modelopt_row_weight_scale
                                : !group->legacy_weight_scale)) {
        return false;
    }
    cublasStatus_t status = cublasLtCreate(&group->lt);
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasLtMatmulDescCreate(&group->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    }
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    const cublasLtMatmulMatrixScale_t scalar = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    float *a_scale = group->unit_scale;
    float *b_scale = group->modelopt_scalar ? group->input_scale : group->unit_scale;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scalar, sizeof(scalar));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scalar, sizeof(scalar));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &a_scale, sizeof(a_scale));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulDescSetAttribute(
            group->desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &b_scale, sizeof(b_scale));
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &group->a_layout, CUDA_R_8F_E4M3, group->cols, group->rows, group->cols);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &group->b_layout, CUDA_R_8F_E4M3, group->cols,
            AXIOM_QWEN38_FP8_BATCH, group->cols);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &group->c_layout, CUDA_R_32F, group->rows,
            AXIOM_QWEN38_FP8_BATCH, group->rows);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatrixLayoutCreate(
            &group->d_layout, CUDA_R_32F, group->rows,
            AXIOM_QWEN38_FP8_BATCH, group->rows);
    cublasLtMatmulPreference_t preference = nullptr;
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceCreate(&preference);
    if (status == CUBLAS_STATUS_SUCCESS) status = cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kWorkspaceBytes, sizeof(kWorkspaceBytes));
    if (status == CUBLAS_STATUS_SUCCESS) status = select_fp8_algorithm(
            group->lt, group->desc,
            group->a_layout, group->b_layout, group->c_layout, group->d_layout,
            preference, group->weight, group->input_quantized, group->workspace,
            group->rows, group->cols, &group->heuristic);
    if (preference) (void)cublasLtMatmulPreferenceDestroy(preference);
    if (status != CUBLAS_STATUS_SUCCESS) {
        destroy_group_lt(group);
        return false;
    }
    group->tensor_core = true;
    return true;
}

int quantize_group_input(
        axiom_qwen38_fp8_projection_group *group,
        const float *input,
        cudaStream_t stream) {
    if (group->modelopt_scalar) {
        const dim3 grid(
                (group->cols + kThreads - 1u) / kThreads,
                AXIOM_QWEN38_FP8_BATCH);
        quantize_columns_static_e4m3_kernel<<<grid, kThreads, 0, stream>>>(
                input, group->input_quantized, group->input_scale_host, group->cols);
    } else {
        quantize_columns_e4m3_kernel<<<AXIOM_QWEN38_FP8_BATCH, kThreads, 0, stream>>>(
                input, group->input_quantized, group->input_scale, group->cols);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_group_reference(
        axiom_qwen38_fp8_projection_group *group,
        float *out,
        cudaStream_t stream) {
    if (group->modelopt_scalar) {
        fp8_group_scalar_scaled_reference_kernel<<<
                dim3(group->rows, AXIOM_QWEN38_FP8_BATCH), kThreads, 0, stream>>>(
                group->weight, group->modelopt_row_weight_scale,
                group->input_quantized, group->input_scale_host,
                out, group->rows, group->cols);
    } else {
        fp8_group_row_scaled_reference_kernel<<<
                dim3(group->rows, AXIOM_QWEN38_FP8_BATCH), kThreads, 0, stream>>>(
                group->weight, group->legacy_weight_scale,
                group->input_quantized, group->input_scale,
                out, group->rows, group->cols);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_group_tensor_core(
        axiom_qwen38_fp8_projection_group *group,
        float *out,
        cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasLtMatmul(
            group->lt, group->desc, &alpha,
            group->weight, group->a_layout,
            group->input_quantized, group->b_layout,
            &beta, out, group->c_layout,
            out, group->d_layout,
            &group->heuristic.algo, group->workspace,
            group->heuristic.workspaceSize, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return AXIOM_ERR_RUNTIME;
    const uint64_t count = static_cast<uint64_t>(group->rows) * AXIOM_QWEN38_FP8_BATCH;
    const uint32_t blocks = static_cast<uint32_t>((count + kThreads - 1u) / kThreads);
    if (group->modelopt_scalar) {
        fp8_group_scalar_scale_bf16_epilogue_kernel<<<blocks, kThreads, 0, stream>>>(
                out, group->modelopt_row_weight_scale, group->rows);
    } else {
        fp8_group_row_scale_bf16_epilogue_kernel<<<blocks, kThreads, 0, stream>>>(
                out, group->legacy_weight_scale, group->input_scale, group->rows);
    }
    return cudaPeekAtLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

}  // namespace

extern "C" int axiom_qwen38_fp8_linear_load(
        axiom_model *model,
        int device,
        const char *weight_name,
        const char *scale_name,
        axiom_qwen38_fp8_linear **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !weight_name || !weight_name[0] ||
        !scale_name || !scale_name[0] || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_tensor_info weight_info{};
    weight_info.abi_version = AXIOM_ABI_VERSION;
    axiom_tensor_info scale_info{};
    scale_info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, weight_name, &weight_info);
    if (rc == AXIOM_OK) rc = axiom_model_tensor_info_get(model, scale_name, &scale_info);
    if (rc != AXIOM_OK) return rc;
    if (weight_info.dtype != AXIOM_TENSOR_DTYPE_F8_E4M3 || weight_info.rank != 2u ||
        weight_info.shape[0] == 0u || weight_info.shape[1] == 0u ||
        weight_info.shape[0] > std::numeric_limits<uint32_t>::max() ||
        weight_info.shape[1] > std::numeric_limits<uint32_t>::max() ||
        scale_info.dtype != AXIOM_TENSOR_DTYPE_BF16 || scale_info.rank != 2u ||
        scale_info.shape[0] != weight_info.shape[0] || scale_info.shape[1] != 1u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t weight_bytes = 0u;
    if (!checked_mul_u64(weight_info.shape[0], weight_info.shape[1], &weight_bytes) ||
        weight_info.byte_count != weight_bytes ||
        scale_info.byte_count != weight_info.shape[0] * sizeof(uint16_t)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t rows = static_cast<uint32_t>(weight_info.shape[0]);
    const uint32_t cols = static_cast<uint32_t>(weight_info.shape[1]);
    std::vector<uint16_t> host_scale;
    try {
        host_scale.resize(rows);
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    rc = axiom_model_tensor_read_slice(model, scale_name, 0u, host_scale.data(), scale_info.byte_count);
    if (rc != AXIOM_OK) return rc;
    if (!valid_bf16_scales(host_scale)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_qwen38_fp8_linear *linear = new (std::nothrow) axiom_qwen38_fp8_linear();
    if (!linear) return AXIOM_ERR_BUDGET;
    linear->device = device;
    linear->rows = rows;
    linear->cols = cols;
    const uint64_t input_bytes = static_cast<uint64_t>(cols) * AXIOM_QWEN38_FP8_BATCH;
    cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void **>(&linear->weight), static_cast<size_t>(weight_bytes));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->weight_scale), host_scale.size() * sizeof(uint16_t));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->input_quantized), static_cast<size_t>(input_bytes));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->input_scale), AXIOM_QWEN38_FP8_BATCH * sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->unit_scale), sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&linear->workspace, kWorkspaceBytes);
    if (cuda_status != cudaSuccess) {
        axiom_qwen38_fp8_linear_destroy(linear);
        return cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    rc = upload_tensor_chunked(model, weight_name, linear->weight, weight_bytes);
    const float one = 1.0f;
    if (rc == AXIOM_OK && cudaMemcpy(
            linear->weight_scale, host_scale.data(), host_scale.size() * sizeof(uint16_t),
            cudaMemcpyHostToDevice) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK && cudaMemcpy(
            linear->unit_scale, &one, sizeof(one), cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_fp8_linear_destroy(linear);
        return rc;
    }
    (void)configure_tensor_core(linear);  // A missing heuristic is a supported reference fallback.
    linear->device_bytes = weight_bytes + scale_info.byte_count + input_bytes +
            AXIOM_QWEN38_FP8_BATCH * sizeof(float) + sizeof(float) + kWorkspaceBytes;
    *out = linear;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_fp8_linear_load_modelopt(
        axiom_model *model,
        int device,
        const char *weight_name,
        const char *weight_scale_name,
        const char *input_scale_name,
        axiom_qwen38_fp8_linear **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !weight_name || !weight_name[0] ||
        !weight_scale_name || !weight_scale_name[0] ||
        !input_scale_name || !input_scale_name[0] || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_tensor_info weight_info{};
    weight_info.abi_version = AXIOM_ABI_VERSION;
    axiom_tensor_info weight_scale_info{};
    weight_scale_info.abi_version = AXIOM_ABI_VERSION;
    axiom_tensor_info input_scale_info{};
    input_scale_info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, weight_name, &weight_info);
    if (rc == AXIOM_OK) {
        rc = axiom_model_tensor_info_get(model, weight_scale_name, &weight_scale_info);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_model_tensor_info_get(model, input_scale_name, &input_scale_info);
    }
    if (rc != AXIOM_OK) return rc;
    if (weight_info.dtype != AXIOM_TENSOR_DTYPE_F8_E4M3 || weight_info.rank != 2u ||
        weight_info.shape[0] == 0u || weight_info.shape[1] == 0u ||
        weight_info.shape[0] > std::numeric_limits<uint32_t>::max() ||
        weight_info.shape[1] > std::numeric_limits<uint32_t>::max() ||
        !exact_f32_scalar(weight_scale_info) || !exact_f32_scalar(input_scale_info)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t weight_bytes = 0u;
    if (!checked_mul_u64(weight_info.shape[0], weight_info.shape[1], &weight_bytes) ||
        weight_info.byte_count != weight_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    float host_weight_scale = 0.0f;
    float host_input_scale = 0.0f;
    rc = axiom_model_tensor_read_slice(
            model, weight_scale_name, 0u, &host_weight_scale, sizeof(host_weight_scale));
    if (rc == AXIOM_OK) {
        rc = axiom_model_tensor_read_slice(
                model, input_scale_name, 0u, &host_input_scale, sizeof(host_input_scale));
    }
    if (rc != AXIOM_OK) return rc;
    if (!valid_positive_f32(host_weight_scale) || !valid_positive_f32(host_input_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_qwen38_fp8_linear *linear = new (std::nothrow) axiom_qwen38_fp8_linear();
    if (!linear) return AXIOM_ERR_BUDGET;
    linear->device = device;
    linear->rows = static_cast<uint32_t>(weight_info.shape[0]);
    linear->cols = static_cast<uint32_t>(weight_info.shape[1]);
    linear->modelopt_scalar = true;
    linear->modelopt_weight_scale_host = host_weight_scale;
    linear->modelopt_input_scale_host = host_input_scale;
    const uint64_t input_bytes =
            static_cast<uint64_t>(linear->cols) * AXIOM_QWEN38_FP8_BATCH;
    cudaError_t cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->weight), static_cast<size_t>(weight_bytes));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->modelopt_weight_scale), sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->input_quantized), static_cast<size_t>(input_bytes));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->input_scale), sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
            reinterpret_cast<void **>(&linear->unit_scale), sizeof(float));
    if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(&linear->workspace, kWorkspaceBytes);
    if (cuda_status != cudaSuccess) {
        axiom_qwen38_fp8_linear_destroy(linear);
        return cuda_status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    rc = upload_tensor_chunked(model, weight_name, linear->weight, weight_bytes);
    const float one = 1.0f;
    if (rc == AXIOM_OK && cudaMemcpy(
            linear->modelopt_weight_scale, &host_weight_scale, sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK && cudaMemcpy(
            linear->input_scale, &host_input_scale, sizeof(float),
            cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK && cudaMemcpy(
            linear->unit_scale, &one, sizeof(one), cudaMemcpyHostToDevice) != cudaSuccess) {
        rc = AXIOM_ERR_CUDA;
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_fp8_linear_destroy(linear);
        return rc;
    }
    (void)configure_tensor_core(linear);
    linear->device_bytes = weight_bytes + input_bytes + 3u * sizeof(float) + kWorkspaceBytes;
    *out = linear;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_fp8_linear_load_auto(
        axiom_model *model,
        int device,
        const char *base,
        axiom_qwen38_fp8_linear **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !base || !base[0] || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    try {
        const std::string prefix(base);
        const std::string weight = prefix + ".weight";
        const std::string weight_scale = prefix + ".weight_scale";
        const std::string input_scale = prefix + ".input_scale";
        axiom_tensor_info scale_info{};
        scale_info.abi_version = AXIOM_ABI_VERSION;
        int rc = axiom_model_tensor_info_get(model, weight_scale.c_str(), &scale_info);
        if (rc != AXIOM_OK) return rc;
        if (scale_info.dtype == AXIOM_TENSOR_DTYPE_BF16) {
            return axiom_qwen38_fp8_linear_load(
                    model, device, weight.c_str(), weight_scale.c_str(), out);
        }
        if (scale_info.dtype == AXIOM_TENSOR_DTYPE_F32 && scale_info.rank == 0u) {
            return axiom_qwen38_fp8_linear_load_modelopt(
                    model, device, weight.c_str(), weight_scale.c_str(), input_scale.c_str(), out);
        }
        return AXIOM_ERR_INVALID_ARGUMENT;
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
}

extern "C" void axiom_qwen38_fp8_linear_destroy(axiom_qwen38_fp8_linear *linear) {
    if (!linear) return;
    if (linear->device >= 0) (void)cudaSetDevice(linear->device);
    destroy_lt(linear);
    if (linear->workspace) (void)cudaFree(linear->workspace);
    if (linear->unit_scale) (void)cudaFree(linear->unit_scale);
    if (linear->input_scale) (void)cudaFree(linear->input_scale);
    if (linear->input_quantized) (void)cudaFree(linear->input_quantized);
    if (linear->modelopt_weight_scale) (void)cudaFree(linear->modelopt_weight_scale);
    if (linear->weight_scale) (void)cudaFree(linear->weight_scale);
    if (linear->weight) (void)cudaFree(linear->weight);
    delete linear;
}

extern "C" uint32_t axiom_qwen38_fp8_linear_rows(const axiom_qwen38_fp8_linear *linear) {
    return linear ? linear->rows : 0u;
}

extern "C" uint32_t axiom_qwen38_fp8_linear_cols(const axiom_qwen38_fp8_linear *linear) {
    return linear ? linear->cols : 0u;
}

extern "C" uint64_t axiom_qwen38_fp8_linear_device_bytes(const axiom_qwen38_fp8_linear *linear) {
    return linear ? linear->device_bytes : 0u;
}

extern "C" int axiom_qwen38_fp8_linear_tensor_core_enabled(const axiom_qwen38_fp8_linear *linear) {
    return linear && linear->tensor_core ? 1 : 0;
}

extern "C" int axiom_qwen38_fp8_linear_forward_reference_f32_device(
        axiom_qwen38_fp8_linear *linear,
        const float *input,
        float *out,
        void *stream) {
    if (!linear || !input || !out || !linear->weight || !linear->input_quantized ||
        !linear->input_scale ||
        (linear->modelopt_scalar ? !linear->modelopt_weight_scale : !linear->weight_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int rc = quantize_input(linear, input, cuda_stream);
    if (rc == AXIOM_OK) rc = launch_reference(linear, out, cuda_stream);
    return rc;
}

extern "C" int axiom_qwen38_fp8_linear_forward_f32_device(
        axiom_qwen38_fp8_linear *linear,
        const float *input,
        float *out,
        void *stream) {
    if (!linear || !input || !out || !linear->weight || !linear->input_quantized ||
        !linear->input_scale ||
        (linear->modelopt_scalar ? !linear->modelopt_weight_scale : !linear->weight_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int rc = quantize_input(linear, input, cuda_stream);
    if (rc != AXIOM_OK) return rc;
    if (linear->tensor_core) return launch_tensor_core(linear, out, cuda_stream);
    return launch_reference(linear, out, cuda_stream);
}

extern "C" int axiom_qwen38_fp8_linear_prepare_input_f32_device(
        axiom_qwen38_fp8_linear *prepared,
        const float *input,
        void *stream) {
    if (!prepared || !input || !prepared->modelopt_scalar ||
        !prepared->input_quantized || !prepared->input_scale ||
        prepared->cols == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(prepared->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    return quantize_input(prepared, input, static_cast<cudaStream_t>(stream));
}

extern "C" int axiom_qwen38_fp8_linear_forward_prepared_f32_device(
        axiom_qwen38_fp8_linear *linear,
        const axiom_qwen38_fp8_linear *prepared,
        float *out,
        void *stream) {
    if (!linear || !prepared || !out || !linear->modelopt_scalar ||
        !prepared->modelopt_scalar || !linear->weight || !prepared->input_quantized ||
        !linear->input_scale || !prepared->input_scale ||
        linear->device != prepared->device || linear->cols == 0u ||
        linear->cols != prepared->cols ||
        std::memcmp(&linear->modelopt_input_scale_host,
                    &prepared->modelopt_input_scale_host,
                    sizeof(linear->modelopt_input_scale_host)) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (linear->tensor_core) {
        return launch_tensor_core_prepared(
                linear, prepared->input_quantized, out, cuda_stream);
    }
    return launch_reference_prepared(
            linear, prepared->input_quantized, out, cuda_stream);
}

extern "C" int axiom_qwen38_fp8_projection_group_create(
        axiom_model *model,
        int device,
        const axiom_qwen38_fp8_projection_group_config *config,
        axiom_qwen38_fp8_projection_group **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !config || !out ||
        config->abi_version != AXIOM_ABI_VERSION || config->flags != 0u ||
        config->projection_count < 2u ||
        config->projection_count > AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    try {
        std::vector<std::string> weight_names;
        std::vector<uint64_t> weight_bytes;
        std::vector<uint16_t> legacy_scales;
        std::vector<float> modelopt_row_scales;
        weight_names.reserve(config->projection_count);
        weight_bytes.reserve(config->projection_count);

        uint32_t schema = 0u;
        uint32_t cols = 0u;
        uint32_t total_rows = 0u;
        uint32_t row_offsets[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX + 1u]{};
        float common_input_scale = 0.0f;
        uint32_t common_input_scale_bits = 0u;
        bool have_common_input_scale = false;
        uint64_t total_weight_bytes = 0u;

        for (uint32_t projection = 0u;
             projection < config->projection_count; ++projection) {
            const axiom_qwen38_fp8_projection_group_member &member =
                    config->projections[projection];
            if (!member.weight_name || !member.weight_name[0] ||
                !member.weight_scale_name || !member.weight_scale_name[0]) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            axiom_tensor_info weight_info{};
            weight_info.abi_version = AXIOM_ABI_VERSION;
            axiom_tensor_info weight_scale_info{};
            weight_scale_info.abi_version = AXIOM_ABI_VERSION;
            int rc = axiom_model_tensor_info_get(model, member.weight_name, &weight_info);
            if (rc == AXIOM_OK) {
                rc = axiom_model_tensor_info_get(
                        model, member.weight_scale_name, &weight_scale_info);
            }
            if (rc != AXIOM_OK) return rc;
            if (weight_info.dtype != AXIOM_TENSOR_DTYPE_F8_E4M3 ||
                weight_info.rank != 2u || weight_info.shape[0] == 0u ||
                weight_info.shape[1] == 0u ||
                weight_info.shape[0] > std::numeric_limits<uint32_t>::max() ||
                weight_info.shape[1] > std::numeric_limits<uint32_t>::max()) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            uint64_t projection_weight_bytes = 0u;
            if (!checked_mul_u64(
                        weight_info.shape[0], weight_info.shape[1],
                        &projection_weight_bytes) ||
                weight_info.byte_count != projection_weight_bytes) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            const uint32_t projection_rows = static_cast<uint32_t>(weight_info.shape[0]);
            const uint32_t projection_cols = static_cast<uint32_t>(weight_info.shape[1]);
            if ((projection != 0u && projection_cols != cols) ||
                projection_rows > std::numeric_limits<uint32_t>::max() - total_rows ||
                projection_weight_bytes >
                        std::numeric_limits<uint64_t>::max() - total_weight_bytes) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            if (projection == 0u) cols = projection_cols;

            uint32_t member_schema = 0u;
            if (weight_scale_info.dtype == AXIOM_TENSOR_DTYPE_BF16) {
                member_schema = AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_LEGACY_ROW;
                if ((member.input_scale_name && member.input_scale_name[0]) ||
                    weight_scale_info.rank != 2u ||
                    weight_scale_info.shape[0] != projection_rows ||
                    weight_scale_info.shape[1] != 1u ||
                    weight_scale_info.byte_count !=
                            static_cast<uint64_t>(projection_rows) * sizeof(uint16_t)) {
                    return AXIOM_ERR_INVALID_ARGUMENT;
                }
                std::vector<uint16_t> scales(projection_rows);
                rc = axiom_model_tensor_read_slice(
                        model, member.weight_scale_name, 0u, scales.data(),
                        weight_scale_info.byte_count);
                if (rc != AXIOM_OK) return rc;
                if (!valid_bf16_scales(scales)) return AXIOM_ERR_INVALID_ARGUMENT;
                legacy_scales.insert(legacy_scales.end(), scales.begin(), scales.end());
            } else if (exact_f32_scalar(weight_scale_info)) {
                member_schema = AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_MODELOPT_SCALAR;
                if (!member.input_scale_name || !member.input_scale_name[0]) {
                    return AXIOM_ERR_INVALID_ARGUMENT;
                }
                axiom_tensor_info input_scale_info{};
                input_scale_info.abi_version = AXIOM_ABI_VERSION;
                rc = axiom_model_tensor_info_get(
                        model, member.input_scale_name, &input_scale_info);
                if (rc != AXIOM_OK) return rc;
                if (!exact_f32_scalar(input_scale_info)) return AXIOM_ERR_INVALID_ARGUMENT;
                float weight_scale = 0.0f;
                float input_scale = 0.0f;
                rc = axiom_model_tensor_read_slice(
                        model, member.weight_scale_name, 0u,
                        &weight_scale, sizeof(weight_scale));
                if (rc == AXIOM_OK) {
                    rc = axiom_model_tensor_read_slice(
                            model, member.input_scale_name, 0u,
                            &input_scale, sizeof(input_scale));
                }
                if (rc != AXIOM_OK) return rc;
                if (!valid_positive_f32(weight_scale) ||
                    !valid_positive_f32(input_scale)) {
                    return AXIOM_ERR_INVALID_ARGUMENT;
                }
                uint32_t input_scale_bits = 0u;
                std::memcpy(&input_scale_bits, &input_scale, sizeof(input_scale_bits));
                if (have_common_input_scale &&
                    input_scale_bits != common_input_scale_bits) {
                    return AXIOM_ERR_INVALID_ARGUMENT;
                }
                have_common_input_scale = true;
                common_input_scale = input_scale;
                common_input_scale_bits = input_scale_bits;
                modelopt_row_scales.insert(
                        modelopt_row_scales.end(), projection_rows, weight_scale);
            } else {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            if (projection != 0u && member_schema != schema) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            schema = member_schema;
            weight_names.emplace_back(member.weight_name);
            weight_bytes.push_back(projection_weight_bytes);
            total_weight_bytes += projection_weight_bytes;
            total_rows += projection_rows;
            row_offsets[projection + 1u] = total_rows;
        }

        uint64_t expected_weight_bytes = 0u;
        if (!checked_mul_u64(total_rows, cols, &expected_weight_bytes) ||
            expected_weight_bytes != total_weight_bytes || total_rows == 0u || cols == 0u) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (schema == AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_LEGACY_ROW &&
            legacy_scales.size() != total_rows) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (schema == AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_MODELOPT_SCALAR &&
            (!have_common_input_scale || modelopt_row_scales.size() != total_rows)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;

        axiom_qwen38_fp8_projection_group *group =
                new (std::nothrow) axiom_qwen38_fp8_projection_group();
        if (!group) return AXIOM_ERR_BUDGET;
        group->device = device;
        group->projection_count = config->projection_count;
        group->cols = cols;
        group->rows = total_rows;
        std::memcpy(group->row_offsets, row_offsets, sizeof(row_offsets));
        group->modelopt_scalar =
                schema == AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_MODELOPT_SCALAR;
        group->input_scale_host = common_input_scale;

        const uint64_t input_bytes =
                static_cast<uint64_t>(cols) * AXIOM_QWEN38_FP8_BATCH;
        const uint64_t scale_bytes = static_cast<uint64_t>(total_rows) *
                (group->modelopt_scalar ? sizeof(float) : sizeof(uint16_t));
        const uint64_t input_scale_bytes = group->modelopt_scalar
                ? sizeof(float)
                : AXIOM_QWEN38_FP8_BATCH * sizeof(float);
        cudaError_t cuda_status = cudaMalloc(
                reinterpret_cast<void **>(&group->weight),
                static_cast<size_t>(total_weight_bytes));
        if (cuda_status == cudaSuccess && group->modelopt_scalar) {
            cuda_status = cudaMalloc(
                    reinterpret_cast<void **>(&group->modelopt_row_weight_scale),
                    static_cast<size_t>(scale_bytes));
        } else if (cuda_status == cudaSuccess) {
            cuda_status = cudaMalloc(
                    reinterpret_cast<void **>(&group->legacy_weight_scale),
                    static_cast<size_t>(scale_bytes));
        }
        if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
                reinterpret_cast<void **>(&group->input_quantized),
                static_cast<size_t>(input_bytes));
        if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
                reinterpret_cast<void **>(&group->input_scale),
                static_cast<size_t>(input_scale_bytes));
        if (cuda_status == cudaSuccess) cuda_status = cudaMalloc(
                reinterpret_cast<void **>(&group->unit_scale), sizeof(float));
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMalloc(&group->workspace, kWorkspaceBytes);
        }
        if (cuda_status != cudaSuccess) {
            axiom_qwen38_fp8_projection_group_destroy(group);
            return cuda_status == cudaErrorMemoryAllocation
                    ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
        }

        int rc = AXIOM_OK;
        uint64_t destination_offset = 0u;
        for (uint32_t projection = 0u;
             projection < config->projection_count && rc == AXIOM_OK; ++projection) {
            rc = upload_tensor_chunked(
                    model, weight_names[projection].c_str(),
                    group->weight + destination_offset, weight_bytes[projection]);
            destination_offset += weight_bytes[projection];
        }
        const float one = 1.0f;
        if (rc == AXIOM_OK && group->modelopt_scalar && cudaMemcpy(
                    group->modelopt_row_weight_scale, modelopt_row_scales.data(),
                    static_cast<size_t>(scale_bytes), cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK && !group->modelopt_scalar && cudaMemcpy(
                    group->legacy_weight_scale, legacy_scales.data(),
                    static_cast<size_t>(scale_bytes), cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK && group->modelopt_scalar && cudaMemcpy(
                    group->input_scale, &common_input_scale, sizeof(common_input_scale),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK && cudaMemcpy(
                    group->unit_scale, &one, sizeof(one),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
        }
        if (rc != AXIOM_OK) {
            axiom_qwen38_fp8_projection_group_destroy(group);
            return rc;
        }
        (void)configure_group_tensor_core(group);
        group->device_bytes = total_weight_bytes + scale_bytes + input_bytes +
                input_scale_bytes + sizeof(float) + kWorkspaceBytes;
        *out = group;
        return AXIOM_OK;
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
}

extern "C" int axiom_qwen38_fp8_projection_group_load_auto(
        axiom_model *model,
        int device,
        const char *const *bases,
        uint32_t projection_count,
        axiom_qwen38_fp8_projection_group **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !bases || !out || projection_count < 2u ||
        projection_count > AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    try {
        std::string weights[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX];
        std::string weight_scales[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX];
        std::string input_scales[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX];
        axiom_qwen38_fp8_projection_group_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.projection_count = projection_count;
        for (uint32_t projection = 0u; projection < projection_count; ++projection) {
            if (!bases[projection] || !bases[projection][0]) {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
            const std::string prefix(bases[projection]);
            weights[projection] = prefix + ".weight";
            weight_scales[projection] = prefix + ".weight_scale";
            axiom_tensor_info scale_info{};
            scale_info.abi_version = AXIOM_ABI_VERSION;
            const int rc = axiom_model_tensor_info_get(
                    model, weight_scales[projection].c_str(), &scale_info);
            if (rc != AXIOM_OK) return rc;
            config.projections[projection].weight_name = weights[projection].c_str();
            config.projections[projection].weight_scale_name =
                    weight_scales[projection].c_str();
            if (scale_info.dtype == AXIOM_TENSOR_DTYPE_BF16) {
                config.projections[projection].input_scale_name = nullptr;
            } else if (exact_f32_scalar(scale_info)) {
                input_scales[projection] = prefix + ".input_scale";
                config.projections[projection].input_scale_name =
                        input_scales[projection].c_str();
            } else {
                return AXIOM_ERR_INVALID_ARGUMENT;
            }
        }
        return axiom_qwen38_fp8_projection_group_create(
                model, device, &config, out);
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
}

extern "C" void axiom_qwen38_fp8_projection_group_destroy(
        axiom_qwen38_fp8_projection_group *group) {
    if (!group) return;
    if (group->device >= 0) (void)cudaSetDevice(group->device);
    destroy_group_lt(group);
    if (group->workspace) (void)cudaFree(group->workspace);
    if (group->unit_scale) (void)cudaFree(group->unit_scale);
    if (group->input_scale) (void)cudaFree(group->input_scale);
    if (group->input_quantized) (void)cudaFree(group->input_quantized);
    if (group->modelopt_row_weight_scale) {
        (void)cudaFree(group->modelopt_row_weight_scale);
    }
    if (group->legacy_weight_scale) (void)cudaFree(group->legacy_weight_scale);
    if (group->weight) (void)cudaFree(group->weight);
    delete group;
}

extern "C" int axiom_qwen38_fp8_projection_group_info_get(
        const axiom_qwen38_fp8_projection_group *group,
        axiom_qwen38_fp8_projection_group_info *out) {
    if (!group || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi_version = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi_version;
    out->projection_count = group->projection_count;
    out->input_features = group->cols;
    out->total_output_features = group->rows;
    std::memcpy(out->row_offsets, group->row_offsets, sizeof(group->row_offsets));
    out->scale_schema = group->modelopt_scalar
            ? AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_MODELOPT_SCALAR
            : AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_LEGACY_ROW;
    out->tensor_core_enabled = group->tensor_core ? 1u : 0u;
    out->bf16_output_boundary = 1u;
    out->device_bytes = group->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_fp8_projection_group_forward_reference_f32_device(
        axiom_qwen38_fp8_projection_group *group,
        const float *input,
        float *concatenated_out,
        void *stream) {
    if (!group || !input || !concatenated_out || !group->weight ||
        !group->input_quantized || !group->input_scale || !group->unit_scale ||
        (group->modelopt_scalar ? !group->modelopt_row_weight_scale
                                : !group->legacy_weight_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(group->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int rc = quantize_group_input(group, input, cuda_stream);
    if (rc == AXIOM_OK) rc = launch_group_reference(group, concatenated_out, cuda_stream);
    return rc;
}

extern "C" int axiom_qwen38_fp8_projection_group_forward_f32_device(
        axiom_qwen38_fp8_projection_group *group,
        const float *input,
        float *concatenated_out,
        void *stream) {
    if (!group || !input || !concatenated_out || !group->weight ||
        !group->input_quantized || !group->input_scale || !group->unit_scale ||
        (group->modelopt_scalar ? !group->modelopt_row_weight_scale
                                : !group->legacy_weight_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(group->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int rc = quantize_group_input(group, input, cuda_stream);
    if (rc != AXIOM_OK) return rc;
    if (group->tensor_core) {
        return launch_group_tensor_core(group, concatenated_out, cuda_stream);
    }
    return launch_group_reference(group, concatenated_out, cuda_stream);
}
