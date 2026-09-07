/* Native CUDA executor for the BF16 Qwen3.8 MTP block. */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_attention.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"

namespace {

constexpr uint32_t kHidden = AXIOM_QWEN38_MTP_HIDDEN;
constexpr uint32_t kIntermediate = AXIOM_QWEN38_MTP_INTERMEDIATE;
constexpr uint32_t kHeads = AXIOM_QWEN38_MTP_ATTENTION_HEADS;
constexpr uint32_t kKvHeads = AXIOM_QWEN38_MTP_KV_HEADS;
constexpr uint32_t kHeadDim = AXIOM_QWEN38_MTP_HEAD_DIM;
constexpr uint32_t kQDim = kHeads * kHeadDim;
constexpr uint32_t kKvDim = kKvHeads * kHeadDim;
constexpr uint32_t kQGateDim = kQDim * 2u;
constexpr uint32_t kRopeDim = AXIOM_QWEN38_ATTENTION_ROPE_DIM;
constexpr uint32_t kVocab = AXIOM_QWEN38_MTP_VOCAB;
constexpr uint32_t kMaxColumns = AXIOM_QWEN38_MTP_BATCH;
constexpr uint32_t kFusionInput = AXIOM_QWEN38_MTP_FC_INPUT;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kGqaHeads = kHeads / kKvHeads;
constexpr uint32_t kAttentionThreads = kGqaHeads * 32u;
constexpr uint32_t kAttentionSplits = 8u;
constexpr uint32_t kAttentionTileTokens = 16u;
constexpr uint32_t kTop1Blocks = 256u;
constexpr uint32_t kKvPageTokens = AXIOM_QWEN38_MTP_COMPUTE_KV_PAGE_TOKENS;
constexpr uint64_t kKvPageBytes = AXIOM_QWEN38_MTP_COMPUTE_KV_PAGE_BYTES;
constexpr uint64_t kKvPageElements = static_cast<uint64_t>(kKvPageTokens) * kKvDim;
constexpr uint64_t kCublasWorkspaceBytes = 32ull * 1024ull * 1024ull;
constexpr float kRmsEps = 1.0e-6f;
constexpr float kRopeTheta = AXIOM_QWEN38_TARGET_ROPE_THETA;
constexpr float kYarnFactor = AXIOM_QWEN38_TARGET_YARN_FACTOR;
constexpr float kYarnOriginalContext = AXIOM_QWEN38_TARGET_YARN_ORIGINAL_CONTEXT;
constexpr float kYarnBetaFast = AXIOM_QWEN38_TARGET_YARN_BETA_FAST;
constexpr float kYarnBetaSlow = AXIOM_QWEN38_TARGET_YARN_BETA_SLOW;
constexpr float kYarnMscale = AXIOM_QWEN38_TARGET_YARN_MSCALE;
constexpr float kE4m3MaxFinite = 448.0f;
constexpr uint64_t kDerivedFp8WeightElements =
        static_cast<uint64_t>(kHidden) * kFusionInput +
        static_cast<uint64_t>(kQGateDim) * kHidden +
        2ull * kKvDim * kHidden +
        static_cast<uint64_t>(kHidden) * kQDim +
        2ull * kIntermediate * kHidden +
        static_cast<uint64_t>(kHidden) * kIntermediate;
constexpr uint64_t kDerivedFp8ScaleElements =
        static_cast<uint64_t>(kHidden) + kQGateDim + 2ull * kKvDim +
        kHidden + 2ull * kIntermediate + kHidden;

static_assert(kHidden == 5120u && kIntermediate == 17408u,
              "Qwen3.8 MTP dense geometry changed");
static_assert(kQDim == 6144u && kKvDim == 1024u && kQGateDim == 12288u,
              "Qwen3.8 MTP attention geometry changed");
static_assert(kGqaHeads == 6u && kAttentionThreads == 192u,
              "Qwen3.8 MTP GQA geometry changed");
static_assert(kFusionInput == 2u * kHidden,
              "Qwen3.8 MTP fusion geometry changed");
static_assert(kDerivedFp8WeightElements == 424673280ull &&
              kDerivedFp8ScaleElements == 64512ull,
              "Qwen3.8 MTP derived FP8 geometry changed");
static_assert(kKvPageTokens == 256u && kKvPageElements == 262144u &&
              kKvPageBytes == 2u * kKvPageElements,
              "Qwen3.8 MTP persistence page geometry changed");

struct mtp_weights {
    const uint16_t *fc = nullptr;
    const uint16_t *input_norm = nullptr;
    const uint16_t *mlp_down = nullptr;
    const uint16_t *mlp_gate = nullptr;
    const uint16_t *mlp_up = nullptr;
    const uint16_t *post_attention_norm = nullptr;
    const uint16_t *k_norm = nullptr;
    const uint16_t *k_proj = nullptr;
    const uint16_t *o_proj = nullptr;
    const uint16_t *q_norm = nullptr;
    const uint16_t *q_proj = nullptr;
    const uint16_t *v_proj = nullptr;
    const uint16_t *final_norm = nullptr;
    const uint16_t *pre_fc_embedding_norm = nullptr;
    const uint16_t *pre_fc_hidden_norm = nullptr;
};

struct mtp_fp8_projection {
    uint8_t *weight = nullptr;
    float *scales = nullptr;
};

struct mtp_derived_fp8_weights {
    mtp_fp8_projection fc{};
    mtp_fp8_projection q_proj{};
    mtp_fp8_projection k_proj{};
    mtp_fp8_projection v_proj{};
    mtp_fp8_projection o_proj{};
    mtp_fp8_projection mlp_gate{};
    mtp_fp8_projection mlp_up{};
    mtp_fp8_projection mlp_down{};
};

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

int cublas_status(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return AXIOM_OK;
    return status == CUBLAS_STATUS_ALLOC_FAILED ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

float bf16_host_to_float(uint16_t bits) {
    uint32_t expanded = static_cast<uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

uint16_t float_host_to_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16u);
}

uint8_t float_host_to_e4m3fn(float value) {
    const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
    const float magnitude = std::fabs(value);
    if (std::isnan(value)) return static_cast<uint8_t>(sign | 0x7fu);
    if (!std::isfinite(value) || magnitude >= kE4m3MaxFinite) {
        return static_cast<uint8_t>(sign | 0x7eu);
    }
    if (magnitude < 0.015625f) {
        const int mantissa = static_cast<int>(std::nearbyint(magnitude * 512.0f));
        return mantissa >= 8 ? static_cast<uint8_t>(sign | 0x08u)
                             : static_cast<uint8_t>(sign | mantissa);
    }
    int exponent = std::ilogb(magnitude) + 7;
    int mantissa = static_cast<int>(std::nearbyint(
            (magnitude / std::ldexp(1.0f, exponent - 7) - 1.0f) * 8.0f));
    if (mantissa >= 8) {
        ++exponent;
        mantissa = 0;
    }
    if (exponent >= 15) return static_cast<uint8_t>(sign | 0x7eu);
    return static_cast<uint8_t>(sign | (exponent << 3u) | mantissa);
}

float e4m3fn_to_float(uint8_t code) {
    const uint32_t magnitude = static_cast<uint32_t>(code) & 0x7fu;
    if (magnitude == 0x7fu) return std::numeric_limits<float>::quiet_NaN();
    const float sign = (code & 0x80u) != 0u ? -1.0f : 1.0f;
    const uint32_t exponent = (magnitude >> 3u) & 0x0fu;
    const uint32_t mantissa = magnitude & 0x07u;
    if (exponent == 0u) return sign * std::ldexp(static_cast<float>(mantissa), -9);
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                             static_cast<int>(exponent) - 7);
}

bool mtp_gemv_m1_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_MTP_GEMV_M1");
    return !value || value[0] == '\0' ||
            !(value[0] == '0' && value[1] == '\0');
}

bool mtp_fp8_weight_gemv_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_MTP_FP8_WEIGHT_GEMV");
    return value && value[0] == '1' && value[1] == '\0';
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int add_bytes(uint64_t *total, uint64_t amount) {
    uint64_t next = 0u;
    if (!total || !checked_add(*total, amount, &next)) return AXIOM_ERR_BUDGET;
    *total = next;
    return AXIOM_OK;
}

int buffer_pointer(const axiom_device_buffer *buffer, void **out) {
    if (!buffer || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_device_buffer_cuda_pointer(buffer, out);
}

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ __forceinline__ uint16_t float_to_bf16(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float round_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
}

__device__ __forceinline__ uint8_t encode_e4m3fn(float value) {
    return static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float decode_e4m3fn(uint8_t code) {
    const uint32_t magnitude = static_cast<uint32_t>(code) & 0x7fu;
    if (magnitude == 0x7fu) return nanf("");
    const uint32_t exponent = (static_cast<uint32_t>(code) >> 3u) & 0x0fu;
    const uint32_t mantissa = static_cast<uint32_t>(code) & 0x07u;
    const uint32_t sign = (static_cast<uint32_t>(code) & 0x80u) << 24u;
    if (exponent == 0u) {
        const float value = ldexpf(static_cast<float>(mantissa), -9);
        return sign != 0u ? -value : value;
    }
    return __uint_as_float(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
}

__global__ void quantize_bf16_rows_e4m3_kernel(
        const uint16_t *__restrict__ input,
        uint8_t *__restrict__ quantized,
        float *__restrict__ scales,
        uint32_t rows,
        uint32_t cols,
        uint32_t *__restrict__ status) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    const uint16_t *in = input + static_cast<uint64_t>(row) * cols;
    float local_max = 0.0f;
    bool invalid = false;
    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        const float value = bf16_to_float(in[col]);
        if (isfinite(value)) {
            local_max = fmaxf(local_max, fabsf(value));
        } else {
            invalid = true;
        }
    }
    if (invalid) {
        (void)atomicCAS(status, static_cast<uint32_t>(AXIOM_OK),
                        static_cast<uint32_t>(AXIOM_ERR_RUNTIME));
    }
    __shared__ float maxima[kThreads];
    maxima[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
        __syncthreads();
    }
    const float scale = maxima[0] == 0.0f
            ? 1.0f : maxima[0] / kE4m3MaxFinite;
    if (tid == 0u) scales[row] = scale;
    uint8_t *out = quantized + static_cast<uint64_t>(row) * cols;
    for (uint32_t col = tid; col < cols; col += blockDim.x) {
        out[col] = encode_e4m3fn(bf16_to_float(in[col]) / scale);
    }
}

__device__ __forceinline__ float yarn_inv_frequency(uint32_t pair) {
    const float dim = static_cast<float>(kRopeDim);
    const float exponent = (2.0f * static_cast<float>(pair)) / dim;
    const float inv = 1.0f / powf(kRopeTheta, exponent);
    const float correction_fast = dim *
            logf(kYarnOriginalContext / (kYarnBetaFast * 6.283185307179586f)) /
            (2.0f * logf(kRopeTheta));
    const float correction_slow = dim *
            logf(kYarnOriginalContext / (kYarnBetaSlow * 6.283185307179586f)) /
            (2.0f * logf(kRopeTheta));
    const float low = floorf(fminf(correction_fast, correction_slow));
    const float high = ceilf(fmaxf(correction_fast, correction_slow));
    float ramp = high > low
            ? (static_cast<float>(pair) - low) / (high - low)
            : static_cast<float>(pair) >= high ? 1.0f : 0.0f;
    ramp = fminf(1.0f, fmaxf(0.0f, ramp));
    return inv * (1.0f - ramp) + (inv / kYarnFactor) * ramp;
}

__device__ __forceinline__ float native_inv_frequency(uint32_t pair) {
    const float exponent = (2.0f * static_cast<float>(pair)) /
            static_cast<float>(kRopeDim);
    return 1.0f / powf(kRopeTheta, exponent);
}

__global__ void resolve_device_position_kernel(
        const uint32_t *__restrict__ base_position,
        uint32_t position_offset,
        uint32_t columns,
        uint32_t max_context,
        uint32_t *__restrict__ async_status,
        uint32_t *__restrict__ resolved_position) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) {
        resolved_position[0] = 0u;
        return;
    }
    const uint64_t candidate = static_cast<uint64_t>(base_position[0]) +
            static_cast<uint64_t>(position_offset);
    const bool invalid = candidate > static_cast<uint64_t>(UINT32_MAX) ||
            candidate >= static_cast<uint64_t>(max_context) ||
            static_cast<uint64_t>(columns) >
                    static_cast<uint64_t>(max_context) - candidate;
    if (invalid) {
        (void)atomicCAS(async_status, static_cast<uint32_t>(AXIOM_OK),
                        static_cast<uint32_t>(AXIOM_ERR_BUDGET));
        resolved_position[0] = 0u;
        return;
    }
    resolved_position[0] = static_cast<uint32_t>(candidate);
}

__global__ void sanitize_device_tokens_kernel(
        const uint32_t *__restrict__ input,
        uint32_t *__restrict__ output,
        uint32_t columns,
        uint32_t *__restrict__ async_status) {
    const uint32_t column = threadIdx.x;
    if (blockIdx.x != 0u || column >= columns) return;
    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) {
        output[column] = 0u;
        return;
    }
    const uint32_t token = input[column];
    if (token >= kVocab) {
        output[column] = 0u;
        (void)atomicCAS(async_status, static_cast<uint32_t>(AXIOM_OK),
                        static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
        return;
    }
    output[column] = token;
}

__global__ void f32_to_bf16_stage_kernel(
        const float *__restrict__ input,
        uint16_t *__restrict__ output,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = float_to_bf16(input[index]);
}

/* cuBLAS' GEMM selector is tuned for matrix shapes and leaves substantial
 * bandwidth unused for the autoregressive M=1 projections.  Every Qwen3.8
 * MTP K dimension is even, so one block owns one output row and streams BF16
 * pairs through a deterministic FP32 reduction.  The activation has already
 * crossed the same RN-even BF16 boundary used by the M8 cuBLAS path. */
__global__ void bf16_gemv_m1_f32_kernel(
        const uint16_t *__restrict__ weight,
        const uint16_t *__restrict__ input,
        float *__restrict__ output,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    const uint32_t pairs = cols >> 1u;
    const __nv_bfloat162 *weight_pairs =
            reinterpret_cast<const __nv_bfloat162 *>(
                    weight + static_cast<uint64_t>(row) * cols);
    const __nv_bfloat162 *input_pairs =
            reinterpret_cast<const __nv_bfloat162 *>(input);
    float partial = 0.0f;
    for (uint32_t pair = tid; pair < pairs; pair += blockDim.x) {
        const float2 w = __bfloat1622float2(weight_pairs[pair]);
        const float2 x = __bfloat1622float2(input_pairs[pair]);
        partial = fmaf(w.x, x.x, partial);
        partial = fmaf(w.y, x.y, partial);
    }
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[kThreads / 32u];
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (lane == 0u) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0u) {
        partial = lane < kThreads / 32u ? warp_sums[lane] : 0.0f;
        for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0u) output[row] = partial;
    }
}

__global__ void fp8_weight_bf16_gemv_m1_f32_kernel(
        const uint8_t *__restrict__ weight,
        const float *__restrict__ scales,
        const uint16_t *__restrict__ input,
        float *__restrict__ output,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    const uint32_t pairs = cols >> 1u;
    const uint8_t *row_weight = weight + static_cast<uint64_t>(row) * cols;
    const __nv_bfloat162 *input_pairs =
            reinterpret_cast<const __nv_bfloat162 *>(input);
    const float scale = scales[row];
    float partial = 0.0f;
    for (uint32_t pair = tid; pair < pairs; pair += blockDim.x) {
        const uint32_t col = pair << 1u;
        const float2 x = __bfloat1622float2(input_pairs[pair]);
        partial = fmaf(decode_e4m3fn(row_weight[col]) * scale, x.x, partial);
        partial = fmaf(decode_e4m3fn(row_weight[col + 1u]) * scale, x.y, partial);
    }
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    __shared__ float warp_sums[kThreads / 32u];
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (lane == 0u) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0u) {
        partial = lane < kThreads / 32u ? warp_sums[lane] : 0.0f;
        for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
            partial += __shfl_down_sync(0xffffffffu, partial, offset);
        }
        if (lane == 0u) output[row] = partial;
    }
}

__global__ void round_bf16_inplace_kernel(float *values, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = round_bf16(values[index]);
}

int launch_round_bf16(float *values, uint64_t count, cudaStream_t stream) {
    if (!values || count == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t blocks = (count + kThreads - 1u) / kThreads;
    if (blocks > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    round_bf16_inplace_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0, stream>>>(
            values, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void rmsnorm_zero_centered_kernel(
        const uint16_t *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ output,
        uint32_t features,
        uint32_t columns) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= columns) return;
    const float *x = input + static_cast<uint64_t>(column) * features;
    float *y = output + static_cast<uint64_t>(column) * features;
    __shared__ double sums[kThreads];
    double sum = 0.0;
    for (uint32_t feature = tid; feature < features; feature += blockDim.x) {
        const double value = static_cast<double>(x[feature]);
        sum += value * value;
    }
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = static_cast<float>(
            1.0 / sqrt(sums[0] / static_cast<double>(features) + kRmsEps));
    for (uint32_t feature = tid; feature < features; feature += blockDim.x) {
        y[feature] = round_bf16(
                x[feature] * inverse * (1.0f + bf16_to_float(weight[feature])));
    }
}

int launch_rmsnorm(
        const uint16_t *weight,
        const float *input,
        float *output,
        uint32_t features,
        uint32_t columns,
        cudaStream_t stream) {
    if (!weight || !input || !output || features == 0u || columns == 0u ||
        columns > kMaxColumns) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    rmsnorm_zero_centered_kernel<<<columns, kThreads, 0, stream>>>(
            weight, input, output, features, columns);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void pack_fusion_kernel(
        const float *__restrict__ embedding,
        const float *__restrict__ hidden,
        float *__restrict__ output,
        uint32_t columns) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kFusionInput) * columns;
    if (index >= count) return;
    const uint32_t column = static_cast<uint32_t>(index / kFusionInput);
    const uint32_t feature = static_cast<uint32_t>(index -
            static_cast<uint64_t>(column) * kFusionInput);
    output[index] = feature < kHidden
            ? embedding[static_cast<uint64_t>(column) * kHidden + feature]
            : hidden[static_cast<uint64_t>(column) * kHidden + feature - kHidden];
}

__global__ void split_q_gate_round_kv_kernel(
        const float *__restrict__ q_gate,
        float *__restrict__ q,
        float *__restrict__ gate,
        float *__restrict__ k,
        float *__restrict__ v,
        uint32_t columns) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t q_count = static_cast<uint64_t>(kQDim) * columns;
    const uint64_t kv_count = static_cast<uint64_t>(kKvDim) * columns;
    if (index < q_count) {
        const uint32_t column = static_cast<uint32_t>(index / kQDim);
        const uint32_t element = static_cast<uint32_t>(index -
                static_cast<uint64_t>(column) * kQDim);
        const uint32_t head = element / kHeadDim;
        const uint32_t dim = element - head * kHeadDim;
        const uint64_t source = static_cast<uint64_t>(column) * kQGateDim +
                static_cast<uint64_t>(head) * 2u * kHeadDim + dim;
        q[index] = round_bf16(q_gate[source]);
        gate[index] = round_bf16(q_gate[source + kHeadDim]);
    } else if (index < q_count + kv_count) {
        const uint64_t kv_index = index - q_count;
        k[kv_index] = round_bf16(k[kv_index]);
        v[kv_index] = round_bf16(v[kv_index]);
    }
}

template <typename CacheT>
__device__ __forceinline__ CacheT encode_cache(float value);

template <>
__device__ __forceinline__ uint16_t encode_cache<uint16_t>(float value) {
    return float_to_bf16(value);
}

template <>
__device__ __forceinline__ uint8_t encode_cache<uint8_t>(float value) {
    return encode_e4m3fn(value);
}

template <typename CacheT>
__device__ __forceinline__ float decode_cache(CacheT value);

template <>
__device__ __forceinline__ float decode_cache<uint16_t>(uint16_t value) {
    return bf16_to_float(value);
}

template <>
__device__ __forceinline__ float decode_cache<uint8_t>(uint8_t value) {
    return decode_e4m3fn(value);
}

template <typename CacheT, bool DevicePosition>
__global__ void qk_norm_rope_store_kernel(
        float *__restrict__ q,
        float *__restrict__ k,
        const float *__restrict__ v,
        const uint16_t *__restrict__ q_weight,
        const uint16_t *__restrict__ k_weight,
        CacheT *__restrict__ k_cache,
        CacheT *__restrict__ v_cache,
        uint32_t first_position,
        const uint32_t *__restrict__ first_position_device,
        const uint32_t *__restrict__ async_status,
        uint32_t columns,
        uint32_t max_context,
        bool yarn_enabled) {
    const uint32_t combined_head = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (column >= columns || combined_head >= kHeads + kKvHeads || tid >= kHeadDim) return;
    if constexpr (DevicePosition) {
        if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) return;
        first_position = first_position_device[0];
    }
    const bool is_k = combined_head >= kHeads;
    const uint32_t head = is_k ? combined_head - kHeads : combined_head;
    const uint32_t heads = is_k ? kKvHeads : kHeads;
    float *row = (is_k ? k : q) +
            (static_cast<uint64_t>(column) * heads + head) * kHeadDim;
    const uint16_t *weight = is_k ? k_weight : q_weight;
    __shared__ double sums[kThreads];
    __shared__ float normalized[kHeadDim];
    const double value = static_cast<double>(row[tid]);
    sums[tid] = value * value;
    __syncthreads();
    for (uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = static_cast<float>(
            1.0 / sqrt(sums[0] / static_cast<double>(kHeadDim) + kRmsEps));
    normalized[tid] = row[tid] * inverse * (1.0f + bf16_to_float(weight[tid]));
    __syncthreads();
    constexpr uint32_t kRopeHalf = kRopeDim / 2u;
    const uint32_t position = first_position + column;
    if (tid < kRopeHalf) {
        const float frequency = yarn_enabled
                ? yarn_inv_frequency(tid) : native_inv_frequency(tid);
        const float angle = static_cast<float>(position) * frequency;
        const float scale = yarn_enabled ? kYarnMscale : 1.0f;
        const float cosine = cosf(angle) * scale;
        const float sine = sinf(angle) * scale;
        const float first = normalized[tid];
        const float second = normalized[tid + kRopeHalf];
        row[tid] = round_bf16(first * cosine - second * sine);
        row[tid + kRopeHalf] = round_bf16(first * sine + second * cosine);
    } else if (tid >= kRopeDim) {
        row[tid] = round_bf16(normalized[tid]);
    }
    __syncthreads();
    if (is_k && position < max_context) {
        const uint64_t cache_index =
                (static_cast<uint64_t>(position) * kKvHeads + head) * kHeadDim + tid;
        k_cache[cache_index] = encode_cache<CacheT>(row[tid]);
        const uint64_t value_index =
                (static_cast<uint64_t>(column) * kKvHeads + head) * kHeadDim + tid;
        v_cache[cache_index] = encode_cache<CacheT>(round_bf16(v[value_index]));
    }
}

__host__ __device__ __forceinline__ uint64_t split_stats_index(
        uint32_t row, uint32_t kv_head, uint32_t split, uint32_t gqa_head) {
    return (((static_cast<uint64_t>(row) * kKvHeads + kv_head) * kAttentionSplits + split) *
            kGqaHeads + gqa_head);
}

__host__ __device__ __forceinline__ uint64_t split_value_index(
        uint32_t row, uint32_t kv_head, uint32_t split, uint32_t gqa_head,
        uint32_t dim) {
    return split_stats_index(row, kv_head, split, gqa_head) * kHeadDim + dim;
}

template <typename CacheT, bool DevicePosition>
__global__ void causal_gqa_splitk_kernel(
        const float *__restrict__ q,
        const CacheT *__restrict__ k_cache,
        const CacheT *__restrict__ v_cache,
        float *__restrict__ split_values,
        float *__restrict__ split_maxima,
        float *__restrict__ split_denominators,
        uint32_t first_position,
        const uint32_t *__restrict__ first_position_device,
        const uint32_t *__restrict__ async_status,
        uint32_t columns) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t split = blockIdx.z;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || row >= columns || split >= kAttentionSplits) return;
    if constexpr (DevicePosition) {
        if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) return;
        first_position = first_position_device[0];
    }
    __shared__ float tile_k[kAttentionTileTokens * kHeadDim];
    __shared__ float tile_v[kAttentionTileTokens * kHeadDim];
    const bool active = warp < kGqaHeads;
    const uint32_t query_head = kv_head * kGqaHeads + warp;
    const uint64_t q_base =
            (static_cast<uint64_t>(row) * kHeads + query_head) * kHeadDim;
    float query_values[kHeadDim / 32u];
    float accumulators[kHeadDim / 32u];
    if (active) {
#pragma unroll
        for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
            query_values[part] = q[q_base + lane + part * 32u];
            accumulators[part] = 0.0f;
        }
    }
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    const uint32_t cache_tokens = first_position + row + 1u;
    const uint32_t token_begin = static_cast<uint32_t>(
            (static_cast<uint64_t>(cache_tokens) * split) / kAttentionSplits);
    const uint32_t token_end = static_cast<uint32_t>(
            (static_cast<uint64_t>(cache_tokens) * (split + 1u)) / kAttentionSplits);
    const float attention_scale = rsqrtf(static_cast<float>(kHeadDim));
    for (uint32_t tile_begin = token_begin; tile_begin < token_end;
         tile_begin += kAttentionTileTokens) {
        const uint32_t tile_count = min(kAttentionTileTokens, token_end - tile_begin);
        const uint32_t tile_values = tile_count * kHeadDim;
        for (uint32_t index = threadIdx.x; index < tile_values; index += blockDim.x) {
            const uint32_t tile_token = index / kHeadDim;
            const uint32_t dim = index - tile_token * kHeadDim;
            const uint64_t cache_index =
                    (static_cast<uint64_t>(tile_begin + tile_token) * kKvHeads + kv_head) *
                    kHeadDim + dim;
            tile_k[index] = decode_cache<CacheT>(k_cache[cache_index]);
            tile_v[index] = decode_cache<CacheT>(v_cache[cache_index]);
        }
        __syncthreads();
        if (active) {
            for (uint32_t tile_token = 0u; tile_token < tile_count; ++tile_token) {
                const uint32_t tile_offset = tile_token * kHeadDim;
                float dot = 0.0f;
#pragma unroll
                for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
                    dot = fmaf(query_values[part],
                               tile_k[tile_offset + lane + part * 32u], dot);
                }
                for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
                    dot += __shfl_down_sync(0xffffffffu, dot, offset);
                }
                const float score = __shfl_sync(0xffffffffu, dot, 0u) * attention_scale;
                const float next_maximum = fmaxf(maximum, score);
                const float correction = __expf(maximum - next_maximum);
                const float weight = __expf(score - next_maximum);
                denominator = denominator * correction + weight;
#pragma unroll
                for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
                    const uint32_t dim = lane + part * 32u;
                    accumulators[part] = fmaf(
                            weight, tile_v[tile_offset + dim],
                            accumulators[part] * correction);
                }
                maximum = next_maximum;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    const uint64_t stats = split_stats_index(row, kv_head, split, warp);
    if (lane == 0u) {
        split_maxima[stats] = maximum;
        split_denominators[stats] = denominator;
    }
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        const uint32_t dim = lane + part * 32u;
        split_values[split_value_index(row, kv_head, split, warp, dim)] =
                accumulators[part];
    }
}

__global__ void merge_gqa_splitk_kernel(
        const float *__restrict__ split_values,
        const float *__restrict__ split_maxima,
        const float *__restrict__ split_denominators,
        float *__restrict__ output,
        uint32_t columns) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || row >= columns || warp >= kGqaHeads) return;
    float accumulators[kHeadDim / 32u]{};
    float maximum = -CUDART_INF_F;
    float denominator = 0.0f;
    for (uint32_t split = 0u; split < kAttentionSplits; ++split) {
        const uint64_t stats = split_stats_index(row, kv_head, split, warp);
        const float partial_denominator = split_denominators[stats];
        if (partial_denominator == 0.0f) continue;
        const float partial_maximum = split_maxima[stats];
        const float next_maximum = fmaxf(maximum, partial_maximum);
        const float correction = __expf(maximum - next_maximum);
        const float partial_correction = __expf(partial_maximum - next_maximum);
        denominator = denominator * correction + partial_denominator * partial_correction;
#pragma unroll
        for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
            const uint32_t dim = lane + part * 32u;
            const float partial = split_values[
                    split_value_index(row, kv_head, split, warp, dim)];
            accumulators[part] = fmaf(
                    partial_correction, partial, accumulators[part] * correction);
        }
        maximum = next_maximum;
    }
    const uint32_t query_head = kv_head * kGqaHeads + warp;
    const uint64_t base = (static_cast<uint64_t>(row) * kHeads + query_head) * kHeadDim;
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        const uint32_t dim = lane + part * 32u;
        output[base + dim] = round_bf16(
                denominator > 0.0f ? accumulators[part] / denominator : 0.0f);
    }
}

__global__ void sigmoid_gate_kernel(
        const float *__restrict__ attention,
        const float *__restrict__ gate,
        float *__restrict__ output,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = round_bf16(
                attention[index] * (1.0f / (1.0f + expf(-gate[index]))));
    }
}

__global__ void residual_add_kernel(
        const float *__restrict__ left,
        const float *__restrict__ right,
        float *__restrict__ output,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = round_bf16(left[index] + right[index]);
}

__global__ void silu_multiply_kernel(
        float *__restrict__ gate,
        const float *__restrict__ up,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const float value = gate[index];
        gate[index] = round_bf16((value / (1.0f + expf(-value))) * up[index]);
    }
}

__device__ __forceinline__ void top1_select(
        float candidate_value,
        uint32_t candidate_id,
        float *best_value,
        uint32_t *best_id) {
    if (candidate_value > *best_value ||
        (candidate_value == *best_value && candidate_id < *best_id)) {
        *best_value = candidate_value;
        *best_id = candidate_id;
    }
}

__device__ __forceinline__ void top1_warp_reduce(float *best_value, uint32_t *best_id) {
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const float candidate_value = __shfl_down_sync(0xffffffffu, *best_value, offset);
        const uint32_t candidate_id = __shfl_down_sync(0xffffffffu, *best_id, offset);
        top1_select(candidate_value, candidate_id, best_value, best_id);
    }
}

__global__ void top1_stage1_kernel(
        const float *__restrict__ logits,
        float *__restrict__ block_values,
        uint32_t *__restrict__ block_ids) {
    const uint32_t tid = threadIdx.x;
    float best_value = -CUDART_INF_F;
    uint32_t best_id = UINT32_MAX;
    for (uint32_t token = blockIdx.x * blockDim.x + tid;
         token < kVocab; token += gridDim.x * blockDim.x) {
        const float value = logits[token];
        if (isfinite(value)) top1_select(value, token, &best_value, &best_id);
    }
    top1_warp_reduce(&best_value, &best_id);
    __shared__ float warp_values[kThreads / 32u];
    __shared__ uint32_t warp_ids[kThreads / 32u];
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (lane == 0u) {
        warp_values[warp] = best_value;
        warp_ids[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0u) {
        best_value = lane < kThreads / 32u ? warp_values[lane] : -CUDART_INF_F;
        best_id = lane < kThreads / 32u ? warp_ids[lane] : UINT32_MAX;
        top1_warp_reduce(&best_value, &best_id);
        if (lane == 0u) {
            block_values[blockIdx.x] = best_value;
            block_ids[blockIdx.x] = best_id;
        }
    }
}

__global__ void top1_stage2_kernel(
        const float *__restrict__ block_values,
        const uint32_t *__restrict__ block_ids,
        uint32_t *__restrict__ output,
        uint32_t *__restrict__ async_status) {
    const uint32_t tid = threadIdx.x;
    float best_value = tid < kTop1Blocks ? block_values[tid] : -CUDART_INF_F;
    uint32_t best_id = tid < kTop1Blocks ? block_ids[tid] : UINT32_MAX;
    top1_warp_reduce(&best_value, &best_id);
    __shared__ float warp_values[kThreads / 32u];
    __shared__ uint32_t warp_ids[kThreads / 32u];
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (lane == 0u) {
        warp_values[warp] = best_value;
        warp_ids[warp] = best_id;
    }
    __syncthreads();
    if (warp == 0u) {
        best_value = lane < kThreads / 32u ? warp_values[lane] : -CUDART_INF_F;
        best_id = lane < kThreads / 32u ? warp_ids[lane] : UINT32_MAX;
        top1_warp_reduce(&best_value, &best_id);
        if (lane == 0u) {
            if (best_id == UINT32_MAX && async_status) {
                (void)atomicCAS(async_status, static_cast<uint32_t>(AXIOM_OK),
                                static_cast<uint32_t>(AXIOM_ERR_RUNTIME));
            }
            output[0] = !async_status ||
                    async_status[0] == static_cast<uint32_t>(AXIOM_OK)
                    ? best_id : UINT32_MAX;
        }
    }
}

template <bool ConfigureStream>
int gemm_bf16_weight_bf16_f32(
        cublasHandle_t handle,
        const uint16_t *weight,
        uint32_t rows,
        uint32_t cols,
        const float *input,
        float *output,
        uint32_t columns,
        uint16_t *activation_stage,
        uint64_t activation_capacity,
        cudaStream_t stream,
        const uint8_t *fp8_weight = nullptr,
        const float *fp8_scales = nullptr,
        bool fp8_weight_enabled = false,
        bool reuse_staged_activation = false) {
    if (!handle || !weight || !input || !output || !activation_stage || rows == 0u ||
        cols == 0u || columns == 0u ||
        (fp8_weight_enabled && (!fp8_weight || !fp8_scales))) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t activation_elements = 0u;
    if (!checked_mul(cols, columns, &activation_elements) ||
        activation_elements > activation_capacity ||
        activation_elements > std::numeric_limits<uint32_t>::max()) {
        return AXIOM_ERR_BUDGET;
    }
    if (!reuse_staged_activation) {
        f32_to_bf16_stage_kernel<<<
                static_cast<uint32_t>((activation_elements + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(input, activation_stage, activation_elements);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    if (columns == 1u && fp8_weight_enabled) {
        if ((cols & 1u) != 0u) return AXIOM_ERR_UNSUPPORTED_BACKEND;
        fp8_weight_bf16_gemv_m1_f32_kernel<<<rows, kThreads, 0, stream>>>(
                fp8_weight, fp8_scales, activation_stage, output, rows, cols);
        return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    }
    if (columns == 1u && (cols & 1u) == 0u && mtp_gemv_m1_enabled()) {
        bf16_gemv_m1_f32_kernel<<<rows, kThreads, 0, stream>>>(
                weight, activation_stage, output, rows, cols);
        return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    }
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    if constexpr (ConfigureStream) status = cublasSetStream(handle, stream);
    if (status == CUBLAS_STATUS_SUCCESS) {
        constexpr float alpha = 1.0f;
        constexpr float beta = 0.0f;
        status = cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(cols),
                &alpha, weight, CUDA_R_16BF, static_cast<int>(cols),
                activation_stage, CUDA_R_16BF, static_cast<int>(cols),
                &beta, output, CUDA_R_32F, static_cast<int>(rows),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    return cublas_status(status);
}

int mtp_tensor_pointer(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_tensor tensor,
        const uint16_t **output) {
    if (output) *output = nullptr;
    if (!mtp || !output) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_device_buffer *buffer = nullptr;
    int rc = axiom_qwen38_mtp_tensor_buffer_get(mtp, tensor, &buffer);
    void *pointer = nullptr;
    if (rc == AXIOM_OK) rc = buffer_pointer(buffer, &pointer);
    if (rc != AXIOM_OK || !pointer) return rc == AXIOM_OK ? AXIOM_ERR_RUNTIME : rc;
    *output = static_cast<const uint16_t *>(pointer);
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_mtp_compute {
    const axiom_qwen38_mtp *mtp = nullptr;
    axiom_runtime *runtime = nullptr;
    int device = -1;
    uint32_t max_context = 0u;
    uint32_t cache_dtype = 0u;
    axiom_qwen38_rope_profile rope_profile =
            AXIOM_QWEN38_ROPE_PROFILE_INVALID;
    uint32_t committed_position = 0u;
    uint32_t transaction_anchor = 0u;
    uint32_t staged_end_position = 0u;
    axiom_qwen38_mtp_transaction_state transaction_state =
            AXIOM_QWEN38_MTP_TRANSACTION_IDLE;
    cudaStream_t transaction_stream = nullptr;
    bool device_graph_capture_active = false;
    bool device_session_active = false;
    uint32_t device_session_anchor = 0u;
    cudaStream_t device_enqueue_stream = nullptr;
    uint64_t checkpoint_bytes = 0u;
    uint64_t workspace_bytes = 0u;
    uint64_t kv_cache_bytes = 0u;
    uint64_t device_bytes = 0u;
    bool fp8_weight_gemv_enabled = false;
    std::vector<uint8_t> imported_kv_pages;
    cublasHandle_t cublas = nullptr;
    axiom_qwen38_mtp_target_binding target{};
    mtp_weights weights{};
    mtp_derived_fp8_weights derived_fp8_weights{};

    axiom_device_buffer *derived_fp8_weight_buffer = nullptr;
    axiom_device_buffer *derived_fp8_scale_buffer = nullptr;
    axiom_device_buffer *embedding_buffer = nullptr;
    axiom_device_buffer *embedding_norm_buffer = nullptr;
    axiom_device_buffer *hidden_norm_buffer = nullptr;
    axiom_device_buffer *fusion_buffer = nullptr;
    axiom_device_buffer *hidden_buffer = nullptr;
    axiom_device_buffer *residual_buffer = nullptr;
    axiom_device_buffer *norm_buffer = nullptr;
    axiom_device_buffer *q_gate_buffer = nullptr;
    axiom_device_buffer *q_buffer = nullptr;
    axiom_device_buffer *gate_buffer = nullptr;
    axiom_device_buffer *k_buffer = nullptr;
    axiom_device_buffer *v_buffer = nullptr;
    axiom_device_buffer *attention_buffer = nullptr;
    axiom_device_buffer *gated_attention_buffer = nullptr;
    axiom_device_buffer *linear_buffer = nullptr;
    axiom_device_buffer *mlp_gate_buffer = nullptr;
    axiom_device_buffer *mlp_up_buffer = nullptr;
    axiom_device_buffer *final_hidden_buffer = nullptr;
    axiom_device_buffer *logits_buffer = nullptr;
    axiom_device_buffer *activation_stage_buffer = nullptr;
    axiom_device_buffer *cublas_workspace_buffer = nullptr;
    axiom_device_buffer *split_values_buffer = nullptr;
    axiom_device_buffer *split_maxima_buffer = nullptr;
    axiom_device_buffer *split_denominators_buffer = nullptr;
    axiom_device_buffer *top1_values_buffer = nullptr;
    axiom_device_buffer *top1_ids_buffer = nullptr;
    axiom_device_buffer *resolved_position_buffer = nullptr;
    axiom_device_buffer *safe_token_ids_buffer = nullptr;
    axiom_device_buffer *k_cache_buffer = nullptr;
    axiom_device_buffer *v_cache_buffer = nullptr;

    uint8_t *derived_fp8_weight_storage = nullptr;
    float *derived_fp8_scale_storage = nullptr;
    float *embedding = nullptr;
    float *embedding_norm = nullptr;
    float *hidden_norm = nullptr;
    float *fusion = nullptr;
    float *hidden = nullptr;
    float *residual = nullptr;
    float *norm = nullptr;
    float *q_gate = nullptr;
    float *q = nullptr;
    float *gate = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *attention = nullptr;
    float *gated_attention = nullptr;
    float *linear = nullptr;
    float *mlp_gate = nullptr;
    float *mlp_up = nullptr;
    float *final_hidden = nullptr;
    float *logits = nullptr;
    uint16_t *activation_stage = nullptr;
    uint64_t activation_stage_capacity = 0u;
    void *cublas_workspace = nullptr;
    float *split_values = nullptr;
    float *split_maxima = nullptr;
    float *split_denominators = nullptr;
    float *top1_values = nullptr;
    uint32_t *top1_ids = nullptr;
    uint32_t *resolved_position = nullptr;
    uint32_t *safe_token_ids = nullptr;
    void *k_cache = nullptr;
    void *v_cache = nullptr;
};

namespace {

uint32_t kv_logical_pages(const axiom_qwen38_mtp_compute *compute) {
    return compute
            ? compute->max_context / kKvPageTokens +
                    (compute->max_context % kKvPageTokens != 0u ? 1u : 0u)
            : 0u;
}

int mtp_kv_page_args(
        const axiom_qwen38_mtp_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes) {
    if (!compute || layer != 0u || !host_page || host_page_bytes != kKvPageBytes ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || compute->device_session_active ||
        compute->device_enqueue_stream || !compute->k_cache || !compute->v_cache ||
        (compute->cache_dtype != AXIOM_TENSOR_DTYPE_BF16 &&
         compute->cache_dtype != AXIOM_TENSOR_DTYPE_F8_E4M3) ||
        logical_page >= kv_logical_pages(compute) ||
        compute->imported_kv_pages.size() != kv_logical_pages(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

bool imported_prefix_covers(
        const axiom_qwen38_mtp_compute *compute,
        uint32_t position) {
    if (!compute || position > compute->max_context) return false;
    const uint32_t required_pages = position / kKvPageTokens +
            (position % kKvPageTokens != 0u ? 1u : 0u);
    if (required_pages > compute->imported_kv_pages.size()) return false;
    for (uint32_t page = 0u; page < required_pages; ++page) {
        if (compute->imported_kv_pages[page] == 0u) return false;
    }
    return true;
}

bool e4m3_record_finite(
        const uint8_t *record,
        size_t valid_elements) {
    if (!record || valid_elements > kKvPageElements) return false;
    const uint8_t *v_plane = record + kKvPageBytes / 2u;
    for (size_t index = 0u; index < valid_elements; ++index) {
        if ((record[index] & 0x7fu) == 0x7fu ||
            (v_plane[index] & 0x7fu) == 0x7fu) {
            return false;
        }
    }
    return true;
}

void destroy_buffer(axiom_device_buffer **buffer, void **pointer = nullptr) {
    if (pointer) *pointer = nullptr;
    if (!buffer) return;
    axiom_device_buffer_destroy(*buffer);
    *buffer = nullptr;
}

void destroy_workspace(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return;
    destroy_buffer(&compute->derived_fp8_scale_buffer,
                   reinterpret_cast<void **>(&compute->derived_fp8_scale_storage));
    destroy_buffer(&compute->derived_fp8_weight_buffer,
                   reinterpret_cast<void **>(&compute->derived_fp8_weight_storage));
    compute->derived_fp8_weights = {};
    destroy_buffer(&compute->v_cache_buffer, &compute->v_cache);
    destroy_buffer(&compute->k_cache_buffer, &compute->k_cache);
    destroy_buffer(&compute->resolved_position_buffer,
                   reinterpret_cast<void **>(&compute->resolved_position));
    destroy_buffer(&compute->safe_token_ids_buffer,
                   reinterpret_cast<void **>(&compute->safe_token_ids));
    destroy_buffer(&compute->top1_ids_buffer, reinterpret_cast<void **>(&compute->top1_ids));
    destroy_buffer(&compute->top1_values_buffer, reinterpret_cast<void **>(&compute->top1_values));
    destroy_buffer(&compute->split_denominators_buffer,
                   reinterpret_cast<void **>(&compute->split_denominators));
    destroy_buffer(&compute->split_maxima_buffer,
                   reinterpret_cast<void **>(&compute->split_maxima));
    destroy_buffer(&compute->split_values_buffer,
                   reinterpret_cast<void **>(&compute->split_values));
    destroy_buffer(&compute->cublas_workspace_buffer, &compute->cublas_workspace);
    destroy_buffer(&compute->activation_stage_buffer,
                   reinterpret_cast<void **>(&compute->activation_stage));
    destroy_buffer(&compute->logits_buffer, reinterpret_cast<void **>(&compute->logits));
    destroy_buffer(&compute->final_hidden_buffer,
                   reinterpret_cast<void **>(&compute->final_hidden));
    destroy_buffer(&compute->mlp_up_buffer, reinterpret_cast<void **>(&compute->mlp_up));
    destroy_buffer(&compute->mlp_gate_buffer, reinterpret_cast<void **>(&compute->mlp_gate));
    destroy_buffer(&compute->linear_buffer, reinterpret_cast<void **>(&compute->linear));
    destroy_buffer(&compute->gated_attention_buffer,
                   reinterpret_cast<void **>(&compute->gated_attention));
    destroy_buffer(&compute->attention_buffer, reinterpret_cast<void **>(&compute->attention));
    destroy_buffer(&compute->v_buffer, reinterpret_cast<void **>(&compute->v));
    destroy_buffer(&compute->k_buffer, reinterpret_cast<void **>(&compute->k));
    destroy_buffer(&compute->gate_buffer, reinterpret_cast<void **>(&compute->gate));
    destroy_buffer(&compute->q_buffer, reinterpret_cast<void **>(&compute->q));
    destroy_buffer(&compute->q_gate_buffer, reinterpret_cast<void **>(&compute->q_gate));
    destroy_buffer(&compute->norm_buffer, reinterpret_cast<void **>(&compute->norm));
    destroy_buffer(&compute->residual_buffer, reinterpret_cast<void **>(&compute->residual));
    destroy_buffer(&compute->hidden_buffer, reinterpret_cast<void **>(&compute->hidden));
    destroy_buffer(&compute->fusion_buffer, reinterpret_cast<void **>(&compute->fusion));
    destroy_buffer(&compute->hidden_norm_buffer, reinterpret_cast<void **>(&compute->hidden_norm));
    destroy_buffer(&compute->embedding_norm_buffer,
                   reinterpret_cast<void **>(&compute->embedding_norm));
    destroy_buffer(&compute->embedding_buffer, reinterpret_cast<void **>(&compute->embedding));
    compute->workspace_bytes = 0u;
    compute->kv_cache_bytes = 0u;
    compute->device_bytes = 0u;
}

int allocate_buffer(
        axiom_qwen38_mtp_compute *compute,
        axiom_device_buffer **buffer,
        void **pointer,
        uint64_t bytes,
        uint64_t *category) {
    if (!compute || !buffer || !pointer || !category || bytes == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = axiom_device_buffer_create(compute->runtime, buffer, bytes);
    if (rc == AXIOM_OK) rc = buffer_pointer(*buffer, pointer);
    if (rc == AXIOM_OK) rc = add_bytes(category, bytes);
    return rc;
}

int load_weights(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = mtp_tensor_pointer(compute->mtp, AXIOM_QWEN38_MTP_TENSOR_FC_WEIGHT,
                                &compute->weights.fc);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_INPUT_LAYERNORM_WEIGHT,
            &compute->weights.input_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_DOWN_PROJ_WEIGHT,
            &compute->weights.mlp_down);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_GATE_PROJ_WEIGHT,
            &compute->weights.mlp_gate);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_UP_PROJ_WEIGHT,
            &compute->weights.mlp_up);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_POST_ATTENTION_LAYERNORM_WEIGHT,
            &compute->weights.post_attention_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_NORM_WEIGHT,
            &compute->weights.k_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_PROJ_WEIGHT,
            &compute->weights.k_proj);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_O_PROJ_WEIGHT,
            &compute->weights.o_proj);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_NORM_WEIGHT,
            &compute->weights.q_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_PROJ_WEIGHT,
            &compute->weights.q_proj);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_V_PROJ_WEIGHT,
            &compute->weights.v_proj);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_NORM_WEIGHT,
            &compute->weights.final_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_EMBEDDING_WEIGHT,
            &compute->weights.pre_fc_embedding_norm);
    if (rc == AXIOM_OK) rc = mtp_tensor_pointer(
            compute->mtp, AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_HIDDEN_WEIGHT,
            &compute->weights.pre_fc_hidden_norm);
    return rc;
}

int bind_derived_fp8_weights(axiom_qwen38_mtp_compute *compute) {
    if (!compute || !compute->derived_fp8_weight_storage ||
        !compute->derived_fp8_scale_storage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t weight_offset = 0u;
    uint64_t scale_offset = 0u;
    auto bind = [&](mtp_fp8_projection *projection,
                    uint32_t rows, uint32_t cols) {
        uint64_t elements = 0u;
        uint64_t next_weight_offset = 0u;
        uint64_t next_scale_offset = 0u;
        if (!projection || !checked_mul(rows, cols, &elements) ||
            !checked_add(weight_offset, elements, &next_weight_offset) ||
            !checked_add(scale_offset, rows, &next_scale_offset) ||
            next_weight_offset > kDerivedFp8WeightElements ||
            next_scale_offset > kDerivedFp8ScaleElements) {
            return false;
        }
        projection->weight = compute->derived_fp8_weight_storage + weight_offset;
        projection->scales = compute->derived_fp8_scale_storage + scale_offset;
        weight_offset = next_weight_offset;
        scale_offset = next_scale_offset;
        return true;
    };
    const bool valid =
            bind(&compute->derived_fp8_weights.fc, kHidden, kFusionInput) &&
            bind(&compute->derived_fp8_weights.q_proj, kQGateDim, kHidden) &&
            bind(&compute->derived_fp8_weights.k_proj, kKvDim, kHidden) &&
            bind(&compute->derived_fp8_weights.v_proj, kKvDim, kHidden) &&
            bind(&compute->derived_fp8_weights.o_proj, kHidden, kQDim) &&
            bind(&compute->derived_fp8_weights.mlp_gate, kIntermediate, kHidden) &&
            bind(&compute->derived_fp8_weights.mlp_up, kIntermediate, kHidden) &&
            bind(&compute->derived_fp8_weights.mlp_down, kHidden, kIntermediate);
    return valid && weight_offset == kDerivedFp8WeightElements &&
                    scale_offset == kDerivedFp8ScaleElements
            ? AXIOM_OK : AXIOM_ERR_RUNTIME;
}

int allocate_derived_fp8_weights(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!compute->fp8_weight_gemv_enabled) return AXIOM_OK;
    int rc = allocate_buffer(
            compute, &compute->derived_fp8_weight_buffer,
            reinterpret_cast<void **>(&compute->derived_fp8_weight_storage),
            kDerivedFp8WeightElements * sizeof(uint8_t),
            &compute->workspace_bytes);
    if (rc == AXIOM_OK) {
        rc = allocate_buffer(
                compute, &compute->derived_fp8_scale_buffer,
                reinterpret_cast<void **>(&compute->derived_fp8_scale_storage),
                kDerivedFp8ScaleElements * sizeof(float),
                &compute->workspace_bytes);
    }
    if (rc == AXIOM_OK) rc = bind_derived_fp8_weights(compute);
    return rc;
}

int launch_quantize_bf16_rows_e4m3(
        const uint16_t *input,
        const mtp_fp8_projection &projection,
        uint32_t rows,
        uint32_t cols,
        uint32_t *status,
        cudaStream_t stream) {
    if (!input || !projection.weight || !projection.scales ||
        rows == 0u || cols == 0u || !status) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    quantize_bf16_rows_e4m3_kernel<<<rows, kThreads, 0, stream>>>(
            input, projection.weight, projection.scales, rows, cols, status);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int initialize_derived_fp8_weights(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!compute->fp8_weight_gemv_enabled) return AXIOM_OK;
    if (!compute->resolved_position) return AXIOM_ERR_RUNTIME;
    const cudaStream_t stream = nullptr;
    cudaError_t status = cudaMemsetAsync(
            compute->resolved_position, 0, sizeof(uint32_t), stream);
    int rc = cuda_status(status);
    auto quantize = [&](const uint16_t *input,
                        const mtp_fp8_projection &projection,
                        uint32_t rows, uint32_t cols) {
        return launch_quantize_bf16_rows_e4m3(
                input, projection, rows, cols,
                compute->resolved_position, stream);
    };
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.fc, compute->derived_fp8_weights.fc,
            kHidden, kFusionInput);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.q_proj, compute->derived_fp8_weights.q_proj,
            kQGateDim, kHidden);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.k_proj, compute->derived_fp8_weights.k_proj,
            kKvDim, kHidden);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.v_proj, compute->derived_fp8_weights.v_proj,
            kKvDim, kHidden);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.o_proj, compute->derived_fp8_weights.o_proj,
            kHidden, kQDim);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.mlp_gate, compute->derived_fp8_weights.mlp_gate,
            kIntermediate, kHidden);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.mlp_up, compute->derived_fp8_weights.mlp_up,
            kIntermediate, kHidden);
    if (rc == AXIOM_OK) rc = quantize(
            compute->weights.mlp_down, compute->derived_fp8_weights.mlp_down,
            kHidden, kIntermediate);
    uint32_t host_status = static_cast<uint32_t>(AXIOM_OK);
    if (rc == AXIOM_OK) {
        status = cudaMemcpyAsync(
                &host_status, compute->resolved_position, sizeof(host_status),
                cudaMemcpyDeviceToHost, stream);
        rc = cuda_status(status);
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaStreamSynchronize(stream));
    if (rc == AXIOM_OK && host_status != static_cast<uint32_t>(AXIOM_OK)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    return rc;
}

template <bool ConfigureStream>
int gemm_native_mtp_projection(
        axiom_qwen38_mtp_compute *compute,
        const uint16_t *weight,
        const mtp_fp8_projection &derived_fp8_weight,
        uint32_t rows,
        uint32_t cols,
        const float *input,
        float *output,
        uint32_t columns,
        cudaStream_t stream,
        bool reuse_staged_activation = false) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    return gemm_bf16_weight_bf16_f32<ConfigureStream>(
            compute->cublas, weight, rows, cols, input, output, columns,
            compute->activation_stage, compute->activation_stage_capacity, stream,
            derived_fp8_weight.weight, derived_fp8_weight.scales,
            compute->fp8_weight_gemv_enabled, reuse_staged_activation);
}

int allocate_workspace(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t hidden_bytes = static_cast<uint64_t>(kHidden) * kMaxColumns * sizeof(float);
    const uint64_t fusion_bytes = static_cast<uint64_t>(kFusionInput) * kMaxColumns * sizeof(float);
    const uint64_t q_gate_bytes = static_cast<uint64_t>(kQGateDim) * kMaxColumns * sizeof(float);
    const uint64_t q_bytes = static_cast<uint64_t>(kQDim) * kMaxColumns * sizeof(float);
    const uint64_t kv_bytes = static_cast<uint64_t>(kKvDim) * kMaxColumns * sizeof(float);
    const uint64_t intermediate_bytes =
            static_cast<uint64_t>(kIntermediate) * kMaxColumns * sizeof(float);
    const uint64_t logits_bytes = static_cast<uint64_t>(kVocab) * kMaxColumns * sizeof(float);
    const uint64_t activation_elements = static_cast<uint64_t>(kIntermediate) * kMaxColumns;
    const uint64_t activation_bytes = activation_elements * sizeof(uint16_t);
    const uint64_t split_stats = static_cast<uint64_t>(kMaxColumns) * kKvHeads *
            kAttentionSplits * kGqaHeads;
    const uint64_t split_values_bytes = split_stats * kHeadDim * sizeof(float);
    const uint64_t split_stats_bytes = split_stats * sizeof(float);
    const uint64_t cache_element_bytes =
            compute->cache_dtype == AXIOM_TENSOR_DTYPE_BF16
                    ? sizeof(uint16_t) : sizeof(uint8_t);
    uint64_t cache_elements = 0u;
    uint64_t cache_bytes = 0u;
    if (!checked_mul(compute->max_context, kKvDim, &cache_elements) ||
        !checked_mul(cache_elements, cache_element_bytes, &cache_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    auto workspace = [&](axiom_device_buffer **buffer, void **pointer, uint64_t bytes) {
        return allocate_buffer(compute, buffer, pointer, bytes, &compute->workspace_bytes);
    };
    int rc = workspace(&compute->embedding_buffer,
                       reinterpret_cast<void **>(&compute->embedding), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->embedding_norm_buffer,
            reinterpret_cast<void **>(&compute->embedding_norm), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->hidden_norm_buffer,
            reinterpret_cast<void **>(&compute->hidden_norm), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->fusion_buffer,
            reinterpret_cast<void **>(&compute->fusion), fusion_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->hidden_buffer,
            reinterpret_cast<void **>(&compute->hidden), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->residual_buffer,
            reinterpret_cast<void **>(&compute->residual), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->norm_buffer,
            reinterpret_cast<void **>(&compute->norm), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->q_gate_buffer,
            reinterpret_cast<void **>(&compute->q_gate), q_gate_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->q_buffer,
            reinterpret_cast<void **>(&compute->q), q_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->gate_buffer,
            reinterpret_cast<void **>(&compute->gate), q_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->k_buffer,
            reinterpret_cast<void **>(&compute->k), kv_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->v_buffer,
            reinterpret_cast<void **>(&compute->v), kv_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->attention_buffer,
            reinterpret_cast<void **>(&compute->attention), q_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->gated_attention_buffer,
            reinterpret_cast<void **>(&compute->gated_attention), q_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->linear_buffer,
            reinterpret_cast<void **>(&compute->linear), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->mlp_gate_buffer,
            reinterpret_cast<void **>(&compute->mlp_gate), intermediate_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->mlp_up_buffer,
            reinterpret_cast<void **>(&compute->mlp_up), intermediate_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->final_hidden_buffer,
            reinterpret_cast<void **>(&compute->final_hidden), hidden_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->logits_buffer,
            reinterpret_cast<void **>(&compute->logits), logits_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->activation_stage_buffer,
            reinterpret_cast<void **>(&compute->activation_stage), activation_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->cublas_workspace_buffer,
            &compute->cublas_workspace, kCublasWorkspaceBytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->split_values_buffer,
            reinterpret_cast<void **>(&compute->split_values), split_values_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->split_maxima_buffer,
            reinterpret_cast<void **>(&compute->split_maxima), split_stats_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->split_denominators_buffer,
            reinterpret_cast<void **>(&compute->split_denominators), split_stats_bytes);
    if (rc == AXIOM_OK) rc = workspace(&compute->top1_values_buffer,
            reinterpret_cast<void **>(&compute->top1_values),
            static_cast<uint64_t>(kTop1Blocks) * sizeof(float));
    if (rc == AXIOM_OK) rc = workspace(&compute->top1_ids_buffer,
            reinterpret_cast<void **>(&compute->top1_ids),
            static_cast<uint64_t>(kTop1Blocks) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = workspace(&compute->resolved_position_buffer,
            reinterpret_cast<void **>(&compute->resolved_position), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = workspace(&compute->safe_token_ids_buffer,
            reinterpret_cast<void **>(&compute->safe_token_ids),
            static_cast<uint64_t>(kMaxColumns) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_derived_fp8_weights(compute);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->k_cache_buffer,
            &compute->k_cache, cache_bytes, &compute->kv_cache_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->v_cache_buffer,
            &compute->v_cache, cache_bytes, &compute->kv_cache_bytes);
    compute->activation_stage_capacity = activation_elements;
    if (rc == AXIOM_OK &&
        (!checked_add(compute->checkpoint_bytes, compute->workspace_bytes,
                      &compute->device_bytes) ||
         !checked_add(compute->device_bytes, compute->kv_cache_bytes,
                      &compute->device_bytes))) {
        rc = AXIOM_ERR_BUDGET;
    }
    if (rc == AXIOM_OK) {
        rc = cublas_status(cublasSetWorkspace(
                compute->cublas, compute->cublas_workspace,
                static_cast<size_t>(kCublasWorkspaceBytes)));
    }
    return rc;
}

template <bool DevicePosition>
int enqueue_attention(
        axiom_qwen38_mtp_compute *compute,
        uint32_t first_position,
        const uint32_t *first_position_device,
        uint32_t *async_status,
        uint32_t columns,
        cudaStream_t stream) {
    const uint64_t q_count = static_cast<uint64_t>(kQDim) * columns;
    const uint64_t split_count = static_cast<uint64_t>(columns) * kKvHeads *
            kAttentionSplits * kGqaHeads;
    const uint64_t split_value_count = split_count * kHeadDim;
    cudaError_t status = cudaMemsetAsync(
            compute->split_denominators, 0,
            static_cast<size_t>(split_count * sizeof(float)), stream);
    if (status != cudaSuccess) return cuda_status(status);
    if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_BF16) {
        qk_norm_rope_store_kernel<uint16_t, DevicePosition><<<
                dim3(kHeads + kKvHeads, columns), kThreads, 0, stream>>>(
                compute->q, compute->k, compute->v,
                compute->weights.q_norm, compute->weights.k_norm,
                static_cast<uint16_t *>(compute->k_cache),
                static_cast<uint16_t *>(compute->v_cache),
                first_position, first_position_device, async_status,
                columns, compute->max_context,
                compute->rope_profile == AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        causal_gqa_splitk_kernel<uint16_t, DevicePosition><<<
                dim3(kKvHeads, columns, kAttentionSplits), kAttentionThreads, 0, stream>>>(
                compute->q, static_cast<const uint16_t *>(compute->k_cache),
                static_cast<const uint16_t *>(compute->v_cache),
                compute->split_values, compute->split_maxima,
                compute->split_denominators, first_position,
                first_position_device, async_status, columns);
    } else {
        qk_norm_rope_store_kernel<uint8_t, DevicePosition><<<
                dim3(kHeads + kKvHeads, columns), kThreads, 0, stream>>>(
                compute->q, compute->k, compute->v,
                compute->weights.q_norm, compute->weights.k_norm,
                static_cast<uint8_t *>(compute->k_cache),
                static_cast<uint8_t *>(compute->v_cache),
                first_position, first_position_device, async_status,
                columns, compute->max_context,
                compute->rope_profile == AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        causal_gqa_splitk_kernel<uint8_t, DevicePosition><<<
                dim3(kKvHeads, columns, kAttentionSplits), kAttentionThreads, 0, stream>>>(
                compute->q, static_cast<const uint8_t *>(compute->k_cache),
                static_cast<const uint8_t *>(compute->v_cache),
                compute->split_values, compute->split_maxima,
                compute->split_denominators, first_position,
                first_position_device, async_status, columns);
    }
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    merge_gqa_splitk_kernel<<<
            dim3(kKvHeads, columns), kAttentionThreads, 0, stream>>>(
            compute->split_values, compute->split_maxima,
            compute->split_denominators, compute->attention, columns);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    (void)q_count;
    (void)split_value_count;
    return AXIOM_OK;
}

struct mtp_forward_view {
    const uint32_t *token_ids_device;
    const float *prior_hidden_device;
    float *out_hidden_device;
    float *out_logits_device;
    uint32_t *out_top1_device;
    uint32_t first_position;
    const uint32_t *first_position_device;
    uint32_t *async_status_device;
    uint32_t columns;
    void *stream;
};

template <bool DevicePosition>
int enqueue_forward_impl(
        axiom_qwen38_mtp_compute *compute,
        const mtp_forward_view &request) {
    const uint32_t columns = request.columns;
    const cudaStream_t stream = static_cast<cudaStream_t>(request.stream);
    const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * columns;
    const uint64_t q_count = static_cast<uint64_t>(kQDim) * columns;
    const uint64_t intermediate_count = static_cast<uint64_t>(kIntermediate) * columns;
    int rc = compute->target.embed_f32_device(
            compute->target.user_data, request.token_ids_device,
            compute->embedding, columns, request.stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.pre_fc_embedding_norm, compute->embedding,
            compute->embedding_norm, kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.pre_fc_hidden_norm, request.prior_hidden_device,
            compute->hidden_norm, kHidden, columns, stream);
    if (rc == AXIOM_OK) {
        const uint64_t fusion_count = static_cast<uint64_t>(kFusionInput) * columns;
        pack_fusion_kernel<<<
                static_cast<uint32_t>((fusion_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->embedding_norm, compute->hidden_norm,
                                      compute->fusion, columns);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.fc, compute->derived_fp8_weights.fc,
            kHidden, kFusionInput, compute->fusion, compute->hidden,
            columns, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(compute->hidden, hidden_count, stream);
    if (rc == AXIOM_OK) {
        const cudaError_t status = cudaMemcpyAsync(
                compute->residual, compute->hidden,
                static_cast<size_t>(hidden_count * sizeof(float)),
                cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) rc = cuda_status(status);
    }
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.input_norm, compute->hidden, compute->norm,
            kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.q_proj,
            compute->derived_fp8_weights.q_proj, kQGateDim, kHidden,
            compute->norm, compute->q_gate, columns, stream);
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.k_proj,
            compute->derived_fp8_weights.k_proj, kKvDim, kHidden,
            compute->norm, compute->k, columns, stream, true);
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.v_proj,
            compute->derived_fp8_weights.v_proj, kKvDim, kHidden,
            compute->norm, compute->v, columns, stream, true);
    if (rc == AXIOM_OK) {
        const uint64_t split_count = q_count + static_cast<uint64_t>(kKvDim) * columns;
        split_q_gate_round_kv_kernel<<<
                static_cast<uint32_t>((split_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->q_gate, compute->q, compute->gate,
                                      compute->k, compute->v, columns);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = enqueue_attention<DevicePosition>(
            compute, request.first_position, request.first_position_device,
            request.async_status_device, columns, stream);
    if (rc == AXIOM_OK) {
        sigmoid_gate_kernel<<<
                static_cast<uint32_t>((q_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->attention, compute->gate,
                                      compute->gated_attention, q_count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.o_proj,
            compute->derived_fp8_weights.o_proj, kHidden, kQDim,
            compute->gated_attention, compute->linear, columns, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(compute->linear, hidden_count, stream);
    if (rc == AXIOM_OK) {
        residual_add_kernel<<<
                static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->residual, compute->linear,
                                      compute->hidden, hidden_count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.post_attention_norm, compute->hidden, compute->norm,
            kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.mlp_gate,
            compute->derived_fp8_weights.mlp_gate, kIntermediate, kHidden,
            compute->norm, compute->mlp_gate, columns, stream);
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.mlp_up,
            compute->derived_fp8_weights.mlp_up, kIntermediate, kHidden,
            compute->norm, compute->mlp_up, columns, stream, true);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->mlp_gate, intermediate_count, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->mlp_up, intermediate_count, stream);
    if (rc == AXIOM_OK) {
        silu_multiply_kernel<<<
                static_cast<uint32_t>((intermediate_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->mlp_gate, compute->mlp_up,
                                      intermediate_count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = gemm_native_mtp_projection<!DevicePosition>(
            compute, compute->weights.mlp_down,
            compute->derived_fp8_weights.mlp_down, kHidden, kIntermediate,
            compute->mlp_gate, compute->linear, columns, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(compute->linear, hidden_count, stream);
    if (rc == AXIOM_OK) {
        residual_add_kernel<<<
                static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->hidden, compute->linear,
                                      compute->residual, hidden_count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.final_norm, compute->residual, compute->final_hidden,
            kHidden, columns, stream);
    if (rc == AXIOM_OK) {
        const cudaError_t status = cudaMemcpyAsync(
                request.out_hidden_device, compute->final_hidden,
                static_cast<size_t>(hidden_count * sizeof(float)),
                cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) rc = cuda_status(status);
    }
    float *logits = request.out_logits_device
            ? request.out_logits_device : compute->logits;
    if (rc == AXIOM_OK && (request.out_logits_device || request.out_top1_device)) {
        rc = compute->target.lm_head_f32_device(
                compute->target.user_data, compute->final_hidden, logits,
                columns, request.stream);
    }
    for (uint32_t column = 0u;
         column < columns && rc == AXIOM_OK && request.out_top1_device; ++column) {
        top1_stage1_kernel<<<kTop1Blocks, kThreads, 0, stream>>>(
                logits + static_cast<uint64_t>(column) * kVocab,
                compute->top1_values, compute->top1_ids);
        if (cudaGetLastError() != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
            break;
        }
        top1_stage2_kernel<<<1u, kThreads, 0, stream>>>(
                compute->top1_values, compute->top1_ids,
                request.out_top1_device + column,
                request.async_status_device);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    return rc;
}

int enqueue_forward(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_forward_request *request) {
    const mtp_forward_view view{
            request->token_ids_device,
            request->prior_hidden_device,
            request->out_hidden_device,
            request->out_logits_device,
            request->out_top1_device,
            request->first_position,
            nullptr,
            nullptr,
            request->columns,
            request->stream};
    return enqueue_forward_impl<false>(compute, view);
}

int enqueue_forward_device(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_forward_request_v1 *request) {
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    sanitize_device_tokens_kernel<<<1u, kMaxColumns, 0, stream>>>(
            request->token_ids_device, compute->safe_token_ids,
            request->columns, request->async_status_device);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    resolve_device_position_kernel<<<1u, 1u, 0, stream>>>(
            request->base_position_device, request->position_offset,
            request->columns, compute->max_context,
            request->async_status_device, compute->resolved_position);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    const mtp_forward_view view{
            compute->safe_token_ids,
            request->prior_hidden_device,
            request->out_hidden_device,
            request->out_logits_device,
            request->out_top1_device,
            0u,
            compute->resolved_position,
            request->async_status_device,
            request->columns,
            request->stream};
    return enqueue_forward_impl<true>(compute, view);
}

int enqueue_kv_catchup_device(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_kv_catchup_request_v1 *request) {
    const uint32_t columns = request->columns;
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * columns;
    const uint64_t kv_count = static_cast<uint64_t>(kKvDim) * columns;

    sanitize_device_tokens_kernel<<<1u, kMaxColumns, 0, stream>>>(
            request->token_ids_device, compute->safe_token_ids,
            columns, request->async_status_device);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    resolve_device_position_kernel<<<1u, 1u, 0, stream>>>(
            request->base_position_device, request->position_offset,
            columns, compute->max_context,
            request->async_status_device, compute->resolved_position);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    int rc = compute->target.embed_f32_device(
            compute->target.user_data, compute->safe_token_ids,
            compute->embedding, columns, request->stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.pre_fc_embedding_norm, compute->embedding,
            compute->embedding_norm, kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.pre_fc_hidden_norm, request->prior_hidden_device,
            compute->hidden_norm, kHidden, columns, stream);
    if (rc == AXIOM_OK) {
        const uint64_t fusion_count = static_cast<uint64_t>(kFusionInput) * columns;
        pack_fusion_kernel<<<
                static_cast<uint32_t>((fusion_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->embedding_norm, compute->hidden_norm,
                                      compute->fusion, columns);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32<false>(
            compute->cublas, compute->weights.fc, kHidden, kFusionInput,
            compute->fusion, compute->hidden, columns, compute->activation_stage,
            compute->activation_stage_capacity, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->hidden, hidden_count, stream);
    if (rc == AXIOM_OK) {
        const cudaError_t status = cudaMemcpyAsync(
                compute->residual, compute->hidden,
                static_cast<size_t>(hidden_count * sizeof(float)),
                cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) rc = cuda_status(status);
    }
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->weights.input_norm, compute->hidden, compute->norm,
            kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32<false>(
            compute->cublas, compute->weights.k_proj, kKvDim, kHidden,
            compute->norm, compute->k, columns, compute->activation_stage,
            compute->activation_stage_capacity, stream);
    if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32<false>(
            compute->cublas, compute->weights.v_proj, kKvDim, kHidden,
            compute->norm, compute->v, columns, compute->activation_stage,
            compute->activation_stage_capacity, stream, nullptr, nullptr,
            false, true);
    if (rc == AXIOM_OK) {
        const uint64_t q_count = static_cast<uint64_t>(kQDim) * columns;
        const uint64_t split_count = q_count + kv_count;
        split_q_gate_round_kv_kernel<<<
                static_cast<uint32_t>((split_count + kThreads - 1u) / kThreads),
                kThreads, 0, stream>>>(compute->q_gate, compute->q,
                                      compute->gate, compute->k, compute->v,
                                      columns);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        const bool yarn_enabled = compute->rope_profile ==
                AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M;
        if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_BF16) {
            qk_norm_rope_store_kernel<uint16_t, true><<<
                    dim3(kHeads + kKvHeads, columns), kThreads, 0, stream>>>(
                    compute->q, compute->k, compute->v,
                    compute->weights.q_norm, compute->weights.k_norm,
                    static_cast<uint16_t *>(compute->k_cache),
                    static_cast<uint16_t *>(compute->v_cache),
                    0u, compute->resolved_position,
                    request->async_status_device,
                    columns, compute->max_context, yarn_enabled);
        } else if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_F8_E4M3) {
            qk_norm_rope_store_kernel<uint8_t, true><<<
                    dim3(kHeads + kKvHeads, columns), kThreads, 0, stream>>>(
                    compute->q, compute->k, compute->v,
                    compute->weights.q_norm, compute->weights.k_norm,
                    static_cast<uint8_t *>(compute->k_cache),
                    static_cast<uint8_t *>(compute->v_cache),
                    0u, compute->resolved_position,
                    request->async_status_device,
                    columns, compute->max_context, yarn_enabled);
        } else {
            return AXIOM_ERR_UNSUPPORTED_BACKEND;
        }
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    return rc;
}

}  // namespace

extern "C" int axiom_qwen38_mtp_compute_create_v1(
        const axiom_qwen38_mtp *mtp,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_mtp_compute_config *config,
        uint64_t config_bytes,
        const axiom_qwen38_mtp_target_binding *target,
        axiom_qwen38_mtp_compute **out) {
    if (out) *out = nullptr;
    if (!config || config_bytes != sizeof(*config) ||
        config->abi_version != AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION ||
        config->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_mtp_compute_config_v2 config_v2{};
    config_v2.abi_version = AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION;
    config_v2.struct_size = sizeof(config_v2);
    config_v2.max_context = config->max_context;
    config_v2.cache_dtype = config->cache_dtype;
    config_v2.rope_profile = AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M;
    return axiom_qwen38_mtp_compute_create_v2(
            mtp, runtime, device, &config_v2, sizeof(config_v2), target, out);
}

extern "C" int axiom_qwen38_mtp_compute_create_v2(
        const axiom_qwen38_mtp *mtp,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_mtp_compute_config_v2 *config,
        uint64_t config_bytes,
        const axiom_qwen38_mtp_target_binding *target,
        axiom_qwen38_mtp_compute **out) {
    if (out) *out = nullptr;
    if (!mtp || !runtime || device < 0 || !config || !target || !out ||
        config_bytes != sizeof(*config) ||
        config->abi_version != AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION ||
        config->struct_size != sizeof(*config) ||
        target->abi_version != AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION ||
        config->max_context == 0u || !target->embed_f32_device ||
        !target->lm_head_f32_device ||
        config->reserved != 0u || config->flags != 0u || target->flags != 0u ||
        (config->rope_profile != AXIOM_QWEN38_ROPE_PROFILE_NATIVE_262K &&
         config->rope_profile != AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M) ||
        (config->rope_profile == AXIOM_QWEN38_ROPE_PROFILE_NATIVE_262K &&
         config->max_context > 262144u) ||
        (config->rope_profile == AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M &&
         config->max_context > 1048576u) ||
        (config->cache_dtype != AXIOM_TENSOR_DTYPE_BF16 &&
         config->cache_dtype != AXIOM_TENSOR_DTYPE_F8_E4M3)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t runtime_device = 0u;
    int rc = axiom_runtime_device_id(runtime, &runtime_device);
    if (rc != AXIOM_OK) return rc;
    if (runtime_device != static_cast<uint32_t>(device)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_mtp_compute *compute = new (std::nothrow) axiom_qwen38_mtp_compute();
    if (!compute) return AXIOM_ERR_BUDGET;
    compute->mtp = mtp;
    compute->runtime = runtime;
    compute->device = device;
    compute->max_context = config->max_context;
    compute->cache_dtype = config->cache_dtype;
    compute->rope_profile =
            static_cast<axiom_qwen38_rope_profile>(config->rope_profile);
    compute->fp8_weight_gemv_enabled = mtp_fp8_weight_gemv_enabled();
    compute->target = *target;
    compute->checkpoint_bytes = axiom_qwen38_mtp_device_bytes(mtp);
    try {
        compute->imported_kv_pages.assign(kv_logical_pages(compute), 0u);
    } catch (...) {
        delete compute;
        return AXIOM_ERR_BUDGET;
    }
    rc = cublas_status(cublasCreate(&compute->cublas));
    if (rc == AXIOM_OK) {
        rc = cublas_status(cublasSetPointerMode(
                compute->cublas, CUBLAS_POINTER_MODE_HOST));
    }
    if (rc == AXIOM_OK) rc = load_weights(compute);
    if (rc == AXIOM_OK) rc = allocate_workspace(compute);
    if (rc == AXIOM_OK) rc = initialize_derived_fp8_weights(compute);
    if (rc != AXIOM_OK) {
        axiom_qwen38_mtp_compute_destroy(compute);
        return rc;
    }
    *out = compute;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_mtp_compute_destroy(axiom_qwen38_mtp_compute *compute) {
    if (!compute) return;
    if (compute->device >= 0) (void)cudaSetDevice(compute->device);
    destroy_workspace(compute);
    if (compute->cublas) (void)cublasDestroy(compute->cublas);
    compute->cublas = nullptr;
    delete compute;
}

extern "C" int axiom_qwen38_mtp_compute_reset(axiom_qwen38_mtp_compute *compute) {
    if (!compute || compute->transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_OPEN ||
        compute->device_graph_capture_active || compute->device_session_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = 0u;
    compute->transaction_anchor = 0u;
    compute->staged_end_position = 0u;
    compute->transaction_stream = nullptr;
    compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_IDLE;
    compute->device_session_anchor = 0u;
    compute->device_enqueue_stream = nullptr;
    std::fill(compute->imported_kv_pages.begin(),
              compute->imported_kv_pages.end(), 0u);
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_kv_page_export(
        const axiom_qwen38_mtp_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        void *host_page,
        uint64_t host_page_bytes) {
    int rc = mtp_kv_page_args(
            compute, layer, logical_page, host_page, host_page_bytes);
    const uint64_t page_start = static_cast<uint64_t>(logical_page) * kKvPageTokens;
    if (rc != AXIOM_OK || page_start >= compute->committed_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc != AXIOM_OK) return rc;

    const uint64_t valid_tokens = std::min<uint64_t>(
            kKvPageTokens, compute->committed_position - page_start);
    const size_t valid_elements = static_cast<size_t>(valid_tokens * kKvDim);
    std::vector<uint8_t> record;
    try {
        record.assign(static_cast<size_t>(kKvPageBytes), 0u);
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    uint8_t *record_v = record.data() + kKvPageBytes / 2u;
    const uint64_t device_offset = page_start * kKvDim;

    if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_F8_E4M3) {
        cudaError_t status = cudaMemcpy(
                record.data(), static_cast<const uint8_t *>(compute->k_cache) + device_offset,
                valid_elements, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                    record_v, static_cast<const uint8_t *>(compute->v_cache) + device_offset,
                    valid_elements, cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) return cuda_status(status);
    } else if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_BF16) {
        std::vector<uint16_t> k_plane;
        std::vector<uint16_t> v_plane;
        try {
            k_plane.resize(valid_elements);
            v_plane.resize(valid_elements);
        } catch (...) {
            return AXIOM_ERR_BUDGET;
        }
        const size_t valid_bytes = valid_elements * sizeof(uint16_t);
        cudaError_t status = cudaMemcpy(
                k_plane.data(), static_cast<const uint16_t *>(compute->k_cache) + device_offset,
                valid_bytes, cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                    v_plane.data(),
                    static_cast<const uint16_t *>(compute->v_cache) + device_offset,
                    valid_bytes, cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) return cuda_status(status);
        for (size_t index = 0u; index < valid_elements; ++index) {
            const float k_value = bf16_host_to_float(k_plane[index]);
            const float v_value = bf16_host_to_float(v_plane[index]);
            if (!std::isfinite(k_value) || !std::isfinite(v_value)) {
                return AXIOM_ERR_RUNTIME;
            }
            record[index] = float_host_to_e4m3fn(k_value);
            record_v[index] = float_host_to_e4m3fn(v_value);
        }
    } else {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    if (!e4m3_record_finite(record.data(), valid_elements)) {
        return AXIOM_ERR_RUNTIME;
    }
    std::memcpy(host_page, record.data(), static_cast<size_t>(kKvPageBytes));
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_kv_page_import(
        axiom_qwen38_mtp_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes) {
    int rc = mtp_kv_page_args(
            compute, layer, logical_page, host_page, host_page_bytes);
    const uint64_t page_start = static_cast<uint64_t>(logical_page) * kKvPageTokens;
    if (rc != AXIOM_OK || page_start < compute->committed_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t valid_tokens = std::min<uint64_t>(
            kKvPageTokens, compute->max_context - page_start);
    const size_t valid_elements = static_cast<size_t>(valid_tokens * kKvDim);
    const uint8_t *record = static_cast<const uint8_t *>(host_page);
    const uint8_t *record_v = record + kKvPageBytes / 2u;
    if (!e4m3_record_finite(record, valid_elements)) return AXIOM_ERR_RUNTIME;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc != AXIOM_OK) return rc;
    const uint64_t device_offset = page_start * kKvDim;

    if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_F8_E4M3) {
        cudaError_t status = cudaMemcpy(
                static_cast<uint8_t *>(compute->k_cache) + device_offset,
                record, valid_elements, cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                    static_cast<uint8_t *>(compute->v_cache) + device_offset,
                    record_v, valid_elements, cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) return cuda_status(status);
    } else if (compute->cache_dtype == AXIOM_TENSOR_DTYPE_BF16) {
        std::vector<uint16_t> k_plane;
        std::vector<uint16_t> v_plane;
        try {
            k_plane.resize(valid_elements);
            v_plane.resize(valid_elements);
        } catch (...) {
            return AXIOM_ERR_BUDGET;
        }
        for (size_t index = 0u; index < valid_elements; ++index) {
            k_plane[index] = float_host_to_bf16(e4m3fn_to_float(record[index]));
            v_plane[index] = float_host_to_bf16(e4m3fn_to_float(record_v[index]));
        }
        const size_t valid_bytes = valid_elements * sizeof(uint16_t);
        cudaError_t status = cudaMemcpy(
                static_cast<uint16_t *>(compute->k_cache) + device_offset,
                k_plane.data(), valid_bytes, cudaMemcpyHostToDevice);
        if (status == cudaSuccess) {
            status = cudaMemcpy(
                    static_cast<uint16_t *>(compute->v_cache) + device_offset,
                    v_plane.data(), valid_bytes, cudaMemcpyHostToDevice);
        }
        if (status != cudaSuccess) return cuda_status(status);
    } else {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc == AXIOM_OK) compute->imported_kv_pages[logical_page] = 1u;
    return rc;
}

extern "C" int axiom_qwen38_mtp_compute_restore_position(
        axiom_qwen38_mtp_compute *compute,
        uint32_t position) {
    if (!compute || compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || compute->device_session_active ||
        position > compute->max_context ||
        (position > compute->committed_position &&
         !imported_prefix_covers(compute, position))) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const bool consumed_import = position > compute->committed_position;
    compute->committed_position = position;
    compute->transaction_anchor = position;
    compute->staged_end_position = position;
    compute->transaction_stream = nullptr;
    compute->device_session_anchor = position;
    compute->device_enqueue_stream = nullptr;
    if (consumed_import) {
        std::fill(compute->imported_kv_pages.begin(),
                  compute->imported_kv_pages.end(), 0u);
    }
    return AXIOM_OK;
}

extern "C" uint32_t axiom_qwen38_mtp_compute_position(
        const axiom_qwen38_mtp_compute *compute) {
    return compute ? compute->committed_position : 0u;
}

extern "C" uint64_t axiom_qwen38_mtp_compute_device_bytes(
        const axiom_qwen38_mtp_compute *compute) {
    return compute ? compute->device_bytes : 0u;
}

extern "C" axiom_qwen38_rope_profile axiom_qwen38_mtp_compute_rope_profile(
        const axiom_qwen38_mtp_compute *compute) {
    return compute ? compute->rope_profile : AXIOM_QWEN38_ROPE_PROFILE_INVALID;
}

extern "C" int axiom_qwen38_mtp_compute_info_get(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out) {
    if (!compute || !out ||
        out->abi_version != AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->device = static_cast<uint32_t>(compute->device);
    out->max_context = compute->max_context;
    out->hidden_size = kHidden;
    out->intermediate_size = kIntermediate;
    out->attention_heads = kHeads;
    out->key_value_heads = kKvHeads;
    out->head_dim = kHeadDim;
    out->max_columns = kMaxColumns;
    out->cache_dtype = compute->cache_dtype;
    out->mtp_layers = AXIOM_QWEN38_MTP_LAYERS;
    out->vocab_size = kVocab;
    out->transaction_state = static_cast<uint32_t>(compute->transaction_state);
    out->transaction_begin_position = compute->transaction_anchor;
    out->committed_position = compute->committed_position;
    out->staged_position = compute->staged_end_position;
    out->checkpoint_device_bytes = compute->checkpoint_bytes;
    out->workspace_device_bytes = compute->workspace_bytes;
    out->kv_cache_device_bytes = compute->kv_cache_bytes;
    out->device_bytes = compute->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_info_get_v1(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_device_info_v1 *out,
        uint64_t info_bytes) {
    if (!compute || !out || info_bytes != sizeof(*out) ||
        out->abi_version != AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION ||
        out->struct_size != sizeof(*out)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    const uint32_t size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->struct_size = size;
    out->device_session_active = compute->device_session_active ? 1u : 0u;
    out->graph_capture_active = compute->device_graph_capture_active ? 1u : 0u;
    out->device_session_begin_position = compute->device_session_active
            ? compute->device_session_anchor : compute->committed_position;
    out->committed_position = compute->committed_position;
    out->max_context = compute->max_context;
    out->rope_profile = static_cast<uint32_t>(compute->rope_profile);
    out->workspace_device_bytes = compute->workspace_bytes;
    out->device_bytes = compute->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_graph_capture_begin(
        axiom_qwen38_mtp_compute *compute) {
    if (!compute ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_session_active || compute->device_graph_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->device_graph_capture_active = true;
    compute->device_enqueue_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_graph_capture_end(
        axiom_qwen38_mtp_compute *compute) {
    if (!compute ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_session_active || !compute->device_graph_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->device_graph_capture_active = false;
    compute->device_enqueue_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_session_begin(
        axiom_qwen38_mtp_compute *compute,
        uint32_t expected_host_position) {
    if (!compute ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || compute->device_session_active ||
        expected_host_position != compute->committed_position ||
        expected_host_position > compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->device_session_anchor = expected_host_position;
    compute->device_session_active = true;
    compute->device_enqueue_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_session_end(
        axiom_qwen38_mtp_compute *compute,
        uint32_t materialized_position) {
    if (!compute ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || !compute->device_session_active ||
        materialized_position < compute->device_session_anchor ||
        materialized_position > compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = materialized_position;
    compute->transaction_anchor = materialized_position;
    compute->staged_end_position = materialized_position;
    compute->transaction_stream = nullptr;
    compute->device_session_anchor = materialized_position;
    compute->device_session_active = false;
    compute->device_enqueue_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_device_session_abort(
        axiom_qwen38_mtp_compute *compute) {
    if (!compute ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || !compute->device_session_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = compute->device_session_anchor;
    compute->transaction_anchor = compute->device_session_anchor;
    compute->staged_end_position = compute->device_session_anchor;
    compute->transaction_stream = nullptr;
    compute->device_session_active = false;
    compute->device_enqueue_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_forward_device_enqueue_v1(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_forward_request_v1 *request,
        uint64_t request_bytes) {
    if (!compute || !request || request_bytes != sizeof(*request) ||
        request->abi_version != AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION ||
        request->struct_size != sizeof(*request) ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        (!compute->device_graph_capture_active && !compute->device_session_active) ||
        !request->token_ids_device || !request->prior_hidden_device ||
        !request->out_hidden_device || !request->base_position_device ||
        !request->async_status_device || !request->stream || request->flags != 0u ||
        request->columns == 0u || request->columns > kMaxColumns) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    if (compute->device_enqueue_stream && compute->device_enqueue_stream != stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (!compute->device_enqueue_stream) {
        int rc = cublas_status(cublasSetStream(compute->cublas, stream));
        if (rc == AXIOM_OK) {
            rc = cublas_status(cublasSetWorkspace(
                    compute->cublas, compute->cublas_workspace,
                    static_cast<size_t>(kCublasWorkspaceBytes)));
        }
        if (rc != AXIOM_OK) return rc;
        compute->device_enqueue_stream = stream;
    }
    return enqueue_forward_device(compute, request);
}

extern "C" int axiom_qwen38_mtp_compute_kv_catchup_device_enqueue_v1(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_kv_catchup_request_v1 *request,
        uint64_t request_bytes) {
    if (!compute || !request || request_bytes != sizeof(*request) ||
        request->abi_version != AXIOM_QWEN38_MTP_COMPUTE_KV_CATCHUP_ABI_VERSION ||
        request->struct_size != sizeof(*request) ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        (!compute->device_graph_capture_active && !compute->device_session_active) ||
        !request->token_ids_device || !request->prior_hidden_device ||
        !request->base_position_device || !request->async_status_device ||
        !request->stream || request->flags != 0u || request->columns == 0u ||
        request->columns > kMaxColumns ||
        (compute->cache_dtype != AXIOM_TENSOR_DTYPE_BF16 &&
         compute->cache_dtype != AXIOM_TENSOR_DTYPE_F8_E4M3) ||
        (compute->rope_profile != AXIOM_QWEN38_ROPE_PROFILE_NATIVE_262K &&
         compute->rope_profile != AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    if (compute->device_enqueue_stream && compute->device_enqueue_stream != stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (!compute->device_enqueue_stream) {
        int rc = cublas_status(cublasSetStream(compute->cublas, stream));
        if (rc == AXIOM_OK) {
            rc = cublas_status(cublasSetWorkspace(
                    compute->cublas, compute->cublas_workspace,
                    static_cast<size_t>(kCublasWorkspaceBytes)));
        }
        if (rc != AXIOM_OK) return rc;
        compute->device_enqueue_stream = stream;
    }
    return enqueue_kv_catchup_device(compute, request);
}

extern "C" int axiom_qwen38_mtp_compute_transaction_begin(
        axiom_qwen38_mtp_compute *compute,
        uint32_t first_position,
        void *stream) {
    if (!compute || compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute->device_graph_capture_active || compute->device_session_active ||
        first_position != compute->committed_position ||
        first_position >= compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->transaction_anchor = first_position;
    compute->staged_end_position = first_position;
    compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_OPEN;
    compute->transaction_stream = static_cast<cudaStream_t>(stream);
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_forward(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_forward_request *request) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!request || request->abi_version != AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION ||
        compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_OPEN ||
        !request->token_ids_device ||
        !request->prior_hidden_device || !request->out_hidden_device ||
        request->flags != 0u ||
        request->columns == 0u || request->columns > kMaxColumns ||
        request->first_position != compute->staged_end_position ||
        static_cast<cudaStream_t>(request->stream) != compute->transaction_stream ||
        request->first_position > compute->max_context ||
        request->columns > compute->max_context - request->first_position) {
        if (compute->transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_OPEN) {
            compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_FAILED;
        }
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) {
        compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_FAILED;
        return AXIOM_ERR_CUDA;
    }
    const int rc = enqueue_forward(compute, request);
    if (rc == AXIOM_OK) {
        compute->staged_end_position += request->columns;
    } else {
        compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_FAILED;
    }
    return rc;
}

extern "C" int axiom_qwen38_mtp_compute_transaction_commit_prefix(
        axiom_qwen38_mtp_compute *compute,
        uint32_t consumed_columns,
        void *stream) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    if (compute->transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_OPEN ||
        static_cast<cudaStream_t>(stream) != compute->transaction_stream ||
        consumed_columns > compute->staged_end_position - compute->transaction_anchor) {
        if (compute->transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_OPEN) {
            compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_FAILED;
        }
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = compute->transaction_anchor + consumed_columns;
    compute->transaction_anchor = compute->committed_position;
    compute->staged_end_position = compute->committed_position;
    compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_IDLE;
    compute->transaction_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_compute_transaction_abort(
        axiom_qwen38_mtp_compute *compute,
        void *stream) {
    if (!compute || compute->transaction_state == AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        static_cast<cudaStream_t>(stream) != compute->transaction_stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = compute->transaction_anchor;
    compute->staged_end_position = compute->transaction_anchor;
    compute->transaction_state = AXIOM_QWEN38_MTP_TRANSACTION_IDLE;
    compute->transaction_stream = nullptr;
    return AXIOM_OK;
}
