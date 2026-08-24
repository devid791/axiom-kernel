/* Native CUDA execution for the BF16 Qwen3.8-27B-DSpark sidecar. */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"
#define AXIOM_QWEN38_DSPARK_COMPUTE_IMPLEMENTATION 1
#include "axiom/qwen38_dspark_compute.h"

namespace {

constexpr uint32_t kHidden = AXIOM_QWEN38_DSPARK_HIDDEN;
constexpr uint32_t kIntermediate = AXIOM_QWEN38_DSPARK_INTERMEDIATE;
constexpr uint32_t kLayers = AXIOM_QWEN38_DSPARK_LAYERS;
constexpr uint32_t kHeads = AXIOM_QWEN38_DSPARK_HEADS;
constexpr uint32_t kKvHeads = AXIOM_QWEN38_DSPARK_KV_HEADS;
constexpr uint32_t kHeadDim = AXIOM_QWEN38_DSPARK_HEAD_DIM;
constexpr uint32_t kKvDim = kKvHeads * kHeadDim;
constexpr uint32_t kVocab = AXIOM_QWEN38_DSPARK_VOCAB;
constexpr uint32_t kFusionInput = AXIOM_QWEN38_DSPARK_FUSION_INPUT;
constexpr uint32_t kBlock = AXIOM_QWEN38_DSPARK_BLOCK_SIZE;
constexpr uint32_t kVerifyWidth = AXIOM_QWEN38_DSPARK_VERIFY_WIDTH;
constexpr uint32_t kRank = AXIOM_QWEN38_DSPARK_MARKOV_RANK;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kGreedyBlocks = 256u;
constexpr uint32_t kHeadThreads = 128u;
constexpr uint32_t kKvPageTokens = 256u;
constexpr uint64_t kKvPageBytes = 524288u;
/* Two query heads per block keeps the 40:8 GQA K/V tile resident while
 * retaining 7 * 8 * ceil(5 / 2) = 168 blocks for the short DSpark block.
 * A full five-head block would expose more reuse but under-fills Blackwell at
 * B=7; pairs are the useful occupancy/reuse point for this fixed geometry. */
constexpr uint32_t kGqaHeadsPerBlock = 2u;
constexpr uint32_t kGqaRatio = kHeads / kKvHeads;
constexpr uint32_t kGqaHeadGroups =
        (kGqaRatio + kGqaHeadsPerBlock - 1u) / kGqaHeadsPerBlock;
constexpr uint32_t kGqaBlockThreads = kGqaHeadsPerBlock * kHeadDim;
constexpr uint32_t kAttentionTileKeys = 4u;
constexpr uint32_t kAttentionWarps = kGqaBlockThreads / 32u;
constexpr uint32_t kDefaultAttentionTileKeys = 32u;
constexpr uint32_t kAttentionMaxSplitK = 4u;
constexpr uint64_t kCublasWorkspaceBytes = 32ull * 1024ull * 1024ull;
constexpr float kRmsEps = 1.0e-6f;
constexpr float kRopeTheta = 10000000.0f;
/* Match the Qwen3.8 1M YaRN override: native 262,144, factor 4. */
constexpr float kYarnFactor = 4.0f;
constexpr float kYarnOriginalContext = 262144.0f;
constexpr float kYarnBetaFast = 32.0f;
constexpr float kYarnBetaSlow = 1.0f;
constexpr float kYarnMscale = 1.138629436111989f; /* 1 + 0.1 * ln(4). */
constexpr float kAttentionScale = 1.0f / 11.313708498984760f;  // 1/sqrt(128)

static_assert(kKvDim == 1024u, "DSpark KV geometry changed");
static_assert(kHeads / kKvHeads == 5u, "DSpark GQA ratio changed");
static_assert(kGqaBlockThreads == 256u, "DSpark paired-GQA launch geometry changed");
static_assert(kFusionInput == 5u * kHidden, "DSpark fusion geometry changed");

struct dspark_layer_weights {
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
};

int cuda_status(const cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

int cublas_status(const cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) return AXIOM_OK;
    return status == CUBLAS_STATUS_ALLOC_FAILED ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

float bf16_host_to_float(const uint16_t bits) {
    uint32_t expanded = static_cast<uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

uint16_t float_host_to_bf16(const float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16u);
}

uint8_t float_host_to_e4m3fn(const float value) {
    const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
    const float magnitude = std::fabs(value);
    if (std::isnan(value)) return static_cast<uint8_t>(sign | 0x7fu);
    if (!std::isfinite(value) || magnitude >= 448.0f) {
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

float e4m3fn_to_float(const uint8_t code) {
    const uint32_t magnitude = static_cast<uint32_t>(code) & 0x7fu;
    if (magnitude == 0x7fu) return std::numeric_limits<float>::quiet_NaN();
    const float sign = (code & 0x80u) != 0u ? -1.0f : 1.0f;
    const uint32_t exponent = (magnitude >> 3u) & 0x0fu;
    const uint32_t mantissa = magnitude & 0x07u;
    if (exponent == 0u) return sign * std::ldexp(static_cast<float>(mantissa), -9);
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                             static_cast<int>(exponent) - 7);
}

bool checked_mul(const uint64_t a, const uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool checked_add(const uint64_t a, const uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int add_bytes(uint64_t *total, const uint64_t amount) {
    uint64_t next = 0u;
    if (!total || !checked_add(*total, amount, &next)) return AXIOM_ERR_BUDGET;
    *total = next;
    return AXIOM_OK;
}

int buffer_pointer(const axiom_device_buffer *buffer, void **out) {
    if (!buffer || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_device_buffer_cuda_pointer(buffer, out);
}

__device__ __forceinline__ float bf16_to_float(const uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ __forceinline__ uint16_t float_to_bf16(const float value) {
    return __bfloat16_as_ushort(__float2bfloat16(value));
}

/* Exact SGLang/Hugging Face YaRN blend used by the DSpark checkpoint. */
__device__ __forceinline__ float yarn_inv_frequency(const uint32_t pair) {
    const float dim = static_cast<float>(kHeadDim);
    const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(kHeadDim);
    const float base = powf(kRopeTheta, exponent);
    const float inv = 1.0f / base;
    const float correction_fast = dim *
            logf(kYarnOriginalContext / (kYarnBetaFast * 6.283185307179586f)) /
            (2.0f * logf(kRopeTheta));
    const float correction_slow = dim *
            logf(kYarnOriginalContext / (kYarnBetaSlow * 6.283185307179586f)) /
            (2.0f * logf(kRopeTheta));
    const float low = floorf(correction_fast < correction_slow ? correction_fast : correction_slow);
    const float high = ceilf(correction_fast > correction_slow ? correction_fast : correction_slow);
    float ramp = high > low
            ? (static_cast<float>(pair) - low) / (high - low)
            : static_cast<float>(pair) >= high ? 1.0f : 0.0f;
    ramp = fminf(1.0f, fmaxf(0.0f, ramp));
    const float interpolated = inv / kYarnFactor;
    return inv * (1.0f - ramp) + interpolated * ramp;
}

__global__ void pack_target_taps_kernel(
        const float *__restrict__ taps,
        float *__restrict__ fusion_input,
        const uint32_t columns) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kFusionInput) * columns;
    if (element >= count) return;
    const uint32_t column = static_cast<uint32_t>(element / kFusionInput);
    const uint32_t feature = static_cast<uint32_t>(element - static_cast<uint64_t>(column) * kFusionInput);
    const uint32_t tap = feature / kHidden;
    const uint32_t hidden = feature - tap * kHidden;
    fusion_input[element] = taps[(static_cast<uint64_t>(tap) * columns + column) * kHidden + hidden];
}

/* Blackwell cuBLAS does not provide the mixed BF16-weight/F32-activation
 * GEMM used by the first draft implementation for the DSpark shapes. Every
 * dense projection therefore reuses this resident staging plane and calls the
 * supported BF16 x BF16 -> F32 path. */
__global__ void f32_to_bf16_stage_kernel(
        const float *__restrict__ input,
        uint16_t *__restrict__ out,
        const uint64_t count) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < count) out[element] = float_to_bf16(input[element]);
}

/* The released draft is a BF16 network. cuBLAS writes F32 in Axiom so that
 * the surrounding custom kernels can consume it, therefore explicitly round
 * every public BF16 activation boundary before the next nonlinear operation.
 * Merely rounding at the next GEMM is insufficient for residuals, RMSNorm,
 * Q/K normalization and attention. */
__global__ void round_f32_to_bf16_inplace_kernel(float *values, const uint64_t count) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < count) values[element] = bf16_to_float(float_to_bf16(values[element]));
}

int launch_round_bf16(float *values, const uint64_t count, const cudaStream_t stream) {
    if (!values || count == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    round_f32_to_bf16_inplace_kernel<<<
            static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(values, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void rmsnorm_bf16_kernel(
        const uint16_t *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out,
        const uint32_t features,
        const uint32_t columns) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= columns) return;
    const float *x = input + static_cast<uint64_t>(column) * features;
    float *y = out + static_cast<uint64_t>(column) * features;
    __shared__ float sums[kThreads];
    float partial = 0.0f;
    for (uint32_t feature = tid; feature < features; feature += blockDim.x) {
        partial = fmaf(x[feature], x[feature], partial);
    }
    sums[tid] = partial;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2u; offset != 0u; offset >>= 1u) {
        if (tid < offset) sums[tid] += sums[tid + offset];
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / static_cast<float>(features) + kRmsEps);
    for (uint32_t feature = tid; feature < features; feature += blockDim.x) {
        y[feature] = x[feature] * inverse * bf16_to_float(weight[feature]);
    }
}

__global__ void qk_rmsnorm_rope_bf16_kernel(
        const uint16_t *__restrict__ weight,
        float *__restrict__ x,
        const uint32_t heads,
        const uint32_t columns,
        const uint32_t first_position,
        const uint32_t *__restrict__ first_position_device) {
    const uint32_t block = blockIdx.x;
    const uint32_t column = block / heads;
    const uint32_t head = block - column * heads;
    const uint32_t tid = threadIdx.x;
    if (column >= columns || tid >= kHeadDim) return;
    const uint64_t offset = static_cast<uint64_t>(column) * heads * kHeadDim +
            static_cast<uint64_t>(head) * kHeadDim;
    __shared__ float reductions[kHeadThreads];
    const float value = x[offset + tid];
    reductions[tid] = value * value;
    __syncthreads();
    for (uint32_t stride = kHeadThreads / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    const float normalized = value *
            rsqrtf(reductions[0] / static_cast<float>(kHeadDim) + kRmsEps) *
            bf16_to_float(weight[tid]);
    /* SGLang's fused table kernel materializes normalized Q/K as BF16 before
     * applying RoPE, then stores the rotated result as BF16 again. */
    x[offset + tid] = bf16_to_float(float_to_bf16(normalized));
    __syncthreads();
    if (tid < kHeadDim / 2u) {
        const uint32_t other = tid + kHeadDim / 2u;
        const uint32_t position_base = first_position_device ? first_position_device[0] : first_position;
        const float angle = static_cast<float>(position_base + column) * yarn_inv_frequency(tid);
        const float c = cosf(angle) * kYarnMscale;
        const float s = sinf(angle) * kYarnMscale;
        const float a = x[offset + tid];
        const float b = x[offset + other];
        x[offset + tid] = a * c - b * s;
        x[offset + other] = a * s + b * c;
    }
}

__global__ void store_kv_bf16_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        uint16_t *__restrict__ k_cache,
        uint16_t *__restrict__ v_cache,
        const uint32_t first_position,
        const uint32_t columns,
        const uint32_t *__restrict__ first_position_device,
        const uint32_t max_context,
        uint32_t *__restrict__ async_status) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kKvDim) * columns;
    if (element >= count) return;
    /* Full-cycle graph replay has no host branch after greedy acceptance.
     * Do not persist speculative target taps if acceptance marked the cycle
     * invalid; all upstream scratch writes are transient. */
    if (async_status && *async_status != static_cast<uint32_t>(AXIOM_OK)) return;
    const uint32_t column = static_cast<uint32_t>(element / kKvDim);
    const uint32_t feature = static_cast<uint32_t>(element - static_cast<uint64_t>(column) * kKvDim);
    const uint32_t position_base = first_position_device ? first_position_device[0] : first_position;
    if (position_base > max_context || column >= max_context - position_base) {
        if (async_status && element == 0u) *async_status = static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT);
        return;
    }
    const uint64_t destination = static_cast<uint64_t>(position_base + column) * kKvDim + feature;
    k_cache[destination] = float_to_bf16(k[element]);
    v_cache[destination] = float_to_bf16(v[element]);
}

__global__ void fill_noise_block_kernel(
        uint32_t *__restrict__ tokens,
        const uint32_t anchor_token,
        const uint32_t columns,
        const uint32_t *__restrict__ anchor_token_device) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;
    const uint32_t anchor = anchor_token_device ? anchor_token_device[0] : anchor_token;
    tokens[column] = column == 0u ? anchor : AXIOM_QWEN38_DSPARK_MASK_TOKEN_ID;
}

__global__ void residual_add_kernel(
        const float *__restrict__ left,
        const float *__restrict__ right,
        float *__restrict__ out,
        const uint64_t count) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element < count) out[element] = left[element] + right[element];
}

__global__ void silu_multiply_kernel(
        float *__restrict__ gate,
        const float *__restrict__ up,
        const uint64_t count) {
    const uint64_t element = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= count) return;
    const float x = gate[element];
    gate[element] = (x / (1.0f + expf(-x))) * up[element];
}

/* One block evaluates two Q heads which share one GQA K/V head.  The 4x128
 * K/V tile is read once from global memory and reused by both query heads.
 *
 * The arithmetic per Q head deliberately mirrors the prior 128-thread
 * kernel: one warp owns one key dot-product, the four 32-wide pieces are
 * accumulated in ascending dimension order, then online softmax advances in
 * ascending key order.  Thus the BF16 materialization boundaries and F32
 * association remain unchanged; only the K/V transport moves through shared
 * memory.  Local K/V remain fully visible to every temporal query, preserving
 * DSpark's encoder-only/non-causal block semantics. */
__global__ void noncausal_gqa_attention_bf16_kernel(
        const float *__restrict__ q,
        const float *__restrict__ local_k,
        const float *__restrict__ local_v,
        const uint16_t *__restrict__ k_cache,
        const uint16_t *__restrict__ v_cache,
        float *__restrict__ out,
        const uint32_t cache_tokens,
        const uint32_t columns,
        const uint32_t *__restrict__ cache_tokens_device) {
    const uint32_t packed = blockIdx.x;
    const uint32_t groups_per_column = kKvHeads * kGqaHeadGroups;
    const uint32_t column = packed / groups_per_column;
    const uint32_t within_column = packed - column * groups_per_column;
    const uint32_t kv_head = within_column / kGqaHeadGroups;
    const uint32_t head_group = within_column - kv_head * kGqaHeadGroups;
    const uint32_t tid = threadIdx.x;
    if (column >= columns || tid >= kGqaBlockThreads) return;
    const uint32_t first_head_in_group = head_group * kGqaHeadsPerBlock;
    const uint32_t valid_heads = first_head_in_group < kGqaRatio
            ? ((kGqaRatio - first_head_in_group) < kGqaHeadsPerBlock
                       ? (kGqaRatio - first_head_in_group)
                       : kGqaHeadsPerBlock)
            : 0u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t dot_head = warp / kAttentionTileKeys;
    const uint32_t dot_key = warp - dot_head * kAttentionTileKeys;
    const uint32_t value_head = tid / kHeadDim;
    const uint32_t value_dim = tid - value_head * kHeadDim;
    const uint32_t head_base = kv_head * kGqaRatio + first_head_in_group;
    float accumulator = 0.0f;
    __shared__ float shared_k[kAttentionTileKeys][kHeadDim];
    __shared__ float shared_v[kAttentionTileKeys][kHeadDim];
    __shared__ float scores[kGqaHeadsPerBlock][kAttentionTileKeys];
    __shared__ float rescales[kGqaHeadsPerBlock][kAttentionTileKeys];
    __shared__ float probabilities[kGqaHeadsPerBlock][kAttentionTileKeys];
    __shared__ float running_maximum[kGqaHeadsPerBlock];
    __shared__ float running_normalizer[kGqaHeadsPerBlock];
    const uint32_t effective_cache_tokens = cache_tokens_device ? cache_tokens_device[0] : cache_tokens;
    const uint32_t total_keys = effective_cache_tokens + columns;
    if (tid < valid_heads) {
        running_maximum[tid] = -CUDART_INF_F;
        running_normalizer[tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t key_base = 0u; key_base < total_keys; key_base += kAttentionTileKeys) {
        /* Each K/V element is loaded once for both query heads.  Cache entries
         * are BF16; local entries already carry the explicit BF16 boundary in
         * F32, so this transport does not alter either representation. */
        #pragma unroll
        for (uint32_t load = tid; load < kAttentionTileKeys * kHeadDim;
             load += kGqaBlockThreads) {
            const uint32_t key_slot = load / kHeadDim;
            const uint32_t dim = load - key_slot * kHeadDim;
            const uint32_t key = key_base + key_slot;
            float key_value = 0.0f;
            float value = 0.0f;
            if (key < total_keys) {
                if (key < effective_cache_tokens) {
                    const uint64_t offset = static_cast<uint64_t>(key) * kKvDim +
                            kv_head * kHeadDim + dim;
                    key_value = bf16_to_float(k_cache[offset]);
                    value = bf16_to_float(v_cache[offset]);
                } else {
                    const uint32_t local_column = key - effective_cache_tokens;
                    const uint64_t offset = static_cast<uint64_t>(local_column) * kKvDim +
                            kv_head * kHeadDim + dim;
                    key_value = local_k[offset];
                    value = local_v[offset];
                }
            }
            shared_k[key_slot][dim] = key_value;
            shared_v[key_slot][dim] = value;
        }
        __syncthreads();

        const uint32_t key = key_base + dot_key;
        float dot = -CUDART_INF_F;
        if (dot_head < valid_heads && key < total_keys) {
            const uint64_t q_offset = static_cast<uint64_t>(column) * kHidden +
                    static_cast<uint64_t>(head_base + dot_head) * kHeadDim;
            dot = 0.0f;
            #pragma unroll
            for (uint32_t part = 0u; part < 4u; ++part) {
                const uint32_t dim = lane + part * 32u;
                dot = fmaf(q[q_offset + dim], shared_k[dot_key][dim], dot);
            }
            for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
                dot += __shfl_down_sync(0xffffffffu, dot, static_cast<int>(offset));
            }
        }
        if (lane == 0u && dot_head < valid_heads) {
            scores[dot_head][dot_key] = dot * kAttentionScale;
        }
        __syncthreads();

        if (tid < valid_heads) {
            float maximum = running_maximum[tid];
            float normalizer = running_normalizer[tid];
            #pragma unroll
            for (uint32_t group = 0u; group < kAttentionTileKeys; ++group) {
                if (key_base + group >= total_keys) {
                    rescales[tid][group] = 1.0f;
                    probabilities[tid][group] = 0.0f;
                    continue;
                }
                const float next_maximum = fmaxf(maximum, scores[tid][group]);
                const float rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_maximum);
                const float probability = expf(scores[tid][group] - next_maximum);
                normalizer = normalizer * rescale + probability;
                maximum = next_maximum;
                rescales[tid][group] = rescale;
                probabilities[tid][group] = probability;
            }
            running_maximum[tid] = maximum;
            running_normalizer[tid] = normalizer;
        }
        __syncthreads();

        if (value_head < valid_heads) {
            #pragma unroll
            for (uint32_t group = 0u; group < kAttentionTileKeys; ++group) {
                const uint32_t group_key = key_base + group;
                if (group_key >= total_keys) continue;
                accumulator = accumulator * rescales[value_head][group] +
                        probabilities[value_head][group] * shared_v[group][value_dim];
            }
        }
        __syncthreads();
    }
    if (value_head < valid_heads) {
        const uint64_t q_offset = static_cast<uint64_t>(column) * kHidden +
                static_cast<uint64_t>(head_base + value_head) * kHeadDim;
        out[q_offset + value_dim] = accumulator / running_normalizer[value_head];
    }
}

/* Wider key tiles keep the exact per-head arithmetic order of the reference
 * tile-4 kernel while halving (or better) the number of block barriers.  Each
 * of the eight warps owns one or more key slots and computes both query heads
 * which share a GQA K/V head.  K/V are still loaded once per pair of heads.
 * Tile width is selected before graph capture, so replay has no host branch. */
template <uint32_t TileKeys, uint32_t SplitK = 1u>
__global__ void noncausal_gqa_attention_bf16_wide_kernel(
        const float *__restrict__ q,
        const float *__restrict__ local_k,
        const float *__restrict__ local_v,
        const uint16_t *__restrict__ k_cache,
        const uint16_t *__restrict__ v_cache,
        float *__restrict__ out,
        float *__restrict__ split_values,
        float *__restrict__ split_maxima,
        float *__restrict__ split_denominators,
        const uint32_t cache_tokens,
        const uint32_t columns,
        const uint32_t *__restrict__ cache_tokens_device) {
    static_assert(TileKeys >= kAttentionWarps && TileKeys % kAttentionWarps == 0u,
                  "wide attention tile must map evenly across warps");
    static_assert(SplitK == 1u || SplitK == 2u || SplitK == 4u,
                  "unsupported DSpark attention split count");
    const uint32_t packed = blockIdx.x;
    const uint32_t groups_per_column = kKvHeads * kGqaHeadGroups;
    const uint32_t column = packed / groups_per_column;
    const uint32_t within_column = packed - column * groups_per_column;
    const uint32_t kv_head = within_column / kGqaHeadGroups;
    const uint32_t head_group = within_column - kv_head * kGqaHeadGroups;
    const uint32_t tid = threadIdx.x;
    if (column >= columns || tid >= kGqaBlockThreads) return;
    const uint32_t first_head_in_group = head_group * kGqaHeadsPerBlock;
    const uint32_t valid_heads = first_head_in_group < kGqaRatio
            ? ((kGqaRatio - first_head_in_group) < kGqaHeadsPerBlock
                       ? (kGqaRatio - first_head_in_group)
                       : kGqaHeadsPerBlock)
            : 0u;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t value_head = tid / kHeadDim;
    const uint32_t value_dim = tid - value_head * kHeadDim;
    const uint32_t head_base = kv_head * kGqaRatio + first_head_in_group;
    float accumulator = 0.0f;
    __shared__ float shared_k[TileKeys][kHeadDim];
    __shared__ float shared_v[TileKeys][kHeadDim];
    __shared__ float scores[kGqaHeadsPerBlock][TileKeys];
    __shared__ float rescales[kGqaHeadsPerBlock][TileKeys];
    __shared__ float probabilities[kGqaHeadsPerBlock][TileKeys];
    __shared__ float running_maximum[kGqaHeadsPerBlock];
    __shared__ float running_normalizer[kGqaHeadsPerBlock];
    const uint32_t effective_cache_tokens = cache_tokens_device ? cache_tokens_device[0] : cache_tokens;
    const uint32_t total_keys = effective_cache_tokens + columns;
    const uint32_t split = SplitK == 1u ? 0u : blockIdx.y;
    const uint32_t split_span = (total_keys + SplitK - 1u) / SplitK;
    const uint32_t key_start = split * split_span;
    const uint32_t key_end = min(key_start + split_span, total_keys);
    if (tid < valid_heads) {
        running_maximum[tid] = -CUDART_INF_F;
        running_normalizer[tid] = 0.0f;
    }
    __syncthreads();

    for (uint32_t key_base = key_start; key_base < key_end; key_base += TileKeys) {
        #pragma unroll
        for (uint32_t load = tid; load < TileKeys * kHeadDim;
             load += kGqaBlockThreads) {
            const uint32_t key_slot = load / kHeadDim;
            const uint32_t dim = load - key_slot * kHeadDim;
            const uint32_t key = key_base + key_slot;
            float key_value = 0.0f;
            float value = 0.0f;
            if (key < key_end) {
                if (key < effective_cache_tokens) {
                    const uint64_t offset = static_cast<uint64_t>(key) * kKvDim +
                            kv_head * kHeadDim + dim;
                    key_value = bf16_to_float(k_cache[offset]);
                    value = bf16_to_float(v_cache[offset]);
                } else {
                    const uint32_t local_column = key - effective_cache_tokens;
                    const uint64_t offset = static_cast<uint64_t>(local_column) * kKvDim +
                            kv_head * kHeadDim + dim;
                    key_value = local_k[offset];
                    value = local_v[offset];
                }
            }
            shared_k[key_slot][dim] = key_value;
            shared_v[key_slot][dim] = value;
        }
        __syncthreads();

        #pragma unroll
        for (uint32_t warp_key = 0u; warp_key < TileKeys / kAttentionWarps; ++warp_key) {
            const uint32_t key_slot = warp + warp_key * kAttentionWarps;
            const uint32_t key = key_base + key_slot;
            float dot0 = -CUDART_INF_F;
            float dot1 = -CUDART_INF_F;
            if (key < key_end && valid_heads != 0u) {
                const uint64_t q_offset0 = static_cast<uint64_t>(column) * kHidden +
                        static_cast<uint64_t>(head_base) * kHeadDim;
                dot0 = 0.0f;
                #pragma unroll
                for (uint32_t part = 0u; part < 4u; ++part) {
                    const uint32_t dim = lane + part * 32u;
                    dot0 = fmaf(q[q_offset0 + dim], shared_k[key_slot][dim], dot0);
                }
                if (valid_heads > 1u) {
                    const uint64_t q_offset1 = q_offset0 + kHeadDim;
                    dot1 = 0.0f;
                    #pragma unroll
                    for (uint32_t part = 0u; part < 4u; ++part) {
                        const uint32_t dim = lane + part * 32u;
                        dot1 = fmaf(q[q_offset1 + dim], shared_k[key_slot][dim], dot1);
                    }
                }
                for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
                    dot0 += __shfl_down_sync(0xffffffffu, dot0, static_cast<int>(offset));
                    dot1 += __shfl_down_sync(0xffffffffu, dot1, static_cast<int>(offset));
                }
            }
            if (lane == 0u) {
                if (valid_heads != 0u) scores[0][key_slot] = dot0 * kAttentionScale;
                if (valid_heads > 1u) scores[1][key_slot] = dot1 * kAttentionScale;
            }
        }
        __syncthreads();

        if (tid < valid_heads) {
            float maximum = running_maximum[tid];
            float normalizer = running_normalizer[tid];
            #pragma unroll
            for (uint32_t group = 0u; group < TileKeys; ++group) {
                if (key_base + group >= key_end) {
                    rescales[tid][group] = 1.0f;
                    probabilities[tid][group] = 0.0f;
                    continue;
                }
                const float next_maximum = fmaxf(maximum, scores[tid][group]);
                const float rescale = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_maximum);
                const float probability = expf(scores[tid][group] - next_maximum);
                normalizer = normalizer * rescale + probability;
                maximum = next_maximum;
                rescales[tid][group] = rescale;
                probabilities[tid][group] = probability;
            }
            running_maximum[tid] = maximum;
            running_normalizer[tid] = normalizer;
        }
        __syncthreads();

        if (value_head < valid_heads) {
            #pragma unroll
            for (uint32_t group = 0u; group < TileKeys; ++group) {
                const uint32_t group_key = key_base + group;
                if (group_key >= key_end) continue;
                accumulator = accumulator * rescales[value_head][group] +
                        probabilities[value_head][group] * shared_v[group][value_dim];
            }
        }
        __syncthreads();
    }
    if (value_head < valid_heads) {
        const uint64_t q_offset = static_cast<uint64_t>(column) * kHidden +
                static_cast<uint64_t>(head_base + value_head) * kHeadDim;
        if constexpr (SplitK == 1u) {
            out[q_offset + value_dim] = accumulator / running_normalizer[value_head];
        } else {
            const uint64_t stat_index =
                    (static_cast<uint64_t>(column) * kHeads + head_base + value_head) *
                            kAttentionMaxSplitK + split;
            split_values[stat_index * kHeadDim + value_dim] = accumulator;
        }
    }
    if constexpr (SplitK > 1u) {
        if (tid < valid_heads) {
            const uint64_t stat_index =
                    (static_cast<uint64_t>(column) * kHeads + head_base + tid) *
                            kAttentionMaxSplitK + split;
            split_maxima[stat_index] = running_maximum[tid];
            split_denominators[stat_index] = running_normalizer[tid];
        }
    }
}

template <uint32_t SplitK>
__global__ void merge_noncausal_gqa_attention_split_kernel(
        const float *__restrict__ split_values,
        const float *__restrict__ split_maxima,
        const float *__restrict__ split_denominators,
        float *__restrict__ out,
        uint32_t columns) {
    static_assert(SplitK == 2u || SplitK == 4u,
                  "unsupported DSpark attention merge split count");
    const uint32_t packed = blockIdx.x;
    const uint32_t column = packed / kHeads;
    const uint32_t head = packed - column * kHeads;
    const uint32_t dim = threadIdx.x;
    if (column >= columns || dim >= kHeadDim) return;
    const uint64_t stat_base =
            (static_cast<uint64_t>(column) * kHeads + head) * kAttentionMaxSplitK;
    __shared__ float rescale[SplitK];
    __shared__ float denominator;
    if (dim == 0u) {
        float maximum = -CUDART_INF_F;
        #pragma unroll
        for (uint32_t split = 0u; split < SplitK; ++split) {
            maximum = fmaxf(maximum, split_maxima[stat_base + split]);
        }
        float normalizer = 0.0f;
        #pragma unroll
        for (uint32_t split = 0u; split < SplitK; ++split) {
            const float scale = expf(split_maxima[stat_base + split] - maximum);
            rescale[split] = scale;
            normalizer += split_denominators[stat_base + split] * scale;
        }
        denominator = normalizer;
    }
    __syncthreads();
    float accumulator = 0.0f;
    #pragma unroll
    for (uint32_t split = 0u; split < SplitK; ++split) {
        accumulator += split_values[(stat_base + split) * kHeadDim + dim] * rescale[split];
    }
    const uint64_t out_index = static_cast<uint64_t>(column) * kHidden +
            static_cast<uint64_t>(head) * kHeadDim + dim;
    out[out_index] = accumulator / denominator;
}

__global__ void gather_markov_rank_kernel(
        const uint16_t *__restrict__ w1,
        const uint32_t *__restrict__ previous_token,
        float *__restrict__ rank) {
    const uint32_t feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= kRank) return;
    const uint32_t token = previous_token[0];
    rank[feature] = token < kVocab
            ? bf16_to_float(w1[static_cast<uint64_t>(token) * kRank + feature])
            : 0.0f;
}

__global__ void greedy_top1_kernel(
        const float *__restrict__ logits,
        uint32_t *__restrict__ out_token) {
    const uint32_t tid = threadIdx.x;
    float best_value = -CUDART_INF_F;
    uint32_t best_id = UINT32_MAX;
    for (uint32_t token = tid; token < kVocab; token += blockDim.x) {
        const float value = logits[token];
        if (isfinite(value) && (value > best_value || (value == best_value && token < best_id))) {
            best_value = value;
            best_id = token;
        }
    }
    __shared__ float values[kThreads];
    __shared__ uint32_t ids[kThreads];
    values[tid] = best_value;
    ids[tid] = best_id;
    __syncthreads();
    for (uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            const float right_value = values[tid + stride];
            const uint32_t right_id = ids[tid + stride];
            if (right_value > values[tid] ||
                (right_value == values[tid] && right_id < ids[tid])) {
                values[tid] = right_value;
                ids[tid] = right_id;
            }
        }
        __syncthreads();
    }
    if (tid == 0u) out_token[0] = ids[0];
}

__device__ __forceinline__ void top1_select(
        const float candidate_value,
        const uint32_t candidate_id,
        float *best_value,
        uint32_t *best_id) {
    if (candidate_value > *best_value ||
        (candidate_value == *best_value && candidate_id < *best_id)) {
        *best_value = candidate_value;
        *best_id = candidate_id;
    }
}

__device__ __forceinline__ void top1_warp_reduce(
        float *best_value,
        uint32_t *best_id) {
    constexpr uint32_t mask = 0xffffffffu;
    for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
        const float candidate_value = __shfl_down_sync(mask, *best_value, offset);
        const uint32_t candidate_id = __shfl_down_sync(mask, *best_id, offset);
        top1_select(candidate_value, candidate_id, best_value, best_id);
    }
}

/* The Markov epilogue already touches every vocabulary row.  Fuse its exact
 * BF16 public-boundary round with a hierarchical top-1 reduction instead of
 * rescanning 248,320 logits on one CUDA block. */
__global__ void add_bf16_rounded_top1_stage1_kernel(
        float *__restrict__ base_logits,
        const float *__restrict__ markov_bias,
        float *__restrict__ block_values,
        uint32_t *__restrict__ block_ids) {
    const uint32_t tid = threadIdx.x;
    float best_value = -CUDART_INF_F;
    uint32_t best_id = UINT32_MAX;
    for (uint32_t token = blockIdx.x * blockDim.x + tid;
         token < kVocab;
         token += gridDim.x * blockDim.x) {
        const float base = bf16_to_float(float_to_bf16(base_logits[token]));
        const float bias = bf16_to_float(float_to_bf16(markov_bias[token]));
        const float value = bf16_to_float(float_to_bf16(base + bias));
        base_logits[token] = value;
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
        uint32_t *__restrict__ out_token) {
    const uint32_t tid = threadIdx.x;
    float best_value = tid < kGreedyBlocks ? block_values[tid] : -CUDART_INF_F;
    uint32_t best_id = tid < kGreedyBlocks ? block_ids[tid] : UINT32_MAX;
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
        if (lane == 0u) out_token[0] = best_id;
    }
}

/* All control-plane kernels are intentionally one tiny launch each.  They
 * replace the former D2H proposal copy + host longest-prefix loop and are
 * consumed by a target M8 verifier on the same CUDA stream. */
__global__ void build_verify_input_kernel(
        const uint32_t *__restrict__ anchor,
        const uint32_t *__restrict__ proposal,
        uint32_t *__restrict__ verify_tokens) {
    const uint32_t index = threadIdx.x;
    if (index == 0u) verify_tokens[0] = anchor[0];
    if (index < kBlock) verify_tokens[index + 1u] = proposal[index];
}

__global__ void accept_greedy_kernel(
        const uint32_t *__restrict__ anchor,
        const uint32_t *__restrict__ proposal,
        const uint32_t *__restrict__ target_tokens,
        const float *__restrict__ target_logits,
        const uint32_t *__restrict__ commit_limit,
        uint32_t *__restrict__ terminal_state,
        const uint32_t stop_token_count,
        const uint32_t stop_token_0,
        const uint32_t stop_token_1,
        uint32_t *__restrict__ accepted_prefix,
        uint32_t *__restrict__ target_commit_prefix,
        uint32_t *__restrict__ continuation_token,
        float *__restrict__ continuation_logit,
        uint32_t *__restrict__ async_status) {
    if (threadIdx.x != 0u || blockIdx.x != 0u) return;
    uint32_t prefix = 0u;
    if (!anchor || !proposal || !target_tokens || !target_logits || !commit_limit ||
        !terminal_state || !accepted_prefix ||
        !target_commit_prefix || !continuation_token || !continuation_logit || !async_status) {
        return;
    }
    if (terminal_state[0] != 0u) {
        *accepted_prefix = 0u;
        *target_commit_prefix = 0u;
        *continuation_token = anchor[0];
        *continuation_logit = 0.0f;
        terminal_state[0] = 2u;
        return;
    }
    const uint32_t limit = commit_limit[0];
    if (limit == 0u || limit > kVerifyWidth) {
        *async_status = static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT);
        return;
    }
    for (uint32_t index = 0u; index < kVerifyWidth; ++index) {
        if (target_tokens[index] >= kVocab || !isfinite(target_logits[index])) {
            *async_status = static_cast<uint32_t>(AXIOM_ERR_RUNTIME);
            return;
        }
    }
    for (; prefix < kBlock; ++prefix) {
        const uint32_t draft = proposal[prefix];
        const uint32_t target = target_tokens[prefix];
        if (draft >= kVocab) {
            *async_status = static_cast<uint32_t>(AXIOM_ERR_RUNTIME);
            return;
        }
        if (draft != target) break;
    }
    prefix = min(prefix, limit - 1u);
    for (uint32_t index = 0u; index < prefix; ++index) {
        const uint32_t token = proposal[index];
        if ((stop_token_count > 0u && token == stop_token_0) ||
            (stop_token_count > 1u && token == stop_token_1)) {
            prefix = index;
            break;
        }
    }
    const uint32_t continuation = target_tokens[prefix];
    *accepted_prefix = prefix;
    *target_commit_prefix = prefix + 1u;
    *continuation_token = continuation;
    *continuation_logit = target_logits[prefix];
    if ((stop_token_count > 0u && continuation == stop_token_0) ||
        (stop_token_count > 1u && continuation == stop_token_1)) {
        terminal_state[0] = 1u;
    }
}

__global__ void advance_device_position_kernel(
        uint32_t *__restrict__ anchor_token,
        uint32_t *__restrict__ position,
        const uint32_t *__restrict__ accepted_prefix,
        const uint32_t *__restrict__ continuation_token,
        uint32_t *__restrict__ terminal_state,
        uint32_t *__restrict__ async_status,
        const uint32_t max_context) {
    if (threadIdx.x != 0u || blockIdx.x != 0u || !anchor_token || !position ||
        !accepted_prefix || !continuation_token || !terminal_state || !async_status) {
        return;
    }
    if (*async_status != static_cast<uint32_t>(AXIOM_OK)) return;
    if (terminal_state[0] == 2u) return;
    const uint32_t accepted = accepted_prefix[0];
    const uint32_t current = position[0];
    if (accepted > kBlock || current > max_context ||
        static_cast<uint64_t>(current) + 1u + accepted > max_context) {
        *async_status = static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT);
        return;
    }
    anchor_token[0] = continuation_token[0];
    position[0] = current + 1u + accepted;
    if (terminal_state[0] == 1u) terminal_state[0] = 2u;
}

/* PyTorch/SGLang executes the shared target head, the BF16 Markov projection,
 * and their elementwise sum with BF16 outputs.  Axiom's CUDA primitives keep
 * F32 accumulators, so materialize the same public BF16 boundaries explicitly
 * before greedy sampling. */
__global__ void add_bf16_rounded_kernel(
        float *__restrict__ base_logits,
        const float *__restrict__ markov_bias,
        const uint32_t count) {
    const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= count) return;
    const float base = bf16_to_float(float_to_bf16(base_logits[element]));
    const float bias = bf16_to_float(float_to_bf16(markov_bias[element]));
    base_logits[element] = bf16_to_float(float_to_bf16(base + bias));
}

__global__ void confidence_kernel(
        const float *__restrict__ hidden,
        const float *__restrict__ rank,
        const uint16_t *__restrict__ weight,
        const uint16_t *__restrict__ bias,
        float *__restrict__ out_confidence) {
    const uint32_t tid = threadIdx.x;
    float partial = 0.0f;
    for (uint32_t feature = tid; feature < kHidden; feature += blockDim.x) {
        partial = fmaf(hidden[feature], bf16_to_float(weight[feature]), partial);
    }
    for (uint32_t feature = tid; feature < kRank; feature += blockDim.x) {
        partial = fmaf(rank[feature], bf16_to_float(weight[kHidden + feature]), partial);
    }
    __shared__ float sums[kThreads];
    sums[tid] = partial;
    __syncthreads();
    for (uint32_t stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (tid == 0u) {
        const float value = sums[0] + bf16_to_float(bias[0]);
        out_confidence[0] = 1.0f / (1.0f + expf(-value));
    }
}

int launch_rmsnorm(
        const uint16_t *weight,
        const float *input,
        float *out,
        const uint32_t features,
        const uint32_t columns,
        const cudaStream_t stream) {
    if (!weight || !input || !out || features == 0u || columns == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    rmsnorm_bf16_kernel<<<columns, kThreads, 0, stream>>>(weight, input, out, features, columns);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int launch_qk_norm_rope(
        const uint16_t *weight,
        float *x,
        const uint32_t heads,
        const uint32_t columns,
        const uint32_t first_position,
        const uint32_t *first_position_device,
        const cudaStream_t stream) {
    if (!weight || !x || heads == 0u || columns == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    qk_rmsnorm_rope_bf16_kernel<<<heads * columns, kHeadThreads, 0, stream>>>(
            weight, x, heads, columns, first_position, first_position_device);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

bool fused_top1_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_DSPARK_FUSED_TOP1");
    return !(value && value[0] == '0' && value[1] == '\0');
}

uint32_t dspark_attention_tile_keys() {
    const char *value = std::getenv("AXIOM_QWEN38_DSPARK_ATTENTION_TILE");
    if (!value || value[0] == '\0') return kDefaultAttentionTileKeys;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || end == value || end[0] != '\0') return kDefaultAttentionTileKeys;
    return parsed == 4ul || parsed == 8ul || parsed == 16ul || parsed == 32ul
            ? static_cast<uint32_t>(parsed)
            : kDefaultAttentionTileKeys;
}

uint32_t dspark_attention_split_k() {
    const char *value = std::getenv("AXIOM_QWEN38_DSPARK_ATTENTION_SPLIT_K");
    if (!value || value[0] == '\0') return 4u;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!end || end == value || end[0] != '\0') return 4u;
    return parsed == 1ul || parsed == 2ul || parsed == 4ul
            ? static_cast<uint32_t>(parsed) : 4u;
}

int gemm_bf16_weight_bf16_f32(
        cublasHandle_t handle,
        const uint16_t *weight,
        const uint32_t rows,
        const uint32_t cols,
        const float *input,
        float *out,
        const uint32_t columns,
        const float beta,
        uint16_t *activation_stage,
        const uint64_t activation_stage_capacity,
        const cudaStream_t stream) {
    if (!handle || !weight || !input || !out || !activation_stage || rows == 0u || cols == 0u || columns == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t activation_elements = 0u;
    if (!checked_mul(cols, columns, &activation_elements) || activation_elements > activation_stage_capacity ||
        activation_elements > std::numeric_limits<uint32_t>::max()) {
        return AXIOM_ERR_BUDGET;
    }
    f32_to_bf16_stage_kernel<<<static_cast<uint32_t>((activation_elements + kThreads - 1u) / kThreads),
                              kThreads, 0, stream>>>(input, activation_stage, activation_elements);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status == CUBLAS_STATUS_SUCCESS) {
        constexpr float alpha = 1.0f;
        status = cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(rows), static_cast<int>(columns), static_cast<int>(cols),
                &alpha,
                weight, CUDA_R_16BF, static_cast<int>(cols),
                activation_stage, CUDA_R_16BF, static_cast<int>(cols),
                &beta,
                out, CUDA_R_32F, static_cast<int>(rows),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    const char *debug = std::getenv("AXIOM_QWEN38_SPEC_DEBUG");
    if (status != CUBLAS_STATUS_SUCCESS && debug && debug[0] == '1' && debug[1] == '\0') {
        std::fprintf(stderr,
                     "qwen38_dspark_gemm cublas_status=%d rows=%u cols=%u columns=%u beta=%.9g\n",
                     static_cast<int>(status), rows, cols, columns, static_cast<double>(beta));
    }
    return cublas_status(status);
}

int view_pointer(
        const axiom_qwen38_dspark *dspark,
        const int device,
        const uint32_t layer,
        const axiom_qwen38_dspark_tensor tensor,
        const uint32_t rank,
        const uint64_t d0,
        const uint64_t d1,
        const uint16_t **out) {
    if (out) *out = nullptr;
    if (!dspark || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_dspark_tensor_view view{};
    view.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_qwen38_dspark_tensor_view_get(dspark, layer, tensor, &view);
    if (rc != AXIOM_OK) return rc;
    if (view.dtype != AXIOM_TENSOR_DTYPE_BF16 || view.rank != rank || view.shape[0] != d0 ||
        (rank == 2u && view.shape[1] != d1) || !view.device_buffer) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t count = d0;
    if (rank == 2u && !checked_mul(count, d1, &count)) return AXIOM_ERR_BUDGET;
    uint64_t bytes = 0u;
    if (!checked_mul(count, sizeof(uint16_t), &bytes) || view.byte_count != bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t buffer_device = 0u;
    rc = axiom_device_buffer_device_id(view.device_buffer, &buffer_device);
    if (rc != AXIOM_OK || buffer_device != static_cast<uint32_t>(device)) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    void *pointer = nullptr;
    rc = buffer_pointer(view.device_buffer, &pointer);
    if (rc != AXIOM_OK || !pointer) return rc == AXIOM_OK ? AXIOM_ERR_RUNTIME : rc;
    *out = static_cast<const uint16_t *>(pointer);
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_dspark_compute {
    const axiom_qwen38_dspark *dspark = nullptr;  // borrowed; owner outlives compute.
    axiom_runtime *runtime = nullptr;             // borrowed; owner outlives compute.
    int device = -1;
    uint32_t max_context = 0u;
    uint32_t committed_position = 0u;
    uint32_t transaction_anchor = 0u;
    uint32_t staged_end_position = 0u;
    bool transaction_open = false;
    bool transaction_has_staged_taps = false;
    bool device_session_active = false;
    bool device_graph_capture_active = false;
    bool proposal_graph_ready = false;
    uint64_t workspace_bytes = 0u;
    uint64_t kv_cache_bytes = 0u;
    uint64_t device_bytes = 0u;
    cublasHandle_t cublas = nullptr;
    axiom_qwen38_dspark_target_binding target{};

    const uint16_t *fc = nullptr;
    const uint16_t *hidden_norm = nullptr;
    const uint16_t *final_norm = nullptr;
    const uint16_t *confidence_weight = nullptr;
    const uint16_t *confidence_bias = nullptr;
    const uint16_t *markov_w1 = nullptr;
    const uint16_t *markov_w2 = nullptr;
    dspark_layer_weights layers[kLayers]{};

    axiom_device_buffer *fusion_input_buffer = nullptr;
    axiom_device_buffer *activation_stage_buffer = nullptr;
    axiom_device_buffer *cublas_workspace_buffer = nullptr;
    axiom_device_buffer *fused_buffer = nullptr;
    axiom_device_buffer *token_buffer = nullptr;
    axiom_device_buffer *hidden_buffer = nullptr;
    axiom_device_buffer *residual_buffer = nullptr;
    axiom_device_buffer *norm_buffer = nullptr;
    axiom_device_buffer *q_buffer = nullptr;
    axiom_device_buffer *k_buffer = nullptr;
    axiom_device_buffer *v_buffer = nullptr;
    axiom_device_buffer *attention_buffer = nullptr;
    axiom_device_buffer *attention_split_values_buffer = nullptr;
    axiom_device_buffer *attention_split_maxima_buffer = nullptr;
    axiom_device_buffer *attention_split_denominators_buffer = nullptr;
    axiom_device_buffer *linear_buffer = nullptr;
    axiom_device_buffer *gate_buffer = nullptr;
    axiom_device_buffer *up_buffer = nullptr;
    axiom_device_buffer *logits_buffer = nullptr;
    axiom_device_buffer *markov_bias_buffer = nullptr;
    axiom_device_buffer *rank_buffer = nullptr;
    axiom_device_buffer *greedy_block_values_buffer = nullptr;
    axiom_device_buffer *greedy_block_ids_buffer = nullptr;
    axiom_device_buffer *device_anchor_token_buffer = nullptr;
    axiom_device_buffer *device_anchor_position_buffer = nullptr;
    axiom_device_buffer *device_proposal_tokens_buffer = nullptr;
    axiom_device_buffer *device_verify_tokens_buffer = nullptr;
    axiom_device_buffer *device_accepted_prefix_buffer = nullptr;
    axiom_device_buffer *device_target_commit_prefix_buffer = nullptr;
    axiom_device_buffer *device_commit_limit_buffer = nullptr;
    axiom_device_buffer *device_continuation_token_buffer = nullptr;
    axiom_device_buffer *device_continuation_logit_buffer = nullptr;
    axiom_device_buffer *device_async_status_buffer = nullptr;
    axiom_device_buffer *device_terminal_state_buffer = nullptr;
    axiom_device_buffer *device_history_buffer = nullptr;
    axiom_device_buffer *k_cache_buffer[kLayers]{};
    axiom_device_buffer *v_cache_buffer[kLayers]{};

    float *fusion_input = nullptr;
    uint16_t *activation_stage = nullptr;
    uint64_t activation_stage_capacity = 0u;
    void *cublas_workspace = nullptr;
    float *fused = nullptr;
    uint32_t *tokens = nullptr;
    float *hidden = nullptr;
    float *residual = nullptr;
    float *norm = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *attention = nullptr;
    float *attention_split_values = nullptr;
    float *attention_split_maxima = nullptr;
    float *attention_split_denominators = nullptr;
    float *linear = nullptr;
    float *gate = nullptr;
    float *up = nullptr;
    float *logits = nullptr;
    float *markov_bias = nullptr;
    float *rank = nullptr;
    float *greedy_block_values = nullptr;
    uint32_t *greedy_block_ids = nullptr;
    uint32_t *device_anchor_token = nullptr;
    uint32_t *device_anchor_position = nullptr;
    uint32_t *device_proposal_tokens = nullptr;
    uint32_t *device_verify_tokens = nullptr;
    uint32_t *device_accepted_prefix = nullptr;
    uint32_t *device_target_commit_prefix = nullptr;
    uint32_t *device_commit_limit = nullptr;
    uint32_t *device_continuation_token = nullptr;
    float *device_continuation_logit = nullptr;
    uint32_t *device_async_status = nullptr;
    uint32_t *device_terminal_state = nullptr;
    axiom_qwen38_dspark_device_history *device_history = nullptr;
    uint32_t stop_token_count = 0u;
    uint32_t stop_token_ids[2]{};
    uint64_t device_control_bytes = 0u;
    cudaStream_t proposal_graph_capture_stream = nullptr;
    cudaGraph_t proposal_graph = nullptr;
    cudaGraphExec_t proposal_graph_exec = nullptr;
    uint16_t *k_cache[kLayers]{};
    uint16_t *v_cache[kLayers]{};
};

namespace {

void destroy_workspace(axiom_qwen38_dspark_compute *compute) {
    if (!compute) return;
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        axiom_device_buffer_destroy(compute->v_cache_buffer[layer]);
        axiom_device_buffer_destroy(compute->k_cache_buffer[layer]);
        compute->v_cache_buffer[layer] = nullptr;
        compute->k_cache_buffer[layer] = nullptr;
    }
    axiom_device_buffer_destroy(compute->greedy_block_ids_buffer);
    axiom_device_buffer_destroy(compute->greedy_block_values_buffer);
    axiom_device_buffer_destroy(compute->rank_buffer);
    axiom_device_buffer_destroy(compute->device_async_status_buffer);
    axiom_device_buffer_destroy(compute->device_terminal_state_buffer);
    axiom_device_buffer_destroy(compute->device_history_buffer);
    axiom_device_buffer_destroy(compute->device_continuation_logit_buffer);
    axiom_device_buffer_destroy(compute->device_continuation_token_buffer);
    axiom_device_buffer_destroy(compute->device_target_commit_prefix_buffer);
    axiom_device_buffer_destroy(compute->device_commit_limit_buffer);
    axiom_device_buffer_destroy(compute->device_accepted_prefix_buffer);
    axiom_device_buffer_destroy(compute->device_verify_tokens_buffer);
    axiom_device_buffer_destroy(compute->device_proposal_tokens_buffer);
    axiom_device_buffer_destroy(compute->device_anchor_position_buffer);
    axiom_device_buffer_destroy(compute->device_anchor_token_buffer);
    axiom_device_buffer_destroy(compute->markov_bias_buffer);
    axiom_device_buffer_destroy(compute->logits_buffer);
    axiom_device_buffer_destroy(compute->up_buffer);
    axiom_device_buffer_destroy(compute->gate_buffer);
    axiom_device_buffer_destroy(compute->linear_buffer);
    axiom_device_buffer_destroy(compute->attention_split_denominators_buffer);
    axiom_device_buffer_destroy(compute->attention_split_maxima_buffer);
    axiom_device_buffer_destroy(compute->attention_split_values_buffer);
    axiom_device_buffer_destroy(compute->attention_buffer);
    axiom_device_buffer_destroy(compute->v_buffer);
    axiom_device_buffer_destroy(compute->k_buffer);
    axiom_device_buffer_destroy(compute->q_buffer);
    axiom_device_buffer_destroy(compute->norm_buffer);
    axiom_device_buffer_destroy(compute->residual_buffer);
    axiom_device_buffer_destroy(compute->hidden_buffer);
    axiom_device_buffer_destroy(compute->token_buffer);
    axiom_device_buffer_destroy(compute->fused_buffer);
    axiom_device_buffer_destroy(compute->fusion_input_buffer);
    axiom_device_buffer_destroy(compute->activation_stage_buffer);
    axiom_device_buffer_destroy(compute->cublas_workspace_buffer);
    compute->greedy_block_ids_buffer = nullptr;
    compute->greedy_block_values_buffer = nullptr;
    compute->rank_buffer = nullptr;
    compute->device_async_status_buffer = nullptr;
    compute->device_terminal_state_buffer = nullptr;
    compute->device_history_buffer = nullptr;
    compute->device_continuation_logit_buffer = nullptr;
    compute->device_continuation_token_buffer = nullptr;
    compute->device_target_commit_prefix_buffer = nullptr;
    compute->device_commit_limit_buffer = nullptr;
    compute->device_accepted_prefix_buffer = nullptr;
    compute->device_verify_tokens_buffer = nullptr;
    compute->device_proposal_tokens_buffer = nullptr;
    compute->device_anchor_position_buffer = nullptr;
    compute->device_anchor_token_buffer = nullptr;
    compute->markov_bias_buffer = nullptr;
    compute->logits_buffer = nullptr;
    compute->up_buffer = nullptr;
    compute->gate_buffer = nullptr;
    compute->linear_buffer = nullptr;
    compute->attention_split_denominators_buffer = nullptr;
    compute->attention_split_maxima_buffer = nullptr;
    compute->attention_split_values_buffer = nullptr;
    compute->attention_buffer = nullptr;
    compute->v_buffer = nullptr;
    compute->k_buffer = nullptr;
    compute->q_buffer = nullptr;
    compute->norm_buffer = nullptr;
    compute->residual_buffer = nullptr;
    compute->hidden_buffer = nullptr;
    compute->token_buffer = nullptr;
    compute->fused_buffer = nullptr;
    compute->fusion_input_buffer = nullptr;
    compute->activation_stage_buffer = nullptr;
    compute->cublas_workspace_buffer = nullptr;
    compute->greedy_block_values = nullptr;
    compute->greedy_block_ids = nullptr;
    compute->device_anchor_token = nullptr;
    compute->device_anchor_position = nullptr;
    compute->device_proposal_tokens = nullptr;
    compute->device_verify_tokens = nullptr;
    compute->device_accepted_prefix = nullptr;
    compute->device_target_commit_prefix = nullptr;
    compute->device_commit_limit = nullptr;
    compute->device_continuation_token = nullptr;
    compute->device_continuation_logit = nullptr;
    compute->device_async_status = nullptr;
    compute->device_terminal_state = nullptr;
    compute->device_history = nullptr;
    compute->device_control_bytes = 0u;
}

int allocate_buffer(
        axiom_qwen38_dspark_compute *compute,
        axiom_device_buffer **buffer,
        void **pointer,
        const uint64_t bytes,
        uint64_t *account) {
    if (!compute || !buffer || !pointer || !account || bytes == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    *buffer = nullptr;
    *pointer = nullptr;
    int rc = axiom_device_buffer_create(compute->runtime, buffer, bytes);
    if (rc != AXIOM_OK) return rc;
    rc = buffer_pointer(*buffer, pointer);
    if (rc == AXIOM_OK && !*pointer) rc = AXIOM_ERR_RUNTIME;
    if (rc == AXIOM_OK) rc = add_bytes(account, bytes);
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(*buffer);
        *buffer = nullptr;
        *pointer = nullptr;
    }
    return rc;
}

int allocate_workspace(axiom_qwen38_dspark_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t hidden_block = 0u;
    uint64_t intermediate_block = 0u;
    uint64_t fusion_block = 0u;
    uint64_t kv_inject_block = 0u;
    uint64_t logits_block = 0u;
    uint64_t cache_block = 0u;
    uint64_t attention_split_values_block = 0u;
    uint64_t attention_split_stats_block = 0u;
    if (!checked_mul(static_cast<uint64_t>(kHidden) * kBlock, sizeof(float), &hidden_block) ||
        !checked_mul(static_cast<uint64_t>(kIntermediate) * kBlock, sizeof(float), &intermediate_block) ||
        !checked_mul(static_cast<uint64_t>(kFusionInput) * kVerifyWidth, sizeof(float), &fusion_block) ||
        !checked_mul(static_cast<uint64_t>(kKvDim) * kVerifyWidth, sizeof(float), &kv_inject_block) ||
        !checked_mul(static_cast<uint64_t>(kVocab) * kBlock, sizeof(float), &logits_block) ||
        !checked_mul(static_cast<uint64_t>(kBlock) * kHeads * kAttentionMaxSplitK,
                     kHeadDim * sizeof(float), &attention_split_values_block) ||
        !checked_mul(static_cast<uint64_t>(kBlock) * kHeads * kAttentionMaxSplitK,
                     sizeof(float), &attention_split_stats_block) ||
        !checked_mul(static_cast<uint64_t>(compute->max_context) * kKvDim, sizeof(uint16_t), &cache_block)) {
        return AXIOM_ERR_BUDGET;
    }
    const uint64_t fused_block = static_cast<uint64_t>(kHidden) * kVerifyWidth * sizeof(float);
    const uint64_t activation_stage_block = static_cast<uint64_t>(kFusionInput) * kVerifyWidth * sizeof(uint16_t);
    const auto allocate_device_control = [&](axiom_device_buffer **buffer, void **pointer, uint64_t bytes) -> int {
        const int local_rc = allocate_buffer(compute, buffer, pointer, bytes, &compute->workspace_bytes);
        if (local_rc != AXIOM_OK) return local_rc;
        return add_bytes(&compute->device_control_bytes, bytes);
    };
    int rc = allocate_buffer(compute, &compute->fusion_input_buffer,
                             reinterpret_cast<void **>(&compute->fusion_input), fusion_block,
                             &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->activation_stage_buffer,
                                              reinterpret_cast<void **>(&compute->activation_stage),
                                              activation_stage_block, &compute->workspace_bytes);
    if (rc == AXIOM_OK) compute->activation_stage_capacity =
            static_cast<uint64_t>(kFusionInput) * kVerifyWidth;
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->cublas_workspace_buffer,
                                              &compute->cublas_workspace, kCublasWorkspaceBytes,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->fused_buffer,
                                              reinterpret_cast<void **>(&compute->fused), fused_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->token_buffer,
                                              reinterpret_cast<void **>(&compute->tokens),
                                              static_cast<uint64_t>(kBlock) * sizeof(uint32_t),
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->hidden_buffer,
                                              reinterpret_cast<void **>(&compute->hidden), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->residual_buffer,
                                              reinterpret_cast<void **>(&compute->residual), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->norm_buffer,
                                              reinterpret_cast<void **>(&compute->norm), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->q_buffer,
                                              reinterpret_cast<void **>(&compute->q), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->k_buffer,
                                              reinterpret_cast<void **>(&compute->k), kv_inject_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->v_buffer,
                                              reinterpret_cast<void **>(&compute->v), kv_inject_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->attention_buffer,
                                              reinterpret_cast<void **>(&compute->attention), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(
            compute, &compute->attention_split_values_buffer,
            reinterpret_cast<void **>(&compute->attention_split_values),
            attention_split_values_block, &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(
            compute, &compute->attention_split_maxima_buffer,
            reinterpret_cast<void **>(&compute->attention_split_maxima),
            attention_split_stats_block, &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(
            compute, &compute->attention_split_denominators_buffer,
            reinterpret_cast<void **>(&compute->attention_split_denominators),
            attention_split_stats_block, &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->linear_buffer,
                                              reinterpret_cast<void **>(&compute->linear), hidden_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->gate_buffer,
                                              reinterpret_cast<void **>(&compute->gate), intermediate_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->up_buffer,
                                              reinterpret_cast<void **>(&compute->up), intermediate_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->logits_buffer,
                                              reinterpret_cast<void **>(&compute->logits), logits_block,
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->markov_bias_buffer,
                                              reinterpret_cast<void **>(&compute->markov_bias),
                                              static_cast<uint64_t>(kVocab) * sizeof(float),
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(compute, &compute->rank_buffer,
                                              reinterpret_cast<void **>(&compute->rank),
                                              static_cast<uint64_t>(kRank) * sizeof(float),
                                              &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(
            compute, &compute->greedy_block_values_buffer,
            reinterpret_cast<void **>(&compute->greedy_block_values),
            static_cast<uint64_t>(kGreedyBlocks) * sizeof(float),
            &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_buffer(
            compute, &compute->greedy_block_ids_buffer,
            reinterpret_cast<void **>(&compute->greedy_block_ids),
            static_cast<uint64_t>(kGreedyBlocks) * sizeof(uint32_t),
            &compute->workspace_bytes);
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_anchor_token_buffer,
            reinterpret_cast<void **>(&compute->device_anchor_token), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_anchor_position_buffer,
            reinterpret_cast<void **>(&compute->device_anchor_position), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_proposal_tokens_buffer,
            reinterpret_cast<void **>(&compute->device_proposal_tokens),
            static_cast<uint64_t>(kBlock) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_verify_tokens_buffer,
            reinterpret_cast<void **>(&compute->device_verify_tokens),
            static_cast<uint64_t>(kVerifyWidth) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_accepted_prefix_buffer,
            reinterpret_cast<void **>(&compute->device_accepted_prefix), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_target_commit_prefix_buffer,
            reinterpret_cast<void **>(&compute->device_target_commit_prefix), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_commit_limit_buffer,
            reinterpret_cast<void **>(&compute->device_commit_limit), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_continuation_token_buffer,
            reinterpret_cast<void **>(&compute->device_continuation_token), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_continuation_logit_buffer,
            reinterpret_cast<void **>(&compute->device_continuation_logit), sizeof(float));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_async_status_buffer,
            reinterpret_cast<void **>(&compute->device_async_status), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_terminal_state_buffer,
            reinterpret_cast<void **>(&compute->device_terminal_state), sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = allocate_device_control(
            &compute->device_history_buffer,
            reinterpret_cast<void **>(&compute->device_history),
            sizeof(axiom_qwen38_dspark_device_history));
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        rc = allocate_buffer(compute, &compute->k_cache_buffer[layer],
                             reinterpret_cast<void **>(&compute->k_cache[layer]), cache_block,
                             &compute->kv_cache_bytes);
        if (rc == AXIOM_OK) {
            rc = allocate_buffer(compute, &compute->v_cache_buffer[layer],
                                 reinterpret_cast<void **>(&compute->v_cache[layer]), cache_block,
                                 &compute->kv_cache_bytes);
        }
    }
    if (rc == AXIOM_OK) {
        const uint32_t default_commit_limit = kVerifyWidth;
        const cudaError_t status = cudaMemcpy(
                compute->device_commit_limit, &default_commit_limit,
                sizeof(default_commit_limit), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) rc = cuda_status(status);
    }
    if (rc == AXIOM_OK) {
        const cudaError_t status = cudaMemset(
                compute->device_terminal_state, 0, sizeof(uint32_t));
        if (status != cudaSuccess) rc = cuda_status(status);
    }
    if (rc == AXIOM_OK && !checked_add(compute->workspace_bytes, compute->kv_cache_bytes,
                                       &compute->device_bytes)) {
        rc = AXIOM_ERR_BUDGET;
    }
    if (rc == AXIOM_OK) rc = cublas_status(cublasSetWorkspace(
            compute->cublas, compute->cublas_workspace, static_cast<size_t>(kCublasWorkspaceBytes)));
    return rc;
}

int load_weights(axiom_qwen38_dspark_compute *compute) {
    if (!compute) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = view_pointer(compute->dspark, compute->device, AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                          AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT, 2u, kHidden, kFusionInput,
                          &compute->fc);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_HIDDEN_NORM_WEIGHT,
                                          1u, kHidden, 0u, &compute->hidden_norm);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_FINAL_NORM_WEIGHT,
                                          1u, kHidden, 0u, &compute->final_norm);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT,
                                          2u, 1u, AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES,
                                          &compute->confidence_weight);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS,
                                          1u, 1u, 0u, &compute->confidence_bias);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W1_WEIGHT,
                                          2u, kVocab, kRank, &compute->markov_w1);
    if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device,
                                          AXIOM_QWEN38_DSPARK_GLOBAL_LAYER,
                                          AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W2_WEIGHT,
                                          2u, kVocab, kRank, &compute->markov_w2);
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        dspark_layer_weights &weights = compute->layers[layer];
        rc = view_pointer(compute->dspark, compute->device, layer,
                          AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT,
                          1u, kHidden, 0u, &weights.input_norm);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_DOWN_PROJ_WEIGHT,
                                               2u, kHidden, kIntermediate, &weights.mlp_down);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_GATE_PROJ_WEIGHT,
                                               2u, kIntermediate, kHidden, &weights.mlp_gate);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_UP_PROJ_WEIGHT,
                                               2u, kIntermediate, kHidden, &weights.mlp_up);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_POST_ATTENTION_LAYERNORM_WEIGHT,
                                               1u, kHidden, 0u, &weights.post_attention_norm);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_NORM_WEIGHT,
                                               1u, kHeadDim, 0u, &weights.k_norm);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_PROJ_WEIGHT,
                                               2u, kKvDim, kHidden, &weights.k_proj);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_O_PROJ_WEIGHT,
                                               2u, kHidden, kHidden, &weights.o_proj);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_NORM_WEIGHT,
                                               1u, kHeadDim, 0u, &weights.q_norm);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_PROJ_WEIGHT,
                                               2u, kHidden, kHidden, &weights.q_proj);
        if (rc == AXIOM_OK) rc = view_pointer(compute->dspark, compute->device, layer,
                                               AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_V_PROJ_WEIGHT,
                                               2u, kKvDim, kHidden, &weights.v_proj);
    }
    return rc;
}

/* Shared proposal body for the eager ABI and the fixed-shape CUDA graph.  The
 * graph variant reads anchor/position from resident device scalars so replay
 * does not bake either value into a launch parameter. */
int enqueue_proposal(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t anchor_token,
        const uint32_t anchor_position,
        const uint32_t *anchor_token_device,
        const uint32_t *anchor_position_device,
        const uint32_t columns,
        uint32_t *out_tokens,
        float *out_logits,
        float *out_confidence,
        const cudaStream_t stream) {
    if (!compute || !out_tokens || columns == 0u || columns > kBlock) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t attention_tile = dspark_attention_tile_keys();
    const uint32_t attention_split_k = dspark_attention_split_k();
    fill_noise_block_kernel<<<(columns + 31u) / 32u, 32u, 0, stream>>>(
            compute->tokens, anchor_token, columns, anchor_token_device);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = compute->target.embed_f32_device(
            compute->target.user_data, compute->tokens, compute->hidden, columns,
            reinterpret_cast<void *>(stream));
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        const dspark_layer_weights &weights = compute->layers[layer];
        rc = launch_rmsnorm(weights.input_norm, compute->hidden, compute->norm, kHidden, columns, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.q_proj, kHidden, kHidden, compute->norm, compute->q,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.k_proj, kKvDim, kHidden, compute->norm, compute->k,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.v_proj, kKvDim, kHidden, compute->norm, compute->v,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->q, static_cast<uint64_t>(kHidden) * columns, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->k, static_cast<uint64_t>(kKvDim) * columns, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->v, static_cast<uint64_t>(kKvDim) * columns, stream);
        if (rc == AXIOM_OK) rc = launch_qk_norm_rope(
                weights.q_norm, compute->q, kHeads, columns, anchor_position,
                anchor_position_device, stream);
        if (rc == AXIOM_OK) rc = launch_qk_norm_rope(
                weights.k_norm, compute->k, kKvHeads, columns, anchor_position,
                anchor_position_device, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->q, static_cast<uint64_t>(kHidden) * columns, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->k, static_cast<uint64_t>(kKvDim) * columns, stream);
        if (rc == AXIOM_OK) {
            const uint32_t blocks = kKvHeads * kGqaHeadGroups * columns;
            if (attention_split_k == 2u) {
                noncausal_gqa_attention_bf16_wide_kernel<32u, 2u><<<
                        dim3(blocks, 2u), kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention,
                        compute->attention_split_values, compute->attention_split_maxima,
                        compute->attention_split_denominators, anchor_position,
                        columns, anchor_position_device);
                if (cudaGetLastError() == cudaSuccess) {
                    merge_noncausal_gqa_attention_split_kernel<2u><<<
                            columns * kHeads, kHeadDim, 0, stream>>>(
                            compute->attention_split_values, compute->attention_split_maxima,
                            compute->attention_split_denominators, compute->attention, columns);
                } else {
                    rc = AXIOM_ERR_CUDA;
                }
            } else if (attention_split_k == 4u) {
                noncausal_gqa_attention_bf16_wide_kernel<32u, 4u><<<
                        dim3(blocks, 4u), kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention,
                        compute->attention_split_values, compute->attention_split_maxima,
                        compute->attention_split_denominators, anchor_position,
                        columns, anchor_position_device);
                if (cudaGetLastError() == cudaSuccess) {
                    merge_noncausal_gqa_attention_split_kernel<4u><<<
                            columns * kHeads, kHeadDim, 0, stream>>>(
                            compute->attention_split_values, compute->attention_split_maxima,
                            compute->attention_split_denominators, compute->attention, columns);
                } else {
                    rc = AXIOM_ERR_CUDA;
                }
            } else if (attention_tile == 8u) {
                noncausal_gqa_attention_bf16_wide_kernel<8u><<<
                        blocks, kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention, nullptr, nullptr, nullptr,
                        anchor_position,
                        columns, anchor_position_device);
            } else if (attention_tile == 16u) {
                noncausal_gqa_attention_bf16_wide_kernel<16u><<<
                        blocks, kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention, nullptr, nullptr, nullptr,
                        anchor_position,
                        columns, anchor_position_device);
            } else if (attention_tile == 32u) {
                noncausal_gqa_attention_bf16_wide_kernel<32u><<<
                        blocks, kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention, nullptr, nullptr, nullptr,
                        anchor_position,
                        columns, anchor_position_device);
            } else {
                noncausal_gqa_attention_bf16_kernel<<<
                        blocks, kGqaBlockThreads, 0, stream>>>(
                        compute->q, compute->k, compute->v, compute->k_cache[layer],
                        compute->v_cache[layer], compute->attention, anchor_position,
                        columns, anchor_position_device);
            }
            if (rc == AXIOM_OK && cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.o_proj, kHidden, kHidden, compute->attention, compute->linear,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * columns;
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->linear, hidden_count, stream);
        if (rc == AXIOM_OK) {
            residual_add_kernel<<<static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                    compute->hidden, compute->linear, compute->residual, hidden_count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->residual, hidden_count, stream);
        if (rc == AXIOM_OK) rc = launch_rmsnorm(
                weights.post_attention_norm, compute->residual, compute->norm, kHidden, columns, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.mlp_gate, kIntermediate, kHidden, compute->norm, compute->gate,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.mlp_up, kIntermediate, kHidden, compute->norm, compute->up,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        const uint64_t intermediate_count = static_cast<uint64_t>(kIntermediate) * columns;
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->gate, intermediate_count, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->up, intermediate_count, stream);
        if (rc == AXIOM_OK) {
            silu_multiply_kernel<<<static_cast<uint32_t>((intermediate_count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                    compute->gate, compute->up, intermediate_count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.mlp_down, kHidden, kIntermediate, compute->gate, compute->linear,
                columns, 0.0f, compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->linear, hidden_count, stream);
        if (rc == AXIOM_OK) {
            residual_add_kernel<<<static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                    compute->residual, compute->linear, compute->hidden, hidden_count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = launch_round_bf16(compute->hidden, hidden_count, stream);
    }
    if (rc == AXIOM_OK) rc = launch_rmsnorm(compute->final_norm, compute->hidden, compute->residual,
                                             kHidden, columns, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->residual, static_cast<uint64_t>(kHidden) * columns, stream);
    if (rc == AXIOM_OK) rc = compute->target.lm_head_f32_device(
            compute->target.user_data, compute->residual, compute->logits, columns,
            reinterpret_cast<void *>(stream));
    const bool fused_top1 = fused_top1_enabled();
    for (uint32_t column = 0u; column < columns && rc == AXIOM_OK; ++column) {
        const uint32_t *previous = column == 0u ? compute->tokens : out_tokens + column - 1u;
        gather_markov_rank_kernel<<<(kRank + 127u) / 128u, 128u, 0, stream>>>(
                compute->markov_w1, previous, compute->rank);
        if (cudaGetLastError() != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
            break;
        }
        float *logits = compute->logits + static_cast<uint64_t>(column) * kVocab;
        rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, compute->markov_w2, kVocab, kRank, compute->rank,
                compute->markov_bias, 1u, 0.0f,
                compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc != AXIOM_OK) break;
        if (fused_top1) {
            add_bf16_rounded_top1_stage1_kernel<<<kGreedyBlocks, kThreads, 0, stream>>>(
                    logits, compute->markov_bias,
                    compute->greedy_block_values, compute->greedy_block_ids);
            if (cudaGetLastError() != cudaSuccess) {
                rc = AXIOM_ERR_CUDA;
                break;
            }
            top1_stage2_kernel<<<1u, kThreads, 0, stream>>>(
                    compute->greedy_block_values, compute->greedy_block_ids,
                    out_tokens + column);
        } else {
            add_bf16_rounded_kernel<<<(kVocab + kThreads - 1u) / kThreads, kThreads, 0, stream>>>(
                    logits, compute->markov_bias, kVocab);
            if (cudaGetLastError() != cudaSuccess) {
                rc = AXIOM_ERR_CUDA;
                break;
            }
            greedy_top1_kernel<<<1u, kThreads, 0, stream>>>(
                    logits, out_tokens + column);
        }
        if (cudaGetLastError() != cudaSuccess) {
            rc = AXIOM_ERR_CUDA;
            break;
        }
        if (out_confidence) {
            confidence_kernel<<<1u, kThreads, 0, stream>>>(
                    compute->residual + static_cast<uint64_t>(column) * kHidden,
                    compute->rank, compute->confidence_weight, compute->confidence_bias,
                    out_confidence + column);
            if (cudaGetLastError() != cudaSuccess) {
                rc = AXIOM_ERR_CUDA;
                break;
            }
        }
    }
    if (rc == AXIOM_OK && out_logits) {
        const uint64_t bytes = static_cast<uint64_t>(kVocab) * columns * sizeof(float);
        const cudaError_t copy = cudaMemcpyAsync(out_logits, compute->logits, static_cast<size_t>(bytes),
                                                  cudaMemcpyDeviceToDevice, stream);
        if (copy != cudaSuccess) rc = cuda_status(copy);
    }
    return rc;
}

}  // namespace

extern "C" void axiom_qwen38_dspark_compute_destroy(axiom_qwen38_dspark_compute *compute) {
    if (!compute) return;
    if (compute->device >= 0) (void) cudaSetDevice(compute->device);
    if (compute->proposal_graph_exec) (void) cudaGraphExecDestroy(compute->proposal_graph_exec);
    if (compute->proposal_graph) (void) cudaGraphDestroy(compute->proposal_graph);
    if (compute->proposal_graph_capture_stream) {
        (void) cudaStreamDestroy(compute->proposal_graph_capture_stream);
    }
    compute->proposal_graph_exec = nullptr;
    compute->proposal_graph = nullptr;
    compute->proposal_graph_capture_stream = nullptr;
    compute->proposal_graph_ready = false;
    if (compute->cublas) (void) cublasDestroy(compute->cublas);
    compute->cublas = nullptr;
    destroy_workspace(compute);
    delete compute;
}

extern "C" int axiom_qwen38_dspark_compute_create(
        const axiom_qwen38_dspark *,
        axiom_runtime *,
        const int,
        const axiom_qwen38_dspark_compute_config *,
        const axiom_qwen38_dspark_target_binding *,
        axiom_qwen38_dspark_compute **out) {
    if (out) *out = nullptr;
    return AXIOM_ERR_INVALID_ARGUMENT;
}

extern "C" int axiom_qwen38_dspark_compute_create_v2(
        const axiom_qwen38_dspark *dspark,
        axiom_runtime *runtime,
        const int device,
        const axiom_qwen38_dspark_compute_config *config,
        const uint64_t config_bytes,
        const axiom_qwen38_dspark_target_binding *target,
        axiom_qwen38_dspark_compute **out) {
    if (out) *out = nullptr;
    if (config_bytes != sizeof(axiom_qwen38_dspark_compute_config)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!dspark || !runtime || device < 0 || !config || !target || !out ||
        config->abi_version != AXIOM_ABI_VERSION || target->abi_version != AXIOM_ABI_VERSION ||
        config->max_context == 0u || config->max_context > 262144u ||
        config->stop_token_count > 2u ||
        !target->embed_f32_device || !target->lm_head_f32_device) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0u; index < config->stop_token_count; ++index) {
        if (config->stop_token_ids[index] >= kVocab) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t runtime_device = 0u;
    int rc = axiom_runtime_device_id(runtime, &runtime_device);
    if (rc != AXIOM_OK || runtime_device != static_cast<uint32_t>(device)) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    axiom_qwen38_dspark_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_qwen38_dspark_info_get(dspark, &info);
    if (rc != AXIOM_OK || info.hidden_size != kHidden || info.intermediate_size != kIntermediate ||
        info.draft_layers != kLayers || info.attention_heads != kHeads ||
        info.key_value_heads != kKvHeads || info.head_dim != kHeadDim || info.vocab_size != kVocab ||
        info.block_size != kBlock || info.verify_width != kVerifyWidth ||
        info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_dspark_compute *compute = new (std::nothrow) axiom_qwen38_dspark_compute();
    if (!compute) return AXIOM_ERR_BUDGET;
    compute->dspark = dspark;
    compute->runtime = runtime;
    compute->device = device;
    compute->max_context = config->max_context;
    compute->stop_token_count = config->stop_token_count;
    for (uint32_t index = 0u; index < config->stop_token_count; ++index) {
        compute->stop_token_ids[index] = config->stop_token_ids[index];
    }
    compute->target = *target;
    rc = cublas_status(cublasCreate(&compute->cublas));
    if (rc == AXIOM_OK) rc = cublas_status(cublasSetPointerMode(compute->cublas, CUBLAS_POINTER_MODE_HOST));
    if (rc == AXIOM_OK) rc = load_weights(compute);
    if (rc == AXIOM_OK) rc = allocate_workspace(compute);
    if (rc != AXIOM_OK) {
        axiom_qwen38_dspark_compute_destroy(compute);
        return rc;
    }
    *out = compute;
    return AXIOM_OK;
}

extern "C" uint64_t axiom_qwen38_dspark_compute_device_bytes(
        const axiom_qwen38_dspark_compute *compute) {
    return compute ? compute->device_bytes : 0u;
}

extern "C" int axiom_qwen38_dspark_compute_info_get(
        const axiom_qwen38_dspark_compute *compute,
        axiom_qwen38_dspark_compute_info *out) {
    if (!compute || !out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->device = static_cast<uint32_t>(compute->device);
    out->max_context = compute->max_context;
    out->hidden_size = kHidden;
    out->attention_heads = kHeads;
    out->key_value_heads = kKvHeads;
    out->head_dim = kHeadDim;
    out->draft_layers = kLayers;
    out->block_size = kBlock;
    out->cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
    out->explicit_streams = 1u;
    out->transaction_open = compute->transaction_open ? 1u : 0u;
    out->committed_position = compute->committed_position;
    out->checkpoint_device_bytes = axiom_qwen38_dspark_device_bytes(compute->dspark);
    out->workspace_device_bytes = compute->workspace_bytes;
    out->kv_cache_device_bytes = compute->kv_cache_bytes;
    out->device_bytes = compute->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_restore_position(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t position) {
    if (!compute || compute->transaction_open || compute->device_session_active ||
        position > compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    compute->committed_position = position;
    compute->transaction_anchor = position;
    compute->staged_end_position = position;
    compute->transaction_has_staged_taps = false;
    return AXIOM_OK;
}

static int dspark_kv_page_args(
        const axiom_qwen38_dspark_compute *compute,
        const uint32_t layer,
        const uint32_t logical_page,
        const void *host_page,
        const uint64_t host_page_bytes) {
    if (!compute || layer >= kLayers || !host_page || host_page_bytes != kKvPageBytes ||
        logical_page >= (compute->max_context + kKvPageTokens - 1u) / kKvPageTokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_kv_page_export(
        const axiom_qwen38_dspark_compute *compute,
        const uint32_t layer,
        const uint32_t logical_page,
        void *host_page,
        const uint64_t host_page_bytes) {
    int rc = dspark_kv_page_args(compute, layer, logical_page, host_page, host_page_bytes);
    if (rc != AXIOM_OK || compute->transaction_open) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t page_elements = static_cast<uint64_t>(kKvPageTokens) * kKvDim;
    const uint64_t page_start = static_cast<uint64_t>(logical_page) * kKvPageTokens;
    const uint64_t valid_tokens = std::min<uint64_t>(kKvPageTokens, compute->max_context - page_start);
    const size_t valid_elements = static_cast<size_t>(valid_tokens * kKvDim);
    std::vector<uint16_t> k_plane;
    std::vector<uint16_t> v_plane;
    try {
        k_plane.resize(static_cast<size_t>(page_elements));
        v_plane.resize(static_cast<size_t>(page_elements));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    const size_t valid_bytes = valid_elements * sizeof(uint16_t);
    const uint64_t device_offset = page_start * kKvDim;
    cudaError_t status = cudaMemcpy(
            k_plane.data(), compute->k_cache[layer] + device_offset,
            valid_bytes, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
                v_plane.data(), compute->v_cache[layer] + device_offset,
                valid_bytes, cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess) return cuda_status(status);
    std::memset(host_page, 0, static_cast<size_t>(host_page_bytes));
    uint8_t *target_k = static_cast<uint8_t *>(host_page);
    uint8_t *target_v = target_k + kKvPageBytes / 2u;
    for (size_t index = 0u; index < valid_elements; ++index) {
        target_k[index] = float_host_to_e4m3fn(bf16_host_to_float(k_plane[index]));
        target_v[index] = float_host_to_e4m3fn(bf16_host_to_float(v_plane[index]));
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_kv_page_import(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t layer,
        const uint32_t logical_page,
        const void *host_page,
        const uint64_t host_page_bytes) {
    int rc = dspark_kv_page_args(compute, layer, logical_page, host_page, host_page_bytes);
    if (rc != AXIOM_OK || compute->transaction_open) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t page_elements = static_cast<uint64_t>(kKvPageTokens) * kKvDim;
    const uint64_t page_start = static_cast<uint64_t>(logical_page) * kKvPageTokens;
    const uint64_t valid_tokens = std::min<uint64_t>(kKvPageTokens, compute->max_context - page_start);
    const size_t valid_elements = static_cast<size_t>(valid_tokens * kKvDim);
    std::vector<uint16_t> k_plane;
    std::vector<uint16_t> v_plane;
    try {
        k_plane.resize(static_cast<size_t>(page_elements));
        v_plane.resize(static_cast<size_t>(page_elements));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    const uint8_t *source_k = static_cast<const uint8_t *>(host_page);
    const uint8_t *source_v = source_k + kKvPageBytes / 2u;
    for (size_t index = 0u; index < valid_elements; ++index) {
        k_plane[index] = float_host_to_bf16(e4m3fn_to_float(source_k[index]));
        v_plane[index] = float_host_to_bf16(e4m3fn_to_float(source_v[index]));
    }
    const size_t valid_bytes = valid_elements * sizeof(uint16_t);
    const uint64_t device_offset = page_start * kKvDim;
    cudaError_t status = cudaMemcpy(
            compute->k_cache[layer] + device_offset, k_plane.data(),
            valid_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
                compute->v_cache[layer] + device_offset, v_plane.data(),
                valid_bytes, cudaMemcpyHostToDevice);
    }
    return cuda_status(status);
}

extern "C" int axiom_qwen38_dspark_compute_transaction_begin(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t anchor_position,
        void *stream) {
    (void) stream;
    if (!compute || compute->transaction_open || anchor_position >= compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (anchor_position != compute->committed_position) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    compute->transaction_open = true;
    compute->transaction_anchor = anchor_position;
    compute->staged_end_position = anchor_position;
    compute->transaction_has_staged_taps = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_transaction_commit(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t accepted_prefix,
        void *stream) {
    (void) stream;
    if (!compute || !compute->transaction_open || accepted_prefix > kBlock) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t next = static_cast<uint64_t>(compute->transaction_anchor) + 1u + accepted_prefix;
    if (next > compute->max_context) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!compute->transaction_has_staged_taps || compute->staged_end_position < next) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    compute->committed_position = static_cast<uint32_t>(next);
    compute->transaction_open = false;
    compute->transaction_has_staged_taps = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_transaction_abort(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    (void) stream;
    if (!compute || !compute->transaction_open) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    compute->committed_position = compute->transaction_anchor;
    compute->transaction_open = false;
    compute->transaction_has_staged_taps = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_inject_target(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_inject_request *request) {
    if (!compute || !request || request->abi_version != AXIOM_ABI_VERSION || !request->target_taps ||
        request->columns == 0u || request->columns > kVerifyWidth ||
        request->target_position >= compute->max_context ||
        static_cast<uint64_t>(request->target_position) + request->columns > compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (compute->transaction_open && request->target_position != compute->transaction_anchor) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!compute->transaction_open && request->target_position != compute->committed_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    const uint64_t tap_count = static_cast<uint64_t>(kFusionInput) * request->columns;
    pack_target_taps_kernel<<<static_cast<uint32_t>((tap_count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
            request->target_taps, compute->fusion_input, request->columns);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = gemm_bf16_weight_bf16_f32(compute->cublas, compute->fc, kHidden, kFusionInput,
                                        compute->fusion_input, compute->fused, request->columns, 0.0f,
                                        compute->activation_stage, compute->activation_stage_capacity, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->fused, static_cast<uint64_t>(kHidden) * request->columns, stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(compute->hidden_norm, compute->fused, compute->fused,
                                             kHidden, request->columns, stream);
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        const dspark_layer_weights &weights = compute->layers[layer];
        rc = gemm_bf16_weight_bf16_f32(compute->cublas, weights.k_proj, kKvDim, kHidden,
                                        compute->fused, compute->k, request->columns, 0.0f,
                                        compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.v_proj, kKvDim, kHidden, compute->fused, compute->v,
                request->columns, 0.0f, compute->activation_stage,
                compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(
                compute->k, static_cast<uint64_t>(kKvDim) * request->columns, stream);
        if (rc == AXIOM_OK) rc = launch_qk_norm_rope(weights.k_norm, compute->k, kKvHeads,
                                                      request->columns, request->target_position, nullptr, stream);
        if (rc == AXIOM_OK) {
            const uint64_t count = static_cast<uint64_t>(kKvDim) * request->columns;
            store_kv_bf16_kernel<<<static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                    compute->k, compute->v, compute->k_cache[layer], compute->v_cache[layer],
                    request->target_position, request->columns, nullptr,
                    compute->max_context, nullptr);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
    }
    if (rc == AXIOM_OK && compute->transaction_open) {
        compute->transaction_has_staged_taps = true;
        compute->staged_end_position = request->target_position + request->columns;
    } else if (rc == AXIOM_OK) {
        const uint32_t next = request->target_position + request->columns;
        if (next > compute->committed_position) compute->committed_position = next;
    }
    return rc;
}

extern "C" int axiom_qwen38_dspark_compute_propose(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_propose_request *request) {
    if (!compute || !request || request->abi_version != AXIOM_ABI_VERSION || !request->out_tokens ||
        request->proposal_tokens == 0u || request->proposal_tokens > kBlock ||
        request->anchor_position >= compute->max_context ||
        static_cast<uint64_t>(request->anchor_position) + request->proposal_tokens > compute->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!compute->transaction_open || request->anchor_position != compute->transaction_anchor) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    return enqueue_proposal(compute, request->anchor_token, request->anchor_position,
                            nullptr, nullptr, request->proposal_tokens,
                            request->out_tokens, request->out_logits,
                            request->out_confidence, stream);
}

namespace {

bool device_controls_ready(const axiom_qwen38_dspark_compute *compute) {
    return compute && compute->device_anchor_token && compute->device_anchor_position &&
            compute->device_proposal_tokens && compute->device_verify_tokens &&
            compute->device_accepted_prefix && compute->device_target_commit_prefix &&
            compute->device_commit_limit &&
            compute->device_continuation_token &&
            compute->device_continuation_logit && compute->device_async_status &&
            compute->device_terminal_state && compute->device_history;
}

bool device_operation_active(const axiom_qwen38_dspark_compute *compute) {
    return compute && (compute->device_session_active || compute->device_graph_capture_active);
}

__global__ void pack_device_history_kernel(
        const uint32_t *proposal_tokens,
        const uint32_t *accepted_prefix,
        const uint32_t *continuation_token,
        const uint32_t *async_status,
        const uint32_t *next_position,
        const uint32_t *committed_tokens,
        axiom_qwen38_dspark_device_history *output) {
    const uint32_t index = static_cast<uint32_t>(threadIdx.x);
    if (index < 7u) output->proposal_tokens[index] = proposal_tokens[index];
    if (index == 0u) {
        output->accepted_prefix = accepted_prefix[0];
        output->continuation_token = continuation_token[0];
        output->async_status = async_status[0];
        output->next_position = next_position[0];
        output->committed_tokens = committed_tokens[0];
    }
}

int clear_device_step_state(axiom_qwen38_dspark_compute *compute, cudaStream_t stream) {
    if (!device_controls_ready(compute)) return AXIOM_ERR_RUNTIME;
    cudaError_t status = cudaMemsetAsync(compute->device_accepted_prefix, 0, sizeof(uint32_t), stream);
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(compute->device_target_commit_prefix, 0, sizeof(uint32_t), stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(compute->device_continuation_token, 0, sizeof(uint32_t), stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(compute->device_continuation_logit, 0, sizeof(float), stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemsetAsync(compute->device_async_status, 0, sizeof(uint32_t), stream);
    }
    return cuda_status(status);
}

bool valid_device_target_view(const axiom_qwen38_dspark_compute_device_target_view *target) {
    return target && target->abi_version == AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION &&
            target->target_taps_device && target->target_token_ids_device &&
            target->target_logits_device &&
            target->columns == kVerifyWidth && target->stream != nullptr && target->flags == 0u;
}

}  // namespace

extern "C" int axiom_qwen38_dspark_compute_device_history_pack_enqueue(
        const uint32_t *,
        const uint32_t *,
        const uint32_t *,
        const uint32_t *,
        const uint32_t *,
        axiom_qwen38_dspark_device_history *,
        void *) {
    return AXIOM_ERR_INVALID_ARGUMENT;
}

extern "C" int axiom_qwen38_dspark_compute_device_history_pack_enqueue_v2(
        const uint32_t *proposal_tokens_device,
        const uint32_t *accepted_prefix_device,
        const uint32_t *continuation_token_device,
        const uint32_t *async_status_device,
        const uint32_t *next_position_device,
        const uint32_t *committed_tokens_device,
        axiom_qwen38_dspark_device_history *output_device,
        const uint64_t output_device_bytes,
        void *stream) {
    if (output_device_bytes != sizeof(axiom_qwen38_dspark_device_history)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!proposal_tokens_device || !accepted_prefix_device ||
        !continuation_token_device || !async_status_device ||
        !next_position_device || !committed_tokens_device || !output_device || !stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    pack_device_history_kernel<<<1u, 32u, 0, static_cast<cudaStream_t>(stream)>>>(
            proposal_tokens_device, accepted_prefix_device,
            continuation_token_device, async_status_device,
            next_position_device, committed_tokens_device, output_device);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_dspark_compute_device_state_get(
        const axiom_qwen38_dspark_compute *,
        axiom_qwen38_dspark_compute_device_state *) {
    return AXIOM_ERR_INVALID_ARGUMENT;
}

extern "C" int axiom_qwen38_dspark_compute_device_state_get_v2(
        const axiom_qwen38_dspark_compute *compute,
        axiom_qwen38_dspark_compute_device_state *out,
        const uint64_t out_bytes) {
    if (out_bytes != sizeof(axiom_qwen38_dspark_compute_device_state)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!compute || !out || out->abi_version != AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION ||
        !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->graph_ready = compute->proposal_graph_ready ? 1u : 0u;
    out->device_session_active = compute->device_session_active ? 1u : 0u;
    out->proposal_tokens = kBlock;
    out->verify_width = kVerifyWidth;
    out->resident_device_bytes = compute->device_control_bytes;
    out->anchor_token_device = compute->device_anchor_token;
    out->anchor_position_device = compute->device_anchor_position;
    out->proposal_tokens_device = compute->device_proposal_tokens;
    out->verify_tokens_device = compute->device_verify_tokens;
    out->accepted_prefix_device = compute->device_accepted_prefix;
    out->target_commit_prefix_device = compute->device_target_commit_prefix;
    out->continuation_token_device = compute->device_continuation_token;
    out->continuation_logit_device = compute->device_continuation_logit;
    out->async_status_device = compute->device_async_status;
    out->terminal_state_device = compute->device_terminal_state;
    out->history_device = compute->device_history;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_device_graph_prepare(
        axiom_qwen38_dspark_compute *compute) {
    if (!compute || !device_controls_ready(compute) || compute->transaction_open ||
        compute->device_session_active || compute->device_graph_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (compute->proposal_graph_ready) return AXIOM_OK;

    cudaError_t status = cudaStreamCreateWithFlags(
            &compute->proposal_graph_capture_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return cuda_status(status);
    const cudaStream_t stream = compute->proposal_graph_capture_stream;

    status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
        (void) cudaStreamDestroy(compute->proposal_graph_capture_stream);
        compute->proposal_graph_capture_stream = nullptr;
        return cuda_status(status);
    }
    const int rc = enqueue_proposal(compute, 0u, 0u, compute->device_anchor_token,
                                    compute->device_anchor_position, kBlock,
                                    compute->device_proposal_tokens, nullptr, nullptr, stream);
    cudaGraph_t graph = nullptr;
    const cudaError_t end_status = cudaStreamEndCapture(stream, &graph);
    if (rc != AXIOM_OK || end_status != cudaSuccess || !graph) {
        if (graph) (void) cudaGraphDestroy(graph);
        (void) cudaStreamDestroy(compute->proposal_graph_capture_stream);
        compute->proposal_graph_capture_stream = nullptr;
        return rc != AXIOM_OK ? rc : cuda_status(end_status);
    }
    cudaGraphExec_t graph_exec = nullptr;
    status = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0u);
    (void) cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
        (void) cudaStreamDestroy(compute->proposal_graph_capture_stream);
        compute->proposal_graph_capture_stream = nullptr;
        return cuda_status(status);
    }
    compute->proposal_graph_exec = graph_exec;
    compute->proposal_graph_ready = true;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_device_graph_capture_begin(
        axiom_qwen38_dspark_compute *compute) {
    if (!compute || !device_controls_ready(compute) || compute->transaction_open ||
        compute->device_session_active || compute->device_graph_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    compute->device_graph_capture_active = true;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_device_graph_capture_end(
        axiom_qwen38_dspark_compute *compute) {
    if (!compute || !compute->device_graph_capture_active) return AXIOM_ERR_INVALID_ARGUMENT;
    compute->device_graph_capture_active = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_compute_device_session_begin(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_session_request *request) {
    if (!compute || !request ||
        request->abi_version != AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION ||
        !request->anchor_token_device || !request->anchor_position_device || !request->stream ||
        request->flags != 0u || !device_controls_ready(compute) || compute->transaction_open ||
        compute->device_session_active || compute->device_graph_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    cudaError_t status = cudaSuccess;
    if (compute->device_anchor_token != request->anchor_token_device) {
        status = cudaMemcpyAsync(compute->device_anchor_token, request->anchor_token_device,
                                 sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess && compute->device_anchor_position != request->anchor_position_device) {
        status = cudaMemcpyAsync(compute->device_anchor_position, request->anchor_position_device,
                                 sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    }
    int rc = status == cudaSuccess ? clear_device_step_state(compute, stream) : cuda_status(status);
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemsetAsync(
                compute->device_terminal_state, 0, sizeof(uint32_t), stream));
    }
    if (rc == AXIOM_OK) compute->device_session_active = true;
    return rc;
}

extern "C" int axiom_qwen38_dspark_compute_device_session_abort(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    if (!compute || !stream || !compute->device_session_active) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = clear_device_step_state(compute, static_cast<cudaStream_t>(stream));
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemsetAsync(
                compute->device_terminal_state, 0, sizeof(uint32_t),
                static_cast<cudaStream_t>(stream)));
    }
    if (rc == AXIOM_OK) compute->device_session_active = false;
    return rc;
}

extern "C" int axiom_qwen38_dspark_compute_device_graph_replay(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_session_request *request) {
    if (!compute || !request ||
        request->abi_version != AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION ||
        !request->anchor_token_device || !request->anchor_position_device || !request->stream ||
        request->flags != 0u || !compute->device_session_active || !compute->proposal_graph_ready ||
        !compute->proposal_graph_exec || !device_controls_ready(compute)) {
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    cudaError_t status = cudaSuccess;
    if (compute->device_anchor_token != request->anchor_token_device) {
        status = cudaMemcpyAsync(compute->device_anchor_token, request->anchor_token_device,
                                 sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    }
    if (status == cudaSuccess && compute->device_anchor_position != request->anchor_position_device) {
        status = cudaMemcpyAsync(compute->device_anchor_position, request->anchor_position_device,
                                 sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
    }
    int rc = status == cudaSuccess ? clear_device_step_state(compute, stream) : cuda_status(status);
    if (rc != AXIOM_OK) return rc;
    status = cudaGraphLaunch(compute->proposal_graph_exec, stream);
    return cuda_status(status);
}

extern "C" int axiom_qwen38_dspark_compute_device_commit_limit_set(
        axiom_qwen38_dspark_compute *compute,
        const uint32_t max_commit_tokens,
        void *stream) {
    if (!compute || !stream || !device_controls_ready(compute) ||
        max_commit_tokens == 0u || max_commit_tokens > kVerifyWidth) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t status = cudaMemcpyAsync(
            compute->device_commit_limit, &max_commit_tokens, sizeof(max_commit_tokens),
            cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
    return cuda_status(status);
}

extern "C" int axiom_qwen38_dspark_compute_device_cycle_begin_enqueue(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    if (!compute || !stream || !device_operation_active(compute) || !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    return clear_device_step_state(compute, static_cast<cudaStream_t>(stream));
}

extern "C" int axiom_qwen38_dspark_compute_device_propose_enqueue(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    if (!compute || !stream || !device_operation_active(compute) || !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    return enqueue_proposal(compute, 0u, 0u, compute->device_anchor_token,
                            compute->device_anchor_position, kBlock,
                            compute->device_proposal_tokens, nullptr, nullptr,
                            static_cast<cudaStream_t>(stream));
}

extern "C" int axiom_qwen38_dspark_compute_device_build_verify_input(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    if (!compute || !stream || !device_operation_active(compute) || !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    build_verify_input_kernel<<<1u, kVerifyWidth, 0, static_cast<cudaStream_t>(stream)>>>(
            compute->device_anchor_token, compute->device_proposal_tokens, compute->device_verify_tokens);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_dspark_compute_device_accept_greedy(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_target_view *target) {
    if (!compute || !valid_device_target_view(target) || !device_operation_active(compute) ||
        !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    accept_greedy_kernel<<<1u, 1u, 0, static_cast<cudaStream_t>(target->stream)>>>(
            compute->device_anchor_token, compute->device_proposal_tokens,
            target->target_token_ids_device, target->target_logits_device,
            compute->device_commit_limit, compute->device_terminal_state,
            compute->stop_token_count, compute->stop_token_ids[0], compute->stop_token_ids[1],
            compute->device_accepted_prefix, compute->device_target_commit_prefix,
            compute->device_continuation_token,
            compute->device_continuation_logit, compute->device_async_status);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_dspark_compute_device_inject_target(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_target_view *target) {
    if (!compute || !valid_device_target_view(target) || !device_operation_active(compute) ||
        !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t stream = static_cast<cudaStream_t>(target->stream);
    constexpr uint32_t columns = kVerifyWidth;
    const uint64_t tap_count = static_cast<uint64_t>(kFusionInput) * columns;
    pack_target_taps_kernel<<<static_cast<uint32_t>((tap_count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
            target->target_taps_device, compute->fusion_input, columns);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = gemm_bf16_weight_bf16_f32(compute->cublas, compute->fc, kHidden, kFusionInput,
                                        compute->fusion_input, compute->fused, columns, 0.0f,
                                        compute->activation_stage, compute->activation_stage_capacity, stream);
    if (rc == AXIOM_OK) rc = launch_round_bf16(
            compute->fused, static_cast<uint64_t>(kHidden) * columns, stream);
    if (rc == AXIOM_OK) rc = launch_rmsnorm(
            compute->hidden_norm, compute->fused, compute->fused, kHidden, columns, stream);
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        const dspark_layer_weights &weights = compute->layers[layer];
        rc = gemm_bf16_weight_bf16_f32(compute->cublas, weights.k_proj, kKvDim, kHidden,
                                        compute->fused, compute->k, columns, 0.0f,
                                        compute->activation_stage, compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = gemm_bf16_weight_bf16_f32(
                compute->cublas, weights.v_proj, kKvDim, kHidden, compute->fused, compute->v,
                columns, 0.0f, compute->activation_stage,
                compute->activation_stage_capacity, stream);
        if (rc == AXIOM_OK) rc = launch_round_bf16(
                compute->k, static_cast<uint64_t>(kKvDim) * columns, stream);
        if (rc == AXIOM_OK) rc = launch_qk_norm_rope(
                weights.k_norm, compute->k, kKvHeads, columns, 0u,
                compute->device_anchor_position, stream);
        if (rc == AXIOM_OK) {
            const uint64_t count = static_cast<uint64_t>(kKvDim) * columns;
            store_kv_bf16_kernel<<<static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
                    compute->k, compute->v, compute->k_cache[layer], compute->v_cache[layer],
                    0u, columns, compute->device_anchor_position, compute->max_context,
                    compute->device_async_status);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
    }
    return rc;
}

extern "C" int axiom_qwen38_dspark_compute_device_advance(
        axiom_qwen38_dspark_compute *compute,
        void *stream) {
    if (!compute || !stream || !device_operation_active(compute) || !device_controls_ready(compute)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(compute->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    advance_device_position_kernel<<<1u, 1u, 0, static_cast<cudaStream_t>(stream)>>>(
            compute->device_anchor_token, compute->device_anchor_position,
            compute->device_accepted_prefix, compute->device_continuation_token,
            compute->device_terminal_state, compute->device_async_status,
            compute->max_context);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}
