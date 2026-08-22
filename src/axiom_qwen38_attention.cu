/* Native resident full-attention mixer for unsloth/Qwen3.8-27B-NVFP4. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_attention.h"
#include "axiom/qwen38_flashinfer.h"
#include "axiom/qwen38_fp8.h"
#include "axiom/qwen38_kv_tier.h"

namespace {

constexpr uint32_t kHidden = AXIOM_QWEN38_ATTENTION_HIDDEN;
constexpr uint32_t kBatch = AXIOM_QWEN38_ATTENTION_BATCH;
constexpr uint32_t kHeads = AXIOM_QWEN38_ATTENTION_HEADS;
constexpr uint32_t kKvHeads = AXIOM_QWEN38_ATTENTION_KV_HEADS;
constexpr uint32_t kHeadDim = AXIOM_QWEN38_ATTENTION_HEAD_DIM;
constexpr uint32_t kRopeDim = AXIOM_QWEN38_ATTENTION_ROPE_DIM;
constexpr uint32_t kQDim = kHeads * kHeadDim;
constexpr uint32_t kKvDim = kKvHeads * kHeadDim;
constexpr uint32_t kQGateDim = kHeads * kHeadDim * 2u;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kGqaHeads = kHeads / kKvHeads;
constexpr uint32_t kAttentionSplitK = 8u;
constexpr uint32_t kAttentionTileTokens = 16u;
constexpr uint32_t kAttentionThreads = kGqaHeads * 32u;
/* Keep the operator-configured hottest logical pages resident per target
 * layer. The current page is one of these slots; older pages continue to use
 * the durable NVMe tier. The M8/FlashInfer path deliberately remains fixed at
 * eight pages (2,048 tokens), while the scalar paged provider may use a larger
 * bounded HBM page pool to reduce cold reads. */
constexpr uint32_t kDefaultStreamHotPages = 256u;
constexpr uint32_t kTemporalHotPages = 8u;
constexpr uint32_t kTemporalHotTokens =
        kTemporalHotPages * AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
constexpr uint32_t kFlashInferWorkspaceChunks = kTemporalHotPages;
constexpr uint64_t kFlashInferWorkspaceBytes =
        static_cast<uint64_t>(kFlashInferWorkspaceChunks) * kBatch * kHeads *
                (kHeadDim * sizeof(uint16_t) + sizeof(float));
constexpr uint32_t kMaxStreamHotPages = 256u;
constexpr uint32_t kKvParityMaxContext = 64u;
constexpr float kEps = 1.0e-6f;
constexpr float kRopeTheta = 1.0e7f;
/* Qwen3.8 1M extension profile from the SGLang/Qwen3.5 serving recipe.
 * The checkpoint is native to 262,144 tokens; factor=4 extends it to roughly
 * 1M. Keep the target-side blend identical to DSpark so Q/K use one geometry. */
constexpr float kYarnFactor = 4.0f;
constexpr float kYarnOriginalContext = 262144.0f;
constexpr float kYarnBetaFast = 32.0f;
constexpr float kYarnBetaSlow = 1.0f;
constexpr float kYarnMscale = 1.138629436111989f; /* 1 + 0.1 * ln(4). */
/* RadixArk Qwen3.8 D256 target has no calibrated K/V scale tensors. */
constexpr float kKvFp8Scale = 1.0f;
constexpr float kKvFp8Descale = 1.0f;
constexpr uint32_t kKvParityMaxAbsBits = 0u;
constexpr uint32_t kKvParitySampleCount = 1u;
constexpr uint32_t kKvParityNonfiniteCount = 2u;
constexpr uint32_t kKvParityAttentionMaxAbsBits = 3u;
constexpr uint32_t kKvParityMetricCount = 4u;

static_assert(kQDim == 6144u && kKvDim == 1024u && kQGateDim == 12288u,
              "Qwen3.8 attention geometry changed");
static_assert(kGqaHeads == 6u && kAttentionThreads == 192u,
              "Qwen3.8 GQA geometry changed");

__device__ __forceinline__ float qwen38_yarn_inv_frequency(const uint32_t pair) {
    const float dim = static_cast<float>(kRopeDim);
    const float exponent = (2.0f * static_cast<float>(pair)) / dim;
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
    return inv * (1.0f - ramp) + (inv / kYarnFactor) * ramp;
}

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

uint32_t stream_hot_pages_from_env() {
    const char *text = std::getenv("AXIOM_QWEN38_KV_HOT_PAGES");
    if (!text || !text[0]) return kDefaultStreamHotPages;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0u) return kDefaultStreamHotPages;
    return static_cast<uint32_t>(std::min<unsigned long>(parsed, kMaxStreamHotPages));
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

int buffer_pointer(const axiom_device_buffer *buffer, void **out) {
    return axiom_device_buffer_cuda_pointer(buffer, out);
}

bool env_disabled(const char *name) {
    const char *value = std::getenv(name);
    return value && (value[0] == '0' || value[0] == 'f' || value[0] == 'F');
}

bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && (value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
                     value[0] == 'y' || value[0] == 'Y');
}

int tensor_element_count(
        axiom_model *model,
        const char *name,
        axiom_tensor_dtype expected_dtype,
        uint32_t expected_rank,
        uint64_t expected_count) {
    if (!model || !name) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.dtype != expected_dtype || info.rank != expected_rank) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t count = 1u;
    for (uint32_t dimension = 0u; dimension < expected_rank; ++dimension) {
        if (info.shape[dimension] == 0u || !checked_mul(count, info.shape[dimension], &count)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    uint64_t bytes = 0u;
    if (count != expected_count || !checked_mul(count, sizeof(uint16_t), &bytes) ||
        info.byte_count != bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

int upload_bf16_as_f32(
        axiom_model *model,
        axiom_runtime *runtime,
        const char *name,
        uint32_t rank,
        uint64_t count,
        bool add_one,
        axiom_device_buffer **out) {
    if (out) *out = nullptr;
    int rc = tensor_element_count(model, name, AXIOM_TENSOR_DTYPE_BF16, rank, count);
    if (rc != AXIOM_OK) return rc;
    std::vector<uint16_t> encoded;
    std::vector<float> decoded;
    try {
        encoded.resize(static_cast<size_t>(count));
        decoded.resize(static_cast<size_t>(count));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    uint64_t got = 0u;
    const uint64_t bytes = count * sizeof(uint16_t);
    rc = axiom_model_tensor_read(model, name, encoded.data(), bytes, &got);
    if (rc != AXIOM_OK || got != bytes) return rc == AXIOM_OK ? AXIOM_ERR_IO : rc;
    for (uint64_t i = 0u; i < count; ++i) {
        union { uint32_t u; float f; } value{};
        value.u = static_cast<uint32_t>(encoded[static_cast<size_t>(i)]) << 16u;
        decoded[static_cast<size_t>(i)] = add_one ? 1.0f + value.f : value.f;
    }
    axiom_device_buffer *buffer = nullptr;
    rc = axiom_device_buffer_create(runtime, &buffer, count * sizeof(float));
    if (rc == AXIOM_OK) rc = axiom_device_buffer_upload(buffer, 0u, decoded.data(), count * sizeof(float));
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(buffer);
        return rc;
    }
    *out = buffer;
    return AXIOM_OK;
}

/* The SGLang target keeps decoder activations in BF16 at every public
 * attention boundary.  Scratch remains F32 in Axiom, while this helper
 * materializes the same values before another projection, KV write, or output
 * projection consumes them. */
__device__ __forceinline__ float qwen38_attention_round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ uint8_t qwen38_kv_encode_e4m3fn_scale1(float value) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(
            value / kKvFp8Scale, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float qwen38_kv_decode_e4m3fn_scale1(uint8_t code) {
    const uint32_t magnitude = static_cast<uint32_t>(code) & 0x7fu;
    if (magnitude == 0x7fu) return nanf("");
    const uint32_t exponent = (static_cast<uint32_t>(code) >> 3u) & 0x0fu;
    const uint32_t mantissa = static_cast<uint32_t>(code) & 0x07u;
    const uint32_t sign = (static_cast<uint32_t>(code) & 0x80u) << 24u;
    if (exponent == 0u) {
        const float value = ldexpf(static_cast<float>(mantissa), -9);
        return sign != 0u ? -value * kKvFp8Descale : value * kKvFp8Descale;
    }
    const uint32_t fp32 = sign | ((exponent + 120u) << 23u) | (mantissa << 20u);
    return __uint_as_float(fp32) * kKvFp8Descale;
}

__device__ __forceinline__ void qwen38_kv_record_fp8_parity(
        uint32_t *metrics,
        float bf16_materialized,
        float fp8_roundtrip) {
    if (!metrics) return;
    if (!isfinite(bf16_materialized) || !isfinite(fp8_roundtrip)) {
        atomicAdd(metrics + kKvParityNonfiniteCount, 1u);
        return;
    }
    const float abs_error = fabsf(fp8_roundtrip - bf16_materialized);
    atomicMax(metrics + kKvParityMaxAbsBits, __float_as_uint(abs_error));
    atomicAdd(metrics + kKvParitySampleCount, 1u);
}

__device__ __forceinline__ void qwen38_kv_store_e4m3fn_scale1(
        uint8_t *cache,
        uint64_t index,
        float bf16_materialized,
        uint32_t *parity_metrics) {
    const uint8_t code = qwen38_kv_encode_e4m3fn_scale1(bf16_materialized);
    cache[index] = code;
    qwen38_kv_record_fp8_parity(
            parity_metrics, bf16_materialized, qwen38_kv_decode_e4m3fn_scale1(code));
}

__host__ __device__ __forceinline__ uint64_t qwen38_attention_split_stats_index(
        uint32_t row,
        uint32_t kv_head,
        uint32_t split,
        uint32_t gqa_head) {
    return (((static_cast<uint64_t>(row) * kKvHeads + kv_head) * kAttentionSplitK + split) *
            kGqaHeads + gqa_head);
}

__host__ __device__ __forceinline__ uint64_t qwen38_attention_split_value_index(
        uint32_t row,
        uint32_t kv_head,
        uint32_t split,
        uint32_t gqa_head,
        uint32_t dim) {
    return qwen38_attention_split_stats_index(row, kv_head, split, gqa_head) * kHeadDim + dim;
}

__global__ void qwen38_kv_record_attention_parity_kernel(
        const float *__restrict__ fp8_attention,
        const float *__restrict__ bf16_cache_attention,
        uint64_t count,
        uint32_t *__restrict__ metrics) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!metrics || index >= count) return;
    const float fp8_value = fp8_attention[index];
    const float reference_value = bf16_cache_attention[index];
    if (!isfinite(fp8_value) || !isfinite(reference_value)) {
        atomicAdd(metrics + kKvParityNonfiniteCount, 1u);
        return;
    }
    atomicMax(metrics + kKvParityAttentionMaxAbsBits,
              __float_as_uint(fabsf(fp8_value - reference_value)));
}

__global__ void qwen38_attention_rmsnorm8_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch) return;
    const float *x = input + static_cast<uint64_t>(column) * kHidden;
    float *y = out + static_cast<uint64_t>(column) * kHidden;
    __shared__ float sums[kThreads];
    float sum = 0.0f;
    for (uint32_t i = tid; i < kHidden; i += blockDim.x) sum = fmaf(x[i], x[i], sum);
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(sums[0] / static_cast<float>(kHidden) + kEps);
    for (uint32_t i = tid; i < kHidden; i += blockDim.x) {
        y[i] = qwen38_attention_round_bf16(x[i] * inv * weight[i]);
    }
}

__global__ void qwen38_attention_round_bf16_inplace_kernel(
        float *__restrict__ values,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = qwen38_attention_round_bf16(values[index]);
    }
}

/* FlashInfer's generated D256 specialization is BF16 Q/O while Axiom keeps
 * public intermediates as F32 values already rounded at model boundaries.
 * These two persistent staging conversions are graph-capture-safe and add no
 * allocation or host round trip to temporal verify. */
__global__ void qwen38_attention_f32_to_bf16_kernel(
        const float *__restrict__ input,
        uint16_t *__restrict__ out,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) out[index] = __bfloat16_as_ushort(__float2bfloat16_rn(input[index]));
}

__global__ void qwen38_attention_bf16_to_f32_kernel(
        const uint16_t *__restrict__ input,
        float *__restrict__ out,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const __nv_bfloat16 value = reinterpret_cast<const __nv_bfloat16 *>(input)[index];
        out[index] = __bfloat162float(value);
    }
}

__global__ void qwen38_attention_temporal8_kv_length_kernel(
        uint32_t *__restrict__ kv_length,
        const uint32_t *__restrict__ base_position) {
    if (threadIdx.x == 0u && blockIdx.x == 0u) {
        kv_length[0] = base_position[0] + kBatch;
    }
}

int flashinfer_convert_q_to_bf16(
        const float *input, uint16_t *out, cudaStream_t stream) {
    if (!input || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t count = static_cast<uint64_t>(kQDim) * kBatch;
    qwen38_attention_f32_to_bf16_kernel<<<
            static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
            input, out, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int flashinfer_convert_out_to_f32(
        const uint16_t *input, float *out, cudaStream_t stream) {
    if (!input || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t count = static_cast<uint64_t>(kQDim) * kBatch;
    qwen38_attention_bf16_to_f32_kernel<<<
            static_cast<uint32_t>((count + kThreads - 1u) / kThreads), kThreads, 0, stream>>>(
            input, out, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int round_bf16_inplace(float *values, uint64_t count, cudaStream_t stream = nullptr) {
    if (!values || count == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t grid = (count + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    qwen38_attention_round_bf16_inplace_kernel<<<static_cast<uint32_t>(grid), kThreads, 0, stream>>>(
            values, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void qwen38_split_q_gate_kernel(
        const float *__restrict__ q_gate,
        float *__restrict__ q,
        float *__restrict__ gate) {
    const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (element >= kQDim || column >= kBatch) return;
    const uint32_t head = element / kHeadDim;
    const uint32_t dim = element - head * kHeadDim;
    const uint64_t source = static_cast<uint64_t>(column) * kQGateDim +
            static_cast<uint64_t>(head) * (2u * kHeadDim) + dim;
    const uint64_t destination = static_cast<uint64_t>(column) * kQDim + element;
    q[destination] = q_gate[source];
    gate[destination] = q_gate[source + kHeadDim];
}

/* q_proj returns interleaved [q, gate] per head.  The target consumes the
 * rounded q and gate values, while k_proj must be rounded before q/k norm.
 * Fuse those independent conversions and omit the otherwise dead rounded
 * q_gate scratch. */
__global__ void qwen38_split_q_gate_round_k8_kernel(
        const float *__restrict__ q_gate,
        float *__restrict__ q,
        float *__restrict__ gate,
        float *__restrict__ k) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t q_count = static_cast<uint64_t>(kQDim) * kBatch;
    const uint64_t k_count = static_cast<uint64_t>(kKvDim) * kBatch;
    if (index < q_count) {
        const uint32_t column = static_cast<uint32_t>(index / kQDim);
        const uint32_t element = static_cast<uint32_t>(index -
                static_cast<uint64_t>(column) * kQDim);
        const uint32_t head = element / kHeadDim;
        const uint32_t dim = element - head * kHeadDim;
        const uint64_t source = static_cast<uint64_t>(column) * kQGateDim +
                static_cast<uint64_t>(head) * (2u * kHeadDim) + dim;
        q[index] = qwen38_attention_round_bf16(q_gate[source]);
        gate[index] = qwen38_attention_round_bf16(q_gate[source + kHeadDim]);
    } else if (index < q_count + k_count) {
        const uint64_t k_index = index - q_count;
        k[k_index] = qwen38_attention_round_bf16(k[k_index]);
    }
}

int split_q_gate_round_k8(
        const float *q_gate,
        float *q,
        float *gate,
        float *k,
        cudaStream_t stream) {
    if (!q_gate || !q || !gate || !k) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t count = (static_cast<uint64_t>(kQDim) + kKvDim) * kBatch;
    const uint64_t grid = (count + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    qwen38_split_q_gate_round_k8_kernel<<<static_cast<uint32_t>(grid), kThreads, 0, stream>>>(
            q_gate, q, gate, k);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void qwen38_sigmoid_gate_kernel(
        const float *__restrict__ attention,
        const float *__restrict__ gate,
        float *__restrict__ out) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kQDim) * kBatch;
    if (index >= count) return;
    out[index] = qwen38_attention_round_bf16(
            attention[index] * (1.0f / (1.0f + expf(-gate[index]))));
}

__global__ void qwen38_qk_norm_partial_rope_cache8_kernel(
        float *__restrict__ q,
        float *__restrict__ k,
        const float *__restrict__ q_weight,
        const float *__restrict__ k_weight,
        const float *__restrict__ v,
        uint8_t *__restrict__ k_cache,
        uint8_t *__restrict__ v_cache,
        float *__restrict__ k_cache_reference,
        float *__restrict__ v_cache_reference,
        uint32_t cache_columns,
        uint32_t max_context,
        uint32_t position,
        bool yarn_enabled,
        uint32_t *__restrict__ parity_metrics) {
    const uint32_t combined_head = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch || combined_head >= kHeads + kKvHeads) return;
    const bool is_k = combined_head >= kHeads;
    const uint32_t head = is_k ? combined_head - kHeads : combined_head;
    const uint32_t heads = is_k ? kKvHeads : kHeads;
    float *row = (is_k ? k : q) +
            (static_cast<uint64_t>(column) * heads + head) * kHeadDim;
    const float *weight = is_k ? k_weight : q_weight;
    __shared__ double sums[kThreads];
    __shared__ float normalized[kHeadDim];
    double sum = 0.0;
    for (uint32_t dim = tid; dim < kHeadDim; dim += blockDim.x) {
        const double value = static_cast<double>(row[dim]);
        sum += value * value;
    }
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = static_cast<float>(
            1.0 / sqrt(sums[0] / static_cast<double>(kHeadDim) + static_cast<double>(kEps)));
    if (tid < kHeadDim) normalized[tid] = row[tid] * inverse * weight[tid];
    __syncthreads();
    constexpr uint32_t kRopeHalf = kRopeDim / 2u;
    if (tid < kRopeHalf) {
        const float frequency = yarn_enabled
                ? qwen38_yarn_inv_frequency(tid)
                : powf(kRopeTheta, -2.0f * static_cast<float>(tid) /
                        static_cast<float>(kRopeDim));
        const float angle = static_cast<float>(position) * frequency;
        const float scale = yarn_enabled ? kYarnMscale : 1.0f;
        const float cosine = cosf(angle) * scale;
        const float sine = sinf(angle) * scale;
        const float first = normalized[tid];
        const float second = normalized[tid + kRopeHalf];
        row[tid] = qwen38_attention_round_bf16(first * cosine - second * sine);
        row[tid + kRopeHalf] = qwen38_attention_round_bf16(first * sine + second * cosine);
    } else if (tid >= kRopeDim && tid < kHeadDim) {
        row[tid] = qwen38_attention_round_bf16(normalized[tid]);
    }
    __syncthreads();
    if (is_k && tid < kHeadDim) {
        const uint64_t cache_base =
                (static_cast<uint64_t>(cache_columns == 1u ? 0u : column) * max_context +
                 position) * kKvDim;
        const uint64_t cache_index = cache_base + static_cast<uint64_t>(head) * kHeadDim + tid;
        const float key = row[tid];
        const float value = qwen38_attention_round_bf16(
                v[(static_cast<uint64_t>(column) * kKvHeads + head) * kHeadDim + tid]);
        qwen38_kv_store_e4m3fn_scale1(k_cache, cache_index, key, parity_metrics);
        qwen38_kv_store_e4m3fn_scale1(v_cache, cache_index, value, parity_metrics);
        if (k_cache_reference) k_cache_reference[cache_index] = key;
        if (v_cache_reference) v_cache_reference[cache_index] = value;
    }
}

/* Six Q heads share each Qwen3.8 KV head.  The previous kernel launched one
 * CTA per Q head and therefore reloaded the same K/V plane six times while
 * synchronizing eight times for every cache token.  This tiled split-K path
 * loads each FP8 K/V value once per CTA, fans it out to the six warps, and
 * stores an online-softmax partial.  The merge keeps chronological split
 * order and BF16 materializes the same public attention boundary. */
template <bool kTemporal, uint32_t kSplits>
__global__ void qwen38_attention_core_gqa_splitk8_kernel(
        const float *__restrict__ q,
        const uint8_t *__restrict__ k_cache,
        const uint8_t *__restrict__ v_cache,
        float *__restrict__ split_values,
        float *__restrict__ split_maxima,
        float *__restrict__ split_denominators,
        uint32_t scalar_cache_tokens,
        uint32_t base_position_host,
        const uint32_t *__restrict__ base_position_device,
        uint32_t max_context) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t split = blockIdx.z;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || row >= kBatch || split >= kSplits) return;

    uint32_t cache_tokens = scalar_cache_tokens;
    uint64_t cache_column = static_cast<uint64_t>(row) * max_context * kKvDim;
    if constexpr (kTemporal) {
        const uint32_t base_position = base_position_device
                ? base_position_device[0] : base_position_host;
        if (base_position > max_context || kBatch > max_context - base_position) return;
        cache_tokens = base_position + row + 1u;
        cache_column = 0u;
    }
    if (cache_tokens > max_context) return;

    __shared__ float tile_k[kAttentionTileTokens * kHeadDim];
    __shared__ float tile_v[kAttentionTileTokens * kHeadDim];
    constexpr uint32_t kWarpMask = 0xffffffffu;
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
    float maximum = -3.4028234663852886e+38F;
    float denominator = 0.0f;
    const uint32_t token_begin = static_cast<uint32_t>(
            (static_cast<uint64_t>(cache_tokens) * split) / kSplits);
    const uint32_t token_end = static_cast<uint32_t>(
            (static_cast<uint64_t>(cache_tokens) * (split + 1u)) / kSplits);
    const float attention_scale = rsqrtf(static_cast<float>(kHeadDim));

    for (uint32_t tile_begin = token_begin; tile_begin < token_end;
         tile_begin += kAttentionTileTokens) {
        const uint32_t tile_count = min(kAttentionTileTokens, token_end - tile_begin);
        const uint32_t tile_values = tile_count * kHeadDim;
        for (uint32_t index = threadIdx.x; index < tile_values; index += blockDim.x) {
            const uint32_t tile_token = index / kHeadDim;
            const uint32_t dim = index - tile_token * kHeadDim;
            const uint64_t cache_index = cache_column +
                    (static_cast<uint64_t>(tile_begin + tile_token) * kKvHeads + kv_head) *
                    kHeadDim + dim;
            tile_k[index] = qwen38_kv_decode_e4m3fn_scale1(k_cache[cache_index]);
            tile_v[index] = qwen38_kv_decode_e4m3fn_scale1(v_cache[cache_index]);
        }
        __syncthreads();
        if (active) {
            for (uint32_t tile_token = 0u; tile_token < tile_count; ++tile_token) {
                const uint32_t tile_offset = tile_token * kHeadDim;
                float dot = 0.0f;
#pragma unroll
                for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
                    dot = fmaf(query_values[part], tile_k[tile_offset + lane + part * 32u], dot);
                }
                for (uint32_t offset = 16u; offset != 0u; offset >>= 1u) {
                    dot += __shfl_down_sync(kWarpMask, dot, offset);
                }
                const float score = __shfl_sync(kWarpMask, dot, 0) * attention_scale;
                const float next_maximum = fmaxf(maximum, score);
                const float correction = __expf(maximum - next_maximum);
                const float weight = __expf(score - next_maximum);
                denominator = denominator * correction + weight;
#pragma unroll
                for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
                    const uint32_t dim = lane + part * 32u;
                    accumulators[part] = fmaf(
                            weight, tile_v[tile_offset + dim], accumulators[part] * correction);
                }
                maximum = next_maximum;
            }
        }
        __syncthreads();
    }
    if (!active) return;
    const uint64_t stats_index = qwen38_attention_split_stats_index(row, kv_head, split, warp);
    if (lane == 0u) {
        split_maxima[stats_index] = maximum;
        split_denominators[stats_index] = denominator;
    }
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        split_values[qwen38_attention_split_value_index(
                row, kv_head, split, warp, lane + part * 32u)] = accumulators[part];
    }
}

__global__ void qwen38_attention_merge_gqa_splitk8_kernel(
        const float *__restrict__ split_values,
        const float *__restrict__ split_maxima,
        const float *__restrict__ split_denominators,
        float *__restrict__ out) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || row >= kBatch || warp >= kGqaHeads) return;
    constexpr uint32_t kWarpMask = 0xffffffffu;
    (void)kWarpMask;
    float accumulators[kHeadDim / 32u]{};
    float maximum = -3.4028234663852886e+38F;
    float denominator = 0.0f;
    for (uint32_t split = 0u; split < kAttentionSplitK; ++split) {
        const uint64_t stats_index = qwen38_attention_split_stats_index(row, kv_head, split, warp);
        const float partial_denominator = split_denominators[stats_index];
        if (partial_denominator == 0.0f) continue;
        const float partial_maximum = split_maxima[stats_index];
        const float next_maximum = fmaxf(maximum, partial_maximum);
        const float correction = __expf(maximum - next_maximum);
        const float partial_correction = __expf(partial_maximum - next_maximum);
        denominator = denominator * correction + partial_denominator * partial_correction;
#pragma unroll
        for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
            const uint32_t dim = lane + part * 32u;
            const float partial = split_values[qwen38_attention_split_value_index(
                    row, kv_head, split, warp, dim)];
            accumulators[part] = fmaf(partial_correction, partial, accumulators[part] * correction);
        }
        maximum = next_maximum;
    }
    const uint32_t query_head = kv_head * kGqaHeads + warp;
    const uint64_t out_base =
            (static_cast<uint64_t>(row) * kHeads + query_head) * kHeadDim;
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        const uint32_t dim = lane + part * 32u;
        out[out_base + dim] = qwen38_attention_round_bf16(
                denominator > 0.0f ? accumulators[part] / denominator : 0.0f);
    }
}

/* Scalar and host-M8 validation must retain one chronological reduction
 * order so the temporal transaction can prove exact parity against eight
 * sequential forward_token calls. Device graph decode uses the tiled path
 * above (or FlashInfer); this compact reference kernel is deliberately kept
 * off that hot path. */
template <bool kTemporal>
__global__ void qwen38_attention_core_fp8_reference_kernel(
        const float *__restrict__ q,
        const uint8_t *__restrict__ k_cache,
        const uint8_t *__restrict__ v_cache,
        float *__restrict__ out,
        uint32_t scalar_cache_tokens,
        uint32_t cache_columns,
        uint32_t base_position_host,
        const uint32_t *__restrict__ base_position_device,
        uint32_t max_context) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t dim = threadIdx.x;
    if (query_head >= kHeads || row >= kBatch || dim >= kHeadDim) return;
    uint32_t cache_tokens = scalar_cache_tokens;
    uint64_t cache_column = static_cast<uint64_t>(cache_columns == 1u ? 0u : row) *
            max_context * kKvDim;
    if constexpr (kTemporal) {
        const uint32_t base_position = base_position_device
                ? base_position_device[0] : base_position_host;
        if (base_position > max_context || kBatch > max_context - base_position) return;
        cache_tokens = base_position + row + 1u;
        cache_column = 0u;
    }
    if (cache_tokens > max_context) return;
    const uint32_t kv_head = query_head / kGqaHeads;
    const uint64_t q_base =
            (static_cast<uint64_t>(row) * kHeads + query_head) * kHeadDim;
    extern __shared__ float scratch[];
    const float query = q[q_base + dim];
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    float maximum = -3.4028234663852886e+38F;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (uint32_t token = 0u; token < cache_tokens; ++token) {
        const uint64_t cache_base = cache_column +
                (static_cast<uint64_t>(token) * kKvHeads + kv_head) * kHeadDim;
        scratch[dim] = query * qwen38_kv_decode_e4m3fn_scale1(k_cache[cache_base + dim]);
        __syncthreads();
        for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
            if (dim < stride) scratch[dim] += scratch[dim + stride];
            __syncthreads();
        }
        const float score = scratch[0] * scale;
        __syncthreads();
        const float next_maximum = fmaxf(maximum, score);
        const float correction = __expf(maximum - next_maximum);
        const float weight = __expf(score - next_maximum);
        denominator = denominator * correction + weight;
        accumulator = accumulator * correction +
                weight * qwen38_kv_decode_e4m3fn_scale1(v_cache[cache_base + dim]);
        maximum = next_maximum;
    }
    out[q_base + dim] = qwen38_attention_round_bf16(
            denominator > 0.0f ? accumulator / denominator : 0.0f);
}

/* The shadow path exists only for the short-context numerical gate. It keeps
 * the pre-FP8 BF16-materialized F32 cache semantics as the reference. */
template <bool kTemporal>
__global__ void qwen38_attention_core_bf16_cache_reference_kernel(
        const float *__restrict__ q,
        const float *__restrict__ k_cache,
        const float *__restrict__ v_cache,
        float *__restrict__ out,
        uint32_t scalar_cache_tokens,
        uint32_t cache_columns,
        uint32_t base_position_host,
        const uint32_t *__restrict__ base_position_device,
        uint32_t max_context) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t dim = threadIdx.x;
    if (query_head >= kHeads || row >= kBatch || dim >= kHeadDim) return;
    uint32_t cache_tokens = scalar_cache_tokens;
    uint64_t cache_column = static_cast<uint64_t>(cache_columns == 1u ? 0u : row) *
            max_context * kKvDim;
    if constexpr (kTemporal) {
        const uint32_t base_position = base_position_device
                ? base_position_device[0] : base_position_host;
        if (base_position > max_context || kBatch > max_context - base_position) return;
        cache_tokens = base_position + row + 1u;
        cache_column = 0u;
    }
    if (cache_tokens > max_context) return;
    const uint32_t kv_head = query_head / kGqaHeads;
    const uint64_t q_base =
            (static_cast<uint64_t>(row) * kHeads + query_head) * kHeadDim;
    extern __shared__ float scratch[];
    const float query = q[q_base + dim];
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    float maximum = -3.4028234663852886e+38F;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (uint32_t token = 0u; token < cache_tokens; ++token) {
        const uint64_t cache_base = cache_column +
                (static_cast<uint64_t>(token) * kKvHeads + kv_head) * kHeadDim;
        scratch[dim] = query * k_cache[cache_base + dim];
        __syncthreads();
        for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
            if (dim < stride) scratch[dim] += scratch[dim + stride];
            __syncthreads();
        }
        const float score = scratch[0] * scale;
        __syncthreads();
        const float next_maximum = fmaxf(maximum, score);
        const float correction = __expf(maximum - next_maximum);
        const float weight = __expf(score - next_maximum);
        denominator = denominator * correction + weight;
        accumulator = accumulator * correction + weight * v_cache[cache_base + dim];
        maximum = next_maximum;
    }
    out[q_base + dim] = qwen38_attention_round_bf16(
            denominator > 0.0f ? accumulator / denominator : 0.0f);
}

/* Compatibility scalar/reference mode still owns its historical F32 cache,
 * but writes the exact same BF16-materialized K/V values into the packed
 * E4M3FN cache as well. A later temporal transaction can therefore retain
 * causal prompt history without the former F32 cache bandwidth. */
__global__ void qwen38_cache_f32_to_e4m3fn8_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        uint8_t *__restrict__ k_cache,
        uint8_t *__restrict__ v_cache,
        uint32_t cache_columns,
        uint32_t max_context,
        uint32_t position,
        uint32_t *__restrict__ parity_metrics) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kBatch) * kKvDim;
    if (index >= count) return;
    const uint32_t column = static_cast<uint32_t>(index / kKvDim);
    const uint32_t feature = static_cast<uint32_t>(index - static_cast<uint64_t>(column) * kKvDim);
    if (cache_columns == 1u && column != 0u) return;
    const uint64_t cache_index =
            (static_cast<uint64_t>(cache_columns == 1u ? 0u : column) * max_context + position) *
                    kKvDim + feature;
    qwen38_kv_store_e4m3fn_scale1(k_cache, cache_index, k[index], parity_metrics);
    qwen38_kv_store_e4m3fn_scale1(v_cache, cache_index, v[index], parity_metrics);
}

__global__ void qwen38_qk_norm_partial_rope_cache_temporal8_kernel(
        float *__restrict__ q,
        float *__restrict__ k,
        const float *__restrict__ q_weight,
        const float *__restrict__ k_weight,
        const float *__restrict__ v,
        uint8_t *__restrict__ k_cache,
        uint8_t *__restrict__ v_cache,
        float *__restrict__ k_cache_reference,
        float *__restrict__ v_cache_reference,
        uint32_t base_position_host,
        const uint32_t *__restrict__ base_position_device,
        uint32_t max_context,
        bool yarn_enabled,
        uint32_t *__restrict__ parity_metrics) {
    const uint32_t combined_head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    const uint32_t base_position = base_position_device ? base_position_device[0] : base_position_host;
    if (base_position > max_context || kBatch > max_context - base_position) return;
    if (token >= kBatch || combined_head >= kHeads + kKvHeads) return;
    const bool is_k = combined_head >= kHeads;
    const uint32_t head = is_k ? combined_head - kHeads : combined_head;
    const uint32_t heads = is_k ? kKvHeads : kHeads;
    float *row = (is_k ? k : q) +
            (static_cast<uint64_t>(token) * heads + head) * kHeadDim;
    const float *weight = is_k ? k_weight : q_weight;
    __shared__ double sums[kThreads];
    __shared__ float normalized[kHeadDim];
    double sum = 0.0;
    for (uint32_t dim = tid; dim < kHeadDim; dim += blockDim.x) {
        const double value = static_cast<double>(row[dim]);
        sum += value * value;
    }
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = static_cast<float>(
            1.0 / sqrt(sums[0] / static_cast<double>(kHeadDim) + static_cast<double>(kEps)));
    if (tid < kHeadDim) normalized[tid] = row[tid] * inverse * weight[tid];
    __syncthreads();
    constexpr uint32_t kRopeHalf = kRopeDim / 2u;
    const uint32_t position = base_position + token;
    if (tid < kRopeHalf) {
        const float frequency = yarn_enabled
                ? qwen38_yarn_inv_frequency(tid)
                : powf(kRopeTheta, -2.0f * static_cast<float>(tid) /
                        static_cast<float>(kRopeDim));
        const float angle = static_cast<float>(position) * frequency;
        const float scale = yarn_enabled ? kYarnMscale : 1.0f;
        const float cosine = cosf(angle) * scale;
        const float sine = sinf(angle) * scale;
        const float first = normalized[tid];
        const float second = normalized[tid + kRopeHalf];
        row[tid] = qwen38_attention_round_bf16(first * cosine - second * sine);
        row[tid + kRopeHalf] = qwen38_attention_round_bf16(first * sine + second * cosine);
    } else if (tid >= kRopeDim && tid < kHeadDim) {
        row[tid] = qwen38_attention_round_bf16(normalized[tid]);
    }
    __syncthreads();
    if (is_k && tid < kHeadDim) {
        const uint64_t cache_base =
                static_cast<uint64_t>(position) * kKvDim +
                static_cast<uint64_t>(head) * kHeadDim;
        const float key = row[tid];
        const float value = qwen38_attention_round_bf16(
                v[(static_cast<uint64_t>(token) * kKvHeads + head) * kHeadDim + tid]);
        qwen38_kv_store_e4m3fn_scale1(k_cache, cache_base + tid, key, parity_metrics);
        qwen38_kv_store_e4m3fn_scale1(v_cache, cache_base + tid, value, parity_metrics);
        if (k_cache_reference) k_cache_reference[cache_base + tid] = key;
        if (v_cache_reference) v_cache_reference[cache_base + tid] = value;
    }
}

/* Scalar/reference mode uses the runtime's generic RoPE launcher for legacy
 * checkpoints. The native target must use the same YaRN table as the temporal
 * kernels, otherwise scalar validation and M8 would carry different phases. */
__global__ void qwen38_rope_neox_partial_yarn_kernel(
        float *__restrict__ x,
        uint32_t heads,
        uint32_t position,
        bool yarn_enabled) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= heads || tid >= kHeadDim) return;
    float *row = x + static_cast<uint64_t>(head) * kHeadDim;
    constexpr uint32_t kRopeHalf = kRopeDim / 2u;
    if (tid < kRopeHalf) {
        const float frequency = yarn_enabled
                ? qwen38_yarn_inv_frequency(tid)
                : powf(kRopeTheta, -2.0f * static_cast<float>(tid) /
                        static_cast<float>(kRopeDim));
        const float angle = static_cast<float>(position) * frequency;
        const float scale = yarn_enabled ? kYarnMscale : 1.0f;
        const float cosine = cosf(angle) * scale;
        const float sine = sinf(angle) * scale;
        const float first = row[tid];
        const float second = row[tid + kRopeHalf];
        row[tid] = first * cosine - second * sine;
        row[tid + kRopeHalf] = first * sine + second * cosine;
    }
}

int launch_qwen38_rope_neox_partial_yarn(
        float *x,
        uint32_t heads,
        uint32_t position,
        bool yarn_enabled,
        cudaStream_t stream) {
    if (!x || heads == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    qwen38_rope_neox_partial_yarn_kernel<<<heads, kHeadDim, 0, stream>>>(
            x, heads, position, yarn_enabled);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* The streaming provider keeps exactly one logical target page on device for
 * the current layer.  K and V are packed consecutively so the same aligned
 * page can be passed to the NVMe tier without a second staging format. */
__global__ void qwen38_stream_store_kv_page_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        uint8_t *__restrict__ page,
        uint32_t token_offset) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kKvDim) return;
    const uint64_t page_index = static_cast<uint64_t>(token_offset) * kKvDim + index;
    page[page_index] = qwen38_kv_encode_e4m3fn_scale1(k[index]);
    page[AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u + page_index] =
            qwen38_kv_encode_e4m3fn_scale1(qwen38_attention_round_bf16(v[index]));
}

__global__ void qwen38_stream_store_hot_kv_row_kernel(
        const float *__restrict__ k,
        const float *__restrict__ v,
        uint8_t *__restrict__ hot_k_cache,
        uint8_t *__restrict__ hot_v_cache,
        uint32_t position) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kKvDim || position >= kTemporalHotTokens) return;
    const uint64_t cache_index = static_cast<uint64_t>(position) * kKvDim + index;
    hot_k_cache[cache_index] = qwen38_kv_encode_e4m3fn_scale1(k[index]);
    hot_v_cache[cache_index] = qwen38_kv_encode_e4m3fn_scale1(
            qwen38_attention_round_bf16(v[index]));
}

__global__ void qwen38_stream_init_stats_kernel(
        float *__restrict__ maxima,
        float *__restrict__ denominators,
        float *__restrict__ values) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < kHeads) {
        maxima[index] = -3.4028234663852886e+38F;
        denominators[index] = 0.0f;
    }
    const uint64_t value_count = static_cast<uint64_t>(kHeads) * kHeadDim;
    if (index < value_count) values[index] = 0.0f;
}

/* Correctness-first page reducer. It is the resident reference kernel's
 * chronological online-softmax loop with its state carried across pages.
 * This keeps the dense-vs-paged gate meaningful; the split-K provider below
 * remains available behind AXIOM_QWEN38_KV_STREAMING_FAST for later tuning. */
__global__ void qwen38_stream_reference_page_kernel(
        const float *__restrict__ q,
        const uint8_t *__restrict__ k_page,
        const uint8_t *__restrict__ v_page,
        uint32_t page_tokens,
        float *__restrict__ maxima,
        float *__restrict__ denominators,
        float *__restrict__ values) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t dim = threadIdx.x;
    if (query_head >= kHeads || dim >= kHeadDim || page_tokens >
            AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS) return;
    const uint64_t q_base = static_cast<uint64_t>(query_head) * kHeadDim;
    const uint64_t state_base = q_base;
    extern __shared__ float scratch[];
    const float query = q[q_base + dim];
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    float maximum = maxima[query_head];
    float denominator = denominators[query_head];
    float accumulator = values[state_base + dim];
    for (uint32_t token = 0u; token < page_tokens; ++token) {
        const uint64_t cache_base = static_cast<uint64_t>(token) * kKvDim +
                static_cast<uint64_t>(query_head / kGqaHeads) * kHeadDim;
        scratch[dim] = query * qwen38_kv_decode_e4m3fn_scale1(k_page[cache_base + dim]);
        __syncthreads();
        for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
            if (dim < stride) scratch[dim] += scratch[dim + stride];
            __syncthreads();
        }
        const float score = scratch[0] * scale;
        __syncthreads();
        const float next_maximum = fmaxf(maximum, score);
        const float correction = __expf(maximum - next_maximum);
        const float weight = __expf(score - next_maximum);
        denominator = denominator * correction + weight;
        accumulator = accumulator * correction + weight *
                qwen38_kv_decode_e4m3fn_scale1(v_page[cache_base + dim]);
        maximum = next_maximum;
    }
    if (dim == 0u) {
        maxima[query_head] = maximum;
        denominators[query_head] = denominator;
    }
    values[state_base + dim] = accumulator;
}

/* Device-paged fast path for the resident hot window. The page table contains
 * the fixed device addresses of the HBM slots; logical pages map to slots with
 * the same ring rule used by stream_read_page(). The online-softmax loop is
 * intentionally the same chronological loop as the page-at-a-time reference,
 * but all resident logical pages are traversed by one launch per layer. */
__global__ void qwen38_stream_reference_hot_pages_kernel(
        const float *__restrict__ q,
        const uint8_t *const *__restrict__ page_table,
        uint32_t hot_pages,
        uint32_t page_count,
        uint32_t current_page_tokens,
        float *__restrict__ out) {
    const uint32_t query_head = blockIdx.x;
    const uint32_t dim = threadIdx.x;
    if (query_head >= kHeads || dim >= kHeadDim || hot_pages == 0u || page_count == 0u) return;
    const uint64_t q_base = static_cast<uint64_t>(query_head) * kHeadDim;
    extern __shared__ float scratch[];
    const float query = q[q_base + dim];
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    float maximum = -3.4028234663852886e+38F;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    for (uint32_t logical_page = 0u; logical_page < page_count; ++logical_page) {
        const uint8_t *page = page_table[logical_page % hot_pages];
        const uint32_t page_tokens = logical_page + 1u == page_count
                ? current_page_tokens : AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
        for (uint32_t token = 0u; token < page_tokens; ++token) {
            const uint64_t cache_base = static_cast<uint64_t>(token) * kKvDim +
                    static_cast<uint64_t>(query_head / kGqaHeads) * kHeadDim;
            scratch[dim] = query * qwen38_kv_decode_e4m3fn_scale1(
                    page[cache_base + dim]);
            __syncthreads();
            for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
                if (dim < stride) scratch[dim] += scratch[dim + stride];
                __syncthreads();
            }
            const float score = scratch[0] * scale;
            __syncthreads();
            const float next_maximum = fmaxf(maximum, score);
            const float correction = __expf(maximum - next_maximum);
            const float weight = __expf(score - next_maximum);
            denominator = denominator * correction + weight;
            accumulator = accumulator * correction + weight *
                    qwen38_kv_decode_e4m3fn_scale1(
                            page[AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u + cache_base + dim]);
            maximum = next_maximum;
        }
    }
    out[q_base + dim] = qwen38_attention_round_bf16(
            denominator > 0.0f ? accumulator / denominator : 0.0f);
}

/* Merge the split-K result for one resident page into the chronological
 * online-softmax state.  Each GQA warp owns one query head; distinct KV heads
 * write disjoint portions of the state, so no atomics are needed. */
__global__ void qwen38_stream_accumulate_page_kernel(
        const float *__restrict__ split_values,
        const float *__restrict__ split_maxima,
        const float *__restrict__ split_denominators,
        float *__restrict__ maxima,
        float *__restrict__ denominators,
        float *__restrict__ values) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || warp >= kGqaHeads) return;
    const uint32_t query_head = kv_head * kGqaHeads + warp;
    float accumulators[kHeadDim / 32u];
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        accumulators[part] = values[static_cast<uint64_t>(query_head) * kHeadDim +
                lane + part * 32u];
    }
    float maximum = maxima[query_head];
    float denominator = denominators[query_head];
    for (uint32_t split = 0u; split < kAttentionSplitK; ++split) {
        const uint64_t stats_index = qwen38_attention_split_stats_index(
                0u, kv_head, split, warp);
        const float partial_denominator = split_denominators[stats_index];
        if (partial_denominator == 0.0f) continue;
        const float partial_maximum = split_maxima[stats_index];
        const float next_maximum = fmaxf(maximum, partial_maximum);
        const float correction = __expf(maximum - next_maximum);
        const float partial_correction = __expf(partial_maximum - next_maximum);
        denominator = denominator * correction + partial_denominator * partial_correction;
#pragma unroll
        for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
            const uint32_t dim = lane + part * 32u;
            const float partial = split_values[qwen38_attention_split_value_index(
                    0u, kv_head, split, warp, dim)];
            accumulators[part] = fmaf(
                    partial_correction, partial, accumulators[part] * correction);
        }
        maximum = next_maximum;
    }
    if (lane == 0u) {
        maxima[query_head] = maximum;
        denominators[query_head] = denominator;
    }
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        values[static_cast<uint64_t>(query_head) * kHeadDim + lane + part * 32u] =
                accumulators[part];
    }
}

__global__ void qwen38_stream_finalize_stats_kernel(
        const float *__restrict__ maxima,
        const float *__restrict__ denominators,
        const float *__restrict__ values,
        float *__restrict__ out) {
    const uint32_t kv_head = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    if (kv_head >= kKvHeads || warp >= kGqaHeads) return;
    const uint32_t query_head = kv_head * kGqaHeads + warp;
    const float denominator = denominators[query_head];
    const uint64_t base = static_cast<uint64_t>(query_head) * kHeadDim;
#pragma unroll
    for (uint32_t part = 0u; part < kHeadDim / 32u; ++part) {
        const uint32_t dim = lane + part * 32u;
        out[base + dim] = qwen38_attention_round_bf16(
                denominator > 0.0f ? values[base + dim] / denominator : 0.0f);
    }
}

__global__ void qwen38_stream_replicate_row8_kernel(float *__restrict__ values) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t row_width = static_cast<uint64_t>(kQDim);
    if (index >= row_width) return;
    for (uint32_t row = 1u; row < kBatch; ++row) {
        values[static_cast<uint64_t>(row) * row_width + index] = values[index];
    }
}

}  // namespace

struct axiom_qwen38_attention_layer {
    axiom_runtime *runtime = nullptr;
    int device = -1;
    uint32_t layer = 0u;
    uint32_t max_context = 0u;
    uint32_t position = 0u;
    uint32_t spec_base_position = 0u;
    uint64_t device_bytes = 0u;
    bool spec_active = false;
    bool spec_forwarded = false;
    cudaStream_t spec_stream = nullptr;
    bool spec_device_position = false;
    /* `AXIOM_QWEN38_ATTENTION_BATCHED=0` is a diagnostic/reference mode.
     * It keeps the old F32 cache solely for an external numerical comparison;
     * production uses fixed-scale E4M3FN K/V. */
    bool scalar_f32_cache_mode = false;
    /* The temporal M8 path is a single sequence. For long-context serving the
     * scalar compatibility lanes share that same causal cache; retaining eight
     * duplicate columns would multiply KV memory by eight without adding state.
     * Set AXIOM_QWEN38_COMPACT_KV=0 only for the independent-column diagnostic. */
    uint32_t cache_columns = kBatch;
    bool yarn_enabled = true;
    bool kv_fp8_parity_enabled = false;
    bool shared_projection_input = true;
    bool parallel_projection_streams = true;
    cudaStream_t k_projection_stream = nullptr;
    cudaStream_t v_projection_stream = nullptr;
    cudaEvent_t projection_input_ready = nullptr;
    cudaEvent_t k_projection_done = nullptr;
    cudaEvent_t v_projection_done = nullptr;
    bool streaming_kv = false;
    /* Short-context DSpark/M8 uses the same target weights and a compact
     * contiguous hot cache. Long-context requests stay on the durable page
     * provider; the API selects this mode only when the complete request fits
     * in the hot window. */
    bool streaming_temporal_hot = false;
    axiom_qwen38_kv_tier *kv_tier = nullptr;
    uint32_t tier_layer = 0u;
    uint32_t stream_hot_pages = 0u;
    uint32_t stream_current_page = 0u;
    bool stream_page_dirty = false;
    void *stream_host_page = nullptr;
    /* Set only after a non-capturing warm launch of the fixed FlashInfer
     * BF16-Q/E4M3-KV/BF16-O D256 specialization. */
    bool flashinfer_ready = false;
    bool flashinfer_split_kv = false;

    axiom_qwen38_fp8_linear *q_proj = nullptr;
    axiom_qwen38_fp8_linear *k_proj = nullptr;
    axiom_qwen38_fp8_linear *v_proj = nullptr;
    axiom_qwen38_fp8_linear *o_proj = nullptr;

    axiom_device_buffer *input_norm_weight = nullptr;
    axiom_device_buffer *q_norm_weight = nullptr;
    axiom_device_buffer *k_norm_weight = nullptr;
    axiom_device_buffer *norm = nullptr;
    axiom_device_buffer *q_gate = nullptr;
    axiom_device_buffer *q = nullptr;
    axiom_device_buffer *gate = nullptr;
    axiom_device_buffer *k = nullptr;
    axiom_device_buffer *v = nullptr;
    axiom_device_buffer *attention = nullptr;
    axiom_device_buffer *gated_attention = nullptr;
    axiom_device_buffer *attention_split_values = nullptr;
    axiom_device_buffer *attention_split_maxima = nullptr;
    axiom_device_buffer *attention_split_denominators = nullptr;
    axiom_device_buffer *flashinfer_q_bf16 = nullptr;
    axiom_device_buffer *flashinfer_attention_bf16 = nullptr;
    axiom_device_buffer *flashinfer_kv_length_device = nullptr;
    axiom_device_buffer *flashinfer_split_kv_workspace = nullptr;
    axiom_device_buffer *attention_reference = nullptr;
    axiom_device_buffer *k_cache = nullptr;
    axiom_device_buffer *v_cache = nullptr;
    axiom_device_buffer *k_cache_f32 = nullptr;
    axiom_device_buffer *v_cache_f32 = nullptr;
    axiom_device_buffer *kv_fp8_parity_metrics = nullptr;
    axiom_device_buffer *stream_current_page_device = nullptr;
    axiom_device_buffer *stream_hot_page_table = nullptr;
    axiom_device_buffer **stream_hot_page_device = nullptr;
    uint32_t *stream_hot_page_ids = nullptr;
    axiom_device_buffer *stream_io_page_device = nullptr;
    axiom_device_buffer *stream_maxima = nullptr;
    axiom_device_buffer *stream_denominators = nullptr;
    axiom_device_buffer *stream_values = nullptr;
};

namespace {

int qwen38_attention_project_qkv(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *q_gate,
        float *k,
        float *v,
        void *stream) {
    if (!layer || !input || !q_gate || !k || !v) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = AXIOM_OK;
    if (layer->parallel_projection_streams) {
        const cudaStream_t main_stream = static_cast<cudaStream_t>(stream);
        if (layer->shared_projection_input) {
            rc = axiom_qwen38_fp8_linear_prepare_input_f32_device(
                    layer->q_proj, input, stream);
        }
        if (rc == AXIOM_OK) {
            rc = cuda_status(cudaEventRecord(layer->projection_input_ready, main_stream));
        }
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                layer->k_projection_stream, layer->projection_input_ready, 0u));
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                layer->v_projection_stream, layer->projection_input_ready, 0u));
        if (rc != AXIOM_OK) return rc;

        if (layer->shared_projection_input) {
            rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                    layer->q_proj, layer->q_proj, q_gate, stream);
            if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                    layer->k_proj, layer->q_proj, k,
                    reinterpret_cast<void *>(layer->k_projection_stream));
            if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(
                    layer->k_projection_done, layer->k_projection_stream));
            if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                    layer->v_proj, layer->q_proj, v,
                    reinterpret_cast<void *>(layer->v_projection_stream));
        } else {
            rc = axiom_qwen38_fp8_linear_forward_f32_device(
                    layer->q_proj, input, q_gate, stream);
            if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(
                    layer->k_proj, input, k,
                    reinterpret_cast<void *>(layer->k_projection_stream));
            if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(
                    layer->k_projection_done, layer->k_projection_stream));
            if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(
                    layer->v_proj, input, v,
                    reinterpret_cast<void *>(layer->v_projection_stream));
        }
        if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(
                layer->v_projection_done, layer->v_projection_stream));
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                main_stream, layer->k_projection_done, 0u));
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                main_stream, layer->v_projection_done, 0u));
        return rc;
    }
    if (layer->shared_projection_input) {
        rc = axiom_qwen38_fp8_linear_prepare_input_f32_device(
                layer->q_proj, input, stream);
        if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                layer->q_proj, layer->q_proj, q_gate, stream);
        if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                layer->k_proj, layer->q_proj, k, stream);
        if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_prepared_f32_device(
                layer->v_proj, layer->q_proj, v, stream);
        return rc;
    }
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->q_proj, input, q_gate, stream);
    if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->k_proj, input, k, stream);
    if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->v_proj, input, v, stream);
    return rc;
}

}  // namespace

extern "C" void axiom_qwen38_attention_layer_destroy(axiom_qwen38_attention_layer *layer) {
    if (!layer) return;
    if (layer->device >= 0) (void)cudaSetDevice(layer->device);
    if (layer->v_projection_stream) (void)cudaStreamDestroy(layer->v_projection_stream);
    if (layer->k_projection_stream) (void)cudaStreamDestroy(layer->k_projection_stream);
    if (layer->v_projection_done) (void)cudaEventDestroy(layer->v_projection_done);
    if (layer->k_projection_done) (void)cudaEventDestroy(layer->k_projection_done);
    if (layer->projection_input_ready) (void)cudaEventDestroy(layer->projection_input_ready);
    axiom_qwen38_fp8_linear_destroy(layer->o_proj);
    axiom_qwen38_fp8_linear_destroy(layer->v_proj);
    axiom_qwen38_fp8_linear_destroy(layer->k_proj);
    axiom_qwen38_fp8_linear_destroy(layer->q_proj);
    axiom_qwen38_kv_tier_host_page_free(layer->stream_host_page);
    layer->stream_host_page = nullptr;
    axiom_device_buffer_destroy(layer->stream_values);
    axiom_device_buffer_destroy(layer->stream_denominators);
    axiom_device_buffer_destroy(layer->stream_maxima);
    axiom_device_buffer_destroy(layer->stream_io_page_device);
    for (uint32_t slot = 0u; slot < layer->stream_hot_pages; ++slot) {
        axiom_device_buffer_destroy(layer->stream_hot_page_device[slot]);
    }
    delete[] layer->stream_hot_page_device;
    delete[] layer->stream_hot_page_ids;
    layer->stream_hot_page_device = nullptr;
    layer->stream_hot_page_ids = nullptr;
    layer->stream_hot_pages = 0u;
    layer->stream_current_page_device = nullptr;
    axiom_device_buffer_destroy(layer->stream_hot_page_table);
    layer->stream_hot_page_table = nullptr;
    axiom_device_buffer_destroy(layer->kv_fp8_parity_metrics);
    axiom_device_buffer_destroy(layer->v_cache_f32);
    axiom_device_buffer_destroy(layer->k_cache_f32);
    axiom_device_buffer_destroy(layer->v_cache);
    axiom_device_buffer_destroy(layer->k_cache);
    axiom_device_buffer_destroy(layer->attention_reference);
    axiom_device_buffer_destroy(layer->flashinfer_split_kv_workspace);
    axiom_device_buffer_destroy(layer->flashinfer_kv_length_device);
    axiom_device_buffer_destroy(layer->flashinfer_attention_bf16);
    axiom_device_buffer_destroy(layer->flashinfer_q_bf16);
    axiom_device_buffer_destroy(layer->attention_split_denominators);
    axiom_device_buffer_destroy(layer->attention_split_maxima);
    axiom_device_buffer_destroy(layer->attention_split_values);
    axiom_device_buffer_destroy(layer->gated_attention);
    axiom_device_buffer_destroy(layer->attention);
    axiom_device_buffer_destroy(layer->v);
    axiom_device_buffer_destroy(layer->k);
    axiom_device_buffer_destroy(layer->gate);
    axiom_device_buffer_destroy(layer->q);
    axiom_device_buffer_destroy(layer->q_gate);
    axiom_device_buffer_destroy(layer->norm);
    axiom_device_buffer_destroy(layer->k_norm_weight);
    axiom_device_buffer_destroy(layer->q_norm_weight);
    axiom_device_buffer_destroy(layer->input_norm_weight);
    delete layer;
}

extern "C" int axiom_qwen38_attention_layer_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        uint32_t layer_index,
        uint32_t max_context,
        axiom_qwen38_attention_layer **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || !out || device < 0 || max_context == 0u || (layer_index % 4u) != 3u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_attention_layer *layer = new (std::nothrow) axiom_qwen38_attention_layer();
    if (!layer) return AXIOM_ERR_BUDGET;
    layer->runtime = runtime;
    layer->device = device;
    layer->layer = layer_index;
    layer->max_context = max_context;
    layer->streaming_kv = env_enabled("AXIOM_QWEN38_KV_STREAMING");
    layer->streaming_temporal_hot = layer->streaming_kv &&
            env_enabled("AXIOM_QWEN38_KV_TEMPORAL8");
    layer->shared_projection_input =
            !env_disabled("AXIOM_QWEN38_ATTENTION_SHARED_FP8_INPUT");
    layer->parallel_projection_streams =
            !env_disabled("AXIOM_QWEN38_PARALLEL_PROJECTIONS");
    if (layer->parallel_projection_streams) {
        cudaError_t status = cudaStreamCreateWithFlags(
                &layer->k_projection_stream, cudaStreamNonBlocking);
        if (status == cudaSuccess) status = cudaStreamCreateWithFlags(
                &layer->v_projection_stream, cudaStreamNonBlocking);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(
                &layer->projection_input_ready, cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(
                &layer->k_projection_done, cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(
                &layer->v_projection_done, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            const int create_rc = cuda_status(status);
            axiom_qwen38_attention_layer_destroy(layer);
            return create_rc;
        }
    }
    if (layer->streaming_kv) {
        layer->stream_hot_pages = stream_hot_pages_from_env();
        layer->stream_hot_page_device =
                new (std::nothrow) axiom_device_buffer *[layer->stream_hot_pages]{};
        layer->stream_hot_page_ids = new (std::nothrow) uint32_t[layer->stream_hot_pages];
        if (!layer->stream_hot_page_device || !layer->stream_hot_page_ids) {
            axiom_qwen38_attention_layer_destroy(layer);
            return AXIOM_ERR_BUDGET;
        }
    }

    char prefix[128]{};
    const int prefix_written = std::snprintf(
            prefix, sizeof(prefix), "model.language_model.layers.%u.", layer_index);
    if (prefix_written <= 0 || static_cast<size_t>(prefix_written) >= sizeof(prefix)) {
        axiom_qwen38_attention_layer_destroy(layer);
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    char name[192]{};
    auto named = [&](const char *suffix) -> const char * {
        const int written = std::snprintf(name, sizeof(name), "%s%s", prefix, suffix);
        return written > 0 && static_cast<size_t>(written) < sizeof(name) ? name : nullptr;
    };
    auto load_fp8 = [&](const char *base_suffix,
                        axiom_qwen38_fp8_linear **target) -> int {
        const char *base = named(base_suffix);
        if (!base) return AXIOM_ERR_INVALID_ARGUMENT;
        char base_copy[192]{};
        const int written = std::snprintf(base_copy, sizeof(base_copy), "%s", base);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(base_copy)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        return axiom_qwen38_fp8_linear_load_auto(model, device, base_copy, target);
    };
    auto load_weight = [&](const char *suffix, uint64_t count, axiom_device_buffer **target) -> int {
        const char *tensor = named(suffix);
        return tensor ? upload_bf16_as_f32(model, runtime, tensor, 1u, count, true, target)
                      : AXIOM_ERR_INVALID_ARGUMENT;
    };
    int rc = load_weight("input_layernorm.weight", kHidden, &layer->input_norm_weight);
    if (rc == AXIOM_OK) rc = load_weight("self_attn.q_norm.weight", kHeadDim, &layer->q_norm_weight);
    if (rc == AXIOM_OK) rc = load_weight("self_attn.k_norm.weight", kHeadDim, &layer->k_norm_weight);
    if (rc == AXIOM_OK) rc = load_fp8("self_attn.q_proj", &layer->q_proj);
    if (rc == AXIOM_OK) rc = load_fp8("self_attn.k_proj", &layer->k_proj);
    if (rc == AXIOM_OK) rc = load_fp8("self_attn.v_proj", &layer->v_proj);
    if (rc == AXIOM_OK) rc = load_fp8("self_attn.o_proj", &layer->o_proj);
    if (rc != AXIOM_OK ||
        axiom_qwen38_fp8_linear_rows(layer->q_proj) != kQGateDim ||
        axiom_qwen38_fp8_linear_cols(layer->q_proj) != kHidden ||
        axiom_qwen38_fp8_linear_rows(layer->k_proj) != kKvDim ||
        axiom_qwen38_fp8_linear_cols(layer->k_proj) != kHidden ||
        axiom_qwen38_fp8_linear_rows(layer->v_proj) != kKvDim ||
        axiom_qwen38_fp8_linear_cols(layer->v_proj) != kHidden ||
        axiom_qwen38_fp8_linear_rows(layer->o_proj) != kHidden ||
        axiom_qwen38_fp8_linear_cols(layer->o_proj) != kQDim) {
        axiom_qwen38_attention_layer_destroy(layer);
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }

    const uint64_t hidden_batch = static_cast<uint64_t>(kHidden) * kBatch * sizeof(float);
    const uint64_t q_gate_batch = static_cast<uint64_t>(kQGateDim) * kBatch * sizeof(float);
    const uint64_t q_batch = static_cast<uint64_t>(kQDim) * kBatch * sizeof(float);
    const uint64_t q_batch_bf16 = static_cast<uint64_t>(kQDim) * kBatch * sizeof(uint16_t);
    const uint64_t kv_batch = static_cast<uint64_t>(kKvDim) * kBatch * sizeof(float);
    uint64_t cache_elements = 0u;
    uint64_t cache_fp8_bytes = 0u;
    uint64_t cache_f32_bytes = 0u;
    uint64_t split_values = 0u;
    uint64_t split_values_bytes = 0u;
    uint64_t split_stats = 0u;
    uint64_t split_stats_bytes = 0u;
    layer->cache_columns = (!env_disabled("AXIOM_QWEN38_COMPACT_KV") && max_context > 8192u)
            ? 1u : kBatch;
    const uint64_t resident_cache_context = layer->streaming_kv
            ? (layer->streaming_temporal_hot ? kTemporalHotTokens : 0u) : max_context;
    if (!checked_mul(resident_cache_context, layer->cache_columns, &cache_elements) ||
        !checked_mul(cache_elements, kKvDim, &cache_elements) ||
        !checked_mul(cache_elements, sizeof(uint8_t), &cache_fp8_bytes) ||
        !checked_mul(cache_elements, sizeof(float), &cache_f32_bytes) ||
        !checked_mul(static_cast<uint64_t>(kBatch) * kHeads, kAttentionSplitK, &split_values) ||
        !checked_mul(split_values, kHeadDim, &split_values) ||
        !checked_mul(split_values, sizeof(float), &split_values_bytes) ||
        !checked_mul(static_cast<uint64_t>(kBatch) * kHeads, kAttentionSplitK, &split_stats) ||
        !checked_mul(split_stats, sizeof(float), &split_stats_bytes)) {
        axiom_qwen38_attention_layer_destroy(layer);
        return AXIOM_ERR_BUDGET;
    }
    layer->scalar_f32_cache_mode = !layer->streaming_kv &&
            env_disabled("AXIOM_QWEN38_ATTENTION_BATCHED");
    layer->yarn_enabled = !env_disabled("AXIOM_QWEN38_YARN");
    const std::pair<uint64_t, axiom_device_buffer **> common_allocations[] = {
        {hidden_batch, &layer->norm}, {q_gate_batch, &layer->q_gate},
        {q_batch, &layer->q}, {q_batch, &layer->gate},
        {kv_batch, &layer->k}, {kv_batch, &layer->v},
        {q_batch, &layer->attention}, {q_batch, &layer->gated_attention},
        {split_values_bytes, &layer->attention_split_values},
        {split_stats_bytes, &layer->attention_split_maxima},
        {split_stats_bytes, &layer->attention_split_denominators},
    };
    for (const auto &allocation : common_allocations) {
        rc = axiom_device_buffer_create(runtime, allocation.second, allocation.first);
        if (rc != AXIOM_OK) {
            axiom_qwen38_attention_layer_destroy(layer);
            return rc;
        }
    }
    if (!layer->streaming_kv || layer->streaming_temporal_hot) {
        const std::pair<uint64_t, axiom_device_buffer **> resident_allocations[] = {
            {q_batch_bf16, &layer->flashinfer_q_bf16},
            {q_batch_bf16, &layer->flashinfer_attention_bf16},
            {sizeof(uint32_t), &layer->flashinfer_kv_length_device},
            {cache_fp8_bytes, &layer->k_cache}, {cache_fp8_bytes, &layer->v_cache},
        };
        for (const auto &allocation : resident_allocations) {
            rc = axiom_device_buffer_create(runtime, allocation.second, allocation.first);
            if (rc != AXIOM_OK) {
                axiom_qwen38_attention_layer_destroy(layer);
                return rc;
            }
        }
        if (resident_cache_context != 0u && resident_cache_context <= kTemporalHotTokens &&
            !env_disabled("AXIOM_QWEN38_FLASHINFER_SPLIT_KV")) {
            rc = axiom_device_buffer_create(
                    runtime, &layer->flashinfer_split_kv_workspace,
                    kFlashInferWorkspaceBytes);
            if (rc != AXIOM_OK) {
                axiom_qwen38_attention_layer_destroy(layer);
                return rc;
            }
            layer->flashinfer_split_kv = true;
        }
    }
    if (layer->streaming_kv) {
        for (uint32_t slot = 0u; slot < layer->stream_hot_pages && rc == AXIOM_OK; ++slot) {
            rc = axiom_device_buffer_create(
                    runtime, &layer->stream_hot_page_device[slot],
                    AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        }
        if (rc == AXIOM_OK) {
            layer->stream_current_page_device = layer->stream_hot_page_device[0];
        }
        if (rc == AXIOM_OK) {
            rc = axiom_device_buffer_create(
                    runtime, &layer->stream_hot_page_table,
                    static_cast<uint64_t>(layer->stream_hot_pages) * sizeof(void *));
        }
        if (rc == AXIOM_OK) {
            std::vector<void *> hot_page_ptrs(layer->stream_hot_pages, nullptr);
            for (uint32_t slot = 0u; slot < layer->stream_hot_pages && rc == AXIOM_OK; ++slot) {
                rc = buffer_pointer(layer->stream_hot_page_device[slot], &hot_page_ptrs[slot]);
            }
            if (rc == AXIOM_OK) {
                rc = axiom_device_buffer_upload(
                        layer->stream_hot_page_table, 0u, hot_page_ptrs.data(),
                        static_cast<uint64_t>(hot_page_ptrs.size()) * sizeof(void *));
            }
        }
        const std::pair<uint64_t, axiom_device_buffer **> streaming_allocations[] = {
            {AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES, &layer->stream_io_page_device},
            {static_cast<uint64_t>(kHeads) * sizeof(float), &layer->stream_maxima},
            {static_cast<uint64_t>(kHeads) * sizeof(float), &layer->stream_denominators},
            {static_cast<uint64_t>(kQDim) * sizeof(float), &layer->stream_values},
        };
        for (const auto &allocation : streaming_allocations) {
            if (rc != AXIOM_OK) break;
            rc = axiom_device_buffer_create(runtime, allocation.second, allocation.first);
        }
        if (rc != AXIOM_OK) {
            axiom_qwen38_attention_layer_destroy(layer);
            return rc;
        }
    }
    if (layer->scalar_f32_cache_mode) {
        rc = axiom_device_buffer_create(runtime, &layer->k_cache_f32, cache_f32_bytes);
        if (rc == AXIOM_OK) {
            rc = axiom_device_buffer_create(runtime, &layer->v_cache_f32, cache_f32_bytes);
        }
        if (rc != AXIOM_OK) {
            axiom_qwen38_attention_layer_destroy(layer);
            return rc;
        }
    }
    rc = axiom_qwen38_attention_layer_reset(layer);
    if (rc != AXIOM_OK) {
        axiom_qwen38_attention_layer_destroy(layer);
        return rc;
    }

    /* The generated dispatcher configures dynamic shared memory and queries
     * occupancy. Do that once outside any CUDA graph capture. A failure is
     * deliberately non-fatal: the capture-safe tiled GQA implementation
     * remains available as the native fallback. */
    if ((!layer->streaming_kv || layer->streaming_temporal_hot) &&
        max_context >= kBatch && !env_disabled("AXIOM_QWEN38_FLASHINFER")) {
        void *flashinfer_q_bf16 = nullptr;
        void *flashinfer_attention_bf16 = nullptr;
        void *flashinfer_split_kv_workspace = nullptr;
        void *k_cache = nullptr;
        void *v_cache = nullptr;
        int warm_rc = buffer_pointer(layer->flashinfer_q_bf16, &flashinfer_q_bf16);
        if (warm_rc == AXIOM_OK) {
            warm_rc = buffer_pointer(layer->flashinfer_attention_bf16, &flashinfer_attention_bf16);
        }
        if (warm_rc == AXIOM_OK) warm_rc = buffer_pointer(layer->k_cache, &k_cache);
        if (warm_rc == AXIOM_OK) warm_rc = buffer_pointer(layer->v_cache, &v_cache);
        if (warm_rc == AXIOM_OK && layer->flashinfer_split_kv) {
            warm_rc = buffer_pointer(
                    layer->flashinfer_split_kv_workspace,
                    &flashinfer_split_kv_workspace);
        }
        if (warm_rc == AXIOM_OK) {
            const uint32_t warm_context = layer->streaming_temporal_hot
                    ? kTemporalHotTokens : max_context;
            const uint32_t warm_kv_len = warm_context < 1024u ? warm_context : 1024u;
            warm_rc = axiom_qwen38_flashinfer_temporal8_bf16_e4m3_device(
                    static_cast<const uint16_t *>(flashinfer_q_bf16),
                    static_cast<const uint8_t *>(k_cache),
                    static_cast<const uint8_t *>(v_cache),
                    static_cast<uint16_t *>(flashinfer_attention_bf16),
                    warm_kv_len, nullptr,
                    static_cast<uint16_t *>(flashinfer_split_kv_workspace), nullptr);
        }
        if (warm_rc == AXIOM_OK && cudaStreamSynchronize(nullptr) == cudaSuccess) {
            layer->flashinfer_ready = true;
        } else {
            /* Clear the launch error so later fallback work can proceed. */
            (void)cudaGetLastError();
        }
    }

    uint64_t bytes = 0u;
    const uint64_t allocations_bytes[] = {
        axiom_qwen38_fp8_linear_device_bytes(layer->q_proj),
        axiom_qwen38_fp8_linear_device_bytes(layer->k_proj),
        axiom_qwen38_fp8_linear_device_bytes(layer->v_proj),
        axiom_qwen38_fp8_linear_device_bytes(layer->o_proj),
        static_cast<uint64_t>(kHidden + kHeadDim + kHeadDim) * sizeof(float),
        hidden_batch, q_gate_batch, q_batch, q_batch, kv_batch, kv_batch,
        q_batch, q_batch, split_values_bytes, split_stats_bytes, split_stats_bytes,
        (!layer->streaming_kv && !layer->streaming_temporal_hot) ? 0u : q_batch_bf16,
        (!layer->streaming_kv && !layer->streaming_temporal_hot) ? 0u : q_batch_bf16,
        (!layer->streaming_kv && !layer->streaming_temporal_hot) ? 0u : sizeof(uint32_t),
        layer->flashinfer_split_kv ? kFlashInferWorkspaceBytes : 0u,
        resident_cache_context == 0u ? 0u : cache_fp8_bytes,
        resident_cache_context == 0u ? 0u : cache_fp8_bytes,
    };
    for (uint64_t amount : allocations_bytes) {
        if (!checked_add(bytes, amount, &bytes)) {
            axiom_qwen38_attention_layer_destroy(layer);
            return AXIOM_ERR_BUDGET;
        }
    }
    if (layer->scalar_f32_cache_mode &&
        (!checked_add(bytes, cache_f32_bytes, &bytes) ||
         !checked_add(bytes, cache_f32_bytes, &bytes))) {
        axiom_qwen38_attention_layer_destroy(layer);
        return AXIOM_ERR_BUDGET;
    }
    if (layer->streaming_kv) {
        if (!checked_add(bytes, static_cast<uint64_t>(layer->stream_hot_pages + 1u) *
                AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES, &bytes) ||
            !checked_add(bytes, static_cast<uint64_t>(layer->stream_hot_pages) * sizeof(void *),
                         &bytes) ||
            !checked_add(bytes, static_cast<uint64_t>(kHeads) * sizeof(float), &bytes) ||
            !checked_add(bytes, static_cast<uint64_t>(kHeads) * sizeof(float), &bytes) ||
            !checked_add(bytes, static_cast<uint64_t>(kQDim) * sizeof(float), &bytes)) {
            axiom_qwen38_attention_layer_destroy(layer);
            return AXIOM_ERR_BUDGET;
        }
    }
    layer->device_bytes = bytes;
    *out = layer;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_reset(axiom_qwen38_attention_layer *layer) {
    if (!layer || layer->max_context == 0u ||
        (layer->streaming_kv
                ? (!layer->stream_current_page_device || !layer->stream_hot_page_device[0] ||
                   !layer->stream_hot_page_table ||
                   !layer->stream_io_page_device ||
                   !layer->stream_maxima || !layer->stream_denominators || !layer->stream_values)
                : (!layer->k_cache || !layer->v_cache))) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (layer->streaming_kv) {
        void *current_page = nullptr;
        void *io_page = nullptr;
        void *maxima = nullptr;
        void *denominators = nullptr;
        void *values = nullptr;
        void *hot_k_cache = nullptr;
        void *hot_v_cache = nullptr;
        void *flashinfer_q = nullptr;
        void *flashinfer_attention = nullptr;
        void *flashinfer_kv_length = nullptr;
        layer->stream_current_page_device = layer->stream_hot_page_device[0];
        int stream_rc = buffer_pointer(layer->stream_current_page_device, &current_page);
        if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->stream_io_page_device, &io_page);
        if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->stream_maxima, &maxima);
        if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->stream_denominators, &denominators);
        if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->stream_values, &values);
        if (stream_rc == AXIOM_OK && layer->streaming_temporal_hot) {
            stream_rc = buffer_pointer(layer->k_cache, &hot_k_cache);
            if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->v_cache, &hot_v_cache);
            if (stream_rc == AXIOM_OK) stream_rc = buffer_pointer(layer->flashinfer_q_bf16, &flashinfer_q);
            if (stream_rc == AXIOM_OK) {
                stream_rc = buffer_pointer(layer->flashinfer_attention_bf16, &flashinfer_attention);
            }
            if (stream_rc == AXIOM_OK) {
                stream_rc = buffer_pointer(layer->flashinfer_kv_length_device, &flashinfer_kv_length);
            }
        }
        if (stream_rc != AXIOM_OK) return stream_rc;
        cudaError_t status = cudaMemset(
                current_page, 0, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        for (uint32_t slot = 1u;
             status == cudaSuccess && slot < layer->stream_hot_pages; ++slot) {
            void *hot_page = nullptr;
            stream_rc = buffer_pointer(layer->stream_hot_page_device[slot], &hot_page);
            if (stream_rc != AXIOM_OK) return stream_rc;
            status = cudaMemset(hot_page, 0, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        }
        if (status == cudaSuccess) {
            status = cudaMemset(io_page, 0, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        }
        if (status == cudaSuccess) status = cudaMemset(maxima, 0, kHeads * sizeof(float));
        if (status == cudaSuccess) status = cudaMemset(denominators, 0, kHeads * sizeof(float));
        if (status == cudaSuccess) status = cudaMemset(values, 0, kQDim * sizeof(float));
        if (status == cudaSuccess && layer->streaming_temporal_hot) {
            status = cudaMemset(
                    hot_k_cache, 0,
                    static_cast<size_t>(kTemporalHotTokens) * kKvDim * sizeof(uint8_t));
        }
        if (status == cudaSuccess && layer->streaming_temporal_hot) {
            status = cudaMemset(
                    hot_v_cache, 0,
                    static_cast<size_t>(kTemporalHotTokens) * kKvDim * sizeof(uint8_t));
        }
        if (status == cudaSuccess && layer->streaming_temporal_hot) {
            status = cudaMemset(flashinfer_q, 0, kQDim * kBatch * sizeof(uint16_t));
        }
        if (status == cudaSuccess && layer->streaming_temporal_hot) {
            status = cudaMemset(flashinfer_attention, 0, kQDim * kBatch * sizeof(uint16_t));
        }
        if (status == cudaSuccess && layer->streaming_temporal_hot) {
            status = cudaMemset(flashinfer_kv_length, 0, sizeof(uint32_t));
        }
        if (status != cudaSuccess) return cuda_status(status);
        layer->position = 0u;
        layer->stream_current_page = 0u;
        layer->stream_page_dirty = false;
        for (uint32_t slot = 0u; slot < layer->stream_hot_pages; ++slot) {
            layer->stream_hot_page_ids[slot] = std::numeric_limits<uint32_t>::max();
        }
        layer->stream_hot_page_ids[0] = 0u;
        layer->spec_base_position = 0u;
        layer->spec_active = false;
        layer->spec_forwarded = false;
        layer->spec_stream = nullptr;
        layer->spec_device_position = false;
        return AXIOM_OK;
    }
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    void *k_cache_f32 = nullptr;
    void *v_cache_f32 = nullptr;
    void *parity_metrics = nullptr;
    void *flashinfer_q_bf16 = nullptr;
    void *flashinfer_attention_bf16 = nullptr;
    void *flashinfer_kv_length_device = nullptr;
    int rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc == AXIOM_OK && layer->k_cache_f32) {
        rc = buffer_pointer(layer->k_cache_f32, &k_cache_f32);
        if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache_f32, &v_cache_f32);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_metrics) {
        rc = buffer_pointer(layer->kv_fp8_parity_metrics, &parity_metrics);
    }
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->flashinfer_q_bf16, &flashinfer_q_bf16);
    if (rc == AXIOM_OK) {
        rc = buffer_pointer(layer->flashinfer_attention_bf16, &flashinfer_attention_bf16);
    }
    if (rc == AXIOM_OK) {
        rc = buffer_pointer(layer->flashinfer_kv_length_device, &flashinfer_kv_length_device);
    }
    if (rc != AXIOM_OK) return rc;
    uint64_t elements = 0u;
    uint64_t fp8_bytes = 0u;
    uint64_t f32_bytes = 0u;
    if (!checked_mul(static_cast<uint64_t>(layer->max_context), layer->cache_columns, &elements) ||
        !checked_mul(elements, kKvDim, &elements) ||
        !checked_mul(elements, sizeof(uint8_t), &fp8_bytes) ||
        !checked_mul(elements, sizeof(float), &f32_bytes) ||
        fp8_bytes > std::numeric_limits<size_t>::max() ||
        f32_bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_BUDGET;
    }
    cudaError_t status = cudaMemset(k_cache, 0, static_cast<size_t>(fp8_bytes));
    if (status == cudaSuccess) status = cudaMemset(v_cache, 0, static_cast<size_t>(fp8_bytes));
    if (status == cudaSuccess && layer->k_cache_f32) {
        status = cudaMemset(k_cache_f32, 0, static_cast<size_t>(f32_bytes));
    }
    if (status == cudaSuccess && layer->v_cache_f32) {
        status = cudaMemset(v_cache_f32, 0, static_cast<size_t>(f32_bytes));
    }
    if (status == cudaSuccess && parity_metrics) {
        status = cudaMemset(parity_metrics, 0, kKvParityMetricCount * sizeof(uint32_t));
    }
    const size_t q_bf16_bytes = static_cast<size_t>(kQDim) * kBatch * sizeof(uint16_t);
    if (status == cudaSuccess) status = cudaMemset(flashinfer_q_bf16, 0, q_bf16_bytes);
    if (status == cudaSuccess) status = cudaMemset(flashinfer_attention_bf16, 0, q_bf16_bytes);
    if (status == cudaSuccess) status = cudaMemset(flashinfer_kv_length_device, 0, sizeof(uint32_t));
    if (status != cudaSuccess) return cuda_status(status);
    layer->position = 0u;
    layer->spec_base_position = 0u;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    layer->spec_device_position = false;
    return AXIOM_OK;
}

namespace {

int stream_wait_page_io(
        axiom_qwen38_kv_tier *tier,
        const uint32_t operation,
        const uint32_t layer,
        const uint32_t logical_page,
        void *host_page) {
    if (!tier || !host_page) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_kv_tier_request request{};
    request.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    request.operation = operation;
    request.family = AXIOM_QWEN38_KV_TIER_TARGET;
    request.layer = layer;
    request.logical_page = logical_page;
    request.host_page = host_page;
    request.host_page_bytes = AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES;
    request.user_data = logical_page;
    int rc = axiom_qwen38_kv_tier_submit(tier, &request);
    if (rc != AXIOM_OK) return rc;
    axiom_qwen38_kv_tier_completion completion{};
    uint32_t completed = 0u;
    rc = axiom_qwen38_kv_tier_wait(tier, 1u, &completion, 1u, &completed);
    if (rc != AXIOM_OK || completed != 1u || completion.result != AXIOM_OK ||
        completion.transferred_bytes != AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES) {
        return rc == AXIOM_OK ? AXIOM_ERR_IO : rc;
    }
    return AXIOM_OK;
}

int stream_flush_current_page(axiom_qwen38_attention_layer *layer) {
    if (!layer || !layer->streaming_kv || !layer->kv_tier || !layer->stream_host_page ||
        !layer->stream_current_page_device) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!layer->stream_page_dirty) return AXIOM_OK;
    void *current_page = nullptr;
    int rc = buffer_pointer(layer->stream_current_page_device, &current_page);
    if (rc != AXIOM_OK) return rc;
    cudaError_t status = cudaMemcpy(
            layer->stream_host_page, current_page,
            AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_status(status);
    rc = stream_wait_page_io(
            layer->kv_tier, AXIOM_QWEN38_KV_TIER_WRITE, layer->tier_layer,
            layer->stream_current_page, layer->stream_host_page);
    if (rc == AXIOM_OK) layer->stream_page_dirty = false;
    return rc;
}

int stream_read_page(
        axiom_qwen38_attention_layer *layer,
        const uint32_t logical_page,
        void *io_page,
        const uint8_t **out_page) {
    if (!layer || !layer->streaming_kv || !layer->kv_tier || !layer->stream_host_page ||
        !out_page || logical_page >= layer->stream_current_page) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t slot = logical_page % layer->stream_hot_pages;
    if (layer->stream_hot_page_ids[slot] == logical_page) {
        void *hot_page = nullptr;
        const int hot_rc = buffer_pointer(layer->stream_hot_page_device[slot], &hot_page);
        if (hot_rc != AXIOM_OK) return hot_rc;
        *out_page = static_cast<const uint8_t *>(hot_page);
        return AXIOM_OK;
    }
    if (!io_page) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = stream_wait_page_io(
            layer->kv_tier, AXIOM_QWEN38_KV_TIER_READ, layer->tier_layer,
            logical_page, layer->stream_host_page);
    if (rc != AXIOM_OK) return rc;
    cudaError_t status = cudaMemcpy(
            io_page, layer->stream_host_page,
            AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return cuda_status(status);
    *out_page = static_cast<const uint8_t *>(io_page);
    return AXIOM_OK;
}

int forward_streaming_kv(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out) {
    if (!layer || !input || !out || !layer->streaming_kv || !layer->kv_tier ||
        layer->spec_active || layer->position >= layer->max_context ||
        !layer->stream_host_page) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t expected_page = layer->position / AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
    const bool page_boundary = layer->position != 0u &&
            layer->position % AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS == 0u;
    if (expected_page != layer->stream_current_page &&
        !(page_boundary && expected_page == layer->stream_current_page + 1u)) {
        return AXIOM_ERR_RUNTIME;
    }
    if (page_boundary) {
        int rc = stream_flush_current_page(layer);
        if (rc != AXIOM_OK) return rc;
        ++layer->stream_current_page;
        const uint32_t slot = layer->stream_current_page % layer->stream_hot_pages;
        layer->stream_current_page_device = layer->stream_hot_page_device[slot];
        layer->stream_hot_page_ids[slot] = layer->stream_current_page;
        void *current_page = nullptr;
        rc = buffer_pointer(layer->stream_current_page_device, &current_page);
        if (rc != AXIOM_OK) return rc;
        const cudaError_t clear_status = cudaMemset(
                current_page, 0, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        if (clear_status != cudaSuccess) return cuda_status(clear_status);
    }

    void *input_norm_weight = nullptr;
    void *norm = nullptr;
    void *q_gate = nullptr;
    void *q = nullptr;
    void *gate = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *q_norm_weight = nullptr;
    void *k_norm_weight = nullptr;
    void *attention = nullptr;
    void *gated_attention = nullptr;
    void *split_values = nullptr;
    void *split_maxima = nullptr;
    void *split_denominators = nullptr;
    void *current_page = nullptr;
    void *io_page = nullptr;
    void *stream_maxima = nullptr;
    void *stream_denominators = nullptr;
    void *stream_values = nullptr;
    void *hot_k_cache = nullptr;
    void *hot_v_cache = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &input_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_gate, &q_gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q, &q);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gate, &gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k, &k);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v, &v);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_norm_weight, &q_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_norm_weight, &k_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention, &attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated_attention, &gated_attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_values, &split_values);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_maxima, &split_maxima);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_denominators, &split_denominators);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->stream_current_page_device, &current_page);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->stream_io_page_device, &io_page);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->stream_maxima, &stream_maxima);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->stream_denominators, &stream_denominators);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->stream_values, &stream_values);
    if (rc == AXIOM_OK && layer->streaming_temporal_hot) {
        rc = buffer_pointer(layer->k_cache, &hot_k_cache);
        if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &hot_v_cache);
    }
    if (rc != AXIOM_OK) return rc;

    qwen38_attention_rmsnorm8_kernel<<<kBatch, kThreads>>>(
            static_cast<const float *>(input_norm_weight), input,
            static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_attention_project_qkv(
            layer, static_cast<const float *>(norm), static_cast<float *>(q_gate),
            static_cast<float *>(k), static_cast<float *>(v), nullptr);
    if (rc != AXIOM_OK) return rc;
    rc = split_q_gate_round_k8(
            static_cast<const float *>(q_gate), static_cast<float *>(q),
            static_cast<float *>(gate), static_cast<float *>(k), nullptr);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_runtime_qk_rmsnorm_w_f32_device(
            layer->runtime, layer->q, 0u, layer->q_norm_weight, 0u,
            kHeads, kHeadDim, kEps);
    if (rc == AXIOM_OK) rc = axiom_runtime_qk_rmsnorm_w_f32_device(
            layer->runtime, layer->k, 0u, layer->k_norm_weight, 0u,
            kKvHeads, kHeadDim, kEps);
    if (rc == AXIOM_OK) rc = launch_qwen38_rope_neox_partial_yarn(
            static_cast<float *>(q), kHeads, layer->position, layer->yarn_enabled, nullptr);
    if (rc == AXIOM_OK) rc = launch_qwen38_rope_neox_partial_yarn(
            static_cast<float *>(k), kKvHeads, layer->position, layer->yarn_enabled, nullptr);
    if (rc == AXIOM_OK) rc = round_bf16_inplace(static_cast<float *>(q), kQDim);
    if (rc == AXIOM_OK) rc = round_bf16_inplace(static_cast<float *>(k), kKvDim);
    if (rc != AXIOM_OK) return rc;
    qwen38_stream_store_kv_page_kernel<<<
            (kKvDim + kThreads - 1u) / kThreads, kThreads>>>(
            static_cast<const float *>(k), static_cast<const float *>(v),
            static_cast<uint8_t *>(current_page),
            layer->position % AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    layer->stream_page_dirty = true;
    if (layer->streaming_temporal_hot && layer->position < kTemporalHotTokens) {
        qwen38_stream_store_hot_kv_row_kernel<<<
                (kKvDim + kThreads - 1u) / kThreads, kThreads>>>(
                static_cast<const float *>(k), static_cast<const float *>(v),
                static_cast<uint8_t *>(hot_k_cache), static_cast<uint8_t *>(hot_v_cache),
                layer->position);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }

    const uint32_t current_page_tokens =
            layer->position % AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS + 1u;
    const uint32_t page_count = layer->stream_current_page + 1u;
    const bool all_pages_hot = page_count <= layer->stream_hot_pages;
    if (all_pages_hot) {
        void *page_table = nullptr;
        rc = buffer_pointer(layer->stream_hot_page_table, &page_table);
        if (rc != AXIOM_OK) return rc;
        qwen38_stream_reference_hot_pages_kernel<<<
                kHeads, kHeadDim, static_cast<size_t>(kHeadDim) * sizeof(float)>>>(
                static_cast<const float *>(q),
                reinterpret_cast<const uint8_t *const *>(page_table),
                layer->stream_hot_pages, page_count, current_page_tokens,
                static_cast<float *>(attention));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    } else {
        qwen38_stream_init_stats_kernel<<<
                (kQDim + kThreads - 1u) / kThreads, kThreads>>>(
                static_cast<float *>(stream_maxima), static_cast<float *>(stream_denominators),
                static_cast<float *>(stream_values));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        const bool streaming_fast = env_enabled("AXIOM_QWEN38_KV_STREAMING_FAST");
        for (uint32_t logical_page = 0u;
             logical_page <= layer->stream_current_page; ++logical_page) {
            const uint8_t *page = static_cast<const uint8_t *>(current_page);
            uint32_t page_tokens = AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
            if (logical_page == layer->stream_current_page) {
                page_tokens = current_page_tokens;
            } else {
                rc = stream_read_page(layer, logical_page, io_page, &page);
                if (rc != AXIOM_OK) return rc;
            }
            if (!streaming_fast) {
                qwen38_stream_reference_page_kernel<<<
                        kHeads, kHeadDim, static_cast<size_t>(kHeadDim) * sizeof(float)>>>(
                        static_cast<const float *>(q), page,
                        page + AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                        page_tokens, static_cast<float *>(stream_maxima),
                        static_cast<float *>(stream_denominators),
                        static_cast<float *>(stream_values));
            } else {
                cudaError_t clear_split_status = cudaMemset(
                        split_denominators, 0,
                        static_cast<size_t>(kBatch) * kHeads * kAttentionSplitK * sizeof(float));
                if (clear_split_status != cudaSuccess) return cuda_status(clear_split_status);
                qwen38_attention_core_gqa_splitk8_kernel<false, 1u><<<
                        dim3(kKvHeads, 1u, 1u), kAttentionThreads>>>(
                        static_cast<const float *>(q), page,
                        page + AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                        static_cast<float *>(split_values), static_cast<float *>(split_maxima),
                        static_cast<float *>(split_denominators), page_tokens, 0u, nullptr,
                        AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS);
                if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
                qwen38_stream_accumulate_page_kernel<<<kKvHeads, kAttentionThreads>>>(
                        static_cast<const float *>(split_values),
                        static_cast<const float *>(split_maxima),
                        static_cast<const float *>(split_denominators),
                        static_cast<float *>(stream_maxima),
                        static_cast<float *>(stream_denominators),
                        static_cast<float *>(stream_values));
            }
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        }
        qwen38_stream_finalize_stats_kernel<<<kKvHeads, kAttentionThreads>>>(
                static_cast<const float *>(stream_maxima),
                static_cast<const float *>(stream_denominators),
                static_cast<const float *>(stream_values), static_cast<float *>(attention));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    qwen38_stream_replicate_row8_kernel<<<
            (kQDim + kThreads - 1u) / kThreads, kThreads>>>(static_cast<float *>(attention));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t total_q = static_cast<uint64_t>(kQDim) * kBatch;
    qwen38_sigmoid_gate_kernel<<<
            static_cast<uint32_t>((total_q + kThreads - 1u) / kThreads), kThreads>>>(
            static_cast<const float *>(attention), static_cast<const float *>(gate),
            static_cast<float *>(gated_attention));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->o_proj, static_cast<const float *>(gated_attention), out, nullptr);
    if (rc == AXIOM_OK) ++layer->position;
    return rc;
}

}  // namespace

extern "C" int axiom_qwen38_attention_layer_kv_tier_bind(
        axiom_qwen38_attention_layer *layer,
        axiom_qwen38_kv_tier *tier,
        const uint32_t tier_layer) {
    if (!layer || !tier || !layer->streaming_kv || layer->spec_active ||
        layer->position != 0u || tier_layer >= AXIOM_QWEN38_KV_TIER_TARGET_LAYERS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    /* The HTTP scheduler serializes native work. At position zero the same
     * device hot-page allocation can be rebound to another generation-owned
     * NVMe file; this is what gives a logical session its own durable tier
     * without reallocating the 16-layer VRAM working set. */
    if (layer->kv_tier && layer->tier_layer != tier_layer) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!layer->stream_host_page) {
        int rc = axiom_qwen38_kv_tier_host_page_alloc(
                AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES, &layer->stream_host_page);
        if (rc != AXIOM_OK) return rc;
    }
    layer->kv_tier = tier;
    layer->tier_layer = tier_layer;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_kv_tier_flush(
        axiom_qwen38_attention_layer *layer) {
    if (!layer || !layer->streaming_kv || !layer->kv_tier) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return stream_flush_current_page(layer);
}

extern "C" uint32_t axiom_qwen38_attention_layer_position(const axiom_qwen38_attention_layer *layer) {
    return layer ? layer->position : 0u;
}

extern "C" uint64_t axiom_qwen38_attention_layer_device_bytes(const axiom_qwen38_attention_layer *layer) {
    return layer ? layer->device_bytes : 0u;
}

extern "C" int axiom_qwen38_attention_layer_restore_position(
        axiom_qwen38_attention_layer *layer,
        const uint32_t position) {
    if (!layer || position > layer->max_context || layer->spec_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (layer->streaming_kv) {
        const uint32_t current_page = position == 0u
                ? 0u : (position - 1u) / AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
        if (current_page >= (layer->max_context + AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS - 1u) /
                AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        const uint32_t slot = current_page % layer->stream_hot_pages;
        if (position != 0u && layer->stream_hot_page_ids[slot] != current_page) {
            /* The caller must import every page needed to seed the hot window
             * before installing the logical position. */
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        layer->stream_current_page = current_page;
        layer->stream_current_page_device = layer->stream_hot_page_device[slot];
        layer->stream_page_dirty = false;
    }
    layer->position = position;
    layer->spec_base_position = position;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_materialize_device_position(
        axiom_qwen38_attention_layer *layer,
        const uint32_t position) {
    if (!layer || position > layer->max_context || layer->spec_active ||
        (layer->streaming_kv && !layer->streaming_temporal_hot) ||
        (layer->streaming_temporal_hot && position > kTemporalHotTokens)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!layer->streaming_kv) {
        layer->position = position;
        layer->spec_base_position = position;
        return AXIOM_OK;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    void *hot_k_cache = nullptr;
    void *hot_v_cache = nullptr;
    int rc = buffer_pointer(layer->k_cache, &hot_k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &hot_v_cache);
    if (rc != AXIOM_OK) return rc;

    const uint32_t current_page = position == 0u
            ? 0u : (position - 1u) / AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
    const uint32_t page_count = position == 0u ? 1u : current_page + 1u;
    for (uint32_t logical_page = 0u; logical_page < page_count; ++logical_page) {
        const uint32_t slot = logical_page % layer->stream_hot_pages;
        void *page = nullptr;
        rc = buffer_pointer(layer->stream_hot_page_device[slot], &page);
        if (rc != AXIOM_OK) return rc;
        cudaError_t status = cudaMemset(
                page, 0, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES);
        const uint32_t page_start =
                logical_page * AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
        const uint32_t valid_tokens = position > page_start
                ? std::min<uint32_t>(
                        AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS,
                        position - page_start)
                : 0u;
        const size_t valid_bytes = static_cast<size_t>(valid_tokens) * kKvDim;
        const size_t source_offset = static_cast<size_t>(page_start) * kKvDim;
        if (status == cudaSuccess && valid_bytes != 0u) {
            status = cudaMemcpy(
                    page, static_cast<const uint8_t *>(hot_k_cache) + source_offset,
                    valid_bytes, cudaMemcpyDeviceToDevice);
        }
        if (status == cudaSuccess && valid_bytes != 0u) {
            status = cudaMemcpy(
                    static_cast<uint8_t *>(page) +
                            AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                    static_cast<const uint8_t *>(hot_v_cache) + source_offset,
                    valid_bytes, cudaMemcpyDeviceToDevice);
        }
        if (status != cudaSuccess) return cuda_status(status);
        layer->stream_hot_page_ids[slot] = logical_page;
    }
    const uint32_t current_slot = current_page % layer->stream_hot_pages;
    layer->stream_current_page = current_page;
    layer->stream_current_page_device = layer->stream_hot_page_device[current_slot];
    layer->stream_page_dirty = position != 0u;
    layer->position = position;
    layer->spec_base_position = position;
    return AXIOM_OK;
}

namespace {

int attention_kv_page_args(
        const axiom_qwen38_attention_layer *layer,
        const uint32_t logical_page,
        const void *host_page,
        const uint64_t host_page_bytes) {
    if (!layer || !host_page || host_page_bytes != AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES ||
        logical_page >= (layer->max_context + AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS - 1u) /
                AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

}  // namespace

extern "C" int axiom_qwen38_attention_layer_kv_page_export(
        const axiom_qwen38_attention_layer *layer,
        const uint32_t logical_page,
        void *host_page,
        const uint64_t host_page_bytes) {
    int rc = attention_kv_page_args(layer, logical_page, host_page, host_page_bytes);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t page_start = static_cast<uint64_t>(logical_page) *
            AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
    if (layer->streaming_kv &&
        (!layer->streaming_temporal_hot || page_start >= kTemporalHotTokens)) {
        const uint32_t slot = logical_page % layer->stream_hot_pages;
        if (layer->stream_hot_page_ids[slot] != logical_page) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        void *page = nullptr;
        rc = buffer_pointer(layer->stream_hot_page_device[slot], &page);
        if (rc != AXIOM_OK) return rc;
        return cuda_status(cudaMemcpy(
                host_page, page, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES,
                cudaMemcpyDeviceToHost));
    }
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc != AXIOM_OK) return rc;
    std::memset(host_page, 0, static_cast<size_t>(host_page_bytes));
    const uint64_t resident_limit = layer->streaming_temporal_hot
            ? std::min<uint64_t>(layer->max_context, kTemporalHotTokens)
            : layer->max_context;
    const uint64_t valid_tokens = std::min<uint64_t>(
            AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS,
            resident_limit - page_start);
    const size_t valid_bytes = static_cast<size_t>(valid_tokens * kKvDim * sizeof(uint8_t));
    const uint64_t device_offset = page_start * kKvDim * sizeof(uint8_t);
    cudaError_t status = cudaMemcpy(
            static_cast<uint8_t *>(host_page),
            static_cast<const uint8_t *>(k_cache) + device_offset,
            valid_bytes, cudaMemcpyDeviceToHost);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
                static_cast<uint8_t *>(host_page) +
                        AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                static_cast<const uint8_t *>(v_cache) + device_offset,
                valid_bytes, cudaMemcpyDeviceToHost);
    }
    return cuda_status(status);
}

extern "C" int axiom_qwen38_attention_layer_kv_page_import(
        axiom_qwen38_attention_layer *layer,
        const uint32_t logical_page,
        const void *host_page,
        const uint64_t host_page_bytes) {
    int rc = attention_kv_page_args(layer, logical_page, host_page, host_page_bytes);
    if (rc != AXIOM_OK || layer->spec_active) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (layer->streaming_kv) {
        const uint32_t slot = logical_page % layer->stream_hot_pages;
        void *hot_page = nullptr;
        rc = buffer_pointer(layer->stream_hot_page_device[slot], &hot_page);
        if (rc != AXIOM_OK) return rc;
        cudaError_t status = cudaMemcpy(
                hot_page, host_page, AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES,
                cudaMemcpyHostToDevice);
        if (status != cudaSuccess) return cuda_status(status);
        const uint64_t page_start = static_cast<uint64_t>(logical_page) *
                AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
        if (layer->streaming_temporal_hot && page_start < kTemporalHotTokens) {
            void *hot_k_cache = nullptr;
            void *hot_v_cache = nullptr;
            rc = buffer_pointer(layer->k_cache, &hot_k_cache);
            if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &hot_v_cache);
            if (rc != AXIOM_OK) return rc;
            const uint64_t valid_tokens = std::min<uint64_t>(
                    AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS,
                    kTemporalHotTokens - page_start);
            const size_t valid_bytes = static_cast<size_t>(valid_tokens) * kKvDim;
            const size_t destination_offset = static_cast<size_t>(page_start) * kKvDim;
            status = cudaMemcpy(
                    static_cast<uint8_t *>(hot_k_cache) + destination_offset,
                    host_page, valid_bytes, cudaMemcpyHostToDevice);
            if (status == cudaSuccess) {
                status = cudaMemcpy(
                        static_cast<uint8_t *>(hot_v_cache) + destination_offset,
                        static_cast<const uint8_t *>(host_page) +
                                AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                        valid_bytes, cudaMemcpyHostToDevice);
            }
            if (status != cudaSuccess) return cuda_status(status);
        }
        layer->stream_hot_page_ids[slot] = logical_page;
        if (logical_page == layer->stream_current_page) {
            layer->stream_current_page_device = layer->stream_hot_page_device[slot];
            layer->stream_page_dirty = false;
        }
        return AXIOM_OK;
    }
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc != AXIOM_OK) return rc;
    const uint64_t page_start = static_cast<uint64_t>(logical_page) *
            AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS;
    const uint64_t valid_tokens = std::min<uint64_t>(
            AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS, layer->max_context - page_start);
    const size_t valid_bytes = static_cast<size_t>(valid_tokens * kKvDim * sizeof(uint8_t));
    const uint64_t device_offset = page_start * kKvDim * sizeof(uint8_t);
    cudaError_t status = cudaMemcpy(
            static_cast<uint8_t *>(k_cache) + device_offset,
            static_cast<const uint8_t *>(host_page), valid_bytes, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
                static_cast<uint8_t *>(v_cache) + device_offset,
                static_cast<const uint8_t *>(host_page) +
                        AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES / 2u,
                valid_bytes, cudaMemcpyHostToDevice);
    }
    return cuda_status(status);
}

extern "C" int axiom_qwen38_attention_layer_kv_fp8_parity_enable(
        axiom_qwen38_attention_layer *layer) {
    if (!layer || !layer->runtime || layer->spec_active || layer->position != 0u ||
        layer->max_context > kKvParityMaxContext) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    uint64_t cache_elements = 0u;
    uint64_t cache_f32_bytes = 0u;
    const uint64_t attention_reference_bytes =
            static_cast<uint64_t>(kQDim) * kBatch * sizeof(float);
    if (!checked_mul(static_cast<uint64_t>(layer->max_context), layer->cache_columns, &cache_elements) ||
        !checked_mul(cache_elements, kKvDim, &cache_elements) ||
        !checked_mul(cache_elements, sizeof(float), &cache_f32_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    uint64_t extra_bytes = 0u;
    if (!layer->k_cache_f32 && !checked_add(extra_bytes, cache_f32_bytes, &extra_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    if (!layer->v_cache_f32 && !checked_add(extra_bytes, cache_f32_bytes, &extra_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    if (!layer->attention_reference &&
        !checked_add(extra_bytes, attention_reference_bytes, &extra_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    if (!layer->kv_fp8_parity_metrics &&
        !checked_add(extra_bytes,
                     static_cast<uint64_t>(kKvParityMetricCount) * sizeof(uint32_t),
                     &extra_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    uint64_t next_bytes = 0u;
    if (!checked_add(layer->device_bytes, extra_bytes, &next_bytes)) return AXIOM_ERR_BUDGET;

    bool made_k_reference = false;
    bool made_v_reference = false;
    bool made_attention_reference = false;
    bool made_metrics = false;
    int rc = AXIOM_OK;
    if (!layer->k_cache_f32) {
        rc = axiom_device_buffer_create(layer->runtime, &layer->k_cache_f32, cache_f32_bytes);
        made_k_reference = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK && !layer->v_cache_f32) {
        rc = axiom_device_buffer_create(layer->runtime, &layer->v_cache_f32, cache_f32_bytes);
        made_v_reference = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK && !layer->attention_reference) {
        rc = axiom_device_buffer_create(
                layer->runtime, &layer->attention_reference, attention_reference_bytes);
        made_attention_reference = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK && !layer->kv_fp8_parity_metrics) {
        rc = axiom_device_buffer_create(
                layer->runtime, &layer->kv_fp8_parity_metrics,
                static_cast<uint64_t>(kKvParityMetricCount) * sizeof(uint32_t));
        made_metrics = rc == AXIOM_OK;
    }
    if (rc != AXIOM_OK) {
        if (made_metrics) axiom_device_buffer_destroy(layer->kv_fp8_parity_metrics);
        if (made_attention_reference) axiom_device_buffer_destroy(layer->attention_reference);
        if (made_v_reference) axiom_device_buffer_destroy(layer->v_cache_f32);
        if (made_k_reference) axiom_device_buffer_destroy(layer->k_cache_f32);
        if (made_metrics) layer->kv_fp8_parity_metrics = nullptr;
        if (made_attention_reference) layer->attention_reference = nullptr;
        if (made_v_reference) layer->v_cache_f32 = nullptr;
        if (made_k_reference) layer->k_cache_f32 = nullptr;
        return rc;
    }
    void *metrics = nullptr;
    rc = buffer_pointer(layer->kv_fp8_parity_metrics, &metrics);
    if (rc != AXIOM_OK) return rc;
    const cudaError_t status = cudaMemset(
            metrics, 0, kKvParityMetricCount * sizeof(uint32_t));
    if (status != cudaSuccess) return cuda_status(status);
    layer->device_bytes = next_bytes;
    layer->kv_fp8_parity_enabled = true;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_kv_fp8_parity_get(
        const axiom_qwen38_attention_layer *layer,
        axiom_qwen38_attention_kv_fp8_parity *out) {
    if (!layer || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_ATTENTION_KV_FP8_PARITY_ABI_VERSION;
    out->scale = kKvFp8Scale;
    out->descale = kKvFp8Descale;
    out->enabled = layer->kv_fp8_parity_enabled ? 1u : 0u;
    if (!layer->kv_fp8_parity_metrics) return AXIOM_OK;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    void *metrics_device = nullptr;
    const int rc = buffer_pointer(layer->kv_fp8_parity_metrics, &metrics_device);
    if (rc != AXIOM_OK) return rc;
    uint32_t metrics[kKvParityMetricCount]{};
    const cudaError_t status = cudaMemcpy(
            metrics, metrics_device, sizeof(metrics), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_status(status);
    union { uint32_t bits; float value; } cache_error{};
    union { uint32_t bits; float value; } attention_error{};
    cache_error.bits = metrics[kKvParityMaxAbsBits];
    attention_error.bits = metrics[kKvParityAttentionMaxAbsBits];
    out->sampled_values = metrics[kKvParitySampleCount];
    out->nonfinite_values = metrics[kKvParityNonfiniteCount];
    out->max_cache_abs_error = cache_error.value;
    out->max_attention_abs_error = attention_error.value;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_forward_f32_device(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        void *stream) {
    if (!layer || !input || !out || stream != nullptr || layer->spec_active || !layer->runtime ||
        layer->position >= layer->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (layer->streaming_kv) return forward_streaming_kv(layer, input, out);
    void *input_norm_weight = nullptr;
    void *norm = nullptr;
    void *q_gate = nullptr;
    void *q = nullptr;
    void *gate = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    void *k_cache_f32 = nullptr;
    void *v_cache_f32 = nullptr;
    void *attention = nullptr;
    void *gated_attention = nullptr;
    void *q_norm_weight = nullptr;
    void *k_norm_weight = nullptr;
    void *split_values = nullptr;
    void *split_maxima = nullptr;
    void *split_denominators = nullptr;
    void *attention_reference = nullptr;
    void *parity_metrics = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &input_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_gate, &q_gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q, &q);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gate, &gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k, &k);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v, &v);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc == AXIOM_OK &&
        (layer->scalar_f32_cache_mode || layer->kv_fp8_parity_enabled)) {
        rc = buffer_pointer(layer->k_cache_f32, &k_cache_f32);
        if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache_f32, &v_cache_f32);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->kv_fp8_parity_metrics, &parity_metrics);
    }
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention, &attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated_attention, &gated_attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_norm_weight, &q_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_norm_weight, &k_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_values, &split_values);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_maxima, &split_maxima);
    if (rc == AXIOM_OK) {
        rc = buffer_pointer(layer->attention_split_denominators, &split_denominators);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->attention_reference, &attention_reference);
    }
    if (rc != AXIOM_OK) return rc;

    qwen38_attention_rmsnorm8_kernel<<<kBatch, kThreads>>>(
            static_cast<const float *>(input_norm_weight), input, static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_attention_project_qkv(
            layer, static_cast<const float *>(norm), static_cast<float *>(q_gate),
            static_cast<float *>(k), static_cast<float *>(v), nullptr);
    if (rc != AXIOM_OK) return rc;
    const bool use_batched = !layer->scalar_f32_cache_mode;
    if (use_batched) {
        rc = split_q_gate_round_k8(
                static_cast<const float *>(q_gate), static_cast<float *>(q),
                static_cast<float *>(gate), static_cast<float *>(k), nullptr);
        if (rc != AXIOM_OK) return rc;
    } else {
        rc = round_bf16_inplace(
                static_cast<float *>(q_gate), static_cast<uint64_t>(kQGateDim) * kBatch);
        if (rc == AXIOM_OK) {
            rc = round_bf16_inplace(static_cast<float *>(k), static_cast<uint64_t>(kKvDim) * kBatch);
        }
        if (rc == AXIOM_OK) {
            rc = round_bf16_inplace(static_cast<float *>(v), static_cast<uint64_t>(kKvDim) * kBatch);
        }
        if (rc != AXIOM_OK) return rc;
        qwen38_split_q_gate_kernel<<<
                dim3((kQDim + kThreads - 1u) / kThreads, kBatch), kThreads>>>(
                static_cast<const float *>(q_gate), static_cast<float *>(q),
                static_cast<float *>(gate));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }

    if (use_batched) {
        qwen38_qk_norm_partial_rope_cache8_kernel<<<
                dim3(kHeads + kKvHeads, kBatch), kThreads>>>(
                static_cast<float *>(q), static_cast<float *>(k),
                static_cast<const float *>(q_norm_weight),
                static_cast<const float *>(k_norm_weight), static_cast<const float *>(v),
                static_cast<uint8_t *>(k_cache), static_cast<uint8_t *>(v_cache),
                static_cast<float *>(k_cache_f32), static_cast<float *>(v_cache_f32),
                layer->cache_columns, layer->max_context, layer->position, layer->yarn_enabled,
                static_cast<uint32_t *>(parity_metrics));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        qwen38_attention_core_fp8_reference_kernel<false><<<
                dim3(kHeads, kBatch), kHeadDim,
                static_cast<size_t>(kHeadDim) * sizeof(float)>>>(
                static_cast<const float *>(q), static_cast<const uint8_t *>(k_cache),
                static_cast<const uint8_t *>(v_cache), static_cast<float *>(attention),
                layer->position + 1u, layer->cache_columns, 0u, nullptr, layer->max_context);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        if (layer->kv_fp8_parity_enabled) {
            qwen38_attention_core_bf16_cache_reference_kernel<false><<<
                    dim3(kHeads, kBatch), kHeadDim,
                    static_cast<size_t>(kHeadDim) * sizeof(float)>>>(
                    static_cast<const float *>(q), static_cast<const float *>(k_cache_f32),
                    static_cast<const float *>(v_cache_f32), static_cast<float *>(attention_reference),
                    layer->position + 1u, layer->cache_columns, 0u, nullptr, layer->max_context);
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
            const uint64_t attention_count = static_cast<uint64_t>(kQDim) * kBatch;
            qwen38_kv_record_attention_parity_kernel<<<
                    static_cast<uint32_t>((attention_count + kThreads - 1u) / kThreads), kThreads>>>(
                    static_cast<const float *>(attention),
                    static_cast<const float *>(attention_reference), attention_count,
                    static_cast<uint32_t *>(parity_metrics));
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        }
    } else {
        const uint64_t q_bytes = static_cast<uint64_t>(kQDim) * sizeof(float);
        const uint64_t kv_bytes = static_cast<uint64_t>(kKvDim) * sizeof(float);
        const uint64_t cache_column_bytes =
                static_cast<uint64_t>(layer->max_context) * kv_bytes;
        for (uint32_t column = 0u; column < kBatch; ++column) {
            const uint64_t q_offset = static_cast<uint64_t>(column) * q_bytes;
            const uint64_t kv_offset = static_cast<uint64_t>(column) * kv_bytes;
            rc = axiom_runtime_qk_rmsnorm_w_f32_device(
                    layer->runtime, layer->q, q_offset, layer->q_norm_weight, 0u,
                    kHeads, kHeadDim, kEps);
            if (rc == AXIOM_OK) {
                rc = axiom_runtime_qk_rmsnorm_w_f32_device(
                        layer->runtime, layer->k, kv_offset, layer->k_norm_weight, 0u,
                        kKvHeads, kHeadDim, kEps);
            }
            if (rc == AXIOM_OK) {
                rc = launch_qwen38_rope_neox_partial_yarn(
                        reinterpret_cast<float *>(static_cast<uint8_t *>(q) + q_offset),
                        kHeads, layer->position, layer->yarn_enabled, nullptr);
            }
            if (rc == AXIOM_OK) {
                rc = launch_qwen38_rope_neox_partial_yarn(
                        reinterpret_cast<float *>(static_cast<uint8_t *>(k) + kv_offset),
                        kKvHeads, layer->position, layer->yarn_enabled, nullptr);
            }
            if (rc == AXIOM_OK) {
                rc = round_bf16_inplace(
                        static_cast<float *>(q) + static_cast<uint64_t>(column) * kQDim, kQDim);
            }
            if (rc == AXIOM_OK) {
                rc = round_bf16_inplace(
                        static_cast<float *>(k) + static_cast<uint64_t>(column) * kKvDim, kKvDim);
            }
            if (rc != AXIOM_OK) return rc;
            const uint64_t cache_offset =
                    static_cast<uint64_t>(layer->cache_columns == 1u ? 0u : column) *
                    cache_column_bytes +
                    static_cast<uint64_t>(layer->position) * kv_bytes;
            cudaError_t status = cudaMemcpyAsync(
                    static_cast<uint8_t *>(k_cache_f32) + cache_offset,
                    static_cast<const uint8_t *>(k) + kv_offset,
                    static_cast<size_t>(kv_bytes), cudaMemcpyDeviceToDevice, nullptr);
            if (status == cudaSuccess) {
                status = cudaMemcpyAsync(
                        static_cast<uint8_t *>(v_cache_f32) + cache_offset,
                        static_cast<const uint8_t *>(v) + kv_offset,
                        static_cast<size_t>(kv_bytes), cudaMemcpyDeviceToDevice, nullptr);
            }
            if (status != cudaSuccess) return cuda_status(status);
            rc = axiom_runtime_attention_core_fast_f32_device(
                    layer->runtime, layer->q, q_offset, layer->k_cache_f32,
                    cache_offset, layer->v_cache_f32, cache_offset,
                    layer->attention, q_offset, kHeads, kKvHeads, kHeadDim,
                    layer->position + 1u, 0u);
            if (rc != AXIOM_OK) return rc;
        }
        const uint64_t kv_count = static_cast<uint64_t>(kKvDim) * kBatch;
        qwen38_cache_f32_to_e4m3fn8_kernel<<<
                static_cast<uint32_t>((kv_count + kThreads - 1u) / kThreads), kThreads>>>(
                static_cast<const float *>(k), static_cast<const float *>(v),
                static_cast<uint8_t *>(k_cache), static_cast<uint8_t *>(v_cache),
                layer->cache_columns, layer->max_context, layer->position,
                static_cast<uint32_t *>(parity_metrics));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    const uint64_t total_q = static_cast<uint64_t>(kQDim) * kBatch;
    if (!use_batched) {
        rc = round_bf16_inplace(static_cast<float *>(attention), total_q);
        if (rc != AXIOM_OK) return rc;
    }
    qwen38_sigmoid_gate_kernel<<<static_cast<uint32_t>((total_q + kThreads - 1u) / kThreads), kThreads>>>(
            static_cast<const float *>(attention), static_cast<const float *>(gate),
            static_cast<float *>(gated_attention));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->o_proj, static_cast<const float *>(gated_attention), out, nullptr);
    if (rc == AXIOM_OK) ++layer->position;
    return rc;
}

extern "C" int axiom_qwen38_attention_layer_spec_begin(
        axiom_qwen38_attention_layer *layer,
        void *stream) {
    if (!layer || layer->spec_active ||
        (layer->streaming_kv && !layer->streaming_temporal_hot) ||
        layer->position > layer->max_context ||
        kBatch > layer->max_context - layer->position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t cache_context = layer->streaming_temporal_hot
            ? kTemporalHotTokens : layer->max_context;
    if (layer->position > cache_context || kBatch > cache_context - layer->position) {
        return AXIOM_ERR_BUDGET;
    }
    layer->spec_base_position = layer->position;
    layer->spec_active = true;
    layer->spec_forwarded = false;
    layer->spec_stream = static_cast<cudaStream_t>(stream);
    layer->spec_device_position = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_forward_temporal8_f32_device(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        void *stream) {
    if (!layer || !input || !out || !layer->spec_active || layer->spec_device_position ||
        (layer->streaming_kv && !layer->streaming_temporal_hot) ||
        layer->spec_forwarded ||
        layer->spec_stream != static_cast<cudaStream_t>(stream) || !layer->runtime ||
        layer->spec_base_position > layer->max_context ||
        kBatch > layer->max_context - layer->spec_base_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t cache_context = layer->streaming_temporal_hot
            ? kTemporalHotTokens : layer->max_context;
    if (layer->spec_base_position > cache_context ||
        kBatch > cache_context - layer->spec_base_position) {
        return AXIOM_ERR_BUDGET;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    void *input_norm_weight = nullptr;
    void *q_norm_weight = nullptr;
    void *k_norm_weight = nullptr;
    void *norm = nullptr;
    void *q_gate = nullptr;
    void *q = nullptr;
    void *gate = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    void *k_cache_f32 = nullptr;
    void *v_cache_f32 = nullptr;
    void *parity_metrics = nullptr;
    void *attention = nullptr;
    void *gated_attention = nullptr;
    void *split_values = nullptr;
    void *split_maxima = nullptr;
    void *split_denominators = nullptr;
    void *attention_reference = nullptr;
    void *flashinfer_q_bf16 = nullptr;
    void *flashinfer_attention_bf16 = nullptr;
    void *flashinfer_split_kv_workspace = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &input_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_norm_weight, &q_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_norm_weight, &k_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_gate, &q_gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q, &q);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gate, &gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k, &k);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v, &v);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->k_cache_f32, &k_cache_f32);
        if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache_f32, &v_cache_f32);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->kv_fp8_parity_metrics, &parity_metrics);
    }
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention, &attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated_attention, &gated_attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_values, &split_values);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_maxima, &split_maxima);
    if (rc == AXIOM_OK) {
        rc = buffer_pointer(layer->attention_split_denominators, &split_denominators);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->attention_reference, &attention_reference);
    }
    if (rc == AXIOM_OK && layer->flashinfer_ready) {
        rc = buffer_pointer(layer->flashinfer_q_bf16, &flashinfer_q_bf16);
        if (rc == AXIOM_OK) {
            rc = buffer_pointer(layer->flashinfer_attention_bf16, &flashinfer_attention_bf16);
        }
        if (rc == AXIOM_OK && layer->flashinfer_split_kv) {
            rc = buffer_pointer(
                    layer->flashinfer_split_kv_workspace,
                    &flashinfer_split_kv_workspace);
        }
    }
    if (rc != AXIOM_OK) return rc;

    qwen38_attention_rmsnorm8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(input_norm_weight), input, static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_attention_project_qkv(
            layer, static_cast<const float *>(norm), static_cast<float *>(q_gate),
            static_cast<float *>(k), static_cast<float *>(v), stream);
    if (rc != AXIOM_OK) return rc;
    rc = split_q_gate_round_k8(
            static_cast<const float *>(q_gate), static_cast<float *>(q),
            static_cast<float *>(gate), static_cast<float *>(k), cuda_stream);
    if (rc != AXIOM_OK) return rc;
    qwen38_qk_norm_partial_rope_cache_temporal8_kernel<<<
            dim3(kHeads + kKvHeads, kBatch), kThreads, 0, cuda_stream>>>(
            static_cast<float *>(q), static_cast<float *>(k),
                static_cast<const float *>(q_norm_weight), static_cast<const float *>(k_norm_weight),
                static_cast<const float *>(v), static_cast<uint8_t *>(k_cache),
            static_cast<uint8_t *>(v_cache), static_cast<float *>(k_cache_f32),
            static_cast<float *>(v_cache_f32), layer->spec_base_position, nullptr,
            cache_context, layer->yarn_enabled, static_cast<uint32_t *>(parity_metrics));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    /* Host M8 is the scalar-parity oracle. Keep its exact fallback reduction
     * order; production decode reaches the FlashInfer specialization through
     * the device-position / CUDA-graph entry point below. */
    if (layer->spec_device_position && layer->flashinfer_ready &&
        !layer->kv_fp8_parity_enabled) {
        rc = flashinfer_convert_q_to_bf16(
                static_cast<const float *>(q), static_cast<uint16_t *>(flashinfer_q_bf16),
                cuda_stream);
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_flashinfer_temporal8_bf16_e4m3_device(
                    static_cast<const uint16_t *>(flashinfer_q_bf16),
                    static_cast<const uint8_t *>(k_cache), static_cast<const uint8_t *>(v_cache),
                    static_cast<uint16_t *>(flashinfer_attention_bf16),
                    layer->spec_base_position + kBatch, nullptr,
                    static_cast<uint16_t *>(flashinfer_split_kv_workspace), stream);
        }
        if (rc == AXIOM_OK) {
            rc = flashinfer_convert_out_to_f32(
                    static_cast<const uint16_t *>(flashinfer_attention_bf16),
                    static_cast<float *>(attention), cuda_stream);
        }
        if (rc != AXIOM_OK) return rc;
    } else {
        qwen38_attention_core_fp8_reference_kernel<true><<<
                dim3(kHeads, kBatch), kHeadDim,
                static_cast<size_t>(kHeadDim) * sizeof(float), cuda_stream>>>(
                static_cast<const float *>(q), static_cast<const uint8_t *>(k_cache),
                static_cast<const uint8_t *>(v_cache), static_cast<float *>(attention),
                0u, layer->cache_columns, layer->spec_base_position, nullptr, cache_context);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    if (layer->kv_fp8_parity_enabled) {
        qwen38_attention_core_bf16_cache_reference_kernel<true><<<
                dim3(kHeads, kBatch), kHeadDim, static_cast<size_t>(kHeadDim) * sizeof(float),
                cuda_stream>>>(
                static_cast<const float *>(q), static_cast<const float *>(k_cache_f32),
                static_cast<const float *>(v_cache_f32), static_cast<float *>(attention_reference),
                0u, layer->cache_columns, layer->spec_base_position, nullptr, cache_context);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        const uint64_t attention_count = static_cast<uint64_t>(kQDim) * kBatch;
        qwen38_kv_record_attention_parity_kernel<<<
                static_cast<uint32_t>((attention_count + kThreads - 1u) / kThreads), kThreads,
                0, cuda_stream>>>(
                static_cast<const float *>(attention),
                static_cast<const float *>(attention_reference), attention_count,
                static_cast<uint32_t *>(parity_metrics));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    const uint64_t total_q = static_cast<uint64_t>(kQDim) * kBatch;
    qwen38_sigmoid_gate_kernel<<<
            static_cast<uint32_t>((total_q + kThreads - 1u) / kThreads), kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(attention), static_cast<const float *>(gate),
            static_cast<float *>(gated_attention));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->o_proj, static_cast<const float *>(gated_attention), out, stream);
    if (rc == AXIOM_OK) layer->spec_forwarded = true;
    return rc;
}

extern "C" int axiom_qwen38_attention_layer_spec_begin_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *base_position_device,
        void *stream) {
    if (!layer || !base_position_device || layer->spec_active ||
        (layer->streaming_kv && !layer->streaming_temporal_hot)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    layer->spec_base_position = 0u;
    layer->spec_active = true;
    layer->spec_forwarded = false;
    layer->spec_stream = static_cast<cudaStream_t>(stream);
    layer->spec_device_position = true;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_forward_temporal8_f32_device_position(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        const uint32_t *base_position_device,
        void *stream) {
    if (!layer || !input || !out || !base_position_device || !layer->spec_active ||
        !layer->spec_device_position || layer->spec_forwarded ||
        layer->spec_stream != static_cast<cudaStream_t>(stream) || !layer->runtime ||
        (layer->streaming_kv && !layer->streaming_temporal_hot)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t cache_context = layer->streaming_temporal_hot
            ? kTemporalHotTokens : layer->max_context;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    void *input_norm_weight = nullptr;
    void *q_norm_weight = nullptr;
    void *k_norm_weight = nullptr;
    void *norm = nullptr;
    void *q_gate = nullptr;
    void *q = nullptr;
    void *gate = nullptr;
    void *k = nullptr;
    void *v = nullptr;
    void *k_cache = nullptr;
    void *v_cache = nullptr;
    void *k_cache_f32 = nullptr;
    void *v_cache_f32 = nullptr;
    void *parity_metrics = nullptr;
    void *attention = nullptr;
    void *gated_attention = nullptr;
    void *split_values = nullptr;
    void *split_maxima = nullptr;
    void *split_denominators = nullptr;
    void *attention_reference = nullptr;
    void *flashinfer_q_bf16 = nullptr;
    void *flashinfer_attention_bf16 = nullptr;
    void *flashinfer_kv_length_device = nullptr;
    void *flashinfer_split_kv_workspace = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &input_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_norm_weight, &q_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_norm_weight, &k_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q_gate, &q_gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->q, &q);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gate, &gate);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k, &k);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v, &v);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->k_cache, &k_cache);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache, &v_cache);
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->k_cache_f32, &k_cache_f32);
        if (rc == AXIOM_OK) rc = buffer_pointer(layer->v_cache_f32, &v_cache_f32);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->kv_fp8_parity_metrics, &parity_metrics);
    }
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention, &attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated_attention, &gated_attention);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_values, &split_values);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->attention_split_maxima, &split_maxima);
    if (rc == AXIOM_OK) {
        rc = buffer_pointer(layer->attention_split_denominators, &split_denominators);
    }
    if (rc == AXIOM_OK && layer->kv_fp8_parity_enabled) {
        rc = buffer_pointer(layer->attention_reference, &attention_reference);
    }
    if (rc == AXIOM_OK && layer->flashinfer_ready) {
        rc = buffer_pointer(layer->flashinfer_q_bf16, &flashinfer_q_bf16);
        if (rc == AXIOM_OK) {
            rc = buffer_pointer(layer->flashinfer_attention_bf16, &flashinfer_attention_bf16);
        }
        if (rc == AXIOM_OK) {
            rc = buffer_pointer(layer->flashinfer_kv_length_device, &flashinfer_kv_length_device);
        }
        if (rc == AXIOM_OK && layer->flashinfer_split_kv) {
            rc = buffer_pointer(
                    layer->flashinfer_split_kv_workspace,
                    &flashinfer_split_kv_workspace);
        }
    }
    if (rc != AXIOM_OK) return rc;

    qwen38_attention_rmsnorm8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(input_norm_weight), input, static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_attention_project_qkv(
            layer, static_cast<const float *>(norm), static_cast<float *>(q_gate),
            static_cast<float *>(k), static_cast<float *>(v), stream);
    if (rc != AXIOM_OK) return rc;
    rc = split_q_gate_round_k8(
            static_cast<const float *>(q_gate), static_cast<float *>(q),
            static_cast<float *>(gate), static_cast<float *>(k), cuda_stream);
    if (rc != AXIOM_OK) return rc;
    qwen38_qk_norm_partial_rope_cache_temporal8_kernel<<<
            dim3(kHeads + kKvHeads, kBatch), kThreads, 0, cuda_stream>>>(
            static_cast<float *>(q), static_cast<float *>(k),
                static_cast<const float *>(q_norm_weight), static_cast<const float *>(k_norm_weight),
                static_cast<const float *>(v), static_cast<uint8_t *>(k_cache),
            static_cast<uint8_t *>(v_cache), static_cast<float *>(k_cache_f32),
            static_cast<float *>(v_cache_f32), 0u, base_position_device, cache_context,
            layer->yarn_enabled, static_cast<uint32_t *>(parity_metrics));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (layer->flashinfer_ready && !layer->kv_fp8_parity_enabled) {
        qwen38_attention_temporal8_kv_length_kernel<<<1u, 1u, 0, cuda_stream>>>(
                static_cast<uint32_t *>(flashinfer_kv_length_device), base_position_device);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        rc = flashinfer_convert_q_to_bf16(
                static_cast<const float *>(q), static_cast<uint16_t *>(flashinfer_q_bf16),
                cuda_stream);
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_flashinfer_temporal8_bf16_e4m3_device(
                    static_cast<const uint16_t *>(flashinfer_q_bf16),
                    static_cast<const uint8_t *>(k_cache), static_cast<const uint8_t *>(v_cache),
                    static_cast<uint16_t *>(flashinfer_attention_bf16), cache_context,
                    static_cast<const uint32_t *>(flashinfer_kv_length_device),
                    static_cast<uint16_t *>(flashinfer_split_kv_workspace), stream);
        }
        if (rc == AXIOM_OK) {
            rc = flashinfer_convert_out_to_f32(
                    static_cast<const uint16_t *>(flashinfer_attention_bf16),
                    static_cast<float *>(attention), cuda_stream);
        }
        if (rc != AXIOM_OK) return rc;
    } else {
        qwen38_attention_core_gqa_splitk8_kernel<true, kAttentionSplitK><<<
                dim3(kKvHeads, kBatch, kAttentionSplitK), kAttentionThreads, 0, cuda_stream>>>(
                static_cast<const float *>(q), static_cast<const uint8_t *>(k_cache),
                static_cast<const uint8_t *>(v_cache), static_cast<float *>(split_values),
                static_cast<float *>(split_maxima), static_cast<float *>(split_denominators),
                0u, 0u, base_position_device, cache_context);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        qwen38_attention_merge_gqa_splitk8_kernel<<<
                dim3(kKvHeads, kBatch), kAttentionThreads, 0, cuda_stream>>>(
                static_cast<const float *>(split_values), static_cast<const float *>(split_maxima),
                static_cast<const float *>(split_denominators), static_cast<float *>(attention));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    if (layer->kv_fp8_parity_enabled) {
        qwen38_attention_core_bf16_cache_reference_kernel<true><<<
                dim3(kHeads, kBatch), kHeadDim, static_cast<size_t>(kHeadDim) * sizeof(float),
                cuda_stream>>>(
                static_cast<const float *>(q), static_cast<const float *>(k_cache_f32),
                static_cast<const float *>(v_cache_f32), static_cast<float *>(attention_reference),
                0u, layer->cache_columns, 0u, base_position_device, cache_context);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        const uint64_t attention_count = static_cast<uint64_t>(kQDim) * kBatch;
        qwen38_kv_record_attention_parity_kernel<<<
                static_cast<uint32_t>((attention_count + kThreads - 1u) / kThreads), kThreads,
                0, cuda_stream>>>(
                static_cast<const float *>(attention),
                static_cast<const float *>(attention_reference), attention_count,
                static_cast<uint32_t *>(parity_metrics));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    const uint64_t total_q = static_cast<uint64_t>(kQDim) * kBatch;
    qwen38_sigmoid_gate_kernel<<<
            static_cast<uint32_t>((total_q + kThreads - 1u) / kThreads), kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(attention), static_cast<const float *>(gate),
            static_cast<float *>(gated_attention));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->o_proj, static_cast<const float *>(gated_attention), out, stream);
    if (rc == AXIOM_OK) layer->spec_forwarded = true;
    return rc;
}

extern "C" int axiom_qwen38_attention_layer_spec_commit_prefix_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *consumed_tokens_device,
        void *stream) {
    if (!layer || !consumed_tokens_device || !layer->spec_active || !layer->spec_forwarded ||
        !layer->spec_device_position || layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    /* Logical cache position belongs to the controller's device scalar. The
     * uncommitted suffix is overwritten at that next scalar position. */
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    layer->spec_device_position = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_spec_finalize_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *consumed_tokens_device,
        const uint32_t *async_status_device,
        void *stream) {
    if (!layer || !consumed_tokens_device || !async_status_device || !layer->spec_active ||
        !layer->spec_forwarded || !layer->spec_device_position ||
        layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    /* Both success and failure leave suffix KV non-authoritative. A later
     * temporal write uses the controller position and replaces it; unlike
     * GDN there is no recurrent prefix tensor to copy. */
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    layer->spec_device_position = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_spec_commit_prefix(
        axiom_qwen38_attention_layer *layer,
        uint32_t consumed_tokens,
        void *stream) {
    if (!layer || !layer->spec_active || !layer->spec_forwarded || layer->spec_device_position ||
        layer->spec_stream != static_cast<cudaStream_t>(stream) ||
        consumed_tokens == 0u || consumed_tokens > kBatch ||
        layer->spec_base_position > layer->max_context ||
        consumed_tokens > layer->max_context - layer->spec_base_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    layer->position = layer->spec_base_position + consumed_tokens;
    layer->spec_base_position = layer->position;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    layer->spec_device_position = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_attention_layer_spec_abort(
        axiom_qwen38_attention_layer *layer,
        void *stream) {
    if (!layer || !layer->spec_active ||
        layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!layer->spec_device_position) layer->position = layer->spec_base_position;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    layer->spec_device_position = false;
    return AXIOM_OK;
}
