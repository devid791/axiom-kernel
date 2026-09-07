/* Native resident Gated DeltaNet mixer for unsloth/Qwen3.8-27B-NVFP4. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_bf16_linear.h"
#include "axiom/qwen38_fp8.h"
#include "axiom/qwen38_gdn.h"

namespace {

constexpr uint32_t kHidden = AXIOM_QWEN38_GDN_HIDDEN;
constexpr uint32_t kBatch = AXIOM_QWEN38_GDN_BATCH;
constexpr uint32_t kConvDim = AXIOM_QWEN38_GDN_CONV_DIM;
constexpr uint32_t kKeyHeads = AXIOM_QWEN38_GDN_KEY_HEADS;
constexpr uint32_t kValueHeads = AXIOM_QWEN38_GDN_VALUE_HEADS;
constexpr uint32_t kHeadDim = AXIOM_QWEN38_GDN_HEAD_DIM;
constexpr uint32_t kKeyDim = kKeyHeads * kHeadDim;
constexpr uint32_t kValueDim = kValueHeads * kHeadDim;
constexpr uint32_t kProjectionDim = kConvDim + kValueDim;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kDefaultTemporalValueTile = 8u;
constexpr float kEps = 1.0e-6f;

static_assert(kConvDim == kKeyDim + kKeyDim + kValueDim, "Qwen3.8 QKV layout changed");
static_assert(kHeadDim % kDefaultTemporalValueTile == 0u,
              "Temporal GDN tile must divide V");

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
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

int device_buffer_create(axiom_runtime *runtime, uint64_t bytes, axiom_device_buffer **out) {
    if (out) *out = nullptr;
    if (!runtime || !out || bytes == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_device_buffer_create(runtime, out, bytes);
}

int buffer_pointer(const axiom_device_buffer *buffer, void **out) {
    return axiom_device_buffer_cuda_pointer(buffer, out);
}

bool env_disabled(const char *name) {
    const char *value = std::getenv(name);
    return value && (value[0] == '0' || value[0] == 'f' || value[0] == 'F');
}

uint32_t temporal_value_tile_from_env() {
    const char *value = std::getenv("AXIOM_QWEN38_GDN_TEMPORAL_VALUE_TILE");
    if (!value || !value[0]) return kDefaultTemporalValueTile;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0') return kDefaultTemporalValueTile;
    switch (parsed) {
        case 8u:
        case 16u:
        case 32u:
        case 64u:
            return static_cast<uint32_t>(parsed);
        default:
            return kDefaultTemporalValueTile;
    }
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
    for (uint32_t dim = 0u; dim < expected_rank; ++dim) {
        if (info.shape[dim] == 0u || !checked_mul(count, info.shape[dim], &count)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    uint64_t bytes = 0u;
    if (count != expected_count || !checked_mul(count, 2u, &bytes) || info.byte_count != bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

int read_bf16_f32(
        axiom_model *model,
        const char *name,
        uint32_t expected_rank,
        uint64_t count,
        bool add_one,
        std::vector<float> *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = tensor_element_count(model, name, AXIOM_TENSOR_DTYPE_BF16, expected_rank, count);
    if (rc != AXIOM_OK) return rc;
    if (count > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) return AXIOM_ERR_BUDGET;
    std::vector<uint16_t> encoded;
    try {
        encoded.resize(static_cast<size_t>(count));
        out->resize(static_cast<size_t>(count));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    uint64_t got = 0u;
    rc = axiom_model_tensor_read(model, name, encoded.data(), count * sizeof(uint16_t), &got);
    if (rc != AXIOM_OK || got != count * sizeof(uint16_t)) return rc == AXIOM_OK ? AXIOM_ERR_IO : rc;
    for (uint64_t i = 0u; i < count; ++i) {
        union { uint32_t u; float f; } value{};
        value.u = static_cast<uint32_t>(encoded[static_cast<size_t>(i)]) << 16u;
        (*out)[static_cast<size_t>(i)] = add_one ? 1.0f + value.f : value.f;
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
    std::vector<float> values;
    int rc = read_bf16_f32(model, name, rank, count, add_one, &values);
    if (rc != AXIOM_OK) return rc;
    uint64_t bytes = 0u;
    if (!checked_mul(count, sizeof(float), &bytes)) return AXIOM_ERR_BUDGET;
    axiom_device_buffer *buffer = nullptr;
    rc = device_buffer_create(runtime, bytes, &buffer);
    if (rc == AXIOM_OK) rc = axiom_device_buffer_upload(buffer, 0u, values.data(), bytes);
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(buffer);
        return rc;
    }
    *out = buffer;
    return AXIOM_OK;
}

__device__ __forceinline__ float qwen38_gdn_round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

#include "axiom_qwen38_gdn_register.cuh"

/* Input RMSNorm is a public BF16 boundary in the target graph.  Materialize
 * that boundary in the norm kernel itself rather than enqueueing a second
 * elementwise pass. */
__global__ void qwen38_rmsnorm8_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch) return;
    const float *x = input + static_cast<uint64_t>(column) * kHidden;
    float *y = out + static_cast<uint64_t>(column) * kHidden;
    static_assert(kHidden % kThreads == 0u, "RMSNorm requires complete per-thread chunks");
    static_assert(kThreads == 256u, "Exact reduction requires the existing 256-thread block");
    constexpr uint32_t kValuesPerThread = kHidden / kThreads;
    // Unrolled constant indices allow 20 FP32 input/merged values to stay in
    // registers across normalization; no extra BF16 rounding is introduced.
    float values[kValuesPerThread];
    __shared__ float sums[kThreads];
    float sum = 0.0f;
#pragma unroll
    for (uint32_t slot = 0u; slot < kValuesPerThread; ++slot) {
        const uint32_t i = tid + slot * kThreads;
        values[slot] = x[i];
        sum = fmaf(values[slot], values[slot], sum);
    }
    sums[tid] = sum;
    __syncthreads();
    // Preserve the descending FP32 tree: shared 128/64/32, warp 16..1.
    for (uint32_t stride = kThreads / 2u; stride >= 32u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (tid < 32u) {
        float total = sums[tid];
#pragma unroll
        for (uint32_t stride = 16u; stride != 0u; stride >>= 1u) {
            const float other = __shfl_down_sync(0xffffffffu, total, stride);
            if (tid < stride) total = __fadd_rn(total, other);
        }
        if (tid == 0u) sums[0] = total;
    }
    __syncthreads();
    const float inv = rsqrtf(sums[0] / static_cast<float>(kHidden) + kEps);
#pragma unroll
    for (uint32_t slot = 0u; slot < kValuesPerThread; ++slot) {
        const uint32_t i = tid + slot * kThreads;
        y[i] = qwen38_gdn_round_bf16(values[slot] * inv * weight[i]);
    }
}

/* All public Qwen3.8 activations are BF16.  Axiom keeps scratch buffers F32
 * for the native kernels, so materialize only the externally-visible module
 * boundaries before the next quantized/BF16 operation consumes them.  The GDN
 * recurrent state intentionally remains F32: it is an internal accumulator,
 * not a decoder activation tensor. */
__global__ void qwen38_round_bf16_inplace_kernel(float *__restrict__ values, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = qwen38_gdn_round_bf16(values[index]);
    }
}

int round_bf16_inplace(float *values, uint64_t count, cudaStream_t stream = nullptr) {
    if (!values || count == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t grid = (count + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    qwen38_round_bf16_inplace_kernel<<<static_cast<uint32_t>(grid), kThreads, 0, stream>>>(
            values, count);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* The grouped FP8 qkv+z projection already materializes its BF16 boundary.
 * The two BF16 linears still produce F32 scratch, so round only a and b. */
__global__ void qwen38_gdn_round_ab8_kernel(
        float *__restrict__ a,
        float *__restrict__ b) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t ab_count = static_cast<uint64_t>(kValueHeads) * kBatch;
    if (index < ab_count) {
        a[index] = qwen38_gdn_round_bf16(a[index]);
    } else if (index < 2u * ab_count) {
        const uint64_t offset = index - ab_count;
        b[offset] = qwen38_gdn_round_bf16(b[offset]);
    }
}

int round_ab8(
        float *a,
        float *b,
        cudaStream_t stream) {
    if (!a || !b) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t count = 2u * static_cast<uint64_t>(kValueHeads) * kBatch;
    const uint64_t grid = (count + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    qwen38_gdn_round_ab8_kernel<<<static_cast<uint32_t>(grid), kThreads, 0, stream>>>(a, b);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

/* Compatibility materialization for the deliberately slow generic-runtime
 * path.  The native and temporal paths consume the grouped row stride in
 * place and never execute this copy. */
__global__ void qwen38_gdn_unpack_projection8_kernel(
        const float *__restrict__ projection,
        float *__restrict__ qkv,
        float *__restrict__ z) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kProjectionDim) * kBatch;
    if (index >= count) return;
    const uint32_t column = static_cast<uint32_t>(index / kProjectionDim);
    const uint32_t row = static_cast<uint32_t>(index % kProjectionDim);
    if (row < kConvDim) {
        qkv[static_cast<uint64_t>(column) * kConvDim + row] = projection[index];
    } else {
        z[static_cast<uint64_t>(column) * kValueDim + row - kConvDim] = projection[index];
    }
}

int unpack_projection8(
        const float *projection,
        float *qkv,
        float *z,
        cudaStream_t stream) {
    if (!projection || !qkv || !z) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t count = static_cast<uint64_t>(kProjectionDim) * kBatch;
    const uint64_t grid = (count + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
    qwen38_gdn_unpack_projection8_kernel<<<
            static_cast<uint32_t>(grid), kThreads, 0, stream>>>(projection, qkv, z);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__global__ void qwen38_gdn_gates_kernel(
        const float *__restrict__ a,
        const float *__restrict__ b,
        const float *__restrict__ a_log,
        const float *__restrict__ dt_bias,
        float *__restrict__ g,
        float *__restrict__ beta) {
    const uint32_t head = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (head >= kValueHeads || column >= kBatch) return;
    const uint64_t index = static_cast<uint64_t>(column) * kValueHeads + head;
    const float av = a[index] + dt_bias[head];
    const float softplus = av > 20.0f ? av : log1pf(expf(av));
    g[index] = -expf(a_log[head]) * softplus;
    beta[index] = 1.0f / (1.0f + expf(-b[index]));
}

__global__ void qwen38_gdn_conv1d_silu8_kernel(
        const float *__restrict__ grouped_projection,
        const float *__restrict__ weight,
        float *__restrict__ ring,
        float *__restrict__ out) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t column = blockIdx.y;
    if (channel >= kConvDim || column >= kBatch) return;
    const uint64_t projection_index =
            static_cast<uint64_t>(column) * kProjectionDim + channel;
    const uint64_t output_index = static_cast<uint64_t>(column) * kConvDim + channel;
    const uint64_t ring_index =
            (static_cast<uint64_t>(column) * kConvDim + channel) * 3u;
    const uint64_t weight_index = static_cast<uint64_t>(channel) * 4u;
    const float value = grouped_projection[projection_index];
    const float acc = weight[weight_index] * ring[ring_index] +
            weight[weight_index + 1u] * ring[ring_index + 1u] +
            weight[weight_index + 2u] * ring[ring_index + 2u] +
            weight[weight_index + 3u] * value;
    out[output_index] = qwen38_gdn_round_bf16(acc / (1.0f + __expf(-acc)));
    ring[ring_index] = ring[ring_index + 1u];
    ring[ring_index + 1u] = ring[ring_index + 2u];
    ring[ring_index + 2u] = value;
}

__global__ void qwen38_gdn_recurrence8_kernel(
        const float *__restrict__ qkv,
        const float *__restrict__ g,
        const float *__restrict__ beta,
        float *__restrict__ state,
        float *__restrict__ out) {
    const uint32_t head = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t dim = threadIdx.x;
    if (head >= kValueHeads || column >= kBatch || dim >= kHeadDim) return;
    const uint32_t key_head = head / (kValueHeads / kKeyHeads);
    const uint64_t qkv_base = static_cast<uint64_t>(column) * kConvDim;
    const float *q = qkv + qkv_base + static_cast<uint64_t>(key_head) * kHeadDim;
    const float *k = qkv + qkv_base + kKeyDim + static_cast<uint64_t>(key_head) * kHeadDim;
    const float *v = qkv + qkv_base + 2u * kKeyDim + static_cast<uint64_t>(head) * kHeadDim;
    const uint64_t state_stride =
            static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
    float *head_state = state + static_cast<uint64_t>(column) * state_stride +
            static_cast<uint64_t>(head) * kHeadDim * kHeadDim;
    extern __shared__ float scratch[];
    float *qq = scratch;
    float *kk = scratch + kHeadDim;
    float *sums = scratch + 2u * kHeadDim;
    const float q_value = q[dim];
    const float k_value = k[dim];
    const float v_value = v[dim];
    sums[dim] = q_value * q_value;
    __syncthreads();
    for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
        if (dim < stride) sums[dim] += sums[dim + stride];
        __syncthreads();
    }
    const float inverse_q = rsqrtf(sums[0] + 1.0e-6f);
    __syncthreads();
    sums[dim] = k_value * k_value;
    __syncthreads();
    for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
        if (dim < stride) sums[dim] += sums[dim + stride];
        __syncthreads();
    }
    const float inverse_k = rsqrtf(sums[0] + 1.0e-6f);
    __syncthreads();
    qq[dim] = q_value * inverse_q * rsqrtf(static_cast<float>(kHeadDim));
    kk[dim] = k_value * inverse_k;
    __syncthreads();
    const uint64_t gate_index = static_cast<uint64_t>(column) * kValueHeads + head;
    const float decay = __expf(g[gate_index]);
    const float update = beta[gate_index];
    for (uint32_t row = 0u; row < kHeadDim; ++row) {
        head_state[static_cast<uint64_t>(row) * kHeadDim + dim] *= decay;
    }
    float projected = 0.0f;
    for (uint32_t row = 0u; row < kHeadDim; ++row) {
        projected += head_state[static_cast<uint64_t>(row) * kHeadDim + dim] * kk[row];
    }
    const float delta = (v_value - projected) * update;
    for (uint32_t row = 0u; row < kHeadDim; ++row) {
        head_state[static_cast<uint64_t>(row) * kHeadDim + dim] += kk[row] * delta;
    }
    float value = 0.0f;
    for (uint32_t row = 0u; row < kHeadDim; ++row) {
        value += head_state[static_cast<uint64_t>(row) * kHeadDim + dim] * qq[row];
    }
    out[(static_cast<uint64_t>(column) * kValueHeads + head) * kHeadDim + dim] =
            qwen38_gdn_round_bf16(value);
}

__global__ void qwen38_gdn_gated_norm8_kernel(
        const float *__restrict__ recurrent,
        const float *__restrict__ weight,
        const float *__restrict__ grouped_projection,
        float *__restrict__ out) {
    const uint32_t head = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t dim = threadIdx.x;
    if (head >= kValueHeads || column >= kBatch || dim >= kHeadDim) return;
    const uint64_t base =
            (static_cast<uint64_t>(column) * kValueHeads + head) * kHeadDim;
    extern __shared__ float sums[];
    const float recurrent_value = recurrent[base + dim];
    sums[dim] = recurrent_value * recurrent_value;
    __syncthreads();
    for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
        if (dim < stride) sums[dim] += sums[dim + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / static_cast<float>(kHeadDim) + kEps);
    const uint64_t z_index = static_cast<uint64_t>(column) * kProjectionDim +
            kConvDim + static_cast<uint64_t>(head) * kHeadDim + dim;
    const float gate = grouped_projection[z_index];
    out[base + dim] = qwen38_gdn_round_bf16(recurrent_value * inverse * weight[dim] *
            (gate / (1.0f + __expf(-gate))));
}

__global__ void qwen38_gdn_conv1d_silu_temporal8_kernel(
        const float *__restrict__ grouped_projection,
        const float *__restrict__ weight,
        const float *__restrict__ ring,
        float *__restrict__ final_ring,
        float *__restrict__ out) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kConvDim) return;
    float ring0 = ring[static_cast<uint64_t>(channel) * 3u];
    float ring1 = ring[static_cast<uint64_t>(channel) * 3u + 1u];
    float ring2 = ring[static_cast<uint64_t>(channel) * 3u + 2u];
    const uint64_t weight_index = static_cast<uint64_t>(channel) * 4u;
    for (uint32_t token = 0u; token < kBatch; ++token) {
        const uint64_t projection_index =
                static_cast<uint64_t>(token) * kProjectionDim + channel;
        const uint64_t output_index = static_cast<uint64_t>(token) * kConvDim + channel;
        const float value = grouped_projection[projection_index];
        const float acc = weight[weight_index] * ring0 +
                weight[weight_index + 1u] * ring1 +
                weight[weight_index + 2u] * ring2 +
                weight[weight_index + 3u] * value;
        out[output_index] = qwen38_gdn_round_bf16(acc / (1.0f + __expf(-acc)));
        ring0 = ring1;
        ring1 = ring2;
        ring2 = value;
    }
    final_ring[static_cast<uint64_t>(channel) * 3u] = ring0;
    final_ring[static_cast<uint64_t>(channel) * 3u + 1u] = ring1;
    final_ring[static_cast<uint64_t>(channel) * 3u + 2u] = ring2;
}

__global__ void qwen38_gdn_normalize_qk_temporal8_kernel(
        float *__restrict__ qkv) {
    const uint32_t token = blockIdx.x;
    const uint32_t key_head = blockIdx.y;
    const uint32_t dim = threadIdx.x;
    if (token >= kBatch || key_head >= kKeyHeads || dim >= kHeadDim) return;
    const uint64_t qkv_base = static_cast<uint64_t>(token) * kConvDim;
    float *q = qkv + qkv_base + static_cast<uint64_t>(key_head) * kHeadDim;
    float *k = qkv + qkv_base + kKeyDim +
            static_cast<uint64_t>(key_head) * kHeadDim;
    extern __shared__ float sums[];
    float *q_sums = sums;
    float *k_sums = sums + kHeadDim;
    const float q_value = q[dim];
    const float k_value = k[dim];
    q_sums[dim] = q_value * q_value;
    k_sums[dim] = k_value * k_value;
    __syncthreads();
    for (uint32_t stride = kHeadDim >> 1u; stride != 0u; stride >>= 1u) {
        if (dim < stride) {
            q_sums[dim] += q_sums[dim + stride];
            k_sums[dim] += k_sums[dim + stride];
        }
        __syncthreads();
    }
    const float inverse_q = rsqrtf(q_sums[0] + 1.0e-6f);
    const float inverse_k = rsqrtf(k_sums[0] + 1.0e-6f);
    q[dim] = q_value * inverse_q * rsqrtf(static_cast<float>(kHeadDim));
    k[dim] = k_value * inverse_k;
}

/* FlashInfer's MTP recurrence keeps a small V tile resident while advancing
 * every temporal row.  This native variant uses shared memory for the tile so
 * each output column can retain Axiom's original row-ordered FP32 reductions.
 * It reads the initial state once and writes each committable prefix once. */
template <uint32_t ValueTile>
__global__ void qwen38_gdn_recurrence_temporal8_kernel(
        const float *__restrict__ normalized_qkv,
        const float *__restrict__ g,
        const float *__restrict__ beta,
        const float *__restrict__ state,
        float *__restrict__ final_state,
        float *__restrict__ out,
        float *__restrict__ delta_history,
        float *__restrict__ decay_history) {
    static_assert(ValueTile != 0u && kHeadDim % ValueTile == 0u,
                  "Temporal GDN value tile must divide the head dimension");
    const uint32_t value_tile = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t row = threadIdx.x;
    if (value_tile >= kHeadDim / ValueTile ||
        head >= kValueHeads || row >= kHeadDim) {
        return;
    }
    const uint32_t value_start = value_tile * ValueTile;
    const uint32_t key_head = head / (kValueHeads / kKeyHeads);
    extern __shared__ float scratch[];
    constexpr uint32_t kStateTileElements = kHeadDim * ValueTile;
    float *state_tile = scratch;
    float *qq = state_tile + kStateTileElements;
    float *kk = qq + kHeadDim;
    float *delta = kk + kHeadDim;

#pragma unroll
    for (uint32_t flat = row; flat < kStateTileElements; flat += kHeadDim) {
        const uint32_t state_row = flat / ValueTile;
        const uint32_t local_value = flat % ValueTile;
        state_tile[flat] = state[
                (static_cast<uint64_t>(head) * kHeadDim + state_row) * kHeadDim +
                value_start + local_value];
    }
    __syncthreads();

    for (uint32_t token = 0u; token < kBatch; ++token) {
        const uint64_t qkv_base = static_cast<uint64_t>(token) * kConvDim;
        const float *q = normalized_qkv + qkv_base +
                static_cast<uint64_t>(key_head) * kHeadDim;
        const float *k = normalized_qkv + qkv_base + kKeyDim +
                static_cast<uint64_t>(key_head) * kHeadDim;
        const float *v = normalized_qkv + qkv_base + 2u * kKeyDim +
                static_cast<uint64_t>(head) * kHeadDim + value_start;
        qq[row] = q[row];
        kk[row] = k[row];
        __syncthreads();

        const uint64_t gate_index = static_cast<uint64_t>(token) * kValueHeads + head;
        const float decay = __expf(g[gate_index]);
        const float update = beta[gate_index];
        if (value_tile == 0u && row == 0u) {
            decay_history[gate_index] = decay;
        }

#pragma unroll
        for (uint32_t flat = row; flat < kStateTileElements; flat += kHeadDim) {
            state_tile[flat] *= decay;
        }
        __syncthreads();

        if (row < ValueTile) {
            float projected = 0.0f;
#pragma unroll 8
            for (uint32_t state_row = 0u; state_row < kHeadDim; ++state_row) {
                projected = fmaf(
                        state_tile[state_row * ValueTile + row],
                        kk[state_row], projected);
            }
            delta[row] = (v[row] - projected) * update;
            delta_history[(static_cast<uint64_t>(token) * kValueHeads + head) *
                    kHeadDim + value_start + row] = delta[row];
        }
        __syncthreads();

#pragma unroll
        for (uint32_t flat = row; flat < kStateTileElements; flat += kHeadDim) {
            const uint32_t state_row = flat / ValueTile;
            const uint32_t local_value = flat % ValueTile;
            const float updated = fmaf(kk[state_row], delta[local_value], state_tile[flat]);
            state_tile[flat] = updated;
        }
        __syncthreads();

        if (row < ValueTile) {
            float value = 0.0f;
#pragma unroll 8
            for (uint32_t state_row = 0u; state_row < kHeadDim; ++state_row) {
                value = fmaf(
                        state_tile[state_row * ValueTile + row],
                        qq[state_row], value);
            }
            const uint32_t value_dim = value_start + row;
            out[(static_cast<uint64_t>(token) * kValueHeads + head) * kHeadDim +
                    value_dim] = qwen38_gdn_round_bf16(value);
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t flat = row; flat < kStateTileElements; flat += kHeadDim) {
        const uint32_t state_row = flat / ValueTile;
        const uint32_t local_value = flat % ValueTile;
        final_state[(static_cast<uint64_t>(head) * kHeadDim + state_row) * kHeadDim +
                value_start + local_value] = state_tile[flat];
    }
}

template <uint32_t ValueTile>
int launch_gdn_recurrence_temporal8(
        const float *normalized_qkv,
        const float *g,
        const float *beta,
        const float *state,
        float *final_state,
        float *out,
        float *delta_history,
        float *decay_history,
        cudaStream_t stream) {
    constexpr size_t kSharedFloats =
            static_cast<size_t>(kHeadDim) * ValueTile + 2u * kHeadDim + ValueTile;
    qwen38_gdn_recurrence_temporal8_kernel<ValueTile><<<
            dim3(kHeadDim / ValueTile, kValueHeads), kHeadDim,
            kSharedFloats * sizeof(float), stream>>>(
            normalized_qkv, g, beta, state, final_state, out,
            delta_history, decay_history);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__device__ __forceinline__ void qwen38_gdn_materialize_prefix_index(
        const uint64_t index,
        float *__restrict__ ring,
        float *__restrict__ state,
        const float *__restrict__ grouped_projection,
        const float *__restrict__ normalized_qkv,
        const float *__restrict__ delta_history,
        const float *__restrict__ decay_history,
        const uint32_t consumed) {
    if (index < kConvDim) {
        if (consumed == kBatch) {
            const uint64_t final_base =
                    static_cast<uint64_t>(kBatch - 1u) * kConvDim * 3u + index * 3u;
            ring[index * 3u] = ring[final_base];
            ring[index * 3u + 1u] = ring[final_base + 1u];
            ring[index * 3u + 2u] = ring[final_base + 2u];
        } else {
        float ring0 = ring[index * 3u];
        float ring1 = ring[index * 3u + 1u];
        float ring2 = ring[index * 3u + 2u];
        for (uint32_t token = 0u; token < consumed; ++token) {
            ring0 = ring1;
            ring1 = ring2;
            ring2 = grouped_projection[
                    static_cast<uint64_t>(token) * kProjectionDim + index];
        }
        ring[index * 3u] = ring0;
        ring[index * 3u + 1u] = ring1;
        ring[index * 3u + 2u] = ring2;
        }
    }
    const uint64_t state_count =
            static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
    if (index < state_count) {
        if (consumed == kBatch) {
            state[index] = state[static_cast<uint64_t>(kBatch - 1u) * state_count + index];
            return;
        }
        const uint32_t head = static_cast<uint32_t>(
                index / (static_cast<uint64_t>(kHeadDim) * kHeadDim));
        const uint32_t state_row = static_cast<uint32_t>(
                (index / kHeadDim) % kHeadDim);
        const uint32_t value_dim = static_cast<uint32_t>(index % kHeadDim);
        const uint32_t key_head = head / (kValueHeads / kKeyHeads);
        float value = state[index];
        for (uint32_t token = 0u; token < consumed; ++token) {
            const float key = normalized_qkv[
                    static_cast<uint64_t>(token) * kConvDim + kKeyDim +
                    static_cast<uint64_t>(key_head) * kHeadDim + state_row];
            const float delta = delta_history[
                    (static_cast<uint64_t>(token) * kValueHeads + head) *
                    kHeadDim + value_dim];
            const float decay = decay_history[
                    static_cast<uint64_t>(token) * kValueHeads + head];
            value *= decay;
            value = fmaf(key, delta, value);
        }
        state[index] = value;
    }
}

__global__ void qwen38_gdn_materialize_prefix_kernel(
        float *__restrict__ ring,
        float *__restrict__ state,
        const float *__restrict__ grouped_projection,
        const float *__restrict__ normalized_qkv,
        const float *__restrict__ delta_history,
        const float *__restrict__ decay_history,
        const uint32_t consumed) {
    if (consumed == 0u || consumed > kBatch) return;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    qwen38_gdn_materialize_prefix_index(
            index, ring, state, grouped_projection, normalized_qkv,
            delta_history, decay_history, consumed);
}

/* One graph node handles commit and fail-closed abort.  The temporal forward
 * leaves durable state untouched, so an invalid async status needs no copy. */
__global__ void qwen38_gdn_materialize_prefix_device_kernel(
        float *__restrict__ ring,
        float *__restrict__ state,
        const float *__restrict__ grouped_projection,
        const float *__restrict__ normalized_qkv,
        const float *__restrict__ delta_history,
        const float *__restrict__ decay_history,
        const uint32_t *__restrict__ consumed_tokens,
        const uint32_t *__restrict__ async_status) {
    const uint32_t consumed = consumed_tokens[0];
    if ((async_status && async_status[0] != 0u) || consumed == 0u || consumed > kBatch) return;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    qwen38_gdn_materialize_prefix_index(
            index, ring, state, grouped_projection, normalized_qkv,
            delta_history, decay_history, consumed);
}

}  // namespace

struct axiom_qwen38_gdn_layer {
    axiom_runtime *runtime = nullptr;
    int device = -1;
    uint32_t layer = 0u;
    uint32_t temporal_value_tile = kDefaultTemporalValueTile;
    bool temporal_register_state = false;
    bool bf16_pair = true;
    bool parallel_projection_streams = true;
    cudaStream_t ab_projection_stream = nullptr;
    cudaEvent_t projection_input_ready = nullptr;
    cudaEvent_t ab_projection_done = nullptr;
    uint64_t device_bytes = 0u;

    axiom_qwen38_fp8_projection_group *qkv_z = nullptr;
    axiom_qwen38_fp8_linear *out_proj = nullptr;
    axiom_qwen38_bf16_linear *a = nullptr;
    axiom_qwen38_bf16_linear *b = nullptr;

    axiom_device_buffer *input_norm_weight = nullptr;
    axiom_device_buffer *conv_weight = nullptr;
    axiom_device_buffer *a_log = nullptr;
    axiom_device_buffer *dt_bias = nullptr;
    axiom_device_buffer *ssm_norm_weight = nullptr;

    axiom_device_buffer *ring = nullptr;
    axiom_device_buffer *state = nullptr;
    /* Compact transactional journal: one delta vector and decay scalar per
     * temporal token/head. Durable ring/state remain untouched until the
     * accepted prefix is known. */
    axiom_device_buffer *spec_delta = nullptr;
    axiom_device_buffer *spec_decay = nullptr;
    axiom_device_buffer *norm = nullptr;
    axiom_device_buffer *projection_out = nullptr;
    /* Used only by AXIOM_QWEN38_GDN_BATCHED=0 compatibility execution. */
    axiom_device_buffer *qkv_out = nullptr;
    axiom_device_buffer *z_out = nullptr;
    axiom_device_buffer *a_out = nullptr;
    axiom_device_buffer *b_out = nullptr;
    axiom_device_buffer *g = nullptr;
    axiom_device_buffer *beta = nullptr;
    axiom_device_buffer *conv = nullptr;
    axiom_device_buffer *recurrent = nullptr;
    axiom_device_buffer *gated = nullptr;
    bool spec_active = false;
    bool spec_forwarded = false;
    cudaStream_t spec_stream = nullptr;
};

namespace {

int qwen38_gdn_project_inputs(
        axiom_qwen38_gdn_layer *layer,
        const float *input,
        float *projection,
        float *a,
        float *b,
        void *stream) {
    if (!layer || !input || !projection || !a || !b) return AXIOM_ERR_INVALID_ARGUMENT;
    const cudaStream_t main_stream = static_cast<cudaStream_t>(stream);
    int rc = AXIOM_OK;
    if (layer->parallel_projection_streams) {
        rc = cuda_status(cudaEventRecord(layer->projection_input_ready, main_stream));
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                layer->ab_projection_stream, layer->projection_input_ready, 0u));
        if (rc == AXIOM_OK) rc = axiom_qwen38_fp8_projection_group_forward_f32_device(
                layer->qkv_z, input, projection, stream);
        void *ab_stream = reinterpret_cast<void *>(layer->ab_projection_stream);
        if (rc == AXIOM_OK && layer->bf16_pair) {
            rc = axiom_qwen38_bf16_linear_pair_forward_f32_device(
                    layer->a, layer->b, input, a, b, ab_stream);
        } else if (rc == AXIOM_OK) {
            rc = axiom_qwen38_bf16_linear_forward_f32_device(
                    layer->a, input, a, ab_stream);
            if (rc == AXIOM_OK) rc = axiom_qwen38_bf16_linear_forward_f32_device(
                    layer->b, input, b, ab_stream);
        }
        if (rc == AXIOM_OK) rc = cuda_status(cudaEventRecord(
                layer->ab_projection_done, layer->ab_projection_stream));
        if (rc == AXIOM_OK) rc = cuda_status(cudaStreamWaitEvent(
                main_stream, layer->ab_projection_done, 0u));
        return rc;
    }
    rc = axiom_qwen38_fp8_projection_group_forward_f32_device(
            layer->qkv_z, input, projection, stream);
    if (rc == AXIOM_OK && layer->bf16_pair) {
        rc = axiom_qwen38_bf16_linear_pair_forward_f32_device(
                layer->a, layer->b, input, a, b, stream);
    } else if (rc == AXIOM_OK) {
        rc = axiom_qwen38_bf16_linear_forward_f32_device(layer->a, input, a, stream);
        if (rc == AXIOM_OK) rc = axiom_qwen38_bf16_linear_forward_f32_device(
                layer->b, input, b, stream);
    }
    return rc;
}

}  // namespace

extern "C" void axiom_qwen38_gdn_layer_destroy(axiom_qwen38_gdn_layer *layer) {
    if (!layer) return;
    if (layer->device >= 0) (void)cudaSetDevice(layer->device);
    if (layer->ab_projection_stream) (void)cudaStreamDestroy(layer->ab_projection_stream);
    if (layer->ab_projection_done) (void)cudaEventDestroy(layer->ab_projection_done);
    if (layer->projection_input_ready) (void)cudaEventDestroy(layer->projection_input_ready);
    axiom_qwen38_fp8_linear_destroy(layer->out_proj);
    axiom_qwen38_fp8_projection_group_destroy(layer->qkv_z);
    axiom_qwen38_bf16_linear_destroy(layer->b);
    axiom_qwen38_bf16_linear_destroy(layer->a);
    axiom_device_buffer_destroy(layer->gated);
    axiom_device_buffer_destroy(layer->recurrent);
    axiom_device_buffer_destroy(layer->conv);
    axiom_device_buffer_destroy(layer->beta);
    axiom_device_buffer_destroy(layer->g);
    axiom_device_buffer_destroy(layer->b_out);
    axiom_device_buffer_destroy(layer->a_out);
    axiom_device_buffer_destroy(layer->z_out);
    axiom_device_buffer_destroy(layer->qkv_out);
    axiom_device_buffer_destroy(layer->projection_out);
    axiom_device_buffer_destroy(layer->norm);
    axiom_device_buffer_destroy(layer->spec_decay);
    axiom_device_buffer_destroy(layer->spec_delta);
    axiom_device_buffer_destroy(layer->state);
    axiom_device_buffer_destroy(layer->ring);
    axiom_device_buffer_destroy(layer->ssm_norm_weight);
    axiom_device_buffer_destroy(layer->dt_bias);
    axiom_device_buffer_destroy(layer->a_log);
    axiom_device_buffer_destroy(layer->conv_weight);
    axiom_device_buffer_destroy(layer->input_norm_weight);
    delete layer;
}

extern "C" int axiom_qwen38_gdn_layer_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        uint32_t layer_index,
        axiom_qwen38_gdn_layer **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || !out || device < 0 || (layer_index % 4u) == 3u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_qwen38_gdn_layer *layer = new (std::nothrow) axiom_qwen38_gdn_layer();
    if (!layer) return AXIOM_ERR_BUDGET;
    layer->runtime = runtime;
    layer->device = device;
    layer->layer = layer_index;
    layer->temporal_value_tile = temporal_value_tile_from_env();
    // Capture the exact opt-in once per layer, keeping CUDA graph dispatch stable.
    const char *register_state = std::getenv("AXIOM_QWEN38_GDN_REGISTER_STATE");
    layer->temporal_register_state =
            register_state && register_state[0] == '1' && register_state[1] == '\0';
    layer->bf16_pair = !env_disabled("AXIOM_QWEN38_GDN_BF16_PAIR");
    layer->parallel_projection_streams =
            !env_disabled("AXIOM_QWEN38_PARALLEL_PROJECTIONS");
    if (layer->parallel_projection_streams) {
        cudaError_t status = cudaStreamCreateWithFlags(
                &layer->ab_projection_stream, cudaStreamNonBlocking);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(
                &layer->projection_input_ready, cudaEventDisableTiming);
        if (status == cudaSuccess) status = cudaEventCreateWithFlags(
                &layer->ab_projection_done, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            const int create_rc = cuda_status(status);
            axiom_qwen38_gdn_layer_destroy(layer);
            return create_rc;
        }
    }

    char prefix[128]{};
    const int prefix_len = std::snprintf(
            prefix, sizeof(prefix), "model.language_model.layers.%u.linear_attn.", layer_index);
    if (prefix_len <= 0 || static_cast<size_t>(prefix_len) >= sizeof(prefix)) {
        axiom_qwen38_gdn_layer_destroy(layer);
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    char name[192]{};
    auto full_name = [&](const char *suffix) -> const char * {
        const int written = std::snprintf(name, sizeof(name), "%s%s", prefix, suffix);
        return written > 0 && static_cast<size_t>(written) < sizeof(name) ? name : nullptr;
    };
    auto load_fp8 = [&](const char *base_suffix,
                        axiom_qwen38_fp8_linear **target) -> int {
        const char *base = full_name(base_suffix);
        if (!base) return AXIOM_ERR_INVALID_ARGUMENT;
        char base_copy[192]{};
        const int written = std::snprintf(base_copy, sizeof(base_copy), "%s", base);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(base_copy)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        return axiom_qwen38_fp8_linear_load_auto(model, device, base_copy, target);
    };
    auto load_qkv_z_group = [&]() -> int {
        char qkv_base[192]{};
        char z_base[192]{};
        const int qkv_written = std::snprintf(
                qkv_base, sizeof(qkv_base), "%sin_proj_qkv", prefix);
        const int z_written = std::snprintf(
                z_base, sizeof(z_base), "%sin_proj_z", prefix);
        if (qkv_written <= 0 || static_cast<size_t>(qkv_written) >= sizeof(qkv_base) ||
            z_written <= 0 || static_cast<size_t>(z_written) >= sizeof(z_base)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        const char *bases[2] = {qkv_base, z_base};
        return axiom_qwen38_fp8_projection_group_load_auto(
                model, device, bases, 2u, &layer->qkv_z);
    };
    auto load_bf16 = [&](const char *suffix, axiom_qwen38_bf16_linear **target) -> int {
        const char *tensor = full_name(suffix);
        return tensor ? axiom_qwen38_bf16_linear_load(model, device, tensor, target)
                      : AXIOM_ERR_INVALID_ARGUMENT;
    };
    auto load_f32 = [&](const char *suffix, uint32_t rank, uint64_t count, bool add_one,
                        axiom_device_buffer **target) -> int {
        const char *tensor = full_name(suffix);
        return tensor ? upload_bf16_as_f32(model, runtime, tensor, rank, count, add_one, target)
                      : AXIOM_ERR_INVALID_ARGUMENT;
    };

    int rc = AXIOM_OK;
    {
        char norm_name[160]{};
        const int written = std::snprintf(
                norm_name, sizeof(norm_name),
                "model.language_model.layers.%u.input_layernorm.weight", layer_index);
        rc = written > 0 && static_cast<size_t>(written) < sizeof(norm_name)
                ? upload_bf16_as_f32(model, runtime, norm_name, 1u, kHidden, true,
                                     &layer->input_norm_weight)
                : AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK) rc = load_qkv_z_group();
    if (rc == AXIOM_OK) rc = load_bf16("in_proj_a.weight", &layer->a);
    if (rc == AXIOM_OK) rc = load_bf16("in_proj_b.weight", &layer->b);
    if (rc == AXIOM_OK) rc = load_fp8("out_proj", &layer->out_proj);
    if (rc == AXIOM_OK) rc = load_f32("A_log", 1u, kValueHeads, false, &layer->a_log);
    if (rc == AXIOM_OK) rc = load_f32("dt_bias", 1u, kValueHeads, false, &layer->dt_bias);
    if (rc == AXIOM_OK) rc = load_f32("conv1d.weight", 3u, static_cast<uint64_t>(kConvDim) * 4u,
                                      false, &layer->conv_weight);
    if (rc == AXIOM_OK) rc = load_f32("norm.weight", 1u, kHeadDim, false, &layer->ssm_norm_weight);
    axiom_qwen38_fp8_projection_group_info projection_info{};
    projection_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_fp8_projection_group_info_get(
                layer->qkv_z, &projection_info);
    }
    if (rc != AXIOM_OK || projection_info.projection_count != 2u ||
        projection_info.input_features != kHidden ||
        projection_info.total_output_features != kProjectionDim ||
        projection_info.row_offsets[0] != 0u ||
        projection_info.row_offsets[1] != kConvDim ||
        projection_info.row_offsets[2] != kProjectionDim ||
        projection_info.bf16_output_boundary != 1u ||
        axiom_qwen38_fp8_linear_rows(layer->out_proj) != kHidden ||
        axiom_qwen38_fp8_linear_cols(layer->out_proj) != kValueDim ||
        axiom_qwen38_bf16_linear_rows(layer->a) != kValueHeads ||
        axiom_qwen38_bf16_linear_cols(layer->a) != kHidden ||
        axiom_qwen38_bf16_linear_rows(layer->b) != kValueHeads ||
        axiom_qwen38_bf16_linear_cols(layer->b) != kHidden) {
        axiom_qwen38_gdn_layer_destroy(layer);
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    if (projection_info.tensor_core_enabled != 1u) {
        axiom_qwen38_gdn_layer_destroy(layer);
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }

    const uint64_t hidden_batch = static_cast<uint64_t>(kHidden) * kBatch * sizeof(float);
    const uint64_t conv_batch = static_cast<uint64_t>(kConvDim) * kBatch * sizeof(float);
    const uint64_t value_batch = static_cast<uint64_t>(kValueDim) * kBatch * sizeof(float);
    const uint64_t projection_batch =
            static_cast<uint64_t>(kProjectionDim) * kBatch * sizeof(float);
    const uint64_t gates_batch = static_cast<uint64_t>(kValueHeads) * kBatch * sizeof(float);
    const uint64_t ring_bytes = static_cast<uint64_t>(kConvDim) * 3u * kBatch * sizeof(float);
    const uint64_t state_bytes = static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim * kBatch * sizeof(float);
    const uint64_t spec_delta_bytes =
            static_cast<uint64_t>(kBatch) * kValueHeads * kHeadDim * sizeof(float);
    const uint64_t spec_decay_bytes =
            static_cast<uint64_t>(kBatch) * kValueHeads * sizeof(float);
    const std::pair<uint64_t, axiom_device_buffer **> allocations[] = {
        {ring_bytes, &layer->ring}, {state_bytes, &layer->state},
        {spec_delta_bytes, &layer->spec_delta},
        {spec_decay_bytes, &layer->spec_decay},
        {hidden_batch, &layer->norm}, {projection_batch, &layer->projection_out},
        {conv_batch, &layer->qkv_out},
        {value_batch, &layer->z_out}, {gates_batch, &layer->a_out},
        {gates_batch, &layer->b_out}, {gates_batch, &layer->g},
        {gates_batch, &layer->beta}, {conv_batch, &layer->conv},
        {value_batch, &layer->recurrent}, {value_batch, &layer->gated},
    };
    for (const auto &allocation : allocations) {
        rc = device_buffer_create(runtime, allocation.first, allocation.second);
        if (rc != AXIOM_OK) {
            axiom_qwen38_gdn_layer_destroy(layer);
            return rc;
        }
    }
    rc = axiom_qwen38_gdn_layer_reset(layer);
    if (rc != AXIOM_OK) {
        axiom_qwen38_gdn_layer_destroy(layer);
        return rc;
    }

    uint64_t bytes = 0u;
    const uint64_t resident_bytes[] = {
        projection_info.device_bytes,
        axiom_qwen38_fp8_linear_device_bytes(layer->out_proj),
        axiom_qwen38_bf16_linear_device_bytes(layer->a),
        axiom_qwen38_bf16_linear_device_bytes(layer->b),
        static_cast<uint64_t>(kHidden) * sizeof(float),
        static_cast<uint64_t>(kConvDim) * 4u * sizeof(float),
        static_cast<uint64_t>(kValueHeads) * 2u * sizeof(float),
        static_cast<uint64_t>(kHeadDim) * sizeof(float),
        ring_bytes, state_bytes, spec_delta_bytes, spec_decay_bytes,
        hidden_batch, projection_batch, conv_batch, value_batch,
        gates_batch, gates_batch, gates_batch, gates_batch, conv_batch,
        value_batch, value_batch,
    };
    for (uint64_t amount : resident_bytes) {
        if (!checked_add(bytes, amount, &bytes)) {
            axiom_qwen38_gdn_layer_destroy(layer);
            return AXIOM_ERR_BUDGET;
        }
    }
    layer->device_bytes = bytes;
    *out = layer;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_gdn_layer_reset(axiom_qwen38_gdn_layer *layer) {
    if (!layer || !layer->ring || !layer->state || layer->device < 0) return AXIOM_ERR_INVALID_ARGUMENT;
    void *ring = nullptr;
    void *state = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t zero_ring = cudaMemset(ring, 0, static_cast<size_t>(kConvDim) * 3u * kBatch * sizeof(float));
    if (zero_ring != cudaSuccess) return cuda_status(zero_ring);
    const cudaError_t zero_state = cudaMemset(
            state, 0, static_cast<size_t>(kValueHeads) * kHeadDim * kHeadDim * kBatch * sizeof(float));
    if (zero_state == cudaSuccess) {
        layer->spec_active = false;
        layer->spec_forwarded = false;
        layer->spec_stream = nullptr;
    }
    return cuda_status(zero_state);
}

extern "C" uint64_t axiom_qwen38_gdn_layer_device_bytes(const axiom_qwen38_gdn_layer *layer) {
    return layer ? layer->device_bytes : 0u;
}

extern "C" int axiom_qwen38_gdn_layer_state_export(
        const axiom_qwen38_gdn_layer *layer,
        void *host_snapshot,
        const uint64_t host_snapshot_bytes) {
    if (!layer || !host_snapshot || host_snapshot_bytes != AXIOM_QWEN38_GDN_SNAPSHOT_BYTES ||
        !layer->ring || !layer->state || layer->device < 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *ring = nullptr;
    void *state = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t ring_status = cudaMemcpy(
            host_snapshot, ring, AXIOM_QWEN38_GDN_RING_BYTES, cudaMemcpyDeviceToHost);
    if (ring_status != cudaSuccess) return cuda_status(ring_status);
    const cudaError_t state_status = cudaMemcpy(
            static_cast<uint8_t *>(host_snapshot) + AXIOM_QWEN38_GDN_RING_BYTES,
            state, AXIOM_QWEN38_GDN_STATE_BYTES, cudaMemcpyDeviceToHost);
    return cuda_status(state_status);
}

extern "C" int axiom_qwen38_gdn_layer_state_import(
        axiom_qwen38_gdn_layer *layer,
        const void *host_snapshot,
        const uint64_t host_snapshot_bytes) {
    if (!layer || !host_snapshot || host_snapshot_bytes != AXIOM_QWEN38_GDN_SNAPSHOT_BYTES ||
        !layer->ring || !layer->state || layer->device < 0 || layer->spec_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *ring = nullptr;
    void *state = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    cudaError_t status = cudaMemcpy(
            ring, host_snapshot, AXIOM_QWEN38_GDN_RING_BYTES, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        status = cudaMemcpy(
                state,
                static_cast<const uint8_t *>(host_snapshot) + AXIOM_QWEN38_GDN_RING_BYTES,
                AXIOM_QWEN38_GDN_STATE_BYTES, cudaMemcpyHostToDevice);
    }
    /* The durable snapshot represents lane zero. Clear speculative/independent
     * lanes so a subsequent temporal transaction cannot consume stale state. */
    if (status == cudaSuccess) {
        status = cudaMemset(
                static_cast<uint8_t *>(ring) + AXIOM_QWEN38_GDN_RING_BYTES,
                0, AXIOM_QWEN38_GDN_RING_BYTES * (AXIOM_QWEN38_GDN_BATCH - 1u));
    }
    if (status == cudaSuccess) {
        status = cudaMemset(
                static_cast<uint8_t *>(state) + AXIOM_QWEN38_GDN_STATE_BYTES,
                0, AXIOM_QWEN38_GDN_STATE_BYTES * (AXIOM_QWEN38_GDN_BATCH - 1u));
    }
    return cuda_status(status);
}

extern "C" int axiom_qwen38_gdn_layer_forward_f32_device(
        axiom_qwen38_gdn_layer *layer,
        const float *input,
        float *out,
        void *stream) {
    if (!layer || !input || !out || stream != nullptr || layer->spec_active || !layer->runtime ||
        !layer->qkv_z || !layer->input_norm_weight || !layer->norm ||
        !layer->projection_out || !layer->qkv_out || !layer->z_out ||
        !layer->a_out || !layer->b_out || !layer->g || !layer->beta || !layer->conv ||
        !layer->recurrent || !layer->gated) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    void *norm_weight = nullptr;
    void *norm = nullptr;
    void *projection = nullptr;
    void *qkv = nullptr;
    void *z = nullptr;
    void *a = nullptr;
    void *b = nullptr;
    void *a_log = nullptr;
    void *dt_bias = nullptr;
    void *conv_weight = nullptr;
    void *ring = nullptr;
    void *state = nullptr;
    void *ssm_norm_weight = nullptr;
    void *conv = nullptr;
    void *recurrent = nullptr;
    void *g = nullptr;
    void *beta = nullptr;
    void *gated = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->projection_out, &projection);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->qkv_out, &qkv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->z_out, &z);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->a_out, &a);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->b_out, &b);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->a_log, &a_log);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->dt_bias, &dt_bias);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv_weight, &conv_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->ssm_norm_weight, &ssm_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->g, &g);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->beta, &beta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv, &conv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->recurrent, &recurrent);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated, &gated);
    if (rc != AXIOM_OK) return rc;

    qwen38_rmsnorm8_kernel<<<kBatch, kThreads>>>(
            static_cast<const float *>(norm_weight), input, static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_gdn_project_inputs(
            layer, static_cast<const float *>(norm), static_cast<float *>(projection),
            static_cast<float *>(a), static_cast<float *>(b), nullptr);
    if (rc != AXIOM_OK) return rc;
    rc = round_ab8(static_cast<float *>(a), static_cast<float *>(b), nullptr);
    if (rc != AXIOM_OK) return rc;

    qwen38_gdn_gates_kernel<<<dim3((kValueHeads + 127u) / 128u, kBatch), 128u>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<const float *>(a_log), static_cast<const float *>(dt_bias),
            static_cast<float *>(g), static_cast<float *>(beta));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    if (!env_disabled("AXIOM_QWEN38_GDN_BATCHED")) {
        qwen38_gdn_conv1d_silu8_kernel<<<
                dim3((kConvDim + kThreads - 1u) / kThreads, kBatch), kThreads>>>(
                static_cast<const float *>(projection), static_cast<const float *>(conv_weight),
                static_cast<float *>(ring), static_cast<float *>(conv));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        qwen38_gdn_recurrence8_kernel<<<
                dim3(kValueHeads, kBatch), kHeadDim,
                static_cast<size_t>(3u * kHeadDim) * sizeof(float)>>>(
                static_cast<const float *>(conv), static_cast<const float *>(g),
                static_cast<const float *>(beta), static_cast<float *>(state),
                static_cast<float *>(recurrent));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        qwen38_gdn_gated_norm8_kernel<<<
                dim3(kValueHeads, kBatch), kHeadDim,
                static_cast<size_t>(kHeadDim) * sizeof(float)>>>(
                static_cast<const float *>(recurrent),
                static_cast<const float *>(ssm_norm_weight),
                static_cast<const float *>(projection), static_cast<float *>(gated));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    } else {
        rc = unpack_projection8(
                static_cast<const float *>(projection), static_cast<float *>(qkv),
                static_cast<float *>(z), nullptr);
        if (rc != AXIOM_OK) return rc;
        const uint64_t conv_bytes = static_cast<uint64_t>(kConvDim) * sizeof(float);
        const uint64_t key_bytes = static_cast<uint64_t>(kKeyDim) * sizeof(float);
        const uint64_t value_bytes = static_cast<uint64_t>(kValueDim) * sizeof(float);
        const uint64_t ring_bytes = static_cast<uint64_t>(kConvDim) * 3u * sizeof(float);
        const uint64_t state_bytes =
                static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim * sizeof(float);
        const uint64_t gates_bytes = static_cast<uint64_t>(kValueHeads) * sizeof(float);
        for (uint32_t column = 0u; column < kBatch; ++column) {
            const uint64_t qkv_offset = static_cast<uint64_t>(column) * conv_bytes;
            const uint64_t gate_offset = static_cast<uint64_t>(column) * gates_bytes;
            const uint64_t ring_offset = static_cast<uint64_t>(column) * ring_bytes;
            const uint64_t state_offset = static_cast<uint64_t>(column) * state_bytes;
            const uint64_t value_offset = static_cast<uint64_t>(column) * value_bytes;
            rc = axiom_runtime_deltanet_conv1d_silu_f32_device(
                    layer->runtime, layer->qkv_out, qkv_offset, layer->conv_weight, 0u,
                    layer->ring, ring_offset, layer->conv, qkv_offset, kConvDim);
            if (rc == AXIOM_OK) {
                rc = round_bf16_inplace(
                        static_cast<float *>(conv) + static_cast<uint64_t>(column) * kConvDim,
                        kConvDim);
            }
            if (rc == AXIOM_OK) {
                rc = axiom_runtime_deltanet_recurrence_f32_device(
                        layer->runtime, layer->conv, qkv_offset, qkv_offset + key_bytes,
                        qkv_offset + 2u * key_bytes, layer->g, gate_offset,
                        layer->beta, gate_offset, layer->state, state_offset,
                        layer->recurrent, value_offset, kValueHeads, kKeyHeads, kHeadDim);
            }
            if (rc == AXIOM_OK) {
                rc = round_bf16_inplace(
                        static_cast<float *>(recurrent) + static_cast<uint64_t>(column) * kValueDim,
                        kValueDim);
            }
            if (rc == AXIOM_OK) {
                rc = axiom_runtime_deltanet_gated_norm_f32_device(
                        layer->runtime, layer->recurrent, value_offset,
                        layer->ssm_norm_weight, 0u, layer->z_out, value_offset,
                        layer->gated, value_offset, kValueHeads, kHeadDim, kEps);
            }
            if (rc != AXIOM_OK) return rc;
        }
        /* The generic runtime path still produces its gated output in F32.
         * Keep its public BF16 boundary explicit; the native path fuses this
         * conversion in qwen38_gdn_gated_norm8_kernel above. */
        rc = round_bf16_inplace(
                static_cast<float *>(gated), static_cast<uint64_t>(kValueDim) * kBatch);
        if (rc != AXIOM_OK) return rc;
    }
    return axiom_qwen38_fp8_linear_forward_f32_device(
            layer->out_proj, static_cast<const float *>(gated), out, nullptr);
}

extern "C" int axiom_qwen38_gdn_layer_spec_begin(
        axiom_qwen38_gdn_layer *layer,
        void *stream) {
    if (!layer || layer->spec_active || !layer->ring || !layer->state ||
        !layer->spec_delta || !layer->spec_decay) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    layer->spec_active = true;
    layer->spec_forwarded = false;
    layer->spec_stream = cuda_stream;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_gdn_layer_forward_temporal8_f32_device(
        axiom_qwen38_gdn_layer *layer,
        const float *input,
        float *out,
        void *stream) {
    if (!layer || !input || !out || !layer->spec_active ||
        layer->spec_forwarded ||
        layer->spec_stream != static_cast<cudaStream_t>(stream) ||
        !layer->runtime || !layer->qkv_z || !layer->input_norm_weight || !layer->norm ||
        !layer->projection_out || !layer->a_out || !layer->b_out || !layer->g || !layer->beta ||
        !layer->conv || !layer->recurrent || !layer->gated ||
        !layer->spec_delta || !layer->spec_decay) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    void *norm_weight = nullptr;
    void *norm = nullptr;
    void *projection = nullptr;
    void *a = nullptr;
    void *b = nullptr;
    void *a_log = nullptr;
    void *dt_bias = nullptr;
    void *conv_weight = nullptr;
    void *ring = nullptr;
    void *state = nullptr;
    void *ssm_norm_weight = nullptr;
    void *g = nullptr;
    void *beta = nullptr;
    void *conv = nullptr;
    void *recurrent = nullptr;
    void *gated = nullptr;
    void *spec_delta = nullptr;
    void *spec_decay = nullptr;
    int rc = buffer_pointer(layer->input_norm_weight, &norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->projection_out, &projection);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->a_out, &a);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->b_out, &b);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->a_log, &a_log);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->dt_bias, &dt_bias);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv_weight, &conv_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->ssm_norm_weight, &ssm_norm_weight);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->g, &g);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->beta, &beta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv, &conv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->recurrent, &recurrent);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->gated, &gated);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_delta, &spec_delta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_decay, &spec_decay);
    if (rc != AXIOM_OK) return rc;

    qwen38_rmsnorm8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(norm_weight), input, static_cast<float *>(norm));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = qwen38_gdn_project_inputs(
            layer, static_cast<const float *>(norm), static_cast<float *>(projection),
            static_cast<float *>(a), static_cast<float *>(b), stream);
    if (rc != AXIOM_OK) return rc;
    rc = round_ab8(static_cast<float *>(a), static_cast<float *>(b), cuda_stream);
    if (rc != AXIOM_OK) return rc;
    qwen38_gdn_gates_kernel<<<
            dim3((kValueHeads + 127u) / 128u, kBatch), 128u, 0, cuda_stream>>>(
            static_cast<const float *>(a), static_cast<const float *>(b),
            static_cast<const float *>(a_log), static_cast<const float *>(dt_bias),
            static_cast<float *>(g), static_cast<float *>(beta));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_gdn_conv1d_silu_temporal8_kernel<<<
            (kConvDim + kThreads - 1u) / kThreads, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(projection), static_cast<const float *>(conv_weight),
            static_cast<const float *>(ring),
            static_cast<float *>(ring) +
                    static_cast<uint64_t>(kBatch - 1u) * kConvDim * 3u,
            static_cast<float *>(conv));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_gdn_normalize_qk_temporal8_kernel<<<
            dim3(kBatch, kKeyHeads), kHeadDim,
            static_cast<size_t>(2u * kHeadDim) * sizeof(float), cuda_stream>>>(
            static_cast<float *>(conv));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    float *final_state = static_cast<float *>(state) +
            static_cast<uint64_t>(kBatch - 1u) * kValueHeads * kHeadDim * kHeadDim;
    if (layer->temporal_register_state) {
        // The qualified register path is fixed at tile16; temporal_value_tile
        // (AXIOM_QWEN38_GDN_TEMPORAL_VALUE_TILE) applies only to the fallback.
        rc = launch_register_recurrence<16u>(
                static_cast<const float *>(conv), static_cast<const float *>(g),
                static_cast<const float *>(beta), static_cast<const float *>(state),
                final_state, static_cast<float *>(recurrent),
                static_cast<float *>(spec_delta), static_cast<float *>(spec_decay),
                cuda_stream);
    } else
    switch (layer->temporal_value_tile) {
        case 16u:
            rc = launch_gdn_recurrence_temporal8<16u>(
                    static_cast<const float *>(conv), static_cast<const float *>(g),
                    static_cast<const float *>(beta), static_cast<const float *>(state),
                    final_state, static_cast<float *>(recurrent),
                    static_cast<float *>(spec_delta), static_cast<float *>(spec_decay),
                    cuda_stream);
            break;
        case 32u:
            rc = launch_gdn_recurrence_temporal8<32u>(
                    static_cast<const float *>(conv), static_cast<const float *>(g),
                    static_cast<const float *>(beta), static_cast<const float *>(state),
                    final_state, static_cast<float *>(recurrent),
                    static_cast<float *>(spec_delta), static_cast<float *>(spec_decay),
                    cuda_stream);
            break;
        case 64u:
            rc = launch_gdn_recurrence_temporal8<64u>(
                    static_cast<const float *>(conv), static_cast<const float *>(g),
                    static_cast<const float *>(beta), static_cast<const float *>(state),
                    final_state, static_cast<float *>(recurrent),
                    static_cast<float *>(spec_delta), static_cast<float *>(spec_decay),
                    cuda_stream);
            break;
        default:
            rc = launch_gdn_recurrence_temporal8<8u>(
                    static_cast<const float *>(conv), static_cast<const float *>(g),
                    static_cast<const float *>(beta), static_cast<const float *>(state),
                    final_state, static_cast<float *>(recurrent),
                    static_cast<float *>(spec_delta), static_cast<float *>(spec_decay),
                    cuda_stream);
            break;
    }
    if (rc != AXIOM_OK) return rc;
    qwen38_gdn_gated_norm8_kernel<<<
            dim3(kValueHeads, kBatch), kHeadDim,
            static_cast<size_t>(kHeadDim) * sizeof(float), cuda_stream>>>(
            static_cast<const float *>(recurrent), static_cast<const float *>(ssm_norm_weight),
            static_cast<const float *>(projection), static_cast<float *>(gated));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = axiom_qwen38_fp8_linear_forward_f32_device(
            layer->out_proj, static_cast<const float *>(gated), out, stream);
    if (rc == AXIOM_OK) layer->spec_forwarded = true;
    return rc;
}

extern "C" int axiom_qwen38_gdn_layer_spec_commit_prefix(
        axiom_qwen38_gdn_layer *layer,
        uint32_t consumed_tokens,
        void *stream) {
    if (!layer || !layer->spec_active || !layer->spec_forwarded ||
        layer->spec_stream != static_cast<cudaStream_t>(stream) ||
        consumed_tokens == 0u ||
        consumed_tokens > kBatch) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    void *ring = nullptr;
    void *state = nullptr;
    void *projection = nullptr;
    void *conv = nullptr;
    void *spec_delta = nullptr;
    void *spec_decay = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->projection_out, &projection);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv, &conv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_delta, &spec_delta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_decay, &spec_decay);
    if (rc != AXIOM_OK) return rc;
    constexpr uint64_t kStateCount =
            static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
    constexpr uint64_t kMaterializeCount =
            kStateCount > kConvDim ? kStateCount : kConvDim;
    constexpr uint64_t kGrid = (kMaterializeCount + kThreads - 1u) / kThreads;
    static_assert(kGrid <= std::numeric_limits<uint32_t>::max(), "GDN commit grid overflow");
    qwen38_gdn_materialize_prefix_kernel<<<
            static_cast<uint32_t>(kGrid), kThreads, 0, cuda_stream>>>(
            static_cast<float *>(ring), static_cast<float *>(state),
            static_cast<const float *>(projection), static_cast<const float *>(conv),
            static_cast<const float *>(spec_delta), static_cast<const float *>(spec_decay),
            consumed_tokens);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_gdn_layer_spec_commit_prefix_device(
        axiom_qwen38_gdn_layer *layer,
        const uint32_t *consumed_tokens_device,
        void *stream) {
    if (!layer || !consumed_tokens_device || !layer->spec_active || !layer->spec_forwarded ||
        layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    void *ring = nullptr;
    void *state = nullptr;
    void *projection = nullptr;
    void *conv = nullptr;
    void *spec_delta = nullptr;
    void *spec_decay = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->projection_out, &projection);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv, &conv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_delta, &spec_delta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_decay, &spec_decay);
    if (rc != AXIOM_OK) return rc;
    constexpr uint64_t kStateCount =
            static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
    constexpr uint64_t kMaterializeCount =
            kStateCount > kConvDim ? kStateCount : kConvDim;
    constexpr uint64_t kGrid = (kMaterializeCount + kThreads - 1u) / kThreads;
    static_assert(kGrid <= std::numeric_limits<uint32_t>::max(), "GDN commit grid overflow");
    qwen38_gdn_materialize_prefix_device_kernel<<<
            static_cast<uint32_t>(kGrid), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<float *>(ring), static_cast<float *>(state),
            static_cast<const float *>(projection), static_cast<const float *>(conv),
            static_cast<const float *>(spec_delta), static_cast<const float *>(spec_decay),
            consumed_tokens_device, nullptr);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_gdn_layer_spec_finalize_device(
        axiom_qwen38_gdn_layer *layer,
        const uint32_t *consumed_tokens_device,
        const uint32_t *async_status_device,
        void *stream) {
    if (!layer || !consumed_tokens_device || !async_status_device || !layer->spec_active ||
        !layer->spec_forwarded || layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    void *ring = nullptr;
    void *state = nullptr;
    void *projection = nullptr;
    void *conv = nullptr;
    void *spec_delta = nullptr;
    void *spec_decay = nullptr;
    int rc = buffer_pointer(layer->ring, &ring);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->state, &state);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->projection_out, &projection);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->conv, &conv);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_delta, &spec_delta);
    if (rc == AXIOM_OK) rc = buffer_pointer(layer->spec_decay, &spec_decay);
    if (rc != AXIOM_OK) return rc;
    constexpr uint64_t kStateCount =
            static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim;
    constexpr uint64_t kMaterializeCount =
            kStateCount > kConvDim ? kStateCount : kConvDim;
    constexpr uint64_t kGrid = (kMaterializeCount + kThreads - 1u) / kThreads;
    static_assert(kGrid <= std::numeric_limits<uint32_t>::max(), "GDN commit grid overflow");
    qwen38_gdn_materialize_prefix_device_kernel<<<
            static_cast<uint32_t>(kGrid), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<float *>(ring), static_cast<float *>(state),
            static_cast<const float *>(projection), static_cast<const float *>(conv),
            static_cast<const float *>(spec_delta), static_cast<const float *>(spec_decay),
            consumed_tokens_device, async_status_device);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_gdn_layer_spec_abort(
        axiom_qwen38_gdn_layer *layer,
        void *stream) {
    if (!layer || !layer->spec_active ||
        layer->spec_stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(layer->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    /* Temporal forward journals candidate deltas without mutating durable
     * ring/state, so abort is intentionally allocation- and copy-free. */
    layer->spec_active = false;
    layer->spec_forwarded = false;
    layer->spec_stream = nullptr;
    return AXIOM_OK;
}
