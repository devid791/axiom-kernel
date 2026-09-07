#include "axiom/qwen4exp/mtp_provider.hpp"

#include "axiom/qwen4exp/attention.hpp"
#include "axiom/qwen4exp/bf16_linear.hpp"
#include "axiom/qwen4exp/mhc.hpp"
#include "axiom/qwen4exp/moe.hpp"
#include "axiom/qwen4exp/qsa.hpp"
#include "axiom/qwen4exp/shared_expert.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr unsigned kThreads = 256u;
constexpr std::size_t kQProjection = 12288u;
constexpr std::size_t kKvProjection = 512u;
constexpr std::size_t kAttentionOutput = 6144u;
constexpr std::size_t kIndexProjection = 640u;
constexpr std::size_t kIndexQuery = 512u;
constexpr std::size_t kIndexKey = 128u;
constexpr std::size_t kSelectedCapacity = qsa::checkpoint_selected_capacity;
constexpr std::size_t kExpertGateUpElements =
        2u * kMtpIntermediate * kMtpHidden;
constexpr std::size_t kExpertDownElements =
        kMtpHidden * kMtpIntermediate;
constexpr std::size_t kExpertGateUpBytes =
        kExpertGateUpElements * sizeof(std::uint16_t);
constexpr std::size_t kExpertDownBytes =
        kExpertDownElements * sizeof(std::uint16_t);
constexpr float kRmsEpsilon = 1.0e-6F;

struct tensor_contract {
    const char *name;
    std::uint8_t rank;
    std::array<std::uint64_t, 3> shape;
};

constexpr std::array<tensor_contract, kMtpTensorCount> kContracts{{
    {"mtp.fc_embedding.weight", 2u, {2560u, 2560u, 0u}},
    {"mtp.fc_hidden.weight", 2u, {2560u, 2560u, 0u}},
    {"mtp.hyper_connection_mixer.hc_norm.weight", 1u, {10240u, 0u, 0u}},
    {"mtp.hyper_connection_mixer.input_mix_weight_down.weight", 2u, {320u, 10240u, 0u}},
    {"mtp.hyper_connection_mixer.input_mix_weight_up.weight", 2u, {10240u, 320u, 0u}},
    {"mtp.pre_fc_norm_embedding.weight", 1u, {2560u, 0u, 0u}},
    {"mtp.pre_fc_norm_hidden.weight", 1u, {10240u, 0u, 0u}},
    {"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight", 2u, {4u, 10240u, 0u}},
    {"mtp.layers.0.attn_hyper_connection.hc_norm.weight", 1u, {10240u, 0u, 0u}},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight", 2u, {320u, 10240u, 0u}},
    {"mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight", 2u, {10240u, 320u, 0u}},
    {"mtp.layers.0.mlp.experts.down_proj", 3u, {512u, 2560u, 640u}},
    {"mtp.layers.0.mlp.experts.gate_up_proj", 3u, {512u, 1280u, 2560u}},
    {"mtp.layers.0.mlp.gate.weight", 2u, {512u, 2560u, 0u}},
    {"mtp.layers.0.mlp.shared_expert.down_proj.weight", 2u, {2560u, 640u, 0u}},
    {"mtp.layers.0.mlp.shared_expert.gate_proj.weight", 2u, {640u, 2560u, 0u}},
    {"mtp.layers.0.mlp.shared_expert.up_proj.weight", 2u, {640u, 2560u, 0u}},
    {"mtp.layers.0.mlp.shared_expert_gate.weight", 2u, {1u, 2560u, 0u}},
    {"mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight", 2u, {4u, 10240u, 0u}},
    {"mtp.layers.0.mlp_hyper_connection.hc_norm.weight", 1u, {10240u, 0u, 0u}},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight", 2u, {320u, 10240u, 0u}},
    {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight", 2u, {10240u, 320u, 0u}},
    {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight", 2u, {640u, 2560u, 0u}},
    {"mtp.layers.0.self_attn.indexer.k_layernorm.weight", 1u, {128u, 0u, 0u}},
    {"mtp.layers.0.self_attn.indexer.q_layernorm.weight", 1u, {128u, 0u, 0u}},
    {"mtp.layers.0.self_attn.k_norm.weight", 1u, {256u, 0u, 0u}},
    {"mtp.layers.0.self_attn.k_proj.weight", 2u, {512u, 2560u, 0u}},
    {"mtp.layers.0.self_attn.o_proj.weight", 2u, {2560u, 6144u, 0u}},
    {"mtp.layers.0.self_attn.q_norm.weight", 1u, {256u, 0u, 0u}},
    {"mtp.layers.0.self_attn.q_proj.weight", 2u, {12288u, 2560u, 0u}},
    {"mtp.layers.0.self_attn.v_proj.weight", 2u, {512u, 2560u, 0u}},
}};

void set_error(std::string *error, const std::string &message) noexcept {
    if (error == nullptr) return;
    try {
        *error = "qwen4_exp MTP: " + message;
    } catch (...) {
    }
}

mtp_status fail(mtp_status status,
                std::string *error,
                const std::string &message) noexcept {
    set_error(error, message);
    return status;
}

bool checked_mul(std::size_t left,
                 std::size_t right,
                 std::size_t *out) noexcept {
    if (out == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

bool current_sm120_device(int *device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, *device) == cudaSuccess &&
           properties.major == 12 && properties.minor == 0;
}

bool device_range_valid(const void *pointer,
                        std::size_t bytes,
                        int expected_device) noexcept {
    if (pointer == nullptr || bytes == 0u) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        return false;
    }
    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    const auto address = static_cast<CUdeviceptr>(
            reinterpret_cast<std::uintptr_t>(pointer));
    if (cuMemGetAddressRange(&allocation_base, &allocation_bytes, address) !=
        CUDA_SUCCESS) {
        return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - allocation_base);
    return offset <= allocation_bytes && bytes <= allocation_bytes - offset;
}

float bf16_to_f32_host(std::uint16_t encoded) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

mtp_status map_linear(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return mtp_status::ok;
        case bf16_linear_status::invalid_argument:
            return mtp_status::invalid_argument;
        case bf16_linear_status::unsupported_config:
            return mtp_status::unsupported_config;
        case bf16_linear_status::tensor_not_found:
            return mtp_status::tensor_not_found;
        case bf16_linear_status::dtype_mismatch:
        case bf16_linear_status::shape_mismatch:
            return mtp_status::tensor_contract_mismatch;
        case bf16_linear_status::checkpoint_io_error:
            return mtp_status::checkpoint_io_error;
        case bf16_linear_status::allocation_failure:
            return mtp_status::allocation_failure;
        case bf16_linear_status::unsupported_device:
            return mtp_status::unsupported_device;
        case bf16_linear_status::invalid_device_pointer:
            return mtp_status::invalid_device_pointer;
        case bf16_linear_status::size_overflow:
        case bf16_linear_status::cuda_error:
        case bf16_linear_status::cublas_error:
            return mtp_status::linear_error;
    }
    return mtp_status::linear_error;
}

mtp_status map_qsa(qsa::status status) noexcept {
    switch (status) {
        case qsa::status::ok: return mtp_status::ok;
        case qsa::status::null_pointer:
        case qsa::status::invalid_argument:
            return mtp_status::invalid_argument;
        case qsa::status::invalid_config:
            return mtp_status::unsupported_config;
        case qsa::status::capacity_exceeded:
            return mtp_status::capacity_exceeded;
        case qsa::status::arithmetic_overflow:
        case qsa::status::non_finite:
        case qsa::status::invalid_visible_index:
            return mtp_status::qsa_error;
        case qsa::status::cuda_failure:
            return mtp_status::cuda_error;
    }
    return mtp_status::qsa_error;
}

mtp_status map_attention(attention::status status) noexcept {
    switch (status) {
        case attention::status::ok: return mtp_status::ok;
        case attention::status::null_pointer:
        case attention::status::invalid_argument:
            return mtp_status::invalid_argument;
        case attention::status::invalid_config:
            return mtp_status::unsupported_config;
        case attention::status::invalid_state:
            return mtp_status::invalid_state;
        case attention::status::capacity_exceeded:
            return mtp_status::capacity_exceeded;
        case attention::status::arithmetic_overflow:
        case attention::status::insufficient_workspace:
        case attention::status::invalid_selected_count:
        case attention::status::invalid_selected_index:
        case attention::status::non_finite:
            return mtp_status::attention_error;
        case attention::status::cuda_failure:
            return mtp_status::cuda_error;
    }
    return mtp_status::attention_error;
}

mtp_status load_linear(const checkpoint_catalog &catalog,
                       const std::string &name,
                       std::size_t input,
                       std::size_t output,
                       std::size_t max_batch,
                       const mtp_provider_config &provider_config,
                       cudaStream_t stream,
                       std::unique_ptr<resident_bf16_linear> *linear,
                       std::string *error) noexcept {
    bf16_linear_config config{};
    config.input_features = input;
    config.output_features = output;
    config.max_batch = max_batch;
    config.upload_chunk_bytes = provider_config.upload_chunk_bytes;
    config.blas_workspace_bytes = provider_config.blas_workspace_bytes;
    return map_linear(resident_bf16_linear::load(
            catalog, name, config, stream, linear, error));
}

mtp_status load_bf16(const checkpoint_catalog &catalog,
                     const std::string &name,
                     std::initializer_list<std::uint64_t> shape,
                     std::uint16_t **output,
                     std::uint64_t *resident_bytes,
                     std::string *error) noexcept {
    if (output == nullptr || *output != nullptr) {
        return fail(mtp_status::invalid_argument, error,
                    "invalid BF16 load destination: " + name);
    }
    const tensor_span *span = catalog.find(name);
    if (span == nullptr) {
        return fail(mtp_status::tensor_not_found, error,
                    "missing tensor " + name);
    }
    if (span->dtype != tensor_dtype::bf16 || span->rank != shape.size()) {
        return fail(mtp_status::tensor_contract_mismatch, error,
                    "BF16 dtype/rank mismatch for " + name);
    }
    std::size_t index = 0u;
    std::uint64_t elements = 1u;
    for (std::uint64_t dimension : shape) {
        if (span->shape[index++] != dimension ||
            (dimension != 0u &&
             elements > std::numeric_limits<std::uint64_t>::max() / dimension)) {
            return fail(mtp_status::tensor_contract_mismatch, error,
                        "shape mismatch for " + name);
        }
        elements *= dimension;
    }
    if (elements > std::numeric_limits<std::size_t>::max() /
                           sizeof(std::uint16_t) ||
        span->bytes != elements * sizeof(std::uint16_t)) {
        return fail(mtp_status::tensor_contract_mismatch, error,
                    "byte length mismatch for " + name);
    }
    const std::size_t bytes = static_cast<std::size_t>(span->bytes);
    std::vector<std::uint16_t> host(elements);
    std::string read_error;
    if (!catalog.read_range(name, 0u, host.data(), bytes, &read_error)) {
        return fail(mtp_status::checkpoint_io_error, error,
                    read_error.empty() ? "read failed for " + name : read_error);
    }
    for (std::uint16_t value : host) {
        if (!std::isfinite(bf16_to_f32_host(value))) {
            return fail(mtp_status::tensor_contract_mismatch, error,
                        "non-finite BF16 value in " + name);
        }
    }
    if (cudaMalloc(reinterpret_cast<void **>(output), bytes) != cudaSuccess) {
        return fail(mtp_status::allocation_failure, error,
                    "GPU allocation failed for " + name);
    }
    if (cudaMemcpy(*output, host.data(), bytes, cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        (void)cudaFree(*output);
        *output = nullptr;
        return fail(mtp_status::cuda_error, error,
                    "GPU upload failed for " + name);
    }
    if (resident_bytes != nullptr) *resident_bytes += bytes;
    return mtp_status::ok;
}

__device__ __forceinline__ float stable_sigmoid(float value) {
    if (value >= 0.0F) {
        const float inverse = expf(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = expf(value);
    return exponential / (1.0F + exponential);
}

__device__ __forceinline__ float bf16_at(const std::uint16_t *values,
                                         std::size_t index) {
    return __bfloat162float(
            reinterpret_cast<const __nv_bfloat16 *>(values)[index]);
}

__device__ __forceinline__ void record_compute_error(
        std::uint32_t *status,
        mtp_status value) {
    (void)atomicCAS(status, 0u, static_cast<std::uint32_t>(value));
}

__global__ void gemma_rmsnorm_kernel(const float *input,
                                     const std::uint16_t *weight,
                                     std::size_t width,
                                     std::size_t rows,
                                     float *output,
                                     std::uint32_t *status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;
    __shared__ float reduction[kThreads];
    float sum = 0.0F;
    bool invalid = false;
    for (std::size_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        const float value = input[row * width + column];
        invalid = invalid || !isfinite(value);
        sum = fmaf(value, value, sum);
    }
    if (invalid || !isfinite(sum)) {
        record_compute_error(status, mtp_status::non_finite);
        sum = 0.0F;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / static_cast<float>(width) +
                                 kRmsEpsilon);
    for (std::size_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        const float scale = 1.0F + bf16_at(weight, column);
        const float value = input[row * width + column] * inverse * scale;
        if (!isfinite(value)) {
            record_compute_error(status, mtp_status::non_finite);
            output[row * width + column] = 0.0F;
        } else {
            output[row * width + column] = value;
        }
    }
}

__global__ void fusion_add_kernel(const float *embedding,
                                  const float *hidden,
                                  float *output) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index < kMtpResidual) {
        output[index] = hidden[index] + embedding[index % kMtpHidden];
    }
}

__global__ void grouped_rmsnorm_kernel(const float *input,
                                       const std::uint16_t *weight,
                                       float *output,
                                       std::uint32_t *status) {
    const std::size_t stream = static_cast<std::size_t>(blockIdx.x);
    if (stream >= kMtpStreams) return;
    __shared__ float reduction[kThreads];
    float sum = 0.0F;
    bool invalid = false;
    const std::size_t base = stream * kMtpHidden;
    for (std::size_t column = threadIdx.x; column < kMtpHidden;
         column += blockDim.x) {
        const float value = input[base + column];
        invalid = invalid || !isfinite(value);
        sum = fmaf(value, value, sum);
    }
    if (invalid || !isfinite(sum)) {
        record_compute_error(status, mtp_status::non_finite);
        sum = 0.0F;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / static_cast<float>(kMtpHidden) +
                                 kRmsEpsilon);
    for (std::size_t column = threadIdx.x; column < kMtpHidden;
         column += blockDim.x) {
        const float value = input[base + column] * inverse *
                            (1.0F + bf16_at(weight, base + column));
        if (!isfinite(value)) {
            record_compute_error(status, mtp_status::non_finite);
            output[base + column] = 0.0F;
        } else {
            output[base + column] = value;
        }
    }
}

__global__ void scaled_silu_kernel(float *values) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index < kMtpRank) {
        const float scaled = values[index] / static_cast<float>(kMtpStreams);
        values[index] = scaled * stable_sigmoid(scaled);
    }
}

__global__ void mhc_combine_kernel(const float *normalized,
                                   const float *mix_logits,
                                   const float *injection_logits,
                                   float *mixed,
                                   float *injection,
                                   std::uint32_t *status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index < kMtpHidden) {
        float sum = 0.0F;
        for (std::size_t stream = 0u; stream < kMtpStreams; ++stream) {
            const std::size_t source = stream * kMtpHidden + index;
            sum += stable_sigmoid(mix_logits[source]) * normalized[source];
        }
        const float value = sum / static_cast<float>(kMtpStreams);
        if (!isfinite(value)) {
            record_compute_error(status, mtp_status::non_finite);
            mixed[index] = 0.0F;
        } else {
            mixed[index] = value;
        }
    }
    if (index < kMtpStreams) {
        const float value = 2.0F * stable_sigmoid(
                injection_logits[index] / static_cast<float>(kMtpStreams));
        if (!isfinite(value)) {
            record_compute_error(status, mtp_status::non_finite);
            injection[index] = 0.0F;
        } else {
            injection[index] = value;
        }
    }
}

__global__ void mhc_finalize_kernel(const float *normalized,
                                    const float *mix_logits,
                                    float *mixed,
                                    std::uint32_t *status) {
    const std::size_t hidden = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                               threadIdx.x;
    if (hidden >= kMtpHidden) return;
    float sum = 0.0F;
    for (std::size_t stream = 0u; stream < kMtpStreams; ++stream) {
        const std::size_t index = stream * kMtpHidden + hidden;
        sum += stable_sigmoid(mix_logits[index]) * normalized[index];
    }
    const float value = sum / static_cast<float>(kMtpStreams);
    if (!isfinite(value)) {
        record_compute_error(status, mtp_status::non_finite);
        mixed[hidden] = 0.0F;
    } else {
        mixed[hidden] = value;
    }
}

__global__ void projection_to_bf16_kernel(const float *input,
                                          std::uint16_t *output,
                                          std::size_t elements,
                                          attention::status *status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index >= elements) return;
    float value = input[index];
    if (!isfinite(value)) {
        (void)atomicCAS(reinterpret_cast<unsigned int *>(status),
                        static_cast<unsigned int>(attention::status::ok),
                        static_cast<unsigned int>(attention::status::non_finite));
        value = 0.0F;
    }
    reinterpret_cast<__nv_bfloat16 *>(output)[index] = __float2bfloat16_rn(value);
}

__global__ void split_index_projection_kernel(
        const float *projection,
        std::size_t cache_start,
        std::uint16_t *query_bf16,
        std::uint16_t *index_key_cache_bf16,
        qsa::status *status) {
    const std::size_t lane = threadIdx.x;
    if (lane >= kIndexProjection) return;
    float value = projection[lane];
    if (!isfinite(value)) {
        (void)atomicCAS(reinterpret_cast<unsigned int *>(status),
                        static_cast<unsigned int>(qsa::status::ok),
                        static_cast<unsigned int>(qsa::status::non_finite));
        value = 0.0F;
    }
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    if (lane < kIndexQuery) {
        reinterpret_cast<__nv_bfloat16 *>(query_bf16)[lane] = converted;
    } else {
        reinterpret_cast<__nv_bfloat16 *>(index_key_cache_bf16)
                [cache_start * kIndexKey + lane - kIndexQuery] = converted;
    }
}

__global__ void fill_indices_kernel(std::int32_t *indices,
                                    std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index < count) indices[index] = static_cast<std::int32_t>(index);
}

__global__ void store_count_kernel(std::uint32_t *output,
                                   std::uint32_t value) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) *output = value;
}

__global__ void bf16_to_f32_kernel(const std::uint16_t *input,
                                   float *output,
                                   std::size_t elements,
                                   attention::status *status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index >= elements) return;
    const float value = bf16_at(input, index);
    if (!isfinite(value)) {
        (void)atomicCAS(reinterpret_cast<unsigned int *>(status),
                        static_cast<unsigned int>(attention::status::ok),
                        static_cast<unsigned int>(attention::status::non_finite));
        output[index] = 0.0F;
    } else {
        output[index] = value;
    }
}

__global__ void expert_gate_up_kernel(const std::uint16_t *gate_up_slots,
                                      const float *input,
                                      float *intermediate,
                                      std::uint32_t *status) {
    const std::size_t ordinal = static_cast<std::size_t>(blockIdx.x);
    const std::size_t slot = ordinal / kMtpIntermediate;
    const std::size_t row = ordinal % kMtpIntermediate;
    if (slot >= kMtpTopK) return;
    __shared__ float gate_sums[kThreads];
    __shared__ float up_sums[kThreads];
    const std::size_t slot_base = slot * kExpertGateUpElements;
    const std::size_t gate_base = slot_base + row * kMtpHidden;
    const std::size_t up_base = slot_base +
            (kMtpIntermediate + row) * kMtpHidden;
    float gate = 0.0F;
    float up = 0.0F;
    bool invalid = false;
    for (std::size_t column = threadIdx.x; column < kMtpHidden;
         column += blockDim.x) {
        const float x = input[column];
        const float g = bf16_at(gate_up_slots, gate_base + column);
        const float u = bf16_at(gate_up_slots, up_base + column);
        invalid = invalid || !isfinite(x) || !isfinite(g) || !isfinite(u);
        gate = fmaf(g, x, gate);
        up = fmaf(u, x, up);
    }
    if (invalid || !isfinite(gate) || !isfinite(up)) {
        record_compute_error(status, mtp_status::non_finite);
        gate = 0.0F;
        up = 0.0F;
    }
    gate_sums[threadIdx.x] = gate;
    up_sums[threadIdx.x] = up;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            gate_sums[threadIdx.x] += gate_sums[threadIdx.x + stride];
            up_sums[threadIdx.x] += up_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const float activation = gate_sums[0] * stable_sigmoid(gate_sums[0]) *
                                 up_sums[0];
        if (!isfinite(activation)) {
            record_compute_error(status, mtp_status::non_finite);
            intermediate[slot * kMtpIntermediate + row] = 0.0F;
        } else {
            intermediate[slot * kMtpIntermediate + row] = activation;
        }
    }
}

__global__ void expert_down_kernel(const std::uint16_t *down_slots,
                                   const float *intermediate,
                                   const float *router_weights,
                                   float *output,
                                   std::uint32_t *status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= kMtpHidden) return;
    __shared__ float reduction[kThreads];
    float sum = 0.0F;
    bool invalid = false;
    for (std::size_t slot = 0u; slot < kMtpTopK; ++slot) {
        const float route = router_weights[slot];
        const std::size_t weight_base = slot * kExpertDownElements +
                                        row * kMtpIntermediate;
        const std::size_t activation_base = slot * kMtpIntermediate;
        float slot_sum = 0.0F;
        for (std::size_t column = threadIdx.x; column < kMtpIntermediate;
             column += blockDim.x) {
            const float activation = intermediate[activation_base + column];
            const float weight = bf16_at(down_slots, weight_base + column);
            invalid = invalid || !isfinite(activation) || !isfinite(weight) ||
                      !isfinite(route);
            slot_sum = fmaf(weight, activation, slot_sum);
        }
        sum = fmaf(route, slot_sum, sum);
    }
    if (invalid || !isfinite(sum)) {
        record_compute_error(status, mtp_status::non_finite);
        sum = 0.0F;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) output[row] = reduction[0];
}

__global__ void vector_sum_kernel(const float *left,
                                  const float *right,
                                  float *output,
                                  std::uint32_t *status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index >= kMtpHidden) return;
    const float value = left[index] + right[index];
    if (!isfinite(value)) {
        record_compute_error(status, mtp_status::non_finite);
        output[index] = 0.0F;
    } else {
        output[index] = value;
    }
}

__global__ void argmax_kernel(const float *logits,
                              std::uint32_t *token,
                              float *selected,
                              std::uint32_t *status) {
    __shared__ float values[kThreads];
    __shared__ std::uint32_t indices[kThreads];
    float best = -INFINITY;
    std::uint32_t best_index = UINT32_MAX;
    bool invalid = false;
    for (std::size_t index = threadIdx.x; index < kMtpVocab;
         index += blockDim.x) {
        const float value = logits[index];
        if (!isfinite(value)) {
            invalid = true;
            continue;
        }
        if (value > best || (value == best && index < best_index)) {
            best = value;
            best_index = static_cast<std::uint32_t>(index);
        }
    }
    if (invalid) record_compute_error(status, mtp_status::non_finite);
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const std::uint32_t other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x] ||
                (other_value == values[threadIdx.x] &&
                 other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        if (indices[0] == UINT32_MAX) {
            record_compute_error(status, mtp_status::non_finite);
            *token = 0u;
            *selected = 0.0F;
        } else {
            *token = indices[0];
            *selected = values[0];
        }
    }
}

struct mhc_weights {
    std::uint16_t *norm = nullptr;
    std::unique_ptr<resident_bf16_linear> down;
    std::unique_ptr<resident_bf16_linear> up;
    std::unique_ptr<resident_bf16_linear> inject;
};

}  // namespace

const char *mtp_status_string(mtp_status status) noexcept {
    switch (status) {
        case mtp_status::ok: return "ok";
        case mtp_status::invalid_argument: return "invalid_argument";
        case mtp_status::unsupported_config: return "unsupported_config";
        case mtp_status::unsupported_device: return "unsupported_device";
        case mtp_status::tensor_not_found: return "tensor_not_found";
        case mtp_status::tensor_contract_mismatch: return "tensor_contract_mismatch";
        case mtp_status::checkpoint_io_error: return "checkpoint_io_error";
        case mtp_status::allocation_failure: return "allocation_failure";
        case mtp_status::invalid_device_pointer: return "invalid_device_pointer";
        case mtp_status::invalid_state: return "invalid_state";
        case mtp_status::capacity_exceeded: return "capacity_exceeded";
        case mtp_status::linear_error: return "linear_error";
        case mtp_status::qsa_error: return "qsa_error";
        case mtp_status::attention_error: return "attention_error";
        case mtp_status::router_error: return "router_error";
        case mtp_status::expert_paging_error: return "expert_paging_error";
        case mtp_status::expert_compute_error: return "expert_compute_error";
        case mtp_status::non_finite: return "non_finite";
        case mtp_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

mtp_status mtp_validate_config(const mtp_provider_config &config) noexcept {
    if (config.max_context == 0u || config.max_context > kMtpMaxContext ||
        config.upload_chunk_bytes < 4096u ||
        config.upload_chunk_bytes > 64u * 1024u * 1024u ||
        (config.upload_chunk_bytes % sizeof(std::uint16_t)) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > 64u * 1024u * 1024u ||
        (config.blas_workspace_bytes % 256u) != 0u) {
        return mtp_status::unsupported_config;
    }
    return mtp_status::ok;
}

mtp_status mtp_validate_checkpoint_contract(
        const checkpoint_catalog &catalog,
        mtp_contract_report *report,
        std::string *error) noexcept {
    if (report == nullptr) {
        return fail(mtp_status::invalid_argument, error, "null contract report");
    }
    *report = {};
    std::unordered_set<std::string> expected;
    try {
        expected.reserve(kContracts.size());
        for (const tensor_contract &contract : kContracts) {
            expected.emplace(contract.name);
            const tensor_span *span = catalog.find(contract.name);
            if (span == nullptr) {
                return fail(mtp_status::tensor_not_found, error,
                            "missing required tensor " +
                                    std::string(contract.name));
            }
            if (span->dtype != tensor_dtype::bf16 ||
                span->rank != contract.rank) {
                return fail(mtp_status::tensor_contract_mismatch, error,
                            "dtype/rank mismatch for " +
                                    std::string(contract.name));
            }
            std::uint64_t elements = 1u;
            for (std::size_t dimension = 0u; dimension < contract.rank;
                 ++dimension) {
                if (span->shape[dimension] != contract.shape[dimension] ||
                    (contract.shape[dimension] != 0u &&
                     elements > std::numeric_limits<std::uint64_t>::max() /
                                        contract.shape[dimension])) {
                    return fail(mtp_status::tensor_contract_mismatch, error,
                                "shape mismatch for " +
                                        std::string(contract.name));
                }
                elements *= contract.shape[dimension];
            }
            if (span->bytes != elements * sizeof(std::uint16_t)) {
                return fail(mtp_status::tensor_contract_mismatch, error,
                            "byte length mismatch for " +
                                    std::string(contract.name));
            }
            ++report->tensor_count;
            report->tensor_bytes += span->bytes;
        }
        std::size_t observed = 0u;
        for (std::size_t index = 0u; index < catalog.tensor_count(); ++index) {
            const tensor_span *span = catalog.at(index);
            if (span == nullptr || span->name.rfind("mtp.", 0u) != 0u) continue;
            ++observed;
            if (expected.erase(span->name) == 0u) {
                return fail(mtp_status::tensor_contract_mismatch, error,
                            "unexpected MTP tensor " + span->name);
            }
        }
        if (observed != kMtpTensorCount || !expected.empty()) {
            return fail(mtp_status::tensor_contract_mismatch, error,
                        "MTP namespace is not exactly 31 tensors");
        }
        const tensor_span *head = catalog.find("lm_head.weight");
        if (head == nullptr || head->dtype != tensor_dtype::bf16 ||
            head->rank != 2u || head->shape[0] != kMtpVocab ||
            head->shape[1] != kMtpHidden ||
            head->bytes != kMtpVocab * kMtpHidden * sizeof(std::uint16_t)) {
            return fail(mtp_status::tensor_contract_mismatch, error,
                        "shared lm_head.weight contract mismatch");
        }
        report->exact_namespace = true;
        report->all_bf16 = true;
        report->single_full_attention_layer = true;
        report->shared_lm_head = true;
        return mtp_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(mtp_status::allocation_failure, error,
                    "host allocation failed during MTP admission");
    } catch (...) {
        return fail(mtp_status::tensor_contract_mismatch, error,
                    "unexpected MTP admission failure");
    }
}

struct mtp_provider::impl {
    mtp_provider_config config{};
    mtp_contract_report contract{};
    std::shared_ptr<checkpoint_catalog> catalog;
    int device = -1;
    bool ready = false;
    std::uint64_t resident_bytes = 0u;

    std::uint16_t *pre_fc_embedding_norm = nullptr;
    std::uint16_t *pre_fc_hidden_norm = nullptr;
    std::unique_ptr<resident_bf16_linear> fc_embedding;
    std::unique_ptr<resident_bf16_linear> fc_hidden;

    mhc_weights attention_mhc;
    mhc_weights mlp_mhc;
    mhc_weights final_mhc;

    std::unique_ptr<resident_bf16_linear> q_projection;
    std::unique_ptr<resident_bf16_linear> k_projection;
    std::unique_ptr<resident_bf16_linear> v_projection;
    std::unique_ptr<resident_bf16_linear> index_projection;
    std::unique_ptr<resident_bf16_linear> o_projection;
    std::uint16_t *q_norm = nullptr;
    std::uint16_t *k_norm = nullptr;
    std::uint16_t *index_q_norm = nullptr;
    std::uint16_t *index_k_norm = nullptr;

    std::unique_ptr<resident_bf16_linear> router;
    std::uint16_t *shared_gate = nullptr;
    std::uint16_t *shared_up = nullptr;
    std::uint16_t *shared_down = nullptr;
    std::uint16_t *shared_output_gate = nullptr;
    std::unique_ptr<resident_bf16_linear> lm_head;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && device >= 0 && previous != device &&
                             cudaSetDevice(device) == cudaSuccess;
        const std::array<std::uint16_t **, 12> pointers{{
            &pre_fc_embedding_norm,
            &pre_fc_hidden_norm,
            &attention_mhc.norm,
            &mlp_mhc.norm,
            &final_mhc.norm,
            &q_norm,
            &k_norm,
            &index_q_norm,
            &index_k_norm,
            &shared_gate,
            &shared_up,
            &shared_down,
        }};
        for (std::uint16_t **pointer : pointers) {
            if (*pointer != nullptr) (void)cudaFree(*pointer);
            *pointer = nullptr;
        }
        if (shared_output_gate != nullptr) (void)cudaFree(shared_output_gate);
        shared_output_gate = nullptr;
        if (changed) (void)cudaSetDevice(previous);
    }
};

namespace {

struct host_control_block {
    std::array<std::uint32_t, kMtpTopK> route_indices{};
    std::array<float, kMtpTopK> route_weights{};
    std::uint32_t router_status = 0u;
    std::uint32_t compute_status = 0u;
    std::uint32_t shared_status = 0u;
    std::uint32_t token = 0u;
    float selected_logit = 0.0F;
};

}  // namespace

struct mtp_session::impl {
    mtp_provider::impl *owner = nullptr;
    int device = -1;
    cudaStream_t bound_stream = nullptr;
    std::size_t capacity = 0u;
    std::size_t committed = 0u;
    bool transaction_open = false;
    bool staged = false;
    bool poisoned = false;
    bool ready = false;

    std::vector<void *> device_allocations;
    std::vector<void *> host_allocations;

    attention::cache_state attention_cache{};
    std::uint16_t *key_cache = nullptr;
    std::uint16_t *value_cache = nullptr;
    std::uint16_t *index_key_cache = nullptr;

    std::uint16_t *linear_input_bf16 = nullptr;
    void *blas_workspace = nullptr;
    float *normalized_embedding = nullptr;
    float *normalized_hidden = nullptr;
    float *fc_embedding_output = nullptr;
    float *fc_hidden_output = nullptr;
    float *fused_hidden = nullptr;

    float *mhc_normalized = nullptr;
    float *mhc_low_rank = nullptr;
    float *mhc_mix_logits = nullptr;
    float *mhc_injection_logits = nullptr;
    float *mhc_mixed = nullptr;
    float *mhc_injection = nullptr;
    float *attention_residual = nullptr;
    float *qsa_output = nullptr;
    float *moe_input = nullptr;
    float *moe_output = nullptr;

    float *q_projection = nullptr;
    float *k_projection = nullptr;
    float *v_projection = nullptr;
    float *index_projection = nullptr;
    std::uint16_t *q_projection_bf16 = nullptr;
    std::uint16_t *k_projection_bf16 = nullptr;
    std::uint16_t *v_projection_bf16 = nullptr;
    std::uint16_t *index_query_bf16 = nullptr;
    float *index_query_prepared = nullptr;
    std::int32_t *visible_indices = nullptr;
    float *pooled_index_keys = nullptr;
    float *index_scores = nullptr;
    void *selection_workspace = nullptr;
    std::size_t selection_workspace_bytes = 0u;
    std::int32_t *selected_indices = nullptr;
    std::uint32_t *selected_count = nullptr;
    void *attention_workspace = nullptr;
    std::size_t attention_workspace_bytes = 0u;
    std::uint16_t *attention_output_bf16 = nullptr;
    float *attention_output_f32 = nullptr;
    qsa::status *qsa_status = nullptr;
    attention::status *attention_status = nullptr;

    float *router_logits = nullptr;
    std::uint32_t *route_indices = nullptr;
    float *route_weights = nullptr;
    std::uint32_t *router_status = nullptr;
    std::uint16_t *expert_gate_up_slots = nullptr;
    std::uint16_t *expert_down_slots = nullptr;
    float *expert_intermediate = nullptr;
    float *routed_output = nullptr;
    float *shared_intermediate = nullptr;
    float *shared_gate_value = nullptr;
    std::uint32_t *shared_status = nullptr;
    float *shared_output = nullptr;
    std::array<std::uint32_t, kMtpTopK> cached_experts{};

    float *final_hidden = nullptr;
    float *logits = nullptr;
    std::uint32_t *selected_token = nullptr;
    float *selected_logit = nullptr;
    std::uint32_t *compute_status = nullptr;

    host_control_block *host_control = nullptr;
    std::uint16_t *host_gate_up_slots = nullptr;
    std::uint16_t *host_down_slots = nullptr;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && device >= 0 && previous != device &&
                             cudaSetDevice(device) == cudaSuccess;
        for (void *pointer : device_allocations) {
            if (pointer != nullptr) (void)cudaFree(pointer);
        }
        for (void *pointer : host_allocations) {
            if (pointer != nullptr) (void)cudaFreeHost(pointer);
        }
        if (changed) (void)cudaSetDevice(previous);
    }

    template <typename T>
    bool allocate_device(std::size_t elements,
                         T **output,
                         std::string *error,
                         const char *label) noexcept {
        if (output == nullptr || *output != nullptr || elements == 0u ||
            elements > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            set_error(error, std::string("invalid allocation for ") + label);
            return false;
        }
        void *pointer = nullptr;
        if (cudaMalloc(&pointer, elements * sizeof(T)) != cudaSuccess) {
            set_error(error, std::string("GPU allocation failed for ") + label);
            return false;
        }
        try {
            device_allocations.push_back(pointer);
        } catch (...) {
            (void)cudaFree(pointer);
            set_error(error, std::string("allocation ledger failed for ") + label);
            return false;
        }
        *output = static_cast<T *>(pointer);
        return true;
    }

    bool allocate_bytes(std::size_t bytes,
                        void **output,
                        std::string *error,
                        const char *label) noexcept {
        return allocate_device<unsigned char>(
                bytes, reinterpret_cast<unsigned char **>(output), error, label);
    }

    bool allocate_host(std::size_t bytes,
                       void **output,
                       std::string *error,
                       const char *label) noexcept {
        if (output == nullptr || *output != nullptr || bytes == 0u) {
            set_error(error, std::string("invalid host allocation for ") + label);
            return false;
        }
        void *pointer = nullptr;
        if (cudaHostAlloc(&pointer, bytes, cudaHostAllocPortable) != cudaSuccess) {
            set_error(error, std::string("pinned allocation failed for ") + label);
            return false;
        }
        try {
            host_allocations.push_back(pointer);
        } catch (...) {
            (void)cudaFreeHost(pointer);
            set_error(error, std::string("host allocation ledger failed for ") + label);
            return false;
        }
        *output = pointer;
        return true;
    }
};

namespace {

mtp_status load_mhc_weights(const checkpoint_catalog &catalog,
                            const std::string &prefix,
                            bool with_injection,
                            const mtp_provider_config &config,
                            cudaStream_t stream,
                            mhc_weights *weights,
                            std::uint64_t *resident_bytes,
                            std::string *error) noexcept {
    mtp_status status = load_bf16(
            catalog, prefix + "hc_norm.weight", {kMtpResidual},
            &weights->norm, resident_bytes, error);
    if (status != mtp_status::ok) return status;
    status = load_linear(
            catalog, prefix + "input_mix_weight_down.weight",
            kMtpResidual, kMtpRank, 1u, config, stream,
            &weights->down, error);
    if (status != mtp_status::ok) return status;
    *resident_bytes += weights->down->weight_bytes();
    status = load_linear(
            catalog, prefix + "input_mix_weight_up.weight",
            kMtpRank, kMtpResidual, 1u, config, stream,
            &weights->up, error);
    if (status != mtp_status::ok) return status;
    *resident_bytes += weights->up->weight_bytes();
    if (with_injection) {
        status = load_linear(
                catalog, prefix + "block_inject_weight.weight",
                kMtpResidual, kMtpStreams, 1u, config, stream,
                &weights->inject, error);
        if (status != mtp_status::ok) return status;
        *resident_bytes += weights->inject->weight_bytes();
    }
    return mtp_status::ok;
}

mtp_status run_linear(resident_bf16_linear *linear,
                      const float *input,
                      std::size_t batch,
                      float *output,
                      mtp_session::impl &session,
                      cudaStream_t stream) noexcept {
    if (linear == nullptr) return mtp_status::invalid_state;
    const bf16_linear_scratch scratch{
        session.linear_input_bf16,
        kMtpResidual * sizeof(std::uint16_t),
        session.blas_workspace,
        session.owner->config.blas_workspace_bytes,
    };
    return map_linear(linear->forward(input, batch, scratch, output, stream));
}

mtp_status run_mhc_prepare(const mhc_weights &weights,
                           const float *input,
                           mtp_session::impl &session,
                           cudaStream_t stream) noexcept {
    if (weights.norm == nullptr || !weights.down || !weights.up ||
        !weights.inject) {
        return mtp_status::invalid_state;
    }
    grouped_rmsnorm_kernel<<<kMtpStreams, kThreads, 0u, stream>>>(
            input, weights.norm, session.mhc_normalized,
            session.compute_status);
    if (cudaPeekAtLastError() != cudaSuccess) return mtp_status::cuda_error;
    mtp_status status = run_linear(
            weights.down.get(), session.mhc_normalized, 1u,
            session.mhc_low_rank, session, stream);
    if (status != mtp_status::ok) return status;
    scaled_silu_kernel<<<1u, kThreads, 0u, stream>>>(session.mhc_low_rank);
    if (cudaPeekAtLastError() != cudaSuccess) return mtp_status::cuda_error;
    status = run_linear(weights.up.get(), session.mhc_low_rank, 1u,
                        session.mhc_mix_logits, session, stream);
    if (status != mtp_status::ok) return status;
    status = run_linear(weights.inject.get(), session.mhc_normalized, 1u,
                        session.mhc_injection_logits, session, stream);
    if (status != mtp_status::ok) return status;
    const unsigned blocks = static_cast<unsigned>(
            (kMtpHidden + kThreads - 1u) / kThreads);
    mhc_combine_kernel<<<blocks, kThreads, 0u, stream>>>(
            session.mhc_normalized, session.mhc_mix_logits,
            session.mhc_injection_logits, session.mhc_mixed,
            session.mhc_injection, session.compute_status);
    return cudaPeekAtLastError() == cudaSuccess ? mtp_status::ok
                                                : mtp_status::cuda_error;
}

mtp_status run_mhc_finalize(const mhc_weights &weights,
                            const float *input,
                            mtp_session::impl &session,
                            cudaStream_t stream) noexcept {
    if (weights.norm == nullptr || !weights.down || !weights.up) {
        return mtp_status::invalid_state;
    }
    grouped_rmsnorm_kernel<<<kMtpStreams, kThreads, 0u, stream>>>(
            input, weights.norm, session.mhc_normalized,
            session.compute_status);
    if (cudaPeekAtLastError() != cudaSuccess) return mtp_status::cuda_error;
    mtp_status status = run_linear(
            weights.down.get(), session.mhc_normalized, 1u,
            session.mhc_low_rank, session, stream);
    if (status != mtp_status::ok) return status;
    scaled_silu_kernel<<<1u, kThreads, 0u, stream>>>(session.mhc_low_rank);
    if (cudaPeekAtLastError() != cudaSuccess) return mtp_status::cuda_error;
    status = run_linear(weights.up.get(), session.mhc_low_rank, 1u,
                        session.mhc_mix_logits, session, stream);
    if (status != mtp_status::ok) return status;
    const unsigned blocks = static_cast<unsigned>(
            (kMtpHidden + kThreads - 1u) / kThreads);
    mhc_finalize_kernel<<<blocks, kThreads, 0u, stream>>>(
            session.mhc_normalized, session.mhc_mix_logits,
            session.final_hidden, session.compute_status);
    return cudaPeekAtLastError() == cudaSuccess ? mtp_status::ok
                                                : mtp_status::cuda_error;
}

mtp_status abort_transaction(mtp_session::impl &session,
                             cudaStream_t stream,
                             std::string *error,
                             mtp_status original,
                             const std::string &message) noexcept {
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    if (session.attention_cache.transaction_open) {
        const attention::status rollback =
                attention::rollback(&session.attention_cache);
        if (rollback != attention::status::ok) {
            session.poisoned = true;
            session.transaction_open = false;
            session.staged = false;
            return fail(mtp_status::invalid_state, error,
                        message + "; attention rollback failed");
        }
    }
    session.transaction_open = false;
    session.staged = false;
    return fail(original, error, message);
}

}  // namespace

mtp_provider::mtp_provider() = default;
mtp_provider::~mtp_provider() = default;
mtp_provider::mtp_provider(mtp_provider &&) noexcept = default;
mtp_provider &mtp_provider::operator=(mtp_provider &&) noexcept = default;

mtp_status mtp_provider::load(
        const checkpoint_catalog &catalog,
        const mtp_provider_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<mtp_provider> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(mtp_status::invalid_argument, error,
                    "null provider output");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const mtp_status config_status = mtp_validate_config(config);
    if (config_status != mtp_status::ok) {
        return fail(config_status, error, "invalid provider configuration");
    }
    mtp_contract_report contract{};
    const mtp_status contract_status =
            mtp_validate_checkpoint_contract(catalog, &contract, error);
    if (contract_status != mtp_status::ok) return contract_status;
    int device = -1;
    if (!current_sm120_device(&device)) {
        return fail(mtp_status::unsupported_device, error,
                    "the native MTP provider requires an active SM120 GPU");
    }
    try {
        auto result = std::make_unique<mtp_provider>();
        result->impl_ = std::make_unique<impl>();
        impl &state = *result->impl_;
        state.config = config;
        state.contract = contract;
        state.device = device;

        std::unique_ptr<checkpoint_catalog> reopened;
        std::string reopen_error;
        if (!checkpoint_catalog::open(catalog.model_root(), &reopened,
                                      &reopen_error)) {
            return fail(mtp_status::checkpoint_io_error, error,
                        reopen_error.empty() ? "failed to reopen pinned checkpoint"
                                             : reopen_error);
        }
        state.catalog = std::shared_ptr<checkpoint_catalog>(std::move(reopened));

        mtp_status status = load_bf16(
                *state.catalog, "mtp.pre_fc_norm_embedding.weight",
                {kMtpHidden}, &state.pre_fc_embedding_norm,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog, "mtp.pre_fc_norm_hidden.weight",
                {kMtpResidual}, &state.pre_fc_hidden_norm,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_linear(
                *state.catalog, "mtp.fc_embedding.weight",
                kMtpHidden, kMtpHidden, 1u, config,
                initialization_stream, &state.fc_embedding, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.fc_embedding->weight_bytes();
        status = load_linear(
                *state.catalog, "mtp.fc_hidden.weight",
                kMtpHidden, kMtpHidden, kMtpStreams, config,
                initialization_stream, &state.fc_hidden, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.fc_hidden->weight_bytes();

        status = load_mhc_weights(
                *state.catalog, "mtp.layers.0.attn_hyper_connection.", true,
                config, initialization_stream, &state.attention_mhc,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_mhc_weights(
                *state.catalog, "mtp.layers.0.mlp_hyper_connection.", true,
                config, initialization_stream, &state.mlp_mhc,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_mhc_weights(
                *state.catalog, "mtp.hyper_connection_mixer.", false,
                config, initialization_stream, &state.final_mhc,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;

        const std::string attention_prefix = "mtp.layers.0.self_attn.";
        status = load_linear(
                *state.catalog, attention_prefix + "q_proj.weight",
                kMtpHidden, kQProjection, 1u, config,
                initialization_stream, &state.q_projection, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.q_projection->weight_bytes();
        status = load_linear(
                *state.catalog, attention_prefix + "k_proj.weight",
                kMtpHidden, kKvProjection, 1u, config,
                initialization_stream, &state.k_projection, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.k_projection->weight_bytes();
        status = load_linear(
                *state.catalog, attention_prefix + "v_proj.weight",
                kMtpHidden, kKvProjection, 1u, config,
                initialization_stream, &state.v_projection, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.v_projection->weight_bytes();
        status = load_linear(
                *state.catalog,
                attention_prefix + "indexer.index_qk_proj.weight",
                kMtpHidden, kIndexProjection, 1u, config,
                initialization_stream, &state.index_projection, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.index_projection->weight_bytes();
        status = load_linear(
                *state.catalog, attention_prefix + "o_proj.weight",
                kAttentionOutput, kMtpHidden, 1u, config,
                initialization_stream, &state.o_projection, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.o_projection->weight_bytes();
        status = load_bf16(
                *state.catalog, attention_prefix + "q_norm.weight", {256u},
                &state.q_norm, &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog, attention_prefix + "k_norm.weight", {256u},
                &state.k_norm, &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog,
                attention_prefix + "indexer.q_layernorm.weight", {128u},
                &state.index_q_norm, &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog,
                attention_prefix + "indexer.k_layernorm.weight", {128u},
                &state.index_k_norm, &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;

        const std::string mlp_prefix = "mtp.layers.0.mlp.";
        status = load_linear(
                *state.catalog, mlp_prefix + "gate.weight",
                kMtpHidden, kMtpExperts, 1u, config,
                initialization_stream, &state.router, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.router->weight_bytes();
        status = load_bf16(
                *state.catalog, mlp_prefix + "shared_expert.gate_proj.weight",
                {kMtpIntermediate, kMtpHidden}, &state.shared_gate,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog, mlp_prefix + "shared_expert.up_proj.weight",
                {kMtpIntermediate, kMtpHidden}, &state.shared_up,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog, mlp_prefix + "shared_expert.down_proj.weight",
                {kMtpHidden, kMtpIntermediate}, &state.shared_down,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;
        status = load_bf16(
                *state.catalog, mlp_prefix + "shared_expert_gate.weight",
                {1u, kMtpHidden}, &state.shared_output_gate,
                &state.resident_bytes, error);
        if (status != mtp_status::ok) return status;

        status = load_linear(
                *state.catalog, "lm_head.weight", kMtpHidden, kMtpVocab,
                1u, config, initialization_stream, &state.lm_head, error);
        if (status != mtp_status::ok) return status;
        state.resident_bytes += state.lm_head->weight_bytes();
        if (cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(mtp_status::cuda_error, error,
                        "initialization stream failed");
        }
        state.ready = true;
        *out = std::move(result);
        return mtp_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(mtp_status::allocation_failure, error,
                    "host allocation failed while loading provider");
    } catch (const std::exception &exception) {
        return fail(mtp_status::checkpoint_io_error, error, exception.what());
    } catch (...) {
        return fail(mtp_status::checkpoint_io_error, error,
                    "unexpected provider load failure");
    }
}

mtp_status mtp_provider::create_session(
        std::size_t context_capacity,
        cudaStream_t initialization_stream,
        std::unique_ptr<mtp_session> *out,
        std::string *error) noexcept {
    if (out == nullptr || impl_ == nullptr || !impl_->ready) {
        return fail(mtp_status::invalid_argument, error,
                    "invalid session creation request");
    }
    out->reset();
    if (context_capacity < qsa::checkpoint_config.compress_ratio ||
        context_capacity > impl_->config.max_context) {
        return fail(mtp_status::capacity_exceeded, error,
                    "session context capacity is outside the admitted range");
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return fail(mtp_status::unsupported_device, error,
                    "session creation used the wrong CUDA device");
    }
    try {
        auto result = std::make_unique<mtp_session>();
        result->impl_ = std::make_unique<mtp_session::impl>();
        mtp_session::impl &session = *result->impl_;
        session.owner = impl_.get();
        session.device = impl_->device;
        session.bound_stream = initialization_stream;
        session.capacity = context_capacity;
        session.cached_experts.fill(UINT32_MAX);

        const std::size_t cache_elements = context_capacity *
                attention::kCheckpointKvHeads * attention::kCheckpointHeadDim;
        const std::size_t max_blocks =
                context_capacity / qsa::checkpoint_config.compress_ratio;
        if (!session.allocate_device(cache_elements, &session.key_cache, error,
                                     "MTP key cache") ||
            !session.allocate_device(cache_elements, &session.value_cache, error,
                                     "MTP value cache") ||
            !session.allocate_device(context_capacity * kIndexKey,
                                     &session.index_key_cache, error,
                                     "MTP index key cache") ||
            !session.allocate_device(kMtpResidual, &session.linear_input_bf16,
                                     error, "linear BF16 input") ||
            !session.allocate_bytes(impl_->config.blas_workspace_bytes,
                                    &session.blas_workspace, error,
                                    "cuBLAS workspace") ||
            !session.allocate_device(kMtpHidden, &session.normalized_embedding,
                                     error, "normalized embedding") ||
            !session.allocate_device(kMtpResidual, &session.normalized_hidden,
                                     error, "normalized target hidden") ||
            !session.allocate_device(kMtpHidden, &session.fc_embedding_output,
                                     error, "projected embedding") ||
            !session.allocate_device(kMtpResidual, &session.fc_hidden_output,
                                     error, "projected target hidden") ||
            !session.allocate_device(kMtpResidual, &session.fused_hidden,
                                     error, "MTP fused hidden") ||
            !session.allocate_device(kMtpResidual, &session.mhc_normalized,
                                     error, "mHC normalized") ||
            !session.allocate_device(kMtpRank, &session.mhc_low_rank,
                                     error, "mHC low rank") ||
            !session.allocate_device(kMtpResidual, &session.mhc_mix_logits,
                                     error, "mHC mix logits") ||
            !session.allocate_device(kMtpStreams, &session.mhc_injection_logits,
                                     error, "mHC injection logits") ||
            !session.allocate_device(kMtpHidden, &session.mhc_mixed,
                                     error, "mHC mixed input") ||
            !session.allocate_device(kMtpStreams, &session.mhc_injection,
                                     error, "mHC injection") ||
            !session.allocate_device(kMtpResidual, &session.attention_residual,
                                     error, "attention residual") ||
            !session.allocate_device(kMtpHidden, &session.qsa_output,
                                     error, "QSA output") ||
            !session.allocate_device(kMtpHidden, &session.moe_input,
                                     error, "MoE input") ||
            !session.allocate_device(kMtpHidden, &session.moe_output,
                                     error, "MoE output") ||
            !session.allocate_device(kQProjection, &session.q_projection,
                                     error, "Q projection") ||
            !session.allocate_device(kKvProjection, &session.k_projection,
                                     error, "K projection") ||
            !session.allocate_device(kKvProjection, &session.v_projection,
                                     error, "V projection") ||
            !session.allocate_device(kIndexProjection, &session.index_projection,
                                     error, "index projection") ||
            !session.allocate_device(kQProjection, &session.q_projection_bf16,
                                     error, "Q projection BF16") ||
            !session.allocate_device(kKvProjection, &session.k_projection_bf16,
                                     error, "K projection BF16") ||
            !session.allocate_device(kKvProjection, &session.v_projection_bf16,
                                     error, "V projection BF16") ||
            !session.allocate_device(kIndexQuery, &session.index_query_bf16,
                                     error, "index query BF16") ||
            !session.allocate_device(kIndexQuery, &session.index_query_prepared,
                                     error, "prepared index query") ||
            !session.allocate_device(context_capacity, &session.visible_indices,
                                     error, "visible indices") ||
            !session.allocate_device(max_blocks * kIndexKey,
                                     &session.pooled_index_keys, error,
                                     "pooled index keys") ||
            !session.allocate_device(max_blocks, &session.index_scores,
                                     error, "index scores") ||
            !session.allocate_device(kSelectedCapacity,
                                     &session.selected_indices, error,
                                     "selected indices") ||
            !session.allocate_device(1u, &session.selected_count, error,
                                     "selected count") ||
            !session.allocate_device(kAttentionOutput,
                                     &session.attention_output_bf16, error,
                                     "attention output BF16") ||
            !session.allocate_device(kAttentionOutput,
                                     &session.attention_output_f32, error,
                                     "attention output F32") ||
            !session.allocate_device(1u, &session.qsa_status, error,
                                     "QSA status") ||
            !session.allocate_device(1u, &session.attention_status, error,
                                     "attention status") ||
            !session.allocate_device(kMtpExperts, &session.router_logits, error,
                                     "router logits") ||
            !session.allocate_device(kMtpTopK, &session.route_indices, error,
                                     "route indices") ||
            !session.allocate_device(kMtpTopK, &session.route_weights, error,
                                     "route weights") ||
            !session.allocate_device(1u, &session.router_status, error,
                                     "router status") ||
            !session.allocate_device(kMtpTopK * kExpertGateUpElements,
                                     &session.expert_gate_up_slots, error,
                                     "BF16 gate/up expert slots") ||
            !session.allocate_device(kMtpTopK * kExpertDownElements,
                                     &session.expert_down_slots, error,
                                     "BF16 down expert slots") ||
            !session.allocate_device(kMtpTopK * kMtpIntermediate,
                                     &session.expert_intermediate, error,
                                     "expert intermediate") ||
            !session.allocate_device(kMtpHidden, &session.routed_output, error,
                                     "routed output") ||
            !session.allocate_device(kMtpIntermediate,
                                     &session.shared_intermediate, error,
                                     "shared intermediate") ||
            !session.allocate_device(1u, &session.shared_gate_value, error,
                                     "shared output gate") ||
            !session.allocate_device(1u, &session.shared_status, error,
                                     "shared expert status") ||
            !session.allocate_device(kMtpHidden, &session.shared_output, error,
                                     "shared expert output") ||
            !session.allocate_device(kMtpHidden, &session.final_hidden, error,
                                     "final MTP hidden") ||
            !session.allocate_device(kMtpVocab, &session.logits, error,
                                     "MTP logits") ||
            !session.allocate_device(1u, &session.selected_token, error,
                                     "selected token") ||
            !session.allocate_device(1u, &session.selected_logit, error,
                                     "selected logit") ||
            !session.allocate_device(1u, &session.compute_status, error,
                                     "MTP compute status")) {
            return mtp_status::allocation_failure;
        }

        if (qsa::cuda_select_workspace_bytes(
                    qsa::checkpoint_config, max_blocks,
                    &session.selection_workspace_bytes) != qsa::status::ok ||
            session.selection_workspace_bytes == 0u ||
            attention::workspace_bytes(
                    attention::checkpoint_config, 1u,
                    &session.attention_workspace_bytes) != attention::status::ok ||
            session.attention_workspace_bytes == 0u ||
            !session.allocate_bytes(session.selection_workspace_bytes,
                                    &session.selection_workspace, error,
                                    "QSA selection workspace") ||
            !session.allocate_bytes(session.attention_workspace_bytes,
                                    &session.attention_workspace, error,
                                    "attention workspace")) {
            return fail(mtp_status::allocation_failure, error,
                        "failed to allocate QSA workspaces");
        }
        if (!session.allocate_host(sizeof(host_control_block),
                                   reinterpret_cast<void **>(&session.host_control),
                                   error, "MTP host control") ||
            !session.allocate_host(kMtpTopK * kExpertGateUpBytes,
                                   reinterpret_cast<void **>(
                                           &session.host_gate_up_slots),
                                   error, "pinned gate/up expert slots") ||
            !session.allocate_host(kMtpTopK * kExpertDownBytes,
                                   reinterpret_cast<void **>(
                                           &session.host_down_slots),
                                   error, "pinned down expert slots")) {
            return mtp_status::allocation_failure;
        }
        *session.host_control = {};

        const attention::status cache_status = attention::cache_initialize(
                &session.attention_cache, session.key_cache,
                session.value_cache, context_capacity, 0u);
        if (cache_status != attention::status::ok) {
            return fail(map_attention(cache_status), error,
                        "attention cache initialization failed");
        }
        if (cudaMemsetAsync(session.compute_status, 0, sizeof(std::uint32_t),
                            initialization_stream) != cudaSuccess ||
            cudaMemsetAsync(session.router_status, 0, sizeof(std::uint32_t),
                            initialization_stream) != cudaSuccess ||
            cudaMemsetAsync(session.shared_status, 0, sizeof(std::uint32_t),
                            initialization_stream) != cudaSuccess ||
            cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(mtp_status::cuda_error, error,
                        "session initialization stream failed");
        }
        session.ready = true;
        *out = std::move(result);
        return mtp_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(mtp_status::allocation_failure, error,
                    "host allocation failed while creating session");
    } catch (...) {
        return fail(mtp_status::allocation_failure, error,
                    "unexpected session creation failure");
    }
}

const mtp_provider_config &mtp_provider::config() const noexcept {
    static const mtp_provider_config empty{};
    return impl_ ? impl_->config : empty;
}

mtp_contract_report mtp_provider::contract() const noexcept {
    return impl_ ? impl_->contract : mtp_contract_report{};
}

std::uint64_t mtp_provider::resident_weight_bytes() const noexcept {
    return impl_ ? impl_->resident_bytes : 0u;
}

int mtp_provider::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool mtp_provider::initialized() const noexcept {
    return impl_ && impl_->ready;
}

mtp_session::mtp_session() = default;
mtp_session::~mtp_session() = default;
mtp_session::mtp_session(mtp_session &&) noexcept = default;
mtp_session &mtp_session::operator=(mtp_session &&) noexcept = default;

mtp_status mtp_session::stage_token(
        const float *input_embedding_2560_f32,
        const float *target_hidden_4x2560_f32,
        const float *full_cos_64_f32,
        const float *full_sin_64_f32,
        std::size_t position_count,
        float *output_hidden_4x2560_f32,
        mtp_prediction *prediction,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (prediction != nullptr) *prediction = {};
    if (impl_ == nullptr || !impl_->ready || impl_->owner == nullptr ||
        !impl_->owner->ready || prediction == nullptr ||
        input_embedding_2560_f32 == nullptr ||
        target_hidden_4x2560_f32 == nullptr ||
        full_cos_64_f32 == nullptr || full_sin_64_f32 == nullptr ||
        output_hidden_4x2560_f32 == nullptr) {
        return fail(mtp_status::invalid_argument, error,
                    "invalid stage_token arguments");
    }
    impl &session = *impl_;
    if (session.poisoned || session.transaction_open || session.staged) {
        return fail(mtp_status::invalid_state, error,
                    "session is poisoned or already has an open transaction");
    }
    if (stream != session.bound_stream) {
        return fail(mtp_status::invalid_state, error,
                    "MTP session stream changed without an event handoff");
    }
    if (session.committed >= session.capacity ||
        position_count <= session.committed ||
        position_count > session.capacity) {
        return fail(mtp_status::capacity_exceeded, error,
                    "MTP position exceeds session cache capacity");
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != session.device) {
        return fail(mtp_status::unsupported_device, error,
                    "stage_token used the wrong CUDA device");
    }
    std::size_t rope_elements = 0u;
    if (!checked_mul(position_count,
                     attention::kCheckpointRotaryDim,
                     &rope_elements) ||
        !device_range_valid(input_embedding_2560_f32,
                            kMtpHidden * sizeof(float), session.device) ||
        !device_range_valid(target_hidden_4x2560_f32,
                            kMtpResidual * sizeof(float), session.device) ||
        !device_range_valid(output_hidden_4x2560_f32,
                            kMtpResidual * sizeof(float), session.device) ||
        !device_range_valid(full_cos_64_f32,
                            rope_elements * sizeof(float), session.device) ||
        !device_range_valid(full_sin_64_f32,
                            rope_elements * sizeof(float), session.device)) {
        return fail(mtp_status::invalid_device_pointer, error,
                    "stage_token received an invalid device range");
    }

    mtp_status status = map_qsa(qsa::cuda_reset_status(
            session.qsa_status, stream));
    if (status != mtp_status::ok) return status;
    status = map_attention(attention::reset_device_status(
            session.attention_status, stream));
    if (status != mtp_status::ok) return status;
    if (cudaMemsetAsync(session.compute_status, 0, sizeof(std::uint32_t),
                        stream) != cudaSuccess ||
        cudaMemsetAsync(session.router_status, 0, sizeof(std::uint32_t),
                        stream) != cudaSuccess ||
        cudaMemsetAsync(session.shared_status, 0, sizeof(std::uint32_t),
                        stream) != cudaSuccess) {
        return fail(mtp_status::cuda_error, error,
                    "failed to reset MTP device status");
    }
    const attention::status begin =
            attention::begin_transaction(&session.attention_cache);
    if (begin != attention::status::ok) {
        return fail(map_attention(begin), error,
                    "failed to begin MTP attention transaction");
    }
    session.transaction_open = true;

    gemma_rmsnorm_kernel<<<1u, kThreads, 0u, stream>>>(
            input_embedding_2560_f32,
            session.owner->pre_fc_embedding_norm,
            kMtpHidden, 1u, session.normalized_embedding,
            session.compute_status);
    gemma_rmsnorm_kernel<<<1u, kThreads, 0u, stream>>>(
            target_hidden_4x2560_f32,
            session.owner->pre_fc_hidden_norm,
            kMtpResidual, 1u, session.normalized_hidden,
            session.compute_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error, mtp_status::cuda_error,
                                 "MTP input normalization launch failed");
    }
    status = run_linear(session.owner->fc_embedding.get(),
                        session.normalized_embedding, 1u,
                        session.fc_embedding_output, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP embedding projection failed");
    }
    status = run_linear(session.owner->fc_hidden.get(),
                        session.normalized_hidden, kMtpStreams,
                        session.fc_hidden_output, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP target-hidden projection failed");
    }
    fusion_add_kernel<<<
            static_cast<unsigned>((kMtpResidual + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.fc_embedding_output, session.fc_hidden_output,
            session.fused_hidden);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error, mtp_status::cuda_error,
                                 "MTP fusion launch failed");
    }

    status = run_mhc_prepare(session.owner->attention_mhc,
                             session.fused_hidden, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP attention mHC prepare failed");
    }

    status = run_linear(session.owner->q_projection.get(), session.mhc_mixed,
                        1u, session.q_projection, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP Q projection failed");
    }
    projection_to_bf16_kernel<<<
            static_cast<unsigned>((kQProjection + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.q_projection, session.q_projection_bf16,
            kQProjection, session.attention_status);
    status = run_linear(session.owner->k_projection.get(), session.mhc_mixed,
                        1u, session.k_projection, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP K projection failed");
    }
    projection_to_bf16_kernel<<<
            static_cast<unsigned>((kKvProjection + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.k_projection, session.k_projection_bf16,
            kKvProjection, session.attention_status);
    status = run_linear(session.owner->v_projection.get(), session.mhc_mixed,
                        1u, session.v_projection, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP V projection failed");
    }
    projection_to_bf16_kernel<<<
            static_cast<unsigned>((kKvProjection + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.v_projection, session.v_projection_bf16,
            kKvProjection, session.attention_status);
    status = run_linear(session.owner->index_projection.get(),
                        session.mhc_mixed, 1u, session.index_projection,
                        session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP QSA index projection failed");
    }
    split_index_projection_kernel<<<1u, kIndexProjection, 0u, stream>>>(
            session.index_projection, session.committed,
            session.index_query_bf16, session.index_key_cache,
            session.qsa_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error, mtp_status::cuda_error,
                                 "MTP projection conversion failed");
    }

    attention::staged_batch staged_batch{};
    status = map_attention(attention::stage_cuda(
            attention::checkpoint_config, &session.attention_cache,
            session.q_projection_bf16, session.k_projection_bf16,
            session.v_projection_bf16, session.owner->q_norm,
            session.owner->k_norm,
            full_cos_64_f32 +
                    session.committed * attention::kCheckpointRotaryDim,
            full_sin_64_f32 +
                    session.committed * attention::kCheckpointRotaryDim,
            1u, session.attention_workspace,
            session.attention_workspace_bytes, &staged_batch,
            session.attention_status, stream));
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP attention cache staging failed");
    }

    status = map_qsa(qsa::prepare_queries_cuda_bf16(
            qsa::checkpoint_config,
            reinterpret_cast<const __nv_bfloat16 *>(
                    session.index_query_bf16),
            reinterpret_cast<const __nv_bfloat16 *>(
                    session.owner->index_q_norm),
            full_cos_64_f32 +
                    session.committed * attention::kCheckpointRotaryDim,
            full_sin_64_f32 +
                    session.committed * attention::kCheckpointRotaryDim,
            1u, session.index_query_prepared, session.qsa_status, stream));
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP QSA query preparation failed");
    }
    const std::size_t visible_count = session.committed + 1u;
    fill_indices_kernel<<<
            static_cast<unsigned>((visible_count + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(session.visible_indices, visible_count);
    std::size_t block_count = 0u;
    status = map_qsa(qsa::complete_block_count(
            qsa::checkpoint_config, visible_count, &block_count));
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP QSA block count failed");
    }
    std::size_t produced_blocks = 0u;
    status = map_qsa(qsa::pool_keys_cuda_bf16(
            qsa::checkpoint_config,
            reinterpret_cast<const __nv_bfloat16 *>(session.index_key_cache),
            visible_count, session.visible_indices, visible_count,
            reinterpret_cast<const __nv_bfloat16 *>(
                    session.owner->index_k_norm),
            full_cos_64_f32, full_sin_64_f32, position_count,
            session.pooled_index_keys,
            session.capacity / qsa::checkpoint_config.compress_ratio,
            &produced_blocks, session.qsa_status, stream));
    if (status != mtp_status::ok || produced_blocks != block_count) {
        return abort_transaction(
                session, stream, error,
                status == mtp_status::ok ? mtp_status::qsa_error : status,
                "MTP QSA key pooling failed");
    }
    status = map_qsa(qsa::score_blocks_cuda_f32(
            qsa::checkpoint_config, session.index_query_prepared,
            session.pooled_index_keys, block_count, session.index_scores,
            session.qsa_status, stream));
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP QSA score computation failed");
    }
    std::size_t selected_count_host = 0u;
    status = map_qsa(qsa::select_tokens_cuda_f32(
            qsa::checkpoint_config, session.index_scores, block_count,
            session.visible_indices, visible_count, visible_count,
            session.selected_indices, kSelectedCapacity,
            session.selection_workspace, session.selection_workspace_bytes,
            &selected_count_host, session.qsa_status, stream));
    if (status != mtp_status::ok || selected_count_host == 0u ||
        selected_count_host > kSelectedCapacity ||
        selected_count_host > std::numeric_limits<std::uint32_t>::max()) {
        return abort_transaction(
                session, stream, error,
                status == mtp_status::ok ? mtp_status::qsa_error : status,
                "MTP QSA token selection failed");
    }
    store_count_kernel<<<1u, 1u, 0u, stream>>>(
            session.selected_count,
            static_cast<std::uint32_t>(selected_count_host));
    status = map_attention(attention::forward_qsa_cuda(
            attention::checkpoint_config, &session.attention_cache,
            staged_batch, session.selected_indices, session.selected_count,
            kSelectedCapacity, session.attention_output_bf16,
            session.attention_status, stream));
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP sparse attention failed");
    }
    bf16_to_f32_kernel<<<
            static_cast<unsigned>((kAttentionOutput + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.attention_output_bf16, session.attention_output_f32,
            kAttentionOutput, session.attention_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error, mtp_status::cuda_error,
                                 "MTP attention output conversion failed");
    }
    status = run_linear(session.owner->o_projection.get(),
                        session.attention_output_f32, 1u,
                        session.qsa_output, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP attention output projection failed");
    }
    const MhcStatus attention_reinject = mhc_reinject_cuda(
            mhc_qwen4_exp_config(), session.fused_hidden,
            session.qsa_output, session.mhc_injection, 1u,
            session.attention_residual, stream);
    if (attention_reinject != MhcStatus::kOk) {
        return abort_transaction(session, stream, error,
                                 mtp_status::cuda_error,
                                 "MTP attention mHC reinjection failed");
    }

    status = run_mhc_prepare(session.owner->mlp_mhc,
                             session.attention_residual, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP MLP mHC prepare failed");
    }
    if (cudaMemcpyAsync(session.moe_input, session.mhc_mixed,
                        kMtpHidden * sizeof(float), cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
        return abort_transaction(session, stream, error, mtp_status::cuda_error,
                                 "MTP MoE input staging failed");
    }
    status = run_linear(session.owner->router.get(), session.moe_input, 1u,
                        session.router_logits, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP router projection failed");
    }
    const moe_status route_status = moe_router_topk_cuda(
            moe_qwen4_exp_config(), session.router_logits,
            session.route_indices, session.route_weights,
            session.router_status, stream);
    if (route_status != moe_status::ok ||
        cudaMemcpyAsync(session.host_control->route_indices.data(),
                        session.route_indices,
                        kMtpTopK * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(session.host_control->route_weights.data(),
                        session.route_weights,
                        kMtpTopK * sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->router_status,
                        session.router_status, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->compute_status,
                        session.compute_status, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return abort_transaction(session, stream, error,
                                 mtp_status::router_error,
                                 "MTP route collection failed");
    }
    if (session.host_control->router_status !=
            static_cast<std::uint32_t>(moe_status::ok)) {
        return abort_transaction(session, stream, error,
                                 mtp_status::router_error,
                                 "MTP router rejected non-finite logits");
    }
    if (session.host_control->compute_status != 0u) {
        return abort_transaction(session, stream, error,
                                 mtp_status::non_finite,
                                 "MTP pre-route graph produced non-finite data");
    }

    std::array<bool, kMtpTopK> slot_miss{};
    std::uint64_t nvme_bytes = 0u;
    std::uint32_t cache_hits = 0u;
    const std::string gate_up_name =
            "mtp.layers.0.mlp.experts.gate_up_proj";
    const std::string down_name =
            "mtp.layers.0.mlp.experts.down_proj";
    for (std::size_t slot = 0u; slot < kMtpTopK; ++slot) {
        const std::uint32_t expert =
                session.host_control->route_indices[slot];
        if (expert >= kMtpExperts) {
            return abort_transaction(session, stream, error,
                                     mtp_status::router_error,
                                     "MTP router returned an invalid expert id");
        }
        if (session.cached_experts[slot] == expert) {
            ++cache_hits;
            continue;
        }
        slot_miss[slot] = true;
        std::string read_error;
        if (!session.owner->catalog->read_range(
                    gate_up_name,
                    static_cast<std::uint64_t>(expert) * kExpertGateUpBytes,
                    session.host_gate_up_slots + slot * kExpertGateUpElements,
                    kExpertGateUpBytes, &read_error) ||
            !session.owner->catalog->read_range(
                    down_name,
                    static_cast<std::uint64_t>(expert) * kExpertDownBytes,
                    session.host_down_slots + slot * kExpertDownElements,
                    kExpertDownBytes, &read_error)) {
            return abort_transaction(
                    session, stream, error, mtp_status::expert_paging_error,
                    read_error.empty() ? "MTP expert NVMe read failed"
                                       : read_error);
        }
        nvme_bytes += kExpertGateUpBytes + kExpertDownBytes;
        if (cudaMemcpyAsync(
                    session.expert_gate_up_slots +
                            slot * kExpertGateUpElements,
                    session.host_gate_up_slots +
                            slot * kExpertGateUpElements,
                    kExpertGateUpBytes, cudaMemcpyHostToDevice, stream) !=
                    cudaSuccess ||
            cudaMemcpyAsync(
                    session.expert_down_slots + slot * kExpertDownElements,
                    session.host_down_slots + slot * kExpertDownElements,
                    kExpertDownBytes, cudaMemcpyHostToDevice, stream) !=
                    cudaSuccess) {
            return abort_transaction(session, stream, error,
                                     mtp_status::expert_paging_error,
                                     "MTP expert GPU upload failed");
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return abort_transaction(session, stream, error,
                                 mtp_status::expert_paging_error,
                                 "MTP expert paging stream failed");
    }
    for (std::size_t slot = 0u; slot < kMtpTopK; ++slot) {
        if (slot_miss[slot]) {
            session.cached_experts[slot] =
                    session.host_control->route_indices[slot];
        }
    }

    expert_gate_up_kernel<<<
            static_cast<unsigned>(kMtpTopK * kMtpIntermediate),
            kThreads, 0u, stream>>>(
            session.expert_gate_up_slots, session.moe_input,
            session.expert_intermediate, session.compute_status);
    expert_down_kernel<<<kMtpHidden, kThreads, 0u, stream>>>(
            session.expert_down_slots, session.expert_intermediate,
            session.route_weights, session.routed_output,
            session.compute_status);
    const shared_expert_weights_bf16 shared_weights{
        session.owner->shared_gate,
        session.owner->shared_up,
        session.owner->shared_down,
        session.owner->shared_output_gate,
    };
    const shared_expert_workspace_f32 shared_workspace{
        session.shared_intermediate,
        session.shared_gate_value,
        session.shared_status,
    };
    const shared_expert_status shared_launch =
            shared_expert_forward_f32_cuda(
                    session.device, shared_expert_qwen4_exp_config(),
                    shared_weights, session.moe_input, shared_workspace,
                    session.shared_output, stream);
    if (shared_launch != shared_expert_status::ok ||
        cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error,
                                 mtp_status::expert_compute_error,
                                 "MTP BF16 expert launch failed");
    }
    vector_sum_kernel<<<
            static_cast<unsigned>((kMtpHidden + kThreads - 1u) / kThreads),
            kThreads, 0u, stream>>>(
            session.routed_output, session.shared_output,
            session.moe_output, session.compute_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return abort_transaction(session, stream, error,
                                 mtp_status::expert_compute_error,
                                 "MTP expert output sum failed");
    }
    const MhcStatus mlp_reinject = mhc_reinject_cuda(
            mhc_qwen4_exp_config(), session.attention_residual,
            session.moe_output, session.mhc_injection, 1u,
            output_hidden_4x2560_f32, stream);
    if (mlp_reinject != MhcStatus::kOk) {
        return abort_transaction(session, stream, error,
                                 mtp_status::cuda_error,
                                 "MTP MLP mHC reinjection failed");
    }
    status = run_mhc_finalize(session.owner->final_mhc,
                              output_hidden_4x2560_f32, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP final mixer failed");
    }
    status = run_linear(session.owner->lm_head.get(), session.final_hidden,
                        1u, session.logits, session, stream);
    if (status != mtp_status::ok) {
        return abort_transaction(session, stream, error, status,
                                 "MTP shared LM head failed");
    }
    argmax_kernel<<<1u, kThreads, 0u, stream>>>(
            session.logits, session.selected_token,
            session.selected_logit, session.compute_status);
    if (cudaPeekAtLastError() != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->token,
                        session.selected_token, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->selected_logit,
                        session.selected_logit, sizeof(float),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->compute_status,
                        session.compute_status, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&session.host_control->shared_status,
                        session.shared_status, sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return abort_transaction(session, stream, error,
                                 mtp_status::cuda_error,
                                 "MTP output collection failed");
    }

    qsa::status collected_qsa = qsa::status::cuda_failure;
    if (qsa::cuda_collect_status(session.qsa_status, stream,
                                 &collected_qsa) != qsa::status::ok ||
        collected_qsa != qsa::status::ok) {
        return abort_transaction(session, stream, error,
                                 mtp_status::qsa_error,
                                 "MTP QSA device status failed");
    }
    attention::status collected_attention = attention::status::cuda_failure;
    if (attention::collect_device_status(
                session.attention_status, &collected_attention, stream) !=
                attention::status::ok ||
        collected_attention != attention::status::ok) {
        return abort_transaction(session, stream, error,
                                 mtp_status::attention_error,
                                 "MTP attention device status failed");
    }
    if (session.host_control->compute_status != 0u ||
        session.host_control->shared_status !=
                static_cast<std::uint32_t>(shared_expert_status::ok) ||
        !std::isfinite(session.host_control->selected_logit)) {
        return abort_transaction(session, stream, error,
                                 mtp_status::non_finite,
                                 "MTP graph failed its finite-output gate");
    }

    prediction->token_id = session.host_control->token;
    prediction->selected_logit = session.host_control->selected_logit;
    prediction->experts = session.host_control->route_indices;
    prediction->router_weights = session.host_control->route_weights;
    prediction->nvme_bytes_read = nvme_bytes;
    prediction->expert_cache_hits = cache_hits;
    session.staged = true;
    return mtp_status::ok;
}

mtp_status mtp_session::commit(cudaStream_t stream,
                               std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        !impl_->transaction_open || !impl_->staged ||
        stream != impl_->bound_stream) {
        return fail(mtp_status::invalid_state, error,
                    "no prepared MTP transaction to commit");
    }
    const attention::status status =
            attention::commit_prefix(&impl_->attention_cache, 1u);
    if (status != attention::status::ok) {
        impl_->poisoned = true;
        return fail(map_attention(status), error,
                    "MTP attention commit failed");
    }
    ++impl_->committed;
    impl_->transaction_open = false;
    impl_->staged = false;
    return mtp_status::ok;
}

mtp_status mtp_session::rollback(cudaStream_t stream,
                                 std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        !impl_->transaction_open || stream != impl_->bound_stream) {
        return fail(mtp_status::invalid_state, error,
                    "no MTP transaction to roll back");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        impl_->poisoned = true;
        return fail(mtp_status::cuda_error, error,
                    "MTP rollback stream failed");
    }
    const attention::status status =
            attention::rollback(&impl_->attention_cache);
    if (status != attention::status::ok) {
        impl_->poisoned = true;
        return fail(map_attention(status), error,
                    "MTP attention rollback failed");
    }
    impl_->transaction_open = false;
    impl_->staged = false;
    return mtp_status::ok;
}

mtp_status mtp_session::truncate_committed(
        std::size_t committed_tokens,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        impl_->transaction_open || impl_->staged ||
        stream != impl_->bound_stream || committed_tokens > impl_->committed) {
        return fail(mtp_status::invalid_state, error,
                    "invalid MTP committed-tail truncation");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        impl_->poisoned = true;
        return fail(mtp_status::cuda_error, error,
                    "MTP truncation stream failed");
    }
    if (!impl_->attention_cache.initialized ||
        impl_->attention_cache.transaction_open ||
        impl_->attention_cache.staged_tokens != 0u ||
        impl_->attention_cache.committed_tokens != impl_->committed) {
        impl_->poisoned = true;
        return fail(mtp_status::invalid_state, error,
                    "MTP cache metadata diverged before truncation");
    }
    impl_->attention_cache.committed_tokens = committed_tokens;
    impl_->committed = committed_tokens;
    return mtp_status::ok;
}

mtp_status mtp_session::reset(cudaStream_t stream,
                              std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || stream != impl_->bound_stream) {
        return fail(mtp_status::invalid_state, error,
                    "invalid MTP reset request");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        impl_->poisoned = true;
        return fail(mtp_status::cuda_error, error,
                    "MTP reset stream failed");
    }
    if (impl_->attention_cache.transaction_open &&
        attention::rollback(&impl_->attention_cache) != attention::status::ok) {
        impl_->poisoned = true;
        return fail(mtp_status::invalid_state, error,
                    "MTP reset rollback failed");
    }
    if (attention::cache_reset(&impl_->attention_cache) !=
        attention::status::ok) {
        impl_->poisoned = true;
        return fail(mtp_status::invalid_state, error,
                    "MTP attention cache reset failed");
    }
    impl_->committed = 0u;
    impl_->transaction_open = false;
    impl_->staged = false;
    impl_->poisoned = false;
    return mtp_status::ok;
}

mtp_session_state mtp_session::state() const noexcept {
    mtp_session_state result{};
    if (impl_ != nullptr) {
        result.context_capacity = impl_->capacity;
        result.committed_tokens = impl_->committed;
        result.transaction_open = impl_->transaction_open;
        result.staged = impl_->staged;
        result.poisoned = impl_->poisoned;
    }
    return result;
}

bool mtp_session::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
