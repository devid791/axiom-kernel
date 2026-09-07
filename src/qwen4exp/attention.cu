#include "axiom/qwen4exp/attention.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp::attention {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256u;

static_assert(sizeof(status) == sizeof(std::uint32_t));

[[nodiscard]] bool add_overflow(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* output) noexcept {
    if (output == nullptr || lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        return true;
    }
    *output = lhs + rhs;
    return false;
}

[[nodiscard]] bool multiply_overflow(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* output) noexcept {
    if (output == nullptr ||
        (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return true;
    }
    *output = lhs * rhs;
    return false;
}

[[nodiscard]] bool is_power_of_two(std::uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] bool same_config(const config& lhs, const config& rhs) noexcept {
    return lhs.query_heads == rhs.query_heads && lhs.kv_heads == rhs.kv_heads &&
           lhs.head_dim == rhs.head_dim && lhs.rotary_dim == rhs.rotary_dim &&
           lhs.max_context == rhs.max_context &&
           lhs.max_selected_tokens == rhs.max_selected_tokens &&
           lhs.rms_epsilon == rhs.rms_epsilon;
}

[[nodiscard]] status cuda_status(cudaError_t value) noexcept {
    return value == cudaSuccess ? status::ok : status::cuda_failure;
}

[[nodiscard]] status visible_tokens(
    const cache_state& state,
    std::size_t* output) noexcept {
    if (output == nullptr) {
        return status::null_pointer;
    }
    if (add_overflow(state.committed_tokens, state.staged_tokens, output)) {
        return status::arithmetic_overflow;
    }
    return status::ok;
}

[[nodiscard]] status validate_cache(const cache_state* state) noexcept {
    if (state == nullptr) {
        return status::null_pointer;
    }
    if (!state->initialized || state->key_bf16 == nullptr ||
        state->value_bf16 == nullptr || state->capacity_tokens == 0u ||
        state->capacity_tokens > kCheckpointMaxContext) {
        return status::invalid_state;
    }
    std::size_t visible = 0u;
    const status length_status = visible_tokens(*state, &visible);
    if (length_status != status::ok) {
        return length_status;
    }
    if (visible > state->capacity_tokens) {
        return status::invalid_state;
    }
    if (!state->transaction_open && state->staged_tokens != 0u) {
        return status::invalid_state;
    }
    return status::ok;
}

[[nodiscard]] std::uintptr_t align_up(std::uintptr_t value) noexcept {
    return (value + (kWorkspaceAlignment - 1u)) &
           ~(static_cast<std::uintptr_t>(kWorkspaceAlignment) - 1u);
}

__device__ float load_bf16(const std::uint16_t* values, std::size_t index) {
    return __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(values)[index]);
}

__device__ void store_bf16(
    std::uint16_t* values,
    std::size_t index,
    float value) {
    reinterpret_cast<__nv_bfloat16*>(values)[index] = __float2bfloat16_rn(value);
}

__device__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ void report_error(status* output, status value) {
    if (output != nullptr) {
        atomicCAS(
            reinterpret_cast<unsigned int*>(output),
            static_cast<unsigned int>(status::ok),
            static_cast<unsigned int>(value));
    }
}

__device__ float reduce_sum(float value, float* shared) {
    const unsigned int lane = threadIdx.x;
    shared[lane] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            shared[lane] += shared[lane + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void prepare_query_kernel(
    config value,
    const std::uint16_t* q_proj,
    const std::uint16_t* norm_weight,
    const float* cosine,
    const float* sine,
    std::size_t row_count,
    float* prepared_queries,
    status* device_status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= row_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::size_t token = row / value.query_heads;
    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = shared + value.head_dim;
    __shared__ unsigned int valid;
    if (lane == 0u) {
        valid = 1u;
    }
    __syncthreads();

    const std::size_t q_offset = row * (2u * value.head_dim);
    float raw = load_bf16(q_proj, q_offset + lane);
    const float weight = load_bf16(norm_weight, lane);
    if (!isfinite(raw) || !isfinite(weight)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        raw = 0.0F;
    }
    const float square_sum = reduce_sum(raw * raw, reduction);
    float inverse_rms = rsqrtf(
        square_sum / static_cast<float>(value.head_dim) + value.rms_epsilon);
    if (!isfinite(inverse_rms)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        inverse_rms = 0.0F;
    }
    float normalized_value = raw * inverse_rms * (1.0F + weight);
    normalized_value = round_bf16(normalized_value);
    if (!isfinite(normalized_value)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        normalized_value = 0.0F;
    }
    normalized[lane] = normalized_value;
    __syncthreads();

    float output = normalized_value;
    if (lane < value.rotary_dim) {
        const std::uint32_t half = value.rotary_dim / 2u;
        const std::uint32_t pair = lane < half ? lane + half : lane - half;
        const float cos_value = cosine[token * value.rotary_dim + lane];
        const float sin_value = sine[token * value.rotary_dim + lane];
        if (!isfinite(cos_value) || !isfinite(sin_value)) {
            report_error(device_status, status::non_finite);
            atomicExch(&valid, 0u);
            output = 0.0F;
        } else {
            const float rotated_half =
                lane < half ? -normalized[pair] : normalized[pair];
            output = normalized_value * cos_value + rotated_half * sin_value;
        }
    }
    if (!isfinite(output)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        output = 0.0F;
    }
    __syncthreads();
    prepared_queries[row * value.head_dim + lane] = valid != 0u ? output : 0.0F;
}

__global__ void prepare_kv_kernel(
    config value,
    const std::uint16_t* k_proj,
    const std::uint16_t* v_proj,
    const std::uint16_t* norm_weight,
    const float* cosine,
    const float* sine,
    std::size_t token_count,
    std::size_t cache_start,
    std::uint16_t* key_cache,
    std::uint16_t* value_cache,
    status* device_status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    const std::size_t row_count = token_count * value.kv_heads;
    if (row >= row_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::size_t token = row / value.kv_heads;
    const std::size_t kv_head = row % value.kv_heads;
    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = shared + value.head_dim;
    __shared__ unsigned int valid;
    if (lane == 0u) {
        valid = 1u;
    }
    __syncthreads();

    const std::size_t source = row * value.head_dim + lane;
    float raw_key = load_bf16(k_proj, source);
    float raw_value = load_bf16(v_proj, source);
    const float weight = load_bf16(norm_weight, lane);
    if (!isfinite(raw_key) || !isfinite(raw_value) || !isfinite(weight)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        raw_key = 0.0F;
        raw_value = 0.0F;
    }
    const float square_sum = reduce_sum(raw_key * raw_key, reduction);
    float inverse_rms = rsqrtf(
        square_sum / static_cast<float>(value.head_dim) + value.rms_epsilon);
    if (!isfinite(inverse_rms)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        inverse_rms = 0.0F;
    }
    float normalized_value = raw_key * inverse_rms * (1.0F + weight);
    normalized_value = round_bf16(normalized_value);
    if (!isfinite(normalized_value)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        normalized_value = 0.0F;
    }
    normalized[lane] = normalized_value;
    __syncthreads();

    float output_key = normalized_value;
    if (lane < value.rotary_dim) {
        const std::uint32_t half = value.rotary_dim / 2u;
        const std::uint32_t pair = lane < half ? lane + half : lane - half;
        const float cos_value = cosine[token * value.rotary_dim + lane];
        const float sin_value = sine[token * value.rotary_dim + lane];
        if (!isfinite(cos_value) || !isfinite(sin_value)) {
            report_error(device_status, status::non_finite);
            atomicExch(&valid, 0u);
            output_key = 0.0F;
        } else {
            const float rotated_half =
                lane < half ? -normalized[pair] : normalized[pair];
            output_key = normalized_value * cos_value + rotated_half * sin_value;
        }
    }
    if (!isfinite(output_key)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
        output_key = 0.0F;
    }
    __syncthreads();

    const std::size_t destination =
        ((cache_start + token) * value.kv_heads + kv_head) * value.head_dim + lane;
    if (valid == 0u) {
        output_key = 0.0F;
        raw_value = 0.0F;
    }
    store_bf16(key_cache, destination, output_key);
    store_bf16(value_cache, destination, raw_value);
}

__device__ float stable_sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + expf(-value));
    }
    const float exponent = expf(value);
    return exponent / (1.0F + exponent);
}

__global__ void qsa_attention_kernel(
    config value,
    const std::uint16_t* q_proj,
    const float* prepared_queries,
    const std::uint16_t* key_cache,
    const std::uint16_t* value_cache,
    std::size_t visible_count,
    std::size_t query_start,
    std::size_t query_count,
    const std::int32_t* selected_indices,
    const std::uint32_t* selected_counts,
    std::size_t selected_stride,
    std::uint16_t* output,
    status* device_status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    const std::size_t row_count = query_count * value.query_heads;
    if (row >= row_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::size_t token = row / value.query_heads;
    const std::size_t query_head = row % value.query_heads;
    const std::size_t kv_head = query_head / (value.query_heads / value.kv_heads);
    const std::size_t query_position = query_start + token;
    extern __shared__ float reduction[];
    __shared__ unsigned int valid;
    __shared__ std::uint32_t count;
    __shared__ std::int32_t current_index;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probability;

    if (lane == 0u) {
        valid = 1u;
        count = selected_counts[token];
        maximum = -CUDART_INF_F;
        denominator = 0.0F;
        probability = 0.0F;
        if (count == 0u || count > selected_stride ||
            count > value.max_selected_tokens) {
            report_error(device_status, status::invalid_selected_count);
            valid = 0u;
            count = 0u;
        }
    }
    __syncthreads();

    const float query = prepared_queries[row * value.head_dim + lane];
    if (!isfinite(query)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
    }
    __syncthreads();

    for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        if (lane == 0u) {
            current_index = selected_indices[token * selected_stride + ordinal];
            if (current_index < 0 ||
                static_cast<std::size_t>(current_index) >= visible_count ||
                static_cast<std::size_t>(current_index) > query_position) {
                report_error(device_status, status::invalid_selected_index);
                valid = 0u;
            }
        }
        __syncthreads();
        if (valid != 0u) {
            const std::size_t key_offset =
                (static_cast<std::size_t>(current_index) * value.kv_heads + kv_head) *
                    value.head_dim +
                lane;
            const float key = load_bf16(key_cache, key_offset);
            const float dot = reduce_sum(query * key, reduction);
            if (lane == 0u) {
                const float score = dot / sqrtf(static_cast<float>(value.head_dim));
                if (!isfinite(score)) {
                    report_error(device_status, status::non_finite);
                    valid = 0u;
                } else {
                    maximum = fmaxf(maximum, score);
                }
            }
        }
        __syncthreads();
    }

    for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        if (lane == 0u) {
            current_index = selected_indices[token * selected_stride + ordinal];
        }
        __syncthreads();
        if (valid != 0u) {
            const std::size_t key_offset =
                (static_cast<std::size_t>(current_index) * value.kv_heads + kv_head) *
                    value.head_dim +
                lane;
            const float key = load_bf16(key_cache, key_offset);
            const float dot = reduce_sum(query * key, reduction);
            if (lane == 0u) {
                const float score = dot / sqrtf(static_cast<float>(value.head_dim));
                denominator += expf(score - maximum);
                if (!isfinite(denominator)) {
                    report_error(device_status, status::non_finite);
                    valid = 0u;
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0u && (denominator <= 0.0F || !isfinite(denominator))) {
        report_error(device_status, status::non_finite);
        valid = 0u;
    }
    __syncthreads();

    float accumulator = 0.0F;
    for (std::uint32_t ordinal = 0u; ordinal < count; ++ordinal) {
        if (lane == 0u) {
            current_index = selected_indices[token * selected_stride + ordinal];
        }
        __syncthreads();
        if (valid != 0u) {
            const std::size_t cache_offset =
                (static_cast<std::size_t>(current_index) * value.kv_heads + kv_head) *
                    value.head_dim +
                lane;
            const float key = load_bf16(key_cache, cache_offset);
            const float dot = reduce_sum(query * key, reduction);
            if (lane == 0u) {
                const float score = dot / sqrtf(static_cast<float>(value.head_dim));
                probability = expf(score - maximum) / denominator;
                if (!isfinite(probability)) {
                    report_error(device_status, status::non_finite);
                    valid = 0u;
                    probability = 0.0F;
                }
            }
            __syncthreads();
            accumulator += probability * load_bf16(value_cache, cache_offset);
        }
        __syncthreads();
    }

    const std::size_t gate_offset =
        (token * value.query_heads + query_head) * (2u * value.head_dim) +
        value.head_dim + lane;
    const float gate = load_bf16(q_proj, gate_offset);
    if (!isfinite(gate) || !isfinite(accumulator)) {
        report_error(device_status, status::non_finite);
        atomicExch(&valid, 0u);
    }
    __syncthreads();
    const float gated = valid != 0u ? accumulator * stable_sigmoid(gate) : 0.0F;
    store_bf16(output, row * value.head_dim + lane, gated);
}

}  // namespace

const char* status_string(status value) noexcept {
    switch (value) {
        case status::ok:
            return "ok";
        case status::null_pointer:
            return "null_pointer";
        case status::invalid_config:
            return "invalid_config";
        case status::invalid_argument:
            return "invalid_argument";
        case status::invalid_state:
            return "invalid_state";
        case status::capacity_exceeded:
            return "capacity_exceeded";
        case status::arithmetic_overflow:
            return "arithmetic_overflow";
        case status::insufficient_workspace:
            return "insufficient_workspace";
        case status::invalid_selected_count:
            return "invalid_selected_count";
        case status::invalid_selected_index:
            return "invalid_selected_index";
        case status::non_finite:
            return "non_finite";
        case status::cuda_failure:
            return "cuda_failure";
    }
    return "unknown";
}

status validate_config(const config& value) noexcept {
    if (value.query_heads == 0u || value.kv_heads == 0u ||
        value.query_heads % value.kv_heads != 0u || value.head_dim == 0u ||
        value.head_dim > 1024u || !is_power_of_two(value.head_dim) ||
        value.rotary_dim > value.head_dim || value.rotary_dim % 2u != 0u ||
        value.max_context == 0u || value.max_context > kCheckpointMaxContext ||
        value.max_selected_tokens == 0u ||
        value.max_selected_tokens > value.max_context ||
        !std::isfinite(value.rms_epsilon) || value.rms_epsilon <= 0.0F) {
        return status::invalid_config;
    }
    return status::ok;
}

status validate_checkpoint_config(const config& value) noexcept {
    return same_config(value, checkpoint_config) ? status::ok : status::invalid_config;
}

float attention_scale(const config& value) noexcept {
    return validate_config(value) == status::ok
               ? 1.0F / std::sqrt(static_cast<float>(value.head_dim))
               : 0.0F;
}

status workspace_bytes(
    const config& value,
    std::size_t token_count,
    std::size_t* bytes) noexcept {
    if (bytes == nullptr) {
        return status::null_pointer;
    }
    *bytes = 0u;
    const status config_status = validate_config(value);
    if (config_status != status::ok) {
        return config_status;
    }
    if (token_count == 0u || token_count > value.max_context) {
        return status::invalid_argument;
    }
    std::size_t elements = 0u;
    std::size_t payload = 0u;
    if (multiply_overflow(token_count, value.query_heads, &elements) ||
        multiply_overflow(elements, value.head_dim, &elements) ||
        multiply_overflow(elements, sizeof(float), &payload) ||
        add_overflow(payload, kWorkspaceAlignment - 1u, bytes)) {
        *bytes = 0u;
        return status::arithmetic_overflow;
    }
    return status::ok;
}

status cache_initialize(
    cache_state* state,
    std::uint16_t* external_key_bf16,
    std::uint16_t* external_value_bf16,
    std::size_t capacity_tokens,
    std::size_t initial_committed_tokens) noexcept {
    if (state == nullptr || external_key_bf16 == nullptr ||
        external_value_bf16 == nullptr) {
        return status::null_pointer;
    }
    *state = {};
    if (capacity_tokens == 0u || capacity_tokens > kCheckpointMaxContext ||
        initial_committed_tokens > capacity_tokens) {
        return status::invalid_argument;
    }
    state->key_bf16 = external_key_bf16;
    state->value_bf16 = external_value_bf16;
    state->capacity_tokens = capacity_tokens;
    state->committed_tokens = initial_committed_tokens;
    state->initialized = true;
    return status::ok;
}

status cache_reset(cache_state* state) noexcept {
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    state->committed_tokens = 0u;
    state->staged_tokens = 0u;
    state->transaction_open = false;
    return status::ok;
}

status cache_get_lengths(
    const cache_state* state,
    cache_lengths* lengths) noexcept {
    if (lengths == nullptr) {
        return status::null_pointer;
    }
    *lengths = {};
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    std::size_t visible = 0u;
    const status length_status = visible_tokens(*state, &visible);
    if (length_status != status::ok) {
        return length_status;
    }
    lengths->committed = state->committed_tokens;
    lengths->staged = state->staged_tokens;
    lengths->visible = visible;
    lengths->capacity = state->capacity_tokens;
    return status::ok;
}

status begin_transaction(cache_state* state) noexcept {
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    if (state->transaction_open) {
        return status::invalid_state;
    }
    state->staged_tokens = 0u;
    state->transaction_open = true;
    return status::ok;
}

status commit_prefix(cache_state* state, std::size_t accepted_tokens) noexcept {
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    if (!state->transaction_open) {
        return status::invalid_state;
    }
    if (accepted_tokens > state->staged_tokens) {
        return status::invalid_argument;
    }
    std::size_t committed = 0u;
    if (add_overflow(state->committed_tokens, accepted_tokens, &committed) ||
        committed > state->capacity_tokens) {
        return status::arithmetic_overflow;
    }
    state->committed_tokens = committed;
    state->staged_tokens = 0u;
    state->transaction_open = false;
    return status::ok;
}

status rollback(cache_state* state) noexcept {
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    if (!state->transaction_open) {
        return status::invalid_state;
    }
    state->staged_tokens = 0u;
    state->transaction_open = false;
    return status::ok;
}

status reset_device_status(status* device_status, cudaStream_t stream) noexcept {
    if (device_status == nullptr) {
        return status::null_pointer;
    }
    return cuda_status(cudaMemsetAsync(device_status, 0, sizeof(status), stream));
}

status collect_device_status(
    const status* device_status,
    status* host_status,
    cudaStream_t stream) noexcept {
    if (device_status == nullptr || host_status == nullptr) {
        return status::null_pointer;
    }
    cudaError_t result = cudaMemcpyAsync(
        host_status,
        device_status,
        sizeof(status),
        cudaMemcpyDeviceToHost,
        stream);
    if (result != cudaSuccess) {
        return status::cuda_failure;
    }
    result = cudaStreamSynchronize(stream);
    return cuda_status(result);
}

status stage_cuda(
    const config& value,
    cache_state* state,
    const std::uint16_t* q_proj_bf16,
    const std::uint16_t* k_proj_bf16,
    const std::uint16_t* v_proj_bf16,
    const std::uint16_t* q_norm_weight_bf16,
    const std::uint16_t* k_norm_weight_bf16,
    const float* cos_f32,
    const float* sin_f32,
    std::size_t token_count,
    void* workspace,
    std::size_t workspace_capacity_bytes,
    staged_batch* batch,
    status* device_status,
    cudaStream_t stream) noexcept {
    if (batch == nullptr) {
        return status::null_pointer;
    }
    *batch = {};
    const status config_status = validate_config(value);
    if (config_status != status::ok) {
        return config_status;
    }
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    if (!state->transaction_open) {
        return status::invalid_state;
    }
    if (q_proj_bf16 == nullptr || k_proj_bf16 == nullptr ||
        v_proj_bf16 == nullptr || q_norm_weight_bf16 == nullptr ||
        k_norm_weight_bf16 == nullptr || cos_f32 == nullptr ||
        sin_f32 == nullptr || workspace == nullptr || device_status == nullptr) {
        return status::null_pointer;
    }
    if (token_count == 0u || token_count > value.max_context) {
        return status::invalid_argument;
    }
    std::size_t start = 0u;
    const status length_status = visible_tokens(*state, &start);
    if (length_status != status::ok) {
        return length_status;
    }
    std::size_t end = 0u;
    if (add_overflow(start, token_count, &end)) {
        return status::arithmetic_overflow;
    }
    if (end > state->capacity_tokens || end > value.max_context) {
        return status::capacity_exceeded;
    }
    std::size_t required = 0u;
    const status workspace_status = workspace_bytes(value, token_count, &required);
    if (workspace_status != status::ok) {
        return workspace_status;
    }
    if (workspace_capacity_bytes < required) {
        return status::insufficient_workspace;
    }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(workspace);
    if (base > std::numeric_limits<std::uintptr_t>::max() -
                   (kWorkspaceAlignment - 1u)) {
        return status::arithmetic_overflow;
    }
    const std::uintptr_t aligned = align_up(base);
    const std::size_t prefix = static_cast<std::size_t>(aligned - base);
    if (prefix > workspace_capacity_bytes || required - (kWorkspaceAlignment - 1u) >
                                                workspace_capacity_bytes - prefix) {
        return status::insufficient_workspace;
    }
    auto* prepared = reinterpret_cast<float*>(aligned);

    std::size_t query_rows = 0u;
    std::size_t kv_rows = 0u;
    if (multiply_overflow(token_count, value.query_heads, &query_rows) ||
        multiply_overflow(token_count, value.kv_heads, &kv_rows) ||
        query_rows > std::numeric_limits<unsigned int>::max() ||
        kv_rows > std::numeric_limits<unsigned int>::max()) {
        return status::arithmetic_overflow;
    }
    const std::size_t shared_bytes =
        2u * static_cast<std::size_t>(value.head_dim) * sizeof(float);
    prepare_query_kernel<<<
        static_cast<unsigned int>(query_rows),
        value.head_dim,
        shared_bytes,
        stream>>>(
        value,
        q_proj_bf16,
        q_norm_weight_bf16,
        cos_f32,
        sin_f32,
        query_rows,
        prepared,
        device_status);
    if (cudaGetLastError() != cudaSuccess) {
        return status::cuda_failure;
    }
    prepare_kv_kernel<<<
        static_cast<unsigned int>(kv_rows),
        value.head_dim,
        shared_bytes,
        stream>>>(
        value,
        k_proj_bf16,
        v_proj_bf16,
        k_norm_weight_bf16,
        cos_f32,
        sin_f32,
        token_count,
        start,
        state->key_bf16,
        state->value_bf16,
        device_status);
    if (cudaGetLastError() != cudaSuccess) {
        return status::cuda_failure;
    }

    state->staged_tokens += token_count;
    batch->start_position = start;
    batch->token_count = token_count;
    batch->prepared_queries_f32 = prepared;
    batch->q_proj_bf16 = q_proj_bf16;
    return status::ok;
}

status forward_qsa_cuda(
    const config& value,
    const cache_state* state,
    const staged_batch& batch,
    const std::int32_t* selected_indices,
    const std::uint32_t* selected_counts,
    std::size_t selected_stride,
    std::uint16_t* output_bf16,
    status* device_status,
    cudaStream_t stream) noexcept {
    const status config_status = validate_config(value);
    if (config_status != status::ok) {
        return config_status;
    }
    const status cache_status = validate_cache(state);
    if (cache_status != status::ok) {
        return cache_status;
    }
    if (batch.prepared_queries_f32 == nullptr || batch.q_proj_bf16 == nullptr ||
        selected_indices == nullptr || selected_counts == nullptr ||
        output_bf16 == nullptr || device_status == nullptr) {
        return status::null_pointer;
    }
    if (batch.token_count == 0u || selected_stride == 0u ||
        selected_stride > value.max_selected_tokens) {
        return status::invalid_argument;
    }
    std::size_t batch_end = 0u;
    if (add_overflow(batch.start_position, batch.token_count, &batch_end)) {
        return status::arithmetic_overflow;
    }
    std::size_t visible = 0u;
    const status length_status = visible_tokens(*state, &visible);
    if (length_status != status::ok) {
        return length_status;
    }
    if (batch_end > visible || visible > value.max_context) {
        return status::invalid_state;
    }
    std::size_t rows = 0u;
    if (multiply_overflow(batch.token_count, value.query_heads, &rows) ||
        rows > std::numeric_limits<unsigned int>::max()) {
        return status::arithmetic_overflow;
    }
    const std::size_t shared_bytes =
        static_cast<std::size_t>(value.head_dim) * sizeof(float);
    qsa_attention_kernel<<<
        static_cast<unsigned int>(rows),
        value.head_dim,
        shared_bytes,
        stream>>>(
        value,
        batch.q_proj_bf16,
        batch.prepared_queries_f32,
        state->key_bf16,
        state->value_bf16,
        visible,
        batch.start_position,
        batch.token_count,
        selected_indices,
        selected_counts,
        selected_stride,
        output_bf16,
        device_status);
    return cuda_status(cudaGetLastError());
}

}  // namespace axiom::qwen4exp::attention
