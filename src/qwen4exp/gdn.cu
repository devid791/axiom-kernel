#include "axiom/qwen4exp/gdn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr int kConvThreads = 256;
constexpr int kRecurrentThreads = 128;

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t* result) noexcept {
    if (result == nullptr || lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool checked_mul(std::size_t lhs, std::size_t rhs, std::size_t* result) noexcept {
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

GdnStatus validate_and_measure(const GdnConfig& config, GdnFootprint* footprint) noexcept {
    if (config.batch == 0 || config.key_heads == 0 || config.value_heads == 0 ||
        config.key_head_dim == 0 || config.value_head_dim == 0 || config.conv_kernel == 0 ||
        !std::isfinite(config.rms_epsilon) || config.rms_epsilon <= 0.0F ||
        !std::isfinite(config.l2_epsilon) || config.l2_epsilon <= 0.0F) {
        return GdnStatus::kInvalidArgument;
    }
    if (config.batch > kGdnMaxBatch || config.key_heads > kGdnMaxKeyHeads ||
        config.value_heads > kGdnMaxValueHeads || config.key_head_dim > kGdnMaxHeadDim ||
        config.value_head_dim > kGdnMaxHeadDim || config.conv_kernel != kGdnQwen4ExpConvKernel ||
        config.value_heads % config.key_heads != 0) {
        return GdnStatus::kUnsupportedConfig;
    }

    GdnFootprint measured{};
    std::size_t twice_keys = 0;
    std::size_t state_per_head = 0;
    std::size_t state_per_batch = 0;
    std::size_t state_floats = 0;
    std::size_t total_state_floats = 0;
    if (!checked_mul(config.key_heads, config.key_head_dim, &measured.key_elements) ||
        !checked_mul(config.value_heads, config.value_head_dim, &measured.value_elements) ||
        !checked_mul(measured.key_elements, 2, &twice_keys) ||
        !checked_add(twice_keys, measured.value_elements, &measured.conv_channels) ||
        !checked_mul(config.batch, measured.conv_channels, &measured.step_workspace_floats) ||
        !checked_mul(measured.step_workspace_floats, config.conv_kernel, &measured.conv_state_floats) ||
        !checked_mul(config.key_head_dim, config.value_head_dim, &state_per_head) ||
        !checked_mul(config.value_heads, state_per_head, &state_per_batch) ||
        !checked_mul(config.batch, state_per_batch, &state_floats) ||
        !checked_add(measured.conv_state_floats, state_floats, &total_state_floats) ||
        !checked_mul(total_state_floats, sizeof(float), &measured.device_state_bytes)) {
        return GdnStatus::kSizeOverflow;
    }
    measured.recurrent_state_floats = state_floats;
    if (footprint != nullptr) {
        *footprint = measured;
    }
    return GdnStatus::kOk;
}

GdnStatus validate_host_state(
    const GdnConfig& config,
    const GdnHostStateView& state,
    GdnFootprint* footprint) noexcept {
    GdnFootprint measured{};
    const GdnStatus status = validate_and_measure(config, &measured);
    if (status != GdnStatus::kOk) {
        return status;
    }
    if (state.conv_state == nullptr || state.recurrent_state == nullptr ||
        state.conv_cursor == nullptr || state.tokens_seen == nullptr) {
        return GdnStatus::kInvalidArgument;
    }
    if (state.conv_state_floats < measured.conv_state_floats ||
        state.recurrent_state_floats < measured.recurrent_state_floats ||
        *state.conv_cursor >= config.conv_kernel) {
        return GdnStatus::kStateMismatch;
    }
    if (footprint != nullptr) {
        *footprint = measured;
    }
    return GdnStatus::kOk;
}

bool valid_conv_arguments(
    const float* projected_qkv,
    const GdnWeightsView& weights,
    float* convolved_qkv) noexcept {
    return projected_qkv != nullptr && weights.conv_weight != nullptr && convolved_qkv != nullptr;
}

bool valid_recurrent_arguments(
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const GdnWeightsView& weights,
    float* output) noexcept {
    return convolved_qkv != nullptr && z != nullptr && a != nullptr && b != nullptr &&
           weights.A_log != nullptr && weights.dt_bias != nullptr &&
           weights.norm_weight != nullptr && output != nullptr;
}

float host_sigmoid(float value) noexcept {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float host_softplus(float value) noexcept {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

GdnStatus validate_cuda_state(
    const GdnDeviceState* state,
    GdnFootprint* footprint,
    cudaDeviceProp* properties) noexcept {
    if (state == nullptr) {
        return GdnStatus::kInvalidArgument;
    }
    if (!state->initialized) {
        return GdnStatus::kUninitialized;
    }
    GdnFootprint measured{};
    const GdnStatus status = validate_and_measure(state->config, &measured);
    if (status != GdnStatus::kOk) {
        return status;
    }
    if (state->conv_state == nullptr || state->recurrent_state == nullptr ||
        state->conv_state_floats != measured.conv_state_floats ||
        state->recurrent_state_floats != measured.recurrent_state_floats ||
        state->conv_cursor >= state->config.conv_kernel) {
        return GdnStatus::kStateMismatch;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (current_device != state->device) {
        return GdnStatus::kStateMismatch;
    }
    cudaDeviceProp local_properties{};
    if (cudaGetDeviceProperties(&local_properties, current_device) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (local_properties.major != 12 || local_properties.minor != 0) {
        return GdnStatus::kUnsupportedDevice;
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (footprint != nullptr) {
        *footprint = measured;
    }
    if (properties != nullptr) {
        *properties = local_properties;
    }
    return GdnStatus::kOk;
}

__device__ float device_sigmoid(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + expf(-value));
    }
    const float exponential = expf(value);
    return exponential / (1.0F + exponential);
}

__device__ float device_softplus(float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return expf(value);
    }
    return log1pf(expf(value));
}

__global__ void causal_conv_update_kernel(
    const float* projected_qkv,
    const float* conv_weight,
    const float* conv_bias,
    float* conv_state,
    std::size_t batch,
    std::size_t channels,
    std::size_t kernel_size,
    std::uint32_t cursor,
    float* convolved_qkv) {
    const std::size_t linear_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = batch * channels;
    if (linear_index >= total) {
        return;
    }
    const std::size_t batch_index = linear_index / channels;
    const std::size_t channel = linear_index - batch_index * channels;
    float* channel_state =
        conv_state + (batch_index * channels + channel) * kernel_size;
    channel_state[cursor] = projected_qkv[linear_index];

    float sum = conv_bias == nullptr ? 0.0F : conv_bias[channel];
    for (std::size_t tap = 0; tap < kernel_size; ++tap) {
        const std::size_t ring_index = (static_cast<std::size_t>(cursor) + 1 + tap) % kernel_size;
        sum += conv_weight[channel * kernel_size + tap] * channel_state[ring_index];
    }
    convolved_qkv[linear_index] = sum * device_sigmoid(sum);
}

__global__ void recurrent_delta_update_kernel(
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const float* A_log,
    const float* dt_bias,
    const float* norm_weight,
    float* recurrent_state,
    std::size_t key_heads,
    std::size_t value_heads,
    std::size_t key_head_dim,
    std::size_t value_head_dim,
    std::size_t key_elements,
    std::size_t conv_channels,
    float l2_epsilon,
    float rms_epsilon,
    float* output) {
    const std::size_t head_linear = blockIdx.x;
    const std::size_t batch_index = head_linear / value_heads;
    const std::size_t value_head = head_linear - batch_index * value_heads;
    const std::size_t repeat = value_heads / key_heads;
    const std::size_t key_head = value_head / repeat;
    const float* token_qkv = convolved_qkv + batch_index * conv_channels;
    const float* query = token_qkv + key_head * key_head_dim;
    const float* key = token_qkv + key_elements + key_head * key_head_dim;
    const float* value = token_qkv + 2 * key_elements + value_head * value_head_dim;
    float* state = recurrent_state +
        (batch_index * value_heads + value_head) * key_head_dim * value_head_dim;
    float* head_output = output +
        (batch_index * value_heads + value_head) * value_head_dim;
    const float* head_z = z +
        (batch_index * value_heads + value_head) * value_head_dim;

    __shared__ float shared_inv_query;
    __shared__ float shared_inv_key;
    __shared__ float shared_decay;
    __shared__ float shared_beta;
    __shared__ float shared_rms_inverse;

    if (threadIdx.x == 0) {
        float query_square_sum = 0.0F;
        float key_square_sum = 0.0F;
        for (std::size_t index = 0; index < key_head_dim; ++index) {
            query_square_sum += query[index] * query[index];
            key_square_sum += key[index] * key[index];
        }
        shared_inv_query = rsqrtf(query_square_sum + l2_epsilon);
        shared_inv_key = rsqrtf(key_square_sum + l2_epsilon);
        const std::size_t scalar_index = batch_index * value_heads + value_head;
        const float g = -expf(A_log[value_head]) *
                        device_softplus(a[scalar_index] + dt_bias[value_head]);
        shared_decay = expf(g);
        shared_beta = device_sigmoid(b[scalar_index]);
    }
    __syncthreads();

    const float query_scale = rsqrtf(static_cast<float>(key_head_dim));
    for (std::size_t value_index = threadIdx.x; value_index < value_head_dim;
         value_index += blockDim.x) {
        float memory_value = 0.0F;
        for (std::size_t key_index = 0; key_index < key_head_dim; ++key_index) {
            const std::size_t state_index = key_index * value_head_dim + value_index;
            const float decayed = state[state_index] * shared_decay;
            state[state_index] = decayed;
            memory_value += decayed * (key[key_index] * shared_inv_key);
        }
        const float delta = (value[value_index] - memory_value) * shared_beta;
        float queried = 0.0F;
        for (std::size_t key_index = 0; key_index < key_head_dim; ++key_index) {
            const std::size_t state_index = key_index * value_head_dim + value_index;
            const float updated =
                state[state_index] + (key[key_index] * shared_inv_key) * delta;
            state[state_index] = updated;
            queried += updated * (query[key_index] * shared_inv_query);
        }
        head_output[value_index] = queried * query_scale;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float square_sum = 0.0F;
        for (std::size_t value_index = 0; value_index < value_head_dim; ++value_index) {
            square_sum += head_output[value_index] * head_output[value_index];
        }
        shared_rms_inverse =
            rsqrtf(square_sum / static_cast<float>(value_head_dim) + rms_epsilon);
    }
    __syncthreads();

    for (std::size_t value_index = threadIdx.x; value_index < value_head_dim;
         value_index += blockDim.x) {
        const float normalized =
            head_output[value_index] * shared_rms_inverse * norm_weight[value_index];
        head_output[value_index] = normalized * device_sigmoid(head_z[value_index]);
    }
}

// One thread owns one convolution channel and advances that channel through
// every row. This preserves causal order without launching once per token.
__global__ void causal_conv_sequence_kernel(
    const float* projected_qkv,
    const float* conv_weight,
    const float* conv_bias,
    float* conv_state,
    std::size_t token_count,
    std::size_t channels,
    std::size_t kernel_size,
    std::uint32_t initial_cursor,
    float* convolved_qkv) {
    const std::size_t channel =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    float* channel_state = conv_state + channel * kernel_size;
    for (std::size_t token = 0; token < token_count; ++token) {
        const std::size_t cursor =
            (static_cast<std::size_t>(initial_cursor) + token) % kernel_size;
        channel_state[cursor] = projected_qkv[token * channels + channel];

        float sum = conv_bias == nullptr ? 0.0F : conv_bias[channel];
        for (std::size_t tap = 0; tap < kernel_size; ++tap) {
            const std::size_t ring_index = (cursor + 1 + tap) % kernel_size;
            sum += conv_weight[channel * kernel_size + tap] *
                   channel_state[ring_index];
        }
        convolved_qkv[token * channels + channel] =
            sum * device_sigmoid(sum);
    }
}

// One block owns one value head. All rows therefore update the same recurrent
// matrix sequentially inside the block, with synchronization at each stage.
__global__ void recurrent_delta_sequence_kernel(
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const float* A_log,
    const float* dt_bias,
    const float* norm_weight,
    float* recurrent_state,
    std::size_t token_count,
    std::size_t key_heads,
    std::size_t value_heads,
    std::size_t key_head_dim,
    std::size_t value_head_dim,
    std::size_t key_elements,
    std::size_t conv_channels,
    float l2_epsilon,
    float rms_epsilon,
    float* output) {
    const std::size_t value_head = blockIdx.x;
    const std::size_t repeat = value_heads / key_heads;
    const std::size_t key_head = value_head / repeat;
    float* state = recurrent_state +
        value_head * key_head_dim * value_head_dim;

    __shared__ float shared_inv_query;
    __shared__ float shared_inv_key;
    __shared__ float shared_decay;
    __shared__ float shared_beta;
    __shared__ float shared_rms_inverse;

    const float query_scale = rsqrtf(static_cast<float>(key_head_dim));
    for (std::size_t token = 0; token < token_count; ++token) {
        const float* token_qkv = convolved_qkv + token * conv_channels;
        const float* query = token_qkv + key_head * key_head_dim;
        const float* key = token_qkv + key_elements + key_head * key_head_dim;
        const float* value = token_qkv + 2 * key_elements +
                             value_head * value_head_dim;
        float* head_output = output +
            (token * value_heads + value_head) * value_head_dim;
        const float* head_z = z +
            (token * value_heads + value_head) * value_head_dim;

        if (threadIdx.x == 0) {
            float query_square_sum = 0.0F;
            float key_square_sum = 0.0F;
            for (std::size_t index = 0; index < key_head_dim; ++index) {
                query_square_sum += query[index] * query[index];
                key_square_sum += key[index] * key[index];
            }
            shared_inv_query = rsqrtf(query_square_sum + l2_epsilon);
            shared_inv_key = rsqrtf(key_square_sum + l2_epsilon);
            const std::size_t scalar_index = token * value_heads + value_head;
            const float g = -expf(A_log[value_head]) *
                            device_softplus(a[scalar_index] + dt_bias[value_head]);
            shared_decay = expf(g);
            shared_beta = device_sigmoid(b[scalar_index]);
        }
        __syncthreads();

        for (std::size_t value_index = threadIdx.x;
             value_index < value_head_dim;
             value_index += blockDim.x) {
            float memory_value = 0.0F;
            for (std::size_t key_index = 0; key_index < key_head_dim;
                 ++key_index) {
                const std::size_t state_index =
                    key_index * value_head_dim + value_index;
                const float decayed = state[state_index] * shared_decay;
                state[state_index] = decayed;
                memory_value += decayed * (key[key_index] * shared_inv_key);
            }
            const float delta =
                (value[value_index] - memory_value) * shared_beta;
            float queried = 0.0F;
            for (std::size_t key_index = 0; key_index < key_head_dim;
                 ++key_index) {
                const std::size_t state_index =
                    key_index * value_head_dim + value_index;
                const float updated = state[state_index] +
                    (key[key_index] * shared_inv_key) * delta;
                state[state_index] = updated;
                queried += updated * (query[key_index] * shared_inv_query);
            }
            head_output[value_index] = queried * query_scale;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            float square_sum = 0.0F;
            for (std::size_t value_index = 0; value_index < value_head_dim;
                 ++value_index) {
                square_sum += head_output[value_index] * head_output[value_index];
            }
            shared_rms_inverse = rsqrtf(
                square_sum / static_cast<float>(value_head_dim) + rms_epsilon);
        }
        __syncthreads();

        for (std::size_t value_index = threadIdx.x;
             value_index < value_head_dim;
             value_index += blockDim.x) {
            const float normalized = head_output[value_index] *
                shared_rms_inverse * norm_weight[value_index];
            head_output[value_index] =
                normalized * device_sigmoid(head_z[value_index]);
        }
        __syncthreads();
    }
}

}  // namespace

GdnConfig gdn_qwen4_exp_config(std::size_t batch) noexcept {
    GdnConfig config{};
    config.batch = batch;
    return config;
}

GdnStatus gdn_validate_config(const GdnConfig& config) noexcept {
    return validate_and_measure(config, nullptr);
}

bool gdn_is_qwen4_exp_contract(const GdnConfig& config) noexcept {
    return gdn_validate_config(config) == GdnStatus::kOk &&
           config.key_heads == kGdnQwen4ExpKeyHeads &&
           config.value_heads == kGdnQwen4ExpValueHeads &&
           config.key_head_dim == kGdnQwen4ExpKeyHeadDim &&
           config.value_head_dim == kGdnQwen4ExpValueHeadDim &&
           config.conv_kernel == kGdnQwen4ExpConvKernel &&
           config.rms_epsilon == kGdnQwen4ExpRmsEpsilon &&
           config.l2_epsilon == kGdnQwen4ExpL2Epsilon;
}

GdnStatus gdn_footprint(const GdnConfig& config, GdnFootprint* footprint) noexcept {
    if (footprint == nullptr) {
        return GdnStatus::kInvalidArgument;
    }
    *footprint = GdnFootprint{};
    return validate_and_measure(config, footprint);
}

const char* gdn_status_string(GdnStatus status) noexcept {
    switch (status) {
        case GdnStatus::kOk: return "ok";
        case GdnStatus::kInvalidArgument: return "invalid_argument";
        case GdnStatus::kUnsupportedConfig: return "unsupported_config";
        case GdnStatus::kDimensionMismatch: return "dimension_mismatch";
        case GdnStatus::kSizeOverflow: return "size_overflow";
        case GdnStatus::kUninitialized: return "uninitialized";
        case GdnStatus::kStateMismatch: return "state_mismatch";
        case GdnStatus::kUnsupportedDevice: return "unsupported_device";
        case GdnStatus::kCudaError: return "cuda_error";
    }
    return "unknown";
}

GdnStatus gdn_host_state_reset(
    const GdnConfig& config,
    const GdnHostStateView& state) noexcept {
    GdnFootprint footprint{};
    const GdnStatus status = validate_host_state(config, state, &footprint);
    if (status != GdnStatus::kOk) {
        return status;
    }
    std::fill_n(state.conv_state, footprint.conv_state_floats, 0.0F);
    std::fill_n(state.recurrent_state, footprint.recurrent_state_floats, 0.0F);
    *state.conv_cursor = 0;
    *state.tokens_seen = 0;
    return GdnStatus::kOk;
}

GdnStatus gdn_causal_conv_update_host(
    const GdnConfig& config,
    const float* projected_qkv,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    float* convolved_qkv) noexcept {
    GdnFootprint footprint{};
    const GdnStatus status = validate_host_state(config, state, &footprint);
    if (status != GdnStatus::kOk) {
        return status;
    }
    if (!valid_conv_arguments(projected_qkv, weights, convolved_qkv)) {
        return GdnStatus::kInvalidArgument;
    }
    if (*state.tokens_seen == std::numeric_limits<std::uint64_t>::max()) {
        return GdnStatus::kSizeOverflow;
    }
    const std::size_t cursor = *state.conv_cursor;
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        for (std::size_t channel = 0; channel < footprint.conv_channels; ++channel) {
            const std::size_t token_index = batch_index * footprint.conv_channels + channel;
            float* channel_state = state.conv_state +
                token_index * config.conv_kernel;
            channel_state[cursor] = projected_qkv[token_index];
            float sum = weights.conv_bias == nullptr ? 0.0F : weights.conv_bias[channel];
            for (std::size_t tap = 0; tap < config.conv_kernel; ++tap) {
                const std::size_t ring_index = (cursor + 1 + tap) % config.conv_kernel;
                sum += weights.conv_weight[channel * config.conv_kernel + tap] *
                       channel_state[ring_index];
            }
            convolved_qkv[token_index] = sum * host_sigmoid(sum);
        }
    }
    *state.conv_cursor = static_cast<std::uint32_t>((cursor + 1) % config.conv_kernel);
    ++(*state.tokens_seen);
    return GdnStatus::kOk;
}

GdnStatus gdn_recurrent_delta_update_host(
    const GdnConfig& config,
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    float* output) noexcept {
    GdnFootprint footprint{};
    const GdnStatus status = validate_host_state(config, state, &footprint);
    if (status != GdnStatus::kOk) {
        return status;
    }
    if (!valid_recurrent_arguments(convolved_qkv, z, a, b, weights, output)) {
        return GdnStatus::kInvalidArgument;
    }

    const std::size_t repeat = config.value_heads / config.key_heads;
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(config.key_head_dim));
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        const float* token_qkv =
            convolved_qkv + batch_index * footprint.conv_channels;
        for (std::size_t value_head = 0; value_head < config.value_heads; ++value_head) {
            const std::size_t key_head = value_head / repeat;
            const float* query = token_qkv + key_head * config.key_head_dim;
            const float* key = token_qkv + footprint.key_elements +
                               key_head * config.key_head_dim;
            const float* value = token_qkv + 2 * footprint.key_elements +
                                 value_head * config.value_head_dim;
            float* matrix = state.recurrent_state +
                (batch_index * config.value_heads + value_head) *
                    config.key_head_dim * config.value_head_dim;
            float* head_output = output +
                (batch_index * config.value_heads + value_head) * config.value_head_dim;
            const float* head_z = z +
                (batch_index * config.value_heads + value_head) * config.value_head_dim;

            float query_square_sum = 0.0F;
            float key_square_sum = 0.0F;
            for (std::size_t index = 0; index < config.key_head_dim; ++index) {
                query_square_sum += query[index] * query[index];
                key_square_sum += key[index] * key[index];
            }
            const float inv_query = 1.0F / std::sqrt(query_square_sum + config.l2_epsilon);
            const float inv_key = 1.0F / std::sqrt(key_square_sum + config.l2_epsilon);
            const std::size_t scalar_index =
                batch_index * config.value_heads + value_head;
            const float g = -std::exp(weights.A_log[value_head]) *
                            host_softplus(a[scalar_index] + weights.dt_bias[value_head]);
            const float decay = std::exp(g);
            const float beta = host_sigmoid(b[scalar_index]);

            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                float memory_value = 0.0F;
                for (std::size_t key_index = 0; key_index < config.key_head_dim;
                     ++key_index) {
                    const std::size_t state_index =
                        key_index * config.value_head_dim + value_index;
                    matrix[state_index] *= decay;
                    memory_value += matrix[state_index] * (key[key_index] * inv_key);
                }
                const float delta = (value[value_index] - memory_value) * beta;
                float queried = 0.0F;
                for (std::size_t key_index = 0; key_index < config.key_head_dim;
                     ++key_index) {
                    const std::size_t state_index =
                        key_index * config.value_head_dim + value_index;
                    matrix[state_index] += (key[key_index] * inv_key) * delta;
                    queried += matrix[state_index] * (query[key_index] * inv_query);
                }
                head_output[value_index] = queried * query_scale;
            }

            float square_sum = 0.0F;
            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                square_sum += head_output[value_index] * head_output[value_index];
            }
            const float rms_inverse = 1.0F / std::sqrt(
                square_sum / static_cast<float>(config.value_head_dim) + config.rms_epsilon);
            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                head_output[value_index] = head_output[value_index] * rms_inverse *
                    weights.norm_weight[value_index] * host_sigmoid(head_z[value_index]);
            }
        }
    }
    return GdnStatus::kOk;
}

GdnStatus gdn_step_host(
    const GdnConfig& config,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    const GdnStepOutputs& outputs) noexcept {
    GdnFootprint footprint{};
    const GdnStatus state_status = validate_host_state(config, state, &footprint);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (!valid_conv_arguments(inputs.projected_qkv, weights, outputs.convolved_qkv) ||
        !valid_recurrent_arguments(
            outputs.convolved_qkv, inputs.z, inputs.a, inputs.b, weights, outputs.output)) {
        return GdnStatus::kInvalidArgument;
    }
    const GdnStatus conv_status = gdn_causal_conv_update_host(
        config, inputs.projected_qkv, weights, state, outputs.convolved_qkv);
    if (conv_status != GdnStatus::kOk) {
        return conv_status;
    }
    return gdn_recurrent_delta_update_host(
        config,
        outputs.convolved_qkv,
        inputs.z,
        inputs.a,
        inputs.b,
        weights,
        state,
        outputs.output);
}

GdnStatus gdn_forward_sequence_host(
    const GdnConfig& config,
    std::size_t token_count,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    const GdnStepOutputs& outputs) noexcept {
    GdnFootprint footprint{};
    const GdnStatus state_status = validate_host_state(config, state, &footprint);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (config.batch != 1u || token_count == 0u ||
        token_count > kGdnMaxSequenceTokens) {
        return GdnStatus::kUnsupportedConfig;
    }
    if (!valid_conv_arguments(inputs.projected_qkv, weights, outputs.convolved_qkv) ||
        !valid_recurrent_arguments(
            outputs.convolved_qkv, inputs.z, inputs.a, inputs.b, weights, outputs.output)) {
        return GdnStatus::kInvalidArgument;
    }
    if (*state.tokens_seen >
        std::numeric_limits<std::uint64_t>::max() - token_count) {
        return GdnStatus::kSizeOverflow;
    }

    const std::size_t qkv_stride = footprint.conv_channels;
    const std::size_t value_stride = footprint.value_elements;
    const std::size_t scalar_stride = config.value_heads;
    for (std::size_t token = 0u; token < token_count; ++token) {
        const GdnStepInputs token_inputs{
            inputs.projected_qkv + token * qkv_stride,
            inputs.z + token * value_stride,
            inputs.a + token * scalar_stride,
            inputs.b + token * scalar_stride,
        };
        const GdnStepOutputs token_outputs{
            outputs.convolved_qkv + token * qkv_stride,
            outputs.output + token * value_stride,
        };
        const GdnStatus status =
            gdn_step_host(config, token_inputs, weights, state, token_outputs);
        if (status != GdnStatus::kOk) {
            return status;
        }
    }
    return GdnStatus::kOk;
}

GdnStatus gdn_device_state_init(
    GdnDeviceState* state,
    const GdnConfig& config,
    cudaStream_t stream) noexcept {
    if (state == nullptr) {
        return GdnStatus::kInvalidArgument;
    }
    if (state->initialized || state->conv_state != nullptr || state->recurrent_state != nullptr) {
        return GdnStatus::kStateMismatch;
    }
    GdnFootprint footprint{};
    const GdnStatus config_status = validate_and_measure(config, &footprint);
    if (config_status != GdnStatus::kOk) {
        return config_status;
    }
    int device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (properties.major != 12 || properties.minor != 0) {
        return GdnStatus::kUnsupportedDevice;
    }
    const std::size_t recurrent_bytes = footprint.recurrent_state_floats * sizeof(float);
    const std::size_t conv_bytes = footprint.conv_state_floats * sizeof(float);
    float* conv_state = nullptr;
    float* recurrent_state = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&conv_state), conv_bytes) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&recurrent_state), recurrent_bytes) != cudaSuccess) {
        (void)cudaFree(conv_state);
        return GdnStatus::kCudaError;
    }

    state->config = config;
    state->conv_state = conv_state;
    state->recurrent_state = recurrent_state;
    state->conv_state_floats = footprint.conv_state_floats;
    state->recurrent_state_floats = footprint.recurrent_state_floats;
    state->conv_cursor = 0;
    state->tokens_seen = 0;
    state->device = device;
    state->initialized = true;
    const GdnStatus reset_status = gdn_device_state_reset(state, stream);
    if (reset_status != GdnStatus::kOk) {
        (void)gdn_device_state_release(state);
        return reset_status;
    }
    return GdnStatus::kOk;
}

GdnStatus gdn_device_state_reset(GdnDeviceState* state, cudaStream_t stream) noexcept {
    GdnFootprint footprint{};
    const GdnStatus status = validate_cuda_state(state, &footprint, nullptr);
    if (status != GdnStatus::kOk) {
        return status;
    }
    if (cudaMemsetAsync(
            state->conv_state, 0, footprint.conv_state_floats * sizeof(float), stream) !=
            cudaSuccess ||
        cudaMemsetAsync(
            state->recurrent_state,
            0,
            footprint.recurrent_state_floats * sizeof(float),
            stream) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    state->conv_cursor = 0;
    state->tokens_seen = 0;
    return GdnStatus::kOk;
}

GdnStatus gdn_device_state_release(GdnDeviceState* state) noexcept {
    if (state == nullptr) {
        return GdnStatus::kInvalidArgument;
    }
    if (!state->initialized && state->conv_state == nullptr && state->recurrent_state == nullptr) {
        *state = GdnDeviceState{};
        return GdnStatus::kOk;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    if (state->device >= 0 && current_device != state->device) {
        return GdnStatus::kStateMismatch;
    }
    cudaError_t conv_status = cudaSuccess;
    cudaError_t recurrent_status = cudaSuccess;
    if (state->conv_state != nullptr) {
        conv_status = cudaFree(state->conv_state);
    }
    if (state->recurrent_state != nullptr) {
        recurrent_status = cudaFree(state->recurrent_state);
    }
    *state = GdnDeviceState{};
    return conv_status == cudaSuccess && recurrent_status == cudaSuccess
        ? GdnStatus::kOk
        : GdnStatus::kCudaError;
}

GdnStatus gdn_causal_conv_update_cuda(
    GdnDeviceState* state,
    const float* projected_qkv,
    const GdnWeightsView& weights,
    float* convolved_qkv,
    cudaStream_t stream) noexcept {
    GdnFootprint footprint{};
    cudaDeviceProp properties{};
    const GdnStatus state_status = validate_cuda_state(state, &footprint, &properties);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (!valid_conv_arguments(projected_qkv, weights, convolved_qkv)) {
        return GdnStatus::kInvalidArgument;
    }
    if (state->tokens_seen == std::numeric_limits<std::uint64_t>::max()) {
        return GdnStatus::kSizeOverflow;
    }
    const std::size_t element_count = footprint.step_workspace_floats;
    const std::size_t block_count =
        (element_count + static_cast<std::size_t>(kConvThreads) - 1) /
        static_cast<std::size_t>(kConvThreads);
    if (block_count == 0 || block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
        return GdnStatus::kDimensionMismatch;
    }
    causal_conv_update_kernel<<<
        static_cast<unsigned int>(block_count), kConvThreads, 0, stream>>>(
        projected_qkv,
        weights.conv_weight,
        weights.conv_bias,
        state->conv_state,
        state->config.batch,
        footprint.conv_channels,
        state->config.conv_kernel,
        state->conv_cursor,
        convolved_qkv);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return GdnStatus::kCudaError;
    }
    state->conv_cursor = static_cast<std::uint32_t>(
        (static_cast<std::size_t>(state->conv_cursor) + 1) % state->config.conv_kernel);
    ++state->tokens_seen;
    return GdnStatus::kOk;
}

GdnStatus gdn_recurrent_delta_update_cuda(
    GdnDeviceState* state,
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const GdnWeightsView& weights,
    float* output,
    cudaStream_t stream) noexcept {
    GdnFootprint footprint{};
    cudaDeviceProp properties{};
    const GdnStatus state_status = validate_cuda_state(state, &footprint, &properties);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (!valid_recurrent_arguments(convolved_qkv, z, a, b, weights, output)) {
        return GdnStatus::kInvalidArgument;
    }
    std::size_t block_count = 0;
    if (!checked_mul(state->config.batch, state->config.value_heads, &block_count)) {
        return GdnStatus::kSizeOverflow;
    }
    if (block_count == 0 || block_count > static_cast<std::size_t>(properties.maxGridSize[0])) {
        return GdnStatus::kDimensionMismatch;
    }
    recurrent_delta_update_kernel<<<
        static_cast<unsigned int>(block_count), kRecurrentThreads, 0, stream>>>(
        convolved_qkv,
        z,
        a,
        b,
        weights.A_log,
        weights.dt_bias,
        weights.norm_weight,
        state->recurrent_state,
        state->config.key_heads,
        state->config.value_heads,
        state->config.key_head_dim,
        state->config.value_head_dim,
        footprint.key_elements,
        footprint.conv_channels,
        state->config.l2_epsilon,
        state->config.rms_epsilon,
        output);
    return cudaPeekAtLastError() == cudaSuccess
        ? GdnStatus::kOk
        : GdnStatus::kCudaError;
}

GdnStatus gdn_step_cuda(
    GdnDeviceState* state,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream) noexcept {
    GdnFootprint footprint{};
    const GdnStatus state_status = validate_cuda_state(state, &footprint, nullptr);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (!valid_conv_arguments(inputs.projected_qkv, weights, outputs.convolved_qkv) ||
        !valid_recurrent_arguments(
            outputs.convolved_qkv, inputs.z, inputs.a, inputs.b, weights, outputs.output)) {
        return GdnStatus::kInvalidArgument;
    }
    const GdnStatus conv_status = gdn_causal_conv_update_cuda(
        state, inputs.projected_qkv, weights, outputs.convolved_qkv, stream);
    if (conv_status != GdnStatus::kOk) {
        return conv_status;
    }
    return gdn_recurrent_delta_update_cuda(
        state,
        outputs.convolved_qkv,
        inputs.z,
        inputs.a,
        inputs.b,
        weights,
        outputs.output,
        stream);
}

GdnStatus gdn_forward_sequence_cuda(
    GdnDeviceState* state,
    std::size_t token_count,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream) noexcept {
    GdnFootprint footprint{};
    cudaDeviceProp properties{};
    const GdnStatus state_status =
        validate_cuda_state(state, &footprint, &properties);
    if (state_status != GdnStatus::kOk) {
        return state_status;
    }
    if (state->config.batch != 1u || token_count == 0u ||
        token_count > kGdnMaxSequenceTokens) {
        return GdnStatus::kUnsupportedConfig;
    }
    if (!valid_conv_arguments(inputs.projected_qkv, weights, outputs.convolved_qkv) ||
        !valid_recurrent_arguments(
            outputs.convolved_qkv, inputs.z, inputs.a, inputs.b, weights, outputs.output)) {
        return GdnStatus::kInvalidArgument;
    }
    if (state->tokens_seen >
        std::numeric_limits<std::uint64_t>::max() - token_count) {
        return GdnStatus::kSizeOverflow;
    }

    const std::size_t conv_blocks =
        (footprint.conv_channels + static_cast<std::size_t>(kConvThreads) - 1u) /
        static_cast<std::size_t>(kConvThreads);
    const std::size_t recurrent_blocks = state->config.value_heads;
    if (conv_blocks == 0u || recurrent_blocks == 0u ||
        conv_blocks > static_cast<std::size_t>(properties.maxGridSize[0]) ||
        recurrent_blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
        return GdnStatus::kDimensionMismatch;
    }

    const std::uint32_t initial_cursor = state->conv_cursor;
    causal_conv_sequence_kernel<<<
        static_cast<unsigned int>(conv_blocks), kConvThreads, 0, stream>>>(
        inputs.projected_qkv,
        weights.conv_weight,
        weights.conv_bias,
        state->conv_state,
        token_count,
        footprint.conv_channels,
        state->config.conv_kernel,
        initial_cursor,
        outputs.convolved_qkv);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return GdnStatus::kCudaError;
    }

    recurrent_delta_sequence_kernel<<<
        static_cast<unsigned int>(recurrent_blocks), kRecurrentThreads, 0, stream>>>(
        outputs.convolved_qkv,
        inputs.z,
        inputs.a,
        inputs.b,
        weights.A_log,
        weights.dt_bias,
        weights.norm_weight,
        state->recurrent_state,
        token_count,
        state->config.key_heads,
        state->config.value_heads,
        state->config.key_head_dim,
        state->config.value_head_dim,
        footprint.key_elements,
        footprint.conv_channels,
        state->config.l2_epsilon,
        state->config.rms_epsilon,
        outputs.output);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return GdnStatus::kCudaError;
    }

    state->conv_cursor = static_cast<std::uint32_t>(
        (static_cast<std::size_t>(initial_cursor) + token_count) %
        state->config.conv_kernel);
    state->tokens_seen += token_count;
    return GdnStatus::kOk;
}

}  // namespace axiom::qwen4exp
