#include "axiom/qwen4exp/ple_compute.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kMaxHidden = 4096u;
constexpr std::size_t kMaxStreams = 8u;
constexpr std::size_t kMaxEmbedding = 4096u;
constexpr std::size_t kMaxSpeculative = 256u;
constexpr unsigned kThreads = 256u;

bool multiply_overflow(std::size_t left,
                       std::size_t right,
                       std::size_t *result) noexcept {
    if (result == nullptr) return true;
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left) {
        return true;
    }
    *result = left * right;
    return false;
}

bool add_overflow(std::size_t left,
                  std::size_t right,
                  std::size_t *result) noexcept {
    if (result == nullptr || right > std::numeric_limits<std::size_t>::max() - left) {
        return true;
    }
    *result = left + right;
    return false;
}

__device__ float bf16(const std::uint16_t *bits, std::size_t index) {
    const __nv_bfloat16_raw raw{bits[index]};
    return __bfloat162float(__nv_bfloat16(raw));
}

__device__ float block_sum(float value) {
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = kThreads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

__global__ void bf16_matvec_kernel(const std::uint16_t *weight,
                                   const float *input,
                                   std::size_t rows,
                                   std::size_t columns,
                                   float *output) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0F;
    const std::size_t base = row * columns;
    for (std::size_t column = threadIdx.x; column < columns; column += blockDim.x) {
        sum = fmaf(bf16(weight, base + column), input[column], sum);
    }
    const float total = block_sum(sum);
    if (threadIdx.x == 0u) output[row] = total;
}

__global__ void grouped_norm_kernel(const float *input,
                                    const std::uint16_t *weight,
                                    std::size_t hidden,
                                    std::size_t streams,
                                    float epsilon,
                                    float *output) {
    const std::size_t stream_index = blockIdx.x;
    if (stream_index >= streams) return;
    const std::size_t base = stream_index * hidden;
    float squares = 0.0F;
    for (std::size_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const float value = input[base + column];
        squares = fmaf(value, value, squares);
    }
    const float total = block_sum(squares);
    const float inverse = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    for (std::size_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        const std::size_t index = base + column;
        output[index] = input[index] * inverse * (1.0F + bf16(weight, index));
    }
}

__global__ void gate_value_kernel(const float *key,
                                  const float *query,
                                  const float *value,
                                  std::size_t hidden,
                                  std::size_t streams,
                                  float *gated) {
    const std::size_t stream_index = blockIdx.x;
    if (stream_index >= streams) return;
    const std::size_t base = stream_index * hidden;
    float dot = 0.0F;
    for (std::size_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        dot = fmaf(key[base + column], query[base + column], dot);
    }
    float gate = block_sum(dot) * rsqrtf(static_cast<float>(hidden));
    const float magnitude = sqrtf(fmaxf(fabsf(gate), 1.0e-6F));
    gate = copysignf(magnitude, gate);
    const float probability = 1.0F / (1.0F + expf(-gate));
    for (std::size_t column = threadIdx.x; column < hidden; column += blockDim.x) {
        gated[base + column] = probability * value[column];
    }
}

__device__ float prior_frame_value(const float *committed,
                                   const float *staged,
                                   std::size_t channels,
                                   std::size_t committed_next,
                                   std::size_t committed_valid,
                                   std::size_t staged_index,
                                   std::size_t distance,
                                   std::size_t channel) {
    if (distance <= staged_index) {
        return staged[(staged_index - distance) * channels + channel];
    }
    const std::size_t committed_distance = distance - staged_index;
    if (committed_distance == 0u || committed_distance > committed_valid) return 0.0F;
    const std::size_t frame =
            (committed_next + kPleComputeHistory - committed_distance) %
            kPleComputeHistory;
    return committed[frame * channels + channel];
}

__global__ void dilated_conv_residual_kernel(
        const float *committed,
        const float *staged,
        const float *gated,
        const std::uint16_t *conv,
        std::size_t channels,
        std::size_t committed_next,
        std::size_t committed_valid,
        std::size_t staged_index,
        float *output) {
    const std::size_t channel =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    float convolved = 0.0F;
    for (std::size_t tap = 0u; tap < kPleComputeTaps; ++tap) {
        const std::size_t distance =
                (kPleComputeTaps - 1u - tap) * kPleComputeDilation;
        const float value = prior_frame_value(
                committed, staged, channels, committed_next, committed_valid,
                staged_index, distance, channel);
        convolved = fmaf(bf16(conv, channel * kPleComputeTaps + tap),
                         value, convolved);
    }
    const float activated = convolved / (1.0F + expf(-convolved));
    output[channel] = gated[channel] + activated;
}

__global__ void commit_frames_kernel(float *committed,
                                     const float *staged,
                                     std::size_t channels,
                                     std::size_t committed_next,
                                     std::size_t accepted) {
    const std::size_t channel =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t frame = blockIdx.y;
    if (frame >= accepted || channel >= channels) return;
    const std::size_t target = (committed_next + frame) % kPleComputeHistory;
    committed[target * channels + channel] = staged[frame * channels + channel];
}

ple_compute_status immediate_launch_status() noexcept {
    return cudaGetLastError() == cudaSuccess ? ple_compute_status::ok
                                             : ple_compute_status::cuda_error;
}

ple_compute_status validate_state(const ple_compute_device_state *state) noexcept {
    if (state == nullptr || !state->initialized || state->device < 0 ||
        state->committed_history == nullptr || state->staged_frames == nullptr) {
        return ple_compute_status::invalid_state;
    }
    if (ple_compute_validate_config(state->config) != ple_compute_status::ok ||
        state->committed_next >= kPleComputeHistory ||
        state->committed_valid > kPleComputeHistory ||
        state->staged_count > state->config.max_speculative_tokens) {
        return ple_compute_status::invalid_state;
    }
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return ple_compute_status::cuda_error;
    return current == state->device ? ple_compute_status::ok
                                    : ple_compute_status::invalid_state;
}

}  // namespace

ple_compute_config ple_compute_qwen4_exp_config() noexcept { return {}; }

ple_compute_status ple_compute_validate_config(
        const ple_compute_config &config) noexcept {
    if (config.hidden == 0u || config.streams == 0u || config.embedding == 0u ||
        config.max_speculative_tokens == 0u ||
        !std::isfinite(config.rms_epsilon) || config.rms_epsilon <= 0.0F) {
        return ple_compute_status::invalid_argument;
    }
    if (config.hidden > kMaxHidden || config.streams > kMaxStreams ||
        config.embedding > kMaxEmbedding ||
        config.max_speculative_tokens > kMaxSpeculative ||
        config.hidden % 32u != 0u || config.embedding % 32u != 0u) {
        return ple_compute_status::unsupported_config;
    }
    std::size_t channels = 0u;
    std::size_t committed = 0u;
    std::size_t staged = 0u;
    std::size_t bytes = 0u;
    if (multiply_overflow(config.hidden, config.streams, &channels) ||
        multiply_overflow(channels, kPleComputeHistory, &committed) ||
        multiply_overflow(channels, config.max_speculative_tokens, &staged) ||
        add_overflow(committed, staged, &bytes) ||
        multiply_overflow(bytes, sizeof(float), &bytes)) {
        return ple_compute_status::size_overflow;
    }
    return ple_compute_status::ok;
}

bool ple_compute_is_qwen4_exp_contract(
        const ple_compute_config &config) noexcept {
    return ple_compute_validate_config(config) == ple_compute_status::ok &&
           config.hidden == kPleComputeHidden &&
           config.streams == kPleComputeStreams &&
           config.embedding == kPleComputeEmbedding &&
           config.rms_epsilon == 1.0e-6F;
}

ple_compute_status ple_compute_get_footprint(
        const ple_compute_config &config,
        ple_compute_footprint *footprint) noexcept {
    if (footprint == nullptr) return ple_compute_status::invalid_argument;
    const ple_compute_status valid = ple_compute_validate_config(config);
    if (valid != ple_compute_status::ok) return valid;
    ple_compute_footprint result{};
    if (multiply_overflow(config.hidden, config.streams, &result.channels) ||
        multiply_overflow(result.channels, kPleComputeHistory,
                          &result.committed_state_floats) ||
        multiply_overflow(result.channels, config.max_speculative_tokens,
                          &result.staged_state_floats)) {
        return ple_compute_status::size_overflow;
    }
    std::size_t four_channels = 0u;
    if (multiply_overflow(result.channels, 4u, &four_channels) ||
        add_overflow(four_channels, config.hidden, &result.scratch_floats)) {
        return ple_compute_status::size_overflow;
    }
    std::size_t state_floats = 0u;
    if (add_overflow(result.committed_state_floats,
                     result.staged_state_floats, &state_floats) ||
        multiply_overflow(state_floats, sizeof(float), &result.state_bytes)) {
        return ple_compute_status::size_overflow;
    }
    *footprint = result;
    return ple_compute_status::ok;
}

const char *ple_compute_status_string(ple_compute_status status) noexcept {
    switch (status) {
        case ple_compute_status::ok: return "ok";
        case ple_compute_status::invalid_argument: return "invalid_argument";
        case ple_compute_status::unsupported_config: return "unsupported_config";
        case ple_compute_status::invalid_state: return "invalid_state";
        case ple_compute_status::size_overflow: return "size_overflow";
        case ple_compute_status::unsupported_device: return "unsupported_device";
        case ple_compute_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

ple_compute_status ple_compute_device_state_init(
        ple_compute_device_state *state,
        const ple_compute_config &config,
        cudaStream_t stream) noexcept {
    if (state == nullptr || state->initialized) return ple_compute_status::invalid_argument;
    ple_compute_footprint footprint{};
    const ple_compute_status valid = ple_compute_get_footprint(config, &footprint);
    if (valid != ple_compute_status::ok) return valid;
    int device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return ple_compute_status::cuda_error;
    }
    if (properties.major < 12) return ple_compute_status::unsupported_device;
    float *committed = nullptr;
    float *staged = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&committed),
                   footprint.committed_state_floats * sizeof(float)) != cudaSuccess) {
        return ple_compute_status::cuda_error;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&staged),
                   footprint.staged_state_floats * sizeof(float)) != cudaSuccess) {
        (void)cudaFree(committed);
        return ple_compute_status::cuda_error;
    }
    if (cudaMemsetAsync(committed, 0,
                        footprint.committed_state_floats * sizeof(float), stream) != cudaSuccess ||
        cudaMemsetAsync(staged, 0,
                        footprint.staged_state_floats * sizeof(float), stream) != cudaSuccess) {
        (void)cudaFree(staged);
        (void)cudaFree(committed);
        return ple_compute_status::cuda_error;
    }
    state->config = config;
    state->committed_history = committed;
    state->staged_frames = staged;
    state->committed_next = 0u;
    state->committed_valid = 0u;
    state->staged_count = 0u;
    state->committed_tokens = 0u;
    state->device = device;
    state->initialized = true;
    state->transaction_open = false;
    return ple_compute_status::ok;
}

ple_compute_status ple_compute_device_state_reset(
        ple_compute_device_state *state,
        cudaStream_t stream) noexcept {
    const ple_compute_status valid = validate_state(state);
    if (valid != ple_compute_status::ok) return valid;
    if (state->transaction_open) return ple_compute_status::invalid_state;
    ple_compute_footprint footprint{};
    if (ple_compute_get_footprint(state->config, &footprint) != ple_compute_status::ok) {
        return ple_compute_status::invalid_state;
    }
    if (cudaMemsetAsync(state->committed_history, 0,
                        footprint.committed_state_floats * sizeof(float), stream) != cudaSuccess ||
        cudaMemsetAsync(state->staged_frames, 0,
                        footprint.staged_state_floats * sizeof(float), stream) != cudaSuccess) {
        return ple_compute_status::cuda_error;
    }
    state->committed_next = 0u;
    state->committed_valid = 0u;
    state->staged_count = 0u;
    state->committed_tokens = 0u;
    return ple_compute_status::ok;
}

ple_compute_status ple_compute_device_state_release(
        ple_compute_device_state *state) noexcept {
    if (state == nullptr) return ple_compute_status::invalid_argument;
    ple_compute_status result = ple_compute_status::ok;
    if (state->staged_frames != nullptr && cudaFree(state->staged_frames) != cudaSuccess) {
        result = ple_compute_status::cuda_error;
    }
    if (state->committed_history != nullptr &&
        cudaFree(state->committed_history) != cudaSuccess) {
        result = ple_compute_status::cuda_error;
    }
    *state = {};
    return result;
}

ple_compute_status ple_compute_begin_transaction(
        ple_compute_device_state *state) noexcept {
    const ple_compute_status valid = validate_state(state);
    if (valid != ple_compute_status::ok) return valid;
    if (state->transaction_open) return ple_compute_status::invalid_state;
    state->staged_count = 0u;
    state->transaction_open = true;
    return ple_compute_status::ok;
}

ple_compute_status ple_compute_stage_token_cuda(
        ple_compute_device_state *state,
        const ple_compute_weights &weights,
        const float *embedding,
        const float *hidden_states,
        const ple_compute_scratch &scratch,
        float *output,
        cudaStream_t stream) noexcept {
    const ple_compute_status valid = validate_state(state);
    if (valid != ple_compute_status::ok) return valid;
    if (!state->transaction_open ||
        state->staged_count >= state->config.max_speculative_tokens) {
        return ple_compute_status::invalid_state;
    }
    if (weights.key_proj == nullptr || weights.value_proj == nullptr ||
        weights.norm_key == nullptr || weights.norm_query == nullptr ||
        weights.norm_conv == nullptr || weights.conv == nullptr ||
        embedding == nullptr || hidden_states == nullptr || output == nullptr ||
        scratch.key == nullptr || scratch.value == nullptr ||
        scratch.query == nullptr || scratch.gated == nullptr ||
        scratch.normalized == nullptr) {
        return ple_compute_status::invalid_argument;
    }
    const std::size_t hidden = state->config.hidden;
    const std::size_t streams = state->config.streams;
    const std::size_t channels = hidden * streams;
    if (output == embedding || output == hidden_states ||
        output == scratch.key || output == scratch.value ||
        output == scratch.query || output == scratch.gated ||
        output == scratch.normalized) {
        return ple_compute_status::invalid_argument;
    }

    bf16_matvec_kernel<<<static_cast<unsigned>(channels), kThreads, 0, stream>>>(
            weights.key_proj, embedding, channels, state->config.embedding,
            scratch.key);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    bf16_matvec_kernel<<<static_cast<unsigned>(hidden), kThreads, 0, stream>>>(
            weights.value_proj, embedding, hidden, state->config.embedding,
            scratch.value);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    grouped_norm_kernel<<<static_cast<unsigned>(streams), kThreads, 0, stream>>>(
            scratch.key, weights.norm_key, hidden, streams,
            state->config.rms_epsilon, scratch.key);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    grouped_norm_kernel<<<static_cast<unsigned>(streams), kThreads, 0, stream>>>(
            hidden_states, weights.norm_query, hidden, streams,
            state->config.rms_epsilon, scratch.query);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    gate_value_kernel<<<static_cast<unsigned>(streams), kThreads, 0, stream>>>(
            scratch.key, scratch.query, scratch.value, hidden, streams,
            scratch.gated);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    grouped_norm_kernel<<<static_cast<unsigned>(streams), kThreads, 0, stream>>>(
            scratch.gated, weights.norm_conv, hidden, streams,
            state->config.rms_epsilon, scratch.normalized);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;

    float *staged = state->staged_frames + state->staged_count * channels;
    if (cudaMemcpyAsync(staged, scratch.normalized, channels * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return ple_compute_status::cuda_error;
    }
    const unsigned blocks = static_cast<unsigned>((channels + kThreads - 1u) / kThreads);
    dilated_conv_residual_kernel<<<blocks, kThreads, 0, stream>>>(
            state->committed_history, state->staged_frames, scratch.gated,
            weights.conv, channels, state->committed_next,
            state->committed_valid, state->staged_count, output);
    if (immediate_launch_status() != ple_compute_status::ok) return ple_compute_status::cuda_error;
    ++state->staged_count;
    return ple_compute_status::ok;
}

ple_compute_status ple_compute_commit_prefix_cuda(
        ple_compute_device_state *state,
        std::size_t accepted_tokens,
        cudaStream_t stream) noexcept {
    const ple_compute_status valid = validate_state(state);
    if (valid != ple_compute_status::ok) return valid;
    if (!state->transaction_open || accepted_tokens > state->staged_count) {
        return ple_compute_status::invalid_state;
    }
    if (accepted_tokens != 0u) {
        const std::size_t channels = state->config.hidden * state->config.streams;
        const unsigned blocks = static_cast<unsigned>((channels + kThreads - 1u) / kThreads);
        const dim3 grid(blocks, static_cast<unsigned>(accepted_tokens), 1u);
        commit_frames_kernel<<<grid, kThreads, 0, stream>>>(
                state->committed_history, state->staged_frames, channels,
                state->committed_next, accepted_tokens);
        if (immediate_launch_status() != ple_compute_status::ok) {
            return ple_compute_status::cuda_error;
        }
        state->committed_next =
                (state->committed_next + accepted_tokens) % kPleComputeHistory;
        state->committed_valid =
                state->committed_valid + accepted_tokens > kPleComputeHistory
                        ? kPleComputeHistory
                        : state->committed_valid + accepted_tokens;
        state->committed_tokens += accepted_tokens;
    }
    state->staged_count = 0u;
    state->transaction_open = false;
    return ple_compute_status::ok;
}

ple_compute_status ple_compute_rollback(
        ple_compute_device_state *state) noexcept {
    const ple_compute_status valid = validate_state(state);
    if (valid != ple_compute_status::ok) return valid;
    if (!state->transaction_open) return ple_compute_status::invalid_state;
    state->staged_count = 0u;
    state->transaction_open = false;
    return ple_compute_status::ok;
}

}  // namespace axiom::qwen4exp
