#include "axiom/qwen4exp/mhc.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr unsigned int kCudaThreads = 256;

[[nodiscard]] bool multiply_overflows(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* product) noexcept {
    if (product == nullptr) {
        return true;
    }
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return true;
    }
    *product = lhs * rhs;
    return false;
}

[[nodiscard]] bool add_overflows(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* sum) noexcept {
    if (sum == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return true;
    }
    *sum = lhs + rhs;
    return false;
}

[[nodiscard]] MhcStatus validate_launch(
    const MhcConfig& config,
    std::size_t token_count,
    std::size_t* residual,
    std::size_t* required_workspace) noexcept {
    const MhcStatus config_status = mhc_validate_config(config);
    if (config_status != MhcStatus::kOk) {
        return config_status;
    }
    if (token_count == 0) {
        return MhcStatus::kInvalidArgument;
    }
    if (token_count > kMhcMaxTokensPerLaunch) {
        return MhcStatus::kUnsupportedConfig;
    }

    if (multiply_overflows(config.hidden, config.streams, residual)) {
        return MhcStatus::kSizeOverflow;
    }
    std::size_t per_token = 0;
    if (add_overflows(*residual, config.rank, &per_token) ||
        multiply_overflows(token_count, per_token, required_workspace)) {
        return MhcStatus::kSizeOverflow;
    }
    return MhcStatus::kOk;
}

[[nodiscard]] bool mix_weights_valid(const MhcWeightsView& weights) noexcept {
    return weights.hc_norm_weight != nullptr &&
           weights.input_mix_weight_down != nullptr &&
           weights.input_mix_weight_up != nullptr;
}

[[nodiscard]] float sigmoid_host(float value) noexcept {
    if (value >= 0.0F) {
        const float exp_neg = std::exp(-value);
        return 1.0F / (1.0F + exp_neg);
    }
    const float exp_pos = std::exp(value);
    return exp_pos / (1.0F + exp_pos);
}

[[nodiscard]] float silu_host(float value) noexcept {
    return value * sigmoid_host(value);
}

void normalize_host(
    const MhcConfig& config,
    const float* hyper_input,
    const float* norm_weight,
    std::size_t token_count,
    float* normalized) noexcept {
    const std::size_t residual = config.hidden * config.streams;
    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t stream_index = 0; stream_index < config.streams; ++stream_index) {
            const std::size_t row = token * config.streams + stream_index;
            const std::size_t input_base = token * residual + stream_index * config.hidden;
            float square_sum = 0.0F;
            for (std::size_t hidden_index = 0; hidden_index < config.hidden; ++hidden_index) {
                const float value = hyper_input[input_base + hidden_index];
                square_sum = std::fma(value, value, square_sum);
            }
            const float inverse_rms =
                1.0F / std::sqrt(square_sum / static_cast<float>(config.hidden) + config.rms_epsilon);
            const std::size_t normalized_base = row * config.hidden;
            const std::size_t weight_base = stream_index * config.hidden;
            for (std::size_t hidden_index = 0; hidden_index < config.hidden; ++hidden_index) {
                normalized[normalized_base + hidden_index] =
                    hyper_input[input_base + hidden_index] * inverse_rms *
                    (1.0F + norm_weight[weight_base + hidden_index]);
            }
        }
    }
}

void low_rank_host(
    const MhcConfig& config,
    const float* normalized,
    const float* weight_down,
    std::size_t token_count,
    float* low_rank) noexcept {
    const std::size_t residual = config.hidden * config.streams;
    const float stream_scale = 1.0F / static_cast<float>(config.streams);
    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t rank_index = 0; rank_index < config.rank; ++rank_index) {
            float projected = 0.0F;
            const std::size_t weight_base = rank_index * residual;
            const std::size_t input_base = token * residual;
            for (std::size_t residual_index = 0; residual_index < residual; ++residual_index) {
                projected = std::fma(
                    weight_down[weight_base + residual_index],
                    normalized[input_base + residual_index],
                    projected);
            }
            low_rank[token * config.rank + rank_index] = silu_host(projected * stream_scale);
        }
    }
}

void mixed_input_host(
    const MhcConfig& config,
    const float* normalized,
    const float* low_rank,
    const float* weight_up,
    std::size_t token_count,
    float* output) noexcept {
    const std::size_t residual = config.hidden * config.streams;
    const float stream_scale = 1.0F / static_cast<float>(config.streams);
    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t hidden_index = 0; hidden_index < config.hidden; ++hidden_index) {
            float mixed = 0.0F;
            for (std::size_t stream_index = 0; stream_index < config.streams; ++stream_index) {
                const std::size_t residual_index = stream_index * config.hidden + hidden_index;
                const std::size_t weight_base = residual_index * config.rank;
                float projected = 0.0F;
                for (std::size_t rank_index = 0; rank_index < config.rank; ++rank_index) {
                    projected = std::fma(
                        weight_up[weight_base + rank_index],
                        low_rank[token * config.rank + rank_index],
                        projected);
                }
                mixed = std::fma(
                    sigmoid_host(projected),
                    normalized[token * residual + residual_index],
                    mixed);
            }
            output[token * config.hidden + hidden_index] = mixed * stream_scale;
        }
    }
}

void injection_host(
    const MhcConfig& config,
    const float* normalized,
    const float* block_inject_weight,
    std::size_t token_count,
    float* injection_weights) noexcept {
    const std::size_t residual = config.hidden * config.streams;
    const float stream_scale = 1.0F / static_cast<float>(config.streams);
    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t stream_index = 0; stream_index < config.streams; ++stream_index) {
            float projected = 0.0F;
            const std::size_t weight_base = stream_index * residual;
            const std::size_t input_base = token * residual;
            for (std::size_t residual_index = 0; residual_index < residual; ++residual_index) {
                projected = std::fma(
                    block_inject_weight[weight_base + residual_index],
                    normalized[input_base + residual_index],
                    projected);
            }
            injection_weights[token * config.streams + stream_index] =
                2.0F * sigmoid_host(projected * stream_scale);
        }
    }
}

__device__ float sigmoid_device(float value) {
    if (value >= 0.0F) {
        const float exp_neg = expf(-value);
        return 1.0F / (1.0F + exp_neg);
    }
    const float exp_pos = expf(value);
    return exp_pos / (1.0F + exp_pos);
}

__global__ void normalize_kernel(
    const float* hyper_input,
    const float* norm_weight,
    float* normalized,
    std::size_t token_count,
    std::size_t hidden,
    std::size_t streams,
    float epsilon) {
    const std::size_t row = blockIdx.x;
    if (row >= token_count * streams) {
        return;
    }

    const std::size_t stream_index = row % streams;
    const std::size_t token = row / streams;
    const std::size_t residual = hidden * streams;
    const std::size_t input_base = token * residual + stream_index * hidden;

    __shared__ float square_sums[kCudaThreads];
    float local_sum = 0.0F;
    for (std::size_t hidden_index = threadIdx.x; hidden_index < hidden; hidden_index += blockDim.x) {
        const float value = hyper_input[input_base + hidden_index];
        local_sum = fmaf(value, value, local_sum);
    }
    square_sums[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            square_sums[threadIdx.x] += square_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(square_sums[0] / static_cast<float>(hidden) + epsilon);
    const std::size_t weight_base = stream_index * hidden;
    const std::size_t normalized_base = row * hidden;
    for (std::size_t hidden_index = threadIdx.x; hidden_index < hidden; hidden_index += blockDim.x) {
        normalized[normalized_base + hidden_index] =
            hyper_input[input_base + hidden_index] * inverse_rms *
            (1.0F + norm_weight[weight_base + hidden_index]);
    }
}

__global__ void low_rank_kernel(
    const float* normalized,
    const float* weight_down,
    float* low_rank,
    std::size_t token_count,
    std::size_t residual,
    std::size_t rank,
    std::size_t streams) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t output_count = token_count * rank;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t token = output_index / rank;
    const std::size_t rank_index = output_index % rank;
    float projected = 0.0F;
    for (std::size_t residual_index = 0; residual_index < residual; ++residual_index) {
        projected = fmaf(
            weight_down[rank_index * residual + residual_index],
            normalized[token * residual + residual_index],
            projected);
    }
    const float scaled = projected / static_cast<float>(streams);
    low_rank[output_index] = scaled * sigmoid_device(scaled);
}

__global__ void mixed_input_kernel(
    const float* normalized,
    const float* low_rank,
    const float* weight_up,
    float* output,
    std::size_t token_count,
    std::size_t hidden,
    std::size_t streams,
    std::size_t rank) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t output_count = token_count * hidden;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t token = output_index / hidden;
    const std::size_t hidden_index = output_index % hidden;
    const std::size_t residual = hidden * streams;
    float mixed = 0.0F;
    for (std::size_t stream_index = 0; stream_index < streams; ++stream_index) {
        const std::size_t residual_index = stream_index * hidden + hidden_index;
        float projected = 0.0F;
        for (std::size_t rank_index = 0; rank_index < rank; ++rank_index) {
            projected = fmaf(
                weight_up[residual_index * rank + rank_index],
                low_rank[token * rank + rank_index],
                projected);
        }
        mixed = fmaf(
            sigmoid_device(projected),
            normalized[token * residual + residual_index],
            mixed);
    }
    output[output_index] = mixed / static_cast<float>(streams);
}

__global__ void injection_kernel(
    const float* normalized,
    const float* block_inject_weight,
    float* injection_weights,
    std::size_t token_count,
    std::size_t residual,
    std::size_t streams) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t output_count = token_count * streams;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t token = output_index / streams;
    const std::size_t stream_index = output_index % streams;
    float projected = 0.0F;
    for (std::size_t residual_index = 0; residual_index < residual; ++residual_index) {
        projected = fmaf(
            block_inject_weight[stream_index * residual + residual_index],
            normalized[token * residual + residual_index],
            projected);
    }
    injection_weights[output_index] =
        2.0F * sigmoid_device(projected / static_cast<float>(streams));
}

__global__ void reinject_kernel(
    const float* hyper_input,
    const float* block_output,
    const float* injection_weights,
    float* output,
    std::size_t token_count,
    std::size_t hidden,
    std::size_t streams) {
    const std::size_t output_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t residual = hidden * streams;
    const std::size_t output_count = token_count * residual;
    if (output_index >= output_count) {
        return;
    }

    const std::size_t token = output_index / residual;
    const std::size_t in_token = output_index % residual;
    const std::size_t stream_index = in_token / hidden;
    const std::size_t hidden_index = in_token % hidden;
    output[output_index] = hyper_input[output_index] +
                           injection_weights[token * streams + stream_index] *
                               block_output[token * hidden + hidden_index];
}

[[nodiscard]] MhcStatus launch_status() noexcept {
    return cudaPeekAtLastError() == cudaSuccess ? MhcStatus::kOk : MhcStatus::kCudaError;
}

[[nodiscard]] dim3 one_dimensional_grid(std::size_t elements) noexcept {
    return dim3(static_cast<unsigned int>((elements + kCudaThreads - 1U) / kCudaThreads));
}

[[nodiscard]] MhcStatus launch_mix_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* output,
    float* workspace,
    cudaStream_t stream) noexcept {
    const std::size_t residual = config.hidden * config.streams;
    float* normalized = workspace;
    float* low_rank = workspace + token_count * residual;

    const std::size_t norm_rows = token_count * config.streams;
    normalize_kernel<<<static_cast<unsigned int>(norm_rows), kCudaThreads, 0, stream>>>(
        hyper_input,
        weights.hc_norm_weight,
        normalized,
        token_count,
        config.hidden,
        config.streams,
        config.rms_epsilon);
    if (launch_status() != MhcStatus::kOk) {
        return MhcStatus::kCudaError;
    }

    const std::size_t low_rank_count = token_count * config.rank;
    low_rank_kernel<<<one_dimensional_grid(low_rank_count), kCudaThreads, 0, stream>>>(
        normalized,
        weights.input_mix_weight_down,
        low_rank,
        token_count,
        residual,
        config.rank,
        config.streams);
    if (launch_status() != MhcStatus::kOk) {
        return MhcStatus::kCudaError;
    }

    const std::size_t mixed_count = token_count * config.hidden;
    mixed_input_kernel<<<one_dimensional_grid(mixed_count), kCudaThreads, 0, stream>>>(
        normalized,
        low_rank,
        weights.input_mix_weight_up,
        output,
        token_count,
        config.hidden,
        config.streams,
        config.rank);
    return launch_status();
}

}  // namespace

MhcConfig mhc_qwen4_exp_config() noexcept {
    return MhcConfig{};
}

MhcStatus mhc_validate_config(const MhcConfig& config) noexcept {
    if (config.hidden == 0 || config.streams == 0 || config.rank == 0 ||
        !std::isfinite(config.rms_epsilon) || config.rms_epsilon <= 0.0F) {
        return MhcStatus::kInvalidArgument;
    }
    if (config.hidden > kMhcMaxHidden || config.streams < 2 || config.streams > kMhcMaxStreams ||
        config.rank > kMhcMaxRank || config.rms_epsilon > 1.0F) {
        return MhcStatus::kUnsupportedConfig;
    }
    std::size_t residual = 0;
    if (multiply_overflows(config.hidden, config.streams, &residual)) {
        return MhcStatus::kSizeOverflow;
    }
    if (residual > kMhcMaxResidual) {
        return MhcStatus::kUnsupportedConfig;
    }
    return MhcStatus::kOk;
}

bool mhc_is_qwen4_exp_contract(const MhcConfig& config) noexcept {
    return mhc_validate_config(config) == MhcStatus::kOk &&
           config.hidden == kMhcQwen4ExpHidden &&
           config.streams == kMhcQwen4ExpStreams &&
           config.rank == kMhcQwen4ExpRank &&
           config.rms_epsilon == kMhcQwen4ExpRmsEpsilon;
}

const char* mhc_status_string(MhcStatus status) noexcept {
    switch (status) {
        case MhcStatus::kOk:
            return "ok";
        case MhcStatus::kInvalidArgument:
            return "invalid_argument";
        case MhcStatus::kUnsupportedConfig:
            return "unsupported_config";
        case MhcStatus::kSizeOverflow:
            return "size_overflow";
        case MhcStatus::kInsufficientWorkspace:
            return "insufficient_workspace";
        case MhcStatus::kCudaError:
            return "cuda_error";
    }
    return "unknown_status";
}

std::size_t mhc_residual_size(const MhcConfig& config) noexcept {
    if (mhc_validate_config(config) != MhcStatus::kOk) {
        return 0;
    }
    return config.hidden * config.streams;
}

std::size_t mhc_workspace_floats(const MhcConfig& config, std::size_t token_count) noexcept {
    std::size_t residual = 0;
    std::size_t required_workspace = 0;
    if (validate_launch(config, token_count, &residual, &required_workspace) != MhcStatus::kOk) {
        return 0;
    }
    return required_workspace;
}

MhcStatus mhc_prepare_host(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* mixed_input,
    float* injection_weights,
    float* workspace,
    std::size_t workspace_floats) noexcept {
    std::size_t residual = 0;
    std::size_t required_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &required_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || mixed_input == nullptr || injection_weights == nullptr || workspace == nullptr ||
        !mix_weights_valid(weights) || weights.block_inject_weight == nullptr) {
        return MhcStatus::kInvalidArgument;
    }
    if (workspace_floats < required_workspace) {
        return MhcStatus::kInsufficientWorkspace;
    }

    float* normalized = workspace;
    float* low_rank = workspace + token_count * residual;
    normalize_host(config, hyper_input, weights.hc_norm_weight, token_count, normalized);
    low_rank_host(config, normalized, weights.input_mix_weight_down, token_count, low_rank);
    mixed_input_host(config, normalized, low_rank, weights.input_mix_weight_up, token_count, mixed_input);
    injection_host(config, normalized, weights.block_inject_weight, token_count, injection_weights);
    return MhcStatus::kOk;
}

MhcStatus mhc_reinject_host(
    const MhcConfig& config,
    const float* hyper_input,
    const float* block_output,
    const float* injection_weights,
    std::size_t token_count,
    float* output) noexcept {
    std::size_t residual = 0;
    std::size_t ignored_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &ignored_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || block_output == nullptr || injection_weights == nullptr || output == nullptr) {
        return MhcStatus::kInvalidArgument;
    }

    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t stream_index = 0; stream_index < config.streams; ++stream_index) {
            for (std::size_t hidden_index = 0; hidden_index < config.hidden; ++hidden_index) {
                const std::size_t output_index =
                    token * residual + stream_index * config.hidden + hidden_index;
                output[output_index] = hyper_input[output_index] +
                                       injection_weights[token * config.streams + stream_index] *
                                           block_output[token * config.hidden + hidden_index];
            }
        }
    }
    return MhcStatus::kOk;
}

MhcStatus mhc_finalize_host(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* output,
    float* workspace,
    std::size_t workspace_floats) noexcept {
    std::size_t residual = 0;
    std::size_t required_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &required_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || output == nullptr || workspace == nullptr || !mix_weights_valid(weights)) {
        return MhcStatus::kInvalidArgument;
    }
    if (workspace_floats < required_workspace) {
        return MhcStatus::kInsufficientWorkspace;
    }

    float* normalized = workspace;
    float* low_rank = workspace + token_count * residual;
    normalize_host(config, hyper_input, weights.hc_norm_weight, token_count, normalized);
    low_rank_host(config, normalized, weights.input_mix_weight_down, token_count, low_rank);
    mixed_input_host(config, normalized, low_rank, weights.input_mix_weight_up, token_count, output);
    return MhcStatus::kOk;
}

MhcStatus mhc_prepare_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* mixed_input,
    float* injection_weights,
    float* workspace,
    std::size_t workspace_floats,
    cudaStream_t stream) noexcept {
    std::size_t residual = 0;
    std::size_t required_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &required_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || mixed_input == nullptr || injection_weights == nullptr || workspace == nullptr ||
        !mix_weights_valid(weights) || weights.block_inject_weight == nullptr) {
        return MhcStatus::kInvalidArgument;
    }
    if (workspace_floats < required_workspace) {
        return MhcStatus::kInsufficientWorkspace;
    }

    const MhcStatus mix_status = launch_mix_cuda(
        config, hyper_input, weights, token_count, mixed_input, workspace, stream);
    if (mix_status != MhcStatus::kOk) {
        return mix_status;
    }

    float* normalized = workspace;
    const std::size_t injection_count = token_count * config.streams;
    injection_kernel<<<one_dimensional_grid(injection_count), kCudaThreads, 0, stream>>>(
        normalized,
        weights.block_inject_weight,
        injection_weights,
        token_count,
        residual,
        config.streams);
    return launch_status();
}

MhcStatus mhc_reinject_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const float* block_output,
    const float* injection_weights,
    std::size_t token_count,
    float* output,
    cudaStream_t stream) noexcept {
    std::size_t residual = 0;
    std::size_t ignored_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &ignored_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || block_output == nullptr || injection_weights == nullptr || output == nullptr) {
        return MhcStatus::kInvalidArgument;
    }

    const std::size_t output_count = token_count * residual;
    reinject_kernel<<<one_dimensional_grid(output_count), kCudaThreads, 0, stream>>>(
        hyper_input,
        block_output,
        injection_weights,
        output,
        token_count,
        config.hidden,
        config.streams);
    return launch_status();
}

MhcStatus mhc_finalize_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* output,
    float* workspace,
    std::size_t workspace_floats,
    cudaStream_t stream) noexcept {
    std::size_t residual = 0;
    std::size_t required_workspace = 0;
    const MhcStatus launch_status_value =
        validate_launch(config, token_count, &residual, &required_workspace);
    if (launch_status_value != MhcStatus::kOk) {
        return launch_status_value;
    }
    if (hyper_input == nullptr || output == nullptr || workspace == nullptr || !mix_weights_valid(weights)) {
        return MhcStatus::kInvalidArgument;
    }
    if (workspace_floats < required_workspace) {
        return MhcStatus::kInsufficientWorkspace;
    }
    return launch_mix_cuda(config, hyper_input, weights, token_count, output, workspace, stream);
}

}  // namespace axiom::qwen4exp
