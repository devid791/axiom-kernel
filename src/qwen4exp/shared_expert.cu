#include "axiom/qwen4exp/shared_expert.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr std::uint32_t kThreads = 256u;
constexpr std::uint32_t kMaximumDimension = 65536u;

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t *result) noexcept {
    if (result == nullptr ||
        (right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

float bf16_to_f32_host(std::uint16_t encoded) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

void zero_host(const shared_expert_config &config,
               float *scratch_intermediate,
               float *scratch_output_gate,
               float *output) noexcept {
    if (scratch_intermediate != nullptr) {
        std::fill(scratch_intermediate,
                  scratch_intermediate + config.intermediate, 0.0f);
    }
    if (scratch_output_gate != nullptr) *scratch_output_gate = 0.0f;
    if (output != nullptr) std::fill(output, output + config.hidden, 0.0f);
}

bool pointer_visible_to_device(const void *pointer, int device) noexcept {
    cudaPointerAttributes attributes{};
    const cudaError_t result = cudaPointerGetAttributes(&attributes, pointer);
    if (result != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attributes.type == cudaMemoryTypeManaged) return true;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
#else
    if (attributes.isManaged != 0) return true;
    return attributes.memoryType == cudaMemoryTypeDevice && attributes.device == device;
#endif
}

bool aligned_pointer(const void *pointer, std::uintptr_t alignment) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

__device__ __forceinline__ float bf16_to_f32_device(
        const std::uint16_t *values,
        std::size_t index) {
    return __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16 *>(values + index));
}

__device__ __forceinline__ void record_status(
        std::uint32_t *status,
        shared_expert_status value) {
    (void)atomicCAS(status, 0u, static_cast<std::uint32_t>(value));
}

__global__ void output_gate_kernel(const std::uint16_t *weight,
                                   const float *input,
                                   std::uint32_t hidden,
                                   float *output_gate,
                                   std::uint32_t *status) {
    __shared__ float sums[kThreads];
    float local = 0.0f;
    bool invalid = false;
    for (std::uint32_t column = threadIdx.x;
         column < hidden;
         column += blockDim.x) {
        const float input_value = input[column];
        const float weight_value = bf16_to_f32_device(weight, column);
        if (!isfinite(input_value) || !isfinite(weight_value)) {
            invalid = true;
        } else {
            local = fmaf(weight_value, input_value, local);
        }
    }
    if (invalid || !isfinite(local)) {
        record_status(status, shared_expert_status::non_finite);
        local = 0.0f;
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const float sum = sums[0];
        const float gate = 1.0f / (1.0f + expf(-sum));
        if (*status != static_cast<std::uint32_t>(shared_expert_status::ok) ||
            !isfinite(sum) || !isfinite(gate)) {
            if (!isfinite(sum) || !isfinite(gate)) {
                record_status(status, shared_expert_status::non_finite);
            }
            *output_gate = 0.0f;
        } else {
            *output_gate = gate;
        }
    }
}

__global__ void gate_up_silu_kernel(const std::uint16_t *gate_weight,
                                    const std::uint16_t *up_weight,
                                    const float *input,
                                    std::uint32_t hidden,
                                    float *intermediate,
                                    std::uint32_t *status) {
    const std::uint32_t row = blockIdx.x;
    if (*status != static_cast<std::uint32_t>(shared_expert_status::ok)) {
        if (threadIdx.x == 0u) intermediate[row] = 0.0f;
        return;
    }
    __shared__ float gate_sums[kThreads];
    __shared__ float up_sums[kThreads];
    float gate_local = 0.0f;
    float up_local = 0.0f;
    bool invalid = false;
    const std::size_t row_offset = static_cast<std::size_t>(row) * hidden;
    for (std::uint32_t column = threadIdx.x;
         column < hidden;
         column += blockDim.x) {
        const float input_value = input[column];
        const float gate_value = bf16_to_f32_device(
                gate_weight, row_offset + column);
        const float up_value = bf16_to_f32_device(
                up_weight, row_offset + column);
        if (!isfinite(input_value) || !isfinite(gate_value) ||
            !isfinite(up_value)) {
            invalid = true;
        } else {
            gate_local = fmaf(gate_value, input_value, gate_local);
            up_local = fmaf(up_value, input_value, up_local);
        }
    }
    if (invalid || !isfinite(gate_local) || !isfinite(up_local)) {
        record_status(status, shared_expert_status::non_finite);
        gate_local = 0.0f;
        up_local = 0.0f;
    }
    gate_sums[threadIdx.x] = gate_local;
    up_sums[threadIdx.x] = up_local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            gate_sums[threadIdx.x] += gate_sums[threadIdx.x + stride];
            up_sums[threadIdx.x] += up_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const float gate = gate_sums[0];
        const float up = up_sums[0];
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up;
        if (*status != static_cast<std::uint32_t>(shared_expert_status::ok) ||
            !isfinite(gate) || !isfinite(up) || !isfinite(value)) {
            if (!isfinite(gate) || !isfinite(up) || !isfinite(value)) {
                record_status(status, shared_expert_status::non_finite);
            }
            intermediate[row] = 0.0f;
        } else {
            intermediate[row] = value;
        }
    }
}

__global__ void down_kernel(const std::uint16_t *down_weight,
                            const float *intermediate,
                            const float *output_gate,
                            std::uint32_t intermediate_size,
                            float *output,
                            std::uint32_t *status) {
    const std::uint32_t row = blockIdx.x;
    if (*status != static_cast<std::uint32_t>(shared_expert_status::ok)) {
        if (threadIdx.x == 0u) output[row] = 0.0f;
        return;
    }
    __shared__ float sums[kThreads];
    float local = 0.0f;
    bool invalid = false;
    const std::size_t row_offset =
            static_cast<std::size_t>(row) * intermediate_size;
    for (std::uint32_t column = threadIdx.x;
         column < intermediate_size;
         column += blockDim.x) {
        const float activation = intermediate[column];
        const float weight_value = bf16_to_f32_device(
                down_weight, row_offset + column);
        if (!isfinite(activation) || !isfinite(weight_value)) {
            invalid = true;
        } else {
            local = fmaf(weight_value, activation, local);
        }
    }
    if (invalid || !isfinite(local)) {
        record_status(status, shared_expert_status::non_finite);
        local = 0.0f;
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        const float value = sums[0] * *output_gate;
        if (*status != static_cast<std::uint32_t>(shared_expert_status::ok) ||
            !isfinite(value)) {
            if (!isfinite(value)) {
                record_status(status, shared_expert_status::non_finite);
            }
            output[row] = 0.0f;
        } else {
            output[row] = value;
        }
    }
}

__global__ void fail_closed_output_kernel(float *output,
                                          std::uint32_t hidden,
                                          const std::uint32_t *status) {
    if (*status == static_cast<std::uint32_t>(shared_expert_status::ok)) return;
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hidden;
         index += blockDim.x * gridDim.x) {
        output[index] = 0.0f;
    }
}

bool launch_ok() noexcept { return cudaPeekAtLastError() == cudaSuccess; }

}  // namespace

shared_expert_config shared_expert_qwen4_exp_config() noexcept { return {}; }

shared_expert_status shared_expert_validate_config(
        const shared_expert_config &config) noexcept {
    if (config.hidden == 0u || config.intermediate == 0u) {
        return shared_expert_status::invalid_argument;
    }
    std::uint64_t elements = 0u;
    std::uint64_t bytes = 0u;
    if (!checked_multiply(config.hidden, config.intermediate, &elements) ||
        !checked_multiply(elements, sizeof(std::uint16_t), &bytes)) {
        return shared_expert_status::overflow;
    }
    if (config.hidden > kMaximumDimension ||
        config.intermediate > kMaximumDimension) {
        return shared_expert_status::unsupported_config;
    }
    return shared_expert_status::ok;
}

bool shared_expert_is_qwen4_exp_contract(
        const shared_expert_config &config) noexcept {
    return shared_expert_validate_config(config) == shared_expert_status::ok &&
           config.hidden == kSharedExpertHidden &&
           config.intermediate == kSharedExpertIntermediate;
}

const char *shared_expert_status_string(shared_expert_status status) noexcept {
    switch (status) {
        case shared_expert_status::ok: return "ok";
        case shared_expert_status::invalid_argument: return "invalid_argument";
        case shared_expert_status::unsupported_config: return "unsupported_config";
        case shared_expert_status::overflow: return "overflow";
        case shared_expert_status::unsupported_device: return "unsupported_device";
        case shared_expert_status::invalid_device_pointer: return "invalid_device_pointer";
        case shared_expert_status::non_finite: return "non_finite";
        case shared_expert_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

shared_expert_status shared_expert_tensor_spec_for(
        const shared_expert_config &config,
        shared_expert_tensor tensor,
        shared_expert_tensor_spec *spec) noexcept {
    const shared_expert_status validation = shared_expert_validate_config(config);
    if (validation != shared_expert_status::ok) return validation;
    if (spec == nullptr) return shared_expert_status::invalid_argument;

    shared_expert_tensor_spec result{};
    switch (tensor) {
        case shared_expert_tensor::gate_projection:
        case shared_expert_tensor::up_projection:
            result.rows = config.intermediate;
            result.columns = config.hidden;
            break;
        case shared_expert_tensor::down_projection:
            result.rows = config.hidden;
            result.columns = config.intermediate;
            break;
        case shared_expert_tensor::output_gate:
            result.rows = 1u;
            result.columns = config.hidden;
            break;
        default:
            return shared_expert_status::invalid_argument;
    }
    std::uint64_t elements = 0u;
    if (!checked_multiply(result.rows, result.columns, &elements) ||
        !checked_multiply(elements, sizeof(std::uint16_t), &result.bytes)) {
        return shared_expert_status::overflow;
    }
    *spec = result;
    return shared_expert_status::ok;
}

shared_expert_status shared_expert_tensor_name(
        std::uint32_t layer,
        shared_expert_tensor tensor,
        char *destination,
        std::size_t destination_bytes) noexcept {
    if (layer >= kSharedExpertLayers || destination == nullptr ||
        destination_bytes == 0u) {
        return shared_expert_status::invalid_argument;
    }
    const char *format = nullptr;
    switch (tensor) {
        case shared_expert_tensor::gate_projection:
            format = "model.language_model.layers.%u.mlp.shared_expert.gate_proj.weight";
            break;
        case shared_expert_tensor::up_projection:
            format = "model.language_model.layers.%u.mlp.shared_expert.up_proj.weight";
            break;
        case shared_expert_tensor::down_projection:
            format = "model.language_model.layers.%u.mlp.shared_expert.down_proj.weight";
            break;
        case shared_expert_tensor::output_gate:
            format = "model.language_model.layers.%u.mlp.shared_expert_gate.weight";
            break;
        default:
            return shared_expert_status::invalid_argument;
    }
    const int written = std::snprintf(destination, destination_bytes, format, layer);
    if (written < 0 || static_cast<std::size_t>(written) >= destination_bytes) {
        destination[0] = '\0';
        return shared_expert_status::invalid_argument;
    }
    return shared_expert_status::ok;
}

shared_expert_status shared_expert_forward_f32_host(
        const shared_expert_config &config,
        const shared_expert_weights_bf16 &weights,
        const float *input,
        float *scratch_intermediate,
        float *scratch_output_gate,
        float *output) noexcept {
    const shared_expert_status validation = shared_expert_validate_config(config);
    if (validation != shared_expert_status::ok) return validation;
    if (weights.gate == nullptr || weights.up == nullptr || weights.down == nullptr ||
        weights.output_gate == nullptr || input == nullptr ||
        scratch_intermediate == nullptr || scratch_output_gate == nullptr ||
        output == nullptr) {
        return shared_expert_status::invalid_argument;
    }
    zero_host(config, scratch_intermediate, scratch_output_gate, output);

    float output_gate_sum = 0.0f;
    for (std::uint32_t column = 0u; column < config.hidden; ++column) {
        const float input_value = input[column];
        const float weight_value = bf16_to_f32_host(weights.output_gate[column]);
        if (!std::isfinite(input_value) || !std::isfinite(weight_value)) {
            return shared_expert_status::non_finite;
        }
        output_gate_sum = std::fma(weight_value, input_value, output_gate_sum);
    }
    const float output_gate = 1.0f / (1.0f + std::exp(-output_gate_sum));
    if (!std::isfinite(output_gate_sum) || !std::isfinite(output_gate)) {
        return shared_expert_status::non_finite;
    }
    *scratch_output_gate = output_gate;

    for (std::uint32_t row = 0u; row < config.intermediate; ++row) {
        float gate = 0.0f;
        float up = 0.0f;
        const std::size_t offset = static_cast<std::size_t>(row) * config.hidden;
        for (std::uint32_t column = 0u; column < config.hidden; ++column) {
            const float input_value = input[column];
            const float gate_value = bf16_to_f32_host(weights.gate[offset + column]);
            const float up_value = bf16_to_f32_host(weights.up[offset + column]);
            if (!std::isfinite(input_value) || !std::isfinite(gate_value) ||
                !std::isfinite(up_value)) {
                zero_host(config, scratch_intermediate, scratch_output_gate, output);
                return shared_expert_status::non_finite;
            }
            gate = std::fma(gate_value, input_value, gate);
            up = std::fma(up_value, input_value, up);
        }
        const float value = (gate / (1.0f + std::exp(-gate))) * up;
        if (!std::isfinite(gate) || !std::isfinite(up) || !std::isfinite(value)) {
            zero_host(config, scratch_intermediate, scratch_output_gate, output);
            return shared_expert_status::non_finite;
        }
        scratch_intermediate[row] = value;
    }

    for (std::uint32_t row = 0u; row < config.hidden; ++row) {
        float sum = 0.0f;
        const std::size_t offset =
                static_cast<std::size_t>(row) * config.intermediate;
        for (std::uint32_t column = 0u; column < config.intermediate; ++column) {
            const float weight_value = bf16_to_f32_host(weights.down[offset + column]);
            const float activation = scratch_intermediate[column];
            if (!std::isfinite(weight_value) || !std::isfinite(activation)) {
                zero_host(config, scratch_intermediate, scratch_output_gate, output);
                return shared_expert_status::non_finite;
            }
            sum = std::fma(weight_value, activation, sum);
        }
        const float value = sum * output_gate;
        if (!std::isfinite(sum) || !std::isfinite(value)) {
            zero_host(config, scratch_intermediate, scratch_output_gate, output);
            return shared_expert_status::non_finite;
        }
        output[row] = value;
    }
    return shared_expert_status::ok;
}

shared_expert_status shared_expert_forward_f32_cuda(
        int device,
        const shared_expert_config &config,
        const shared_expert_weights_bf16 &weights,
        const float *input,
        const shared_expert_workspace_f32 &workspace,
        float *output,
        cudaStream_t stream) noexcept {
    const shared_expert_status validation = shared_expert_validate_config(config);
    if (validation != shared_expert_status::ok) return validation;
    if (device < 0 || weights.gate == nullptr || weights.up == nullptr ||
        weights.down == nullptr || weights.output_gate == nullptr ||
        input == nullptr || workspace.intermediate == nullptr ||
        workspace.output_gate == nullptr || workspace.device_status == nullptr ||
        output == nullptr || input == output) {
        return shared_expert_status::invalid_argument;
    }
    if (!aligned_pointer(weights.gate, alignof(std::uint16_t)) ||
        !aligned_pointer(weights.up, alignof(std::uint16_t)) ||
        !aligned_pointer(weights.down, alignof(std::uint16_t)) ||
        !aligned_pointer(weights.output_gate, alignof(std::uint16_t)) ||
        !aligned_pointer(input, alignof(float)) ||
        !aligned_pointer(workspace.intermediate, alignof(float)) ||
        !aligned_pointer(workspace.output_gate, alignof(float)) ||
        !aligned_pointer(workspace.device_status, alignof(std::uint32_t)) ||
        !aligned_pointer(output, alignof(float))) {
        return shared_expert_status::invalid_argument;
    }

    int device_count = 0;
    int current_device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device >= device_count ||
        cudaGetDevice(&current_device) != cudaSuccess || current_device != device ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return shared_expert_status::unsupported_device;
    }
    if (properties.major * 10 + properties.minor < 120) {
        return shared_expert_status::unsupported_device;
    }

    const void *const pointers[] = {
        weights.gate, weights.up, weights.down, weights.output_gate, input,
        workspace.intermediate, workspace.output_gate,
        workspace.device_status, output,
    };
    for (const void *pointer : pointers) {
        if (!pointer_visible_to_device(pointer, device)) {
            return shared_expert_status::invalid_device_pointer;
        }
    }

    const std::size_t intermediate_bytes =
            static_cast<std::size_t>(config.intermediate) * sizeof(float);
    const std::size_t output_bytes =
            static_cast<std::size_t>(config.hidden) * sizeof(float);
    if (cudaMemsetAsync(workspace.device_status, 0,
                        sizeof(*workspace.device_status), stream) != cudaSuccess ||
        cudaMemsetAsync(workspace.intermediate, 0, intermediate_bytes, stream) != cudaSuccess ||
        cudaMemsetAsync(workspace.output_gate, 0,
                        sizeof(*workspace.output_gate), stream) != cudaSuccess ||
        cudaMemsetAsync(output, 0, output_bytes, stream) != cudaSuccess) {
        return shared_expert_status::cuda_error;
    }

    output_gate_kernel<<<1u, kThreads, 0u, stream>>>(
            weights.output_gate, input, config.hidden,
            workspace.output_gate, workspace.device_status);
    if (!launch_ok()) return shared_expert_status::cuda_error;

    gate_up_silu_kernel<<<config.intermediate, kThreads, 0u, stream>>>(
            weights.gate, weights.up, input, config.hidden,
            workspace.intermediate, workspace.device_status);
    if (!launch_ok()) return shared_expert_status::cuda_error;

    down_kernel<<<config.hidden, kThreads, 0u, stream>>>(
            weights.down, workspace.intermediate, workspace.output_gate,
            config.intermediate, output, workspace.device_status);
    if (!launch_ok()) return shared_expert_status::cuda_error;

    const std::uint32_t blocks = std::min<std::uint32_t>(
            256u, (config.hidden + kThreads - 1u) / kThreads);
    fail_closed_output_kernel<<<blocks, kThreads, 0u, stream>>>(
            output, config.hidden, workspace.device_status);
    return launch_ok() ? shared_expert_status::ok
                       : shared_expert_status::cuda_error;
}

}  // namespace axiom::qwen4exp
