#include "axiom/qwen4exp/rope.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp::rope {
namespace {

constexpr unsigned int kThreads = 256u;

bool same_config(const config& left, const config& right) noexcept {
    return left.rotary_dim == right.rotary_dim &&
           left.max_context == right.max_context && left.theta == right.theta &&
           left.mrope_sections == right.mrope_sections;
}

status cuda_status(cudaError_t result) noexcept {
    return result == cudaSuccess ? status::ok : status::cuda_failure;
}

std::uint32_t axis_for_frequency(
    std::uint32_t frequency,
    const std::array<std::uint32_t, 3>& sections) noexcept {
    if (frequency % 3u == 1u && frequency < sections[1] * 3u) return 1u;
    if (frequency % 3u == 2u && frequency < sections[2] * 3u) return 2u;
    return 0u;
}

__device__ void report_error(status* output, status value) {
    atomicCAS(
        reinterpret_cast<unsigned int*>(output),
        static_cast<unsigned int>(status::ok),
        static_cast<unsigned int>(value));
}

__device__ std::uint32_t device_axis_for_frequency(std::uint32_t frequency) {
    // Device plans admit only checkpoint_config (11/11/10), so keep the hot
    // path POD-only instead of relying on std::array's host-only operator[].
    if (frequency % 3u == 1u && frequency < 33u) return 1u;
    if (frequency % 3u == 2u && frequency < 30u) return 2u;
    return 0u;
}

__global__ void generate_mrope_kernel(
    config value,
    const float* inverse_frequency,
    const std::uint32_t* positions,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = token_count * value.rotary_dim;
    if (index >= elements) return;
    const std::size_t token = index / value.rotary_dim;
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % value.rotary_dim);
    const std::uint32_t frequency = dimension % (value.rotary_dim / 2u);
    const std::uint32_t axis = device_axis_for_frequency(frequency);
    const std::uint32_t position = positions[axis * token_count + token];
    if (position >= value.max_context) {
        report_error(device_status, status::invalid_argument);
        cosine[index] = 0.0F;
        sine[index] = 0.0F;
        return;
    }
    const float angle = static_cast<float>(position) * inverse_frequency[frequency];
    const float cos_value = cosf(angle);
    const float sin_value = sinf(angle);
    if (!isfinite(cos_value) || !isfinite(sin_value)) {
        report_error(device_status, status::non_finite);
        cosine[index] = 0.0F;
        sine[index] = 0.0F;
        return;
    }
    cosine[index] = cos_value;
    sine[index] = sin_value;
}

__global__ void generate_text_kernel(
    config value,
    const float* inverse_frequency,
    std::uint32_t start_position,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = token_count * value.rotary_dim;
    if (index >= elements) return;
    const std::size_t token = index / value.rotary_dim;
    const std::uint32_t dimension =
        static_cast<std::uint32_t>(index % value.rotary_dim);
    const std::uint32_t frequency = dimension % (value.rotary_dim / 2u);
    const std::uint64_t position =
        static_cast<std::uint64_t>(start_position) + token;
    if (position >= value.max_context) {
        report_error(device_status, status::invalid_argument);
        cosine[index] = 0.0F;
        sine[index] = 0.0F;
        return;
    }
    const float angle = static_cast<float>(position) * inverse_frequency[frequency];
    cosine[index] = cosf(angle);
    sine[index] = sinf(angle);
}

status validate_plan(const device_plan* plan) noexcept {
    if (plan == nullptr) return status::null_pointer;
    if (!plan->initialized || plan->inverse_frequency == nullptr ||
        validate_checkpoint_config(plan->value) != status::ok) {
        return status::invalid_state;
    }
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return status::cuda_failure;
    return current == plan->device ? status::ok : status::invalid_state;
}

status launch_shape(
    const config& value,
    std::size_t token_count,
    unsigned int* blocks) noexcept {
    if (blocks == nullptr) return status::null_pointer;
    *blocks = 0u;
    if (token_count == 0u || token_count > value.max_context) {
        return status::invalid_argument;
    }
    if (token_count > std::numeric_limits<std::size_t>::max() / value.rotary_dim) {
        return status::arithmetic_overflow;
    }
    const std::size_t elements = token_count * value.rotary_dim;
    const std::size_t grid = (elements + kThreads - 1u) / kThreads;
    if (grid > std::numeric_limits<unsigned int>::max()) {
        return status::arithmetic_overflow;
    }
    *blocks = static_cast<unsigned int>(grid);
    return status::ok;
}

}  // namespace

const char* status_string(status value) noexcept {
    switch (value) {
        case status::ok: return "ok";
        case status::null_pointer: return "null_pointer";
        case status::invalid_config: return "invalid_config";
        case status::invalid_argument: return "invalid_argument";
        case status::arithmetic_overflow: return "arithmetic_overflow";
        case status::unsupported_device: return "unsupported_device";
        case status::invalid_state: return "invalid_state";
        case status::non_finite: return "non_finite";
        case status::cuda_failure: return "cuda_failure";
    }
    return "unknown";
}

status validate_config(const config& value) noexcept {
    if (value.rotary_dim == 0u || value.rotary_dim % 2u != 0u ||
        value.rotary_dim != kRotaryDim || value.max_context == 0u ||
        value.max_context > kMaxContext || !std::isfinite(value.theta) ||
        value.theta <= 1.0F || value.mrope_sections[0] == 0u ||
        value.mrope_sections[1] == 0u || value.mrope_sections[2] == 0u ||
        value.mrope_sections[0] + value.mrope_sections[1] +
                value.mrope_sections[2] != value.rotary_dim / 2u) {
        return status::invalid_config;
    }
    return status::ok;
}

status validate_checkpoint_config(const config& value) noexcept {
    return same_config(value, checkpoint_config) ? status::ok
                                                  : status::invalid_config;
}

status initialize_device_plan(
    device_plan* plan,
    const config& value,
    cudaStream_t stream) noexcept {
    if (plan == nullptr) return status::null_pointer;
    if (plan->initialized || plan->inverse_frequency != nullptr) {
        return status::invalid_state;
    }
    const status config_status = validate_checkpoint_config(value);
    if (config_status != status::ok) return config_status;
    int device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return status::cuda_failure;
    }
    if (properties.major != 12 || properties.minor != 0) {
        return status::unsupported_device;
    }
    float host_inverse[kRotaryHalf]{};
    for (std::uint32_t index = 0u; index < kRotaryHalf; ++index) {
        host_inverse[index] = std::pow(
            value.theta,
            -2.0F * static_cast<float>(index) /
                static_cast<float>(value.rotary_dim));
    }
    float* device_inverse = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_inverse),
                   sizeof(host_inverse)) != cudaSuccess) {
        return status::cuda_failure;
    }
    cudaError_t result = cudaMemcpyAsync(
        device_inverse,
        host_inverse,
        sizeof(host_inverse),
        cudaMemcpyHostToDevice,
        stream);
    if (result == cudaSuccess) result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess) {
        static_cast<void>(cudaFree(device_inverse));
        return status::cuda_failure;
    }
    plan->value = value;
    plan->inverse_frequency = device_inverse;
    plan->device = device;
    plan->initialized = true;
    return status::ok;
}

status release_device_plan(device_plan* plan) noexcept {
    if (plan == nullptr) return status::null_pointer;
    if (!plan->initialized || plan->inverse_frequency == nullptr) {
        return status::invalid_state;
    }
    const cudaError_t result = cudaFree(plan->inverse_frequency);
    *plan = {};
    return cuda_status(result);
}

status generate_host(
    const config& value,
    const std::uint32_t* positions,
    std::size_t token_count,
    float* cosine,
    float* sine) noexcept {
    const status config_status = validate_checkpoint_config(value);
    if (config_status != status::ok) return config_status;
    if (positions == nullptr || cosine == nullptr || sine == nullptr) {
        return status::null_pointer;
    }
    if (token_count == 0u || token_count > value.max_context) {
        return status::invalid_argument;
    }
    float inverse[kRotaryHalf]{};
    for (std::uint32_t index = 0u; index < kRotaryHalf; ++index) {
        inverse[index] = std::pow(
            value.theta,
            -2.0F * static_cast<float>(index) /
                static_cast<float>(value.rotary_dim));
    }
    for (std::size_t token = 0u; token < token_count; ++token) {
        for (std::uint32_t dimension = 0u;
             dimension < value.rotary_dim;
             ++dimension) {
            const std::uint32_t frequency = dimension % kRotaryHalf;
            const std::uint32_t axis =
                axis_for_frequency(frequency, value.mrope_sections);
            const std::uint32_t position = positions[axis * token_count + token];
            if (position >= value.max_context) return status::invalid_argument;
            const float angle = static_cast<float>(position) * inverse[frequency];
            const std::size_t output = token * value.rotary_dim + dimension;
            cosine[output] = std::cos(angle);
            sine[output] = std::sin(angle);
        }
    }
    return status::ok;
}

status reset_device_status(status* device_status, cudaStream_t stream) noexcept {
    if (device_status == nullptr) return status::null_pointer;
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
    if (result == cudaSuccess) result = cudaStreamSynchronize(stream);
    return cuda_status(result);
}

status generate_mrope_cuda(
    const device_plan* plan,
    const std::uint32_t* positions,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status,
    cudaStream_t stream) noexcept {
    const status plan_status = validate_plan(plan);
    if (plan_status != status::ok) return plan_status;
    if (positions == nullptr || cosine == nullptr || sine == nullptr ||
        device_status == nullptr) return status::null_pointer;
    unsigned int blocks = 0u;
    const status shape_status = launch_shape(plan->value, token_count, &blocks);
    if (shape_status != status::ok) return shape_status;
    generate_mrope_kernel<<<blocks, kThreads, 0, stream>>>(
        plan->value,
        plan->inverse_frequency,
        positions,
        token_count,
        cosine,
        sine,
        device_status);
    return cuda_status(cudaGetLastError());
}

status generate_text_cuda(
    const device_plan* plan,
    std::uint32_t start_position,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status,
    cudaStream_t stream) noexcept {
    const status plan_status = validate_plan(plan);
    if (plan_status != status::ok) return plan_status;
    if (cosine == nullptr || sine == nullptr || device_status == nullptr) {
        return status::null_pointer;
    }
    if (start_position >= plan->value.max_context ||
        token_count > plan->value.max_context - start_position) {
        return status::invalid_argument;
    }
    unsigned int blocks = 0u;
    const status shape_status = launch_shape(plan->value, token_count, &blocks);
    if (shape_status != status::ok) return shape_status;
    generate_text_kernel<<<blocks, kThreads, 0, stream>>>(
        plan->value,
        plan->inverse_frequency,
        start_position,
        token_count,
        cosine,
        sine,
        device_status);
    return cuda_status(cudaGetLastError());
}

}  // namespace axiom::qwen4exp::rope
