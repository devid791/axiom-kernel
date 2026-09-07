#include "axiom/qwen4exp/qsa.hpp"

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace axiom::qwen4exp::qsa {
namespace {

constexpr std::size_t k_workspace_alignment = 256;
constexpr std::size_t k_max_score_shared_bytes = 48U * 1024U;

struct candidate {
    float score;
    std::uint32_t block;
};

[[nodiscard]] bool add_overflow(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (out == nullptr || a > std::numeric_limits<std::size_t>::max() - b) {
        return true;
    }
    *out = a + b;
    return false;
}

[[nodiscard]] bool mul_overflow(std::size_t a, std::size_t b, std::size_t* out) noexcept {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) {
        return true;
    }
    *out = a * b;
    return false;
}

[[nodiscard]] std::uint32_t next_power_of_two(std::uint32_t value) noexcept {
    if (value <= 1) {
        return 1;
    }
    --value;
    value |= value >> 1U;
    value |= value >> 2U;
    value |= value >> 4U;
    value |= value >> 8U;
    value |= value >> 16U;
    return value + 1U;
}

[[nodiscard]] bool finite(float value) noexcept {
    return std::isfinite(value);
}

[[nodiscard]] bool same_config(const config& a, const config& b) noexcept {
    return a.query_heads == b.query_heads && a.kv_heads == b.kv_heads && a.head_dim == b.head_dim &&
           a.rotary_dim == b.rotary_dim && a.compress_ratio == b.compress_ratio &&
           a.token_budget == b.token_budget && a.max_context == b.max_context &&
           a.rms_epsilon == b.rms_epsilon;
}

[[nodiscard]] status validate_count(const config& cfg, std::size_t count) noexcept {
    return count <= static_cast<std::size_t>(cfg.max_context) ? status::ok : status::invalid_argument;
}

[[nodiscard]] status validate_visible_host(
    const std::int32_t* visible,
    std::size_t count,
    std::size_t key_count,
    std::size_t position_count) noexcept {
    if (count != 0 && visible == nullptr) {
        return status::null_pointer;
    }
    std::int32_t previous = -1;
    for (std::size_t i = 0; i < count; ++i) {
        const std::int32_t index = visible[i];
        if (index < 0 || static_cast<std::size_t>(index) >= key_count ||
            static_cast<std::size_t>(index) >= position_count || (i != 0 && index <= previous)) {
            return status::invalid_visible_index;
        }
        previous = index;
    }
    return status::ok;
}

[[nodiscard]] bool better(const candidate& lhs, const candidate& rhs) noexcept {
    return lhs.score > rhs.score || (lhs.score == rhs.score && lhs.block < rhs.block);
}

void rotate_partial_host(
    float* vector,
    const float* cos,
    const float* sin,
    std::uint32_t rotary_dim) noexcept {
    const std::uint32_t half = rotary_dim / 2U;
    for (std::uint32_t i = 0; i < half; ++i) {
        const float first = vector[i];
        const float second = vector[i + half];
        vector[i] = first * cos[i] - second * sin[i];
        vector[i + half] = second * cos[i + half] + first * sin[i + half];
    }
}

[[nodiscard]] status validate_host_common(
    const config& cfg,
    std::size_t count,
    const void* first,
    const void* second,
    const void* third,
    const void* fourth,
    void* output) noexcept {
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    const status count_status = validate_count(cfg, count);
    if (count_status != status::ok) {
        return count_status;
    }
    if (count != 0 && (first == nullptr || second == nullptr || third == nullptr || fourth == nullptr ||
                       output == nullptr)) {
        return status::null_pointer;
    }
    return status::ok;
}

[[nodiscard]] status cuda_result(cudaError_t result) noexcept {
    return result == cudaSuccess ? status::ok : status::cuda_failure;
}

[[nodiscard]] status validate_cuda_common(
    const config& cfg,
    std::size_t count,
    const void* first,
    const void* second,
    const void* third,
    const void* fourth,
    void* output,
    const status* device_status) noexcept {
    const status host_status = validate_host_common(cfg, count, first, second, third, fourth, output);
    if (host_status != status::ok) {
        return host_status;
    }
    return device_status == nullptr ? status::null_pointer : status::ok;
}

__device__ void report_device_error(status* output, status value) {
    if (output != nullptr) {
        atomicCAS(
            reinterpret_cast<unsigned int*>(output),
            static_cast<unsigned int>(status::ok),
            static_cast<unsigned int>(value));
    }
}

template <typename T>
__device__ float load_as_float(const T* values, std::size_t index) {
    return static_cast<float>(values[index]);
}

template <>
__device__ float load_as_float<__nv_bfloat16>(const __nv_bfloat16* values, std::size_t index) {
    return __bfloat162float(values[index]);
}

template <typename T>
__global__ void prepare_queries_kernel(
    config cfg,
    const T* raw_queries,
    const T* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t row_count,
    float* output,
    status* device_status) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= row_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t threads = blockDim.x;
    extern __shared__ float shared[];
    float* values = shared;
    float* reduction = shared + threads;

    float raw = 0.0F;
    float weight = 0.0F;
    if (lane < cfg.head_dim) {
        raw = load_as_float(raw_queries, row * cfg.head_dim + lane);
        weight = load_as_float(rms_weight, lane);
        if (!isfinite(raw) || !isfinite(weight)) {
            report_device_error(device_status, status::non_finite);
            raw = 0.0F;
            weight = 0.0F;
        }
        values[lane] = raw;
        const float square = raw * raw;
        if (!isfinite(square)) {
            report_device_error(device_status, status::non_finite);
            reduction[lane] = 0.0F;
        } else {
            reduction[lane] = square;
        }
    } else {
        values[lane] = 0.0F;
        reduction[lane] = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t stride = threads / 2U; stride != 0; stride /= 2U) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        reduction[0] = rsqrtf(reduction[0] / static_cast<float>(cfg.head_dim) + cfg.rms_epsilon);
        if (!isfinite(reduction[0])) {
            report_device_error(device_status, status::non_finite);
            reduction[0] = 0.0F;
        }
    }
    __syncthreads();

    if (lane < cfg.head_dim) {
        values[lane] = values[lane] * reduction[0] * (1.0F + weight);
        if (!isfinite(values[lane])) {
            report_device_error(device_status, status::non_finite);
            values[lane] = 0.0F;
        }
    }
    __syncthreads();

    if (lane < cfg.head_dim) {
        float result = values[lane];
        if (lane < cfg.rotary_dim) {
            const std::uint32_t half = cfg.rotary_dim / 2U;
            const std::uint32_t pair = lane < half ? lane + half : lane - half;
            const std::size_t query = row / cfg.query_heads;
            const float cosine = cos[query * cfg.rotary_dim + lane];
            const float sine = sin[query * cfg.rotary_dim + lane];
            if (!isfinite(cosine) || !isfinite(sine)) {
                report_device_error(device_status, status::non_finite);
                result = 0.0F;
            } else {
                const float rotated = lane < half ? -values[pair] : values[pair];
                result = values[lane] * cosine + rotated * sine;
                if (!isfinite(result)) {
                    report_device_error(device_status, status::non_finite);
                    result = 0.0F;
                }
            }
        }
        output[row * cfg.head_dim + lane] = result;
    }
}

__global__ void validate_visible_kernel(
    const std::int32_t* visible,
    std::size_t count,
    std::size_t key_count,
    std::size_t position_count,
    status* device_status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::int32_t value = visible[index];
    if (value < 0 || static_cast<std::size_t>(value) >= key_count ||
        static_cast<std::size_t>(value) >= position_count || (index != 0 && value <= visible[index - 1])) {
        report_device_error(device_status, status::invalid_visible_index);
    }
}

template <typename T>
__global__ void pool_keys_kernel(
    config cfg,
    const T* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible,
    std::size_t block_count,
    const T* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    status* device_status) {
    const std::size_t group = static_cast<std::size_t>(blockIdx.x);
    if (group >= block_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t threads = blockDim.x;
    extern __shared__ float shared[];
    float* values = shared;
    float* reduction = shared + threads;

    bool valid_group = true;
    const std::size_t visible_offset = group * cfg.compress_ratio;
    for (std::uint32_t item = 0; item < cfg.compress_ratio; ++item) {
        const std::int32_t token = visible[visible_offset + item];
        valid_group = valid_group && token >= 0 && static_cast<std::size_t>(token) < key_count &&
                      static_cast<std::size_t>(token) < position_count;
    }
    if (!valid_group) {
        report_device_error(device_status, status::invalid_visible_index);
    }

    float pooled = 0.0F;
    float weight = 0.0F;
    if (lane < cfg.head_dim && valid_group) {
        for (std::uint32_t item = 0; item < cfg.compress_ratio; ++item) {
            const std::size_t token = static_cast<std::size_t>(visible[visible_offset + item]);
            const float value = load_as_float(raw_keys, token * cfg.head_dim + lane);
            if (!isfinite(value)) {
                report_device_error(device_status, status::non_finite);
            } else {
                pooled += value;
            }
        }
        pooled /= static_cast<float>(cfg.compress_ratio);
        weight = load_as_float(rms_weight, lane);
        if (!isfinite(weight)) {
            report_device_error(device_status, status::non_finite);
            weight = 0.0F;
        }
        values[lane] = pooled;
        const float square = pooled * pooled;
        if (!isfinite(pooled) || !isfinite(square)) {
            report_device_error(device_status, status::non_finite);
            values[lane] = 0.0F;
            reduction[lane] = 0.0F;
        } else {
            reduction[lane] = square;
        }
    } else {
        values[lane] = 0.0F;
        reduction[lane] = 0.0F;
    }
    __syncthreads();

    for (std::uint32_t stride = threads / 2U; stride != 0; stride /= 2U) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        reduction[0] = rsqrtf(reduction[0] / static_cast<float>(cfg.head_dim) + cfg.rms_epsilon);
        if (!isfinite(reduction[0])) {
            report_device_error(device_status, status::non_finite);
            reduction[0] = 0.0F;
        }
    }
    __syncthreads();

    if (lane < cfg.head_dim) {
        values[lane] = values[lane] * reduction[0] * (1.0F + weight);
        if (!isfinite(values[lane])) {
            report_device_error(device_status, status::non_finite);
            values[lane] = 0.0F;
        }
    }
    __syncthreads();

    if (lane < cfg.head_dim) {
        float result = values[lane];
        if (lane < cfg.rotary_dim && valid_group) {
            const std::uint32_t half = cfg.rotary_dim / 2U;
            const std::uint32_t pair = lane < half ? lane + half : lane - half;
            const std::size_t position = static_cast<std::size_t>(visible[visible_offset]);
            const float cosine = full_cos[position * cfg.rotary_dim + lane];
            const float sine = full_sin[position * cfg.rotary_dim + lane];
            if (!isfinite(cosine) || !isfinite(sine)) {
                report_device_error(device_status, status::non_finite);
                result = 0.0F;
            } else {
                const float rotated = lane < half ? -values[pair] : values[pair];
                result = values[lane] * cosine + rotated * sine;
                if (!isfinite(result)) {
                    report_device_error(device_status, status::non_finite);
                    result = 0.0F;
                }
            }
        }
        output[group * cfg.head_dim + lane] = valid_group ? result : 0.0F;
    }
}

__global__ void score_blocks_kernel(
    config cfg,
    const float* query,
    const float* keys,
    std::size_t block_count,
    float* scores,
    status* device_status) {
    const std::size_t key_block = static_cast<std::size_t>(blockIdx.x);
    if (key_block >= block_count) {
        return;
    }
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t threads = blockDim.x;
    extern __shared__ float shared[];

    if (lane == 0) {
        shared[cfg.query_heads * threads] = 0.0F;
    }
    for (std::uint32_t head = 0; head < cfg.query_heads; ++head) {
        float product = 0.0F;
        if (lane < cfg.head_dim) {
            const float q = query[static_cast<std::size_t>(head) * cfg.head_dim + lane];
            const float k = keys[key_block * cfg.head_dim + lane];
            if (!isfinite(q) || !isfinite(k)) {
                report_device_error(device_status, status::non_finite);
            } else {
                product = q * k;
                if (!isfinite(product)) {
                    report_device_error(device_status, status::non_finite);
                    product = 0.0F;
                }
            }
        }
        shared[static_cast<std::size_t>(head) * threads + lane] = product;
    }
    __syncthreads();

    for (std::uint32_t head = 0; head < cfg.query_heads; ++head) {
        float* head_values = shared + static_cast<std::size_t>(head) * threads;
        for (std::uint32_t stride = threads / 2U; stride != 0; stride /= 2U) {
            if (lane < stride) {
                head_values[lane] += head_values[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0) {
            if (!isfinite(head_values[0])) {
                report_device_error(device_status, status::non_finite);
            } else {
                shared[cfg.query_heads * threads] += fmaxf(head_values[0], 0.0F);
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        const float score = shared[cfg.query_heads * threads] / sqrtf(static_cast<float>(cfg.head_dim));
        if (!isfinite(score)) {
            report_device_error(device_status, status::non_finite);
            scores[key_block] = 0.0F;
        } else {
            scores[key_block] = score;
        }
    }
}

__global__ void validate_scores_and_init_kernel(
    const float* scores,
    std::size_t count,
    std::int32_t* indices,
    status* device_status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float value = scores[index];
    if (!isfinite(value)) {
        report_device_error(device_status, status::non_finite);
    }
    indices[index] = static_cast<std::int32_t>(index);
}

__global__ void expand_tokens_kernel(
    config cfg,
    const std::int32_t* sorted_blocks,
    std::size_t selected_blocks,
    const std::int32_t* visible,
    std::size_t complete_blocks,
    std::size_t visible_count,
    std::size_t key_count,
    std::int32_t* output,
    std::size_t output_count,
    status* device_status) {
    const std::size_t output_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= output_count) {
        return;
    }
    const std::size_t block_token_count = selected_blocks * cfg.compress_ratio;
    std::size_t visible_offset = 0;
    if (output_index < block_token_count) {
        const std::size_t rank = output_index / cfg.compress_ratio;
        const std::size_t within = output_index % cfg.compress_ratio;
        const std::int32_t block = sorted_blocks[rank];
        if (block < 0 || static_cast<std::size_t>(block) >= complete_blocks) {
            report_device_error(device_status, status::invalid_visible_index);
            output[output_index] = -1;
            return;
        }
        visible_offset = static_cast<std::size_t>(block) * cfg.compress_ratio + within;
    } else {
        visible_offset = complete_blocks * cfg.compress_ratio + (output_index - block_token_count);
    }
    if (visible_offset >= visible_count) {
        report_device_error(device_status, status::invalid_visible_index);
        output[output_index] = -1;
        return;
    }
    const std::int32_t token = visible[visible_offset];
    if (token < 0 || static_cast<std::size_t>(token) >= key_count) {
        report_device_error(device_status, status::invalid_visible_index);
        output[output_index] = -1;
        return;
    }
    output[output_index] = token;
}

__global__ void scatter_mask_kernel(
    const std::int32_t* selected,
    std::size_t count,
    std::size_t kv_length,
    std::uint8_t* mask,
    status* device_status) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const std::int32_t token = selected[index];
    if (token < 0 || static_cast<std::size_t>(token) >= kv_length) {
        report_device_error(device_status, status::invalid_visible_index);
        return;
    }
    mask[token] = 1U;
}

[[nodiscard]] status cub_temporary_bytes(std::size_t block_count, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    *out = 0;
    if (block_count == 0) {
        return status::ok;
    }
    if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return status::arithmetic_overflow;
    }
    std::size_t bytes = 0;
    const cudaError_t result = cub::DeviceRadixSort::SortPairsDescending(
        nullptr,
        bytes,
        static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr),
        static_cast<const std::int32_t*>(nullptr),
        static_cast<std::int32_t*>(nullptr),
        static_cast<int>(block_count),
        0,
        8 * sizeof(float),
        nullptr);
    if (result != cudaSuccess) {
        return status::cuda_failure;
    }
    *out = bytes;
    return status::ok;
}

struct device_selection_layout {
    float* sorted_scores = nullptr;
    std::int32_t* input_indices = nullptr;
    std::int32_t* sorted_indices = nullptr;
    void* cub_temporary = nullptr;
    std::size_t cub_bytes = 0;
};

[[nodiscard]] bool align_up(
    std::uintptr_t value,
    std::size_t alignment,
    std::uintptr_t* output) noexcept {
    if (output == nullptr || alignment == 0 || (alignment & (alignment - 1U)) != 0 ||
        value > std::numeric_limits<std::uintptr_t>::max() - (alignment - 1U)) {
        return false;
    }
    *output = (value + alignment - 1U) & ~(static_cast<std::uintptr_t>(alignment) - 1U);
    return true;
}

[[nodiscard]] status make_device_layout(
    void* workspace,
    std::size_t workspace_bytes,
    std::size_t block_count,
    device_selection_layout* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    *out = {};
    std::size_t cub_bytes = 0;
    status result = cub_temporary_bytes(block_count, &cub_bytes);
    if (result != status::ok) {
        return result;
    }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(workspace);
    const std::uintptr_t end = base + workspace_bytes;
    if (end < base) {
        return status::arithmetic_overflow;
    }
    std::uintptr_t cursor = 0;
    if (!align_up(base, k_workspace_alignment, &cursor)) {
        return status::arithmetic_overflow;
    }
    std::size_t array_bytes = 0;
    if (mul_overflow(block_count, sizeof(float), &array_bytes) || cursor > end ||
        array_bytes > static_cast<std::size_t>(end - cursor)) {
        return status::capacity_exceeded;
    }
    out->sorted_scores = reinterpret_cast<float*>(cursor);
    cursor += array_bytes;
    if (!align_up(cursor, alignof(std::int32_t), &cursor)) {
        return status::arithmetic_overflow;
    }
    if (mul_overflow(block_count, sizeof(std::int32_t), &array_bytes) || cursor > end ||
        array_bytes > static_cast<std::size_t>(end - cursor)) {
        return status::capacity_exceeded;
    }
    out->input_indices = reinterpret_cast<std::int32_t*>(cursor);
    cursor += array_bytes;
    if (!align_up(cursor, alignof(std::int32_t), &cursor)) {
        return status::arithmetic_overflow;
    }
    if (cursor > end || array_bytes > static_cast<std::size_t>(end - cursor)) {
        return status::capacity_exceeded;
    }
    out->sorted_indices = reinterpret_cast<std::int32_t*>(cursor);
    cursor += array_bytes;
    if (!align_up(cursor, k_workspace_alignment, &cursor)) {
        return status::arithmetic_overflow;
    }
    if (cursor > end || cub_bytes > static_cast<std::size_t>(end - cursor)) {
        return status::capacity_exceeded;
    }
    out->cub_temporary = reinterpret_cast<void*>(cursor);
    out->cub_bytes = cub_bytes;
    return status::ok;
}

template <typename T>
[[nodiscard]] status launch_prepare_queries(
    const config& cfg,
    const T* raw_queries,
    const T* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output,
    status* device_status,
    cudaStream_t stream) noexcept {
    const status validation = validate_cuda_common(
        cfg, query_count, raw_queries, rms_weight, cos, sin, output, device_status);
    if (validation != status::ok || query_count == 0) {
        return validation;
    }
    std::size_t rows = 0;
    if (mul_overflow(query_count, cfg.query_heads, &rows) ||
        rows > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        return status::arithmetic_overflow;
    }
    const std::uint32_t threads = next_power_of_two(cfg.head_dim);
    const std::size_t shared_bytes = 2U * threads * sizeof(float);
    prepare_queries_kernel<<<static_cast<unsigned int>(rows), threads, shared_bytes, stream>>>(
        cfg, raw_queries, rms_weight, cos, sin, rows, output, device_status);
    return cuda_result(cudaPeekAtLastError());
}

template <typename T>
[[nodiscard]] status launch_pool_keys(
    const config& cfg,
    const T* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const T* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count,
    status* device_status,
    cudaStream_t stream) noexcept {
    if (output_block_count == nullptr || device_status == nullptr) {
        return status::null_pointer;
    }
    *output_block_count = 0;
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (key_count > cfg.max_context || position_count > cfg.max_context || visible_count > cfg.max_context) {
        return status::invalid_argument;
    }
    if (visible_count != 0 && visible_indices == nullptr) {
        return status::null_pointer;
    }
    std::size_t blocks = visible_count / cfg.compress_ratio;
    if (blocks > output_block_capacity) {
        return status::capacity_exceeded;
    }
    if (blocks != 0 &&
        (raw_keys == nullptr || rms_weight == nullptr || full_cos == nullptr || full_sin == nullptr || output == nullptr)) {
        return status::null_pointer;
    }
    *output_block_count = blocks;
    if (visible_count != 0) {
        constexpr unsigned int threads = 256;
        const unsigned int grid = static_cast<unsigned int>((visible_count + threads - 1U) / threads);
        validate_visible_kernel<<<grid, threads, 0, stream>>>(
            visible_indices, visible_count, key_count, position_count, device_status);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return status::cuda_failure;
        }
    }
    if (blocks == 0) {
        return status::ok;
    }
    if (blocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        return status::arithmetic_overflow;
    }
    const std::uint32_t threads = next_power_of_two(cfg.head_dim);
    const std::size_t shared_bytes = 2U * threads * sizeof(float);
    pool_keys_kernel<<<static_cast<unsigned int>(blocks), threads, shared_bytes, stream>>>(
        cfg,
        raw_keys,
        key_count,
        visible_indices,
        blocks,
        rms_weight,
        full_cos,
        full_sin,
        position_count,
        output,
        device_status);
    return cuda_result(cudaPeekAtLastError());
}

}  // namespace

const char* status_string(status value) noexcept {
    switch (value) {
        case status::ok: return "ok";
        case status::null_pointer: return "null pointer";
        case status::invalid_config: return "invalid config";
        case status::invalid_argument: return "invalid argument";
        case status::capacity_exceeded: return "capacity exceeded";
        case status::arithmetic_overflow: return "arithmetic overflow";
        case status::non_finite: return "non-finite input";
        case status::invalid_visible_index: return "invalid visible index";
        case status::cuda_failure: return "CUDA failure";
    }
    return "unknown status";
}

status validate_config(const config& cfg) noexcept {
    if (cfg.query_heads == 0 || cfg.kv_heads != 1 || cfg.head_dim == 0 || cfg.head_dim > 1024 ||
        cfg.rotary_dim > cfg.head_dim || (cfg.rotary_dim & 1U) != 0 || cfg.compress_ratio == 0 ||
        cfg.token_budget == 0 || cfg.token_budget % cfg.compress_ratio != 0 || cfg.max_context == 0 ||
        cfg.max_context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        !finite(cfg.rms_epsilon) || cfg.rms_epsilon <= 0.0F) {
        return status::invalid_config;
    }
    const std::uint32_t threads = next_power_of_two(cfg.head_dim);
    std::size_t score_shared = 0;
    if (mul_overflow(cfg.query_heads, threads, &score_shared) ||
        add_overflow(score_shared, 1, &score_shared) ||
        mul_overflow(score_shared, sizeof(float), &score_shared) || score_shared > k_max_score_shared_bytes) {
        return status::invalid_config;
    }
    const std::uint64_t maximum_selection = static_cast<std::uint64_t>(cfg.token_budget) + cfg.compress_ratio - 1U;
    return maximum_selection <= std::numeric_limits<std::size_t>::max() ? status::ok : status::invalid_config;
}

status validate_checkpoint_config(const config& cfg) noexcept {
    const status result = validate_config(cfg);
    return result == status::ok && same_config(cfg, checkpoint_config) ? status::ok : status::invalid_config;
}

status block_topk(const config& cfg, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    const status result = validate_config(cfg);
    if (result != status::ok) {
        return result;
    }
    *out = cfg.token_budget / cfg.compress_ratio;
    return status::ok;
}

status complete_block_count(const config& cfg, std::size_t visible_count, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    const status result = validate_config(cfg);
    if (result != status::ok) {
        return result;
    }
    if (visible_count > cfg.max_context) {
        return status::invalid_argument;
    }
    *out = visible_count / cfg.compress_ratio;
    return status::ok;
}

status selected_token_capacity(const config& cfg, std::size_t visible_count, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    std::size_t blocks = 0;
    status result = complete_block_count(cfg, visible_count, &blocks);
    if (result != status::ok) {
        return result;
    }
    std::size_t topk = 0;
    result = block_topk(cfg, &topk);
    if (result != status::ok) {
        return result;
    }
    const std::size_t selected_blocks = std::min(blocks, topk);
    std::size_t count = 0;
    if (mul_overflow(selected_blocks, cfg.compress_ratio, &count) ||
        add_overflow(count, visible_count % cfg.compress_ratio, &count)) {
        return status::arithmetic_overflow;
    }
    *out = count;
    return status::ok;
}

status host_select_workspace_bytes(const config& cfg, std::size_t block_count, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    std::size_t topk = 0;
    const status result = block_topk(cfg, &topk);
    if (result != status::ok) {
        return result;
    }
    if (block_count > cfg.max_context / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    const std::size_t keep = std::min(block_count, topk);
    if (keep == 0) {
        *out = 0;
        return status::ok;
    }
    std::size_t bytes = 0;
    if (mul_overflow(keep, sizeof(candidate), &bytes) ||
        add_overflow(bytes, alignof(candidate) - 1U, out)) {
        return status::arithmetic_overflow;
    }
    return status::ok;
}

status cuda_select_workspace_bytes(const config& cfg, std::size_t block_count, std::size_t* out) noexcept {
    if (out == nullptr) {
        return status::null_pointer;
    }
    *out = 0;
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (block_count > cfg.max_context / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    if (block_count == 0) {
        return status::ok;
    }
    std::size_t cub_bytes = 0;
    status result = cub_temporary_bytes(block_count, &cub_bytes);
    if (result != status::ok) {
        return result;
    }
    std::size_t array_bytes = 0;
    std::size_t total = k_workspace_alignment - 1U;
    if (mul_overflow(block_count, sizeof(float), &array_bytes) || add_overflow(total, array_bytes, &total) ||
        add_overflow(total, alignof(std::int32_t) - 1U, &total) ||
        mul_overflow(block_count, sizeof(std::int32_t), &array_bytes) || add_overflow(total, array_bytes, &total) ||
        add_overflow(total, alignof(std::int32_t) - 1U, &total) || add_overflow(total, array_bytes, &total) ||
        add_overflow(total, k_workspace_alignment - 1U, &total) || add_overflow(total, cub_bytes, &total)) {
        return status::arithmetic_overflow;
    }
    *out = total;
    return status::ok;
}

status prepare_queries_host_f32(
    const config& cfg,
    const float* raw_queries,
    const float* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output) noexcept {
    const status validation = validate_host_common(
        cfg, query_count, raw_queries, rms_weight, cos, sin, output);
    if (validation != status::ok || query_count == 0) {
        return validation;
    }
    for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
        if (!finite(rms_weight[dim])) {
            return status::non_finite;
        }
    }
    for (std::size_t query = 0; query < query_count; ++query) {
        for (std::uint32_t dim = 0; dim < cfg.rotary_dim; ++dim) {
            if (!finite(cos[query * cfg.rotary_dim + dim]) || !finite(sin[query * cfg.rotary_dim + dim])) {
                return status::non_finite;
            }
        }
        for (std::uint32_t head = 0; head < cfg.query_heads; ++head) {
            const std::size_t row = query * cfg.query_heads + head;
            const float* input = raw_queries + row * cfg.head_dim;
            float* result = output + row * cfg.head_dim;
            double sum_squares = 0.0;
            for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
                if (!finite(input[dim])) {
                    return status::non_finite;
                }
                sum_squares += static_cast<double>(input[dim]) * input[dim];
            }
            const float inverse = 1.0F / std::sqrt(
                static_cast<float>(sum_squares / cfg.head_dim) + cfg.rms_epsilon);
            if (!finite(inverse)) {
                return status::non_finite;
            }
            for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
                result[dim] = input[dim] * inverse * (1.0F + rms_weight[dim]);
                if (!finite(result[dim])) {
                    return status::non_finite;
                }
            }
            rotate_partial_host(
                result,
                cos + query * cfg.rotary_dim,
                sin + query * cfg.rotary_dim,
                cfg.rotary_dim);
            for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
                if (!finite(result[dim])) {
                    return status::non_finite;
                }
            }
        }
    }
    return status::ok;
}

status pool_keys_host_f32(
    const config& cfg,
    const float* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const float* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count) noexcept {
    if (output_block_count == nullptr) {
        return status::null_pointer;
    }
    *output_block_count = 0;
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (key_count > cfg.max_context || position_count > cfg.max_context || visible_count > cfg.max_context) {
        return status::invalid_argument;
    }
    const status visible_status = validate_visible_host(
        visible_indices, visible_count, key_count, position_count);
    if (visible_status != status::ok) {
        return visible_status;
    }
    const std::size_t blocks = visible_count / cfg.compress_ratio;
    if (blocks > output_block_capacity) {
        return status::capacity_exceeded;
    }
    if (blocks != 0 &&
        (raw_keys == nullptr || rms_weight == nullptr || full_cos == nullptr || full_sin == nullptr || output == nullptr)) {
        return status::null_pointer;
    }
    for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
        if (!finite(rms_weight[dim])) {
            return status::non_finite;
        }
    }
    for (std::size_t block = 0; block < blocks; ++block) {
        float* result = output + block * cfg.head_dim;
        double sum_squares = 0.0;
        for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
            double sum = 0.0;
            for (std::uint32_t item = 0; item < cfg.compress_ratio; ++item) {
                const std::size_t token = static_cast<std::size_t>(
                    visible_indices[block * cfg.compress_ratio + item]);
                const float value = raw_keys[token * cfg.head_dim + dim];
                if (!finite(value)) {
                    return status::non_finite;
                }
                sum += value;
            }
            result[dim] = static_cast<float>(sum / cfg.compress_ratio);
            if (!finite(result[dim])) {
                return status::non_finite;
            }
            sum_squares += static_cast<double>(result[dim]) * result[dim];
        }
        const float inverse = 1.0F / std::sqrt(
            static_cast<float>(sum_squares / cfg.head_dim) + cfg.rms_epsilon);
        if (!finite(inverse)) {
            return status::non_finite;
        }
        for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
            result[dim] = result[dim] * inverse * (1.0F + rms_weight[dim]);
            if (!finite(result[dim])) {
                return status::non_finite;
            }
        }
        const std::size_t position = static_cast<std::size_t>(
            visible_indices[block * cfg.compress_ratio]);
        const float* cosine = full_cos + position * cfg.rotary_dim;
        const float* sine = full_sin + position * cfg.rotary_dim;
        for (std::uint32_t dim = 0; dim < cfg.rotary_dim; ++dim) {
            if (!finite(cosine[dim]) || !finite(sine[dim])) {
                return status::non_finite;
            }
        }
        rotate_partial_host(result, cosine, sine, cfg.rotary_dim);
        for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
            if (!finite(result[dim])) {
                return status::non_finite;
            }
        }
    }
    *output_block_count = blocks;
    return status::ok;
}

status score_blocks_host_f32(
    const config& cfg,
    const float* prepared_query,
    const float* pooled_keys,
    std::size_t block_count,
    float* scores) noexcept {
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (block_count > cfg.max_context / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    if (block_count != 0 && (prepared_query == nullptr || pooled_keys == nullptr || scores == nullptr)) {
        return status::null_pointer;
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(cfg.head_dim));
    for (std::size_t block = 0; block < block_count; ++block) {
        double score = 0.0;
        for (std::uint32_t head = 0; head < cfg.query_heads; ++head) {
            double dot = 0.0;
            for (std::uint32_t dim = 0; dim < cfg.head_dim; ++dim) {
                const float q = prepared_query[static_cast<std::size_t>(head) * cfg.head_dim + dim];
                const float k = pooled_keys[block * cfg.head_dim + dim];
                if (!finite(q) || !finite(k)) {
                    return status::non_finite;
                }
                dot += static_cast<double>(q) * k;
            }
            score += std::max(dot, 0.0);
        }
        scores[block] = static_cast<float>(score) * scale;
        if (!finite(scores[block])) {
            return status::non_finite;
        }
    }
    return status::ok;
}

status select_tokens_host_f32(
    const config& cfg,
    const float* scores,
    std::size_t block_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    std::size_t key_count,
    std::int32_t* selected_tokens,
    std::size_t selected_capacity,
    void* workspace,
    std::size_t workspace_bytes,
    std::size_t* selected_count) noexcept {
    if (selected_count == nullptr) {
        return status::null_pointer;
    }
    *selected_count = 0;
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (key_count > cfg.max_context || visible_count > cfg.max_context ||
        block_count != visible_count / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    const status visible_status = validate_visible_host(
        visible_indices, visible_count, key_count, key_count);
    if (visible_status != status::ok) {
        return visible_status;
    }
    std::size_t required_output = 0;
    status result = selected_token_capacity(cfg, visible_count, &required_output);
    if (result != status::ok) {
        return result;
    }
    if (required_output > selected_capacity) {
        return status::capacity_exceeded;
    }
    if (required_output != 0 && selected_tokens == nullptr) {
        return status::null_pointer;
    }
    std::size_t required_workspace = 0;
    result = host_select_workspace_bytes(cfg, block_count, &required_workspace);
    if (result != status::ok) {
        return result;
    }
    if (required_workspace > workspace_bytes || (required_workspace != 0 && workspace == nullptr)) {
        return status::capacity_exceeded;
    }
    if (block_count != 0 && scores == nullptr) {
        return status::null_pointer;
    }

    std::size_t topk = 0;
    result = block_topk(cfg, &topk);
    if (result != status::ok) {
        return result;
    }
    const std::size_t keep = std::min(block_count, topk);
    std::uintptr_t aligned_workspace = 0;
    if (required_workspace != 0 &&
        !align_up(reinterpret_cast<std::uintptr_t>(workspace), alignof(candidate), &aligned_workspace)) {
        return status::arithmetic_overflow;
    }
    auto* best = reinterpret_cast<candidate*>(aligned_workspace);
    std::size_t filled = 0;
    for (std::size_t block = 0; block < block_count; ++block) {
        if (!finite(scores[block])) {
            return status::non_finite;
        }
        const candidate current{scores[block], static_cast<std::uint32_t>(block)};
        std::size_t insertion = 0;
        while (insertion < filled && better(best[insertion], current)) {
            ++insertion;
        }
        if (insertion >= keep) {
            continue;
        }
        if (filled < keep) {
            ++filled;
        }
        for (std::size_t move = filled - 1; move > insertion; --move) {
            best[move] = best[move - 1];
        }
        best[insertion] = current;
    }

    std::size_t output_index = 0;
    for (std::size_t rank = 0; rank < keep; ++rank) {
        const std::size_t block = best[rank].block;
        for (std::uint32_t item = 0; item < cfg.compress_ratio; ++item) {
            selected_tokens[output_index++] = visible_indices[block * cfg.compress_ratio + item];
        }
    }
    const std::size_t tail_start = block_count * cfg.compress_ratio;
    for (std::size_t tail = tail_start; tail < visible_count; ++tail) {
        selected_tokens[output_index++] = visible_indices[tail];
    }
    *selected_count = output_index;
    return status::ok;
}

status build_mask_host(
    const std::int32_t* selected_tokens,
    std::size_t selected_count,
    std::size_t kv_length,
    std::uint8_t* mask) noexcept {
    if (kv_length != 0 && mask == nullptr) {
        return status::null_pointer;
    }
    if (selected_count != 0 && selected_tokens == nullptr) {
        return status::null_pointer;
    }
    if (kv_length > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return status::invalid_argument;
    }
    if (kv_length != 0) {
        std::memset(mask, 0, kv_length);
    }
    for (std::size_t i = 0; i < selected_count; ++i) {
        const std::int32_t token = selected_tokens[i];
        if (token < 0 || static_cast<std::size_t>(token) >= kv_length) {
            return status::invalid_visible_index;
        }
        mask[token] = 1U;
    }
    return status::ok;
}

status cuda_reset_status(status* device_status, cudaStream_t stream) noexcept {
    if (device_status == nullptr) {
        return status::null_pointer;
    }
    return cuda_result(cudaMemsetAsync(device_status, 0, sizeof(status), stream));
}

status cuda_collect_status(const status* device_status, cudaStream_t stream, status* host_status) noexcept {
    if (device_status == nullptr || host_status == nullptr) {
        return status::null_pointer;
    }
    std::uint32_t raw = 0;
    cudaError_t result = cudaMemcpyAsync(&raw, device_status, sizeof(raw), cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) {
        return status::cuda_failure;
    }
    result = cudaStreamSynchronize(stream);
    if (result != cudaSuccess || raw > static_cast<std::uint32_t>(status::cuda_failure)) {
        return status::cuda_failure;
    }
    *host_status = static_cast<status>(raw);
    return status::ok;
}

status prepare_queries_cuda_f32(
    const config& cfg,
    const float* raw_queries,
    const float* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output,
    status* device_status,
    cudaStream_t stream) noexcept {
    return launch_prepare_queries(
        cfg, raw_queries, rms_weight, cos, sin, query_count, output, device_status, stream);
}

status prepare_queries_cuda_bf16(
    const config& cfg,
    const __nv_bfloat16* raw_queries,
    const __nv_bfloat16* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output,
    status* device_status,
    cudaStream_t stream) noexcept {
    return launch_prepare_queries(
        cfg, raw_queries, rms_weight, cos, sin, query_count, output, device_status, stream);
}

status pool_keys_cuda_f32(
    const config& cfg,
    const float* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const float* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count,
    status* device_status,
    cudaStream_t stream) noexcept {
    return launch_pool_keys(
        cfg,
        raw_keys,
        key_count,
        visible_indices,
        visible_count,
        rms_weight,
        full_cos,
        full_sin,
        position_count,
        output,
        output_block_capacity,
        output_block_count,
        device_status,
        stream);
}

status pool_keys_cuda_bf16(
    const config& cfg,
    const __nv_bfloat16* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const __nv_bfloat16* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count,
    status* device_status,
    cudaStream_t stream) noexcept {
    return launch_pool_keys(
        cfg,
        raw_keys,
        key_count,
        visible_indices,
        visible_count,
        rms_weight,
        full_cos,
        full_sin,
        position_count,
        output,
        output_block_capacity,
        output_block_count,
        device_status,
        stream);
}

status score_blocks_cuda_f32(
    const config& cfg,
    const float* prepared_query,
    const float* pooled_keys,
    std::size_t block_count,
    float* scores,
    status* device_status,
    cudaStream_t stream) noexcept {
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (block_count > cfg.max_context / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    if (device_status == nullptr ||
        (block_count != 0 && (prepared_query == nullptr || pooled_keys == nullptr || scores == nullptr))) {
        return status::null_pointer;
    }
    if (block_count == 0) {
        return status::ok;
    }
    if (block_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
        return status::arithmetic_overflow;
    }
    const std::uint32_t threads = next_power_of_two(cfg.head_dim);
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(cfg.query_heads) * threads + 1U) * sizeof(float);
    score_blocks_kernel<<<static_cast<unsigned int>(block_count), threads, shared_bytes, stream>>>(
        cfg, prepared_query, pooled_keys, block_count, scores, device_status);
    return cuda_result(cudaPeekAtLastError());
}

status select_tokens_cuda_f32(
    const config& cfg,
    const float* scores,
    std::size_t block_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    std::size_t key_count,
    std::int32_t* selected_tokens,
    std::size_t selected_capacity,
    void* workspace,
    std::size_t workspace_bytes,
    std::size_t* selected_count,
    status* device_status,
    cudaStream_t stream) noexcept {
    if (selected_count == nullptr || device_status == nullptr) {
        return status::null_pointer;
    }
    *selected_count = 0;
    const status cfg_status = validate_config(cfg);
    if (cfg_status != status::ok) {
        return cfg_status;
    }
    if (key_count > cfg.max_context || visible_count > cfg.max_context ||
        block_count != visible_count / cfg.compress_ratio) {
        return status::invalid_argument;
    }
    if (visible_count != 0 && visible_indices == nullptr) {
        return status::null_pointer;
    }
    std::size_t output_count = 0;
    status result = selected_token_capacity(cfg, visible_count, &output_count);
    if (result != status::ok) {
        return result;
    }
    if (output_count > selected_capacity) {
        return status::capacity_exceeded;
    }
    if (output_count != 0 && selected_tokens == nullptr) {
        return status::null_pointer;
    }
    if (visible_count != 0) {
        constexpr unsigned int threads = 256;
        const unsigned int grid = static_cast<unsigned int>((visible_count + threads - 1U) / threads);
        validate_visible_kernel<<<grid, threads, 0, stream>>>(
            visible_indices, visible_count, key_count, key_count, device_status);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return status::cuda_failure;
        }
    }

    std::size_t topk = 0;
    result = block_topk(cfg, &topk);
    if (result != status::ok) {
        return result;
    }
    const std::size_t selected_blocks = std::min(block_count, topk);
    const std::int32_t* sorted_indices = nullptr;
    device_selection_layout layout{};
    if (block_count != 0) {
        if (scores == nullptr || workspace == nullptr) {
            return status::null_pointer;
        }
        std::size_t required_workspace = 0;
        result = cuda_select_workspace_bytes(cfg, block_count, &required_workspace);
        if (result != status::ok) {
            return result;
        }
        if (workspace_bytes < required_workspace) {
            return status::capacity_exceeded;
        }
        result = make_device_layout(workspace, workspace_bytes, block_count, &layout);
        if (result != status::ok) {
            return result;
        }
        constexpr unsigned int threads = 256;
        const unsigned int grid = static_cast<unsigned int>((block_count + threads - 1U) / threads);
        validate_scores_and_init_kernel<<<grid, threads, 0, stream>>>(
            scores, block_count, layout.input_indices, device_status);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return status::cuda_failure;
        }
        const cudaError_t sort_result = cub::DeviceRadixSort::SortPairsDescending(
            layout.cub_temporary,
            layout.cub_bytes,
            scores,
            layout.sorted_scores,
            layout.input_indices,
            layout.sorted_indices,
            static_cast<int>(block_count),
            0,
            8 * sizeof(float),
            stream);
        if (sort_result != cudaSuccess) {
            return status::cuda_failure;
        }
        sorted_indices = layout.sorted_indices;
    }

    if (output_count != 0) {
        constexpr unsigned int threads = 256;
        const unsigned int grid = static_cast<unsigned int>((output_count + threads - 1U) / threads);
        expand_tokens_kernel<<<grid, threads, 0, stream>>>(
            cfg,
            sorted_indices,
            selected_blocks,
            visible_indices,
            block_count,
            visible_count,
            key_count,
            selected_tokens,
            output_count,
            device_status);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return status::cuda_failure;
        }
    }
    *selected_count = output_count;
    return status::ok;
}

status build_mask_cuda(
    const std::int32_t* selected_tokens,
    std::size_t selected_count,
    std::size_t kv_length,
    std::uint8_t* mask,
    status* device_status,
    cudaStream_t stream) noexcept {
    if (device_status == nullptr || (kv_length != 0 && mask == nullptr) ||
        (selected_count != 0 && selected_tokens == nullptr)) {
        return status::null_pointer;
    }
    if (kv_length > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        return status::invalid_argument;
    }
    if (kv_length != 0) {
        const cudaError_t memset_result = cudaMemsetAsync(mask, 0, kv_length, stream);
        if (memset_result != cudaSuccess) {
            return status::cuda_failure;
        }
    }
    if (selected_count != 0) {
        constexpr unsigned int threads = 256;
        if (selected_count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) * threads) {
            return status::arithmetic_overflow;
        }
        const unsigned int grid = static_cast<unsigned int>((selected_count + threads - 1U) / threads);
        scatter_mask_kernel<<<grid, threads, 0, stream>>>(
            selected_tokens, selected_count, kv_length, mask, device_status);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return status::cuda_failure;
        }
    }
    return status::ok;
}

}  // namespace axiom::qwen4exp::qsa
