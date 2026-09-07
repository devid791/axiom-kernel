#include "axiom/qwen4exp/output_head.hpp"

#include "axiom/qwen4exp/bf16_linear.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kThreads = 256u;
constexpr std::size_t kMaximumTokens = 64u;
constexpr std::size_t kMinimumUploadChunk = 4u * 1024u;
constexpr std::size_t kMaximumUploadChunk = 64u * 1024u * 1024u;
constexpr std::size_t kMaximumBlasWorkspace = 64u * 1024u * 1024u;
constexpr std::size_t kWorkspaceAlignment = 256u;

constexpr char kEmbeddingName[] =
        "model.language_model.embed_tokens.weight";
constexpr char kLegacyFinalNormName[] =
        "model.language_model.norm.weight";
constexpr char kGlobalNormName[] =
        "model.language_model.hyper_connection_mixer.hc_norm.weight";
constexpr char kGlobalDownName[] =
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";
constexpr char kGlobalUpName[] =
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight";
constexpr char kLmHeadName[] = "lm_head.weight";

bool multiply_overflow(std::size_t left, std::size_t right,
                       std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return true;
    }
    *result = left * right;
    return false;
}

output_head_status fail(output_head_status status, std::string *error,
                        const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return status;
}

float bf16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0u) {
        return static_cast<std::uint16_t>((bits >> 16u) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float round_bf16(float value) noexcept {
    return bf16_to_float(float_to_bf16(value));
}

float sigmoid_stable(float value) noexcept {
    if (value >= 0.0F) {
        const float inverse = std::exp(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

bool current_blackwell_device(int *device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, *device) != cudaSuccess) return false;
    return properties.major == 12 && properties.minor == 0;
}

bool device_range_valid(const void *pointer, std::size_t required_bytes,
                        int expected_device, std::size_t alignment) noexcept {
    if (pointer == nullptr || required_bytes == 0u) return false;
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
    if (alignment != 0u && address % alignment != 0u) return false;

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status = cudaPointerGetAttributes(&attributes, pointer);
    if (pointer_status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        return false;
    }

    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    if (cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                             static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS ||
        allocation_base == 0u) {
        return false;
    }
    const std::uintptr_t base = static_cast<std::uintptr_t>(allocation_base);
    if (address < base) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes && required_bytes <= allocation_bytes - offset;
}

bool ranges_overlap(const void *left, std::size_t left_bytes,
                    const void *right, std::size_t right_bytes) noexcept {
    const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left);
    const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
        right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
        return true;
    }
    const std::uintptr_t left_end = left_begin + left_bytes;
    const std::uintptr_t right_end = right_begin + right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

struct device_range {
    const void *pointer = nullptr;
    std::size_t bytes = 0u;
    std::size_t alignment = 0u;
};

template <std::size_t Count>
bool valid_disjoint_ranges(const std::array<device_range, Count> &ranges,
                           int device) noexcept {
    for (const device_range &range : ranges) {
        if (!device_range_valid(range.pointer, range.bytes, device,
                                range.alignment)) {
            return false;
        }
    }
    for (std::size_t left = 0u; left < Count; ++left) {
        for (std::size_t right = left + 1u; right < Count; ++right) {
            if (ranges_overlap(ranges[left].pointer, ranges[left].bytes,
                               ranges[right].pointer, ranges[right].bytes)) {
                return false;
            }
        }
    }
    return true;
}

bool exact_span(const tensor_span *span, tensor_dtype dtype,
                std::uint32_t rank, std::initializer_list<std::uint64_t> shape,
                std::uint64_t bytes) noexcept {
    if (span == nullptr || span->dtype != dtype || span->rank != rank ||
        span->bytes != bytes || shape.size() != rank) {
        return false;
    }
    std::size_t index = 0u;
    for (const std::uint64_t dimension : shape) {
        if (span->shape[index++] != dimension) return false;
    }
    return true;
}

class pinned_buffer final {
public:
    pinned_buffer() = default;
    ~pinned_buffer() {
        if (pointer_ != nullptr) (void)cudaFreeHost(pointer_);
    }
    pinned_buffer(const pinned_buffer &) = delete;
    pinned_buffer &operator=(const pinned_buffer &) = delete;

    bool allocate(std::size_t bytes) noexcept {
        return bytes != 0u && pointer_ == nullptr &&
               cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault) == cudaSuccess;
    }
    void *get() const noexcept { return pointer_; }

private:
    void *pointer_ = nullptr;
};

output_head_status upload_tensor(const checkpoint_catalog &catalog,
                                 const tensor_span &span,
                                 std::size_t upload_chunk_bytes,
                                 cudaStream_t stream,
                                 std::uint16_t **destination,
                                 std::string *error) noexcept {
    if (destination == nullptr || span.bytes == 0u ||
        span.bytes > static_cast<std::uint64_t>(
                             std::numeric_limits<std::size_t>::max())) {
        return output_head_status::size_overflow;
    }
    *destination = nullptr;
    const std::size_t tensor_bytes = static_cast<std::size_t>(span.bytes);
    if (cudaMalloc(reinterpret_cast<void **>(destination), tensor_bytes) !=
        cudaSuccess) {
        return fail(output_head_status::allocation_failure, error,
                    "device allocation failed for " + span.name);
    }
    const std::size_t staging_bytes = std::min(upload_chunk_bytes, tensor_bytes);
    pinned_buffer staging;
    if (!staging.allocate(staging_bytes)) {
        (void)cudaFree(*destination);
        *destination = nullptr;
        return fail(output_head_status::allocation_failure, error,
                    "pinned staging allocation failed for " + span.name);
    }

    std::uint64_t offset = 0u;
    while (offset < span.bytes) {
        const std::size_t amount = static_cast<std::size_t>(
                std::min<std::uint64_t>(staging_bytes, span.bytes - offset));
        std::string read_error;
        if (!catalog.read_range(span.name, offset, staging.get(), amount,
                                &read_error)) {
            (void)cudaFree(*destination);
            *destination = nullptr;
            return fail(output_head_status::checkpoint_io_error, error,
                        read_error.empty() ? "checkpoint read failed for " + span.name
                                           : read_error);
        }
        if (cudaMemcpyAsync(reinterpret_cast<unsigned char *>(*destination) + offset,
                            staging.get(), amount, cudaMemcpyHostToDevice,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaFree(*destination);
            *destination = nullptr;
            return fail(output_head_status::cuda_error, error,
                        "GPU upload failed for " + span.name);
        }
        offset += amount;
    }
    return output_head_status::ok;
}

output_head_status map_linear_status(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return output_head_status::ok;
        case bf16_linear_status::invalid_argument:
            return output_head_status::invalid_argument;
        case bf16_linear_status::unsupported_config:
            return output_head_status::unsupported_config;
        case bf16_linear_status::tensor_not_found:
            return output_head_status::tensor_not_found;
        case bf16_linear_status::dtype_mismatch:
        case bf16_linear_status::shape_mismatch:
            return output_head_status::tensor_contract_mismatch;
        case bf16_linear_status::size_overflow:
            return output_head_status::size_overflow;
        case bf16_linear_status::unsupported_device:
            return output_head_status::unsupported_device;
        case bf16_linear_status::invalid_device_pointer:
            return output_head_status::invalid_device_pointer;
        case bf16_linear_status::checkpoint_io_error:
            return output_head_status::checkpoint_io_error;
        case bf16_linear_status::allocation_failure:
            return output_head_status::allocation_failure;
        case bf16_linear_status::cuda_error:
            return output_head_status::cuda_error;
        case bf16_linear_status::cublas_error:
            return output_head_status::linear_error;
    }
    return output_head_status::linear_error;
}

__device__ float device_sigmoid(float value) {
    if (value >= 0.0F) {
        const float inverse = expf(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = expf(value);
    return exponential / (1.0F + exponential);
}

__global__ void embedding_repeat_kernel(const std::uint32_t *token_ids,
                                        std::size_t tokens,
                                        const std::uint16_t *embedding,
                                        float *hyper_input,
                                        std::uint32_t *device_status) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = tokens * kOutputHeadResidual;
    if (index >= elements) return;
    const std::size_t token = index / kOutputHeadResidual;
    const std::size_t hidden = index % kOutputHeadHidden;
    const std::uint32_t token_id = token_ids[token];
    if (token_id >= kOutputHeadVocab) {
        atomicExch(device_status, 1u);
        hyper_input[index] = 0.0F;
        return;
    }
    const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16 *>(
            embedding + static_cast<std::size_t>(token_id) * kOutputHeadHidden +
            hidden);
    hyper_input[index] = __bfloat162float(raw);
}

__global__ void grouped_rmsnorm_kernel(const float *input,
                                       const std::uint16_t *weight,
                                       std::size_t rows,
                                       float *output) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;
    float sum = 0.0F;
    for (std::size_t hidden = threadIdx.x; hidden < kOutputHeadHidden;
         hidden += blockDim.x) {
        const float value = input[row * kOutputHeadHidden + hidden];
        sum += value * value;
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse = rsqrtf(
            reduction[0] / static_cast<float>(kOutputHeadHidden) +
            kOutputHeadRmsEpsilon);
    const std::size_t stream = row % kOutputHeadStreams;
    for (std::size_t hidden = threadIdx.x; hidden < kOutputHeadHidden;
         hidden += blockDim.x) {
        const __nv_bfloat16 raw = *reinterpret_cast<const __nv_bfloat16 *>(
                weight + stream * kOutputHeadHidden + hidden);
        const std::size_t offset = row * kOutputHeadHidden + hidden;
        output[offset] = input[offset] * inverse *
                         (1.0F + __bfloat162float(raw));
    }
}

__global__ void scaled_silu_kernel(float *values, std::size_t elements) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float scaled = values[index] /
                         static_cast<float>(kOutputHeadStreams);
    values[index] = scaled * device_sigmoid(scaled);
}

__global__ void global_combine_kernel(const float *normalized,
                                      const float *mix_logits,
                                      std::size_t tokens,
                                      float *mixed_hidden) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t elements = tokens * kOutputHeadHidden;
    if (index >= elements) return;
    const std::size_t token = index / kOutputHeadHidden;
    const std::size_t hidden = index % kOutputHeadHidden;
    float sum = 0.0F;
    for (std::size_t stream = 0u; stream < kOutputHeadStreams; ++stream) {
        const std::size_t source = token * kOutputHeadResidual +
                                   stream * kOutputHeadHidden + hidden;
        sum += device_sigmoid(mix_logits[source]) * normalized[source];
    }
    mixed_hidden[index] = sum / static_cast<float>(kOutputHeadStreams);
}

struct argmax_pair {
    float value;
    std::uint32_t token;
};

__device__ argmax_pair better_pair(argmax_pair left,
                                   argmax_pair right) {
    if (right.value > left.value ||
        (right.value == left.value && right.token < left.token)) {
        return right;
    }
    return left;
}

__global__ void argmax_kernel(const float *logits, std::size_t tokens,
                              std::uint32_t *selected_tokens,
                              float *selected_logits) {
    const std::size_t batch = static_cast<std::size_t>(blockIdx.x);
    if (batch >= tokens) return;
    argmax_pair best{-FLT_MAX, 0u};
    const float *row = logits + batch * kOutputHeadVocab;
    for (std::size_t token = threadIdx.x; token < kOutputHeadVocab;
         token += blockDim.x) {
        const float value = row[token];
        if (!isnan(value)) {
            best = better_pair(best,
                               {value, static_cast<std::uint32_t>(token)});
        }
    }
    __shared__ argmax_pair reduction[kThreads];
    reduction[threadIdx.x] = best;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = better_pair(
                    reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        selected_tokens[batch] = reduction[0].token;
        selected_logits[batch] = reduction[0].value;
    }
}

__device__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31u);
}

__global__ void sample_kernel(const float *logits, std::size_t tokens,
                              float temperature, std::uint64_t seed,
                              std::uint64_t sequence,
                              std::uint32_t *selected_tokens,
                              float *selected_logits) {
    const std::size_t batch = static_cast<std::size_t>(blockIdx.x);
    if (batch >= tokens) return;
    const float *row = logits + batch * kOutputHeadVocab;

    float local_max = -FLT_MAX;
    for (std::size_t token = threadIdx.x; token < kOutputHeadVocab;
         token += blockDim.x) {
        const float value = row[token];
        if (!isnan(value)) local_max = fmaxf(local_max, value);
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] =
                    fmaxf(reduction[threadIdx.x],
                          reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduction[0];
    float local_sum = 0.0F;
    if (isfinite(maximum)) {
        for (std::size_t token = threadIdx.x; token < kOutputHeadVocab;
             token += blockDim.x) {
            const float value = row[token];
            if (!isnan(value)) {
                local_sum += expf((value - maximum) / temperature);
            }
        }
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x != 0u) return;

    const float total = reduction[0];
    if (!(total > 0.0F) || !isfinite(total)) {
        selected_tokens[batch] = 0u;
        selected_logits[batch] = row[0];
        return;
    }
    const std::uint64_t random_bits = splitmix64(
            seed ^ (sequence + 0x9e3779b97f4a7c15ULL * (batch + 1u)));
    const double uniform =
            (static_cast<double>(random_bits >> 11u) + 0.5) *
            (1.0 / 9007199254740992.0);
    const double target = uniform * static_cast<double>(total);
    double cumulative = 0.0;
    std::uint32_t chosen = static_cast<std::uint32_t>(kOutputHeadVocab - 1u);
    for (std::uint32_t token = 0u; token < kOutputHeadVocab; ++token) {
        const float value = row[token];
        if (!isnan(value)) {
            cumulative += static_cast<double>(
                    expf((value - maximum) / temperature));
        }
        if (cumulative >= target) {
            chosen = token;
            break;
        }
    }
    selected_tokens[batch] = chosen;
    selected_logits[batch] = row[chosen];
}

}  // namespace

struct resident_output_head::impl {
    output_head_config config{};
    output_head_workspace_requirements requirements{};
    output_head_tying tying = output_head_tying::untied_checkpoint_head;
    output_head_finalization finalization =
            output_head_finalization::global_hyper_connection_mixer;
    std::uint16_t *embedding = nullptr;
    std::uint16_t *global_norm = nullptr;
    std::unique_ptr<resident_bf16_linear> global_down;
    std::unique_ptr<resident_bf16_linear> global_up;
    std::unique_ptr<resident_bf16_linear> lm_head;
    std::uint64_t resident_bytes = 0u;
    int device = -1;
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && device >= 0 && previous != device &&
                             cudaSetDevice(device) == cudaSuccess;
        lm_head.reset();
        global_up.reset();
        global_down.reset();
        if (global_norm != nullptr) (void)cudaFree(global_norm);
        if (embedding != nullptr) (void)cudaFree(embedding);
        global_norm = nullptr;
        embedding = nullptr;
        if (changed) (void)cudaSetDevice(previous);
    }
};

const char *output_head_status_string(output_head_status status) noexcept {
    switch (status) {
        case output_head_status::ok: return "ok";
        case output_head_status::invalid_argument: return "invalid_argument";
        case output_head_status::unsupported_config: return "unsupported_config";
        case output_head_status::tensor_not_found: return "tensor_not_found";
        case output_head_status::tensor_contract_mismatch:
            return "tensor_contract_mismatch";
        case output_head_status::size_overflow: return "size_overflow";
        case output_head_status::unsupported_device: return "unsupported_device";
        case output_head_status::allocation_failure: return "allocation_failure";
        case output_head_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case output_head_status::checkpoint_io_error: return "checkpoint_io_error";
        case output_head_status::cuda_error: return "cuda_error";
        case output_head_status::linear_error: return "linear_error";
        case output_head_status::device_rejected_input:
            return "device_rejected_input";
    }
    return "unknown";
}

output_head_status output_head_validate_config(
        const output_head_config &config) noexcept {
    if (config.max_tokens == 0u) return output_head_status::invalid_argument;
    if (config.max_tokens > kMaximumTokens ||
        config.upload_chunk_bytes < kMinimumUploadChunk ||
        config.upload_chunk_bytes > kMaximumUploadChunk ||
        config.upload_chunk_bytes % sizeof(std::uint16_t) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > kMaximumBlasWorkspace ||
        config.blas_workspace_bytes % kWorkspaceAlignment != 0u) {
        return output_head_status::unsupported_config;
    }
    output_head_workspace_requirements requirements{};
    return output_head_get_workspace_requirements(config, &requirements);
}

output_head_status output_head_get_workspace_requirements(
        const output_head_config &config,
        output_head_workspace_requirements *requirements) noexcept {
    if (requirements == nullptr || config.max_tokens == 0u) {
        return output_head_status::invalid_argument;
    }
    if (config.max_tokens > kMaximumTokens ||
        config.upload_chunk_bytes < kMinimumUploadChunk ||
        config.upload_chunk_bytes > kMaximumUploadChunk ||
        config.upload_chunk_bytes % sizeof(std::uint16_t) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > kMaximumBlasWorkspace ||
        config.blas_workspace_bytes % kWorkspaceAlignment != 0u) {
        return output_head_status::unsupported_config;
    }
    output_head_workspace_requirements result{};
    std::size_t elements = 0u;
    if (multiply_overflow(config.max_tokens, kOutputHeadResidual, &elements) ||
        multiply_overflow(elements, sizeof(float),
                          &result.hyper_input_f32_bytes)) {
        return output_head_status::size_overflow;
    }
    result.normalized_f32_bytes = result.hyper_input_f32_bytes;
    result.mix_logits_f32_bytes = result.hyper_input_f32_bytes;
    if (multiply_overflow(config.max_tokens, kOutputHeadRank, &elements) ||
        multiply_overflow(elements, sizeof(float),
                          &result.low_rank_f32_bytes) ||
        multiply_overflow(config.max_tokens, kOutputHeadHidden, &elements) ||
        multiply_overflow(elements, sizeof(float),
                          &result.mixed_hidden_f32_bytes) ||
        multiply_overflow(config.max_tokens, kOutputHeadResidual, &elements) ||
        multiply_overflow(elements, sizeof(std::uint16_t),
                          &result.linear_input_bf16_bytes) ||
        multiply_overflow(config.max_tokens, kOutputHeadVocab, &elements) ||
        multiply_overflow(elements, sizeof(float), &result.logits_f32_bytes)) {
        return output_head_status::size_overflow;
    }
    result.blas_workspace_bytes = config.blas_workspace_bytes;
    *requirements = result;
    return output_head_status::ok;
}

output_head_status output_head_global_mix_reference(
        const float *hyper_input_f32, const std::uint16_t *hc_norm_bf16,
        const std::uint16_t *down_bf16, const std::uint16_t *up_bf16,
        std::size_t tokens, float *normalized_f32, float *low_rank_f32,
        float *mix_logits_f32, float *mixed_hidden_f32) noexcept {
    if (hyper_input_f32 == nullptr || hc_norm_bf16 == nullptr ||
        down_bf16 == nullptr || up_bf16 == nullptr || tokens == 0u ||
        tokens > kMaximumTokens || normalized_f32 == nullptr ||
        low_rank_f32 == nullptr || mix_logits_f32 == nullptr ||
        mixed_hidden_f32 == nullptr) {
        return output_head_status::invalid_argument;
    }
    for (std::size_t token = 0u; token < tokens; ++token) {
        for (std::size_t stream = 0u; stream < kOutputHeadStreams; ++stream) {
            const std::size_t base = token * kOutputHeadResidual +
                                     stream * kOutputHeadHidden;
            float square_sum = 0.0F;
            for (std::size_t hidden = 0u; hidden < kOutputHeadHidden; ++hidden) {
                const float value = hyper_input_f32[base + hidden];
                square_sum += value * value;
            }
            const float inverse = 1.0F / std::sqrt(
                    square_sum / static_cast<float>(kOutputHeadHidden) +
                    kOutputHeadRmsEpsilon);
            for (std::size_t hidden = 0u; hidden < kOutputHeadHidden; ++hidden) {
                const std::size_t offset = base + hidden;
                normalized_f32[offset] =
                        hyper_input_f32[offset] * inverse *
                        (1.0F + bf16_to_float(
                                        hc_norm_bf16[stream * kOutputHeadHidden +
                                                     hidden]));
            }
        }
        for (std::size_t row = 0u; row < kOutputHeadRank; ++row) {
            float sum = 0.0F;
            for (std::size_t column = 0u; column < kOutputHeadResidual; ++column) {
                sum = std::fma(
                        round_bf16(normalized_f32[token * kOutputHeadResidual +
                                                  column]),
                        bf16_to_float(down_bf16[row * kOutputHeadResidual +
                                                column]),
                        sum);
            }
            const float scaled = sum / static_cast<float>(kOutputHeadStreams);
            low_rank_f32[token * kOutputHeadRank + row] =
                    scaled * sigmoid_stable(scaled);
        }
        for (std::size_t row = 0u; row < kOutputHeadResidual; ++row) {
            float sum = 0.0F;
            for (std::size_t column = 0u; column < kOutputHeadRank; ++column) {
                sum = std::fma(
                        round_bf16(low_rank_f32[token * kOutputHeadRank + column]),
                        bf16_to_float(up_bf16[row * kOutputHeadRank + column]),
                        sum);
            }
            mix_logits_f32[token * kOutputHeadResidual + row] = sum;
        }
        for (std::size_t hidden = 0u; hidden < kOutputHeadHidden; ++hidden) {
            float sum = 0.0F;
            for (std::size_t stream = 0u; stream < kOutputHeadStreams; ++stream) {
                const std::size_t source = token * kOutputHeadResidual +
                                           stream * kOutputHeadHidden + hidden;
                sum += sigmoid_stable(mix_logits_f32[source]) *
                       normalized_f32[source];
            }
            mixed_hidden_f32[token * kOutputHeadHidden + hidden] =
                    sum / static_cast<float>(kOutputHeadStreams);
        }
    }
    return output_head_status::ok;
}

resident_output_head::resident_output_head() = default;
resident_output_head::~resident_output_head() = default;
resident_output_head::resident_output_head(resident_output_head &&) noexcept = default;
resident_output_head &resident_output_head::operator=(
        resident_output_head &&) noexcept = default;

output_head_status resident_output_head::load(
        const checkpoint_catalog &catalog, const output_head_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<resident_output_head> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(output_head_status::invalid_argument, error,
                    "output head destination is null");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const output_head_status validation = output_head_validate_config(config);
    if (validation != output_head_status::ok) {
        return fail(validation, error, "invalid output head configuration");
    }

    constexpr std::uint64_t embedding_bytes =
            static_cast<std::uint64_t>(kOutputHeadVocab) * kOutputHeadHidden *
            sizeof(std::uint16_t);
    constexpr std::uint64_t norm_bytes =
            static_cast<std::uint64_t>(kOutputHeadResidual) *
            sizeof(std::uint16_t);
    const tensor_span *embedding = catalog.find(kEmbeddingName);
    const tensor_span *global_norm = catalog.find(kGlobalNormName);
    const tensor_span *global_down = catalog.find(kGlobalDownName);
    const tensor_span *global_up = catalog.find(kGlobalUpName);
    const tensor_span *lm_head = catalog.find(kLmHeadName);
    if (!exact_span(embedding, tensor_dtype::bf16, 2u,
                    {kOutputHeadVocab, kOutputHeadHidden}, embedding_bytes) ||
        !exact_span(global_norm, tensor_dtype::bf16, 1u,
                    {kOutputHeadResidual}, norm_bytes) ||
        !exact_span(global_down, tensor_dtype::bf16, 2u,
                    {kOutputHeadRank, kOutputHeadResidual},
                    static_cast<std::uint64_t>(kOutputHeadRank) *
                            kOutputHeadResidual * sizeof(std::uint16_t)) ||
        !exact_span(global_up, tensor_dtype::bf16, 2u,
                    {kOutputHeadResidual, kOutputHeadRank},
                    static_cast<std::uint64_t>(kOutputHeadResidual) *
                            kOutputHeadRank * sizeof(std::uint16_t))) {
        return fail(output_head_status::tensor_contract_mismatch, error,
                    "embedding or global hyper-connection tensor contract mismatch");
    }
    if (catalog.find(kLegacyFinalNormName) != nullptr) {
        return fail(output_head_status::tensor_contract_mismatch, error,
                    "qwen4_exp must finalize through the global hyper-connection "
                    "mixer, not model.language_model.norm.weight");
    }
    if (lm_head != nullptr &&
        !exact_span(lm_head, tensor_dtype::bf16, 2u,
                    {kOutputHeadVocab, kOutputHeadHidden}, embedding_bytes)) {
        return fail(output_head_status::tensor_contract_mismatch, error,
                    "lm_head.weight contract mismatch");
    }

    int device = -1;
    if (!current_blackwell_device(&device)) {
        return fail(output_head_status::unsupported_device, error,
                    "qwen4_exp output head requires an active SM120 device");
    }

    try {
        auto result = std::make_unique<resident_output_head>();
        result->impl_ = std::make_unique<impl>();
        result->impl_->config = config;
        result->impl_->device = device;
        output_head_status status = output_head_get_workspace_requirements(
                config, &result->impl_->requirements);
        if (status != output_head_status::ok) return status;

        status = upload_tensor(catalog, *embedding, config.upload_chunk_bytes,
                               initialization_stream,
                               &result->impl_->embedding, error);
        if (status != output_head_status::ok) return status;
        status = upload_tensor(catalog, *global_norm, config.upload_chunk_bytes,
                               initialization_stream,
                               &result->impl_->global_norm, error);
        if (status != output_head_status::ok) return status;

        auto load_linear = [&](const std::string &name, std::uint64_t inputs,
                               std::uint64_t outputs,
                               std::unique_ptr<resident_bf16_linear> *linear) {
            bf16_linear_config linear_config{};
            linear_config.input_features = inputs;
            linear_config.output_features = outputs;
            linear_config.max_batch = config.max_tokens;
            linear_config.upload_chunk_bytes = config.upload_chunk_bytes;
            linear_config.blas_workspace_bytes = config.blas_workspace_bytes;
            std::string linear_error;
            const bf16_linear_status linear_status = resident_bf16_linear::load(
                    catalog, name, linear_config, initialization_stream, linear,
                    &linear_error);
            const output_head_status mapped = map_linear_status(linear_status);
            if (mapped != output_head_status::ok && error != nullptr) {
                *error = linear_error.empty() ? "linear load failed for " + name
                                              : linear_error;
            }
            return mapped;
        };

        status = load_linear(kGlobalDownName, kOutputHeadResidual,
                             kOutputHeadRank, &result->impl_->global_down);
        if (status == output_head_status::ok) {
            status = load_linear(kGlobalUpName, kOutputHeadRank,
                                 kOutputHeadResidual,
                                 &result->impl_->global_up);
        }
        const char *head_name = lm_head != nullptr ? kLmHeadName : kEmbeddingName;
        if (status == output_head_status::ok) {
            status = load_linear(head_name, kOutputHeadHidden,
                                 kOutputHeadVocab, &result->impl_->lm_head);
        }
        if (status != output_head_status::ok) return status;

        result->impl_->tying =
                lm_head != nullptr ? output_head_tying::untied_checkpoint_head
                                   : output_head_tying::tied_to_embedding;
        result->impl_->resident_bytes =
                embedding_bytes + norm_bytes +
                result->impl_->global_down->weight_bytes() +
                result->impl_->global_up->weight_bytes() +
                result->impl_->lm_head->weight_bytes();
        result->impl_->ready = true;
        *out = std::move(result);
        return output_head_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(output_head_status::allocation_failure, error,
                    "host allocation failed while loading output head");
    } catch (const std::exception &exception) {
        return fail(output_head_status::allocation_failure, error,
                    std::string("output head load exception: ") + exception.what());
    } catch (...) {
        return fail(output_head_status::allocation_failure, error,
                    "unknown output head load exception");
    }
}

output_head_status resident_output_head::embedding_repeat(
        const std::uint32_t *token_ids, std::size_t tokens,
        const output_head_scratch &scratch, float *hyper_input_f32,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || token_ids == nullptr ||
        hyper_input_f32 == nullptr || scratch.device_status == nullptr ||
        tokens == 0u || tokens > impl_->config.max_tokens) {
        return output_head_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return output_head_status::unsupported_device;
    }
    std::size_t id_bytes = 0u;
    std::size_t hyper_bytes = 0u;
    if (multiply_overflow(tokens, sizeof(std::uint32_t), &id_bytes) ||
        multiply_overflow(tokens, kOutputHeadResidual * sizeof(float),
                          &hyper_bytes)) {
        return output_head_status::size_overflow;
    }
    const std::array<device_range, 3> ranges{{
            {token_ids, id_bytes, alignof(std::uint32_t)},
            {hyper_input_f32, hyper_bytes, alignof(float)},
            {scratch.device_status, sizeof(std::uint32_t),
             alignof(std::uint32_t)},
    }};
    if (scratch.device_status_bytes < sizeof(std::uint32_t) ||
        !valid_disjoint_ranges(ranges, impl_->device)) {
        return output_head_status::invalid_device_pointer;
    }
    if (cudaMemsetAsync(scratch.device_status, 0, sizeof(std::uint32_t),
                        stream) != cudaSuccess) {
        return output_head_status::cuda_error;
    }
    const std::size_t elements = tokens * kOutputHeadResidual;
    const std::size_t blocks = (elements + kThreads - 1u) / kThreads;
    embedding_repeat_kernel<<<static_cast<unsigned>(blocks), kThreads, 0,
                              stream>>>(token_ids, tokens, impl_->embedding,
                                       hyper_input_f32,
                                       scratch.device_status);
    return cudaPeekAtLastError() == cudaSuccess ? output_head_status::ok
                                                : output_head_status::cuda_error;
}

output_head_status resident_output_head::global_mix(
        const float *hyper_input_f32, std::size_t tokens,
        const output_head_scratch &scratch, float *mixed_hidden_f32,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || hyper_input_f32 == nullptr ||
        mixed_hidden_f32 == nullptr || scratch.normalized == nullptr ||
        scratch.low_rank == nullptr || scratch.mix_logits == nullptr ||
        scratch.linear_input_bf16 == nullptr ||
        scratch.blas_workspace == nullptr || tokens == 0u ||
        tokens > impl_->config.max_tokens) {
        return output_head_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return output_head_status::unsupported_device;
    }
    std::size_t residual_bytes = 0u;
    std::size_t rank_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    std::size_t linear_input_bytes = 0u;
    if (multiply_overflow(tokens, kOutputHeadResidual * sizeof(float),
                          &residual_bytes) ||
        multiply_overflow(tokens, kOutputHeadRank * sizeof(float),
                          &rank_bytes) ||
        multiply_overflow(tokens, kOutputHeadHidden * sizeof(float),
                          &hidden_bytes) ||
        multiply_overflow(tokens,
                          kOutputHeadResidual * sizeof(std::uint16_t),
                          &linear_input_bytes)) {
        return output_head_status::size_overflow;
    }
    const std::array<device_range, 7> ranges{{
            {hyper_input_f32, residual_bytes, alignof(float)},
            {scratch.normalized, residual_bytes, alignof(float)},
            {scratch.low_rank, rank_bytes, alignof(float)},
            {scratch.mix_logits, residual_bytes, alignof(float)},
            {mixed_hidden_f32, hidden_bytes, alignof(float)},
            {scratch.linear_input_bf16, linear_input_bytes,
             alignof(std::uint16_t)},
            {scratch.blas_workspace, impl_->config.blas_workspace_bytes,
             kWorkspaceAlignment},
    }};
    if (scratch.linear_input_bf16_bytes < linear_input_bytes ||
        scratch.blas_workspace_bytes < impl_->config.blas_workspace_bytes ||
        !valid_disjoint_ranges(ranges, impl_->device)) {
        return output_head_status::invalid_device_pointer;
    }

    grouped_rmsnorm_kernel<<<
            static_cast<unsigned>(tokens * kOutputHeadStreams), kThreads, 0,
            stream>>>(hyper_input_f32, impl_->global_norm,
                      tokens * kOutputHeadStreams, scratch.normalized);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return output_head_status::cuda_error;
    }
    const bf16_linear_scratch linear_scratch{
            scratch.linear_input_bf16, scratch.linear_input_bf16_bytes,
            scratch.blas_workspace, scratch.blas_workspace_bytes};
    bf16_linear_status linear = impl_->global_down->forward(
            scratch.normalized, tokens, linear_scratch, scratch.low_rank,
            stream);
    if (linear != bf16_linear_status::ok) return map_linear_status(linear);
    const std::size_t rank_elements = tokens * kOutputHeadRank;
    const std::size_t rank_blocks =
            (rank_elements + kThreads - 1u) / kThreads;
    scaled_silu_kernel<<<static_cast<unsigned>(rank_blocks), kThreads, 0,
                         stream>>>(scratch.low_rank, rank_elements);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return output_head_status::cuda_error;
    }
    linear = impl_->global_up->forward(scratch.low_rank, tokens,
                                       linear_scratch, scratch.mix_logits,
                                       stream);
    if (linear != bf16_linear_status::ok) return map_linear_status(linear);
    const std::size_t hidden_elements = tokens * kOutputHeadHidden;
    const std::size_t hidden_blocks =
            (hidden_elements + kThreads - 1u) / kThreads;
    global_combine_kernel<<<static_cast<unsigned>(hidden_blocks), kThreads, 0,
                            stream>>>(scratch.normalized, scratch.mix_logits,
                                     tokens, mixed_hidden_f32);
    return cudaPeekAtLastError() == cudaSuccess ? output_head_status::ok
                                                : output_head_status::cuda_error;
}

output_head_status resident_output_head::logits(
        const float *mixed_hidden_f32, std::size_t tokens,
        const output_head_scratch &scratch, float *logits_f32,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || mixed_hidden_f32 == nullptr ||
        logits_f32 == nullptr || scratch.linear_input_bf16 == nullptr ||
        scratch.blas_workspace == nullptr || tokens == 0u ||
        tokens > impl_->config.max_tokens) {
        return output_head_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return output_head_status::unsupported_device;
    }
    std::size_t hidden_bytes = 0u;
    std::size_t input_bf16_bytes = 0u;
    std::size_t logits_bytes = 0u;
    if (multiply_overflow(tokens, kOutputHeadHidden * sizeof(float),
                          &hidden_bytes) ||
        multiply_overflow(tokens,
                          kOutputHeadHidden * sizeof(std::uint16_t),
                          &input_bf16_bytes) ||
        multiply_overflow(tokens, kOutputHeadVocab * sizeof(float),
                          &logits_bytes)) {
        return output_head_status::size_overflow;
    }
    const std::array<device_range, 4> ranges{{
            {mixed_hidden_f32, hidden_bytes, alignof(float)},
            {logits_f32, logits_bytes, alignof(float)},
            {scratch.linear_input_bf16, input_bf16_bytes,
             alignof(std::uint16_t)},
            {scratch.blas_workspace, impl_->config.blas_workspace_bytes,
             kWorkspaceAlignment},
    }};
    if (scratch.linear_input_bf16_bytes < input_bf16_bytes ||
        scratch.blas_workspace_bytes < impl_->config.blas_workspace_bytes ||
        !valid_disjoint_ranges(ranges, impl_->device)) {
        return output_head_status::invalid_device_pointer;
    }
    const bf16_linear_scratch linear_scratch{
            scratch.linear_input_bf16, scratch.linear_input_bf16_bytes,
            scratch.blas_workspace, scratch.blas_workspace_bytes};
    return map_linear_status(impl_->lm_head->forward(
            mixed_hidden_f32, tokens, linear_scratch, logits_f32, stream));
}

output_head_status resident_output_head::embedding_global_mix_logits(
        const std::uint32_t *token_ids, std::size_t tokens,
        const output_head_scratch &scratch, cudaStream_t stream) noexcept {
    if (scratch.hyper_input == nullptr || scratch.mixed_hidden == nullptr ||
        scratch.logits == nullptr) {
        return output_head_status::invalid_argument;
    }
    output_head_status status = embedding_repeat(
            token_ids, tokens, scratch, scratch.hyper_input, stream);
    if (status == output_head_status::ok) {
        status = global_mix(scratch.hyper_input, tokens, scratch,
                            scratch.mixed_hidden, stream);
    }
    if (status == output_head_status::ok) {
        status = logits(scratch.mixed_hidden, tokens, scratch, scratch.logits,
                        stream);
    }
    return status;
}

output_head_status resident_output_head::argmax(
        const float *logits_f32, std::size_t tokens,
        std::uint32_t *token_ids, float *selected_logits,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || logits_f32 == nullptr ||
        token_ids == nullptr || selected_logits == nullptr || tokens == 0u ||
        tokens > impl_->config.max_tokens) {
        return output_head_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return output_head_status::unsupported_device;
    }
    std::size_t logits_bytes = 0u;
    std::size_t id_bytes = 0u;
    std::size_t value_bytes = 0u;
    if (multiply_overflow(tokens, kOutputHeadVocab * sizeof(float),
                          &logits_bytes) ||
        multiply_overflow(tokens, sizeof(std::uint32_t), &id_bytes) ||
        multiply_overflow(tokens, sizeof(float), &value_bytes)) {
        return output_head_status::size_overflow;
    }
    const std::array<device_range, 3> ranges{{
            {logits_f32, logits_bytes, alignof(float)},
            {token_ids, id_bytes, alignof(std::uint32_t)},
            {selected_logits, value_bytes, alignof(float)},
    }};
    if (!valid_disjoint_ranges(ranges, impl_->device)) {
        return output_head_status::invalid_device_pointer;
    }
    argmax_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
            logits_f32, tokens, token_ids, selected_logits);
    return cudaPeekAtLastError() == cudaSuccess ? output_head_status::ok
                                                : output_head_status::cuda_error;
}

output_head_status resident_output_head::sample(
        const float *logits_f32, std::size_t tokens,
        const output_head_sampling_config &sampling,
        std::uint32_t *token_ids, float *selected_logits,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || logits_f32 == nullptr ||
        token_ids == nullptr || selected_logits == nullptr || tokens == 0u ||
        tokens > impl_->config.max_tokens ||
        !std::isfinite(sampling.temperature) ||
        sampling.temperature <= 0.0F || sampling.temperature > 100.0F) {
        return output_head_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != impl_->device) {
        return output_head_status::unsupported_device;
    }
    std::size_t logits_bytes = 0u;
    std::size_t id_bytes = 0u;
    std::size_t value_bytes = 0u;
    if (multiply_overflow(tokens, kOutputHeadVocab * sizeof(float),
                          &logits_bytes) ||
        multiply_overflow(tokens, sizeof(std::uint32_t), &id_bytes) ||
        multiply_overflow(tokens, sizeof(float), &value_bytes)) {
        return output_head_status::size_overflow;
    }
    const std::array<device_range, 3> ranges{{
            {logits_f32, logits_bytes, alignof(float)},
            {token_ids, id_bytes, alignof(std::uint32_t)},
            {selected_logits, value_bytes, alignof(float)},
    }};
    if (!valid_disjoint_ranges(ranges, impl_->device)) {
        return output_head_status::invalid_device_pointer;
    }
    sample_kernel<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
            logits_f32, tokens, sampling.temperature, sampling.seed,
            sampling.sequence, token_ids, selected_logits);
    return cudaPeekAtLastError() == cudaSuccess ? output_head_status::ok
                                                : output_head_status::cuda_error;
}

output_head_status resident_output_head::collect_device_status(
        const output_head_scratch &scratch, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || scratch.device_status == nullptr ||
        scratch.device_status_bytes < sizeof(std::uint32_t) ||
        !device_range_valid(scratch.device_status, sizeof(std::uint32_t),
                            impl_->device, alignof(std::uint32_t))) {
        return output_head_status::invalid_argument;
    }
    std::uint32_t host_status = 0u;
    if (cudaMemcpyAsync(&host_status, scratch.device_status,
                        sizeof(host_status), cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return output_head_status::cuda_error;
    }
    return host_status == 0u ? output_head_status::ok
                             : output_head_status::device_rejected_input;
}

const output_head_config &resident_output_head::config() const noexcept {
    static const output_head_config empty{};
    return impl_ ? impl_->config : empty;
}

output_head_workspace_requirements
resident_output_head::workspace_requirements() const noexcept {
    return impl_ ? impl_->requirements : output_head_workspace_requirements{};
}

output_head_tying resident_output_head::tying() const noexcept {
    return impl_ ? impl_->tying : output_head_tying::tied_to_embedding;
}

output_head_finalization resident_output_head::finalization() const noexcept {
    return impl_ ? impl_->finalization
                 : output_head_finalization::global_hyper_connection_mixer;
}

std::uint64_t resident_output_head::resident_weight_bytes() const noexcept {
    return impl_ ? impl_->resident_bytes : 0u;
}

int resident_output_head::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool resident_output_head::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
