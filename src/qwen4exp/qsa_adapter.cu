#include "axiom/qwen4exp/qsa_adapter.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kMinimumUploadChunk = 4u * 1024u;
constexpr std::size_t kMaximumUploadChunk = 64u * 1024u * 1024u;
constexpr std::size_t kMaximumBlasWorkspace = 64u * 1024u * 1024u;
constexpr std::size_t kWorkspaceAlignment = 256u;
constexpr std::size_t kRotaryDim = attention::kCheckpointRotaryDim;
constexpr std::size_t kSelectedStride = qsa::checkpoint_selected_capacity;

bool checked_mul(std::size_t left,
                 std::size_t right,
                 std::size_t* result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left,
                 std::size_t right,
                 std::size_t* result) noexcept {
    if (result == nullptr ||
        right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

QsaAdapterStatus fail(QsaAdapterStatus status,
                      std::string* error,
                      const std::string& message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return status;
}

bool current_sm120_device(int* device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, *device) == cudaSuccess &&
           properties.major == 12 && properties.minor == 0;
}

struct DeviceRange {
    const void* pointer = nullptr;
    std::size_t bytes = 0u;
    std::size_t alignment = 1u;
};

bool device_range_valid(const DeviceRange& range, int expected_device) noexcept {
    if (range.pointer == nullptr || range.bytes == 0u || range.alignment == 0u) {
        return false;
    }
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(range.pointer);
    if ((address % range.alignment) != 0u) return false;

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status =
        cudaPointerGetAttributes(&attributes, range.pointer);
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
    if (cuMemGetAddressRange(
            &allocation_base,
            &allocation_bytes,
            static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS) {
        return false;
    }
    const std::uintptr_t base = static_cast<std::uintptr_t>(allocation_base);
    if (address < base || range.bytes > allocation_bytes) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes - range.bytes;
}

bool ranges_overlap(const DeviceRange& left, const DeviceRange& right) noexcept {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left.pointer);
    const std::uintptr_t right_begin =
        reinterpret_cast<std::uintptr_t>(right.pointer);
    if (left.bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
        right.bytes > std::numeric_limits<std::uintptr_t>::max() - right_begin) {
        return true;
    }
    const std::uintptr_t left_end = left_begin + left.bytes;
    const std::uintptr_t right_end = right_begin + right.bytes;
    return left_begin < right_end && right_begin < left_end;
}

QsaAdapterStatus map_linear_status(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return QsaAdapterStatus::kOk;
        case bf16_linear_status::invalid_argument:
            return QsaAdapterStatus::kInvalidArgument;
        case bf16_linear_status::unsupported_config:
            return QsaAdapterStatus::kUnsupportedConfig;
        case bf16_linear_status::tensor_not_found:
            return QsaAdapterStatus::kTensorNotFound;
        case bf16_linear_status::dtype_mismatch:
            return QsaAdapterStatus::kDtypeMismatch;
        case bf16_linear_status::shape_mismatch:
            return QsaAdapterStatus::kShapeMismatch;
        case bf16_linear_status::size_overflow:
            return QsaAdapterStatus::kSizeOverflow;
        case bf16_linear_status::unsupported_device:
            return QsaAdapterStatus::kUnsupportedDevice;
        case bf16_linear_status::invalid_device_pointer:
            return QsaAdapterStatus::kInvalidDevicePointer;
        case bf16_linear_status::checkpoint_io_error:
            return QsaAdapterStatus::kCheckpointIoError;
        case bf16_linear_status::allocation_failure:
            return QsaAdapterStatus::kAllocationFailure;
        case bf16_linear_status::cuda_error:
        case bf16_linear_status::cublas_error:
            return QsaAdapterStatus::kLinearError;
    }
    return QsaAdapterStatus::kLinearError;
}

QsaAdapterStatus map_qsa_status(qsa::status status) noexcept {
    switch (status) {
        case qsa::status::ok: return QsaAdapterStatus::kOk;
        case qsa::status::null_pointer:
        case qsa::status::invalid_argument:
            return QsaAdapterStatus::kInvalidArgument;
        case qsa::status::invalid_config:
            return QsaAdapterStatus::kUnsupportedConfig;
        case qsa::status::arithmetic_overflow:
            return QsaAdapterStatus::kSizeOverflow;
        case qsa::status::capacity_exceeded:
            return QsaAdapterStatus::kInvalidDevicePointer;
        case qsa::status::non_finite:
        case qsa::status::invalid_visible_index:
            return QsaAdapterStatus::kQsaError;
        case qsa::status::cuda_failure:
            return QsaAdapterStatus::kCudaError;
    }
    return QsaAdapterStatus::kQsaError;
}

QsaAdapterStatus map_attention_status(attention::status status) noexcept {
    switch (status) {
        case attention::status::ok: return QsaAdapterStatus::kOk;
        case attention::status::null_pointer:
        case attention::status::invalid_argument:
            return QsaAdapterStatus::kInvalidArgument;
        case attention::status::invalid_config:
            return QsaAdapterStatus::kUnsupportedConfig;
        case attention::status::invalid_state:
            return QsaAdapterStatus::kStateMismatch;
        case attention::status::capacity_exceeded:
        case attention::status::insufficient_workspace:
            return QsaAdapterStatus::kInvalidDevicePointer;
        case attention::status::arithmetic_overflow:
            return QsaAdapterStatus::kSizeOverflow;
        case attention::status::invalid_selected_count:
        case attention::status::invalid_selected_index:
        case attention::status::non_finite:
            return QsaAdapterStatus::kAttentionError;
        case attention::status::cuda_failure:
            return QsaAdapterStatus::kCudaError;
    }
    return QsaAdapterStatus::kAttentionError;
}

std::string tensor_prefix(std::size_t layer_index) {
    return "model.language_model.layers." + std::to_string(layer_index) +
           ".self_attn.";
}

QsaAdapterStatus load_linear(
    const checkpoint_catalog& catalog,
    const std::string& name,
    std::size_t input_features,
    std::size_t output_features,
    const QsaAdapterConfig& adapter_config,
    cudaStream_t stream,
    std::unique_ptr<resident_bf16_linear>* output,
    std::string* error) noexcept {
    bf16_linear_config config{};
    config.input_features = input_features;
    config.output_features = output_features;
    config.max_batch = adapter_config.max_batch;
    config.upload_chunk_bytes = adapter_config.upload_chunk_bytes;
    config.blas_workspace_bytes = adapter_config.blas_workspace_bytes;
    return map_linear_status(resident_bf16_linear::load(
        catalog, name, config, stream, output, error));
}

QsaAdapterStatus load_bf16_vector(
    const checkpoint_catalog& catalog,
    const std::string& name,
    std::size_t elements,
    cudaStream_t stream,
    std::uint16_t** output,
    std::string* error) noexcept {
    if (output == nullptr || *output != nullptr || name.empty() || elements == 0u) {
        return fail(QsaAdapterStatus::kInvalidArgument, error,
                    "invalid BF16 vector load request: " + name);
    }
    const tensor_span* span = catalog.find(name);
    if (span == nullptr) {
        return fail(QsaAdapterStatus::kTensorNotFound, error,
                    "checkpoint tensor not found: " + name);
    }
    if (span->dtype != tensor_dtype::bf16) {
        return fail(QsaAdapterStatus::kDtypeMismatch, error,
                    "checkpoint tensor is not BF16: " + name);
    }
    std::size_t bytes = 0u;
    if (span->rank != 1u || span->shape[0] != elements ||
        !checked_mul(elements, sizeof(std::uint16_t), &bytes) ||
        span->bytes != bytes) {
        return fail(QsaAdapterStatus::kShapeMismatch, error,
                    "checkpoint BF16 vector shape mismatch: " + name);
    }

    std::uint16_t* device_output = nullptr;
    void* host_staging = nullptr;
    const auto cleanup = [&]() noexcept {
        if (host_staging != nullptr) (void)cudaFreeHost(host_staging);
        if (device_output != nullptr) (void)cudaFree(device_output);
    };
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), bytes) != cudaSuccess ||
        cudaHostAlloc(&host_staging, bytes, cudaHostAllocPortable) != cudaSuccess) {
        cleanup();
        return fail(QsaAdapterStatus::kAllocationFailure, error,
                    "BF16 vector staging allocation failed: " + name);
    }
    std::string read_error;
    if (!catalog.read_range(name, 0u, host_staging, bytes, &read_error)) {
        cleanup();
        return fail(QsaAdapterStatus::kCheckpointIoError, error,
                    read_error.empty()
                        ? "checkpoint vector read failed: " + name
                        : read_error);
    }
    if (cudaMemcpyAsync(device_output,
                        host_staging,
                        bytes,
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        cleanup();
        return fail(QsaAdapterStatus::kCudaError, error,
                    "BF16 vector upload failed: " + name);
    }
    const cudaError_t free_status = cudaFreeHost(host_staging);
    host_staging = nullptr;
    if (free_status != cudaSuccess) {
        cleanup();
        return fail(QsaAdapterStatus::kCudaError, error,
                    "BF16 vector staging release failed: " + name);
    }
    *output = device_output;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus validate_cache(const QsaLayerCache* cache) noexcept {
    if (cache == nullptr || !cache->initialized ||
        cache->index_key_bf16 == nullptr || cache->capacity_tokens == 0u ||
        cache->capacity_tokens > kQsaAdapterMaxContext ||
        cache->committed_tokens > cache->capacity_tokens ||
        cache->staged_tokens > cache->capacity_tokens - cache->committed_tokens ||
        cache->transaction_open != cache->attention_cache.transaction_open ||
        cache->committed_tokens != cache->attention_cache.committed_tokens ||
        cache->staged_tokens != cache->attention_cache.staged_tokens ||
        cache->capacity_tokens != cache->attention_cache.capacity_tokens) {
        return QsaAdapterStatus::kStateMismatch;
    }
    attention::cache_lengths lengths{};
    const attention::status status =
        attention::cache_get_lengths(&cache->attention_cache, &lengths);
    if (status != attention::status::ok ||
        lengths.committed != cache->committed_tokens ||
        lengths.staged != cache->staged_tokens ||
        lengths.capacity != cache->capacity_tokens) {
        return QsaAdapterStatus::kStateMismatch;
    }
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus abort_transaction(QsaLayerCache* cache) noexcept {
    if (cache == nullptr || !cache->initialized || !cache->transaction_open) {
        return QsaAdapterStatus::kStateMismatch;
    }
    const attention::status status = attention::rollback(&cache->attention_cache);
    if (status != attention::status::ok) return QsaAdapterStatus::kStateMismatch;
    cache->staged_tokens = 0u;
    cache->transaction_open = false;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus fail_transaction(QsaLayerCache* cache,
                                  QsaAdapterStatus original) noexcept {
    return abort_transaction(cache) == QsaAdapterStatus::kOk
        ? original
        : QsaAdapterStatus::kStateMismatch;
}

__device__ void report_qsa_error(qsa::status* output, qsa::status value) {
    if (output != nullptr) {
        atomicCAS(reinterpret_cast<unsigned int*>(output),
                  static_cast<unsigned int>(qsa::status::ok),
                  static_cast<unsigned int>(value));
    }
}

__device__ void report_attention_error(attention::status* output,
                                       attention::status value) {
    if (output != nullptr) {
        atomicCAS(reinterpret_cast<unsigned int*>(output),
                  static_cast<unsigned int>(attention::status::ok),
                  static_cast<unsigned int>(value));
    }
}

__global__ void projection_to_bf16_kernel(
    const float* input,
    std::uint16_t* output,
    std::size_t elements,
    attention::status* device_status) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = input[index];
    if (!isfinite(value)) {
        report_attention_error(device_status, attention::status::non_finite);
        value = 0.0F;
    }
    reinterpret_cast<__nv_bfloat16*>(output)[index] =
        __float2bfloat16_rn(value);
}

__global__ void split_index_projection_kernel(
    const float* projection,
    std::size_t token_count,
    std::size_t cache_start,
    std::uint16_t* query_bf16,
    std::uint16_t* index_key_cache_bf16,
    qsa::status* device_status) {
    const std::size_t token = static_cast<std::size_t>(blockIdx.x);
    const std::size_t lane = threadIdx.x;
    if (token >= token_count || lane >= kQsaAdapterIndexerProjectionSize) return;
    float value = projection[token * kQsaAdapterIndexerProjectionSize + lane];
    if (!isfinite(value)) {
        report_qsa_error(device_status, qsa::status::non_finite);
        value = 0.0F;
    }
    const __nv_bfloat16 converted = __float2bfloat16_rn(value);
    if (lane < kQsaAdapterIndexerQuerySize) {
        reinterpret_cast<__nv_bfloat16*>(query_bf16)
            [token * kQsaAdapterIndexerQuerySize + lane] = converted;
    } else {
        reinterpret_cast<__nv_bfloat16*>(index_key_cache_bf16)
            [(cache_start + token) * kQsaAdapterIndexerKeySize +
             (lane - kQsaAdapterIndexerQuerySize)] = converted;
    }
}

__global__ void fill_visible_indices_kernel(std::int32_t* output,
                                            std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = static_cast<std::int32_t>(index);
}

__global__ void store_selected_count_kernel(std::uint32_t* output,
                                            std::size_t token,
                                            std::uint32_t count) {
    if (threadIdx.x == 0u && blockIdx.x == 0u) output[token] = count;
}

__global__ void attention_bf16_to_f32_kernel(
    const std::uint16_t* input,
    float* output,
    std::size_t elements,
    attention::status* device_status) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = __bfloat162float(
        reinterpret_cast<const __nv_bfloat16*>(input)[index]);
    if (!isfinite(value)) {
        report_attention_error(device_status, attention::status::non_finite);
        value = 0.0F;
    }
    output[index] = value;
}

QsaAdapterStatus launch_projection_conversion(
    const float* input,
    std::uint16_t* output,
    std::size_t elements,
    attention::status* device_status,
    cudaStream_t stream) noexcept {
    constexpr unsigned int threads = 256u;
    if (elements == 0u ||
        elements > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) *
                       threads) {
        return QsaAdapterStatus::kSizeOverflow;
    }
    const unsigned int blocks = static_cast<unsigned int>(
        (elements + threads - 1u) / threads);
    projection_to_bf16_kernel<<<blocks, threads, 0u, stream>>>(
        input, output, elements, device_status);
    return cudaPeekAtLastError() == cudaSuccess
        ? QsaAdapterStatus::kOk
        : QsaAdapterStatus::kCudaError;
}

}  // namespace

struct QsaLayerAdapter::impl {
    QsaAdapterConfig config{};
    std::unique_ptr<resident_bf16_linear> q_projection;
    std::unique_ptr<resident_bf16_linear> k_projection;
    std::unique_ptr<resident_bf16_linear> v_projection;
    std::unique_ptr<resident_bf16_linear> o_projection;
    std::unique_ptr<resident_bf16_linear> index_projection;
    std::uint16_t* q_norm_weight = nullptr;
    std::uint16_t* k_norm_weight = nullptr;
    std::uint16_t* index_q_norm_weight = nullptr;
    std::uint16_t* index_k_norm_weight = nullptr;
    int device = -1;
    bool ready = false;

    ~impl() {
        if (q_norm_weight != nullptr) (void)cudaFree(q_norm_weight);
        if (k_norm_weight != nullptr) (void)cudaFree(k_norm_weight);
        if (index_q_norm_weight != nullptr) (void)cudaFree(index_q_norm_weight);
        if (index_k_norm_weight != nullptr) (void)cudaFree(index_k_norm_weight);
    }
};

const char* qsa_adapter_status_string(QsaAdapterStatus status) noexcept {
    switch (status) {
        case QsaAdapterStatus::kOk: return "ok";
        case QsaAdapterStatus::kInvalidArgument: return "invalid_argument";
        case QsaAdapterStatus::kUnsupportedLayer: return "unsupported_layer";
        case QsaAdapterStatus::kUnsupportedConfig: return "unsupported_config";
        case QsaAdapterStatus::kTensorNotFound: return "tensor_not_found";
        case QsaAdapterStatus::kDtypeMismatch: return "dtype_mismatch";
        case QsaAdapterStatus::kShapeMismatch: return "shape_mismatch";
        case QsaAdapterStatus::kSizeOverflow: return "size_overflow";
        case QsaAdapterStatus::kCheckpointIoError: return "checkpoint_io_error";
        case QsaAdapterStatus::kAllocationFailure: return "allocation_failure";
        case QsaAdapterStatus::kUnsupportedDevice: return "unsupported_device";
        case QsaAdapterStatus::kInvalidDevicePointer:
            return "invalid_device_pointer";
        case QsaAdapterStatus::kStateMismatch: return "state_mismatch";
        case QsaAdapterStatus::kLinearError: return "linear_error";
        case QsaAdapterStatus::kQsaError: return "qsa_error";
        case QsaAdapterStatus::kAttentionError: return "attention_error";
        case QsaAdapterStatus::kDeviceRejected: return "device_rejected";
        case QsaAdapterStatus::kCudaError: return "cuda_error";
    }
    return "unknown";
}

QsaAdapterStatus qsa_adapter_validate_config(
    const QsaAdapterConfig& config) noexcept {
    if (config.layer_index >= kQsaAdapterLayerCount ||
        config.max_batch == 0u || config.max_context == 0u) {
        return QsaAdapterStatus::kInvalidArgument;
    }
    if ((config.layer_index % 4u) != 3u) {
        return QsaAdapterStatus::kUnsupportedLayer;
    }
    if (config.max_batch > kQsaAdapterMaxBatch ||
        config.max_context < qsa::checkpoint_config.compress_ratio ||
        config.max_context > kQsaAdapterMaxContext ||
        config.upload_chunk_bytes < kMinimumUploadChunk ||
        config.upload_chunk_bytes > kMaximumUploadChunk ||
        (config.upload_chunk_bytes % sizeof(std::uint16_t)) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > kMaximumBlasWorkspace ||
        (config.blas_workspace_bytes % kWorkspaceAlignment) != 0u ||
        qsa::validate_checkpoint_config(qsa::checkpoint_config) != qsa::status::ok ||
        attention::validate_checkpoint_config(attention::checkpoint_config) !=
            attention::status::ok) {
        return QsaAdapterStatus::kUnsupportedConfig;
    }
    const std::array<std::array<std::size_t, 2>, 3> shapes{{
        {kQsaAdapterHiddenSize, kQsaAdapterQueryProjectionSize},
        {kQsaAdapterHiddenSize, kQsaAdapterKvProjectionSize},
        {kQsaAdapterAttentionOutputSize, kQsaAdapterHiddenSize},
    }};
    for (const auto& shape : shapes) {
        bf16_linear_config linear{};
        linear.input_features = shape[0];
        linear.output_features = shape[1];
        linear.max_batch = config.max_batch;
        linear.upload_chunk_bytes = config.upload_chunk_bytes;
        linear.blas_workspace_bytes = config.blas_workspace_bytes;
        if (bf16_linear_validate_config(linear) != bf16_linear_status::ok) {
            return QsaAdapterStatus::kUnsupportedConfig;
        }
    }
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus qsa_adapter_workspace_requirements(
    const QsaAdapterConfig& config,
    QsaAdapterWorkspaceRequirements* requirements) noexcept {
    if (requirements == nullptr) return QsaAdapterStatus::kInvalidArgument;
    *requirements = QsaAdapterWorkspaceRequirements{};
    const QsaAdapterStatus validation = qsa_adapter_validate_config(config);
    if (validation != QsaAdapterStatus::kOk) return validation;

    const auto batch_bytes = [&](std::size_t width,
                                 std::size_t element_bytes,
                                 std::size_t* output) noexcept {
        std::size_t elements = 0u;
        return checked_mul(config.max_batch, width, &elements) &&
               checked_mul(elements, element_bytes, output);
    };
    std::size_t max_blocks = config.max_context / qsa::checkpoint_config.compress_ratio;
    if (!batch_bytes(kQsaAdapterAttentionOutputSize,
                     sizeof(std::uint16_t),
                     &requirements->linear_input_bf16_bytes) ||
        !batch_bytes(kQsaAdapterQueryProjectionSize,
                     sizeof(float),
                     &requirements->q_projection_f32_bytes) ||
        !batch_bytes(kQsaAdapterKvProjectionSize,
                     sizeof(float),
                     &requirements->k_projection_f32_bytes) ||
        !batch_bytes(kQsaAdapterKvProjectionSize,
                     sizeof(float),
                     &requirements->v_projection_f32_bytes) ||
        !batch_bytes(kQsaAdapterIndexerProjectionSize,
                     sizeof(float),
                     &requirements->index_projection_f32_bytes) ||
        !batch_bytes(kQsaAdapterQueryProjectionSize,
                     sizeof(std::uint16_t),
                     &requirements->q_projection_bf16_bytes) ||
        !batch_bytes(kQsaAdapterKvProjectionSize,
                     sizeof(std::uint16_t),
                     &requirements->k_projection_bf16_bytes) ||
        !batch_bytes(kQsaAdapterKvProjectionSize,
                     sizeof(std::uint16_t),
                     &requirements->v_projection_bf16_bytes) ||
        !batch_bytes(kQsaAdapterIndexerQuerySize,
                     sizeof(std::uint16_t),
                     &requirements->index_query_bf16_bytes) ||
        !batch_bytes(kQsaAdapterIndexerQuerySize,
                     sizeof(float),
                     &requirements->index_query_prepared_f32_bytes) ||
        !checked_mul(config.max_context,
                     sizeof(std::int32_t),
                     &requirements->visible_indices_bytes) ||
        !checked_mul(max_blocks,
                     kQsaAdapterIndexerKeySize,
                     &requirements->pooled_index_keys_f32_bytes) ||
        !checked_mul(requirements->pooled_index_keys_f32_bytes,
                     sizeof(float),
                     &requirements->pooled_index_keys_f32_bytes) ||
        !checked_mul(max_blocks,
                     sizeof(float),
                     &requirements->index_scores_f32_bytes) ||
        qsa::cuda_select_workspace_bytes(
            qsa::checkpoint_config,
            max_blocks,
            &requirements->selection_workspace_bytes) != qsa::status::ok ||
        !batch_bytes(kSelectedStride,
                     sizeof(std::int32_t),
                     &requirements->selected_indices_bytes) ||
        !batch_bytes(1u,
                     sizeof(std::uint32_t),
                     &requirements->selected_counts_bytes) ||
        attention::workspace_bytes(attention::checkpoint_config,
                                   config.max_batch,
                                   &requirements->attention_workspace_bytes) !=
            attention::status::ok ||
        !batch_bytes(kQsaAdapterAttentionOutputSize,
                     sizeof(std::uint16_t),
                     &requirements->attention_output_bf16_bytes) ||
        !batch_bytes(kQsaAdapterAttentionOutputSize,
                     sizeof(float),
                     &requirements->attention_output_f32_bytes)) {
        *requirements = QsaAdapterWorkspaceRequirements{};
        return QsaAdapterStatus::kSizeOverflow;
    }
    requirements->blas_workspace_bytes = config.blas_workspace_bytes;

    const std::array<std::size_t, 20> parts{{
        requirements->linear_input_bf16_bytes,
        requirements->blas_workspace_bytes,
        requirements->q_projection_f32_bytes,
        requirements->k_projection_f32_bytes,
        requirements->v_projection_f32_bytes,
        requirements->index_projection_f32_bytes,
        requirements->q_projection_bf16_bytes,
        requirements->k_projection_bf16_bytes,
        requirements->v_projection_bf16_bytes,
        requirements->index_query_bf16_bytes,
        requirements->index_query_prepared_f32_bytes,
        requirements->visible_indices_bytes,
        requirements->pooled_index_keys_f32_bytes,
        requirements->index_scores_f32_bytes,
        requirements->selection_workspace_bytes,
        requirements->selected_indices_bytes,
        requirements->selected_counts_bytes,
        requirements->attention_workspace_bytes,
        requirements->attention_output_bf16_bytes,
        requirements->attention_output_f32_bytes,
    }};
    for (const std::size_t part : parts) {
        if (!checked_add(requirements->total_device_bytes,
                         part,
                         &requirements->total_device_bytes)) {
            *requirements = QsaAdapterWorkspaceRequirements{};
            return QsaAdapterStatus::kSizeOverflow;
        }
    }
    if (!checked_add(requirements->total_device_bytes,
                     requirements->qsa_device_status_bytes,
                     &requirements->total_device_bytes) ||
        !checked_add(requirements->total_device_bytes,
                     requirements->attention_device_status_bytes,
                     &requirements->total_device_bytes)) {
        *requirements = QsaAdapterWorkspaceRequirements{};
        return QsaAdapterStatus::kSizeOverflow;
    }
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus qsa_adapter_cache_initialize(
    QsaLayerCache* cache,
    std::uint16_t* external_key_bf16,
    std::uint16_t* external_value_bf16,
    std::uint16_t* external_index_key_bf16,
    std::size_t capacity_tokens,
    std::size_t initial_committed_tokens) noexcept {
    if (cache == nullptr || external_index_key_bf16 == nullptr) {
        return QsaAdapterStatus::kInvalidArgument;
    }
    *cache = QsaLayerCache{};
    if (capacity_tokens == 0u || capacity_tokens > kQsaAdapterMaxContext ||
        initial_committed_tokens > capacity_tokens) {
        return QsaAdapterStatus::kInvalidArgument;
    }
    const attention::status status = attention::cache_initialize(
        &cache->attention_cache,
        external_key_bf16,
        external_value_bf16,
        capacity_tokens,
        initial_committed_tokens);
    if (status != attention::status::ok) return map_attention_status(status);
    cache->index_key_bf16 = external_index_key_bf16;
    cache->capacity_tokens = capacity_tokens;
    cache->committed_tokens = initial_committed_tokens;
    cache->initialized = true;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus qsa_adapter_cache_reset(QsaLayerCache* cache) noexcept {
    const QsaAdapterStatus validation = validate_cache(cache);
    if (validation != QsaAdapterStatus::kOk) return validation;
    const attention::status status = attention::cache_reset(&cache->attention_cache);
    if (status != attention::status::ok) return map_attention_status(status);
    cache->committed_tokens = 0u;
    cache->staged_tokens = 0u;
    cache->transaction_open = false;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus qsa_adapter_cache_get_lengths(
    const QsaLayerCache* cache,
    QsaAdapterCacheLengths* lengths) noexcept {
    if (lengths == nullptr) return QsaAdapterStatus::kInvalidArgument;
    *lengths = QsaAdapterCacheLengths{};
    const QsaAdapterStatus validation = validate_cache(cache);
    if (validation != QsaAdapterStatus::kOk) return validation;
    lengths->committed = cache->committed_tokens;
    lengths->staged = cache->staged_tokens;
    lengths->visible = cache->committed_tokens + cache->staged_tokens;
    lengths->capacity = cache->capacity_tokens;
    return QsaAdapterStatus::kOk;
}

QsaLayerAdapter::QsaLayerAdapter() = default;
QsaLayerAdapter::~QsaLayerAdapter() = default;
QsaLayerAdapter::QsaLayerAdapter(QsaLayerAdapter&&) noexcept = default;
QsaLayerAdapter& QsaLayerAdapter::operator=(QsaLayerAdapter&&) noexcept = default;

QsaAdapterStatus QsaLayerAdapter::load(
    const checkpoint_catalog& catalog,
    const QsaAdapterConfig& config,
    cudaStream_t initialization_stream,
    std::unique_ptr<QsaLayerAdapter>* out,
    std::string* error) noexcept {
    if (out == nullptr) {
        return fail(QsaAdapterStatus::kInvalidArgument, error,
                    "null QSA adapter output");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const QsaAdapterStatus validation = qsa_adapter_validate_config(config);
    if (validation != QsaAdapterStatus::kOk) {
        return fail(validation, error,
                    "invalid QSA adapter configuration: " +
                        std::string(qsa_adapter_status_string(validation)));
    }
    int device = -1;
    if (!current_sm120_device(&device)) {
        return fail(QsaAdapterStatus::kUnsupportedDevice, error,
                    "QSA adapter requires an active SM120 Blackwell device");
    }

    try {
        auto result = std::make_unique<QsaLayerAdapter>();
        result->impl_ = std::make_unique<impl>();
        result->impl_->config = config;
        result->impl_->device = device;
        const std::string prefix = tensor_prefix(config.layer_index);

        using LinearMember = std::unique_ptr<resident_bf16_linear> impl::*;
        struct LinearLoad {
            const char* suffix;
            std::size_t input;
            std::size_t output;
            LinearMember member;
        };
        const std::array<LinearLoad, 5> linears{{
            {"q_proj.weight", kQsaAdapterHiddenSize,
             kQsaAdapterQueryProjectionSize, &impl::q_projection},
            {"k_proj.weight", kQsaAdapterHiddenSize,
             kQsaAdapterKvProjectionSize, &impl::k_projection},
            {"v_proj.weight", kQsaAdapterHiddenSize,
             kQsaAdapterKvProjectionSize, &impl::v_projection},
            {"o_proj.weight", kQsaAdapterAttentionOutputSize,
             kQsaAdapterHiddenSize, &impl::o_projection},
            {"indexer.index_qk_proj.weight", kQsaAdapterHiddenSize,
             kQsaAdapterIndexerProjectionSize, &impl::index_projection},
        }};
        for (const LinearLoad& linear : linears) {
            const QsaAdapterStatus status = load_linear(
                catalog,
                prefix + linear.suffix,
                linear.input,
                linear.output,
                config,
                initialization_stream,
                &(result->impl_.get()->*(linear.member)),
                error);
            if (status != QsaAdapterStatus::kOk) return status;
        }

        using VectorMember = std::uint16_t* impl::*;
        struct VectorLoad {
            const char* suffix;
            std::size_t elements;
            VectorMember member;
        };
        const std::array<VectorLoad, 4> vectors{{
            {"q_norm.weight", attention::kCheckpointHeadDim,
             &impl::q_norm_weight},
            {"k_norm.weight", attention::kCheckpointHeadDim,
             &impl::k_norm_weight},
            {"indexer.q_layernorm.weight", qsa::checkpoint_config.head_dim,
             &impl::index_q_norm_weight},
            {"indexer.k_layernorm.weight", qsa::checkpoint_config.head_dim,
             &impl::index_k_norm_weight},
        }};
        for (const VectorLoad& vector : vectors) {
            const QsaAdapterStatus status = load_bf16_vector(
                catalog,
                prefix + vector.suffix,
                vector.elements,
                initialization_stream,
                &(result->impl_.get()->*(vector.member)),
                error);
            if (status != QsaAdapterStatus::kOk) return status;
        }

        result->impl_->ready = true;
        *out = std::move(result);
        return QsaAdapterStatus::kOk;
    } catch (const std::bad_alloc&) {
        return fail(QsaAdapterStatus::kAllocationFailure, error,
                    "host allocation failed while loading QSA adapter");
    } catch (const std::exception& exception) {
        return fail(QsaAdapterStatus::kAllocationFailure, error,
                    std::string("QSA adapter initialization exception: ") +
                        exception.what());
    } catch (...) {
        return fail(QsaAdapterStatus::kAllocationFailure, error,
                    "unknown QSA adapter initialization exception");
    }
}

QsaAdapterStatus QsaLayerAdapter::begin_transaction(
    QsaLayerCache* cache,
    const QsaAdapterScratch& scratch,
    cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready) return QsaAdapterStatus::kStateMismatch;
    const QsaAdapterStatus cache_status = validate_cache(cache);
    if (cache_status != QsaAdapterStatus::kOk) return cache_status;
    if (cache->transaction_open || cache->staged_tokens != 0u ||
        scratch.qsa_device_status == nullptr ||
        scratch.attention_device_status == nullptr) {
        return QsaAdapterStatus::kStateMismatch;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess) {
        return QsaAdapterStatus::kCudaError;
    }
    if (active_device != impl_->device) return QsaAdapterStatus::kUnsupportedDevice;
    const std::array<DeviceRange, 2> status_ranges{{
        {scratch.qsa_device_status, sizeof(qsa::status), alignof(qsa::status)},
        {scratch.attention_device_status,
         sizeof(attention::status),
         alignof(attention::status)},
    }};
    for (const DeviceRange& range : status_ranges) {
        if (!device_range_valid(range, impl_->device)) {
            return QsaAdapterStatus::kInvalidDevicePointer;
        }
    }
    if (ranges_overlap(status_ranges[0], status_ranges[1])) {
        return QsaAdapterStatus::kInvalidDevicePointer;
    }
    QsaAdapterStatus status = map_qsa_status(qsa::cuda_reset_status(
        scratch.qsa_device_status, stream));
    if (status != QsaAdapterStatus::kOk) return status;
    status = map_attention_status(attention::reset_device_status(
        scratch.attention_device_status, stream));
    if (status != QsaAdapterStatus::kOk) return status;
    const attention::status begin_status =
        attention::begin_transaction(&cache->attention_cache);
    if (begin_status != attention::status::ok) {
        return map_attention_status(begin_status);
    }
    cache->transaction_open = true;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus QsaLayerAdapter::forward(
    const float* hidden_states_f32,
    std::size_t token_count,
    const float* full_cos_f32,
    const float* full_sin_f32,
    std::size_t position_count,
    QsaLayerCache* cache,
    const QsaAdapterScratch& scratch,
    float* output_f32,
    cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || hidden_states_f32 == nullptr ||
        full_cos_f32 == nullptr || full_sin_f32 == nullptr ||
        output_f32 == nullptr || token_count == 0u ||
        token_count > impl_->config.max_batch) {
        return QsaAdapterStatus::kInvalidArgument;
    }
    const QsaAdapterStatus cache_status = validate_cache(cache);
    if (cache_status != QsaAdapterStatus::kOk) return cache_status;
    if (!cache->transaction_open || cache->staged_tokens != 0u ||
        cache->capacity_tokens > impl_->config.max_context) {
        return QsaAdapterStatus::kStateMismatch;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess) {
        return fail_transaction(cache, QsaAdapterStatus::kCudaError);
    }
    if (active_device != impl_->device) {
        return fail_transaction(cache, QsaAdapterStatus::kUnsupportedDevice);
    }
    const std::size_t start = cache->committed_tokens;
    std::size_t end = 0u;
    if (!checked_add(start, token_count, &end)) {
        return fail_transaction(cache, QsaAdapterStatus::kSizeOverflow);
    }
    if (end > cache->capacity_tokens || end > impl_->config.max_context ||
        position_count < end || position_count > impl_->config.max_context) {
        return fail_transaction(cache, QsaAdapterStatus::kInvalidArgument);
    }

    QsaAdapterConfig active_config = impl_->config;
    active_config.max_batch = token_count;
    active_config.max_context = cache->capacity_tokens;
    QsaAdapterWorkspaceRequirements required{};
    QsaAdapterStatus status =
        qsa_adapter_workspace_requirements(active_config, &required);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    if (scratch.linear_input_bf16_bytes < required.linear_input_bf16_bytes ||
        scratch.blas_workspace_bytes < required.blas_workspace_bytes ||
        scratch.q_projection_f32_bytes < required.q_projection_f32_bytes ||
        scratch.k_projection_f32_bytes < required.k_projection_f32_bytes ||
        scratch.v_projection_f32_bytes < required.v_projection_f32_bytes ||
        scratch.index_projection_f32_bytes < required.index_projection_f32_bytes ||
        scratch.q_projection_bf16_bytes < required.q_projection_bf16_bytes ||
        scratch.k_projection_bf16_bytes < required.k_projection_bf16_bytes ||
        scratch.v_projection_bf16_bytes < required.v_projection_bf16_bytes ||
        scratch.index_query_bf16_bytes < required.index_query_bf16_bytes ||
        scratch.index_query_prepared_f32_bytes <
            required.index_query_prepared_f32_bytes ||
        scratch.visible_indices_bytes < required.visible_indices_bytes ||
        scratch.pooled_index_keys_f32_bytes <
            required.pooled_index_keys_f32_bytes ||
        scratch.index_scores_f32_bytes < required.index_scores_f32_bytes ||
        scratch.selection_workspace_bytes < required.selection_workspace_bytes ||
        scratch.selected_indices_bytes < required.selected_indices_bytes ||
        scratch.selected_counts_bytes < required.selected_counts_bytes ||
        scratch.attention_workspace_bytes < required.attention_workspace_bytes ||
        scratch.attention_output_bf16_bytes <
            required.attention_output_bf16_bytes ||
        scratch.attention_output_f32_bytes <
            required.attention_output_f32_bytes) {
        return fail_transaction(cache, QsaAdapterStatus::kInvalidDevicePointer);
    }

    std::size_t hidden_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t cos_bytes = 0u;
    std::size_t main_cache_bytes = 0u;
    std::size_t index_cache_bytes = 0u;
    if (!checked_mul(token_count, kQsaAdapterHiddenSize, &hidden_bytes) ||
        !checked_mul(hidden_bytes, sizeof(float), &hidden_bytes) ||
        !checked_mul(token_count, kQsaAdapterHiddenSize, &output_bytes) ||
        !checked_mul(output_bytes, sizeof(float), &output_bytes) ||
        !checked_mul(position_count, kRotaryDim, &cos_bytes) ||
        !checked_mul(cos_bytes, sizeof(float), &cos_bytes) ||
        !checked_mul(cache->capacity_tokens,
                     attention::kCheckpointKvHeads *
                         attention::kCheckpointHeadDim,
                     &main_cache_bytes) ||
        !checked_mul(main_cache_bytes, sizeof(std::uint16_t), &main_cache_bytes) ||
        !checked_mul(cache->capacity_tokens,
                     kQsaAdapterIndexerKeySize,
                     &index_cache_bytes) ||
        !checked_mul(index_cache_bytes,
                     sizeof(std::uint16_t),
                     &index_cache_bytes)) {
        return fail_transaction(cache, QsaAdapterStatus::kSizeOverflow);
    }

    const std::array<DeviceRange, 25> ranges{{
        {hidden_states_f32, hidden_bytes, alignof(float)},
        {output_f32, output_bytes, alignof(float)},
        {full_cos_f32, cos_bytes, alignof(float)},
        {full_sin_f32, cos_bytes, alignof(float)},
        {cache->attention_cache.key_bf16, main_cache_bytes, alignof(std::uint16_t)},
        {cache->attention_cache.value_bf16, main_cache_bytes, alignof(std::uint16_t)},
        {cache->index_key_bf16, index_cache_bytes, alignof(std::uint16_t)},
        {scratch.linear_input_bf16,
         required.linear_input_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.blas_workspace,
         required.blas_workspace_bytes,
         kWorkspaceAlignment},
        {scratch.q_projection, required.q_projection_f32_bytes, alignof(float)},
        {scratch.k_projection, required.k_projection_f32_bytes, alignof(float)},
        {scratch.v_projection, required.v_projection_f32_bytes, alignof(float)},
        {scratch.index_projection,
         required.index_projection_f32_bytes,
         alignof(float)},
        {scratch.q_projection_bf16,
         required.q_projection_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.k_projection_bf16,
         required.k_projection_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.v_projection_bf16,
         required.v_projection_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.index_query_bf16,
         required.index_query_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.index_query_prepared,
         required.index_query_prepared_f32_bytes,
         alignof(float)},
        {scratch.visible_indices,
         required.visible_indices_bytes,
         alignof(std::int32_t)},
        {scratch.pooled_index_keys,
         required.pooled_index_keys_f32_bytes,
         alignof(float)},
        {scratch.index_scores,
         required.index_scores_f32_bytes,
         alignof(float)},
        {scratch.selection_workspace,
         required.selection_workspace_bytes,
         1u},
        {scratch.selected_indices,
         required.selected_indices_bytes,
         alignof(std::int32_t)},
        {scratch.selected_counts,
         required.selected_counts_bytes,
         alignof(std::uint32_t)},
        {scratch.attention_workspace,
         required.attention_workspace_bytes,
         1u},
    }};
    const std::array<DeviceRange, 4> trailing_ranges{{
        {scratch.attention_output_bf16,
         required.attention_output_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.attention_output_f32,
         required.attention_output_f32_bytes,
         alignof(float)},
        {scratch.qsa_device_status, sizeof(qsa::status), alignof(qsa::status)},
        {scratch.attention_device_status,
         sizeof(attention::status),
         alignof(attention::status)},
    }};
    for (const DeviceRange& range : ranges) {
        if (!device_range_valid(range, impl_->device)) {
            return fail_transaction(cache, QsaAdapterStatus::kInvalidDevicePointer);
        }
    }
    for (const DeviceRange& range : trailing_ranges) {
        if (!device_range_valid(range, impl_->device)) {
            return fail_transaction(cache, QsaAdapterStatus::kInvalidDevicePointer);
        }
    }
    for (std::size_t left = 0u; left < ranges.size(); ++left) {
        for (std::size_t right = left + 1u; right < ranges.size(); ++right) {
            if (ranges_overlap(ranges[left], ranges[right])) {
                return fail_transaction(cache,
                                        QsaAdapterStatus::kInvalidDevicePointer);
            }
        }
        for (const DeviceRange& right : trailing_ranges) {
            if (ranges_overlap(ranges[left], right)) {
                return fail_transaction(cache,
                                        QsaAdapterStatus::kInvalidDevicePointer);
            }
        }
    }
    for (std::size_t left = 0u; left < trailing_ranges.size(); ++left) {
        for (std::size_t right = left + 1u;
             right < trailing_ranges.size();
             ++right) {
            if (ranges_overlap(trailing_ranges[left], trailing_ranges[right])) {
                return fail_transaction(cache,
                                        QsaAdapterStatus::kInvalidDevicePointer);
            }
        }
    }

    const bf16_linear_scratch linear_scratch{
        scratch.linear_input_bf16,
        scratch.linear_input_bf16_bytes,
        scratch.blas_workspace,
        scratch.blas_workspace_bytes,
    };
    const auto run_linear = [&](resident_bf16_linear* linear,
                                const float* input,
                                float* output) noexcept {
        return map_linear_status(linear->forward(
            input, token_count, linear_scratch, output, stream));
    };
    status = run_linear(
        impl_->q_projection.get(), hidden_states_f32, scratch.q_projection);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    status = launch_projection_conversion(
        scratch.q_projection,
        scratch.q_projection_bf16,
        token_count * kQsaAdapterQueryProjectionSize,
        scratch.attention_device_status,
        stream);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

    status = run_linear(
        impl_->k_projection.get(), hidden_states_f32, scratch.k_projection);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    status = launch_projection_conversion(
        scratch.k_projection,
        scratch.k_projection_bf16,
        token_count * kQsaAdapterKvProjectionSize,
        scratch.attention_device_status,
        stream);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

    status = run_linear(
        impl_->v_projection.get(), hidden_states_f32, scratch.v_projection);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    status = launch_projection_conversion(
        scratch.v_projection,
        scratch.v_projection_bf16,
        token_count * kQsaAdapterKvProjectionSize,
        scratch.attention_device_status,
        stream);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

    status = run_linear(
        impl_->index_projection.get(), hidden_states_f32, scratch.index_projection);
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    split_index_projection_kernel<<<
        static_cast<unsigned int>(token_count),
        static_cast<unsigned int>(kQsaAdapterIndexerProjectionSize),
        0u,
        stream>>>(
        scratch.index_projection,
        token_count,
        start,
        scratch.index_query_bf16,
        cache->index_key_bf16,
        scratch.qsa_device_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return fail_transaction(cache, QsaAdapterStatus::kCudaError);
    }

    attention::staged_batch staged{};
    status = map_attention_status(attention::stage_cuda(
        attention::checkpoint_config,
        &cache->attention_cache,
        scratch.q_projection_bf16,
        scratch.k_projection_bf16,
        scratch.v_projection_bf16,
        impl_->q_norm_weight,
        impl_->k_norm_weight,
        full_cos_f32 + start * kRotaryDim,
        full_sin_f32 + start * kRotaryDim,
        token_count,
        scratch.attention_workspace,
        scratch.attention_workspace_bytes,
        &staged,
        scratch.attention_device_status,
        stream));
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
    cache->staged_tokens = token_count;

    status = map_qsa_status(qsa::prepare_queries_cuda_bf16(
        qsa::checkpoint_config,
        reinterpret_cast<const __nv_bfloat16*>(scratch.index_query_bf16),
        reinterpret_cast<const __nv_bfloat16*>(impl_->index_q_norm_weight),
        full_cos_f32 + start * kRotaryDim,
        full_sin_f32 + start * kRotaryDim,
        token_count,
        scratch.index_query_prepared,
        scratch.qsa_device_status,
        stream));
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

    constexpr unsigned int visible_threads = 256u;
    for (std::size_t token = 0u; token < token_count; ++token) {
        const std::size_t visible_count = start + token + 1u;
        const unsigned int visible_blocks = static_cast<unsigned int>(
            (visible_count + visible_threads - 1u) / visible_threads);
        fill_visible_indices_kernel<<<visible_blocks, visible_threads, 0u, stream>>>(
            scratch.visible_indices, visible_count);
        if (cudaPeekAtLastError() != cudaSuccess) {
            return fail_transaction(cache, QsaAdapterStatus::kCudaError);
        }

        std::size_t block_count = 0u;
        status = map_qsa_status(qsa::complete_block_count(
            qsa::checkpoint_config, visible_count, &block_count));
        if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);
        std::size_t produced_blocks = 0u;
        status = map_qsa_status(qsa::pool_keys_cuda_bf16(
            qsa::checkpoint_config,
            reinterpret_cast<const __nv_bfloat16*>(cache->index_key_bf16),
            visible_count,
            scratch.visible_indices,
            visible_count,
            reinterpret_cast<const __nv_bfloat16*>(impl_->index_k_norm_weight),
            full_cos_f32,
            full_sin_f32,
            position_count,
            scratch.pooled_index_keys,
            active_config.max_context / qsa::checkpoint_config.compress_ratio,
            &produced_blocks,
            scratch.qsa_device_status,
            stream));
        if (status != QsaAdapterStatus::kOk || produced_blocks != block_count) {
            return fail_transaction(
                cache,
                status == QsaAdapterStatus::kOk
                    ? QsaAdapterStatus::kStateMismatch
                    : status);
        }
        status = map_qsa_status(qsa::score_blocks_cuda_f32(
            qsa::checkpoint_config,
            scratch.index_query_prepared +
                token * kQsaAdapterIndexerQuerySize,
            scratch.pooled_index_keys,
            block_count,
            scratch.index_scores,
            scratch.qsa_device_status,
            stream));
        if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

        std::size_t selected_count = 0u;
        status = map_qsa_status(qsa::select_tokens_cuda_f32(
            qsa::checkpoint_config,
            scratch.index_scores,
            block_count,
            scratch.visible_indices,
            visible_count,
            visible_count,
            scratch.selected_indices + token * kSelectedStride,
            kSelectedStride,
            scratch.selection_workspace,
            scratch.selection_workspace_bytes,
            &selected_count,
            scratch.qsa_device_status,
            stream));
        if (status != QsaAdapterStatus::kOk ||
            selected_count == 0u || selected_count > kSelectedStride ||
            selected_count > std::numeric_limits<std::uint32_t>::max()) {
            return fail_transaction(
                cache,
                status == QsaAdapterStatus::kOk
                    ? QsaAdapterStatus::kQsaError
                    : status);
        }
        store_selected_count_kernel<<<1u, 1u, 0u, stream>>>(
            scratch.selected_counts,
            token,
            static_cast<std::uint32_t>(selected_count));
        if (cudaPeekAtLastError() != cudaSuccess) {
            return fail_transaction(cache, QsaAdapterStatus::kCudaError);
        }
    }

    status = map_attention_status(attention::forward_qsa_cuda(
        attention::checkpoint_config,
        &cache->attention_cache,
        staged,
        scratch.selected_indices,
        scratch.selected_counts,
        kSelectedStride,
        scratch.attention_output_bf16,
        scratch.attention_device_status,
        stream));
    if (status != QsaAdapterStatus::kOk) return fail_transaction(cache, status);

    const std::size_t attention_elements =
        token_count * kQsaAdapterAttentionOutputSize;
    constexpr unsigned int conversion_threads = 256u;
    const unsigned int conversion_blocks = static_cast<unsigned int>(
        (attention_elements + conversion_threads - 1u) / conversion_threads);
    attention_bf16_to_f32_kernel<<<
        conversion_blocks, conversion_threads, 0u, stream>>>(
        scratch.attention_output_bf16,
        scratch.attention_output_f32,
        attention_elements,
        scratch.attention_device_status);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return fail_transaction(cache, QsaAdapterStatus::kCudaError);
    }
    status = run_linear(
        impl_->o_projection.get(), scratch.attention_output_f32, output_f32);
    return status == QsaAdapterStatus::kOk
        ? QsaAdapterStatus::kOk
        : fail_transaction(cache, status);
}

QsaAdapterStatus QsaLayerAdapter::collect_device_status(
    QsaLayerCache* cache,
    const QsaAdapterScratch& scratch,
    QsaAdapterCollectedStatus* collected,
    cudaStream_t stream) noexcept {
    if (collected == nullptr) return QsaAdapterStatus::kInvalidArgument;
    *collected = QsaAdapterCollectedStatus{};
    if (!impl_ || !impl_->ready) return QsaAdapterStatus::kStateMismatch;
    const QsaAdapterStatus cache_status = validate_cache(cache);
    if (cache_status != QsaAdapterStatus::kOk) return cache_status;
    if (!cache->transaction_open || cache->staged_tokens == 0u ||
        scratch.qsa_device_status == nullptr ||
        scratch.attention_device_status == nullptr) {
        return QsaAdapterStatus::kStateMismatch;
    }
    qsa::status indexer_status = qsa::status::cuda_failure;
    const qsa::status qsa_collect = qsa::cuda_collect_status(
        scratch.qsa_device_status, stream, &indexer_status);
    if (qsa_collect != qsa::status::ok) {
        return fail_transaction(cache, map_qsa_status(qsa_collect));
    }
    attention::status attention_status = attention::status::cuda_failure;
    const attention::status attention_collect = attention::collect_device_status(
        scratch.attention_device_status, &attention_status, stream);
    if (attention_collect != attention::status::ok) {
        return fail_transaction(cache, map_attention_status(attention_collect));
    }
    collected->indexer = indexer_status;
    collected->attention_core = attention_status;
    if (indexer_status != qsa::status::ok ||
        attention_status != attention::status::ok) {
        return fail_transaction(cache, QsaAdapterStatus::kDeviceRejected);
    }
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus QsaLayerAdapter::commit_prefix(
    QsaLayerCache* cache,
    std::size_t accepted_tokens) noexcept {
    if (!impl_ || !impl_->ready) return QsaAdapterStatus::kStateMismatch;
    const QsaAdapterStatus cache_status = validate_cache(cache);
    if (cache_status != QsaAdapterStatus::kOk) return cache_status;
    if (!cache->transaction_open || accepted_tokens > cache->staged_tokens) {
        return QsaAdapterStatus::kInvalidArgument;
    }
    const attention::status status = attention::commit_prefix(
        &cache->attention_cache, accepted_tokens);
    if (status != attention::status::ok) return map_attention_status(status);
    cache->committed_tokens += accepted_tokens;
    cache->staged_tokens = 0u;
    cache->transaction_open = false;
    return QsaAdapterStatus::kOk;
}

QsaAdapterStatus QsaLayerAdapter::rollback(QsaLayerCache* cache) noexcept {
    if (!impl_ || !impl_->ready) return QsaAdapterStatus::kStateMismatch;
    const QsaAdapterStatus cache_status = validate_cache(cache);
    if (cache_status != QsaAdapterStatus::kOk) return cache_status;
    return abort_transaction(cache);
}

const QsaAdapterConfig& QsaLayerAdapter::config() const noexcept {
    static const QsaAdapterConfig empty{};
    return impl_ ? impl_->config : empty;
}

std::size_t QsaLayerAdapter::layer_index() const noexcept {
    return impl_ ? impl_->config.layer_index : 0u;
}

int QsaLayerAdapter::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool QsaLayerAdapter::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
