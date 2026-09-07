#include "axiom/qwen4exp/bf16_linear.hpp"

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <pthread.h>
#include <string>
#include <unordered_map>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kMinimumUploadChunk = 4u * 1024u;
constexpr std::size_t kMaximumUploadChunk = 64u * 1024u * 1024u;
constexpr std::size_t kMaximumBlasWorkspace = 64u * 1024u * 1024u;
constexpr std::size_t kWorkspaceAlignment = 256u;
constexpr unsigned kConvertThreads = 256u;

bool multiply_overflow(std::uint64_t left,
                       std::uint64_t right,
                       std::uint64_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)) {
        return true;
    }
    *result = left * right;
    return false;
}

bool size_multiply_overflow(std::size_t left,
                            std::size_t right,
                            std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return true;
    }
    *result = left * right;
    return false;
}

bf16_linear_status fail(bf16_linear_status status,
                        std::string *error,
                        const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return status;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0u) {
        return static_cast<std::uint16_t>((bits >> 16u) | 0x0040u);
    }
    const std::uint32_t rounding_bias = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16u);
}

bool current_blackwell_device(int *device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, *device) != cudaSuccess) return false;
    return properties.major == 12 && properties.minor == 0;
}

bool device_range_valid(const void *pointer,
                        std::size_t required_bytes,
                        int expected_device,
                        std::size_t alignment) noexcept {
    if (pointer == nullptr || required_bytes == 0u) return false;
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
    if (alignment != 0u && address % alignment != 0u) return false;

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status = cudaPointerGetAttributes(&attributes, pointer);
    if (pointer_status != cudaSuccess) {
        // A rejected host/stale pointer must not poison the next valid launch
        // through CUDA's per-thread last-error slot.
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

bool ranges_overlap(const void *left,
                    std::size_t left_bytes,
                    const void *right,
                    std::size_t right_bytes) noexcept {
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

class pinned_buffer final {
public:
    pinned_buffer() = default;
    ~pinned_buffer() {
        if (pointer_ != nullptr) (void)cudaFreeHost(pointer_);
    }
    pinned_buffer(const pinned_buffer &) = delete;
    pinned_buffer &operator=(const pinned_buffer &) = delete;

    cudaError_t allocate(std::size_t bytes) noexcept {
        if (pointer_ != nullptr || bytes == 0u) return cudaErrorInvalidValue;
        return cudaHostAlloc(&pointer_, bytes, cudaHostAllocDefault);
    }

    cudaError_t release() noexcept {
        if (pointer_ == nullptr) return cudaSuccess;
        void *released = pointer_;
        pointer_ = nullptr;
        return cudaFreeHost(released);
    }

    [[nodiscard]] void *get() const noexcept { return pointer_; }

private:
    void *pointer_ = nullptr;
};

__global__ void convert_f32_to_bf16(const float *input,
                                    std::uint16_t *output,
                                    std::size_t elements) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const __nv_bfloat16 converted = __float2bfloat16_rn(input[index]);
    output[index] = reinterpret_cast<const __nv_bfloat16_raw &>(converted).x;
}

struct bf16_device_execution_context final {
    cublasHandle_t handle = nullptr;
    int device = -1;
    pthread_mutex_t dispatch_mutex = PTHREAD_MUTEX_INITIALIZER;

    ~bf16_device_execution_context() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && previous != device && device >= 0 &&
                             cudaSetDevice(device) == cudaSuccess;
        if (handle != nullptr) {
            (void)cublasDestroy(handle);
            handle = nullptr;
        }
        (void)pthread_mutex_destroy(&dispatch_mutex);
        if (changed) (void)cudaSetDevice(previous);
    }
};

struct bf16_context_registry_entry {
    std::weak_ptr<bf16_device_execution_context> context;
    std::uint64_t handles_created = 0u;
    std::uint64_t acquisitions = 0u;
};

pthread_mutex_t &bf16_context_registry_mutex() {
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    return mutex;
}

class pthread_lock_guard final {
public:
    explicit pthread_lock_guard(pthread_mutex_t *mutex) noexcept
        : mutex_(mutex), locked_(mutex != nullptr && pthread_mutex_lock(mutex) == 0) {}
    ~pthread_lock_guard() {
        if (locked_) (void)pthread_mutex_unlock(mutex_);
    }
    pthread_lock_guard(const pthread_lock_guard &) = delete;
    pthread_lock_guard &operator=(const pthread_lock_guard &) = delete;
    [[nodiscard]] bool locked() const noexcept { return locked_; }

private:
    pthread_mutex_t *mutex_ = nullptr;
    bool locked_ = false;
};

std::unordered_map<int, bf16_context_registry_entry> &bf16_context_registry() {
    static std::unordered_map<int, bf16_context_registry_entry> registry;
    return registry;
}

bf16_linear_status acquire_bf16_execution_context(
        int device,
        std::shared_ptr<bf16_device_execution_context> *out) noexcept {
    if (device < 0 || out == nullptr) return bf16_linear_status::invalid_argument;
    out->reset();
    try {
        pthread_lock_guard lock(&bf16_context_registry_mutex());
        if (!lock.locked()) return bf16_linear_status::cublas_error;
        auto &entry = bf16_context_registry()[device];
        if (std::shared_ptr<bf16_device_execution_context> existing =
                    entry.context.lock()) {
            ++entry.acquisitions;
            *out = std::move(existing);
            return bf16_linear_status::ok;
        }

        auto created = std::make_shared<bf16_device_execution_context>();
        created->device = device;
        if (cublasCreate(&created->handle) != CUBLAS_STATUS_SUCCESS ||
            cublasSetPointerMode(created->handle,
                                 CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS ||
            cublasSetAtomicsMode(created->handle,
                                 CUBLAS_ATOMICS_NOT_ALLOWED) != CUBLAS_STATUS_SUCCESS ||
            cublasSetMathMode(created->handle,
                              CUBLAS_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
            return bf16_linear_status::cublas_error;
        }
        entry.context = created;
        ++entry.handles_created;
        ++entry.acquisitions;
        *out = std::move(created);
        return bf16_linear_status::ok;
    } catch (const std::bad_alloc &) {
        return bf16_linear_status::allocation_failure;
    } catch (...) {
        return bf16_linear_status::cublas_error;
    }
}

}  // namespace

struct resident_bf16_linear::impl {
    bf16_linear_config config{};
    std::string tensor_name;
    std::uint16_t *weight = nullptr;
    std::shared_ptr<bf16_device_execution_context> execution_context;
    std::uint64_t weight_bytes = 0u;
    int device = -1;
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && previous != device && device >= 0 &&
                             cudaSetDevice(device) == cudaSuccess;
        if (weight != nullptr) {
            (void)cudaFree(weight);
            weight = nullptr;
        }
        if (changed) (void)cudaSetDevice(previous);
    }
};

bf16_linear_status bf16_linear_get_execution_context_metrics(
        int device,
        bf16_linear_execution_context_metrics *metrics) noexcept {
    if (device < 0 || metrics == nullptr) {
        return bf16_linear_status::invalid_argument;
    }
    *metrics = {};
    try {
        pthread_lock_guard lock(&bf16_context_registry_mutex());
        if (!lock.locked()) return bf16_linear_status::cublas_error;
        const auto found = bf16_context_registry().find(device);
        if (found == bf16_context_registry().end()) {
            return bf16_linear_status::ok;
        }
        metrics->handles_created = found->second.handles_created;
        metrics->context_acquisitions = found->second.acquisitions;
        if (!found->second.context.expired()) {
            metrics->live_contexts = 1u;
            metrics->live_handles = 1u;
        }
        return bf16_linear_status::ok;
    } catch (...) {
        return bf16_linear_status::allocation_failure;
    }
}

const char *bf16_linear_status_string(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return "ok";
        case bf16_linear_status::invalid_argument: return "invalid_argument";
        case bf16_linear_status::unsupported_config: return "unsupported_config";
        case bf16_linear_status::tensor_not_found: return "tensor_not_found";
        case bf16_linear_status::dtype_mismatch: return "dtype_mismatch";
        case bf16_linear_status::shape_mismatch: return "shape_mismatch";
        case bf16_linear_status::size_overflow: return "size_overflow";
        case bf16_linear_status::unsupported_device: return "unsupported_device";
        case bf16_linear_status::invalid_device_pointer: return "invalid_device_pointer";
        case bf16_linear_status::checkpoint_io_error: return "checkpoint_io_error";
        case bf16_linear_status::allocation_failure: return "allocation_failure";
        case bf16_linear_status::cuda_error: return "cuda_error";
        case bf16_linear_status::cublas_error: return "cublas_error";
    }
    return "unknown";
}

bf16_linear_status bf16_linear_validate_config(
        const bf16_linear_config &config) noexcept {
    if (config.input_features == 0u || config.output_features == 0u ||
        config.max_batch == 0u) {
        return bf16_linear_status::invalid_argument;
    }
    if (config.input_features > static_cast<std::uint64_t>(INT_MAX) ||
        config.output_features > static_cast<std::uint64_t>(INT_MAX) ||
        config.max_batch > static_cast<std::size_t>(INT_MAX) ||
        config.upload_chunk_bytes < kMinimumUploadChunk ||
        config.upload_chunk_bytes > kMaximumUploadChunk ||
        (config.upload_chunk_bytes % sizeof(std::uint16_t)) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > kMaximumBlasWorkspace ||
        (config.blas_workspace_bytes % kWorkspaceAlignment) != 0u) {
        return bf16_linear_status::unsupported_config;
    }

    std::uint64_t elements = 0u;
    std::uint64_t bytes = 0u;
    if (multiply_overflow(config.input_features, config.output_features, &elements) ||
        multiply_overflow(elements, sizeof(std::uint16_t), &bytes) ||
        bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return bf16_linear_status::size_overflow;
    }
    std::size_t input_elements = 0u;
    std::size_t input_bf16_bytes = 0u;
    std::size_t input_f32_bytes = 0u;
    std::size_t output_elements = 0u;
    std::size_t output_f32_bytes = 0u;
    if (size_multiply_overflow(config.max_batch,
                               static_cast<std::size_t>(config.input_features),
                               &input_elements) ||
        size_multiply_overflow(input_elements, sizeof(std::uint16_t),
                               &input_bf16_bytes) ||
        size_multiply_overflow(input_elements, sizeof(float), &input_f32_bytes) ||
        size_multiply_overflow(config.max_batch,
                               static_cast<std::size_t>(config.output_features),
                               &output_elements) ||
        size_multiply_overflow(output_elements, sizeof(float),
                               &output_f32_bytes)) {
        return bf16_linear_status::size_overflow;
    }
    return bf16_linear_status::ok;
}

bf16_linear_status bf16_linear_get_workspace_requirements(
        const bf16_linear_config &config,
        bf16_linear_workspace_requirements *requirements) noexcept {
    if (requirements == nullptr) return bf16_linear_status::invalid_argument;
    *requirements = {};
    const bf16_linear_status validation = bf16_linear_validate_config(config);
    if (validation != bf16_linear_status::ok) return validation;
    std::size_t elements = 0u;
    if (size_multiply_overflow(config.max_batch,
                               static_cast<std::size_t>(config.input_features),
                               &elements) ||
        size_multiply_overflow(elements, sizeof(std::uint16_t),
                               &requirements->input_bf16_bytes)) {
        return bf16_linear_status::size_overflow;
    }
    requirements->blas_workspace_bytes = config.blas_workspace_bytes;
    return bf16_linear_status::ok;
}

bf16_linear_status bf16_linear_reference_f32(
        const bf16_linear_config &config,
        const std::uint16_t *weight_bf16,
        const float *input_f32,
        std::size_t batch,
        float *output_f32) noexcept {
    const bf16_linear_status validation = bf16_linear_validate_config(config);
    if (validation != bf16_linear_status::ok) return validation;
    if (weight_bf16 == nullptr || input_f32 == nullptr || output_f32 == nullptr ||
        batch == 0u || batch > config.max_batch) {
        return bf16_linear_status::invalid_argument;
    }
    const std::size_t input_features =
            static_cast<std::size_t>(config.input_features);
    const std::size_t output_features =
            static_cast<std::size_t>(config.output_features);
    for (std::size_t token = 0u; token < batch; ++token) {
        for (std::size_t row = 0u; row < output_features; ++row) {
            float sum = 0.0F;
            for (std::size_t column = 0u; column < input_features; ++column) {
                const float input = bf16_to_float(float_to_bf16(
                        input_f32[token * input_features + column]));
                const float weight = bf16_to_float(
                        weight_bf16[row * input_features + column]);
                sum = std::fma(input, weight, sum);
            }
            output_f32[token * output_features + row] = sum;
        }
    }
    return bf16_linear_status::ok;
}

resident_bf16_linear::resident_bf16_linear() = default;
resident_bf16_linear::~resident_bf16_linear() = default;
resident_bf16_linear::resident_bf16_linear(resident_bf16_linear &&) noexcept = default;
resident_bf16_linear &resident_bf16_linear::operator=(
        resident_bf16_linear &&) noexcept = default;

bf16_linear_status resident_bf16_linear::load(
        const checkpoint_catalog &catalog,
        const std::string &tensor_name,
        const bf16_linear_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<resident_bf16_linear> *out,
        std::string *error) noexcept {
    if (out == nullptr || tensor_name.empty()) {
        return fail(bf16_linear_status::invalid_argument, error,
                    "invalid BF16 linear load arguments");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const bf16_linear_status validation = bf16_linear_validate_config(config);
    if (validation != bf16_linear_status::ok) {
        return fail(validation, error,
                    std::string("invalid BF16 linear config: ") +
                    bf16_linear_status_string(validation));
    }

    const tensor_span *span = catalog.find(tensor_name);
    if (span == nullptr) {
        return fail(bf16_linear_status::tensor_not_found, error,
                    "checkpoint tensor not found: " + tensor_name);
    }
    if (span->dtype != tensor_dtype::bf16) {
        return fail(bf16_linear_status::dtype_mismatch, error,
                    "checkpoint tensor is not BF16: " + tensor_name);
    }
    if (span->rank != 2u || span->shape[0] != config.output_features ||
        span->shape[1] != config.input_features) {
        return fail(bf16_linear_status::shape_mismatch, error,
                    "checkpoint tensor rank/shape mismatch: " + tensor_name);
    }
    std::uint64_t elements = 0u;
    std::uint64_t expected_bytes = 0u;
    if (multiply_overflow(config.input_features, config.output_features, &elements) ||
        multiply_overflow(elements, sizeof(std::uint16_t), &expected_bytes)) {
        return fail(bf16_linear_status::size_overflow, error,
                    "checkpoint tensor byte count overflow: " + tensor_name);
    }
    if (span->bytes != expected_bytes) {
        return fail(bf16_linear_status::shape_mismatch, error,
                    "checkpoint tensor byte length mismatch: " + tensor_name);
    }

    int device = -1;
    if (!current_blackwell_device(&device)) {
        return fail(bf16_linear_status::unsupported_device, error,
                    "BF16 linear requires an active SM120 Blackwell device");
    }

    try {
        auto result = std::make_unique<resident_bf16_linear>();
        result->impl_ = std::make_unique<impl>();
        result->impl_->config = config;
        result->impl_->tensor_name = tensor_name;
        result->impl_->weight_bytes = expected_bytes;
        result->impl_->device = device;

        if (cudaMalloc(reinterpret_cast<void **>(&result->impl_->weight),
                       static_cast<std::size_t>(expected_bytes)) != cudaSuccess) {
            return fail(bf16_linear_status::allocation_failure, error,
                        "device allocation failed for BF16 tensor: " + tensor_name);
        }

        const std::size_t staging_bytes = std::min<std::size_t>(
                config.upload_chunk_bytes, static_cast<std::size_t>(expected_bytes));
        pinned_buffer staging;
        if (staging.allocate(staging_bytes) != cudaSuccess) {
            return fail(bf16_linear_status::allocation_failure, error,
                        "pinned staging allocation failed for BF16 tensor: " + tensor_name);
        }

        bf16_linear_status upload_status = bf16_linear_status::ok;
        std::uint64_t offset = 0u;
        while (offset < expected_bytes) {
            const std::size_t amount = static_cast<std::size_t>(
                    std::min<std::uint64_t>(staging_bytes, expected_bytes - offset));
            std::string read_error;
            if (!catalog.read_range(tensor_name, offset, staging.get(), amount, &read_error)) {
                upload_status = fail(bf16_linear_status::checkpoint_io_error, error,
                                     read_error.empty()
                                             ? "checkpoint range read failed: " + tensor_name
                                             : read_error);
                break;
            }
            if (cudaMemcpyAsync(
                        reinterpret_cast<unsigned char *>(result->impl_->weight) + offset,
                        staging.get(), amount, cudaMemcpyHostToDevice,
                        initialization_stream) != cudaSuccess ||
                cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
                upload_status = fail(bf16_linear_status::cuda_error, error,
                                     "checkpoint GPU upload failed: " + tensor_name);
                break;
            }
            offset += amount;
        }
        const cudaError_t free_staging = staging.release();
        if (upload_status != bf16_linear_status::ok) return upload_status;
        if (free_staging != cudaSuccess) {
            return fail(bf16_linear_status::cuda_error, error,
                        "pinned staging release failed: " + tensor_name);
        }

        const bf16_linear_status context_status = acquire_bf16_execution_context(
                device, &result->impl_->execution_context);
        if (context_status != bf16_linear_status::ok) {
            return fail(context_status, error,
                        "cuBLAS initialization failed for BF16 tensor: " + tensor_name);
        }
        result->impl_->ready = true;
        *out = std::move(result);
        return bf16_linear_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(bf16_linear_status::allocation_failure, error,
                    "host allocation failed while loading BF16 tensor: " + tensor_name);
    } catch (const std::exception &exception) {
        return fail(bf16_linear_status::allocation_failure, error,
                    std::string("BF16 tensor load exception: ") + exception.what());
    } catch (...) {
        return fail(bf16_linear_status::allocation_failure, error,
                    "unknown BF16 tensor load exception");
    }
}

bf16_linear_status resident_bf16_linear::forward(
        const float *input_f32,
        std::size_t batch,
        const bf16_linear_scratch &scratch,
        float *output_f32,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || impl_->weight == nullptr ||
        !impl_->execution_context || impl_->execution_context->handle == nullptr) {
        return bf16_linear_status::invalid_argument;
    }
    if (input_f32 == nullptr || output_f32 == nullptr || batch == 0u ||
        batch > impl_->config.max_batch || scratch.input_bf16 == nullptr ||
        scratch.blas_workspace == nullptr) {
        return bf16_linear_status::invalid_argument;
    }

    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess || active_device != impl_->device) {
        return bf16_linear_status::unsupported_device;
    }
    std::size_t input_elements = 0u;
    std::size_t input_f32_bytes = 0u;
    std::size_t input_bf16_bytes = 0u;
    std::size_t output_elements = 0u;
    std::size_t output_bytes = 0u;
    if (size_multiply_overflow(batch,
                               static_cast<std::size_t>(impl_->config.input_features),
                               &input_elements) ||
        size_multiply_overflow(input_elements, sizeof(float), &input_f32_bytes) ||
        size_multiply_overflow(input_elements, sizeof(std::uint16_t),
                               &input_bf16_bytes) ||
        size_multiply_overflow(batch,
                               static_cast<std::size_t>(impl_->config.output_features),
                               &output_elements) ||
        size_multiply_overflow(output_elements, sizeof(float), &output_bytes)) {
        return bf16_linear_status::size_overflow;
    }
    if (scratch.input_bf16_bytes < input_bf16_bytes ||
        scratch.blas_workspace_bytes < impl_->config.blas_workspace_bytes ||
        !device_range_valid(input_f32, input_f32_bytes, impl_->device, alignof(float)) ||
        !device_range_valid(output_f32, output_bytes, impl_->device, alignof(float)) ||
        !device_range_valid(scratch.input_bf16, input_bf16_bytes,
                            impl_->device, alignof(std::uint16_t)) ||
        !device_range_valid(scratch.blas_workspace,
                            impl_->config.blas_workspace_bytes,
                            impl_->device, kWorkspaceAlignment) ||
        ranges_overlap(input_f32, input_f32_bytes, output_f32, output_bytes) ||
        ranges_overlap(input_f32, input_f32_bytes,
                       scratch.input_bf16, input_bf16_bytes) ||
        ranges_overlap(input_f32, input_f32_bytes,
                       scratch.blas_workspace, impl_->config.blas_workspace_bytes) ||
        ranges_overlap(output_f32, output_bytes,
                       scratch.input_bf16, input_bf16_bytes) ||
        ranges_overlap(output_f32, output_bytes,
                       scratch.blas_workspace, impl_->config.blas_workspace_bytes) ||
        ranges_overlap(scratch.input_bf16, input_bf16_bytes,
                       scratch.blas_workspace, impl_->config.blas_workspace_bytes)) {
        return bf16_linear_status::invalid_device_pointer;
    }

    const std::size_t blocks_size =
            (input_elements + kConvertThreads - 1u) / kConvertThreads;
    if (blocks_size == 0u ||
        blocks_size > static_cast<std::size_t>(std::numeric_limits<unsigned>::max())) {
        return bf16_linear_status::size_overflow;
    }
    convert_f32_to_bf16<<<static_cast<unsigned>(blocks_size), kConvertThreads, 0, stream>>>(
            input_f32, scratch.input_bf16, input_elements);
    if (cudaPeekAtLastError() != cudaSuccess) return bf16_linear_status::cuda_error;

    pthread_lock_guard dispatch_lock(&impl_->execution_context->dispatch_mutex);
    if (!dispatch_lock.locked()) return bf16_linear_status::cublas_error;
    if (cublasSetStream(impl_->execution_context->handle, stream) !=
                CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(impl_->execution_context->handle, scratch.blas_workspace,
                           impl_->config.blas_workspace_bytes) != CUBLAS_STATUS_SUCCESS) {
        return bf16_linear_status::cublas_error;
    }

    const int output_features = static_cast<int>(impl_->config.output_features);
    const int input_features = static_cast<int>(impl_->config.input_features);
    const int tokens = static_cast<int>(batch);
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const cublasStatus_t gemm = cublasGemmEx(
            impl_->execution_context->handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            output_features,
            tokens,
            input_features,
            &alpha,
            impl_->weight,
            CUDA_R_16BF,
            input_features,
            scratch.input_bf16,
            CUDA_R_16BF,
            input_features,
            &beta,
            output_f32,
            CUDA_R_32F,
            output_features,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return gemm == CUBLAS_STATUS_SUCCESS ? bf16_linear_status::ok
                                         : bf16_linear_status::cublas_error;
}

const bf16_linear_config &resident_bf16_linear::config() const noexcept {
    static const bf16_linear_config empty{};
    return impl_ ? impl_->config : empty;
}

const std::string &resident_bf16_linear::tensor_name() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->tensor_name : empty;
}

std::uint64_t resident_bf16_linear::weight_bytes() const noexcept {
    return impl_ ? impl_->weight_bytes : 0u;
}

int resident_bf16_linear::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool resident_bf16_linear::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
