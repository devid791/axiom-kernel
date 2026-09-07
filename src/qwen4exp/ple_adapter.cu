// Match the native threaded CUDA TUs: libstdc++ mutex/condition_variable
// needs GNU pthread declarations even when nvcc globally uses -U_GNU_SOURCE.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cstddef>
// nvcc may preinclude glibc features before this TU's _GNU_SOURCE definition.
// In that case libstdc++ advertises GNU timed-pthread functions whose
// declarations are hidden. This worker uses only untimed waits/locks.
#if defined(__GLIBCXX__) && !defined(__USE_GNU)
#undef _GLIBCXX_USE_PTHREAD_COND_CLOCKWAIT
#undef _GLIBCXX_USE_PTHREAD_MUTEX_CLOCKLOCK
#endif
#include "axiom/qwen4exp/ple_adapter.hpp"

#include <condition_variable>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

// CPU-only test seam: compile the very same single-job worker without any
// CUDA implementation/linkage. No production behavior is switched by this.
namespace axiom::qwen4exp::detail {
struct ple_io_result {
    ple_adapter_status code = ple_adapter_status::ok;
    std::string message;
};

class ple_io_worker final {
public:
    using operation = ple_io_result (*)(void *, const ple_head_ids &);
    ple_io_worker() = default;
    ple_io_worker(const ple_io_worker &) = delete;
    ple_io_worker &operator=(const ple_io_worker &) = delete;
    ~ple_io_worker() { shutdown(); }

    bool start() noexcept {
        if (thread_.joinable() || stopping_) return false;
        try { thread_ = std::thread([this] { run(); }); }
        catch (...) { return false; }
        return true;
    }

    bool submit(operation function, void *context, const ple_head_ids &ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable() || stopping_ || active_ || function == nullptr)
            return false;
        function_ = function;
        context_ = context;
        ids_ = ids; // Never retain a caller's token/ID array.
        active_ = queued_ = true;
        done_ = false;
        changed_.notify_one();
        return true;
    }

    ple_io_result collect() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!active_) return {ple_adapter_status::invalid_state, "no PLE I/O job"};
        changed_.wait(lock, [this] { return done_; });
        active_ = done_ = false;
        return std::move(result_);
    }

    // Stop accepting work, finish at most the one accepted job, and join.
    // A blocking pread is not interruptible through this interface.
    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            changed_.notify_all();
        }
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() noexcept {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            changed_.wait(lock, [this] { return queued_ || stopping_; });
            if (!queued_) return;
            const auto function = function_;
            void *context = context_;
            const auto ids = ids_;
            queued_ = false;
            lock.unlock();
            ple_io_result result;
            try { result = function(context, ids); }
            catch (const std::bad_alloc &) {
                result.code = ple_adapter_status::resource_exhausted;
            } catch (...) {
                result.code = ple_adapter_status::embedding_error;
            }
            lock.lock();
            result_ = std::move(result);
            done_ = true;
            changed_.notify_all();
        }
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    std::thread thread_;
    operation function_ = nullptr;
    void *context_ = nullptr;
    ple_head_ids ids_{};
    ple_io_result result_{};
    bool active_ = false, queued_ = false, done_ = false, stopping_ = false;
};
} // namespace axiom::qwen4exp::detail

#ifndef AXIOM_QWEN4EXP_PLE_WORKER_TEST_ONLY
#include "axiom/qwen4exp/ple_compute.hpp"
#include "axiom/qwen4exp/ple_embedding.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

constexpr char kPlePrefix[] = "model.language_model.layers.1.ple.";
constexpr char kMetadataPrefix[] =
        "model.language_model.layers.1.ple.ple_embedding.";
constexpr std::size_t kHidden = kPleComputeHidden;
constexpr std::size_t kStreams = kPleComputeStreams;
constexpr std::size_t kChannels = kHidden * kStreams;
constexpr std::size_t kEmbedding = kPleComputeEmbedding;
constexpr std::size_t kWorkspaceFloats =
        kEmbedding + kChannels + kHidden + 4u * kChannels;
constexpr unsigned kThreads = 256u;

struct tensor_contract {
    const char *suffix = nullptr;
    tensor_dtype dtype = tensor_dtype::unknown;
    std::uint32_t rank = 0u;
    std::array<std::uint64_t, 3> shape{};
};

constexpr std::array<tensor_contract, kPleAdapterResidentWeightTensors>
        kWeightContracts{{
                {"key_proj.weight", tensor_dtype::bf16, 2u,
                 {kChannels, kEmbedding, 0u}},
                {"value_proj.weight", tensor_dtype::bf16, 2u,
                 {kHidden, kEmbedding, 0u}},
                {"norm_key.weight", tensor_dtype::bf16, 1u,
                 {kChannels, 0u, 0u}},
                {"norm_query.weight", tensor_dtype::bf16, 1u,
                 {kChannels, 0u, 0u}},
                {"norm_conv.weight", tensor_dtype::bf16, 1u,
                 {kChannels, 0u, 0u}},
                {"conv1d.weight", tensor_dtype::bf16, 3u,
                 {kChannels, 1u, kPleComputeTaps}},
        }};

struct metadata_contract {
    const char *suffix = nullptr;
    std::size_t elements = 0u;
};

constexpr std::array<metadata_contract, kPleAdapterMetadataTensors>
        kMetadataContracts{{
                {"layer_multipliers", kPleNgramOrder},
                {"ngram_heads_offsets", kPleHeadCount},
                {"ngram_heads_vocab_sizes", kPleHeadCount},
        }};

bool checked_mul(std::size_t left,
                 std::size_t right,
                 std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left,
                 std::size_t right,
                 std::size_t *result) noexcept {
    if (result == nullptr ||
        right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

ple_adapter_status fail(ple_adapter_status code,
                        std::string *error,
                        const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return code;
}

bool starts_with(const std::string &value, const char *prefix) noexcept {
    const std::size_t length = std::strlen(prefix);
    return value.size() >= length &&
           value.compare(0u, length, prefix, length) == 0;
}

std::size_t contract_elements(const tensor_contract &contract,
                              bool *valid) noexcept {
    if (valid == nullptr || contract.rank == 0u ||
        contract.rank > contract.shape.size()) {
        if (valid != nullptr) *valid = false;
        return 0u;
    }
    std::size_t elements = 1u;
    for (std::uint32_t index = 0u; index < contract.rank; ++index) {
        if (contract.shape[index] == 0u ||
            contract.shape[index] >
                    static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max()) ||
            !checked_mul(elements,
                         static_cast<std::size_t>(contract.shape[index]),
                         &elements)) {
            *valid = false;
            return 0u;
        }
    }
    *valid = true;
    return elements;
}

bool current_supported_device(int *device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, *device) == cudaSuccess &&
           properties.major >= 12;
}

struct device_range {
    const void *pointer = nullptr;
    std::size_t bytes = 0u;
};

bool valid_device_range(const device_range &range,
                        int expected_device) noexcept {
    if (range.pointer == nullptr || range.bytes == 0u) return false;
    cudaPointerAttributes attributes{};
    const cudaError_t attribute_status =
            cudaPointerGetAttributes(&attributes, range.pointer);
    if (attribute_status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        return false;
    }

    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    const CUresult range_status = cuMemGetAddressRange(
            &allocation_base, &allocation_bytes,
            static_cast<CUdeviceptr>(
                    reinterpret_cast<std::uintptr_t>(range.pointer)));
    if (range_status != CUDA_SUCCESS) return false;
    const std::uintptr_t address =
            reinterpret_cast<std::uintptr_t>(range.pointer);
    const std::uintptr_t base =
            static_cast<std::uintptr_t>(allocation_base);
    if (address < base || range.bytes > allocation_bytes) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes - range.bytes;
}

bool ranges_overlap(const device_range &left,
                    const device_range &right) noexcept {
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

ple_adapter_status map_embedding_status(
        ple_embedding_status_code code) noexcept {
    switch (code) {
        case ple_embedding_status_code::ok:
            return ple_adapter_status::ok;
        case ple_embedding_status_code::invalid_argument:
            return ple_adapter_status::invalid_argument;
        case ple_embedding_status_code::invalid_checkpoint:
            return ple_adapter_status::invalid_checkpoint;
        case ple_embedding_status_code::out_of_range:
            return ple_adapter_status::invalid_checkpoint;
        case ple_embedding_status_code::io_error:
            return ple_adapter_status::checkpoint_io_error;
        case ple_embedding_status_code::non_finite:
            return ple_adapter_status::invalid_checkpoint;
        case ple_embedding_status_code::resource_exhausted:
            return ple_adapter_status::resource_exhausted;
    }
    return ple_adapter_status::embedding_error;
}

ple_adapter_status map_ple_status(ple_status_code code) noexcept {
    switch (code) {
        case ple_status_code::ok:
            return ple_adapter_status::ok;
        case ple_status_code::invalid_argument:
            return ple_adapter_status::invalid_argument;
        case ple_status_code::invalid_metadata:
            return ple_adapter_status::metadata_mismatch;
        case ple_status_code::invalid_state:
            return ple_adapter_status::invalid_state;
        case ple_status_code::out_of_range:
            return ple_adapter_status::invalid_argument;
        case ple_status_code::arithmetic_overflow:
            return ple_adapter_status::size_overflow;
        case ple_status_code::resource_exhausted:
            return ple_adapter_status::resource_exhausted;
    }
    return ple_adapter_status::invalid_state;
}

ple_adapter_status map_compute_status(ple_compute_status status) noexcept {
    switch (status) {
        case ple_compute_status::ok:
            return ple_adapter_status::ok;
        case ple_compute_status::invalid_argument:
            return ple_adapter_status::invalid_argument;
        case ple_compute_status::unsupported_config:
            return ple_adapter_status::unsupported_config;
        case ple_compute_status::invalid_state:
            return ple_adapter_status::invalid_state;
        case ple_compute_status::size_overflow:
            return ple_adapter_status::size_overflow;
        case ple_compute_status::unsupported_device:
            return ple_adapter_status::unsupported_device;
        case ple_compute_status::cuda_error:
            return ple_adapter_status::cuda_error;
    }
    return ple_adapter_status::compute_error;
}

ple_adapter_status validate_tensor(
        const checkpoint_catalog &catalog,
        const std::string &name,
        const tensor_contract &contract,
        const tensor_span **span_out,
        std::string *error) noexcept {
    if (span_out == nullptr) {
        return fail(ple_adapter_status::invalid_argument, error,
                    "null PLE tensor output");
    }
    const tensor_span *span = catalog.find(name);
    if (span == nullptr) {
        return fail(ple_adapter_status::tensor_not_found, error,
                    "checkpoint tensor not found: " + name);
    }
    if (span->dtype != contract.dtype) {
        return fail(ple_adapter_status::dtype_mismatch, error,
                    "checkpoint tensor dtype mismatch: " + name);
    }
    if (span->rank != contract.rank) {
        return fail(ple_adapter_status::shape_mismatch, error,
                    "checkpoint tensor rank mismatch: " + name);
    }
    for (std::uint32_t index = 0u; index < contract.rank; ++index) {
        if (span->shape[index] != contract.shape[index]) {
            return fail(ple_adapter_status::shape_mismatch, error,
                        "checkpoint tensor shape mismatch: " + name);
        }
    }
    bool valid_elements = false;
    const std::size_t elements = contract_elements(contract, &valid_elements);
    std::size_t expected_bytes = 0u;
    if (!valid_elements || !checked_mul(elements, sizeof(std::uint16_t),
                                         &expected_bytes) ||
        span->bytes != expected_bytes) {
        return fail(ple_adapter_status::shape_mismatch, error,
                    "checkpoint tensor byte count mismatch: " + name);
    }
    *span_out = span;
    return ple_adapter_status::ok;
}

template <std::size_t Size>
ple_adapter_status read_i64_metadata(
        const checkpoint_catalog &catalog,
        const char *suffix,
        std::array<std::int64_t, Size> *output,
        std::string *error) noexcept {
    if (suffix == nullptr || output == nullptr) {
        return fail(ple_adapter_status::invalid_argument, error,
                    "invalid PLE metadata read request");
    }
    std::string name;
    try {
        name = std::string(kMetadataPrefix) + suffix;
    } catch (...) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not construct PLE metadata name");
    }
    const tensor_span *span = catalog.find(name);
    if (span == nullptr) {
        return fail(ple_adapter_status::tensor_not_found, error,
                    "checkpoint metadata not found: " + name);
    }
    if (span->dtype != tensor_dtype::i64) {
        return fail(ple_adapter_status::dtype_mismatch, error,
                    "checkpoint metadata is not I64: " + name);
    }
    if (span->rank != 1u || span->shape[0] != Size ||
        span->bytes != Size * sizeof(std::int64_t)) {
        return fail(ple_adapter_status::shape_mismatch, error,
                    "checkpoint metadata shape mismatch: " + name);
    }
    std::string read_error;
    if (!catalog.read_range(name, 0u, output->data(),
                            output->size() * sizeof((*output)[0]),
                            &read_error)) {
        return fail(ple_adapter_status::checkpoint_io_error, error,
                    read_error.empty() ? "PLE metadata range read failed: " + name
                                       : read_error);
    }
    return ple_adapter_status::ok;
}

ple_adapter_status load_bf16_weight(
        const checkpoint_catalog &catalog,
        const tensor_span &span,
        std::size_t upload_chunk_bytes,
        cudaStream_t stream,
        std::uint16_t **device_output,
        std::string *error) noexcept {
    if (device_output == nullptr || *device_output != nullptr ||
        span.dtype != tensor_dtype::bf16 || span.bytes == 0u ||
        upload_chunk_bytes == 0u) {
        return fail(ple_adapter_status::invalid_argument, error,
                    "invalid BF16 PLE weight load request");
    }
    if (span.bytes > static_cast<std::uint64_t>(
                             std::numeric_limits<std::size_t>::max())) {
        return fail(ple_adapter_status::size_overflow, error,
                    "PLE weight is too large for this host: " + span.name);
    }
    const std::size_t bytes = static_cast<std::size_t>(span.bytes);
    const std::size_t staging_bytes = std::min(upload_chunk_bytes, bytes);
    void *device = nullptr;
    void *staging = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess ||
        cudaHostAlloc(&staging, staging_bytes, cudaHostAllocPortable) !=
                cudaSuccess) {
        if (staging != nullptr) (void)cudaFreeHost(staging);
        if (device != nullptr) (void)cudaFree(device);
        return fail(ple_adapter_status::resource_exhausted, error,
                    "bounded PLE weight staging allocation failed: " +
                            span.name);
    }

    std::size_t offset = 0u;
    while (offset < bytes) {
        const std::size_t amount = std::min(staging_bytes, bytes - offset);
        std::string read_error;
        if (!catalog.read_range(span.name, offset, staging, amount,
                                &read_error)) {
            (void)cudaFreeHost(staging);
            (void)cudaFree(device);
            return fail(ple_adapter_status::checkpoint_io_error, error,
                        read_error.empty()
                                ? "PLE BF16 range read failed: " + span.name
                                : read_error);
        }
        if (cudaMemcpyAsync(static_cast<unsigned char *>(device) + offset,
                            staging, amount, cudaMemcpyHostToDevice, stream) !=
                    cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            (void)cudaFreeHost(staging);
            (void)cudaFree(device);
            return fail(ple_adapter_status::cuda_error, error,
                        "PLE BF16 upload failed: " + span.name);
        }
        offset += amount;
    }
    const cudaError_t release_status = cudaFreeHost(staging);
    if (release_status != cudaSuccess) {
        (void)cudaFree(device);
        return fail(ple_adapter_status::cuda_error, error,
                    "PLE BF16 staging release failed: " + span.name);
    }
    *device_output = static_cast<std::uint16_t *>(device);
    return ple_adapter_status::ok;
}

__global__ void add_ple_before_attention_hyper_connection_kernel(
        const float *hidden,
        const float *ple,
        float *output,
        std::size_t elements) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = hidden[index] + ple[index];
}

}  // namespace

struct ple_layer_adapter::impl {
    ple_adapter_config config{};
    ple_metadata metadata{};
    ple_embedding_reader embedding_reader{};
    ple_compute_weights compute_weights{};
    std::array<std::uint16_t *, kPleAdapterResidentWeightTensors>
            resident_weights{};
    ple_adapter_footprint footprint{};
    int device = -1;
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        if (device >= 0 && (!have_previous || previous != device)) {
            (void)cudaSetDevice(device);
        }
        for (std::uint16_t *pointer : resident_weights) {
            if (pointer != nullptr) (void)cudaFree(pointer);
        }
        if (have_previous && previous >= 0 && previous != device) {
            (void)cudaSetDevice(previous);
        }
    }
};

struct ple_layer_session::impl {
    const ple_layer_adapter::impl *owner = nullptr;
    ple_session_state token_state{};
    ple_embedding_row_cache row_cache{};
    ple_compute_device_state compute_state{};
    float *host_embedding = nullptr;
    float *device_workspace = nullptr;
    float *device_embedding = nullptr;
    ple_compute_scratch compute_scratch{};
    float *device_ple_output = nullptr;
    ple_head_ids prefetched_ids{};
    std::uint64_t nvme_bytes = 0u;
    bool prefetched = false;
    bool poisoned = false;
    bool ready = false;
    detail::ple_io_worker io_worker{};
    bool io_pending = false;
    std::uint64_t visible_hits = 0u;
    std::uint64_t visible_misses = 0u;

    static detail::ple_io_result gather(void *context, const ple_head_ids &ids) {
        auto &state = *static_cast<impl *>(context);
        const auto result = state.owner->embedding_reader.gather_f32(
                ids, state.host_embedding, kEmbedding, &state.row_cache);
        return {map_embedding_status(result.code), result.message};
    }

    detail::ple_io_result drain_io() {
        if (!io_pending) return {};
        auto result = io_worker.collect(); // Acquires all worker writes.
        io_pending = false;
        const auto before = visible_misses;
        visible_hits = row_cache.hits();
        visible_misses = row_cache.misses();
        if (result.code == ple_adapter_status::ok) {
            if (visible_misses < before ||
                visible_misses - before > kPleAdapterRowsPerToken ||
                visible_misses - before >
                        (std::numeric_limits<std::uint64_t>::max() - nvme_bytes) /
                                kPleAdapterColdRowBytes) {
                result = {ple_adapter_status::size_overflow, "PLE cache accounting overflow"};
            } else {
                nvme_bytes += (visible_misses - before) * kPleAdapterColdRowBytes;
            }
        }
        return result;
    }

    ~impl() {
        // Must precede cache destruction, pinned-buffer free and owner access.
        io_worker.shutdown();
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const int target = owner == nullptr ? -1 : owner->device;
        if (target >= 0 && (!have_previous || previous != target)) {
            (void)cudaSetDevice(target);
        }
        if (compute_state.transaction_open) {
            (void)ple_compute_rollback(&compute_state);
        }
        if (token_state.transaction_open()) (void)token_state.rollback();
        if (compute_state.initialized) {
            (void)ple_compute_device_state_release(&compute_state);
        }
        if (device_workspace != nullptr) (void)cudaFree(device_workspace);
        if (host_embedding != nullptr) (void)cudaFreeHost(host_embedding);
        if (have_previous && previous >= 0 && previous != target) {
            (void)cudaSetDevice(previous);
        }
    }
};

namespace {

template <typename SessionState>
ple_adapter_status abort_open_transaction(SessionState *state,
                                          std::string *error,
                                          const std::string &message,
                                          ple_adapter_status original) noexcept {
    if (state == nullptr) return original;
    (void)state->drain_io(); // Discard cancelled data; never publish it.
    bool rollback_ok = true;
    if (state->compute_state.transaction_open) {
        rollback_ok = ple_compute_rollback(&state->compute_state) ==
                              ple_compute_status::ok &&
                      rollback_ok;
    }
    if (state->token_state.transaction_open()) {
        rollback_ok = state->token_state.rollback().ok() && rollback_ok;
    }
    state->prefetched = false;
    if (!rollback_ok) {
        state->poisoned = true;
        return fail(ple_adapter_status::invalid_state, error,
                    message + "; rollback failed and session is poisoned");
    }
    return fail(original, error, message);
}

}  // namespace

const char *ple_adapter_status_string(ple_adapter_status status) noexcept {
    switch (status) {
        case ple_adapter_status::ok: return "ok";
        case ple_adapter_status::invalid_argument: return "invalid_argument";
        case ple_adapter_status::unsupported_layer: return "unsupported_layer";
        case ple_adapter_status::unsupported_config: return "unsupported_config";
        case ple_adapter_status::invalid_checkpoint: return "invalid_checkpoint";
        case ple_adapter_status::tensor_not_found: return "tensor_not_found";
        case ple_adapter_status::dtype_mismatch: return "dtype_mismatch";
        case ple_adapter_status::shape_mismatch: return "shape_mismatch";
        case ple_adapter_status::metadata_mismatch: return "metadata_mismatch";
        case ple_adapter_status::size_overflow: return "size_overflow";
        case ple_adapter_status::checkpoint_io_error: return "checkpoint_io_error";
        case ple_adapter_status::resource_exhausted: return "resource_exhausted";
        case ple_adapter_status::unsupported_device: return "unsupported_device";
        case ple_adapter_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case ple_adapter_status::invalid_state: return "invalid_state";
        case ple_adapter_status::cuda_error: return "cuda_error";
        case ple_adapter_status::embedding_error: return "embedding_error";
        case ple_adapter_status::compute_error: return "compute_error";
    }
    return "unknown";
}

ple_adapter_status ple_adapter_validate_config(
        const ple_adapter_config &config) noexcept {
    if (config.layer_index != kPleAdapterLayerIndex) {
        return ple_adapter_status::unsupported_layer;
    }
    if (config.max_speculative_tokens == 0u ||
        config.max_speculative_tokens > 256u ||
        config.row_cache_slots < 4u ||
        config.row_cache_slots % 4u != 0u ||
        config.upload_chunk_bytes < sizeof(std::uint16_t) ||
        config.upload_chunk_bytes % sizeof(std::uint16_t) != 0u) {
        return ple_adapter_status::unsupported_config;
    }
    return ple_adapter_status::ok;
}

ple_adapter_status ple_adapter_get_footprint(
        const ple_adapter_config &config,
        ple_adapter_footprint *footprint) noexcept {
    if (footprint == nullptr) return ple_adapter_status::invalid_argument;
    const ple_adapter_status valid = ple_adapter_validate_config(config);
    if (valid != ple_adapter_status::ok) return valid;

    ple_adapter_footprint result{};
    for (const tensor_contract &contract : kWeightContracts) {
        bool valid_elements = false;
        const std::size_t elements =
                contract_elements(contract, &valid_elements);
        std::size_t bytes = 0u;
        if (!valid_elements ||
            !checked_mul(elements, sizeof(std::uint16_t), &bytes) ||
            !checked_add(result.resident_weight_bytes, bytes,
                         &result.resident_weight_bytes)) {
            return ple_adapter_status::size_overflow;
        }
    }

    ple_compute_config compute_config = ple_compute_qwen4_exp_config();
    compute_config.max_speculative_tokens = config.max_speculative_tokens;
    ple_compute_footprint compute_footprint{};
    const ple_compute_status compute_status =
            ple_compute_get_footprint(compute_config, &compute_footprint);
    if (compute_status != ple_compute_status::ok) {
        return map_compute_status(compute_status);
    }
    result.device_state_bytes = compute_footprint.state_bytes;
    if (!checked_mul(kWorkspaceFloats, sizeof(float),
                     &result.device_workspace_bytes) ||
        !checked_mul(kEmbedding, sizeof(float),
                     &result.pinned_prefetch_bytes) ||
        !checked_mul(config.row_cache_slots,
                     kPleEmbeddingWidth * sizeof(float),
                     &result.row_cache_payload_bytes)) {
        return ple_adapter_status::size_overflow;
    }
    result.max_cold_nvme_bytes_per_token = kPleAdapterColdPrefetchBytes;
    result.host_to_device_bytes_per_token = kEmbedding * sizeof(float);
    *footprint = result;
    return ple_adapter_status::ok;
}

ple_layer_adapter::ple_layer_adapter() = default;
ple_layer_adapter::~ple_layer_adapter() = default;
ple_layer_adapter::ple_layer_adapter(ple_layer_adapter &&) noexcept = default;
ple_layer_adapter &ple_layer_adapter::operator=(
        ple_layer_adapter &&) noexcept = default;

ple_adapter_status ple_layer_adapter::load(
        const checkpoint_catalog &catalog,
        const ple_adapter_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<ple_layer_adapter> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(ple_adapter_status::invalid_argument, error,
                    "null PLE adapter output");
    }
    out->reset();
    const ple_adapter_status config_status =
            ple_adapter_validate_config(config);
    if (config_status != ple_adapter_status::ok) {
        return fail(config_status, error,
                    config_status == ple_adapter_status::unsupported_layer
                            ? "qwen4_exp PLE is admitted only on zero-based layer 1"
                            : "unsupported qwen4_exp PLE adapter configuration");
    }

    int device = -1;
    if (!current_supported_device(&device)) {
        return fail(ple_adapter_status::unsupported_device, error,
                    "qwen4_exp PLE requires a CUDA compute capability 12.x device");
    }

    std::size_t ple_tensor_count = 0u;
    for (std::size_t index = 0u; index < catalog.tensor_count(); ++index) {
        const tensor_span *span = catalog.at(index);
        if (span == nullptr) {
            return fail(ple_adapter_status::invalid_checkpoint, error,
                        "checkpoint catalog contains a null tensor span");
        }
        if (span->name.find(".ple.") == std::string::npos) continue;
        ++ple_tensor_count;
        if (!starts_with(span->name, kPlePrefix)) {
            return fail(ple_adapter_status::unsupported_layer, error,
                        "checkpoint contains PLE outside zero-based layer 1: " +
                                span->name);
        }
    }
    if (ple_tensor_count != kPleAdapterCheckpointTensors) {
        return fail(ple_adapter_status::invalid_checkpoint, error,
                    "unexpected PLE checkpoint tensor count: " +
                            std::to_string(ple_tensor_count));
    }

    std::unique_ptr<impl> loaded;
    try {
        loaded = std::make_unique<impl>();
    } catch (...) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not allocate PLE adapter state");
    }
    loaded->config = config;
    loaded->device = device;
    loaded->metadata.unigram_vocab_size = 248320;
    loaded->metadata.eos_token_id = 248044;

    ple_adapter_status status = read_i64_metadata(
            catalog, kMetadataContracts[0].suffix,
            &loaded->metadata.layer_multipliers, error);
    if (status != ple_adapter_status::ok) return status;
    status = read_i64_metadata(
            catalog, kMetadataContracts[1].suffix,
            &loaded->metadata.ngram_heads_offsets, error);
    if (status != ple_adapter_status::ok) return status;
    status = read_i64_metadata(
            catalog, kMetadataContracts[2].suffix,
            &loaded->metadata.ngram_heads_vocab_sizes, error);
    if (status != ple_adapter_status::ok) return status;

    const ple_status metadata_status =
            ple_validate_metadata(loaded->metadata);
    if (!metadata_status) {
        return fail(map_ple_status(metadata_status.code), error,
                    metadata_status.message);
    }
    std::array<std::int64_t, kPleNgramOrder> expected_multipliers{};
    std::array<std::int64_t, kPleHeadCount> expected_offsets{};
    std::array<std::int64_t, kPleHeadCount> expected_sizes{};
    if (!ple_derive_layer_multipliers(
                 248320, 0u, 1234u, &expected_multipliers) ||
        !ple_derive_head_layout(
                 20000000, 0u, &expected_offsets, &expected_sizes) ||
        expected_multipliers != loaded->metadata.layer_multipliers ||
        expected_offsets != loaded->metadata.ngram_heads_offsets ||
        expected_sizes != loaded->metadata.ngram_heads_vocab_sizes) {
        return fail(ple_adapter_status::metadata_mismatch, error,
                    "PLE metadata differs from the deterministic qwen4_exp layer-0 PLE contract");
    }

    const ple_embedding_status reader_status =
            loaded->embedding_reader.configure(&catalog);
    if (!reader_status) {
        return fail(map_embedding_status(reader_status.code), error,
                    reader_status.message);
    }

    for (std::size_t index = 0u; index < kWeightContracts.size(); ++index) {
        std::string name;
        try {
            name = std::string(kPlePrefix) + kWeightContracts[index].suffix;
        } catch (...) {
            return fail(ple_adapter_status::resource_exhausted, error,
                        "could not construct PLE weight name");
        }
        const tensor_span *span = nullptr;
        status = validate_tensor(catalog, name, kWeightContracts[index],
                                 &span, error);
        if (status != ple_adapter_status::ok) return status;
        status = load_bf16_weight(
                catalog, *span, config.upload_chunk_bytes,
                initialization_stream, &loaded->resident_weights[index], error);
        if (status != ple_adapter_status::ok) return status;
    }

    loaded->compute_weights.key_proj = loaded->resident_weights[0];
    loaded->compute_weights.value_proj = loaded->resident_weights[1];
    loaded->compute_weights.norm_key = loaded->resident_weights[2];
    loaded->compute_weights.norm_query = loaded->resident_weights[3];
    loaded->compute_weights.norm_conv = loaded->resident_weights[4];
    loaded->compute_weights.conv = loaded->resident_weights[5];
    status = ple_adapter_get_footprint(config, &loaded->footprint);
    if (status != ple_adapter_status::ok) {
        return fail(status, error, "could not derive PLE adapter footprint");
    }
    loaded->ready = true;

    std::unique_ptr<ple_layer_adapter> result;
    try {
        result = std::make_unique<ple_layer_adapter>();
    } catch (...) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not allocate PLE adapter owner");
    }
    result->impl_ = std::move(loaded);
    *out = std::move(result);
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_adapter::create_session(
        cudaStream_t initialization_stream,
        std::unique_ptr<ple_layer_session> *out,
        std::string *error) const noexcept {
    if (out == nullptr) {
        return fail(ple_adapter_status::invalid_argument, error,
                    "null PLE session output");
    }
    out->reset();
    if (impl_ == nullptr || !impl_->ready) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE adapter is not initialized");
    }
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != impl_->device) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE session must be created on the adapter device");
    }

    std::unique_ptr<ple_layer_session::impl> state;
    try {
        state = std::make_unique<ple_layer_session::impl>();
    } catch (...) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not allocate PLE session state");
    }
    state->owner = impl_.get();
    ple_status token_status = state->token_state.configure(
            impl_->metadata, 1u, impl_->config.max_speculative_tokens);
    if (!token_status) {
        return fail(map_ple_status(token_status.code), error,
                    token_status.message);
    }
    const ple_embedding_status cache_status =
            state->row_cache.configure(impl_->config.row_cache_slots);
    if (!cache_status) {
        return fail(map_embedding_status(cache_status.code), error,
                    cache_status.message);
    }

    ple_compute_config compute_config = ple_compute_qwen4_exp_config();
    compute_config.max_speculative_tokens =
            impl_->config.max_speculative_tokens;
    const ple_compute_status init_status = ple_compute_device_state_init(
            &state->compute_state, compute_config, initialization_stream);
    if (init_status != ple_compute_status::ok) {
        return fail(map_compute_status(init_status), error,
                    std::string("PLE compute state initialization failed: ") +
                            ple_compute_status_string(init_status));
    }

    if (cudaHostAlloc(reinterpret_cast<void **>(&state->host_embedding),
                      kEmbedding * sizeof(float), cudaHostAllocPortable) !=
                cudaSuccess ||
        cudaMalloc(reinterpret_cast<void **>(&state->device_workspace),
                   kWorkspaceFloats * sizeof(float)) != cudaSuccess) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "PLE session prefetch/workspace allocation failed");
    }
    float *cursor = state->device_workspace;
    state->device_embedding = cursor;
    cursor += kEmbedding;
    state->compute_scratch.key = cursor;
    cursor += kChannels;
    state->compute_scratch.value = cursor;
    cursor += kHidden;
    state->compute_scratch.query = cursor;
    cursor += kChannels;
    state->compute_scratch.gated = cursor;
    cursor += kChannels;
    state->compute_scratch.normalized = cursor;
    cursor += kChannels;
    state->device_ple_output = cursor;
    cursor += kChannels;
    if (cursor != state->device_workspace + kWorkspaceFloats ||
        cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
        return fail(ple_adapter_status::cuda_error, error,
                    "PLE session workspace initialization failed");
    }
    if (!state->io_worker.start()) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not start persistent PLE I/O worker");
    }
    state->ready = true;

    std::unique_ptr<ple_layer_session> result;
    try {
        result = std::make_unique<ple_layer_session>();
    } catch (...) {
        return fail(ple_adapter_status::resource_exhausted, error,
                    "could not allocate PLE session owner");
    }
    result->impl_ = std::move(state);
    *out = std::move(result);
    return ple_adapter_status::ok;
}

const ple_adapter_config &ple_layer_adapter::config() const noexcept {
    static const ple_adapter_config empty{};
    return impl_ == nullptr ? empty : impl_->config;
}

const ple_metadata &ple_layer_adapter::metadata() const noexcept {
    static const ple_metadata empty{};
    return impl_ == nullptr ? empty : impl_->metadata;
}

ple_adapter_footprint ple_layer_adapter::footprint() const noexcept {
    return impl_ == nullptr ? ple_adapter_footprint{} : impl_->footprint;
}

float ple_layer_adapter::embedding_scale() const noexcept {
    return impl_ == nullptr ? 0.0F : impl_->embedding_reader.scale();
}

std::size_t ple_layer_adapter::layer_index() const noexcept {
    return impl_ == nullptr ? std::numeric_limits<std::size_t>::max()
                            : impl_->config.layer_index;
}

int ple_layer_adapter::device() const noexcept {
    return impl_ == nullptr ? -1 : impl_->device;
}

bool ple_layer_adapter::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

ple_layer_session::ple_layer_session() = default;
ple_layer_session::~ple_layer_session() = default;
ple_layer_session::ple_layer_session(ple_layer_session &&) noexcept = default;
ple_layer_session &ple_layer_session::operator=(
        ple_layer_session &&) noexcept = default;

ple_adapter_status ple_layer_session::begin_transaction(
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE session is not ready");
    }
    if (impl_->token_state.transaction_open() ||
        impl_->compute_state.transaction_open || impl_->prefetched || impl_->io_pending) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE transaction is already open");
    }
    const ple_status token_status = impl_->token_state.begin_transaction();
    if (!token_status) {
        return fail(map_ple_status(token_status.code), error,
                    token_status.message);
    }
    const ple_compute_status compute_status =
            ple_compute_begin_transaction(&impl_->compute_state);
    if (compute_status != ple_compute_status::ok) {
        (void)impl_->token_state.rollback();
        return fail(map_compute_status(compute_status), error,
                    "PLE compute transaction begin failed");
    }
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::prefetch_token(
        std::int64_t token_id,
        cudaStream_t stream,
        std::string *error) noexcept {
    const auto status = start_prefetch_token(token_id, error);
    return status == ple_adapter_status::ok ? finish_prefetch_token(stream, error) : status;
}

ple_adapter_status ple_layer_session::start_prefetch_token(
        std::int64_t token_id,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        impl_->owner == nullptr ||
        !impl_->token_state.transaction_open() ||
        !impl_->compute_state.transaction_open || impl_->prefetched || impl_->io_pending ||
        impl_->token_state.staged_tokens() !=
                impl_->compute_state.staged_count) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE token prefetch requires a balanced open transaction");
    }

    ple_head_ids ids{};
    ple_status token_status = impl_->token_state.stage_token(token_id, &ids);
    if (!token_status) {
        return abort_open_transaction(
                impl_.get(), error, token_status.message,
                map_ple_status(token_status.code));
    }
    const float marker = 0.0F;
    std::array<float, kPleConvKernelSize> ignored_taps{};
    token_status = impl_->token_state.stage_conv_frame(
            &marker, 1u, ignored_taps.data(), ignored_taps.size());
    if (!token_status) {
        return abort_open_transaction(
                impl_.get(), error, token_status.message,
                map_ple_status(token_status.code));
    }

    if (!impl_->io_worker.submit(&impl::gather, impl_.get(), ids)) {
        return abort_open_transaction(
                impl_.get(), error, "PLE I/O worker rejected submission",
                ple_adapter_status::invalid_state);
    }
    impl_->io_pending = true;
    impl_->prefetched_ids = ids;
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::finish_prefetch_token(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        !impl_->io_pending || impl_->prefetched ||
        !impl_->token_state.transaction_open() ||
        !impl_->compute_state.transaction_open) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE finish requires one pending I/O job");
    }
    auto result = impl_->drain_io();
    if (result.code != ple_adapter_status::ok) {
        return abort_open_transaction(
                impl_.get(), error,
                result.message.empty() ? "PLE asynchronous gather failed" : result.message,
                result.code);
    }

    if (cudaMemcpyAsync(impl_->device_embedding, impl_->host_embedding,
                        kEmbedding * sizeof(float), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return abort_open_transaction(
                impl_.get(), error, "PLE prefetched embedding upload failed",
                ple_adapter_status::cuda_error);
    }
    impl_->prefetched = true;
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::stage_prefetched_and_add_cuda(
        const float *hidden_before_ple,
        float *hidden_after_ple,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        impl_->owner == nullptr || !impl_->prefetched ||
        !impl_->token_state.transaction_open() ||
        !impl_->compute_state.transaction_open ||
        impl_->token_state.staged_tokens() !=
                impl_->compute_state.staged_count + 1u) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE CUDA stage requires one prefetched token");
    }
    constexpr std::size_t kHiddenBytes = kChannels * sizeof(float);
    const device_range input{hidden_before_ple, kHiddenBytes};
    const device_range output{hidden_after_ple, kHiddenBytes};
    const device_range internal{
            impl_->device_workspace, kWorkspaceFloats * sizeof(float)};
    if (!valid_device_range(input, impl_->owner->device) ||
        !valid_device_range(output, impl_->owner->device)) {
        return abort_open_transaction(
                impl_.get(), error, "PLE input/output is not a valid device range",
                ple_adapter_status::invalid_device_pointer);
    }
    if ((hidden_before_ple != hidden_after_ple &&
         ranges_overlap(input, output)) ||
        ranges_overlap(input, internal) || ranges_overlap(output, internal)) {
        return abort_open_transaction(
                impl_.get(), error, "PLE input/output overlaps adapter scratch",
                ple_adapter_status::invalid_argument);
    }

    const ple_compute_status compute_status = ple_compute_stage_token_cuda(
            &impl_->compute_state, impl_->owner->compute_weights,
            impl_->device_embedding, hidden_before_ple,
            impl_->compute_scratch, impl_->device_ple_output, stream);
    if (compute_status != ple_compute_status::ok) {
        return abort_open_transaction(
                impl_.get(), error,
                std::string("PLE compute stage failed: ") +
                        ple_compute_status_string(compute_status),
                map_compute_status(compute_status));
    }

    const unsigned blocks = static_cast<unsigned>(
            (kChannels + kThreads - 1u) / kThreads);
    add_ple_before_attention_hyper_connection_kernel<<<
            blocks, kThreads, 0u, stream>>>(
                    hidden_before_ple, impl_->device_ple_output,
                    hidden_after_ple, kChannels);
    if (cudaGetLastError() != cudaSuccess) {
        return abort_open_transaction(
                impl_.get(), error,
                "hidden += PLE launch failed before attn_hyper_connection",
                ple_adapter_status::cuda_error);
    }
    impl_->prefetched = false;
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::commit_prefix(
        std::size_t accepted_tokens,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->poisoned ||
        impl_->prefetched || impl_->io_pending || !impl_->token_state.transaction_open() ||
        !impl_->compute_state.transaction_open ||
        impl_->token_state.staged_tokens() !=
                impl_->compute_state.staged_count ||
        accepted_tokens > impl_->compute_state.staged_count) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE commit requires a balanced complete transaction");
    }

    const ple_compute_status compute_status =
            ple_compute_commit_prefix_cuda(
                    &impl_->compute_state, accepted_tokens, stream);
    if (compute_status != ple_compute_status::ok) {
        return abort_open_transaction(
                impl_.get(), error, "PLE compute commit failed",
                map_compute_status(compute_status));
    }
    const ple_status token_status =
            impl_->token_state.commit_prefix(accepted_tokens);
    if (!token_status) {
        impl_->poisoned = true;
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE token commit diverged after device commit; session is poisoned");
    }
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::rollback(
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready ||
        (!impl_->token_state.transaction_open() &&
         !impl_->compute_state.transaction_open)) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE rollback requires an open transaction");
    }
    (void)impl_->drain_io(); // A cancelled gather is not an answer/state update.
    bool ok = true;
    if (impl_->compute_state.transaction_open) {
        ok = ple_compute_rollback(&impl_->compute_state) ==
                     ple_compute_status::ok &&
             ok;
    }
    if (impl_->token_state.transaction_open()) {
        ok = impl_->token_state.rollback().ok() && ok;
    }
    impl_->prefetched = false;
    if (!ok) {
        impl_->poisoned = true;
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE rollback failed; session is poisoned");
    }
    return ple_adapter_status::ok;
}

ple_adapter_status ple_layer_session::reset(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready) {
        return fail(ple_adapter_status::invalid_state, error,
                    "PLE reset requires an initialized session");
    }
    (void)impl_->drain_io();
    if (impl_->compute_state.transaction_open) {
        (void)ple_compute_rollback(&impl_->compute_state);
    }
    if (impl_->token_state.transaction_open()) {
        (void)impl_->token_state.rollback();
    }
    const ple_status token_status = impl_->token_state.reset();
    if (!token_status) {
        impl_->poisoned = true;
        return fail(map_ple_status(token_status.code), error,
                    token_status.message);
    }
    const ple_compute_status compute_status =
            ple_compute_device_state_reset(&impl_->compute_state, stream);
    if (compute_status != ple_compute_status::ok) {
        impl_->poisoned = true;
        return fail(map_compute_status(compute_status), error,
                    "PLE compute reset failed");
    }
    impl_->row_cache.clear();
    impl_->visible_hits = impl_->visible_misses = 0u;
    impl_->prefetched_ids.fill(0);
    impl_->nvme_bytes = 0u;
    impl_->prefetched = false;
    impl_->poisoned = false;
    return ple_adapter_status::ok;
}

const ple_head_ids &ple_layer_session::prefetched_head_ids() const noexcept {
    static const ple_head_ids empty{};
    return impl_ == nullptr ? empty : impl_->prefetched_ids;
}

std::size_t ple_layer_session::staged_tokens() const noexcept {
    return impl_ == nullptr ? 0u : impl_->token_state.staged_tokens();
}

std::uint64_t ple_layer_session::committed_tokens() const noexcept {
    return impl_ == nullptr ? 0u : impl_->token_state.committed_tokens();
}

std::uint64_t ple_layer_session::cache_hits() const noexcept {
    return impl_ == nullptr ? 0u : impl_->visible_hits;
}

std::uint64_t ple_layer_session::cache_misses() const noexcept {
    return impl_ == nullptr ? 0u : impl_->visible_misses;
}

std::uint64_t ple_layer_session::nvme_bytes_read() const noexcept {
    return impl_ == nullptr ? 0u : impl_->nvme_bytes;
}

bool ple_layer_session::transaction_open() const noexcept {
    return impl_ != nullptr && impl_->token_state.transaction_open() &&
           impl_->compute_state.transaction_open;
}

bool ple_layer_session::token_prefetched() const noexcept {
    return impl_ != nullptr && impl_->prefetched;
}

bool ple_layer_session::prefetch_pending() const noexcept {
    return impl_ != nullptr && impl_->io_pending;
}

bool ple_layer_session::poisoned() const noexcept {
    return impl_ == nullptr || impl_->poisoned;
}

bool ple_layer_session::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

}  // namespace axiom::qwen4exp
#endif // AXIOM_QWEN4EXP_PLE_WORKER_TEST_ONLY
