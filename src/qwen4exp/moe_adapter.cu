#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "axiom/qwen4exp/moe_adapter.hpp"

#include "axiom/qwen4exp/bf16_linear.hpp"
#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/expert_bridge.hpp"
#include "axiom/qwen4exp/moe.hpp"
#include "axiom/qwen4exp/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kRouterLogits = kMoeExperts;
constexpr std::size_t kSelectedExperts = kMoeTopK;

class spin_mutex final {
public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
        }
    }
    void unlock() noexcept { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class scoped_spin_lock final {
public:
    explicit scoped_spin_lock(spin_mutex &mutex) noexcept : mutex_(mutex) {
        mutex_.lock();
    }
    ~scoped_spin_lock() { mutex_.unlock(); }
    scoped_spin_lock(const scoped_spin_lock &) = delete;
    scoped_spin_lock &operator=(const scoped_spin_lock &) = delete;

private:
    spin_mutex &mutex_;
};

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) {
        *error = "qwen4_exp routed-MoE adapter: " + message;
    }
}

moe_adapter_status fail(moe_adapter_status status,
                        std::string *error,
                        const std::string &message) {
    set_error(error, message);
    return status;
}

std::string cuda_message(const char *operation, cudaError_t status) {
    return std::string(operation) + ": " + cudaGetErrorString(status);
}

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t *result) noexcept {
    if (result == nullptr ||
        (right != 0u && left > std::numeric_limits<std::size_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t *result) noexcept {
    if (result == nullptr || left > std::numeric_limits<std::uint64_t>::max() - right) {
        return false;
    }
    *result = left + right;
    return true;
}

bool bf16_finite(std::uint16_t value) noexcept {
    return (value & 0x7f80u) != 0x7f80u;
}

bool pointer_visible_to_device(const void *pointer, int device) noexcept {
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attributes.type == cudaMemoryTypeManaged) return true;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
#else
    return attributes.memoryType == cudaMemoryTypeDevice && attributes.device == device;
#endif
}

bool ranges_overlap(const void *left,
                    std::size_t left_bytes,
                    const void *right,
                    std::size_t right_bytes) noexcept {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
        right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
        return true;
    }
    const auto left_end = left_begin + left_bytes;
    const auto right_end = right_begin + right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

moe_adapter_status map_bf16_status(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return moe_adapter_status::ok;
        case bf16_linear_status::invalid_argument:
            return moe_adapter_status::invalid_argument;
        case bf16_linear_status::unsupported_config:
        case bf16_linear_status::shape_mismatch:
        case bf16_linear_status::dtype_mismatch:
        case bf16_linear_status::size_overflow:
            return moe_adapter_status::unsupported_config;
        case bf16_linear_status::unsupported_device:
            return moe_adapter_status::unsupported_device;
        case bf16_linear_status::invalid_device_pointer:
            return moe_adapter_status::invalid_device_pointer;
        case bf16_linear_status::tensor_not_found:
        case bf16_linear_status::checkpoint_io_error:
            return moe_adapter_status::checkpoint_error;
        case bf16_linear_status::allocation_failure:
            return moe_adapter_status::allocation_failure;
        case bf16_linear_status::cuda_error:
        case bf16_linear_status::cublas_error:
            return moe_adapter_status::cuda_error;
    }
    return moe_adapter_status::kernel_error;
}

moe_adapter_status map_moe_status(moe_status status) noexcept {
    switch (status) {
        case moe_status::ok: return moe_adapter_status::ok;
        case moe_status::invalid_argument:
            return moe_adapter_status::invalid_argument;
        case moe_status::unsupported_config:
            return moe_adapter_status::unsupported_config;
        case moe_status::non_finite:
            return moe_adapter_status::non_finite;
        case moe_status::cuda_error:
            return moe_adapter_status::cuda_error;
        case moe_status::kernel_error:
            return moe_adapter_status::kernel_error;
    }
    return moe_adapter_status::kernel_error;
}

moe_adapter_status map_shared_status(shared_expert_status status) noexcept {
    switch (status) {
        case shared_expert_status::ok: return moe_adapter_status::ok;
        case shared_expert_status::invalid_argument:
            return moe_adapter_status::invalid_argument;
        case shared_expert_status::unsupported_config:
        case shared_expert_status::overflow:
            return moe_adapter_status::unsupported_config;
        case shared_expert_status::unsupported_device:
            return moe_adapter_status::unsupported_device;
        case shared_expert_status::invalid_device_pointer:
            return moe_adapter_status::invalid_device_pointer;
        case shared_expert_status::non_finite:
            return moe_adapter_status::non_finite;
        case shared_expert_status::cuda_error:
            return moe_adapter_status::cuda_error;
    }
    return moe_adapter_status::kernel_error;
}

struct route_host_snapshot {
    std::array<std::uint32_t, kSelectedExperts> indices{};
    std::array<float, kSelectedExperts> weights{};
    std::uint32_t status = static_cast<std::uint32_t>(moe_status::ok);
};

__global__ void validate_moe_outputs_kernel(
        const float *routed,
        const float *shared,
        const std::uint32_t *shared_status,
        std::uint32_t *adapter_status,
        std::size_t elements) {
    if (blockIdx.x == 0u && threadIdx.x == 0u &&
        *shared_status != static_cast<std::uint32_t>(shared_expert_status::ok)) {
        atomicCAS(adapter_status,
                  static_cast<std::uint32_t>(moe_adapter_status::ok),
                  static_cast<std::uint32_t>(moe_adapter_status::non_finite));
    }
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        if (!isfinite(routed[index]) || !isfinite(shared[index])) {
            atomicCAS(adapter_status,
                      static_cast<std::uint32_t>(moe_adapter_status::ok),
                      static_cast<std::uint32_t>(moe_adapter_status::non_finite));
        }
    }
}

__global__ void combine_moe_outputs_kernel(
        const float *routed,
        const float *shared,
        const std::uint32_t *adapter_status,
        float *output,
        std::size_t elements) {
    const bool valid = *adapter_status ==
                       static_cast<std::uint32_t>(moe_adapter_status::ok);
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                             threadIdx.x;
         index < elements;
         index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        output[index] = valid ? routed[index] + shared[index] : 0.0f;
    }
}

bool launch_ok() noexcept {
    return cudaGetLastError() == cudaSuccess;
}

std::string router_tensor_name(std::uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) +
           ".mlp.gate.weight";
}

bool shared_tensor_name_checked(std::uint32_t layer,
                                shared_expert_tensor tensor,
                                std::string *name,
                                std::string *error) {
    if (name == nullptr) return false;
    std::array<char, 160> buffer{};
    const shared_expert_status status = shared_expert_tensor_name(
            layer, tensor, buffer.data(), buffer.size());
    if (status != shared_expert_status::ok) {
        set_error(error, std::string("cannot derive shared expert tensor name: ") +
                         shared_expert_status_string(status));
        return false;
    }
    *name = buffer.data();
    return true;
}

bool load_shared_tensor(const checkpoint_catalog &catalog,
                        const std::string &name,
                        std::uint64_t rows,
                        std::uint64_t columns,
                        cudaStream_t stream,
                        const std::uint16_t **device_pointer,
                        std::uint64_t *resident_bytes,
                        std::string *error) {
    if (device_pointer == nullptr || resident_bytes == nullptr) return false;
    *device_pointer = nullptr;
    const tensor_span *span = catalog.find(name);
    std::uint64_t elements = 0u;
    if (rows != 0u && columns > std::numeric_limits<std::uint64_t>::max() / rows) {
        set_error(error, "shared expert tensor shape overflows: " + name);
        return false;
    }
    elements = rows * columns;
    if (elements > std::numeric_limits<std::uint64_t>::max() /
                           sizeof(std::uint16_t)) {
        set_error(error, "shared expert tensor byte count overflows: " + name);
        return false;
    }
    const std::uint64_t bytes = elements * sizeof(std::uint16_t);
    if (span == nullptr || span->dtype != tensor_dtype::bf16 || span->rank != 2u ||
        span->shape[0] != rows || span->shape[1] != columns || span->bytes != bytes ||
        bytes > std::numeric_limits<std::size_t>::max()) {
        set_error(error, "shared expert tensor contract mismatch: " + name);
        return false;
    }

    try {
        std::vector<std::uint16_t> host(static_cast<std::size_t>(elements));
        std::string read_error;
        if (!catalog.read_range(name, 0u, host.data(),
                                static_cast<std::size_t>(bytes), &read_error)) {
            set_error(error, read_error.empty()
                                     ? "shared expert checkpoint read failed: " + name
                                     : read_error);
            return false;
        }
        if (!std::all_of(host.begin(), host.end(), bf16_finite)) {
            set_error(error, "shared expert contains non-finite BF16 data: " + name);
            return false;
        }
        std::uint16_t *allocated = nullptr;
        cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&allocated),
                                        static_cast<std::size_t>(bytes));
        if (status != cudaSuccess) {
            set_error(error, cuda_message("cudaMalloc(shared expert)", status));
            return false;
        }
        status = cudaMemcpyAsync(allocated, host.data(),
                                 static_cast<std::size_t>(bytes),
                                 cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            (void)cudaFree(allocated);
            set_error(error, cuda_message("cudaMemcpyAsync(shared expert)", status));
            return false;
        }
        status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            (void)cudaFree(allocated);
            set_error(error, cuda_message("cudaStreamSynchronize(shared expert)", status));
            return false;
        }
        if (!checked_add(*resident_bytes, bytes, resident_bytes)) {
            (void)cudaFree(allocated);
            set_error(error, "shared expert resident-byte accounting overflow");
            return false;
        }
        *device_pointer = allocated;
        return true;
    } catch (const std::bad_alloc &) {
        set_error(error, "host staging allocation failed for shared expert: " + name);
        return false;
    } catch (...) {
        set_error(error, "unexpected shared expert load failure: " + name);
        return false;
    }
}

}  // namespace

struct routed_moe_layer_adapter::impl {
    moe_adapter_options options{};
    moe_config config = moe_qwen4_exp_config();
    shared_expert_config shared_config = shared_expert_qwen4_exp_config();
    std::unique_ptr<resident_bf16_linear> router;
    std::unique_ptr<expert_bridge> bridge;

    shared_expert_weights_bf16 shared_weights{};
    bf16_linear_scratch router_scratch{};
    std::uint16_t *router_input_bf16 = nullptr;
    void *router_blas_workspace = nullptr;
    float *router_logits = nullptr;
    std::uint32_t *router_indices = nullptr;
    float *router_weights = nullptr;
    std::uint32_t *router_status = nullptr;
    float *routed_mid = nullptr;
    float *routed_output = nullptr;
    float *shared_intermediate = nullptr;
    float *shared_gate = nullptr;
    std::uint32_t *shared_status = nullptr;
    float *shared_output = nullptr;
    std::uint32_t *forward_status = nullptr;

    route_host_snapshot *host_route = nullptr;
    std::uint32_t *host_forward_status = nullptr;
    cudaEvent_t route_event = nullptr;
    cudaEvent_t forward_event = nullptr;
    cudaStream_t route_stream = nullptr;

    std::uint64_t resident_gpu_bytes = 0u;
    mutable spin_mutex mutex;
    bool route_in_flight = false;
    bool generation_active = false;
    bool initialized = false;

    ~impl() {
        if (options.device >= 0 && cudaSetDevice(options.device) == cudaSuccess) {
            if (route_in_flight && route_event != nullptr) {
                (void)cudaEventSynchronize(route_event);
            }
            (void)cudaEventDestroy(forward_event);
            (void)cudaEventDestroy(route_event);
            (void)cudaFreeHost(host_forward_status);
            (void)cudaFreeHost(host_route);
            (void)cudaFree(forward_status);
            (void)cudaFree(shared_output);
            (void)cudaFree(shared_status);
            (void)cudaFree(shared_gate);
            (void)cudaFree(shared_intermediate);
            (void)cudaFree(routed_output);
            (void)cudaFree(routed_mid);
            (void)cudaFree(router_status);
            (void)cudaFree(router_weights);
            (void)cudaFree(router_indices);
            (void)cudaFree(router_logits);
            (void)cudaFree(router_blas_workspace);
            (void)cudaFree(router_input_bf16);
            (void)cudaFree(const_cast<std::uint16_t *>(shared_weights.output_gate));
            (void)cudaFree(const_cast<std::uint16_t *>(shared_weights.down));
            (void)cudaFree(const_cast<std::uint16_t *>(shared_weights.up));
            (void)cudaFree(const_cast<std::uint16_t *>(shared_weights.gate));
        }
    }

    void release_generation() noexcept {
        scoped_spin_lock lock(mutex);
        generation_active = false;
    }
};

struct routed_moe_layer_adapter::prepared_generation::state {
    std::shared_ptr<routed_moe_layer_adapter::impl> owner;
    expert_bridge::prepared_generation bridge_generation;
    std::array<std::uint32_t, kSelectedExperts> experts{};
    std::array<float, kSelectedExperts> weights{};
    cudaStream_t stream = nullptr;
    bool forward_is_enqueued = false;
    bool arena_is_handed_off = false;
    bool terminal = false;
    bool is_committed = false;
    moe_adapter_status completed_status = moe_adapter_status::ok;

    bool rollback(std::string *error) noexcept {
        if (terminal) {
            return !is_committed;
        }
        if (forward_is_enqueued && owner != nullptr && owner->forward_event != nullptr) {
            const cudaError_t status = cudaEventSynchronize(owner->forward_event);
            if (status != cudaSuccess) {
                set_error(error, cuda_message("cudaEventSynchronize(forward rollback)",
                                              status));
                return false;
            }
        }
        bool rolled_back = true;
        if (bridge_generation.valid() && !bridge_generation.committed()) {
            rolled_back = bridge_generation.rollback(error);
        }
        bridge_generation = expert_bridge::prepared_generation{};
        if (owner != nullptr) owner->release_generation();
        terminal = true;
        is_committed = false;
        owner.reset();
        return rolled_back;
    }

    void rollback_noexcept() noexcept {
        std::string ignored;
        (void)rollback(&ignored);
    }

    ~state() { rollback_noexcept(); }
};

const char *moe_adapter_status_string(moe_adapter_status status) noexcept {
    switch (status) {
        case moe_adapter_status::ok: return "ok";
        case moe_adapter_status::invalid_argument: return "invalid_argument";
        case moe_adapter_status::unsupported_config: return "unsupported_config";
        case moe_adapter_status::unsupported_device: return "unsupported_device";
        case moe_adapter_status::invalid_device_pointer: return "invalid_device_pointer";
        case moe_adapter_status::checkpoint_error: return "checkpoint_error";
        case moe_adapter_status::allocation_failure: return "allocation_failure";
        case moe_adapter_status::router_error: return "router_error";
        case moe_adapter_status::pager_error: return "pager_error";
        case moe_adapter_status::non_finite: return "non_finite";
        case moe_adapter_status::busy: return "busy";
        case moe_adapter_status::cuda_error: return "cuda_error";
        case moe_adapter_status::kernel_error: return "kernel_error";
        case moe_adapter_status::transaction_error: return "transaction_error";
    }
    return "unknown";
}

routed_moe_layer_adapter::prepared_generation::prepared_generation() noexcept = default;
routed_moe_layer_adapter::prepared_generation::~prepared_generation() = default;
routed_moe_layer_adapter::prepared_generation::prepared_generation(
        prepared_generation &&) noexcept = default;
routed_moe_layer_adapter::prepared_generation &
routed_moe_layer_adapter::prepared_generation::operator=(
        prepared_generation &&) noexcept = default;
routed_moe_layer_adapter::prepared_generation::prepared_generation(
        std::unique_ptr<state> state_value) noexcept
    : state_(std::move(state_value)) {}

bool routed_moe_layer_adapter::prepared_generation::valid() const noexcept {
    return state_ != nullptr && !state_->terminal && state_->owner != nullptr &&
           state_->bridge_generation.valid();
}

bool routed_moe_layer_adapter::prepared_generation::committed() const noexcept {
    return state_ != nullptr && state_->terminal && state_->is_committed;
}

bool routed_moe_layer_adapter::prepared_generation::forward_enqueued() const noexcept {
    return valid() && state_->forward_is_enqueued;
}

bool routed_moe_layer_adapter::prepared_generation::arena_handed_off() const noexcept {
    return valid() && state_->forward_is_enqueued &&
           state_->arena_is_handed_off;
}

std::uint64_t routed_moe_layer_adapter::prepared_generation::generation_id() const noexcept {
    return valid() ? state_->bridge_generation.generation_id() : 0u;
}

const std::uint32_t *
routed_moe_layer_adapter::prepared_generation::selected_experts() const noexcept {
    return valid() ? state_->experts.data() : nullptr;
}

const float *
routed_moe_layer_adapter::prepared_generation::router_weights() const noexcept {
    return valid() ? state_->weights.data() : nullptr;
}

std::uint32_t
routed_moe_layer_adapter::prepared_generation::selected_count() const noexcept {
    return valid() ? kMoeTopK : 0u;
}

moe_adapter_status
routed_moe_layer_adapter::prepared_generation::device_status() const noexcept {
    return state_ != nullptr ? state_->completed_status
                             : moe_adapter_status::transaction_error;
}

moe_adapter_status routed_moe_layer_adapter::prepared_generation::forward(
        const float *input,
        float *output,
        cudaStream_t stream_value,
        std::string *error) noexcept {
    if (!valid() || input == nullptr || output == nullptr || input == output) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "invalid prepared generation or forward pointers");
    }
    if (state_->forward_is_enqueued) {
        return fail(moe_adapter_status::busy, error,
                    "forward was already enqueued for this generation");
    }
    if (stream_value != state_->stream) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "forward stream differs from the route/pager stream");
    }
    const std::shared_ptr<routed_moe_layer_adapter::impl> owner = state_->owner;
    const std::size_t vector_bytes = static_cast<std::size_t>(kMoeHidden) * sizeof(float);
    if (!pointer_visible_to_device(input, owner->options.device) ||
        !pointer_visible_to_device(output, owner->options.device) ||
        ranges_overlap(input, vector_bytes, output, vector_bytes)) {
        return fail(moe_adapter_status::invalid_device_pointer, error,
                    "input/output are not distinct visible device ranges");
    }

    auto synchronize_failure = [&](moe_adapter_status status,
                                   const std::string &message) {
        (void)cudaStreamSynchronize(stream_value);
        return fail(status, error, message);
    };

    cudaError_t cuda_status = cudaMemsetAsync(owner->forward_status, 0,
                                               sizeof(*owner->forward_status),
                                               stream_value);
    if (cuda_status != cudaSuccess) {
        return synchronize_failure(moe_adapter_status::cuda_error,
                                   cuda_message("cudaMemsetAsync(forward status)",
                                                cuda_status));
    }

    const moe_status routed_status = moe_forward_slots_f32_cuda(
            owner->options.device, owner->config,
            state_->bridge_generation.physical_slot_capacity(),
            state_->bridge_generation.planes(),
            state_->bridge_generation.slot_indices_device(),
            owner->router_weights, input, owner->routed_mid,
            owner->routed_output, stream_value);
    if (routed_status != moe_status::ok) {
        return synchronize_failure(map_moe_status(routed_status),
                                   std::string("routed expert enqueue failed: ") +
                                           moe_status_string(routed_status));
    }

    const shared_expert_workspace_f32 shared_workspace{
        owner->shared_intermediate, owner->shared_gate, owner->shared_status};
    const shared_expert_status shared_status = shared_expert_forward_f32_cuda(
            owner->options.device, owner->shared_config, owner->shared_weights,
            input, shared_workspace, owner->shared_output, stream_value);
    if (shared_status != shared_expert_status::ok) {
        return synchronize_failure(map_shared_status(shared_status),
                                   std::string("shared expert enqueue failed: ") +
                                           shared_expert_status_string(shared_status));
    }

    constexpr unsigned threads = 256u;
    constexpr unsigned blocks = 10u;
    validate_moe_outputs_kernel<<<blocks, threads, 0u, stream_value>>>(
            owner->routed_output, owner->shared_output, owner->shared_status,
            owner->forward_status, kMoeHidden);
    if (!launch_ok()) {
        return synchronize_failure(moe_adapter_status::cuda_error,
                                   "validate-MoE-output kernel launch failed");
    }
    combine_moe_outputs_kernel<<<blocks, threads, 0u, stream_value>>>(
            owner->routed_output, owner->shared_output, owner->forward_status,
            output, kMoeHidden);
    if (!launch_ok()) {
        return synchronize_failure(moe_adapter_status::cuda_error,
                                   "combine-MoE-output kernel launch failed");
    }
    cuda_status = cudaMemcpyAsync(owner->host_forward_status, owner->forward_status,
                                  sizeof(*owner->host_forward_status),
                                  cudaMemcpyDeviceToHost, stream_value);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaEventRecord(owner->forward_event, stream_value);
    }
    if (cuda_status != cudaSuccess) {
        return synchronize_failure(moe_adapter_status::cuda_error,
                                   cuda_message("record forward completion", cuda_status));
    }
    if (!state_->bridge_generation.handoff_after_consumers(stream_value,
                                                            error)) {
        return synchronize_failure(
                moe_adapter_status::transaction_error,
                error != nullptr && !error->empty()
                        ? *error
                        : "stream-ordered expert arena handoff failed");
    }
    state_->arena_is_handed_off = true;
    state_->forward_is_enqueued = true;
    return moe_adapter_status::ok;
}

bool routed_moe_layer_adapter::prepared_generation::commit(
        std::string *error) noexcept {
    if (!valid() || !state_->forward_is_enqueued) {
        set_error(error, "cannot commit before a valid forward is enqueued");
        return false;
    }
    const cudaError_t event_status = cudaEventSynchronize(state_->owner->forward_event);
    if (event_status != cudaSuccess) {
        set_error(error, cuda_message("cudaEventSynchronize(forward commit)",
                                      event_status));
        return false;
    }
    const auto result = static_cast<moe_adapter_status>(
            *state_->owner->host_forward_status);
    state_->completed_status = result;
    if (result != moe_adapter_status::ok) {
        std::string rollback_error;
        const bool rolled_back = state_->rollback(&rollback_error);
        set_error(error, std::string("device rejected routed-MoE generation: ") +
                         moe_adapter_status_string(result) +
                         (rolled_back || rollback_error.empty()
                                  ? std::string{}
                                  : "; rollback: " + rollback_error));
        return false;
    }
    if (!state_->bridge_generation.commit(error)) {
        state_->completed_status = moe_adapter_status::transaction_error;
        state_->bridge_generation = expert_bridge::prepared_generation{};
        state_->owner->release_generation();
        state_->terminal = true;
        state_->is_committed = false;
        state_->owner.reset();
        return false;
    }
    state_->bridge_generation = expert_bridge::prepared_generation{};
    state_->owner->release_generation();
    state_->terminal = true;
    state_->is_committed = true;
    state_->owner.reset();
    return true;
}

bool routed_moe_layer_adapter::prepared_generation::rollback(
        std::string *error) noexcept {
    if (state_ == nullptr) {
        set_error(error, "cannot roll back an empty generation");
        return false;
    }
    return state_->rollback(error);
}

routed_moe_layer_adapter::routed_moe_layer_adapter() = default;
routed_moe_layer_adapter::~routed_moe_layer_adapter() = default;
routed_moe_layer_adapter::routed_moe_layer_adapter(
        routed_moe_layer_adapter &&) noexcept = default;
routed_moe_layer_adapter &routed_moe_layer_adapter::operator=(
        routed_moe_layer_adapter &&) noexcept = default;

moe_adapter_status routed_moe_layer_adapter::load(
        const std::string &model_root,
        const moe_adapter_options &options,
        cudaStream_t initialization_stream,
        std::unique_ptr<routed_moe_layer_adapter> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "adapter output is null");
    }
    out->reset();
    if (model_root.empty() || options.device < 0 ||
        options.layer >= kExpertBridgeLayers || options.prefetch_workers == 0u ||
        options.blas_workspace_bytes == 0u ||
        options.blas_workspace_bytes % 256u != 0u) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "invalid model root, device, layer, workers or BLAS workspace");
    }

    try {
        int device_count = 0;
        int current_device = -1;
        cudaDeviceProp properties{};
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            options.device >= device_count ||
            cudaSetDevice(options.device) != cudaSuccess ||
            cudaGetDevice(&current_device) != cudaSuccess ||
            current_device != options.device ||
            cudaGetDeviceProperties(&properties, options.device) != cudaSuccess ||
            properties.major * 10 + properties.minor < 120) {
            return fail(moe_adapter_status::unsupported_device, error,
                        "qwen4_exp routed-MoE requires an available SM120 device");
        }

        auto result = std::make_unique<routed_moe_layer_adapter>();
        result->impl_ = std::make_shared<impl>();
        result->impl_->options = options;
        impl &state = *result->impl_;
        if (!moe_is_qwen4_exp_contract(state.config) ||
            !shared_expert_is_qwen4_exp_contract(state.shared_config)) {
            return fail(moe_adapter_status::unsupported_config, error,
                        "compiled MoE/shared-expert contracts are not qwen4_exp exact");
        }

        std::unique_ptr<checkpoint_catalog> owned_checkpoint;
        const checkpoint_catalog *checkpoint =
                options.shared_checkpoint_catalog;
        std::string checkpoint_error;
        if (checkpoint == nullptr) {
            if (!checkpoint_catalog::open(model_root, &owned_checkpoint,
                                          &checkpoint_error) ||
                owned_checkpoint == nullptr) {
                return fail(moe_adapter_status::checkpoint_error, error,
                            checkpoint_error.empty()
                                    ? "cannot open pinned checkpoint"
                                    : checkpoint_error);
            }
            checkpoint = owned_checkpoint.get();
        }

        const bf16_linear_config router_config{
            kMoeHidden, kMoeExperts, 1u, 8u * 1024u * 1024u,
            options.blas_workspace_bytes};
        const std::string router_name = router_tensor_name(options.layer);
        std::string load_error;
        const bf16_linear_status router_load = resident_bf16_linear::load(
                *checkpoint, router_name, router_config, initialization_stream,
                &state.router, &load_error);
        if (router_load != bf16_linear_status::ok) {
            return fail(map_bf16_status(router_load), error,
                        load_error.empty()
                                ? std::string("router load failed: ") +
                                          bf16_linear_status_string(router_load)
                                : load_error);
        }
        state.resident_gpu_bytes = state.router->weight_bytes();

        constexpr std::array<shared_expert_tensor, 4> shared_tensors{
            shared_expert_tensor::gate_projection,
            shared_expert_tensor::up_projection,
            shared_expert_tensor::down_projection,
            shared_expert_tensor::output_gate,
        };
        std::array<const std::uint16_t **, 4> destinations{
            &state.shared_weights.gate,
            &state.shared_weights.up,
            &state.shared_weights.down,
            &state.shared_weights.output_gate,
        };
        for (std::size_t index = 0u; index < shared_tensors.size(); ++index) {
            shared_expert_tensor_spec spec{};
            const shared_expert_status spec_status = shared_expert_tensor_spec_for(
                    state.shared_config, shared_tensors[index], &spec);
            std::string name;
            if (spec_status != shared_expert_status::ok ||
                !shared_tensor_name_checked(options.layer, shared_tensors[index],
                                            &name, error) ||
                !load_shared_tensor(*checkpoint, name, spec.rows, spec.columns,
                                    initialization_stream, destinations[index],
                                    &state.resident_gpu_bytes, error)) {
                return spec_status == shared_expert_status::ok
                               ? moe_adapter_status::checkpoint_error
                               : map_shared_status(spec_status);
            }
        }

        std::shared_ptr<const expert_checkpoint_catalog> expert_catalog =
                options.shared_expert_catalog;
        if (!expert_catalog) {
            if (!expert_checkpoint_catalog::open(
                        model_root, pinned_expert_checkpoint_identity(),
                        &expert_catalog, &load_error)) {
                return fail(moe_adapter_status::checkpoint_error, error,
                            load_error.empty()
                                    ? "cannot open routed-expert catalog"
                                    : load_error);
            }
        }
        const expert_bridge_options bridge_options{
            options.device, kMoeTopK, options.ram_capacity_bytes,
            options.prefetch_workers, options.shared_expert_arena,
            options.owned_direct_pread};
        if (!expert_bridge::open(expert_catalog, options.layer, bridge_options,
                                 &state.bridge, &load_error)) {
            return fail(moe_adapter_status::pager_error, error,
                        load_error.empty() ? "cannot open bounded expert bridge"
                                           : load_error);
        }
        const expert_bridge_metrics bridge_metrics = state.bridge->metrics();
        if (!checked_add(state.resident_gpu_bytes,
                         bridge_metrics.gpu_owned_bytes,
                         &state.resident_gpu_bytes)) {
            return fail(moe_adapter_status::allocation_failure, error,
                        "resident GPU byte accounting overflow");
        }

        bf16_linear_workspace_requirements requirements{};
        const bf16_linear_status requirement_status =
                bf16_linear_get_workspace_requirements(router_config, &requirements);
        if (requirement_status != bf16_linear_status::ok) {
            return fail(map_bf16_status(requirement_status), error,
                        "router workspace requirement derivation failed");
        }

        auto allocate = [&](auto **pointer, std::size_t count,
                            const char *label) -> bool {
            using value_type = std::remove_pointer_t<
                    std::remove_reference_t<decltype(*pointer)>>;
            std::size_t bytes = 0u;
            if (!checked_multiply(count, sizeof(value_type), &bytes)) {
                set_error(error, std::string(label) + " allocation overflows");
                return false;
            }
            const cudaError_t allocation = cudaMalloc(
                    reinterpret_cast<void **>(pointer), bytes);
            if (allocation != cudaSuccess) {
                set_error(error, cuda_message(label, allocation));
                return false;
            }
            return checked_add(state.resident_gpu_bytes, bytes,
                               &state.resident_gpu_bytes);
        };

        if (!allocate(&state.router_input_bf16,
                      requirements.input_bf16_bytes / sizeof(std::uint16_t),
                      "cudaMalloc(router input BF16)") ||
            cudaMalloc(&state.router_blas_workspace,
                       requirements.blas_workspace_bytes) != cudaSuccess ||
            !checked_add(state.resident_gpu_bytes,
                         requirements.blas_workspace_bytes,
                         &state.resident_gpu_bytes) ||
            !allocate(&state.router_logits, kRouterLogits,
                      "cudaMalloc(router logits)") ||
            !allocate(&state.router_indices, kSelectedExperts,
                      "cudaMalloc(router indices)") ||
            !allocate(&state.router_weights, kSelectedExperts,
                      "cudaMalloc(router weights)") ||
            !allocate(&state.router_status, 1u,
                      "cudaMalloc(router status)") ||
            !allocate(&state.routed_mid,
                      static_cast<std::size_t>(kMoeTopK) * kMoeIntermediate,
                      "cudaMalloc(routed intermediate)") ||
            !allocate(&state.routed_output, kMoeHidden,
                      "cudaMalloc(routed output)") ||
            !allocate(&state.shared_intermediate, kSharedExpertIntermediate,
                      "cudaMalloc(shared intermediate)") ||
            !allocate(&state.shared_gate, 1u,
                      "cudaMalloc(shared gate)") ||
            !allocate(&state.shared_status, 1u,
                      "cudaMalloc(shared status)") ||
            !allocate(&state.shared_output, kMoeHidden,
                      "cudaMalloc(shared output)") ||
            !allocate(&state.forward_status, 1u,
                      "cudaMalloc(forward status)")) {
            return moe_adapter_status::allocation_failure;
        }
        state.router_scratch = bf16_linear_scratch{
            state.router_input_bf16, requirements.input_bf16_bytes,
            state.router_blas_workspace, requirements.blas_workspace_bytes};

        cudaError_t allocation = cudaHostAlloc(
                reinterpret_cast<void **>(&state.host_route),
                sizeof(route_host_snapshot), cudaHostAllocPortable);
        if (allocation == cudaSuccess) {
            allocation = cudaHostAlloc(
                    reinterpret_cast<void **>(&state.host_forward_status),
                    sizeof(std::uint32_t), cudaHostAllocPortable);
        }
        if (allocation == cudaSuccess) {
            allocation = cudaEventCreateWithFlags(&state.route_event,
                                                  cudaEventDisableTiming);
        }
        if (allocation == cudaSuccess) {
            allocation = cudaEventCreateWithFlags(&state.forward_event,
                                                  cudaEventDisableTiming);
        }
        if (allocation != cudaSuccess) {
            return fail(moe_adapter_status::allocation_failure, error,
                        cuda_message("allocate pinned route state/events", allocation));
        }
        *state.host_route = route_host_snapshot{};
        *state.host_forward_status =
                static_cast<std::uint32_t>(moe_adapter_status::ok);
        state.initialized = true;
        *out = std::move(result);
        return moe_adapter_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(moe_adapter_status::allocation_failure, error,
                    "host allocation failed during adapter load");
    } catch (const std::exception &exception) {
        return fail(moe_adapter_status::checkpoint_error, error,
                    std::string("adapter load exception: ") + exception.what());
    } catch (...) {
        return fail(moe_adapter_status::checkpoint_error, error,
                    "unknown adapter load exception");
    }
}

moe_adapter_status routed_moe_layer_adapter::enqueue_route(
        const float *input,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->initialized || input == nullptr) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "adapter is not initialized or route input is null");
    }
    scoped_spin_lock lock(impl_->mutex);
    if (impl_->route_in_flight || impl_->generation_active) {
        return fail(moe_adapter_status::busy, error,
                    "the bounded route/slot set is already in use");
    }
    if (!pointer_visible_to_device(input, impl_->options.device)) {
        return fail(moe_adapter_status::invalid_device_pointer, error,
                    "route input is not visible to the configured device");
    }

    const bf16_linear_status linear_status = impl_->router->forward(
            input, 1u, impl_->router_scratch, impl_->router_logits, stream);
    if (linear_status != bf16_linear_status::ok) {
        return fail(map_bf16_status(linear_status), error,
                    std::string("router projection failed: ") +
                            bf16_linear_status_string(linear_status));
    }
    const moe_status route_status = moe_router_topk_cuda(
            impl_->config, impl_->router_logits, impl_->router_indices,
            impl_->router_weights, impl_->router_status, stream);
    if (route_status != moe_status::ok) {
        (void)cudaStreamSynchronize(stream);
        return fail(map_moe_status(route_status), error,
                    std::string("top-k router failed: ") +
                            moe_status_string(route_status));
    }
    cudaError_t status = cudaMemcpyAsync(
            impl_->host_route->indices.data(), impl_->router_indices,
            sizeof(impl_->host_route->indices), cudaMemcpyDeviceToHost, stream);
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(
                impl_->host_route->weights.data(), impl_->router_weights,
                sizeof(impl_->host_route->weights), cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(&impl_->host_route->status, impl_->router_status,
                                 sizeof(impl_->host_route->status),
                                 cudaMemcpyDeviceToHost, stream);
    }
    if (status == cudaSuccess) {
        status = cudaEventRecord(impl_->route_event, stream);
    }
    if (status != cudaSuccess) {
        (void)cudaStreamSynchronize(stream);
        return fail(moe_adapter_status::cuda_error, error,
                    cuda_message("record asynchronous route", status));
    }
    impl_->route_stream = stream;
    impl_->route_in_flight = true;
    return moe_adapter_status::ok;
}

moe_adapter_status routed_moe_layer_adapter::prepare_route(
        cudaStream_t stream,
        prepared_generation *out,
        std::string *error) noexcept {
    if (!impl_ || !impl_->initialized || out == nullptr) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "adapter is not initialized or generation output is null");
    }
    *out = prepared_generation{};
    scoped_spin_lock lock(impl_->mutex);
    if (!impl_->route_in_flight || impl_->generation_active) {
        return fail(moe_adapter_status::busy, error,
                    "no unique completed route is available for preparation");
    }
    if (stream != impl_->route_stream) {
        return fail(moe_adapter_status::invalid_argument, error,
                    "prepare stream differs from enqueue_route stream");
    }
    const cudaError_t wait_status = cudaEventSynchronize(impl_->route_event);
    impl_->route_in_flight = false;
    if (wait_status != cudaSuccess) {
        return fail(moe_adapter_status::cuda_error, error,
                    cuda_message("cudaEventSynchronize(route)", wait_status));
    }
    const auto route_status = static_cast<moe_status>(impl_->host_route->status);
    if (route_status != moe_status::ok) {
        return fail(map_moe_status(route_status), error,
                    std::string("device router rejected input: ") +
                            moe_status_string(route_status));
    }

    std::array<bool, kMoeExperts> observed{};
    float weight_sum = 0.0f;
    for (std::size_t slot = 0u; slot < kSelectedExperts; ++slot) {
        const std::uint32_t expert = impl_->host_route->indices[slot];
        const float weight = impl_->host_route->weights[slot];
        if (expert >= kMoeExperts || observed[expert] ||
            !std::isfinite(weight) || weight <= 0.0f) {
            return fail(moe_adapter_status::router_error, error,
                        "router produced duplicate/out-of-range ids or invalid weights");
        }
        observed[expert] = true;
        weight_sum += weight;
    }
    if (!std::isfinite(weight_sum) || std::abs(weight_sum - 1.0f) > 2.0e-5f) {
        return fail(moe_adapter_status::router_error, error,
                    "normalized top-10 router weights do not sum to one");
    }

    expert_bridge::prepared_generation bridge_generation;
    std::string bridge_error;
    if (!impl_->bridge->prepare(impl_->host_route->indices.data(), kMoeTopK,
                                stream, &bridge_generation, &bridge_error)) {
        const bool shared_busy = bridge_error.find("arena is busy") !=
                                 std::string::npos;
        return fail(shared_busy ? moe_adapter_status::busy
                                : moe_adapter_status::pager_error, error,
                    bridge_error.empty() ? "expert bridge preparation failed"
                                         : bridge_error);
    }
    try {
        auto prepared_state = std::make_unique<prepared_generation::state>();
        prepared_state->owner = impl_;
        prepared_state->bridge_generation = std::move(bridge_generation);
        prepared_state->experts = impl_->host_route->indices;
        prepared_state->weights = impl_->host_route->weights;
        prepared_state->stream = stream;
        impl_->generation_active = true;
        *out = prepared_generation(std::move(prepared_state));
        return moe_adapter_status::ok;
    } catch (const std::bad_alloc &) {
        std::string rollback_error;
        (void)bridge_generation.rollback(&rollback_error);
        return fail(moe_adapter_status::allocation_failure, error,
                    "prepared-generation allocation failed");
    } catch (...) {
        std::string rollback_error;
        (void)bridge_generation.rollback(&rollback_error);
        return fail(moe_adapter_status::transaction_error, error,
                    "prepared-generation construction failed");
    }
}

std::uint32_t routed_moe_layer_adapter::layer() const noexcept {
    return impl_ ? impl_->options.layer : 0u;
}

int routed_moe_layer_adapter::device() const noexcept {
    return impl_ ? impl_->options.device : -1;
}

bool routed_moe_layer_adapter::initialized() const noexcept {
    return impl_ && impl_->initialized;
}

ExpertPagerMetrics routed_moe_layer_adapter::pager_metrics() const noexcept {
    // The bridge is immutable after load; its pager owns snapshot locking.
    // Do not take the adapter execution lock, which spans route preparation.
    return impl_ && impl_->bridge
            ? impl_->bridge->pager_metrics() : ExpertPagerMetrics{};
}

moe_adapter_metrics routed_moe_layer_adapter::metrics() const noexcept {
    moe_adapter_metrics result{};
    if (!impl_) return result;
    scoped_spin_lock lock(impl_->mutex);
    result.layer = impl_->options.layer;
    result.bounded_slots = kMoeTopK;
    result.resident_gpu_bytes = impl_->resident_gpu_bytes;
    const expert_bridge_metrics bridge = impl_->bridge
            ? impl_->bridge->metrics() : expert_bridge_metrics{};
    result.expert_gpu_owned_bytes = bridge.gpu_owned_bytes;
    result.expert_gpu_shared_bytes = bridge.gpu_shared_bytes;
    const ExpertPagerMetrics pager = impl_->bridge
            ? impl_->bridge->pager_metrics() : ExpertPagerMetrics{};
    result.pager_requests = pager.requests;
    result.pager_ram_hits = pager.ram_hits;
    result.pager_ram_misses = pager.ram_misses;
    result.pager_nvme_reads = pager.nvme_reads;
    result.pager_nvme_bytes = pager.nvme_bytes_read;
    result.pager_active_generations = pager.active_generations;
    result.route_in_flight = impl_->route_in_flight;
    result.generation_active = impl_->generation_active;
    return result;
}

}  // namespace axiom::qwen4exp
