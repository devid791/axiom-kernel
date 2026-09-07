#include "axiom/qwen4exp/gdn_transaction.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

static_assert(sizeof(GdnTransactionDeviceMetadata) == 16u);
static_assert(alignof(GdnTransactionDeviceMetadata) >= alignof(float));

bool checked_add(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* result) noexcept {
    if (result == nullptr || lhs > std::numeric_limits<std::size_t>::max() - rhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool checked_mul(
    std::size_t lhs,
    std::size_t rhs,
    std::size_t* result) noexcept {
    if (result == nullptr ||
        (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool same_config(const GdnConfig& lhs, const GdnConfig& rhs) noexcept {
    return lhs.batch == rhs.batch &&
           lhs.key_heads == rhs.key_heads &&
           lhs.value_heads == rhs.value_heads &&
           lhs.key_head_dim == rhs.key_head_dim &&
           lhs.value_head_dim == rhs.value_head_dim &&
           lhs.conv_kernel == rhs.conv_kernel &&
           lhs.rms_epsilon == rhs.rms_epsilon &&
           lhs.l2_epsilon == rhs.l2_epsilon;
}

GdnTransactionStatus map_gdn_status(GdnStatus status) noexcept {
    switch (status) {
        case GdnStatus::kOk:
            return GdnTransactionStatus::kOk;
        case GdnStatus::kInvalidArgument:
            return GdnTransactionStatus::kInvalidArgument;
        case GdnStatus::kUnsupportedConfig:
            return GdnTransactionStatus::kUnsupportedConfig;
        case GdnStatus::kDimensionMismatch:
            return GdnTransactionStatus::kDimensionMismatch;
        case GdnStatus::kSizeOverflow:
            return GdnTransactionStatus::kSizeOverflow;
        case GdnStatus::kUninitialized:
            return GdnTransactionStatus::kUninitialized;
        case GdnStatus::kStateMismatch:
            return GdnTransactionStatus::kStateMismatch;
        case GdnStatus::kUnsupportedDevice:
            return GdnTransactionStatus::kUnsupportedDevice;
        case GdnStatus::kCudaError:
            return GdnTransactionStatus::kCudaError;
    }
    return GdnTransactionStatus::kGdnError;
}

bool device_pointer_matches(const void* pointer, int expected_device) noexcept {
    if (pointer == nullptr) {
        return false;
    }
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        return false;
    }
    return attributes.type == cudaMemoryTypeDevice &&
           attributes.device == expected_device;
}

GdnTransactionStatus validate_bound_state(
    const GdnTransactionState* transaction,
    cudaStream_t stream,
    bool allow_metadata_recovery) noexcept {
    if (transaction == nullptr) {
        return GdnTransactionStatus::kInvalidArgument;
    }
    if (!transaction->initialized) {
        return GdnTransactionStatus::kUninitialized;
    }
    if (transaction->state == nullptr ||
        transaction->snapshot_allocation == nullptr ||
        transaction->snapshot_metadata == nullptr ||
        transaction->snapshot_conv_state == nullptr ||
        transaction->snapshot_recurrent_state == nullptr ||
        transaction->bound_conv_state == nullptr ||
        transaction->bound_recurrent_state == nullptr) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (stream != transaction->stream) {
        return GdnTransactionStatus::kStreamMismatch;
    }

    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    if (current_device != transaction->device ||
        transaction->state->device != transaction->device) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (!transaction->state->initialized ||
        transaction->state->conv_state != transaction->bound_conv_state ||
        transaction->state->recurrent_state != transaction->bound_recurrent_state ||
        !same_config(transaction->state->config, transaction->bound_config)) {
        return GdnTransactionStatus::kStateMismatch;
    }

    GdnFootprint footprint{};
    const GdnStatus footprint_status =
        gdn_footprint(transaction->bound_config, &footprint);
    if (footprint_status != GdnStatus::kOk) {
        return map_gdn_status(footprint_status);
    }
    std::size_t expected_conv_bytes = 0u;
    std::size_t expected_recurrent_bytes = 0u;
    std::size_t expected_state_bytes = 0u;
    std::size_t expected_allocation_bytes = 0u;
    if (!checked_mul(
            footprint.conv_state_floats,
            sizeof(float),
            &expected_conv_bytes) ||
        !checked_mul(
            footprint.recurrent_state_floats,
            sizeof(float),
            &expected_recurrent_bytes) ||
        !checked_add(
            expected_conv_bytes,
            expected_recurrent_bytes,
            &expected_state_bytes) ||
        !checked_add(
            sizeof(GdnTransactionDeviceMetadata),
            expected_state_bytes,
            &expected_allocation_bytes)) {
        return GdnTransactionStatus::kSizeOverflow;
    }
    if (expected_state_bytes != footprint.device_state_bytes ||
        transaction->state->conv_state_floats != footprint.conv_state_floats ||
        transaction->state->recurrent_state_floats != footprint.recurrent_state_floats ||
        transaction->conv_state_bytes != expected_conv_bytes ||
        transaction->recurrent_state_bytes != expected_recurrent_bytes ||
        transaction->snapshot_allocation_bytes != expected_allocation_bytes) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (!allow_metadata_recovery &&
        transaction->state->conv_cursor >= transaction->bound_config.conv_kernel) {
        return GdnTransactionStatus::kStateMismatch;
    }
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus validate_expected_metadata(
    const GdnTransactionState* transaction) noexcept {
    if (transaction->staged_steps >
        std::numeric_limits<std::uint64_t>::max() -
            transaction->committed_tokens_seen) {
        return GdnTransactionStatus::kSizeOverflow;
    }
    const std::uint64_t expected_tokens =
        transaction->committed_tokens_seen + transaction->staged_steps;
    const std::size_t cursor_advance = static_cast<std::size_t>(
        transaction->staged_steps % transaction->bound_config.conv_kernel);
    const std::uint32_t expected_cursor = static_cast<std::uint32_t>(
        (static_cast<std::size_t>(transaction->committed_conv_cursor) +
         cursor_advance) %
        transaction->bound_config.conv_kernel);
    if (transaction->state->tokens_seen != expected_tokens ||
        transaction->state->conv_cursor != expected_cursor) {
        return GdnTransactionStatus::kStateMismatch;
    }
    return GdnTransactionStatus::kOk;
}

__global__ void capture_metadata_kernel(
    GdnTransactionDeviceMetadata* destination,
    std::uint32_t conv_cursor,
    std::uint64_t tokens_seen) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        destination->tokens_seen = tokens_seen;
        destination->conv_cursor = conv_cursor;
        destination->reserved = 0u;
    }
}

}  // namespace

const char* gdn_transaction_status_string(GdnTransactionStatus status) noexcept {
    switch (status) {
        case GdnTransactionStatus::kOk:
            return "ok";
        case GdnTransactionStatus::kInvalidArgument:
            return "invalid_argument";
        case GdnTransactionStatus::kUnsupportedConfig:
            return "unsupported_config";
        case GdnTransactionStatus::kDimensionMismatch:
            return "dimension_mismatch";
        case GdnTransactionStatus::kSizeOverflow:
            return "size_overflow";
        case GdnTransactionStatus::kSnapshotTooLarge:
            return "snapshot_too_large";
        case GdnTransactionStatus::kUninitialized:
            return "uninitialized";
        case GdnTransactionStatus::kAlreadyInitialized:
            return "already_initialized";
        case GdnTransactionStatus::kStateMismatch:
            return "state_mismatch";
        case GdnTransactionStatus::kUnsupportedDevice:
            return "unsupported_device";
        case GdnTransactionStatus::kStreamMismatch:
            return "stream_mismatch";
        case GdnTransactionStatus::kTransactionOpen:
            return "transaction_open";
        case GdnTransactionStatus::kNoTransaction:
            return "no_transaction";
        case GdnTransactionStatus::kPartialAcceptUnsupported:
            return "partial_accept_unsupported";
        case GdnTransactionStatus::kCudaError:
            return "cuda_error";
        case GdnTransactionStatus::kGdnError:
            return "gdn_error";
    }
    return "unknown";
}

GdnTransactionStatus gdn_transaction_state_init(
    GdnTransactionState* transaction,
    GdnDeviceState* state,
    cudaStream_t stream) noexcept {
    if (transaction == nullptr || state == nullptr) {
        return GdnTransactionStatus::kInvalidArgument;
    }
    if (transaction->initialized || transaction->state != nullptr ||
        transaction->snapshot_allocation != nullptr) {
        return GdnTransactionStatus::kAlreadyInitialized;
    }
    if (!state->initialized) {
        return GdnTransactionStatus::kUninitialized;
    }

    GdnFootprint footprint{};
    const GdnStatus footprint_status = gdn_footprint(state->config, &footprint);
    if (footprint_status != GdnStatus::kOk) {
        return map_gdn_status(footprint_status);
    }
    if (state->conv_state == nullptr || state->recurrent_state == nullptr ||
        state->conv_state_floats != footprint.conv_state_floats ||
        state->recurrent_state_floats != footprint.recurrent_state_floats ||
        state->conv_cursor >= state->config.conv_kernel) {
        return GdnTransactionStatus::kStateMismatch;
    }

    int device = -1;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    if (properties.major != 12 || properties.minor != 0) {
        return GdnTransactionStatus::kUnsupportedDevice;
    }
    if (state->device != device ||
        !device_pointer_matches(state->conv_state, device) ||
        !device_pointer_matches(state->recurrent_state, device)) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (stream != nullptr) {
        unsigned int flags = 0u;
        if (cudaStreamGetFlags(stream, &flags) != cudaSuccess) {
            return GdnTransactionStatus::kCudaError;
        }
    }

    std::size_t conv_bytes = 0u;
    std::size_t recurrent_bytes = 0u;
    std::size_t state_bytes = 0u;
    std::size_t allocation_bytes = 0u;
    if (!checked_mul(
            footprint.conv_state_floats, sizeof(float), &conv_bytes) ||
        !checked_mul(
            footprint.recurrent_state_floats,
            sizeof(float),
            &recurrent_bytes) ||
        !checked_add(conv_bytes, recurrent_bytes, &state_bytes) ||
        !checked_add(
            sizeof(GdnTransactionDeviceMetadata),
            state_bytes,
            &allocation_bytes)) {
        return GdnTransactionStatus::kSizeOverflow;
    }
    if (state_bytes != footprint.device_state_bytes) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (allocation_bytes > kGdnTransactionMaxSnapshotBytes) {
        return GdnTransactionStatus::kSnapshotTooLarge;
    }

    std::byte* allocation = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&allocation), allocation_bytes) !=
        cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }

    transaction->state = state;
    transaction->bound_config = state->config;
    transaction->bound_conv_state = state->conv_state;
    transaction->bound_recurrent_state = state->recurrent_state;
    transaction->snapshot_allocation = allocation;
    transaction->snapshot_metadata =
        reinterpret_cast<GdnTransactionDeviceMetadata*>(allocation);
    transaction->snapshot_conv_state = reinterpret_cast<float*>(
        allocation + sizeof(GdnTransactionDeviceMetadata));
    transaction->snapshot_recurrent_state = reinterpret_cast<float*>(
        allocation + sizeof(GdnTransactionDeviceMetadata) + conv_bytes);
    transaction->snapshot_allocation_bytes = allocation_bytes;
    transaction->conv_state_bytes = conv_bytes;
    transaction->recurrent_state_bytes = recurrent_bytes;
    transaction->committed_conv_cursor = 0u;
    transaction->committed_tokens_seen = 0u;
    transaction->staged_steps = 0u;
    transaction->stream = stream;
    transaction->device = device;
    transaction->last_gdn_status = GdnStatus::kOk;
    transaction->initialized = true;
    transaction->transaction_open = false;
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus gdn_transaction_state_release(
    GdnTransactionState* transaction) noexcept {
    if (transaction == nullptr) {
        return GdnTransactionStatus::kInvalidArgument;
    }
    if (!transaction->initialized) {
        if (transaction->snapshot_allocation == nullptr &&
            transaction->state == nullptr) {
            *transaction = GdnTransactionState{};
            return GdnTransactionStatus::kOk;
        }
        return GdnTransactionStatus::kStateMismatch;
    }
    if (transaction->transaction_open) {
        return GdnTransactionStatus::kTransactionOpen;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    if (current_device != transaction->device) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (transaction->snapshot_allocation == nullptr) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (cudaFree(transaction->snapshot_allocation) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    *transaction = GdnTransactionState{};
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus gdn_transaction_begin(
    GdnTransactionState* transaction,
    cudaStream_t stream) noexcept {
    const GdnTransactionStatus validation =
        validate_bound_state(transaction, stream, false);
    if (validation != GdnTransactionStatus::kOk) {
        return validation;
    }
    if (transaction->transaction_open) {
        return GdnTransactionStatus::kTransactionOpen;
    }

    const std::uint32_t cursor = transaction->state->conv_cursor;
    const std::uint64_t tokens = transaction->state->tokens_seen;
    capture_metadata_kernel<<<1u, 1u, 0u, stream>>>(
        transaction->snapshot_metadata, cursor, tokens);
    if (cudaPeekAtLastError() != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    if (cudaMemcpyAsync(
            transaction->snapshot_conv_state,
            transaction->state->conv_state,
            transaction->conv_state_bytes,
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess ||
        cudaMemcpyAsync(
            transaction->snapshot_recurrent_state,
            transaction->state->recurrent_state,
            transaction->recurrent_state_bytes,
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }

    transaction->committed_conv_cursor = cursor;
    transaction->committed_tokens_seen = tokens;
    transaction->staged_steps = 0u;
    transaction->last_gdn_status = GdnStatus::kOk;
    transaction->transaction_open = true;
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus gdn_transaction_stage(
    GdnTransactionState* transaction,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream) noexcept {
    const GdnTransactionStatus validation =
        validate_bound_state(transaction, stream, false);
    if (validation != GdnTransactionStatus::kOk) {
        return validation;
    }
    if (!transaction->transaction_open) {
        return GdnTransactionStatus::kNoTransaction;
    }
    const GdnTransactionStatus metadata_status =
        validate_expected_metadata(transaction);
    if (metadata_status != GdnTransactionStatus::kOk) {
        return metadata_status;
    }
    if (transaction->staged_steps ==
            std::numeric_limits<std::uint64_t>::max() ||
        transaction->committed_tokens_seen ==
            std::numeric_limits<std::uint64_t>::max() -
                transaction->staged_steps) {
        return GdnTransactionStatus::kSizeOverflow;
    }

    const GdnStatus gdn_status = gdn_step_cuda(
        transaction->state, inputs, weights, outputs, stream);
    transaction->last_gdn_status = gdn_status;
    if (gdn_status != GdnStatus::kOk) {
        return map_gdn_status(gdn_status);
    }
    ++transaction->staged_steps;
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus gdn_transaction_commit_all(
    GdnTransactionState* transaction,
    cudaStream_t stream) noexcept {
    const GdnTransactionStatus validation =
        validate_bound_state(transaction, stream, false);
    if (validation != GdnTransactionStatus::kOk) {
        return validation;
    }
    if (!transaction->transaction_open) {
        return GdnTransactionStatus::kNoTransaction;
    }
    const GdnTransactionStatus metadata_status =
        validate_expected_metadata(transaction);
    if (metadata_status != GdnTransactionStatus::kOk) {
        return metadata_status;
    }
    transaction->staged_steps = 0u;
    transaction->transaction_open = false;
    transaction->last_gdn_status = GdnStatus::kOk;
    return GdnTransactionStatus::kOk;
}

GdnTransactionStatus gdn_transaction_commit_prefix(
    GdnTransactionState* transaction,
    std::uint64_t accepted_steps,
    cudaStream_t stream) noexcept {
    const GdnTransactionStatus validation =
        validate_bound_state(transaction, stream, false);
    if (validation != GdnTransactionStatus::kOk) {
        return validation;
    }
    if (!transaction->transaction_open) {
        return GdnTransactionStatus::kNoTransaction;
    }
    const GdnTransactionStatus metadata_status =
        validate_expected_metadata(transaction);
    if (metadata_status != GdnTransactionStatus::kOk) {
        return metadata_status;
    }
    if (accepted_steps > transaction->staged_steps) {
        return GdnTransactionStatus::kInvalidArgument;
    }
    return GdnTransactionStatus::kPartialAcceptUnsupported;
}

GdnTransactionStatus gdn_transaction_rollback(
    GdnTransactionState* transaction,
    cudaStream_t stream) noexcept {
    const GdnTransactionStatus validation =
        validate_bound_state(transaction, stream, true);
    if (validation != GdnTransactionStatus::kOk) {
        return validation;
    }
    if (!transaction->transaction_open) {
        return GdnTransactionStatus::kNoTransaction;
    }
    if (transaction->committed_conv_cursor >=
        transaction->bound_config.conv_kernel) {
        return GdnTransactionStatus::kStateMismatch;
    }
    if (cudaMemcpyAsync(
            transaction->state->conv_state,
            transaction->snapshot_conv_state,
            transaction->conv_state_bytes,
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess ||
        cudaMemcpyAsync(
            transaction->state->recurrent_state,
            transaction->snapshot_recurrent_state,
            transaction->recurrent_state_bytes,
            cudaMemcpyDeviceToDevice,
            stream) != cudaSuccess) {
        return GdnTransactionStatus::kCudaError;
    }
    transaction->state->conv_cursor = transaction->committed_conv_cursor;
    transaction->state->tokens_seen = transaction->committed_tokens_seen;
    transaction->staged_steps = 0u;
    transaction->transaction_open = false;
    transaction->last_gdn_status = GdnStatus::kOk;
    return GdnTransactionStatus::kOk;
}

}  // namespace axiom::qwen4exp
