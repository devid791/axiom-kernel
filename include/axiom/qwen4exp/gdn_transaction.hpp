#pragma once

#include "axiom/qwen4exp/gdn.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

// A transaction snapshots one complete GDN state.  The bound keeps malformed
// or unexpectedly batched configurations from turning speculative support
// into an unbounded device allocation.  Qwen4-Exp batch-one state is well
// below this limit.
inline constexpr std::size_t kGdnTransactionMaxSnapshotBytes =
    256u * 1024u * 1024u;

enum class GdnTransactionStatus : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kUnsupportedConfig,
    kDimensionMismatch,
    kSizeOverflow,
    kSnapshotTooLarge,
    kUninitialized,
    kAlreadyInitialized,
    kStateMismatch,
    kUnsupportedDevice,
    kStreamMismatch,
    kTransactionOpen,
    kNoTransaction,
    kPartialAcceptUnsupported,
    kCudaError,
    kGdnError,
};

// This header is stored in the same single CUDA allocation as the two exact
// state backups.  A tiny kernel captures it on the transaction stream so the
// device snapshot has the same ordering as the state copies.
struct alignas(16) GdnTransactionDeviceMetadata {
    std::uint64_t tokens_seen = 0;
    std::uint32_t conv_cursor = 0;
    std::uint32_t reserved = 0;
};

// Additive non-owning wrapper around an initialized GdnDeviceState.  It owns
// only one bounded, contiguous CUDA snapshot allocation and never owns or
// releases the wrapped GDN state.  Treat this structure as non-copyable.
//
// The wrapped state must be accessed exclusively through this wrapper while a
// transaction is open.  Every operation is tied to the stream supplied at
// init; using another stream fails closed instead of introducing a race.
struct GdnTransactionState {
    GdnDeviceState* state = nullptr;
    GdnConfig bound_config{};
    float* bound_conv_state = nullptr;
    float* bound_recurrent_state = nullptr;

    std::byte* snapshot_allocation = nullptr;
    GdnTransactionDeviceMetadata* snapshot_metadata = nullptr;
    float* snapshot_conv_state = nullptr;
    float* snapshot_recurrent_state = nullptr;
    std::size_t snapshot_allocation_bytes = 0;
    std::size_t conv_state_bytes = 0;
    std::size_t recurrent_state_bytes = 0;

    std::uint32_t committed_conv_cursor = 0;
    std::uint64_t committed_tokens_seen = 0;
    std::uint64_t staged_steps = 0;
    cudaStream_t stream = nullptr;
    int device = -1;
    GdnStatus last_gdn_status = GdnStatus::kOk;
    bool initialized = false;
    bool transaction_open = false;
};

[[nodiscard]] const char* gdn_transaction_status_string(
    GdnTransactionStatus status) noexcept;

// Allocates the sole transaction snapshot.  The GDN state itself must already
// be initialized on the current sm_120 device and remains caller-owned.
[[nodiscard]] GdnTransactionStatus gdn_transaction_state_init(
    GdnTransactionState* transaction,
    GdnDeviceState* state,
    cudaStream_t stream = nullptr) noexcept;

// Release is intentionally outside the hot path and may call cudaFree.  An
// open transaction must first be committed or rolled back.
[[nodiscard]] GdnTransactionStatus gdn_transaction_state_release(
    GdnTransactionState* transaction) noexcept;

// begin/stage/commit/rollback are allocation-free and never synchronize the
// device.  CUDA state copies and metadata capture are ordered on the bound
// stream.  Host metadata is restored immediately on rollback because the base
// GdnDeviceState stores cursor/tokens on the host.
[[nodiscard]] GdnTransactionStatus gdn_transaction_begin(
    GdnTransactionState* transaction,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] GdnTransactionStatus gdn_transaction_stage(
    GdnTransactionState* transaction,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] GdnTransactionStatus gdn_transaction_commit_all(
    GdnTransactionState* transaction,
    cudaStream_t stream = nullptr) noexcept;

// GDN recurrent state cannot publish an arbitrary speculative prefix from a
// single backup.  This API is explicit and always fail-closed: the decoder
// must rollback and replay the accepted prefix before commit_all().
[[nodiscard]] GdnTransactionStatus gdn_transaction_commit_prefix(
    GdnTransactionState* transaction,
    std::uint64_t accepted_steps,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] GdnTransactionStatus gdn_transaction_rollback(
    GdnTransactionState* transaction,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp
