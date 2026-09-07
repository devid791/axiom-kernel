#ifndef AXIOM_QWEN4EXP_PLE_ADAPTER_HPP
#define AXIOM_QWEN4EXP_PLE_ADAPTER_HPP

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kPleAdapterDecoderLayers = 48u;
inline constexpr std::size_t kPleAdapterLayerIndex = 1u;
inline constexpr std::size_t kPleAdapterCheckpointTensors = 138u;
inline constexpr std::size_t kPleAdapterResidentWeightTensors = 6u;
inline constexpr std::size_t kPleAdapterMetadataTensors = 3u;
inline constexpr std::size_t kPleAdapterColdRowBytes = 160u;
inline constexpr std::size_t kPleAdapterRowsPerToken = 16u;
inline constexpr std::size_t kPleAdapterColdPrefetchBytes =
        kPleAdapterColdRowBytes * kPleAdapterRowsPerToken;

enum class ple_adapter_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_layer,
    unsupported_config,
    invalid_checkpoint,
    tensor_not_found,
    dtype_mismatch,
    shape_mismatch,
    metadata_mismatch,
    size_overflow,
    checkpoint_io_error,
    resource_exhausted,
    unsupported_device,
    invalid_device_pointer,
    invalid_state,
    cuda_error,
    embedding_error,
    compute_error,
};

struct ple_adapter_config {
    /* HF ple_layer_ids=[2] is one-indexed; the checkpoint path is layer 1. */
    std::size_t layer_index = kPleAdapterLayerIndex;
    std::size_t max_speculative_tokens = 32u;
    std::size_t row_cache_slots = 256u;
    std::size_t upload_chunk_bytes = 2u * 1024u * 1024u;
};

struct ple_adapter_footprint {
    std::size_t resident_weight_bytes = 0u;
    std::size_t device_state_bytes = 0u;
    std::size_t device_workspace_bytes = 0u;
    std::size_t pinned_prefetch_bytes = 0u;
    std::size_t row_cache_payload_bytes = 0u;
    std::size_t max_cold_nvme_bytes_per_token = 0u;
    std::size_t host_to_device_bytes_per_token = 0u;
};

[[nodiscard]] const char *ple_adapter_status_string(
        ple_adapter_status status) noexcept;

[[nodiscard]] ple_adapter_status ple_adapter_validate_config(
        const ple_adapter_config &config) noexcept;

[[nodiscard]] ple_adapter_status ple_adapter_get_footprint(
        const ple_adapter_config &config,
        ple_adapter_footprint *footprint) noexcept;

class ple_layer_session;

/* Additive, isolated adapter for the single real qwen4_exp PLE layer.
 *
 * load() validates the exact pinned checkpoint contract: only zero-based
 * decoder layer 1 may own PLE; its three I64 metadata tensors, 128 FP8 shards,
 * one BF16 calibration scalar and six BF16 compute tensors must all match.
 * The 51.2 GB FP8 table remains range-readable on NVMe.  Only the six compute
 * tensors are resident on the selected GPU.
 *
 * Initialization may allocate and synchronize the supplied stream while
 * bounded checkpoint chunks are copied.  It never modifies the checkpoint.
 * The checkpoint catalog and this adapter must outlive all sessions created
 * from it. */
class ple_layer_adapter final {
public:
    ple_layer_adapter();
    ~ple_layer_adapter();
    ple_layer_adapter(ple_layer_adapter &&) noexcept;
    ple_layer_adapter &operator=(ple_layer_adapter &&) noexcept;
    ple_layer_adapter(const ple_layer_adapter &) = delete;
    ple_layer_adapter &operator=(const ple_layer_adapter &) = delete;

    [[nodiscard]] static ple_adapter_status load(
            const checkpoint_catalog &catalog,
            const ple_adapter_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<ple_layer_adapter> *out,
            std::string *error) noexcept;

    /* A session owns EOS-aware token history, the explicit bounded row cache,
     * the transactional convolution state, pinned prefetch staging and all
     * CUDA scratch.  Session construction is an initialization boundary and
     * may allocate/synchronize. */
    [[nodiscard]] ple_adapter_status create_session(
            cudaStream_t initialization_stream,
            std::unique_ptr<ple_layer_session> *out,
            std::string *error) const noexcept;

    [[nodiscard]] const ple_adapter_config &config() const noexcept;
    [[nodiscard]] const ple_metadata &metadata() const noexcept;
    [[nodiscard]] ple_adapter_footprint footprint() const noexcept;
    [[nodiscard]] float embedding_scale() const noexcept;
    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class ple_layer_session;
};

/* One sequence/session.  Calls are intentionally not internally synchronized;
 * use one CUDA stream throughout a transaction or order streams with events.
 *
 * prefetch_token() is the explicit cold/warm paging boundary.  It computes
 * EOS-aware n-gram row IDs, performs at most sixteen bounded 160-byte
 * NVMe reads on cache misses, then copies exactly 2560 F32 elements to GPU.
 * It may synchronize the supplied stream so its pinned buffer can be reused.
 *
 * stage_prefetched_and_add_cuda() is the allocation- and synchronization-free
 * hot path.  It implements the official ordering and returns
 *
 *   hidden_after_ple = hidden_before_ple + ple(hidden_before_ple, token)
 *
 * which is the tensor that must be passed next to attn_hyper_connection.
 * It never substitutes or invokes that existing Axiom component. */
class ple_layer_session final {
public:
    ple_layer_session();
    ~ple_layer_session();
    ple_layer_session(ple_layer_session &&) noexcept;
    ple_layer_session &operator=(ple_layer_session &&) noexcept;
    ple_layer_session(const ple_layer_session &) = delete;
    ple_layer_session &operator=(const ple_layer_session &) = delete;

    [[nodiscard]] ple_adapter_status begin_transaction(
            std::string *error = nullptr) noexcept;

    [[nodiscard]] ple_adapter_status prefetch_token(
            std::int64_t token_id,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    /* Split form of the synchronous wrapper above. start stages CPU token
     * history and submits exactly one copied-ID job to a persistent session
     * worker; it performs no CUDA calls. finish collects errors, uploads and
     * synchronizes before making token_prefetched() true. No second start or
     * commit is allowed until this job is consumed or rolled back. Metrics
     * reflect collected jobs only, never race with worker cache mutations.
     * rollback/reset/destruction drain accepted I/O before reusing/freeing
     * staging. This is not an interruptible pread or concurrent session API. */
    [[nodiscard]] ple_adapter_status start_prefetch_token(
            std::int64_t token_id,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] ple_adapter_status finish_prefetch_token(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] ple_adapter_status stage_prefetched_and_add_cuda(
            const float *hidden_before_ple,
            float *hidden_after_ple,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] ple_adapter_status commit_prefix(
            std::size_t accepted_tokens,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] ple_adapter_status rollback(
            std::string *error = nullptr) noexcept;

    [[nodiscard]] ple_adapter_status reset(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] const ple_head_ids &prefetched_head_ids() const noexcept;
    [[nodiscard]] std::size_t staged_tokens() const noexcept;
    [[nodiscard]] std::uint64_t committed_tokens() const noexcept;
    [[nodiscard]] std::uint64_t cache_hits() const noexcept;
    [[nodiscard]] std::uint64_t cache_misses() const noexcept;
    [[nodiscard]] std::uint64_t nvme_bytes_read() const noexcept;
    [[nodiscard]] bool transaction_open() const noexcept;
    [[nodiscard]] bool token_prefetched() const noexcept;
    [[nodiscard]] bool prefetch_pending() const noexcept;
    [[nodiscard]] bool poisoned() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class ple_layer_adapter;
};

}  // namespace axiom::qwen4exp

#endif
