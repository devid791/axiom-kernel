#ifndef AXIOM_QWEN4EXP_DECODER_LAYER_HPP
#define AXIOM_QWEN4EXP_DECODER_LAYER_HPP

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/provider_plan.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

// A value-returning declaration needs no pager implementation headers here.
// Keep <future>/<mutex> out of unrelated CUDA translation units.
struct ExpertPagerMetrics;

class expert_slot_arena;
class expert_checkpoint_catalog;

inline constexpr std::size_t kDecoderLayerCount = 48u;
inline constexpr std::size_t kDecoderHidden = 2560u;
inline constexpr std::size_t kDecoderStreams = 4u;
inline constexpr std::size_t kDecoderResidual = 10240u;
inline constexpr std::size_t kDecoderMaxContext = 262144u;
inline constexpr std::size_t kDecoderMaxSpeculativeTokens = 4u;

enum class decoder_layer_kind : std::uint8_t {
    gated_deltanet = 0,
    qsa = 1,
};

enum class decoder_layer_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_layer,
    unsupported_config,
    invalid_checkpoint,
    allocation_failure,
    invalid_device_pointer,
    invalid_state,
    capacity_exceeded,
    mhc_error,
    gdn_error,
    qsa_error,
    ple_error,
    moe_error,
    transaction_error,
    cuda_error,
};

struct decoder_layer_config {
    std::size_t layer_index = 0u;
    std::size_t max_context = kDecoderMaxContext;
    std::size_t max_batch = 1u;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
    std::uint64_t moe_ram_capacity_bytes = 64ull * 1024ull * 1024ull;
    std::size_t moe_prefetch_workers = 2u;
    std::shared_ptr<expert_slot_arena> shared_expert_arena;
    std::shared_ptr<const expert_checkpoint_catalog> shared_expert_catalog;
    bool enable_ple = true;
    bool moe_owned_direct_pread = false;
};

struct decoder_layer_state {
    std::size_t layer_index = 0u;
    decoder_layer_kind kind = decoder_layer_kind::gated_deltanet;
    std::size_t context_capacity = 0u;
    std::uint64_t committed_tokens = 0u;
    std::size_t staged_tokens = 0u;
    bool has_ple = false;
    bool transaction_open = false;
    bool prepared_to_commit = false;
    bool poisoned = false;
};

[[nodiscard]] const char *decoder_layer_status_string(
        decoder_layer_status status) noexcept;
[[nodiscard]] decoder_layer_kind decoder_layer_kind_for(
        std::size_t layer_index) noexcept;
[[nodiscard]] bool decoder_layer_has_ple(std::size_t layer_index) noexcept;
[[nodiscard]] decoder_layer_status decoder_layer_validate_config(
        const decoder_layer_config &config) noexcept;

/* Proves the real 48-layer checkpoint schedule without relying on a report:
 * layers 3,7,...,47 must expose QSA and every other layer Gated DeltaNet;
 * both mHC sites and routed-MoE must exist on every layer; PLE tensors may
 * occur only under zero-based layer 1. */
[[nodiscard]] decoder_layer_status decoder_layer_validate_checkpoint_schedule(
        const checkpoint_catalog &catalog,
        std::string *error = nullptr) noexcept;

class decoder_layer_session;
class decoder_model_transaction;

/* Immutable weights for one exact qwen4_exp decoder layer.  The object is
 * additive and provider-local: it neither registers nor mutates an existing
 * Axiom model.  The owner must outlive every session created from it. */
class decoder_layer final {
public:
    decoder_layer();
    ~decoder_layer();
    decoder_layer(decoder_layer &&) noexcept;
    decoder_layer &operator=(decoder_layer &&) noexcept;
    decoder_layer(const decoder_layer &) = delete;
    decoder_layer &operator=(const decoder_layer &) = delete;

    [[nodiscard]] static decoder_layer_status load(
            const checkpoint_catalog &catalog,
            const decoder_layer_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<decoder_layer> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_layer_status create_session(
            std::size_t context_capacity,
            cudaStream_t initialization_stream,
            std::unique_ptr<decoder_layer_session> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] const decoder_layer_config &config() const noexcept;
    [[nodiscard]] decoder_layer_kind kind() const noexcept;
    [[nodiscard]] bool has_ple() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

    /* Forwards the MoE's thread-safe, value-only pager snapshot. The layer
     * must remain alive and unmoved; no payload retention or CUDA calls. */
    [[nodiscard]] ExpertPagerMetrics expert_pager_metrics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class decoder_layer_session;
};

/* Per-sequence state and bounded CUDA scratch for one decoder layer.
 * stage_token() implements the exact HF v5.16.1 order:
 *
 *   optional hidden += PLE (only layer 1)
 *   attention mHC prepare -> GDN or QSA -> mHC reinject
 *   MLP mHC prepare -> routed MoE + shared expert -> mHC reinject
 *
 * The staged state is invisible until commit().  Any component failure rolls
 * back QSA/GDN/PLE and the bounded expert generation together.  Decode is
 * deliberately one token per transaction; prefill/continuous batching is a
 * model-level scheduler concern and is rejected here instead of approximated. */
class decoder_layer_session final {
public:
    decoder_layer_session();
    ~decoder_layer_session();
    decoder_layer_session(decoder_layer_session &&) noexcept;
    decoder_layer_session &operator=(decoder_layer_session &&) noexcept;
    decoder_layer_session(const decoder_layer_session &) = delete;
    decoder_layer_session &operator=(const decoder_layer_session &) = delete;

    /* Start only the first PLE token's host I/O before layer 0 executes.
     * Does not open the outer decoder transaction or need hidden activations.
     * Non-PLE layers are a no-op. stage_sequence consumes the matching token;
     * rollback/reset/destruction also drain an otherwise unopened layer. */
    [[nodiscard]] decoder_layer_status prefetch_first_ple_token(
            std::int64_t token_id,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_layer_status stage_token(
            std::int64_t token_id,
            const float *residual_4x2560_f32,
            const float *full_cos_64_f32,
            const float *full_sin_64_f32,
            std::size_t position_count,
            float *output_4x2560_f32,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    /* Stage one contiguous causal token block.  Residual tensors are
     * token-major [token_count,4,2560].  QSA/GDN see one sequence, not a
     * batch of independent sessions.  The whole block is published or
     * rolled back atomically by the model transaction. */
    [[nodiscard]] decoder_layer_status stage_sequence(
            const std::int64_t *token_ids,
            std::size_t token_count,
            const float *residual_4x2560_f32,
            const float *full_cos_64_f32,
            const float *full_sin_64_f32,
            std::size_t position_count,
            float *output_4x2560_f32,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_layer_status commit(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] decoder_layer_status rollback(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] decoder_layer_status reset(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_layer_state state() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class decoder_layer;
    friend class decoder_model_transaction;
};

/* Atomic publication coordinator for a complete decoder stack.  Staging is
 * performed layer by layer on one stream.  prepare_commit() issues exactly
 * one model-wide cudaStreamSynchronize, validates every device result and all
 * transactional boundaries, and releases only ephemeral MoE pager slots.
 * It does not publish QSA, GDN or PLE history.  Any failure rolls every open
 * layer back, including layers staged before the failing layer.
 *
 * commit_prepared_noexcept() is the publication phase: PLE is published first
 * (the only fixed-shape CUDA state copy), then the already-validated QSA/GDN
 * metadata.  No inference, route, paging, allocation or synchronization is
 * permitted in this phase. */
class decoder_model_transaction final {
public:
    [[nodiscard]] static decoder_layer_status prepare_commit(
            decoder_layer_session *const *layers,
            std::size_t layer_count,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] static bool commit_prepared_noexcept(
            decoder_layer_session *const *layers,
            std::size_t layer_count) noexcept;

    [[nodiscard]] static decoder_layer_status rollback_all(
            decoder_layer_session *const *layers,
            std::size_t layer_count,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
};

}  // namespace axiom::qwen4exp

#endif
