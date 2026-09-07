#ifndef AXIOM_QWEN4EXP_TEXT_MODEL_HPP
#define AXIOM_QWEN4EXP_TEXT_MODEL_HPP

#include "axiom/qwen4exp/decoder_residency.hpp"
#include "axiom/qwen4exp/expert_bridge.hpp"
#include "axiom/qwen4exp/multimodal_text.hpp"
#include "axiom/qwen4exp/output_head.hpp"
#include "axiom/qwen4exp/rope.hpp"
#include "axiom/qwen4exp/session_state.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kTextModelLayerCount = 48u;
inline constexpr std::size_t kTextModelNativeContext = 262144u;
inline constexpr std::size_t kTextModelVocab = 248320u;

enum class text_model_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    unsupported_device,
    admission_error,
    checkpoint_error,
    allocation_failure,
    residency_error,
    decoder_layer_error,
    output_head_error,
    rope_error,
    session_error,
    capacity_exceeded,
    token_rejected,
    transaction_error,
    cuda_error,
    invalid_state,
};

enum class text_selection_mode : std::uint8_t {
    argmax = 0,
    sample,
};

struct text_model_config {
    /* The admitted Flash-Next checkpoint is native 262K.  A different model
     * context is a different provider contract and is rejected here. */
    std::size_t max_context = kTextModelNativeContext;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
    std::uint64_t moe_ram_capacity_bytes_per_layer =
            64ull * 1024ull * 1024ull;
    std::size_t moe_prefetch_workers_per_layer = 2u;
    /* Zero preserves the transient top10 arena. Nonzero explicitly enables
     * a checkpoint/layer/expert-keyed GPU cache shared by all decoder layers. */
    std::uint32_t expert_gpu_cache_slots = 0u;
    /* Optional exclusive cold expert cache shared by all layers. A zero
     * budget preserves the existing independent host pager behavior. */
    std::uint64_t expert_host_cold_cache_bytes = 0u;
    std::uint64_t decoder_hard_limit_bytes = kDecoderResidencyHardLimit;
    std::uint64_t decoder_safety_margin_bytes =
            kDecoderResidencyDefaultMargin;
    std::uint64_t decoder_adapter_allowance_bytes_per_layer =
            40ull * 1024ull * 1024ull;

    /* Full verification hashes all 206 pinned weight shards.  It is enabled
     * by default so a successful load proves payload identity, not only JSON
     * metadata and Safetensors structure. */
    bool verify_payload_hashes = true;
    bool moe_owned_direct_pread = false;
};

struct text_model_metrics {
    double load_ms = 0.0;
    std::size_t admitted_layers = 0u;
    std::size_t loaded_layers = 0u;
    std::uint64_t output_head_resident_bytes = 0u;
    std::uint64_t expert_arena_resident_bytes = 0u;
    std::uint64_t checkpoint_payload_bytes = 0u;
    bool payload_hashes_verified = false;
    decoder_residency_metrics decoder_residency{};
};

struct text_session_config {
    /* Physical CUDA state is bounded to this value and may never exceed the
     * model's native 262K limit. */
    std::size_t context_capacity = kTextModelNativeContext;
};

struct text_generation_config {
    text_selection_mode mode = text_selection_mode::argmax;
    output_head_sampling_config sampling{};
};

struct text_token_result {
    std::uint32_t token_id = 0u;
    float selected_logit = 0.0F;
    std::uint64_t committed_context = 0u;
    double latency_ms = 0.0;
    bool sampled = false;
};

/* Stable device views exported only across the native target/MTP handoff.
 * MTP position zero corresponds to target position one, hence the RoPE table
 * pointers are already shifted by one row and position_count is the visible
 * MTP length.  The views remain valid until the next text_session operation
 * on the same stream. */
struct text_mtp_inputs_view {
    const float *input_embedding_2560_f32 = nullptr;
    const float *target_hidden_4x2560_f32 = nullptr;
    const float *full_cos_64_f32 = nullptr;
    const float *full_sin_64_f32 = nullptr;
    std::size_t position_count = 0u;

    [[nodiscard]] bool complete() const noexcept {
        return input_embedding_2560_f32 != nullptr &&
               target_hidden_4x2560_f32 != nullptr &&
               full_cos_64_f32 != nullptr && full_sin_64_f32 != nullptr &&
               position_count != 0u;
    }
};

struct text_session_metrics {
    std::uint64_t prompt_input_tokens = 0u;
    std::uint64_t prompt_visual_tokens = 0u;
    std::uint64_t decode_input_tokens = 0u;
    std::uint64_t generated_tokens = 0u;
    std::uint64_t committed_transactions = 0u;
    std::uint64_t rolled_back_transactions = 0u;
    std::uint64_t resets = 0u;
    double ttft_ms = 0.0;
    double last_decode_ms = 0.0;
    double total_decode_ms = 0.0;
    bool ttft_measured = false;
};

struct text_session_view {
    model_session_view decoder{};
    text_session_metrics metrics{};
    std::uint32_t next_mrope_text_position = 0u;
    bool multimodal_mrope_active = false;
    bool next_mrope_text_position_available = false;
    bool initialized = false;
};

[[nodiscard]] const char *text_model_status_string(
        text_model_status status) noexcept;
[[nodiscard]] text_model_status text_model_validate_config(
        const text_model_config &config) noexcept;
[[nodiscard]] text_model_status text_session_validate_config(
        const text_session_config &config) noexcept;

class text_session;
class speculative_session;

/* Immutable complete text model for the exact pinned qwen4_exp checkpoint:
 * embedding + a shared bounded residency pool for all 48 decoder layers +
 * global hyper mixer + untied LM head.  The object owns the native RoPE plan
 * and no Python runtime is involved.
 *
 * Output-head execution is serialized inside the model because the resident
 * tensor-core tail owns one cuBLAS execution context.  Decoder/KV/PLE/GDN
 * state remains independent per text_session. */
class resident_text_model final {
public:
    resident_text_model();
    ~resident_text_model();
    resident_text_model(resident_text_model &&) noexcept;
    resident_text_model &operator=(resident_text_model &&) noexcept;
    resident_text_model(const resident_text_model &) = delete;
    resident_text_model &operator=(const resident_text_model &) = delete;

    [[nodiscard]] static text_model_status load(
            const std::string &model_root,
            const text_model_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<resident_text_model> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] text_model_status create_session(
            const text_session_config &config,
            cudaStream_t stream,
            std::unique_ptr<text_session> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] const text_model_config &config() const noexcept;
    [[nodiscard]] const text_model_metrics &metrics() const noexcept;
    [[nodiscard]] expert_slot_arena_metrics expert_cache_metrics() const noexcept;
    /* Sum of pager snapshots from valid resident decoder layers (up to 48).
     * All counters, bytes, limits, reservations and active counts are added.
     * peak_ram_bytes and peak_direct_staging_bytes are SUMS OF PER-LAYER
     * PEAKS, not observed simultaneous/global peaks. Staging reservations
     * remain separate from cached RAM bytes; neither measures process RSS.
     * Each pager snapshot is thread-safe, but the aggregate is not globally
     * atomic during activity; use a quiescent boundary for aligned deltas.
     * Bounded stack-only snapshots retain no payload and allocate/synchronize
     * no GPU resources. Model lifetime must be stable (no move/destruction).
     * Missing/uninitialized layers contribute zero. */
    [[nodiscard]] ExpertPagerMetrics host_expert_cache_metrics() const noexcept;
    [[nodiscard]] const std::string &model_root() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class text_session;
};

/* Synchronous transaction boundary around asynchronous CUDA work.  Each
 * input token is atomically staged through all 48 decoder layers.  Prefill
 * emits only the final next-token result; intermediate prompt tokens never
 * execute the LM head.  decode() consumes exactly one token and emits exactly
 * one next-token result.  Any failure rolls the currently open model-wide
 * transaction back before returning. */
class text_session final {
public:
    text_session();
    ~text_session();
    text_session(text_session &&) noexcept;
    text_session &operator=(text_session &&) noexcept;
    text_session(const text_session &) = delete;
    text_session &operator=(const text_session &) = delete;

    [[nodiscard]] text_model_status prefill(
            const std::uint32_t *token_ids,
            std::size_t token_count,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error = nullptr) noexcept;

    /* Processor-expanded multimodal prefill.  Projected rows come directly
     * from the native vision adapter and replace image/video placeholder
     * embeddings one-for-one.  The exact axis-major MRoPE positions are
     * installed transactionally for every prompt token; subsequent decode
     * continues at max(MRoPE)+1. */
    [[nodiscard]] text_model_status prefill_multimodal(
            const multimodal_text_prompt_view &prompt,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] text_model_status decode(
            std::uint32_t input_token,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] text_model_status reset(
            std::string *error = nullptr) noexcept;

    [[nodiscard]] text_session_view view() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    [[nodiscard]] text_model_status prefill_token_for_speculation(
            std::uint32_t token,
            const float *projected_visual_embedding,
            const std::uint32_t *mrope_positions_t_h_w,
            bool final,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error) noexcept;

    [[nodiscard]] text_model_status prepare_mtp_inputs(
            std::uint32_t next_input_token,
            const float *projected_visual_embedding,
            const std::uint32_t *mrope_positions_t_h_w,
            text_mtp_inputs_view *view,
            std::string *error) noexcept;

    [[nodiscard]] text_model_status prepare_mtp_token_embedding(
            std::uint32_t token,
            const float **embedding_2560_f32,
            std::string *error) noexcept;

    [[nodiscard]] text_model_status stage_decode_sequence_for_speculation(
            const std::uint32_t *input_tokens,
            std::size_t token_count,
            text_token_result *next_tokens,
            std::string *error) noexcept;

    [[nodiscard]] text_model_status commit_staged_speculative_sequence(
            std::string *error) noexcept;
    [[nodiscard]] text_model_status rollback_staged_speculative_sequence(
            std::string *error) noexcept;

    [[nodiscard]] text_model_status finish_speculative_multimodal_prefill(
            std::uint32_t next_text_position,
            bool next_text_position_available,
            std::string *error) noexcept;

    struct impl;
    std::unique_ptr<impl> impl_;
    friend class resident_text_model;
    friend class speculative_session;
};

}  // namespace axiom::qwen4exp

#endif
