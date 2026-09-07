#ifndef AXIOM_QWEN4EXP_SPECULATIVE_MODEL_HPP
#define AXIOM_QWEN4EXP_SPECULATIVE_MODEL_HPP

#include "axiom/qwen4exp/mtp_provider.hpp"
#include "axiom/qwen4exp/text_model.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

/* The legacy endpoint ABI remains width one for unit-test adapters.  The
 * native qwen4_exp compositor reuses that exact MTP head autoregressively and
 * verifies a bounded causal block in the target. */
inline constexpr std::size_t kSpeculativeDraftWidth = 1u;
inline constexpr std::size_t kSpeculativeMaxDraftWidth = 4u;
inline constexpr std::size_t kSpeculativeDefaultDraftWidth = 3u;

enum class speculative_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    target_error,
    draft_error,
    rollback_error,
    invalid_state,
};

enum class speculative_path : std::uint8_t {
    target_only = 0,
    draft_verified_accepted,
    draft_verified_rejected,
    draft_error_target_only,
    draft_commit_error_target_only,
};

struct speculative_model_config {
    text_model_config target{};
    mtp_provider_config draft{};

    /* When false, an unavailable or non-exact MTP graph is never admitted and
     * the exact same target model remains usable.  When true, load fails
     * closed unless the pinned qwen4_exp MTP graph is resident. */
    bool require_exact_draft = false;
};

struct speculative_model_metrics {
    text_model_metrics target{};
    mtp_contract_report draft_contract{};
    std::uint64_t draft_resident_weight_bytes = 0u;
    bool exact_draft_admitted = false;
    bool target_only_fallback = true;
};

struct speculative_session_config {
    std::size_t context_capacity = kTextModelNativeContext;
    std::size_t proposal_width = kSpeculativeDefaultDraftWidth;
    /* Explicit target-only qualification path; never silently changes output
     * model. The same admitted resident target and tokenizer are used. */
    bool enable_draft = true;
};

/* All pointers are device pointers accepted verbatim by mtp_session.  The
 * current resident_text_model public API does not expose these tensors.  A
 * caller may supply them only if it owns an equivalent native target staging
 * boundary; otherwise step() deliberately takes the target-only path. */
struct speculative_draft_inputs {
    const float *input_embedding_2560_f32 = nullptr;
    const float *target_hidden_4x2560_f32 = nullptr;
    const float *full_cos_64_f32 = nullptr;
    const float *full_sin_64_f32 = nullptr;
    std::size_t position_count = 0u;
    float *output_hidden_4x2560_f32 = nullptr;

    [[nodiscard]] bool complete() const noexcept {
        return input_embedding_2560_f32 != nullptr &&
               target_hidden_4x2560_f32 != nullptr &&
               full_cos_64_f32 != nullptr && full_sin_64_f32 != nullptr &&
               position_count != 0u &&
               output_hidden_4x2560_f32 != nullptr;
    }
};

struct speculative_proposal {
    mtp_prediction prediction{};
    bool staged = false;
};

struct speculative_step_result {
    text_token_result authoritative{};
    speculative_proposal proposal{};
    speculative_path path = speculative_path::target_only;
    bool draft_was_available = false;
    bool accepted = false;
    std::string diagnostic;
};

struct speculative_metrics {
    std::uint64_t target_attempts = 0u;
    std::uint64_t target_tokens = 0u;
    std::uint64_t drafted_tokens = 0u;
    std::uint64_t accepted_tokens = 0u;
    std::uint64_t rejected_tokens = 0u;
    std::uint64_t target_only_tokens = 0u;
    std::uint64_t primed_context_tokens = 0u;
    std::uint64_t draft_stage_failures = 0u;
    std::uint64_t draft_commit_failures = 0u;
    std::uint64_t rollback_attempts = 0u;
    std::uint64_t rollback_failures = 0u;
    std::uint64_t disconnect_rollbacks = 0u;
    std::uint64_t resets = 0u;
    std::uint64_t block_cycles = 0u;
    std::uint64_t block_verified_rows = 0u;
    std::uint64_t block_emitted_tokens = 0u;

    [[nodiscard]] double acceptance_rate() const noexcept {
        return drafted_tokens == 0u
                ? 0.0
                : static_cast<double>(accepted_tokens) /
                          static_cast<double>(drafted_tokens);
    }
};

struct speculative_session_view {
    speculative_metrics metrics{};
    bool proposal_open = false;
    bool draft_available = false;
    bool draft_synchronized = false;
    bool disconnected = false;
};

[[nodiscard]] const char *speculative_status_string(
        speculative_status status) noexcept;
[[nodiscard]] const char *speculative_path_string(
        speculative_path path) noexcept;
[[nodiscard]] speculative_status speculative_model_validate_config(
        const speculative_model_config &config) noexcept;
[[nodiscard]] speculative_status speculative_session_validate_config(
        const speculative_session_config &config) noexcept;

[[nodiscard]] constexpr bool speculative_accepts(
        std::uint32_t draft_token,
        std::uint32_t target_token) noexcept {
    return draft_token == target_token;
}

[[nodiscard]] constexpr bool speculative_exact_mtp_contract(
        const mtp_contract_report &contract) noexcept {
    return contract.tensor_count == kMtpTensorCount &&
           contract.exact_namespace && contract.all_bf16 &&
           contract.single_full_attention_layer && contract.shared_lm_head;
}

/* Small public endpoints keep the transaction policy independently testable
 * without loading 135 GB of weights.  Native adapters around text_session and
 * mtp_session are private to speculative_model.cu. */
class speculative_target_endpoint {
public:
    virtual ~speculative_target_endpoint() = default;
    [[nodiscard]] virtual text_model_status decode(
            std::uint32_t input_token,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error) noexcept = 0;
    [[nodiscard]] virtual text_model_status reset(
            std::string *error) noexcept = 0;
};

class speculative_draft_endpoint {
public:
    virtual ~speculative_draft_endpoint() = default;
    [[nodiscard]] virtual mtp_status stage(
            const speculative_draft_inputs &inputs,
            mtp_prediction *prediction,
            std::string *error) noexcept = 0;
    [[nodiscard]] virtual mtp_status commit(
            std::string *error) noexcept = 0;
    [[nodiscard]] virtual mtp_status rollback(
            std::string *error) noexcept = 0;
    [[nodiscard]] virtual mtp_status reset(
            std::string *error) noexcept = 0;
    [[nodiscard]] virtual bool transaction_open() const noexcept = 0;
};

/* Deterministic fail-closed policy shared by native and unit-test endpoints.
 * Endpoint lifetimes must exceed this object.  The staged MTP row represents
 * the authoritative input token, so successful target verification commits
 * that row on both a prediction match and mismatch.  Only target/MTP errors
 * roll it back and disable drafting. */
class speculative_transaction_driver final {
public:
    speculative_transaction_driver(
            speculative_target_endpoint *target,
            speculative_draft_endpoint *draft) noexcept
        : target_(target), draft_(draft),
          draft_enabled_(draft != nullptr),
          draft_synchronized_(draft != nullptr) {}

    ~speculative_transaction_driver() {
        if (proposal_open_ ||
            (draft_ != nullptr && draft_->transaction_open())) {
            std::string ignored;
            (void)abort_open(false, &ignored);
        }
    }

    speculative_transaction_driver(const speculative_transaction_driver &) =
            delete;
    speculative_transaction_driver &operator=(
            const speculative_transaction_driver &) = delete;

    [[nodiscard]] speculative_status target_only(
            std::uint32_t input_token,
            const text_generation_config &generation,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept {
        if (target_ == nullptr || result == nullptr || disconnected_ ||
            proposal_open_) {
            return fail(speculative_status::invalid_state, error,
                        "target-only step used an invalid transaction state");
        }
        *result = {};
        ++metrics_.target_attempts;
        text_token_result target_result{};
        std::string detail;
        const text_model_status status = target_->decode(
                input_token, generation, &target_result, &detail);
        if (status != text_model_status::ok) {
            return fail(speculative_status::target_error, error,
                        detail.empty()
                                ? "target decode failed"
                                : detail);
        }
        ++metrics_.target_tokens;
        ++metrics_.target_only_tokens;
        result->authoritative = target_result;
        result->path = speculative_path::target_only;
        result->draft_was_available = draft_enabled_;
        return speculative_status::ok;
    }

    [[nodiscard]] speculative_status propose(
            const speculative_draft_inputs &inputs,
            speculative_proposal *proposal,
            std::string *error = nullptr) noexcept {
        if (proposal == nullptr || target_ == nullptr || draft_ == nullptr ||
            !draft_enabled_ || !draft_synchronized_ || disconnected_ ||
            proposal_open_) {
            return fail(speculative_status::invalid_state, error,
                        "draft proposal is unavailable in the current state");
        }
        if (!inputs.complete()) {
            return fail(speculative_status::invalid_argument, error,
                        "draft proposal requires complete device inputs");
        }
        *proposal = {};
        mtp_prediction prediction{};
        std::string detail;
        const mtp_status status = draft_->stage(inputs, &prediction, &detail);
        if (status != mtp_status::ok) {
            ++metrics_.draft_stage_failures;
            if (draft_->transaction_open()) {
                std::string rollback_detail;
                (void)abort_open(false, &rollback_detail);
                if (!rollback_detail.empty()) {
                    if (!detail.empty()) detail += "; ";
                    detail += rollback_detail;
                }
            }
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::draft_error, error,
                        detail.empty()
                                ? "MTP stage failed"
                                : detail);
        }
        if (!draft_->transaction_open()) {
            ++metrics_.draft_stage_failures;
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::draft_error, error,
                        "MTP stage returned without an open transaction");
        }
        proposal_open_ = true;
        current_.prediction = prediction;
        current_.staged = true;
        *proposal = current_;
        return speculative_status::ok;
    }

    [[nodiscard]] speculative_status verify(
            std::uint32_t input_token,
            const text_generation_config &generation,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept {
        if (target_ == nullptr || result == nullptr || disconnected_ ||
            !proposal_open_ || !current_.staged || draft_ == nullptr) {
            return fail(speculative_status::invalid_state, error,
                        "target verification requires one open proposal");
        }
        *result = {};
        result->proposal = current_;
        result->draft_was_available = true;

        ++metrics_.target_attempts;
        text_token_result target_result{};
        std::string target_detail;
        const text_model_status target_status = target_->decode(
                input_token, generation, &target_result, &target_detail);
        if (target_status != text_model_status::ok) {
            std::string rollback_detail;
            const speculative_status rollback_status =
                    abort_open(false, &rollback_detail);
            if (!rollback_detail.empty()) {
                if (!target_detail.empty()) target_detail += "; ";
                target_detail += rollback_detail;
            }
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(rollback_status == speculative_status::ok
                                ? speculative_status::target_error
                                : speculative_status::rollback_error,
                        error,
                        target_detail.empty()
                                ? "target decode failed"
                                : target_detail);
        }

        ++metrics_.target_tokens;
        ++metrics_.drafted_tokens;
        result->authoritative = target_result;
        const bool accepted = speculative_accepts(
                current_.prediction.token_id, target_result.token_id);
        std::string commit_detail;
        const mtp_status commit_status = draft_->commit(&commit_detail);
        proposal_open_ = false;
        current_ = {};
        if (commit_status != mtp_status::ok || draft_->transaction_open()) {
            ++metrics_.draft_commit_failures;
            if (draft_->transaction_open()) {
                std::string rollback_detail;
                const speculative_status rollback_status =
                        abort_open(false, &rollback_detail);
                if (!rollback_detail.empty()) {
                    if (!commit_detail.empty()) commit_detail += "; ";
                    commit_detail += rollback_detail;
                }
                (void)rollback_status;
            }
            draft_enabled_ = false;
            draft_synchronized_ = false;
            ++metrics_.target_only_tokens;
            result->path = speculative_path::draft_commit_error_target_only;
            result->accepted = false;
            result->diagnostic = commit_detail.empty()
                    ? "MTP commit failed"
                    : commit_detail;
            return speculative_status::ok;
        }

        if (accepted) {
            ++metrics_.accepted_tokens;
            result->path = speculative_path::draft_verified_accepted;
            result->accepted = true;
        } else {
            ++metrics_.rejected_tokens;
            result->path = speculative_path::draft_verified_rejected;
            result->accepted = false;
        }
        return speculative_status::ok;
    }

    /* Prompt priming commits the MTP KV row because the staged input token is
     * authoritative prompt context.  Its prediction is not scored as an
     * accepted/rejected draft. */
    [[nodiscard]] speculative_status commit_context(
            std::string *error = nullptr) noexcept {
        if (draft_ == nullptr || !draft_enabled_ || !draft_synchronized_ ||
            disconnected_ || !proposal_open_ || !current_.staged) {
            return fail(speculative_status::invalid_state, error,
                        "MTP context commit requires one open proposal");
        }
        std::string detail;
        const mtp_status status = draft_->commit(&detail);
        proposal_open_ = false;
        current_ = {};
        if (status != mtp_status::ok || draft_->transaction_open()) {
            ++metrics_.draft_commit_failures;
            if (draft_->transaction_open()) {
                std::string rollback_detail;
                (void)abort_open(false, &rollback_detail);
                if (!rollback_detail.empty()) {
                    if (!detail.empty()) detail += "; ";
                    detail += rollback_detail;
                }
            }
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::draft_error, error,
                        detail.empty() ? "MTP context commit failed" : detail);
        }
        ++metrics_.primed_context_tokens;
        return speculative_status::ok;
    }

    [[nodiscard]] const speculative_proposal *current_proposal() const noexcept {
        return proposal_open_ && current_.staged ? &current_ : nullptr;
    }

    [[nodiscard]] speculative_status commit_open_for_block(
            speculative_proposal *proposal,
            std::string *error = nullptr) noexcept {
        if (proposal == nullptr || draft_ == nullptr || !draft_enabled_ ||
            !draft_synchronized_ || disconnected_ || !proposal_open_ ||
            !current_.staged) {
            return fail(speculative_status::invalid_state, error,
                        "MTP block root requires one open proposal");
        }
        *proposal = current_;
        std::string detail;
        const mtp_status status = draft_->commit(&detail);
        proposal_open_ = false;
        current_ = {};
        if (status != mtp_status::ok || draft_->transaction_open()) {
            ++metrics_.draft_commit_failures;
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::draft_error, error,
                        detail.empty() ? "MTP block-root commit failed" : detail);
        }
        return speculative_status::ok;
    }

    void record_native_block(std::size_t verified_rows,
                             std::size_t accepted_drafts,
                             std::size_t emitted_tokens) noexcept {
        ++metrics_.block_cycles;
        metrics_.block_verified_rows += verified_rows;
        metrics_.block_emitted_tokens += emitted_tokens;
        ++metrics_.target_attempts;
        metrics_.target_tokens += verified_rows;
        metrics_.drafted_tokens += verified_rows;
        metrics_.accepted_tokens += accepted_drafts;
        if (accepted_drafts < verified_rows) ++metrics_.rejected_tokens;
    }

    [[nodiscard]] speculative_status step(
            std::uint32_t input_token,
            const text_generation_config &generation,
            const speculative_draft_inputs *draft_inputs,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept {
        if (result == nullptr) {
            return fail(speculative_status::invalid_argument, error,
                        "speculative step output is null");
        }
        if (draft_inputs == nullptr || !draft_enabled_ ||
            !draft_synchronized_) {
            return target_only(input_token, generation, result, error);
        }

        speculative_proposal proposal{};
        std::string proposal_detail;
        const speculative_status proposal_status =
                propose(*draft_inputs, &proposal, &proposal_detail);
        if (proposal_status == speculative_status::ok) {
            return verify(input_token, generation, result, error);
        }

        const speculative_status target_status =
                target_only(input_token, generation, result, error);
        if (target_status == speculative_status::ok) {
            result->proposal = proposal;
            result->draft_was_available = true;
            result->path = speculative_path::draft_error_target_only;
            result->diagnostic = proposal_detail;
        }
        return target_status;
    }

    [[nodiscard]] speculative_status disconnect(
            std::string *error = nullptr) noexcept {
        if (disconnected_) return speculative_status::ok;
        speculative_status status = speculative_status::ok;
        if (proposal_open_ ||
            (draft_ != nullptr && draft_->transaction_open())) {
            ++metrics_.disconnect_rollbacks;
            status = abort_open(true, error);
        }
        disconnected_ = true;
        draft_enabled_ = false;
        draft_synchronized_ = false;
        return status;
    }

    [[nodiscard]] speculative_status reset(
            std::string *error = nullptr) noexcept {
        if (target_ == nullptr) {
            return fail(speculative_status::invalid_state, error,
                        "speculative target endpoint is null");
        }
        std::string detail;
        if (proposal_open_ ||
            (draft_ != nullptr && draft_->transaction_open())) {
            const speculative_status rollback = abort_open(false, &detail);
            if (rollback != speculative_status::ok) {
                disconnected_ = true;
                draft_enabled_ = false;
                draft_synchronized_ = false;
                return fail(rollback, error, detail);
            }
        }
        std::string target_detail;
        const text_model_status target_status = target_->reset(&target_detail);
        std::string draft_detail;
        const mtp_status draft_status = draft_ == nullptr
                ? mtp_status::invalid_state
                : draft_->reset(&draft_detail);
        ++metrics_.resets;
        disconnected_ = false;
        if (target_status != text_model_status::ok) {
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::target_error, error,
                        target_detail.empty()
                                ? "target reset failed"
                                : target_detail);
        }
        draft_enabled_ = draft_ != nullptr && draft_status == mtp_status::ok;
        draft_synchronized_ = draft_enabled_;
        if (draft_ != nullptr && draft_status != mtp_status::ok &&
            error != nullptr) {
            *error = draft_detail.empty()
                    ? "MTP reset failed"
                    : draft_detail;
        }
        return speculative_status::ok;
    }

    void disable_draft() noexcept {
        if (proposal_open_ ||
            (draft_ != nullptr && draft_->transaction_open())) {
            std::string ignored;
            (void)abort_open(false, &ignored);
        }
        draft_enabled_ = false;
        draft_synchronized_ = false;
    }

    [[nodiscard]] speculative_session_view view() const noexcept {
        speculative_session_view result{};
        result.metrics = metrics_;
        result.proposal_open = proposal_open_;
        result.draft_available = draft_enabled_;
        result.draft_synchronized = draft_synchronized_;
        result.disconnected = disconnected_;
        return result;
    }

private:
    [[nodiscard]] static speculative_status fail(
            speculative_status status,
            std::string *error,
            const std::string &message) noexcept {
        if (error != nullptr) *error = message;
        return status;
    }

    [[nodiscard]] speculative_status abort_open(
            bool disconnect,
            std::string *error) noexcept {
        if (!proposal_open_ &&
            (draft_ == nullptr || !draft_->transaction_open())) {
            return speculative_status::ok;
        }
        ++metrics_.rollback_attempts;
        std::string detail;
        const mtp_status status = draft_ == nullptr
                ? mtp_status::invalid_state
                : draft_->rollback(&detail);
        proposal_open_ = false;
        current_ = {};
        if (status != mtp_status::ok) {
            ++metrics_.rollback_failures;
            draft_enabled_ = false;
            draft_synchronized_ = false;
            return fail(speculative_status::rollback_error, error,
                        detail.empty()
                                ? (disconnect
                                           ? "disconnect rollback failed"
                                           : "draft rollback failed")
                                : detail);
        }
        return speculative_status::ok;
    }

    speculative_target_endpoint *target_ = nullptr;
    speculative_draft_endpoint *draft_ = nullptr;
    speculative_proposal current_{};
    speculative_metrics metrics_{};
    bool proposal_open_ = false;
    bool draft_enabled_ = false;
    bool draft_synchronized_ = false;
    bool disconnected_ = false;
};

class speculative_session;

class speculative_model final {
public:
    speculative_model();
    ~speculative_model();
    speculative_model(speculative_model &&) noexcept;
    speculative_model &operator=(speculative_model &&) noexcept;
    speculative_model(const speculative_model &) = delete;
    speculative_model &operator=(const speculative_model &) = delete;

    [[nodiscard]] static speculative_status load(
            const std::string &model_root,
            const speculative_model_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<speculative_model> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] speculative_status create_session(
            const speculative_session_config &config,
            cudaStream_t stream,
            std::unique_ptr<speculative_session> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] const speculative_model_config &config() const noexcept;
    [[nodiscard]] const speculative_model_metrics &metrics() const noexcept;
    [[nodiscard]] expert_slot_arena_metrics expert_cache_metrics() const noexcept;
    /* Target-only host pager aggregate; inherits resident_text_model's
     * thread-safe snapshot/quiescent-boundary and sum-of-layer-peaks semantics.
     * No payload retention or GPU allocation/synchronization. */
    [[nodiscard]] ExpertPagerMetrics host_expert_cache_metrics() const noexcept;
    [[nodiscard]] const std::string &draft_diagnostic() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

class speculative_session final {
public:
    speculative_session();
    ~speculative_session();
    speculative_session(speculative_session &&) noexcept;
    speculative_session &operator=(speculative_session &&) noexcept;
    speculative_session(const speculative_session &) = delete;
    speculative_session &operator=(const speculative_session &) = delete;

    [[nodiscard]] speculative_status target_prefill(
            const std::uint32_t *token_ids,
            std::size_t token_count,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status target_prefill_multimodal(
            const multimodal_text_prompt_view &prompt,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status propose(
            const speculative_draft_inputs &inputs,
            speculative_proposal *proposal,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status verify(
            std::uint32_t input_token,
            const text_generation_config &generation,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status step(
            std::uint32_t input_token,
            const text_generation_config &generation,
            const speculative_draft_inputs *draft_inputs,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept;
    /* Native continuous path.  It verifies the pending MTP proposal against
     * the authoritative target token, commits the MTP context row on either
     * match or mismatch, then stages the next proposal from the new target
     * hidden state.  Target output is never substituted by an unverified
     * draft. */
    [[nodiscard]] speculative_status decode(
            std::uint32_t input_token,
            const text_generation_config &generation,
            speculative_step_result *result,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status disconnect(
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_status reset(
            std::string *error = nullptr) noexcept;
    [[nodiscard]] speculative_session_view view() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    [[nodiscard]] speculative_status stage_native_proposal(
            std::uint32_t next_input_token,
            const float *projected_visual_embedding,
            const std::uint32_t *mrope_positions_t_h_w,
            std::string *error) noexcept;

    struct impl;
    std::unique_ptr<impl> impl_;
    friend class speculative_model;
};

}  // namespace axiom::qwen4exp

#endif
