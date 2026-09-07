#ifndef AXIOM_QWEN4EXP_PLE_HPP
#define AXIOM_QWEN4EXP_PLE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace axiom::qwen4exp {

/* Qwen4-Exp PLE is fixed to two n-gram orders: eight bigram heads followed by
 * eight trigram heads.  The depthwise short convolution has four taps at
 * t-9, t-6, t-3 and t, hence nine prior frames are sufficient. */
inline constexpr std::size_t kPleNgramOrder = 3u;
inline constexpr std::size_t kPleHeadsPerNgram = 8u;
inline constexpr std::size_t kPleHeadCount = 16u;
inline constexpr std::size_t kPleTokenContext = 2u;
inline constexpr std::size_t kPleConvKernelSize = 4u;
inline constexpr std::size_t kPleConvDilation = 3u;
inline constexpr std::size_t kPleConvHistory = 9u;

using ple_head_ids = std::array<std::int64_t, kPleHeadCount>;

enum class ple_status_code : std::uint8_t {
    ok = 0,
    invalid_argument,
    invalid_metadata,
    invalid_state,
    out_of_range,
    arithmetic_overflow,
    resource_exhausted,
};

struct ple_status {
    ple_status_code code = ple_status_code::ok;
    const char *message = "ok";

    constexpr bool ok() const noexcept {
        return code == ple_status_code::ok;
    }
    constexpr explicit operator bool() const noexcept { return ok(); }
};

/* The three arrays map one-to-one to the I64 checkpoint tensors:
 *
 *   ple_embedding.layer_multipliers           [3]
 *   ple_embedding.ngram_heads_offsets         [16]
 *   ple_embedding.ngram_heads_vocab_sizes     [16]
 *
 * The embedding payload itself is intentionally outside this CPU/state API.
 */
struct ple_metadata {
    std::int64_t unigram_vocab_size = 0;
    std::int64_t eos_token_id = -1;
    std::array<std::int64_t, kPleNgramOrder> layer_multipliers{};
    std::array<std::int64_t, kPleHeadCount> ngram_heads_offsets{};
    std::array<std::int64_t, kPleHeadCount> ngram_heads_vocab_sizes{};
};

/* Exact deterministic derivation used by the official Transformers
 * Qwen4-Exp implementation.  These helpers let admission compare checkpoint
 * metadata against architecture parameters instead of trusting either one in
 * isolation. */
ple_status ple_derive_layer_multipliers(
        std::int64_t unigram_vocab_size,
        std::uint64_t ple_layer_index,
        std::uint64_t seed,
        std::array<std::int64_t, kPleNgramOrder> *multipliers) noexcept;

ple_status ple_derive_head_layout(
        std::int64_t ngram_vocab_size_base,
        std::uint64_t ple_layer_index,
        std::array<std::int64_t, kPleHeadCount> *offsets,
        std::array<std::int64_t, kPleHeadCount> *vocab_sizes) noexcept;

ple_status ple_validate_metadata(const ple_metadata &metadata) noexcept;

/* Compute the 16 embedding row IDs for one token.  left_context is ordered
 * oldest-to-newest and must contain the two immediately preceding token IDs,
 * with EOS used as left padding.  A previous EOS masks older tokens, matching
 * Transformers' _shift_right_ignore_eos semantics. */
ple_status ple_compute_head_ids(
        const ple_metadata &metadata,
        const std::array<std::int64_t, kPleTokenContext> &left_context,
        std::int64_t token_id,
        ple_head_ids *ids) noexcept;

/* Per-session CPU state for PLE hashing and the dilated short convolution.
 *
 * Committed state is immutable while a transaction is open.  Speculative
 * steps are staged separately; commit_prefix(k) publishes exactly the first k
 * token/conv pairs and discards the suffix, while rollback() discards all
 * staged state.  This object is deliberately session-owned and not internally
 * synchronized.
 *
 * stage_conv_frame() accepts the normalized gated-value frame that feeds the
 * official depthwise convolution and returns a flattened tap window in kernel
 * order [t-9, t-6, t-3, t], each tap containing conv_channels contiguous
 * floats.  It assembles state only; it does not apply convolution weights,
 * SiLU, projection weights, or the FP8 PLE embedding gather.
 */
class ple_session_state {
public:
    ple_session_state() = default;
    ~ple_session_state() = default;

    ple_session_state(const ple_session_state &) = delete;
    ple_session_state &operator=(const ple_session_state &) = delete;
    ple_session_state(ple_session_state &&other) noexcept;
    ple_session_state &operator=(ple_session_state &&other) noexcept;

    ple_status configure(const ple_metadata &metadata,
                         std::size_t conv_channels,
                         std::size_t max_speculative_tokens) noexcept;
    ple_status reset() noexcept;

    ple_status begin_transaction() noexcept;
    ple_status stage_token(std::int64_t token_id,
                           ple_head_ids *ids) noexcept;
    ple_status stage_conv_frame(const float *frame,
                                std::size_t frame_elements,
                                float *tap_window,
                                std::size_t tap_window_elements) noexcept;
    ple_status commit_prefix(std::size_t accepted_tokens) noexcept;
    ple_status rollback() noexcept;

    /* Copy the committed nine-frame state in chronological order with zero
     * left padding.  The output must contain exactly
     * kPleConvHistory * conv_channels elements. */
    ple_status copy_committed_conv_history(
            float *output, std::size_t output_elements) const noexcept;

    bool configured() const noexcept { return configured_; }
    bool transaction_open() const noexcept { return transaction_open_; }
    std::size_t conv_channels() const noexcept { return conv_channels_; }
    std::size_t conv_window_elements() const noexcept {
        return conv_channels_ * kPleConvKernelSize;
    }
    std::size_t staged_tokens() const noexcept { return staged_tokens_.size(); }
    std::uint64_t committed_tokens() const noexcept {
        return committed_token_count_;
    }
    std::size_t committed_conv_frames() const noexcept {
        return committed_conv_valid_;
    }
    std::array<std::int64_t, kPleTokenContext> committed_context() const noexcept {
        return committed_context_;
    }

private:
    std::int64_t prior_token(std::size_t distance) const noexcept;
    const float *prior_conv_frame(std::size_t distance,
                                  std::size_t staged_count) const noexcept;
    void append_committed_conv_frame(const float *frame) noexcept;
    void close_transaction() noexcept;

    ple_metadata metadata_{};
    bool configured_ = false;
    bool transaction_open_ = false;
    std::size_t conv_channels_ = 0u;
    std::size_t max_speculative_tokens_ = 0u;

    std::array<std::int64_t, kPleTokenContext> committed_context_{};
    std::uint64_t committed_token_count_ = 0u;

    /* Ring buffer of exactly kPleConvHistory frames.  next identifies the
     * slot replaced by the next committed frame. */
    std::vector<float> committed_conv_history_;
    std::size_t committed_conv_next_ = 0u;
    std::size_t committed_conv_valid_ = 0u;

    std::vector<std::int64_t> staged_tokens_;
    std::vector<float> staged_conv_frames_;
    std::size_t staged_conv_count_ = 0u;
};

}  // namespace axiom::qwen4exp

#endif
