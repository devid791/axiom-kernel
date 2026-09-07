#include "axiom/qwen4exp/ple.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <utility>

namespace axiom::qwen4exp {
namespace {

/* Oracle: huggingface/transformers
 * src/transformers/models/qwen4_exp/modular_qwen4_exp.py
 * commit fc5c5bde8e656dad91cbf34e61940d984b1c7b91. */
constexpr std::uint64_t kSplitMixGamma = 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t kSplitMixM1 = 0xbf58476d1ce4e5b9ull;
constexpr std::uint64_t kSplitMixM2 = 0x94d049bb133111ebull;
constexpr std::uint64_t kLayerPrime = 10007ull;

constexpr ple_status success() noexcept {
    return {ple_status_code::ok, "ok"};
}

constexpr ple_status failure(ple_status_code code,
                             const char *message) noexcept {
    return {code, message};
}

std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += kSplitMixGamma;
    value = (value ^ (value >> 30u)) * kSplitMixM1;
    value = (value ^ (value >> 27u)) * kSplitMixM2;
    return value ^ (value >> 31u);
}

bool is_prime(std::int64_t value) noexcept {
    if (value < 2) return false;
    if ((value & 1ll) == 0) return value == 2;
    for (std::int64_t divisor = 3;
         divisor <= value / divisor;
         divisor += 2) {
        if (value % divisor == 0) return false;
    }
    return true;
}

ple_status validate_token(const ple_metadata &metadata,
                          std::int64_t token) noexcept {
    if (token < 0 || token >= metadata.unigram_vocab_size) {
        return failure(ple_status_code::out_of_range,
                       "PLE token ID is outside the unigram vocabulary");
    }
    return success();
}

ple_status compute_ids_validated(
        const ple_metadata &metadata,
        const std::array<std::int64_t, kPleTokenContext> &left_context,
        std::int64_t token_id,
        ple_head_ids *ids) noexcept {
    if (!ids) {
        return failure(ple_status_code::invalid_argument,
                       "PLE head-ID output is null");
    }
    ple_status status = validate_token(metadata, token_id);
    if (!status) return status;
    status = validate_token(metadata, left_context[0]);
    if (!status) return status;
    status = validate_token(metadata, left_context[1]);
    if (!status) return status;

    const std::int64_t eos = metadata.eos_token_id;
    const std::int64_t shift_one =
            left_context[1] == eos ? eos : left_context[1];
    const std::int64_t shift_two =
            left_context[1] == eos || left_context[0] == eos
                    ? eos
                    : left_context[0];

    /* validate_metadata() proves every token/multiplier product is bounded by
     * INT64_MAX.  Unsigned arithmetic then expresses the bitwise XOR without
     * implementation-defined signed conversions. */
    const std::uint64_t current =
            static_cast<std::uint64_t>(token_id) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[0]);
    const std::uint64_t previous =
            static_cast<std::uint64_t>(shift_one) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[1]);
    const std::uint64_t previous_two =
            static_cast<std::uint64_t>(shift_two) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[2]);

    const std::uint64_t bigram = current ^ previous;
    const std::uint64_t trigram = bigram ^ previous_two;
    ple_head_ids result{};
    for (std::size_t head = 0; head < kPleHeadCount; ++head) {
        const std::uint64_t mixed =
                head < kPleHeadsPerNgram ? bigram : trigram;
        const std::uint64_t vocab = static_cast<std::uint64_t>(
                metadata.ngram_heads_vocab_sizes[head]);
        const std::uint64_t offset = static_cast<std::uint64_t>(
                metadata.ngram_heads_offsets[head]);
        const std::uint64_t row = offset + mixed % vocab;
        if (row > static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max())) {
            return failure(ple_status_code::arithmetic_overflow,
                           "PLE embedding row overflows I64");
        }
        result[head] = static_cast<std::int64_t>(row);
    }
    *ids = result;
    return success();
}

}  // namespace

ple_status ple_derive_layer_multipliers(
        std::int64_t unigram_vocab_size,
        std::uint64_t ple_layer_index,
        std::uint64_t seed,
        std::array<std::int64_t, kPleNgramOrder> *multipliers) noexcept {
    if (!multipliers) {
        return failure(ple_status_code::invalid_argument,
                       "PLE multiplier output is null");
    }
    if (unigram_vocab_size <= 0) {
        return failure(ple_status_code::invalid_argument,
                       "PLE unigram vocabulary must be positive");
    }

    const auto max_long = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max());
    const std::uint64_t multiplier_max =
            max_long / static_cast<std::uint64_t>(unigram_vocab_size);
    const std::uint64_t half_bound =
            std::max<std::uint64_t>(1u, multiplier_max / 2u);
    const std::uint64_t base_seed = seed + kLayerPrime * ple_layer_index;

    std::array<std::int64_t, kPleNgramOrder> result{};
    for (std::size_t index = 0; index < kPleNgramOrder; ++index) {
        const std::uint64_t value =
                base_seed + kSplitMixGamma * (index + 1u);
        const std::uint64_t multiplier =
                2u * (splitmix64(value) % half_bound) + 1u;
        if (multiplier > max_long) {
            return failure(ple_status_code::arithmetic_overflow,
                           "PLE multiplier overflows I64");
        }
        result[index] = static_cast<std::int64_t>(multiplier);
    }
    *multipliers = result;
    return success();
}

ple_status ple_derive_head_layout(
        std::int64_t ngram_vocab_size_base,
        std::uint64_t ple_layer_index,
        std::array<std::int64_t, kPleHeadCount> *offsets,
        std::array<std::int64_t, kPleHeadCount> *vocab_sizes) noexcept {
    if (!offsets || !vocab_sizes) {
        return failure(ple_status_code::invalid_argument,
                       "PLE head-layout output is null");
    }
    if (ngram_vocab_size_base < 2) {
        return failure(ple_status_code::invalid_argument,
                       "PLE n-gram vocabulary base must be at least two");
    }
    if (ple_layer_index >
        (std::numeric_limits<std::uint64_t>::max() - kPleHeadCount) /
                kPleHeadCount) {
        return failure(ple_status_code::arithmetic_overflow,
                       "PLE layer index overflows head derivation");
    }

    const std::uint64_t first = ple_layer_index * kPleHeadCount;
    const std::uint64_t required = first + kPleHeadCount;
    std::uint64_t found = 0u;
    std::size_t output_index = 0u;
    std::int64_t candidate = ngram_vocab_size_base - 1;
    std::int64_t total = 0;
    std::array<std::int64_t, kPleHeadCount> result_offsets{};
    std::array<std::int64_t, kPleHeadCount> result_sizes{};

    while (found < required) {
        if (candidate == std::numeric_limits<std::int64_t>::max()) {
            return failure(ple_status_code::arithmetic_overflow,
                           "PLE prime search overflowed I64");
        }
        ++candidate;
        if (!is_prime(candidate)) continue;
        if (found >= first) {
            if (output_index >= kPleHeadCount) {
                return failure(ple_status_code::invalid_state,
                               "PLE prime derivation produced excess heads");
            }
            result_offsets[output_index] = total;
            result_sizes[output_index] = candidate;
            if (candidate > std::numeric_limits<std::int64_t>::max() - total) {
                return failure(ple_status_code::arithmetic_overflow,
                               "PLE combined head vocabulary overflows I64");
            }
            total += candidate;
            ++output_index;
        }
        ++found;
    }
    if (output_index != kPleHeadCount) {
        return failure(ple_status_code::invalid_state,
                       "PLE prime derivation produced too few heads");
    }
    *offsets = result_offsets;
    *vocab_sizes = result_sizes;
    return success();
}

ple_status ple_validate_metadata(const ple_metadata &metadata) noexcept {
    if (metadata.unigram_vocab_size <= 0) {
        return failure(ple_status_code::invalid_metadata,
                       "PLE unigram vocabulary must be positive");
    }
    if (metadata.eos_token_id < 0 ||
        metadata.eos_token_id >= metadata.unigram_vocab_size) {
        return failure(ple_status_code::invalid_metadata,
                       "PLE EOS token is outside the unigram vocabulary");
    }

    const std::int64_t multiplier_max =
            std::numeric_limits<std::int64_t>::max() /
            metadata.unigram_vocab_size;
    for (std::int64_t multiplier : metadata.layer_multipliers) {
        if (multiplier <= 0 || (multiplier & 1ll) == 0 ||
            multiplier > multiplier_max) {
            return failure(ple_status_code::invalid_metadata,
                           "PLE layer multiplier violates the odd bounded contract");
        }
    }

    std::int64_t expected_offset = 0;
    std::int64_t previous_size = 0;
    for (std::size_t head = 0; head < kPleHeadCount; ++head) {
        const std::int64_t size = metadata.ngram_heads_vocab_sizes[head];
        if (!is_prime(size) || (head > 0u && size <= previous_size)) {
            return failure(ple_status_code::invalid_metadata,
                           "PLE head vocabulary sizes must be increasing primes");
        }
        if (metadata.ngram_heads_offsets[head] != expected_offset) {
            return failure(ple_status_code::invalid_metadata,
                           "PLE head offsets are not contiguous");
        }
        if (size > std::numeric_limits<std::int64_t>::max() -
                           expected_offset) {
            return failure(ple_status_code::invalid_metadata,
                           "PLE combined head vocabulary overflows I64");
        }
        expected_offset += size;
        previous_size = size;
    }
    return success();
}

ple_status ple_compute_head_ids(
        const ple_metadata &metadata,
        const std::array<std::int64_t, kPleTokenContext> &left_context,
        std::int64_t token_id,
        ple_head_ids *ids) noexcept {
    const ple_status status = ple_validate_metadata(metadata);
    if (!status) return status;
    return compute_ids_validated(metadata, left_context, token_id, ids);
}

ple_session_state::ple_session_state(ple_session_state &&other) noexcept {
    *this = std::move(other);
}

ple_session_state &ple_session_state::operator=(
        ple_session_state &&other) noexcept {
    if (this == &other) return *this;
    metadata_ = other.metadata_;
    configured_ = other.configured_;
    transaction_open_ = other.transaction_open_;
    conv_channels_ = other.conv_channels_;
    max_speculative_tokens_ = other.max_speculative_tokens_;
    committed_context_ = other.committed_context_;
    committed_token_count_ = other.committed_token_count_;
    committed_conv_history_ = std::move(other.committed_conv_history_);
    committed_conv_next_ = other.committed_conv_next_;
    committed_conv_valid_ = other.committed_conv_valid_;
    staged_tokens_ = std::move(other.staged_tokens_);
    staged_conv_frames_ = std::move(other.staged_conv_frames_);
    staged_conv_count_ = other.staged_conv_count_;

    other.metadata_ = {};
    other.configured_ = false;
    other.transaction_open_ = false;
    other.conv_channels_ = 0u;
    other.max_speculative_tokens_ = 0u;
    other.committed_context_.fill(0);
    other.committed_token_count_ = 0u;
    other.committed_conv_next_ = 0u;
    other.committed_conv_valid_ = 0u;
    other.staged_conv_count_ = 0u;
    return *this;
}

ple_status ple_session_state::configure(
        const ple_metadata &metadata,
        std::size_t conv_channels,
        std::size_t max_speculative_tokens) noexcept {
    if (transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "cannot reconfigure PLE during a transaction");
    }
    const ple_status metadata_status = ple_validate_metadata(metadata);
    if (!metadata_status) return metadata_status;
    if (conv_channels == 0u || max_speculative_tokens == 0u) {
        return failure(ple_status_code::invalid_argument,
                       "PLE conv channels and transaction capacity must be positive");
    }
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (conv_channels > max_size / kPleConvHistory ||
        conv_channels > max_size / kPleConvKernelSize ||
        conv_channels > max_size / max_speculative_tokens) {
        return failure(ple_status_code::arithmetic_overflow,
                       "PLE state dimensions overflow size_t");
    }

    try {
        std::vector<float> committed(
                conv_channels * kPleConvHistory, 0.0f);
        std::vector<std::int64_t> staged_tokens;
        std::vector<float> staged_frames;
        staged_tokens.reserve(max_speculative_tokens);
        staged_frames.reserve(conv_channels * max_speculative_tokens);

        metadata_ = metadata;
        conv_channels_ = conv_channels;
        max_speculative_tokens_ = max_speculative_tokens;
        committed_conv_history_.swap(committed);
        staged_tokens_.swap(staged_tokens);
        staged_conv_frames_.swap(staged_frames);
    } catch (const std::bad_alloc &) {
        return failure(ple_status_code::resource_exhausted,
                       "PLE state allocation failed");
    } catch (...) {
        return failure(ple_status_code::resource_exhausted,
                       "PLE state allocation was rejected");
    }

    configured_ = true;
    transaction_open_ = false;
    committed_context_.fill(metadata_.eos_token_id);
    committed_token_count_ = 0u;
    committed_conv_next_ = 0u;
    committed_conv_valid_ = 0u;
    staged_conv_count_ = 0u;
    return success();
}

ple_status ple_session_state::reset() noexcept {
    if (!configured_) {
        return failure(ple_status_code::invalid_state,
                       "PLE session is not configured");
    }
    if (transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "cannot reset PLE during a transaction");
    }
    committed_context_.fill(metadata_.eos_token_id);
    committed_token_count_ = 0u;
    std::fill(committed_conv_history_.begin(),
              committed_conv_history_.end(), 0.0f);
    committed_conv_next_ = 0u;
    committed_conv_valid_ = 0u;
    staged_tokens_.clear();
    staged_conv_frames_.clear();
    staged_conv_count_ = 0u;
    return success();
}

ple_status ple_session_state::begin_transaction() noexcept {
    if (!configured_) {
        return failure(ple_status_code::invalid_state,
                       "PLE session is not configured");
    }
    if (transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "PLE transaction is already open");
    }
    staged_tokens_.clear();
    staged_conv_frames_.clear();
    staged_conv_count_ = 0u;
    transaction_open_ = true;
    return success();
}

std::int64_t ple_session_state::prior_token(
        std::size_t distance) const noexcept {
    if (distance == 0u) return metadata_.eos_token_id;
    if (distance <= staged_tokens_.size()) {
        return staged_tokens_[staged_tokens_.size() - distance];
    }
    const std::size_t committed_distance = distance - staged_tokens_.size();
    if (committed_distance == 1u) return committed_context_[1];
    if (committed_distance == 2u) return committed_context_[0];
    return metadata_.eos_token_id;
}

ple_status ple_session_state::stage_token(
        std::int64_t token_id, ple_head_ids *ids) noexcept {
    if (!transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "PLE token staging requires an open transaction");
    }
    if (!ids) {
        return failure(ple_status_code::invalid_argument,
                       "PLE head-ID output is null");
    }
    if (staged_tokens_.size() != staged_conv_count_) {
        return failure(ple_status_code::invalid_state,
                       "PLE conv frame for the previous token is pending");
    }
    if (staged_tokens_.size() >= max_speculative_tokens_) {
        return failure(ple_status_code::out_of_range,
                       "PLE transaction capacity exceeded");
    }

    const std::array<std::int64_t, kPleTokenContext> context{
            prior_token(2u), prior_token(1u)};
    const ple_status status =
            compute_ids_validated(metadata_, context, token_id, ids);
    if (!status) return status;
    try {
        staged_tokens_.push_back(token_id);
    } catch (...) {
        return failure(ple_status_code::resource_exhausted,
                       "PLE token staging allocation failed");
    }
    return success();
}

const float *ple_session_state::prior_conv_frame(
        std::size_t distance, std::size_t staged_count) const noexcept {
    if (distance == 0u) return nullptr;
    if (distance <= staged_count) {
        const std::size_t index = staged_count - distance;
        return staged_conv_frames_.data() + index * conv_channels_;
    }
    const std::size_t committed_distance = distance - staged_count;
    if (committed_distance == 0u ||
        committed_distance > committed_conv_valid_) {
        return nullptr;
    }
    const std::size_t slot =
            (committed_conv_next_ + kPleConvHistory - committed_distance) %
            kPleConvHistory;
    return committed_conv_history_.data() + slot * conv_channels_;
}

ple_status ple_session_state::stage_conv_frame(
        const float *frame,
        std::size_t frame_elements,
        float *tap_window,
        std::size_t tap_window_elements) noexcept {
    if (!transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "PLE conv staging requires an open transaction");
    }
    if (!frame || !tap_window) {
        return failure(ple_status_code::invalid_argument,
                       "PLE conv input or tap output is null");
    }
    if (frame_elements != conv_channels_ ||
        tap_window_elements != conv_channels_ * kPleConvKernelSize) {
        return failure(ple_status_code::invalid_argument,
                       "PLE conv input or tap-window size is invalid");
    }
    if (staged_tokens_.size() != staged_conv_count_ + 1u) {
        return failure(ple_status_code::invalid_state,
                       "PLE conv staging has no matching pending token");
    }
    for (std::size_t channel = 0; channel < conv_channels_; ++channel) {
        if (!std::isfinite(frame[channel])) {
            return failure(ple_status_code::invalid_argument,
                           "PLE conv input contains a non-finite value");
        }
    }

    const std::size_t prior_count = staged_conv_count_;
    const std::size_t current_offset = staged_conv_frames_.size();
    try {
        staged_conv_frames_.insert(
                staged_conv_frames_.end(), frame, frame + conv_channels_);
    } catch (...) {
        return failure(ple_status_code::resource_exhausted,
                       "PLE conv staging allocation failed");
    }

    constexpr std::array<std::size_t, kPleConvKernelSize - 1u> kDistances{
            9u, 6u, 3u};
    for (std::size_t tap = 0; tap < kDistances.size(); ++tap) {
        float *destination = tap_window + tap * conv_channels_;
        const float *source = prior_conv_frame(kDistances[tap], prior_count);
        if (source) {
            std::copy_n(source, conv_channels_, destination);
        } else {
            std::fill_n(destination, conv_channels_, 0.0f);
        }
    }
    std::copy_n(staged_conv_frames_.data() + current_offset,
                conv_channels_,
                tap_window + (kPleConvKernelSize - 1u) * conv_channels_);
    ++staged_conv_count_;
    return success();
}

void ple_session_state::append_committed_conv_frame(
        const float *frame) noexcept {
    std::copy_n(frame, conv_channels_,
                committed_conv_history_.data() +
                        committed_conv_next_ * conv_channels_);
    committed_conv_next_ =
            (committed_conv_next_ + 1u) % kPleConvHistory;
    committed_conv_valid_ =
            std::min(kPleConvHistory, committed_conv_valid_ + 1u);
}

void ple_session_state::close_transaction() noexcept {
    staged_tokens_.clear();
    staged_conv_frames_.clear();
    staged_conv_count_ = 0u;
    transaction_open_ = false;
}

ple_status ple_session_state::commit_prefix(
        std::size_t accepted_tokens) noexcept {
    if (!transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "PLE commit requires an open transaction");
    }
    if (staged_tokens_.size() != staged_conv_count_) {
        return failure(ple_status_code::invalid_state,
                       "PLE commit rejected an incomplete token/conv pair");
    }
    if (accepted_tokens > staged_tokens_.size()) {
        return failure(ple_status_code::out_of_range,
                       "PLE accepted prefix exceeds the staged transaction");
    }
    if (accepted_tokens >
        std::numeric_limits<std::uint64_t>::max() - committed_token_count_) {
        return failure(ple_status_code::arithmetic_overflow,
                       "PLE committed token counter overflowed");
    }

    for (std::size_t index = 0; index < accepted_tokens; ++index) {
        committed_context_[0] = committed_context_[1];
        committed_context_[1] = staged_tokens_[index];
        append_committed_conv_frame(
                staged_conv_frames_.data() + index * conv_channels_);
    }
    committed_token_count_ += accepted_tokens;
    close_transaction();
    return success();
}

ple_status ple_session_state::rollback() noexcept {
    if (!transaction_open_) {
        return failure(ple_status_code::invalid_state,
                       "PLE rollback requires an open transaction");
    }
    close_transaction();
    return success();
}

ple_status ple_session_state::copy_committed_conv_history(
        float *output, std::size_t output_elements) const noexcept {
    if (!configured_) {
        return failure(ple_status_code::invalid_state,
                       "PLE session is not configured");
    }
    if (!output) {
        return failure(ple_status_code::invalid_argument,
                       "PLE committed-history output is null");
    }
    if (output_elements != conv_channels_ * kPleConvHistory) {
        return failure(ple_status_code::invalid_argument,
                       "PLE committed-history output size is invalid");
    }

    std::fill_n(output, output_elements, 0.0f);
    const std::size_t destination_start =
            kPleConvHistory - committed_conv_valid_;
    for (std::size_t index = 0; index < committed_conv_valid_; ++index) {
        const std::size_t distance = committed_conv_valid_ - index;
        const float *source = prior_conv_frame(distance, 0u);
        if (!source) {
            return failure(ple_status_code::invalid_state,
                           "PLE committed conv ring is inconsistent");
        }
        std::copy_n(source, conv_channels_,
                    output + (destination_start + index) * conv_channels_);
    }
    return success();
}

}  // namespace axiom::qwen4exp
