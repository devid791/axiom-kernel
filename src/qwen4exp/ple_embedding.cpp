#include "axiom/qwen4exp/ple_embedding.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace axiom::qwen4exp {
namespace {

ple_embedding_status status(ple_embedding_status_code code,
                            std::string message) noexcept {
    ple_embedding_status result;
    result.code = code;
    try {
        result.message = std::move(message);
    } catch (...) {
        result.code = ple_embedding_status_code::resource_exhausted;
        result.message = "resource exhausted while reporting PLE embedding status";
    }
    return result;
}

float bf16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline constexpr std::size_t kCacheWays = 4u;

std::size_t cache_set(std::uint64_t row_id, std::size_t slots) noexcept {
    row_id ^= row_id >> 30u;
    row_id *= 0xbf58476d1ce4e5b9ULL;
    row_id ^= row_id >> 27u;
    row_id *= 0x94d049bb133111ebULL;
    row_id ^= row_id >> 31u;
    const std::size_t sets = slots / kCacheWays;
    return static_cast<std::size_t>(row_id % static_cast<std::uint64_t>(sets));
}

}  // namespace

float ple_e4m3fn_to_f32(std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x80u) != 0u;
    const unsigned exponent = (encoded >> 3u) & 0x0fu;
    const unsigned mantissa = encoded & 0x07u;
    if (exponent == 15u && mantissa == 7u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float value = 0.0f;
    if (exponent == 0u) {
        value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                           static_cast<int>(exponent) - 7);
    }
    return negative ? -value : value;
}

ple_embedding_status ple_embedding_row_cache::configure(
        std::size_t slots) noexcept {
    if (slots < kCacheWays || slots % kCacheWays != 0u) {
        return status(ple_embedding_status_code::invalid_argument,
                      "PLE row cache slots must be a non-zero multiple of four");
    }
    try {
        entries_.assign(slots, entry{});
    } catch (const std::bad_alloc &) {
        return status(ple_embedding_status_code::resource_exhausted,
                      "could not allocate bounded PLE row cache");
    } catch (...) {
        return status(ple_embedding_status_code::resource_exhausted,
                      "unexpected failure allocating PLE row cache");
    }
    hits_ = 0;
    misses_ = 0;
    clock_ = 0;
    return {};
}

void ple_embedding_row_cache::clear() noexcept {
    for (entry &item : entries_) item.valid = false;
    hits_ = 0;
    misses_ = 0;
    clock_ = 0;
}

bool ple_embedding_row_cache::lookup(std::uint64_t row_id,
                                     float *row,
                                     std::size_t row_elements) noexcept {
    if (row == nullptr || row_elements != kPleEmbeddingWidth || entries_.empty()) {
        return false;
    }
    const std::size_t base = cache_set(row_id, entries_.size()) * kCacheWays;
    for (std::size_t way = 0; way < kCacheWays; ++way) {
        entry &item = entries_[base + way];
        if (item.valid && item.row_id == row_id) {
            item.stamp = ++clock_;
            std::copy(item.values.begin(), item.values.end(), row);
            ++hits_;
            return true;
        }
    }
    ++misses_;
    return false;
}

ple_embedding_status ple_embedding_row_cache::insert(
        std::uint64_t row_id,
        const float *row,
        std::size_t row_elements) noexcept {
    if (row == nullptr || row_elements != kPleEmbeddingWidth || entries_.empty()) {
        return status(ple_embedding_status_code::invalid_argument,
                      "invalid PLE row cache insertion");
    }
    const std::size_t base = cache_set(row_id, entries_.size()) * kCacheWays;
    entry *victim = &entries_[base];
    for (std::size_t way = 0; way < kCacheWays; ++way) {
        entry &candidate = entries_[base + way];
        if (!candidate.valid) {
            victim = &candidate;
            break;
        }
        if (candidate.stamp < victim->stamp) victim = &candidate;
    }
    entry &item = *victim;
    item.row_id = row_id;
    item.stamp = ++clock_;
    item.valid = true;
    std::copy(row, row + kPleEmbeddingWidth, item.values.begin());
    return {};
}

ple_embedding_status ple_embedding_reader::configure(
        const checkpoint_catalog *catalog) noexcept {
    catalog_ = nullptr;
    scale_ = 0.0f;
    if (catalog == nullptr) {
        return status(ple_embedding_status_code::invalid_argument,
                      "null checkpoint catalog for PLE embedding");
    }

    try {
        for (std::size_t shard = 0; shard < kPleEmbeddingShardCount; ++shard) {
            shard_names_[shard] =
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_" +
                    std::to_string(shard) + ".weight";
            const tensor_span *span = catalog->find(shard_names_[shard]);
            const std::uint64_t expected_bytes =
                    static_cast<std::uint64_t>(kPleEmbeddingRowsPerShard) *
                    static_cast<std::uint64_t>(kPleEmbeddingWidth);
            if (span == nullptr || span->dtype != tensor_dtype::f8_e4m3 ||
                span->rank != 2u ||
                span->shape[0] != kPleEmbeddingRowsPerShard ||
                span->shape[1] != kPleEmbeddingWidth ||
                span->bytes != expected_bytes) {
                return status(ple_embedding_status_code::invalid_checkpoint,
                              "invalid PLE FP8 shard descriptor: " + shard_names_[shard]);
            }
        }
    } catch (const std::bad_alloc &) {
        return status(ple_embedding_status_code::resource_exhausted,
                      "could not construct PLE shard catalog");
    } catch (...) {
        return status(ple_embedding_status_code::invalid_checkpoint,
                      "unexpected PLE shard catalog failure");
    }

    constexpr char scale_name[] =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
    const tensor_span *scale_span = catalog->find(scale_name);
    if (scale_span == nullptr || scale_span->dtype != tensor_dtype::bf16 ||
        scale_span->rank != 1u || scale_span->shape[0] != 1u ||
        scale_span->bytes != sizeof(std::uint16_t)) {
        return status(ple_embedding_status_code::invalid_checkpoint,
                      "invalid PLE BF16 scale descriptor");
    }
    std::uint16_t scale_bits = 0;
    std::string read_error;
    if (!catalog->read_range(scale_name, 0, &scale_bits, sizeof(scale_bits),
                             &read_error)) {
        return status(ple_embedding_status_code::io_error,
                      "could not read PLE BF16 scale: " + read_error);
    }
    const float decoded_scale = bf16_to_f32(scale_bits);
    if (!std::isfinite(decoded_scale) || decoded_scale <= 0.0f ||
        decoded_scale >= 0.001f) {
        return status(ple_embedding_status_code::invalid_checkpoint,
                      "PLE BF16 scale is non-finite or outside calibrated bounds");
    }

    catalog_ = catalog;
    scale_ = decoded_scale;
    return {};
}

ple_embedding_status ple_embedding_reader::read_row_f32(
        std::uint64_t row_id,
        float *row,
        std::size_t row_elements) const noexcept {
    if (catalog_ == nullptr || row == nullptr ||
        row_elements != kPleEmbeddingWidth) {
        return status(ple_embedding_status_code::invalid_argument,
                      "invalid PLE embedding row read");
    }
    if (row_id >= kPleEmbeddingRows) {
        return status(ple_embedding_status_code::out_of_range,
                      "PLE embedding row id is out of range");
    }
    const std::size_t shard = static_cast<std::size_t>(
            row_id / static_cast<std::uint64_t>(kPleEmbeddingRowsPerShard));
    const std::uint64_t local_row =
            row_id % static_cast<std::uint64_t>(kPleEmbeddingRowsPerShard);
    const std::uint64_t byte_offset =
            local_row * static_cast<std::uint64_t>(kPleEmbeddingWidth);
    std::array<std::uint8_t, kPleEmbeddingWidth> encoded{};
    std::string read_error;
    if (!catalog_->read_range(shard_names_[shard], byte_offset,
                              encoded.data(), encoded.size(), &read_error)) {
        return status(ple_embedding_status_code::io_error,
                      "could not read PLE embedding row: " + read_error);
    }
    for (std::size_t column = 0; column < kPleEmbeddingWidth; ++column) {
        const float value = ple_e4m3fn_to_f32(encoded[column]);
        if (!std::isfinite(value)) {
            return status(ple_embedding_status_code::non_finite,
                          "PLE embedding contains an E4M3FN NaN encoding");
        }
        row[column] = value * scale_;
    }
    return {};
}

ple_embedding_status ple_embedding_reader::gather_f32(
        const ple_head_ids &row_ids,
        float *output,
        std::size_t output_elements,
        ple_embedding_row_cache *cache) const noexcept {
    if (catalog_ == nullptr || output == nullptr ||
        output_elements != kPleEmbeddingOutputElements) {
        return status(ple_embedding_status_code::invalid_argument,
                      "invalid PLE embedding gather output");
    }
    std::array<float, kPleEmbeddingWidth> row{};
    for (std::size_t head = 0; head < kPleHeadCount; ++head) {
        if (row_ids[head] < 0 ||
            static_cast<std::uint64_t>(row_ids[head]) >= kPleEmbeddingRows) {
            return status(ple_embedding_status_code::out_of_range,
                          "PLE embedding gather row id is out of range");
        }
        const std::uint64_t row_id = static_cast<std::uint64_t>(row_ids[head]);
        const bool cached = cache != nullptr &&
                cache->lookup(row_id, row.data(), row.size());
        if (!cached) {
            ple_embedding_status read_status =
                    read_row_f32(row_id, row.data(), row.size());
            if (!read_status) return read_status;
            if (cache != nullptr) {
                ple_embedding_status cache_status =
                        cache->insert(row_id, row.data(), row.size());
                if (!cache_status) return cache_status;
            }
        }
        std::copy(row.begin(), row.end(),
                  output + head * kPleEmbeddingWidth);
    }
    return {};
}

}  // namespace axiom::qwen4exp
