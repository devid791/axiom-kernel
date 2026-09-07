#ifndef AXIOM_QWEN4EXP_PLE_EMBEDDING_HPP
#define AXIOM_QWEN4EXP_PLE_EMBEDDING_HPP

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace axiom::qwen4exp {

inline constexpr std::size_t kPleEmbeddingShardCount = 128u;
inline constexpr std::size_t kPleEmbeddingRowsPerShard = 2500012u;
inline constexpr std::size_t kPleEmbeddingWidth = 160u;
inline constexpr std::uint64_t kPleEmbeddingRows =
        static_cast<std::uint64_t>(kPleEmbeddingShardCount) *
        static_cast<std::uint64_t>(kPleEmbeddingRowsPerShard);
inline constexpr std::size_t kPleEmbeddingOutputElements =
        kPleHeadCount * kPleEmbeddingWidth;

enum class ple_embedding_status_code : std::uint8_t {
    ok = 0,
    invalid_argument,
    invalid_checkpoint,
    out_of_range,
    io_error,
    non_finite,
    resource_exhausted,
};

struct ple_embedding_status {
    ple_embedding_status_code code = ple_embedding_status_code::ok;
    std::string message = "ok";

    bool ok() const noexcept { return code == ple_embedding_status_code::ok; }
    explicit operator bool() const noexcept { return ok(); }
};

/* Decode one finite E4M3FN byte.  NaN encodings produce NaN; signed zero is
 * preserved.  This function is shared by checkpoint gates and later CUDA
 * staging code so the on-disk contract has one explicit definition. */
float ple_e4m3fn_to_f32(std::uint8_t encoded) noexcept;

/* Small, session-owned four-way set-associative row cache.  It is intentionally bounded
 * and contains dequantized rows only; the OS page cache remains the larger
 * NVMe->RAM tier.  No synchronization is performed inside the cache. */
class ple_embedding_row_cache final {
public:
    ple_embedding_status configure(std::size_t slots) noexcept;
    void clear() noexcept;

    bool lookup(std::uint64_t row_id, float *row,
                std::size_t row_elements) noexcept;
    ple_embedding_status insert(std::uint64_t row_id, const float *row,
                                std::size_t row_elements) noexcept;

    std::size_t slots() const noexcept { return entries_.size(); }
    std::uint64_t hits() const noexcept { return hits_; }
    std::uint64_t misses() const noexcept { return misses_; }

private:
    struct entry {
        std::uint64_t row_id = 0;
        std::uint64_t stamp = 0;
        bool valid = false;
        std::array<float, kPleEmbeddingWidth> values{};
    };

    std::vector<entry> entries_;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::uint64_t clock_ = 0;
};

/* Immutable range reader for the pinned FP8 PLE table.  configure() validates
 * all 128 Safetensors spans and the non-unit BF16 calibration scalar.  A
 * gather performs 16 bounded row reads at most and never materializes a shard
 * or the complete 51.2 GB embedding.  The catalog must outlive this object. */
class ple_embedding_reader final {
public:
    ple_embedding_status configure(
            const checkpoint_catalog *catalog) noexcept;

    ple_embedding_status gather_f32(
            const ple_head_ids &row_ids,
            float *output,
            std::size_t output_elements,
            ple_embedding_row_cache *cache = nullptr) const noexcept;

    bool configured() const noexcept { return catalog_ != nullptr; }
    float scale() const noexcept { return scale_; }
    std::uint64_t total_rows() const noexcept { return kPleEmbeddingRows; }

private:
    ple_embedding_status read_row_f32(
            std::uint64_t row_id,
            float *row,
            std::size_t row_elements) const noexcept;

    const checkpoint_catalog *catalog_ = nullptr;
    float scale_ = 0.0f;
    std::array<std::string, kPleEmbeddingShardCount> shard_names_{};
};

}  // namespace axiom::qwen4exp

#endif
