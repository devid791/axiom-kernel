#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple.hpp"
#include "axiom/qwen4exp/ple_embedding.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>

namespace q4 = axiom::qwen4exp;

namespace {

bool check(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "qwen4exp-ple-embedding-test: %s\n", message);
    return condition;
}

std::uint64_t rss_bytes() {
    std::ifstream stream("/proc/self/statm");
    std::uint64_t pages = 0;
    std::uint64_t resident = 0;
    if (!(stream >> pages >> resident)) return 0;
    return resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
}

float independent_e4m3(std::uint8_t encoded) {
    const int sign = (encoded & 0x80u) != 0u ? -1 : 1;
    const int exponent = static_cast<int>((encoded >> 3u) & 0x0fu);
    const int mantissa = static_cast<int>(encoded & 0x07u);
    if (exponent == 15 && mantissa == 7) return std::nanf("");
    const float magnitude = exponent == 0
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                         exponent - 7);
    return static_cast<float>(sign) * magnitude;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    bool ok = true;
    ok = check(q4::ple_e4m3fn_to_f32(0x00u) == 0.0f,
               "E4M3 zero decode mismatch") && ok;
    ok = check(q4::ple_e4m3fn_to_f32(0x01u) == std::ldexp(1.0f, -9),
               "E4M3 minimum subnormal mismatch") && ok;
    ok = check(q4::ple_e4m3fn_to_f32(0x7eu) == 448.0f,
               "E4M3 maximum finite mismatch") && ok;
    ok = check(q4::ple_e4m3fn_to_f32(0xfeu) == -448.0f,
               "E4M3 negative maximum mismatch") && ok;
    ok = check(std::isnan(q4::ple_e4m3fn_to_f32(0x7fu)),
               "E4M3 NaN encoding was accepted as finite") && ok;

    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    if (!q4::checkpoint_catalog::open(argv[1], &catalog, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    q4::ple_embedding_reader reader;
    const q4::ple_embedding_status configure_status = reader.configure(catalog.get());
    if (!configure_status) {
        std::fprintf(stderr, "%s\n", configure_status.message.c_str());
        return 1;
    }
    ok = check(reader.total_rows() == 320001536ull,
               "unexpected PLE total row count") && ok;
    ok = check(std::isfinite(reader.scale()) && reader.scale() > 0.0f &&
                       reader.scale() < 0.001f,
               "invalid PLE dequant scale") && ok;

    q4::ple_metadata metadata;
    metadata.unigram_vocab_size = 248320;
    metadata.eos_token_id = 248044;
    ok = check(q4::ple_derive_layer_multipliers(
                       248320, 0u, 1234u, &metadata.layer_multipliers).ok(),
               "could not derive PLE multipliers") && ok;
    ok = check(q4::ple_derive_head_layout(
                       20000000, 0u, &metadata.ngram_heads_offsets,
                       &metadata.ngram_heads_vocab_sizes).ok(),
               "could not derive PLE head layout") && ok;
    q4::ple_head_ids ids{};
    ok = check(q4::ple_compute_head_ids(
                       metadata, {248044, 248044}, 101, &ids).ok(),
               "could not derive PLE row ids") && ok;

    q4::ple_embedding_row_cache cache;
    ok = check(cache.configure(64u).ok(), "could not configure bounded row cache") && ok;
    std::array<float, q4::kPleEmbeddingOutputElements> first{};
    const std::uint64_t rss_before = rss_bytes();
    const q4::ple_embedding_status first_status =
            reader.gather_f32(ids, first.data(), first.size(), &cache);
    const std::uint64_t rss_after = rss_bytes();
    if (!first_status) {
        std::fprintf(stderr, "%s\n", first_status.message.c_str());
        return 1;
    }
    ok = check(rss_after <= rss_before + 8ull * 1024ull * 1024ull,
               "PLE gather materialized an unbounded payload") && ok;
    ok = check(std::all_of(first.begin(), first.end(),
                           [](float value) { return std::isfinite(value); }),
               "PLE gather produced a non-finite value") && ok;

    /* Independently read and decode each selected row.  This catches shard
     * mapping, local-row offsets and scale application on the real 51.2 GB
     * checkpoint rather than merely exercising synthetic bytes. */
    for (std::size_t head = 0; head < q4::kPleHeadCount; ++head) {
        const std::uint64_t row_id = static_cast<std::uint64_t>(ids[head]);
        const std::size_t shard = static_cast<std::size_t>(
                row_id / q4::kPleEmbeddingRowsPerShard);
        const std::uint64_t local = row_id % q4::kPleEmbeddingRowsPerShard;
        const std::string name =
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_" +
                std::to_string(shard) + ".weight";
        std::array<std::uint8_t, q4::kPleEmbeddingWidth> encoded{};
        if (!catalog->read_range(name,
                                 local * q4::kPleEmbeddingWidth,
                                 encoded.data(), encoded.size(), &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 1;
        }
        for (std::size_t column = 0; column < q4::kPleEmbeddingWidth; ++column) {
            const float expected = independent_e4m3(encoded[column]) * reader.scale();
            const float actual = first[head * q4::kPleEmbeddingWidth + column];
            if (!(std::isfinite(expected) && std::abs(expected - actual) <= 1.0e-9f)) {
                ok = check(false, "real PLE row differs from independent decode") && ok;
                break;
            }
        }
    }

    std::array<float, q4::kPleEmbeddingOutputElements> second{};
    const std::uint64_t hits_before = cache.hits();
    ok = check(reader.gather_f32(ids, second.data(), second.size(), &cache).ok(),
               "cached PLE gather failed") && ok;
    ok = check(first == second, "cached PLE gather changed output") && ok;
    ok = check(cache.hits() - hits_before == q4::kPleHeadCount,
               "second PLE gather did not hit every cached row") && ok;

    q4::ple_head_ids invalid = ids;
    invalid[0] = static_cast<std::int64_t>(q4::kPleEmbeddingRows);
    ok = check(reader.gather_f32(invalid, second.data(), second.size(), &cache).code ==
                       q4::ple_embedding_status_code::out_of_range,
               "out-of-range PLE row did not fail closed") && ok;
    ok = check(reader.gather_f32(ids, second.data(), second.size() - 1u, &cache).code ==
                       q4::ple_embedding_status_code::invalid_argument,
               "short PLE output did not fail closed") && ok;

    if (!ok) return 1;
    std::printf("qwen4exp-ple-embedding-test: pass scale=%.9g rows=%llu cache_hits=%llu rss_delta=%lld\n",
                reader.scale(),
                static_cast<unsigned long long>(reader.total_rows()),
                static_cast<unsigned long long>(cache.hits()),
                static_cast<long long>(rss_after) - static_cast<long long>(rss_before));
    return 0;
}
