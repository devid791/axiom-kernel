#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <unistd.h>

namespace {

std::uint64_t rss_bytes() {
    std::ifstream stream("/proc/self/statm");
    std::uint64_t pages = 0;
    std::uint64_t resident = 0;
    if (!(stream >> pages >> resident)) return 0;
    return resident * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
}

float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool check(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "checkpoint_test: %s\n", message);
    return condition;
}

template <std::size_t Size>
bool read_i64_array(const axiom::qwen4exp::checkpoint_catalog &catalog,
                    const char *name,
                    std::array<std::int64_t, Size> *values,
                    std::string *error) {
    return values != nullptr &&
           catalog.read_range(name, 0, values->data(),
                              values->size() * sizeof((*values)[0]), error);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    std::unique_ptr<axiom::qwen4exp::checkpoint_catalog> catalog;
    std::string error;
    if (!axiom::qwen4exp::checkpoint_catalog::open(argv[1], &catalog, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    bool ok = check(catalog && catalog->tensor_count() == 296475u,
                    "unexpected catalog cardinality");

    const auto *expert = catalog->find(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight");
    ok = check(expert && expert->dtype == axiom::qwen4exp::tensor_dtype::u8 &&
                       expert->rank == 2u && expert->shape[0] == 640u &&
                       expert->shape[1] == 1280u && expert->bytes == 819200u,
               "expert ModelOpt span mismatch") && ok;

    const auto *ple = catalog->find(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight");
    ok = check(ple && ple->dtype == axiom::qwen4exp::tensor_dtype::f8_e4m3 &&
                    ple->rank == 2u && ple->shape[0] == 2500012u &&
                    ple->shape[1] == 160u && ple->bytes == 400001920u,
               "PLE FP8 span mismatch") && ok;

    const std::uint64_t rss_before = rss_bytes();
    unsigned char sample[160]{};
    if (!catalog->read_range(ple ? ple->name : std::string{}, 123456u * 160u,
                             sample, sizeof(sample), &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        ok = false;
    }
    const std::uint64_t rss_after = rss_bytes();
    ok = check(rss_after <= rss_before + 8ull * 1024ull * 1024ull,
               "range read materialized an unexpectedly large buffer") && ok;

    std::uint16_t scale_bits = 0;
    const std::string scale_name =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale";
    if (!catalog->read_range(scale_name, 0, &scale_bits, sizeof(scale_bits), &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        ok = false;
    } else {
        const float scale = bf16_to_f32(scale_bits);
        ok = check(std::isfinite(scale) && scale > 0.0f && scale < 0.001f,
                   "PLE BF16 scale is not the calibrated non-unit value") && ok;
    }


    axiom::qwen4exp::ple_metadata ple_metadata;
    ple_metadata.unigram_vocab_size = 248320;
    ple_metadata.eos_token_id = 248044;
    const bool ple_metadata_read =
            read_i64_array(
                    *catalog,
                    "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                    &ple_metadata.layer_multipliers, &error) &&
            read_i64_array(
                    *catalog,
                    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                    &ple_metadata.ngram_heads_offsets, &error) &&
            read_i64_array(
                    *catalog,
                    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
                    &ple_metadata.ngram_heads_vocab_sizes, &error);
    if (!ple_metadata_read) {
        std::fprintf(stderr, "%s\n", error.c_str());
        ok = false;
    } else {
        ok = check(axiom::qwen4exp::ple_validate_metadata(ple_metadata).ok(),
                   "real checkpoint PLE metadata failed semantic validation") && ok;
        std::array<std::int64_t, axiom::qwen4exp::kPleNgramOrder> derived_multipliers{};
        std::array<std::int64_t, axiom::qwen4exp::kPleHeadCount> derived_offsets{};
        std::array<std::int64_t, axiom::qwen4exp::kPleHeadCount> derived_vocab_sizes{};
        const bool derived =
                axiom::qwen4exp::ple_derive_layer_multipliers(
                        248320, 0u, 1234u, &derived_multipliers).ok() &&
                axiom::qwen4exp::ple_derive_head_layout(
                        20000000, 0u, &derived_offsets, &derived_vocab_sizes).ok();
        ok = check(derived &&
                           derived_multipliers == ple_metadata.layer_multipliers &&
                           derived_offsets == ple_metadata.ngram_heads_offsets &&
                           derived_vocab_sizes == ple_metadata.ngram_heads_vocab_sizes,
                   "real checkpoint PLE metadata differs from deterministic architecture contract") && ok;
    }

    unsigned char guard = 0;
    std::string expected_error;
    ok = check(!catalog->read_range(ple ? ple->name : std::string{},
                                    ple ? ple->bytes : 0u, &guard, 1u,
                                    &expected_error),
               "out-of-range tensor read did not fail closed") && ok;
    ok = check(!catalog->read_range("not.a.tensor", 0, &guard, 1u,
                                    &expected_error),
               "unknown tensor read did not fail closed") && ok;

    if (!ok) return 1;
    std::printf("checkpoint_test: OK tensors=%zu rss_delta=%lld ple_scale=validated ple_metadata=validated\n",
                catalog->tensor_count(),
                static_cast<long long>(rss_after) - static_cast<long long>(rss_before));
    return 0;
}
