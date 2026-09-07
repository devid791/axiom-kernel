#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple_adapter.hpp"
#include "axiom/qwen4exp/ple_compute.hpp"
#include "axiom/qwen4exp/ple_embedding.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

constexpr std::size_t kHidden = 2560u;
constexpr std::size_t kStreams = 4u;
constexpr std::size_t kChannels = kHidden * kStreams;
constexpr std::size_t kEmbedding = 2560u;
constexpr char kPlePrefix[] = "model.language_model.layers.1.ple.";
constexpr char kMetadataPrefix[] =
        "model.language_model.layers.1.ple.ple_embedding.";

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void cuda_require(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void adapter_require(q4::ple_adapter_status actual,
                     q4::ple_adapter_status expected,
                     const std::string &operation,
                     const std::string &detail = {}) {
    if (actual != expected) {
        fail(operation + ": expected " +
             q4::ple_adapter_status_string(expected) + ", got " +
             q4::ple_adapter_status_string(actual) +
             (detail.empty() ? std::string{} : " (" + detail + ")"));
    }
}

class stream_owner final {
public:
    stream_owner() {
        cuda_require(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~stream_owner() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    stream_owner(const stream_owner &) = delete;
    stream_owner &operator=(const stream_owner &) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class device_buffer final {
public:
    explicit device_buffer(std::size_t elements) : elements_(elements) {
        require(elements_ != 0u, "zero-sized device allocation");
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements_ * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    [[nodiscard]] T *get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return elements_ * sizeof(T);
    }
    void upload(const std::vector<T> &values) {
        require(values.size() == elements_, "device upload size mismatch");
        cuda_require(cudaMemcpy(pointer_, values.data(), bytes(),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy H2D");
    }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> values(elements_);
        cuda_require(cudaMemcpy(values.data(), pointer_, bytes(),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy D2H");
        return values;
    }

private:
    T *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

float bf16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float independent_e4m3fn(std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x80u) != 0u;
    const unsigned exponent = (encoded >> 3u) & 0x0fu;
    const unsigned mantissa = encoded & 0x07u;
    if (exponent == 15u && mantissa == 7u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float magnitude = exponent == 0u
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F,
                         static_cast<int>(exponent) - 7);
    return negative ? -magnitude : magnitude;
}

template <typename T>
std::vector<T> read_tensor(const q4::checkpoint_catalog &catalog,
                           const std::string &name,
                           q4::tensor_dtype dtype,
                           std::size_t expected_elements) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr, "missing oracle tensor: " + name);
    require(span->dtype == dtype, "oracle tensor dtype mismatch: " + name);
    require(span->bytes == expected_elements * sizeof(T),
            "oracle tensor byte count mismatch: " + name);
    std::vector<T> values(expected_elements);
    constexpr std::size_t kReadChunk = 2u * 1024u * 1024u;
    std::size_t offset = 0u;
    while (offset < span->bytes) {
        const std::size_t amount = std::min(
                kReadChunk,
                static_cast<std::size_t>(span->bytes) - offset);
        std::string error;
        require(catalog.read_range(
                        name, offset,
                        reinterpret_cast<unsigned char *>(values.data()) +
                                offset,
                        amount, &error),
                error.empty() ? "oracle range read failed: " + name : error);
        offset += amount;
    }
    return values;
}

q4::ple_metadata read_metadata(const q4::checkpoint_catalog &catalog) {
    q4::ple_metadata metadata{};
    metadata.unigram_vocab_size = 248320;
    metadata.eos_token_id = 248044;
    const auto multipliers = read_tensor<std::int64_t>(
            catalog, std::string(kMetadataPrefix) + "layer_multipliers",
            q4::tensor_dtype::i64, q4::kPleNgramOrder);
    const auto offsets = read_tensor<std::int64_t>(
            catalog, std::string(kMetadataPrefix) + "ngram_heads_offsets",
            q4::tensor_dtype::i64, q4::kPleHeadCount);
    const auto sizes = read_tensor<std::int64_t>(
            catalog,
            std::string(kMetadataPrefix) + "ngram_heads_vocab_sizes",
            q4::tensor_dtype::i64, q4::kPleHeadCount);
    std::copy(multipliers.begin(), multipliers.end(),
              metadata.layer_multipliers.begin());
    std::copy(offsets.begin(), offsets.end(),
              metadata.ngram_heads_offsets.begin());
    std::copy(sizes.begin(), sizes.end(),
              metadata.ngram_heads_vocab_sizes.begin());
    require(q4::ple_validate_metadata(metadata).ok(),
            "independent checkpoint metadata validation failed");
    return metadata;
}

q4::ple_head_ids independent_head_ids(
        const q4::ple_metadata &metadata,
        const std::array<std::int64_t, q4::kPleTokenContext> &context,
        std::int64_t token) {
    require(token >= 0 && token < metadata.unigram_vocab_size,
            "invalid oracle token");
    const std::int64_t eos = metadata.eos_token_id;
    const std::int64_t previous = context[1] == eos ? eos : context[1];
    const std::int64_t previous_two =
            context[1] == eos || context[0] == eos ? eos : context[0];
    const std::uint64_t current_mix =
            static_cast<std::uint64_t>(token) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[0]);
    const std::uint64_t previous_mix =
            static_cast<std::uint64_t>(previous) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[1]);
    const std::uint64_t previous_two_mix =
            static_cast<std::uint64_t>(previous_two) *
            static_cast<std::uint64_t>(metadata.layer_multipliers[2]);
    const std::uint64_t bigram = current_mix ^ previous_mix;
    const std::uint64_t trigram = bigram ^ previous_two_mix;
    q4::ple_head_ids ids{};
    for (std::size_t head = 0u; head < ids.size(); ++head) {
        const std::uint64_t mixed =
                head < q4::kPleHeadsPerNgram ? bigram : trigram;
        ids[head] = metadata.ngram_heads_offsets[head] +
                    static_cast<std::int64_t>(
                            mixed % static_cast<std::uint64_t>(
                                            metadata.ngram_heads_vocab_sizes[head]));
    }
    return ids;
}

float read_embedding_scale(const q4::checkpoint_catalog &catalog) {
    const auto bits = read_tensor<std::uint16_t>(
            catalog,
            std::string(kMetadataPrefix) +
                    "ngram_embedding.weight_scale",
            q4::tensor_dtype::bf16, 1u);
    const float scale = bf16_to_f32(bits[0]);
    require(std::isfinite(scale) && scale > 0.0F && scale < 0.001F,
            "invalid independent PLE calibration scalar");
    return scale;
}

std::vector<float> read_embedding_rows(
        const q4::checkpoint_catalog &catalog,
        const q4::ple_head_ids &ids,
        float scale) {
    std::vector<float> embedding(kEmbedding);
    for (std::size_t head = 0u; head < ids.size(); ++head) {
        require(ids[head] >= 0 &&
                        static_cast<std::uint64_t>(ids[head]) <
                                q4::kPleEmbeddingRows,
                "oracle PLE row is out of range");
        const std::uint64_t row = static_cast<std::uint64_t>(ids[head]);
        const std::size_t shard = static_cast<std::size_t>(
                row / q4::kPleEmbeddingRowsPerShard);
        const std::uint64_t local =
                row % q4::kPleEmbeddingRowsPerShard;
        const std::string name =
                std::string(kMetadataPrefix) + "ngram_embedding.shard_" +
                std::to_string(shard) + ".weight";
        std::array<std::uint8_t, q4::kPleEmbeddingWidth> encoded{};
        std::string error;
        require(catalog.read_range(
                        name, local * q4::kPleEmbeddingWidth,
                        encoded.data(), encoded.size(), &error),
                error.empty() ? "oracle FP8 row range read failed" : error);
        for (std::size_t column = 0u;
             column < q4::kPleEmbeddingWidth; ++column) {
            const float decoded = independent_e4m3fn(encoded[column]);
            require(std::isfinite(decoded),
                    "oracle FP8 row contains NaN");
            embedding[head * q4::kPleEmbeddingWidth + column] =
                    decoded * scale;
        }
    }
    return embedding;
}

struct host_weights {
    std::vector<std::uint16_t> key;
    std::vector<std::uint16_t> value;
    std::vector<std::uint16_t> norm_key;
    std::vector<std::uint16_t> norm_query;
    std::vector<std::uint16_t> norm_conv;
    std::vector<std::uint16_t> conv;
};

host_weights read_weights(const q4::checkpoint_catalog &catalog) {
    host_weights weights{};
    weights.key = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "key_proj.weight",
            q4::tensor_dtype::bf16, kChannels * kEmbedding);
    weights.value = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "value_proj.weight",
            q4::tensor_dtype::bf16, kHidden * kEmbedding);
    weights.norm_key = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "norm_key.weight",
            q4::tensor_dtype::bf16, kChannels);
    weights.norm_query = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "norm_query.weight",
            q4::tensor_dtype::bf16, kChannels);
    weights.norm_conv = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "norm_conv.weight",
            q4::tensor_dtype::bf16, kChannels);
    weights.conv = read_tensor<std::uint16_t>(
            catalog, std::string(kPlePrefix) + "conv1d.weight",
            q4::tensor_dtype::bf16,
            kChannels * q4::kPleComputeTaps);
    return weights;
}

std::vector<float> matvec(const std::vector<std::uint16_t> &weight,
                          const std::vector<float> &input,
                          std::size_t rows,
                          std::size_t columns) {
    require(weight.size() == rows * columns && input.size() == columns,
            "oracle matvec geometry mismatch");
    std::vector<float> output(rows);
    for (std::size_t row = 0u; row < rows; ++row) {
        double sum = 0.0;
        const std::size_t base = row * columns;
        for (std::size_t column = 0u; column < columns; ++column) {
            sum += static_cast<double>(bf16_to_f32(weight[base + column])) *
                   static_cast<double>(input[column]);
        }
        output[row] = static_cast<float>(sum);
    }
    return output;
}

std::vector<float> grouped_rms_norm(
        const std::vector<float> &input,
        const std::vector<std::uint16_t> &weight) {
    require(input.size() == kChannels && weight.size() == kChannels,
            "oracle grouped RMSNorm geometry mismatch");
    std::vector<float> output(kChannels);
    for (std::size_t stream = 0u; stream < kStreams; ++stream) {
        const std::size_t base = stream * kHidden;
        double squares = 0.0;
        for (std::size_t column = 0u; column < kHidden; ++column) {
            const double value = input[base + column];
            squares += value * value;
        }
        const float inverse = 1.0F / std::sqrt(
                static_cast<float>(squares / static_cast<double>(kHidden)) +
                1.0e-6F);
        for (std::size_t column = 0u; column < kHidden; ++column) {
            const std::size_t index = base + column;
            output[index] = input[index] * inverse *
                            (1.0F + bf16_to_f32(weight[index]));
        }
    }
    return output;
}

std::vector<float> official_first_token_oracle(
        const host_weights &weights,
        const std::vector<float> &embedding,
        const std::vector<float> &hidden_before_ple) {
    require(embedding.size() == kEmbedding &&
                    hidden_before_ple.size() == kChannels,
            "oracle PLE input geometry mismatch");
    std::vector<float> key = matvec(
            weights.key, embedding, kChannels, kEmbedding);
    const std::vector<float> value = matvec(
            weights.value, embedding, kHidden, kEmbedding);
    key = grouped_rms_norm(key, weights.norm_key);
    const std::vector<float> query =
            grouped_rms_norm(hidden_before_ple, weights.norm_query);
    std::vector<float> gated(kChannels);
    for (std::size_t stream = 0u; stream < kStreams; ++stream) {
        const std::size_t base = stream * kHidden;
        double dot = 0.0;
        for (std::size_t column = 0u; column < kHidden; ++column) {
            dot += static_cast<double>(key[base + column]) *
                   static_cast<double>(query[base + column]);
        }
        float gate = static_cast<float>(
                dot / std::sqrt(static_cast<double>(kHidden)));
        gate = std::copysign(
                std::sqrt(std::max(std::abs(gate), 1.0e-6F)), gate);
        const float probability = 1.0F / (1.0F + std::exp(-gate));
        for (std::size_t column = 0u; column < kHidden; ++column) {
            gated[base + column] = probability * value[column];
        }
    }
    const std::vector<float> normalized =
            grouped_rms_norm(gated, weights.norm_conv);
    std::vector<float> hidden_after_ple(kChannels);
    for (std::size_t channel = 0u; channel < kChannels; ++channel) {
        /* First token has zero prior conv history.  Official Conv1d ordering
         * therefore leaves only the t (last) tap active. */
        const float convolved =
                bf16_to_f32(weights.conv[
                        channel * q4::kPleComputeTaps +
                        (q4::kPleComputeTaps - 1u)]) *
                normalized[channel];
        const float ple = gated[channel] +
                          convolved / (1.0F + std::exp(-convolved));
        /* This is the contract under test: hidden += PLE occurs before the
         * existing attention hyper-connection consumes the tensor. */
        hidden_after_ple[channel] = hidden_before_ple[channel] + ple;
    }
    return hidden_after_ple;
}

float compare_output(const std::vector<float> &actual,
                     const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "output comparison size mismatch");
    float max_abs = 0.0F;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]) && std::isfinite(expected[index]),
                "PLE output contains a non-finite value");
        const float error = std::abs(actual[index] - expected[index]);
        max_abs = std::max(max_abs, error);
        const float tolerance = 4.0e-4F + 2.0e-4F * std::abs(expected[index]);
        require(error <= tolerance,
                "real checkpoint PLE CPU/CUDA oracle mismatch at element " +
                        std::to_string(index));
    }
    return max_abs;
}

void test_wrong_layer_admission(const q4::checkpoint_catalog &catalog,
                                cudaStream_t stream) {
    for (const std::size_t layer : {0u, 2u, 47u, 48u}) {
        q4::ple_adapter_config config{};
        config.layer_index = layer;
        std::unique_ptr<q4::ple_layer_adapter> rejected;
        std::string error;
        adapter_require(q4::ple_layer_adapter::load(
                                catalog, config, stream, &rejected, &error),
                        q4::ple_adapter_status::unsupported_layer,
                        "wrong-layer admission", error);
        require(rejected == nullptr,
                "wrong-layer admission returned an adapter");
    }
}

void test_real_layer(const q4::checkpoint_catalog &catalog,
                     cudaStream_t stream) {
    q4::ple_adapter_config config{};
    config.layer_index = 1u;
    config.max_speculative_tokens = 8u;
    config.row_cache_slots = 64u;
    config.upload_chunk_bytes = 2u * 1024u * 1024u;
    std::unique_ptr<q4::ple_layer_adapter> adapter;
    std::string error;
    adapter_require(q4::ple_layer_adapter::load(
                            catalog, config, stream, &adapter, &error),
                    q4::ple_adapter_status::ok,
                    "real PLE adapter load", error);
    require(adapter != nullptr && adapter->initialized() &&
                    adapter->layer_index() == 1u,
            "real PLE adapter did not initialize layer 1");
    const q4::ple_adapter_footprint footprint = adapter->footprint();
    require(footprint.resident_weight_bytes == 65679360u &&
                    footprint.max_cold_nvme_bytes_per_token == 2560u &&
                    footprint.host_to_device_bytes_per_token == 10240u,
            "PLE adapter footprint mismatch");

    const q4::ple_metadata metadata = read_metadata(catalog);
    const float scale = read_embedding_scale(catalog);
    require(metadata.layer_multipliers == adapter->metadata().layer_multipliers &&
                    metadata.ngram_heads_offsets ==
                            adapter->metadata().ngram_heads_offsets &&
                    metadata.ngram_heads_vocab_sizes ==
                            adapter->metadata().ngram_heads_vocab_sizes &&
                    adapter->embedding_scale() == scale,
            "adapter metadata/scale differs from independent checkpoint read");

    std::unique_ptr<q4::ple_layer_session> session;
    adapter_require(adapter->create_session(stream, &session, &error),
                    q4::ple_adapter_status::ok,
                    "PLE session creation", error);
    require(session != nullptr && session->initialized() &&
                    !session->poisoned(),
            "PLE session did not initialize");

    std::vector<float> hidden(kChannels);
    for (std::size_t index = 0u; index < hidden.size(); ++index) {
        hidden[index] = 0.025F * std::sin(
                static_cast<float>((index * 17u + 11u) % 1009u) * 0.013F);
    }
    device_buffer<float> device_hidden(kChannels);
    device_buffer<float> device_output(kChannels);
    device_hidden.upload(hidden);

    const std::int64_t first_token = 101;
    const q4::ple_head_ids first_ids = independent_head_ids(
            metadata, {metadata.eos_token_id, metadata.eos_token_id},
            first_token);

    /* Warm kernels and fill the explicit row cache, then roll back all token
     * and convolution state. */
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "warm begin", error);
    adapter_require(session->prefetch_token(first_token, stream, &error),
                    q4::ple_adapter_status::ok, "warm prefetch", error);
    require(session->prefetched_head_ids() == first_ids,
            "initial EOS-aware PLE row IDs mismatch");
    adapter_require(session->stage_prefetched_and_add_cuda(
                            device_hidden.get(), device_output.get(), stream,
                            &error),
                    q4::ple_adapter_status::ok, "warm stage", error);
    adapter_require(session->rollback(&error), q4::ple_adapter_status::ok,
                    "warm rollback", error);
    cuda_require(cudaStreamSynchronize(stream), "warm synchronize");
    require(session->committed_tokens() == 0u,
            "rollback changed committed token history");

    const std::uint64_t hits_before = session->cache_hits();
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "measured begin", error);
    adapter_require(session->prefetch_token(first_token, stream, &error),
                    q4::ple_adapter_status::ok, "measured prefetch", error);
    require(session->cache_hits() - hits_before == q4::kPleHeadCount,
            "warm PLE prefetch did not hit all explicit cache rows");

    std::size_t free_before = 0u;
    std::size_t total_before = 0u;
    cuda_require(cudaMemGetInfo(&free_before, &total_before),
                 "cudaMemGetInfo before PLE hot path");
    adapter_require(session->stage_prefetched_and_add_cuda(
                            device_hidden.get(), device_output.get(), stream,
                            &error),
                    q4::ple_adapter_status::ok, "measured stage", error);
    cuda_require(cudaStreamSynchronize(stream), "measured stage synchronize");
    std::size_t free_after = 0u;
    std::size_t total_after = 0u;
    cuda_require(cudaMemGetInfo(&free_after, &total_after),
                 "cudaMemGetInfo after PLE hot path");
    const bool sanitizer_instrumented =
            std::getenv("AXIOM_QWEN4EXP_COMPUTE_SANITIZER") != nullptr;
    if (!sanitizer_instrumented) {
        require(free_before == free_after && total_before == total_after,
                "PLE hot path changed the CUDA allocation footprint");
    }

    const host_weights weights = read_weights(catalog);
    const std::vector<float> embedding =
            read_embedding_rows(catalog, first_ids, scale);
    const std::vector<float> expected =
            official_first_token_oracle(weights, embedding, hidden);
    const std::vector<float> actual = device_output.download();
    const float max_abs = compare_output(actual, expected);
    adapter_require(session->commit_prefix(1u, stream, &error),
                    q4::ple_adapter_status::ok, "first commit", error);
    require(session->committed_tokens() == 1u,
            "first PLE token did not commit");

    /* A second token sees the committed first token and EOS as its older
     * context.  Rollback must preserve that committed history. */
    const std::int64_t second_token = 202;
    const q4::ple_head_ids second_ids = independent_head_ids(
            metadata, {metadata.eos_token_id, first_token}, second_token);
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "second begin", error);
    adapter_require(session->prefetch_token(second_token, stream, &error),
                    q4::ple_adapter_status::ok, "second prefetch", error);
    require(session->prefetched_head_ids() == second_ids,
            "committed token history was not used by PLE hashing");
    adapter_require(session->stage_prefetched_and_add_cuda(
                            device_hidden.get(), device_output.get(), stream,
                            &error),
                    q4::ple_adapter_status::ok, "second stage", error);
    adapter_require(session->rollback(&error), q4::ple_adapter_status::ok,
                    "second rollback", error);
    require(session->committed_tokens() == 1u,
            "second rollback damaged committed token history");

    /* Commit EOS, then prove that both older n-gram positions are masked back
     * to EOS for the next token, matching _shift_right_ignore_eos. */
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "EOS begin", error);
    adapter_require(session->prefetch_token(
                            metadata.eos_token_id, stream, &error),
                    q4::ple_adapter_status::ok, "EOS prefetch", error);
    adapter_require(session->stage_prefetched_and_add_cuda(
                            device_hidden.get(), device_output.get(), stream,
                            &error),
                    q4::ple_adapter_status::ok, "EOS stage", error);
    adapter_require(session->commit_prefix(1u, stream, &error),
                    q4::ple_adapter_status::ok, "EOS commit", error);

    const std::int64_t after_eos_token = 303;
    const q4::ple_head_ids after_eos_ids = independent_head_ids(
            metadata, {metadata.eos_token_id, metadata.eos_token_id},
            after_eos_token);
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "after-EOS begin", error);
    adapter_require(session->prefetch_token(after_eos_token, stream, &error),
                    q4::ple_adapter_status::ok, "after-EOS prefetch", error);
    require(session->prefetched_head_ids() == after_eos_ids,
            "PLE token history did not reset at EOS");
    adapter_require(session->rollback(&error), q4::ple_adapter_status::ok,
                    "after-EOS rollback", error);

    /* Fail-closed pointer validation must close, not strand, a transaction. */
    adapter_require(session->begin_transaction(&error),
                    q4::ple_adapter_status::ok, "fail-closed begin", error);
    adapter_require(session->prefetch_token(404, stream, &error),
                    q4::ple_adapter_status::ok, "fail-closed prefetch", error);
    std::vector<float> invalid_host_output(kChannels);
    adapter_require(session->stage_prefetched_and_add_cuda(
                            hidden.data(), invalid_host_output.data(), stream,
                            &error),
                    q4::ple_adapter_status::invalid_device_pointer,
                    "fail-closed host pointer", error);
    require(!session->transaction_open() && !session->token_prefetched() &&
                    !session->poisoned(),
            "invalid pointer did not close the PLE transaction cleanly");

    // Split boundary keeps the synchronous wrapper's compute contract intact.
    adapter_require(session->begin_transaction(&error), q4::ple_adapter_status::ok,
                    "async begin", error);
    adapter_require(session->start_prefetch_token(505, &error), q4::ple_adapter_status::ok,
                    "async start", error);
    require(session->prefetch_pending() && !session->token_prefetched(),
            "pending I/O was marked GPU-ready");
    adapter_require(session->start_prefetch_token(506, &error), q4::ple_adapter_status::invalid_state,
                    "second async start rejected", error);
    adapter_require(session->commit_prefix(0, stream, &error), q4::ple_adapter_status::invalid_state,
                    "pending commit rejected", error);
    adapter_require(session->finish_prefetch_token(stream, &error), q4::ple_adapter_status::ok,
                    "async finish", error);
    require(!session->prefetch_pending() && session->token_prefetched(), "async finish not ready");
    adapter_require(session->stage_prefetched_and_add_cuda(device_hidden.get(), device_output.get(),
                    stream, &error), q4::ple_adapter_status::ok, "async stage", error);
    adapter_require(session->rollback(&error), q4::ple_adapter_status::ok, "async rollback", error);

    adapter_require(session->begin_transaction(&error), q4::ple_adapter_status::ok,
                    "pending rollback begin", error);
    adapter_require(session->start_prefetch_token(606, &error), q4::ple_adapter_status::ok,
                    "pending rollback start", error);
    adapter_require(session->rollback(&error), q4::ple_adapter_status::ok, "pending rollback", error);
    require(!session->prefetch_pending() && !session->transaction_open(), "rollback left I/O pending");
    adapter_require(session->begin_transaction(&error), q4::ple_adapter_status::ok,
                    "pending reset begin", error);
    adapter_require(session->start_prefetch_token(707, &error), q4::ple_adapter_status::ok,
                    "pending reset start", error);
    adapter_require(session->reset(stream, &error), q4::ple_adapter_status::ok, "pending reset", error);
    require(!session->prefetch_pending() && session->committed_tokens() == 0 &&
                    session->cache_hits() == 0 && session->cache_misses() == 0,
            "reset did not drain before clearing cache/history");

    std::unique_ptr<q4::ple_layer_session> doomed;
    adapter_require(adapter->create_session(stream, &doomed, &error), q4::ple_adapter_status::ok,
                    "destruction session", error);
    adapter_require(doomed->begin_transaction(&error), q4::ple_adapter_status::ok,
                    "destruction begin", error);
    adapter_require(doomed->start_prefetch_token(808, &error), q4::ple_adapter_status::ok,
                    "destruction start", error);
    doomed.reset(); // Must join before freeing staging or accessing its owner.

    cuda_require(cudaStreamSynchronize(stream), "final PLE synchronize");
    std::printf(
            "qwen4exp-ple-adapter-test: PASS layer=1 ple_tensors=%zu "
            "resident_bytes=%zu scale=%.9g max_abs=%.9g cache=%llu/%llu "
            "nvme_bytes=%llu hot_alloc_gate=%s order=hidden_plus_ple_before_mhc\n",
            q4::kPleAdapterCheckpointTensors,
            footprint.resident_weight_bytes,
            adapter->embedding_scale(), max_abs,
            static_cast<unsigned long long>(session->cache_hits()),
            static_cast<unsigned long long>(session->cache_misses()),
            static_cast<unsigned long long>(session->nvme_bytes_read()),
            sanitizer_instrumented ? "sanitizer-instrumented" : "delta-zero");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            fail(std::string("usage: ") + argv[0] + " MODEL_DIR");
        }
        int devices = 0;
        cuda_require(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        require(devices > 0, "PLE adapter test requires a CUDA device");
        cuda_require(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, 0),
                     "cudaGetDeviceProperties");
        require(properties.major >= 12,
                "PLE adapter test requires compute capability 12.x");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(
                        argv[1], &catalog, &error),
                error.empty() ? "could not open pinned checkpoint" : error);
        require(catalog != nullptr && catalog->tensor_count() == 296475u,
                "unexpected pinned checkpoint tensor count");
        stream_owner stream;
        test_wrong_layer_admission(*catalog, stream.get());
        test_real_layer(*catalog, stream.get());
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-ple-adapter-test: FAIL %s\n",
                     exception.what());
        return 1;
    }
}
