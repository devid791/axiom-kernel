#include "axiom/qwen4exp/output_head.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr char kEmbeddingName[] =
        "model.language_model.embed_tokens.weight";
constexpr char kLegacyFinalNormName[] =
        "model.language_model.norm.weight";
constexpr char kGlobalNormName[] =
        "model.language_model.hyper_connection_mixer.hc_norm.weight";
constexpr char kGlobalDownName[] =
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight";
constexpr char kGlobalUpName[] =
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight";
constexpr char kLmHeadName[] = "lm_head.weight";

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

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0u) {
        return static_cast<std::uint16_t>((bits >> 16u) | 0x0040u);
    }
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float round_bf16(float value) {
    return bf16_to_float(float_to_bf16(value));
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
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class device_allocation final {
public:
    device_allocation() = default;
    explicit device_allocation(std::size_t bytes) { allocate(bytes); }
    ~device_allocation() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_allocation(const device_allocation &) = delete;
    device_allocation &operator=(const device_allocation &) = delete;
    device_allocation(device_allocation &&) = delete;
    device_allocation &operator=(device_allocation &&) = delete;
    void allocate(std::size_t bytes) {
        require(pointer_ == nullptr && bytes != 0u,
                "invalid device allocation request");
        cuda_require(cudaMalloc(&pointer_, bytes), "cudaMalloc");
        bytes_ = bytes;
    }
    template <typename T>
    T *as() const noexcept {
        return static_cast<T *>(pointer_);
    }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void *pointer_ = nullptr;
    std::size_t bytes_ = 0u;
};

std::vector<std::uint16_t> read_bf16(const q4::checkpoint_catalog &catalog,
                                     const std::string &name,
                                     std::size_t elements) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr, "missing real tensor: " + name);
    require(span->dtype == q4::tensor_dtype::bf16 &&
                    span->bytes == elements * sizeof(std::uint16_t),
            "unexpected BF16 tensor contract: " + name);
    std::vector<std::uint16_t> result(elements);
    std::string error;
    require(catalog.read_range(name, 0u, result.data(), span->bytes, &error),
            error.empty() ? "real tensor read failed: " + name : error);
    return result;
}

std::vector<std::uint16_t> read_bf16_row(
        const q4::checkpoint_catalog &catalog, const std::string &name,
        std::size_t row, std::size_t columns) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr && span->dtype == q4::tensor_dtype::bf16 &&
                    span->rank == 2u && span->shape[1] == columns &&
                    row < span->shape[0],
            "unexpected row tensor contract: " + name);
    std::vector<std::uint16_t> result(columns);
    std::string error;
    require(catalog.read_range(
                    name, row * columns * sizeof(std::uint16_t), result.data(),
                    result.size() * sizeof(std::uint16_t), &error),
            error.empty() ? "real tensor row read failed: " + name : error);
    return result;
}

struct comparison {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
};

comparison compare_values(const std::vector<float> &actual,
                          const std::vector<float> &expected,
                          float absolute_limit, float relative_limit,
                          const char *label) {
    require(actual.size() == expected.size(),
            std::string(label) + " size mismatch");
    comparison result{};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]),
                std::string(label) + " produced a non-finite value");
        const float absolute = std::fabs(actual[index] - expected[index]);
        const float relative =
                absolute / std::max(1.0e-5F, std::fabs(expected[index]));
        result.max_absolute = std::max(result.max_absolute, absolute);
        result.max_relative = std::max(result.max_relative, relative);
        if (absolute > absolute_limit && relative > relative_limit) {
            fail(std::string(label) + " mismatch at index " +
                 std::to_string(index) + ": actual=" +
                 std::to_string(actual[index]) + " expected=" +
                 std::to_string(expected[index]));
        }
    }
    return result;
}

std::uint64_t fnv1a(const void *data, std::size_t bytes) {
    const auto *octets = static_cast<const unsigned char *>(data);
    std::uint64_t digest = 1469598103934665603ULL;
    for (std::size_t index = 0u; index < bytes; ++index) {
        digest ^= octets[index];
        digest *= 1099511628211ULL;
    }
    return digest;
}

float head_row_reference(const std::vector<std::uint16_t> &row,
                         const std::vector<float> &hidden) {
    require(row.size() == q4::kOutputHeadHidden &&
                    hidden.size() == q4::kOutputHeadHidden,
            "LM-head subset oracle shape mismatch");
    float sum = 0.0F;
    for (std::size_t column = 0u; column < q4::kOutputHeadHidden; ++column) {
        sum = std::fma(round_bf16(hidden[column]),
                       bf16_to_float(row[column]), sum);
    }
    return sum;
}

void validate_catalog_contract(const q4::checkpoint_catalog &catalog) {
    const q4::tensor_span *embedding = catalog.find(kEmbeddingName);
    const q4::tensor_span *norm = catalog.find(kGlobalNormName);
    const q4::tensor_span *down = catalog.find(kGlobalDownName);
    const q4::tensor_span *up = catalog.find(kGlobalUpName);
    const q4::tensor_span *head = catalog.find(kLmHeadName);
    require(embedding != nullptr && embedding->dtype == q4::tensor_dtype::bf16 &&
                    embedding->rank == 2u &&
                    embedding->shape[0] == q4::kOutputHeadVocab &&
                    embedding->shape[1] == q4::kOutputHeadHidden,
            "real embedding contract mismatch");
    require(catalog.find(kLegacyFinalNormName) == nullptr,
            "unexpected standalone final norm in qwen4_exp checkpoint");
    require(norm != nullptr && norm->dtype == q4::tensor_dtype::bf16 &&
                    norm->rank == 1u &&
                    norm->shape[0] == q4::kOutputHeadResidual,
            "real global-mixer norm contract mismatch");
    require(down != nullptr && down->dtype == q4::tensor_dtype::bf16 &&
                    down->rank == 2u &&
                    down->shape[0] == q4::kOutputHeadRank &&
                    down->shape[1] == q4::kOutputHeadResidual,
            "real global-mixer down contract mismatch");
    require(up != nullptr && up->dtype == q4::tensor_dtype::bf16 &&
                    up->rank == 2u &&
                    up->shape[0] == q4::kOutputHeadResidual &&
                    up->shape[1] == q4::kOutputHeadRank,
            "real global-mixer up contract mismatch");
    require(head != nullptr && head->dtype == q4::tensor_dtype::bf16 &&
                    head->rank == 2u &&
                    head->shape[0] == q4::kOutputHeadVocab &&
                    head->shape[1] == q4::kOutputHeadHidden,
            "real untied LM-head contract mismatch");
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    try {
        int device = -1;
        cuda_require(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "output-head gate requires the SM120 path");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error),
                error.empty() ? "checkpoint catalog open failed" : error);
        require(catalog != nullptr, "checkpoint catalog is null");
        validate_catalog_contract(*catalog);

        q4::output_head_config invalid{};
        invalid.max_tokens = 0u;
        require(q4::output_head_validate_config(invalid) ==
                        q4::output_head_status::invalid_argument,
                "zero-token config did not fail closed");

        q4::output_head_config config{};
        config.max_tokens = 1u;
        config.upload_chunk_bytes = 16u * 1024u * 1024u;
        config.blas_workspace_bytes = 4u * 1024u * 1024u;
        stream_owner stream;
        std::unique_ptr<q4::resident_output_head> output_head;
        const auto load_start = std::chrono::steady_clock::now();
        const q4::output_head_status load_status = q4::resident_output_head::load(
                *catalog, config, stream.get(), &output_head, &error);
        const auto load_end = std::chrono::steady_clock::now();
        require(load_status == q4::output_head_status::ok && output_head != nullptr,
                error.empty() ? std::string("output-head load failed: ") +
                                        q4::output_head_status_string(load_status)
                              : error);
        require(output_head->initialized() && output_head->device() == device,
                "resident output head did not initialize on the active device");
        require(output_head->tying() ==
                        q4::output_head_tying::untied_checkpoint_head,
                "real checkpoint was not classified as untied");
        require(output_head->finalization() ==
                        q4::output_head_finalization::
                                global_hyper_connection_mixer,
                "real checkpoint finalization is not the global mixer");
        constexpr std::uint64_t expected_resident_bytes = 2555924480ULL;
        require(output_head->resident_weight_bytes() == expected_resident_bytes,
                "resident output-tail byte accounting mismatch");

        const q4::output_head_workspace_requirements requirements =
                output_head->workspace_requirements();
        device_allocation d_hyper(requirements.hyper_input_f32_bytes);
        device_allocation d_normalized(requirements.normalized_f32_bytes);
        device_allocation d_low_rank(requirements.low_rank_f32_bytes);
        device_allocation d_mix_logits(requirements.mix_logits_f32_bytes);
        device_allocation d_mixed(requirements.mixed_hidden_f32_bytes);
        device_allocation d_linear_input(requirements.linear_input_bf16_bytes);
        device_allocation d_logits(requirements.logits_f32_bytes);
        device_allocation d_blas(requirements.blas_workspace_bytes);
        device_allocation d_status(requirements.device_status_bytes);
        device_allocation d_input_token(sizeof(std::uint32_t));
        device_allocation d_argmax_token(sizeof(std::uint32_t));
        device_allocation d_argmax_logit(sizeof(float));
        device_allocation d_sample_token_a(sizeof(std::uint32_t));
        device_allocation d_sample_token_b(sizeof(std::uint32_t));
        device_allocation d_sample_logit_a(sizeof(float));
        device_allocation d_sample_logit_b(sizeof(float));

        q4::output_head_scratch scratch{};
        scratch.hyper_input = d_hyper.as<float>();
        scratch.normalized = d_normalized.as<float>();
        scratch.low_rank = d_low_rank.as<float>();
        scratch.mix_logits = d_mix_logits.as<float>();
        scratch.mixed_hidden = d_mixed.as<float>();
        scratch.linear_input_bf16 = d_linear_input.as<std::uint16_t>();
        scratch.linear_input_bf16_bytes = d_linear_input.bytes();
        scratch.logits = d_logits.as<float>();
        scratch.blas_workspace = d_blas.as<void>();
        scratch.blas_workspace_bytes = d_blas.bytes();
        scratch.device_status = d_status.as<std::uint32_t>();
        scratch.device_status_bytes = d_status.bytes();

        constexpr std::uint32_t source_token = 248053u;
        cuda_require(cudaMemcpyAsync(d_input_token.as<std::uint32_t>(),
                                     &source_token, sizeof(source_token),
                                     cudaMemcpyHostToDevice, stream.get()),
                     "copy source token");
        require(output_head->embedding_global_mix_logits(
                        d_input_token.as<std::uint32_t>(), 1u, scratch,
                        stream.get()) == q4::output_head_status::ok,
                "embedding -> global mixer -> head enqueue failed");
        require(output_head->argmax(
                        scratch.logits, 1u, d_argmax_token.as<std::uint32_t>(),
                        d_argmax_logit.as<float>(), stream.get()) ==
                        q4::output_head_status::ok,
                "argmax enqueue failed");
        const q4::output_head_sampling_config sampling{0.8F, 0x5a17ULL, 37u};
        require(output_head->sample(
                        scratch.logits, 1u, sampling,
                        d_sample_token_a.as<std::uint32_t>(),
                        d_sample_logit_a.as<float>(), stream.get()) ==
                        q4::output_head_status::ok &&
                        output_head->sample(
                                scratch.logits, 1u, sampling,
                                d_sample_token_b.as<std::uint32_t>(),
                                d_sample_logit_b.as<float>(), stream.get()) ==
                                q4::output_head_status::ok,
                "deterministic sampling enqueue failed");
        require(output_head->collect_device_status(scratch, stream.get()) ==
                        q4::output_head_status::ok,
                "valid token was rejected asynchronously");

        std::vector<float> gpu_hyper(q4::kOutputHeadResidual);
        std::vector<float> gpu_mixed(q4::kOutputHeadHidden);
        std::vector<float> gpu_logits(q4::kOutputHeadVocab);
        cuda_require(cudaMemcpy(gpu_hyper.data(), scratch.hyper_input,
                                gpu_hyper.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy repeated embedding");
        cuda_require(cudaMemcpy(gpu_mixed.data(), scratch.mixed_hidden,
                                gpu_mixed.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy mixed hidden");
        cuda_require(cudaMemcpy(gpu_logits.data(), scratch.logits,
                                gpu_logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy logits");

        std::uint32_t gpu_argmax = 0u;
        float gpu_argmax_logit = 0.0F;
        std::uint32_t sample_a = 0u;
        std::uint32_t sample_b = 0u;
        float sample_logit_a = 0.0F;
        float sample_logit_b = 0.0F;
        cuda_require(cudaMemcpy(&gpu_argmax,
                                d_argmax_token.as<std::uint32_t>(),
                                sizeof(gpu_argmax), cudaMemcpyDeviceToHost),
                     "copy argmax token");
        cuda_require(cudaMemcpy(&gpu_argmax_logit,
                                d_argmax_logit.as<float>(),
                                sizeof(gpu_argmax_logit),
                                cudaMemcpyDeviceToHost),
                     "copy argmax logit");
        cuda_require(cudaMemcpy(&sample_a,
                                d_sample_token_a.as<std::uint32_t>(),
                                sizeof(sample_a), cudaMemcpyDeviceToHost),
                     "copy sample A token");
        cuda_require(cudaMemcpy(&sample_b,
                                d_sample_token_b.as<std::uint32_t>(),
                                sizeof(sample_b), cudaMemcpyDeviceToHost),
                     "copy sample B token");
        cuda_require(cudaMemcpy(&sample_logit_a,
                                d_sample_logit_a.as<float>(),
                                sizeof(sample_logit_a), cudaMemcpyDeviceToHost),
                     "copy sample A logit");
        cuda_require(cudaMemcpy(&sample_logit_b,
                                d_sample_logit_b.as<float>(),
                                sizeof(sample_logit_b), cudaMemcpyDeviceToHost),
                     "copy sample B logit");

        const std::vector<std::uint16_t> embedding_row = read_bf16_row(
                *catalog, kEmbeddingName, source_token, q4::kOutputHeadHidden);
        std::vector<float> expected_hyper(q4::kOutputHeadResidual);
        for (std::size_t stream_index = 0u;
             stream_index < q4::kOutputHeadStreams; ++stream_index) {
            for (std::size_t hidden = 0u; hidden < q4::kOutputHeadHidden;
                 ++hidden) {
                expected_hyper[stream_index * q4::kOutputHeadHidden + hidden] =
                        bf16_to_float(embedding_row[hidden]);
            }
        }
        require(std::memcmp(gpu_hyper.data(), expected_hyper.data(),
                            gpu_hyper.size() * sizeof(float)) == 0,
                "embedding repeat is not byte-exact");

        const std::vector<std::uint16_t> norm = read_bf16(
                *catalog, kGlobalNormName, q4::kOutputHeadResidual);
        const std::vector<std::uint16_t> down = read_bf16(
                *catalog, kGlobalDownName,
                q4::kOutputHeadRank * q4::kOutputHeadResidual);
        const std::vector<std::uint16_t> up = read_bf16(
                *catalog, kGlobalUpName,
                q4::kOutputHeadResidual * q4::kOutputHeadRank);
        std::vector<float> cpu_normalized(q4::kOutputHeadResidual);
        std::vector<float> cpu_low_rank(q4::kOutputHeadRank);
        std::vector<float> cpu_mix_logits(q4::kOutputHeadResidual);
        std::vector<float> cpu_mixed(q4::kOutputHeadHidden);
        require(q4::output_head_global_mix_reference(
                        expected_hyper.data(), norm.data(), down.data(), up.data(),
                        1u, cpu_normalized.data(), cpu_low_rank.data(),
                        cpu_mix_logits.data(), cpu_mixed.data()) ==
                        q4::output_head_status::ok,
                "real-weight CPU global-mixer oracle failed");
        const comparison mixed_error = compare_values(
                gpu_mixed, cpu_mixed, 6.0e-3F, 8.0e-3F, "global mixer");

        constexpr std::uint32_t subset_ids[]{0u, 1u, 42u, 151643u,
                                              248319u};
        float head_subset_max_absolute = 0.0F;
        float head_subset_max_relative = 0.0F;
        for (const std::uint32_t token : subset_ids) {
            const std::vector<std::uint16_t> row = read_bf16_row(
                    *catalog, kLmHeadName, token, q4::kOutputHeadHidden);
            const float expected = head_row_reference(row, gpu_mixed);
            const float actual = gpu_logits[token];
            const float absolute = std::fabs(actual - expected);
            const float relative =
                    absolute / std::max(1.0e-5F, std::fabs(expected));
            head_subset_max_absolute =
                    std::max(head_subset_max_absolute, absolute);
            head_subset_max_relative =
                    std::max(head_subset_max_relative, relative);
            require(absolute <= 3.0e-3F || relative <= 3.0e-3F,
                    "real LM-head subset oracle mismatch for token " +
                            std::to_string(token));
        }

        std::uint32_t cpu_argmax = 0u;
        float cpu_argmax_logit = gpu_logits[0];
        for (std::uint32_t token = 1u; token < q4::kOutputHeadVocab; ++token) {
            if (gpu_logits[token] > cpu_argmax_logit) {
                cpu_argmax = token;
                cpu_argmax_logit = gpu_logits[token];
            }
        }
        require(gpu_argmax == cpu_argmax &&
                        std::memcmp(&gpu_argmax_logit, &cpu_argmax_logit,
                                    sizeof(float)) == 0,
                "CUDA argmax differs from deterministic CPU scan");
        require(sample_a == sample_b &&
                        std::memcmp(&sample_logit_a, &sample_logit_b,
                                    sizeof(float)) == 0 &&
                        sample_a < q4::kOutputHeadVocab &&
                        sample_logit_a == gpu_logits[sample_a],
                "categorical sampling is not deterministic");

        std::vector<float> second_logits(q4::kOutputHeadVocab);
        require(output_head->embedding_global_mix_logits(
                        d_input_token.as<std::uint32_t>(), 1u, scratch,
                        stream.get()) == q4::output_head_status::ok &&
                        output_head->collect_device_status(scratch, stream.get()) ==
                                q4::output_head_status::ok,
                "second deterministic tail execution failed");
        cuda_require(cudaMemcpy(second_logits.data(), scratch.logits,
                                second_logits.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy second logits");
        require(std::memcmp(gpu_logits.data(), second_logits.data(),
                            gpu_logits.size() * sizeof(float)) == 0,
                "full logits are not bitwise deterministic");

        constexpr std::uint32_t invalid_token =
                static_cast<std::uint32_t>(q4::kOutputHeadVocab);
        cuda_require(cudaMemcpyAsync(d_input_token.as<std::uint32_t>(),
                                     &invalid_token, sizeof(invalid_token),
                                     cudaMemcpyHostToDevice, stream.get()),
                     "copy invalid token");
        require(output_head->embedding_repeat(
                        d_input_token.as<std::uint32_t>(), 1u, scratch,
                        scratch.hyper_input, stream.get()) ==
                        q4::output_head_status::ok &&
                        output_head->collect_device_status(scratch, stream.get()) ==
                                q4::output_head_status::device_rejected_input,
                "out-of-range token did not fail closed");

        const auto load_milliseconds =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        load_end - load_start)
                        .count();
        std::printf(
                "qwen4exp-output-head-test: PASS sm=%d%d tied=false "
                "finalizer=global_hyper_connection_mixer resident_bytes=%llu "
                "load_ms=%lld token=%u mixed_abs=%.9g mixed_rel=%.9g "
                "head_subset_abs=%.9g head_subset_rel=%.9g argmax=%u "
                "sample=%u logits_digest=%016llx deterministic=2/2\n",
                properties.major, properties.minor,
                static_cast<unsigned long long>(
                        output_head->resident_weight_bytes()),
                static_cast<long long>(load_milliseconds), source_token,
                mixed_error.max_absolute, mixed_error.max_relative,
                head_subset_max_absolute, head_subset_max_relative, gpu_argmax,
                sample_a,
                static_cast<unsigned long long>(
                        fnv1a(gpu_logits.data(),
                              gpu_logits.size() * sizeof(float))));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-output-head-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
