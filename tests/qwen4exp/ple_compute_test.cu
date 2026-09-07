#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/ple_compute.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        require(count_ != 0u, "zero-sized device buffer");
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data_),
                                count_ * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() { if (data_ != nullptr) (void)cudaFree(data_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    T *data() noexcept { return data_; }
    const T *data() const noexcept { return data_; }
    void upload(const std::vector<T> &values) {
        require(values.size() == count_, "upload size mismatch");
        require_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(T),
                                cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
    std::vector<T> download() const {
        std::vector<T> values(count_);
        require_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T),
                                cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        return values;
    }
private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_f32(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<std::uint16_t> make_weight(std::size_t count,
                                       float scale,
                                       std::size_t seed) {
    std::vector<std::uint16_t> result(count);
    for (std::size_t index = 0u; index < count; ++index) {
        const float value = scale * std::sin(
                static_cast<float>((index + seed) % 997u) * 0.071F);
        result[index] = f32_to_bf16(value);
    }
    return result;
}

struct host_weights {
    std::vector<std::uint16_t> key;
    std::vector<std::uint16_t> value;
    std::vector<std::uint16_t> norm_key;
    std::vector<std::uint16_t> norm_query;
    std::vector<std::uint16_t> norm_conv;
    std::vector<std::uint16_t> conv;
};

struct host_state {
    std::vector<float> committed;
    std::vector<float> staged;
    std::size_t next = 0u;
    std::size_t valid = 0u;
};

std::vector<float> matvec(const std::vector<std::uint16_t> &weight,
                          const std::vector<float> &input,
                          std::size_t rows,
                          std::size_t columns) {
    std::vector<float> result(rows, 0.0F);
    for (std::size_t row = 0u; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0u; column < columns; ++column) {
            sum += static_cast<double>(bf16_to_f32(weight[row * columns + column])) *
                   static_cast<double>(input[column]);
        }
        result[row] = static_cast<float>(sum);
    }
    return result;
}

std::vector<float> group_norm(const std::vector<float> &input,
                              const std::vector<std::uint16_t> &weight,
                              std::size_t hidden,
                              std::size_t streams,
                              float epsilon) {
    std::vector<float> result(input.size());
    for (std::size_t stream = 0u; stream < streams; ++stream) {
        const std::size_t base = stream * hidden;
        double squares = 0.0;
        for (std::size_t column = 0u; column < hidden; ++column) {
            squares += static_cast<double>(input[base + column]) *
                       static_cast<double>(input[base + column]);
        }
        const float inverse = 1.0F / std::sqrt(
                static_cast<float>(squares / static_cast<double>(hidden)) + epsilon);
        for (std::size_t column = 0u; column < hidden; ++column) {
            const std::size_t index = base + column;
            result[index] = input[index] * inverse *
                            (1.0F + bf16_to_f32(weight[index]));
        }
    }
    return result;
}

float prior(const host_state &state,
            std::size_t channels,
            std::size_t staged_index,
            std::size_t distance,
            std::size_t channel) {
    if (distance <= staged_index) {
        return state.staged[(staged_index - distance) * channels + channel];
    }
    const std::size_t committed_distance = distance - staged_index;
    if (committed_distance == 0u || committed_distance > state.valid) return 0.0F;
    const std::size_t frame =
            (state.next + q4::kPleComputeHistory - committed_distance) %
            q4::kPleComputeHistory;
    return state.committed[frame * channels + channel];
}

std::vector<float> host_step(const q4::ple_compute_config &config,
                             const host_weights &weights,
                             const std::vector<float> &embedding,
                             const std::vector<float> &hidden_states,
                             host_state *state) {
    const std::size_t channels = config.hidden * config.streams;
    std::vector<float> key = matvec(
            weights.key, embedding, channels, config.embedding);
    std::vector<float> value = matvec(
            weights.value, embedding, config.hidden, config.embedding);
    key = group_norm(key, weights.norm_key, config.hidden, config.streams,
                     config.rms_epsilon);
    const std::vector<float> query = group_norm(
            hidden_states, weights.norm_query, config.hidden, config.streams,
            config.rms_epsilon);
    std::vector<float> gated(channels);
    for (std::size_t stream = 0u; stream < config.streams; ++stream) {
        const std::size_t base = stream * config.hidden;
        double dot = 0.0;
        for (std::size_t column = 0u; column < config.hidden; ++column) {
            dot += static_cast<double>(key[base + column]) *
                   static_cast<double>(query[base + column]);
        }
        float gate = static_cast<float>(dot) /
                     std::sqrt(static_cast<float>(config.hidden));
        gate = std::copysign(std::sqrt(std::max(std::abs(gate), 1.0e-6F)), gate);
        const float probability = 1.0F / (1.0F + std::exp(-gate));
        for (std::size_t column = 0u; column < config.hidden; ++column) {
            gated[base + column] = probability * value[column];
        }
    }
    const std::vector<float> normalized = group_norm(
            gated, weights.norm_conv, config.hidden, config.streams,
            config.rms_epsilon);
    const std::size_t staged_index = state->staged.size() / channels;
    state->staged.insert(state->staged.end(), normalized.begin(), normalized.end());
    std::vector<float> output(channels);
    for (std::size_t channel = 0u; channel < channels; ++channel) {
        double convolved = 0.0;
        for (std::size_t tap = 0u; tap < q4::kPleComputeTaps; ++tap) {
            const std::size_t distance =
                    (q4::kPleComputeTaps - 1u - tap) * q4::kPleComputeDilation;
            convolved += static_cast<double>(bf16_to_f32(
                    weights.conv[channel * q4::kPleComputeTaps + tap])) *
                    static_cast<double>(prior(*state, channels, staged_index,
                                              distance, channel));
        }
        const float conv = static_cast<float>(convolved);
        output[channel] = gated[channel] + conv / (1.0F + std::exp(-conv));
    }
    return output;
}

void host_commit(host_state *state,
                 std::size_t channels,
                 std::size_t accepted) {
    require(state != nullptr && accepted * channels <= state->staged.size(),
            "invalid host commit");
    for (std::size_t frame = 0u; frame < accepted; ++frame) {
        const std::size_t target =
                (state->next + frame) % q4::kPleComputeHistory;
        std::copy_n(state->staged.data() + frame * channels, channels,
                    state->committed.data() + target * channels);
    }
    state->next = (state->next + accepted) % q4::kPleComputeHistory;
    state->valid = std::min(q4::kPleComputeHistory, state->valid + accepted);
    state->staged.clear();
}

void compare(const std::vector<float> &actual,
             const std::vector<float> &expected,
             float *max_error) {
    require(actual.size() == expected.size(), "comparison size mismatch");
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const float error = std::abs(actual[index] - expected[index]);
        *max_error = std::max(*max_error, error);
        const float tolerance = 2.0e-4F + 3.0e-5F * std::abs(expected[index]);
        require(error <= tolerance, "PLE CUDA output differs from host oracle");
    }
}

void test_compute() {
    q4::ple_compute_config config;
    config.hidden = 32u;
    config.streams = 2u;
    config.embedding = 32u;
    config.max_speculative_tokens = 4u;
    const std::size_t channels = config.hidden * config.streams;
    require(q4::ple_compute_validate_config(config) == q4::ple_compute_status::ok,
            "reduced config rejected");
    require(q4::ple_compute_is_qwen4_exp_contract(
                    q4::ple_compute_qwen4_exp_config()),
            "production contract rejected");
    q4::ple_compute_footprint footprint{};
    require(q4::ple_compute_get_footprint(config, &footprint) ==
                    q4::ple_compute_status::ok &&
            footprint.channels == channels,
            "footprint mismatch");

    host_weights host{
        make_weight(channels * config.embedding, 0.035F, 1u),
        make_weight(config.hidden * config.embedding, 0.04F, 3u),
        make_weight(channels, 0.02F, 5u),
        make_weight(channels, 0.02F, 7u),
        make_weight(channels, 0.02F, 11u),
        make_weight(channels * q4::kPleComputeTaps, 0.025F, 13u)};
    device_buffer<std::uint16_t> d_key(host.key.size()); d_key.upload(host.key);
    device_buffer<std::uint16_t> d_value(host.value.size()); d_value.upload(host.value);
    device_buffer<std::uint16_t> d_norm_key(host.norm_key.size()); d_norm_key.upload(host.norm_key);
    device_buffer<std::uint16_t> d_norm_query(host.norm_query.size()); d_norm_query.upload(host.norm_query);
    device_buffer<std::uint16_t> d_norm_conv(host.norm_conv.size()); d_norm_conv.upload(host.norm_conv);
    device_buffer<std::uint16_t> d_conv(host.conv.size()); d_conv.upload(host.conv);
    const q4::ple_compute_weights weights{
        d_key.data(), d_value.data(), d_norm_key.data(), d_norm_query.data(),
        d_norm_conv.data(), d_conv.data()};
    device_buffer<float> d_embedding(config.embedding);
    device_buffer<float> d_hidden(channels);
    device_buffer<float> d_key_scratch(channels);
    device_buffer<float> d_value_scratch(config.hidden);
    device_buffer<float> d_query_scratch(channels);
    device_buffer<float> d_gated_scratch(channels);
    device_buffer<float> d_normalized_scratch(channels);
    device_buffer<float> d_output(channels);
    const q4::ple_compute_scratch scratch{
        d_key_scratch.data(), d_value_scratch.data(), d_query_scratch.data(),
        d_gated_scratch.data(), d_normalized_scratch.data()};

    q4::ple_compute_device_state device_state{};
    require(q4::ple_compute_device_state_init(&device_state, config) ==
                    q4::ple_compute_status::ok,
            "device state init failed");
    host_state reference{
        std::vector<float>(channels * q4::kPleComputeHistory, 0.0F), {}, 0u, 0u};
    float max_error = 0.0F;
    std::size_t token_seed = 0u;
    auto stage = [&]() {
        std::vector<float> embedding(config.embedding);
        std::vector<float> hidden_states(channels);
        for (std::size_t index = 0u; index < embedding.size(); ++index) {
            embedding[index] = 0.15F * std::cos(
                    static_cast<float>(index + token_seed * 7u) * 0.09F);
        }
        for (std::size_t index = 0u; index < hidden_states.size(); ++index) {
            hidden_states[index] = 0.2F * std::sin(
                    static_cast<float>(index + token_seed * 11u) * 0.05F);
        }
        const std::vector<float> expected = host_step(
                config, host, embedding, hidden_states, &reference);
        d_embedding.upload(embedding);
        d_hidden.upload(hidden_states);
        require(q4::ple_compute_stage_token_cuda(
                        &device_state, weights, d_embedding.data(), d_hidden.data(),
                        scratch, d_output.data()) == q4::ple_compute_status::ok,
                "device PLE stage failed");
        require_cuda(cudaDeviceSynchronize(), "PLE stage synchronize");
        compare(d_output.download(), expected, &max_error);
        ++token_seed;
    };

    require(q4::ple_compute_begin_transaction(&device_state) ==
                    q4::ple_compute_status::ok,
            "transaction A begin failed");
    for (unsigned token = 0u; token < 4u; ++token) stage();
    require(q4::ple_compute_commit_prefix_cuda(&device_state, 2u) ==
                    q4::ple_compute_status::ok,
            "transaction A commit failed");
    host_commit(&reference, channels, 2u);

    require(q4::ple_compute_begin_transaction(&device_state) ==
                    q4::ple_compute_status::ok,
            "transaction B begin failed");
    for (unsigned token = 0u; token < 3u; ++token) stage();
    require(q4::ple_compute_rollback(&device_state) == q4::ple_compute_status::ok,
            "transaction B rollback failed");
    reference.staged.clear();

    require(q4::ple_compute_begin_transaction(&device_state) ==
                    q4::ple_compute_status::ok,
            "transaction C begin failed");
    for (unsigned token = 0u; token < 4u; ++token) stage();
    require(q4::ple_compute_commit_prefix_cuda(&device_state, 4u) ==
                    q4::ple_compute_status::ok,
            "transaction C commit failed");
    host_commit(&reference, channels, 4u);
    require_cuda(cudaDeviceSynchronize(), "PLE commit synchronize");

    std::vector<float> committed(channels * q4::kPleComputeHistory);
    require_cuda(cudaMemcpy(committed.data(), device_state.committed_history,
                            committed.size() * sizeof(float),
                            cudaMemcpyDeviceToHost), "copy committed history");
    compare(committed, reference.committed, &max_error);
    require(device_state.committed_tokens == 6u &&
            device_state.committed_valid == 6u &&
            device_state.committed_next == 6u,
            "transaction counters mismatch");
    require(q4::ple_compute_device_state_release(&device_state) ==
                    q4::ple_compute_status::ok,
            "device state release failed");
    std::printf("qwen4exp-ple-compute synthetic max_abs=%.9g committed=6\n",
                max_error);
}

void test_real_checkpoint(const char *model_root) {
    if (model_root == nullptr) return;
    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    require(q4::checkpoint_catalog::open(model_root, &catalog, &error),
            "could not open real checkpoint: " + error);
    struct expected_tensor {
        const char *suffix;
        std::uint64_t rows;
        std::uint64_t columns;
    };
    const expected_tensor tensors[] = {
        {"key_proj.weight", 10240u, 2560u},
        {"value_proj.weight", 2560u, 2560u},
        {"norm_key.weight", 1u, 10240u},
        {"norm_query.weight", 1u, 10240u},
        {"norm_conv.weight", 1u, 10240u},
        {"conv1d.weight", 10240u, 4u},
    };
    std::size_t finite_samples = 0u;
    for (const expected_tensor &expected : tensors) {
        const std::string name =
                "model.language_model.layers.1.ple." + std::string(expected.suffix);
        const q4::tensor_span *span = catalog->find(name);
        require(span != nullptr && span->dtype == q4::tensor_dtype::bf16,
                "missing real PLE tensor: " + name);
        if (std::string(expected.suffix).find("norm_") == 0u) {
            require(span->rank == 1u && span->shape[0] == expected.columns,
                    "real PLE norm geometry mismatch");
        } else if (std::string(expected.suffix) == "conv1d.weight") {
            require(span->rank == 3u && span->shape[0] == expected.rows &&
                    span->shape[1] == 1u && span->shape[2] == expected.columns,
                    "real PLE conv geometry mismatch");
        } else {
            require(span->rank == 2u && span->shape[0] == expected.rows &&
                    span->shape[1] == expected.columns,
                    "real PLE projection geometry mismatch");
        }
        const std::size_t samples = static_cast<std::size_t>(
                std::min<std::uint64_t>(span->bytes / 2u, 16u));
        std::vector<std::uint16_t> bits(samples);
        require(catalog->read_range(name, 0u, bits.data(), bits.size() * 2u, &error),
                "real PLE sample read failed: " + error);
        for (std::uint16_t item : bits) {
            require(std::isfinite(bf16_to_f32(item)),
                    "real PLE tensor contains non-finite sample");
            ++finite_samples;
        }
    }
    std::printf("qwen4exp-ple-compute checkpoint tensors=6 finite_samples=%zu\n",
                finite_samples);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        int devices = 0;
        require_cuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
        require(devices > 0, "PLE compute test requires a CUDA device");
        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        test_compute();
        test_real_checkpoint(argc == 2 ? argv[1] : nullptr);
        std::puts("qwen4exp-ple-compute-test: pass");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "qwen4exp-ple-compute-test: %s\n", error.what());
        return 1;
    }
}
