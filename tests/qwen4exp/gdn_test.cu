#include "axiom/qwen4exp/gdn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

void require_status(q4::GdnStatus actual, q4::GdnStatus expected, const std::string& label) {
    if (actual != expected) {
        fail(label + ": expected " + q4::gdn_status_string(expected) + ", got " +
             q4::gdn_status_string(actual));
    }
}

void require_cuda(cudaError_t status, const std::string& label) {
    if (status != cudaSuccess) {
        fail(label + ": " + cudaGetErrorString(status));
    }
}

float deterministic_value(std::uint64_t seed, std::size_t index, float scale) {
    std::uint64_t value = seed + 0x9E3779B97F4A7C15ULL * (index + 1ULL);
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    const std::int32_t signed_value =
        static_cast<std::int32_t>((value >> 32U) & 0xFFFFU) - 32768;
    return scale * static_cast<float>(signed_value) / 32768.0F;
}

void fill_deterministic(std::vector<float>* values, std::uint64_t seed, float scale) {
    require(values != nullptr, "fill_deterministic null vector");
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = deterministic_value(seed, index, scale);
    }
}

float sigmoid_reference(float value) {
    if (value >= 0.0F) {
        return 1.0F / (1.0F + std::exp(-value));
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

float softplus_reference(float value) {
    if (value > 20.0F) {
        return value;
    }
    if (value < -20.0F) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

struct Comparison {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
    std::size_t max_index = 0;
};

Comparison compare_vectors(
    const std::vector<float>& expected,
    const std::vector<float>& actual,
    float absolute_tolerance,
    float relative_tolerance,
    const std::string& label) {
    require(expected.size() == actual.size(), label + ": size mismatch");
    Comparison result{};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::isfinite(expected[index]), label + ": non-finite expected value");
        require(std::isfinite(actual[index]), label + ": non-finite actual value");
        const float absolute = std::abs(actual[index] - expected[index]);
        const float relative = absolute / std::max(std::abs(expected[index]), 1.0e-7F);
        if (absolute > result.max_absolute) {
            result.max_absolute = absolute;
            result.max_index = index;
        }
        result.max_relative = std::max(result.max_relative, relative);
        const float allowed = absolute_tolerance + relative_tolerance * std::abs(expected[index]);
        if (absolute > allowed) {
            fail(label + ": mismatch at " + std::to_string(index) +
                 ", expected=" + std::to_string(expected[index]) +
                 ", actual=" + std::to_string(actual[index]) +
                 ", abs=" + std::to_string(absolute) +
                 ", allowed=" + std::to_string(allowed));
        }
    }
    return result;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        require(count_ > 0, "DeviceBuffer cannot be empty");
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
            "cudaMalloc DeviceBuffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    void copy_from(const std::vector<T>& host, cudaStream_t stream) {
        require(host.size() == count_, "DeviceBuffer copy_from size mismatch");
        require_cuda(
            cudaMemcpyAsync(
                data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D");
    }

    void copy_to(std::vector<T>* host, cudaStream_t stream) const {
        require(host != nullptr && host->size() == count_, "DeviceBuffer copy_to size mismatch");
        require_cuda(
            cudaMemcpyAsync(
                host->data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync D2H");
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() {
        if (stream_ != nullptr) {
            (void)cudaStreamDestroy(stream_);
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { require_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }

private:
    cudaStream_t stream_ = nullptr;
};

struct OracleState {
    std::vector<float> chronological_conv;
    std::vector<float> recurrent;
    std::uint64_t tokens_seen = 0;
};

OracleState make_oracle_state(const q4::GdnFootprint& footprint) {
    OracleState state{};
    state.chronological_conv.assign(footprint.conv_state_floats, 0.0F);
    state.recurrent.assign(footprint.recurrent_state_floats, 0.0F);
    return state;
}

// Independent, straightforward scalar transcription of the official
// Transformers single-token path. Convolution history is shifted
// chronologically rather than using the module's ring layout.
void oracle_step(
    const q4::GdnConfig& config,
    const q4::GdnFootprint& footprint,
    const std::vector<float>& projected_qkv,
    const std::vector<float>& z,
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& conv_weight,
    const std::vector<float>& A_log,
    const std::vector<float>& dt_bias,
    const std::vector<float>& norm_weight,
    OracleState* state,
    std::vector<float>* convolved_qkv,
    std::vector<float>* output) {
    require(state != nullptr && convolved_qkv != nullptr && output != nullptr, "oracle null output");
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        for (std::size_t channel = 0; channel < footprint.conv_channels; ++channel) {
            const std::size_t token_index = batch_index * footprint.conv_channels + channel;
            float* history = state->chronological_conv.data() +
                token_index * config.conv_kernel;
            for (std::size_t tap = 0; tap + 1 < config.conv_kernel; ++tap) {
                history[tap] = history[tap + 1];
            }
            history[config.conv_kernel - 1] = projected_qkv[token_index];
            float sum = 0.0F;
            for (std::size_t tap = 0; tap < config.conv_kernel; ++tap) {
                sum += conv_weight[channel * config.conv_kernel + tap] * history[tap];
            }
            (*convolved_qkv)[token_index] = sum * sigmoid_reference(sum);
        }
    }

    const std::size_t repeat = config.value_heads / config.key_heads;
    const float query_scale = 1.0F / std::sqrt(static_cast<float>(config.key_head_dim));
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        const float* token = convolved_qkv->data() + batch_index * footprint.conv_channels;
        for (std::size_t value_head = 0; value_head < config.value_heads; ++value_head) {
            const std::size_t key_head = value_head / repeat;
            const float* query = token + key_head * config.key_head_dim;
            const float* key = token + footprint.key_elements + key_head * config.key_head_dim;
            const float* value = token + 2 * footprint.key_elements +
                                 value_head * config.value_head_dim;
            float* matrix = state->recurrent.data() +
                (batch_index * config.value_heads + value_head) *
                    config.key_head_dim * config.value_head_dim;
            float* head_output = output->data() +
                (batch_index * config.value_heads + value_head) * config.value_head_dim;
            const float* head_z = z.data() +
                (batch_index * config.value_heads + value_head) * config.value_head_dim;

            float query_sum = 0.0F;
            float key_sum = 0.0F;
            for (std::size_t key_index = 0; key_index < config.key_head_dim; ++key_index) {
                query_sum += query[key_index] * query[key_index];
                key_sum += key[key_index] * key[key_index];
            }
            const float inv_query = 1.0F / std::sqrt(query_sum + config.l2_epsilon);
            const float inv_key = 1.0F / std::sqrt(key_sum + config.l2_epsilon);
            const std::size_t scalar_index = batch_index * config.value_heads + value_head;
            const float g = -std::exp(A_log[value_head]) *
                            softplus_reference(a[scalar_index] + dt_bias[value_head]);
            const float decay = std::exp(g);
            const float beta = sigmoid_reference(b[scalar_index]);

            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                float memory_value = 0.0F;
                for (std::size_t key_index = 0; key_index < config.key_head_dim;
                     ++key_index) {
                    const std::size_t matrix_index =
                        key_index * config.value_head_dim + value_index;
                    matrix[matrix_index] *= decay;
                    memory_value += matrix[matrix_index] * key[key_index] * inv_key;
                }
                const float delta = (value[value_index] - memory_value) * beta;
                float queried = 0.0F;
                for (std::size_t key_index = 0; key_index < config.key_head_dim;
                     ++key_index) {
                    const std::size_t matrix_index =
                        key_index * config.value_head_dim + value_index;
                    matrix[matrix_index] += key[key_index] * inv_key * delta;
                    queried += matrix[matrix_index] * query[key_index] * inv_query;
                }
                head_output[value_index] = queried * query_scale;
            }
            float square_sum = 0.0F;
            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                square_sum += head_output[value_index] * head_output[value_index];
            }
            const float rms_inverse = 1.0F / std::sqrt(
                square_sum / static_cast<float>(config.value_head_dim) + config.rms_epsilon);
            for (std::size_t value_index = 0; value_index < config.value_head_dim;
                 ++value_index) {
                head_output[value_index] = head_output[value_index] * rms_inverse *
                    norm_weight[value_index] * sigmoid_reference(head_z[value_index]);
            }
        }
    }
    ++state->tokens_seen;
}

void test_contract_fail_closed_and_real_state(std::size_t* real_state_bytes) {
    const q4::GdnConfig exact = q4::gdn_qwen4_exp_config();
    require_status(q4::gdn_validate_config(exact), q4::GdnStatus::kOk, "real config");
    require(q4::gdn_is_qwen4_exp_contract(exact), "real contract not recognized");
    q4::GdnFootprint footprint{};
    require_status(q4::gdn_footprint(exact, &footprint), q4::GdnStatus::kOk, "real footprint");
    require(footprint.key_elements == 2048, "real key elements mismatch");
    require(footprint.value_elements == 6144, "real value elements mismatch");
    require(footprint.conv_channels == 10240, "real convolution channels mismatch");
    require(footprint.conv_state_floats == 40960, "real convolution state mismatch");
    require(footprint.recurrent_state_floats == 786432, "real recurrent state mismatch");
    require(footprint.step_workspace_floats == 10240, "real workspace mismatch");
    require(footprint.device_state_bytes == 3309568, "real device bytes mismatch");
    require(footprint.device_state_bytes < 4U * 1024U * 1024U, "real state unexpectedly large");
    require_status(
        q4::gdn_footprint(exact, nullptr),
        q4::GdnStatus::kInvalidArgument,
        "null footprint");

    q4::GdnConfig invalid = exact;
    invalid.value_heads = 47;
    require_status(
        q4::gdn_validate_config(invalid),
        q4::GdnStatus::kUnsupportedConfig,
        "non-integral head repeat");
    invalid = exact;
    invalid.conv_kernel = 3;
    require_status(
        q4::gdn_validate_config(invalid),
        q4::GdnStatus::kUnsupportedConfig,
        "wrong convolution kernel");
    invalid = exact;
    invalid.key_head_dim = 0;
    require_status(
        q4::gdn_validate_config(invalid),
        q4::GdnStatus::kInvalidArgument,
        "zero head dimension");
    invalid = exact;
    invalid.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
    require_status(
        q4::gdn_validate_config(invalid),
        q4::GdnStatus::kInvalidArgument,
        "NaN epsilon");
    invalid = exact;
    invalid.batch = q4::kGdnMaxBatch + 1;
    require_status(
        q4::gdn_validate_config(invalid),
        q4::GdnStatus::kUnsupportedConfig,
        "oversized batch");

    q4::GdnDeviceState uninitialized{};
    require_status(
        q4::gdn_device_state_reset(&uninitialized),
        q4::GdnStatus::kUninitialized,
        "reset uninitialized");

    cudaDeviceProp properties{};
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    require_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    require(properties.major == 12 && properties.minor == 0, "test requires SM120");

    Stream stream;
    q4::GdnDeviceState real_state{};
    require_status(
        q4::gdn_device_state_init(&real_state, exact, stream.get()),
        q4::GdnStatus::kOk,
        "real state init");
    require(real_state.initialized, "real state not initialized");
    require(real_state.conv_state_floats == footprint.conv_state_floats, "real conv count");
    require(
        real_state.recurrent_state_floats == footprint.recurrent_state_floats,
        "real recurrent count");
    require_status(
        q4::gdn_device_state_reset(&real_state, stream.get()),
        q4::GdnStatus::kOk,
        "real state reset");
    stream.synchronize();
    require_status(
        q4::gdn_device_state_release(&real_state),
        q4::GdnStatus::kOk,
        "real state release");
    require(!real_state.initialized && real_state.conv_state == nullptr, "real state not cleared");
    *real_state_bytes = footprint.device_state_bytes;
}

void test_analytic_sigmoid_gated_rmsnorm() {
    q4::GdnConfig config{};
    config.key_heads = 1;
    config.value_heads = 1;
    config.key_head_dim = 2;
    config.value_head_dim = 2;
    q4::GdnFootprint footprint{};
    require_status(q4::gdn_footprint(config, &footprint), q4::GdnStatus::kOk, "analytic footprint");

    std::vector<float> conv_state(footprint.conv_state_floats, 0.0F);
    std::vector<float> recurrent_state(footprint.recurrent_state_floats, 0.0F);
    std::uint32_t cursor = 0;
    std::uint64_t tokens = 0;
    const q4::GdnHostStateView state{
        conv_state.data(), conv_state.size(), recurrent_state.data(), recurrent_state.size(),
        &cursor, &tokens};
    const std::vector<float> convolved_qkv{1.0F, 0.0F, 1.0F, 0.0F, 2.0F, 4.0F};
    const std::vector<float> z{0.0F, 0.0F};
    const std::vector<float> a{0.0F};
    const std::vector<float> b{0.0F};
    const std::vector<float> A_log{0.0F};
    const std::vector<float> dt_bias{0.0F};
    const std::vector<float> norm_weight{1.0F, 2.0F};
    std::vector<float> output(2, 0.0F);
    q4::GdnWeightsView weights{};
    weights.A_log = A_log.data();
    weights.dt_bias = dt_bias.data();
    weights.norm_weight = norm_weight.data();
    require_status(
        q4::gdn_recurrent_delta_update_host(
            config,
            convolved_qkv.data(),
            z.data(),
            a.data(),
            b.data(),
            weights,
            state,
            output.data()),
        q4::GdnStatus::kOk,
        "analytic recurrent update");

    const float normalized_component = 1.0F / std::sqrt(1.0F + config.l2_epsilon);
    const float raw0 = normalized_component * normalized_component * 1.0F /
                       std::sqrt(2.0F);
    const float raw1 = normalized_component * normalized_component * 2.0F /
                       std::sqrt(2.0F);
    const float rms_inverse = 1.0F / std::sqrt(
        (raw0 * raw0 + raw1 * raw1) / 2.0F + config.rms_epsilon);
    const std::vector<float> expected{
        raw0 * rms_inverse * 1.0F * 0.5F,
        raw1 * rms_inverse * 2.0F * 0.5F};
    (void)compare_vectors(expected, output, 2.0e-6F, 2.0e-6F, "analytic sigmoid RMSNorm");
}

float test_short_sequence_host_cuda() {
    q4::GdnConfig config{};
    config.batch = 2;
    config.key_heads = 2;
    config.value_heads = 6;
    config.key_head_dim = 8;
    config.value_head_dim = 8;
    q4::GdnFootprint footprint{};
    require_status(q4::gdn_footprint(config, &footprint), q4::GdnStatus::kOk, "small footprint");

    std::vector<float> conv_weight(footprint.conv_channels * config.conv_kernel);
    std::vector<float> A_log(config.value_heads);
    std::vector<float> dt_bias(config.value_heads);
    std::vector<float> norm_weight(config.value_head_dim);
    fill_deterministic(&conv_weight, 11, 0.12F);
    for (std::size_t channel = 0; channel < footprint.conv_channels; ++channel) {
        conv_weight[channel * config.conv_kernel + config.conv_kernel - 1] += 0.75F;
    }
    for (std::size_t head = 0; head < config.value_heads; ++head) {
        A_log[head] = std::log(0.05F + 0.025F * static_cast<float>(head + 1));
        dt_bias[head] = deterministic_value(17, head, 0.25F);
    }
    for (std::size_t index = 0; index < norm_weight.size(); ++index) {
        norm_weight[index] = 1.0F + deterministic_value(19, index, 0.15F);
    }
    const q4::GdnWeightsView host_weights{
        conv_weight.data(), nullptr, A_log.data(), dt_bias.data(), norm_weight.data()};

    std::vector<float> host_conv_state(footprint.conv_state_floats, 0.0F);
    std::vector<float> host_recurrent_state(footprint.recurrent_state_floats, 0.0F);
    std::uint32_t host_cursor = 0;
    std::uint64_t host_tokens = 0;
    const q4::GdnHostStateView host_state{
        host_conv_state.data(), host_conv_state.size(),
        host_recurrent_state.data(), host_recurrent_state.size(),
        &host_cursor, &host_tokens};
    require_status(
        q4::gdn_host_state_reset(config, host_state),
        q4::GdnStatus::kOk,
        "small host reset");
    OracleState oracle = make_oracle_state(footprint);

    Stream stream;
    q4::GdnDeviceState device_state{};
    require_status(
        q4::gdn_device_state_init(&device_state, config, stream.get()),
        q4::GdnStatus::kOk,
        "small device state init");

    DeviceBuffer<float> d_conv_weight(conv_weight.size());
    DeviceBuffer<float> d_A_log(A_log.size());
    DeviceBuffer<float> d_dt_bias(dt_bias.size());
    DeviceBuffer<float> d_norm_weight(norm_weight.size());
    DeviceBuffer<float> d_projected(footprint.step_workspace_floats);
    DeviceBuffer<float> d_z(config.batch * config.value_heads * config.value_head_dim);
    DeviceBuffer<float> d_a(config.batch * config.value_heads);
    DeviceBuffer<float> d_b(config.batch * config.value_heads);
    DeviceBuffer<float> d_convolved(footprint.step_workspace_floats);
    DeviceBuffer<float> d_output(config.batch * config.value_heads * config.value_head_dim);
    d_conv_weight.copy_from(conv_weight, stream.get());
    d_A_log.copy_from(A_log, stream.get());
    d_dt_bias.copy_from(dt_bias, stream.get());
    d_norm_weight.copy_from(norm_weight, stream.get());
    const q4::GdnWeightsView device_weights{
        d_conv_weight.data(), nullptr, d_A_log.data(), d_dt_bias.data(), d_norm_weight.data()};

    std::vector<float> first_projected;
    std::vector<float> first_z;
    std::vector<float> first_a;
    std::vector<float> first_b;
    std::vector<float> first_expected_output;
    float max_cuda_absolute = 0.0F;
    constexpr std::size_t kSequenceLength = 7;
    for (std::size_t token = 0; token < kSequenceLength; ++token) {
        std::vector<float> projected(footprint.step_workspace_floats);
        std::vector<float> z(config.batch * config.value_heads * config.value_head_dim);
        std::vector<float> a(config.batch * config.value_heads);
        std::vector<float> b(config.batch * config.value_heads);
        fill_deterministic(&projected, 1000 + token * 13, 0.45F);
        fill_deterministic(&z, 2000 + token * 17, 1.2F);
        fill_deterministic(&a, 3000 + token * 19, 0.55F);
        fill_deterministic(&b, 4000 + token * 23, 0.8F);

        std::vector<float> oracle_convolved(footprint.step_workspace_floats, 0.0F);
        std::vector<float> oracle_output(
            config.batch * config.value_heads * config.value_head_dim, 0.0F);
        oracle_step(
            config,
            footprint,
            projected,
            z,
            a,
            b,
            conv_weight,
            A_log,
            dt_bias,
            norm_weight,
            &oracle,
            &oracle_convolved,
            &oracle_output);

        std::vector<float> host_convolved(footprint.step_workspace_floats, 0.0F);
        std::vector<float> host_output(oracle_output.size(), 0.0F);
        const q4::GdnStepInputs host_inputs{
            projected.data(), z.data(), a.data(), b.data()};
        const q4::GdnStepOutputs host_outputs{host_convolved.data(), host_output.data()};
        require_status(
            q4::gdn_step_host(config, host_inputs, host_weights, host_state, host_outputs),
            q4::GdnStatus::kOk,
            "host sequence step");
        (void)compare_vectors(
            oracle_convolved, host_convolved, 2.0e-6F, 2.0e-6F, "host causal conv");
        (void)compare_vectors(
            oracle_output, host_output, 3.0e-6F, 3.0e-6F, "host recurrent output");

        d_projected.copy_from(projected, stream.get());
        d_z.copy_from(z, stream.get());
        d_a.copy_from(a, stream.get());
        d_b.copy_from(b, stream.get());
        const q4::GdnStepInputs device_inputs{
            d_projected.data(), d_z.data(), d_a.data(), d_b.data()};
        const q4::GdnStepOutputs device_outputs{d_convolved.data(), d_output.data()};
        require_status(
            q4::gdn_step_cuda(
                &device_state, device_inputs, device_weights, device_outputs, stream.get()),
            q4::GdnStatus::kOk,
            "CUDA sequence step");
        std::vector<float> cuda_convolved(oracle_convolved.size(), 0.0F);
        std::vector<float> cuda_output(oracle_output.size(), 0.0F);
        d_convolved.copy_to(&cuda_convolved, stream.get());
        d_output.copy_to(&cuda_output, stream.get());
        stream.synchronize();
        const Comparison conv_comparison = compare_vectors(
            oracle_convolved, cuda_convolved, 4.0e-6F, 4.0e-6F, "CUDA causal conv");
        const Comparison output_comparison = compare_vectors(
            oracle_output, cuda_output, 4.0e-5F, 5.0e-5F, "CUDA recurrent output");
        max_cuda_absolute = std::max(
            max_cuda_absolute,
            std::max(conv_comparison.max_absolute, output_comparison.max_absolute));

        if (token == 0) {
            first_projected = projected;
            first_z = z;
            first_a = a;
            first_b = b;
            first_expected_output = oracle_output;
        }
    }

    require(host_tokens == kSequenceLength, "host token counter mismatch");
    require(device_state.tokens_seen == kSequenceLength, "CUDA token counter mismatch");
    require(host_cursor == kSequenceLength % config.conv_kernel, "host cursor mismatch");
    require(
        device_state.conv_cursor == kSequenceLength % config.conv_kernel,
        "CUDA cursor mismatch");
    (void)compare_vectors(
        oracle.recurrent,
        host_recurrent_state,
        4.0e-6F,
        4.0e-6F,
        "host final recurrent state");
    std::vector<float> cuda_recurrent(footprint.recurrent_state_floats, 0.0F);
    require_cuda(
        cudaMemcpyAsync(
            cuda_recurrent.data(),
            device_state.recurrent_state,
            cuda_recurrent.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream.get()),
        "copy recurrent state");
    stream.synchronize();
    const Comparison state_comparison = compare_vectors(
        oracle.recurrent,
        cuda_recurrent,
        5.0e-5F,
        6.0e-5F,
        "CUDA final recurrent state");
    max_cuda_absolute = std::max(max_cuda_absolute, state_comparison.max_absolute);

    // Reset must reproduce token zero. Exercise the explicit conv and recurrent
    // APIs separately rather than the composed helper.
    require_status(
        q4::gdn_host_state_reset(config, host_state),
        q4::GdnStatus::kOk,
        "host reproducibility reset");
    require_status(
        q4::gdn_device_state_reset(&device_state, stream.get()),
        q4::GdnStatus::kOk,
        "CUDA reproducibility reset");
    std::vector<float> host_convolved(footprint.step_workspace_floats, 0.0F);
    std::vector<float> host_output(first_expected_output.size(), 0.0F);
    require_status(
        q4::gdn_causal_conv_update_host(
            config, first_projected.data(), host_weights, host_state, host_convolved.data()),
        q4::GdnStatus::kOk,
        "host explicit conv");
    require_status(
        q4::gdn_recurrent_delta_update_host(
            config,
            host_convolved.data(),
            first_z.data(),
            first_a.data(),
            first_b.data(),
            host_weights,
            host_state,
            host_output.data()),
        q4::GdnStatus::kOk,
        "host explicit recurrence");
    (void)compare_vectors(
        first_expected_output, host_output, 3.0e-6F, 3.0e-6F, "host reset reproducibility");

    d_projected.copy_from(first_projected, stream.get());
    d_z.copy_from(first_z, stream.get());
    d_a.copy_from(first_a, stream.get());
    d_b.copy_from(first_b, stream.get());
    require_status(
        q4::gdn_causal_conv_update_cuda(
            &device_state,
            d_projected.data(),
            device_weights,
            d_convolved.data(),
            stream.get()),
        q4::GdnStatus::kOk,
        "CUDA explicit conv");
    require_status(
        q4::gdn_recurrent_delta_update_cuda(
            &device_state,
            d_convolved.data(),
            d_z.data(),
            d_a.data(),
            d_b.data(),
            device_weights,
            d_output.data(),
            stream.get()),
        q4::GdnStatus::kOk,
        "CUDA explicit recurrence");
    std::vector<float> reset_cuda_output(first_expected_output.size(), 0.0F);
    d_output.copy_to(&reset_cuda_output, stream.get());
    stream.synchronize();
    const Comparison reset_comparison = compare_vectors(
        first_expected_output,
        reset_cuda_output,
        4.0e-5F,
        5.0e-5F,
        "CUDA reset reproducibility");
    max_cuda_absolute = std::max(max_cuda_absolute, reset_comparison.max_absolute);

    // All arguments are checked before the composed call mutates state.
    require_status(
        q4::gdn_device_state_reset(&device_state, stream.get()),
        q4::GdnStatus::kOk,
        "CUDA fail-closed reset");
    q4::GdnStepInputs invalid_inputs{
        d_projected.data(), nullptr, d_a.data(), d_b.data()};
    const q4::GdnStepOutputs valid_outputs{d_convolved.data(), d_output.data()};
    require_status(
        q4::gdn_step_cuda(
            &device_state, invalid_inputs, device_weights, valid_outputs, stream.get()),
        q4::GdnStatus::kInvalidArgument,
        "CUDA null z fail closed");
    require(device_state.tokens_seen == 0, "invalid CUDA step mutated token counter");
    device_state.tokens_seen = std::numeric_limits<std::uint64_t>::max();
    require_status(
        q4::gdn_causal_conv_update_cuda(
            &device_state,
            d_projected.data(),
            device_weights,
            d_convolved.data(),
            stream.get()),
        q4::GdnStatus::kSizeOverflow,
        "CUDA token counter overflow");
    device_state.tokens_seen = 0;

    require_status(
        q4::gdn_device_state_release(&device_state),
        q4::GdnStatus::kOk,
        "small device state release");
    return max_cuda_absolute;
}

float test_contiguous_sequence_block() {
    constexpr std::size_t kTokenCount = q4::kGdnMaxSequenceTokens;
    q4::GdnConfig config{};
    config.batch = 1u;
    config.key_heads = 2u;
    config.value_heads = 6u;
    config.key_head_dim = 8u;
    config.value_head_dim = 8u;
    q4::GdnFootprint footprint{};
    require_status(q4::gdn_footprint(config, &footprint),
                   q4::GdnStatus::kOk,
                   "sequence footprint");

    std::vector<float> conv_weight(
        footprint.conv_channels * config.conv_kernel);
    std::vector<float> A_log(config.value_heads);
    std::vector<float> dt_bias(config.value_heads);
    std::vector<float> norm_weight(config.value_head_dim);
    fill_deterministic(&conv_weight, 5101u, 0.12F);
    for (std::size_t channel = 0u; channel < footprint.conv_channels; ++channel) {
        conv_weight[channel * config.conv_kernel + config.conv_kernel - 1u] +=
            0.75F;
    }
    for (std::size_t head = 0u; head < config.value_heads; ++head) {
        A_log[head] = std::log(0.06F + 0.02F * static_cast<float>(head + 1u));
        dt_bias[head] = deterministic_value(5103u, head, 0.2F);
    }
    for (std::size_t index = 0u; index < norm_weight.size(); ++index) {
        norm_weight[index] = 1.0F + deterministic_value(5105u, index, 0.1F);
    }
    const q4::GdnWeightsView host_weights{
        conv_weight.data(), nullptr, A_log.data(), dt_bias.data(),
        norm_weight.data()};

    const std::size_t qkv_rows = kTokenCount * footprint.conv_channels;
    const std::size_t value_rows = kTokenCount * footprint.value_elements;
    const std::size_t scalar_rows = kTokenCount * config.value_heads;
    std::vector<float> projected(qkv_rows);
    std::vector<float> z(value_rows);
    std::vector<float> a(scalar_rows);
    std::vector<float> b(scalar_rows);
    fill_deterministic(&projected, 5201u, 0.4F);
    fill_deterministic(&z, 5203u, 1.1F);
    fill_deterministic(&a, 5205u, 0.5F);
    fill_deterministic(&b, 5207u, 0.7F);

    std::vector<float> prime_projected(footprint.conv_channels);
    std::vector<float> prime_z(footprint.value_elements);
    std::vector<float> prime_a(config.value_heads);
    std::vector<float> prime_b(config.value_heads);
    fill_deterministic(&prime_projected, 5301u, 0.35F);
    fill_deterministic(&prime_z, 5303u, 1.0F);
    fill_deterministic(&prime_a, 5305u, 0.45F);
    fill_deterministic(&prime_b, 5307u, 0.65F);

    struct HostOwnedState {
        std::vector<float> conv;
        std::vector<float> recurrent;
        std::uint32_t cursor = 0u;
        std::uint64_t tokens = 0u;
        q4::GdnHostStateView view{};

        explicit HostOwnedState(const q4::GdnFootprint& size)
            : conv(size.conv_state_floats, 0.0F),
              recurrent(size.recurrent_state_floats, 0.0F) {
            view = q4::GdnHostStateView{
                conv.data(), conv.size(), recurrent.data(), recurrent.size(),
                &cursor, &tokens};
        }
    } host_repeated(footprint), host_block(footprint);
    require_status(q4::gdn_host_state_reset(config, host_repeated.view),
                   q4::GdnStatus::kOk,
                   "repeated host reset");
    require_status(q4::gdn_host_state_reset(config, host_block.view),
                   q4::GdnStatus::kOk,
                   "block host reset");

    std::vector<float> prime_convolved(footprint.conv_channels, 0.0F);
    std::vector<float> prime_output(footprint.value_elements, 0.0F);
    const q4::GdnStepInputs prime_inputs{
        prime_projected.data(), prime_z.data(), prime_a.data(), prime_b.data()};
    const q4::GdnStepOutputs prime_outputs{
        prime_convolved.data(), prime_output.data()};
    require_status(q4::gdn_step_host(
                       config, prime_inputs, host_weights,
                       host_repeated.view, prime_outputs),
                   q4::GdnStatus::kOk,
                   "prime repeated host state");
    require_status(q4::gdn_step_host(
                       config, prime_inputs, host_weights,
                       host_block.view, prime_outputs),
                   q4::GdnStatus::kOk,
                   "prime block host state");

    std::vector<float> host_repeated_convolved(qkv_rows, 0.0F);
    std::vector<float> host_repeated_output(value_rows, 0.0F);
    for (std::size_t token = 0u; token < kTokenCount; ++token) {
        const q4::GdnStepInputs token_inputs{
            projected.data() + token * footprint.conv_channels,
            z.data() + token * footprint.value_elements,
            a.data() + token * config.value_heads,
            b.data() + token * config.value_heads};
        const q4::GdnStepOutputs token_outputs{
            host_repeated_convolved.data() + token * footprint.conv_channels,
            host_repeated_output.data() + token * footprint.value_elements};
        require_status(q4::gdn_step_host(
                           config, token_inputs, host_weights,
                           host_repeated.view, token_outputs),
                       q4::GdnStatus::kOk,
                       "repeated host sequence token");
    }
    std::vector<float> host_block_convolved(qkv_rows, 0.0F);
    std::vector<float> host_block_output(value_rows, 0.0F);
    const q4::GdnStepInputs sequence_inputs{
        projected.data(), z.data(), a.data(), b.data()};
    const q4::GdnStepOutputs host_sequence_outputs{
        host_block_convolved.data(), host_block_output.data()};
    require_status(q4::gdn_forward_sequence_host(
                       config, kTokenCount, sequence_inputs, host_weights,
                       host_block.view, host_sequence_outputs),
                   q4::GdnStatus::kOk,
                   "host contiguous sequence");
    (void)compare_vectors(host_repeated_convolved, host_block_convolved,
                          0.0F, 0.0F, "host sequence convolution");
    (void)compare_vectors(host_repeated_output, host_block_output,
                          0.0F, 0.0F, "host sequence output");
    (void)compare_vectors(host_repeated.conv, host_block.conv,
                          0.0F, 0.0F, "host sequence convolution state");
    (void)compare_vectors(host_repeated.recurrent, host_block.recurrent,
                          0.0F, 0.0F, "host sequence recurrent state");
    require(host_block.tokens == 1u + kTokenCount &&
                host_block.cursor == (1u + kTokenCount) % config.conv_kernel,
            "host sequence metadata advanced incorrectly");

    const std::uint64_t host_tokens_before_failure = host_block.tokens;
    const std::uint32_t host_cursor_before_failure = host_block.cursor;
    require_status(q4::gdn_forward_sequence_host(
                       config, 0u, sequence_inputs, host_weights,
                       host_block.view, host_sequence_outputs),
                   q4::GdnStatus::kUnsupportedConfig,
                   "zero-width host sequence rejection");
    q4::GdnStepInputs invalid_host_inputs = sequence_inputs;
    invalid_host_inputs.z = nullptr;
    require_status(q4::gdn_forward_sequence_host(
                       config, kTokenCount, invalid_host_inputs, host_weights,
                       host_block.view, host_sequence_outputs),
                   q4::GdnStatus::kInvalidArgument,
                   "null host sequence input rejection");
    require(host_block.tokens == host_tokens_before_failure &&
                host_block.cursor == host_cursor_before_failure,
            "invalid host sequence mutated metadata");

    Stream stream;
    q4::GdnDeviceState device_repeated{};
    q4::GdnDeviceState device_block{};
    require_status(q4::gdn_device_state_init(
                       &device_repeated, config, stream.get()),
                   q4::GdnStatus::kOk,
                   "repeated device state init");
    require_status(q4::gdn_device_state_init(
                       &device_block, config, stream.get()),
                   q4::GdnStatus::kOk,
                   "block device state init");

    DeviceBuffer<float> d_conv_weight(conv_weight.size());
    DeviceBuffer<float> d_A_log(A_log.size());
    DeviceBuffer<float> d_dt_bias(dt_bias.size());
    DeviceBuffer<float> d_norm_weight(norm_weight.size());
    DeviceBuffer<float> d_projected(qkv_rows);
    DeviceBuffer<float> d_z(value_rows);
    DeviceBuffer<float> d_a(scalar_rows);
    DeviceBuffer<float> d_b(scalar_rows);
    DeviceBuffer<float> d_repeated_convolved(qkv_rows);
    DeviceBuffer<float> d_repeated_output(value_rows);
    DeviceBuffer<float> d_block_convolved(qkv_rows);
    DeviceBuffer<float> d_block_output(value_rows);
    d_conv_weight.copy_from(conv_weight, stream.get());
    d_A_log.copy_from(A_log, stream.get());
    d_dt_bias.copy_from(dt_bias, stream.get());
    d_norm_weight.copy_from(norm_weight, stream.get());
    const q4::GdnWeightsView device_weights{
        d_conv_weight.data(), nullptr, d_A_log.data(), d_dt_bias.data(),
        d_norm_weight.data()};

    std::vector<float> upload_projected(qkv_rows, 0.0F);
    std::vector<float> upload_z(value_rows, 0.0F);
    std::vector<float> upload_a(scalar_rows, 0.0F);
    std::vector<float> upload_b(scalar_rows, 0.0F);
    std::copy(prime_projected.begin(), prime_projected.end(),
              upload_projected.begin());
    std::copy(prime_z.begin(), prime_z.end(), upload_z.begin());
    std::copy(prime_a.begin(), prime_a.end(), upload_a.begin());
    std::copy(prime_b.begin(), prime_b.end(), upload_b.begin());
    d_projected.copy_from(upload_projected, stream.get());
    d_z.copy_from(upload_z, stream.get());
    d_a.copy_from(upload_a, stream.get());
    d_b.copy_from(upload_b, stream.get());
    const q4::GdnStepInputs device_inputs{
        d_projected.data(), d_z.data(), d_a.data(), d_b.data()};
    const q4::GdnStepOutputs repeated_outputs{
        d_repeated_convolved.data(), d_repeated_output.data()};
    const q4::GdnStepOutputs block_outputs{
        d_block_convolved.data(), d_block_output.data()};
    require_status(q4::gdn_step_cuda(
                       &device_repeated, device_inputs, device_weights,
                       repeated_outputs, stream.get()),
                   q4::GdnStatus::kOk,
                   "prime repeated CUDA state");
    require_status(q4::gdn_step_cuda(
                       &device_block, device_inputs, device_weights,
                       block_outputs, stream.get()),
                   q4::GdnStatus::kOk,
                   "prime block CUDA state");

    d_projected.copy_from(projected, stream.get());
    d_z.copy_from(z, stream.get());
    d_a.copy_from(a, stream.get());
    d_b.copy_from(b, stream.get());
    for (std::size_t token = 0u; token < kTokenCount; ++token) {
        const q4::GdnStepInputs token_inputs{
            d_projected.data() + token * footprint.conv_channels,
            d_z.data() + token * footprint.value_elements,
            d_a.data() + token * config.value_heads,
            d_b.data() + token * config.value_heads};
        const q4::GdnStepOutputs token_outputs{
            d_repeated_convolved.data() + token * footprint.conv_channels,
            d_repeated_output.data() + token * footprint.value_elements};
        require_status(q4::gdn_step_cuda(
                           &device_repeated, token_inputs, device_weights,
                           token_outputs, stream.get()),
                       q4::GdnStatus::kOk,
                       "repeated CUDA sequence token");
    }
    require_status(q4::gdn_forward_sequence_cuda(
                       &device_block, kTokenCount, device_inputs,
                       device_weights, block_outputs, stream.get()),
                   q4::GdnStatus::kOk,
                   "CUDA contiguous sequence");

    std::vector<float> repeated_convolved(qkv_rows, 0.0F);
    std::vector<float> repeated_output(value_rows, 0.0F);
    std::vector<float> block_convolved(qkv_rows, 0.0F);
    std::vector<float> block_output(value_rows, 0.0F);
    std::vector<float> repeated_conv_state(footprint.conv_state_floats, 0.0F);
    std::vector<float> block_conv_state(footprint.conv_state_floats, 0.0F);
    std::vector<float> repeated_recurrent(
        footprint.recurrent_state_floats, 0.0F);
    std::vector<float> block_recurrent(
        footprint.recurrent_state_floats, 0.0F);
    d_repeated_convolved.copy_to(&repeated_convolved, stream.get());
    d_repeated_output.copy_to(&repeated_output, stream.get());
    d_block_convolved.copy_to(&block_convolved, stream.get());
    d_block_output.copy_to(&block_output, stream.get());
    require_cuda(cudaMemcpyAsync(
                     repeated_conv_state.data(), device_repeated.conv_state,
                     repeated_conv_state.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, stream.get()),
                 "copy repeated sequence convolution state");
    require_cuda(cudaMemcpyAsync(
                     block_conv_state.data(), device_block.conv_state,
                     block_conv_state.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, stream.get()),
                 "copy block sequence convolution state");
    require_cuda(cudaMemcpyAsync(
                     repeated_recurrent.data(), device_repeated.recurrent_state,
                     repeated_recurrent.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, stream.get()),
                 "copy repeated sequence recurrent state");
    require_cuda(cudaMemcpyAsync(
                     block_recurrent.data(), device_block.recurrent_state,
                     block_recurrent.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, stream.get()),
                 "copy block sequence recurrent state");
    stream.synchronize();

    float maximum_error = 0.0F;
    const Comparison conv = compare_vectors(
        repeated_convolved, block_convolved,
        4.0e-6F, 4.0e-6F, "CUDA block convolution");
    const Comparison output = compare_vectors(
        repeated_output, block_output,
        4.0e-5F, 5.0e-5F, "CUDA block output");
    const Comparison conv_state = compare_vectors(
        repeated_conv_state, block_conv_state,
        4.0e-6F, 4.0e-6F, "CUDA block convolution state");
    const Comparison recurrent_state = compare_vectors(
        repeated_recurrent, block_recurrent,
        5.0e-5F, 6.0e-5F, "CUDA block recurrent state");
    maximum_error = std::max(
        std::max(conv.max_absolute, output.max_absolute),
        std::max(conv_state.max_absolute, recurrent_state.max_absolute));
    require(device_block.tokens_seen == 1u + kTokenCount &&
                device_block.conv_cursor ==
                    (1u + kTokenCount) % config.conv_kernel,
            "CUDA sequence metadata advanced incorrectly");

    const std::uint64_t device_tokens_before_failure = device_block.tokens_seen;
    const std::uint32_t device_cursor_before_failure = device_block.conv_cursor;
    require_status(q4::gdn_forward_sequence_cuda(
                       &device_block, kTokenCount + 1u, device_inputs,
                       device_weights, block_outputs, stream.get()),
                   q4::GdnStatus::kUnsupportedConfig,
                   "oversized CUDA sequence rejection");
    q4::GdnStepInputs invalid_device_inputs = device_inputs;
    invalid_device_inputs.z = nullptr;
    require_status(q4::gdn_forward_sequence_cuda(
                       &device_block, kTokenCount, invalid_device_inputs,
                       device_weights, block_outputs, stream.get()),
                   q4::GdnStatus::kInvalidArgument,
                   "null CUDA sequence input rejection");
    require(device_block.tokens_seen == device_tokens_before_failure &&
                device_block.conv_cursor == device_cursor_before_failure,
            "invalid CUDA sequence mutated metadata");
    device_block.tokens_seen =
        std::numeric_limits<std::uint64_t>::max() - 2u;
    require_status(q4::gdn_forward_sequence_cuda(
                       &device_block, kTokenCount, device_inputs,
                       device_weights, block_outputs, stream.get()),
                   q4::GdnStatus::kSizeOverflow,
                   "CUDA sequence token overflow rejection");
    require(device_block.conv_cursor == device_cursor_before_failure,
            "overflowing CUDA sequence mutated cursor");

    require_status(q4::gdn_device_state_release(&device_repeated),
                   q4::GdnStatus::kOk,
                   "repeated device state release");
    require_status(q4::gdn_device_state_release(&device_block),
                   q4::GdnStatus::kOk,
                   "block device state release");
    return maximum_error;
}

}  // namespace

int main() {
    try {
        std::size_t real_state_bytes = 0;
        test_contract_fail_closed_and_real_state(&real_state_bytes);
        test_analytic_sigmoid_gated_rmsnorm();
        const float max_cuda_absolute = test_short_sequence_host_cuda();
        const float max_sequence_absolute = test_contiguous_sequence_block();
        std::cout << "gdn_test: OK sm=120 sequence_tokens=7 real_state_bytes="
                  << real_state_bytes << " max_cuda_abs=" << max_cuda_absolute
                  << " block_width=" << q4::kGdnMaxSequenceTokens
                  << " block_max_abs=" << max_sequence_absolute << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gdn_test: FAIL " << error.what() << '\n';
        return 1;
    }
}
