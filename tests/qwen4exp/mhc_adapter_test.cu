#include "axiom/qwen4exp/mhc_adapter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

void cuda_require(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class device_buffer final {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_), count * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() { if (pointer_ != nullptr) (void)cudaFree(pointer_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    T *get() noexcept { return pointer_; }
    std::size_t bytes() const noexcept { return count_ * sizeof(T); }
private:
    T *pointer_ = nullptr;
    std::size_t count_ = 0u;
};

std::vector<float> load_bf16(const q4::checkpoint_catalog &catalog,
                             const std::string &name,
                             const std::vector<std::uint64_t> &shape) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr && span->dtype == q4::tensor_dtype::bf16 &&
                    span->rank == shape.size(),
            "BF16 tensor contract missing: " + name);
    std::size_t count = 1u;
    for (std::size_t index = 0u; index < shape.size(); ++index) {
        require(span->shape[index] == shape[index], "BF16 tensor shape mismatch: " + name);
        count *= static_cast<std::size_t>(shape[index]);
    }
    require(span->bytes == count * sizeof(std::uint16_t),
            "BF16 tensor byte mismatch: " + name);
    std::vector<std::uint16_t> bits(count);
    std::string error;
    require(catalog.read_range(name, 0u, bits.data(), bits.size() * sizeof(bits[0]), &error),
            error.empty() ? "BF16 tensor read failed: " + name : error);
    std::vector<float> output(count);
    for (std::size_t index = 0u; index < count; ++index) {
        const std::uint32_t expanded = static_cast<std::uint32_t>(bits[index]) << 16u;
        std::memcpy(&output[index], &expanded, sizeof(float));
        require(std::isfinite(output[index]), "non-finite BF16 checkpoint value: " + name);
    }
    return output;
}

struct comparison {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
};

comparison compare(const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float absolute_limit, float relative_limit,
                   const char *label) {
    require(actual.size() == expected.size(), std::string(label) + " size mismatch");
    comparison result{};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), std::string(label) + " non-finite output");
        const float absolute = std::abs(actual[index] - expected[index]);
        const float relative = absolute / std::max(std::abs(expected[index]), 1.0e-4F);
        result.max_absolute = std::max(result.max_absolute, absolute);
        result.max_relative = std::max(result.max_relative, relative);
        if (absolute > absolute_limit && relative > relative_limit) {
            fail(std::string(label) + " parity mismatch at " + std::to_string(index) +
                 ": actual=" + std::to_string(actual[index]) +
                 " expected=" + std::to_string(expected[index]));
        }
    }
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
    const std::uint32_t expanded =
            static_cast<std::uint32_t>(float_to_bf16(value)) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &expanded, sizeof(result));
    return result;
}

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float inverse = std::exp(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0F + exponential);
}

void linear_reference(const std::vector<float> &weight,
                      std::size_t output_features,
                      std::size_t input_features,
                      const std::vector<float> &input,
                      std::vector<float> *output) {
    require(input.size() == input_features, "linear reference input size mismatch");
    require(weight.size() == output_features * input_features,
            "linear reference weight size mismatch");
    output->assign(output_features, 0.0F);
    for (std::size_t row = 0u; row < output_features; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0u; column < input_features; ++column) {
            sum = std::fma(round_bf16(input[column]),
                           weight[row * input_features + column], sum);
        }
        (*output)[row] = sum;
    }
}

void tensor_core_reference(const std::vector<float> &input,
                           const std::vector<float> &norm,
                           const std::vector<float> &down,
                           const std::vector<float> &up,
                           const std::vector<float> &inject,
                           std::vector<float> *mixed,
                           std::vector<float> *injection) {
    require(input.size() == 10240u && norm.size() == 10240u,
            "tensor-core oracle residual size mismatch");
    std::vector<float> normalized(10240u);
    for (std::size_t stream = 0u; stream < 4u; ++stream) {
        float square_sum = 0.0F;
        for (std::size_t hidden = 0u; hidden < 2560u; ++hidden) {
            const float value = input[stream * 2560u + hidden];
            square_sum += value * value;
        }
        const float inverse = 1.0F / std::sqrt(square_sum / 2560.0F + 1.0e-6F);
        for (std::size_t hidden = 0u; hidden < 2560u; ++hidden) {
            const std::size_t index = stream * 2560u + hidden;
            normalized[index] = input[index] * inverse * (1.0F + norm[index]);
        }
    }
    std::vector<float> low_rank;
    linear_reference(down, 320u, 10240u, normalized, &low_rank);
    for (float &value : low_rank) {
        const float scaled = value / 4.0F;
        value = scaled * sigmoid(scaled);
    }
    std::vector<float> mix_logits;
    std::vector<float> injection_logits;
    linear_reference(up, 10240u, 320u, low_rank, &mix_logits);
    linear_reference(inject, 4u, 10240u, normalized, &injection_logits);
    mixed->assign(2560u, 0.0F);
    for (std::size_t hidden = 0u; hidden < 2560u; ++hidden) {
        float sum = 0.0F;
        for (std::size_t stream = 0u; stream < 4u; ++stream) {
            const std::size_t index = stream * 2560u + hidden;
            sum += sigmoid(mix_logits[index]) * normalized[index];
        }
        (*mixed)[hidden] = sum / 4.0F;
    }
    injection->resize(4u);
    for (std::size_t stream = 0u; stream < 4u; ++stream) {
        (*injection)[stream] = 2.0F * sigmoid(injection_logits[stream] / 4.0F);
    }
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
        cuda_require(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "mHC adapter gate requires sm_120");

        q4::mhc_adapter_config invalid{};
        invalid.max_batch = 0u;
        require(q4::mhc_adapter_validate_config(invalid) ==
                        q4::mhc_adapter_status::invalid_argument,
                "zero batch config did not fail closed");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error), error);
        const std::string base =
                "model.language_model.layers.0.attn_hyper_connection.";
        std::vector<float> norm = load_bf16(
                *catalog, base + "hc_norm.weight", {10240u});
        std::vector<float> down = load_bf16(
                *catalog, base + "input_mix_weight_down.weight", {320u, 10240u});
        std::vector<float> up = load_bf16(
                *catalog, base + "input_mix_weight_up.weight", {10240u, 320u});
        std::vector<float> inject = load_bf16(
                *catalog, base + "block_inject_weight.weight", {4u, 10240u});

        q4::mhc_adapter_config config{};
        config.max_batch = 2u;
        config.upload_chunk_bytes = 256u * 1024u;
        std::unique_ptr<q4::resident_mhc_adapter> adapter;
        require(q4::resident_mhc_adapter::load(
                        *catalog, 0u, q4::mhc_adapter_site::attention, config,
                        nullptr, &adapter, &error) == q4::mhc_adapter_status::ok &&
                        adapter && adapter->initialized(),
                error.empty() ? "resident mHC adapter load failed" : error);
        require(adapter->layer() == 0u &&
                        adapter->site() == q4::mhc_adapter_site::attention &&
                        adapter->device() == device,
                "resident mHC adapter identity mismatch");

        q4::mhc_adapter_workspace_requirements requirements{};
        require(q4::mhc_adapter_get_workspace_requirements(config, &requirements) ==
                        q4::mhc_adapter_status::ok,
                "mHC adapter workspace derivation failed");
        device_buffer<float> d_input(config.max_batch * 10240u);
        device_buffer<float> d_normalized(requirements.normalized_f32_bytes / sizeof(float));
        device_buffer<float> d_low_rank(requirements.low_rank_f32_bytes / sizeof(float));
        device_buffer<float> d_mix(requirements.mix_logits_f32_bytes / sizeof(float));
        device_buffer<float> d_inject_logits(
                requirements.injection_logits_f32_bytes / sizeof(float));
        device_buffer<float> d_mixed(config.max_batch * 2560u);
        device_buffer<float> d_injection(config.max_batch * 4u);
        device_buffer<float> d_block(config.max_batch * 2560u);
        device_buffer<float> d_reinjected(config.max_batch * 10240u);
        device_buffer<std::uint16_t> d_linear_input(
                requirements.linear_input_bf16_bytes / sizeof(std::uint16_t));
        device_buffer<std::uint8_t> d_blas(requirements.blas_workspace_bytes);
        const q4::mhc_adapter_scratch scratch{
            d_normalized.get(), d_low_rank.get(), d_mix.get(), d_inject_logits.get(),
            d_linear_input.get(), d_linear_input.bytes(),
            d_blas.get(), d_blas.bytes()};

        constexpr std::size_t batch = 1u;
        std::vector<float> input(batch * 10240u);
        for (std::size_t index = 0u; index < input.size(); ++index) {
            input[index] = 0.125F * std::sin(static_cast<float>(index + 5u) * 0.0031F) +
                           0.03125F * std::cos(static_cast<float>(index + 17u) * 0.007F);
        }
        q4::MhcWeightsView weights{
            norm.data(), down.data(), up.data(), inject.data()};
        const q4::MhcConfig host_config = q4::mhc_qwen4_exp_config();
        const std::size_t workspace_floats = q4::mhc_workspace_floats(host_config, batch);
        std::vector<float> host_workspace(workspace_floats);
        std::vector<float> official_f32_mixed(batch * 2560u);
        std::vector<float> official_f32_injection(batch * 4u);
        require(q4::mhc_prepare_host(
                        host_config, input.data(), weights, batch,
                        official_f32_mixed.data(), official_f32_injection.data(),
                        host_workspace.data(), host_workspace.size()) == q4::MhcStatus::kOk,
                "host mHC oracle failed");
        std::vector<float> expected_mixed;
        std::vector<float> expected_injection;
        tensor_core_reference(input, norm, down, up, inject,
                              &expected_mixed, &expected_injection);

        cuda_require(cudaMemcpy(d_input.get(), input.data(), input.size() * sizeof(float),
                                cudaMemcpyHostToDevice), "copy input");
        require(adapter->prepare(d_input.get(), batch, scratch, d_mixed.get(),
                                 d_injection.get(), nullptr) == q4::mhc_adapter_status::ok,
                "CUDA mHC adapter prepare failed");
        std::vector<float> actual_mixed(expected_mixed.size());
        std::vector<float> actual_injection(expected_injection.size());
        cuda_require(cudaMemcpy(actual_mixed.data(), d_mixed.get(),
                                actual_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy mixed");
        cuda_require(cudaMemcpy(actual_injection.data(), d_injection.get(),
                                actual_injection.size() * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy injection");
        const comparison mixed_error = compare(
                actual_mixed, expected_mixed, 5.0e-5F, 5.0e-4F, "mixed input");
        const comparison injection_error = compare(
                actual_injection, expected_injection, 5.0e-5F, 5.0e-4F, "injection");
        const comparison f32_drift = compare(
                actual_mixed, official_f32_mixed, 5.0e-3F, 5.0F, "F32 semantic drift");

        std::vector<float> block(batch * 2560u);
        for (std::size_t index = 0u; index < block.size(); ++index) {
            block[index] = 0.05F * std::sin(static_cast<float>(index + 1u) * 0.011F);
        }
        std::vector<float> expected_reinjected(batch * 10240u);
        require(q4::mhc_reinject_host(
                        host_config, input.data(), block.data(), actual_injection.data(),
                        batch, expected_reinjected.data()) == q4::MhcStatus::kOk,
                "host mHC reinject oracle failed");
        cuda_require(cudaMemcpy(d_block.get(), block.data(), block.size() * sizeof(float),
                                cudaMemcpyHostToDevice), "copy block");
        require(adapter->reinject(d_input.get(), d_block.get(), d_injection.get(), batch,
                                  d_reinjected.get(), nullptr) ==
                        q4::mhc_adapter_status::ok,
                "CUDA mHC reinject failed");
        std::vector<float> actual_reinjected(expected_reinjected.size());
        cuda_require(cudaMemcpy(actual_reinjected.data(), d_reinjected.get(),
                                actual_reinjected.size() * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy reinjected");
        const comparison reinject_error = compare(
                actual_reinjected, expected_reinjected, 1.0e-5F, 1.0e-5F, "reinject");

        std::vector<float> first_run = actual_mixed;
        require(adapter->prepare(d_input.get(), batch, scratch, d_mixed.get(),
                                 d_injection.get(), nullptr) == q4::mhc_adapter_status::ok,
                "repeat mHC prepare failed");
        cuda_require(cudaMemcpy(actual_mixed.data(), d_mixed.get(),
                                actual_mixed.size() * sizeof(float), cudaMemcpyDeviceToHost),
                     "copy repeat mixed");
        require(std::memcmp(first_run.data(), actual_mixed.data(),
                            first_run.size() * sizeof(float)) == 0,
                "mHC adapter output is not repeatable");

        require(adapter->prepare(input.data(), batch, scratch, d_mixed.get(),
                                 d_injection.get(), nullptr) ==
                        q4::mhc_adapter_status::invalid_device_pointer,
                "host pointer did not fail closed");
        require(adapter->prepare(d_input.get(), batch, scratch, d_mixed.get(),
                                 d_injection.get(), nullptr) == q4::mhc_adapter_status::ok,
                "rejected pointer poisoned the next valid launch");
        cuda_require(cudaDeviceSynchronize(), "final synchronize");

        std::printf(
                "qwen4exp-mhc-adapter-test: PASS sm=120 layer=0 site=attention "
                "resident=bf16-tensor-core batch=1 mixed_abs=%.9g mixed_rel=%.9g "
                "inject_abs=%.9g f32_drift_abs=%.9g reinject_abs=%.9g deterministic=2/2\n",
                mixed_error.max_absolute, mixed_error.max_relative,
                injection_error.max_absolute, f32_drift.max_absolute,
                reinject_error.max_absolute);
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-mhc-adapter-test: FAIL: %s\n", exception.what());
        return 1;
    }
}
