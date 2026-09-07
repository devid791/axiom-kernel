#include "axiom/qwen4exp/mhc.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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

void require_status(q4::MhcStatus actual, q4::MhcStatus expected, const std::string& label) {
    if (actual != expected) {
        fail(label + ": expected " + q4::mhc_status_string(expected) + ", got " +
             q4::mhc_status_string(actual));
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
    const std::int32_t signed_value = static_cast<std::int32_t>((value >> 32U) & 0xFFFFU) - 32768;
    return scale * static_cast<float>(signed_value) / 32768.0F;
}

void fill_deterministic(std::vector<float>* values, std::uint64_t seed, float scale) {
    require(values != nullptr, "fill_deterministic null vector");
    for (std::size_t index = 0; index < values->size(); ++index) {
        (*values)[index] = deterministic_value(seed, index, scale);
    }
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
    Comparison comparison;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(std::isfinite(expected[index]), label + ": non-finite host oracle");
        require(std::isfinite(actual[index]), label + ": non-finite CUDA result");
        const float absolute = std::abs(actual[index] - expected[index]);
        const float denominator = std::max(std::abs(expected[index]), 1.0e-7F);
        const float relative = absolute / denominator;
        if (absolute > comparison.max_absolute) {
            comparison.max_absolute = absolute;
            comparison.max_index = index;
        }
        comparison.max_relative = std::max(comparison.max_relative, relative);
        const float allowed = absolute_tolerance + relative_tolerance * std::abs(expected[index]);
        if (absolute > allowed) {
            fail(label + ": mismatch at index " + std::to_string(index) +
                 ", expected=" + std::to_string(expected[index]) +
                 ", actual=" + std::to_string(actual[index]) +
                 ", abs=" + std::to_string(absolute) +
                 ", allowed=" + std::to_string(allowed));
        }
    }
    return comparison;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        require(count > 0, "DeviceBuffer cannot be empty");
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc");
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
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

    void copy_from(const std::vector<T>& host, cudaStream_t stream) {
        require(host.size() == count_, "copy_from size mismatch");
        require_cuda(
            cudaMemcpyAsync(data_, host.data(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync H2D");
    }

    void copy_to(std::vector<T>* host, cudaStream_t stream) const {
        require(host != nullptr && host->size() == count_, "copy_to size mismatch");
        require_cuda(
            cudaMemcpyAsync(host->data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync D2H");
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

void test_contract_and_invalid_inputs() {
    const q4::MhcConfig exact = q4::mhc_qwen4_exp_config();
    require_status(q4::mhc_validate_config(exact), q4::MhcStatus::kOk, "exact config");
    require(q4::mhc_is_qwen4_exp_contract(exact), "exact Qwen4-Exp contract was not recognized");
    require(q4::mhc_residual_size(exact) == q4::kMhcQwen4ExpResidual, "exact residual mismatch");

    q4::MhcConfig invalid = exact;
    invalid.hidden = 0;
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kInvalidArgument, "zero hidden");
    invalid = exact;
    invalid.streams = 1;
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kUnsupportedConfig, "one stream");
    invalid = exact;
    invalid.streams = q4::kMhcMaxStreams + 1;
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kUnsupportedConfig, "too many streams");
    invalid = exact;
    invalid.hidden = q4::kMhcMaxHidden + 1;
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kUnsupportedConfig, "hidden limit");
    invalid = exact;
    invalid.rank = q4::kMhcMaxRank + 1;
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kUnsupportedConfig, "rank limit");
    invalid = exact;
    invalid.rms_epsilon = std::numeric_limits<float>::quiet_NaN();
    require_status(q4::mhc_validate_config(invalid), q4::MhcStatus::kInvalidArgument, "NaN epsilon");
    require(q4::mhc_workspace_floats(exact, 0) == 0, "zero-token workspace must fail closed");
    require(
        q4::mhc_workspace_floats(exact, q4::kMhcMaxTokensPerLaunch + 1) == 0,
        "oversized launch workspace must fail closed");

    const q4::MhcConfig small{2, 2, 1, 1.0e-6F};
    const std::size_t residual = q4::mhc_residual_size(small);
    const std::size_t workspace_size = q4::mhc_workspace_floats(small, 1);
    std::vector<float> hyper_input(residual, 0.0F);
    std::vector<float> norm_weight(residual, 0.0F);
    std::vector<float> down(small.rank * residual, 0.0F);
    std::vector<float> up(residual * small.rank, 0.0F);
    std::vector<float> inject(small.streams * residual, 0.0F);
    std::vector<float> mixed(small.hidden, 0.0F);
    std::vector<float> injection(small.streams, 0.0F);
    std::vector<float> workspace(workspace_size, 0.0F);
    const q4::MhcWeightsView weights{
        norm_weight.data(), down.data(), up.data(), inject.data()};

    require_status(
        q4::mhc_prepare_host(
            small,
            nullptr,
            weights,
            1,
            mixed.data(),
            injection.data(),
            workspace.data(),
            workspace.size()),
        q4::MhcStatus::kInvalidArgument,
        "null host input");
    require_status(
        q4::mhc_prepare_host(
            small,
            hyper_input.data(),
            weights,
            1,
            mixed.data(),
            injection.data(),
            workspace.data(),
            workspace.size() - 1),
        q4::MhcStatus::kInsufficientWorkspace,
        "short host workspace");
    q4::MhcWeightsView missing_inject = weights;
    missing_inject.block_inject_weight = nullptr;
    require_status(
        q4::mhc_prepare_host(
            small,
            hyper_input.data(),
            missing_inject,
            1,
            mixed.data(),
            injection.data(),
            workspace.data(),
            workspace.size()),
        q4::MhcStatus::kInvalidArgument,
        "missing block inject");
    require_status(
        q4::mhc_finalize_host(
            small,
            hyper_input.data(),
            missing_inject,
            1,
            mixed.data(),
            workspace.data(),
            workspace.size()),
        q4::MhcStatus::kOk,
        "finalize must not require block inject");
}

void test_analytic_host_oracle() {
    const q4::MhcConfig config{2, 2, 1, 1.0e-6F};
    const std::size_t residual = q4::mhc_residual_size(config);
    const std::size_t workspace_size = q4::mhc_workspace_floats(config, 1);
    const std::vector<float> hyper_input{3.0F, 4.0F, 0.0F, 5.0F};
    const std::vector<float> norm_weight{0.0F, 0.25F, -0.5F, 0.0F};
    const std::vector<float> down(config.rank * residual, 0.0F);
    const std::vector<float> up(residual * config.rank, 0.0F);
    const std::vector<float> inject(config.streams * residual, 0.0F);
    const q4::MhcWeightsView weights{
        norm_weight.data(), down.data(), up.data(), inject.data()};

    std::vector<float> mixed(config.hidden, 0.0F);
    std::vector<float> injection(config.streams, 0.0F);
    std::vector<float> workspace(workspace_size, 0.0F);
    require_status(
        q4::mhc_prepare_host(
            config,
            hyper_input.data(),
            weights,
            1,
            mixed.data(),
            injection.data(),
            workspace.data(),
            workspace.size()),
        q4::MhcStatus::kOk,
        "analytic prepare");

    const float stream0_inverse = 1.0F / std::sqrt((9.0F + 16.0F) / 2.0F + config.rms_epsilon);
    const float stream1_inverse = 1.0F / std::sqrt(25.0F / 2.0F + config.rms_epsilon);
    const float normalized_00 = 3.0F * stream0_inverse;
    const float normalized_01 = 4.0F * stream0_inverse * 1.25F;
    const float normalized_10 = 0.0F * stream1_inverse * 0.5F;
    const float normalized_11 = 5.0F * stream1_inverse;
    const std::vector<float> expected_mixed{
        (normalized_00 + normalized_10) * 0.25F,
        (normalized_01 + normalized_11) * 0.25F};
    (void)compare_vectors(expected_mixed, mixed, 1.0e-6F, 1.0e-6F, "analytic mixed");
    require(injection[0] == 1.0F && injection[1] == 1.0F, "zero inject weights must produce unit injection");

    const std::vector<float> block_output{0.25F, -0.5F};
    std::vector<float> reinjected(residual, 0.0F);
    require_status(
        q4::mhc_reinject_host(
            config,
            hyper_input.data(),
            block_output.data(),
            injection.data(),
            1,
            reinjected.data()),
        q4::MhcStatus::kOk,
        "analytic reinject");
    const std::vector<float> expected_reinjected{3.25F, 3.5F, 0.25F, 4.5F};
    (void)compare_vectors(expected_reinjected, reinjected, 0.0F, 0.0F, "analytic reinjected");

    std::vector<float> mixed_repeat(config.hidden, 0.0F);
    std::vector<float> injection_repeat(config.streams, 0.0F);
    std::vector<float> workspace_repeat(workspace_size, 0.0F);
    require_status(
        q4::mhc_prepare_host(
            config,
            hyper_input.data(),
            weights,
            1,
            mixed_repeat.data(),
            injection_repeat.data(),
            workspace_repeat.data(),
            workspace_repeat.size()),
        q4::MhcStatus::kOk,
        "deterministic repeat");
    require(
        std::memcmp(mixed.data(), mixed_repeat.data(), mixed.size() * sizeof(float)) == 0 &&
            std::memcmp(injection.data(), injection_repeat.data(), injection.size() * sizeof(float)) == 0,
        "host reference is not bit-deterministic");
}

struct OracleSummary {
    Comparison mixed;
    Comparison injection;
    Comparison reinjected;
    Comparison finalized;
};

OracleSummary run_cuda_oracle(
    const q4::MhcConfig& config,
    std::size_t token_count,
    float absolute_tolerance,
    float relative_tolerance,
    const std::string& label) {
    const std::size_t residual = q4::mhc_residual_size(config);
    const std::size_t workspace_size = q4::mhc_workspace_floats(config, token_count);
    require(residual != 0 && workspace_size != 0, label + ": invalid test shape");

    std::vector<float> hyper_input(token_count * residual);
    std::vector<float> norm_weight(residual);
    std::vector<float> down(config.rank * residual);
    std::vector<float> up(residual * config.rank);
    std::vector<float> inject(config.streams * residual);
    std::vector<float> block_output(token_count * config.hidden);
    fill_deterministic(&hyper_input, 0x1001ULL, 0.75F);
    fill_deterministic(&norm_weight, 0x2002ULL, 0.125F);
    fill_deterministic(&down, 0x3003ULL, 0.0020F);
    fill_deterministic(&up, 0x4004ULL, 0.0200F);
    fill_deterministic(&inject, 0x5005ULL, 0.0010F);
    fill_deterministic(&block_output, 0x6006ULL, 0.30F);

    const q4::MhcWeightsView host_weights{
        norm_weight.data(), down.data(), up.data(), inject.data()};
    std::vector<float> host_mixed(token_count * config.hidden, 0.0F);
    std::vector<float> host_injection(token_count * config.streams, 0.0F);
    std::vector<float> host_reinjected(token_count * residual, 0.0F);
    std::vector<float> host_final(token_count * config.hidden, 0.0F);
    std::vector<float> host_workspace(workspace_size, 0.0F);
    require_status(
        q4::mhc_prepare_host(
            config,
            hyper_input.data(),
            host_weights,
            token_count,
            host_mixed.data(),
            host_injection.data(),
            host_workspace.data(),
            host_workspace.size()),
        q4::MhcStatus::kOk,
        label + " host prepare");
    require_status(
        q4::mhc_reinject_host(
            config,
            hyper_input.data(),
            block_output.data(),
            host_injection.data(),
            token_count,
            host_reinjected.data()),
        q4::MhcStatus::kOk,
        label + " host reinject");
    require_status(
        q4::mhc_finalize_host(
            config,
            host_reinjected.data(),
            host_weights,
            token_count,
            host_final.data(),
            host_workspace.data(),
            host_workspace.size()),
        q4::MhcStatus::kOk,
        label + " host finalize");

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), label + " stream create");

    DeviceBuffer<float> device_hyper(hyper_input.size());
    DeviceBuffer<float> device_norm(norm_weight.size());
    DeviceBuffer<float> device_down(down.size());
    DeviceBuffer<float> device_up(up.size());
    DeviceBuffer<float> device_inject(inject.size());
    DeviceBuffer<float> device_block(block_output.size());
    DeviceBuffer<float> device_mixed(host_mixed.size());
    DeviceBuffer<float> device_injection(host_injection.size());
    DeviceBuffer<float> device_reinjected(host_reinjected.size());
    DeviceBuffer<float> device_final(host_final.size());
    DeviceBuffer<float> device_workspace(workspace_size);

    device_hyper.copy_from(hyper_input, stream);
    device_norm.copy_from(norm_weight, stream);
    device_down.copy_from(down, stream);
    device_up.copy_from(up, stream);
    device_inject.copy_from(inject, stream);
    device_block.copy_from(block_output, stream);
    const q4::MhcWeightsView device_weights{
        device_norm.data(), device_down.data(), device_up.data(), device_inject.data()};

    require_status(
        q4::mhc_prepare_cuda(
            config,
            device_hyper.data(),
            device_weights,
            token_count,
            device_mixed.data(),
            device_injection.data(),
            device_workspace.data(),
            device_workspace.size(),
            stream),
        q4::MhcStatus::kOk,
        label + " CUDA prepare");
    require_status(
        q4::mhc_reinject_cuda(
            config,
            device_hyper.data(),
            device_block.data(),
            device_injection.data(),
            token_count,
            device_reinjected.data(),
            stream),
        q4::MhcStatus::kOk,
        label + " CUDA reinject");
    require_status(
        q4::mhc_finalize_cuda(
            config,
            device_reinjected.data(),
            device_weights,
            token_count,
            device_final.data(),
            device_workspace.data(),
            device_workspace.size(),
            stream),
        q4::MhcStatus::kOk,
        label + " CUDA finalize");

    std::vector<float> cuda_mixed(host_mixed.size(), 0.0F);
    std::vector<float> cuda_injection(host_injection.size(), 0.0F);
    std::vector<float> cuda_reinjected(host_reinjected.size(), 0.0F);
    std::vector<float> cuda_final(host_final.size(), 0.0F);
    device_mixed.copy_to(&cuda_mixed, stream);
    device_injection.copy_to(&cuda_injection, stream);
    device_reinjected.copy_to(&cuda_reinjected, stream);
    device_final.copy_to(&cuda_final, stream);
    require_cuda(cudaStreamSynchronize(stream), label + " stream synchronize");
    require_cuda(cudaStreamDestroy(stream), label + " stream destroy");

    return OracleSummary{
        compare_vectors(host_mixed, cuda_mixed, absolute_tolerance, relative_tolerance, label + " mixed"),
        compare_vectors(
            host_injection,
            cuda_injection,
            absolute_tolerance,
            relative_tolerance,
            label + " injection"),
        compare_vectors(
            host_reinjected,
            cuda_reinjected,
            absolute_tolerance,
            relative_tolerance,
            label + " reinjected"),
        compare_vectors(host_final, cuda_final, absolute_tolerance, relative_tolerance, label + " finalized")};
}

void print_summary(const std::string& label, const OracleSummary& summary) {
    std::cout << label
              << " max_abs={mixed:" << summary.mixed.max_absolute
              << ",injection:" << summary.injection.max_absolute
              << ",reinject:" << summary.reinjected.max_absolute
              << ",finalize:" << summary.finalized.max_absolute << "}\n";
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "CUDA gate requires one device");
        require_cuda(cudaSetDevice(0), "cudaSetDevice");

        test_contract_and_invalid_inputs();
        test_analytic_host_oracle();

        constexpr float kSmallAbsoluteTolerance = 3.0e-5F;
        constexpr float kSmallRelativeTolerance = 3.0e-5F;
        const OracleSummary small_summary = run_cuda_oracle(
            q4::MhcConfig{17, 4, 7, 1.0e-6F},
            3,
            kSmallAbsoluteTolerance,
            kSmallRelativeTolerance,
            "parametric");
        print_summary("parametric", small_summary);

        constexpr float kExactAbsoluteTolerance = 3.0e-4F;
        constexpr float kExactRelativeTolerance = 3.0e-4F;
        const OracleSummary exact_summary = run_cuda_oracle(
            q4::mhc_qwen4_exp_config(),
            1,
            kExactAbsoluteTolerance,
            kExactRelativeTolerance,
            "qwen4_exp_exact");
        print_summary("qwen4_exp_exact", exact_summary);

        require_cuda(cudaDeviceSynchronize(), "final cudaDeviceSynchronize");
        std::cout << "PASS qwen4_exp mHC host/CUDA oracle, analytic semantics, bounds and invalid inputs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL qwen4_exp mHC: " << error.what() << '\n';
        return 1;
    }
}
