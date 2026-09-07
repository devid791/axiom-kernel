#include "axiom/qwen4exp/vision_adapter.hpp"

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

std::uint64_t fnv1a(const float *values, std::size_t count) {
    std::uint64_t digest = 1469598103934665603ull;
    for (std::size_t index = 0u; index < count; ++index) {
        std::uint32_t bits = 0u;
        std::memcpy(&bits, values + index, sizeof(bits));
        for (unsigned byte = 0u; byte < 4u; ++byte) {
            digest ^= (bits >> (byte * 8u)) & 0xffu;
            digest *= 1099511628211ull;
        }
    }
    return digest;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2, "usage: qwen4exp-vision-adapter-test MODEL_ROOT");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error),
                error.empty() ? "checkpoint catalog open failed" : error);

        q4::vision_contract_report contract{};
        const q4::vision_status admitted =
                q4::vision_admit_checkpoint(*catalog, &contract, &error);
        require(admitted == q4::vision_status::ok,
                std::string("vision admission failed: ") +
                        q4::vision_status_string(admitted) + ": " + error);
        require(contract.tensor_count == 333u, "unexpected visual tensor count");
        require(contract.tensor_bytes == 897862112ull,
                "unexpected visual tensor bytes");
        require(contract.depth == 27u && contract.hidden_size == 1152u &&
                        contract.intermediate_size == 4304u &&
                        contract.num_heads == 16u && contract.output_size == 2560u,
                "unexpected Flash-Next visual geometry");

        q4::vision_provider_config config{};
        config.device = 0;
        config.max_patch_tokens = 4u;
        std::unique_ptr<q4::resident_vision_provider> provider;
        error.clear();
        const q4::vision_status loaded = q4::resident_vision_provider::load(
                *catalog, config, nullptr, &provider, &error);
        require(loaded == q4::vision_status::ok,
                std::string("vision load failed: ") +
                        q4::vision_status_string(loaded) + ": " + error);
        require(provider != nullptr && provider->initialized(),
                "vision provider was not initialized");
        const q4::vision_provider_info info = provider->info();
        require(info.contract.output_size == 2560u &&
                        info.resident_weight_bytes == 897862112ull &&
                        info.resident_total_bytes >= info.resident_weight_bytes,
                "resident vision provider accounting mismatch");

        constexpr std::size_t kPatches = 4u;
        std::vector<float> patches(kPatches * q4::kVisionPatchFeatures);
        for (std::size_t index = 0u; index < patches.size(); ++index) {
            const float phase = static_cast<float>((index * 17u) % 257u) / 128.0F;
            patches[index] = phase - 1.0F;
        }
        const q4::vision_grid grid{1u, 2u, 2u};
        std::vector<float> first(q4::kVisionOutputSize, 0.0F);
        std::vector<float> second(q4::kVisionOutputSize, 0.0F);
        std::size_t first_tokens = 0u;
        std::size_t second_tokens = 0u;
        const q4::vision_status first_status = provider->forward_patches(
                patches.data(), patches.size(), &grid, 1u, first.data(),
                first.size(), &first_tokens);
        require(first_status == q4::vision_status::ok,
                std::string("first real vision forward failed: ") +
                        q4::vision_status_string(first_status) + ": " +
                        cudaGetErrorString(cudaGetLastError()));
        const q4::vision_status second_status = provider->forward_patches(
                patches.data(), patches.size(), &grid, 1u, second.data(),
                second.size(), &second_tokens);
        require(second_status == q4::vision_status::ok,
                std::string("second real vision forward failed: ") +
                        q4::vision_status_string(second_status) + ": " +
                        cudaGetErrorString(cudaGetLastError()));
        require(first_tokens == 1u && second_tokens == 1u,
                "2x2 merger did not produce exactly one token");

        float *device_patches = nullptr;
        float *device_output = nullptr;
        require(cudaMalloc(reinterpret_cast<void **>(&device_patches),
                           patches.size() * sizeof(float)) == cudaSuccess &&
                        cudaMalloc(reinterpret_cast<void **>(&device_output),
                                   first.size() * sizeof(float)) == cudaSuccess,
                "device-forward buffer allocation failed");
        require(cudaMemcpy(device_patches, patches.data(),
                           patches.size() * sizeof(float),
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "device-forward patch upload failed");
        std::size_t device_tokens = 0u;
        const q4::vision_status device_status =
                provider->forward_patches_device(
                        device_patches, patches.size(), &grid, 1u,
                        device_output, first.size(), &device_tokens, nullptr);
        require(device_status == q4::vision_status::ok &&
                        device_tokens == 1u,
                std::string("real device vision forward failed: ") +
                        q4::vision_status_string(device_status));
        std::vector<float> device_result(first.size(), 0.0F);
        require(cudaMemcpy(device_result.data(), device_output,
                           device_result.size() * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "device-forward result download failed");
        require(cudaFree(device_output) == cudaSuccess &&
                        cudaFree(device_patches) == cudaSuccess,
                "device-forward buffer release failed");

        float minimum = first.front();
        float maximum = first.front();
        double squared_norm = 0.0;
        float max_difference = 0.0F;
        for (std::size_t index = 0u; index < first.size(); ++index) {
            require(std::isfinite(first[index]) && std::isfinite(second[index]),
                    "vision forward emitted non-finite output");
            minimum = std::min(minimum, first[index]);
            maximum = std::max(maximum, first[index]);
            squared_norm += static_cast<double>(first[index]) * first[index];
            max_difference = std::max(
                    max_difference, std::fabs(first[index] - second[index]));
            max_difference = std::max(
                    max_difference,
                    std::fabs(first[index] - device_result[index]));
        }
        require(squared_norm > 0.0 && maximum > minimum,
                "vision forward output is trivial");
        require(max_difference == 0.0F,
                "vision forward is not bitwise deterministic on one device");

        std::printf(
                "qwen4exp-vision-adapter-test: PASS tensors=%zu bytes=%llu "
                "depth=%zu hidden=%zu output=%zu patches=%zu merged=%zu "
                "device_bytes=%llu range=[%.9g,%.9g] l2=%.9g "
                "device_handoff=direct deterministic_abs=%.9g "
                "digest=%016llx\n",
                contract.tensor_count,
                static_cast<unsigned long long>(contract.tensor_bytes),
                contract.depth, contract.hidden_size, contract.output_size,
                kPatches, first_tokens,
                static_cast<unsigned long long>(info.resident_total_bytes),
                minimum, maximum, std::sqrt(squared_norm), max_difference,
                static_cast<unsigned long long>(fnv1a(first.data(), first.size())));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-vision-adapter-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
