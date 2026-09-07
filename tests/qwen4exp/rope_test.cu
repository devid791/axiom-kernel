#include "axiom/qwen4exp/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qr = axiom::qwen4exp::rope;

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t result, const char* where) {
    if (result != cudaSuccess) {
        fail(std::string(where) + ": " + cudaGetErrorString(result));
    }
}

void require_status(qr::status actual, qr::status expected, const char* where) {
    if (actual != expected) {
        fail(std::string(where) + ": expected " + qr::status_string(expected) +
             ", got " + qr::status_string(actual));
    }
}

template <typename T>
class device_buffer {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        require_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() { static_cast<void>(cudaFree(data_)); }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    void upload(const std::vector<T>& values, cudaStream_t stream) {
        require(values.size() <= count_, "upload overflow");
        require_cuda(cudaMemcpyAsync(data_, values.data(), values.size() * sizeof(T),
                                     cudaMemcpyHostToDevice, stream),
                     "upload");
    }
    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(count_);
        require_cuda(cudaMemcpyAsync(values.data(), data_, count_ * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream),
                     "download");
        require_cuda(cudaStreamSynchronize(stream), "download sync");
        return values;
    }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0u;
};

float compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "comparison size mismatch");
    float maximum = 0.0F;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    require(maximum <= 2.0e-5F, "RoPE CPU/CUDA drift exceeds tolerance");
    return maximum;
}

}  // namespace

int main() {
    try {
        require_status(qr::validate_checkpoint_config(qr::checkpoint_config),
                       qr::status::ok, "checkpoint config");
        int device = -1;
        cudaDeviceProp properties{};
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "gate requires sm_120");

        cudaStream_t stream = nullptr;
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                     "cudaStreamCreate");
        qr::device_plan plan{};
        require_status(qr::initialize_device_plan(&plan, qr::checkpoint_config, stream),
                       qr::status::ok, "plan init");

        constexpr std::size_t tokens = 4u;
        const std::vector<std::uint32_t> positions{
            0u, 1u, 1024u, 262143u,
            0u, 7u, 23u, 99u,
            0u, 9u, 31u, 101u,
        };
        std::vector<float> expected_cos(tokens * qr::kRotaryDim);
        std::vector<float> expected_sin(tokens * qr::kRotaryDim);
        require_status(qr::generate_host(qr::checkpoint_config, positions.data(), tokens,
                                         expected_cos.data(), expected_sin.data()),
                       qr::status::ok, "host mrope");

        device_buffer<std::uint32_t> device_positions(positions.size());
        device_buffer<float> device_cos(expected_cos.size());
        device_buffer<float> device_sin(expected_sin.size());
        device_buffer<qr::status> device_status(1u);
        device_positions.upload(positions, stream);
        require_status(qr::reset_device_status(device_status.get(), stream),
                       qr::status::ok, "reset status");
        require_status(qr::generate_mrope_cuda(&plan, device_positions.get(), tokens,
                                               device_cos.get(), device_sin.get(),
                                               device_status.get(), stream),
                       qr::status::ok, "CUDA mrope");
        qr::status collected = qr::status::cuda_failure;
        require_status(qr::collect_device_status(device_status.get(), &collected, stream),
                       qr::status::ok, "collect mrope");
        require_status(collected, qr::status::ok, "device mrope status");
        const float mrope_cos_abs = compare(device_cos.download(stream), expected_cos);
        const float mrope_sin_abs = compare(device_sin.download(stream), expected_sin);

        constexpr std::uint32_t text_start = 4096u;
        std::vector<std::uint32_t> text_positions(3u * tokens);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            for (std::size_t token = 0u; token < tokens; ++token) {
                text_positions[axis * tokens + token] =
                    text_start + static_cast<std::uint32_t>(token);
            }
        }
        require_status(qr::generate_host(qr::checkpoint_config, text_positions.data(),
                                         tokens, expected_cos.data(), expected_sin.data()),
                       qr::status::ok, "host text");
        require_status(qr::reset_device_status(device_status.get(), stream),
                       qr::status::ok, "reset text status");
        require_status(qr::generate_text_cuda(&plan, text_start, tokens,
                                              device_cos.get(), device_sin.get(),
                                              device_status.get(), stream),
                       qr::status::ok, "CUDA text");
        require_status(qr::collect_device_status(device_status.get(), &collected, stream),
                       qr::status::ok, "collect text");
        require_status(collected, qr::status::ok, "device text status");
        const float text_cos_abs = compare(device_cos.download(stream), expected_cos);
        const float text_sin_abs = compare(device_sin.download(stream), expected_sin);

        require_status(qr::generate_text_cuda(&plan, qr::kMaxContext - 1u, 2u,
                                              device_cos.get(), device_sin.get(),
                                              device_status.get(), stream),
                       qr::status::invalid_argument, "context bound");
        require_status(qr::release_device_plan(&plan), qr::status::ok, "plan release");
        require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        std::cout << "qwen4exp-rope-test: PASS sm=120 dim=64 theta=1e7 "
                  << "sections=11/11/10 context=262144 mrope_cos_abs="
                  << mrope_cos_abs << " mrope_sin_abs=" << mrope_sin_abs
                  << " text_cos_abs=" << text_cos_abs
                  << " text_sin_abs=" << text_sin_abs << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4exp-rope-test: FAIL: " << error.what() << '\n';
        return 1;
    }
}
