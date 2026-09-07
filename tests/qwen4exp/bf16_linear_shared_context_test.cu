#include "axiom/qwen4exp/bf16_linear.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

constexpr const char *kTensor =
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
constexpr std::size_t kLinears = 8u;
constexpr std::size_t kRounds = 64u;

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
    explicit device_buffer(std::size_t elements) : elements_(elements) {
        require(elements != 0u, "zero-sized device buffer");
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements * sizeof(T)),
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

private:
    T *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

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

struct lane final {
    explicit lane(const q4::bf16_linear_workspace_requirements &requirements,
                  const q4::bf16_linear_config &config)
        : input(static_cast<std::size_t>(config.input_features)),
          output(static_cast<std::size_t>(config.output_features)),
          input_bf16(requirements.input_bf16_bytes / sizeof(std::uint16_t)),
          blas_workspace(requirements.blas_workspace_bytes) {}

    stream_owner stream;
    device_buffer<float> input;
    device_buffer<float> output;
    device_buffer<std::uint16_t> input_bf16;
    device_buffer<unsigned char> blas_workspace;

    [[nodiscard]] q4::bf16_linear_scratch scratch() const noexcept {
        return {input_bf16.get(), input_bf16.bytes(),
                blas_workspace.get(), blas_workspace.bytes()};
    }
};

std::uint64_t digest(const std::vector<float> &values) {
    std::uint64_t value = 1469598103934665603ull;
    const auto *bytes = reinterpret_cast<const unsigned char *>(values.data());
    for (std::size_t index = 0u; index < values.size() * sizeof(float); ++index) {
        value ^= bytes[index];
        value *= 1099511628211ull;
    }
    return value;
}

struct test_result {
    std::uint64_t handles_before = 0u;
    std::uint64_t handles_after = 0u;
    std::uint64_t acquisitions = 0u;
    std::size_t live_handles = 0u;
    std::size_t vram_used_bytes = 0u;
    std::size_t steady_growth_bytes = 0u;
    std::uint64_t output_digest = 0u;
};

test_result shared_context_gate(const std::string &model_root) {
    int device = -1;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_require(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require(properties.major == 12 && properties.minor == 0,
            "test requires the strict sm_120 Blackwell path");

#ifndef AXIOM_BF16_BASELINE
    q4::bf16_linear_execution_context_metrics initial{};
    require(q4::bf16_linear_get_execution_context_metrics(device, &initial) ==
                    q4::bf16_linear_status::ok,
            "initial execution-context metrics failed");
    require(initial.live_handles == 0u && initial.live_contexts == 0u,
            "test process unexpectedly started with a live context");
#endif

    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    require(q4::checkpoint_catalog::open(model_root, &catalog, &error),
            error.empty() ? "checkpoint catalog open failed" : error);
    const q4::tensor_span *span = catalog->find(kTensor);
    require(span != nullptr && span->dtype == q4::tensor_dtype::bf16 &&
                    span->rank == 2u && span->shape[0] == 2560u &&
                    span->shape[1] == 640u && span->bytes == 3276800u,
            "qualification tensor descriptor mismatch");

    q4::bf16_linear_config config{};
    config.input_features = 640u;
    config.output_features = 2560u;
    config.max_batch = 1u;
    config.upload_chunk_bytes = 256u * 1024u;
    config.blas_workspace_bytes = 4u * 1024u * 1024u;
    q4::bf16_linear_workspace_requirements requirements{};
    require(q4::bf16_linear_get_workspace_requirements(config, &requirements) ==
                    q4::bf16_linear_status::ok,
            "workspace requirements failed");

    stream_owner initialization_stream;
    std::size_t free_before = 0u;
    std::size_t total_bytes = 0u;
    cuda_require(cudaMemGetInfo(&free_before, &total_bytes), "cudaMemGetInfo before");

    std::vector<std::unique_ptr<q4::resident_bf16_linear>> linears;
    linears.reserve(kLinears);
    for (std::size_t index = 0u; index < kLinears; ++index) {
        std::unique_ptr<q4::resident_bf16_linear> linear;
        const q4::bf16_linear_status status = q4::resident_bf16_linear::load(
                *catalog, kTensor, config, initialization_stream.get(),
                &linear, &error);
        require(status == q4::bf16_linear_status::ok && linear &&
                        linear->initialized(),
                error.empty() ? std::string("linear load failed: ") +
                                        q4::bf16_linear_status_string(status)
                              : error);
        linears.push_back(std::move(linear));
    }

#ifndef AXIOM_BF16_BASELINE
    q4::bf16_linear_execution_context_metrics loaded{};
    require(q4::bf16_linear_get_execution_context_metrics(device, &loaded) ==
                    q4::bf16_linear_status::ok,
            "loaded execution-context metrics failed");
    require(loaded.handles_created - initial.handles_created == 1u &&
                    loaded.context_acquisitions - initial.context_acquisitions ==
                            kLinears &&
                    loaded.live_contexts == 1u && loaded.live_handles == 1u,
            "N resident linears did not share exactly one cuBLAS handle");
#endif

    std::size_t free_after_load = 0u;
    cuda_require(cudaMemGetInfo(&free_after_load, &total_bytes),
                 "cudaMemGetInfo after load");
    require(free_after_load < free_before, "load did not consume device memory");

    std::vector<std::unique_ptr<lane>> lanes;
    lanes.reserve(kLinears);
    std::array<float, 640> host_input{};
    for (std::size_t index = 0u; index < host_input.size(); ++index) {
        host_input[index] = static_cast<float>((index * 37u) % 211u) / 211.0F - 0.5F;
    }
    for (std::size_t index = 0u; index < kLinears; ++index) {
        lanes.push_back(std::make_unique<lane>(requirements, config));
        cuda_require(cudaMemcpyAsync(lanes.back()->input.get(), host_input.data(),
                                     sizeof(host_input), cudaMemcpyHostToDevice,
                                     lanes.back()->stream.get()),
                     "cudaMemcpyAsync input");
    }

    std::atomic<bool> start{false};
    std::array<q4::bf16_linear_status, kLinears> statuses{};
    std::vector<std::thread> workers;
    workers.reserve(kLinears);
    for (std::size_t index = 0u; index < kLinears; ++index) {
        workers.emplace_back([&, index]() {
            if (cudaSetDevice(device) != cudaSuccess) {
                statuses[index] = q4::bf16_linear_status::cuda_error;
                return;
            }
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            statuses[index] = linears[index]->forward(
                    lanes[index]->input.get(), 1u, lanes[index]->scratch(),
                    lanes[index]->output.get(), lanes[index]->stream.get());
        });
    }
    start.store(true, std::memory_order_release);
    for (std::thread &worker : workers) worker.join();
    for (std::size_t index = 0u; index < kLinears; ++index) {
        require(statuses[index] == q4::bf16_linear_status::ok,
                "concurrent shared-handle forward failed");
        cuda_require(cudaStreamSynchronize(lanes[index]->stream.get()),
                     "cudaStreamSynchronize concurrent");
    }

    std::vector<float> baseline(2560u);
    cuda_require(cudaMemcpy(baseline.data(), lanes[0]->output.get(),
                            baseline.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "cudaMemcpy baseline");
    for (std::size_t index = 1u; index < kLinears; ++index) {
        std::vector<float> actual(2560u);
        cuda_require(cudaMemcpy(actual.data(), lanes[index]->output.get(),
                                actual.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy actual");
        require(std::memcmp(actual.data(), baseline.data(),
                            baseline.size() * sizeof(float)) == 0,
                "shared-handle output is not bit-identical across linears");
    }

    // Warm the complete path before proving repeated forwards do not grow the
    // CUDA allocation footprint.  All forward scratch remains caller-owned.
    for (std::size_t index = 0u; index < kLinears; ++index) {
        require(linears[index]->forward(
                        lanes[index]->input.get(), 1u, lanes[index]->scratch(),
                        lanes[index]->output.get(), lanes[index]->stream.get()) ==
                        q4::bf16_linear_status::ok,
                "warm forward failed");
    }
    cuda_require(cudaDeviceSynchronize(), "cudaDeviceSynchronize warm");
    std::size_t free_before_steady = 0u;
    cuda_require(cudaMemGetInfo(&free_before_steady, &total_bytes),
                 "cudaMemGetInfo steady before");
    for (std::size_t round = 0u; round < kRounds; ++round) {
        for (std::size_t index = 0u; index < kLinears; ++index) {
            require(linears[index]->forward(
                            lanes[index]->input.get(), 1u, lanes[index]->scratch(),
                            lanes[index]->output.get(), lanes[index]->stream.get()) ==
                            q4::bf16_linear_status::ok,
                    "steady forward failed");
        }
    }
    cuda_require(cudaDeviceSynchronize(), "cudaDeviceSynchronize steady");
    std::size_t free_after_steady = 0u;
    cuda_require(cudaMemGetInfo(&free_after_steady, &total_bytes),
                 "cudaMemGetInfo steady after");
    const std::size_t steady_growth = free_after_steady < free_before_steady
            ? free_before_steady - free_after_steady
            : 0u;
    require(steady_growth == 0u,
            "forward path grew device memory after warm-up");

    test_result result{};
    result.handles_before = kLinears;
#ifdef AXIOM_BF16_BASELINE
    result.handles_after = kLinears;
    result.acquisitions = kLinears;
    result.live_handles = kLinears;
#else
    result.handles_after = loaded.live_handles;
    result.acquisitions = loaded.context_acquisitions -
                          initial.context_acquisitions;
    result.live_handles = loaded.live_handles;
#endif
    result.vram_used_bytes = free_before - free_after_load;
    result.steady_growth_bytes = steady_growth;
    result.output_digest = digest(baseline);

    linears.clear();
#ifndef AXIOM_BF16_BASELINE
    q4::bf16_linear_execution_context_metrics released{};
    require(q4::bf16_linear_get_execution_context_metrics(device, &released) ==
                    q4::bf16_linear_status::ok &&
                    released.live_contexts == 0u && released.live_handles == 0u,
            "weak registry retained the execution context after last owner");
#endif
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        const test_result result = shared_context_gate(argv[1]);
        std::printf(
                "qwen4exp-bf16-shared-context-test: PASS linears=%zu "
                "handles_before=%llu handles_after=%llu acquisitions=%llu "
                "vram_used=%zu steady_growth=%zu concurrent=bit-identical "
                "output=%016llx\n",
                kLinears,
                static_cast<unsigned long long>(result.handles_before),
                static_cast<unsigned long long>(result.handles_after),
                static_cast<unsigned long long>(result.acquisitions),
                result.vram_used_bytes, result.steady_growth_bytes,
                static_cast<unsigned long long>(result.output_digest));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-bf16-shared-context-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
