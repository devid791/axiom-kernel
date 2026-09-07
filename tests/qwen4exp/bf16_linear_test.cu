#include "axiom/qwen4exp/bf16_linear.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

constexpr const char *kRealTensor =
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight";
constexpr const char *kU8Tensor =
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight";

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

float bf16_to_float(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
class device_buffer final {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t elements) { allocate(elements); }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0u)) {}
    device_buffer &operator=(device_buffer &&other) noexcept {
        if (this == &other) return *this;
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
        pointer_ = std::exchange(other.pointer_, nullptr);
        elements_ = std::exchange(other.elements_, 0u);
        return *this;
    }
    void allocate(std::size_t elements) {
        require(pointer_ == nullptr && elements != 0u,
                "invalid device_buffer allocation");
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements * sizeof(T)),
                     "cudaMalloc");
        elements_ = elements;
    }
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

void synthetic_reference_gate() {
    q4::bf16_linear_config config{};
    config.input_features = 4u;
    config.output_features = 3u;
    config.max_batch = 2u;
    config.upload_chunk_bytes = 4096u;
    config.blas_workspace_bytes = 4096u;
    require(q4::bf16_linear_validate_config(config) == q4::bf16_linear_status::ok,
            "synthetic config was rejected");

    constexpr std::array<float, 12> weight_f32{
        1.0F, 2.0F, -1.0F, 0.5F,
        0.0F, -1.0F, 2.0F, 1.0F,
        0.25F, 0.5F, 1.0F, -2.0F,
    };
    std::array<std::uint16_t, weight_f32.size()> weight{};
    std::transform(weight_f32.begin(), weight_f32.end(), weight.begin(),
                   float_to_bf16);
    constexpr std::array<float, 8> input{
        1.0F, 2.0F, 3.0F, 4.0F,
        -1.0F, 0.5F, 2.0F, -2.0F,
    };
    constexpr std::array<float, 6> expected{4.0F, 8.0F, -3.75F,
                                            -3.0F, 1.5F, 6.0F};
    std::array<float, expected.size()> output{};
    require(q4::bf16_linear_reference_f32(config, weight.data(), input.data(),
                                           2u, output.data()) ==
                    q4::bf16_linear_status::ok,
            "synthetic host reference failed");
    require(output == expected, "synthetic host reference differs from exact oracle");

    q4::bf16_linear_config invalid = config;
    invalid.max_batch = 0u;
    require(q4::bf16_linear_validate_config(invalid) ==
                    q4::bf16_linear_status::invalid_argument,
            "zero batch config did not fail closed");
    invalid = config;
    invalid.upload_chunk_bytes = 4097u;
    require(q4::bf16_linear_validate_config(invalid) ==
                    q4::bf16_linear_status::unsupported_config,
            "odd upload chunk did not fail closed");
}

void independent_reference(const q4::bf16_linear_config &config,
                           const std::vector<std::uint16_t> &weight,
                           const std::vector<float> &input,
                           std::size_t batch,
                           std::vector<float> *output) {
    const std::size_t inputs = static_cast<std::size_t>(config.input_features);
    const std::size_t outputs = static_cast<std::size_t>(config.output_features);
    output->assign(batch * outputs, 0.0F);
    for (std::size_t token = 0u; token < batch; ++token) {
        for (std::size_t row = 0u; row < outputs; ++row) {
            float sum = 0.0F;
            for (std::size_t column = 0u; column < inputs; ++column) {
                const float rounded_input = bf16_to_float(float_to_bf16(
                        input[token * inputs + column]));
                const float decoded_weight =
                        bf16_to_float(weight[row * inputs + column]);
                sum = std::fma(rounded_input, decoded_weight, sum);
            }
            (*output)[token * outputs + row] = sum;
        }
    }
}

struct comparison_result {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
};

comparison_result compare_outputs(const std::vector<float> &actual,
                                  const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "output sizes differ");
    comparison_result result{};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), "CUDA output is non-finite");
        const float absolute = std::fabs(actual[index] - expected[index]);
        const float relative = absolute / std::max(1.0e-5F, std::fabs(expected[index]));
        result.max_absolute = std::max(result.max_absolute, absolute);
        result.max_relative = std::max(result.max_relative, relative);
        const float tolerance = 2.0e-3F + 2.0e-3F * std::fabs(expected[index]);
        if (absolute > tolerance) {
            fail("CUDA output exceeds BF16/F32 oracle tolerance at index " +
                 std::to_string(index));
        }
    }
    return result;
}

std::uint64_t fnv1a(const void *data, std::size_t bytes) {
    const auto *octets = static_cast<const unsigned char *>(data);
    std::uint64_t digest = 1469598103934665603ull;
    for (std::size_t index = 0u; index < bytes; ++index) {
        digest ^= octets[index];
        digest *= 1099511628211ull;
    }
    return digest;
}

struct real_result {
    int compute_capability = 0;
    float batch_one_max_absolute = 0.0F;
    float batch_five_max_absolute = 0.0F;
    float batch_five_max_relative = 0.0F;
    std::uint64_t checkpoint_digest = 0u;
    std::uint64_t output_digest = 0u;
    long long load_milliseconds = 0;
};

real_result real_checkpoint_gate(const std::string &model_root) {
    int device = -1;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_require(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require(properties.major == 12 && properties.minor == 0,
            "test requires the strict sm_120 Blackwell path");

    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    require(q4::checkpoint_catalog::open(model_root, &catalog, &error),
            error.empty() ? "checkpoint catalog open failed" : error);
    require(catalog != nullptr, "checkpoint catalog is null");
    const q4::tensor_span *span = catalog->find(kRealTensor);
    require(span != nullptr && span->dtype == q4::tensor_dtype::bf16 &&
                    span->rank == 2u && span->shape[0] == 2560u &&
                    span->shape[1] == 640u && span->bytes == 3276800u,
            "real shared-expert tensor descriptor mismatch");

    q4::bf16_linear_config config{};
    config.input_features = 640u;
    config.output_features = 2560u;
    config.max_batch = 8u;
    config.upload_chunk_bytes = 256u * 1024u;
    config.blas_workspace_bytes = 4u * 1024u * 1024u;

    stream_owner stream;
    std::unique_ptr<q4::resident_bf16_linear> linear;
    q4::bf16_linear_config wrong_shape = config;
    wrong_shape.output_features = 2559u;
    require(q4::resident_bf16_linear::load(
                    *catalog, kRealTensor, wrong_shape, stream.get(), &linear, &error) ==
                    q4::bf16_linear_status::shape_mismatch && !linear,
            "shape mismatch admission did not fail closed");

    q4::bf16_linear_config u8_config = config;
    u8_config.input_features = 1280u;
    u8_config.output_features = 640u;
    require(q4::resident_bf16_linear::load(
                    *catalog, kU8Tensor, u8_config, stream.get(), &linear, &error) ==
                    q4::bf16_linear_status::dtype_mismatch && !linear,
            "non-BF16 admission did not fail closed");

    const auto load_begin = std::chrono::steady_clock::now();
    const q4::bf16_linear_status load_status = q4::resident_bf16_linear::load(
            *catalog, kRealTensor, config, stream.get(), &linear, &error);
    const auto load_end = std::chrono::steady_clock::now();
    require(load_status == q4::bf16_linear_status::ok && linear &&
                    linear->initialized(),
            error.empty() ? std::string("real BF16 tensor load failed: ") +
                                    q4::bf16_linear_status_string(load_status)
                          : error);
    require(linear->tensor_name() == kRealTensor &&
                    linear->weight_bytes() == span->bytes &&
                    linear->device() == device,
            "resident BF16 linear metadata mismatch");

    std::vector<std::uint16_t> host_weight(
            static_cast<std::size_t>(span->bytes / sizeof(std::uint16_t)));
    require(catalog->read_range(kRealTensor, 0u, host_weight.data(),
                                static_cast<std::size_t>(span->bytes), &error),
            error.empty() ? "bounded full tensor read failed" : error);
    const std::uint64_t checkpoint_digest =
            fnv1a(host_weight.data(), static_cast<std::size_t>(span->bytes));
    require(checkpoint_digest != 1469598103934665603ull,
            "real tensor digest did not change");

    q4::bf16_linear_workspace_requirements requirements{};
    require(q4::bf16_linear_get_workspace_requirements(config, &requirements) ==
                    q4::bf16_linear_status::ok,
            "workspace derivation failed");
    device_buffer<float> device_input(
            config.max_batch * static_cast<std::size_t>(config.input_features));
    device_buffer<float> device_output(
            config.max_batch * static_cast<std::size_t>(config.output_features));
    device_buffer<std::uint16_t> device_input_bf16(
            requirements.input_bf16_bytes / sizeof(std::uint16_t));
    device_buffer<unsigned char> device_blas_workspace(
            requirements.blas_workspace_bytes);
    const q4::bf16_linear_scratch scratch{
        device_input_bf16.get(), device_input_bf16.bytes(),
        device_blas_workspace.get(), device_blas_workspace.bytes(),
    };

    auto execute = [&](std::size_t batch, std::vector<float> *host_output) {
        std::vector<float> host_input(
                batch * static_cast<std::size_t>(config.input_features));
        for (std::size_t index = 0u; index < host_input.size(); ++index) {
            host_input[index] =
                    0.125F * std::sin(static_cast<float>(index + 3u) * 0.017F) +
                    0.0625F * std::cos(static_cast<float>(index + 11u) * 0.031F);
        }
        cuda_require(cudaMemcpyAsync(device_input.get(), host_input.data(),
                                     host_input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream.get()),
                     "cudaMemcpyAsync input");
        require(linear->forward(device_input.get(), batch, scratch,
                                device_output.get(), stream.get()) ==
                        q4::bf16_linear_status::ok,
                "resident BF16 forward enqueue failed");
        host_output->resize(batch * static_cast<std::size_t>(config.output_features));
        cuda_require(cudaMemcpyAsync(host_output->data(), device_output.get(),
                                     host_output->size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream.get()),
                     "cudaMemcpyAsync output");
        cuda_require(cudaStreamSynchronize(stream.get()), "cudaStreamSynchronize");

        std::vector<float> expected;
        independent_reference(config, host_weight, host_input, batch, &expected);
        return compare_outputs(*host_output, expected);
    };

    std::vector<float> batch_one;
    const comparison_result one = execute(1u, &batch_one);
    std::vector<float> baseline;
    const comparison_result five = execute(5u, &baseline);
    for (unsigned repetition = 0u; repetition < 5u; ++repetition) {
        std::vector<float> repeated;
        (void)execute(5u, &repeated);
        require(repeated.size() == baseline.size() &&
                        std::memcmp(repeated.data(), baseline.data(),
                                    baseline.size() * sizeof(float)) == 0,
                "cuBLAS BF16 path is not deterministic across repetitions");
    }

    q4::bf16_linear_scratch short_scratch = scratch;
    short_scratch.input_bf16_bytes = sizeof(std::uint16_t);
    require(linear->forward(device_input.get(), 1u, short_scratch,
                            device_output.get(), stream.get()) ==
                    q4::bf16_linear_status::invalid_device_pointer,
            "undersized scratch did not fail closed");
    require(linear->forward(device_input.get(), config.max_batch + 1u, scratch,
                            device_output.get(), stream.get()) ==
                    q4::bf16_linear_status::invalid_argument,
            "oversized prefill batch did not fail closed");
    require(linear->forward(device_output.get(), 1u, scratch,
                            device_output.get(), stream.get()) ==
                    q4::bf16_linear_status::invalid_device_pointer,
            "overlapping input/output did not fail closed");
    std::array<float, 640> host_pointer{};
    require(linear->forward(host_pointer.data(), 1u, scratch,
                            device_output.get(), stream.get()) ==
                    q4::bf16_linear_status::invalid_device_pointer,
            "host input pointer did not fail closed");
    std::vector<float> after_rejected_pointer;
    (void)execute(1u, &after_rejected_pointer);
    require(after_rejected_pointer.size() == batch_one.size() &&
                    std::memcmp(after_rejected_pointer.data(), batch_one.data(),
                                batch_one.size() * sizeof(float)) == 0,
            "a rejected pointer poisoned the following valid CUDA launch");

    real_result result{};
    result.compute_capability = properties.major * 10 + properties.minor;
    result.batch_one_max_absolute = one.max_absolute;
    result.batch_five_max_absolute = five.max_absolute;
    result.batch_five_max_relative = five.max_relative;
    result.checkpoint_digest = checkpoint_digest;
    result.output_digest = fnv1a(baseline.data(), baseline.size() * sizeof(float));
    result.load_milliseconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_begin)
                    .count();
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        synthetic_reference_gate();
        const real_result result = real_checkpoint_gate(argv[1]);
        std::printf(
                "qwen4exp-bf16-linear-test: PASS sm=%d backend=cublasGemmEx-bf16-tensor-op "
                "real_shape=2560x640 batch=1,5 max_abs=%.9g/%.9g max_rel=%.9g "
                "deterministic=5/5 load_ms=%lld checkpoint=%016llx output=%016llx\n",
                result.compute_capability,
                result.batch_one_max_absolute,
                result.batch_five_max_absolute,
                result.batch_five_max_relative,
                result.load_milliseconds,
                static_cast<unsigned long long>(result.checkpoint_digest),
                static_cast<unsigned long long>(result.output_digest));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-bf16-linear-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
