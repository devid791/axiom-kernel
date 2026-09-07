#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void cuda_require(cudaError_t result, const char *operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

std::uint16_t f32_to_bf16(float value) {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t exponent = bits & 0x7f800000u;
    const std::uint32_t mantissa = bits & 0x007fffffu;
    if (exponent == 0x7f800000u && mantissa != 0u) {
        return static_cast<std::uint16_t>((bits >> 16u) | 0x0040u);
    }
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>(bits >> 16u);
}

float bf16_to_f32(std::uint16_t encoded) {
    const std::uint32_t bits = static_cast<std::uint32_t>(encoded) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename Type>
class device_storage final {
public:
    explicit device_storage(std::size_t elements) : elements_(elements) {
        if (elements_ == 0u ||
            elements_ > std::numeric_limits<std::size_t>::max() / sizeof(Type)) {
            throw std::runtime_error("invalid device allocation size");
        }
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements_ * sizeof(Type)),
                     "cudaMalloc");
    }

    ~device_storage() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }

    device_storage(const device_storage &) = delete;
    device_storage &operator=(const device_storage &) = delete;

    Type *get() noexcept { return pointer_; }
    const Type *get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return elements_; }

private:
    Type *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

template <typename Type>
void upload(device_storage<Type> &destination,
            const std::vector<Type> &source,
            cudaStream_t stream) {
    require(destination.size() == source.size(), "upload size mismatch");
    cuda_require(cudaMemcpyAsync(destination.get(), source.data(),
                                 source.size() * sizeof(Type),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync upload");
}

struct host_weights {
    std::vector<std::uint16_t> gate;
    std::vector<std::uint16_t> up;
    std::vector<std::uint16_t> down;
    std::vector<std::uint16_t> output_gate;
};

host_weights make_weights(const q4::shared_expert_config &config) {
    const std::size_t projection_elements =
            static_cast<std::size_t>(config.hidden) * config.intermediate;
    host_weights result;
    result.gate.resize(projection_elements);
    result.up.resize(projection_elements);
    result.down.resize(projection_elements);
    result.output_gate.resize(config.hidden);
    for (std::size_t index = 0u; index < projection_elements; ++index) {
        result.gate[index] = f32_to_bf16(
                0.035f * std::sin(static_cast<float>(index + 1u) * 0.173f));
        result.up[index] = f32_to_bf16(
                0.041f * std::cos(static_cast<float>(index + 3u) * 0.137f));
        result.down[index] = f32_to_bf16(
                0.029f * std::sin(static_cast<float>(index + 7u) * 0.097f));
    }
    for (std::size_t index = 0u; index < result.output_gate.size(); ++index) {
        result.output_gate[index] = f32_to_bf16(
                0.023f * std::cos(static_cast<float>(index + 5u) * 0.113f));
    }
    return result;
}

void independent_oracle(const q4::shared_expert_config &config,
                        const host_weights &weights,
                        const std::vector<float> &input,
                        std::vector<float> *intermediate,
                        float *output_gate,
                        std::vector<float> *output) {
    require(intermediate != nullptr && output_gate != nullptr && output != nullptr,
            "null independent oracle output");
    intermediate->assign(config.intermediate, 0.0f);
    output->assign(config.hidden, 0.0f);

    double gate_sum = 0.0;
    for (std::uint32_t column = 0u; column < config.hidden; ++column) {
        gate_sum += static_cast<double>(bf16_to_f32(weights.output_gate[column])) *
                    static_cast<double>(input[column]);
    }
    *output_gate = static_cast<float>(1.0 / (1.0 + std::exp(-gate_sum)));
    for (std::uint32_t row = 0u; row < config.intermediate; ++row) {
        double gate = 0.0;
        double up = 0.0;
        const std::size_t offset = static_cast<std::size_t>(row) * config.hidden;
        for (std::uint32_t column = 0u; column < config.hidden; ++column) {
            gate += static_cast<double>(bf16_to_f32(weights.gate[offset + column])) *
                    static_cast<double>(input[column]);
            up += static_cast<double>(bf16_to_f32(weights.up[offset + column])) *
                  static_cast<double>(input[column]);
        }
        (*intermediate)[row] = static_cast<float>(
                (gate / (1.0 + std::exp(-gate))) * up);
    }
    for (std::uint32_t row = 0u; row < config.hidden; ++row) {
        double sum = 0.0;
        const std::size_t offset =
                static_cast<std::size_t>(row) * config.intermediate;
        for (std::uint32_t column = 0u; column < config.intermediate; ++column) {
            sum += static_cast<double>(bf16_to_f32(weights.down[offset + column])) *
                   static_cast<double>((*intermediate)[column]);
        }
        (*output)[row] = static_cast<float>(sum * *output_gate);
    }
}

float maximum_absolute_error(const std::vector<float> &left,
                             const std::vector<float> &right) {
    require(left.size() == right.size(), "error vector size mismatch");
    float maximum = 0.0f;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        require(std::isfinite(left[index]) && std::isfinite(right[index]),
                "non-finite comparison value");
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

struct synthetic_result {
    float host_max_error = 0.0f;
    float cuda_max_error = 0.0f;
    int compute_capability = 0;
};

synthetic_result run_synthetic_gate() {
    const q4::shared_expert_config exact = q4::shared_expert_qwen4_exp_config();
    require(q4::shared_expert_is_qwen4_exp_contract(exact),
            "exact qwen4_exp shared expert contract rejected");
    q4::shared_expert_tensor_spec spec{};
    require(q4::shared_expert_tensor_spec_for(
                    exact, q4::shared_expert_tensor::gate_projection, &spec) ==
                    q4::shared_expert_status::ok &&
            spec.rows == 640u && spec.columns == 2560u &&
            spec.bytes == 3276800u,
            "exact gate projection contract mismatch");
    require(q4::shared_expert_tensor_spec_for(
                    exact, q4::shared_expert_tensor::down_projection, &spec) ==
                    q4::shared_expert_status::ok &&
            spec.rows == 2560u && spec.columns == 640u &&
            spec.bytes == 3276800u,
            "exact down projection contract mismatch");
    require(q4::shared_expert_tensor_spec_for(
                    exact, q4::shared_expert_tensor::output_gate, &spec) ==
                    q4::shared_expert_status::ok &&
            spec.rows == 1u && spec.columns == 2560u && spec.bytes == 5120u,
            "exact output gate contract mismatch");
    char name[128]{};
    require(q4::shared_expert_tensor_name(
                    0u, q4::shared_expert_tensor::gate_projection,
                    name, sizeof(name)) == q4::shared_expert_status::ok &&
            std::string(name) ==
                    "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "checkpoint tensor name mismatch");
    char short_name[8]{};
    require(q4::shared_expert_tensor_name(
                    0u, q4::shared_expert_tensor::gate_projection,
                    short_name, sizeof(short_name)) ==
                    q4::shared_expert_status::invalid_argument,
            "short tensor-name buffer did not fail closed");
    require(q4::shared_expert_tensor_name(
                    q4::kSharedExpertLayers,
                    q4::shared_expert_tensor::gate_projection,
                    name, sizeof(name)) == q4::shared_expert_status::invalid_argument,
            "out-of-range layer did not fail closed");
    q4::shared_expert_config overflowing{};
    overflowing.hidden = std::numeric_limits<std::uint32_t>::max();
    overflowing.intermediate = std::numeric_limits<std::uint32_t>::max();
    require(q4::shared_expert_validate_config(overflowing) ==
                    q4::shared_expert_status::overflow,
            "overflowing config was not rejected");

    q4::shared_expert_config config{};
    config.hidden = 37u;
    config.intermediate = 19u;
    const host_weights weights = make_weights(config);
    std::vector<float> input(config.hidden);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = 0.21f * std::sin(static_cast<float>(index + 11u) * 0.071f);
    }

    std::vector<float> oracle_intermediate;
    std::vector<float> oracle_output;
    float oracle_gate = 0.0f;
    independent_oracle(config, weights, input, &oracle_intermediate,
                       &oracle_gate, &oracle_output);

    std::vector<float> host_intermediate(config.intermediate, 0.0f);
    std::vector<float> host_output(config.hidden, 0.0f);
    float host_gate = 0.0f;
    const q4::shared_expert_weights_bf16 host_weight_view{
        weights.gate.data(), weights.up.data(), weights.down.data(),
        weights.output_gate.data(),
    };
    require(q4::shared_expert_forward_f32_host(
                    config, host_weight_view, input.data(),
                    host_intermediate.data(), &host_gate, host_output.data()) ==
                    q4::shared_expert_status::ok,
            "allocation-free host reference failed");
    const float host_error = maximum_absolute_error(host_output, oracle_output);
    require(host_error <= 2.0e-7f && std::abs(host_gate - oracle_gate) <= 1.0e-7f,
            "host reference differs from independent oracle");
    std::vector<float> invalid_host_input = input;
    invalid_host_input[2] = std::numeric_limits<float>::infinity();
    require(q4::shared_expert_forward_f32_host(
                    config, host_weight_view, invalid_host_input.data(),
                    host_intermediate.data(), &host_gate, host_output.data()) ==
                    q4::shared_expert_status::non_finite,
            "host reference accepted non-finite input");
    require(std::all_of(host_output.begin(), host_output.end(),
                        [](float value) { return value == 0.0f; }),
            "host reference did not zero non-finite output fail-closed");

    int device = -1;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_require(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require(properties.major * 10 + properties.minor >= 120,
            "shared expert test requires SM120 or newer");

    cudaStream_t stream = nullptr;
    cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");
    try {
        device_storage<std::uint16_t> device_gate(weights.gate.size());
        device_storage<std::uint16_t> device_up(weights.up.size());
        device_storage<std::uint16_t> device_down(weights.down.size());
        device_storage<std::uint16_t> device_output_gate_weight(
                weights.output_gate.size());
        device_storage<float> device_input(input.size());
        device_storage<float> device_intermediate(config.intermediate);
        device_storage<float> device_gate_scalar(1u);
        device_storage<std::uint32_t> device_status(1u);
        device_storage<float> device_output(config.hidden);

        upload(device_gate, weights.gate, stream);
        upload(device_up, weights.up, stream);
        upload(device_down, weights.down, stream);
        upload(device_output_gate_weight, weights.output_gate, stream);
        upload(device_input, input, stream);

        const q4::shared_expert_weights_bf16 device_weight_view{
            device_gate.get(), device_up.get(), device_down.get(),
            device_output_gate_weight.get(),
        };
        const q4::shared_expert_workspace_f32 workspace{
            device_intermediate.get(), device_gate_scalar.get(), device_status.get(),
        };
        require(q4::shared_expert_forward_f32_cuda(
                        device, config, device_weight_view, device_input.get(),
                        workspace, device_output.get(), stream) ==
                        q4::shared_expert_status::ok,
                "CUDA shared expert enqueue failed");
        std::vector<float> cuda_output(config.hidden, 0.0f);
        float cuda_gate = 0.0f;
        std::uint32_t cuda_status = 0u;
        cuda_require(cudaMemcpyAsync(cuda_output.data(), device_output.get(),
                                     cuda_output.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync output");
        cuda_require(cudaMemcpyAsync(&cuda_gate, device_gate_scalar.get(),
                                     sizeof(cuda_gate), cudaMemcpyDeviceToHost,
                                     stream),
                     "cudaMemcpyAsync gate");
        cuda_require(cudaMemcpyAsync(&cuda_status, device_status.get(),
                                     sizeof(cuda_status), cudaMemcpyDeviceToHost,
                                     stream),
                     "cudaMemcpyAsync status");
        cuda_require(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        require(cuda_status == static_cast<std::uint32_t>(
                                       q4::shared_expert_status::ok),
                "CUDA shared expert reported a semantic error");
        const float cuda_error = maximum_absolute_error(cuda_output, oracle_output);
        require(cuda_error <= 2.0e-6f && std::abs(cuda_gate - oracle_gate) <= 2.0e-7f,
                "CUDA shared expert differs from independent oracle");

        const std::vector<float> first_output = cuda_output;
        for (int repetition = 0; repetition < 4; ++repetition) {
            require(q4::shared_expert_forward_f32_cuda(
                            device, config, device_weight_view, device_input.get(),
                            workspace, device_output.get(), stream) ==
                            q4::shared_expert_status::ok,
                    "repeated CUDA shared expert enqueue failed");
            cuda_require(cudaMemcpyAsync(cuda_output.data(), device_output.get(),
                                         cuda_output.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream),
                         "cudaMemcpyAsync repeated output");
            cuda_require(cudaStreamSynchronize(stream),
                         "cudaStreamSynchronize repeated");
            require(cuda_output == first_output,
                    "CUDA shared expert is not deterministic");
        }

        q4::shared_expert_weights_bf16 invalid_weight_view = device_weight_view;
        invalid_weight_view.gate = weights.gate.data();
        require(q4::shared_expert_forward_f32_cuda(
                        device, config, invalid_weight_view, device_input.get(),
                        workspace, device_output.get(), stream) ==
                        q4::shared_expert_status::invalid_device_pointer,
                "host weight pointer was accepted by CUDA adapter");

        std::vector<float> non_finite_input = input;
        non_finite_input[3] = std::numeric_limits<float>::quiet_NaN();
        upload(device_input, non_finite_input, stream);
        require(q4::shared_expert_forward_f32_cuda(
                        device, config, device_weight_view, device_input.get(),
                        workspace, device_output.get(), stream) ==
                        q4::shared_expert_status::ok,
                "non-finite CUDA gate did not enqueue");
        cuda_require(cudaMemcpyAsync(cuda_output.data(), device_output.get(),
                                     cuda_output.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync non-finite output");
        cuda_require(cudaMemcpyAsync(&cuda_status, device_status.get(),
                                     sizeof(cuda_status), cudaMemcpyDeviceToHost,
                                     stream),
                     "cudaMemcpyAsync non-finite status");
        cuda_require(cudaStreamSynchronize(stream),
                     "cudaStreamSynchronize non-finite");
        require(cuda_status == static_cast<std::uint32_t>(
                                       q4::shared_expert_status::non_finite),
                "non-finite input was not reported asynchronously");
        require(std::all_of(cuda_output.begin(), cuda_output.end(),
                            [](float value) { return value == 0.0f; }),
                "non-finite CUDA output was not zeroed fail-closed");

        cuda_require(cudaStreamDestroy(stream), "cudaStreamDestroy");
        stream = nullptr;
        synthetic_result result{};
        result.host_max_error = host_error;
        result.cuda_max_error = cuda_error;
        result.compute_capability = properties.major * 10 + properties.minor;
        return result;
    } catch (...) {
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        throw;
    }
}

std::uint64_t validate_real_checkpoint(const std::string &model_root) {
    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    if (!q4::checkpoint_catalog::open(model_root, &catalog, &error)) {
        throw std::runtime_error(
                error.empty() ? "could not open checkpoint catalog" : error);
    }
    require(catalog != nullptr, "checkpoint catalog is null");

    constexpr std::array<q4::shared_expert_tensor, 4> tensors{
        q4::shared_expert_tensor::gate_projection,
        q4::shared_expert_tensor::up_projection,
        q4::shared_expert_tensor::down_projection,
        q4::shared_expert_tensor::output_gate,
    };
    constexpr std::array<std::uint32_t, 2> layers{0u, 47u};
    const q4::shared_expert_config config = q4::shared_expert_qwen4_exp_config();
    std::uint64_t digest = 1469598103934665603ull;
    for (const std::uint32_t layer : layers) {
        for (const q4::shared_expert_tensor tensor : tensors) {
            char name[128]{};
            require(q4::shared_expert_tensor_name(
                            layer, tensor, name, sizeof(name)) ==
                            q4::shared_expert_status::ok,
                    "could not construct real checkpoint tensor name");
            q4::shared_expert_tensor_spec expected{};
            require(q4::shared_expert_tensor_spec_for(config, tensor, &expected) ==
                            q4::shared_expert_status::ok,
                    "could not derive real checkpoint tensor spec");
            const q4::tensor_span *span = catalog->find(name);
            require(span != nullptr, "shared expert tensor missing from checkpoint");
            require(span->dtype == q4::tensor_dtype::bf16 && span->rank == 2u &&
                            span->shape[0] == expected.rows &&
                            span->shape[1] == expected.columns &&
                            span->bytes == expected.bytes,
                    "shared expert checkpoint dtype/shape/span mismatch");

            const std::array<std::uint64_t, 3> offsets{
                0u,
                (span->bytes / 2u) & ~std::uint64_t{1u},
                span->bytes - sizeof(std::uint16_t),
            };
            for (const std::uint64_t offset : offsets) {
                std::uint16_t encoded = 0u;
                if (!catalog->read_range(name, offset, &encoded,
                                         sizeof(encoded), &error)) {
                    throw std::runtime_error(
                            error.empty() ? "real checkpoint range read failed"
                                          : error);
                }
                require(std::isfinite(bf16_to_f32(encoded)),
                        "real checkpoint shared expert contains sampled non-finite BF16");
                digest ^= encoded;
                digest *= 1099511628211ull;
            }
        }
    }
    require(digest != 1469598103934665603ull,
            "real checkpoint sample digest did not change");
    return digest;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        const synthetic_result synthetic = run_synthetic_gate();
        const std::uint64_t checkpoint_digest = validate_real_checkpoint(argv[1]);
        std::printf(
                "qwen4exp-shared-expert-test: PASS sm=%d host_max_abs=%.9g "
                "cuda_max_abs=%.9g real_tensors=8 sampled_bf16=24 digest=%016llx\n",
                synthetic.compute_capability,
                synthetic.host_max_error,
                synthetic.cuda_max_error,
                static_cast<unsigned long long>(checkpoint_digest));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-shared-expert-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
