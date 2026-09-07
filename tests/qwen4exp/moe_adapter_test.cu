#include "axiom/qwen4exp/moe_adapter.hpp"

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/moe.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
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

template <typename Type>
class device_buffer final {
public:
    explicit device_buffer(std::size_t elements) : elements_(elements) {
        require(elements != 0u, "zero-sized device allocation");
        cuda_require(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements * sizeof(Type)), "cudaMalloc");
    }
    ~device_buffer() { if (pointer_ != nullptr) (void)cudaFree(pointer_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    Type *get() noexcept { return pointer_; }
    const Type *get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return elements_; }

private:
    Type *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

class stream_guard final {
public:
    stream_guard() {
        cuda_require(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~stream_guard() { if (stream_ != nullptr) (void)cudaStreamDestroy(stream_); }
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

float bf16_to_f32(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::uint16_t f32_to_bf16(float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t lower = bits & 0xffffu;
    const std::uint32_t upper = bits >> 16u;
    const std::uint32_t rounded = upper +
            ((lower > 0x8000u || (lower == 0x8000u && (upper & 1u) != 0u)) ? 1u : 0u);
    return static_cast<std::uint16_t>(rounded);
}

float e2m1(std::uint8_t nibble) noexcept {
    static constexpr float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const float magnitude = values[nibble & 7u];
    return (nibble & 8u) != 0u ? -magnitude : magnitude;
}

float e4m3(std::uint8_t encoded) noexcept {
    const bool negative = (encoded & 0x80u) != 0u;
    const unsigned exponent = (encoded >> 3u) & 15u;
    const unsigned mantissa = encoded & 7u;
    const float value = exponent == 0u
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                         static_cast<int>(exponent) - 7);
    return negative ? -value : value;
}

template <typename Type>
std::vector<Type> read_tensor(const q4::checkpoint_catalog &catalog,
                              const std::string &name,
                              q4::tensor_dtype dtype,
                              std::initializer_list<std::uint64_t> shape) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr, "missing checkpoint tensor: " + name);
    require(span->dtype == dtype && span->rank == shape.size(),
            "dtype/rank mismatch: " + name);
    std::uint64_t elements = 1u;
    std::size_t dimension = 0u;
    for (const std::uint64_t expected : shape) {
        require(span->shape[dimension++] == expected,
                "shape mismatch: " + name);
        require(expected == 0u ||
                        elements <= std::numeric_limits<std::uint64_t>::max() / expected,
                "element-count overflow: " + name);
        elements *= expected;
    }
    const std::uint64_t bytes = elements * sizeof(Type);
    require(span->bytes == bytes && elements <= std::numeric_limits<std::size_t>::max(),
            "byte-count mismatch: " + name);
    std::vector<Type> values(static_cast<std::size_t>(elements));
    std::string error;
    require(catalog.read_range(name, 0u, values.data(), values.size() * sizeof(Type),
                               &error),
            error.empty() ? "checkpoint read failed: " + name : error);
    return values;
}

float read_scalar_f32(const q4::checkpoint_catalog &catalog,
                      const std::string &name) {
    const q4::tensor_span *span = catalog.find(name);
    require(span != nullptr && span->dtype == q4::tensor_dtype::f32 &&
                    span->rank == 0u && span->bytes == sizeof(float),
            "scalar contract mismatch: " + name);
    float value = 0.0f;
    std::string error;
    require(catalog.read_range(name, 0u, &value, sizeof(value), &error),
            error.empty() ? "scalar read failed: " + name : error);
    require(std::isfinite(value) && value > 0.0f,
            "invalid positive ModelOpt scalar: " + name);
    return value;
}

std::string layer_prefix(std::uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + ".mlp.";
}

struct route_oracle {
    std::array<std::uint32_t, q4::kMoeTopK> experts{};
    std::array<float, q4::kMoeTopK> weights{};
};

route_oracle independent_router(const q4::checkpoint_catalog &catalog,
                                std::uint32_t layer,
                                const std::vector<float> &input) {
    require(input.size() == q4::kMoeHidden, "router input shape mismatch");
    const auto weight = read_tensor<std::uint16_t>(
            catalog, layer_prefix(layer) + "gate.weight",
            q4::tensor_dtype::bf16, {q4::kMoeExperts, q4::kMoeHidden});
    std::array<float, q4::kMoeExperts> logits{};
    for (std::uint32_t expert = 0u; expert < q4::kMoeExperts; ++expert) {
        float sum = 0.0f;
        const std::size_t base = static_cast<std::size_t>(expert) * q4::kMoeHidden;
        for (std::uint32_t column = 0u; column < q4::kMoeHidden; ++column) {
            const float rounded_input = bf16_to_f32(f32_to_bf16(input[column]));
            sum = std::fma(bf16_to_f32(weight[base + column]), rounded_input, sum);
        }
        require(std::isfinite(sum), "non-finite independent router logit");
        logits[expert] = sum;
    }

    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::array<float, q4::kMoeExperts> probabilities{};
    float denominator = 0.0f;
    for (std::uint32_t expert = 0u; expert < q4::kMoeExperts; ++expert) {
        probabilities[expert] = std::exp(logits[expert] - maximum);
        denominator += probabilities[expert];
    }
    require(std::isfinite(denominator) && denominator > 0.0f,
            "invalid independent router softmax");
    for (float &value : probabilities) value /= denominator;

    std::array<std::uint32_t, q4::kMoeExperts> order{};
    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(order.begin(), order.begin() + q4::kMoeTopK, order.end(),
                      [&](std::uint32_t left, std::uint32_t right) {
                          return probabilities[left] > probabilities[right] ||
                                 (probabilities[left] == probabilities[right] &&
                                  left < right);
                      });
    route_oracle result{};
    float selected_sum = 0.0f;
    for (std::uint32_t slot = 0u; slot < q4::kMoeTopK; ++slot) {
        result.experts[slot] = order[slot];
        selected_sum += probabilities[order[slot]];
    }
    for (std::uint32_t slot = 0u; slot < q4::kMoeTopK; ++slot) {
        result.weights[slot] = probabilities[result.experts[slot]] / selected_sum;
    }
    return result;
}

float nvfp4_matvec_row(const std::vector<std::uint8_t> &weight,
                       const std::vector<std::uint8_t> &scales,
                       float global_scale,
                       std::uint32_t rows,
                       std::uint32_t columns,
                       std::uint32_t row,
                       const float *input) {
    require(row < rows && columns % 16u == 0u,
            "invalid independent NVFP4 matvec geometry");
    const std::size_t weight_base =
            static_cast<std::size_t>(row) * (columns / 2u);
    const std::size_t scale_base =
            static_cast<std::size_t>(row) * (columns / 16u);
    float sum = 0.0f;
    for (std::uint32_t column = 0u; column < columns; ++column) {
        const std::uint8_t packed = weight[weight_base + column / 2u];
        const std::uint8_t nibble = (column & 1u) == 0u
                ? packed & 15u : packed >> 4u;
        sum = std::fma(e2m1(nibble) * e4m3(scales[scale_base + column / 16u]) *
                               global_scale,
                       input[column], sum);
    }
    return sum;
}

void independent_routed_experts(
        const q4::checkpoint_catalog &catalog,
        std::uint32_t layer,
        const route_oracle &route,
        const std::vector<float> &input,
        std::vector<float> *output) {
    require(output != nullptr, "null routed oracle output");
    output->assign(q4::kMoeHidden, 0.0f);
    const std::string prefix = layer_prefix(layer) + "experts.";
    std::vector<float> intermediate(q4::kMoeIntermediate);

    for (std::uint32_t slot = 0u; slot < q4::kMoeTopK; ++slot) {
        const std::string expert = prefix + std::to_string(route.experts[slot]) + ".";
        const auto gate_weight = read_tensor<std::uint8_t>(
                catalog, expert + "gate_proj.weight", q4::tensor_dtype::u8,
                {q4::kMoeIntermediate, q4::kMoeHidden / 2u});
        const auto gate_scale = read_tensor<std::uint8_t>(
                catalog, expert + "gate_proj.weight_scale",
                q4::tensor_dtype::f8_e4m3,
                {q4::kMoeIntermediate, q4::kMoeHidden / 16u});
        const auto up_weight = read_tensor<std::uint8_t>(
                catalog, expert + "up_proj.weight", q4::tensor_dtype::u8,
                {q4::kMoeIntermediate, q4::kMoeHidden / 2u});
        const auto up_scale = read_tensor<std::uint8_t>(
                catalog, expert + "up_proj.weight_scale",
                q4::tensor_dtype::f8_e4m3,
                {q4::kMoeIntermediate, q4::kMoeHidden / 16u});
        const auto down_weight = read_tensor<std::uint8_t>(
                catalog, expert + "down_proj.weight", q4::tensor_dtype::u8,
                {q4::kMoeHidden, q4::kMoeIntermediate / 2u});
        const auto down_scale = read_tensor<std::uint8_t>(
                catalog, expert + "down_proj.weight_scale",
                q4::tensor_dtype::f8_e4m3,
                {q4::kMoeHidden, q4::kMoeIntermediate / 16u});
        const float gate_global = read_scalar_f32(
                catalog, expert + "gate_proj.weight_scale_2");
        const float up_global = read_scalar_f32(
                catalog, expert + "up_proj.weight_scale_2");
        const float down_global = read_scalar_f32(
                catalog, expert + "down_proj.weight_scale_2");

        /* input_scale is calibration metadata for a quantized activation
         * path.  The adapter consumes F32 activations, so it is admitted and
         * validated but is not a multiplicative model parameter. */
        (void)read_scalar_f32(catalog, expert + "gate_proj.input_scale");
        (void)read_scalar_f32(catalog, expert + "up_proj.input_scale");
        (void)read_scalar_f32(catalog, expert + "down_proj.input_scale");

        for (std::uint32_t row = 0u; row < q4::kMoeIntermediate; ++row) {
            const float gate = nvfp4_matvec_row(
                    gate_weight, gate_scale, gate_global,
                    q4::kMoeIntermediate, q4::kMoeHidden, row, input.data());
            const float up = nvfp4_matvec_row(
                    up_weight, up_scale, up_global,
                    q4::kMoeIntermediate, q4::kMoeHidden, row, input.data());
            intermediate[row] = (gate / (1.0f + std::exp(-gate))) * up;
        }
        for (std::uint32_t row = 0u; row < q4::kMoeHidden; ++row) {
            const float down = nvfp4_matvec_row(
                    down_weight, down_scale, down_global,
                    q4::kMoeHidden, q4::kMoeIntermediate, row,
                    intermediate.data());
            (*output)[row] = std::fma(route.weights[slot], down, (*output)[row]);
        }
    }
}

void independent_shared_expert(const q4::checkpoint_catalog &catalog,
                               std::uint32_t layer,
                               const std::vector<float> &input,
                               std::vector<float> *output) {
    require(output != nullptr, "null shared oracle output");
    const std::string prefix = layer_prefix(layer) + "shared_expert.";
    const auto gate = read_tensor<std::uint16_t>(
            catalog, prefix + "gate_proj.weight", q4::tensor_dtype::bf16,
            {q4::kMoeIntermediate, q4::kMoeHidden});
    const auto up = read_tensor<std::uint16_t>(
            catalog, prefix + "up_proj.weight", q4::tensor_dtype::bf16,
            {q4::kMoeIntermediate, q4::kMoeHidden});
    const auto down = read_tensor<std::uint16_t>(
            catalog, prefix + "down_proj.weight", q4::tensor_dtype::bf16,
            {q4::kMoeHidden, q4::kMoeIntermediate});
    const auto output_gate_weight = read_tensor<std::uint16_t>(
            catalog, layer_prefix(layer) + "shared_expert_gate.weight",
            q4::tensor_dtype::bf16, {1u, q4::kMoeHidden});

    double output_gate_sum = 0.0;
    for (std::uint32_t column = 0u; column < q4::kMoeHidden; ++column) {
        output_gate_sum += static_cast<double>(bf16_to_f32(output_gate_weight[column])) *
                           static_cast<double>(input[column]);
    }
    const float output_gate = static_cast<float>(
            1.0 / (1.0 + std::exp(-output_gate_sum)));
    std::vector<float> intermediate(q4::kMoeIntermediate);
    for (std::uint32_t row = 0u; row < q4::kMoeIntermediate; ++row) {
        double gate_value = 0.0;
        double up_value = 0.0;
        const std::size_t base = static_cast<std::size_t>(row) * q4::kMoeHidden;
        for (std::uint32_t column = 0u; column < q4::kMoeHidden; ++column) {
            gate_value += static_cast<double>(bf16_to_f32(gate[base + column])) *
                          static_cast<double>(input[column]);
            up_value += static_cast<double>(bf16_to_f32(up[base + column])) *
                        static_cast<double>(input[column]);
        }
        intermediate[row] = static_cast<float>(
                (gate_value / (1.0 + std::exp(-gate_value))) * up_value);
    }
    output->assign(q4::kMoeHidden, 0.0f);
    for (std::uint32_t row = 0u; row < q4::kMoeHidden; ++row) {
        double value = 0.0;
        const std::size_t base = static_cast<std::size_t>(row) * q4::kMoeIntermediate;
        for (std::uint32_t column = 0u; column < q4::kMoeIntermediate; ++column) {
            value += static_cast<double>(bf16_to_f32(down[base + column])) *
                     static_cast<double>(intermediate[column]);
        }
        (*output)[row] = static_cast<float>(value * output_gate);
    }
}

std::uint64_t fnv1a(const void *data, std::size_t bytes) noexcept {
    const auto *values = static_cast<const std::uint8_t *>(data);
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

struct comparison_result {
    float maximum_absolute = 0.0f;
    float relative_rms = 0.0f;
};

comparison_result compare(const std::vector<float> &actual,
                          const std::vector<float> &expected) {
    require(actual.size() == expected.size(), "comparison shape mismatch");
    double squared_error = 0.0;
    double squared_reference = 0.0;
    comparison_result result{};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]) && std::isfinite(expected[index]),
                "non-finite output comparison");
        const float error = std::abs(actual[index] - expected[index]);
        result.maximum_absolute = std::max(result.maximum_absolute, error);
        squared_error += static_cast<double>(error) * error;
        squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
    result.relative_rms = static_cast<float>(
            std::sqrt(squared_error / std::max(squared_reference, 1.0e-30)));
    return result;
}

std::vector<float> execute_generation(q4::routed_moe_layer_adapter &adapter,
                                      const std::vector<float> &input,
                                      device_buffer<float> &device_input,
                                      device_buffer<float> &device_output,
                                      cudaStream_t stream,
                                      route_oracle *observed_route,
                                      bool check_busy) {
    cuda_require(cudaMemcpyAsync(device_input.get(), input.data(),
                                 input.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, stream),
                 "cudaMemcpyAsync input");
    std::string error;
    require(adapter.enqueue_route(device_input.get(), stream, &error) ==
                    q4::moe_adapter_status::ok,
            error.empty() ? "enqueue_route failed" : error);
    if (check_busy) {
        require(adapter.enqueue_route(device_input.get(), stream, &error) ==
                        q4::moe_adapter_status::busy,
                "overlapping route was not rejected");
    }
    q4::routed_moe_layer_adapter::prepared_generation generation;
    require(adapter.prepare_route(stream, &generation, &error) ==
                    q4::moe_adapter_status::ok,
            error.empty() ? "prepare_route failed" : error);
    require(generation.valid() && generation.selected_count() == q4::kMoeTopK,
            "prepared generation contract mismatch");
    if (observed_route != nullptr) {
        std::copy_n(generation.selected_experts(), q4::kMoeTopK,
                    observed_route->experts.begin());
        std::copy_n(generation.router_weights(), q4::kMoeTopK,
                    observed_route->weights.begin());
    }
    require(generation.forward(device_input.get(), device_output.get(), stream,
                               &error) == q4::moe_adapter_status::ok,
            error.empty() ? "prepared forward failed" : error);
    require(generation.forward_enqueued(), "forward enqueue state was not recorded");
    require(generation.commit(&error),
            error.empty() ? "generation commit failed" : error);
    require(generation.committed() &&
                    generation.device_status() == q4::moe_adapter_status::ok,
            "committed generation did not retain green status");

    std::vector<float> output(q4::kMoeHidden);
    cuda_require(cudaMemcpy(output.data(), device_output.get(),
                            output.size() * sizeof(float), cudaMemcpyDeviceToHost),
                 "cudaMemcpy output");
    return output;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }
    try {
        int device_count = 0;
        cuda_require(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "routed-MoE adapter test requires CUDA");
        cuda_require(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        cuda_require(cudaGetDeviceProperties(&properties, 0),
                     "cudaGetDeviceProperties");
        require(properties.major * 10 + properties.minor >= 120,
                "routed-MoE adapter requires SM120");

        stream_guard stream;
        constexpr std::uint32_t layer = 0u;
        q4::moe_adapter_options options{};
        options.device = 0;
        options.layer = layer;
        options.ram_capacity_bytes = 64ull * 1024ull * 1024ull;
        options.prefetch_workers = 2u;

        const auto load_begin = std::chrono::steady_clock::now();
        std::unique_ptr<q4::routed_moe_layer_adapter> adapter;
        std::string error;
        const q4::moe_adapter_status load_status =
                q4::routed_moe_layer_adapter::load(
                        argv[1], options, stream.get(), &adapter, &error);
        require(load_status == q4::moe_adapter_status::ok && adapter &&
                        adapter->initialized(),
                error.empty() ? std::string("adapter load failed: ") +
                                        q4::moe_adapter_status_string(load_status)
                              : error);
        const auto load_end = std::chrono::steady_clock::now();

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error) && catalog,
                error.empty() ? "checkpoint catalog open failed" : error);

        std::vector<float> input(q4::kMoeHidden);
        for (std::size_t index = 0u; index < input.size(); ++index) {
            input[index] = 0.0625f * std::sin(static_cast<float>(index + 1u) * 0.013f) +
                           0.03125f * std::cos(static_cast<float>(index + 7u) * 0.029f);
        }
        device_buffer<float> device_input(q4::kMoeHidden);
        device_buffer<float> device_output(q4::kMoeHidden);

        const route_oracle expected_route = independent_router(*catalog, layer, input);
        route_oracle observed_route{};
        const std::vector<float> first = execute_generation(
                *adapter, input, device_input, device_output, stream.get(),
                &observed_route, true);
        require(observed_route.experts == expected_route.experts,
                "real BF16 router selected different experts than independent oracle");
        float maximum_route_weight_error = 0.0f;
        for (std::size_t slot = 0u; slot < q4::kMoeTopK; ++slot) {
            maximum_route_weight_error = std::max(
                    maximum_route_weight_error,
                    std::abs(observed_route.weights[slot] -
                             expected_route.weights[slot]));
        }
        require(maximum_route_weight_error <= 2.0e-5f,
                "real top-10 weights differ from independent F32 softmax oracle");

        std::vector<float> routed;
        std::vector<float> shared;
        independent_routed_experts(*catalog, layer, expected_route, input, &routed);
        independent_shared_expert(*catalog, layer, input, &shared);
        std::vector<float> expected(q4::kMoeHidden);
        for (std::size_t index = 0u; index < expected.size(); ++index) {
            expected[index] = routed[index] + shared[index];
        }
        const comparison_result accuracy = compare(first, expected);
        require(accuracy.maximum_absolute <= 2.0e-3f &&
                        accuracy.relative_rms <= 8.0e-4f,
                "real routed+shared output differs from independent checkpoint oracle");

        const std::vector<float> repeated = execute_generation(
                *adapter, input, device_input, device_output, stream.get(),
                nullptr, false);
        require(repeated.size() == first.size() &&
                        std::memcmp(repeated.data(), first.data(),
                                    first.size() * sizeof(float)) == 0,
                "real routed-MoE adapter is not bitwise deterministic");

        cuda_require(cudaMemcpyAsync(device_input.get(), input.data(),
                                     input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream.get()),
                     "cudaMemcpyAsync rollback input");
        require(adapter->enqueue_route(device_input.get(), stream.get(), &error) ==
                        q4::moe_adapter_status::ok,
                "rollback route enqueue failed");
        q4::routed_moe_layer_adapter::prepared_generation rolled_back;
        require(adapter->prepare_route(stream.get(), &rolled_back, &error) ==
                        q4::moe_adapter_status::ok && rolled_back.valid(),
                error.empty() ? "rollback preparation failed" : error);
        require(rolled_back.rollback(&error),
                error.empty() ? "explicit expert transaction rollback failed" : error);
        require(!rolled_back.valid() && !rolled_back.committed(),
                "rolled-back generation remained live");

        std::array<float, q4::kMoeHidden> host_pointer{};
        require(adapter->enqueue_route(host_pointer.data(), stream.get(), &error) ==
                        q4::moe_adapter_status::invalid_device_pointer,
                "host route pointer did not fail closed");
        std::vector<float> non_finite = input;
        non_finite[17] = std::numeric_limits<float>::quiet_NaN();
        cuda_require(cudaMemcpyAsync(device_input.get(), non_finite.data(),
                                     non_finite.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream.get()),
                     "cudaMemcpyAsync non-finite input");
        require(adapter->enqueue_route(device_input.get(), stream.get(), &error) ==
                        q4::moe_adapter_status::ok,
                "non-finite route did not enqueue for device-side rejection");
        q4::routed_moe_layer_adapter::prepared_generation rejected;
        require(adapter->prepare_route(stream.get(), &rejected, &error) ==
                        q4::moe_adapter_status::non_finite,
                "non-finite route was not rejected before pager transaction");

        const q4::moe_adapter_metrics metrics = adapter->metrics();
        require(metrics.layer == layer && metrics.bounded_slots == q4::kMoeTopK &&
                        !metrics.route_in_flight && !metrics.generation_active &&
                        metrics.pager_active_generations == 0u &&
                        metrics.pager_requests >= 360u &&
                        metrics.pager_nvme_reads >= 120u,
                "bounded pager/transaction metrics are inconsistent");

        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                load_end - load_begin).count();
        std::printf(
                "qwen4exp-moe-adapter-test: PASS sm=%d layer=%u "
                "router=bf16-512x2560-softmax-f32-top10 route_abs=%.9g "
                "output_abs=%.9g output_rel_rms=%.9g deterministic=2/2 "
                "transaction=commit+rollback slots=%u resident_bytes=%llu "
                "pager_requests=%llu nvme_reads=%llu nvme_bytes=%llu "
                "load_ms=%lld output=%016llx\n",
                properties.major * 10 + properties.minor, layer,
                maximum_route_weight_error,
                accuracy.maximum_absolute, accuracy.relative_rms,
                metrics.bounded_slots,
                static_cast<unsigned long long>(metrics.resident_gpu_bytes),
                static_cast<unsigned long long>(metrics.pager_requests),
                static_cast<unsigned long long>(metrics.pager_nvme_reads),
                static_cast<unsigned long long>(metrics.pager_nvme_bytes),
                static_cast<long long>(load_ms),
                static_cast<unsigned long long>(
                        fnv1a(first.data(), first.size() * sizeof(float))));
        return 0;
    } catch (const std::exception &exception) {
        std::fprintf(stderr, "qwen4exp-moe-adapter-test: FAIL: %s\n",
                     exception.what());
        return 1;
    }
}
