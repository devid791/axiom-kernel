#include "axiom/qwen4exp/gdn_adapter.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
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
    if (!condition) fail(message);
}

void cuda_require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void adapter_require(q4::GdnAdapterStatus actual,
                     q4::GdnAdapterStatus expected,
                     const std::string& operation) {
    if (actual != expected) {
        fail(operation + ": expected " +
             q4::gdn_adapter_status_string(expected) + ", got " +
             q4::gdn_adapter_status_string(actual));
    }
}

void gdn_require(q4::GdnStatus actual,
                 q4::GdnStatus expected,
                 const std::string& operation) {
    if (actual != expected) {
        fail(operation + ": expected " + q4::gdn_status_string(expected) +
             ", got " + q4::gdn_status_string(actual));
    }
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

std::uint16_t float_to_bf16(float value) noexcept {
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

template <typename T>
class DeviceBuffer final {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t elements) { allocate(elements); }
    ~DeviceBuffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0u)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
        pointer_ = std::exchange(other.pointer_, nullptr);
        elements_ = std::exchange(other.elements_, 0u);
        return *this;
    }
    void allocate(std::size_t elements) {
        require(pointer_ == nullptr && elements != 0u,
                "invalid device buffer allocation");
        cuda_require(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                                elements * sizeof(T)),
                     "cudaMalloc");
        elements_ = elements;
    }
    [[nodiscard]] T* get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return elements_ * sizeof(T);
    }

private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

class StreamOwner final {
public:
    StreamOwner() {
        cuda_require(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~StreamOwner() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

struct TensorSpec {
    const char* suffix;
    q4::tensor_dtype dtype;
    std::uint32_t rank;
    std::array<std::uint64_t, 3> shape;
    std::size_t elements;
};

constexpr std::array<TensorSpec, 9> kLayerTensorSpecs{{
    {"A_log", q4::tensor_dtype::bf16, 1u, {48u, 0u, 0u}, 48u},
    {"conv1d.weight", q4::tensor_dtype::bf16, 3u,
     {10240u, 1u, 4u}, 40960u},
    {"dt_bias", q4::tensor_dtype::bf16, 1u, {48u, 0u, 0u}, 48u},
    {"in_proj_a.weight", q4::tensor_dtype::bf16, 2u,
     {48u, 2560u, 0u}, 48u * 2560u},
    {"in_proj_b.weight", q4::tensor_dtype::bf16, 2u,
     {48u, 2560u, 0u}, 48u * 2560u},
    {"in_proj_qkv.weight", q4::tensor_dtype::bf16, 2u,
     {10240u, 2560u, 0u}, 10240u * 2560u},
    {"in_proj_z.weight", q4::tensor_dtype::bf16, 2u,
     {6144u, 2560u, 0u}, 6144u * 2560u},
    {"norm.weight", q4::tensor_dtype::bf16, 1u, {128u, 0u, 0u}, 128u},
    {"out_proj.weight", q4::tensor_dtype::bf16, 2u,
     {2560u, 6144u, 0u}, 2560u * 6144u},
}};

std::string layer_prefix(std::size_t layer) {
    return "model.language_model.layers." + std::to_string(layer) +
           ".linear_attn.";
}

void validate_real_descriptors(const q4::checkpoint_catalog& catalog,
                               std::size_t layer) {
    const std::string prefix = layer_prefix(layer);
    for (const TensorSpec& spec : kLayerTensorSpecs) {
        const std::string name = prefix + spec.suffix;
        const q4::tensor_span* span = catalog.find(name);
        require(span != nullptr, "missing real checkpoint tensor: " + name);
        require(span->dtype == spec.dtype, "dtype mismatch: " + name);
        require(span->rank == spec.rank, "rank mismatch: " + name);
        for (std::uint32_t dimension = 0u; dimension < spec.rank; ++dimension) {
            require(span->shape[dimension] == spec.shape[dimension],
                    "shape mismatch: " + name);
        }
        require(span->bytes == spec.elements * sizeof(std::uint16_t),
                "byte count mismatch: " + name);
    }
}

std::vector<std::uint16_t> read_bf16_tensor(
    const q4::checkpoint_catalog& catalog,
    const std::string& name,
    std::size_t expected_elements) {
    const q4::tensor_span* span = catalog.find(name);
    require(span != nullptr && span->dtype == q4::tensor_dtype::bf16 &&
                span->bytes == expected_elements * sizeof(std::uint16_t),
            "invalid BF16 oracle tensor: " + name);
    std::vector<std::uint16_t> result(expected_elements);
    constexpr std::size_t kReadChunk = 1024u * 1024u;
    std::size_t offset = 0u;
    while (offset < span->bytes) {
        const std::size_t amount =
            std::min(kReadChunk, static_cast<std::size_t>(span->bytes) - offset);
        std::string error;
        require(catalog.read_range(name,
                                   offset,
                                   reinterpret_cast<unsigned char*>(result.data()) + offset,
                                   amount,
                                   &error),
                error.empty() ? "oracle range read failed: " + name : error);
        offset += amount;
    }
    return result;
}

struct HostWeights {
    std::vector<std::uint16_t> qkv;
    std::vector<std::uint16_t> z;
    std::vector<std::uint16_t> a;
    std::vector<std::uint16_t> b;
    std::vector<std::uint16_t> out;
    std::vector<float> conv;
    std::vector<float> A_log;
    std::vector<float> dt_bias;
    std::vector<float> norm;

    [[nodiscard]] q4::GdnWeightsView view() const noexcept {
        return q4::GdnWeightsView{
            conv.data(), nullptr, A_log.data(), dt_bias.data(), norm.data()};
    }
};

std::vector<float> decode_bf16(const std::vector<std::uint16_t>& encoded) {
    std::vector<float> decoded(encoded.size());
    std::transform(encoded.begin(), encoded.end(), decoded.begin(), bf16_to_float);
    return decoded;
}

HostWeights load_host_weights(const q4::checkpoint_catalog& catalog,
                              std::size_t layer) {
    const std::string prefix = layer_prefix(layer);
    HostWeights weights{};
    weights.qkv = read_bf16_tensor(
        catalog, prefix + "in_proj_qkv.weight", 10240u * 2560u);
    weights.z = read_bf16_tensor(
        catalog, prefix + "in_proj_z.weight", 6144u * 2560u);
    weights.a = read_bf16_tensor(
        catalog, prefix + "in_proj_a.weight", 48u * 2560u);
    weights.b = read_bf16_tensor(
        catalog, prefix + "in_proj_b.weight", 48u * 2560u);
    weights.out = read_bf16_tensor(
        catalog, prefix + "out_proj.weight", 2560u * 6144u);
    weights.conv = decode_bf16(read_bf16_tensor(
        catalog, prefix + "conv1d.weight", 10240u * 4u));
    weights.A_log = decode_bf16(read_bf16_tensor(
        catalog, prefix + "A_log", 48u));
    weights.dt_bias = decode_bf16(read_bf16_tensor(
        catalog, prefix + "dt_bias", 48u));
    weights.norm = decode_bf16(read_bf16_tensor(
        catalog, prefix + "norm.weight", 128u));
    return weights;
}

/* Independent scalar projection oracle.  It deliberately does not call
 * bf16_linear_reference_f32: input rounding, BF16 decode, and accumulation
 * are implemented locally, while matching the adapter's BF16-input/F32-
 * accumulation contract. */
void project_reference(const std::vector<std::uint16_t>& weight,
                       std::size_t input_features,
                       std::size_t output_features,
                       const std::vector<float>& input,
                       std::size_t batch,
                       std::vector<float>* output) {
    require(output != nullptr &&
                weight.size() == input_features * output_features &&
                input.size() == batch * input_features,
            "invalid independent projection oracle inputs");
    output->assign(batch * output_features, 0.0F);
    std::vector<float> rounded(input_features);
    for (std::size_t token = 0u; token < batch; ++token) {
        for (std::size_t column = 0u; column < input_features; ++column) {
            rounded[column] = bf16_to_float(float_to_bf16(
                input[token * input_features + column]));
        }
        for (std::size_t row = 0u; row < output_features; ++row) {
            const std::uint16_t* row_weight =
                weight.data() + row * input_features;
            float sum = 0.0F;
            for (std::size_t column = 0u; column < input_features; ++column) {
                sum = std::fma(rounded[column],
                               bf16_to_float(row_weight[column]),
                               sum);
            }
            (*output)[token * output_features + row] = sum;
        }
    }
}

struct HostState {
    q4::GdnConfig config{};
    q4::GdnFootprint footprint{};
    std::vector<float> conv;
    std::vector<float> recurrent;
    std::uint32_t cursor = 0u;
    std::uint64_t tokens = 0u;

    explicit HostState(std::size_t batch)
        : config(q4::gdn_qwen4_exp_config(batch)) {
        gdn_require(q4::gdn_footprint(config, &footprint),
                    q4::GdnStatus::kOk,
                    "host state footprint");
        conv.resize(footprint.conv_state_floats);
        recurrent.resize(footprint.recurrent_state_floats);
        reset();
    }

    void reset() {
        q4::GdnHostStateView state = view();
        gdn_require(q4::gdn_host_state_reset(config, state),
                    q4::GdnStatus::kOk,
                    "host state reset");
    }

    [[nodiscard]] q4::GdnHostStateView view() noexcept {
        return q4::GdnHostStateView{
            conv.data(), conv.size(), recurrent.data(), recurrent.size(),
            &cursor, &tokens};
    }
};

struct HostStep {
    std::vector<float> qkv;
    std::vector<float> z;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> convolved;
    std::vector<float> gdn_output;
    std::vector<float> output;
};

HostStep host_step(const HostWeights& weights,
                   const std::vector<float>& input,
                   std::size_t batch,
                   HostState* state) {
    require(state != nullptr && state->config.batch == batch,
            "host step state mismatch");
    HostStep result{};
    project_reference(weights.qkv, 2560u, 10240u,
                      input, batch, &result.qkv);
    project_reference(weights.z, 2560u, 6144u,
                      input, batch, &result.z);
    project_reference(weights.b, 2560u, 48u,
                      input, batch, &result.b);
    project_reference(weights.a, 2560u, 48u,
                      input, batch, &result.a);
    result.convolved.resize(batch * 10240u);
    result.gdn_output.resize(batch * 6144u);
    const q4::GdnStepInputs inputs{
        result.qkv.data(), result.z.data(), result.a.data(), result.b.data()};
    const q4::GdnStepOutputs outputs{
        result.convolved.data(), result.gdn_output.data()};
    q4::GdnHostStateView host_state = state->view();
    gdn_require(q4::gdn_step_host(state->config,
                                  inputs,
                                  weights.view(),
                                  host_state,
                                  outputs),
                q4::GdnStatus::kOk,
                "real host GDN step");
    project_reference(weights.out, 6144u, 2560u,
                      result.gdn_output, batch, &result.output);
    return result;
}

struct Comparison {
    float max_absolute = 0.0F;
    float max_relative = 0.0F;
};

Comparison compare(const std::vector<float>& actual,
                   const std::vector<float>& expected,
                   float absolute_tolerance,
                   float relative_tolerance,
                   const std::string& label) {
    require(actual.size() == expected.size(), label + " size mismatch");
    Comparison result{};
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), label + " produced non-finite value");
        const float absolute = std::fabs(actual[index] - expected[index]);
        const float relative = absolute /
            std::max(1.0e-6F, std::fabs(expected[index]));
        result.max_absolute = std::max(result.max_absolute, absolute);
        result.max_relative = std::max(result.max_relative, relative);
        const float allowed = absolute_tolerance +
            relative_tolerance * std::fabs(expected[index]);
        if (absolute > allowed) {
            fail(label + " exceeds tolerance at index " +
                 std::to_string(index) + ": actual=" +
                 std::to_string(actual[index]) + " expected=" +
                 std::to_string(expected[index]) + " allowed=" +
                 std::to_string(allowed));
        }
    }
    return result;
}

struct DeviceScratch {
    DeviceBuffer<std::uint16_t> linear_input;
    DeviceBuffer<unsigned char> blas;
    DeviceBuffer<float> qkv;
    DeviceBuffer<float> z;
    DeviceBuffer<float> a;
    DeviceBuffer<float> b;
    DeviceBuffer<float> convolved;
    DeviceBuffer<float> gdn_output;
    q4::GdnAdapterScratch view{};

    explicit DeviceScratch(const q4::GdnAdapterWorkspaceRequirements& required)
        : linear_input(required.linear_input_bf16_bytes /
                       sizeof(std::uint16_t)),
          blas(required.blas_workspace_bytes),
          qkv(required.projected_qkv_f32_bytes / sizeof(float)),
          z(required.z_f32_bytes / sizeof(float)),
          a(required.a_f32_bytes / sizeof(float)),
          b(required.b_f32_bytes / sizeof(float)),
          convolved(required.convolved_qkv_f32_bytes / sizeof(float)),
          gdn_output(required.gdn_output_f32_bytes / sizeof(float)) {
        view = q4::GdnAdapterScratch{
            linear_input.get(), linear_input.bytes(),
            blas.get(), blas.bytes(),
            qkv.get(), qkv.bytes(),
            z.get(), z.bytes(),
            a.get(), a.bytes(),
            b.get(), b.bytes(),
            convolved.get(), convolved.bytes(),
            gdn_output.get(), gdn_output.bytes(),
        };
    }
};

std::vector<float> make_input(std::size_t batch, std::size_t step) {
    std::vector<float> input(batch * 2560u);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        const float phase = static_cast<float>(index + 17u * step + 1u);
        input[index] = 0.03125F * std::sin(phase * 0.013F) +
                       0.015625F * std::cos(phase * 0.029F);
    }
    return input;
}

struct DeviceStepResult {
    std::vector<float> output;
    std::vector<float> qkv;
    std::vector<float> z;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> convolved;
    std::vector<float> gdn_output;
};

DeviceStepResult device_step(q4::GdnLayerAdapter* adapter,
                             const std::vector<float>& input,
                             std::size_t batch,
                             q4::GdnDeviceState* state,
                             DeviceScratch* scratch,
                             DeviceBuffer<float>* device_input,
                             DeviceBuffer<float>* device_output,
                             cudaStream_t stream,
                             bool contiguous_sequence = false) {
    require(adapter != nullptr && state != nullptr && scratch != nullptr &&
                device_input != nullptr && device_output != nullptr,
            "invalid device step arguments");
    cuda_require(cudaMemcpyAsync(device_input->get(),
                                 input.data(),
                                 input.size() * sizeof(float),
                                 cudaMemcpyHostToDevice,
                                 stream),
                 "cudaMemcpyAsync adapter input");
    const q4::GdnAdapterStatus forward_status = contiguous_sequence
        ? adapter->forward_sequence(device_input->get(),
                                    batch,
                                    state,
                                    scratch->view,
                                    device_output->get(),
                                    stream)
        : adapter->forward(device_input->get(),
                           batch,
                           state,
                           scratch->view,
                           device_output->get(),
                           stream);
    adapter_require(forward_status,
                    q4::GdnAdapterStatus::kOk,
                    contiguous_sequence
                        ? "GDN adapter forward_sequence"
                        : "GDN adapter forward");

    DeviceStepResult result{};
    result.output.resize(batch * 2560u);
    result.qkv.resize(batch * 10240u);
    result.z.resize(batch * 6144u);
    result.a.resize(batch * 48u);
    result.b.resize(batch * 48u);
    result.convolved.resize(batch * 10240u);
    result.gdn_output.resize(batch * 6144u);
    const auto copy = [&](void* destination,
                          const void* source,
                          std::size_t bytes,
                          const char* operation) {
        cuda_require(cudaMemcpyAsync(destination, source, bytes,
                                     cudaMemcpyDeviceToHost, stream),
                     operation);
    };
    copy(result.output.data(), device_output->get(),
         result.output.size() * sizeof(float), "copy output");
    copy(result.qkv.data(), scratch->view.projected_qkv,
         result.qkv.size() * sizeof(float), "copy qkv");
    copy(result.z.data(), scratch->view.z,
         result.z.size() * sizeof(float), "copy z");
    copy(result.a.data(), scratch->view.a,
         result.a.size() * sizeof(float), "copy a");
    copy(result.b.data(), scratch->view.b,
         result.b.size() * sizeof(float), "copy b");
    copy(result.convolved.data(), scratch->view.convolved_qkv,
         result.convolved.size() * sizeof(float), "copy convolved qkv");
    copy(result.gdn_output.data(), scratch->view.gdn_output,
         result.gdn_output.size() * sizeof(float), "copy GDN output");
    cuda_require(cudaStreamSynchronize(stream), "synchronize GDN adapter step");
    return result;
}

void compare_step(const DeviceStepResult& device,
                  const HostStep& host,
                  float* maximum_output_error) {
    const Comparison qkv = compare(device.qkv, host.qkv,
                                   2.0e-4F, 3.0e-3F, "qkv projection");
    const Comparison z = compare(device.z, host.z,
                                 2.0e-4F, 3.0e-3F, "z projection");
    const Comparison a = compare(device.a, host.a,
                                 2.0e-4F, 3.0e-3F, "a projection");
    const Comparison b = compare(device.b, host.b,
                                 2.0e-4F, 3.0e-3F, "b projection");
    const Comparison convolved = compare(device.convolved, host.convolved,
                                         5.0e-4F, 5.0e-3F,
                                         "causal convolution");
    const Comparison gdn = compare(device.gdn_output, host.gdn_output,
                                   2.0e-3F, 1.0e-2F,
                                   "recurrent GDN output");
    const Comparison output = compare(device.output, host.output,
                                      3.0e-3F, 1.0e-2F,
                                      "adapter output");
    require(qkv.max_absolute < 1.0F && z.max_absolute < 1.0F &&
                a.max_absolute < 1.0F && b.max_absolute < 1.0F &&
                convolved.max_absolute < 1.0F &&
                gdn.max_absolute < 1.0F,
            "intermediate comparison sanity gate failed");
    *maximum_output_error = std::max(*maximum_output_error,
                                     output.max_absolute);
}

void verify_small_device_weights(const q4::GdnLayerAdapter& adapter,
                                 const HostWeights& host) {
    const q4::GdnWeightsView device = adapter.device_weights();
    require(device.conv_weight != nullptr && device.conv_bias == nullptr &&
                device.A_log != nullptr && device.dt_bias != nullptr &&
                device.norm_weight != nullptr,
            "adapter device weight view is incomplete");
    auto exact_copy = [](const float* source,
                         const std::vector<float>& expected,
                         const char* label) {
        std::vector<float> actual(expected.size());
        cuda_require(cudaMemcpy(actual.data(), source,
                                actual.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     label);
        require(actual.size() == expected.size() &&
                    std::memcmp(actual.data(), expected.data(),
                                actual.size() * sizeof(float)) == 0,
                std::string(label) + " differs after BF16->F32 conversion");
    };
    exact_copy(device.conv_weight, host.conv, "copy real conv weight");
    exact_copy(device.A_log, host.A_log, "copy real A_log");
    exact_copy(device.dt_bias, host.dt_bias, "copy real dt_bias");
    exact_copy(device.norm_weight, host.norm, "copy real norm weight");
}

struct TestResult {
    int sm = 0;
    std::size_t resident_weight_bytes = 0u;
    std::size_t scratch_bytes = 0u;
    float maximum_output_error = 0.0F;
    std::size_t sequence_steps = 0u;
    std::size_t tested_batch = 0u;
};

TestResult run_real_gate(const std::string& model_root) {
    int device = -1;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_require(cudaGetDeviceProperties(&properties, device),
                 "cudaGetDeviceProperties");
    require(properties.major == 12 && properties.minor == 0,
            "GDN adapter test requires strict sm_120");

    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    require(q4::checkpoint_catalog::open(model_root, &catalog, &error),
            error.empty() ? "checkpoint catalog open failed" : error);
    require(catalog != nullptr, "checkpoint catalog is null");
    validate_real_descriptors(*catalog, 0u);

    for (std::size_t layer = 0u; layer < 48u; ++layer) {
        q4::GdnAdapterConfig probe{};
        probe.layer_index = layer;
        probe.max_batch = 1u;
        const q4::GdnAdapterStatus expected = (layer % 4u) == 3u
            ? q4::GdnAdapterStatus::kUnsupportedLayer
            : q4::GdnAdapterStatus::kOk;
        adapter_require(q4::gdn_adapter_validate_config(probe),
                        expected,
                        "48-layer schedule admission");
    }
    q4::GdnAdapterConfig invalid{};
    invalid.layer_index = 48u;
    adapter_require(q4::gdn_adapter_validate_config(invalid),
                    q4::GdnAdapterStatus::kInvalidArgument,
                    "out-of-range layer admission");

    StreamOwner stream;
    q4::GdnAdapterConfig config{};
    config.layer_index = 0u;
    config.max_batch = q4::kGdnMaxSequenceTokens;
    config.upload_chunk_bytes = 256u * 1024u;
    config.blas_workspace_bytes = 4u * 1024u * 1024u;

    std::unique_ptr<q4::GdnLayerAdapter> rejected;
    q4::GdnAdapterConfig qsa_config = config;
    qsa_config.layer_index = 3u;
    adapter_require(q4::GdnLayerAdapter::load(*catalog,
                                              qsa_config,
                                              stream.get(),
                                              &rejected,
                                              &error),
                    q4::GdnAdapterStatus::kUnsupportedLayer,
                    "QSA layer load rejection");
    require(!rejected, "QSA layer unexpectedly created a GDN adapter");

    std::unique_ptr<q4::GdnLayerAdapter> adapter;
    adapter_require(q4::GdnLayerAdapter::load(*catalog,
                                              config,
                                              stream.get(),
                                              &adapter,
                                              &error),
                    q4::GdnAdapterStatus::kOk,
                    error.empty() ? "real layer-0 adapter load" : error);
    require(adapter && adapter->initialized() && adapter->layer_index() == 0u &&
                adapter->device() == device,
            "loaded adapter metadata mismatch");

    const HostWeights weights = load_host_weights(*catalog, 0u);
    verify_small_device_weights(*adapter, weights);
    q4::GdnAdapterWorkspaceRequirements requirements{};
    adapter_require(q4::gdn_adapter_workspace_requirements(config, &requirements),
                    q4::GdnAdapterStatus::kOk,
                    "adapter workspace derivation");
    require(requirements.linear_input_bf16_bytes == 4u * 6144u * 2u &&
                requirements.projected_qkv_f32_bytes == 4u * 10240u * 4u &&
                requirements.z_f32_bytes == 4u * 6144u * 4u &&
                requirements.a_f32_bytes == 4u * 48u * 4u &&
                requirements.gdn_output_f32_bytes == 4u * 6144u * 4u,
            "adapter workspace exact-size contract mismatch");

    DeviceScratch scratch(requirements);
    DeviceBuffer<float> device_input(4u * 2560u);
    DeviceBuffer<float> device_output(4u * 2560u);
    q4::GdnDeviceState state{};
    gdn_require(q4::gdn_device_state_init(
                    &state, q4::gdn_qwen4_exp_config(1u), stream.get()),
                q4::GdnStatus::kOk,
                "batch-one device state init");
    cuda_require(cudaStreamSynchronize(stream.get()), "state init synchronize");
    HostState host_state(1u);
    float maximum_output_error = 0.0F;
    std::vector<float> first_device_output;

    for (std::size_t step = 0u; step < 2u; ++step) {
        const std::vector<float> input = make_input(1u, step);
        const HostStep host = host_step(weights, input, 1u, &host_state);
        const DeviceStepResult actual = device_step(adapter.get(),
                                                    input,
                                                    1u,
                                                    &state,
                                                    &scratch,
                                                    &device_input,
                                                    &device_output,
                                                    stream.get());
        compare_step(actual, host, &maximum_output_error);
        if (step == 0u) first_device_output = actual.output;
        require(state.tokens_seen == step + 1u &&
                    host_state.tokens == step + 1u,
                "sequential token counter mismatch");

        std::vector<float> device_conv(host_state.conv.size());
        std::vector<float> device_recurrent(host_state.recurrent.size());
        cuda_require(cudaMemcpy(device_conv.data(), state.conv_state,
                                device_conv.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy convolution state");
        cuda_require(cudaMemcpy(device_recurrent.data(), state.recurrent_state,
                                device_recurrent.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "copy recurrent state");
        (void)compare(device_conv, host_state.conv,
                      5.0e-4F, 5.0e-3F, "convolution state");
        (void)compare(device_recurrent, host_state.recurrent,
                      2.0e-3F, 1.0e-2F, "recurrent state");
    }

    std::size_t free_before = 0u;
    std::size_t total_before = 0u;
    cuda_require(cudaMemGetInfo(&free_before, &total_before),
                 "cudaMemGetInfo before allocation-free replay");
    gdn_require(q4::gdn_device_state_reset(&state, stream.get()),
                q4::GdnStatus::kOk,
                "device state reset");
    host_state.reset();
    const std::vector<float> reset_input = make_input(1u, 0u);
    const DeviceStepResult replay = device_step(adapter.get(),
                                                reset_input,
                                                1u,
                                                &state,
                                                &scratch,
                                                &device_input,
                                                &device_output,
                                                stream.get());
    require(replay.output.size() == first_device_output.size() &&
                std::memcmp(replay.output.data(), first_device_output.data(),
                            replay.output.size() * sizeof(float)) == 0,
            "reset replay is not bit-identical");
    std::size_t free_after = 0u;
    std::size_t total_after = 0u;
    cuda_require(cudaMemGetInfo(&free_after, &total_after),
                 "cudaMemGetInfo after allocation-free replay");
    require(total_before == total_after && free_before == free_after,
            "forward/reset replay changed CUDA allocation footprint");

    q4::GdnAdapterScratch short_scratch = scratch.view;
    short_scratch.z_f32_bytes = sizeof(float);
    adapter_require(adapter->forward(device_input.get(),
                                     1u,
                                     &state,
                                     short_scratch,
                                     device_output.get(),
                                     stream.get()),
                    q4::GdnAdapterStatus::kInvalidDevicePointer,
                    "undersized scratch rejection");
    adapter_require(adapter->forward(reset_input.data(),
                                     1u,
                                     &state,
                                     scratch.view,
                                     device_output.get(),
                                     stream.get()),
                    q4::GdnAdapterStatus::kInvalidDevicePointer,
                    "host input pointer rejection");

    q4::GdnDeviceState batch_state{};
    gdn_require(q4::gdn_device_state_init(
                    &batch_state, q4::gdn_qwen4_exp_config(2u), stream.get()),
                q4::GdnStatus::kOk,
                "batch-two device state init");
    cuda_require(cudaStreamSynchronize(stream.get()),
                 "batch state init synchronize");
    adapter_require(adapter->forward(device_input.get(),
                                     1u,
                                     &batch_state,
                                     scratch.view,
                                     device_output.get(),
                                     stream.get()),
                    q4::GdnAdapterStatus::kStateMismatch,
                    "batch/state mismatch rejection");
    adapter_require(adapter->forward_sequence(device_input.get(),
                                              2u,
                                              &batch_state,
                                              scratch.view,
                                              device_output.get(),
                                              stream.get()),
                    q4::GdnAdapterStatus::kStateMismatch,
                    "sequence requires batch-one state");

    HostState batch_host_state(2u);
    const std::vector<float> batch_input = make_input(2u, 5u);
    const HostStep batch_host = host_step(weights,
                                          batch_input,
                                          2u,
                                          &batch_host_state);
    const DeviceStepResult batch_actual = device_step(adapter.get(),
                                                      batch_input,
                                                      2u,
                                                      &batch_state,
                                                      &scratch,
                                                      &device_input,
                                                      &device_output,
                                                      stream.get());
    compare_step(batch_actual, batch_host, &maximum_output_error);
    require(batch_state.tokens_seen == 1u && batch_host_state.tokens == 1u,
            "batch-two token counter mismatch");

    // Compare one four-row projection plus causal block execution against four
    // independent adapter calls over exactly the same contiguous input rows.
    constexpr std::size_t kSequenceTokens = q4::kGdnMaxSequenceTokens;
    const std::vector<float> sequence_input = make_input(kSequenceTokens, 19u);
    HostStep repeated_sequence{};
    const auto append = [](std::vector<float>* destination,
                           const std::vector<float>& source) {
        destination->insert(destination->end(), source.begin(), source.end());
    };
    gdn_require(q4::gdn_device_state_reset(&state, stream.get()),
                q4::GdnStatus::kOk,
                "repeated sequence state reset");
    cuda_require(cudaStreamSynchronize(stream.get()),
                 "repeated sequence reset synchronize");
    for (std::size_t token = 0u; token < kSequenceTokens; ++token) {
        const auto begin = sequence_input.begin() +
            static_cast<std::ptrdiff_t>(token * 2560u);
        const std::vector<float> token_input(begin, begin + 2560u);
        const DeviceStepResult token_result = device_step(adapter.get(),
                                                          token_input,
                                                          1u,
                                                          &state,
                                                          &scratch,
                                                          &device_input,
                                                          &device_output,
                                                          stream.get());
        append(&repeated_sequence.output, token_result.output);
        append(&repeated_sequence.qkv, token_result.qkv);
        append(&repeated_sequence.z, token_result.z);
        append(&repeated_sequence.a, token_result.a);
        append(&repeated_sequence.b, token_result.b);
        append(&repeated_sequence.convolved, token_result.convolved);
        append(&repeated_sequence.gdn_output, token_result.gdn_output);
    }
    require(state.tokens_seen == kSequenceTokens && state.conv_cursor == 0u,
            "repeated sequence metadata mismatch");
    std::vector<float> repeated_sequence_conv(state.conv_state_floats);
    std::vector<float> repeated_sequence_recurrent(
        state.recurrent_state_floats);
    cuda_require(cudaMemcpy(repeated_sequence_conv.data(),
                            state.conv_state,
                            repeated_sequence_conv.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy repeated adapter convolution state");
    cuda_require(cudaMemcpy(repeated_sequence_recurrent.data(),
                            state.recurrent_state,
                            repeated_sequence_recurrent.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy repeated adapter recurrent state");

    gdn_require(q4::gdn_device_state_reset(&state, stream.get()),
                q4::GdnStatus::kOk,
                "block sequence state reset");
    cuda_require(cudaStreamSynchronize(stream.get()),
                 "block sequence reset synchronize");
    const DeviceStepResult block_sequence = device_step(adapter.get(),
                                                        sequence_input,
                                                        kSequenceTokens,
                                                        &state,
                                                        &scratch,
                                                        &device_input,
                                                        &device_output,
                                                        stream.get(),
                                                        true);
    compare_step(block_sequence, repeated_sequence, &maximum_output_error);
    require(state.tokens_seen == kSequenceTokens && state.conv_cursor == 0u,
            "block sequence metadata mismatch");
    std::vector<float> block_sequence_conv(state.conv_state_floats);
    std::vector<float> block_sequence_recurrent(state.recurrent_state_floats);
    cuda_require(cudaMemcpy(block_sequence_conv.data(),
                            state.conv_state,
                            block_sequence_conv.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy block adapter convolution state");
    cuda_require(cudaMemcpy(block_sequence_recurrent.data(),
                            state.recurrent_state,
                            block_sequence_recurrent.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy block adapter recurrent state");
    (void)compare(block_sequence_conv,
                  repeated_sequence_conv,
                  5.0e-4F,
                  5.0e-3F,
                  "block adapter convolution state");
    (void)compare(block_sequence_recurrent,
                  repeated_sequence_recurrent,
                  2.0e-3F,
                  1.0e-2F,
                  "block adapter recurrent state");

    // Warmed sequence replay must not alter the CUDA allocation footprint.
    gdn_require(q4::gdn_device_state_reset(&state, stream.get()),
                q4::GdnStatus::kOk,
                "allocation-free sequence reset");
    cuda_require(cudaStreamSynchronize(stream.get()),
                 "allocation-free sequence reset synchronize");
    std::size_t sequence_free_before = 0u;
    std::size_t sequence_total_before = 0u;
    cuda_require(cudaMemGetInfo(&sequence_free_before,
                                &sequence_total_before),
                 "cudaMemGetInfo before sequence replay");
    (void)device_step(adapter.get(),
                      sequence_input,
                      kSequenceTokens,
                      &state,
                      &scratch,
                      &device_input,
                      &device_output,
                      stream.get(),
                      true);
    std::size_t sequence_free_after = 0u;
    std::size_t sequence_total_after = 0u;
    cuda_require(cudaMemGetInfo(&sequence_free_after,
                                &sequence_total_after),
                 "cudaMemGetInfo after sequence replay");
    require(sequence_total_before == sequence_total_after &&
                sequence_free_before == sequence_free_after,
            "forward_sequence changed CUDA allocation footprint");

    gdn_require(q4::gdn_device_state_reset(&state, stream.get()),
                q4::GdnStatus::kOk,
                "sequence failure reset");
    cuda_require(cudaStreamSynchronize(stream.get()),
                 "sequence failure reset synchronize");
    adapter_require(adapter->forward_sequence(device_input.get(),
                                              0u,
                                              &state,
                                              scratch.view,
                                              device_output.get(),
                                              stream.get()),
                    q4::GdnAdapterStatus::kInvalidArgument,
                    "zero-width adapter sequence rejection");
    adapter_require(adapter->forward_sequence(device_input.get(),
                                              kSequenceTokens + 1u,
                                              &state,
                                              scratch.view,
                                              device_output.get(),
                                              stream.get()),
                    q4::GdnAdapterStatus::kInvalidArgument,
                    "oversized adapter sequence rejection");
    adapter_require(adapter->forward_sequence(device_input.get(),
                                              kSequenceTokens,
                                              &state,
                                              short_scratch,
                                              device_output.get(),
                                              stream.get()),
                    q4::GdnAdapterStatus::kInvalidDevicePointer,
                    "short sequence scratch rejection");
    adapter_require(adapter->forward_sequence(sequence_input.data(),
                                              kSequenceTokens,
                                              &state,
                                              scratch.view,
                                              device_output.get(),
                                              stream.get()),
                    q4::GdnAdapterStatus::kInvalidDevicePointer,
                    "host sequence pointer rejection");
    require(state.tokens_seen == 0u && state.conv_cursor == 0u,
            "invalid adapter sequence mutated state metadata");

    gdn_require(q4::gdn_device_state_release(&batch_state),
                q4::GdnStatus::kOk,
                "batch-two state release");
    gdn_require(q4::gdn_device_state_release(&state),
                q4::GdnStatus::kOk,
                "batch-one state release");

    std::size_t resident_weight_bytes = 0u;
    for (const TensorSpec& spec : kLayerTensorSpecs) {
        resident_weight_bytes += spec.elements *
            (std::string(spec.suffix) == "A_log" ||
             std::string(spec.suffix) == "conv1d.weight" ||
             std::string(spec.suffix) == "dt_bias" ||
             std::string(spec.suffix) == "norm.weight"
                 ? sizeof(float)
                 : sizeof(std::uint16_t));
    }
    return TestResult{
        properties.major * 10 + properties.minor,
        resident_weight_bytes,
        requirements.total_device_bytes,
        maximum_output_error,
        kSequenceTokens,
        2u,
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: gdn-adapter-test MODEL_ROOT");
        const TestResult result = run_real_gate(argv[1]);
        std::cout << "gdn_adapter_test: OK sm=" << result.sm
                  << " layer=0 sequence_steps=" << result.sequence_steps
                  << " batch=" << result.tested_batch
                  << " resident_weight_bytes=" << result.resident_weight_bytes
                  << " scratch_bytes=" << result.scratch_bytes
                  << " max_output_abs=" << result.maximum_output_error
                  << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "gdn_adapter_test: FAIL: " << exception.what() << '\n';
        return 1;
    }
}
