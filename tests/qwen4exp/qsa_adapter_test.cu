#include "axiom/qwen4exp/qsa_adapter.hpp"

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

constexpr std::size_t kTokens = 2u;
constexpr std::size_t kCapacity = 8u;
constexpr float kProjectionTolerance = 2.5e-3F;
constexpr float kOutputTolerance = 1.5e-2F;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const std::string& where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::QsaAdapterStatus actual,
                    q4::QsaAdapterStatus expected,
                    const std::string& where) {
    if (actual != expected) {
        throw std::runtime_error(
            where + ": expected " + q4::qsa_adapter_status_string(expected) +
            ", got " + q4::qsa_adapter_status_string(actual));
    }
}

template <typename T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t elements) { reset(elements); }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;
    device_buffer(device_buffer&& other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0u)) {}
    device_buffer& operator=(device_buffer&& other) noexcept {
        if (this != &other) {
            if (pointer_ != nullptr) (void)cudaFree(pointer_);
            pointer_ = std::exchange(other.pointer_, nullptr);
            elements_ = std::exchange(other.elements_, 0u);
        }
        return *this;
    }

    void reset(std::size_t elements) {
        require(elements != 0u, "zero-sized device allocation");
        if (pointer_ != nullptr) require_cuda(cudaFree(pointer_), "cudaFree");
        pointer_ = nullptr;
        elements_ = 0u;
        require_cuda(
            cudaMalloc(reinterpret_cast<void**>(&pointer_), elements * sizeof(T)),
            "cudaMalloc");
        elements_ = elements;
    }

    T* get() noexcept { return pointer_; }
    const T* get() const noexcept { return pointer_; }
    std::size_t elements() const noexcept { return elements_; }
    std::size_t bytes() const noexcept { return elements_ * sizeof(T); }

private:
    T* pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<std::uint16_t>((bits + rounding) >> 16u);
}

float bf16_to_float(std::uint16_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<std::uint16_t> read_bf16_tensor(
    const q4::checkpoint_catalog& catalog,
    const std::string& name,
    std::uint64_t rows,
    std::uint64_t columns = 0u) {
    const q4::tensor_span* span = catalog.find(name);
    require(span != nullptr, "missing real tensor: " + name);
    require(span->dtype == q4::tensor_dtype::bf16,
            "real tensor is not BF16: " + name);
    const bool vector = columns == 0u;
    require(span->rank == (vector ? 1u : 2u),
            "real tensor rank mismatch: " + name);
    require(span->shape[0] == rows && (vector || span->shape[1] == columns),
            "real tensor shape mismatch: " + name);
    const std::uint64_t elements = rows * (vector ? 1u : columns);
    require(elements <= std::numeric_limits<std::size_t>::max(),
            "tensor element count overflow: " + name);
    std::vector<std::uint16_t> result(static_cast<std::size_t>(elements));
    std::string error;
    require(catalog.read_range(name, 0u, result.data(), result.size() * 2u, &error),
            error.empty() ? "tensor read failed: " + name : error);
    return result;
}

std::vector<float> independent_linear(
    const std::vector<std::uint16_t>& weights,
    std::size_t rows,
    std::size_t columns,
    const float* input) {
    require(weights.size() == rows * columns, "independent linear shape mismatch");
    std::vector<float> rounded_input(columns);
    for (std::size_t column = 0u; column < columns; ++column) {
        rounded_input[column] = bf16_to_float(float_to_bf16(input[column]));
    }
    std::vector<float> output(rows, 0.0F);
    for (std::size_t row = 0u; row < rows; ++row) {
        float sum = 0.0F;
        const std::size_t offset = row * columns;
        for (std::size_t column = 0u; column < columns; ++column) {
            sum = std::fma(
                rounded_input[column],
                bf16_to_float(weights[offset + column]),
                sum);
        }
        output[row] = sum;
    }
    return output;
}

float maximum_error(const std::vector<float>& actual,
                    const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "error vectors differ in size");
    float maximum = 0.0F;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum;
}

std::pair<std::vector<float>, std::vector<float>> make_rope(std::size_t positions) {
    std::vector<float> cosine(positions * q4::attention::kCheckpointRotaryDim);
    std::vector<float> sine(cosine.size());
    constexpr double theta = 10000000.0;
    constexpr std::size_t rotary = q4::attention::kCheckpointRotaryDim;
    constexpr std::size_t half = rotary / 2u;
    for (std::size_t position = 0u; position < positions; ++position) {
        for (std::size_t dim = 0u; dim < half; ++dim) {
            const double frequency =
                std::pow(theta, -2.0 * static_cast<double>(dim) /
                                    static_cast<double>(rotary));
            const float angle = static_cast<float>(
                static_cast<double>(position) * frequency);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            cosine[position * rotary + dim] = c;
            cosine[position * rotary + dim + half] = c;
            sine[position * rotary + dim] = s;
            sine[position * rotary + dim + half] = s;
        }
    }
    return {std::move(cosine), std::move(sine)};
}

struct fixture {
    explicit fixture(const q4::QsaAdapterWorkspaceRequirements& required)
        : linear_input(required.linear_input_bf16_bytes / 2u),
          blas(required.blas_workspace_bytes),
          q_projection(required.q_projection_f32_bytes / 4u),
          k_projection(required.k_projection_f32_bytes / 4u),
          v_projection(required.v_projection_f32_bytes / 4u),
          index_projection(required.index_projection_f32_bytes / 4u),
          q_projection_bf16(required.q_projection_bf16_bytes / 2u),
          k_projection_bf16(required.k_projection_bf16_bytes / 2u),
          v_projection_bf16(required.v_projection_bf16_bytes / 2u),
          index_query_bf16(required.index_query_bf16_bytes / 2u),
          index_query_prepared(required.index_query_prepared_f32_bytes / 4u),
          visible_indices(required.visible_indices_bytes / 4u),
          pooled_index_keys(required.pooled_index_keys_f32_bytes / 4u),
          index_scores(required.index_scores_f32_bytes / 4u),
          selection(required.selection_workspace_bytes),
          selected_indices(required.selected_indices_bytes / 4u),
          selected_counts(required.selected_counts_bytes / 4u),
          attention_workspace(required.attention_workspace_bytes),
          attention_output_bf16(required.attention_output_bf16_bytes / 2u),
          attention_output_f32(required.attention_output_f32_bytes / 4u),
          qsa_status(1u),
          attention_status(1u),
          key_cache(kCapacity * q4::attention::kCheckpointKvHeads *
                    q4::attention::kCheckpointHeadDim),
          value_cache(kCapacity * q4::attention::kCheckpointKvHeads *
                      q4::attention::kCheckpointHeadDim),
          index_key_cache(kCapacity * q4::kQsaAdapterIndexerKeySize),
          input(kTokens * q4::kQsaAdapterHiddenSize),
          invalid_input(q4::kQsaAdapterHiddenSize),
          output(kTokens * q4::kQsaAdapterHiddenSize),
          cosine(3u * q4::attention::kCheckpointRotaryDim),
          sine(3u * q4::attention::kCheckpointRotaryDim) {
        scratch.linear_input_bf16 = linear_input.get();
        scratch.linear_input_bf16_bytes = linear_input.bytes();
        scratch.blas_workspace = blas.get();
        scratch.blas_workspace_bytes = blas.bytes();
        scratch.q_projection = q_projection.get();
        scratch.q_projection_f32_bytes = q_projection.bytes();
        scratch.k_projection = k_projection.get();
        scratch.k_projection_f32_bytes = k_projection.bytes();
        scratch.v_projection = v_projection.get();
        scratch.v_projection_f32_bytes = v_projection.bytes();
        scratch.index_projection = index_projection.get();
        scratch.index_projection_f32_bytes = index_projection.bytes();
        scratch.q_projection_bf16 = q_projection_bf16.get();
        scratch.q_projection_bf16_bytes = q_projection_bf16.bytes();
        scratch.k_projection_bf16 = k_projection_bf16.get();
        scratch.k_projection_bf16_bytes = k_projection_bf16.bytes();
        scratch.v_projection_bf16 = v_projection_bf16.get();
        scratch.v_projection_bf16_bytes = v_projection_bf16.bytes();
        scratch.index_query_bf16 = index_query_bf16.get();
        scratch.index_query_bf16_bytes = index_query_bf16.bytes();
        scratch.index_query_prepared = index_query_prepared.get();
        scratch.index_query_prepared_f32_bytes = index_query_prepared.bytes();
        scratch.visible_indices = visible_indices.get();
        scratch.visible_indices_bytes = visible_indices.bytes();
        scratch.pooled_index_keys = pooled_index_keys.get();
        scratch.pooled_index_keys_f32_bytes = pooled_index_keys.bytes();
        scratch.index_scores = index_scores.get();
        scratch.index_scores_f32_bytes = index_scores.bytes();
        scratch.selection_workspace = selection.get();
        scratch.selection_workspace_bytes = selection.bytes();
        scratch.selected_indices = selected_indices.get();
        scratch.selected_indices_bytes = selected_indices.bytes();
        scratch.selected_counts = selected_counts.get();
        scratch.selected_counts_bytes = selected_counts.bytes();
        scratch.attention_workspace = attention_workspace.get();
        scratch.attention_workspace_bytes = attention_workspace.bytes();
        scratch.attention_output_bf16 = attention_output_bf16.get();
        scratch.attention_output_bf16_bytes = attention_output_bf16.bytes();
        scratch.attention_output_f32 = attention_output_f32.get();
        scratch.attention_output_f32_bytes = attention_output_f32.bytes();
        scratch.qsa_device_status = qsa_status.get();
        scratch.attention_device_status = attention_status.get();
    }

    device_buffer<std::uint16_t> linear_input;
    device_buffer<std::uint8_t> blas;
    device_buffer<float> q_projection;
    device_buffer<float> k_projection;
    device_buffer<float> v_projection;
    device_buffer<float> index_projection;
    device_buffer<std::uint16_t> q_projection_bf16;
    device_buffer<std::uint16_t> k_projection_bf16;
    device_buffer<std::uint16_t> v_projection_bf16;
    device_buffer<std::uint16_t> index_query_bf16;
    device_buffer<float> index_query_prepared;
    device_buffer<std::int32_t> visible_indices;
    device_buffer<float> pooled_index_keys;
    device_buffer<float> index_scores;
    device_buffer<std::uint8_t> selection;
    device_buffer<std::int32_t> selected_indices;
    device_buffer<std::uint32_t> selected_counts;
    device_buffer<std::uint8_t> attention_workspace;
    device_buffer<std::uint16_t> attention_output_bf16;
    device_buffer<float> attention_output_f32;
    device_buffer<q4::qsa::status> qsa_status;
    device_buffer<q4::attention::status> attention_status;
    device_buffer<std::uint16_t> key_cache;
    device_buffer<std::uint16_t> value_cache;
    device_buffer<std::uint16_t> index_key_cache;
    device_buffer<float> input;
    device_buffer<float> invalid_input;
    device_buffer<float> output;
    device_buffer<float> cosine;
    device_buffer<float> sine;
    q4::QsaAdapterScratch scratch{};
    q4::QsaLayerCache cache{};
};

struct run_capture {
    std::vector<float> output;
    std::vector<float> q_projection;
    std::vector<float> k_projection;
    std::vector<float> v_projection;
    std::vector<float> index_projection;
    std::vector<std::uint16_t> attention_output;
    std::array<std::uint32_t, kTokens> selected_counts{};
    std::vector<std::int32_t> selected_indices;
};

run_capture capture_valid_run(q4::QsaLayerAdapter* adapter,
                              fixture* memory,
                              cudaStream_t stream) {
    require_status(adapter->begin_transaction(
                       &memory->cache, memory->scratch, stream),
                   q4::QsaAdapterStatus::kOk,
                   "begin valid transaction");
    require_status(adapter->forward(
                       memory->input.get(),
                       kTokens,
                       memory->cosine.get(),
                       memory->sine.get(),
                       3u,
                       &memory->cache,
                       memory->scratch,
                       memory->output.get(),
                       stream),
                   q4::QsaAdapterStatus::kOk,
                   "valid forward");
    q4::QsaAdapterCollectedStatus collected{};
    require_status(adapter->collect_device_status(
                       &memory->cache, memory->scratch, &collected, stream),
                   q4::QsaAdapterStatus::kOk,
                   "collect valid status");
    require(collected.indexer == q4::qsa::status::ok &&
                collected.attention_core == q4::attention::status::ok,
            "valid run reported a device error");

    run_capture result{};
    result.output.resize(kTokens * q4::kQsaAdapterHiddenSize);
    result.q_projection.resize(kTokens * q4::kQsaAdapterQueryProjectionSize);
    result.k_projection.resize(kTokens * q4::kQsaAdapterKvProjectionSize);
    result.v_projection.resize(kTokens * q4::kQsaAdapterKvProjectionSize);
    result.index_projection.resize(kTokens * q4::kQsaAdapterIndexerProjectionSize);
    result.attention_output.resize(
        kTokens * q4::kQsaAdapterAttentionOutputSize);
    result.selected_indices.assign(
        kTokens * q4::qsa::checkpoint_selected_capacity, -1);
    require_cuda(cudaMemcpy(result.output.data(), memory->output.get(),
                            result.output.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy output");
    require_cuda(cudaMemcpy(result.q_projection.data(), memory->q_projection.get(),
                            result.q_projection.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy q projection");
    require_cuda(cudaMemcpy(result.k_projection.data(), memory->k_projection.get(),
                            result.k_projection.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy k projection");
    require_cuda(cudaMemcpy(result.v_projection.data(), memory->v_projection.get(),
                            result.v_projection.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy v projection");
    require_cuda(cudaMemcpy(result.index_projection.data(),
                            memory->index_projection.get(),
                            result.index_projection.size() * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 "copy index projection");
    require_cuda(cudaMemcpy(result.attention_output.data(),
                            memory->attention_output_bf16.get(),
                            result.attention_output.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost),
                 "copy attention output");
    require_cuda(cudaMemcpy(result.selected_counts.data(),
                            memory->selected_counts.get(),
                            sizeof(result.selected_counts),
                            cudaMemcpyDeviceToHost),
                 "copy selected counts");
    for (std::size_t token = 0u; token < kTokens; ++token) {
        require(result.selected_counts[token] <=
                    q4::qsa::checkpoint_selected_capacity,
                "device selected count exceeds row capacity");
        require_cuda(cudaMemcpy(
                         result.selected_indices.data() +
                             token * q4::qsa::checkpoint_selected_capacity,
                         memory->selected_indices.get() +
                             token * q4::qsa::checkpoint_selected_capacity,
                         result.selected_counts[token] * sizeof(std::int32_t),
                         cudaMemcpyDeviceToHost),
                     "copy initialized selected indices");
    }
    return result;
}

void validate_selected_mask(const run_capture& capture) {
    require(capture.selected_counts[0] == 1u &&
                capture.selected_counts[1] == 2u,
            "short causal QSA counts differ from [1,2]");
    require(capture.selected_indices[0] == 0,
            "first QSA row did not select token zero");
    const std::size_t second = q4::qsa::checkpoint_selected_capacity;
    require(capture.selected_indices[second] == 0 &&
                capture.selected_indices[second + 1u] == 1,
            "second QSA row did not select its complete causal tail");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: qsa-adapter-test MODEL_ROOT");
        int device = -1;
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "QSA adapter gate requires an actual SM120 device");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error), error);

        q4::QsaAdapterConfig config{};
        config.layer_index = 3u;
        config.max_batch = kTokens;
        config.max_context = kCapacity;
        config.upload_chunk_bytes = 8u * 1024u * 1024u;
        require_status(q4::qsa_adapter_validate_config(config),
                       q4::QsaAdapterStatus::kOk,
                       "validate real config");
        q4::QsaAdapterConfig rejected = config;
        rejected.layer_index = 0u;
        require_status(q4::qsa_adapter_validate_config(rejected),
                       q4::QsaAdapterStatus::kUnsupportedLayer,
                       "reject non-QSA layer");

        q4::QsaAdapterWorkspaceRequirements requirements{};
        require_status(q4::qsa_adapter_workspace_requirements(
                           config, &requirements),
                       q4::QsaAdapterStatus::kOk,
                       "workspace requirements");
        require(requirements.total_device_bytes != 0u,
                "workspace contract returned zero bytes");

        cudaStream_t stream = nullptr;
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
        const auto stream_guard = std::unique_ptr<
            std::remove_pointer_t<cudaStream_t>, void (*)(cudaStream_t)>(
            stream,
            [](cudaStream_t value) {
                if (value != nullptr) (void)cudaStreamDestroy(value);
            });

        std::unique_ptr<q4::QsaLayerAdapter> adapter;
        require_status(q4::QsaLayerAdapter::load(
                           *catalog, config, stream, &adapter, &error),
                       q4::QsaAdapterStatus::kOk,
                       error.empty() ? "load real QSA adapter" : error);
        require(adapter && adapter->initialized() && adapter->layer_index() == 3u,
                "real QSA adapter did not initialize");

        fixture memory(requirements);
        require_status(q4::qsa_adapter_cache_initialize(
                           &memory.cache,
                           memory.key_cache.get(),
                           memory.value_cache.get(),
                           memory.index_key_cache.get(),
                           kCapacity),
                       q4::QsaAdapterStatus::kOk,
                       "initialize coherent caches");

        std::vector<float> input(kTokens * q4::kQsaAdapterHiddenSize);
        for (std::size_t token = 0u; token < kTokens; ++token) {
            for (std::size_t dim = 0u; dim < q4::kQsaAdapterHiddenSize; ++dim) {
                input[token * q4::kQsaAdapterHiddenSize + dim] =
                    0.0125F * std::sin(
                        static_cast<float>((token + 1u) * (dim + 3u)) *
                        0.0078125F);
            }
        }
        const auto rope = make_rope(3u);
        require_cuda(cudaMemcpyAsync(memory.input.get(), input.data(),
                                     input.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "upload mixed hidden");
        require_cuda(cudaMemcpyAsync(memory.cosine.get(), rope.first.data(),
                                     rope.first.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "upload cosine");
        require_cuda(cudaMemcpyAsync(memory.sine.get(), rope.second.data(),
                                     rope.second.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "upload sine");
        require_cuda(cudaStreamSynchronize(stream), "synchronize input upload");

        const run_capture first = capture_valid_run(adapter.get(), &memory, stream);
        validate_selected_mask(first);

        const std::string prefix = "model.language_model.layers.3.self_attn.";
        const auto q_weight = read_bf16_tensor(
            *catalog, prefix + "q_proj.weight",
            q4::kQsaAdapterQueryProjectionSize,
            q4::kQsaAdapterHiddenSize);
        const auto k_weight = read_bf16_tensor(
            *catalog, prefix + "k_proj.weight",
            q4::kQsaAdapterKvProjectionSize,
            q4::kQsaAdapterHiddenSize);
        const auto v_weight = read_bf16_tensor(
            *catalog, prefix + "v_proj.weight",
            q4::kQsaAdapterKvProjectionSize,
            q4::kQsaAdapterHiddenSize);
        const auto index_weight = read_bf16_tensor(
            *catalog, prefix + "indexer.index_qk_proj.weight",
            q4::kQsaAdapterIndexerProjectionSize,
            q4::kQsaAdapterHiddenSize);
        const auto o_weight = read_bf16_tensor(
            *catalog, prefix + "o_proj.weight",
            q4::kQsaAdapterHiddenSize,
            q4::kQsaAdapterAttentionOutputSize);
        (void)read_bf16_tensor(
            *catalog, prefix + "q_norm.weight",
            q4::attention::kCheckpointHeadDim);
        (void)read_bf16_tensor(
            *catalog, prefix + "k_norm.weight",
            q4::attention::kCheckpointHeadDim);
        (void)read_bf16_tensor(
            *catalog, prefix + "indexer.q_layernorm.weight",
            q4::qsa::checkpoint_config.head_dim);
        (void)read_bf16_tensor(
            *catalog, prefix + "indexer.k_layernorm.weight",
            q4::qsa::checkpoint_config.head_dim);

        const std::vector<float> oracle_q = independent_linear(
            q_weight,
            q4::kQsaAdapterQueryProjectionSize,
            q4::kQsaAdapterHiddenSize,
            input.data());
        const std::vector<float> oracle_k = independent_linear(
            k_weight,
            q4::kQsaAdapterKvProjectionSize,
            q4::kQsaAdapterHiddenSize,
            input.data());
        const std::vector<float> oracle_v = independent_linear(
            v_weight,
            q4::kQsaAdapterKvProjectionSize,
            q4::kQsaAdapterHiddenSize,
            input.data());
        const std::vector<float> oracle_index = independent_linear(
            index_weight,
            q4::kQsaAdapterIndexerProjectionSize,
            q4::kQsaAdapterHiddenSize,
            input.data());
        const std::vector<float> gpu_q(
            first.q_projection.begin(),
            first.q_projection.begin() + q4::kQsaAdapterQueryProjectionSize);
        const std::vector<float> gpu_k(
            first.k_projection.begin(),
            first.k_projection.begin() + q4::kQsaAdapterKvProjectionSize);
        const std::vector<float> gpu_v(
            first.v_projection.begin(),
            first.v_projection.begin() + q4::kQsaAdapterKvProjectionSize);
        const std::vector<float> gpu_index(
            first.index_projection.begin(),
            first.index_projection.begin() + q4::kQsaAdapterIndexerProjectionSize);
        const float q_error = maximum_error(gpu_q, oracle_q);
        const float k_error = maximum_error(gpu_k, oracle_k);
        const float v_error = maximum_error(gpu_v, oracle_v);
        const float index_error = maximum_error(gpu_index, oracle_index);
        require(q_error <= kProjectionTolerance &&
                    k_error <= kProjectionTolerance &&
                    v_error <= kProjectionTolerance &&
                    index_error <= kProjectionTolerance,
                "real BF16 projection exceeds independent CPU tolerance");

        std::vector<std::uint16_t> oracle_attention(
            q4::kQsaAdapterAttentionOutputSize);
        for (std::size_t head = 0u;
             head < q4::attention::kCheckpointQueryHeads;
             ++head) {
            const std::size_t kv_head =
                head / q4::attention::kCheckpointGqaRatio;
            for (std::size_t dim = 0u;
                 dim < q4::attention::kCheckpointHeadDim;
                 ++dim) {
                const std::size_t gate_row =
                    head * q4::attention::kCheckpointQHeadStride +
                    q4::attention::kCheckpointHeadDim + dim;
                const float gate = bf16_to_float(float_to_bf16(oracle_q[gate_row]));
                const std::size_t value_row =
                    kv_head * q4::attention::kCheckpointHeadDim + dim;
                const float value =
                    bf16_to_float(float_to_bf16(oracle_v[value_row]));
                const float sigmoid = 1.0F / (1.0F + std::exp(-gate));
                oracle_attention[
                    head * q4::attention::kCheckpointHeadDim + dim] =
                    float_to_bf16(value * sigmoid);
            }
        }
        float attention_error = 0.0F;
        for (std::size_t index = 0u; index < oracle_attention.size(); ++index) {
            attention_error = std::max(
                attention_error,
                std::abs(bf16_to_float(first.attention_output[index]) -
                         bf16_to_float(oracle_attention[index])));
        }
        require(attention_error <= kProjectionTolerance,
                "single-token causal attention differs from independent oracle");

        std::vector<float> oracle_attention_f32(oracle_attention.size());
        for (std::size_t index = 0u; index < oracle_attention.size(); ++index) {
            oracle_attention_f32[index] = bf16_to_float(oracle_attention[index]);
        }
        const std::vector<float> oracle_output = independent_linear(
            o_weight,
            q4::kQsaAdapterHiddenSize,
            q4::kQsaAdapterAttentionOutputSize,
            oracle_attention_f32.data());
        const std::vector<float> gpu_output(
            first.output.begin(),
            first.output.begin() + q4::kQsaAdapterHiddenSize);
        const float output_error = maximum_error(gpu_output, oracle_output);
        require(output_error <= kOutputTolerance,
                "real QSA token output differs from independent CPU oracle");

        require_status(adapter->rollback(&memory.cache),
                       q4::QsaAdapterStatus::kOk,
                       "rollback first valid run");
        q4::QsaAdapterCacheLengths lengths{};
        require_status(q4::qsa_adapter_cache_get_lengths(&memory.cache, &lengths),
                       q4::QsaAdapterStatus::kOk,
                       "lengths after rollback");
        require(lengths.committed == 0u && lengths.staged == 0u &&
                    lengths.visible == 0u,
                "rollback left a visible cache suffix");

        const run_capture second = capture_valid_run(adapter.get(), &memory, stream);
        require(first.output.size() == second.output.size() &&
                    std::memcmp(first.output.data(), second.output.data(),
                                first.output.size() * sizeof(float)) == 0,
                "QSA adapter output is not bit-deterministic after rollback");
        require_status(adapter->commit_prefix(&memory.cache, kTokens),
                       q4::QsaAdapterStatus::kOk,
                       "commit deterministic run");
        require_status(q4::qsa_adapter_cache_get_lengths(&memory.cache, &lengths),
                       q4::QsaAdapterStatus::kOk,
                       "lengths after commit");
        require(lengths.committed == kTokens && lengths.staged == 0u &&
                    lengths.visible == kTokens,
                "main KV and indexer cache did not commit together");

        std::vector<float> invalid(q4::kQsaAdapterHiddenSize, 0.0F);
        invalid[17] = std::numeric_limits<float>::quiet_NaN();
        require_cuda(cudaMemcpyAsync(memory.invalid_input.get(), invalid.data(),
                                     invalid.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream),
                     "upload invalid mixed hidden");
        require_cuda(cudaStreamSynchronize(stream), "sync invalid upload");
        require_status(adapter->begin_transaction(
                           &memory.cache, memory.scratch, stream),
                       q4::QsaAdapterStatus::kOk,
                       "begin rejected transaction");
        require_status(adapter->forward(
                           memory.invalid_input.get(),
                           1u,
                           memory.cosine.get(),
                           memory.sine.get(),
                           3u,
                           &memory.cache,
                           memory.scratch,
                           memory.output.get(),
                           stream),
                       q4::QsaAdapterStatus::kOk,
                       "enqueue invalid forward");
        q4::QsaAdapterCollectedStatus rejected_status{};
        require_status(adapter->collect_device_status(
                           &memory.cache,
                           memory.scratch,
                           &rejected_status,
                           stream),
                       q4::QsaAdapterStatus::kDeviceRejected,
                       "fail-closed device collection");
        require(rejected_status.indexer != q4::qsa::status::ok ||
                    rejected_status.attention_core != q4::attention::status::ok,
                "invalid device input was not reported");
        require_status(q4::qsa_adapter_cache_get_lengths(&memory.cache, &lengths),
                       q4::QsaAdapterStatus::kOk,
                       "lengths after device rejection");
        require(lengths.committed == kTokens && lengths.staged == 0u &&
                    lengths.visible == kTokens,
                "device rejection exposed an uncommitted cache suffix");

        std::cout
            << "qwen4exp-qsa-adapter-test: PASS sm=120 layer=3 mixed_hidden=2560 "
            << "qkv_o=real index_qk=real qsa=4x128+1x128 block=4 top=512 "
            << "q_abs=" << q_error
            << " k_abs=" << k_error
            << " v_abs=" << v_error
            << " index_abs=" << index_error
            << " attention_abs=" << attention_error
            << " output_abs=" << output_error
            << " deterministic=2/2 transaction=rollback+commit+fail_closed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "qwen4exp-qsa-adapter-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
