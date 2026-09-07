#include "axiom/qwen4exp/decoder_layer.hpp"
#include "axiom/qwen4exp/rope.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

constexpr std::size_t kCapacity = 4u;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const std::string &where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(where + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::decoder_layer_status actual,
                    q4::decoder_layer_status expected,
                    const std::string &where,
                    const std::string &detail = {}) {
    if (actual != expected) {
        throw std::runtime_error(
                where + ": expected " +
                q4::decoder_layer_status_string(expected) + ", got " +
                q4::decoder_layer_status_string(actual) +
                (detail.empty() ? std::string{} : "; " + detail));
    }
}

class stream_owner {
public:
    stream_owner() { require_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    ~stream_owner() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    stream_owner(const stream_owner &) = delete;
    stream_owner &operator=(const stream_owner &) = delete;
    cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class device_buffer {
public:
    device_buffer() = default;
    explicit device_buffer(std::size_t elements) { reset(elements); }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    device_buffer(device_buffer &&other) noexcept
        : pointer_(std::exchange(other.pointer_, nullptr)),
          elements_(std::exchange(other.elements_, 0u)) {}
    device_buffer &operator=(device_buffer &&other) noexcept {
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
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                                elements * sizeof(T)),
                     "cudaMalloc");
        elements_ = elements;
    }
    void upload(const std::vector<T> &values, cudaStream_t stream) {
        require(values.size() == elements_, "device upload size mismatch");
        require_cuda(cudaMemcpyAsync(pointer_, values.data(),
                                     values.size() * sizeof(T),
                                     cudaMemcpyHostToDevice, stream),
                     "cudaMemcpyAsync H2D");
    }
    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(elements_);
        require_cuda(cudaMemcpyAsync(values.data(), pointer_,
                                     values.size() * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync D2H");
        require_cuda(cudaStreamSynchronize(stream), "download synchronize");
        return values;
    }
    T *get() noexcept { return pointer_; }
    const T *get() const noexcept { return pointer_; }

private:
    T *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

struct rope_tables {
    device_buffer<float> cosine{kCapacity * q4::rope::kRotaryDim};
    device_buffer<float> sine{kCapacity * q4::rope::kRotaryDim};

    explicit rope_tables(cudaStream_t stream) {
        std::vector<std::uint32_t> positions(3u * kCapacity);
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            for (std::size_t token = 0u; token < kCapacity; ++token) {
                positions[axis * kCapacity + token] =
                        static_cast<std::uint32_t>(token);
            }
        }
        std::vector<float> host_cos(kCapacity * q4::rope::kRotaryDim);
        std::vector<float> host_sin(host_cos.size());
        require(q4::rope::generate_host(q4::rope::checkpoint_config,
                                       positions.data(), kCapacity,
                                       host_cos.data(), host_sin.data()) ==
                        q4::rope::status::ok,
                "host RoPE generation failed");
        cosine.upload(host_cos, stream);
        sine.upload(host_sin, stream);
    }
};

std::vector<float> make_input(std::uint32_t seed) {
    std::vector<float> values(q4::kDecoderResidual);
    for (std::size_t index = 0u; index < values.size(); ++index) {
        const float angle = static_cast<float>(
                (index * 37u + static_cast<std::size_t>(seed) * 19u) % 4093u) *
                            0.0021F;
        values[index] = 0.018F * std::sin(angle) +
                        0.011F * std::cos(angle * 0.37F);
    }
    return values;
}

void require_finite(const std::vector<float> &values,
                    const std::string &where) {
    require(std::all_of(values.begin(), values.end(),
                        [](float value) { return std::isfinite(value); }),
            where + " contains a non-finite value");
}

float maximum_absolute_difference(const std::vector<float> &left,
                                  const std::vector<float> &right) {
    require(left.size() == right.size(), "comparison size mismatch");
    float maximum = 0.0F;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        maximum = std::max(maximum, std::abs(left[index] - right[index]));
    }
    return maximum;
}

std::unique_ptr<q4::decoder_layer> load_layer(
        const q4::checkpoint_catalog &catalog,
        std::size_t layer_index,
        bool enable_ple,
        cudaStream_t stream) {
    q4::decoder_layer_config config{};
    config.layer_index = layer_index;
    config.max_context = kCapacity;
    config.max_batch = q4::kDecoderMaxSpeculativeTokens;
    config.moe_ram_capacity_bytes = 64ull * 1024ull * 1024ull;
    config.moe_prefetch_workers = 2u;
    config.enable_ple = enable_ple;
    std::unique_ptr<q4::decoder_layer> layer;
    std::string error;
    require_status(q4::decoder_layer::load(
                           catalog, config, stream, &layer, &error),
                   q4::decoder_layer_status::ok,
                   "real decoder layer load", error);
    require(layer != nullptr && layer->initialized() &&
                    layer->config().layer_index == layer_index,
            "loaded decoder layer identity mismatch");
    return layer;
}

std::unique_ptr<q4::decoder_layer_session> create_session(
        q4::decoder_layer *layer,
        cudaStream_t stream) {
    std::unique_ptr<q4::decoder_layer_session> session;
    std::string error;
    require_status(layer->create_session(kCapacity, stream, &session, &error),
                   q4::decoder_layer_status::ok,
                   "real decoder session creation", error);
    require(session != nullptr && session->initialized(),
            "decoder session did not initialize");
    return session;
}

q4::decoder_layer_status stage(
        q4::decoder_layer_session *session,
        std::int64_t token,
        const float *input,
        float *output,
        const rope_tables *rope,
        cudaStream_t stream,
        std::string *error) {
    return session->stage_token(
            token, input,
            rope == nullptr ? nullptr : rope->cosine.get(),
            rope == nullptr ? nullptr : rope->sine.get(),
            rope == nullptr ? 0u : kCapacity,
            output, stream, error);
}

float test_real_layer(const q4::checkpoint_catalog &catalog,
                      std::size_t layer_index,
                      bool enable_ple,
                      const rope_tables *rope,
                      cudaStream_t stream) {
    auto layer = load_layer(catalog, layer_index, enable_ple, stream);
    auto session = create_session(layer.get(), stream);
    require(layer->kind() == q4::decoder_layer_kind_for(layer_index),
            "decoder layer schedule kind mismatch");
    require(layer->has_ple() ==
                    (enable_ple && q4::decoder_layer_has_ple(layer_index)),
            "decoder PLE admission mismatch");

    const std::vector<float> input = make_input(
            static_cast<std::uint32_t>(layer_index + 1u));
    device_buffer<float> device_input(q4::kDecoderResidual);
    device_buffer<float> device_output(q4::kDecoderResidual);
    device_input.upload(input, stream);

    std::string error;
    require_status(stage(session.get(), 101, device_input.get(),
                         device_output.get(), rope, stream, &error),
                   q4::decoder_layer_status::ok,
                   "first deterministic stage", error);
    const std::vector<float> first = device_output.download(stream);
    require_finite(first, "first decoder output");
    require(session->state().transaction_open &&
                    session->state().committed_tokens == 0u,
            "staged decoder state was published early");
    require_status(session->rollback(stream, &error),
                   q4::decoder_layer_status::ok,
                   "first deterministic rollback", error);

    if (layer->has_ple()) {
        require_status(session->prefetch_first_ple_token(101, stream, &error),
                       q4::decoder_layer_status::ok, "early unopened PLE start", error);
        require(!session->state().transaction_open, "early PLE opened outer transaction");
        q4::decoder_layer_session *only[] = {session.get()};
        require_status(q4::decoder_model_transaction::rollback_all(only, 1, stream, &error),
                       q4::decoder_layer_status::ok, "rollback unopened PLE layer", error);
        require_status(session->prefetch_first_ple_token(101, stream, &error),
                       q4::decoder_layer_status::ok, "restart after unopened rollback", error);
        require_status(session->reset(stream, &error), q4::decoder_layer_status::ok,
                       "reset unopened PLE layer", error);
        require_status(session->prefetch_first_ple_token(101, stream, &error),
                       q4::decoder_layer_status::ok, "early PLE replay start", error);
        require_status(stage(session.get(), 102, device_input.get(), device_output.get(),
                             rope, stream, &error), q4::decoder_layer_status::invalid_state,
                       "mismatched early token rejected", error);
        require_status(session->rollback(stream, &error), q4::decoder_layer_status::ok,
                       "rollback mismatched early token", error);
        require_status(session->prefetch_first_ple_token(101, stream, &error),
                       q4::decoder_layer_status::ok, "early PLE matching token", error);
    }
    require_status(stage(session.get(), 101, device_input.get(),
                         device_output.get(), rope, stream, &error),
                   q4::decoder_layer_status::ok,
                   "repeat deterministic stage", error);
    const std::vector<float> second = device_output.download(stream);
    const float maximum = maximum_absolute_difference(first, second);
    require(std::memcmp(first.data(), second.data(),
                        first.size() * sizeof(float)) == 0,
            "decoder rollback/replay is not bit-identical");
    require_status(session->rollback(stream, &error),
                   q4::decoder_layer_status::ok,
                   "repeat deterministic rollback", error);

    require_status(stage(session.get(), 101, device_input.get(),
                         device_output.get(), rope, stream, &error),
                   q4::decoder_layer_status::ok,
                   "commit stage", error);
    require_status(session->commit(stream, &error),
                   q4::decoder_layer_status::ok,
                   "single-layer atomic commit", error);
    require_cuda(cudaStreamSynchronize(stream), "single-layer commit completion");
    require(session->state().committed_tokens == 1u &&
                    !session->state().transaction_open &&
                    !session->state().prepared_to_commit,
            "single-layer commit state mismatch");

    require_status(stage(session.get(), 202, device_input.get(),
                         device_output.get(), rope, stream, &error),
                   q4::decoder_layer_status::ok,
                   "post-commit rollback stage", error);
    require_status(session->rollback(stream, &error),
                   q4::decoder_layer_status::ok,
                   "post-commit rollback", error);
    require(session->state().committed_tokens == 1u,
            "rollback changed previously committed state");
    require_status(session->reset(stream, &error),
                   q4::decoder_layer_status::ok,
                   "decoder session reset", error);
    require(session->state().committed_tokens == 0u,
            "decoder reset did not clear committed state");

    constexpr std::size_t block_tokens = q4::kDecoderMaxSpeculativeTokens;
    std::vector<float> block_input(block_tokens * q4::kDecoderResidual);
    for (std::size_t row = 0u; row < block_tokens; ++row) {
        std::copy(input.begin(), input.end(),
                  block_input.begin() + row * q4::kDecoderResidual);
    }
    device_buffer<float> device_block_input(block_input.size());
    device_buffer<float> device_block_output(block_input.size());
    device_block_input.upload(block_input, stream);
    const std::array<std::int64_t, block_tokens> block_ids{{401, 402, 403, 404}};
    if (layer->has_ple()) {
        require_status(session->prefetch_first_ple_token(block_ids[0], stream, &error),
                       q4::decoder_layer_status::ok, "first-row block prefetch", error);
    }
    require_status(session->stage_sequence(
                           block_ids.data(), block_tokens,
                           device_block_input.get(),
                           rope == nullptr ? nullptr : rope->cosine.get(),
                           rope == nullptr ? nullptr : rope->sine.get(),
                           rope == nullptr ? 0u : kCapacity,
                           device_block_output.get(), stream, &error),
                   q4::decoder_layer_status::ok,
                   "four-token causal decoder stage", error);
    require(session->state().transaction_open &&
                    session->state().staged_tokens == block_tokens &&
                    session->state().committed_tokens == 0u,
            "four-token decoder stage published early");
    const std::vector<float> block_result =
            device_block_output.download(stream);
    require_finite(block_result, "four-token decoder output");
    require_status(session->commit(stream, &error),
                   q4::decoder_layer_status::ok,
                   "four-token atomic commit", error);
    require(session->state().committed_tokens == block_tokens &&
                    session->state().staged_tokens == 0u &&
                    !session->state().transaction_open,
            "four-token decoder commit state mismatch");
    require_status(session->reset(stream, &error),
                   q4::decoder_layer_status::ok,
                   "four-token decoder reset", error);
    return maximum;
}

void test_cross_layer_atomicity(const q4::checkpoint_catalog &catalog,
                                const rope_tables &rope,
                                cudaStream_t stream) {
    auto layer0 = load_layer(catalog, 0u, false, stream);
    std::cerr << "decoder-gate: atomic loaded layer 0\n";
    auto layer3 = load_layer(catalog, 3u, false, stream);
    std::cerr << "decoder-gate: atomic loaded layer 3\n";
    auto session0 = create_session(layer0.get(), stream);
    auto session3 = create_session(layer3.get(), stream);
    std::cerr << "decoder-gate: atomic sessions ready\n";
    device_buffer<float> input(q4::kDecoderResidual);
    device_buffer<float> layer0_output(q4::kDecoderResidual);
    device_buffer<float> layer3_output(q4::kDecoderResidual);
    input.upload(make_input(77u), stream);

    std::string error;
    require_status(stage(session0.get(), 301, input.get(), layer0_output.get(),
                         nullptr, stream, &error),
                   q4::decoder_layer_status::ok,
                   "cross-layer first stage", error);
    std::cerr << "decoder-gate: atomic staged prior layer\n";
    std::vector<float> invalid_host_output(q4::kDecoderResidual);
    const q4::decoder_layer_status failed = stage(
            session3.get(), 301, layer0_output.get(),
            invalid_host_output.data(), &rope, stream, &error);
    require(failed != q4::decoder_layer_status::ok,
            "intentional downstream layer failure was accepted");
    std::cerr << "decoder-gate: atomic downstream failure observed\n";
    q4::decoder_layer_session *failed_stack[] = {
        session0.get(), session3.get()};
    require_status(q4::decoder_model_transaction::rollback_all(
                           failed_stack, 2u, stream, &error),
                   q4::decoder_layer_status::ok,
                   "rollback all after downstream layer failure", error);
    std::cerr << "decoder-gate: atomic rollback-all passed\n";
    require(!session0->state().transaction_open &&
                    session0->state().committed_tokens == 0u &&
                    !session3->state().transaction_open &&
                    session3->state().committed_tokens == 0u,
            "downstream failure did not roll back every prior layer");

    require_status(stage(session0.get(), 302, input.get(), layer0_output.get(),
                         nullptr, stream, &error),
                   q4::decoder_layer_status::ok,
                   "atomic stack GDN stage", error);
    std::cerr << "decoder-gate: atomic restaged GDN\n";
    require_status(stage(session3.get(), 302, layer0_output.get(),
                         layer3_output.get(), &rope, stream, &error),
                   q4::decoder_layer_status::ok,
                   "atomic stack QSA stage", error);
    std::cerr << "decoder-gate: atomic staged QSA\n";
    q4::decoder_layer_session *stack[] = {session0.get(), session3.get()};
    require_status(q4::decoder_model_transaction::prepare_commit(
                           stack, 2u, stream, &error),
                   q4::decoder_layer_status::ok,
                   "model-wide prepare commit", error);
    std::cerr << "decoder-gate: atomic prepare passed\n";
    require(session0->state().prepared_to_commit &&
                    session3->state().prepared_to_commit &&
                    session0->state().committed_tokens == 0u &&
                    session3->state().committed_tokens == 0u,
            "prepare_commit published semantic state");
    require(q4::decoder_model_transaction::commit_prepared_noexcept(stack, 2u),
            "prepared model publication violated an invariant");
    std::cerr << "decoder-gate: atomic publish passed\n";
    require_cuda(cudaStreamSynchronize(stream), "atomic stack completion");
    require(session0->state().committed_tokens == 1u &&
                    session3->state().committed_tokens == 1u &&
                    !session0->state().transaction_open &&
                    !session3->state().transaction_open,
            "atomic model publication did not commit both layers");
    std::cerr << "decoder-gate: atomic assertions passed\n";
}

void test_schedule(const q4::checkpoint_catalog &catalog) {
    std::string error;
    require_status(q4::decoder_layer_validate_checkpoint_schedule(
                           catalog, &error),
                   q4::decoder_layer_status::ok,
                   "real 48-layer schedule", error);
    std::size_t gdn = 0u;
    std::size_t qsa = 0u;
    std::size_t ple = 0u;
    for (std::size_t layer = 0u; layer < q4::kDecoderLayerCount; ++layer) {
        if (q4::decoder_layer_kind_for(layer) ==
            q4::decoder_layer_kind::qsa) {
            ++qsa;
        } else {
            ++gdn;
        }
        if (q4::decoder_layer_has_ple(layer)) ++ple;
    }
    require(gdn == 36u && qsa == 12u && ple == 1u,
            "compiled 48-layer schedule count mismatch");
    q4::decoder_layer_config invalid{};
    invalid.layer_index = 48u;
    require_status(q4::decoder_layer_validate_config(invalid),
                   q4::decoder_layer_status::invalid_argument,
                   "out-of-range layer rejection");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2 || argc == 3,
                "usage: decoder-layer-test MODEL_ROOT [gdn|qsa|ple|atomic|all]");
        const std::string mode = argc == 3 ? argv[2] : "all";
        require(mode == "gdn" || mode == "qsa" || mode == "ple" ||
                        mode == "atomic" || mode == "all",
                "invalid decoder gate mode");
        int device = -1;
        cudaDeviceProp properties{};
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12 && properties.minor == 0,
                "decoder layer gate requires SM120");

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        std::string error;
        require(q4::checkpoint_catalog::open(argv[1], &catalog, &error) &&
                        catalog != nullptr,
                error.empty() ? "checkpoint catalog open failed" : error);
        test_schedule(*catalog);

        stream_owner stream;
        rope_tables rope(stream.get());
        require_cuda(cudaStreamSynchronize(stream.get()), "RoPE upload");
        float gdn_abs = 0.0F;
        float qsa_abs = 0.0F;
        float ple_abs = 0.0F;
        if (mode == "gdn" || mode == "all") {
            std::cerr << "decoder-gate: begin real GDN layer 0\n";
            gdn_abs = test_real_layer(
                    *catalog, 0u, false, nullptr, stream.get());
            std::cerr << "decoder-gate: end real GDN layer 0\n";
        }
        if (mode == "qsa" || mode == "all") {
            std::cerr << "decoder-gate: begin real QSA layer 3\n";
            qsa_abs = test_real_layer(
                    *catalog, 3u, false, &rope, stream.get());
            std::cerr << "decoder-gate: end real QSA layer 3\n";
        }
        if (mode == "ple" || mode == "all") {
            std::cerr << "decoder-gate: begin real PLE/GDN layer 1\n";
            ple_abs = test_real_layer(
                    *catalog, 1u, true, nullptr, stream.get());
            std::cerr << "decoder-gate: end real PLE/GDN layer 1\n";
        }
        if (mode == "atomic" || mode == "all") {
            std::cerr << "decoder-gate: begin cross-layer atomicity\n";
            test_cross_layer_atomicity(*catalog, rope, stream.get());
            std::cerr << "decoder-gate: end cross-layer atomicity\n";
        }

        std::cout
                << "qwen4exp-decoder-layer-test: PASS schedule=36GDN+12QSA "
                   "ple_layer=1 real_layers=0,1,3 deterministic_abs="
                << gdn_abs << ',' << ple_abs << ',' << qsa_abs
                << " atomic=model_prepare+noexcept_publish "
                   "downstream_failure=rollback_all\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-decoder-layer-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
