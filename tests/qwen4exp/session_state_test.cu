#include "axiom/qwen4exp/session_state.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

/* Public-contract fake for the model-level coordinator test.  It deliberately
 * contains no model math: the real decoder_layer implementation owns that.
 * This fake makes cross-layer failure injection deterministic without loading
 * 135 GB of weights into a transaction unit test. */
namespace axiom::qwen4exp {
namespace {

std::size_t fake_next_layer = 0u;
std::size_t fake_failure_layer = kModelSessionNoFailedLayer;
std::int64_t fake_failure_token = std::numeric_limits<std::int64_t>::min();
std::array<std::array<std::size_t, kModelSessionLayerCount>, 2u>
        fake_rollbacks{};

}  // namespace

struct decoder_layer::impl {
    decoder_layer_config config{};
    int device = -1;
    std::size_t sessions_created = 0u;
    bool ready = false;
};

struct decoder_layer_session::impl {
    decoder_layer_state state{};
    cudaStream_t stream = nullptr;
    std::size_t session_slot = 0u;
    bool ready = false;
};

const char *decoder_layer_status_string(decoder_layer_status status) noexcept {
    switch (status) {
        case decoder_layer_status::ok: return "ok";
        case decoder_layer_status::invalid_argument: return "invalid_argument";
        case decoder_layer_status::unsupported_layer:
            return "unsupported_layer";
        case decoder_layer_status::unsupported_config:
            return "unsupported_config";
        case decoder_layer_status::invalid_checkpoint:
            return "invalid_checkpoint";
        case decoder_layer_status::allocation_failure:
            return "allocation_failure";
        case decoder_layer_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case decoder_layer_status::invalid_state: return "invalid_state";
        case decoder_layer_status::capacity_exceeded:
            return "capacity_exceeded";
        case decoder_layer_status::mhc_error: return "mhc_error";
        case decoder_layer_status::gdn_error: return "gdn_error";
        case decoder_layer_status::qsa_error: return "qsa_error";
        case decoder_layer_status::ple_error: return "ple_error";
        case decoder_layer_status::moe_error: return "moe_error";
        case decoder_layer_status::transaction_error:
            return "transaction_error";
        case decoder_layer_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

decoder_layer_kind decoder_layer_kind_for(std::size_t index) noexcept {
    return index % 4u == 3u ? decoder_layer_kind::qsa
                            : decoder_layer_kind::gated_deltanet;
}

bool decoder_layer_has_ple(std::size_t index) noexcept { return index == 1u; }

decoder_layer_status decoder_layer_validate_config(
        const decoder_layer_config &config) noexcept {
    return config.layer_index < kDecoderLayerCount &&
                   config.max_context > 0u &&
                   config.max_context <= kDecoderMaxContext
            ? decoder_layer_status::ok
            : decoder_layer_status::unsupported_config;
}

decoder_layer::decoder_layer() : impl_(new impl()) {
    impl_->config.layer_index = fake_next_layer++;
    impl_->config.max_context = kDecoderMaxContext;
    impl_->config.max_batch = 1u;
    (void)cudaGetDevice(&impl_->device);
    impl_->ready = impl_->config.layer_index < kDecoderLayerCount;
}

decoder_layer::~decoder_layer() = default;
decoder_layer::decoder_layer(decoder_layer &&) noexcept = default;
decoder_layer &decoder_layer::operator=(decoder_layer &&) noexcept = default;

decoder_layer_status decoder_layer::create_session(
        std::size_t context_capacity,
        cudaStream_t initialization_stream,
        std::unique_ptr<decoder_layer_session> *out,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || out == nullptr || context_capacity == 0u ||
        context_capacity > impl_->config.max_context ||
        impl_->sessions_created >= 2u) {
        if (error != nullptr) *error = "fake create_session rejected input";
        return decoder_layer_status::invalid_argument;
    }
    std::unique_ptr<decoder_layer_session> result(new decoder_layer_session());
    result->impl_.reset(new decoder_layer_session::impl());
    result->impl_->state.layer_index = impl_->config.layer_index;
    result->impl_->state.kind = kind();
    result->impl_->state.context_capacity = context_capacity;
    result->impl_->state.committed_tokens = 0u;
    result->impl_->state.has_ple = has_ple();
    result->impl_->stream = initialization_stream;
    result->impl_->session_slot = impl_->sessions_created++;
    result->impl_->ready = true;
    *out = std::move(result);
    return decoder_layer_status::ok;
}

const decoder_layer_config &decoder_layer::config() const noexcept {
    static const decoder_layer_config empty{};
    return impl_ ? impl_->config : empty;
}

decoder_layer_kind decoder_layer::kind() const noexcept {
    return impl_ ? decoder_layer_kind_for(impl_->config.layer_index)
                 : decoder_layer_kind::gated_deltanet;
}

bool decoder_layer::has_ple() const noexcept {
    return impl_ && decoder_layer_has_ple(impl_->config.layer_index);
}

int decoder_layer::device() const noexcept { return impl_ ? impl_->device : -1; }
bool decoder_layer::initialized() const noexcept {
    return impl_ && impl_->ready;
}

decoder_layer_session::decoder_layer_session() = default;
decoder_layer_session::~decoder_layer_session() = default;
decoder_layer_session::decoder_layer_session(
        decoder_layer_session &&) noexcept = default;
decoder_layer_session &decoder_layer_session::operator=(
        decoder_layer_session &&) noexcept = default;

decoder_layer_status decoder_layer_session::stage_token(
        std::int64_t token_id,
        const float *residual_4x2560_f32,
        const float *,
        const float *,
        std::size_t position_count,
        float *output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || residual_4x2560_f32 == nullptr ||
        output_4x2560_f32 == nullptr || position_count == 0u ||
        stream != impl_->stream || impl_->state.transaction_open ||
        impl_->state.committed_tokens >= impl_->state.context_capacity) {
        if (error != nullptr) *error = "fake stage rejected input";
        return decoder_layer_status::invalid_state;
    }
    impl_->state.transaction_open = true;
    impl_->state.prepared_to_commit = false;
    if (impl_->state.layer_index == fake_failure_layer &&
        token_id == fake_failure_token) {
        if (error != nullptr) *error = "injected layer failure";
        return decoder_layer_status::transaction_error;
    }
    const cudaError_t copy = cudaMemcpyAsync(
            output_4x2560_f32,
            residual_4x2560_f32,
            kDecoderResidual * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream);
    if (copy != cudaSuccess) {
        if (error != nullptr) *error = "fake residual copy failed";
        return decoder_layer_status::cuda_error;
    }
    impl_->state.staged_tokens = 1u;
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::stage_sequence(
        const std::int64_t *token_ids,
        std::size_t token_count,
        const float *residual_4x2560_f32,
        const float *,
        const float *,
        std::size_t position_count,
        float *output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || token_ids == nullptr || token_count == 0u ||
        token_count > kDecoderMaxSpeculativeTokens ||
        residual_4x2560_f32 == nullptr || output_4x2560_f32 == nullptr ||
        position_count == 0u || stream != impl_->stream ||
        impl_->state.transaction_open ||
        impl_->state.committed_tokens + token_count >
                impl_->state.context_capacity) {
        if (error != nullptr) *error = "fake sequence stage rejected input";
        return decoder_layer_status::invalid_state;
    }
    impl_->state.transaction_open = true;
    impl_->state.prepared_to_commit = false;
    if (impl_->state.layer_index == fake_failure_layer &&
        token_ids[0] == fake_failure_token) {
        if (error != nullptr) *error = "injected sequence layer failure";
        return decoder_layer_status::transaction_error;
    }
    const cudaError_t copy = cudaMemcpyAsync(
            output_4x2560_f32, residual_4x2560_f32,
            token_count * kDecoderResidual * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
    if (copy != cudaSuccess) {
        if (error != nullptr) *error = "fake sequence residual copy failed";
        return decoder_layer_status::cuda_error;
    }
    impl_->state.staged_tokens = token_count;
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::commit(
        cudaStream_t stream,
        std::string *error) noexcept {
    decoder_layer_session *layers[] = {this};
    const decoder_layer_status prepared =
            decoder_model_transaction::prepare_commit(
                    layers, 1u, stream, error);
    if (prepared != decoder_layer_status::ok) return prepared;
    return decoder_model_transaction::commit_prepared_noexcept(layers, 1u)
            ? decoder_layer_status::ok
            : decoder_layer_status::transaction_error;
}

decoder_layer_status decoder_layer_session::rollback(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || stream != impl_->stream ||
        !impl_->state.transaction_open) {
        if (error != nullptr) *error = "fake rollback rejected state";
        return decoder_layer_status::transaction_error;
    }
    ++fake_rollbacks[impl_->session_slot][impl_->state.layer_index];
    impl_->state.transaction_open = false;
    impl_->state.prepared_to_commit = false;
    impl_->state.staged_tokens = 0u;
    return decoder_layer_status::ok;
}

decoder_layer_status decoder_layer_session::reset(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || stream != impl_->stream) {
        if (error != nullptr) *error = "fake reset rejected state";
        return decoder_layer_status::invalid_state;
    }
    impl_->state.committed_tokens = 0u;
    impl_->state.staged_tokens = 0u;
    impl_->state.transaction_open = false;
    impl_->state.prepared_to_commit = false;
    impl_->state.poisoned = false;
    return decoder_layer_status::ok;
}

decoder_layer_state decoder_layer_session::state() const noexcept {
    return impl_ ? impl_->state : decoder_layer_state{};
}

bool decoder_layer_session::initialized() const noexcept {
    return impl_ && impl_->ready;
}

decoder_layer_status decoder_model_transaction::prepare_commit(
        decoder_layer_session *const *layers,
        std::size_t layer_count,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        if (error != nullptr) *error = "fake prepare rejected layer set";
        return decoder_layer_status::invalid_argument;
    }
    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session *const layer = layers[index];
        if (layer == nullptr || !layer->impl_ || !layer->impl_->ready ||
            layer->impl_->stream != stream ||
            !layer->impl_->state.transaction_open ||
            layer->impl_->state.prepared_to_commit ||
            layer->impl_->state.poisoned) {
            if (error != nullptr) *error = "fake prepare rejected layer state";
            return decoder_layer_status::invalid_state;
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        if (error != nullptr) *error = "fake prepare stream failed";
        return decoder_layer_status::cuda_error;
    }
    for (std::size_t index = 0u; index < layer_count; ++index) {
        layers[index]->impl_->state.prepared_to_commit = true;
    }
    return decoder_layer_status::ok;
}

bool decoder_model_transaction::commit_prepared_noexcept(
        decoder_layer_session *const *layers,
        std::size_t layer_count) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        return false;
    }
    for (std::size_t index = 0u; index < layer_count; ++index) {
        decoder_layer_session *const layer = layers[index];
        if (layer == nullptr || !layer->impl_ ||
            !layer->impl_->state.transaction_open ||
            !layer->impl_->state.prepared_to_commit) {
            return false;
        }
    }
    for (std::size_t index = 0u; index < layer_count; ++index) {
        layers[index]->impl_->state.committed_tokens +=
                layers[index]->impl_->state.staged_tokens;
        layers[index]->impl_->state.staged_tokens = 0u;
        layers[index]->impl_->state.transaction_open = false;
        layers[index]->impl_->state.prepared_to_commit = false;
    }
    return true;
}

decoder_layer_status decoder_model_transaction::rollback_all(
        decoder_layer_session *const *layers,
        std::size_t layer_count,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (layers == nullptr || layer_count == 0u ||
        layer_count > kDecoderLayerCount) {
        if (error != nullptr) *error = "fake rollback rejected layer set";
        return decoder_layer_status::invalid_argument;
    }
    for (std::size_t offset = layer_count; offset > 0u; --offset) {
        decoder_layer_session *const layer = layers[offset - 1u];
        if (layer == nullptr || !layer->impl_ || !layer->impl_->ready ||
            layer->impl_->stream != stream) {
            if (error != nullptr) *error = "fake rollback invalid layer";
            return decoder_layer_status::invalid_state;
        }
        if (layer->impl_->state.transaction_open) {
            const decoder_layer_status status = layer->rollback(stream, error);
            if (status != decoder_layer_status::ok) return status;
        } else {
            layer->impl_->state.prepared_to_commit = false;
        }
    }
    return decoder_layer_status::ok;
}

}  // namespace axiom::qwen4exp

namespace {

[[noreturn]] void die(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) die(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        die(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::model_session_status actual,
                    q4::model_session_status wanted,
                    const std::string &operation,
                    const std::string &detail = {}) {
    if (actual != wanted) {
        die(operation + ": expected " +
            q4::model_session_status_string(wanted) + ", got " +
            q4::model_session_status_string(actual) +
            (detail.empty() ? std::string() : ": " + detail));
    }
}

class stream final {
public:
    stream() { require_cuda(cudaStreamCreate(&value_), "cudaStreamCreate"); }
    ~stream() {
        if (value_ != nullptr) (void)cudaStreamDestroy(value_);
    }
    stream(const stream &) = delete;
    stream &operator=(const stream &) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return value_; }

private:
    cudaStream_t value_ = nullptr;
};

template <typename T>
class device_buffer final {
public:
    explicit device_buffer(std::size_t elements) : elements_(elements) {
        require(elements > 0u, "device buffer cannot be empty");
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer_), bytes()),
                     "cudaMalloc");
    }
    ~device_buffer() {
        if (pointer_ != nullptr) (void)cudaFree(pointer_);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
    [[nodiscard]] T *get() noexcept { return pointer_; }
    [[nodiscard]] const T *get() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t bytes() const noexcept {
        return elements_ * sizeof(T);
    }

private:
    T *pointer_ = nullptr;
    std::size_t elements_ = 0u;
};

void check_view(const q4::model_session_state &session,
                std::uint64_t committed,
                bool open,
                const std::string &label) {
    const q4::model_session_view view = session.view();
    require(view.initialized && !view.poisoned,
            label + " is not a healthy initialized session");
    require(view.committed_tokens == committed,
            label + " committed token count mismatch");
    require(view.transaction_open == open,
            label + " transaction state mismatch");
}

void stage_and_commit(q4::model_session_state *session,
                      std::int64_t token,
                      std::size_t position_count,
                      const device_buffer<float> &input,
                      const device_buffer<float> &cosine,
                      const device_buffer<float> &sine,
                      device_buffer<float> *output,
                      cudaStream_t cuda_stream) {
    std::string error;
    require_status(session->begin(cuda_stream, &error),
                   q4::model_session_status::ok, "begin", error);
    require_status(session->stage_all(
                           token,
                           input.get(),
                           cosine.get(),
                           sine.get(),
                           position_count,
                           output->get(),
                           cuda_stream,
                           &error),
                   q4::model_session_status::ok, "stage_all", error);
    require_status(session->commit_all(cuda_stream, &error),
                   q4::model_session_status::ok, "commit_all", error);
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        require(device_count > 0, "session-state test requires a CUDA device");
        require_cuda(cudaSetDevice(0), "cudaSetDevice");
        cudaDeviceProp properties{};
        require_cuda(cudaGetDeviceProperties(&properties, 0),
                     "cudaGetDeviceProperties");
        require(properties.major >= 12,
                "session-state test requires compute capability 12.x");

        q4::model_session_config native_context{};
        native_context.context_capacity = q4::kModelSessionMaxContext;
        require_status(q4::model_session_validate_config(native_context),
                       q4::model_session_status::ok,
                       "native 262K context admission");
        native_context.context_capacity = q4::kModelSessionMaxContext + 1u;
        require_status(q4::model_session_validate_config(native_context),
                       q4::model_session_status::unsupported_config,
                       "context above native contract rejection");

        std::array<std::unique_ptr<q4::decoder_layer>,
                   q4::kModelSessionLayerCount>
                owned_layers{};
        q4::model_decoder_layers layers{};
        for (std::size_t index = 0u; index < layers.size(); ++index) {
            owned_layers[index].reset(new q4::decoder_layer());
            layers[index] = owned_layers[index].get();
        }

        stream stream_a;
        stream stream_b;
        q4::model_session_config config{};
        config.context_capacity = 4u;
        std::unique_ptr<q4::model_session_state> session_a;
        std::unique_ptr<q4::model_session_state> session_b;
        std::string error;
        require_status(q4::model_session_state::create(
                               layers, config, stream_a.get(), &session_a,
                               &error),
                       q4::model_session_status::ok,
                       "create session A", error);
        require_status(q4::model_session_state::create(
                               layers, config, stream_b.get(), &session_b,
                               &error),
                       q4::model_session_status::ok,
                       "create session B", error);
        require(session_a && session_b && session_a.get() != session_b.get(),
                "two independent model sessions were not created");

        device_buffer<float> input(q4::kDecoderResidual);
        device_buffer<float> output_a(q4::kDecoderResidual);
        device_buffer<float> output_b(q4::kDecoderResidual);
        device_buffer<float> cosine(config.context_capacity * 64u);
        device_buffer<float> sine(config.context_capacity * 64u);
        std::vector<float> host_input(q4::kDecoderResidual);
        for (std::size_t index = 0u; index < host_input.size(); ++index) {
            host_input[index] = static_cast<float>((index * 17u) % 257u) /
                                257.0F;
        }
        require_cuda(cudaMemcpy(input.get(), host_input.data(), input.bytes(),
                                cudaMemcpyHostToDevice),
                     "upload residual input");
        require_cuda(cudaMemset(cosine.get(), 0, cosine.bytes()),
                     "initialize cosine");
        require_cuda(cudaMemset(sine.get(), 0, sine.bytes()),
                     "initialize sine");

        stage_and_commit(session_a.get(), 101, 1u, input, cosine, sine,
                         &output_a, stream_a.get());
        check_view(*session_a, 1u, false, "session A after commit");
        check_view(*session_b, 0u, false, "session B isolation after A commit");

        require_status(session_a->begin(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "begin explicit rollback", error);
        require_status(session_a->stage_all(
                               202, input.get(), cosine.get(), sine.get(), 2u,
                               output_a.get(), stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "stage explicit rollback", error);
        require_status(session_a->rollback_all(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "rollback all 48 layers", error);
        check_view(*session_a, 1u, false, "session A after rollback");
        for (std::size_t layer = 0u;
             layer < q4::kModelSessionLayerCount;
             ++layer) {
            require(q4::fake_rollbacks[0u][layer] == 1u,
                    "explicit rollback did not visit every A layer");
            require(q4::fake_rollbacks[1u][layer] == 0u,
                    "A rollback contaminated session B");
        }

        q4::fake_failure_layer = 17u;
        q4::fake_failure_token = 777;
        require_status(session_a->begin(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "begin injected failure", error);
        require_status(session_a->stage_all(
                               777, input.get(), cosine.get(), sine.get(), 2u,
                               output_a.get(), stream_a.get(), &error),
                       q4::model_session_status::layer_error,
                       "injected stage failure", error);
        const q4::model_session_view failed_view = session_a->view();
        require(!failed_view.transaction_open && !failed_view.poisoned &&
                        failed_view.committed_tokens == 1u &&
                        failed_view.last_failed_layer == 17u,
                "injected failure did not fail closed");
        require_status(session_a->validate(&error),
                       q4::model_session_status::ok,
                       "validate after injected rollback", error);
        for (std::size_t layer = 0u; layer <= 17u; ++layer) {
            require(q4::fake_rollbacks[0u][layer] == 2u,
                    "injected failure did not roll back an opened A layer");
        }
        for (std::size_t layer = 18u;
             layer < q4::kModelSessionLayerCount;
             ++layer) {
            require(q4::fake_rollbacks[0u][layer] == 1u,
                    "injected failure touched an unopened A layer");
        }
        check_view(*session_b, 0u, false,
                   "session B isolation after injected A failure");

        q4::fake_failure_layer = q4::kModelSessionNoFailedLayer;
        q4::fake_failure_token = std::numeric_limits<std::int64_t>::min();
        stage_and_commit(session_a.get(), 303, 2u, input, cosine, sine,
                         &output_a, stream_a.get());
        check_view(*session_a, 2u, false,
                   "session A reusable after injected failure");

        require_status(session_a->begin(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "begin pointer failure", error);
        require_status(session_a->stage_all(
                               404, host_input.data(), cosine.get(), sine.get(),
                               3u, output_a.get(), stream_a.get(), &error),
                       q4::model_session_status::invalid_device_pointer,
                       "host pointer rejection", error);
        require_status(session_a->rollback_all(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "rollback empty coordinator transaction", error);
        check_view(*session_a, 2u, false,
                   "session A after pointer rejection");

        require_status(session_a->begin(stream_b.get(), &error),
                       q4::model_session_status::stream_mismatch,
                       "stream isolation", error);
        check_view(*session_a, 2u, false,
                   "session A after wrong-stream rejection");

        stage_and_commit(session_b.get(), 505, 1u, input, cosine, sine,
                         &output_b, stream_b.get());
        check_view(*session_b, 1u, false, "session B independent commit");
        check_view(*session_a, 2u, false,
                   "session A isolation after B commit");

        require_status(session_a->reset(stream_a.get(), &error),
                       q4::model_session_status::ok,
                       "reset session A", error);
        check_view(*session_a, 0u, false, "session A reset");
        check_view(*session_b, 1u, false,
                   "session B isolation after A reset");

        require_status(session_b->reset(stream_b.get(), &error),
                       q4::model_session_status::ok,
                       "reset session B before block", error);
        constexpr std::size_t block_tokens =
                q4::kDecoderMaxSpeculativeTokens;
        device_buffer<float> block_input(
                block_tokens * q4::kDecoderResidual);
        device_buffer<float> block_output(
                block_tokens * q4::kDecoderResidual);
        std::vector<float> host_block(block_tokens * q4::kDecoderResidual);
        for (std::size_t row = 0u; row < block_tokens; ++row) {
            std::copy(host_input.begin(), host_input.end(),
                      host_block.begin() + row * q4::kDecoderResidual);
        }
        require_cuda(cudaMemcpy(block_input.get(), host_block.data(),
                                block_input.bytes(), cudaMemcpyHostToDevice),
                     "upload block residual input");
        const std::array<std::int64_t, block_tokens> block_ids{{
                601, 602, 603, 604}};
        require_status(session_b->begin(stream_b.get(), &error),
                       q4::model_session_status::ok,
                       "begin four-token block", error);
        require_status(session_b->stage_all_sequence(
                               block_ids.data(), block_tokens,
                               block_input.get(), cosine.get(), sine.get(),
                               block_tokens, block_output.get(), stream_b.get(),
                               &error),
                       q4::model_session_status::ok,
                       "stage four-token model block", error);
        require(session_b->view().staged_tokens == block_tokens &&
                        session_b->view().committed_tokens == 0u,
                "four-token model block published early");
        require_status(session_b->commit_all(stream_b.get(), &error),
                       q4::model_session_status::ok,
                       "commit four-token model block", error);
        check_view(*session_b, block_tokens, false,
                   "session B four-token commit");

        require_cuda(cudaStreamSynchronize(stream_a.get()),
                     "synchronize stream A");
        require_cuda(cudaStreamSynchronize(stream_b.get()),
                     "synchronize stream B");
        std::cout
                << "qwen4exp-session-state-test: PASS layers=48 sessions=2 "
                   "context_max=262144 transaction=begin+stage_all+commit_all+"
                   "rollback_all block_width=4 reset=all failure_layer=17 "
                   "isolation=no-contamination\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-session-state-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
