#include "axiom/qwen4exp/session_state.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <utility>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kResidualElements =
        kDecoderMaxSpeculativeTokens * kDecoderStreams * kDecoderHidden;
constexpr std::size_t kResidualBytes = kResidualElements * sizeof(float);

model_session_status fail(model_session_status status,
                          std::string *error,
                          const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return status;
}

bool expected_qsa(std::size_t layer_index) noexcept {
    return layer_index % 4u == 3u;
}

bool pointer_on_device(const void *pointer, int device) noexcept {
    if (pointer == nullptr) return false;
    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    return attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
#else
    return attributes.memoryType == cudaMemoryTypeDevice &&
           attributes.device == device;
#endif
}

bool ranges_overlap(const void *left,
                    std::size_t left_bytes,
                    const void *right,
                    std::size_t right_bytes) noexcept {
    const std::uintptr_t left_begin =
            reinterpret_cast<std::uintptr_t>(left);
    const std::uintptr_t right_begin =
            reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
        right_begin >
                std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
        return true;
    }
    const std::uintptr_t left_end = left_begin + left_bytes;
    const std::uintptr_t right_end = right_begin + right_bytes;
    return left_begin < right_end && right_begin < left_end;
}

std::string layer_message(std::size_t layer_index,
                          const char *operation,
                          decoder_layer_status status,
                          const std::string &detail) {
    std::ostringstream stream;
    stream << "qwen4_exp layer " << layer_index << ' ' << operation
           << " failed: " << decoder_layer_status_string(status);
    if (!detail.empty()) stream << ": " << detail;
    return stream.str();
}

}  // namespace

struct model_session_state::impl {
    model_session_config config{};
    std::array<decoder_layer *, kModelSessionLayerCount> layers{};
    std::array<std::unique_ptr<decoder_layer_session>,
               kModelSessionLayerCount>
            sessions{};
    std::array<float *, 2u> residual_ping_pong{{nullptr, nullptr}};
    cudaStream_t stream = nullptr;
    std::uint64_t committed_tokens = 0u;
    std::uint64_t staged_tokens = 0u;
    std::size_t staged_layers = 0u;
    std::size_t last_failed_layer = kModelSessionNoFailedLayer;
    int device = -1;
    bool transaction_open = false;
    bool poisoned = false;
    bool ready = false;

    ~impl() {
        int previous_device = -1;
        const bool have_previous = cudaGetDevice(&previous_device) == cudaSuccess;
        if (device >= 0 && (!have_previous || previous_device != device)) {
            (void)cudaSetDevice(device);
        }
        for (float *pointer : residual_ping_pong) {
            if (pointer != nullptr) (void)cudaFree(pointer);
        }
        if (have_previous && previous_device >= 0 && previous_device != device) {
            (void)cudaSetDevice(previous_device);
        }
    }

    bool stream_matches(cudaStream_t candidate) const noexcept {
        return candidate == stream;
    }

    model_session_status rollback_open_layers(
            cudaStream_t candidate,
            std::size_t *failed_layer,
            std::string *detail) noexcept {
        std::array<decoder_layer_session *, kModelSessionLayerCount> raw{};
        for (std::size_t index = 0u; index < raw.size(); ++index) {
            raw[index] = sessions[index].get();
        }
        std::string rollback_detail;
        const decoder_layer_status status =
                decoder_model_transaction::rollback_all(
                        raw.data(), raw.size(), candidate, &rollback_detail);
        if (status == decoder_layer_status::ok) {
            if (failed_layer != nullptr) {
                *failed_layer = kModelSessionNoFailedLayer;
            }
            if (detail != nullptr) detail->clear();
            return model_session_status::ok;
        }
        std::size_t first_failure = kModelSessionNoFailedLayer;
        for (std::size_t index = 0u; index < sessions.size(); ++index) {
            if (sessions[index] && sessions[index]->state().transaction_open) {
                first_failure = index;
                break;
            }
        }
        if (failed_layer != nullptr) *failed_layer = first_failure;
        if (detail != nullptr) {
            *detail = rollback_detail.empty()
                    ? "decoder model rollback failed"
                    : rollback_detail;
        }
        return model_session_status::rollback_error;
    }

    bool reset_every_layer(cudaStream_t candidate,
                           std::size_t *failed_layer,
                           std::string *detail) noexcept {
        bool all_ok = true;
        std::size_t first_failure = kModelSessionNoFailedLayer;
        std::string first_detail;
        for (std::size_t index = 0u; index < kModelSessionLayerCount; ++index) {
            if (!sessions[index]) continue;
            std::string layer_error;
            const decoder_layer_status status =
                    sessions[index]->reset(candidate, &layer_error);
            if (status != decoder_layer_status::ok && all_ok) {
                all_ok = false;
                first_failure = index;
                first_detail =
                        layer_message(index, "reset", status, layer_error);
            }
        }
        if (failed_layer != nullptr) *failed_layer = first_failure;
        if (detail != nullptr) *detail = first_detail;
        return all_ok;
    }
};

const char *model_session_status_string(model_session_status status) noexcept {
    switch (status) {
        case model_session_status::ok: return "ok";
        case model_session_status::invalid_argument: return "invalid_argument";
        case model_session_status::unsupported_config:
            return "unsupported_config";
        case model_session_status::size_overflow: return "size_overflow";
        case model_session_status::allocation_failure:
            return "allocation_failure";
        case model_session_status::unsupported_device:
            return "unsupported_device";
        case model_session_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case model_session_status::stream_mismatch: return "stream_mismatch";
        case model_session_status::invalid_state: return "invalid_state";
        case model_session_status::capacity_exceeded:
            return "capacity_exceeded";
        case model_session_status::transaction_open: return "transaction_open";
        case model_session_status::no_transaction: return "no_transaction";
        case model_session_status::stage_already_complete:
            return "stage_already_complete";
        case model_session_status::layer_error: return "layer_error";
        case model_session_status::rollback_error: return "rollback_error";
        case model_session_status::reset_error: return "reset_error";
        case model_session_status::poisoned: return "poisoned";
        case model_session_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

model_session_status model_session_validate_config(
        const model_session_config &config) noexcept {
    if (config.context_capacity == 0u ||
        config.context_capacity > kModelSessionMaxContext) {
        return model_session_status::unsupported_config;
    }
    return model_session_status::ok;
}

model_session_state::model_session_state() = default;
model_session_state::~model_session_state() = default;
model_session_state::model_session_state(model_session_state &&) noexcept =
        default;
model_session_state &model_session_state::operator=(
        model_session_state &&) noexcept = default;

model_session_status model_session_state::create(
        const model_decoder_layers &layers,
        const model_session_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<model_session_state> *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return fail(model_session_status::invalid_argument, error,
                    "model session output pointer is null");
    }
    out->reset();
    const model_session_status config_status =
            model_session_validate_config(config);
    if (config_status != model_session_status::ok) {
        return fail(config_status, error,
                    "context_capacity must be in [1,262144]");
    }
    try {
        int active_device = -1;
        if (cudaGetDevice(&active_device) != cudaSuccess) {
            return fail(model_session_status::cuda_error, error,
                        "cudaGetDevice failed while creating model session");
        }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, active_device) != cudaSuccess) {
            return fail(model_session_status::cuda_error, error,
                        "cudaGetDeviceProperties failed");
        }
        if (properties.major < 12) {
            return fail(model_session_status::unsupported_device, error,
                        "qwen4_exp model session requires compute capability 12.x");
        }

        std::unique_ptr<model_session_state> result(
                new model_session_state());
        result->impl_.reset(new impl());
        impl &state = *result->impl_;
        state.config = config;
        state.layers = layers;
        state.stream = initialization_stream;
        state.device = active_device;

        for (std::size_t index = 0u; index < kModelSessionLayerCount;
             ++index) {
            decoder_layer *const layer = layers[index];
            if (layer == nullptr || !layer->initialized()) {
                return fail(model_session_status::invalid_argument, error,
                            "all 48 immutable decoder layers must be initialized");
            }
            const decoder_layer_config &layer_config = layer->config();
            const decoder_layer_kind wanted_kind =
                    expected_qsa(index) ? decoder_layer_kind::qsa
                                        : decoder_layer_kind::gated_deltanet;
            if (layer_config.layer_index != index ||
                layer->kind() != wanted_kind ||
                layer->has_ple() != (index == 1u) ||
                layer->device() != active_device ||
                layer_config.max_context < config.context_capacity) {
                std::ostringstream message;
                message << "decoder layer binding mismatch at layer " << index;
                return fail(model_session_status::unsupported_config, error,
                            message.str());
            }
            std::string layer_error;
            const decoder_layer_status layer_status = layer->create_session(
                    config.context_capacity,
                    initialization_stream,
                    &state.sessions[index],
                    &layer_error);
            if (layer_status != decoder_layer_status::ok ||
                !state.sessions[index] ||
                !state.sessions[index]->initialized()) {
                return fail(
                        model_session_status::layer_error,
                        error,
                        layer_message(index, "session creation", layer_status,
                                      layer_error));
            }
        }

        for (float *&pointer : state.residual_ping_pong) {
            if (cudaMalloc(reinterpret_cast<void **>(&pointer),
                           kResidualBytes) != cudaSuccess) {
                return fail(model_session_status::allocation_failure, error,
                            "failed to allocate bounded residual ping-pong buffer");
            }
        }
        state.ready = true;
        const model_session_status validation = result->validate(error);
        if (validation != model_session_status::ok) return validation;
        *out = std::move(result);
        return model_session_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(model_session_status::allocation_failure, error,
                    "host allocation failed while creating model session");
    } catch (const std::exception &exception) {
        return fail(model_session_status::allocation_failure, error,
                    std::string("model session creation exception: ") +
                            exception.what());
    } catch (...) {
        return fail(model_session_status::allocation_failure, error,
                    "unknown model session creation exception");
    }
}

model_session_status model_session_state::validate(std::string *error) const
        noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    const impl &state = *impl_;
    if (state.poisoned) {
        return fail(model_session_status::poisoned, error,
                    "model session is poisoned and requires reset");
    }
    if (model_session_validate_config(state.config) !=
            model_session_status::ok ||
        state.device < 0 || state.residual_ping_pong[0] == nullptr ||
        state.residual_ping_pong[1] == nullptr ||
        state.committed_tokens > state.config.context_capacity ||
        state.staged_tokens > kDecoderMaxSpeculativeTokens ||
        state.staged_layers > kModelSessionLayerCount) {
        return fail(model_session_status::invalid_state, error,
                    "model session coordinator metadata is inconsistent");
    }
    if ((!state.transaction_open &&
         (state.staged_tokens != 0u || state.staged_layers != 0u)) ||
        (state.transaction_open && state.staged_layers == 0u &&
         state.staged_tokens != 0u) ||
        (state.transaction_open && state.staged_layers != 0u &&
         (state.staged_layers != kModelSessionLayerCount ||
          state.staged_tokens == 0u))) {
        return fail(model_session_status::invalid_state, error,
                    "model transaction metadata is inconsistent");
    }

    for (std::size_t index = 0u; index < kModelSessionLayerCount; ++index) {
        const decoder_layer *const layer = state.layers[index];
        const std::unique_ptr<decoder_layer_session> &session =
                state.sessions[index];
        if (layer == nullptr || !layer->initialized() || !session ||
            !session->initialized()) {
            return fail(model_session_status::invalid_state, error,
                        "model session lost a decoder layer binding");
        }
        const decoder_layer_state layer_state = session->state();
        const decoder_layer_kind wanted_kind =
                expected_qsa(index) ? decoder_layer_kind::qsa
                                    : decoder_layer_kind::gated_deltanet;
        const bool should_be_open =
                state.transaction_open &&
                state.staged_layers == kModelSessionLayerCount;
        if (layer_state.layer_index != index ||
            layer_state.kind != wanted_kind ||
            layer_state.context_capacity != state.config.context_capacity ||
            layer_state.committed_tokens != state.committed_tokens ||
            layer_state.staged_tokens !=
                    (should_be_open ? state.staged_tokens : 0u) ||
            layer_state.has_ple != (index == 1u) || layer_state.poisoned ||
            layer_state.prepared_to_commit ||
            layer_state.transaction_open != should_be_open) {
            std::ostringstream message;
            message << "decoder layer session state mismatch at layer "
                    << index;
            return fail(model_session_status::invalid_state, error,
                        message.str());
        }
    }
    return model_session_status::ok;
}

model_session_status model_session_state::begin(cudaStream_t stream,
                                                std::string *error) noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    impl &state = *impl_;
    if (state.poisoned) {
        return fail(model_session_status::poisoned, error,
                    "model session is poisoned and requires reset");
    }
    if (!state.stream_matches(stream)) {
        return fail(model_session_status::stream_mismatch, error,
                    "model session transaction used a different CUDA stream");
    }
    if (state.transaction_open) {
        return fail(model_session_status::transaction_open, error,
                    "model session transaction is already open");
    }
    const model_session_status validation = validate(error);
    if (validation != model_session_status::ok) return validation;
    if (state.committed_tokens >= state.config.context_capacity) {
        return fail(model_session_status::capacity_exceeded, error,
                    "model session reached its physical context capacity");
    }
    state.transaction_open = true;
    state.staged_tokens = 0u;
    state.staged_layers = 0u;
    state.last_failed_layer = kModelSessionNoFailedLayer;
    return model_session_status::ok;
}

model_session_status model_session_state::stage_all(
        std::int64_t token_id,
        const float *residual_input_4x2560_f32,
        const float *full_cos_64_f32,
        const float *full_sin_64_f32,
        std::size_t position_count,
        float *residual_output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    return stage_all_sequence(
            &token_id, 1u, residual_input_4x2560_f32, full_cos_64_f32,
            full_sin_64_f32, position_count,
            residual_output_4x2560_f32, stream, error);
}

model_session_status model_session_state::stage_all_sequence(
        const std::int64_t *token_ids,
        std::size_t token_count,
        const float *residual_input_4x2560_f32,
        const float *full_cos_64_f32,
        const float *full_sin_64_f32,
        std::size_t position_count,
        float *residual_output_4x2560_f32,
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    impl &state = *impl_;
    if (state.poisoned) {
        return fail(model_session_status::poisoned, error,
                    "model session is poisoned and requires reset");
    }
    if (!state.stream_matches(stream)) {
        return fail(model_session_status::stream_mismatch, error,
                    "model session stage used a different CUDA stream");
    }
    if (!state.transaction_open) {
        return fail(model_session_status::no_transaction, error,
                    "stage_all requires begin()");
    }
    if (token_ids == nullptr || token_count == 0u ||
        token_count > kDecoderMaxSpeculativeTokens) {
        return fail(model_session_status::invalid_argument, error,
                    "invalid contiguous model sequence");
    }
    if (state.staged_layers != 0u || state.staged_tokens != 0u) {
        return fail(model_session_status::stage_already_complete, error,
                    "one token is already staged in this model transaction");
    }
    if (token_count > state.config.context_capacity ||
        position_count < state.committed_tokens + token_count ||
        position_count > state.config.context_capacity ||
        state.committed_tokens > state.config.context_capacity - token_count) {
        return fail(model_session_status::capacity_exceeded, error,
                    "RoPE position or committed context exceeds capacity");
    }
    const std::array<const void *, 4u> inputs{{
            residual_input_4x2560_f32,
            full_cos_64_f32,
            full_sin_64_f32,
            residual_output_4x2560_f32,
    }};
    for (const void *pointer : inputs) {
        if (!pointer_on_device(pointer, state.device)) {
            return fail(model_session_status::invalid_device_pointer, error,
                        "stage_all requires device pointers on the bound GPU");
        }
    }
    const std::size_t active_residual_bytes = token_count *
            kDecoderStreams * kDecoderHidden * sizeof(float);
    if (ranges_overlap(residual_input_4x2560_f32, active_residual_bytes,
                       residual_output_4x2560_f32, active_residual_bytes)) {
        return fail(model_session_status::invalid_device_pointer, error,
                    "model residual input and output must not overlap");
    }

    // IDs depend only on token history. Launch the sole PLE layer's first
    // gather before layer 0; its consumer still waits at layer 1. Rollback
    // must include this pending work even though layer 1 is not staged yet.
    for (std::size_t index = 0u; index < kModelSessionLayerCount; ++index) {
        if (!state.layers[index]->has_ple()) continue;
        std::string detail;
        const auto status = state.sessions[index]->prefetch_first_ple_token(
                token_ids[0], stream, &detail);
        if (status != decoder_layer_status::ok) {
            state.last_failed_layer = index;
            std::string rollback_detail;
            const auto rollback = state.rollback_open_layers(stream, nullptr, &rollback_detail);
            state.transaction_open = false;
            state.staged_tokens = state.staged_layers = 0u;
            if (rollback != model_session_status::ok) state.poisoned = true;
            return fail(rollback == model_session_status::ok
                                ? model_session_status::layer_error
                                : model_session_status::rollback_error,
                        error, layer_message(index, "early PLE prefetch", status, detail) +
                                (rollback_detail.empty() ? "" : "; rollback: " + rollback_detail));
        }
    }

    const float *source = residual_input_4x2560_f32;
    for (std::size_t index = 0u; index < kModelSessionLayerCount; ++index) {
        float *destination = nullptr;
        if (index + 1u == kModelSessionLayerCount) {
            destination = residual_output_4x2560_f32;
        } else {
            destination = state.residual_ping_pong[index % 2u];
        }
        std::string layer_error;
        const decoder_layer_status layer_status =
                state.sessions[index]->stage_sequence(
                        token_ids, token_count,
                        source,
                        full_cos_64_f32,
                        full_sin_64_f32,
                        position_count,
                        destination,
                        stream,
                        &layer_error);
        if (layer_status != decoder_layer_status::ok) {
            state.last_failed_layer = index;
            std::size_t rollback_layer = kModelSessionNoFailedLayer;
            std::string rollback_detail;
            const model_session_status rollback_status =
                    state.rollback_open_layers(
                            stream, &rollback_layer, &rollback_detail);
            state.transaction_open = false;
            state.staged_tokens = 0u;
            state.staged_layers = 0u;
            if (rollback_status != model_session_status::ok) {
                state.poisoned = true;
                if (rollback_layer != kModelSessionNoFailedLayer) {
                    state.last_failed_layer = rollback_layer;
                }
                return fail(
                        model_session_status::rollback_error,
                        error,
                        layer_message(index, "stage", layer_status,
                                      layer_error) +
                                "; cross-layer " + rollback_detail);
            }
            return fail(model_session_status::layer_error,
                        error,
                        layer_message(index, "stage", layer_status,
                                      layer_error));
        }
        state.staged_layers = index + 1u;
        source = destination;
    }
    state.staged_tokens = token_count;
    return validate(error);
}

model_session_status model_session_state::commit_all(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    impl &state = *impl_;
    if (!state.stream_matches(stream)) {
        return fail(model_session_status::stream_mismatch, error,
                    "model session commit used a different CUDA stream");
    }
    if (!state.transaction_open) {
        return fail(model_session_status::no_transaction, error,
                    "commit_all requires an open model transaction");
    }
    if (state.staged_tokens == 0u ||
        state.staged_tokens > kDecoderMaxSpeculativeTokens ||
        state.staged_layers != kModelSessionLayerCount) {
        return fail(model_session_status::invalid_state, error,
                    "commit_all requires one bounded sequence through all 48 layers");
    }
    const model_session_status validation = validate(error);
    if (validation != model_session_status::ok) return validation;

    std::array<decoder_layer_session *, kModelSessionLayerCount> raw{};
    for (std::size_t index = 0u; index < raw.size(); ++index) {
        raw[index] = state.sessions[index].get();
    }
    std::string prepare_error;
    const decoder_layer_status prepare_status =
            decoder_model_transaction::prepare_commit(
                    raw.data(), raw.size(), stream, &prepare_error);
    if (prepare_status != decoder_layer_status::ok) {
        std::size_t rollback_layer = kModelSessionNoFailedLayer;
        std::string rollback_detail;
        (void)state.rollback_open_layers(
                stream, &rollback_layer, &rollback_detail);
        state.staged_tokens = 0u;
        state.staged_layers = 0u;
        state.transaction_open = false;
        if (rollback_layer != kModelSessionNoFailedLayer) {
            state.last_failed_layer = rollback_layer;
            state.poisoned = true;
            return fail(model_session_status::rollback_error, error,
                        prepare_error + "; rollback: " + rollback_detail);
        }
        return fail(model_session_status::layer_error, error,
                    prepare_error.empty()
                            ? "decoder model prepare_commit failed"
                            : prepare_error);
    }
    if (!decoder_model_transaction::commit_prepared_noexcept(
                raw.data(), raw.size())) {
        std::size_t rollback_layer = kModelSessionNoFailedLayer;
        std::string rollback_detail;
        (void)state.rollback_open_layers(
                stream, &rollback_layer, &rollback_detail);
        std::size_t reset_layer = kModelSessionNoFailedLayer;
        std::string reset_detail;
        const bool reset_ok = state.reset_every_layer(
                stream, &reset_layer, &reset_detail);
        state.committed_tokens = 0u;
        state.staged_tokens = 0u;
        state.staged_layers = 0u;
        state.transaction_open = false;
        state.poisoned = true;
        state.last_failed_layer = reset_layer != kModelSessionNoFailedLayer
                ? reset_layer
                : rollback_layer;
        return fail(
                model_session_status::poisoned,
                error,
                reset_ok
                        ? "prepared decoder publication invariant failed; "
                          "session was reset and poisoned"
                        : "prepared decoder publication invariant failed; " +
                                  (rollback_detail.empty()
                                           ? reset_detail
                                           : rollback_detail + "; " +
                                                     reset_detail));
    }

    state.committed_tokens += state.staged_tokens;
    state.staged_tokens = 0u;
    state.staged_layers = 0u;
    state.transaction_open = false;
    state.last_failed_layer = kModelSessionNoFailedLayer;
    return validate(error);
}

model_session_status model_session_state::rollback_all(
        cudaStream_t stream,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    impl &state = *impl_;
    if (!state.stream_matches(stream)) {
        return fail(model_session_status::stream_mismatch, error,
                    "model session rollback used a different CUDA stream");
    }
    if (!state.transaction_open) {
        return fail(model_session_status::no_transaction, error,
                    "rollback_all requires an open model transaction");
    }
    std::size_t failed_layer = kModelSessionNoFailedLayer;
    std::string detail;
    const model_session_status rollback_status =
            state.rollback_open_layers(stream, &failed_layer, &detail);
    state.transaction_open = false;
    state.staged_tokens = 0u;
    state.staged_layers = 0u;
    if (rollback_status != model_session_status::ok) {
        state.poisoned = true;
        state.last_failed_layer = failed_layer;
        return fail(model_session_status::rollback_error, error, detail);
    }
    state.last_failed_layer = kModelSessionNoFailedLayer;
    return validate(error);
}

model_session_status model_session_state::reset(cudaStream_t stream,
                                                std::string *error) noexcept {
    if (!impl_ || !impl_->ready) {
        return fail(model_session_status::invalid_state, error,
                    "model session is not initialized");
    }
    impl &state = *impl_;
    if (!state.stream_matches(stream)) {
        return fail(model_session_status::stream_mismatch, error,
                    "model session reset used a different CUDA stream");
    }
    std::size_t rollback_layer = kModelSessionNoFailedLayer;
    std::string rollback_detail;
    (void)state.rollback_open_layers(stream, &rollback_layer,
                                     &rollback_detail);
    std::size_t reset_layer = kModelSessionNoFailedLayer;
    std::string reset_detail;
    const bool reset_ok =
            state.reset_every_layer(stream, &reset_layer, &reset_detail);
    state.committed_tokens = 0u;
    state.staged_tokens = 0u;
    state.staged_layers = 0u;
    state.transaction_open = false;
    state.last_failed_layer = reset_layer;
    state.poisoned = !reset_ok;
    if (!reset_ok) {
        return fail(model_session_status::reset_error, error, reset_detail);
    }
    state.last_failed_layer = kModelSessionNoFailedLayer;
    return validate(error);
}

model_session_view model_session_state::view() const noexcept {
    model_session_view result{};
    if (!impl_) return result;
    result.context_capacity = impl_->config.context_capacity;
    result.committed_tokens = impl_->committed_tokens;
    result.staged_tokens = impl_->staged_tokens;
    result.staged_layers = impl_->staged_layers;
    result.last_failed_layer = impl_->last_failed_layer;
    result.device = impl_->device;
    result.transaction_open = impl_->transaction_open;
    result.poisoned = impl_->poisoned;
    result.initialized = impl_->ready;
    return result;
}

bool model_session_state::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
