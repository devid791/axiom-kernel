#include "axiom/qwen4exp/text_model.hpp"

#include "axiom/qwen4exp_admission.hpp"
#include "axiom/qwen4exp/expert_bridge.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

using steady_clock = std::chrono::steady_clock;

double elapsed_ms(const steady_clock::time_point start) noexcept {
    return std::chrono::duration<double, std::milli>(
                   steady_clock::now() - start)
            .count();
}

void set_error(std::string *error, const std::string &message) noexcept {
    if (error == nullptr) return;
    try {
        *error = message;
    } catch (...) {
    }
}

text_model_status fail(text_model_status status,
                       std::string *error,
                       const std::string &message) noexcept {
    set_error(error, message);
    return status;
}

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

text_model_status map_session_status(model_session_status status) noexcept {
    switch (status) {
        case model_session_status::ok: return text_model_status::ok;
        case model_session_status::invalid_argument:
            return text_model_status::invalid_argument;
        case model_session_status::unsupported_config:
        case model_session_status::unsupported_device:
            return text_model_status::unsupported_config;
        case model_session_status::allocation_failure:
            return text_model_status::allocation_failure;
        case model_session_status::capacity_exceeded:
            return text_model_status::capacity_exceeded;
        case model_session_status::rollback_error:
        case model_session_status::poisoned:
            return text_model_status::transaction_error;
        case model_session_status::cuda_error:
            return text_model_status::cuda_error;
        case model_session_status::invalid_device_pointer:
        case model_session_status::stream_mismatch:
        case model_session_status::invalid_state:
        case model_session_status::transaction_open:
        case model_session_status::no_transaction:
        case model_session_status::stage_already_complete:
        case model_session_status::layer_error:
        case model_session_status::reset_error:
        case model_session_status::size_overflow:
            return text_model_status::session_error;
    }
    return text_model_status::session_error;
}

text_model_status map_output_status(output_head_status status) noexcept {
    switch (status) {
        case output_head_status::ok: return text_model_status::ok;
        case output_head_status::invalid_argument:
            return text_model_status::invalid_argument;
        case output_head_status::unsupported_config:
        case output_head_status::unsupported_device:
            return text_model_status::unsupported_config;
        case output_head_status::allocation_failure:
            return text_model_status::allocation_failure;
        case output_head_status::device_rejected_input:
            return text_model_status::token_rejected;
        case output_head_status::cuda_error:
            return text_model_status::cuda_error;
        case output_head_status::tensor_not_found:
        case output_head_status::tensor_contract_mismatch:
        case output_head_status::checkpoint_io_error:
            return text_model_status::checkpoint_error;
        case output_head_status::size_overflow:
        case output_head_status::invalid_device_pointer:
        case output_head_status::linear_error:
            return text_model_status::output_head_error;
    }
    return text_model_status::output_head_error;
}

text_model_status map_residency_status(
        decoder_residency_status status) noexcept {
    switch (status) {
        case decoder_residency_status::ok:
            return text_model_status::ok;
        case decoder_residency_status::invalid_argument:
            return text_model_status::invalid_argument;
        case decoder_residency_status::unsupported_device:
            return text_model_status::unsupported_device;
        case decoder_residency_status::invalid_checkpoint:
            return text_model_status::checkpoint_error;
        case decoder_residency_status::arithmetic_overflow:
        case decoder_residency_status::budget_exceeded:
        case decoder_residency_status::busy:
        case decoder_residency_status::invalid_state:
            return text_model_status::residency_error;
        case decoder_residency_status::allocation_failure:
            return text_model_status::allocation_failure;
        case decoder_residency_status::layer_load_failure:
            return text_model_status::decoder_layer_error;
        case decoder_residency_status::cuda_error:
            return text_model_status::cuda_error;
    }
    return text_model_status::residency_error;
}

text_model_status map_rope_status(rope::status status) noexcept {
    switch (status) {
        case rope::status::ok: return text_model_status::ok;
        case rope::status::null_pointer:
        case rope::status::invalid_argument:
            return text_model_status::invalid_argument;
        case rope::status::invalid_config:
        case rope::status::unsupported_device:
            return text_model_status::unsupported_config;
        case rope::status::cuda_failure:
            return text_model_status::cuda_error;
        case rope::status::arithmetic_overflow:
        case rope::status::invalid_state:
        case rope::status::non_finite:
            return text_model_status::rope_error;
    }
    return text_model_status::rope_error;
}

text_model_status map_multimodal_status(
        multimodal_text_status status) noexcept {
    switch (status) {
        case multimodal_text_status::ok: return text_model_status::ok;
        case multimodal_text_status::invalid_argument:
            return text_model_status::invalid_argument;
        case multimodal_text_status::unsupported_contract:
        case multimodal_text_status::malformed_vision_span:
        case multimodal_text_status::vision_token_mismatch:
            return text_model_status::unsupported_config;
        case multimodal_text_status::invalid_mrope:
            return text_model_status::rope_error;
        case multimodal_text_status::size_overflow:
            return text_model_status::capacity_exceeded;
        case multimodal_text_status::unsupported_device:
            return text_model_status::unsupported_device;
        case multimodal_text_status::invalid_device_pointer:
        case multimodal_text_status::non_finite_embedding:
            return text_model_status::token_rejected;
        case multimodal_text_status::cuda_error:
            return text_model_status::cuda_error;
    }
    return text_model_status::invalid_state;
}

bool generation_valid(const text_generation_config &generation) noexcept {
    if (generation.mode == text_selection_mode::argmax) return true;
    return generation.mode == text_selection_mode::sample &&
           std::isfinite(generation.sampling.temperature) &&
           generation.sampling.temperature > 0.0F &&
           generation.sampling.temperature <= 100.0F;
}

}  // namespace

struct resident_text_model::impl {
    text_model_config config{};
    text_model_metrics metrics{};
    std::string model_root;
    admission_report admission{};
    std::unique_ptr<checkpoint_catalog> catalog;
    std::shared_ptr<const expert_checkpoint_catalog> expert_catalog;
    std::shared_ptr<expert_slot_arena> expert_arena;
    std::unique_ptr<decoder_residency_manager> decoder_pool;
    std::unique_ptr<resident_output_head> output_head;
    rope::device_plan rope_plan{};
    std::mutex execution_mutex;
    int device = -1;
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = device >= 0 && (!have_previous || previous != device);
        if (changed) (void)cudaSetDevice(device);
        decoder_pool.reset();
        expert_arena.reset();
        expert_catalog.reset();
        output_head.reset();
        if (rope_plan.initialized) (void)rope::release_device_plan(&rope_plan);
        catalog.reset();
        if (changed && have_previous && previous >= 0) {
            (void)cudaSetDevice(previous);
        }
    }
};

struct text_session::impl {
    resident_text_model::impl *owner = nullptr;
    text_session_config config{};
    text_session_metrics metrics{};
    std::unique_ptr<model_session_state> decoder;
    output_head_scratch scratch{};
    std::vector<void *> allocations;
    float *decoder_output = nullptr;
    const float *latest_decoder_hidden = nullptr;
    float *rope_cosine = nullptr;
    float *rope_sine = nullptr;
    rope::status *rope_device_status = nullptr;
    std::uint32_t *mrope_position = nullptr;
    std::uint32_t *input_token = nullptr;
    std::uint32_t *selected_token = nullptr;
    float *selected_logit = nullptr;
    cudaStream_t stream = nullptr;
    int device = -1;
    std::uint32_t next_mrope_text_position = 0u;
    bool multimodal_mrope_active = false;
    bool next_mrope_text_position_available = false;
    steady_clock::time_point speculative_prefill_started{};
    bool speculative_prefill_active = false;
    bool rope_table_text_pristine = true;
    std::size_t staged_speculative_tokens = 0u;
    steady_clock::time_point staged_speculative_started{};
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = device >= 0 && (!have_previous || previous != device);
        if (changed) (void)cudaSetDevice(device);
        decoder.reset();
        for (auto iterator = allocations.rbegin(); iterator != allocations.rend();
             ++iterator) {
            if (*iterator != nullptr) (void)cudaFree(*iterator);
        }
        if (changed && have_previous && previous >= 0) {
            (void)cudaSetDevice(previous);
        }
    }

    bool allocate_bytes(void **destination,
                        std::size_t bytes,
                        const char *label,
                        std::string *error) noexcept {
        if (destination == nullptr || bytes == 0u) {
            set_error(error, std::string("invalid allocation request for ") + label);
            return false;
        }
        *destination = nullptr;
        void *pointer = nullptr;
        const cudaError_t status = cudaMalloc(&pointer, bytes);
        if (status != cudaSuccess) {
            set_error(error, std::string("cudaMalloc(") + label + "): " +
                                     cudaGetErrorString(status));
            return false;
        }
        try {
            allocations.push_back(pointer);
        } catch (...) {
            (void)cudaFree(pointer);
            set_error(error, std::string("allocation registry failed for ") + label);
            return false;
        }
        *destination = pointer;
        return true;
    }

    template <typename T>
    bool allocate(T **destination,
                  std::size_t bytes,
                  const char *label,
                  std::string *error) noexcept {
        return allocate_bytes(reinterpret_cast<void **>(destination), bytes,
                              label, error);
    }

    text_model_status restore_text_rope_table(std::string *error) noexcept {
        if (owner == nullptr || !owner->ready || rope_cosine == nullptr ||
            rope_sine == nullptr || rope_device_status == nullptr) {
            return fail(text_model_status::invalid_state, error,
                        "cannot restore the native text RoPE table");
        }
        rope::status status =
                rope::reset_device_status(rope_device_status, stream);
        if (status == rope::status::ok) {
            status = rope::generate_text_cuda(
                    &owner->rope_plan, 0u, config.context_capacity,
                    rope_cosine, rope_sine, rope_device_status, stream);
        }
        rope::status collected = rope::status::cuda_failure;
        if (status == rope::status::ok) {
            status = rope::collect_device_status(
                    rope_device_status, &collected, stream);
        }
        if (status != rope::status::ok || collected != rope::status::ok) {
            const rope::status reported = status != rope::status::ok
                    ? status
                    : collected;
            return fail(map_rope_status(reported), error,
                        std::string("native text RoPE restoration failed: ") +
                                rope::status_string(reported));
        }
        rope_table_text_pristine = true;
        return text_model_status::ok;
    }

    text_model_status install_mrope_row(
            std::size_t row,
            const std::uint32_t *positions_t_h_w,
            std::string *error) noexcept {
        if (owner == nullptr || !owner->ready || positions_t_h_w == nullptr ||
            mrope_position == nullptr || row >= config.context_capacity) {
            return fail(text_model_status::invalid_argument, error,
                        "invalid MRoPE row installation request");
        }
        for (std::size_t axis = 0u; axis < kMultimodalPositionAxes; ++axis) {
            if (positions_t_h_w[axis] >= rope::kMaxContext) {
                return fail(text_model_status::rope_error, error,
                            "MRoPE position exceeds the native 262K domain");
            }
        }
        if (cudaMemcpyAsync(mrope_position, positions_t_h_w,
                            kMultimodalPositionAxes * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cannot upload one exact MRoPE position row");
        }
        rope::status status =
                rope::reset_device_status(rope_device_status, stream);
        if (status == rope::status::ok) {
            status = rope::generate_mrope_cuda(
                    &owner->rope_plan, mrope_position, 1u,
                    rope_cosine + row * rope::kRotaryDim,
                    rope_sine + row * rope::kRotaryDim,
                    rope_device_status, stream);
        }
        rope::status collected = rope::status::cuda_failure;
        if (status == rope::status::ok) {
            status = rope::collect_device_status(
                    rope_device_status, &collected, stream);
        }
        if (status != rope::status::ok || collected != rope::status::ok) {
            const rope::status reported = status != rope::status::ok
                    ? status
                    : collected;
            return fail(map_rope_status(reported), error,
                        std::string("exact MRoPE row generation failed: ") +
                                rope::status_string(reported));
        }
        rope_table_text_pristine = false;
        return text_model_status::ok;
    }

    text_model_status abort_transaction(text_model_status cause,
                                        const std::string &detail,
                                        std::string *error) noexcept {
        if (decoder != nullptr && decoder->view().transaction_open) {
            std::string rollback_error;
            const model_session_status rollback =
                    decoder->rollback_all(stream, &rollback_error);
            ++metrics.rolled_back_transactions;
            if (rollback != model_session_status::ok) {
                return fail(text_model_status::transaction_error, error,
                            detail + "; global rollback failed: " +
                                    (rollback_error.empty()
                                             ? model_session_status_string(rollback)
                                             : rollback_error));
            }
        } else {
            /* stage_all() performs fail-closed cross-layer rollback itself. */
            ++metrics.rolled_back_transactions;
        }
        return fail(cause, error, detail);
    }

    text_model_status run_token(std::uint32_t token,
                                const float *projected_visual_embedding,
                                const std::uint32_t *mrope_positions_t_h_w,
                                bool produce_output,
                                const text_generation_config &generation,
                                text_token_result *result,
                                std::string *error) noexcept {
        if (!ready || owner == nullptr || !owner->ready || decoder == nullptr) {
            return fail(text_model_status::invalid_state, error,
                        "text session or resident model is not initialized");
        }
        if (produce_output && (result == nullptr || !generation_valid(generation))) {
            return fail(text_model_status::invalid_argument, error,
                        "output token destination or generation config is invalid");
        }

        const model_session_view before = decoder->view();
        if (!before.initialized || before.poisoned) {
            return fail(text_model_status::invalid_state, error,
                        "decoder session is not usable");
        }
        if (before.committed_tokens >= config.context_capacity) {
            return fail(text_model_status::capacity_exceeded, error,
                        "text session reached its bounded context capacity");
        }

        if (mrope_positions_t_h_w != nullptr) {
            const text_model_status rope_status = install_mrope_row(
                    static_cast<std::size_t>(before.committed_tokens),
                    mrope_positions_t_h_w, error);
            if (rope_status != text_model_status::ok) return rope_status;
        }

        const steady_clock::time_point started = steady_clock::now();
        std::unique_lock<std::mutex> execution_lock(owner->execution_mutex);
        std::string detail;
        model_session_status session_status = decoder->begin(stream, &detail);
        if (session_status != model_session_status::ok) {
            return fail(map_session_status(session_status), error,
                        detail.empty() ? model_session_status_string(session_status)
                                       : detail);
        }

        if (cudaMemsetAsync(scratch.device_status, 0,
                            scratch.device_status_bytes, stream) != cudaSuccess ||
            cudaMemcpyAsync(input_token, &token, sizeof(token),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return abort_transaction(text_model_status::cuda_error,
                                     "cannot stage input token on the GPU", error);
        }

        output_head_status output_status = output_head_status::ok;
        if (projected_visual_embedding == nullptr) {
            output_status = owner->output_head->embedding_repeat(
                    input_token, 1u, scratch, scratch.hyper_input, stream);
            if (output_status != output_head_status::ok) {
                return abort_transaction(
                        map_output_status(output_status),
                        std::string("embedding_repeat failed: ") +
                                output_head_status_string(output_status),
                        error);
            }
        } else {
            const multimodal_text_status visual_status =
                    repeat_projected_vision_embedding_cuda(
                            projected_visual_embedding, scratch.hyper_input,
                            scratch.device_status, device, stream);
            if (visual_status != multimodal_text_status::ok) {
                return abort_transaction(
                        map_multimodal_status(visual_status),
                        std::string("projected vision injection failed: ") +
                                multimodal_text_status_string(visual_status),
                        error);
            }
        }

        session_status = decoder->stage_all(
                static_cast<std::int64_t>(token), scratch.hyper_input,
                rope_cosine, rope_sine,
                static_cast<std::size_t>(before.committed_tokens + 1u),
                decoder_output, stream, &detail);
        if (session_status != model_session_status::ok) {
            return abort_transaction(
                    map_session_status(session_status),
                    detail.empty()
                            ? std::string("48-layer stage failed: ") +
                                      model_session_status_string(session_status)
                            : detail,
                    error);
        }

        if (produce_output) {
            output_status = owner->output_head->global_mix(
                    decoder_output, 1u, scratch, scratch.mixed_hidden, stream);
            if (output_status == output_head_status::ok) {
                output_status = owner->output_head->logits(
                        scratch.mixed_hidden, 1u, scratch, scratch.logits, stream);
            }
            if (output_status == output_head_status::ok) {
                if (generation.mode == text_selection_mode::argmax) {
                    output_status = owner->output_head->argmax(
                            scratch.logits, 1u, selected_token, selected_logit,
                            stream);
                } else {
                    output_status = owner->output_head->sample(
                            scratch.logits, 1u, generation.sampling,
                            selected_token, selected_logit, stream);
                }
            }
            if (output_status != output_head_status::ok) {
                return abort_transaction(
                        map_output_status(output_status),
                        std::string("model output tail failed: ") +
                                output_head_status_string(output_status),
                        error);
            }
        }

        output_status = owner->output_head->collect_device_status(scratch, stream);
        if (output_status != output_head_status::ok) {
            return abort_transaction(
                    map_output_status(output_status),
                    std::string("device status rejected token: ") +
                            output_head_status_string(output_status),
                    error);
        }

        std::uint32_t host_token = 0u;
        float host_logit = 0.0F;
        if (produce_output &&
            (cudaMemcpyAsync(&host_token, selected_token, sizeof(host_token),
                             cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
             cudaMemcpyAsync(&host_logit, selected_logit, sizeof(host_logit),
                             cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
             cudaStreamSynchronize(stream) != cudaSuccess)) {
            return abort_transaction(text_model_status::cuda_error,
                                     "cannot collect selected next token", error);
        }

        session_status = decoder->commit_all(stream, &detail);
        if (session_status != model_session_status::ok) {
            return abort_transaction(
                    map_session_status(session_status),
                    detail.empty()
                            ? std::string("model-wide commit failed: ") +
                                      model_session_status_string(session_status)
                            : detail,
                    error);
        }

        const double token_ms = elapsed_ms(started);
        latest_decoder_hidden = decoder_output;
        ++metrics.committed_transactions;
        if (produce_output) {
            ++metrics.generated_tokens;
            text_token_result published{};
            published.token_id = host_token;
            published.selected_logit = host_logit;
            published.committed_context = decoder->view().committed_tokens;
            published.latency_ms = token_ms;
            published.sampled = generation.mode == text_selection_mode::sample;
            *result = published;
        }
        return text_model_status::ok;
    }
};

const char *text_model_status_string(text_model_status status) noexcept {
    switch (status) {
        case text_model_status::ok: return "ok";
        case text_model_status::invalid_argument: return "invalid_argument";
        case text_model_status::unsupported_config: return "unsupported_config";
        case text_model_status::unsupported_device: return "unsupported_device";
        case text_model_status::admission_error: return "admission_error";
        case text_model_status::checkpoint_error: return "checkpoint_error";
        case text_model_status::allocation_failure: return "allocation_failure";
        case text_model_status::residency_error: return "residency_error";
        case text_model_status::decoder_layer_error:
            return "decoder_layer_error";
        case text_model_status::output_head_error: return "output_head_error";
        case text_model_status::rope_error: return "rope_error";
        case text_model_status::session_error: return "session_error";
        case text_model_status::capacity_exceeded: return "capacity_exceeded";
        case text_model_status::token_rejected: return "token_rejected";
        case text_model_status::transaction_error: return "transaction_error";
        case text_model_status::cuda_error: return "cuda_error";
        case text_model_status::invalid_state: return "invalid_state";
    }
    return "unknown";
}

text_model_status text_model_validate_config(
        const text_model_config &config) noexcept {
    if (config.max_context != kTextModelNativeContext ||
        config.upload_chunk_bytes < 256u * 1024u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes % 256u != 0u ||
        config.moe_ram_capacity_bytes_per_layer == 0u ||
        config.moe_prefetch_workers_per_layer == 0u ||
        (config.expert_host_cold_cache_bytes != 0u &&
         config.expert_gpu_cache_slots == 0u) ||
        (config.expert_gpu_cache_slots != 0u &&
         (config.expert_gpu_cache_slots < kMoeTopK ||
          config.expert_gpu_cache_slots > kTextModelLayerCount * kMoeExperts)) ||
        config.decoder_hard_limit_bytes == 0u ||
        config.decoder_hard_limit_bytes > kDecoderResidencyHardLimit ||
        config.decoder_safety_margin_bytes == 0u ||
        config.decoder_safety_margin_bytes >=
                config.decoder_hard_limit_bytes ||
        config.decoder_adapter_allowance_bytes_per_layer == 0u) {
        return text_model_status::unsupported_config;
    }
    return text_model_status::ok;
}

text_model_status text_session_validate_config(
        const text_session_config &config) noexcept {
    if (config.context_capacity == 0u ||
        config.context_capacity > kTextModelNativeContext) {
        return text_model_status::unsupported_config;
    }
    return text_model_status::ok;
}

resident_text_model::resident_text_model() = default;
resident_text_model::~resident_text_model() = default;
resident_text_model::resident_text_model(resident_text_model &&) noexcept =
        default;
resident_text_model &resident_text_model::operator=(
        resident_text_model &&) noexcept = default;

text_model_status resident_text_model::load(
        const std::string &model_root,
        const text_model_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<resident_text_model> *out,
        std::string *error) noexcept {
    if (out == nullptr || model_root.empty()) {
        return fail(text_model_status::invalid_argument, error,
                    "model root or output destination is invalid");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const text_model_status config_status = text_model_validate_config(config);
    if (config_status != text_model_status::ok) {
        return fail(config_status, error,
                    "qwen4_exp text model requires the exact native 262K contract");
    }

    const steady_clock::time_point started = steady_clock::now();
    try {
        auto result = std::make_unique<resident_text_model>();
        result->impl_ = std::make_unique<impl>();
        impl &state = *result->impl_;
        state.config = config;
        state.model_root = model_root;

        std::string detail;
        if (!inspect_checkpoint(model_root, !config.verify_payload_hashes,
                                &state.admission, &detail)) {
            return fail(text_model_status::admission_error, error,
                        detail.empty() ? "pinned checkpoint admission failed"
                                       : detail);
        }
        const bool structurally_admitted = state.admission.identity_valid &&
                                           state.admission.config_valid &&
                                           state.admission.quantization_valid &&
                                           state.admission.tensor_contract_valid &&
                                           state.admission.metadata_admitted &&
                                           state.admission.checkpoint_complete;
        const bool payload_admitted = !config.verify_payload_hashes ||
                                      (state.admission.payload_hashes_valid &&
                                       state.admission.checkpoint_admitted);
        if (!structurally_admitted || !payload_admitted) {
            return fail(text_model_status::admission_error, error,
                        "checkpoint does not match the pinned qwen4_exp identity");
        }

        if (!checkpoint_catalog::open(model_root, &state.catalog, &detail) ||
            state.catalog == nullptr) {
            return fail(text_model_status::checkpoint_error, error,
                        detail.empty() ? "cannot open pinned checkpoint catalog"
                                       : detail);
        }
        const decoder_layer_status schedule =
                decoder_layer_validate_checkpoint_schedule(*state.catalog, &detail);
        if (schedule != decoder_layer_status::ok) {
            return fail(text_model_status::checkpoint_error, error,
                        detail.empty()
                                ? std::string("decoder schedule rejected: ") +
                                          decoder_layer_status_string(schedule)
                                : detail);
        }

        detail.clear();
        if (!expert_checkpoint_catalog::open(
                    model_root, pinned_expert_checkpoint_identity(),
                    &state.expert_catalog, &detail) ||
            state.expert_catalog == nullptr) {
            return fail(text_model_status::checkpoint_error, error,
                        detail.empty()
                                ? "cannot open shared routed-expert catalog"
                                : detail);
        }

        if (cudaGetDevice(&state.device) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cudaGetDevice failed during text model load");
        }
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, state.device) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cudaGetDeviceProperties failed during text model load");
        }
        if (properties.major != 12) {
            return fail(text_model_status::unsupported_device, error,
                        "qwen4_exp native provider requires an SM120 GPU");
        }

        std::size_t clean_free_bytes = 0u;
        std::size_t cuda_total_bytes = 0u;
        if (cudaMemGetInfo(&clean_free_bytes, &cuda_total_bytes) !=
            cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cudaMemGetInfo failed before text model allocations");
        }

        const rope::status rope_status = rope::initialize_device_plan(
                &state.rope_plan, rope::checkpoint_config,
                initialization_stream);
        if (rope_status != rope::status::ok) {
            return fail(map_rope_status(rope_status), error,
                        std::string("native RoPE plan load failed: ") +
                                rope::status_string(rope_status));
        }

        output_head_config head_config{};
        head_config.max_tokens = kDecoderMaxSpeculativeTokens;
        head_config.upload_chunk_bytes = config.upload_chunk_bytes;
        head_config.blas_workspace_bytes = config.blas_workspace_bytes;
        const output_head_status head_status = resident_output_head::load(
                *state.catalog, head_config, initialization_stream,
                &state.output_head, &detail);
        if (head_status != output_head_status::ok ||
            state.output_head == nullptr || !state.output_head->initialized()) {
            return fail(map_output_status(head_status), error,
                        detail.empty()
                                ? std::string("output head load failed: ") +
                                          output_head_status_string(head_status)
                                : detail);
        }

        detail.clear();
        const bool arena_created = config.expert_gpu_cache_slots == 0u
                ? expert_slot_arena::create(state.device, kMoeTopK,
                                           &state.expert_arena, &detail)
                : expert_slot_arena::create(state.device, kMoeTopK,
                                           config.expert_gpu_cache_slots,
                                           config.expert_host_cold_cache_bytes,
                                           &state.expert_arena, &detail);
        if (!arena_created ||
            state.expert_arena == nullptr) {
            return fail(text_model_status::allocation_failure, error,
                        detail.empty()
                                ? "shared expert-slot arena allocation failed"
                                : detail);
        }
        const expert_slot_arena_metrics arena_metrics =
                state.expert_arena->metrics();
        if (arena_metrics.device != state.device ||
            arena_metrics.bounded_slots != kMoeTopK ||
            (config.expert_gpu_cache_slots == 0u
                     ? arena_metrics.capacity_bytes != 27648160u
                     : (!arena_metrics.persistent_cache ||
                        arena_metrics.physical_slots != config.expert_gpu_cache_slots))) {
            return fail(text_model_status::invalid_state, error,
                        "shared expert-slot arena contract mismatch");
        }

        decoder_residency_config residency_config{};
        residency_config.hard_limit_bytes = config.decoder_hard_limit_bytes;
        residency_config.safety_margin_bytes =
                config.decoder_safety_margin_bytes;
        residency_config.clean_free_bytes = clean_free_bytes;
        residency_config.max_context = config.max_context;
        residency_config.upload_chunk_bytes = config.upload_chunk_bytes;
        residency_config.blas_workspace_bytes = config.blas_workspace_bytes;
        residency_config.moe_ram_capacity_bytes_per_layer =
                config.moe_ram_capacity_bytes_per_layer;
        residency_config.moe_prefetch_workers_per_layer =
                config.moe_prefetch_workers_per_layer;
        residency_config.shared_expert_arena = state.expert_arena;
        residency_config.shared_expert_catalog = state.expert_catalog;
        residency_config.enable_ple = true;
        residency_config.moe_owned_direct_pread = config.moe_owned_direct_pread;
        residency_config.adapter_allowance_bytes_per_layer =
                config.decoder_adapter_allowance_bytes_per_layer;

        detail.clear();
        decoder_residency_status residency_status =
                decoder_residency_manager::create(
                        *state.catalog, residency_config,
                        initialization_stream, &state.decoder_pool, &detail);
        if (residency_status != decoder_residency_status::ok ||
            state.decoder_pool == nullptr ||
            !state.decoder_pool->initialized()) {
            return fail(map_residency_status(residency_status), error,
                        detail.empty()
                                ? std::string("decoder residency create failed: ") +
                                          decoder_residency_status_string(
                                                  residency_status)
                                : detail);
        }
        if (state.decoder_pool->mode() ==
            decoder_residency_mode::resident_all) {
            detail.clear();
            residency_status = state.decoder_pool->load_all(
                    initialization_stream, &detail);
            if (residency_status != decoder_residency_status::ok) {
                return fail(map_residency_status(residency_status), error,
                            detail.empty()
                                    ? std::string(
                                              "decoder residency load failed: ") +
                                              decoder_residency_status_string(
                                                      residency_status)
                                    : detail);
            }
        }
        if (cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "complete text model initialization stream failed");
        }

        state.metrics.load_ms = elapsed_ms(started);
        state.metrics.decoder_residency = state.decoder_pool->metrics();
        state.metrics.admitted_layers =
                state.metrics.decoder_residency.all_48_admitted
                        ? kTextModelLayerCount
                        : 0u;
        state.metrics.loaded_layers =
                state.metrics.decoder_residency.resident_layers;
        state.metrics.output_head_resident_bytes =
                state.output_head->resident_weight_bytes();
        state.metrics.expert_arena_resident_bytes =
                state.expert_arena->metrics().capacity_bytes;
        state.metrics.checkpoint_payload_bytes =
                state.admission.tensor_payload_bytes;
        state.metrics.payload_hashes_verified =
                state.admission.payload_hashes_valid;
        state.ready = true;
        *out = std::move(result);
        return text_model_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(text_model_status::allocation_failure, error,
                    "host allocation failed during complete text model load");
    } catch (const std::exception &exception) {
        return fail(text_model_status::invalid_state, error,
                    std::string("text model load exception: ") +
                            exception.what());
    } catch (...) {
        return fail(text_model_status::invalid_state, error,
                    "unknown text model load exception");
    }
}

text_model_status resident_text_model::create_session(
        const text_session_config &config,
        cudaStream_t stream,
        std::unique_ptr<text_session> *out,
        std::string *error) noexcept {
    if (out == nullptr || impl_ == nullptr || !impl_->ready) {
        return fail(text_model_status::invalid_argument, error,
                    "resident text model or session output is invalid");
    }
    out->reset();
    const text_model_status config_status = text_session_validate_config(config);
    if (config_status != text_model_status::ok ||
        config.context_capacity > impl_->config.max_context) {
        return fail(text_model_status::unsupported_config, error,
                    "session context must be in [1,262144]");
    }
    try {
        int active_device = -1;
        if (cudaGetDevice(&active_device) != cudaSuccess ||
            active_device != impl_->device) {
            return fail(text_model_status::unsupported_device, error,
                        "text session must be created on the model GPU");
        }

        auto result = std::make_unique<text_session>();
        result->impl_ = std::make_unique<text_session::impl>();
        text_session::impl &session = *result->impl_;
        session.owner = impl_.get();
        session.config = config;
        session.stream = stream;
        session.device = active_device;

        if (impl_->decoder_pool == nullptr ||
            !impl_->decoder_pool->initialized()) {
            return fail(text_model_status::invalid_state, error,
                        "decoder residency pool is not initialized");
        }
        if (!impl_->decoder_pool->all_48_resident()) {
            return fail(text_model_status::residency_error, error,
                        "bounded decoder residency requires the shared "
                        "session binding contract");
        }
        const model_decoder_layers layer_pointers =
                impl_->decoder_pool->resident_layer_array();
        model_session_config decoder_config{};
        decoder_config.context_capacity = config.context_capacity;
        std::string detail;
        const model_session_status decoder_status = model_session_state::create(
                layer_pointers, decoder_config, stream, &session.decoder,
                &detail);
        if (decoder_status != model_session_status::ok ||
            session.decoder == nullptr || !session.decoder->initialized()) {
            return fail(map_session_status(decoder_status), error,
                        detail.empty()
                                ? std::string("model session creation failed: ") +
                                          model_session_status_string(decoder_status)
                                : detail);
        }

        const output_head_workspace_requirements requirements =
                impl_->output_head->workspace_requirements();
        auto allocate = [&](auto **pointer, std::size_t bytes,
                            const char *label) {
            return session.allocate(pointer, bytes, label, error);
        };
        if (!allocate(&session.scratch.hyper_input,
                      requirements.hyper_input_f32_bytes, "head hyper input") ||
            !allocate(&session.scratch.normalized,
                      requirements.normalized_f32_bytes, "head normalized") ||
            !allocate(&session.scratch.low_rank,
                      requirements.low_rank_f32_bytes, "head low rank") ||
            !allocate(&session.scratch.mix_logits,
                      requirements.mix_logits_f32_bytes, "head mix logits") ||
            !allocate(&session.scratch.mixed_hidden,
                      requirements.mixed_hidden_f32_bytes, "head mixed hidden") ||
            !allocate(&session.scratch.linear_input_bf16,
                      requirements.linear_input_bf16_bytes,
                      "head BF16 linear input") ||
            !allocate(&session.scratch.logits,
                      requirements.logits_f32_bytes, "full vocabulary logits") ||
            !session.allocate_bytes(&session.scratch.blas_workspace,
                                    requirements.blas_workspace_bytes,
                                    "head BLAS workspace", error) ||
            !allocate(&session.scratch.device_status,
                      requirements.device_status_bytes, "head device status") ||
            !allocate(&session.decoder_output,
                      kDecoderMaxSpeculativeTokens * kDecoderResidual *
                              sizeof(float),
                      "decoder residual output") ||
            !allocate(&session.input_token,
                      kDecoderMaxSpeculativeTokens * sizeof(std::uint32_t),
                      "input token") ||
            !allocate(&session.selected_token,
                      kDecoderMaxSpeculativeTokens * sizeof(std::uint32_t),
                      "selected token") ||
            !allocate(&session.selected_logit,
                      kDecoderMaxSpeculativeTokens * sizeof(float),
                      "selected logit") ||
            !allocate(&session.rope_device_status, sizeof(rope::status),
                      "RoPE device status") ||
            !allocate(&session.mrope_position,
                      kMultimodalPositionAxes * sizeof(std::uint32_t),
                      "one MRoPE position row")) {
            return text_model_status::allocation_failure;
        }
        session.scratch.linear_input_bf16_bytes =
                requirements.linear_input_bf16_bytes;
        session.scratch.blas_workspace_bytes =
                requirements.blas_workspace_bytes;
        session.scratch.device_status_bytes = requirements.device_status_bytes;

        std::size_t rope_elements = 0u;
        std::size_t rope_bytes = 0u;
        if (!checked_multiply(config.context_capacity, rope::kRotaryDim,
                              &rope_elements) ||
            !checked_multiply(rope_elements, sizeof(float), &rope_bytes) ||
            !allocate(&session.rope_cosine, rope_bytes, "native RoPE cosine") ||
            !allocate(&session.rope_sine, rope_bytes, "native RoPE sine")) {
            return fail(text_model_status::allocation_failure, error,
                        "bounded native RoPE table allocation failed");
        }

        rope::status rope_status = rope::reset_device_status(
                session.rope_device_status, stream);
        if (rope_status == rope::status::ok) {
            rope_status = rope::generate_text_cuda(
                    &impl_->rope_plan, 0u, config.context_capacity,
                    session.rope_cosine, session.rope_sine,
                    session.rope_device_status, stream);
        }
        rope::status collected = rope::status::cuda_failure;
        if (rope_status == rope::status::ok) {
            rope_status = rope::collect_device_status(
                    session.rope_device_status, &collected, stream);
        }
        if (rope_status != rope::status::ok || collected != rope::status::ok) {
            const rope::status reported = rope_status != rope::status::ok
                    ? rope_status
                    : collected;
            return fail(map_rope_status(reported), error,
                        std::string("native text RoPE table failed: ") +
                                rope::status_string(reported));
        }
        if (cudaMemsetAsync(session.scratch.device_status, 0,
                            session.scratch.device_status_bytes, stream) !=
                    cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "text session initialization stream failed");
        }

        session.latest_decoder_hidden = session.decoder_output;
        session.ready = true;
        *out = std::move(result);
        return text_model_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(text_model_status::allocation_failure, error,
                    "host allocation failed during text session creation");
    } catch (const std::exception &exception) {
        return fail(text_model_status::invalid_state, error,
                    std::string("text session creation exception: ") +
                            exception.what());
    } catch (...) {
        return fail(text_model_status::invalid_state, error,
                    "unknown text session creation exception");
    }
}

const text_model_config &resident_text_model::config() const noexcept {
    static const text_model_config empty{};
    return impl_ ? impl_->config : empty;
}

const text_model_metrics &resident_text_model::metrics() const noexcept {
    static const text_model_metrics empty{};
    return impl_ ? impl_->metrics : empty;
}

expert_slot_arena_metrics resident_text_model::expert_cache_metrics() const noexcept {
    return impl_ && impl_->expert_arena
            ? impl_->expert_arena->metrics() : expert_slot_arena_metrics{};
}

ExpertPagerMetrics resident_text_model::host_expert_cache_metrics() const noexcept {
    ExpertPagerMetrics result{};
    if (!impl_ || !impl_->decoder_pool) return result;
    // Resident layers are retained for the model lifetime. The pool locks
    // only to copy its fixed-size pointer array; each pager locks its snapshot.
    const auto layers = impl_->decoder_pool->resident_layer_array();
    for (const decoder_layer *layer : layers) {
        if (layer == nullptr || !layer->initialized()) continue;
        const ExpertPagerMetrics pager = layer->expert_pager_metrics();
        result.requests += pager.requests;
        result.prefetches += pager.prefetches;
        result.ram_hits += pager.ram_hits;
        result.ram_misses += pager.ram_misses;
        result.deduplicated_waits += pager.deduplicated_waits;
        result.evictions += pager.evictions;
        result.nvme_reads += pager.nvme_reads;
        result.nvme_bytes_read += pager.nvme_bytes_read;
        result.nvme_read_latency_ns += pager.nvme_read_latency_ns;
        result.checksum_failures += pager.checksum_failures;
        result.io_failures += pager.io_failures;
        result.gpu_uploads += pager.gpu_uploads;
        result.gpu_upload_bytes += pager.gpu_upload_bytes;
        result.gpu_releases += pager.gpu_releases;
        result.gpu_release_failures += pager.gpu_release_failures;
        result.current_ram_bytes += pager.current_ram_bytes;
        // These peaks are sums of per-layer peaks, not global observed peaks.
        result.peak_ram_bytes += pager.peak_ram_bytes;
        result.cache_entries += pager.cache_entries;
        result.active_leases += pager.active_leases;
        result.active_generations += pager.active_generations;
        result.direct_read_calls += pager.direct_read_calls;
        result.direct_requested_bytes += pager.direct_requested_bytes;
        result.direct_bytes_read += pager.direct_bytes_read;
        result.direct_payload_bytes += pager.direct_payload_bytes;
        result.direct_staging_limit_bytes += pager.direct_staging_limit_bytes;
        result.current_direct_staging_bytes += pager.current_direct_staging_bytes;
        result.peak_direct_staging_bytes += pager.peak_direct_staging_bytes;
    }
    return result;
}

const std::string &resident_text_model::model_root() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->model_root : empty;
}

int resident_text_model::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool resident_text_model::initialized() const noexcept {
    return impl_ && impl_->ready;
}

text_session::text_session() = default;
text_session::~text_session() = default;
text_session::text_session(text_session &&) noexcept = default;
text_session &text_session::operator=(text_session &&) noexcept = default;

text_model_status text_session::prefill(
        const std::uint32_t *token_ids,
        std::size_t token_count,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || token_ids == nullptr || token_count == 0u ||
        next_token == nullptr || !generation_valid(generation)) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid text prefill request");
    }
    const model_session_view initial = impl_->decoder->view();
    if (initial.committed_tokens != 0u || initial.transaction_open) {
        return fail(text_model_status::invalid_state, error,
                    "prefill requires a fresh or reset text session");
    }
    if (token_count > impl_->config.context_capacity) {
        return fail(text_model_status::capacity_exceeded, error,
                    "prompt exceeds the bounded text session capacity");
    }
    if (!impl_->rope_table_text_pristine) {
        const text_model_status restored =
                impl_->restore_text_rope_table(error);
        if (restored != text_model_status::ok) return restored;
    }
    impl_->multimodal_mrope_active = false;
    impl_->next_mrope_text_position = 0u;
    impl_->next_mrope_text_position_available = false;

    const steady_clock::time_point started = steady_clock::now();
    const text_session_metrics before_metrics = impl_->metrics;
    for (std::size_t index = 0u; index < token_count; ++index) {
        text_token_result final_result{};
        const bool final = index + 1u == token_count;
        text_model_status status = impl_->run_token(
                token_ids[index], nullptr, nullptr, final, generation,
                final ? &final_result : nullptr, error);
        if (status != text_model_status::ok) {
            const std::uint64_t rollback_delta =
                    impl_->metrics.rolled_back_transactions -
                    before_metrics.rolled_back_transactions;
            std::string reset_error;
            const model_session_status reset =
                    impl_->decoder->reset(impl_->stream, &reset_error);
            impl_->metrics = before_metrics;
            impl_->metrics.rolled_back_transactions += rollback_delta;
            ++impl_->metrics.resets;
            if (reset != model_session_status::ok) {
                return fail(text_model_status::transaction_error, error,
                            "prefill failed and session reset failed: " +
                                    (reset_error.empty()
                                             ? model_session_status_string(reset)
                                             : reset_error));
            }
            return status;
        }
        ++impl_->metrics.prompt_input_tokens;
        if (final) *next_token = final_result;
    }
    if (!impl_->metrics.ttft_measured) {
        impl_->metrics.ttft_ms = elapsed_ms(started);
        impl_->metrics.ttft_measured = true;
    }
    return text_model_status::ok;
}

text_model_status text_session::prefill_multimodal(
        const multimodal_text_prompt_view &prompt,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || next_token == nullptr ||
        !generation_valid(generation)) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid multimodal prefill request");
    }
    const model_session_view initial = impl_->decoder->view();
    if (initial.committed_tokens != 0u || initial.transaction_open) {
        return fail(text_model_status::invalid_state, error,
                    "multimodal prefill requires a fresh or reset session");
    }
    if (prompt.token_count > impl_->config.context_capacity) {
        return fail(text_model_status::capacity_exceeded, error,
                    "multimodal prompt exceeds the bounded session capacity");
    }

    multimodal_text_prompt_report report{};
    const multimodal_text_status admitted = validate_multimodal_text_prompt(
            prompt, impl_->device, &report);
    if (admitted != multimodal_text_status::ok) {
        return fail(map_multimodal_status(admitted), error,
                    std::string("multimodal prompt rejected: ") +
                            multimodal_text_status_string(admitted));
    }

    const steady_clock::time_point started = steady_clock::now();
    const text_session_metrics before_metrics = impl_->metrics;
    std::size_t visual_index = 0u;
    for (std::size_t index = 0u; index < prompt.token_count; ++index) {
        const std::uint32_t token = prompt.token_ids[index];
        const bool visual = token == kImageTokenId || token == kVideoTokenId;
        const float *projected = nullptr;
        if (visual) {
            projected = prompt.projected_vision_embeddings_device +
                        visual_index * kMultimodalHidden;
            ++visual_index;
        }
        const std::array<std::uint32_t, kMultimodalPositionAxes> positions{{
                prompt.mrope_positions[index],
                prompt.mrope_positions[prompt.token_count + index],
                prompt.mrope_positions[2u * prompt.token_count + index],
        }};
        text_token_result final_result{};
        const bool final = index + 1u == prompt.token_count;
        const text_model_status status = impl_->run_token(
                token, projected, positions.data(), final, generation,
                final ? &final_result : nullptr, error);
        if (status != text_model_status::ok) {
            const std::uint64_t rollback_delta =
                    impl_->metrics.rolled_back_transactions -
                    before_metrics.rolled_back_transactions;
            std::string reset_error;
            const model_session_status reset =
                    impl_->decoder->reset(impl_->stream, &reset_error);
            impl_->metrics = before_metrics;
            impl_->metrics.rolled_back_transactions += rollback_delta;
            ++impl_->metrics.resets;
            impl_->multimodal_mrope_active = false;
            impl_->next_mrope_text_position = 0u;
            impl_->next_mrope_text_position_available = false;
            if (reset != model_session_status::ok) {
                return fail(text_model_status::transaction_error, error,
                            "multimodal prefill failed and reset failed: " +
                                    (reset_error.empty()
                                             ? model_session_status_string(reset)
                                             : reset_error));
            }
            return status;
        }
        ++impl_->metrics.prompt_input_tokens;
        if (visual) ++impl_->metrics.prompt_visual_tokens;
        if (final) *next_token = final_result;
    }
    if (visual_index != report.vision_tokens) {
        std::string reset_error;
        (void)impl_->decoder->reset(impl_->stream, &reset_error);
        impl_->metrics = before_metrics;
        ++impl_->metrics.resets;
        impl_->multimodal_mrope_active = false;
        impl_->next_mrope_text_position = 0u;
        impl_->next_mrope_text_position_available = false;
        return fail(text_model_status::invalid_state, error,
                    "multimodal visual row accounting diverged after admission");
    }
    impl_->multimodal_mrope_active = true;
    impl_->next_mrope_text_position = report.next_text_position;
    impl_->next_mrope_text_position_available =
            report.next_text_position_available;
    if (!impl_->metrics.ttft_measured) {
        impl_->metrics.ttft_ms = elapsed_ms(started);
        impl_->metrics.ttft_measured = true;
    }
    return text_model_status::ok;
}

text_model_status text_session::prefill_token_for_speculation(
        std::uint32_t token,
        const float *projected_visual_embedding,
        const std::uint32_t *mrope_positions_t_h_w,
        bool final,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || !generation_valid(generation) ||
        (final && next_token == nullptr) || (!final && next_token != nullptr)) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid speculative prefill token request");
    }
    const model_session_view before = impl_->decoder->view();
    if (before.transaction_open ||
        before.committed_tokens >= impl_->config.context_capacity) {
        return fail(text_model_status::invalid_state, error,
                    "speculative prefill requires bounded committed state");
    }
    if (before.committed_tokens == 0u) {
        if (impl_->speculative_prefill_active) {
            return fail(text_model_status::invalid_state, error,
                        "speculative prefill is already active");
        }
        if (mrope_positions_t_h_w == nullptr &&
            !impl_->rope_table_text_pristine) {
            const text_model_status restored =
                    impl_->restore_text_rope_table(error);
            if (restored != text_model_status::ok) return restored;
        }
        impl_->multimodal_mrope_active = false;
        impl_->next_mrope_text_position = 0u;
        impl_->next_mrope_text_position_available = false;
        impl_->speculative_prefill_started = steady_clock::now();
        impl_->speculative_prefill_active = true;
    } else if (!impl_->speculative_prefill_active) {
        return fail(text_model_status::invalid_state, error,
                    "speculative prefill continuation was not opened");
    }

    text_token_result produced{};
    const text_model_status status = impl_->run_token(
            token, projected_visual_embedding, mrope_positions_t_h_w, final,
            generation, final ? &produced : nullptr, error);
    if (status != text_model_status::ok) return status;

    ++impl_->metrics.prompt_input_tokens;
    if (projected_visual_embedding != nullptr) {
        ++impl_->metrics.prompt_visual_tokens;
    }
    if (final) {
        *next_token = produced;
        if (!impl_->metrics.ttft_measured) {
            impl_->metrics.ttft_ms =
                    elapsed_ms(impl_->speculative_prefill_started);
            impl_->metrics.ttft_measured = true;
        }
        impl_->speculative_prefill_active = false;
    }
    return text_model_status::ok;
}

text_model_status text_session::prepare_mtp_inputs(
        std::uint32_t next_input_token,
        const float *projected_visual_embedding,
        const std::uint32_t *mrope_positions_t_h_w,
        text_mtp_inputs_view *view,
        std::string *error) noexcept {
    if (view != nullptr) *view = {};
    if (!impl_ || !impl_->ready || view == nullptr ||
        next_input_token >= kTextModelVocab) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid target-to-MTP handoff request");
    }
    const model_session_view state = impl_->decoder->view();
    if (!state.initialized || state.poisoned || state.transaction_open ||
        state.committed_tokens == 0u ||
        state.committed_tokens >= impl_->config.context_capacity) {
        return fail(text_model_status::invalid_state, error,
                    "target-to-MTP handoff requires committed target state and "
                    "one free context position");
    }
    if (mrope_positions_t_h_w != nullptr) {
        const text_model_status rope_status = impl_->install_mrope_row(
                static_cast<std::size_t>(state.committed_tokens),
                mrope_positions_t_h_w, error);
        if (rope_status != text_model_status::ok) return rope_status;
    }

    std::unique_lock<std::mutex> execution_lock(
            impl_->owner->execution_mutex);
    output_head_status output_status = output_head_status::ok;
    if (projected_visual_embedding == nullptr) {
        if (cudaMemcpyAsync(impl_->input_token, &next_input_token,
                            sizeof(next_input_token), cudaMemcpyHostToDevice,
                            impl_->stream) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cannot upload the MTP input token");
        }
        output_status = impl_->owner->output_head->embedding_repeat(
                impl_->input_token, 1u, impl_->scratch,
                impl_->scratch.hyper_input, impl_->stream);
    } else {
        if (cudaMemsetAsync(impl_->scratch.device_status, 0,
                            impl_->scratch.device_status_bytes,
                            impl_->stream) != cudaSuccess) {
            return fail(text_model_status::cuda_error, error,
                        "cannot reset the visual MTP handoff status");
        }
        const multimodal_text_status visual_status =
                repeat_projected_vision_embedding_cuda(
                        projected_visual_embedding, impl_->scratch.hyper_input,
                        impl_->scratch.device_status, impl_->device,
                        impl_->stream);
        if (visual_status != multimodal_text_status::ok) {
            return fail(map_multimodal_status(visual_status), error,
                        std::string("visual MTP handoff failed: ") +
                                multimodal_text_status_string(visual_status));
        }
    }
    if (output_status != output_head_status::ok) {
        return fail(map_output_status(output_status), error,
                    std::string("MTP input embedding failed: ") +
                            output_head_status_string(output_status));
    }
    output_status = impl_->owner->output_head->collect_device_status(
            impl_->scratch, impl_->stream);
    if (output_status != output_head_status::ok) {
        return fail(map_output_status(output_status), error,
                    std::string("MTP input embedding was rejected: ") +
                            output_head_status_string(output_status));
    }

    text_mtp_inputs_view prepared{};
    prepared.input_embedding_2560_f32 = impl_->scratch.hyper_input;
    prepared.target_hidden_4x2560_f32 = impl_->latest_decoder_hidden;
    prepared.full_cos_64_f32 = impl_->rope_cosine + rope::kRotaryDim;
    prepared.full_sin_64_f32 = impl_->rope_sine + rope::kRotaryDim;
    prepared.position_count =
            static_cast<std::size_t>(state.committed_tokens);
    *view = prepared;
    return text_model_status::ok;
}

text_model_status text_session::prepare_mtp_token_embedding(
        std::uint32_t token,
        const float **embedding_2560_f32,
        std::string *error) noexcept {
    if (embedding_2560_f32 != nullptr) *embedding_2560_f32 = nullptr;
    if (!impl_ || !impl_->ready || embedding_2560_f32 == nullptr ||
        token >= kTextModelVocab || impl_->decoder->view().transaction_open) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid recursive MTP embedding request");
    }
    std::unique_lock<std::mutex> execution_lock(
            impl_->owner->execution_mutex);
    if (cudaMemsetAsync(impl_->scratch.device_status, 0,
                        impl_->scratch.device_status_bytes,
                        impl_->stream) != cudaSuccess ||
        cudaMemcpyAsync(impl_->input_token, &token, sizeof(token),
                        cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess) {
        return fail(text_model_status::cuda_error, error,
                    "cannot upload recursive MTP token");
    }
    output_head_status status = impl_->owner->output_head->embedding_repeat(
            impl_->input_token, 1u, impl_->scratch,
            impl_->scratch.hyper_input, impl_->stream);
    if (status == output_head_status::ok) {
        status = impl_->owner->output_head->collect_device_status(
                impl_->scratch, impl_->stream);
    }
    if (status != output_head_status::ok) {
        return fail(map_output_status(status), error,
                    std::string("recursive MTP embedding failed: ") +
                            output_head_status_string(status));
    }
    *embedding_2560_f32 = impl_->scratch.hyper_input;
    return text_model_status::ok;
}

text_model_status text_session::stage_decode_sequence_for_speculation(
        const std::uint32_t *input_tokens,
        std::size_t token_count,
        text_token_result *next_tokens,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || input_tokens == nullptr ||
        next_tokens == nullptr || token_count < 2u ||
        token_count > kDecoderMaxSpeculativeTokens ||
        impl_->multimodal_mrope_active ||
        impl_->staged_speculative_tokens != 0u) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid text-only speculative block request");
    }
    const model_session_view before = impl_->decoder->view();
    if (!before.initialized || before.poisoned || before.transaction_open ||
        before.committed_tokens == 0u ||
        token_count > impl_->config.context_capacity ||
        before.committed_tokens >
                impl_->config.context_capacity - token_count) {
        return fail(text_model_status::invalid_state, error,
                    "speculative block requires bounded committed target state");
    }
    for (std::size_t index = 0u; index < token_count; ++index) {
        if (input_tokens[index] >= kTextModelVocab) {
            return fail(text_model_status::token_rejected, error,
                        "speculative block contains an invalid token id");
        }
    }

    const steady_clock::time_point started = steady_clock::now();
    std::unique_lock<std::mutex> execution_lock(
            impl_->owner->execution_mutex);
    std::string detail;
    model_session_status session_status =
            impl_->decoder->begin(impl_->stream, &detail);
    if (session_status != model_session_status::ok) {
        return fail(map_session_status(session_status), error,
                    detail.empty() ? model_session_status_string(session_status)
                                   : detail);
    }

    if (cudaMemsetAsync(impl_->scratch.device_status, 0,
                        impl_->scratch.device_status_bytes,
                        impl_->stream) != cudaSuccess ||
        cudaMemcpyAsync(impl_->input_token, input_tokens,
                        token_count * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess) {
        return impl_->abort_transaction(text_model_status::cuda_error,
                                        "cannot upload speculative token block",
                                        error);
    }
    output_head_status output_status =
            impl_->owner->output_head->embedding_repeat(
                    impl_->input_token, token_count, impl_->scratch,
                    impl_->scratch.hyper_input, impl_->stream);
    if (output_status != output_head_status::ok) {
        return impl_->abort_transaction(
                map_output_status(output_status),
                std::string("speculative embedding block failed: ") +
                        output_head_status_string(output_status), error);
    }

    std::array<std::int64_t, kDecoderMaxSpeculativeTokens> signed_tokens{};
    for (std::size_t index = 0u; index < token_count; ++index) {
        signed_tokens[index] = static_cast<std::int64_t>(input_tokens[index]);
    }
    session_status = impl_->decoder->stage_all_sequence(
            signed_tokens.data(), token_count, impl_->scratch.hyper_input,
            impl_->rope_cosine, impl_->rope_sine,
            static_cast<std::size_t>(before.committed_tokens) + token_count,
            impl_->decoder_output, impl_->stream, &detail);
    if (session_status != model_session_status::ok) {
        return impl_->abort_transaction(
                map_session_status(session_status),
                detail.empty() ? "48-layer speculative block failed" : detail,
                error);
    }

    output_status = impl_->owner->output_head->global_mix(
            impl_->decoder_output, token_count, impl_->scratch,
            impl_->scratch.mixed_hidden, impl_->stream);
    if (output_status == output_head_status::ok) {
        output_status = impl_->owner->output_head->logits(
                impl_->scratch.mixed_hidden, token_count, impl_->scratch,
                impl_->scratch.logits, impl_->stream);
    }
    if (output_status == output_head_status::ok) {
        output_status = impl_->owner->output_head->argmax(
                impl_->scratch.logits, token_count, impl_->selected_token,
                impl_->selected_logit, impl_->stream);
    }
    if (output_status == output_head_status::ok) {
        output_status = impl_->owner->output_head->collect_device_status(
                impl_->scratch, impl_->stream);
    }
    if (output_status != output_head_status::ok) {
        return impl_->abort_transaction(
                map_output_status(output_status),
                std::string("speculative output block failed: ") +
                        output_head_status_string(output_status), error);
    }

    std::array<std::uint32_t, kDecoderMaxSpeculativeTokens> host_tokens{};
    std::array<float, kDecoderMaxSpeculativeTokens> host_logits{};
    if (cudaMemcpyAsync(host_tokens.data(), impl_->selected_token,
                        token_count * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess ||
        cudaMemcpyAsync(host_logits.data(), impl_->selected_logit,
                        token_count * sizeof(float),
                        cudaMemcpyDeviceToHost, impl_->stream) != cudaSuccess ||
        cudaStreamSynchronize(impl_->stream) != cudaSuccess) {
        return impl_->abort_transaction(text_model_status::cuda_error,
                                        "cannot collect speculative target block",
                                        error);
    }
    const double block_ms = elapsed_ms(started);
    for (std::size_t index = 0u; index < token_count; ++index) {
        next_tokens[index] = {};
        next_tokens[index].token_id = host_tokens[index];
        next_tokens[index].selected_logit = host_logits[index];
        next_tokens[index].committed_context =
                before.committed_tokens + index + 1u;
        next_tokens[index].latency_ms = block_ms;
        next_tokens[index].sampled = false;
    }
    impl_->staged_speculative_tokens = token_count;
    impl_->staged_speculative_started = started;
    return text_model_status::ok;
}

text_model_status text_session::commit_staged_speculative_sequence(
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || impl_->staged_speculative_tokens == 0u ||
        !impl_->decoder->view().transaction_open) {
        return fail(text_model_status::invalid_state, error,
                    "no speculative target block is staged");
    }
    std::unique_lock<std::mutex> execution_lock(
            impl_->owner->execution_mutex);
    const std::size_t token_count = impl_->staged_speculative_tokens;
    std::string detail;
    const model_session_status status =
            impl_->decoder->commit_all(impl_->stream, &detail);
    if (status != model_session_status::ok) {
        impl_->staged_speculative_tokens = 0u;
        return fail(map_session_status(status), error,
                    detail.empty() ? "speculative target commit failed" : detail);
    }
    const double measured = elapsed_ms(impl_->staged_speculative_started);
    impl_->latest_decoder_hidden = impl_->decoder_output +
            (token_count - 1u) * kDecoderResidual;
    ++impl_->metrics.committed_transactions;
    impl_->metrics.decode_input_tokens += token_count;
    impl_->metrics.generated_tokens += token_count;
    impl_->metrics.last_decode_ms = measured;
    impl_->metrics.total_decode_ms += measured;
    impl_->staged_speculative_tokens = 0u;
    return text_model_status::ok;
}

text_model_status text_session::rollback_staged_speculative_sequence(
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || impl_->staged_speculative_tokens == 0u ||
        !impl_->decoder->view().transaction_open) {
        return fail(text_model_status::invalid_state, error,
                    "no speculative target block is staged");
    }
    std::unique_lock<std::mutex> execution_lock(
            impl_->owner->execution_mutex);
    std::string detail;
    const model_session_status status =
            impl_->decoder->rollback_all(impl_->stream, &detail);
    impl_->staged_speculative_tokens = 0u;
    ++impl_->metrics.rolled_back_transactions;
    if (status != model_session_status::ok) {
        return fail(map_session_status(status), error,
                    detail.empty() ? "speculative target rollback failed" : detail);
    }
    return text_model_status::ok;
}

text_model_status text_session::finish_speculative_multimodal_prefill(
        std::uint32_t next_text_position,
        bool next_text_position_available,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || impl_->speculative_prefill_active ||
        (next_text_position_available &&
         next_text_position >= rope::kMaxContext)) {
        return fail(text_model_status::invalid_state, error,
                    "cannot finalize speculative multimodal continuation");
    }
    impl_->multimodal_mrope_active = true;
    impl_->next_mrope_text_position = next_text_position;
    impl_->next_mrope_text_position_available =
            next_text_position_available;
    return text_model_status::ok;
}

text_model_status text_session::decode(
        std::uint32_t input_token,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (!impl_ || !impl_->ready || next_token == nullptr ||
        !generation_valid(generation)) {
        return fail(text_model_status::invalid_argument, error,
                    "invalid one-token decode request");
    }
    const model_session_view before = impl_->decoder->view();
    if (before.committed_tokens == 0u || before.transaction_open) {
        return fail(text_model_status::invalid_state, error,
                    "decode requires a committed prompt and no open transaction");
    }
    const steady_clock::time_point started = steady_clock::now();
    std::array<std::uint32_t, kMultimodalPositionAxes> positions{};
    const std::uint32_t *position_pointer = nullptr;
    if (impl_->multimodal_mrope_active) {
        if (!impl_->next_mrope_text_position_available) {
            return fail(text_model_status::capacity_exceeded, error,
                        "multimodal MRoPE reached the native 262K domain");
        }
        positions.fill(impl_->next_mrope_text_position);
        position_pointer = positions.data();
    }
    const text_model_status status = impl_->run_token(
            input_token, nullptr, position_pointer, true, generation,
            next_token, error);
    if (status == text_model_status::ok) {
        const double measured = elapsed_ms(started);
        ++impl_->metrics.decode_input_tokens;
        impl_->metrics.last_decode_ms = measured;
        impl_->metrics.total_decode_ms += measured;
        if (impl_->multimodal_mrope_active) {
            if (impl_->next_mrope_text_position + 1u < rope::kMaxContext) {
                ++impl_->next_mrope_text_position;
            } else {
                impl_->next_mrope_text_position_available = false;
            }
        }
    }
    return status;
}

text_model_status text_session::reset(std::string *error) noexcept {
    if (!impl_ || !impl_->ready || impl_->decoder == nullptr) {
        return fail(text_model_status::invalid_state, error,
                    "text session is not initialized");
    }
    try {
        std::unique_lock<std::mutex> execution_lock(
                impl_->owner->execution_mutex);
        std::string detail;
        const model_session_status status =
                impl_->decoder->reset(impl_->stream, &detail);
        if (status != model_session_status::ok) {
            return fail(map_session_status(status), error,
                        detail.empty() ? model_session_status_string(status)
                                       : detail);
        }
        const std::uint64_t reset_count = impl_->metrics.resets + 1u;
        impl_->metrics = text_session_metrics{};
        impl_->metrics.resets = reset_count;
        impl_->multimodal_mrope_active = false;
        impl_->next_mrope_text_position = 0u;
        impl_->next_mrope_text_position_available = false;
        impl_->speculative_prefill_active = false;
        impl_->staged_speculative_tokens = 0u;
        impl_->latest_decoder_hidden = impl_->decoder_output;
        return text_model_status::ok;
    } catch (const std::exception &exception) {
        return fail(text_model_status::invalid_state, error,
                    std::string("text session reset exception: ") +
                            exception.what());
    } catch (...) {
        return fail(text_model_status::invalid_state, error,
                    "unknown text session reset exception");
    }
}

text_session_view text_session::view() const noexcept {
    text_session_view result{};
    if (!impl_) return result;
    if (impl_->decoder) result.decoder = impl_->decoder->view();
    result.metrics = impl_->metrics;
    result.next_mrope_text_position = impl_->next_mrope_text_position;
    result.multimodal_mrope_active = impl_->multimodal_mrope_active;
    result.next_mrope_text_position_available =
            impl_->next_mrope_text_position_available;
    result.initialized = impl_->ready;
    return result;
}

bool text_session::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
