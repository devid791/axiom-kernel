#include "axiom/qwen4exp/speculative_model.hpp"

#include "axiom/qwen4exp/checkpoint.hpp"

#include <array>
#include <exception>
#include <new>
#include <utility>

namespace axiom::qwen4exp {
namespace {

speculative_status fail(speculative_status status,
                        std::string *error,
                        const std::string &message) noexcept {
    if (error != nullptr) *error = message;
    return status;
}

class native_target_endpoint final : public speculative_target_endpoint {
public:
    explicit native_target_endpoint(text_session *session) noexcept
        : session_(session) {}

    text_model_status decode(
            std::uint32_t input_token,
            const text_generation_config &generation,
            text_token_result *next_token,
            std::string *error) noexcept override {
        return session_ == nullptr
                ? text_model_status::invalid_state
                : session_->decode(input_token, generation, next_token, error);
    }

    text_model_status reset(std::string *error) noexcept override {
        return session_ == nullptr
                ? text_model_status::invalid_state
                : session_->reset(error);
    }

private:
    text_session *session_ = nullptr;
};

class native_draft_endpoint final : public speculative_draft_endpoint {
public:
    native_draft_endpoint(mtp_session *session, cudaStream_t stream) noexcept
        : session_(session), stream_(stream) {}

    mtp_status stage(const speculative_draft_inputs &inputs,
                     mtp_prediction *prediction,
                     std::string *error) noexcept override {
        if (session_ == nullptr) return mtp_status::invalid_state;
        return session_->stage_token(
                inputs.input_embedding_2560_f32,
                inputs.target_hidden_4x2560_f32,
                inputs.full_cos_64_f32,
                inputs.full_sin_64_f32,
                inputs.position_count,
                inputs.output_hidden_4x2560_f32,
                prediction,
                stream_,
                error);
    }

    mtp_status commit(std::string *error) noexcept override {
        return session_ == nullptr
                ? mtp_status::invalid_state
                : session_->commit(stream_, error);
    }

    mtp_status rollback(std::string *error) noexcept override {
        return session_ == nullptr
                ? mtp_status::invalid_state
                : session_->rollback(stream_, error);
    }

    mtp_status reset(std::string *error) noexcept override {
        return session_ == nullptr
                ? mtp_status::invalid_state
                : session_->reset(stream_, error);
    }

    bool transaction_open() const noexcept override {
        return session_ != nullptr && session_->state().transaction_open;
    }

private:
    mtp_session *session_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

}  // namespace

struct speculative_model::impl {
    speculative_model_config config{};
    speculative_model_metrics metrics{};
    std::unique_ptr<resident_text_model> target;
    std::unique_ptr<mtp_provider> draft;
    std::string draft_diagnostic;
    bool ready = false;
};

struct speculative_session::impl {
    std::unique_ptr<text_session> target;
    std::unique_ptr<mtp_session> draft;
    std::unique_ptr<native_target_endpoint> target_endpoint;
    std::unique_ptr<native_draft_endpoint> draft_endpoint;
    std::unique_ptr<speculative_transaction_driver> driver;
    std::array<float *, 2u> draft_output_hidden{{nullptr, nullptr}};
    const float *latest_draft_hidden = nullptr;
    std::array<text_token_result, kSpeculativeMaxDraftWidth> queued{};
    std::array<std::uint32_t, kSpeculativeMaxDraftWidth> queued_inputs{};
    std::size_t queue_size = 0u;
    std::size_t queue_index = 0u;
    std::size_t proposal_width = kSpeculativeDefaultDraftWidth;
    cudaStream_t stream = nullptr;
    int device = -1;
    bool require_exact_draft = false;
    bool ready = false;

    ~impl() {
        if (driver != nullptr) {
            std::string ignored;
            (void)driver->disconnect(&ignored);
        }
        driver.reset();
        draft_endpoint.reset();
        target_endpoint.reset();
        draft.reset();
        target.reset();
        if (draft_output_hidden[0] != nullptr ||
            draft_output_hidden[1] != nullptr) {
            int previous = -1;
            const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
            if (device >= 0 && (!have_previous || previous != device)) {
                (void)cudaSetDevice(device);
            }
            for (float *pointer : draft_output_hidden) {
                if (pointer != nullptr) (void)cudaFree(pointer);
            }
            if (have_previous && previous >= 0 && previous != device) {
                (void)cudaSetDevice(previous);
            }
        }
    }
};

const char *speculative_status_string(speculative_status status) noexcept {
    switch (status) {
        case speculative_status::ok: return "ok";
        case speculative_status::invalid_argument: return "invalid_argument";
        case speculative_status::unsupported_config: return "unsupported_config";
        case speculative_status::target_error: return "target_error";
        case speculative_status::draft_error: return "draft_error";
        case speculative_status::rollback_error: return "rollback_error";
        case speculative_status::invalid_state: return "invalid_state";
    }
    return "unknown";
}

const char *speculative_path_string(speculative_path path) noexcept {
    switch (path) {
        case speculative_path::target_only: return "target_only";
        case speculative_path::draft_verified_accepted:
            return "draft_verified_accepted";
        case speculative_path::draft_verified_rejected:
            return "draft_verified_rejected";
        case speculative_path::draft_error_target_only:
            return "draft_error_target_only";
        case speculative_path::draft_commit_error_target_only:
            return "draft_commit_error_target_only";
    }
    return "unknown";
}

speculative_status speculative_model_validate_config(
        const speculative_model_config &config) noexcept {
    if (text_model_validate_config(config.target) != text_model_status::ok ||
        mtp_validate_config(config.draft) != mtp_status::ok ||
        config.draft.max_context > config.target.max_context) {
        return speculative_status::unsupported_config;
    }
    return speculative_status::ok;
}

speculative_status speculative_session_validate_config(
        const speculative_session_config &config) noexcept {
    return config.context_capacity == 0u ||
                   config.context_capacity > kTextModelNativeContext ||
                   (config.enable_draft && (config.proposal_width < 2u ||
                   config.proposal_width > kSpeculativeMaxDraftWidth))
            ? speculative_status::unsupported_config
            : speculative_status::ok;
}

speculative_model::speculative_model() = default;
speculative_model::~speculative_model() = default;
speculative_model::speculative_model(speculative_model &&) noexcept = default;
speculative_model &speculative_model::operator=(
        speculative_model &&) noexcept = default;

speculative_status speculative_model::load(
        const std::string &model_root,
        const speculative_model_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<speculative_model> *out,
        std::string *error) noexcept {
    if (model_root.empty() || out == nullptr) {
        return fail(speculative_status::invalid_argument, error,
                    "model root or speculative output is invalid");
    }
    out->reset();
    if (error != nullptr) error->clear();
    if (speculative_model_validate_config(config) != speculative_status::ok) {
        return fail(speculative_status::unsupported_config, error,
                    "speculative model requires compatible native 262K target "
                    "and qwen4_exp MTP configurations");
    }

    try {
        auto result = std::make_unique<speculative_model>();
        result->impl_ = std::make_unique<impl>();
        impl &state = *result->impl_;
        state.config = config;

        std::string detail;
        const text_model_status target_status = resident_text_model::load(
                model_root, config.target, initialization_stream,
                &state.target, &detail);
        if (target_status != text_model_status::ok || state.target == nullptr ||
            !state.target->initialized()) {
            return fail(speculative_status::target_error, error,
                        detail.empty()
                                ? std::string("target load failed: ") +
                                          text_model_status_string(target_status)
                                : detail);
        }
        state.metrics.target = state.target->metrics();

        std::unique_ptr<checkpoint_catalog> catalog;
        detail.clear();
        if (!checkpoint_catalog::open(model_root, &catalog, &detail) ||
            catalog == nullptr) {
            state.draft_diagnostic = detail.empty()
                    ? "cannot open checkpoint catalog for exact MTP admission"
                    : detail;
        } else {
            mtp_contract_report contract{};
            detail.clear();
            const mtp_status contract_status =
                    mtp_validate_checkpoint_contract(
                            *catalog, &contract, &detail);
            state.metrics.draft_contract = contract;
            if (contract_status != mtp_status::ok ||
                !speculative_exact_mtp_contract(contract)) {
                state.draft_diagnostic = detail.empty()
                        ? std::string("exact pinned MTP contract rejected: ") +
                                  mtp_status_string(contract_status)
                        : detail;
            } else {
                detail.clear();
                const mtp_status draft_status = mtp_provider::load(
                        *catalog, config.draft, initialization_stream,
                        &state.draft, &detail);
                if (draft_status != mtp_status::ok || state.draft == nullptr ||
                    !state.draft->initialized() ||
                    state.draft->device() != state.target->device()) {
                    state.draft.reset();
                    state.draft_diagnostic = detail.empty()
                            ? std::string("exact MTP load failed: ") +
                                      mtp_status_string(draft_status)
                            : detail;
                }
            }
        }

        if (state.draft != nullptr) {
            state.metrics.draft_contract = state.draft->contract();
            state.metrics.draft_resident_weight_bytes =
                    state.draft->resident_weight_bytes();
            state.metrics.exact_draft_admitted =
                    speculative_exact_mtp_contract(
                            state.metrics.draft_contract);
            state.metrics.target_only_fallback = false;
            state.draft_diagnostic.clear();
        }

        if (config.require_exact_draft &&
            !state.metrics.exact_draft_admitted) {
            return fail(speculative_status::draft_error, error,
                        state.draft_diagnostic.empty()
                                ? "required exact pinned MTP graph is unavailable"
                                : state.draft_diagnostic);
        }

        state.ready = true;
        *out = std::move(result);
        return speculative_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(speculative_status::invalid_state, error,
                    "host allocation failed during speculative model load");
    } catch (const std::exception &exception) {
        return fail(speculative_status::invalid_state, error,
                    std::string("speculative model load exception: ") +
                            exception.what());
    } catch (...) {
        return fail(speculative_status::invalid_state, error,
                    "unknown speculative model load exception");
    }
}

speculative_status speculative_model::create_session(
        const speculative_session_config &config,
        cudaStream_t stream,
        std::unique_ptr<speculative_session> *out,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->target == nullptr ||
        out == nullptr) {
        return fail(speculative_status::invalid_argument, error,
                    "speculative model or session output is invalid");
    }
    out->reset();
    if (speculative_session_validate_config(config) != speculative_status::ok ||
        config.context_capacity > impl_->config.target.max_context ||
        config.context_capacity > impl_->config.draft.max_context) {
        return fail(speculative_status::unsupported_config, error,
                    "speculative session context exceeds its admitted contract");
    }

    try {
        auto result = std::make_unique<speculative_session>();
        result->impl_ = std::make_unique<speculative_session::impl>();
        speculative_session::impl &state = *result->impl_;
        state.stream = stream;
        state.device = impl_->target->device();
        state.require_exact_draft = config.enable_draft && impl_->config.require_exact_draft;
        state.proposal_width = config.proposal_width;

        text_session_config target_config{};
        target_config.context_capacity = config.context_capacity;
        std::string detail;
        const text_model_status target_status = impl_->target->create_session(
                target_config, stream, &state.target, &detail);
        if (target_status != text_model_status::ok || state.target == nullptr ||
            !state.target->initialized()) {
            return fail(speculative_status::target_error, error,
                        detail.empty()
                                ? std::string("target session failed: ") +
                                          text_model_status_string(target_status)
                                : detail);
        }

        if (config.enable_draft && impl_->draft != nullptr) {
            detail.clear();
            const mtp_status draft_status = impl_->draft->create_session(
                    config.context_capacity, stream, &state.draft, &detail);
            if (draft_status != mtp_status::ok || state.draft == nullptr ||
                !state.draft->initialized()) {
                state.draft.reset();
                if (impl_->config.require_exact_draft) {
                    return fail(speculative_status::draft_error, error,
                                detail.empty()
                                        ? std::string("MTP session failed: ") +
                                                  mtp_status_string(draft_status)
                                        : detail);
                }
            }
        }

        if (state.draft != nullptr) {
            bool allocation_ok = true;
            for (float *&pointer : state.draft_output_hidden) {
                if (cudaMalloc(reinterpret_cast<void **>(&pointer),
                               kMtpResidual * sizeof(float)) != cudaSuccess) {
                    allocation_ok = false;
                    break;
                }
            }
            if (!allocation_ok) {
                state.draft.reset();
                if (state.require_exact_draft) {
                    return fail(speculative_status::draft_error, error,
                                "cannot allocate bounded MTP chain handoffs");
                }
                (void)cudaGetLastError();
            }
        }

        state.target_endpoint =
                std::make_unique<native_target_endpoint>(state.target.get());
        if (state.draft != nullptr) {
            state.draft_endpoint = std::make_unique<native_draft_endpoint>(
                    state.draft.get(), stream);
        }
        state.driver = std::make_unique<speculative_transaction_driver>(
                state.target_endpoint.get(), state.draft_endpoint.get());
        state.ready = true;
        *out = std::move(result);
        return speculative_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(speculative_status::invalid_state, error,
                    "host allocation failed during speculative session creation");
    } catch (const std::exception &exception) {
        return fail(speculative_status::invalid_state, error,
                    std::string("speculative session exception: ") +
                            exception.what());
    } catch (...) {
        return fail(speculative_status::invalid_state, error,
                    "unknown speculative session creation exception");
    }
}

const speculative_model_config &speculative_model::config() const noexcept {
    static const speculative_model_config empty{};
    return impl_ == nullptr ? empty : impl_->config;
}

const speculative_model_metrics &speculative_model::metrics() const noexcept {
    static const speculative_model_metrics empty{};
    return impl_ == nullptr ? empty : impl_->metrics;
}

expert_slot_arena_metrics speculative_model::expert_cache_metrics() const noexcept {
    return impl_ && impl_->target
            ? impl_->target->expert_cache_metrics() : expert_slot_arena_metrics{};
}

ExpertPagerMetrics speculative_model::host_expert_cache_metrics() const noexcept {
    return impl_ && impl_->target
            ? impl_->target->host_expert_cache_metrics() : ExpertPagerMetrics{};
}

const std::string &speculative_model::draft_diagnostic() const noexcept {
    static const std::string empty;
    return impl_ == nullptr ? empty : impl_->draft_diagnostic;
}

bool speculative_model::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

speculative_session::speculative_session() = default;
speculative_session::~speculative_session() = default;
speculative_session::speculative_session(speculative_session &&) noexcept =
        default;
speculative_session &speculative_session::operator=(
        speculative_session &&) noexcept = default;

speculative_status speculative_session::stage_native_proposal(
        std::uint32_t next_input_token,
        const float *projected_visual_embedding,
        const std::uint32_t *mrope_positions_t_h_w,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->target == nullptr ||
        impl_->draft == nullptr || impl_->driver == nullptr ||
        impl_->draft_output_hidden[0] == nullptr) {
        return fail(speculative_status::draft_error, error,
                    "native MTP proposal path is unavailable");
    }
    const text_session_view target_state = impl_->target->view();
    const mtp_session_state draft_state = impl_->draft->state();
    if (!target_state.initialized || draft_state.poisoned ||
        draft_state.transaction_open ||
        target_state.decoder.committed_tokens == 0u ||
        target_state.decoder.committed_tokens !=
                draft_state.committed_tokens + 1u) {
        impl_->driver->disable_draft();
        return fail(speculative_status::draft_error, error,
                    "target/MTP context alignment check failed");
    }

    text_mtp_inputs_view native{};
    std::string detail;
    const text_model_status handoff = impl_->target->prepare_mtp_inputs(
            next_input_token, projected_visual_embedding,
            mrope_positions_t_h_w, &native, &detail);
    if (handoff != text_model_status::ok || !native.complete()) {
        impl_->driver->disable_draft();
        return fail(speculative_status::draft_error, error,
                    detail.empty()
                            ? std::string("target-to-MTP handoff failed: ") +
                                      text_model_status_string(handoff)
                            : detail);
    }

    speculative_draft_inputs inputs{};
    inputs.input_embedding_2560_f32 = native.input_embedding_2560_f32;
    inputs.target_hidden_4x2560_f32 = native.target_hidden_4x2560_f32;
    inputs.full_cos_64_f32 = native.full_cos_64_f32;
    inputs.full_sin_64_f32 = native.full_sin_64_f32;
    inputs.position_count = native.position_count;
    inputs.output_hidden_4x2560_f32 = impl_->draft_output_hidden[0];
    speculative_proposal proposal{};
    detail.clear();
    const speculative_status status =
            impl_->driver->propose(inputs, &proposal, &detail);
    if (status != speculative_status::ok) {
        impl_->driver->disable_draft();
        return fail(status, error,
                    detail.empty() ? "native MTP proposal failed" : detail);
    }
    impl_->latest_draft_hidden = impl_->draft_output_hidden[0];
    return speculative_status::ok;
}

speculative_status speculative_session::target_prefill(
        const std::uint32_t *token_ids,
        std::size_t token_count,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->target == nullptr ||
        impl_->driver == nullptr) {
        return fail(speculative_status::invalid_state, error,
                    "speculative session is not initialized");
    }
    if (token_ids == nullptr || token_count == 0u || next_token == nullptr) {
        return fail(speculative_status::invalid_argument, error,
                    "invalid speculative target prefill request");
    }
    if (impl_->draft == nullptr || impl_->draft_output_hidden[0] == nullptr) {
        const text_model_status status = impl_->target->prefill(
                token_ids, token_count, generation, next_token, error);
        return status == text_model_status::ok
                ? speculative_status::ok
                : speculative_status::target_error;
    }

    for (std::size_t index = 0u; index < token_count; ++index) {
        const bool final = index + 1u == token_count;
        text_token_result produced{};
        const text_model_status target_status =
                impl_->target->prefill_token_for_speculation(
                        token_ids[index], nullptr, nullptr, final, generation,
                        final ? &produced : nullptr, error);
        if (target_status != text_model_status::ok) {
            std::string ignored;
            (void)impl_->driver->reset(&ignored);
            return speculative_status::target_error;
        }
        if (final) *next_token = produced;

        const std::uint32_t candidate =
                final ? produced.token_id : token_ids[index + 1u];
        std::string draft_detail;
        speculative_status draft_status = stage_native_proposal(
                candidate, nullptr, nullptr, &draft_detail);
        if (draft_status == speculative_status::ok && !final) {
            draft_status = impl_->driver->commit_context(&draft_detail);
        }
        if (draft_status != speculative_status::ok) {
            impl_->driver->disable_draft();
            if (impl_->require_exact_draft) {
                std::string ignored;
                (void)impl_->driver->reset(&ignored);
                return fail(speculative_status::draft_error, error,
                            draft_detail.empty()
                                    ? "required MTP prompt priming failed"
                                    : draft_detail);
            }
            if (error != nullptr) error->clear();
        }
    }
    return speculative_status::ok;
}

speculative_status speculative_session::target_prefill_multimodal(
        const multimodal_text_prompt_view &prompt,
        const text_generation_config &generation,
        text_token_result *next_token,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->target == nullptr ||
        impl_->driver == nullptr || next_token == nullptr) {
        return fail(speculative_status::invalid_state, error,
                    "speculative multimodal session is not initialized");
    }
    multimodal_text_prompt_report report{};
    const multimodal_text_status admitted = validate_multimodal_text_prompt(
            prompt, impl_->device, &report);
    if (admitted != multimodal_text_status::ok) {
        return fail(speculative_status::unsupported_config, error,
                    std::string("multimodal prompt rejected: ") +
                            multimodal_text_status_string(admitted));
    }
    if (impl_->draft == nullptr || impl_->draft_output_hidden[0] == nullptr) {
        const text_model_status status = impl_->target->prefill_multimodal(
                prompt, generation, next_token, error);
        return status == text_model_status::ok
                ? speculative_status::ok
                : speculative_status::target_error;
    }

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
        const bool final = index + 1u == prompt.token_count;
        text_token_result produced{};
        const text_model_status target_status =
                impl_->target->prefill_token_for_speculation(
                        token, projected, positions.data(), final, generation,
                        final ? &produced : nullptr, error);
        if (target_status != text_model_status::ok) {
            std::string ignored;
            (void)impl_->driver->reset(&ignored);
            return speculative_status::target_error;
        }
        if (final) *next_token = produced;

        std::uint32_t candidate = produced.token_id;
        const float *candidate_projected = nullptr;
        std::array<std::uint32_t, kMultimodalPositionAxes> candidate_positions{};
        bool candidate_available = true;
        if (!final) {
            candidate = prompt.token_ids[index + 1u];
            const bool candidate_visual =
                    candidate == kImageTokenId || candidate == kVideoTokenId;
            if (candidate_visual) {
                candidate_projected =
                        prompt.projected_vision_embeddings_device +
                        visual_index * kMultimodalHidden;
            }
            candidate_positions = {{
                    prompt.mrope_positions[index + 1u],
                    prompt.mrope_positions[prompt.token_count + index + 1u],
                    prompt.mrope_positions[
                            2u * prompt.token_count + index + 1u],
            }};
        } else if (report.next_text_position_available) {
            candidate_positions.fill(report.next_text_position);
        } else {
            candidate_available = false;
        }

        std::string draft_detail;
        speculative_status draft_status = speculative_status::ok;
        if (candidate_available) {
            draft_status = stage_native_proposal(
                    candidate, candidate_projected, candidate_positions.data(),
                    &draft_detail);
            if (draft_status == speculative_status::ok && !final) {
                draft_status = impl_->driver->commit_context(&draft_detail);
            }
        } else {
            impl_->driver->disable_draft();
        }
        if (draft_status != speculative_status::ok) {
            impl_->driver->disable_draft();
            if (impl_->require_exact_draft) {
                std::string ignored;
                (void)impl_->driver->reset(&ignored);
                return fail(speculative_status::draft_error, error,
                            draft_detail.empty()
                                    ? "required multimodal MTP priming failed"
                                    : draft_detail);
            }
            if (error != nullptr) error->clear();
        }
    }

    const text_model_status finish =
            impl_->target->finish_speculative_multimodal_prefill(
                    report.next_text_position,
                    report.next_text_position_available, error);
    if (finish != text_model_status::ok) {
        std::string ignored;
        (void)impl_->driver->reset(&ignored);
        return speculative_status::target_error;
    }
    return speculative_status::ok;
}

speculative_status speculative_session::propose(
        const speculative_draft_inputs &inputs,
        speculative_proposal *proposal,
        std::string *error) noexcept {
    return impl_ == nullptr || !impl_->ready || impl_->driver == nullptr
            ? fail(speculative_status::invalid_state, error,
                   "speculative session is not initialized")
            : impl_->driver->propose(inputs, proposal, error);
}

speculative_status speculative_session::verify(
        std::uint32_t input_token,
        const text_generation_config &generation,
        speculative_step_result *result,
        std::string *error) noexcept {
    return impl_ == nullptr || !impl_->ready || impl_->driver == nullptr
            ? fail(speculative_status::invalid_state, error,
                   "speculative session is not initialized")
            : impl_->driver->verify(input_token, generation, result, error);
}

speculative_status speculative_session::step(
        std::uint32_t input_token,
        const text_generation_config &generation,
        const speculative_draft_inputs *draft_inputs,
        speculative_step_result *result,
        std::string *error) noexcept {
    return impl_ == nullptr || !impl_->ready || impl_->driver == nullptr
            ? fail(speculative_status::invalid_state, error,
                   "speculative session is not initialized")
            : impl_->driver->step(
                      input_token, generation, draft_inputs, result, error);
}

speculative_status speculative_session::decode(
        std::uint32_t input_token,
        const text_generation_config &generation,
        speculative_step_result *result,
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->driver == nullptr ||
        impl_->target == nullptr || result == nullptr) {
        return fail(speculative_status::invalid_state, error,
                    "native speculative decode session is not initialized");
    }

    if (impl_->queue_index < impl_->queue_size) {
        const std::size_t index = impl_->queue_index;
        if (input_token != impl_->queued_inputs[index]) {
            return fail(speculative_status::invalid_state, error,
                        "caller token diverged from the committed speculative queue");
        }
        *result = {};
        result->authoritative = impl_->queued[index];
        result->path = speculative_path::draft_verified_accepted;
        result->draft_was_available = true;
        result->accepted = true;
        ++impl_->queue_index;
        if (impl_->queue_index == impl_->queue_size) {
            impl_->queue_index = 0u;
            impl_->queue_size = 0u;
        }
        return speculative_status::ok;
    }

    const speculative_session_view before = impl_->driver->view();
    const text_session_view target_before = impl_->target->view();
    const bool block_eligible = before.proposal_open &&
            before.draft_available && before.draft_synchronized &&
            impl_->draft != nullptr && impl_->latest_draft_hidden != nullptr &&
            generation.mode == text_selection_mode::argmax &&
            !target_before.multimodal_mrope_active &&
            target_before.decoder.context_capacity >= impl_->proposal_width &&
            target_before.decoder.committed_tokens <=
                    target_before.decoder.context_capacity -
                            impl_->proposal_width;

    if (block_eligible) {
        const std::size_t width = impl_->proposal_width;
        const std::uint64_t target_committed =
                target_before.decoder.committed_tokens;
        std::array<std::uint32_t, kSpeculativeMaxDraftWidth> drafts{};
        std::array<std::uint32_t, kSpeculativeMaxDraftWidth> verify_inputs{};
        std::array<text_token_result, kSpeculativeMaxDraftWidth> verified{};
        speculative_proposal root_proposal{};

        auto target_fallback = [&](const std::string &diagnostic) noexcept {
            impl_->driver->disable_draft();
            if (impl_->draft != nullptr) {
                const mtp_session_state draft_state = impl_->draft->state();
                if (!draft_state.poisoned && !draft_state.transaction_open &&
                    draft_state.committed_tokens >= target_committed) {
                    std::string ignored;
                    (void)impl_->draft->truncate_committed(
                            static_cast<std::size_t>(target_committed),
                            impl_->stream, &ignored);
                }
            }
            *result = {};
            std::string target_detail;
            const text_model_status target_status = impl_->target->decode(
                    input_token, generation, &result->authoritative,
                    &target_detail);
            if (target_status != text_model_status::ok) {
                return fail(speculative_status::target_error, error,
                            target_detail.empty()
                                    ? "target fallback after MTP block failed"
                                    : target_detail);
            }
            result->path = speculative_path::draft_error_target_only;
            result->draft_was_available = true;
            result->diagnostic = diagnostic;
            return speculative_status::ok;
        };

        std::string detail;
        speculative_status block_status =
                impl_->driver->commit_open_for_block(&root_proposal, &detail);
        if (block_status != speculative_status::ok) {
            if (impl_->require_exact_draft) {
                return fail(block_status, error,
                            detail.empty() ? "required MTP block root failed"
                                           : detail);
            }
            return target_fallback(detail.empty()
                                           ? "MTP block root commit failed"
                                           : detail);
        }
        drafts[0] = root_proposal.prediction.token_id;
        verify_inputs[0] = input_token;

        const float *previous_hidden = impl_->latest_draft_hidden;
        std::size_t last_buffer = previous_hidden ==
                        impl_->draft_output_hidden[0]
                ? 0u
                : 1u;
        bool chain_ok = true;
        for (std::size_t index = 1u; index < width; ++index) {
            text_mtp_inputs_view native{};
            detail.clear();
            const text_model_status embedding_status =
                    impl_->target->prepare_mtp_inputs(
                            drafts[index - 1u], nullptr, nullptr,
                            &native, &detail);
            if (embedding_status != text_model_status::ok ||
                native.input_embedding_2560_f32 == nullptr ||
                native.full_cos_64_f32 == nullptr ||
                native.full_sin_64_f32 == nullptr) {
                chain_ok = false;
                if (detail.empty()) detail = "recursive MTP embedding failed";
                break;
            }
            const mtp_session_state draft_state = impl_->draft->state();
            const std::size_t output_buffer = 1u - last_buffer;
            mtp_prediction prediction{};
            const mtp_status draft_status = impl_->draft->stage_token(
                    native.input_embedding_2560_f32, previous_hidden,
                    native.full_cos_64_f32, native.full_sin_64_f32,
                    draft_state.committed_tokens + 1u,
                    impl_->draft_output_hidden[output_buffer], &prediction,
                    impl_->stream, &detail);
            if (draft_status != mtp_status::ok) {
                chain_ok = false;
                if (detail.empty()) {
                    detail = std::string("recursive MTP stage failed: ") +
                            mtp_status_string(draft_status);
                }
                break;
            }
            const mtp_status commit_status =
                    impl_->draft->commit(impl_->stream, &detail);
            if (commit_status != mtp_status::ok) {
                chain_ok = false;
                if (detail.empty()) {
                    detail = std::string("recursive MTP commit failed: ") +
                            mtp_status_string(commit_status);
                }
                break;
            }
            drafts[index] = prediction.token_id;
            previous_hidden = impl_->draft_output_hidden[output_buffer];
            last_buffer = output_buffer;
        }
        impl_->latest_draft_hidden = previous_hidden;
        if (!chain_ok) {
            if (impl_->require_exact_draft) {
                return fail(speculative_status::draft_error, error, detail);
            }
            return target_fallback(detail);
        }
        for (std::size_t index = 1u; index < width; ++index) {
            verify_inputs[index] = drafts[index - 1u];
        }

        detail.clear();
        const text_model_status stage_status =
                impl_->target->stage_decode_sequence_for_speculation(
                        verify_inputs.data(), width, verified.data(), &detail);
        if (stage_status != text_model_status::ok) {
            if (impl_->require_exact_draft) {
                return fail(speculative_status::target_error, error,
                            detail.empty() ? "target block stage failed" : detail);
            }
            return target_fallback(detail.empty()
                                           ? "target block stage failed"
                                           : detail);
        }

        std::size_t accepted = 0u;
        while (accepted < width &&
               drafts[accepted] == verified[accepted].token_id) {
            ++accepted;
        }
        const std::size_t emitted =
                accepted == width ? width : accepted + 1u;

        if (accepted == width) {
            detail.clear();
            const text_model_status commit_status =
                    impl_->target->commit_staged_speculative_sequence(&detail);
            if (commit_status != text_model_status::ok) {
                return fail(speculative_status::target_error, error,
                            detail.empty() ? "target block commit failed" : detail);
            }
        } else {
            detail.clear();
            const text_model_status rollback_status =
                    impl_->target->rollback_staged_speculative_sequence(&detail);
            if (rollback_status != text_model_status::ok) {
                return fail(speculative_status::rollback_error, error,
                            detail.empty() ? "target block rollback failed" : detail);
            }
            const std::size_t wanted_draft =
                    static_cast<std::size_t>(target_committed - 1u) + emitted;
            const mtp_status truncate_status = impl_->draft->truncate_committed(
                    wanted_draft, impl_->stream, &detail);
            if (truncate_status != mtp_status::ok) {
                return fail(speculative_status::rollback_error, error,
                            detail.empty() ? "MTP suffix truncation failed" : detail);
            }
            for (std::size_t index = 0u; index < emitted; ++index) {
                text_token_result replayed{};
                const text_model_status replay_status = impl_->target->decode(
                        verify_inputs[index], generation, &replayed, &detail);
                if (replay_status != text_model_status::ok ||
                    replayed.token_id != verified[index].token_id) {
                    return fail(speculative_status::target_error, error,
                                detail.empty()
                                        ? "target prefix replay diverged"
                                        : detail);
                }
                verified[index] = replayed;
            }
        }

        impl_->driver->record_native_block(width, accepted, emitted);
        *result = {};
        result->authoritative = verified[0];
        result->proposal = root_proposal;
        result->draft_was_available = true;
        result->accepted = accepted != 0u;
        result->path = accepted == 0u
                ? speculative_path::draft_verified_rejected
                : speculative_path::draft_verified_accepted;
        result->diagnostic = "mtp_block width=" + std::to_string(width) +
                " accepted=" + std::to_string(accepted);

        impl_->queue_size = emitted > 0u ? emitted - 1u : 0u;
        impl_->queue_index = 0u;
        for (std::size_t index = 1u; index < emitted; ++index) {
            impl_->queued[index - 1u] = verified[index];
            impl_->queued_inputs[index - 1u] = verified[index - 1u].token_id;
        }

        const text_session_view committed_view = impl_->target->view();
        std::array<std::uint32_t, kMultimodalPositionAxes> positions{};
        const std::uint32_t *position_pointer = nullptr;
        if (committed_view.multimodal_mrope_active) {
            positions.fill(committed_view.next_mrope_text_position);
            position_pointer = positions.data();
        }
        detail.clear();
        const speculative_status next_status = stage_native_proposal(
                verified[emitted - 1u].token_id, nullptr, position_pointer,
                &detail);
        if (next_status != speculative_status::ok) {
            impl_->driver->disable_draft();
            result->diagnostic += "; next proposal unavailable: " + detail;
            if (impl_->require_exact_draft) {
                return fail(speculative_status::draft_error, error, detail);
            }
            if (error != nullptr) error->clear();
        }
        return speculative_status::ok;
    }

    speculative_status status = speculative_status::ok;
    if (before.proposal_open) {
        status = impl_->driver->verify(
                input_token, generation, result, error);
    } else {
        status = impl_->driver->target_only(
                input_token, generation, result, error);
    }
    if (status != speculative_status::ok) return status;

    const speculative_session_view after_verify = impl_->driver->view();
    if (!before.proposal_open || !after_verify.draft_available ||
        !after_verify.draft_synchronized) {
        return speculative_status::ok;
    }

    const text_session_view target_view = impl_->target->view();
    std::array<std::uint32_t, kMultimodalPositionAxes> positions{};
    const std::uint32_t *position_pointer = nullptr;
    if (target_view.multimodal_mrope_active) {
        if (!target_view.next_mrope_text_position_available) {
            impl_->driver->disable_draft();
            return speculative_status::ok;
        }
        positions.fill(target_view.next_mrope_text_position);
        position_pointer = positions.data();
    }

    std::string draft_detail;
    const speculative_status draft_status = stage_native_proposal(
            result->authoritative.token_id, nullptr, position_pointer,
            &draft_detail);
    if (draft_status == speculative_status::ok) {
        return speculative_status::ok;
    }
    impl_->driver->disable_draft();
    result->diagnostic = draft_detail;
    if (impl_->require_exact_draft) {
        return fail(speculative_status::draft_error, error,
                    draft_detail.empty()
                            ? "required continuous MTP proposal failed"
                            : draft_detail);
    }
    if (error != nullptr) error->clear();
    return speculative_status::ok;
}

speculative_status speculative_session::disconnect(
        std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->driver == nullptr) {
        return fail(speculative_status::invalid_state, error,
                    "speculative session is not initialized");
    }
    impl_->queue_index = 0u;
    impl_->queue_size = 0u;
    return impl_->driver->disconnect(error);
}

speculative_status speculative_session::reset(std::string *error) noexcept {
    if (impl_ == nullptr || !impl_->ready || impl_->driver == nullptr) {
        return fail(speculative_status::invalid_state, error,
                    "speculative session is not initialized");
    }
    impl_->queue_index = 0u;
    impl_->queue_size = 0u;
    impl_->latest_draft_hidden = nullptr;
    return impl_->driver->reset(error);
}

speculative_session_view speculative_session::view() const noexcept {
    return impl_ == nullptr || impl_->driver == nullptr
            ? speculative_session_view{}
            : impl_->driver->view();
}

bool speculative_session::initialized() const noexcept {
    return impl_ != nullptr && impl_->ready;
}

}  // namespace axiom::qwen4exp
