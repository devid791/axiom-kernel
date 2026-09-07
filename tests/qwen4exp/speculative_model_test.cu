#include "axiom/qwen4exp/speculative_model.hpp"
#include "axiom/qwen4exp/vision_adapter.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void require_status(q4::speculative_status actual,
                    q4::speculative_status expected,
                    const std::string &where) {
    if (actual != expected) {
        throw std::runtime_error(
                where + ": expected status " +
                std::to_string(static_cast<unsigned>(expected)) +
                ", got " +
                std::to_string(static_cast<unsigned>(actual)));
    }
}

struct fake_target final : q4::speculative_target_endpoint {
    fake_target(std::vector<std::uint32_t> values,
                std::vector<std::string> *event_log)
        : tokens(std::move(values)), events(event_log) {}

    std::vector<std::uint32_t> tokens;
    std::vector<std::string> *events = nullptr;
    std::size_t cursor = 0u;
    bool fail_next = false;
    bool fail_reset = false;

    q4::text_model_status decode(
            std::uint32_t,
            const q4::text_generation_config &,
            q4::text_token_result *next_token,
            std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("target");
        if (fail_next) {
            fail_next = false;
            if (error != nullptr) *error = "injected target failure";
            return q4::text_model_status::transaction_error;
        }
        if (next_token == nullptr || cursor >= tokens.size()) {
            if (error != nullptr) *error = "fake target exhausted";
            return q4::text_model_status::invalid_state;
        }
        next_token->token_id = tokens[cursor++];
        next_token->selected_logit = 7.0F;
        next_token->committed_context = cursor;
        return q4::text_model_status::ok;
    }

    q4::text_model_status reset(std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("target_reset");
        if (fail_reset) {
            if (error != nullptr) *error = "injected target reset failure";
            return q4::text_model_status::transaction_error;
        }
        cursor = 0u;
        return q4::text_model_status::ok;
    }
};

struct fake_draft final : q4::speculative_draft_endpoint {
    std::uint32_t token = 0u;
    std::vector<std::string> *events = nullptr;
    q4::mtp_status stage_status = q4::mtp_status::ok;
    q4::mtp_status commit_status = q4::mtp_status::ok;
    q4::mtp_status rollback_status = q4::mtp_status::ok;
    q4::mtp_status reset_status = q4::mtp_status::ok;
    bool leave_open_on_stage_error = false;
    bool open = false;
    std::uint64_t stages = 0u;
    std::uint64_t commits = 0u;
    std::uint64_t rollbacks = 0u;

    q4::mtp_status stage(const q4::speculative_draft_inputs &,
                         q4::mtp_prediction *prediction,
                         std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("stage");
        ++stages;
        if (stage_status != q4::mtp_status::ok) {
            open = leave_open_on_stage_error;
            if (error != nullptr) *error = "injected MTP stage failure";
            return stage_status;
        }
        if (prediction == nullptr) return q4::mtp_status::invalid_argument;
        open = true;
        prediction->token_id = token;
        prediction->selected_logit = 3.0F;
        return q4::mtp_status::ok;
    }

    q4::mtp_status commit(std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("commit");
        ++commits;
        if (!open) return q4::mtp_status::invalid_state;
        if (commit_status != q4::mtp_status::ok) {
            if (error != nullptr) *error = "injected MTP commit failure";
            return commit_status;
        }
        open = false;
        return q4::mtp_status::ok;
    }

    q4::mtp_status rollback(std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("rollback");
        ++rollbacks;
        if (!open) return q4::mtp_status::invalid_state;
        if (rollback_status != q4::mtp_status::ok) {
            if (error != nullptr) *error = "injected MTP rollback failure";
            return rollback_status;
        }
        open = false;
        return q4::mtp_status::ok;
    }

    q4::mtp_status reset(std::string *error) noexcept override {
        if (events != nullptr) events->emplace_back("draft_reset");
        if (reset_status != q4::mtp_status::ok) {
            if (error != nullptr) *error = "injected MTP reset failure";
            return reset_status;
        }
        open = false;
        return q4::mtp_status::ok;
    }

    bool transaction_open() const noexcept override { return open; }
};

q4::speculative_draft_inputs complete_inputs() {
    static float storage[5]{};
    q4::speculative_draft_inputs inputs{};
    inputs.input_embedding_2560_f32 = &storage[0];
    inputs.target_hidden_4x2560_f32 = &storage[1];
    inputs.full_cos_64_f32 = &storage[2];
    inputs.full_sin_64_f32 = &storage[3];
    inputs.position_count = 1u;
    inputs.output_hidden_4x2560_f32 = &storage[4];
    return inputs;
}

q4::text_generation_config greedy_generation() {
    q4::text_generation_config generation{};
    generation.mode = q4::text_selection_mode::argmax;
    return generation;
}

void accepted_gate() {
    std::vector<std::string> events;
    fake_target target{{17u}, &events};
    fake_draft draft{};
    draft.token = 17u;
    draft.events = &events;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result result{};
    std::string error;
    require_status(driver.step(5u, greedy_generation(), &inputs, &result,
                               &error),
                   q4::speculative_status::ok, "accepted step");
    require(error.empty(), "accepted step returned an error");
    require(result.authoritative.token_id == 17u && result.accepted,
            "accepted step changed the target token");
    require(result.path == q4::speculative_path::draft_verified_accepted,
            "accepted path was not reported");
    require(events == std::vector<std::string>({"stage", "target", "commit"}),
            "accepted transaction order is wrong");
    const q4::speculative_session_view view = driver.view();
    require(view.metrics.drafted_tokens == 1u &&
                    view.metrics.accepted_tokens == 1u &&
                    view.metrics.rejected_tokens == 0u &&
                    view.metrics.acceptance_rate() == 1.0,
            "accepted metrics are wrong");
    require(view.draft_available && view.draft_synchronized &&
                    !view.proposal_open,
            "accepted draft did not stay synchronized");
}

void mismatch_commits_context_and_continues() {
    std::vector<std::string> events;
    fake_target target{{23u, 29u}, &events};
    fake_draft draft{};
    draft.token = 19u;
    draft.events = &events;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result rejected{};
    require_status(driver.step(7u, greedy_generation(), &inputs, &rejected),
                   q4::speculative_status::ok, "mismatch step");
    require(rejected.authoritative.token_id == 23u && !rejected.accepted,
            "mismatch did not preserve the target token");
    require(rejected.path == q4::speculative_path::draft_verified_rejected,
            "mismatch path was not reported");
    require(events == std::vector<std::string>({"stage", "target", "commit"}),
            "mismatch did not commit its authoritative input context");

    draft.token = 29u;
    q4::speculative_step_result continued{};
    require_status(driver.step(23u, greedy_generation(), &inputs, &continued),
                   q4::speculative_status::ok, "post-mismatch continued step");
    require(continued.authoritative.token_id == 29u && continued.accepted &&
                    continued.path ==
                            q4::speculative_path::draft_verified_accepted,
            "post-mismatch continuous draft changed target correctness");
    require(draft.stages == 2u && draft.commits == 2u,
            "synchronized MTP did not continue after a mismatch");
    const q4::speculative_session_view view = driver.view();
    require(view.metrics.rejected_tokens == 1u &&
                    view.metrics.accepted_tokens == 1u &&
                    view.metrics.drafted_tokens == 2u &&
                    view.metrics.rollback_attempts == 0u &&
                    view.metrics.target_only_tokens == 0u &&
                    view.draft_available && view.draft_synchronized,
            "mismatch metrics/state are wrong");
}

void prompt_context_commit_is_not_scored_as_a_draft() {
    std::vector<std::string> events;
    fake_target target{{}, &events};
    fake_draft draft{};
    draft.token = 71u;
    draft.events = &events;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_proposal proposal{};
    require_status(driver.propose(inputs, &proposal),
                   q4::speculative_status::ok, "prompt proposal");
    require_status(driver.commit_context(), q4::speculative_status::ok,
                   "prompt context commit");
    const q4::speculative_session_view view = driver.view();
    require(events == std::vector<std::string>({"stage", "commit"}) &&
                    view.metrics.primed_context_tokens == 1u &&
                    view.metrics.drafted_tokens == 0u &&
                    !view.proposal_open && view.draft_synchronized,
            "prompt priming polluted verification metrics or state");
}

void stage_error_rolls_back_then_uses_target() {
    std::vector<std::string> events;
    fake_target target{{31u}, &events};
    fake_draft draft{};
    draft.events = &events;
    draft.stage_status = q4::mtp_status::qsa_error;
    draft.leave_open_on_stage_error = true;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result result{};
    require_status(driver.step(11u, greedy_generation(), &inputs, &result),
                   q4::speculative_status::ok, "stage-error fallback");
    require(result.authoritative.token_id == 31u &&
                    result.path ==
                            q4::speculative_path::draft_error_target_only,
            "stage-error fallback did not preserve target output");
    require(events ==
                    std::vector<std::string>({"stage", "rollback", "target"}),
            "stage error did not roll back before fallback");
    const q4::speculative_session_view view = driver.view();
    require(view.metrics.draft_stage_failures == 1u &&
                    view.metrics.rollback_attempts == 1u &&
                    view.metrics.target_only_tokens == 1u,
            "stage-error metrics are wrong");
}

void target_error_rolls_back() {
    std::vector<std::string> events;
    fake_target target{{}, &events};
    target.fail_next = true;
    fake_draft draft{};
    draft.token = 37u;
    draft.events = &events;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result result{};
    require_status(driver.step(13u, greedy_generation(), &inputs, &result),
                   q4::speculative_status::target_error,
                   "target-error transaction");
    require(events == std::vector<std::string>({"stage", "target", "rollback"}),
            "target error did not roll back the proposal");
    require(!draft.open && driver.view().metrics.rollback_attempts == 1u,
            "target-error rollback did not close the draft");
}

void commit_error_falls_back_without_changing_target() {
    std::vector<std::string> events;
    fake_target target{{41u}, &events};
    fake_draft draft{};
    draft.token = 41u;
    draft.events = &events;
    draft.commit_status = q4::mtp_status::attention_error;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result result{};
    require_status(driver.step(17u, greedy_generation(), &inputs, &result),
                   q4::speculative_status::ok, "commit-error fallback");
    require(result.authoritative.token_id == 41u && !result.accepted &&
                    result.path == q4::speculative_path::
                                           draft_commit_error_target_only,
            "commit error changed the authoritative token");
    require(events == std::vector<std::string>(
                              {"stage", "target", "commit", "rollback"}),
            "commit error did not trigger rollback");
    const q4::speculative_session_view view = driver.view();
    require(view.metrics.draft_commit_failures == 1u &&
                    view.metrics.rollback_attempts == 1u &&
                    view.metrics.target_only_tokens == 1u &&
                    !view.draft_available,
            "commit-error fail-closed state is wrong");
}

void disconnect_rolls_back_open_proposal() {
    std::vector<std::string> events;
    fake_target target{{43u}, &events};
    fake_draft draft{};
    draft.token = 43u;
    draft.events = &events;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_proposal proposal{};
    require_status(driver.propose(inputs, &proposal),
                   q4::speculative_status::ok, "disconnect proposal");
    require(proposal.staged && driver.view().proposal_open,
            "disconnect test did not open a proposal");
    require_status(driver.disconnect(), q4::speculative_status::ok,
                   "disconnect rollback");
    require(events == std::vector<std::string>({"stage", "rollback"}) &&
                    !draft.open,
            "disconnect did not roll back the open MTP transaction");
    const q4::speculative_session_view view = driver.view();
    require(view.disconnected && view.metrics.disconnect_rollbacks == 1u &&
                    view.metrics.rollback_attempts == 1u,
            "disconnect metrics are wrong");
}

void rollback_failure_fails_closed_on_disconnect() {
    fake_target target{{}, nullptr};
    fake_draft draft{};
    draft.token = 53u;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_proposal proposal{};
    require_status(driver.propose(inputs, &proposal),
                   q4::speculative_status::ok,
                   "rollback-failure proposal");
    draft.rollback_status = q4::mtp_status::cuda_error;
    require_status(driver.disconnect(), q4::speculative_status::rollback_error,
                   "rollback-failure disconnect");
    require(!driver.view().draft_available && driver.view().disconnected &&
                    driver.view().metrics.rollback_failures == 1u,
            "rollback failure did not fail closed");
    draft.open = false;
}

void reset_reenables_a_clean_draft() {
    fake_target target{{59u}, nullptr};
    fake_draft draft{};
    draft.token = 59u;
    draft.commit_status = q4::mtp_status::attention_error;
    q4::speculative_transaction_driver driver(&target, &draft);
    const q4::speculative_draft_inputs inputs = complete_inputs();
    q4::speculative_step_result failed_commit{};
    require_status(driver.step(23u, greedy_generation(), &inputs,
                               &failed_commit),
                   q4::speculative_status::ok, "pre-reset commit failure");
    require(!driver.view().draft_available,
            "commit failure did not disable the draft before reset");
    draft.commit_status = q4::mtp_status::ok;
    draft.token = 59u;
    require_status(driver.reset(), q4::speculative_status::ok,
                   "coordinated reset");
    require(driver.view().draft_available &&
                    driver.view().draft_synchronized &&
                    driver.view().metrics.resets == 1u,
            "coordinated reset did not re-enable a clean draft");
    q4::speculative_step_result accepted{};
    require_status(driver.step(23u, greedy_generation(), &inputs, &accepted),
                   q4::speculative_status::ok, "post-reset accepted step");
    require(accepted.accepted && accepted.authoritative.token_id == 59u,
            "post-reset deterministic acceptance failed");
}

void target_only_is_identical() {
    fake_target target{{67u}, nullptr};
    q4::speculative_transaction_driver driver(&target, nullptr);
    q4::speculative_step_result result{};
    require_status(driver.step(29u, greedy_generation(), nullptr, &result),
                   q4::speculative_status::ok, "explicit target-only step");
    require(result.authoritative.token_id == 67u &&
                    result.path == q4::speculative_path::target_only &&
                    !result.accepted &&
                    driver.view().metrics.target_only_tokens == 1u,
            "target-only path changed the target result");
}

void native_runtime_gate(const std::string &model_root) {
    cudaStream_t stream = nullptr;
    require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
                    cudaSuccess,
            "cannot create native MTP runtime stream");
    try {
        q4::speculative_model_config config{};
        config.target.verify_payload_hashes = false;
        config.require_exact_draft = true;
        std::unique_ptr<q4::speculative_model> model;
        std::string error;
        const q4::speculative_status load_status =
                q4::speculative_model::load(
                        model_root, config, stream, &model, &error);
        require_status(load_status, q4::speculative_status::ok,
                       "native speculative model load: " + error);
        require(model && model->initialized() &&
                        model->metrics().exact_draft_admitted &&
                        !model->metrics().target_only_fallback,
                "native model did not admit the exact MTP graph");

        q4::speculative_session_config session_config{};
        session_config.context_capacity = 64u;
        session_config.proposal_width = q4::kSpeculativeDefaultDraftWidth;
        std::unique_ptr<q4::speculative_session> session;
        error.clear();
        const q4::speculative_status session_status = model->create_session(
                session_config, stream, &session, &error);
        require_status(session_status, q4::speculative_status::ok,
                       "native speculative session: " + error);
        require(session && session->initialized(),
                "native speculative session is not initialized");

        const std::uint32_t prompt[]{101u, 102u, 103u};
        q4::text_token_result first{};
        error.clear();
        const q4::speculative_status prefill_status = session->target_prefill(
                prompt, 3u, greedy_generation(), &first, &error);
        require_status(prefill_status, q4::speculative_status::ok,
                       "native target/MTP prefill: " + error);
        q4::speculative_session_view state = session->view();
        require(state.proposal_open && state.draft_available &&
                        state.draft_synchronized &&
                        state.metrics.primed_context_tokens == 2u &&
                        state.metrics.drafted_tokens == 0u,
                "native prompt did not leave an aligned MTP proposal");

        q4::speculative_step_result decoded{};
        error.clear();
        const q4::speculative_status decode_status = session->decode(
                first.token_id, greedy_generation(), &decoded, &error);
        require_status(decode_status, q4::speculative_status::ok,
                       "native continuous MTP decode: " + error);
        state = session->view();
        const std::uint64_t accepted = state.metrics.accepted_tokens;
        const std::uint64_t emitted = state.metrics.block_emitted_tokens;
        require(decoded.authoritative.committed_context == 4u &&
                        state.metrics.block_cycles == 1u &&
                        state.metrics.block_verified_rows ==
                                q4::kSpeculativeDefaultDraftWidth &&
                        state.metrics.target_tokens ==
                                q4::kSpeculativeDefaultDraftWidth &&
                        state.metrics.drafted_tokens ==
                                q4::kSpeculativeDefaultDraftWidth &&
                        accepted <= q4::kSpeculativeDefaultDraftWidth &&
                        state.metrics.rejected_tokens ==
                                (accepted == q4::kSpeculativeDefaultDraftWidth
                                         ? 0u
                                         : 1u) &&
                        emitted ==
                                (accepted == q4::kSpeculativeDefaultDraftWidth
                                         ? q4::kSpeculativeDefaultDraftWidth
                                         : accepted + 1u) &&
                        state.proposal_open && state.draft_available &&
                        state.draft_synchronized,
                "native width-3 MTP block state diverged");

        std::uint32_t queued_input = decoded.authoritative.token_id;
        for (std::uint64_t index = 1u; index < emitted; ++index) {
            q4::speculative_step_result queued{};
            error.clear();
            require_status(session->decode(
                                   queued_input, greedy_generation(), &queued,
                                   &error),
                           q4::speculative_status::ok,
                           "native verified-token queue: " + error);
            require(queued.authoritative.committed_context == 4u + index &&
                            queued.path ==
                                    q4::speculative_path::
                                            draft_verified_accepted &&
                            queued.accepted,
                    "verified target token was not drained exactly once");
            queued_input = queued.authoritative.token_id;
        }
        const q4::speculative_session_view block_state = session->view();
        require(block_state.metrics.block_cycles == 1u &&
                        block_state.metrics.block_verified_rows ==
                                q4::kSpeculativeDefaultDraftWidth &&
                        block_state.metrics.block_emitted_tokens == emitted &&
                        block_state.proposal_open &&
                        block_state.draft_available &&
                        block_state.draft_synchronized,
                "draining verified tokens changed the native block state");

        error.clear();
        require_status(session->disconnect(&error),
                       q4::speculative_status::ok,
                       "native MTP disconnect: " + error);
        state = session->view();
        require(state.disconnected && !state.proposal_open &&
                        state.metrics.disconnect_rollbacks == 1u,
                "native MTP disconnect did not roll back the pending proposal");

        session.reset();

        std::unique_ptr<q4::checkpoint_catalog> catalog;
        error.clear();
        require(q4::checkpoint_catalog::open(model_root, &catalog, &error) &&
                        catalog != nullptr,
                "vision checkpoint catalog open failed: " + error);
        q4::vision_provider_config vision_config{};
        int active_device = -1;
        require(cudaGetDevice(&active_device) == cudaSuccess &&
                        active_device >= 0,
                "cannot resolve the active CUDA device for vision");
        vision_config.device = active_device;
        vision_config.max_patch_tokens = 4u;
        std::unique_ptr<q4::resident_vision_provider> vision;
        error.clear();
        const q4::vision_status vision_load =
                q4::resident_vision_provider::load(
                        *catalog, vision_config, stream, &vision, &error);
        require(vision_load == q4::vision_status::ok && vision &&
                        vision->initialized(),
                std::string("native vision load failed: ") +
                        q4::vision_status_string(vision_load) + ": " + error);

        constexpr std::size_t patch_count = 4u;
        std::vector<float> patches(
                patch_count * q4::kVisionPatchFeatures);
        for (std::size_t index = 0u; index < patches.size(); ++index) {
            patches[index] =
                    static_cast<float>((index * 17u) % 257u) / 128.0F - 1.0F;
        }
        float *device_patches = nullptr;
        float *device_projected = nullptr;
        require(cudaMalloc(reinterpret_cast<void **>(&device_patches),
                           patches.size() * sizeof(float)) == cudaSuccess &&
                        cudaMalloc(reinterpret_cast<void **>(&device_projected),
                                   q4::kVisionOutputSize * sizeof(float)) ==
                                cudaSuccess,
                "cannot allocate direct vision/text handoff buffers");
        require(cudaMemcpyAsync(device_patches, patches.data(),
                                patches.size() * sizeof(float),
                                cudaMemcpyHostToDevice, stream) == cudaSuccess,
                "cannot upload deterministic vision patches");
        require(cudaStreamSynchronize(stream) == cudaSuccess,
                "cannot publish deterministic patches to the synchronous "
                "vision provider");
        const q4::vision_grid grid{1u, 2u, 2u};
        std::size_t projected_tokens = 0u;
        const q4::vision_status vision_forward =
                vision->forward_patches_device(
                        device_patches, patches.size(), &grid, 1u,
                        device_projected, q4::kVisionOutputSize,
                        &projected_tokens, nullptr);
        require(vision_forward == q4::vision_status::ok &&
                        projected_tokens == 1u,
                std::string("native vision forward failed: ") +
                        q4::vision_status_string(vision_forward));

        const std::uint32_t multimodal_tokens[]{
                q4::kVisionStartTokenId, q4::kImageTokenId,
                q4::kVisionEndTokenId, 104u};
        const std::uint32_t multimodal_positions[]{
                0u, 1u, 2u, 3u,
                0u, 0u, 2u, 3u,
                0u, 0u, 2u, 3u,
        };
        q4::multimodal_text_prompt_view multimodal{};
        multimodal.token_ids = multimodal_tokens;
        multimodal.token_count = 4u;
        multimodal.projected_vision_embeddings_device = device_projected;
        multimodal.projected_vision_tokens = 1u;
        multimodal.mrope_positions = multimodal_positions;
        multimodal.mrope_position_tokens = 4u;

        error.clear();
        const q4::speculative_status multimodal_session_status =
                model->create_session(
                        session_config, stream, &session, &error);
        require_status(multimodal_session_status, q4::speculative_status::ok,
                       "native multimodal session: " + error);
        q4::text_token_result multimodal_first{};
        error.clear();
        const q4::speculative_status multimodal_prefill =
                session->target_prefill_multimodal(
                        multimodal, greedy_generation(), &multimodal_first,
                        &error);
        require_status(multimodal_prefill, q4::speculative_status::ok,
                       "native vision/MRoPE/MTP prefill: " + error);
        state = session->view();
        require(state.proposal_open && state.draft_available &&
                        state.draft_synchronized &&
                        state.metrics.primed_context_tokens == 3u,
                "vision prompt did not prime an aligned MTP context");

        q4::speculative_step_result multimodal_decoded{};
        error.clear();
        const q4::speculative_status multimodal_decode = session->decode(
                multimodal_first.token_id, greedy_generation(),
                &multimodal_decoded, &error);
        require_status(multimodal_decode, q4::speculative_status::ok,
                       "native vision continuous MTP decode: " + error);
        state = session->view();
        require(multimodal_decoded.authoritative.committed_context == 5u &&
                        state.metrics.drafted_tokens == 1u &&
                        state.proposal_open && state.draft_synchronized,
                "vision/MRoPE target and MTP continuation diverged");
        error.clear();
        require_status(session->disconnect(&error),
                       q4::speculative_status::ok,
                       "native multimodal disconnect: " + error);

        require(cudaFree(device_projected) == cudaSuccess &&
                        cudaFree(device_patches) == cudaSuccess,
                "cannot release direct vision/text handoff buffers");
        device_projected = nullptr;
        device_patches = nullptr;
        vision.reset();
        catalog.reset();

        std::cout
                << "qwen4exp-speculative-model-runtime: PASS "
                << "target=resident48 mtp=exact primed=2 verified="
                << q4::kSpeculativeDefaultDraftWidth
                << " accepted=" << block_state.metrics.accepted_tokens
                << " rejected=" << block_state.metrics.rejected_tokens
                << " emitted=" << block_state.metrics.block_emitted_tokens
                << " continuation=staged vision=resident27 "
                << "mrope=11/11/10 multimodal_mtp=continuous "
                << "disconnect=rollback\n";
        session.reset();
        model.reset();
        require(cudaStreamDestroy(stream) == cudaSuccess,
                "cannot destroy native MTP runtime stream");
        stream = nullptr;
    } catch (...) {
        if (stream != nullptr) (void)cudaStreamDestroy(stream);
        throw;
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 1 || argc == 2,
                "usage: speculative-model-test [MODEL_ROOT]");
        static_assert(q4::kSpeculativeDraftWidth == 1u,
                      "public MTP ABI admits one proposal per transaction");
        static_assert(q4::kSpeculativeDefaultDraftWidth == 3u &&
                              q4::kSpeculativeMaxDraftWidth == 4u,
                      "native speculative compositor must remain width 3/4");
        static_assert(q4::speculative_accepts(7u, 7u),
                      "equal tokens must be accepted");
        static_assert(!q4::speculative_accepts(7u, 8u),
                      "different tokens must be rejected");
        constexpr q4::mtp_contract_report exact_contract{
            q4::kMtpTensorCount, 1u, true, true, true, true};
        static_assert(q4::speculative_exact_mtp_contract(exact_contract),
                      "exact qwen4_exp MTP contract must be admitted");
        constexpr q4::mtp_contract_report historical_or_partial_contract{
            q4::kMtpTensorCount, 1u, false, true, true, true};
        static_assert(!q4::speculative_exact_mtp_contract(
                              historical_or_partial_contract),
                      "non-exact MTP namespace must fail closed");
        q4::speculative_session_config invalid_narrow{};
        q4::speculative_session_config target_only_config{};
        target_only_config.enable_draft = false;
        target_only_config.proposal_width = 0u;
        require_status(q4::speculative_session_validate_config(target_only_config),
                       q4::speculative_status::ok,
                       "explicit target-only session admission");
        invalid_narrow.proposal_width = 1u;
        require_status(q4::speculative_session_validate_config(invalid_narrow),
                       q4::speculative_status::unsupported_config,
                       "native width-one session rejection");
        q4::speculative_session_config invalid_wide{};
        invalid_wide.proposal_width = 5u;
        require_status(q4::speculative_session_validate_config(invalid_wide),
                       q4::speculative_status::unsupported_config,
                       "native width-five session rejection");

        accepted_gate();
        mismatch_commits_context_and_continues();
        prompt_context_commit_is_not_scored_as_a_draft();
        stage_error_rolls_back_then_uses_target();
        target_error_rolls_back();
        commit_error_falls_back_without_changing_target();
        disconnect_rolls_back_open_proposal();
        rollback_failure_fails_closed_on_disconnect();
        reset_reenables_a_clean_draft();
        target_only_is_identical();

        if (argc == 2) native_runtime_gate(argv[1]);

        std::cout
                << "qwen4exp-speculative-model-test: PASS "
                << "accept=deterministic mismatch=commit+continue "
                << "prompt=primed-not-scored "
                << "stage_error=rollback+target_only "
                << "target_error=rollback commit_error=target_only "
                << "disconnect=rollback rollback_failure=fail_closed "
                << "reset=resync target_only=authoritative\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-speculative-model-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
