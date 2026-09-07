#include "axiom/qwen4exp/text_model.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::text_model_status actual,
                    q4::text_model_status wanted,
                    const char *operation,
                    const std::string &detail) {
    if (actual == wanted) return;
    fail(std::string(operation) + ": expected " +
         q4::text_model_status_string(wanted) + ", got " +
         q4::text_model_status_string(actual) +
         (detail.empty() ? std::string{} : ": " + detail));
}

class stream_owner final {
public:
    stream_owner() {
        require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");
    }
    ~stream_owner() {
        if (stream_ != nullptr) (void)cudaStreamDestroy(stream_);
    }
    stream_owner(const stream_owner &) = delete;
    stream_owner &operator=(const stream_owner &) = delete;
    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

bool bitwise_equal(float left, float right) {
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 ||
        (argc == 3 && std::string(argv[2]) != "--skip-payload-hash")) {
        std::cerr << "usage: " << argv[0]
                  << " MODEL_ROOT [--skip-payload-hash]\n";
        return 2;
    }

    try {
        int device = -1;
        cudaDeviceProp properties{};
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12,
                "real qwen4_exp text gate requires an SM120 GPU");

        stream_owner stream;
        q4::text_model_config model_config{};
        model_config.verify_payload_hashes = argc == 2;
        require(q4::text_model_validate_config(model_config) ==
                        q4::text_model_status::ok,
                "compiled text model config is not the native contract");
        q4::text_model_config invalid_budget = model_config;
        invalid_budget.decoder_safety_margin_bytes =
                invalid_budget.decoder_hard_limit_bytes;
        require(q4::text_model_validate_config(invalid_budget) ==
                        q4::text_model_status::unsupported_config,
                "invalid decoder residency budget was accepted");

        std::unique_ptr<q4::resident_text_model> model;
        std::string error;
        require_status(q4::resident_text_model::load(
                               argv[1], model_config, stream.get(), &model,
                               &error),
                       q4::text_model_status::ok, "bounded model load", error);
        require(model != nullptr && model->initialized(),
                "bounded text model did not initialize");
        const q4::text_model_metrics model_metrics = model->metrics();
        require(model_metrics.admitted_layers == q4::kTextModelLayerCount &&
                        model_metrics.decoder_residency.all_48_admitted,
                "bounded model did not admit all 48 checkpoint layers");
        require(model_metrics.loaded_layers <= q4::kTextModelLayerCount &&
                        model_metrics.decoder_residency.resident_capacity > 0u &&
                        model_metrics.decoder_residency.resident_capacity <=
                                q4::kTextModelLayerCount &&
                        model_metrics.decoder_residency.peak_observed_bytes <=
                                model_metrics.decoder_residency.usable_limit_bytes,
                "decoder residency metrics violate the bounded contract");
        if (model_metrics.decoder_residency.mode ==
            q4::decoder_residency_mode::resident_all) {
            require(model_metrics.loaded_layers == q4::kTextModelLayerCount &&
                            model_metrics.decoder_residency.no_reload_decode_path,
                    "resident-all model is not fully resident and reload-free");
        } else {
            require(model_metrics.loaded_layers <=
                            model_metrics.decoder_residency.resident_capacity,
                    "bounded-window model exceeded its resident capacity");
        }
        require(model_metrics.load_ms > 0.0,
                "complete text model load time was not measured");
        require(model_metrics.output_head_resident_bytes == 2555924480ULL,
                "real output-head resident byte contract changed");
        require(model_metrics.expert_arena_resident_bytes == 27648160ULL,
                "shared expert-slot arena byte contract changed");
        require(!model_config.verify_payload_hashes ||
                        model_metrics.payload_hashes_verified,
                "full checkpoint payload identity was not verified");

        q4::text_session_config session_config{};
        session_config.context_capacity = 4u;
        std::unique_ptr<q4::text_session> first;
        std::unique_ptr<q4::text_session> second;
        error.clear();
        require_status(model->create_session(session_config, stream.get(),
                                             &first, &error),
                       q4::text_model_status::ok, "first session creation",
                       error);
        error.clear();
        require_status(model->create_session(session_config, stream.get(),
                                             &second, &error),
                       q4::text_model_status::ok, "second session creation",
                       error);
        require(first && second && first->initialized() && second->initialized(),
                "fresh text sessions did not initialize");

        q4::text_generation_config greedy{};
        greedy.mode = q4::text_selection_mode::argmax;
        constexpr std::uint32_t prompt_token = 248053u;
        q4::text_token_result first_next{};
        q4::text_token_result second_next{};
        error.clear();
        require_status(first->prefill(&prompt_token, 1u, greedy, &first_next,
                                      &error),
                       q4::text_model_status::ok, "first real prefill", error);
        error.clear();
        require_status(second->prefill(&prompt_token, 1u, greedy, &second_next,
                                       &error),
                       q4::text_model_status::ok, "second real prefill", error);
        require(first_next.token_id < q4::kTextModelVocab &&
                        std::isfinite(first_next.selected_logit),
                "first generated token is invalid");
        require(first_next.token_id == second_next.token_id &&
                        bitwise_equal(first_next.selected_logit,
                                      second_next.selected_logit),
                "two fresh sessions are not bitwise deterministic");
        require(first_next.committed_context == 1u &&
                        second_next.committed_context == 1u,
                "prefill did not atomically commit one prompt token");

        q4::text_token_result decoded{};
        error.clear();
        require_status(first->decode(first_next.token_id, greedy, &decoded,
                                     &error),
                       q4::text_model_status::ok, "one-token real decode",
                       error);
        require(decoded.token_id < q4::kTextModelVocab &&
                        std::isfinite(decoded.selected_logit) &&
                        decoded.committed_context == 2u,
                "one-token decode result is invalid");

        const q4::text_session_view before_rejection = second->view();
        q4::text_token_result rejected_result{};
        error.clear();
        const q4::text_model_status rejected = second->decode(
                static_cast<std::uint32_t>(q4::kTextModelVocab), greedy,
                &rejected_result, &error);
        require(rejected != q4::text_model_status::ok,
                "out-of-vocabulary token was not rejected");
        const q4::text_session_view after_rejection = second->view();
        require(after_rejection.decoder.committed_tokens ==
                        before_rejection.decoder.committed_tokens &&
                        !after_rejection.decoder.transaction_open &&
                        !after_rejection.decoder.poisoned &&
                        after_rejection.metrics.rolled_back_transactions >= 1u,
                "failed decode did not roll back the complete 48-layer stack");

        error.clear();
        require_status(second->reset(&error), q4::text_model_status::ok,
                       "complete session reset", error);
        const q4::text_session_view reset_view = second->view();
        require(reset_view.decoder.committed_tokens == 0u &&
                        !reset_view.decoder.transaction_open &&
                        !reset_view.decoder.poisoned &&
                        reset_view.metrics.resets == 1u,
                "session reset did not clear all model state");

        const q4::text_session_view first_view = first->view();
        require(first_view.decoder.committed_tokens == 2u &&
                        first_view.metrics.prompt_input_tokens == 1u &&
                        first_view.metrics.decode_input_tokens == 1u &&
                        first_view.metrics.generated_tokens == 2u &&
                        first_view.metrics.committed_transactions == 2u &&
                        first_view.metrics.ttft_measured &&
                        first_view.metrics.ttft_ms > 0.0 &&
                        first_view.metrics.last_decode_ms > 0.0,
                "measured text session metrics are incomplete");

        std::cout << std::fixed << std::setprecision(3)
                  << "qwen4exp-text-model-test: PASS sm="
                  << properties.major << properties.minor
                  << " admitted_layers=" << model_metrics.admitted_layers
                  << " resident_layers=" << model_metrics.loaded_layers
                  << " residency_mode="
                  << q4::decoder_residency_mode_string(
                             model_metrics.decoder_residency.mode)
                  << " residency_capacity="
                  << model_metrics.decoder_residency.resident_capacity
                  << " residency_peak_bytes="
                  << model_metrics.decoder_residency.peak_observed_bytes
                  << " context_native=" << q4::kTextModelNativeContext
                  << " session_capacity=" << session_config.context_capacity
                  << " payload_hash="
                  << (model_metrics.payload_hashes_verified ? "verified" : "skipped")
                  << " generated_first=" << first_next.token_id
                  << " generated_decode=" << decoded.token_id
                  << " deterministic=2/2"
                  << " transaction=commit+rollback+reset"
                  << " load_ms=" << model_metrics.load_ms
                  << " ttft_ms=" << first_view.metrics.ttft_ms
                  << " decode_ms=" << first_view.metrics.last_decode_ms
                  << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-text-model-test: FAIL: " << exception.what()
                  << '\n';
        return 1;
    }
}
