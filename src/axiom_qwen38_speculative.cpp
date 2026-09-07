#include "axiom/qwen38_speculative.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#include "axiom/qwen38_dspark_compute.h"

#ifndef AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION
#define AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION 0u
#endif

#if AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION == 1u
#include <cuda_runtime_api.h>
#define AXIOM_QWEN38_SPECULATIVE_BACKEND_READY 1
#else
#define AXIOM_QWEN38_SPECULATIVE_BACKEND_READY 0
#endif

#if AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
namespace {

using steady_clock = std::chrono::steady_clock;

bool debug_enabled() {
    const char *value = std::getenv("AXIOM_QWEN38_SPEC_DEBUG");
    return value && value[0] == '1' && value[1] == '\0';
}

void debug_prefill(const char *stage, int status, uint32_t position) {
    if (debug_enabled()) {
        std::fprintf(stderr, "qwen38_spec_prefill stage=%s status=%d position=%u\n",
                     stage, status, position);
    }
}

uint64_t elapsed_ns(steady_clock::time_point begin, steady_clock::time_point end) {
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return value > 0 ? static_cast<uint64_t>(value) : 0u;
}

void add_saturated(uint64_t *counter, uint64_t value) {
    if (!counter) return;
    if (value > std::numeric_limits<uint64_t>::max() - *counter) {
        *counter = std::numeric_limits<uint64_t>::max();
    } else {
        *counter += value;
    }
}

}  // namespace
#endif

struct axiom_qwen38_speculative {
    axiom_qwen38_model *target = nullptr;
    const axiom_qwen38_dspark *dspark = nullptr;
    axiom_qwen38_dspark_compute *compute = nullptr;
    int device = -1;
    mutable std::mutex mutex;
    bool poisoned = false;
    bool device_session_started = false;
    bool has_device_target = false;
    bool device_cycle_graph_ready = false;
    void *device_session_stream = nullptr;
    axiom_qwen38_speculative_counters counters{};
    axiom_qwen38_speculative_device_target_binding device_target{};
#if AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    cudaGraphExec_t device_cycle_graph_exec = nullptr;
    uint32_t *proposal_device = nullptr;
    uint32_t *proposal_host = nullptr;
#endif
};

#if AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
namespace {

static_assert(AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS == AXIOM_QWEN38_DSPARK_BLOCK_SIZE,
              "DSpark block ABI mismatch");
static_assert(AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ==
                      AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH,
              "target temporal verify ABI mismatch");
static_assert(AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT ==
                      AXIOM_QWEN38_DSPARK_TARGET_FEATURES,
              "target tap ABI mismatch");

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

uint32_t longest_prefix(
        const uint32_t draft[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS],
        const uint32_t target[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH]) {
    uint32_t prefix = 0u;
    while (prefix < AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS &&
           draft[prefix] == target[prefix]) {
        ++prefix;
    }
    return prefix;
}

bool valid_target_capabilities(const axiom_qwen38_model_dspark_temporal_capabilities &caps) {
    return caps.abi_version == AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION &&
            caps.temporal_m8_available == 1u &&
            caps.temporal_verify_width == AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH &&
            caps.draft_block_size == AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS &&
            caps.target_tap_count == AXIOM_QWEN38_DSPARK_TARGET_FEATURES &&
            caps.hidden_size == AXIOM_QWEN38_DSPARK_HIDDEN &&
            caps.requires_gdn_causal_rows == 1u &&
            caps.requires_attention_triangular_kv == 1u &&
            caps.requires_prefix_state_install == 1u &&
            caps.temporal_m8_implemented == 1u && caps.temporal_m8_validated == 1u;
}

bool valid_dspark_contract(const axiom_qwen38_dspark_forward_contract &contract) {
    constexpr uint32_t expected_taps[AXIOM_QWEN38_DSPARK_TARGET_FEATURES] = {
            4u, 16u, 28u, 40u, 52u};
    bool taps_match = true;
    for (uint32_t index = 0u;
         index < AXIOM_QWEN38_DSPARK_TARGET_FEATURES; ++index) {
        taps_match = taps_match &&
                contract.target_layer_ids[index] == expected_taps[index];
    }
    return contract.abi_version == AXIOM_ABI_VERSION && taps_match &&
            contract.scalar_dtype == AXIOM_TENSOR_DTYPE_F32 &&
            contract.hidden_size == AXIOM_QWEN38_DSPARK_HIDDEN &&
            contract.target_feature_count == AXIOM_QWEN38_DSPARK_TARGET_FEATURES &&
            contract.draft_layers == AXIOM_QWEN38_DSPARK_LAYERS &&
            contract.attention_heads == AXIOM_QWEN38_DSPARK_HEADS &&
            contract.key_value_heads == AXIOM_QWEN38_DSPARK_KV_HEADS &&
            contract.head_dim == AXIOM_QWEN38_DSPARK_HEAD_DIM &&
            contract.block_size == AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS &&
            contract.verify_width == AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH &&
            contract.mask_token_id == AXIOM_QWEN38_DSPARK_MASK_TOKEN_ID &&
            contract.markov_rank == AXIOM_QWEN38_DSPARK_MARKOV_RANK &&
            contract.confidence_features == AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES &&
            contract.requires_draft_kv_injection == 1u &&
            contract.requires_draft_kv_transaction == 1u &&
            contract.uses_target_embedding == 1u && contract.uses_target_lm_head == 1u &&
            contract.attention_is_noncausal_over_draft_block == 1u &&
            contract.rmsnorm_zero_centered == 0u &&
            contract.max_context == 262144u &&
            contract.rms_norm_eps == 1.0e-6f &&
            contract.confidence_head_alpha == 1.0f &&
            contract.rope_theta == 10000000.0f &&
            contract.yarn_factor == 32.0f &&
            contract.yarn_beta_fast == 32.0f &&
            contract.yarn_beta_slow == 1.0f &&
            contract.yarn_original_context == 8192u;
}

bool valid_device_target_binding(const axiom_qwen38_speculative_device_target_binding *binding) {
    return binding &&
            binding->abi_version == AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION &&
            binding->transaction_begin && binding->transaction_verify &&
            binding->transaction_commit && binding->transaction_abort && binding->transaction_finalize &&
            (binding->flags & ~AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_FLAG_CAPTURE_SAFE) == 0u;
}

bool valid_device_verify_view(const axiom_qwen38_speculative_device_verify_view &view) {
    return view.abi_version == AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION &&
            view.target_taps_device && view.target_token_ids_device && view.target_logits_device &&
            view.target_tap_count == AXIOM_QWEN38_DSPARK_TARGET_FEATURES &&
            view.target_tap_tokens == AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH &&
            view.target_tap_hidden_size == AXIOM_QWEN38_DSPARK_HIDDEN && view.reserved == 0u;
}

int abort_device_step(axiom_qwen38_speculative *speculative, void *stream, bool target_open) {
    int first = AXIOM_OK;
    if (target_open && speculative->has_device_target) {
        const int rc = speculative->device_target.transaction_abort(
                speculative->device_target.user_data, stream);
        if (rc != AXIOM_OK) first = rc;
    }
    if (speculative->device_session_started) {
        const int rc = axiom_qwen38_dspark_compute_device_session_abort(
                speculative->compute, stream);
        speculative->device_session_started = false;
        speculative->device_session_stream = nullptr;
        if (first == AXIOM_OK && rc != AXIOM_OK) first = rc;
    }
    return first;
}

/* Capture exactly one fixed gamma=7 cycle. The target binding is required to
 * own a graph-safe device transaction: these host callbacks run while capture
 * is being built, never on replay. Every replay is therefore only CUDA graph
 * nodes (proposal, causal M8, accept, tap injection, finalize, advance). */
int prepare_device_cycle_graph(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_device_target_binding *binding) {
    if (!speculative || !binding ||
        (binding->flags & AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_FLAG_CAPTURE_SAFE) == 0u ||
        speculative->device_cycle_graph_exec || speculative->device_cycle_graph_ready) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_dspark_compute_device_state state{};
    state.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    int rc = axiom_qwen38_dspark_compute_device_state_get(speculative->compute, &state);
    if (rc != AXIOM_OK) return rc;
    if (state.device_session_active != 0u) return AXIOM_ERR_INVALID_ARGUMENT;

    cudaStream_t stream = nullptr;
    cudaError_t cuda_rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_rc != cudaSuccess) return cuda_status(cuda_rc);
    bool compute_capture_open = false;
    bool stream_capture_open = false;
    bool target_open = false;
    cudaGraph_t graph = nullptr;

    rc = axiom_qwen38_dspark_compute_device_graph_capture_begin(speculative->compute);
    if (rc == AXIOM_OK) compute_capture_open = true;
    if (rc == AXIOM_OK) {
        cuda_rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (cuda_rc != cudaSuccess) rc = cuda_status(cuda_rc);
        else stream_capture_open = true;
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_cycle_begin_enqueue(
                speculative->compute, reinterpret_cast<void *>(stream));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_propose_enqueue(
                speculative->compute, reinterpret_cast<void *>(stream));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_build_verify_input(
                speculative->compute, reinterpret_cast<void *>(stream));
    }
    if (rc == AXIOM_OK) {
        rc = binding->transaction_begin(
                binding->user_data, state.anchor_position_device, reinterpret_cast<void *>(stream));
        if (rc == AXIOM_OK) target_open = true;
    }

    axiom_qwen38_speculative_device_verify_view verify{};
    verify.abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = binding->transaction_verify(
                binding->user_data, state.verify_tokens_device, state.anchor_position_device,
                AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH, reinterpret_cast<void *>(stream), &verify);
    }
    if (rc == AXIOM_OK && !valid_device_verify_view(verify)) rc = AXIOM_ERR_RUNTIME;

    axiom_qwen38_dspark_compute_device_target_view target_view{};
    if (rc == AXIOM_OK) {
        target_view.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
        target_view.target_taps_device = verify.target_taps_device;
        target_view.target_token_ids_device = verify.target_token_ids_device;
        target_view.target_logits_device = verify.target_logits_device;
        target_view.columns = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
        target_view.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_dspark_compute_device_accept_greedy(
                speculative->compute, &target_view);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_inject_target(
                speculative->compute, &target_view);
    }
    if (rc == AXIOM_OK) {
        rc = binding->transaction_finalize(
                binding->user_data, state.target_commit_prefix_device, state.async_status_device,
                reinterpret_cast<void *>(stream));
        if (rc == AXIOM_OK) target_open = false;
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_advance(
                speculative->compute, reinterpret_cast<void *>(stream));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_device_history_pack_enqueue(
                state.proposal_tokens_device, state.proposal_confidence_device,
                state.accepted_prefix_device,
                state.continuation_token_device, state.async_status_device,
                state.anchor_position_device, state.target_commit_prefix_device,
                state.history_device,
                reinterpret_cast<void *>(stream));
    }

    cudaError_t end_rc = cudaSuccess;
    if (stream_capture_open) {
        end_rc = cudaStreamEndCapture(stream, &graph);
        stream_capture_open = false;
    }
    if (compute_capture_open) {
        const int capture_end_rc = axiom_qwen38_dspark_compute_device_graph_capture_end(
                speculative->compute);
        compute_capture_open = false;
        if (rc == AXIOM_OK && capture_end_rc != AXIOM_OK) rc = capture_end_rc;
    }
    if (rc == AXIOM_OK && end_rc != cudaSuccess) rc = cuda_status(end_rc);
    if (rc != AXIOM_OK || !graph) {
        if (target_open) (void) binding->transaction_abort(binding->user_data, reinterpret_cast<void *>(stream));
        if (graph) (void) cudaGraphDestroy(graph);
        (void) cudaStreamDestroy(stream);
        return rc != AXIOM_OK ? rc : AXIOM_ERR_CUDA;
    }

    cudaGraphExec_t graph_exec = nullptr;
    cuda_rc = cudaGraphInstantiate(&graph_exec, graph, 0u);
    (void) cudaGraphDestroy(graph);
    (void) cudaStreamDestroy(stream);
    if (cuda_rc != cudaSuccess) return cuda_status(cuda_rc);
    speculative->device_cycle_graph_exec = graph_exec;
    speculative->device_cycle_graph_ready = true;
    return AXIOM_OK;
}

bool valid_verify_result(
        const axiom_qwen38_model_dspark_verify_block8_result &result,
        uint32_t snapshot_position,
        const uint32_t draft[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS],
        uint32_t *out_prefix) {
    if (!out_prefix || result.abi_version != AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION ||
        result.snapshot_position != snapshot_position ||
        result.position_after_verify != snapshot_position + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
        result.target_tap_token_start_position != snapshot_position ||
        result.target_tap_count != AXIOM_QWEN38_DSPARK_TARGET_FEATURES ||
        result.target_tap_tokens != AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
        result.target_tap_hidden_size != AXIOM_QWEN38_DSPARK_HIDDEN ||
        !result.target_aux_hidden) {
        return false;
    }
    for (uint32_t index = 0u; index < AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS; ++index) {
        if (draft[index] >= AXIOM_QWEN38_DSPARK_VOCAB) return false;
    }
    for (uint32_t index = 0u; index < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++index) {
        if (result.target_token_ids[index] >= AXIOM_QWEN38_DSPARK_VOCAB ||
            !std::isfinite(result.target_logits[index])) {
            return false;
        }
    }
    const uint32_t prefix = longest_prefix(draft, result.target_token_ids);
    if (result.accepted_draft_prefix != prefix ||
        result.bonus_token_id != result.target_token_ids[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS] ||
        !std::isfinite(result.bonus_logit)) {
        return false;
    }
    *out_prefix = prefix;
    return true;
}

int abort_open_transactions(
        axiom_qwen38_speculative *speculative,
        axiom_qwen38_model_transaction **target_transaction,
        bool *compute_transaction_open) {
    int first_error = AXIOM_OK;
    if (target_transaction && *target_transaction) {
        const int rc = axiom_qwen38_model_transaction_abort(*target_transaction);
        *target_transaction = nullptr;
        if (rc != AXIOM_OK) first_error = rc;
    }
    if (compute_transaction_open && *compute_transaction_open) {
        const int rc = axiom_qwen38_dspark_compute_transaction_abort(
                speculative->compute, nullptr);
        *compute_transaction_open = false;
        if (first_error == AXIOM_OK && rc != AXIOM_OK) first_error = rc;
    }
    return first_error;
}

void record_failure(
        axiom_qwen38_speculative *speculative,
        int status,
        bool aborted,
        bool poison,
        uint64_t total_ns) {
    speculative->counters.last_status = static_cast<uint32_t>(status);
    add_saturated(&speculative->counters.failed_steps, 1u);
    if (aborted) add_saturated(&speculative->counters.aborted_steps, 1u);
    add_saturated(&speculative->counters.total_ns, total_ns);
    if (poison) speculative->poisoned = true;
    speculative->counters.poisoned = speculative->poisoned ? 1u : 0u;
}

void record_prefill_failure(
        axiom_qwen38_speculative *speculative,
        int status,
        bool poison,
        uint64_t failed_tokens,
        uint64_t target_ns,
        uint64_t commit_ns,
        uint64_t inject_ns,
        uint64_t total_ns) {
    speculative->counters.last_status = static_cast<uint32_t>(status);
    add_saturated(&speculative->counters.prefill_failed_tokens, failed_tokens);
    add_saturated(&speculative->counters.prefill_target_ns, target_ns);
    add_saturated(&speculative->counters.prefill_commit_ns, commit_ns);
    add_saturated(&speculative->counters.prefill_inject_ns, inject_ns);
    add_saturated(&speculative->counters.prefill_total_ns, total_ns);
    if (poison) speculative->poisoned = true;
    speculative->counters.poisoned = speculative->poisoned ? 1u : 0u;
}

}  // namespace
#endif

extern "C" int axiom_qwen38_speculative_backend_available(void) {
    return AXIOM_QWEN38_SPECULATIVE_BACKEND_READY;
}

extern "C" int axiom_qwen38_speculative_create(
        axiom_qwen38_model *target,
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_speculative_config *config,
        axiom_qwen38_speculative **out) {
    if (out) *out = nullptr;
    if (!target || !dspark || !compute || !config || !out ||
        config->abi_version != AXIOM_ABI_VERSION || config->device < 0 ||
        config->reserved != 0u || config->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(target, &capabilities);
    if (rc != AXIOM_OK) return rc;
    if (!valid_target_capabilities(capabilities)) return AXIOM_ERR_NOT_IMPLEMENTED;

    axiom_qwen38_dspark_forward_contract contract{};
    contract.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_qwen38_dspark_forward_contract_get(dspark, &contract);
    if (rc != AXIOM_OK) return rc;
    if (!valid_dspark_contract(contract)) return AXIOM_ERR_INVALID_ARGUMENT;

    axiom_qwen38_dspark_compute_info compute_info{};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_qwen38_dspark_compute_info_get(compute, &compute_info);
    if (rc != AXIOM_OK) return rc;
    if (compute_info.device != static_cast<uint32_t>(config->device) ||
        compute_info.block_size != AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ||
        compute_info.transaction_open != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_qwen38_speculative *speculative = new (std::nothrow) axiom_qwen38_speculative();
    if (!speculative) return AXIOM_ERR_BUDGET;
    speculative->target = target;
    speculative->dspark = dspark;
    speculative->compute = compute;
    speculative->device = config->device;
    speculative->counters.abi_version = AXIOM_ABI_VERSION;
    if (cudaSetDevice(config->device) != cudaSuccess) {
        delete speculative;
        return AXIOM_ERR_CUDA;
    }
    void *device_buffer = nullptr;
    cudaError_t cuda_rc = cudaMalloc(
            &device_buffer,
            AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS * sizeof(uint32_t));
    if (cuda_rc == cudaSuccess) {
        speculative->proposal_device = static_cast<uint32_t *>(device_buffer);
        void *host_buffer = nullptr;
        cuda_rc = cudaHostAlloc(
                &host_buffer,
                AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS * sizeof(uint32_t),
                cudaHostAllocPortable);
        speculative->proposal_host = static_cast<uint32_t *>(host_buffer);
    }
    if (cuda_rc != cudaSuccess) {
        axiom_qwen38_speculative_destroy(speculative);
        return cuda_status(cuda_rc);
    }
    *out = speculative;
    return AXIOM_OK;
#endif
}

extern "C" void axiom_qwen38_speculative_destroy(axiom_qwen38_speculative *speculative) {
    if (!speculative) return;
#if AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    if (speculative->device >= 0) (void)cudaSetDevice(speculative->device);
    if (speculative->device_cycle_graph_exec) {
        (void)cudaGraphExecDestroy(speculative->device_cycle_graph_exec);
        speculative->device_cycle_graph_exec = nullptr;
    }
    if (speculative->proposal_host) (void)cudaFreeHost(speculative->proposal_host);
    if (speculative->proposal_device) (void)cudaFree(speculative->proposal_device);
#endif
    delete speculative;
}

extern "C" int axiom_qwen38_speculative_device_target_bind(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_device_target_binding *binding) {
    if (!speculative || !binding) return AXIOM_ERR_INVALID_ARGUMENT;
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    if (!valid_device_target_binding(binding)) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((binding->flags & AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_FLAG_CAPTURE_SAFE) == 0u) {
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (speculative->poisoned || speculative->has_device_target ||
        speculative->device_session_started) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_dspark_compute_device_state state{};
    state.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    int rc = axiom_qwen38_dspark_compute_device_state_get(speculative->compute, &state);
    if (rc != AXIOM_OK) return rc;
    if (state.device_session_active != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    rc = prepare_device_cycle_graph(speculative, binding);
    if (rc != AXIOM_OK) return rc;
    speculative->device_target = *binding;
    speculative->has_device_target = true;
    return AXIOM_OK;
#endif
}

extern "C" int axiom_qwen38_speculative_device_target_bind_qwen38_model(
        axiom_qwen38_speculative *speculative) {
    if (!speculative || !speculative->target) return AXIOM_ERR_INVALID_ARGUMENT;
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    axiom_qwen38_speculative_device_target_binding binding{};
    binding.abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
    const int rc = axiom_qwen38_model_dspark_device_target_binding_get(
            speculative->target, &binding);
    if (rc != AXIOM_OK) return rc;
    return axiom_qwen38_speculative_device_target_bind(speculative, &binding);
#endif
}

extern "C" int axiom_qwen38_speculative_prefill_token(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_prefill_request *request,
        axiom_qwen38_speculative_prefill_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_ABI_VERSION || out->abi_version != AXIOM_ABI_VERSION ||
        request->token_id >= AXIOM_QWEN38_DSPARK_VOCAB || request->stream != nullptr ||
        request->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    const auto total_begin = steady_clock::now();
    add_saturated(&speculative->counters.prefill_attempted_tokens, 1u);
    if (speculative->poisoned) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, AXIOM_ERR_RUNTIME, true, 1u, 0u, 0u, 0u, total);
        return AXIOM_ERR_RUNTIME;
    }

    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
            speculative->target, &capabilities);
    axiom_qwen38_dspark_compute_info compute_info{};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_info_get(speculative->compute, &compute_info);
    }
    const uint32_t snapshot_position = axiom_qwen38_model_position(speculative->target);
    if (rc != AXIOM_OK || !valid_target_capabilities(capabilities) ||
        compute_info.transaction_open != 0u || compute_info.committed_position != snapshot_position) {
        const int status = rc != AXIOM_OK
                ? rc
                : (!valid_target_capabilities(capabilities)
                           ? AXIOM_ERR_NOT_IMPLEMENTED : AXIOM_ERR_RUNTIME);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(speculative, status, false, 1u, 0u, 0u, 0u, total);
        return status;
    }

    axiom_qwen38_model_transaction *target_transaction = nullptr;
    const auto target_begin = steady_clock::now();
    rc = axiom_qwen38_model_transaction_begin(
            speculative->target, &target_transaction);
    axiom_qwen38_model_dspark_prefill_token_result target_result{};
    target_result.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = request->embedding_device
                ? axiom_qwen38_model_transaction_prefill_embedding(
                        target_transaction, request->token_id,
                        request->embedding_device, nullptr, &target_result)
                : axiom_qwen38_model_transaction_prefill_token(
                        target_transaction, request->token_id, nullptr, &target_result);
    }
    debug_prefill("target", rc, snapshot_position);
    const uint64_t target_ns = elapsed_ns(target_begin, steady_clock::now());
    if (rc == AXIOM_OK &&
        (target_result.abi_version != AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION ||
         target_result.token_position != snapshot_position ||
         target_result.target_tap_count != AXIOM_QWEN38_DSPARK_TARGET_FEATURES ||
         target_result.temporal_tokens != 1u ||
         target_result.hidden_size != AXIOM_QWEN38_DSPARK_HIDDEN ||
         !target_result.target_aux_hidden ||
         target_result.target_token_id >= AXIOM_QWEN38_DSPARK_VOCAB ||
         !std::isfinite(target_result.target_logit))) {
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc != AXIOM_OK) {
        int abort_rc = AXIOM_OK;
        if (target_transaction) {
            abort_rc = axiom_qwen38_model_transaction_abort(target_transaction);
            target_transaction = nullptr;
        }
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, rc, abort_rc != AXIOM_OK, 1u, target_ns, 0u, 0u, total);
        return rc;
    }

    const auto inject_begin = steady_clock::now();
    axiom_qwen38_dspark_compute_inject_request inject{};
    inject.abi_version = AXIOM_ABI_VERSION;
    inject.target_taps = target_result.target_aux_hidden;
    inject.target_position = snapshot_position;
    inject.columns = 1u;
    inject.stream = nullptr;
    rc = axiom_qwen38_dspark_compute_inject_target(speculative->compute, &inject);
    debug_prefill("inject", rc, snapshot_position);
    const bool compute_advanced = rc == AXIOM_OK;
    compute_info = {};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_info_get(speculative->compute, &compute_info);
    }
    debug_prefill("inject_info", rc, compute_info.committed_position);
    if (rc == AXIOM_OK &&
        (compute_info.transaction_open != 0u ||
         compute_info.committed_position != snapshot_position + 1u)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    const uint64_t inject_ns = elapsed_ns(inject_begin, steady_clock::now());
    if (rc != AXIOM_OK) {
        int abort_rc = AXIOM_OK;
        if (target_transaction) {
            abort_rc = axiom_qwen38_model_transaction_abort(target_transaction);
            target_transaction = nullptr;
        }
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, rc, compute_advanced || abort_rc != AXIOM_OK, 1u,
                target_ns, 0u, inject_ns, total);
        return rc;
    }

    const auto commit_begin = steady_clock::now();
    rc = axiom_qwen38_model_transaction_commit_prefix(target_transaction, 1u);
    debug_prefill("target_commit", rc, axiom_qwen38_model_position(speculative->target));
    /* commit_prefix consumes the opaque handle on both success and failure. */
    target_transaction = nullptr;
    const uint64_t commit_ns = elapsed_ns(commit_begin, steady_clock::now());
    if (rc == AXIOM_OK && axiom_qwen38_model_position(speculative->target) != snapshot_position + 1u) {
        rc = AXIOM_ERR_RUNTIME;
    }
    const uint64_t total_ns = elapsed_ns(total_begin, steady_clock::now());
    if (rc != AXIOM_OK) {
        /* DSpark already advanced; a failed authoritative commit cannot be
         * reconciled without resetting both objects. */
        record_prefill_failure(
                speculative, rc, true, 1u, target_ns, commit_ns, inject_ns, total_ns);
        return rc;
    }

    axiom_qwen38_speculative_prefill_result result{};
    result.abi_version = AXIOM_ABI_VERSION;
    result.token_position = snapshot_position;
    result.position_after_commit = snapshot_position + 1u;
    result.target_token_id = target_result.target_token_id;
    result.target_logit = target_result.target_logit;
    result.target_ns = target_ns;
    result.commit_ns = commit_ns;
    result.inject_ns = inject_ns;
    result.total_ns = total_ns;
    add_saturated(&speculative->counters.prefill_committed_tokens, 1u);
    add_saturated(&speculative->counters.prefill_target_ns, target_ns);
    add_saturated(&speculative->counters.prefill_commit_ns, commit_ns);
    add_saturated(&speculative->counters.prefill_inject_ns, inject_ns);
    add_saturated(&speculative->counters.prefill_total_ns, total_ns);
    speculative->counters.last_status = AXIOM_OK;
    speculative->counters.poisoned = 0u;
    *out = result;
    return AXIOM_OK;
#endif
}

extern "C" int axiom_qwen38_speculative_prefill_block8(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_prefill_block8_request *request,
        axiom_qwen38_speculative_prefill_block8_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_ABI_VERSION || out->abi_version != AXIOM_ABI_VERSION ||
        request->stream != nullptr || request->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t time = 0u; time < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++time) {
        if (request->token_ids[time] >= AXIOM_QWEN38_DSPARK_VOCAB) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    constexpr uint64_t kTokens = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    const auto total_begin = steady_clock::now();
    add_saturated(&speculative->counters.prefill_attempted_tokens, kTokens);
    if (speculative->poisoned) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, AXIOM_ERR_RUNTIME, true, kTokens, 0u, 0u, 0u, total);
        return AXIOM_ERR_RUNTIME;
    }

    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
            speculative->target, &capabilities);
    axiom_qwen38_dspark_compute_info compute_info{};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_info_get(speculative->compute, &compute_info);
    }
    const uint32_t snapshot_position = axiom_qwen38_model_position(speculative->target);
    if (rc != AXIOM_OK || !valid_target_capabilities(capabilities) ||
        compute_info.transaction_open != 0u || compute_info.committed_position != snapshot_position) {
        const int status = rc != AXIOM_OK
                ? rc
                : (!valid_target_capabilities(capabilities)
                           ? AXIOM_ERR_NOT_IMPLEMENTED : AXIOM_ERR_RUNTIME);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, status, false, kTokens, 0u, 0u, 0u, total);
        return status;
    }

    axiom_qwen38_model_transaction *target_transaction = nullptr;
    const auto target_begin = steady_clock::now();
    rc = axiom_qwen38_model_transaction_begin(
            speculative->target, &target_transaction);
    axiom_qwen38_model_dspark_verify_block8_result target_result{};
    target_result.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_verify_block8(
                target_transaction, request->token_ids, nullptr, &target_result);
    }
    debug_prefill("target_block8", rc, snapshot_position);
    const uint64_t target_ns = elapsed_ns(target_begin, steady_clock::now());
    if (rc == AXIOM_OK &&
        (target_result.abi_version != AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION ||
         target_result.snapshot_position != snapshot_position ||
         target_result.position_after_verify !=
                 snapshot_position + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
         target_result.target_tap_token_start_position != snapshot_position ||
         target_result.target_tap_count != AXIOM_QWEN38_DSPARK_TARGET_FEATURES ||
         target_result.target_tap_tokens != AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH ||
         target_result.target_tap_hidden_size != AXIOM_QWEN38_DSPARK_HIDDEN ||
         !target_result.target_aux_hidden)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    for (uint32_t time = 0u;
         rc == AXIOM_OK && time < AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH; ++time) {
        if (target_result.target_token_ids[time] >= AXIOM_QWEN38_DSPARK_VOCAB ||
            !std::isfinite(target_result.target_logits[time])) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc != AXIOM_OK) {
        int abort_rc = AXIOM_OK;
        if (target_transaction) {
            abort_rc = axiom_qwen38_model_transaction_abort(target_transaction);
            target_transaction = nullptr;
        }
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, rc, abort_rc != AXIOM_OK, kTokens,
                target_ns, 0u, 0u, total);
        return rc;
    }

    const auto inject_begin = steady_clock::now();
    axiom_qwen38_dspark_compute_inject_request inject{};
    inject.abi_version = AXIOM_ABI_VERSION;
    inject.target_taps = target_result.target_aux_hidden;
    inject.target_position = snapshot_position;
    inject.columns = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    inject.stream = nullptr;
    rc = axiom_qwen38_dspark_compute_inject_target(speculative->compute, &inject);
    debug_prefill("inject_block8", rc, snapshot_position);
    const bool compute_advanced = rc == AXIOM_OK;
    compute_info = {};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_info_get(speculative->compute, &compute_info);
    }
    if (rc == AXIOM_OK &&
        (compute_info.transaction_open != 0u ||
         compute_info.committed_position !=
                 snapshot_position + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    const uint64_t inject_ns = elapsed_ns(inject_begin, steady_clock::now());
    if (rc != AXIOM_OK) {
        int abort_rc = AXIOM_OK;
        if (target_transaction) {
            abort_rc = axiom_qwen38_model_transaction_abort(target_transaction);
            target_transaction = nullptr;
        }
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_prefill_failure(
                speculative, rc, compute_advanced || abort_rc != AXIOM_OK, kTokens,
                target_ns, 0u, inject_ns, total);
        return rc;
    }

    const auto commit_begin = steady_clock::now();
    rc = axiom_qwen38_model_transaction_commit_prefix(
            target_transaction, AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH);
    target_transaction = nullptr;
    const uint64_t commit_ns = elapsed_ns(commit_begin, steady_clock::now());
    if (rc == AXIOM_OK && axiom_qwen38_model_position(speculative->target) !=
            snapshot_position + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH) {
        rc = AXIOM_ERR_RUNTIME;
    }
    const uint64_t total_ns = elapsed_ns(total_begin, steady_clock::now());
    if (rc != AXIOM_OK) {
        record_prefill_failure(
                speculative, rc, true, kTokens,
                target_ns, commit_ns, inject_ns, total_ns);
        return rc;
    }

    axiom_qwen38_speculative_prefill_block8_result result{};
    result.abi_version = AXIOM_ABI_VERSION;
    result.token_start_position = snapshot_position;
    result.position_after_commit =
            snapshot_position + AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
    std::memcpy(result.target_token_ids, target_result.target_token_ids,
                sizeof(result.target_token_ids));
    std::memcpy(result.target_logits, target_result.target_logits,
                sizeof(result.target_logits));
    result.target_ns = target_ns;
    result.commit_ns = commit_ns;
    result.inject_ns = inject_ns;
    result.total_ns = total_ns;
    add_saturated(&speculative->counters.prefill_committed_tokens, kTokens);
    add_saturated(&speculative->counters.prefill_target_ns, target_ns);
    add_saturated(&speculative->counters.prefill_commit_ns, commit_ns);
    add_saturated(&speculative->counters.prefill_inject_ns, inject_ns);
    add_saturated(&speculative->counters.prefill_total_ns, total_ns);
    speculative->counters.last_status = AXIOM_OK;
    speculative->counters.poisoned = 0u;
    *out = result;
    return AXIOM_OK;
#endif
}

extern "C" int axiom_qwen38_speculative_step(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_step_request *request,
        axiom_qwen38_speculative_step_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_ABI_VERSION || out->abi_version != AXIOM_ABI_VERSION ||
        request->anchor_token_id >= AXIOM_QWEN38_DSPARK_VOCAB || request->stream != nullptr ||
        request->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    const auto total_begin = steady_clock::now();
    add_saturated(&speculative->counters.attempted_steps, 1u);
    if (speculative->poisoned) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, AXIOM_ERR_RUNTIME, false, true, total);
        return AXIOM_ERR_RUNTIME;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, AXIOM_ERR_CUDA, false, false, total);
        return AXIOM_ERR_CUDA;
    }

    axiom_qwen38_dspark_compute_info compute_info{};
    compute_info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_qwen38_dspark_compute_info_get(speculative->compute, &compute_info);
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
                speculative->target, &capabilities);
    }
    const uint32_t snapshot_position = axiom_qwen38_model_position(speculative->target);
    if (rc != AXIOM_OK || !valid_target_capabilities(capabilities) ||
        compute_info.transaction_open != 0u || compute_info.committed_position != snapshot_position) {
        const int status = rc != AXIOM_OK
                ? rc
                : (!valid_target_capabilities(capabilities)
                           ? AXIOM_ERR_NOT_IMPLEMENTED : AXIOM_ERR_RUNTIME);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, status, false, false, total);
        return status;
    }

    bool compute_transaction_open = false;
    axiom_qwen38_model_transaction *target_transaction = nullptr;
    rc = axiom_qwen38_dspark_compute_transaction_begin(
            speculative->compute, snapshot_position, nullptr);
    if (rc == AXIOM_OK) compute_transaction_open = true;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_begin(
                speculative->target, &target_transaction);
    }
    if (rc != AXIOM_OK) {
        const bool had_transaction = compute_transaction_open || target_transaction;
        const int abort_rc = abort_open_transactions(
                speculative, &target_transaction, &compute_transaction_open);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, rc, had_transaction, abort_rc != AXIOM_OK, total);
        return rc;
    }

    const auto propose_begin = steady_clock::now();
    axiom_qwen38_dspark_compute_propose_request propose{};
    propose.abi_version = AXIOM_ABI_VERSION;
    propose.anchor_token = request->anchor_token_id;
    propose.anchor_position = snapshot_position;
    propose.proposal_tokens = AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS;
    propose.out_tokens = speculative->proposal_device;
    propose.stream = nullptr;
    rc = axiom_qwen38_dspark_compute_propose(speculative->compute, &propose);
    const uint64_t propose_ns = elapsed_ns(propose_begin, steady_clock::now());

    const auto copy_begin = steady_clock::now();
    if (rc == AXIOM_OK) {
        cudaError_t cuda_rc = cudaMemcpyAsync(
                speculative->proposal_host, speculative->proposal_device,
                AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, nullptr);
        if (cuda_rc == cudaSuccess) cuda_rc = cudaStreamSynchronize(nullptr);
        if (cuda_rc != cudaSuccess) rc = cuda_status(cuda_rc);
    }
    const uint64_t copy_ns = elapsed_ns(copy_begin, steady_clock::now());

    uint32_t verify_input[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH]{};
    verify_input[0] = request->anchor_token_id;
    if (rc == AXIOM_OK) {
        std::memcpy(
                verify_input + 1u, speculative->proposal_host,
                AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS * sizeof(uint32_t));
    }
    axiom_qwen38_model_dspark_verify_block8_result verify{};
    verify.abi_version = AXIOM_ABI_VERSION;
    const auto verify_begin = steady_clock::now();
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_verify_block8(
                target_transaction, verify_input, nullptr, &verify);
    }
    const uint64_t verify_ns = elapsed_ns(verify_begin, steady_clock::now());

    uint32_t accepted_prefix = 0u;
    if (rc == AXIOM_OK && !valid_verify_result(
            verify, snapshot_position, speculative->proposal_host, &accepted_prefix)) {
        rc = AXIOM_ERR_RUNTIME;
    }

    const auto inject_begin = steady_clock::now();
    if (rc == AXIOM_OK) {
        axiom_qwen38_dspark_compute_inject_request inject{};
        inject.abi_version = AXIOM_ABI_VERSION;
        inject.target_taps = verify.target_aux_hidden;
        inject.target_position = snapshot_position;
        inject.columns = AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH;
        inject.stream = nullptr;
        rc = axiom_qwen38_dspark_compute_inject_target(speculative->compute, &inject);
    }
    const uint64_t inject_ns = elapsed_ns(inject_begin, steady_clock::now());

    if (rc != AXIOM_OK) {
        const int abort_rc = abort_open_transactions(
                speculative, &target_transaction, &compute_transaction_open);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        add_saturated(&speculative->counters.propose_ns, propose_ns);
        add_saturated(&speculative->counters.proposal_copy_ns, copy_ns);
        add_saturated(&speculative->counters.verify_ns, verify_ns);
        add_saturated(&speculative->counters.inject_ns, inject_ns);
        record_failure(speculative, rc, true, abort_rc != AXIOM_OK, total);
        return rc;
    }

    axiom_qwen38_speculative_step_result result{};
    result.abi_version = AXIOM_ABI_VERSION;
    result.snapshot_position = snapshot_position;
    result.position_after_verify = verify.position_after_verify;
    result.position_after_commit = snapshot_position + 1u + accepted_prefix;
    result.accepted_draft_prefix = accepted_prefix;
    result.emitted_token_count = accepted_prefix + 1u;
    std::memcpy(result.draft_token_ids, speculative->proposal_host,
                sizeof(result.draft_token_ids));
    std::memcpy(result.target_token_ids, verify.target_token_ids,
                sizeof(result.target_token_ids));
    std::memcpy(result.target_logits, verify.target_logits,
                sizeof(result.target_logits));
    for (uint32_t index = 0u; index < accepted_prefix; ++index) {
        result.emitted_token_ids[index] = speculative->proposal_host[index];
    }
    result.continuation_token_id = verify.target_token_ids[accepted_prefix];
    result.continuation_logit = verify.target_logits[accepted_prefix];
    result.emitted_token_ids[accepted_prefix] = result.continuation_token_id;
    result.full_block_accept = accepted_prefix == AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS ? 1u : 0u;
    result.inject_ns = inject_ns;
    result.propose_ns = propose_ns;
    result.proposal_copy_ns = copy_ns;
    result.verify_ns = verify_ns;

    const auto commit_begin = steady_clock::now();
    rc = axiom_qwen38_model_transaction_commit_prefix(
            target_transaction, 1u + accepted_prefix);
    /* commit_prefix consumes the opaque handle on both success and failure. */
    target_transaction = nullptr;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_transaction_commit(
                speculative->compute, accepted_prefix, nullptr);
        if (rc == AXIOM_OK) compute_transaction_open = false;
    }
    const uint64_t commit_ns = elapsed_ns(commit_begin, steady_clock::now());
    result.commit_ns = commit_ns;
    result.total_ns = elapsed_ns(total_begin, steady_clock::now());
    if (rc != AXIOM_OK) {
        (void)abort_open_transactions(
                speculative, &target_transaction, &compute_transaction_open);
        add_saturated(&speculative->counters.propose_ns, propose_ns);
        add_saturated(&speculative->counters.proposal_copy_ns, copy_ns);
        add_saturated(&speculative->counters.verify_ns, verify_ns);
        add_saturated(&speculative->counters.inject_ns, inject_ns);
        add_saturated(&speculative->counters.commit_ns, commit_ns);
        record_failure(speculative, rc, true, true, result.total_ns);
        return rc;
    }

    add_saturated(&speculative->counters.committed_steps, 1u);
    if (result.full_block_accept) {
        add_saturated(&speculative->counters.full_accept_steps, 1u);
    } else {
        add_saturated(&speculative->counters.correction_steps, 1u);
    }
    add_saturated(&speculative->counters.proposed_draft_tokens,
                  AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS);
    add_saturated(&speculative->counters.accepted_draft_tokens, accepted_prefix);
    add_saturated(&speculative->counters.rejected_draft_tokens,
                  AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS - accepted_prefix);
    add_saturated(&speculative->counters.emitted_tokens, result.emitted_token_count);
    add_saturated(&speculative->counters.authoritative_tail_tokens, 1u);
    add_saturated(&speculative->counters.inject_ns, inject_ns);
    add_saturated(&speculative->counters.propose_ns, propose_ns);
    add_saturated(&speculative->counters.proposal_copy_ns, copy_ns);
    add_saturated(&speculative->counters.verify_ns, verify_ns);
    add_saturated(&speculative->counters.commit_ns, commit_ns);
    add_saturated(&speculative->counters.total_ns, result.total_ns);
    speculative->counters.last_status = AXIOM_OK;
    speculative->counters.last_accepted_draft_prefix = accepted_prefix;
    speculative->counters.poisoned = 0u;
    *out = result;
    return AXIOM_OK;
#endif
}

extern "C" int axiom_qwen38_speculative_device_step_enqueue(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_device_step_request *request,
        axiom_qwen38_speculative_device_step_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION ||
        out->abi_version != AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION ||
        !request->anchor_token_device || !request->anchor_position_device || !request->stream ||
        request->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    const auto total_begin = steady_clock::now();
    add_saturated(&speculative->counters.attempted_steps, 1u);
    if (speculative->poisoned) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, AXIOM_ERR_RUNTIME, false, true, total);
        return AXIOM_ERR_RUNTIME;
    }
    if (!speculative->has_device_target) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, AXIOM_ERR_NOT_IMPLEMENTED, false, false, total);
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, AXIOM_ERR_CUDA, false, false, total);
        return AXIOM_ERR_CUDA;
    }

    axiom_qwen38_dspark_compute_device_state state{};
    state.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    int rc = axiom_qwen38_dspark_compute_device_state_get(speculative->compute, &state);
    if (rc == AXIOM_OK && (!speculative->device_cycle_graph_ready ||
                            !speculative->device_cycle_graph_exec)) {
        rc = AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (rc != AXIOM_OK) {
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, rc, false, false, total);
        return rc;
    }

    axiom_qwen38_dspark_compute_device_session_request session{};
    session.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    session.anchor_token_device = request->anchor_token_device;
    session.anchor_position_device = request->anchor_position_device;
    session.stream = request->stream;
    if (!speculative->device_session_started) {
        rc = axiom_qwen38_dspark_compute_device_session_begin(speculative->compute, &session);
        if (rc == AXIOM_OK) {
            speculative->device_session_started = true;
            speculative->device_session_stream = request->stream;
            rc = axiom_qwen38_model_dspark_device_session_begin(
                    speculative->target);
        }
    } else if (request->anchor_token_device != state.anchor_token_device ||
               request->anchor_position_device != state.anchor_position_device ||
               request->stream != speculative->device_session_stream) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaGraphLaunch(
                speculative->device_cycle_graph_exec, static_cast<cudaStream_t>(request->stream)));
    }

    if (rc != AXIOM_OK) {
        const bool had_device_session = speculative->device_session_started;
        void *abort_stream = had_device_session ? speculative->device_session_stream : request->stream;
        const int abort_rc = abort_device_step(speculative, abort_stream, false);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        record_failure(speculative, rc, had_device_session,
                       abort_rc != AXIOM_OK, total);
        return rc;
    }

    state = {};
    state.abi_version = AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION;
    rc = axiom_qwen38_dspark_compute_device_state_get(speculative->compute, &state);
    if (rc != AXIOM_OK) {
        const int abort_rc = abort_device_step(speculative, request->stream, false);
        const uint64_t total = elapsed_ns(total_begin, steady_clock::now());
        (void) abort_rc;
        record_failure(speculative, rc, true, true, total);
        return rc;
    }

    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->graph_replayed = 1u;
    out->proposal_tokens = state.proposal_tokens;
    out->verify_width = state.verify_width;
    out->proposal_tokens_device = state.proposal_tokens_device;
    out->verify_tokens_device = state.verify_tokens_device;
    out->accepted_prefix_device = state.accepted_prefix_device;
    out->target_commit_prefix_device = state.target_commit_prefix_device;
    out->continuation_token_device = state.continuation_token_device;
    out->continuation_logit_device = state.continuation_logit_device;
    out->next_anchor_token_device = state.anchor_token_device;
    out->next_anchor_position_device = state.anchor_position_device;
    out->anchor_position_device = state.anchor_position_device;
    out->async_status_device = state.async_status_device;
    out->history_device = state.history_device;

    /* The status is enqueue-time only. Accepted-prefix and emission counters
     * remain device-owned and are intentionally not read back here. */
    add_saturated(&speculative->counters.committed_steps, 1u);
    add_saturated(&speculative->counters.proposed_draft_tokens,
                  AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS);
    speculative->counters.last_status = AXIOM_OK;
    speculative->counters.poisoned = 0u;
    return AXIOM_OK;
#endif
}

extern "C" int axiom_qwen38_speculative_device_session_end(
        axiom_qwen38_speculative *speculative,
        void *stream,
        const uint32_t *committed_token_ids,
        const uint32_t committed_token_count) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    (void)stream;
    (void)committed_token_ids;
    (void)committed_token_count;
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (!stream ||
        (speculative->device_session_started &&
         stream != speculative->device_session_stream) ||
        (committed_token_count != 0u && !committed_token_ids)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        speculative->poisoned = true;
        return AXIOM_ERR_CUDA;
    }

    /* The clear is ordered after every graph replay on the same nonblocking
     * stream.  Draining once after it therefore establishes the ownership
     * handoff to the ABI-v1/default-stream canonical path without a race. */
    int rc = speculative->device_session_started
            ? abort_device_step(speculative, stream, false)
            : AXIOM_OK;
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_restore_position(
                speculative->target, committed_token_count);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_committed_history_install(
                speculative->target, committed_token_ids, committed_token_count);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_dspark_compute_restore_position(
                speculative->compute, committed_token_count);
    }
    if (rc != AXIOM_OK) speculative->poisoned = true;
    return rc;
#endif
}

extern "C" int axiom_qwen38_speculative_device_commit_limit_set(
        axiom_qwen38_speculative *speculative,
        const uint32_t max_commit_tokens,
        void *stream) {
    if (!speculative || !stream) return AXIOM_ERR_INVALID_ARGUMENT;
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    (void)max_commit_tokens;
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (speculative->poisoned) return AXIOM_ERR_RUNTIME;
    return axiom_qwen38_dspark_compute_device_commit_limit_set(
            speculative->compute, max_commit_tokens, stream);
#endif
}

extern "C" int axiom_qwen38_speculative_counters_get(
        const axiom_qwen38_speculative *speculative,
        axiom_qwen38_speculative_counters *out) {
    if (!speculative || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
#if !AXIOM_QWEN38_SPECULATIVE_BACKEND_READY
    return AXIOM_ERR_NOT_IMPLEMENTED;
#else
    std::lock_guard<std::mutex> lock(speculative->mutex);
    *out = speculative->counters;
    return AXIOM_OK;
#endif
}
