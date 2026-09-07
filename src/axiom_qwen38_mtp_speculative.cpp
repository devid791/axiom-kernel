#include "axiom/qwen38_mtp_speculative.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#include "axiom/qwen38_mtp_device_control.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kVocab = AXIOM_QWEN38_MODEL_VOCAB;
constexpr uint32_t kDraft = AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
constexpr uint32_t kWidth = AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH;
constexpr uint32_t kMaxStops = AXIOM_QWEN38_MTP_SPECULATIVE_MAX_STOP_TOKENS;

static_assert(kMaxStops == AXIOM_QWEN38_MTP_DEVICE_CONTROL_MAX_STOP_TOKENS,
              "controller and device stop bounds must match");
static_assert(kHidden == AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_HIDDEN,
              "controller persistent hidden geometry changed");

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    if (status == cudaErrorMemoryAllocation) return AXIOM_ERR_BUDGET;
    return AXIOM_ERR_CUDA;
}

void add_saturated(uint64_t *value, uint64_t increment) {
    if (!value) return;
    const uint64_t room = std::numeric_limits<uint64_t>::max() - *value;
    *value += std::min(room, increment);
}

int fast_graph_min_position_from_env(uint32_t *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = 0u;
    const char *raw = std::getenv("AXIOM_QWEN38_MTP_FAST_GRAPH_MIN_POSITION");
    if (!raw) return AXIOM_OK;
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = static_cast<uint32_t>(parsed);
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_mtp_speculative {
    axiom_qwen38_model *target = nullptr;
    axiom_qwen38_mtp_compute *mtp = nullptr;
    int device = -1;
    bool poisoned = false;
    bool device_graph_ready = false;
    bool device_stop_tokens_configured = false;
    bool device_session_started = false;
    void *device_session_stream = nullptr;
    void *device_pending_stream = nullptr;
    uint32_t device_session_start_position = 0u;
    uint32_t device_fast_graph_min_position = 0u;
    mutable std::mutex mutex;
    cudaGraphExec_t device_graph_exec_exact = nullptr;
    cudaGraphExec_t device_graph_exec_fast = nullptr;
    cudaGraphExec_t device_session_graph_exec = nullptr;

    uint32_t *verify_tokens_device = nullptr;
    uint32_t *proposal_device = nullptr;
    uint32_t *device_anchor_token = nullptr;
    uint32_t *device_anchor_position = nullptr;
    uint32_t *device_accepted_prefix = nullptr;
    uint32_t *device_commit_count = nullptr;
    uint32_t *device_commit_limit = nullptr;
    uint32_t *device_stop_token_ids = nullptr;
    uint32_t *device_stop_token_count = nullptr;
    uint32_t *device_emitted_token_count = nullptr;
    uint32_t *device_continuation_token = nullptr;
    float *device_continuation_logit = nullptr;
    uint32_t *device_emitted_tokens = nullptr;
    uint32_t *device_stop_detected = nullptr;
    uint32_t *device_matched_stop_token = nullptr;
    uint32_t *device_next_anchor_token = nullptr;
    uint32_t *device_next_anchor_position = nullptr;
    uint32_t *device_async_status = nullptr;
    uint32_t *host_seed_position = nullptr;
    float *draft_hidden[2]{};
    float *pending_hidden[2]{};
    uint32_t pending_index = 0u;
    float *device_pending_hidden = nullptr;
    float *verified_hidden = nullptr;
    float *catchup_prior = nullptr;
    float *catchup_output = nullptr;

    uint64_t attempted_steps = 0u;
    uint64_t committed_steps = 0u;
    uint64_t proposed_tokens = 0u;
    uint64_t accepted_tokens = 0u;
    uint64_t emitted_tokens = 0u;
    uint64_t full_accept_steps = 0u;
    uint64_t correction_steps = 0u;
    uint64_t failed_steps = 0u;
};

namespace {

float *current_pending(axiom_qwen38_mtp_speculative *speculative) {
    return speculative->pending_hidden[speculative->pending_index];
}

float *candidate_pending(axiom_qwen38_mtp_speculative *speculative) {
    return speculative->pending_hidden[speculative->pending_index ^ 1u];
}

bool controller_graph_state_valid(
        const axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative || speculative->device_session_graph_exec) return false;
    if (!speculative->device_graph_ready) {
        return !speculative->device_graph_exec_exact &&
                !speculative->device_graph_exec_fast;
    }
    return speculative->device_graph_exec_exact &&
            (speculative->device_fast_graph_min_position == 0u
                     ? !speculative->device_graph_exec_fast
                     : speculative->device_graph_exec_fast != nullptr);
}

int controller_idle_positions_locked(
        const axiom_qwen38_mtp_speculative *speculative,
        uint32_t *target_position,
        uint32_t *mtp_position) {
    if (!speculative || !target_position || !mtp_position || speculative->poisoned ||
        speculative->device_session_started || speculative->device_session_stream ||
        speculative->device_session_start_position != 0u ||
        speculative->pending_index > 1u || !controller_graph_state_valid(speculative) ||
        !speculative->target || !speculative->mtp ||
        !speculative->pending_hidden[0] || !speculative->pending_hidden[1] ||
        !speculative->device_pending_hidden || !speculative->host_seed_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_mtp_compute_info compute_info{};
    compute_info.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    int rc = axiom_qwen38_mtp_compute_info_get(speculative->mtp, &compute_info);
    axiom_qwen38_mtp_compute_device_info_v1 device_info{};
    device_info.abi_version = AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION;
    device_info.struct_size = sizeof(device_info);
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_mtp_compute_device_info_get_v1(
                speculative->mtp, &device_info, sizeof(device_info));
    }
    if (rc != AXIOM_OK ||
        compute_info.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute_info.transaction_begin_position != compute_info.committed_position ||
        compute_info.staged_position != compute_info.committed_position ||
        device_info.device_session_active != 0u ||
        device_info.graph_capture_active != 0u ||
        device_info.committed_position != compute_info.committed_position) {
        return rc != AXIOM_OK ? rc : AXIOM_ERR_INVALID_ARGUMENT;
    }
    *target_position = axiom_qwen38_model_position(speculative->target);
    *mtp_position = compute_info.committed_position;
    return AXIOM_OK;
}

int reset_controller_device_cursors(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t position) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
    auto clear = [](void *pointer, size_t bytes) {
        return pointer ? cuda_status(cudaMemset(pointer, 0, bytes))
                       : AXIOM_ERR_INVALID_ARGUMENT;
    };
    int rc = clear(speculative->verify_tokens_device,
                   static_cast<size_t>(kWidth) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->proposal_device,
            static_cast<size_t>(kDraft) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_anchor_token, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_accepted_prefix, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_commit_count, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_emitted_token_count, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_continuation_token, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_continuation_logit, sizeof(float));
    if (rc == AXIOM_OK) rc = clear(speculative->device_emitted_tokens,
            static_cast<size_t>(kWidth) * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_stop_detected, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_matched_stop_token, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_next_anchor_token, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = clear(speculative->device_async_status, sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            speculative->device_anchor_position, &position, sizeof(position),
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            speculative->device_next_anchor_position, &position, sizeof(position),
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) *speculative->host_seed_position = position;
    return rc;
}

void destroy_device_storage(axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return;
    (void)cudaSetDevice(speculative->device);
    if (speculative->device_graph_exec_fast) {
        (void)cudaGraphExecDestroy(speculative->device_graph_exec_fast);
        speculative->device_graph_exec_fast = nullptr;
    }
    if (speculative->device_graph_exec_exact) {
        (void)cudaGraphExecDestroy(speculative->device_graph_exec_exact);
        speculative->device_graph_exec_exact = nullptr;
    }
    speculative->device_session_graph_exec = nullptr;
    (void)cudaFreeHost(speculative->host_seed_position);
    (void)cudaFree(speculative->catchup_output);
    (void)cudaFree(speculative->catchup_prior);
    (void)cudaFree(speculative->verified_hidden);
    (void)cudaFree(speculative->device_pending_hidden);
    (void)cudaFree(speculative->pending_hidden[1]);
    (void)cudaFree(speculative->pending_hidden[0]);
    (void)cudaFree(speculative->draft_hidden[1]);
    (void)cudaFree(speculative->draft_hidden[0]);
    (void)cudaFree(speculative->device_async_status);
    (void)cudaFree(speculative->device_next_anchor_position);
    (void)cudaFree(speculative->device_next_anchor_token);
    (void)cudaFree(speculative->device_emitted_tokens);
    (void)cudaFree(speculative->device_continuation_logit);
    (void)cudaFree(speculative->device_continuation_token);
    (void)cudaFree(speculative->device_matched_stop_token);
    (void)cudaFree(speculative->device_stop_detected);
    (void)cudaFree(speculative->device_emitted_token_count);
    (void)cudaFree(speculative->device_stop_token_count);
    (void)cudaFree(speculative->device_stop_token_ids);
    (void)cudaFree(speculative->device_commit_limit);
    (void)cudaFree(speculative->device_commit_count);
    (void)cudaFree(speculative->device_accepted_prefix);
    (void)cudaFree(speculative->device_anchor_position);
    (void)cudaFree(speculative->device_anchor_token);
    (void)cudaFree(speculative->proposal_device);
    (void)cudaFree(speculative->verify_tokens_device);
    speculative->catchup_output = nullptr;
    speculative->catchup_prior = nullptr;
    speculative->verified_hidden = nullptr;
    speculative->device_pending_hidden = nullptr;
    speculative->pending_hidden[0] = nullptr;
    speculative->pending_hidden[1] = nullptr;
    speculative->draft_hidden[0] = nullptr;
    speculative->draft_hidden[1] = nullptr;
    speculative->device_async_status = nullptr;
    speculative->host_seed_position = nullptr;
    speculative->device_next_anchor_position = nullptr;
    speculative->device_next_anchor_token = nullptr;
    speculative->device_emitted_tokens = nullptr;
    speculative->device_continuation_logit = nullptr;
    speculative->device_continuation_token = nullptr;
    speculative->device_matched_stop_token = nullptr;
    speculative->device_stop_detected = nullptr;
    speculative->device_emitted_token_count = nullptr;
    speculative->device_stop_token_count = nullptr;
    speculative->device_stop_token_ids = nullptr;
    speculative->device_commit_limit = nullptr;
    speculative->device_commit_count = nullptr;
    speculative->device_accepted_prefix = nullptr;
    speculative->device_anchor_position = nullptr;
    speculative->device_anchor_token = nullptr;
    speculative->proposal_device = nullptr;
    speculative->verify_tokens_device = nullptr;
    speculative->device_graph_ready = false;
    speculative->device_pending_stream = nullptr;
}

int allocate_device_storage(axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const size_t hidden_row_bytes = static_cast<size_t>(kHidden) * sizeof(float);
    const size_t hidden_width_bytes = static_cast<size_t>(kWidth) * hidden_row_bytes;
    int rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->verify_tokens_device),
            static_cast<size_t>(kWidth) * sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->proposal_device),
            static_cast<size_t>(kDraft) * sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_anchor_token), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_anchor_position), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_accepted_prefix), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_commit_count), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_commit_limit), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_stop_token_ids),
            static_cast<size_t>(kMaxStops) * sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_stop_token_count), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_emitted_token_count), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_continuation_token), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_continuation_logit), sizeof(float)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_emitted_tokens),
            static_cast<size_t>(kWidth) * sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_stop_detected), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_matched_stop_token), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_next_anchor_token), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_next_anchor_position), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_async_status), sizeof(uint32_t)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMallocHost(
            reinterpret_cast<void **>(&speculative->host_seed_position), sizeof(uint32_t)));
    for (uint32_t index = 0u; index < 2u && rc == AXIOM_OK; ++index) {
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&speculative->draft_hidden[index]),
                hidden_row_bytes));
    }
    for (uint32_t index = 0u; index < 2u && rc == AXIOM_OK; ++index) {
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&speculative->pending_hidden[index]),
                hidden_row_bytes));
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->device_pending_hidden), hidden_row_bytes));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->verified_hidden), hidden_width_bytes));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->catchup_prior), hidden_width_bytes));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMalloc(
            reinterpret_cast<void **>(&speculative->catchup_output), hidden_width_bytes));
    if (rc != AXIOM_OK) destroy_device_storage(speculative);
    return rc;
}

bool state_positions_match(const axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative || !speculative->target || !speculative->mtp) return false;
    return axiom_qwen38_model_position(speculative->target) ==
            axiom_qwen38_mtp_compute_position(speculative->mtp);
}

bool target_capabilities_valid(const axiom_qwen38_model *target) {
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    const int rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
            target, &capabilities);
    return rc == AXIOM_OK && capabilities.temporal_m8_available == 1u &&
            capabilities.temporal_verify_width == kWidth &&
            capabilities.draft_block_size == kDraft &&
            capabilities.hidden_size == kHidden &&
            capabilities.requires_gdn_causal_rows == 1u &&
            capabilities.requires_attention_triangular_kv == 1u &&
            capabilities.requires_prefix_state_install == 1u &&
            capabilities.temporal_m8_implemented == 1u &&
            capabilities.temporal_m8_validated == 1u;
}

int abort_target(axiom_qwen38_model_transaction **transaction) {
    if (!transaction || !*transaction) return AXIOM_OK;
    const int rc = axiom_qwen38_model_transaction_abort(*transaction);
    *transaction = nullptr;
    return rc;
}

int abort_mtp(axiom_qwen38_mtp_compute *mtp, bool *open) {
    if (!open || !*open) return AXIOM_OK;
    const int rc = axiom_qwen38_mtp_compute_transaction_abort(mtp, nullptr);
    *open = false;
    return rc;
}

int abort_both(
        axiom_qwen38_mtp_speculative *speculative,
        axiom_qwen38_model_transaction **target_transaction,
        bool *mtp_open) {
    const int target_rc = abort_target(target_transaction);
    const int mtp_rc = abort_mtp(speculative->mtp, mtp_open);
    if (target_rc != AXIOM_OK || mtp_rc != AXIOM_OK) speculative->poisoned = true;
    return target_rc != AXIOM_OK ? target_rc : mtp_rc;
}

int copy_verified_hidden(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_model_dspark_capture_view &view,
        uint32_t expected_start_position) {
    if (!speculative || view.abi_version != AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_ABI_VERSION ||
        view.semantics_version != AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_SEMANTICS_VERSION ||
        view.token_start_position != expected_start_position || view.tokens != kWidth ||
        view.hidden_size != kHidden ||
        view.layout != AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_LAYOUT_TAP_TIME_HIDDEN ||
        view.storage != AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_STORAGE_F32_BF16_MATERIALIZED ||
        !view.target_last_hidden) {
        return AXIOM_ERR_RUNTIME;
    }
    return cuda_status(cudaMemcpyAsync(
            speculative->verified_hidden, view.target_last_hidden,
            static_cast<size_t>(kWidth) * kHidden * sizeof(float),
            cudaMemcpyDeviceToDevice, nullptr));
}

int stage_mtp_forward(
        axiom_qwen38_mtp_speculative *speculative,
        const uint32_t *tokens_device,
        const float *prior_hidden,
        float *out_hidden,
        uint32_t *out_top1,
        uint32_t columns,
        uint32_t first_position) {
    axiom_qwen38_mtp_compute_forward_request request{};
    request.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    request.token_ids_device = tokens_device;
    request.prior_hidden_device = prior_hidden;
    request.out_hidden_device = out_hidden;
    request.out_top1_device = out_top1;
    request.columns = columns;
    request.first_position = first_position;
    request.stream = nullptr;
    return axiom_qwen38_mtp_compute_forward(speculative->mtp, &request);
}

int stage_mtp_forward_device(
        axiom_qwen38_mtp_speculative *speculative,
        const uint32_t *tokens_device,
        const float *prior_hidden,
        float *out_hidden,
        uint32_t *out_top1,
        uint32_t columns,
        uint32_t position_offset,
        cudaStream_t stream) {
    axiom_qwen38_mtp_compute_device_forward_request_v1 request{};
    request.abi_version = AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.token_ids_device = tokens_device;
    request.prior_hidden_device = prior_hidden;
    request.out_hidden_device = out_hidden;
    request.out_top1_device = out_top1;
    request.base_position_device = speculative->device_anchor_position;
    request.position_offset = position_offset;
    request.columns = columns;
    request.async_status_device = speculative->device_async_status;
    request.stream = reinterpret_cast<void *>(stream);
    return axiom_qwen38_mtp_compute_forward_device_enqueue_v1(
            speculative->mtp, &request, sizeof(request));
}

int stage_mtp_kv_catchup_device(
        axiom_qwen38_mtp_speculative *speculative,
        const uint32_t *tokens_device,
        const float *prior_hidden,
        uint32_t columns,
        uint32_t position_offset,
        cudaStream_t stream) {
    axiom_qwen38_mtp_compute_device_kv_catchup_request_v1 request{};
    request.abi_version = AXIOM_QWEN38_MTP_COMPUTE_KV_CATCHUP_ABI_VERSION;
    request.struct_size = sizeof(request);
    request.token_ids_device = tokens_device;
    request.prior_hidden_device = prior_hidden;
    request.base_position_device = speculative->device_anchor_position;
    request.position_offset = position_offset;
    request.columns = columns;
    request.async_status_device = speculative->device_async_status;
    request.stream = reinterpret_cast<void *>(stream);
    return axiom_qwen38_mtp_compute_kv_catchup_device_enqueue_v1(
            speculative->mtp, &request, sizeof(request));
}

bool valid_target_device_view(
        const axiom_qwen38_model_dspark_device_target_verify_view &view) {
    return view.abi_version == AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION &&
            view.target_token_ids_device && view.target_logits_device &&
            view.target_tap_count == AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT &&
            view.target_tap_tokens == kWidth && view.target_tap_hidden_size == kHidden &&
            view.reserved == 0u;
}

bool valid_target_hidden_view(
        const axiom_qwen38_model_dspark_device_hidden_view &view) {
    return view.abi_version ==
                    AXIOM_QWEN38_MODEL_DSPARK_DEVICE_HIDDEN_VIEW_ABI_VERSION &&
            view.struct_size == sizeof(view) &&
            view.semantics_version ==
                    AXIOM_QWEN38_MODEL_DSPARK_DEVICE_HIDDEN_SEMANTICS_VERSION &&
            view.temporal_tokens == kWidth && view.hidden_size == kHidden &&
            view.token_stride_elements == kHidden &&
            view.layout == AXIOM_QWEN38_MODEL_DSPARK_DEVICE_HIDDEN_LAYOUT_TOKEN_HIDDEN &&
            view.storage ==
                    AXIOM_QWEN38_MODEL_DSPARK_DEVICE_HIDDEN_STORAGE_F32_BF16_MATERIALIZED &&
            view.flags == 0u && view.target_last_hidden_device;
}

int capture_device_graph(
        axiom_qwen38_mtp_speculative *speculative,
        cudaGraphExec_t *out) {
    if (!speculative || !out || *out || speculative->device_graph_ready ||
        speculative->device_graph_exec_exact ||
        speculative->device_graph_exec_fast ||
        speculative->device_session_started) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cudaStream_t stream = nullptr;
    cudaError_t cuda_rc = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_rc != cudaSuccess) return cuda_status(cuda_rc);

    bool compute_capture_open = false;
    bool stream_capture_open = false;
    bool target_open = false;
    cudaGraph_t graph = nullptr;
    int rc = axiom_qwen38_mtp_compute_device_graph_capture_begin(speculative->mtp);
    if (rc == AXIOM_OK) compute_capture_open = true;
    if (rc == AXIOM_OK) {
        cuda_rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (cuda_rc == cudaSuccess) stream_capture_open = true;
        else rc = cuda_status(cuda_rc);
    }
    for (uint32_t index = 0u; index < kDraft && rc == AXIOM_OK; ++index) {
        const uint32_t *token = index == 0u
                ? speculative->device_anchor_token
                : speculative->proposal_device + index - 1u;
        const float *prior = index == 0u
                ? speculative->device_pending_hidden
                : speculative->draft_hidden[(index - 1u) & 1u];
        rc = stage_mtp_forward_device(
                speculative, token, prior,
                speculative->draft_hidden[index & 1u],
                speculative->proposal_device + index, 1u, index, stream);
    }
    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_device_build_verify_request_v1 request{};
        request.abi_version = AXIOM_QWEN38_MTP_DEVICE_CONTROL_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_device = speculative->device_anchor_token;
        request.draft_token_ids_device = speculative->proposal_device;
        request.verify_token_ids_device = speculative->verify_tokens_device;
        request.async_status_device = speculative->device_async_status;
        request.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_mtp_device_build_verify_input_enqueue_v1(
                &request, sizeof(request));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_device_transaction_begin(
                speculative->target, speculative->device_anchor_position,
                reinterpret_cast<void *>(stream));
        target_open = rc == AXIOM_OK;
    }
    axiom_qwen38_model_dspark_device_target_verify_view target_view{};
    target_view.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_device_transaction_verify(
                speculative->target, speculative->verify_tokens_device,
                speculative->device_anchor_position, kWidth,
                reinterpret_cast<void *>(stream), &target_view);
    }
    if (rc == AXIOM_OK && !valid_target_device_view(target_view)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    axiom_qwen38_model_dspark_device_hidden_view hidden_view{};
    hidden_view.abi_version =
            AXIOM_QWEN38_MODEL_DSPARK_DEVICE_HIDDEN_VIEW_ABI_VERSION;
    hidden_view.struct_size = sizeof(hidden_view);
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_device_transaction_hidden_view_get(
                speculative->target, &hidden_view);
    }
    if (rc == AXIOM_OK && !valid_target_hidden_view(hidden_view)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_device_accept_request_v2 request{};
        request.abi_version = AXIOM_QWEN38_MTP_DEVICE_ACCEPT_V2_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_device = speculative->device_anchor_token;
        request.anchor_position_device = speculative->device_anchor_position;
        request.draft_token_ids_device = speculative->proposal_device;
        request.target_top1_device = target_view.target_token_ids_device;
        request.target_top1_logits_device = target_view.target_logits_device;
        request.commit_limit_device = speculative->device_commit_limit;
        request.stop_token_ids_device = speculative->device_stop_token_ids;
        request.stop_token_count_device = speculative->device_stop_token_count;
        request.accepted_prefix_device = speculative->device_accepted_prefix;
        request.target_commit_count_device = speculative->device_commit_count;
        request.emitted_token_count_device = speculative->device_emitted_token_count;
        request.continuation_token_device = speculative->device_continuation_token;
        request.continuation_logit_device = speculative->device_continuation_logit;
        request.emitted_token_ids_device = speculative->device_emitted_tokens;
        request.stop_detected_device = speculative->device_stop_detected;
        request.matched_stop_token_device = speculative->device_matched_stop_token;
        request.next_anchor_token_device = speculative->device_next_anchor_token;
        request.next_anchor_position_device = speculative->device_next_anchor_position;
        request.async_status_device = speculative->device_async_status;
        request.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_mtp_device_accept_greedy_enqueue_v2(
                &request, sizeof(request));
    }
    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_device_build_catchup_request_v1 request{};
        request.abi_version = AXIOM_QWEN38_MTP_DEVICE_CONTROL_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.pending_hidden_device = speculative->device_pending_hidden;
        request.target_final_hidden_device = hidden_view.target_last_hidden_device;
        request.catchup_prior_device = speculative->catchup_prior;
        request.async_status_device = speculative->device_async_status;
        request.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_mtp_device_build_catchup_prior_enqueue_v1(
                &request, sizeof(request));
    }
    if (rc == AXIOM_OK) {
        /* Preserve the checkpoint's exact M8 GEMM shape for all cache rows.
         * Although proposal zero already touched the anchor row, reducing the
         * catch-up to M7 changes cuBLAS accumulation and is not bit-identical.
         * Stop after K/V store: attention, MLP, final hidden and LM-head output
         * are never consumed. */
        rc = stage_mtp_kv_catchup_device(
                speculative, speculative->verify_tokens_device,
                speculative->catchup_prior, kWidth, 0u, stream);
    }
    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_device_select_pending_request_v2 request{};
        request.abi_version = AXIOM_QWEN38_MTP_DEVICE_SELECT_V2_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.target_final_hidden_device = hidden_view.target_last_hidden_device;
        request.target_commit_count_device = speculative->device_commit_count;
        request.next_pending_hidden_device = candidate_pending(speculative);
        request.async_status_device = speculative->device_async_status;
        request.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_mtp_device_select_next_pending_enqueue_v2(
                &request, sizeof(request));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_device_transaction_finalize(
                speculative->target, speculative->device_commit_count,
                speculative->device_async_status, reinterpret_cast<void *>(stream));
        if (rc == AXIOM_OK) target_open = false;
    }
    if (rc == AXIOM_OK) {
        axiom_qwen38_mtp_device_publish_state_request_v2 request{};
        request.abi_version = AXIOM_QWEN38_MTP_DEVICE_PUBLISH_V2_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.candidate_anchor_token_device = speculative->device_next_anchor_token;
        request.candidate_anchor_position_device = speculative->device_next_anchor_position;
        request.candidate_pending_hidden_device = candidate_pending(speculative);
        request.target_commit_count_device = speculative->device_commit_count;
        request.async_status_device = speculative->device_async_status;
        request.authoritative_anchor_token_device = speculative->device_anchor_token;
        request.authoritative_anchor_position_device = speculative->device_anchor_position;
        request.authoritative_pending_hidden_device = speculative->device_pending_hidden;
        request.stream = reinterpret_cast<void *>(stream);
        rc = axiom_qwen38_mtp_device_publish_state_enqueue_v2(
                &request, sizeof(request));
    }

    cudaError_t end_rc = cudaSuccess;
    if (stream_capture_open) {
        end_rc = cudaStreamEndCapture(stream, &graph);
        stream_capture_open = false;
    }
    if (compute_capture_open) {
        const int capture_rc =
                axiom_qwen38_mtp_compute_device_graph_capture_end(speculative->mtp);
        compute_capture_open = false;
        if (rc == AXIOM_OK && capture_rc != AXIOM_OK) rc = capture_rc;
    }
    if (rc == AXIOM_OK && end_rc != cudaSuccess) rc = cuda_status(end_rc);
    if (rc != AXIOM_OK || !graph) {
        if (target_open) {
            (void)axiom_qwen38_model_dspark_device_transaction_abort(
                    speculative->target, reinterpret_cast<void *>(stream));
        }
        if (graph) (void)cudaGraphDestroy(graph);
        (void)cudaStreamDestroy(stream);
        return rc != AXIOM_OK ? rc : AXIOM_ERR_CUDA;
    }

    cudaGraphExec_t graph_exec = nullptr;
    cuda_rc = cudaGraphInstantiate(&graph_exec, graph, 0u);
    (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
    if (cuda_rc != cudaSuccess) return cuda_status(cuda_rc);
    *out = graph_exec;
    return AXIOM_OK;
}

int destroy_graph_exec(cudaGraphExec_t *graph_exec) {
    if (!graph_exec || !*graph_exec) return AXIOM_OK;
    const cudaError_t rc = cudaGraphExecDestroy(*graph_exec);
    if (rc == cudaSuccess) *graph_exec = nullptr;
    return cuda_status(rc);
}

int prepare_device_graph(axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative || speculative->device_graph_ready ||
        speculative->device_graph_exec_exact ||
        speculative->device_graph_exec_fast ||
        speculative->device_session_graph_exec ||
        speculative->device_session_started) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    cudaGraphExec_t exact = nullptr;
    cudaGraphExec_t fast = nullptr;
    int rc = axiom_qwen38_model_dspark_device_temporal_exact_set(
            speculative->target, 1);
    if (rc == AXIOM_OK) rc = capture_device_graph(speculative, &exact);

    const bool capture_fast = speculative->device_fast_graph_min_position > 0u;
    if (capture_fast && rc == AXIOM_OK) {
        rc = axiom_qwen38_model_dspark_device_temporal_exact_set(
                speculative->target, 0);
    }
    if (capture_fast && rc == AXIOM_OK) {
        rc = capture_device_graph(speculative, &fast);
    }
    if (capture_fast) {
        const int restore_rc =
                axiom_qwen38_model_dspark_device_temporal_exact_set(
                        speculative->target, 1);
        if (rc == AXIOM_OK && restore_rc != AXIOM_OK) rc = restore_rc;
    }

    if (rc != AXIOM_OK) {
        (void)destroy_graph_exec(&fast);
        (void)destroy_graph_exec(&exact);
        /* Retain any handle CUDA refused to destroy so reset/destruction can
         * retry instead of silently leaking an unreachable executable. */
        speculative->device_graph_exec_fast = fast;
        speculative->device_graph_exec_exact = exact;
        speculative->poisoned = true;
        return rc;
    }

    speculative->device_graph_exec_exact = exact;
    speculative->device_graph_exec_fast = fast;
    speculative->device_graph_ready = true;
    return AXIOM_OK;
}

int rollback_mtp_after_target_commit_failure(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t snapshot_position) {
    const int rc = axiom_qwen38_mtp_compute_restore_position(
            speculative->mtp, snapshot_position);
    speculative->poisoned = true;
    return rc;
}

}  // namespace

extern "C" int axiom_qwen38_mtp_speculative_create_v1(
        axiom_qwen38_model *target,
        axiom_qwen38_mtp_compute *mtp_compute,
        const axiom_qwen38_mtp_speculative_config *config,
        uint64_t config_bytes,
        axiom_qwen38_mtp_speculative **out) {
    if (out) *out = nullptr;
    if (!target || !mtp_compute || !config || !out ||
        config_bytes != sizeof(*config) ||
        config->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        config->struct_size != sizeof(*config) || config->device > INT32_MAX ||
        config->reserved != 0u || config->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_mtp_compute_info compute_info{};
    compute_info.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    int rc = axiom_qwen38_mtp_compute_info_get(mtp_compute, &compute_info);
    uint32_t fast_graph_min_position = 0u;
    if (rc == AXIOM_OK) {
        rc = fast_graph_min_position_from_env(&fast_graph_min_position);
    }
    const axiom_qwen38_rope_profile target_rope =
            axiom_qwen38_model_rope_profile(target);
    const axiom_qwen38_rope_profile mtp_rope =
            axiom_qwen38_mtp_compute_rope_profile(mtp_compute);
    if (rc != AXIOM_OK || compute_info.device != config->device ||
        axiom_qwen38_model_device(target) != static_cast<int>(config->device) ||
        target_rope == AXIOM_QWEN38_ROPE_PROFILE_INVALID ||
        mtp_rope == AXIOM_QWEN38_ROPE_PROFILE_INVALID ||
        target_rope != mtp_rope ||
        compute_info.hidden_size != kHidden || compute_info.vocab_size != kVocab ||
        compute_info.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
        compute_info.committed_position != 0u ||
        axiom_qwen38_model_position(target) != 0u ||
        !target_capabilities_valid(target)) {
        return rc != AXIOM_OK ? rc : AXIOM_ERR_INVALID_ARGUMENT;
    }
    auto *speculative = new (std::nothrow) axiom_qwen38_mtp_speculative();
    if (!speculative) return AXIOM_ERR_BUDGET;
    speculative->target = target;
    speculative->mtp = mtp_compute;
    speculative->device = static_cast<int>(config->device);
    speculative->device_fast_graph_min_position = fast_graph_min_position;
    rc = allocate_device_storage(speculative);
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->pending_hidden[0], 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->pending_hidden[1], 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->device_pending_hidden, 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->device_async_status, 0, sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        const uint32_t commit_limit = kWidth;
        rc = cuda_status(cudaMemcpy(
                speculative->device_commit_limit, &commit_limit,
                sizeof(commit_limit), cudaMemcpyHostToDevice));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->device_stop_token_ids, 0,
                static_cast<size_t>(kMaxStops) * sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->device_stop_token_count, 0, sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaDeviceSynchronize());
    if (rc != AXIOM_OK) {
        destroy_device_storage(speculative);
        delete speculative;
        return rc;
    }
    *out = speculative;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_mtp_speculative_destroy(
        axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return;
    (void)cudaSetDevice(speculative->device);
    (void)cudaDeviceSynchronize();
    destroy_device_storage(speculative);
    delete speculative;
}

extern "C" int axiom_qwen38_mtp_speculative_reset(
        axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = cuda_status(cudaDeviceSynchronize());
    const int fast_destroy_rc = destroy_graph_exec(
            &speculative->device_graph_exec_fast);
    const int exact_destroy_rc = destroy_graph_exec(
            &speculative->device_graph_exec_exact);
    if (rc == AXIOM_OK && fast_destroy_rc != AXIOM_OK) rc = fast_destroy_rc;
    if (rc == AXIOM_OK && exact_destroy_rc != AXIOM_OK) rc = exact_destroy_rc;
    if (rc == AXIOM_OK) rc = axiom_qwen38_model_reset(speculative->target);
    if (rc == AXIOM_OK) rc = axiom_qwen38_mtp_compute_reset(speculative->mtp);
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->pending_hidden[0], 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->pending_hidden[1], 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemset(
                speculative->device_pending_hidden, 0,
                static_cast<size_t>(kHidden) * sizeof(float)));
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaDeviceSynchronize());
    speculative->pending_index = 0u;
    speculative->device_graph_ready = false;
    speculative->device_session_started = false;
    speculative->device_session_stream = nullptr;
    speculative->device_pending_stream = nullptr;
    speculative->device_session_start_position = 0u;
    speculative->device_session_graph_exec = nullptr;
    speculative->poisoned = rc != AXIOM_OK;
    return rc;
}

extern "C" int axiom_qwen38_mtp_speculative_persistent_state_export_v1(
        axiom_qwen38_mtp_speculative *speculative,
        axiom_qwen38_mtp_speculative_persistent_state_v1 *out,
        uint64_t state_bytes) {
    if (!speculative || !out || state_bytes != sizeof(*out) ||
        out->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_STATE_ABI_VERSION ||
        out->struct_size != sizeof(*out)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(speculative->mutex);
    uint32_t target_position = 0u;
    uint32_t mtp_position = 0u;
    int rc = controller_idle_positions_locked(
            speculative, &target_position, &mtp_position);
    if (rc != AXIOM_OK || target_position != mtp_position) {
        return rc != AXIOM_OK ? rc : AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        speculative->poisoned = true;
        return AXIOM_ERR_CUDA;
    }
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc != AXIOM_OK) {
        speculative->poisoned = true;
        return rc;
    }
    speculative->device_pending_stream = nullptr;

    axiom_qwen38_mtp_speculative_persistent_state_v1 snapshot{};
    snapshot.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_STATE_ABI_VERSION;
    snapshot.struct_size = sizeof(snapshot);
    snapshot.position = target_position;
    snapshot.hidden_elements = kHidden;
    snapshot.flags = 0u;
    rc = cuda_status(cudaMemcpy(
            snapshot.pending_hidden, current_pending(speculative),
            sizeof(snapshot.pending_hidden), cudaMemcpyDeviceToHost));
    if (rc == AXIOM_OK) {
        for (uint32_t index = 0u; index < kHidden; ++index) {
            if (!std::isfinite(snapshot.pending_hidden[index])) {
                rc = AXIOM_ERR_RUNTIME;
                break;
            }
        }
    }
    if (rc != AXIOM_OK) {
        if (rc == AXIOM_ERR_CUDA) speculative->poisoned = true;
        return rc;
    }
    std::memcpy(out, &snapshot, sizeof(snapshot));
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_persistent_state_import_v1(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_persistent_state_v1 *state,
        uint64_t state_bytes) {
    if (!speculative || !state || state_bytes != sizeof(*state) ||
        state->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_STATE_ABI_VERSION ||
        state->struct_size != sizeof(*state) || state->hidden_elements != kHidden ||
        state->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0u; index < kHidden; ++index) {
        if (!std::isfinite(state->pending_hidden[index])) return AXIOM_ERR_RUNTIME;
    }

    std::lock_guard<std::mutex> lock(speculative->mutex);
    uint32_t target_position = 0u;
    uint32_t mtp_position = 0u;
    int rc = controller_idle_positions_locked(
            speculative, &target_position, &mtp_position);
    if (rc != AXIOM_OK || state->position != target_position ||
        state->position != mtp_position) {
        return rc != AXIOM_OK ? rc : AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        speculative->poisoned = true;
        return AXIOM_ERR_CUDA;
    }
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            speculative->pending_hidden[0], state->pending_hidden,
            sizeof(state->pending_hidden), cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            speculative->pending_hidden[1], state->pending_hidden,
            sizeof(state->pending_hidden), cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            speculative->device_pending_hidden, state->pending_hidden,
            sizeof(state->pending_hidden), cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) {
        rc = reset_controller_device_cursors(speculative, state->position);
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaDeviceSynchronize());
    if (rc != AXIOM_OK) {
        speculative->poisoned = true;
        return rc;
    }
    speculative->pending_index = 0u;
    speculative->device_pending_stream = nullptr;
    speculative->device_session_graph_exec = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_recycle_to_zero(
        axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    uint32_t target_position = 0u;
    uint32_t mtp_position = 0u;
    int rc = controller_idle_positions_locked(
            speculative, &target_position, &mtp_position);
    if (rc != AXIOM_OK || target_position != 0u) {
        return rc != AXIOM_OK ? rc : AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        speculative->poisoned = true;
        return AXIOM_ERR_CUDA;
    }
    rc = cuda_status(cudaDeviceSynchronize());
    if (rc == AXIOM_OK) rc = axiom_qwen38_mtp_compute_reset(speculative->mtp);
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemset(
            speculative->pending_hidden[0], 0,
            static_cast<size_t>(kHidden) * sizeof(float)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemset(
            speculative->pending_hidden[1], 0,
            static_cast<size_t>(kHidden) * sizeof(float)));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemset(
            speculative->device_pending_hidden, 0,
            static_cast<size_t>(kHidden) * sizeof(float)));
    if (rc == AXIOM_OK) rc = reset_controller_device_cursors(speculative, 0u);
    if (rc == AXIOM_OK) rc = cuda_status(cudaDeviceSynchronize());
    if (rc == AXIOM_OK &&
        (axiom_qwen38_model_position(speculative->target) != 0u ||
         axiom_qwen38_mtp_compute_position(speculative->mtp) != 0u)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc != AXIOM_OK) {
        speculative->poisoned = true;
        return rc;
    }
    speculative->pending_index = 0u;
    speculative->device_session_started = false;
    speculative->device_session_stream = nullptr;
    speculative->device_pending_stream = nullptr;
    speculative->device_session_start_position = 0u;
    speculative->device_session_graph_exec = nullptr;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_info_get(
        const axiom_qwen38_mtp_speculative *speculative,
        axiom_qwen38_mtp_speculative_info *out) {
    if (!speculative || !out ||
        out->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        out->struct_size != sizeof(*out)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(speculative->mutex);
    const uint32_t requested_abi = out->abi_version;
    const uint32_t requested_size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->struct_size = requested_size;
    out->position = axiom_qwen38_mtp_compute_position(speculative->mtp);
    out->poisoned = speculative->poisoned ? 1u : 0u;
    out->attempted_steps = speculative->attempted_steps;
    out->committed_steps = speculative->committed_steps;
    out->proposed_tokens = speculative->proposed_tokens;
    out->accepted_tokens = speculative->accepted_tokens;
    out->emitted_tokens = speculative->emitted_tokens;
    out->full_accept_steps = speculative->full_accept_steps;
    out->correction_steps = speculative->correction_steps;
    out->failed_steps = speculative->failed_steps;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_prefill_token(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t token_id,
        axiom_qwen38_mtp_speculative_prefill_result *out) {
    if (!speculative || !out ||
        out->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        out->struct_size != sizeof(*out) || token_id >= kVocab) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    const uint32_t requested_size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->struct_size = requested_size;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    /* A captured graph owns executable topology, not immutable KV/state.
     * Persistent sessions may restore a committed prefix and append a new
     * prompt suffix before the next graph launch.  Host prefill is therefore
     * legal while graph executables are resident, provided no device session
     * is active and the target/MTP watermarks still agree. */
    if (speculative->poisoned || speculative->device_session_started ||
        !state_positions_match(speculative) ||
        !target_capabilities_valid(speculative->target)) {
        return AXIOM_ERR_RUNTIME;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t snapshot = axiom_qwen38_model_position(speculative->target);
    axiom_qwen38_model_transaction *target_transaction = nullptr;
    bool mtp_open = false;
    int rc = axiom_qwen38_model_transaction_begin(
            speculative->target, &target_transaction);
    axiom_qwen38_model_dspark_prefill_token_result target_result{};
    target_result.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_prefill_token(
                target_transaction, token_id, nullptr, &target_result);
    }
    axiom_qwen38_model_dspark_capture_view capture{};
    capture.abi_version = AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_dspark_capture_view_get(
                target_transaction, &capture);
    }
    if (rc == AXIOM_OK) rc = copy_verified_hidden(speculative, capture, snapshot);
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                speculative->verify_tokens_device, &token_id, sizeof(token_id),
                cudaMemcpyHostToDevice, nullptr));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_mtp_compute_transaction_begin(
                speculative->mtp, snapshot, nullptr);
        mtp_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        rc = stage_mtp_forward(
                speculative, speculative->verify_tokens_device,
                current_pending(speculative), speculative->catchup_output,
                nullptr, 1u, snapshot);
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                candidate_pending(speculative), speculative->verified_hidden,
                static_cast<size_t>(kHidden) * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr));
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaStreamSynchronize(nullptr));
    if (rc != AXIOM_OK) {
        (void)abort_both(speculative, &target_transaction, &mtp_open);
        speculative->poisoned = speculative->poisoned || rc == AXIOM_ERR_CUDA;
        return rc;
    }
    rc = axiom_qwen38_mtp_compute_transaction_commit_prefix(
            speculative->mtp, 1u, nullptr);
    if (rc == AXIOM_OK) mtp_open = false;
    if (rc != AXIOM_OK) {
        (void)abort_both(speculative, &target_transaction, &mtp_open);
        return rc;
    }
    rc = axiom_qwen38_model_transaction_commit_prefix(target_transaction, 1u);
    target_transaction = nullptr;
    if (rc != AXIOM_OK) {
        (void)rollback_mtp_after_target_commit_failure(speculative, snapshot);
        return rc;
    }
    speculative->pending_index ^= 1u;
    out->token_position = snapshot;
    out->target_token_id = target_result.target_token_id;
    out->target_logit = target_result.target_logit;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_step(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_step_request *request,
        axiom_qwen38_mtp_speculative_step_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        request->struct_size != sizeof(*request) ||
        out->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        out->struct_size != sizeof(*out) || request->reserved != 0u ||
        request->anchor_token_id >= kVocab) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    const uint32_t requested_size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->struct_size = requested_size;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (speculative->poisoned || speculative->device_graph_ready ||
        speculative->device_session_started || !state_positions_match(speculative) ||
        !target_capabilities_valid(speculative->target)) {
        return AXIOM_ERR_RUNTIME;
    }
    add_saturated(&speculative->attempted_steps, 1u);
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        add_saturated(&speculative->failed_steps, 1u);
        return AXIOM_ERR_CUDA;
    }
    const uint32_t snapshot = axiom_qwen38_model_position(speculative->target);
    uint32_t verify_input[kWidth]{};
    verify_input[0] = request->anchor_token_id;
    int rc = cuda_status(cudaMemcpyAsync(
            speculative->verify_tokens_device, &request->anchor_token_id,
            sizeof(request->anchor_token_id), cudaMemcpyHostToDevice, nullptr));
    bool mtp_open = false;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_mtp_compute_transaction_begin(
                speculative->mtp, snapshot, nullptr);
        mtp_open = rc == AXIOM_OK;
    }
    for (uint32_t index = 0u; index < kDraft && rc == AXIOM_OK; ++index) {
        const uint32_t *token = index == 0u
                ? speculative->verify_tokens_device
                : speculative->proposal_device + index - 1u;
        const float *prior = index == 0u
                ? current_pending(speculative)
                : speculative->draft_hidden[(index - 1u) & 1u];
        rc = stage_mtp_forward(
                speculative, token, prior,
                speculative->draft_hidden[index & 1u],
                speculative->proposal_device + index,
                1u, snapshot + index);
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpy(
                verify_input + 1u, speculative->proposal_device,
                static_cast<size_t>(kDraft) * sizeof(uint32_t),
                cudaMemcpyDeviceToHost));
    }
    axiom_qwen38_model_transaction *target_transaction = nullptr;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_begin(
                speculative->target, &target_transaction);
    }
    axiom_qwen38_model_dspark_verify_block8_result verify{};
    verify.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_verify_block8(
                target_transaction, verify_input, nullptr, &verify);
    }
    axiom_qwen38_model_dspark_capture_view capture{};
    capture.abi_version = AXIOM_QWEN38_MODEL_DSPARK_CAPTURE_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_dspark_capture_view_get(
                target_transaction, &capture);
    }
    uint32_t accepted = 0u;
    if (rc == AXIOM_OK) {
        accepted = verify.accepted_draft_prefix;
        if (verify.snapshot_position != snapshot || accepted > kDraft ||
            verify.position_after_verify != snapshot + kWidth) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    for (uint32_t index = 0u; index < kDraft && rc == AXIOM_OK; ++index) {
        if ((index < accepted && verify.target_token_ids[index] != verify_input[index + 1u]) ||
            (index == accepted && accepted < kDraft &&
             verify.target_token_ids[index] == verify_input[index + 1u])) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) rc = copy_verified_hidden(speculative, capture, snapshot);
    if (rc != AXIOM_OK) {
        (void)abort_both(speculative, &target_transaction, &mtp_open);
        add_saturated(&speculative->failed_steps, 1u);
        speculative->poisoned = speculative->poisoned || rc == AXIOM_ERR_CUDA;
        return rc;
    }

    rc = abort_mtp(speculative->mtp, &mtp_open);
    const uint32_t committed_inputs = 1u + accepted;
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                speculative->verify_tokens_device, verify_input,
                static_cast<size_t>(committed_inputs) * sizeof(uint32_t),
                cudaMemcpyHostToDevice, nullptr));
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                speculative->catchup_prior, current_pending(speculative),
                static_cast<size_t>(kHidden) * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr));
    }
    if (rc == AXIOM_OK && committed_inputs > 1u) {
        rc = cuda_status(cudaMemcpyAsync(
                speculative->catchup_prior + kHidden,
                speculative->verified_hidden,
                static_cast<size_t>(committed_inputs - 1u) * kHidden * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_mtp_compute_transaction_begin(
                speculative->mtp, snapshot, nullptr);
        mtp_open = rc == AXIOM_OK;
    }
    if (rc == AXIOM_OK) {
        rc = stage_mtp_forward(
                speculative, speculative->verify_tokens_device,
                speculative->catchup_prior, speculative->catchup_output,
                nullptr, committed_inputs, snapshot);
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpyAsync(
                candidate_pending(speculative),
                speculative->verified_hidden +
                        static_cast<size_t>(committed_inputs - 1u) * kHidden,
                static_cast<size_t>(kHidden) * sizeof(float),
                cudaMemcpyDeviceToDevice, nullptr));
    }
    if (rc == AXIOM_OK) rc = cuda_status(cudaStreamSynchronize(nullptr));
    if (rc != AXIOM_OK) {
        (void)abort_both(speculative, &target_transaction, &mtp_open);
        add_saturated(&speculative->failed_steps, 1u);
        speculative->poisoned = speculative->poisoned || rc == AXIOM_ERR_CUDA;
        return rc;
    }
    rc = axiom_qwen38_mtp_compute_transaction_commit_prefix(
            speculative->mtp, committed_inputs, nullptr);
    if (rc == AXIOM_OK) mtp_open = false;
    if (rc != AXIOM_OK) {
        (void)abort_both(speculative, &target_transaction, &mtp_open);
        add_saturated(&speculative->failed_steps, 1u);
        return rc;
    }
    rc = axiom_qwen38_model_transaction_commit_prefix(
            target_transaction, committed_inputs);
    target_transaction = nullptr;
    if (rc != AXIOM_OK) {
        (void)rollback_mtp_after_target_commit_failure(speculative, snapshot);
        add_saturated(&speculative->failed_steps, 1u);
        return rc;
    }
    speculative->pending_index ^= 1u;

    out->snapshot_position = snapshot;
    out->position_after_commit = snapshot + committed_inputs;
    out->accepted_draft_prefix = accepted;
    out->emitted_token_count = committed_inputs;
    out->continuation_token_id = verify.target_token_ids[accepted];
    out->full_block_accept = accepted == kDraft ? 1u : 0u;
    std::memcpy(out->draft_token_ids, verify_input + 1u,
                static_cast<size_t>(kDraft) * sizeof(uint32_t));
    std::memcpy(out->target_token_ids, verify.target_token_ids,
                static_cast<size_t>(kWidth) * sizeof(uint32_t));
    std::memcpy(out->target_logits, verify.target_logits,
                static_cast<size_t>(kWidth) * sizeof(float));
    for (uint32_t index = 0u; index < accepted; ++index) {
        out->emitted_token_ids[index] = verify_input[index + 1u];
    }
    out->emitted_token_ids[accepted] = out->continuation_token_id;

    add_saturated(&speculative->committed_steps, 1u);
    add_saturated(&speculative->proposed_tokens, kDraft);
    add_saturated(&speculative->accepted_tokens, accepted);
    add_saturated(&speculative->emitted_tokens, committed_inputs);
    if (accepted == kDraft) add_saturated(&speculative->full_accept_steps, 1u);
    else add_saturated(&speculative->correction_steps, 1u);
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_device_prepare(
        axiom_qwen38_mtp_speculative *speculative) {
    if (!speculative) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (speculative->poisoned || !speculative->device_stop_tokens_configured ||
        speculative->device_session_started ||
        speculative->device_graph_ready || !state_positions_match(speculative) ||
        !target_capabilities_valid(speculative->target)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    return prepare_device_graph(speculative);
}

extern "C" int axiom_qwen38_mtp_speculative_device_stop_tokens_set(
        axiom_qwen38_mtp_speculative *speculative,
        const uint32_t *stop_token_ids,
        uint32_t stop_token_count) {
    if (!speculative || stop_token_count > kMaxStops ||
        (stop_token_count != 0u && !stop_token_ids)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t bounded_ids[kMaxStops]{};
    for (uint32_t index = 0u; index < stop_token_count; ++index) {
        if (stop_token_ids[index] >= kVocab) return AXIOM_ERR_INVALID_ARGUMENT;
        for (uint32_t prior = 0u; prior < index; ++prior) {
            if (stop_token_ids[prior] == stop_token_ids[index])
                return AXIOM_ERR_INVALID_ARGUMENT;
        }
        bounded_ids[index] = stop_token_ids[index];
    }

    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (speculative->poisoned || speculative->device_graph_ready ||
        speculative->device_session_started) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    int rc = cuda_status(cudaMemcpy(
            speculative->device_stop_token_ids, bounded_ids,
            sizeof(bounded_ids), cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpy(
                speculative->device_stop_token_count, &stop_token_count,
                sizeof(stop_token_count), cudaMemcpyHostToDevice));
    }
    if (rc == AXIOM_OK) speculative->device_stop_tokens_configured = true;
    return rc;
}

extern "C" int axiom_qwen38_mtp_speculative_device_step_enqueue(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_device_step_request *request,
        axiom_qwen38_mtp_speculative_device_step_result *out) {
    if (!speculative || !request || !out ||
        request->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION ||
        request->struct_size != sizeof(*request) ||
        out->abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION ||
        out->struct_size != sizeof(*out) || !request->anchor_token_device ||
        !request->anchor_position_device || !request->stream || request->flags != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    const uint32_t requested_size = out->struct_size;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->struct_size = requested_size;

    std::lock_guard<std::mutex> lock(speculative->mutex);
    add_saturated(&speculative->attempted_steps, 1u);
    if (speculative->poisoned || !speculative->device_graph_ready ||
        !speculative->device_graph_exec_exact ||
        (speculative->device_fast_graph_min_position > 0u &&
         !speculative->device_graph_exec_fast)) {
        add_saturated(&speculative->failed_steps, 1u);
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        add_saturated(&speculative->failed_steps, 1u);
        return AXIOM_ERR_CUDA;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(request->stream);
    int rc = AXIOM_OK;
    if (!speculative->device_session_started) {
        const uint32_t start_position = axiom_qwen38_model_position(speculative->target);
        cudaGraphExec_t selected_graph =
                speculative->device_fast_graph_min_position > 0u &&
                        start_position >= speculative->device_fast_graph_min_position
                ? speculative->device_graph_exec_fast
                : speculative->device_graph_exec_exact;
        if (speculative->device_session_graph_exec || !selected_graph) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (start_position != axiom_qwen38_mtp_compute_position(speculative->mtp)) {
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK && speculative->device_pending_stream &&
            request->stream != speculative->device_pending_stream) {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (rc == AXIOM_OK) {
            rc = cuda_status(cudaMemsetAsync(
                    speculative->device_async_status, 0u,
                    sizeof(uint32_t), stream));
        }
        if (rc == AXIOM_OK && request->anchor_token_device !=
                speculative->device_anchor_token) {
            rc = cuda_status(cudaMemcpyAsync(
                    speculative->device_anchor_token, request->anchor_token_device,
                    sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));
        }
        if (rc == AXIOM_OK) {
            *speculative->host_seed_position = start_position;
            rc = cuda_status(cudaMemcpyAsync(
                    speculative->device_anchor_position,
                    speculative->host_seed_position, sizeof(uint32_t),
                    cudaMemcpyHostToDevice, stream));
        }
        if (rc == AXIOM_OK) {
            rc = cuda_status(cudaMemcpyAsync(
                    speculative->device_pending_hidden, current_pending(speculative),
                    static_cast<size_t>(kHidden) * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream));
        }
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_mtp_compute_device_session_begin(
                    speculative->mtp, start_position);
        }
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_model_dspark_device_session_begin(speculative->target);
        }
        if (rc == AXIOM_OK) {
            speculative->device_session_started = true;
            speculative->device_session_stream = request->stream;
            speculative->device_pending_stream = nullptr;
            speculative->device_session_start_position = start_position;
            speculative->device_session_graph_exec = selected_graph;
        } else {
            (void)axiom_qwen38_mtp_compute_device_session_abort(speculative->mtp);
        }
    } else if (request->stream != speculative->device_session_stream ||
               request->anchor_token_device != speculative->device_anchor_token ||
               request->anchor_position_device != speculative->device_anchor_position ||
               !speculative->device_session_graph_exec) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaGraphLaunch(
                speculative->device_session_graph_exec, stream));
    }
    if (rc != AXIOM_OK) {
        if (speculative->device_session_started) {
            (void)cudaStreamSynchronize(stream);
            (void)axiom_qwen38_model_restore_position(
                    speculative->target,
                    speculative->device_session_start_position);
            (void)axiom_qwen38_mtp_compute_device_session_abort(speculative->mtp);
            speculative->device_session_started = false;
            speculative->device_session_stream = nullptr;
            speculative->device_pending_stream = nullptr;
            speculative->device_session_start_position = 0u;
            speculative->device_session_graph_exec = nullptr;
        }
        speculative->poisoned = true;
        add_saturated(&speculative->failed_steps, 1u);
        return rc;
    }

    out->graph_replayed = 1u;
    out->draft_tokens = kDraft;
    out->verify_width = kWidth;
    out->draft_token_ids_device = speculative->proposal_device;
    out->verify_token_ids_device = speculative->verify_tokens_device;
    out->accepted_prefix_device = speculative->device_accepted_prefix;
    out->target_commit_count_device = speculative->device_commit_count;
    out->emitted_token_count_device = speculative->device_emitted_token_count;
    out->continuation_token_device = speculative->device_continuation_token;
    out->continuation_logit_device = speculative->device_continuation_logit;
    out->emitted_token_ids_device = speculative->device_emitted_tokens;
    out->stop_detected_device = speculative->device_stop_detected;
    out->matched_stop_token_device = speculative->device_matched_stop_token;
    out->next_anchor_token_device = speculative->device_anchor_token;
    out->next_anchor_position_device = speculative->device_anchor_position;
    out->async_status_device = speculative->device_async_status;
    add_saturated(&speculative->committed_steps, 1u);
    add_saturated(&speculative->proposed_tokens, kDraft);
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_speculative_device_commit_limit_set(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t max_commit_tokens,
        void *stream) {
    if (!speculative || !stream || max_commit_tokens == 0u ||
        max_commit_tokens > kWidth) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (!speculative->device_graph_ready || speculative->poisoned ||
        (speculative->device_session_started &&
         stream != speculative->device_session_stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!speculative->device_session_started) {
        if (speculative->device_pending_stream &&
            stream != speculative->device_pending_stream) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        speculative->device_pending_stream = stream;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    return cuda_status(cudaMemcpyAsync(
            speculative->device_commit_limit, &max_commit_tokens,
            sizeof(max_commit_tokens), cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream)));
}

extern "C" int axiom_qwen38_mtp_speculative_device_session_end(
        axiom_qwen38_mtp_speculative *speculative,
        void *stream,
        const uint32_t *committed_token_ids,
        uint32_t committed_token_count) {
    if (!speculative || !stream ||
        (committed_token_count != 0u && !committed_token_ids)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(speculative->mutex);
    if (!speculative->device_session_started ||
        stream != speculative->device_session_stream ||
        committed_token_count < speculative->device_session_start_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(speculative->device) != cudaSuccess) {
        speculative->device_session_graph_exec = nullptr;
        speculative->poisoned = true;
        return AXIOM_ERR_CUDA;
    }
    uint32_t async_status = AXIOM_OK;
    uint32_t device_position = 0u;
    int transfer_rc = cuda_status(cudaMemcpyAsync(
            current_pending(speculative), speculative->device_pending_hidden,
            static_cast<size_t>(kHidden) * sizeof(float),
            cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)));
    if (transfer_rc == AXIOM_OK) {
        transfer_rc = cuda_status(cudaMemcpyAsync(
                &async_status, speculative->device_async_status,
                sizeof(async_status), cudaMemcpyDeviceToHost,
                static_cast<cudaStream_t>(stream)));
    }
    if (transfer_rc == AXIOM_OK) {
        transfer_rc = cuda_status(cudaMemcpyAsync(
                &device_position, speculative->device_anchor_position,
                sizeof(device_position), cudaMemcpyDeviceToHost,
                static_cast<cudaStream_t>(stream)));
    }
    if (transfer_rc == AXIOM_OK) {
        transfer_rc = cuda_status(cudaStreamSynchronize(
                static_cast<cudaStream_t>(stream)));
    }

    const bool valid_device_position = transfer_rc == AXIOM_OK &&
            device_position >= speculative->device_session_start_position;
    int target_rc = AXIOM_OK;
    int mtp_rc = AXIOM_OK;
    int history_rc = AXIOM_OK;
    if (valid_device_position) {
        target_rc = axiom_qwen38_model_restore_position(
                speculative->target, device_position);
        mtp_rc = axiom_qwen38_mtp_compute_device_session_end(
                speculative->mtp, device_position);
        if (device_position != committed_token_count) {
            history_rc = AXIOM_ERR_RUNTIME;
        } else if (target_rc == AXIOM_OK) {
            history_rc = axiom_qwen38_model_committed_history_install(
                    speculative->target, committed_token_ids,
                    committed_token_count);
        }
    } else {
        target_rc = axiom_qwen38_model_restore_position(
                speculative->target,
                speculative->device_session_start_position);
        mtp_rc = axiom_qwen38_mtp_compute_device_session_abort(speculative->mtp);
        history_rc = AXIOM_ERR_RUNTIME;
    }

    speculative->device_session_started = false;
    speculative->device_session_stream = nullptr;
    speculative->device_pending_stream = nullptr;
    speculative->device_session_start_position = 0u;
    speculative->device_session_graph_exec = nullptr;

    int rc = transfer_rc;
    if (rc == AXIOM_OK && target_rc != AXIOM_OK) rc = target_rc;
    if (rc == AXIOM_OK && mtp_rc != AXIOM_OK) rc = mtp_rc;
    if (rc == AXIOM_OK && history_rc != AXIOM_OK) rc = history_rc;
    if (rc == AXIOM_OK && async_status != static_cast<uint32_t>(AXIOM_OK))
        rc = static_cast<int>(async_status);
    if (rc != AXIOM_OK) speculative->poisoned = true;
    return rc;
}
