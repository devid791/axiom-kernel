#include "axiom/qwen38_mtp_device_control.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr uint32_t kDraftTokens = AXIOM_QWEN38_MTP_DEVICE_CONTROL_DRAFT_TOKENS;
constexpr uint32_t kVerifyWidth = AXIOM_QWEN38_MTP_DEVICE_CONTROL_VERIFY_WIDTH;
constexpr uint32_t kHidden = AXIOM_QWEN38_MTP_DEVICE_CONTROL_HIDDEN;
constexpr uint32_t kVocab = AXIOM_QWEN38_MTP_DEVICE_CONTROL_VOCAB;
constexpr uint32_t kInvalidToken = AXIOM_QWEN38_MTP_DEVICE_CONTROL_INVALID_TOKEN_ID;
constexpr uint32_t kMaxStopTokens = AXIOM_QWEN38_MTP_DEVICE_CONTROL_MAX_STOP_TOKENS;
constexpr uint32_t kThreads = 256u;

static_assert(kDraftTokens == 7u, "native MTP device control requires gamma-7");
static_assert(kVerifyWidth == 8u, "native MTP target verification requires M8");
static_assert(kDraftTokens + 1u == kVerifyWidth, "verify width must include one anchor");
static_assert(kHidden == 5120u, "Qwen3.8-27B hidden width changed");
static_assert(kVocab == 248320u, "Qwen3.8 vocabulary changed");
static_assert(kMaxStopTokens == 16u, "stop-token ABI bound changed");

template <typename Request>
bool valid_named_revision(
        const Request *request,
        const uint64_t request_bytes,
        const uint32_t abi_version) {
    return request && request_bytes == sizeof(Request) &&
           request->abi_version == abi_version &&
           request->struct_size == sizeof(Request) && request->flags == 0u;
}

template <typename Request>
bool valid_revision(const Request *request, const uint64_t request_bytes) {
    return valid_named_revision(
            request, request_bytes, AXIOM_QWEN38_MTP_DEVICE_CONTROL_ABI_VERSION);
}

template <std::size_t Count>
bool all_distinct(const void *const (&pointers)[Count]) {
    for (std::size_t left = 0u; left < Count; ++left) {
        if (!pointers[left]) return false;
        for (std::size_t right = left + 1u; right < Count; ++right) {
            if (pointers[left] == pointers[right]) return false;
        }
    }
    return true;
}

int launch_status() {
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

__device__ __forceinline__ void publish_status(
        uint32_t *__restrict__ async_status,
        const uint32_t error) {
    (void)atomicCAS(async_status, static_cast<uint32_t>(AXIOM_OK), error);
}

__global__ void build_verify_input_kernel(
        const uint32_t *__restrict__ anchor,
        const uint32_t *__restrict__ draft,
        uint32_t *__restrict__ verify,
        uint32_t *__restrict__ async_status) {
    const uint32_t index = threadIdx.x;
    if (index >= kVerifyWidth) return;

    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) {
        verify[index] = 0u;
        return;
    }
    const uint32_t token = index == 0u ? anchor[0] : draft[index - 1u];
    if (token >= kVocab) {
        verify[index] = 0u;
        publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
        return;
    }
    verify[index] = token;
}

__device__ __forceinline__ void write_failed_accept_outputs(
        const uint32_t anchor,
        const uint32_t position,
        uint32_t *__restrict__ accepted_prefix,
        uint32_t *__restrict__ target_commit_count,
        uint32_t *__restrict__ continuation_token,
        float *__restrict__ continuation_logit,
        uint32_t *__restrict__ emitted,
        uint32_t *__restrict__ next_anchor,
        uint32_t *__restrict__ next_position) {
    accepted_prefix[0] = 0u;
    target_commit_count[0] = 0u;
    continuation_token[0] = kInvalidToken;
    continuation_logit[0] = __int_as_float(0x7fffffff);
    for (uint32_t index = 0u; index < kVerifyWidth; ++index) emitted[index] = kInvalidToken;
    next_anchor[0] = anchor;
    next_position[0] = position;
}

__device__ __forceinline__ bool token_is_stop(
        const uint32_t token,
        const uint32_t *__restrict__ stop_token_ids,
        const uint32_t stop_token_count) {
    for (uint32_t index = 0u; index < stop_token_count; ++index) {
        if (token == stop_token_ids[index]) return true;
    }
    return false;
}

__device__ __forceinline__ void write_failed_accept_outputs_v2(
        const uint32_t anchor,
        const uint32_t position,
        uint32_t *__restrict__ accepted_prefix,
        uint32_t *__restrict__ target_commit_count,
        uint32_t *__restrict__ emitted_token_count,
        uint32_t *__restrict__ continuation_token,
        float *__restrict__ continuation_logit,
        uint32_t *__restrict__ emitted,
        uint32_t *__restrict__ stop_detected,
        uint32_t *__restrict__ matched_stop_token,
        uint32_t *__restrict__ next_anchor,
        uint32_t *__restrict__ next_position) {
    write_failed_accept_outputs(
            anchor, position, accepted_prefix, target_commit_count,
            continuation_token, continuation_logit, emitted, next_anchor, next_position);
    emitted_token_count[0] = 0u;
    stop_detected[0] = 0u;
    matched_stop_token[0] = kInvalidToken;
}

__global__ void accept_greedy_kernel(
        const uint32_t *__restrict__ anchor,
        const uint32_t *__restrict__ position,
        const uint32_t *__restrict__ draft,
        const uint32_t *__restrict__ target_top1,
        const float *__restrict__ target_top1_logits,
        const uint32_t *__restrict__ commit_limit,
        uint32_t *__restrict__ accepted_prefix,
        uint32_t *__restrict__ target_commit_count,
        uint32_t *__restrict__ continuation_token,
        float *__restrict__ continuation_logit,
        uint32_t *__restrict__ emitted,
        uint32_t *__restrict__ next_anchor,
        uint32_t *__restrict__ next_position,
        uint32_t *__restrict__ async_status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;

    const uint32_t current_anchor = anchor[0];
    const uint32_t current_position = position[0];
    write_failed_accept_outputs(
            current_anchor, current_position, accepted_prefix, target_commit_count,
            continuation_token, continuation_logit, emitted, next_anchor, next_position);

    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) return;

    const uint32_t limit = commit_limit[0];
    if (limit == 0u || limit > kVerifyWidth || current_anchor >= kVocab) {
        publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
        return;
    }
    for (uint32_t index = 0u; index < kDraftTokens; ++index) {
        if (draft[index] >= kVocab) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            return;
        }
    }
    for (uint32_t index = 0u; index < kVerifyWidth; ++index) {
        if (target_top1[index] >= kVocab) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            return;
        }
        if (!isfinite(target_top1_logits[index])) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_RUNTIME));
            return;
        }
    }

    uint32_t prefix = 0u;
    while (prefix < kDraftTokens && draft[prefix] == target_top1[prefix]) ++prefix;
    const uint32_t maximum_prefix = limit - 1u;
    if (prefix > maximum_prefix) prefix = maximum_prefix;
    const uint32_t committed = prefix + 1u;
    if (current_position > kInvalidToken - committed) {
        publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_BUDGET));
        return;
    }

    const uint32_t continuation = target_top1[prefix];
    accepted_prefix[0] = prefix;
    target_commit_count[0] = committed;
    continuation_token[0] = continuation;
    continuation_logit[0] = target_top1_logits[prefix];
    for (uint32_t index = 0u; index < prefix; ++index) emitted[index] = draft[index];
    emitted[prefix] = continuation;
    next_anchor[0] = continuation;
    next_position[0] = current_position + committed;
}

__global__ void accept_greedy_stop_aware_kernel(
        const uint32_t *__restrict__ anchor,
        const uint32_t *__restrict__ position,
        const uint32_t *__restrict__ draft,
        const uint32_t *__restrict__ target_top1,
        const float *__restrict__ target_top1_logits,
        const uint32_t *__restrict__ commit_limit,
        const uint32_t *__restrict__ stop_token_ids,
        const uint32_t *__restrict__ stop_token_count,
        uint32_t *__restrict__ accepted_prefix,
        uint32_t *__restrict__ target_commit_count,
        uint32_t *__restrict__ emitted_token_count,
        uint32_t *__restrict__ continuation_token,
        float *__restrict__ continuation_logit,
        uint32_t *__restrict__ emitted,
        uint32_t *__restrict__ stop_detected,
        uint32_t *__restrict__ matched_stop_token,
        uint32_t *__restrict__ next_anchor,
        uint32_t *__restrict__ next_position,
        uint32_t *__restrict__ async_status) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;

    const uint32_t current_anchor = anchor[0];
    const uint32_t current_position = position[0];
    write_failed_accept_outputs_v2(
            current_anchor, current_position, accepted_prefix, target_commit_count,
            emitted_token_count, continuation_token, continuation_logit, emitted,
            stop_detected, matched_stop_token, next_anchor, next_position);

    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) return;

    const uint32_t limit = commit_limit[0];
    const uint32_t active_stop_tokens = stop_token_count[0];
    if (limit == 0u || limit > kVerifyWidth || current_anchor >= kVocab ||
        active_stop_tokens > kMaxStopTokens) {
        publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
        return;
    }
    for (uint32_t index = 0u; index < active_stop_tokens; ++index) {
        if (stop_token_ids[index] >= kVocab) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            return;
        }
    }
    for (uint32_t index = 0u; index < kDraftTokens; ++index) {
        if (draft[index] >= kVocab) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            return;
        }
    }
    for (uint32_t index = 0u; index < kVerifyWidth; ++index) {
        if (target_top1[index] >= kVocab) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            return;
        }
        if (!isfinite(target_top1_logits[index])) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_RUNTIME));
            return;
        }
    }

    if (token_is_stop(current_anchor, stop_token_ids, active_stop_tokens)) {
        accepted_prefix[0] = 0u;
        target_commit_count[0] = 0u;
        emitted_token_count[0] = 0u;
        stop_detected[0] = 1u;
        matched_stop_token[0] = current_anchor;
        return;
    }

    uint32_t prefix = 0u;
    while (prefix < kDraftTokens && draft[prefix] == target_top1[prefix]) ++prefix;
    const uint32_t maximum_prefix = limit - 1u;
    if (prefix > maximum_prefix) prefix = maximum_prefix;

    for (uint32_t index = 0u; index < prefix; ++index) {
        if (!token_is_stop(draft[index], stop_token_ids, active_stop_tokens)) continue;

        const uint32_t terminal_prefix = index;
        const uint32_t committed = index + 1u;
        if (current_position > kInvalidToken - committed) {
            publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_BUDGET));
            return;
        }
        accepted_prefix[0] = terminal_prefix;
        target_commit_count[0] = committed;
        emitted_token_count[0] = committed;
        continuation_token[0] = draft[index];
        continuation_logit[0] = target_top1_logits[index];
        for (uint32_t emitted_index = 0u; emitted_index < terminal_prefix;
             ++emitted_index) {
            emitted[emitted_index] = draft[emitted_index];
        }
        emitted[terminal_prefix] = draft[index];
        stop_detected[0] = 1u;
        matched_stop_token[0] = draft[index];
        next_anchor[0] = draft[index];
        next_position[0] = current_position + committed;
        return;
    }

    const uint32_t committed = prefix + 1u;
    if (current_position > kInvalidToken - committed) {
        publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_BUDGET));
        return;
    }

    const uint32_t continuation = target_top1[prefix];
    accepted_prefix[0] = prefix;
    target_commit_count[0] = committed;
    emitted_token_count[0] = committed;
    continuation_token[0] = continuation;
    continuation_logit[0] = target_top1_logits[prefix];
    for (uint32_t index = 0u; index < prefix; ++index) emitted[index] = draft[index];
    emitted[prefix] = continuation;
    if (token_is_stop(continuation, stop_token_ids, active_stop_tokens)) {
        stop_detected[0] = 1u;
        matched_stop_token[0] = continuation;
    }
    next_anchor[0] = continuation;
    next_position[0] = current_position + committed;
}

__global__ void build_catchup_prior_kernel(
        const float *__restrict__ pending,
        const float *__restrict__ target_hidden,
        float *__restrict__ catchup_prior,
        const uint32_t *__restrict__ async_status) {
    const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t kElements = kHidden * kVerifyWidth;
    if (element >= kElements) return;

    if (async_status[0] != static_cast<uint32_t>(AXIOM_OK)) {
        catchup_prior[element] = 0.0f;
        return;
    }
    const uint32_t column = element / kHidden;
    const uint32_t hidden = element - column * kHidden;
    catchup_prior[element] = column == 0u
            ? pending[hidden]
            : target_hidden[(column - 1u) * kHidden + hidden];
}

__global__ void select_next_pending_kernel(
        const float *__restrict__ target_hidden,
        const uint32_t *__restrict__ target_commit_count,
        float *__restrict__ next_pending,
        uint32_t *__restrict__ async_status) {
    __shared__ uint32_t selected_row;
    if (threadIdx.x == 0u) {
        selected_row = kInvalidToken;
        if (async_status[0] == static_cast<uint32_t>(AXIOM_OK)) {
            const uint32_t committed = target_commit_count[0];
            if (committed == 0u || committed > kVerifyWidth) {
                publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            } else {
                selected_row = committed - 1u;
            }
        }
    }
    __syncthreads();
    if (selected_row == kInvalidToken) return;

    const uint64_t input_base = static_cast<uint64_t>(selected_row) * kHidden;
    for (uint32_t hidden = threadIdx.x; hidden < kHidden; hidden += blockDim.x) {
        next_pending[hidden] = target_hidden[input_base + hidden];
    }
}

__global__ void select_next_pending_commit_aware_kernel(
        const float *__restrict__ target_hidden,
        const uint32_t *__restrict__ target_commit_count,
        float *__restrict__ next_pending,
        uint32_t *__restrict__ async_status) {
    __shared__ uint32_t selected_row;
    if (threadIdx.x == 0u) {
        selected_row = kInvalidToken;
        if (async_status[0] == static_cast<uint32_t>(AXIOM_OK)) {
            const uint32_t committed = target_commit_count[0];
            if (committed > kVerifyWidth) {
                publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            } else if (committed != 0u) {
                selected_row = committed - 1u;
            }
        }
    }
    __syncthreads();
    if (selected_row == kInvalidToken) return;

    const uint64_t input_base = static_cast<uint64_t>(selected_row) * kHidden;
    for (uint32_t hidden = threadIdx.x; hidden < kHidden; hidden += blockDim.x) {
        next_pending[hidden] = target_hidden[input_base + hidden];
    }
}

__global__ void publish_state_kernel(
        const uint32_t *__restrict__ candidate_anchor,
        const uint32_t *__restrict__ candidate_position,
        const float *__restrict__ candidate_pending,
        const uint32_t *__restrict__ async_status,
        uint32_t *__restrict__ authoritative_anchor,
        uint32_t *__restrict__ authoritative_position,
        float *__restrict__ authoritative_pending) {
    __shared__ uint32_t publish;
    if (threadIdx.x == 0u) {
        publish = async_status[0] == static_cast<uint32_t>(AXIOM_OK) ? 1u : 0u;
    }
    __syncthreads();
    if (publish == 0u) return;

    if (threadIdx.x == 0u) {
        authoritative_anchor[0] = candidate_anchor[0];
        authoritative_position[0] = candidate_position[0];
    }
    for (uint32_t hidden = threadIdx.x; hidden < kHidden; hidden += blockDim.x) {
        authoritative_pending[hidden] = candidate_pending[hidden];
    }
}

__global__ void publish_state_commit_aware_kernel(
        const uint32_t *__restrict__ candidate_anchor,
        const uint32_t *__restrict__ candidate_position,
        const float *__restrict__ candidate_pending,
        const uint32_t *__restrict__ target_commit_count,
        uint32_t *__restrict__ async_status,
        uint32_t *__restrict__ authoritative_anchor,
        uint32_t *__restrict__ authoritative_position,
        float *__restrict__ authoritative_pending) {
    __shared__ uint32_t publish;
    if (threadIdx.x == 0u) {
        publish = 0u;
        if (async_status[0] == static_cast<uint32_t>(AXIOM_OK)) {
            const uint32_t committed = target_commit_count[0];
            if (committed > kVerifyWidth) {
                publish_status(async_status, static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT));
            } else if (committed != 0u) {
                publish = 1u;
            }
        }
    }
    __syncthreads();
    if (publish == 0u) return;

    if (threadIdx.x == 0u) {
        authoritative_anchor[0] = candidate_anchor[0];
        authoritative_position[0] = candidate_position[0];
    }
    for (uint32_t hidden = threadIdx.x; hidden < kHidden; hidden += blockDim.x) {
        authoritative_pending[hidden] = candidate_pending[hidden];
    }
}

}  // namespace

extern "C" int axiom_qwen38_mtp_device_build_verify_input_enqueue_v1(
        const axiom_qwen38_mtp_device_build_verify_request_v1 *request,
        const uint64_t request_bytes) {
    if (!valid_revision(request, request_bytes) || !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->anchor_token_device,
        request->draft_token_ids_device,
        request->verify_token_ids_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    build_verify_input_kernel<<<1u, kVerifyWidth, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->anchor_token_device, request->draft_token_ids_device,
            request->verify_token_ids_device, request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_accept_greedy_enqueue_v1(
        const axiom_qwen38_mtp_device_accept_request_v1 *request,
        const uint64_t request_bytes) {
    if (!valid_revision(request, request_bytes) || !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->anchor_token_device,
        request->anchor_position_device,
        request->draft_token_ids_device,
        request->target_top1_device,
        request->target_top1_logits_device,
        request->commit_limit_device,
        request->accepted_prefix_device,
        request->target_commit_count_device,
        request->continuation_token_device,
        request->continuation_logit_device,
        request->emitted_token_ids_device,
        request->next_anchor_token_device,
        request->next_anchor_position_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    accept_greedy_kernel<<<1u, 1u, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->anchor_token_device, request->anchor_position_device,
            request->draft_token_ids_device, request->target_top1_device,
            request->target_top1_logits_device, request->commit_limit_device,
            request->accepted_prefix_device, request->target_commit_count_device,
            request->continuation_token_device, request->continuation_logit_device,
            request->emitted_token_ids_device, request->next_anchor_token_device,
            request->next_anchor_position_device, request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_accept_greedy_enqueue_v2(
        const axiom_qwen38_mtp_device_accept_request_v2 *request,
        const uint64_t request_bytes) {
    if (!valid_named_revision(
                request, request_bytes, AXIOM_QWEN38_MTP_DEVICE_ACCEPT_V2_ABI_VERSION) ||
        !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->anchor_token_device,
        request->anchor_position_device,
        request->draft_token_ids_device,
        request->target_top1_device,
        request->target_top1_logits_device,
        request->commit_limit_device,
        request->stop_token_ids_device,
        request->stop_token_count_device,
        request->accepted_prefix_device,
        request->target_commit_count_device,
        request->emitted_token_count_device,
        request->continuation_token_device,
        request->continuation_logit_device,
        request->emitted_token_ids_device,
        request->stop_detected_device,
        request->matched_stop_token_device,
        request->next_anchor_token_device,
        request->next_anchor_position_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    accept_greedy_stop_aware_kernel<<<
            1u, 1u, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->anchor_token_device, request->anchor_position_device,
            request->draft_token_ids_device, request->target_top1_device,
            request->target_top1_logits_device, request->commit_limit_device,
            request->stop_token_ids_device, request->stop_token_count_device,
            request->accepted_prefix_device, request->target_commit_count_device,
            request->emitted_token_count_device, request->continuation_token_device,
            request->continuation_logit_device, request->emitted_token_ids_device,
            request->stop_detected_device, request->matched_stop_token_device,
            request->next_anchor_token_device, request->next_anchor_position_device,
            request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_build_catchup_prior_enqueue_v1(
        const axiom_qwen38_mtp_device_build_catchup_request_v1 *request,
        const uint64_t request_bytes) {
    if (!valid_revision(request, request_bytes) || !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->pending_hidden_device,
        request->target_final_hidden_device,
        request->catchup_prior_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    constexpr uint32_t elements = kHidden * kVerifyWidth;
    constexpr uint32_t blocks = (elements + kThreads - 1u) / kThreads;
    build_catchup_prior_kernel<<<blocks, kThreads, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->pending_hidden_device, request->target_final_hidden_device,
            request->catchup_prior_device, request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_select_next_pending_enqueue_v1(
        const axiom_qwen38_mtp_device_select_pending_request_v1 *request,
        const uint64_t request_bytes) {
    if (!valid_revision(request, request_bytes) || !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->target_final_hidden_device,
        request->target_commit_count_device,
        request->next_pending_hidden_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    select_next_pending_kernel<<<1u, kThreads, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->target_final_hidden_device, request->target_commit_count_device,
            request->next_pending_hidden_device, request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_select_next_pending_enqueue_v2(
        const axiom_qwen38_mtp_device_select_pending_request_v2 *request,
        const uint64_t request_bytes) {
    if (!valid_named_revision(
                request, request_bytes, AXIOM_QWEN38_MTP_DEVICE_SELECT_V2_ABI_VERSION) ||
        !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->target_final_hidden_device,
        request->target_commit_count_device,
        request->next_pending_hidden_device,
        request->async_status_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    select_next_pending_commit_aware_kernel<<<
            1u, kThreads, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->target_final_hidden_device, request->target_commit_count_device,
            request->next_pending_hidden_device, request->async_status_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_publish_state_enqueue_v1(
        const axiom_qwen38_mtp_device_publish_state_request_v1 *request,
        const uint64_t request_bytes) {
    if (!valid_named_revision(
                request, request_bytes, AXIOM_QWEN38_MTP_DEVICE_PUBLISH_V1_ABI_VERSION) ||
        !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->candidate_anchor_token_device,
        request->candidate_anchor_position_device,
        request->candidate_pending_hidden_device,
        request->async_status_device,
        request->authoritative_anchor_token_device,
        request->authoritative_anchor_position_device,
        request->authoritative_pending_hidden_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    publish_state_kernel<<<1u, kThreads, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->candidate_anchor_token_device,
            request->candidate_anchor_position_device,
            request->candidate_pending_hidden_device,
            request->async_status_device,
            request->authoritative_anchor_token_device,
            request->authoritative_anchor_position_device,
            request->authoritative_pending_hidden_device);
    return launch_status();
}

extern "C" int axiom_qwen38_mtp_device_publish_state_enqueue_v2(
        const axiom_qwen38_mtp_device_publish_state_request_v2 *request,
        const uint64_t request_bytes) {
    if (!valid_named_revision(
                request, request_bytes, AXIOM_QWEN38_MTP_DEVICE_PUBLISH_V2_ABI_VERSION) ||
        !request->stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const void *const pointers[] = {
        request->candidate_anchor_token_device,
        request->candidate_anchor_position_device,
        request->candidate_pending_hidden_device,
        request->target_commit_count_device,
        request->async_status_device,
        request->authoritative_anchor_token_device,
        request->authoritative_anchor_position_device,
        request->authoritative_pending_hidden_device,
    };
    if (!all_distinct(pointers)) return AXIOM_ERR_INVALID_ARGUMENT;

    publish_state_commit_aware_kernel<<<
            1u, kThreads, 0, static_cast<cudaStream_t>(request->stream)>>>(
            request->candidate_anchor_token_device,
            request->candidate_anchor_position_device,
            request->candidate_pending_hidden_device,
            request->target_commit_count_device,
            request->async_status_device,
            request->authoritative_anchor_token_device,
            request->authoritative_anchor_position_device,
            request->authoritative_pending_hidden_device);
    return launch_status();
}
