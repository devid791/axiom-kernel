#ifndef AXIOM_QWEN38_MTP_DEVICE_CONTROL_H
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_H

/*
 * Stateless CUDA device-control helpers for native Qwen3.8 MTP decoding.
 *
 * Every pointer field addresses persistent caller-owned CUDA memory.  The
 * enqueue functions retain no host request pointer, allocate no storage,
 * synchronize never, and perform no device-to-host transfer.  A request may
 * therefore be issued during CUDA graph capture.  Device pointer values used
 * while capturing must remain valid until the resulting graph executable is
 * destroyed; outside capture they must remain valid until stream completion.
 *
 * The caller must select the CUDA device that owns the explicit stream and
 * buffers before enqueueing.  Pointer ranges in one request may not overlap.
 * The ABI validates host-visible structure metadata and non-null pointer
 * fields, but intentionally performs no cudaPointerGetAttributes query in the
 * decode path.
 */

#include <stdint.h>

#include "axiom/axiom.h"
#include "axiom/qwen38_mtp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_DRAFT_TOKENS 7u
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_VERIFY_WIDTH AXIOM_QWEN38_MTP_BATCH
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_HIDDEN AXIOM_QWEN38_MTP_HIDDEN
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_VOCAB AXIOM_QWEN38_MTP_VOCAB
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_INVALID_TOKEN_ID 0xffffffffu
#define AXIOM_QWEN38_MTP_DEVICE_ACCEPT_V2_ABI_VERSION 2u
#define AXIOM_QWEN38_MTP_DEVICE_SELECT_V2_ABI_VERSION 2u
#define AXIOM_QWEN38_MTP_DEVICE_PUBLISH_V1_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_DEVICE_PUBLISH_V2_ABI_VERSION 2u
#define AXIOM_QWEN38_MTP_DEVICE_CONTROL_MAX_STOP_TOKENS 16u

/* Build the target temporal-M8 input as:
 *
 *   verify_token_ids_device = [anchor, draft_0, ..., draft_6]
 *
 * Invalid input token IDs are replaced with token zero before target
 * execution and atomically publish AXIOM_ERR_INVALID_ARGUMENT to
 * async_status_device.  An already-failed status produces an all-zero safe
 * verify input.  The acceptance operation independently validates every
 * token again before allowing a commit.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *anchor_token_device;      /* CUDA U32[1]. */
    const uint32_t *draft_token_ids_device;   /* CUDA U32[7]. */
    uint32_t *verify_token_ids_device;        /* CUDA U32[8]. */
    uint32_t *async_status_device;            /* CUDA U32[1], initially AXIOM_OK. */
    void *stream;                             /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                           /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_build_verify_request_v1;

/* Exact target-authoritative greedy acceptance.
 *
 * Let K be the longest prefix in [0,7] for which
 * draft_token_ids_device[i] == target_top1_device[i].  The device commit
 * limit L is required to be in [1,8], includes the anchor row, and clamps K
 * to min(K, L - 1).  On success the outputs are:
 *
 *   accepted_prefix       = K
 *   target_commit_count   = K + 1                 (range [1,8])
 *   continuation_token    = target_top1[K]
 *   continuation_logit    = target_top1_logits[K]
 *   emitted[0..K-1]       = draft[0..K-1]
 *   emitted[K]            = continuation_token
 *   emitted[K+1..7]       = INVALID_TOKEN_ID
 *   next_anchor_token     = continuation_token
 *   next_anchor_position  = anchor_position + K + 1
 *
 * All eight target token IDs, all seven draft IDs, and the anchor ID must be
 * in the Qwen vocabulary.  All eight target top-1 logits must be finite.
 * Invalid tokens/limits publish AXIOM_ERR_INVALID_ARGUMENT, non-finite target
 * logits publish AXIOM_ERR_RUNTIME, and position overflow publishes
 * AXIOM_ERR_BUDGET.  Errors are fail-closed: target_commit_count becomes zero
 * and no next-session state is advanced.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *anchor_token_device;          /* CUDA U32[1]. */
    const uint32_t *anchor_position_device;       /* CUDA U32[1]. */
    const uint32_t *draft_token_ids_device;       /* CUDA U32[7]. */
    const uint32_t *target_top1_device;            /* CUDA U32[8]. */
    const float *target_top1_logits_device;        /* CUDA F32[8]. */
    const uint32_t *commit_limit_device;           /* CUDA U32[1], in [1,8]. */
    uint32_t *accepted_prefix_device;              /* CUDA U32[1]. */
    uint32_t *target_commit_count_device;          /* CUDA U32[1], success [1,8]. */
    uint32_t *continuation_token_device;           /* CUDA U32[1]. */
    float *continuation_logit_device;              /* CUDA F32[1]. */
    uint32_t *emitted_token_ids_device;            /* CUDA U32[8]. */
    uint32_t *next_anchor_token_device;            /* CUDA U32[1]. */
    uint32_t *next_anchor_position_device;         /* CUDA U32[1]. */
    uint32_t *async_status_device;                 /* CUDA U32[1], initially AXIOM_OK. */
    void *stream;                                  /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                                /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_accept_request_v1;

/* Stop-aware target-authoritative greedy acceptance.
 *
 * This revision preserves the v1 symbol and semantics and adds a bounded,
 * device-resident stop-token set.  stop_token_count_device is in [0,16]; a
 * zero count disables stop matching without changing the captured graph.
 * The stop-token array remains a fixed U32[16] allocation even when the
 * active count is zero, so every graph-captured pointer stays stable.
 *
 * Stop tokens are output boundaries and are never committed as accepted
 * draft inputs.  The rules match the production DSpark contract exactly:
 *
 *   stop at anchor       -> accepted=0, commit=0, emitted=0; terminal no-op
 *   stop at draft i      -> accepted=i, commit=i+1, emitted=i+1
 *
 * In the second case draft i is exposed as the authoritative continuation
 * target_top1[i], so output includes the stop but never a token after it.
 * A stop reached as the ordinary target continuation uses the unchanged
 * N=K+1 contract.  The explicit emitted count distinguishes an anchor no-op
 * from ordinary correction.
 *
 * All active stop IDs, target/draft/anchor IDs, the device stop count and all
 * target logits are validated before any state advances.  Invalid values,
 * non-finite logits, a pre-failed asynchronous status or position overflow
 * fail closed with target_commit_count and emitted_token_count both zero.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *anchor_token_device;          /* CUDA U32[1]. */
    const uint32_t *anchor_position_device;       /* CUDA U32[1]. */
    const uint32_t *draft_token_ids_device;       /* CUDA U32[7]. */
    const uint32_t *target_top1_device;            /* CUDA U32[8]. */
    const float *target_top1_logits_device;        /* CUDA F32[8]. */
    const uint32_t *commit_limit_device;           /* CUDA U32[1], in [1,8]. */
    const uint32_t *stop_token_ids_device;         /* CUDA U32[16]. */
    const uint32_t *stop_token_count_device;       /* CUDA U32[1], in [0,16]. */
    uint32_t *accepted_prefix_device;              /* CUDA U32[1]. */
    uint32_t *target_commit_count_device;          /* CUDA U32[1], success [0,8]. */
    uint32_t *emitted_token_count_device;          /* CUDA U32[1], success [0,8]. */
    uint32_t *continuation_token_device;           /* CUDA U32[1]. */
    float *continuation_logit_device;              /* CUDA F32[1]. */
    uint32_t *emitted_token_ids_device;            /* CUDA U32[8]. */
    uint32_t *stop_detected_device;                /* CUDA U32[1], boolean. */
    uint32_t *matched_stop_token_device;           /* CUDA U32[1] or INVALID. */
    uint32_t *next_anchor_token_device;            /* CUDA U32[1]. */
    uint32_t *next_anchor_position_device;         /* CUDA U32[1]. */
    uint32_t *async_status_device;                 /* CUDA U32[1], initially AXIOM_OK. */
    void *stream;                                  /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                                /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_accept_request_v2;

/* Build the shifted target catch-up prior expected by the MTP executor.
 * Target hidden storage is temporal-major F32 [8][5120].  MTP prior storage
 * is its column-major F32 [5120,8] contract, physically one contiguous hidden
 * row per temporal column.  This operation writes exactly:
 *
 *   prior[:,0] = pending[:]
 *   prior[:,1] = target_final_hidden[0,:]
 *   ...
 *   prior[:,7] = target_final_hidden[6,:]
 *
 * Target row H7 is intentionally not part of this shifted prior.  If the
 * asynchronous status is already failed, the output is zero-filled so a
 * subsequently captured but logically aborted MTP forward remains numerically
 * safe.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const float *pending_hidden_device;       /* CUDA F32[5120]. */
    const float *target_final_hidden_device;  /* CUDA F32[8][5120]. */
    float *catchup_prior_device;              /* CUDA F32[5120,8], column-major. */
    const uint32_t *async_status_device;      /* CUDA U32[1]. */
    void *stream;                             /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                           /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_build_catchup_request_v1;

/* Select the next persistent predecessor state without a host-side branch:
 *
 *   next_pending[:] = target_final_hidden[target_commit_count - 1, :]
 *
 * target_commit_count must be in [1,8].  A failed incoming status leaves the
 * existing next_pending buffer untouched.  An invalid count atomically sets
 * AXIOM_ERR_INVALID_ARGUMENT and also leaves it untouched.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const float *target_final_hidden_device;  /* CUDA F32[8][5120]. */
    const uint32_t *target_commit_count_device; /* CUDA U32[1], in [1,8]. */
    float *next_pending_hidden_device;        /* CUDA F32[5120]. */
    uint32_t *async_status_device;            /* CUDA U32[1]. */
    void *stream;                             /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                           /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_select_pending_request_v1;

/* Commit-aware selector for graph paths that may terminate at the current
 * anchor.  Counts in [1,8] select row count-1 exactly as v1; count zero is a
 * successful no-op that preserves next_pending in full.  Counts above eight
 * publish AXIOM_ERR_INVALID_ARGUMENT and also preserve next_pending. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const float *target_final_hidden_device;  /* CUDA F32[8][5120]. */
    const uint32_t *target_commit_count_device; /* CUDA U32[1], in [0,8]. */
    float *next_pending_hidden_device;        /* CUDA F32[5120]. */
    uint32_t *async_status_device;            /* CUDA U32[1]. */
    void *stream;                             /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                           /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_select_pending_request_v2;

/* Transactionally publish one candidate decode state on the supplied CUDA
 * stream.  A single kernel snapshots async_status once.  If it is AXIOM_OK,
 * the candidate anchor, position and all 5120 pending-hidden elements become
 * authoritative before any later operation on the stream can observe them.
 * For every other status the kernel performs no authoritative write, thereby
 * preserving all three old values.  The operation allocates nothing,
 * synchronizes never and has fixed CUDA-graph topology.
 *
 * Candidate and authoritative ranges must be separate persistent allocations
 * and may not overlap.  Transactionality is stream ordered: callers must use
 * the same stream or an explicit CUDA dependency for readers/writers.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *candidate_anchor_token_device;    /* CUDA U32[1]. */
    const uint32_t *candidate_anchor_position_device; /* CUDA U32[1]. */
    const float *candidate_pending_hidden_device;     /* CUDA F32[5120]. */
    const uint32_t *async_status_device;              /* CUDA U32[1]. */
    uint32_t *authoritative_anchor_token_device;      /* CUDA U32[1]. */
    uint32_t *authoritative_anchor_position_device;   /* CUDA U32[1]. */
    float *authoritative_pending_hidden_device;       /* CUDA F32[5120]. */
    void *stream;                                     /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                                   /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_publish_state_request_v1;

/* Commit-aware transactional publish.  This revision adds a device commit
 * count to v1: [1,8] publishes all three candidate fields, zero preserves all
 * authoritative fields without error, and values above eight fail closed by
 * publishing AXIOM_ERR_INVALID_ARGUMENT while preserving all state. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *candidate_anchor_token_device;    /* CUDA U32[1]. */
    const uint32_t *candidate_anchor_position_device; /* CUDA U32[1]. */
    const float *candidate_pending_hidden_device;     /* CUDA F32[5120]. */
    const uint32_t *target_commit_count_device;       /* CUDA U32[1], in [0,8]. */
    uint32_t *async_status_device;                    /* CUDA U32[1]. */
    uint32_t *authoritative_anchor_token_device;      /* CUDA U32[1]. */
    uint32_t *authoritative_anchor_position_device;   /* CUDA U32[1]. */
    float *authoritative_pending_hidden_device;       /* CUDA F32[5120]. */
    void *stream;                                     /* Explicit cudaStream_t; non-NULL. */
    uint64_t flags;                                   /* Reserved; must be zero. */
} axiom_qwen38_mtp_device_publish_state_request_v2;

/* request_bytes and request->struct_size must both exactly match the named
 * revision.  No function accepts an older/smaller structure implicitly. */
int axiom_qwen38_mtp_device_build_verify_input_enqueue_v1(
        const axiom_qwen38_mtp_device_build_verify_request_v1 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_accept_greedy_enqueue_v1(
        const axiom_qwen38_mtp_device_accept_request_v1 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_accept_greedy_enqueue_v2(
        const axiom_qwen38_mtp_device_accept_request_v2 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_build_catchup_prior_enqueue_v1(
        const axiom_qwen38_mtp_device_build_catchup_request_v1 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_select_next_pending_enqueue_v1(
        const axiom_qwen38_mtp_device_select_pending_request_v1 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_select_next_pending_enqueue_v2(
        const axiom_qwen38_mtp_device_select_pending_request_v2 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_publish_state_enqueue_v1(
        const axiom_qwen38_mtp_device_publish_state_request_v1 *request,
        uint64_t request_bytes);

int axiom_qwen38_mtp_device_publish_state_enqueue_v2(
        const axiom_qwen38_mtp_device_publish_state_request_v2 *request,
        uint64_t request_bytes);

#ifdef __cplusplus
}
#endif

#endif
