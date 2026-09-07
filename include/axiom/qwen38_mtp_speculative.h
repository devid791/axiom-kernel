#ifndef AXIOM_QWEN38_MTP_SPECULATIVE_H
#define AXIOM_QWEN38_MTP_SPECULATIVE_H

/*
 * Lossless Qwen3.8 native-MTP speculative controller.
 *
 * This controller is the correctness/reference path used to qualify the
 * device-resident production executor.  It joins one target M8 transaction
 * with one native MTP transaction, accepts only an exact greedy prefix, then
 * rewrites the authoritative MTP suffix from the target's BF16-materialized
 * final hidden states before either side is committed.
 *
 * No target-only or scalar fallback exists in this contract.  A failure
 * aborts both open transactions.  A failure after either logical commit
 * poisons the controller and must be followed by reset or destruction.
 */

#include <stdint.h>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp_compute.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS 7u
#define AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH 8u
#define AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION 2u
#define AXIOM_QWEN38_MTP_SPECULATIVE_MAX_STOP_TOKENS 16u
#define AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_STATE_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_HIDDEN 5120u

typedef struct axiom_qwen38_mtp_speculative axiom_qwen38_mtp_speculative;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t device;
    uint32_t reserved;
    uint64_t flags;
} axiom_qwen38_mtp_speculative_config;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t position;
    uint32_t poisoned;
    uint64_t attempted_steps;
    uint64_t committed_steps;
    uint64_t proposed_tokens;
    uint64_t accepted_tokens;
    uint64_t emitted_tokens;
    uint64_t full_accept_steps;
    uint64_t correction_steps;
    uint64_t failed_steps;
} axiom_qwen38_mtp_speculative_info;

/* Caller-owned, fixed-size host snapshot of the native-MTP predecessor
 * frontier. The header and F32[5120] payload are part of the ABI; flags must
 * be zero and hidden_elements must equal 5120. No target or MTP K/V bytes are
 * embedded here. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t position;
    uint32_t hidden_elements;
    uint64_t flags;
    float pending_hidden[AXIOM_QWEN38_MTP_SPECULATIVE_PERSISTENT_HIDDEN];
} axiom_qwen38_mtp_speculative_persistent_state_v1;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t token_position;
    uint32_t target_token_id;
    float target_logit;
    uint32_t reserved;
} axiom_qwen38_mtp_speculative_prefill_result;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t anchor_token_id;
    uint32_t reserved;
} axiom_qwen38_mtp_speculative_step_request;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t snapshot_position;
    uint32_t position_after_commit;
    uint32_t accepted_draft_prefix;
    uint32_t emitted_token_count;
    uint32_t continuation_token_id;
    uint32_t full_block_accept;
    uint32_t draft_token_ids[AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS];
    uint32_t target_token_ids[AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH];
    float target_logits[AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH];
    uint32_t emitted_token_ids[AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH];
} axiom_qwen38_mtp_speculative_step_result;

/* Device-only CUDA-graph decode. Both request scalars are persistent CUDA
 * U32[1]. The first enqueue copies the anchor token but derives the cache
 * position from the target/MTP host watermark; the caller-supplied position
 * is never authoritative. Later enqueues must pass the returned next-anchor
 * pointers and the same explicit stream. Result pointers are transient: a
 * caller batching replays must enqueue any result copy on that stream before
 * submitting the next replay. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *anchor_token_device;
    const uint32_t *anchor_position_device;
    void *stream;
    uint64_t flags;
} axiom_qwen38_mtp_speculative_device_step_request;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t graph_replayed;
    uint32_t draft_tokens;
    uint32_t verify_width;
    uint32_t reserved;
    const uint32_t *draft_token_ids_device;
    const uint32_t *verify_token_ids_device;
    const uint32_t *accepted_prefix_device;
    const uint32_t *target_commit_count_device;
    const uint32_t *emitted_token_count_device;
    const uint32_t *continuation_token_device;
    const float *continuation_logit_device;
    const uint32_t *emitted_token_ids_device;
    const uint32_t *stop_detected_device;
    const uint32_t *matched_stop_token_device;
    const uint32_t *next_anchor_token_device;
    const uint32_t *next_anchor_position_device;
    const uint32_t *async_status_device;
} axiom_qwen38_mtp_speculative_device_step_result;

/* The controller borrows target and MTP compute objects.  They must outlive
 * it and must not be used concurrently through another session. */
int axiom_qwen38_mtp_speculative_create_v1(
        axiom_qwen38_model *target,
        axiom_qwen38_mtp_compute *mtp_compute,
        const axiom_qwen38_mtp_speculative_config *config,
        uint64_t config_bytes,
        axiom_qwen38_mtp_speculative **out);

void axiom_qwen38_mtp_speculative_destroy(
        axiom_qwen38_mtp_speculative *speculative);

/* Cold control-plane reset.  It synchronizes the default stream, resets both
 * borrowed state machines to position zero, and reinstalls the all-zero
 * initial pending hidden row required by the Qwen MTP shift contract. */
int axiom_qwen38_mtp_speculative_reset(
        axiom_qwen38_mtp_speculative *speculative);

/* Synchronous persistent-frontier handoff. The controller retains ownership
 * of all CUDA buffers; the caller owns the size-checked host structure.
 * Export/import are legal only while the controller is non-poisoned and
 * idle, with no target/MTP transaction or device session and with equal
 * target/MTP positions. Import additionally requires snapshot.position to
 * equal both already-restored positions, copies the pending row into both
 * controller buffers and the device-authoritative pending buffer, normalizes
 * pending_index, and synchronizes before returning. */
int axiom_qwen38_mtp_speculative_persistent_state_export_v1(
        axiom_qwen38_mtp_speculative *speculative,
        axiom_qwen38_mtp_speculative_persistent_state_v1 *out,
        uint64_t state_bytes);
int axiom_qwen38_mtp_speculative_persistent_state_import_v1(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_persistent_state_v1 *state,
        uint64_t state_bytes);

/* Synchronous resident-controller recycle. The borrowed target must already
 * be idle at position zero and no device session may exist. The call resets
 * only MTP logical state plus pending/controller cursors; captured exact and
 * fast CUDA graph executables remain owned by the controller and are not
 * destroyed or recaptured. A failure after mutation poisons the controller. */
int axiom_qwen38_mtp_speculative_recycle_to_zero(
        axiom_qwen38_mtp_speculative *speculative);

int axiom_qwen38_mtp_speculative_info_get(
        const axiom_qwen38_mtp_speculative *speculative,
        axiom_qwen38_mtp_speculative_info *out);

/* Reference prompt ingestion.  Target and MTP consume exactly one token and
 * commit together.  This function is intentionally host-observable and is
 * used by the parity gate; the production hot path uses the later device
 * enqueue ABI and may not silently call this function. */
int axiom_qwen38_mtp_speculative_prefill_token(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t token_id,
        axiom_qwen38_mtp_speculative_prefill_result *out);

/* One exact greedy gamma-7 reference cycle.  It emits the accepted MTP prefix
 * followed by one authoritative target token. */
int axiom_qwen38_mtp_speculative_step(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_step_request *request,
        axiom_qwen38_mtp_speculative_step_result *out);

/* Prepare captures exactly one fixed gamma-7 proposal, target M8,
 * target-authoritative accept, fixed-width MTP catch-up and conditional
 * finalize cycle.  Call it once after host prefill and before device enqueue.
 * An unavailable graph fails closed; no host/reference fallback is selected. */
int axiom_qwen38_mtp_speculative_device_prepare(
        axiom_qwen38_mtp_speculative *speculative);

/* Cold, explicit token-stop contract. It must be configured before graph
 * preparation, including an intentional zero-count configuration. Active IDs
 * must be unique Qwen vocabulary IDs; no multi-token sequence is inferred. */
int axiom_qwen38_mtp_speculative_device_stop_tokens_set(
        axiom_qwen38_mtp_speculative *speculative,
        const uint32_t *stop_token_ids,
        uint32_t stop_token_count);

int axiom_qwen38_mtp_speculative_device_step_enqueue(
        axiom_qwen38_mtp_speculative *speculative,
        const axiom_qwen38_mtp_speculative_device_step_request *request,
        axiom_qwen38_mtp_speculative_device_step_result *out);

/* Bound the next graph cycle to [1,8] authoritative emitted tokens. */
int axiom_qwen38_mtp_speculative_device_commit_limit_set(
        axiom_qwen38_mtp_speculative *speculative,
        uint32_t max_commit_tokens,
        void *stream);

/* Cold ownership handoff after all queued graph work.  The full committed
 * history is required to restore target replay/session metadata exactly. */
int axiom_qwen38_mtp_speculative_device_session_end(
        axiom_qwen38_mtp_speculative *speculative,
        void *stream,
        const uint32_t *committed_token_ids,
        uint32_t committed_token_count);

#ifdef __cplusplus
}
#endif

#endif
