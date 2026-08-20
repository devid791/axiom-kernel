#ifndef AXIOM_QWEN38_SPECULATIVE_H
#define AXIOM_QWEN38_SPECULATIVE_H

/*
 * Native single-stream Qwen3.8 DSpark speculative controller.
 *
 * The executable backend is enabled only when the target advertises a true
 * same-session temporal-M8 verifier: causal GDN rows, triangular full-
 * attention KV, temporal target taps [5,8,5120], and suffix rollback on
 * commit_prefix.  Eight independent batch lanes and M1 replay loops are not
 * compatible and must leave backend_available() equal to zero.
 *
 * Once that contract is present, one step stages a seven-token DSpark proposal, verifies
 * [anchor, draft_0, ..., draft_6] through the target model's transactional
 * width-eight verifier, accepts the longest greedy prefix, and emits the
 * accepted draft tokens followed by one authoritative target token.  The
 * latter is the full-block bonus when all seven draft tokens match and the
 * corrective token at the first mismatch otherwise.
 *
 * The controller borrows target, loader and compute objects.  It never owns
 * or converts their weights.  Calls are serialized per controller.  Any
 * failure before target commit aborts both transactions; a failure after the
 * authoritative target commits poisons this controller so callers can fall
 * back to the target model instead of continuing with divergent draft state.
 */

#include <stdint.h>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"
#include "axiom/qwen38_dspark_compute.h"
#include "axiom/qwen38_model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS 7u
#define AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH 8u

typedef struct axiom_qwen38_dspark_compute axiom_qwen38_dspark_compute;
typedef struct axiom_qwen38_speculative axiom_qwen38_speculative;

/*
 * Optional device-native target transaction ABI for Dirac/Axiom executors.
 * The current Qwen target ABI accepts host U32[8] input and therefore cannot
 * implement this contract without an adapter.  A null binding never falls
 * back from `device_step_enqueue` to the legacy host path: it returns
 * AXIOM_ERR_NOT_IMPLEMENTED instead.
 *
 * All pointer fields below address persistent CUDA memory.  A callback must
 * enqueue exclusively on `stream`, retain no transient caller pointers, make
 * no allocation or synchronization in its decode path, and leave its output
 * view valid until the next callback on that target session. `commit` takes
 * `target_commit_prefix_device[0]` in [1,8], the count of target-input rows
 * to retain including the anchor; `abort` restores the pre-verify suffix on
 * the same stream.
 */
#define AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION 1u
/* Binding flags. A capture-safe target has stable device addresses and its
 * callbacks only enqueue graph-capturable CUDA work; the full controller
 * refuses to bind anything else. */
#define AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_FLAG_CAPTURE_SAFE (1ull << 0)

typedef struct {
    uint32_t abi_version;
    const float *target_taps_device;          /* F32 [5][8][5120], tap-major */
    const uint32_t *target_token_ids_device;  /* U32[8], prediction after each input */
    /* F32[8] top-1 values; invalid rows must be NaN. This lets the
     * controller fail closed without a host-side target-status read. */
    const float *target_logits_device;
    uint32_t target_tap_count;
    uint32_t target_tap_tokens;
    uint32_t target_tap_hidden_size;
    uint32_t reserved;
} axiom_qwen38_speculative_device_verify_view;

typedef int (*axiom_qwen38_speculative_device_transaction_begin_fn)(
        void *user_data,
        const uint32_t *anchor_position_device,
        void *stream);
typedef int (*axiom_qwen38_speculative_device_transaction_verify_fn)(
        void *user_data,
        const uint32_t *verify_tokens_device,
        const uint32_t *anchor_position_device,
        uint32_t verify_width,
        void *stream,
        axiom_qwen38_speculative_device_verify_view *out);
typedef int (*axiom_qwen38_speculative_device_transaction_commit_fn)(
        void *user_data,
        const uint32_t *target_commit_prefix_device,
        void *stream);
typedef int (*axiom_qwen38_speculative_device_transaction_abort_fn)(
        void *user_data,
        void *stream);
/* Graph-only finalization. It must retain `target_commit_prefix_device[0]`
 * inputs when `async_status_device[0] == AXIOM_OK`, otherwise restore the
 * target transaction. It is the device-only equivalent of host commit/abort.
 */
typedef int (*axiom_qwen38_speculative_device_transaction_finalize_fn)(
        void *user_data,
        const uint32_t *target_commit_prefix_device,
        const uint32_t *async_status_device,
        void *stream);

typedef struct {
    uint32_t abi_version;
    void *user_data;
    axiom_qwen38_speculative_device_transaction_begin_fn transaction_begin;
    axiom_qwen38_speculative_device_transaction_verify_fn transaction_verify;
    axiom_qwen38_speculative_device_transaction_commit_fn transaction_commit;
    axiom_qwen38_speculative_device_transaction_abort_fn transaction_abort;
    axiom_qwen38_speculative_device_transaction_finalize_fn transaction_finalize;
    uint64_t flags;
} axiom_qwen38_speculative_device_target_binding;

typedef struct {
    uint32_t abi_version;
    int32_t device;
    uint32_t reserved;
    uint64_t flags; /* Reserved; must be zero in ABI v1. */
} axiom_qwen38_speculative_config;

typedef struct {
    uint32_t abi_version;
    uint32_t anchor_token_id;
    void *stream;  /* ABI v1 requires NULL: target verification is default-stream only. */
    uint64_t flags; /* Must be zero in ABI v1. */
} axiom_qwen38_speculative_step_request;

typedef struct {
    uint32_t abi_version;
    uint32_t snapshot_position;
    uint32_t position_after_verify;
    uint32_t position_after_commit;
    uint32_t accepted_draft_prefix;
    uint32_t emitted_token_count;
    uint32_t draft_token_ids[AXIOM_QWEN38_SPECULATIVE_DRAFT_TOKENS];
    uint32_t target_token_ids[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH];
    float target_logits[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH];
    uint32_t emitted_token_ids[AXIOM_QWEN38_SPECULATIVE_VERIFY_WIDTH];
    uint32_t continuation_token_id;
    float continuation_logit;
    uint32_t full_block_accept;
    uint32_t reserved;
    uint64_t inject_ns;
    uint64_t propose_ns;
    uint64_t proposal_copy_ns;
    uint64_t verify_ns;
    uint64_t commit_ns;
    uint64_t total_ns;
} axiom_qwen38_speculative_step_result;

/* Device-only enqueue request/result. The two request scalars are CUDA U32[1]
 * values. They seed the first graph replay only. After that, the graph carries
 * its authoritative continuation/position internally, so subsequent requests
 * must pass the `next_anchor_*_device` pointers returned below. Result
 * pointers are borrowed from the controller/compute object and become invalid
 * on destroy. */
typedef struct {
    uint32_t abi_version;
    const uint32_t *anchor_token_device;
    const uint32_t *anchor_position_device;
    void *stream;
    uint64_t flags;
} axiom_qwen38_speculative_device_step_request;

typedef struct {
    uint32_t abi_version;
    uint32_t graph_replayed;
    uint32_t proposal_tokens;
    uint32_t verify_width;
    const uint32_t *proposal_tokens_device;
    const uint32_t *verify_tokens_device;
    const uint32_t *accepted_prefix_device;
    const uint32_t *target_commit_prefix_device;
    const uint32_t *continuation_token_device;
    const float *continuation_logit_device;
    const uint32_t *next_anchor_token_device;
    const uint32_t *next_anchor_position_device;
    const uint32_t *anchor_position_device;
    const uint32_t *async_status_device;
    const axiom_qwen38_dspark_device_history *history_device;
} axiom_qwen38_speculative_device_step_result;

/* Sequential prompt ingestion. The target consumes `token_id` transactionally,
 * exposes its temporal row-zero taps, and the controller installs those taps
 * into the DSpark cache at the same absolute position. `target_token_id` is
 * the authoritative next-token prediction and can be chained as the first
 * speculative anchor after the final prompt token. */
typedef struct {
    uint32_t abi_version;
    uint32_t token_id;
    void *stream;  /* ABI v1 requires NULL. */
    uint64_t flags; /* Must be zero in ABI v1. */
    /* Optional CUDA F32[5120] multimodal embedding. NULL preserves the
     * ordinary tokenizer-embedding path. The logical token_id is still
     * committed for replay/session bookkeeping. */
    const float *embedding_device;
} axiom_qwen38_speculative_prefill_request;

typedef struct {
    uint32_t abi_version;
    uint32_t token_position;
    uint32_t position_after_commit;
    uint32_t target_token_id;
    float target_logit;
    uint32_t reserved;
    uint64_t target_ns;
    uint64_t commit_ns;
    uint64_t inject_ns;
    uint64_t total_ns;
} axiom_qwen38_speculative_prefill_result;

typedef struct {
    uint32_t abi_version;
    uint32_t poisoned;
    uint32_t last_status;
    uint32_t last_accepted_draft_prefix;
    uint64_t attempted_steps;
    uint64_t committed_steps;
    uint64_t aborted_steps;
    uint64_t failed_steps;
    uint64_t full_accept_steps;
    uint64_t correction_steps;
    uint64_t proposed_draft_tokens;
    uint64_t accepted_draft_tokens;
    uint64_t rejected_draft_tokens;
    uint64_t emitted_tokens;
    uint64_t authoritative_tail_tokens;
    uint64_t prefill_attempted_tokens;
    uint64_t prefill_committed_tokens;
    uint64_t prefill_failed_tokens;
    uint64_t prefill_target_ns;
    uint64_t prefill_commit_ns;
    uint64_t prefill_inject_ns;
    uint64_t prefill_total_ns;
    uint64_t inject_ns;
    uint64_t propose_ns;
    uint64_t proposal_copy_ns;
    uint64_t verify_ns;
    uint64_t commit_ns;
    uint64_t total_ns;
} axiom_qwen38_speculative_counters;

/* Returns one only when both the true temporal-M8 target transaction ABI and
 * matching native DSpark transaction ABI were available while this
 * translation unit was compiled.  create() additionally requires the bound
 * model instance to have passed its temporal-M8 parity validation gate. */
int axiom_qwen38_speculative_backend_available(void);

int axiom_qwen38_speculative_create(
        axiom_qwen38_model *target,
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_speculative_config *config,
        axiom_qwen38_speculative **out);

void axiom_qwen38_speculative_destroy(axiom_qwen38_speculative *speculative);

/* Bind the optional Dirac-style device target after legacy creation.  This is
 * deliberately separate from the v1 create-config so binaries built against
 * the pre-device header keep their exact config size. Binding captures the
 * complete fixed seven-token cycle (proposal, target M8, accept, injection,
 * finalize) once; an unavailable graph returns an error and leaves
 * device_step_enqueue fail-closed. */
int axiom_qwen38_speculative_device_target_bind(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_device_target_binding *binding);

/* Native Qwen target factory supplied by the target executor. It returns the
 * capture-safe, persistent-buffer binding required above; separated here to
 * avoid making qwen38_model.h depend on speculative controller types. */
int axiom_qwen38_model_dspark_device_target_binding_get(
        axiom_qwen38_model *model,
        axiom_qwen38_speculative_device_target_binding *out);

/* Convenience full-chain binding for the target passed to speculative_create.
 * It calls the native factory then captures the complete fixed-cycle graph. */
int axiom_qwen38_speculative_device_target_bind_qwen38_model(
        axiom_qwen38_speculative *speculative);

int axiom_qwen38_speculative_prefill_token(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_prefill_request *request,
        axiom_qwen38_speculative_prefill_result *out);

int axiom_qwen38_speculative_step(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_step_request *request,
        axiom_qwen38_speculative_step_result *out);

/* Enqueue one complete greedy gamma=7 device-only step through the optional
 * Dirac target ABI.  There is intentionally no host-compatible fallback:
 * target/device graph incompatibility is returned before enqueuing a partial
 * transaction.  The caller may read or synchronize `async_status_device` at
 * an explicit boundary outside decode. */
int axiom_qwen38_speculative_device_step_enqueue(
        axiom_qwen38_speculative *speculative,
        const axiom_qwen38_speculative_device_step_request *request,
        axiom_qwen38_speculative_device_step_result *out);

int axiom_qwen38_speculative_counters_get(
        const axiom_qwen38_speculative *speculative,
        axiom_qwen38_speculative_counters *out);

#ifdef __cplusplus
}
#endif

#endif
