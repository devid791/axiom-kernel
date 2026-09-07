#ifndef AXIOM_QWEN38_MTP_COMPUTE_H
#define AXIOM_QWEN38_MTP_COMPUTE_H

/*
 * Native CUDA execution contract for the Qwen3.8 MTP decoder block.
 *
 * The compute object borrows the immutable BF16 weights owned by
 * axiom_qwen38_mtp and the immutable target binding supplied at creation.
 * It owns only its CUDA workspaces and MTP K/V cache. The MTP object,
 * runtime, target callback storage, and target callback user_data must
 * outlive the compute object.
 *
 * This ABI represents exactly one temporal sequence. `columns` is a causal
 * run of consecutive positions for that sequence, never a request batch.
 * All token, hidden, logit, and top-1 pointers are CUDA device pointers.
 * Transaction begin, forward, commit, and abort enqueue work on one supplied
 * stream, perform no allocation, perform no device/host synchronization, and
 * never retain request-owned transient pointers after returning.
 */

#include <stdint.h>

#include "axiom/axiom.h"
#include "axiom/qwen38_attention.h"
#include "axiom/qwen38_kv_tier.h"
#include "axiom/qwen38_mtp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION 1u
#define AXIOM_QWEN38_MTP_COMPUTE_MAX_COLUMNS 8u
#define AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION 2u
#define AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_COMPUTE_KV_CATCHUP_ABI_VERSION 1u
#define AXIOM_QWEN38_MTP_COMPUTE_KV_LAYERS 1u
#define AXIOM_QWEN38_MTP_COMPUTE_KV_PAGE_TOKENS \
    AXIOM_QWEN38_KV_TIER_PAGE_TOKENS
#define AXIOM_QWEN38_MTP_COMPUTE_KV_PAGE_BYTES \
    AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES

typedef struct axiom_qwen38_mtp_compute axiom_qwen38_mtp_compute;

/* These callback signatures intentionally match the existing DSpark target
 * callbacks. `columns` is temporal width in [1,8]. Hidden storage is CUDA F32
 * column-major [hidden_size, columns]; logits are CUDA F32 column-major
 * [vocab_size, columns]. Implementations enqueue on `stream`, do not
 * synchronize, and do not retain any transient pointer. */
typedef int (*axiom_qwen38_mtp_target_embed_f32_device_fn)(
        void *user_data,
        const uint32_t *token_ids,
        float *out_hidden,
        uint32_t columns,
        void *stream);

typedef int (*axiom_qwen38_mtp_target_lm_head_f32_device_fn)(
        void *user_data,
        const float *hidden,
        float *out_logits,
        uint32_t columns,
        void *stream);

typedef struct {
    uint32_t abi_version;
    uint32_t max_context;
    /* axiom_tensor_dtype value; unsupported formats fail creation. */
    uint32_t cache_dtype;
    uint64_t flags;
} axiom_qwen38_mtp_compute_config;

/* Revision 2 makes the immutable RoPE geometry explicit.  It is deliberately
 * a new structure and entry point so revision-1 binary layouts do not change.
 * The legacy v1 factory retains its original YaRN4 behavior for ABI
 * compatibility; production controllers must use v2 and compare the result
 * with the loaded target profile. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t max_context;
    uint32_t cache_dtype;
    uint32_t rope_profile; /* axiom_qwen38_rope_profile */
    uint32_t reserved;
    uint64_t flags;
} axiom_qwen38_mtp_compute_config_v2;

typedef struct {
    uint32_t abi_version;
    void *user_data;
    axiom_qwen38_mtp_target_embed_f32_device_fn embed_f32_device;
    axiom_qwen38_mtp_target_lm_head_f32_device_fn lm_head_f32_device;
    uint64_t flags;
} axiom_qwen38_mtp_target_binding;

typedef enum {
    AXIOM_QWEN38_MTP_TRANSACTION_IDLE = 0,
    AXIOM_QWEN38_MTP_TRANSACTION_OPEN = 1,
    AXIOM_QWEN38_MTP_TRANSACTION_FAILED = 2,
} axiom_qwen38_mtp_transaction_state;

typedef struct {
    uint32_t abi_version;
    uint32_t device;
    uint32_t max_context;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t mtp_layers;
    uint32_t vocab_size;
    uint32_t max_columns;
    uint32_t cache_dtype;
    uint32_t transaction_state;
    uint32_t transaction_begin_position;
    uint32_t staged_position;
    uint32_t committed_position;
    uint64_t checkpoint_device_bytes;
    uint64_t workspace_device_bytes;
    uint64_t kv_cache_device_bytes;
    /* Total resident bytes: checkpoint + workspace + K/V cache. */
    uint64_t device_bytes;
} axiom_qwen38_mtp_compute_info;

/* Execute consecutive MTP cache positions
 * [first_position, first_position + columns).
 *
 * `token_ids_device` is CUDA U32[columns]. `prior_hidden_device` is the exact
 * predecessor state supplied by the caller as CUDA F32
 * [hidden_size, columns]: column t is paired with token t and is not shifted
 * or inferred by the executor. During autoregressive drafting, call with one
 * column and feed the preceding call's output hidden into the next call.
 * During target catch-up, the caller may supply the already-known shifted
 * target hidden states as a wider temporal run.
 *
 * `out_hidden_device` is mandatory CUDA F32 [hidden_size, columns] after the
 * final MTP norm. `out_logits_device`, when non-NULL, receives CUDA F32
 * [vocab_size, columns] from the borrowed target LM head. `out_top1_device`,
 * when non-NULL, receives CUDA U32[columns]. Logits and top-1 are independently
 * optional; all output pointers may alias neither inputs nor each other.
 */
typedef struct {
    uint32_t abi_version;
    const uint32_t *token_ids_device;
    const float *prior_hidden_device;
    float *out_hidden_device;
    float *out_logits_device;
    uint32_t *out_top1_device;
    uint32_t columns;
    uint32_t first_position;
    void *stream;
    uint64_t flags;
} axiom_qwen38_mtp_compute_forward_request;

/* Revision-1 device-position forward contract.  Unlike the host transaction
 * request above, the first cache position is resolved on the GPU as
 * `base_position_device[0] + position_offset`.  This makes a fixed request
 * graph-replayable while the caller updates only persistent device memory.
 *
 * `async_status_device` is caller-owned CUDA U32[1], initialized to AXIOM_OK
 * before a session.  The executor preserves the first non-zero status.  A
 * position overflow or context violation substitutes safe position zero and
 * prevents authoritative cache writes.  When top-1 is requested, a column
 * containing no finite logit publishes AXIOM_ERR_RUNTIME and UINT32_MAX.
 *
 * Every pointer is a persistent CUDA device pointer.  Pointer ranges may not
 * overlap.  The explicit stream must be non-NULL.  request_bytes and
 * struct_size must exactly match this revision; flags must be zero. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *token_ids_device;
    const float *prior_hidden_device;
    float *out_hidden_device;
    float *out_logits_device;
    uint32_t *out_top1_device;
    const uint32_t *base_position_device;
    uint32_t position_offset;
    uint32_t columns;
    uint32_t *async_status_device;
    void *stream;
    uint64_t flags;
} axiom_qwen38_mtp_compute_device_forward_request_v1;

/* Revision-1 device-position KV-only catch-up contract.  This is the
 * target-authoritative cache installation prefix of the full MTP forward:
 * token sanitization and target embedding, both pre-FC RMSNorms, fusion, FC,
 * input RMSNorm, Q/K/V projection, the joint Q/K normalization and RoPE
 * kernel, and MTP K/V store.  Preserving the complete prefix keeps M8 cache
 * bytes bit-identical to the full checkpoint forward.  It deliberately
 * produces no hidden/logit/token output and never executes attention,
 * O projection, MLP, final norm, or LM head.
 *
 * `prior_hidden_device` is CUDA F32 [hidden_size, columns] containing the
 * exact target hidden state paired with each token.  The first cache position
 * is resolved entirely on device as
 * `base_position_device[0] + position_offset`.  `async_status_device` is a
 * caller-owned CUDA U32[1] initialized to AXIOM_OK before the device session;
 * the operation preserves the first error.  Invalid tokens or positions use
 * safe workspace values but cannot write authoritative cache entries.
 *
 * Every pointer is a persistent CUDA device pointer.  Pointer ranges may not
 * overlap.  The explicit stream must be non-NULL.  request_bytes and
 * struct_size must exactly match this revision; flags must be zero. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *token_ids_device;
    const float *prior_hidden_device;
    const uint32_t *base_position_device;
    uint32_t position_offset;
    uint32_t columns;
    uint32_t *async_status_device;
    void *stream;
    uint64_t flags;
} axiom_qwen38_mtp_compute_device_kv_catchup_request_v1;

/* Size-checked control-plane snapshot for the device-position path. */
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t device_session_active;
    uint32_t graph_capture_active;
    uint32_t device_session_begin_position;
    uint32_t committed_position;
    uint32_t max_context;
    uint32_t rope_profile; /* axiom_qwen38_rope_profile */
    uint64_t workspace_device_bytes;
    uint64_t device_bytes;
} axiom_qwen38_mtp_compute_device_info_v1;

/* Create validates `config_bytes == sizeof(*config)` before reading any
 * config field. `abi_version` in every revision-1 structure must equal
 * AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION. Unsupported cache dtypes or target
 * bindings are rejected; they never select a slower or scalar fallback.
 * Creation may allocate and synchronize. The decode hot path may not. */
int axiom_qwen38_mtp_compute_create_v1(
        const axiom_qwen38_mtp *mtp,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_mtp_compute_config *config,
        uint64_t config_bytes,
        const axiom_qwen38_mtp_target_binding *target,
        axiom_qwen38_mtp_compute **out);
int axiom_qwen38_mtp_compute_create_v2(
        const axiom_qwen38_mtp *mtp,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_mtp_compute_config_v2 *config,
        uint64_t config_bytes,
        const axiom_qwen38_mtp_target_binding *target,
        axiom_qwen38_mtp_compute **out);

void axiom_qwen38_mtp_compute_destroy(axiom_qwen38_mtp_compute *compute);

/* Reset discards every logical MTP cache entry and returns to IDLE at
 * position zero. It does not need to clear unreachable physical cache bytes.
 * Restore is a cold control-plane operation: no transaction may be open, the
 * position must not exceed max_context, and the implementation must reject a
 * position for which matching MTP K/V bytes are not known to be resident.
 * A complete imported page prefix authorizes one explicit forward restore;
 * page import itself never changes the logical position. */
int axiom_qwen38_mtp_compute_reset(axiom_qwen38_mtp_compute *compute);
int axiom_qwen38_mtp_compute_restore_position(
        axiom_qwen38_mtp_compute *compute,
        uint32_t position);

/* Synchronous storage bridge for the single native-MTP cache layer. `layer`
 * must be exactly zero. The caller owns `host_page`, whose size must be
 * exactly 524288 bytes and whose explicit wire layout is E4M3
 * K[256,4,256] followed by E4M3 V[256,4,256]. Internal BF16 caches are
 * converted element-by-element; internal E4M3 caches are copied without a
 * layout or dtype reinterpretation. Both calls synchronize outstanding CUDA
 * work before touching cache storage and return only after the transfer is
 * complete. They are illegal during host transactions, graph capture, or a
 * device session. Import records residency but never advances position. */
int axiom_qwen38_mtp_compute_kv_page_export(
        const axiom_qwen38_mtp_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        void *host_page,
        uint64_t host_page_bytes);
int axiom_qwen38_mtp_compute_kv_page_import(
        axiom_qwen38_mtp_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes);

uint32_t axiom_qwen38_mtp_compute_position(
        const axiom_qwen38_mtp_compute *compute);
uint64_t axiom_qwen38_mtp_compute_device_bytes(
        const axiom_qwen38_mtp_compute *compute);
axiom_qwen38_rope_profile axiom_qwen38_mtp_compute_rope_profile(
        const axiom_qwen38_mtp_compute *compute);
int axiom_qwen38_mtp_compute_info_get(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out);
int axiom_qwen38_mtp_compute_device_info_get_v1(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_device_info_v1 *out,
        uint64_t info_bytes);

/* Capture markers protect the compute-owned fixed workspace while callers
 * record one or more device forwards into an externally managed CUDA graph.
 * Begin is legal only while the host transaction is IDLE and no device
 * session exists.  No CUDA capture is started or ended by these markers. */
int axiom_qwen38_mtp_compute_device_graph_capture_begin(
        axiom_qwen38_mtp_compute *compute);
int axiom_qwen38_mtp_compute_device_graph_capture_end(
        axiom_qwen38_mtp_compute *compute);

/* A device session owns the logical MTP sequence while a captured graph is
 * replayed or device-position forwards are enqueued directly.  Begin must
 * match the current committed host position.  After synchronizing the caller
 * stream and checking async_status_device, end materializes a monotonic
 * position in host metadata.  Abort restores the begin position. */
int axiom_qwen38_mtp_compute_device_session_begin(
        axiom_qwen38_mtp_compute *compute,
        uint32_t expected_host_position);
int axiom_qwen38_mtp_compute_device_session_end(
        axiom_qwen38_mtp_compute *compute,
        uint32_t materialized_position);
int axiom_qwen38_mtp_compute_device_session_abort(
        axiom_qwen38_mtp_compute *compute);

/* Graph-capturable enqueue: no allocation, synchronization, or D2H transfer.
 * It is legal only inside a capture marker or an active device session. */
int axiom_qwen38_mtp_compute_forward_device_enqueue_v1(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_forward_request_v1 *request,
        uint64_t request_bytes);

/* Graph-capturable target-authoritative MTP K/V catch-up.  It performs no
 * allocation, host synchronization, D2H transfer, or fallback, and is legal
 * only inside a compute capture marker or an active device session.  BF16 and
 * FP8 cache storage and the immutable compute RoPE profile are honored exactly
 * as by the full device forward. */
int axiom_qwen38_mtp_compute_kv_catchup_device_enqueue_v1(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_device_kv_catchup_request_v1 *request,
        uint64_t request_bytes);

/* Transaction invariants:
 *
 *   - committed_position C is the exclusive end of authoritative MTP K/V;
 *   - begin(C) is accepted only in IDLE and fixes both transaction start and
 *     staged_position to C;
 *   - each forward must start exactly at staged_position, use 1..8 columns,
 *     remain within max_context, and advances only staged_position;
 *   - commit_prefix(N), where 0 <= N <= staged_position - C, exposes exactly
 *     [C, C + N), sets committed_position to C + N, and returns to IDLE;
 *   - abort leaves committed_position at C and returns to IDLE;
 *   - cache bytes above committed_position are unreachable stale storage and
 *     must be overwritten before they can become authoritative.
 *
 * All calls in one transaction must use the same CUDA stream. Any validation,
 * callback, or launch error observed after begin moves the transaction to
 * FAILED without changing committed_position. FAILED rejects further forward
 * and commit calls; only abort, reset, or destroy is legal. No error may
 * silently invoke a non-MTP, scalar, host-synchronized, or eager fallback.
 */
int axiom_qwen38_mtp_compute_transaction_begin(
        axiom_qwen38_mtp_compute *compute,
        uint32_t first_position,
        void *stream);
int axiom_qwen38_mtp_compute_forward(
        axiom_qwen38_mtp_compute *compute,
        const axiom_qwen38_mtp_compute_forward_request *request);
int axiom_qwen38_mtp_compute_transaction_commit_prefix(
        axiom_qwen38_mtp_compute *compute,
        uint32_t committed_columns,
        void *stream);
int axiom_qwen38_mtp_compute_transaction_abort(
        axiom_qwen38_mtp_compute *compute,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
