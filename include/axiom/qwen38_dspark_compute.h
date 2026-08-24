#ifndef AXIOM_QWEN38_DSPARK_COMPUTE_H
#define AXIOM_QWEN38_DSPARK_COMPUTE_H

/*
 * Native resident CUDA execution for Qwen3.8-27B-DSpark.
 *
 * This object borrows the immutable BF16 tensor views owned by
 * axiom_qwen38_dspark and owns only its CUDA workspaces, BF16 draft KV cache,
 * and cuBLAS handle.  It never converts or duplicates the checkpoint.  The
 * target Qwen model remains the owner of its embedding table and output head;
 * those two operations are deliberately injected as asynchronous device
 * callbacks so a target implementation can reuse its existing resident
 * NVFP4/FP8 kernels.
 *
 * The first execution ABI intentionally describes one speculative sequence.
 * Every multi-column input is a temporal run for that one sequence, never a
 * set of independent B8 sessions.  A DSpark noise block has one anchor token
 * followed by up to six MASK slots; its seven logits predict up to seven
 * tokens.  All caller buffers are CUDA device pointers.  Calls enqueue work
 * on the supplied cudaStream_t encoded as void *, return after launch
 * validation, and never synchronize or allocate in their hot path.
 */

#include <stdint.h>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_qwen38_dspark_compute axiom_qwen38_dspark_compute;

/* DSpark compute layout revision.  The library-wide AXIOM_ABI_VERSION still
 * identifies the common Axiom API, while this revision protects DSpark-only
 * structures whose layouts changed independently.  New source is routed to
 * size-checked v2 symbols below; the legacy binary symbols remain exported
 * only to reject callers compiled against revision 1 without dereferencing
 * their smaller buffers. */
#define AXIOM_QWEN38_DSPARK_COMPUTE_LAYOUT_VERSION 2u

/* Target callbacks borrow target-owned CUDA storage.  `columns` is a temporal
 * width in [1,7], not a request batch. Token/hidden/logit storage is
 * column-major [hidden_size, columns] and [vocab_size, columns], with column
 * t corresponding to absolute position `first_position + t` in the caller's
 * target executor. Implementations must enqueue work on `stream`, must not
 * synchronize, and must not retain transient pointers after returning. */
typedef int (*axiom_qwen38_dspark_target_embed_f32_device_fn)(
        void *user_data,
        const uint32_t *token_ids,
        float *out_hidden,
        uint32_t columns,
        void *stream);

typedef int (*axiom_qwen38_dspark_target_lm_head_f32_device_fn)(
        void *user_data,
        const float *hidden,
        float *out_logits,
        uint32_t columns,
        void *stream);

typedef struct {
    uint32_t abi_version;
    uint32_t max_context;
    /* Greedy graph termination is device-owned so a stop token in an early
     * cycle of a batched replay cannot let later cycles advance recurrent
     * state. Zero disables stop handling for low-level parity tools. */
    uint32_t stop_token_count;
    uint32_t stop_token_ids[2];
    uint64_t flags;
} axiom_qwen38_dspark_compute_config;

typedef struct {
    uint32_t abi_version;
    void *user_data;
    axiom_qwen38_dspark_target_embed_f32_device_fn embed_f32_device;
    axiom_qwen38_dspark_target_lm_head_f32_device_fn lm_head_f32_device;
    uint64_t flags;
} axiom_qwen38_dspark_target_binding;

typedef struct {
    uint32_t abi_version;
    uint32_t device;
    uint32_t max_context;
    uint32_t hidden_size;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t draft_layers;
    uint32_t block_size;
    uint32_t cache_dtype;
    uint32_t explicit_streams;
    uint32_t transaction_open;
    uint32_t committed_position;
    uint64_t checkpoint_device_bytes;
    uint64_t workspace_device_bytes;
    uint64_t kv_cache_device_bytes;
    uint64_t device_bytes;
} axiom_qwen38_dspark_compute_info;

/*
 * Device-resident DSpark control plane used by a graph-capable target
 * executor.  These pointers are owned by `compute`, remain valid until it is
 * destroyed, and are deliberately never host-mirrored in the decode hot path.
 *
 * A caller starts one device session with an initial anchor/position, replays
 * the fixed seven-token proposal graph, invokes its target M8 verifier with
 * `verify_tokens_device`, calls `device_accept_greedy`, injects the returned
 * temporal taps, commits its own target transaction, then calls
 * `device_advance`.  All scalar inputs and outputs below are CUDA device
 * pointers.  Reading `async_status_device` is the caller's responsibility;
 * no API in this plane synchronizes to inspect it.
 */
#define AXIOM_QWEN38_DSPARK_COMPUTE_DEVICE_ABI_VERSION \
    AXIOM_QWEN38_DSPARK_COMPUTE_LAYOUT_VERSION

typedef struct axiom_qwen38_dspark_device_history axiom_qwen38_dspark_device_history;

typedef struct {
    uint32_t abi_version;
    uint32_t graph_ready;
    uint32_t device_session_active;
    uint32_t proposal_tokens;
    uint32_t verify_width;
    uint32_t reserved0;
    uint64_t resident_device_bytes;
    const uint32_t *anchor_token_device;      /* U32[1] */
    const uint32_t *anchor_position_device;   /* U32[1], next target cache slot */
    const uint32_t *proposal_tokens_device;   /* U32[7] */
    const uint32_t *verify_tokens_device;     /* U32[8], [anchor,draft0,...,draft6] */
    const uint32_t *accepted_prefix_device;   /* U32[1], in [0,7] */
    /* U32[1], normally in [1,8] and exactly 1 + accepted prefix. Zero is a
     * terminal no-op for cycles already queued behind a selected stop token. */
    const uint32_t *target_commit_prefix_device;
    const uint32_t *continuation_token_device;/* U32[1], target[accepted_prefix] */
    const float *continuation_logit_device;   /* F32[1], optional target logits path */
    const uint32_t *async_status_device;      /* U32[1], AXIOM_OK or an AXIOM error */
    /* 0=active, 1=stop selected in this cycle, 2=terminal/no-op. */
    const uint32_t *terminal_state_device;
    axiom_qwen38_dspark_device_history *history_device;
} axiom_qwen38_dspark_compute_device_state;

/* Compact device-to-host history record used by the native API after one
 * graph replay.  Keeping the five scalar results next to the seven proposal
 * ids turns small D2H transactions into one ordered transfer.  The added
 * committed-token authority field is part of DSpark device ABI revision 2. */
struct axiom_qwen38_dspark_device_history {
    uint32_t proposal_tokens[7];
    uint32_t accepted_prefix;
    uint32_t continuation_token;
    uint32_t async_status;
    uint32_t next_position;
    /* Exact authoritative inputs committed by this replay. Zero identifies a
     * terminal no-op cycle already queued behind the cycle that selected EOS. */
    uint32_t committed_tokens;
};

/* Enqueue the compacting kernel and one contiguous record write on `stream`.
 * All input pointers are borrowed device addresses and the output record is
 * caller-owned device memory large enough for the struct above. */
int axiom_qwen38_dspark_compute_device_history_pack_enqueue_v2(
        const uint32_t *proposal_tokens_device,
        const uint32_t *accepted_prefix_device,
        const uint32_t *continuation_token_device,
        const uint32_t *async_status_device,
        const uint32_t *next_position_device,
        const uint32_t *committed_tokens_device,
        axiom_qwen38_dspark_device_history *output_device,
        uint64_t output_device_bytes,
        void *stream);

/* Legacy revision-1 symbol.  It is retained for deterministic binary
 * compatibility failure and always returns AXIOM_ERR_INVALID_ARGUMENT. */
int axiom_qwen38_dspark_compute_device_history_pack_enqueue(
        const uint32_t *proposal_tokens_device,
        const uint32_t *accepted_prefix_device,
        const uint32_t *continuation_token_device,
        const uint32_t *async_status_device,
        const uint32_t *next_position_device,
        axiom_qwen38_dspark_device_history *output_device,
        void *stream);

/* Both fields are CUDA-device U32[1] values.  The compute object copies them
 * into its resident graph controls on `stream`; neither pointer is retained.
 */
typedef struct {
    uint32_t abi_version;
    const uint32_t *anchor_token_device;
    const uint32_t *anchor_position_device;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_compute_device_session_request;

/* Target output views are temporal, never batch lanes.  The target verifier
 * must write them on the same stream before `device_accept_greedy` or
 * `device_inject_target` is queued. */
typedef struct {
    uint32_t abi_version;
    const float *target_taps_device;          /* F32 tap-major [5][8][5120] */
    const uint32_t *target_token_ids_device;  /* U32[8] */
    /* F32[8] top-1 values. Every row must be finite; a target-side invalid
     * row is represented as NaN so device_accept_greedy fails closed before
     * any temporal taps are injected. */
    const float *target_logits_device;
    uint32_t columns;                         /* exactly 8 for M8 injection */
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_compute_device_target_view;

/* Fuses target hidden-state taps ordered exactly as DSpark's
 * target_layer_ids=[4,16,28,40,52], then injects K/V for all five draft
 * decoder layers. `target_taps` is device F32 logical
 * [target_feature=5][temporal_column=columns][hidden=5120], contiguous in
 * that order. The CUDA pack stage makes FC column t equal to
 * concat(tap0[t], tap1[t], tap2[t], tap3[t], tap4[t]) with shape
 * [25600, columns]. `columns` is a temporal target-forward width in [1,8].
 * It writes cache positions [target_position, target_position + columns).
 * Inject positions must be monotonic in normal serving; stale slots above a
 * rolled-back watermark are harmless and are overwritten before use. */
typedef struct {
    uint32_t abi_version;
    const float *target_taps;
    uint32_t target_position;
    uint32_t columns;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_compute_inject_request;

/* Generates a single DSpark block.  The decoder consumes an anchor token at
 * `anchor_position` plus MASK slots; logits at positions [0,proposal_tokens)
 * are Markov-biased and greedily chained to produce `out_tokens`.
 *
 * `out_tokens` is mandatory device U32[proposal_tokens].  `out_logits`, when
 * supplied, receives device F32[vocab_size, proposal_tokens] after Markov
 * bias. `out_confidence`, when supplied, receives device F32[proposal_tokens]
 * after sigmoid.  Cache entries in [0, anchor_position) must already have
 * been injected from the corresponding target hidden-state taps. */
typedef struct {
    uint32_t abi_version;
    uint32_t anchor_token;
    uint32_t anchor_position;
    uint32_t proposal_tokens;
    uint32_t *out_tokens;
    float *out_logits;
    float *out_confidence;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_compute_propose_request;

/* The transaction API is deliberately logical: DSpark's noncausal noise block
 * uses local K/V only, so `propose` never appends draft K/V to the persistent
 * injected cache.  begin/commit/abort expose the same target-verification
 * boundary as the target model. Begin is made at q, where x_q is the target's
 * unconsumed anchor. After target temporal verification of
 * [x_q,draft0,...,draft6], commit(L) advances the watermark to q + 1 + L:
 * x_q and the accepted L-token prefix are now consumed and their target taps
 * may be retained. While a transaction is open, inject_target must stage taps
 * beginning at q; commit(L) requires at least 1+L staged temporal columns.
 * Abort restores the watermark to q. Neither call synchronizes or clears
 * cache bytes. */
int axiom_qwen38_dspark_compute_transaction_begin(
        axiom_qwen38_dspark_compute *compute,
        uint32_t anchor_position,
        void *stream);
int axiom_qwen38_dspark_compute_transaction_commit(
        axiom_qwen38_dspark_compute *compute,
        uint32_t accepted_prefix,
        void *stream);
int axiom_qwen38_dspark_compute_transaction_abort(
        axiom_qwen38_dspark_compute *compute,
        void *stream);

/* `dspark` and `target` storage must outlive `compute`. `max_context` is a
 * real BF16 K/V-cache allocation limit, not the 262K checkpoint declaration;
 * set it to the serving context budget. */
int axiom_qwen38_dspark_compute_create_v2(
        const axiom_qwen38_dspark *dspark,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_dspark_compute_config *config,
        uint64_t config_bytes,
        const axiom_qwen38_dspark_target_binding *target,
        axiom_qwen38_dspark_compute **out);

/* Legacy revision-1 symbol.  No config fields are read; the call is rejected
 * before any access to the historical, smaller config layout. */
int axiom_qwen38_dspark_compute_create(
        const axiom_qwen38_dspark *dspark,
        axiom_runtime *runtime,
        int device,
        const axiom_qwen38_dspark_compute_config *config,
        const axiom_qwen38_dspark_target_binding *target,
        axiom_qwen38_dspark_compute **out);

void axiom_qwen38_dspark_compute_destroy(axiom_qwen38_dspark_compute *compute);
uint64_t axiom_qwen38_dspark_compute_device_bytes(
        const axiom_qwen38_dspark_compute *compute);
int axiom_qwen38_dspark_compute_info_get(
        const axiom_qwen38_dspark_compute *compute,
        axiom_qwen38_dspark_compute_info *out);
/* Set the committed position after restoring the corresponding target/draft
 * KV pages. This is a cold-start restore operation; no transaction may be
 * open and no CUDA work is enqueued. */
int axiom_qwen38_dspark_compute_restore_position(
        axiom_qwen38_dspark_compute *compute,
        uint32_t position);

/* Synchronous storage-tier bridge for one DSpark BF16 cache layer. The tier
 * record is E4M3 K followed by E4M3 V; conversion is explicit and therefore
 * never silently aliases the BF16 device cache. */
int axiom_qwen38_dspark_compute_kv_page_export(
        const axiom_qwen38_dspark_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        void *host_page,
        uint64_t host_page_bytes);
int axiom_qwen38_dspark_compute_kv_page_import(
        axiom_qwen38_dspark_compute *compute,
        uint32_t layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes);

int axiom_qwen38_dspark_compute_inject_target(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_inject_request *request);
int axiom_qwen38_dspark_compute_propose(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_propose_request *request);

/*
 * Fixed-shape CUDA-graph half of the DSpark decode pipeline.  Preparation
 * allocates/captures once and is outside the decode hot path.  Replay,
 * verify-input assembly, greedy acceptance, injection and advancement only
 * enqueue work on the supplied stream; they allocate and synchronize never.
 *
 * Graph replay is deliberately unavailable rather than silently eager when
 * capture fails.  This keeps a device-only caller from accidentally taking a
 * host-synchronized path.
 */
int axiom_qwen38_dspark_compute_device_state_get_v2(
        const axiom_qwen38_dspark_compute *compute,
        axiom_qwen38_dspark_compute_device_state *out,
        uint64_t out_bytes);
/* Legacy revision-1 symbol.  The old, smaller output buffer is never
 * inspected or cleared and the call always fails closed. */
int axiom_qwen38_dspark_compute_device_state_get(
        const axiom_qwen38_dspark_compute *compute,
        axiom_qwen38_dspark_compute_device_state *out);
int axiom_qwen38_dspark_compute_device_graph_prepare(
        axiom_qwen38_dspark_compute *compute);
/* Enter/leave full-cycle graph capture. These calls alter only host-side
 * ownership state; all CUDA work between them must be captured on the caller
 * stream. They cannot run while a live device session exists. */
int axiom_qwen38_dspark_compute_device_graph_capture_begin(
        axiom_qwen38_dspark_compute *compute);
int axiom_qwen38_dspark_compute_device_graph_capture_end(
        axiom_qwen38_dspark_compute *compute);
int axiom_qwen38_dspark_compute_device_session_begin(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_session_request *request);
int axiom_qwen38_dspark_compute_device_session_abort(
        axiom_qwen38_dspark_compute *compute,
        void *stream);
int axiom_qwen38_dspark_compute_device_graph_replay(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_session_request *request);
/* Set the maximum authoritative input prefix for the next and subsequent
 * graph replays. `max_commit_tokens` includes the anchor and is in [1,8].
 * The graph clamps natural acceptance to this device-resident boundary. */
int axiom_qwen38_dspark_compute_device_commit_limit_set(
        axiom_qwen38_dspark_compute *compute,
        uint32_t max_commit_tokens,
        void *stream);
/* Eager enqueue primitives used only while preparing a larger fixed-cycle
 * graph. They read the resident anchor/position controls and have no graph,
 * host copy, allocation, or synchronization of their own. */
int axiom_qwen38_dspark_compute_device_cycle_begin_enqueue(
        axiom_qwen38_dspark_compute *compute,
        void *stream);
int axiom_qwen38_dspark_compute_device_propose_enqueue(
        axiom_qwen38_dspark_compute *compute,
        void *stream);
int axiom_qwen38_dspark_compute_device_build_verify_input(
        axiom_qwen38_dspark_compute *compute,
        void *stream);
int axiom_qwen38_dspark_compute_device_accept_greedy(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_target_view *target);
int axiom_qwen38_dspark_compute_device_inject_target(
        axiom_qwen38_dspark_compute *compute,
        const axiom_qwen38_dspark_compute_device_target_view *target);
int axiom_qwen38_dspark_compute_device_advance(
        axiom_qwen38_dspark_compute *compute,
        void *stream);

/* Source-compatibility shims route every caller rebuilt with this header to
 * the size-checked revision-2 symbols.  Define the implementation guard only
 * in the translation unit that exports both the legacy and v2 symbols. */
#if !defined(AXIOM_QWEN38_DSPARK_COMPUTE_IMPLEMENTATION)
#define axiom_qwen38_dspark_compute_create(dspark, runtime, device, config, target, out) \
    axiom_qwen38_dspark_compute_create_v2(                                      \
            (dspark), (runtime), (device), (config),                            \
            (uint64_t) sizeof(*(config)), (target), (out))
#define axiom_qwen38_dspark_compute_device_state_get(compute, out) \
    axiom_qwen38_dspark_compute_device_state_get_v2(              \
            (compute), (out), (uint64_t) sizeof(*(out)))
#define axiom_qwen38_dspark_compute_device_history_pack_enqueue(                 \
        proposal, accepted, continuation, status, position, committed, output, stream) \
    axiom_qwen38_dspark_compute_device_history_pack_enqueue_v2(                  \
            (proposal), (accepted), (continuation), (status), (position),        \
            (committed), (output), (uint64_t) sizeof(*(output)), (stream))
#endif

#ifdef __cplusplus
}
#endif

#endif
