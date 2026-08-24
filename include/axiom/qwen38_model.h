#ifndef AXIOM_QWEN38_MODEL_H
#define AXIOM_QWEN38_MODEL_H

/* Native, local Axiom forward for unsloth/Qwen3.8-27B-NVFP4.
 *
 * This is the first complete decoder graph: embedding -> 64 token-mixer/MLP
 * blocks -> final RMS norm -> original FP8 LM head.  It exposes both a
 * compatibility single-token entry and its native eight-column forward.
 * It is not yet a continuous-batching scheduler.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_qwen38_model axiom_qwen38_model;
typedef struct axiom_qwen38_model_transaction axiom_qwen38_model_transaction;
typedef struct axiom_qwen38_kv_tier axiom_qwen38_kv_tier;

#define AXIOM_QWEN38_MODEL_VOCAB 248320u

/* DSpark's target verifier has a fixed temporal width: one accepted anchor,
 * seven draft tokens, and an eighth target-only bonus prediction.  The
 * temporal hidden layout is F32 device memory, tap-major and then temporal:
 *   target_aux_hidden[((tap * 8 + time) * 5120) + hidden]
 * `time == 0` is the input token at `token_start_position`. */
#define AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION 1u
#define AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT 5u
#define AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH 8u
#define AXIOM_QWEN38_MODEL_DSPARK_BLOCK_SIZE 7u

typedef struct axiom_qwen38_model_dspark_temporal_capabilities {
    uint32_t abi_version;
    /* This is a runtime gate.  It remains zero until
     * axiom_qwen38_model_dspark_temporal8_validate() has compared the M8
     * path with the scalar causal path on this model instance. */
    uint32_t temporal_m8_available;
    uint32_t temporal_verify_width;
    uint32_t draft_block_size;
    uint32_t target_tap_count;
    uint32_t hidden_size;
    uint32_t requires_gdn_causal_rows;
    uint32_t requires_attention_triangular_kv;
    uint32_t requires_prefix_state_install;
    uint32_t temporal_m8_implemented;
    uint32_t temporal_m8_validated;
} axiom_qwen38_model_dspark_temporal_capabilities;

typedef struct axiom_qwen38_model_dspark_temporal_taps {
    uint32_t abi_version;
    uint32_t target_layer_ids[AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT];
    uint32_t token_start_position;
    uint32_t temporal_tokens;
    uint32_t hidden_size;
    /* Borrowed CUDA F32 pointer; valid until this model starts its next
     * temporal verification, is reset, or is destroyed. */
    const float *target_aux_hidden;
} axiom_qwen38_model_dspark_temporal_taps;

typedef struct axiom_qwen38_model_dspark_verify_block8_result {
    uint32_t abi_version;
    uint32_t snapshot_position;
    uint32_t position_after_verify;
    uint32_t target_tap_token_start_position;
    uint32_t target_tap_count;
    uint32_t target_tap_tokens;
    uint32_t target_tap_hidden_size;
    const float *target_aux_hidden;
    /* Target prediction after each input [anchor,d1,...,d7]. */
    uint32_t target_token_ids[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH];
    float target_logits[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH];
    /* Number in [0,7] for which target[i] == input[i+1].  The bonus is
     * target[7], after the seven draft comparisons. */
    uint32_t accepted_draft_prefix;
    uint32_t bonus_token_id;
    float bonus_logit;
} axiom_qwen38_model_dspark_verify_block8_result;

/* Device-resident form of the target greedy reduction.  The array is owned
 * by the target model and lives in CUDA memory.  It has exactly eight rows in
 * temporal order `[anchor,d1,...,d7]`; `invalid != 0` is a fail-closed
 * non-finite-logit indication.  This layout is deliberately public so a
 * DSpark acceptance kernel can consume it without a D2H copy. */
typedef struct axiom_qwen38_model_dspark_device_top1_result {
    float value;
    uint32_t token_id;
    uint32_t invalid;
} axiom_qwen38_model_dspark_device_top1_result;

/* Asynchronous M8 result.  All pointer fields are borrowed CUDA pointers
 * valid until the next target forward/reset/destroy.  The caller must order
 * its consuming CUDA work after the supplied verification stream; this call
 * intentionally does not synchronize or inspect the device top-1 values on
 * the host. */
typedef struct axiom_qwen38_model_dspark_verify_block8_device_result {
    uint32_t abi_version;
    uint32_t snapshot_position;
    uint32_t position_after_verify;
    uint32_t target_tap_token_start_position;
    uint32_t target_tap_count;
    uint32_t target_tap_tokens;
    uint32_t target_tap_hidden_size;
    uint32_t target_top1_count;
    const float *target_aux_hidden;
    const axiom_qwen38_model_dspark_device_top1_result *target_top1_results;
} axiom_qwen38_model_dspark_verify_block8_device_result;

/* Full device-controller view. All arrays are borrowed, persistent CUDA
 * allocations owned by the target model and have stable addresses across
 * CUDA-graph replays. `target_token_ids_device` and
 * `target_logits_device` are temporal U32/F32 [8] arrays, unlike the legacy
 * packed top-1 result ABI above. */
typedef struct axiom_qwen38_model_dspark_device_target_verify_view {
    uint32_t abi_version;
    const float *target_taps_device;          /* F32 [5][8][5120], tap-major */
    const uint32_t *target_token_ids_device;  /* U32 [8] */
    const float *target_logits_device;        /* F32 [8] */
    uint32_t target_tap_count;
    uint32_t target_tap_tokens;
    uint32_t target_tap_hidden_size;
    uint32_t reserved;
} axiom_qwen38_model_dspark_device_target_verify_view;

typedef struct axiom_qwen38_model_dspark_temporal_validation_result {
    uint32_t abi_version;
    uint32_t passed;
    uint32_t tested_position;
    uint32_t tested_tokens;
    float max_logit_abs_error;
    float max_tap_abs_error;
    /* Captured target taps must already be exactly BF16-materialized values.
     * This is distinct from the downstream DSpark FC staging round. */
    float max_tap_bf16_materialization_abs_error;
} axiom_qwen38_model_dspark_temporal_validation_result;

typedef struct axiom_qwen38_model_dspark_prefill_token_result {
    uint32_t abi_version;
    /* Absolute position of the consumed prompt token. */
    uint32_t token_position;
    uint32_t target_tap_count;
    uint32_t temporal_tokens;
    uint32_t hidden_size;
    /* Borrowed CUDA F32 pointer, tap-major `[5,1,5120]`.  Valid until the
     * next prefill/temporal verification, reset, or model destruction. */
    const float *target_aux_hidden;
    uint32_t target_token_id;
    float target_logit;
} axiom_qwen38_model_dspark_prefill_token_result;

/* `max_context` applies to the sixteen full-attention KV caches. */
int axiom_qwen38_model_create(
        const char *model_path,
        int device,
        uint32_t max_context,
        axiom_qwen38_model **out);

void axiom_qwen38_model_destroy(axiom_qwen38_model *model);
int axiom_qwen38_model_reset(axiom_qwen38_model *model);
uint32_t axiom_qwen38_model_position(const axiom_qwen38_model *model);
uint64_t axiom_qwen38_model_device_bytes(const axiom_qwen38_model *model);
/* Borrowed checkpoint handle used by native auxiliary modules (for example
 * the Qwen vision tower).  The model owns it; callers must not close or
 * destroy the returned handle and must release their auxiliary module before
 * destroying the model. */
axiom_model *axiom_qwen38_model_checkpoint(axiom_qwen38_model *model);
/* Restore all mutable target state that is not represented by the paged
 * full-attention records: the GDN recurrent lanes and logical position. */
uint64_t axiom_qwen38_model_recurrent_state_bytes(void);
int axiom_qwen38_model_recurrent_state_export(
        const axiom_qwen38_model *model,
        void *host_snapshot,
        uint64_t host_snapshot_bytes);
int axiom_qwen38_model_recurrent_state_import(
        axiom_qwen38_model *model,
        const void *host_snapshot,
        uint64_t host_snapshot_bytes);
int axiom_qwen38_model_restore_position(
        axiom_qwen38_model *model,
        uint32_t position);
/* Install the exact logical token history for an already-restored or
 * device-committed model state. This changes host metadata only: the caller
 * must first synchronize CUDA work and restore the matching device position.
 * It is the explicit bridge back from device-authoritative graph decode. */
int axiom_qwen38_model_committed_history_install(
        axiom_qwen38_model *model,
        const uint32_t *token_ids,
        uint32_t token_count);
/* Synchronous page bridge for the native target's sixteen full-attention
 * layers. `attention_layer` is the compact index in [0,15], not the decoder
 * layer id. This is a storage-tier bridge; it is not used during graph replay. */
int axiom_qwen38_model_kv_page_export(
        const axiom_qwen38_model *model,
        uint32_t attention_layer,
        uint32_t logical_page,
        void *host_page,
        uint64_t host_page_bytes);
int axiom_qwen38_model_kv_page_import(
        axiom_qwen38_model *model,
        uint32_t attention_layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes);
int axiom_qwen38_model_kv_tier_bind(
        axiom_qwen38_model *model,
        axiom_qwen38_kv_tier *tier);
int axiom_qwen38_model_kv_tier_flush(
        axiom_qwen38_model *model,
        uint32_t committed_tokens);

/* Execute one token at the current position and return greedy top-1.  This is
 * a compatibility wrapper over the native eight-column execution width. */
int axiom_qwen38_model_forward_token(
        axiom_qwen38_model *model,
        uint32_t token_id,
        uint32_t *out_token_id,
        float *out_logit);

/* Execute one token and copy the first temporal column's full LM-head logits
 * to host memory.  This is the scalar sampling bridge for API requests that
 * use Qwen's non-greedy temperature/top-k/top-p defaults.  The caller must
 * provide AXIOM_QWEN38_MODEL_VOCAB floats.  The native DSpark graph remains
 * the greedy fast path; sampling is deliberately explicit at this boundary so
 * it cannot silently consume a top-1 reduction as if it were a distribution. */
int axiom_qwen38_model_forward_token_logits(
        axiom_qwen38_model *model,
        uint32_t token_id,
        float *out_logits,
        uint32_t out_logits_capacity,
        uint32_t *out_greedy_token_id,
        float *out_greedy_logit);

/* Execute one logical position from a caller-owned CUDA F32[5120] embedding
 * instead of the tokenizer embedding row. The logical token id is used only
 * for position/history validation; this is the native multimodal bridge. */
int axiom_qwen38_model_forward_embedding(
        axiom_qwen38_model *model,
        uint32_t logical_token_id,
        const float *embedding_device,
        uint32_t *out_token_id,
        float *out_logit);
int axiom_qwen38_model_forward_embedding_logits(
        axiom_qwen38_model *model,
        uint32_t logical_token_id,
        const float *embedding_device,
        float *out_logits,
        uint32_t out_logits_capacity,
        uint32_t *out_greedy_token_id,
        float *out_greedy_logit);

/* Execute eight independent token columns at the same logical position.
 * This is the native execution width of the NVFP4/FP8 kernels, not a
 * same-session temporal verifier.  Use the transaction APIs below for
 * DSpark's causal [anchor,d1,...,d7] target forward. */
int axiom_qwen38_model_forward_batch8(
        axiom_qwen38_model *model,
        const uint32_t token_ids[8],
        uint32_t out_token_ids[8],
        float out_logits[8]);

/* DSpark compute binding callbacks.  These enqueue only device work and
 * accept the compute executor's temporal columns in [1,7], column-major.
 * `user_data` is the target model itself.  They borrow model-owned pad-to-8
 * staging buffers; callers must serialize them against target forward calls.
 * They intentionally do not advance target position or touch its KV/GDN
 * state. */
int axiom_qwen38_model_dspark_target_embed_f32_device(
        void *user_data,
        const uint32_t *token_ids,
        float *out_hidden,
        uint32_t columns,
        void *stream);
int axiom_qwen38_model_dspark_target_lm_head_f32_device(
        void *user_data,
        const float *hidden,
        float *out_logits,
        uint32_t columns,
        void *stream);

/* Runtime capability and one-time parity gate for the same-session M8
 * verifier.  Validation temporarily replays the committed single-session
 * history, compares the temporal path with eight scalar causal forwards
 * (including all five taps), and restores the original committed state.
 * It is intentionally rejected after an independent forward_batch8 call,
 * because that API represents eight different sequences. */
int axiom_qwen38_model_dspark_temporal_capabilities_get(
        const axiom_qwen38_model *model,
        axiom_qwen38_model_dspark_temporal_capabilities *out);
int axiom_qwen38_model_dspark_temporal8_validate(
        axiom_qwen38_model *model,
        const uint32_t input_tokens[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH],
        float absolute_tolerance,
        axiom_qwen38_model_dspark_temporal_validation_result *out);

/* A transaction owns all lower-layer recurrent/KV snapshots.  It accepts one
 * M8 temporal forward only.  Commit retains input tokens [0,prefix_count);
 * abort restores the exact pre-verify state.  These legacy entry points use
 * the default CUDA stream and return host top-1 values for compatibility. */
int axiom_qwen38_model_transaction_begin(
        axiom_qwen38_model *model,
        axiom_qwen38_model_transaction **out);
int axiom_qwen38_model_transaction_temporal_taps_get(
        const axiom_qwen38_model_transaction *transaction,
        axiom_qwen38_model_dspark_temporal_taps *out);
/* Same transaction/rollback contract as verify_block8, but consumes exactly
 * one prompt token when the caller commits prefix 1.  Internally this uses
 * the causal M8 verifier and installs only temporal row zero, so the target
 * can be aborted if DSpark tap injection fails. */
int axiom_qwen38_model_transaction_prefill_token(
        axiom_qwen38_model_transaction *transaction,
        uint32_t token_id,
        void *stream,
        axiom_qwen38_model_dspark_prefill_token_result *out);

/* Vision-prefill bridge. `embedding_device` is one CUDA F32[5120] vector
 * produced by the native Qwen vision merger. The target consumes it at the
 * current logical position with the same transactional M8/tap path as a
 * normal token; `token_id` is retained only as the committed logical
 * placeholder for session replay. */
int axiom_qwen38_model_transaction_prefill_embedding(
        axiom_qwen38_model_transaction *transaction,
        uint32_t token_id,
        const float *embedding_device,
        void *stream,
        axiom_qwen38_model_dspark_prefill_token_result *out);
int axiom_qwen38_model_transaction_verify_block8(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t input_tokens[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH],
        void *stream,
        axiom_qwen38_model_dspark_verify_block8_result *out);
int axiom_qwen38_model_transaction_commit_prefix(
        axiom_qwen38_model_transaction *transaction,
        uint32_t input_prefix_count);
int axiom_qwen38_model_transaction_abort(
        axiom_qwen38_model_transaction *transaction);

/* Explicit-stream device ABI for the target side of DSpark.  Start, verify,
 * and commit/abort must use the same `cudaStream_t` represented as `void *`.
 * `input_tokens_host` is validated and retained for the target's optional
 * replay/parity history; `input_tokens_device` is the authoritative CUDA
 * `[8]` input consumed by the forward.  Keeping the small host mirror avoids
 * a D2H token read during commit while the large top-1 result remains wholly
 * device-resident. */
int axiom_qwen38_model_transaction_begin_stream(
        axiom_qwen38_model *model,
        void *stream,
        axiom_qwen38_model_transaction **out);
int axiom_qwen38_model_transaction_verify_block8_device(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t input_tokens_host[AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH],
        const uint32_t *input_tokens_device,
        void *stream,
        axiom_qwen38_model_dspark_verify_block8_device_result *out);
int axiom_qwen38_model_transaction_commit_prefix_stream(
        axiom_qwen38_model_transaction *transaction,
        uint32_t input_prefix_count,
        void *stream);
int axiom_qwen38_model_transaction_abort_stream(
        axiom_qwen38_model_transaction *transaction,
        void *stream);

/* Device-only target transaction for the CUDA-graph DSpark controller.
 * Every scalar pointer is persistent CUDA U32[1], all calls use one stream,
 * and no call copies device data to host. `anchor_position_device` is the
 * controller's authoritative absolute cache position. The model's legacy
 * host position/history APIs are deliberately unavailable after this mode is
 * entered, until model_reset(). */
int axiom_qwen38_model_dspark_device_transaction_begin(
        axiom_qwen38_model *model,
        const uint32_t *anchor_position_device,
        void *stream);
int axiom_qwen38_model_dspark_device_transaction_verify(
        axiom_qwen38_model *model,
        const uint32_t *verify_tokens_device,
        const uint32_t *anchor_position_device,
        uint32_t verify_width,
        void *stream,
        axiom_qwen38_model_dspark_device_target_verify_view *out);
/* Eager device path: prefix is CUDA U32[1] in [1,8], anchor included. */
int axiom_qwen38_model_dspark_device_transaction_commit(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        void *stream);
/* Graph-safe finalize: status==0 commits the device prefix; nonzero restores
 * all transactional GDN state on device. */
int axiom_qwen38_model_dspark_device_transaction_finalize(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        const uint32_t *async_status_device,
        void *stream);
int axiom_qwen38_model_dspark_device_transaction_abort(
        axiom_qwen38_model *model,
        void *stream);

/* Mark the target position/KV metadata as device-authoritative for a CUDA
 * graph replay session. Graph capture invokes the transaction callbacks only
 * once, so a resident graph executor must renew this ownership explicitly on
 * every later request before its first replay. The matching session handoff
 * materializes the device position through axiom_qwen38_model_restore_position.
 */
int axiom_qwen38_model_dspark_device_session_begin(
        axiom_qwen38_model *model);

#ifdef __cplusplus
}
#endif

#endif
