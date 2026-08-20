#ifndef AXIOM_QWEN38_GDN_H
#define AXIOM_QWEN38_GDN_H

/* Native Qwen3.8 Gated DeltaNet mixer.
 *
 * This owns one non-attention Qwen3.8 layer's original checkpoint weights and
 * its eight independent recurrent states. The qkv and z input projections are
 * resident as one exact FP8 projection group and execute as one M8 cuBLASLt
 * GEMM. Inputs and outputs are CUDA F32 column-major matrices [5120, 8]. The
 * result is the linear-attention mixer output before its residual connection.
 *
 * The scalar compatibility entry point remains default-stream-only. The
 * temporal verifier is fully stream ordered: begin, forward, and commit/abort
 * must receive the same CUDA stream (or all NULL for the default stream).
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_GDN_HIDDEN 5120u
#define AXIOM_QWEN38_GDN_BATCH 8u
#define AXIOM_QWEN38_GDN_CONV_DIM 10240u
#define AXIOM_QWEN38_GDN_KEY_HEADS 16u
#define AXIOM_QWEN38_GDN_VALUE_HEADS 48u
#define AXIOM_QWEN38_GDN_HEAD_DIM 128u
#define AXIOM_QWEN38_GDN_RING_BYTES (AXIOM_QWEN38_GDN_CONV_DIM * 3u * sizeof(float))
#define AXIOM_QWEN38_GDN_STATE_BYTES \
    (AXIOM_QWEN38_GDN_VALUE_HEADS * AXIOM_QWEN38_GDN_HEAD_DIM * \
     AXIOM_QWEN38_GDN_HEAD_DIM * sizeof(float))
#define AXIOM_QWEN38_GDN_SNAPSHOT_BYTES \
    (AXIOM_QWEN38_GDN_RING_BYTES + AXIOM_QWEN38_GDN_STATE_BYTES)

typedef struct axiom_qwen38_gdn_layer axiom_qwen38_gdn_layer;

/* Loads one Gated DeltaNet layer. Full-attention layers (3, 7, ... 63) are
 * rejected. `runtime` must be a CUDA Axiom runtime on `device`. */
int axiom_qwen38_gdn_layer_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        uint32_t layer,
        axiom_qwen38_gdn_layer **out);

void axiom_qwen38_gdn_layer_destroy(axiom_qwen38_gdn_layer *layer);

/* Clear only the recurrent convolution and DeltaNet state; resident weights
 * and activation scratch remain allocated. */
int axiom_qwen38_gdn_layer_reset(axiom_qwen38_gdn_layer *layer);

uint64_t axiom_qwen38_gdn_layer_device_bytes(const axiom_qwen38_gdn_layer *layer);

/* Export/import the committed lane-zero recurrent state. The snapshot is
 * intentionally independent of the speculative backup lanes: it is the
 * durable state needed to resume a prompt prefix after a model reset. */
int axiom_qwen38_gdn_layer_state_export(
        const axiom_qwen38_gdn_layer *layer,
        void *host_snapshot,
        uint64_t host_snapshot_bytes);
int axiom_qwen38_gdn_layer_state_import(
        axiom_qwen38_gdn_layer *layer,
        const void *host_snapshot,
        uint64_t host_snapshot_bytes);

/* Run all eight independent streams. `stream` must be NULL in this version. */
int axiom_qwen38_gdn_layer_forward_f32_device(
        axiom_qwen38_gdn_layer *layer,
        const float *input,
        float *out,
        void *stream);

/* Same-session temporal verification primitive for DSpark. The eight input
 * columns are consecutive token positions, not independent sequences. A
 * transaction must be opened first. Each recurrent prefix state is retained
 * in one of the existing eight state lanes so commit_prefix can select the
 * exact accepted prefix without replay. `stream` is accepted and must remain
 * identical across the transaction. */
int axiom_qwen38_gdn_layer_spec_begin(
        axiom_qwen38_gdn_layer *layer,
        void *stream);
int axiom_qwen38_gdn_layer_forward_temporal8_f32_device(
        axiom_qwen38_gdn_layer *layer,
        const float *input,
        float *out,
        void *stream);
int axiom_qwen38_gdn_layer_spec_commit_prefix(
        axiom_qwen38_gdn_layer *layer,
        uint32_t consumed_tokens,
        void *stream);
/* Device-only commit form. `consumed_tokens_device` is CUDA U32[1] in
 * [1,8] and includes the anchor row. It selects the already-computed causal
 * state lane without a host copy or stream synchronization. */
int axiom_qwen38_gdn_layer_spec_commit_prefix_device(
        axiom_qwen38_gdn_layer *layer,
        const uint32_t *consumed_tokens_device,
        void *stream);
/* Graph-safe device finalize. If `async_status_device[0] != 0`, restore the
 * transactional backup; otherwise install `consumed_tokens_device[0]` in
 * [1,8]. Both inputs are CUDA U32[1] and no host branch or D2H read occurs. */
int axiom_qwen38_gdn_layer_spec_finalize_device(
        axiom_qwen38_gdn_layer *layer,
        const uint32_t *consumed_tokens_device,
        const uint32_t *async_status_device,
        void *stream);
int axiom_qwen38_gdn_layer_spec_abort(
        axiom_qwen38_gdn_layer *layer,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
