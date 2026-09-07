#ifndef AXIOM_QWEN38_ATTENTION_H
#define AXIOM_QWEN38_ATTENTION_H

/* Native Qwen3.8 full-attention mixer. Inputs and outputs are CUDA F32
 * column-major [5120,8]. A layer owns eight independent E4M3FN K/V caches
 * with a shared monotonic position. RadixArk's D256 target uses fixed
 * E4M3FN cache storage with scale=descale=1.0; no inferred checkpoint scale
 * is ever introduced. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_ATTENTION_HIDDEN 5120u
#define AXIOM_QWEN38_ATTENTION_BATCH 8u
#define AXIOM_QWEN38_ATTENTION_HEADS 24u
#define AXIOM_QWEN38_ATTENTION_KV_HEADS 4u
#define AXIOM_QWEN38_ATTENTION_HEAD_DIM 256u
#define AXIOM_QWEN38_ATTENTION_ROPE_DIM 64u
#define AXIOM_QWEN38_TARGET_ROPE_THETA 10000000.0f
#define AXIOM_QWEN38_TARGET_YARN_FACTOR 4.0f
#define AXIOM_QWEN38_TARGET_YARN_ORIGINAL_CONTEXT 262144.0f
#define AXIOM_QWEN38_TARGET_YARN_BETA_FAST 32.0f
#define AXIOM_QWEN38_TARGET_YARN_BETA_SLOW 1.0f
#define AXIOM_QWEN38_TARGET_YARN_MSCALE 1.138629436111989f
#define AXIOM_QWEN38_ATTENTION_KV_FP8_PARITY_ABI_VERSION 1u
#define AXIOM_QWEN38_ATTENTION_KV_PAGE_TOKENS 256u
#define AXIOM_QWEN38_ATTENTION_KV_PAGE_BYTES 524288u

/* RoPE geometry is immutable for the lifetime of one loaded target.  A
 * context budget never changes this profile: native 262K and YaRN4 1M must
 * use separate model/cache namespaces. */
typedef enum axiom_qwen38_rope_profile {
    AXIOM_QWEN38_ROPE_PROFILE_INVALID = 0,
    AXIOM_QWEN38_ROPE_PROFILE_NATIVE_262K = 1,
    AXIOM_QWEN38_ROPE_PROFILE_YARN4_1M = 2,
} axiom_qwen38_rope_profile;

typedef struct axiom_qwen38_attention_layer axiom_qwen38_attention_layer;
typedef struct axiom_qwen38_kv_tier axiom_qwen38_kv_tier;

/* Optional short-context numerical gate for fixed-scale E4M3FN K/V. Enable
 * it before the first forward. It compares both cache values and attention
 * output against the former F32 storage of the same BF16-materialized values.
 * The explicit get call may synchronize; normal decode and CUDA-graph replay
 * never do. */
typedef struct axiom_qwen38_attention_kv_fp8_parity {
    uint32_t abi_version;
    uint32_t enabled;
    uint32_t sampled_values;
    uint32_t nonfinite_values;
    float max_cache_abs_error;
    float max_attention_abs_error;
    float scale;
    float descale;
} axiom_qwen38_attention_kv_fp8_parity;

/* Loads a full-attention layer (3, 7, ..., 63). `max_context` must be nonzero;
 * the initial implementation advances all eight columns together. */
int axiom_qwen38_attention_layer_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        uint32_t layer,
        uint32_t max_context,
        axiom_qwen38_attention_layer **out);

void axiom_qwen38_attention_layer_destroy(axiom_qwen38_attention_layer *layer);
int axiom_qwen38_attention_layer_reset(axiom_qwen38_attention_layer *layer);
uint32_t axiom_qwen38_attention_layer_position(const axiom_qwen38_attention_layer *layer);
uint64_t axiom_qwen38_attention_layer_device_bytes(const axiom_qwen38_attention_layer *layer);
axiom_qwen38_rope_profile axiom_qwen38_attention_layer_rope_profile(
        const axiom_qwen38_attention_layer *layer);
/* Restore a position after page import. For the paged provider this also
 * rebuilds the logical-to-hot-page map and leaves the current page clean. */
int axiom_qwen38_attention_layer_restore_position(
        axiom_qwen38_attention_layer *layer,
        uint32_t position);
/* Materialize a device-authoritative temporal-hot prefix into the paged HBM
 * view and install its host position. CUDA graph decode writes the compact
 * contiguous cache; canonical decode and NVMe persistence consume pages. */
int axiom_qwen38_attention_layer_materialize_device_position(
        axiom_qwen38_attention_layer *layer,
        uint32_t position);

/* Export/import one logical temporal page from lane zero. The host layout is
 * E4M3 K[256,4,256] followed by E4M3 V[256,4,256], exactly one tier record.
 * These calls are synchronous and intentionally outside CUDA-graph replay. */
int axiom_qwen38_attention_layer_kv_page_export(
        const axiom_qwen38_attention_layer *layer,
        uint32_t logical_page,
        void *host_page,
        uint64_t host_page_bytes);
int axiom_qwen38_attention_layer_kv_page_import(
        axiom_qwen38_attention_layer *layer,
        uint32_t logical_page,
        const void *host_page,
        uint64_t host_page_bytes);

/* Qualification-only synchronous export of the four internal attention
 * boundaries. Host buffers use column-major [column_count,24*256] F32. This
 * API is intentionally outside graph replay and permits an active host
 * temporal transaction only when it uses the default stream. */
int axiom_qwen38_attention_layer_validation_export(
        const axiom_qwen38_attention_layer *layer,
        uint32_t first_column,
        uint32_t column_count,
        float *host_q,
        float *host_gate,
        float *host_attention,
        float *host_gated_attention,
        uint64_t host_elements);

/* Qualification-only byte-exact export of the paged provider and its
 * contiguous temporal-hot mirror for one logical page.  A non-streaming
 * layer has only one canonical resident view, which is returned in both
 * outputs so the common temporal validator remains usable offline. */
int axiom_qwen38_attention_layer_validation_kv_views_export(
        const axiom_qwen38_attention_layer *layer,
        uint32_t logical_page,
        void *host_paged,
        void *host_temporal_hot,
        uint64_t host_page_bytes);
/* Bind the target layer to the durable tier. With AXIOM_QWEN38_KV_STREAMING=1
 * this enables the real page provider: current-page K/V stays on device,
 * completed pages are flushed and cold pages are read before online attention. */
int axiom_qwen38_attention_layer_kv_tier_bind(
        axiom_qwen38_attention_layer *layer,
        axiom_qwen38_kv_tier *tier,
        uint32_t tier_layer);
int axiom_qwen38_attention_layer_kv_tier_flush(
        axiom_qwen38_attention_layer *layer);
int axiom_qwen38_attention_layer_kv_fp8_parity_enable(
        axiom_qwen38_attention_layer *layer);
int axiom_qwen38_attention_layer_kv_fp8_parity_get(
        const axiom_qwen38_attention_layer *layer,
        axiom_qwen38_attention_kv_fp8_parity *out);

/* Capture-time selector for the device temporal backend. It is rejected
 * while a speculative transaction is active; already instantiated CUDA
 * graphs retain the kernels captured under the selected value. */
int axiom_qwen38_attention_layer_device_temporal_exact_set(
        axiom_qwen38_attention_layer *layer,
        int enabled);
int axiom_qwen38_attention_layer_device_temporal_exact_can_set(
        const axiom_qwen38_attention_layer *layer,
        int enabled);

/* Scalar compatibility path; `stream` must be NULL. */
int axiom_qwen38_attention_layer_forward_f32_device(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        void *stream);

/* Same-session temporal DSpark verifier. Columns are consecutive positions
 * and share cache lane zero. Uncommitted suffix KV is harmless because abort
 * restores the logical position and the next transaction overwrites it. An
 * explicit CUDA stream is accepted, but every call in a transaction must use
 * the identical stream. */
int axiom_qwen38_attention_layer_spec_begin(
        axiom_qwen38_attention_layer *layer,
        void *stream);
int axiom_qwen38_attention_layer_forward_temporal8_f32_device(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        void *stream);
int axiom_qwen38_attention_layer_spec_commit_prefix(
        axiom_qwen38_attention_layer *layer,
        uint32_t consumed_tokens,
        void *stream);
int axiom_qwen38_attention_layer_spec_abort(
        axiom_qwen38_attention_layer *layer,
        void *stream);

/* Graph/device-controller variant. `base_position_device` is a persistent
 * CUDA U32[1] owned by the controller. The temporal kernels read it on the
 * supplied stream, so no host position update or D2H read is needed between
 * speculative steps. `consumed_tokens_device` is CUDA U32[1] in [1,8]. */
int axiom_qwen38_attention_layer_spec_begin_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *base_position_device,
        void *stream);
int axiom_qwen38_attention_layer_forward_temporal8_f32_device_position(
        axiom_qwen38_attention_layer *layer,
        const float *input,
        float *out,
        const uint32_t *base_position_device,
        void *stream);
int axiom_qwen38_attention_layer_spec_commit_prefix_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *consumed_tokens_device,
        void *stream);
int axiom_qwen38_attention_layer_spec_finalize_device(
        axiom_qwen38_attention_layer *layer,
        const uint32_t *consumed_tokens_device,
        const uint32_t *async_status_device,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
