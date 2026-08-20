#ifndef AXIOM_QWEN38_FP8_H
#define AXIOM_QWEN38_FP8_H

/*
 * Native FP8 linear primitive for Qwen3.8-27B NVFP4 checkpoints.
 *
 * The legacy compressed-tensors representation is:
 *
 *   weight       F8_E4M3 [rows, cols]
 *   weight_scale BF16     [rows, 1]
 *
 * Each of the eight F32 input columns is dynamically quantized to E4M3 with
 * its own F32 scale.  Inputs are column-major [cols, 8] and outputs are
 * column-major [rows, 8].  The numerical contract is:
 *
 *   sx[b]    = max(abs(x[:, b])) / 448
 *   xq[:, b] = e4m3_satfinite(x[:, b] / sx[b])
 *   y[r, b]  = weight_scale[r] * sx[b]
 *              * sum_c(e4m3(weight[r, c]) * e4m3(xq[c, b]))
 *
 * A zero input column uses sx=1 and quantizes to all zeroes.  The resident
 * object owns the original E4M3 weight bytes, the original BF16 row scales,
 * and activation scratch; it does not convert or rewrite model weights.
 *
 * cuBLASLt has no direct BF16 per-output-row scale mode for this checkpoint.
 * The fast path therefore performs the raw E4M3 x E4M3 -> F32 matmul on FP8
 * tensor cores with unit scalar scales, then applies the mathematically
 * separable BF16-row x F32-column scale in one CUDA epilogue.  If cuBLASLt has
 * no heuristic for a particular shape, the same object remains usable through
 * a correct CUDA reference kernel.
 *
 * NVIDIA ModelOpt projections are also consumed directly:
 *
 *   weight       F8_E4M3 [rows, cols]
 *   weight_scale F32      []
 *   input_scale  F32      []
 *
 * ModelOpt input quantization uses its fixed checkpoint scale and cuBLASLt
 * applies both scalar multipliers without a post-GEMM row epilogue.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_FP8_BATCH 8u
#define AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX 4u

#define AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_LEGACY_ROW 1u
#define AXIOM_QWEN38_FP8_PROJECTION_GROUP_SCHEMA_MODELOPT_SCALAR 2u

typedef struct axiom_qwen38_fp8_linear axiom_qwen38_fp8_linear;
typedef struct axiom_qwen38_fp8_projection_group axiom_qwen38_fp8_projection_group;

/* One member of a load-time projection concatenation.  ModelOpt members name
 * all three tensors.  A legacy member names weight/weight_scale and leaves
 * input_scale_name NULL. */
typedef struct axiom_qwen38_fp8_projection_group_member {
    const char *weight_name;
    const char *weight_scale_name;
    const char *input_scale_name;
} axiom_qwen38_fp8_projection_group_member;

typedef struct axiom_qwen38_fp8_projection_group_config {
    uint32_t abi_version;
    uint32_t projection_count;
    axiom_qwen38_fp8_projection_group_member
            projections[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX];
    uint64_t flags; /* Must be zero. */
} axiom_qwen38_fp8_projection_group_config;

typedef struct axiom_qwen38_fp8_projection_group_info {
    uint32_t abi_version;
    uint32_t projection_count;
    uint32_t input_features;
    uint32_t total_output_features;
    /* Projection i occupies rows [row_offsets[i], row_offsets[i + 1]) in
     * every output column. */
    uint32_t row_offsets[AXIOM_QWEN38_FP8_PROJECTION_GROUP_MAX + 1u];
    uint32_t scale_schema;
    uint32_t tensor_core_enabled;
    uint32_t bf16_output_boundary;
    uint64_t device_bytes;
} axiom_qwen38_fp8_projection_group_info;

/* Load one named projection from an already-open safetensors model.  Shapes
 * and dtypes are inferred and strictly checked.  `weight_name` must name an
 * F8_E4M3 rank-2 tensor; `scale_name` must name BF16 [rows,1]. */
int axiom_qwen38_fp8_linear_load(
        axiom_model *model,
        int device,
        const char *weight_name,
        const char *scale_name,
        axiom_qwen38_fp8_linear **out);

/* Strict ModelOpt scalar-scale loader. */
int axiom_qwen38_fp8_linear_load_modelopt(
        axiom_model *model,
        int device,
        const char *weight_name,
        const char *weight_scale_name,
        const char *input_scale_name,
        axiom_qwen38_fp8_linear **out);

/* Detect the scale schema below `base` and dispatch to exactly one of the two
 * strict loaders.  `base` excludes the final .weight suffix. */
int axiom_qwen38_fp8_linear_load_auto(
        axiom_model *model,
        int device,
        const char *base,
        axiom_qwen38_fp8_linear **out);

void axiom_qwen38_fp8_linear_destroy(axiom_qwen38_fp8_linear *linear);

uint32_t axiom_qwen38_fp8_linear_rows(const axiom_qwen38_fp8_linear *linear);
uint32_t axiom_qwen38_fp8_linear_cols(const axiom_qwen38_fp8_linear *linear);
uint64_t axiom_qwen38_fp8_linear_device_bytes(const axiom_qwen38_fp8_linear *linear);
int axiom_qwen38_fp8_linear_tensor_core_enabled(const axiom_qwen38_fp8_linear *linear);

/* Launch the fixed-batch reference forward.  `input` and `out` are CUDA F32
 * pointers.  `stream` is a cudaStream_t represented as void*; NULL selects
 * the default stream.  The call is asynchronous apart from launch checks. */
int axiom_qwen38_fp8_linear_forward_f32_device(
        axiom_qwen38_fp8_linear *linear,
        const float *input,
        float *out,
        void *stream);

/* Numerical gate entry: force the CUDA reference dot product even when a
 * cuBLASLt tensor-core heuristic is available. */
int axiom_qwen38_fp8_linear_forward_reference_f32_device(
        axiom_qwen38_fp8_linear *linear,
        const float *input,
        float *out,
        void *stream);

/* Create one resident concatenated weight plane.  All members must use the
 * same scale schema and K.  ModelOpt input_scale tensors must additionally be
 * bit-identical F32 scalars.  Weight rows are concatenated in member order;
 * no member weight remains resident as a second allocation. */
int axiom_qwen38_fp8_projection_group_create(
        axiom_model *model,
        int device,
        const axiom_qwen38_fp8_projection_group_config *config,
        axiom_qwen38_fp8_projection_group **out);

/* Convenience loader.  Each base excludes .weight; the scale schema is
 * detected from .weight_scale exactly as for the single-projection loader. */
int axiom_qwen38_fp8_projection_group_load_auto(
        axiom_model *model,
        int device,
        const char *const *bases,
        uint32_t projection_count,
        axiom_qwen38_fp8_projection_group **out);

void axiom_qwen38_fp8_projection_group_destroy(
        axiom_qwen38_fp8_projection_group *group);

int axiom_qwen38_fp8_projection_group_info_get(
        const axiom_qwen38_fp8_projection_group *group,
        axiom_qwen38_fp8_projection_group_info *out);

/* One input quantization, one cuBLASLt M=8 GEMM, and one scale/round epilogue.
 * Input is CUDA F32 [input_features,8].  Output is CUDA F32
 * [total_output_features,8], but every stored value is rounded RN-even to the
 * exact BF16 boundary expected by Qwen3.8.  The call is asynchronous on
 * stream and uses no output staging or device-to-device copy. */
int axiom_qwen38_fp8_projection_group_forward_f32_device(
        axiom_qwen38_fp8_projection_group *group,
        const float *input,
        float *concatenated_out,
        void *stream);

/* Numerical gate entry with the same quantization and BF16 boundary. */
int axiom_qwen38_fp8_projection_group_forward_reference_f32_device(
        axiom_qwen38_fp8_projection_group *group,
        const float *input,
        float *concatenated_out,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
