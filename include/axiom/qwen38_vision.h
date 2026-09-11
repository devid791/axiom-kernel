#ifndef AXIOM_QWEN38_VISION_H
#define AXIOM_QWEN38_VISION_H

/* Native Qwen3.8/Qwen3.5 vision tower for the installed 27B checkpoint.
 *
 * The public boundary consumes the processor's already patchified layout:
 * [channel, temporal_patch, patch_y, patch_x] flattened per patch as
 * C * temporal_patch_size * patch_size * patch_size.  Keeping this boundary
 * explicit lets the image/video decoders stay independent from the CUDA
 * tower, while making the tensor order auditable against the official
 * Qwen3-VL processor.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_VISION_ABI_VERSION 1u
#define AXIOM_QWEN38_VISION_DEPTH 27u /* checkpoint vision_config.depth */
#define AXIOM_QWEN38_VISION_HIDDEN 1152u
#define AXIOM_QWEN38_VISION_INTERMEDIATE 4304u
#define AXIOM_QWEN38_VISION_HEADS 16u
#define AXIOM_QWEN38_VISION_HEAD_DIM 72u
#define AXIOM_QWEN38_VISION_PATCH_SIZE 16u
#define AXIOM_QWEN38_VISION_TEMPORAL_PATCH 2u
#define AXIOM_QWEN38_VISION_MERGE_SIZE 2u
#define AXIOM_QWEN38_VISION_POSITION_SIDE 48u
#define AXIOM_QWEN38_VISION_MERGED_HIDDEN 4608u
#define AXIOM_QWEN38_VISION_OUTPUT 5120u
#define AXIOM_QWEN38_VISION_PATCH_FEATURES 1536u

typedef struct axiom_qwen38_vision axiom_qwen38_vision;

typedef struct axiom_qwen38_vision_grid {
    uint32_t temporal;
    uint32_t height;
    uint32_t width;
} axiom_qwen38_vision_grid;

typedef struct axiom_qwen38_vision_info {
    uint32_t abi_version;
    uint32_t depth;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t patch_size;
    uint32_t temporal_patch_size;
    uint32_t spatial_merge_size;
    uint32_t position_grid_side;
    uint32_t patch_feature_count;
    uint32_t output_size;
    uint32_t max_tokens;
    uint64_t device_bytes;
    uint64_t loaded_tensor_count;
    uint64_t loaded_tensor_bytes;
} axiom_qwen38_vision_info;

/* Loads and validates all model.visual.* tensors. max_tokens is the reusable
 * scratch capacity, not an input limit. Larger forwards allocate temporary
 * scratch against actual free GPU memory and restore the resident workspace
 * on success or failure. No text weights are loaded by this object. */
int axiom_qwen38_vision_create(
        axiom_model *model,
        int device,
        uint32_t max_tokens,
        axiom_qwen38_vision **out);

void axiom_qwen38_vision_destroy(axiom_qwen38_vision *vision);
int axiom_qwen38_vision_info_get(
        const axiom_qwen38_vision *vision,
        axiom_qwen38_vision_info *out);

/* Runs the complete patch embed -> 27-block vision tower -> 2x2 merger.
 * `patch_values` is host F32 [sum(grid.t*grid.h*grid.w),1536] in the exact
 * Qwen3-VL processor order. `out_host` receives BF16-materialized F32 values
 * [sum(grid.t*(grid.h/2)*(grid.w/2)),5120]. */
int axiom_qwen38_vision_forward_patches(
        axiom_qwen38_vision *vision,
        const float *patch_values,
        uint64_t patch_value_count,
        const axiom_qwen38_vision_grid *grids,
        uint32_t grid_count,
        float *out_host,
        uint64_t out_value_capacity,
        uint32_t *out_tokens);

/* Device-direct sibling used by the multimodal target prefill. Input and
 * output are CUDA F32 buffers; output is BF16-materialized F32. */
int axiom_qwen38_vision_forward_patches_device(
        axiom_qwen38_vision *vision,
        const float *patch_values_device,
        uint64_t patch_value_count,
        const axiom_qwen38_vision_grid *grids,
        uint32_t grid_count,
        float *out_device,
        uint64_t out_value_capacity,
        uint32_t *out_tokens);

#ifdef __cplusplus
}
#endif

#endif
