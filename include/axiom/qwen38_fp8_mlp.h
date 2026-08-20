#ifndef AXIOM_QWEN38_FP8_MLP_H
#define AXIOM_QWEN38_FP8_MLP_H

/* Resident dense SwiGLU MLP for Qwen3.8 layers 56..63.  These eight layers
 * are stored by Unsloth as F8_E4M3 weights plus BF16 per-output-row scales,
 * unlike layers 0..55 which use packed NVFP4. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_qwen38_fp8_mlp axiom_qwen38_fp8_mlp;

int axiom_qwen38_fp8_mlp_load(
        axiom_model *model,
        int device,
        uint32_t layer,
        axiom_qwen38_fp8_mlp **out);

void axiom_qwen38_fp8_mlp_destroy(axiom_qwen38_fp8_mlp *mlp);

uint64_t axiom_qwen38_fp8_mlp_device_bytes(const axiom_qwen38_fp8_mlp *mlp);

/* Batch-8 F32 device matrices: [5120,8] -> [5120,8], column-major. */
int axiom_qwen38_fp8_mlp_forward_f32_device(
        axiom_qwen38_fp8_mlp *mlp,
        const float *input,
        float *out,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
