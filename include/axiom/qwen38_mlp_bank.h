#ifndef AXIOM_QWEN38_MLP_BANK_H
#define AXIOM_QWEN38_MLP_BANK_H

/* Complete resident MLP dispatch for original Unsloth Qwen3.8-27B-NVFP4.
 * Layers 0..55 execute native compressed-tensors NVFP4; layers 56..63
 * execute native F8_E4M3 plus BF16-row-scale.  The split is checkpoint
 * metadata, not a conversion choice. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_MLP_LAYER_COUNT 64u

typedef struct axiom_qwen38_mlp_bank axiom_qwen38_mlp_bank;

int axiom_qwen38_mlp_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_mlp_bank **out);

void axiom_qwen38_mlp_bank_destroy(axiom_qwen38_mlp_bank *bank);

uint64_t axiom_qwen38_mlp_bank_device_bytes(const axiom_qwen38_mlp_bank *bank);

/* `layer` is [0,63].  Input and output are batch-8, column-major F32 CUDA
 * matrices [5120,8]. */
int axiom_qwen38_mlp_bank_forward_f32_device(
        axiom_qwen38_mlp_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
