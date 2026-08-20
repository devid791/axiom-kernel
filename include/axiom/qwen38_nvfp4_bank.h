#ifndef AXIOM_QWEN38_NVFP4_BANK_H
#define AXIOM_QWEN38_NVFP4_BANK_H

/* Axiom-owned resident NVFP4 MLP bank.  It contains 56 layers for the
 * compressed-tensors checkpoint and 64 layers for the RadixArk ModelOpt
 * checkpoint, selected from the actual tail-layer tensor schema. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_qwen38_nvfp4_bank axiom_qwen38_nvfp4_bank;

int axiom_qwen38_nvfp4_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_nvfp4_bank **out);

void axiom_qwen38_nvfp4_bank_destroy(axiom_qwen38_nvfp4_bank *bank);

uint64_t axiom_qwen38_nvfp4_bank_device_bytes(const axiom_qwen38_nvfp4_bank *bank);

uint32_t axiom_qwen38_nvfp4_bank_layer_count(const axiom_qwen38_nvfp4_bank *bank);

/* `layer` is [0,layer_count).  The input/output ABI is the batch-8 device F32 ABI of
 * axiom_qwen38_nvfp4_mlp_forward_f32_device. */
int axiom_qwen38_nvfp4_bank_forward_f32_device(
        axiom_qwen38_nvfp4_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
