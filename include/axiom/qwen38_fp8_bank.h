#ifndef AXIOM_QWEN38_FP8_BANK_H
#define AXIOM_QWEN38_FP8_BANK_H

/* Axiom-owned resident bank for the eight dense FP8 MLP layers in the
 * original unsloth/Qwen3.8-27B-NVFP4 checkpoint.  Layers 56..63 use
 * F8_E4M3 weights plus BF16 per-output-row scales; layers 0..55 belong to
 * the separate NVFP4 bank. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_FP8_BANK_FIRST_LAYER 56u
#define AXIOM_QWEN38_FP8_BANK_LAYER_COUNT 8u
#define AXIOM_QWEN38_FP8_BANK_LAST_LAYER \
        (AXIOM_QWEN38_FP8_BANK_FIRST_LAYER + AXIOM_QWEN38_FP8_BANK_LAYER_COUNT - 1u)

typedef struct axiom_qwen38_fp8_bank axiom_qwen38_fp8_bank;

/* Load and retain all three MLP projections for layers 56..63.  A partial
 * load is destroyed atomically before an error is returned. */
int axiom_qwen38_fp8_bank_load(
        axiom_model *model,
        int device,
        axiom_qwen38_fp8_bank **out);

void axiom_qwen38_fp8_bank_destroy(axiom_qwen38_fp8_bank *bank);

uint64_t axiom_qwen38_fp8_bank_device_bytes(const axiom_qwen38_fp8_bank *bank);

/* `layer` is an absolute decoder-layer index in [56,63].  Input and output
 * use the batch-8, column-major device-F32 ABI of
 * axiom_qwen38_fp8_mlp_forward_f32_device. */
int axiom_qwen38_fp8_bank_forward_f32_device(
        axiom_qwen38_fp8_bank *bank,
        uint32_t layer,
        const float *input,
        float *out,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
