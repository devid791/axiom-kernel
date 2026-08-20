#ifndef AXIOM_QWEN38_NVFP4_H
#define AXIOM_QWEN38_NVFP4_H

/*
 * Native building blocks for the Unsloth Qwen3.8-27B-NVFP4 checkpoint.
 *
 * This is the compressed-tensors NVFP4 convention used by
 * unsloth/Qwen3.8-27B-NVFP4, not a converted RadixArk/SGLang layout:
 *
 *   weight_packed : U8       [rows, cols / 2], two E2M1 values per byte
 *   weight_scale  : F8_E4M3  [rows, cols / 16]
 *   weight_global : F32      [1], dequantization divides by this value
 *   input_scale   : U8       [cols / 16], dynamically generated F8_E4M3
 *   input_global  : F32      [1], dequantization divides by this value
 *
 * The reference entry points intentionally use scalar accumulation.  They are
 * the numerical oracle for the later Blackwell tensor-core implementation;
 * callers must not use them as the performance serving path.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantize one f32 activation row to the checkpoint's W4A4 activation form.
 * `out_packed` is cols/2 bytes and `out_scale` is cols/16 bytes.  `stream` is
 * a cudaStream_t represented as void*; NULL denotes the default stream. */
int axiom_qwen38_nvfp4_quantize_reference_f32(
        int device,
        const float *input,
        float input_global_scale,
        uint8_t *out_packed,
        uint8_t *out_scale,
        uint32_t cols,
        void *stream);

/* Dequantize-and-matvec the canonical Unsloth W4A4 representation.  Input is
 * already quantized by axiom_qwen38_nvfp4_quantize_reference_f32. */
int axiom_qwen38_nvfp4_w4a4_reference_matvec_f32(
        int device,
        const uint8_t *weight_packed,
        const uint8_t *weight_scale,
        float weight_global_scale,
        const uint8_t *input_packed,
        const uint8_t *input_scale,
        float input_global_scale,
        float *out,
        uint32_t rows,
        uint32_t cols,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
