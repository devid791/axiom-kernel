#ifndef AXIOM_QWEN38_BF16_LINEAR_H
#define AXIOM_QWEN38_BF16_LINEAR_H

/* Resident BF16 linear primitive for the small dense projections in the
 * original Qwen3.8 safetensors checkpoint, notably Gated DeltaNet
 * linear_attn.in_proj_a/b. Checkpoint weights stay in their original BF16
 * [rows, cols] representation. Device inputs are column-major F32 [cols, 8]
 * and outputs are column-major F32 [rows, 8]. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_BF16_LINEAR_BATCH 8u

typedef struct axiom_qwen38_bf16_linear axiom_qwen38_bf16_linear;

/* `weight_name` must resolve to a rank-2 BF16 safetensors tensor. Rows and
 * columns are inferred from tensor metadata and validated against byte_count
 * before any allocation or read. */
int axiom_qwen38_bf16_linear_load(
        axiom_model *model,
        int device,
        const char *weight_name,
        axiom_qwen38_bf16_linear **out);

/* Build a dense BF16 shadow from an NVIDIA ModelOpt NVFP4 linear.  This is
 * intentionally an explicit opt-in path for algorithms whose selector was
 * trained against a dense target LM head. The checkpoint remains untouched;
 * the object owns a dequantized BF16 copy on the device and uses the ModelOpt
 * E2M1/F8_E4M3/F32 scale semantics. */
int axiom_qwen38_bf16_linear_load_nvfp4_dequantized(
        axiom_model *model,
        int device,
        const char *base,
        uint32_t rows,
        uint32_t cols,
        axiom_qwen38_bf16_linear **out);

void axiom_qwen38_bf16_linear_destroy(axiom_qwen38_bf16_linear *linear);

uint32_t axiom_qwen38_bf16_linear_rows(const axiom_qwen38_bf16_linear *linear);
uint32_t axiom_qwen38_bf16_linear_cols(const axiom_qwen38_bf16_linear *linear);
uint64_t axiom_qwen38_bf16_linear_device_bytes(const axiom_qwen38_bf16_linear *linear);

/* Reference CUDA forward with F32 accumulation. `stream` is a cudaStream_t
 * represented as void*; NULL selects the default stream. The call is
 * asynchronous apart from immediate launch validation. */
int axiom_qwen38_bf16_linear_forward_f32_device(
        axiom_qwen38_bf16_linear *linear,
        const float *input,
        float *out,
        void *stream);

/* Dynamic-width form used by proposal selectors.  `columns` is [1,8], with
 * column-major F32 input [cols,columns] and output [rows,columns]. */
int axiom_qwen38_bf16_linear_forward_f32_device_m(
        axiom_qwen38_bf16_linear *linear,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
