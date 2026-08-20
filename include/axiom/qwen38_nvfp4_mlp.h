#ifndef AXIOM_QWEN38_NVFP4_MLP_H
#define AXIOM_QWEN38_NVFP4_MLP_H

/*
 * Resident Axiom execution for one Qwen3.8 NVFP4 MLP layer.
 *
 * Both native checkpoint representations are accepted without conversion:
 *
 *   compressed-tensors: weight_packed, weight_scale,
 *                       input_global_scale, weight_global_scale
 *   NVIDIA ModelOpt:    weight, weight_scale, input_scale, weight_scale_2
 *
 * Packed E2M1 values stay in their original nibble order and `weight_scale`
 * stays F8_E4M3.  At load time
 * Axiom creates the compact 128x4 scale *view* required by Blackwell's FP4
 * tensor cores; it never converts, requantizes, or replaces the weights.
 *
 * The component is deliberately an opaque Axiom-owned resident object.  It
 * is the serving primitive used by the Qwen3.8 runner, not an Unsloth or
 * SGLang runtime.  Inputs and outputs are CUDA F32 matrices in column-major
 * [features, batch] order.  The serving hot path accepts one through eight
 * columns without padding the matrix M dimension.  The legacy entry points
 * retain their fixed width of eight for ABI compatibility.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_NVFP4_HIDDEN 5120u
#define AXIOM_QWEN38_NVFP4_FFN 17408u
#define AXIOM_QWEN38_NVFP4_TC_BATCH 8u
#define AXIOM_QWEN38_NVFP4_TC_LAYERS 56u
#define AXIOM_QWEN38_NVFP4_MODEL_LAYERS 64u

typedef struct axiom_qwen38_nvfp4_mlp axiom_qwen38_nvfp4_mlp;
typedef struct axiom_qwen38_nvfp4_linear axiom_qwen38_nvfp4_linear;

/* Load and upload the exact gate/up/down tensors for one NVFP4 MLP layer.
 * `layer` must be in [0,63] and the selected layer must expose a supported
 * NVFP4 tensor family.  The object owns all allocations and may be
 * shared only after the caller serializes forward calls on its CUDA stream. */
int axiom_qwen38_nvfp4_mlp_load(
        axiom_model *model,
        int device,
        uint32_t layer,
        axiom_qwen38_nvfp4_mlp **out);

void axiom_qwen38_nvfp4_mlp_destroy(axiom_qwen38_nvfp4_mlp *mlp);

/* Bytes owned in device memory: raw Unsloth weights, tensor-core scale views,
 * activation workspace, and cuBLASLt workspace. */
uint64_t axiom_qwen38_nvfp4_mlp_device_bytes(const axiom_qwen38_nvfp4_mlp *mlp);

/* One when the resident object merged gate/up exactly; zero means the
 * checkpoint-scale-safe two-projection fallback is active. */
uint32_t axiom_qwen38_nvfp4_mlp_uses_fused_gate_up(
        const axiom_qwen38_nvfp4_mlp *mlp);

/*
 * Execute out = down(silu(gate(x)) * up(x)) for eight columns.  `input` is
 * [5120,8] and `out` is [5120,8], column-major F32 device memory.  `stream`
 * is a cudaStream_t passed as void*; NULL means default stream.  The call is
 * asynchronous except for immediate CUDA launch validation.
 */
int axiom_qwen38_nvfp4_mlp_forward_f32_device(
        axiom_qwen38_nvfp4_mlp *mlp,
        const float *input,
        float *out,
        void *stream);

/* Dynamic-width form of the resident MLP forward.  `columns` is [1,8]; input
 * is [5120,columns] and output is [5120,columns].  Gate and up are executed as
 * one resident packed NVFP4 projection when their checkpoint-global scales
 * permit an exact merge.  SiLU*up and the down-projection activation
 * quantization are fused.  The call allocates nothing and is asynchronous. */
int axiom_qwen38_nvfp4_mlp_forward_f32_device_m(
        axiom_qwen38_nvfp4_mlp *mlp,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream);

/* Generic ModelOpt/legacy NVFP4 linear used by the RadixArk lm_head.  `base`
 * is the tensor prefix (for example "lm_head"); the loader consumes the same
 * four-tensor schema documented above and keeps the checkpoint bytes intact. */
int axiom_qwen38_nvfp4_linear_load(
        axiom_model *model,
        int device,
        const char *base,
        uint32_t rows,
        uint32_t cols,
        axiom_qwen38_nvfp4_linear **out);

void axiom_qwen38_nvfp4_linear_destroy(axiom_qwen38_nvfp4_linear *linear);

uint32_t axiom_qwen38_nvfp4_linear_rows(const axiom_qwen38_nvfp4_linear *linear);
uint32_t axiom_qwen38_nvfp4_linear_cols(const axiom_qwen38_nvfp4_linear *linear);
uint64_t axiom_qwen38_nvfp4_linear_device_bytes(const axiom_qwen38_nvfp4_linear *linear);

int axiom_qwen38_nvfp4_linear_forward_f32_device(
        axiom_qwen38_nvfp4_linear *linear,
        const float *input,
        float *out,
        void *stream);

/* Dynamic-M, device-direct NVFP4 linear.  `columns` is [1,8].  `input` and
 * `out` are caller-owned CUDA F32 column-major matrices [cols,columns] and
 * [rows,columns].  The result is written directly to `out`: there is no
 * internal output staging or device-to-device copy.  All descriptors and
 * algorithms are selected at load time; the call is allocation-free and
 * asynchronous on `stream`.  Calls sharing one opaque object must be ordered
 * on the same stream or explicitly synchronized by the caller. */
int axiom_qwen38_nvfp4_linear_forward_f32_device_m(
        axiom_qwen38_nvfp4_linear *linear,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
