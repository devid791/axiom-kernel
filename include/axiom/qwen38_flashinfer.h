#ifndef AXIOM_QWEN38_FLASHINFER_H
#define AXIOM_QWEN38_FLASHINFER_H

/* Fixed native port of the generated FlashInfer single-prefill specialization
 * used by the Qwen3.8 target verifier:
 *
 *   Q   BF16 [8,24,256] NHD
 *   K/V E4M3 [T,4,256] NHD, unit scale
 *   O   BF16 [8,24,256] NHD
 *
 * No Python, TVM, or SGLang component is linked at runtime.  The optional
 * device length lets a captured CUDA graph reuse one launch while the target
 * transaction advances its causal cache position entirely on device. */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_FLASHINFER_TEMPORAL8_ABI_VERSION 2u

/* `kv_len_host` configures the generated launch. It must be >= 8. When
 * `kv_len_device` is non-NULL, the kernel reads its U32 value for its actual
 * causal boundary; the caller must ensure it is in [8, kv_len_host].
 * `split_kv_tmp_bf16` optionally points to FlashInfer workspace sized by the
 * caller for the dispatcher's chunk count; NULL preserves the unsplit path. */
int axiom_qwen38_flashinfer_temporal8_bf16_e4m3_device(
        const uint16_t *q_bf16,
        const uint8_t *k_e4m3,
        const uint8_t *v_e4m3,
        uint16_t *out_bf16,
        uint32_t kv_len_host,
        const uint32_t *kv_len_device,
        uint16_t *split_kv_tmp_bf16,
        void *stream);

#ifdef __cplusplus
}
#endif

#endif
