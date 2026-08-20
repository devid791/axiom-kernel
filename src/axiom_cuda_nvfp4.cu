/* axiom_cuda_nvfp4.cu — native NVFP4 expert matvec for DeepSeek V4 Flash on Blackwell.
 *
 * Weight-only dequant GEMV for the routed-expert GEMMs, matching the on-disk NVFP4
 * layout of nvidia/DeepSeek-V4-Flash-NVFP4:
 *   weight        : U8       [rows, cols/2]   2x signed-E2M1 nibbles/byte (low nibble = even col)
 *   block_scale   : F8_E4M3  [rows, cols/16]  per-group-16 block scale (as uint8 e4m3 codes)
 *   global_scale  : f32                        per-tensor scale (weight_scale_2)
 * value(r,c) = e2m1(nibble) * e4m3(block_scale[r, c/16]) * global_scale
 *
 * Compiled into libaxiom. Validated bit-exact vs a host oracle by
 * tests/axiom_nvfp4_smoke.cpp on the GB10. Kernel-math-proven earlier via the
 * standalone tools/axiom_e2m1_probe.cu.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include "axiom/axiom.h"

#define AXIOM_NVFP4_GROUP 16u

__device__ static inline float axnv_e2m1(uint8_t nib) {
    float v;
    switch (nib & 7u) {
        case 0: v = 0.0f; break; case 1: v = 0.5f; break; case 2: v = 1.0f; break; case 3: v = 1.5f; break;
        case 4: v = 2.0f; break; case 5: v = 3.0f; break; case 6: v = 4.0f; break; default: v = 6.0f; break;
    }
    return (nib & 8u) ? -v : v;
}
/* OCP FP8 e4m3 (e4m3fn): s eeee mmm, exp bias 7, no inf. */
__device__ static inline float axnv_e4m3(uint8_t b) {
    uint32_t s = (b >> 7) & 1u, e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v = (e == 0u) ? (float)m * (0.015625f / 8.0f)
                        : (1.0f + (float)m / 8.0f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}

__global__ static void axnv_matvec_kernel(const uint8_t *w, const uint8_t *bscale, float global,
                                          const float *x, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint32_t groups = cols / AXIOM_NVFP4_GROUP;
    float acc = 0.0f;
    for (uint32_t k = 0; k < cols; ++k) {
        uint64_t idx = (uint64_t)r * cols + k;
        uint8_t byte = w[idx >> 1];
        uint8_t nib = (idx & 1u) ? (byte >> 4) : (byte & 0x0f);
        float bs = axnv_e4m3(bscale[(uint64_t)r * groups + (k / AXIOM_NVFP4_GROUP)]);
        acc += axnv_e2m1(nib) * bs * global * x[k];
    }
    out[r] = acc;
}

/* Raw-device-pointer entry (pointers already offset by caller; device is the CUDA
 * device index). Returns AXIOM_OK / AXIOM_ERR_*. A buffer-handle adapter that bridges
 * axiom_device_buffer -> raw ptr will wrap this at the ds4 expert-GEMM call sites. */
extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (cols % AXIOM_NVFP4_GROUP) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u);
    axnv_matvec_kernel<<<grd, blk>>>(weight, block_scale, global_scale, input, out, rows, cols);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cudaDeviceSynchronize() != cudaSuccess) return AXIOM_ERR_CUDA;
    return AXIOM_OK;
}

/* ---- FP8 path: attention + shared experts. weight F8_E4M3, scale F8_E8M0 on 128x128
 * blocks (weight_block_size=[128,128]). value(r,c) = e4m3(w) * 2^(e8m0(scale[r/128,c/128])-127).
 * DeepSeek-V4-Flash keeps attn/shared in FP8 (only routed experts are NVFP4). ---- */
__device__ static inline float axnv_e8m0(uint8_t b) {
    /* unsigned 8-bit exponent (bias 127): value = 2^(b-127). 0xff = NaN per spec; scales aren't NaN. */
    return exp2f((float)((int)b - 127));
}
__global__ static void axfp8_matvec_kernel(const uint8_t *w, const uint8_t *scale, const float *x,
                                           float *out, uint32_t rows, uint32_t cols, uint32_t sc_cols) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float acc = 0.0f;
    uint32_t sr = r >> 7;   /* row / 128 */
    for (uint32_t c = 0; c < cols; ++c) {
        float wv = axnv_e4m3(w[(uint64_t)r * cols + c]);
        float s = axnv_e8m0(scale[(uint64_t)sr * sc_cols + (c >> 7)]);
        acc += wv * s * x[c];
    }
    out[r] = acc;
}
extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (rows % 128u) != 0u || (cols % 128u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u);
    axfp8_matvec_kernel<<<grd, blk>>>(weight, block_scale, input, out, rows, cols, cols / 128u);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cudaDeviceSynchronize() != cudaSuccess) return AXIOM_ERR_CUDA;
    return AXIOM_OK;
}

/* ---- axpby: out[i] = beta*out[i] + alpha*in[i]. Used to accumulate the topk routed
 * expert outputs into the MoE result (beta=0 on the first expert overwrites; beta=1
 * accumulates the rest), scaling each by its router weight (alpha). ---- */
__global__ static void axnv_axpby_kernel(float *out, const float *in, float alpha, float beta, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = beta * out[i] + alpha * in[i];
}
extern "C" int axiom_cuda_axpby_f32(int device, float *out, const float *in,
                                    float alpha, float beta, uint32_t n) {
    if (!out || !in || n == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(256), grd((n + 255u) / 256u);
    axnv_axpby_kernel<<<grd, blk>>>(out, in, alpha, beta, n);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cudaDeviceSynchronize() != cudaSuccess) return AXIOM_ERR_CUDA;
    return AXIOM_OK;
}

/* ==================== stream-aware _async variants (ADDITIVE) ====================
 * Audit item: every entry above ends in an unconditional cudaDeviceSynchronize, which
 * costs ~1600 device-wide syncs per token when the host MoE loop drives them. The
 * _async variants below launch the SAME kernels with the SAME validation on a
 * caller-provided stream (NULL = default stream; per-thread under
 * --default-stream per-thread) and follow the axiom_cuda_finish_after_launch
 * convention (src/axiom_cuda.cu:197-201): AXIOM_CUDA_LAUNCH_ASYNC enabled or stream
 * capturing -> return right after launch; otherwise a STREAM-scoped synchronize (so
 * the default behavior stays synchronous-correct, without serializing other streams).
 * The canonical helper is `static` in axiom_cuda.cu, so the 3-line logic is replicated
 * here — with the getenv CACHED at first use (the audit flagged the per-call getenv;
 * truthiness matches axiom_cuda_env_enabled, src/axiom_cuda.cu:102-107; flipping the
 * variable mid-process is not observed). The sync entries above are untouched
 * (bit-compat: same kernels, same launch geometry, same default stream). */
#include <cstdlib>
#include <cstring>

static bool axnv_launch_async_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = std::getenv("AXIOM_CUDA_LAUNCH_ASYNC");
        cached = (v && v[0] != '\0' && std::strcmp(v, "0") != 0 &&
                  std::strcmp(v, "false") != 0 && std::strcmp(v, "False") != 0 &&
                  std::strcmp(v, "FALSE") != 0) ? 1 : 0;
    }
    return cached != 0;
}
static bool axnv_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
}
static int axnv_finish_after_launch(cudaStream_t stream) {
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axnv_launch_async_enabled() || axnv_stream_capturing(stream)) return AXIOM_OK;
    return cudaStreamSynchronize(stream) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32_async(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols, void *stream) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (cols % AXIOM_NVFP4_GROUP) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u);
    axnv_matvec_kernel<<<grd, blk, 0, (cudaStream_t)stream>>>(
            weight, block_scale, global_scale, input, out, rows, cols);
    return axnv_finish_after_launch((cudaStream_t)stream);
}

extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32_async(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols, void *stream) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (rows % 128u) != 0u || (cols % 128u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u);
    axfp8_matvec_kernel<<<grd, blk, 0, (cudaStream_t)stream>>>(
            weight, block_scale, input, out, rows, cols, cols / 128u);
    return axnv_finish_after_launch((cudaStream_t)stream);
}

extern "C" int axiom_cuda_axpby_f32_async(int device, float *out, const float *in,
                                          float alpha, float beta, uint32_t n, void *stream) {
    if (!out || !in || n == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(256), grd((n + 255u) / 256u);
    axnv_axpby_kernel<<<grd, blk, 0, (cudaStream_t)stream>>>(out, in, alpha, beta, n);
    return axnv_finish_after_launch((cudaStream_t)stream);
}
