/* axiom_cuda_pme.cu — AXIOM unified memory-injected serving kernel (PME fused GEMV).
 *
 * One kernel pass computes
 *     y = dequant_NVFP4(W_base) . x  +  (B . (A . x)) * pack_scale
 * i.e. the frozen NVFP4 (or FP8) base weight AND a hot-attachable rank-r adapter
 * ("memory pack" — the serving-side counterpart of the PME/EXP-067 parametric-memory
 * research: LoRA-style per-site adapters trained to inject facts/memories) applied
 * per-request WITHOUT dequantizing, merging, or requantizing the base model.
 *
 * HOUSE GATE (OFF == identical to the base kernel):
 * pack_a == NULL MUST be bit-identical to the plain kernels in src/axiom_cuda_nvfp4.cu.
 * To that end:
 *   - the e2m1/e4m3/e8m0 decode helpers below are copied VERBATIM from
 *     src/axiom_cuda_nvfp4.cu (axnv_e2m1:20, axnv_e4m3:29, axnv_e8m0:71);
 *   - the base accumulation loops are copied VERBATIM from axnv_matvec_kernel:36-50
 *     and axfp8_matvec_kernel:75-87 — same fp ops, same order, same expression trees;
 *   - the ENTIRE adapter block is guarded by `pack_a != NULL` (a kernel argument,
 *     uniform across the grid, so the branch never diverges within a block) and runs
 *     strictly OUTSIDE the base accumulation: base partial sums are untouched;
 *   - launch geometry (blockDim 64) matches the plain kernels.
 * The only structural delta vs the plain kernels: the `r >= rows` early-return moves
 * BELOW the cooperative pack staging, because the staging __syncthreads() must be
 * reached by every thread of the block (the last block may carry r >= rows threads;
 * returning before a __syncthreads() is UB). This changes no arithmetic: per output
 * row the executed fp op sequence is identical.
 * Residual risk: nvcc -O3 --use_fast_math could in principle schedule/contract the
 * (identical) base loop differently in this TU. The expression trees are identical and
 * FMA contraction is per-expression, so this is not expected; it is PROVEN on GB10 by
 * tests/axiom_pme_fused_smoke.cpp, which runs BOTH kernels and memcmps the outputs.
 * Contingency if that gate ever fails on a new toolchain: dispatch pack==NULL calls to
 * the plain entries at the host level (bit-identity by construction); documented in
 * docs/axiom_pme_fused_kernel_design.md.
 *
 * Adapter math (standard LoRA semantics; PME artifacts not present in this checkout —
 * see UNRESOLVED in the design doc): A f32 [rank, cols] row-major, B f32 [rows, rank]
 * row-major, pack_scale = alpha / rank precomputed by the caller. rank <= 256.
 * t = A.x is staged cooperatively in shared memory once per block (redundantly across
 * blocks — adapter traffic r*(in+out)*4 bytes is negligible vs base weight traffic at
 * r <= 256; see the design doc perf notes), then each row adds dot(B[row], t)*pack_scale.
 *
 * Compiled with the SAME nvcc flags as axiom_cuda_nvfp4.cu (see the append-only
 * Makefile block). Not yet a member of AXIOM_OBJS/libaxiom.so (append-only Makefile
 * constraint); the smoke links this object directly next to libaxiom.
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "axiom/axiom.h"

#define AXIOM_NVFP4_GROUP 16u

/* ---- decode helpers: copied VERBATIM from src/axiom_cuda_nvfp4.cu (bit-identity
 * requirement — do NOT "improve" these). ---- */
__device__ static inline float axpme_e2m1(uint8_t nib) {
    float v;
    switch (nib & 7u) {
        case 0: v = 0.0f; break; case 1: v = 0.5f; break; case 2: v = 1.0f; break; case 3: v = 1.5f; break;
        case 4: v = 2.0f; break; case 5: v = 3.0f; break; case 6: v = 4.0f; break; default: v = 6.0f; break;
    }
    return (nib & 8u) ? -v : v;
}
/* OCP FP8 e4m3 (e4m3fn): s eeee mmm, exp bias 7, no inf. */
__device__ static inline float axpme_e4m3(uint8_t b) {
    uint32_t s = (b >> 7) & 1u, e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v = (e == 0u) ? (float)m * (0.015625f / 8.0f)
                        : (1.0f + (float)m / 8.0f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}
__device__ static inline float axpme_e8m0(uint8_t b) {
    /* unsigned 8-bit exponent (bias 127): value = 2^(b-127). 0xff = NaN per spec; scales aren't NaN. */
    return exp2f((float)((int)b - 127));
}

/* Cooperative rank-r staging: t[0..pack_rank) = A . x, computed once per block into
 * shared memory. Every thread of the block participates (call this BEFORE any
 * r >= rows return — the __syncthreads() must be reached by the whole block).
 * Deterministic accumulation: each t[j] is one thread's sequential f32 loop over
 * k = 0..cols-1 (documented op order; the oracle tolerance covers f32-vs-f64 drift). */
__device__ static inline void axpme_stage_pack(
        const float *pack_a, uint32_t pack_rank, const float *x, uint32_t cols, float *t) {
    for (uint32_t j = threadIdx.x; j < pack_rank; j += blockDim.x) {
        const float *arow = pack_a + (uint64_t)j * cols;
        float tj = 0.0f;
        for (uint32_t k = 0; k < cols; ++k) tj += arow[k] * x[k];
        t[j] = tj;
    }
    __syncthreads();
}

/* ================= NVFP4 base (routed experts) + fused memory pack ================= */
__global__ static void axpme_nvfp4_matvec_kernel(
        const uint8_t *w, const uint8_t *bscale, float global,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *x, float *out, uint32_t rows, uint32_t cols) {
    __shared__ float t[AXIOM_PME_MAX_RANK];
    /* Uniform branch: pack_a is a kernel argument, identical for every thread. */
    if (pack_a != NULL) axpme_stage_pack(pack_a, pack_rank, x, cols, t);
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return; /* safe: no __syncthreads() below this point */
    /* ---- base NVFP4 accumulation: copied VERBATIM from axnv_matvec_kernel
     * (src/axiom_cuda_nvfp4.cu:36-50). Do NOT restructure — the pack==NULL path of
     * this kernel must execute exactly these fp ops in exactly this order. ---- */
    uint32_t groups = cols / AXIOM_NVFP4_GROUP;
    float acc = 0.0f;
    for (uint32_t k = 0; k < cols; ++k) {
        uint64_t idx = (uint64_t)r * cols + k;
        uint8_t byte = w[idx >> 1];
        uint8_t nib = (idx & 1u) ? (byte >> 4) : (byte & 0x0f);
        float bs = axpme_e4m3(bscale[(uint64_t)r * groups + (k / AXIOM_NVFP4_GROUP)]);
        acc += axpme_e2m1(nib) * bs * global * x[k];
    }
    /* ---- fused memory-pack update: entirely absent when pack_a == NULL, appended
     * strictly AFTER the base accumulation completed (base partials untouched). ---- */
    if (pack_a != NULL) {
        const float *brow = pack_b + (uint64_t)r * pack_rank;
        float upd = 0.0f;
        for (uint32_t j = 0; j < pack_rank; ++j) upd += brow[j] * t[j];
        acc += upd * pack_scale;
    }
    out[r] = acc;
}

/* Raw-device-pointer entry (pointers already offset by caller; device = CUDA device
 * index). pack_a == NULL (with pack_b == NULL) => memory pack OFF => output bit-identical
 * to axiom_cuda_e2m1_nvfp4_matvec_f32. Returns AXIOM_OK / AXIOM_ERR_*. */
extern "C" int axiom_cuda_e2m1_nvfp4_pme_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (cols % AXIOM_NVFP4_GROUP) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((pack_a == NULL) != (pack_b == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pack_a != NULL && (pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK))
        return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u); /* same geometry as the plain kernel */
    axpme_nvfp4_matvec_kernel<<<grd, blk>>>(weight, block_scale, global_scale,
            pack_a, pack_b, pack_a != NULL ? pack_rank : 0u, pack_scale,
            input, out, rows, cols);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cudaDeviceSynchronize() != cudaSuccess) return AXIOM_ERR_CUDA;
    return AXIOM_OK;
}

/* ============ FP8 base (attn / shared experts) + fused memory pack ============
 * PME/EXP-067 trains attn AND ffn sites; DeepSeek-V4-Flash keeps attn/shared in FP8
 * (E4M3 weight + E8M0 128x128 block scale), so those sites need this variant. */
__global__ static void axpme_fp8_matvec_kernel(
        const uint8_t *w, const uint8_t *scale,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *x, float *out, uint32_t rows, uint32_t cols, uint32_t sc_cols) {
    __shared__ float t[AXIOM_PME_MAX_RANK];
    if (pack_a != NULL) axpme_stage_pack(pack_a, pack_rank, x, cols, t);
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return; /* safe: no __syncthreads() below this point */
    /* ---- base FP8 accumulation: copied VERBATIM from axfp8_matvec_kernel
     * (src/axiom_cuda_nvfp4.cu:75-87). ---- */
    float acc = 0.0f;
    uint32_t sr = r >> 7;   /* row / 128 */
    for (uint32_t c = 0; c < cols; ++c) {
        float wv = axpme_e4m3(w[(uint64_t)r * cols + c]);
        float s = axpme_e8m0(scale[(uint64_t)sr * sc_cols + (c >> 7)]);
        acc += wv * s * x[c];
    }
    if (pack_a != NULL) {
        const float *brow = pack_b + (uint64_t)r * pack_rank;
        float upd = 0.0f;
        for (uint32_t j = 0; j < pack_rank; ++j) upd += brow[j] * t[j];
        acc += upd * pack_scale;
    }
    out[r] = acc;
}

extern "C" int axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols) {
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (rows % 128u) != 0u || (cols % 128u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((pack_a == NULL) != (pack_b == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pack_a != NULL && (pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK))
        return AXIOM_ERR_INVALID_ARGUMENT;
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    dim3 blk(64), grd((rows + 63u) / 64u);
    axpme_fp8_matvec_kernel<<<grd, blk>>>(weight, block_scale,
            pack_a, pack_b, pack_a != NULL ? pack_rank : 0u, pack_scale,
            input, out, rows, cols, cols / 128u);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cudaDeviceSynchronize() != cudaSuccess) return AXIOM_ERR_CUDA;
    return AXIOM_OK;
}

/* ================= indexed MoE variant (per-expert memory slices) =================
 * Routed-expert sites carry ONE contiguous all-experts pack allocation:
 *     pack_a_all f32 [experts, rank, cols]   (per-expert A at expert*rank*cols floats)
 *     pack_b_all f32 [experts, rows, rank]   (per-expert B at expert*rows*rank floats)
 * `weight`/`block_scale` are the ALREADY-OFFSET per-expert base planes — exactly what
 * the host MoE loop (axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device,
 * src/axiom_runtime.cpp:7379-7402) computes today via e*stride, so this slots into
 * that loop as a drop-in replacement for the plain matvec call. Per-expert pack_scale
 * is the caller's (mirrors the per-expert global-scale download pattern there).
 * pack_a_all == NULL => OFF => bit-identical to the plain matvec (forwards NULL).
 * experts is capped at 65536 so expert*rank*cols*4 cannot overflow u64. */
extern "C" int axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *pack_a_all, const float *pack_b_all,
        uint32_t experts, uint32_t expert, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols) {
    const float *pa = NULL, *pb = NULL;
    if ((pack_a_all == NULL) != (pack_b_all == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pack_a_all != NULL) {
        if (experts == 0u || experts > 65536u || expert >= experts ||
            pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK)
            return AXIOM_ERR_INVALID_ARGUMENT;
        pa = pack_a_all + (uint64_t)expert * pack_rank * cols;   /* <= 2^16*2^8*2^32 elems: fits u64 */
        pb = pack_b_all + (uint64_t)expert * rows * pack_rank;
    }
    return axiom_cuda_e2m1_nvfp4_pme_matvec_f32(device, weight, block_scale, global_scale,
            pa, pb, pack_rank, pack_scale, input, out, rows, cols);
}

/* ================= backend_buffer(+offset) device adapters =================
 * Bridge opaque axiom_device_buffer backend handles (+ byte offsets) to the raw entries
 * above, mirroring the adapter pattern of src/axiom_cuda.cu:18043-18113.
 *
 * ABI NOTE (deliberate, documented): the canonical buffer struct is PRIVATE to
 * src/axiom_cuda.cu:33-37:
 *     struct axiom_cuda_buffer { int device; void *ptr; uint64_t bytes; };
 * The established home for _device adapters is axiom_cuda.cu next to that struct, but
 * this change is zero-touch to owned files, so we mirror the 3-field layout locally
 * and cast the opaque handle to the mirror. The static_asserts below pin the mirror to
 * the exact offsets the :18043-18113 adapters rely on (int at 0, ptr at 8, bytes at 16
 * on LP64). RISK: if axiom_cuda.cu ever reorders/extends those leading fields, these
 * adapters read garbage — the long-term fix is moving these thin functions into
 * axiom_cuda.cu (pure copy-paste; raw entries are unaffected). Tracked as UNRESOLVED
 * in docs/axiom_pme_fused_kernel_design.md.
 *
 * Unlike the :18043 adapters (whose bounds are validated by axiom_runtime.cpp wrappers),
 * no runtime wrapper exists yet for the PME entries, so these adapters validate every
 * span themselves against the mirrored `bytes` field, plus 4-byte alignment of all f32
 * offsets. Size products cannot overflow u64: rows,cols < 2^32 (=> rows*cols/2 < 2^63)
 * and pack_rank <= 256 (=> rank*cols*4 <= 2^42, rows*rank*4 <= 2^42). */
struct axiom_pme_buffer_view { int device; void *ptr; uint64_t bytes; };
static_assert(offsetof(axiom_pme_buffer_view, device) == 0, "axiom_cuda_buffer mirror: device at 0");
static_assert(offsetof(axiom_pme_buffer_view, ptr) == 8, "axiom_cuda_buffer mirror: ptr at 8");
static_assert(offsetof(axiom_pme_buffer_view, bytes) == 16, "axiom_cuda_buffer mirror: bytes at 16");
static_assert(sizeof(axiom_pme_buffer_view) == 24, "axiom_cuda_buffer mirror: 24 bytes");

static bool axpme_span_ok(const void *handle, uint64_t off, uint64_t need) {
    const axiom_pme_buffer_view *b = (const axiom_pme_buffer_view *)handle;
    return b != NULL && b->ptr != NULL && off <= b->bytes && need <= b->bytes - off;
}
static const uint8_t *axpme_cptr(const void *handle, uint64_t off) {
    return (const uint8_t *)((const axiom_pme_buffer_view *)handle)->ptr + off;
}
static uint8_t *axpme_mptr(void *handle, uint64_t off) {
    return (uint8_t *)((axiom_pme_buffer_view *)handle)->ptr + off;
}

extern "C" int axiom_cuda_e2m1_nvfp4_pme_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset, float global_scale,
        const void *pack_a, uint64_t pack_a_offset,
        const void *pack_b, uint64_t pack_b_offset,
        uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols) {
    (void)cuda_runtime;
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (cols % AXIOM_NVFP4_GROUP) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((pack_a == NULL) != (pack_b == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((input_offset % 4u) != 0u || (out_offset % 4u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!axpme_span_ok(weight, weight_offset, (uint64_t)rows * (cols / 2u)) ||
        !axpme_span_ok(block_scale, block_scale_offset, (uint64_t)rows * (cols / AXIOM_NVFP4_GROUP)) ||
        !axpme_span_ok(input, input_offset, (uint64_t)cols * 4u) ||
        !axpme_span_ok(out, out_offset, (uint64_t)rows * 4u)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pack_a != NULL) {
        if (pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK ||
            (pack_a_offset % 4u) != 0u || (pack_b_offset % 4u) != 0u ||
            !axpme_span_ok(pack_a, pack_a_offset, (uint64_t)pack_rank * cols * 4u) ||
            !axpme_span_ok(pack_b, pack_b_offset, (uint64_t)rows * pack_rank * 4u))
            return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_e2m1_nvfp4_pme_matvec_f32(
            ((const axiom_pme_buffer_view *)weight)->device,
            axpme_cptr(weight, weight_offset),
            axpme_cptr(block_scale, block_scale_offset),
            global_scale,
            pack_a != NULL ? (const float *)axpme_cptr(pack_a, pack_a_offset) : NULL,
            pack_b != NULL ? (const float *)axpme_cptr(pack_b, pack_b_offset) : NULL,
            pack_rank, pack_scale,
            (const float *)axpme_cptr(input, input_offset),
            (float *)axpme_mptr(out, out_offset),
            rows, cols);
}

extern "C" int axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset,
        const void *pack_a, uint64_t pack_a_offset,
        const void *pack_b, uint64_t pack_b_offset,
        uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols) {
    (void)cuda_runtime;
    if (!weight || !block_scale || !input || !out || rows == 0u || cols == 0u ||
        (rows % 128u) != 0u || (cols % 128u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((pack_a == NULL) != (pack_b == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    if ((input_offset % 4u) != 0u || (out_offset % 4u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!axpme_span_ok(weight, weight_offset, (uint64_t)rows * cols) ||
        !axpme_span_ok(block_scale, block_scale_offset, (uint64_t)(rows / 128u) * (cols / 128u)) ||
        !axpme_span_ok(input, input_offset, (uint64_t)cols * 4u) ||
        !axpme_span_ok(out, out_offset, (uint64_t)rows * 4u)) return AXIOM_ERR_INVALID_ARGUMENT;
    if (pack_a != NULL) {
        if (pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK ||
            (pack_a_offset % 4u) != 0u || (pack_b_offset % 4u) != 0u ||
            !axpme_span_ok(pack_a, pack_a_offset, (uint64_t)pack_rank * cols * 4u) ||
            !axpme_span_ok(pack_b, pack_b_offset, (uint64_t)rows * pack_rank * 4u))
            return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32(
            ((const axiom_pme_buffer_view *)weight)->device,
            axpme_cptr(weight, weight_offset),
            axpme_cptr(block_scale, block_scale_offset),
            pack_a != NULL ? (const float *)axpme_cptr(pack_a, pack_a_offset) : NULL,
            pack_b != NULL ? (const float *)axpme_cptr(pack_b, pack_b_offset) : NULL,
            pack_rank, pack_scale,
            (const float *)axpme_cptr(input, input_offset),
            (float *)axpme_mptr(out, out_offset),
            rows, cols);
}

extern "C" int axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset, float global_scale,
        const void *pack_a_all, uint64_t pack_a_all_offset,
        const void *pack_b_all, uint64_t pack_b_all_offset,
        uint32_t experts, uint32_t expert, uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols) {
    if ((pack_a_all == NULL) != (pack_b_all == NULL)) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t a_off = pack_a_all_offset, b_off = pack_b_all_offset;
    if (pack_a_all != NULL) {
        if (experts == 0u || experts > 65536u || expert >= experts ||
            pack_rank == 0u || pack_rank > AXIOM_PME_MAX_RANK)
            return AXIOM_ERR_INVALID_ARGUMENT;
        /* Validate the FULL all-experts spans, then narrow to the expert slice by
         * offset (experts<=2^16, rank<=2^8, cols/rows<2^32 => products fit u64). */
        if (!axpme_span_ok(pack_a_all, pack_a_all_offset, (uint64_t)experts * pack_rank * cols * 4u) ||
            !axpme_span_ok(pack_b_all, pack_b_all_offset, (uint64_t)experts * rows * pack_rank * 4u))
            return AXIOM_ERR_INVALID_ARGUMENT;
        a_off += (uint64_t)expert * pack_rank * cols * 4u;
        b_off += (uint64_t)expert * rows * pack_rank * 4u;
    }
    return axiom_cuda_e2m1_nvfp4_pme_matvec_f32_device(
            cuda_runtime, weight, weight_offset, block_scale, block_scale_offset, global_scale,
            pack_a_all, a_off, pack_b_all, b_off, pack_rank, pack_scale,
            input, input_offset, out, out_offset, rows, cols);
}
