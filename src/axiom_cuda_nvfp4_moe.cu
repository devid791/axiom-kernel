/* axiom_cuda_nvfp4_moe.cu — fused, device-resident NVFP4 routed-expert MoE for
 * DeepSeek V4 Flash on Blackwell (audit item #1: replaces the host-orchestrated
 * per-expert loop of axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device,
 * src/axiom_runtime.cpp:7335-7405, which per FFN layer per token performed a blocking
 * D2H of the router indices/weights, re-downloaded all per-expert global scales, and
 * issued ~30 sequential kernel launches each ending in cudaDeviceSynchronize).
 *
 * This TU launches exactly TWO kernels per token per FFN layer, all inputs on-device:
 *   kernel 1 (gate+up+silu, grid.y = slot): for each router slot j with expert
 *       e = indices[j],  mid[j][r] = silu(gate_e[r]·x) * (up_e[r]·x)
 *   kernel 2 (weighted down):  out[r] = sum_j router_w[j] * gscale_down[e_j] *
 *       (down_{e_j}[r] · mid[j])                                (out is OVERWRITTEN)
 * No cudaDeviceSynchronize inside; launches go to a caller-provided stream (NULL =
 * default stream) and completion follows the axiom_cuda_finish_after_launch convention
 * (src/axiom_cuda.cu:197-201): when AXIOM_CUDA_LAUNCH_ASYNC is enabled (CACHED getenv —
 * the audit flagged the per-call getenv) or the stream is capturing, return right after
 * launch; otherwise cudaStreamSynchronize (stream-scoped, not device-wide).
 *
 * Weight layout = exactly what the loader produces and the host loop consumes
 * (src/axiom_runtime.cpp:7374-7397): de-blocked combined planes, expert e at e*stride:
 *   weight       U8  [experts, rows, cols/2]   2x signed-E2M1 nibbles/byte (low = even col)
 *   block_scale  U8  [experts, rows, cols/16]  per-group-16 e4m3 codes
 *   global_scale F32 [experts]                 per-expert per-tensor scale (ON DEVICE)
 * gate/up: rows = expert_hidden, cols = hidden; down: rows = hidden, cols = expert_hidden.
 *
 * Kernel shape (proven in-house template: the Q8K fused MoE chain, src/axiom_cuda.cu
 * :13835-13890 / gate+up rowstage / down_q8k_sum6_qwarp32): warp-per-row GEMV. Each lane
 * loads ONE uint4 (16 B = 32 nibbles = 32 cols = 2 scale groups) per step, adjacent
 * lanes -> adjacent 16 B chunks, so each warp-level weight load is a fully-coalesced
 * 512 B contiguous span — the DRAM-bound stream (the FP4 planes) is perfectly coalesced,
 * row-major within the warp. e2m1 decodes through a 16-entry __constant__ LUT; the e4m3
 * block scale decodes through a 256-entry __constant__ LUT ONCE per group-16 (2 lookups
 * per uint4, not per element). x / mid are read via __ldg (read-only path); they are
 * small (hidden*4 = 16 KB, topk*expert_hidden*4 = 48 KB at DS4 dims), fully L1/L2
 * resident after first touch, so their per-lane 32-float spans cost L1 bandwidth, not
 * DRAM (see docs/axiom_nvfp4_fused_moe_design.md for the traffic model).
 *
 * NUMERICS: NOT bit-exact vs the host loop, by design. The matvec kernel it replaces
 * (axnv_matvec_kernel, src/axiom_cuda_nvfp4.cu:36-50) accumulates
 * e2m1*bs*global*x per element in strict column order; this kernel accumulates
 * per-group-16 partial dots, multiplies each by its block scale, sums per-lane partials
 * in lane-strided order, warp-reduces, and applies the global scale once at the end.
 * Same real-number value, different fp32 association order => parity bar is RELATIVE
 * error (see tests/axiom_nvfp4_smoke.cpp fused section and the design doc), matching
 * how the Q8K fused chain is validated.
 *
 * Compiled with the same nvcc recipe as axiom_cuda_nvfp4.o (Makefile:69-70). NOT yet a
 * member of AXIOM_OBJS (that := line is mid-file and not append-only for this change);
 * the standalone smoke links build/axiom_cuda_nvfp4_moe.o explicitly — adding it to
 * AXIOM_OBJS is a one-line edit deferred to the libaxiom owner (UNRESOLVED in the
 * design doc; until then the axiom_runtime.cpp wrapper leaves one lazily-bound
 * undefined symbol in libaxiom.so that only resolves for binaries linking this .o).
 */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "axiom/axiom.h"

#define AXIOM_NVFP4_GROUP 16u
/* One uint4 per lane-step = 32 nibbles = 32 cols = 2 group-16 scale groups. */
#define AXNVM_CHUNK_COLS 32u
#define AXNVM_BLOCK_THREADS 128u   /* 4 warps -> 4 rows per block */
#define AXNVM_WARPS_PER_BLOCK (AXNVM_BLOCK_THREADS / 32u)
#define AXNVM_MAX_TOPK 64u         /* matches the host-loop entry's idx[64] cap */
#define AXNVM_MAX_EXPERTS 65536u   /* PME precedent: keeps e*stride products in u64 */
#define AXNVM_MAX_DIM (1u << 20)   /* rows/cols cap: 2^16 * 2^20 * 2^19 B < 2^56 fits u64 */

/* ---- __constant__ decode LUTs ----
 * e2m1: value(nib) for the 16 signed-E2M1 codes (sign bit 3). Identical to axnv_e2m1
 * (src/axiom_cuda_nvfp4.cu:20-27).
 * e4m3: value(byte) for all 256 OCP e4m3fn codes computed by the same formula as
 * axnv_e4m3 (src/axiom_cuda_nvfp4.cu:29-34): e==0 -> m*2^-9, else (1+m/8)*2^(e-7),
 * sign bit 7. NOTE 0x7f/0xff decode to +/-480 (the formula value) — the existing
 * kernel and both python oracles use the same formula and do NOT NaN-special-case;
 * block scales are never NaN. Every entry is an exact dyadic float. */
__constant__ static float axnvm_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
__constant__ static float axnvm_e4m3_lut[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 480.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, -480.0f,
};

/* dot of the 8 e2m1 nibbles packed in one little-endian u32 (memory bytes b0..b3 ->
 * cols 0..7: byte t = cols 2t (low nibble) / 2t+1 (high nibble)) with x[0..7]. */
__device__ __forceinline__ static float axnvm_dot8(uint32_t w, const float *__restrict__ x) {
    float s = 0.0f;
    #pragma unroll
    for (uint32_t b = 0; b < 4u; ++b) {
        const uint32_t byte = (w >> (8u * b)) & 0xffu;
        s += axnvm_e2m1_lut[byte & 0x0fu] * __ldg(x + 2u * b);
        s += axnvm_e2m1_lut[byte >> 4u]  * __ldg(x + 2u * b + 1u);
    }
    return s;
}

/* ---- kernel 1: fused gate+up+silu for all topk slots ----
 * grid = (ceil(expert_hidden / AXNVM_WARPS_PER_BLOCK), topk), block = 128.
 * warp-per-row; blockIdx.y = router slot. Router indices are consumed ON DEVICE; an
 * out-of-range index cannot surface an error from inside a kernel, so it degrades to a
 * zero contribution (mid row = 0 here; kernel 2 skips the slot) — same convention as
 * the Q8K down kernel's `if (expert >= experts) continue;` (src/axiom_cuda.cu:11554).
 * The host loop errors instead; documented as an accepted divergence in the design doc. */
__global__ static void axnvm_moe_gate_up_silu_kernel(
        const uint8_t *__restrict__ gate_w, const uint8_t *__restrict__ gate_bs,
        const float *__restrict__ gate_gs,
        const uint8_t *__restrict__ up_w, const uint8_t *__restrict__ up_bs,
        const float *__restrict__ up_gs,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ x,
        float *__restrict__ mid,          /* [topk, expert_hidden] */
        uint32_t experts, uint32_t hidden, uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = (uint32_t)blockIdx.x * AXNVM_WARPS_PER_BLOCK + warp;
    const uint32_t slot = blockIdx.y;
    if (row >= expert_hidden) return;   /* whole warp exits together (warp-per-row) */
    const uint32_t e = indices[slot];
    if (e >= experts) {
        if (lane == 0u) mid[(uint64_t)slot * expert_hidden + row] = 0.0f;
        return;
    }
    const uint64_t row_wbytes = hidden >> 1;             /* cols/2 */
    const uint64_t row_groups = hidden >> 4;             /* cols/16 */
    const uint64_t w_off  = ((uint64_t)e * expert_hidden + row) * row_wbytes;
    const uint64_t bs_off = ((uint64_t)e * expert_hidden + row) * row_groups;
    const uint4 *gw = (const uint4 *)(gate_w + w_off);
    const uint4 *uw = (const uint4 *)(up_w + w_off);
    const uint8_t *gbs = gate_bs + bs_off;
    const uint8_t *ubs = up_bs + bs_off;
    const uint32_t chunks = hidden / AXNVM_CHUNK_COLS;   /* uint4 chunks per row */
    float accg = 0.0f, accu = 0.0f;
    for (uint32_t c = lane; c < chunks; c += 32u) {
        const uint4 vg = __ldg(gw + c);
        const uint4 vu = __ldg(uw + c);
        /* two e4m3 block scales per chunk — decoded ONCE per group-16 via the LUT */
        const float sg0 = axnvm_e4m3_lut[__ldg(gbs + 2u * c)];
        const float sg1 = axnvm_e4m3_lut[__ldg(gbs + 2u * c + 1u)];
        const float su0 = axnvm_e4m3_lut[__ldg(ubs + 2u * c)];
        const float su1 = axnvm_e4m3_lut[__ldg(ubs + 2u * c + 1u)];
        const float *xc = x + (uint64_t)c * AXNVM_CHUNK_COLS;
        accg += sg0 * (axnvm_dot8(vg.x, xc)      + axnvm_dot8(vg.y, xc + 8u));
        accg += sg1 * (axnvm_dot8(vg.z, xc + 16u) + axnvm_dot8(vg.w, xc + 24u));
        accu += su0 * (axnvm_dot8(vu.x, xc)      + axnvm_dot8(vu.y, xc + 8u));
        accu += su1 * (axnvm_dot8(vu.z, xc + 16u) + axnvm_dot8(vu.w, xc + 24u));
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        accg += __shfl_down_sync(0xffffffffu, accg, off);
        accu += __shfl_down_sync(0xffffffffu, accu, off);
    }
    if (lane == 0u) {
        /* per-expert global scales applied once per output element (hoisted out of the
         * per-element product — see the NUMERICS note in the file header) */
        const float g = accg * __ldg(gate_gs + e);
        const float u = accu * __ldg(up_gs + e);
        /* silu formula identical to axiom_silu_mul_f32_kernel (src/axiom_cuda.cu:17308) */
        mid[(uint64_t)slot * expert_hidden + row] = (g / (1.0f + expf(-g))) * u;
    }
}

/* ---- kernel 2: weighted down-projection summed over all topk slots ----
 * grid = ceil(hidden / AXNVM_WARPS_PER_BLOCK), block = 128; warp-per-row over hidden.
 * out[row] is OVERWRITTEN with the full MoE sum (same net effect as the host loop's
 * axpby beta=0-then-1 chain — the caller's moe_out prior contents are dead). */
__global__ static void axnvm_moe_down_weighted_kernel(
        const uint8_t *__restrict__ down_w, const uint8_t *__restrict__ down_bs,
        const float *__restrict__ down_gs,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ router_w,
        const float *__restrict__ mid,    /* [topk, expert_hidden] */
        float *__restrict__ out,          /* [hidden] */
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = (uint32_t)blockIdx.x * AXNVM_WARPS_PER_BLOCK + warp;
    if (row >= hidden) return;          /* whole warp exits together */
    const uint64_t row_wbytes = expert_hidden >> 1;
    const uint64_t row_groups = expert_hidden >> 4;
    const uint32_t chunks = expert_hidden / AXNVM_CHUNK_COLS;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < topk; ++slot) {
        const uint32_t e = indices[slot];
        if (e >= experts) continue;     /* zero contribution; kernel 1 zeroed mid[slot] */
        const uint4 *dw = (const uint4 *)(down_w + ((uint64_t)e * hidden + row) * row_wbytes);
        const uint8_t *dbs = down_bs + ((uint64_t)e * hidden + row) * row_groups;
        const float *m = mid + (uint64_t)slot * expert_hidden;
        float acc = 0.0f;
        for (uint32_t c = lane; c < chunks; c += 32u) {
            const uint4 v = __ldg(dw + c);
            const float s0 = axnvm_e4m3_lut[__ldg(dbs + 2u * c)];
            const float s1 = axnvm_e4m3_lut[__ldg(dbs + 2u * c + 1u)];
            const float *mc = m + (uint64_t)c * AXNVM_CHUNK_COLS;
            acc += s0 * (axnvm_dot8(v.x, mc)       + axnvm_dot8(v.y, mc + 8u));
            acc += s1 * (axnvm_dot8(v.z, mc + 16u) + axnvm_dot8(v.w, mc + 24u));
        }
        for (uint32_t off = 16u; off > 0u; off >>= 1u) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0u) total += __ldg(router_w + slot) * __ldg(down_gs + e) * acc;
    }
    if (lane == 0u) out[row] = total;
}

/* ---- axiom_cuda_finish_after_launch convention, local replica ----
 * The canonical helper (src/axiom_cuda.cu:197-201) is `static` in that TU, so the
 * 3-line logic is replicated here — with the getenv CACHED (audit flag: the per-call
 * getenv of AXIOM_CUDA_LAUNCH_ASYNC shows up in per-token profiles). Same truthiness
 * as axiom_cuda_env_enabled (src/axiom_cuda.cu:102-107); the snapshot is taken at
 * first use, so flipping the variable mid-process is not observed (documented in the
 * design doc). Sync is STREAM-scoped, not device-wide. */
static bool axnvm_launch_async_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *v = std::getenv("AXIOM_CUDA_LAUNCH_ASYNC");
        cached = (v && v[0] != '\0' && std::strcmp(v, "0") != 0 &&
                  std::strcmp(v, "false") != 0 && std::strcmp(v, "False") != 0 &&
                  std::strcmp(v, "FALSE") != 0) ? 1 : 0;
    }
    return cached != 0;
}
static bool axnvm_stream_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
}
static int axnvm_finish_after_launch(cudaStream_t stream) {
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (axnvm_launch_async_enabled() || axnvm_stream_capturing(stream)) return AXIOM_OK;
    return cudaStreamSynchronize(stream) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

static bool axnvm_ptr_uint4_aligned(const void *p) {
    return ((uintptr_t)p & 15u) == 0u;   /* uint4 loads require 16-byte alignment */
}

/* ---- raw-device-pointer entry ----
 * Pointers are the PLANE BASES (expert 0); per-expert offsets are computed in-kernel
 * from the on-device router indices — nothing is downloaded to the host. stream is a
 * cudaStream_t (NULL = default stream; with --default-stream per-thread that is the
 * per-thread stream). Dimension caps keep every in-kernel u64 offset product exact:
 * experts <= 2^16, rows/cols <= 2^20 => e*rows*cols/2 < 2^55. hidden/expert_hidden
 * must be multiples of 32 (uint4 = 32 nibbles) and the weight planes 16-byte aligned
 * (cudaMalloc bases always are; cols%32==0 keeps every row and expert stride aligned). */
extern "C" int axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32(
        int device,
        const uint8_t *gate_w, const uint8_t *gate_bs, const float *gate_gs,
        const uint8_t *up_w, const uint8_t *up_bs, const float *up_gs,
        const uint8_t *down_w, const uint8_t *down_bs, const float *down_gs,
        const uint32_t *router_indices, const float *router_weights,
        const float *input, float *scratch_mid, float *out, void *stream,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden) {
    if (!gate_w || !gate_bs || !gate_gs || !up_w || !up_bs || !up_gs ||
        !down_w || !down_bs || !down_gs || !router_indices || !router_weights ||
        !input || !scratch_mid || !out ||
        experts == 0u || experts > AXNVM_MAX_EXPERTS ||
        topk == 0u || topk > AXNVM_MAX_TOPK ||
        hidden == 0u || hidden > AXNVM_MAX_DIM || (hidden % AXNVM_CHUNK_COLS) != 0u ||
        expert_hidden == 0u || expert_hidden > AXNVM_MAX_DIM ||
        (expert_hidden % AXNVM_CHUNK_COLS) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!axnvm_ptr_uint4_aligned(gate_w) || !axnvm_ptr_uint4_aligned(up_w) ||
        !axnvm_ptr_uint4_aligned(down_w)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    cudaStream_t s = (cudaStream_t)stream;
    const dim3 blk(AXNVM_BLOCK_THREADS);
    const dim3 grd1((expert_hidden + AXNVM_WARPS_PER_BLOCK - 1u) / AXNVM_WARPS_PER_BLOCK, topk);
    const dim3 grd2((hidden + AXNVM_WARPS_PER_BLOCK - 1u) / AXNVM_WARPS_PER_BLOCK);
    axnvm_moe_gate_up_silu_kernel<<<grd1, blk, 0, s>>>(
            gate_w, gate_bs, gate_gs, up_w, up_bs, up_gs,
            router_indices, input, scratch_mid, experts, hidden, expert_hidden);
    axnvm_moe_down_weighted_kernel<<<grd2, blk, 0, s>>>(
            down_w, down_bs, down_gs, router_indices, router_weights,
            scratch_mid, out, experts, topk, hidden, expert_hidden);
    return axnvm_finish_after_launch(s);
}

/* ---- backend_buffer(+offset) device adapter ----
 * Bridges opaque axiom_device_buffer backend handles to the raw entry. Same
 * deliberate, documented mirror of the axiom_cuda.cu-private buffer struct as
 * axiom_cuda_pme.cu:243-247 (struct axiom_cuda_buffer { int device; void *ptr;
 * uint64_t bytes; }, src/axiom_cuda.cu:33-37) — this TU is zero-touch to axiom_cuda.cu,
 * so the 3-field layout is mirrored and pinned by static_asserts. RISK + long-term fix
 * identical to the PME note (move the adapter into axiom_cuda.cu); tracked UNRESOLVED
 * in docs/axiom_nvfp4_fused_moe_design.md. Every span is validated here with
 * overflow-checked u64 products; the axiom_runtime.cpp wrapper re-validates at the
 * axiom_device_buffer level (belt and braces, house idiom). */
struct axnvm_buffer_view { int device; void *ptr; uint64_t bytes; };
static_assert(offsetof(axnvm_buffer_view, device) == 0, "axiom_cuda_buffer mirror: device at 0");
static_assert(offsetof(axnvm_buffer_view, ptr) == 8, "axiom_cuda_buffer mirror: ptr at 8");
static_assert(offsetof(axnvm_buffer_view, bytes) == 16, "axiom_cuda_buffer mirror: bytes at 16");
static_assert(sizeof(axnvm_buffer_view) == 24, "axiom_cuda_buffer mirror: 24 bytes");

static int axnvm_mul_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = a * b;
    return AXIOM_OK;
}
/* plane bytes = experts * rows * (cols/div), overflow-checked. */
static int axnvm_plane_bytes_checked(uint32_t experts, uint32_t rows, uint32_t cols,
                                     uint32_t div, uint64_t *out) {
    uint64_t bytes = 0;
    int rc = axnvm_mul_u64_checked(experts, rows, &bytes);
    rc = rc == AXIOM_OK ? axnvm_mul_u64_checked(bytes, cols / div, &bytes) : rc;
    if (rc == AXIOM_OK) *out = bytes;
    return rc;
}
static bool axnvm_span_ok(const void *handle, uint64_t off, uint64_t need) {
    const axnvm_buffer_view *b = (const axnvm_buffer_view *)handle;
    return b != NULL && b->ptr != NULL && off <= b->bytes && need <= b->bytes - off;
}
static const uint8_t *axnvm_cptr(const void *handle, uint64_t off) {
    return (const uint8_t *)((const axnvm_buffer_view *)handle)->ptr + off;
}
static uint8_t *axnvm_mptr(void *handle, uint64_t off) {
    return (uint8_t *)((axnvm_buffer_view *)handle)->ptr + off;
}

/* Plane buffers carry no separate offsets (whole combined tensors, matching the host
 * loop's argument surface); input/indices/weights/mid/out take byte offsets.
 * scratch_mid must hold the full [topk, expert_hidden] f32 mid plane. */
extern "C" int axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32_device(
        void *cuda_runtime,
        const void *gate_w, const void *gate_bs, const void *gate_gs,
        const void *up_w, const void *up_bs, const void *up_gs,
        const void *down_w, const void *down_bs, const void *down_gs,
        const void *input, uint64_t input_offset,
        const void *router_indices, uint64_t router_indices_offset,
        const void *router_weights, uint64_t router_weights_offset,
        void *scratch_mid, uint64_t scratch_mid_offset,
        void *out, uint64_t out_offset,
        void *stream,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden) {
    (void)cuda_runtime;   /* device index comes from the buffer views (PME pattern) */
    if (!gate_w || !gate_bs || !gate_gs || !up_w || !up_bs || !up_gs ||
        !down_w || !down_bs || !down_gs || !input || !router_indices ||
        !router_weights || !scratch_mid || !out ||
        experts == 0u || experts > AXNVM_MAX_EXPERTS ||
        topk == 0u || topk > AXNVM_MAX_TOPK ||
        hidden == 0u || hidden > AXNVM_MAX_DIM || (hidden % AXNVM_CHUNK_COLS) != 0u ||
        expert_hidden == 0u || expert_hidden > AXNVM_MAX_DIM ||
        (expert_hidden % AXNVM_CHUNK_COLS) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if ((input_offset % 4u) != 0u || (router_indices_offset % 4u) != 0u ||
        (router_weights_offset % 4u) != 0u || (scratch_mid_offset % 4u) != 0u ||
        (out_offset % 4u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t gu_w_bytes = 0, gu_bs_bytes = 0, dn_w_bytes = 0, dn_bs_bytes = 0, mid_bytes = 0;
    int rc = axnvm_plane_bytes_checked(experts, expert_hidden, hidden, 2u, &gu_w_bytes);
    rc = rc == AXIOM_OK ? axnvm_plane_bytes_checked(experts, expert_hidden, hidden, 16u, &gu_bs_bytes) : rc;
    rc = rc == AXIOM_OK ? axnvm_plane_bytes_checked(experts, hidden, expert_hidden, 2u, &dn_w_bytes) : rc;
    rc = rc == AXIOM_OK ? axnvm_plane_bytes_checked(experts, hidden, expert_hidden, 16u, &dn_bs_bytes) : rc;
    rc = rc == AXIOM_OK ? axnvm_mul_u64_checked((uint64_t)topk * expert_hidden, 4u, &mid_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    const uint64_t gs_bytes = (uint64_t)experts * sizeof(float);
    if (!axnvm_span_ok(gate_w, 0, gu_w_bytes) || !axnvm_span_ok(gate_bs, 0, gu_bs_bytes) ||
        !axnvm_span_ok(gate_gs, 0, gs_bytes) ||
        !axnvm_span_ok(up_w, 0, gu_w_bytes) || !axnvm_span_ok(up_bs, 0, gu_bs_bytes) ||
        !axnvm_span_ok(up_gs, 0, gs_bytes) ||
        !axnvm_span_ok(down_w, 0, dn_w_bytes) || !axnvm_span_ok(down_bs, 0, dn_bs_bytes) ||
        !axnvm_span_ok(down_gs, 0, gs_bytes) ||
        !axnvm_span_ok(input, input_offset, (uint64_t)hidden * sizeof(float)) ||
        !axnvm_span_ok(router_indices, router_indices_offset, (uint64_t)topk * sizeof(uint32_t)) ||
        !axnvm_span_ok(router_weights, router_weights_offset, (uint64_t)topk * sizeof(float)) ||
        !axnvm_span_ok(scratch_mid, scratch_mid_offset, mid_bytes) ||
        !axnvm_span_ok(out, out_offset, (uint64_t)hidden * sizeof(float))) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    /* all views must live on the weight plane's device (single-GPU chain) */
    const int device = ((const axnvm_buffer_view *)gate_w)->device;
    const void *views[14] = { gate_w, gate_bs, gate_gs, up_w, up_bs, up_gs,
                              down_w, down_bs, down_gs, input, router_indices,
                              router_weights, scratch_mid, out };
    for (int i = 0; i < 14; ++i) {
        if (((const axnvm_buffer_view *)views[i])->device != device) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    return axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32(
            device,
            axnvm_cptr(gate_w, 0), axnvm_cptr(gate_bs, 0), (const float *)axnvm_cptr(gate_gs, 0),
            axnvm_cptr(up_w, 0), axnvm_cptr(up_bs, 0), (const float *)axnvm_cptr(up_gs, 0),
            axnvm_cptr(down_w, 0), axnvm_cptr(down_bs, 0), (const float *)axnvm_cptr(down_gs, 0),
            (const uint32_t *)axnvm_cptr(router_indices, router_indices_offset),
            (const float *)axnvm_cptr(router_weights, router_weights_offset),
            (const float *)axnvm_cptr(input, input_offset),
            (float *)axnvm_mptr(scratch_mid, scratch_mid_offset),
            (float *)axnvm_mptr(out, out_offset),
            stream,
            experts, topk, hidden, expert_hidden);
}
