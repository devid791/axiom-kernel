/* P2: PRECISE (non-fast-math) compile unit. Compiled WITHOUT --use_fast_math (see the Makefile
 * rule for axiom_cuda_precise.o) so powf/cosf/sinf are precise libm — byte-close to the host
 * rope_neox (tools/axiom_qwen.cpp:540-551). The fast-math trig in the main (--use_fast_math)
 * axiom_cuda.cu made the device attention core systematically oracle-worse (~86% vs host). This
 * unit is self-contained: it takes a RAW device pointer + cuda device id (no libaxiom internal
 * structs), so it needs no shared header; the axiom_cuda.cu thunk extracts the pointer + delegates.
 *
 * Dense NeoX RoPE: half-split pairing (ic, ic+head_dim/2); one block per head, grid-stride over
 * ic; no reduction and each ic touches only {ic, ic+half} so in-place is safe. */
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

static bool axiom_precise_env_enabled(const char *name) {
    const char *v = std::getenv(name);
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0 &&
           std::strcmp(v, "false") != 0 && std::strcmp(v, "False") != 0 &&
           std::strcmp(v, "FALSE") != 0;
}

static bool axiom_precise_skip_sync(void) {
    if (axiom_precise_env_enabled("AXIOM_CUDA_LAUNCH_ASYNC")) return true;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t err = cudaStreamIsCapturing(cudaStreamPerThread, &status);
    if (err == cudaSuccess && status != cudaStreamCaptureStatusNone) return true;
    status = cudaStreamCaptureStatusNone;
    err = cudaStreamIsCapturing((cudaStream_t)0, &status);
    return err == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

static int axiom_precise_finish(cudaError_t err) {
    if (err != cudaSuccess) return -2;
    if (axiom_precise_skip_sync()) return 0;
    err = cudaDeviceSynchronize();
    return err == cudaSuccess ? 0 : -3;
}

__global__ static void axiom_qwen_rope_neox_precise_kernel(
        float *x, uint32_t head_dim, uint32_t pos, float theta) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t half = head_dim / 2u;
    float *row = x + (uint64_t)head * head_dim;
    for (uint32_t ic = (uint32_t)threadIdx.x; ic < half; ic += blockDim.x) {
        const float freq = powf(theta, -2.0f * (float)ic / (float)head_dim);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = row[ic], x1 = row[ic + half];
        row[ic]        = x0 * c - x1 * s;
        row[ic + half] = x0 * s + x1 * c;
    }
}

/* device = cuda device id; x = device pointer (already offset by the caller). Returns 0 on
 * success, negative on a CUDA error. */
extern "C" int axiom_rope_neox_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t pos, float theta) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    axiom_qwen_rope_neox_precise_kernel<<<heads, 256>>>(x, head_dim, pos, theta);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}

/* P2 STEP 7 (Nemotron): PARTIAL NeoX RoPE — rotate only the first n_rot dims (pairs ic,
 * ic+n_rot/2) and PASS THROUGH the rest [n_rot, head_dim). half = n_rot/2 (NOT head_dim/2)
 * and the freq denominator is n_rot — byte-faithful to the host rope_neox_partial
 * (tools/axiom_nemotron.cpp:144). Dims [n_rot, head_dim) are never written, so in-place is
 * safe and the pass-through dims keep their values exactly. Nemotron has NO qk-norm, so this
 * is a standalone rope (unlike the qwen fused qknorm+rope). One block per head, grid-stride
 * over ic; precise libm (non-fast-math unit) matches the host trig. */
__global__ static void axiom_nemotron_rope_neox_partial_precise_kernel(
        float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, float theta) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t half = n_rot / 2u;
    float *row = x + (uint64_t)head * head_dim;
    for (uint32_t ic = (uint32_t)threadIdx.x; ic < half; ic += blockDim.x) {
        const float freq = powf(theta, -2.0f * (float)ic / (float)n_rot);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = row[ic], x1 = row[ic + half];
        row[ic]        = x0 * c - x1 * s;
        row[ic + half] = x0 * s + x1 * c;
    }
}

/* device = cuda device id; x = device pointer (already offset by the caller). Returns 0 on
 * success, negative on a CUDA error. */
extern "C" int axiom_rope_neox_partial_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos, float theta) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    axiom_nemotron_rope_neox_partial_precise_kernel<<<heads, 256>>>(x, head_dim, n_rot, pos, theta);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}

/* P2 launch-fusion: per-head QK-RMSNorm(+weight) FUSED with the NeoX rope in ONE launch
 * (block = head_dim, one thread per dim). Replaces the separate qk_rmsnorm_w + rope_neox kernels
 * (2 launches -> 1 per tensor) to cut the device_attn per-layer launch count -> flips the small-
 * model launch-bound slowdown toward a universal win. qk-norm uses DOUBLE accumulation (matches
 * the host qk_norm); rope uses precise libm (non-fast-math unit). head_dim must be a power of two
 * and <= 1024. */
__global__ static void axiom_qwen_qknorm_rope_precise_kernel(
        float *x, const float *w, uint32_t head_dim, uint32_t pos, float theta, float eps) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    float *row = x + (uint64_t)head * head_dim;
    extern __shared__ double dsm[];                 /* [head_dim] double reduce */
    float *xs = (float *)(dsm + head_dim);          /* [head_dim] qk-normed values */
    const float xv = row[tid];
    dsm[tid] = (double)xv * (double)xv;
    __syncthreads();
    for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
        if (tid < s) dsm[tid] += dsm[tid + s];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(dsm[0] / (double)head_dim + (double)eps));
    xs[tid] = xv * inv * w[tid];                     /* qk-normed value for dim tid */
    __syncthreads();
    const uint32_t half = head_dim >> 1;
    if (tid < half) {
        const float freq = powf(theta, -2.0f * (float)tid / (float)head_dim);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = xs[tid], x1 = xs[tid + half];
        row[tid]        = x0 * c - x1 * s;
        row[tid + half] = x0 * s + x1 * c;
    }
}

__global__ static void axiom_qwen_qknorm_rope_pos_precise_kernel(
        float *x, const float *w, const uint32_t *pos_ptr,
        uint32_t head_dim, float theta, float eps) {
    const uint32_t pos = pos_ptr[0];
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    float *row = x + (uint64_t)head * head_dim;
    extern __shared__ double dsm[];
    float *xs = (float *)(dsm + head_dim);
    const float xv = row[tid];
    dsm[tid] = (double)xv * (double)xv;
    __syncthreads();
    for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
        if (tid < s) dsm[tid] += dsm[tid + s];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(dsm[0] / (double)head_dim + (double)eps));
    xs[tid] = xv * inv * w[tid];
    __syncthreads();
    const uint32_t half = head_dim >> 1;
    if (tid < half) {
        const float freq = powf(theta, -2.0f * (float)tid / (float)head_dim);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = xs[tid], x1 = xs[tid + half];
        row[tid]        = x0 * c - x1 * s;
        row[tid + half] = x0 * s + x1 * c;
    }
}

__global__ static void axiom_qwen_qknorm_rope_pos_dual_precise_kernel(
        float *q, const float *qw,
        float *k, const float *kw,
        const uint32_t *pos_ptr,
        uint32_t q_heads, uint32_t head_dim, float theta, float eps) {
    const uint32_t ghead = (uint32_t)blockIdx.x;
    const uint32_t tid = (uint32_t)threadIdx.x;
    const int is_q = ghead < q_heads;
    const uint32_t head = is_q ? ghead : (ghead - q_heads);
    float *row = (is_q ? q : k) + (uint64_t)head * head_dim;
    const float *w = is_q ? qw : kw;
    const uint32_t pos = pos_ptr[0];
    extern __shared__ double dsm[];
    float *xs = (float *)(dsm + head_dim);
    const float xv = row[tid];
    dsm[tid] = (double)xv * (double)xv;
    __syncthreads();
    for (uint32_t s = head_dim >> 1; s > 0u; s >>= 1u) {
        if (tid < s) dsm[tid] += dsm[tid + s];
        __syncthreads();
    }
    const float inv = (float)(1.0 / sqrt(dsm[0] / (double)head_dim + (double)eps));
    xs[tid] = xv * inv * w[tid];
    __syncthreads();
    const uint32_t half = head_dim >> 1;
    if (tid < half) {
        const float freq = powf(theta, -2.0f * (float)tid / (float)head_dim);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = xs[tid], x1 = xs[tid + half];
        row[tid]        = x0 * c - x1 * s;
        row[tid + half] = x0 * s + x1 * c;
    }
}

extern "C" int axiom_qknorm_rope_precise_launch(
        int device, float *x, const float *w, uint32_t heads, uint32_t head_dim,
        uint32_t pos, float theta, float eps) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    const size_t shbytes = (size_t)head_dim * sizeof(double) + (size_t)head_dim * sizeof(float);
    axiom_qwen_qknorm_rope_precise_kernel<<<heads, head_dim, shbytes>>>(x, w, head_dim, pos, theta, eps);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}

extern "C" int axiom_qknorm_rope_pos_precise_launch(
        int device, float *x, const float *w, const uint32_t *pos,
        uint32_t heads, uint32_t head_dim, float theta, float eps) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    const size_t shbytes = (size_t)head_dim * sizeof(double) + (size_t)head_dim * sizeof(float);
    axiom_qwen_qknorm_rope_pos_precise_kernel<<<heads, head_dim, shbytes>>>(
            x, w, pos, head_dim, theta, eps);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}

extern "C" int axiom_qknorm_rope_pos_dual_precise_launch(
        int device,
        float *q, const float *qw,
        float *k, const float *kw,
        const uint32_t *pos,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim,
        float theta, float eps) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    const size_t shbytes = (size_t)head_dim * sizeof(double) + (size_t)head_dim * sizeof(float);
    axiom_qwen_qknorm_rope_pos_dual_precise_kernel<<<q_heads + k_heads, head_dim, shbytes>>>(
            q, qw, k, kw, pos, q_heads, head_dim, theta, eps);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}

/* P2 STEP 7 (Gemma-4): DECOUPLED partial NeoX RoPE. Like the Nemotron partial rope above, but
 * the rotated-pair COUNT (active_pairs) and the freq DENOMINATOR (exp_dim) are SEPARATE params —
 * gemma-4's GLOBAL layers rotate only 64 pairs yet use a denominator of 512 (active_pairs != exp_dim),
 * whereas the Nemotron kernel ties both to n_rot. Pair (ic, ic+head_dim/2); rotate ic in
 * [0, min(active_pairs, head_dim/2)) with freq = theta^(-2*ic/exp_dim); pass the rest through
 * unchanged. Byte-faithful to the host rope() in tools/axiom_gemma4.cpp (half = head_dim/2, NOT
 * active_pairs/2). One block per head, grid-stride over ic; precise libm (non-fast-math unit). */
__global__ static void axiom_gemma4_rope_decoupled_partial_precise_kernel(
        float *x, uint32_t head_dim, uint32_t active_pairs, uint32_t exp_dim,
        uint32_t pos, float theta) {
    const uint32_t head = (uint32_t)blockIdx.x;
    const uint32_t half = head_dim / 2u;
    float *row = x + (uint64_t)head * head_dim;
    for (uint32_t ic = (uint32_t)threadIdx.x; ic < active_pairs && ic < half; ic += blockDim.x) {
        const float freq = powf(theta, -2.0f * (float)ic / (float)exp_dim);
        const float ang = (float)pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x0 = row[ic], x1 = row[ic + half];
        row[ic]        = x0 * c - x1 * s;
        row[ic + half] = x0 * s + x1 * c;
    }
}

/* device = cuda device id; x = device pointer (already offset by the caller). Returns 0 on
 * success, negative on a CUDA error. */
extern "C" int axiom_rope_neox_decoupled_partial_precise_launch(
        int device, float *x, uint32_t heads, uint32_t head_dim, uint32_t active_pairs,
        uint32_t exp_dim, uint32_t pos, float theta) {
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return -1;
    axiom_gemma4_rope_decoupled_partial_precise_kernel<<<heads, 256>>>(
            x, head_dim, active_pairs, exp_dim, pos, theta);
    err = cudaGetLastError();
    return axiom_precise_finish(err);
}
