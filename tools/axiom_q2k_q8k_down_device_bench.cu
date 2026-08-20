#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

#define QK_K 256u
#define Q2_K_BLOCK_BYTES 84u

typedef struct {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16u];
} axiom_q8k_block;

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int init = 0;
    LARGE_INTEGER t;
    if (!init) {
        QueryPerformanceFrequency(&freq);
        init = 1;
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#endif
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v == 0 || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_u32_zero_ok(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_double(double *values, uint32_t n) {
    qsort(values, n, sizeof(double), cmp_double);
    return (n & 1u) ? values[n / 2u] : (values[n / 2u - 1u] + values[n / 2u]) * 0.5;
}

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8u);
}

static void fill_input(float *x, uint32_t cols) {
    for (uint32_t i = 0; i < cols; ++i) {
        const int a = (int)(i % 67u) - 33;
        const int b = (int)((i * 17u + 5u) % 19u) - 9;
        x[i] = (float)a * 0.0015f + (float)b * 0.0002f;
    }
}

static void fill_q2(uint8_t *buf, uint32_t rows, uint32_t cols) {
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t *row = buf + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
            uint8_t *scales = blk;
            uint8_t *qs = blk + 16u;
            const uint32_t base = r * 131u + b * 977u + 17u;
            for (uint32_t il = 0; il < 16u; ++il) {
                const uint32_t scale = 1u + ((base + il * 7u) % 15u);
                const uint32_t minv = (base >> (il & 7u)) % 5u;
                scales[il] = (uint8_t)((minv << 4u) | scale);
            }
            for (uint32_t i = 0; i < 64u; ++i) {
                uint8_t v = 0;
                for (uint32_t k = 0; k < 4u; ++k) {
                    const uint32_t q = (base + i * 13u + k * 29u + (i >> 2u)) & 3u;
                    v |= (uint8_t)(q << (2u * k));
                }
                qs[i] = v;
            }
            put_f16(blk + 80u, 0x3000u);
            put_f16(blk + 82u, 0x2800u);
        }
    }
}

__device__ static uint16_t dev_le16(const uint8_t *p) {
    return *(const uint16_t *)p;
}

__device__ static float dev_f16_to_f32(uint16_t h) {
    __half_raw raw;
    raw.x = h;
    return __half2float(__half(raw));
}

__device__ static float q2_k_dot_f32(
        const uint8_t *__restrict__ row,
        const float *__restrict__ input,
        uint32_t blocks) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float d = dev_f16_to_f32(dev_le16(blk + 80u));
        const float dmin = dev_f16_to_f32(dev_le16(blk + 82u));
        for (uint32_t il = 0; il < 16u; ++il) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4u);
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const uint64_t base = (uint64_t)b * QK_K + chunk * 128u +
                    ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16u; ++i) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * __ldg(input + base + i);
            }
        }
    }
    return acc;
}

__device__ static float q2_k_dot_f32_warp(
        const uint8_t *__restrict__ row,
        const float *__restrict__ input,
        uint32_t blocks) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t pair = lane >> 4u;
    const uint32_t i = lane & 15u;
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float d = dev_f16_to_f32(dev_le16(blk + 80u));
        const float dmin = dev_f16_to_f32(dev_le16(blk + 82u));
        const float *x_ptr = input + (uint64_t)b * QK_K + lane;
        for (uint32_t segment = 0; segment < 8u; ++segment) {
            const uint32_t chunk = segment >> 2u;
            const uint32_t qseg = segment & 3u;
            const uint32_t il = chunk * 8u + qseg * 2u + pair;
            const uint8_t sc = scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4u);
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const uint32_t shift = qseg * 2u;
            const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
            acc += w * __ldg(x_ptr);
            x_ptr += 32u;
        }
    }
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    return acc;
}

__device__ static float q2_k_dot_q8k_ref(
        const uint8_t *__restrict__ row,
        const axiom_q8k_block *__restrict__ input,
        uint32_t blocks) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float dall = input[b].d * dev_f16_to_f32(dev_le16(blk + 80u));
        const float dmin = input[b].d * dev_f16_to_f32(dev_le16(blk + 82u));
        int32_t isum = 0;
        int32_t summs = 0;
        for (uint32_t il = 0; il < 16u; ++il) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = scales[il];
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const int8_t *xq = input[b].qs + il * 16u;
            int32_t qsum = 0;
            for (uint32_t i = 0; i < 16u; ++i) {
                qsum += (int32_t)((q[i] >> shift) & 3u) * (int32_t)__ldg(xq + i);
            }
            isum += (int32_t)(sc & 0x0fu) * qsum;
            summs += (int32_t)input[b].bsums[il] * (int32_t)(sc >> 4u);
        }
        acc += dall * (float)isum - dmin * (float)summs;
    }
    return acc;
}

__device__ __forceinline__ static int32_t pack_q2_4(const uint8_t *q, uint32_t shift, uint32_t lane4) {
    const uint32_t i = lane4 * 4u;
    const uint32_t q0 = (q[i] >> shift) & 3u;
    const uint32_t q1 = (q[i + 1u] >> shift) & 3u;
    const uint32_t q2 = (q[i + 2u] >> shift) & 3u;
    const uint32_t q3 = (q[i + 3u] >> shift) & 3u;
    return (int32_t)(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
}

__device__ __forceinline__ static int32_t pack_q8_4(const int8_t *q, uint32_t lane4) {
    const uint32_t i = lane4 * 4u;
    return (int32_t)((uint32_t)(uint8_t)q[i] |
           ((uint32_t)(uint8_t)q[i + 1u] << 8u) |
           ((uint32_t)(uint8_t)q[i + 2u] << 16u) |
           ((uint32_t)(uint8_t)q[i + 3u] << 24u));
}

__device__ static float q2_k_dot_q8k_qwarp4(
        const uint8_t *__restrict__ row,
        const axiom_q8k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane4,
        uint32_t mask) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float dall = input[b].d * dev_f16_to_f32(dev_le16(blk + 80u));
        const float dmin = input[b].d * dev_f16_to_f32(dev_le16(blk + 82u));
        int32_t isum = 0;
        int32_t summs = 0;
        for (uint32_t il = 0; il < 16u; ++il) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = scales[il];
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const int8_t *xq = input[b].qs + il * 16u;
            const int32_t qsum = __dp4a(pack_q2_4(q, shift, lane4), pack_q8_4(xq, lane4), 0);
            isum += (int32_t)(sc & 0x0fu) * qsum;
            if (lane4 == 0u) {
                summs += (int32_t)input[b].bsums[il] * (int32_t)(sc >> 4u);
            }
        }
        for (uint32_t offset = 2u; offset > 0u; offset >>= 1u) {
            isum += __shfl_down_sync(mask, isum, offset, 4);
        }
        if (lane4 == 0u) acc += dall * (float)isum - dmin * (float)summs;
    }
    return acc;
}

__device__ static float q2_k_dot_q8k_qwarp8(
        const uint8_t *__restrict__ row,
        const axiom_q8k_block *__restrict__ input,
        uint32_t blocks,
        uint32_t lane8,
        uint32_t mask8) {
    const uint32_t half = lane8 >> 2u;
    const uint32_t lane4 = lane8 & 3u;
    const uint32_t mask4 = 0x0fu << ((threadIdx.x & 24u) + half * 4u);
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
        const uint8_t *scales = blk;
        const uint8_t *qs = blk + 16u;
        const float dall = input[b].d * dev_f16_to_f32(dev_le16(blk + 80u));
        const float dmin = input[b].d * dev_f16_to_f32(dev_le16(blk + 82u));
        int32_t isum = 0;
        int32_t summs = 0;
        for (uint32_t il0 = 0; il0 < 16u; il0 += 2u) {
            const uint32_t il = il0 + half;
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = scales[il];
            const uint8_t *q = qs + 32u * chunk + 16u * pair;
            const int8_t *xq = input[b].qs + il * 16u;
            const int32_t qsum = __dp4a(pack_q2_4(q, shift, lane4), pack_q8_4(xq, lane4), 0);
            isum += (int32_t)(sc & 0x0fu) * qsum;
            if (lane4 == 0u) {
                summs += (int32_t)input[b].bsums[il] * (int32_t)(sc >> 4u);
            }
        }
        for (uint32_t offset = 2u; offset > 0u; offset >>= 1u) {
            isum += __shfl_down_sync(mask4, isum, offset, 4);
        }
        isum += __shfl_down_sync(mask8, isum, 4, 8);
        summs += __shfl_down_sync(mask8, summs, 4, 8);
        if (lane8 == 0u) acc += dall * (float)isum - dmin * (float)summs;
    }
    return acc;
}

__global__ static void q8k_pack_f32_kernel(const float *input, axiom_q8k_block *out, uint32_t blocks) {
    const uint32_t b = (uint32_t)blockIdx.x;
    if (b >= blocks) return;
    const uint32_t lane = threadIdx.x;
    __shared__ float absmax[QK_K];
    __shared__ float maxv[QK_K];
    __shared__ float iscale;
    const float x = input[(uint64_t)b * QK_K + lane];
    absmax[lane] = fabsf(x);
    maxv[lane] = x;
    __syncthreads();
    for (uint32_t stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride && absmax[lane + stride] > absmax[lane]) {
            absmax[lane] = absmax[lane + stride];
            maxv[lane] = maxv[lane + stride];
        }
        __syncthreads();
    }
    if (absmax[0] == 0.0f) {
        out[b].qs[lane] = 0;
        if (lane < QK_K / 16u) out[b].bsums[lane] = 0;
        if (lane == 0u) out[b].d = 0.0f;
        return;
    }
    if (lane == 0u) iscale = -127.0f / maxv[0];
    __syncthreads();
    int q = (int)lrintf(iscale * x);
    q = q < -128 ? -128 : q > 127 ? 127 : q;
    out[b].qs[lane] = (int8_t)q;
    __syncthreads();
    if (lane < QK_K / 16u) {
        int sum = 0;
        for (uint32_t i = 0; i < 16u; ++i) sum += out[b].qs[lane * 16u + i];
        out[b].bsums[lane] = (int16_t)sum;
    }
    if (lane == 0u) out[b].d = 1.0f / iscale;
}

__global__ static void q2k_f32_scalar_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    out[row] = q2_k_dot_f32(weight + (uint64_t)row * row_bytes, input, blocks);
}

__global__ static void q2k_f32_warp_kernel(
        const uint8_t *weight,
        const float *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    const float acc = q2_k_dot_f32_warp(weight + (uint64_t)row * row_bytes, input, blocks);
    if (lane == 0u) out[row] = acc;
}

__global__ static void q2k_q8k_ref_kernel(
        const uint8_t *weight,
        const axiom_q8k_block *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    out[row] = q2_k_dot_q8k_ref(weight + (uint64_t)row * row_bytes, input, blocks);
}

__global__ static void q2k_q8k_dp4a_kernel(
        const uint8_t *weight,
        const axiom_q8k_block *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 2u;
    const uint32_t lane4 = lane & 3u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t rows_per_block = warps_per_block * 8u;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + warp * 8u + qwarp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    const uint32_t mask = 0x0fu << (qwarp * 4u);
    const float acc = q2_k_dot_q8k_qwarp4(weight + (uint64_t)row * row_bytes, input, blocks, lane4, mask);
    if (lane4 == 0u) out[row] = acc;
}

__global__ static void q2k_q8k_qwarp8_kernel(
        const uint8_t *weight,
        const axiom_q8k_block *input,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t qwarp = lane >> 3u;
    const uint32_t lane8 = lane & 7u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t rows_per_block = warps_per_block * 4u;
    const uint32_t row = (uint32_t)blockIdx.x * rows_per_block + warp * 4u + qwarp;
    if (row >= rows) return;
    const uint32_t blocks = cols / QK_K;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    const uint32_t mask = 0xffu << (qwarp * 8u);
    const float acc = q2_k_dot_q8k_qwarp8(weight + (uint64_t)row * row_bytes, input, blocks, lane8, mask);
    if (lane8 == 0u) out[row] = acc;
}

static int check_cuda(cudaError_t err, const char *where) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(err));
    return 0;
}

static double run_f32_scalar(
        const uint8_t *w,
        const float *x,
        float *y,
        uint32_t rows,
        uint32_t cols,
        uint32_t reps,
        uint32_t block) {
    const int grid = (int)((rows + block - 1u) / block);
    const double t0 = now_ms();
    for (uint32_t i = 0; i < reps; ++i) {
        q2k_f32_scalar_kernel<<<grid, (int)block>>>(w, x, y, rows, cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1.0;
    const double t1 = now_ms();
    return ((double)reps * 1000.0) / (t1 - t0);
}

static double run_f32_warp(
        const uint8_t *w,
        const float *x,
        float *y,
        uint32_t rows,
        uint32_t cols,
        uint32_t reps,
        uint32_t block) {
    const uint32_t warps = block >> 5u;
    const int grid = (int)((rows + warps - 1u) / warps);
    const double t0 = now_ms();
    for (uint32_t i = 0; i < reps; ++i) {
        q2k_f32_warp_kernel<<<grid, (int)block>>>(w, x, y, rows, cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1.0;
    const double t1 = now_ms();
    return ((double)reps * 1000.0) / (t1 - t0);
}

static double run_q8k_dp4a(
        const uint8_t *w,
        const axiom_q8k_block *x,
        float *y,
        uint32_t rows,
        uint32_t cols,
        uint32_t reps,
        uint32_t block) {
    const uint32_t warps = block >> 5u;
    const uint32_t rows_per_block = warps * 8u;
    const int grid = (int)((rows + rows_per_block - 1u) / rows_per_block);
    const double t0 = now_ms();
    for (uint32_t i = 0; i < reps; ++i) {
        q2k_q8k_dp4a_kernel<<<grid, (int)block>>>(w, x, y, rows, cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1.0;
    const double t1 = now_ms();
    return ((double)reps * 1000.0) / (t1 - t0);
}

static double run_q8k_qwarp8(
        const uint8_t *w,
        const axiom_q8k_block *x,
        float *y,
        uint32_t rows,
        uint32_t cols,
        uint32_t reps,
        uint32_t block) {
    const uint32_t warps = block >> 5u;
    const uint32_t rows_per_block = warps * 4u;
    const int grid = (int)((rows + rows_per_block - 1u) / rows_per_block);
    const double t0 = now_ms();
    for (uint32_t i = 0; i < reps; ++i) {
        q2k_q8k_qwarp8_kernel<<<grid, (int)block>>>(w, x, y, rows, cols);
    }
    cudaError_t err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -1.0;
    const double t1 = now_ms();
    return ((double)reps * 1000.0) / (t1 - t0);
}

static void compare_outputs(
        const float *a,
        const float *b,
        uint32_t rows,
        double *mean_abs,
        double *rmse,
        double *rel_l2,
        double *cosine,
        float *max_abs,
        uint32_t *max_idx) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double a_sq = 0.0;
    double b_sq = 0.0;
    double dot = 0.0;
    *max_abs = 0.0f;
    *max_idx = 0;
    for (uint32_t i = 0; i < rows; ++i) {
        const double d = (double)b[i] - (double)a[i];
        const double ad = fabs(d);
        if (ad > (double)*max_abs) {
            *max_abs = (float)ad;
            *max_idx = i;
        }
        sum_abs += ad;
        sum_sq += d * d;
        a_sq += (double)a[i] * (double)a[i];
        b_sq += (double)b[i] * (double)b[i];
        dot += (double)a[i] * (double)b[i];
    }
    *mean_abs = sum_abs / (double)rows;
    *rmse = sqrt(sum_sq / (double)rows);
    *rel_l2 = a_sq > 0.0 ? sqrt(sum_sq / a_sq) : 0.0;
    *cosine = (a_sq > 0.0 && b_sq > 0.0) ? dot / (sqrt(a_sq) * sqrt(b_sq)) : 0.0;
}

int main(int argc, char **argv) {
    uint32_t rows = 4096;
    uint32_t cols = 2048;
    uint32_t reps = 200;
    uint32_t warmup = 10;
    uint32_t samples = 5;
    uint32_t reps_per_sample = 0;
    if (argc > 1 && !parse_u32(argv[1], &rows)) return 2;
    if (argc > 2 && !parse_u32(argv[2], &cols)) return 2;
    if (argc > 3 && !parse_u32(argv[3], &reps)) return 2;
    if (argc > 4 && !parse_u32_zero_ok(argv[4], &warmup)) return 2;
    if (argc > 5 && !parse_u32(argv[5], &samples)) return 2;
    if (argc > 6 && !parse_u32(argv[6], &reps_per_sample)) return 2;
    if (argc > 7 || rows == 0 || cols == 0 || (cols % QK_K) != 0 || samples == 0) return 2;
    if (reps_per_sample == 0) reps_per_sample = reps;

    const uint32_t blocks = cols / QK_K;
    const size_t row_bytes = (size_t)blocks * Q2_K_BLOCK_BYTES;
    const size_t weight_bytes = (size_t)rows * row_bytes;
    const size_t input_bytes = (size_t)cols * sizeof(float);
    const size_t out_bytes = (size_t)rows * sizeof(float);
    const size_t q8k_bytes = (size_t)blocks * sizeof(axiom_q8k_block);

    uint8_t *weights = (uint8_t *)malloc(weight_bytes);
    float *input = (float *)malloc(input_bytes);
    float *base = (float *)malloc(out_bytes);
    float *warp = (float *)malloc(out_bytes);
    float *qref = (float *)malloc(out_bytes);
    float *cand = (float *)malloc(out_bytes);
    float *alt = (float *)malloc(out_bytes);
    double *scalar_samples = (double *)malloc((size_t)samples * sizeof(double));
    double *warp_samples = (double *)malloc((size_t)samples * sizeof(double));
    double *cand_samples = (double *)malloc((size_t)samples * sizeof(double));
    double *alt_samples = (double *)malloc((size_t)samples * sizeof(double));
    if (!weights || !input || !base || !warp || !qref || !cand || !alt ||
        !scalar_samples || !warp_samples || !cand_samples || !alt_samples) {
        fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    fill_q2(weights, rows, cols);
    fill_input(input, cols);

    uint8_t *weights_dev = NULL;
    float *input_dev = NULL;
    float *base_dev = NULL;
    float *warp_dev = NULL;
    float *qref_dev = NULL;
    float *cand_dev = NULL;
    float *alt_dev = NULL;
    axiom_q8k_block *q8k_dev = NULL;

    cudaError_t err = cudaMalloc(&weights_dev, weight_bytes);
    err = err == cudaSuccess ? cudaMalloc(&input_dev, input_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&base_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&warp_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&qref_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&cand_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&alt_dev, out_bytes) : err;
    err = err == cudaSuccess ? cudaMalloc(&q8k_dev, q8k_bytes) : err;
    err = err == cudaSuccess ? cudaMemcpy(weights_dev, weights, weight_bytes, cudaMemcpyHostToDevice) : err;
    err = err == cudaSuccess ? cudaMemcpy(input_dev, input, input_bytes, cudaMemcpyHostToDevice) : err;
    if (!check_cuda(err, "setup")) return 1;

    q8k_pack_f32_kernel<<<blocks, 256>>>(input_dev, q8k_dev, blocks);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (!check_cuda(err, "q8k_pack")) return 1;

    const uint32_t scalar_block = 256;
    const uint32_t warp_block = 256;
    const uint32_t cand_block = 128;
    const uint32_t alt_block = 128;
    const uint32_t scalar_grid = (rows + scalar_block - 1u) / scalar_block;
    const uint32_t warp_grid = (rows + (warp_block >> 5u) - 1u) / (warp_block >> 5u);
    const uint32_t cand_grid = (rows + ((cand_block >> 5u) * 8u) - 1u) / ((cand_block >> 5u) * 8u);
    const uint32_t alt_grid = (rows + ((alt_block >> 5u) * 4u) - 1u) / ((alt_block >> 5u) * 4u);

    for (uint32_t i = 0; i < warmup; ++i) {
        q2k_f32_scalar_kernel<<<(int)scalar_grid, (int)scalar_block>>>(weights_dev, input_dev, base_dev, rows, cols);
        q2k_f32_warp_kernel<<<(int)warp_grid, (int)warp_block>>>(weights_dev, input_dev, warp_dev, rows, cols);
        q2k_q8k_dp4a_kernel<<<(int)cand_grid, (int)cand_block>>>(weights_dev, q8k_dev, cand_dev, rows, cols);
        q2k_q8k_qwarp8_kernel<<<(int)alt_grid, (int)alt_block>>>(weights_dev, q8k_dev, alt_dev, rows, cols);
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (!check_cuda(err, "warmup")) return 1;

    for (uint32_t s = 0; s < samples; ++s) {
        scalar_samples[s] = run_f32_scalar(weights_dev, input_dev, base_dev, rows, cols, reps_per_sample, scalar_block);
        warp_samples[s] = run_f32_warp(weights_dev, input_dev, warp_dev, rows, cols, reps_per_sample, warp_block);
        cand_samples[s] = run_q8k_dp4a(weights_dev, q8k_dev, cand_dev, rows, cols, reps_per_sample, cand_block);
        alt_samples[s] = run_q8k_qwarp8(weights_dev, q8k_dev, alt_dev, rows, cols, reps_per_sample, alt_block);
        if (scalar_samples[s] < 0.0 || warp_samples[s] < 0.0 || cand_samples[s] < 0.0 || alt_samples[s] < 0.0) {
            fprintf(stderr, "benchmark launch failed\n");
            return 1;
        }
    }

    const double pack_t0 = now_ms();
    for (uint32_t i = 0; i < reps_per_sample; ++i) {
        q8k_pack_f32_kernel<<<blocks, 256>>>(input_dev, q8k_dev, blocks);
    }
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (!check_cuda(err, "pack_bench")) return 1;
    const double pack_t1 = now_ms();
    const double pack_iter_s = ((double)reps_per_sample * 1000.0) / (pack_t1 - pack_t0);

    q2k_f32_scalar_kernel<<<(int)scalar_grid, (int)scalar_block>>>(weights_dev, input_dev, base_dev, rows, cols);
    q2k_f32_warp_kernel<<<(int)warp_grid, (int)warp_block>>>(weights_dev, input_dev, warp_dev, rows, cols);
    q2k_q8k_ref_kernel<<<(int)scalar_grid, (int)scalar_block>>>(weights_dev, q8k_dev, qref_dev, rows, cols);
    q2k_q8k_dp4a_kernel<<<(int)cand_grid, (int)cand_block>>>(weights_dev, q8k_dev, cand_dev, rows, cols);
    q2k_q8k_qwarp8_kernel<<<(int)alt_grid, (int)alt_block>>>(weights_dev, q8k_dev, alt_dev, rows, cols);
    err = cudaGetLastError();
    if (err == cudaSuccess) err = cudaDeviceSynchronize();
    if (!check_cuda(err, "final_kernels")) return 1;

    err = cudaMemcpy(base, base_dev, out_bytes, cudaMemcpyDeviceToHost);
    err = err == cudaSuccess ? cudaMemcpy(warp, warp_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    err = err == cudaSuccess ? cudaMemcpy(qref, qref_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    err = err == cudaSuccess ? cudaMemcpy(cand, cand_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    err = err == cudaSuccess ? cudaMemcpy(alt, alt_dev, out_bytes, cudaMemcpyDeviceToHost) : err;
    if (!check_cuda(err, "download")) return 1;

    double warp_mean_abs = 0.0;
    double warp_rmse = 0.0;
    double warp_rel_l2 = 0.0;
    double warp_cosine = 0.0;
    float warp_max_abs = 0.0f;
    uint32_t warp_max_idx = 0;
    compare_outputs(base, warp, rows, &warp_mean_abs, &warp_rmse, &warp_rel_l2,
                    &warp_cosine, &warp_max_abs, &warp_max_idx);

    double fp32_mean_abs = 0.0;
    double fp32_rmse = 0.0;
    double fp32_rel_l2 = 0.0;
    double fp32_cosine = 0.0;
    float fp32_max_abs = 0.0f;
    uint32_t fp32_max_idx = 0;
    compare_outputs(base, cand, rows, &fp32_mean_abs, &fp32_rmse, &fp32_rel_l2,
                    &fp32_cosine, &fp32_max_abs, &fp32_max_idx);

    double q8_mean_abs = 0.0;
    double q8_rmse = 0.0;
    double q8_rel_l2 = 0.0;
    double q8_cosine = 0.0;
    float q8_max_abs = 0.0f;
    uint32_t q8_max_idx = 0;
    compare_outputs(qref, cand, rows, &q8_mean_abs, &q8_rmse, &q8_rel_l2,
                    &q8_cosine, &q8_max_abs, &q8_max_idx);

    double alt_q8_mean_abs = 0.0;
    double alt_q8_rmse = 0.0;
    double alt_q8_rel_l2 = 0.0;
    double alt_q8_cosine = 0.0;
    float alt_q8_max_abs = 0.0f;
    uint32_t alt_q8_max_idx = 0;
    compare_outputs(qref, alt, rows, &alt_q8_mean_abs, &alt_q8_rmse, &alt_q8_rel_l2,
                    &alt_q8_cosine, &alt_q8_max_abs, &alt_q8_max_idx);

    const double scalar_med = median_double(scalar_samples, samples);
    const double warp_med = median_double(warp_samples, samples);
    const double cand_med = median_double(cand_samples, samples);
    const double alt_med = median_double(alt_samples, samples);
    const double elems = (double)rows * (double)cols;
    const double scalar_gelem_s = scalar_med * elems / 1000000000.0;
    const double warp_gelem_s = warp_med * elems / 1000000000.0;
    const double cand_gelem_s = cand_med * elems / 1000000000.0;
    const double alt_gelem_s = alt_med * elems / 1000000000.0;

    const int warp_ok = isfinite(warp_cosine) && warp_max_abs <= 0.0001f && warp_rel_l2 <= 0.000001;
    const int q8_ok = isfinite(q8_cosine) && q8_max_abs <= 0.002f && q8_rel_l2 <= 0.00005 && q8_cosine >= 0.999999;
    const int alt_q8_ok = isfinite(alt_q8_cosine) && alt_q8_max_abs <= 0.002f && alt_q8_rel_l2 <= 0.00005 && alt_q8_cosine >= 0.999999;
    const int fp32_ok = isfinite(fp32_cosine) && fp32_rel_l2 <= 0.02 && fp32_cosine >= 0.9995;
    const int ok = warp_ok && q8_ok && alt_q8_ok;

    printf("q2k_q8k_down rows=%u cols=%u reps=%u warmup=%u samples=%u reps_per_sample=%u "
           "f32_scalar_iter_s=%.3f f32_warp_iter_s=%.3f q8k_dp4a_iter_s=%.3f q8k_qwarp8_iter_s=%.3f pack_iter_s=%.3f "
           "speedup_vs_f32_scalar=%.3f speedup_vs_f32_warp=%.3f qwarp8_speedup_vs_f32_warp=%.3f qwarp8_speedup_vs_q8k_dp4a=%.3f "
           "f32_scalar_gelem_s=%.3f f32_warp_gelem_s=%.3f q8k_dp4a_gelem_s=%.3f q8k_qwarp8_gelem_s=%.3f "
           "warp_max_abs=%.9f warp_rel_l2=%.9f warp_cosine=%.9f warp_status=%s "
           "fp32_max_abs=%.9f fp32_mean_abs=%.9f fp32_rmse=%.9f fp32_rel_l2=%.9f fp32_cosine=%.9f fp32_max_idx=%u base=%.9f cand=%.9f fp32_status=%s "
           "q8_max_abs=%.9f q8_mean_abs=%.9f q8_rmse=%.9f q8_rel_l2=%.9f q8_cosine=%.9f q8_max_idx=%u qref=%.9f qcand=%.9f q8_status=%s "
           "qwarp8_q8_max_abs=%.9f qwarp8_q8_mean_abs=%.9f qwarp8_q8_rmse=%.9f qwarp8_q8_rel_l2=%.9f qwarp8_q8_cosine=%.9f qwarp8_q8_max_idx=%u qref_alt=%.9f qwarp8=%.9f qwarp8_q8_status=%s status=%s\n",
           rows, cols, reps, warmup, samples, reps_per_sample,
           scalar_med, warp_med, cand_med, alt_med, pack_iter_s,
           scalar_med > 0.0 ? cand_med / scalar_med : 0.0,
           warp_med > 0.0 ? cand_med / warp_med : 0.0,
           warp_med > 0.0 ? alt_med / warp_med : 0.0,
           cand_med > 0.0 ? alt_med / cand_med : 0.0,
           scalar_gelem_s, warp_gelem_s, cand_gelem_s, alt_gelem_s,
           warp_max_abs, warp_rel_l2, warp_cosine, warp_ok ? "pass" : "warn",
           fp32_max_abs, fp32_mean_abs, fp32_rmse, fp32_rel_l2, fp32_cosine,
           fp32_max_idx, base[fp32_max_idx], cand[fp32_max_idx],
           fp32_ok ? "pass" : "warn",
           q8_max_abs, q8_mean_abs, q8_rmse, q8_rel_l2, q8_cosine,
           q8_max_idx, qref[q8_max_idx], cand[q8_max_idx],
           q8_ok ? "pass" : "warn",
           alt_q8_max_abs, alt_q8_mean_abs, alt_q8_rmse, alt_q8_rel_l2, alt_q8_cosine,
           alt_q8_max_idx, qref[alt_q8_max_idx], alt[alt_q8_max_idx],
           alt_q8_ok ? "pass" : "warn",
           ok ? "pass" : "warn");

    cudaFree(q8k_dev);
    cudaFree(alt_dev);
    cudaFree(cand_dev);
    cudaFree(qref_dev);
    cudaFree(warp_dev);
    cudaFree(base_dev);
    cudaFree(input_dev);
    cudaFree(weights_dev);
    free(alt_samples);
    free(cand_samples);
    free(warp_samples);
    free(scalar_samples);
    free(alt);
    free(cand);
    free(qref);
    free(warp);
    free(base);
    free(input);
    free(weights);
    return ok ? 0 : 3;
}
