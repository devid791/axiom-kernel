#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float d;
    int8_t qs[32];
} q8_tile32;

static void cuda_check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda_error %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static void fill_q8(uint8_t *w, uint32_t rows, uint32_t cols) {
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *blk = w + (uint64_t)r * row_bytes + (uint64_t)b * 34u;
            blk[0] = 0;
            blk[1] = 0x3c;
            for (uint32_t i = 0; i < 32u; ++i) {
                blk[2u + i] = (uint8_t)((int)((r * 17u + b * 13u + i * 7u) & 255u) - 128);
            }
        }
    }
}

static void fill_xq(q8_tile32 *x, uint32_t blocks) {
    for (uint32_t b = 0; b < blocks; ++b) {
        x[b].d = 1.0f / 127.0f;
        for (uint32_t i = 0; i < 32u; ++i) {
            x[b].qs[i] = (int8_t)((int)((b * 11u + i * 5u) & 255u) - 128);
        }
    }
}

__device__ __forceinline__ static uint16_t le16(const uint8_t *p) {
    return *(const uint16_t *)p;
}

__device__ __forceinline__ static float f16_to_f32(uint16_t h) {
    __half_raw raw;
    raw.x = h;
    return __half2float(__half(raw));
}

__device__ __forceinline__ static int32_t pack4(const int8_t *p, uint32_t off) {
    const uint8_t *q = (const uint8_t *)(p + off);
    const uint32_t v = (uint32_t)q[0] |
            ((uint32_t)q[1] << 8u) |
            ((uint32_t)q[2] << 16u) |
            ((uint32_t)q[3] << 24u);
    return (int32_t)v;
}

__device__ __forceinline__ static int32_t dot_i8x32(const int8_t *w, const int8_t *x) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        sum = __dp4a(pack4(w, i), pack4(x, i), sum);
    }
    return sum;
}

__global__ static void q8_load_only_kernel(
        const uint8_t *__restrict__ weight,
        const q8_tile32 *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const q8_tile32 *x = input + b;
        int32_t sum = (int32_t)le16(blk);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            sum += pack4(wq, i) ^ pack4(x->qs, i);
        }
        acc += (float)sum * x->d;
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[row] = acc;
}

__global__ static void q8_dot_kernel(
        const uint8_t *__restrict__ weight,
        const q8_tile32 *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const uint32_t blocks = cols / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    const uint8_t *r = weight + (uint64_t)row * row_bytes;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 32u) {
        const uint8_t *blk = r + (uint64_t)b * 34u;
        const float wd = f16_to_f32(le16(blk));
        const int8_t *wq = (const int8_t *)(blk + 2u);
        const q8_tile32 *x = input + b;
        acc += wd * x->d * (float)dot_i8x32(wq, x->qs);
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0u) out[row] = acc;
}

static float run_ms(
        void (*kernel)(const uint8_t *, const q8_tile32 *, float *, uint32_t, uint32_t),
        const uint8_t *w,
        const q8_tile32 *x,
        float *out,
        uint32_t rows,
        uint32_t cols,
        uint32_t iters) {
    const int block = 256;
    const int warps = block / 32;
    const int grid = (int)((rows + (uint32_t)warps - 1u) / (uint32_t)warps);
    cudaEvent_t a, b;
    cuda_check(cudaEventCreate(&a), "event_create_a");
    cuda_check(cudaEventCreate(&b), "event_create_b");
    for (uint32_t i = 0; i < 8u; ++i) {
        kernel<<<grid, block>>>(w, x, out, rows, cols);
    }
    cuda_check(cudaDeviceSynchronize(), "warmup");
    cuda_check(cudaEventRecord(a), "event_record_a");
    for (uint32_t i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(w, x, out, rows, cols);
    }
    cuda_check(cudaEventRecord(b), "event_record_b");
    cuda_check(cudaEventSynchronize(b), "event_sync_b");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / (float)iters;
}

static void bench_shape(const char *name, uint32_t rows, uint32_t cols, uint32_t iters) {
    const uint32_t blocks = cols / 32u;
    const uint64_t weight_bytes = (uint64_t)rows * blocks * 34u;
    const uint64_t input_bytes = (uint64_t)blocks * sizeof(q8_tile32);
    uint8_t *wh = (uint8_t *)malloc((size_t)weight_bytes);
    q8_tile32 *xh = (q8_tile32 *)malloc((size_t)input_bytes);
    if (!wh || !xh) {
        fprintf(stderr, "host_alloc_failed\n");
        exit(1);
    }
    fill_q8(wh, rows, cols);
    fill_xq(xh, blocks);

    uint8_t *wd = NULL;
    q8_tile32 *xd = NULL;
    float *out = NULL;
    cuda_check(cudaMalloc(&wd, (size_t)weight_bytes), "malloc_w");
    cuda_check(cudaMalloc(&xd, (size_t)input_bytes), "malloc_x");
    cuda_check(cudaMalloc(&out, (size_t)rows * sizeof(float)), "malloc_out");
    cuda_check(cudaMemcpy(wd, wh, (size_t)weight_bytes, cudaMemcpyHostToDevice), "copy_w");
    cuda_check(cudaMemcpy(xd, xh, (size_t)input_bytes, cudaMemcpyHostToDevice), "copy_x");

    const float load_ms = run_ms(q8_load_only_kernel, wd, xd, out, rows, cols, iters);
    const float dot_ms = run_ms(q8_dot_kernel, wd, xd, out, rows, cols, iters);
    const double bytes_per_iter = (double)weight_bytes + (double)rows * (double)blocks * sizeof(q8_tile32);
    const double load_gbs = bytes_per_iter / ((double)load_ms / 1000.0) / 1.0e9;
    const double dot_gbs = bytes_per_iter / ((double)dot_ms / 1000.0) / 1.0e9;
    printf("{\"shape\":\"%s\",\"rows\":%u,\"cols\":%u,\"iters\":%u,"
           "\"bytes_per_iter\":%.0f,\"load_ms\":%.6f,\"dot_ms\":%.6f,"
           "\"dot_over_load\":%.6f,\"load_gb_s\":%.3f,\"dot_effective_gb_s\":%.3f}\n",
           name, rows, cols, iters, bytes_per_iter, load_ms, dot_ms,
           (double)dot_ms / (double)load_ms, load_gbs, dot_gbs);

    cudaFree(out);
    cudaFree(xd);
    cudaFree(wd);
    free(xh);
    free(wh);
}

int main(int argc, char **argv) {
    uint32_t iters = 1000u;
    if (argc > 1) iters = (uint32_t)strtoul(argv[1], NULL, 10);
    if (iters == 0) iters = 1000u;
    cuda_check(cudaSetDevice(0), "set_device");
    bench_shape("q8_direct_4096x8192", 4096u, 8192u, iters);
    bench_shape("q8_grouped_flat_8192x4096", 8192u, 4096u, iters);
    return 0;
}
