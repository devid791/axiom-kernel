/* axiom_e2m1_probe.cu — NVFP4 production-decode proof for GB10 (sm_121).
 *
 * Proves the EXACT DeepSeek-V4-Flash-NVFP4 routed-expert decode on Blackwell before
 * wiring into libaxiom. Confirmed on-disk layout (from a real shard header):
 *   weight        : U8    [out, in/2]     2x E2M1 nibbles/byte (signed FP4)
 *   weight_scale  : F8_E4M3 [out, in/16]  per-group-16 block scale (FP8 e4m3)
 *   weight_scale_2: F32    []             per-tensor global scale
 *   => value(r,c) = e2m1(nibble) * e4m3(block_scale[r, c/16]) * global_scale_2
 * Self-contained (CUDA runtime only). Build:
 *   nvcc -O3 -arch=sm_121 -o bin/axiom-e2m1-probe tools/axiom_e2m1_probe.cu
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>

#define GROUP 16u

__device__ __host__ static inline float e2m1_mag(uint32_t m) {
    switch (m & 7u) {
        case 0: return 0.0f; case 1: return 0.5f; case 2: return 1.0f; case 3: return 1.5f;
        case 4: return 2.0f; case 5: return 3.0f; case 6: return 4.0f; default: return 6.0f;
    }
}
__device__ __host__ static inline float e2m1_decode(uint8_t nib) {
    float v = e2m1_mag(nib & 7u);
    return (nib & 8u) ? -v : v;
}
/* OCP FP8 e4m3 (e4m3fn): s eeee mmm, exp bias 7, no inf; 0x7f/0xff = NaN. */
__device__ __host__ static inline float e4m3_decode(uint8_t b) {
    uint32_t s = (b >> 7) & 1u, e = (b >> 3) & 0xfu, m = b & 0x7u;
    float v;
    if (e == 0u) v = (float)m * (0.015625f / 8.0f);           /* subnormal: 2^-6 * m/8 */
    else         v = (1.0f + (float)m / 8.0f) * exp2f((float)((int)e - 7)); /* normal */
    return s ? -v : v;
}

__global__ static void e2m1_matvec_kernel(const uint8_t *qs, const uint8_t *bscale_e4m3,
                                          float global, const float *x, float *out,
                                          uint32_t rows, uint32_t cols) {
    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    uint32_t groups = cols / GROUP;
    float acc = 0.0f;
    for (uint32_t k = 0; k < cols; ++k) {
        uint64_t idx = (uint64_t)r * cols + k;
        uint8_t byte = qs[idx >> 1];
        uint8_t nib = (idx & 1u) ? (byte >> 4) : (byte & 0x0f);
        float bs = e4m3_decode(bscale_e4m3[(uint64_t)r * groups + (k / GROUP)]);
        acc += e2m1_decode(nib) * bs * global * x[k];
    }
    out[r] = acc;
}

static void ck(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { fprintf(stderr, "axiom-e2m1-probe: %s: %s\n", what, cudaGetErrorString(e)); exit(2); }
}

int main(void) {
    int dev = 0; cudaDeviceProp p;
    ck(cudaGetDevice(&dev), "getdevice");
    ck(cudaGetDeviceProperties(&p, dev), "getprops");
    printf("axiom-e2m1-probe: device %s sm_%d%d\n", p.name, p.major, p.minor);

    /* self-check the e4m3 table against known codes */
    printf("axiom-e2m1-probe: e4m3 0x38=%.4f 0x30=%.4f 0x40=%.4f 0x3c=%.4f (want 1,0.5,2,1.5)\n",
           e4m3_decode(0x38), e4m3_decode(0x30), e4m3_decode(0x40), e4m3_decode(0x3c));

    const uint32_t rows = 64, cols = 256;
    const uint32_t groups = cols / GROUP;
    const uint64_t nqbytes = (uint64_t)rows * cols / 2;
    const float global = 0.5f;

    uint8_t *qs = (uint8_t *)malloc(nqbytes);
    uint8_t *bscale = (uint8_t *)malloc((size_t)rows * groups);
    float *x = (float *)malloc(cols * sizeof(float));
    float *out_host = (float *)malloc(rows * sizeof(float));
    float *out_dev = (float *)malloc(rows * sizeof(float));
    static const uint8_t e4m3_codes[4] = {0x38, 0x30, 0x40, 0x3c}; /* 1.0,0.5,2.0,1.5 */

    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t k = 0; k < cols; ++k) {
            uint8_t nib = (uint8_t)(((r * 131u + k * 17u + 3u) >> 1) & 0x0fu);
            uint64_t idx = (uint64_t)r * cols + k;
            if (idx & 1u) qs[idx >> 1] = (uint8_t)((qs[idx >> 1] & 0x0f) | (nib << 4));
            else          qs[idx >> 1] = (uint8_t)((qs[idx >> 1] & 0xf0) | nib);
        }
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t g = 0; g < groups; ++g)
            bscale[r * groups + g] = e4m3_codes[(r + g) % 4];
    for (uint32_t k = 0; k < cols; ++k) x[k] = (float)((int)(k % 7u) - 3) * 0.5f;

    for (uint32_t r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (uint32_t k = 0; k < cols; ++k) {
            uint64_t idx = (uint64_t)r * cols + k;
            uint8_t byte = qs[idx >> 1];
            uint8_t nib = (idx & 1u) ? (byte >> 4) : (byte & 0x0f);
            double bs = (double)e4m3_decode(bscale[r * groups + (k / GROUP)]);
            acc += (double)e2m1_decode(nib) * bs * (double)global * (double)x[k];
        }
        out_host[r] = (float)acc;
    }

    uint8_t *d_qs, *d_bscale; float *d_x, *d_out;
    ck(cudaMalloc(&d_qs, nqbytes), "malloc qs");
    ck(cudaMalloc(&d_bscale, (size_t)rows * groups), "malloc bscale");
    ck(cudaMalloc(&d_x, cols * sizeof(float)), "malloc x");
    ck(cudaMalloc(&d_out, rows * sizeof(float)), "malloc out");
    ck(cudaMemcpy(d_qs, qs, nqbytes, cudaMemcpyHostToDevice), "cpy qs");
    ck(cudaMemcpy(d_bscale, bscale, (size_t)rows * groups, cudaMemcpyHostToDevice), "cpy bscale");
    ck(cudaMemcpy(d_x, x, cols * sizeof(float), cudaMemcpyHostToDevice), "cpy x");

    dim3 blk(64), grd((rows + 63) / 64);
    e2m1_matvec_kernel<<<grd, blk>>>(d_qs, d_bscale, global, d_x, d_out, rows, cols);
    ck(cudaGetLastError(), "launch");
    ck(cudaDeviceSynchronize(), "sync");
    ck(cudaMemcpy(out_dev, d_out, rows * sizeof(float), cudaMemcpyDeviceToHost), "cpy out");

    float max_abs = 0.0f; uint32_t worst = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        float d = fabsf(out_dev[r] - out_host[r]);
        if (d > max_abs) { max_abs = d; worst = r; }
    }
    int ok = max_abs < 1.0e-3f;
    printf("axiom-e2m1-probe: rows=%u cols=%u group=%u global=%.3f max_abs=%.9g (row %u host=%.6f dev=%.6f)\n",
           rows, cols, GROUP, global, max_abs, worst, out_host[worst], out_dev[worst]);
    printf("axiom-e2m1-probe: %s\n", ok ? "OK" : "FAIL");

    cudaFree(d_qs); cudaFree(d_bscale); cudaFree(d_x); cudaFree(d_out);
    free(qs); free(bscale); free(x); free(out_host); free(out_dev);
    return ok ? 0 : 1;
}
