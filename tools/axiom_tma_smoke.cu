#include <cuda_runtime.h>
#include <cuda/barrier>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

static constexpr uint32_t CHUNK_BYTES = 32768u;
static constexpr uint32_t TOTAL_BYTES = 2u * 1024u * 1024u;

__global__ static void axiom_scalar_copy_kernel(const uint8_t *src, uint8_t *dst, uint32_t chunks) {
    __shared__ __align__(16) uint8_t smem[CHUNK_BYTES];
    const uint32_t chunk = (uint32_t)blockIdx.x;
    if (chunk >= chunks) return;
    const uint8_t *in = src + (uint64_t)chunk * CHUNK_BYTES;
    uint8_t *out = dst + (uint64_t)chunk * CHUNK_BYTES;
    for (uint32_t i = threadIdx.x; i < CHUNK_BYTES; i += blockDim.x) {
        smem[i] = in[i];
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < CHUNK_BYTES; i += blockDim.x) {
        out[i] = (uint8_t)(smem[i] + 1u);
    }
}

__global__ static void axiom_tma_copy_kernel(const uint8_t *src, uint8_t *dst, uint32_t chunks) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    __shared__ __align__(16) uint8_t smem[CHUNK_BYTES];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();

    const uint32_t chunk = (uint32_t)blockIdx.x;
    if (chunk >= chunks) return;
    const uint8_t *in = src + (uint64_t)chunk * CHUNK_BYTES;
    uint8_t *out = dst + (uint64_t)chunk * CHUNK_BYTES;
    if (threadIdx.x == 0) {
        cuda::memcpy_async(smem, in, cuda::aligned_size_t<16>(CHUNK_BYTES), bar);
    }
    barrier_t::arrival_token token = bar.arrive();
    bar.wait(cuda::std::move(token));
    for (uint32_t i = threadIdx.x; i < CHUNK_BYTES; i += blockDim.x) {
        out[i] = (uint8_t)(smem[i] + 1u);
    }
#else
    (void)src;
    (void)dst;
    (void)chunks;
#endif
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

int main(int argc, char **argv) {
    uint32_t reps = 200;
    if (argc > 1) reps = (uint32_t)strtoul(argv[1], NULL, 10);
    if (reps == 0) reps = 1;
    const uint32_t chunks = TOTAL_BYTES / CHUNK_BYTES;
    uint8_t *host = (uint8_t *)malloc(TOTAL_BYTES);
    if (!host) return 1;
    for (uint32_t i = 0; i < TOTAL_BYTES; ++i) host[i] = (uint8_t)(i * 131u + 17u);

    uint8_t *src = NULL;
    uint8_t *dst = NULL;
    cudaError_t err = cudaMalloc(&src, TOTAL_BYTES);
    err = err == cudaSuccess ? cudaMalloc(&dst, TOTAL_BYTES) : err;
    err = err == cudaSuccess ? cudaMemcpy(src, host, TOTAL_BYTES, cudaMemcpyHostToDevice) : err;
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda init failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEvent_t s0, s1;
    cudaEventCreate(&s0);
    cudaEventCreate(&s1);
    const dim3 grid(chunks);
    const dim3 block(256);

    axiom_scalar_copy_kernel<<<grid, block>>>(src, dst, chunks);
    axiom_tma_copy_kernel<<<grid, block>>>(src, dst, chunks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "warmup failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventRecord(s0);
    for (uint32_t i = 0; i < reps; ++i) {
        axiom_scalar_copy_kernel<<<grid, block>>>(src, dst, chunks);
    }
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);
    const double scalar_ms = elapsed_ms(s0, s1);

    cudaEventRecord(s0);
    for (uint32_t i = 0; i < reps; ++i) {
        axiom_tma_copy_kernel<<<grid, block>>>(src, dst, chunks);
    }
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);
    const double tma_ms = elapsed_ms(s0, s1);

    uint8_t check[32];
    err = cudaMemcpy(check, dst, sizeof(check), cudaMemcpyDeviceToHost);
    int ok = err == cudaSuccess;
    for (uint32_t i = 0; i < sizeof(check) && ok; ++i) {
        const uint8_t expected = (uint8_t)(host[i] + 1u);
        ok = check[i] == expected;
    }

    const double mb = ((double)TOTAL_BYTES * (double)reps) / (1024.0 * 1024.0);
    printf("axiom_tma_smoke status=%s reps=%u bytes=%u chunk=%u scalar_ms=%.6f tma_ms=%.6f scalar_gbs=%.3f tma_gbs=%.3f speedup=%.3f\n",
           ok ? "pass" : "fail",
           reps,
           TOTAL_BYTES,
           CHUNK_BYTES,
           scalar_ms,
           tma_ms,
           scalar_ms > 0.0 ? mb / scalar_ms : 0.0,
           tma_ms > 0.0 ? mb / tma_ms : 0.0,
           tma_ms > 0.0 ? scalar_ms / tma_ms : 0.0);

    cudaFree(dst);
    cudaFree(src);
    free(host);
    return ok ? 0 : 2;
}
