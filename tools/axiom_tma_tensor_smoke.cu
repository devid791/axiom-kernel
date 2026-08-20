#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

static constexpr uint32_t CHUNK_BYTES = 256u;
static constexpr uint32_t CHUNK_ELEMS = CHUNK_BYTES / 4u;
static constexpr uint32_t TOTAL_BYTES = 2u * 1024u * 1024u;

__global__ static void axiom_tensor_map_copy_kernel(const __grid_constant__ CUtensorMap tensor_map, uint8_t *dst, uint32_t chunks) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) init(&bar, blockDim.x);
    __syncthreads();

    const uint32_t chunk = (uint32_t)blockIdx.x;
    if (chunk >= chunks) return;
    uint8_t *out = dst + (uint64_t)chunk * CHUNK_BYTES;
    if (threadIdx.x == 0) {
        cuda::device::barrier_expect_tx(bar, CHUNK_BYTES);
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(
            smem, &tensor_map, 0, (int)chunk, bar);
    }
    barrier_t::arrival_token token = bar.arrive();
    bar.wait(cuda::std::move(token));
    for (uint32_t i = threadIdx.x; i < CHUNK_BYTES; i += blockDim.x) {
        out[i] = (uint8_t)(smem[i] + 1u);
    }
#else
    (void)tensor_map;
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
        free(host);
        return 1;
    }

    CUresult cu = cuInit(0);
    if (cu != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed: %d\n", (int)cu);
        return 1;
    }
    alignas(64) CUtensorMap tensor_map;
    const cuuint64_t global_dim[2] = {CHUNK_ELEMS, TOTAL_BYTES / CHUNK_BYTES};
    const cuuint64_t global_stride[1] = {CHUNK_BYTES};
    const cuuint32_t box_dim[2] = {CHUNK_ELEMS, 1u};
    const cuuint32_t element_stride[2] = {1u, 1u};
    cu = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        2,
        src,
        global_dim,
        global_stride,
        box_dim,
        element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (cu != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed: %d\n", (int)cu);
        return 1;
    }

    cudaEvent_t s0, s1;
    cudaEventCreate(&s0);
    cudaEventCreate(&s1);
    const uint32_t chunks = TOTAL_BYTES / CHUNK_BYTES;
    const dim3 grid(chunks);
    const dim3 block(256);

    axiom_tensor_map_copy_kernel<<<grid, block, CHUNK_BYTES>>>(tensor_map, dst, chunks);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "warmup failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaEventRecord(s0);
    for (uint32_t i = 0; i < reps; ++i) {
        axiom_tensor_map_copy_kernel<<<grid, block, CHUNK_BYTES>>>(tensor_map, dst, chunks);
    }
    cudaEventRecord(s1);
    cudaEventSynchronize(s1);
    const double tensor_ms = elapsed_ms(s0, s1);

    uint8_t check[32];
    err = cudaMemcpy(check, dst, sizeof(check), cudaMemcpyDeviceToHost);
    int ok = err == cudaSuccess;
    for (uint32_t i = 0; i < sizeof(check) && ok; ++i) {
        ok = check[i] == (uint8_t)(host[i] + 1u);
    }

    const double mb = ((double)TOTAL_BYTES * (double)reps) / (1024.0 * 1024.0);
    printf("axiom_tma_tensor_smoke status=%s reps=%u bytes=%u chunk=%u tensor_ms=%.6f tensor_gbs=%.3f\n",
           ok ? "pass" : "fail",
           reps,
           TOTAL_BYTES,
           CHUNK_BYTES,
           tensor_ms,
           tensor_ms > 0.0 ? mb / tensor_ms : 0.0);

    cudaFree(dst);
    cudaFree(src);
    free(host);
    return ok ? 0 : 2;
}
