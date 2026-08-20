#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IQ2_XXS_BLOCK_BYTES 66u
#define Q2_K_BLOCK_BYTES 84u

static void check(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "cuda_error %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static int cmp_float(const void *a, const void *b) {
    const float fa = *(const float *)a;
    const float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static float median(float *v, uint32_t n) {
    qsort(v, n, sizeof(float), cmp_float);
    return (n & 1u) ? v[n / 2u] : 0.5f * (v[n / 2u - 1u] + v[n / 2u]);
}

__device__ __forceinline__ static uint32_t load_u32(const uint8_t *p) {
    return *(const uint32_t *)p;
}

__global__ static void moe_gateup_iq2_load_only_kernel(
        const uint8_t *__restrict__ gate,
        const uint8_t *__restrict__ up,
        const uint32_t *__restrict__ indices,
        uint32_t *__restrict__ out,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    const uint32_t t = (uint32_t)blockIdx.y;
    if (row >= expert_hidden || t >= topk) return;
    const uint32_t blocks = hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * IQ2_XXS_BLOCK_BYTES;
    const uint32_t eid = indices[t];
    const uint64_t off = ((uint64_t)eid * expert_hidden + row) * row_bytes;
    uint32_t acc = 0u;
    for (uint32_t p = lane * 4u; p < row_bytes; p += 128u) {
        acc ^= load_u32(gate + off + p);
        acc += load_u32(up + off + p);
    }
    for (uint32_t s = 16u; s > 0u; s >>= 1u) acc ^= __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0u) out[t * expert_hidden + row] = acc;
}

__global__ static void moe_down_q2_load_only_kernel(
        const uint8_t *__restrict__ down,
        const uint32_t *__restrict__ indices,
        uint32_t *__restrict__ out,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp;
    if (row >= hidden) return;
    const uint32_t blocks = expert_hidden / 256u;
    const uint64_t row_bytes = (uint64_t)blocks * Q2_K_BLOCK_BYTES;
    uint32_t acc = 0u;
    for (uint32_t t = 0u; t < topk; ++t) {
        const uint32_t eid = indices[t];
        const uint64_t off = ((uint64_t)eid * hidden + row) * row_bytes;
        for (uint32_t p = lane * 4u; p < row_bytes; p += 128u) {
            acc += load_u32(down + off + p);
        }
    }
    for (uint32_t s = 16u; s > 0u; s >>= 1u) acc ^= __shfl_down_sync(0xffffffffu, acc, s);
    if (lane == 0u) out[row] = acc;
}

static float time_gateup(
        const uint8_t *gate,
        const uint8_t *up,
        const uint32_t *indices,
        uint32_t *out,
        uint32_t groups,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t reps) {
    const int block = 256;
    const int warps = block / 32;
    const dim3 grid((expert_hidden + warps - 1u) / warps, topk, 1);
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event_a");
    check(cudaEventCreate(&b), "event_b");
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t off = (i % groups) * topk;
        moe_gateup_iq2_load_only_kernel<<<grid, block>>>(gate, up, indices + off, out, topk, hidden, expert_hidden);
    }
    check(cudaDeviceSynchronize(), "warmup_gateup");
    check(cudaEventRecord(a), "record_a");
    for (uint32_t i = 0; i < reps; ++i) {
        const uint32_t off = (i % groups) * topk;
        moe_gateup_iq2_load_only_kernel<<<grid, block>>>(gate, up, indices + off, out, topk, hidden, expert_hidden);
    }
    check(cudaEventRecord(b), "record_b");
    check(cudaEventSynchronize(b), "sync_b");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed_gateup");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / (float)reps;
}

static float time_down(
        const uint8_t *down,
        const uint32_t *indices,
        uint32_t *out,
        uint32_t groups,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t reps) {
    const int block = 256;
    const int warps = block / 32;
    const dim3 grid((hidden + warps - 1u) / warps, 1, 1);
    cudaEvent_t a, b;
    check(cudaEventCreate(&a), "event_a");
    check(cudaEventCreate(&b), "event_b");
    for (uint32_t i = 0; i < 4u; ++i) {
        const uint32_t off = (i % groups) * topk;
        moe_down_q2_load_only_kernel<<<grid, block>>>(down, indices + off, out, topk, hidden, expert_hidden);
    }
    check(cudaDeviceSynchronize(), "warmup_down");
    check(cudaEventRecord(a), "record_a");
    for (uint32_t i = 0; i < reps; ++i) {
        const uint32_t off = (i % groups) * topk;
        moe_down_q2_load_only_kernel<<<grid, block>>>(down, indices + off, out, topk, hidden, expert_hidden);
    }
    check(cudaEventRecord(b), "record_b");
    check(cudaEventSynchronize(b), "sync_b");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, a, b), "elapsed_down");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / (float)reps;
}

int main(int argc, char **argv) {
    uint32_t experts = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 10) : 256u;
    uint32_t topk = argc > 2 ? (uint32_t)strtoul(argv[2], 0, 10) : 6u;
    uint32_t hidden = argc > 3 ? (uint32_t)strtoul(argv[3], 0, 10) : 4096u;
    uint32_t expert_hidden = argc > 4 ? (uint32_t)strtoul(argv[4], 0, 10) : 2048u;
    uint32_t reps = argc > 5 ? (uint32_t)strtoul(argv[5], 0, 10) : 42u;
    uint32_t samples = argc > 6 ? (uint32_t)strtoul(argv[6], 0, 10) : 7u;
    if (!experts || !topk || topk > experts || !hidden || !expert_hidden || !reps || !samples) return 2;
    if ((hidden % 256u) || (expert_hidden % 256u)) return 2;
    /* gate/up rows are (hidden/256)*66 bytes; an odd IQ2_XXS block count makes
       row_bytes 2 (mod 4), which breaks moe_gateup_iq2_load_only_kernel's 4-byte
       load stride (misaligned global load) and overruns the final row by 2 bytes.
       Require an even block count so row_bytes stays a multiple of 4. */
    if ((hidden / 256u) & 1u) return 2;

    const uint32_t groups = experts / topk;
    const uint64_t gate_bytes = (uint64_t)experts * expert_hidden * (hidden / 256u) * IQ2_XXS_BLOCK_BYTES;
    const uint64_t down_bytes = (uint64_t)experts * hidden * (expert_hidden / 256u) * Q2_K_BLOCK_BYTES;
    const uint64_t gateup_active = 2ull * topk * expert_hidden * (hidden / 256u) * IQ2_XXS_BLOCK_BYTES;
    const uint64_t down_active = (uint64_t)topk * hidden * (expert_hidden / 256u) * Q2_K_BLOCK_BYTES;
    uint8_t *gate = 0, *up = 0, *down = 0;
    uint32_t *indices = 0, *out = 0;
    check(cudaSetDevice(0), "set_device");
    check(cudaMalloc(&gate, (size_t)gate_bytes), "malloc_gate");
    check(cudaMalloc(&up, (size_t)gate_bytes), "malloc_up");
    check(cudaMalloc(&down, (size_t)down_bytes), "malloc_down");
    check(cudaMalloc(&indices, (size_t)groups * topk * sizeof(uint32_t)), "malloc_indices");
    check(cudaMalloc(&out, (size_t)(topk * expert_hidden + hidden) * sizeof(uint32_t)), "malloc_out");
    check(cudaMemset(gate, 0x13, (size_t)gate_bytes), "memset_gate");
    check(cudaMemset(up, 0x37, (size_t)gate_bytes), "memset_up");
    check(cudaMemset(down, 0x71, (size_t)down_bytes), "memset_down");
    uint32_t *ih = (uint32_t *)malloc((size_t)groups * topk * sizeof(uint32_t));
    for (uint32_t g = 0; g < groups; ++g) {
        for (uint32_t t = 0; t < topk; ++t) ih[g * topk + t] = (g * topk + t) % experts;
    }
    check(cudaMemcpy(indices, ih, (size_t)groups * topk * sizeof(uint32_t), cudaMemcpyHostToDevice), "copy_indices");

    float *gate_samples = (float *)malloc(samples * sizeof(float));
    float *down_samples = (float *)malloc(samples * sizeof(float));
    for (uint32_t s = 0; s < samples; ++s) {
        gate_samples[s] = time_gateup(gate, up, indices, out, groups, topk, hidden, expert_hidden, reps);
        down_samples[s] = time_down(down, indices, out, groups, topk, hidden, expert_hidden, reps);
    }
    const float gate_ms = median(gate_samples, samples);
    const float down_ms = median(down_samples, samples);
    const float total_ms = gate_ms + down_ms;
    const uint64_t active = gateup_active + down_active;
    const double gbs = (double)active / ((double)total_ms / 1000.0) / 1.0e9;
    const double reference_full_ms = 0.451785;
    printf("{\"experts\":%u,\"topk\":%u,\"hidden\":%u,\"expert_hidden\":%u,"
           "\"reps\":%u,\"samples\":%u,\"gateup_active_bytes\":%llu,"
           "\"down_active_bytes\":%llu,\"moe_active_bytes\":%llu,"
           "\"gateup_load_ms\":%.6f,\"down_load_ms\":%.6f,"
           "\"moe_load_ms\":%.6f,\"load_gb_s\":%.3f,"
           "\"reference_full_ms\":%.6f,\"load_fraction_vs_reference\":%.6f,"
           "\"compute_tax_ms_vs_reference\":%.6f}\n",
           experts, topk, hidden, expert_hidden, reps, samples,
           (unsigned long long)gateup_active,
           (unsigned long long)down_active,
           (unsigned long long)active,
           gate_ms, down_ms, total_ms, gbs,
           reference_full_ms, (double)total_ms / reference_full_ms,
           reference_full_ms - (double)total_ms);

    free(down_samples);
    free(gate_samples);
    free(ih);
    cudaFree(out);
    cudaFree(indices);
    cudaFree(down);
    cudaFree(up);
    cudaFree(gate);
    return 0;
}
