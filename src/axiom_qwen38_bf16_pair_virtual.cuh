#pragma once

// Internal implementation: include inside bf16_linear.cu's anonymous namespace
// after decode_bf16. Shared by production and the standalone microbenchmark.
// Preserve 256 virtual partials and the reference FP flags (no fast math).
static_assert(AXIOM_QWEN38_BF16_LINEAR_BATCH == 8, "requires batch eight");

template <unsigned Threads>
__global__ void bf16_linear_pair_batch8_virtual_kernel(
        const uint16_t *__restrict__ first_weight,
        const uint16_t *__restrict__ second_weight,
        const float *__restrict__ input,
        float *__restrict__ first_out, float *__restrict__ second_out,
        uint32_t rows, uint32_t cols) {
    static_assert(Threads == 32 || Threads == 64 || Threads == 128,
                  "only bounded physical block variants");
    constexpr unsigned Groups = 256 / Threads;
    const unsigned row = blockIdx.x, column = blockIdx.y;
    const unsigned thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_BF16_LINEAR_BATCH) return;
    const uint16_t *first = first_weight + uint64_t(row) * cols;
    const uint16_t *second = second_weight + uint64_t(row) * cols;
    const float *x = input + uint64_t(column) * cols;
    float a[Groups] = {}, b[Groups] = {};
    // Each slot is one ORIGINAL thread, not a reassociated contiguous sum.
    // For a fixed group: c = thread + group*Threads + iteration*256.
    // Keep the baseline FP32 multiply-add expression and accumulation order.
    for (unsigned iteration = 0; iteration < (cols + 255u) / 256u; ++iteration) {
#pragma unroll
        for (unsigned group = 0; group < Groups; ++group) {
            const unsigned c = thread + group * Threads + iteration * 256u;
            if (c < cols) {
                const float value = x[c];
                a[group] += decode_bf16(first[c]) * value;
                b[group] += decode_bf16(second[c]) * value;
            }
        }
    }
    // Original strides 128,64,32 (only those >= Threads) live in registers.
    // Unrolling is intended to scalarize slots; attributes/SASS must confirm.
#pragma unroll
    for (unsigned stride = Groups / 2; stride; stride >>= 1) {
#pragma unroll
        for (unsigned group = 0; group < stride; ++group) {
            a[group] += a[group + stride];
            b[group] += b[group + stride];
        }
    }
    // Retain the baseline shared-memory tree, including its ordering/barriers.
    // This is NOT the previously tested simple last-warp reduction change.
    __shared__ float ar[Threads], br[Threads];
    ar[thread] = a[0]; br[thread] = b[0];
    __syncthreads();
    for (unsigned stride = Threads / 2; stride; stride >>= 1) {
        if (thread < stride) {
            ar[thread] += ar[thread + stride];
            br[thread] += br[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0) {
        const uint64_t index = uint64_t(column) * rows + row;
        first_out[index] = ar[0]; second_out[index] = br[0];
    }
}
