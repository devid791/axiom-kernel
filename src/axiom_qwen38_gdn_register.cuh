// Included inside gdn.cu's anonymous namespace after constants and BF16 rounding.
// Preserve the qualified nonvolatile fused-row loops: compiler spills are
// intentional; this variant measured faster than the volatile spill-free probe.
// Production dispatch uses tile16 regardless of temporal_value_tile.

template <uint32_t ValueTile>
__global__ void register_recurrence_kernel(
        const float *__restrict__ normalized_qkv,
        const float *__restrict__ g, const float *__restrict__ beta,
        const float *__restrict__ input_state,
        float *__restrict__ final_state, float *__restrict__ out,
        float *__restrict__ delta_history, float *__restrict__ decay_history) {
    static_assert(kHeadDim == 128u, "Register probe requires 128 state rows");
    static_assert(ValueTile == 8u || ValueTile == 16u || ValueTile == 32u,
                  "Supported value tiles: 8, 16, 32");
    const uint32_t lane = threadIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t value_start = blockIdx.x * ValueTile;
    const uint32_t column = value_start + lane;
    const uint32_t key_head = head / (kValueHeads / kKeyHeads);
    const bool owns_column = lane < ValueTile;
    __shared__ float qq[128];
    __shared__ float kk[128];
    float state[128];

    if (owns_column) {
#pragma unroll 128
        for (uint32_t row = 0; row < 128u; ++row)
            state[row] = input_state[
                (static_cast<uint64_t>(head) * kHeadDim + row) * kHeadDim + column];
    }
    // Keep tokens rolled: only the state-row accesses must be fully unrolled.
#pragma unroll 1
    for (uint32_t token = 0; token < kBatch; ++token) {
        const uint64_t qkv_base = static_cast<uint64_t>(token) * kConvDim;
        const uint64_t key_base = qkv_base + static_cast<uint64_t>(key_head) * kHeadDim;
#pragma unroll 4
        for (uint32_t chunk = 0; chunk < 4u; ++chunk) {
            const uint32_t row = lane + chunk * 32u;
            qq[row] = normalized_qkv[key_base + row];
            kk[row] = normalized_qkv[key_base + kKeyDim + row];
        }
        __syncthreads();
        if (owns_column) {
            const uint64_t gate_index = static_cast<uint64_t>(token) * kValueHeads + head;
            const float decay = __expf(g[gate_index]);
            const float update = beta[gate_index];
            if (blockIdx.x == 0u && lane == 0u)
                decay_history[gate_index] = decay;
            // Separate rounded multiply, then strictly serial row-order FMAs.
            // __fmul_rn forbids contraction of decay into the projection/update.
            float projected = 0.0f;
#pragma unroll 128
            for (uint32_t row = 0; row < 128u; ++row) {
                state[row] = __fmul_rn(state[row], decay);
                projected = fmaf(state[row], kk[row], projected);
            }
            const float v = normalized_qkv[qkv_base + 2u * kKeyDim +
                static_cast<uint64_t>(head) * kHeadDim + column];
            const float delta = (v - projected) * update;
            const uint64_t output_index = gate_index * kHeadDim + column;
            delta_history[output_index] = delta;
            float value = 0.0f;
#pragma unroll 128
            for (uint32_t row = 0; row < 128u; ++row) {
                state[row] = fmaf(kk[row], delta, state[row]);
                value = fmaf(state[row], qq[row], value);
            }
            // Match baseline ABI: BF16-rounded values in float storage.
            out[output_index] = qwen38_gdn_round_bf16(value);
        }
        // Non-owning lanes must not overwrite shared Q/K for the next token
        // before owning lanes have finished consuming the current token.
        __syncthreads();
    }
    if (owns_column) {
#pragma unroll 128
        for (uint32_t row = 0; row < 128u; ++row)
            final_state[(static_cast<uint64_t>(head) * kHeadDim + row) * kHeadDim +
                        column] = state[row];
    }
}

template <uint32_t ValueTile>
int launch_register_recurrence(
        const float *qkv, const float *g, const float *beta,
        const float *input_state, float *final_state, float *out,
        float *delta, float *decay, cudaStream_t stream) {
    register_recurrence_kernel<ValueTile><<<
        dim3(kHeadDim / ValueTile, kValueHeads), 32, 0, stream>>>(
        qkv, g, beta, input_state, final_state, out, delta, decay);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}
