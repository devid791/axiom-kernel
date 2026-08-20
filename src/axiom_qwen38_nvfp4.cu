/*
 * Native reference W4A4 path for unsloth/Qwen3.8-27B-NVFP4.
 *
 * The checkpoint uses compressed-tensors NVFP4: E2M1 values packed low-nibble
 * first, F8_E4M3 local scales for groups of 16, and F32 global scales stored
 * as quantization multipliers.  Therefore dequantization is q * local/global.
 *
 * This file is deliberately the exact, inspectable oracle before the serving
 * path moves the same layout to Blackwell tensor cores.  It is not a fallback
 * backend and it does not load any non-Unsloth checkpoint.
 */
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>

#include "axiom/axiom.h"
#include "axiom/qwen38_nvfp4.h"

namespace {

constexpr uint32_t kGroup = 16u;
constexpr uint8_t kE4m3Epsilon = 0x20u;  // +0.125, the compressed-tensors zero-scale guard.
constexpr uint8_t kE4m3MaxFinite = 0x7eu;

__device__ __forceinline__ float e2m1(uint8_t code) {
    constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = values[code & 7u];
    return (code & 8u) ? -v : v;
}

/* F8_E4M3FN as serialized by safetensors.  0x7f/0xff are NaN and are
 * invalid for checkpoint or runtime scale planes. */
__device__ __forceinline__ float e4m3fn(uint8_t code) {
    if ((code & 0x7fu) == 0x7fu) return nanf("");
    const uint32_t sign = code >> 7u;
    const uint32_t exp = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    float value = exp == 0u
            ? ldexpf(static_cast<float>(mantissa), -9)
            : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                     static_cast<int>(exp) - 7);
    return sign ? -value : value;
}

__device__ __forceinline__ uint8_t encode_e4m3fn_scale(float value) {
    uint8_t code = static_cast<uint8_t>(__nv_cvt_float_to_fp8(
            value, __NV_SATFINITE, __NV_E4M3));
    const uint8_t mag = code & 0x7fu;
    if (mag == 0u) return kE4m3Epsilon;
    if (mag == 0x7fu) return static_cast<uint8_t>((code & 0x80u) | kE4m3MaxFinite);
    return code;
}

__global__ void qwen38_quantize_reference_kernel(
        const float *input,
        float input_global_scale,
        uint8_t *out_packed,
        uint8_t *out_scale,
        uint32_t groups) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    const uint32_t first = group * kGroup;

    float max_abs = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kGroup; ++i) {
        max_abs = fmaxf(max_abs, fabsf(input[first + i]));
    }

    const uint8_t scale_code = encode_e4m3fn_scale(
            (max_abs * (1.0f / 6.0f)) * input_global_scale);
    const float local_scale = e4m3fn(scale_code) / input_global_scale;
    out_scale[group] = scale_code;

    #pragma unroll
    for (uint32_t pair = 0; pair < kGroup / 2u; ++pair) {
        const float low_value = input[first + pair * 2u] / local_scale;
        const float high_value = input[first + pair * 2u + 1u] / local_scale;
        const uint8_t low = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                low_value, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        const uint8_t high = static_cast<uint8_t>(__nv_cvt_float_to_fp4(
                high_value, __NV_E2M1, cudaRoundNearest)) & 0x0fu;
        out_packed[group * (kGroup / 2u) + pair] = static_cast<uint8_t>(low | (high << 4u));
    }
}

__global__ void qwen38_w4a4_reference_matvec_kernel(
        const uint8_t *weight_packed,
        const uint8_t *weight_scale,
        float weight_global_scale,
        const uint8_t *input_packed,
        const uint8_t *input_scale,
        float input_global_scale,
        float *out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t groups = cols / kGroup;
    const uint64_t packed_row = static_cast<uint64_t>(row) * (cols / 2u);
    const uint64_t scale_row = static_cast<uint64_t>(row) * groups;
    float acc = 0.0f;

    for (uint32_t group = 0; group < groups; ++group) {
        const float ws = e4m3fn(weight_scale[scale_row + group]) / weight_global_scale;
        const float xs = e4m3fn(input_scale[group]) / input_global_scale;
        #pragma unroll
        for (uint32_t pair = 0; pair < kGroup / 2u; ++pair) {
            const uint8_t w = weight_packed[packed_row + group * (kGroup / 2u) + pair];
            const uint8_t x = input_packed[group * (kGroup / 2u) + pair];
            acc = fmaf(e2m1(w & 0x0fu) * ws, e2m1(x & 0x0fu) * xs, acc);
            acc = fmaf(e2m1(w >> 4u) * ws, e2m1(x >> 4u) * xs, acc);
        }
    }
    out[row] = acc;
}

int finish(cudaStream_t stream) {
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    return cudaStreamSynchronize(stream) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

bool valid_scale(float value) {
    return std::isfinite(value) && value > 0.0f;
}

}  // namespace

extern "C" int axiom_qwen38_nvfp4_quantize_reference_f32(
        int device,
        const float *input,
        float input_global_scale,
        uint8_t *out_packed,
        uint8_t *out_scale,
        uint32_t cols,
        void *stream) {
    if (!input || !out_packed || !out_scale || cols == 0u || (cols % kGroup) != 0u ||
        !valid_scale(input_global_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint32_t groups = cols / kGroup;
    constexpr uint32_t threads = 128u;
    const dim3 grid((groups + threads - 1u) / threads);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    qwen38_quantize_reference_kernel<<<grid, threads, 0, cuda_stream>>>(
            input, input_global_scale, out_packed, out_scale, groups);
    return finish(cuda_stream);
}

extern "C" int axiom_qwen38_nvfp4_w4a4_reference_matvec_f32(
        int device,
        const uint8_t *weight_packed,
        const uint8_t *weight_scale,
        float weight_global_scale,
        const uint8_t *input_packed,
        const uint8_t *input_scale,
        float input_global_scale,
        float *out,
        uint32_t rows,
        uint32_t cols,
        void *stream) {
    if (!weight_packed || !weight_scale || !input_packed || !input_scale || !out ||
        rows == 0u || cols == 0u || (cols % kGroup) != 0u ||
        !valid_scale(weight_global_scale) || !valid_scale(input_global_scale)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    constexpr uint32_t threads = 64u;
    const dim3 grid((rows + threads - 1u) / threads);
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    qwen38_w4a4_reference_matvec_kernel<<<grid, threads, 0, cuda_stream>>>(
            weight_packed, weight_scale, weight_global_scale,
            input_packed, input_scale, input_global_scale, out, rows, cols);
    return finish(cuda_stream);
}
