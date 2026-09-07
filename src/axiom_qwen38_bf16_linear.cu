/* Native resident BF16 x F32 reference linear for Qwen3.8 batch 8. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_bf16_linear.h"

namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint64_t kUploadChunkBytes = 64ull * 1024ull * 1024ull;
constexpr uint32_t kNvfp4Group = 16u;
constexpr uint32_t kNvfp4NibbleValues = 16u;

__device__ __forceinline__ float decode_bf16(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

#include "axiom_qwen38_bf16_pair_virtual.cuh"

__global__ void bf16_linear_batch8_reference_kernel(
        const uint16_t *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_BF16_LINEAR_BATCH) return;

    const uint16_t *w = weight + static_cast<uint64_t>(row) * cols;
    const float *x = input + static_cast<uint64_t>(column) * cols;
    float partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        partial += decode_bf16(w[c]) * x[c];
    }

    __shared__ float reduction[kThreads];
    reduction[thread] = partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) reduction[thread] += reduction[thread + stride];
        __syncthreads();
    }
    if (thread == 0u) {
        out[static_cast<uint64_t>(column) * rows + row] = reduction[0];
    }
}

__global__ void bf16_linear_pair_batch8_reference_kernel(
        const uint16_t *__restrict__ first_weight,
        const uint16_t *__restrict__ second_weight,
        const float *__restrict__ input,
        float *__restrict__ first_out,
        float *__restrict__ second_out,
        uint32_t rows,
        uint32_t cols) {
    const uint32_t row = blockIdx.x;
    const uint32_t column = blockIdx.y;
    const uint32_t thread = threadIdx.x;
    if (row >= rows || column >= AXIOM_QWEN38_BF16_LINEAR_BATCH) return;

    const uint16_t *first = first_weight + static_cast<uint64_t>(row) * cols;
    const uint16_t *second = second_weight + static_cast<uint64_t>(row) * cols;
    const float *x = input + static_cast<uint64_t>(column) * cols;
    float first_partial = 0.0f;
    float second_partial = 0.0f;
    for (uint32_t c = thread; c < cols; c += blockDim.x) {
        const float value = x[c];
        first_partial += decode_bf16(first[c]) * value;
        second_partial += decode_bf16(second[c]) * value;
    }

    __shared__ float first_reduction[kThreads];
    __shared__ float second_reduction[kThreads];
    first_reduction[thread] = first_partial;
    second_reduction[thread] = second_partial;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (thread < stride) {
            first_reduction[thread] += first_reduction[thread + stride];
            second_reduction[thread] += second_reduction[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0u) {
        const uint64_t index = static_cast<uint64_t>(column) * rows + row;
        first_out[index] = first_reduction[0];
        second_out[index] = second_reduction[0];
    }
}

__global__ void f32_to_bf16_columns_kernel(
        const float *__restrict__ input,
        uint16_t *__restrict__ output,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __bfloat16_as_ushort(__float2bfloat16_rn(input[index]));
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

int upload_tensor_chunked(
        axiom_model *model,
        const char *name,
        uint16_t *device_dst,
        uint64_t bytes) {
    if (!model || !name || !device_dst || bytes == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t capacity = bytes < kUploadChunkBytes ? bytes : kUploadChunkBytes;
    std::vector<uint8_t> chunk;
    try {
        chunk.resize(static_cast<size_t>(capacity));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    uint8_t *dst = reinterpret_cast<uint8_t *>(device_dst);
    for (uint64_t offset = 0u; offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const uint64_t count = remaining < capacity ? remaining : capacity;
        int rc = axiom_model_tensor_read_slice(model, name, offset, chunk.data(), count);
        if (rc != AXIOM_OK) return rc;
        const cudaError_t status = cudaMemcpy(
                dst + offset, chunk.data(), static_cast<size_t>(count), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
        }
        offset += count;
    }
    return AXIOM_OK;
}

__host__ __device__ float decode_e2m1_host_device(uint8_t code) {
    constexpr float values[kNvfp4NibbleValues] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return values[code & 0x0fu];
}

__host__ __device__ float decode_e4m3fn_host_device(uint8_t code) {
    if ((code & 0x7fu) == 0x7fu) return NAN;
    const uint32_t exponent = (code >> 3u) & 0x0fu;
    const uint32_t mantissa = code & 0x07u;
    const float value = exponent == 0u
            ? std::ldexp(static_cast<float>(mantissa), -9)
            : std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f,
                         static_cast<int>(exponent) - 7);
    return (code & 0x80u) ? -value : value;
}

uint16_t float_to_bf16_host(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

bool exact_f32_scalar(const axiom_tensor_info &info) {
    const bool scalar_shape = info.rank == 0u ||
            (info.rank == 1u && info.shape[0] == 1u);
    return info.dtype == AXIOM_TENSOR_DTYPE_F32 && scalar_shape &&
            info.byte_count == sizeof(float);
}

int tensor_info(
        axiom_model *model,
        const std::string &name,
        axiom_tensor_info *out) {
    if (!model || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_ABI_VERSION;
    return axiom_model_tensor_info_get(model, name.c_str(), out);
}

int upload_nvfp4_dequantized_rows(
        axiom_model *model,
        const std::string &weight_name,
        const std::string &scale_name,
        uint32_t rows,
        uint32_t cols,
        float weight_global_scale,
        uint16_t *device_weight) {
    if (!model || !device_weight || rows == 0u || cols == 0u ||
        (cols % kNvfp4Group) != 0u || !std::isfinite(weight_global_scale) ||
        weight_global_scale <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t groups = cols / kNvfp4Group;
    /* Keep the host working set bounded.  The output chunk is only ~2.6 MiB
     * at 256 rows for the Qwen3.8 vocabulary head. */
    constexpr uint32_t kRowsPerChunk = 256u;
    const uint32_t chunk_rows = rows < kRowsPerChunk ? rows : kRowsPerChunk;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    std::vector<uint16_t> dense;
    try {
        packed.resize(static_cast<size_t>(chunk_rows) * (cols / 2u));
        scales.resize(static_cast<size_t>(chunk_rows) * groups);
        dense.resize(static_cast<size_t>(chunk_rows) * cols);
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    const uint64_t packed_row_bytes = static_cast<uint64_t>(cols) / 2u;
    const uint64_t scale_row_bytes = groups;
    const uint64_t dense_row_bytes = static_cast<uint64_t>(cols) * sizeof(uint16_t);
    for (uint32_t row_start = 0u; row_start < rows;) {
        const uint32_t count = (rows - row_start) < chunk_rows
                ? rows - row_start : chunk_rows;
        const uint64_t packed_bytes = static_cast<uint64_t>(count) * packed_row_bytes;
        const uint64_t scale_bytes = static_cast<uint64_t>(count) * scale_row_bytes;
        int rc = axiom_model_tensor_read_slice(
                model, weight_name.c_str(), static_cast<uint64_t>(row_start) * packed_row_bytes,
                packed.data(), packed_bytes);
        if (rc == AXIOM_OK) rc = axiom_model_tensor_read_slice(
                model, scale_name.c_str(), static_cast<uint64_t>(row_start) * scale_row_bytes,
                scales.data(), scale_bytes);
        if (rc != AXIOM_OK) return rc;
        for (uint32_t row = 0u; row < count; ++row) {
            const uint8_t *packed_row = packed.data() + static_cast<size_t>(row) * packed_row_bytes;
            const uint8_t *scale_row = scales.data() + static_cast<size_t>(row) * scale_row_bytes;
            uint16_t *dense_row = dense.data() + static_cast<size_t>(row) * cols;
            for (uint32_t group = 0u; group < groups; ++group) {
                const float local_scale = decode_e4m3fn_host_device(scale_row[group]);
                if (!std::isfinite(local_scale) || local_scale <= 0.0f) {
                    return AXIOM_ERR_INVALID_ARGUMENT;
                }
                const float scale = local_scale * weight_global_scale;
                for (uint32_t pair = 0u; pair < kNvfp4Group / 2u; ++pair) {
                    const uint8_t packed_pair = packed_row[
                            group * (kNvfp4Group / 2u) + pair];
                    const uint32_t first = group * kNvfp4Group + pair * 2u;
                    dense_row[first] = float_to_bf16_host(
                            decode_e2m1_host_device(packed_pair & 0x0fu) * scale);
                    dense_row[first + 1u] = float_to_bf16_host(
                            decode_e2m1_host_device(packed_pair >> 4u) * scale);
                }
            }
        }
        const cudaError_t status = cudaMemcpy(
                reinterpret_cast<uint8_t *>(device_weight) +
                        static_cast<uint64_t>(row_start) * dense_row_bytes,
                dense.data(), static_cast<size_t>(static_cast<uint64_t>(count) * dense_row_bytes),
                cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
        }
        row_start += count;
    }
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_bf16_linear {
    int device = -1;
    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint16_t *weight = nullptr;
    uint16_t *input_bf16 = nullptr;
    cublasHandle_t cublas = nullptr;
    bool nvfp4_dequantized = false;
    bool pair_virtual128 = false;
    uint64_t device_bytes = 0u;
};

extern "C" int axiom_qwen38_bf16_linear_load(
        axiom_model *model,
        int device,
        const char *weight_name,
        axiom_qwen38_bf16_linear **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !weight_name || !weight_name[0] || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, weight_name, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.dtype != AXIOM_TENSOR_DTYPE_BF16 || info.rank != 2u ||
        info.shape[0] == 0u || info.shape[1] == 0u ||
        info.shape[0] > std::numeric_limits<uint32_t>::max() ||
        info.shape[1] > std::numeric_limits<uint32_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t elements = 0u;
    uint64_t weight_bytes = 0u;
    if (!checked_mul_u64(info.shape[0], info.shape[1], &elements) ||
        !checked_mul_u64(elements, sizeof(uint16_t), &weight_bytes) ||
        info.byte_count != weight_bytes ||
        weight_bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;

    axiom_qwen38_bf16_linear *linear = new (std::nothrow) axiom_qwen38_bf16_linear();
    if (!linear) return AXIOM_ERR_BUDGET;
    linear->device = device;
    linear->rows = static_cast<uint32_t>(info.shape[0]);
    linear->cols = static_cast<uint32_t>(info.shape[1]);
    // Snapshot at weight load, outside forward/graph capture. Only exact "1"
    // opts in; changing the environment requires reloading both paired handles.
    const char *pair_virtual128 = std::getenv("AXIOM_QWEN38_BF16_PAIR_VIRTUAL128");
    linear->pair_virtual128 = pair_virtual128 && std::strcmp(pair_virtual128, "1") == 0;

    const cudaError_t allocation = cudaMalloc(
            reinterpret_cast<void **>(&linear->weight), static_cast<size_t>(weight_bytes));
    if (allocation != cudaSuccess) {
        axiom_qwen38_bf16_linear_destroy(linear);
        return allocation == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    rc = upload_tensor_chunked(model, weight_name, linear->weight, weight_bytes);
    if (rc != AXIOM_OK) {
        axiom_qwen38_bf16_linear_destroy(linear);
        return rc;
    }
    linear->device_bytes = weight_bytes;
    *out = linear;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_bf16_linear_load_nvfp4_dequantized(
        axiom_model *model,
        int device,
        const char *base,
        uint32_t rows,
        uint32_t cols,
        axiom_qwen38_bf16_linear **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || !base || !base[0] || !out || rows == 0u || cols == 0u ||
        (cols % kNvfp4Group) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const std::string prefix(base);
    const std::string weight_name = prefix + ".weight";
    const std::string scale_name = prefix + ".weight_scale";
    const std::string input_scale_name = prefix + ".input_scale";
    const std::string global_name = prefix + ".weight_scale_2";
    axiom_tensor_info weight_info{};
    axiom_tensor_info scale_info{};
    axiom_tensor_info input_scale_info{};
    axiom_tensor_info global_info{};
    int rc = tensor_info(model, weight_name, &weight_info);
    if (rc == AXIOM_OK) rc = tensor_info(model, scale_name, &scale_info);
    if (rc == AXIOM_OK) rc = tensor_info(model, input_scale_name, &input_scale_info);
    if (rc == AXIOM_OK) rc = tensor_info(model, global_name, &global_info);
    const uint64_t weight_bytes = static_cast<uint64_t>(rows) * (cols / 2u);
    const uint64_t scale_bytes = static_cast<uint64_t>(rows) * (cols / kNvfp4Group);
    if (rc != AXIOM_OK ||
        weight_info.dtype != AXIOM_TENSOR_DTYPE_U8 || weight_info.rank != 2u ||
        weight_info.shape[0] != rows || weight_info.shape[1] != cols / 2u ||
        weight_info.byte_count != weight_bytes ||
        scale_info.dtype != AXIOM_TENSOR_DTYPE_F8_E4M3 || scale_info.rank != 2u ||
        scale_info.shape[0] != rows || scale_info.shape[1] != cols / kNvfp4Group ||
        scale_info.byte_count != scale_bytes || !exact_f32_scalar(input_scale_info) ||
        !exact_f32_scalar(global_info)) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    float input_scale = 0.0f;
    float weight_global_scale = 0.0f;
    rc = axiom_model_tensor_read_slice(
            model, input_scale_name.c_str(), 0u, &input_scale, sizeof(input_scale));
    if (rc == AXIOM_OK) rc = axiom_model_tensor_read_slice(
            model, global_name.c_str(), 0u, &weight_global_scale, sizeof(weight_global_scale));
    if (rc != AXIOM_OK || !std::isfinite(input_scale) || input_scale <= 0.0f ||
        !std::isfinite(weight_global_scale) || weight_global_scale <= 0.0f) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_bf16_linear *linear = new (std::nothrow) axiom_qwen38_bf16_linear();
    if (!linear) return AXIOM_ERR_BUDGET;
    linear->device = device;
    linear->rows = rows;
    linear->cols = cols;
    linear->nvfp4_dequantized = true;
    cudaError_t status = cudaMalloc(
            reinterpret_cast<void **>(&linear->weight),
            static_cast<size_t>(static_cast<uint64_t>(rows) * cols * sizeof(uint16_t)));
    if (status == cudaSuccess) status = cudaMalloc(
            reinterpret_cast<void **>(&linear->input_bf16),
            static_cast<size_t>(static_cast<uint64_t>(cols) * AXIOM_QWEN38_BF16_LINEAR_BATCH *
                                sizeof(uint16_t)));
    if (status != cudaSuccess) {
        axiom_qwen38_bf16_linear_destroy(linear);
        return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
    }
    rc = upload_nvfp4_dequantized_rows(
            model, weight_name, scale_name, rows, cols, weight_global_scale, linear->weight);
    if (rc == AXIOM_OK && cublasCreate(&linear->cublas) != CUBLAS_STATUS_SUCCESS) {
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_bf16_linear_destroy(linear);
        return rc;
    }
    linear->device_bytes = static_cast<uint64_t>(rows) * cols * sizeof(uint16_t) +
            static_cast<uint64_t>(cols) * AXIOM_QWEN38_BF16_LINEAR_BATCH * sizeof(uint16_t);
    *out = linear;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_bf16_linear_destroy(axiom_qwen38_bf16_linear *linear) {
    if (!linear) return;
    if (linear->device >= 0) (void)cudaSetDevice(linear->device);
    if (linear->cublas) (void)cublasDestroy(linear->cublas);
    if (linear->input_bf16) (void)cudaFree(linear->input_bf16);
    if (linear->weight) (void)cudaFree(linear->weight);
    delete linear;
}

extern "C" uint32_t axiom_qwen38_bf16_linear_rows(
        const axiom_qwen38_bf16_linear *linear) {
    return linear ? linear->rows : 0u;
}

extern "C" uint32_t axiom_qwen38_bf16_linear_cols(
        const axiom_qwen38_bf16_linear *linear) {
    return linear ? linear->cols : 0u;
}

extern "C" uint64_t axiom_qwen38_bf16_linear_device_bytes(
        const axiom_qwen38_bf16_linear *linear) {
    return linear ? linear->device_bytes : 0u;
}

extern "C" int axiom_qwen38_bf16_linear_forward_f32_device(
        axiom_qwen38_bf16_linear *linear,
        const float *input,
        float *out,
        void *stream) {
    if (linear && linear->nvfp4_dequantized) {
        return axiom_qwen38_bf16_linear_forward_f32_device_m(
                linear, input, out, AXIOM_QWEN38_BF16_LINEAR_BATCH, stream);
    }
    if (!linear || !input || !out || !linear->weight ||
        linear->rows == 0u || linear->cols == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    bf16_linear_batch8_reference_kernel<<<
            dim3(linear->rows, AXIOM_QWEN38_BF16_LINEAR_BATCH),
            kThreads, 0, cuda_stream>>>(
            linear->weight, input, out, linear->rows, linear->cols);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_bf16_linear_pair_forward_f32_device(
        axiom_qwen38_bf16_linear *first,
        axiom_qwen38_bf16_linear *second,
        const float *input,
        float *first_out,
        float *second_out,
        void *stream) {
    if (!first || !second || !input || !first_out || !second_out ||
        !first->weight || !second->weight || first->nvfp4_dequantized ||
        second->nvfp4_dequantized || first->device != second->device ||
        first->rows == 0u || first->cols == 0u ||
        first->rows != second->rows || first->cols != second->cols) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(first->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (first->pair_virtual128 && second->pair_virtual128) {
        bf16_linear_pair_batch8_virtual_kernel<128><<<
                dim3(first->rows, AXIOM_QWEN38_BF16_LINEAR_BATCH),
                128, 0, cuda_stream>>>(
                first->weight, second->weight, input, first_out, second_out,
                first->rows, first->cols);
    } else {
        bf16_linear_pair_batch8_reference_kernel<<<
                dim3(first->rows, AXIOM_QWEN38_BF16_LINEAR_BATCH),
                kThreads, 0, cuda_stream>>>(
                first->weight, second->weight, input, first_out, second_out,
                first->rows, first->cols);
    }
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_bf16_linear_forward_f32_device_m(
        axiom_qwen38_bf16_linear *linear,
        const float *input,
        float *out,
        uint32_t columns,
        void *stream) {
    if (!linear || !input || !out || !linear->nvfp4_dequantized || !linear->weight ||
        !linear->input_bf16 || !linear->cublas || columns == 0u ||
        columns > AXIOM_QWEN38_BF16_LINEAR_BATCH) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(linear->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const uint64_t input_count = static_cast<uint64_t>(linear->cols) * columns;
    f32_to_bf16_columns_kernel<<<
            static_cast<uint32_t>((input_count + kThreads - 1u) / kThreads),
            kThreads, 0, cuda_stream>>>(input, linear->input_bf16, input_count);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cublasSetStream(linear->cublas, cuda_stream) != CUBLAS_STATUS_SUCCESS) {
        return AXIOM_ERR_CUDA;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
            linear->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(linear->rows), static_cast<int>(columns),
            static_cast<int>(linear->cols), &alpha,
            linear->weight, CUDA_R_16BF, static_cast<int>(linear->cols),
            linear->input_bf16, CUDA_R_16BF, static_cast<int>(linear->cols),
            &beta, out, CUDA_R_32F, static_cast<int>(linear->rows),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS ? AXIOM_OK : AXIOM_ERR_CUDA;
}
