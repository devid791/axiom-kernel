/* Native qwen4_exp / Flash-Next vision tower.
 *
 * This TU intentionally owns the whole vision path instead of routing image
 * features through a Python or third-party model wrapper. The implementation
 * is a correctness-first CUDA path: BF16 checkpoint weights, BF16-materialized
 * activation boundaries, variable-length non-causal attention per video frame,
 * and the checkpoint's 2x2 patch merger. It is kept separate from the text
 * model until the standalone gate and multimodal API parity gates pass.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen4exp/vision_adapter.hpp"
#include "axiom/qwen38_vision_preprocess.hpp"

#define Q4V_ABI_VERSION 1u
#define Q4V_DEPTH 27u
#define Q4V_HIDDEN 1152u
#define Q4V_INTERMEDIATE 4304u
#define Q4V_HEADS 16u
#define Q4V_HEAD_DIM 72u
#define Q4V_PATCH_SIZE 16u
#define Q4V_TEMPORAL_PATCH 2u
#define Q4V_MERGE_SIZE 2u
#define Q4V_POSITION_SIDE 48u
#define Q4V_MERGED_HIDDEN 4608u
#define Q4V_OUTPUT 2560u
#define Q4V_PATCH_FEATURES 1536u

struct qwen4exp_vision_core;
struct qwen4exp_vision_core_grid {
    uint32_t temporal;
    uint32_t height;
    uint32_t width;
};
struct qwen4exp_vision_core_info {
    uint32_t abi_version;
    uint32_t depth;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t patch_size;
    uint32_t temporal_patch_size;
    uint32_t spatial_merge_size;
    uint32_t position_grid_side;
    uint32_t patch_feature_count;
    uint32_t output_size;
    uint32_t max_tokens;
    uint64_t device_bytes;
    uint64_t loaded_tensor_count;
    uint64_t loaded_tensor_bytes;
};

extern "C" int qwen4exp_vision_core_create(
        axiom_model *, int, uint32_t, qwen4exp_vision_core **);
extern "C" void qwen4exp_vision_core_destroy(qwen4exp_vision_core *);
extern "C" int qwen4exp_vision_core_info_get(
        const qwen4exp_vision_core *, qwen4exp_vision_core_info *);
extern "C" int qwen4exp_vision_core_forward_patches(
        qwen4exp_vision_core *, const float *, uint64_t,
        const qwen4exp_vision_core_grid *, uint32_t, float *, uint64_t,
        uint32_t *);
extern "C" int qwen4exp_vision_core_forward_patches_device(
        qwen4exp_vision_core *, const float *, uint64_t,
        const qwen4exp_vision_core_grid *, uint32_t, float *, uint64_t,
        uint32_t *);

namespace {

constexpr uint32_t kDepth = Q4V_DEPTH;
constexpr uint32_t kVisionHidden = Q4V_HIDDEN;
constexpr uint32_t kIntermediate = Q4V_INTERMEDIATE;
constexpr uint32_t kHeads = Q4V_HEADS;
constexpr uint32_t kHeadDim = Q4V_HEAD_DIM;
constexpr uint32_t kPatchFeatures = Q4V_PATCH_FEATURES;
constexpr uint32_t kPatchSize = Q4V_PATCH_SIZE;
constexpr uint32_t kTemporalPatch = Q4V_TEMPORAL_PATCH;
constexpr uint32_t kMergeSize = Q4V_MERGE_SIZE;
constexpr uint32_t kMergeUnit = kMergeSize * kMergeSize;
constexpr uint32_t kPositionSide = Q4V_POSITION_SIDE;
constexpr uint32_t kPositionEntries = kPositionSide * kPositionSide;
constexpr uint32_t kOutput = Q4V_OUTPUT;
constexpr uint32_t kMaxThreads = 256u;
constexpr uint64_t kUploadChunkBytes = 64ull * 1024ull * 1024ull;
constexpr float kLayerNormEps = 1.0e-6f;
constexpr float kVisionRopeTheta = 10000.0f;
constexpr uint32_t kRotaryDim = kHeadDim / 2u;

static_assert(kVisionHidden == kHeads * kHeadDim, "vision head geometry changed");
static_assert(kRotaryDim == 36u, "vision rotary geometry changed");

struct vision_linear {
    uint32_t rows = 0u;
    uint32_t cols = 0u;
    uint16_t *weight = nullptr;
    uint16_t *bias = nullptr;
    uint64_t device_bytes = 0u;
};

struct vision_norm {
    uint16_t *weight = nullptr;
    uint16_t *bias = nullptr;
    uint64_t device_bytes = 0u;
};

struct vision_block {
    vision_norm norm1;
    vision_norm norm2;
    vision_linear qkv;
    vision_linear proj;
    vision_linear fc1;
    vision_linear fc2;
};

struct vision_metadata {
    uint32_t patch_count = 0u;
    uint32_t merged_count = 0u;
    uint32_t segment_count = 0u;
    std::vector<uint32_t> interpolation_indices;
    std::vector<float> interpolation_weights;
    std::vector<int32_t> position_ids;
    std::vector<uint32_t> segment_ids;
    std::vector<uint32_t> cu_seqlens;
};

}  // namespace

/* This definition must live in the global namespace: the public header
 * forward-declares the C ABI handle with this exact tag. */
struct qwen4exp_vision_core {
    int device = -1;
    uint32_t max_tokens = 0u;
    uint64_t loaded_tensor_count = 0u;
    uint64_t loaded_tensor_bytes = 0u;
    uint64_t device_bytes = 0u;
    cublasHandle_t cublas = nullptr;

    vision_linear patch_embed;
    uint16_t *pos_embed = nullptr;
    uint64_t pos_embed_bytes = 0u;
    vision_block blocks[Q4V_DEPTH];
    vision_norm merger_norm;
    vision_linear merger_fc1;
    vision_linear merger_fc2;

    /* Per-call metadata and activations. They are bounded by max_tokens and
     * are owned by this object, so a future API worker can serialize calls on
     * the object without allocating on the hot path. */
    float *patches = nullptr;
    float *hidden = nullptr;
    float *normed = nullptr;
    float *qkv = nullptr;
    float *attention = nullptr;
    float *fc1 = nullptr;
    float *merger_input = nullptr;
    float *merger_fc1_out = nullptr;
    uint16_t *linear_input_bf16 = nullptr;
    uint32_t *interpolation_indices = nullptr;
    float *interpolation_weights = nullptr;
    int32_t *position_ids = nullptr;
    uint32_t *segment_ids = nullptr;
    uint32_t *cu_seqlens = nullptr;
};

namespace {

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

int alloc_device(void **out, uint64_t bytes) {
    if (!out || bytes == 0u || bytes > std::numeric_limits<size_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    return cuda_status(cudaMalloc(out, static_cast<size_t>(bytes)));
}

__device__ __forceinline__ float bf16_to_float_device(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float bf16_materialize_device(float value) {
    return bf16_to_float_device(float_to_bf16_bits_device(value));
}

int tensor_shape_matches(
        axiom_model *model,
        const char *name,
        uint32_t rank,
        const uint64_t *shape,
        axiom_tensor_info *out_info) {
    if (!model || !name || !shape || rank == 0u || rank > AXIOM_MAX_TENSOR_DIMS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.dtype != AXIOM_TENSOR_DTYPE_BF16 || info.rank != rank) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t count = 1u;
    for (uint32_t i = 0u; i < rank; ++i) {
        if (shape[i] == 0u || info.shape[i] != shape[i] || !checked_mul(count, shape[i], &count)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    uint64_t bytes = 0u;
    if (!checked_mul(count, sizeof(uint16_t), &bytes) || info.byte_count != bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (out_info) *out_info = info;
    return AXIOM_OK;
}

int upload_bf16_tensor(
        axiom_model *model,
        const char *name,
        uint32_t rank,
        const uint64_t *shape,
        uint16_t **out,
        uint64_t *out_bytes) {
    if (out) *out = nullptr;
    if (out_bytes) *out_bytes = 0u;
    if (!out || !out_bytes) return AXIOM_ERR_INVALID_ARGUMENT;

    axiom_tensor_info info{};
    int rc = tensor_shape_matches(model, name, rank, shape, &info);
    if (rc != AXIOM_OK) return rc;
    uint16_t *device = nullptr;
    rc = alloc_device(reinterpret_cast<void **>(&device), info.byte_count);
    if (rc != AXIOM_OK) return rc;

    const uint64_t chunk_capacity = std::min<uint64_t>(info.byte_count, kUploadChunkBytes);
    std::vector<uint8_t> chunk;
    try {
        chunk.resize(static_cast<size_t>(chunk_capacity));
    } catch (...) {
        cudaFree(device);
        return AXIOM_ERR_BUDGET;
    }
    for (uint64_t offset = 0u; offset < info.byte_count;) {
        const uint64_t count = std::min<uint64_t>(chunk_capacity, info.byte_count - offset);
        rc = axiom_model_tensor_read_slice(model, name, offset, chunk.data(), count);
        if (rc == AXIOM_OK) {
            rc = cuda_status(cudaMemcpy(
                    reinterpret_cast<uint8_t *>(device) + offset,
                    chunk.data(), static_cast<size_t>(count), cudaMemcpyHostToDevice));
        }
        if (rc != AXIOM_OK) {
            cudaFree(device);
            return rc;
        }
        offset += count;
    }
    *out = device;
    *out_bytes = info.byte_count;
    return AXIOM_OK;
}

int load_norm(
        axiom_model *model,
        const std::string &prefix,
        vision_norm *out,
        uint32_t hidden) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t shape[] = {hidden};
    const std::string weight = prefix + ".weight";
    const std::string bias = prefix + ".bias";
    int rc = upload_bf16_tensor(model, weight.c_str(), 1u, shape, &out->weight, &out->device_bytes);
    if (rc == AXIOM_OK) {
        uint64_t bias_bytes = 0u;
        rc = upload_bf16_tensor(model, bias.c_str(), 1u, shape, &out->bias, &bias_bytes);
        if (rc == AXIOM_OK && !checked_add(out->device_bytes, bias_bytes, &out->device_bytes)) {
            rc = AXIOM_ERR_BUDGET;
        }
    }
    return rc;
}

int load_linear(
        axiom_model *model,
        const std::string &prefix,
        vision_linear *out,
        uint32_t rows,
        uint32_t cols) {
    if (!out || rows == 0u || cols == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_shape[] = {rows, cols};
    const uint64_t bias_shape[] = {rows};
    const std::string weight = prefix + ".weight";
    const std::string bias = prefix + ".bias";
    uint64_t weight_bytes = 0u;
    int rc = upload_bf16_tensor(
            model, weight.c_str(), 2u, weight_shape, &out->weight, &weight_bytes);
    if (rc == AXIOM_OK) {
        uint64_t bias_bytes = 0u;
        rc = upload_bf16_tensor(model, bias.c_str(), 1u, bias_shape, &out->bias, &bias_bytes);
        if (rc == AXIOM_OK && !checked_add(weight_bytes, bias_bytes, &out->device_bytes)) {
            rc = AXIOM_ERR_BUDGET;
        }
    }
    out->rows = rows;
    out->cols = cols;
    return rc;
}

void destroy_norm(vision_norm *norm) {
    if (!norm) return;
    cudaFree(norm->bias);
    cudaFree(norm->weight);
    *norm = vision_norm{};
}

void destroy_linear(vision_linear *linear) {
    if (!linear) return;
    cudaFree(linear->bias);
    cudaFree(linear->weight);
    *linear = vision_linear{};
}

__global__ void f32_to_bf16_kernel(
        const float *__restrict__ input,
        uint16_t *__restrict__ output,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = float_to_bf16_bits_device(input[index]);
}

__global__ void add_bias_materialize_kernel(
        float *__restrict__ output,
        const uint16_t *__restrict__ bias,
        uint32_t rows,
        uint32_t tokens) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(rows) * tokens;
    if (index >= count) return;
    const uint32_t row = static_cast<uint32_t>(index % rows);
    output[index] = bf16_materialize_device(output[index] + bf16_to_float_device(bias[row]));
}

__global__ void layer_norm_kernel(
        const float *__restrict__ input,
        float *__restrict__ output,
        const uint16_t *__restrict__ weight,
        const uint16_t *__restrict__ bias,
        uint32_t tokens,
        uint32_t hidden,
        float eps) {
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (token >= tokens) return;

    __shared__ float sums[kMaxThreads];
    __shared__ float squares[kMaxThreads];
    float sum = 0.0f;
    float square = 0.0f;
    const float *row = input + static_cast<uint64_t>(token) * hidden;
    for (uint32_t index = lane; index < hidden; index += blockDim.x) {
        const float value = row[index];
        sum += value;
        square += value * value;
    }
    sums[lane] = sum;
    squares[lane] = square;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            squares[lane] += squares[lane + stride];
        }
        __syncthreads();
    }
    const float mean = sums[0] / static_cast<float>(hidden);
    const float variance = fmaxf(0.0f, squares[0] / static_cast<float>(hidden) - mean * mean);
    const float inv = rsqrtf(variance + eps);
    float *out_row = output + static_cast<uint64_t>(token) * hidden;
    for (uint32_t index = lane; index < hidden; index += blockDim.x) {
        const float gamma = bf16_to_float_device(weight[index]);
        const float beta = bf16_to_float_device(bias[index]);
        out_row[index] = bf16_materialize_device((row[index] - mean) * inv * gamma + beta);
    }
}

__global__ void residual_add_kernel(
        float *__restrict__ residual,
        const float *__restrict__ add,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) residual[index] = bf16_materialize_device(residual[index] + add[index]);
}

__device__ __forceinline__ float gelu_tanh(float x) {
    constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(kSqrtTwoOverPi * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_materialize_kernel(float *values, uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) values[index] = bf16_materialize_device(gelu_tanh(values[index]));
}

__global__ void add_interpolated_position_kernel(
        const float *input,
        float *output,
        const uint16_t *__restrict__ pos_embed,
        const uint32_t *__restrict__ indices,
        const float *__restrict__ weights,
        uint32_t tokens,
        uint32_t hidden) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(tokens) * hidden;
    if (index >= count) return;
    const uint32_t token = static_cast<uint32_t>(index / hidden);
    const uint32_t dim = static_cast<uint32_t>(index % hidden);
    const uint32_t *row_indices = indices + static_cast<uint64_t>(token) * 4u;
    const float *row_weights = weights + static_cast<uint64_t>(token) * 4u;
    float position = 0.0f;
    for (uint32_t tap = 0u; tap < 4u; ++tap) {
        position += row_weights[tap] * bf16_to_float_device(
                pos_embed[static_cast<uint64_t>(row_indices[tap]) * hidden + dim]);
    }
    output[index] = bf16_materialize_device(input[index] + position);
}

__device__ __forceinline__ float rotary_value(
        const float *base,
        uint32_t dimension,
        int32_t height_position,
        int32_t width_position) {
    const uint32_t half = dimension < kRotaryDim ? dimension : dimension - kRotaryDim;
    const uint32_t axis = half < (kRotaryDim / 2u) ? 0u : 1u;
    const uint32_t pair = axis == 0u ? half : half - (kRotaryDim / 2u);
    const float position = static_cast<float>(axis == 0u ? height_position : width_position);
    const float inv_freq = 1.0f / powf(kVisionRopeTheta,
            (2.0f * static_cast<float>(pair)) / static_cast<float>(kRotaryDim));
    const float phase = position * inv_freq;
    const float cosine = cosf(phase);
    const float sine = sinf(phase);
    const float partner = dimension < kRotaryDim ? base[dimension + kRotaryDim]
                                                 : base[dimension - kRotaryDim];
    return base[dimension] * cosine + (dimension < kRotaryDim ? -partner : partner) * sine;
}

__global__ void vision_attention_kernel(
        const float *__restrict__ qkv,
        float *__restrict__ output,
        const int32_t *__restrict__ position_ids,
        const uint32_t *__restrict__ segment_ids,
        const uint32_t *__restrict__ cu_seqlens,
        uint32_t tokens) {
    const uint32_t query_token = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (query_token >= tokens || head >= kHeads) return;

    __shared__ float query[kHeadDim];
    __shared__ float reduction[kMaxThreads];
    const float *query_base = qkv + static_cast<uint64_t>(query_token) * (3u * kVisionHidden) +
            static_cast<uint64_t>(head) * kHeadDim;
    const int32_t q_height = position_ids[static_cast<uint64_t>(query_token) * 2u];
    const int32_t q_width = position_ids[static_cast<uint64_t>(query_token) * 2u + 1u];
    for (uint32_t dimension = lane; dimension < kHeadDim; dimension += blockDim.x) {
        query[dimension] = rotary_value(query_base, dimension, q_height, q_width);
    }
    __syncthreads();

    const uint32_t segment = segment_ids[query_token];
    const uint32_t begin = cu_seqlens[segment];
    const uint32_t end = cu_seqlens[segment + 1u];
    const float scale = rsqrtf(static_cast<float>(kHeadDim));
    float local_max = -1.0e30f;
    const float *key_base = qkv;
    for (uint32_t key_token = begin + lane; key_token < end; key_token += blockDim.x) {
        const float *key = key_base + static_cast<uint64_t>(key_token) * (3u * kVisionHidden) +
                kVisionHidden + static_cast<uint64_t>(head) * kHeadDim;
        const int32_t key_height = position_ids[static_cast<uint64_t>(key_token) * 2u];
        const int32_t key_width = position_ids[static_cast<uint64_t>(key_token) * 2u + 1u];
        float dot = 0.0f;
        for (uint32_t dimension = 0u; dimension < kHeadDim; ++dimension) {
            dot += query[dimension] * rotary_value(key, dimension, key_height, key_width);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduction[lane] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    const float max_score = reduction[0];
    float local_sum = 0.0f;
    for (uint32_t key_token = begin + lane; key_token < end; key_token += blockDim.x) {
        const float *key = key_base + static_cast<uint64_t>(key_token) * (3u * kVisionHidden) +
                kVisionHidden + static_cast<uint64_t>(head) * kHeadDim;
        const int32_t key_height = position_ids[static_cast<uint64_t>(key_token) * 2u];
        const int32_t key_width = position_ids[static_cast<uint64_t>(key_token) * 2u + 1u];
        float dot = 0.0f;
        for (uint32_t dimension = 0u; dimension < kHeadDim; ++dimension) {
            dot += query[dimension] * rotary_value(key, dimension, key_height, key_width);
        }
        local_sum += expf(dot * scale - max_score);
    }
    reduction[lane] = local_sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float denominator = fmaxf(reduction[0], 1.0e-20f);
    float *out = output + (static_cast<uint64_t>(query_token) * kHeads + head) * kHeadDim;
    for (uint32_t dimension = lane; dimension < kHeadDim; dimension += blockDim.x) {
        float accumulator = 0.0f;
        for (uint32_t key_token = begin; key_token < end; ++key_token) {
            const float *key = qkv + static_cast<uint64_t>(key_token) * (3u * kVisionHidden) +
                    kVisionHidden + static_cast<uint64_t>(head) * kHeadDim;
            const int32_t key_height = position_ids[static_cast<uint64_t>(key_token) * 2u];
            const int32_t key_width = position_ids[static_cast<uint64_t>(key_token) * 2u + 1u];
            float dot = 0.0f;
            for (uint32_t qdim = 0u; qdim < kHeadDim; ++qdim) {
                dot += query[qdim] * rotary_value(key, qdim, key_height, key_width);
            }
            const float probability = expf(dot * scale - max_score) / denominator;
            const float sample = *(qkv + static_cast<uint64_t>(key_token) * (3u * kVisionHidden) +
                    2u * kVisionHidden + static_cast<uint64_t>(head) * kHeadDim + dimension);
            accumulator += probability * sample;
        }
        out[dimension] = bf16_materialize_device(accumulator);
    }
}

__global__ void gather_merger_kernel(
        const float *__restrict__ input,
        float *__restrict__ output,
        uint32_t merged_tokens) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(merged_tokens) * Q4V_MERGED_HIDDEN;
    if (index >= count) return;
    const uint32_t merged = static_cast<uint32_t>(index / Q4V_MERGED_HIDDEN);
    const uint32_t offset = static_cast<uint32_t>(index % Q4V_MERGED_HIDDEN);
    const uint32_t patch = merged * kMergeUnit + offset / kVisionHidden;
    const uint32_t dimension = offset % kVisionHidden;
    output[index] = input[static_cast<uint64_t>(patch) * kVisionHidden + dimension];
}

int linear_forward(
        qwen4exp_vision_core *vision,
        const vision_linear &linear,
        const float *input,
        float *output,
        uint32_t tokens) {
    if (!vision || !linear.weight || !linear.bias || !input || !output || tokens == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t input_count = static_cast<uint64_t>(tokens) * linear.cols;
    const uint32_t blocks = static_cast<uint32_t>((input_count + kMaxThreads - 1u) / kMaxThreads);
    f32_to_bf16_kernel<<<blocks, kMaxThreads>>>(input, vision->linear_input_bf16, input_count);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (cublasSetStream(vision->cublas, nullptr) != CUBLAS_STATUS_SUCCESS) return AXIOM_ERR_CUDA;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
            vision->cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(linear.rows),
            static_cast<int>(tokens),
            static_cast<int>(linear.cols),
            &alpha,
            linear.weight,
            CUDA_R_16BF,
            static_cast<int>(linear.cols),
            vision->linear_input_bf16,
            CUDA_R_16BF,
            static_cast<int>(linear.cols),
            &beta,
            output,
            CUDA_R_32F,
            static_cast<int>(linear.rows),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) return AXIOM_ERR_CUDA;
    const uint64_t output_count = static_cast<uint64_t>(tokens) * linear.rows;
    const uint32_t output_blocks = static_cast<uint32_t>((output_count + kMaxThreads - 1u) / kMaxThreads);
    add_bias_materialize_kernel<<<output_blocks, kMaxThreads>>>(
            output, linear.bias, linear.rows, tokens);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int norm_forward(
        const vision_norm &norm,
        const float *input,
        float *output,
        uint32_t tokens) {
    if (!norm.weight || !norm.bias || !input || !output || tokens == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    layer_norm_kernel<<<tokens, kMaxThreads>>>(
            input, output, norm.weight, norm.bias, tokens, kVisionHidden, kLayerNormEps);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

int build_metadata(
        const qwen4exp_vision_core_grid *grids,
        uint32_t grid_count,
        vision_metadata *out) {
    if (!grids || grid_count == 0u || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    vision_metadata result;
    result.cu_seqlens.push_back(0u);
    uint64_t patch_count = 0u;
    uint64_t merged_count = 0u;
    uint32_t segment = 0u;
    for (uint32_t image = 0u; image < grid_count; ++image) {
        const qwen4exp_vision_core_grid grid = grids[image];
        if (grid.temporal == 0u || grid.height < kMergeSize || grid.width < kMergeSize ||
            (grid.height % kMergeSize) != 0u || (grid.width % kMergeSize) != 0u) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        uint64_t frame_patches = 0u;
        if (!checked_mul(grid.height, grid.width, &frame_patches) ||
            !checked_mul(frame_patches, grid.temporal, &frame_patches) ||
            !checked_add(patch_count, frame_patches, &patch_count)) {
            return AXIOM_ERR_BUDGET;
        }
        uint64_t frame_merged = static_cast<uint64_t>(grid.height / kMergeSize) *
                (grid.width / kMergeSize);
        if (!checked_mul(frame_merged, grid.temporal, &frame_merged) ||
            !checked_add(merged_count, frame_merged, &merged_count)) {
            return AXIOM_ERR_BUDGET;
        }
        for (uint32_t time = 0u; time < grid.temporal; ++time) {
            for (uint32_t within = 0u; within < grid.height * grid.width; ++within) {
                const uint32_t blocks_w = grid.width / kMergeSize;
                const uint32_t in_col = within % kMergeSize;
                const uint32_t in_row = (within / kMergeSize) % kMergeSize;
                const uint32_t block_col = (within / kMergeUnit) % blocks_w;
                const uint32_t block_row = within / (kMergeUnit * blocks_w);
                const uint32_t row = block_row * kMergeSize + in_row;
                const uint32_t col = block_col * kMergeSize + in_col;
                const float source_h = static_cast<float>(row) * static_cast<float>(kPositionSide - 1u) /
                        static_cast<float>(std::max<uint32_t>(grid.height - 1u, 1u));
                const float source_w = static_cast<float>(col) * static_cast<float>(kPositionSide - 1u) /
                        static_cast<float>(std::max<uint32_t>(grid.width - 1u, 1u));
                const uint32_t floor_h = static_cast<uint32_t>(std::floor(source_h));
                const uint32_t floor_w = static_cast<uint32_t>(std::floor(source_w));
                const uint32_t high_h = std::min<uint32_t>(floor_h + 1u, kPositionSide - 1u);
                const uint32_t high_w = std::min<uint32_t>(floor_w + 1u, kPositionSide - 1u);
                const float high_h_weight = source_h - static_cast<float>(floor_h);
                const float high_w_weight = source_w - static_cast<float>(floor_w);
                result.interpolation_indices.push_back(floor_h * kPositionSide + floor_w);
                result.interpolation_indices.push_back(floor_h * kPositionSide + high_w);
                result.interpolation_indices.push_back(high_h * kPositionSide + floor_w);
                result.interpolation_indices.push_back(high_h * kPositionSide + high_w);
                result.interpolation_weights.push_back((1.0f - high_h_weight) * (1.0f - high_w_weight));
                result.interpolation_weights.push_back((1.0f - high_h_weight) * high_w_weight);
                result.interpolation_weights.push_back(high_h_weight * (1.0f - high_w_weight));
                result.interpolation_weights.push_back(high_h_weight * high_w_weight);
                result.position_ids.push_back(static_cast<int32_t>(row));
                result.position_ids.push_back(static_cast<int32_t>(col));
                result.segment_ids.push_back(segment);
            }
            const uint64_t frame_end = static_cast<uint64_t>(result.cu_seqlens.back()) +
                    static_cast<uint64_t>(grid.height) * grid.width;
            if (frame_end > std::numeric_limits<uint32_t>::max()) return AXIOM_ERR_BUDGET;
            result.cu_seqlens.push_back(static_cast<uint32_t>(frame_end));
            ++segment;
        }
    }
    if (patch_count == 0u || merged_count == 0u || patch_count > std::numeric_limits<uint32_t>::max() ||
        merged_count > std::numeric_limits<uint32_t>::max() || segment == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    result.patch_count = static_cast<uint32_t>(patch_count);
    result.merged_count = static_cast<uint32_t>(merged_count);
    result.segment_count = segment;
    *out = std::move(result);
    return AXIOM_OK;
}

int upload_metadata(qwen4exp_vision_core *vision, const vision_metadata &metadata) {
    if (!vision || metadata.patch_count > vision->max_tokens ||
        metadata.cu_seqlens.size() > vision->max_tokens + 1u) {
        return AXIOM_ERR_BUDGET;
    }
    const uint64_t index_bytes = static_cast<uint64_t>(metadata.patch_count) * 4u * sizeof(uint32_t);
    const uint64_t weight_bytes = static_cast<uint64_t>(metadata.patch_count) * 4u * sizeof(float);
    const uint64_t position_bytes = static_cast<uint64_t>(metadata.patch_count) * 2u * sizeof(int32_t);
    const uint64_t segment_bytes = static_cast<uint64_t>(metadata.patch_count) * sizeof(uint32_t);
    const uint64_t cu_bytes = static_cast<uint64_t>(metadata.cu_seqlens.size()) * sizeof(uint32_t);
    int rc = cuda_status(cudaMemcpy(
            vision->interpolation_indices, metadata.interpolation_indices.data(), index_bytes,
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            vision->interpolation_weights, metadata.interpolation_weights.data(), weight_bytes,
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            vision->position_ids, metadata.position_ids.data(), position_bytes,
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            vision->segment_ids, metadata.segment_ids.data(), segment_bytes,
            cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            vision->cu_seqlens, metadata.cu_seqlens.data(), cu_bytes,
            cudaMemcpyHostToDevice));
    return rc;
}

int forward_device(
        qwen4exp_vision_core *vision,
        const float *patch_values_device,
        const vision_metadata &metadata,
        float *out_device) {
    if (!vision || !patch_values_device || !out_device || metadata.patch_count == 0u ||
        metadata.patch_count > vision->max_tokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = upload_metadata(vision, metadata);
    if (rc != AXIOM_OK) return rc;
    const uint64_t patch_bytes = static_cast<uint64_t>(metadata.patch_count) * kPatchFeatures * sizeof(float);
    if (cudaMemcpy(vision->patches, patch_values_device, patch_bytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
        return AXIOM_ERR_CUDA;
    }
    rc = linear_forward(vision, vision->patch_embed, vision->patches, vision->hidden, metadata.patch_count);
    if (rc == AXIOM_OK) {
        const uint64_t count = static_cast<uint64_t>(metadata.patch_count) * kVisionHidden;
        const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
        add_interpolated_position_kernel<<<blocks, kMaxThreads>>>(
                vision->hidden, vision->hidden, vision->pos_embed,
                vision->interpolation_indices, vision->interpolation_weights,
                metadata.patch_count, kVisionHidden);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    for (uint32_t layer = 0u; layer < kDepth && rc == AXIOM_OK; ++layer) {
        const vision_block &block = vision->blocks[layer];
        rc = norm_forward(block.norm1, vision->hidden, vision->normed, metadata.patch_count);
        if (rc == AXIOM_OK) rc = linear_forward(
                vision, block.qkv, vision->normed, vision->qkv, metadata.patch_count);
        if (rc == AXIOM_OK) {
            dim3 grid(metadata.patch_count, kHeads);
            vision_attention_kernel<<<grid, kMaxThreads>>>(
                    vision->qkv, vision->attention, vision->position_ids,
                    vision->segment_ids, vision->cu_seqlens, metadata.patch_count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = linear_forward(
                vision, block.proj, vision->attention, vision->normed, metadata.patch_count);
        if (rc == AXIOM_OK) {
            const uint64_t count = static_cast<uint64_t>(metadata.patch_count) * kVisionHidden;
            const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
            residual_add_kernel<<<blocks, kMaxThreads>>>(vision->hidden, vision->normed, count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = norm_forward(block.norm2, vision->hidden, vision->normed, metadata.patch_count);
        if (rc == AXIOM_OK) rc = linear_forward(
                vision, block.fc1, vision->normed, vision->fc1, metadata.patch_count);
        if (rc == AXIOM_OK) {
            const uint64_t count = static_cast<uint64_t>(metadata.patch_count) * kIntermediate;
            const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
            gelu_materialize_kernel<<<blocks, kMaxThreads>>>(vision->fc1, count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) rc = linear_forward(
                vision, block.fc2, vision->fc1, vision->normed, metadata.patch_count);
        if (rc == AXIOM_OK) {
            const uint64_t count = static_cast<uint64_t>(metadata.patch_count) * kVisionHidden;
            const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
            residual_add_kernel<<<blocks, kMaxThreads>>>(vision->hidden, vision->normed, count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
    }
    if (rc == AXIOM_OK) rc = norm_forward(
            vision->merger_norm, vision->hidden, vision->normed, metadata.patch_count);
    if (rc == AXIOM_OK) {
        const uint64_t count = static_cast<uint64_t>(metadata.merged_count) * Q4V_MERGED_HIDDEN;
        const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
        gather_merger_kernel<<<blocks, kMaxThreads>>>(
                vision->normed, vision->merger_input, metadata.merged_count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = linear_forward(
            vision, vision->merger_fc1, vision->merger_input,
            vision->merger_fc1_out, metadata.merged_count);
    if (rc == AXIOM_OK) {
        const uint64_t count = static_cast<uint64_t>(metadata.merged_count) * kVisionHidden * kMergeUnit;
        const uint32_t blocks = static_cast<uint32_t>((count + kMaxThreads - 1u) / kMaxThreads);
        gelu_materialize_kernel<<<blocks, kMaxThreads>>>(vision->merger_fc1_out, count);
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) rc = linear_forward(
            vision, vision->merger_fc2, vision->merger_fc1_out,
            out_device, metadata.merged_count);
    if (rc == AXIOM_OK && cudaDeviceSynchronize() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    return rc;
}

uint64_t linear_bytes(const vision_linear &linear) { return linear.device_bytes; }
uint64_t norm_bytes(const vision_norm &norm) { return norm.device_bytes; }

void add_bytes(uint64_t *total, uint64_t amount, bool *ok) {
    if (!*ok || !checked_add(*total, amount, total)) *ok = false;
}

}  // namespace

extern "C" int qwen4exp_vision_core_create(
        axiom_model *model,
        int device,
        uint32_t max_tokens,
        qwen4exp_vision_core **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || max_tokens < kMergeUnit || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen4exp_vision_core *vision = new (std::nothrow) qwen4exp_vision_core();
    if (!vision) return AXIOM_ERR_BUDGET;
    vision->device = device;
    vision->max_tokens = max_tokens;
    int rc = cublasCreate(&vision->cublas) == CUBLAS_STATUS_SUCCESS ? AXIOM_OK : AXIOM_ERR_CUDA;

        const uint64_t patch_shape[] = {kVisionHidden, 3u, kTemporalPatch,
                                    kPatchSize, kPatchSize};
    /* Conv3D is flattened as [out, in, temporal, y, x]; the linear forward
     * below consumes that same flattened row-major representation. */
    if (rc == AXIOM_OK) {
        rc = upload_bf16_tensor(model, "model.visual.patch_embed.proj.weight", 5u,
                patch_shape, &vision->patch_embed.weight, &vision->patch_embed.device_bytes);
        if (rc == AXIOM_OK) {
            const uint64_t bias_shape[] = {kVisionHidden};
            uint64_t bias_bytes = 0u;
            rc = upload_bf16_tensor(model, "model.visual.patch_embed.proj.bias", 1u,
                    bias_shape, &vision->patch_embed.bias, &bias_bytes);
            if (rc == AXIOM_OK) {
                vision->patch_embed.rows = kVisionHidden;
                vision->patch_embed.cols = kPatchFeatures;
                if (!checked_add(vision->patch_embed.device_bytes, bias_bytes,
                                 &vision->patch_embed.device_bytes)) rc = AXIOM_ERR_BUDGET;
            }
        }
    }
    if (rc == AXIOM_OK) {
        const uint64_t pos_shape[] = {kPositionEntries, kVisionHidden};
        rc = upload_bf16_tensor(model, "model.visual.pos_embed.weight", 2u,
                pos_shape, &vision->pos_embed, &vision->pos_embed_bytes);
    }
    for (uint32_t layer = 0u; layer < kDepth && rc == AXIOM_OK; ++layer) {
        const std::string prefix = "model.visual.blocks." + std::to_string(layer);
        rc = load_norm(model, prefix + ".norm1", &vision->blocks[layer].norm1, kVisionHidden);
        if (rc == AXIOM_OK) rc = load_norm(model, prefix + ".norm2", &vision->blocks[layer].norm2, kVisionHidden);
        if (rc == AXIOM_OK) rc = load_linear(model, prefix + ".attn.qkv",
                &vision->blocks[layer].qkv, 3u * kVisionHidden, kVisionHidden);
        if (rc == AXIOM_OK) rc = load_linear(model, prefix + ".attn.proj",
                &vision->blocks[layer].proj, kVisionHidden, kVisionHidden);
        if (rc == AXIOM_OK) rc = load_linear(model, prefix + ".mlp.linear_fc1",
                &vision->blocks[layer].fc1, kIntermediate, kVisionHidden);
        if (rc == AXIOM_OK) rc = load_linear(model, prefix + ".mlp.linear_fc2",
                &vision->blocks[layer].fc2, kVisionHidden, kIntermediate);
    }
    if (rc == AXIOM_OK) rc = load_norm(
            model, "model.visual.merger.norm", &vision->merger_norm, kVisionHidden);
    if (rc == AXIOM_OK) rc = load_linear(
            model, "model.visual.merger.linear_fc1", &vision->merger_fc1,
            Q4V_MERGED_HIDDEN, Q4V_MERGED_HIDDEN);
    if (rc == AXIOM_OK) rc = load_linear(
            model, "model.visual.merger.linear_fc2", &vision->merger_fc2,
            kOutput, Q4V_MERGED_HIDDEN);

    const uint64_t n = max_tokens;
    const uint64_t max_merged = (n + kMergeUnit - 1u) / kMergeUnit;
    const uint64_t max_linear_cols = Q4V_MERGED_HIDDEN;
    auto alloc_float = [&](float **pointer, uint64_t count) -> int {
        return alloc_device(reinterpret_cast<void **>(pointer), count * sizeof(float));
    };
    if (rc == AXIOM_OK) rc = alloc_float(&vision->patches, n * kPatchFeatures);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->hidden, n * kVisionHidden);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->normed, n * kVisionHidden);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->qkv, n * (3u * kVisionHidden));
    if (rc == AXIOM_OK) rc = alloc_float(&vision->attention, n * kVisionHidden);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->fc1, n * kIntermediate);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->merger_input, max_merged * max_linear_cols);
    if (rc == AXIOM_OK) rc = alloc_float(&vision->merger_fc1_out, max_merged * max_linear_cols);
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->linear_input_bf16),
            n * max_linear_cols * sizeof(uint16_t));
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->interpolation_indices),
            n * 4u * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->interpolation_weights),
            n * 4u * sizeof(float));
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->position_ids), n * 2u * sizeof(int32_t));
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->segment_ids), n * sizeof(uint32_t));
    if (rc == AXIOM_OK) rc = alloc_device(
            reinterpret_cast<void **>(&vision->cu_seqlens), (n + 1u) * sizeof(uint32_t));
    if (rc != AXIOM_OK) {
        qwen4exp_vision_core_destroy(vision);
        return rc;
    }

    bool total_ok = true;
    uint64_t loaded_count = 0u;
    uint64_t loaded_bytes = 0u;
    auto account_linear = [&](const vision_linear &linear) {
        add_bytes(&loaded_bytes, linear_bytes(linear), &total_ok);
        loaded_count += 2u;
    };
    auto account_norm = [&](const vision_norm &norm) {
        add_bytes(&loaded_bytes, norm_bytes(norm), &total_ok);
        loaded_count += 2u;
    };
    account_linear(vision->patch_embed);
    add_bytes(&loaded_bytes, vision->pos_embed_bytes, &total_ok);
    ++loaded_count;
    for (const vision_block &block : vision->blocks) {
        account_norm(block.norm1);
        account_norm(block.norm2);
        account_linear(block.qkv);
        account_linear(block.proj);
        account_linear(block.fc1);
        account_linear(block.fc2);
    }
    account_norm(vision->merger_norm);
    account_linear(vision->merger_fc1);
    account_linear(vision->merger_fc2);
    if (!total_ok) {
        qwen4exp_vision_core_destroy(vision);
        return AXIOM_ERR_BUDGET;
    }
    vision->loaded_tensor_count = loaded_count;
    vision->loaded_tensor_bytes = loaded_bytes;
    uint64_t device_bytes = loaded_bytes;
    const uint64_t scratch_bytes =
            n * kPatchFeatures * sizeof(float) + n * kVisionHidden * sizeof(float) * 3u +
            n * (3u * kVisionHidden) * sizeof(float) + n * kVisionHidden * sizeof(float) +
            n * kIntermediate * sizeof(float) + max_merged * max_linear_cols * sizeof(float) * 2u +
            n * max_linear_cols * sizeof(uint16_t) + n * 4u * sizeof(uint32_t) +
            n * 4u * sizeof(float) + n * 2u * sizeof(int32_t) + n * sizeof(uint32_t) +
            (n + 1u) * sizeof(uint32_t);
    if (!checked_add(device_bytes, scratch_bytes, &device_bytes)) {
        qwen4exp_vision_core_destroy(vision);
        return AXIOM_ERR_BUDGET;
    }
    vision->device_bytes = device_bytes;
    *out = vision;
    return AXIOM_OK;
}

extern "C" void qwen4exp_vision_core_destroy(qwen4exp_vision_core *vision) {
    if (!vision) return;
    if (vision->device >= 0) cudaSetDevice(vision->device);
    for (vision_block &block : vision->blocks) {
        destroy_linear(&block.fc2);
        destroy_linear(&block.fc1);
        destroy_linear(&block.proj);
        destroy_linear(&block.qkv);
        destroy_norm(&block.norm2);
        destroy_norm(&block.norm1);
    }
    destroy_linear(&vision->merger_fc2);
    destroy_linear(&vision->merger_fc1);
    destroy_norm(&vision->merger_norm);
    destroy_linear(&vision->patch_embed);
    cudaFree(vision->pos_embed);
    cudaFree(vision->cu_seqlens);
    cudaFree(vision->segment_ids);
    cudaFree(vision->position_ids);
    cudaFree(vision->interpolation_weights);
    cudaFree(vision->interpolation_indices);
    cudaFree(vision->linear_input_bf16);
    cudaFree(vision->merger_fc1_out);
    cudaFree(vision->merger_input);
    cudaFree(vision->fc1);
    cudaFree(vision->attention);
    cudaFree(vision->qkv);
    cudaFree(vision->normed);
    cudaFree(vision->hidden);
    cudaFree(vision->patches);
    if (vision->cublas) cublasDestroy(vision->cublas);
    delete vision;
}

extern "C" int qwen4exp_vision_core_info_get(
        const qwen4exp_vision_core *vision,
        qwen4exp_vision_core_info *out) {
    if (!vision || !out || out->abi_version != Q4V_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    *out = qwen4exp_vision_core_info{};
    out->abi_version = abi;
    out->depth = kDepth;
    out->hidden_size = kVisionHidden;
    out->intermediate_size = kIntermediate;
    out->num_heads = kHeads;
    out->head_dim = kHeadDim;
    out->patch_size = Q4V_PATCH_SIZE;
    out->temporal_patch_size = Q4V_TEMPORAL_PATCH;
    out->spatial_merge_size = kMergeSize;
    out->position_grid_side = kPositionSide;
    out->patch_feature_count = kPatchFeatures;
    out->output_size = kOutput;
    out->max_tokens = vision->max_tokens;
    out->device_bytes = vision->device_bytes;
    out->loaded_tensor_count = vision->loaded_tensor_count;
    out->loaded_tensor_bytes = vision->loaded_tensor_bytes;
    return AXIOM_OK;
}

extern "C" int qwen4exp_vision_core_forward_patches(
        qwen4exp_vision_core *vision,
        const float *patch_values,
        uint64_t patch_value_count,
        const qwen4exp_vision_core_grid *grids,
        uint32_t grid_count,
        float *out_host,
        uint64_t out_value_capacity,
        uint32_t *out_tokens) {
    if (out_tokens) *out_tokens = 0u;
    if (!vision || !patch_values || !grids || grid_count == 0u || !out_host || !out_tokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    vision_metadata metadata;
    int rc = build_metadata(grids, grid_count, &metadata);
    if (rc != AXIOM_OK) return rc;
    uint64_t expected_patch_values = 0u;
    uint64_t expected_output_values = 0u;
    if (!checked_mul(metadata.patch_count, kPatchFeatures, &expected_patch_values) ||
        !checked_mul(metadata.merged_count, kOutput, &expected_output_values) ||
        patch_value_count != expected_patch_values || out_value_capacity < expected_output_values) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (metadata.patch_count > vision->max_tokens) return AXIOM_ERR_BUDGET;
    if (cudaSetDevice(vision->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t patch_bytes = expected_patch_values * sizeof(float);
    const uint64_t output_bytes = expected_output_values * sizeof(float);
    float *patch_device = nullptr;
    float *output_device = nullptr;
    rc = alloc_device(reinterpret_cast<void **>(&patch_device), patch_bytes);
    if (rc == AXIOM_OK) rc = alloc_device(reinterpret_cast<void **>(&output_device), output_bytes);
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            patch_device, patch_values, patch_bytes, cudaMemcpyHostToDevice));
    if (rc == AXIOM_OK) rc = forward_device(vision, patch_device, metadata, output_device);
    if (rc == AXIOM_OK) rc = cuda_status(cudaMemcpy(
            out_host, output_device, output_bytes, cudaMemcpyDeviceToHost));
    cudaFree(output_device);
    cudaFree(patch_device);
    if (rc == AXIOM_OK) *out_tokens = metadata.merged_count;
    return rc;
}

extern "C" int qwen4exp_vision_core_forward_patches_device(
        qwen4exp_vision_core *vision,
        const float *patch_values_device,
        uint64_t patch_value_count,
        const qwen4exp_vision_core_grid *grids,
        uint32_t grid_count,
        float *out_device,
        uint64_t out_value_capacity,
        uint32_t *out_tokens) {
    if (out_tokens) *out_tokens = 0u;
    if (!vision || !patch_values_device || !grids || grid_count == 0u ||
        !out_device || !out_tokens) return AXIOM_ERR_INVALID_ARGUMENT;
    vision_metadata metadata;
    int rc = build_metadata(grids, grid_count, &metadata);
    if (rc != AXIOM_OK) return rc;
    uint64_t expected_patch_values = 0u;
    uint64_t expected_output_values = 0u;
    if (!checked_mul(metadata.patch_count, kPatchFeatures, &expected_patch_values) ||
        !checked_mul(metadata.merged_count, kOutput, &expected_output_values) ||
        patch_value_count != expected_patch_values || out_value_capacity < expected_output_values) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (metadata.patch_count > vision->max_tokens) return AXIOM_ERR_BUDGET;
    if (cudaSetDevice(vision->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    rc = forward_device(vision, patch_values_device, metadata, out_device);
    if (rc == AXIOM_OK) *out_tokens = metadata.merged_count;
    return rc;
}

namespace axiom::qwen4exp {
namespace {

vision_status map_core_status(int status) noexcept {
    switch (status) {
        case AXIOM_OK: return vision_status::ok;
        case AXIOM_ERR_INVALID_ARGUMENT: return vision_status::invalid_argument;
        case AXIOM_ERR_BUDGET: return vision_status::allocation_failure;
        case AXIOM_ERR_CUDA: return vision_status::cuda_error;
        case AXIOM_ERR_IO: return vision_status::checkpoint_io_error;
        default: return vision_status::unsupported_config;
    }
}

vision_status vision_fail(vision_status status,
                          std::string *error,
                          const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = "qwen4_exp vision: " + message;
        } catch (...) {
        }
    }
    return status;
}

bool checked_product(std::initializer_list<std::uint64_t> shape,
                     std::uint64_t *elements) noexcept {
    if (elements == nullptr || shape.size() == 0u) return false;
    std::uint64_t value = 1u;
    for (const std::uint64_t dimension : shape) {
        if (dimension == 0u ||
            value > std::numeric_limits<std::uint64_t>::max() / dimension) {
            return false;
        }
        value *= dimension;
    }
    *elements = value;
    return true;
}

vision_status expected_visual_tensor(
        const checkpoint_catalog &catalog,
        const std::string &name,
        std::initializer_list<std::uint64_t> shape,
        std::uint64_t *bytes,
        std::string *error) noexcept {
    const tensor_span *span = catalog.find(name);
    if (span == nullptr) {
        return vision_fail(vision_status::tensor_not_found, error,
                           "missing tensor " + name);
    }
    if (span->dtype != tensor_dtype::bf16 ||
        span->rank != shape.size()) {
        return vision_fail(vision_status::tensor_contract_mismatch, error,
                           "dtype/rank mismatch for " + name);
    }
    std::uint64_t elements = 0u;
    if (!checked_product(shape, &elements) ||
        elements > std::numeric_limits<std::uint64_t>::max() /
                           sizeof(std::uint16_t)) {
        return vision_fail(vision_status::size_overflow, error,
                           "shape overflow for " + name);
    }
    std::size_t dimension = 0u;
    for (const std::uint64_t expected : shape) {
        if (span->shape[dimension++] != expected) {
            return vision_fail(vision_status::tensor_contract_mismatch, error,
                               "shape mismatch for " + name);
        }
    }
    const std::uint64_t expected_bytes =
            elements * sizeof(std::uint16_t);
    if (span->bytes != expected_bytes ||
        *bytes > std::numeric_limits<std::uint64_t>::max() - expected_bytes) {
        return vision_fail(vision_status::tensor_contract_mismatch, error,
                           "byte-count mismatch for " + name);
    }
    *bytes += expected_bytes;
    return vision_status::ok;
}

axiom::qwen38::vision::rgb_image copy_rgb_frame(
        const vision_rgb8_frame &source,
        bool *ok) {
    axiom::qwen38::vision::rgb_image result{};
    *ok = false;
    if (source.pixels == nullptr || source.width == 0u ||
        source.height == 0u ||
        source.row_stride_bytes <
                static_cast<std::size_t>(source.width) * 3u) {
        return result;
    }
    const std::size_t tight_row =
            static_cast<std::size_t>(source.width) * 3u;
    if (source.height >
        std::numeric_limits<std::size_t>::max() / tight_row) {
        return result;
    }
    result.width = source.width;
    result.height = source.height;
    try {
        result.rgb.resize(tight_row * source.height);
    } catch (...) {
        return {};
    }
    for (std::uint32_t row = 0u; row < source.height; ++row) {
        std::memcpy(result.rgb.data() + static_cast<std::size_t>(row) * tight_row,
                    source.pixels +
                            static_cast<std::size_t>(row) *
                                    source.row_stride_bytes,
                    tight_row);
    }
    *ok = true;
    return result;
}

vision_status convert_preprocessed(
        axiom::qwen38::vision::preprocessed_media &&source,
        vision_preprocessed_media *out,
        std::string *error) noexcept {
    if (out == nullptr) {
        return vision_fail(vision_status::invalid_argument, error,
                           "preprocess output is null");
    }
    vision_preprocessed_media converted{};
    converted.grid = vision_grid{
        source.grid.temporal, source.grid.height, source.grid.width};
    converted.resized_width = source.resized_width;
    converted.resized_height = source.resized_height;
    converted.source_frames = source.source_frames;
    converted.patches = std::move(source.patches);
    *out = std::move(converted);
    return vision_status::ok;
}

}  // namespace

std::uint64_t vision_preprocessed_media::patch_count() const noexcept {
    return static_cast<std::uint64_t>(grid.temporal) * grid.height * grid.width;
}

std::uint64_t vision_preprocessed_media::merged_token_count() const noexcept {
    if ((grid.height % kVisionSpatialMergeSize) != 0u ||
        (grid.width % kVisionSpatialMergeSize) != 0u) {
        return 0u;
    }
    return static_cast<std::uint64_t>(grid.temporal) *
           (grid.height / kVisionSpatialMergeSize) *
           (grid.width / kVisionSpatialMergeSize);
}

const char *vision_status_string(vision_status status) noexcept {
    switch (status) {
        case vision_status::ok: return "ok";
        case vision_status::invalid_argument: return "invalid_argument";
        case vision_status::unsupported_config: return "unsupported_config";
        case vision_status::tensor_not_found: return "tensor_not_found";
        case vision_status::tensor_contract_mismatch:
            return "tensor_contract_mismatch";
        case vision_status::size_overflow: return "size_overflow";
        case vision_status::checkpoint_io_error: return "checkpoint_io_error";
        case vision_status::unsupported_device: return "unsupported_device";
        case vision_status::allocation_failure: return "allocation_failure";
        case vision_status::cuda_error: return "cuda_error";
        case vision_status::cublas_error: return "cublas_error";
    }
    return "unknown";
}

vision_status vision_preprocess_image_rgb8(
        const vision_rgb8_frame &image,
        std::uint32_t min_pixels,
        std::uint32_t max_pixels,
        vision_preprocessed_media *out,
        std::string *error) noexcept {
    bool copied = false;
    auto native = copy_rgb_frame(image, &copied);
    if (!copied) {
        return vision_fail(vision_status::invalid_argument, error,
                           "invalid RGB8 image");
    }
    axiom::qwen38::vision::preprocessed_media prepared{};
    std::string detail;
    if (!axiom::qwen38::vision::preprocess_image(
                native, min_pixels, max_pixels, &prepared, &detail)) {
        return vision_fail(vision_status::invalid_argument, error,
                           detail.empty() ? "image preprocessing failed" : detail);
    }
    return convert_preprocessed(std::move(prepared), out, error);
}

vision_status vision_preprocess_video_rgb8(
        const vision_rgb8_frame *frames,
        std::size_t frame_count,
        std::uint32_t min_pixels,
        std::uint32_t max_pixels,
        vision_preprocessed_media *out,
        std::string *error) noexcept {
    if (frames == nullptr || frame_count == 0u) {
        return vision_fail(vision_status::invalid_argument, error,
                           "empty RGB8 video frame sequence");
    }
    std::vector<axiom::qwen38::vision::rgb_image> native;
    try {
        native.reserve(frame_count);
        for (std::size_t index = 0u; index < frame_count; ++index) {
            bool copied = false;
            auto frame = copy_rgb_frame(frames[index], &copied);
            if (!copied) {
                return vision_fail(vision_status::invalid_argument, error,
                                   "invalid RGB8 video frame");
            }
            native.push_back(std::move(frame));
        }
    } catch (...) {
        return vision_fail(vision_status::allocation_failure, error,
                           "video frame staging allocation failed");
    }
    axiom::qwen38::vision::preprocessed_media prepared{};
    std::string detail;
    if (!axiom::qwen38::vision::preprocess_video_frames(
                native, min_pixels, max_pixels, &prepared, &detail)) {
        return vision_fail(vision_status::invalid_argument, error,
                           detail.empty() ? "video preprocessing failed" : detail);
    }
    return convert_preprocessed(std::move(prepared), out, error);
}

vision_status vision_admit_checkpoint(
        const checkpoint_catalog &catalog,
        vision_contract_report *report,
        std::string *error) noexcept {
    if (report == nullptr) {
        return vision_fail(vision_status::invalid_argument, error,
                           "contract report is null");
    }
    std::size_t visual_count = 0u;
    for (std::size_t index = 0u; index < catalog.tensor_count(); ++index) {
        const tensor_span *span = catalog.at(index);
        if (span != nullptr && span->name.rfind("model.visual.", 0u) == 0u) {
            ++visual_count;
        }
    }
    if (visual_count != kVisionTensorCount) {
        return vision_fail(vision_status::tensor_contract_mismatch, error,
                           "visual tensor count is not exactly 333");
    }

    std::uint64_t bytes = 0u;
    const auto admit = [&](const std::string &name,
                           std::initializer_list<std::uint64_t> shape) noexcept {
        return expected_visual_tensor(catalog, name, shape, &bytes, error);
    };
    vision_status status = admit(
            "model.visual.patch_embed.proj.weight",
            {kVisionHiddenSize, 3u, kVisionTemporalPatchSize,
             kVisionPatchSize, kVisionPatchSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.patch_embed.proj.bias", {kVisionHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.pos_embed.weight",
                   {kVisionPositionCount, kVisionHiddenSize});
    if (status != vision_status::ok) return status;

    for (std::size_t layer = 0u; layer < kVisionDepth; ++layer) {
        const std::string prefix =
                "model.visual.blocks." + std::to_string(layer);
        status = admit(prefix + ".norm1.weight", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".norm1.bias", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".norm2.weight", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".norm2.bias", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".attn.qkv.weight",
                       {3u * kVisionHiddenSize, kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".attn.qkv.bias",
                       {3u * kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".attn.proj.weight",
                       {kVisionHiddenSize, kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".attn.proj.bias", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".mlp.linear_fc1.weight",
                       {kVisionIntermediateSize, kVisionHiddenSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".mlp.linear_fc1.bias",
                       {kVisionIntermediateSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".mlp.linear_fc2.weight",
                       {kVisionHiddenSize, kVisionIntermediateSize});
        if (status != vision_status::ok) return status;
        status = admit(prefix + ".mlp.linear_fc2.bias", {kVisionHiddenSize});
        if (status != vision_status::ok) return status;
    }
    status = admit("model.visual.merger.norm.weight", {kVisionHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.merger.norm.bias", {kVisionHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.merger.linear_fc1.weight",
                   {kVisionMergedHiddenSize, kVisionMergedHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.merger.linear_fc1.bias",
                   {kVisionMergedHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.merger.linear_fc2.weight",
                   {kVisionOutputSize, kVisionMergedHiddenSize});
    if (status != vision_status::ok) return status;
    status = admit("model.visual.merger.linear_fc2.bias", {kVisionOutputSize});
    if (status != vision_status::ok) return status;
    constexpr std::uint64_t kPinnedVisionBytes = 897862112ull;
    if (bytes != kPinnedVisionBytes) {
        return vision_fail(vision_status::tensor_contract_mismatch, error,
                           "visual tensor bytes do not match pinned checkpoint");
    }
    *report = vision_contract_report{
        kVisionTensorCount, bytes, kVisionDepth, kVisionHiddenSize,
        kVisionIntermediateSize, kVisionHeadCount, kVisionPatchSize,
        kVisionTemporalPatchSize, kVisionSpatialMergeSize, kVisionOutputSize};
    return vision_status::ok;
}

struct resident_vision_provider::impl {
    qwen4exp_vision_core *core = nullptr;
    vision_provider_info provider_info{};

    ~impl() {
        qwen4exp_vision_core_destroy(core);
    }
};

resident_vision_provider::resident_vision_provider() = default;
resident_vision_provider::~resident_vision_provider() = default;
resident_vision_provider::resident_vision_provider(
        resident_vision_provider &&) noexcept = default;
resident_vision_provider &resident_vision_provider::operator=(
        resident_vision_provider &&) noexcept = default;

vision_status resident_vision_provider::load(
        const checkpoint_catalog &catalog,
        const vision_provider_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<resident_vision_provider> *out,
        std::string *error) noexcept {
    if (out == nullptr || config.device < 0 ||
        config.max_patch_tokens < kVisionSpatialMergeSize *
                                          kVisionSpatialMergeSize) {
        return vision_fail(vision_status::invalid_argument, error,
                           "invalid provider load request");
    }
    out->reset();
    vision_contract_report contract{};
    const vision_status admitted =
            vision_admit_checkpoint(catalog, &contract, error);
    if (admitted != vision_status::ok) return admitted;
    int count = 0;
    cudaDeviceProp properties{};
    if (cudaGetDeviceCount(&count) != cudaSuccess ||
        config.device >= count ||
        cudaGetDeviceProperties(&properties, config.device) != cudaSuccess ||
        properties.major != 12 || properties.minor != 0) {
        return vision_fail(vision_status::unsupported_device, error,
                           "Flash-Next vision requires SM120");
    }
    if (initialization_stream != nullptr &&
        cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
        return vision_fail(vision_status::cuda_error, error,
                           "initialization stream synchronization failed");
    }

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = static_cast<std::uint32_t>(config.device);
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_config);
    axiom_model *model = nullptr;
    if (rc == AXIOM_OK) {
        axiom_model_config model_config{};
        model_config.abi_version = AXIOM_ABI_VERSION;
        model_config.path = catalog.model_root().c_str();
        model_config.name = "qwen4-exp-vision";
        model_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        model_config.placement.abi_version = AXIOM_ABI_VERSION;
        model_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        rc = axiom_model_open(runtime, &model, &model_config);
    }
    std::unique_ptr<resident_vision_provider> result;
    try {
        result = std::make_unique<resident_vision_provider>();
        result->impl_ = std::make_unique<impl>();
    } catch (...) {
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return vision_fail(vision_status::allocation_failure, error,
                           "vision provider allocation failed");
    }
    if (rc == AXIOM_OK) {
        rc = qwen4exp_vision_core_create(
                model, config.device,
                static_cast<std::uint32_t>(config.max_patch_tokens),
                &result->impl_->core);
    }
    qwen4exp_vision_core_info core_info{};
    core_info.abi_version = Q4V_ABI_VERSION;
    if (rc == AXIOM_OK) {
        rc = qwen4exp_vision_core_info_get(
                result->impl_->core, &core_info);
    }
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    if (rc != AXIOM_OK) {
        return vision_fail(map_core_status(rc), error,
                           std::string("native tower load failed: ") +
                                   axiom_status_string(rc));
    }
    if (core_info.loaded_tensor_count != kVisionTensorCount ||
        core_info.loaded_tensor_bytes != contract.tensor_bytes ||
        core_info.output_size != kVisionOutputSize ||
        core_info.depth != kVisionDepth) {
        return vision_fail(vision_status::tensor_contract_mismatch, error,
                           "loaded tower does not match admitted contract");
    }
    result->impl_->provider_info = vision_provider_info{
        contract, config.max_patch_tokens, core_info.loaded_tensor_bytes,
        core_info.device_bytes, config.device};
    *out = std::move(result);
    return vision_status::ok;
}

vision_status resident_vision_provider::forward_patches(
        const float *patch_values_host,
        std::size_t patch_value_count,
        const vision_grid *grids,
        std::size_t grid_count,
        float *output_host,
        std::size_t output_value_capacity,
        std::size_t *output_tokens,
        cudaStream_t stream) noexcept {
    if (impl_ == nullptr || impl_->core == nullptr || grids == nullptr ||
        grid_count == 0u || grid_count > UINT32_MAX ||
        output_tokens == nullptr ||
        stream != nullptr) {
        return vision_status::invalid_argument;
    }
    std::vector<qwen4exp_vision_core_grid> native;
    try {
        native.reserve(grid_count);
        for (std::size_t index = 0u; index < grid_count; ++index) {
            native.push_back(qwen4exp_vision_core_grid{
                grids[index].temporal, grids[index].height, grids[index].width});
        }
    } catch (...) {
        return vision_status::allocation_failure;
    }
    std::uint32_t produced = 0u;
    const int rc = qwen4exp_vision_core_forward_patches(
            impl_->core, patch_values_host,
            static_cast<std::uint64_t>(patch_value_count), native.data(),
            static_cast<std::uint32_t>(native.size()), output_host,
            static_cast<std::uint64_t>(output_value_capacity), &produced);
    if (rc == AXIOM_OK) *output_tokens = produced;
    return map_core_status(rc);
}

vision_status resident_vision_provider::forward_patches_device(
        const float *patch_values_device,
        std::size_t patch_value_count,
        const vision_grid *grids,
        std::size_t grid_count,
        float *output_device,
        std::size_t output_value_capacity,
        std::size_t *output_tokens,
        cudaStream_t stream) noexcept {
    if (impl_ == nullptr || impl_->core == nullptr || grids == nullptr ||
        grid_count == 0u || grid_count > UINT32_MAX ||
        output_tokens == nullptr ||
        stream != nullptr) {
        return vision_status::invalid_argument;
    }
    std::vector<qwen4exp_vision_core_grid> native;
    try {
        native.reserve(grid_count);
        for (std::size_t index = 0u; index < grid_count; ++index) {
            native.push_back(qwen4exp_vision_core_grid{
                grids[index].temporal, grids[index].height, grids[index].width});
        }
    } catch (...) {
        return vision_status::allocation_failure;
    }
    std::uint32_t produced = 0u;
    const int rc = qwen4exp_vision_core_forward_patches_device(
            impl_->core, patch_values_device,
            static_cast<std::uint64_t>(patch_value_count), native.data(),
            static_cast<std::uint32_t>(native.size()), output_device,
            static_cast<std::uint64_t>(output_value_capacity), &produced);
    if (rc == AXIOM_OK) *output_tokens = produced;
    return map_core_status(rc);
}

vision_provider_info resident_vision_provider::info() const noexcept {
    return impl_ != nullptr ? impl_->provider_info : vision_provider_info{};
}

bool resident_vision_provider::initialized() const noexcept {
    return impl_ != nullptr && impl_->core != nullptr;
}

}  // namespace axiom::qwen4exp
