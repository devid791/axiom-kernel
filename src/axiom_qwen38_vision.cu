/* Native Qwen3.8/Qwen3.5 vision tower.
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
#include <stdexcept>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_vision.h"
#include "axiom/vision_memory_budget.hpp"

namespace {

constexpr uint32_t kDepth = AXIOM_QWEN38_VISION_DEPTH;
constexpr uint32_t kVisionHidden = AXIOM_QWEN38_VISION_HIDDEN;
constexpr uint32_t kIntermediate = AXIOM_QWEN38_VISION_INTERMEDIATE;
constexpr uint32_t kHeads = AXIOM_QWEN38_VISION_HEADS;
constexpr uint32_t kHeadDim = AXIOM_QWEN38_VISION_HEAD_DIM;
constexpr uint32_t kPatchFeatures = AXIOM_QWEN38_VISION_PATCH_FEATURES;
constexpr uint32_t kPatchSize = AXIOM_QWEN38_VISION_PATCH_SIZE;
constexpr uint32_t kTemporalPatch = AXIOM_QWEN38_VISION_TEMPORAL_PATCH;
constexpr uint32_t kMergeSize = AXIOM_QWEN38_VISION_MERGE_SIZE;
constexpr uint32_t kMergeUnit = kMergeSize * kMergeSize;
constexpr uint32_t kPositionSide = AXIOM_QWEN38_VISION_POSITION_SIDE;
constexpr uint32_t kPositionEntries = kPositionSide * kPositionSide;
constexpr uint32_t kOutput = AXIOM_QWEN38_VISION_OUTPUT;
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

struct vision_scratch {
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

}  // namespace

/* The ABI handle owns immutable weights and a small reusable workspace.
 * Large calls temporarily grow only scratch, never reload or copy weights. */
struct axiom_qwen38_vision : vision_scratch {
    int device = -1;
    uint32_t max_tokens = 0u;
    uint64_t loaded_tensor_count = 0u;
    uint64_t loaded_tensor_bytes = 0u;
    uint64_t device_bytes = 0u;
    cublasHandle_t cublas = nullptr;
    vision_linear patch_embed;
    uint16_t *pos_embed = nullptr;
    uint64_t pos_embed_bytes = 0u;
    vision_block blocks[AXIOM_QWEN38_VISION_DEPTH];
    vision_norm merger_norm;
    vision_linear merger_fc1;
    vision_linear merger_fc2;
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

void free_scratch(vision_scratch &s) {
    cudaFree(s.cu_seqlens); cudaFree(s.segment_ids); cudaFree(s.position_ids);
    cudaFree(s.interpolation_weights); cudaFree(s.interpolation_indices);
    cudaFree(s.linear_input_bf16); cudaFree(s.merger_fc1_out); cudaFree(s.merger_input);
    cudaFree(s.fc1); cudaFree(s.attention); cudaFree(s.qkv); cudaFree(s.normed);
    cudaFree(s.hidden); cudaFree(s.patches);
    s = vision_scratch{};
}

uint64_t scratch_size(uint32_t tokens) {
    const uint64_t n = tokens, merged = (n + kMergeUnit - 1u) / kMergeUnit;
    return n * (kPatchFeatures + 6ull * kVisionHidden + kIntermediate) * sizeof(float) +
        merged * AXIOM_QWEN38_VISION_MERGED_HIDDEN * sizeof(float) * 2u +
        n * AXIOM_QWEN38_VISION_MERGED_HIDDEN * sizeof(uint16_t) +
        n * (4u * sizeof(uint32_t) + 4u * sizeof(float) + 2u * sizeof(int32_t) + sizeof(uint32_t)) +
        (n + 1u) * sizeof(uint32_t);
}

int allocate_scratch(vision_scratch &s, uint32_t tokens) {
    const uint64_t n = tokens, merged = (n + kMergeUnit - 1u) / kMergeUnit;
    int rc = AXIOM_OK;
    auto allocate = [&](auto **pointer, uint64_t bytes) {
        if (rc == AXIOM_OK) rc = alloc_device(reinterpret_cast<void **>(pointer), bytes);
    };
    allocate(&s.patches, n * kPatchFeatures * sizeof(float));
    allocate(&s.hidden, n * kVisionHidden * sizeof(float));
    allocate(&s.normed, n * kVisionHidden * sizeof(float));
    allocate(&s.qkv, n * 3u * kVisionHidden * sizeof(float));
    allocate(&s.attention, n * kVisionHidden * sizeof(float));
    allocate(&s.fc1, n * kIntermediate * sizeof(float));
    allocate(&s.merger_input, merged * AXIOM_QWEN38_VISION_MERGED_HIDDEN * sizeof(float));
    allocate(&s.merger_fc1_out, merged * AXIOM_QWEN38_VISION_MERGED_HIDDEN * sizeof(float));
    allocate(&s.linear_input_bf16, n * AXIOM_QWEN38_VISION_MERGED_HIDDEN * sizeof(uint16_t));
    allocate(&s.interpolation_indices, n * 4u * sizeof(uint32_t));
    allocate(&s.interpolation_weights, n * 4u * sizeof(float));
    allocate(&s.position_ids, n * 2u * sizeof(int32_t));
    allocate(&s.segment_ids, n * sizeof(uint32_t));
    allocate(&s.cu_seqlens, (n + 1u) * sizeof(uint32_t));
    if (rc != AXIOM_OK) free_scratch(s);
    return rc;
}

/* Transactional, call-scoped growth. A rejected allocation cannot poison the
 * resident workspace. All capacity/device accounting is restored on every exit. */
struct scoped_scratch {
    axiom_qwen38_vision *vision;
    vision_scratch previous;
    uint32_t previous_tokens = 0u;
    explicit scoped_scratch(axiom_qwen38_vision *value) : vision(value) {}
    scoped_scratch(const scoped_scratch &) = delete;
    scoped_scratch &operator=(const scoped_scratch &) = delete;
    int ensure(uint32_t tokens, uint64_t io_bytes) {
        if (tokens <= vision->max_tokens) return AXIOM_OK;
        if (tokens > INT32_MAX) return AXIOM_ERR_BUDGET;
        size_t available = 0u, total = 0u;
        if (cudaMemGetInfo(&available, &total) != cudaSuccess) return AXIOM_ERR_CUDA;
        uint64_t needed = 0u;
        if (!checked_add(scratch_size(tokens), io_bytes, &needed) ||
            !checked_add(needed, 128ull * 1024ull * 1024ull, &needed) || needed > available)
            return AXIOM_ERR_BUDGET;
        vision_scratch replacement;
        const int rc = allocate_scratch(replacement, tokens);
        if (rc != AXIOM_OK) return rc;
        previous = static_cast<vision_scratch &>(*vision);
        previous_tokens = vision->max_tokens;
        static_cast<vision_scratch &>(*vision) = replacement;
        vision->max_tokens = tokens;
        return AXIOM_OK;
    }
    ~scoped_scratch() {
        if (!previous_tokens) return;
        free_scratch(*vision);
        static_cast<vision_scratch &>(*vision) = previous;
        vision->max_tokens = previous_tokens;
    }
};

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
        const float *__restrict__ input,
        float *__restrict__ output,
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

/* Materialize the SAME rotary_value once per Q/K dimension, rather than
 * recomputing sin/cos/pow for each query/key/value combination. All raw partner
 * dimensions are loaded before any in-place write. V and dot/softmax reduction
 * order are unchanged; this is not a different attention approximation. */
__global__ void vision_rotary_kernel(float *qkv, const int32_t *position_ids, uint32_t tokens) {
    const uint32_t token = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    if (token >= tokens || head >= kHeads) return;
    __shared__ float query[kHeadDim];
    __shared__ float key[kHeadDim];
    float *q = qkv + static_cast<uint64_t>(token) * (3u * kVisionHidden) + head * kHeadDim;
    float *k = q + kVisionHidden;
    if (lane < kHeadDim) { query[lane] = q[lane]; key[lane] = k[lane]; }
    __syncthreads();
    if (lane < kHeadDim) {
        const int32_t h = position_ids[static_cast<uint64_t>(token) * 2u];
        const int32_t w = position_ids[static_cast<uint64_t>(token) * 2u + 1u];
        q[lane] = rotary_value(query, lane, h, w);
        k[lane] = rotary_value(key, lane, h, w);
    }
}

__global__ void vision_attention_kernel(
        const float *__restrict__ qkv,
        float *__restrict__ output,
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
    for (uint32_t dimension = lane; dimension < kHeadDim; dimension += blockDim.x) {
        query[dimension] = query_base[dimension];
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
        float dot = 0.0f;
        for (uint32_t dimension = 0u; dimension < kHeadDim; ++dimension) {
            dot += query[dimension] * key[dimension];
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
        float dot = 0.0f;
        for (uint32_t dimension = 0u; dimension < kHeadDim; ++dimension) {
            dot += query[dimension] * key[dimension];
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
            float dot = 0.0f;
            for (uint32_t qdim = 0u; qdim < kHeadDim; ++qdim) {
                dot += query[qdim] * key[qdim];
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
    const uint64_t count = static_cast<uint64_t>(merged_tokens) * AXIOM_QWEN38_VISION_MERGED_HIDDEN;
    if (index >= count) return;
    const uint32_t merged = static_cast<uint32_t>(index / AXIOM_QWEN38_VISION_MERGED_HIDDEN);
    const uint32_t offset = static_cast<uint32_t>(index % AXIOM_QWEN38_VISION_MERGED_HIDDEN);
    const uint32_t patch = merged * kMergeUnit + offset / kVisionHidden;
    const uint32_t dimension = offset % kVisionHidden;
    output[index] = input[static_cast<uint64_t>(patch) * kVisionHidden + dimension];
}

int linear_forward(
        axiom_qwen38_vision *vision,
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
        const axiom_qwen38_vision_grid *grids,
        uint32_t grid_count,
        vision_metadata *out) try {
    if (!grids || grid_count == 0u || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t total_patches = 0u, total_segments = 0u;
    for (uint32_t i = 0u; i < grid_count; ++i) {
        const auto &g = grids[i];
        if (!g.temporal || g.height < kMergeSize || g.width < kMergeSize ||
            g.height % kMergeSize || g.width % kMergeSize) return AXIOM_ERR_INVALID_ARGUMENT;
        uint64_t n = 0u;
        if (!checked_mul(g.height, g.width, &n) || !checked_mul(n, g.temporal, &n) ||
            !checked_add(total_patches, n, &total_patches) || total_patches > INT32_MAX ||
            !checked_add(total_segments, g.temporal, &total_segments)) return AXIOM_ERR_BUDGET;
    }
    const uint64_t bytes = total_patches * (4u * sizeof(uint32_t) + 4u * sizeof(float) + 2u * sizeof(int32_t) + sizeof(uint32_t)) +
            (total_segments + 1u) * sizeof(uint32_t);
    if (!axiom::vision_memory::host_allocation_fits(bytes)) return AXIOM_ERR_BUDGET;
    vision_metadata result;
    result.interpolation_indices.reserve(total_patches * 4u);
    result.interpolation_weights.reserve(total_patches * 4u);
    result.position_ids.reserve(total_patches * 2u);
    result.segment_ids.reserve(total_patches);
    result.cu_seqlens.reserve(total_segments + 1u);
    result.cu_seqlens.push_back(0u);
    uint64_t patch_count = 0u;
    uint64_t merged_count = 0u;
    uint32_t segment = 0u;
    for (uint32_t image = 0u; image < grid_count; ++image) {
        const axiom_qwen38_vision_grid grid = grids[image];
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
} catch (const std::bad_alloc &) {
    return AXIOM_ERR_BUDGET;
} catch (const std::length_error &) {
    return AXIOM_ERR_BUDGET;
}

int upload_metadata(axiom_qwen38_vision *vision, const vision_metadata &metadata) {
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
        axiom_qwen38_vision *vision,
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
            vision_rotary_kernel<<<grid, kMaxThreads>>>(vision->qkv, vision->position_ids, metadata.patch_count);
            if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
        }
        if (rc == AXIOM_OK) {
            dim3 grid(metadata.patch_count, kHeads);
            vision_attention_kernel<<<grid, kMaxThreads>>>(
                    vision->qkv, vision->attention,
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
        const uint64_t count = static_cast<uint64_t>(metadata.merged_count) * AXIOM_QWEN38_VISION_MERGED_HIDDEN;
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

extern "C" int axiom_qwen38_vision_create(
        axiom_model *model,
        int device,
        uint32_t max_tokens,
        axiom_qwen38_vision **out) {
    if (out) *out = nullptr;
    if (!model || device < 0 || max_tokens < kMergeUnit || !out) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (cudaSetDevice(device) != cudaSuccess) return AXIOM_ERR_CUDA;
    axiom_qwen38_vision *vision = new (std::nothrow) axiom_qwen38_vision();
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
            AXIOM_QWEN38_VISION_MERGED_HIDDEN, AXIOM_QWEN38_VISION_MERGED_HIDDEN);
    if (rc == AXIOM_OK) rc = load_linear(
            model, "model.visual.merger.linear_fc2", &vision->merger_fc2,
            kOutput, AXIOM_QWEN38_VISION_MERGED_HIDDEN);

    if (rc == AXIOM_OK) rc = allocate_scratch(*vision, max_tokens);
    if (rc != AXIOM_OK) {
        axiom_qwen38_vision_destroy(vision);
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
        axiom_qwen38_vision_destroy(vision);
        return AXIOM_ERR_BUDGET;
    }
    vision->loaded_tensor_count = loaded_count;
    vision->loaded_tensor_bytes = loaded_bytes;
    uint64_t device_bytes = loaded_bytes;
    const uint64_t scratch_bytes = scratch_size(max_tokens);
    if (!checked_add(device_bytes, scratch_bytes, &device_bytes)) {
        axiom_qwen38_vision_destroy(vision);
        return AXIOM_ERR_BUDGET;
    }
    vision->device_bytes = device_bytes;
    *out = vision;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_vision_destroy(axiom_qwen38_vision *vision) {
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
    free_scratch(*vision);
    if (vision->cublas) cublasDestroy(vision->cublas);
    delete vision;
}

extern "C" int axiom_qwen38_vision_info_get(
        const axiom_qwen38_vision *vision,
        axiom_qwen38_vision_info *out) {
    if (!vision || !out || out->abi_version != AXIOM_QWEN38_VISION_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    *out = axiom_qwen38_vision_info{};
    out->abi_version = abi;
    out->depth = kDepth;
    out->hidden_size = kVisionHidden;
    out->intermediate_size = kIntermediate;
    out->num_heads = kHeads;
    out->head_dim = kHeadDim;
    out->patch_size = AXIOM_QWEN38_VISION_PATCH_SIZE;
    out->temporal_patch_size = AXIOM_QWEN38_VISION_TEMPORAL_PATCH;
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

extern "C" int axiom_qwen38_vision_forward_patches(
        axiom_qwen38_vision *vision,
        const float *patch_values,
        uint64_t patch_value_count,
        const axiom_qwen38_vision_grid *grids,
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
    if (cudaSetDevice(vision->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const uint64_t patch_bytes = expected_patch_values * sizeof(float);
    const uint64_t output_bytes = expected_output_values * sizeof(float);
    scoped_scratch scratch(vision);
    rc = scratch.ensure(metadata.patch_count, patch_bytes + output_bytes);
    if (rc != AXIOM_OK) return rc;
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

extern "C" int axiom_qwen38_vision_forward_patches_device(
        axiom_qwen38_vision *vision,
        const float *patch_values_device,
        uint64_t patch_value_count,
        const axiom_qwen38_vision_grid *grids,
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
    if (cudaSetDevice(vision->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    scoped_scratch scratch(vision);
    rc = scratch.ensure(metadata.patch_count, 0u);
    if (rc != AXIOM_OK) return rc;
    rc = forward_device(vision, patch_values_device, metadata, out_device);
    if (rc == AXIOM_OK) *out_tokens = metadata.merged_count;
    return rc;
}
