/* Complete native Axiom Qwen3.8-27B-NVFP4 decoder integration. */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_attention.h"
#include "axiom/qwen38_bf16_linear.h"
#include "axiom/qwen38_fp8.h"
#include "axiom/qwen38_gdn.h"
#include "axiom/qwen38_kv_tier.h"
#include "axiom/qwen38_mlp_bank.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_nvfp4_mlp.h"
#include "axiom/qwen38_speculative.h"

namespace {

constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kLayers = 64u;
constexpr uint32_t kBatch = 8u;
constexpr uint32_t kVocab = 248320u;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kTargetTapCount = AXIOM_QWEN38_MODEL_DSPARK_TARGET_TAP_COUNT;
constexpr uint32_t kTemporalWidth = AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH;
constexpr uint32_t kTargetAttentionLayers = 16u;
constexpr float kEps = 1.0e-6f;
constexpr uint64_t kEmbeddingBytes =
        static_cast<uint64_t>(kVocab) * kHidden * sizeof(uint16_t);
constexpr uint64_t kEmbeddingUploadChunkBytes = 64u * 1024u * 1024u;
constexpr uint32_t kTargetTapLayerIds[kTargetTapCount] = {4u, 16u, 28u, 40u, 52u};

using qwen38_top1_result = axiom_qwen38_model_dspark_device_top1_result;

static_assert(sizeof(qwen38_top1_result) == 12u, "top-1 result ABI must stay compact");

int cuda_status(cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

int buffer_pointer(const axiom_device_buffer *buffer, void **out) {
    return axiom_device_buffer_cuda_pointer(buffer, out);
}

int upload_zero_centered_norm(
        axiom_model *model,
        axiom_runtime *runtime,
        const char *name,
        axiom_device_buffer **out) {
    if (out) *out = nullptr;
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.dtype != AXIOM_TENSOR_DTYPE_BF16 || info.rank != 1u || info.shape[0] != kHidden ||
        info.byte_count != static_cast<uint64_t>(kHidden) * sizeof(uint16_t)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<uint16_t> encoded(kHidden);
    std::vector<float> decoded(kHidden);
    uint64_t got = 0u;
    rc = axiom_model_tensor_read(model, name, encoded.data(), info.byte_count, &got);
    if (rc != AXIOM_OK || got != info.byte_count) return rc == AXIOM_OK ? AXIOM_ERR_IO : rc;
    for (uint32_t i = 0u; i < kHidden; ++i) {
        union { uint32_t u; float f; } value{};
        value.u = static_cast<uint32_t>(encoded[i]) << 16u;
        decoded[i] = 1.0f + value.f;
    }
    axiom_device_buffer *buffer = nullptr;
    rc = axiom_device_buffer_create(runtime, &buffer, static_cast<uint64_t>(kHidden) * sizeof(float));
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_upload(
                buffer, 0u, decoded.data(), static_cast<uint64_t>(kHidden) * sizeof(float));
    }
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(buffer);
        return rc;
    }
    *out = buffer;
    return AXIOM_OK;
}

/* The old path read one BF16 embedding row from safetensors, converted it on
 * the host, copied it back to the GPU, and did that once per logical column.
 * Keep the original BF16 tensor resident instead: decode needs only eight
 * 5120-wide gathers per step. */
int load_embedding_device(
        axiom_model *checkpoint,
        axiom_runtime *runtime,
        axiom_device_buffer **out) {
    if (out) *out = nullptr;
    if (!checkpoint || !runtime || !out) return AXIOM_ERR_INVALID_ARGUMENT;

    constexpr const char *kEmbeddingName = "model.language_model.embed_tokens.weight";
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(checkpoint, kEmbeddingName, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.dtype != AXIOM_TENSOR_DTYPE_BF16 || info.rank != 2u ||
        info.shape[0] != kVocab || info.shape[1] != kHidden ||
        info.byte_count != kEmbeddingBytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    axiom_device_buffer *buffer = nullptr;
    rc = axiom_device_buffer_create(runtime, &buffer, info.byte_count);
    if (rc != AXIOM_OK) return rc;
    try {
        std::vector<uint8_t> chunk(static_cast<size_t>(
                std::min<uint64_t>(kEmbeddingUploadChunkBytes, info.byte_count)));
        for (uint64_t offset = 0u; offset < info.byte_count;) {
            const uint64_t bytes = std::min<uint64_t>(
                    static_cast<uint64_t>(chunk.size()), info.byte_count - offset);
            rc = axiom_model_tensor_read_slice(
                    checkpoint, kEmbeddingName, offset, chunk.data(), bytes);
            if (rc == AXIOM_OK) {
                rc = axiom_device_buffer_upload(buffer, offset, chunk.data(), bytes);
            }
            if (rc != AXIOM_OK) break;
            offset += bytes;
        }
    } catch (...) {
        rc = AXIOM_ERR_BUDGET;
    }
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(buffer);
        return rc;
    }
    *out = buffer;
    return AXIOM_OK;
}

/* SGLang's decoder residual lives in the model activation dtype (BF16): its
 * fused add/RMSNorm stores the summed residual before the next decoder layer
 * sees or captures it.  Keep Axiom's resident storage F32, but materialize
 * that same BF16 boundary here.  Rounding both inputs is deliberate: mixer
 * linears currently return F32 staging buffers even though their upstream
 * tensor-core output is BF16. */
__device__ __forceinline__ float qwen38_round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void qwen38_add_bf16_inplace_kernel(
        float *__restrict__ dst,
        const float *__restrict__ addend,
        uint64_t count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float lhs = qwen38_round_bf16(dst[index]);
    const float rhs = qwen38_round_bf16(addend[index]);
    dst[index] = qwen38_round_bf16(lhs + rhs);
}

/* Exact composition of the existing residual materialization, RMSNorm, and
 * BF16 output staging.  Keeping the same per-thread fmaf/reduction order is
 * important: temporal parity compares this path against the scalar B8 path
 * at zero tolerance. */
__global__ void qwen38_add_bf16_rmsnorm8_kernel(
        float *__restrict__ residual,
        const float *__restrict__ addend,
        const float *__restrict__ weight,
        float *__restrict__ norm) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch) return;
    float *residual_column = residual + static_cast<uint64_t>(column) * kHidden;
    const float *addend_column = addend + static_cast<uint64_t>(column) * kHidden;
    float *norm_column = norm + static_cast<uint64_t>(column) * kHidden;
    __shared__ float sums[kThreads];
    float sum = 0.0f;
    for (uint32_t feature = tid; feature < kHidden; feature += blockDim.x) {
        const float lhs = qwen38_round_bf16(residual_column[feature]);
        const float rhs = qwen38_round_bf16(addend_column[feature]);
        const float merged = qwen38_round_bf16(lhs + rhs);
        residual_column[feature] = merged;
        sum = fmaf(merged, merged, sum);
    }
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / static_cast<float>(kHidden) + kEps);
    for (uint32_t feature = tid; feature < kHidden; feature += blockDim.x) {
        norm_column[feature] = qwen38_round_bf16(
                residual_column[feature] * inverse * weight[feature]);
    }
}

__global__ void qwen38_rmsnorm_bf16_8_kernel(
        const float *__restrict__ weight,
        const float *__restrict__ input,
        float *__restrict__ out) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch) return;
    const float *input_column = input + static_cast<uint64_t>(column) * kHidden;
    float *out_column = out + static_cast<uint64_t>(column) * kHidden;
    __shared__ float sums[kThreads];
    float sum = 0.0f;
    for (uint32_t feature = tid; feature < kHidden; feature += blockDim.x) {
        sum = fmaf(input_column[feature], input_column[feature], sum);
    }
    sums[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(sums[0] / static_cast<float>(kHidden) + kEps);
    for (uint32_t feature = tid; feature < kHidden; feature += blockDim.x) {
        out_column[feature] = qwen38_round_bf16(
                input_column[feature] * inverse * weight[feature]);
    }
}

__global__ void qwen38_embedding_gather8_kernel(
        const uint16_t *__restrict__ embedding,
        const uint32_t *__restrict__ token_ids,
        float *__restrict__ out) {
    const uint32_t column = blockIdx.x;
    if (column >= kBatch) return;
    const uint32_t token_id = token_ids[column];
    const __nv_bfloat16 *src = reinterpret_cast<const __nv_bfloat16 *>(
            embedding + static_cast<uint64_t>(token_id) * kHidden);
    float *dst = out + static_cast<uint64_t>(column) * kHidden;
    for (uint32_t feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
        dst[feature] = __bfloat162float(src[feature]);
    }
}

__global__ void qwen38_embedding_replicate8_kernel(
        const float *__restrict__ input,
        float *__restrict__ out) {
    const uint32_t column = blockIdx.x;
    if (column >= kBatch) return;
    float *dst = out + static_cast<uint64_t>(column) * kHidden;
    for (uint32_t feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
        /* Vision output is already BF16-materialized by the native tower. */
        dst[feature] = input[feature];
    }
}

__global__ void qwen38_embedding_gather_columns_kernel(
        const uint16_t *__restrict__ embedding,
        const uint32_t *__restrict__ token_ids,
        float *__restrict__ out,
        uint32_t columns) {
    const uint32_t column = blockIdx.x;
    if (column >= columns) return;
    const uint32_t token_id = token_ids[column];
    float *dst = out + static_cast<uint64_t>(column) * kHidden;
    if (token_id >= kVocab) {
        for (uint32_t feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
            dst[feature] = 0.0f;
        }
        return;
    }
    const __nv_bfloat16 *src = reinterpret_cast<const __nv_bfloat16 *>(
            embedding + static_cast<uint64_t>(token_id) * kHidden);
    for (uint32_t feature = threadIdx.x; feature < kHidden; feature += blockDim.x) {
        dst[feature] = __bfloat162float(src[feature]);
    }
}

__global__ void qwen38_capture_tap8_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ taps,
        uint32_t tap_index) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = static_cast<uint64_t>(kHidden) * kBatch;
    if (index >= count || tap_index >= kTargetTapCount) return;
    taps[static_cast<uint64_t>(tap_index) * count + index] = hidden[index];
}

/* The scalar oracle still executes native B8, but only lane zero represents
 * its causal session.  Store that lane in the matching temporal column so the
 * parity gate compares the exact public scalar path against temporal M8. */
__global__ void qwen38_capture_tap_column_kernel(
        const float *__restrict__ hidden,
        float *__restrict__ taps,
        uint32_t tap_index,
        uint32_t temporal_column) {
    const uint32_t hidden_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidden_index >= kHidden || tap_index >= kTargetTapCount || temporal_column >= kBatch) {
        return;
    }
    const uint64_t destination =
            (static_cast<uint64_t>(tap_index) * kBatch + temporal_column) * kHidden + hidden_index;
    taps[destination] = hidden[hidden_index];
}

__global__ void qwen38_extract_temporal_tap0_kernel(
        const float *__restrict__ temporal_taps,
        float *__restrict__ prefill_taps) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t count = kTargetTapCount * kHidden;
    if (index >= count) return;
    const uint32_t tap = index / kHidden;
    const uint32_t hidden_index = index % kHidden;
    prefill_taps[static_cast<uint64_t>(tap) * kHidden + hidden_index] =
            temporal_taps[(static_cast<uint64_t>(tap) * kBatch) * kHidden + hidden_index];
}

int target_tap_index(uint32_t layer) {
    for (uint32_t tap = 0u; tap < kTargetTapCount; ++tap) {
        if (kTargetTapLayerIds[tap] == layer) return static_cast<int>(tap);
    }
    return -1;
}

/* CUDA's __float2bfloat16_rn is IEEE round-to-nearest-even.  The temporal
 * validation already copies the tap tensor to host, so reproduce that finite
 * conversion here to assert that capture happened after materialization, not
 * merely before DSpark's later FC-input staging round. */
float host_round_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16u) & 1u);
    bits &= 0xffff0000u;
    float rounded = 0.0f;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

/* The LM head stays FP8 tensor-core GEMM.  Only its 8 MiB logits tensor used
 * to cross PCIe/UMA every decode step for a CPU scan.  Select exact greedy
 * top-1 on device; equal values retain the lowest token id, matching the old
 * left-to-right host scan.  `invalid` preserves the previous fail-closed
 * finite-logit check without downloading all logits. */
__global__ void qwen38_top1_8_kernel(
        const float *__restrict__ logits,
        qwen38_top1_result *__restrict__ results) {
    const uint32_t column = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (column >= kBatch) return;
    const float *column_logits = logits + static_cast<uint64_t>(column) * kVocab;

    float best = -CUDART_INF_F;
    uint32_t best_id = 0xffffffffu;
    uint32_t invalid = 0u;
    for (uint32_t token = tid; token < kVocab; token += blockDim.x) {
        const float value = column_logits[token];
        if (!isfinite(value)) {
            invalid = 1u;
        } else if (value > best || (value == best && token < best_id)) {
            best = value;
            best_id = token;
        }
    }

    __shared__ float values[kThreads];
    __shared__ uint32_t ids[kThreads];
    __shared__ uint32_t invalids[kThreads];
    values[tid] = best;
    ids[tid] = best_id;
    invalids[tid] = invalid;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            const float right_value = values[tid + stride];
            const uint32_t right_id = ids[tid + stride];
            const float left_value = values[tid];
            const uint32_t left_id = ids[tid];
            if (right_value > left_value ||
                (right_value == left_value && right_id < left_id)) {
                values[tid] = right_value;
                ids[tid] = right_id;
            }
            invalids[tid] |= invalids[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        results[column].value = values[0];
        results[column].token_id = ids[0];
        results[column].invalid = invalids[0];
    }
}

/* The legacy top-1 ABI is compact per-column structs. The graph controller
 * needs contiguous temporal ID/logit arrays, so split them once on device. */
__global__ void qwen38_split_top1_8_kernel(
        const qwen38_top1_result *__restrict__ results,
        uint32_t *__restrict__ token_ids,
        float *__restrict__ logits) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= kBatch) return;
    /* The graph acceptance path has no host error branch.  Preserve the
     * packed reduction's fail-closed non-finite signal in its scalar view so
     * the downstream CUDA kernel can reject the whole cycle. */
    if (results[column].invalid != 0u) {
        token_ids[column] = 0u;
        logits[column] = nanf("");
        return;
    }
    token_ids[column] = results[column].token_id;
    logits[column] = results[column].value;
}

/* Device mode deliberately has no host-side context-limit branch.  Full
 * attention leaves its temporary rows untouched when this fixed M8 window
 * would overflow; turn that condition into a fail-closed acceptance input
 * before the controller can consume stale logits.  GDN's graph finalize then
 * restores its recurrent backup on the resulting async failure. */
__global__ void qwen38_validate_temporal_position_top1_kernel(
        const uint32_t *__restrict__ base_position_device,
        const uint32_t *__restrict__ verify_tokens_device,
        uint32_t max_context,
        uint32_t *__restrict__ token_ids,
        float *__restrict__ logits) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= kBatch) return;
    const uint32_t base_position = base_position_device[0];
    if (verify_tokens_device[column] >= kVocab || base_position > max_context ||
        kBatch > max_context - base_position) {
        token_ids[column] = 0u;
        logits[column] = nanf("");
    }
}

}  // namespace

struct qwen38_layer_entry {
    axiom_qwen38_gdn_layer *gdn = nullptr;
    axiom_qwen38_attention_layer *attention = nullptr;
    axiom_device_buffer *post_norm_weight = nullptr;
};

struct axiom_qwen38_model {
    int device = -1;
    uint32_t max_context = 0u;
    uint32_t position = 0u;
    uint64_t device_bytes = 0u;
    axiom_runtime *runtime = nullptr;
    axiom_model *checkpoint = nullptr;
    qwen38_layer_entry layers[kLayers]{};
    axiom_qwen38_mlp_bank *mlp_bank = nullptr;
    axiom_qwen38_fp8_linear *lm_head_fp8 = nullptr;
    axiom_qwen38_nvfp4_linear *lm_head_nvfp4 = nullptr;
    axiom_device_buffer *final_norm_weight = nullptr;
    axiom_device_buffer *embedding = nullptr;
    axiom_device_buffer *token_ids = nullptr;
    axiom_device_buffer *hidden = nullptr;
    axiom_device_buffer *mixer = nullptr;
    axiom_device_buffer *norm = nullptr;
    axiom_device_buffer *mlp = nullptr;
    axiom_device_buffer *final_hidden = nullptr;
    axiom_device_buffer *logits = nullptr;
    axiom_device_buffer *top1_results = nullptr;
    axiom_device_buffer *device_target_token_ids = nullptr;
    axiom_device_buffer *device_target_logits = nullptr;

    /* DSpark temporal verifier workspace.  Both buffers have logical shape
     * [5,8,5120] F32, tap-major.  `validation_taps` is only used by the
     * explicit scalar-versus-temporal parity gate. */
    axiom_device_buffer *temporal_taps = nullptr;
    axiom_device_buffer *validation_taps = nullptr;
    axiom_device_buffer *prefill_taps = nullptr;
    axiom_device_buffer *dspark_callback_hidden = nullptr;
    axiom_device_buffer *dspark_callback_logits = nullptr;

    axiom_qwen38_model_transaction *active_transaction = nullptr;
    axiom_qwen38_kv_tier *kv_tier = nullptr;
    std::vector<uint32_t> committed_tokens;
    bool committed_history_valid = true;
    bool suppress_history_mutation = false;
    bool scalar_tap_capture_active = false;
    uint32_t scalar_tap_capture_time = 0u;
    bool temporal_m8_validated = false;
    bool device_position_authoritative = false;
};

struct axiom_qwen38_model_transaction {
    axiom_qwen38_model *model = nullptr;
    uint32_t snapshot_position = 0u;
    uint32_t input_tokens[kTemporalWidth]{};
    /* All mutable GDN/KV state for one verify is ordered on this stream. */
    cudaStream_t stream = nullptr;
    bool forwarded = false;
    bool device_only = false;
};

extern "C" void axiom_qwen38_model_destroy(axiom_qwen38_model *model) {
    if (!model) return;
    if (model->active_transaction) {
        (void)axiom_qwen38_model_transaction_abort_stream(
                model->active_transaction, model->active_transaction->stream);
    }
    axiom_qwen38_nvfp4_linear_destroy(model->lm_head_nvfp4);
    axiom_qwen38_fp8_linear_destroy(model->lm_head_fp8);
    axiom_qwen38_mlp_bank_destroy(model->mlp_bank);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        axiom_device_buffer_destroy(model->layers[layer].post_norm_weight);
        axiom_qwen38_attention_layer_destroy(model->layers[layer].attention);
        axiom_qwen38_gdn_layer_destroy(model->layers[layer].gdn);
    }
    axiom_device_buffer_destroy(model->logits);
    axiom_device_buffer_destroy(model->final_hidden);
    axiom_device_buffer_destroy(model->mlp);
    axiom_device_buffer_destroy(model->norm);
    axiom_device_buffer_destroy(model->mixer);
    axiom_device_buffer_destroy(model->hidden);
    axiom_device_buffer_destroy(model->top1_results);
    axiom_device_buffer_destroy(model->device_target_logits);
    axiom_device_buffer_destroy(model->device_target_token_ids);
    axiom_device_buffer_destroy(model->token_ids);
    axiom_device_buffer_destroy(model->embedding);
    axiom_device_buffer_destroy(model->final_norm_weight);
    axiom_device_buffer_destroy(model->validation_taps);
    axiom_device_buffer_destroy(model->temporal_taps);
    axiom_device_buffer_destroy(model->prefill_taps);
    axiom_device_buffer_destroy(model->dspark_callback_logits);
    axiom_device_buffer_destroy(model->dspark_callback_hidden);
    axiom_model_close(model->checkpoint);
    axiom_runtime_destroy(model->runtime);
    delete model;
}

extern "C" axiom_model *axiom_qwen38_model_checkpoint(axiom_qwen38_model *model) {
    return model ? model->checkpoint : nullptr;
}

extern "C" int axiom_qwen38_model_create(
        const char *model_path,
        int device,
        uint32_t max_context,
        axiom_qwen38_model **out) {
    if (out) *out = nullptr;
    if (!model_path || !model_path[0] || !out || device < 0 || max_context == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = new (std::nothrow) axiom_qwen38_model();
    if (!model) return AXIOM_ERR_BUDGET;
    model->device = device;
    model->max_context = max_context;

    axiom_config runtime_config{};
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = static_cast<uint32_t>(device);
    int rc = axiom_runtime_create(&model->runtime, &runtime_config);
    if (rc != AXIOM_OK) {
        axiom_qwen38_model_destroy(model);
        return rc;
    }

    axiom_model_config checkpoint_config{};
    checkpoint_config.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.path = model_path;
    checkpoint_config.name = "unsloth-qwen3.8-27b-nvfp4";
    checkpoint_config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    checkpoint_config.placement.abi_version = AXIOM_ABI_VERSION;
    checkpoint_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    rc = axiom_model_open(model->runtime, &model->checkpoint, &checkpoint_config);
    if (rc != AXIOM_OK) {
        axiom_qwen38_model_destroy(model);
        return rc;
    }

    rc = load_embedding_device(model->checkpoint, model->runtime, &model->embedding);

    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        char name[160]{};
        const int written = std::snprintf(
                name, sizeof(name), "model.language_model.layers.%u.post_attention_layernorm.weight", layer);
        rc = written > 0 && static_cast<size_t>(written) < sizeof(name)
                ? upload_zero_centered_norm(model->checkpoint, model->runtime, name,
                                            &model->layers[layer].post_norm_weight)
                : AXIOM_ERR_INVALID_ARGUMENT;
        if (rc == AXIOM_OK) {
            if ((layer % 4u) == 3u) {
                rc = axiom_qwen38_attention_layer_load(
                        model->checkpoint, model->runtime, device, layer, max_context,
                        &model->layers[layer].attention);
            } else {
                rc = axiom_qwen38_gdn_layer_load(
                        model->checkpoint, model->runtime, device, layer, &model->layers[layer].gdn);
            }
        }
    }
    if (rc == AXIOM_OK) rc = axiom_qwen38_mlp_bank_load(model->checkpoint, device, &model->mlp_bank);
    if (rc == AXIOM_OK) {
        axiom_tensor_info head_info{};
        head_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_model_tensor_info_get(model->checkpoint, "lm_head.weight", &head_info);
        if (rc == AXIOM_OK && head_info.dtype == AXIOM_TENSOR_DTYPE_U8) {
            rc = axiom_qwen38_nvfp4_linear_load(
                    model->checkpoint, device, "lm_head", kVocab, kHidden,
                    &model->lm_head_nvfp4);
        } else if (rc == AXIOM_OK && head_info.dtype == AXIOM_TENSOR_DTYPE_F8_E4M3) {
            rc = axiom_qwen38_fp8_linear_load_auto(
                    model->checkpoint, device, "lm_head", &model->lm_head_fp8);
        } else if (rc == AXIOM_OK) {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    if (rc == AXIOM_OK && model->lm_head_nvfp4 &&
        (axiom_qwen38_nvfp4_linear_rows(model->lm_head_nvfp4) != kVocab ||
         axiom_qwen38_nvfp4_linear_cols(model->lm_head_nvfp4) != kHidden)) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK && model->lm_head_fp8 &&
        (axiom_qwen38_fp8_linear_rows(model->lm_head_fp8) != kVocab ||
         axiom_qwen38_fp8_linear_cols(model->lm_head_fp8) != kHidden)) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK) {
        rc = upload_zero_centered_norm(
                model->checkpoint, model->runtime, "model.language_model.norm.weight", &model->final_norm_weight);
    }
    const uint64_t hidden_batch_bytes = static_cast<uint64_t>(kHidden) * kBatch * sizeof(float);
    const uint64_t temporal_taps_bytes =
            static_cast<uint64_t>(kTargetTapCount) * kHidden * kBatch * sizeof(float);
    const uint64_t prefill_taps_bytes =
            static_cast<uint64_t>(kTargetTapCount) * kHidden * sizeof(float);
    const uint64_t logits_batch_bytes = static_cast<uint64_t>(kVocab) * kBatch * sizeof(float);
    axiom_device_buffer **workspace[] = {
        &model->hidden, &model->mixer, &model->norm, &model->mlp, &model->final_hidden,
    };
    for (axiom_device_buffer **slot : workspace) {
        if (rc == AXIOM_OK) rc = axiom_device_buffer_create(model->runtime, slot, hidden_batch_bytes);
    }
    if (rc == AXIOM_OK) rc = axiom_device_buffer_create(model->runtime, &model->logits, logits_batch_bytes);
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->token_ids, static_cast<uint64_t>(kBatch) * sizeof(uint32_t));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->top1_results,
                static_cast<uint64_t>(kBatch) * sizeof(qwen38_top1_result));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->device_target_token_ids,
                static_cast<uint64_t>(kBatch) * sizeof(uint32_t));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->device_target_logits,
                static_cast<uint64_t>(kBatch) * sizeof(float));
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(model->runtime, &model->temporal_taps, temporal_taps_bytes);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(model->runtime, &model->validation_taps, temporal_taps_bytes);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(model->runtime, &model->prefill_taps, prefill_taps_bytes);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->dspark_callback_hidden, hidden_batch_bytes);
    }
    if (rc == AXIOM_OK) {
        rc = axiom_device_buffer_create(
                model->runtime, &model->dspark_callback_logits, logits_batch_bytes);
    }
    if (rc != AXIOM_OK) {
        axiom_qwen38_model_destroy(model);
        return rc;
    }

    uint64_t bytes = 0u;
    const uint64_t direct[] = {
        axiom_qwen38_mlp_bank_device_bytes(model->mlp_bank),
        axiom_qwen38_nvfp4_linear_device_bytes(model->lm_head_nvfp4),
        axiom_qwen38_fp8_linear_device_bytes(model->lm_head_fp8),
        kEmbeddingBytes,
        static_cast<uint64_t>(kHidden) * sizeof(float), hidden_batch_bytes,
        hidden_batch_bytes, hidden_batch_bytes, hidden_batch_bytes, hidden_batch_bytes,
        logits_batch_bytes, static_cast<uint64_t>(kBatch) * sizeof(uint32_t),
        static_cast<uint64_t>(kBatch) * sizeof(qwen38_top1_result),
        static_cast<uint64_t>(kBatch) * sizeof(uint32_t),
        static_cast<uint64_t>(kBatch) * sizeof(float),
        temporal_taps_bytes, temporal_taps_bytes, prefill_taps_bytes,
        hidden_batch_bytes, logits_batch_bytes,
    };
    for (uint64_t amount : direct) {
        if (!checked_add(bytes, amount, &bytes)) {
            axiom_qwen38_model_destroy(model);
            return AXIOM_ERR_BUDGET;
        }
    }
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        if (!checked_add(bytes, static_cast<uint64_t>(kHidden) * sizeof(float), &bytes)) {
            axiom_qwen38_model_destroy(model);
            return AXIOM_ERR_BUDGET;
        }
        const uint64_t mixer_bytes = model->layers[layer].attention
                ? axiom_qwen38_attention_layer_device_bytes(model->layers[layer].attention)
                : axiom_qwen38_gdn_layer_device_bytes(model->layers[layer].gdn);
        if (!checked_add(bytes, mixer_bytes, &bytes)) {
            axiom_qwen38_model_destroy(model);
            return AXIOM_ERR_BUDGET;
        }
    }
    model->device_bytes = bytes;
    *out = model;
    return AXIOM_OK;
}

int model_reset_state(axiom_qwen38_model *model) {
    if (!model || model->active_transaction) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = AXIOM_OK;
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_reset(model->layers[layer].attention);
        } else {
            rc = axiom_qwen38_gdn_layer_reset(model->layers[layer].gdn);
        }
    }
    if (rc == AXIOM_OK) model->position = 0u;
    return rc;
}

extern "C" int axiom_qwen38_model_reset(axiom_qwen38_model *model) {
    /* The temporal parity result describes this immutable model/configuration,
     * not the contents of one request's KV history. Preserve a passed gate
     * across the request reset so production does not replay the full eight
     * scalar forwards before every short-context M8 request. A failed forward
     * clears the flag before reaching this reset and therefore remains
     * fail-closed. */
    const bool temporal_gate_was_validated = model && model->temporal_m8_validated;
    const int rc = model_reset_state(model);
    if (rc == AXIOM_OK) {
        model->committed_tokens.clear();
        model->committed_history_valid = true;
        model->scalar_tap_capture_active = false;
        model->scalar_tap_capture_time = 0u;
        model->temporal_m8_validated = temporal_gate_was_validated;
        model->device_position_authoritative = false;
    }
    return rc;
}

extern "C" uint32_t axiom_qwen38_model_position(const axiom_qwen38_model *model) {
    return model ? model->position : 0u;
}

extern "C" uint64_t axiom_qwen38_model_device_bytes(const axiom_qwen38_model *model) {
    return model ? model->device_bytes : 0u;
}

extern "C" uint64_t axiom_qwen38_model_recurrent_state_bytes(void) {
    uint64_t bytes = 0u;
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        if ((layer % 4u) == 3u) continue;
        bytes += AXIOM_QWEN38_GDN_SNAPSHOT_BYTES;
    }
    return bytes;
}

extern "C" int axiom_qwen38_model_recurrent_state_export(
        const axiom_qwen38_model *model,
        void *host_snapshot,
        const uint64_t host_snapshot_bytes) {
    if (!model || !host_snapshot || host_snapshot_bytes != axiom_qwen38_model_recurrent_state_bytes() ||
        model->active_transaction) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint8_t *cursor = static_cast<uint8_t *>(host_snapshot);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        if ((layer % 4u) == 3u) continue;
        const int rc = axiom_qwen38_gdn_layer_state_export(
                model->layers[layer].gdn, cursor, AXIOM_QWEN38_GDN_SNAPSHOT_BYTES);
        if (rc != AXIOM_OK) return rc;
        cursor += AXIOM_QWEN38_GDN_SNAPSHOT_BYTES;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_recurrent_state_import(
        axiom_qwen38_model *model,
        const void *host_snapshot,
        const uint64_t host_snapshot_bytes) {
    if (!model || !host_snapshot || host_snapshot_bytes != axiom_qwen38_model_recurrent_state_bytes() ||
        model->active_transaction) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint8_t *cursor = static_cast<const uint8_t *>(host_snapshot);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        if ((layer % 4u) == 3u) continue;
        const int rc = axiom_qwen38_gdn_layer_state_import(
                model->layers[layer].gdn, cursor, AXIOM_QWEN38_GDN_SNAPSHOT_BYTES);
        if (rc != AXIOM_OK) return rc;
        cursor += AXIOM_QWEN38_GDN_SNAPSHOT_BYTES;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_restore_position(
        axiom_qwen38_model *model,
        const uint32_t position) {
    if (!model || position > model->max_context || model->active_transaction) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = AXIOM_OK;
    for (uint32_t layer = 0u; layer < kLayers && rc == AXIOM_OK; ++layer) {
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_restore_position(
                    model->layers[layer].attention, position);
        }
    }
    if (rc == AXIOM_OK) {
        model->position = position;
        model->committed_tokens.clear();
        model->committed_history_valid = false;
        model->device_position_authoritative = false;
    }
    return rc;
}

extern "C" int axiom_qwen38_model_kv_page_export(
        const axiom_qwen38_model *model,
        const uint32_t attention_layer,
        const uint32_t logical_page,
        void *host_page,
        const uint64_t host_page_bytes) {
    if (!model || attention_layer >= kTargetAttentionLayers || !host_page) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t layer_id = 3u + attention_layer * 4u;
    return axiom_qwen38_attention_layer_kv_page_export(
            model->layers[layer_id].attention, logical_page, host_page, host_page_bytes);
}

extern "C" int axiom_qwen38_model_kv_page_import(
        axiom_qwen38_model *model,
        const uint32_t attention_layer,
        const uint32_t logical_page,
        const void *host_page,
        const uint64_t host_page_bytes) {
    if (!model || attention_layer >= kTargetAttentionLayers || !host_page) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t layer_id = 3u + attention_layer * 4u;
    return axiom_qwen38_attention_layer_kv_page_import(
            model->layers[layer_id].attention, logical_page, host_page, host_page_bytes);
}

extern "C" int axiom_qwen38_model_kv_tier_bind(
        axiom_qwen38_model *model,
        axiom_qwen38_kv_tier *tier) {
    if (!model || !tier || model->active_transaction || model->position != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_kv_tier_info tier_info{};
    tier_info.abi_version = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
    int rc = axiom_qwen38_kv_tier_info_get(tier, &tier_info);
    if (rc != AXIOM_OK || tier_info.plan.max_context != model->max_context) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    for (uint32_t attention_layer = 0u;
         attention_layer < kTargetAttentionLayers && rc == AXIOM_OK; ++attention_layer) {
        const uint32_t layer_id = 3u + attention_layer * 4u;
        rc = axiom_qwen38_attention_layer_kv_tier_bind(
                model->layers[layer_id].attention, tier, attention_layer);
    }
    if (rc == AXIOM_OK) model->kv_tier = tier;
    return rc;
}

extern "C" int axiom_qwen38_model_kv_tier_flush(
        axiom_qwen38_model *model,
        const uint32_t committed_tokens) {
    if (!model || !model->kv_tier || model->active_transaction ||
        committed_tokens > model->position || committed_tokens > model->max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = AXIOM_OK;
    for (uint32_t attention_layer = 0u;
         attention_layer < kTargetAttentionLayers && rc == AXIOM_OK; ++attention_layer) {
        const uint32_t layer_id = 3u + attention_layer * 4u;
        rc = axiom_qwen38_attention_layer_kv_tier_flush(
                model->layers[layer_id].attention);
    }
    if (rc == AXIOM_OK) rc = axiom_qwen38_kv_tier_commit(model->kv_tier, committed_tokens);
    return rc;
}

extern "C" int axiom_qwen38_model_dspark_target_embed_f32_device(
        void *user_data,
        const uint32_t *token_ids,
        float *out_hidden,
        uint32_t columns,
        void *stream) {
    axiom_qwen38_model *model = static_cast<axiom_qwen38_model *>(user_data);
    if (!model || !token_ids || !out_hidden || columns == 0u ||
        columns > kBatch || !model->embedding) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *embedding = nullptr;
    int rc = buffer_pointer(model->embedding, &embedding);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_embedding_gather_columns_kernel<<<
            columns, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const uint16_t *>(embedding), token_ids, out_hidden, columns);
    return cudaGetLastError() == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
}

extern "C" int axiom_qwen38_model_dspark_target_lm_head_f32_device(
        void *user_data,
        const float *hidden,
        float *out_logits,
        uint32_t columns,
        void *stream) {
    axiom_qwen38_model *model = static_cast<axiom_qwen38_model *>(user_data);
    if (!model || !hidden || !out_logits || columns == 0u ||
        columns > kBatch ||
        (!model->lm_head_nvfp4 && !model->lm_head_fp8)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    /* DSpark only asks for one through seven proposal columns.  The resident
     * NVFP4 head now has an exact dynamic-M entry point, so keep those caller
     * buffers device-direct instead of padding to M8 through two D2D copies.
     * The FP8 fallback is fixed-M8 and intentionally retains its staging
     * path below. */
    if (model->lm_head_nvfp4 && columns < kBatch) {
        return axiom_qwen38_nvfp4_linear_forward_f32_device_m(
                model->lm_head_nvfp4, hidden, out_logits, columns, stream);
    }
    void *staged_hidden = nullptr;
    void *staged_logits = nullptr;
    int rc = buffer_pointer(model->dspark_callback_hidden, &staged_hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->dspark_callback_logits, &staged_logits);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const size_t hidden_bytes = static_cast<size_t>(columns) * kHidden * sizeof(float);
    cudaError_t status = cudaMemcpyAsync(
            staged_hidden, hidden, hidden_bytes, cudaMemcpyDeviceToDevice, cuda_stream);
    if (status != cudaSuccess) return cuda_status(status);
    if (model->lm_head_nvfp4) {
        rc = axiom_qwen38_nvfp4_linear_forward_f32_device(
                model->lm_head_nvfp4, static_cast<const float *>(staged_hidden),
                static_cast<float *>(staged_logits), stream);
    } else {
        rc = axiom_qwen38_fp8_linear_forward_f32_device(
                model->lm_head_fp8, static_cast<const float *>(staged_hidden),
                static_cast<float *>(staged_logits), stream);
    }
    if (rc != AXIOM_OK) return rc;
    const size_t logits_bytes = static_cast<size_t>(columns) * kVocab * sizeof(float);
    status = cudaMemcpyAsync(
            out_logits, staged_logits, logits_bytes, cudaMemcpyDeviceToDevice, cuda_stream);
    return status == cudaSuccess ? AXIOM_OK : cuda_status(status);
}

int qwen38_model_forward_batch8_internal(
        axiom_qwen38_model *model,
        const uint32_t token_ids[kBatch],
        const float *embedding_override_device,
        uint32_t out_token_ids[kBatch],
        float out_logits[kBatch]) {
    if (out_token_ids) {
        for (uint32_t column = 0u; column < kBatch; ++column) out_token_ids[column] = 0u;
    }
    if (out_logits) {
        for (uint32_t column = 0u; column < kBatch; ++column) out_logits[column] = 0.0f;
    }
    if (!model || !token_ids || !out_token_ids || !out_logits || !model->checkpoint || !model->runtime ||
        model->active_transaction || model->device_position_authoritative ||
        model->position >= model->max_context ||
        (model->scalar_tap_capture_active && model->scalar_tap_capture_time >= kBatch)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t column = 0u; column < kBatch; ++column) {
        if (token_ids[column] >= kVocab) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *embedding = nullptr;
    void *token_ids_device = nullptr;
    void *hidden = nullptr;
    void *mixer = nullptr;
    void *norm = nullptr;
    void *mlp = nullptr;
    void *final_hidden = nullptr;
    void *logits = nullptr;
    void *top1_results = nullptr;
    void *scalar_capture_taps = nullptr;
    int rc = embedding_override_device ? AXIOM_OK : buffer_pointer(model->embedding, &embedding);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->token_ids, &token_ids_device);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->hidden, &hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mixer, &mixer);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mlp, &mlp);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->final_hidden, &final_hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->logits, &logits);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->top1_results, &top1_results);
    if (rc == AXIOM_OK && model->scalar_tap_capture_active) {
        rc = buffer_pointer(model->validation_taps, &scalar_capture_taps);
    }
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    cudaError_t status = cudaMemcpy(
            token_ids_device, token_ids, static_cast<size_t>(kBatch) * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return cuda_status(status);
    if (embedding_override_device) {
        qwen38_embedding_replicate8_kernel<<<kBatch, kThreads>>>(
                embedding_override_device, static_cast<float *>(hidden));
    } else {
        qwen38_embedding_gather8_kernel<<<kBatch, kThreads>>>(
                static_cast<const uint16_t *>(embedding),
                static_cast<const uint32_t *>(token_ids_device), static_cast<float *>(hidden));
    }
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * kBatch;
    const uint32_t grid = static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        /* SGLang/SpecForge capture target feature id N at the input of
         * decoder layer N (the committed residual after layer N-1).  The
         * DSpark checkpoint was trained against that exact convention. */
        const int tap_index = target_tap_index(layer);
        if (tap_index >= 0 && model->scalar_tap_capture_active) {
            qwen38_capture_tap_column_kernel<<<
                    (kHidden + kThreads - 1u) / kThreads, kThreads>>>(
                    static_cast<const float *>(hidden), static_cast<float *>(scalar_capture_taps),
                    static_cast<uint32_t>(tap_index), model->scalar_tap_capture_time);
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        }
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_forward_f32_device(
                    model->layers[layer].attention, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), nullptr);
        } else {
            rc = axiom_qwen38_gdn_layer_forward_f32_device(
                    model->layers[layer].gdn, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), nullptr);
        }
        if (rc != AXIOM_OK) return rc;
        void *post_weight = nullptr;
        rc = buffer_pointer(model->layers[layer].post_norm_weight, &post_weight);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_rmsnorm8_kernel<<<kBatch, kThreads>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mixer),
                static_cast<const float *>(post_weight), static_cast<float *>(norm));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        rc = axiom_qwen38_mlp_bank_forward_f32_device(
                model->mlp_bank, layer, static_cast<const float *>(norm), static_cast<float *>(mlp), nullptr);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_inplace_kernel<<<grid, kThreads>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mlp), hidden_count);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    void *final_weight = nullptr;
    rc = buffer_pointer(model->final_norm_weight, &final_weight);
    if (rc != AXIOM_OK) return rc;
    qwen38_rmsnorm_bf16_8_kernel<<<kBatch, kThreads>>>(
            static_cast<const float *>(final_weight), static_cast<const float *>(hidden),
            static_cast<float *>(final_hidden));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (model->lm_head_nvfp4) {
        rc = axiom_qwen38_nvfp4_linear_forward_f32_device(
                model->lm_head_nvfp4, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), nullptr);
    } else {
        rc = axiom_qwen38_fp8_linear_forward_f32_device(
                model->lm_head_fp8, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), nullptr);
    }
    if (rc != AXIOM_OK) return rc;
    qwen38_top1_8_kernel<<<kBatch, kThreads>>>(
            static_cast<const float *>(logits), static_cast<qwen38_top1_result *>(top1_results));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_top1_result results[kBatch]{};
    status = cudaMemcpy(
            results, top1_results, sizeof(results), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) return cuda_status(status);
    for (uint32_t column = 0u; column < kBatch; ++column) {
        if (results[column].invalid != 0u || results[column].token_id >= kVocab ||
            !std::isfinite(results[column].value)) {
            return AXIOM_ERR_CUDA;
        }
        out_token_ids[column] = results[column].token_id;
        out_logits[column] = results[column].value;
    }
    ++model->position;
    if (!model->suppress_history_mutation) model->committed_history_valid = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_forward_batch8(
        axiom_qwen38_model *model,
        const uint32_t token_ids[kBatch],
        uint32_t out_token_ids[kBatch],
        float out_logits[kBatch]) {
    return qwen38_model_forward_batch8_internal(
            model, token_ids, nullptr, out_token_ids, out_logits);
}

extern "C" int axiom_qwen38_model_forward_embedding(
        axiom_qwen38_model *model,
        uint32_t logical_token_id,
        const float *embedding_device,
        uint32_t *out_token_id,
        float *out_logit) {
    if (out_token_id) *out_token_id = 0u;
    if (out_logit) *out_logit = 0.0f;
    if (!model || !embedding_device || !out_token_id || !out_logit ||
        logical_token_id >= kVocab || model->active_transaction ||
        model->device_position_authoritative) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t inputs[kBatch]{};
    uint32_t outputs[kBatch]{};
    float logits[kBatch]{};
    for (uint32_t column = 0u; column < kBatch; ++column) inputs[column] = logical_token_id;
    const int rc = qwen38_model_forward_batch8_internal(
            model, inputs, embedding_device, outputs, logits);
    if (rc == AXIOM_OK) {
        *out_token_id = outputs[0];
        *out_logit = logits[0];
        /* A visual embedding cannot be reconstructed from a tokenizer id;
         * do not let the M8 validation replay treat this history as textual. */
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
    } else {
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
    }
    return rc;
}

extern "C" int axiom_qwen38_model_forward_embedding_logits(
        axiom_qwen38_model *model,
        uint32_t logical_token_id,
        const float *embedding_device,
        float *out_logits,
        uint32_t out_logits_capacity,
        uint32_t *out_greedy_token_id,
        float *out_greedy_logit) {
    if (out_greedy_token_id) *out_greedy_token_id = 0u;
    if (out_greedy_logit) *out_greedy_logit = 0.0f;
    if (!model || !embedding_device || !out_logits || out_logits_capacity < kVocab ||
        !out_greedy_token_id || !out_greedy_logit || logical_token_id >= kVocab ||
        model->active_transaction || model->device_position_authoritative) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t inputs[kBatch]{};
    uint32_t outputs[kBatch]{};
    float top1_logits[kBatch]{};
    for (uint32_t column = 0u; column < kBatch; ++column) inputs[column] = logical_token_id;
    const int rc = qwen38_model_forward_batch8_internal(
            model, inputs, embedding_device, outputs, top1_logits);
    if (rc != AXIOM_OK) {
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
        return rc;
    }
    void *device_logits = nullptr;
    const int pointer_rc = buffer_pointer(model->logits, &device_logits);
    if (pointer_rc != AXIOM_OK) return pointer_rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t copy = cudaMemcpy(
            out_logits, device_logits, static_cast<size_t>(kVocab) * sizeof(float),
            cudaMemcpyDeviceToHost);
    if (copy != cudaSuccess) return cuda_status(copy);
    for (uint32_t token = 0u; token < kVocab; ++token) {
        if (!std::isfinite(out_logits[token])) return AXIOM_ERR_CUDA;
    }
    *out_greedy_token_id = outputs[0];
    *out_greedy_logit = top1_logits[0];
    model->committed_history_valid = false;
    model->temporal_m8_validated = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_forward_token(
        axiom_qwen38_model *model,
        uint32_t token_id,
        uint32_t *out_token_id,
        float *out_logit) {
    if (out_token_id) *out_token_id = 0u;
    if (out_logit) *out_logit = 0.0f;
    if (!out_token_id || !out_logit || !model || model->active_transaction ||
        model->device_position_authoritative) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t inputs[kBatch]{};
    uint32_t outputs[kBatch]{};
    float logits[kBatch]{};
    for (uint32_t column = 0u; column < kBatch; ++column) inputs[column] = token_id;
    const bool history_was_valid = model->committed_history_valid;
    model->suppress_history_mutation = true;
    const int rc = axiom_qwen38_model_forward_batch8(model, inputs, outputs, logits);
    model->suppress_history_mutation = false;
    if (rc == AXIOM_OK) {
        *out_token_id = outputs[0];
        *out_logit = logits[0];
        if (history_was_valid) {
            try {
                model->committed_tokens.push_back(token_id);
                model->committed_history_valid = true;
            } catch (...) {
                /* The forward is still valid; only the optional future M8
                 * parity gate loses its replay history. */
                model->committed_history_valid = false;
            }
        }
    } else {
        /* A CUDA failure may have occurred after a mixer cache/state write. */
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
    }
    return rc;
}

extern "C" int axiom_qwen38_model_forward_token_logits(
        axiom_qwen38_model *model,
        uint32_t token_id,
        float *out_logits,
        uint32_t out_logits_capacity,
        uint32_t *out_greedy_token_id,
        float *out_greedy_logit) {
    if (out_greedy_token_id) *out_greedy_token_id = 0u;
    if (out_greedy_logit) *out_greedy_logit = 0.0f;
    if (!model || !out_logits || out_logits_capacity < kVocab ||
        !out_greedy_token_id || !out_greedy_logit) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t inputs[kBatch]{};
    uint32_t outputs[kBatch]{};
    float top1_logits[kBatch]{};
    for (uint32_t column = 0u; column < kBatch; ++column) inputs[column] = token_id;
    const int rc = axiom_qwen38_model_forward_batch8(model, inputs, outputs, top1_logits);
    if (rc != AXIOM_OK) return rc;

    void *device_logits = nullptr;
    const int pointer_rc = buffer_pointer(model->logits, &device_logits);
    if (pointer_rc != AXIOM_OK) return pointer_rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t copy = cudaMemcpy(
            out_logits, device_logits, static_cast<size_t>(kVocab) * sizeof(float),
            cudaMemcpyDeviceToHost);
    if (copy != cudaSuccess) return cuda_status(copy);
    for (uint32_t token = 0u; token < kVocab; ++token) {
        if (!std::isfinite(out_logits[token])) return AXIOM_ERR_CUDA;
    }
    *out_greedy_token_id = outputs[0];
    *out_greedy_logit = top1_logits[0];
    return AXIOM_OK;
}

namespace {

int model_spec_begin_layers(
        axiom_qwen38_model *model,
        void *stream,
        uint32_t *out_begun) {
    if (out_begun) *out_begun = 0u;
    if (!model || !out_begun) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_spec_begin(model->layers[layer].attention, stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_begin(model->layers[layer].gdn, stream);
        }
        if (rc != AXIOM_OK) return rc;
        *out_begun = layer + 1u;
    }
    return AXIOM_OK;
}

int model_spec_abort_layers(
        axiom_qwen38_model *model,
        uint32_t layers,
        void *stream) {
    if (!model || layers > kLayers) return AXIOM_ERR_INVALID_ARGUMENT;
    int first_error = AXIOM_OK;
    while (layers != 0u) {
        --layers;
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layers].attention) {
            rc = axiom_qwen38_attention_layer_spec_abort(model->layers[layers].attention, stream);
        } else if (model->layers[layers].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_abort(model->layers[layers].gdn, stream);
        }
        if (first_error == AXIOM_OK && rc != AXIOM_OK) first_error = rc;
    }
    return first_error;
}

int model_spec_commit_layers(
        axiom_qwen38_model *model,
        uint32_t consumed_tokens,
        void *stream) {
    if (!model || consumed_tokens == 0u || consumed_tokens > kTemporalWidth) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_spec_commit_prefix(
                    model->layers[layer].attention, consumed_tokens, stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_commit_prefix(
                    model->layers[layer].gdn, consumed_tokens, stream);
        }
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

int model_device_spec_begin_layers(
        axiom_qwen38_model *model,
        const uint32_t *anchor_position_device,
        void *stream,
        uint32_t *out_begun) {
    if (out_begun) *out_begun = 0u;
    if (!model || !anchor_position_device || !out_begun) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_spec_begin_device(
                    model->layers[layer].attention, anchor_position_device, stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_begin(model->layers[layer].gdn, stream);
        }
        if (rc != AXIOM_OK) return rc;
        *out_begun = layer + 1u;
    }
    return AXIOM_OK;
}

int model_device_spec_commit_layers(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        void *stream) {
    if (!model || !target_commit_prefix_device) return AXIOM_ERR_INVALID_ARGUMENT;
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_spec_commit_prefix_device(
                    model->layers[layer].attention, target_commit_prefix_device, stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_commit_prefix_device(
                    model->layers[layer].gdn, target_commit_prefix_device, stream);
        }
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

int model_device_spec_finalize_layers(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        const uint32_t *async_status_device,
        void *stream) {
    if (!model || !target_commit_prefix_device || !async_status_device) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        int rc = AXIOM_ERR_INVALID_ARGUMENT;
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_spec_finalize_device(
                    model->layers[layer].attention, target_commit_prefix_device,
                    async_status_device, stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_spec_finalize_device(
                    model->layers[layer].gdn, target_commit_prefix_device,
                    async_status_device, stream);
        }
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

void fill_temporal_taps_metadata(
        uint32_t token_start_position,
        const float *device_taps,
        axiom_qwen38_model_dspark_temporal_taps *out) {
    if (!out) return;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    for (uint32_t tap = 0u; tap < kTargetTapCount; ++tap) {
        out->target_layer_ids[tap] = kTargetTapLayerIds[tap];
    }
    out->token_start_position = token_start_position;
    out->temporal_tokens = kTemporalWidth;
    out->hidden_size = kHidden;
    out->target_aux_hidden = device_taps;
}

}  // namespace

extern "C" int axiom_qwen38_model_dspark_temporal_capabilities_get(
        const axiom_qwen38_model *model,
        axiom_qwen38_model_dspark_temporal_capabilities *out) {
    if (out) *out = {};
    if (!model || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    out->temporal_m8_available =
            model->temporal_m8_validated && model->committed_history_valid ? 1u : 0u;
    out->temporal_verify_width = kTemporalWidth;
    out->draft_block_size = AXIOM_QWEN38_MODEL_DSPARK_BLOCK_SIZE;
    out->target_tap_count = kTargetTapCount;
    out->hidden_size = kHidden;
    out->requires_gdn_causal_rows = 1u;
    out->requires_attention_triangular_kv = 1u;
    out->requires_prefix_state_install = 1u;
    out->temporal_m8_implemented = 1u;
    out->temporal_m8_validated = model->temporal_m8_validated ? 1u : 0u;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_transaction_begin_stream(
        axiom_qwen38_model *model,
        void *stream,
        axiom_qwen38_model_transaction **out) {
    if (out) *out = nullptr;
    if (!model || !out || model->active_transaction || model->device_position_authoritative ||
        !model->committed_history_valid ||
        model->scalar_tap_capture_active || model->position > model->max_context ||
        kTemporalWidth > model->max_context - model->position ||
        model->committed_tokens.size() != static_cast<size_t>(model->position)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model_transaction *transaction =
            new (std::nothrow) axiom_qwen38_model_transaction();
    if (!transaction) return AXIOM_ERR_BUDGET;
    uint32_t begun = 0u;
    const int rc = model_spec_begin_layers(model, stream, &begun);
    if (rc != AXIOM_OK) {
        (void)model_spec_abort_layers(model, begun, stream);
        delete transaction;
        return rc;
    }
    transaction->model = model;
    transaction->snapshot_position = model->position;
    transaction->stream = static_cast<cudaStream_t>(stream);
    model->active_transaction = transaction;
    *out = transaction;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_transaction_begin(
        axiom_qwen38_model *model,
        axiom_qwen38_model_transaction **out) {
    return axiom_qwen38_model_transaction_begin_stream(model, nullptr, out);
}

extern "C" int axiom_qwen38_model_transaction_temporal_taps_get(
        const axiom_qwen38_model_transaction *transaction,
        axiom_qwen38_model_dspark_temporal_taps *out) {
    if (out) *out = {};
    if (!transaction || !out || !transaction->model ||
        transaction->model->active_transaction != transaction || !transaction->forwarded ||
        transaction->device_only) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *taps = nullptr;
    const int rc = buffer_pointer(transaction->model->temporal_taps, &taps);
    if (rc != AXIOM_OK) return rc;
    fill_temporal_taps_metadata(
            transaction->snapshot_position, static_cast<const float *>(taps), out);
    return AXIOM_OK;
}

namespace {

/* Common enqueue path for the legacy host result and the new device-result
 * ABI.  It intentionally has no D2H operation: callers choosing the legacy
 * API perform their compatibility copy after this returns. */
int model_transaction_verify_block8_enqueue(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t input_tokens_host[kTemporalWidth],
        const uint32_t *input_tokens_device,
        cudaStream_t cuda_stream,
        const float *embedding_override_device,
        axiom_qwen38_model_dspark_verify_block8_device_result *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!transaction || !input_tokens_host || !input_tokens_device || !out || !transaction->model ||
        transaction->model->active_transaction != transaction || transaction->forwarded ||
        transaction->stream != cuda_stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    if (model->position != transaction->snapshot_position ||
        transaction->snapshot_position > model->max_context ||
        kTemporalWidth > model->max_context - transaction->snapshot_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) {
        if (input_tokens_host[time] >= kVocab) return AXIOM_ERR_INVALID_ARGUMENT;
    }

    void *embedding = nullptr;
    void *hidden = nullptr;
    void *mixer = nullptr;
    void *norm = nullptr;
    void *mlp = nullptr;
    void *final_hidden = nullptr;
    void *logits = nullptr;
    void *top1_results = nullptr;
    void *temporal_taps = nullptr;
    int rc = embedding_override_device ? AXIOM_OK : buffer_pointer(model->embedding, &embedding);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->hidden, &hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mixer, &mixer);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mlp, &mlp);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->final_hidden, &final_hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->logits, &logits);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->top1_results, &top1_results);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->temporal_taps, &temporal_taps);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    if (embedding_override_device) {
        qwen38_embedding_replicate8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
                embedding_override_device, static_cast<float *>(hidden));
    } else {
        qwen38_embedding_gather8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
                static_cast<const uint16_t *>(embedding), input_tokens_device,
                static_cast<float *>(hidden));
    }
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * kBatch;
    const uint32_t hidden_grid =
            static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        /* Match SGLang's Qwen3.8 auxiliary-hidden convention: feature layer
         * id N is the residual entering layer N, not the output of layer N. */
        const int tap_index = target_tap_index(layer);
        if (tap_index >= 0) {
            qwen38_capture_tap8_kernel<<<hidden_grid, kThreads, 0, cuda_stream>>>(
                    static_cast<const float *>(hidden), static_cast<float *>(temporal_taps),
                    static_cast<uint32_t>(tap_index));
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        }
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_forward_temporal8_f32_device(
                    model->layers[layer].attention, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), cuda_stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_forward_temporal8_f32_device(
                    model->layers[layer].gdn, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), cuda_stream);
        } else {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (rc != AXIOM_OK) return rc;
        void *post_weight = nullptr;
        rc = buffer_pointer(model->layers[layer].post_norm_weight, &post_weight);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_rmsnorm8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mixer),
                static_cast<const float *>(post_weight), static_cast<float *>(norm));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        rc = axiom_qwen38_mlp_bank_forward_f32_device(
                model->mlp_bank, layer, static_cast<const float *>(norm),
                static_cast<float *>(mlp), cuda_stream);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_inplace_kernel<<<hidden_grid, kThreads, 0, cuda_stream>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mlp), hidden_count);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    void *final_weight = nullptr;
    rc = buffer_pointer(model->final_norm_weight, &final_weight);
    if (rc != AXIOM_OK) return rc;
    qwen38_rmsnorm_bf16_8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(final_weight), static_cast<const float *>(hidden),
            static_cast<float *>(final_hidden));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (model->lm_head_nvfp4) {
        rc = axiom_qwen38_nvfp4_linear_forward_f32_device(
                model->lm_head_nvfp4, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), cuda_stream);
    } else {
        rc = axiom_qwen38_fp8_linear_forward_f32_device(
                model->lm_head_fp8, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), cuda_stream);
    }
    if (rc != AXIOM_OK) return rc;
    qwen38_top1_8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(logits), static_cast<qwen38_top1_result *>(top1_results));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    out->snapshot_position = transaction->snapshot_position;
    out->position_after_verify = transaction->snapshot_position + kTemporalWidth;
    out->target_tap_token_start_position = transaction->snapshot_position;
    out->target_tap_count = kTargetTapCount;
    out->target_tap_tokens = kTemporalWidth;
    out->target_tap_hidden_size = kHidden;
    out->target_top1_count = kTemporalWidth;
    out->target_aux_hidden = static_cast<const float *>(temporal_taps);
    out->target_top1_results = static_cast<const qwen38_top1_result *>(top1_results);
    std::memcpy(transaction->input_tokens, input_tokens_host, sizeof(transaction->input_tokens));
    transaction->forwarded = true;
    return AXIOM_OK;
}

/* Full device-only counterpart. The controller owns the token/position
 * scalars and all values remain device-resident through the greedy split. */
int model_device_transaction_verify_block8_enqueue(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t *verify_tokens_device,
        const uint32_t *anchor_position_device,
        cudaStream_t cuda_stream,
        axiom_qwen38_model_dspark_device_target_verify_view *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!transaction || !verify_tokens_device || !anchor_position_device || !out ||
        !transaction->model || !transaction->device_only ||
        transaction->model->active_transaction != transaction || transaction->forwarded ||
        transaction->stream != cuda_stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    void *embedding = nullptr;
    void *hidden = nullptr;
    void *mixer = nullptr;
    void *norm = nullptr;
    void *mlp = nullptr;
    void *final_hidden = nullptr;
    void *logits = nullptr;
    void *top1_results = nullptr;
    void *temporal_taps = nullptr;
    void *device_token_ids = nullptr;
    void *device_top1_logits = nullptr;
    int rc = buffer_pointer(model->embedding, &embedding);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->hidden, &hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mixer, &mixer);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->norm, &norm);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->mlp, &mlp);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->final_hidden, &final_hidden);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->logits, &logits);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->top1_results, &top1_results);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->temporal_taps, &temporal_taps);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->device_target_token_ids, &device_token_ids);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->device_target_logits, &device_top1_logits);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;

    /* This gather checks IDs on device. A malformed external token produces
     * a zero column instead of indexing the resident embedding out of range;
     * the post-reduction device validator turns it into NaN so the
     * controller's async status takes the authoritative rollback path. */
    qwen38_embedding_gather_columns_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const uint16_t *>(embedding), verify_tokens_device,
            static_cast<float *>(hidden), kBatch);
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    const uint64_t hidden_count = static_cast<uint64_t>(kHidden) * kBatch;
    const uint32_t hidden_grid =
            static_cast<uint32_t>((hidden_count + kThreads - 1u) / kThreads);
    for (uint32_t layer = 0u; layer < kLayers; ++layer) {
        const int tap_index = target_tap_index(layer);
        if (tap_index >= 0) {
            qwen38_capture_tap8_kernel<<<hidden_grid, kThreads, 0, cuda_stream>>>(
                    static_cast<const float *>(hidden), static_cast<float *>(temporal_taps),
                    static_cast<uint32_t>(tap_index));
            if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        }
        if (model->layers[layer].attention) {
            rc = axiom_qwen38_attention_layer_forward_temporal8_f32_device_position(
                    model->layers[layer].attention, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), anchor_position_device, cuda_stream);
        } else if (model->layers[layer].gdn) {
            rc = axiom_qwen38_gdn_layer_forward_temporal8_f32_device(
                    model->layers[layer].gdn, static_cast<const float *>(hidden),
                    static_cast<float *>(mixer), cuda_stream);
        } else {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (rc != AXIOM_OK) return rc;
        void *post_weight = nullptr;
        rc = buffer_pointer(model->layers[layer].post_norm_weight, &post_weight);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_rmsnorm8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mixer),
                static_cast<const float *>(post_weight), static_cast<float *>(norm));
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
        rc = axiom_qwen38_mlp_bank_forward_f32_device(
                model->mlp_bank, layer, static_cast<const float *>(norm),
                static_cast<float *>(mlp), cuda_stream);
        if (rc != AXIOM_OK) return rc;
        qwen38_add_bf16_inplace_kernel<<<hidden_grid, kThreads, 0, cuda_stream>>>(
                static_cast<float *>(hidden), static_cast<const float *>(mlp), hidden_count);
        if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    }
    void *final_weight = nullptr;
    rc = buffer_pointer(model->final_norm_weight, &final_weight);
    if (rc != AXIOM_OK) return rc;
    qwen38_rmsnorm_bf16_8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(final_weight), static_cast<const float *>(hidden),
            static_cast<float *>(final_hidden));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    if (model->lm_head_nvfp4) {
        rc = axiom_qwen38_nvfp4_linear_forward_f32_device(
                model->lm_head_nvfp4, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), cuda_stream);
    } else {
        rc = axiom_qwen38_fp8_linear_forward_f32_device(
                model->lm_head_fp8, static_cast<const float *>(final_hidden),
                static_cast<float *>(logits), cuda_stream);
    }
    if (rc != AXIOM_OK) return rc;
    qwen38_top1_8_kernel<<<kBatch, kThreads, 0, cuda_stream>>>(
            static_cast<const float *>(logits), static_cast<qwen38_top1_result *>(top1_results));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_split_top1_8_kernel<<<1u, kThreads, 0, cuda_stream>>>(
            static_cast<const qwen38_top1_result *>(top1_results),
            static_cast<uint32_t *>(device_token_ids), static_cast<float *>(device_top1_logits));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_validate_temporal_position_top1_kernel<<<1u, kThreads, 0, cuda_stream>>>(
            anchor_position_device, verify_tokens_device, model->max_context,
            static_cast<uint32_t *>(device_token_ids),
            static_cast<float *>(device_top1_logits));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;

    out->target_taps_device = static_cast<const float *>(temporal_taps);
    out->target_token_ids_device = static_cast<const uint32_t *>(device_token_ids);
    out->target_logits_device = static_cast<const float *>(device_top1_logits);
    out->target_tap_count = kTargetTapCount;
    out->target_tap_tokens = kTemporalWidth;
    out->target_tap_hidden_size = kHidden;
    transaction->forwarded = true;
    return AXIOM_OK;
}

}  // namespace

extern "C" int axiom_qwen38_model_transaction_verify_block8(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t input_tokens[kTemporalWidth],
        void *stream,
        axiom_qwen38_model_dspark_verify_block8_result *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!transaction || !input_tokens || !out || stream != nullptr || !transaction->model ||
        transaction->stream != nullptr || transaction->device_only) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    void *token_ids_device = nullptr;
    int rc = buffer_pointer(transaction->model->token_ids, &token_ids_device);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(transaction->model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    const cudaError_t copy_status = cudaMemcpy(
            token_ids_device, input_tokens, static_cast<size_t>(kTemporalWidth) * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    if (copy_status != cudaSuccess) return cuda_status(copy_status);
    axiom_qwen38_model_dspark_verify_block8_device_result device_result{};
    rc = model_transaction_verify_block8_enqueue(
            transaction, input_tokens, static_cast<const uint32_t *>(token_ids_device), nullptr,
            nullptr, &device_result);
    if (rc != AXIOM_OK) return rc;
    qwen38_top1_result results[kBatch]{};
    const cudaError_t result_status = cudaMemcpy(
            results, device_result.target_top1_results, sizeof(results), cudaMemcpyDeviceToHost);
    if (result_status != cudaSuccess) return cuda_status(result_status);
    out->snapshot_position = device_result.snapshot_position;
    out->position_after_verify = device_result.position_after_verify;
    out->target_tap_token_start_position = device_result.target_tap_token_start_position;
    out->target_tap_count = device_result.target_tap_count;
    out->target_tap_tokens = device_result.target_tap_tokens;
    out->target_tap_hidden_size = device_result.target_tap_hidden_size;
    out->target_aux_hidden = device_result.target_aux_hidden;
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) {
        if (results[time].invalid != 0u || results[time].token_id >= kVocab ||
            !std::isfinite(results[time].value)) {
            return AXIOM_ERR_CUDA;
        }
        out->target_token_ids[time] = results[time].token_id;
        out->target_logits[time] = results[time].value;
    }
    uint32_t accepted = 0u;
    while (accepted < AXIOM_QWEN38_MODEL_DSPARK_BLOCK_SIZE &&
           out->target_token_ids[accepted] == input_tokens[accepted + 1u]) {
        ++accepted;
    }
    out->accepted_draft_prefix = accepted;
    out->bonus_token_id = out->target_token_ids[kTemporalWidth - 1u];
    out->bonus_logit = out->target_logits[kTemporalWidth - 1u];
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_transaction_verify_block8_device(
        axiom_qwen38_model_transaction *transaction,
        const uint32_t input_tokens_host[kTemporalWidth],
        const uint32_t *input_tokens_device,
        void *stream,
        axiom_qwen38_model_dspark_verify_block8_device_result *out) {
    if (!transaction || transaction->device_only) return AXIOM_ERR_INVALID_ARGUMENT;
    return model_transaction_verify_block8_enqueue(
            transaction, input_tokens_host, input_tokens_device,
            static_cast<cudaStream_t>(stream), nullptr, out);
}

extern "C" int axiom_qwen38_model_transaction_prefill_token(
        axiom_qwen38_model_transaction *transaction,
        uint32_t token_id,
        void *stream,
        axiom_qwen38_model_dspark_prefill_token_result *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!transaction || !out || stream != nullptr || !transaction->model || transaction->device_only ||
        transaction->model->active_transaction != transaction || transaction->forwarded ||
        token_id >= kVocab) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    if (!model->temporal_m8_validated) return AXIOM_ERR_NOT_IMPLEMENTED;
    uint32_t repeated_tokens[kTemporalWidth]{};
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) repeated_tokens[time] = token_id;
    axiom_qwen38_model_dspark_verify_block8_result temporal{};
    temporal.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    int rc = axiom_qwen38_model_transaction_verify_block8(
            transaction, repeated_tokens, nullptr, &temporal);
    if (rc != AXIOM_OK) return rc;
    void *temporal_taps = nullptr;
    void *prefill_taps = nullptr;
    rc = buffer_pointer(model->temporal_taps, &temporal_taps);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->prefill_taps, &prefill_taps);
    if (rc != AXIOM_OK) return rc;
    if (cudaSetDevice(model->device) != cudaSuccess) return AXIOM_ERR_CUDA;
    qwen38_extract_temporal_tap0_kernel<<<
            (kTargetTapCount * kHidden + kThreads - 1u) / kThreads, kThreads>>>(
            static_cast<const float *>(temporal_taps), static_cast<float *>(prefill_taps));
    if (cudaGetLastError() != cudaSuccess) return AXIOM_ERR_CUDA;
    out->token_position = temporal.snapshot_position;
    out->target_tap_count = kTargetTapCount;
    out->temporal_tokens = 1u;
    out->hidden_size = kHidden;
    out->target_aux_hidden = static_cast<const float *>(prefill_taps);
    out->target_token_id = temporal.target_token_ids[0];
    out->target_logit = temporal.target_logits[0];
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_transaction_prefill_embedding(
        axiom_qwen38_model_transaction *transaction,
        uint32_t token_id,
        const float *embedding_device,
        void *stream,
        axiom_qwen38_model_dspark_prefill_token_result *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!transaction || !out || !embedding_device || stream != nullptr ||
        !transaction->model || transaction->device_only ||
        transaction->model->active_transaction != transaction || transaction->forwarded ||
        token_id >= kVocab || !transaction->model->temporal_m8_validated) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    uint32_t repeated_tokens[kTemporalWidth]{};
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) repeated_tokens[time] = token_id;
    void *token_ids_device = nullptr;
    int rc = buffer_pointer(model->token_ids, &token_ids_device);
    if (rc == AXIOM_OK && cudaSetDevice(model->device) != cudaSuccess) rc = AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpy(
                token_ids_device, repeated_tokens,
                static_cast<size_t>(kTemporalWidth) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    axiom_qwen38_model_dspark_verify_block8_device_result device_result{};
    if (rc == AXIOM_OK) {
        rc = model_transaction_verify_block8_enqueue(
                transaction, repeated_tokens, static_cast<const uint32_t *>(token_ids_device),
                nullptr, embedding_device, &device_result);
    }
    qwen38_top1_result top1[kBatch]{};
    if (rc == AXIOM_OK) {
        rc = cuda_status(cudaMemcpy(
                top1, device_result.target_top1_results, sizeof(top1), cudaMemcpyDeviceToHost));
    }
    if (rc == AXIOM_OK &&
        (top1[0].invalid != 0u || top1[0].token_id >= kVocab || !std::isfinite(top1[0].value))) {
        rc = AXIOM_ERR_CUDA;
    }
    void *temporal_taps = nullptr;
    void *prefill_taps = nullptr;
    if (rc == AXIOM_OK) rc = buffer_pointer(model->temporal_taps, &temporal_taps);
    if (rc == AXIOM_OK) rc = buffer_pointer(model->prefill_taps, &prefill_taps);
    if (rc == AXIOM_OK) {
        qwen38_extract_temporal_tap0_kernel<<<
                (kTargetTapCount * kHidden + kThreads - 1u) / kThreads, kThreads>>>(
                static_cast<const float *>(temporal_taps), static_cast<float *>(prefill_taps));
        if (cudaGetLastError() != cudaSuccess) rc = AXIOM_ERR_CUDA;
    }
    if (rc == AXIOM_OK) {
        out->token_position = transaction->snapshot_position;
        out->target_tap_count = kTargetTapCount;
        out->temporal_tokens = 1u;
        out->hidden_size = kHidden;
        out->target_aux_hidden = static_cast<const float *>(prefill_taps);
        out->target_token_id = top1[0].token_id;
        out->target_logit = top1[0].value;
    }
    return rc;
}

extern "C" int axiom_qwen38_model_transaction_commit_prefix_stream(
        axiom_qwen38_model_transaction *transaction,
        uint32_t input_prefix_count,
        void *stream) {
    if (!transaction || !transaction->model || !transaction->forwarded || transaction->device_only ||
        input_prefix_count == 0u || input_prefix_count > kTemporalWidth ||
        transaction->stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    if (model->active_transaction != transaction || model->position != transaction->snapshot_position) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const int rc = model_spec_commit_layers(model, input_prefix_count, stream);
    if (rc != AXIOM_OK) {
        /* A failed lower-layer commit cannot safely be rolled back after an
         * earlier layer has installed its prefix.  Disable future temporal
         * use instead of silently exposing mixed state. */
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
        model->active_transaction = nullptr;
        delete transaction;
        return rc;
    }
    model->position = transaction->snapshot_position + input_prefix_count;
    if (model->committed_history_valid) {
        try {
            model->committed_tokens.insert(
                    model->committed_tokens.end(), transaction->input_tokens,
                    transaction->input_tokens + input_prefix_count);
        } catch (...) {
            model->committed_history_valid = false;
            model->temporal_m8_validated = false;
        }
    }
    model->active_transaction = nullptr;
    delete transaction;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_transaction_commit_prefix(
        axiom_qwen38_model_transaction *transaction,
        uint32_t input_prefix_count) {
    return axiom_qwen38_model_transaction_commit_prefix_stream(
            transaction, input_prefix_count, nullptr);
}

extern "C" int axiom_qwen38_model_transaction_abort_stream(
        axiom_qwen38_model_transaction *transaction,
        void *stream) {
    if (!transaction || !transaction->model ||
        transaction->model->active_transaction != transaction ||
        transaction->stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model *model = transaction->model;
    const int rc = model_spec_abort_layers(model, kLayers, stream);
    if (rc != AXIOM_OK) {
        model->committed_history_valid = false;
        model->temporal_m8_validated = false;
    }
    /* model->position remains the committed snapshot position. */
    model->active_transaction = nullptr;
    delete transaction;
    return rc;
}

extern "C" int axiom_qwen38_model_transaction_abort(
        axiom_qwen38_model_transaction *transaction) {
    return axiom_qwen38_model_transaction_abort_stream(transaction, nullptr);
}

extern "C" int axiom_qwen38_model_dspark_device_transaction_begin(
        axiom_qwen38_model *model,
        const uint32_t *anchor_position_device,
        void *stream) {
    if (!model || !anchor_position_device || model->active_transaction ||
        model->scalar_tap_capture_active) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model_transaction *transaction =
            new (std::nothrow) axiom_qwen38_model_transaction();
    if (!transaction) return AXIOM_ERR_BUDGET;
    uint32_t begun = 0u;
    const int rc = model_device_spec_begin_layers(
            model, anchor_position_device, stream, &begun);
    if (rc != AXIOM_OK) {
        (void)model_spec_abort_layers(model, begun, stream);
        delete transaction;
        return rc;
    }
    transaction->model = model;
    transaction->stream = static_cast<cudaStream_t>(stream);
    transaction->device_only = true;
    model->active_transaction = transaction;
    /* The controller owns the authoritative position/history from here. */
    model->device_position_authoritative = true;
    model->committed_history_valid = false;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_model_dspark_device_transaction_verify(
        axiom_qwen38_model *model,
        const uint32_t *verify_tokens_device,
        const uint32_t *anchor_position_device,
        uint32_t verify_width,
        void *stream,
        axiom_qwen38_model_dspark_device_target_verify_view *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!model || !verify_tokens_device || !anchor_position_device || !out ||
        verify_width != kTemporalWidth || !model->active_transaction ||
        !model->active_transaction->device_only) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return model_device_transaction_verify_block8_enqueue(
            model->active_transaction, verify_tokens_device, anchor_position_device,
            static_cast<cudaStream_t>(stream), out);
}

extern "C" int axiom_qwen38_model_dspark_device_transaction_commit(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        void *stream) {
    if (!model || !target_commit_prefix_device || !model->active_transaction ||
        !model->active_transaction->device_only || !model->active_transaction->forwarded ||
        model->active_transaction->stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model_transaction *transaction = model->active_transaction;
    const int rc = model_device_spec_commit_layers(
            model, target_commit_prefix_device, stream);
    if (rc != AXIOM_OK) {
        model->temporal_m8_validated = false;
    }
    model->active_transaction = nullptr;
    delete transaction;
    return rc;
}

extern "C" int axiom_qwen38_model_dspark_device_transaction_finalize(
        axiom_qwen38_model *model,
        const uint32_t *target_commit_prefix_device,
        const uint32_t *async_status_device,
        void *stream) {
    if (!model || !target_commit_prefix_device || !async_status_device ||
        !model->active_transaction || !model->active_transaction->device_only ||
        !model->active_transaction->forwarded ||
        model->active_transaction->stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_qwen38_model_transaction *transaction = model->active_transaction;
    const int rc = model_device_spec_finalize_layers(
            model, target_commit_prefix_device, async_status_device, stream);
    if (rc != AXIOM_OK) {
        model->temporal_m8_validated = false;
    }
    model->active_transaction = nullptr;
    delete transaction;
    return rc;
}

extern "C" int axiom_qwen38_model_dspark_device_transaction_abort(
        axiom_qwen38_model *model,
        void *stream) {
    if (!model || !model->active_transaction || !model->active_transaction->device_only ||
        model->active_transaction->stream != static_cast<cudaStream_t>(stream)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_qwen38_model_transaction_abort_stream(model->active_transaction, stream);
}

namespace {

int qwen38_speculative_device_begin_callback(
        void *user_data,
        const uint32_t *anchor_position_device,
        void *stream) {
    return axiom_qwen38_model_dspark_device_transaction_begin(
            static_cast<axiom_qwen38_model *>(user_data), anchor_position_device, stream);
}

int qwen38_speculative_device_verify_callback(
        void *user_data,
        const uint32_t *verify_tokens_device,
        const uint32_t *anchor_position_device,
        uint32_t verify_width,
        void *stream,
        axiom_qwen38_speculative_device_verify_view *out) {
    if (out) *out = {};
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_qwen38_model_dspark_device_target_verify_view target_view{};
    target_view.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    const int rc = axiom_qwen38_model_dspark_device_transaction_verify(
            static_cast<axiom_qwen38_model *>(user_data), verify_tokens_device,
            anchor_position_device, verify_width, stream, &target_view);
    if (rc != AXIOM_OK) return rc;
    out->abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
    out->target_taps_device = target_view.target_taps_device;
    out->target_token_ids_device = target_view.target_token_ids_device;
    out->target_logits_device = target_view.target_logits_device;
    out->target_tap_count = target_view.target_tap_count;
    out->target_tap_tokens = target_view.target_tap_tokens;
    out->target_tap_hidden_size = target_view.target_tap_hidden_size;
    return AXIOM_OK;
}

int qwen38_speculative_device_commit_callback(
        void *user_data,
        const uint32_t *target_commit_prefix_device,
        void *stream) {
    return axiom_qwen38_model_dspark_device_transaction_commit(
            static_cast<axiom_qwen38_model *>(user_data), target_commit_prefix_device, stream);
}

int qwen38_speculative_device_abort_callback(void *user_data, void *stream) {
    return axiom_qwen38_model_dspark_device_transaction_abort(
            static_cast<axiom_qwen38_model *>(user_data), stream);
}

int qwen38_speculative_device_finalize_callback(
        void *user_data,
        const uint32_t *target_commit_prefix_device,
        const uint32_t *async_status_device,
        void *stream) {
    return axiom_qwen38_model_dspark_device_transaction_finalize(
            static_cast<axiom_qwen38_model *>(user_data), target_commit_prefix_device,
            async_status_device, stream);
}

}  // namespace

/* The speculative header owns the binding type to avoid a public include
 * cycle. Its controller exposes this factory declaration alongside bind(). */
extern "C" int axiom_qwen38_model_dspark_device_target_binding_get(
        axiom_qwen38_model *model,
        axiom_qwen38_speculative_device_target_binding *out) {
    if (out) *out = {};
    if (!model || !out || model->active_transaction) return AXIOM_ERR_INVALID_ARGUMENT;
    out->abi_version = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_ABI_VERSION;
    out->user_data = model;
    out->transaction_begin = qwen38_speculative_device_begin_callback;
    out->transaction_verify = qwen38_speculative_device_verify_callback;
    out->transaction_commit = qwen38_speculative_device_commit_callback;
    out->transaction_abort = qwen38_speculative_device_abort_callback;
    out->transaction_finalize = qwen38_speculative_device_finalize_callback;
    out->flags = AXIOM_QWEN38_SPECULATIVE_DEVICE_TARGET_FLAG_CAPTURE_SAFE;
    return AXIOM_OK;
}

namespace {

/* Resetting/replaying is used only by the explicit correctness gate.  It lets
 * one resident 27B model compare scalar and temporal execution without a
 * second full weight copy. */
int model_restore_committed_history(axiom_qwen38_model *model) {
    if (!model || model->active_transaction || !model->committed_history_valid ||
        model->committed_tokens.size() > static_cast<size_t>(model->max_context)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const bool old_suppress = model->suppress_history_mutation;
    const bool old_capture = model->scalar_tap_capture_active;
    const uint32_t old_capture_time = model->scalar_tap_capture_time;
    model->suppress_history_mutation = true;
    model->scalar_tap_capture_active = false;
    int rc = model_reset_state(model);
    uint32_t inputs[kBatch]{};
    uint32_t outputs[kBatch]{};
    float logits[kBatch]{};
    for (size_t index = 0u; index < model->committed_tokens.size() && rc == AXIOM_OK; ++index) {
        const uint32_t token = model->committed_tokens[index];
        for (uint32_t column = 0u; column < kBatch; ++column) inputs[column] = token;
        rc = axiom_qwen38_model_forward_batch8(model, inputs, outputs, logits);
    }
    model->suppress_history_mutation = old_suppress;
    model->scalar_tap_capture_active = old_capture;
    model->scalar_tap_capture_time = old_capture_time;
    if (rc != AXIOM_OK ||
        model->position != static_cast<uint32_t>(model->committed_tokens.size())) {
        model->committed_history_valid = false;
        return rc == AXIOM_OK ? AXIOM_ERR_CUDA : rc;
    }
    return AXIOM_OK;
}

int model_restore_history_snapshot(
        axiom_qwen38_model *model,
        const std::vector<uint32_t> &history) {
    if (!model || model->active_transaction || history.size() > static_cast<size_t>(model->max_context)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    try {
        model->committed_tokens = history;
    } catch (...) {
        model->committed_history_valid = false;
        return AXIOM_ERR_BUDGET;
    }
    model->committed_history_valid = true;
    return model_restore_committed_history(model);
}

}  // namespace

extern "C" int axiom_qwen38_model_dspark_temporal8_validate(
        axiom_qwen38_model *model,
        const uint32_t input_tokens[kTemporalWidth],
        float absolute_tolerance,
        axiom_qwen38_model_dspark_temporal_validation_result *out) {
    if (out) {
        *out = {};
        out->abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    }
    if (!model || !input_tokens || !out || model->active_transaction ||
        model->device_position_authoritative ||
        !model->committed_history_valid || model->scalar_tap_capture_active ||
        !std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0f ||
        model->position > model->max_context ||
        kTemporalWidth > model->max_context - model->position ||
        model->committed_tokens.size() != static_cast<size_t>(model->position)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) {
        if (input_tokens[time] >= kVocab) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<uint32_t> original_history;
    try {
        original_history = model->committed_tokens;
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    const uint32_t original_position = model->position;
    out->tested_position = original_position;
    out->tested_tokens = kTemporalWidth;
    model->temporal_m8_validated = false;

    uint32_t scalar_tokens[kTemporalWidth]{};
    float scalar_logits[kTemporalWidth]{};
    uint32_t repeated[kBatch]{};
    uint32_t scalar_outputs[kBatch]{};
    float scalar_output_logits[kBatch]{};
    const bool old_suppress = model->suppress_history_mutation;
    model->suppress_history_mutation = true;
    model->scalar_tap_capture_active = true;
    int rc = AXIOM_OK;
    for (uint32_t time = 0u; time < kTemporalWidth && rc == AXIOM_OK; ++time) {
        model->scalar_tap_capture_time = time;
        for (uint32_t column = 0u; column < kBatch; ++column) repeated[column] = input_tokens[time];
        rc = axiom_qwen38_model_forward_batch8(
                model, repeated, scalar_outputs, scalar_output_logits);
        if (rc == AXIOM_OK) {
            scalar_tokens[time] = scalar_outputs[0];
            scalar_logits[time] = scalar_output_logits[0];
        }
    }
    model->scalar_tap_capture_active = false;
    model->scalar_tap_capture_time = 0u;
    model->suppress_history_mutation = old_suppress;
    if (rc != AXIOM_OK) {
        (void)model_restore_history_snapshot(model, original_history);
        return rc;
    }
    rc = model_restore_history_snapshot(model, original_history);
    if (rc != AXIOM_OK) return rc;

    axiom_qwen38_model_transaction *transaction = nullptr;
    axiom_qwen38_model_dspark_verify_block8_result temporal{};
    temporal.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    rc = axiom_qwen38_model_transaction_begin(model, &transaction);
    if (rc == AXIOM_OK) {
        rc = axiom_qwen38_model_transaction_verify_block8(
                transaction, input_tokens, nullptr, &temporal);
    }
    std::vector<float> scalar_tap_values;
    std::vector<float> temporal_tap_values;
    if (rc == AXIOM_OK) {
        const uint64_t tap_elements =
                static_cast<uint64_t>(kTargetTapCount) * kTemporalWidth * kHidden;
        if (tap_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            rc = AXIOM_ERR_BUDGET;
        } else {
            try {
                scalar_tap_values.resize(static_cast<size_t>(tap_elements));
                temporal_tap_values.resize(static_cast<size_t>(tap_elements));
            } catch (...) {
                rc = AXIOM_ERR_BUDGET;
            }
        }
        if (rc == AXIOM_OK) {
            void *scalar_taps = nullptr;
            void *temporal_taps = nullptr;
            rc = buffer_pointer(model->validation_taps, &scalar_taps);
            if (rc == AXIOM_OK) rc = buffer_pointer(model->temporal_taps, &temporal_taps);
            if (rc == AXIOM_OK && cudaSetDevice(model->device) != cudaSuccess) rc = AXIOM_ERR_CUDA;
            cudaError_t status = cudaSuccess;
            if (rc == AXIOM_OK) {
                status = cudaMemcpy(scalar_tap_values.data(), scalar_taps,
                        scalar_tap_values.size() * sizeof(float), cudaMemcpyDeviceToHost);
            }
            if (rc == AXIOM_OK && status == cudaSuccess) {
                status = cudaMemcpy(temporal_tap_values.data(), temporal_taps,
                        temporal_tap_values.size() * sizeof(float), cudaMemcpyDeviceToHost);
            }
            if (rc == AXIOM_OK && status != cudaSuccess) rc = cuda_status(status);
        }
    }
    if (transaction) {
        const int abort_rc = axiom_qwen38_model_transaction_abort(transaction);
        transaction = nullptr;
        if (rc == AXIOM_OK && abort_rc != AXIOM_OK) rc = abort_rc;
    }
    if (rc != AXIOM_OK) {
        (void)model_restore_history_snapshot(model, original_history);
        return rc;
    }

    bool tokens_match = true;
    float max_logit_error = 0.0f;
    for (uint32_t time = 0u; time < kTemporalWidth; ++time) {
        tokens_match &= temporal.target_token_ids[time] == scalar_tokens[time];
        const float error = std::fabs(temporal.target_logits[time] - scalar_logits[time]);
        if (!std::isfinite(error)) {
            max_logit_error = std::numeric_limits<float>::infinity();
        } else {
            max_logit_error = std::max(max_logit_error, error);
        }
    }
    float max_tap_error = 0.0f;
    for (size_t index = 0u; index < scalar_tap_values.size(); ++index) {
        const float error = std::fabs(temporal_tap_values[index] - scalar_tap_values[index]);
        if (!std::isfinite(error)) {
            max_tap_error = std::numeric_limits<float>::infinity();
            break;
        }
        max_tap_error = std::max(max_tap_error, error);
    }
    float max_tap_bf16_materialization_error = 0.0f;
    const auto accumulate_tap_bf16_materialization_error = [&](const std::vector<float> &tap_values) {
        for (float value : tap_values) {
            if (!std::isfinite(value)) {
                max_tap_bf16_materialization_error = std::numeric_limits<float>::infinity();
                break;
            }
            const float error = std::fabs(value - host_round_bf16(value));
            max_tap_bf16_materialization_error =
                    std::max(max_tap_bf16_materialization_error, error);
        }
    };
    accumulate_tap_bf16_materialization_error(scalar_tap_values);
    if (std::isfinite(max_tap_bf16_materialization_error)) {
        accumulate_tap_bf16_materialization_error(temporal_tap_values);
    }
    out->max_logit_abs_error = max_logit_error;
    out->max_tap_abs_error = max_tap_error;
    out->max_tap_bf16_materialization_abs_error = max_tap_bf16_materialization_error;
    if (!tokens_match || max_logit_error > absolute_tolerance ||
        max_tap_error > absolute_tolerance || max_tap_bf16_materialization_error != 0.0f) {
        return AXIOM_OK;
    }

    /* Prefix-install probe: each commit must preserve exactly 1, 4, or 8
     * temporal positions.  Reset/replay returns to the caller's snapshot
     * after each probe, so validation itself never advances the session. */
    constexpr uint32_t kCommitProbeCounts[] = {1u, 4u, 8u};
    for (uint32_t count : kCommitProbeCounts) {
        transaction = nullptr;
        rc = axiom_qwen38_model_transaction_begin(model, &transaction);
        if (rc == AXIOM_OK) {
            axiom_qwen38_model_dspark_verify_block8_result ignored{};
            ignored.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
            rc = axiom_qwen38_model_transaction_verify_block8(
                    transaction, input_tokens, nullptr, &ignored);
        }
        if (rc == AXIOM_OK) {
            rc = axiom_qwen38_model_transaction_commit_prefix(transaction, count);
            transaction = nullptr;  // commit consumes the handle on both outcomes.
        } else if (transaction) {
            const int abort_rc = axiom_qwen38_model_transaction_abort(transaction);
            transaction = nullptr;
            if (abort_rc != AXIOM_OK) rc = abort_rc;
        }
        if (rc != AXIOM_OK || model->position != original_position + count) {
            if (rc == AXIOM_OK) rc = AXIOM_ERR_CUDA;
            (void)model_restore_history_snapshot(model, original_history);
            return rc;
        }
        rc = model_restore_history_snapshot(model, original_history);
        if (rc != AXIOM_OK || model->position != original_position) {
            return rc == AXIOM_OK ? AXIOM_ERR_CUDA : rc;
        }
    }
    model->temporal_m8_validated = true;
    out->passed = 1u;
    return AXIOM_OK;
}
