#ifndef AXIOM_QWEN38_MTP_H
#define AXIOM_QWEN38_MTP_H

/*
 * Resident loader for the native MTP block shipped with Qwen3.8 checkpoints.
 *
 * The block may live in the historical model_mtp.safetensors sidecar or be
 * embedded in a normal model-*.safetensors checkpoint shard. It is loaded in
 * its original BF16 representation and contains one full-attention decoder
 * block plus the input fusion and RMSNorm weights; it does not contain a
 * second vocabulary embedding table or LM head. Those are shared with the
 * base Qwen3.8 checkpoint.
 *
 * This component is a loader and a strict forward ABI contract, not an MTP
 * executor. All MTP RMSNorm tensors use Qwen3.5's zero-centered form
 * `x * rsqrt(mean(x^2) + eps) * (1 + weight)`. A speculative executor borrows
 * these immutable device buffers, provides the base model's post-final-norm
 * hidden state and embedding for a proposed token, and owns transactional MTP
 * KV state separately.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_MTP_HIDDEN 5120u
#define AXIOM_QWEN38_MTP_INTERMEDIATE 17408u
#define AXIOM_QWEN38_MTP_BATCH 8u
#define AXIOM_QWEN38_MTP_LAYERS 1u
#define AXIOM_QWEN38_MTP_ATTENTION_HEADS 24u
#define AXIOM_QWEN38_MTP_KV_HEADS 4u
#define AXIOM_QWEN38_MTP_HEAD_DIM 256u
#define AXIOM_QWEN38_MTP_VOCAB 248320u
#define AXIOM_QWEN38_MTP_FC_INPUT 10240u
#define AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT 15u

typedef struct axiom_qwen38_mtp axiom_qwen38_mtp;

/* The order is stable so a future CUDA executor can acquire immutable weight
 * buffers without duplicating checkpoint-name string handling. */
typedef enum {
    AXIOM_QWEN38_MTP_TENSOR_FC_WEIGHT = 0,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_INPUT_LAYERNORM_WEIGHT = 1,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_DOWN_PROJ_WEIGHT = 2,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_GATE_PROJ_WEIGHT = 3,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_UP_PROJ_WEIGHT = 4,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_POST_ATTENTION_LAYERNORM_WEIGHT = 5,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_NORM_WEIGHT = 6,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_PROJ_WEIGHT = 7,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_O_PROJ_WEIGHT = 8,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_NORM_WEIGHT = 9,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_PROJ_WEIGHT = 10,
    AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_V_PROJ_WEIGHT = 11,
    AXIOM_QWEN38_MTP_TENSOR_NORM_WEIGHT = 12,
    AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_EMBEDDING_WEIGHT = 13,
    AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_HIDDEN_WEIGHT = 14,
} axiom_qwen38_mtp_tensor;

typedef struct {
    uint32_t abi_version;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t mtp_layers;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t tensor_count;
    axiom_tensor_dtype resident_dtype;
    uint64_t checkpoint_bytes;
    uint64_t device_bytes;
} axiom_qwen38_mtp_info;

typedef struct {
    uint32_t abi_version;
    axiom_qwen38_mtp_tensor tensor;
    axiom_tensor_dtype dtype;
    uint32_t rank;
    uint64_t shape[AXIOM_MAX_TENSOR_DIMS];
    uint64_t byte_count;
    char name[128];
} axiom_qwen38_mtp_tensor_info;

/* Contract for the executor that will consume this resident sidecar.
 *
 * `target_hidden`, `proposed_token_embedding`, and `draft_hidden` use CUDA
 * F32 column-major [hidden_size, batch] buffers.  `target_hidden` is the
 * base decoder's post-final-RMSNorm output. `proposed_token_embedding` comes
 * from model.language_model.embed_tokens.weight for the draft token.  The MTP
 * output is post mtp.norm and must be projected with the base lm_head.
 */
typedef struct {
    uint32_t abi_version;
    uint32_t scalar_dtype;
    uint32_t hidden_size;
    uint32_t batch;
    uint32_t mtp_layers;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t uses_base_embedding;
    uint32_t uses_base_lm_head;
    uint32_t requires_mtp_kv_transaction;
    uint32_t requires_target_cache_transaction;
    uint32_t output_is_post_mtp_norm;
    uint32_t zero_centered_rmsnorm;
    uint32_t attention_output_gate;
} axiom_qwen38_mtp_forward_contract;

typedef struct {
    uint32_t abi_version;
    const float *target_hidden;
    const float *proposed_token_embedding;
    float *draft_hidden;
    uint32_t batch;
    uint32_t position;
    uint32_t spec_step;
    void *stream;
    uint64_t flags;
} axiom_qwen38_mtp_forward_request;

/* `model` must have been opened on the Qwen3.8 model directory, not on a
 * single safetensors file. All fifteen MTP tensors must come atomically from
 * one model_mtp.safetensors or model-*.safetensors file in that opened
 * checkpoint, retain their original BF16 dtype, and match the Qwen3.8
 * geometry. `runtime` must be a CUDA Axiom runtime for `device`. */
int axiom_qwen38_mtp_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        axiom_qwen38_mtp **out);

void axiom_qwen38_mtp_destroy(axiom_qwen38_mtp *mtp);
uint64_t axiom_qwen38_mtp_device_bytes(const axiom_qwen38_mtp *mtp);

int axiom_qwen38_mtp_info_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_info *out);

int axiom_qwen38_mtp_tensor_info_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_tensor tensor,
        axiom_qwen38_mtp_tensor_info *out);

/* Borrow an immutable Axiom-owned BF16 weight buffer. The pointer remains
 * valid only while `mtp` remains alive. A CUDA executor may obtain its raw
 * CUDA address through axiom_device_buffer_cuda_pointer(). */
int axiom_qwen38_mtp_tensor_buffer_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_tensor tensor,
        const axiom_device_buffer **out);

int axiom_qwen38_mtp_forward_contract_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_forward_contract *out);

/* Validate a planned MTP launch against this sidecar's ABI. This does not
 * execute the MTP block or mutate any cache. The first executor must use the
 * default CUDA stream (NULL) because the current native Qwen3.8 graph does. */
int axiom_qwen38_mtp_forward_request_validate(
        const axiom_qwen38_mtp *mtp,
        const axiom_qwen38_mtp_forward_request *request);

#ifdef __cplusplus
}
#endif

#endif
