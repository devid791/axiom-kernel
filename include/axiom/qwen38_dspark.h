#ifndef AXIOM_QWEN38_DSPARK_H
#define AXIOM_QWEN38_DSPARK_H

/*
 * Resident loader and execution contract for the BF16 DSpark draft checkpoint
 * Qwen3.8-27B-DSpark.
 *
 * This sidecar is a five-layer Qwen3-style DFlash decoder. It consumes five
 * auxiliary target hidden-state taps (layers 4, 16, 28, 40 and 52), fuses
 * them through fc + hidden_norm, injects per-layer K/V into its draft cache,
 * and proposes up to seven tokens. It shares both the embedding table and
 * LM head with the target Qwen3.8 model. Markov and confidence heads are
 * resident in this sidecar.
 *
 * The loader retains the original BF16 bytes. It does not execute a draft
 * graph, create a KV cache, or mutate target state; those are deliberately
 * made explicit by the request contracts below.
 */

#include <stdint.h>

#include "axiom/axiom.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN38_DSPARK_HIDDEN 5120u
#define AXIOM_QWEN38_DSPARK_INTERMEDIATE 10240u
#define AXIOM_QWEN38_DSPARK_LAYERS 5u
#define AXIOM_QWEN38_DSPARK_HEADS 40u
#define AXIOM_QWEN38_DSPARK_KV_HEADS 8u
#define AXIOM_QWEN38_DSPARK_HEAD_DIM 128u
#define AXIOM_QWEN38_DSPARK_VOCAB 248320u
#define AXIOM_QWEN38_DSPARK_TARGET_FEATURES 5u
#define AXIOM_QWEN38_DSPARK_TARGET_LAYERS 64u
#define AXIOM_QWEN38_DSPARK_FUSION_INPUT 25600u
#define AXIOM_QWEN38_DSPARK_BLOCK_SIZE 7u
#define AXIOM_QWEN38_DSPARK_VERIFY_WIDTH 8u
#define AXIOM_QWEN38_DSPARK_MASK_TOKEN_ID 248077u
#define AXIOM_QWEN38_DSPARK_MARKOV_RANK 256u
#define AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES 5376u
#define AXIOM_QWEN38_DSPARK_GLOBAL_TENSOR_COUNT 7u
#define AXIOM_QWEN38_DSPARK_LAYER_TENSOR_COUNT 11u
#define AXIOM_QWEN38_DSPARK_TENSOR_COUNT 62u
#define AXIOM_QWEN38_DSPARK_GLOBAL_LAYER UINT32_MAX

typedef struct axiom_qwen38_dspark axiom_qwen38_dspark;

/* Global tensor kinds require layer == AXIOM_QWEN38_DSPARK_GLOBAL_LAYER;
 * layer-local kinds require 0 <= layer < AXIOM_QWEN38_DSPARK_LAYERS. */
typedef enum {
    AXIOM_QWEN38_DSPARK_TENSOR_INVALID = 0,
    AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT = 1,
    AXIOM_QWEN38_DSPARK_TENSOR_HIDDEN_NORM_WEIGHT = 2,
    AXIOM_QWEN38_DSPARK_TENSOR_FINAL_NORM_WEIGHT = 3,
    AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT = 4,
    AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS = 5,
    AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W1_WEIGHT = 6,
    AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W2_WEIGHT = 7,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT = 8,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_DOWN_PROJ_WEIGHT = 9,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_GATE_PROJ_WEIGHT = 10,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_UP_PROJ_WEIGHT = 11,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_POST_ATTENTION_LAYERNORM_WEIGHT = 12,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_NORM_WEIGHT = 13,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_PROJ_WEIGHT = 14,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_O_PROJ_WEIGHT = 15,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_NORM_WEIGHT = 16,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_PROJ_WEIGHT = 17,
    AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_V_PROJ_WEIGHT = 18,
} axiom_qwen38_dspark_tensor;

typedef struct {
    uint32_t abi_version;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t draft_layers;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t target_model_layers;
    uint32_t target_feature_count;
    uint32_t target_layer_ids[AXIOM_QWEN38_DSPARK_TARGET_FEATURES];
    uint32_t block_size;
    uint32_t verify_width;
    uint32_t mask_token_id;
    uint32_t markov_rank;
    uint32_t confidence_features;
    uint32_t tensor_count;
    axiom_tensor_dtype resident_dtype;
    uint64_t checkpoint_bytes;
    uint64_t device_bytes;
} axiom_qwen38_dspark_info;

/* A typed, borrowed immutable view. `device_buffer` remains valid until the
 * parent DSpark object is destroyed. It contains raw BF16 bytes in checkpoint
 * row-major order; a CUDA executor may borrow the CUDA address via
 * axiom_device_buffer_cuda_pointer(). */
typedef struct {
    uint32_t abi_version;
    axiom_qwen38_dspark_tensor tensor;
    uint32_t layer;
    axiom_tensor_dtype dtype;
    uint32_t rank;
    uint64_t shape[AXIOM_MAX_TENSOR_DIMS];
    uint64_t byte_count;
    char name[128];
    const axiom_device_buffer *device_buffer;
} axiom_qwen38_dspark_tensor_view;

/* Overall contract for a native DSpark executor. All activation pointers in
 * the request types below are CUDA F32 column-major buffers. DSpark uses
 * conventional Qwen3 RMSNorm scale weights (no +1 zero-centering) and the
 * Qwen3.8 1M YaRN RoPE profile: theta=1e7, factor=4, native context=262144,
 * beta_fast=32, beta_slow=1. */
typedef struct {
    uint32_t abi_version;
    uint32_t scalar_dtype;
    uint32_t hidden_size;
    uint32_t target_feature_count;
    uint32_t target_layer_ids[AXIOM_QWEN38_DSPARK_TARGET_FEATURES];
    uint32_t draft_layers;
    uint32_t attention_heads;
    uint32_t key_value_heads;
    uint32_t head_dim;
    uint32_t block_size;
    uint32_t verify_width;
    uint32_t mask_token_id;
    uint32_t markov_rank;
    uint32_t confidence_features;
    uint32_t uses_target_embedding;
    uint32_t uses_target_lm_head;
    uint32_t requires_draft_kv_injection;
    uint32_t requires_draft_kv_transaction;
    uint32_t attention_is_noncausal_over_draft_block;
    uint32_t rmsnorm_zero_centered;
    uint32_t max_context;
    float rms_norm_eps;
    float confidence_head_alpha;
    float rope_theta;
    float yarn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_original_context;
} axiom_qwen38_dspark_forward_contract;

/* Fuse the five target auxiliary features in target_layer_ids order. Input is
 * [target_feature_count * hidden_size, batch]; output is [hidden_size, batch]
 * after fc.weight and hidden_norm.weight. This result seeds K/V injection for
 * all five draft layers. */
typedef struct {
    uint32_t abi_version;
    const float *target_aux_hidden;
    float *fused_hidden;
    uint32_t batch;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_fusion_request;

/* Seed the five draft layer KV caches from a fused target state. `positions`
 * contains one absolute position for every logical sequence. The executor
 * owns the cache and must make this operation transactional with the target
 * verification window. */
typedef struct {
    uint32_t abi_version;
    const float *fused_hidden;
    const uint32_t *positions;
    uint32_t batch;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_kv_injection_request;

/* Run a masked/noise draft block after K/V injection. `token_count` is in
 * [1, block_size]. token_ids and positions are column-major [token_count,
 * batch]. `out_hidden` is [hidden_size, token_count * batch] after norm.weight
 * and is projected by the target LM head. */
typedef struct {
    uint32_t abi_version;
    const uint32_t *token_ids;
    const uint32_t *positions;
    float *out_hidden;
    uint32_t batch;
    uint32_t token_count;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_draft_request;

/* Markov/confidence postprocess for a draft block. Base logits and draft
 * hidden states are [vocab_size, token_count * batch] and [hidden_size,
 * token_count * batch]; anchor_tokens has one committed token per sequence.
 * Outputs retain the same column-major token ordering. */
typedef struct {
    uint32_t abi_version;
    const float *base_logits;
    const float *draft_hidden;
    const uint32_t *anchor_tokens;
    float *out_logits;
    float *out_confidence;
    uint32_t batch;
    uint32_t token_count;
    void *stream;
    uint64_t flags;
} axiom_qwen38_dspark_markov_request;

/* `model` must have been opened on the DSpark directory. The loader accepts
 * exactly its one model.safetensors file: 62 original BF16 tensors with the
 * five-layer architecture encoded above. `runtime` must be CUDA on `device`. */
int axiom_qwen38_dspark_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        axiom_qwen38_dspark **out);

void axiom_qwen38_dspark_destroy(axiom_qwen38_dspark *dspark);
uint64_t axiom_qwen38_dspark_device_bytes(const axiom_qwen38_dspark *dspark);

int axiom_qwen38_dspark_info_get(
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_info *out);

int axiom_qwen38_dspark_tensor_view_get(
        const axiom_qwen38_dspark *dspark,
        uint32_t layer,
        axiom_qwen38_dspark_tensor tensor,
        axiom_qwen38_dspark_tensor_view *out);

int axiom_qwen38_dspark_forward_contract_get(
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_forward_contract *out);

/* Validation only: none of these functions execute a kernel or mutate a
 * cache. The first native executor must use the default CUDA stream (NULL),
 * matching the current Qwen3.8 integration path. */
int axiom_qwen38_dspark_fusion_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_fusion_request *request);
int axiom_qwen38_dspark_kv_injection_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_kv_injection_request *request);
int axiom_qwen38_dspark_draft_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_draft_request *request);
int axiom_qwen38_dspark_markov_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_markov_request *request);

#ifdef __cplusplus
}
#endif

#endif
