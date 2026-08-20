#ifndef AXIOM_AXIOM_H
#define AXIOM_AXIOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_ABI_VERSION 1u
#define AXIOM_MAX_TENSOR_DIMS 8u
#define AXIOM_Q8_K_BLOCK_BYTES 292u
#define AXIOM_MAX_GOAL_ID 64u
#define AXIOM_MAX_AGENT_ID 64u
#define AXIOM_MAX_AGENT_ROLE 64u
#define AXIOM_MAX_AGENT_OBJECTIVE 384u

typedef enum {
    AXIOM_OK = 0,
    AXIOM_ERR_INVALID_ARGUMENT = 1,
    AXIOM_ERR_UNSUPPORTED_BACKEND = 2,
    AXIOM_ERR_CUDA = 3,
    AXIOM_ERR_RUNTIME = 4,
    AXIOM_ERR_NOT_IMPLEMENTED = 5,
    AXIOM_ERR_IO = 6,
    AXIOM_ERR_BUDGET = 7,
} axiom_status;

typedef enum {
    AXIOM_BACKEND_CUDA = 1,
    AXIOM_BACKEND_CLUSTER = 2,
} axiom_backend;

typedef struct axiom_runtime axiom_runtime;
typedef struct axiom_model axiom_model;
typedef struct axiom_entity axiom_entity;
typedef struct axiom_session axiom_session;
typedef struct axiom_tokenizer axiom_tokenizer;
typedef struct axiom_latent_link axiom_latent_link;
typedef struct axiom_cluster axiom_cluster;
typedef struct axiom_device_buffer axiom_device_buffer;

#define AXIOM_TOKEN_ID_INVALID UINT32_MAX

typedef enum {
    AXIOM_MODEL_FORMAT_NATIVE = 1,
    AXIOM_MODEL_FORMAT_GGUF = 2,
    AXIOM_MODEL_FORMAT_SAFETENSORS = 3,
} axiom_model_format;

typedef enum {
    AXIOM_TOKENIZER_FORMAT_HF_JSON = 1,
    AXIOM_TOKENIZER_FORMAT_NATIVE = 2,
} axiom_tokenizer_format;

typedef enum {
    AXIOM_TENSOR_DTYPE_UNKNOWN = 0,
    AXIOM_TENSOR_DTYPE_F32 = 1,
    AXIOM_TENSOR_DTYPE_F16 = 2,
    AXIOM_TENSOR_DTYPE_BF16 = 3,
    AXIOM_TENSOR_DTYPE_I64 = 4,
    AXIOM_TENSOR_DTYPE_I32 = 5,
    AXIOM_TENSOR_DTYPE_U8 = 6,
    AXIOM_TENSOR_DTYPE_F8_E4M3 = 7,
    AXIOM_TENSOR_DTYPE_F8_E5M2 = 8,
    AXIOM_TENSOR_DTYPE_Q8_0 = 9,
    AXIOM_TENSOR_DTYPE_Q2_K = 10,
    AXIOM_TENSOR_DTYPE_IQ2_XXS = 11,
    AXIOM_TENSOR_DTYPE_Q4_K = 12,
} axiom_tensor_dtype;

typedef enum {
    AXIOM_SCHEDULER_EXCLUSIVE = 1,
    AXIOM_SCHEDULER_INTERLEAVE = 2,
    AXIOM_SCHEDULER_BATCH_COMPATIBLE = 3,
    AXIOM_SCHEDULER_SWARM = 4,
    AXIOM_SCHEDULER_LATENT_RECURSIVE = 5,
} axiom_scheduler_mode;

typedef enum {
    AXIOM_LATENT_BF16 = 1,
    AXIOM_LATENT_F16 = 2,
    AXIOM_LATENT_F32 = 3,
} axiom_latent_dtype;

typedef enum {
    AXIOM_LATENT_LINK_INNER = 1,
    AXIOM_LATENT_LINK_OUTER = 2,
} axiom_latent_link_kind;

typedef enum {
    AXIOM_PLACEMENT_LOCAL_DEVICE = 1,
    AXIOM_PLACEMENT_LOCAL_NODE = 2,
    AXIOM_PLACEMENT_CLUSTER_NODE = 3,
    AXIOM_PLACEMENT_CLUSTER_MESH = 4,
} axiom_placement_kind;

typedef enum {
    AXIOM_TRANSPORT_INPROC = 1,
    AXIOM_TRANSPORT_SHM = 2,
    AXIOM_TRANSPORT_TCP = 3,
    AXIOM_TRANSPORT_QUIC = 4,
    AXIOM_TRANSPORT_RDMA = 5,
} axiom_transport_kind;

typedef enum {
    AXIOM_AGENT_COORDINATOR = 1,
    AXIOM_AGENT_WORKER = 2,
    AXIOM_AGENT_MODEL_WORKER = 3,
} axiom_agent_kind;

typedef enum {
    AXIOM_AGENT_CREATED = 1,
    AXIOM_AGENT_DISPATCHED = 2,
    AXIOM_AGENT_RECEIVED = 3,
} axiom_agent_status;

typedef struct {
    uint32_t abi_version;
    axiom_backend backend;
    uint32_t device;
    uint64_t flags;
} axiom_config;

typedef struct {
    uint32_t abi_version;
    axiom_transport_kind transport;
    const char *listen_addr;
    const char *join_addr;
    uint32_t node_id;
    uint32_t node_count;
    uint64_t flags;
} axiom_cluster_config;

typedef struct {
    uint32_t abi_version;
    axiom_transport_kind transport;
    char listen_addr[128];
    char join_addr[128];
    uint32_t node_id;
    uint32_t node_count;
    uint32_t connected_peers;
    uint32_t protocol_version;
    uint64_t flags;
} axiom_cluster_info;

typedef struct {
    uint32_t abi_version;
    axiom_placement_kind kind;
    uint32_t node_id;
    uint32_t device_id;
    uint64_t memory_budget_bytes;
    uint64_t flags;
} axiom_placement;

typedef struct {
    uint32_t abi_version;
    const char *goal_id;
    const char *objective;
    uint64_t budget_tokens;
    uint64_t flags;
} axiom_goal_config;

typedef struct {
    uint32_t abi_version;
    axiom_agent_kind kind;
    const char *agent_id;
    const char *goal_id;
    const char *role;
    const char *objective;
    axiom_placement placement;
    uint32_t target_node;
    uint32_t target_device;
    uint64_t task_id;
    uint64_t session_id;
    uint64_t budget_tokens;
    uint64_t flags;
} axiom_agent_spawn_config;

typedef struct {
    uint32_t abi_version;
    axiom_agent_kind kind;
    axiom_agent_status status;
    char agent_id[AXIOM_MAX_AGENT_ID];
    char goal_id[AXIOM_MAX_GOAL_ID];
    char role[AXIOM_MAX_AGENT_ROLE];
    char objective[AXIOM_MAX_AGENT_OBJECTIVE];
    axiom_placement_kind placement;
    uint32_t source_node;
    uint32_t source_device;
    uint32_t target_node;
    uint32_t target_device;
    uint64_t task_id;
    uint64_t session_id;
    uint64_t budget_tokens;
    uint64_t sequence;
    uint64_t flags;
} axiom_agent_spawn_info;

typedef struct {
    uint32_t abi_version;
    char name[128];
    char path[512];
    char model_type[64];
    char architecture[128];
    axiom_model_format format;
    uint64_t bytes;
    uint64_t safetensors_bytes;
    uint64_t weight_index_total_bytes;
    uint64_t safetensors_header_data_bytes;
    uint64_t memory_budget_bytes;
    uint32_t safetensors_file_count;
    uint32_t weight_tensor_count;
    uint32_t safetensors_header_tensor_count;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t num_hidden_layers;
    uint32_t num_attention_heads;
    uint32_t num_key_value_heads;
    uint32_t vocab_size;
    uint32_t max_context;
    uint32_t entity_count;
} axiom_model_info;

typedef struct {
    uint32_t abi_version;
    char name[256];
    char file[256];
    axiom_tensor_dtype dtype;
    uint32_t rank;
    uint64_t shape[AXIOM_MAX_TENSOR_DIMS];
    uint64_t data_offset_begin;
    uint64_t data_offset_end;
    uint64_t file_offset_begin;
    uint64_t file_offset_end;
    uint64_t byte_count;
} axiom_tensor_info;

typedef struct {
    uint32_t abi_version;
    char name[128];
    char path[512];
    axiom_tokenizer_format format;
    uint64_t tokenizer_json_bytes;
    uint64_t tokenizer_hash;
    uint64_t chat_template_hash;
    uint32_t vocab_size;
    uint32_t added_tokens;
    uint32_t endoftext_token_id;
    uint32_t im_start_token_id;
    uint32_t im_end_token_id;
    uint32_t tool_call_token_id;
    uint32_t tool_call_end_token_id;
} axiom_tokenizer_info;

typedef struct {
    uint32_t abi_version;
    char name[128];
    char role[128];
    uint64_t memory_budget_bytes;
    uint32_t session_count;
} axiom_entity_info;

typedef struct {
    char name[128];
    int major;
    int minor;
    uint64_t total_global_mem;
    int multi_processor_count;
} axiom_device_info;

typedef struct {
    uint32_t abi_version;
    const char *path;
    const char *name;
    axiom_model_format format;
    axiom_placement placement;
    uint64_t memory_budget_bytes;
    uint32_t max_context;
    uint64_t flags;
} axiom_model_config;

typedef struct {
    uint32_t abi_version;
    const char *path;
    const char *name;
    axiom_tokenizer_format format;
    uint64_t flags;
} axiom_tokenizer_config;

typedef struct {
    uint32_t abi_version;
    const char *name;
    const char *role;
    uint64_t memory_budget_bytes;
    uint64_t flags;
} axiom_entity_config;

typedef struct {
    uint32_t abi_version;
    uint32_t max_context;
    uint64_t kv_budget_bytes;
    uint64_t flags;
} axiom_session_config;

typedef struct {
    uint32_t abi_version;
    axiom_scheduler_mode mode;
    uint32_t max_steps;
    uint64_t flags;
} axiom_scheduler_step_options;

typedef struct {
    uint32_t abi_version;
    axiom_latent_dtype dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
    void *device_ptr;
} axiom_latent_frame;

typedef struct {
    uint32_t abi_version;
    axiom_latent_dtype dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
    uint32_t source_node;
    uint32_t source_device;
    uint64_t shard_id;
    uint64_t session_id;
    uint64_t content_hash;
    uint64_t payload_bytes;
    uint64_t flags;
} axiom_latent_shard;

typedef struct {
    uint32_t abi_version;
    axiom_latent_link_kind kind;
    axiom_latent_dtype dtype;
    uint32_t source_width;
    uint32_t target_width;
    uint32_t hidden_width;
    uint32_t rank;
    float eps;
    uint64_t flags;
} axiom_latent_link_config;

typedef struct {
    uint32_t abi_version;
    axiom_latent_dtype dtype;
    const float *pre_ln_weight;
    const float *pre_ln_bias;
    const float *proj1_weight;
    const float *proj1_bias;
    const float *proj2_weight;
    const float *proj2_bias;
    const float *residual_weight;
    const float *residual_bias;
    const float *post_ln_weight;
    const float *post_ln_bias;
    uint64_t flags;
} axiom_latent_link_weights_f32;

typedef struct {
    uint32_t abi_version;
    uint32_t header_abi_version;
    uint32_t runtime_abi_version;
    uint32_t struct_size;
    const char *version;
    const char *backend;
    const char *build_target;
    uint64_t flags;
} axiom_abi_info;

uint32_t axiom_abi_version(void);
const char *axiom_version(void);
int axiom_abi_info_get(axiom_abi_info *out);
int axiom_abi_check(uint32_t header_abi_version);
const char *axiom_status_string(int status);

int axiom_runtime_create(axiom_runtime **out, const axiom_config *config);
void axiom_runtime_destroy(axiom_runtime *runtime);

int axiom_runtime_device_count(uint32_t *out_count);
int axiom_runtime_probe_device(uint32_t device, axiom_device_info *out);
int axiom_runtime_device_id(const axiom_runtime *runtime, uint32_t *out_device);
int axiom_runtime_probe(axiom_runtime *runtime, axiom_device_info *out);
int axiom_runtime_q8_0_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_q8_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q2_k_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_q2k_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_iq2_xxs_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_iq2xxs_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
int axiom_device_buffer_create(
        axiom_runtime *runtime,
        axiom_device_buffer **out,
        uint64_t bytes);
void axiom_device_buffer_destroy(axiom_device_buffer *buffer);
int axiom_device_buffer_upload(
        axiom_device_buffer *buffer,
        uint64_t offset,
        const void *src_host,
        uint64_t bytes);
int axiom_device_buffer_download(
        axiom_device_buffer *buffer,
        uint64_t offset,
        void *dst_host,
        uint64_t bytes);
int axiom_device_buffer_copy(
        axiom_device_buffer *dst,
        uint64_t dst_offset,
        const axiom_device_buffer *src,
        uint64_t src_offset,
        uint64_t bytes);
int axiom_device_buffer_device_id(const axiom_device_buffer *buffer, uint32_t *out_device);
/* Borrow the native CUDA allocation for composition with an Axiom-owned CUDA
 * primitive.  CUDA backend only.  The pointer remains owned by `buffer`: it
 * becomes invalid when the buffer or its runtime is destroyed, and callers
 * must never pass it to cudaFree. */
int axiom_device_buffer_cuda_pointer(const axiom_device_buffer *buffer, void **out_pointer);
int axiom_runtime_rmsnorm_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
/* High-precision rmsnorm (double sum-of-squares + (float)(1.0/sqrt(...)) inverse) —
 * matches the host reference rmsnorm math so the on-device-glue forward tracks the
 * llama.cpp oracle as closely as the host path. Same ABI as the plain entry. */
int axiom_runtime_rmsnorm_f32_hp_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
/* Qwen B1 decode fast path rmsnorm: FP32 reduction + rsqrtf, oracle-gated for
 * B1 throughput work only. Non-B1/shared callers should keep using the hp entry. */
int axiom_runtime_rmsnorm_f32_b1_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
int axiom_runtime_rmsnorm_f32_dual_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t count,
        float eps);
int axiom_runtime_rmsnorm_f32_q8k_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        axiom_device_buffer *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count,
        float eps);
int axiom_runtime_head_rmsnorm_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
/* P2: dense NeoX RoPE (half-split pairing), faithful to the host rope_neox. */
int axiom_runtime_rope_neox_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t pos,
        float theta);
/* P2 STEP 7: PARTIAL NeoX RoPE — rotate only the first n_rot dims (pairs ic, ic+n_rot/2),
 * pass-through [n_rot, head_dim). half=n_rot/2 and freq denominator=n_rot. Faithful to the host
 * rope_neox_partial (Nemotron: n_rot=64 of head_dim=128). NO qk-norm fused (Nemotron has none). */
int axiom_runtime_rope_neox_partial_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos,
        float theta);
/* P2 launch-fusion: fused per-head QK-RMSNorm(+weight) + NeoX rope (one launch per tensor). */
int axiom_runtime_qknorm_rope_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x, uint64_t x_offset,
        const axiom_device_buffer *w, uint64_t w_offset,
        uint32_t heads, uint32_t head_dim, uint32_t pos, float theta, float eps);
int axiom_runtime_qknorm_rope_pos_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x, uint64_t x_offset,
        const axiom_device_buffer *w, uint64_t w_offset,
        const axiom_device_buffer *pos, uint64_t pos_offset,
        uint32_t heads, uint32_t head_dim, float theta, float eps);
int axiom_runtime_qknorm_rope_pos_dual_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *qw, uint64_t qw_offset,
        axiom_device_buffer *k, uint64_t k_offset,
        const axiom_device_buffer *kw, uint64_t kw_offset,
        const axiom_device_buffer *pos, uint64_t pos_offset,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim, float theta, float eps);
/* P2: per-head QK-RMSNorm with learned weight, faithful to the host qk_norm. */
int axiom_runtime_qk_rmsnorm_w_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        const axiom_device_buffer *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
/* P2: device-resident GQA causal attention core over the on-device KV cache (the keystone
 * round-trip eliminator). cache_tokens = pos+1; cache layout token-major (t*kv_heads+kv_head)*hd. */
int axiom_runtime_attention_core_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens);
/* P2 STEP 6: optimized block-per-head GQA attention (score computed once + online softmax).
 * SLIDING WINDOW: window>0 restricts the softmax to the last `window` keys
 * [cache_tokens-window, cache_tokens) (gemma sliding layers); window==0 => full causal
 * (global/full layers + non-sliding families like qwen/nemotron) — byte-identical to pre-window. */
int axiom_runtime_attention_core_fast_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens,
        uint32_t window);
/* P2 STEP 7 (Gemma-4): DECOUPLED partial NeoX RoPE — rotate the first `active_pairs` pairs
 * (ic, ic+head_dim/2) with freq = theta^(-2*ic/exp_dim); pass the rest through. active_pairs (the
 * rotated range) and exp_dim (the freq denominator) are SEPARATE (gemma-4 global: rotate 64 pairs
 * of head_dim 512, denominator 512). Faithful to the host rope() in tools/axiom_gemma4.cpp. */
int axiom_runtime_rope_neox_decoupled_partial_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t active_pairs,
        uint32_t exp_dim,
        uint32_t pos,
        float theta);
/* P2 STEP 7 (Gemma-4): per-head RMSNorm out = row[i]*inv*w[i] (PLAIN weight, double sum-of-squares
 * + (float)(1.0/sqrt(ss/hd+eps)) inverse), faithful to the gemma-4 host rmsnorm1p/qknorm1p (NOTE:
 * gemma-4 uses PLAIN *w, NOT *(1+w)). w may be NULL -> v-norm no-scale path (out = row[i]*inv). */
int axiom_runtime_gemma4_qknorm1p_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        const axiom_device_buffer *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
/* P2 STEP 7 (Gemma-4): block-per-head online-softmax GQA/MQA attention with a CALLER-SUPPLIED
 * scale (gemma-4 uses scale=1.0 — raw dot product, NO 1/sqrt(hd)). Handles MQA (kv_heads=1) and
 * GQA (kv_heads=8). Cache layout token-major (t*kv_heads+kvh)*head_dim, cache_tokens = pos+1. */
int axiom_runtime_attention_core_scale_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens, float scale,
        uint32_t window);
/* W5a (continuous batching): PAGED attention cores — the SAME online-softmax math as the two
 * contiguous cores above (BYTE-IDENTICAL output given identical logical KV: paging is pure
 * storage relocation, K/V bytes are read in the same strictly ascending-t order), but K/V live
 * in device-resident paged pools addressed through per-active-seq block tables:
 *   lb = t/block_tokens, slot = t%block_tokens, pb = block_tables[a*tbl_stride + lb]
 *   element base = (((uint64_t)pb*n_layers + layer)*block_tokens + slot)*pool_stride
 *                  + kvh*head_dim
 * Layouts: q/out are per-active-seq token-major [active][q_heads*head_dim] (active=1 for
 * prefill); block_tables is a DEVICE int32 array [active][tbl_stride]; cache_tokens a DEVICE
 * uint32 array [active] of per-seq lengths (ragged decode in ONE launch; the partial last
 * block falls out of the t<cache_tokens[a] bound). pool_stride is the ELEMENT stride of one
 * (block,layer,slot) row and is SEPARATE from kv_heads*head_dim (gemma-4 later packs its true
 * KVD into a wider uniform slab); must be >= kv_heads*head_dim. Host metadata (free-list,
 * block tables, refcounts) stays host — the engine uploads tables/lengths before the launch.
 * window: same semantics as the contiguous cores, computed PER SEQ from cache_tokens[a].
 * head_dim must be a power of two and <= 1024 (one block per (head, seq), block = head_dim). */
int axiom_runtime_attention_core_paged_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_pool, uint64_t k_offset,
        const axiom_device_buffer *v_pool, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *block_tables, uint64_t tbl_offset,
        const axiom_device_buffer *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t window);
/* Qwen B=1 decode vector paged attention: split the KV sequence into fixed graph-capturable
 * chunks and combine online-softmax partials. This deliberately changes reduction order and is
 * therefore oracle-anchored, not bit-identical. Unsupported shapes return
 * AXIOM_ERR_UNSUPPORTED_BACKEND so callers can use the scalar paged core. */
int axiom_runtime_attention_core_paged_b1_vec_reserve(
        axiom_runtime *runtime,
        uint32_t active, uint32_t tbl_stride, uint32_t block_tokens,
        uint32_t q_heads, uint32_t head_dim, uint32_t window);
int axiom_runtime_attention_core_paged_b1_vec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_pool, uint64_t k_offset,
        const axiom_device_buffer *v_pool, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *block_tables, uint64_t tbl_offset,
        const axiom_device_buffer *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t window);
/* W5a: paged attention core with a CALLER-SUPPLIED scale (gemma-4 uses 1.0) — paged variant of
 * axiom_runtime_attention_core_scale_f32_device; same addressing/layouts as the paged core. */
int axiom_runtime_attention_core_scale_paged_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_pool, uint64_t k_offset,
        const axiom_device_buffer *v_pool, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *block_tables, uint64_t tbl_offset,
        const axiom_device_buffer *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale, uint32_t window);
/* W5a: scatter ONE staged K row and ONE staged V row per active seq (k_stage/v_stage are
 * [active][kv_dim] f32 each — the current token's k/v for `layer`) into the paged pools at
 * per-seq token position pos[a] (pos is a DEVICE uint32 array [active]; positions may differ
 * per seq), via the same block-table addressing as the paged attention cores. kv_dim may
 * exceed 1024 (grid-stride row loop, up to 4096+). Lanes [kv_dim, pool_stride) of the target
 * row are left untouched. */
int axiom_runtime_kv_pool_store_paged_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *k_stage, uint64_t k_stage_offset,
        const axiom_device_buffer *v_stage, uint64_t v_stage_offset,
        axiom_device_buffer *k_pool, uint64_t k_pool_offset,
        axiom_device_buffer *v_pool, uint64_t v_pool_offset,
        const axiom_device_buffer *block_tables, uint64_t tbl_offset,
        const axiom_device_buffer *pos, uint64_t pos_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t kv_dim);
/* Gated DeltaNet (Qwen3.6/Qwen3-Next) device kernels */
int axiom_runtime_deltanet_conv1d_silu_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *in, uint64_t in_off,
        const axiom_device_buffer *w, uint64_t w_off, axiom_device_buffer *ring, uint64_t ring_off,
        axiom_device_buffer *out, uint64_t out_off, uint32_t conv_dim);
int axiom_runtime_deltanet_recurrence_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *qkv, uint64_t q_off, uint64_t k_off, uint64_t v_off,
        const axiom_device_buffer *g, uint64_t g_off, const axiom_device_buffer *beta, uint64_t beta_off,
        axiom_device_buffer *state, uint64_t state_off, axiom_device_buffer *out, uint64_t out_off,
        uint32_t n_v_heads, uint32_t n_k_heads, uint32_t head_dim);
int axiom_runtime_deltanet_gated_norm_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *o, uint64_t o_off,
        const axiom_device_buffer *wnorm, uint64_t wnorm_off, const axiom_device_buffer *z, uint64_t z_off,
        axiom_device_buffer *out, uint64_t out_off, uint32_t n_v_heads, uint32_t head_dim, float eps);
int axiom_runtime_head_rmsnorm_f32_dual_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
int axiom_runtime_deepseek_fp8_kv_quantize_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot);
int axiom_runtime_deepseek_fp8_kv_quantize_dual_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot);
int axiom_runtime_f32_f16_round_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t count);
int axiom_runtime_f32_f16_round_dual_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t count);
int axiom_runtime_add_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *a,
        uint64_t a_offset,
        const axiom_device_buffer *b,
        uint64_t b_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count);
int axiom_runtime_silu_mul_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count);
/* GeGLU (gelu-tanh gate) * up — out = 0.5*g*(1+tanh(0.7978845608*(g+0.044715*g^3)))*up,
 * computed in double (precise tanh, not --use_fast_math degraded). Matches the host
 * geglu math in the gemma3/gemma4 runners; shared by both on-device FFN paths. */
int axiom_runtime_geglu_mul_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count);
/* LayerNorm WITH bias (Nemotron) — out = ((in-mean)*inv)*w + b, mean/var in double,
 * inv=(float)(1.0/sqrt(var+eps)); matches the host layernorm math. */
int axiom_runtime_layernorm_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32, uint64_t weight_offset,
        const axiom_device_buffer *bias_f32, uint64_t bias_offset,
        const axiom_device_buffer *input, uint64_t input_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t count, float eps);
/* ReLU^2 (squared ReLU), Nemotron sequential FFN: out = (max(in,0))^2. */
int axiom_runtime_relu2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input, uint64_t input_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t count);
int axiom_runtime_silu_mul_clamp_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float clamp_abs);
int axiom_runtime_pack_f32_to_q8k_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count);
int axiom_runtime_weighted_sum_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *inputs,
        uint64_t inputs_offset,
        const axiom_device_buffer *weights,
        uint64_t weights_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t slots,
        uint32_t count);
int axiom_runtime_f16_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f16,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_f16_dual_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_a_f16,
        uint64_t weight_a_offset,
        const axiom_device_buffer *weight_b_f16,
        uint64_t weight_b_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_a,
        uint64_t out_a_offset,
        axiom_device_buffer *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
int axiom_runtime_f16_embedding_streams_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_f16,
        uint64_t embedding_offset,
        axiom_device_buffer *out_streams,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t hidden,
        uint32_t streams);
int axiom_runtime_q8_0_embedding_gather_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_q8,
        uint64_t embedding_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t token_count,
        uint32_t hidden);
int axiom_runtime_q8_0_embedding_gather_token_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_q8,
        uint64_t embedding_offset,
        const axiom_device_buffer *token_id,
        uint64_t token_id_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t token_count,
        uint32_t hidden);
int axiom_runtime_q8_0_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
/* Native DeepSeek-V4-Flash weight matvecs — single-output GEMV out[rows]=dequant(W)·input[cols].
 * Offsets are in bytes. NVFP4 (routed experts): weight U8 [rows,cols/2] (2x signed-E2M1/byte,
 * low nibble=even col) + block_scale U8 e4m3 [rows,cols/16] (group 16) + global_scale f32;
 * requires cols%16==0. FP8 (attn/shared): weight U8 e4m3 [rows,cols] + block_scale U8 e8m0
 * [rows/128,cols/128]; requires rows%128==0 && cols%128==0. Bit-exact vs host oracle on GB10. */
int axiom_runtime_e2m1_nvfp4_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight,
        uint64_t weight_offset,
        const axiom_device_buffer *block_scale,
        uint64_t block_scale_offset,
        float global_scale,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_fp8_e4m3_e8m0_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight,
        uint64_t weight_offset,
        const axiom_device_buffer *block_scale,
        uint64_t block_scale_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
/* Native FP8 shared-expert FFN (DeepSeek-V4-Flash): out = down_fp8 · silu_mul(gate_fp8·x, up_fp8·x).
 * Composes three FP8 (E4M3+E8M0 128x128) matvecs + silu_mul. gate/up: rows=expert_hidden,
 * cols=hidden; down: rows=hidden, cols=expert_hidden (all 128-divisible). Two [expert_hidden]
 * f32 scratch buffers (gate, up). NVFP4/FP8 wiring 3/3 building block. */
int axiom_runtime_deepseek_shared_expert_fp8_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_w, uint64_t gate_w_offset,
        const axiom_device_buffer *gate_scale, uint64_t gate_scale_offset,
        const axiom_device_buffer *up_w, uint64_t up_w_offset,
        const axiom_device_buffer *up_scale, uint64_t up_scale_offset,
        const axiom_device_buffer *down_w, uint64_t down_w_offset,
        const axiom_device_buffer *down_scale, uint64_t down_scale_offset,
        const axiom_device_buffer *input, uint64_t input_offset,
        axiom_device_buffer *scratch_gate, uint64_t scratch_gate_offset,
        axiom_device_buffer *scratch_up, uint64_t scratch_up_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t hidden, uint32_t expert_hidden);
/* out[i] = beta*out[i] + alpha*in[i] (n f32). Accumulates weighted routed-expert outputs. */
int axiom_runtime_axpby_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *in, uint64_t in_offset,
        float alpha, float beta, uint32_t n);
/* Native NVFP4 routed-expert MoE (single token): out = sum_j w[j] * down_e . silu_mul(gate_e.x, up_e.x)
 * over the topk experts from router_indices/router_weights (downloaded to host). gate/up/down are the
 * de-blocked combined NVFP4 planes (weight _w, block-scale _bs) + per-expert f32 global _gs; expert e
 * at e*stride. Three [expert_hidden]/[hidden] scratch buffers. Correctness-first host loop. */
int axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_w, const axiom_device_buffer *gate_bs, const axiom_device_buffer *gate_gs,
        const axiom_device_buffer *up_w, const axiom_device_buffer *up_bs, const axiom_device_buffer *up_gs,
        const axiom_device_buffer *down_w, const axiom_device_buffer *down_bs, const axiom_device_buffer *down_gs,
        const axiom_device_buffer *input, uint64_t input_offset,
        const axiom_device_buffer *router_indices, uint64_t router_indices_offset,
        const axiom_device_buffer *router_weights, uint64_t router_weights_offset,
        axiom_device_buffer *scratch_gate,
        axiom_device_buffer *scratch_up,
        axiom_device_buffer *scratch_expert_out,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden);
int axiom_runtime_q8_0_matvec2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_matvec4_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        const axiom_device_buffer *input2,
        uint64_t input2_offset,
        const axiom_device_buffer *input3,
        uint64_t input3_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        axiom_device_buffer *out2,
        uint64_t out2_offset,
        axiom_device_buffer *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols);
/* Dynamic-B batched Q8_0 matvec: Y = W * X for B input vectors sharing the same
 * Q8_0 weight matrix W (rows x cols). Foundational primitive for continuous
 * batching. The dispatch mirrors axiom_runtime_q8_0_matvec_f32_device: by
 * default it takes the prequant warp path, so each output element Y[b][r] is
 * bit-for-bit identical to running axiom_runtime_q8_0_matvec_f32_device on X[b]
 * with the same (default) environment. Setting AXIOM_DS4_Q8_PREQ_DISABLE falls
 * back to the plain-float warp path on both, also bit-for-bit identical.
 *   x:        device buffer, row-major [batch][cols] f32.
 *   y:        device buffer, f32; layout selected by y_layout.
 *   y_layout: 0 -> [batch][rows] (per-seq contiguous, recommended);
 *             1 -> [rows][batch] (per-row contiguous / interleaved).
 */
int axiom_runtime_q8_0_batched_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *x,
        uint64_t x_offset,
        axiom_device_buffer *y,
        uint64_t y_offset,
        uint32_t y_layout,
        uint32_t batch,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_soa_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_soa_matvec2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_soa_matvec4_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        const axiom_device_buffer *input2,
        uint64_t input2_offset,
        const axiom_device_buffer *input3,
        uint64_t input3_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        axiom_device_buffer *out2,
        uint64_t out2_offset,
        axiom_device_buffer *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_dual_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_a_q8,
        uint64_t weight_a_offset,
        const axiom_device_buffer *weight_b_q8,
        uint64_t weight_b_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_a,
        uint64_t out_a_offset,
        axiom_device_buffer *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
int axiom_runtime_q8_0_qkv_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q_q8,
        uint64_t weight_q_offset,
        const axiom_device_buffer *weight_k_q8,
        uint64_t weight_k_offset,
        const axiom_device_buffer *weight_v_q8,
        uint64_t weight_v_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_q,
        uint64_t out_q_offset,
        axiom_device_buffer *out_k,
        uint64_t out_k_offset,
        axiom_device_buffer *out_v,
        uint64_t out_v_offset,
        uint32_t rows_q,
        uint32_t rows_k,
        uint32_t rows_v,
        uint32_t cols);
int axiom_runtime_q8_0_matvec_add_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_dual_matvec_silu_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_gate_q8,
        uint64_t weight_gate_offset,
        const axiom_device_buffer *weight_up_q8,
        uint64_t weight_up_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q8_0_dual_matvec2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_a_q8,
        uint64_t weight_a_offset,
        const axiom_device_buffer *weight_b_q8,
        uint64_t weight_b_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out_a0,
        uint64_t out_a0_offset,
        axiom_device_buffer *out_b0,
        uint64_t out_b0_offset,
        axiom_device_buffer *out_a1,
        uint64_t out_a1_offset,
        axiom_device_buffer *out_b1,
        uint64_t out_b1_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
int axiom_runtime_q8_0_matvec_argmax_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index,
        float *out_value);
int axiom_runtime_q8_0_matvec2_argmax_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index0,
        float *out_value0,
        uint32_t *out_index1,
        float *out_value1);
int axiom_runtime_f32_argmax_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t count,
        uint32_t *out_index,
        float *out_value);
int axiom_runtime_f32_argmax_to_buffer_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t count,
        axiom_device_buffer *out_token_id,
        uint64_t out_token_id_offset);
int axiom_runtime_q8_0_grouped_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
int axiom_runtime_q8_0_soa_grouped_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
int axiom_runtime_q8_0_soa_grouped_matvec2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
int axiom_runtime_q8_0_soa_grouped_matvec4_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        const axiom_device_buffer *input2,
        uint64_t input2_offset,
        const axiom_device_buffer *input3,
        uint64_t input3_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        axiom_device_buffer *out2,
        uint64_t out2_offset,
        axiom_device_buffer *out3,
        uint64_t out3_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
int axiom_runtime_q2_k_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q2k,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_q4_k_matvec_q8k_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q4k,
        uint64_t weight_offset,
        const axiom_device_buffer *input_q8k,
        uint64_t input_q8k_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_iq2_xxs_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_iq2xxs,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_iq2_xxs_matvec_f32_warp_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_iq2xxs,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
int axiom_runtime_deepseek_moe_topk_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_indexed_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_indexed_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *scratch_mid,
        uint64_t scratch_mid_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_gate_up_indexed_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        axiom_device_buffer *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_gate_up_indexed_q8k_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        axiom_device_buffer *scratch_q8k,
        uint64_t scratch_q8k_offset,
        axiom_device_buffer *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_down_indexed_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *mid,
        uint64_t mid_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_q8k_full_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *scratch_xq,
        uint64_t scratch_xq_offset,
        axiom_device_buffer *scratch_midq,
        uint64_t scratch_midq_offset,
        axiom_device_buffer *scratch_mid,
        uint64_t scratch_mid_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_q8k_full2_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *indices0,
        uint64_t indices0_offset,
        const axiom_device_buffer *router_weights0,
        uint64_t router_weights0_offset,
        axiom_device_buffer *scratch_xq0,
        uint64_t scratch_xq0_offset,
        axiom_device_buffer *scratch_midq0,
        uint64_t scratch_midq0_offset,
        axiom_device_buffer *scratch_mid0,
        uint64_t scratch_mid0_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        const axiom_device_buffer *indices1,
        uint64_t indices1_offset,
        const axiom_device_buffer *router_weights1,
        uint64_t router_weights1_offset,
        axiom_device_buffer *scratch_xq1,
        uint64_t scratch_xq1_offset,
        axiom_device_buffer *scratch_midq1,
        uint64_t scratch_midq1_offset,
        axiom_device_buffer *scratch_mid1,
        uint64_t scratch_mid1_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_q8k_full4_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_iq2xxs,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_iq2xxs,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q2k,
        uint64_t down_offset,
        const axiom_device_buffer *input[4],
        const uint64_t input_offset[4],
        const axiom_device_buffer *indices[4],
        const uint64_t indices_offset[4],
        const axiom_device_buffer *router_weights[4],
        const uint64_t router_weights_offset[4],
        axiom_device_buffer *scratch_xq[4],
        const uint64_t scratch_xq_offset[4],
        axiom_device_buffer *scratch_midq[4],
        const uint64_t scratch_midq_offset[4],
        axiom_device_buffer *scratch_mid[4],
        const uint64_t scratch_mid_offset[4],
        axiom_device_buffer *out[4],
        const uint64_t out_offset[4],
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_moe_q4k_full_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_all_q4k,
        uint64_t gate_offset,
        const axiom_device_buffer *up_all_q4k,
        uint64_t up_offset,
        const axiom_device_buffer *down_all_q4k,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        const axiom_device_buffer *router_weights,
        uint64_t router_weights_offset,
        axiom_device_buffer *scratch_xq,
        uint64_t scratch_xq_offset,
        axiom_device_buffer *scratch_midq,
        uint64_t scratch_midq_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_router_topk_f16_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *router_f16,
        uint64_t router_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_indices,
        uint64_t out_indices_offset,
        axiom_device_buffer *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
int axiom_runtime_deepseek_router_topk_biased_f16_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *router_f16,
        uint64_t router_offset,
        const axiom_device_buffer *bias_f32,
        uint64_t bias_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_indices,
        uint64_t out_indices_offset,
        axiom_device_buffer *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
int axiom_runtime_deepseek_router_topk_biased_f32_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *router_f32,
        uint64_t router_offset,
        const axiom_device_buffer *bias_f32,
        uint64_t bias_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_indices,
        uint64_t out_indices_offset,
        axiom_device_buffer *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
int axiom_runtime_deepseek_router_topk_biased_f16_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *router_f16,
        uint64_t router_offset,
        const axiom_device_buffer *bias_f32,
        uint64_t bias_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *scratch_logits,
        uint64_t scratch_logits_offset,
        axiom_device_buffer *out_indices,
        uint64_t out_indices_offset,
        axiom_device_buffer *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
int axiom_runtime_deepseek_router_hash_f16_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *router_f16,
        uint64_t router_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *indices,
        uint64_t indices_offset,
        axiom_device_buffer *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
int axiom_runtime_deepseek_shared_expert_q8_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_q8,
        uint64_t gate_offset,
        const axiom_device_buffer *up_q8,
        uint64_t up_offset,
        const axiom_device_buffer *down_q8,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_shared_expert_q8_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_q8,
        uint64_t gate_offset,
        const axiom_device_buffer *up_q8,
        uint64_t up_offset,
        const axiom_device_buffer *down_q8,
        uint64_t down_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *scratch_mid,
        uint64_t scratch_mid_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_shared_expert_q8_soa_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_qs_i8,
        const axiom_device_buffer *gate_scales_f16,
        const axiom_device_buffer *up_qs_i8,
        const axiom_device_buffer *up_scales_f16,
        const axiom_device_buffer *down_qs_i8,
        const axiom_device_buffer *down_scales_f16,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *scratch_mid,
        uint64_t scratch_mid_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_shared_expert_q8_soa_f32_scratch2_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_qs_i8,
        const axiom_device_buffer *gate_scales_f16,
        const axiom_device_buffer *up_qs_i8,
        const axiom_device_buffer *up_scales_f16,
        const axiom_device_buffer *down_qs_i8,
        const axiom_device_buffer *down_scales_f16,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *scratch_mid0,
        uint64_t scratch_mid0_offset,
        axiom_device_buffer *scratch_mid1,
        uint64_t scratch_mid1_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_shared_gate_up_q8_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_q8,
        uint64_t gate_offset,
        const axiom_device_buffer *up_q8,
        uint64_t up_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *mid,
        uint64_t mid_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_sliding_attention_single_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
int axiom_runtime_deepseek_sliding_attention_ring_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *current_kv,
        uint64_t current_kv_offset,
        const axiom_device_buffer *ring_kv,
        uint64_t ring_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
int axiom_runtime_deepseek_sliding_attention_ring2_causal_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q0,
        uint64_t q0_offset,
        const axiom_device_buffer *q1,
        uint64_t q1_offset,
        const axiom_device_buffer *kv0,
        uint64_t kv0_offset,
        const axiom_device_buffer *kv1,
        uint64_t kv1_offset,
        const axiom_device_buffer *ring_kv,
        uint64_t ring_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
int axiom_runtime_deepseek_sliding_attention_ring4_causal_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q0,
        uint64_t q0_offset,
        const axiom_device_buffer *q1,
        uint64_t q1_offset,
        const axiom_device_buffer *q2,
        uint64_t q2_offset,
        const axiom_device_buffer *q3,
        uint64_t q3_offset,
        const axiom_device_buffer *kv0,
        uint64_t kv0_offset,
        const axiom_device_buffer *kv1,
        uint64_t kv1_offset,
        const axiom_device_buffer *kv2,
        uint64_t kv2_offset,
        const axiom_device_buffer *kv3,
        uint64_t kv3_offset,
        const axiom_device_buffer *ring_kv,
        uint64_t ring_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        axiom_device_buffer *out2,
        uint64_t out2_offset,
        axiom_device_buffer *out3,
        uint64_t out3_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
int axiom_runtime_deepseek_attention_raw_comp_ring2_causal_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q0,
        uint64_t q0_offset,
        const axiom_device_buffer *q1,
        uint64_t q1_offset,
        const axiom_device_buffer *kv0,
        uint64_t kv0_offset,
        const axiom_device_buffer *kv1,
        uint64_t kv1_offset,
        const axiom_device_buffer *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const axiom_device_buffer *comp_kv,
        uint64_t comp_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        float eps);
int axiom_runtime_deepseek_attention_raw_comp_ring4_causal_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q0,
        uint64_t q0_offset,
        const axiom_device_buffer *q1,
        uint64_t q1_offset,
        const axiom_device_buffer *q2,
        uint64_t q2_offset,
        const axiom_device_buffer *q3,
        uint64_t q3_offset,
        const axiom_device_buffer *kv0,
        uint64_t kv0_offset,
        const axiom_device_buffer *kv1,
        uint64_t kv1_offset,
        const axiom_device_buffer *kv2,
        uint64_t kv2_offset,
        const axiom_device_buffer *kv3,
        uint64_t kv3_offset,
        const axiom_device_buffer *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const axiom_device_buffer *comp_kv,
        uint64_t comp_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        axiom_device_buffer *out2,
        uint64_t out2_offset,
        axiom_device_buffer *out3,
        uint64_t out3_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        uint32_t comp_count2,
        uint32_t comp_count3,
        float eps);
int axiom_runtime_deepseek_rope_tail_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t position,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse);
int axiom_runtime_deepseek_rope_tail_dual_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input0,
        uint64_t input0_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        uint32_t position0,
        const axiom_device_buffer *input1,
        uint64_t input1_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t position1,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        uint32_t n_ctx_orig,
        uint32_t inverse);
int axiom_runtime_deepseek_attention_multi_kv_single_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t kv_count,
        float eps);
int axiom_runtime_deepseek_attention_current_history_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *current_kv,
        uint64_t current_kv_offset,
        const axiom_device_buffer *history_kv,
        uint64_t history_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t has_history,
        float eps);
int axiom_runtime_deepseek_attention_raw_comp_ring_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const axiom_device_buffer *comp_kv,
        uint64_t comp_kv_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        float eps);
int axiom_runtime_deepseek_csa_indexer_qat_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim);
int axiom_runtime_deepseek_csa_indexer_topk_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *index_weights,
        uint64_t index_weights_offset,
        const axiom_device_buffer *index_comp,
        uint64_t index_comp_offset,
        axiom_device_buffer *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk);
int axiom_runtime_deepseek_csa_indexer_topk_scratch_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *index_weights,
        uint64_t index_weights_offset,
        const axiom_device_buffer *index_comp,
        uint64_t index_comp_offset,
        axiom_device_buffer *scores,
        uint64_t scores_offset,
        axiom_device_buffer *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk);
int axiom_runtime_deepseek_attention_raw_selected_comp_ring_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const axiom_device_buffer *comp_kv,
        uint64_t comp_kv_offset,
        const axiom_device_buffer *selected,
        uint64_t selected_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps);
int axiom_runtime_deepseek_attention_raw_selected_comp_ring_trusted_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q,
        uint64_t q_offset,
        const axiom_device_buffer *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const axiom_device_buffer *comp_kv,
        uint64_t comp_kv_offset,
        const axiom_device_buffer *selected,
        uint64_t selected_offset,
        const axiom_device_buffer *attn_sink,
        uint64_t attn_sink_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps);
int axiom_runtime_deepseek_csa_cold_window_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim);
int axiom_runtime_deepseek_csa_ring_window_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head);
int axiom_runtime_deepseek_csa_ring_window_count_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count);
int axiom_runtime_deepseek_csa_state_pool_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t head_dim,
        uint32_t use_previous);
int axiom_runtime_deepseek_hca_window_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim);
int axiom_runtime_deepseek_hca_ring_window_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head);
int axiom_runtime_deepseek_hca_ring_window_count_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *bias,
        uint64_t bias_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count);
int axiom_runtime_deepseek_hc_pre_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *fn_f16,
        uint64_t fn_offset,
        const axiom_device_buffer *scale_f32,
        uint64_t scale_offset,
        const axiom_device_buffer *base_f32,
        uint64_t base_offset,
        const axiom_device_buffer *streams,
        uint64_t streams_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        axiom_device_buffer *post,
        uint64_t post_offset,
        axiom_device_buffer *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps);
int axiom_runtime_deepseek_hc_pre_fn_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *fn_f32,
        uint64_t fn_offset,
        const axiom_device_buffer *scale_f32,
        uint64_t scale_offset,
        const axiom_device_buffer *base_f32,
        uint64_t base_offset,
        const axiom_device_buffer *streams,
        uint64_t streams_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        axiom_device_buffer *post,
        uint64_t post_offset,
        axiom_device_buffer *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps);
int axiom_runtime_deepseek_hc_post_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *x,
        uint64_t x_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        const axiom_device_buffer *post,
        uint64_t post_offset,
        const axiom_device_buffer *comb,
        uint64_t comb_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden);
int axiom_runtime_q8_0_matvec_hc_post_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *block_out,
        uint64_t block_out_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        const axiom_device_buffer *post,
        uint64_t post_offset,
        const axiom_device_buffer *comb,
        uint64_t comb_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t cols);
int axiom_runtime_deepseek_ffn_hc_post_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *shared,
        uint64_t shared_offset,
        const axiom_device_buffer *moe,
        uint64_t moe_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        const axiom_device_buffer *post,
        uint64_t post_offset,
        const axiom_device_buffer *comb,
        uint64_t comb_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden);
int axiom_runtime_deepseek_ffn_hc_post_dual_out_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *shared,
        uint64_t shared_offset,
        const axiom_device_buffer *moe,
        uint64_t moe_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        const axiom_device_buffer *post,
        uint64_t post_offset,
        const axiom_device_buffer *comb,
        uint64_t comb_offset,
        axiom_device_buffer *out_a,
        uint64_t out_a_offset,
        axiom_device_buffer *out_b,
        uint64_t out_b_offset,
        uint32_t hidden);
int axiom_runtime_deepseek_ffn_hc_post2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *shared0,
        uint64_t shared0_offset,
        const axiom_device_buffer *moe0,
        uint64_t moe0_offset,
        const axiom_device_buffer *residual0,
        uint64_t residual0_offset,
        const axiom_device_buffer *post0,
        uint64_t post0_offset,
        const axiom_device_buffer *comb0,
        uint64_t comb0_offset,
        const axiom_device_buffer *shared1,
        uint64_t shared1_offset,
        const axiom_device_buffer *moe1,
        uint64_t moe1_offset,
        const axiom_device_buffer *residual1,
        uint64_t residual1_offset,
        const axiom_device_buffer *post1,
        uint64_t post1_offset,
        const axiom_device_buffer *comb1,
        uint64_t comb1_offset,
        axiom_device_buffer *out0,
        uint64_t out0_offset,
        axiom_device_buffer *out1,
        uint64_t out1_offset,
        uint32_t hidden);
int axiom_runtime_deepseek_shared_down_ffn_hc_post_q8_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *down_q8,
        uint64_t down_offset,
        const axiom_device_buffer *shared_mid,
        uint64_t shared_mid_offset,
        const axiom_device_buffer *moe,
        uint64_t moe_offset,
        const axiom_device_buffer *residual,
        uint64_t residual_offset,
        const axiom_device_buffer *post,
        uint64_t post_offset,
        const axiom_device_buffer *comb,
        uint64_t comb_offset,
        axiom_device_buffer *out_a,
        uint64_t out_a_offset,
        axiom_device_buffer *out_b,
        uint64_t out_b_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
int axiom_runtime_deepseek_output_hc_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *fn_f16,
        uint64_t fn_offset,
        const axiom_device_buffer *scale_f32,
        uint64_t scale_offset,
        const axiom_device_buffer *base_f32,
        uint64_t base_offset,
        const axiom_device_buffer *streams,
        uint64_t streams_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps);
int axiom_runtime_deepseek_output_hc_fn_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *fn_f32,
        uint64_t fn_offset,
        const axiom_device_buffer *scale_f32,
        uint64_t scale_offset,
        const axiom_device_buffer *base_f32,
        uint64_t base_offset,
        const axiom_device_buffer *streams,
        uint64_t streams_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps);

int axiom_cluster_create(axiom_cluster **out, const axiom_cluster_config *config);
void axiom_cluster_destroy(axiom_cluster *cluster);
int axiom_cluster_info_get(const axiom_cluster *cluster, axiom_cluster_info *out);
int axiom_cluster_accept(axiom_cluster *listener, axiom_cluster **out_peer);
int axiom_cluster_send_latent(
        axiom_cluster *cluster,
        const axiom_latent_shard *shard,
        const void *payload,
        uint64_t bytes);
int axiom_cluster_recv_latent(
        axiom_cluster *cluster,
        axiom_latent_shard *out_shard,
        void *payload,
        uint64_t payload_capacity,
        uint64_t *out_bytes);
int axiom_cluster_goal_begin(
        axiom_cluster *cluster,
        const axiom_goal_config *config);
int axiom_cluster_spawn_agent(
        axiom_cluster *cluster,
        const axiom_agent_spawn_config *config,
        axiom_agent_spawn_info *out);
int axiom_cluster_recv_agent_spawn(
        axiom_cluster *cluster,
        axiom_agent_spawn_info *out);
int axiom_runtime_attach_cluster(axiom_runtime *runtime, axiom_cluster *cluster);

int axiom_model_open(
        axiom_runtime *runtime,
        axiom_model **out,
        const axiom_model_config *config);
void axiom_model_close(axiom_model *model);
int axiom_model_info_get(axiom_model *model, axiom_model_info *out);
int axiom_model_tensor_info_get(
        axiom_model *model,
        const char *name,
        axiom_tensor_info *out);
int axiom_model_tensor_read(
        axiom_model *model,
        const char *name,
        void *out_data,
        uint64_t out_capacity,
        uint64_t *out_bytes);
int axiom_model_tensor_read_slice(
        axiom_model *model,
        const char *name,
        uint64_t byte_offset,
        void *out_data,
        uint64_t byte_count);
int axiom_model_reside_layer_span(
        axiom_model *model,
        uint32_t layer_start,
        uint32_t layer_end,
        uint64_t max_resident_bytes,
        uint64_t *out_resident_bytes,
        uint32_t *out_tensor_count);
/* P4-C: DEVICE twin of the host residency cache — upload the named tensor's
 * bytes ONCE to the device (sourced through the same single read path as host
 * residency, so the device bytes are exactly the disk bytes). The device
 * buffer is owned by the model and freed at axiom_model_close. After this,
 * axiom_model_linear_bf16_f32 / _rank3_slice_f32 on this tensor execute the
 * SAME matvec kernel on the resident buffer, skipping the per-call weight
 * upload — bit-identical output, only the weight bytes' residence changes.
 * Idempotent (re-residing is a no-op). out_bytes (optional) receives the
 * tensor's device byte count. Family-agnostic: keyed by tensor name. */
int axiom_model_tensor_device_resident(
        axiom_model *model,
        const char *name,
        uint64_t *out_bytes);
/* P4-C introspection: cumulative count of model linear/matvec calls served by
 * the per-op host-upload path vs the device-resident path (the profile proof
 * that resident tensors are never re-uploaded). */
int axiom_model_linear_path_counters(
        axiom_model *model,
        uint64_t *out_host_path_calls,
        uint64_t *out_device_resident_calls);
int axiom_model_embed_token_f32(
        axiom_model *model,
        uint32_t token_id,
        float *out_host,
        uint32_t out_count);
int axiom_model_linear_bf16_f32(
        axiom_model *model,
        const char *weight_name,
        const char *bias_name,
        const float *input_host,
        uint32_t input_count,
        float *out_host,
        uint32_t out_count);
int axiom_model_linear_bf16_rank3_slice_f32(
        axiom_model *model,
        const char *weight_name,
        uint32_t slice,
        const float *input_host,
        uint32_t input_count,
        float *out_host,
        uint32_t out_count);
int axiom_model_rmsnorm_f32(
        axiom_model *model,
        const char *weight_name,
        const float *input_host,
        float *out_host,
        uint32_t count,
        float eps);
int axiom_model_rope_f32(
        axiom_model *model,
        const float *input_host,
        float *out_host,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t position,
        float rope_theta);
int axiom_model_attention_single_f32(
        axiom_model *model,
        const float *q_host,
        const float *k_host,
        const float *v_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim);
int axiom_model_attention_cache_f32(
        axiom_model *model,
        const float *q_host,
        const float *k_cache_host,
        const float *v_cache_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens);
int axiom_model_silu_mul_f32(
        axiom_model *model,
        const float *gate_host,
        const float *up_host,
        float *out_host,
        uint32_t count);
int axiom_model_topk_f32(
        axiom_model *model,
        const float *input_host,
        uint32_t count,
        uint32_t k,
        uint32_t *out_indices_host,
        float *out_values_host);
int axiom_model_tied_lm_head_top1_f32(
        axiom_model *model,
        const float *hidden_host,
        uint32_t hidden_count,
        uint32_t *out_token_id,
        float *out_logit);

int axiom_tokenizer_open(axiom_tokenizer **out, const axiom_tokenizer_config *config);
void axiom_tokenizer_close(axiom_tokenizer *tokenizer);
int axiom_tokenizer_info_get(axiom_tokenizer *tokenizer, axiom_tokenizer_info *out);
/* Return the exact id of an added/special token without relying on a BPE
 * encode pass. This is used by multimodal prompt builders for vision_start,
 * vision_end, image_pad and video_pad. */
int axiom_tokenizer_token_id(
        axiom_tokenizer *tokenizer,
        const char *token,
        uint32_t *out_token_id);
int axiom_tokenizer_same_identity(
        axiom_tokenizer *a,
        axiom_tokenizer *b,
        int *out_same);
int axiom_tokenizer_decode_token(
        axiom_tokenizer *tokenizer,
        uint32_t token_id,
        char *out_text,
        uint32_t out_capacity,
        uint32_t *out_bytes);
int axiom_tokenizer_decode_ids(
        axiom_tokenizer *tokenizer,
        const uint32_t *token_ids,
        uint32_t token_count,
        char *out_text,
        uint32_t out_capacity,
        uint32_t *out_bytes);
int axiom_tokenizer_encode_text(
        axiom_tokenizer *tokenizer,
        const char *text,
        uint32_t *out_token_ids,
        uint32_t out_capacity,
        uint32_t *out_count);

int axiom_entity_create(
        axiom_runtime *runtime,
        axiom_entity **out,
        axiom_model *model,
        const axiom_entity_config *config);
void axiom_entity_destroy(axiom_entity *entity);
int axiom_entity_info_get(axiom_entity *entity, axiom_entity_info *out);

int axiom_session_create(
        axiom_entity *entity,
        axiom_session **out,
        const axiom_session_config *config);
void axiom_session_destroy(axiom_session *session);

int axiom_latent_link_create(
        axiom_runtime *runtime,
        axiom_latent_link **out,
        const axiom_latent_link_config *config);
void axiom_latent_link_destroy(axiom_latent_link *link);
int axiom_latent_link_load_f32(
        axiom_latent_link *link,
        const axiom_latent_link_weights_f32 *weights);

int axiom_latent_link_apply(
        axiom_latent_link *link,
        const axiom_latent_frame *source,
        axiom_latent_frame *target);

int axiom_scheduler_step(
        axiom_runtime *runtime,
        const axiom_scheduler_step_options *options);

int axiom_smoke_vector_add(
        axiom_runtime *runtime,
        const float *a_host,
        const float *b_host,
        float *out_host,
        size_t count);

/* ---- AXIOM PME fused serving kernels (memory-injected GEMV) — APPEND-ONLY block ----
 * One kernel pass: y = dequant(W_base)·x + (B·(A·x))*pack_scale. Base = the proven
 * NVFP4/FP8 matvecs (src/axiom_cuda_nvfp4.cu); pack = hot-attachable rank-r LoRA-style
 * "memory pack" (PME/EXP-067 serving side): A f32 [pack_rank, cols] row-major,
 * B f32 [rows, pack_rank] row-major, pack_scale = alpha/rank precomputed by caller,
 * 1 <= pack_rank <= AXIOM_PME_MAX_RANK. pack_a==NULL && pack_b==NULL => pack OFF =>
 * output BIT-IDENTICAL to the plain matvec (house OFF-gate; proven by
 * tests/axiom_pme_fused_smoke.cpp memcmp on GB10). Raw entries take device pointers
 * already offset by the caller; _device entries take opaque axiom_device_buffer
 * backend handles + byte offsets and self-validate every span (runtime-level wrappers
 * come later). Kernels live in src/axiom_cuda_pme.cu (NOT yet in AXIOM_OBJS/libaxiom —
 * link build/axiom_cuda_pme.o explicitly; see the Makefile pme append-block).
 * Design: docs/axiom_pme_fused_kernel_design.md. */
#define AXIOM_PME_MAX_RANK 256u
int axiom_cuda_e2m1_nvfp4_pme_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);
int axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *pack_a, const float *pack_b, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);
/* Indexed routed-expert variant: pack_a_all f32 [experts, pack_rank, cols], pack_b_all
 * f32 [experts, rows, pack_rank] (one contiguous all-experts allocation); weight/
 * block_scale are the ALREADY-OFFSET per-expert base planes (host MoE loop e*stride
 * pattern). experts <= 65536. pack_a_all==NULL => OFF (bit-identical). */
int axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *pack_a_all, const float *pack_b_all,
        uint32_t experts, uint32_t expert, uint32_t pack_rank, float pack_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);
int axiom_cuda_e2m1_nvfp4_pme_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset, float global_scale,
        const void *pack_a, uint64_t pack_a_offset,
        const void *pack_b, uint64_t pack_b_offset,
        uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols);
int axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset,
        const void *pack_a, uint64_t pack_a_offset,
        const void *pack_b, uint64_t pack_b_offset,
        uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols);
int axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32_device(
        void *cuda_runtime,
        const void *weight, uint64_t weight_offset,
        const void *block_scale, uint64_t block_scale_offset, float global_scale,
        const void *pack_a_all, uint64_t pack_a_all_offset,
        const void *pack_b_all, uint64_t pack_b_all_offset,
        uint32_t experts, uint32_t expert, uint32_t pack_rank, float pack_scale,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t rows, uint32_t cols);

/* ---- Fused NVFP4 routed-expert MoE + de-synced native entries — APPEND-ONLY block ----
 * Device-resident replacement for the host-orchestrated NVFP4 MoE loop
 * (axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device above, which stays as
 * the oracle): router indices/weights and per-expert global scales are consumed ON
 * DEVICE, two kernels total (fused gate+up+silu -> mid[topk, expert_hidden]; weighted
 * down-projection summed over the topk slots -> out OVERWRITTEN with the MoE result).
 * Planes are the same de-blocked combined NVFP4 tensors (weight U8 [experts,rows,cols/2],
 * block_scale U8 [experts,rows,cols/16], global_scale F32 [experts]); requires
 * hidden%32==0 && expert_hidden%32==0 (uint4 nibble loads), topk<=64, experts<=65536.
 * `stream` is a cudaStream_t (NULL = default stream); no cudaDeviceSynchronize inside —
 * completion follows the finish-after-launch convention (AXIOM_CUDA_LAUNCH_ASYNC or
 * stream capture => return after launch, else stream-scoped synchronize). NOT bit-exact
 * vs the host loop (different fp32 accumulation order; parity bar = relative error, see
 * tests/axiom_nvfp4_smoke.cpp fused section). Kernels live in
 * src/axiom_cuda_nvfp4_moe.cu (NOT yet in AXIOM_OBJS/libaxiom — link
 * build/axiom_cuda_nvfp4_moe.o explicitly; see the Makefile append-block).
 * Design: docs/axiom_nvfp4_fused_moe_design.md. */
int axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32(
        int device,
        const uint8_t *gate_w, const uint8_t *gate_bs, const float *gate_gs,
        const uint8_t *up_w, const uint8_t *up_bs, const float *up_gs,
        const uint8_t *down_w, const uint8_t *down_bs, const float *down_gs,
        const uint32_t *router_indices, const float *router_weights,
        const float *input, float *scratch_mid, float *out, void *stream,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden);
int axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32_device(
        void *cuda_runtime,
        const void *gate_w, const void *gate_bs, const void *gate_gs,
        const void *up_w, const void *up_bs, const void *up_gs,
        const void *down_w, const void *down_bs, const void *down_gs,
        const void *input, uint64_t input_offset,
        const void *router_indices, uint64_t router_indices_offset,
        const void *router_weights, uint64_t router_weights_offset,
        void *scratch_mid, uint64_t scratch_mid_offset,
        void *out, uint64_t out_offset,
        void *stream,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden);
/* Runtime-level wrapper with the SAME argument surface as the host-loop entry
 * (axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device) so the forward can
 * switch 1:1. Differences vs the host loop, deliberate and validated here:
 *   - scratch_gate is used as the [topk, expert_hidden] f32 mid plane and must be at
 *     least topk*expert_hidden*4 bytes (the DS4 forward passes moe_mid, which has
 *     exactly that size); scratch_up/scratch_expert_out are accepted for surface
 *     parity, validated (non-NULL, same runtime) but NOT touched;
 *   - hidden/expert_hidden must be multiples of 32 (host loop: 16);
 *   - out is overwritten with the full MoE sum (same net effect as the host loop's
 *     beta=0-then-1 axpby chain);
 *   - an out-of-range router index contributes zero instead of returning an error
 *     (indices stay on device; no sync point to pre-validate them). */
int axiom_runtime_deepseek_moe_nvfp4_indexed_fused_f32_scratch_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate_w, const axiom_device_buffer *gate_bs, const axiom_device_buffer *gate_gs,
        const axiom_device_buffer *up_w, const axiom_device_buffer *up_bs, const axiom_device_buffer *up_gs,
        const axiom_device_buffer *down_w, const axiom_device_buffer *down_bs, const axiom_device_buffer *down_gs,
        const axiom_device_buffer *input, uint64_t input_offset,
        const axiom_device_buffer *router_indices, uint64_t router_indices_offset,
        const axiom_device_buffer *router_weights, uint64_t router_weights_offset,
        axiom_device_buffer *scratch_gate,
        axiom_device_buffer *scratch_up,
        axiom_device_buffer *scratch_expert_out,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden);
/* Stream-aware _async variants of the three raw native NVFP4/FP8 entries in
 * src/axiom_cuda_nvfp4.cu. Same kernels, same validation, same launch geometry as the
 * sync originals (which stay untouched, bit-compat); `stream` is a cudaStream_t
 * (NULL = default stream) and completion follows the finish-after-launch convention
 * described above (no unconditional cudaDeviceSynchronize). */
int axiom_cuda_e2m1_nvfp4_matvec_f32_async(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols, void *stream);
int axiom_cuda_fp8_e4m3_e8m0_matvec_f32_async(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols, void *stream);
int axiom_cuda_axpby_f32_async(
        int device, float *out, const float *in,
        float alpha, float beta, uint32_t n, void *stream);

#ifdef __cplusplus
}
#endif

#endif
