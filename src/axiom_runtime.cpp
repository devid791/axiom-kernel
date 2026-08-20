#include "axiom/axiom.h"

#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <atomic>
#include <mutex>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#if AXIOM_ENABLE_RDMA
#include <rdma/rdma_cma.h>
#include <rdma/rsocket.h>
#endif

#if AXIOM_ENABLE_QUIC
#include <openssl/opensslv.h>
#if !defined(OPENSSL_NO_QUIC) && \
        (OPENSSL_VERSION_MAJOR > 3 || (OPENSSL_VERSION_MAJOR == 3 && OPENSSL_VERSION_MINOR >= 5))
#define AXIOM_HAVE_OPENSSL_QUIC 1
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/quic.h>
#include <openssl/rsa.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#else
#define AXIOM_HAVE_OPENSSL_QUIC 0
#endif
#else
#define AXIOM_HAVE_OPENSSL_QUIC 0
#endif

#ifndef AXIOM_BUILD_TARGET
#define AXIOM_BUILD_TARGET "cuda"
#endif

extern "C" int axiom_cuda_runtime_create(void **out, uint32_t device);
extern "C" void axiom_cuda_runtime_destroy(void *cuda_runtime);
extern "C" int axiom_cuda_device_count(uint32_t *out_count);
extern "C" int axiom_cuda_device_probe(uint32_t device, axiom_device_info *out);
extern "C" int axiom_cuda_runtime_probe(void *cuda_runtime, axiom_device_info *out);
extern "C" int axiom_cuda_smoke_vector_add(
        void *cuda_runtime,
        const float *a_host,
        const float *b_host,
        float *out_host,
        size_t count);
extern "C" int axiom_cuda_latent_link_apply(
        void *cuda_runtime,
        void *cuda_link,
        const void *source,
        void *target,
        uint32_t rows,
        uint32_t source_stride,
        uint32_t target_stride);
extern "C" int axiom_cuda_latent_link_create(
        void *cuda_runtime,
        void **out,
        axiom_latent_link_kind kind,
        axiom_latent_dtype dtype,
        uint32_t source_width,
        uint32_t target_width,
        uint32_t hidden_width,
        float eps);
extern "C" void axiom_cuda_latent_link_destroy(void *cuda_link);
extern "C" int axiom_cuda_latent_link_load_f32(
        void *cuda_link,
        const axiom_latent_link_weights_f32 *weights);
extern "C" int axiom_cuda_bf16_to_f32(
        void *cuda_runtime,
        const uint16_t *bf16_host,
        float *out_host,
        size_t count);
extern "C" int axiom_cuda_bf16_matvec_f32(
        void *cuda_runtime,
        const uint16_t *weight_bf16_host,
        const uint16_t *bias_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_bf16_matvec_f32_resident(
        void *cuda_runtime,
        const void *weight_buffer_opaque,
        uint64_t weight_byte_offset,
        const uint16_t *bias_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_rmsnorm_f32(
        void *cuda_runtime,
        const uint16_t *weight_bf16_host,
        const float *input_host,
        float *out_host,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_rmsnorm_f32_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_rmsnorm_f32_hp_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_rmsnorm_f32_b1_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_rmsnorm_f32_dual_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        void *out0,
        uint64_t out0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_rmsnorm_f32_q8k_device(
        void *cuda_runtime,
        const void *weight_f32,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        void *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count,
        float eps);
extern "C" int axiom_cuda_head_rmsnorm_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
extern "C" int axiom_cuda_rope_neox_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t pos,
        float theta);
extern "C" int axiom_cuda_rope_neox_partial_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos,
        float theta);
extern "C" int axiom_cuda_qknorm_rope_f32_device(
        void *cuda_runtime,
        void *x, uint64_t x_offset,
        const void *w, uint64_t w_offset,
        uint32_t heads, uint32_t head_dim, uint32_t pos, float theta, float eps);
extern "C" int axiom_cuda_qknorm_rope_pos_f32_device(
        void *cuda_runtime,
        void *x, uint64_t x_offset,
        const void *w, uint64_t w_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t heads, uint32_t head_dim, float theta, float eps);
extern "C" int axiom_cuda_qknorm_rope_pos_dual_f32_device(
        void *cuda_runtime,
        void *q, uint64_t q_offset,
        const void *qw, uint64_t qw_offset,
        void *k, uint64_t k_offset,
        const void *kw, uint64_t kw_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim, float theta, float eps);
extern "C" int axiom_cuda_qk_rmsnorm_w_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        const void *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
extern "C" int axiom_cuda_rope_neox_decoupled_partial_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t active_pairs,
        uint32_t exp_dim,
        uint32_t pos,
        float theta);
extern "C" int axiom_cuda_gemma4_qknorm1p_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        const void *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
extern "C" int axiom_cuda_attention_core_scale_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens, float scale,
        uint32_t window);
extern "C" int axiom_cuda_attention_core_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens);
extern "C" int axiom_cuda_attention_core_fast_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_cache, uint64_t k_offset,
        const void *v_cache, uint64_t v_offset,
        void *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens,
        uint32_t window);
extern "C" int axiom_cuda_attention_core_paged_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t window);
extern "C" int axiom_cuda_attention_core_paged_b1_vec_reserve(
        void *cuda_runtime,
        uint32_t active,
        uint32_t tbl_stride,
        uint32_t block_tokens,
        uint32_t q_heads,
        uint32_t head_dim,
        uint32_t window);
extern "C" int axiom_cuda_attention_core_paged_b1_vec_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim,
        uint32_t window);
extern "C" int axiom_cuda_attention_core_scale_paged_f32_device(
        void *cuda_runtime,
        const void *q, uint64_t q_offset,
        const void *k_pool, uint64_t k_offset,
        const void *v_pool, uint64_t v_offset,
        void *out, uint64_t out_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale,
        uint32_t window);
extern "C" int axiom_cuda_kv_pool_store_paged_f32_device(
        void *cuda_runtime,
        const void *k_stage, uint64_t k_stage_offset,
        const void *v_stage, uint64_t v_stage_offset,
        void *k_pool, uint64_t k_pool_offset,
        void *v_pool, uint64_t v_pool_offset,
        const void *block_tables, uint64_t tbl_offset,
        const void *pos, uint64_t pos_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t kv_dim);
extern "C" int axiom_cuda_head_rmsnorm_f32_dual_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
extern "C" int axiom_cuda_deepseek_fp8_kv_quantize_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot);
extern "C" int axiom_cuda_deepseek_fp8_kv_quantize_dual_f32_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot);
extern "C" int axiom_cuda_f32_f16_round_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t count);
extern "C" int axiom_cuda_f32_f16_round_dual_device(
        void *cuda_runtime,
        void *x0,
        uint64_t x0_offset,
        void *x1,
        uint64_t x1_offset,
        uint32_t count);
extern "C" int axiom_cuda_add_f32_device(
        void *cuda_runtime,
        const void *a,
        uint64_t a_offset,
        const void *b,
        uint64_t b_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count);
extern "C" int axiom_cuda_silu_mul_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count);
extern "C" int axiom_cuda_geglu_mul_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count);
extern "C" int axiom_cuda_layernorm_f32_device(
        void *cuda_runtime,
        const void *weight_f32, uint64_t weight_offset,
        const void *bias_f32, uint64_t bias_offset,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t count, float eps);
extern "C" int axiom_cuda_relu2_f32_device(
        void *cuda_runtime,
        const void *input, uint64_t input_offset,
        void *out, uint64_t out_offset,
        uint32_t count);
extern "C" int axiom_cuda_silu_mul_clamp_f32_device(
        void *cuda_runtime,
        const void *gate,
        uint64_t gate_offset,
        const void *up,
        uint64_t up_offset,
        void *out,
        uint64_t out_offset,
        uint32_t count,
        float clamp_abs);
extern "C" int axiom_cuda_pack_f32_to_q8k_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        void *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count);
extern "C" int axiom_cuda_weighted_sum_f32_device(
        void *cuda_runtime,
        const void *inputs,
        uint64_t inputs_offset,
        const void *weights,
        uint64_t weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t slots,
        uint32_t count);
extern "C" int axiom_cuda_f16_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_f16,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_f16_dual_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_a_f16,
        uint64_t weight_a_offset,
        const void *weight_b_f16,
        uint64_t weight_b_offset,
        const void *input,
        uint64_t input_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
extern "C" int axiom_cuda_f16_embedding_streams_device(
        void *cuda_runtime,
        const void *embedding_f16,
        uint64_t embedding_offset,
        void *out_streams,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t hidden,
        uint32_t streams);
extern "C" int axiom_cuda_q8_0_embedding_gather_f32_device(
        void *cuda_runtime,
        const void *embedding_q8,
        uint64_t embedding_offset,
        void *out,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t token_count,
        uint32_t hidden);
extern "C" int axiom_cuda_q8_0_embedding_gather_token_f32_device(
        void *cuda_runtime,
        const void *embedding_q8,
        uint64_t embedding_offset,
        const void *token_id,
        uint64_t token_id_offset,
        void *out,
        uint64_t out_offset,
        uint32_t token_count,
        uint32_t hidden);
extern "C" int axiom_cuda_rope_f32(
        void *cuda_runtime,
        const float *input_host,
        float *out_host,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t position,
        float rope_theta);
extern "C" int axiom_cuda_attention_single_f32(
        void *cuda_runtime,
        const float *q_host,
        const float *k_host,
        const float *v_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim);
extern "C" int axiom_cuda_attention_cache_f32(
        void *cuda_runtime,
        const float *q_host,
        const float *k_cache_host,
        const float *v_cache_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens);
extern "C" int axiom_cuda_silu_mul_f32(
        void *cuda_runtime,
        const float *gate_host,
        const float *up_host,
        float *out_host,
        uint32_t count);
extern "C" int axiom_cuda_topk_f32(
        void *cuda_runtime,
        const float *input_host,
        uint32_t count,
        uint32_t k,
        uint32_t *out_indices_host,
        float *out_values_host);
extern "C" int axiom_cuda_q8_0_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_q8_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q2_k_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_q2k_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_iq2_xxs_matvec_f32(
        void *cuda_runtime,
        const uint8_t *weight_iq2xxs_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_device_buffer_create(
        void *cuda_runtime,
        void **out,
        uint64_t bytes);
extern "C" void axiom_cuda_device_buffer_destroy(void *buffer);
extern "C" int axiom_cuda_device_buffer_upload(
        void *buffer,
        uint64_t offset,
        const void *src_host,
        uint64_t bytes);
extern "C" int axiom_cuda_device_buffer_download(
        void *buffer,
        uint64_t offset,
        void *dst_host,
        uint64_t bytes);
extern "C" int axiom_cuda_device_buffer_copy(
        void *dst,
        uint64_t dst_offset,
        const void *src,
        uint64_t src_offset,
        uint64_t bytes);
extern "C" int axiom_cuda_device_buffer_device(const void *buffer, uint32_t *out_device);
extern "C" int axiom_cuda_device_buffer_pointer(const void *buffer, void **out_pointer);
extern "C" int axiom_cuda_q8_0_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32_device(
        void *cuda_runtime,
        const void *weight,
        uint64_t weight_offset,
        const void *block_scale,
        uint64_t block_scale_offset,
        float global_scale,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32_device(
        void *cuda_runtime,
        const void *weight,
        uint64_t weight_offset,
        const void *block_scale,
        uint64_t block_scale_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_axpby_f32_device(
        void *cuda_runtime,
        void *out,
        uint64_t out_offset,
        const void *in,
        uint64_t in_offset,
        float alpha,
        float beta,
        uint32_t n);
extern "C" int axiom_cuda_q8_0_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_batched_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *x,
        uint64_t x_offset,
        void *y,
        uint64_t y_offset,
        uint32_t y_layout,
        uint32_t batch,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_soa_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_soa_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_soa_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_dual_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_a_q8,
        uint64_t weight_a_offset,
        const void *weight_b_q8,
        uint64_t weight_b_offset,
        const void *input,
        uint64_t input_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_qkv_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q_q8,
        uint64_t weight_q_offset,
        const void *weight_k_q8,
        uint64_t weight_k_offset,
        const void *weight_v_q8,
        uint64_t weight_v_offset,
        const void *input,
        uint64_t input_offset,
        void *out_q,
        uint64_t out_q_offset,
        void *out_k,
        uint64_t out_k_offset,
        void *out_v,
        uint64_t out_v_offset,
        uint32_t rows_q,
        uint32_t rows_k,
        uint32_t rows_v,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_matvec_add_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        const void *residual,
        uint64_t residual_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_dual_matvec_silu_f32_device(
        void *cuda_runtime,
        const void *weight_gate_q8,
        uint64_t weight_gate_offset,
        const void *weight_up_q8,
        uint64_t weight_up_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_dual_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_a_q8,
        uint64_t weight_a_offset,
        const void *weight_b_q8,
        uint64_t weight_b_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out_a0,
        uint64_t out_a0_offset,
        void *out_b0,
        uint64_t out_b0_offset,
        void *out_a1,
        uint64_t out_a1_offset,
        void *out_b1,
        uint64_t out_b1_offset,
        uint32_t rows_a,
        uint32_t rows_b,
        uint32_t cols);
extern "C" int axiom_cuda_q8_0_matvec_argmax_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index,
        float *out_value);
extern "C" int axiom_cuda_q8_0_matvec2_argmax_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index0,
        float *out_value0,
        uint32_t *out_index1,
        float *out_value1);
extern "C" int axiom_cuda_f32_argmax_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        uint32_t count,
        uint32_t *out_index,
        float *out_value);
extern "C" int axiom_cuda_f32_argmax_to_buffer_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        uint32_t count,
        void *out_token_id,
        uint64_t out_token_id_offset);
extern "C" int axiom_cuda_q8_0_grouped_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
extern "C" int axiom_cuda_q8_0_soa_grouped_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
extern "C" int axiom_cuda_q8_0_soa_grouped_matvec2_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
extern "C" int axiom_cuda_q8_0_soa_grouped_matvec4_f32_device(
        void *cuda_runtime,
        const void *weight_qs_i8,
        const void *weight_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *input2,
        uint64_t input2_offset,
        const void *input3,
        uint64_t input3_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t groups,
        uint32_t rows_per_group,
        uint32_t input_per_group);
extern "C" int axiom_cuda_q2_k_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_q2k,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_q4_k_matvec_q8k_device(
        void *cuda_runtime,
        const void *weight_q4k,
        uint64_t weight_offset,
        const void *input_q8k,
        uint64_t input_q8k_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_iq2_xxs_matvec_f32_device(
        void *cuda_runtime,
        const void *weight_iq2xxs,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_iq2_xxs_matvec_f32_warp_device(
        void *cuda_runtime,
        const void *weight_iq2xxs,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols);
extern "C" int axiom_cuda_deepseek_moe_topk_f32_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_indexed_f32_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_indexed_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_gate_up_indexed_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_gate_up_indexed_q8k_scratch_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *scratch_q8k,
        uint64_t scratch_q8k_offset,
        void *out_mid,
        uint64_t out_mid_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_down_indexed_f32_device(
        void *cuda_runtime,
        const void *down_q2k,
        uint64_t down_offset,
        const void *mid,
        uint64_t mid_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_q8k_full_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_xq,
        uint64_t scratch_xq_offset,
        void *scratch_midq,
        uint64_t scratch_midq_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_q8k_full2_device(
        void *cuda_runtime,
        const void *gate_iq2xxs,
        uint64_t gate_offset,
        const void *up_iq2xxs,
        uint64_t up_offset,
        const void *down_q2k,
        uint64_t down_offset,
        const void *input0,
        uint64_t input0_offset,
        const void *indices0,
        uint64_t indices0_offset,
        const void *router_weights0,
        uint64_t router_weights0_offset,
        void *scratch_xq0,
        uint64_t scratch_xq0_offset,
        void *scratch_midq0,
        uint64_t scratch_midq0_offset,
        void *scratch_mid0,
        uint64_t scratch_mid0_offset,
        void *out0,
        uint64_t out0_offset,
        const void *input1,
        uint64_t input1_offset,
        const void *indices1,
        uint64_t indices1_offset,
        const void *router_weights1,
        uint64_t router_weights1_offset,
        void *scratch_xq1,
        uint64_t scratch_xq1_offset,
        void *scratch_midq1,
        uint64_t scratch_midq1_offset,
        void *scratch_mid1,
        uint64_t scratch_mid1_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_moe_q4k_full_device(
        void *cuda_runtime,
        const void *gate_q4k,
        uint64_t gate_offset,
        const void *up_q4k,
        uint64_t up_offset,
        const void *down_q4k,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        const void *router_weights,
        uint64_t router_weights_offset,
        void *scratch_xq,
        uint64_t scratch_xq_offset,
        void *scratch_midq,
        uint64_t scratch_midq_offset,
        void *out,
        uint64_t out_offset,
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_router_topk_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_router_topk_biased_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_router_topk_biased_f32_f32_device(
        void *cuda_runtime,
        const void *router_f32,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_router_topk_biased_f16_f32_scratch_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *bias_f32,
        uint64_t bias_offset,
        const void *input,
        uint64_t input_offset,
        void *scratch_logits,
        uint64_t scratch_logits_offset,
        void *out_indices,
        uint64_t out_indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_router_hash_f16_f32_device(
        void *cuda_runtime,
        const void *router_f16,
        uint64_t router_offset,
        const void *input,
        uint64_t input_offset,
        const void *indices,
        uint64_t indices_offset,
        void *out_weights,
        uint64_t out_weights_offset,
        uint32_t experts,
        uint32_t hidden,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_shared_expert_q8_f32_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *down_q8,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_shared_expert_q8_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *down_q8,
        uint64_t down_offset,
        const void *input,
        uint64_t input_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch_device(
        void *cuda_runtime,
        const void *gate_qs_i8,
        const void *gate_scales_f16,
        const void *up_qs_i8,
        const void *up_scales_f16,
        const void *down_qs_i8,
        const void *down_scales_f16,
        const void *input,
        uint64_t input_offset,
        void *scratch_mid,
        uint64_t scratch_mid_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch2_device(
        void *cuda_runtime,
        const void *gate_qs_i8,
        const void *gate_scales_f16,
        const void *up_qs_i8,
        const void *up_scales_f16,
        const void *down_qs_i8,
        const void *down_scales_f16,
        const void *input0,
        uint64_t input0_offset,
        const void *input1,
        uint64_t input1_offset,
        void *scratch_mid0,
        uint64_t scratch_mid0_offset,
        void *scratch_mid1,
        uint64_t scratch_mid1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_shared_gate_up_q8_f32_device(
        void *cuda_runtime,
        const void *gate_q8,
        uint64_t gate_offset,
        const void *up_q8,
        uint64_t up_offset,
        const void *input,
        uint64_t input_offset,
        void *mid,
        uint64_t mid_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_sliding_attention_single_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *kv,
        uint64_t kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps);
extern "C" int axiom_cuda_deepseek_sliding_attention_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *current_kv,
        uint64_t current_kv_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
extern "C" int axiom_cuda_deepseek_sliding_attention_ring2_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
extern "C" int axiom_cuda_deepseek_sliding_attention_ring4_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *q2,
        uint64_t q2_offset,
        const void *q3,
        uint64_t q3_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *kv2,
        uint64_t kv2_offset,
        const void *kv3,
        uint64_t kv3_offset,
        const void *ring_kv,
        uint64_t ring_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
        uint64_t out3_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        float eps);
extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring2_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head0,
        uint32_t raw_count0,
        uint32_t comp_count0,
        uint32_t comp_count1,
        float eps);
extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring4_causal_f32_device(
        void *cuda_runtime,
        const void *q0,
        uint64_t q0_offset,
        const void *q1,
        uint64_t q1_offset,
        const void *q2,
        uint64_t q2_offset,
        const void *q3,
        uint64_t q3_offset,
        const void *kv0,
        uint64_t kv0_offset,
        const void *kv1,
        uint64_t kv1_offset,
        const void *kv2,
        uint64_t kv2_offset,
        const void *kv3,
        uint64_t kv3_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        void *out2,
        uint64_t out2_offset,
        void *out3,
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
extern "C" int axiom_cuda_deepseek_rope_tail_f32_device(
        void *cuda_runtime,
        const void *input,
        uint64_t input_offset,
        void *out,
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
extern "C" int axiom_cuda_deepseek_rope_tail_dual_f32_device(
        void *cuda_runtime,
        const void *input0,
        uint64_t input0_offset,
        void *out0,
        uint64_t out0_offset,
        uint32_t position0,
        const void *input1,
        uint64_t input1_offset,
        void *out1,
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
extern "C" int axiom_cuda_deepseek_attention_multi_kv_single_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *kv,
        uint64_t kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t kv_count,
        float eps);
extern "C" int axiom_cuda_deepseek_attention_current_history_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *current_kv,
        uint64_t current_kv_offset,
        const void *history_kv,
        uint64_t history_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t has_history,
        float eps);
extern "C" int axiom_cuda_deepseek_attention_raw_comp_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        float eps);
extern "C" int axiom_cuda_deepseek_csa_indexer_qat_f32_device(
        void *cuda_runtime,
        void *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim);
extern "C" int axiom_cuda_deepseek_csa_indexer_topk_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *index_weights,
        uint64_t index_weights_offset,
        const void *index_comp,
        uint64_t index_comp_offset,
        void *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_csa_indexer_topk_scratch_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *index_weights,
        uint64_t index_weights_offset,
        const void *index_comp,
        uint64_t index_comp_offset,
        void *scores,
        uint64_t scores_offset,
        void *selected,
        uint64_t selected_offset,
        uint32_t comp_count,
        uint32_t topk);
extern "C" int axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *selected,
        uint64_t selected_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps);
extern "C" int axiom_cuda_deepseek_attention_raw_selected_comp_ring_trusted_f32_device(
        void *cuda_runtime,
        const void *q,
        uint64_t q_offset,
        const void *raw_kv_ring,
        uint64_t raw_kv_ring_offset,
        const void *comp_kv,
        uint64_t comp_kv_offset,
        const void *selected,
        uint64_t selected_offset,
        const void *attn_sink,
        uint64_t attn_sink_offset,
        void *out,
        uint64_t out_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t raw_slots,
        uint32_t raw_head,
        uint32_t raw_count,
        uint32_t comp_count,
        uint32_t selected_count,
        float eps);
extern "C" int axiom_cuda_deepseek_csa_cold_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim);
extern "C" int axiom_cuda_deepseek_csa_ring_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head);
extern "C" int axiom_cuda_deepseek_csa_ring_window_count_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count);
extern "C" int axiom_cuda_deepseek_csa_state_pool_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        void *out,
        uint64_t out_offset,
        uint32_t head_dim,
        uint32_t use_previous);
extern "C" int axiom_cuda_deepseek_hca_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim);
extern "C" int axiom_cuda_deepseek_hca_ring_window_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head);
extern "C" int axiom_cuda_deepseek_hca_ring_window_count_f32_device(
        void *cuda_runtime,
        const void *kv,
        uint64_t kv_offset,
        const void *gate,
        uint64_t gate_offset,
        const void *bias,
        uint64_t bias_offset,
        void *out,
        uint64_t out_offset,
        uint32_t ratio,
        uint32_t head_dim,
        uint32_t ring_head,
        uint32_t ring_count);
extern "C" int axiom_cuda_deepseek_hc_pre_f32_device(
        void *cuda_runtime,
        const void *fn_f16,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        void *post,
        uint64_t post_offset,
        void *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps);
extern "C" int axiom_cuda_deepseek_hc_pre_fn_f32_device(
        void *cuda_runtime,
        const void *fn_f32,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        void *post,
        uint64_t post_offset,
        void *comb,
        uint64_t comb_offset,
        uint32_t hidden,
        float eps);
extern "C" int axiom_cuda_deepseek_hc_post_f32_device(
        void *cuda_runtime,
        const void *x,
        uint64_t x_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden);
extern "C" int axiom_cuda_q8_0_matvec_hc_post_f32_device(
        void *cuda_runtime,
        const void *weight_q8,
        uint64_t weight_offset,
        const void *input,
        uint64_t input_offset,
        void *block_out,
        uint64_t block_out_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        uint32_t cols);
extern "C" int axiom_cuda_deepseek_ffn_hc_post_f32_device(
        void *cuda_runtime,
        const void *shared,
        uint64_t shared_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden);
extern "C" int axiom_cuda_deepseek_ffn_hc_post_dual_out_f32_device(
        void *cuda_runtime,
        const void *shared,
        uint64_t shared_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t hidden);
extern "C" int axiom_cuda_deepseek_ffn_hc_post2_f32_device(
        void *cuda_runtime,
        const void *shared0,
        uint64_t shared0_offset,
        const void *moe0,
        uint64_t moe0_offset,
        const void *residual0,
        uint64_t residual0_offset,
        const void *post0,
        uint64_t post0_offset,
        const void *comb0,
        uint64_t comb0_offset,
        const void *shared1,
        uint64_t shared1_offset,
        const void *moe1,
        uint64_t moe1_offset,
        const void *residual1,
        uint64_t residual1_offset,
        const void *post1,
        uint64_t post1_offset,
        const void *comb1,
        uint64_t comb1_offset,
        void *out0,
        uint64_t out0_offset,
        void *out1,
        uint64_t out1_offset,
        uint32_t hidden);
extern "C" int axiom_cuda_deepseek_shared_down_ffn_hc_post_q8_f32_device(
        void *cuda_runtime,
        const void *down_q8,
        uint64_t down_offset,
        const void *shared_mid,
        uint64_t shared_mid_offset,
        const void *moe,
        uint64_t moe_offset,
        const void *residual,
        uint64_t residual_offset,
        const void *post,
        uint64_t post_offset,
        const void *comb,
        uint64_t comb_offset,
        void *out_a,
        uint64_t out_a_offset,
        void *out_b,
        uint64_t out_b_offset,
        uint32_t hidden,
        uint32_t expert_hidden);
extern "C" int axiom_cuda_deepseek_output_hc_f32_device(
        void *cuda_runtime,
        const void *fn_f16,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps);
extern "C" int axiom_cuda_deepseek_output_hc_fn_f32_device(
        void *cuda_runtime,
        const void *fn_f32,
        uint64_t fn_offset,
        const void *scale_f32,
        uint64_t scale_offset,
        const void *base_f32,
        uint64_t base_offset,
        const void *streams,
        uint64_t streams_offset,
        void *out,
        uint64_t out_offset,
        uint32_t hidden,
        float eps);

struct axiom_tensor_record {
    std::string name;
    std::string file;
    axiom_tensor_dtype dtype;
    uint32_t rank;
    uint64_t shape[AXIOM_MAX_TENSOR_DIMS];
    uint64_t data_offset_begin;
    uint64_t data_offset_end;
    uint64_t file_offset_begin;
    uint64_t file_offset_end;
    uint64_t byte_count;
};

struct axiom_cluster {
    axiom_transport_kind transport;
    std::string listen_addr;
    std::string join_addr;
    std::string goal_id;
    std::string goal_objective;
    uint32_t node_id;
    uint32_t node_count;
    uint32_t refs;
    uint32_t connected_peers;
    int listen_fd;
    int peer_fd;
#if AXIOM_HAVE_OPENSSL_QUIC
    SSL_CTX *quic_ctx;
    SSL *quic_listener;
    SSL *quic_conn;
#endif
    uint64_t sequence;
    uint64_t flags;
    uint64_t goal_budget_tokens;
    uint64_t goal_flags;
    bool is_listener;
    bool awaiting_ack;
    bool goal_active;
    std::vector<axiom_agent_spawn_info> agents;
};

struct axiom_runtime {
    axiom_backend backend;
    void *backend_runtime;
    axiom_cluster *cluster;
    uint32_t device;
    uint32_t model_count;
    uint32_t entity_count;
    uint32_t session_count;
};

struct axiom_device_buffer {
    axiom_runtime *runtime;
    void *backend_buffer;
    uint32_t device;
    uint64_t bytes;
};

struct axiom_latent_link {
    axiom_runtime *runtime;
    axiom_latent_link_kind kind;
    axiom_latent_dtype dtype;
    uint32_t source_width;
    uint32_t target_width;
    uint32_t hidden_width;
    uint32_t rank;
    float eps;
    void *backend_link;
    bool loaded;
};

struct axiom_model {
    axiom_runtime *runtime;
    std::string path;
    std::string name;
    std::string model_type;
    std::string architecture;
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
    std::vector<axiom_tensor_record> tensors;
    // Weight residency (criterion #1): load each owned tensor ONCE into a
    // persistent host buffer, keyed by name, then serve every read from RAM
    // instead of re-opening+seeking+reading the file per op. Combined with
    // per-node layer-span sharding this is what gives real cross-node capacity.
    // Generic: keyed by name + raw bytes, so it serves any dtype/family.
    std::unordered_map<std::string, std::vector<uint8_t>> resident;
    std::mutex resident_mutex;
    bool resident_enabled = true;   // AXIOM_MODEL_NO_RESIDENT=1 => per-op disk read (old behavior)
    // P4-C: DEVICE twin of the host residency cache — tensors uploaded ONCE to
    // the device (axiom_model_tensor_device_resident), keyed by name, owned by
    // the model, freed at close. Guarded by resident_mutex; device_resident_any
    // is the lock-free fast-path gate so the default (empty-map) linear path
    // pays nothing. The path counters are the per-op-upload profile proof.
    std::unordered_map<std::string, axiom_device_buffer *> device_resident;
    std::atomic<bool> device_resident_any{false};
    std::atomic<uint64_t> linear_host_path_calls{0};
    std::atomic<uint64_t> linear_device_resident_calls{0};
};

struct axiom_tokenizer {
    std::string path;
    std::string name;
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
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<std::string, uint32_t> merge_ranks;
    std::vector<std::pair<std::string, uint32_t>> added_token_pieces;
};

struct axiom_entity {
    axiom_runtime *runtime;
    axiom_model *model;
    std::string name;
    std::string role;
    uint64_t memory_budget_bytes;
    uint32_t session_count;
};

struct axiom_session {
    axiom_entity *entity;
    uint32_t max_context;
    uint64_t kv_budget_bytes;
};

static bool axiom_read_file(const std::filesystem::path &path, std::string *out);
static bool axiom_extract_json_string(
        const std::string &json,
        const char *key,
        std::string *out);
static bool axiom_find_json_object_range(
        const std::string &json,
        const char *key,
        size_t *out_start,
        size_t *out_end);
static void axiom_model_load_config(axiom_model *model, const std::filesystem::path &dir);
static void axiom_model_load_weight_index(axiom_model *model, const std::filesystem::path &dir);
static int axiom_model_load_safetensors_headers(axiom_model *model, const std::filesystem::path &dir);
static const axiom_tensor_record *axiom_model_find_tensor(
        axiom_model *model,
        const char *name);

enum {
    AXIOM_CLUSTER_PROTOCOL_VERSION = 1,
    AXIOM_CLUSTER_WIRE_HEADER_BYTES = 96,
};

static constexpr uint32_t AXIOM_CLUSTER_WIRE_MAGIC = 0x31584d41u; /* AXM1 */
static constexpr uint16_t AXIOM_CLUSTER_FRAME_HELLO = 1u;
static constexpr uint16_t AXIOM_CLUSTER_FRAME_HELLO_ACK = 2u;
static constexpr uint16_t AXIOM_CLUSTER_FRAME_LATENT = 3u;
static constexpr uint16_t AXIOM_CLUSTER_FRAME_GOAL = 4u;
static constexpr uint16_t AXIOM_CLUSTER_FRAME_AGENT_SPAWN = 5u;
static constexpr uint32_t AXIOM_CLUSTER_AGENT_WIRE_MAGIC = 0x31544741u; /* AGT1 */
static constexpr uint32_t AXIOM_CLUSTER_AGENT_WIRE_VERSION = 1u;
static constexpr uint32_t AXIOM_CLUSTER_AGENT_WIRE_BYTES =
        72u + AXIOM_MAX_GOAL_ID + AXIOM_MAX_AGENT_ID +
        AXIOM_MAX_AGENT_ROLE + AXIOM_MAX_AGENT_OBJECTIVE;

struct axiom_cluster_wire_frame {
    uint16_t kind;
    uint32_t flags;
    uint64_t sequence;
    uint64_t payload_bytes;
    uint64_t shard_id;
    uint64_t session_id;
    uint64_t content_hash;
    uint32_t dtype;
    uint32_t rows;
    uint32_t cols;
    uint32_t stride;
    uint32_t source_node;
    uint32_t source_device;
};

static void axiom_store_le16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
}

static void axiom_store_le32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8) & 0xffu);
    p[2] = (uint8_t)((v >> 16) & 0xffu);
    p[3] = (uint8_t)((v >> 24) & 0xffu);
}

static void axiom_store_le64(uint8_t *p, uint64_t v) {
    for (uint32_t i = 0; i < 8; ++i) {
        p[i] = (uint8_t)((v >> (i * 8u)) & 0xffu);
    }
}

static uint16_t axiom_load_le16(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t axiom_load_le32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t axiom_load_le64(const uint8_t *p) {
    uint64_t v = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        v |= ((uint64_t)p[i]) << (i * 8u);
    }
    return v;
}

static uint64_t axiom_fnv1a64(const void *data, uint64_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < bytes; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void axiom_wire_store_cstr(uint8_t *dst, size_t dst_size, const char *src) {
    if (!dst || dst_size == 0) return;
    std::memset(dst, 0, dst_size);
    if (!src) return;
    const size_t n = strnlen(src, dst_size - 1);
    std::memcpy(dst, src, n);
}

static std::string axiom_wire_load_cstr(const uint8_t *src, size_t src_size) {
    if (!src || src_size == 0) return std::string();
    const void *nul = std::memchr(src, 0, src_size);
    const size_t n = nul ? (size_t)((const uint8_t *)nul - src) : src_size;
    return std::string((const char *)src, n);
}

static bool axiom_agent_kind_valid(axiom_agent_kind kind) {
    return kind == AXIOM_AGENT_COORDINATOR ||
           kind == AXIOM_AGENT_WORKER ||
           kind == AXIOM_AGENT_MODEL_WORKER;
}

static bool axiom_placement_kind_valid(axiom_placement_kind kind) {
    return kind == AXIOM_PLACEMENT_LOCAL_DEVICE ||
           kind == AXIOM_PLACEMENT_LOCAL_NODE ||
           kind == AXIOM_PLACEMENT_CLUSTER_NODE ||
           kind == AXIOM_PLACEMENT_CLUSTER_MESH;
}

static void axiom_agent_wire_encode(
        const axiom_agent_spawn_info *info,
        uint8_t *out) {
    std::memset(out, 0, AXIOM_CLUSTER_AGENT_WIRE_BYTES);
    axiom_store_le32(out + 0, AXIOM_CLUSTER_AGENT_WIRE_MAGIC);
    axiom_store_le32(out + 4, AXIOM_CLUSTER_AGENT_WIRE_VERSION);
    axiom_store_le32(out + 8, (uint32_t)info->kind);
    axiom_store_le32(out + 12, (uint32_t)info->status);
    axiom_store_le32(out + 16, (uint32_t)info->placement);
    axiom_store_le32(out + 20, info->source_node);
    axiom_store_le32(out + 24, info->source_device);
    axiom_store_le32(out + 28, info->target_node);
    axiom_store_le32(out + 32, info->target_device);
    axiom_store_le64(out + 40, info->task_id);
    axiom_store_le64(out + 48, info->session_id);
    axiom_store_le64(out + 56, info->budget_tokens);
    axiom_store_le64(out + 64, info->flags);
    size_t off = 72;
    axiom_wire_store_cstr(out + off, AXIOM_MAX_GOAL_ID, info->goal_id);
    off += AXIOM_MAX_GOAL_ID;
    axiom_wire_store_cstr(out + off, AXIOM_MAX_AGENT_ID, info->agent_id);
    off += AXIOM_MAX_AGENT_ID;
    axiom_wire_store_cstr(out + off, AXIOM_MAX_AGENT_ROLE, info->role);
    off += AXIOM_MAX_AGENT_ROLE;
    axiom_wire_store_cstr(out + off, AXIOM_MAX_AGENT_OBJECTIVE, info->objective);
}

static int axiom_agent_wire_decode(
        const uint8_t *in,
        uint64_t bytes,
        axiom_agent_spawn_info *out) {
    if (!in || !out || bytes != AXIOM_CLUSTER_AGENT_WIRE_BYTES) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_load_le32(in + 0) != AXIOM_CLUSTER_AGENT_WIRE_MAGIC ||
        axiom_load_le32(in + 4) != AXIOM_CLUSTER_AGENT_WIRE_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    std::string goal_id = axiom_wire_load_cstr(in + 72, AXIOM_MAX_GOAL_ID);
    std::string agent_id = axiom_wire_load_cstr(
            in + 72 + AXIOM_MAX_GOAL_ID,
            AXIOM_MAX_AGENT_ID);
    std::string role = axiom_wire_load_cstr(
            in + 72 + AXIOM_MAX_GOAL_ID + AXIOM_MAX_AGENT_ID,
            AXIOM_MAX_AGENT_ROLE);
    std::string objective = axiom_wire_load_cstr(
            in + 72 + AXIOM_MAX_GOAL_ID + AXIOM_MAX_AGENT_ID + AXIOM_MAX_AGENT_ROLE,
            AXIOM_MAX_AGENT_OBJECTIVE);
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->kind = (axiom_agent_kind)axiom_load_le32(in + 8);
    out->status = (axiom_agent_status)axiom_load_le32(in + 12);
    out->placement = (axiom_placement_kind)axiom_load_le32(in + 16);
    out->source_node = axiom_load_le32(in + 20);
    out->source_device = axiom_load_le32(in + 24);
    out->target_node = axiom_load_le32(in + 28);
    out->target_device = axiom_load_le32(in + 32);
    out->task_id = axiom_load_le64(in + 40);
    out->session_id = axiom_load_le64(in + 48);
    out->budget_tokens = axiom_load_le64(in + 56);
    out->flags = axiom_load_le64(in + 64);
    axiom_wire_store_cstr((uint8_t *)out->goal_id, sizeof(out->goal_id), goal_id.c_str());
    axiom_wire_store_cstr((uint8_t *)out->agent_id, sizeof(out->agent_id), agent_id.c_str());
    axiom_wire_store_cstr((uint8_t *)out->role, sizeof(out->role), role.c_str());
    axiom_wire_store_cstr((uint8_t *)out->objective, sizeof(out->objective), objective.c_str());
    return AXIOM_OK;
}

static void axiom_wire_encode(const axiom_cluster_wire_frame *frame, uint8_t *out) {
    std::memset(out, 0, AXIOM_CLUSTER_WIRE_HEADER_BYTES);
    axiom_store_le32(out + 0, AXIOM_CLUSTER_WIRE_MAGIC);
    axiom_store_le16(out + 4, AXIOM_CLUSTER_PROTOCOL_VERSION);
    axiom_store_le16(out + 6, frame->kind);
    axiom_store_le32(out + 8, AXIOM_CLUSTER_WIRE_HEADER_BYTES);
    axiom_store_le32(out + 12, frame->flags);
    axiom_store_le64(out + 16, frame->sequence);
    axiom_store_le64(out + 24, frame->payload_bytes);
    axiom_store_le64(out + 32, frame->shard_id);
    axiom_store_le64(out + 40, frame->session_id);
    axiom_store_le64(out + 48, frame->content_hash);
    axiom_store_le32(out + 56, frame->dtype);
    axiom_store_le32(out + 60, frame->rows);
    axiom_store_le32(out + 64, frame->cols);
    axiom_store_le32(out + 68, frame->stride);
    axiom_store_le32(out + 72, frame->source_node);
    axiom_store_le32(out + 76, frame->source_device);
}

static int axiom_wire_decode(const uint8_t *in, axiom_cluster_wire_frame *frame) {
    if (axiom_load_le32(in + 0) != AXIOM_CLUSTER_WIRE_MAGIC ||
        axiom_load_le16(in + 4) != AXIOM_CLUSTER_PROTOCOL_VERSION ||
        axiom_load_le32(in + 8) != AXIOM_CLUSTER_WIRE_HEADER_BYTES) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::memset(frame, 0, sizeof(*frame));
    frame->kind = axiom_load_le16(in + 6);
    frame->flags = axiom_load_le32(in + 12);
    frame->sequence = axiom_load_le64(in + 16);
    frame->payload_bytes = axiom_load_le64(in + 24);
    frame->shard_id = axiom_load_le64(in + 32);
    frame->session_id = axiom_load_le64(in + 40);
    frame->content_hash = axiom_load_le64(in + 48);
    frame->dtype = axiom_load_le32(in + 56);
    frame->rows = axiom_load_le32(in + 60);
    frame->cols = axiom_load_le32(in + 64);
    frame->stride = axiom_load_le32(in + 68);
    frame->source_node = axiom_load_le32(in + 72);
    frame->source_device = axiom_load_le32(in + 76);
    return AXIOM_OK;
}

static bool axiom_cluster_transport_is_stream(axiom_transport_kind transport) {
    return transport == AXIOM_TRANSPORT_TCP ||
           transport == AXIOM_TRANSPORT_RDMA ||
           transport == AXIOM_TRANSPORT_QUIC;
}

static bool axiom_cluster_has_stream_peer(const axiom_cluster *cluster) {
    if (!cluster) return false;
#if AXIOM_HAVE_OPENSSL_QUIC
    if (cluster->transport == AXIOM_TRANSPORT_QUIC) return cluster->quic_conn != nullptr;
#endif
    return cluster->peer_fd >= 0;
}

static void axiom_transport_close(axiom_transport_kind transport, int *fd) {
    if (fd && *fd >= 0) {
#if AXIOM_ENABLE_RDMA
        if (transport == AXIOM_TRANSPORT_RDMA) {
            rclose(*fd);
            *fd = -1;
            return;
        }
#else
        (void)transport;
#endif
        close(*fd);
        *fd = -1;
    }
}

static ssize_t axiom_transport_send_raw(
        axiom_transport_kind transport,
        int fd,
        const void *data,
        size_t bytes,
        int flags) {
#if AXIOM_ENABLE_RDMA
    if (transport == AXIOM_TRANSPORT_RDMA) {
        return rsend(fd, data, bytes, flags);
    }
#else
    (void)transport;
#endif
    return send(fd, data, bytes, flags);
}

static ssize_t axiom_transport_recv_raw(
        axiom_transport_kind transport,
        int fd,
        void *data,
        size_t bytes,
        int flags) {
#if AXIOM_ENABLE_RDMA
    if (transport == AXIOM_TRANSPORT_RDMA) {
        return rrecv(fd, data, bytes, flags);
    }
#else
    (void)transport;
#endif
    return recv(fd, data, bytes, flags);
}

static int axiom_socket_write_all(
        axiom_transport_kind transport,
        int fd,
        const void *data,
        uint64_t bytes) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t remaining = bytes - done;
        const size_t chunk = remaining > (uint64_t)std::numeric_limits<int>::max()
                ? (size_t)std::numeric_limits<int>::max()
                : (size_t)remaining;
#ifdef MSG_NOSIGNAL
        const int send_flags = MSG_NOSIGNAL;
#else
        const int send_flags = 0;
#endif
        const ssize_t n = axiom_transport_send_raw(transport, fd, p + done, chunk, send_flags);
        if (n < 0) {
            if (errno == EINTR) continue;
            return AXIOM_ERR_IO;
        }
        if (n == 0) return AXIOM_ERR_IO;
        done += (uint64_t)n;
    }
    return AXIOM_OK;
}

static int axiom_socket_read_all(
        axiom_transport_kind transport,
        int fd,
        void *data,
        uint64_t bytes) {
    uint8_t *p = (uint8_t *)data;
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t remaining = bytes - done;
        const size_t chunk = remaining > (uint64_t)std::numeric_limits<int>::max()
                ? (size_t)std::numeric_limits<int>::max()
                : (size_t)remaining;
        const ssize_t n = axiom_transport_recv_raw(transport, fd, p + done, chunk, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return AXIOM_ERR_IO;
        }
        if (n == 0) return AXIOM_ERR_IO;
        done += (uint64_t)n;
    }
    return AXIOM_OK;
}

static int axiom_cluster_write_all(axiom_cluster *cluster, const void *data, uint64_t bytes) {
    if (!cluster) return AXIOM_ERR_INVALID_ARGUMENT;
#if AXIOM_HAVE_OPENSSL_QUIC
    if (cluster->transport == AXIOM_TRANSPORT_QUIC) {
        if (!cluster->quic_conn) return AXIOM_ERR_INVALID_ARGUMENT;
        const uint8_t *p = (const uint8_t *)data;
        uint64_t done = 0;
        while (done < bytes) {
            const uint64_t remaining = bytes - done;
            const size_t chunk = remaining > (uint64_t)std::numeric_limits<int>::max()
                    ? (size_t)std::numeric_limits<int>::max()
                    : (size_t)remaining;
            size_t wrote = 0;
            if (SSL_write_ex(cluster->quic_conn, p + done, chunk, &wrote) != 1 || wrote == 0) {
                return AXIOM_ERR_IO;
            }
            done += (uint64_t)wrote;
        }
        return AXIOM_OK;
    }
#endif
    if (cluster->peer_fd < 0) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_socket_write_all(cluster->transport, cluster->peer_fd, data, bytes);
}

static int axiom_cluster_read_all(axiom_cluster *cluster, void *data, uint64_t bytes) {
    if (!cluster) return AXIOM_ERR_INVALID_ARGUMENT;
#if AXIOM_HAVE_OPENSSL_QUIC
    if (cluster->transport == AXIOM_TRANSPORT_QUIC) {
        if (!cluster->quic_conn) return AXIOM_ERR_INVALID_ARGUMENT;
        uint8_t *p = (uint8_t *)data;
        uint64_t done = 0;
        while (done < bytes) {
            const uint64_t remaining = bytes - done;
            const size_t chunk = remaining > (uint64_t)std::numeric_limits<int>::max()
                    ? (size_t)std::numeric_limits<int>::max()
                    : (size_t)remaining;
            size_t read_bytes = 0;
            if (SSL_read_ex(cluster->quic_conn, p + done, chunk, &read_bytes) != 1 ||
                read_bytes == 0) {
                return AXIOM_ERR_IO;
            }
            done += (uint64_t)read_bytes;
        }
        return AXIOM_OK;
    }
#endif
    if (cluster->peer_fd < 0) return AXIOM_ERR_INVALID_ARGUMENT;
    return axiom_socket_read_all(cluster->transport, cluster->peer_fd, data, bytes);
}

static int axiom_cluster_drain(axiom_cluster *cluster, uint64_t bytes) {
    uint8_t scratch[4096];
    uint64_t done = 0;
    while (done < bytes) {
        const uint64_t remaining = bytes - done;
        const uint64_t want = remaining < sizeof(scratch) ? remaining : sizeof(scratch);
        const int rc = axiom_cluster_read_all(cluster, scratch, want);
        if (rc != AXIOM_OK) return rc;
        done += want;
    }
    return AXIOM_OK;
}

static int axiom_socket_set_nodelay(int fd) {
    int yes = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0
            ? AXIOM_OK
            : AXIOM_ERR_IO;
}

static int axiom_parse_tcp_addr(
        const std::string &addr,
        bool bind_addr,
        std::string *out_host,
        std::string *out_port) {
    if (!out_host || !out_port || addr.empty()) return AXIOM_ERR_INVALID_ARGUMENT;
    const size_t colon = addr.rfind(':');
    if (colon == std::string::npos || colon + 1 >= addr.size()) return AXIOM_ERR_INVALID_ARGUMENT;
    std::string host = addr.substr(0, colon);
    std::string port = addr.substr(colon + 1);
    if (host == "*") host.clear();
    if (host.empty()) host = bind_addr ? "0.0.0.0" : "127.0.0.1";
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(port.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0' || parsed > 65535ul) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_host = host;
    *out_port = port;
    return AXIOM_OK;
}

static int axiom_tcp_listen(
        const std::string &addr,
        uint32_t backlog,
        int *out_fd,
        std::string *out_resolved) {
    if (!out_fd || !out_resolved) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_fd = -1;
    std::string host;
    std::string port;
    int rc = axiom_parse_tcp_addr(addr, true, &host, &port);
    if (rc != AXIOM_OK) return rc;

    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo *results = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &results) != 0) {
        return AXIOM_ERR_IO;
    }

    int fd = -1;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            const int listen_backlog = backlog == 0 ? 1 : (int)backlog;
            if (listen(fd, listen_backlog) == 0) break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) return AXIOM_ERR_IO;

    sockaddr_in sin;
    socklen_t sin_len = sizeof(sin);
    std::memset(&sin, 0, sizeof(sin));
    if (getsockname(fd, (sockaddr *)&sin, &sin_len) != 0) {
        close(fd);
        return AXIOM_ERR_IO;
    }
    char ip[INET_ADDRSTRLEN];
    const uint32_t addr_be = sin.sin_addr.s_addr;
    if (addr_be == htonl(INADDR_ANY)) {
        std::strncpy(ip, "127.0.0.1", sizeof(ip));
        ip[sizeof(ip) - 1] = '\0';
    } else if (!inet_ntop(AF_INET, &sin.sin_addr, ip, sizeof(ip))) {
        close(fd);
        return AXIOM_ERR_IO;
    }
    char resolved[160];
    std::snprintf(resolved, sizeof(resolved), "%s:%u", ip, (unsigned)ntohs(sin.sin_port));
    *out_resolved = resolved;
    *out_fd = fd;
    return AXIOM_OK;
}

static int axiom_tcp_connect(const std::string &addr, int *out_fd) {
    if (!out_fd) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_fd = -1;
    std::string host;
    std::string port;
    int rc = axiom_parse_tcp_addr(addr, false, &host, &port);
    if (rc != AXIOM_OK) return rc;
    if (host == "0.0.0.0") host = "127.0.0.1";

    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *results = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &results) != 0) {
        return AXIOM_ERR_IO;
    }

    int fd = -1;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) return AXIOM_ERR_IO;
    rc = axiom_socket_set_nodelay(fd);
    if (rc != AXIOM_OK) {
        close(fd);
        return rc;
    }
    *out_fd = fd;
    return AXIOM_OK;
}

#if AXIOM_ENABLE_RDMA
static bool axiom_rdma_device_available(void) {
    int count = 0;
    ibv_context **devices = rdma_get_devices(&count);
    if (devices) rdma_free_devices(devices);
    return count > 0;
}

static int axiom_rdma_set_nodelay(int fd) {
    int yes = 1;
    return rsetsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) == 0
            ? AXIOM_OK
            : AXIOM_ERR_IO;
}

static int axiom_rdma_listen(
        const std::string &addr,
        uint32_t backlog,
        int *out_fd,
        std::string *out_resolved) {
    if (!out_fd || !out_resolved) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_fd = -1;
    if (!axiom_rdma_device_available()) return AXIOM_ERR_IO;
    std::string host;
    std::string port;
    int rc = axiom_parse_tcp_addr(addr, true, &host, &port);
    if (rc != AXIOM_OK) return rc;

    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo *results = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &results) != 0) {
        return AXIOM_ERR_IO;
    }

    int fd = -1;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = rsocket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        (void)rsetsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (rbind(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            const int listen_backlog = backlog == 0 ? 1 : (int)backlog;
            if (rlisten(fd, listen_backlog) == 0) break;
        }
        rclose(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) return AXIOM_ERR_IO;

    sockaddr_in sin;
    socklen_t sin_len = sizeof(sin);
    std::memset(&sin, 0, sizeof(sin));
    if (rgetsockname(fd, (sockaddr *)&sin, &sin_len) != 0) {
        rclose(fd);
        return AXIOM_ERR_IO;
    }
    char ip[INET_ADDRSTRLEN];
    const uint32_t addr_be = sin.sin_addr.s_addr;
    if (addr_be == htonl(INADDR_ANY)) {
        std::strncpy(ip, "127.0.0.1", sizeof(ip));
        ip[sizeof(ip) - 1] = '\0';
    } else if (!inet_ntop(AF_INET, &sin.sin_addr, ip, sizeof(ip))) {
        rclose(fd);
        return AXIOM_ERR_IO;
    }
    char resolved[160];
    std::snprintf(resolved, sizeof(resolved), "%s:%u", ip, (unsigned)ntohs(sin.sin_port));
    *out_resolved = resolved;
    *out_fd = fd;
    return AXIOM_OK;
}

static int axiom_rdma_connect(const std::string &addr, int *out_fd) {
    if (!out_fd) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_fd = -1;
    if (!axiom_rdma_device_available()) return AXIOM_ERR_IO;
    std::string host;
    std::string port;
    int rc = axiom_parse_tcp_addr(addr, false, &host, &port);
    if (rc != AXIOM_OK) return rc;
    if (host == "0.0.0.0") host = "127.0.0.1";

    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *results = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &results) != 0) {
        return AXIOM_ERR_IO;
    }

    int fd = -1;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = rsocket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (rconnect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        rclose(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) return AXIOM_ERR_IO;
    rc = axiom_rdma_set_nodelay(fd);
    if (rc != AXIOM_OK) {
        rclose(fd);
        return rc;
    }
    *out_fd = fd;
    return AXIOM_OK;
}
#endif

#if AXIOM_HAVE_OPENSSL_QUIC
static const unsigned char axiom_quic_alpn[] = {
    11, 'a', 'x', 'i', 'o', 'm', '-', 'q', 'u', 'i', 'c', '1'
};

static int axiom_quic_select_alpn(
        SSL *,
        const unsigned char **out,
        unsigned char *out_len,
        const unsigned char *in,
        unsigned int in_len,
        void *) {
    if (SSL_select_next_proto(
                (unsigned char **)out,
                out_len,
                axiom_quic_alpn,
                sizeof(axiom_quic_alpn),
                in,
                in_len) == OPENSSL_NPN_NEGOTIATED) {
        return SSL_TLSEXT_ERR_OK;
    }
    return SSL_TLSEXT_ERR_ALERT_FATAL;
}

static int axiom_quic_make_self_signed(SSL_CTX *ctx) {
    int rc = AXIOM_ERR_IO;
    EVP_PKEY *pkey = nullptr;
    EVP_PKEY_CTX *pkey_ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, nullptr);
    X509 *cert = nullptr;
    X509_NAME *name = nullptr;
    if (!ctx || !pkey_ctx) goto done;
    if (EVP_PKEY_keygen_init(pkey_ctx) != 1) goto done;
    if (EVP_PKEY_CTX_set_rsa_keygen_bits(pkey_ctx, 2048) != 1) goto done;
    if (EVP_PKEY_keygen(pkey_ctx, &pkey) != 1 || !pkey) goto done;

    cert = X509_new();
    if (!cert) goto done;
    if (ASN1_INTEGER_set(X509_get_serialNumber(cert), 1) != 1) goto done;
    if (!X509_gmtime_adj(X509_getm_notBefore(cert), 0)) goto done;
    if (!X509_gmtime_adj(X509_getm_notAfter(cert), 24 * 60 * 60)) goto done;
    if (X509_set_pubkey(cert, pkey) != 1) goto done;
    name = X509_get_subject_name(cert);
    if (!name) goto done;
    if (X509_NAME_add_entry_by_txt(
                name,
                "CN",
                MBSTRING_ASC,
                (const unsigned char *)"axiom-quic",
                -1,
                -1,
                0) != 1) {
        goto done;
    }
    if (X509_set_issuer_name(cert, name) != 1) goto done;
    if (!X509_sign(cert, pkey, EVP_sha256())) goto done;
    if (SSL_CTX_use_certificate(ctx, cert) != 1) goto done;
    if (SSL_CTX_use_PrivateKey(ctx, pkey) != 1) goto done;
    if (SSL_CTX_check_private_key(ctx) != 1) goto done;
    rc = AXIOM_OK;

done:
    X509_free(cert);
    EVP_PKEY_free(pkey);
    EVP_PKEY_CTX_free(pkey_ctx);
    return rc;
}

static SSL_CTX *axiom_quic_server_ctx_create(void) {
    SSL_CTX *ctx = SSL_CTX_new(OSSL_QUIC_server_method());
    if (!ctx) return nullptr;
    SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
    SSL_CTX_set_alpn_select_cb(ctx, axiom_quic_select_alpn, nullptr);
    if (axiom_quic_make_self_signed(ctx) != AXIOM_OK) {
        SSL_CTX_free(ctx);
        return nullptr;
    }
    return ctx;
}

static SSL_CTX *axiom_quic_client_ctx_create(void) {
    SSL_CTX *ctx = SSL_CTX_new(OSSL_QUIC_client_method());
    if (!ctx) return nullptr;
    SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr);
    return ctx;
}

static int axiom_quic_sockaddr_to_bio_addr(const sockaddr *sa, BIO_ADDR **out) {
    if (!sa || !out || sa->sa_family != AF_INET) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    const sockaddr_in *sin = (const sockaddr_in *)sa;
    BIO_ADDR *addr = BIO_ADDR_new();
    if (!addr) return AXIOM_ERR_IO;
    if (BIO_ADDR_rawmake(
                addr,
                AF_INET,
                &sin->sin_addr,
                sizeof(sin->sin_addr),
                ntohs(sin->sin_port)) != 1) {
        BIO_ADDR_free(addr);
        return AXIOM_ERR_IO;
    }
    *out = addr;
    return AXIOM_OK;
}

static int axiom_quic_resolve_udp(
        const std::string &addr,
        bool bind_addr,
        addrinfo **out_results) {
    if (!out_results) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_results = nullptr;
    std::string host;
    std::string port;
    int rc = axiom_parse_tcp_addr(addr, bind_addr, &host, &port);
    if (rc != AXIOM_OK) return rc;
    if (!bind_addr && host == "0.0.0.0") host = "127.0.0.1";
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;
    hints.ai_flags = bind_addr ? AI_PASSIVE : 0;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, out_results) != 0) {
        return AXIOM_ERR_IO;
    }
    return AXIOM_OK;
}

static int axiom_quic_listen(
        axiom_cluster *cluster,
        const std::string &addr,
        std::string *out_resolved) {
    if (!cluster || !out_resolved) return AXIOM_ERR_INVALID_ARGUMENT;
    addrinfo *results = nullptr;
    int rc = axiom_quic_resolve_udp(addr, true, &results);
    if (rc != AXIOM_OK) return rc;

    int fd = -1;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int yes = 1;
        (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && BIO_socket_nbio(fd, 1) == 1) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0) return AXIOM_ERR_IO;

    SSL_CTX *ctx = axiom_quic_server_ctx_create();
    if (!ctx) {
        close(fd);
        return AXIOM_ERR_IO;
    }
    SSL *listener = SSL_new_listener(ctx, 0);
    if (!listener) {
        SSL_CTX_free(ctx);
        close(fd);
        return AXIOM_ERR_IO;
    }
    if (SSL_set_fd(listener, fd) != 1 || SSL_set_blocking_mode(listener, 1) != 1 ||
        SSL_listen(listener) != 1) {
        SSL_free(listener);
        SSL_CTX_free(ctx);
        close(fd);
        return AXIOM_ERR_IO;
    }

    sockaddr_in sin;
    socklen_t sin_len = sizeof(sin);
    std::memset(&sin, 0, sizeof(sin));
    if (getsockname(fd, (sockaddr *)&sin, &sin_len) != 0) {
        SSL_free(listener);
        SSL_CTX_free(ctx);
        close(fd);
        return AXIOM_ERR_IO;
    }
    char ip[INET_ADDRSTRLEN];
    const uint32_t addr_be = sin.sin_addr.s_addr;
    if (addr_be == htonl(INADDR_ANY)) {
        std::strncpy(ip, "127.0.0.1", sizeof(ip));
        ip[sizeof(ip) - 1] = '\0';
    } else if (!inet_ntop(AF_INET, &sin.sin_addr, ip, sizeof(ip))) {
        SSL_free(listener);
        SSL_CTX_free(ctx);
        close(fd);
        return AXIOM_ERR_IO;
    }
    char resolved[160];
    std::snprintf(resolved, sizeof(resolved), "%s:%u", ip, (unsigned)ntohs(sin.sin_port));
    cluster->quic_ctx = ctx;
    cluster->quic_listener = listener;
    cluster->listen_fd = fd;
    *out_resolved = resolved;
    return AXIOM_OK;
}

static int axiom_quic_connect(
        axiom_cluster *cluster,
        const std::string &addr) {
    if (!cluster) return AXIOM_ERR_INVALID_ARGUMENT;
    addrinfo *results = nullptr;
    int rc = axiom_quic_resolve_udp(addr, false, &results);
    if (rc != AXIOM_OK) return rc;

    int fd = -1;
    BIO_ADDR *peer_addr = nullptr;
    for (addrinfo *ai = results; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0 && BIO_socket_nbio(fd, 1) == 1 &&
            axiom_quic_sockaddr_to_bio_addr(ai->ai_addr, &peer_addr) == AXIOM_OK) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(results);
    if (fd < 0 || !peer_addr) {
        BIO_ADDR_free(peer_addr);
        return AXIOM_ERR_IO;
    }

    SSL_CTX *ctx = axiom_quic_client_ctx_create();
    SSL *conn = ctx ? SSL_new(ctx) : nullptr;
    if (!ctx || !conn) {
        SSL_free(conn);
        SSL_CTX_free(ctx);
        BIO_ADDR_free(peer_addr);
        close(fd);
        return AXIOM_ERR_IO;
    }
    if (SSL_set_fd(conn, fd) != 1 ||
        SSL_set_blocking_mode(conn, 1) != 1 ||
        SSL_set_tlsext_host_name(conn, "axiom-quic") != 1 ||
        SSL_set_alpn_protos(conn, axiom_quic_alpn, sizeof(axiom_quic_alpn)) != 0 ||
        SSL_set1_initial_peer_addr(conn, peer_addr) != 1 ||
        SSL_set_default_stream_mode(conn, SSL_DEFAULT_STREAM_MODE_AUTO_BIDI) != 1 ||
        SSL_connect(conn) != 1) {
        SSL_free(conn);
        SSL_CTX_free(ctx);
        BIO_ADDR_free(peer_addr);
        close(fd);
        return AXIOM_ERR_IO;
    }
    BIO_ADDR_free(peer_addr);
    cluster->quic_ctx = ctx;
    cluster->quic_conn = conn;
    cluster->peer_fd = fd;
    return AXIOM_OK;
}
#endif

static int axiom_wire_write_frame(
        axiom_cluster *cluster,
        const axiom_cluster_wire_frame *frame,
        const void *payload) {
    uint8_t header[AXIOM_CLUSTER_WIRE_HEADER_BYTES];
    axiom_wire_encode(frame, header);
    int rc = axiom_cluster_write_all(cluster, header, sizeof(header));
    if (rc != AXIOM_OK) return rc;
    if (frame->payload_bytes > 0) {
        if (!payload) return AXIOM_ERR_INVALID_ARGUMENT;
        rc = axiom_cluster_write_all(cluster, payload, frame->payload_bytes);
    }
    return rc;
}

static int axiom_wire_read_frame(
        axiom_cluster *cluster,
        axiom_cluster_wire_frame *frame) {
    uint8_t header[AXIOM_CLUSTER_WIRE_HEADER_BYTES];
    int rc = axiom_cluster_read_all(cluster, header, sizeof(header));
    if (rc != AXIOM_OK) return rc;
    return axiom_wire_decode(header, frame);
}

static int axiom_cluster_write_control(axiom_cluster *cluster, uint16_t kind) {
    if (!cluster) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!axiom_cluster_has_stream_peer(cluster)) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cluster_wire_frame frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.kind = kind;
    frame.flags = (uint32_t)(cluster->flags & 0xffffffffu);
    frame.sequence = ++cluster->sequence;
    frame.source_node = cluster->node_id;
    return axiom_wire_write_frame(cluster, &frame, nullptr);
}

static int axiom_cluster_read_control(axiom_cluster *cluster, uint16_t expected_kind) {
    if (!cluster) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!axiom_cluster_has_stream_peer(cluster)) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_cluster_wire_frame frame;
    const int rc = axiom_wire_read_frame(cluster, &frame);
    if (rc != AXIOM_OK) return rc;
    if (frame.kind != expected_kind || frame.payload_bytes != 0) {
        if (frame.payload_bytes > 0) (void)axiom_cluster_drain(cluster, frame.payload_bytes);
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

static int axiom_cluster_ensure_ack(axiom_cluster *cluster) {
    if (!cluster || !cluster->awaiting_ack) return AXIOM_OK;
    const int rc = axiom_cluster_read_control(cluster, AXIOM_CLUSTER_FRAME_HELLO_ACK);
    if (rc == AXIOM_OK) cluster->awaiting_ack = false;
    return rc;
}

static axiom_cluster *axiom_cluster_alloc_base(
        axiom_transport_kind transport,
        uint32_t node_id,
        uint32_t node_count,
        uint64_t flags) {
    axiom_cluster *cluster = new (std::nothrow) axiom_cluster;
    if (!cluster) return nullptr;
    cluster->transport = transport;
    cluster->node_id = node_id;
    cluster->node_count = node_count;
    cluster->refs = 1;
    cluster->connected_peers = 0;
    cluster->listen_fd = -1;
    cluster->peer_fd = -1;
#if AXIOM_HAVE_OPENSSL_QUIC
    cluster->quic_ctx = nullptr;
    cluster->quic_listener = nullptr;
    cluster->quic_conn = nullptr;
#endif
    cluster->sequence = 0;
    cluster->flags = flags;
    cluster->goal_budget_tokens = 0;
    cluster->goal_flags = 0;
    cluster->is_listener = false;
    cluster->awaiting_ack = false;
    cluster->goal_active = false;
    return cluster;
}

static void axiom_cluster_retain(axiom_cluster *cluster) {
    if (cluster) ++cluster->refs;
}

static void axiom_cluster_release(axiom_cluster *cluster) {
    if (!cluster) return;
    if (cluster->refs > 1) {
        --cluster->refs;
        return;
    }
#if AXIOM_HAVE_OPENSSL_QUIC
    SSL_free(cluster->quic_conn);
    SSL_free(cluster->quic_listener);
    SSL_CTX_free(cluster->quic_ctx);
#endif
    axiom_transport_close(cluster->transport, &cluster->peer_fd);
    axiom_transport_close(cluster->transport, &cluster->listen_fd);
    delete cluster;
}

static void axiom_copy_cstr(char *dst, size_t dst_size, const std::string &src) {
    if (!dst || dst_size == 0) return;
    std::strncpy(dst, src.c_str(), dst_size - 1);
    dst[dst_size - 1] = '\0';
}

static bool axiom_cstr_nonempty_fits(const char *s, size_t max_bytes) {
    if (!s || !s[0] || max_bytes == 0) return false;
    return std::strlen(s) < max_bytes;
}

uint32_t axiom_abi_version(void) {
    return AXIOM_ABI_VERSION;
}

const char *axiom_version(void) {
    return "axiom-kernel 0.1.0 abi=1 cuda";
}

int axiom_abi_info_get(axiom_abi_info *out) {
    if (!out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->header_abi_version = AXIOM_ABI_VERSION;
    out->runtime_abi_version = AXIOM_ABI_VERSION;
    out->struct_size = (uint32_t)sizeof(*out);
    out->version = axiom_version();
    out->backend = "cuda";
    out->build_target = AXIOM_BUILD_TARGET;
    out->flags = 0;
    return AXIOM_OK;
}

int axiom_abi_check(uint32_t header_abi_version) {
    return header_abi_version == AXIOM_ABI_VERSION ? AXIOM_OK : AXIOM_ERR_INVALID_ARGUMENT;
}

const char *axiom_status_string(int status) {
    switch (status) {
    case AXIOM_OK:
        return "ok";
    case AXIOM_ERR_INVALID_ARGUMENT:
        return "invalid argument";
    case AXIOM_ERR_UNSUPPORTED_BACKEND:
        return "unsupported backend";
    case AXIOM_ERR_CUDA:
        return "cuda error";
    case AXIOM_ERR_RUNTIME:
        return "runtime error";
    case AXIOM_ERR_NOT_IMPLEMENTED:
        return "not implemented";
    case AXIOM_ERR_IO:
        return "io error";
    case AXIOM_ERR_BUDGET:
        return "budget exceeded";
    default:
        return "unknown error";
    }
}

int axiom_runtime_create(axiom_runtime **out, const axiom_config *config) {
    if (!out || !config || config->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (config->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }

    axiom_runtime *runtime = (axiom_runtime *)std::calloc(1, sizeof(*runtime));
    if (!runtime) return AXIOM_ERR_RUNTIME;
    runtime->backend = config->backend;
    runtime->device = config->device;

    const int rc = axiom_cuda_runtime_create(&runtime->backend_runtime, config->device);
    if (rc != AXIOM_OK) {
        std::free(runtime);
        return rc;
    }

    *out = runtime;
    return AXIOM_OK;
}

int axiom_runtime_device_count(uint32_t *out_count) {
    if (!out_count) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_count = 0;
    return axiom_cuda_device_count(out_count);
}

int axiom_runtime_probe_device(uint32_t device, axiom_device_info *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    return axiom_cuda_device_probe(device, out);
}

int axiom_runtime_device_id(const axiom_runtime *runtime, uint32_t *out_device) {
    if (!runtime || !out_device) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    *out_device = runtime->device;
    return AXIOM_OK;
}

void axiom_runtime_destroy(axiom_runtime *runtime) {
    if (!runtime) return;
    axiom_cluster_release(runtime->cluster);
    if (runtime->backend == AXIOM_BACKEND_CUDA) {
        axiom_cuda_runtime_destroy(runtime->backend_runtime);
    }
    std::free(runtime);
}

int axiom_runtime_probe(axiom_runtime *runtime, axiom_device_info *out) {
    if (!runtime || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_runtime_probe(runtime->backend_runtime, out);
}

int axiom_runtime_q8_0_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_q8_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!runtime || !weight_q8_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_q8_0_matvec_f32(
            runtime->backend_runtime, weight_q8_host, input_host, out_host, rows, cols);
}

int axiom_runtime_q2_k_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_q2k_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!runtime || !weight_q2k_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_q2_k_matvec_f32(
            runtime->backend_runtime, weight_q2k_host, input_host, out_host, rows, cols);
}

int axiom_runtime_iq2_xxs_matvec_f32(
        axiom_runtime *runtime,
        const uint8_t *weight_iq2xxs_host,
        const float *input_host,
        float *out_host,
        uint32_t rows,
        uint32_t cols) {
    if (!runtime || !weight_iq2xxs_host || !input_host || !out_host ||
        rows == 0 || cols == 0 || (cols % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_iq2_xxs_matvec_f32(
            runtime->backend_runtime, weight_iq2xxs_host, input_host, out_host, rows, cols);
}

int axiom_device_buffer_create(
        axiom_runtime *runtime,
        axiom_device_buffer **out,
        uint64_t bytes) {
    if (!runtime || !out || bytes == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = nullptr;
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    axiom_device_buffer *buffer = (axiom_device_buffer *)std::calloc(1, sizeof(*buffer));
    if (!buffer) return AXIOM_ERR_RUNTIME;
    int rc = axiom_cuda_device_buffer_create(runtime->backend_runtime, &buffer->backend_buffer, bytes);
    if (rc != AXIOM_OK) {
        std::free(buffer);
        return rc;
    }
    buffer->runtime = runtime;
    buffer->device = runtime->device;
    buffer->bytes = bytes;
    *out = buffer;
    return AXIOM_OK;
}

void axiom_device_buffer_destroy(axiom_device_buffer *buffer) {
    if (!buffer) return;
    if (buffer->backend_buffer) axiom_cuda_device_buffer_destroy(buffer->backend_buffer);
    std::free(buffer);
}

int axiom_device_buffer_upload(
        axiom_device_buffer *buffer,
        uint64_t offset,
        const void *src_host,
        uint64_t bytes) {
    if (!buffer || !src_host || bytes == 0 || offset > buffer->bytes ||
        bytes > buffer->bytes - offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_device_buffer_upload(buffer->backend_buffer, offset, src_host, bytes);
}

int axiom_device_buffer_download(
        axiom_device_buffer *buffer,
        uint64_t offset,
        void *dst_host,
        uint64_t bytes) {
    if (!buffer || !dst_host || bytes == 0 || offset > buffer->bytes ||
        bytes > buffer->bytes - offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_device_buffer_download(buffer->backend_buffer, offset, dst_host, bytes);
}

int axiom_device_buffer_copy(
        axiom_device_buffer *dst,
        uint64_t dst_offset,
        const axiom_device_buffer *src,
        uint64_t src_offset,
        uint64_t bytes) {
    if (!dst || !src || bytes == 0 ||
        dst_offset > dst->bytes || bytes > dst->bytes - dst_offset ||
        src_offset > src->bytes || bytes > src->bytes - src_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!dst->runtime || !src->runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    if (dst->runtime->backend != AXIOM_BACKEND_CUDA ||
        src->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_device_buffer_copy(
            dst->backend_buffer,
            dst_offset,
            src->backend_buffer,
            src_offset,
            bytes);
}

int axiom_device_buffer_device_id(const axiom_device_buffer *buffer, uint32_t *out_device) {
    if (!buffer || !out_device) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_device = buffer->device;
    return AXIOM_OK;
}

int axiom_device_buffer_cuda_pointer(const axiom_device_buffer *buffer, void **out_pointer) {
    if (out_pointer) *out_pointer = nullptr;
    if (!buffer || !out_pointer || !buffer->runtime || !buffer->backend_buffer) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (buffer->runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_device_buffer_pointer(buffer->backend_buffer, out_pointer);
}

static int axiom_check_device_matvec_args(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        const axiom_device_buffer *out,
        uint64_t out_offset,
        uint64_t weight_bytes,
        uint64_t input_bytes,
        uint64_t out_bytes) {
    if (!runtime || !weight || !input || !out ||
        weight->runtime != runtime || input->runtime != runtime || out->runtime != runtime ||
        weight_offset > weight->bytes || weight_bytes > weight->bytes - weight_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return AXIOM_OK;
}

static bool axiom_ranges_overlap(
        const axiom_device_buffer *a,
        uint64_t a_offset,
        uint64_t a_bytes,
        const axiom_device_buffer *b,
        uint64_t b_offset,
        uint64_t b_bytes) {
    if (a != b || a_bytes == 0 || b_bytes == 0) return false;
    const uint64_t a_end = a_offset + a_bytes;
    const uint64_t b_end = b_offset + b_bytes;
    return a_offset < b_end && b_offset < a_end;
}

static int axiom_mul_u64_checked(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    if (a != 0 && b > UINT64_MAX / a) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = a * b;
    return AXIOM_OK;
}

static bool axiom_u32_is_power_of_two(uint32_t x) {
    return x != 0 && (x & (x - 1u)) == 0;
}

static int axiom_check_device_buffer_span(
        const axiom_device_buffer *buffer,
        uint64_t offset,
        uint64_t bytes) {
    if (!buffer || offset > buffer->bytes || bytes > buffer->bytes - offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

static int axiom_ds_moe_iq2_bytes_checked(
        uint32_t experts,
        uint32_t expert_hidden,
        uint32_t hidden,
        uint64_t *out) {
    uint64_t bytes = 0;
    int rc = axiom_mul_u64_checked(experts, expert_hidden, &bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, hidden / 256u, &bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, 66u, &bytes) : rc;
    if (rc == AXIOM_OK) *out = bytes;
    return rc;
}

static int axiom_ds_moe_q2_bytes_checked(
        uint32_t experts,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint64_t *out) {
    uint64_t bytes = 0;
    int rc = axiom_mul_u64_checked(experts, hidden, &bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, expert_hidden / 256u, &bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, 84u, &bytes) : rc;
    if (rc == AXIOM_OK) *out = bytes;
    return rc;
}

static int axiom_ds_moe_q4_bytes_checked(
        uint32_t experts,
        uint32_t rows,
        uint32_t cols,
        uint64_t *out) {
    uint64_t bytes = 0;
    int rc = axiom_mul_u64_checked(experts, rows, &bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, cols / 256u, &bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, 144u, &bytes) : rc;
    if (rc == AXIOM_OK) *out = bytes;
    return rc;
}

int axiom_runtime_rmsnorm_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_f32, weight_offset, input, input_offset, out, out_offset,
            bytes, bytes, bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_rmsnorm_f32_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

int axiom_runtime_rmsnorm_f32_hp_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_f32, weight_offset, input, input_offset, out, out_offset,
            bytes, bytes, bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_rmsnorm_f32_hp_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

int axiom_runtime_rmsnorm_f32_b1_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float eps) {
    if (count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_f32, weight_offset, input, input_offset, out, out_offset,
            bytes, bytes, bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_rmsnorm_f32_b1_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

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
        float eps) {
    if (!runtime || !weight_f32 || !input0 || !out0 || !input1 || !out1 || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA ||
        weight_f32->runtime != runtime || input0->runtime != runtime ||
        out0->runtime != runtime || input1->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (weight_offset > weight_f32->bytes || bytes > weight_f32->bytes - weight_offset ||
        input0_offset > input0->bytes || bytes > input0->bytes - input0_offset ||
        out0_offset > out0->bytes || bytes > out0->bytes - out0_offset ||
        input1_offset > input1->bytes || bytes > input1->bytes - input1_offset ||
        out1_offset > out1->bytes || bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_rmsnorm_f32_dual_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer,
            weight_offset,
            input0->backend_buffer,
            input0_offset,
            out0->backend_buffer,
            out0_offset,
            input1->backend_buffer,
            input1_offset,
            out1->backend_buffer,
            out1_offset,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

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
        float eps) {
    if (count == 0 || (count % 256u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    uint64_t q8k_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)(count / 256u), (uint64_t)AXIOM_Q8_K_BLOCK_BYTES, &q8k_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_f32, weight_offset, input, input_offset, out, out_offset,
            bytes, bytes, bytes);
    if (rc != AXIOM_OK) return rc;
    if (!out_q8k || out_q8k_offset > out_q8k->bytes || q8k_bytes > out_q8k->bytes - out_q8k_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_rmsnorm_f32_q8k_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            out_q8k->backend_buffer,
            out_q8k_offset,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

int axiom_runtime_head_rmsnorm_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!runtime || !x || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_head_rmsnorm_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            heads,
            head_dim,
            eps);
}

int axiom_runtime_rope_neox_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t pos,
        float theta) {
    if (!runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_rope_neox_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            heads,
            head_dim,
            pos,
            theta);
}

int axiom_runtime_rope_neox_partial_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos,
        float theta) {
    if (!runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u) ||
        n_rot == 0 || (n_rot & 1u) || n_rot > head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_rope_neox_partial_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            heads,
            head_dim,
            n_rot,
            pos,
            theta);
}

int axiom_runtime_qknorm_rope_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x, uint64_t x_offset,
        const axiom_device_buffer *w, uint64_t w_offset,
        uint32_t heads, uint32_t head_dim, uint32_t pos, float theta, float eps) {
    if (!runtime || !x || !w || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_qknorm_rope_f32_device(
            runtime->backend_runtime,
            x->backend_buffer, x_offset,
            w->backend_buffer, w_offset,
            heads, head_dim, pos, theta, eps);
}

int axiom_runtime_qknorm_rope_pos_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x, uint64_t x_offset,
        const axiom_device_buffer *w, uint64_t w_offset,
        const axiom_device_buffer *pos, uint64_t pos_offset,
        uint32_t heads, uint32_t head_dim, float theta, float eps) {
    if (!runtime || !x || !w || !pos || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime ||
        w->runtime != runtime || pos->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset ||
        w_offset > w->bytes || (uint64_t)head_dim * sizeof(float) > w->bytes - w_offset ||
        pos_offset > pos->bytes || sizeof(uint32_t) > pos->bytes - pos_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_qknorm_rope_pos_f32_device(
            runtime->backend_runtime,
            x->backend_buffer, x_offset,
            w->backend_buffer, w_offset,
            pos->backend_buffer, pos_offset,
            heads, head_dim, theta, eps);
}

int axiom_runtime_qknorm_rope_pos_dual_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *qw, uint64_t qw_offset,
        axiom_device_buffer *k, uint64_t k_offset,
        const axiom_device_buffer *kw, uint64_t kw_offset,
        const axiom_device_buffer *pos, uint64_t pos_offset,
        uint32_t q_heads, uint32_t k_heads, uint32_t head_dim, float theta, float eps) {
    if (!runtime || !q || !qw || !k || !kw || !pos ||
        q_heads == 0 || k_heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA ||
        q->runtime != runtime || qw->runtime != runtime ||
        k->runtime != runtime || kw->runtime != runtime ||
        pos->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)q_heads * head_dim * sizeof(float);
    const uint64_t k_bytes = (uint64_t)k_heads * head_dim * sizeof(float);
    const uint64_t w_bytes = (uint64_t)head_dim * sizeof(float);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        k_offset > k->bytes || k_bytes > k->bytes - k_offset ||
        qw_offset > qw->bytes || w_bytes > qw->bytes - qw_offset ||
        kw_offset > kw->bytes || w_bytes > kw->bytes - kw_offset ||
        pos_offset > pos->bytes || sizeof(uint32_t) > pos->bytes - pos_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_qknorm_rope_pos_dual_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            qw->backend_buffer, qw_offset,
            k->backend_buffer, k_offset,
            kw->backend_buffer, kw_offset,
            pos->backend_buffer, pos_offset,
            q_heads, k_heads, head_dim, theta, eps);
}

int axiom_runtime_qk_rmsnorm_w_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        const axiom_device_buffer *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!runtime || !x || !w || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_qk_rmsnorm_w_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            w->backend_buffer,
            w_offset,
            heads,
            head_dim,
            eps);
}

int axiom_runtime_attention_core_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens) {
    if (!runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_attention_core_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_cache->backend_buffer, k_offset,
            v_cache->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            q_heads, kv_heads, head_dim, cache_tokens);
}

int axiom_runtime_attention_core_fast_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens,
        uint32_t window) {
    if (!runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_attention_core_fast_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_cache->backend_buffer, k_offset,
            v_cache->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            q_heads, kv_heads, head_dim, cache_tokens, window);
}

/* P2 STEP 7 (Gemma-4): DECOUPLED partial NeoX rope — active_pairs (rotated range) and exp_dim
 * (freq denominator) are SEPARATE params (global rotates 64 pairs but denominator 512). */
int axiom_runtime_rope_neox_decoupled_partial_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t active_pairs,
        uint32_t exp_dim,
        uint32_t pos,
        float theta) {
    if (!runtime || !x || heads == 0 || head_dim == 0 || (head_dim & 1u) ||
        active_pairs == 0 || exp_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_rope_neox_decoupled_partial_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            heads,
            head_dim,
            active_pairs,
            exp_dim,
            pos,
            theta);
}

/* P2 STEP 7 (Gemma-4): per-head RMSNorm out=row*inv*w (PLAIN weight, double-accum) faithful to the
 * gemma-4 host rmsnorm1p/qknorm1p. w may be NULL -> v-norm no-scale path (out=row*inv). */
int axiom_runtime_gemma4_qknorm1p_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        const axiom_device_buffer *w,
        uint64_t w_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!runtime || !x || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_gemma4_qknorm1p_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            w ? w->backend_buffer : nullptr,
            w_offset,
            heads,
            head_dim,
            eps);
}

/* P2 STEP 7 (Gemma-4): block-per-head online-softmax GQA attention with a caller scale
 * (gemma-4 uses 1.0 — raw dot product). Mirrors axiom_runtime_attention_core_fast_f32_device. */
int axiom_runtime_attention_core_scale_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_cache, uint64_t k_offset,
        const axiom_device_buffer *v_cache, uint64_t v_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t cache_tokens, float scale,
        uint32_t window) {
    if (!runtime || !q || !k_cache || !v_cache || !out ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_attention_core_scale_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_cache->backend_buffer, k_offset,
            v_cache->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            q_heads, kv_heads, head_dim, cache_tokens, scale, window);
}

/* W5a (continuous batching): shared validation for the PAGED attention-core wrappers — mirrors
 * axiom_runtime_attention_core_fast_f32_device plus span checks for every buffer whose extent
 * is fully determined by the params (q/out [active][q_heads*head_dim] f32, block_tables
 * [active][tbl_stride] i32, cache_tokens [active] u32). The pools' extent depends on the
 * engine-owned physical block count (not a param), so they get runtime-membership checks only. */
static int axiom_attention_core_paged_args_check(
        axiom_runtime *runtime,
        const axiom_device_buffer *q, uint64_t q_offset,
        const axiom_device_buffer *k_pool,
        const axiom_device_buffer *v_pool,
        const axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *block_tables, uint64_t tbl_offset,
        const axiom_device_buffer *cache_tokens, uint64_t ct_offset,
        uint32_t active, uint32_t tbl_stride,
        uint32_t n_layers, uint32_t layer, uint32_t block_tokens, uint32_t pool_stride,
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim) {
    if (!runtime || !q || !k_pool || !v_pool || !out || !block_tables || !cache_tokens ||
        active == 0 || tbl_stride == 0 || n_layers == 0 || layer >= n_layers ||
        block_tokens == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        (q_heads % kv_heads) != 0 ||
        (uint64_t)pool_stride < (uint64_t)kv_heads * head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || k_pool->runtime != runtime || v_pool->runtime != runtime ||
        out->runtime != runtime || block_tables->runtime != runtime ||
        cache_tokens->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t qo_bytes = 0, tbl_bytes = 0, ct_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)active * q_heads, head_dim, &qo_bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(qo_bytes, sizeof(float), &qo_bytes) : rc;
    rc = rc == AXIOM_OK
            ? axiom_mul_u64_checked((uint64_t)active * tbl_stride, sizeof(int32_t), &tbl_bytes)
            : rc;
    rc = rc == AXIOM_OK
            ? axiom_mul_u64_checked((uint64_t)active, sizeof(uint32_t), &ct_bytes)
            : rc;
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_buffer_span(q, q_offset, qo_bytes);
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(out, out_offset, qo_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(block_tables, tbl_offset, tbl_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(cache_tokens, ct_offset, ct_bytes) : rc;
    return rc;
}

/* W5a: PAGED attention core (1/sqrt(head_dim) scale) — same math/online-softmax as
 * axiom_runtime_attention_core_fast_f32_device (byte-identical given identical logical KV) but
 * K/V are read from the device-resident paged pools through per-active-seq block tables.
 * See include/axiom/axiom.h for the addressing contract and layouts. */
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
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t window) {
    int rc = axiom_attention_core_paged_args_check(
            runtime, q, q_offset, k_pool, v_pool, out, out_offset,
            block_tables, tbl_offset, cache_tokens, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_attention_core_paged_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_pool->backend_buffer, k_offset,
            v_pool->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            block_tables->backend_buffer, tbl_offset,
            cache_tokens->backend_buffer, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, window);
}

int axiom_runtime_attention_core_paged_b1_vec_reserve(
        axiom_runtime *runtime,
        uint32_t active, uint32_t tbl_stride, uint32_t block_tokens,
        uint32_t q_heads, uint32_t head_dim, uint32_t window) {
    if (!runtime || active == 0 || tbl_stride == 0 || block_tokens == 0 ||
        q_heads == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_attention_core_paged_b1_vec_reserve(
            runtime->backend_runtime,
            active, tbl_stride, block_tokens, q_heads, head_dim, window);
}

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
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, uint32_t window) {
    int rc = axiom_attention_core_paged_args_check(
            runtime, q, q_offset, k_pool, v_pool, out, out_offset,
            block_tables, tbl_offset, cache_tokens, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_attention_core_paged_b1_vec_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_pool->backend_buffer, k_offset,
            v_pool->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            block_tables->backend_buffer, tbl_offset,
            cache_tokens->backend_buffer, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, window);
}

/* W5a: PAGED attention core with a CALLER-SUPPLIED scale (gemma-4 uses 1.0) — paged variant of
 * axiom_runtime_attention_core_scale_f32_device. */
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
        uint32_t q_heads, uint32_t kv_heads, uint32_t head_dim, float scale, uint32_t window) {
    int rc = axiom_attention_core_paged_args_check(
            runtime, q, q_offset, k_pool, v_pool, out, out_offset,
            block_tables, tbl_offset, cache_tokens, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_attention_core_scale_paged_f32_device(
            runtime->backend_runtime,
            q->backend_buffer, q_offset,
            k_pool->backend_buffer, k_offset,
            v_pool->backend_buffer, v_offset,
            out->backend_buffer, out_offset,
            block_tables->backend_buffer, tbl_offset,
            cache_tokens->backend_buffer, ct_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride,
            q_heads, kv_heads, head_dim, scale, window);
}

/* W5a: scatter one staged K row + one staged V row per active seq ([active][kv_dim] f32 each)
 * into the paged pools at per-seq token position pos[a] (DEVICE u32 [active]) via the same
 * block-table addressing as the paged attention cores. */
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
        uint32_t kv_dim) {
    if (!runtime || !k_stage || !v_stage || !k_pool || !v_pool || !block_tables || !pos ||
        active == 0 || tbl_stride == 0 || n_layers == 0 || layer >= n_layers ||
        block_tokens == 0 || kv_dim == 0 || pool_stride < kv_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (k_stage->runtime != runtime || v_stage->runtime != runtime ||
        k_pool->runtime != runtime || v_pool->runtime != runtime ||
        block_tables->runtime != runtime || pos->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t stage_bytes = 0, tbl_bytes = 0, pos_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)active * kv_dim, sizeof(float), &stage_bytes);
    rc = rc == AXIOM_OK
            ? axiom_mul_u64_checked((uint64_t)active * tbl_stride, sizeof(int32_t), &tbl_bytes)
            : rc;
    rc = rc == AXIOM_OK
            ? axiom_mul_u64_checked((uint64_t)active, sizeof(uint32_t), &pos_bytes)
            : rc;
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_buffer_span(k_stage, k_stage_offset, stage_bytes);
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(v_stage, v_stage_offset, stage_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(block_tables, tbl_offset, tbl_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(pos, pos_offset, pos_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_kv_pool_store_paged_f32_device(
            runtime->backend_runtime,
            k_stage->backend_buffer, k_stage_offset,
            v_stage->backend_buffer, v_stage_offset,
            k_pool->backend_buffer, k_pool_offset,
            v_pool->backend_buffer, v_pool_offset,
            block_tables->backend_buffer, tbl_offset,
            pos->backend_buffer, pos_offset,
            active, tbl_stride, n_layers, layer, block_tokens, pool_stride, kv_dim);
}

/* ---- Gated DeltaNet (Qwen3.6/Qwen3-Next) device thunks ---- */
extern "C" int axiom_cuda_deltanet_conv1d_silu_f32_device(void*, const void*, uint64_t, const void*, uint64_t, void*, uint64_t, void*, uint64_t, uint32_t);
extern "C" int axiom_cuda_deltanet_recurrence_f32_device(void*, const void*, uint64_t, uint64_t, uint64_t, const void*, uint64_t, const void*, uint64_t, void*, uint64_t, void*, uint64_t, uint32_t, uint32_t, uint32_t);
extern "C" int axiom_cuda_deltanet_gated_norm_f32_device(void*, const void*, uint64_t, const void*, uint64_t, const void*, uint64_t, void*, uint64_t, uint32_t, uint32_t, float);

int axiom_runtime_deltanet_conv1d_silu_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *in, uint64_t in_off,
        const axiom_device_buffer *w, uint64_t w_off, axiom_device_buffer *ring, uint64_t ring_off,
        axiom_device_buffer *out, uint64_t out_off, uint32_t conv_dim) {
    if (!runtime || !in || !w || !ring || !out || conv_dim == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (in->runtime != runtime || w->runtime != runtime ||
        ring->runtime != runtime || out->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t conv_bytes = 0, weight_bytes = 0, ring_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)conv_dim, sizeof(float), &conv_bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(conv_bytes, 4u, &weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(conv_bytes, 3u, &ring_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_buffer_span(in, in_off, conv_bytes);
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(w, w_off, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(ring, ring_off, ring_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(out, out_off, conv_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_deltanet_conv1d_silu_f32_device(runtime->backend_runtime,
            in->backend_buffer, in_off, w->backend_buffer, w_off,
            ring->backend_buffer, ring_off, out->backend_buffer, out_off, conv_dim);
}
int axiom_runtime_deltanet_recurrence_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *qkv, uint64_t q_off, uint64_t k_off, uint64_t v_off,
        const axiom_device_buffer *g, uint64_t g_off, const axiom_device_buffer *beta, uint64_t beta_off,
        axiom_device_buffer *state, uint64_t state_off, axiom_device_buffer *out, uint64_t out_off,
        uint32_t n_v_heads, uint32_t n_k_heads, uint32_t head_dim) {
    if (!runtime || !qkv || !g || !beta || !state || !out ||
        n_v_heads == 0 || n_k_heads == 0 || head_dim == 0 ||
        head_dim > 1024u || !axiom_u32_is_power_of_two(head_dim) ||
        n_v_heads < n_k_heads || (n_v_heads % n_k_heads) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (qkv->runtime != runtime || g->runtime != runtime ||
        beta->runtime != runtime || state->runtime != runtime || out->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t kh_items = 0, vh_items = 0, kh_bytes = 0, vh_bytes = 0;
    uint64_t gate_bytes = 0, state_items = 0, state_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)n_k_heads, (uint64_t)head_dim, &kh_items);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)n_v_heads, (uint64_t)head_dim, &vh_items) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(kh_items, sizeof(float), &kh_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(vh_items, sizeof(float), &vh_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)n_v_heads, sizeof(float), &gate_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(vh_items, (uint64_t)head_dim, &state_items) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(state_items, sizeof(float), &state_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_buffer_span(qkv, q_off, kh_bytes);
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(qkv, k_off, kh_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(qkv, v_off, vh_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(g, g_off, gate_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(beta, beta_off, gate_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(state, state_off, state_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(out, out_off, vh_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_deltanet_recurrence_f32_device(runtime->backend_runtime,
            qkv->backend_buffer, q_off, k_off, v_off, g->backend_buffer, g_off, beta->backend_buffer, beta_off,
            state->backend_buffer, state_off, out->backend_buffer, out_off, n_v_heads, n_k_heads, head_dim);
}
int axiom_runtime_deltanet_gated_norm_f32_device(
        axiom_runtime *runtime, const axiom_device_buffer *o, uint64_t o_off,
        const axiom_device_buffer *wnorm, uint64_t wnorm_off, const axiom_device_buffer *z, uint64_t z_off,
        axiom_device_buffer *out, uint64_t out_off, uint32_t n_v_heads, uint32_t head_dim, float eps) {
    if (!runtime || !o || !wnorm || !z || !out ||
        n_v_heads == 0 || head_dim == 0 || head_dim > 1024u ||
        !axiom_u32_is_power_of_two(head_dim) || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (o->runtime != runtime || wnorm->runtime != runtime ||
        z->runtime != runtime || out->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t head_bytes = 0, out_items = 0, out_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)head_dim, sizeof(float), &head_bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)n_v_heads, (uint64_t)head_dim, &out_items) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(out_items, sizeof(float), &out_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_buffer_span(o, o_off, out_bytes);
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(wnorm, wnorm_off, head_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(z, z_off, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_check_device_buffer_span(out, out_off, out_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_deltanet_gated_norm_f32_device(runtime->backend_runtime,
            o->backend_buffer, o_off, wnorm->backend_buffer, wnorm_off, z->backend_buffer, z_off,
            out->backend_buffer, out_off, n_v_heads, head_dim, eps);
}

int axiom_runtime_head_rmsnorm_f32_dual_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t heads,
        uint32_t head_dim,
        float eps) {
    if (!runtime || !x0 || !x1 || heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA ||
        x0->runtime != runtime || x1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (x0_offset > x0->bytes || bytes > x0->bytes - x0_offset ||
        x1_offset > x1->bytes || bytes > x1->bytes - x1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_head_rmsnorm_f32_dual_device(
            runtime->backend_runtime,
            x0->backend_buffer,
            x0_offset,
            x1->backend_buffer,
            x1_offset,
            heads,
            head_dim,
            eps);
}

int axiom_runtime_deepseek_fp8_kv_quantize_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (!runtime || !x || rows == 0 || head_dim == 0 || n_rot >= head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)rows * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_fp8_kv_quantize_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            rows,
            head_dim,
            n_rot);
}

int axiom_runtime_deepseek_fp8_kv_quantize_dual_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t rows,
        uint32_t head_dim,
        uint32_t n_rot) {
    if (!runtime || !x0 || !x1 || rows == 0 || head_dim == 0 || n_rot >= head_dim) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA ||
        x0->runtime != runtime || x1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)rows * head_dim * sizeof(float);
    if (x0_offset > x0->bytes || bytes > x0->bytes - x0_offset ||
        x1_offset > x1->bytes || bytes > x1->bytes - x1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_fp8_kv_quantize_dual_f32_device(
            runtime->backend_runtime,
            x0->backend_buffer,
            x0_offset,
            x1->backend_buffer,
            x1_offset,
            rows,
            head_dim,
            n_rot);
}

int axiom_runtime_f32_f16_round_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t count) {
    if (!runtime || !x || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA || x->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f32_f16_round_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            count);
}

int axiom_runtime_f32_f16_round_dual_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x0,
        uint64_t x0_offset,
        axiom_device_buffer *x1,
        uint64_t x1_offset,
        uint32_t count) {
    if (!runtime || !x0 || !x1 || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA ||
        x0->runtime != runtime || x1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (x0_offset > x0->bytes || bytes > x0->bytes - x0_offset ||
        x1_offset > x1->bytes || bytes > x1->bytes - x1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f32_f16_round_dual_device(
            runtime->backend_runtime,
            x0->backend_buffer,
            x0_offset,
            x1->backend_buffer,
            x1_offset,
            count);
}

int axiom_runtime_add_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *a,
        uint64_t a_offset,
        const axiom_device_buffer *b,
        uint64_t b_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!runtime || !a || !b || !out || count == 0 ||
        a->runtime != runtime || b->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (a_offset > a->bytes || bytes > a->bytes - a_offset ||
        b_offset > b->bytes || bytes > b->bytes - b_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_add_f32_device(
            runtime->backend_runtime,
            a->backend_buffer,
            a_offset,
            b->backend_buffer,
            b_offset,
            out->backend_buffer,
            out_offset,
            count);
}

int axiom_runtime_silu_mul_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!runtime || !gate || !up || !out || count == 0 ||
        gate->runtime != runtime || up->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (gate_offset > gate->bytes || bytes > gate->bytes - gate_offset ||
        up_offset > up->bytes || bytes > up->bytes - up_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_silu_mul_f32_device(
            runtime->backend_runtime,
            gate->backend_buffer,
            gate_offset,
            up->backend_buffer,
            up_offset,
            out->backend_buffer,
            out_offset,
            count);
}

int axiom_runtime_geglu_mul_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count) {
    if (!runtime || !gate || !up || !out || count == 0 ||
        gate->runtime != runtime || up->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (gate_offset > gate->bytes || bytes > gate->bytes - gate_offset ||
        up_offset > up->bytes || bytes > up->bytes - up_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_geglu_mul_f32_device(
            runtime->backend_runtime,
            gate->backend_buffer,
            gate_offset,
            up->backend_buffer,
            up_offset,
            out->backend_buffer,
            out_offset,
            count);
}

int axiom_runtime_layernorm_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f32, uint64_t weight_offset,
        const axiom_device_buffer *bias_f32, uint64_t bias_offset,
        const axiom_device_buffer *input, uint64_t input_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t count, float eps) {
    if (!runtime || !weight_f32 || !bias_f32 || !input || !out || count == 0 ||
        weight_f32->runtime != runtime || bias_f32->runtime != runtime ||
        input->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (weight_offset > weight_f32->bytes || bytes > weight_f32->bytes - weight_offset ||
        bias_offset > bias_f32->bytes || bytes > bias_f32->bytes - bias_offset ||
        input_offset > input->bytes || bytes > input->bytes - input_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_layernorm_f32_device(
            runtime->backend_runtime,
            weight_f32->backend_buffer, weight_offset,
            bias_f32->backend_buffer, bias_offset,
            input->backend_buffer, input_offset,
            out->backend_buffer, out_offset,
            count, eps > 0.0f ? eps : 1.0e-5f);
}

int axiom_runtime_relu2_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input, uint64_t input_offset,
        axiom_device_buffer *out, uint64_t out_offset,
        uint32_t count) {
    if (!runtime || !input || !out || count == 0 ||
        input->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (input_offset > input->bytes || bytes > input->bytes - input_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_relu2_f32_device(
            runtime->backend_runtime,
            input->backend_buffer, input_offset,
            out->backend_buffer, out_offset,
            count);
}

int axiom_runtime_silu_mul_clamp_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        const axiom_device_buffer *up,
        uint64_t up_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t count,
        float clamp_abs) {
    if (!runtime || !gate || !up || !out || count == 0 || clamp_abs < 0.0f ||
        gate->runtime != runtime || up->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (gate_offset > gate->bytes || bytes > gate->bytes - gate_offset ||
        up_offset > up->bytes || bytes > up->bytes - up_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_silu_mul_clamp_f32_device(
            runtime->backend_runtime,
            gate->backend_buffer,
            gate_offset,
            up->backend_buffer,
            up_offset,
            out->backend_buffer,
            out_offset,
            count,
            clamp_abs);
}

int axiom_runtime_pack_f32_to_q8k_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out_q8k,
        uint64_t out_q8k_offset,
        uint32_t count) {
    if (!runtime || !input || !out_q8k || count == 0 || (count % 256u) != 0u ||
        input->runtime != runtime || out_q8k->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    uint64_t q8k_bytes = 0;
    int rc = axiom_mul_u64_checked((uint64_t)(count / 256u), (uint64_t)AXIOM_Q8_K_BLOCK_BYTES, &q8k_bytes);
    if (rc != AXIOM_OK) return rc;
    if (input_offset > input->bytes || bytes > input->bytes - input_offset ||
        out_q8k_offset > out_q8k->bytes || q8k_bytes > out_q8k->bytes - out_q8k_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_pack_f32_to_q8k_device(
            runtime->backend_runtime,
            input->backend_buffer,
            input_offset,
            out_q8k->backend_buffer,
            out_q8k_offset,
            count);
}

int axiom_runtime_weighted_sum_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *inputs,
        uint64_t inputs_offset,
        const axiom_device_buffer *weights,
        uint64_t weights_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t slots,
        uint32_t count) {
    if (!runtime || !inputs || !weights || !out || slots == 0 || count == 0 ||
        inputs->runtime != runtime || weights->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    uint64_t input_items = 0;
    int rc = axiom_mul_u64_checked((uint64_t)slots, (uint64_t)count, &input_items);
    if (rc != AXIOM_OK) return rc;
    uint64_t inputs_bytes = 0;
    uint64_t weights_bytes = 0;
    uint64_t out_bytes = 0;
    rc = axiom_mul_u64_checked(input_items, (uint64_t)sizeof(float), &inputs_bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)slots, (uint64_t)sizeof(float), &weights_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)count, (uint64_t)sizeof(float), &out_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    if (inputs_offset > inputs->bytes || inputs_bytes > inputs->bytes - inputs_offset ||
        weights_offset > weights->bytes || weights_bytes > weights->bytes - weights_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_weighted_sum_f32_device(
            runtime->backend_runtime,
            inputs->backend_buffer,
            inputs_offset,
            weights->backend_buffer,
            weights_offset,
            out->backend_buffer,
            out_offset,
            slots,
            count);
}

int axiom_runtime_f16_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_f16,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * cols * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_f16, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_f16_matvec_f32_device(
            runtime->backend_runtime,
            weight_f16->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows_a == 0 || rows_b == 0 || cols == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_a_bytes = (uint64_t)rows_a * cols * sizeof(uint16_t);
    const uint64_t weight_b_bytes = (uint64_t)rows_b * cols * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_a_bytes = (uint64_t)rows_a * sizeof(float);
    const uint64_t out_b_bytes = (uint64_t)rows_b * sizeof(float);
    int rc = axiom_check_device_matvec_args(
            runtime, weight_a_f16, weight_a_offset, input, input_offset, out_a, out_a_offset,
            weight_a_bytes, input_bytes, out_a_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_b_f16, weight_b_offset, input, input_offset, out_b, out_b_offset,
            weight_b_bytes, input_bytes, out_b_bytes);
    if (rc != AXIOM_OK) return rc;
    if (axiom_ranges_overlap(input, input_offset, input_bytes, out_a, out_a_offset, out_a_bytes) ||
        axiom_ranges_overlap(input, input_offset, input_bytes, out_b, out_b_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_a, out_a_offset, out_a_bytes, out_b, out_b_offset, out_b_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f16_dual_matvec_f32_device(
            runtime->backend_runtime,
            weight_a_f16->backend_buffer,
            weight_a_offset,
            weight_b_f16->backend_buffer,
            weight_b_offset,
            input->backend_buffer,
            input_offset,
            out_a->backend_buffer,
            out_a_offset,
            out_b->backend_buffer,
            out_b_offset,
            rows_a,
            rows_b,
            cols);
}

int axiom_runtime_f16_embedding_streams_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_f16,
        uint64_t embedding_offset,
        axiom_device_buffer *out_streams,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t hidden,
        uint32_t streams) {
    if (!runtime || !embedding_f16 || !out_streams || hidden == 0 || streams == 0 ||
        embedding_f16->runtime != runtime || out_streams->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t row_end = embedding_offset + ((uint64_t)token_id + 1u) * hidden * sizeof(uint16_t);
    const uint64_t out_end = out_offset + (uint64_t)hidden * streams * sizeof(float);
    if (row_end > embedding_f16->bytes || out_end > out_streams->bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f16_embedding_streams_device(
            runtime->backend_runtime,
            embedding_f16->backend_buffer,
            embedding_offset,
            out_streams->backend_buffer,
            out_offset,
            token_id,
            hidden,
            streams);
}

int axiom_runtime_q8_0_embedding_gather_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_q8,
        uint64_t embedding_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t token_id,
        uint32_t token_count,
        uint32_t hidden) {
    if (!runtime || !embedding_q8 || !out || token_count == 0 || hidden == 0 ||
        (hidden % 32u) != 0u || token_id >= token_count ||
        embedding_q8->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t emb_bytes = row_bytes * (uint64_t)token_count;
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (embedding_offset > embedding_q8->bytes || emb_bytes > embedding_q8->bytes - embedding_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_embedding_gather_f32_device(
            runtime->backend_runtime,
            embedding_q8->backend_buffer,
            embedding_offset,
            out->backend_buffer,
            out_offset,
            token_id,
            token_count,
            hidden);
}

int axiom_runtime_q8_0_embedding_gather_token_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *embedding_q8,
        uint64_t embedding_offset,
        const axiom_device_buffer *token_id,
        uint64_t token_id_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t token_count,
        uint32_t hidden) {
    if (!runtime || !embedding_q8 || !token_id || !out || token_count == 0 || hidden == 0 ||
        (hidden % 32u) != 0u ||
        embedding_q8->runtime != runtime || token_id->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t emb_bytes = row_bytes * (uint64_t)token_count;
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (embedding_offset > embedding_q8->bytes || emb_bytes > embedding_q8->bytes - embedding_offset ||
        token_id_offset > token_id->bytes || sizeof(uint32_t) > token_id->bytes - token_id_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_embedding_gather_token_f32_device(
            runtime->backend_runtime,
            embedding_q8->backend_buffer,
            embedding_offset,
            token_id->backend_buffer,
            token_id_offset,
            out->backend_buffer,
            out_offset,
            token_count,
            hidden);
}

int axiom_runtime_q8_0_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_q8, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_q8_0_matvec_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 16u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 2u);
    const uint64_t scale_bytes  = (uint64_t)rows * (cols / 16u);
    const uint64_t input_bytes  = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes    = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    if (!block_scale || block_scale->runtime != runtime ||
        block_scale_offset > block_scale->bytes ||
        scale_bytes > block_scale->bytes - block_scale_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_e2m1_nvfp4_matvec_f32_device(
            runtime->backend_runtime,
            weight->backend_buffer,
            weight_offset,
            block_scale->backend_buffer,
            block_scale_offset,
            global_scale,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (rows % 128u) != 0u || (cols % 128u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t weight_bytes = (uint64_t)rows * cols;
    const uint64_t scale_bytes  = (uint64_t)(rows / 128u) * (cols / 128u);
    const uint64_t input_bytes  = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes    = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    if (!block_scale || block_scale->runtime != runtime ||
        block_scale_offset > block_scale->bytes ||
        scale_bytes > block_scale->bytes - block_scale_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_fp8_e4m3_e8m0_matvec_f32_device(
            runtime->backend_runtime,
            weight->backend_buffer,
            weight_offset,
            block_scale->backend_buffer,
            block_scale_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (!runtime || !weight_q8 || !input0 || !input1 || !out0 || !out1 ||
        weight_q8->runtime != runtime || input0->runtime != runtime ||
        input1->runtime != runtime || out0->runtime != runtime ||
        out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec2_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (!runtime || !weight_q8 || !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        weight_q8->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        input2->runtime != runtime || input3->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime ||
        out2->runtime != runtime || out3->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        input2_offset > input2->bytes || input_bytes > input2->bytes - input2_offset ||
        input3_offset > input3->bytes || input_bytes > input3->bytes - input3_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset ||
        out2_offset > out2->bytes || out_bytes > out2->bytes - out2_offset ||
        out3_offset > out3->bytes || out_bytes > out3->bytes - out3_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out2, out2_offset, out_bytes, out3, out3_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec4_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            input2->backend_buffer,
            input2_offset,
            input3->backend_buffer,
            input3_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            out2->backend_buffer,
            out2_offset,
            out3->backend_buffer,
            out3_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (batch == 0 || rows == 0 || cols == 0 || (cols % 32u) != 0u || y_layout > 1u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!runtime || !weight_q8 || !x || !y ||
        weight_q8->runtime != runtime || x->runtime != runtime || y->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    uint64_t weight_bytes = 0;
    uint64_t x_bytes = 0;
    uint64_t y_bytes = 0;
    if (axiom_mul_u64_checked(rows, row_bytes, &weight_bytes) != AXIOM_OK ||
        axiom_mul_u64_checked(batch, (uint64_t)cols * sizeof(float), &x_bytes) != AXIOM_OK ||
        axiom_mul_u64_checked(batch, (uint64_t)rows * sizeof(float), &y_bytes) != AXIOM_OK) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_check_device_buffer_span(weight_q8, weight_offset, weight_bytes) != AXIOM_OK ||
        axiom_check_device_buffer_span(x, x_offset, x_bytes) != AXIOM_OK ||
        axiom_check_device_buffer_span(y, y_offset, y_bytes) != AXIOM_OK) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(x, x_offset, x_bytes, y, y_offset, y_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_batched_matvec_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            x->backend_buffer,
            x_offset,
            y->backend_buffer,
            y_offset,
            y_layout,
            batch,
            rows,
            cols);
}

int axiom_runtime_q8_0_soa_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_qs_i8,
        const axiom_device_buffer *weight_scales_f16,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t qs_bytes = (uint64_t)rows * cols;
    const uint64_t scale_bytes = (uint64_t)rows * (cols / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 || !input || !out ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_matvec_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t qs_bytes = (uint64_t)rows * cols;
    const uint64_t scale_bytes = (uint64_t)rows * (cols / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 || !input0 || !input1 || !out0 || !out1 ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_matvec2_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t qs_bytes = (uint64_t)rows * cols;
    const uint64_t scale_bytes = (uint64_t)rows * (cols / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        input2->runtime != runtime || input3->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime ||
        out2->runtime != runtime || out3->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        input2_offset > input2->bytes || input_bytes > input2->bytes - input2_offset ||
        input3_offset > input3->bytes || input_bytes > input3->bytes - input3_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset ||
        out2_offset > out2->bytes || out_bytes > out2->bytes - out2_offset ||
        out3_offset > out3->bytes || out_bytes > out3->bytes - out3_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out2, out2_offset, out_bytes, out3, out3_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_matvec4_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            input2->backend_buffer,
            input2_offset,
            input3->backend_buffer,
            input3_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            out2->backend_buffer,
            out2_offset,
            out3->backend_buffer,
            out3_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_a_bytes = (uint64_t)rows_a * row_bytes;
    const uint64_t weight_b_bytes = (uint64_t)rows_b * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_a_bytes = (uint64_t)rows_a * sizeof(float);
    const uint64_t out_b_bytes = (uint64_t)rows_b * sizeof(float);
    int rc = axiom_check_device_matvec_args(
            runtime, weight_a_q8, weight_a_offset, input, input_offset, out_a, out_a_offset,
            weight_a_bytes, input_bytes, out_a_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_b_q8, weight_b_offset, input, input_offset, out_b, out_b_offset,
            weight_b_bytes, input_bytes, out_b_bytes);
    if (rc != AXIOM_OK) return rc;
    if (axiom_ranges_overlap(input, input_offset, input_bytes, out_a, out_a_offset, out_a_bytes) ||
        axiom_ranges_overlap(input, input_offset, input_bytes, out_b, out_b_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_a, out_a_offset, out_a_bytes, out_b, out_b_offset, out_b_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_dual_matvec_f32_device(
            runtime->backend_runtime,
            weight_a_q8->backend_buffer,
            weight_a_offset,
            weight_b_q8->backend_buffer,
            weight_b_offset,
            input->backend_buffer,
            input_offset,
            out_a->backend_buffer,
            out_a_offset,
            out_b->backend_buffer,
            out_b_offset,
            rows_a,
            rows_b,
            cols);
}

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
        uint32_t cols) {
    if (rows_q == 0 || rows_k == 0 || rows_v == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_q_bytes = (uint64_t)rows_q * row_bytes;
    const uint64_t weight_k_bytes = (uint64_t)rows_k * row_bytes;
    const uint64_t weight_v_bytes = (uint64_t)rows_v * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_q_bytes = (uint64_t)rows_q * sizeof(float);
    const uint64_t out_k_bytes = (uint64_t)rows_k * sizeof(float);
    const uint64_t out_v_bytes = (uint64_t)rows_v * sizeof(float);
    int rc = axiom_check_device_matvec_args(
            runtime, weight_q_q8, weight_q_offset, input, input_offset, out_q, out_q_offset,
            weight_q_bytes, input_bytes, out_q_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_k_q8, weight_k_offset, input, input_offset, out_k, out_k_offset,
            weight_k_bytes, input_bytes, out_k_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_v_q8, weight_v_offset, input, input_offset, out_v, out_v_offset,
            weight_v_bytes, input_bytes, out_v_bytes);
    if (rc != AXIOM_OK) return rc;
    if (axiom_ranges_overlap(input, input_offset, input_bytes, out_q, out_q_offset, out_q_bytes) ||
        axiom_ranges_overlap(input, input_offset, input_bytes, out_k, out_k_offset, out_k_bytes) ||
        axiom_ranges_overlap(input, input_offset, input_bytes, out_v, out_v_offset, out_v_bytes) ||
        axiom_ranges_overlap(out_q, out_q_offset, out_q_bytes, out_k, out_k_offset, out_k_bytes) ||
        axiom_ranges_overlap(out_q, out_q_offset, out_q_bytes, out_v, out_v_offset, out_v_bytes) ||
        axiom_ranges_overlap(out_k, out_k_offset, out_k_bytes, out_v, out_v_offset, out_v_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_qkv_matvec_f32_device(
            runtime->backend_runtime,
            weight_q_q8->backend_buffer,
            weight_q_offset,
            weight_k_q8->backend_buffer,
            weight_k_offset,
            weight_v_q8->backend_buffer,
            weight_v_offset,
            input->backend_buffer,
            input_offset,
            out_q->backend_buffer,
            out_q_offset,
            out_k->backend_buffer,
            out_k_offset,
            out_v->backend_buffer,
            out_v_offset,
            rows_q,
            rows_k,
            rows_v,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u || !runtime || !residual ||
        residual->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    int rc = axiom_check_device_matvec_args(
            runtime, weight_q8, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    if (residual_offset > residual->bytes || out_bytes > residual->bytes - residual_offset ||
        axiom_ranges_overlap(input, input_offset, input_bytes, out, out_offset, out_bytes) ||
        axiom_ranges_overlap(input, input_offset, input_bytes, residual, residual_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec_add_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            residual->backend_buffer,
            residual_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    int rc = axiom_check_device_matvec_args(
            runtime, weight_gate_q8, weight_gate_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    rc = axiom_check_device_matvec_args(
            runtime, weight_up_q8, weight_up_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    if (axiom_ranges_overlap(input, input_offset, input_bytes, out, out_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_dual_matvec_silu_f32_device(
            runtime->backend_runtime,
            weight_gate_q8->backend_buffer,
            weight_gate_offset,
            weight_up_q8->backend_buffer,
            weight_up_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t cols) {
    if (rows_a == 0 || rows_b == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t row_bytes = (uint64_t)(cols / 32u) * 34u;
    const uint64_t weight_a_bytes = (uint64_t)rows_a * row_bytes;
    const uint64_t weight_b_bytes = (uint64_t)rows_b * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_a_bytes = (uint64_t)rows_a * sizeof(float);
    const uint64_t out_b_bytes = (uint64_t)rows_b * sizeof(float);
    if (!runtime || !weight_a_q8 || !weight_b_q8 || !input0 || !input1 ||
        !out_a0 || !out_b0 || !out_a1 || !out_b1 ||
        weight_a_q8->runtime != runtime || weight_b_q8->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        out_a0->runtime != runtime || out_b0->runtime != runtime ||
        out_a1->runtime != runtime || out_b1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (weight_a_offset > weight_a_q8->bytes || weight_a_bytes > weight_a_q8->bytes - weight_a_offset ||
        weight_b_offset > weight_b_q8->bytes || weight_b_bytes > weight_b_q8->bytes - weight_b_offset ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        out_a0_offset > out_a0->bytes || out_a_bytes > out_a0->bytes - out_a0_offset ||
        out_b0_offset > out_b0->bytes || out_b_bytes > out_b0->bytes - out_b0_offset ||
        out_a1_offset > out_a1->bytes || out_a_bytes > out_a1->bytes - out_a1_offset ||
        out_b1_offset > out_b1->bytes || out_b_bytes > out_b1->bytes - out_b1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out_a0, out_a0_offset, out_a_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out_b0, out_b0_offset, out_b_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out_a1, out_a1_offset, out_a_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out_b1, out_b1_offset, out_b_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out_a0, out_a0_offset, out_a_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out_b0, out_b0_offset, out_b_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out_a1, out_a1_offset, out_a_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out_b1, out_b1_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_a0, out_a0_offset, out_a_bytes, out_b0, out_b0_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_a0, out_a0_offset, out_a_bytes, out_a1, out_a1_offset, out_a_bytes) ||
        axiom_ranges_overlap(out_a0, out_a0_offset, out_a_bytes, out_b1, out_b1_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_b0, out_b0_offset, out_b_bytes, out_a1, out_a1_offset, out_a_bytes) ||
        axiom_ranges_overlap(out_b0, out_b0_offset, out_b_bytes, out_b1, out_b1_offset, out_b_bytes) ||
        axiom_ranges_overlap(out_a1, out_a1_offset, out_a_bytes, out_b1, out_b1_offset, out_b_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_dual_matvec2_f32_device(
            runtime->backend_runtime,
            weight_a_q8->backend_buffer,
            weight_a_offset,
            weight_b_q8->backend_buffer,
            weight_b_offset,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            out_a0->backend_buffer,
            out_a0_offset,
            out_b0->backend_buffer,
            out_b0_offset,
            out_a1->backend_buffer,
            out_a1_offset,
            out_b1->backend_buffer,
            out_b1_offset,
            rows_a,
            rows_b,
            cols);
}

int axiom_runtime_f32_argmax_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t count,
        uint32_t *out_index,
        float *out_value) {
    if (!runtime || !input || !out_index || !out_value || count == 0 ||
        input->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    if (input_offset > input->bytes || bytes > input->bytes - input_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f32_argmax_device(
            runtime->backend_runtime,
            input->backend_buffer,
            input_offset,
            count,
            out_index,
            out_value);
}

int axiom_runtime_f32_argmax_to_buffer_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t count,
        axiom_device_buffer *out_token_id,
        uint64_t out_token_id_offset) {
    if (!runtime || !input || !out_token_id || count == 0 ||
        input->runtime != runtime || out_token_id->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t input_bytes = (uint64_t)count * sizeof(float);
    if (input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_token_id_offset > out_token_id->bytes ||
        sizeof(uint32_t) > out_token_id->bytes - out_token_id_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_f32_argmax_to_buffer_device(
            runtime->backend_runtime,
            input->backend_buffer,
            input_offset,
            count,
            out_token_id->backend_buffer,
            out_token_id_offset);
}

int axiom_runtime_q8_0_matvec_argmax_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q8,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        uint32_t rows,
        uint32_t cols,
        uint32_t *out_index,
        float *out_value) {
    if (!runtime || !weight_q8 || !input || !out_index || !out_value ||
        weight_q8->runtime != runtime || input->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec_argmax_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            rows,
            cols,
            out_index,
            out_value);
}

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
        float *out_value1) {
    if (!runtime || !weight_q8 || !input0 || !input1 ||
        !out_index0 || !out_value0 || !out_index1 || !out_value1 ||
        weight_q8->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (rows == 0 || cols == 0 || (cols % 32u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec2_argmax_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            rows,
            cols,
            out_index0,
            out_value0,
            out_index1,
            out_value1);
}

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
        uint32_t input_per_group) {
    if (!runtime || !weight_q8 || !input || !out ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u ||
        weight_q8->runtime != runtime || input->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t weight_bytes =
            (uint64_t)groups * rows_per_group * (input_per_group / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)groups * input_per_group * sizeof(float);
    const uint64_t out_bytes = (uint64_t)groups * rows_per_group * sizeof(float);
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_grouped_matvec_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            groups,
            rows_per_group,
            input_per_group);
}

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
        uint32_t input_per_group) {
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 || !input || !out ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const uint64_t qs_bytes = rows * input_per_group;
    const uint64_t scale_bytes = rows * (input_per_group / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)groups * input_per_group * sizeof(float);
    const uint64_t out_bytes = rows * sizeof(float);
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_grouped_matvec_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            groups,
            rows_per_group,
            input_per_group);
}

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
        uint32_t input_per_group) {
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 || !input0 || !input1 || !out0 || !out1 ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const uint64_t qs_bytes = rows * input_per_group;
    const uint64_t scale_bytes = rows * (input_per_group / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)groups * input_per_group * sizeof(float);
    const uint64_t out_bytes = rows * sizeof(float);
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_grouped_matvec2_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            groups,
            rows_per_group,
            input_per_group);
}

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
        uint32_t input_per_group) {
    if (!runtime || !weight_qs_i8 || !weight_scales_f16 ||
        !input0 || !input1 || !input2 || !input3 ||
        !out0 || !out1 || !out2 || !out3 ||
        groups == 0 || rows_per_group == 0 || input_per_group == 0 ||
        (input_per_group % 32u) != 0u ||
        weight_qs_i8->runtime != runtime || weight_scales_f16->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        input2->runtime != runtime || input3->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime ||
        out2->runtime != runtime || out3->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    const uint64_t rows = (uint64_t)groups * rows_per_group;
    const uint64_t qs_bytes = rows * input_per_group;
    const uint64_t scale_bytes = rows * (input_per_group / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)groups * input_per_group * sizeof(float);
    const uint64_t out_bytes = rows * sizeof(float);
    if (qs_bytes > weight_qs_i8->bytes || scale_bytes > weight_scales_f16->bytes ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        input2_offset > input2->bytes || input_bytes > input2->bytes - input2_offset ||
        input3_offset > input3->bytes || input_bytes > input3->bytes - input3_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset ||
        out2_offset > out2->bytes || out_bytes > out2->bytes - out2_offset ||
        out3_offset > out3->bytes || out_bytes > out3->bytes - out3_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(input0, input0_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input0, input0_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input1, input1_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input2, input2_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out0, out0_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(input3, input3_offset, input_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out1, out1_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out2, out2_offset, out_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, out_bytes, out3, out3_offset, out_bytes) ||
        axiom_ranges_overlap(out2, out2_offset, out_bytes, out3, out3_offset, out_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_soa_grouped_matvec4_f32_device(
            runtime->backend_runtime,
            weight_qs_i8->backend_buffer,
            weight_scales_f16->backend_buffer,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            input2->backend_buffer,
            input2_offset,
            input3->backend_buffer,
            input3_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            out2->backend_buffer,
            out2_offset,
            out3->backend_buffer,
            out3_offset,
            groups,
            rows_per_group,
            input_per_group);
}

int axiom_runtime_q2_k_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q2k,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 256u) * 84u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_q2k, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_q2_k_matvec_f32_device(
            runtime->backend_runtime,
            weight_q2k->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

int axiom_runtime_q4_k_matvec_q8k_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_q4k,
        uint64_t weight_offset,
        const axiom_device_buffer *input_q8k,
        uint64_t input_q8k_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t blocks = (uint64_t)(cols / 256u);
    const uint64_t weight_bytes = (uint64_t)rows * blocks * 144u;
    const uint64_t input_bytes = blocks * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_q4k, weight_offset, input_q8k, input_q8k_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_q4_k_matvec_q8k_device(
            runtime->backend_runtime,
            weight_q4k->backend_buffer,
            weight_offset,
            input_q8k->backend_buffer,
            input_q8k_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

int axiom_runtime_iq2_xxs_matvec_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_iq2xxs,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 256u) * 66u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_iq2xxs, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_iq2_xxs_matvec_f32_device(
            runtime->backend_runtime,
            weight_iq2xxs->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

int axiom_runtime_iq2_xxs_matvec_f32_warp_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *weight_iq2xxs,
        uint64_t weight_offset,
        const axiom_device_buffer *input,
        uint64_t input_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t rows,
        uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256u) != 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t weight_bytes = (uint64_t)rows * (cols / 256u) * 66u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    const int rc = axiom_check_device_matvec_args(
            runtime, weight_iq2xxs, weight_offset, input, input_offset, out, out_offset,
            weight_bytes, input_bytes, out_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_iq2_xxs_matvec_f32_warp_device(
            runtime->backend_runtime,
            weight_iq2xxs->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            rows,
            cols);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_iq2xxs || !up_iq2xxs || !down_q2k || !input ||
        !router_weights || !out || experts == 0 || experts > 16 ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_iq2xxs->runtime != runtime || up_iq2xxs->runtime != runtime ||
        down_q2k->runtime != runtime || input->runtime != runtime ||
        router_weights->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t router_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_iq2xxs->bytes || gate_bytes > gate_iq2xxs->bytes - gate_offset ||
        up_offset > up_iq2xxs->bytes || up_bytes > up_iq2xxs->bytes - up_offset ||
        down_offset > down_q2k->bytes || down_bytes > down_q2k->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        router_weights_offset > router_weights->bytes ||
        router_bytes > router_weights->bytes - router_weights_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_topk_f32_device(
            runtime->backend_runtime,
            gate_iq2xxs->backend_buffer,
            gate_offset,
            up_iq2xxs->backend_buffer,
            up_offset,
            down_q2k->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            out->backend_buffer,
            out_offset,
            experts,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !down_all_q2k || !input ||
        !indices || !router_weights || !out || experts == 0 || topk == 0 ||
        topk > 16 || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        down_all_q2k->runtime != runtime || input->runtime != runtime ||
        indices->runtime != runtime || router_weights->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        down_offset > down_all_q2k->bytes || down_bytes > down_all_q2k->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        router_weights_offset > router_weights->bytes ||
        router_bytes > router_weights->bytes - router_weights_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_indexed_f32_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer,
            gate_offset,
            up_all_iq2xxs->backend_buffer,
            up_offset,
            down_all_q2k->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            out->backend_buffer,
            out_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !down_all_q2k || !input ||
        !indices || !router_weights || !scratch_mid || !out || experts == 0 || topk == 0 ||
        topk > 16 || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        down_all_q2k->runtime != runtime || input->runtime != runtime ||
        indices->runtime != runtime || router_weights->runtime != runtime ||
        scratch_mid->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        down_offset > down_all_q2k->bytes || down_bytes > down_all_q2k->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        router_weights_offset > router_weights->bytes ||
        router_bytes > router_weights->bytes - router_weights_offset ||
        scratch_mid_offset > scratch_mid->bytes || scratch_bytes > scratch_mid->bytes - scratch_mid_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_indexed_f32_scratch_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer,
            gate_offset,
            up_all_iq2xxs->backend_buffer,
            up_offset,
            down_all_q2k->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            scratch_mid->backend_buffer,
            scratch_mid_offset,
            out->backend_buffer,
            out_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !input ||
        !indices || !out_mid || experts == 0 || topk == 0 || topk > 16 ||
        topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        input->runtime != runtime || indices->runtime != runtime ||
        out_mid->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t mid_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        out_mid_offset > out_mid->bytes || mid_bytes > out_mid->bytes - out_mid_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_gate_up_indexed_f32_scratch_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer,
            gate_offset,
            up_all_iq2xxs->backend_buffer,
            up_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            out_mid->backend_buffer,
            out_mid_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !input ||
        !indices || !scratch_q8k || !out_mid || experts == 0 || topk == 0 ||
        topk > 16 || topk > experts || hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        input->runtime != runtime || indices->runtime != runtime ||
        scratch_q8k->runtime != runtime || out_mid->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t q8k_bytes = (uint64_t)(hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t mid_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        scratch_q8k_offset > scratch_q8k->bytes || q8k_bytes > scratch_q8k->bytes - scratch_q8k_offset ||
        out_mid_offset > out_mid->bytes || mid_bytes > out_mid->bytes - out_mid_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_gate_up_indexed_q8k_scratch_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer,
            gate_offset,
            up_all_iq2xxs->backend_buffer,
            up_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            scratch_q8k->backend_buffer,
            scratch_q8k_offset,
            out_mid->backend_buffer,
            out_mid_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !down_all_q2k || !mid || !indices || !router_weights || !out ||
        experts == 0 || topk == 0 || topk > 16 || topk > experts ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (down_all_q2k->runtime != runtime || mid->runtime != runtime ||
        indices->runtime != runtime || router_weights->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes);
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t mid_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (down_offset > down_all_q2k->bytes || down_bytes > down_all_q2k->bytes - down_offset ||
        mid_offset > mid->bytes || mid_bytes > mid->bytes - mid_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        router_weights_offset > router_weights->bytes ||
        router_bytes > router_weights->bytes - router_weights_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_down_indexed_f32_device(
            runtime->backend_runtime,
            down_all_q2k->backend_buffer,
            down_offset,
            mid->backend_buffer,
            mid_offset,
            indices->backend_buffer,
            indices_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            out->backend_buffer,
            out_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !down_all_q2k || !input ||
        !indices || !router_weights || !scratch_xq || !scratch_midq || !scratch_mid || !out ||
        experts == 0 || topk != 6u || topk > experts ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        down_all_q2k->runtime != runtime || input->runtime != runtime ||
        indices->runtime != runtime || router_weights->runtime != runtime ||
        scratch_xq->runtime != runtime || scratch_midq->runtime != runtime ||
        scratch_mid->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)(hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t midq_bytes = (uint64_t)topk * (expert_hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t mid_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        down_offset > down_all_q2k->bytes || down_bytes > down_all_q2k->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        router_weights_offset > router_weights->bytes || router_bytes > router_weights->bytes - router_weights_offset ||
        scratch_xq_offset > scratch_xq->bytes || xq_bytes > scratch_xq->bytes - scratch_xq_offset ||
        scratch_midq_offset > scratch_midq->bytes || midq_bytes > scratch_midq->bytes - scratch_midq_offset ||
        scratch_mid_offset > scratch_mid->bytes || mid_bytes > scratch_mid->bytes - scratch_mid_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_q8k_full_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer,
            gate_offset,
            up_all_iq2xxs->backend_buffer,
            up_offset,
            down_all_q2k->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            scratch_xq->backend_buffer,
            scratch_xq_offset,
            scratch_midq->backend_buffer,
            scratch_midq_offset,
            scratch_mid->backend_buffer,
            scratch_mid_offset,
            out->backend_buffer,
            out_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_iq2xxs || !up_all_iq2xxs || !down_all_q2k ||
        !input0 || !indices0 || !router_weights0 || !scratch_xq0 || !scratch_midq0 || !scratch_mid0 || !out0 ||
        !input1 || !indices1 || !router_weights1 || !scratch_xq1 || !scratch_midq1 || !scratch_mid1 || !out1 ||
        experts == 0 || topk != 6u || topk > experts ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_iq2xxs->runtime != runtime || up_all_iq2xxs->runtime != runtime ||
        down_all_q2k->runtime != runtime ||
        input0->runtime != runtime || indices0->runtime != runtime || router_weights0->runtime != runtime ||
        scratch_xq0->runtime != runtime || scratch_midq0->runtime != runtime ||
        scratch_mid0->runtime != runtime || out0->runtime != runtime ||
        input1->runtime != runtime || indices1->runtime != runtime || router_weights1->runtime != runtime ||
        scratch_xq1->runtime != runtime || scratch_midq1->runtime != runtime ||
        scratch_mid1->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_iq2_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q2_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)(hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t midq_bytes = (uint64_t)topk * (expert_hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t mid_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_all_iq2xxs->bytes || gate_bytes > gate_all_iq2xxs->bytes - gate_offset ||
        up_offset > up_all_iq2xxs->bytes || up_bytes > up_all_iq2xxs->bytes - up_offset ||
        down_offset > down_all_q2k->bytes || down_bytes > down_all_q2k->bytes - down_offset ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        indices0_offset > indices0->bytes || indices_bytes > indices0->bytes - indices0_offset ||
        router_weights0_offset > router_weights0->bytes || router_bytes > router_weights0->bytes - router_weights0_offset ||
        scratch_xq0_offset > scratch_xq0->bytes || xq_bytes > scratch_xq0->bytes - scratch_xq0_offset ||
        scratch_midq0_offset > scratch_midq0->bytes || midq_bytes > scratch_midq0->bytes - scratch_midq0_offset ||
        scratch_mid0_offset > scratch_mid0->bytes || mid_bytes > scratch_mid0->bytes - scratch_mid0_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        indices1_offset > indices1->bytes || indices_bytes > indices1->bytes - indices1_offset ||
        router_weights1_offset > router_weights1->bytes || router_bytes > router_weights1->bytes - router_weights1_offset ||
        scratch_xq1_offset > scratch_xq1->bytes || xq_bytes > scratch_xq1->bytes - scratch_xq1_offset ||
        scratch_midq1_offset > scratch_midq1->bytes || midq_bytes > scratch_midq1->bytes - scratch_midq1_offset ||
        scratch_mid1_offset > scratch_mid1->bytes || mid_bytes > scratch_mid1->bytes - scratch_mid1_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_q8k_full2_device(
            runtime->backend_runtime,
            gate_all_iq2xxs->backend_buffer, gate_offset,
            up_all_iq2xxs->backend_buffer, up_offset,
            down_all_q2k->backend_buffer, down_offset,
            input0->backend_buffer, input0_offset,
            indices0->backend_buffer, indices0_offset,
            router_weights0->backend_buffer, router_weights0_offset,
            scratch_xq0->backend_buffer, scratch_xq0_offset,
            scratch_midq0->backend_buffer, scratch_midq0_offset,
            scratch_mid0->backend_buffer, scratch_mid0_offset,
            out0->backend_buffer, out0_offset,
            input1->backend_buffer, input1_offset,
            indices1->backend_buffer, indices1_offset,
            router_weights1->backend_buffer, router_weights1_offset,
            scratch_xq1->backend_buffer, scratch_xq1_offset,
            scratch_midq1->backend_buffer, scratch_midq1_offset,
            scratch_mid1->backend_buffer, scratch_mid1_offset,
            out1->backend_buffer, out1_offset,
            experts, topk, hidden, expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!input || !input_offset || !indices || !indices_offset ||
        !router_weights || !router_weights_offset || !scratch_xq ||
        !scratch_xq_offset || !scratch_midq || !scratch_midq_offset ||
        !scratch_mid || !scratch_mid_offset || !out || !out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < 4u; i++) {
        if (!input[i] || !indices[i] || !router_weights[i] ||
            !scratch_xq[i] || !scratch_midq[i] || !scratch_mid[i] || !out[i]) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    int rc = axiom_runtime_deepseek_moe_q8k_full2_device(
            runtime,
            gate_all_iq2xxs, gate_offset,
            up_all_iq2xxs, up_offset,
            down_all_q2k, down_offset,
            input[0], input_offset[0],
            indices[0], indices_offset[0],
            router_weights[0], router_weights_offset[0],
            scratch_xq[0], scratch_xq_offset[0],
            scratch_midq[0], scratch_midq_offset[0],
            scratch_mid[0], scratch_mid_offset[0],
            out[0], out_offset[0],
            input[1], input_offset[1],
            indices[1], indices_offset[1],
            router_weights[1], router_weights_offset[1],
            scratch_xq[1], scratch_xq_offset[1],
            scratch_midq[1], scratch_midq_offset[1],
            scratch_mid[1], scratch_mid_offset[1],
            out[1], out_offset[1],
            experts, topk, hidden, expert_hidden);
    if (rc != AXIOM_OK) return rc;
    return axiom_runtime_deepseek_moe_q8k_full2_device(
            runtime,
            gate_all_iq2xxs, gate_offset,
            up_all_iq2xxs, up_offset,
            down_all_q2k, down_offset,
            input[2], input_offset[2],
            indices[2], indices_offset[2],
            router_weights[2], router_weights_offset[2],
            scratch_xq[2], scratch_xq_offset[2],
            scratch_midq[2], scratch_midq_offset[2],
            scratch_mid[2], scratch_mid_offset[2],
            out[2], out_offset[2],
            input[3], input_offset[3],
            indices[3], indices_offset[3],
            router_weights[3], router_weights_offset[3],
            scratch_xq[3], scratch_xq_offset[3],
            scratch_midq[3], scratch_midq_offset[3],
            scratch_mid[3], scratch_mid_offset[3],
            out[3], out_offset[3],
            experts, topk, hidden, expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_all_q4k || !up_all_q4k || !down_all_q4k || !input ||
        !indices || !router_weights || !scratch_xq || !scratch_midq || !out ||
        experts == 0 || topk != 6u || topk > experts ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 256u) != 0u || (expert_hidden % 256u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_all_q4k->runtime != runtime || up_all_q4k->runtime != runtime ||
        down_all_q4k->runtime != runtime || input->runtime != runtime ||
        indices->runtime != runtime || router_weights->runtime != runtime ||
        scratch_xq->runtime != runtime || scratch_midq->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gate_bytes = 0;
    uint64_t down_bytes = 0;
    int size_rc = axiom_ds_moe_q4_bytes_checked(experts, expert_hidden, hidden, &gate_bytes);
    size_rc = size_rc == AXIOM_OK ? axiom_ds_moe_q4_bytes_checked(experts, hidden, expert_hidden, &down_bytes) : size_rc;
    if (size_rc != AXIOM_OK) return size_rc;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t xq_bytes = (uint64_t)(hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t midq_bytes = (uint64_t)topk * (expert_hidden / 256u) * AXIOM_Q8_K_BLOCK_BYTES;
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_all_q4k->bytes || gate_bytes > gate_all_q4k->bytes - gate_offset ||
        up_offset > up_all_q4k->bytes || up_bytes > up_all_q4k->bytes - up_offset ||
        down_offset > down_all_q4k->bytes || down_bytes > down_all_q4k->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        router_weights_offset > router_weights->bytes || router_bytes > router_weights->bytes - router_weights_offset ||
        scratch_xq_offset > scratch_xq->bytes || xq_bytes > scratch_xq->bytes - scratch_xq_offset ||
        scratch_midq_offset > scratch_midq->bytes || midq_bytes > scratch_midq->bytes - scratch_midq_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_q4k_full_device(
            runtime->backend_runtime,
            gate_all_q4k->backend_buffer,
            gate_offset,
            up_all_q4k->backend_buffer,
            up_offset,
            down_all_q4k->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            router_weights->backend_buffer,
            router_weights_offset,
            scratch_xq->backend_buffer,
            scratch_xq_offset,
            scratch_midq->backend_buffer,
            scratch_midq_offset,
            out->backend_buffer,
            out_offset,
            experts,
            topk,
            hidden,
            expert_hidden);
}

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
        uint32_t topk) {
    if (!runtime || !router_f16 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (router_f16->runtime != runtime || input->runtime != runtime ||
        out_indices->runtime != runtime || out_weights->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t router_bytes = (uint64_t)experts * hidden * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    if (router_offset > router_f16->bytes || router_bytes > router_f16->bytes - router_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_indices_offset > out_indices->bytes ||
        indices_bytes > out_indices->bytes - out_indices_offset ||
        out_weights_offset > out_weights->bytes ||
        weights_bytes > out_weights->bytes - out_weights_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_router_topk_f16_f32_device(
            runtime->backend_runtime,
            router_f16->backend_buffer,
            router_offset,
            input->backend_buffer,
            input_offset,
            out_indices->backend_buffer,
            out_indices_offset,
            out_weights->backend_buffer,
            out_weights_offset,
            experts,
            hidden,
            topk);
}

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
        uint32_t topk) {
    if (!runtime || !router_f16 || !bias_f32 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (router_f16->runtime != runtime || bias_f32->runtime != runtime ||
        input->runtime != runtime || out_indices->runtime != runtime ||
        out_weights->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t router_bytes = (uint64_t)experts * hidden * sizeof(uint16_t);
    const uint64_t bias_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    if (router_offset > router_f16->bytes || router_bytes > router_f16->bytes - router_offset ||
        bias_offset > bias_f32->bytes || bias_bytes > bias_f32->bytes - bias_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_indices_offset > out_indices->bytes ||
        indices_bytes > out_indices->bytes - out_indices_offset ||
        out_weights_offset > out_weights->bytes ||
        weights_bytes > out_weights->bytes - out_weights_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_router_topk_biased_f16_f32_device(
            runtime->backend_runtime,
            router_f16->backend_buffer,
            router_offset,
            bias_f32->backend_buffer,
            bias_offset,
            input->backend_buffer,
            input_offset,
            out_indices->backend_buffer,
            out_indices_offset,
            out_weights->backend_buffer,
            out_weights_offset,
            experts,
            hidden,
            topk);
}

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
        uint32_t topk) {
    if (!runtime || !router_f32 || !bias_f32 || !input || !out_indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (router_f32->runtime != runtime || bias_f32->runtime != runtime ||
        input->runtime != runtime || out_indices->runtime != runtime ||
        out_weights->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t router_bytes = (uint64_t)experts * hidden * sizeof(float);
    const uint64_t bias_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    if (router_offset > router_f32->bytes || router_bytes > router_f32->bytes - router_offset ||
        bias_offset > bias_f32->bytes || bias_bytes > bias_f32->bytes - bias_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_indices_offset > out_indices->bytes ||
        indices_bytes > out_indices->bytes - out_indices_offset ||
        out_weights_offset > out_weights->bytes ||
        weights_bytes > out_weights->bytes - out_weights_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_router_topk_biased_f32_f32_device(
            runtime->backend_runtime,
            router_f32->backend_buffer,
            router_offset,
            bias_f32->backend_buffer,
            bias_offset,
            input->backend_buffer,
            input_offset,
            out_indices->backend_buffer,
            out_indices_offset,
            out_weights->backend_buffer,
            out_weights_offset,
            experts,
            hidden,
            topk);
}

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
        uint32_t topk) {
    if (!runtime || !router_f16 || !bias_f32 || !input || !scratch_logits ||
        !out_indices || !out_weights || experts == 0 || hidden == 0 ||
        topk == 0 || topk > 16 || topk > experts) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (router_f16->runtime != runtime || bias_f32->runtime != runtime ||
        input->runtime != runtime || scratch_logits->runtime != runtime ||
        out_indices->runtime != runtime || out_weights->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t router_bytes = (uint64_t)experts * hidden * sizeof(uint16_t);
    const uint64_t bias_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    if (router_offset > router_f16->bytes || router_bytes > router_f16->bytes - router_offset ||
        bias_offset > bias_f32->bytes || bias_bytes > bias_f32->bytes - bias_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        scratch_logits_offset > scratch_logits->bytes ||
        scratch_bytes > scratch_logits->bytes - scratch_logits_offset ||
        out_indices_offset > out_indices->bytes ||
        indices_bytes > out_indices->bytes - out_indices_offset ||
        out_weights_offset > out_weights->bytes ||
        weights_bytes > out_weights->bytes - out_weights_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_router_topk_biased_f16_f32_scratch_device(
            runtime->backend_runtime,
            router_f16->backend_buffer,
            router_offset,
            bias_f32->backend_buffer,
            bias_offset,
            input->backend_buffer,
            input_offset,
            scratch_logits->backend_buffer,
            scratch_logits_offset,
            out_indices->backend_buffer,
            out_indices_offset,
            out_weights->backend_buffer,
            out_weights_offset,
            experts,
            hidden,
            topk);
}

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
        uint32_t topk) {
    if (!runtime || !router_f16 || !input || !indices || !out_weights ||
        experts == 0 || hidden == 0 || topk == 0 || topk > 16) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (router_f16->runtime != runtime || input->runtime != runtime ||
        indices->runtime != runtime || out_weights->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t router_bytes = (uint64_t)experts * hidden * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    if (router_offset > router_f16->bytes || router_bytes > router_f16->bytes - router_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        indices_offset > indices->bytes || indices_bytes > indices->bytes - indices_offset ||
        out_weights_offset > out_weights->bytes ||
        weights_bytes > out_weights->bytes - out_weights_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_router_hash_f16_f32_device(
            runtime->backend_runtime,
            router_f16->backend_buffer,
            router_offset,
            input->backend_buffer,
            input_offset,
            indices->backend_buffer,
            indices_offset,
            out_weights->backend_buffer,
            out_weights_offset,
            experts,
            hidden,
            topk);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_q8 || !up_q8 || !down_q8 || !input || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_q8->runtime != runtime || up_q8->runtime != runtime ||
        down_q8->runtime != runtime || input->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    const uint64_t gate_bytes = (uint64_t)expert_hidden * (hidden / 32u) * 34u;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t down_bytes = (uint64_t)hidden * (expert_hidden / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_q8->bytes || gate_bytes > gate_q8->bytes - gate_offset ||
        up_offset > up_q8->bytes || up_bytes > up_q8->bytes - up_offset ||
        down_offset > down_q8->bytes || down_bytes > down_q8->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_shared_expert_q8_f32_device(
            runtime->backend_runtime,
            gate_q8->backend_buffer,
            gate_offset,
            up_q8->backend_buffer,
            up_offset,
            down_q8->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_q8 || !up_q8 || !down_q8 || !input || !scratch_mid || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_q8->runtime != runtime || up_q8->runtime != runtime ||
        down_q8->runtime != runtime || input->runtime != runtime ||
        scratch_mid->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    const uint64_t gate_bytes = (uint64_t)expert_hidden * (hidden / 32u) * 34u;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t down_bytes = (uint64_t)hidden * (expert_hidden / 32u) * 34u;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_offset > gate_q8->bytes || gate_bytes > gate_q8->bytes - gate_offset ||
        up_offset > up_q8->bytes || up_bytes > up_q8->bytes - up_offset ||
        down_offset > down_q8->bytes || down_bytes > down_q8->bytes - down_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        scratch_mid_offset > scratch_mid->bytes || scratch_bytes > scratch_mid->bytes - scratch_mid_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_shared_expert_q8_f32_scratch_device(
            runtime->backend_runtime,
            gate_q8->backend_buffer,
            gate_offset,
            up_q8->backend_buffer,
            up_offset,
            down_q8->backend_buffer,
            down_offset,
            input->backend_buffer,
            input_offset,
            scratch_mid->backend_buffer,
            scratch_mid_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            expert_hidden);
}

/* Native FP8 shared-expert FFN: out = down_fp8 . silu_mul(gate_fp8.x, up_fp8.x).
 * Pure composition of the (validated) FP8 matvec + silu_mul primitives — each callee
 * re-validates args/bounds/backend, so this stays a thin orchestrator. */
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
        uint32_t hidden, uint32_t expert_hidden) {
    if (!runtime || hidden == 0 || expert_hidden == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    /* gate = gate_fp8 . x   (rows=expert_hidden, cols=hidden) */
    int rc = axiom_runtime_fp8_e4m3_e8m0_matvec_f32_device(
            runtime, gate_w, gate_w_offset, gate_scale, gate_scale_offset,
            input, input_offset, scratch_gate, scratch_gate_offset,
            expert_hidden, hidden);
    /* up = up_fp8 . x */
    rc = rc == AXIOM_OK ? axiom_runtime_fp8_e4m3_e8m0_matvec_f32_device(
            runtime, up_w, up_w_offset, up_scale, up_scale_offset,
            input, input_offset, scratch_up, scratch_up_offset,
            expert_hidden, hidden) : rc;
    /* mid = silu(gate) * up   (in place into scratch_gate) */
    rc = rc == AXIOM_OK ? axiom_runtime_silu_mul_f32_device(
            runtime, scratch_gate, scratch_gate_offset, scratch_up, scratch_up_offset,
            scratch_gate, scratch_gate_offset, expert_hidden) : rc;
    /* out = down_fp8 . mid   (rows=hidden, cols=expert_hidden) */
    rc = rc == AXIOM_OK ? axiom_runtime_fp8_e4m3_e8m0_matvec_f32_device(
            runtime, down_w, down_w_offset, down_scale, down_scale_offset,
            scratch_gate, scratch_gate_offset, out, out_offset,
            hidden, expert_hidden) : rc;
    return rc;
}

int axiom_runtime_axpby_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *out, uint64_t out_offset,
        const axiom_device_buffer *in, uint64_t in_offset,
        float alpha, float beta, uint32_t n) {
    if (!runtime || !out || !in || n == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (out->runtime != runtime || in->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    if (out_offset > out->bytes || bytes > out->bytes - out_offset ||
        in_offset > in->bytes || bytes > in->bytes - in_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_axpby_f32_device(
            runtime->backend_runtime, out->backend_buffer, out_offset,
            in->backend_buffer, in_offset, alpha, beta, n);
}

/* Native NVFP4 routed-expert MoE (single token) — correctness-first host-orchestrated form.
 * Downloads the topk router indices/weights, then for each selected expert e composes
 *   out += weight[j] * ( down_e . silu_mul(gate_e . x, up_e . x) )
 * via the proven per-expert NVFP4 matvec + silu_mul + axpby. Weight/scale planes are the
 * de-blocked combined tensors (expert e at e*stride); gscale is the per-expert f32 global.
 * gate/up: rows=expert_hidden, cols=hidden; down: rows=hidden, cols=expert_hidden.
 * This favors correctness (bit-faithful to the validated kernels) over speed; a fused
 * indexed kernel can replace it later without changing the caller. */
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
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden) {
    if (!runtime || !gate_w || !gate_bs || !gate_gs || !up_w || !up_bs || !up_gs ||
        !down_w || !down_bs || !down_gs || !input || !router_indices || !router_weights ||
        !scratch_gate || !scratch_up || !scratch_expert_out || !out ||
        experts == 0u || topk == 0u || topk > 64u || hidden == 0u || expert_hidden == 0u ||
        (hidden % 16u) != 0u || (expert_hidden % 16u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;

    uint32_t idx[64];
    float wts[64];
    int rc = axiom_device_buffer_download((axiom_device_buffer *)router_indices,
            router_indices_offset, idx, (uint64_t)topk * sizeof(uint32_t));
    rc = rc == AXIOM_OK ? axiom_device_buffer_download((axiom_device_buffer *)router_weights,
            router_weights_offset, wts, (uint64_t)topk * sizeof(float)) : rc;
    if (rc != AXIOM_OK) return rc;

    float *ggs = (float *)malloc((size_t)experts * sizeof(float));
    float *ugs = (float *)malloc((size_t)experts * sizeof(float));
    float *dgs = (float *)malloc((size_t)experts * sizeof(float));
    if (!ggs || !ugs || !dgs) { free(ggs); free(ugs); free(dgs); return AXIOM_ERR_RUNTIME; }
    rc = axiom_device_buffer_download((axiom_device_buffer *)gate_gs, 0, ggs, (uint64_t)experts * sizeof(float));
    rc = rc == AXIOM_OK ? axiom_device_buffer_download((axiom_device_buffer *)up_gs, 0, ugs, (uint64_t)experts * sizeof(float)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download((axiom_device_buffer *)down_gs, 0, dgs, (uint64_t)experts * sizeof(float)) : rc;
    if (rc != AXIOM_OK) { free(ggs); free(ugs); free(dgs); return rc; }

    const uint64_t gu_w_stride  = (uint64_t)expert_hidden * (hidden / 2u);
    const uint64_t gu_bs_stride = (uint64_t)expert_hidden * (hidden / 16u);
    const uint64_t dn_w_stride  = (uint64_t)hidden * (expert_hidden / 2u);
    const uint64_t dn_bs_stride = (uint64_t)hidden * (expert_hidden / 16u);

    for (uint32_t j = 0; j < topk && rc == AXIOM_OK; ++j) {
        const uint32_t e = idx[j];
        if (e >= experts) { rc = AXIOM_ERR_INVALID_ARGUMENT; break; }
        const float wj = wts[j];
        /* gate_e . x -> scratch_gate  (rows=expert_hidden, cols=hidden) */
        rc = axiom_runtime_e2m1_nvfp4_matvec_f32_device(
                runtime, gate_w, e * gu_w_stride, gate_bs, e * gu_bs_stride, ggs[e],
                input, input_offset, scratch_gate, 0, expert_hidden, hidden);
        /* up_e . x -> scratch_up */
        rc = rc == AXIOM_OK ? axiom_runtime_e2m1_nvfp4_matvec_f32_device(
                runtime, up_w, e * gu_w_stride, up_bs, e * gu_bs_stride, ugs[e],
                input, input_offset, scratch_up, 0, expert_hidden, hidden) : rc;
        /* silu(gate) * up -> scratch_gate */
        rc = rc == AXIOM_OK ? axiom_runtime_silu_mul_f32_device(
                runtime, scratch_gate, 0, scratch_up, 0, scratch_gate, 0, expert_hidden) : rc;
        /* down_e . mid -> scratch_expert_out  (rows=hidden, cols=expert_hidden) */
        rc = rc == AXIOM_OK ? axiom_runtime_e2m1_nvfp4_matvec_f32_device(
                runtime, down_w, e * dn_w_stride, down_bs, e * dn_bs_stride, dgs[e],
                scratch_gate, 0, scratch_expert_out, 0, hidden, expert_hidden) : rc;
        /* out = (j==0 ? 0 : 1)*out + wj*expert_out */
        rc = rc == AXIOM_OK ? axiom_runtime_axpby_f32_device(
                runtime, out, out_offset, scratch_expert_out, 0,
                wj, (j == 0u) ? 0.0f : 1.0f, hidden) : rc;
    }
    free(ggs); free(ugs); free(dgs);
    return rc;
}

/* NVFP4 plane bytes = experts * rows * (cols/div), overflow-checked (house idiom of
 * axiom_ds_moe_iq2_bytes_checked). div = 2 for the nibble plane, 16 for the group-16
 * block-scale plane. */
static int axiom_ds_moe_nvfp4_plane_bytes_checked(
        uint32_t experts, uint32_t rows, uint32_t cols, uint32_t div, uint64_t *out) {
    uint64_t bytes = 0;
    int rc = axiom_mul_u64_checked(experts, rows, &bytes);
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked(bytes, cols / div, &bytes) : rc;
    if (rc == AXIOM_OK) *out = bytes;
    return rc;
}

/* Native NVFP4 routed-expert MoE (single token) — FUSED, device-resident form.
 * Same argument surface as the host-loop entry above so the forward switches 1:1;
 * the host loop STAYS as the oracle. Router indices/weights and the per-expert f32
 * global scales are consumed ON DEVICE (no D2H, no per-expert gscale re-download),
 * and the whole MoE runs as two kernel launches with no cudaDeviceSynchronize
 * (src/axiom_cuda_nvfp4_moe.cu; completion follows the finish-after-launch
 * convention). Semantics deltas vs the host loop — deliberate, documented in
 * include/axiom/axiom.h and docs/axiom_nvfp4_fused_moe_design.md:
 *   - scratch_gate is the [topk, expert_hidden] f32 mid plane (>= topk*expert_hidden*4
 *     bytes; the DS4 forward passes moe_mid); scratch_up/scratch_expert_out are only
 *     surface parity (validated, untouched);
 *   - hidden/expert_hidden must be %32 (uint4 nibble loads), not just %16;
 *   - out is overwritten with the full sum (same net effect as the axpby chain);
 *   - an out-of-range router index contributes zero instead of erroring (indices are
 *     never downloaded, so there is no sync point at which to reject them);
 *   - NOT bit-exact vs the host loop (different fp32 accumulation order); the parity
 *     bar is relative error, enforced by tests/axiom_nvfp4_smoke.cpp (fused section).
 * Validation follows the full axiom_runtime_deepseek_moe_indexed_f32_scratch_device
 * idiom: arg/backend/ownership checks + overflow-checked byte spans for every buffer. */
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
        uint32_t experts, uint32_t topk, uint32_t hidden, uint32_t expert_hidden) {
    if (!runtime || !gate_w || !gate_bs || !gate_gs || !up_w || !up_bs || !up_gs ||
        !down_w || !down_bs || !down_gs || !input || !router_indices || !router_weights ||
        !scratch_gate || !scratch_up || !scratch_expert_out || !out ||
        experts == 0u || experts > 65536u || topk == 0u || topk > 64u ||
        hidden == 0u || expert_hidden == 0u ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_w->runtime != runtime || gate_bs->runtime != runtime || gate_gs->runtime != runtime ||
        up_w->runtime != runtime || up_bs->runtime != runtime || up_gs->runtime != runtime ||
        down_w->runtime != runtime || down_bs->runtime != runtime || down_gs->runtime != runtime ||
        input->runtime != runtime || router_indices->runtime != runtime ||
        router_weights->runtime != runtime || scratch_gate->runtime != runtime ||
        scratch_up->runtime != runtime || scratch_expert_out->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    uint64_t gu_w_bytes = 0, gu_bs_bytes = 0, dn_w_bytes = 0, dn_bs_bytes = 0, mid_bytes = 0;
    int rc = axiom_ds_moe_nvfp4_plane_bytes_checked(experts, expert_hidden, hidden, 2u, &gu_w_bytes);
    rc = rc == AXIOM_OK ? axiom_ds_moe_nvfp4_plane_bytes_checked(experts, expert_hidden, hidden, 16u, &gu_bs_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_ds_moe_nvfp4_plane_bytes_checked(experts, hidden, expert_hidden, 2u, &dn_w_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_ds_moe_nvfp4_plane_bytes_checked(experts, hidden, expert_hidden, 16u, &dn_bs_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_mul_u64_checked((uint64_t)topk * expert_hidden, sizeof(float), &mid_bytes) : rc;
    if (rc != AXIOM_OK) return rc;
    const uint64_t gs_bytes = (uint64_t)experts * sizeof(float);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t router_bytes = (uint64_t)topk * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_w->bytes < gu_w_bytes || gate_bs->bytes < gu_bs_bytes || gate_gs->bytes < gs_bytes ||
        up_w->bytes < gu_w_bytes || up_bs->bytes < gu_bs_bytes || up_gs->bytes < gs_bytes ||
        down_w->bytes < dn_w_bytes || down_bs->bytes < dn_bs_bytes || down_gs->bytes < gs_bytes ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        router_indices_offset > router_indices->bytes ||
        indices_bytes > router_indices->bytes - router_indices_offset ||
        router_weights_offset > router_weights->bytes ||
        router_bytes > router_weights->bytes - router_weights_offset ||
        scratch_gate->bytes < mid_bytes ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32_device(
            runtime->backend_runtime,
            gate_w->backend_buffer, gate_bs->backend_buffer, gate_gs->backend_buffer,
            up_w->backend_buffer, up_bs->backend_buffer, up_gs->backend_buffer,
            down_w->backend_buffer, down_bs->backend_buffer, down_gs->backend_buffer,
            input->backend_buffer, input_offset,
            router_indices->backend_buffer, router_indices_offset,
            router_weights->backend_buffer, router_weights_offset,
            scratch_gate->backend_buffer, 0,
            out->backend_buffer, out_offset,
            /*stream=*/NULL,
            experts, topk, hidden, expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_qs_i8 || !gate_scales_f16 || !up_qs_i8 || !up_scales_f16 ||
        !down_qs_i8 || !down_scales_f16 || !input || !scratch_mid || !out ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_qs_i8->runtime != runtime || gate_scales_f16->runtime != runtime ||
        up_qs_i8->runtime != runtime || up_scales_f16->runtime != runtime ||
        down_qs_i8->runtime != runtime || down_scales_f16->runtime != runtime ||
        input->runtime != runtime || scratch_mid->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    const uint64_t gate_qs_bytes = (uint64_t)expert_hidden * hidden;
    const uint64_t gate_scale_bytes = (uint64_t)expert_hidden * (hidden / 32u) * sizeof(uint16_t);
    const uint64_t up_qs_bytes = gate_qs_bytes;
    const uint64_t up_scale_bytes = gate_scale_bytes;
    const uint64_t down_qs_bytes = (uint64_t)hidden * expert_hidden;
    const uint64_t down_scale_bytes = (uint64_t)hidden * (expert_hidden / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_qs_i8->bytes < gate_qs_bytes || gate_scales_f16->bytes < gate_scale_bytes ||
        up_qs_i8->bytes < up_qs_bytes || up_scales_f16->bytes < up_scale_bytes ||
        down_qs_i8->bytes < down_qs_bytes || down_scales_f16->bytes < down_scale_bytes ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        scratch_mid_offset > scratch_mid->bytes || scratch_bytes > scratch_mid->bytes - scratch_mid_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch_device(
            runtime->backend_runtime,
            gate_qs_i8->backend_buffer,
            gate_scales_f16->backend_buffer,
            up_qs_i8->backend_buffer,
            up_scales_f16->backend_buffer,
            down_qs_i8->backend_buffer,
            down_scales_f16->backend_buffer,
            input->backend_buffer,
            input_offset,
            scratch_mid->backend_buffer,
            scratch_mid_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_qs_i8 || !gate_scales_f16 || !up_qs_i8 || !up_scales_f16 ||
        !down_qs_i8 || !down_scales_f16 || !input0 || !input1 ||
        !scratch_mid0 || !scratch_mid1 || !out0 || !out1 ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_qs_i8->runtime != runtime || gate_scales_f16->runtime != runtime ||
        up_qs_i8->runtime != runtime || up_scales_f16->runtime != runtime ||
        down_qs_i8->runtime != runtime || down_scales_f16->runtime != runtime ||
        input0->runtime != runtime || input1->runtime != runtime ||
        scratch_mid0->runtime != runtime || scratch_mid1->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    const uint64_t gate_qs_bytes = (uint64_t)expert_hidden * hidden;
    const uint64_t gate_scale_bytes = (uint64_t)expert_hidden * (hidden / 32u) * sizeof(uint16_t);
    const uint64_t up_qs_bytes = gate_qs_bytes;
    const uint64_t up_scale_bytes = gate_scale_bytes;
    const uint64_t down_qs_bytes = (uint64_t)hidden * expert_hidden;
    const uint64_t down_scale_bytes = (uint64_t)hidden * (expert_hidden / 32u) * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t scratch_bytes = (uint64_t)expert_hidden * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (gate_qs_i8->bytes < gate_qs_bytes || gate_scales_f16->bytes < gate_scale_bytes ||
        up_qs_i8->bytes < up_qs_bytes || up_scales_f16->bytes < up_scale_bytes ||
        down_qs_i8->bytes < down_qs_bytes || down_scales_f16->bytes < down_scale_bytes ||
        input0_offset > input0->bytes || input_bytes > input0->bytes - input0_offset ||
        input1_offset > input1->bytes || input_bytes > input1->bytes - input1_offset ||
        scratch_mid0_offset > scratch_mid0->bytes ||
        scratch_bytes > scratch_mid0->bytes - scratch_mid0_offset ||
        scratch_mid1_offset > scratch_mid1->bytes ||
        scratch_bytes > scratch_mid1->bytes - scratch_mid1_offset ||
        out0_offset > out0->bytes || out_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || out_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    return axiom_cuda_deepseek_shared_expert_q8_soa_f32_scratch2_device(
            runtime->backend_runtime,
            gate_qs_i8->backend_buffer,
            gate_scales_f16->backend_buffer,
            up_qs_i8->backend_buffer,
            up_scales_f16->backend_buffer,
            down_qs_i8->backend_buffer,
            down_scales_f16->backend_buffer,
            input0->backend_buffer,
            input0_offset,
            input1->backend_buffer,
            input1_offset,
            scratch_mid0->backend_buffer,
            scratch_mid0_offset,
            scratch_mid1->backend_buffer,
            scratch_mid1_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            hidden,
            expert_hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !gate_q8 || !up_q8 || !input || !mid ||
        hidden == 0 || expert_hidden == 0 ||
        (hidden % 32u) != 0u || (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (gate_q8->runtime != runtime || up_q8->runtime != runtime ||
        input->runtime != runtime || mid->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t gate_bytes = (uint64_t)expert_hidden * (hidden / 32u) * 34u;
    const uint64_t up_bytes = gate_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)expert_hidden * sizeof(float);
    if (gate_offset > gate_q8->bytes || gate_bytes > gate_q8->bytes - gate_offset ||
        up_offset > up_q8->bytes || up_bytes > up_q8->bytes - up_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        mid_offset > mid->bytes || mid_bytes > mid->bytes - mid_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_shared_gate_up_q8_f32_device(
            runtime->backend_runtime,
            gate_q8->backend_buffer,
            gate_offset,
            up_q8->backend_buffer,
            up_offset,
            input->backend_buffer,
            input_offset,
            mid->backend_buffer,
            mid_offset,
            hidden,
            expert_hidden);
}

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
        float eps) {
    if (!runtime || !q || !kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || kv->runtime != runtime ||
        attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    const uint64_t out_bytes = q_bytes;
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        kv_offset > kv->bytes || kv_bytes > kv->bytes - kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_sliding_attention_single_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            kv->backend_buffer,
            kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            eps);
}

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
        float eps) {
    if (!runtime || !q || !current_kv || !ring_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_slots > 128u ||
        ring_head >= ring_slots || ring_count > ring_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || current_kv->runtime != runtime ||
        ring_kv->runtime != runtime || attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t current_kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t ring_bytes = (uint64_t)ring_slots * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    const uint64_t out_bytes = q_bytes;
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        current_kv_offset > current_kv->bytes || current_kv_bytes > current_kv->bytes - current_kv_offset ||
        ring_kv_offset > ring_kv->bytes || ring_bytes > ring_kv->bytes - ring_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_sliding_attention_ring_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            current_kv->backend_buffer,
            current_kv_offset,
            ring_kv->backend_buffer,
            ring_kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            ring_slots,
            ring_head,
            ring_count,
            eps);
}

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
        float eps) {
    if (!runtime || !q0 || !q1 || !kv0 || !kv1 || !ring_kv || !attn_sink || !out0 || !out1 ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_head >= ring_slots ||
        ring_count > ring_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q0->runtime != runtime || q1->runtime != runtime ||
        kv0->runtime != runtime || kv1->runtime != runtime ||
        ring_kv->runtime != runtime || attn_sink->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t ring_bytes = (uint64_t)ring_slots * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q0_offset > q0->bytes || q_bytes > q0->bytes - q0_offset ||
        q1_offset > q1->bytes || q_bytes > q1->bytes - q1_offset ||
        kv0_offset > kv0->bytes || kv_bytes > kv0->bytes - kv0_offset ||
        kv1_offset > kv1->bytes || kv_bytes > kv1->bytes - kv1_offset ||
        ring_kv_offset > ring_kv->bytes || ring_bytes > ring_kv->bytes - ring_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out0_offset > out0->bytes || q_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || q_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_sliding_attention_ring2_causal_f32_device(
            runtime->backend_runtime,
            q0->backend_buffer,
            q0_offset,
            q1->backend_buffer,
            q1_offset,
            kv0->backend_buffer,
            kv0_offset,
            kv1->backend_buffer,
            kv1_offset,
            ring_kv->backend_buffer,
            ring_kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            heads,
            head_dim,
            ring_slots,
            ring_head,
            ring_count,
            eps);
}

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
        float eps) {
    if (!runtime || !q0 || !q1 || !q2 || !q3 ||
        !kv0 || !kv1 || !kv2 || !kv3 || !ring_kv || !attn_sink ||
        !out0 || !out1 || !out2 || !out3 ||
        heads == 0 || head_dim == 0 || ring_slots == 0 || ring_head >= ring_slots ||
        ring_count > ring_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q0->runtime != runtime || q1->runtime != runtime ||
        q2->runtime != runtime || q3->runtime != runtime ||
        kv0->runtime != runtime || kv1->runtime != runtime ||
        kv2->runtime != runtime || kv3->runtime != runtime ||
        ring_kv->runtime != runtime || attn_sink->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime ||
        out2->runtime != runtime || out3->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t ring_bytes = (uint64_t)ring_slots * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q0_offset > q0->bytes || q_bytes > q0->bytes - q0_offset ||
        q1_offset > q1->bytes || q_bytes > q1->bytes - q1_offset ||
        q2_offset > q2->bytes || q_bytes > q2->bytes - q2_offset ||
        q3_offset > q3->bytes || q_bytes > q3->bytes - q3_offset ||
        kv0_offset > kv0->bytes || kv_bytes > kv0->bytes - kv0_offset ||
        kv1_offset > kv1->bytes || kv_bytes > kv1->bytes - kv1_offset ||
        kv2_offset > kv2->bytes || kv_bytes > kv2->bytes - kv2_offset ||
        kv3_offset > kv3->bytes || kv_bytes > kv3->bytes - kv3_offset ||
        ring_kv_offset > ring_kv->bytes || ring_bytes > ring_kv->bytes - ring_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out0_offset > out0->bytes || q_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || q_bytes > out1->bytes - out1_offset ||
        out2_offset > out2->bytes || q_bytes > out2->bytes - out2_offset ||
        out3_offset > out3->bytes || q_bytes > out3->bytes - out3_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(out0, out0_offset, q_bytes, out1, out1_offset, q_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, q_bytes, out2, out2_offset, q_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, q_bytes, out3, out3_offset, q_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, q_bytes, out2, out2_offset, q_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, q_bytes, out3, out3_offset, q_bytes) ||
        axiom_ranges_overlap(out2, out2_offset, q_bytes, out3, out3_offset, q_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_sliding_attention_ring4_causal_f32_device(
            runtime->backend_runtime,
            q0->backend_buffer, q0_offset,
            q1->backend_buffer, q1_offset,
            q2->backend_buffer, q2_offset,
            q3->backend_buffer, q3_offset,
            kv0->backend_buffer, kv0_offset,
            kv1->backend_buffer, kv1_offset,
            kv2->backend_buffer, kv2_offset,
            kv3->backend_buffer, kv3_offset,
            ring_kv->backend_buffer, ring_kv_offset,
            attn_sink->backend_buffer, attn_sink_offset,
            out0->backend_buffer, out0_offset,
            out1->backend_buffer, out1_offset,
            out2->backend_buffer, out2_offset,
            out3->backend_buffer, out3_offset,
            heads, head_dim, ring_slots, ring_head, ring_count, eps);
}

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
        uint32_t inverse) {
    if (!runtime || !input || !out || heads == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot % 2u) != 0u ||
        freq_base <= 0.0f || freq_scale <= 0.0f || beta_fast <= 0.0f || beta_slow <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (input->runtime != runtime || out->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (input_offset > input->bytes || bytes > input->bytes - input_offset ||
        out_offset > out->bytes || bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_rope_tail_f32_device(
            runtime->backend_runtime,
            input->backend_buffer,
            input_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            n_rot,
            position,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            n_ctx_orig,
            inverse);
}

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
        uint32_t inverse) {
    if (!runtime || !input0 || !out0 || !input1 || !out1 || heads == 0 || head_dim == 0 ||
        n_rot == 0 || n_rot > head_dim || (n_rot % 2u) != 0u ||
        freq_base <= 0.0f || freq_scale <= 0.0f || beta_fast <= 0.0f || beta_slow <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (input0->runtime != runtime || out0->runtime != runtime ||
        input1->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t bytes = (uint64_t)heads * head_dim * sizeof(float);
    if (input0_offset > input0->bytes || bytes > input0->bytes - input0_offset ||
        out0_offset > out0->bytes || bytes > out0->bytes - out0_offset ||
        input1_offset > input1->bytes || bytes > input1->bytes - input1_offset ||
        out1_offset > out1->bytes || bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_rope_tail_dual_f32_device(
            runtime->backend_runtime,
            input0->backend_buffer,
            input0_offset,
            out0->backend_buffer,
            out0_offset,
            position0,
            input1->backend_buffer,
            input1_offset,
            out1->backend_buffer,
            out1_offset,
            position1,
            heads,
            head_dim,
            n_rot,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            n_ctx_orig,
            inverse);
}

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
        float eps) {
    if (!runtime || !q || !kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || kv_count == 0 || kv_count > 32 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || kv->runtime != runtime ||
        attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)kv_count * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    const uint64_t out_bytes = q_bytes;
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        kv_offset > kv->bytes || kv_bytes > kv->bytes - kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_multi_kv_single_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            kv->backend_buffer,
            kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            kv_count,
            eps);
}

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
        float eps) {
    if (!runtime || !q || !current_kv || !history_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || has_history > 1u || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || current_kv->runtime != runtime || history_kv->runtime != runtime ||
        attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        current_kv_offset > current_kv->bytes || kv_bytes > current_kv->bytes - current_kv_offset ||
        history_kv_offset > history_kv->bytes || kv_bytes > history_kv->bytes - history_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || q_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_current_history_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            current_kv->backend_buffer,
            current_kv_offset,
            history_kv->backend_buffer,
            history_kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            has_history,
            eps);
}

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
        float eps) {
    if (!runtime || !q || !raw_kv_ring || !comp_kv || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head >= raw_slots ||
        raw_count == 0 || raw_count > raw_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || raw_kv_ring->runtime != runtime ||
        comp_kv->runtime != runtime || attn_sink->runtime != runtime ||
        out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_slots * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)comp_count * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        raw_kv_ring_offset > raw_kv_ring->bytes || raw_bytes > raw_kv_ring->bytes - raw_kv_ring_offset ||
        comp_kv_offset > comp_kv->bytes || comp_bytes > comp_kv->bytes - comp_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || q_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_raw_comp_ring_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            raw_kv_ring->backend_buffer,
            raw_kv_ring_offset,
            comp_kv->backend_buffer,
            comp_kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            raw_slots,
            raw_head,
            raw_count,
            comp_count,
            eps);
}

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
        float eps) {
    if (!runtime || !q0 || !q1 || !kv0 || !kv1 || !raw_kv_ring || !comp_kv ||
        !attn_sink || !out0 || !out1 || heads == 0 || head_dim == 0 ||
        raw_slots == 0 || raw_head0 >= raw_slots || raw_count0 == 0 ||
        raw_count0 > raw_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q0->runtime != runtime || q1->runtime != runtime ||
        kv0->runtime != runtime || kv1->runtime != runtime ||
        raw_kv_ring->runtime != runtime || comp_kv->runtime != runtime ||
        attn_sink->runtime != runtime || out0->runtime != runtime ||
        out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_slots * head_dim * sizeof(float);
    const uint32_t comp_cap = comp_count0 > comp_count1 ? comp_count0 : comp_count1;
    const uint64_t comp_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q0_offset > q0->bytes || q_bytes > q0->bytes - q0_offset ||
        q1_offset > q1->bytes || q_bytes > q1->bytes - q1_offset ||
        kv0_offset > kv0->bytes || kv_bytes > kv0->bytes - kv0_offset ||
        kv1_offset > kv1->bytes || kv_bytes > kv1->bytes - kv1_offset ||
        raw_kv_ring_offset > raw_kv_ring->bytes || raw_bytes > raw_kv_ring->bytes - raw_kv_ring_offset ||
        comp_kv_offset > comp_kv->bytes || comp_bytes > comp_kv->bytes - comp_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out0_offset > out0->bytes || q_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || q_bytes > out1->bytes - out1_offset ||
        raw_count0 + comp_count0 > 16384u ||
        raw_count1 + comp_count1 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_raw_comp_ring2_causal_f32_device(
            runtime->backend_runtime,
            q0->backend_buffer,
            q0_offset,
            q1->backend_buffer,
            q1_offset,
            kv0->backend_buffer,
            kv0_offset,
            kv1->backend_buffer,
            kv1_offset,
            raw_kv_ring->backend_buffer,
            raw_kv_ring_offset,
            comp_kv->backend_buffer,
            comp_kv_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            heads,
            head_dim,
            raw_slots,
            raw_head0,
            raw_count0,
            comp_count0,
            comp_count1,
            eps);
}

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
        float eps) {
    if (!runtime || !q0 || !q1 || !q2 || !q3 ||
        !kv0 || !kv1 || !kv2 || !kv3 || !raw_kv_ring || !comp_kv ||
        !attn_sink || !out0 || !out1 || !out2 || !out3 ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head0 >= raw_slots ||
        raw_count0 == 0 || raw_count0 > raw_slots || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q0->runtime != runtime || q1->runtime != runtime ||
        q2->runtime != runtime || q3->runtime != runtime ||
        kv0->runtime != runtime || kv1->runtime != runtime ||
        kv2->runtime != runtime || kv3->runtime != runtime ||
        raw_kv_ring->runtime != runtime || comp_kv->runtime != runtime ||
        attn_sink->runtime != runtime || out0->runtime != runtime ||
        out1->runtime != runtime || out2->runtime != runtime ||
        out3->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t raw_count1 = raw_count0 < raw_slots ? raw_count0 + 1u : raw_slots;
    const uint32_t raw_count2 = raw_count1 < raw_slots ? raw_count1 + 1u : raw_slots;
    const uint32_t raw_count3 = raw_count2 < raw_slots ? raw_count2 + 1u : raw_slots;
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_slots * head_dim * sizeof(float);
    uint32_t comp_cap = comp_count0 > comp_count1 ? comp_count0 : comp_count1;
    comp_cap = comp_cap > comp_count2 ? comp_cap : comp_count2;
    comp_cap = comp_cap > comp_count3 ? comp_cap : comp_count3;
    const uint64_t comp_bytes = (uint64_t)comp_cap * head_dim * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q0_offset > q0->bytes || q_bytes > q0->bytes - q0_offset ||
        q1_offset > q1->bytes || q_bytes > q1->bytes - q1_offset ||
        q2_offset > q2->bytes || q_bytes > q2->bytes - q2_offset ||
        q3_offset > q3->bytes || q_bytes > q3->bytes - q3_offset ||
        kv0_offset > kv0->bytes || kv_bytes > kv0->bytes - kv0_offset ||
        kv1_offset > kv1->bytes || kv_bytes > kv1->bytes - kv1_offset ||
        kv2_offset > kv2->bytes || kv_bytes > kv2->bytes - kv2_offset ||
        kv3_offset > kv3->bytes || kv_bytes > kv3->bytes - kv3_offset ||
        raw_kv_ring_offset > raw_kv_ring->bytes || raw_bytes > raw_kv_ring->bytes - raw_kv_ring_offset ||
        comp_kv_offset > comp_kv->bytes || comp_bytes > comp_kv->bytes - comp_kv_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out0_offset > out0->bytes || q_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || q_bytes > out1->bytes - out1_offset ||
        out2_offset > out2->bytes || q_bytes > out2->bytes - out2_offset ||
        out3_offset > out3->bytes || q_bytes > out3->bytes - out3_offset ||
        raw_count0 + comp_count0 > 16384u ||
        raw_count1 + comp_count1 > 16384u ||
        raw_count2 + comp_count2 > 16384u ||
        raw_count3 + comp_count3 > 16384u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (axiom_ranges_overlap(out0, out0_offset, q_bytes, out1, out1_offset, q_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, q_bytes, out2, out2_offset, q_bytes) ||
        axiom_ranges_overlap(out0, out0_offset, q_bytes, out3, out3_offset, q_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, q_bytes, out2, out2_offset, q_bytes) ||
        axiom_ranges_overlap(out1, out1_offset, q_bytes, out3, out3_offset, q_bytes) ||
        axiom_ranges_overlap(out2, out2_offset, q_bytes, out3, out3_offset, q_bytes)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_raw_comp_ring4_causal_f32_device(
            runtime->backend_runtime,
            q0->backend_buffer, q0_offset,
            q1->backend_buffer, q1_offset,
            q2->backend_buffer, q2_offset,
            q3->backend_buffer, q3_offset,
            kv0->backend_buffer, kv0_offset,
            kv1->backend_buffer, kv1_offset,
            kv2->backend_buffer, kv2_offset,
            kv3->backend_buffer, kv3_offset,
            raw_kv_ring->backend_buffer, raw_kv_ring_offset,
            comp_kv->backend_buffer, comp_kv_offset,
            attn_sink->backend_buffer, attn_sink_offset,
            out0->backend_buffer, out0_offset,
            out1->backend_buffer, out1_offset,
            out2->backend_buffer, out2_offset,
            out3->backend_buffer, out3_offset,
            heads, head_dim, raw_slots, raw_head0, raw_count0,
            comp_count0, comp_count1, comp_count2, comp_count3, eps);
}

int axiom_runtime_deepseek_csa_indexer_qat_f32_device(
        axiom_runtime *runtime,
        axiom_device_buffer *x,
        uint64_t x_offset,
        uint32_t rows,
        uint32_t head_dim) {
    if (!runtime || !x || rows == 0 || head_dim != 128u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (x->runtime != runtime) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint64_t bytes = (uint64_t)rows * head_dim * sizeof(float);
    if (x_offset > x->bytes || bytes > x->bytes - x_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_indexer_qat_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            rows,
            head_dim);
}

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
        uint32_t topk) {
    if (!runtime || !q || !index_weights || !index_comp || !selected ||
        comp_count == 0 || topk == 0 || topk > 512u || topk > comp_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || index_weights->runtime != runtime ||
        index_comp->runtime != runtime || selected->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = 64ull * 128ull * sizeof(float);
    const uint64_t weight_bytes = 64ull * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)comp_count * 128ull * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)topk * sizeof(uint32_t);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        index_weights_offset > index_weights->bytes ||
        weight_bytes > index_weights->bytes - index_weights_offset ||
        index_comp_offset > index_comp->bytes ||
        comp_bytes > index_comp->bytes - index_comp_offset ||
        selected_offset > selected->bytes ||
        selected_bytes > selected->bytes - selected_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_indexer_topk_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            index_weights->backend_buffer,
            index_weights_offset,
            index_comp->backend_buffer,
            index_comp_offset,
            selected->backend_buffer,
            selected_offset,
            comp_count,
            topk);
}

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
        uint32_t topk) {
    if (!runtime || !q || !index_weights || !index_comp || !scores || !selected ||
        comp_count == 0 || topk == 0 || topk > 512u || topk > comp_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || index_weights->runtime != runtime ||
        index_comp->runtime != runtime || scores->runtime != runtime ||
        selected->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = 64ull * 128ull * sizeof(float);
    const uint64_t weight_bytes = 64ull * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)comp_count * 128ull * sizeof(float);
    const uint64_t score_bytes = (uint64_t)comp_count * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)topk * sizeof(uint32_t);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        index_weights_offset > index_weights->bytes ||
        weight_bytes > index_weights->bytes - index_weights_offset ||
        index_comp_offset > index_comp->bytes ||
        comp_bytes > index_comp->bytes - index_comp_offset ||
        scores_offset > scores->bytes ||
        score_bytes > scores->bytes - scores_offset ||
        selected_offset > selected->bytes ||
        selected_bytes > selected->bytes - selected_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_indexer_topk_scratch_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            index_weights->backend_buffer,
            index_weights_offset,
            index_comp->backend_buffer,
            index_comp_offset,
            scores->backend_buffer,
            scores_offset,
            selected->backend_buffer,
            selected_offset,
            comp_count,
            topk);
}

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
        float eps) {
    if (!runtime || !q || !raw_kv_ring || !comp_kv || !selected || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head >= raw_slots ||
        raw_count == 0 || raw_count > raw_slots || selected_count > 512u ||
        selected_count > comp_count || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || raw_kv_ring->runtime != runtime ||
        comp_kv->runtime != runtime || selected->runtime != runtime ||
        attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_slots * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)comp_count * head_dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)selected_count * sizeof(uint32_t);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        raw_kv_ring_offset > raw_kv_ring->bytes || raw_bytes > raw_kv_ring->bytes - raw_kv_ring_offset ||
        comp_kv_offset > comp_kv->bytes || comp_bytes > comp_kv->bytes - comp_kv_offset ||
        selected_offset > selected->bytes || selected_bytes > selected->bytes - selected_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || q_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_raw_selected_comp_ring_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            raw_kv_ring->backend_buffer,
            raw_kv_ring_offset,
            comp_kv->backend_buffer,
            comp_kv_offset,
            selected->backend_buffer,
            selected_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            raw_slots,
            raw_head,
            raw_count,
            comp_count,
            selected_count,
            eps);
}

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
        float eps) {
    if (!runtime || !q || !raw_kv_ring || !comp_kv || !selected || !attn_sink || !out ||
        heads == 0 || head_dim == 0 || raw_slots == 0 || raw_head >= raw_slots ||
        raw_count == 0 || raw_count > raw_slots || selected_count > 512u ||
        selected_count > comp_count || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (q->runtime != runtime || raw_kv_ring->runtime != runtime ||
        comp_kv->runtime != runtime || selected->runtime != runtime ||
        attn_sink->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t q_bytes = (uint64_t)heads * head_dim * sizeof(float);
    const uint64_t raw_bytes = (uint64_t)raw_slots * head_dim * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)comp_count * head_dim * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)selected_count * sizeof(uint32_t);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    if (q_offset > q->bytes || q_bytes > q->bytes - q_offset ||
        raw_kv_ring_offset > raw_kv_ring->bytes || raw_bytes > raw_kv_ring->bytes - raw_kv_ring_offset ||
        comp_kv_offset > comp_kv->bytes || comp_bytes > comp_kv->bytes - comp_kv_offset ||
        selected_offset > selected->bytes || selected_bytes > selected->bytes - selected_offset ||
        attn_sink_offset > attn_sink->bytes || sink_bytes > attn_sink->bytes - attn_sink_offset ||
        out_offset > out->bytes || q_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_attention_raw_selected_comp_ring_trusted_f32_device(
            runtime->backend_runtime,
            q->backend_buffer,
            q_offset,
            raw_kv_ring->backend_buffer,
            raw_kv_ring_offset,
            comp_kv->backend_buffer,
            comp_kv_offset,
            selected->backend_buffer,
            selected_offset,
            attn_sink->backend_buffer,
            attn_sink_offset,
            out->backend_buffer,
            out_offset,
            heads,
            head_dim,
            raw_slots,
            raw_head,
            raw_count,
            comp_count,
            selected_count,
            eps);
}

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
        uint32_t head_dim) {
    if (!runtime || !kv || !gate || !bias || !out || ratio == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (kv->runtime != runtime || gate->runtime != runtime ||
        bias->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t window_bytes = (uint64_t)ratio * 2u * head_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_offset > kv->bytes || window_bytes > kv->bytes - kv_offset ||
        gate_offset > gate->bytes || window_bytes > gate->bytes - gate_offset ||
        bias_offset > bias->bytes || window_bytes > bias->bytes - bias_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_cold_window_f32_device(
            runtime->backend_runtime,
            kv->backend_buffer,
            kv_offset,
            gate->backend_buffer,
            gate_offset,
            bias->backend_buffer,
            bias_offset,
            out->backend_buffer,
            out_offset,
            ratio,
            head_dim);
}

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
        uint32_t ring_head) {
    return axiom_runtime_deepseek_csa_ring_window_count_f32_device(
            runtime, kv, kv_offset, gate, gate_offset, bias, bias_offset,
            out, out_offset, ratio, head_dim, ring_head, ratio);
}

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
        uint32_t ring_count) {
    if (!runtime || !kv || !gate || !bias || !out ||
        ratio == 0 || head_dim == 0 || ring_head >= ratio || ring_count > ratio) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (kv->runtime != runtime || gate->runtime != runtime ||
        bias->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t window_bytes = (uint64_t)ratio * 2u * head_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_offset > kv->bytes || window_bytes > kv->bytes - kv_offset ||
        gate_offset > gate->bytes || window_bytes > gate->bytes - gate_offset ||
        bias_offset > bias->bytes || window_bytes > bias->bytes - bias_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_ring_window_count_f32_device(
            runtime->backend_runtime,
            kv->backend_buffer,
            kv_offset,
            gate->backend_buffer,
            gate_offset,
            bias->backend_buffer,
            bias_offset,
            out->backend_buffer,
            out_offset,
            ratio,
            head_dim,
            ring_head,
            ring_count);
}

int axiom_runtime_deepseek_csa_state_pool_f32_device(
        axiom_runtime *runtime,
        const axiom_device_buffer *kv,
        uint64_t kv_offset,
        const axiom_device_buffer *gate,
        uint64_t gate_offset,
        axiom_device_buffer *out,
        uint64_t out_offset,
        uint32_t head_dim,
        uint32_t use_previous) {
    if (!runtime || !kv || !gate || !out || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (kv->runtime != runtime || gate->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t state_bytes = 16ull * head_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_offset > kv->bytes || state_bytes > kv->bytes - kv_offset ||
        gate_offset > gate->bytes || state_bytes > gate->bytes - gate_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_csa_state_pool_f32_device(
            runtime->backend_runtime,
            kv->backend_buffer,
            kv_offset,
            gate->backend_buffer,
            gate_offset,
            out->backend_buffer,
            out_offset,
            head_dim,
            use_previous);
}

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
        uint32_t head_dim) {
    if (!runtime || !kv || !gate || !bias || !out || ratio == 0 || head_dim == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (kv->runtime != runtime || gate->runtime != runtime ||
        bias->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t window_bytes = (uint64_t)ratio * head_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_offset > kv->bytes || window_bytes > kv->bytes - kv_offset ||
        gate_offset > gate->bytes || window_bytes > gate->bytes - gate_offset ||
        bias_offset > bias->bytes || window_bytes > bias->bytes - bias_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_hca_window_f32_device(
            runtime->backend_runtime,
            kv->backend_buffer,
            kv_offset,
            gate->backend_buffer,
            gate_offset,
            bias->backend_buffer,
            bias_offset,
            out->backend_buffer,
            out_offset,
            ratio,
            head_dim);
}

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
        uint32_t ring_head) {
    return axiom_runtime_deepseek_hca_ring_window_count_f32_device(
            runtime, kv, kv_offset, gate, gate_offset, bias, bias_offset,
            out, out_offset, ratio, head_dim, ring_head, ratio);
}

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
        uint32_t ring_count) {
    if (!runtime || !kv || !gate || !bias || !out ||
        ratio == 0 || head_dim == 0 || ring_head >= ratio || ring_count > ratio) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (kv->runtime != runtime || gate->runtime != runtime ||
        bias->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t window_bytes = (uint64_t)ratio * head_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)head_dim * sizeof(float);
    if (kv_offset > kv->bytes || window_bytes > kv->bytes - kv_offset ||
        gate_offset > gate->bytes || window_bytes > gate->bytes - gate_offset ||
        bias_offset > bias->bytes || window_bytes > bias->bytes - bias_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_hca_ring_window_count_f32_device(
            runtime->backend_runtime,
            kv->backend_buffer,
            kv_offset,
            gate->backend_buffer,
            gate_offset,
            bias->backend_buffer,
            bias_offset,
            out->backend_buffer,
            out_offset,
            ratio,
            head_dim,
            ring_head,
            ring_count);
}

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
        float eps) {
    if (!runtime || !fn_f16 || !scale_f32 || !base_f32 || !streams || !out || !post || !comb ||
        hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (fn_f16->runtime != runtime || scale_f32->runtime != runtime || base_f32->runtime != runtime ||
        streams->runtime != runtime || out->runtime != runtime || post->runtime != runtime ||
        comb->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t flat = (uint64_t)4u * hidden;
    const uint64_t fn_bytes = (uint64_t)24u * flat * sizeof(uint16_t);
    const uint64_t scale_bytes = 3u * sizeof(float);
    const uint64_t base_bytes = 24u * sizeof(float);
    const uint64_t streams_bytes = flat * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (fn_offset > fn_f16->bytes || fn_bytes > fn_f16->bytes - fn_offset ||
        scale_offset > scale_f32->bytes || scale_bytes > scale_f32->bytes - scale_offset ||
        base_offset > base_f32->bytes || base_bytes > base_f32->bytes - base_offset ||
        streams_offset > streams->bytes || streams_bytes > streams->bytes - streams_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_hc_pre_f32_device(
            runtime->backend_runtime,
            fn_f16->backend_buffer,
            fn_offset,
            scale_f32->backend_buffer,
            scale_offset,
            base_f32->backend_buffer,
            base_offset,
            streams->backend_buffer,
            streams_offset,
            out->backend_buffer,
            out_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            hidden,
            eps);
}

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
        float eps) {
    if (!runtime || !fn_f32 || !scale_f32 || !base_f32 || !streams || !out || !post || !comb ||
        hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (fn_f32->runtime != runtime || scale_f32->runtime != runtime || base_f32->runtime != runtime ||
        streams->runtime != runtime || out->runtime != runtime || post->runtime != runtime ||
        comb->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t flat = (uint64_t)4u * hidden;
    const uint64_t fn_bytes = (uint64_t)24u * flat * sizeof(float);
    const uint64_t scale_bytes = 3u * sizeof(float);
    const uint64_t base_bytes = 24u * sizeof(float);
    const uint64_t streams_bytes = flat * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (fn_offset > fn_f32->bytes || fn_bytes > fn_f32->bytes - fn_offset ||
        scale_offset > scale_f32->bytes || scale_bytes > scale_f32->bytes - scale_offset ||
        base_offset > base_f32->bytes || base_bytes > base_f32->bytes - base_offset ||
        streams_offset > streams->bytes || streams_bytes > streams->bytes - streams_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_hc_pre_fn_f32_device(
            runtime->backend_runtime,
            fn_f32->backend_buffer,
            fn_offset,
            scale_f32->backend_buffer,
            scale_offset,
            base_f32->backend_buffer,
            base_offset,
            streams->backend_buffer,
            streams_offset,
            out->backend_buffer,
            out_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            hidden,
            eps);
}

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
        uint32_t hidden) {
    if (!runtime || !x || !residual || !post || !comb || !out || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (x->runtime != runtime || residual->runtime != runtime || post->runtime != runtime ||
        comb->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4u * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (x_offset > x->bytes || hidden_bytes > x->bytes - x_offset ||
        residual_offset > residual->bytes || streams_bytes > residual->bytes - residual_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset ||
        out_offset > out->bytes || streams_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_hc_post_f32_device(
            runtime->backend_runtime,
            x->backend_buffer,
            x_offset,
            residual->backend_buffer,
            residual_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            out->backend_buffer,
            out_offset,
            hidden);
}

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
        uint32_t cols) {
    if (!runtime || !weight_q8 || !input || !block_out || !residual || !post ||
        !comb || !out || hidden == 0 || cols == 0 || (cols % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (weight_q8->runtime != runtime || input->runtime != runtime ||
        block_out->runtime != runtime || residual->runtime != runtime ||
        post->runtime != runtime || comb->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t blocks = cols / 32u;
    const uint64_t weight_bytes = (uint64_t)hidden * blocks * 34u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4u * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (weight_offset > weight_q8->bytes || weight_bytes > weight_q8->bytes - weight_offset ||
        input_offset > input->bytes || input_bytes > input->bytes - input_offset ||
        block_out_offset > block_out->bytes || hidden_bytes > block_out->bytes - block_out_offset ||
        residual_offset > residual->bytes || streams_bytes > residual->bytes - residual_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset ||
        out_offset > out->bytes || streams_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_q8_0_matvec_hc_post_f32_device(
            runtime->backend_runtime,
            weight_q8->backend_buffer,
            weight_offset,
            input->backend_buffer,
            input_offset,
            block_out->backend_buffer,
            block_out_offset,
            residual->backend_buffer,
            residual_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            cols);
}

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
        uint32_t hidden) {
    if (!runtime || !shared || !moe || !residual || !post || !comb || !out || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (shared->runtime != runtime || moe->runtime != runtime || residual->runtime != runtime ||
        post->runtime != runtime || comb->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4u * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (shared_offset > shared->bytes || hidden_bytes > shared->bytes - shared_offset ||
        moe_offset > moe->bytes || hidden_bytes > moe->bytes - moe_offset ||
        residual_offset > residual->bytes || streams_bytes > residual->bytes - residual_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset ||
        out_offset > out->bytes || streams_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_ffn_hc_post_f32_device(
            runtime->backend_runtime,
            shared->backend_buffer,
            shared_offset,
            moe->backend_buffer,
            moe_offset,
            residual->backend_buffer,
            residual_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            out->backend_buffer,
            out_offset,
            hidden);
}

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
        uint32_t hidden) {
    if (!runtime || !shared || !moe || !residual || !post || !comb || !out_a || !out_b || hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (shared->runtime != runtime || moe->runtime != runtime || residual->runtime != runtime ||
        post->runtime != runtime || comb->runtime != runtime ||
        out_a->runtime != runtime || out_b->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4u * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (shared_offset > shared->bytes || hidden_bytes > shared->bytes - shared_offset ||
        moe_offset > moe->bytes || hidden_bytes > moe->bytes - moe_offset ||
        residual_offset > residual->bytes || streams_bytes > residual->bytes - residual_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset ||
        out_a_offset > out_a->bytes || streams_bytes > out_a->bytes - out_a_offset ||
        out_b_offset > out_b->bytes || streams_bytes > out_b->bytes - out_b_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_ffn_hc_post_dual_out_f32_device(
            runtime->backend_runtime,
            shared->backend_buffer,
            shared_offset,
            moe->backend_buffer,
            moe_offset,
            residual->backend_buffer,
            residual_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            out_a->backend_buffer,
            out_a_offset,
            out_b->backend_buffer,
            out_b_offset,
            hidden);
}

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
        uint32_t hidden) {
    if (!runtime || !shared0 || !moe0 || !residual0 || !post0 || !comb0 ||
        !shared1 || !moe1 || !residual1 || !post1 || !comb1 || !out0 || !out1 ||
        hidden == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (shared0->runtime != runtime || moe0->runtime != runtime ||
        residual0->runtime != runtime || post0->runtime != runtime || comb0->runtime != runtime ||
        shared1->runtime != runtime || moe1->runtime != runtime ||
        residual1->runtime != runtime || post1->runtime != runtime || comb1->runtime != runtime ||
        out0->runtime != runtime || out1->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4ull * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (shared0_offset > shared0->bytes || hidden_bytes > shared0->bytes - shared0_offset ||
        moe0_offset > moe0->bytes || hidden_bytes > moe0->bytes - moe0_offset ||
        residual0_offset > residual0->bytes || streams_bytes > residual0->bytes - residual0_offset ||
        post0_offset > post0->bytes || post_bytes > post0->bytes - post0_offset ||
        comb0_offset > comb0->bytes || comb_bytes > comb0->bytes - comb0_offset ||
        shared1_offset > shared1->bytes || hidden_bytes > shared1->bytes - shared1_offset ||
        moe1_offset > moe1->bytes || hidden_bytes > moe1->bytes - moe1_offset ||
        residual1_offset > residual1->bytes || streams_bytes > residual1->bytes - residual1_offset ||
        post1_offset > post1->bytes || post_bytes > post1->bytes - post1_offset ||
        comb1_offset > comb1->bytes || comb_bytes > comb1->bytes - comb1_offset ||
        out0_offset > out0->bytes || streams_bytes > out0->bytes - out0_offset ||
        out1_offset > out1->bytes || streams_bytes > out1->bytes - out1_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_ffn_hc_post2_f32_device(
            runtime->backend_runtime,
            shared0->backend_buffer,
            shared0_offset,
            moe0->backend_buffer,
            moe0_offset,
            residual0->backend_buffer,
            residual0_offset,
            post0->backend_buffer,
            post0_offset,
            comb0->backend_buffer,
            comb0_offset,
            shared1->backend_buffer,
            shared1_offset,
            moe1->backend_buffer,
            moe1_offset,
            residual1->backend_buffer,
            residual1_offset,
            post1->backend_buffer,
            post1_offset,
            comb1->backend_buffer,
            comb1_offset,
            out0->backend_buffer,
            out0_offset,
            out1->backend_buffer,
            out1_offset,
            hidden);
}

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
        uint32_t expert_hidden) {
    if (!runtime || !down_q8 || !shared_mid || !moe || !residual || !post || !comb ||
        !out_a || !out_b || hidden == 0 || expert_hidden == 0 ||
        (expert_hidden % 32u) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (down_q8->runtime != runtime || shared_mid->runtime != runtime ||
        moe->runtime != runtime || residual->runtime != runtime ||
        post->runtime != runtime || comb->runtime != runtime ||
        out_a->runtime != runtime || out_b->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t down_bytes = (uint64_t)hidden * (expert_hidden / 32u) * 34u;
    const uint64_t mid_bytes = (uint64_t)expert_hidden * sizeof(float);
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t streams_bytes = 4u * hidden_bytes;
    const uint64_t post_bytes = 4u * sizeof(float);
    const uint64_t comb_bytes = 16u * sizeof(float);
    if (down_offset > down_q8->bytes || down_bytes > down_q8->bytes - down_offset ||
        shared_mid_offset > shared_mid->bytes || mid_bytes > shared_mid->bytes - shared_mid_offset ||
        moe_offset > moe->bytes || hidden_bytes > moe->bytes - moe_offset ||
        residual_offset > residual->bytes || streams_bytes > residual->bytes - residual_offset ||
        post_offset > post->bytes || post_bytes > post->bytes - post_offset ||
        comb_offset > comb->bytes || comb_bytes > comb->bytes - comb_offset ||
        out_a_offset > out_a->bytes || streams_bytes > out_a->bytes - out_a_offset ||
        out_b_offset > out_b->bytes || streams_bytes > out_b->bytes - out_b_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_shared_down_ffn_hc_post_q8_f32_device(
            runtime->backend_runtime,
            down_q8->backend_buffer,
            down_offset,
            shared_mid->backend_buffer,
            shared_mid_offset,
            moe->backend_buffer,
            moe_offset,
            residual->backend_buffer,
            residual_offset,
            post->backend_buffer,
            post_offset,
            comb->backend_buffer,
            comb_offset,
            out_a->backend_buffer,
            out_a_offset,
            out_b->backend_buffer,
            out_b_offset,
            hidden,
            expert_hidden);
}

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
        float eps) {
    if (!runtime || !fn_f16 || !scale_f32 || !base_f32 || !streams || !out ||
        hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (fn_f16->runtime != runtime || scale_f32->runtime != runtime ||
        base_f32->runtime != runtime || streams->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t flat = (uint64_t)4u * hidden;
    const uint64_t fn_bytes = 4u * flat * sizeof(uint16_t);
    const uint64_t scale_bytes = sizeof(float);
    const uint64_t base_bytes = 4u * sizeof(float);
    const uint64_t streams_bytes = flat * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (fn_offset > fn_f16->bytes || fn_bytes > fn_f16->bytes - fn_offset ||
        scale_offset > scale_f32->bytes || scale_bytes > scale_f32->bytes - scale_offset ||
        base_offset > base_f32->bytes || base_bytes > base_f32->bytes - base_offset ||
        streams_offset > streams->bytes || streams_bytes > streams->bytes - streams_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_output_hc_f32_device(
            runtime->backend_runtime,
            fn_f16->backend_buffer,
            fn_offset,
            scale_f32->backend_buffer,
            scale_offset,
            base_f32->backend_buffer,
            base_offset,
            streams->backend_buffer,
            streams_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            eps);
}

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
        float eps) {
    if (!runtime || !fn_f32 || !scale_f32 || !base_f32 || !streams || !out ||
        hidden == 0 || eps <= 0.0f) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    if (fn_f32->runtime != runtime || scale_f32->runtime != runtime ||
        base_f32->runtime != runtime || streams->runtime != runtime || out->runtime != runtime) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t flat = (uint64_t)4u * hidden;
    const uint64_t fn_bytes = 4u * flat * sizeof(float);
    const uint64_t scale_bytes = sizeof(float);
    const uint64_t base_bytes = 4u * sizeof(float);
    const uint64_t streams_bytes = flat * sizeof(float);
    const uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (fn_offset > fn_f32->bytes || fn_bytes > fn_f32->bytes - fn_offset ||
        scale_offset > scale_f32->bytes || scale_bytes > scale_f32->bytes - scale_offset ||
        base_offset > base_f32->bytes || base_bytes > base_f32->bytes - base_offset ||
        streams_offset > streams->bytes || streams_bytes > streams->bytes - streams_offset ||
        out_offset > out->bytes || out_bytes > out->bytes - out_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return axiom_cuda_deepseek_output_hc_fn_f32_device(
            runtime->backend_runtime,
            fn_f32->backend_buffer,
            fn_offset,
            scale_f32->backend_buffer,
            scale_offset,
            base_f32->backend_buffer,
            base_offset,
            streams->backend_buffer,
            streams_offset,
            out->backend_buffer,
            out_offset,
            hidden,
            eps);
}

int axiom_cluster_create(axiom_cluster **out, const axiom_cluster_config *config) {
    if (!out || !config || config->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (config->transport != AXIOM_TRANSPORT_INPROC &&
        config->transport != AXIOM_TRANSPORT_SHM &&
        config->transport != AXIOM_TRANSPORT_TCP &&
        config->transport != AXIOM_TRANSPORT_QUIC &&
        config->transport != AXIOM_TRANSPORT_RDMA) {
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (config->node_count == 0 || config->node_id >= config->node_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_cluster *cluster = axiom_cluster_alloc_base(
            config->transport,
            config->node_id,
            config->node_count,
            config->flags);
    if (!cluster) return AXIOM_ERR_RUNTIME;
    cluster->listen_addr = config->listen_addr ? config->listen_addr : "";
    cluster->join_addr = config->join_addr ? config->join_addr : "";
    if (axiom_cluster_transport_is_stream(config->transport)) {
        const bool has_listen = !cluster->listen_addr.empty();
        const bool has_join = !cluster->join_addr.empty();
        if (has_listen == has_join) {
            axiom_cluster_release(cluster);
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (has_listen) {
            int fd = -1;
            std::string resolved;
            int rc = AXIOM_ERR_NOT_IMPLEMENTED;
            if (config->transport == AXIOM_TRANSPORT_TCP) {
                rc = axiom_tcp_listen(
                        cluster->listen_addr,
                        cluster->node_count > 1 ? cluster->node_count - 1u : 1u,
                        &fd,
                        &resolved);
            } else if (config->transport == AXIOM_TRANSPORT_RDMA) {
#if AXIOM_ENABLE_RDMA
                rc = axiom_rdma_listen(
                        cluster->listen_addr,
                        cluster->node_count > 1 ? cluster->node_count - 1u : 1u,
                        &fd,
                        &resolved);
#endif
            } else if (config->transport == AXIOM_TRANSPORT_QUIC) {
#if AXIOM_HAVE_OPENSSL_QUIC
                rc = axiom_quic_listen(cluster, cluster->listen_addr, &resolved);
#endif
            }
            if (rc != AXIOM_OK) {
                axiom_cluster_release(cluster);
                return rc;
            }
            if (config->transport != AXIOM_TRANSPORT_QUIC) cluster->listen_fd = fd;
            cluster->listen_addr = resolved;
            cluster->is_listener = true;
        } else {
            int fd = -1;
            int rc = AXIOM_ERR_NOT_IMPLEMENTED;
            if (config->transport == AXIOM_TRANSPORT_TCP) {
                rc = axiom_tcp_connect(cluster->join_addr, &fd);
            } else if (config->transport == AXIOM_TRANSPORT_RDMA) {
#if AXIOM_ENABLE_RDMA
                rc = axiom_rdma_connect(cluster->join_addr, &fd);
#endif
            } else if (config->transport == AXIOM_TRANSPORT_QUIC) {
#if AXIOM_HAVE_OPENSSL_QUIC
                rc = axiom_quic_connect(cluster, cluster->join_addr);
#endif
            }
            if (rc != AXIOM_OK) {
                axiom_cluster_release(cluster);
                return rc;
            }
            if (config->transport != AXIOM_TRANSPORT_QUIC) cluster->peer_fd = fd;
            cluster->connected_peers = 1;
            cluster->awaiting_ack = true;
            const int hello_rc = axiom_cluster_write_control(cluster, AXIOM_CLUSTER_FRAME_HELLO);
            if (hello_rc != AXIOM_OK) {
                axiom_cluster_release(cluster);
                return hello_rc;
            }
        }
    }
    *out = cluster;
    return AXIOM_OK;
}

void axiom_cluster_destroy(axiom_cluster *cluster) {
    axiom_cluster_release(cluster);
}

int axiom_cluster_info_get(const axiom_cluster *cluster, axiom_cluster_info *out) {
    if (!cluster || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t requested_abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = requested_abi;
    out->transport = cluster->transport;
    axiom_copy_cstr(out->listen_addr, sizeof(out->listen_addr), cluster->listen_addr);
    axiom_copy_cstr(out->join_addr, sizeof(out->join_addr), cluster->join_addr);
    out->node_id = cluster->node_id;
    out->node_count = cluster->node_count;
    out->connected_peers = cluster->connected_peers;
    out->protocol_version = AXIOM_CLUSTER_PROTOCOL_VERSION;
    out->flags = cluster->flags;
    return AXIOM_OK;
}

int axiom_cluster_accept(axiom_cluster *listener, axiom_cluster **out_peer) {
    if (!listener || !out_peer || !axiom_cluster_transport_is_stream(listener->transport) ||
        !listener->is_listener || listener->listen_fd < 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_peer = nullptr;
#if AXIOM_HAVE_OPENSSL_QUIC
    if (listener->transport == AXIOM_TRANSPORT_QUIC) {
        if (!listener->quic_listener) return AXIOM_ERR_INVALID_ARGUMENT;
        SSL *conn = SSL_accept_connection(listener->quic_listener, 0);
        if (!conn) return AXIOM_ERR_IO;
        int rc = AXIOM_OK;
        if (SSL_set_blocking_mode(conn, 1) != 1 ||
            SSL_set_default_stream_mode(conn, SSL_DEFAULT_STREAM_MODE_AUTO_BIDI) != 1 ||
            SSL_set_incoming_stream_policy(conn, SSL_INCOMING_STREAM_POLICY_AUTO, 0) != 1) {
            SSL_free(conn);
            return AXIOM_ERR_IO;
        }
        axiom_cluster *peer = axiom_cluster_alloc_base(
                listener->transport,
                listener->node_id,
                listener->node_count,
                listener->flags);
        if (!peer) {
            SSL_free(conn);
            return AXIOM_ERR_RUNTIME;
        }
        SSL_CTX *ctx = SSL_get_SSL_CTX(conn);
        if (ctx && SSL_CTX_up_ref(ctx) == 1) {
            peer->quic_ctx = ctx;
        }
        peer->listen_addr = listener->listen_addr;
        peer->quic_conn = conn;
        peer->connected_peers = 1;
        rc = axiom_cluster_read_control(peer, AXIOM_CLUSTER_FRAME_HELLO);
        if (rc == AXIOM_OK) rc = axiom_cluster_write_control(peer, AXIOM_CLUSTER_FRAME_HELLO_ACK);
        if (rc != AXIOM_OK) {
            axiom_cluster_release(peer);
            return rc;
        }
        ++listener->connected_peers;
        *out_peer = peer;
        return AXIOM_OK;
    }
#endif
    sockaddr_storage ss;
    socklen_t ss_len = sizeof(ss);
    int fd = -1;
    do {
        if (listener->transport == AXIOM_TRANSPORT_RDMA) {
#if AXIOM_ENABLE_RDMA
            fd = raccept(listener->listen_fd, (sockaddr *)&ss, &ss_len);
#else
            return AXIOM_ERR_NOT_IMPLEMENTED;
#endif
        } else {
            fd = accept(listener->listen_fd, (sockaddr *)&ss, &ss_len);
        }
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) return AXIOM_ERR_IO;
    int rc = AXIOM_OK;
    if (listener->transport == AXIOM_TRANSPORT_RDMA) {
#if AXIOM_ENABLE_RDMA
        rc = axiom_rdma_set_nodelay(fd);
#else
        rc = AXIOM_ERR_NOT_IMPLEMENTED;
#endif
    } else {
        rc = axiom_socket_set_nodelay(fd);
    }
    if (rc != AXIOM_OK) {
        axiom_transport_close(listener->transport, &fd);
        return rc;
    }
    axiom_cluster *peer = axiom_cluster_alloc_base(
            listener->transport,
            listener->node_id,
            listener->node_count,
            listener->flags);
    if (!peer) {
        axiom_transport_close(listener->transport, &fd);
        return AXIOM_ERR_RUNTIME;
    }
    peer->listen_addr = listener->listen_addr;
    peer->peer_fd = fd;
    peer->connected_peers = 1;
    rc = axiom_cluster_read_control(peer, AXIOM_CLUSTER_FRAME_HELLO);
    if (rc == AXIOM_OK) rc = axiom_cluster_write_control(peer, AXIOM_CLUSTER_FRAME_HELLO_ACK);
    if (rc != AXIOM_OK) {
        axiom_cluster_release(peer);
        return rc;
    }
    ++listener->connected_peers;
    *out_peer = peer;
    return AXIOM_OK;
}

int axiom_cluster_send_latent(
        axiom_cluster *cluster,
        const axiom_latent_shard *shard,
        const void *payload,
        uint64_t bytes) {
    if (!cluster || !shard || shard->abi_version != AXIOM_ABI_VERSION ||
        !axiom_cluster_transport_is_stream(cluster->transport) ||
        !axiom_cluster_has_stream_peer(cluster) ||
        bytes == 0 || !payload) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = axiom_cluster_ensure_ack(cluster);
    if (rc != AXIOM_OK) return rc;
    axiom_cluster_wire_frame frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.kind = AXIOM_CLUSTER_FRAME_LATENT;
    frame.flags = (uint32_t)(shard->flags & 0xffffffffu);
    frame.sequence = ++cluster->sequence;
    frame.payload_bytes = bytes;
    frame.shard_id = shard->shard_id;
    frame.session_id = shard->session_id;
    frame.content_hash = shard->content_hash ? shard->content_hash : axiom_fnv1a64(payload, bytes);
    frame.dtype = (uint32_t)shard->dtype;
    frame.rows = shard->rows;
    frame.cols = shard->cols;
    frame.stride = shard->stride;
    frame.source_node = shard->source_node;
    frame.source_device = shard->source_device;
    return axiom_wire_write_frame(cluster, &frame, payload);
}

int axiom_cluster_recv_latent(
        axiom_cluster *cluster,
        axiom_latent_shard *out_shard,
        void *payload,
        uint64_t payload_capacity,
        uint64_t *out_bytes) {
    if (!cluster || !out_shard || out_shard->abi_version != AXIOM_ABI_VERSION ||
        !out_bytes || !axiom_cluster_transport_is_stream(cluster->transport) ||
        !axiom_cluster_has_stream_peer(cluster)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_bytes = 0;
    int rc = axiom_cluster_ensure_ack(cluster);
    if (rc != AXIOM_OK) return rc;
    axiom_cluster_wire_frame frame;
    rc = axiom_wire_read_frame(cluster, &frame);
    if (rc != AXIOM_OK) return rc;
    if (frame.kind != AXIOM_CLUSTER_FRAME_LATENT) {
        if (frame.payload_bytes > 0) (void)axiom_cluster_drain(cluster, frame.payload_bytes);
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_bytes = frame.payload_bytes;
    if (frame.payload_bytes > payload_capacity || (frame.payload_bytes > 0 && !payload)) {
        (void)axiom_cluster_drain(cluster, frame.payload_bytes);
        return AXIOM_ERR_BUDGET;
    }
    if (frame.payload_bytes > 0) {
        rc = axiom_cluster_read_all(cluster, payload, frame.payload_bytes);
        if (rc != AXIOM_OK) return rc;
    }
    const uint64_t actual_hash = frame.payload_bytes > 0
            ? axiom_fnv1a64(payload, frame.payload_bytes)
            : 0;
    if (frame.content_hash != 0 && actual_hash != frame.content_hash) {
        return AXIOM_ERR_IO;
    }
    const uint32_t requested_abi = out_shard->abi_version;
    std::memset(out_shard, 0, sizeof(*out_shard));
    out_shard->abi_version = requested_abi;
    out_shard->dtype = (axiom_latent_dtype)frame.dtype;
    out_shard->rows = frame.rows;
    out_shard->cols = frame.cols;
    out_shard->stride = frame.stride;
    out_shard->source_node = frame.source_node;
    out_shard->source_device = frame.source_device;
    out_shard->shard_id = frame.shard_id;
    out_shard->session_id = frame.session_id;
    out_shard->content_hash = frame.content_hash;
    out_shard->payload_bytes = frame.payload_bytes;
    out_shard->flags = frame.flags;
    return AXIOM_OK;
}

int axiom_cluster_goal_begin(
        axiom_cluster *cluster,
        const axiom_goal_config *config) {
    if (!cluster || !config || config->abi_version != AXIOM_ABI_VERSION ||
        !axiom_cstr_nonempty_fits(config->goal_id, AXIOM_MAX_GOAL_ID) ||
        !axiom_cstr_nonempty_fits(config->objective, AXIOM_MAX_AGENT_OBJECTIVE)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    cluster->goal_id = config->goal_id;
    cluster->goal_objective = config->objective;
    cluster->goal_budget_tokens = config->budget_tokens;
    cluster->goal_flags = config->flags;
    cluster->goal_active = true;
    if (!axiom_cluster_has_stream_peer(cluster)) return AXIOM_OK;

    int rc = axiom_cluster_ensure_ack(cluster);
    if (rc != AXIOM_OK) return rc;

    axiom_agent_spawn_info info;
    std::memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    info.kind = AXIOM_AGENT_COORDINATOR;
    info.status = AXIOM_AGENT_DISPATCHED;
    info.placement = AXIOM_PLACEMENT_CLUSTER_MESH;
    info.source_node = cluster->node_id;
    info.source_device = 0;
    info.target_node = cluster->node_id;
    info.target_device = 0;
    info.budget_tokens = config->budget_tokens;
    info.flags = config->flags;
    axiom_wire_store_cstr((uint8_t *)info.goal_id, sizeof(info.goal_id), config->goal_id);
    axiom_wire_store_cstr((uint8_t *)info.agent_id, sizeof(info.agent_id), "goal");
    axiom_wire_store_cstr((uint8_t *)info.role, sizeof(info.role), "coordinator");
    axiom_wire_store_cstr((uint8_t *)info.objective, sizeof(info.objective), config->objective);

    uint8_t wire[AXIOM_CLUSTER_AGENT_WIRE_BYTES];
    axiom_agent_wire_encode(&info, wire);
    axiom_cluster_wire_frame frame;
    std::memset(&frame, 0, sizeof(frame));
    frame.kind = AXIOM_CLUSTER_FRAME_GOAL;
    frame.flags = (uint32_t)(config->flags & 0xffffffffu);
    frame.sequence = ++cluster->sequence;
    frame.payload_bytes = sizeof(wire);
    frame.session_id = 0;
    frame.content_hash = axiom_fnv1a64(wire, sizeof(wire));
    frame.dtype = (uint32_t)info.kind;
    frame.rows = cluster->node_count;
    frame.cols = 0;
    frame.stride = (uint32_t)info.placement;
    frame.source_node = cluster->node_id;
    frame.source_device = 0;
    return axiom_wire_write_frame(cluster, &frame, wire);
}

int axiom_cluster_spawn_agent(
        axiom_cluster *cluster,
        const axiom_agent_spawn_config *config,
        axiom_agent_spawn_info *out) {
    if (!cluster || !config || config->abi_version != AXIOM_ABI_VERSION ||
        !axiom_agent_kind_valid(config->kind) ||
        !axiom_cstr_nonempty_fits(config->agent_id, AXIOM_MAX_AGENT_ID) ||
        !axiom_cstr_nonempty_fits(config->goal_id, AXIOM_MAX_GOAL_ID) ||
        !axiom_cstr_nonempty_fits(config->role, AXIOM_MAX_AGENT_ROLE) ||
        ((!config->objective && !cluster->goal_active) ||
         (config->objective &&
          !axiom_cstr_nonempty_fits(config->objective, AXIOM_MAX_AGENT_OBJECTIVE))) ||
        config->placement.abi_version != AXIOM_ABI_VERSION ||
        !axiom_placement_kind_valid(config->placement.kind) ||
        config->target_node >= cluster->node_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (out && out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const bool has_stream = axiom_cluster_has_stream_peer(cluster);
    int rc = AXIOM_OK;
    if (has_stream) {
        rc = axiom_cluster_ensure_ack(cluster);
        if (rc != AXIOM_OK) return rc;
    }

    axiom_agent_spawn_info info;
    std::memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    info.kind = config->kind;
    info.status = has_stream ? AXIOM_AGENT_DISPATCHED : AXIOM_AGENT_CREATED;
    info.placement = config->placement.kind;
    info.source_node = cluster->node_id;
    info.source_device = config->placement.device_id;
    info.target_node = config->target_node;
    info.target_device = config->target_device;
    info.task_id = config->task_id;
    info.session_id = config->session_id;
    info.budget_tokens = config->budget_tokens;
    info.flags = config->flags;
    axiom_wire_store_cstr((uint8_t *)info.agent_id, sizeof(info.agent_id), config->agent_id);
    axiom_wire_store_cstr((uint8_t *)info.goal_id, sizeof(info.goal_id), config->goal_id);
    axiom_wire_store_cstr((uint8_t *)info.role, sizeof(info.role), config->role);
    axiom_wire_store_cstr(
            (uint8_t *)info.objective,
            sizeof(info.objective),
            config->objective ? config->objective : cluster->goal_objective.c_str());

    if (has_stream) {
        uint8_t wire[AXIOM_CLUSTER_AGENT_WIRE_BYTES];
        axiom_agent_wire_encode(&info, wire);
        axiom_cluster_wire_frame frame;
        std::memset(&frame, 0, sizeof(frame));
        frame.kind = AXIOM_CLUSTER_FRAME_AGENT_SPAWN;
        frame.flags = (uint32_t)(config->flags & 0xffffffffu);
        frame.sequence = ++cluster->sequence;
        frame.payload_bytes = sizeof(wire);
        frame.shard_id = config->task_id;
        frame.session_id = config->session_id;
        frame.content_hash = axiom_fnv1a64(wire, sizeof(wire));
        frame.dtype = (uint32_t)config->kind;
        frame.rows = config->target_node;
        frame.cols = config->target_device;
        frame.stride = (uint32_t)config->placement.kind;
        frame.source_node = cluster->node_id;
        frame.source_device = config->placement.device_id;
        info.sequence = frame.sequence;
        rc = axiom_wire_write_frame(cluster, &frame, wire);
        if (rc != AXIOM_OK) return rc;
    } else {
        info.sequence = ++cluster->sequence;
    }
    cluster->agents.push_back(info);
    if (out) {
        const uint32_t requested_abi = out->abi_version;
        *out = info;
        out->abi_version = requested_abi;
    }
    return AXIOM_OK;
}

int axiom_cluster_recv_agent_spawn(
        axiom_cluster *cluster,
        axiom_agent_spawn_info *out) {
    if (!cluster || !out || out->abi_version != AXIOM_ABI_VERSION ||
        !axiom_cluster_transport_is_stream(cluster->transport) ||
        !axiom_cluster_has_stream_peer(cluster)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    int rc = axiom_cluster_ensure_ack(cluster);
    if (rc != AXIOM_OK) return rc;
    for (;;) {
        axiom_cluster_wire_frame frame;
        rc = axiom_wire_read_frame(cluster, &frame);
        if (rc != AXIOM_OK) return rc;
        if (frame.kind != AXIOM_CLUSTER_FRAME_GOAL &&
            frame.kind != AXIOM_CLUSTER_FRAME_AGENT_SPAWN) {
            if (frame.payload_bytes > 0) (void)axiom_cluster_drain(cluster, frame.payload_bytes);
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (frame.payload_bytes != AXIOM_CLUSTER_AGENT_WIRE_BYTES) {
            if (frame.payload_bytes > 0) (void)axiom_cluster_drain(cluster, frame.payload_bytes);
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        uint8_t wire[AXIOM_CLUSTER_AGENT_WIRE_BYTES];
        rc = axiom_cluster_read_all(cluster, wire, sizeof(wire));
        if (rc != AXIOM_OK) return rc;
        if (frame.content_hash != 0 && axiom_fnv1a64(wire, sizeof(wire)) != frame.content_hash) {
            return AXIOM_ERR_IO;
        }
        axiom_agent_spawn_info info;
        info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_agent_wire_decode(wire, sizeof(wire), &info);
        if (rc != AXIOM_OK) return rc;
        info.sequence = frame.sequence;
        info.source_node = frame.source_node;
        info.source_device = frame.source_device;
        if (frame.kind == AXIOM_CLUSTER_FRAME_GOAL) {
            cluster->goal_id = info.goal_id;
            cluster->goal_objective = info.objective;
            cluster->goal_budget_tokens = info.budget_tokens;
            cluster->goal_flags = info.flags;
            cluster->goal_active = true;
            continue;
        }
        info.status = AXIOM_AGENT_RECEIVED;
        cluster->agents.push_back(info);
        const uint32_t requested_abi = out->abi_version;
        *out = info;
        out->abi_version = requested_abi;
        return AXIOM_OK;
    }
}

int axiom_runtime_attach_cluster(axiom_runtime *runtime, axiom_cluster *cluster) {
    if (!runtime || !cluster) return AXIOM_ERR_INVALID_ARGUMENT;
    if (cluster->transport != AXIOM_TRANSPORT_INPROC &&
        cluster->transport != AXIOM_TRANSPORT_SHM &&
        cluster->transport != AXIOM_TRANSPORT_TCP &&
        cluster->transport != AXIOM_TRANSPORT_RDMA &&
        cluster->transport != AXIOM_TRANSPORT_QUIC) {
        return AXIOM_ERR_NOT_IMPLEMENTED;
    }
    if (runtime->cluster == cluster) return AXIOM_OK;
    axiom_cluster_retain(cluster);
    axiom_cluster_release(runtime->cluster);
    runtime->cluster = cluster;
    return AXIOM_OK;
}

static bool axiom_model_budget_includes_file(const std::filesystem::path &path, axiom_model_format format) {
    const std::string ext = path.extension().string();
    switch (format) {
    case AXIOM_MODEL_FORMAT_SAFETENSORS:
        return ext == ".safetensors";
    case AXIOM_MODEL_FORMAT_GGUF:
        return ext == ".gguf";
    default:
        return true;
    }
}

int axiom_model_open(
        axiom_runtime *runtime,
        axiom_model **out,
        const axiom_model_config *config) {
    if (!runtime || !out || !config || config->abi_version != AXIOM_ABI_VERSION ||
        !config->path || !config->path[0]) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    std::error_code ec;
    const std::filesystem::path p(config->path);
    uint64_t bytes = 0;
    if (std::filesystem::is_regular_file(p, ec)) {
        bytes = (uint64_t)std::filesystem::file_size(p, ec);
        if (ec) return AXIOM_ERR_IO;
    } else if (std::filesystem::is_directory(p, ec)) {
        for (const auto &entry : std::filesystem::recursive_directory_iterator(p, ec)) {
            if (ec) return AXIOM_ERR_IO;
            if (!entry.is_regular_file(ec)) continue;
            if (!axiom_model_budget_includes_file(entry.path(), config->format)) continue;
            bytes += (uint64_t)entry.file_size(ec);
            if (ec) return AXIOM_ERR_IO;
        }
    } else {
        return AXIOM_ERR_IO;
    }
    if (config->memory_budget_bytes != 0 && bytes > config->memory_budget_bytes) {
        return AXIOM_ERR_BUDGET;
    }

    axiom_model *model = new (std::nothrow) axiom_model();
    if (!model) return AXIOM_ERR_RUNTIME;
    model->runtime = runtime;
    model->resident_enabled = (std::getenv("AXIOM_MODEL_NO_RESIDENT") == nullptr);
    model->path = config->path;
    model->name = config->name && config->name[0] ? config->name : p.filename().string();
    model->format = config->format;
    model->bytes = bytes;
    model->memory_budget_bytes = config->memory_budget_bytes;
    model->max_context = config->max_context;
    if (std::filesystem::is_directory(p, ec)) {
        axiom_model_load_config(model, p);
        axiom_model_load_weight_index(model, p);
        for (const auto &entry : std::filesystem::directory_iterator(p, ec)) {
            if (ec) {
                delete model;
                return AXIOM_ERR_IO;
            }
            if (!entry.is_regular_file(ec)) continue;
            if (entry.path().extension() == ".safetensors") {
                model->safetensors_file_count++;
                model->safetensors_bytes += (uint64_t)entry.file_size(ec);
                if (ec) {
                    delete model;
                    return AXIOM_ERR_IO;
                }
            }
        }
        const int rc = axiom_model_load_safetensors_headers(model, p);
        if (rc != AXIOM_OK) {
            delete model;
            return rc;
        }
    }
    runtime->model_count++;
    *out = model;
    return AXIOM_OK;
}

void axiom_model_close(axiom_model *model) {
    if (!model) return;
    // P4-C: free the model-owned device-resident weight buffers (VRAM back).
    for (auto &entry : model->device_resident) {
        axiom_device_buffer_destroy(entry.second);
    }
    model->device_resident.clear();
    if (model->runtime && model->runtime->model_count > 0) model->runtime->model_count--;
    delete model;
}

static void axiom_copy_string(char *dst, size_t cap, const std::string &src) {
    if (!dst || cap == 0) return;
    const size_t n = src.size() < cap - 1 ? src.size() : cap - 1;
    std::memcpy(dst, src.data(), n);
    dst[n] = '\0';
}

int axiom_model_info_get(axiom_model *model, axiom_model_info *out) {
    if (!model || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    axiom_copy_string(out->name, sizeof(out->name), model->name);
    axiom_copy_string(out->path, sizeof(out->path), model->path);
    axiom_copy_string(out->model_type, sizeof(out->model_type), model->model_type);
    axiom_copy_string(out->architecture, sizeof(out->architecture), model->architecture);
    out->format = model->format;
    out->bytes = model->bytes;
    out->safetensors_bytes = model->safetensors_bytes;
    out->weight_index_total_bytes = model->weight_index_total_bytes;
    out->safetensors_header_data_bytes = model->safetensors_header_data_bytes;
    out->memory_budget_bytes = model->memory_budget_bytes;
    out->safetensors_file_count = model->safetensors_file_count;
    out->weight_tensor_count = model->weight_tensor_count;
    out->safetensors_header_tensor_count = model->safetensors_header_tensor_count;
    out->hidden_size = model->hidden_size;
    out->intermediate_size = model->intermediate_size;
    out->num_hidden_layers = model->num_hidden_layers;
    out->num_attention_heads = model->num_attention_heads;
    out->num_key_value_heads = model->num_key_value_heads;
    out->vocab_size = model->vocab_size;
    out->max_context = model->max_context;
    out->entity_count = model->entity_count;
    return AXIOM_OK;
}

int axiom_model_tensor_info_get(
        axiom_model *model,
        const char *name,
        axiom_tensor_info *out) {
    if (!model || !name || !name[0] || !out ||
        out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const axiom_tensor_record *found = axiom_model_find_tensor(model, name);
    if (!found) return AXIOM_ERR_IO;

    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    axiom_copy_string(out->name, sizeof(out->name), found->name);
    axiom_copy_string(out->file, sizeof(out->file), found->file);
    out->dtype = found->dtype;
    out->rank = found->rank;
    for (uint32_t i = 0; i < found->rank && i < AXIOM_MAX_TENSOR_DIMS; ++i) {
        out->shape[i] = found->shape[i];
    }
    out->data_offset_begin = found->data_offset_begin;
    out->data_offset_end = found->data_offset_end;
    out->file_offset_begin = found->file_offset_begin;
    out->file_offset_end = found->file_offset_end;
    out->byte_count = found->byte_count;
    return AXIOM_OK;
}

int axiom_model_tensor_read(
        axiom_model *model,
        const char *name,
        void *out_data,
        uint64_t out_capacity,
        uint64_t *out_bytes) {
    if (!model || !name || !name[0] || !out_data || !out_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_bytes = 0;
    const axiom_tensor_record *found = axiom_model_find_tensor(model, name);
    if (!found) return AXIOM_ERR_IO;
    if (out_capacity < found->byte_count) return AXIOM_ERR_BUDGET;
    const int rc = axiom_model_tensor_read_slice(
            model,
            name,
            0,
            out_data,
            found->byte_count);
    if (rc != AXIOM_OK) return rc;
    *out_bytes = found->byte_count;
    return AXIOM_OK;
}

// Residency core (criterion #1/#2): ensure the WHOLE tensor `found` is loaded
// ONCE into model->resident, then return a pointer to the resident buffer. This
// is the SINGLE disk-read path shared by both per-tensor slice reads and the
// generic per-node layer-span residency API — the resident bytes are exactly
// what the disk read returns, so numerics stay bit-identical for every reader.
// The caller MUST hold model->resident_mutex.
static int axiom_model_resident_ensure_locked(
        axiom_model *model,
        const axiom_tensor_record *found,
        std::vector<uint8_t> **out_buf) {
    auto it = model->resident.find(found->name);
    if (it == model->resident.end()) {
        std::vector<uint8_t> buf;
        try { buf.resize(found->byte_count); } catch (...) { return AXIOM_ERR_IO; }
        const std::filesystem::path rp =
                std::filesystem::path(model->path) / found->file;
        std::ifstream rf(rp, std::ios::binary);
        if (!rf) return AXIOM_ERR_IO;
        rf.seekg((std::streamoff)found->file_offset_begin, std::ios::beg);
        if (!rf) return AXIOM_ERR_IO;
        rf.read((char *)buf.data(), (std::streamsize)found->byte_count);
        if ((uint64_t)rf.gcount() != found->byte_count) return AXIOM_ERR_IO;
        it = model->resident.emplace(found->name, std::move(buf)).first;
    }
    if (out_buf) *out_buf = &it->second;
    return AXIOM_OK;
}

int axiom_model_tensor_read_slice(
        axiom_model *model,
        const char *name,
        uint64_t byte_offset,
        void *out_data,
        uint64_t byte_count) {
    if (!model || !name || !name[0] || !out_data || byte_count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const axiom_tensor_record *found = axiom_model_find_tensor(model, name);
    if (!found) return AXIOM_ERR_IO;
    if (byte_offset > found->byte_count ||
        byte_count > found->byte_count - byte_offset) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    // Residency (criterion #1): load the WHOLE tensor once, then serve all
    // reads (full or slice) from the resident buffer — no per-op disk re-read.
    // Byte-transparent: the resident bytes are exactly what the disk read
    // returns, so numerics stay bit-identical. (Disk IO is done under the lock;
    // the worker request loop is single-threaded today — see resident_mutex.)
    if (model->resident_enabled) {
        std::lock_guard<std::mutex> lk(model->resident_mutex);
        std::vector<uint8_t> *buf = nullptr;
        const int rc = axiom_model_resident_ensure_locked(model, found, &buf);
        if (rc != AXIOM_OK) return rc;
        std::memcpy(out_data, buf->data() + byte_offset, (size_t)byte_count);
        return AXIOM_OK;
    }

    const std::filesystem::path tensor_path =
            std::filesystem::path(model->path) / found->file;
    std::ifstream f(tensor_path, std::ios::binary);
    if (!f) return AXIOM_ERR_IO;
    f.seekg((std::streamoff)(found->file_offset_begin + byte_offset), std::ios::beg);
    if (!f) return AXIOM_ERR_IO;
    f.read((char *)out_data, (std::streamsize)byte_count);
    if ((uint64_t)f.gcount() != byte_count) return AXIOM_ERR_IO;
    return AXIOM_OK;
}

// Generic (model-agnostic) per-node layer-span residency (criterion #2): load
// ONLY the tensors whose transformer layer index falls in [layer_start,
// layer_end] into the residency cache, so a node holds just its own shard of
// the model. Layer parsing is family-agnostic: it locates the ".layers."
// substring anywhere in the tensor name and parses the integer immediately
// after it ("model.layers.5.*", "model.language_model.layers.20.*", ...).
// Tensors with no ".layers." segment (embed_tokens, lm_head, final norm — the
// shared tail) are SKIPPED here; they are not part of any node's layer span.
// Bytes are loaded through the same single disk-read path as slice reads, so
// residency stays byte-identical. A running total of span bytes is enforced
// against max_resident_bytes (0 or UINT64_MAX == unlimited).
int axiom_model_reside_layer_span(
        axiom_model *model,
        uint32_t layer_start,
        uint32_t layer_end,
        uint64_t max_resident_bytes,
        uint64_t *out_resident_bytes,
        uint32_t *out_tensor_count) {
    if (out_resident_bytes) *out_resident_bytes = 0;
    if (out_tensor_count) *out_tensor_count = 0;
    if (!model || layer_end < layer_start) return AXIOM_ERR_INVALID_ARGUMENT;

    const uint64_t limit =
            (max_resident_bytes == 0) ? UINT64_MAX : max_resident_bytes;

    std::lock_guard<std::mutex> lk(model->resident_mutex);
    uint64_t total = 0;
    uint32_t count = 0;
    for (const auto &rec : model->tensors) {
        const std::string &name = rec.name;
        const size_t marker = name.find(".layers.");
        if (marker == std::string::npos) continue;  // shared tail, not a span
        const char *digits = name.c_str() + marker + 8;  // strlen(".layers.")
        if (*digits < '0' || *digits > '9') continue;     // no numeric index
        char *endp = nullptr;
        const unsigned long layer = std::strtoul(digits, &endp, 10);
        if (endp == digits) continue;
        if (layer < layer_start || layer > layer_end) continue;

        // Running budget: reject before reading if this tensor would exceed it.
        if (rec.byte_count > limit || total > limit - rec.byte_count) {
            return AXIOM_ERR_BUDGET;
        }
        std::vector<uint8_t> *buf = nullptr;
        const int rc = axiom_model_resident_ensure_locked(model, &rec, &buf);
        if (rc != AXIOM_OK) return rc;
        total += rec.byte_count;
        ++count;
    }

    if (out_resident_bytes) *out_resident_bytes = total;
    if (out_tensor_count) *out_tensor_count = count;
    return AXIOM_OK;
}

// P4-C: DEVICE twin of host residency — upload the named tensor ONCE to the
// device. The bytes are sourced through the SAME read path as every other
// consumer (host-resident buffer when enabled, single disk read otherwise), so
// the device bytes are exactly the disk bytes: residence changes, bytes don't.
// The buffer is model-owned (freed at axiom_model_close). Idempotent.
int axiom_model_tensor_device_resident(
        axiom_model *model,
        const char *name,
        uint64_t *out_bytes) {
    if (out_bytes) *out_bytes = 0;
    if (!model || !name || !name[0]) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const axiom_tensor_record *found = axiom_model_find_tensor(model, name);
    if (!found) return AXIOM_ERR_IO;
    {
        std::lock_guard<std::mutex> lk(model->resident_mutex);
        if (model->device_resident.find(found->name) != model->device_resident.end()) {
            if (out_bytes) *out_bytes = found->byte_count;
            return AXIOM_OK;  // already device-resident: upload-once means once
        }
    }

    axiom_device_buffer *buffer = nullptr;
    int rc = axiom_device_buffer_create(model->runtime, &buffer, found->byte_count);
    if (rc != AXIOM_OK) return rc;

    if (model->resident_enabled) {
        // Upload straight from the host-resident bytes (no temp copy).
        std::lock_guard<std::mutex> lk(model->resident_mutex);
        std::vector<uint8_t> *host_buf = nullptr;
        rc = axiom_model_resident_ensure_locked(model, found, &host_buf);
        if (rc == AXIOM_OK) {
            rc = axiom_device_buffer_upload(buffer, 0, host_buf->data(), found->byte_count);
        }
    } else {
        // Per-op-read mode (AXIOM_MODEL_NO_RESIDENT=1): one disk read into a
        // transient host buffer, then the one-time upload.
        std::vector<uint8_t> tmp;
        try { tmp.resize((size_t)found->byte_count); } catch (...) { rc = AXIOM_ERR_IO; }
        if (rc == AXIOM_OK) {
            rc = axiom_model_tensor_read_slice(model, name, 0, tmp.data(), found->byte_count);
        }
        if (rc == AXIOM_OK) {
            rc = axiom_device_buffer_upload(buffer, 0, tmp.data(), found->byte_count);
        }
    }
    if (rc != AXIOM_OK) {
        axiom_device_buffer_destroy(buffer);
        return rc;
    }

    {
        std::lock_guard<std::mutex> lk(model->resident_mutex);
        const auto inserted = model->device_resident.emplace(found->name, buffer);
        if (!inserted.second) {
            // Lost a (theoretical) race: keep the first buffer, drop ours.
            axiom_device_buffer_destroy(buffer);
        }
        model->device_resident_any.store(true, std::memory_order_release);
    }
    if (out_bytes) *out_bytes = found->byte_count;
    return AXIOM_OK;
}

// P4-C introspection: how many linear/matvec calls went through the per-op
// host-upload path vs the device-resident path. The gate's profile proof:
// with a fully device-resident span, host_path_calls stays at ZERO.
int axiom_model_linear_path_counters(
        axiom_model *model,
        uint64_t *out_host_path_calls,
        uint64_t *out_device_resident_calls) {
    if (!model) return AXIOM_ERR_INVALID_ARGUMENT;
    if (out_host_path_calls) {
        *out_host_path_calls =
                model->linear_host_path_calls.load(std::memory_order_relaxed);
    }
    if (out_device_resident_calls) {
        *out_device_resident_calls =
                model->linear_device_resident_calls.load(std::memory_order_relaxed);
    }
    return AXIOM_OK;
}

// P4-C helper: device-resident lookup for a weight about to be matvec'd.
// Lock-free when nothing is resident (the default path pays ~nothing).
static axiom_device_buffer *axiom_model_device_resident_find(
        axiom_model *model,
        const char *name) {
    if (!model->device_resident_any.load(std::memory_order_acquire)) return nullptr;
    std::lock_guard<std::mutex> lk(model->resident_mutex);
    const auto it = model->device_resident.find(name);
    return it == model->device_resident.end() ? nullptr : it->second;
}

int axiom_model_embed_token_f32(
        axiom_model *model,
        uint32_t token_id,
        float *out_host,
        uint32_t out_count) {
    if (!model || !out_host || out_count == 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const char *embedding_name = "model.embed_tokens.weight";
    const axiom_tensor_record *embedding =
            axiom_model_find_tensor(model, embedding_name);
    if (!embedding) {
        embedding_name = "model.language_model.embed_tokens.weight";
        embedding = axiom_model_find_tensor(model, embedding_name);
    }
    if (!embedding ||
        embedding->dtype != AXIOM_TENSOR_DTYPE_BF16 ||
        embedding->rank != 2 ||
        embedding->shape[0] == 0 ||
        embedding->shape[1] == 0 ||
        token_id >= embedding->shape[0] ||
        out_count < embedding->shape[1]) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t hidden = embedding->shape[1];
    const uint64_t row_bytes = hidden * sizeof(uint16_t);
    if (row_bytes > std::numeric_limits<size_t>::max()) return AXIOM_ERR_BUDGET;
    std::vector<uint16_t> row((size_t)hidden);
    const int rc = axiom_model_tensor_read_slice(
            model,
            embedding_name,
            (uint64_t)token_id * row_bytes,
            row.data(),
            row_bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_bf16_to_f32(
            model->runtime->backend_runtime,
            row.data(),
            out_host,
            (size_t)hidden);
}

int axiom_model_linear_bf16_f32(
        axiom_model *model,
        const char *weight_name,
        const char *bias_name,
        const float *input_host,
        uint32_t input_count,
        float *out_host,
        uint32_t out_count) {
    if (!model || !weight_name || !weight_name[0] || !input_host || !out_host) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const axiom_tensor_record *weight = axiom_model_find_tensor(model, weight_name);
    if (!weight || weight->dtype != AXIOM_TENSOR_DTYPE_BF16 || weight->rank != 2) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t rows64 = weight->shape[0];
    const uint64_t cols64 = weight->shape[1];
    if (rows64 == 0 || cols64 == 0 ||
        rows64 > std::numeric_limits<uint32_t>::max() ||
        cols64 > std::numeric_limits<uint32_t>::max() ||
        input_count < cols64 ||
        out_count < rows64 ||
        weight->byte_count != rows64 * cols64 * sizeof(uint16_t)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    // P4-C: device-resident fast path — the weight bytes already live on the
    // device (uploaded once); only the per-call weight upload is skipped. The
    // weight is NOT read to host at all. SAME kernel, same launch config.
    axiom_device_buffer *resident_weight =
            axiom_model_device_resident_find(model, weight_name);

    std::vector<uint16_t> weight_host;
    if (!resident_weight) {
        weight_host.resize((size_t)(weight->byte_count / sizeof(uint16_t)));
        uint64_t weight_bytes = 0;
        const int rc = axiom_model_tensor_read(
                model,
                weight_name,
                weight_host.data(),
                weight->byte_count,
                &weight_bytes);
        if (rc != AXIOM_OK) return rc;
    }

    std::vector<uint16_t> bias_host;
    const uint16_t *bias_ptr = nullptr;
    if (bias_name && bias_name[0]) {
        const axiom_tensor_record *bias = axiom_model_find_tensor(model, bias_name);
        if (!bias || bias->dtype != AXIOM_TENSOR_DTYPE_BF16 || bias->rank != 1 ||
            bias->shape[0] != weight->shape[0] ||
            bias->byte_count != weight->shape[0] * sizeof(uint16_t)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
        uint64_t bias_bytes = 0;
        bias_host.resize((size_t)(bias->byte_count / sizeof(uint16_t)));
        const int rc = axiom_model_tensor_read(
                model,
                bias_name,
                bias_host.data(),
                bias->byte_count,
                &bias_bytes);
        if (rc != AXIOM_OK) return rc;
        bias_ptr = bias_host.data();
    }

    if (resident_weight) {
        model->linear_device_resident_calls.fetch_add(1, std::memory_order_relaxed);
        return axiom_cuda_bf16_matvec_f32_resident(
                model->runtime->backend_runtime,
                resident_weight->backend_buffer,
                0,
                bias_ptr,
                input_host,
                out_host,
                (uint32_t)weight->shape[0],
                (uint32_t)weight->shape[1]);
    }
    model->linear_host_path_calls.fetch_add(1, std::memory_order_relaxed);
    return axiom_cuda_bf16_matvec_f32(
            model->runtime->backend_runtime,
            weight_host.data(),
            bias_ptr,
            input_host,
            out_host,
            (uint32_t)weight->shape[0],
            (uint32_t)weight->shape[1]);
}

int axiom_model_linear_bf16_rank3_slice_f32(
        axiom_model *model,
        const char *weight_name,
        uint32_t slice,
        const float *input_host,
        uint32_t input_count,
        float *out_host,
        uint32_t out_count) {
    if (!model || !weight_name || !weight_name[0] || !input_host || !out_host) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const axiom_tensor_record *weight = axiom_model_find_tensor(model, weight_name);
    if (!weight || weight->dtype != AXIOM_TENSOR_DTYPE_BF16 || weight->rank != 3) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t slices64 = weight->shape[0];
    const uint64_t rows64 = weight->shape[1];
    const uint64_t cols64 = weight->shape[2];
    if (slices64 == 0 || rows64 == 0 || cols64 == 0 ||
        slice >= slices64 ||
        rows64 > std::numeric_limits<uint32_t>::max() ||
        cols64 > std::numeric_limits<uint32_t>::max() ||
        input_count < cols64 ||
        out_count < rows64 ||
        weight->byte_count != slices64 * rows64 * cols64 * sizeof(uint16_t)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t slice_bytes = rows64 * cols64 * sizeof(uint16_t);

    // P4-C: device-resident fast path — the WHOLE rank-3 slab lives on the
    // device; the slice is a byte-offset view into the resident buffer. SAME
    // kernel, same launch config, no per-call slice upload.
    axiom_device_buffer *resident_weight =
            axiom_model_device_resident_find(model, weight_name);
    if (resident_weight) {
        model->linear_device_resident_calls.fetch_add(1, std::memory_order_relaxed);
        return axiom_cuda_bf16_matvec_f32_resident(
                model->runtime->backend_runtime,
                resident_weight->backend_buffer,
                (uint64_t)slice * slice_bytes,
                nullptr,
                input_host,
                out_host,
                (uint32_t)rows64,
                (uint32_t)cols64);
    }

    std::vector<uint16_t> weight_host((size_t)(slice_bytes / sizeof(uint16_t)));
    const int rc = axiom_model_tensor_read_slice(
            model,
            weight_name,
            (uint64_t)slice * slice_bytes,
            weight_host.data(),
            slice_bytes);
    if (rc != AXIOM_OK) return rc;

    model->linear_host_path_calls.fetch_add(1, std::memory_order_relaxed);
    return axiom_cuda_bf16_matvec_f32(
            model->runtime->backend_runtime,
            weight_host.data(),
            nullptr,
            input_host,
            out_host,
            (uint32_t)rows64,
            (uint32_t)cols64);
}

int axiom_model_rmsnorm_f32(
        axiom_model *model,
        const char *weight_name,
        const float *input_host,
        float *out_host,
        uint32_t count,
        float eps) {
    if (!model || !weight_name || !weight_name[0] ||
        !input_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const axiom_tensor_record *weight = axiom_model_find_tensor(model, weight_name);
    if (!weight || weight->dtype != AXIOM_TENSOR_DTYPE_BF16 ||
        weight->rank != 1 || weight->shape[0] != count ||
        weight->byte_count != (uint64_t)count * sizeof(uint16_t)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<uint16_t> weight_host(count);
    uint64_t bytes = 0;
    const int rc = axiom_model_tensor_read(
            model,
            weight_name,
            weight_host.data(),
            weight->byte_count,
            &bytes);
    if (rc != AXIOM_OK) return rc;
    return axiom_cuda_rmsnorm_f32(
            model->runtime->backend_runtime,
            weight_host.data(),
            input_host,
            out_host,
            count,
            eps > 0.0f ? eps : 1.0e-6f);
}

int axiom_model_rope_f32(
        axiom_model *model,
        const float *input_host,
        float *out_host,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t position,
        float rope_theta) {
    if (!model || !input_host || !out_host || heads == 0 || head_dim == 0 ||
        (head_dim % 2) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_rope_f32(
            model->runtime->backend_runtime,
            input_host,
            out_host,
            heads,
            head_dim,
            position,
            rope_theta > 0.0f ? rope_theta : 1000000.0f);
}

int axiom_model_attention_single_f32(
        axiom_model *model,
        const float *q_host,
        const float *k_host,
        const float *v_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim) {
    if (!model || !q_host || !k_host || !v_host || !out_host ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_attention_single_f32(
            model->runtime->backend_runtime,
            q_host,
            k_host,
            v_host,
            out_host,
            q_heads,
            kv_heads,
            head_dim);
}

int axiom_model_attention_cache_f32(
        axiom_model *model,
        const float *q_host,
        const float *k_cache_host,
        const float *v_cache_host,
        float *out_host,
        uint32_t q_heads,
        uint32_t kv_heads,
        uint32_t head_dim,
        uint32_t cache_tokens) {
    if (!model || !q_host || !k_cache_host || !v_cache_host || !out_host ||
        q_heads == 0 || kv_heads == 0 || head_dim == 0 || cache_tokens == 0 ||
        (q_heads % kv_heads) != 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_attention_cache_f32(
            model->runtime->backend_runtime,
            q_host,
            k_cache_host,
            v_cache_host,
            out_host,
            q_heads,
            kv_heads,
            head_dim,
            cache_tokens);
}

int axiom_model_silu_mul_f32(
        axiom_model *model,
        const float *gate_host,
        const float *up_host,
        float *out_host,
        uint32_t count) {
    if (!model || !gate_host || !up_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_silu_mul_f32(
            model->runtime->backend_runtime,
            gate_host,
            up_host,
            out_host,
            count);
}

int axiom_model_topk_f32(
        axiom_model *model,
        const float *input_host,
        uint32_t count,
        uint32_t k,
        uint32_t *out_indices_host,
        float *out_values_host) {
    if (!model || !input_host || count == 0 || k == 0 || k > count ||
        !out_indices_host || !out_values_host) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!model->runtime || model->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_topk_f32(
            model->runtime->backend_runtime,
            input_host,
            count,
            k,
            out_indices_host,
            out_values_host);
}

int axiom_model_tied_lm_head_top1_f32(
        axiom_model *model,
        const float *hidden_host,
        uint32_t hidden_count,
        uint32_t *out_token_id,
        float *out_logit) {
    if (!model || !hidden_host || hidden_count == 0 || !out_token_id || !out_logit ||
        hidden_count != model->hidden_size || model->vocab_size == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::vector<float> logits(model->vocab_size);
    const int rc = axiom_model_linear_bf16_f32(
            model,
            "model.embed_tokens.weight",
            nullptr,
            hidden_host,
            hidden_count,
            logits.data(),
            model->vocab_size);
    if (rc != AXIOM_OK) return rc;

    uint32_t best_id = 0;
    float best = logits[0];
    for (uint32_t i = 1; i < model->vocab_size; ++i) {
        if (logits[i] > best) {
            best = logits[i];
            best_id = i;
        }
    }
    *out_token_id = best_id;
    *out_logit = best;
    return AXIOM_OK;
}

static uint64_t axiom_fnv1a_update(uint64_t hash, const unsigned char *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static int axiom_hash_file(
        const std::filesystem::path &path,
        uint64_t *out_bytes,
        uint64_t *out_hash) {
    if (!out_bytes || !out_hash) return AXIOM_ERR_INVALID_ARGUMENT;
    std::ifstream f(path, std::ios::binary);
    if (!f) return AXIOM_ERR_IO;

    uint64_t bytes = 0;
    uint64_t hash = 1469598103934665603ull;
    std::vector<unsigned char> buf(1 << 16);
    while (f) {
        f.read((char *)buf.data(), (std::streamsize)buf.size());
        const std::streamsize got = f.gcount();
        if (got > 0) {
            bytes += (uint64_t)got;
            hash = axiom_fnv1a_update(hash, buf.data(), (size_t)got);
        }
    }
    if (f.bad()) return AXIOM_ERR_IO;
    *out_bytes = bytes;
    *out_hash = hash;
    return AXIOM_OK;
}

static bool axiom_read_file(const std::filesystem::path &path, std::string *out) {
    if (!out) return false;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out->assign(
            std::istreambuf_iterator<char>(f),
            std::istreambuf_iterator<char>());
    return !f.bad();
}

static bool axiom_extract_json_string(
        const std::string &json,
        const char *key,
        std::string *out) {
    if (!out) return false;
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                  json[pos] == '\r' || json[pos] == '\n')) {
        ++pos;
    }
    if (pos >= json.size() || json[pos] != '"') return false;
    ++pos;

    std::string value;
    bool escaped = false;
    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (escaped) {
            switch (c) {
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            case '"':
            case '\\':
            case '/':
                value.push_back(c);
                break;
            default:
                value.push_back(c);
                break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            *out = value;
            return true;
        }
        value.push_back(c);
    }
    return false;
}

static uint32_t axiom_parse_u32_after(const std::string &s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                              s[pos] == '\r' || s[pos] == '\n' ||
                              s[pos] == ':')) {
        ++pos;
    }
    uint64_t v = 0;
    bool any = false;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        any = true;
        v = v * 10u + (uint32_t)(s[pos] - '0');
        if (v > std::numeric_limits<uint32_t>::max()) {
            return AXIOM_TOKEN_ID_INVALID;
        }
        ++pos;
    }
    return any ? (uint32_t)v : AXIOM_TOKEN_ID_INVALID;
}

static uint32_t axiom_extract_json_u32(const std::string &json, const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) return 0;
    const uint32_t value = axiom_parse_u32_after(json, colon + 1);
    return value == AXIOM_TOKEN_ID_INVALID ? 0 : value;
}

static uint64_t axiom_parse_u64_after(const std::string &s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                              s[pos] == '\r' || s[pos] == '\n' ||
                              s[pos] == ':')) {
        ++pos;
    }
    uint64_t v = 0;
    bool any = false;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        any = true;
        const uint64_t next = v * 10u + (uint32_t)(s[pos] - '0');
        if (next < v) return 0;
        v = next;
        ++pos;
    }
    return any ? v : 0;
}

static uint64_t axiom_extract_json_u64(const std::string &json, const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    const size_t colon = json.find(':', pos + needle.size());
    if (colon == std::string::npos) return 0;
    return axiom_parse_u64_after(json, colon + 1);
}

static uint32_t axiom_count_object_entries_after_key(
        const std::string &json,
        const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return 0;
    pos = json.find('{', pos + needle.size());
    if (pos == std::string::npos) return 0;

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    uint32_t count = 0;
    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            ++depth;
            continue;
        }
        if (c == '}') {
            --depth;
            if (depth == 0) break;
            continue;
        }
        if (c == ':' && depth == 1) ++count;
    }
    return count;
}

static size_t axiom_skip_ws(const std::string &s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                              s[pos] == '\r' || s[pos] == '\n' ||
                              s[pos] == ',')) {
        ++pos;
    }
    return pos;
}

static bool axiom_parse_json_string_at(
        const std::string &json,
        size_t *pos,
        std::string *out) {
    if (!pos || !out || *pos >= json.size() || json[*pos] != '"') return false;
    size_t i = *pos + 1;
    std::string value;
    bool escaped = false;
    for (; i < json.size(); ++i) {
        const char c = json[i];
        if (escaped) {
            switch (c) {
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            case '"':
            case '\\':
            case '/':
                value.push_back(c);
                break;
            default:
                value.push_back(c);
                break;
            }
            escaped = false;
            continue;
        }
        if (c == '\\') {
            escaped = true;
            continue;
        }
        if (c == '"') {
            *pos = i + 1;
            *out = std::move(value);
            return true;
        }
        value.push_back(c);
    }
    return false;
}

static size_t axiom_find_json_match(
        const std::string &json,
        size_t pos,
        char open,
        char close) {
    if (pos >= json.size() || json[pos] != open) return std::string::npos;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = pos; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open) {
            ++depth;
        } else if (c == close) {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

static bool axiom_extract_json_u64_array(
        const std::string &json,
        const char *key,
        uint64_t *out,
        uint32_t max_count,
        uint32_t *out_count) {
    if (!out || max_count == 0 || !out_count) return false;
    *out_count = 0;
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find('[', pos + needle.size());
    if (pos == std::string::npos) return false;
    const size_t end = axiom_find_json_match(json, pos, '[', ']');
    if (end == std::string::npos) return false;
    ++pos;
    pos = axiom_skip_ws(json, pos);
    if (pos == end) return true;  /* Safetensors scalar shape: []. */
    while (pos < end && *out_count < max_count) {
        pos = axiom_skip_ws(json, pos);
        if (pos >= end) break;
        if (json[pos] < '0' || json[pos] > '9') {
            ++pos;
            continue;
        }
        out[*out_count] = axiom_parse_u64_after(json, pos);
        ++(*out_count);
        while (pos < end && json[pos] >= '0' && json[pos] <= '9') ++pos;
    }
    return *out_count > 0;
}

static axiom_tensor_dtype axiom_tensor_dtype_from_string(const std::string &dtype) {
    if (dtype == "F32") return AXIOM_TENSOR_DTYPE_F32;
    if (dtype == "F16") return AXIOM_TENSOR_DTYPE_F16;
    if (dtype == "BF16") return AXIOM_TENSOR_DTYPE_BF16;
    if (dtype == "I64") return AXIOM_TENSOR_DTYPE_I64;
    if (dtype == "I32") return AXIOM_TENSOR_DTYPE_I32;
    if (dtype == "U8") return AXIOM_TENSOR_DTYPE_U8;
    if (dtype == "F8_E4M3" || dtype == "F8_E4M3FN") return AXIOM_TENSOR_DTYPE_F8_E4M3;
    if (dtype == "F8_E5M2") return AXIOM_TENSOR_DTYPE_F8_E5M2;
    return AXIOM_TENSOR_DTYPE_UNKNOWN;
}

static int axiom_read_safetensors_header(
        const std::filesystem::path &path,
        std::string *out_header,
        uint64_t *out_data_begin) {
    if (!out_header || !out_data_begin) return AXIOM_ERR_INVALID_ARGUMENT;
    std::ifstream f(path, std::ios::binary);
    if (!f) return AXIOM_ERR_IO;
    unsigned char len_bytes[8] = {0};
    f.read((char *)len_bytes, 8);
    if (f.gcount() != 8) return AXIOM_ERR_IO;
    uint64_t header_len = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        header_len |= ((uint64_t)len_bytes[i]) << (8u * i);
    }
    if (header_len == 0 || header_len > (256ull * 1024ull * 1024ull)) {
        return AXIOM_ERR_IO;
    }
    out_header->assign((size_t)header_len, '\0');
    f.read(out_header->data(), (std::streamsize)header_len);
    if ((uint64_t)f.gcount() != header_len) return AXIOM_ERR_IO;
    *out_data_begin = 8ull + header_len;
    return AXIOM_OK;
}

static bool axiom_parse_safetensors_tensor_block(
        const std::string &name,
        const std::string &file,
        const std::string &block,
        uint64_t file_data_begin,
        axiom_tensor_record *out) {
    if (!out) return false;
    std::string dtype_s;
    uint64_t shape[AXIOM_MAX_TENSOR_DIMS] = {0};
    uint64_t offsets[2] = {0};
    uint32_t rank = 0;
    uint32_t offset_count = 0;
    if (!axiom_extract_json_string(block, "dtype", &dtype_s) ||
        !axiom_extract_json_u64_array(block, "shape", shape, AXIOM_MAX_TENSOR_DIMS, &rank) ||
        !axiom_extract_json_u64_array(block, "data_offsets", offsets, 2, &offset_count) ||
        offset_count != 2 || offsets[1] < offsets[0]) {
        return false;
    }

    out->name = name;
    out->file = file;
    out->dtype = axiom_tensor_dtype_from_string(dtype_s);
    out->rank = rank;
    for (uint32_t i = 0; i < AXIOM_MAX_TENSOR_DIMS; ++i) {
        out->shape[i] = 0;
    }
    for (uint32_t i = 0; i < rank && i < AXIOM_MAX_TENSOR_DIMS; ++i) {
        out->shape[i] = shape[i];
    }
    out->data_offset_begin = offsets[0];
    out->data_offset_end = offsets[1];
    out->file_offset_begin = file_data_begin + offsets[0];
    out->file_offset_end = file_data_begin + offsets[1];
    out->byte_count = offsets[1] - offsets[0];
    return true;
}

static int axiom_parse_safetensors_header(
        axiom_model *model,
        const std::filesystem::path &path,
        const std::string &header,
        uint64_t file_data_begin) {
    if (!model) return AXIOM_ERR_INVALID_ARGUMENT;
    size_t pos = axiom_skip_ws(header, 0);
    if (pos >= header.size() || header[pos] != '{') return AXIOM_ERR_IO;
    ++pos;
    const std::string file = path.filename().string();
    uint64_t max_data_end = 0;

    while (pos < header.size()) {
        pos = axiom_skip_ws(header, pos);
        if (pos >= header.size() || header[pos] == '}') break;
        std::string key;
        if (!axiom_parse_json_string_at(header, &pos, &key)) return AXIOM_ERR_IO;
        pos = axiom_skip_ws(header, pos);
        if (pos >= header.size() || header[pos] != ':') return AXIOM_ERR_IO;
        pos = axiom_skip_ws(header, pos + 1);
        if (pos >= header.size()) return AXIOM_ERR_IO;

        if (header[pos] == '{') {
            const size_t end = axiom_find_json_match(header, pos, '{', '}');
            if (end == std::string::npos) return AXIOM_ERR_IO;
            if (key != "__metadata__") {
                axiom_tensor_record tensor;
                if (axiom_parse_safetensors_tensor_block(
                            key,
                            file,
                            header.substr(pos, end - pos + 1),
                            file_data_begin,
                            &tensor)) {
                    if (tensor.data_offset_end > max_data_end) {
                        max_data_end = tensor.data_offset_end;
                    }
                    model->tensors.push_back(std::move(tensor));
                }
            }
            pos = end + 1;
        } else if (header[pos] == '[') {
            const size_t end = axiom_find_json_match(header, pos, '[', ']');
            if (end == std::string::npos) return AXIOM_ERR_IO;
            pos = end + 1;
        } else if (header[pos] == '"') {
            std::string ignored;
            if (!axiom_parse_json_string_at(header, &pos, &ignored)) return AXIOM_ERR_IO;
        } else {
            while (pos < header.size() && header[pos] != ',' && header[pos] != '}') ++pos;
        }
    }

    model->safetensors_header_data_bytes += max_data_end;
    return AXIOM_OK;
}

static int axiom_model_load_safetensors_headers(
        axiom_model *model,
        const std::filesystem::path &dir) {
    if (!model) return AXIOM_ERR_INVALID_ARGUMENT;
    std::error_code ec;
    std::vector<std::filesystem::path> shards;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) return AXIOM_ERR_IO;
        if (!entry.is_regular_file(ec)) continue;
        if (entry.path().extension() == ".safetensors") {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    model->tensors.clear();
    model->safetensors_header_data_bytes = 0;

    for (const auto &shard : shards) {
        std::string header;
        uint64_t file_data_begin = 0;
        int rc = axiom_read_safetensors_header(shard, &header, &file_data_begin);
        if (rc != AXIOM_OK) return rc;
        rc = axiom_parse_safetensors_header(model, shard, header, file_data_begin);
        if (rc != AXIOM_OK) return rc;
    }
    model->safetensors_header_tensor_count = (uint32_t)model->tensors.size();
    return AXIOM_OK;
}

static const axiom_tensor_record *axiom_model_find_tensor(
        axiom_model *model,
        const char *name) {
    if (!model || !name) return nullptr;
    for (const auto &tensor : model->tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

static std::string axiom_extract_first_architecture(const std::string &json) {
    const std::string needle = "\"architectures\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos);
    if (pos == std::string::npos) return "";
    const size_t start = pos + 1;
    pos = json.find('"', start);
    if (pos == std::string::npos) return "";
    return json.substr(start, pos - start);
}

static void axiom_model_load_config(axiom_model *model, const std::filesystem::path &dir) {
    const auto config_path = dir / "config.json";
    std::string json;
    if (!model || !axiom_read_file(config_path, &json)) return;

    std::string model_type;
    if (axiom_extract_json_string(json, "model_type", &model_type)) {
        model->model_type = model_type;
    }
    model->architecture = axiom_extract_first_architecture(json);
    const std::string *model_json = &json;
    std::string text_config_json;
    size_t text_start = 0;
    size_t text_end = 0;
    if (axiom_find_json_object_range(json, "text_config", &text_start, &text_end)) {
        text_config_json = json.substr(text_start, text_end - text_start);
        model_json = &text_config_json;
        std::string text_model_type;
        if (axiom_extract_json_string(text_config_json, "model_type", &text_model_type)) {
            model->model_type = text_model_type;
        }
    }
    model->hidden_size = axiom_extract_json_u32(*model_json, "hidden_size");
    model->intermediate_size = axiom_extract_json_u32(*model_json, "intermediate_size");
    if (model->intermediate_size == 0) {
        model->intermediate_size = axiom_extract_json_u32(*model_json, "moe_intermediate_size");
    }
    model->num_hidden_layers = axiom_extract_json_u32(*model_json, "num_hidden_layers");
    model->num_attention_heads = axiom_extract_json_u32(*model_json, "num_attention_heads");
    model->num_key_value_heads = axiom_extract_json_u32(*model_json, "num_key_value_heads");
    model->vocab_size = axiom_extract_json_u32(*model_json, "vocab_size");
    const uint32_t max_pos = axiom_extract_json_u32(*model_json, "max_position_embeddings");
    if (max_pos != 0) model->max_context = max_pos;
}

static void axiom_model_load_weight_index(axiom_model *model, const std::filesystem::path &dir) {
    const auto index_path = dir / "model.safetensors.index.json";
    std::string json;
    if (!model || !axiom_read_file(index_path, &json)) return;
    model->weight_index_total_bytes = axiom_extract_json_u64(json, "total_size");
    model->weight_tensor_count = axiom_count_object_entries_after_key(json, "weight_map");
}

static uint32_t axiom_extract_added_token_id(
        const std::string &json,
        const char *content) {
    size_t pos = json.find(content);
    while (pos != std::string::npos) {
        const size_t start = json.rfind('{', pos);
        const size_t end = json.find('}', pos);
        if (start != std::string::npos && end != std::string::npos && start < end) {
            const size_t id = json.find("\"id\"", start);
            if (id != std::string::npos && id < end) {
                const size_t colon = json.find(':', id);
                if (colon != std::string::npos && colon < end) {
                    return axiom_parse_u32_after(json, colon + 1);
                }
            }
        }
        pos = json.find(content, pos + std::strlen(content));
    }
    return AXIOM_TOKEN_ID_INVALID;
}

static uint32_t axiom_count_occurrences(const std::string &s, const char *needle) {
    uint32_t count = 0;
    const size_t n = std::strlen(needle);
    size_t pos = s.find(needle);
    while (pos != std::string::npos) {
        ++count;
        pos = s.find(needle, pos + n);
    }
    return count;
}

static bool axiom_find_json_array_range(
        const std::string &json,
        const char *key,
        size_t *out_start,
        size_t *out_end) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find('[', pos + needle.size());
    if (pos == std::string::npos) return false;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = pos; i < json.size(); ++i) {
        const char c = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                *out_start = pos;
                *out_end = i + 1;
                return true;
            }
        }
    }
    return false;
}

static uint32_t axiom_count_added_tokens(const std::string &json) {
    size_t start = 0;
    size_t end = 0;
    if (!axiom_find_json_array_range(json, "added_tokens", &start, &end)) return 0;
    return axiom_count_occurrences(json.substr(start, end - start), "\"content\"");
}

static uint32_t axiom_count_vocab_entries(const std::string &json) {
    const size_t key = json.find("\"vocab\"");
    if (key == std::string::npos) return 0;
    size_t pos = json.find('{', key);
    if (pos == std::string::npos) return 0;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    uint32_t count = 0;
    for (; pos < json.size(); ++pos) {
        const char c = json[pos];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            ++depth;
            continue;
        }
        if (c == '}') {
            --depth;
            if (depth == 0) break;
            continue;
        }
        if (c == ':' && depth == 1) ++count;
    }
    return count;
}

static bool axiom_find_json_object_range(
        const std::string &json,
        const char *key,
        size_t *out_start,
        size_t *out_end) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find('{', pos + needle.size());
    if (pos == std::string::npos) return false;
    const size_t end = axiom_find_json_match(json, pos, '{', '}');
    if (end == std::string::npos) return false;
    *out_start = pos;
    *out_end = end + 1;
    return true;
}

static bool axiom_parse_json_u32_at(
        const std::string &json,
        size_t *pos,
        uint32_t *out) {
    if (!pos || !out) return false;
    size_t i = axiom_skip_ws(json, *pos);
    uint64_t v = 0;
    bool any = false;
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') {
        any = true;
        v = v * 10u + (uint32_t)(json[i] - '0');
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        ++i;
    }
    if (!any) return false;
    *pos = i;
    *out = (uint32_t)v;
    return true;
}

static bool axiom_utf8_next(
        const std::string &s,
        size_t *pos,
        uint32_t *out_cp,
        size_t *out_start,
        size_t *out_len) {
    if (!pos || !out_cp || !out_start || !out_len || *pos >= s.size()) return false;
    const size_t start = *pos;
    const unsigned char c0 = (unsigned char)s[start];
    uint32_t cp = 0;
    size_t len = 1;
    if ((c0 & 0x80u) == 0) {
        cp = c0;
    } else if ((c0 & 0xe0u) == 0xc0u && start + 1 < s.size()) {
        cp = ((uint32_t)(c0 & 0x1fu) << 6u) |
             ((uint32_t)((unsigned char)s[start + 1] & 0x3fu));
        len = 2;
    } else if ((c0 & 0xf0u) == 0xe0u && start + 2 < s.size()) {
        cp = ((uint32_t)(c0 & 0x0fu) << 12u) |
             ((uint32_t)((unsigned char)s[start + 1] & 0x3fu) << 6u) |
             ((uint32_t)((unsigned char)s[start + 2] & 0x3fu));
        len = 3;
    } else if ((c0 & 0xf8u) == 0xf0u && start + 3 < s.size()) {
        cp = ((uint32_t)(c0 & 0x07u) << 18u) |
             ((uint32_t)((unsigned char)s[start + 1] & 0x3fu) << 12u) |
             ((uint32_t)((unsigned char)s[start + 2] & 0x3fu) << 6u) |
             ((uint32_t)((unsigned char)s[start + 3] & 0x3fu));
        len = 4;
    } else {
        cp = c0;
    }
    *pos = start + len;
    *out_cp = cp;
    *out_start = start;
    *out_len = len;
    return true;
}

static int axiom_byte_decoder_lookup(uint32_t cp) {
    struct ByteDecoderTable {
        int table[512];
        ByteDecoderTable() {
            for (int i = 0; i < 512; ++i) table[i] = -1;
            bool direct[256] = {};
            for (int b = 33; b <= 126; ++b) direct[b] = true;
            for (int b = 161; b <= 172; ++b) direct[b] = true;
            for (int b = 174; b <= 255; ++b) direct[b] = true;
            for (int b = 0; b < 256; ++b) {
                if (direct[b]) table[b] = b;
            }
            int n = 0;
            for (int b = 0; b < 256; ++b) {
                if (!direct[b]) table[256 + n++] = b;
            }
        }
    };
    static const ByteDecoderTable t;
    return cp < 512 ? t.table[cp] : -1;
}

static std::string axiom_byte_decode_token_piece(const std::string &piece) {
    std::string out;
    size_t pos = 0;
    while (pos < piece.size()) {
        uint32_t cp = 0;
        size_t raw_start = 0;
        size_t raw_len = 0;
        if (!axiom_utf8_next(piece, &pos, &cp, &raw_start, &raw_len)) break;
        const int byte = axiom_byte_decoder_lookup(cp);
        if (byte >= 0) {
            out.push_back((char)byte);
        } else {
            out.append(piece.data() + raw_start, raw_len);
        }
    }
    return out;
}

static uint32_t axiom_byte_encoder_codepoint(unsigned char byte) {
    if ((byte >= 33 && byte <= 126) ||
        (byte >= 161 && byte <= 172) ||
        byte >= 174) {
        return byte;
    }
    uint32_t n = 0;
    for (uint32_t b = 0; b < byte; ++b) {
        const bool direct = (b >= 33 && b <= 126) ||
                            (b >= 161 && b <= 172) ||
                            (b >= 174 && b <= 255);
        if (!direct) ++n;
    }
    return 256u + n;
}

static void axiom_append_utf8(std::string *out, uint32_t cp) {
    if (cp <= 0x7f) {
        out->push_back((char)cp);
    } else if (cp <= 0x7ff) {
        out->push_back((char)(0xc0u | (cp >> 6u)));
        out->push_back((char)(0x80u | (cp & 0x3fu)));
    } else if (cp <= 0xffff) {
        out->push_back((char)(0xe0u | (cp >> 12u)));
        out->push_back((char)(0x80u | ((cp >> 6u) & 0x3fu)));
        out->push_back((char)(0x80u | (cp & 0x3fu)));
    } else {
        out->push_back((char)(0xf0u | (cp >> 18u)));
        out->push_back((char)(0x80u | ((cp >> 12u) & 0x3fu)));
        out->push_back((char)(0x80u | ((cp >> 6u) & 0x3fu)));
        out->push_back((char)(0x80u | (cp & 0x3fu)));
    }
}

static std::string axiom_byte_encode_text(const char *text, size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        axiom_append_utf8(
                &out,
                axiom_byte_encoder_codepoint((unsigned char)text[i]));
    }
    return out;
}

static std::string axiom_merge_key(const std::string &a, const std::string &b) {
    std::string key;
    key.reserve(a.size() + b.size() + 1);
    key.append(a);
    key.push_back('\001');
    key.append(b);
    return key;
}

static int axiom_tokenizer_load_vocab(
        const std::string &json,
        axiom_tokenizer *tokenizer) {
    if (!tokenizer) return AXIOM_ERR_INVALID_ARGUMENT;
    size_t start = 0;
    size_t end = 0;
    if (!axiom_find_json_object_range(json, "vocab", &start, &end)) {
        return AXIOM_ERR_IO;
    }

    size_t pos = start + 1;
    while (pos < end) {
        pos = axiom_skip_ws(json, pos);
        if (pos >= end || json[pos] == '}') break;
        std::string token;
        if (!axiom_parse_json_string_at(json, &pos, &token)) return AXIOM_ERR_IO;
        pos = axiom_skip_ws(json, pos);
        if (pos >= end || json[pos] != ':') return AXIOM_ERR_IO;
        ++pos;
        uint32_t id = 0;
        if (!axiom_parse_json_u32_at(json, &pos, &id)) return AXIOM_ERR_IO;
        if (id >= tokenizer->id_to_token.size()) {
            tokenizer->id_to_token.resize((size_t)id + 1);
        }
        tokenizer->token_to_id[token] = id;
        tokenizer->id_to_token[id] = axiom_byte_decode_token_piece(token);
    }
    return AXIOM_OK;
}

static int axiom_tokenizer_load_added_tokens(
        const std::string &json,
        axiom_tokenizer *tokenizer) {
    if (!tokenizer) return AXIOM_ERR_INVALID_ARGUMENT;
    size_t start = 0;
    size_t end = 0;
    if (!axiom_find_json_array_range(json, "added_tokens", &start, &end)) {
        return AXIOM_OK;
    }

    size_t pos = start + 1;
    while (pos < end) {
        pos = axiom_skip_ws(json, pos);
        if (pos >= end || json[pos] == ']') break;
        if (json[pos] != '{') {
            ++pos;
            continue;
        }
        const size_t obj_end = axiom_find_json_match(json, pos, '{', '}');
        if (obj_end == std::string::npos || obj_end > end) return AXIOM_ERR_IO;

        const size_t id_key = json.find("\"id\"", pos);
        const size_t content_key = json.find("\"content\"", pos);
        if (id_key != std::string::npos && id_key < obj_end &&
            content_key != std::string::npos && content_key < obj_end) {
            size_t id_pos = json.find(':', id_key);
            size_t content_pos = json.find(':', content_key);
            if (id_pos == std::string::npos || id_pos >= obj_end ||
                content_pos == std::string::npos || content_pos >= obj_end) {
                return AXIOM_ERR_IO;
            }
            ++id_pos;
            ++content_pos;
            uint32_t id = 0;
            std::string content;
            content_pos = axiom_skip_ws(json, content_pos);
            if (!axiom_parse_json_u32_at(json, &id_pos, &id) ||
                !axiom_parse_json_string_at(json, &content_pos, &content)) {
                return AXIOM_ERR_IO;
            }
            if (id >= tokenizer->id_to_token.size()) {
                tokenizer->id_to_token.resize((size_t)id + 1);
            }
            tokenizer->token_to_id[content] = id;
            tokenizer->added_token_pieces.push_back({content, id});
            tokenizer->id_to_token[id] = std::move(content);
        }
        pos = obj_end + 1;
    }
    return AXIOM_OK;
}

static int axiom_tokenizer_load_merges(
        const std::string &json,
        axiom_tokenizer *tokenizer) {
    if (!tokenizer) return AXIOM_ERR_INVALID_ARGUMENT;
    size_t start = 0;
    size_t end = 0;
    if (!axiom_find_json_array_range(json, "merges", &start, &end)) {
        return AXIOM_OK;
    }

    size_t pos = start + 1;
    uint32_t rank = 0;
    while (pos < end) {
        pos = axiom_skip_ws(json, pos);
        if (pos >= end || json[pos] == ']') break;
        std::string merge;
        if (!axiom_parse_json_string_at(json, &pos, &merge)) {
            ++pos;
            continue;
        }
        const size_t sep = merge.find(' ');
        if (sep != std::string::npos) {
            const std::string left = merge.substr(0, sep);
            const std::string right = merge.substr(sep + 1);
            tokenizer->merge_ranks[axiom_merge_key(left, right)] = rank++;
        }
    }
    return AXIOM_OK;
}

int axiom_tokenizer_open(axiom_tokenizer **out, const axiom_tokenizer_config *config) {
    if (!out || !config || config->abi_version != AXIOM_ABI_VERSION ||
        !config->path || !config->path[0]) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (config->format != AXIOM_TOKENIZER_FORMAT_HF_JSON &&
        config->format != AXIOM_TOKENIZER_FORMAT_NATIVE) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    std::error_code ec;
    const std::filesystem::path p(config->path);
    std::filesystem::path tokenizer_path;
    std::filesystem::path chat_template_path;
    if (std::filesystem::is_directory(p, ec)) {
        tokenizer_path = p / "tokenizer.json";
        const auto chat = p / "chat_template.jinja";
        const auto tokcfg = p / "tokenizer_config.json";
        if (std::filesystem::is_regular_file(chat, ec)) {
            chat_template_path = chat;
        } else if (std::filesystem::is_regular_file(tokcfg, ec)) {
            chat_template_path = tokcfg;
        }
    } else if (std::filesystem::is_regular_file(p, ec)) {
        tokenizer_path = p;
    } else {
        return AXIOM_ERR_IO;
    }
    if (!std::filesystem::is_regular_file(tokenizer_path, ec)) return AXIOM_ERR_IO;

    std::string tokenizer_json;
    if (!axiom_read_file(tokenizer_path, &tokenizer_json)) return AXIOM_ERR_IO;

    axiom_tokenizer *tokenizer = new (std::nothrow) axiom_tokenizer();
    if (!tokenizer) return AXIOM_ERR_RUNTIME;
    tokenizer->path = config->path;
    tokenizer->name = config->name && config->name[0] ? config->name : p.filename().string();
    tokenizer->format = config->format;
    tokenizer->endoftext_token_id = AXIOM_TOKEN_ID_INVALID;
    tokenizer->im_start_token_id = AXIOM_TOKEN_ID_INVALID;
    tokenizer->im_end_token_id = AXIOM_TOKEN_ID_INVALID;
    tokenizer->tool_call_token_id = AXIOM_TOKEN_ID_INVALID;
    tokenizer->tool_call_end_token_id = AXIOM_TOKEN_ID_INVALID;

    int rc = axiom_hash_file(
            tokenizer_path,
            &tokenizer->tokenizer_json_bytes,
            &tokenizer->tokenizer_hash);
    if (rc != AXIOM_OK) {
        delete tokenizer;
        return rc;
    }
    if (!chat_template_path.empty()) {
        std::string chat_source;
        std::string chat_template;
        if (!axiom_read_file(chat_template_path, &chat_source)) {
            delete tokenizer;
            return AXIOM_ERR_IO;
        }
        if (chat_template_path.filename() == "tokenizer_config.json" &&
            axiom_extract_json_string(chat_source, "chat_template", &chat_template)) {
            tokenizer->chat_template_hash = axiom_fnv1a_update(
                    1469598103934665603ull,
                    (const unsigned char *)chat_template.data(),
                    chat_template.size());
        } else {
            tokenizer->chat_template_hash = axiom_fnv1a_update(
                    1469598103934665603ull,
                    (const unsigned char *)chat_source.data(),
                    chat_source.size());
        }
    }

    tokenizer->vocab_size = axiom_count_vocab_entries(tokenizer_json);
    tokenizer->added_tokens = axiom_count_added_tokens(tokenizer_json);
    tokenizer->endoftext_token_id =
            axiom_extract_added_token_id(tokenizer_json, "<|endoftext|>");
    tokenizer->im_start_token_id =
            axiom_extract_added_token_id(tokenizer_json, "<|im_start|>");
    tokenizer->im_end_token_id =
            axiom_extract_added_token_id(tokenizer_json, "<|im_end|>");
    tokenizer->tool_call_token_id =
            axiom_extract_added_token_id(tokenizer_json, "<tool_call>");
    tokenizer->tool_call_end_token_id =
            axiom_extract_added_token_id(tokenizer_json, "</tool_call>");
    rc = axiom_tokenizer_load_vocab(tokenizer_json, tokenizer);
    if (rc != AXIOM_OK) {
        delete tokenizer;
        return rc;
    }
    rc = axiom_tokenizer_load_added_tokens(tokenizer_json, tokenizer);
    if (rc != AXIOM_OK) {
        delete tokenizer;
        return rc;
    }
    rc = axiom_tokenizer_load_merges(tokenizer_json, tokenizer);
    if (rc != AXIOM_OK) {
        delete tokenizer;
        return rc;
    }

    *out = tokenizer;
    return AXIOM_OK;
}

void axiom_tokenizer_close(axiom_tokenizer *tokenizer) {
    delete tokenizer;
}

int axiom_tokenizer_info_get(axiom_tokenizer *tokenizer, axiom_tokenizer_info *out) {
    if (!tokenizer || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    axiom_copy_string(out->name, sizeof(out->name), tokenizer->name);
    axiom_copy_string(out->path, sizeof(out->path), tokenizer->path);
    out->format = tokenizer->format;
    out->tokenizer_json_bytes = tokenizer->tokenizer_json_bytes;
    out->tokenizer_hash = tokenizer->tokenizer_hash;
    out->chat_template_hash = tokenizer->chat_template_hash;
    out->vocab_size = tokenizer->vocab_size;
    out->added_tokens = tokenizer->added_tokens;
    out->endoftext_token_id = tokenizer->endoftext_token_id;
    out->im_start_token_id = tokenizer->im_start_token_id;
    out->im_end_token_id = tokenizer->im_end_token_id;
    out->tool_call_token_id = tokenizer->tool_call_token_id;
    out->tool_call_end_token_id = tokenizer->tool_call_end_token_id;
    return AXIOM_OK;
}

int axiom_tokenizer_token_id(
        axiom_tokenizer *tokenizer,
        const char *token,
        uint32_t *out_token_id) {
    if (out_token_id) *out_token_id = AXIOM_TOKEN_ID_INVALID;
    if (!tokenizer || !token || !token[0] || !out_token_id) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const auto found = tokenizer->token_to_id.find(token);
    if (found == tokenizer->token_to_id.end()) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_token_id = found->second;
    return AXIOM_OK;
}

int axiom_tokenizer_decode_token(
        axiom_tokenizer *tokenizer,
        uint32_t token_id,
        char *out_text,
        uint32_t out_capacity,
        uint32_t *out_bytes) {
    if (!tokenizer || !out_text || out_capacity == 0 || !out_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_bytes = 0;
    if (token_id >= tokenizer->id_to_token.size()) return AXIOM_ERR_INVALID_ARGUMENT;
    const std::string &piece = tokenizer->id_to_token[token_id];
    if (piece.empty() && token_id != 0) return AXIOM_ERR_INVALID_ARGUMENT;
    if (piece.size() + 1 > out_capacity) return AXIOM_ERR_BUDGET;
    if (!piece.empty()) std::memcpy(out_text, piece.data(), piece.size());
    out_text[piece.size()] = '\0';
    *out_bytes = (uint32_t)piece.size();
    return AXIOM_OK;
}

int axiom_tokenizer_decode_ids(
        axiom_tokenizer *tokenizer,
        const uint32_t *token_ids,
        uint32_t token_count,
        char *out_text,
        uint32_t out_capacity,
        uint32_t *out_bytes) {
    if (!tokenizer || !token_ids || token_count == 0 ||
        !out_text || out_capacity == 0 || !out_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint32_t used = 0;
    for (uint32_t i = 0; i < token_count; ++i) {
        const uint32_t id = token_ids[i];
        if (id >= tokenizer->id_to_token.size()) return AXIOM_ERR_INVALID_ARGUMENT;
        const std::string &piece = tokenizer->id_to_token[id];
        if (piece.empty() && id != 0) return AXIOM_ERR_INVALID_ARGUMENT;
        if (piece.size() > (size_t)(out_capacity - used - 1)) return AXIOM_ERR_BUDGET;
        if (!piece.empty()) {
            std::memcpy(out_text + used, piece.data(), piece.size());
            used += (uint32_t)piece.size();
        }
    }
    out_text[used] = '\0';
    *out_bytes = used;
    return AXIOM_OK;
}

static bool axiom_tokenizer_match_added(
        axiom_tokenizer *tokenizer,
        const char *text,
        size_t pos,
        size_t n,
        uint32_t *out_id,
        size_t *out_len) {
    size_t best_len = 0;
    uint32_t best_id = 0;
    for (const auto &item : tokenizer->added_token_pieces) {
        const std::string &piece = item.first;
        if (piece.empty() || pos + piece.size() > n || piece.size() <= best_len) {
            continue;
        }
        if (std::memcmp(text + pos, piece.data(), piece.size()) == 0) {
            best_len = piece.size();
            best_id = item.second;
        }
    }
    if (best_len == 0) return false;
    *out_id = best_id;
    *out_len = best_len;
    return true;
}

static int axiom_tokenizer_emit_id(
        uint32_t id,
        uint32_t *out_token_ids,
        uint32_t out_capacity,
        uint32_t *out_count) {
    if (*out_count >= out_capacity) return AXIOM_ERR_BUDGET;
    out_token_ids[*out_count] = id;
    ++*out_count;
    return AXIOM_OK;
}

static int axiom_tokenizer_emit_bpe_piece(
        axiom_tokenizer *tokenizer,
        const char *text,
        size_t n,
        uint32_t *out_token_ids,
        uint32_t out_capacity,
        uint32_t *out_count) {
    if (n == 0) return AXIOM_OK;
    const std::string encoded = axiom_byte_encode_text(text, n);
    std::vector<std::string> parts;
    size_t pos = 0;
    while (pos < encoded.size()) {
        uint32_t cp = 0;
        size_t raw_start = 0;
        size_t raw_len = 0;
        if (!axiom_utf8_next(encoded, &pos, &cp, &raw_start, &raw_len)) break;
        (void)cp;
        parts.emplace_back(encoded.data() + raw_start, raw_len);
    }

    while (parts.size() > 1) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_i = parts.size();
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            const auto found = tokenizer->merge_ranks.find(
                    axiom_merge_key(parts[i], parts[i + 1]));
            if (found != tokenizer->merge_ranks.end() && found->second < best_rank) {
                best_rank = found->second;
                best_i = i;
            }
        }
        if (best_i == parts.size()) break;
        const std::string left = parts[best_i];
        const std::string right = parts[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(parts.size() - 1);
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i + 1 < parts.size() && parts[i] == left && parts[i + 1] == right) {
                merged.push_back(parts[i] + parts[i + 1]);
                ++i;
            } else {
                merged.push_back(parts[i]);
            }
        }
        parts.swap(merged);
    }

    for (const std::string &part : parts) {
        const auto found = tokenizer->token_to_id.find(part);
        if (found == tokenizer->token_to_id.end()) return AXIOM_ERR_IO;
        const int rc = axiom_tokenizer_emit_id(
                found->second,
                out_token_ids,
                out_capacity,
                out_count);
        if (rc != AXIOM_OK) return rc;
    }
    return AXIOM_OK;
}

static int axiom_tokenizer_emit_pretokenized_piece(
        axiom_tokenizer *tokenizer,
        const char *text,
        size_t n,
        uint32_t *out_token_ids,
        uint32_t out_capacity,
        uint32_t *out_count) {
    auto is_word_byte = [](unsigned char ch) -> bool {
        return std::isalpha(ch) || ch >= 0x80u;
    };
    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < n) {
        const unsigned char c = (unsigned char)text[pos];
        if (c >= '0' && c <= '9') {
            int rc = axiom_tokenizer_emit_bpe_piece(
                    tokenizer,
                    text + plain_start,
                    pos - plain_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            size_t digit_start = pos;
            while (pos < n) {
                const unsigned char d = (unsigned char)text[pos];
                if (d < '0' || d > '9') break;
                ++pos;
            }
            while (digit_start < pos) {
                const size_t chunk = std::min<size_t>(3, pos - digit_start);
                rc = axiom_tokenizer_emit_bpe_piece(
                        tokenizer,
                        text + digit_start,
                        chunk,
                        out_token_ids,
                        out_capacity,
                        out_count);
                if (rc != AXIOM_OK) return rc;
                digit_start += chunk;
            }
            plain_start = pos;
            continue;
        }
        if (std::isspace(c) && c != '\r' && c != '\n') {
            int rc = axiom_tokenizer_emit_bpe_piece(
                    tokenizer,
                    text + plain_start,
                    pos - plain_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            const size_t ws_start = pos;
            while (pos < n &&
                   std::isspace((unsigned char)text[pos]) &&
                   text[pos] != '\r' && text[pos] != '\n') {
                ++pos;
            }
            if (pos < n && is_word_byte((unsigned char)text[pos])) {
                const size_t attach = pos > ws_start ? pos - 1 : pos;
                if (attach > ws_start) {
                    rc = axiom_tokenizer_emit_bpe_piece(
                            tokenizer,
                            text + ws_start,
                            attach - ws_start,
                            out_token_ids,
                            out_capacity,
                            out_count);
                    if (rc != AXIOM_OK) return rc;
                }
                while (pos < n) {
                    const unsigned char w = (unsigned char)text[pos];
                    if (!is_word_byte(w)) break;
                    ++pos;
                }
                rc = axiom_tokenizer_emit_bpe_piece(
                        tokenizer,
                        text + attach,
                        pos - attach,
                        out_token_ids,
                        out_capacity,
                        out_count);
                if (rc != AXIOM_OK) return rc;
                plain_start = pos;
                continue;
            }
            if (pos < n && std::ispunct((unsigned char)text[pos])) {
                size_t piece_start = pos;
                if (pos > ws_start && text[pos - 1] == ' ') {
                    piece_start = pos - 1;
                    for (size_t i = ws_start; i < piece_start; ++i) {
                        rc = axiom_tokenizer_emit_bpe_piece(
                                tokenizer,
                                text + i,
                                1,
                                out_token_ids,
                                out_capacity,
                                out_count);
                        if (rc != AXIOM_OK) return rc;
                    }
                } else {
                    for (size_t i = ws_start; i < pos; ++i) {
                        rc = axiom_tokenizer_emit_bpe_piece(
                                tokenizer,
                                text + i,
                                1,
                                out_token_ids,
                                out_capacity,
                                out_count);
                        if (rc != AXIOM_OK) return rc;
                    }
                }
                if (piece_start < pos) {
                    ++pos;
                    while (pos < n && std::ispunct((unsigned char)text[pos])) ++pos;
                    while (pos < n && (text[pos] == '\r' || text[pos] == '\n')) ++pos;
                    rc = axiom_tokenizer_emit_bpe_piece(
                            tokenizer,
                            text + piece_start,
                            pos - piece_start,
                            out_token_ids,
                            out_capacity,
                            out_count);
                    if (rc != AXIOM_OK) return rc;
                }
                plain_start = pos;
                continue;
            }
            if (pos < n && (text[pos] == '\r' || text[pos] == '\n')) {
                while (pos < n && (text[pos] == '\r' || text[pos] == '\n')) ++pos;
                rc = axiom_tokenizer_emit_bpe_piece(
                        tokenizer,
                        text + ws_start,
                        pos - ws_start,
                        out_token_ids,
                        out_capacity,
                        out_count);
                if (rc != AXIOM_OK) return rc;
                plain_start = pos;
                continue;
            }
            rc = axiom_tokenizer_emit_bpe_piece(
                    tokenizer,
                    text + ws_start,
                    pos - ws_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            plain_start = pos;
            continue;
        }
        if (std::ispunct(c)) {
            size_t piece_start = pos;
            size_t flush_end = pos;
            bool took_leading_space = false;
            if (pos > plain_start && text[pos - 1] == ' ') {
                piece_start = pos - 1;
                flush_end = pos - 1;
                took_leading_space = true;
            }
            int rc = axiom_tokenizer_emit_bpe_piece(
                    tokenizer,
                    text + plain_start,
                    flush_end - plain_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            ++pos;
            if (!took_leading_space && pos < n && std::isalpha((unsigned char)text[pos])) {
                while (pos < n && std::isalpha((unsigned char)text[pos])) ++pos;
            } else {
                while (pos < n && std::ispunct((unsigned char)text[pos])) ++pos;
                while (pos < n && (text[pos] == '\r' || text[pos] == '\n')) ++pos;
            }
            rc = axiom_tokenizer_emit_bpe_piece(
                    tokenizer,
                    text + piece_start,
                    pos - piece_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            plain_start = pos;
            continue;
        }
        ++pos;
    }
    return axiom_tokenizer_emit_bpe_piece(
            tokenizer,
            text + plain_start,
            n - plain_start,
            out_token_ids,
            out_capacity,
            out_count);
}

int axiom_tokenizer_encode_text(
        axiom_tokenizer *tokenizer,
        const char *text,
        uint32_t *out_token_ids,
        uint32_t out_capacity,
        uint32_t *out_count) {
    if (!tokenizer || !text || !out_token_ids || out_capacity == 0 || !out_count) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_count = 0;
    const size_t n = std::strlen(text);
    size_t pos = 0;
    size_t plain_start = 0;
    while (pos < n) {
        uint32_t added_id = 0;
        size_t added_len = 0;
        if (axiom_tokenizer_match_added(
                tokenizer,
                text,
                pos,
                n,
                &added_id,
                &added_len)) {
            int rc = axiom_tokenizer_emit_pretokenized_piece(
                    tokenizer,
                    text + plain_start,
                    pos - plain_start,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            rc = axiom_tokenizer_emit_id(
                    added_id,
                    out_token_ids,
                    out_capacity,
                    out_count);
            if (rc != AXIOM_OK) return rc;
            pos += added_len;
            plain_start = pos;
            continue;
        }
        ++pos;
    }
    return axiom_tokenizer_emit_pretokenized_piece(
            tokenizer,
            text + plain_start,
            n - plain_start,
            out_token_ids,
            out_capacity,
            out_count);
}

int axiom_tokenizer_same_identity(
        axiom_tokenizer *a,
        axiom_tokenizer *b,
        int *out_same) {
    if (!a || !b || !out_same) return AXIOM_ERR_INVALID_ARGUMENT;
    *out_same = a->format == b->format &&
                a->tokenizer_json_bytes == b->tokenizer_json_bytes &&
                a->tokenizer_hash == b->tokenizer_hash &&
                a->vocab_size == b->vocab_size &&
                a->endoftext_token_id == b->endoftext_token_id &&
                a->im_start_token_id == b->im_start_token_id &&
                a->im_end_token_id == b->im_end_token_id;
    return AXIOM_OK;
}

int axiom_entity_create(
        axiom_runtime *runtime,
        axiom_entity **out,
        axiom_model *model,
        const axiom_entity_config *config) {
    if (!runtime || !out || !model || !config ||
        config->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_entity *entity = new (std::nothrow) axiom_entity();
    if (!entity) return AXIOM_ERR_RUNTIME;
    entity->runtime = runtime;
    entity->model = model;
    entity->name = config->name && config->name[0] ? config->name : "entity";
    entity->role = config->role && config->role[0] ? config->role : "";
    entity->memory_budget_bytes = config->memory_budget_bytes;
    model->entity_count++;
    runtime->entity_count++;
    *out = entity;
    return AXIOM_OK;
}

void axiom_entity_destroy(axiom_entity *entity) {
    if (!entity) return;
    if (entity->model && entity->model->entity_count > 0) entity->model->entity_count--;
    if (entity->runtime && entity->runtime->entity_count > 0) entity->runtime->entity_count--;
    delete entity;
}

int axiom_entity_info_get(axiom_entity *entity, axiom_entity_info *out) {
    if (!entity || !out || out->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    axiom_copy_string(out->name, sizeof(out->name), entity->name);
    axiom_copy_string(out->role, sizeof(out->role), entity->role);
    out->memory_budget_bytes = entity->memory_budget_bytes;
    out->session_count = entity->session_count;
    return AXIOM_OK;
}

int axiom_session_create(
        axiom_entity *entity,
        axiom_session **out,
        const axiom_session_config *config) {
    if (!entity || !out || !config || config->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_session *session = new (std::nothrow) axiom_session();
    if (!session) return AXIOM_ERR_RUNTIME;
    session->entity = entity;
    session->max_context = config->max_context;
    session->kv_budget_bytes = config->kv_budget_bytes;
    entity->session_count++;
    if (entity->runtime) entity->runtime->session_count++;
    *out = session;
    return AXIOM_OK;
}

void axiom_session_destroy(axiom_session *session) {
    if (!session) return;
    axiom_entity *entity = session->entity;
    if (entity && entity->session_count > 0) entity->session_count--;
    if (entity && entity->runtime && entity->runtime->session_count > 0) {
        entity->runtime->session_count--;
    }
    delete session;
}

int axiom_latent_link_create(
        axiom_runtime *runtime,
        axiom_latent_link **out,
        const axiom_latent_link_config *config) {
    if (!runtime || !out || !config || config->abi_version != AXIOM_ABI_VERSION ||
        config->source_width == 0 || config->target_width == 0 ||
        config->hidden_width == 0 || config->dtype != AXIOM_LATENT_F32) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_latent_link *link = (axiom_latent_link *)std::calloc(1, sizeof(*link));
    if (!link) return AXIOM_ERR_RUNTIME;
    link->runtime = runtime;
    link->kind = config->kind;
    link->dtype = config->dtype;
    link->source_width = config->source_width;
    link->target_width = config->target_width;
    link->hidden_width = config->hidden_width;
    link->rank = config->rank;
    link->eps = config->eps > 0.0f ? config->eps : 1.0e-5f;

    if (runtime->backend != AXIOM_BACKEND_CUDA) {
        std::free(link);
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    const int rc = axiom_cuda_latent_link_create(
            runtime->backend_runtime,
            &link->backend_link,
            link->kind,
            link->dtype,
            link->source_width,
            link->target_width,
            link->hidden_width,
            link->eps);
    if (rc != AXIOM_OK) {
        std::free(link);
        return rc;
    }
    *out = link;
    return AXIOM_OK;
}

void axiom_latent_link_destroy(axiom_latent_link *link) {
    if (link && link->backend_link) axiom_cuda_latent_link_destroy(link->backend_link);
    std::free(link);
}

int axiom_latent_link_load_f32(
        axiom_latent_link *link,
        const axiom_latent_link_weights_f32 *weights) {
    if (!link || !weights || weights->abi_version != AXIOM_ABI_VERSION ||
        weights->dtype != AXIOM_LATENT_F32 ||
        !weights->pre_ln_weight || !weights->pre_ln_bias ||
        !weights->proj1_weight || !weights->proj1_bias ||
        !weights->proj2_weight || !weights->proj2_bias ||
        !weights->post_ln_weight || !weights->post_ln_bias) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const int rc = axiom_cuda_latent_link_load_f32(link->backend_link, weights);
    if (rc == AXIOM_OK) link->loaded = true;
    return rc;
}

int axiom_latent_link_apply(
        axiom_latent_link *link,
        const axiom_latent_frame *source,
        axiom_latent_frame *target) {
    if (!link || !source || !target ||
        source->abi_version != AXIOM_ABI_VERSION ||
        target->abi_version != AXIOM_ABI_VERSION ||
        !source->device_ptr || !target->device_ptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!link->loaded ||
        source->rows != target->rows ||
        source->cols != link->source_width ||
        target->cols != link->target_width ||
        source->stride < source->cols ||
        target->stride < target->cols ||
        source->dtype != target->dtype ||
        source->dtype != link->dtype) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (link->runtime->backend != AXIOM_BACKEND_CUDA) {
        return AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
    return axiom_cuda_latent_link_apply(
            link->runtime->backend_runtime,
            link->backend_link,
            source->device_ptr,
            target->device_ptr,
            source->rows,
            source->stride,
            target->stride);
}

int axiom_scheduler_step(
        axiom_runtime *runtime,
        const axiom_scheduler_step_options *options) {
    if (!runtime || !options || options->abi_version != AXIOM_ABI_VERSION) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_ERR_NOT_IMPLEMENTED;
}

int axiom_smoke_vector_add(
        axiom_runtime *runtime,
        const float *a_host,
        const float *b_host,
        float *out_host,
        size_t count) {
    if (!runtime || !a_host || !b_host || !out_host || count == 0) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (runtime->backend != AXIOM_BACKEND_CUDA) return AXIOM_ERR_UNSUPPORTED_BACKEND;
    return axiom_cuda_smoke_vector_add(
            runtime->backend_runtime,
            a_host,
            b_host,
            out_host,
            count);
}
