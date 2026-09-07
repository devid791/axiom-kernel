#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

namespace axiom::qwen4exp::qsa {

// Qwen3.8-Flash-Next / qwen4_exp QSA indexer contract.  The implementation is
// intentionally independent from every existing Axiom provider and owns no
// memory: all host/device buffers and CUDA streams are supplied by the caller.
struct config {
    std::uint32_t query_heads = 4;
    std::uint32_t kv_heads = 1;
    std::uint32_t head_dim = 128;
    std::uint32_t rotary_dim = 64;
    std::uint32_t compress_ratio = 4;
    std::uint32_t token_budget = 2048;
    std::uint32_t max_context = 262144;
    float rms_epsilon = 1.0e-6F;
};

inline constexpr config checkpoint_config{};
inline constexpr std::size_t checkpoint_block_topk = 512;
inline constexpr std::size_t checkpoint_selected_capacity = 2051;

enum class status : std::uint32_t {
    ok = 0,
    null_pointer = 1,
    invalid_config = 2,
    invalid_argument = 3,
    capacity_exceeded = 4,
    arithmetic_overflow = 5,
    non_finite = 6,
    invalid_visible_index = 7,
    cuda_failure = 8,
};

[[nodiscard]] const char* status_string(status value) noexcept;
[[nodiscard]] status validate_config(const config& cfg) noexcept;
[[nodiscard]] status validate_checkpoint_config(const config& cfg) noexcept;

[[nodiscard]] status block_topk(const config& cfg, std::size_t* out) noexcept;
[[nodiscard]] status complete_block_count(
    const config& cfg,
    std::size_t visible_count,
    std::size_t* out) noexcept;
[[nodiscard]] status selected_token_capacity(
    const config& cfg,
    std::size_t visible_count,
    std::size_t* out) noexcept;
[[nodiscard]] status host_select_workspace_bytes(
    const config& cfg,
    std::size_t block_count,
    std::size_t* out) noexcept;
[[nodiscard]] status cuda_select_workspace_bytes(
    const config& cfg,
    std::size_t block_count,
    std::size_t* out) noexcept;

// raw_queries is [query_count, query_heads, head_dim].  rms_weight is
// [head_dim] and uses the checkpoint convention scale=(1+weight).  cos/sin are
// [query_count, rotary_dim].  output is F32 with the same logical shape.
[[nodiscard]] status prepare_queries_host_f32(
    const config& cfg,
    const float* raw_queries,
    const float* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output) noexcept;

// visible_indices must be strictly increasing, just like torch.nonzero over
// the causal attention mask.  Only floor(visible_count/compress_ratio) complete
// groups are pooled; the remainder is the causal tail handled by selection.
// raw_keys is [key_count, head_dim], full_cos/full_sin are
// [position_count, rotary_dim], and output is [complete_blocks, head_dim].
[[nodiscard]] status pool_keys_host_f32(
    const config& cfg,
    const float* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const float* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count) noexcept;

// prepared_query is one [query_heads, head_dim] query.  pooled_keys is
// [block_count, head_dim].  scores implements
// sum_h(relu(dot(q_h, key_block))) / sqrt(head_dim).
[[nodiscard]] status score_blocks_host_f32(
    const config& cfg,
    const float* prepared_query,
    const float* pooled_keys,
    std::size_t block_count,
    float* scores) noexcept;

// Deterministic top-k: descending score, then ascending block ordinal for an
// exact tie.  This matches the oracle for non-ties and makes the otherwise
// unspecified torch.topk cutoff deterministic.  Tokens inside each selected
// block retain mask order; the incomplete causal tail is appended unchanged.
[[nodiscard]] status select_tokens_host_f32(
    const config& cfg,
    const float* scores,
    std::size_t block_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    std::size_t key_count,
    std::int32_t* selected_tokens,
    std::size_t selected_capacity,
    void* workspace,
    std::size_t workspace_bytes,
    std::size_t* selected_count) noexcept;

// Builds the boolean mask returned by the official indexer.  Bytes are 0/1.
[[nodiscard]] status build_mask_host(
    const std::int32_t* selected_tokens,
    std::size_t selected_count,
    std::size_t kv_length,
    std::uint8_t* mask) noexcept;

// CUDA calls are stream-ordered and never synchronize internally.  Device
// data errors are reported through device_status; reset it before a pipeline
// and collect it after the last launch.  A non-ok collected status makes every
// produced buffer invalid (fail closed).  Static pointer/shape/capacity errors
// are returned immediately.
[[nodiscard]] status cuda_reset_status(
    status* device_status,
    cudaStream_t stream) noexcept;
[[nodiscard]] status cuda_collect_status(
    const status* device_status,
    cudaStream_t stream,
    status* host_status) noexcept;

[[nodiscard]] status prepare_queries_cuda_f32(
    const config& cfg,
    const float* raw_queries,
    const float* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output,
    status* device_status,
    cudaStream_t stream) noexcept;

// BF16 inputs/weights are promoted to F32 for norm, RoPE and output.
[[nodiscard]] status prepare_queries_cuda_bf16(
    const config& cfg,
    const __nv_bfloat16* raw_queries,
    const __nv_bfloat16* rms_weight,
    const float* cos,
    const float* sin,
    std::size_t query_count,
    float* output,
    status* device_status,
    cudaStream_t stream) noexcept;

[[nodiscard]] status pool_keys_cuda_f32(
    const config& cfg,
    const float* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const float* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count,
    status* device_status,
    cudaStream_t stream) noexcept;

[[nodiscard]] status pool_keys_cuda_bf16(
    const config& cfg,
    const __nv_bfloat16* raw_keys,
    std::size_t key_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    const __nv_bfloat16* rms_weight,
    const float* full_cos,
    const float* full_sin,
    std::size_t position_count,
    float* output,
    std::size_t output_block_capacity,
    std::size_t* output_block_count,
    status* device_status,
    cudaStream_t stream) noexcept;

[[nodiscard]] status score_blocks_cuda_f32(
    const config& cfg,
    const float* prepared_query,
    const float* pooled_keys,
    std::size_t block_count,
    float* scores,
    status* device_status,
    cudaStream_t stream) noexcept;

[[nodiscard]] status select_tokens_cuda_f32(
    const config& cfg,
    const float* scores,
    std::size_t block_count,
    const std::int32_t* visible_indices,
    std::size_t visible_count,
    std::size_t key_count,
    std::int32_t* selected_tokens,
    std::size_t selected_capacity,
    void* workspace,
    std::size_t workspace_bytes,
    std::size_t* selected_count,
    status* device_status,
    cudaStream_t stream) noexcept;

[[nodiscard]] status build_mask_cuda(
    const std::int32_t* selected_tokens,
    std::size_t selected_count,
    std::size_t kv_length,
    std::uint8_t* mask,
    status* device_status,
    cudaStream_t stream) noexcept;

}  // namespace axiom::qwen4exp::qsa
