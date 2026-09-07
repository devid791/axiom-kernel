#ifndef AXIOM_QWEN4EXP_ATTENTION_HPP
#define AXIOM_QWEN4EXP_ATTENTION_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp::attention {

inline constexpr std::uint32_t kCheckpointQueryHeads = 24u;
inline constexpr std::uint32_t kCheckpointKvHeads = 2u;
inline constexpr std::uint32_t kCheckpointHeadDim = 256u;
inline constexpr std::uint32_t kCheckpointRotaryDim = 64u;
inline constexpr std::uint32_t kCheckpointGqaRatio = 12u;
inline constexpr std::uint32_t kCheckpointQHeadStride = 512u;
inline constexpr std::uint32_t kCheckpointOutputDim = 6144u;
inline constexpr std::uint32_t kCheckpointMaxContext = 262144u;
inline constexpr std::uint32_t kCheckpointMaxSelectedTokens = 2051u;
inline constexpr float kCheckpointRmsEpsilon = 1.0e-6F;
inline constexpr float kCheckpointAttentionScale = 0.0625F;

/*
 * Additive qwen4_exp attention contract.  This API owns no memory and is
 * deliberately independent from every existing Axiom provider.  BF16 values
 * are exposed as their raw 16-bit payload so this header remains POD-friendly.
 */
struct config {
    std::uint32_t query_heads = kCheckpointQueryHeads;
    std::uint32_t kv_heads = kCheckpointKvHeads;
    std::uint32_t head_dim = kCheckpointHeadDim;
    std::uint32_t rotary_dim = kCheckpointRotaryDim;
    std::uint32_t max_context = kCheckpointMaxContext;
    std::uint32_t max_selected_tokens = kCheckpointMaxSelectedTokens;
    float rms_epsilon = kCheckpointRmsEpsilon;
};

inline constexpr config checkpoint_config{};

enum class status : std::uint32_t {
    ok = 0u,
    null_pointer,
    invalid_config,
    invalid_argument,
    invalid_state,
    capacity_exceeded,
    arithmetic_overflow,
    insufficient_workspace,
    invalid_selected_count,
    invalid_selected_index,
    non_finite,
    cuda_failure,
};

struct cache_lengths {
    std::size_t committed = 0u;
    std::size_t staged = 0u;
    std::size_t visible = 0u;
    std::size_t capacity = 0u;
};

/* External token-major cache layout: [capacity, kv_heads, head_dim].  Host
 * metadata is not internally locked: one state must be externally serialized,
 * and CUDA work must use one stream or explicit inter-stream events.
 */
struct cache_state {
    std::uint16_t* key_bf16 = nullptr;
    std::uint16_t* value_bf16 = nullptr;
    std::size_t capacity_tokens = 0u;
    std::size_t committed_tokens = 0u;
    std::size_t staged_tokens = 0u;
    bool initialized = false;
    bool transaction_open = false;
};

/* A stage view remains valid while its caller-owned workspace and q_proj input
 * remain valid. prepared_queries_f32 is [tokens, query_heads, head_dim].
 * q_proj_bf16 uses the official [tokens, query_heads, 2*head_dim] layout:
 * query[head_dim] followed by gate[head_dim] for each head.
 */
struct staged_batch {
    std::size_t start_position = 0u;
    std::size_t token_count = 0u;
    const float* prepared_queries_f32 = nullptr;
    const std::uint16_t* q_proj_bf16 = nullptr;
};

[[nodiscard]] const char* status_string(status value) noexcept;
[[nodiscard]] status validate_config(const config& value) noexcept;
[[nodiscard]] status validate_checkpoint_config(const config& value) noexcept;
[[nodiscard]] float attention_scale(const config& value) noexcept;

/* Includes alignment padding and the F32 prepared-query storage. */
[[nodiscard]] status workspace_bytes(
    const config& value,
    std::size_t token_count,
    std::size_t* bytes) noexcept;

[[nodiscard]] status cache_initialize(
    cache_state* state,
    std::uint16_t* external_key_bf16,
    std::uint16_t* external_value_bf16,
    std::size_t capacity_tokens,
    std::size_t initial_committed_tokens = 0u) noexcept;
[[nodiscard]] status cache_reset(cache_state* state) noexcept;
[[nodiscard]] status cache_get_lengths(
    const cache_state* state,
    cache_lengths* lengths) noexcept;
[[nodiscard]] status begin_transaction(cache_state* state) noexcept;
[[nodiscard]] status commit_prefix(
    cache_state* state,
    std::size_t accepted_tokens) noexcept;
[[nodiscard]] status rollback(cache_state* state) noexcept;

/* Device-status handling is explicit. reset is asynchronous; collect is the
 * documented synchronization boundary. stage_cuda and forward_qsa_cuda never
 * allocate or synchronize and report data-dependent failures here.
 */
[[nodiscard]] status reset_device_status(
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] status collect_device_status(
    const status* device_status,
    status* host_status,
    cudaStream_t stream = nullptr) noexcept;

/*
 * Stage a contiguous query/prefill batch at the current visible cache tail.
 * Inputs are device-resident:
 *   q_proj [tokens, query_heads, 2*head_dim] BF16
 *   k_proj [tokens, kv_heads, head_dim] BF16
 *   v_proj [tokens, kv_heads, head_dim] BF16
 *   q/k norm weights [head_dim] BF16, checkpoint convention (1 + weight)
 *   cos/sin [tokens, rotary_dim] F32
 * The normalized/rotated K and V are appended to the external BF16 cache.
 * If device_status later becomes non-ok, the caller must discard output and
 * rollback the transaction (fail closed).
 */
[[nodiscard]] status stage_cuda(
    const config& value,
    cache_state* state,
    const std::uint16_t* q_proj_bf16,
    const std::uint16_t* k_proj_bf16,
    const std::uint16_t* v_proj_bf16,
    const std::uint16_t* q_norm_weight_bf16,
    const std::uint16_t* k_norm_weight_bf16,
    const float* cos_f32,
    const float* sin_f32,
    std::size_t token_count,
    void* workspace,
    std::size_t workspace_capacity_bytes,
    staged_batch* batch,
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;

/*
 * QSA eager attention over caller-provided device lists. selected_indices is
 * [tokens, selected_stride], selected_counts is [tokens], and every index is
 * an absolute cache token ordinal.  Each row is checked against both the
 * current visible cache and its query's causal boundary.  Scores and stable
 * softmax are F32; output is BF16 [tokens, query_heads*head_dim] before o_proj.
 */
[[nodiscard]] status forward_qsa_cuda(
    const config& value,
    const cache_state* state,
    const staged_batch& batch,
    const std::int32_t* selected_indices,
    const std::uint32_t* selected_counts,
    std::size_t selected_stride,
    std::uint16_t* output_bf16,
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp::attention

#endif
