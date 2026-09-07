#ifndef AXIOM_QWEN4EXP_PLE_COMPUTE_HPP
#define AXIOM_QWEN4EXP_PLE_COMPUTE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

inline constexpr std::size_t kPleComputeHidden = 2560u;
inline constexpr std::size_t kPleComputeStreams = 4u;
inline constexpr std::size_t kPleComputeEmbedding = 2560u;
inline constexpr std::size_t kPleComputeHistory = 9u;
inline constexpr std::size_t kPleComputeTaps = 4u;
inline constexpr std::size_t kPleComputeDilation = 3u;

enum class ple_compute_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    invalid_state,
    size_overflow,
    unsupported_device,
    cuda_error,
};

struct ple_compute_config {
    std::size_t hidden = kPleComputeHidden;
    std::size_t streams = kPleComputeStreams;
    std::size_t embedding = kPleComputeEmbedding;
    std::size_t max_speculative_tokens = 32u;
    float rms_epsilon = 1.0e-6F;
};

struct ple_compute_footprint {
    std::size_t channels = 0u;
    std::size_t committed_state_floats = 0u;
    std::size_t staged_state_floats = 0u;
    std::size_t scratch_floats = 0u;
    std::size_t state_bytes = 0u;
};

/* All weights are immutable device-resident BF16 payloads expressed as their
 * raw 16-bit bits.  Layout is the exact pinned checkpoint layout:
 *   key_proj   [streams * hidden, embedding]
 *   value_proj [hidden, embedding]
 *   norm_*     [streams * hidden]
 *   conv       [streams * hidden, 1, 4]
 */
struct ple_compute_weights {
    const std::uint16_t *key_proj = nullptr;
    const std::uint16_t *value_proj = nullptr;
    const std::uint16_t *norm_key = nullptr;
    const std::uint16_t *norm_query = nullptr;
    const std::uint16_t *norm_conv = nullptr;
    const std::uint16_t *conv = nullptr;
};

/* Caller-owned, allocation-free per-step workspace.  Each pointer is device
 * memory. key/query/gated/normalized contain streams*hidden floats; value
 * contains hidden floats.  Buffers must not overlap each other or output. */
struct ple_compute_scratch {
    float *key = nullptr;
    float *value = nullptr;
    float *query = nullptr;
    float *gated = nullptr;
    float *normalized = nullptr;
};

/* Transactional device state for the PLE dilated convolution.  Committed
 * history is never overwritten by speculative steps. commit_prefix(k)
 * publishes only the accepted prefix; rollback discards every staged frame.
 * Host metadata is stream-ordered: all calls for one state must use the same
 * CUDA stream (or be externally ordered by events). */
struct ple_compute_device_state {
    ple_compute_config config{};
    float *committed_history = nullptr;
    float *staged_frames = nullptr;
    std::size_t committed_next = 0u;
    std::size_t committed_valid = 0u;
    std::size_t staged_count = 0u;
    std::uint64_t committed_tokens = 0u;
    int device = -1;
    bool initialized = false;
    bool transaction_open = false;
};

[[nodiscard]] ple_compute_config ple_compute_qwen4_exp_config() noexcept;
[[nodiscard]] ple_compute_status ple_compute_validate_config(
        const ple_compute_config &config) noexcept;
[[nodiscard]] bool ple_compute_is_qwen4_exp_contract(
        const ple_compute_config &config) noexcept;
[[nodiscard]] ple_compute_status ple_compute_get_footprint(
        const ple_compute_config &config,
        ple_compute_footprint *footprint) noexcept;
[[nodiscard]] const char *ple_compute_status_string(
        ple_compute_status status) noexcept;

[[nodiscard]] ple_compute_status ple_compute_device_state_init(
        ple_compute_device_state *state,
        const ple_compute_config &config,
        cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] ple_compute_status ple_compute_device_state_reset(
        ple_compute_device_state *state,
        cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] ple_compute_status ple_compute_device_state_release(
        ple_compute_device_state *state) noexcept;

[[nodiscard]] ple_compute_status ple_compute_begin_transaction(
        ple_compute_device_state *state) noexcept;

/* Exact one-token PLE compute from the official qwen4_exp graph:
 * projections -> grouped (1+w) RMSNorm -> signed-sqrt sigmoid gate ->
 * grouped RMSNorm -> causal dilated depthwise conv + SiLU -> residual add.
 * embedding is the 2560-element result of the bounded FP8 row gather and
 * hidden_states is the 4x2560 hyper-connection residual.  The normalized
 * gated-value frame is staged on device without a host synchronization. */
[[nodiscard]] ple_compute_status ple_compute_stage_token_cuda(
        ple_compute_device_state *state,
        const ple_compute_weights &weights,
        const float *embedding,
        const float *hidden_states,
        const ple_compute_scratch &scratch,
        float *output,
        cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] ple_compute_status ple_compute_commit_prefix_cuda(
        ple_compute_device_state *state,
        std::size_t accepted_tokens,
        cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] ple_compute_status ple_compute_rollback(
        ple_compute_device_state *state) noexcept;

}  // namespace axiom::qwen4exp

#endif
