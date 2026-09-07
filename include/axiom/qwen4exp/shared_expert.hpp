#ifndef AXIOM_QWEN4EXP_SHARED_EXPERT_HPP
#define AXIOM_QWEN4EXP_SHARED_EXPERT_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

inline constexpr std::uint32_t kSharedExpertHidden = 2560u;
inline constexpr std::uint32_t kSharedExpertIntermediate = 640u;
inline constexpr std::uint32_t kSharedExpertLayers = 48u;

enum class shared_expert_status : std::uint32_t {
    ok = 0u,
    invalid_argument,
    unsupported_config,
    overflow,
    unsupported_device,
    invalid_device_pointer,
    non_finite,
    cuda_error,
};

enum class shared_expert_tensor : std::uint8_t {
    gate_projection = 0u,
    up_projection,
    down_projection,
    output_gate,
};

struct shared_expert_config {
    std::uint32_t hidden = kSharedExpertHidden;
    std::uint32_t intermediate = kSharedExpertIntermediate;
};

struct shared_expert_tensor_spec {
    std::uint64_t rows = 0u;
    std::uint64_t columns = 0u;
    std::uint64_t bytes = 0u;
};

/* Row-major BF16 checkpoint planes.  The exact pinned qwen4_exp contract is:
 *   gate/up  [intermediate, hidden]
 *   down     [hidden, intermediate]
 *   out gate [1, hidden]
 * No bias or quantization scale is present for these four tensors. */
struct shared_expert_weights_bf16 {
    const std::uint16_t *gate = nullptr;
    const std::uint16_t *up = nullptr;
    const std::uint16_t *down = nullptr;
    const std::uint16_t *output_gate = nullptr;
};

/* All storage is caller-owned.  The provider never allocates or synchronizes.
 * device_status is written asynchronously with a shared_expert_status value;
 * inspect it only after the supplied stream reaches the call. */
struct shared_expert_workspace_f32 {
    float *intermediate = nullptr;          // config.intermediate elements
    float *output_gate = nullptr;           // one element
    std::uint32_t *device_status = nullptr; // one element
};

[[nodiscard]] shared_expert_config shared_expert_qwen4_exp_config() noexcept;
[[nodiscard]] shared_expert_status shared_expert_validate_config(
        const shared_expert_config &config) noexcept;
[[nodiscard]] bool shared_expert_is_qwen4_exp_contract(
        const shared_expert_config &config) noexcept;
[[nodiscard]] const char *shared_expert_status_string(
        shared_expert_status status) noexcept;

[[nodiscard]] shared_expert_status shared_expert_tensor_spec_for(
        const shared_expert_config &config,
        shared_expert_tensor tensor,
        shared_expert_tensor_spec *spec) noexcept;

/* Writes an exact language-layer checkpoint tensor name without allocating.
 * Valid layers are [0, 47]. */
[[nodiscard]] shared_expert_status shared_expert_tensor_name(
        std::uint32_t layer,
        shared_expert_tensor tensor,
        char *destination,
        std::size_t destination_bytes) noexcept;

/* Allocation-free host reference using BF16 checkpoint weights and F32
 * accumulation.  It implements the inherited official Qwen3-Next semantics:
 *
 *   ffn = down(silu(gate(x)) * up(x))
 *   out = sigmoid(output_gate(x)) * ffn
 */
[[nodiscard]] shared_expert_status shared_expert_forward_f32_host(
        const shared_expert_config &config,
        const shared_expert_weights_bf16 &weights,
        const float *input,
        float *scratch_intermediate,
        float *scratch_output_gate,
        float *output) noexcept;

/* Allocation-free CUDA provider adapter for the same expression.  All
 * pointers must address memory visible to `device`; the current CUDA device
 * must already equal `device`.  A return value of ok means the kernels were
 * enqueued.  Non-finite data is reported asynchronously through device_status
 * and the final output is zeroed fail-closed. */
[[nodiscard]] shared_expert_status shared_expert_forward_f32_cuda(
        int device,
        const shared_expert_config &config,
        const shared_expert_weights_bf16 &weights,
        const float *input,
        const shared_expert_workspace_f32 &workspace,
        float *output,
        cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp

#endif
