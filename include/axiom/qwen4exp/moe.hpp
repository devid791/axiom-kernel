#ifndef AXIOM_QWEN4EXP_MOE_HPP
#define AXIOM_QWEN4EXP_MOE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

inline constexpr std::uint32_t kMoeHidden = 2560u;
inline constexpr std::uint32_t kMoeIntermediate = 640u;
inline constexpr std::uint32_t kMoeExperts = 512u;
inline constexpr std::uint32_t kMoeTopK = 10u;
inline constexpr std::uint32_t kMoeNvfp4Group = 16u;

enum class moe_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    non_finite,
    cuda_error,
    kernel_error,
};

struct moe_config {
    std::uint32_t hidden = kMoeHidden;
    std::uint32_t intermediate = kMoeIntermediate;
    std::uint32_t experts = kMoeExperts;
    std::uint32_t top_k = kMoeTopK;
    bool normalize_topk = true;
};

/* Contiguous slot-major NVFP4 planes.  The pager bridge packs selected global
 * experts into these slots without changing any checkpoint value:
 *   gate/up weight [slots, intermediate, hidden/2]
 *   gate/up scale  [slots, intermediate, hidden/16]
 *   down weight    [slots, hidden, intermediate/2]
 *   down scale     [slots, hidden, intermediate/16]
 *   globals        [slots] F32 weight_scale_2.
 */
struct moe_slot_planes {
    const std::uint8_t *gate_weight = nullptr;
    const std::uint8_t *gate_block_scale = nullptr;
    const float *gate_global_scale = nullptr;
    const std::uint8_t *up_weight = nullptr;
    const std::uint8_t *up_block_scale = nullptr;
    const float *up_global_scale = nullptr;
    const std::uint8_t *down_weight = nullptr;
    const std::uint8_t *down_block_scale = nullptr;
    const float *down_global_scale = nullptr;
};

moe_config moe_qwen4_exp_config() noexcept;
moe_status moe_validate_config(const moe_config &config) noexcept;
bool moe_is_qwen4_exp_contract(const moe_config &config) noexcept;
const char *moe_status_string(moe_status status) noexcept;

/* Official Qwen3-Next router semantics: softmax over all experts, top-k, then
 * normalize the selected probabilities.  With normalize_topk=true the global
 * softmax denominator cancels; implementations compute the numerically stable
 * selected softmax directly.  Ties are resolved by the lower expert id. */
moe_status moe_router_topk_host(const moe_config &config,
                               const float *logits,
                               std::uint32_t *indices,
                               float *weights) noexcept;

/* One-block device router. device_status receives zero on success or the
 * numeric moe_status value; it lets a graph check non-finite input without a
 * hidden host synchronization. */
moe_status moe_router_topk_cuda(const moe_config &config,
                               const float *logits,
                               std::uint32_t *indices,
                               float *weights,
                               std::uint32_t *device_status,
                               cudaStream_t stream = nullptr) noexcept;

/* Run the existing generic Axiom fused NVFP4 MoE kernel on pager-resident
 * slots. slot_indices are device ids into the packed planes, normally 0..9;
 * router_weights follow the official normalized top-k contract.  This wrapper
 * is additive and does not alter the existing Qwen/DeepSeek providers. */
moe_status moe_forward_slots_f32_cuda(
        int device,
        const moe_config &config,
        std::uint32_t resident_slots,
        const moe_slot_planes &planes,
        const std::uint32_t *slot_indices,
        const float *router_weights,
        const float *input,
        float *scratch_mid,
        float *output,
        cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp

#endif
