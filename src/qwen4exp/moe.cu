#include "axiom/qwen4exp/moe.hpp"

#include "axiom/axiom.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr std::uint32_t kMaxExperts = 4096u;
constexpr std::uint32_t kMaxTopK = 64u;
constexpr unsigned kRouterThreads = 256u;
constexpr unsigned kRouterWarps = kRouterThreads / 32u;

// All lanes participate. Comparing the index as well as the value preserves
// the scalar scan's lower-index tie order, including +0.0f versus -0.0f.
__device__ void router_warp_best(float &value, std::uint32_t &index) {
    for (unsigned offset = 16u; offset != 0u; offset >>= 1u) {
        const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
        const std::uint32_t other_index =
                __shfl_down_sync(0xffffffffu, index, offset);
        if (other_value > value ||
            (other_value == value && other_index < index)) {
            value = other_value;
            index = other_index;
        }
    }
}

bool finite_positive(float value) noexcept {
    return std::isfinite(value) && value > 0.0f;
}

__global__ void router_topk_kernel(const float *logits,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   bool normalize_topk,
                                   std::uint32_t *indices,
                                   float *weights,
                                   std::uint32_t *status) {
    // One block, with bounded shared storage for every validated expert count.
    // Only this selection cache is masked; weight math below reads the original
    // logits so its GPU operations and serial accumulation order stay unchanged.
    extern __shared__ float candidates[];
    __shared__ float warp_values[kRouterWarps];
    __shared__ std::uint32_t warp_indices[kRouterWarps];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid % 32u;
    const unsigned warp = tid / 32u;
    bool non_finite = false;
    for (std::uint32_t expert = tid; expert < experts; expert += kRouterThreads) {
        const float value = logits[expert];
        candidates[expert] = value;
        non_finite = non_finite || !isfinite(value);
    }
    // Also publishes all cached logits, and makes rejection block-uniform.
    if (__syncthreads_or(non_finite)) {
        if (tid == 0u) {
            *status = static_cast<std::uint32_t>(moe_status::non_finite);
            for (std::uint32_t slot = 0; slot < top_k; ++slot) {
                indices[slot] = UINT32_MAX;
                weights[slot] = 0.0f;
            }
        }
        return;
    }
    if (tid == 0u) *status = static_cast<std::uint32_t>(moe_status::ok);

    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        float best = -INFINITY;
        std::uint32_t best_index = UINT32_MAX;
        for (std::uint32_t expert = tid; expert < experts; expert += kRouterThreads) {
            const float value = candidates[expert];
            if (value > best || (value == best && expert < best_index)) {
                best = value;
                best_index = expert;
            }
        }
        router_warp_best(best, best_index);
        if (lane == 0u) {
            warp_values[warp] = best;
            warp_indices[warp] = best_index;
        }
        __syncthreads();
        if (warp == 0u) {
            best = lane < kRouterWarps ? warp_values[lane] : -INFINITY;
            best_index = lane < kRouterWarps ? warp_indices[lane] : UINT32_MAX;
            router_warp_best(best, best_index);
            if (lane == 0u) {
                indices[slot] = best_index;
                // Inputs are finite and top_k <= experts, so an unselected
                // finite candidate always beats this exclusion sentinel.
                candidates[best_index] = -INFINITY;
            }
        }
        // Publish the exclusion before any thread starts the next selection;
        // also finish all reads of the warp scratch before it can be reused.
        __syncthreads();
    }

    if (tid != 0u) return;
    const float maximum = logits[indices[0]];
    float selected_sum = 0.0f;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        weights[slot] = expf(logits[indices[slot]] - maximum);
        selected_sum += weights[slot];
    }
    if (!isfinite(selected_sum) || selected_sum <= 0.0f) {
        *status = static_cast<std::uint32_t>(moe_status::non_finite);
        return;
    }
    if (normalize_topk) {
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
            weights[slot] /= selected_sum;
        }
    } else {
        float global_sum = 0.0f;
        for (std::uint32_t expert = 0; expert < experts; ++expert) {
            global_sum += expf(logits[expert] - maximum);
        }
        if (!isfinite(global_sum) || global_sum <= 0.0f) {
            *status = static_cast<std::uint32_t>(moe_status::non_finite);
            return;
        }
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
            weights[slot] /= global_sum;
        }
    }
}

}  // namespace

moe_config moe_qwen4_exp_config() noexcept { return {}; }

moe_status moe_validate_config(const moe_config &config) noexcept {
    if (config.hidden == 0u || config.intermediate == 0u ||
        config.experts == 0u || config.top_k == 0u ||
        config.top_k > config.experts) {
        return moe_status::invalid_argument;
    }
    if (config.hidden > (1u << 20u) ||
        config.intermediate > (1u << 20u) ||
        config.experts > kMaxExperts || config.top_k > kMaxTopK ||
        config.hidden % 32u != 0u || config.intermediate % 32u != 0u) {
        return moe_status::unsupported_config;
    }
    return moe_status::ok;
}

bool moe_is_qwen4_exp_contract(const moe_config &config) noexcept {
    return moe_validate_config(config) == moe_status::ok &&
           config.hidden == kMoeHidden &&
           config.intermediate == kMoeIntermediate &&
           config.experts == kMoeExperts &&
           config.top_k == kMoeTopK && config.normalize_topk;
}

const char *moe_status_string(moe_status status_value) noexcept {
    switch (status_value) {
        case moe_status::ok: return "ok";
        case moe_status::invalid_argument: return "invalid_argument";
        case moe_status::unsupported_config: return "unsupported_config";
        case moe_status::non_finite: return "non_finite";
        case moe_status::cuda_error: return "cuda_error";
        case moe_status::kernel_error: return "kernel_error";
    }
    return "unknown";
}

moe_status moe_router_topk_host(const moe_config &config,
                                const float *logits,
                                std::uint32_t *indices,
                                float *weights) noexcept {
    const moe_status validation = moe_validate_config(config);
    if (validation != moe_status::ok) return validation;
    if (logits == nullptr || indices == nullptr || weights == nullptr) {
        return moe_status::invalid_argument;
    }
    for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
        if (!std::isfinite(logits[expert])) return moe_status::non_finite;
    }
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
        float best = -std::numeric_limits<float>::infinity();
        std::uint32_t best_index = UINT32_MAX;
        for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
            bool used = false;
            for (std::uint32_t previous = 0; previous < slot; ++previous) {
                used = used || indices[previous] == expert;
            }
            const float value = logits[expert];
            if (!used && (value > best ||
                          (value == best && expert < best_index))) {
                best = value;
                best_index = expert;
            }
        }
        indices[slot] = best_index;
    }
    const float maximum = logits[indices[0]];
    float selected_sum = 0.0f;
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
        weights[slot] = std::exp(logits[indices[slot]] - maximum);
        selected_sum += weights[slot];
    }
    if (!finite_positive(selected_sum)) return moe_status::non_finite;
    if (config.normalize_topk) {
        for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
            weights[slot] /= selected_sum;
        }
    } else {
        float global_sum = 0.0f;
        for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
            global_sum += std::exp(logits[expert] - maximum);
        }
        if (!finite_positive(global_sum)) return moe_status::non_finite;
        for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
            weights[slot] /= global_sum;
        }
    }
    return moe_status::ok;
}

moe_status moe_router_topk_cuda(const moe_config &config,
                                const float *logits,
                                std::uint32_t *indices,
                                float *weights,
                                std::uint32_t *device_status,
                                cudaStream_t stream) noexcept {
    const moe_status validation = moe_validate_config(config);
    if (validation != moe_status::ok) return validation;
    if (logits == nullptr || indices == nullptr || weights == nullptr ||
        device_status == nullptr) {
        return moe_status::invalid_argument;
    }
    router_topk_kernel<<<1, kRouterThreads, config.experts * sizeof(float), stream>>>(
            logits, config.experts, config.top_k, config.normalize_topk,
            indices, weights, device_status);
    return cudaGetLastError() == cudaSuccess ? moe_status::ok
                                             : moe_status::cuda_error;
}

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
        cudaStream_t stream) noexcept {
    const moe_status validation = moe_validate_config(config);
    if (validation != moe_status::ok) return validation;
    // Physical slots may span all 48 layers; routing remains local to the
    // layer's 512 experts and bridge-selected indices address the shared arena.
    if (device < 0 || resident_slots == 0u || resident_slots > 48u * config.experts ||
        config.top_k > resident_slots ||
        planes.gate_weight == nullptr || planes.gate_block_scale == nullptr ||
        planes.gate_global_scale == nullptr || planes.up_weight == nullptr ||
        planes.up_block_scale == nullptr || planes.up_global_scale == nullptr ||
        planes.down_weight == nullptr || planes.down_block_scale == nullptr ||
        planes.down_global_scale == nullptr || slot_indices == nullptr ||
        router_weights == nullptr || input == nullptr || scratch_mid == nullptr ||
        output == nullptr) {
        return moe_status::invalid_argument;
    }
    const int result = axiom_cuda_deepseek_moe_nvfp4_indexed_fused_f32(
            device,
            planes.gate_weight, planes.gate_block_scale, planes.gate_global_scale,
            planes.up_weight, planes.up_block_scale, planes.up_global_scale,
            planes.down_weight, planes.down_block_scale, planes.down_global_scale,
            slot_indices, router_weights, input, scratch_mid, output,
            reinterpret_cast<void *>(stream), resident_slots, config.top_k,
            config.hidden, config.intermediate);
    if (result == AXIOM_OK) return moe_status::ok;
    if (result == AXIOM_ERR_INVALID_ARGUMENT) return moe_status::invalid_argument;
    if (result == AXIOM_ERR_CUDA) return moe_status::cuda_error;
    return moe_status::kernel_error;
}

}  // namespace axiom::qwen4exp
