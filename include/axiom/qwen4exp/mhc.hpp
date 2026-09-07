#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

inline constexpr std::size_t kMhcQwen4ExpHidden = 2560;
inline constexpr std::size_t kMhcQwen4ExpStreams = 4;
inline constexpr std::size_t kMhcQwen4ExpResidual = 10240;
inline constexpr std::size_t kMhcQwen4ExpRank = 320;
inline constexpr float kMhcQwen4ExpRmsEpsilon = 1.0e-6F;

// The implementation is intentionally bounded. Admission may select any shape
// within these limits for correctness probes, while the production checkpoint
// must additionally satisfy mhc_is_qwen4_exp_contract().
inline constexpr std::size_t kMhcMaxHidden = kMhcQwen4ExpHidden;
inline constexpr std::size_t kMhcMaxStreams = kMhcQwen4ExpStreams;
inline constexpr std::size_t kMhcMaxResidual = kMhcQwen4ExpResidual;
inline constexpr std::size_t kMhcMaxRank = kMhcQwen4ExpRank;
inline constexpr std::size_t kMhcMaxTokensPerLaunch = 65535;

enum class MhcStatus : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kUnsupportedConfig,
    kSizeOverflow,
    kInsufficientWorkspace,
    kCudaError,
};

struct MhcConfig {
    std::size_t hidden = kMhcQwen4ExpHidden;
    std::size_t streams = kMhcQwen4ExpStreams;
    std::size_t rank = kMhcQwen4ExpRank;
    float rms_epsilon = kMhcQwen4ExpRmsEpsilon;
};

// All matrices are contiguous row-major F32:
//   hc_norm_weight       [streams * hidden]
//   input_mix_weight_down[rank, streams * hidden]
//   input_mix_weight_up  [streams * hidden, rank]
//   block_inject_weight  [streams, streams * hidden]
// block_inject_weight is required by prepare() and ignored by finalize().
struct MhcWeightsView {
    const float* hc_norm_weight = nullptr;
    const float* input_mix_weight_down = nullptr;
    const float* input_mix_weight_up = nullptr;
    const float* block_inject_weight = nullptr;
};

[[nodiscard]] MhcConfig mhc_qwen4_exp_config() noexcept;
[[nodiscard]] MhcStatus mhc_validate_config(const MhcConfig& config) noexcept;
[[nodiscard]] bool mhc_is_qwen4_exp_contract(const MhcConfig& config) noexcept;
[[nodiscard]] const char* mhc_status_string(MhcStatus status) noexcept;

[[nodiscard]] std::size_t mhc_residual_size(const MhcConfig& config) noexcept;

// Workspace contains normalized residual streams followed by the low-rank
// activation. Zero means invalid dimensions, zero tokens, or overflow.
[[nodiscard]] std::size_t mhc_workspace_floats(
    const MhcConfig& config,
    std::size_t token_count) noexcept;

// Exact host reference for the Qwen4-Exp gated residual:
//   normalized = RMSNorm(stream, scale = 1 + weight)
//   low_rank = SiLU(down(normalized) / streams)
//   mix = sigmoid(up(low_rank))
//   mixed_input[h] = mean_s(mix[s,h] * normalized[s,h])
//   injection[s] = 2 * sigmoid(block_inject[s](normalized) / streams)
// Buffers must be non-overlapping. No heap allocation is performed.
[[nodiscard]] MhcStatus mhc_prepare_host(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* mixed_input,
    float* injection_weights,
    float* workspace,
    std::size_t workspace_floats) noexcept;

// Add a block result back into all residual streams:
// output[t,s,h] = hyper_input[t,s,h]
//                 + injection_weights[t,s] * block_output[t,h].
[[nodiscard]] MhcStatus mhc_reinject_host(
    const MhcConfig& config,
    const float* hyper_input,
    const float* block_output,
    const float* injection_weights,
    std::size_t token_count,
    float* output) noexcept;

// Final model-level 4-stream -> hidden collapse. This is the same input mixer
// as prepare(), without block injection.
[[nodiscard]] MhcStatus mhc_finalize_host(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* output,
    float* workspace,
    std::size_t workspace_floats) noexcept;

// CUDA variants use device pointers and enqueue work on stream. They never
// allocate, synchronize, or touch the current production runtime.
[[nodiscard]] MhcStatus mhc_prepare_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* mixed_input,
    float* injection_weights,
    float* workspace,
    std::size_t workspace_floats,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] MhcStatus mhc_reinject_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const float* block_output,
    const float* injection_weights,
    std::size_t token_count,
    float* output,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] MhcStatus mhc_finalize_cuda(
    const MhcConfig& config,
    const float* hyper_input,
    const MhcWeightsView& weights,
    std::size_t token_count,
    float* output,
    float* workspace,
    std::size_t workspace_floats,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp
