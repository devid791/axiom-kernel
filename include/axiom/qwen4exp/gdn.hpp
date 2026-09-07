#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

inline constexpr std::size_t kGdnQwen4ExpKeyHeads = 16;
inline constexpr std::size_t kGdnQwen4ExpValueHeads = 48;
inline constexpr std::size_t kGdnQwen4ExpKeyHeadDim = 128;
inline constexpr std::size_t kGdnQwen4ExpValueHeadDim = 128;
inline constexpr std::size_t kGdnQwen4ExpConvKernel = 4;
inline constexpr float kGdnQwen4ExpRmsEpsilon = 1.0e-6F;
inline constexpr float kGdnQwen4ExpL2Epsilon = 1.0e-6F;
inline constexpr std::size_t kGdnMaxSequenceTokens = 4;

// Reduced configurations are accepted for deterministic correctness probes.
// The production checkpoint must additionally satisfy
// gdn_is_qwen4_exp_contract(). Limits are deliberately bounded so malformed
// metadata cannot turn into unbounded allocations or invalid CUDA grids.
inline constexpr std::size_t kGdnMaxBatch = 65535;
inline constexpr std::size_t kGdnMaxKeyHeads = 64;
inline constexpr std::size_t kGdnMaxValueHeads = 64;
inline constexpr std::size_t kGdnMaxHeadDim = 256;

enum class GdnStatus : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kUnsupportedConfig,
    kDimensionMismatch,
    kSizeOverflow,
    kUninitialized,
    kStateMismatch,
    kUnsupportedDevice,
    kCudaError,
};

struct GdnConfig {
    std::size_t batch = 1;
    std::size_t key_heads = kGdnQwen4ExpKeyHeads;
    std::size_t value_heads = kGdnQwen4ExpValueHeads;
    std::size_t key_head_dim = kGdnQwen4ExpKeyHeadDim;
    std::size_t value_head_dim = kGdnQwen4ExpValueHeadDim;
    std::size_t conv_kernel = kGdnQwen4ExpConvKernel;
    float rms_epsilon = kGdnQwen4ExpRmsEpsilon;
    float l2_epsilon = kGdnQwen4ExpL2Epsilon;
};

struct GdnFootprint {
    std::size_t key_elements = 0;
    std::size_t value_elements = 0;
    std::size_t conv_channels = 0;
    std::size_t conv_state_floats = 0;
    std::size_t recurrent_state_floats = 0;
    std::size_t step_workspace_floats = 0;
    std::size_t device_state_bytes = 0;
};

// Projection outputs for one token. Layouts are contiguous row-major F32:
//   projected_qkv [batch, 2 * key_heads * key_head_dim
//                            + value_heads * value_head_dim]
//   z             [batch, value_heads, value_head_dim]
//   a, b          [batch, value_heads]
struct GdnStepInputs {
    const float* projected_qkv = nullptr;
    const float* z = nullptr;
    const float* a = nullptr;
    const float* b = nullptr;
};

// Checkpoint weights converted to F32 by the provider. conv_bias is optional;
// the pinned Qwen4-Exp checkpoint uses a bias-free depthwise convolution.
// Layouts:
//   conv_weight [conv_channels, conv_kernel]
//   conv_bias   [conv_channels] or nullptr
//   A_log       [value_heads]
//   dt_bias     [value_heads]
//   norm_weight [value_head_dim]
struct GdnWeightsView {
    const float* conv_weight = nullptr;
    const float* conv_bias = nullptr;
    const float* A_log = nullptr;
    const float* dt_bias = nullptr;
    const float* norm_weight = nullptr;
};

// convolved_qkv is caller-owned step workspace [batch, conv_channels]. output
// is [batch, value_heads, value_head_dim]. The two buffers must not overlap.
struct GdnStepOutputs {
    float* convolved_qkv = nullptr;
    float* output = nullptr;
};

// Host state uses the same layouts as the device state and is intentionally a
// non-owning view so the reference path never hides allocation from callers.
struct GdnHostStateView {
    float* conv_state = nullptr;
    std::size_t conv_state_floats = 0;
    float* recurrent_state = nullptr;
    std::size_t recurrent_state_floats = 0;
    std::uint32_t* conv_cursor = nullptr;
    std::uint64_t* tokens_seen = nullptr;
};

// Owning CUDA state. Treat as non-copyable; initialize once and release once.
// Public fields make state residency/auditing explicit to the future provider.
struct GdnDeviceState {
    GdnConfig config{};
    float* conv_state = nullptr;
    float* recurrent_state = nullptr;
    std::size_t conv_state_floats = 0;
    std::size_t recurrent_state_floats = 0;
    std::uint32_t conv_cursor = 0;
    std::uint64_t tokens_seen = 0;
    int device = -1;
    bool initialized = false;
};

[[nodiscard]] GdnConfig gdn_qwen4_exp_config(std::size_t batch = 1) noexcept;
[[nodiscard]] GdnStatus gdn_validate_config(const GdnConfig& config) noexcept;
[[nodiscard]] bool gdn_is_qwen4_exp_contract(const GdnConfig& config) noexcept;
[[nodiscard]] GdnStatus gdn_footprint(
    const GdnConfig& config,
    GdnFootprint* footprint) noexcept;
[[nodiscard]] const char* gdn_status_string(GdnStatus status) noexcept;

[[nodiscard]] GdnStatus gdn_host_state_reset(
    const GdnConfig& config,
    const GdnHostStateView& state) noexcept;

// Exact scalar reference of Transformers causal_conv1d_update for one token:
// append the projected token to the K=4 state, apply depthwise cross-
// correlation over the newest four values, then SiLU.
[[nodiscard]] GdnStatus gdn_causal_conv_update_host(
    const GdnConfig& config,
    const float* projected_qkv,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    float* convolved_qkv) noexcept;

// Exact single-token recurrent gated delta rule used by qwen4_exp:
// L2-normalize Q/K, logical repeat_interleave key_heads -> value_heads,
// decay FP32 state, apply beta delta update, query the updated state, then
// per-head RMSNorm and the checkpoint's sigmoid z gate.
[[nodiscard]] GdnStatus gdn_recurrent_delta_update_host(
    const GdnConfig& config,
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    float* output) noexcept;

[[nodiscard]] GdnStatus gdn_step_host(
    const GdnConfig& config,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    const GdnStepOutputs& outputs) noexcept;

// Execute one contiguous causal sequence while retaining a single batch-one
// state. Input and output rows are token-major and contiguous. This API is not
// equivalent to gdn_step_host() with config.batch=token_count: every row sees
// all state updates made by the preceding rows.
[[nodiscard]] GdnStatus gdn_forward_sequence_host(
    const GdnConfig& config,
    std::size_t token_count,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnHostStateView& state,
    const GdnStepOutputs& outputs) noexcept;

// CUDA state is FP32 for both the depthwise-convolution history and recurrent
// [batch, value_heads, key_head_dim, value_head_dim] matrix. init/reset enqueue
// zeroing on stream; callers must preserve stream ordering before first use.
[[nodiscard]] GdnStatus gdn_device_state_init(
    GdnDeviceState* state,
    const GdnConfig& config,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] GdnStatus gdn_device_state_reset(
    GdnDeviceState* state,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] GdnStatus gdn_device_state_release(GdnDeviceState* state) noexcept;

// CUDA launches are asynchronous and allocation-free after state init. They
// validate dimensions, ownership device, integer products and immediate CUDA
// launch errors fail-closed. Runtime/asynchronous faults surface through the
// caller's stream synchronization.
[[nodiscard]] GdnStatus gdn_causal_conv_update_cuda(
    GdnDeviceState* state,
    const float* projected_qkv,
    const GdnWeightsView& weights,
    float* convolved_qkv,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] GdnStatus gdn_recurrent_delta_update_cuda(
    GdnDeviceState* state,
    const float* convolved_qkv,
    const float* z,
    const float* a,
    const float* b,
    const GdnWeightsView& weights,
    float* output,
    cudaStream_t stream = nullptr) noexcept;

[[nodiscard]] GdnStatus gdn_step_cuda(
    GdnDeviceState* state,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream = nullptr) noexcept;

// Allocation-free contiguous-sequence execution for speculative verification.
// The state contract is always batch=1 and token_count is bounded to
// kGdnMaxSequenceTokens. Convolution and recurrent state are advanced in
// causal token order using two CUDA launches independent of token_count.
[[nodiscard]] GdnStatus gdn_forward_sequence_cuda(
    GdnDeviceState* state,
    std::size_t token_count,
    const GdnStepInputs& inputs,
    const GdnWeightsView& weights,
    const GdnStepOutputs& outputs,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp
