#ifndef AXIOM_QWEN4EXP_ROPE_HPP
#define AXIOM_QWEN4EXP_ROPE_HPP

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp::rope {

inline constexpr std::uint32_t kRotaryDim = 64u;
inline constexpr std::uint32_t kRotaryHalf = 32u;
inline constexpr std::uint32_t kMaxContext = 262144u;
inline constexpr float kTheta = 10000000.0F;
inline constexpr std::array<std::uint32_t, 3> kMropeSections{11u, 11u, 10u};

struct config {
    std::uint32_t rotary_dim = kRotaryDim;
    std::uint32_t max_context = kMaxContext;
    float theta = kTheta;
    std::array<std::uint32_t, 3> mrope_sections = kMropeSections;
};

inline constexpr config checkpoint_config{};

enum class status : std::uint32_t {
    ok = 0u,
    null_pointer,
    invalid_config,
    invalid_argument,
    arithmetic_overflow,
    unsupported_device,
    invalid_state,
    non_finite,
    cuda_failure,
};

struct device_plan {
    config value{};
    float* inverse_frequency = nullptr;
    int device = -1;
    bool initialized = false;
};

[[nodiscard]] const char* status_string(status value) noexcept;
[[nodiscard]] status validate_config(const config& value) noexcept;
[[nodiscard]] status validate_checkpoint_config(const config& value) noexcept;

/* Initialization is the only allocating/synchronizing phase. */
[[nodiscard]] status initialize_device_plan(
    device_plan* plan,
    const config& value = checkpoint_config,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] status release_device_plan(device_plan* plan) noexcept;

/* positions is axis-major [3, token_count]: temporal, height, width.
 * Outputs are token-major [token_count, rotary_dim]. */
[[nodiscard]] status generate_host(
    const config& value,
    const std::uint32_t* positions,
    std::size_t token_count,
    float* cosine,
    float* sine) noexcept;

/* Hot-path CUDA calls allocate and synchronize nothing.  Device status is
 * fail-closed for invalid positions/non-finite output and must be reset by the
 * caller before a pipeline and collected at a chosen synchronization boundary.
 */
[[nodiscard]] status reset_device_status(
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] status collect_device_status(
    const status* device_status,
    status* host_status,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] status generate_mrope_cuda(
    const device_plan* plan,
    const std::uint32_t* positions,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;
[[nodiscard]] status generate_text_cuda(
    const device_plan* plan,
    std::uint32_t start_position,
    std::size_t token_count,
    float* cosine,
    float* sine,
    status* device_status,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp::rope

#endif
