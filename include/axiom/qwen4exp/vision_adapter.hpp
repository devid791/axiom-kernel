#ifndef AXIOM_QWEN4EXP_VISION_ADAPTER_HPP
#define AXIOM_QWEN4EXP_VISION_ADAPTER_HPP

#include "axiom/qwen4exp/checkpoint.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace axiom::qwen4exp {

/* Flash-Next has its own vision contract.  In particular, kVisionOutputSize
 * is 2560 and is intentionally unrelated to the legacy qwen38 5120-wide ABI. */
inline constexpr std::size_t kVisionDepth = 27u;
inline constexpr std::size_t kVisionHiddenSize = 1152u;
inline constexpr std::size_t kVisionIntermediateSize = 4304u;
inline constexpr std::size_t kVisionHeadCount = 16u;
inline constexpr std::size_t kVisionHeadSize = 72u;
inline constexpr std::size_t kVisionPatchSize = 16u;
inline constexpr std::size_t kVisionTemporalPatchSize = 2u;
inline constexpr std::size_t kVisionSpatialMergeSize = 2u;
inline constexpr std::size_t kVisionPositionSide = 48u;
inline constexpr std::size_t kVisionPositionCount = 2304u;
inline constexpr std::size_t kVisionPatchFeatures = 1536u;
inline constexpr std::size_t kVisionMergedHiddenSize = 4608u;
inline constexpr std::size_t kVisionOutputSize = 2560u;
inline constexpr std::size_t kVisionTensorCount = 333u;
inline constexpr std::uint32_t kVisionImageMinPixels = 65536u;
inline constexpr std::uint32_t kVisionImageMaxPixels = 16777216u;
inline constexpr std::uint32_t kVisionVideoMinPixels = 4096u;
inline constexpr std::uint32_t kVisionVideoMaxPixels = 25165824u;

static_assert(kVisionHiddenSize == kVisionHeadCount * kVisionHeadSize);
static_assert(kVisionPatchFeatures ==
              3u * kVisionTemporalPatchSize * kVisionPatchSize * kVisionPatchSize);
static_assert(kVisionMergedHiddenSize ==
              kVisionHiddenSize * kVisionSpatialMergeSize * kVisionSpatialMergeSize);
static_assert(kVisionOutputSize != 5120u,
              "Flash-Next vision output must not alias the qwen38 ABI");

enum class vision_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    tensor_not_found,
    tensor_contract_mismatch,
    size_overflow,
    checkpoint_io_error,
    unsupported_device,
    allocation_failure,
    cuda_error,
    cublas_error,
};

[[nodiscard]] const char *vision_status_string(vision_status status) noexcept;

struct vision_grid {
    std::uint32_t temporal = 0u;  // groups after temporal_patch_size=2
    std::uint32_t height = 0u;    // patches, not pixels
    std::uint32_t width = 0u;     // patches, not pixels
};

struct vision_rgb8_frame {
    const std::uint8_t *pixels = nullptr;
    std::uint32_t width = 0u;
    std::uint32_t height = 0u;
    std::size_t row_stride_bytes = 0u;
};

struct vision_preprocessed_media {
    vision_grid grid{};
    std::uint32_t resized_width = 0u;
    std::uint32_t resized_height = 0u;
    std::uint32_t source_frames = 0u;

    /* Exact HF processor/Conv3D input order.  Patches are spatially ordered
     * [T, H/2, W/2, merge_y, merge_x], and each flattened patch is
     * [C, temporal, patch_y, patch_x]. Values use checkpoint mean/std 0.5. */
    std::vector<float> patches;

    [[nodiscard]] std::uint64_t patch_count() const noexcept;
    [[nodiscard]] std::uint64_t merged_token_count() const noexcept;
};

/* These entry points start from already-decoded, interleaved RGB8.  Codec and
 * network policy stay outside the provider.  Resizing, [-1,1] normalization,
 * odd-frame replication, and HF block-major patchification are native C++17;
 * no Python runtime is involved. */
[[nodiscard]] vision_status vision_preprocess_image_rgb8(
        const vision_rgb8_frame &image,
        std::uint32_t min_pixels,
        std::uint32_t max_pixels,
        vision_preprocessed_media *out,
        std::string *error) noexcept;

[[nodiscard]] vision_status vision_preprocess_video_rgb8(
        const vision_rgb8_frame *frames,
        std::size_t frame_count,
        std::uint32_t min_pixels,
        std::uint32_t max_pixels,
        vision_preprocessed_media *out,
        std::string *error) noexcept;

struct vision_contract_report {
    std::size_t tensor_count = 0u;
    std::uint64_t tensor_bytes = 0u;
    std::size_t depth = 0u;
    std::size_t hidden_size = 0u;
    std::size_t intermediate_size = 0u;
    std::size_t num_heads = 0u;
    std::size_t patch_size = 0u;
    std::size_t temporal_patch_size = 0u;
    std::size_t spatial_merge_size = 0u;
    std::size_t output_size = 0u;
};

/* Fail-closed admission over the public checkpoint catalog.  Exactly the 333
 * pinned model.visual.* names must be present as BF16 with exact ranks, shapes
 * and byte counts; unknown visual tensors are rejected as well as omissions. */
[[nodiscard]] vision_status vision_admit_checkpoint(
        const checkpoint_catalog &catalog,
        vision_contract_report *report,
        std::string *error) noexcept;

struct vision_provider_config {
    int device = 0;
    std::size_t max_patch_tokens = 4096u;
};

struct vision_provider_info {
    vision_contract_report contract{};
    std::size_t max_patch_tokens = 0u;
    std::uint64_t resident_weight_bytes = 0u;
    std::uint64_t resident_total_bytes = 0u;
    int device = -1;
};

/* Complete pinned tower: patch Conv3D -> learned/interpolated position -> 27
 * non-causal transformer blocks -> pre-shuffle LayerNorm -> 2x2 merger.  All
 * 333 BF16 visual tensors are owned by the instance.  Forward methods are
 * synchronous by design so their temporary host metadata cannot outlive the
 * call; one instance therefore admits only one in-flight invocation. */
class resident_vision_provider final {
public:
    resident_vision_provider();
    ~resident_vision_provider();
    resident_vision_provider(resident_vision_provider &&) noexcept;
    resident_vision_provider &operator=(resident_vision_provider &&) noexcept;
    resident_vision_provider(const resident_vision_provider &) = delete;
    resident_vision_provider &operator=(const resident_vision_provider &) = delete;

    [[nodiscard]] static vision_status load(
            const checkpoint_catalog &catalog,
            const vision_provider_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<resident_vision_provider> *out,
            std::string *error) noexcept;

    [[nodiscard]] vision_status forward_patches(
            const float *patch_values_host,
            std::size_t patch_value_count,
            const vision_grid *grids,
            std::size_t grid_count,
            float *output_host,
            std::size_t output_value_capacity,
            std::size_t *output_tokens,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] vision_status forward_patches_device(
            const float *patch_values_device,
            std::size_t patch_value_count,
            const vision_grid *grids,
            std::size_t grid_count,
            float *output_device,
            std::size_t output_value_capacity,
            std::size_t *output_tokens,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] vision_provider_info info() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
