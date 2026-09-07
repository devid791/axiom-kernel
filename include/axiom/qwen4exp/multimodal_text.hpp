#ifndef AXIOM_QWEN4EXP_MULTIMODAL_TEXT_HPP
#define AXIOM_QWEN4EXP_MULTIMODAL_TEXT_HPP

#include "axiom/qwen4exp/rope.hpp"
#include "axiom/qwen4exp/vision_adapter.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace axiom::qwen4exp {

/* Exact special-token contract admitted from the pinned qwen4_exp config. */
inline constexpr std::uint32_t kVisionStartTokenId = 248053u;
inline constexpr std::uint32_t kVisionEndTokenId = 248054u;
inline constexpr std::uint32_t kImageTokenId = 248056u;
inline constexpr std::uint32_t kVideoTokenId = 248057u;
inline constexpr std::size_t kMultimodalPositionAxes = 3u;
inline constexpr std::size_t kMultimodalHidden = kVisionOutputSize;
inline constexpr std::size_t kMultimodalStreams = 4u;
inline constexpr std::size_t kMultimodalResidual =
        kMultimodalStreams * kMultimodalHidden;

static_assert(kMultimodalHidden == 2560u);
static_assert(rope::kMaxContext == 262144u);

enum class multimodal_text_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_contract,
    malformed_vision_span,
    vision_token_mismatch,
    invalid_mrope,
    size_overflow,
    unsupported_device,
    invalid_device_pointer,
    non_finite_embedding,
    cuda_error,
};

[[nodiscard]] const char *multimodal_text_status_string(
        multimodal_text_status status) noexcept;

/* A processor-expanded prompt.  token_ids and mrope_positions are host
 * memory.  mrope_positions is axis-major [3, token_count] in temporal,
 * height, width order.  projected_vision_embeddings_device is the direct F32
 * output of resident_vision_provider::forward_patches_device(), concatenated
 * in placeholder order as [projected_vision_tokens, 2560].
 *
 * Each <|vision_start|> span must contain one or more homogeneous image_pad or
 * video_pad tokens and terminate with <|vision_end|>.  Every visual
 * placeholder consumes exactly one projected row.  Non-visual tokens must
 * carry identical temporal/height/width positions. */
struct multimodal_text_prompt_view {
    const std::uint32_t *token_ids = nullptr;
    std::size_t token_count = 0u;
    const float *projected_vision_embeddings_device = nullptr;
    std::size_t projected_vision_tokens = 0u;
    std::size_t projected_vision_width = kMultimodalHidden;
    const std::uint32_t *mrope_positions = nullptr;
    std::size_t mrope_position_tokens = 0u;
};

struct multimodal_text_prompt_report {
    std::size_t image_tokens = 0u;
    std::size_t video_tokens = 0u;
    std::size_t vision_tokens = 0u;
    std::size_t vision_spans = 0u;
    std::uint32_t maximum_position = 0u;
    std::uint32_t next_text_position = 0u;
    bool next_text_position_available = false;
};

/* Fail-closed host/device admission.  It validates prompt grammar, exact
 * placeholder/projected-row cardinality, the native 262K MRoPE domain, text
 * axis equality, tensor width and the complete device allocation range. */
[[nodiscard]] multimodal_text_status validate_multimodal_text_prompt(
        const multimodal_text_prompt_view &prompt,
        int expected_device,
        multimodal_text_prompt_report *report = nullptr) noexcept;

/* Allocation-free hot-path injection for one already projected visual row.
 * The row is repeated into the four mHC residual streams.  Non-finite input
 * is zeroed and atomically marks device_status nonzero; the caller collects
 * that status at its existing transaction synchronization boundary.  This
 * function never embeds the image/video placeholder token. */
[[nodiscard]] multimodal_text_status repeat_projected_vision_embedding_cuda(
        const float *projected_embedding_2560_f32,
        float *residual_4x2560_f32,
        std::uint32_t *device_status,
        int expected_device,
        cudaStream_t stream = nullptr) noexcept;

}  // namespace axiom::qwen4exp

#endif
