#ifndef AXIOM_QWEN38_VISION_PREPROCESS_HPP
#define AXIOM_QWEN38_VISION_PREPROCESS_HPP

/* Native, dependency-light Qwen3.8 media preprocessing.
 *
 * This module deliberately stops at decoded RGB bytes and the exact patchified
 * tensor consumed by qwen38_vision.h.  It has no Python, subprocess, network,
 * or framework dependency.  Container video is not silently guessed: callers
 * pass an ordered frame sequence until a native container decoder is added.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "axiom/qwen38_vision.h"

namespace axiom::qwen38::vision {

struct rgb_image {
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<uint8_t> rgb;

    bool valid() const noexcept {
        return width != 0u && height != 0u &&
               rgb.size() == static_cast<size_t>(width) * height * 3u;
    }
};

struct preprocessed_media {
    axiom_qwen38_vision_grid grid{};
    uint32_t resized_width = 0u;
    uint32_t resized_height = 0u;
    uint32_t source_frames = 0u;
    std::vector<float> patches; /* [T*H*W, C*2*16*16] */

    uint64_t patch_count() const noexcept {
        return static_cast<uint64_t>(grid.temporal) * grid.height * grid.width;
    }
};

/* Decode an encoded image. `mime` may be empty; PNG/JPEG/WebP magic is then
 * detected.  The output is tightly packed interleaved RGB8. */
bool decode_image_bytes(
        const uint8_t *encoded,
        size_t encoded_size,
        const std::string &mime,
        rgb_image *out,
        std::string *error);

/* Decode a base64 data URL. Remote URLs are intentionally rejected here; the
 * HTTP layer must not introduce an implicit network fetch into preprocessing. */
bool decode_data_url(
        const std::string &data_url,
        rgb_image *out,
        std::string *error);

/* Qwen's image processor defaults from the installed checkpoint.  The values
 * are explicit so an API request cannot accidentally inherit a framework
 * default with a different visual-token budget. */
constexpr uint32_t kImageMinPixels = 65536u;
constexpr uint32_t kImageMaxPixels = 16777216u;
constexpr uint32_t kVideoMinPixels = 4096u;
constexpr uint32_t kVideoMaxPixels = 25165824u;

bool preprocess_image(
        const rgb_image &image,
        uint32_t min_pixels,
        uint32_t max_pixels,
        preprocessed_media *out,
        std::string *error);

/* The native video contract is an ordered sequence of decoded RGB frames.
 * Odd frame counts are padded by repeating the final frame, matching the
 * temporal patch size of two. All frames are resized to one common grid. */
bool preprocess_video_frames(
        const std::vector<rgb_image> &frames,
        uint32_t min_pixels,
        uint32_t max_pixels,
        preprocessed_media *out,
        std::string *error);

} // namespace axiom::qwen38::vision

#endif
