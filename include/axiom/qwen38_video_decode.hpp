#ifndef AXIOM_QWEN38_VIDEO_DECODE_HPP
#define AXIOM_QWEN38_VIDEO_DECODE_HPP

/* Native container video decode boundary for Qwen3.8.
 *
 * This is a C++ libavformat/libavcodec/libswscale integration. It reads only
 * caller-owned memory, never opens a URL, never invokes ffmpeg as a process,
 * and returns an ordered RGB frame sequence to the Qwen preprocessor. The
 * limits are part of the API so a hostile video cannot allocate unbounded
 * host memory before the model's visual-token budget is applied.
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "axiom/qwen38_vision_preprocess.hpp"

namespace axiom::qwen38::vision {

struct video_decode_limits {
    uint32_t max_frames = 64u;
    uint32_t max_width = 4096u;
    uint32_t max_height = 4096u;
    uint64_t max_frame_pixels = 25165824ull;
};

bool decode_video_bytes(
        const uint8_t *encoded,
        size_t encoded_size,
        const std::string &mime,
        const video_decode_limits &limits,
        std::vector<rgb_image> *out,
        std::string *error);

bool decode_video_data_url(
        const std::string &data_url,
        const video_decode_limits &limits,
        std::vector<rgb_image> *out,
        std::string *error);

} // namespace axiom::qwen38::vision

#endif
