/* Native Qwen3.8 image/video preprocessing. */

#include "axiom/qwen38_vision_preprocess.hpp"
#include "axiom/vision_memory_budget.hpp"

#include <csetjmp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

#include <jpeglib.h>
#include <png.h>
#include <webp/decode.h>

namespace axiom::qwen38::vision {
namespace {

constexpr uint32_t kFactor =
        AXIOM_QWEN38_VISION_PATCH_SIZE * AXIOM_QWEN38_VISION_MERGE_SIZE;
constexpr uint32_t kPatch = AXIOM_QWEN38_VISION_PATCH_SIZE;
constexpr uint32_t kTemporal = AXIOM_QWEN38_VISION_TEMPORAL_PATCH;
constexpr float kPi = 3.14159265358979323846f;

void set_error(std::string *error, const char *message) {
    if (error) *error = message ? message : "vision preprocessing error";
}

bool checked_image_bytes(uint32_t width, uint32_t height, size_t *out) {
    if (!out || width == 0u || height == 0u) return false;
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels > std::numeric_limits<size_t>::max() / 3u) return false;
    *out = static_cast<size_t>(pixels * 3u);
    return true;
}

bool has_prefix(const uint8_t *data, size_t size, const char *prefix, size_t length) {
    return data && prefix && size >= length && std::memcmp(data, prefix, length) == 0;
}

std::string lowercase(std::string value) {
    for (char &ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
    }
    return value;
}

enum class encoded_kind { png, jpeg, webp, unknown };

encoded_kind detect_kind(const uint8_t *data, size_t size, const std::string &mime) {
    const std::string lower = lowercase(mime);
    if (lower.find("png") != std::string::npos) return encoded_kind::png;
    if (lower.find("jpeg") != std::string::npos || lower.find("jpg") != std::string::npos) {
        return encoded_kind::jpeg;
    }
    if (lower.find("webp") != std::string::npos) return encoded_kind::webp;
    static constexpr uint8_t kPng[] = {0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au};
    if (has_prefix(data, size, reinterpret_cast<const char *>(kPng), sizeof(kPng))) {
        return encoded_kind::png;
    }
    if (size >= 2u && data[0] == 0xffu && data[1] == 0xd8u) return encoded_kind::jpeg;
    if (size >= 12u && has_prefix(data, size, "RIFF", 4u) &&
        has_prefix(data + 8u, size - 8u, "WEBP", 4u)) return encoded_kind::webp;
    return encoded_kind::unknown;
}

bool decode_png(const uint8_t *encoded, size_t encoded_size, rgb_image *out, std::string *error) {
    if (!encoded || !out || encoded_size == 0u) return false;
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, encoded, encoded_size)) {
        set_error(error, image.message[0] ? image.message : "PNG header decode failed");
        return false;
    }
    image.format = PNG_FORMAT_RGB;
    size_t bytes = 0u;
    if (!checked_image_bytes(image.width, image.height, &bytes)) {
        png_image_free(&image);
        set_error(error, "PNG dimensions exceed the native image budget");
        return false;
    }
    try {
        out->width = image.width;
        out->height = image.height;
        out->rgb.resize(bytes);
    } catch (const std::bad_alloc &) {
        png_image_free(&image);
        out->rgb.clear();
        set_error(error, "PNG allocation exceeded the native image budget");
        return false;
    }
    if (!png_image_finish_read(&image, nullptr, out->rgb.data(), 0, nullptr)) {
        set_error(error, image.message[0] ? image.message : "PNG pixel decode failed");
        png_image_free(&image);
        out->rgb.clear();
        out->width = out->height = 0u;
        return false;
    }
    png_image_free(&image);
    return out->valid();
}

struct jpeg_error_context {
    jpeg_error_mgr manager{};
    std::jmp_buf jump{};
};

extern "C" void jpeg_fail(j_common_ptr common) {
    auto *context = reinterpret_cast<jpeg_error_context *>(common->err);
    longjmp(context->jump, 1);
}

bool decode_jpeg(const uint8_t *encoded, size_t encoded_size, rgb_image *out, std::string *error) {
    if (!encoded || !out || encoded_size == 0u || encoded_size > std::numeric_limits<unsigned long>::max()) {
        set_error(error, "invalid JPEG input");
        return false;
    }
    jpeg_decompress_struct decoder{};
    jpeg_error_context errors{};
    decoder.err = jpeg_std_error(&errors.manager);
    errors.manager.error_exit = jpeg_fail;
    if (setjmp(errors.jump) != 0) {
        jpeg_destroy_decompress(&decoder);
        out->rgb.clear();
        out->width = out->height = 0u;
        set_error(error, "JPEG decode failed");
        return false;
    }
    jpeg_create_decompress(&decoder);
    jpeg_mem_src(&decoder, encoded, static_cast<unsigned long>(encoded_size));
    jpeg_read_header(&decoder, TRUE);
    decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&decoder);
    if (decoder.output_width > std::numeric_limits<uint32_t>::max() ||
        decoder.output_height > std::numeric_limits<uint32_t>::max() ||
        decoder.output_components != 3) {
        jpeg_finish_decompress(&decoder);
        jpeg_destroy_decompress(&decoder);
        set_error(error, "unsupported JPEG dimensions or color layout");
        return false;
    }
    size_t bytes = 0u;
    if (!checked_image_bytes(static_cast<uint32_t>(decoder.output_width),
                             static_cast<uint32_t>(decoder.output_height), &bytes)) {
        jpeg_finish_decompress(&decoder);
        jpeg_destroy_decompress(&decoder);
        set_error(error, "JPEG dimensions exceed the native image budget");
        return false;
    }
    try {
        out->width = static_cast<uint32_t>(decoder.output_width);
        out->height = static_cast<uint32_t>(decoder.output_height);
        out->rgb.resize(bytes);
    } catch (const std::bad_alloc &) {
        jpeg_finish_decompress(&decoder);
        jpeg_destroy_decompress(&decoder);
        out->rgb.clear();
        out->width = out->height = 0u;
        set_error(error, "JPEG allocation exceeded the native image budget");
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(decoder.output_width) * 3u;
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = out->rgb.data() +
                static_cast<size_t>(decoder.output_scanline) * row_bytes;
        jpeg_read_scanlines(&decoder, &row, 1u);
    }
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    return out->valid();
}

bool decode_webp(const uint8_t *encoded, size_t encoded_size, rgb_image *out, std::string *error) {
    if (!encoded || !out || encoded_size == 0u) return false;
    int width = 0;
    int height = 0;
    if (!WebPGetInfo(encoded, encoded_size, &width, &height) || width <= 0 || height <= 0 ||
        static_cast<uint64_t>(width) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(height) > std::numeric_limits<uint32_t>::max()) {
        set_error(error, "WebP header decode failed");
        return false;
    }
    size_t bytes = 0u;
    if (!checked_image_bytes(static_cast<uint32_t>(width), static_cast<uint32_t>(height), &bytes)) {
        set_error(error, "WebP dimensions exceed the native image budget");
        return false;
    }
    uint8_t *decoded = WebPDecodeRGB(encoded, encoded_size, &width, &height);
    if (!decoded) {
        set_error(error, "WebP pixel decode failed");
        return false;
    }
    try {
        out->width = static_cast<uint32_t>(width);
        out->height = static_cast<uint32_t>(height);
        out->rgb.assign(decoded, decoded + bytes);
    } catch (const std::bad_alloc &) {
        WebPFree(decoded);
        out->rgb.clear();
        out->width = out->height = 0u;
        set_error(error, "WebP allocation exceeded the native image budget");
        return false;
    }
    WebPFree(decoded);
    return out->valid();
}

bool decode_base64(const std::string &encoded, std::vector<uint8_t> *out, std::string *error) {
    if (!out) return false;
    out->clear();
    auto value = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    uint32_t accumulator = 0u;
    uint32_t bits = 0u;
    bool padding = false;
    try {
        out->reserve((encoded.size() / 4u) * 3u);
        for (size_t index = 0u; index < encoded.size(); ++index) {
            const unsigned char ch = static_cast<unsigned char>(encoded[index]);
            if (ch == '=' ) {
                padding = true;
                continue;
            }
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
            if (padding) {
                set_error(error, "invalid base64 padding");
                return false;
            }
            const int digit = value(ch);
            if (digit < 0) {
                set_error(error, "data URL is not valid base64");
                return false;
            }
            accumulator = (accumulator << 6u) | static_cast<uint32_t>(digit);
            bits += 6u;
            if (bits >= 8u) {
                bits -= 8u;
                out->push_back(static_cast<uint8_t>((accumulator >> bits) & 0xffu));
            }
        }
    } catch (const std::bad_alloc &) {
        out->clear();
        set_error(error, "data URL allocation exceeded the native image budget");
        return false;
    }
    if (bits >= 6u || (bits != 0u && (accumulator & ((1u << bits) - 1u)) != 0u)) {
        set_error(error, "data URL has truncated base64");
        out->clear();
        return false;
    }
    return !out->empty();
}

float cubic_weight(float x) {
    /* Keys cubic convolution with a=-0.5, the bicubic kernel used by the
     * reference image processors for ordinary Qwen resizing. */
    constexpr float a = -0.5f;
    const float ax = std::fabs(x);
    if (ax <= 1.0f) return (a + 2.0f) * ax * ax * ax - (a + 3.0f) * ax * ax + 1.0f;
    if (ax < 2.0f) return a * ax * ax * ax - 5.0f * a * ax * ax + 8.0f * a * ax - 4.0f * a;
    return 0.0f;
}

uint32_t clamp_index(int64_t index, uint32_t limit) {
    if (index < 0) return 0u;
    if (static_cast<uint64_t>(index) >= limit) return limit - 1u;
    return static_cast<uint32_t>(index);
}

uint8_t sample_bicubic(const rgb_image &image, float source_x, float source_y, uint32_t channel) {
    const int64_t x_base = static_cast<int64_t>(std::floor(source_x));
    const int64_t y_base = static_cast<int64_t>(std::floor(source_y));
    const float x_fraction = source_x - static_cast<float>(x_base);
    const float y_fraction = source_y - static_cast<float>(y_base);
    float value = 0.0f;
    float normalizer = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        const float wy = cubic_weight(static_cast<float>(dy) - y_fraction);
        const uint32_t y = clamp_index(y_base + dy, image.height);
        for (int dx = -1; dx <= 2; ++dx) {
            const float wx = cubic_weight(static_cast<float>(dx) - x_fraction);
            const uint32_t x = clamp_index(x_base + dx, image.width);
            const float weight = wx * wy;
            value += weight * static_cast<float>(image.rgb[
                    (static_cast<size_t>(y) * image.width + x) * 3u + channel]);
            normalizer += weight;
        }
    }
    if (normalizer != 0.0f) value /= normalizer;
    value = std::max(0.0f, std::min(255.0f, value));
    return static_cast<uint8_t>(std::lround(value));
}

bool resize_image(const rgb_image &source, uint32_t width, uint32_t height, rgb_image *out,
                  std::string *error) {
    if (!source.valid() || !out || width == 0u || height == 0u) {
        set_error(error, "invalid RGB image for resize");
        return false;
    }
    size_t bytes = 0u;
    if (!checked_image_bytes(width, height, &bytes)) {
        set_error(error, "resized image exceeds the native image budget");
        return false;
    }
    if (!axiom::vision_memory::host_allocation_fits(bytes)) {
        set_error(error, "insufficient available host memory for resized media");
        return false;
    }
    try {
        out->width = width;
        out->height = height;
        out->rgb.resize(bytes);
    } catch (const std::bad_alloc &) {
        out->rgb.clear();
        out->width = out->height = 0u;
        set_error(error, "resized image allocation exceeded the native image budget");
        return false;
    }
    const float scale_x = static_cast<float>(source.width) / static_cast<float>(width);
    const float scale_y = static_cast<float>(source.height) / static_cast<float>(height);
    for (uint32_t y = 0u; y < height; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        for (uint32_t x = 0u; x < width; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const size_t base = (static_cast<size_t>(y) * width + x) * 3u;
            for (uint32_t channel = 0u; channel < 3u; ++channel) {
                out->rgb[base + channel] = sample_bicubic(source, source_x, source_y, channel);
            }
        }
    }
    return true;
}

bool smart_resize(uint32_t source_height, uint32_t source_width,
                  uint32_t min_pixels, uint32_t max_pixels,
                  uint32_t *out_height, uint32_t *out_width, std::string *error) {
    if (!out_height || !out_width || source_height == 0u || source_width == 0u ||
        min_pixels == 0u || max_pixels < min_pixels) {
        set_error(error, "invalid Qwen resize limits");
        return false;
    }
    const double height = static_cast<double>(source_height);
    const double width = static_cast<double>(source_width);
    const double area = height * width;
    const double min_area = static_cast<double>(min_pixels);
    const double max_area = static_cast<double>(max_pixels);
    double resized_height = std::round(height / kFactor) * kFactor;
    double resized_width = std::round(width / kFactor) * kFactor;
    resized_height = std::max<double>(kFactor, resized_height);
    resized_width = std::max<double>(kFactor, resized_width);
    if (area > max_area) {
        const double beta = std::sqrt(area / max_area);
        resized_height = std::floor(height / beta / kFactor) * kFactor;
        resized_width = std::floor(width / beta / kFactor) * kFactor;
    } else if (area < min_area) {
        const double beta = std::sqrt(min_area / area);
        resized_height = std::ceil(height * beta / kFactor) * kFactor;
        resized_width = std::ceil(width * beta / kFactor) * kFactor;
    }
    resized_height = std::max<double>(kFactor, resized_height);
    resized_width = std::max<double>(kFactor, resized_width);
    while (resized_height * resized_width > max_area && resized_height > kFactor && resized_width > kFactor) {
        if (resized_height / height >= resized_width / width) resized_height -= kFactor;
        else resized_width -= kFactor;
    }
    while (resized_height * resized_width < min_area) {
        if (resized_height / height <= resized_width / width) resized_height += kFactor;
        else resized_width += kFactor;
        if (resized_height * resized_width > max_area) break;
    }
    if (resized_height > std::numeric_limits<uint32_t>::max() ||
        resized_width > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(resized_height) * static_cast<uint64_t>(resized_width) > max_pixels) {
        set_error(error, "Qwen resize exceeds the configured pixel budget");
        return false;
    }
    *out_height = static_cast<uint32_t>(resized_height);
    *out_width = static_cast<uint32_t>(resized_width);
    return true;
}

bool patchify(const std::vector<rgb_image> &frames, uint32_t width, uint32_t height,
              preprocessed_media *out, std::string *error) {
    if (!out || frames.empty() || width == 0u || height == 0u ||
        width % kFactor != 0u || height % kFactor != 0u) {
        set_error(error, "Qwen image grid must be non-empty and divisible by 32");
        return false;
    }
    const uint32_t frame_count = static_cast<uint32_t>(frames.size());
    const uint32_t temporal = (frame_count + kTemporal - 1u) / kTemporal;
    const uint32_t grid_h = height / kPatch;
    const uint32_t grid_w = width / kPatch;
    if (grid_h % AXIOM_QWEN38_VISION_MERGE_SIZE != 0u ||
        grid_w % AXIOM_QWEN38_VISION_MERGE_SIZE != 0u) {
        set_error(error, "Qwen image grid must be divisible by the spatial merge size");
        return false;
    }
    const uint64_t patch_count = static_cast<uint64_t>(temporal) * grid_h * grid_w;
    const uint64_t values = patch_count * AXIOM_QWEN38_VISION_PATCH_FEATURES;
    if (values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        set_error(error, "patch tensor exceeds the native allocation limit");
        return false;
    }
    if (!axiom::vision_memory::host_allocation_fits(values * sizeof(float))) {
        set_error(error, "insufficient available host memory for the native visual patch tensor");
        return false;
    }
    try {
        out->grid.temporal = temporal;
        out->grid.height = grid_h;
        out->grid.width = grid_w;
        out->resized_width = width;
        out->resized_height = height;
        out->source_frames = frame_count;
        out->patches.assign(static_cast<size_t>(values), 0.0f);
    } catch (const std::bad_alloc &) {
        out->patches.clear();
        set_error(error, "patch tensor allocation exceeded the native budget");
        return false;
    }
    /* Official Qwen patchify order: [grid_t, grid_h, grid_w, C, temporal, y, x]. */
    size_t cursor = 0u;
    for (uint32_t time_block = 0u; time_block < temporal; ++time_block) {
        for (uint32_t patch_y = 0u; patch_y < grid_h; ++patch_y) {
            for (uint32_t patch_x = 0u; patch_x < grid_w; ++patch_x) {
                for (uint32_t channel = 0u; channel < 3u; ++channel) {
                    for (uint32_t time = 0u; time < kTemporal; ++time) {
                        const uint32_t frame_index = std::min<uint32_t>(
                                time_block * kTemporal + time, frame_count - 1u);
                        const rgb_image &frame = frames[frame_index];
                        for (uint32_t y = 0u; y < kPatch; ++y) {
                            for (uint32_t x = 0u; x < kPatch; ++x) {
                                const uint8_t pixel = frame.rgb[
                                        (static_cast<size_t>(patch_y * kPatch + y) * width +
                                         patch_x * kPatch + x) * 3u + channel];
                                out->patches[cursor++] =
                                        (static_cast<float>(pixel) / 255.0f - 0.5f) / 0.5f;
                            }
                        }
                    }
                }
            }
        }
    }
    return cursor == out->patches.size();
}

} // namespace

bool decode_image_bytes(const uint8_t *encoded, size_t encoded_size, const std::string &mime,
                        rgb_image *out, std::string *error) {
    if (out) *out = rgb_image{};
    if (!encoded || encoded_size == 0u || !out) {
        set_error(error, "empty encoded image");
        return false;
    }
    switch (detect_kind(encoded, encoded_size, mime)) {
        case encoded_kind::png: return decode_png(encoded, encoded_size, out, error);
        case encoded_kind::jpeg: return decode_jpeg(encoded, encoded_size, out, error);
        case encoded_kind::webp: return decode_webp(encoded, encoded_size, out, error);
        default:
            set_error(error, "unsupported image format; native build supports PNG, JPEG and WebP");
            return false;
    }
}

bool decode_data_url(const std::string &data_url, rgb_image *out, std::string *error) {
    if (out) *out = rgb_image{};
    if (!out || data_url.size() < 5u || data_url.compare(0u, 5u, "data:") != 0) {
        set_error(error, "only base64 data URLs are supported for native media input");
        return false;
    }
    const size_t comma = data_url.find(',');
    if (comma == std::string::npos || comma <= 5u) {
        set_error(error, "malformed data URL");
        return false;
    }
    const std::string metadata = data_url.substr(5u, comma - 5u);
    const size_t semicolon = metadata.find(';');
    const std::string mime = semicolon == std::string::npos
            ? metadata : metadata.substr(0u, semicolon);
    if (metadata.find(";base64") == std::string::npos) {
        set_error(error, "native media input requires a base64 data URL");
        return false;
    }
    std::vector<uint8_t> encoded;
    if (!decode_base64(data_url.substr(comma + 1u), &encoded, error)) return false;
    return decode_image_bytes(encoded.data(), encoded.size(), mime, out, error);
}

bool preprocess_image(const rgb_image &image, uint32_t min_pixels, uint32_t max_pixels,
                     preprocessed_media *out, std::string *error) {
    if (out) *out = preprocessed_media{};
    if (!out || !image.valid()) {
        set_error(error, "invalid RGB image");
        return false;
    }
    uint32_t height = 0u;
    uint32_t width = 0u;
    if (!smart_resize(image.height, image.width, min_pixels, max_pixels, &height, &width, error)) {
        return false;
    }
    rgb_image resized;
    if (!resize_image(image, width, height, &resized, error)) return false;
    std::vector<rgb_image> frames;
    try {
        frames.push_back(std::move(resized));
        if (!axiom::vision_memory::host_allocation_fits(frames.front().rgb.size())) {
            set_error(error, "insufficient available host memory for temporal image copy");
            return false;
        }
        frames.push_back(frames.front());
    } catch (const std::bad_alloc &) {
        set_error(error, "image frame allocation exceeded the native budget");
        return false;
    }
    /* A still image is represented as one temporal group of two identical
     * frames, as required by temporal_patch_size=2. */
    const bool ok = patchify(frames, width, height, out, error);
    if (ok) out->source_frames = 1u;
    return ok;
}

bool preprocess_video_frames(const std::vector<rgb_image> &frames, uint32_t min_pixels,
                            uint32_t max_pixels, preprocessed_media *out, std::string *error) {
    if (out) *out = preprocessed_media{};
    if (!out || frames.empty()) {
        set_error(error, "video requires at least one decoded RGB frame");
        return false;
    }
    const rgb_image &first = frames.front();
    if (!first.valid()) {
        set_error(error, "video contains an invalid first RGB frame");
        return false;
    }
    for (const rgb_image &frame : frames) {
        if (!frame.valid()) {
            set_error(error, "video contains an invalid RGB frame");
            return false;
        }
    }
    uint32_t height = 0u;
    uint32_t width = 0u;
    if (!smart_resize(first.height, first.width, min_pixels, max_pixels, &height, &width, error)) {
        return false;
    }
    std::vector<rgb_image> resized;
    try {
        resized.reserve(frames.size() + 1u);
        for (const rgb_image &frame : frames) {
            rgb_image current;
            if (!resize_image(frame, width, height, &current, error)) return false;
            resized.push_back(std::move(current));
        }
        if ((resized.size() % kTemporal) != 0u) {
            if (!axiom::vision_memory::host_allocation_fits(resized.back().rgb.size())) {
                set_error(error, "insufficient available host memory for temporal frame copy");
                return false;
            }
            resized.push_back(resized.back());
        }
    } catch (const std::bad_alloc &) {
        set_error(error, "video frame allocation exceeded the native budget");
        return false;
    }
    const bool ok = patchify(resized, width, height, out, error);
    if (ok) out->source_frames = static_cast<uint32_t>(frames.size());
    return ok;
}

} // namespace axiom::qwen38::vision
