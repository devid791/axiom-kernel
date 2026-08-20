/* Native in-memory libavformat/libavcodec video decoder. */

#include "axiom/qwen38_video_decode.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

namespace axiom::qwen38::vision {
namespace {

void set_error(std::string *error, const char *message) {
    if (error) *error = message ? message : "native video decode failed";
}

void set_error(std::string *error, const std::string &message) {
    if (error) *error = message;
}

std::string av_error(int code, const char *prefix) {
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, text, sizeof(text));
    return std::string(prefix ? prefix : "libav error") + ": " + text;
}

struct memory_reader {
    const uint8_t *data = nullptr;
    size_t size = 0u;
    size_t offset = 0u;
};

int read_memory(void *opaque, uint8_t *buffer, int buffer_size) {
    auto *reader = static_cast<memory_reader *>(opaque);
    if (!reader || !buffer || buffer_size <= 0) return AVERROR(EINVAL);
    if (reader->offset >= reader->size) return AVERROR_EOF;
    const size_t remaining = reader->size - reader->offset;
    const size_t count = std::min<size_t>(remaining, static_cast<size_t>(buffer_size));
    std::memcpy(buffer, reader->data + reader->offset, count);
    reader->offset += count;
    return static_cast<int>(count);
}

int64_t seek_memory(void *opaque, int64_t offset, int whence) {
    auto *reader = static_cast<memory_reader *>(opaque);
    if (!reader) return AVERROR(EINVAL);
    if (whence == AVSEEK_SIZE) return static_cast<int64_t>(reader->size);
    int64_t base = 0;
    if (whence == SEEK_CUR) base = static_cast<int64_t>(reader->offset);
    else if (whence == SEEK_END) base = static_cast<int64_t>(reader->size);
    else if (whence != SEEK_SET) return AVERROR(EINVAL);
    if (offset < 0 || base > static_cast<int64_t>(reader->size) - offset) {
        return AVERROR(EINVAL);
    }
    const int64_t position = base + offset;
    if (position < 0 || static_cast<uint64_t>(position) > reader->size) return AVERROR(EINVAL);
    reader->offset = static_cast<size_t>(position);
    return position;
}

void free_io(AVIOContext **io) {
    if (!io || !*io) return;
    av_freep(&(*io)->buffer);
    avio_context_free(io);
}

bool decode_base64(const std::string &encoded, std::vector<uint8_t> *out, std::string *error) {
    if (!out) return false;
    out->clear();
    auto digit = [](unsigned char c) -> int {
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
        for (char raw : encoded) {
            const unsigned char c = static_cast<unsigned char>(raw);
            if (c == '=') {
                padding = true;
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
            if (padding) {
                set_error(error, "invalid video data URL base64 padding");
                return false;
            }
            const int value = digit(c);
            if (value < 0) {
                set_error(error, "video data URL is not valid base64");
                return false;
            }
            accumulator = (accumulator << 6u) | static_cast<uint32_t>(value);
            bits += 6u;
            if (bits >= 8u) {
                bits -= 8u;
                out->push_back(static_cast<uint8_t>((accumulator >> bits) & 0xffu));
            }
        }
    } catch (const std::bad_alloc &) {
        out->clear();
        set_error(error, "video data URL allocation exceeded the native budget");
        return false;
    }
    if (bits >= 6u || (bits != 0u && (accumulator & ((1u << bits) - 1u)) != 0u)) {
        out->clear();
        set_error(error, "video data URL has truncated base64");
        return false;
    }
    return !out->empty();
}

} // namespace

bool decode_video_bytes(const uint8_t *encoded, size_t encoded_size, const std::string &mime,
                        const video_decode_limits &limits, std::vector<rgb_image> *out,
                        std::string *error) {
    (void)mime; /* libavformat probes the in-memory container safely. */
    if (out) out->clear();
    if (!encoded || encoded_size == 0u || !out || limits.max_frames == 0u ||
        limits.max_width == 0u || limits.max_height == 0u || limits.max_frame_pixels == 0u) {
        set_error(error, "invalid native video decode limits or empty input");
        return false;
    }
    if (encoded_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        set_error(error, "video container is too large for the native memory reader");
        return false;
    }

    memory_reader reader{encoded, encoded_size, 0u};
    constexpr int kIoBufferSize = 64 * 1024;
    unsigned char *io_buffer = static_cast<unsigned char *>(av_malloc(kIoBufferSize));
    if (!io_buffer) {
        set_error(error, "libav IO buffer allocation failed");
        return false;
    }
    AVIOContext *io = avio_alloc_context(
            io_buffer, kIoBufferSize, 0, &reader, read_memory, nullptr, seek_memory);
    if (!io) {
        av_free(io_buffer);
        set_error(error, "libav IO context allocation failed");
        return false;
    }
    AVFormatContext *format = avformat_alloc_context();
    if (!format) {
        free_io(&io);
        set_error(error, "libav format context allocation failed");
        return false;
    }
    format->pb = io;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    int rc = avformat_open_input(&format, nullptr, nullptr, nullptr);
    if (rc < 0) {
        set_error(error, av_error(rc, "video container open failed"));
        if (format) avformat_close_input(&format);
        else free_io(&io);
        return false;
    }
    rc = avformat_find_stream_info(format, nullptr);
    if (rc < 0) {
        set_error(error, av_error(rc, "video stream discovery failed"));
        avformat_close_input(&format);
        return false;
    }
    int video_stream_index = -1;
    for (unsigned int index = 0u; index < format->nb_streams; ++index) {
        if (format->streams[index] && format->streams[index]->codecpar &&
            format->streams[index]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = static_cast<int>(index);
            break;
        }
    }
    if (video_stream_index < 0) {
        set_error(error, "video container has no video stream");
        avformat_close_input(&format);
        return false;
    }
    AVStream *stream = format->streams[video_stream_index];
    const AVCodecParameters *parameters = stream->codecpar;
    const AVCodec *decoder = avcodec_find_decoder(parameters->codec_id);
    if (!decoder) {
        set_error(error, "no native libav decoder is installed for the video codec");
        avformat_close_input(&format);
        return false;
    }
    if (parameters->width <= 0 || parameters->height <= 0 ||
        static_cast<uint32_t>(parameters->width) > limits.max_width ||
        static_cast<uint32_t>(parameters->height) > limits.max_height ||
        static_cast<uint64_t>(parameters->width) * static_cast<uint64_t>(parameters->height) >
                limits.max_frame_pixels) {
        set_error(error, "video dimensions exceed the native decode budget");
        avformat_close_input(&format);
        return false;
    }
    AVCodecContext *codec = avcodec_alloc_context3(decoder);
    if (!codec) {
        set_error(error, "video codec context allocation failed");
        avformat_close_input(&format);
        return false;
    }
    rc = avcodec_parameters_to_context(codec, parameters);
    if (rc >= 0) rc = avcodec_open2(codec, decoder, nullptr);
    if (rc < 0) {
        set_error(error, av_error(rc, "video decoder open failed"));
        avcodec_free_context(&codec);
        avformat_close_input(&format);
        return false;
    }
    AVPacket *packet = av_packet_alloc();
    AVFrame *decoded = av_frame_alloc();
    if (!packet || !decoded) {
        av_packet_free(&packet);
        av_frame_free(&decoded);
        avcodec_free_context(&codec);
        avformat_close_input(&format);
        set_error(error, "video frame allocation failed");
        return false;
    }
    SwsContext *scaler = nullptr;
    bool failed = false;
    auto receive_frames = [&]() -> bool {
        while (true) {
            const int receive = avcodec_receive_frame(codec, decoded);
            if (receive == AVERROR(EAGAIN) || receive == AVERROR_EOF) return true;
            if (receive < 0) {
                set_error(error, av_error(receive, "video frame decode failed"));
                return false;
            }
            if (decoded->width <= 0 || decoded->height <= 0 ||
                static_cast<uint32_t>(decoded->width) > limits.max_width ||
                static_cast<uint32_t>(decoded->height) > limits.max_height ||
                static_cast<uint64_t>(decoded->width) * static_cast<uint64_t>(decoded->height) >
                        limits.max_frame_pixels) {
                set_error(error, "decoded video frame exceeds the native decode budget");
                return false;
            }
            if (out->size() >= limits.max_frames) {
                set_error(error, "video frame count exceeds the native decode budget");
                return false;
            }
            scaler = sws_getCachedContext(
                    scaler, decoded->width, decoded->height,
                    static_cast<AVPixelFormat>(decoded->format), decoded->width, decoded->height,
                    AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!scaler) {
                set_error(error, "native video RGB conversion context failed");
                return false;
            }
            rgb_image image;
            try {
                image.width = static_cast<uint32_t>(decoded->width);
                image.height = static_cast<uint32_t>(decoded->height);
                image.rgb.resize(static_cast<size_t>(image.width) * image.height * 3u);
            } catch (const std::bad_alloc &) {
                set_error(error, "decoded video frame allocation exceeded the native budget");
                return false;
            }
            uint8_t *destination[4] = {image.rgb.data(), nullptr, nullptr, nullptr};
            int destination_stride[4] = {static_cast<int>(image.width * 3u), 0, 0, 0};
            const int scaled = sws_scale(
                    scaler, decoded->data, decoded->linesize, 0, decoded->height,
                    destination, destination_stride);
            if (scaled != decoded->height || !image.valid()) {
                set_error(error, "native video RGB conversion failed");
                return false;
            }
            try {
                out->push_back(std::move(image));
            } catch (const std::bad_alloc &) {
                set_error(error, "video frame sequence allocation exceeded the native budget");
                return false;
            }
        }
    };
    while (!failed && (rc = av_read_frame(format, packet)) >= 0) {
        if (packet->stream_index == video_stream_index) {
            rc = avcodec_send_packet(codec, packet);
            if (rc < 0 && rc != AVERROR(EAGAIN)) {
                set_error(error, av_error(rc, "video packet decode failed"));
                failed = true;
            } else if (!receive_frames()) {
                failed = true;
            }
        }
        av_packet_unref(packet);
    }
    if (!failed) {
        rc = avcodec_send_packet(codec, nullptr);
        if (rc < 0 && rc != AVERROR_EOF) {
            set_error(error, av_error(rc, "video decoder flush failed"));
            failed = true;
        } else if (!receive_frames()) {
            failed = true;
        }
    }
    if (!failed && out->empty()) set_error(error, "video decoder produced no frames");
    sws_freeContext(scaler);
    av_packet_free(&packet);
    av_frame_free(&decoded);
    avcodec_free_context(&codec);
    avformat_close_input(&format);
    return !failed && !out->empty();
}

bool decode_video_data_url(const std::string &data_url, const video_decode_limits &limits,
                           std::vector<rgb_image> *out, std::string *error) {
    if (out) out->clear();
    if (!out || data_url.size() < 5u || data_url.compare(0u, 5u, "data:") != 0) {
        set_error(error, "only base64 data URLs are supported for native video input");
        return false;
    }
    const size_t comma = data_url.find(',');
    if (comma == std::string::npos || comma <= 5u) {
        set_error(error, "malformed video data URL");
        return false;
    }
    const std::string metadata = data_url.substr(5u, comma - 5u);
    const size_t semicolon = metadata.find(';');
    const std::string mime = semicolon == std::string::npos
            ? metadata : metadata.substr(0u, semicolon);
    if (metadata.find(";base64") == std::string::npos) {
        set_error(error, "native video input requires a base64 data URL");
        return false;
    }
    std::vector<uint8_t> encoded;
    if (!decode_base64(data_url.substr(comma + 1u), &encoded, error)) return false;
    return decode_video_bytes(encoded.data(), encoded.size(), mime, limits, out, error);
}

} // namespace axiom::qwen38::vision
