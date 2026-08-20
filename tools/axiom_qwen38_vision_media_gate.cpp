/* Staging-only native image/container/video preprocessing gate. */

#include <algorithm>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <jpeglib.h>
#include <png.h>
#include <webp/encode.h>

#include "axiom/qwen38_video_decode.hpp"
#include "axiom/qwen38_vision_preprocess.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

namespace {

bool encode_png_fixture(
        const std::vector<uint8_t> &rgb, std::vector<uint8_t> *out,
        std::string *error) {
    if (!out || rgb.size() != 2u * 2u * 3u) return false;
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = 2u;
    image.height = 2u;
    image.format = PNG_FORMAT_RGB;
    size_t bytes = 0u;
    if (!png_image_write_to_memory(
            &image, nullptr, &bytes, 0, rgb.data(), 2u * 3u, nullptr)) {
        if (error) *error = image.message;
        return false;
    }
    try {
        out->resize(bytes);
    } catch (...) {
        if (error) *error = "PNG fixture allocation failed";
        return false;
    }
    if (!png_image_write_to_memory(
            &image, out->data(), &bytes, 0, rgb.data(), 2u * 3u, nullptr)) {
        if (error) *error = image.message;
        return false;
    }
    out->resize(bytes);
    return true;
}

struct jpeg_gate_error {
    jpeg_error_mgr base{};
    jmp_buf jump{};
};

void jpeg_gate_error_exit(j_common_ptr info) {
    auto *error = reinterpret_cast<jpeg_gate_error *>(info->err);
    longjmp(error->jump, 1);
}

bool encode_jpeg_fixture(
        const std::vector<uint8_t> &rgb, std::vector<uint8_t> *out,
        std::string *error) {
    if (!out || rgb.size() != 2u * 2u * 3u) return false;
    jpeg_compress_struct compressor{};
    jpeg_gate_error jpeg_error{};
    compressor.err = jpeg_std_error(&jpeg_error.base);
    jpeg_error.base.error_exit = jpeg_gate_error_exit;
    if (setjmp(jpeg_error.jump) != 0) {
        jpeg_destroy_compress(&compressor);
        if (error) *error = "JPEG fixture encoding failed";
        return false;
    }
    jpeg_create_compress(&compressor);
    unsigned char *encoded = nullptr;
    unsigned long encoded_size = 0u;
    jpeg_mem_dest(&compressor, &encoded, &encoded_size);
    compressor.image_width = 2u;
    compressor.image_height = 2u;
    compressor.input_components = 3;
    compressor.in_color_space = JCS_RGB;
    jpeg_set_defaults(&compressor);
    jpeg_set_quality(&compressor, 90, TRUE);
    jpeg_start_compress(&compressor, TRUE);
    while (compressor.next_scanline < compressor.image_height) {
        JSAMPROW row = const_cast<JSAMPROW>(
                rgb.data() + static_cast<size_t>(compressor.next_scanline) * 2u * 3u);
        jpeg_write_scanlines(&compressor, &row, 1u);
    }
    jpeg_finish_compress(&compressor);
    out->assign(encoded, encoded + encoded_size);
    std::free(encoded);
    jpeg_destroy_compress(&compressor);
    return !out->empty();
}

bool encode_webp_fixture(
        const std::vector<uint8_t> &rgb, std::vector<uint8_t> *out,
        std::string *error) {
    if (!out || rgb.size() != 2u * 2u * 3u) return false;
    uint8_t *encoded = nullptr;
    const size_t encoded_size = WebPEncodeRGB(rgb.data(), 2, 2, 2 * 3, 90.0f, &encoded);
    if (encoded_size == 0u || !encoded) {
        if (error) *error = "WebP fixture encoding failed";
        return false;
    }
    out->assign(encoded, encoded + encoded_size);
    WebPFree(encoded);
    return true;
}

struct memory_writer {
    std::vector<uint8_t> bytes;
    int64_t position = 0;
};

int write_memory(void *opaque, uint8_t *buffer, int buffer_size) {
    auto *writer = static_cast<memory_writer *>(opaque);
    if (!writer || !buffer || buffer_size < 0) return AVERROR(EINVAL);
    try {
        if (writer->position < 0) return AVERROR(EINVAL);
        const size_t position = static_cast<size_t>(writer->position);
        if (position > writer->bytes.size()) writer->bytes.resize(position, 0u);
        if (static_cast<size_t>(buffer_size) > std::numeric_limits<size_t>::max() - position) {
            return AVERROR(ENOMEM);
        }
        const size_t end = position + static_cast<size_t>(buffer_size);
        if (end > writer->bytes.size()) writer->bytes.resize(end);
        std::memcpy(writer->bytes.data() + position, buffer, static_cast<size_t>(buffer_size));
        writer->position += buffer_size;
        return buffer_size;
    } catch (...) {
        return AVERROR(ENOMEM);
    }
}

int64_t seek_memory(void *opaque, int64_t offset, int whence) {
    auto *writer = static_cast<memory_writer *>(opaque);
    if (!writer) return AVERROR(EINVAL);
    if (whence == AVSEEK_SIZE) return static_cast<int64_t>(writer->bytes.size());
    int64_t base = 0;
    if (whence == SEEK_CUR) base = writer->position;
    else if (whence == SEEK_END) base = static_cast<int64_t>(writer->bytes.size());
    else if (whence != SEEK_SET) return AVERROR(EINVAL);
    if (offset < 0 || base > std::numeric_limits<int64_t>::max() - offset) return AVERROR(EINVAL);
    const int64_t position = base + offset;
    if (position < 0) return AVERROR(EINVAL);
    writer->position = position;
    return position;
}

bool encode_fixture(std::vector<uint8_t> *out, std::string *error) {
    if (!out) return false;
    out->clear();
    const AVCodec *encoder = avcodec_find_encoder(AV_CODEC_ID_FFV1);
    if (!encoder) {
        if (error) *error = "FFV1 encoder unavailable for the native gate";
        return false;
    }
    AVFormatContext *format = nullptr;
    int rc = avformat_alloc_output_context2(&format, nullptr, "matroska", nullptr);
    if (rc < 0 || !format) {
        if (error) *error = "Matroska output context allocation failed";
        return false;
    }
    memory_writer writer;
    constexpr int kIoBufferSize = 16 * 1024;
    unsigned char *io_buffer = static_cast<unsigned char *>(av_malloc(kIoBufferSize));
    AVIOContext *io = io_buffer
            ? avio_alloc_context(io_buffer, kIoBufferSize, 1, &writer, nullptr,
                                 write_memory, seek_memory) : nullptr;
    if (!io) {
        av_free(io_buffer);
        avformat_free_context(format);
        if (error) *error = "Matroska memory IO allocation failed";
        return false;
    }
    format->pb = io;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    AVStream *stream = avformat_new_stream(format, nullptr);
    AVCodecContext *codec = avcodec_alloc_context3(encoder);
    AVFrame *frame = av_frame_alloc();
    SwsContext *scaler = nullptr;
    AVPacket *packet = av_packet_alloc();
    bool ok = stream && codec && frame && packet;
    if (ok) {
        codec->codec_id = AV_CODEC_ID_FFV1;
        codec->codec_type = AVMEDIA_TYPE_VIDEO;
        codec->width = 32;
        codec->height = 32;
        codec->time_base = AVRational{1, 3};
        codec->framerate = AVRational{3, 1};
        codec->pix_fmt = AV_PIX_FMT_YUV420P;
        codec->gop_size = 12;
        codec->max_b_frames = 0;
        codec->bit_rate = 200000;
        if ((format->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
            codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        rc = avcodec_open2(codec, encoder, nullptr);
        if (rc >= 0) rc = avcodec_parameters_from_context(stream->codecpar, codec);
        stream->time_base = codec->time_base;
        if (rc >= 0) rc = avformat_write_header(format, nullptr);
        frame->format = codec->pix_fmt;
        frame->width = codec->width;
        frame->height = codec->height;
        if (rc >= 0) rc = av_frame_get_buffer(frame, 32);
        scaler = sws_getContext(32, 32, AV_PIX_FMT_RGB24, 32, 32,
                                AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
        ok = rc >= 0 && scaler != nullptr;
    }
    if (ok) {
        for (int index = 0; index < 3 && ok; ++index) {
            if (av_frame_make_writable(frame) < 0) {
                ok = false;
                break;
            }
            std::vector<uint8_t> rgb(32u * 32u * 3u, 0u);
            const uint8_t colors[3][3] = {{255u, 0u, 0u}, {0u, 255u, 0u}, {0u, 0u, 255u}};
            for (size_t pixel = 0u; pixel < 32u * 32u; ++pixel) {
                rgb[pixel * 3u + 0u] = colors[index][0];
                rgb[pixel * 3u + 1u] = colors[index][1];
                rgb[pixel * 3u + 2u] = colors[index][2];
            }
            uint8_t *source[4] = {rgb.data(), nullptr, nullptr, nullptr};
            int source_stride[4] = {32 * 3, 0, 0, 0};
            if (sws_scale(scaler, source, source_stride, 0, 32,
                          frame->data, frame->linesize) != 32) {
                ok = false;
                break;
            }
            frame->pts = index;
            if (avcodec_send_frame(codec, frame) < 0) {
                ok = false;
                break;
            }
            while (true) {
                rc = avcodec_receive_packet(codec, packet);
                if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
                if (rc < 0) {
                    ok = false;
                    break;
                }
                packet->stream_index = stream->index;
                if (av_interleaved_write_frame(format, packet) < 0) ok = false;
                av_packet_unref(packet);
                if (!ok) break;
            }
        }
    }
    if (ok && avcodec_send_frame(codec, nullptr) >= 0) {
        while (true) {
            rc = avcodec_receive_packet(codec, packet);
            if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
            if (rc < 0) {
                ok = false;
                break;
            }
            packet->stream_index = stream->index;
            if (av_interleaved_write_frame(format, packet) < 0) ok = false;
            av_packet_unref(packet);
            if (!ok) break;
        }
    } else if (ok) {
        ok = false;
    }
    if (ok) ok = av_write_trailer(format) >= 0;
    if (ok) *out = std::move(writer.bytes);
    sws_freeContext(scaler);
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&codec);
    avformat_free_context(format);
    if (error && !ok) *error = "native Matroska fixture encoding failed";
    return ok && !out->empty();
}

int fail(const char *message) {
    std::fprintf(stderr, "VISION_MEDIA_GATE status=fail error=%s\n", message);
    return 1;
}

} // namespace

int main() {
    using namespace axiom::qwen38::vision;
    rgb_image synthetic;
    synthetic.width = 32u;
    synthetic.height = 32u;
    synthetic.rgb.assign(32u * 32u * 3u, 128u);
    preprocessed_media image;
    std::string error;
    if (!preprocess_image(synthetic, 1024u, 4096u, &image, &error)) return fail(error.c_str());
    if (image.grid.temporal != 1u || image.grid.height != 2u || image.grid.width != 2u ||
        image.patch_count() != 4u || image.patches.size() != 4u * AXIOM_QWEN38_VISION_PATCH_FEATURES) {
        return fail("still-image grid/patch count mismatch");
    }
    for (float value : image.patches) {
        if (std::abs(value - ((128.0f / 255.0f - 0.5f) / 0.5f)) > 1.0e-6f) {
            return fail("still-image normalization mismatch");
        }
    }

    const std::vector<uint8_t> encoded_rgb = {
            255u, 0u, 0u,   0u, 255u, 0u,
            0u, 0u, 255u,   255u, 255u, 255u,
    };
    std::vector<uint8_t> encoded;
    if (!encode_png_fixture(encoded_rgb, &encoded, &error)) return fail(error.c_str());
    rgb_image decoded;
    if (!decode_image_bytes(encoded.data(), encoded.size(), "image/png", &decoded, &error) ||
        decoded.width != 2u || decoded.height != 2u || !decoded.valid()) {
        return fail("PNG decode contract mismatch");
    }
    if (!encode_jpeg_fixture(encoded_rgb, &encoded, &error) ||
        !decode_image_bytes(encoded.data(), encoded.size(), "image/jpeg", &decoded, &error) ||
        decoded.width != 2u || decoded.height != 2u || !decoded.valid()) {
        return fail(error.empty() ? "JPEG decode contract mismatch" : error.c_str());
    }
    if (!encode_webp_fixture(encoded_rgb, &encoded, &error) ||
        !decode_image_bytes(encoded.data(), encoded.size(), "image/webp", &decoded, &error) ||
        decoded.width != 2u || decoded.height != 2u || !decoded.valid()) {
        return fail(error.empty() ? "WebP decode contract mismatch" : error.c_str());
    }
    if (decode_image_bytes(
            reinterpret_cast<const uint8_t *>("not-an-image"), 12u,
            "image/avif", &decoded, &error)) {
        return fail("unsupported image format was accepted");
    }

    std::vector<uint8_t> container;
    if (!encode_fixture(&container, &error)) return fail(error.c_str());
    video_decode_limits limits;
    limits.max_frames = 8u;
    limits.max_width = 64u;
    limits.max_height = 64u;
    limits.max_frame_pixels = 4096u;
    std::vector<rgb_image> frames;
    if (!decode_video_bytes(container.data(), container.size(), "video/x-matroska",
                            limits, &frames, &error)) return fail(error.c_str());
    if (frames.size() < 3u || frames.size() > limits.max_frames ||
        frames[0].width != 32u || frames[0].height != 32u ||
        frames[1].rgb == frames[0].rgb || frames[2].rgb == frames[1].rgb) {
        std::fprintf(stderr, "decoded_frames=%zu dims=%ux%u distinct=%d/%d\n",
                     frames.size(), frames.empty() ? 0u : frames[0].width,
                     frames.empty() ? 0u : frames[0].height,
                     frames.size() > 1u && frames[1].rgb != frames[0].rgb,
                     frames.size() > 2u && frames[2].rgb != frames[1].rgb);
        return fail("container decoder frame sequence mismatch");
    }
    preprocessed_media video;
    if (!preprocess_video_frames(frames, 1024u, 4096u, &video, &error)) return fail(error.c_str());
    if (video.source_frames != frames.size() || video.grid.temporal != 2u || video.grid.height != 2u ||
        video.grid.width != 2u || video.patch_count() != 8u ||
        video.patches.size() != 8u * AXIOM_QWEN38_VISION_PATCH_FEATURES) {
        std::fprintf(stderr, "video source=%u frame_count=%zu grid=%ux%ux%u patch_count=%llu values=%zu expected=%u\n",
                     video.source_frames, frames.size(), video.grid.temporal,
                     video.grid.height, video.grid.width,
                     static_cast<unsigned long long>(video.patch_count()),
                     video.patches.size(), 8u * AXIOM_QWEN38_VISION_PATCH_FEATURES);
        return fail("video grid/temporal patch mismatch");
    }
    std::printf("VISION_MEDIA_GATE status=pass container_bytes=%zu frames=%zu "
                "image_grid=%ux%ux%u video_grid=%ux%ux%u video_patches=%zu\n",
                container.size(), frames.size(), image.grid.temporal, image.grid.height,
                image.grid.width, video.grid.temporal, video.grid.height, video.grid.width,
                video.patches.size() / AXIOM_QWEN38_VISION_PATCH_FEATURES);
    return 0;
}
