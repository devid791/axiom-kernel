#include "axiom/qwen4exp/multimodal_text.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4 = axiom::qwen4exp;

namespace {

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void require_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        fail(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_status(q4::multimodal_text_status actual,
                    q4::multimodal_text_status expected,
                    const char *operation) {
    if (actual != expected) {
        fail(std::string(operation) + ": expected " +
             q4::multimodal_text_status_string(expected) + ", got " +
             q4::multimodal_text_status_string(actual));
    }
}

template <typename T>
class device_buffer final {
public:
    explicit device_buffer(std::size_t count) : count_(count) {
        require(count != 0u, "zero-sized device buffer");
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data_),
                                count * sizeof(T)),
                     "cudaMalloc");
    }
    ~device_buffer() { (void)cudaFree(data_); }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;

    T *get() noexcept { return data_; }
    const T *get() const noexcept { return data_; }

    void upload(const std::vector<T> &values, cudaStream_t stream) {
        require(values.size() <= count_, "device upload overflow");
        require_cuda(cudaMemcpyAsync(data_, values.data(),
                                     values.size() * sizeof(T),
                                     cudaMemcpyHostToDevice, stream),
                     "cudaMemcpyAsync H2D");
    }

    std::vector<T> download(cudaStream_t stream) const {
        std::vector<T> values(count_);
        require_cuda(cudaMemcpyAsync(values.data(), data_,
                                     count_ * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream),
                     "cudaMemcpyAsync D2H");
        require_cuda(cudaStreamSynchronize(stream), "download synchronize");
        return values;
    }

private:
    T *data_ = nullptr;
    std::size_t count_ = 0u;
};

}  // namespace

int main() {
    try {
        int device = -1;
        cudaDeviceProp properties{};
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        require_cuda(cudaGetDeviceProperties(&properties, device),
                     "cudaGetDeviceProperties");
        require(properties.major == 12,
                "multimodal text gate requires an SM120 GPU");

        cudaStream_t stream = nullptr;
        require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                     "cudaStreamCreateWithFlags");

        const std::vector<std::uint32_t> tokens{
                42u,
                q4::kVisionStartTokenId,
                q4::kImageTokenId,
                q4::kImageTokenId,
                q4::kImageTokenId,
                q4::kImageTokenId,
                q4::kVisionEndTokenId,
                43u,
        };
        const std::size_t count = tokens.size();
        std::vector<std::uint32_t> positions{
                0u, 1u, 1u, 1u, 1u, 1u, 2u, 3u,
                0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u,
                0u, 1u, 1u, 2u, 1u, 2u, 2u, 3u,
        };
        std::vector<float> projected(4u * q4::kMultimodalHidden);
        for (std::size_t index = 0u; index < projected.size(); ++index) {
            projected[index] =
                    static_cast<float>(static_cast<int>(index % 257u) - 128) /
                    64.0F;
        }
        device_buffer<float> device_projected(projected.size());
        device_buffer<float> device_residual(q4::kMultimodalResidual);
        device_buffer<std::uint32_t> device_status(1u);
        device_projected.upload(projected, stream);

        q4::multimodal_text_prompt_view prompt{};
        prompt.token_ids = tokens.data();
        prompt.token_count = count;
        prompt.projected_vision_embeddings_device = device_projected.get();
        prompt.projected_vision_tokens = 4u;
        prompt.projected_vision_width = q4::kMultimodalHidden;
        prompt.mrope_positions = positions.data();
        prompt.mrope_position_tokens = count;

        q4::multimodal_text_prompt_report report{};
        require_status(q4::validate_multimodal_text_prompt(
                               prompt, device, &report),
                       q4::multimodal_text_status::ok,
                       "valid prompt");
        require(report.image_tokens == 4u && report.video_tokens == 0u &&
                        report.vision_tokens == 4u && report.vision_spans == 1u &&
                        report.maximum_position == 3u &&
                        report.next_text_position == 4u &&
                        report.next_text_position_available,
                "valid prompt report mismatch");

        q4::multimodal_text_prompt_view invalid = prompt;
        invalid.projected_vision_width = q4::kMultimodalHidden - 1u;
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::unsupported_contract,
                       "projected width");
        invalid = prompt;
        invalid.projected_vision_tokens = 3u;
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::vision_token_mismatch,
                       "projected row count");
        invalid = prompt;
        invalid.projected_vision_embeddings_device = projected.data();
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::invalid_device_pointer,
                       "host projected pointer");
        invalid = prompt;
        invalid.projected_vision_embeddings_device =
                device_projected.get() + 3u * q4::kMultimodalHidden;
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::invalid_device_pointer,
                       "truncated device allocation");

        std::vector<std::uint32_t> invalid_positions = positions;
        invalid_positions[count] = 9u;
        invalid = prompt;
        invalid.mrope_positions = invalid_positions.data();
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::invalid_mrope,
                       "text axis mismatch");
        invalid_positions = positions;
        invalid_positions[2u] = q4::rope::kMaxContext;
        invalid.mrope_positions = invalid_positions.data();
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::invalid_mrope,
                       "native context bound");

        std::vector<std::uint32_t> orphan_tokens = tokens;
        orphan_tokens[1u] = q4::kImageTokenId;
        invalid = prompt;
        invalid.token_ids = orphan_tokens.data();
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::malformed_vision_span,
                       "orphan visual placeholder");
        std::vector<std::uint32_t> mixed_tokens = tokens;
        mixed_tokens[4u] = q4::kVideoTokenId;
        invalid.token_ids = mixed_tokens.data();
        require_status(q4::validate_multimodal_text_prompt(
                               invalid, device, nullptr),
                       q4::multimodal_text_status::malformed_vision_span,
                       "mixed visual span");

        require_cuda(cudaMemsetAsync(device_status.get(), 0,
                                     sizeof(std::uint32_t), stream),
                     "status reset");
        require_status(q4::repeat_projected_vision_embedding_cuda(
                               device_projected.get(), device_residual.get(),
                               device_status.get(), device, stream),
                       q4::multimodal_text_status::ok,
                       "projected injection");
        const std::vector<float> residual = device_residual.download(stream);
        const std::vector<std::uint32_t> status = device_status.download(stream);
        require(status[0] == 0u, "finite projected row was rejected");
        for (std::size_t stream_index = 0u;
             stream_index < q4::kMultimodalStreams; ++stream_index) {
            for (std::size_t hidden = 0u; hidden < q4::kMultimodalHidden;
                 ++hidden) {
                require(residual[stream_index * q4::kMultimodalHidden + hidden] ==
                                projected[hidden],
                        "projected row was not repeated bit-exactly");
            }
        }

        projected[17u] = std::numeric_limits<float>::quiet_NaN();
        device_projected.upload(projected, stream);
        require_cuda(cudaMemsetAsync(device_status.get(), 0,
                                     sizeof(std::uint32_t), stream),
                     "non-finite status reset");
        require_status(q4::repeat_projected_vision_embedding_cuda(
                               device_projected.get(), device_residual.get(),
                               device_status.get(), device, stream),
                       q4::multimodal_text_status::ok,
                       "non-finite projected injection");
        const std::vector<float> rejected = device_residual.download(stream);
        const std::vector<std::uint32_t> rejected_status =
                device_status.download(stream);
        require(rejected_status[0] != 0u,
                "non-finite projected row did not fail closed");
        for (std::size_t stream_index = 0u;
             stream_index < q4::kMultimodalStreams; ++stream_index) {
            require(rejected[stream_index * q4::kMultimodalHidden + 17u] == 0.0F,
                    "non-finite projected value was not zeroed");
        }

        require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
        std::cout << "qwen4exp-multimodal-text-test: PASS"
                  << " context=262144 mrope=11/11/10"
                  << " spans=" << report.vision_spans
                  << " visual_tokens=" << report.vision_tokens
                  << " hidden=" << q4::kMultimodalHidden
                  << " streams=" << q4::kMultimodalStreams
                  << " next_text_position=" << report.next_text_position
                  << " finite=accepted non_finite=rejected"
                  << '\n';
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "qwen4exp-multimodal-text-test: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
