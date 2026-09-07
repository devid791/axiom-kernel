#include "axiom/qwen4exp/multimodal_text.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace axiom::qwen4exp {
namespace {

constexpr unsigned kThreads = 256u;

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool is_visual_token(std::uint32_t token) noexcept {
    return token == kImageTokenId || token == kVideoTokenId;
}

bool device_range_valid(const void *pointer,
                        std::size_t bytes,
                        int device) noexcept {
    if (pointer == nullptr || bytes == 0u || device < 0) return false;
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) return false;

    cudaPointerAttributes attributes{};
    const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
#if CUDART_VERSION >= 10000
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) {
        return false;
    }
#else
    if (attributes.memoryType != cudaMemoryTypeDevice ||
        attributes.device != device) {
        return false;
    }
#endif

    CUdeviceptr allocation = 0u;
    std::size_t allocation_bytes = 0u;
    if (cuMemGetAddressRange(&allocation, &allocation_bytes,
                             reinterpret_cast<CUdeviceptr>(pointer)) !=
        CUDA_SUCCESS) {
        return false;
    }
    const std::uintptr_t allocation_begin =
            static_cast<std::uintptr_t>(allocation);
    if (allocation_begin > begin) return false;
    const std::size_t offset = static_cast<std::size_t>(begin - allocation_begin);
    return offset <= allocation_bytes && bytes <= allocation_bytes - offset;
}

bool ranges_overlap(const void *left,
                    std::size_t left_bytes,
                    const void *right,
                    std::size_t right_bytes) noexcept {
    const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left);
    const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
        right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
        return true;
    }
    return left_begin < right_begin + right_bytes &&
           right_begin < left_begin + left_bytes;
}

__global__ void repeat_projected_vision_kernel(
        const float *projected,
        float *residual,
        std::uint32_t *device_status) {
    const std::size_t index =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kMultimodalResidual) return;
    const float value = projected[index % kMultimodalHidden];
    if (!isfinite(value)) {
        atomicExch(device_status, 1u);
        residual[index] = 0.0F;
        return;
    }
    residual[index] = value;
}

}  // namespace

const char *multimodal_text_status_string(
        multimodal_text_status status) noexcept {
    switch (status) {
        case multimodal_text_status::ok: return "ok";
        case multimodal_text_status::invalid_argument: return "invalid_argument";
        case multimodal_text_status::unsupported_contract:
            return "unsupported_contract";
        case multimodal_text_status::malformed_vision_span:
            return "malformed_vision_span";
        case multimodal_text_status::vision_token_mismatch:
            return "vision_token_mismatch";
        case multimodal_text_status::invalid_mrope: return "invalid_mrope";
        case multimodal_text_status::size_overflow: return "size_overflow";
        case multimodal_text_status::unsupported_device:
            return "unsupported_device";
        case multimodal_text_status::invalid_device_pointer:
            return "invalid_device_pointer";
        case multimodal_text_status::non_finite_embedding:
            return "non_finite_embedding";
        case multimodal_text_status::cuda_error: return "cuda_error";
    }
    return "unknown";
}

multimodal_text_status validate_multimodal_text_prompt(
        const multimodal_text_prompt_view &prompt,
        int expected_device,
        multimodal_text_prompt_report *report) noexcept {
    if (report != nullptr) *report = {};
    if (prompt.token_ids == nullptr || prompt.mrope_positions == nullptr ||
        prompt.token_count == 0u ||
        prompt.token_count > rope::kMaxContext ||
        prompt.mrope_position_tokens != prompt.token_count ||
        expected_device < 0) {
        return multimodal_text_status::invalid_argument;
    }
    if (prompt.projected_vision_width != kMultimodalHidden) {
        return multimodal_text_status::unsupported_contract;
    }

    multimodal_text_prompt_report result{};
    std::size_t index = 0u;
    while (index < prompt.token_count) {
        const std::uint32_t token = prompt.token_ids[index];
        if (token >= 248320u) return multimodal_text_status::unsupported_contract;
        if (token == kVisionStartTokenId) {
            const std::size_t first = index + 1u;
            if (first >= prompt.token_count ||
                !is_visual_token(prompt.token_ids[first])) {
                return multimodal_text_status::malformed_vision_span;
            }
            const std::uint32_t visual_kind = prompt.token_ids[first];
            index = first;
            while (index < prompt.token_count &&
                   prompt.token_ids[index] == visual_kind) {
                if (visual_kind == kImageTokenId) {
                    ++result.image_tokens;
                } else {
                    ++result.video_tokens;
                }
                ++result.vision_tokens;
                ++index;
            }
            if (index >= prompt.token_count ||
                prompt.token_ids[index] != kVisionEndTokenId) {
                return multimodal_text_status::malformed_vision_span;
            }
            ++result.vision_spans;
            ++index;
            continue;
        }
        if (token == kVisionEndTokenId || is_visual_token(token)) {
            return multimodal_text_status::malformed_vision_span;
        }
        ++index;
    }

    if (result.vision_tokens == 0u ||
        result.vision_tokens != prompt.projected_vision_tokens ||
        prompt.projected_vision_embeddings_device == nullptr) {
        return multimodal_text_status::vision_token_mismatch;
    }

    std::size_t projected_values = 0u;
    std::size_t projected_bytes = 0u;
    if (!checked_multiply(prompt.projected_vision_tokens,
                          kMultimodalHidden, &projected_values) ||
        !checked_multiply(projected_values, sizeof(float), &projected_bytes)) {
        return multimodal_text_status::size_overflow;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != expected_device) {
        return multimodal_text_status::unsupported_device;
    }
    if (!device_range_valid(prompt.projected_vision_embeddings_device,
                            projected_bytes, expected_device)) {
        return multimodal_text_status::invalid_device_pointer;
    }

    std::uint32_t maximum = 0u;
    for (std::size_t token_index = 0u; token_index < prompt.token_count;
         ++token_index) {
        const std::uint32_t temporal =
                prompt.mrope_positions[token_index];
        const std::uint32_t height =
                prompt.mrope_positions[prompt.token_count + token_index];
        const std::uint32_t width =
                prompt.mrope_positions[2u * prompt.token_count + token_index];
        if (temporal >= rope::kMaxContext || height >= rope::kMaxContext ||
            width >= rope::kMaxContext) {
            return multimodal_text_status::invalid_mrope;
        }
        if (!is_visual_token(prompt.token_ids[token_index]) &&
            (temporal != height || temporal != width)) {
            return multimodal_text_status::invalid_mrope;
        }
        maximum = std::max(maximum, std::max(temporal, std::max(height, width)));
    }
    result.maximum_position = maximum;
    if (maximum + 1u < rope::kMaxContext) {
        result.next_text_position = maximum + 1u;
        result.next_text_position_available = true;
    }
    if (report != nullptr) *report = result;
    return multimodal_text_status::ok;
}

multimodal_text_status repeat_projected_vision_embedding_cuda(
        const float *projected_embedding_2560_f32,
        float *residual_4x2560_f32,
        std::uint32_t *device_status,
        int expected_device,
        cudaStream_t stream) noexcept {
    if (projected_embedding_2560_f32 == nullptr ||
        residual_4x2560_f32 == nullptr || device_status == nullptr ||
        expected_device < 0) {
        return multimodal_text_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess ||
        active_device != expected_device) {
        return multimodal_text_status::unsupported_device;
    }
    constexpr std::size_t projected_bytes =
            kMultimodalHidden * sizeof(float);
    constexpr std::size_t residual_bytes =
            kMultimodalResidual * sizeof(float);
    if (!device_range_valid(projected_embedding_2560_f32, projected_bytes,
                            expected_device) ||
        !device_range_valid(residual_4x2560_f32, residual_bytes,
                            expected_device) ||
        !device_range_valid(device_status, sizeof(std::uint32_t),
                            expected_device) ||
        ranges_overlap(projected_embedding_2560_f32, projected_bytes,
                       residual_4x2560_f32, residual_bytes) ||
        ranges_overlap(projected_embedding_2560_f32, projected_bytes,
                       device_status, sizeof(std::uint32_t)) ||
        ranges_overlap(residual_4x2560_f32, residual_bytes,
                       device_status, sizeof(std::uint32_t))) {
        return multimodal_text_status::invalid_device_pointer;
    }
    const unsigned blocks = static_cast<unsigned>(
            (kMultimodalResidual + kThreads - 1u) / kThreads);
    repeat_projected_vision_kernel<<<blocks, kThreads, 0, stream>>>(
            projected_embedding_2560_f32, residual_4x2560_f32, device_status);
    return cudaPeekAtLastError() == cudaSuccess
            ? multimodal_text_status::ok
            : multimodal_text_status::cuda_error;
}

}  // namespace axiom::qwen4exp
