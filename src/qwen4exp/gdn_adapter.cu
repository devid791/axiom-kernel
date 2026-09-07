#include "axiom/qwen4exp/gdn_adapter.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kMinimumUploadChunk = 4u * 1024u;
constexpr std::size_t kMaximumUploadChunk = 64u * 1024u * 1024u;
constexpr std::size_t kMaximumBlasWorkspace = 64u * 1024u * 1024u;
constexpr std::size_t kWorkspaceAlignment = 256u;

bool checked_mul(std::size_t left,
                 std::size_t right,
                 std::size_t* result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left,
                 std::size_t right,
                 std::size_t* result) noexcept {
    if (result == nullptr ||
        right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    *result = left + right;
    return true;
}

GdnAdapterStatus fail(GdnAdapterStatus status,
                      std::string* error,
                      const std::string& message) noexcept {
    if (error != nullptr) {
        try {
            *error = message;
        } catch (...) {
        }
    }
    return status;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

bool current_sm120_device(int* device) noexcept {
    if (device == nullptr || cudaGetDevice(device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, *device) == cudaSuccess &&
           properties.major == 12 && properties.minor == 0;
}

struct DeviceRange {
    const void* pointer = nullptr;
    std::size_t bytes = 0u;
    std::size_t alignment = 1u;
};

bool device_range_valid(const DeviceRange& range, int expected_device) noexcept {
    if (range.pointer == nullptr || range.bytes == 0u) return false;
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(range.pointer);
    if (range.alignment != 0u && (address % range.alignment) != 0u) return false;

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status =
        cudaPointerGetAttributes(&attributes, range.pointer);
    if (pointer_status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice ||
        attributes.device != expected_device) {
        return false;
    }

    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    const CUresult address_status = cuMemGetAddressRange(
        &allocation_base,
        &allocation_bytes,
        static_cast<CUdeviceptr>(address));
    if (address_status != CUDA_SUCCESS) return false;
    const std::uintptr_t base = static_cast<std::uintptr_t>(allocation_base);
    if (address < base || range.bytes > allocation_bytes) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes - range.bytes;
}

bool ranges_overlap(const DeviceRange& left, const DeviceRange& right) noexcept {
    const std::uintptr_t left_begin =
        reinterpret_cast<std::uintptr_t>(left.pointer);
    const std::uintptr_t right_begin =
        reinterpret_cast<std::uintptr_t>(right.pointer);
    if (left.bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
        right.bytes > std::numeric_limits<std::uintptr_t>::max() - right_begin) {
        return true;
    }
    const std::uintptr_t left_end = left_begin + left.bytes;
    const std::uintptr_t right_end = right_begin + right.bytes;
    return left_begin < right_end && right_begin < left_end;
}

GdnAdapterStatus map_linear_status(bf16_linear_status status) noexcept {
    switch (status) {
        case bf16_linear_status::ok: return GdnAdapterStatus::kOk;
        case bf16_linear_status::invalid_argument:
            return GdnAdapterStatus::kInvalidArgument;
        case bf16_linear_status::unsupported_config:
            return GdnAdapterStatus::kUnsupportedConfig;
        case bf16_linear_status::tensor_not_found:
            return GdnAdapterStatus::kTensorNotFound;
        case bf16_linear_status::dtype_mismatch:
            return GdnAdapterStatus::kDtypeMismatch;
        case bf16_linear_status::shape_mismatch:
            return GdnAdapterStatus::kShapeMismatch;
        case bf16_linear_status::size_overflow:
            return GdnAdapterStatus::kSizeOverflow;
        case bf16_linear_status::unsupported_device:
            return GdnAdapterStatus::kUnsupportedDevice;
        case bf16_linear_status::invalid_device_pointer:
            return GdnAdapterStatus::kInvalidDevicePointer;
        case bf16_linear_status::checkpoint_io_error:
            return GdnAdapterStatus::kCheckpointIoError;
        case bf16_linear_status::allocation_failure:
            return GdnAdapterStatus::kAllocationFailure;
        case bf16_linear_status::cuda_error:
        case bf16_linear_status::cublas_error:
            return GdnAdapterStatus::kLinearError;
    }
    return GdnAdapterStatus::kLinearError;
}

GdnAdapterStatus map_gdn_status(GdnStatus status) noexcept {
    switch (status) {
        case GdnStatus::kOk: return GdnAdapterStatus::kOk;
        case GdnStatus::kInvalidArgument:
            return GdnAdapterStatus::kInvalidArgument;
        case GdnStatus::kUnsupportedConfig:
        case GdnStatus::kDimensionMismatch:
            return GdnAdapterStatus::kUnsupportedConfig;
        case GdnStatus::kSizeOverflow:
            return GdnAdapterStatus::kSizeOverflow;
        case GdnStatus::kUninitialized:
        case GdnStatus::kStateMismatch:
            return GdnAdapterStatus::kStateMismatch;
        case GdnStatus::kUnsupportedDevice:
            return GdnAdapterStatus::kUnsupportedDevice;
        case GdnStatus::kCudaError:
            return GdnAdapterStatus::kGdnError;
    }
    return GdnAdapterStatus::kGdnError;
}

std::string tensor_prefix(std::size_t layer_index) {
    return "model.language_model.layers." + std::to_string(layer_index) +
           ".linear_attn.";
}

struct SmallTensorSpec {
    const char* suffix = nullptr;
    std::uint32_t rank = 0u;
    std::array<std::uint64_t, 3> shape{};
    std::size_t elements = 0u;
};

GdnAdapterStatus load_small_bf16_as_f32(
    const checkpoint_catalog& catalog,
    const std::string& name,
    const SmallTensorSpec& spec,
    std::size_t upload_chunk_bytes,
    cudaStream_t stream,
    float** output,
    std::string* error) noexcept {
    if (output == nullptr || *output != nullptr || name.empty() ||
        spec.rank == 0u || spec.rank > spec.shape.size() || spec.elements == 0u) {
        return fail(GdnAdapterStatus::kInvalidArgument, error,
                    "invalid small-tensor load request: " + name);
    }
    const tensor_span* span = catalog.find(name);
    if (span == nullptr) {
        return fail(GdnAdapterStatus::kTensorNotFound, error,
                    "checkpoint tensor not found: " + name);
    }
    if (span->dtype != tensor_dtype::bf16) {
        return fail(GdnAdapterStatus::kDtypeMismatch, error,
                    "checkpoint tensor is not BF16: " + name);
    }
    if (span->rank != spec.rank) {
        return fail(GdnAdapterStatus::kShapeMismatch, error,
                    "checkpoint tensor rank mismatch: " + name);
    }
    for (std::uint32_t index = 0u; index < spec.rank; ++index) {
        if (span->shape[index] != spec.shape[index]) {
            return fail(GdnAdapterStatus::kShapeMismatch, error,
                        "checkpoint tensor shape mismatch: " + name);
        }
    }
    std::size_t encoded_bytes = 0u;
    std::size_t decoded_bytes = 0u;
    if (!checked_mul(spec.elements, sizeof(std::uint16_t), &encoded_bytes) ||
        !checked_mul(spec.elements, sizeof(float), &decoded_bytes) ||
        span->bytes != encoded_bytes) {
        return fail(GdnAdapterStatus::kSizeOverflow, error,
                    "checkpoint tensor byte count mismatch: " + name);
    }

    float* device_output = nullptr;
    std::uint16_t* encoded = nullptr;
    float* decoded = nullptr;
    const std::size_t staging_bytes =
        std::min(upload_chunk_bytes, encoded_bytes);
    const std::size_t staging_elements =
        std::max<std::size_t>(1u, staging_bytes / sizeof(std::uint16_t));
    const auto cleanup = [&]() noexcept {
        if (encoded != nullptr) (void)cudaFreeHost(encoded);
        if (decoded != nullptr) (void)cudaFreeHost(decoded);
        if (device_output != nullptr) (void)cudaFree(device_output);
    };
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), decoded_bytes) !=
            cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&encoded),
                      staging_elements * sizeof(std::uint16_t),
                      cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&decoded),
                      staging_elements * sizeof(float),
                      cudaHostAllocPortable) != cudaSuccess) {
        cleanup();
        return fail(GdnAdapterStatus::kAllocationFailure, error,
                    "bounded staging allocation failed: " + name);
    }

    std::size_t offset = 0u;
    while (offset < spec.elements) {
        const std::size_t count =
            std::min(staging_elements, spec.elements - offset);
        std::string read_error;
        if (!catalog.read_range(name,
                                offset * sizeof(std::uint16_t),
                                encoded,
                                count * sizeof(std::uint16_t),
                                &read_error)) {
            cleanup();
            return fail(GdnAdapterStatus::kCheckpointIoError, error,
                        read_error.empty()
                            ? "checkpoint range read failed: " + name
                            : read_error);
        }
        for (std::size_t index = 0u; index < count; ++index) {
            decoded[index] = bf16_to_float(encoded[index]);
        }
        if (cudaMemcpyAsync(device_output + offset,
                            decoded,
                            count * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            cleanup();
            return fail(GdnAdapterStatus::kCudaError, error,
                        "small BF16 tensor conversion/upload failed: " + name);
        }
        offset += count;
    }

    const cudaError_t encoded_free = cudaFreeHost(encoded);
    encoded = nullptr;
    const cudaError_t decoded_free = cudaFreeHost(decoded);
    decoded = nullptr;
    if (encoded_free != cudaSuccess || decoded_free != cudaSuccess) {
        cleanup();
        return fail(GdnAdapterStatus::kCudaError, error,
                    "small tensor staging release failed: " + name);
    }
    *output = device_output;
    return GdnAdapterStatus::kOk;
}

GdnAdapterStatus load_linear(
    const checkpoint_catalog& catalog,
    const std::string& name,
    std::size_t input_features,
    std::size_t output_features,
    const GdnAdapterConfig& adapter_config,
    cudaStream_t stream,
    std::unique_ptr<resident_bf16_linear>* output,
    std::string* error) noexcept {
    bf16_linear_config config{};
    config.input_features = input_features;
    config.output_features = output_features;
    config.max_batch = adapter_config.max_batch;
    config.upload_chunk_bytes = adapter_config.upload_chunk_bytes;
    config.blas_workspace_bytes = adapter_config.blas_workspace_bytes;
    const bf16_linear_status status = resident_bf16_linear::load(
        catalog, name, config, stream, output, error);
    return map_linear_status(status);
}

}  // namespace

struct GdnLayerAdapter::impl {
    GdnAdapterConfig config{};
    std::unique_ptr<resident_bf16_linear> in_proj_qkv;
    std::unique_ptr<resident_bf16_linear> in_proj_z;
    std::unique_ptr<resident_bf16_linear> in_proj_b;
    std::unique_ptr<resident_bf16_linear> in_proj_a;
    std::unique_ptr<resident_bf16_linear> out_proj;
    float* conv_weight = nullptr;
    float* A_log = nullptr;
    float* dt_bias = nullptr;
    float* norm_weight = nullptr;
    int device = -1;
    bool ready = false;

    ~impl() {
        if (conv_weight != nullptr) (void)cudaFree(conv_weight);
        if (A_log != nullptr) (void)cudaFree(A_log);
        if (dt_bias != nullptr) (void)cudaFree(dt_bias);
        if (norm_weight != nullptr) (void)cudaFree(norm_weight);
    }

    [[nodiscard]] GdnWeightsView weights() const noexcept {
        return GdnWeightsView{
            conv_weight, nullptr, A_log, dt_bias, norm_weight};
    }
};

const char* gdn_adapter_status_string(GdnAdapterStatus status) noexcept {
    switch (status) {
        case GdnAdapterStatus::kOk: return "ok";
        case GdnAdapterStatus::kInvalidArgument: return "invalid_argument";
        case GdnAdapterStatus::kUnsupportedLayer: return "unsupported_layer";
        case GdnAdapterStatus::kUnsupportedConfig: return "unsupported_config";
        case GdnAdapterStatus::kTensorNotFound: return "tensor_not_found";
        case GdnAdapterStatus::kDtypeMismatch: return "dtype_mismatch";
        case GdnAdapterStatus::kShapeMismatch: return "shape_mismatch";
        case GdnAdapterStatus::kSizeOverflow: return "size_overflow";
        case GdnAdapterStatus::kCheckpointIoError: return "checkpoint_io_error";
        case GdnAdapterStatus::kAllocationFailure: return "allocation_failure";
        case GdnAdapterStatus::kUnsupportedDevice: return "unsupported_device";
        case GdnAdapterStatus::kInvalidDevicePointer:
            return "invalid_device_pointer";
        case GdnAdapterStatus::kStateMismatch: return "state_mismatch";
        case GdnAdapterStatus::kLinearError: return "linear_error";
        case GdnAdapterStatus::kGdnError: return "gdn_error";
        case GdnAdapterStatus::kCudaError: return "cuda_error";
    }
    return "unknown";
}

GdnAdapterStatus gdn_adapter_validate_config(
    const GdnAdapterConfig& config) noexcept {
    if (config.layer_index >= kGdnAdapterLayerCount || config.max_batch == 0u) {
        return GdnAdapterStatus::kInvalidArgument;
    }
    if ((config.layer_index % 4u) == 3u) {
        return GdnAdapterStatus::kUnsupportedLayer;
    }
    if (config.max_batch > kGdnMaxBatch ||
        config.upload_chunk_bytes < kMinimumUploadChunk ||
        config.upload_chunk_bytes > kMaximumUploadChunk ||
        (config.upload_chunk_bytes % sizeof(std::uint16_t)) != 0u ||
        config.blas_workspace_bytes == 0u ||
        config.blas_workspace_bytes > kMaximumBlasWorkspace ||
        (config.blas_workspace_bytes % kWorkspaceAlignment) != 0u) {
        return GdnAdapterStatus::kUnsupportedConfig;
    }
    bf16_linear_config linear{};
    linear.input_features = kGdnAdapterValueSize;
    linear.output_features = kGdnAdapterHiddenSize;
    linear.max_batch = config.max_batch;
    linear.upload_chunk_bytes = config.upload_chunk_bytes;
    linear.blas_workspace_bytes = config.blas_workspace_bytes;
    return bf16_linear_validate_config(linear) == bf16_linear_status::ok
        ? GdnAdapterStatus::kOk
        : GdnAdapterStatus::kUnsupportedConfig;
}

GdnAdapterStatus gdn_adapter_workspace_requirements(
    const GdnAdapterConfig& config,
    GdnAdapterWorkspaceRequirements* requirements) noexcept {
    if (requirements == nullptr) return GdnAdapterStatus::kInvalidArgument;
    *requirements = GdnAdapterWorkspaceRequirements{};
    const GdnAdapterStatus validation = gdn_adapter_validate_config(config);
    if (validation != GdnAdapterStatus::kOk) return validation;

    auto bytes_for = [&](std::size_t width,
                         std::size_t element_bytes,
                         std::size_t* output) noexcept {
        std::size_t elements = 0u;
        return checked_mul(config.max_batch, width, &elements) &&
               checked_mul(elements, element_bytes, output);
    };
    if (!bytes_for(kGdnAdapterValueSize,
                   sizeof(std::uint16_t),
                   &requirements->linear_input_bf16_bytes) ||
        !bytes_for(kGdnAdapterQkvSize,
                   sizeof(float),
                   &requirements->projected_qkv_f32_bytes) ||
        !bytes_for(kGdnAdapterValueSize,
                   sizeof(float),
                   &requirements->z_f32_bytes) ||
        !bytes_for(kGdnAdapterScalarSize,
                   sizeof(float),
                   &requirements->a_f32_bytes) ||
        !bytes_for(kGdnAdapterScalarSize,
                   sizeof(float),
                   &requirements->b_f32_bytes) ||
        !bytes_for(kGdnAdapterQkvSize,
                   sizeof(float),
                   &requirements->convolved_qkv_f32_bytes) ||
        !bytes_for(kGdnAdapterValueSize,
                   sizeof(float),
                   &requirements->gdn_output_f32_bytes)) {
        return GdnAdapterStatus::kSizeOverflow;
    }
    requirements->blas_workspace_bytes = config.blas_workspace_bytes;
    const std::array<std::size_t, 8> parts{
        requirements->linear_input_bf16_bytes,
        requirements->blas_workspace_bytes,
        requirements->projected_qkv_f32_bytes,
        requirements->z_f32_bytes,
        requirements->a_f32_bytes,
        requirements->b_f32_bytes,
        requirements->convolved_qkv_f32_bytes,
        requirements->gdn_output_f32_bytes,
    };
    for (const std::size_t part : parts) {
        if (!checked_add(requirements->total_device_bytes,
                         part,
                         &requirements->total_device_bytes)) {
            return GdnAdapterStatus::kSizeOverflow;
        }
    }
    return GdnAdapterStatus::kOk;
}

GdnLayerAdapter::GdnLayerAdapter() = default;
GdnLayerAdapter::~GdnLayerAdapter() = default;
GdnLayerAdapter::GdnLayerAdapter(GdnLayerAdapter&&) noexcept = default;
GdnLayerAdapter& GdnLayerAdapter::operator=(GdnLayerAdapter&&) noexcept = default;

GdnAdapterStatus GdnLayerAdapter::load(
    const checkpoint_catalog& catalog,
    const GdnAdapterConfig& config,
    cudaStream_t initialization_stream,
    std::unique_ptr<GdnLayerAdapter>* out,
    std::string* error) noexcept {
    if (out == nullptr) {
        return fail(GdnAdapterStatus::kInvalidArgument, error,
                    "null GDN adapter output");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const GdnAdapterStatus validation = gdn_adapter_validate_config(config);
    if (validation != GdnAdapterStatus::kOk) {
        return fail(validation, error,
                    "invalid GDN adapter configuration: " +
                        std::string(gdn_adapter_status_string(validation)));
    }
    int device = -1;
    if (!current_sm120_device(&device)) {
        return fail(GdnAdapterStatus::kUnsupportedDevice, error,
                    "GDN adapter requires an active SM120 Blackwell device");
    }

    try {
        auto result = std::make_unique<GdnLayerAdapter>();
        result->impl_ = std::make_unique<impl>();
        result->impl_->config = config;
        result->impl_->device = device;
        const std::string prefix = tensor_prefix(config.layer_index);

        using LinearMember =
            std::unique_ptr<resident_bf16_linear> impl::*;
        struct LinearLoad {
            const char* suffix;
            std::size_t input;
            std::size_t output;
            LinearMember member;
        };
        const std::array<LinearLoad, 5> linears{{
            {"in_proj_qkv.weight", kGdnAdapterHiddenSize,
             kGdnAdapterQkvSize, &impl::in_proj_qkv},
            {"in_proj_z.weight", kGdnAdapterHiddenSize,
             kGdnAdapterValueSize, &impl::in_proj_z},
            {"in_proj_b.weight", kGdnAdapterHiddenSize,
             kGdnAdapterScalarSize, &impl::in_proj_b},
            {"in_proj_a.weight", kGdnAdapterHiddenSize,
             kGdnAdapterScalarSize, &impl::in_proj_a},
            {"out_proj.weight", kGdnAdapterValueSize,
             kGdnAdapterHiddenSize, &impl::out_proj},
        }};
        for (const LinearLoad& linear : linears) {
            const GdnAdapterStatus status = load_linear(
                catalog,
                prefix + linear.suffix,
                linear.input,
                linear.output,
                config,
                initialization_stream,
                &(result->impl_.get()->*(linear.member)),
                error);
            if (status != GdnAdapterStatus::kOk) return status;
        }

        const std::array<SmallTensorSpec, 4> small{{
            {"conv1d.weight", 3u,
             {kGdnAdapterQkvSize, 1u, kGdnQwen4ExpConvKernel},
             kGdnAdapterQkvSize * kGdnQwen4ExpConvKernel},
            {"A_log", 1u, {kGdnAdapterScalarSize, 0u, 0u},
             kGdnAdapterScalarSize},
            {"dt_bias", 1u, {kGdnAdapterScalarSize, 0u, 0u},
             kGdnAdapterScalarSize},
            {"norm.weight", 1u, {kGdnQwen4ExpValueHeadDim, 0u, 0u},
             kGdnQwen4ExpValueHeadDim},
        }};
        const std::array<float**, 4> destinations{{
            &result->impl_->conv_weight,
            &result->impl_->A_log,
            &result->impl_->dt_bias,
            &result->impl_->norm_weight,
        }};
        for (std::size_t index = 0u; index < small.size(); ++index) {
            const GdnAdapterStatus status = load_small_bf16_as_f32(
                catalog,
                prefix + small[index].suffix,
                small[index],
                config.upload_chunk_bytes,
                initialization_stream,
                destinations[index],
                error);
            if (status != GdnAdapterStatus::kOk) return status;
        }

        result->impl_->ready = true;
        *out = std::move(result);
        return GdnAdapterStatus::kOk;
    } catch (const std::bad_alloc&) {
        return fail(GdnAdapterStatus::kAllocationFailure, error,
                    "host allocation failed while loading GDN adapter");
    } catch (const std::exception& exception) {
        return fail(GdnAdapterStatus::kAllocationFailure, error,
                    std::string("GDN adapter initialization exception: ") +
                        exception.what());
    } catch (...) {
        return fail(GdnAdapterStatus::kAllocationFailure, error,
                    "unknown GDN adapter initialization exception");
    }
}

GdnAdapterStatus GdnLayerAdapter::forward(
    const float* input_f32,
    std::size_t batch,
    GdnDeviceState* state,
    const GdnAdapterScratch& scratch,
    float* output_f32,
    cudaStream_t stream) noexcept {
    return forward_rows(input_f32,
                        batch,
                        false,
                        state,
                        scratch,
                        output_f32,
                        stream);
}

GdnAdapterStatus GdnLayerAdapter::forward_sequence(
    const float* input_f32,
    std::size_t token_count,
    GdnDeviceState* state,
    const GdnAdapterScratch& scratch,
    float* output_f32,
    cudaStream_t stream) noexcept {
    return forward_rows(input_f32,
                        token_count,
                        true,
                        state,
                        scratch,
                        output_f32,
                        stream);
}

GdnAdapterStatus GdnLayerAdapter::forward_rows(
    const float* input_f32,
    std::size_t row_count,
    bool contiguous_sequence,
    GdnDeviceState* state,
    const GdnAdapterScratch& scratch,
    float* output_f32,
    cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || input_f32 == nullptr ||
        output_f32 == nullptr || state == nullptr || row_count == 0u ||
        row_count > impl_->config.max_batch ||
        (contiguous_sequence && row_count > kGdnMaxSequenceTokens)) {
        return GdnAdapterStatus::kInvalidArgument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess) {
        return GdnAdapterStatus::kCudaError;
    }
    if (active_device != impl_->device) {
        return GdnAdapterStatus::kUnsupportedDevice;
    }
    const std::size_t required_state_batch =
        contiguous_sequence ? 1u : row_count;
    if (!state->initialized || state->device != impl_->device ||
        state->config.batch != required_state_batch ||
        !gdn_is_qwen4_exp_contract(state->config)) {
        return GdnAdapterStatus::kStateMismatch;
    }

    GdnAdapterConfig active_config = impl_->config;
    active_config.max_batch = row_count;
    GdnAdapterWorkspaceRequirements required{};
    const GdnAdapterStatus requirement_status =
        gdn_adapter_workspace_requirements(active_config, &required);
    if (requirement_status != GdnAdapterStatus::kOk) return requirement_status;
    if (scratch.linear_input_bf16_bytes < required.linear_input_bf16_bytes ||
        scratch.blas_workspace_bytes < required.blas_workspace_bytes ||
        scratch.projected_qkv_f32_bytes < required.projected_qkv_f32_bytes ||
        scratch.z_f32_bytes < required.z_f32_bytes ||
        scratch.a_f32_bytes < required.a_f32_bytes ||
        scratch.b_f32_bytes < required.b_f32_bytes ||
        scratch.convolved_qkv_f32_bytes < required.convolved_qkv_f32_bytes ||
        scratch.gdn_output_f32_bytes < required.gdn_output_f32_bytes) {
        return GdnAdapterStatus::kInvalidDevicePointer;
    }

    std::size_t hidden_elements = 0u;
    std::size_t input_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!checked_mul(row_count, kGdnAdapterHiddenSize, &hidden_elements) ||
        !checked_mul(hidden_elements, sizeof(float), &input_bytes) ||
        !checked_mul(hidden_elements, sizeof(float), &output_bytes)) {
        return GdnAdapterStatus::kSizeOverflow;
    }
    const std::array<DeviceRange, 10> ranges{{
        {input_f32, input_bytes, alignof(float)},
        {output_f32, output_bytes, alignof(float)},
        {scratch.linear_input_bf16,
         required.linear_input_bf16_bytes,
         alignof(std::uint16_t)},
        {scratch.blas_workspace,
         required.blas_workspace_bytes,
         kWorkspaceAlignment},
        {scratch.projected_qkv,
         required.projected_qkv_f32_bytes,
         alignof(float)},
        {scratch.z, required.z_f32_bytes, alignof(float)},
        {scratch.a, required.a_f32_bytes, alignof(float)},
        {scratch.b, required.b_f32_bytes, alignof(float)},
        {scratch.convolved_qkv,
         required.convolved_qkv_f32_bytes,
         alignof(float)},
        {scratch.gdn_output,
         required.gdn_output_f32_bytes,
         alignof(float)},
    }};
    for (const DeviceRange& range : ranges) {
        if (!device_range_valid(range, impl_->device)) {
            return GdnAdapterStatus::kInvalidDevicePointer;
        }
    }
    for (std::size_t left = 0u; left < ranges.size(); ++left) {
        for (std::size_t right = left + 1u; right < ranges.size(); ++right) {
            if (ranges_overlap(ranges[left], ranges[right])) {
                return GdnAdapterStatus::kInvalidDevicePointer;
            }
        }
    }

    const bf16_linear_scratch linear_scratch{
        scratch.linear_input_bf16,
        scratch.linear_input_bf16_bytes,
        scratch.blas_workspace,
        scratch.blas_workspace_bytes,
    };
    const auto run_linear = [&](resident_bf16_linear* linear,
                                const float* input,
                                float* output) noexcept {
        return map_linear_status(
            linear->forward(input, row_count, linear_scratch, output, stream));
    };
    GdnAdapterStatus status = run_linear(
        impl_->in_proj_qkv.get(), input_f32, scratch.projected_qkv);
    if (status != GdnAdapterStatus::kOk) return status;
    status = run_linear(impl_->in_proj_z.get(), input_f32, scratch.z);
    if (status != GdnAdapterStatus::kOk) return status;
    status = run_linear(impl_->in_proj_b.get(), input_f32, scratch.b);
    if (status != GdnAdapterStatus::kOk) return status;
    status = run_linear(impl_->in_proj_a.get(), input_f32, scratch.a);
    if (status != GdnAdapterStatus::kOk) return status;

    const GdnStepInputs inputs{
        scratch.projected_qkv, scratch.z, scratch.a, scratch.b};
    const GdnStepOutputs outputs{
        scratch.convolved_qkv, scratch.gdn_output};
    status = map_gdn_status(contiguous_sequence
        ? gdn_forward_sequence_cuda(
              state, row_count, inputs, impl_->weights(), outputs, stream)
        : gdn_step_cuda(state, inputs, impl_->weights(), outputs, stream));
    if (status != GdnAdapterStatus::kOk) return status;
    return run_linear(impl_->out_proj.get(), scratch.gdn_output, output_f32);
}

const GdnAdapterConfig& GdnLayerAdapter::config() const noexcept {
    static const GdnAdapterConfig empty{};
    return impl_ ? impl_->config : empty;
}

std::size_t GdnLayerAdapter::layer_index() const noexcept {
    return impl_ ? impl_->config.layer_index : 0u;
}

int GdnLayerAdapter::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool GdnLayerAdapter::initialized() const noexcept {
    return impl_ && impl_->ready;
}

GdnWeightsView GdnLayerAdapter::device_weights() const noexcept {
    return impl_ ? impl_->weights() : GdnWeightsView{};
}

}  // namespace axiom::qwen4exp
