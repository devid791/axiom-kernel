#include "axiom/qwen4exp/mhc_adapter.hpp"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr std::size_t kHidden = kMhcQwen4ExpHidden;
constexpr std::size_t kStreams = kMhcQwen4ExpStreams;
constexpr std::size_t kResidual = kMhcQwen4ExpResidual;
constexpr std::size_t kRank = kMhcQwen4ExpRank;
constexpr std::size_t kWorkspaceAlignment = 256u;
constexpr unsigned kThreads = 256u;

bool multiply_overflow(std::size_t left, std::size_t right,
                       std::size_t *result) noexcept {
    if (result == nullptr ||
        (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)) {
        return true;
    }
    *result = left * right;
    return false;
}

mhc_adapter_status fail(mhc_adapter_status status, std::string *error,
                        const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = "qwen4_exp mHC adapter: " + message;
        } catch (...) {
        }
    }
    return status;
}

const char *site_name(mhc_adapter_site site) noexcept {
    return site == mhc_adapter_site::attention ? "attn_hyper_connection"
                                                : "mlp_hyper_connection";
}

std::string prefix(std::uint32_t layer, mhc_adapter_site site) {
    return "model.language_model.layers." + std::to_string(layer) + "." +
           site_name(site) + ".";
}

bool pointer_range_valid(const void *pointer, std::size_t bytes, int device,
                         std::size_t alignment) noexcept {
    if (pointer == nullptr || bytes == 0u) return false;
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
    if (alignment != 0u && address % alignment != 0u) return false;
    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status = cudaPointerGetAttributes(&attributes, pointer);
    if (pointer_status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device) return false;
    CUdeviceptr allocation_base = 0u;
    std::size_t allocation_bytes = 0u;
    if (cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                             static_cast<CUdeviceptr>(address)) != CUDA_SUCCESS ||
        allocation_base == 0u) {
        return false;
    }
    const std::uintptr_t base = static_cast<std::uintptr_t>(allocation_base);
    if (address < base) return false;
    const std::size_t offset = static_cast<std::size_t>(address - base);
    return offset <= allocation_bytes && bytes <= allocation_bytes - offset;
}

bool ranges_overlap(const void *left, std::size_t left_bytes,
                    const void *right, std::size_t right_bytes) noexcept {
    const std::uintptr_t left_begin = reinterpret_cast<std::uintptr_t>(left);
    const std::uintptr_t right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_begin > std::numeric_limits<std::uintptr_t>::max() - left_bytes ||
        right_begin > std::numeric_limits<std::uintptr_t>::max() - right_bytes) {
        return true;
    }
    return left_begin < right_begin + right_bytes &&
           right_begin < left_begin + left_bytes;
}

float bf16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0F;
    std::memcpy(&value, &expanded, sizeof(value));
    return value;
}

__device__ float sigmoidf_stable(float value) {
    if (value >= 0.0F) {
        const float inverse = expf(-value);
        return 1.0F / (1.0F + inverse);
    }
    const float exponential = expf(value);
    return exponential / (1.0F + exponential);
}

__global__ void grouped_rmsnorm_kernel(const float *input,
                                       const std::uint16_t *weight,
                                       std::size_t rows,
                                       float *output) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;
    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < kHidden;
         column += blockDim.x) {
        const float value = input[row * kHidden + column];
        sum += value * value;
    }
    __shared__ float reduction[kThreads];
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(reduction[0] / static_cast<float>(kHidden) +
                                 kMhcQwen4ExpRmsEpsilon);
    const std::size_t stream = row % kStreams;
    for (std::size_t column = threadIdx.x; column < kHidden;
         column += blockDim.x) {
        const __nv_bfloat16 raw =
                *reinterpret_cast<const __nv_bfloat16 *>(
                        weight + stream * kHidden + column);
        output[row * kHidden + column] = input[row * kHidden + column] * inverse *
                                         (1.0F + __bfloat162float(raw));
    }
}

__global__ void scaled_silu_kernel(float *values, std::size_t count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    if (index >= count) return;
    const float scaled = values[index] / static_cast<float>(kStreams);
    values[index] = scaled * sigmoidf_stable(scaled);
}

__global__ void combine_kernel(const float *normalized, const float *mix_logits,
                               const float *injection_logits, std::size_t batch,
                               float *mixed, float *injection) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    const std::size_t mixed_count = batch * kHidden;
    if (index < mixed_count) {
        const std::size_t token = index / kHidden;
        const std::size_t hidden = index % kHidden;
        float sum = 0.0F;
        for (std::size_t stream = 0u; stream < kStreams; ++stream) {
            const std::size_t source = token * kResidual + stream * kHidden + hidden;
            sum += sigmoidf_stable(mix_logits[source]) * normalized[source];
        }
        mixed[index] = sum / static_cast<float>(kStreams);
    }
    const std::size_t injection_count = batch * kStreams;
    if (index < injection_count) {
        injection[index] = 2.0F * sigmoidf_stable(
                injection_logits[index] / static_cast<float>(kStreams));
    }
}

mhc_adapter_status map_linear_status(bf16_linear_status status) noexcept {
    if (status == bf16_linear_status::ok) return mhc_adapter_status::ok;
    if (status == bf16_linear_status::size_overflow) return mhc_adapter_status::size_overflow;
    if (status == bf16_linear_status::allocation_failure) {
        return mhc_adapter_status::allocation_failure;
    }
    if (status == bf16_linear_status::invalid_device_pointer) {
        return mhc_adapter_status::invalid_device_pointer;
    }
    if (status == bf16_linear_status::cuda_error ||
        status == bf16_linear_status::cublas_error) {
        return mhc_adapter_status::linear_error;
    }
    return mhc_adapter_status::tensor_contract_mismatch;
}

}  // namespace

struct resident_mhc_adapter::impl {
    mhc_adapter_config config{};
    std::uint32_t layer = 0u;
    mhc_adapter_site site = mhc_adapter_site::attention;
    int device = -1;
    std::uint16_t *norm_weight = nullptr;
    std::unique_ptr<resident_bf16_linear> down;
    std::unique_ptr<resident_bf16_linear> up;
    std::unique_ptr<resident_bf16_linear> inject;
    bool ready = false;

    ~impl() {
        int previous = -1;
        const bool have_previous = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = have_previous && device >= 0 && previous != device &&
                             cudaSetDevice(device) == cudaSuccess;
        if (norm_weight != nullptr) (void)cudaFree(norm_weight);
        norm_weight = nullptr;
        if (changed) (void)cudaSetDevice(previous);
    }
};

const char *mhc_adapter_status_string(mhc_adapter_status status) noexcept {
    switch (status) {
        case mhc_adapter_status::ok: return "ok";
        case mhc_adapter_status::invalid_argument: return "invalid_argument";
        case mhc_adapter_status::unsupported_config: return "unsupported_config";
        case mhc_adapter_status::tensor_contract_mismatch: return "tensor_contract_mismatch";
        case mhc_adapter_status::size_overflow: return "size_overflow";
        case mhc_adapter_status::allocation_failure: return "allocation_failure";
        case mhc_adapter_status::invalid_device_pointer: return "invalid_device_pointer";
        case mhc_adapter_status::cuda_error: return "cuda_error";
        case mhc_adapter_status::linear_error: return "linear_error";
    }
    return "unknown";
}

mhc_adapter_status mhc_adapter_validate_config(
        const mhc_adapter_config &config) noexcept {
    if (config.max_batch == 0u || config.max_batch > 65535u) {
        return mhc_adapter_status::invalid_argument;
    }
    bf16_linear_config linear{};
    linear.input_features = kResidual;
    linear.output_features = kRank;
    linear.max_batch = config.max_batch;
    linear.upload_chunk_bytes = config.upload_chunk_bytes;
    linear.blas_workspace_bytes = config.blas_workspace_bytes;
    return map_linear_status(bf16_linear_validate_config(linear));
}

mhc_adapter_status mhc_adapter_get_workspace_requirements(
        const mhc_adapter_config &config,
        mhc_adapter_workspace_requirements *requirements) noexcept {
    if (requirements == nullptr) return mhc_adapter_status::invalid_argument;
    *requirements = {};
    const mhc_adapter_status validation = mhc_adapter_validate_config(config);
    if (validation != mhc_adapter_status::ok) return validation;
    auto bytes = [&](std::size_t elements, std::size_t *out) {
        return !multiply_overflow(config.max_batch, elements, out) &&
               !multiply_overflow(*out, sizeof(float), out);
    };
    if (!bytes(kResidual, &requirements->normalized_f32_bytes) ||
        !bytes(kRank, &requirements->low_rank_f32_bytes) ||
        !bytes(kResidual, &requirements->mix_logits_f32_bytes) ||
        !bytes(kStreams, &requirements->injection_logits_f32_bytes)) {
        return mhc_adapter_status::size_overflow;
    }
    std::size_t input_elements = 0u;
    if (multiply_overflow(config.max_batch, kResidual, &input_elements) ||
        multiply_overflow(input_elements, sizeof(std::uint16_t),
                          &requirements->linear_input_bf16_bytes)) {
        return mhc_adapter_status::size_overflow;
    }
    requirements->blas_workspace_bytes = config.blas_workspace_bytes;
    return mhc_adapter_status::ok;
}

resident_mhc_adapter::resident_mhc_adapter() = default;
resident_mhc_adapter::~resident_mhc_adapter() = default;
resident_mhc_adapter::resident_mhc_adapter(resident_mhc_adapter &&) noexcept = default;
resident_mhc_adapter &resident_mhc_adapter::operator=(resident_mhc_adapter &&) noexcept = default;

mhc_adapter_status resident_mhc_adapter::load(
        const checkpoint_catalog &catalog, std::uint32_t layer,
        mhc_adapter_site site, const mhc_adapter_config &config,
        cudaStream_t initialization_stream,
        std::unique_ptr<resident_mhc_adapter> *out,
        std::string *error) noexcept {
    if (out == nullptr || layer >= 48u ||
        (site != mhc_adapter_site::attention && site != mhc_adapter_site::mlp)) {
        return fail(mhc_adapter_status::invalid_argument, error, "invalid load arguments");
    }
    out->reset();
    if (error != nullptr) error->clear();
    const mhc_adapter_status validation = mhc_adapter_validate_config(config);
    if (validation != mhc_adapter_status::ok) {
        return fail(validation, error, "invalid adapter configuration");
    }
    try {
        auto result = std::make_unique<resident_mhc_adapter>();
        result->impl_ = std::make_unique<impl>();
        result->impl_->config = config;
        result->impl_->layer = layer;
        result->impl_->site = site;
        if (cudaGetDevice(&result->impl_->device) != cudaSuccess) {
            return fail(mhc_adapter_status::cuda_error, error, "cudaGetDevice failed");
        }
        const std::string base = prefix(layer, site);
        const std::string norm_name = base + "hc_norm.weight";
        const tensor_span *norm = catalog.find(norm_name);
        if (norm == nullptr || norm->dtype != tensor_dtype::bf16 || norm->rank != 1u ||
            norm->shape[0] != kResidual || norm->bytes != kResidual * sizeof(std::uint16_t)) {
            return fail(mhc_adapter_status::tensor_contract_mismatch, error,
                        "norm tensor contract mismatch: " + norm_name);
        }
        std::vector<std::uint16_t> host_norm(kResidual);
        std::string read_error;
        if (!catalog.read_range(norm_name, 0u, host_norm.data(),
                                host_norm.size() * sizeof(std::uint16_t), &read_error)) {
            return fail(mhc_adapter_status::tensor_contract_mismatch, error, read_error);
        }
        for (const std::uint16_t value : host_norm) {
            if (!std::isfinite(bf16_to_float(value))) {
                return fail(mhc_adapter_status::tensor_contract_mismatch, error,
                            "norm tensor contains a non-finite value");
            }
        }
        if (cudaMalloc(reinterpret_cast<void **>(&result->impl_->norm_weight),
                       host_norm.size() * sizeof(std::uint16_t)) != cudaSuccess ||
            cudaMemcpyAsync(result->impl_->norm_weight, host_norm.data(),
                            host_norm.size() * sizeof(std::uint16_t),
                            cudaMemcpyHostToDevice, initialization_stream) != cudaSuccess ||
            cudaStreamSynchronize(initialization_stream) != cudaSuccess) {
            return fail(mhc_adapter_status::allocation_failure, error,
                        "norm tensor GPU upload failed");
        }

        auto load_linear = [&](const std::string &name, std::uint64_t input,
                               std::uint64_t output,
                               std::unique_ptr<resident_bf16_linear> *linear) {
            bf16_linear_config linear_config{};
            linear_config.input_features = input;
            linear_config.output_features = output;
            linear_config.max_batch = config.max_batch;
            linear_config.upload_chunk_bytes = config.upload_chunk_bytes;
            linear_config.blas_workspace_bytes = config.blas_workspace_bytes;
            std::string linear_error;
            const bf16_linear_status status = resident_bf16_linear::load(
                    catalog, name, linear_config, initialization_stream, linear, &linear_error);
            if (status != bf16_linear_status::ok) {
                if (error != nullptr) *error = linear_error;
                return map_linear_status(status);
            }
            return mhc_adapter_status::ok;
        };
        mhc_adapter_status status = load_linear(
                base + "input_mix_weight_down.weight", kResidual, kRank,
                &result->impl_->down);
        if (status == mhc_adapter_status::ok) {
            status = load_linear(base + "input_mix_weight_up.weight", kRank, kResidual,
                                 &result->impl_->up);
        }
        if (status == mhc_adapter_status::ok) {
            status = load_linear(base + "block_inject_weight.weight", kResidual, kStreams,
                                 &result->impl_->inject);
        }
        if (status != mhc_adapter_status::ok) return status;
        result->impl_->ready = true;
        *out = std::move(result);
        return mhc_adapter_status::ok;
    } catch (const std::bad_alloc &) {
        return fail(mhc_adapter_status::allocation_failure, error, "host allocation failed");
    } catch (const std::exception &exception) {
        return fail(mhc_adapter_status::allocation_failure, error, exception.what());
    } catch (...) {
        return fail(mhc_adapter_status::allocation_failure, error, "unknown load failure");
    }
}

mhc_adapter_status resident_mhc_adapter::prepare(
        const float *hyper_input, std::size_t batch,
        const mhc_adapter_scratch &scratch, float *mixed_input,
        float *injection_weights, cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || hyper_input == nullptr || mixed_input == nullptr ||
        injection_weights == nullptr || scratch.normalized == nullptr ||
        scratch.low_rank == nullptr || scratch.mix_logits == nullptr ||
        scratch.injection_logits == nullptr || scratch.linear_input_bf16 == nullptr ||
        scratch.blas_workspace == nullptr || batch == 0u ||
        batch > impl_->config.max_batch) {
        return mhc_adapter_status::invalid_argument;
    }
    int active_device = -1;
    if (cudaGetDevice(&active_device) != cudaSuccess || active_device != impl_->device) {
        return mhc_adapter_status::cuda_error;
    }
    std::size_t residual_bytes = 0u;
    std::size_t rank_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    std::size_t streams_bytes = 0u;
    std::size_t bf16_bytes = 0u;
    if (multiply_overflow(batch, kResidual * sizeof(float), &residual_bytes) ||
        multiply_overflow(batch, kRank * sizeof(float), &rank_bytes) ||
        multiply_overflow(batch, kHidden * sizeof(float), &hidden_bytes) ||
        multiply_overflow(batch, kStreams * sizeof(float), &streams_bytes) ||
        multiply_overflow(batch, kResidual * sizeof(std::uint16_t), &bf16_bytes)) {
        return mhc_adapter_status::size_overflow;
    }
    struct range { const void *pointer; std::size_t bytes; std::size_t alignment; };
    const std::array<range, 9> ranges{{
        {hyper_input, residual_bytes, alignof(float)},
        {scratch.normalized, residual_bytes, alignof(float)},
        {scratch.low_rank, rank_bytes, alignof(float)},
        {scratch.mix_logits, residual_bytes, alignof(float)},
        {scratch.injection_logits, streams_bytes, alignof(float)},
        {mixed_input, hidden_bytes, alignof(float)},
        {injection_weights, streams_bytes, alignof(float)},
        {scratch.linear_input_bf16, bf16_bytes, alignof(std::uint16_t)},
        {scratch.blas_workspace, impl_->config.blas_workspace_bytes, kWorkspaceAlignment},
    }};
    for (const range &item : ranges) {
        if (!pointer_range_valid(item.pointer, item.bytes, impl_->device, item.alignment)) {
            return mhc_adapter_status::invalid_device_pointer;
        }
    }
    for (std::size_t left = 0u; left < ranges.size(); ++left) {
        for (std::size_t right = left + 1u; right < ranges.size(); ++right) {
            if (ranges_overlap(ranges[left].pointer, ranges[left].bytes,
                               ranges[right].pointer, ranges[right].bytes)) {
                return mhc_adapter_status::invalid_device_pointer;
            }
        }
    }
    if (scratch.linear_input_bf16_bytes < bf16_bytes ||
        scratch.blas_workspace_bytes < impl_->config.blas_workspace_bytes) {
        return mhc_adapter_status::invalid_device_pointer;
    }

    grouped_rmsnorm_kernel<<<static_cast<unsigned>(batch * kStreams), kThreads, 0, stream>>>(
            hyper_input, impl_->norm_weight, batch * kStreams, scratch.normalized);
    if (cudaPeekAtLastError() != cudaSuccess) return mhc_adapter_status::cuda_error;
    const bf16_linear_scratch linear_scratch{
        scratch.linear_input_bf16, scratch.linear_input_bf16_bytes,
        scratch.blas_workspace, scratch.blas_workspace_bytes};
    bf16_linear_status linear = impl_->down->forward(
            scratch.normalized, batch, linear_scratch, scratch.low_rank, stream);
    if (linear != bf16_linear_status::ok) return map_linear_status(linear);
    const std::size_t rank_count = batch * kRank;
    const std::size_t blocks = (rank_count + kThreads - 1u) / kThreads;
    scaled_silu_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            scratch.low_rank, rank_count);
    if (cudaPeekAtLastError() != cudaSuccess) return mhc_adapter_status::cuda_error;
    linear = impl_->up->forward(scratch.low_rank, batch, linear_scratch,
                                scratch.mix_logits, stream);
    if (linear != bf16_linear_status::ok) return map_linear_status(linear);
    linear = impl_->inject->forward(scratch.normalized, batch, linear_scratch,
                                    scratch.injection_logits, stream);
    if (linear != bf16_linear_status::ok) return map_linear_status(linear);
    const std::size_t combine_count = std::max(batch * kHidden, batch * kStreams);
    const std::size_t combine_blocks = (combine_count + kThreads - 1u) / kThreads;
    combine_kernel<<<static_cast<unsigned>(combine_blocks), kThreads, 0, stream>>>(
            scratch.normalized, scratch.mix_logits, scratch.injection_logits,
            batch, mixed_input, injection_weights);
    return cudaPeekAtLastError() == cudaSuccess ? mhc_adapter_status::ok
                                                : mhc_adapter_status::cuda_error;
}

mhc_adapter_status resident_mhc_adapter::reinject(
        const float *hyper_input, const float *block_output,
        const float *injection_weights, std::size_t batch, float *output,
        cudaStream_t stream) noexcept {
    if (!impl_ || !impl_->ready || batch == 0u || batch > impl_->config.max_batch) {
        return mhc_adapter_status::invalid_argument;
    }
    const MhcStatus status = mhc_reinject_cuda(
            mhc_qwen4_exp_config(), hyper_input, block_output, injection_weights,
            batch, output, stream);
    if (status == MhcStatus::kOk) return mhc_adapter_status::ok;
    if (status == MhcStatus::kSizeOverflow) return mhc_adapter_status::size_overflow;
    if (status == MhcStatus::kCudaError) return mhc_adapter_status::cuda_error;
    return mhc_adapter_status::invalid_argument;
}

std::uint32_t resident_mhc_adapter::layer() const noexcept {
    return impl_ ? impl_->layer : 0u;
}

mhc_adapter_site resident_mhc_adapter::site() const noexcept {
    return impl_ ? impl_->site : mhc_adapter_site::attention;
}

int resident_mhc_adapter::device() const noexcept {
    return impl_ ? impl_->device : -1;
}

bool resident_mhc_adapter::initialized() const noexcept {
    return impl_ && impl_->ready;
}

}  // namespace axiom::qwen4exp
