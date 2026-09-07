#ifndef AXIOM_QWEN4EXP_GDN_ADAPTER_HPP
#define AXIOM_QWEN4EXP_GDN_ADAPTER_HPP

#include "axiom/qwen4exp/bf16_linear.hpp"
#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/gdn.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kGdnAdapterLayerCount = 48u;
inline constexpr std::size_t kGdnAdapterHiddenSize = 2560u;
inline constexpr std::size_t kGdnAdapterQkvSize = 10240u;
inline constexpr std::size_t kGdnAdapterValueSize = 6144u;
inline constexpr std::size_t kGdnAdapterScalarSize = 48u;

enum class GdnAdapterStatus : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kUnsupportedLayer,
    kUnsupportedConfig,
    kTensorNotFound,
    kDtypeMismatch,
    kShapeMismatch,
    kSizeOverflow,
    kCheckpointIoError,
    kAllocationFailure,
    kUnsupportedDevice,
    kInvalidDevicePointer,
    kStateMismatch,
    kLinearError,
    kGdnError,
    kCudaError,
};

struct GdnAdapterConfig {
    std::size_t layer_index = 0u;
    std::size_t max_batch = 1u;
    std::size_t upload_chunk_bytes = 256u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct GdnAdapterWorkspaceRequirements {
    std::size_t linear_input_bf16_bytes = 0u;
    std::size_t blas_workspace_bytes = 0u;
    std::size_t projected_qkv_f32_bytes = 0u;
    std::size_t z_f32_bytes = 0u;
    std::size_t a_f32_bytes = 0u;
    std::size_t b_f32_bytes = 0u;
    std::size_t convolved_qkv_f32_bytes = 0u;
    std::size_t gdn_output_f32_bytes = 0u;
    std::size_t total_device_bytes = 0u;
};

/* All ranges are caller-owned device allocations on the adapter's device.
 * The BF16 conversion buffer and cuBLAS workspace are deliberately shared by
 * the five serial linears; stream ordering prevents reuse before consumption.
 * None of the float ranges may overlap one another, the input, or the output. */
struct GdnAdapterScratch {
    std::uint16_t* linear_input_bf16 = nullptr;
    std::size_t linear_input_bf16_bytes = 0u;
    void* blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0u;
    float* projected_qkv = nullptr;
    std::size_t projected_qkv_f32_bytes = 0u;
    float* z = nullptr;
    std::size_t z_f32_bytes = 0u;
    float* a = nullptr;
    std::size_t a_f32_bytes = 0u;
    float* b = nullptr;
    std::size_t b_f32_bytes = 0u;
    float* convolved_qkv = nullptr;
    std::size_t convolved_qkv_f32_bytes = 0u;
    float* gdn_output = nullptr;
    std::size_t gdn_output_f32_bytes = 0u;
};

[[nodiscard]] const char* gdn_adapter_status_string(
    GdnAdapterStatus status) noexcept;

[[nodiscard]] GdnAdapterStatus gdn_adapter_validate_config(
    const GdnAdapterConfig& config) noexcept;

[[nodiscard]] GdnAdapterStatus gdn_adapter_workspace_requirements(
    const GdnAdapterConfig& config,
    GdnAdapterWorkspaceRequirements* requirements) noexcept;

/* Additive checkpoint adapter for one real qwen4_exp GatedDeltaNet layer.
 * load() is the only phase allowed to allocate or synchronize: it validates
 * all nine checkpoint tensors, loads five BF16 matrices through
 * resident_bf16_linear, and converts the four small BF16 tensors to immutable
 * F32 device storage in bounded chunks.  The checkpoint itself is untouched.
 *
 * forward() accepts one generation step for `batch` independent sequences.
 * It only enqueues projection -> gdn_step_cuda -> output projection on the
 * supplied stream.  The caller owns both scratch and GdnDeviceState, whose
 * exact Qwen4Exp batch contract must match the call. */
class GdnLayerAdapter final {
public:
    GdnLayerAdapter();
    ~GdnLayerAdapter();
    GdnLayerAdapter(GdnLayerAdapter&&) noexcept;
    GdnLayerAdapter& operator=(GdnLayerAdapter&&) noexcept;
    GdnLayerAdapter(const GdnLayerAdapter&) = delete;
    GdnLayerAdapter& operator=(const GdnLayerAdapter&) = delete;

    [[nodiscard]] static GdnAdapterStatus load(
        const checkpoint_catalog& catalog,
        const GdnAdapterConfig& config,
        cudaStream_t initialization_stream,
        std::unique_ptr<GdnLayerAdapter>* out,
        std::string* error) noexcept;

    [[nodiscard]] GdnAdapterStatus forward(
        const float* input_f32,
        std::size_t batch,
        GdnDeviceState* state,
        const GdnAdapterScratch& scratch,
        float* output_f32,
        cudaStream_t stream) noexcept;

    /* Project and execute one contiguous causal sequence. The adapter linears
     * consume token_count rows in one call, while the retained GDN state must
     * remain batch=1. This is deliberately separate from forward(), whose rows
     * represent independent sequences. */
    [[nodiscard]] GdnAdapterStatus forward_sequence(
        const float* input_f32,
        std::size_t token_count,
        GdnDeviceState* state,
        const GdnAdapterScratch& scratch,
        float* output_f32,
        cudaStream_t stream) noexcept;

    [[nodiscard]] const GdnAdapterConfig& config() const noexcept;
    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] GdnWeightsView device_weights() const noexcept;

private:
    [[nodiscard]] GdnAdapterStatus forward_rows(
        const float* input_f32,
        std::size_t row_count,
        bool contiguous_sequence,
        GdnDeviceState* state,
        const GdnAdapterScratch& scratch,
        float* output_f32,
        cudaStream_t stream) noexcept;

    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
