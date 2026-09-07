#ifndef AXIOM_QWEN4EXP_BF16_LINEAR_HPP
#define AXIOM_QWEN4EXP_BF16_LINEAR_HPP

#include "axiom/qwen4exp/checkpoint.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

enum class bf16_linear_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    tensor_not_found,
    dtype_mismatch,
    shape_mismatch,
    size_overflow,
    unsupported_device,
    invalid_device_pointer,
    checkpoint_io_error,
    allocation_failure,
    cuda_error,
    cublas_error,
};

struct bf16_linear_config {
    std::uint64_t input_features = 0u;
    std::uint64_t output_features = 0u;
    std::size_t max_batch = 1u;

    /* Initialization stages at most this many checkpoint bytes in pinned RAM.
     * The value must be even because the source tensor is BF16. */
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;

    /* cuBLAS receives exactly this caller-owned workspace on every forward.
     * Keeping it fixed makes the execution contract bounded and repeatable. */
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct bf16_linear_workspace_requirements {
    std::size_t input_bf16_bytes = 0u;
    std::size_t blas_workspace_bytes = 0u;
};

/* Read-only process metrics for the per-device execution-context registry.
 * They are intended for qualification and telemetry; querying them never
 * creates a CUDA context or a cuBLAS handle. */
struct bf16_linear_execution_context_metrics {
    std::uint64_t handles_created = 0u;
    std::uint64_t context_acquisitions = 0u;
    std::size_t live_contexts = 0u;
    std::size_t live_handles = 0u;
};

[[nodiscard]] bf16_linear_status bf16_linear_get_execution_context_metrics(
        int device,
        bf16_linear_execution_context_metrics *metrics) noexcept;

/* Every pointer is device memory on the resident linear's device.  Storage
 * must remain alive until the supplied CUDA stream has completed the call.
 * blas_workspace must be at least 256-byte aligned. */
struct bf16_linear_scratch {
    std::uint16_t *input_bf16 = nullptr;
    std::size_t input_bf16_bytes = 0u;
    void *blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0u;
};

[[nodiscard]] const char *bf16_linear_status_string(
        bf16_linear_status status) noexcept;

[[nodiscard]] bf16_linear_status bf16_linear_validate_config(
        const bf16_linear_config &config) noexcept;

[[nodiscard]] bf16_linear_status bf16_linear_get_workspace_requirements(
        const bf16_linear_config &config,
        bf16_linear_workspace_requirements *requirements) noexcept;

/* Allocation-free host oracle.  It rounds F32 input to BF16 before an F32
 * accumulation, matching the tensor-core execution contract. */
[[nodiscard]] bf16_linear_status bf16_linear_reference_f32(
        const bf16_linear_config &config,
        const std::uint16_t *weight_bf16,
        const float *input_f32,
        std::size_t batch,
        float *output_f32) noexcept;

/* Immutable, device-resident row-major BF16 matrix loaded only through the
 * pinned checkpoint_catalog.  The admitted tensor must be rank-2 BF16 with
 * exact shape [output_features, input_features] and exact byte length.
 *
 * load() may allocate and synchronize its initialization stream while it
 * performs bounded pread -> pinned staging -> GPU copies.  forward() does
 * neither: it only enqueues F32->BF16 conversion and a BF16 tensor-core GEMM.
 * All resident linears on one device share one weak-registry-backed execution
 * context and one cuBLAS handle.  forward() serializes handle configuration
 * and GEMM enqueue while preserving caller-stream asynchrony. */
class resident_bf16_linear final {
public:
    resident_bf16_linear();
    ~resident_bf16_linear();
    resident_bf16_linear(resident_bf16_linear &&) noexcept;
    resident_bf16_linear &operator=(resident_bf16_linear &&) noexcept;
    resident_bf16_linear(const resident_bf16_linear &) = delete;
    resident_bf16_linear &operator=(const resident_bf16_linear &) = delete;

    [[nodiscard]] static bf16_linear_status load(
            const checkpoint_catalog &catalog,
            const std::string &tensor_name,
            const bf16_linear_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<resident_bf16_linear> *out,
            std::string *error) noexcept;

    /* F32 caller input is row-major [batch, input_features]; F32 output is
     * row-major [batch, output_features].  A successful return means work was
     * enqueued, not completed.  No implicit stream/device synchronization is
     * performed. */
    [[nodiscard]] bf16_linear_status forward(
            const float *input_f32,
            std::size_t batch,
            const bf16_linear_scratch &scratch,
            float *output_f32,
            cudaStream_t stream) noexcept;

    [[nodiscard]] const bf16_linear_config &config() const noexcept;
    [[nodiscard]] const std::string &tensor_name() const noexcept;
    [[nodiscard]] std::uint64_t weight_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
