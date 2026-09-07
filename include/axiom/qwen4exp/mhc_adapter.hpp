#ifndef AXIOM_QWEN4EXP_MHC_ADAPTER_HPP
#define AXIOM_QWEN4EXP_MHC_ADAPTER_HPP

#include "axiom/qwen4exp/bf16_linear.hpp"
#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/mhc.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

enum class mhc_adapter_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    tensor_contract_mismatch,
    size_overflow,
    allocation_failure,
    invalid_device_pointer,
    cuda_error,
    linear_error,
};

enum class mhc_adapter_site : std::uint8_t {
    attention = 0,
    mlp = 1,
};

struct mhc_adapter_config {
    std::size_t max_batch = 8u;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct mhc_adapter_workspace_requirements {
    std::size_t normalized_f32_bytes = 0u;
    std::size_t low_rank_f32_bytes = 0u;
    std::size_t mix_logits_f32_bytes = 0u;
    std::size_t injection_logits_f32_bytes = 0u;
    std::size_t linear_input_bf16_bytes = 0u;
    std::size_t blas_workspace_bytes = 0u;
};

// Every pointer is caller-owned device memory. Buffers must not overlap.
// One scratch instance is used serially by the three resident BF16 linears.
struct mhc_adapter_scratch {
    float *normalized = nullptr;       // [batch, 4 * 2560]
    float *low_rank = nullptr;         // [batch, 320]
    float *mix_logits = nullptr;       // [batch, 4 * 2560]
    float *injection_logits = nullptr; // [batch, 4]
    std::uint16_t *linear_input_bf16 = nullptr;
    std::size_t linear_input_bf16_bytes = 0u;
    void *blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0u;
};

[[nodiscard]] const char *mhc_adapter_status_string(
        mhc_adapter_status status) noexcept;
[[nodiscard]] mhc_adapter_status mhc_adapter_validate_config(
        const mhc_adapter_config &config) noexcept;
[[nodiscard]] mhc_adapter_status mhc_adapter_get_workspace_requirements(
        const mhc_adapter_config &config,
        mhc_adapter_workspace_requirements *requirements) noexcept;

// Loads one existing checkpoint mHC site without changing its bytes. Norm is
// kept BF16; down/up/injection matrices remain BF16 in resident tensor-core
// linears. Initialization may allocate and synchronize the supplied stream.
class resident_mhc_adapter final {
public:
    resident_mhc_adapter();
    ~resident_mhc_adapter();
    resident_mhc_adapter(resident_mhc_adapter &&) noexcept;
    resident_mhc_adapter &operator=(resident_mhc_adapter &&) noexcept;
    resident_mhc_adapter(const resident_mhc_adapter &) = delete;
    resident_mhc_adapter &operator=(const resident_mhc_adapter &) = delete;

    [[nodiscard]] static mhc_adapter_status load(
            const checkpoint_catalog &catalog,
            std::uint32_t layer,
            mhc_adapter_site site,
            const mhc_adapter_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<resident_mhc_adapter> *out,
            std::string *error) noexcept;

    // Official Qwen4-Exp GatedResidual prepare:
    // grouped RMSNorm -> SiLU(down/4) -> sigmoid(up) -> stream mean,
    // plus 2*sigmoid(block_inject/4). No allocation or synchronization.
    [[nodiscard]] mhc_adapter_status prepare(
            const float *hyper_input,
            std::size_t batch,
            const mhc_adapter_scratch &scratch,
            float *mixed_input,
            float *injection_weights,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] mhc_adapter_status reinject(
            const float *hyper_input,
            const float *block_output,
            const float *injection_weights,
            std::size_t batch,
            float *output,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] std::uint32_t layer() const noexcept;
    [[nodiscard]] mhc_adapter_site site() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
