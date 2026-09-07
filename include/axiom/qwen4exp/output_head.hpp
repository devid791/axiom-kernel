#ifndef AXIOM_QWEN4EXP_OUTPUT_HEAD_HPP
#define AXIOM_QWEN4EXP_OUTPUT_HEAD_HPP

#include "axiom/qwen4exp/checkpoint.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kOutputHeadHidden = 2560u;
inline constexpr std::size_t kOutputHeadStreams = 4u;
inline constexpr std::size_t kOutputHeadResidual = 10240u;
inline constexpr std::size_t kOutputHeadRank = 320u;
inline constexpr std::size_t kOutputHeadVocab = 248320u;
inline constexpr float kOutputHeadRmsEpsilon = 1.0e-6F;

enum class output_head_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    tensor_not_found,
    tensor_contract_mismatch,
    size_overflow,
    unsupported_device,
    allocation_failure,
    invalid_device_pointer,
    checkpoint_io_error,
    cuda_error,
    linear_error,
    device_rejected_input,
};

enum class output_head_tying : std::uint8_t {
    tied_to_embedding = 0,
    untied_checkpoint_head,
};

enum class output_head_finalization : std::uint8_t {
    global_hyper_connection_mixer = 0,
};

struct output_head_config {
    std::size_t max_tokens = 8u;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct output_head_workspace_requirements {
    std::size_t hyper_input_f32_bytes = 0u;
    std::size_t normalized_f32_bytes = 0u;
    std::size_t low_rank_f32_bytes = 0u;
    std::size_t mix_logits_f32_bytes = 0u;
    std::size_t mixed_hidden_f32_bytes = 0u;
    std::size_t linear_input_bf16_bytes = 0u;
    std::size_t logits_f32_bytes = 0u;
    std::size_t blas_workspace_bytes = 0u;
    std::size_t device_status_bytes = sizeof(std::uint32_t);
};

/* Every pointer is caller-owned device memory and all ranges must be distinct.
 * cudaMalloc alignment is sufficient for every field, including the cuBLAS
 * workspace.  The scratch lifetime extends through completion of the stream. */
struct output_head_scratch {
    float *hyper_input = nullptr;       // [max_tokens, 4 * 2560]
    float *normalized = nullptr;        // [max_tokens, 4 * 2560]
    float *low_rank = nullptr;          // [max_tokens, 320]
    float *mix_logits = nullptr;        // [max_tokens, 4 * 2560]
    float *mixed_hidden = nullptr;      // [max_tokens, 2560]
    std::uint16_t *linear_input_bf16 = nullptr;
    std::size_t linear_input_bf16_bytes = 0u;
    float *logits = nullptr;            // [max_tokens, 248320]
    void *blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0u;
    std::uint32_t *device_status = nullptr;
    std::size_t device_status_bytes = 0u;
};

struct output_head_sampling_config {
    float temperature = 1.0F;
    std::uint64_t seed = 0u;
    std::uint64_t sequence = 0u;
};

[[nodiscard]] const char *output_head_status_string(
        output_head_status status) noexcept;

[[nodiscard]] output_head_status output_head_validate_config(
        const output_head_config &config) noexcept;

[[nodiscard]] output_head_status output_head_get_workspace_requirements(
        const output_head_config &config,
        output_head_workspace_requirements *requirements) noexcept;

/* Allocation-free host oracle for the official model-level
 * Qwen4ExpTextGatedResidual(use_combine=false): grouped zero-centered RMSNorm,
 * SiLU(down/4), sigmoid(up), then the mean of four normalized streams.
 * Linear inputs are rounded to BF16 exactly where the CUDA tensor-core path
 * materializes its BF16 boundaries. */
[[nodiscard]] output_head_status output_head_global_mix_reference(
        const float *hyper_input_f32,
        const std::uint16_t *hc_norm_bf16,
        const std::uint16_t *down_bf16,
        const std::uint16_t *up_bf16,
        std::size_t tokens,
        float *normalized_f32,
        float *low_rank_f32,
        float *mix_logits_f32,
        float *mixed_hidden_f32) noexcept;

/* Resident additive tail for the pinned qwen4_exp checkpoint.  It owns the
 * real BF16 embedding, global hyper-connection mixer, and LM head.  The real
 * checkpoint is untied; if a future admitted catalog omits lm_head.weight,
 * the loader uses the embedding tensor as the tied head contract.
 *
 * load() may allocate and synchronize while staging bounded checkpoint
 * ranges.  Every execution method only enqueues work: no heap allocation and
 * no implicit stream/device synchronization.  One instance is serialized to
 * one in-flight stream; concurrent sessions use separate instances. */
class resident_output_head final {
public:
    resident_output_head();
    ~resident_output_head();
    resident_output_head(resident_output_head &&) noexcept;
    resident_output_head &operator=(resident_output_head &&) noexcept;
    resident_output_head(const resident_output_head &) = delete;
    resident_output_head &operator=(const resident_output_head &) = delete;

    [[nodiscard]] static output_head_status load(
            const checkpoint_catalog &catalog,
            const output_head_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<resident_output_head> *out,
            std::string *error) noexcept;

    /* Decode BF16 token rows to F32 and repeat each embedding into the four
     * residual streams required before decoder layer zero.  Invalid token ids
     * set scratch.device_status and produce a zero row. */
    [[nodiscard]] output_head_status embedding_repeat(
            const std::uint32_t *token_ids,
            std::size_t tokens,
            const output_head_scratch &scratch,
            float *hyper_input_f32,
            cudaStream_t stream = nullptr) noexcept;

    /* Apply the real model-level global mixer after decoder layer 47. */
    [[nodiscard]] output_head_status global_mix(
            const float *hyper_input_f32,
            std::size_t tokens,
            const output_head_scratch &scratch,
            float *mixed_hidden_f32,
            cudaStream_t stream = nullptr) noexcept;

    /* Project already globally-mixed hidden states to full F32 logits. */
    [[nodiscard]] output_head_status logits(
            const float *mixed_hidden_f32,
            std::size_t tokens,
            const output_head_scratch &scratch,
            float *logits_f32,
            cudaStream_t stream = nullptr) noexcept;

    /* Standalone end-to-end tail gate: embedding repeat -> global mixer ->
     * LM head.  Production decoding normally inserts all 48 decoder layers
     * between embedding_repeat() and global_mix(). */
    [[nodiscard]] output_head_status embedding_global_mix_logits(
            const std::uint32_t *token_ids,
            std::size_t tokens,
            const output_head_scratch &scratch,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] output_head_status argmax(
            const float *logits_f32,
            std::size_t tokens,
            std::uint32_t *token_ids,
            float *selected_logits,
            cudaStream_t stream = nullptr) noexcept;

    /* Stateless deterministic categorical sampling over the complete
     * vocabulary.  Identical logits, seed, sequence, and GPU architecture
     * produce the same token; temperature must be finite and positive. */
    [[nodiscard]] output_head_status sample(
            const float *logits_f32,
            std::size_t tokens,
            const output_head_sampling_config &sampling,
            std::uint32_t *token_ids,
            float *selected_logits,
            cudaStream_t stream = nullptr) noexcept;

    /* Explicit synchronization boundary for the asynchronous invalid-token
     * flag.  Execution methods themselves never call it. */
    [[nodiscard]] output_head_status collect_device_status(
            const output_head_scratch &scratch,
            cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] const output_head_config &config() const noexcept;
    [[nodiscard]] output_head_workspace_requirements workspace_requirements() const noexcept;
    [[nodiscard]] output_head_tying tying() const noexcept;
    [[nodiscard]] output_head_finalization finalization() const noexcept;
    [[nodiscard]] std::uint64_t resident_weight_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
