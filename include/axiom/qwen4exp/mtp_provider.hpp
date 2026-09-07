#ifndef AXIOM_QWEN4EXP_MTP_PROVIDER_HPP
#define AXIOM_QWEN4EXP_MTP_PROVIDER_HPP

#include "axiom/qwen4exp/checkpoint.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kMtpHidden = 2560u;
inline constexpr std::size_t kMtpStreams = 4u;
inline constexpr std::size_t kMtpResidual = 10240u;
inline constexpr std::size_t kMtpRank = 320u;
inline constexpr std::size_t kMtpExperts = 512u;
inline constexpr std::size_t kMtpTopK = 10u;
inline constexpr std::size_t kMtpIntermediate = 640u;
inline constexpr std::size_t kMtpVocab = 248320u;
inline constexpr std::size_t kMtpTensorCount = 31u;
inline constexpr std::size_t kMtpMaxContext = 262144u;

enum class mtp_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    unsupported_device,
    tensor_not_found,
    tensor_contract_mismatch,
    checkpoint_io_error,
    allocation_failure,
    invalid_device_pointer,
    invalid_state,
    capacity_exceeded,
    linear_error,
    qsa_error,
    attention_error,
    router_error,
    expert_paging_error,
    expert_compute_error,
    non_finite,
    cuda_error,
};

struct mtp_provider_config {
    std::size_t max_context = kMtpMaxContext;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct mtp_contract_report {
    std::size_t tensor_count = 0u;
    std::uint64_t tensor_bytes = 0u;
    bool exact_namespace = false;
    bool all_bf16 = false;
    bool single_full_attention_layer = false;
    bool shared_lm_head = false;
};

struct mtp_session_state {
    std::size_t context_capacity = 0u;
    std::size_t committed_tokens = 0u;
    bool transaction_open = false;
    bool staged = false;
    bool poisoned = false;
};

struct mtp_prediction {
    std::uint32_t token_id = 0u;
    float selected_logit = 0.0F;
    std::array<std::uint32_t, kMtpTopK> experts{};
    std::array<float, kMtpTopK> router_weights{};
    std::uint64_t nvme_bytes_read = 0u;
    std::uint32_t expert_cache_hits = 0u;
};

[[nodiscard]] const char *mtp_status_string(mtp_status status) noexcept;
[[nodiscard]] mtp_status mtp_validate_config(
        const mtp_provider_config &config) noexcept;

/* Fail-closed proof for the exact pinned qwen4_exp MTP namespace.  It rejects
 * missing, extra, renamed, non-BF16, or reshaped mtp.* tensors.  The admitted
 * graph is the HF/SGLang one-layer contract: Gemma RMSNorm fusion, one
 * full-attention QSA+MoE decoder layer, final four-stream mixer, shared base
 * lm_head.  It deliberately does not accept the historical Qwen3.8 MTP ABI. */
[[nodiscard]] mtp_status mtp_validate_checkpoint_contract(
        const checkpoint_catalog &catalog,
        mtp_contract_report *report,
        std::string *error = nullptr) noexcept;

class mtp_session;

/* Immutable checkpoint-backed MTP weights.  Routed BF16 experts stay cold in
 * the source safetensors and are materialized into ten bounded GPU slots by
 * each session.  All other MTP tensors and the shared lm_head are resident. */
class mtp_provider final {
public:
    struct impl;

    mtp_provider();
    ~mtp_provider();
    mtp_provider(mtp_provider &&) noexcept;
    mtp_provider &operator=(mtp_provider &&) noexcept;
    mtp_provider(const mtp_provider &) = delete;
    mtp_provider &operator=(const mtp_provider &) = delete;

    [[nodiscard]] static mtp_status load(
            const checkpoint_catalog &catalog,
            const mtp_provider_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<mtp_provider> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] mtp_status create_session(
            std::size_t context_capacity,
            cudaStream_t initialization_stream,
            std::unique_ptr<mtp_session> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] const mtp_provider_config &config() const noexcept;
    [[nodiscard]] mtp_contract_report contract() const noexcept;
    [[nodiscard]] std::uint64_t resident_weight_bytes() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    std::unique_ptr<impl> impl_;
    friend class mtp_session;
};

/* One sequence, one token per transaction.  stage_token() executes the real
 * checkpoint graph and is an explicit control-plane boundary: after routing,
 * it synchronizes once to validate device status and page the selected ten
 * BF16 experts from the pinned checkpoint.  commit() publishes exactly one
 * QSA KV row; rollback() publishes none. */
class mtp_session final {
public:
    struct impl;

    mtp_session();
    ~mtp_session();
    mtp_session(mtp_session &&) noexcept;
    mtp_session &operator=(mtp_session &&) noexcept;
    mtp_session(const mtp_session &) = delete;
    mtp_session &operator=(const mtp_session &) = delete;

    [[nodiscard]] mtp_status stage_token(
            const float *input_embedding_2560_f32,
            const float *target_hidden_4x2560_f32,
            const float *full_cos_64_f32,
            const float *full_sin_64_f32,
            std::size_t position_count,
            float *output_hidden_4x2560_f32,
            mtp_prediction *prediction,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] mtp_status commit(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] mtp_status rollback(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    /* Rewind only the logical KV tail after a rejected speculative suffix.
     * Stale rows remain physically allocated but become invisible and are
     * overwritten by subsequent stages. */
    [[nodiscard]] mtp_status truncate_committed(
            std::size_t committed_tokens,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] mtp_status reset(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] mtp_session_state state() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    std::unique_ptr<impl> impl_;
    friend class mtp_provider;
};

}  // namespace axiom::qwen4exp

#endif
