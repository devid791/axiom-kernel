#ifndef AXIOM_QWEN4EXP_QSA_ADAPTER_HPP
#define AXIOM_QWEN4EXP_QSA_ADAPTER_HPP

#include "axiom/qwen4exp/attention.hpp"
#include "axiom/qwen4exp/bf16_linear.hpp"
#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/qsa.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kQsaAdapterLayerCount = 48u;
inline constexpr std::size_t kQsaAdapterHiddenSize = 2560u;
inline constexpr std::size_t kQsaAdapterQueryProjectionSize = 12288u;
inline constexpr std::size_t kQsaAdapterKvProjectionSize = 512u;
inline constexpr std::size_t kQsaAdapterAttentionOutputSize = 6144u;
inline constexpr std::size_t kQsaAdapterIndexerProjectionSize = 640u;
inline constexpr std::size_t kQsaAdapterIndexerQuerySize = 512u;
inline constexpr std::size_t kQsaAdapterIndexerKeySize = 128u;
inline constexpr std::size_t kQsaAdapterMaxBatch = 256u;
inline constexpr std::size_t kQsaAdapterMaxContext = 262144u;

enum class QsaAdapterStatus : std::uint8_t {
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
    kQsaError,
    kAttentionError,
    kDeviceRejected,
    kCudaError,
};

struct QsaAdapterConfig {
    std::size_t layer_index = 3u;
    std::size_t max_batch = 1u;
    std::size_t max_context = kQsaAdapterMaxContext;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
};

struct QsaAdapterWorkspaceRequirements {
    std::size_t linear_input_bf16_bytes = 0u;
    std::size_t blas_workspace_bytes = 0u;
    std::size_t q_projection_f32_bytes = 0u;
    std::size_t k_projection_f32_bytes = 0u;
    std::size_t v_projection_f32_bytes = 0u;
    std::size_t index_projection_f32_bytes = 0u;
    std::size_t q_projection_bf16_bytes = 0u;
    std::size_t k_projection_bf16_bytes = 0u;
    std::size_t v_projection_bf16_bytes = 0u;
    std::size_t index_query_bf16_bytes = 0u;
    std::size_t index_query_prepared_f32_bytes = 0u;
    std::size_t visible_indices_bytes = 0u;
    std::size_t pooled_index_keys_f32_bytes = 0u;
    std::size_t index_scores_f32_bytes = 0u;
    std::size_t selection_workspace_bytes = 0u;
    std::size_t selected_indices_bytes = 0u;
    std::size_t selected_counts_bytes = 0u;
    std::size_t attention_workspace_bytes = 0u;
    std::size_t attention_output_bf16_bytes = 0u;
    std::size_t attention_output_f32_bytes = 0u;
    std::size_t qsa_device_status_bytes = sizeof(qsa::status);
    std::size_t attention_device_status_bytes = sizeof(attention::status);
    std::size_t total_device_bytes = 0u;
};

/* External, caller-owned state for one sequence and one QSA layer.  The main
 * BF16 K/V cache and the raw BF16 indexer-key cache always share the same
 * committed/staged boundary.  One cache must be externally serialized. */
struct QsaLayerCache {
    attention::cache_state attention_cache{};
    std::uint16_t* index_key_bf16 = nullptr;  // [capacity, 128]
    std::size_t capacity_tokens = 0u;
    std::size_t committed_tokens = 0u;
    std::size_t staged_tokens = 0u;
    bool initialized = false;
    bool transaction_open = false;
};

struct QsaAdapterCacheLengths {
    std::size_t committed = 0u;
    std::size_t staged = 0u;
    std::size_t visible = 0u;
    std::size_t capacity = 0u;
};

/* Every pointer is caller-owned device memory on the adapter's device and all
 * ranges must be disjoint.  The two status words are reset by
 * begin_transaction().  forward() performs no allocation or synchronization;
 * collect_device_status() is the explicit synchronization boundary. */
struct QsaAdapterScratch {
    std::uint16_t* linear_input_bf16 = nullptr;
    std::size_t linear_input_bf16_bytes = 0u;
    void* blas_workspace = nullptr;
    std::size_t blas_workspace_bytes = 0u;

    float* q_projection = nullptr;
    std::size_t q_projection_f32_bytes = 0u;
    float* k_projection = nullptr;
    std::size_t k_projection_f32_bytes = 0u;
    float* v_projection = nullptr;
    std::size_t v_projection_f32_bytes = 0u;
    float* index_projection = nullptr;
    std::size_t index_projection_f32_bytes = 0u;

    std::uint16_t* q_projection_bf16 = nullptr;
    std::size_t q_projection_bf16_bytes = 0u;
    std::uint16_t* k_projection_bf16 = nullptr;
    std::size_t k_projection_bf16_bytes = 0u;
    std::uint16_t* v_projection_bf16 = nullptr;
    std::size_t v_projection_bf16_bytes = 0u;
    std::uint16_t* index_query_bf16 = nullptr;
    std::size_t index_query_bf16_bytes = 0u;
    float* index_query_prepared = nullptr;
    std::size_t index_query_prepared_f32_bytes = 0u;

    std::int32_t* visible_indices = nullptr;
    std::size_t visible_indices_bytes = 0u;
    float* pooled_index_keys = nullptr;
    std::size_t pooled_index_keys_f32_bytes = 0u;
    float* index_scores = nullptr;
    std::size_t index_scores_f32_bytes = 0u;
    void* selection_workspace = nullptr;
    std::size_t selection_workspace_bytes = 0u;
    std::int32_t* selected_indices = nullptr;
    std::size_t selected_indices_bytes = 0u;
    std::uint32_t* selected_counts = nullptr;
    std::size_t selected_counts_bytes = 0u;

    void* attention_workspace = nullptr;
    std::size_t attention_workspace_bytes = 0u;
    std::uint16_t* attention_output_bf16 = nullptr;
    std::size_t attention_output_bf16_bytes = 0u;
    float* attention_output_f32 = nullptr;
    std::size_t attention_output_f32_bytes = 0u;

    qsa::status* qsa_device_status = nullptr;
    attention::status* attention_device_status = nullptr;
};

struct QsaAdapterCollectedStatus {
    qsa::status indexer = qsa::status::ok;
    attention::status attention_core = attention::status::ok;
};

[[nodiscard]] const char* qsa_adapter_status_string(
    QsaAdapterStatus status) noexcept;
[[nodiscard]] QsaAdapterStatus qsa_adapter_validate_config(
    const QsaAdapterConfig& config) noexcept;
[[nodiscard]] QsaAdapterStatus qsa_adapter_workspace_requirements(
    const QsaAdapterConfig& config,
    QsaAdapterWorkspaceRequirements* requirements) noexcept;

[[nodiscard]] QsaAdapterStatus qsa_adapter_cache_initialize(
    QsaLayerCache* cache,
    std::uint16_t* external_key_bf16,
    std::uint16_t* external_value_bf16,
    std::uint16_t* external_index_key_bf16,
    std::size_t capacity_tokens,
    std::size_t initial_committed_tokens = 0u) noexcept;
[[nodiscard]] QsaAdapterStatus qsa_adapter_cache_reset(
    QsaLayerCache* cache) noexcept;
[[nodiscard]] QsaAdapterStatus qsa_adapter_cache_get_lengths(
    const QsaLayerCache* cache,
    QsaAdapterCacheLengths* lengths) noexcept;

/* Additive real-checkpoint adapter for qwen4_exp QSA layers 3,7,...,47.
 * load() owns immutable resident q/k/v/o/indexer linears and the four BF16
 * norm vectors.  No existing Axiom provider or model path is consulted. */
class QsaLayerAdapter final {
public:
    QsaLayerAdapter();
    ~QsaLayerAdapter();
    QsaLayerAdapter(QsaLayerAdapter&&) noexcept;
    QsaLayerAdapter& operator=(QsaLayerAdapter&&) noexcept;
    QsaLayerAdapter(const QsaLayerAdapter&) = delete;
    QsaLayerAdapter& operator=(const QsaLayerAdapter&) = delete;

    [[nodiscard]] static QsaAdapterStatus load(
        const checkpoint_catalog& catalog,
        const QsaAdapterConfig& config,
        cudaStream_t initialization_stream,
        std::unique_ptr<QsaLayerAdapter>* out,
        std::string* error) noexcept;

    /* Opens exactly one speculative/prefill batch transaction and resets both
     * asynchronous status words.  No allocation or synchronization. */
    [[nodiscard]] QsaAdapterStatus begin_transaction(
        QsaLayerCache* cache,
        const QsaAdapterScratch& scratch,
        cudaStream_t stream = nullptr) noexcept;

    /* Executes q/k/v/index projections, QSA selection, sparse attention and
     * o projection for one contiguous sequence batch. hidden_states_f32 is
     * exactly the shared [tokens,2560] mixed hidden produced by
     * attn_hyper_connection; both the indexer and main attention consume it.
     * full_cos/full_sin are [position_count,64]. A transaction accepts one
     * forward() call, before the caller performs mHC reinjection. */
    [[nodiscard]] QsaAdapterStatus forward(
        const float* hidden_states_f32,
        std::size_t token_count,
        const float* full_cos_f32,
        const float* full_sin_f32,
        std::size_t position_count,
        QsaLayerCache* cache,
        const QsaAdapterScratch& scratch,
        float* output_f32,
        cudaStream_t stream = nullptr) noexcept;

    /* Explicit synchronization boundary.  Any non-ok device status
     * automatically rolls back both cache tiers before returning
     * kDeviceRejected. */
    [[nodiscard]] QsaAdapterStatus collect_device_status(
        QsaLayerCache* cache,
        const QsaAdapterScratch& scratch,
        QsaAdapterCollectedStatus* collected,
        cudaStream_t stream = nullptr) noexcept;

    [[nodiscard]] QsaAdapterStatus commit_prefix(
        QsaLayerCache* cache,
        std::size_t accepted_tokens) noexcept;
    [[nodiscard]] QsaAdapterStatus rollback(
        QsaLayerCache* cache) noexcept;

    [[nodiscard]] const QsaAdapterConfig& config() const noexcept;
    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
