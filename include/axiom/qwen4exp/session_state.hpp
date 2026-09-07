#ifndef AXIOM_QWEN4EXP_SESSION_STATE_HPP
#define AXIOM_QWEN4EXP_SESSION_STATE_HPP

#include "axiom/qwen4exp/decoder_layer.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::size_t kModelSessionLayerCount = 48u;
inline constexpr std::size_t kModelSessionMaxContext = 262144u;
inline constexpr std::size_t kModelSessionNoFailedLayer =
        static_cast<std::size_t>(-1);

enum class model_session_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_config,
    size_overflow,
    allocation_failure,
    unsupported_device,
    invalid_device_pointer,
    stream_mismatch,
    invalid_state,
    capacity_exceeded,
    transaction_open,
    no_transaction,
    stage_already_complete,
    layer_error,
    rollback_error,
    reset_error,
    poisoned,
    cuda_error,
};

struct model_session_config {
    /* Physical capacity is explicit and bounded by the native 262K contract.
     * The coordinator never silently advertises YaRN or allocates beyond it. */
    std::size_t context_capacity = kModelSessionMaxContext;
};

struct model_session_view {
    std::size_t context_capacity = 0u;
    std::uint64_t committed_tokens = 0u;
    std::uint64_t staged_tokens = 0u;
    std::size_t staged_layers = 0u;
    std::size_t last_failed_layer = kModelSessionNoFailedLayer;
    int device = -1;
    bool transaction_open = false;
    bool poisoned = false;
    bool initialized = false;
};

using model_decoder_layers =
        std::array<decoder_layer *, kModelSessionLayerCount>;

[[nodiscard]] const char *model_session_status_string(
        model_session_status status) noexcept;
[[nodiscard]] model_session_status model_session_validate_config(
        const model_session_config &config) noexcept;

/* Per-model-session coordinator.  It owns exactly one decoder_layer_session
 * created from each of the 48 immutable decoder_layer objects.  Raw GDN, QSA,
 * PLE and MoE state remain private to decoder_layer_session; this class never
 * duplicates their state or math.
 *
 * One model transaction stages one token through all 48 layers.  A layer
 * failure rolls back every opened layer in reverse order before the error is
 * returned.  commit_all() performs a complete cross-layer preflight before
 * entering the decoder-layer no-fail commit phase.  An unexpected commit-time
 * failure poisons and resets the whole model session, so partial state is
 * never exposed as a usable session.
 *
 * The object is externally serialized and bound to the CUDA stream used at
 * creation.  Two instances created from the same immutable layers own wholly
 * independent state and intermediate buffers. */
class model_session_state final {
public:
    model_session_state();
    ~model_session_state();
    model_session_state(model_session_state &&) noexcept;
    model_session_state &operator=(model_session_state &&) noexcept;
    model_session_state(const model_session_state &) = delete;
    model_session_state &operator=(const model_session_state &) = delete;

    [[nodiscard]] static model_session_status create(
            const model_decoder_layers &layers,
            const model_session_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<model_session_state> *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] model_session_status begin(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    /* residual_input/output are [4,2560] F32 device tensors.  full_cos/sin
     * follow decoder_layer_session's public RoPE contract.  Intermediate
     * residuals are held in two bounded per-session device buffers. */
    [[nodiscard]] model_session_status stage_all(
            std::int64_t token_id,
            const float *residual_input_4x2560_f32,
            const float *full_cos_64_f32,
            const float *full_sin_64_f32,
            std::size_t position_count,
            float *residual_output_4x2560_f32,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] model_session_status stage_all_sequence(
            const std::int64_t *token_ids,
            std::size_t token_count,
            const float *residual_input_4x2560_f32,
            const float *full_cos_64_f32,
            const float *full_sin_64_f32,
            std::size_t position_count,
            float *residual_output_4x2560_f32,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] model_session_status validate(
            std::string *error = nullptr) const noexcept;
    [[nodiscard]] model_session_status commit_all(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] model_session_status rollback_all(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;
    [[nodiscard]] model_session_status reset(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] model_session_view view() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
