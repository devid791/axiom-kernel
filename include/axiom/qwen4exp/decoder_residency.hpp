#ifndef AXIOM_QWEN4EXP_DECODER_RESIDENCY_HPP
#define AXIOM_QWEN4EXP_DECODER_RESIDENCY_HPP

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/decoder_layer.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::uint64_t kDecoderResidencyGiB = 1024ull * 1024ull * 1024ull;
inline constexpr std::uint64_t kDecoderResidencyHardLimit = 29ull * kDecoderResidencyGiB;
inline constexpr std::uint64_t kDecoderResidencyDefaultMargin = 1ull * kDecoderResidencyGiB;

enum class decoder_residency_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    unsupported_device,
    invalid_checkpoint,
    arithmetic_overflow,
    budget_exceeded,
    allocation_failure,
    layer_load_failure,
    busy,
    invalid_state,
    cuda_error,
};

enum class decoder_residency_mode : std::uint8_t {
    resident_all = 0,
    bounded_window,
};

enum class decoder_residency_state : std::uint8_t {
    nvme = 0,
    loading,
    resident,
    evicting,
};

struct decoder_residency_config {
    std::uint64_t hard_limit_bytes = kDecoderResidencyHardLimit;
    std::uint64_t safety_margin_bytes = kDecoderResidencyDefaultMargin;

    /* Free bytes observed after CUDA context initialization but before any
     * qwen4_exp model allocation.  Zero asks create() to capture the current
     * value.  Supplying the clean baseline makes the hard limit cover the
     * output head as well as this decoder pool. */
    std::uint64_t clean_free_bytes = 0u;

    std::size_t max_context = kDecoderMaxContext;
    std::size_t upload_chunk_bytes = 8u * 1024u * 1024u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
    std::uint64_t moe_ram_capacity_bytes_per_layer =
            64ull * 1024ull * 1024ull;
    std::size_t moe_prefetch_workers_per_layer = 2u;
    std::shared_ptr<expert_slot_arena> shared_expert_arena;
    std::shared_ptr<const expert_checkpoint_catalog> shared_expert_catalog;
    bool enable_ple = true;
    bool moe_owned_direct_pread = false;

    /* Conservative allowance for each layer's expert GPU slots, events,
     * converted scalar tensors and adapter-owned bounded scratch.  Actual
     * cudaMemGetInfo deltas are authoritative during loading. */
    std::uint64_t adapter_allowance_bytes_per_layer =
            40ull * 1024ull * 1024ull;
};

struct decoder_layer_residency_record {
    std::size_t layer_index = 0u;
    decoder_layer_kind kind = decoder_layer_kind::gated_deltanet;
    decoder_residency_state state = decoder_residency_state::nvme;
    std::uint64_t checkpoint_immutable_bytes = 0u;
    std::uint64_t observed_cuda_bytes = 0u;
    std::uint64_t adapter_overhead_bytes = 0u;
    std::uint64_t load_count = 0u;
    std::uint64_t eviction_count = 0u;
    std::uint32_t pins = 0u;
};

struct decoder_residency_metrics {
    decoder_residency_mode mode = decoder_residency_mode::resident_all;
    std::uint64_t cuda_total_bytes = 0u;
    std::uint64_t clean_free_bytes = 0u;
    std::uint64_t external_bytes_at_create = 0u;
    std::uint64_t hard_limit_bytes = 0u;
    std::uint64_t usable_limit_bytes = 0u;
    std::uint64_t checkpoint_immutable_bytes = 0u;
    std::uint64_t predicted_peak_bytes = 0u;
    std::uint64_t current_observed_bytes = 0u;
    std::uint64_t peak_observed_bytes = 0u;
    std::uint64_t reload_count = 0u;
    std::uint64_t eviction_count = 0u;
    std::uint64_t hot_path_reload_count = 0u;
    std::size_t resident_layers = 0u;
    std::size_t resident_capacity = 0u;
    bool all_48_admitted = false;
    bool no_reload_decode_path = false;
    std::array<decoder_layer_residency_record, kDecoderLayerCount> layer{};
};

class decoder_residency_manager;

/* A lease pins an immutable decoder layer while a caller enqueues work.
 * Leases are runtime-enabled only after all 48 layers are resident.  The
 * bounded_window plan is intentionally diagnostic until decoder session state
 * no longer stores decoder_layer::impl*. */
class decoder_layer_lease final {
public:
    decoder_layer_lease() noexcept;
    ~decoder_layer_lease();
    decoder_layer_lease(decoder_layer_lease &&) noexcept;
    decoder_layer_lease &operator=(decoder_layer_lease &&) noexcept;
    decoder_layer_lease(const decoder_layer_lease &) = delete;
    decoder_layer_lease &operator=(const decoder_layer_lease &) = delete;

    [[nodiscard]] decoder_layer *get() const noexcept;
    [[nodiscard]] decoder_layer &operator*() const noexcept;
    [[nodiscard]] decoder_layer *operator->() const noexcept;
    [[nodiscard]] std::size_t layer_index() const noexcept;
    [[nodiscard]] bool valid() const noexcept;
    void reset() noexcept;

private:
    friend class decoder_residency_manager;
    struct impl;
    explicit decoder_layer_lease(std::unique_ptr<impl> state) noexcept;
    std::unique_ptr<impl> impl_;
};

/* Bounded immutable-weight pool for the exact 48-layer qwen4_exp decoder.
 *
 * If the measured resident set fits under hard_limit-safety_margin, load_all()
 * keeps every layer resident and acquire() performs zero reloads in the decode
 * path.  If a future admitted checkpoint does not fit, create() reports a
 * bounded_window admission plan but load_all(), acquire() and create_session
 * integration fail closed.  Evicting a decoder_layer while a persistent
 * decoder_layer_session stores its impl pointer would be unsafe; runtime layer
 * paging therefore remains disabled until that lifetime is explicitly
 * decoupled and gated.  Session KV/GDN state is never paged here.
 *
 * The catalog and manager must outlive all leases and decoder sessions. */
class decoder_residency_manager final {
public:
    decoder_residency_manager();
    ~decoder_residency_manager();
    decoder_residency_manager(decoder_residency_manager &&) noexcept;
    decoder_residency_manager &operator=(decoder_residency_manager &&) noexcept;
    decoder_residency_manager(const decoder_residency_manager &) = delete;
    decoder_residency_manager &operator=(const decoder_residency_manager &) = delete;

    [[nodiscard]] static decoder_residency_status create(
            const checkpoint_catalog &catalog,
            const decoder_residency_config &config,
            cudaStream_t initialization_stream,
            std::unique_ptr<decoder_residency_manager> *out,
            std::string *error = nullptr) noexcept;

    /* Loads and retains all 48 immutable decoder layers.  Every allocation is
     * measured with cudaMemGetInfo and the operation fails closed before the
     * usable 28 GiB default ceiling can be crossed. */
    [[nodiscard]] decoder_residency_status load_all(
            cudaStream_t initialization_stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_residency_status acquire(
            std::size_t layer_index,
            cudaStream_t stream,
            decoder_layer_lease *lease,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_residency_status release_unpinned(
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] decoder_layer *resident_layer(
            std::size_t layer_index) const noexcept;
    [[nodiscard]] std::array<decoder_layer *, kDecoderLayerCount>
            resident_layer_array() const noexcept;
    [[nodiscard]] decoder_residency_metrics metrics() const noexcept;
    [[nodiscard]] decoder_residency_mode mode() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] bool all_48_resident() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
    friend class decoder_layer_lease;
};

[[nodiscard]] const char *decoder_residency_status_string(
        decoder_residency_status status) noexcept;
[[nodiscard]] const char *decoder_residency_mode_string(
        decoder_residency_mode mode) noexcept;

}  // namespace axiom::qwen4exp

#endif
