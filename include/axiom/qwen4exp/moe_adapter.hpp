#ifndef AXIOM_QWEN4EXP_MOE_ADAPTER_HPP
#define AXIOM_QWEN4EXP_MOE_ADAPTER_HPP

#include "axiom/qwen4exp/expert_pager.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

class expert_slot_arena;
class expert_checkpoint_catalog;
class checkpoint_catalog;

enum class moe_adapter_status : std::uint32_t {
    ok = 0u,
    invalid_argument,
    unsupported_config,
    unsupported_device,
    invalid_device_pointer,
    checkpoint_error,
    allocation_failure,
    router_error,
    pager_error,
    non_finite,
    busy,
    cuda_error,
    kernel_error,
    transaction_error,
};

struct moe_adapter_options {
    int device = 0;
    std::uint32_t layer = 0u;
    std::uint64_t ram_capacity_bytes = 64ull * 1024ull * 1024ull;
    std::size_t prefetch_workers = 2u;
    std::size_t blas_workspace_bytes = 4u * 1024u * 1024u;
    std::shared_ptr<expert_slot_arena> shared_expert_arena;
    /* Optional immutable catalogs owned by the complete model.  Supplying
     * them prevents every decoder layer from reparsing 206 Safetensors
     * shards and rebuilding the 294,912-entry expert index. */
    const checkpoint_catalog *shared_checkpoint_catalog = nullptr;
    std::shared_ptr<const expert_checkpoint_catalog> shared_expert_catalog;
    bool owned_direct_pread = false;
};

struct moe_adapter_metrics {
    std::uint32_t layer = 0u;
    std::uint32_t bounded_slots = 0u;
    std::uint64_t resident_gpu_bytes = 0u;
    std::uint64_t expert_gpu_owned_bytes = 0u;
    std::uint64_t expert_gpu_shared_bytes = 0u;
    std::uint64_t pager_requests = 0u;
    std::uint64_t pager_ram_hits = 0u;
    std::uint64_t pager_ram_misses = 0u;
    std::uint64_t pager_nvme_reads = 0u;
    std::uint64_t pager_nvme_bytes = 0u;
    std::uint64_t pager_active_generations = 0u;
    bool route_in_flight = false;
    bool generation_active = false;
};

[[nodiscard]] const char *moe_adapter_status_string(
        moe_adapter_status status) noexcept;

/* Real qwen4_exp routed-MoE layer adapter.
 *
 * The adapter composes the pinned BF16 router, official top-10/512 routing,
 * bounded expert_bridge/pager slots, the fused NVFP4 routed experts and the
 * BF16 shared expert.  It is deliberately single-generation: one instance
 * owns ten bounded GPU slots and rejects overlap instead of silently reusing
 * them.
 *
 * enqueue_route() and prepared_generation::forward() allocate no memory and
 * perform no host/device synchronization.  prepare_route() is the explicit
 * control-plane boundary: it waits for the ten selected expert ids and lets
 * the pager materialize their immutable NVMe ranges.  commit()/rollback()
 * are transaction boundaries and wait before releasing those slots. */
class routed_moe_layer_adapter final {
private:
    struct impl;

public:
    class prepared_generation final {
    public:
        prepared_generation() noexcept;
        ~prepared_generation();
        prepared_generation(prepared_generation &&) noexcept;
        prepared_generation &operator=(prepared_generation &&) noexcept;
        prepared_generation(const prepared_generation &) = delete;
        prepared_generation &operator=(const prepared_generation &) = delete;

        [[nodiscard]] bool valid() const noexcept;
        [[nodiscard]] bool committed() const noexcept;
        [[nodiscard]] bool forward_enqueued() const noexcept;
        [[nodiscard]] std::uint64_t generation_id() const noexcept;
        [[nodiscard]] const std::uint32_t *selected_experts() const noexcept;
        [[nodiscard]] const float *router_weights() const noexcept;
        [[nodiscard]] std::uint32_t selected_count() const noexcept;
        [[nodiscard]] moe_adapter_status device_status() const noexcept;

        /* Enqueues routed + shared expert execution and the fail-closed sum.
         * input/output are F32 vectors of exactly kMoeHidden elements and
         * must be distinct device ranges on the configured GPU. */
        [[nodiscard]] moe_adapter_status forward(
                const float *input,
                float *output,
                cudaStream_t stream,
                std::string *error = nullptr) noexcept;

        /* forward() performs a stream-ordered arena handoff after its last
         * consumer.  The transaction remains valid for the model-wide
         * commit/rollback boundary while another layer may safely reuse the
         * ten shared physical slots. */
        [[nodiscard]] bool arena_handed_off() const noexcept;

        /* A successful commit proves that all queued CUDA work completed and
         * that the device-side semantic status stayed green. */
        [[nodiscard]] bool commit(std::string *error = nullptr) noexcept;
        [[nodiscard]] bool rollback(std::string *error = nullptr) noexcept;

    private:
        friend class routed_moe_layer_adapter;
        struct state;
        explicit prepared_generation(std::unique_ptr<state> state) noexcept;
        std::unique_ptr<state> state_;
    };

    routed_moe_layer_adapter();
    ~routed_moe_layer_adapter();
    routed_moe_layer_adapter(routed_moe_layer_adapter &&) noexcept;
    routed_moe_layer_adapter &operator=(routed_moe_layer_adapter &&) noexcept;
    routed_moe_layer_adapter(const routed_moe_layer_adapter &) = delete;
    routed_moe_layer_adapter &operator=(const routed_moe_layer_adapter &) = delete;

    /* load() is the only initialization path.  It validates and loads the
     * exact pinned layer router/shared tensors and opens the real expert
     * catalog.  Allocation and synchronization are permitted only here and
     * at the explicit transaction boundaries documented above. */
    [[nodiscard]] static moe_adapter_status load(
            const std::string &model_root,
            const moe_adapter_options &options,
            cudaStream_t initialization_stream,
            std::unique_ptr<routed_moe_layer_adapter> *out,
            std::string *error = nullptr) noexcept;

    /* Stage one: resident BF16 router and deterministic top-10 on CUDA.
     * The result is copied into preallocated pinned memory and guarded by an
     * event; the call itself neither allocates nor synchronizes. */
    [[nodiscard]] moe_adapter_status enqueue_route(
            const float *input,
            cudaStream_t stream,
            std::string *error = nullptr) noexcept;

    /* Stage two: explicit route/pager boundary.  The stream must be the same
     * one used by enqueue_route(); the prepared generation subsequently owns
     * the ten slots until commit or rollback. */
    [[nodiscard]] moe_adapter_status prepare_route(
            cudaStream_t stream,
            prepared_generation *out,
            std::string *error = nullptr) noexcept;

    [[nodiscard]] std::uint32_t layer() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] moe_adapter_metrics metrics() const noexcept;
    /* Thread-safe, fixed-size bridge/pager snapshot; no payload ownership,
     * allocation or CUDA calls. Safe during routing with a stable adapter
     * lifetime (no concurrent move/destruction). Empty if no bridge exists. */
    [[nodiscard]] ExpertPagerMetrics pager_metrics() const noexcept;

private:
    std::shared_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
