#ifndef AXIOM_QWEN4EXP_EXPERT_BRIDGE_HPP
#define AXIOM_QWEN4EXP_EXPERT_BRIDGE_HPP

#include "axiom/qwen4exp/checkpoint.hpp"
#include "axiom/qwen4exp/expert_pager.hpp"
#include "axiom/qwen4exp/moe.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace axiom::qwen4exp {

inline constexpr std::uint32_t kExpertBridgeLayers = 48u;
inline constexpr std::uint32_t kExpertBridgePlanesPerExpert = 12u;
inline constexpr std::uint64_t kExpertBridgeBytesPerExpert = 2764824ull;

/* SHA-256 of the canonical routed-expert descriptor stream.  The stream is
 * ordered by layer/expert/projection/plane and includes every tensor name,
 * shard, dtype, logical shape, exact Safetensors file offset and byte count.
 * It is intentionally separate from the checkpoint payload manifest hash. */
inline constexpr char kExpertLayoutSha256[] =
        "68e996186bea796db29172887f761dd74d37f835edb7400260ec96dc24cd9359";

struct expert_checkpoint_identity {
    std::string repository;
    std::string revision;
    std::string weight_manifest_sha256;
};

expert_checkpoint_identity pinned_expert_checkpoint_identity();

struct expert_tensor_descriptor {
    ExpertKey key;
    std::string name;
    std::string shard;
    std::string file;
    tensor_dtype dtype = tensor_dtype::unknown;
    std::uint32_t rank = 0;
    std::array<std::uint64_t, 5> shape{};
    std::uint64_t file_offset = 0;
    std::uint64_t bytes = 0;
};

class expert_checkpoint_catalog final {
public:
    expert_checkpoint_catalog();
    ~expert_checkpoint_catalog();
    expert_checkpoint_catalog(expert_checkpoint_catalog&&) noexcept;
    expert_checkpoint_catalog& operator=(expert_checkpoint_catalog&&) noexcept;
    expert_checkpoint_catalog(const expert_checkpoint_catalog&) = delete;
    expert_checkpoint_catalog& operator=(const expert_checkpoint_catalog&) = delete;

    static bool open(const std::string& model_root,
                     const expert_checkpoint_identity& identity,
                     std::shared_ptr<const expert_checkpoint_catalog>* out,
                     std::string* error) noexcept;

    const expert_tensor_descriptor* find(std::uint32_t layer,
                                         std::uint32_t expert,
                                         ExpertProjection projection,
                                         ExpertPlane plane) const noexcept;
    const expert_tensor_descriptor* at(std::size_t index) const noexcept;
    std::size_t descriptor_count() const noexcept;
    const std::string& model_root() const noexcept;
    const std::string& layout_sha256() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

struct expert_bridge_options {
    int device = 0;
    std::uint32_t max_slots = kMoeTopK;
    std::uint64_t ram_capacity_bytes = 64ull * 1024ull * 1024ull;
    std::size_t prefetch_workers = 2u;
    std::shared_ptr<class expert_slot_arena> shared_arena;
    bool owned_direct_pread = false;
};

struct expert_slot_arena_metrics {
    int device = -1;
    std::uint32_t bounded_slots = 0u;
    std::uint32_t attached_bridges = 0u;
    std::uint32_t active_leases = 0u;
    std::uint64_t capacity_bytes = 0u;
    std::uint64_t ordered_handoffs = 0u;
    std::uint64_t stream_waits = 0u;
    std::uint32_t physical_slots = 0u;
    bool persistent_cache = false;
    bool poisoned = false;
    std::uint64_t cache_hits = 0u;
    std::uint64_t cache_misses = 0u;
    std::uint64_t upload_bytes = 0u;  // Expert planes only; excludes slot indices.
    std::uint64_t evictions = 0u;
    std::uint64_t host_cold_capacity_bytes = 0u;
    // Includes in-flight promotion/demotion payloads, not just reusable entries.
    std::uint64_t host_cold_current_bytes = 0u;
    std::uint64_t host_cold_peak_bytes = 0u;
    std::uint64_t host_cold_hits = 0u;
    std::uint64_t host_cold_misses = 0u;
    std::uint64_t host_demote_bytes = 0u;  // D2H bytes; input scales stay on CPU.
};

/* Serialized SM120 slot planes. The original create overload is transient;
 * the physical_slots overload explicitly enables an immutable expert LRU. */
class expert_slot_arena final {
public:
    ~expert_slot_arena();
    expert_slot_arena(const expert_slot_arena&) = delete;
    expert_slot_arena& operator=(const expert_slot_arena&) = delete;

    static bool create(int device,
                       std::uint32_t max_slots,
                       std::shared_ptr<expert_slot_arena>* out,
                       std::string* error) noexcept;

    static bool create(int device,
                       std::uint32_t max_active_slots,
                       std::uint32_t physical_slots,
                       std::shared_ptr<expert_slot_arena>* out,
                       std::string* error) noexcept;

    // Nonzero enables exclusive GPU/RAM caching with immutable NVMe backing.
    // The arena-wide budget must fit at least one complete expert payload.
    static bool create(int device,
                       std::uint32_t max_active_slots,
                       std::uint32_t physical_slots,
                       std::uint64_t host_cold_cache_bytes,
                       std::shared_ptr<expert_slot_arena>* out,
                       std::string* error) noexcept;

    expert_slot_arena_metrics metrics() const noexcept;

private:
    friend class expert_bridge;
    struct impl;
    expert_slot_arena();
    std::shared_ptr<impl> impl_;
};

struct expert_bridge_metrics {
    std::uint32_t layer = 0u;
    std::uint32_t bounded_slots = 0u;
    std::uint64_t gpu_capacity_bytes = 0u;
    std::uint64_t gpu_owned_bytes = 0u;
    std::uint64_t gpu_shared_bytes = 0u;
    bool external_arena = false;
    bool generation_active = false;
};

class expert_bridge final {
private:
    struct impl;

public:
    class prepared_generation final {
    public:
        prepared_generation() noexcept;
        ~prepared_generation();
        prepared_generation(prepared_generation&&) noexcept;
        prepared_generation& operator=(prepared_generation&&) noexcept;
        prepared_generation(const prepared_generation&) = delete;
        prepared_generation& operator=(const prepared_generation&) = delete;

        bool valid() const noexcept;
        bool committed() const noexcept;
        std::uint64_t generation_id() const noexcept;
        std::uint32_t slot_count() const noexcept;
        // Pass this, not slot_count()/top_k, as the kernel resident_slots bound.
        std::uint32_t physical_slot_capacity() const noexcept;
        const std::uint32_t* selected_experts() const noexcept;
        const std::uint32_t* slot_indices_host() const noexcept;
        const std::uint32_t* slot_indices_device() const noexcept;
        const moe_slot_planes& planes() const noexcept;
        float input_scale(ExpertProjection projection,
                          std::uint32_t slot) const noexcept;

        bool wait(std::string* error) noexcept;
        /* Records the last arena consumer on stream and releases only the
         * global slot reservation.  Pager leases and transaction state stay
         * alive until commit()/rollback(); a later bridge waits on the arena
         * event before overwriting the shared slots. */
        bool handoff_after_consumers(cudaStream_t stream,
                                     std::string* error) noexcept;
        bool commit(std::string* error) noexcept;
        bool rollback(std::string* error) noexcept;

    private:
        friend class expert_bridge;
        struct state;
        explicit prepared_generation(std::unique_ptr<state> state) noexcept;
        std::unique_ptr<state> state_;
    };

    expert_bridge();
    ~expert_bridge();
    expert_bridge(expert_bridge&&) noexcept;
    expert_bridge& operator=(expert_bridge&&) noexcept;
    expert_bridge(const expert_bridge&) = delete;
    expert_bridge& operator=(const expert_bridge&) = delete;

    static bool open(std::shared_ptr<const expert_checkpoint_catalog> catalog,
                     std::uint32_t layer,
                     const expert_bridge_options& options,
                     std::unique_ptr<expert_bridge>* out,
                     std::string* error) noexcept;

    bool prepare(const std::uint32_t* selected_experts,
                 std::uint32_t count,
                 cudaStream_t stream,
                 prepared_generation* out,
                 std::string* error) noexcept;

    std::uint32_t layer() const noexcept;
    std::uint32_t max_slots() const noexcept;
    std::uint32_t physical_slot_capacity() const noexcept;
    std::uint64_t gpu_capacity_bytes() const noexcept;
    expert_bridge_metrics metrics() const noexcept;
    ExpertPagerMetrics pager_metrics() const noexcept;

private:
    std::shared_ptr<impl> impl_;
};

}  // namespace axiom::qwen4exp

#endif
