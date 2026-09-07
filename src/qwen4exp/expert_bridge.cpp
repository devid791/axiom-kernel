#include "axiom/qwen4exp/expert_bridge.hpp"

#include "axiom/qwen4exp_admission.hpp"
#include "axiom/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>
#include <vector>

namespace axiom::qwen4exp {
namespace {

constexpr std::uint32_t kProjectionCount = 3u;
constexpr std::uint32_t kPlaneCount = 4u;
constexpr std::array<std::size_t, kPlaneCount> kColdPlaneBytes{{819200u, 4u, 102400u, 4u}};
constexpr std::array<std::size_t, kPlaneCount> kColdPlaneOffsets{{0u, 819200u, 819204u, 921604u}};
constexpr std::size_t kColdProjectionBytes = 921608u;
static_assert(kColdProjectionBytes * kProjectionCount == kExpertBridgeBytesPerExpert);

void set_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = "qwen4_exp expert bridge: " + message;
    }
}

bool fail(std::string* error, const std::string& message) {
    set_error(error, message);
    return false;
}

std::string cuda_error(const char* operation, cudaError_t status) {
    return std::string(operation) + ": " + cudaGetErrorString(status);
}

bool checked_multiply(std::uint64_t left,
                      std::uint64_t right,
                      std::uint64_t* result) noexcept {
    if (result == nullptr ||
        (right != 0u && left > std::numeric_limits<std::uint64_t>::max() / right)) {
        return false;
    }
    *result = left * right;
    return true;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t* result) noexcept {
    if (result == nullptr || left > std::numeric_limits<std::uint64_t>::max() - right) {
        return false;
    }
    *result = left + right;
    return true;
}

std::size_t descriptor_index(std::uint32_t layer,
                             std::uint32_t expert,
                             ExpertProjection projection,
                             ExpertPlane plane) noexcept {
    if (layer >= kExpertBridgeLayers || expert >= kMoeExperts ||
        static_cast<std::uint32_t>(projection) >= kProjectionCount ||
        static_cast<std::uint32_t>(plane) >= kPlaneCount) {
        return std::numeric_limits<std::size_t>::max();
    }
    return (((static_cast<std::size_t>(layer) * kMoeExperts + expert) *
             kProjectionCount + static_cast<std::uint32_t>(projection)) *
            kPlaneCount + static_cast<std::uint32_t>(plane));
}

const char* projection_name(ExpertProjection projection) {
    switch (projection) {
        case ExpertProjection::gate: return "gate_proj";
        case ExpertProjection::up: return "up_proj";
        case ExpertProjection::down: return "down_proj";
    }
    throw std::runtime_error("unknown routed-expert projection");
}

const char* plane_name(ExpertPlane plane) {
    switch (plane) {
        case ExpertPlane::weight: return "weight";
        case ExpertPlane::input_scale: return "input_scale";
        case ExpertPlane::weight_scale: return "weight_scale";
        case ExpertPlane::weight_scale_2: return "weight_scale_2";
    }
    throw std::runtime_error("unknown routed-expert plane");
}

std::string tensor_name(std::uint32_t layer,
                        std::uint32_t expert,
                        ExpertProjection projection,
                        ExpertPlane plane) {
    return "model.language_model.layers." + std::to_string(layer) +
           ".mlp.experts." + std::to_string(expert) + "." +
           projection_name(projection) + "." + plane_name(plane);
}

std::string expected_shard(std::uint32_t layer, std::uint32_t expert) {
    const std::uint32_t first = (expert / 128u) * 128u;
    const std::uint32_t last = first + 127u;
    std::array<char, 96> buffer{};
    const int written = std::snprintf(buffer.data(), buffer.size(),
                                      "layer-%05u-experts-%04u-%04u.safetensors",
                                      layer, first, last);
    if (written <= 0 || static_cast<std::size_t>(written) >= buffer.size()) {
        throw std::runtime_error("expert shard name overflow");
    }
    return std::string(buffer.data(), static_cast<std::size_t>(written));
}

std::string join_path(const std::string& root, const std::string& name) {
    return !root.empty() && root.back() == '/' ? root + name : root + "/" + name;
}

bool exact_descriptor(const tensor_span& span,
                      ExpertProjection projection,
                      ExpertPlane plane) noexcept {
    const bool down = projection == ExpertProjection::down;
    if (plane == ExpertPlane::weight) {
        return span.dtype == tensor_dtype::u8 && span.rank == 2u &&
               span.shape[0] == (down ? 2560u : 640u) &&
               span.shape[1] == (down ? 320u : 1280u) &&
               span.bytes == 819200u;
    }
    if (plane == ExpertPlane::weight_scale) {
        return span.dtype == tensor_dtype::f8_e4m3 && span.rank == 2u &&
               span.shape[0] == (down ? 2560u : 640u) &&
               span.shape[1] == (down ? 40u : 160u) &&
               span.bytes == 102400u;
    }
    if (plane == ExpertPlane::input_scale ||
        plane == ExpertPlane::weight_scale_2) {
        return span.dtype == tensor_dtype::f32 && span.rank == 0u && span.bytes == 4u;
    }
    return false;
}

void hash_text(crypto::sha256* hash, const std::string& text) noexcept {
    hash->update(text.data(), text.size());
}

void hash_descriptor(crypto::sha256* hash,
                     const expert_tensor_descriptor& descriptor) {
    std::string canonical;
    canonical.reserve(descriptor.name.size() + descriptor.shard.size() + 128u);
    canonical += descriptor.name;
    canonical.push_back('\t');
    canonical += descriptor.shard;
    canonical.push_back('\t');
    canonical += std::to_string(static_cast<unsigned>(descriptor.dtype));
    canonical.push_back('\t');
    canonical += std::to_string(descriptor.rank);
    for (std::uint32_t index = 0; index < descriptor.rank; ++index) {
        canonical.push_back(',');
        canonical += std::to_string(descriptor.shape[index]);
    }
    canonical.push_back('\t');
    canonical += std::to_string(descriptor.file_offset);
    canonical.push_back('\t');
    canonical += std::to_string(descriptor.bytes);
    canonical.push_back('\n');
    hash_text(hash, canonical);
}

bool valid_positive_scalar(const ExpertByteView& view, float* value) noexcept {
    if (!view.valid() || view.size() != sizeof(float) || value == nullptr) {
        return false;
    }
    std::memcpy(value, view.data(), sizeof(*value));
    return std::isfinite(*value) && *value > 0.0f;
}

bool valid_e4m3_scale_plane(const ExpertByteView& view) noexcept {
    if (!view.valid() || view.size() == 0u) {
        return false;
    }
    for (std::size_t index = 0; index < view.size(); ++index) {
        const std::uint8_t encoded = view.data()[index];
        if ((encoded & 0x80u) != 0u || encoded == 0u || encoded == 0x7fu) {
            return false;
        }
    }
    return true;
}

std::size_t projection_index(ExpertProjection projection) noexcept {
    return static_cast<std::size_t>(projection);
}

}  // namespace

expert_checkpoint_identity pinned_expert_checkpoint_identity() {
    return expert_checkpoint_identity{kRepository, kRevision, kWeightManifestSha256};
}

struct expert_checkpoint_catalog::impl {
    std::string root;
    std::string layout_sha;
    std::vector<expert_tensor_descriptor> descriptors;
};

expert_checkpoint_catalog::expert_checkpoint_catalog() : impl_(std::make_unique<impl>()) {}
expert_checkpoint_catalog::~expert_checkpoint_catalog() = default;
expert_checkpoint_catalog::expert_checkpoint_catalog(expert_checkpoint_catalog&&) noexcept = default;
expert_checkpoint_catalog& expert_checkpoint_catalog::operator=(
        expert_checkpoint_catalog&&) noexcept = default;

bool expert_checkpoint_catalog::open(
        const std::string& model_root,
        const expert_checkpoint_identity& identity,
        std::shared_ptr<const expert_checkpoint_catalog>* out,
        std::string* error) noexcept {
    if (out == nullptr) {
        return fail(error, "catalog output is null");
    }
    out->reset();
    if (identity.repository != kRepository || identity.revision != kRevision ||
        identity.weight_manifest_sha256 != kWeightManifestSha256) {
        return fail(error, "repository, revision or payload fingerprint differs from the pinned contract");
    }
    try {
        std::unique_ptr<checkpoint_catalog> source;
        std::string catalog_error;
        if (!checkpoint_catalog::open(model_root, &source, &catalog_error)) {
            return fail(error, catalog_error);
        }

        auto result = std::make_shared<expert_checkpoint_catalog>();
        result->impl_->root = model_root;
        constexpr std::size_t expected_count =
                static_cast<std::size_t>(kExpertBridgeLayers) * kMoeExperts *
                kProjectionCount * kPlaneCount;
        result->impl_->descriptors.reserve(expected_count);
        crypto::sha256 layout_hash;

        for (std::uint32_t layer = 0; layer < kExpertBridgeLayers; ++layer) {
            for (std::uint32_t expert = 0; expert < kMoeExperts; ++expert) {
                const std::string shard = expected_shard(layer, expert);
                const std::string file = join_path(model_root, shard);
                struct stat status {};
                if (::lstat(file.c_str(), &status) != 0 || S_ISLNK(status.st_mode) ||
                    !S_ISREG(status.st_mode) || status.st_size < 0) {
                    return fail(error, "expert shard is missing, non-regular or a symlink: " + shard);
                }
                const std::uint64_t file_bytes = static_cast<std::uint64_t>(status.st_size);

                for (std::uint32_t projection_value = 0;
                     projection_value < kProjectionCount; ++projection_value) {
                    const auto projection = static_cast<ExpertProjection>(projection_value);
                    for (std::uint32_t plane_value = 0;
                         plane_value < kPlaneCount; ++plane_value) {
                        const auto plane = static_cast<ExpertPlane>(plane_value);
                        const std::string name = tensor_name(layer, expert, projection, plane);
                        const tensor_span* span = source->find(name);
                        if (span == nullptr) {
                            return fail(error, "missing routed-expert tensor: " + name);
                        }
                        if (span->shard != shard || !exact_descriptor(*span, projection, plane)) {
                            return fail(error, "routed-expert tensor descriptor mismatch: " + name);
                        }
                        std::uint64_t end = 0;
                        if (!checked_add(span->file_offset, span->bytes, &end) ||
                            end > file_bytes) {
                            return fail(error, "routed-expert tensor lies outside its immutable shard: " + name);
                        }
                        expert_tensor_descriptor descriptor;
                        descriptor.key = ExpertKey{kWeightManifestSha256, layer, expert,
                                                   projection, plane};
                        descriptor.name = name;
                        descriptor.shard = shard;
                        descriptor.file = file;
                        descriptor.dtype = span->dtype;
                        descriptor.rank = span->rank;
                        descriptor.shape = span->shape;
                        descriptor.file_offset = span->file_offset;
                        descriptor.bytes = span->bytes;
                        hash_descriptor(&layout_hash, descriptor);
                        result->impl_->descriptors.push_back(std::move(descriptor));
                    }
                }
            }
        }

        std::size_t routed_count = 0u;
        constexpr char routed_prefix[] = "model.language_model.layers.";
        for (std::size_t index = 0; index < source->tensor_count(); ++index) {
            const tensor_span* span = source->at(index);
            if (span != nullptr && span->name.compare(
                                           0u, sizeof(routed_prefix) - 1u,
                                           routed_prefix) == 0 &&
                span->name.find(".mlp.experts.") != std::string::npos) {
                ++routed_count;
            }
        }
        if (result->impl_->descriptors.size() != expected_count ||
            routed_count != expected_count) {
            return fail(error, "routed-expert catalog cardinality mismatch");
        }
        result->impl_->layout_sha = crypto::sha256_hex(layout_hash.finish());
        if (result->impl_->layout_sha != kExpertLayoutSha256) {
            return fail(error, "routed-expert layout fingerprint differs from the pinned checkpoint");
        }
        *out = std::move(result);
        return true;
    } catch (const std::exception& exception) {
        return fail(error, std::string("catalog exception: ") + exception.what());
    } catch (...) {
        return fail(error, "catalog exception");
    }
}

const expert_tensor_descriptor* expert_checkpoint_catalog::find(
        std::uint32_t layer,
        std::uint32_t expert,
        ExpertProjection projection,
        ExpertPlane plane) const noexcept {
    const std::size_t index = descriptor_index(layer, expert, projection, plane);
    if (!impl_ || index == std::numeric_limits<std::size_t>::max() ||
        index >= impl_->descriptors.size()) {
        return nullptr;
    }
    const expert_tensor_descriptor& descriptor = impl_->descriptors[index];
    if (descriptor.key.layer != layer || descriptor.key.expert != expert ||
        descriptor.key.projection != projection || descriptor.key.plane != plane) {
        return nullptr;
    }
    return &descriptor;
}

const expert_tensor_descriptor* expert_checkpoint_catalog::at(std::size_t index) const noexcept {
    return impl_ && index < impl_->descriptors.size() ? &impl_->descriptors[index] : nullptr;
}

std::size_t expert_checkpoint_catalog::descriptor_count() const noexcept {
    return impl_ ? impl_->descriptors.size() : 0u;
}

const std::string& expert_checkpoint_catalog::model_root() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->root : empty;
}

const std::string& expert_checkpoint_catalog::layout_sha256() const noexcept {
    static const std::string empty;
    return impl_ ? impl_->layout_sha : empty;
}

struct expert_slot_arena::impl {
    struct cold_entry {
        ExpertKey key;
        std::unique_ptr<std::uint8_t[]> bytes;
        std::uint64_t last_use = 0u;
        bool valid = false;
    };
    struct cache_entry {
        // A representative gate/weight key includes fingerprint, layer, expert.
        ExpertKey key;
        std::array<float, kProjectionCount> input_scales{};
        std::uint64_t last_use = 0u;
        bool valid = false;  // Complete upload enqueued; reuse_event orders readers.
    };
    int device = -1;
    std::uint32_t max_slots = 0u;
    std::uint32_t physical_slots = 0u;
    bool persistent_cache = true;
    bool poisoned = false;
    std::vector<cache_entry> cache;
    std::uint64_t clock = 0u;
    std::uint64_t cache_hits = 0u;
    std::uint64_t cache_misses = 0u;
    std::uint64_t upload_bytes = 0u;
    std::uint64_t evictions = 0u;
    std::uint64_t host_cold_capacity_bytes = 0u;
    std::uint64_t host_cold_current_bytes = 0u;
    std::uint64_t host_cold_peak_bytes = 0u;
    std::uint64_t host_cold_hits = 0u;
    std::uint64_t host_cold_misses = 0u;
    std::uint64_t host_demote_bytes = 0u;
    // Includes nonresident promotion sources until their completion boundary.
    // A second shared owner is a lease and prevents eviction/reclamation.
    std::vector<std::shared_ptr<cold_entry>> cold;
    std::uint8_t* gate_weight = nullptr;
    std::uint8_t* gate_scale = nullptr;
    float* gate_global = nullptr;
    std::uint8_t* up_weight = nullptr;
    std::uint8_t* up_scale = nullptr;
    float* up_global = nullptr;
    std::uint8_t* down_weight = nullptr;
    std::uint8_t* down_scale = nullptr;
    float* down_global = nullptr;
    std::uint32_t* slot_indices = nullptr;
    std::uint64_t gpu_bytes = 0u;
    cudaEvent_t reuse_event = nullptr;
    bool reuse_event_recorded = false;
    std::uint64_t ordered_handoffs = 0u;
    std::uint64_t stream_waits = 0u;
    mutable std::mutex mutex;
    const void* active_owner = nullptr;
    std::uint32_t attached_bridges = 0u;

    void* device_plane(std::uint32_t slot, std::uint32_t projection,
                       std::uint32_t plane) noexcept {
        if (plane == 0u) {
            auto* base = projection == 0u ? gate_weight : projection == 1u ? up_weight : down_weight;
            return base + static_cast<std::size_t>(slot) * 819200u;
        }
        if (plane == 2u) {
            auto* base = projection == 0u ? gate_scale : projection == 1u ? up_scale : down_scale;
            return base + static_cast<std::size_t>(slot) * 102400u;
        }
        auto* base = projection == 0u ? gate_global : projection == 1u ? up_global : down_global;
        return base + slot;
    }

    // Caller holds mutex. Payload bytes remain charged while ANY lease exists.
    void reap_cold() noexcept {
        for (auto it = cold.begin(); it != cold.end();) {
            if (!(*it)->valid && it->use_count() == 1) {
                it = cold.erase(it);
                host_cold_current_bytes -= kExpertBridgeBytesPerExpert;
            } else ++it;
        }
    }

    bool make_cold_room() noexcept {
        reap_cold();
        while (host_cold_capacity_bytes - host_cold_current_bytes < kExpertBridgeBytesPerExpert) {
            auto victim = cold.end();
            for (auto it = cold.begin(); it != cold.end(); ++it) {
                if (it->use_count() == 1 &&
                    (victim == cold.end() || (*it)->last_use < (*victim)->last_use)) victim = it;
            }
            if (victim == cold.end()) return false;
            cold.erase(victim);  // Immutable checkpoint is already the NVMe tier.
            host_cold_current_bytes -= kExpertBridgeBytesPerExpert;
        }
        return true;
    }

    ~impl() {
        if (device >= 0 && cudaSetDevice(device) == cudaSuccess) {
            (void)cudaDeviceSynchronize();
            if (reuse_event != nullptr) (void)cudaEventDestroy(reuse_event);
            (void)cudaFree(slot_indices);
            (void)cudaFree(down_global);
            (void)cudaFree(down_scale);
            (void)cudaFree(down_weight);
            (void)cudaFree(up_global);
            (void)cudaFree(up_scale);
            (void)cudaFree(up_weight);
            (void)cudaFree(gate_global);
            (void)cudaFree(gate_scale);
            (void)cudaFree(gate_weight);
        }
    }
};

expert_slot_arena::expert_slot_arena() = default;
expert_slot_arena::~expert_slot_arena() = default;

bool expert_slot_arena::create(int device,
                               std::uint32_t max_slots,
                               std::shared_ptr<expert_slot_arena>* out,
                               std::string* error) noexcept {
    if (!create(device, max_slots, max_slots, out, error)) return false;
    (*out)->impl_->persistent_cache = false;
    return true;
}

bool expert_slot_arena::create(int device,
                               std::uint32_t max_slots,
                               std::uint32_t physical_slots,
                               std::shared_ptr<expert_slot_arena>* out,
                               std::string* error) noexcept {
    return create(device, max_slots, physical_slots, 0u, out, error);
}

bool expert_slot_arena::create(int device,
                               std::uint32_t max_slots,
                               std::uint32_t physical_slots,
                               std::uint64_t host_cold_cache_bytes,
                               std::shared_ptr<expert_slot_arena>* out,
                               std::string* error) noexcept {
    if (out == nullptr) {
        return fail(error, "expert-slot arena output is null");
    }
    out->reset();
    if (host_cold_cache_bytes != 0u && host_cold_cache_bytes < kExpertBridgeBytesPerExpert) {
        return fail(error, "host cold cache capacity cannot hold one complete expert");
    }
    // The generic NVFP4 kernel bounds physical plane indices to 65536.
    if (device < 0 || max_slots == 0u || max_slots > kMoeTopK ||
        physical_slots < max_slots || physical_slots > 65536u) {
        return fail(error, "invalid expert-slot arena device or slot count");
    }
    try {
        int device_count = 0;
        cudaDeviceProp properties{};
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device >= device_count || cudaSetDevice(device) != cudaSuccess ||
            cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
            properties.major * 10 + properties.minor < 120) {
            return fail(error, "expert-slot arena requires an available SM120 device");
        }
        auto arena = std::shared_ptr<expert_slot_arena>(new expert_slot_arena());
        arena->impl_ = std::make_shared<impl>();
        arena->impl_->device = device;
        arena->impl_->max_slots = max_slots;
        arena->impl_->physical_slots = physical_slots;
        arena->impl_->host_cold_capacity_bytes = host_cold_cache_bytes;
        arena->impl_->cache.resize(physical_slots);
        const cudaError_t event_status = cudaEventCreateWithFlags(
                &arena->impl_->reuse_event, cudaEventDisableTiming);
        if (event_status != cudaSuccess) {
            throw std::runtime_error(cuda_error(
                    "cudaEventCreateWithFlags(expert-slot reuse)",
                    event_status));
        }

        auto allocate = [&](auto** pointer, std::uint64_t bytes) {
            if (bytes > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("GPU arena plane byte count exceeds size_t");
            }
            const cudaError_t status = cudaMalloc(
                    reinterpret_cast<void**>(pointer), static_cast<std::size_t>(bytes));
            if (status != cudaSuccess) {
                throw std::runtime_error(cuda_error("cudaMalloc(expert-slot arena)", status));
            }
            std::uint64_t total = 0u;
            if (!checked_add(arena->impl_->gpu_bytes, bytes, &total)) {
                throw std::runtime_error("GPU arena capacity accounting overflow");
            }
            arena->impl_->gpu_bytes = total;
        };
        std::uint64_t weight_bytes = 0u;
        std::uint64_t scale_bytes = 0u;
        std::uint64_t scalar_bytes = 0u;
        std::uint64_t slot_index_bytes = 0u;
        if (!checked_multiply(819200u, physical_slots, &weight_bytes) ||
            !checked_multiply(102400u, physical_slots, &scale_bytes) ||
            !checked_multiply(sizeof(float), physical_slots, &scalar_bytes) ||
            !checked_multiply(sizeof(std::uint32_t), max_slots, &slot_index_bytes)) {
            return fail(error, "GPU arena plane size overflow");
        }
        allocate(&arena->impl_->gate_weight, weight_bytes);
        allocate(&arena->impl_->gate_scale, scale_bytes);
        allocate(&arena->impl_->gate_global, scalar_bytes);
        allocate(&arena->impl_->up_weight, weight_bytes);
        allocate(&arena->impl_->up_scale, scale_bytes);
        allocate(&arena->impl_->up_global, scalar_bytes);
        allocate(&arena->impl_->down_weight, weight_bytes);
        allocate(&arena->impl_->down_scale, scale_bytes);
        allocate(&arena->impl_->down_global, scalar_bytes);
        allocate(&arena->impl_->slot_indices, slot_index_bytes);
        *out = std::move(arena);
        return true;
    } catch (const std::exception& exception) {
        return fail(error, std::string("expert-slot arena create exception: ") +
                           exception.what());
    } catch (...) {
        return fail(error, "expert-slot arena create exception");
    }
}

expert_slot_arena_metrics expert_slot_arena::metrics() const noexcept {
    expert_slot_arena_metrics result{};
    if (!impl_) return result;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    result.device = impl_->device;
    result.bounded_slots = impl_->max_slots;
    result.attached_bridges = impl_->attached_bridges;
    result.active_leases = impl_->active_owner == nullptr ? 0u : 1u;
    result.capacity_bytes = impl_->gpu_bytes;
    result.ordered_handoffs = impl_->ordered_handoffs;
    result.stream_waits = impl_->stream_waits;
    result.physical_slots = impl_->physical_slots;
    result.persistent_cache = impl_->persistent_cache;
    result.poisoned = impl_->poisoned;
    result.cache_hits = impl_->cache_hits;
    result.cache_misses = impl_->cache_misses;
    result.upload_bytes = impl_->upload_bytes;
    result.evictions = impl_->evictions;
    result.host_cold_capacity_bytes = impl_->host_cold_capacity_bytes;
    result.host_cold_current_bytes = impl_->host_cold_current_bytes;
    result.host_cold_peak_bytes = impl_->host_cold_peak_bytes;
    result.host_cold_hits = impl_->host_cold_hits;
    result.host_cold_misses = impl_->host_cold_misses;
    result.host_demote_bytes = impl_->host_demote_bytes;
    return result;
}

struct expert_bridge::impl {
    std::shared_ptr<const expert_checkpoint_catalog> catalog;
    std::uint32_t layer = 0u;
    expert_bridge_options options;
    std::unique_ptr<ExpertPager> pager;
    std::shared_ptr<expert_slot_arena> arena;
    bool external_arena = false;

    mutable std::mutex mutex;
    bool busy = false;

    ~impl() {
        release_busy();
        if (arena && arena->impl_) {
            std::lock_guard<std::mutex> lock(arena->impl_->mutex);
            if (arena->impl_->attached_bridges != 0u) {
                --arena->impl_->attached_bridges;
            }
        }
    }

    bool acquire_busy(cudaStream_t stream, std::string* error) noexcept {
        std::lock_guard<std::mutex> bridge_lock(mutex);
        if (busy || !arena || !arena->impl_) {
            return fail(error, "shared GPU expert-slot arena is busy");
        }
        std::lock_guard<std::mutex> arena_lock(arena->impl_->mutex);
        if (arena->impl_->poisoned) {
            return fail(error, "shared GPU expert-slot arena is poisoned");
        }
        if (arena->impl_->active_owner != nullptr) {
            return fail(error, "shared GPU expert-slot arena is busy");
        }
        if (arena->impl_->reuse_event_recorded) {
            const cudaError_t wait_status = cudaStreamWaitEvent(
                    stream, arena->impl_->reuse_event, 0u);
            if (wait_status != cudaSuccess) {
                set_error(error, cuda_error(
                        "cudaStreamWaitEvent(expert-slot reuse)", wait_status));
                return false;
            }
            ++arena->impl_->stream_waits;
        }
        arena->impl_->active_owner = this;
        busy = true;
        return true;
    }

    bool handoff_busy(cudaStream_t stream, std::string* error) noexcept {
        std::lock_guard<std::mutex> bridge_lock(mutex);
        if (!busy || !arena || !arena->impl_) {
            return fail(error, "shared GPU expert-slot arena is not held");
        }
        std::lock_guard<std::mutex> arena_lock(arena->impl_->mutex);
        if (arena->impl_->active_owner != this ||
            arena->impl_->reuse_event == nullptr) {
            return fail(error, "shared GPU expert-slot arena owner mismatch");
        }
        const cudaError_t record_status = cudaEventRecord(
                arena->impl_->reuse_event, stream);
        if (record_status != cudaSuccess) {
            return fail(error, cuda_error(
                    "cudaEventRecord(expert-slot handoff)", record_status));
        }
        arena->impl_->reuse_event_recorded = true;
        ++arena->impl_->ordered_handoffs;
        arena->impl_->active_owner = nullptr;
        busy = false;
        return true;
    }

    std::uint32_t* device_slot_indices() const noexcept {
        return arena && arena->impl_ ? arena->impl_->slot_indices : nullptr;
    }

    void release_busy() noexcept {
        std::lock_guard<std::mutex> bridge_lock(mutex);
        if (!busy) return;
        if (arena && arena->impl_) {
            std::lock_guard<std::mutex> arena_lock(arena->impl_->mutex);
            if (arena->impl_->active_owner == this) {
                arena->impl_->active_owner = nullptr;
            }
        }
        busy = false;
    }
};

struct expert_bridge::prepared_generation::state {
    std::shared_ptr<expert_bridge::impl> owner;
    ExpertPager::Generation generation;
    std::uint64_t generation_id = 0u;
    std::vector<ExpertPager::Lease> leases;
    std::vector<ExpertKey> exclusive_source_keys;
    std::vector<std::shared_ptr<expert_slot_arena::impl::cold_entry>> cold_sources;
    bool uploads_succeeded = false;
    std::vector<std::uint32_t> experts;
    std::vector<std::uint32_t> slots;
    std::vector<std::uint32_t> writes;
    std::array<std::vector<float>, kProjectionCount> input_scales;
    moe_slot_planes planes;
    cudaEvent_t ready_event = nullptr;
    bool event_recorded = false;
    bool ready = false;
    bool generation_open = false;
    bool is_committed = false;
    bool usable = true;
    bool released = false;
    bool arena_handed_off = false;
    cudaStream_t stream = nullptr;

    void release_cold_sources(bool uploaded) noexcept {
        if (!owner || cold_sources.empty()) return;
        auto arena = owner->arena->impl_;
        std::lock_guard<std::mutex> lock(arena->mutex);
        for (auto& source : cold_sources) source->valid = !uploaded;
        cold_sources.clear();
        arena->reap_cold();
    }

    void discard_exclusive_sources() noexcept {
        if (!owner) return;
        // All local ExpertByteViews are gone, leases were released, and the
        // transaction was closed. discard_cached accepts only committed entries:
        // in this private pager those necessarily passed a successful upload
        // completion before commit. A failing generation must also retry keys
        // whose removal by an older successful generation its leases deferred.
        for (const auto& key : exclusive_source_keys) {
            try { (void)owner->pager->discard_cached(key); } catch (...) {}
        }
        exclusive_source_keys.clear();
    }

    // Only called while holding the global arena reservation. Synchronize an
    // event on THIS stream, never a device-wide event, before overwriting the
    // victim or reclaiming a source read by an earlier asynchronous upload.
    void demote(std::uint32_t slot) {
        auto arena = owner->arena->impl_;
        {
            std::lock_guard<std::mutex> lock(arena->mutex);
            if (!arena->cache[slot].valid) return;
        }
        auto synchronize_boundary = [&] {
            auto status = cudaEventRecord(ready_event, stream);
            if (status == cudaSuccess) status = cudaEventSynchronize(ready_event);
            if (status != cudaSuccess) {
                throw std::runtime_error(cuda_error("exclusive cache copy boundary", status));
            }
        };
        std::shared_ptr<expert_slot_arena::impl::cold_entry> target;
        {
            std::unique_lock<std::mutex> lock(arena->mutex);
            if (!cold_sources.empty() && arena->host_cold_capacity_bytes -
                    arena->host_cold_current_bytes < kExpertBridgeBytesPerExpert) {
                lock.unlock();
                synchronize_boundary();
                release_cold_sources(true);
                lock.lock();
            }
            if (!arena->make_cold_room()) {
                throw ExpertCapacityError("host cold cache is leased; retry after generation completion");
            }
            target = std::make_shared<expert_slot_arena::impl::cold_entry>();
            target->key = arena->cache[slot].key;
            target->last_use = arena->clock;  // Demotion is a new cold-tier use.
            // Reserve before allocating; allocation failure releases the charge.
            arena->cold.push_back(target);
            arena->host_cold_current_bytes += kExpertBridgeBytesPerExpert;
            arena->host_cold_peak_bytes = std::max(arena->host_cold_peak_bytes,
                                                  arena->host_cold_current_bytes);
        }
        try {
            target->bytes = std::make_unique<std::uint8_t[]>(kExpertBridgeBytesPerExpert);
            for (std::uint32_t p = 0u; p < kProjectionCount; ++p) {
                for (std::uint32_t plane = 0u; plane < kPlaneCount; ++plane) {
                    auto* destination = target->bytes.get() + p * kColdProjectionBytes +
                                        kColdPlaneOffsets[plane];
                    if (plane == 1u) {
                        std::memcpy(destination, &arena->cache[slot].input_scales[p], sizeof(float));
                        continue;
                    }
                    const auto status = cudaMemcpyAsync(destination, arena->device_plane(slot, p, plane),
                                                        kColdPlaneBytes[plane], cudaMemcpyDeviceToHost, stream);
                    if (status != cudaSuccess) {
                        throw std::runtime_error(cuda_error("cudaMemcpyAsync(expert demotion)", status));
                    }
                }
            }
            synchronize_boundary();
            release_cold_sources(true);  // The boundary also covers earlier H2D.
            std::lock_guard<std::mutex> lock(arena->mutex);
            target->valid = true;
            arena->cache[slot].valid = false;
            ++arena->evictions;
            arena->host_demote_bytes += kExpertBridgeBytesPerExpert - 3u * sizeof(float);
        } catch (...) {
            // Even a partial D2H owns its destination until outstanding copies
            // have drained. Keep the old GPU identity unless CUDA is poisoned.
            if (cudaStreamSynchronize(stream) != cudaSuccess) poison_arena();
            target.reset();
            std::lock_guard<std::mutex> lock(arena->mutex);
            arena->reap_cold();
            throw;
        }
    }

    void poison_arena() noexcept {
        if (!owner) return;
        auto arena = owner->arena->impl_;
        std::lock_guard<std::mutex> lock(arena->mutex);
        arena->poisoned = true;
        for (auto& entry : arena->cache) entry.valid = false;
    }

    void invalidate_writes() noexcept {
        if (!owner) return;
        auto arena = owner->arena->impl_;
        std::lock_guard<std::mutex> lock(arena->mutex);
        for (const auto slot : writes) arena->cache[slot].valid = false;
    }

    bool release_lease(std::string* error) noexcept {
        if (!owner || arena_handed_off) return true;
        // A ready_event covers uploads, not consumers queued after prepare().
        // Record their completion even for commit/rollback/destruction without
        // an explicit handoff. Never release a newer same-bridge reservation.
        if (!owner->handoff_busy(stream, error)) {
            (void)cudaStreamSynchronize(stream);
            poison_arena();
            owner->release_busy();
            arena_handed_off = true;
            return false;
        }
        arena_handed_off = true;
        return true;
    }

    bool wait(std::string* error) noexcept {
        if (released || !usable || owner == nullptr) {
            return fail(error, "prepared generation is no longer valid");
        }
        if (ready) {
            return true;
        }
        if (!event_recorded || ready_event == nullptr) {
            return fail(error, "prepared generation has no CUDA completion event");
        }
        const cudaError_t status = cudaEventSynchronize(ready_event);
        if (status != cudaSuccess) {
            poison_arena();
            return fail(error, cuda_error("cudaEventSynchronize", status));
        }
        ready = true;
        uploads_succeeded = true;
        release_cold_sources(true);
        return true;
    }

    void destroy_event() noexcept {
        if (ready_event != nullptr) {
            (void)cudaEventDestroy(ready_event);
            ready_event = nullptr;
        }
        event_recorded = false;
    }

    void release_noexcept() noexcept {
        if (released) {
            return;
        }
        if (owner != nullptr) {
            if (!ready && event_recorded && ready_event != nullptr) {
                if (cudaEventSynchronize(ready_event) != cudaSuccess) poison_arena();
                else uploads_succeeded = true;
                ready = true;
            }
            if (!event_recorded) {
                // Includes exceptions before the upload completion event exists.
                if (cudaStreamSynchronize(stream) != cudaSuccess) poison_arena();
                invalidate_writes();
            }
            release_cold_sources(uploads_succeeded);
            (void)release_lease(nullptr);
            if (generation_open) {
                try {
                    owner->pager->rollback(generation);
                } catch (...) {
                }
                generation_open = false;
            }
            leases.clear();
            discard_exclusive_sources();
        }
        destroy_event();
        usable = false;
        released = true;
        owner.reset();
    }

    ~state() { release_noexcept(); }
};

expert_bridge::prepared_generation::prepared_generation() noexcept = default;
expert_bridge::prepared_generation::~prepared_generation() = default;
expert_bridge::prepared_generation::prepared_generation(prepared_generation&&) noexcept = default;
expert_bridge::prepared_generation& expert_bridge::prepared_generation::operator=(
        prepared_generation&&) noexcept = default;
expert_bridge::prepared_generation::prepared_generation(std::unique_ptr<state> state) noexcept
    : state_(std::move(state)) {}

bool expert_bridge::prepared_generation::valid() const noexcept {
    return state_ != nullptr && state_->usable && !state_->released && state_->owner != nullptr;
}

bool expert_bridge::prepared_generation::committed() const noexcept {
    return state_ != nullptr && state_->is_committed;
}

std::uint64_t expert_bridge::prepared_generation::generation_id() const noexcept {
    return valid() ? state_->generation_id : 0u;
}

std::uint32_t expert_bridge::prepared_generation::slot_count() const noexcept {
    return valid() ? static_cast<std::uint32_t>(state_->experts.size()) : 0u;
}

std::uint32_t expert_bridge::prepared_generation::physical_slot_capacity() const noexcept {
    return valid() ? state_->owner->arena->impl_->physical_slots : 0u;
}

const std::uint32_t* expert_bridge::prepared_generation::selected_experts() const noexcept {
    return valid() ? state_->experts.data() : nullptr;
}

const std::uint32_t* expert_bridge::prepared_generation::slot_indices_host() const noexcept {
    return valid() ? state_->slots.data() : nullptr;
}

const std::uint32_t* expert_bridge::prepared_generation::slot_indices_device() const noexcept {
    return valid() ? state_->owner->device_slot_indices() : nullptr;
}

const moe_slot_planes& expert_bridge::prepared_generation::planes() const noexcept {
    static const moe_slot_planes empty;
    return valid() ? state_->planes : empty;
}

float expert_bridge::prepared_generation::input_scale(
        ExpertProjection projection, std::uint32_t slot) const noexcept {
    const std::size_t index = projection_index(projection);
    return valid() && index < state_->input_scales.size() &&
                   slot < state_->input_scales[index].size()
            ? state_->input_scales[index][slot]
            : std::numeric_limits<float>::quiet_NaN();
}

bool expert_bridge::prepared_generation::wait(std::string* error) noexcept {
    return state_ != nullptr && state_->wait(error);
}

bool expert_bridge::prepared_generation::handoff_after_consumers(
        cudaStream_t stream, std::string* error) noexcept {
    if (!valid() || stream != state_->stream) {
        return fail(error, "invalid prepared generation handoff stream");
    }
    if (state_->arena_handed_off) return true;
    return state_->release_lease(error);
}

bool expert_bridge::prepared_generation::commit(std::string* error) noexcept {
    if (!valid()) {
        return fail(error, "cannot commit an invalid prepared generation");
    }
    if (state_->is_committed) {
        return true;
    }
    if (!state_->wait(error)) {
        state_->release_noexcept();
        return false;
    }
    try {
        if (!state_->release_lease(error)) {
            state_->release_noexcept();
            return false;
        }
        state_->owner->pager->commit(state_->generation);
        state_->generation_open = false;
        state_->is_committed = true;
        state_->leases.clear();
        state_->discard_exclusive_sources();
        state_->release_cold_sources(true);
        state_->destroy_event();
        state_->usable = false;
        state_->released = true;
        state_->owner.reset();
        return true;
    } catch (const std::exception& exception) {
        const std::string message = std::string("pager commit failed: ") + exception.what();
        state_->release_noexcept();
        return fail(error, message);
    } catch (...) {
        state_->release_noexcept();
        return fail(error, "pager commit failed");
    }
}

bool expert_bridge::prepared_generation::rollback(std::string* error) noexcept {
    if (!valid() || state_->is_committed) {
        return fail(error, "cannot roll back an invalid or committed generation");
    }
    if (!state_->wait(error)) {
        state_->release_noexcept();
        return false;
    }
    try {
        if (!state_->release_lease(error)) {
            state_->release_noexcept();
            return false;
        }
        if (state_->generation_open) {
            state_->owner->pager->rollback(state_->generation);
            state_->generation_open = false;
        }
        state_->leases.clear();
        state_->discard_exclusive_sources();
        state_->release_cold_sources(true);
        state_->destroy_event();
        state_->usable = false;
        state_->released = true;
        state_->owner.reset();
        return true;
    } catch (const std::exception& exception) {
        const std::string message = std::string("pager rollback failed: ") + exception.what();
        state_->release_noexcept();
        return fail(error, message);
    } catch (...) {
        state_->release_noexcept();
        return fail(error, "pager rollback failed");
    }
}

expert_bridge::expert_bridge() = default;
expert_bridge::~expert_bridge() = default;
expert_bridge::expert_bridge(expert_bridge&&) noexcept = default;
expert_bridge& expert_bridge::operator=(expert_bridge&&) noexcept = default;

bool expert_bridge::open(
        std::shared_ptr<const expert_checkpoint_catalog> catalog,
        std::uint32_t layer,
        const expert_bridge_options& options,
        std::unique_ptr<expert_bridge>* out,
        std::string* error) noexcept {
    if (out == nullptr) {
        return fail(error, "bridge output is null");
    }
    out->reset();
    if (!catalog || layer >= kExpertBridgeLayers || options.device < 0 ||
        options.max_slots == 0u || options.max_slots > kMoeTopK ||
        options.prefetch_workers == 0u) {
        return fail(error, "invalid bridge catalog, layer, device, slot count or worker count");
    }
    std::uint64_t required_ram = 0u;
    if (!checked_multiply(kExpertBridgeBytesPerExpert, options.max_slots, &required_ram) ||
        options.ram_capacity_bytes < required_ram) {
        return fail(error, "bounded RAM capacity cannot hold one complete selected expert set");
    }
    try {
        auto bridge = std::make_unique<expert_bridge>();
        bridge->impl_ = std::make_shared<impl>();
        bridge->impl_->catalog = std::move(catalog);
        bridge->impl_->layer = layer;
        bridge->impl_->options = options;
        bridge->impl_->external_arena = static_cast<bool>(options.shared_arena);

        std::vector<ExpertRange> ranges;
        ranges.reserve(static_cast<std::size_t>(kMoeExperts) *
                       kProjectionCount * kPlaneCount);
        for (std::uint32_t expert = 0; expert < kMoeExperts; ++expert) {
            for (std::uint32_t projection_value = 0;
                 projection_value < kProjectionCount; ++projection_value) {
                const auto projection = static_cast<ExpertProjection>(projection_value);
                for (std::uint32_t plane_value = 0;
                     plane_value < kPlaneCount; ++plane_value) {
                    const auto plane = static_cast<ExpertPlane>(plane_value);
                    const expert_tensor_descriptor* descriptor =
                            bridge->impl_->catalog->find(layer, expert, projection, plane);
                    if (descriptor == nullptr) {
                        return fail(error, "catalog lost an admitted layer tensor");
                    }
                    ranges.push_back(ExpertRange{descriptor->key, descriptor->file,
                                                 descriptor->file_offset,
                                                 descriptor->bytes, std::nullopt});
                }
            }
        }
        ExpertPagerOptions pager_options;
        pager_options.ram_capacity_bytes = options.ram_capacity_bytes;
        pager_options.prefetch_workers = options.prefetch_workers;
        pager_options.require_checksums = false;
        pager_options.owned_direct_pread = options.owned_direct_pread;
        bridge->impl_->pager =
                std::make_unique<ExpertPager>(pager_options, std::move(ranges));

        if (options.shared_arena) {
            const expert_slot_arena_metrics arena_metrics = options.shared_arena->metrics();
            if (arena_metrics.device != options.device ||
                arena_metrics.bounded_slots != options.max_slots ||
                arena_metrics.capacity_bytes == 0u) {
                return fail(error, "shared expert-slot arena device/slot contract mismatch");
            }
            bridge->impl_->arena = options.shared_arena;
        } else if (!expert_slot_arena::create(options.device, options.max_slots,
                                              &bridge->impl_->arena, error)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(bridge->impl_->arena->impl_->mutex);
            ++bridge->impl_->arena->impl_->attached_bridges;
        }

        *out = std::move(bridge);
        return true;
    } catch (const std::exception& exception) {
        return fail(error, std::string("bridge open exception: ") + exception.what());
    } catch (...) {
        return fail(error, "bridge open exception");
    }
}

bool expert_bridge::prepare(const std::uint32_t* selected_experts,
                            std::uint32_t count,
                            cudaStream_t stream,
                            prepared_generation* out,
                            std::string* error) noexcept {
    if (!impl_ || selected_experts == nullptr || out == nullptr || count == 0u ||
        count > impl_->options.max_slots) {
        return fail(error, "invalid prepare arguments or selected expert count");
    }
    *out = prepared_generation{};
    std::array<bool, kMoeExperts> observed{};
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        if (selected_experts[slot] >= kMoeExperts || observed[selected_experts[slot]]) {
            return fail(error, "selected expert ids are out of range or duplicated");
        }
        observed[selected_experts[slot]] = true;
    }
    if (cudaSetDevice(impl_->options.device) != cudaSuccess) {
        return fail(error, "cannot select expert-slot device");
    }
    if (!impl_->acquire_busy(stream, error)) return false;
    std::unique_ptr<prepared_generation::state> state;
    const std::shared_ptr<expert_slot_arena::impl> arena = impl_->arena->impl_;
    const bool exclusive = arena->host_cold_capacity_bytes != 0u;
    try {
        state = std::make_unique<prepared_generation::state>();
        state->owner = impl_;
        state->stream = stream;
        state->experts.assign(selected_experts, selected_experts + count);
        state->slots.resize(count);
        state->writes.reserve(count);
        if (exclusive) {
            state->cold_sources.reserve(count);
            state->exclusive_source_keys.reserve(static_cast<std::size_t>(count) *
                                                  kProjectionCount * kPlaneCount);
        }
        for (auto& values : state->input_scales) values.assign(count, 0.0f);
        state->planes = moe_slot_planes{
            arena->gate_weight, arena->gate_scale, arena->gate_global,
            arena->up_weight, arena->up_scale, arena->up_global,
            arena->down_weight, arena->down_scale, arena->down_global};
        std::array<bool, kMoeTopK> hits{};
        std::vector<bool> protected_slots(arena->physical_slots, false);
        std::vector<ExpertKey> keys;
        keys.reserve(count);
        for (std::uint32_t index = 0u; index < count; ++index) {
            const auto* descriptor = impl_->catalog->find(
                    impl_->layer, selected_experts[index],
                    ExpertProjection::gate, ExpertPlane::weight);
            if (!descriptor) throw std::runtime_error("selected expert identity missing");
            keys.push_back(descriptor->key);
        }
        {
            std::lock_guard<std::mutex> lock(arena->mutex);
            if (arena->clock > std::numeric_limits<std::uint64_t>::max() - count) {
                throw std::runtime_error("expert cache LRU clock exhausted");
            }
            // Protect ALL batch hits before selecting any victim. Otherwise a
            // miss at position zero can evict an older hit later in the batch.
            if (arena->persistent_cache) {
                for (std::uint32_t index = 0u; index < count; ++index) {
                    for (std::uint32_t slot = 0u; slot < arena->physical_slots; ++slot) {
                        const auto& entry = arena->cache[slot];
                        if (entry.valid && entry.key == keys[index]) {
                            hits[index] = true;
                            protected_slots[slot] = true;
                            state->slots[index] = slot;
                            for (std::size_t p = 0u; p < kProjectionCount; ++p) {
                                state->input_scales[p][index] = entry.input_scales[p];
                            }
                            break;
                        }
                    }
                }
            }
            for (std::uint32_t index = 0u; index < count; ++index) {
                if (hits[index]) {
                    ++arena->cache_hits;
                } else {
                    std::uint32_t victim = arena->physical_slots;
                    if (!arena->persistent_cache) {
                        victim = index;
                    } else {
                        for (std::uint32_t slot = 0u; slot < arena->physical_slots; ++slot) {
                            if (protected_slots[slot]) continue;
                            if (victim == arena->physical_slots || !arena->cache[slot].valid ||
                                arena->cache[slot].last_use < arena->cache[victim].last_use) {
                                victim = slot;
                            }
                            if (!arena->cache[slot].valid) break;
                        }
                    }
                    if (victim == arena->physical_slots) {
                        throw std::runtime_error("expert cache has no unprotected victim");
                    }
                    auto& entry = arena->cache[victim];
                    if (!exclusive) {
                        if (entry.valid) ++arena->evictions;
                        entry.valid = false;  // Before the first possibly partial write.
                        state->writes.push_back(victim);
                        entry.key = keys[index];
                    }
                    protected_slots[victim] = true;
                    state->slots[index] = victim;
                    ++arena->cache_misses;
                }
                if (!exclusive || hits[index]) {
                    arena->cache[state->slots[index]].last_use = ++arena->clock;
                }
            }
        }
        cudaError_t status = cudaSetDevice(impl_->options.device);
        if (status != cudaSuccess) {
            throw std::runtime_error(cuda_error("cudaSetDevice", status));
        }
        status = cudaEventCreateWithFlags(&state->ready_event, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            throw std::runtime_error(cuda_error("cudaEventCreateWithFlags", status));
        }
        state->generation = impl_->pager->begin_generation();
        state->generation_id = state->generation.id();
        state->generation_open = true;
        state->leases.reserve(static_cast<std::size_t>(count) *
                              kProjectionCount * kPlaneCount);

        for (std::uint32_t slot = 0; slot < count; ++slot) {
            if (hits[slot]) continue;  // No pager request, read, validation or H2D.
            const std::uint32_t expert = selected_experts[slot];
            const std::uint32_t physical_slot = state->slots[slot];
            if (exclusive) {
                // Preserve exact old bytes BEFORE the first overwrite. RAM LRU
                // pressure may evict an unleased incoming key here; look it up
                // after room-making so a reported cold hit never reads NVMe.
                state->demote(physical_slot);
                std::shared_ptr<expert_slot_arena::impl::cold_entry> source;
                {
                    std::lock_guard<std::mutex> lock(arena->mutex);
                    auto& entry = arena->cache[physical_slot];
                    entry.valid = false;
                    state->writes.push_back(physical_slot);
                    entry.key = keys[slot];
                    entry.last_use = ++arena->clock;
                    for (const auto& candidate : arena->cold) {
                        if (candidate->valid && candidate->key == keys[slot]) {
                            source = candidate;
                            break;
                        }
                    }
                    if (source) {
                        state->cold_sources.push_back(source);
                        source->valid = false;  // Leased promotion, not a cold resident.
                        ++arena->host_cold_hits;
                    } else ++arena->host_cold_misses;
                }
                if (source) {
                    for (std::uint32_t p = 0u; p < kProjectionCount; ++p) {
                        for (std::uint32_t plane = 0u; plane < kPlaneCount; ++plane) {
                            const auto* bytes = source->bytes.get() + p * kColdProjectionBytes +
                                                kColdPlaneOffsets[plane];
                            if (plane == 1u) {
                                std::memcpy(&state->input_scales[p][slot], bytes, sizeof(float));
                                continue;
                            }
                            status = cudaMemcpyAsync(arena->device_plane(physical_slot, p, plane),
                                                     bytes, kColdPlaneBytes[plane], cudaMemcpyHostToDevice, stream);
                            if (status != cudaSuccess) {
                                throw std::runtime_error(cuda_error("cudaMemcpyAsync(cold promotion)", status));
                            }
                            std::lock_guard<std::mutex> lock(arena->mutex);
                            arena->upload_bytes += kColdPlaneBytes[plane];
                        }
                    }
                    continue;
                }
            }
            for (std::uint32_t projection_value = 0;
                 projection_value < kProjectionCount; ++projection_value) {
                const auto projection = static_cast<ExpertProjection>(projection_value);
                for (std::uint32_t plane_value = 0;
                     plane_value < kPlaneCount; ++plane_value) {
                    const auto plane = static_cast<ExpertPlane>(plane_value);
                    const expert_tensor_descriptor* descriptor =
                            impl_->catalog->find(impl_->layer, expert, projection, plane);
                    if (descriptor == nullptr) {
                        throw std::runtime_error("selected expert descriptor disappeared");
                    }
                    ExpertPager::Lease lease = impl_->pager->request(
                            state->generation, descriptor->key,
                            ExpertRequestOptions{false, true});
                    if (exclusive) state->exclusive_source_keys.push_back(descriptor->key);
                    const ExpertByteView view = lease.view();
                    if (!view.valid() || view.size() != descriptor->bytes) {
                        throw std::runtime_error("pager returned a mismatched expert plane");
                    }

                    void* destination = nullptr;
                    std::size_t copy_bytes = view.size();
                    if (plane == ExpertPlane::weight) {
                        std::uint8_t* base = projection == ExpertProjection::gate
                                ? arena->gate_weight
                                : (projection == ExpertProjection::up
                                           ? arena->up_weight
                                           : arena->down_weight);
                        destination = base + static_cast<std::size_t>(physical_slot) * 819200u;
                    } else if (plane == ExpertPlane::weight_scale) {
                        if (!valid_e4m3_scale_plane(view)) {
                            throw std::runtime_error("invalid E4M3 block scale in selected expert");
                        }
                        std::uint8_t* base = projection == ExpertProjection::gate
                                ? arena->gate_scale
                                : (projection == ExpertProjection::up
                                           ? arena->up_scale
                                           : arena->down_scale);
                        destination = base + static_cast<std::size_t>(physical_slot) * 102400u;
                    } else {
                        float value = 0.0f;
                        if (!valid_positive_scalar(view, &value)) {
                            throw std::runtime_error("invalid ModelOpt scalar in selected expert");
                        }
                        if (plane == ExpertPlane::input_scale) {
                            state->input_scales[projection_index(projection)][slot] = value;
                            state->leases.push_back(std::move(lease));
                            continue;
                        }
                        float* base = projection == ExpertProjection::gate
                                ? arena->gate_global
                                : (projection == ExpertProjection::up
                                           ? arena->up_global
                                           : arena->down_global);
                        destination = base + physical_slot;
                    }
                    status = cudaMemcpyAsync(destination, view.data(), copy_bytes,
                                             cudaMemcpyHostToDevice, stream);
                    if (status != cudaSuccess) {
                        throw std::runtime_error(cuda_error("cudaMemcpyAsync", status));
                    }
                    {
                        std::lock_guard<std::mutex> lock(arena->mutex);
                        arena->upload_bytes += copy_bytes;
                    }
                    state->leases.push_back(std::move(lease));
                }
            }
        }
        const std::size_t slot_bytes = static_cast<std::size_t>(count) * sizeof(std::uint32_t);
        status = cudaMemcpyAsync(arena->slot_indices, state->slots.data(), slot_bytes,
                                 cudaMemcpyHostToDevice, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(cuda_error("cudaMemcpyAsync(slot indices)", status));
        }
        status = cudaEventRecord(state->ready_event, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(cuda_error("cudaEventRecord", status));
        }
        state->event_recorded = true;
        if (arena->persistent_cache) {
            std::lock_guard<std::mutex> lock(arena->mutex);
            for (std::uint32_t index = 0u; index < count; ++index) {
                if (hits[index]) continue;
                auto& entry = arena->cache[state->slots[index]];
                for (std::size_t p = 0u; p < kProjectionCount; ++p) {
                    entry.input_scales[p] = state->input_scales[p][index];
                }
                entry.valid = true;
            }
        }
        *out = prepared_generation(std::move(state));
        return true;
    } catch (const std::exception& exception) {
        if (state) state.reset();
        else impl_->release_busy();
        return fail(error, std::string("prepare failed: ") + exception.what());
    } catch (...) {
        if (state) state.reset();
        else impl_->release_busy();
        return fail(error, "prepare failed");
    }
}

std::uint32_t expert_bridge::layer() const noexcept {
    return impl_ ? impl_->layer : 0u;
}

std::uint32_t expert_bridge::max_slots() const noexcept {
    return impl_ ? impl_->options.max_slots : 0u;
}

std::uint32_t expert_bridge::physical_slot_capacity() const noexcept {
    return impl_ && impl_->arena ? impl_->arena->impl_->physical_slots : 0u;
}

std::uint64_t expert_bridge::gpu_capacity_bytes() const noexcept {
    return impl_ && impl_->arena && impl_->arena->impl_
            ? impl_->arena->impl_->gpu_bytes : 0u;
}

expert_bridge_metrics expert_bridge::metrics() const noexcept {
    expert_bridge_metrics result{};
    if (!impl_) return result;
    result.layer = impl_->layer;
    result.bounded_slots = impl_->options.max_slots;
    result.gpu_capacity_bytes = gpu_capacity_bytes();
    result.external_arena = impl_->external_arena;
    result.gpu_owned_bytes = impl_->external_arena ? 0u : result.gpu_capacity_bytes;
    result.gpu_shared_bytes = impl_->external_arena ? result.gpu_capacity_bytes : 0u;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        result.generation_active = impl_->busy;
    }
    return result;
}

ExpertPagerMetrics expert_bridge::pager_metrics() const noexcept {
    try {
        return impl_ && impl_->pager ? impl_->pager->metrics() : ExpertPagerMetrics{};
    } catch (...) {
        return ExpertPagerMetrics{};
    }
}

}  // namespace axiom::qwen4exp
