#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace axiom::qwen4exp {

namespace detail {
struct ExpertPagerState;
struct ExpertLeaseState;
}  // namespace detail

enum class ExpertProjection : std::uint8_t {
    gate = 0,
    up = 1,
    down = 2,
};

enum class ExpertPlane : std::uint8_t {
    weight = 0,
    input_scale = 1,
    weight_scale = 2,
    weight_scale_2 = 3,
};

struct ExpertKey {
    std::string fingerprint;
    std::uint32_t layer = 0;
    std::uint32_t expert = 0;
    ExpertProjection projection = ExpertProjection::gate;
    ExpertPlane plane = ExpertPlane::weight;

    bool operator==(const ExpertKey& other) const noexcept;
    bool operator!=(const ExpertKey& other) const noexcept { return !(*this == other); }
};

struct ExpertKeyHash {
    std::size_t operator()(const ExpertKey& key) const noexcept;
};

struct Sha256 {
    std::array<std::uint8_t, 32> bytes{};

    static Sha256 from_hex(const std::string& text);
    std::string hex() const;

    bool operator==(const Sha256& other) const noexcept { return bytes == other.bytes; }
    bool operator!=(const Sha256& other) const noexcept { return !(*this == other); }
};

Sha256 sha256_bytes(const std::uint8_t* data, std::size_t size);

struct ExpertRange {
    ExpertKey key;
    std::string file;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::optional<Sha256> checksum;
};

class ExpertPagerError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ExpertAdmissionError : public ExpertPagerError {
public:
    using ExpertPagerError::ExpertPagerError;
};

class ExpertIntegrityError : public ExpertPagerError {
public:
    using ExpertPagerError::ExpertPagerError;
};

class ExpertCapacityError : public ExpertPagerError {
public:
    using ExpertPagerError::ExpertPagerError;
};

class ExpertTransactionError : public ExpertPagerError {
public:
    using ExpertPagerError::ExpertPagerError;
};

class ExpertByteView {
public:
    ExpertByteView() noexcept = default;

    const std::uint8_t* data() const noexcept;
    std::size_t size() const noexcept;
    bool valid() const noexcept;
    explicit operator bool() const noexcept { return valid(); }

private:
    friend struct detail::ExpertPagerState;
    friend struct detail::ExpertLeaseState;

    ExpertByteView(std::shared_ptr<const std::vector<std::uint8_t>> bytes,
                   std::shared_ptr<const std::atomic<bool>> validity) noexcept;

    std::shared_ptr<const std::vector<std::uint8_t>> bytes_;
    std::shared_ptr<const std::atomic<bool>> validity_;
};

struct AbstractGpuSlot {
    std::uint64_t opaque = 0;
    std::uint64_t bytes = 0;

    bool operator==(const AbstractGpuSlot& other) const noexcept {
        return opaque == other.opaque && bytes == other.bytes;
    }
};

struct ExpertGpuCallbacks {
    using Upload = std::function<AbstractGpuSlot(const ExpertKey&,
                                                  const ExpertByteView&,
                                                  std::uint64_t generation)>;
    using Release = std::function<void(const ExpertKey&, const AbstractGpuSlot&)>;

    Upload upload;
    Release release;
};

struct ExpertPagerOptions {
    std::uint64_t ram_capacity_bytes = 0;
    std::size_t prefetch_workers = 2;
    bool require_checksums = false;
    ExpertGpuCallbacks gpu;
    // Opt-in Linux O_DIRECT reads into bounded aligned staging, followed by an
    // exact copy into the same owned-vector snapshots used by the default.
    bool owned_direct_pread = false;
};

struct ExpertRequestOptions {
    bool upload_to_gpu = false;
    bool pin = true;
};

struct ExpertPagerMetrics {
    std::uint64_t requests = 0;
    std::uint64_t prefetches = 0;
    std::uint64_t ram_hits = 0;
    std::uint64_t ram_misses = 0;
    std::uint64_t deduplicated_waits = 0;
    std::uint64_t evictions = 0;
    std::uint64_t nvme_reads = 0;
    std::uint64_t nvme_bytes_read = 0;
    std::uint64_t nvme_read_latency_ns = 0;
    std::uint64_t checksum_failures = 0;
    std::uint64_t io_failures = 0;
    std::uint64_t gpu_uploads = 0;
    std::uint64_t gpu_upload_bytes = 0;
    std::uint64_t gpu_releases = 0;
    std::uint64_t gpu_release_failures = 0;
    std::uint64_t current_ram_bytes = 0;
    std::uint64_t peak_ram_bytes = 0;
    std::uint64_t cache_entries = 0;
    std::uint64_t active_leases = 0;
    std::uint64_t active_generations = 0;
    // Direct syscall counters include failed-load work and EINTR attempts;
    // payload bytes count only ranges passing identity/checksum validation.
    std::uint64_t direct_read_calls = 0;
    std::uint64_t direct_requested_bytes = 0;
    std::uint64_t direct_bytes_read = 0;
    std::uint64_t direct_payload_bytes = 0;
    // Separate from current_ram_bytes; reservations, not machine RAM/RSS.
    std::uint64_t direct_staging_limit_bytes = 0;
    std::uint64_t current_direct_staging_bytes = 0;
    std::uint64_t peak_direct_staging_bytes = 0;
};

class ExpertPager {
public:
    class Generation {
    public:
        Generation() noexcept = default;
        std::uint64_t id() const noexcept { return id_; }
        bool valid() const noexcept { return id_ != 0 && owner_cookie_ != 0; }

    private:
        friend class ExpertPager;
        friend struct detail::ExpertPagerState;

        Generation(std::uint64_t id, std::uint64_t owner_cookie) noexcept
            : id_(id), owner_cookie_(owner_cookie) {}

        std::uint64_t id_ = 0;
        std::uint64_t owner_cookie_ = 0;
    };

    class Lease {
    public:
        Lease() noexcept = default;
        ~Lease();
        Lease(Lease&& other) noexcept;
        Lease& operator=(Lease&& other) noexcept;
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        bool valid() const noexcept;
        ExpertKey key() const;
        std::uint64_t generation() const noexcept;
        ExpertByteView view() const;
        std::optional<AbstractGpuSlot> gpu_slot() const;

    private:
        friend class ExpertPager;
        friend struct detail::ExpertPagerState;

        explicit Lease(std::shared_ptr<detail::ExpertLeaseState> state) noexcept;
        std::shared_ptr<detail::ExpertLeaseState> state_;
    };

    ExpertPager(ExpertPagerOptions options, std::vector<ExpertRange> ranges);
    ~ExpertPager();

    ExpertPager(const ExpertPager&) = delete;
    ExpertPager& operator=(const ExpertPager&) = delete;
    ExpertPager(ExpertPager&&) = delete;
    ExpertPager& operator=(ExpertPager&&) = delete;

    Generation begin_generation();
    Lease request(const Generation& generation,
                  const ExpertKey& key,
                  ExpertRequestOptions options = {});
    std::future<Lease> prefetch(const Generation& generation,
                                const ExpertKey& key,
                                ExpertRequestOptions options = {false, false});

    void pin(Lease& lease);
    void unpin(Lease& lease);
    void commit(const Generation& generation);
    void rollback(const Generation& generation);

    bool contains(const ExpertKey& key) const;
    // True if absent or a committed, unleased/evictable entry was removed.
    // Never revokes a view. GPU release callback errors propagate as usual.
    bool discard_cached(const ExpertKey& key);
    ExpertPagerMetrics metrics() const;

private:
    std::shared_ptr<detail::ExpertPagerState> state_;
};

}  // namespace axiom::qwen4exp
