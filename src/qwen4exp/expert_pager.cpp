#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "axiom/qwen4exp/expert_pager.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <linux/stat.h>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

namespace axiom::qwen4exp {

namespace {

constexpr std::uint64_t kDirectAlignment = 4096;
constexpr std::uint64_t kDirectChunkBytes = 4U * 1024U * 1024U;

std::string errno_message(const char* operation, const std::string& path) {
    std::ostringstream out;
    out << operation << " failed for " << path << ": " << std::strerror(errno);
    return out.str();
}

std::uint32_t rotate_right(std::uint32_t value, unsigned count) noexcept {
    return (value >> count) | (value << (32U - count));
}

class Sha256Engine {
public:
    Sha256Engine() noexcept
        : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

    void update(const std::uint8_t* data, std::size_t size) noexcept {
        if (size == 0) {
            return;
        }
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    Sha256 finish() noexcept {
        const std::uint64_t bit_count = total_bytes_ * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                      block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                  block_.begin() + 56, 0U);
        for (unsigned i = 0; i < 8; ++i) {
            block_[63U - i] = static_cast<std::uint8_t>(bit_count >> (i * 8U));
        }
        transform(block_.data());

        Sha256 digest;
        for (std::size_t i = 0; i < state_.size(); ++i) {
            digest.bytes[i * 4U] = static_cast<std::uint8_t>(state_[i] >> 24U);
            digest.bytes[i * 4U + 1U] = static_cast<std::uint8_t>(state_[i] >> 16U);
            digest.bytes[i * 4U + 2U] = static_cast<std::uint8_t>(state_[i] >> 8U);
            digest.bytes[i * 4U + 3U] = static_cast<std::uint8_t>(state_[i]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* block) noexcept {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
            0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
            0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
            0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
            0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
            0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
            0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
            0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
            0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
            0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
            0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
            0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
            0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
        };

        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0; i < 16; ++i) {
            const std::size_t j = i * 4U;
            words[i] = (static_cast<std::uint32_t>(block[j]) << 24U) |
                       (static_cast<std::uint32_t>(block[j + 1U]) << 16U) |
                       (static_cast<std::uint32_t>(block[j + 2U]) << 8U) |
                       static_cast<std::uint32_t>(block[j + 3U]);
        }
        for (std::size_t i = 16; i < words.size(); ++i) {
            const std::uint32_t s0 = rotate_right(words[i - 15U], 7U) ^
                                     rotate_right(words[i - 15U], 18U) ^
                                     (words[i - 15U] >> 3U);
            const std::uint32_t s1 = rotate_right(words[i - 2U], 17U) ^
                                     rotate_right(words[i - 2U], 19U) ^
                                     (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t s1 = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^
                                     rotate_right(e, 25U);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose + k[i] + words[i];
            const std::uint32_t s0 = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^
                                     rotate_right(a, 22U);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

int hex_digit(char value) noexcept {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

std::size_t hash_combine(std::size_t seed, std::size_t value) noexcept {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

void validate_projection(ExpertProjection projection) {
    switch (projection) {
        case ExpertProjection::gate:
        case ExpertProjection::up:
        case ExpertProjection::down:
            return;
    }
    throw ExpertAdmissionError("unknown qwen4_exp expert projection");
}

void validate_plane(ExpertPlane plane) {
    switch (plane) {
        case ExpertPlane::weight:
        case ExpertPlane::input_scale:
        case ExpertPlane::weight_scale:
        case ExpertPlane::weight_scale_2:
            return;
    }
    throw ExpertAdmissionError("unknown qwen4_exp expert plane");
}

int open_regular_file_without_symlinks(const std::string& path, int extra_flags = 0) {
    if (path.empty() || path.front() != '/') {
        throw ExpertAdmissionError("expert source path must be absolute");
    }

    std::vector<std::string> components;
    std::size_t cursor = 1;
    while (cursor <= path.size()) {
        const std::size_t slash = path.find('/', cursor);
        const std::size_t end = slash == std::string::npos ? path.size() : slash;
        if (end != cursor) {
            std::string component = path.substr(cursor, end - cursor);
            if (component == "." || component == "..") {
                throw ExpertAdmissionError("expert source path contains a traversal component");
            }
            components.push_back(std::move(component));
        }
        if (slash == std::string::npos) {
            break;
        }
        cursor = slash + 1U;
    }
    if (components.empty()) {
        throw ExpertAdmissionError("expert source path does not name a regular file");
    }

    int parent = ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC);
    if (parent < 0) {
        throw ExpertAdmissionError(errno_message("open", "/"));
    }

    for (std::size_t index = 0; index < components.size(); ++index) {
        const bool final = index + 1U == components.size();
        const int flags = final ? (O_RDONLY | O_CLOEXEC | O_NOFOLLOW | extra_flags)
                                : (O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        const int next = ::openat(parent, components[index].c_str(), flags);
        const int saved_errno = errno;
        ::close(parent);
        if (next < 0) {
            errno = saved_errno;
            throw ExpertAdmissionError(errno_message("openat(O_NOFOLLOW)", path));
        }
        parent = next;
    }

    struct stat status {};
    if (::fstat(parent, &status) != 0) {
        const int saved_errno = errno;
        ::close(parent);
        errno = saved_errno;
        throw ExpertAdmissionError(errno_message("fstat", path));
    }
    if (!S_ISREG(status.st_mode)) {
        ::close(parent);
        throw ExpertAdmissionError("expert source is not a regular file: " + path);
    }
    return parent;
}

struct FileIdentity {
    dev_t device = 0;
    ino_t inode = 0;
    off_t size = 0;
    timespec modified{};
    timespec changed{};
};

FileIdentity identity_from_stat(const struct stat& status) noexcept {
    return FileIdentity{status.st_dev, status.st_ino, status.st_size,
                        status.st_mtim, status.st_ctim};
}

bool same_timespec(const timespec& left, const timespec& right) noexcept {
    return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

bool same_identity(const FileIdentity& left, const FileIdentity& right) noexcept {
    return left.device == right.device && left.inode == right.inode &&
           left.size == right.size && same_timespec(left.modified, right.modified) &&
           same_timespec(left.changed, right.changed);
}

struct ChecksumMismatch final : ExpertIntegrityError {
    using ExpertIntegrityError::ExpertIntegrityError;
};

}  // namespace

bool ExpertKey::operator==(const ExpertKey& other) const noexcept {
    return fingerprint == other.fingerprint && layer == other.layer &&
           expert == other.expert && projection == other.projection && plane == other.plane;
}

std::size_t ExpertKeyHash::operator()(const ExpertKey& key) const noexcept {
    std::size_t result = std::hash<std::string>{}(key.fingerprint);
    result = hash_combine(result, std::hash<std::uint32_t>{}(key.layer));
    result = hash_combine(result, std::hash<std::uint32_t>{}(key.expert));
    result = hash_combine(result, static_cast<std::size_t>(key.projection));
    result = hash_combine(result, static_cast<std::size_t>(key.plane));
    return result;
}

Sha256 Sha256::from_hex(const std::string& text) {
    if (text.size() != 64U) {
        throw ExpertAdmissionError("SHA-256 text must contain exactly 64 hexadecimal digits");
    }
    Sha256 result;
    for (std::size_t i = 0; i < result.bytes.size(); ++i) {
        const int high = hex_digit(text[i * 2U]);
        const int low = hex_digit(text[i * 2U + 1U]);
        if (high < 0 || low < 0) {
            throw ExpertAdmissionError("SHA-256 text contains a non-hexadecimal digit");
        }
        result.bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return result;
}

std::string Sha256::hex() const {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2U, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result[i * 2U] = digits[bytes[i] >> 4U];
        result[i * 2U + 1U] = digits[bytes[i] & 0x0fU];
    }
    return result;
}

Sha256 sha256_bytes(const std::uint8_t* data, std::size_t size) {
    if (data == nullptr && size != 0) {
        throw ExpertAdmissionError("cannot hash a null non-empty byte range");
    }
    Sha256Engine engine;
    engine.update(data, size);
    return engine.finish();
}

ExpertByteView::ExpertByteView(
    std::shared_ptr<const std::vector<std::uint8_t>> bytes,
    std::shared_ptr<const std::atomic<bool>> validity) noexcept
    : bytes_(std::move(bytes)), validity_(std::move(validity)) {}

const std::uint8_t* ExpertByteView::data() const noexcept {
    return valid() ? bytes_->data() : nullptr;
}

std::size_t ExpertByteView::size() const noexcept {
    return valid() ? bytes_->size() : 0U;
}

bool ExpertByteView::valid() const noexcept {
    return bytes_ != nullptr && validity_ != nullptr &&
           validity_->load(std::memory_order_acquire);
}

namespace detail {

struct SourceFile {
    std::string path;
    int descriptor = -1;
    int direct_descriptor = -1;
    FileIdentity identity{};

    SourceFile(std::string source_path, int source_descriptor, FileIdentity source_identity)
        : path(std::move(source_path)), descriptor(source_descriptor), identity(source_identity) {}

    ~SourceFile() {
        if (direct_descriptor >= 0) {
            ::close(direct_descriptor);
        }
        if (descriptor >= 0) {
            ::close(descriptor);
        }
    }

    SourceFile(const SourceFile&) = delete;
    SourceFile& operator=(const SourceFile&) = delete;
};

void admit_direct_descriptor(SourceFile& source) {
    // Re-use component-by-component no-symlink admission, then prove that the
    // second open still names the admitted inode and immutable identity.
    source.direct_descriptor = open_regular_file_without_symlinks(source.path, O_DIRECT);
    struct stat status {};
    if (::fstat(source.direct_descriptor, &status) != 0) {
        throw ExpertAdmissionError(errno_message("fstat(O_DIRECT)", source.path));
    }
    if (!same_identity(source.identity, identity_from_stat(status))) {
        throw ExpertIntegrityError("O_DIRECT descriptor differs from admitted source: " + source.path);
    }
    const int flags = ::fcntl(source.direct_descriptor, F_GETFL);
    if (flags < 0 || !(flags & O_DIRECT) || (flags & O_ACCMODE) != O_RDONLY) {
        throw ExpertAdmissionError("read-only O_DIRECT flags not retained: " + source.path);
    }
#if defined(SYS_statx) && defined(STATX_DIOALIGN) && defined(STATX_ATTR_VERITY)
    struct statx alignment {};
    if (::syscall(SYS_statx, source.direct_descriptor, "", AT_EMPTY_PATH,
                  STATX_DIOALIGN, &alignment) != 0) {
        throw ExpertAdmissionError(errno_message("statx(STATX_DIOALIGN)", source.path));
    }
    // fs-verity may silently route O_DIRECT through buffered reads. Reject it.
    if ((alignment.stx_attributes & STATX_ATTR_VERITY) != 0 ||
        !(alignment.stx_mask & STATX_DIOALIGN) ||
        alignment.stx_dio_mem_align == 0 || alignment.stx_dio_offset_align == 0 ||
        kDirectAlignment % alignment.stx_dio_mem_align != 0 ||
        kDirectAlignment % alignment.stx_dio_offset_align != 0) {
        throw ExpertAdmissionError("source lacks verified 4KiB-compatible direct I/O "
                                  "(no buffered fallback): " + source.path);
    }
#else
    throw ExpertAdmissionError("build lacks required Linux direct-I/O alignment reporting");
#endif
}

struct RangeRecord {
    std::shared_ptr<SourceFile> source;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::optional<Sha256> checksum;
};

enum class RamState : std::uint8_t { loading, ready, failed };
enum class GpuState : std::uint8_t { empty, uploading, ready, failed };

struct CacheEntry {
    explicit CacheEntry(ExpertKey cache_key, std::uint64_t bytes)
        : key(std::move(cache_key)), reserved_bytes(bytes) {}

    ExpertKey key;
    std::uint64_t reserved_bytes = 0;
    RamState ram_state = RamState::loading;
    std::shared_ptr<std::vector<std::uint8_t>> data;
    std::exception_ptr ram_error;
    bool ram_committed = false;
    bool discard_pending = false;
    std::unordered_set<std::uint64_t> ram_generations;

    GpuState gpu_state = GpuState::empty;
    std::optional<AbstractGpuSlot> gpu_slot;
    std::shared_ptr<std::atomic<bool>> gpu_valid;
    std::exception_ptr gpu_error;
    bool gpu_committed = false;
    std::unordered_set<std::uint64_t> gpu_generations;

    std::uint64_t frequency = 0;
    std::uint64_t last_touch = 0;
    std::uint64_t pin_count = 0;
    std::condition_variable changed;
};

struct ExpertLeaseState {
    ~ExpertLeaseState();

    std::weak_ptr<ExpertPagerState> owner;
    std::uint64_t id = 0;
    ExpertKey key;
    std::uint64_t generation = 0;
    std::shared_ptr<CacheEntry> entry;
    std::shared_ptr<std::atomic<bool>> validity;
    ExpertByteView view;
    std::optional<AbstractGpuSlot> gpu_slot;
    std::shared_ptr<std::atomic<bool>> gpu_valid;
    std::uint64_t pins = 0;
    std::atomic<bool> active{true};
};

enum class GenerationStatus : std::uint8_t { active, committing, rolling_back };

struct GenerationState {
    explicit GenerationState(std::uint64_t generation_id) : id(generation_id) {}

    std::uint64_t id = 0;
    GenerationStatus status = GenerationStatus::active;
    std::uint64_t in_flight = 0;
    std::unordered_set<ExpertKey, ExpertKeyHash> touched_ram;
    std::unordered_set<ExpertKey, ExpertKeyHash> touched_gpu;
    std::vector<std::weak_ptr<ExpertLeaseState>> leases;
    std::condition_variable idle;
};

struct ReleasedSlot {
    ExpertKey key;
    AbstractGpuSlot slot;
};

struct PrefetchTask {
    std::shared_ptr<GenerationState> generation;
    ExpertKey key;
    ExpertRequestOptions options;
    std::promise<ExpertPager::Lease> result;
};

struct ExpertPagerState : std::enable_shared_from_this<ExpertPagerState> {
    ExpertPagerState(ExpertPagerOptions pager_options, std::vector<ExpertRange> ranges)
        : options(std::move(pager_options)), owner_cookie(next_owner_cookie()) {
        validate_options();
        admit_ranges(std::move(ranges));
    }

    ~ExpertPagerState() { shutdown(); }

    static std::uint64_t next_owner_cookie() noexcept {
        static std::atomic<std::uint64_t> next{1};
        std::uint64_t result = next.fetch_add(1, std::memory_order_relaxed);
        if (result == 0) {
            result = next.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

    void validate_options() {
        if (options.ram_capacity_bytes == 0) {
            throw ExpertAdmissionError("expert RAM cache capacity must be non-zero");
        }
        if (options.prefetch_workers == 0) {
            throw ExpertAdmissionError("expert pager requires at least one prefetch worker");
        }
        if (options.owned_direct_pread) {
            if (options.prefetch_workers >
                std::numeric_limits<std::uint64_t>::max() / kDirectChunkBytes) {
                throw ExpertAdmissionError("direct staging budget overflows uint64_t");
            }
            metrics_value.direct_staging_limit_bytes =
                options.prefetch_workers * kDirectChunkBytes;
        }
        if (static_cast<bool>(options.gpu.upload) != static_cast<bool>(options.gpu.release)) {
            throw ExpertAdmissionError("GPU upload and release callbacks must be configured together");
        }
    }

    void admit_ranges(std::vector<ExpertRange> ranges) {
        if (ranges.empty()) {
            throw ExpertAdmissionError("expert pager manifest is empty");
        }

        std::unordered_map<std::string, std::shared_ptr<SourceFile>> opened;
        for (auto& range : ranges) {
            if (range.key.fingerprint.empty() || range.key.fingerprint.size() > 256U) {
                throw ExpertAdmissionError("expert key fingerprint is empty or unreasonably large");
            }
            validate_projection(range.key.projection);
            validate_plane(range.key.plane);
            if (range.length == 0) {
                throw ExpertAdmissionError("expert range length must be non-zero");
            }
            if (range.length > options.ram_capacity_bytes) {
                throw ExpertAdmissionError("expert range exceeds bounded RAM cache capacity");
            }
            if (range.length > std::numeric_limits<std::size_t>::max()) {
                throw ExpertAdmissionError("expert range cannot be represented by size_t");
            }
            if (range.offset > std::numeric_limits<std::uint64_t>::max() - range.length) {
                throw ExpertAdmissionError("expert file range overflows uint64_t");
            }
            if (range.offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
                range.offset + range.length >
                    static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
                throw ExpertAdmissionError("expert file range overflows off_t");
            }
            if (options.owned_direct_pread && range.offset + range.length >
                static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) -
                    (kDirectAlignment - 1U)) {
                throw ExpertAdmissionError("aligned expert range overflows off_t");
            }
            if (options.require_checksums && !range.checksum.has_value()) {
                throw ExpertAdmissionError("checksum is mandatory for every configured expert range");
            }
            if (manifest.find(range.key) != manifest.end()) {
                throw ExpertAdmissionError("duplicate qwen4_exp expert range key");
            }

            std::shared_ptr<SourceFile> source;
            const auto existing = opened.find(range.file);
            if (existing != opened.end()) {
                source = existing->second;
            } else {
                const int descriptor = open_regular_file_without_symlinks(range.file);
                struct stat status {};
                if (::fstat(descriptor, &status) != 0) {
                    const int saved_errno = errno;
                    ::close(descriptor);
                    errno = saved_errno;
                    throw ExpertAdmissionError(errno_message("fstat", range.file));
                }
                source = std::make_shared<SourceFile>(range.file, descriptor,
                                                      identity_from_stat(status));
                if (options.owned_direct_pread) {
                    admit_direct_descriptor(*source);
                }
                opened.emplace(range.file, source);
            }

            const std::uint64_t file_size = static_cast<std::uint64_t>(source->identity.size);
            if (range.offset > file_size || range.length > file_size - range.offset) {
                throw ExpertAdmissionError("expert range lies outside the immutable source file");
            }

            manifest.emplace(std::move(range.key),
                             RangeRecord{std::move(source), range.offset, range.length,
                                         std::move(range.checksum)});
        }
        sources = std::move(opened);
    }

    void start_workers() {
        std::unique_lock<std::mutex> lock(mutex);
        if (!workers.empty()) {
            return;
        }
        try {
            workers.reserve(options.prefetch_workers);
            for (std::size_t i = 0; i < options.prefetch_workers; ++i) {
                workers.emplace_back([self = shared_from_this()] { self->worker_loop(); });
            }
        } catch (...) {
            accepting = false;
            stopping = true;
            lock.unlock();
            task_available.notify_all();
            for (auto& worker : workers) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            workers.clear();
            throw;
        }
    }

    std::shared_ptr<GenerationState> reserve_generation(
        const ExpertPager::Generation& generation, bool is_prefetch) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!accepting || generation.owner_cookie_ != owner_cookie || generation.id_ == 0) {
            throw ExpertTransactionError("generation does not belong to this expert pager");
        }
        const auto found = generations.find(generation.id_);
        if (found == generations.end() || found->second->status != GenerationStatus::active) {
            throw ExpertTransactionError("generation is closed or unknown");
        }
        if (found->second->in_flight == std::numeric_limits<std::uint64_t>::max()) {
            throw ExpertTransactionError("generation in-flight counter overflow");
        }
        ++found->second->in_flight;
        ++metrics_value.requests;
        if (is_prefetch) {
            ++metrics_value.prefetches;
        }
        return found->second;
    }

    void finish_generation_operation(const std::shared_ptr<GenerationState>& generation) noexcept {
        std::lock_guard<std::mutex> guard(mutex);
        if (generation->in_flight != 0) {
            --generation->in_flight;
        }
        if (generation->in_flight == 0) {
            generation->idle.notify_all();
        }
    }

    // All loaders, including synchronous request() callers, share these slots.
    // This is independent of payload reservations and of OS file-cache memory.
    struct DirectStaging {
        ExpertPagerState& owner;
        std::size_t size;
        void* data = nullptr;

        DirectStaging(ExpertPagerState& state, std::size_t bytes) : owner(state), size(bytes) {
            std::unique_lock<std::mutex> lock(owner.mutex);
            owner.direct_slot_available.wait(lock, [&] {
                return owner.active_direct_loads < owner.options.prefetch_workers;
            });
            ++owner.active_direct_loads;
            owner.metrics_value.current_direct_staging_bytes += size;
            owner.metrics_value.peak_direct_staging_bytes = std::max(
                owner.metrics_value.peak_direct_staging_bytes,
                owner.metrics_value.current_direct_staging_bytes);
            lock.unlock();
            const int rc = ::posix_memalign(&data, kDirectAlignment, size);
            if (rc != 0) {
                release_slot();
                throw ExpertPagerError(std::string("direct staging allocation failed: ") +
                                      std::strerror(rc));
            }
        }

        ~DirectStaging() {
            std::free(data);
            release_slot();
        }
        DirectStaging(const DirectStaging&) = delete;
        DirectStaging& operator=(const DirectStaging&) = delete;

        void release_slot() noexcept {
            std::lock_guard<std::mutex> lock(owner.mutex);
            owner.metrics_value.current_direct_staging_bytes -= size;
            --owner.active_direct_loads;
            owner.direct_slot_available.notify_one();
        }
    };

    void read_direct_range(const RangeRecord& range, std::uint8_t* destination) {
        const std::uint64_t first = range.offset / kDirectAlignment * kDirectAlignment;
        const std::uint64_t end = range.offset + range.length;
        const std::uint64_t padded_end =
            (end + kDirectAlignment - 1U) / kDirectAlignment * kDirectAlignment;
        const std::uint64_t file_size = static_cast<std::uint64_t>(range.source->identity.size);
        DirectStaging staging(*this, static_cast<std::size_t>(
            std::min(kDirectChunkBytes, padded_end - first)));
        std::uint64_t copied = 0;
        for (std::uint64_t offset = first; offset < padded_end;) {
            const std::size_t request = static_cast<std::size_t>(
                std::min<std::uint64_t>(staging.size, padded_end - offset));
            if (offset >= file_size) {
                throw ExpertIntegrityError("aligned direct read starts beyond admitted EOF");
            }
            ssize_t count;
            int saved_errno;
            do {
                count = ::pread(range.source->direct_descriptor, staging.data, request,
                                static_cast<off_t>(offset));
                saved_errno = errno;
                std::lock_guard<std::mutex> guard(mutex);
                ++metrics_value.direct_read_calls;
                metrics_value.direct_requested_bytes += request;
                if (count > 0) metrics_value.direct_bytes_read += static_cast<std::uint64_t>(count);
            } while (count < 0 && saved_errno == EINTR);
            if (count < 0) {
                errno = saved_errno;
                throw ExpertPagerError(errno_message("pread(O_DIRECT; no fallback)", range.source->path));
            }
            const std::uint64_t expected = std::min<std::uint64_t>(request, file_size - offset);
            // Only an exact admitted-EOF short read is valid. Never continue a
            // short read at an unaligned offset or manufacture zero padding.
            if (static_cast<std::uint64_t>(count) != expected ||
                (static_cast<std::size_t>(count) < request && offset + expected < end)) {
                throw ExpertIntegrityError("unexpected short/EOF direct read: " + range.source->path);
            }
            const std::uint64_t copy_start = std::max(offset, range.offset);
            const std::uint64_t copy_end = std::min(offset + expected, end);
            if (copy_end > copy_start) {
                const std::size_t bytes = static_cast<std::size_t>(copy_end - copy_start);
                std::memcpy(destination + (copy_start - range.offset),
                            static_cast<const std::uint8_t*>(staging.data) + (copy_start - offset), bytes);
                copied += bytes;
            }
            offset += request;
        }
        if (copied != range.length) {
            throw ExpertIntegrityError("direct reads did not cover the complete expert range");
        }
    }

    std::shared_ptr<std::vector<std::uint8_t>> read_range(const RangeRecord& range) {
        const auto start = std::chrono::steady_clock::now();
        try {
            const int descriptor = options.owned_direct_pread
                ? range.source->direct_descriptor : range.source->descriptor;
            struct stat before_status {};
            if (::fstat(descriptor, &before_status) != 0) {
                throw ExpertPagerError(errno_message("fstat", range.source->path));
            }
            if (!same_identity(range.source->identity, identity_from_stat(before_status))) {
                throw ExpertIntegrityError("immutable expert source identity changed before pread: " +
                                           range.source->path);
            }

            auto bytes = std::make_shared<std::vector<std::uint8_t>>(
                static_cast<std::size_t>(range.length));
            std::size_t completed = 0;
            if (options.owned_direct_pread) {
                read_direct_range(range, bytes->data());
                completed = bytes->size();
            }
            while (completed < bytes->size()) {
                const off_t offset = static_cast<off_t>(range.offset + completed);
                const std::size_t request = std::min(
                    bytes->size() - completed,
                    static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
                const ssize_t count = ::pread(descriptor,
                                              bytes->data() + completed,
                                              request, offset);
                if (count < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw ExpertPagerError(errno_message("pread", range.source->path));
                }
                if (count == 0) {
                    throw ExpertIntegrityError("unexpected EOF while reading expert range: " +
                                               range.source->path);
                }
                completed += static_cast<std::size_t>(count);
            }

            struct stat after_status {};
            if (::fstat(descriptor, &after_status) != 0) {
                throw ExpertPagerError(errno_message("fstat", range.source->path));
            }
            if (!same_identity(range.source->identity, identity_from_stat(after_status))) {
                throw ExpertIntegrityError("immutable expert source identity changed during pread: " +
                                           range.source->path);
            }
            if (range.checksum.has_value() &&
                sha256_bytes(bytes->data(), bytes->size()) != *range.checksum) {
                throw ChecksumMismatch("SHA-256 mismatch for expert range in " +
                                       range.source->path);
            }

            const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count();
            {
                std::lock_guard<std::mutex> guard(mutex);
                ++metrics_value.nvme_reads;
                metrics_value.nvme_bytes_read += range.length;
                metrics_value.nvme_read_latency_ns += static_cast<std::uint64_t>(elapsed);
                if (options.owned_direct_pread) metrics_value.direct_payload_bytes += range.length;
            }
            return bytes;
        } catch (const ChecksumMismatch&) {
            std::lock_guard<std::mutex> guard(mutex);
            ++metrics_value.checksum_failures;
            throw;
        } catch (...) {
            std::lock_guard<std::mutex> guard(mutex);
            ++metrics_value.io_failures;
            throw;
        }
    }

    bool entry_evictable_locked(const std::shared_ptr<CacheEntry>& entry) const noexcept {
        return entry->ram_state == RamState::ready &&
               entry->gpu_state != GpuState::uploading && entry->pin_count == 0 &&
               entry->ram_generations.empty() && entry->gpu_generations.empty() &&
               entry->data != nullptr && entry->data.use_count() == 1;
    }

    void evict_entry_locked(const std::shared_ptr<CacheEntry>& entry,
                            std::vector<ReleasedSlot>& released) {
        const auto found = cache.find(entry->key);
        if (found == cache.end() || found->second.get() != entry.get() ||
            !entry_evictable_locked(entry)) {
            return;
        }
        if (entry->gpu_state == GpuState::ready && entry->gpu_slot.has_value()) {
            if (entry->gpu_valid) {
                entry->gpu_valid->store(false, std::memory_order_release);
            }
            released.push_back(ReleasedSlot{entry->key, *entry->gpu_slot});
            entry->gpu_slot.reset();
            entry->gpu_valid.reset();
            entry->gpu_state = GpuState::empty;
        }
        if (metrics_value.current_ram_bytes < entry->reserved_bytes) {
            throw ExpertPagerError("expert RAM accounting underflow");
        }
        metrics_value.current_ram_bytes -= entry->reserved_bytes;
        cache.erase(found);
        entry->data.reset();
        ++metrics_value.evictions;
        metrics_value.cache_entries = cache.size();
    }

    void reap_discarded_locked(std::vector<ReleasedSlot>& released) {
        std::vector<std::shared_ptr<CacheEntry>> candidates;
        candidates.reserve(cache.size());
        for (const auto& item : cache) {
            if (item.second->discard_pending && entry_evictable_locked(item.second)) {
                candidates.push_back(item.second);
            }
        }
        for (const auto& entry : candidates) {
            evict_entry_locked(entry, released);
        }
    }

    void reserve_capacity_locked(std::uint64_t required,
                                 std::vector<ReleasedSlot>& released) {
        if (required > options.ram_capacity_bytes) {
            throw ExpertCapacityError("expert range exceeds RAM cache capacity");
        }
        reap_discarded_locked(released);
        if (required <= options.ram_capacity_bytes - metrics_value.current_ram_bytes) {
            return;
        }

        const std::uint64_t need =
            required - (options.ram_capacity_bytes - metrics_value.current_ram_bytes);
        std::vector<std::shared_ptr<CacheEntry>> candidates;
        candidates.reserve(cache.size());
        for (const auto& item : cache) {
            if (entry_evictable_locked(item.second)) {
                candidates.push_back(item.second);
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const std::shared_ptr<CacheEntry>& left,
                     const std::shared_ptr<CacheEntry>& right) {
                      if (left->discard_pending != right->discard_pending) {
                          return left->discard_pending;
                      }
                      if (left->frequency != right->frequency) {
                          return left->frequency < right->frequency;
                      }
                      return left->last_touch < right->last_touch;
                  });

        std::uint64_t reclaimable = 0;
        std::size_t count = 0;
        while (count < candidates.size() && reclaimable < need) {
            if (reclaimable > std::numeric_limits<std::uint64_t>::max() -
                                  candidates[count]->reserved_bytes) {
                throw ExpertCapacityError("expert eviction accounting overflow");
            }
            reclaimable += candidates[count]->reserved_bytes;
            ++count;
        }
        if (reclaimable < need) {
            throw ExpertCapacityError(
                "bounded RAM cache is full and all candidate ranges are leased or transactional");
        }
        for (std::size_t i = 0; i < count; ++i) {
            evict_entry_locked(candidates[i], released);
        }
    }

    void release_slots(std::vector<ReleasedSlot> released, bool propagate_failure) {
        std::exception_ptr first_failure;
        for (const auto& item : released) {
            try {
                options.gpu.release(item.key, item.slot);
                std::lock_guard<std::mutex> guard(mutex);
                ++metrics_value.gpu_releases;
            } catch (...) {
                std::lock_guard<std::mutex> guard(mutex);
                ++metrics_value.gpu_release_failures;
                if (!first_failure) {
                    first_failure = std::current_exception();
                }
            }
        }
        if (propagate_failure && first_failure) {
            throw ExpertPagerError("abstract GPU release callback failed");
        }
    }

    std::shared_ptr<CacheEntry> acquire_ram(
        const std::shared_ptr<GenerationState>& generation, const ExpertKey& key) {
        std::shared_ptr<CacheEntry> entry;
        RangeRecord range;
        bool loader = false;
        std::vector<ReleasedSlot> released;

        try {
            std::unique_lock<std::mutex> lock(mutex);
            const auto range_found = manifest.find(key);
            if (range_found == manifest.end()) {
                throw ExpertAdmissionError("expert key is absent from the immutable range manifest");
            }
            range = range_found->second;

            auto found = cache.find(key);
            if (found != cache.end() && found->second->discard_pending) {
                reap_discarded_locked(released);
                found = cache.find(key);
                if (found != cache.end() && found->second->discard_pending) {
                    throw ExpertCapacityError(
                        "rolled-back expert range still has an outstanding zero-copy view");
                }
            }
            if (found == cache.end()) {
                reserve_capacity_locked(range.length, released);
                entry = std::make_shared<CacheEntry>(key, range.length);
                entry->frequency = 1;
                entry->last_touch = ++touch_clock;
                entry->ram_generations.insert(generation->id);
                generation->touched_ram.insert(key);
                cache.emplace(key, entry);
                metrics_value.current_ram_bytes += range.length;
                metrics_value.peak_ram_bytes =
                    std::max(metrics_value.peak_ram_bytes,
                             metrics_value.current_ram_bytes);
                metrics_value.cache_entries = cache.size();
                ++metrics_value.ram_misses;
                loader = true;
            } else {
                entry = found->second;
                if (!entry->ram_committed) {
                    entry->ram_generations.insert(generation->id);
                    generation->touched_ram.insert(key);
                }
                if (entry->ram_state == RamState::loading) {
                    ++metrics_value.deduplicated_waits;
                    entry->changed.wait(lock, [&entry] {
                        return entry->ram_state != RamState::loading;
                    });
                } else {
                    ++metrics_value.ram_hits;
                }
                if (entry->ram_state == RamState::failed) {
                    std::rethrow_exception(entry->ram_error);
                }
                if (entry->ram_state != RamState::ready || entry->data == nullptr) {
                    throw ExpertPagerError("expert RAM entry reached an invalid state");
                }
                ++entry->frequency;
                entry->last_touch = ++touch_clock;
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            try {
                release_slots(std::move(released), false);
            } catch (...) {
            }
            std::rethrow_exception(failure);
        }

        if (!loader) {
            release_slots(std::move(released), true);
            return entry;
        }

        try {
            release_slots(std::move(released), true);
            auto bytes = read_range(range);
            {
                std::lock_guard<std::mutex> guard(mutex);
                entry->data = std::move(bytes);
                entry->ram_state = RamState::ready;
                entry->changed.notify_all();
            }
            return entry;
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            std::lock_guard<std::mutex> guard(mutex);
            entry->ram_error = failure;
            entry->ram_state = RamState::failed;
            const auto found = cache.find(key);
            if (found != cache.end() && found->second.get() == entry.get()) {
                cache.erase(found);
                if (metrics_value.current_ram_bytes >= entry->reserved_bytes) {
                    metrics_value.current_ram_bytes -= entry->reserved_bytes;
                }
                metrics_value.cache_entries = cache.size();
            }
            entry->changed.notify_all();
            std::rethrow_exception(failure);
        }
    }

    void ensure_gpu(const std::shared_ptr<GenerationState>& generation,
                    const std::shared_ptr<CacheEntry>& entry) {
        if (!options.gpu.upload) {
            throw ExpertAdmissionError("GPU upload requested without an abstract GPU callback");
        }

        bool uploader = false;
        {
            std::unique_lock<std::mutex> lock(mutex);
            if (!entry->gpu_committed) {
                entry->gpu_generations.insert(generation->id);
                generation->touched_gpu.insert(entry->key);
            }
            if (entry->gpu_state == GpuState::uploading) {
                ++metrics_value.deduplicated_waits;
                entry->changed.wait(lock, [&entry] {
                    return entry->gpu_state != GpuState::uploading;
                });
            }
            if (entry->gpu_state == GpuState::failed) {
                std::rethrow_exception(entry->gpu_error);
            }
            if (entry->gpu_state == GpuState::ready) {
                return;
            }
            if (entry->gpu_state != GpuState::empty) {
                throw ExpertPagerError("expert GPU slot reached an invalid state");
            }
            entry->gpu_state = GpuState::uploading;
            uploader = true;
        }

        if (!uploader) {
            return;
        }

        try {
            auto validity = std::make_shared<std::atomic<bool>>(true);
            ExpertByteView view(entry->data, validity);
            AbstractGpuSlot slot;
            try {
                slot = options.gpu.upload(entry->key, view, generation->id);
            } catch (...) {
                validity->store(false, std::memory_order_release);
                throw;
            }
            validity->store(false, std::memory_order_release);
            if (slot.bytes == 0) {
                throw ExpertPagerError("abstract GPU upload returned a zero-byte slot");
            }
            {
                std::lock_guard<std::mutex> guard(mutex);
                entry->gpu_slot = slot;
                entry->gpu_valid = std::make_shared<std::atomic<bool>>(true);
                entry->gpu_state = GpuState::ready;
                ++metrics_value.gpu_uploads;
                metrics_value.gpu_upload_bytes += slot.bytes;
                entry->changed.notify_all();
            }
        } catch (...) {
            const std::exception_ptr failure = std::current_exception();
            std::lock_guard<std::mutex> guard(mutex);
            entry->gpu_error = failure;
            entry->gpu_state = GpuState::failed;
            entry->changed.notify_all();
            std::rethrow_exception(failure);
        }
    }

    ExpertPager::Lease perform_request(const std::shared_ptr<GenerationState>& generation,
                                       const ExpertKey& key,
                                       ExpertRequestOptions request_options) {
        if (request_options.upload_to_gpu && !options.gpu.upload) {
            throw ExpertAdmissionError("GPU upload requested without configured callbacks");
        }
        auto entry = acquire_ram(generation, key);
        if (request_options.upload_to_gpu) {
            ensure_gpu(generation, entry);
        }

        std::lock_guard<std::mutex> guard(mutex);
        if (entry->ram_state != RamState::ready || entry->data == nullptr) {
            throw ExpertPagerError("cannot lease a non-resident expert range");
        }
        auto lease = std::make_shared<ExpertLeaseState>();
        lease->owner = shared_from_this();
        lease->id = next_lease_id++;
        if (lease->id == 0) {
            lease->id = next_lease_id++;
        }
        lease->key = key;
        lease->generation = generation->id;
        lease->entry = entry;
        lease->validity = std::make_shared<std::atomic<bool>>(true);
        lease->view = ExpertByteView(entry->data, lease->validity);
        if (request_options.pin) {
            ++entry->pin_count;
            lease->pins = 1;
        }
        if (request_options.upload_to_gpu) {
            if (entry->gpu_state != GpuState::ready || !entry->gpu_slot.has_value() ||
                !entry->gpu_valid) {
                throw ExpertPagerError("cannot lease an incomplete abstract GPU slot");
            }
            lease->gpu_slot = entry->gpu_slot;
            lease->gpu_valid = entry->gpu_valid;
        }
        ++entry->frequency;
        entry->last_touch = ++touch_clock;
        leases.emplace(lease->id, lease);
        generation->leases.push_back(lease);
        ++metrics_value.active_leases;
        return ExpertPager::Lease(std::move(lease));
    }

    ExpertPager::Lease request(const ExpertPager::Generation& generation,
                               const ExpertKey& key,
                               ExpertRequestOptions request_options) {
        auto reserved = reserve_generation(generation, false);
        try {
            auto lease = perform_request(reserved, key, request_options);
            finish_generation_operation(reserved);
            return lease;
        } catch (...) {
            finish_generation_operation(reserved);
            throw;
        }
    }

    std::future<ExpertPager::Lease> prefetch(const ExpertPager::Generation& generation,
                                             const ExpertKey& key,
                                             ExpertRequestOptions request_options) {
        auto reserved = reserve_generation(generation, true);
        try {
            auto task = std::make_shared<PrefetchTask>();
            task->generation = reserved;
            task->key = key;
            task->options = request_options;
            std::future<ExpertPager::Lease> future = task->result.get_future();
            {
                std::lock_guard<std::mutex> guard(mutex);
                if (!accepting) {
                    throw ExpertTransactionError("expert pager is shutting down");
                }
                tasks.push_back(task);
            }
            task_available.notify_one();
            return future;
        } catch (...) {
            finish_generation_operation(reserved);
            throw;
        }
    }

    void worker_loop() noexcept {
        for (;;) {
            std::shared_ptr<PrefetchTask> task;
            {
                std::unique_lock<std::mutex> lock(mutex);
                task_available.wait(lock, [this] { return stopping || !tasks.empty(); });
                if (tasks.empty()) {
                    if (stopping) {
                        return;
                    }
                    continue;
                }
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            try {
                auto lease = perform_request(task->generation, task->key, task->options);
                task->result.set_value(std::move(lease));
                finish_generation_operation(task->generation);
            } catch (...) {
                const std::exception_ptr failure = std::current_exception();
                finish_generation_operation(task->generation);
                try {
                    task->result.set_exception(failure);
                } catch (...) {
                }
            }
        }
    }

    ExpertPager::Generation begin_generation() {
        std::lock_guard<std::mutex> guard(mutex);
        if (!accepting) {
            throw ExpertTransactionError("expert pager is shutting down");
        }
        std::uint64_t id = next_generation_id++;
        if (id == 0) {
            id = next_generation_id++;
        }
        generations.emplace(id, std::make_shared<GenerationState>(id));
        metrics_value.active_generations = generations.size();
        return ExpertPager::Generation(id, owner_cookie);
    }

    std::shared_ptr<GenerationState> close_generation_locked(
        std::unique_lock<std::mutex>& lock, const ExpertPager::Generation& generation,
        GenerationStatus closing_status) {
        if (generation.owner_cookie_ != owner_cookie || generation.id_ == 0) {
            throw ExpertTransactionError("generation does not belong to this expert pager");
        }
        const auto found = generations.find(generation.id_);
        if (found == generations.end() || found->second->status != GenerationStatus::active) {
            throw ExpertTransactionError("generation is already closed or unknown");
        }
        auto state = found->second;
        state->status = closing_status;
        state->idle.wait(lock, [&state] { return state->in_flight == 0; });
        return state;
    }

    void commit(const ExpertPager::Generation& generation) {
        std::unique_lock<std::mutex> lock(mutex);
        auto state = close_generation_locked(lock, generation, GenerationStatus::committing);
        for (const auto& key : state->touched_ram) {
            const auto found = cache.find(key);
            if (found != cache.end()) {
                found->second->ram_generations.erase(state->id);
                found->second->ram_committed = true;
            }
        }
        for (const auto& key : state->touched_gpu) {
            const auto found = cache.find(key);
            if (found != cache.end()) {
                found->second->gpu_generations.erase(state->id);
                if (found->second->gpu_state == GpuState::ready) {
                    found->second->gpu_committed = true;
                } else if (found->second->gpu_state == GpuState::failed &&
                           found->second->gpu_generations.empty()) {
                    found->second->gpu_state = GpuState::empty;
                    found->second->gpu_error = nullptr;
                }
            }
        }
        generations.erase(state->id);
        metrics_value.active_generations = generations.size();
    }

    void invalidate_lease_locked(const std::shared_ptr<ExpertLeaseState>& lease) {
        if (!lease || !lease->active.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        if (lease->validity) {
            lease->validity->store(false, std::memory_order_release);
        }
        if (lease->entry && lease->pins != 0) {
            if (lease->entry->pin_count < lease->pins) {
                lease->entry->pin_count = 0;
            } else {
                lease->entry->pin_count -= lease->pins;
            }
        }
        lease->pins = 0;
        lease->view = ExpertByteView();
        lease->gpu_slot.reset();
        lease->gpu_valid.reset();
        lease->entry.reset();
        leases.erase(lease->id);
        if (metrics_value.active_leases != 0) {
            --metrics_value.active_leases;
        }
    }

    void rollback(const ExpertPager::Generation& generation) {
        std::vector<ReleasedSlot> released;
        {
            std::unique_lock<std::mutex> lock(mutex);
            auto state =
                close_generation_locked(lock, generation, GenerationStatus::rolling_back);

            for (auto& weak_lease : state->leases) {
                if (auto lease = weak_lease.lock()) {
                    invalidate_lease_locked(lease);
                }
            }
            for (const auto& key : state->touched_gpu) {
                const auto found = cache.find(key);
                if (found == cache.end()) {
                    continue;
                }
                auto& entry = found->second;
                entry->gpu_generations.erase(state->id);
                if (!entry->gpu_committed && entry->gpu_generations.empty()) {
                    if (entry->gpu_state == GpuState::ready && entry->gpu_slot.has_value()) {
                        if (entry->gpu_valid) {
                            entry->gpu_valid->store(false, std::memory_order_release);
                        }
                        released.push_back(ReleasedSlot{entry->key, *entry->gpu_slot});
                    }
                    entry->gpu_slot.reset();
                    entry->gpu_valid.reset();
                    entry->gpu_error = nullptr;
                    entry->gpu_state = GpuState::empty;
                }
            }
            for (const auto& key : state->touched_ram) {
                const auto found = cache.find(key);
                if (found == cache.end()) {
                    continue;
                }
                auto& entry = found->second;
                entry->ram_generations.erase(state->id);
                if (!entry->ram_committed && entry->ram_generations.empty()) {
                    entry->discard_pending = true;
                }
            }
            reap_discarded_locked(released);
            generations.erase(state->id);
            metrics_value.active_generations = generations.size();
        }
        release_slots(std::move(released), true);
    }

    void pin(ExpertPager::Lease& public_lease) {
        auto lease = public_lease.state_;
        if (!lease) {
            throw ExpertTransactionError("cannot pin an empty expert lease");
        }
        std::lock_guard<std::mutex> guard(mutex);
        const auto owner = lease->owner.lock();
        if (owner.get() != this || !lease->active.load(std::memory_order_acquire) ||
            !lease->entry) {
            throw ExpertTransactionError("cannot pin an invalid expert lease");
        }
        if (lease->pins == std::numeric_limits<std::uint64_t>::max() ||
            lease->entry->pin_count == std::numeric_limits<std::uint64_t>::max()) {
            throw ExpertTransactionError("expert lease pin counter overflow");
        }
        ++lease->pins;
        ++lease->entry->pin_count;
    }

    void unpin(ExpertPager::Lease& public_lease) {
        auto lease = public_lease.state_;
        if (!lease) {
            throw ExpertTransactionError("cannot unpin an empty expert lease");
        }
        std::lock_guard<std::mutex> guard(mutex);
        const auto owner = lease->owner.lock();
        if (owner.get() != this || !lease->active.load(std::memory_order_acquire) ||
            !lease->entry) {
            throw ExpertTransactionError("cannot unpin an invalid expert lease");
        }
        if (lease->pins == 0 || lease->entry->pin_count == 0) {
            throw ExpertTransactionError("expert lease is not pinned");
        }
        --lease->pins;
        --lease->entry->pin_count;
    }

    void release_lease(ExpertLeaseState* raw_lease) noexcept {
        if (raw_lease == nullptr) {
            return;
        }
        std::vector<ReleasedSlot> released;
        {
            std::lock_guard<std::mutex> guard(mutex);
            if (!raw_lease->active.exchange(false, std::memory_order_acq_rel)) {
                return;
            }
            if (raw_lease->validity) {
                raw_lease->validity->store(false, std::memory_order_release);
            }
            if (raw_lease->entry && raw_lease->pins != 0) {
                if (raw_lease->entry->pin_count >= raw_lease->pins) {
                    raw_lease->entry->pin_count -= raw_lease->pins;
                } else {
                    raw_lease->entry->pin_count = 0;
                }
            }
            raw_lease->pins = 0;
            raw_lease->view = ExpertByteView();
            raw_lease->gpu_slot.reset();
            raw_lease->gpu_valid.reset();
            raw_lease->entry.reset();
            leases.erase(raw_lease->id);
            if (metrics_value.active_leases != 0) {
                --metrics_value.active_leases;
            }
            try {
                reap_discarded_locked(released);
            } catch (...) {
            }
        }
        try {
            release_slots(std::move(released), false);
        } catch (...) {
        }
    }

    bool contains(const ExpertKey& key) const {
        std::lock_guard<std::mutex> guard(mutex);
        const auto found = cache.find(key);
        return found != cache.end() && !found->second->discard_pending &&
               found->second->ram_state == RamState::ready;
    }

    ExpertPagerMetrics metrics() const {
        std::lock_guard<std::mutex> guard(mutex);
        return metrics_value;
    }

    bool discard_cached(const ExpertKey& key) {
        std::vector<ReleasedSlot> released;
        {
            std::lock_guard<std::mutex> guard(mutex);
            const auto found = cache.find(key);
            if (found == cache.end()) return true;
            // Copy the shared_ptr: evict_entry_locked erases the map node.
            const auto entry = found->second;
            // Map + this local handle must be the only entry owners. This also
            // excludes the interval between acquire_ram() and lease creation.
            if (entry.use_count() != 2 || !entry->ram_committed ||
                !entry_evictable_locked(entry)) return false;
            evict_entry_locked(entry, released);
        }
        release_slots(std::move(released), true);
        return true;
    }

    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> guard(mutex);
            if (shutdown_complete) {
                return;
            }
            accepting = false;
            stopping = true;
        }
        task_available.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();

        std::vector<ReleasedSlot> released;
        {
            std::lock_guard<std::mutex> guard(mutex);
            for (auto& item : leases) {
                if (auto lease = item.second.lock()) {
                    if (lease->active.exchange(false, std::memory_order_acq_rel)) {
                        if (lease->validity) {
                            lease->validity->store(false, std::memory_order_release);
                        }
                        lease->view = ExpertByteView();
                        lease->gpu_slot.reset();
                        lease->gpu_valid.reset();
                        lease->entry.reset();
                        lease->pins = 0;
                    }
                }
            }
            leases.clear();
            for (auto& item : cache) {
                auto& entry = item.second;
                if (entry->gpu_state == GpuState::ready && entry->gpu_slot.has_value()) {
                    if (entry->gpu_valid) {
                        entry->gpu_valid->store(false, std::memory_order_release);
                    }
                    released.push_back(ReleasedSlot{entry->key, *entry->gpu_slot});
                }
                entry->changed.notify_all();
            }
            cache.clear();
            tasks.clear();
            generations.clear();
            metrics_value.current_ram_bytes = 0;
            metrics_value.cache_entries = 0;
            metrics_value.active_leases = 0;
            metrics_value.active_generations = 0;
            shutdown_complete = true;
        }
        try {
            release_slots(std::move(released), false);
        } catch (...) {
        }
    }

    ExpertPagerOptions options;
    const std::uint64_t owner_cookie;
    std::unordered_map<ExpertKey, RangeRecord, ExpertKeyHash> manifest;
    std::unordered_map<std::string, std::shared_ptr<SourceFile>> sources;

    mutable std::mutex mutex;
    std::condition_variable task_available;
    std::condition_variable direct_slot_available;
    std::size_t active_direct_loads = 0;
    std::unordered_map<ExpertKey, std::shared_ptr<CacheEntry>, ExpertKeyHash> cache;
    std::unordered_map<std::uint64_t, std::shared_ptr<GenerationState>> generations;
    std::unordered_map<std::uint64_t, std::weak_ptr<ExpertLeaseState>> leases;
    std::deque<std::shared_ptr<PrefetchTask>> tasks;
    std::vector<std::thread> workers;
    ExpertPagerMetrics metrics_value{};
    std::uint64_t next_generation_id = 1;
    std::uint64_t next_lease_id = 1;
    std::uint64_t touch_clock = 0;
    bool accepting = true;
    bool stopping = false;
    bool shutdown_complete = false;
};

ExpertLeaseState::~ExpertLeaseState() {
    if (auto state = owner.lock()) {
        state->release_lease(this);
    }
}

}  // namespace detail

ExpertPager::Lease::Lease(std::shared_ptr<detail::ExpertLeaseState> state) noexcept
    : state_(std::move(state)) {}

ExpertPager::Lease::~Lease() = default;

ExpertPager::Lease::Lease(Lease&& other) noexcept = default;

ExpertPager::Lease& ExpertPager::Lease::operator=(Lease&& other) noexcept = default;

bool ExpertPager::Lease::valid() const noexcept {
    return state_ != nullptr && state_->active.load(std::memory_order_acquire) &&
           state_->validity != nullptr &&
           state_->validity->load(std::memory_order_acquire);
}

ExpertKey ExpertPager::Lease::key() const {
    if (!state_) {
        throw ExpertTransactionError("empty expert lease has no key");
    }
    return state_->key;
}

std::uint64_t ExpertPager::Lease::generation() const noexcept {
    return state_ ? state_->generation : 0;
}

ExpertByteView ExpertPager::Lease::view() const {
    if (!state_) {
        return {};
    }
    const auto owner = state_->owner.lock();
    if (!owner) {
        return {};
    }
    std::lock_guard<std::mutex> guard(owner->mutex);
    if (!state_->active.load(std::memory_order_acquire)) {
        return {};
    }
    return state_->view;
}

std::optional<AbstractGpuSlot> ExpertPager::Lease::gpu_slot() const {
    if (!state_) {
        return std::nullopt;
    }
    const auto owner = state_->owner.lock();
    if (!owner) {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> guard(owner->mutex);
    if (!state_->active.load(std::memory_order_acquire) || !state_->gpu_slot.has_value() ||
        !state_->gpu_valid || !state_->gpu_valid->load(std::memory_order_acquire)) {
        return std::nullopt;
    }
    return state_->gpu_slot;
}

ExpertPager::ExpertPager(ExpertPagerOptions options, std::vector<ExpertRange> ranges)
    : state_(std::make_shared<detail::ExpertPagerState>(std::move(options),
                                                        std::move(ranges))) {
    state_->start_workers();
}

ExpertPager::~ExpertPager() {
    if (state_) {
        state_->shutdown();
    }
}

ExpertPager::Generation ExpertPager::begin_generation() {
    return state_->begin_generation();
}

ExpertPager::Lease ExpertPager::request(const Generation& generation,
                                        const ExpertKey& key,
                                        ExpertRequestOptions options) {
    return state_->request(generation, key, options);
}

std::future<ExpertPager::Lease> ExpertPager::prefetch(const Generation& generation,
                                                      const ExpertKey& key,
                                                      ExpertRequestOptions options) {
    return state_->prefetch(generation, key, options);
}

void ExpertPager::pin(Lease& lease) {
    state_->pin(lease);
}

void ExpertPager::unpin(Lease& lease) {
    state_->unpin(lease);
}

void ExpertPager::commit(const Generation& generation) {
    state_->commit(generation);
}

void ExpertPager::rollback(const Generation& generation) {
    state_->rollback(generation);
}

bool ExpertPager::contains(const ExpertKey& key) const {
    return state_->contains(key);
}

bool ExpertPager::discard_cached(const ExpertKey& key) {
    return state_->discard_cached(key);
}

ExpertPagerMetrics ExpertPager::metrics() const {
    return state_->metrics();
}

}  // namespace axiom::qwen4exp
