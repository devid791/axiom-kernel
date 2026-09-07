#include "axiom/qwen38_kv_tier.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr uint32_t kAbi = AXIOM_QWEN38_KV_TIER_ABI_VERSION;
constexpr uint32_t kAlignment = AXIOM_QWEN38_KV_TIER_HEADER_BYTES;
constexpr uint32_t kDefaultQueueDepth = 64u;
constexpr uint32_t kMaximumQueueDepth = 1024u;
constexpr uint64_t kKnownFlags =
        AXIOM_QWEN38_KV_TIER_FLAG_PREALLOCATE |
        AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING |
        AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_DIRECT;

bool checked_add(const uint64_t a, const uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool checked_mul(const uint64_t a, const uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool aligned_4096(const void *pointer) {
    return pointer && (reinterpret_cast<uintptr_t>(pointer) & (kAlignment - 1u)) == 0u;
}

uint64_t checksum64(const void *data, const size_t bytes) {
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

struct disk_header {
    char magic[8];
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t max_context;
    uint32_t page_tokens;
    uint32_t logical_pages;
    uint32_t target_layers;
    uint32_t dspark_layers;
    uint32_t reserved0;
    uint64_t target_page_bytes;
    uint64_t dspark_page_bytes;
    uint64_t target_payload_bytes;
    uint64_t dspark_payload_bytes;
    uint64_t file_bytes;
    uint64_t generation;
    uint64_t committed_tokens;
    uint64_t checksum;
    uint8_t reserved[3992];
};

static_assert(sizeof(disk_header) == AXIOM_QWEN38_KV_TIER_HEADER_BYTES,
              "Qwen3.8 KV tier header must be one direct-I/O block");

constexpr uint32_t kTransactionAbi = 1u;

struct transaction_journal_header {
    char magic[8];
    uint32_t abi_version;
    uint32_t header_bytes;
    uint32_t record_count;
    uint32_t logical_page;
    uint32_t base_committed_tokens;
    uint32_t target_layers;
    uint32_t dspark_layers;
    uint32_t target_page_bytes;
    uint32_t dspark_page_bytes;
    uint32_t reserved0;
    uint64_t tier_generation;
    uint64_t tier_file_bytes;
    uint64_t payload_bytes;
    uint64_t checksum;
    uint8_t reserved[4016];
};

static_assert(sizeof(transaction_journal_header) == kAlignment,
              "Qwen3.8 KV transaction header must be one filesystem block");

void fill_header(
        const axiom_qwen38_kv_tier_plan &plan,
        const uint64_t generation,
        const uint32_t committed_tokens,
        disk_header *out) {
    std::memset(out, 0, sizeof(*out));
    const char magic[8] = {'A', 'X', 'Q', '3', '8', 'K', 'V', '1'};
    std::memcpy(out->magic, magic, sizeof(magic));
    out->abi_version = kAbi;
    out->header_bytes = AXIOM_QWEN38_KV_TIER_HEADER_BYTES;
    out->max_context = plan.max_context;
    out->page_tokens = plan.page_tokens;
    out->logical_pages = plan.logical_pages;
    out->target_layers = plan.target_layers;
    out->dspark_layers = plan.dspark_layers;
    out->target_page_bytes = plan.target_page_bytes;
    out->dspark_page_bytes = plan.dspark_page_bytes;
    out->target_payload_bytes = plan.target_payload_bytes;
    out->dspark_payload_bytes = plan.dspark_payload_bytes;
    out->file_bytes = plan.file_bytes;
    out->generation = generation;
    out->committed_tokens = committed_tokens;
    out->checksum = 0u;
    out->checksum = checksum64(out, sizeof(*out));
}

[[maybe_unused]] bool valid_header(
        const disk_header &header, const axiom_qwen38_kv_tier_plan &plan) {
    const char magic[8] = {'A', 'X', 'Q', '3', '8', 'K', 'V', '1'};
    if (std::memcmp(header.magic, magic, sizeof(magic)) != 0 ||
        header.abi_version != kAbi ||
        header.header_bytes != AXIOM_QWEN38_KV_TIER_HEADER_BYTES ||
        header.max_context != plan.max_context ||
        header.page_tokens != plan.page_tokens ||
        header.logical_pages != plan.logical_pages ||
        header.target_layers != plan.target_layers ||
        header.dspark_layers != plan.dspark_layers ||
        header.target_page_bytes != plan.target_page_bytes ||
        header.dspark_page_bytes != plan.dspark_page_bytes ||
        header.target_payload_bytes != plan.target_payload_bytes ||
        header.dspark_payload_bytes != plan.dspark_payload_bytes ||
        header.file_bytes != plan.file_bytes ||
        header.committed_tokens > plan.max_context) {
        return false;
    }
    disk_header copy = header;
    const uint64_t expected = copy.checksum;
    copy.checksum = 0u;
    return checksum64(&copy, sizeof(copy)) == expected;
}

int exact_pread(const int fd, void *buffer, const size_t bytes, const uint64_t offset) {
    size_t done = 0u;
    while (done < bytes) {
        const ssize_t count = pread(
                fd, static_cast<uint8_t *>(buffer) + done, bytes - done,
                static_cast<off_t>(offset + done));
        if (count > 0) {
            done += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            return AXIOM_ERR_IO;
        }
    }
    return AXIOM_OK;
}

int exact_pwrite(const int fd, const void *buffer, const size_t bytes, const uint64_t offset) {
    size_t done = 0u;
    while (done < bytes) {
        const ssize_t count = pwrite(
                fd, static_cast<const uint8_t *>(buffer) + done, bytes - done,
                static_cast<off_t>(offset + done));
        if (count > 0) {
            done += static_cast<size_t>(count);
        } else if (count < 0 && errno == EINTR) {
            continue;
        } else {
            return AXIOM_ERR_IO;
        }
    }
    return AXIOM_OK;
}

bool parent_directory_sync(const std::string &path) {
    const size_t slash = path.rfind('/');
    const std::string directory = slash == std::string::npos
            ? "." : (slash == 0u ? "/" : path.substr(0u, slash));
    const int fd = open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (fd < 0) return false;
    const int rc = fsync(fd);
    close(fd);
    return rc == 0;
}

bool regular_private_file(const int fd, const uint64_t expected_bytes) {
    struct stat status{};
    return fd >= 0 && fstat(fd, &status) == 0 && S_ISREG(status.st_mode) &&
            status.st_nlink == 1 && status.st_uid == geteuid() &&
            status.st_size >= 0 &&
            static_cast<uint64_t>(status.st_size) == expected_bytes;
}

uint64_t journal_checksum(
        const transaction_journal_header &header,
        const std::vector<uint8_t> &payload) {
    transaction_journal_header copy = header;
    copy.checksum = 0u;
    uint64_t hash = checksum64(&copy, sizeof(copy));
    for (const uint8_t byte : payload) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

int page_offset_for_plan(
        const axiom_qwen38_kv_tier_plan &plan,
        const uint32_t family,
        const uint32_t layer,
        const uint32_t logical_page,
        uint64_t *out) {
    if (!out || logical_page >= plan.logical_pages) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t offset = AXIOM_QWEN38_KV_TIER_HEADER_BYTES;
    uint64_t layer_span = 0u;
    uint64_t within = 0u;
    if (family == AXIOM_QWEN38_KV_TIER_TARGET) {
        if (layer >= plan.target_layers ||
            !checked_mul(plan.logical_pages, plan.target_page_bytes, &layer_span) ||
            !checked_mul(layer, layer_span, &within) ||
            !checked_add(offset, within, &offset) ||
            !checked_mul(logical_page, plan.target_page_bytes, &within) ||
            !checked_add(offset, within, &offset)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    } else if (family == AXIOM_QWEN38_KV_TIER_DSPARK) {
        if (layer >= plan.dspark_layers ||
            !checked_add(offset, plan.target_payload_bytes, &offset) ||
            !checked_mul(plan.logical_pages, plan.dspark_page_bytes, &layer_span) ||
            !checked_mul(layer, layer_span, &within) ||
            !checked_add(offset, within, &offset) ||
            !checked_mul(logical_page, plan.dspark_page_bytes, &within) ||
            !checked_add(offset, within, &offset)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    } else {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = offset;
    return AXIOM_OK;
}

struct loaded_journal {
    transaction_journal_header header{};
    disk_header old_tier_header{};
    axiom_qwen38_kv_tier_plan plan{};
    std::vector<uint8_t> payload;
};

int load_transaction_journal(
        const char *path, loaded_journal *out, bool *exists) {
    if (exists) *exists = false;
    if (!path || !path[0] || !out || !exists) return AXIOM_ERR_INVALID_ARGUMENT;
    const int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT ? AXIOM_OK : AXIOM_ERR_IO;
    transaction_journal_header header{};
    int rc = exact_pread(fd, &header, sizeof(header), 0u);
    const char magic[8] = {'A', 'X', 'Q', '3', '8', 'T', 'X', '1'};
    if (rc == AXIOM_OK &&
        (std::memcmp(header.magic, magic, sizeof(magic)) != 0 ||
         header.abi_version != kTransactionAbi || header.header_bytes != kAlignment ||
         header.target_layers != AXIOM_QWEN38_KV_TIER_TARGET_LAYERS ||
         header.dspark_layers != AXIOM_QWEN38_KV_TIER_DSPARK_LAYERS ||
         header.target_page_bytes != AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES ||
         header.dspark_page_bytes != AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES ||
         header.record_count != (header.base_committed_tokens %
                                 AXIOM_QWEN38_KV_TIER_PAGE_TOKENS == 0u
                                 ? 0u : header.target_layers + header.dspark_layers))) {
        rc = AXIOM_ERR_IO;
    }
    uint64_t expected_payload = sizeof(disk_header);
    if (rc == AXIOM_OK) {
        uint64_t target_bytes = 0u;
        uint64_t dspark_bytes = 0u;
        if (header.record_count != 0u &&
            (!checked_mul(header.target_layers, header.target_page_bytes, &target_bytes) ||
             !checked_mul(header.dspark_layers, header.dspark_page_bytes, &dspark_bytes) ||
             !checked_add(expected_payload, target_bytes, &expected_payload) ||
             !checked_add(expected_payload, dspark_bytes, &expected_payload))) {
            rc = AXIOM_ERR_IO;
        }
    }
    if (rc == AXIOM_OK &&
        (header.payload_bytes != expected_payload ||
         header.payload_bytes > std::numeric_limits<size_t>::max() ||
         !regular_private_file(fd, sizeof(header) + header.payload_bytes))) {
        rc = AXIOM_ERR_IO;
    }
    std::vector<uint8_t> payload;
    if (rc == AXIOM_OK) {
        try {
            payload.resize(static_cast<size_t>(header.payload_bytes));
        } catch (...) {
            rc = AXIOM_ERR_BUDGET;
        }
    }
    if (rc == AXIOM_OK && !payload.empty()) {
        rc = exact_pread(fd, payload.data(), payload.size(), sizeof(header));
    }
    close(fd);
    if (rc != AXIOM_OK) return rc;
    if (header.checksum != journal_checksum(header, payload) ||
        payload.size() < sizeof(disk_header)) {
        return AXIOM_ERR_IO;
    }
    disk_header old_header{};
    std::memcpy(&old_header, payload.data(), sizeof(old_header));
    axiom_qwen38_kv_tier_plan plan{};
    plan.abi_version = kAbi;
    rc = axiom_qwen38_kv_tier_plan_get(old_header.max_context, 0u, &plan);
    if (rc != AXIOM_OK || !valid_header(old_header, plan) ||
        header.base_committed_tokens != old_header.committed_tokens ||
        header.tier_generation != old_header.generation ||
        header.tier_file_bytes != old_header.file_bytes ||
        header.logical_page != header.base_committed_tokens /
                AXIOM_QWEN38_KV_TIER_PAGE_TOKENS) {
        return AXIOM_ERR_IO;
    }
    out->header = header;
    out->old_tier_header = old_header;
    out->plan = plan;
    out->payload = std::move(payload);
    *exists = true;
    return AXIOM_OK;
}

int restore_transaction_fd(const int fd, const loaded_journal &journal) {
    if (!regular_private_file(fd, journal.plan.file_bytes)) return AXIOM_ERR_IO;
    size_t cursor = sizeof(disk_header);
    if (journal.header.record_count != 0u) {
        void *page = nullptr;
        if (posix_memalign(&page, kAlignment, AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES) != 0 ||
            !page) return AXIOM_ERR_BUDGET;
        int rc = AXIOM_OK;
        for (uint32_t family = AXIOM_QWEN38_KV_TIER_TARGET;
             family <= AXIOM_QWEN38_KV_TIER_DSPARK && rc == AXIOM_OK; ++family) {
            const uint32_t layers = family == AXIOM_QWEN38_KV_TIER_TARGET
                    ? journal.plan.target_layers : journal.plan.dspark_layers;
            const uint64_t bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
            for (uint32_t layer = 0u; layer < layers && rc == AXIOM_OK; ++layer) {
                if (cursor + bytes > journal.payload.size()) {
                    rc = AXIOM_ERR_IO;
                    break;
                }
                std::memcpy(page, journal.payload.data() + cursor, static_cast<size_t>(bytes));
                uint64_t offset = 0u;
                rc = page_offset_for_plan(
                        journal.plan, family, layer, journal.header.logical_page, &offset);
                if (rc == AXIOM_OK) rc = exact_pwrite(fd, page, static_cast<size_t>(bytes), offset);
                cursor += static_cast<size_t>(bytes);
            }
        }
        std::free(page);
        if (rc != AXIOM_OK || cursor != journal.payload.size()) return AXIOM_ERR_IO;
        if (fdatasync(fd) != 0) return AXIOM_ERR_IO;
    }
    void *raw = nullptr;
    if (posix_memalign(&raw, kAlignment, sizeof(disk_header)) != 0 || !raw) {
        return AXIOM_ERR_BUDGET;
    }
    std::memcpy(raw, &journal.old_tier_header, sizeof(disk_header));
    const int rc = exact_pwrite(fd, raw, sizeof(disk_header), 0u);
    std::free(raw);
    if (rc != AXIOM_OK || fdatasync(fd) != 0) return AXIOM_ERR_IO;
    return AXIOM_OK;
}

int discard_transaction_path(const char *path) {
    if (!path || !path[0]) return AXIOM_ERR_INVALID_ARGUMENT;
    if (unlink(path) != 0 && errno != ENOENT) return AXIOM_ERR_IO;
    return parent_directory_sync(path) ? AXIOM_OK : AXIOM_ERR_IO;
}

struct pending_slot {
    bool active = false;
    uint32_t operation = 0u;
    uint64_t expected_bytes = 0u;
    uint64_t user_data = 0u;
};

#if defined(__linux__)
struct raw_ring {
    int fd = -1;
    void *sq_mapping = MAP_FAILED;
    size_t sq_mapping_bytes = 0u;
    void *cq_mapping = MAP_FAILED;
    size_t cq_mapping_bytes = 0u;
    io_uring_sqe *sqes = nullptr;
    size_t sqes_bytes = 0u;
    uint32_t *sq_head = nullptr;
    uint32_t *sq_tail = nullptr;
    uint32_t *sq_mask = nullptr;
    uint32_t *sq_entries = nullptr;
    uint32_t *sq_array = nullptr;
    uint32_t *cq_head = nullptr;
    uint32_t *cq_tail = nullptr;
    uint32_t *cq_mask = nullptr;
    io_uring_cqe *cqes = nullptr;
};

void ring_destroy(raw_ring *ring) {
    if (!ring) return;
    if (ring->sqes && ring->sqes != MAP_FAILED) munmap(ring->sqes, ring->sqes_bytes);
    if (ring->cq_mapping != MAP_FAILED && ring->cq_mapping != ring->sq_mapping) {
        munmap(ring->cq_mapping, ring->cq_mapping_bytes);
    }
    if (ring->sq_mapping != MAP_FAILED) munmap(ring->sq_mapping, ring->sq_mapping_bytes);
    if (ring->fd >= 0) close(ring->fd);
    *ring = raw_ring{};
}

int ring_create(const uint32_t entries, raw_ring *ring) {
    if (!ring || entries == 0u) return AXIOM_ERR_INVALID_ARGUMENT;
    io_uring_params params{};
    const int fd = static_cast<int>(syscall(__NR_io_uring_setup, entries, &params));
    if (fd < 0) return AXIOM_ERR_UNSUPPORTED_BACKEND;

    ring->fd = fd;
    ring->sq_mapping_bytes = params.sq_off.array + params.sq_entries * sizeof(uint32_t);
    ring->cq_mapping_bytes = params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe);
    if ((params.features & IORING_FEAT_SINGLE_MMAP) != 0u) {
        ring->sq_mapping_bytes = std::max(ring->sq_mapping_bytes, ring->cq_mapping_bytes);
        ring->cq_mapping_bytes = ring->sq_mapping_bytes;
    }
    ring->sq_mapping = mmap(
            nullptr, ring->sq_mapping_bytes, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (ring->sq_mapping == MAP_FAILED) {
        ring_destroy(ring);
        return AXIOM_ERR_RUNTIME;
    }
    if ((params.features & IORING_FEAT_SINGLE_MMAP) != 0u) {
        ring->cq_mapping = ring->sq_mapping;
    } else {
        ring->cq_mapping = mmap(
                nullptr, ring->cq_mapping_bytes, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (ring->cq_mapping == MAP_FAILED) {
            ring_destroy(ring);
            return AXIOM_ERR_RUNTIME;
        }
    }
    ring->sqes_bytes = params.sq_entries * sizeof(io_uring_sqe);
    ring->sqes = static_cast<io_uring_sqe *>(mmap(
            nullptr, ring->sqes_bytes, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES));
    if (ring->sqes == MAP_FAILED) {
        ring_destroy(ring);
        return AXIOM_ERR_RUNTIME;
    }

    uint8_t *sq = static_cast<uint8_t *>(ring->sq_mapping);
    uint8_t *cq = static_cast<uint8_t *>(ring->cq_mapping);
    ring->sq_head = reinterpret_cast<uint32_t *>(sq + params.sq_off.head);
    ring->sq_tail = reinterpret_cast<uint32_t *>(sq + params.sq_off.tail);
    ring->sq_mask = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_mask);
    ring->sq_entries = reinterpret_cast<uint32_t *>(sq + params.sq_off.ring_entries);
    ring->sq_array = reinterpret_cast<uint32_t *>(sq + params.sq_off.array);
    ring->cq_head = reinterpret_cast<uint32_t *>(cq + params.cq_off.head);
    ring->cq_tail = reinterpret_cast<uint32_t *>(cq + params.cq_off.tail);
    ring->cq_mask = reinterpret_cast<uint32_t *>(cq + params.cq_off.ring_mask);
    ring->cqes = reinterpret_cast<io_uring_cqe *>(cq + params.cq_off.cqes);
    return AXIOM_OK;
}
#endif

struct sync_completion {
    uint32_t operation = 0u;
    int32_t result = AXIOM_ERR_IO;
    uint64_t bytes = 0u;
    uint64_t user_data = 0u;
};

}  // namespace

struct axiom_qwen38_kv_tier {
    int fd = -1;
    uint64_t flags = 0u;
    axiom_qwen38_kv_tier_plan plan{};
    uint32_t backend = AXIOM_QWEN38_KV_TIER_BACKEND_SYNC_DIRECT;
    uint32_t queue_depth = 0u;
    uint32_t outstanding = 0u;
    uint32_t committed_tokens = 0u;
    uint64_t generation = 0u;
    uint64_t submitted_reads = 0u;
    uint64_t submitted_writes = 0u;
    uint64_t completed_bytes = 0u;
    uint64_t io_errors = 0u;
    std::vector<pending_slot> slots;
    std::deque<sync_completion> sync_completions;
    mutable std::mutex lock;
#if defined(__linux__)
    raw_ring ring{};
#endif
};

extern "C" int axiom_qwen38_kv_tier_plan_get(
        const uint32_t max_context,
        const uint32_t hot_pages,
        axiom_qwen38_kv_tier_plan *out) {
    if (!out || out->abi_version != kAbi || max_context == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->max_context = max_context;
    out->page_tokens = AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    out->logical_pages = static_cast<uint32_t>(
            (static_cast<uint64_t>(max_context) + out->page_tokens - 1u) / out->page_tokens);
    if (hot_pages > out->logical_pages) return AXIOM_ERR_INVALID_ARGUMENT;
    out->hot_pages = hot_pages;
    out->hot_tokens = hot_pages * out->page_tokens;
    out->target_layers = AXIOM_QWEN38_KV_TIER_TARGET_LAYERS;
    out->dspark_layers = AXIOM_QWEN38_KV_TIER_DSPARK_LAYERS;
    out->target_bytes_per_token = AXIOM_QWEN38_KV_TIER_TARGET_BYTES_PER_TOKEN;
    out->dspark_bytes_per_token = AXIOM_QWEN38_KV_TIER_DSPARK_BYTES_PER_TOKEN;
    out->combined_bytes_per_token = AXIOM_QWEN38_KV_TIER_COMBINED_BYTES_PER_TOKEN;
    out->target_page_bytes = AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES;
    out->dspark_page_bytes = AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES;
    out->combined_page_bytes = AXIOM_QWEN38_KV_TIER_COMBINED_PAGE_BYTES;

    uint64_t target_layer_bytes = 0u;
    uint64_t dspark_layer_bytes = 0u;
    if (!checked_mul(out->logical_pages, out->target_page_bytes, &target_layer_bytes) ||
        !checked_mul(target_layer_bytes, out->target_layers, &out->target_payload_bytes) ||
        !checked_mul(out->logical_pages, out->dspark_page_bytes, &dspark_layer_bytes) ||
        !checked_mul(dspark_layer_bytes, out->dspark_layers, &out->dspark_payload_bytes) ||
        !checked_add(out->target_payload_bytes, out->dspark_payload_bytes, &out->payload_bytes) ||
        !checked_add(AXIOM_QWEN38_KV_TIER_HEADER_BYTES, out->payload_bytes, &out->file_bytes) ||
        !checked_mul(out->hot_pages, out->combined_page_bytes, &out->hot_device_bytes)) {
        return AXIOM_ERR_BUDGET;
    }
    return AXIOM_OK;
}

extern "C" uint64_t axiom_qwen38_kv_tier_family_page_bytes(const uint32_t family) {
    if (family == AXIOM_QWEN38_KV_TIER_TARGET) {
        return AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES;
    }
    if (family == AXIOM_QWEN38_KV_TIER_DSPARK) {
        return AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES;
    }
    return 0u;
}

extern "C" int axiom_qwen38_kv_tier_page_offset(
        const axiom_qwen38_kv_tier *tier,
        const uint32_t family,
        const uint32_t layer,
        const uint32_t logical_page,
        uint64_t *out_offset) {
    if (!tier || !out_offset || logical_page >= tier->plan.logical_pages) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t layer_span = 0u;
    uint64_t offset = AXIOM_QWEN38_KV_TIER_HEADER_BYTES;
    uint64_t within = 0u;
    if (family == AXIOM_QWEN38_KV_TIER_TARGET) {
        if (layer >= tier->plan.target_layers ||
            !checked_mul(tier->plan.logical_pages, tier->plan.target_page_bytes, &layer_span) ||
            !checked_mul(layer, layer_span, &within) ||
            !checked_add(offset, within, &offset) ||
            !checked_mul(logical_page, tier->plan.target_page_bytes, &within) ||
            !checked_add(offset, within, &offset)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    } else if (family == AXIOM_QWEN38_KV_TIER_DSPARK) {
        if (layer >= tier->plan.dspark_layers ||
            !checked_add(offset, tier->plan.target_payload_bytes, &offset) ||
            !checked_mul(tier->plan.logical_pages, tier->plan.dspark_page_bytes, &layer_span) ||
            !checked_mul(layer, layer_span, &within) ||
            !checked_add(offset, within, &offset) ||
            !checked_mul(logical_page, tier->plan.dspark_page_bytes, &within) ||
            !checked_add(offset, within, &offset)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    } else {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_offset = offset;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_kv_tier_host_page_alloc(const uint64_t bytes, void **out) {
    if (out) *out = nullptr;
    if (!out || (bytes != AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES &&
                 bytes != AXIOM_QWEN38_KV_TIER_DSPARK_PAGE_BYTES)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (bytes > std::numeric_limits<size_t>::max()) return AXIOM_ERR_BUDGET;
    void *page = nullptr;
    const int rc = posix_memalign(&page, kAlignment, static_cast<size_t>(bytes));
    if (rc != 0 || !page) return rc == ENOMEM ? AXIOM_ERR_BUDGET : AXIOM_ERR_RUNTIME;
    std::memset(page, 0, static_cast<size_t>(bytes));
    *out = page;
    return AXIOM_OK;
}

extern "C" void axiom_qwen38_kv_tier_host_page_free(void *page) {
    std::free(page);
}

namespace {

int write_header(axiom_qwen38_kv_tier *tier) {
    void *raw = nullptr;
    if (posix_memalign(&raw, kAlignment, sizeof(disk_header)) != 0 || !raw) {
        return AXIOM_ERR_BUDGET;
    }
    disk_header *header = static_cast<disk_header *>(raw);
    fill_header(tier->plan, tier->generation, tier->committed_tokens, header);
    const int rc = exact_pwrite(tier->fd, header, sizeof(*header), 0u);
    std::free(raw);
    if (rc != AXIOM_OK) return rc;
#if defined(__linux__)
    return fdatasync(tier->fd) == 0 ? AXIOM_OK : AXIOM_ERR_IO;
#else
    return fsync(tier->fd) == 0 ? AXIOM_OK : AXIOM_ERR_IO;
#endif
}

[[maybe_unused]] int read_header(const int fd, disk_header *out) {
    void *raw = nullptr;
    if (!out || posix_memalign(&raw, kAlignment, sizeof(disk_header)) != 0 || !raw) {
        return AXIOM_ERR_BUDGET;
    }
    const int rc = exact_pread(fd, raw, sizeof(disk_header), 0u);
    if (rc == AXIOM_OK) std::memcpy(out, raw, sizeof(*out));
    std::free(raw);
    return rc;
}

}  // namespace

extern "C" int axiom_qwen38_kv_tier_create(
        const axiom_qwen38_kv_tier_config *config,
        axiom_qwen38_kv_tier **out) {
    if (out) *out = nullptr;
    if (!config || config->abi_version != kAbi || !config->path || !config->path[0] || !out ||
        config->max_context == 0u || (config->flags & ~kKnownFlags) != 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t depth = config->queue_depth == 0u ? kDefaultQueueDepth : config->queue_depth;
    if (depth == 0u || depth > kMaximumQueueDepth) return AXIOM_ERR_INVALID_ARGUMENT;

    axiom_qwen38_kv_tier_plan plan{};
    plan.abi_version = kAbi;
    int rc = axiom_qwen38_kv_tier_plan_get(config->max_context, config->hot_pages, &plan);
    if (rc != AXIOM_OK || plan.file_bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
        return rc == AXIOM_OK ? AXIOM_ERR_BUDGET : rc;
    }

#if !defined(O_DIRECT)
    return AXIOM_ERR_UNSUPPORTED_BACKEND;
#else
    const int fd = open(config->path, O_RDWR | O_CREAT | O_CLOEXEC | O_DIRECT, 0600);
    if (fd < 0) return AXIOM_ERR_IO;
    struct stat status{};
    if (fstat(fd, &status) != 0 || status.st_size < 0) {
        close(fd);
        return AXIOM_ERR_IO;
    }

    axiom_qwen38_kv_tier *tier = new (std::nothrow) axiom_qwen38_kv_tier();
    if (!tier) {
        close(fd);
        return AXIOM_ERR_BUDGET;
    }
    tier->fd = fd;
    tier->flags = config->flags;
    tier->plan = plan;
    tier->queue_depth = depth;
    tier->slots.resize(depth);

    if (status.st_size == 0) {
        if ((config->flags & AXIOM_QWEN38_KV_TIER_FLAG_PREALLOCATE) != 0u) {
#if defined(__linux__)
            const int allocation = posix_fallocate(fd, 0, static_cast<off_t>(plan.file_bytes));
            rc = allocation == 0 ? AXIOM_OK : AXIOM_ERR_IO;
#else
            rc = AXIOM_ERR_UNSUPPORTED_BACKEND;
#endif
        } else {
            rc = ftruncate(fd, static_cast<off_t>(plan.file_bytes)) == 0 ? AXIOM_OK : AXIOM_ERR_IO;
        }
        tier->generation = 1u;
        tier->committed_tokens = 0u;
        if (rc == AXIOM_OK) rc = write_header(tier);
    } else {
        if (static_cast<uint64_t>(status.st_size) != plan.file_bytes) {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        } else {
            disk_header header{};
            rc = read_header(fd, &header);
            if (rc == AXIOM_OK && !valid_header(header, plan)) rc = AXIOM_ERR_INVALID_ARGUMENT;
            if (rc == AXIOM_OK) {
                tier->generation = header.generation;
                tier->committed_tokens = static_cast<uint32_t>(header.committed_tokens);
            }
        }
    }

#if defined(__linux__)
    if (rc == AXIOM_OK) {
        const int ring_rc = ring_create(depth, &tier->ring);
        if (ring_rc == AXIOM_OK) {
            tier->backend = AXIOM_QWEN38_KV_TIER_BACKEND_IO_URING_DIRECT;
        } else if ((config->flags & AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING) != 0u) {
            rc = ring_rc;
        }
    }
#else
    if ((config->flags & AXIOM_QWEN38_KV_TIER_FLAG_REQUIRE_IO_URING) != 0u) {
        rc = AXIOM_ERR_UNSUPPORTED_BACKEND;
    }
#endif
    if (rc != AXIOM_OK) {
        axiom_qwen38_kv_tier_destroy(tier);
        return rc;
    }
    *out = tier;
    return AXIOM_OK;
#endif
}

extern "C" void axiom_qwen38_kv_tier_destroy(axiom_qwen38_kv_tier *tier) {
    if (!tier) return;
#if defined(__linux__)
    ring_destroy(&tier->ring);
#endif
    if (tier->fd >= 0) close(tier->fd);
    tier->fd = -1;
    delete tier;
}

extern "C" int axiom_qwen38_kv_tier_info_get(
        const axiom_qwen38_kv_tier *tier,
        axiom_qwen38_kv_tier_info *out) {
    if (!tier || !out || out->abi_version != kAbi) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(tier->lock);
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->backend = tier->backend;
    out->direct_io = 1u;
    out->preallocated =
            (tier->flags & AXIOM_QWEN38_KV_TIER_FLAG_PREALLOCATE) != 0u ? 1u : 0u;
    out->queue_depth = tier->queue_depth;
    out->outstanding = tier->outstanding;
    out->committed_tokens = tier->committed_tokens;
    out->plan = tier->plan;
    out->submitted_reads = tier->submitted_reads;
    out->submitted_writes = tier->submitted_writes;
    out->completed_bytes = tier->completed_bytes;
    out->io_errors = tier->io_errors;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_kv_tier_submit(
        axiom_qwen38_kv_tier *tier,
        const axiom_qwen38_kv_tier_request *request) {
    if (!tier || !request || request->abi_version != kAbi ||
        (request->operation != AXIOM_QWEN38_KV_TIER_READ &&
         request->operation != AXIOM_QWEN38_KV_TIER_WRITE) ||
        !aligned_4096(request->host_page)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t expected = axiom_qwen38_kv_tier_family_page_bytes(request->family);
    if (expected == 0u || request->host_page_bytes != expected ||
        expected > std::numeric_limits<uint32_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t offset = 0u;
    int rc = axiom_qwen38_kv_tier_page_offset(
            tier, request->family, request->layer, request->logical_page, &offset);
    if (rc != AXIOM_OK) return rc;

    std::lock_guard<std::mutex> guard(tier->lock);
    if (tier->outstanding >= tier->queue_depth) return AXIOM_ERR_BUDGET;
    if (request->operation == AXIOM_QWEN38_KV_TIER_READ) {
        ++tier->submitted_reads;
    } else {
        ++tier->submitted_writes;
    }

#if defined(__linux__)
    if (tier->backend == AXIOM_QWEN38_KV_TIER_BACKEND_IO_URING_DIRECT) {
        uint32_t slot_index = tier->queue_depth;
        for (uint32_t index = 0u; index < tier->queue_depth; ++index) {
            if (!tier->slots[index].active) {
                slot_index = index;
                break;
            }
        }
        if (slot_index == tier->queue_depth) return AXIOM_ERR_BUDGET;
        const uint32_t head = __atomic_load_n(tier->ring.sq_head, __ATOMIC_ACQUIRE);
        const uint32_t tail = __atomic_load_n(tier->ring.sq_tail, __ATOMIC_RELAXED);
        if (tail - head >= *tier->ring.sq_entries) return AXIOM_ERR_BUDGET;
        const uint32_t sq_index = tail & *tier->ring.sq_mask;
        io_uring_sqe *sqe = &tier->ring.sqes[sq_index];
        std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = request->operation == AXIOM_QWEN38_KV_TIER_READ
                ? IORING_OP_READ : IORING_OP_WRITE;
        sqe->fd = tier->fd;
        sqe->off = offset;
        sqe->addr = reinterpret_cast<uint64_t>(request->host_page);
        sqe->len = static_cast<uint32_t>(expected);
        sqe->user_data = static_cast<uint64_t>(slot_index) + 1u;
        tier->ring.sq_array[sq_index] = sq_index;
        tier->slots[slot_index] = pending_slot{
                true, request->operation, expected, request->user_data};
        __atomic_store_n(tier->ring.sq_tail, tail + 1u, __ATOMIC_RELEASE);
        const int submitted = static_cast<int>(syscall(
                __NR_io_uring_enter, tier->ring.fd, 1u, 0u, 0u, nullptr, 0u));
        if (submitted != 1) {
            tier->slots[slot_index] = pending_slot{};
            return AXIOM_ERR_IO;
        }
        ++tier->outstanding;
        return AXIOM_OK;
    }
#endif

    const int io_rc = request->operation == AXIOM_QWEN38_KV_TIER_READ
            ? exact_pread(tier->fd, request->host_page, static_cast<size_t>(expected), offset)
            : exact_pwrite(tier->fd, request->host_page, static_cast<size_t>(expected), offset);
    tier->sync_completions.push_back(sync_completion{
            request->operation, io_rc, io_rc == AXIOM_OK ? expected : 0u, request->user_data});
    ++tier->outstanding;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_kv_tier_wait(
        axiom_qwen38_kv_tier *tier,
        const uint32_t min_complete,
        axiom_qwen38_kv_tier_completion *completions,
        const uint32_t completion_capacity,
        uint32_t *out_count) {
    if (out_count) *out_count = 0u;
    if (!tier || !out_count || min_complete > completion_capacity ||
        (completion_capacity != 0u && !completions)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> guard(tier->lock);
    if (min_complete > tier->outstanding) return AXIOM_ERR_INVALID_ARGUMENT;
    uint32_t count = 0u;

#if defined(__linux__)
    if (tier->backend == AXIOM_QWEN38_KV_TIER_BACKEND_IO_URING_DIRECT) {
        if (min_complete != 0u) {
            const int entered = static_cast<int>(syscall(
                    __NR_io_uring_enter, tier->ring.fd, 0u, min_complete,
                    IORING_ENTER_GETEVENTS, nullptr, 0u));
            if (entered < 0 && errno != EINTR) return AXIOM_ERR_IO;
        }
        uint32_t head = __atomic_load_n(tier->ring.cq_head, __ATOMIC_RELAXED);
        const uint32_t tail = __atomic_load_n(tier->ring.cq_tail, __ATOMIC_ACQUIRE);
        while (head != tail && count < completion_capacity) {
            const io_uring_cqe &cqe = tier->ring.cqes[head & *tier->ring.cq_mask];
            if (cqe.user_data == 0u || cqe.user_data > tier->slots.size()) {
                return AXIOM_ERR_RUNTIME;
            }
            const uint32_t slot_index = static_cast<uint32_t>(cqe.user_data - 1u);
            pending_slot &slot = tier->slots[slot_index];
            if (!slot.active) return AXIOM_ERR_RUNTIME;
            const bool exact = cqe.res >= 0 && static_cast<uint64_t>(cqe.res) == slot.expected_bytes;
            axiom_qwen38_kv_tier_completion &completion = completions[count++];
            std::memset(&completion, 0, sizeof(completion));
            completion.abi_version = kAbi;
            completion.operation = slot.operation;
            completion.result = exact ? AXIOM_OK : AXIOM_ERR_IO;
            completion.transferred_bytes = cqe.res > 0 ? static_cast<uint64_t>(cqe.res) : 0u;
            completion.user_data = slot.user_data;
            if (exact) {
                tier->completed_bytes += completion.transferred_bytes;
            } else {
                ++tier->io_errors;
            }
            slot = pending_slot{};
            --tier->outstanding;
            ++head;
        }
        __atomic_store_n(tier->ring.cq_head, head, __ATOMIC_RELEASE);
    } else
#endif
    {
        while (!tier->sync_completions.empty() && count < completion_capacity) {
            const sync_completion ready = tier->sync_completions.front();
            tier->sync_completions.pop_front();
            axiom_qwen38_kv_tier_completion &completion = completions[count++];
            std::memset(&completion, 0, sizeof(completion));
            completion.abi_version = kAbi;
            completion.operation = ready.operation;
            completion.result = ready.result;
            completion.transferred_bytes = ready.bytes;
            completion.user_data = ready.user_data;
            if (ready.result == AXIOM_OK) {
                tier->completed_bytes += ready.bytes;
            } else {
                ++tier->io_errors;
            }
            --tier->outstanding;
        }
    }
    if (count < min_complete) return AXIOM_ERR_IO;
    *out_count = count;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_kv_tier_commit(
        axiom_qwen38_kv_tier *tier,
        const uint32_t committed_tokens) {
    if (!tier || committed_tokens > tier->plan.max_context) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(tier->lock);
    if (tier->outstanding != 0u || committed_tokens < tier->committed_tokens) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t previous = tier->committed_tokens;
    tier->committed_tokens = committed_tokens;
    const int rc = write_header(tier);
    if (rc != AXIOM_OK) tier->committed_tokens = previous;
    return rc;
}

extern "C" int axiom_qwen38_kv_tier_reset(axiom_qwen38_kv_tier *tier) {
    if (!tier) return AXIOM_ERR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> guard(tier->lock);
    if (tier->outstanding != 0u || tier->generation == std::numeric_limits<uint64_t>::max()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t previous_generation = tier->generation;
    const uint32_t previous_tokens = tier->committed_tokens;
    ++tier->generation;
    tier->committed_tokens = 0u;
    const int rc = write_header(tier);
    if (rc != AXIOM_OK) {
        tier->generation = previous_generation;
        tier->committed_tokens = previous_tokens;
    }
    return rc;
}

extern "C" int axiom_qwen38_kv_tier_transaction_begin(
        axiom_qwen38_kv_tier *tier,
        const char *journal_path,
        const uint32_t base_committed_tokens) {
    if (!tier || !journal_path || !journal_path[0] || base_committed_tokens == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> guard(tier->lock);
    if (tier->outstanding != 0u || base_committed_tokens != tier->committed_tokens ||
        base_committed_tokens > tier->plan.max_context) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    struct stat existing{};
    if (lstat(journal_path, &existing) == 0 || errno != ENOENT) {
        return AXIOM_ERR_IO;
    }

    disk_header old_header{};
    int rc = read_header(tier->fd, &old_header);
    if (rc != AXIOM_OK || !valid_header(old_header, tier->plan) ||
        old_header.generation != tier->generation ||
        old_header.committed_tokens != base_committed_tokens) {
        return AXIOM_ERR_IO;
    }

    const bool partial = base_committed_tokens % AXIOM_QWEN38_KV_TIER_PAGE_TOKENS != 0u;
    uint64_t payload_bytes = sizeof(disk_header);
    if (partial &&
        (!checked_add(payload_bytes, tier->plan.target_layers * tier->plan.target_page_bytes,
                      &payload_bytes) ||
         !checked_add(payload_bytes, tier->plan.dspark_layers * tier->plan.dspark_page_bytes,
                      &payload_bytes))) {
        return AXIOM_ERR_BUDGET;
    }
    if (payload_bytes > std::numeric_limits<size_t>::max()) return AXIOM_ERR_BUDGET;
    std::vector<uint8_t> payload;
    try {
        payload.resize(static_cast<size_t>(payload_bytes));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    std::memcpy(payload.data(), &old_header, sizeof(old_header));
    size_t cursor = sizeof(old_header);
    if (partial) {
        void *page = nullptr;
        if (posix_memalign(&page, kAlignment, AXIOM_QWEN38_KV_TIER_TARGET_PAGE_BYTES) != 0 ||
            !page) return AXIOM_ERR_BUDGET;
        const uint32_t logical_page =
                base_committed_tokens / AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
        for (uint32_t family = AXIOM_QWEN38_KV_TIER_TARGET;
             family <= AXIOM_QWEN38_KV_TIER_DSPARK && rc == AXIOM_OK; ++family) {
            const uint32_t layers = family == AXIOM_QWEN38_KV_TIER_TARGET
                    ? tier->plan.target_layers : tier->plan.dspark_layers;
            const uint64_t bytes = axiom_qwen38_kv_tier_family_page_bytes(family);
            for (uint32_t layer = 0u; layer < layers && rc == AXIOM_OK; ++layer) {
                uint64_t offset = 0u;
                rc = page_offset_for_plan(tier->plan, family, layer, logical_page, &offset);
                if (rc == AXIOM_OK) {
                    rc = exact_pread(tier->fd, page, static_cast<size_t>(bytes), offset);
                }
                if (rc == AXIOM_OK) {
                    std::memcpy(payload.data() + cursor, page, static_cast<size_t>(bytes));
                    cursor += static_cast<size_t>(bytes);
                }
            }
        }
        std::free(page);
    }
    if (rc != AXIOM_OK || cursor != payload.size()) return AXIOM_ERR_IO;

    transaction_journal_header header{};
    const char magic[8] = {'A', 'X', 'Q', '3', '8', 'T', 'X', '1'};
    std::memcpy(header.magic, magic, sizeof(magic));
    header.abi_version = kTransactionAbi;
    header.header_bytes = sizeof(header);
    header.record_count = partial
            ? tier->plan.target_layers + tier->plan.dspark_layers : 0u;
    header.logical_page =
            base_committed_tokens / AXIOM_QWEN38_KV_TIER_PAGE_TOKENS;
    header.base_committed_tokens = base_committed_tokens;
    header.target_layers = tier->plan.target_layers;
    header.dspark_layers = tier->plan.dspark_layers;
    header.target_page_bytes = static_cast<uint32_t>(tier->plan.target_page_bytes);
    header.dspark_page_bytes = static_cast<uint32_t>(tier->plan.dspark_page_bytes);
    header.tier_generation = tier->generation;
    header.tier_file_bytes = tier->plan.file_bytes;
    header.payload_bytes = payload.size();
    header.checksum = journal_checksum(header, payload);

    const std::string temporary = std::string(journal_path) + ".tmp";
    if (unlink(temporary.c_str()) != 0 && errno != ENOENT) return AXIOM_ERR_IO;
    const int fd = open(
            temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
            0600);
    if (fd < 0) return AXIOM_ERR_IO;
    rc = exact_pwrite(fd, &header, sizeof(header), 0u);
    if (rc == AXIOM_OK) {
        rc = exact_pwrite(fd, payload.data(), payload.size(), sizeof(header));
    }
    if (rc == AXIOM_OK && fdatasync(fd) != 0) rc = AXIOM_ERR_IO;
    if (close(fd) != 0 && rc == AXIOM_OK) rc = AXIOM_ERR_IO;
    if (rc == AXIOM_OK && rename(temporary.c_str(), journal_path) != 0) {
        rc = AXIOM_ERR_IO;
    }
    if (rc == AXIOM_OK && !parent_directory_sync(journal_path)) rc = AXIOM_ERR_IO;
    if (rc != AXIOM_OK) (void)unlink(temporary.c_str());
    return rc;
}

extern "C" int axiom_qwen38_kv_tier_transaction_rollback(
        axiom_qwen38_kv_tier *tier,
        const char *journal_path) {
    if (!tier || !journal_path || !journal_path[0]) return AXIOM_ERR_INVALID_ARGUMENT;
    loaded_journal journal;
    bool exists = false;
    int rc = load_transaction_journal(journal_path, &journal, &exists);
    if (rc != AXIOM_OK || !exists) return rc;
    std::lock_guard<std::mutex> guard(tier->lock);
    if (tier->outstanding != 0u || tier->plan.file_bytes != journal.plan.file_bytes ||
        tier->generation != journal.old_tier_header.generation) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    rc = restore_transaction_fd(tier->fd, journal);
    if (rc == AXIOM_OK) {
        tier->generation = journal.old_tier_header.generation;
        tier->committed_tokens =
                static_cast<uint32_t>(journal.old_tier_header.committed_tokens);
        rc = discard_transaction_path(journal_path);
    }
    return rc;
}

extern "C" int axiom_qwen38_kv_tier_transaction_recover_file(
        const char *tier_path,
        const char *journal_path,
        const uint32_t manifest_committed_tokens) {
    if (!tier_path || !tier_path[0] || !journal_path || !journal_path[0]) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    loaded_journal journal;
    bool exists = false;
    int rc = load_transaction_journal(journal_path, &journal, &exists);
    if (rc != AXIOM_OK || !exists) return rc;
#if !defined(O_DIRECT)
    return AXIOM_ERR_UNSUPPORTED_BACKEND;
#else
    const int fd = open(tier_path, O_RDWR | O_CLOEXEC | O_DIRECT | O_NOFOLLOW);
    if (fd < 0 || !regular_private_file(fd, journal.plan.file_bytes)) {
        if (fd >= 0) close(fd);
        return AXIOM_ERR_IO;
    }
    disk_header current{};
    const int read_rc = read_header(fd, &current);
    const bool current_valid = read_rc == AXIOM_OK && valid_header(current, journal.plan) &&
            current.generation == journal.old_tier_header.generation;
    if (manifest_committed_tokens == journal.header.base_committed_tokens) {
        rc = restore_transaction_fd(fd, journal);
    } else if (current_valid &&
               current.committed_tokens == manifest_committed_tokens) {
        rc = AXIOM_OK;
    } else {
        /* The manifest and tier do not identify either side of the atomic
         * transition. Keep the journal intact and fail closed. */
        rc = AXIOM_ERR_IO;
    }
    if (close(fd) != 0 && rc == AXIOM_OK) rc = AXIOM_ERR_IO;
    if (rc == AXIOM_OK) rc = discard_transaction_path(journal_path);
    return rc;
#endif
}

extern "C" int axiom_qwen38_kv_tier_transaction_discard(
        const char *journal_path) {
    return discard_transaction_path(journal_path);
}
