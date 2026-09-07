#include "axiom/qwen4exp/expert_pager.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <linux/stat.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

#ifdef AXIOM_EXPERT_PAGER_TEST_WRAP_PREAD
// Optional link-time fault injection, confined to this test executable.
static std::atomic<int> direct_fault{0};
static std::atomic<int> mutation_fd{-1};
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void* data, size_t size, off_t offset) {
    const int flags = ::fcntl(fd, F_GETFL);
    const int fault = flags >= 0 && (flags & O_DIRECT) ? direct_fault.exchange(0) : 0;
    if (fault == 1 || fault == 2) {
        errno = fault == 1 ? EINTR : EINVAL;
        return -1;
    }
    if (fault == 5) return 0;
    const ssize_t count = __real_pread(fd, data, size, offset);
    if (fault == 3 && count > 0) return count - 1;
    if (fault == 4 && ::ftruncate(mutation_fd.load(), 0) != 0) return -1;
    return count;
}
#endif

namespace {

using axiom::qwen4exp::AbstractGpuSlot;
using axiom::qwen4exp::ExpertAdmissionError;
using axiom::qwen4exp::ExpertByteView;
using axiom::qwen4exp::ExpertCapacityError;
using axiom::qwen4exp::ExpertIntegrityError;
using axiom::qwen4exp::ExpertKey;
using axiom::qwen4exp::ExpertPager;
using axiom::qwen4exp::ExpertPagerError;
using axiom::qwen4exp::ExpertPagerOptions;
using axiom::qwen4exp::ExpertPlane;
using axiom::qwen4exp::ExpertProjection;
using axiom::qwen4exp::ExpertRange;
using axiom::qwen4exp::ExpertRequestOptions;
using axiom::qwen4exp::Sha256;
using axiom::qwen4exp::sha256_bytes;

[[noreturn]] void fail(const char* expression, const char* file, int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": check failed: " + expression);
}

#define CHECK(expression)             \
    do {                              \
        if (!(expression)) {          \
            fail(#expression, __FILE__, __LINE__); \
        }                             \
    } while (false)

template <typename Exception, typename Callable>
void expect_throw(Callable&& callable) {
    static_assert(std::is_base_of<std::exception, Exception>::value,
                  "expected type must derive from std::exception");
    try {
        std::forward<Callable>(callable)();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

void write_all(int descriptor, const std::uint8_t* data, std::size_t size) {
    std::size_t completed = 0;
    while (completed < size) {
        const ssize_t count = ::write(descriptor, data + completed, size - completed);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
        }
        completed += static_cast<std::size_t>(count);
    }
}

class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& base = "/tmp") {
        std::string pattern = base + "/axiom-qwen4exp-pager-XXXXXX";
        const char* created = ::mkdtemp(pattern.data());
        if (created == nullptr) {
            throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
        }
        path_ = created;
    }

    ~TemporaryDirectory() {
        for (auto it = files_.rbegin(); it != files_.rend(); ++it) {
            ::unlink(it->c_str());
        }
        ::rmdir(path_.c_str());
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    std::string create_file(const std::string& name,
                            const std::vector<std::uint8_t>& bytes) {
        const std::string path = path_ + "/" + name;
        const int descriptor =
            ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (descriptor < 0) {
            throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
        }
        try {
            write_all(descriptor, bytes.data(), bytes.size());
        } catch (...) {
            ::close(descriptor);
            ::unlink(path.c_str());
            throw;
        }
        if (::close(descriptor) != 0) {
            ::unlink(path.c_str());
            throw std::runtime_error(std::string("close failed: ") + std::strerror(errno));
        }
        files_.push_back(path);
        return path;
    }

    std::string create_symlink(const std::string& name, const std::string& target) {
        const std::string path = path_ + "/" + name;
        if (::symlink(target.c_str(), path.c_str()) != 0) {
            throw std::runtime_error(std::string("symlink failed: ") +
                                     std::strerror(errno));
        }
        files_.push_back(path);
        return path;
    }

private:
    std::string path_;
    std::vector<std::string> files_;
};

ExpertKey key(std::uint32_t expert) {
    return ExpertKey{"radixark-7b719225242aacd3", 7U, expert,
                     ExpertProjection::gate, ExpertPlane::weight};
}

std::vector<ExpertRange> make_ranges(const std::string& file,
                                     const std::vector<std::uint8_t>& bytes,
                                     std::size_t range_size,
                                     std::size_t count) {
    std::vector<ExpertRange> ranges;
    ranges.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t offset = i * range_size;
        ranges.push_back(ExpertRange{
            key(static_cast<std::uint32_t>(i)), file, static_cast<std::uint64_t>(offset),
            static_cast<std::uint64_t>(range_size),
            sha256_bytes(bytes.data() + offset, range_size)});
    }
    return ranges;
}

void check_range(const ExpertByteView& view,
                 const std::vector<std::uint8_t>& expected,
                 std::size_t offset,
                 std::size_t size) {
    CHECK(view.valid());
    CHECK(view.size() == size);
    CHECK(std::memcmp(view.data(), expected.data() + offset, size) == 0);
}

void test_sha256() {
    const std::string abc = "abc";
    const Sha256 digest = sha256_bytes(
        reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size());
    CHECK(digest.hex() ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(Sha256::from_hex(digest.hex()) == digest);
    expect_throw<ExpertAdmissionError>([] { Sha256::from_hex("not-a-sha256"); });
}

void test_admission_fail_closed(TemporaryDirectory& temporary,
                                const std::string& source, bool direct = false) {
    ExpertPagerOptions options;
    options.ram_capacity_bytes = 1024;
    options.prefetch_workers = 1;
    options.require_checksums = true;
    options.owned_direct_pread = direct;

    ExpertRange missing_checksum{key(0), source, 0, 64, std::nullopt};
    expect_throw<ExpertAdmissionError>([&] {
        ExpertPager pager(options, {missing_checksum});
    });

    ExpertRange overflow{key(0), source,
                         std::numeric_limits<std::uint64_t>::max() - 3U, 8U,
                         Sha256{}};
    expect_throw<ExpertAdmissionError>([&] { ExpertPager pager(options, {overflow}); });

    const std::string link = temporary.create_symlink(
        direct ? "weights-link-direct.bin" : "weights-link.bin", source);
    ExpertRange symlink_range{key(0), link, 0, 64, Sha256{}};
    expect_throw<ExpertAdmissionError>([&] {
        ExpertPager pager(options, {symlink_range});
    });
}

void test_integrity_failure(TemporaryDirectory& temporary, bool direct = false) {
    std::vector<std::uint8_t> original(256);
    for (std::size_t i = 0; i < original.size(); ++i) {
        original[i] = static_cast<std::uint8_t>((i * 19U + 3U) & 0xffU);
    }
    const Sha256 expected = sha256_bytes(original.data(), original.size());
    std::vector<std::uint8_t> corrupted = original;
    corrupted[91] ^= 0x5aU;
    const std::string path = temporary.create_file(direct ? "corrupted-direct.bin" : "corrupted.bin", corrupted);

    ExpertPagerOptions options;
    options.ram_capacity_bytes = corrupted.size();
    options.prefetch_workers = 1;
    options.require_checksums = true;
    options.owned_direct_pread = direct;
    ExpertRange range{key(40), path, 0, static_cast<std::uint64_t>(corrupted.size()),
                      expected};
    ExpertPager pager(options, {range});
    const auto generation = pager.begin_generation();
    expect_throw<ExpertIntegrityError>([&] { pager.request(generation, range.key); });
    pager.rollback(generation);
    CHECK(pager.metrics().checksum_failures == 1);
    CHECK(pager.metrics().current_ram_bytes == 0);
    CHECK(pager.metrics().current_direct_staging_bytes == 0);
    CHECK(pager.metrics().direct_payload_bytes == 0);
    if (direct) CHECK(pager.metrics().direct_bytes_read == corrupted.size());
}

void test_failed_upload_revokes_callback_view(TemporaryDirectory& temporary, bool direct = false) {
    std::vector<std::uint8_t> bytes(128, 0x3cU);
    const std::string path = temporary.create_file(direct ? "upload-failure-direct.bin" : "upload-failure.bin", bytes);
    ExpertRange range{key(41), path, 0, static_cast<std::uint64_t>(bytes.size()),
                      sha256_bytes(bytes.data(), bytes.size())};

    ExpertByteView retained;
    ExpertPagerOptions options;
    options.ram_capacity_bytes = bytes.size();
    options.prefetch_workers = 1;
    options.require_checksums = true;
    options.owned_direct_pread = direct;
    options.gpu.upload = [&](const ExpertKey&, const ExpertByteView& view, std::uint64_t) {
        retained = view;
        throw std::runtime_error("synthetic upload failure");
        return AbstractGpuSlot{};
    };
    options.gpu.release = [](const ExpertKey&, const AbstractGpuSlot&) {};

    ExpertPager pager(options, {range});
    const auto generation = pager.begin_generation();
    expect_throw<std::runtime_error>([&] {
        pager.request(generation, range.key, {true, true});
    });
    CHECK(!retained.valid());
    pager.rollback(generation);
    retained = ExpertByteView();
    CHECK(pager.metrics().active_generations == 0);
}

void test_paging_transactions_and_concurrency(TemporaryDirectory& temporary, bool direct = false) {
    constexpr std::size_t range_size = 256U * 1024U;
    constexpr std::size_t range_count = 4;
    std::vector<std::uint8_t> bytes(range_size * range_count);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((i * 131U + i / range_size * 29U) & 0xffU);
    }
    const std::string path = temporary.create_file(direct ? "weights-direct.bin" : "weights.bin", bytes);
    auto ranges = make_ranges(path, bytes, range_size, range_count);

    std::atomic<std::uint64_t> next_slot{1};
    std::atomic<int> live_slots{0};
    std::atomic<std::uint64_t> uploads{0};
    std::atomic<std::uint64_t> releases{0};

    ExpertPagerOptions options;
    options.ram_capacity_bytes = range_size * 2U;
    options.prefetch_workers = 2;
    options.require_checksums = true;
    options.owned_direct_pread = direct;
    options.gpu.upload = [&](const ExpertKey&, const ExpertByteView& view,
                             std::uint64_t generation) {
        if (!view.valid() || view.size() != range_size || generation == 0) {
            throw std::runtime_error("invalid abstract GPU upload input");
        }
        ++uploads;
        ++live_slots;
        return AbstractGpuSlot{next_slot.fetch_add(1), view.size()};
    };
    options.gpu.release = [&](const ExpertKey&, const AbstractGpuSlot& slot) {
        if (slot.bytes == 0 || live_slots.fetch_sub(1) <= 0) {
            throw std::runtime_error("invalid abstract GPU release");
        }
        ++releases;
    };

    {
        ExpertPager pager(options, ranges);

        {
            const auto generation = pager.begin_generation();
            auto first = pager.request(generation, key(0), {true, true});
            check_range(first.view(), bytes, 0, range_size);
            CHECK(first.gpu_slot().has_value());
            CHECK(uploads.load() == 1);

            pager.pin(first);
            pager.unpin(first);
            auto duplicate = pager.request(generation, key(0), {true, true});
            CHECK(duplicate.gpu_slot() == first.gpu_slot());
            CHECK(duplicate.view().data() == first.view().data());
            CHECK(uploads.load() == 1);
            pager.commit(generation);
            CHECK(first.valid());
            CHECK(duplicate.valid());
            pager.unpin(first);
            pager.unpin(duplicate);
        }
        CHECK(pager.metrics().active_leases == 0);

        const std::uint64_t reads_before_prefetch = pager.metrics().nvme_reads;
        {
            const auto generation = pager.begin_generation();
            std::vector<std::future<ExpertPager::Lease>> futures;
            for (int i = 0; i < 8; ++i) {
                futures.push_back(pager.prefetch(generation, key(1), {false, false}));
            }
            std::vector<ExpertPager::Lease> leases;
            for (auto& future : futures) {
                leases.push_back(future.get());
                check_range(leases.back().view(), bytes, range_size, range_size);
            }
            pager.commit(generation);
        }
        CHECK(pager.metrics().nvme_reads == reads_before_prefetch + 1U);
        CHECK(pager.metrics().prefetches == 8U);

        {
            const auto generation = pager.begin_generation();
            std::mutex start_mutex;
            std::condition_variable start_condition;
            bool start = false;
            std::atomic<int> ready{0};
            std::atomic<int> failures{0};
            std::vector<std::thread> clients;
            for (int i = 0; i < 8; ++i) {
                clients.emplace_back([&] {
                    {
                        std::unique_lock<std::mutex> lock(start_mutex);
                        ++ready;
                        start_condition.wait(lock, [&] { return start; });
                    }
                    try {
                        auto lease = pager.request(generation, key(1), {false, true});
                        check_range(lease.view(), bytes, range_size, range_size);
                    } catch (...) {
                        ++failures;
                    }
                });
            }
            while (ready.load() != 8) {
                std::this_thread::yield();
            }
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                start = true;
            }
            start_condition.notify_all();
            for (auto& client : clients) {
                client.join();
            }
            CHECK(failures.load() == 0);
            pager.commit(generation);
        }

        // Raise key(0)'s frequency above key(1), then force a bounded-cache eviction.
        {
            const auto generation = pager.begin_generation();
            for (int i = 0; i < 24; ++i) {
                auto lease = pager.request(generation, key(0), {false, false});
                CHECK(lease.valid());
            }
            pager.commit(generation);
        }
        {
            const auto generation = pager.begin_generation();
            auto lease = pager.request(generation, key(2), {false, true});
            check_range(lease.view(), bytes, range_size * 2U, range_size);
            pager.commit(generation);
            pager.unpin(lease);
        }
        CHECK(pager.contains(key(0)));
        CHECK(!pager.contains(key(1)));
        CHECK(pager.contains(key(2)));
        CHECK(pager.metrics().evictions >= 1U);
        CHECK(pager.metrics().current_ram_bytes <= options.ram_capacity_bytes);

        // The tentative GPU slot and RAM range are both discarded by rollback.
        {
            const int slots_before = live_slots.load();
            const auto generation = pager.begin_generation();
            auto lease = pager.request(generation, key(3), {true, true});
            ExpertByteView borrowed = lease.view();
            CHECK(borrowed.valid());
            CHECK(live_slots.load() == slots_before + 1);
            pager.rollback(generation);
            CHECK(!lease.valid());
            CHECK(!borrowed.valid());
            CHECK(!lease.gpu_slot().has_value());
            CHECK(live_slots.load() == slots_before);
            CHECK(!pager.contains(key(3)));
            borrowed = ExpertByteView();
        }

        const auto final_metrics = pager.metrics();
        CHECK(final_metrics.active_generations == 0);
        CHECK(final_metrics.active_leases == 0);
        CHECK(final_metrics.current_ram_bytes <= options.ram_capacity_bytes);
        CHECK(final_metrics.nvme_bytes_read == final_metrics.nvme_reads * range_size);
        CHECK(final_metrics.gpu_uploads == uploads.load());
        CHECK(final_metrics.gpu_releases == releases.load());
        CHECK(final_metrics.current_direct_staging_bytes == 0);
        if (direct) {
            CHECK(final_metrics.direct_bytes_read == final_metrics.nvme_bytes_read);
            CHECK(final_metrics.direct_payload_bytes == final_metrics.nvme_bytes_read);
            CHECK(final_metrics.peak_direct_staging_bytes <= final_metrics.direct_staging_limit_bytes);
        } else {
            CHECK(final_metrics.direct_read_calls == 0);
            CHECK(final_metrics.direct_staging_limit_bytes == 0);
        }
    }
    CHECK(live_slots.load() == 0);
    CHECK(uploads.load() == releases.load());
}

// Independent capability probe. Unsupported hosts explicitly skip real direct
// cases unless --require-direct is used; pager admission must still fail closed.
bool probe_direct_support(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd < 0) return false;
    bool supported = false;
#if defined(SYS_statx) && defined(STATX_DIOALIGN) && defined(STATX_ATTR_VERITY)
    struct statx info {};
    supported = ::syscall(SYS_statx, fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &info) == 0 &&
        (info.stx_mask & STATX_DIOALIGN) && !(info.stx_attributes & STATX_ATTR_VERITY) &&
        info.stx_dio_mem_align != 0 && info.stx_dio_offset_align != 0 &&
        4096 % info.stx_dio_mem_align == 0 && 4096 % info.stx_dio_offset_align == 0;
#endif
    ::close(fd);
    return supported;
}

void test_direct_ranges(TemporaryDirectory& temporary) {
    constexpr std::size_t chunk = 4U * 1024U * 1024U;
    std::vector<std::uint8_t> bytes(2U * chunk + 113U);
    for (std::size_t i = 0; i < bytes.size(); ++i)
        bytes[i] = static_cast<std::uint8_t>((i * 71U) ^ (i >> 12U) ^ (i >> 20U));
    const auto path = temporary.create_file("direct-ranges.bin", bytes);
    const std::vector<std::pair<std::size_t, std::size_t>> spans{
        {0, 4096}, {17, 45}, {4090, 23}, {123, chunk + 37}, {bytes.size() - 71, 71},
    };
    std::vector<ExpertRange> ranges;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        ranges.push_back({key(static_cast<std::uint32_t>(100 + i)), path,
                          spans[i].first, spans[i].second,
                          sha256_bytes(bytes.data() + spans[i].first, spans[i].second)});
    }
    ExpertPagerOptions options;
    options.ram_capacity_bytes = bytes.size();
    options.prefetch_workers = 1;  // Concurrent callers must share this one slot.
    options.require_checksums = true;
    options.owned_direct_pread = true;
    ExpertPager pager(options, ranges);
    auto generation = pager.begin_generation();
    std::vector<std::future<ExpertPager::Lease>> futures;
    for (const auto& range : ranges) {
        futures.push_back(std::async(std::launch::async, [&pager, generation, range] {
            return pager.request(generation, range.key);
        }));
    }
    std::vector<ExpertPager::Lease> leases;
    std::uint64_t expected_requested = 0, expected_returned = 0, expected_payload = 0;
    for (std::size_t i = 0; i < spans.size(); ++i) {
        leases.push_back(futures[i].get());
        check_range(leases.back().view(), bytes, spans[i].first, spans[i].second);
        const auto first = spans[i].first / 4096U * 4096U;
        const auto end = (spans[i].first + spans[i].second + 4095U) / 4096U * 4096U;
        expected_requested += end - first;
        expected_returned += std::min(end, bytes.size()) - first;
        expected_payload += spans[i].second;
    }
    const auto metrics = pager.metrics();
    CHECK(metrics.direct_read_calls == 6);  // Four single chunks plus a two-chunk range.
    CHECK(metrics.direct_requested_bytes == expected_requested);
    CHECK(metrics.direct_bytes_read == expected_returned);
    CHECK(metrics.direct_payload_bytes == expected_payload);
    CHECK(metrics.nvme_bytes_read == expected_payload);
    CHECK(metrics.current_ram_bytes == expected_payload);
    CHECK(metrics.current_direct_staging_bytes == 0);
    CHECK(metrics.peak_direct_staging_bytes == chunk);
    CHECK(metrics.direct_staging_limit_bytes == chunk);
    ExpertByteView retained = leases.front().view();
    pager.rollback(generation);
    CHECK(!retained.valid());
    CHECK(retained.data() == nullptr && retained.size() == 0);
    CHECK(!pager.contains(ranges.front().key));
    CHECK(pager.metrics().current_ram_bytes == spans.front().second);
    retained = ExpertByteView();
    leases.clear();
    // A subsequent reservation reaps the last revoked but retained payload.
    generation = pager.begin_generation();
    auto lease = pager.request(generation, ranges.front().key);
    CHECK(pager.metrics().current_ram_bytes == spans.front().second);
    pager.rollback(generation);
    CHECK(pager.metrics().current_ram_bytes == 0);

    // Smaller-than-page file, optional checksum absent, last aligned block
    // returns only actual EOF bytes and still covers the complete logical range.
    std::vector<std::uint8_t> tiny(31, 0xa7);
    const auto tiny_path = temporary.create_file("direct-tiny.bin", tiny);
    options.require_checksums = false;
    options.ram_capacity_bytes = 7;
    ExpertRange tiny_range{key(120), tiny_path, 24, 7, std::nullopt};
    ExpertPager tiny_pager(options, {tiny_range});
    const auto tiny_generation = tiny_pager.begin_generation();
    auto tiny_lease = tiny_pager.request(tiny_generation, tiny_range.key);
    check_range(tiny_lease.view(), tiny, 24, 7);
    CHECK(tiny_pager.metrics().direct_requested_bytes == 4096);
    CHECK(tiny_pager.metrics().direct_bytes_read == 31);
    CHECK(tiny_pager.metrics().direct_payload_bytes == 7);
    CHECK(tiny_pager.metrics().peak_direct_staging_bytes == 4096);
    tiny_pager.rollback(tiny_generation);
}

void test_direct_mutation_and_capacity(TemporaryDirectory& temporary) {
    std::vector<std::uint8_t> bytes(8192, 0xb3);
    const auto path = temporary.create_file("direct-mutation.bin", bytes);
    ExpertPagerOptions options;
    options.ram_capacity_bytes = bytes.size();
    options.prefetch_workers = 1;
    options.owned_direct_pread = true;
    const auto ranges = make_ranges(path, bytes, 4096, 2);
    ExpertPager pager(options, ranges);
    auto generation = pager.begin_generation();
    auto first = pager.request(generation, ranges[0].key);
    pager.commit(generation);
    // Mutate only a synthetic fixture. The already-owned snapshot stays stable;
    // a new miss must fail identity checks before any additional direct read.
    const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
    CHECK(fd >= 0);
    const int truncate_rc = ::ftruncate(fd, 100);
    ::close(fd);
    CHECK(truncate_rc == 0);
    check_range(first.view(), bytes, 0, 4096);
    generation = pager.begin_generation();
    expect_throw<ExpertIntegrityError>([&] { pager.request(generation, ranges[1].key); });
    CHECK(pager.metrics().direct_read_calls == 1);
    CHECK(pager.metrics().io_failures == 1);
    CHECK(pager.metrics().current_direct_staging_bytes == 0);
    pager.rollback(generation);

    const auto capacity_path = temporary.create_file("direct-capacity.bin", bytes);
    options.ram_capacity_bytes = 4096;
    const auto capacity_ranges = make_ranges(capacity_path, bytes, 4096, 2);
    ExpertPager bounded(options, capacity_ranges);
    const auto bounded_generation = bounded.begin_generation();
    auto held = bounded.request(bounded_generation, capacity_ranges[0].key);
    expect_throw<ExpertCapacityError>([&] {
        bounded.request(bounded_generation, capacity_ranges[1].key);
    });
    CHECK(bounded.metrics().direct_read_calls == 1);
    bounded.rollback(bounded_generation);
    CHECK(bounded.metrics().current_ram_bytes == 0);

    ExpertRange outside{key(121), capacity_path, bytes.size() - 2, 3, std::nullopt};
    expect_throw<ExpertAdmissionError>([&] { ExpertPager invalid(options, {outside}); });
    ExpertRange padded_overflow{key(122), capacity_path,
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) - 1U, 1, std::nullopt};
    expect_throw<ExpertAdmissionError>([&] { ExpertPager invalid(options, {padded_overflow}); });
}

void test_unsupported_direct_source() {
    if (::access("/dev/shm", W_OK | X_OK) != 0) {
        std::cout << "unsupported-direct fixture: SKIP (/dev/shm unavailable)\n";
        return;
    }
    TemporaryDirectory temporary("/dev/shm");
    std::vector<std::uint8_t> bytes(4096, 0x79);
    const auto path = temporary.create_file("unsupported.bin", bytes);
    if (probe_direct_support(path)) {
        std::cout << "unsupported-direct fixture: SKIP (/dev/shm supports direct I/O)\n";
        return;
    }
    ExpertPagerOptions options;
    options.ram_capacity_bytes = bytes.size();
    options.owned_direct_pread = true;
    const auto ranges = make_ranges(path, bytes, bytes.size(), 1);
    expect_throw<ExpertAdmissionError>([&] { ExpertPager invalid(options, ranges); });
    options.owned_direct_pread = false;
    ExpertPager buffered(options, ranges);
    const auto generation = buffered.begin_generation();
    auto lease = buffered.request(generation, ranges[0].key);
    check_range(lease.view(), bytes, 0, bytes.size());
    buffered.rollback(generation);
}

void test_discard_cached(TemporaryDirectory& temporary, bool direct = false) {
    const std::vector<std::uint8_t> bytes(8192, 0x93);
    const auto path = temporary.create_file(direct ? "discard-direct.bin" : "discard.bin", bytes);
    const auto ranges = make_ranges(path, bytes, 4096, 2);
    ExpertPagerOptions options;
    options.ram_capacity_bytes = bytes.size();
    options.prefetch_workers = 1;
    options.owned_direct_pread = direct;
    int uploads = 0, releases = 0;
    ExpertPager* live_pager = nullptr;
    options.gpu.upload = [&](const ExpertKey& item, const ExpertByteView& view, std::uint64_t) {
        CHECK(!live_pager->discard_cached(item));  // Upload in progress.
        return AbstractGpuSlot{static_cast<std::uint64_t>(++uploads), view.size()};
    };
    options.gpu.release = [&](const ExpertKey& item, const AbstractGpuSlot&) {
        // Callback runs outside the pager mutex; reentrant inspection is safe.
        CHECK(!live_pager->contains(item));
        ++releases;
    };
    ExpertPager pager(options, ranges);
    live_pager = &pager;
    CHECK(pager.discard_cached(key(999)));
    CHECK(pager.metrics().evictions == 0);
    auto generation = pager.begin_generation();
    auto lease = pager.request(generation, ranges[0].key, {true, true});
    ExpertByteView retained = lease.view();
    CHECK(!pager.discard_cached(ranges[0].key));
    pager.commit(generation);
    CHECK(!pager.discard_cached(ranges[0].key));  // Still pinned and leased.
    pager.unpin(lease);
    CHECK(!pager.discard_cached(ranges[0].key));  // Unpinned is not unleased.
    CHECK(retained.valid());
    lease = ExpertPager::Lease{};
    CHECK(!retained.valid());
    CHECK(!pager.discard_cached(ranges[0].key));  // Revoked but owning view.
    CHECK(pager.metrics().current_ram_bytes == 4096);
    CHECK(pager.metrics().evictions == 0);
    CHECK(releases == 0);
    retained = ExpertByteView{};
    CHECK(pager.discard_cached(ranges[0].key));
    CHECK(!pager.contains(ranges[0].key));
    CHECK(pager.metrics().current_ram_bytes == 0);
    CHECK(pager.metrics().cache_entries == 0);
    CHECK(pager.metrics().evictions == 1);
    CHECK(pager.metrics().gpu_releases == 1);
    CHECK(uploads == 1 && releases == 1);
    CHECK(pager.discard_cached(ranges[0].key));
    CHECK(pager.metrics().evictions == 1);

    generation = pager.begin_generation();
    {
        auto tentative = pager.request(generation, ranges[1].key, {false, false});
    }
    CHECK(!pager.discard_cached(ranges[1].key));  // Unleased but not committed.
    pager.commit(generation);
    CHECK(pager.discard_cached(ranges[1].key));
    CHECK(pager.metrics().evictions == 2);
    CHECK(pager.metrics().current_ram_bytes == 0);
    CHECK(pager.metrics().gpu_releases == 1);
}

#ifdef AXIOM_EXPERT_PAGER_TEST_WRAP_PREAD
void test_direct_read_faults(TemporaryDirectory& temporary) {
    for (int fault = 1; fault <= 5; ++fault) {
        const std::vector<std::uint8_t> bytes(8192, 0x62);
        const auto path = temporary.create_file("direct-fault-" + std::to_string(fault), bytes);
        ExpertPagerOptions options;
        options.owned_direct_pread = true;
        options.prefetch_workers = 1;
        options.ram_capacity_bytes = 100;
        ExpertRange range{key(150), path, 17, 100, sha256_bytes(bytes.data() + 17, 100)};
        ExpertPager pager(options, {range});
        const auto generation = pager.begin_generation();
        const int writer = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
        CHECK(writer >= 0);
        mutation_fd.store(writer);
        direct_fault.store(fault);
        try {
            if (fault == 1) {
                auto lease = pager.request(generation, range.key);
                check_range(lease.view(), bytes, 17, 100);
                CHECK(pager.metrics().direct_read_calls == 2);
                CHECK(pager.metrics().direct_requested_bytes == 8192);
                CHECK(pager.metrics().direct_bytes_read == 4096);
                CHECK(pager.metrics().direct_payload_bytes == 100);
            } else {
                if (fault == 2) {
                    expect_throw<ExpertPagerError>([&] { pager.request(generation, range.key); });
                } else {
                    expect_throw<ExpertIntegrityError>([&] { pager.request(generation, range.key); });
                }
                CHECK(pager.metrics().direct_read_calls == 1);
                CHECK(pager.metrics().direct_payload_bytes == 0);
                CHECK(pager.metrics().nvme_reads == 0);
                CHECK(pager.metrics().io_failures == 1);
                CHECK(pager.metrics().current_ram_bytes == 0);
            }
            CHECK(direct_fault.load() == 0);
            CHECK(pager.metrics().current_direct_staging_bytes == 0);
            pager.rollback(generation);
        } catch (...) {
            direct_fault.store(0);
            mutation_fd.store(-1);
            ::close(writer);
            throw;
        }
        mutation_fd.store(-1);
        ::close(writer);
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--require-direct"))
            throw std::runtime_error("usage: expert-pager-test [--require-direct]");
        CHECK(!ExpertPagerOptions{}.owned_direct_pread);
        TemporaryDirectory temporary;
        test_sha256();

        std::vector<std::uint8_t> admission_bytes(1024, 0xa5U);
        const std::string admission_source =
            temporary.create_file("admission.bin", admission_bytes);
        test_admission_fail_closed(temporary, admission_source);
        test_integrity_failure(temporary);
        test_failed_upload_revokes_callback_view(temporary);
        test_paging_transactions_and_concurrency(temporary);
        test_discard_cached(temporary);
        test_unsupported_direct_source();
        if (probe_direct_support(admission_source)) {
            test_admission_fail_closed(temporary, admission_source, true);
            test_integrity_failure(temporary, true);
            test_failed_upload_revokes_callback_view(temporary, true);
            test_paging_transactions_and_concurrency(temporary, true);
            test_discard_cached(temporary, true);
            test_direct_ranges(temporary);
            test_direct_mutation_and_capacity(temporary);
#ifdef AXIOM_EXPERT_PAGER_TEST_WRAP_PREAD
            test_direct_read_faults(temporary);
#endif
            std::cout << "owned_direct_pread cases: PASS\n";
        } else {
            ExpertPagerOptions options;
            options.ram_capacity_bytes = admission_bytes.size();
            options.owned_direct_pread = true;
            const auto ranges = make_ranges(admission_source, admission_bytes, admission_bytes.size(), 1);
            expect_throw<ExpertAdmissionError>([&] { ExpertPager invalid(options, ranges); });
            if (argc == 2) throw std::runtime_error("required direct-I/O support unavailable on /tmp");
            std::cout << "owned_direct_pread cases: SKIP (no verified direct I/O on /tmp)\n";
        }

        std::cout << "qwen4_exp expert pager tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen4_exp expert pager tests: FAIL: " << error.what() << '\n';
        return 1;
    }
}
