#include "axiom/qwen38_session_store.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <new>
#include <sstream>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unordered_set>
#include <unistd.h>
#include <utility>

namespace axiom {
namespace qwen38 {
namespace {

constexpr char kManifestMagic[] = "AXQ38SM1";
constexpr uint32_t kManifestAbi = 1u;
constexpr uint32_t kManifestHeaderBytes = 4096u;
constexpr size_t kSessionIdBytes = 128u;
constexpr size_t kNamespaceIdBytes = 33u;
constexpr size_t kNamespaceTextBytes = 1536u;
constexpr size_t kProfileBytes = 32u;
constexpr size_t kMaxRecurrentBytes = 1024u * 1024u * 1024u;
constexpr size_t kMaxGcArtifactsPerNamespace = 16384u;

#pragma pack(push, 1)
struct manifest_disk_header {
    char magic[8];
    uint32_t abi;
    uint32_t header_bytes;
    uint64_t generation;
    uint32_t committed_tokens;
    uint32_t next_token;
    uint32_t token_count;
    uint32_t recurrent_bytes;
    uint64_t token_hash;
    uint64_t payload_bytes;
    uint64_t checksum;
    char namespace_id[kNamespaceIdBytes];
    char session_id[kSessionIdBytes];
    char namespace_text[kNamespaceTextBytes];
    char profile[kProfileBytes];
    uint8_t reserved[kManifestHeaderBytes -
            (8u + 4u + 4u + 8u + 4u + 4u + 4u + 4u +
             8u + 8u + 8u + kNamespaceIdBytes + kSessionIdBytes +
             kNamespaceTextBytes + kProfileBytes)];
};
#pragma pack(pop)
static_assert(sizeof(manifest_disk_header) == kManifestHeaderBytes,
              "Qwen3.8 session manifest header must remain 4096 bytes");

std::atomic<uint64_t> g_temp_sequence{0u};

class scoped_fd {
public:
    explicit scoped_fd(const int value = -1) : value_(value) {}
    scoped_fd(const scoped_fd &) = delete;
    scoped_fd &operator=(const scoped_fd &) = delete;
    ~scoped_fd() { reset(); }

    int get() const { return value_; }
    int release() {
        const int value = value_;
        value_ = -1;
        return value;
    }
    void reset(const int value = -1) {
        if (value_ >= 0) ::close(value_);
        value_ = value;
    }

private:
    int value_ = -1;
};

class scoped_directory {
public:
    explicit scoped_directory(DIR *value = nullptr) : value_(value) {}
    scoped_directory(const scoped_directory &) = delete;
    scoped_directory &operator=(const scoped_directory &) = delete;
    ~scoped_directory() { reset(); }

    DIR *get() const { return value_; }
    void reset(DIR *value = nullptr) {
        if (value_) ::closedir(value_);
        value_ = value;
    }

private:
    DIR *value_ = nullptr;
};

uint64_t fnv_update(uint64_t hash, const void *data, size_t bytes) {
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

std::string hex64(const uint64_t value) {
    char buffer[17]{};
    std::snprintf(buffer, sizeof(buffer), "%016llx",
                  static_cast<unsigned long long>(value));
    return std::string(buffer);
}

std::string hash128(const std::string &text) {
    uint64_t first = 1469598103934665603ull;
    uint64_t second = 1099511628211ull ^ 0x9e3779b97f4a7c15ull;
    first = fnv_update(first, text.data(), text.size());
    second = fnv_update(second, text.data(), text.size());
    second = fnv_update(second, &first, sizeof(first));
    return hex64(first) + hex64(second);
}

bool write_full(const int fd, const void *data, const size_t bytes) {
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    size_t offset = 0u;
    while (offset < bytes) {
        const ssize_t written = ::write(fd, cursor + offset, bytes - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

bool read_full(const int fd, void *data, const size_t bytes) {
    uint8_t *cursor = static_cast<uint8_t *>(data);
    size_t offset = 0u;
    while (offset < bytes) {
        const ssize_t read_bytes = ::read(fd, cursor + offset, bytes - offset);
        if (read_bytes > 0) {
            offset += static_cast<size_t>(read_bytes);
            continue;
        }
        if (read_bytes < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
}

bool make_directory_recursive(const std::string &path) {
    if (path.empty()) return false;
    std::string current;
    size_t start = 0u;
    if (path[0] == '/') {
        current = "/";
        start = 1u;
    }
    while (start <= path.size()) {
        const size_t slash = path.find('/', start);
        const size_t end = slash == std::string::npos ? path.size() : slash;
        if (end > start) {
            if (current.empty() || current == "/") {
                if (current != "/") current.clear();
                current += path.substr(start, end - start);
            } else {
                current += "/" + path.substr(start, end - start);
            }
            if (::mkdir(current.c_str(), 0700) != 0 && errno != EEXIST) return false;
            struct stat status{};
            if (::stat(current.c_str(), &status) != 0 || !S_ISDIR(status.st_mode)) return false;
        }
        if (slash == std::string::npos) break;
        start = slash + 1u;
    }
    return true;
}

bool copy_text(char *destination, const size_t capacity, const std::string &text) {
    if (!destination || text.size() >= capacity) return false;
    std::memset(destination, 0, capacity);
    std::memcpy(destination, text.data(), text.size());
    return true;
}

bool read_text(const char *source, const size_t capacity, std::string *out) {
    if (!source || !out) return false;
    const void *terminator = std::memchr(source, '\0', capacity);
    if (!terminator) return false;
    const char *end = static_cast<const char *>(terminator);
    out->assign(source, static_cast<size_t>(end - source));
    return true;
}

uint64_t header_payload_checksum(
        const manifest_disk_header &source,
        const std::vector<uint8_t> &payload) {
    manifest_disk_header header = source;
    header.checksum = 0u;
    uint64_t hash = 1469598103934665603ull;
    hash = fnv_update(hash, &header, sizeof(header));
    if (!payload.empty()) hash = fnv_update(hash, payload.data(), payload.size());
    return hash;
}

bool parent_directory_sync(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    const std::string directory = slash == std::string::npos
            ? "." : (slash == 0u ? "/" : path.substr(0u, slash));
    const int fd = ::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (fd < 0) return false;
    const int rc = ::fsync(fd);
    ::close(fd);
    return rc == 0;
}

bool read_random_bytes(void *data, const size_t bytes) {
    const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = read_full(fd, data, bytes);
    ::close(fd);
    return ok;
}

bool parse_generation_filename(
        const std::string &name,
        const std::string &session_stem,
        uint64_t *generation) {
    if (!generation) return false;
    const std::string prefix = session_stem + ".g";
    constexpr const char *suffix = ".kv";
    constexpr size_t suffix_size = 3u;
    if (name.size() <= prefix.size() + suffix_size ||
        name.compare(0u, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix_size, suffix_size, suffix) != 0) {
        return false;
    }
    const size_t digits_end = name.size() - suffix_size;
    if (name[prefix.size()] == '0') return false;
    uint64_t value = 0u;
    for (size_t index = prefix.size(); index < digits_end; ++index) {
        const unsigned char character = static_cast<unsigned char>(name[index]);
        if (character < '0' || character > '9') return false;
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) {
            return false;
        }
        value = value * 10u + digit;
    }
    if (value == 0u) return false;
    *generation = value;
    return true;
}

bool is_lower_hex_32(const std::string &text) {
    if (text.size() != 32u) return false;
    for (const unsigned char character : text) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool exact_positive_decimal_components(
        const std::string &text,
        const size_t expected_components) {
    if (text.empty() || expected_components == 0u ||
        text.front() == '.' || text.back() == '.') {
        return false;
    }
    size_t components = 1u;
    uint64_t value = 0u;
    bool have_digit = false;
    for (const unsigned char character : text) {
        if (character >= '0' && character <= '9') {
            if (!have_digit && character == '0') return false;
            const uint64_t digit = static_cast<uint64_t>(character - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10u) {
                return false;
            }
            value = value * 10u + digit;
            have_digit = true;
            continue;
        }
        if (character != '.' || !have_digit || value == 0u ||
            components >= expected_components) {
            return false;
        }
        ++components;
        value = 0u;
        have_digit = false;
    }
    return components == expected_components && have_digit && value != 0u;
}

bool parse_session_artifact(
        const std::string &name,
        std::string *session_stem) {
    if (!session_stem || name.size() <= 32u) return false;
    const std::string stem = name.substr(0u, 32u);
    if (!is_lower_hex_32(stem)) return false;
    const std::string suffix = name.substr(32u);
    bool valid = suffix == ".manifest";
    uint64_t generation = 0u;
    if (!valid) valid = parse_generation_filename(name, stem, &generation);
    constexpr const char *temporary_prefix = ".manifest.tmp.";
    if (!valid && suffix.compare(0u, std::strlen(temporary_prefix),
                                 temporary_prefix) == 0) {
        valid = exact_positive_decimal_components(
                suffix.substr(std::strlen(temporary_prefix)), 2u);
    }
    if (!valid) return false;
    *session_stem = stem;
    return true;
}

bool parse_gc_tombstone(
        const std::string &name,
        std::string *namespace_id) {
    constexpr const char *prefix = ".gc.";
    if (!namespace_id || name.size() <= 4u + 32u + 1u ||
        name.compare(0u, 4u, prefix) != 0) {
        return false;
    }
    const std::string candidate = name.substr(4u, 32u);
    if (!is_lower_hex_32(candidate) || name[36u] != '.' ||
        !exact_positive_decimal_components(name.substr(37u), 3u)) {
        return false;
    }
    *namespace_id = candidate;
    return true;
}

bool rename_directory_noreplace(
        const int root_fd,
        const std::string &source,
        const std::string &destination) {
#ifdef SYS_renameat2
    constexpr unsigned int kRenameNoReplace = 1u;
    if (::syscall(SYS_renameat2, root_fd, source.c_str(), root_fd,
                  destination.c_str(), kRenameNoReplace) == 0) {
        return true;
    }
    if (errno != ENOSYS && errno != EINVAL) return false;
#endif
    /* Old kernels lack renameat2. The daemon is a singleton and lifecycle GC
     * is serialized, so an explicit no-target check preserves the same local
     * invariant on that compatibility path. */
    struct stat status{};
    if (::fstatat(root_fd, destination.c_str(), &status, AT_SYMLINK_NOFOLLOW) == 0) {
        errno = EEXIST;
        return false;
    }
    if (errno != ENOENT) return false;
    return ::renameat(root_fd, source.c_str(), root_fd, destination.c_str()) == 0;
}

uint64_t allocated_bytes(const struct stat &status) {
    if (status.st_blocks <= 0) return 0u;
    const uint64_t blocks = static_cast<uint64_t>(status.st_blocks);
    if (blocks > std::numeric_limits<uint64_t>::max() / 512u) {
        return std::numeric_limits<uint64_t>::max();
    }
    return blocks * 512u;
}

uint64_t add_saturating(const uint64_t left, const uint64_t right) {
    return right > std::numeric_limits<uint64_t>::max() - left
            ? std::numeric_limits<uint64_t>::max() : left + right;
}

uint64_t mtime_seconds(const struct stat &status) {
    return status.st_mtim.tv_sec > 0
            ? static_cast<uint64_t>(status.st_mtim.tv_sec) : 0u;
}

bool private_owned_directory(const struct stat &status) {
    return S_ISDIR(status.st_mode) && status.st_uid == ::geteuid() &&
            (status.st_mode & (S_IRWXG | S_IRWXO)) == 0u;
}

bool validate_private_directory(
        const std::string &path,
        std::string *error) {
    if (!error) return false;
    const int fd = ::open(
            path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    if (fd < 0) {
        *error = "cannot open private persistent KV directory " + path + ": " +
                std::string(std::strerror(errno));
        return false;
    }
    struct stat status{};
    const int stat_rc = ::fstat(fd, &status);
    const int saved_errno = stat_rc == 0 ? 0 : errno;
    const bool valid = stat_rc == 0 && private_owned_directory(status);
    ::close(fd);
    if (!valid) {
        *error = "persistent KV directory is not private and owned by the daemon: " +
                path + (saved_errno != 0
                        ? ": " + std::string(std::strerror(saved_errno)) : "");
    }
    return valid;
}

struct gc_namespace_entry {
    std::string namespace_id;
    std::string directory_name;
    std::string session_stem;
    std::vector<std::string> files;
    uint64_t last_used_unix_seconds = 0u;
    uint64_t allocated_bytes = 0u;
    bool protected_namespace = false;
    bool remove = false;
};

bool scan_gc_namespace(
        const int root_fd,
        const std::string &directory_name,
        const std::string &namespace_id,
        const size_t artifact_budget,
        size_t *total_artifacts,
        int *failure_status,
        gc_namespace_entry *out,
        std::string *error) {
    if (root_fd < 0 || !total_artifacts || !failure_status || !out || !error ||
        !is_lower_hex_32(namespace_id)) return false;
    *failure_status = AXIOM_OK;
    struct stat directory_status{};
    if (::fstatat(root_fd, directory_name.c_str(), &directory_status,
                  AT_SYMLINK_NOFOLLOW) != 0 ||
        !private_owned_directory(directory_status)) {
        *error = "persistent KV session namespace is not a safe directory: " +
                directory_name;
        return false;
    }
    scoped_fd namespace_fd(::openat(
            root_fd, directory_name.c_str(),
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    if (namespace_fd.get() < 0) {
        *error = "cannot open persistent KV session namespace: " +
                std::string(std::strerror(errno));
        return false;
    }
    DIR *raw_directory = ::fdopendir(namespace_fd.get());
    if (!raw_directory) {
        const int saved_errno = errno;
        *error = "cannot scan persistent KV session namespace: " +
                std::string(std::strerror(saved_errno));
        return false;
    }
    (void)namespace_fd.release();
    scoped_directory directory(raw_directory);
    const int fd = ::dirfd(directory.get());

    gc_namespace_entry entry;
    entry.namespace_id = namespace_id;
    entry.directory_name = directory_name;
    entry.last_used_unix_seconds = mtime_seconds(directory_status);
    entry.allocated_bytes = allocated_bytes(directory_status);
    bool ok = true;
    int saved_errno = 0;
    while (ok) {
        errno = 0;
        dirent *item = ::readdir(directory.get());
        if (!item) {
            if (errno != 0) {
                ok = false;
                saved_errno = errno;
            }
            break;
        }
        const std::string name(item->d_name);
        if (name == "." || name == "..") continue;
        struct stat status{};
        if (::fstatat(fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            ok = false;
            saved_errno = errno;
            break;
        }
        std::string stem;
        if (!S_ISREG(status.st_mode) || status.st_nlink != 1 ||
            !parse_session_artifact(name, &stem) ||
            (!entry.session_stem.empty() && entry.session_stem != stem)) {
            ok = false;
            saved_errno = EINVAL;
            break;
        }
        entry.session_stem = std::move(stem);
        if (entry.files.size() >= kMaxGcArtifactsPerNamespace ||
            *total_artifacts >= artifact_budget) {
            ok = false;
            saved_errno = EOVERFLOW;
            *failure_status = AXIOM_ERR_BUDGET;
            break;
        }
        try {
            entry.files.push_back(name);
            ++*total_artifacts;
        } catch (...) {
            ok = false;
            saved_errno = ENOMEM;
            *failure_status = AXIOM_ERR_BUDGET;
            break;
        }
        entry.last_used_unix_seconds = std::max(
                entry.last_used_unix_seconds, mtime_seconds(status));
        entry.allocated_bytes = add_saturating(
                entry.allocated_bytes, allocated_bytes(status));
    }
    if (!ok) {
        *error = "unsafe or unreadable persistent KV session namespace " +
                directory_name + ": " + std::string(std::strerror(saved_errno));
        return false;
    }
    *out = std::move(entry);
    return true;
}

bool remove_gc_namespace(
        const int root_fd,
        const gc_namespace_entry &entry,
        uint64_t *removed_files,
        uint64_t *reclaimed_bytes,
        std::string *error) {
    if (root_fd < 0 || !removed_files || !reclaimed_bytes || !error) return false;
    scoped_fd tombstone_fd(::openat(
            root_fd, entry.directory_name.c_str(),
            O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    if (tombstone_fd.get() < 0) {
        *error = "cannot open persistent KV GC tombstone: " +
                std::string(std::strerror(errno));
        return false;
    }
    const int fd = tombstone_fd.get();
    bool ok = true;
    int saved_errno = 0;
    uint64_t files = 0u;
    struct stat directory_status{};
    const int directory_stat_rc = ::fstat(fd, &directory_status);
    if (directory_stat_rc != 0 || !private_owned_directory(directory_status)) {
        const int status_errno = directory_stat_rc == 0 ? EACCES : errno;
        *error = "cannot stat persistent KV GC tombstone: " +
                std::string(std::strerror(status_errno));
        return false;
    }
    uint64_t bytes = allocated_bytes(directory_status);
    for (const std::string &name : entry.files) {
        struct stat status{};
        if (::fstatat(fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno == ENOENT) continue;
            ok = false;
            saved_errno = errno;
            break;
        }
        if (!S_ISREG(status.st_mode) || status.st_nlink != 1) {
            ok = false;
            saved_errno = EINVAL;
            break;
        }
        if (::unlinkat(fd, name.c_str(), 0) != 0) {
            if (errno == ENOENT) continue;
            ok = false;
            saved_errno = errno;
            break;
        }
        ++files;
        bytes = add_saturating(bytes, allocated_bytes(status));
    }
    if (ok && ::fsync(fd) != 0) {
        ok = false;
        saved_errno = errno;
    }
    tombstone_fd.reset();
    if (ok && ::unlinkat(root_fd, entry.directory_name.c_str(), AT_REMOVEDIR) != 0) {
        ok = false;
        saved_errno = errno;
    }
    if (ok && ::fsync(root_fd) != 0) {
        ok = false;
        saved_errno = errno;
    }
    *removed_files = add_saturating(*removed_files, files);
    *reclaimed_bytes = add_saturating(*reclaimed_bytes, bytes);
    if (!ok) {
        *error = "persistent KV session tombstone cleanup failed: " +
                std::string(std::strerror(saved_errno));
        return false;
    }
    return true;
}

}  // namespace

int qwen38_build_stateful_resume_prompt(
        const qwen38_session_manifest &manifest,
        const std::vector<uint32_t> &session_suffix_ids,
        const uint32_t im_end_token_id,
        const uint32_t endoftext_token_id,
        const uint32_t vocab_size,
        const uint32_t prompt_limit,
        std::vector<uint32_t> *out,
        bool *usable) {
    if (!out || !usable || vocab_size == 0u ||
        manifest.committed_tokens > manifest.token_ids.size()) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    out->clear();
    *usable = false;
    if (manifest.committed_tokens == 0u || session_suffix_ids.empty() ||
        im_end_token_id == AXIOM_TOKEN_ID_INVALID ||
        manifest.next_token >= vocab_size) {
        return AXIOM_OK;
    }
    /* An end-of-text cursor is not a ChatML assistant boundary.  Resetting and
     * replaying the wire prompt is safer than inventing a transition. */
    if (manifest.next_token == endoftext_token_id) return AXIOM_OK;
    const bool needs_chatml_close = manifest.next_token != im_end_token_id;
    const uint64_t candidate_tokens =
            static_cast<uint64_t>(manifest.committed_tokens) + 1u +
            (needs_chatml_close ? 1u : 0u) + session_suffix_ids.size();
    if (candidate_tokens > prompt_limit ||
        candidate_tokens > std::numeric_limits<size_t>::max()) {
        return AXIOM_OK;
    }
    try {
        out->reserve(static_cast<size_t>(candidate_tokens));
        out->insert(
                out->end(), manifest.token_ids.begin(),
                manifest.token_ids.begin() + manifest.committed_tokens);
        out->push_back(manifest.next_token);
        if (needs_chatml_close) out->push_back(im_end_token_id);
        out->insert(out->end(), session_suffix_ids.begin(), session_suffix_ids.end());
    } catch (...) {
        out->clear();
        return AXIOM_ERR_BUDGET;
    }
    *usable = true;
    return AXIOM_OK;
}

qwen38_persistent_session_store::qwen38_persistent_session_store(
        std::string base_path,
        const uint32_t max_context,
        const uint64_t recurrent_state_bytes) {
    configure(std::move(base_path), max_context, recurrent_state_bytes);
}

void qwen38_persistent_session_store::configure(
        std::string base_path,
        const uint32_t max_context,
        const uint64_t recurrent_state_bytes) {
    base_path_ = std::move(base_path);
    root_path_ = base_path_.empty() ? std::string() : base_path_ + ".sessions";
    max_context_ = max_context;
    recurrent_state_bytes_ = recurrent_state_bytes;
}

bool qwen38_persistent_session_store::valid_session_id(const std::string &session_id) {
    if (session_id.empty() || session_id.size() > kSessionIdBytes - 1u) return false;
    for (const unsigned char character : session_id) {
        if (character < 0x20u || character == 0x7fu || character == '/' ||
            character == '\\') return false;
    }
    return true;
}

std::string qwen38_persistent_session_store::generate_session_id() {
    std::array<uint8_t, 16u> bytes{};
    if (!read_random_bytes(bytes.data(), bytes.size())) return {};
    std::ostringstream stream;
    stream << "axiom_";
    for (const uint8_t byte : bytes) {
        stream << std::hex << static_cast<unsigned>(byte >> 4u)
               << static_cast<unsigned>(byte & 0x0fu);
    }
    return stream.str();
}

std::string qwen38_persistent_session_store::namespace_text(
        const qwen38_session_key &key) {
    const std::array<const std::string *, 6u> fields = {
            &key.model_id, &key.target_identity, &key.dspark_identity,
            &key.config_signature, &key.profile, &key.session_id};
    std::string result;
    for (const std::string *field : fields) {
        result += std::to_string(field->size());
        result += ':';
        result += *field;
        result += ';';
    }
    return result;
}

std::string qwen38_persistent_session_store::namespace_id(
        const qwen38_session_key &key) {
    return hash128(namespace_text(key));
}

uint64_t qwen38_persistent_session_store::token_hash(
        const std::vector<uint32_t> &tokens) {
    return tokens.empty() ? 1469598103934665603ull : fnv_update(
            1469598103934665603ull, tokens.data(),
            tokens.size() * sizeof(uint32_t));
}

qwen38_session_paths qwen38_persistent_session_store::paths_for(
        const qwen38_session_key &key,
        const uint64_t generation) const {
    qwen38_session_paths paths;
    paths.namespace_id = namespace_id(key);
    paths.namespace_dir = root_path_ + "/" + paths.namespace_id;
    paths.session_stem = hash128(key.session_id);
    paths.manifest_path = paths.namespace_dir + "/" + paths.session_stem + ".manifest";
    paths.generation = generation;
    paths.tier_path = paths.namespace_dir + "/" + paths.session_stem + ".g" +
            std::to_string(generation) + ".kv";
    return paths;
}

int qwen38_persistent_session_store::prepare(
        const qwen38_session_key &key,
        const uint64_t generation,
        qwen38_session_paths *out,
        std::string *error) const {
    if (!out || !error || !enabled() || generation == 0u ||
        !valid_session_id(key.session_id) || key.model_id.empty() ||
        key.profile.empty()) {
        if (error) *error = "invalid persistent KV session key";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (!make_directory_recursive(root_path_)) {
        *error = "cannot create persistent KV root: " + root_path_;
        return AXIOM_ERR_IO;
    }
    if (!validate_private_directory(root_path_, error)) return AXIOM_ERR_IO;
    *out = paths_for(key, generation);
    if (!make_directory_recursive(out->namespace_dir)) {
        *error = "cannot create persistent KV namespace: " + out->namespace_dir;
        return AXIOM_ERR_IO;
    }
    if (!validate_private_directory(out->namespace_dir, error)) return AXIOM_ERR_IO;
    return AXIOM_OK;
}

int qwen38_persistent_session_store::load(
        const qwen38_session_key &key,
        qwen38_session_manifest *out,
        qwen38_session_paths *paths,
        bool *exists,
        std::string *error) const {
    if (!out || !paths || !exists || !error || !enabled() ||
        !valid_session_id(key.session_id)) {
        if (error) *error = "invalid persistent KV session load arguments";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = qwen38_session_manifest{};
    *exists = false;
    *paths = paths_for(key, 1u);
    struct stat status{};
    if (::stat(paths->manifest_path.c_str(), &status) != 0) {
        if (errno == ENOENT) return AXIOM_OK;
        *error = "cannot stat session manifest: " + std::string(std::strerror(errno));
        return AXIOM_ERR_IO;
    }
    const int fd = ::open(paths->manifest_path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        *error = "cannot open session manifest: " + std::string(std::strerror(errno));
        return AXIOM_ERR_IO;
    }
    manifest_disk_header header{};
    bool ok = read_full(fd, &header, sizeof(header));
    if (!ok) {
        ::close(fd);
        *error = "truncated session manifest header";
        return AXIOM_ERR_IO;
    }
    if (std::memcmp(header.magic, kManifestMagic, sizeof(header.magic)) != 0 ||
        header.abi != kManifestAbi || header.header_bytes != kManifestHeaderBytes ||
        header.generation == 0u || header.committed_tokens > max_context_ ||
        header.token_count > max_context_ || header.committed_tokens > header.token_count ||
        header.recurrent_bytes > kMaxRecurrentBytes ||
        (recurrent_state_bytes_ != 0u && header.recurrent_bytes != recurrent_state_bytes_)) {
        ::close(fd);
        *error = "session manifest contract mismatch";
        return AXIOM_ERR_IO;
    }
    uint64_t expected_payload = static_cast<uint64_t>(header.token_count) * sizeof(uint32_t);
    if (header.recurrent_bytes > std::numeric_limits<uint64_t>::max() - expected_payload) {
        ::close(fd);
        *error = "session manifest payload overflow";
        return AXIOM_ERR_IO;
    }
    expected_payload += header.recurrent_bytes;
    if (header.payload_bytes != expected_payload ||
        header.payload_bytes > static_cast<uint64_t>(max_context_) * sizeof(uint32_t) +
                kMaxRecurrentBytes) {
        ::close(fd);
        *error = "session manifest payload mismatch";
        return AXIOM_ERR_IO;
    }
    struct stat file_status{};
    if (::fstat(fd, &file_status) != 0 ||
        static_cast<uint64_t>(file_status.st_size) !=
                static_cast<uint64_t>(kManifestHeaderBytes) + header.payload_bytes) {
        ::close(fd);
        *error = "session manifest size mismatch";
        return AXIOM_ERR_IO;
    }
    std::vector<uint8_t> payload;
    try {
        payload.resize(static_cast<size_t>(header.payload_bytes));
    } catch (...) {
        ::close(fd);
        *error = "session manifest allocation failed";
        return AXIOM_ERR_BUDGET;
    }
    if (!payload.empty() && !read_full(fd, payload.data(), payload.size())) {
        ::close(fd);
        *error = "truncated session manifest payload";
        return AXIOM_ERR_IO;
    }
    ::close(fd);
    if (header.checksum != header_payload_checksum(header, payload)) {
        *error = "session manifest checksum mismatch";
        return AXIOM_ERR_IO;
    }
    std::string namespace_id_text;
    std::string session_id_text;
    std::string namespace_text_value;
    std::string profile_text;
    if (!read_text(header.namespace_id, sizeof(header.namespace_id), &namespace_id_text) ||
        !read_text(header.session_id, sizeof(header.session_id), &session_id_text) ||
        !read_text(header.namespace_text, sizeof(header.namespace_text), &namespace_text_value) ||
        !read_text(header.profile, sizeof(header.profile), &profile_text)) {
        *error = "session manifest string field is not terminated";
        return AXIOM_ERR_IO;
    }
    const std::string expected_namespace = namespace_id(key);
    const std::string expected_text = namespace_text(key);
    const size_t token_bytes = static_cast<size_t>(header.token_count) * sizeof(uint32_t);
    std::vector<uint32_t> token_ids(header.token_count);
    if (token_bytes != 0u) std::memcpy(token_ids.data(), payload.data(), token_bytes);
    if (namespace_id_text != expected_namespace || session_id_text != key.session_id ||
        namespace_text_value != expected_text || profile_text != key.profile ||
        header.token_hash != token_hash(token_ids)) {
        *error = "session manifest namespace or token checksum mismatch";
        return AXIOM_ERR_IO;
    }
    const size_t recurrent_offset = token_bytes;
    std::vector<uint8_t> recurrent(header.recurrent_bytes);
    if (header.recurrent_bytes != 0u) {
        std::memcpy(recurrent.data(), payload.data() + recurrent_offset,
                    static_cast<size_t>(header.recurrent_bytes));
    }
    *paths = paths_for(key, header.generation);
    *out = qwen38_session_manifest{};
    out->generation = header.generation;
    out->committed_tokens = header.committed_tokens;
    out->next_token = header.next_token;
    out->token_hash = header.token_hash;
    out->session_id = std::move(session_id_text);
    out->namespace_id = std::move(namespace_id_text);
    out->namespace_text = std::move(namespace_text_value);
    out->profile = std::move(profile_text);
    out->token_ids = std::move(token_ids);
    out->recurrent_state = std::move(recurrent);
    *exists = true;
    return AXIOM_OK;
}

int qwen38_persistent_session_store::save(
        const qwen38_session_key &key,
        const qwen38_session_paths &paths,
        const qwen38_session_manifest &manifest,
        std::string *error) const {
    if (!error || !enabled() || manifest.generation == 0u ||
        manifest.generation != paths.generation ||
        manifest.committed_tokens > max_context_ ||
        manifest.token_ids.size() > max_context_ ||
        manifest.token_ids.size() > std::numeric_limits<uint32_t>::max() ||
        manifest.committed_tokens > manifest.token_ids.size() ||
        manifest.recurrent_state.size() > std::numeric_limits<uint32_t>::max() ||
        manifest.recurrent_state.size() != recurrent_state_bytes_ ||
        manifest.session_id != key.session_id ||
        manifest.namespace_id != namespace_id(key) ||
        manifest.namespace_text != namespace_text(key) ||
        manifest.profile != key.profile) {
        if (error) *error = "invalid persistent KV manifest to save";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    qwen38_session_paths prepared;
    int rc = prepare(key, manifest.generation, &prepared, error);
    if (rc != AXIOM_OK) return rc;
    if (prepared.manifest_path != paths.manifest_path ||
        prepared.tier_path != paths.tier_path) {
        *error = "persistent KV manifest path does not match namespace";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint64_t token_bytes =
            static_cast<uint64_t>(manifest.token_ids.size()) * sizeof(uint32_t);
    const uint64_t recurrent_bytes = manifest.recurrent_state.size();
    if (token_bytes > std::numeric_limits<uint64_t>::max() - recurrent_bytes ||
        token_bytes + recurrent_bytes > std::numeric_limits<size_t>::max()) {
        *error = "persistent KV manifest payload overflow";
        return AXIOM_ERR_BUDGET;
    }
    std::vector<uint8_t> payload;
    try {
        payload.resize(static_cast<size_t>(token_bytes + recurrent_bytes));
    } catch (...) {
        *error = "persistent KV manifest allocation failed";
        return AXIOM_ERR_BUDGET;
    }
    if (token_bytes != 0u) {
        std::memcpy(payload.data(), manifest.token_ids.data(),
                    static_cast<size_t>(token_bytes));
    }
    if (recurrent_bytes != 0u) {
        std::memcpy(payload.data() + token_bytes, manifest.recurrent_state.data(),
                    static_cast<size_t>(recurrent_bytes));
    }
    manifest_disk_header header{};
    std::memcpy(header.magic, kManifestMagic, sizeof(header.magic));
    header.abi = kManifestAbi;
    header.header_bytes = kManifestHeaderBytes;
    header.generation = manifest.generation;
    header.committed_tokens = manifest.committed_tokens;
    header.next_token = manifest.next_token;
    header.token_count = static_cast<uint32_t>(manifest.token_ids.size());
    header.recurrent_bytes = static_cast<uint32_t>(recurrent_bytes);
    header.token_hash = token_hash(manifest.token_ids);
    header.payload_bytes = payload.size();
    if (!copy_text(header.namespace_id, sizeof(header.namespace_id), manifest.namespace_id) ||
        !copy_text(header.session_id, sizeof(header.session_id), manifest.session_id) ||
        !copy_text(header.namespace_text, sizeof(header.namespace_text), manifest.namespace_text) ||
        !copy_text(header.profile, sizeof(header.profile), manifest.profile)) {
        *error = "persistent KV manifest string field is too long";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    header.checksum = header_payload_checksum(header, payload);

    const uint64_t temp_id = g_temp_sequence.fetch_add(1u) + 1u;
    const std::string temporary = paths.manifest_path + ".tmp." +
            std::to_string(static_cast<unsigned long long>(::getpid())) + "." +
            std::to_string(static_cast<unsigned long long>(temp_id));
    const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR);
    if (fd < 0) {
        *error = "cannot create temporary session manifest: " +
                std::string(std::strerror(errno));
        return AXIOM_ERR_IO;
    }
    bool ok = write_full(fd, &header, sizeof(header));
    if (ok && !payload.empty()) ok = write_full(fd, payload.data(), payload.size());
    if (ok) ok = ::fsync(fd) == 0;
    const int close_rc = ::close(fd);
    if (close_rc != 0) ok = false;
    if (ok && ::rename(temporary.c_str(), paths.manifest_path.c_str()) != 0) ok = false;
    if (ok) ok = parent_directory_sync(paths.manifest_path);
    if (!ok) {
        *error = "cannot atomically publish session manifest: " +
                std::string(std::strerror(errno));
        (void)::unlink(temporary.c_str());
        return AXIOM_ERR_IO;
    }
    return AXIOM_OK;
}

int qwen38_persistent_session_store::prune_obsolete_generations(
        const qwen38_session_key &key,
        const qwen38_session_paths &committed_paths,
        uint64_t *removed_files,
        std::string *error) const {
    if (removed_files) *removed_files = 0u;
    if (!error || !enabled() || committed_paths.generation == 0u ||
        !valid_session_id(key.session_id)) {
        if (error) *error = "invalid persistent KV generation GC arguments";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const qwen38_session_paths expected = paths_for(key, committed_paths.generation);
    if (expected.namespace_dir != committed_paths.namespace_dir ||
        expected.namespace_id != committed_paths.namespace_id ||
        expected.session_stem != committed_paths.session_stem ||
        expected.manifest_path != committed_paths.manifest_path ||
        expected.tier_path != committed_paths.tier_path) {
        *error = "persistent KV generation GC path does not match namespace";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    DIR *directory = ::opendir(committed_paths.namespace_dir.c_str());
    if (!directory) {
        *error = "cannot open persistent KV namespace for generation GC: " +
                std::string(std::strerror(errno));
        return AXIOM_ERR_IO;
    }
    const int directory_fd = ::dirfd(directory);
    bool ok = directory_fd >= 0;
    uint64_t removed = 0u;
    int saved_errno = ok ? 0 : errno;
    while (ok) {
        errno = 0;
        dirent *entry = ::readdir(directory);
        if (!entry) {
            if (errno != 0) {
                ok = false;
                saved_errno = errno;
            }
            break;
        }
        const std::string name(entry->d_name);
        uint64_t generation = 0u;
        if (!parse_generation_filename(name, committed_paths.session_stem, &generation) ||
            generation >= committed_paths.generation) {
            continue;
        }
        struct stat status{};
        if (::fstatat(directory_fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
            if (errno == ENOENT) continue;
            ok = false;
            saved_errno = errno;
            break;
        }
        if (!S_ISREG(status.st_mode)) continue;
        if (::unlinkat(directory_fd, name.c_str(), 0) != 0) {
            if (errno == ENOENT) continue;
            ok = false;
            saved_errno = errno;
            break;
        }
        ++removed;
    }
    if (ok && removed != 0u && ::fsync(directory_fd) != 0) {
        ok = false;
        saved_errno = errno;
    }
    ::closedir(directory);
    if (removed_files) *removed_files = removed;
    if (!ok) {
        *error = "persistent KV generation GC failed: " +
                std::string(std::strerror(saved_errno));
        return AXIOM_ERR_IO;
    }
    return AXIOM_OK;
}

int qwen38_persistent_session_store::prune_stale_sessions(
        const qwen38_session_gc_policy &policy,
        qwen38_session_gc_result *result,
        std::string *error) const {
    if (result) *result = qwen38_session_gc_result{};
    if (!result || !error || !enabled() || policy.now_unix_seconds == 0u ||
        policy.scan_namespace_budget == 0u || policy.scan_artifact_budget == 0u ||
        policy.scan_namespace_budget > std::numeric_limits<size_t>::max() ||
        policy.scan_artifact_budget > std::numeric_limits<size_t>::max()) {
        if (error) *error = "invalid persistent KV session lifecycle GC arguments";
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    error->clear();

    try {
    scoped_fd root_handle(::open(
            root_path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW));
    const int root_fd = root_handle.get();
    if (root_fd < 0) {
        if (errno == ENOENT) return AXIOM_OK;
        *error = "cannot open persistent KV session root for lifecycle GC: " +
                std::string(std::strerror(errno));
        return AXIOM_ERR_IO;
    }
    struct stat root_status{};
    const int root_stat_rc = ::fstat(root_fd, &root_status);
    if (root_stat_rc != 0 || !private_owned_directory(root_status)) {
        const int saved_errno = root_stat_rc == 0 ? EACCES : errno;
        *error = "persistent KV session root is not private and owned by the daemon";
        if (saved_errno != 0) {
            *error += ": " + std::string(std::strerror(saved_errno));
        }
        return AXIOM_ERR_IO;
    }
    scoped_fd scan_handle(::fcntl(root_fd, F_DUPFD_CLOEXEC, 0));
    if (scan_handle.get() < 0) {
        const int saved_errno = errno;
        *error = "cannot duplicate persistent KV session root for lifecycle GC: " +
                std::string(std::strerror(saved_errno));
        return AXIOM_ERR_IO;
    }
    DIR *raw_root = ::fdopendir(scan_handle.get());
    if (!raw_root) {
        const int saved_errno = errno;
        *error = "cannot scan persistent KV session root for lifecycle GC: " +
                std::string(std::strerror(saved_errno));
        return AXIOM_ERR_IO;
    }
    (void)scan_handle.release();
    scoped_directory root(raw_root);

    std::unordered_set<std::string> protected_namespaces;
    try {
        protected_namespaces.reserve(policy.protected_namespace_ids.size());
        for (const std::string &namespace_id_value : policy.protected_namespace_ids) {
            if (is_lower_hex_32(namespace_id_value)) {
                protected_namespaces.insert(namespace_id_value);
            }
        }
    } catch (...) {
        *error = "persistent KV session lifecycle protection set allocation failed";
        return AXIOM_ERR_BUDGET;
    }
    std::vector<gc_namespace_entry> sessions;
    std::vector<gc_namespace_entry> tombstones;
    size_t total_artifacts = 0u;
    bool scan_ok = true;
    int scan_errno = 0;
    while (scan_ok) {
        errno = 0;
        dirent *item = ::readdir(root.get());
        if (!item) {
            if (errno != 0) {
                scan_ok = false;
                scan_errno = errno;
            }
            break;
        }
        const std::string name(item->d_name);
        if (name == "." || name == "..") continue;
        std::string namespace_id_value;
        const bool tombstone = parse_gc_tombstone(name, &namespace_id_value);
        if (!tombstone) {
            if (!is_lower_hex_32(name)) {
                if (result->unsafe_namespaces == 0u) {
                    *error = "unknown persistent KV session root entry: " + name;
                }
                ++result->unsafe_namespaces;
                continue;
            }
            namespace_id_value = name;
            ++result->scanned_sessions;
        }
        gc_namespace_entry entry;
        std::string scan_error;
        int namespace_scan_status = AXIOM_OK;
        if (!scan_gc_namespace(
                    root_fd, name, namespace_id_value,
                    static_cast<size_t>(policy.scan_artifact_budget),
                    &total_artifacts, &namespace_scan_status,
                    &entry, &scan_error)) {
            if (namespace_scan_status != AXIOM_OK) {
                *error = scan_error;
                return namespace_scan_status;
            }
            if (result->unsafe_namespaces == 0u) *error = scan_error;
            ++result->unsafe_namespaces;
            continue;
        }
        if (sessions.size() + tombstones.size() >=
            static_cast<size_t>(policy.scan_namespace_budget)) {
            scan_ok = false;
            scan_errno = EOVERFLOW;
            break;
        }
        if (tombstone) {
            try {
                tombstones.push_back(std::move(entry));
            } catch (...) {
                scan_ok = false;
                scan_errno = ENOMEM;
            }
            continue;
        }
        entry.protected_namespace =
                protected_namespaces.find(namespace_id_value) !=
                protected_namespaces.end();
        if (entry.protected_namespace) ++result->protected_sessions;
        try {
            sessions.push_back(std::move(entry));
        } catch (...) {
            scan_ok = false;
            scan_errno = ENOMEM;
            break;
        }
    }
    root.reset();
    if (!scan_ok) {
        *error = "persistent KV session lifecycle GC root scan failed: " +
                std::string(std::strerror(scan_errno));
        return (scan_errno == ENOMEM || scan_errno == EOVERFLOW)
                ? AXIOM_ERR_BUDGET
                : AXIOM_ERR_IO;
    }

    /* A prior crash may have occurred after the atomic namespace rename but
     * before the unlink phase. Finish those tombstones before selecting new
     * victims; they are already invisible to session lookup. */
    for (const gc_namespace_entry &entry : tombstones) {
        if (!remove_gc_namespace(
                    root_fd, entry, &result->removed_files,
                    &result->reclaimed_bytes, error)) {
            result->retained_sessions = result->scanned_sessions;
            return AXIOM_ERR_IO;
        }
    }

    uint64_t selected = 0u;
    if (policy.ttl_seconds != 0u) {
        for (gc_namespace_entry &entry : sessions) {
            if (entry.protected_namespace ||
                entry.last_used_unix_seconds > policy.now_unix_seconds) {
                continue;
            }
            const uint64_t age =
                    policy.now_unix_seconds - entry.last_used_unix_seconds;
            if (age >= policy.ttl_seconds) {
                entry.remove = true;
                ++selected;
            }
        }
    }
    std::sort(sessions.begin(), sessions.end(),
              [](const gc_namespace_entry &left,
                 const gc_namespace_entry &right) {
                  if (left.last_used_unix_seconds != right.last_used_unix_seconds) {
                      return left.last_used_unix_seconds < right.last_used_unix_seconds;
                  }
                  return left.namespace_id < right.namespace_id;
              });
    const uint64_t managed_sessions = static_cast<uint64_t>(sessions.size());
    uint64_t projected_sessions = managed_sessions > selected
            ? managed_sessions - selected : 0u;
    if (policy.max_sessions != 0u &&
        projected_sessions > policy.max_sessions) {
        for (gc_namespace_entry &entry : sessions) {
            if (projected_sessions <= policy.max_sessions) break;
            if (entry.protected_namespace || entry.remove) continue;
            entry.remove = true;
            ++selected;
            --projected_sessions;
        }
    }

    result->retained_sessions = result->scanned_sessions;
    for (gc_namespace_entry &entry : sessions) {
        if (!entry.remove) continue;
        const uint64_t sequence = g_temp_sequence.fetch_add(1u) + 1u;
        const std::string tombstone_name = ".gc." + entry.namespace_id + "." +
                std::to_string(static_cast<unsigned long long>(::getpid())) + "." +
                std::to_string(static_cast<unsigned long long>(policy.now_unix_seconds)) + "." +
                std::to_string(static_cast<unsigned long long>(sequence));
        if (!rename_directory_noreplace(
                    root_fd, entry.directory_name, tombstone_name)) {
            const int saved_errno = errno;
            *error = "cannot atomically tombstone stale persistent KV session: " +
                    std::string(std::strerror(saved_errno));
            return AXIOM_ERR_IO;
        }
        if (::fsync(root_fd) != 0) {
            const int saved_errno = errno;
            *error = "cannot sync persistent KV session tombstone rename: " +
                    std::string(std::strerror(saved_errno));
            return AXIOM_ERR_IO;
        }
        entry.directory_name = tombstone_name;
        ++result->removed_sessions;
        if (result->retained_sessions != 0u) --result->retained_sessions;
        if (policy.before_unlink) {
            policy.before_unlink(
                    root_path_ + "/" + tombstone_name,
                    entry.files,
                    policy.before_unlink_context);
        }
        if (!remove_gc_namespace(
                    root_fd, entry, &result->removed_files,
                    &result->reclaimed_bytes, error)) {
            return AXIOM_ERR_IO;
        }
    }
    return AXIOM_OK;
    } catch (const std::bad_alloc &) {
        *error = "persistent KV session lifecycle allocation failed";
        return AXIOM_ERR_BUDGET;
    } catch (...) {
        *error = "persistent KV session lifecycle failed with an unexpected exception";
        return AXIOM_ERR_RUNTIME;
    }
}

}  // namespace qwen38
}  // namespace axiom
