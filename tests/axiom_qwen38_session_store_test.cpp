#include "axiom/qwen38_session_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstring>
#include <openssl/evp.h>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

bool expect(bool condition, const char *message) {
    if (condition) return true;
    std::fprintf(stderr, "qwen38-session-store-test: %s\n", message);
    return false;
}

// Independent, contiguous on-disk oracle. Production seals three disjoint
// vectors, so this catches offsets/order mistakes and proves the v1 wire hash.
bool verify_wire(const std::string &path, uint32_t version) {
    std::ifstream input(path, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 4096u) return false;
    uint32_t actual_version = 0u;
    std::memcpy(&actual_version, bytes.data() + 8u, sizeof(actual_version));
    if (actual_version != version) return false;
    uint64_t stored = 0u;
    std::memcpy(&stored, bytes.data() + 56u, sizeof(stored));
    std::memset(bytes.data() + 56u, 0, sizeof(stored));
    if (version == 1u) {
        uint64_t hash = 1469598103934665603ull;
        for (unsigned char byte : bytes) { hash ^= byte; hash *= 1099511628211ull; }
        return stored == hash;
    }
    // Fixed public v2 disk layout: reserved digest starts after speculative_bytes.
    constexpr size_t digest_offset = 64u + 33u + 128u + 1536u + 32u + 4u;
    unsigned char expected[32], actual[EVP_MAX_MD_SIZE];
    std::memcpy(expected, bytes.data() + digest_offset, sizeof(expected));
    std::memset(bytes.data() + digest_offset, 0, sizeof(expected));
    unsigned int digest_bytes = 0;
    return stored == 0u && EVP_Digest(bytes.data(), bytes.size(), actual,
            &digest_bytes, EVP_sha256(), nullptr) == 1 && digest_bytes == 32u &&
            std::memcmp(expected, actual, 32u) == 0;
}

bool create_empty_file(const std::string &path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    return ::close(fd) == 0;
}

bool set_mtime(const std::string &path, const time_t seconds) {
    struct timespec times[2]{};
    times[0].tv_sec = seconds;
    times[1].tv_sec = seconds;
    return ::utimensat(AT_FDCWD, path.c_str(), times, AT_SYMLINK_NOFOLLOW) == 0;
}

size_t open_fd_count() {
    std::error_code error;
    size_t count = 0u;
    for (std::filesystem::directory_iterator iterator("/proc/self/fd", error), end;
         !error && iterator != end; iterator.increment(error)) {
        ++count;
    }
    return error ? std::numeric_limits<size_t>::max() : count;
}

bool create_lifecycle_fixture(
        axiom::qwen38::qwen38_persistent_session_store *store,
        const axiom::qwen38::qwen38_session_key &key,
        const uint64_t generation,
        const time_t timestamp,
        axiom::qwen38::qwen38_session_paths *paths,
        std::string *error) {
    if (!store || !paths || !error ||
        store->prepare(key, generation, paths, error) != AXIOM_OK ||
        !create_empty_file(paths->tier_path)) {
        return false;
    }
    axiom::qwen38::qwen38_session_manifest manifest;
    manifest.generation = generation;
    manifest.committed_tokens = 1u;
    manifest.next_token = 77u;
    manifest.session_id = key.session_id;
    manifest.namespace_id =
            axiom::qwen38::qwen38_persistent_session_store::namespace_id(key);
    manifest.namespace_text =
            axiom::qwen38::qwen38_persistent_session_store::namespace_text(key);
    manifest.profile = key.profile;
    manifest.token_ids = {42u};
    manifest.recurrent_state.resize(32u, 0x31u);
    return store->save(key, *paths, manifest, error) == AXIOM_OK &&
            set_mtime(paths->tier_path, timestamp) &&
            set_mtime(paths->manifest_path, timestamp) &&
            set_mtime(paths->namespace_dir, timestamp);
}

struct hardlink_race_context {
    std::string outside_file;
    std::string tombstone_directory;
    bool invoked = false;
};

void replace_validated_tier_with_hardlink(
        const std::string &tombstone_directory,
        const std::vector<std::string> &validated_files,
        void *opaque) {
    auto *context = static_cast<hardlink_race_context *>(opaque);
    if (!context) return;
    for (const std::string &name : validated_files) {
        if (name.size() < 3u || name.compare(name.size() - 3u, 3u, ".kv") != 0) {
            continue;
        }
        const std::string target = tombstone_directory + "/" + name;
        if (::unlink(target.c_str()) == 0 &&
            ::link(context->outside_file.c_str(), target.c_str()) == 0) {
            context->tombstone_directory = tombstone_directory;
            context->invoked = true;
        }
        return;
    }
}

}  // namespace

int main(int argc, char **argv) {
    const uint32_t writer_version = argc == 2 && std::string(argv[1]) == "--v2" ? 2u : 1u;
    if (argc > 2 || (argc == 2 && writer_version != 2u)) return 2;
    char directory_template[] = "/tmp/axiom-qwen38-session-test-XXXXXX";
    char *directory = ::mkdtemp(directory_template);
    if (!expect(directory != nullptr, "mkdtemp failed")) return 1;
    const std::string base = std::string(directory) + "/kv-tier.bin";
    constexpr uint64_t kSpeculativeStateBytes = 24u;

    axiom::qwen38::qwen38_persistent_session_store store(
            base, 4096u, 32u, kSpeculativeStateBytes);
    if (!expect(store.set_manifest_version(writer_version), "writer version rejected") ||
        !expect(!store.set_manifest_version(3u), "unknown writer version accepted") ||
        !expect(store.manifest_version() == writer_version, "invalid setter changed version")) return 1;
    axiom::qwen38::qwen38_session_key key;
    key.session_id = "session-alpha";
    key.model_id = "qwen3.8-27b-nvfp4";
    key.target_identity = "target-a";
    key.dspark_identity = "draft-a";
    key.config_signature = "ctx=4096;page=256;yarn=4;kvabi=1";
    key.profile = "ultra-fast";

    axiom::qwen38::qwen38_session_paths paths;
    std::string error;
    bool ok = store.prepare(key, 7u, &paths, &error) == AXIOM_OK;
    axiom::qwen38::qwen38_session_manifest manifest;
    manifest.generation = 7u;
    manifest.committed_tokens = 3u;
    manifest.next_token = 99u;
    manifest.session_id = key.session_id;
    manifest.namespace_id = axiom::qwen38::qwen38_persistent_session_store::namespace_id(key);
    manifest.namespace_text = axiom::qwen38::qwen38_persistent_session_store::namespace_text(key);
    manifest.profile = key.profile;
    manifest.token_ids = {11u, 22u, 33u, 44u};
    manifest.recurrent_state.resize(32u, 0x5au);
    manifest.speculative_state.resize(kSpeculativeStateBytes, 0x3cu);
    ok = ok && store.save(key, paths, manifest, &error) == AXIOM_OK;
    ok = ok && expect(verify_wire(paths.manifest_path, writer_version),
                      "manifest did not match independent on-disk checksum oracle");

    // Both readers accept legacy files, independent of their chosen writer.
    for (uint32_t version : {1u, 2u, 1u, writer_version}) {
        store.set_manifest_version(version);
        ok = ok && store.save(key, paths, manifest, &error) == AXIOM_OK;
        ok = ok && expect(verify_wire(paths.manifest_path, version), "cross-version wire mismatch");
        axiom::qwen38::qwen38_session_manifest roundtrip;
        axiom::qwen38::qwen38_session_paths roundtrip_paths;
        bool roundtrip_exists = false;
        ok = ok && store.load(key, &roundtrip, &roundtrip_paths, &roundtrip_exists, &error) == AXIOM_OK;
        ok = ok && expect(roundtrip_exists && roundtrip.token_ids == manifest.token_ids &&
                         roundtrip.recurrent_state == manifest.recurrent_state &&
                         roundtrip.speculative_state == manifest.speculative_state,
                         "cross-version payload mismatch");
    }
    // Header identity, checksum, reserved tail and each payload region must all
    // be covered by integrity checks; no partial object may escape on failure.
    for (size_t offset : {size_t(8),size_t(12),size_t(28),size_t(48),size_t(56),size_t(1797),size_t(4000),
                          size_t(4096),size_t(4112),size_t(4144)}) {
        const int fd = ::open(paths.manifest_path.c_str(), O_RDWR | O_CLOEXEC);
        unsigned char byte = 0;
        bool changed = fd >= 0 && ::pread(fd, &byte, 1, offset) == 1;
        byte ^= 0x80;
        changed = changed && ::pwrite(fd, &byte, 1, offset) == 1;
        if (fd >= 0) ::close(fd);
        ok = ok && expect(changed, "corruption fixture write failed");
        axiom::qwen38::qwen38_session_manifest rejected;
        axiom::qwen38::qwen38_session_paths rejected_paths;
        bool rejected_exists = false;
        ok = ok && expect(store.load(key, &rejected, &rejected_paths, &rejected_exists, &error) == AXIOM_ERR_IO &&
                          !rejected_exists && rejected.token_ids.empty(), "corruption escaped validation");
        ok = ok && store.save(key, paths, manifest, &error) == AXIOM_OK;
    }

    axiom::qwen38::qwen38_session_manifest loaded;
    axiom::qwen38::qwen38_session_paths loaded_paths;
    bool exists = false;
    ok = ok && store.load(key, &loaded, &loaded_paths, &exists, &error) == AXIOM_OK;
    ok = ok && expect(exists, "saved manifest not found");
    ok = ok && expect(loaded.generation == 7u && loaded.committed_tokens == 3u,
                      "manifest scalar fields did not round-trip");
    ok = ok && expect(loaded.token_ids == manifest.token_ids &&
                              loaded.recurrent_state == manifest.recurrent_state &&
                              loaded.speculative_state == manifest.speculative_state,
                      "manifest payload did not round-trip");
    ok = ok && expect(loaded_paths.tier_path == paths.tier_path,
                      "generation-specific tier path changed");
    ok = ok && expect(paths.transaction_path == paths.tier_path + ".txn",
                      "generation transaction path is not deterministic");

    /* A new store instance models a process restart; no in-memory state is
     * allowed to make this load pass. */
    axiom::qwen38::qwen38_persistent_session_store restarted_store(
            base, 4096u, 32u, kSpeculativeStateBytes);
    restarted_store.set_manifest_version(writer_version);
    axiom::qwen38::qwen38_session_manifest restarted;
    axiom::qwen38::qwen38_session_paths restarted_paths;
    bool restarted_exists = false;
    ok = ok && restarted_store.load(
            key, &restarted, &restarted_paths, &restarted_exists, &error) == AXIOM_OK;
    ok = ok && expect(restarted_exists && restarted.generation == manifest.generation,
                      "manifest did not survive a store restart");
    ok = ok && expect(restarted.token_ids == manifest.token_ids &&
                              restarted.recurrent_state == manifest.recurrent_state &&
                              restarted.speculative_state == manifest.speculative_state,
                      "restart restore payload did not round-trip");

    /* A zero-byte speculative payload remains the legacy DSpark contract.
     * Exercise the explicit four-argument configure path so older manifests
     * remain loadable without synthesizing controller state. */
    const std::string legacy_base = std::string(directory) + "/legacy-zero.bin";
    {
        axiom::qwen38::qwen38_persistent_session_store empty_store(
                std::string(directory) + "/empty.bin", 4096u, 0u, 0u);
        auto empty_key = key;
        empty_key.session_id = "session-empty";
        axiom::qwen38::qwen38_session_paths empty_paths;
        ok = ok && empty_store.prepare(empty_key, 1u, &empty_paths, &error) == AXIOM_OK;
        auto empty = manifest;
        empty.generation = 1u;
        empty.committed_tokens = 0u;
        empty.token_ids.clear();
        empty.recurrent_state.clear();
        empty.speculative_state.clear();
        empty.session_id = empty_key.session_id;
        empty.namespace_id = axiom::qwen38::qwen38_persistent_session_store::namespace_id(empty_key);
        empty.namespace_text = axiom::qwen38::qwen38_persistent_session_store::namespace_text(empty_key);
        for (uint32_t version : {1u, 2u}) {
            empty_store.set_manifest_version(version);
            ok = ok && empty_store.save(empty_key, empty_paths, empty, &error) == AXIOM_OK;
            ok = ok && expect(verify_wire(empty_paths.manifest_path, version), "empty wire checksum failed");
            bool empty_exists = false;
            axiom::qwen38::qwen38_session_manifest restored_empty;
            ok = ok && empty_store.load(empty_key, &restored_empty, &empty_paths,
                                       &empty_exists, &error) == AXIOM_OK;
            ok = ok && expect(empty_exists && restored_empty.token_ids.empty() &&
                             restored_empty.recurrent_state.empty(), "empty roundtrip failed");
        }
    }
    axiom::qwen38::qwen38_persistent_session_store legacy_store;
    legacy_store.configure(legacy_base, 4096u, 32u, 0u);
    auto legacy_key = key;
    legacy_key.session_id = "session-legacy-zero";
    axiom::qwen38::qwen38_session_paths legacy_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &legacy_store, legacy_key, 1u, 100,
                              &legacy_paths, &error),
                      "could not create zero-byte speculative-state fixture");
    axiom::qwen38::qwen38_session_manifest legacy_loaded;
    axiom::qwen38::qwen38_session_paths legacy_loaded_paths;
    bool legacy_exists = false;
    ok = ok && legacy_store.load(
            legacy_key, &legacy_loaded, &legacy_loaded_paths,
            &legacy_exists, &error) == AXIOM_OK;
    ok = ok && expect(legacy_exists && legacy_loaded.speculative_state.empty(),
                      "zero-byte speculative state lost legacy compatibility");

    std::vector<uint32_t> resume_prompt;
    bool resume_usable = false;
    const std::vector<uint32_t> resume_suffix{55u, 66u};
    ok = ok && expect(axiom::qwen38::qwen38_build_stateful_resume_prompt(
                              manifest, resume_suffix, 2u, 3u, 1000u, 16u,
                              &resume_prompt, &resume_usable) == AXIOM_OK &&
                              resume_usable &&
                              resume_prompt == std::vector<uint32_t>(
                                      {11u, 22u, 33u, 99u, 2u, 55u, 66u}),
                      "stateful resume did not append pending token, ChatML close and tail");
    axiom::qwen38::qwen38_session_manifest stopped_manifest = manifest;
    stopped_manifest.next_token = 2u;
    ok = ok && expect(axiom::qwen38::qwen38_build_stateful_resume_prompt(
                              stopped_manifest, resume_suffix, 2u, 3u, 1000u, 16u,
                              &resume_prompt, &resume_usable) == AXIOM_OK &&
                              resume_usable &&
                              resume_prompt == std::vector<uint32_t>(
                                      {11u, 22u, 33u, 2u, 55u, 66u}),
                      "stateful resume duplicated an existing ChatML stop token");
    stopped_manifest.next_token = 3u;
    ok = ok && expect(axiom::qwen38::qwen38_build_stateful_resume_prompt(
                              stopped_manifest, resume_suffix, 2u, 3u, 1000u, 16u,
                              &resume_prompt, &resume_usable) == AXIOM_OK &&
                              !resume_usable && resume_prompt.empty(),
                      "stateful resume accepted a non-ChatML EOS cursor");
    ok = ok && expect(axiom::qwen38::qwen38_build_stateful_resume_prompt(
                              manifest, resume_suffix, 2u, 3u, 1000u, 6u,
                              &resume_prompt, &resume_usable) == AXIOM_OK &&
                              !resume_usable && resume_prompt.empty(),
                      "stateful resume exceeded its selected path budget");
    axiom::qwen38::qwen38_session_manifest malformed_manifest = manifest;
    malformed_manifest.committed_tokens = 9u;
    ok = ok && expect(axiom::qwen38::qwen38_build_stateful_resume_prompt(
                              malformed_manifest, resume_suffix, 2u, 3u, 1000u, 16u,
                              &resume_prompt, &resume_usable) == AXIOM_ERR_INVALID_ARGUMENT,
                      "stateful resume accepted a malformed durable watermark");

    const auto generation5 = store.paths_for(key, 5u);
    const auto generation6 = store.paths_for(key, 6u);
    const auto generation8 = store.paths_for(key, 8u);
    ok = ok && expect(create_empty_file(generation5.tier_path),
                      "could not create generation 5 GC fixture");
    ok = ok && expect(create_empty_file(generation6.tier_path),
                      "could not create generation 6 GC fixture");
    ok = ok && expect(create_empty_file(generation6.transaction_path),
                      "could not create generation 6 transaction GC fixture");
    ok = ok && expect(create_empty_file(generation5.transaction_path + ".tmp"),
                      "could not create generation 5 temporary transaction GC fixture");
    ok = ok && expect(create_empty_file(paths.tier_path),
                      "could not create committed generation GC fixture");
    ok = ok && expect(create_empty_file(generation8.tier_path),
                      "could not create future generation GC fixture");
    const std::string unrelated = paths.namespace_dir + "/unrelated.g1.kv";
    ok = ok && expect(create_empty_file(unrelated),
                      "could not create unrelated GC fixture");
    uint64_t removed_files = 0u;
    ok = ok && store.prune_obsolete_generations(
            key, paths, &removed_files, &error) == AXIOM_OK;
    ok = ok && expect(removed_files == 4u,
                      "GC did not report exactly the obsolete generations");
    ok = ok && expect(::access(generation5.tier_path.c_str(), F_OK) != 0 &&
                              ::access(generation6.tier_path.c_str(), F_OK) != 0 &&
                              ::access(generation6.transaction_path.c_str(), F_OK) != 0 &&
                              ::access((generation5.transaction_path + ".tmp").c_str(), F_OK) != 0,
                      "GC retained an obsolete generation");
    ok = ok && expect(::access(paths.tier_path.c_str(), F_OK) == 0 &&
                              ::access(generation8.tier_path.c_str(), F_OK) == 0 &&
                              ::access(unrelated.c_str(), F_OK) == 0,
                      "GC removed current, future, or unrelated state");
    removed_files = 99u;
    ok = ok && store.prune_obsolete_generations(
            key, paths, &removed_files, &error) == AXIOM_OK;
    ok = ok && expect(removed_files == 0u, "GC retry was not idempotent");

    axiom::qwen38::qwen38_session_key wrong_profile = key;
    wrong_profile.profile = "medium";
    axiom::qwen38::qwen38_session_manifest absent_manifest;
    axiom::qwen38::qwen38_session_paths absent_paths;
    bool absent = true;
    ok = ok && store.load(
            wrong_profile, &absent_manifest, &absent_paths, &absent, &error) == AXIOM_OK;
    ok = ok && expect(!absent, "profile namespace was not isolated");
    ok = ok && expect(axiom::qwen38::qwen38_persistent_session_store::valid_session_id(
                              "session-alpha"), "valid session id rejected");
    ok = ok && expect(!axiom::qwen38::qwen38_persistent_session_store::valid_session_id(
                              "bad/session"), "path traversal session id accepted");
    std::set<std::string> generated_session_ids;
    for (size_t index = 0u; index < 256u; ++index) {
        const std::string generated =
                axiom::qwen38::qwen38_persistent_session_store::generate_session_id();
        ok = ok && expect(generated.size() == 38u &&
                                  generated.compare(0u, 6u, "axiom_") == 0,
                          "generated session id has an invalid shape");
        ok = ok && expect(
                axiom::qwen38::qwen38_persistent_session_store::valid_session_id(generated),
                "generated session id was rejected");
        generated_session_ids.insert(generated);
    }
    ok = ok && expect(generated_session_ids.size() == 256u,
                      "generated session ids were not unique");

    /* Whole-session lifecycle GC is independent from generation pruning.
     * Exercise TTL, LRU pressure, active-session protection, unsafe namespace
     * fail-safe behavior and idempotence entirely inside the temporary root. */
    const std::string lifecycle_base = std::string(directory) + "/lifecycle.bin";
    axiom::qwen38::qwen38_persistent_session_store lifecycle_store(
            lifecycle_base, 4096u, 32u);
    auto lifecycle_key = key;
    lifecycle_key.session_id = "session-protected";
    axiom::qwen38::qwen38_session_paths protected_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &lifecycle_store, lifecycle_key, 1u, 100,
                              &protected_paths, &error),
                      "could not create protected lifecycle fixture");
    lifecycle_key.session_id = "session-old";
    axiom::qwen38::qwen38_session_paths old_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &lifecycle_store, lifecycle_key, 1u, 100,
                              &old_paths, &error),
                      "could not create old lifecycle fixture");
    lifecycle_key.session_id = "session-mid";
    axiom::qwen38::qwen38_session_paths mid_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &lifecycle_store, lifecycle_key, 1u, 500,
                              &mid_paths, &error),
                      "could not create middle lifecycle fixture");
    lifecycle_key.session_id = "session-new";
    axiom::qwen38::qwen38_session_paths new_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &lifecycle_store, lifecycle_key, 1u, 900,
                              &new_paths, &error),
                      "could not create new lifecycle fixture");

    const std::string unsafe_namespace(32u, 'f');
    const std::string unsafe_path = lifecycle_store.root_path() + "/" + unsafe_namespace;
    const std::string outside_sentinel = std::string(directory) + "/owner-sentinel";
    ok = ok && expect(::mkdir(unsafe_path.c_str(), 0700) == 0,
                      "could not create unsafe lifecycle namespace");
    ok = ok && expect(create_empty_file(outside_sentinel),
                      "could not create outside lifecycle sentinel");
    ok = ok && expect(::symlink(
                              outside_sentinel.c_str(),
                              (unsafe_path + "/owner-link").c_str()) == 0,
                      "could not create unsafe lifecycle symlink");
    const std::string unknown_root_entry =
            lifecycle_store.root_path() + "/operator-note";
    ok = ok && expect(create_empty_file(unknown_root_entry),
                      "could not create unknown lifecycle root fixture");
    const std::string malformed_namespace(32u, 'e');
    const std::string malformed_namespace_path =
            lifecycle_store.root_path() + "/" + malformed_namespace;
    const std::string malformed_artifact = malformed_namespace_path + "/" +
            std::string(32u, 'd') + ".g1.2.kv";
    ok = ok && expect(::mkdir(malformed_namespace_path.c_str(), 0700) == 0 &&
                              create_empty_file(malformed_artifact),
                      "could not create malformed lifecycle artifact fixture");
    const std::string malformed_tombstone_path = lifecycle_store.root_path() +
            "/.gc." + std::string(32u, 'c') + ".1";
    ok = ok && expect(::mkdir(malformed_tombstone_path.c_str(), 0700) == 0,
                      "could not create malformed lifecycle tombstone fixture");
    const std::string permissive_namespace_path =
            lifecycle_store.root_path() + "/" + std::string(32u, 'b');
    ok = ok && expect(::mkdir(permissive_namespace_path.c_str(), 0700) == 0 &&
                              ::chmod(permissive_namespace_path.c_str(), 0777) == 0,
                      "could not create permissive lifecycle namespace fixture");
    const std::string leading_zero_namespace_path =
            lifecycle_store.root_path() + "/" + std::string(32u, 'a');
    const std::string leading_zero_artifact = leading_zero_namespace_path + "/" +
            std::string(32u, 'a') + ".g01.kv";
    ok = ok && expect(::mkdir(leading_zero_namespace_path.c_str(), 0700) == 0 &&
                              create_empty_file(leading_zero_artifact),
                      "could not create leading-zero lifecycle artifact fixture");
    const std::string leading_zero_tombstone_path = lifecycle_store.root_path() +
            "/.gc." + std::string(32u, '9') + ".01.1000.1";
    ok = ok && expect(::mkdir(leading_zero_tombstone_path.c_str(), 0700) == 0,
                      "could not create leading-zero lifecycle tombstone fixture");
    const std::string hardlink_namespace_path =
            lifecycle_store.root_path() + "/" + std::string(32u, '8');
    const std::string hardlink_artifact = hardlink_namespace_path + "/" +
            std::string(32u, '8') + ".g1.kv";
    ok = ok && expect(::mkdir(hardlink_namespace_path.c_str(), 0700) == 0 &&
                              ::link(outside_sentinel.c_str(),
                                     hardlink_artifact.c_str()) == 0,
                      "could not create hardlink lifecycle fixture");

    axiom::qwen38::qwen38_session_gc_policy lifecycle_policy;
    lifecycle_policy.now_unix_seconds = 1000u;
    lifecycle_policy.ttl_seconds = 600u;
    lifecycle_policy.max_sessions = 2u;
    lifecycle_policy.protected_namespace_ids = {protected_paths.namespace_id};
    axiom::qwen38::qwen38_session_gc_result lifecycle_result;
    ok = ok && lifecycle_store.prune_stale_sessions(
            lifecycle_policy, &lifecycle_result, &error) == AXIOM_OK;
    ok = ok && expect(lifecycle_result.scanned_sessions == 9u &&
                              lifecycle_result.removed_sessions == 2u &&
                              lifecycle_result.retained_sessions == 7u,
                      "lifecycle GC TTL/LRU accounting is incorrect");
    ok = ok && expect(lifecycle_result.protected_sessions == 1u &&
                              lifecycle_result.unsafe_namespaces == 8u &&
                              lifecycle_result.removed_files == 4u,
                      "lifecycle GC protection or safety accounting is incorrect");
    ok = ok && expect(!error.empty(),
                      "lifecycle GC did not describe the retained unsafe namespace");
    ok = ok && expect(::access(old_paths.namespace_dir.c_str(), F_OK) != 0 &&
                              ::access(mid_paths.namespace_dir.c_str(), F_OK) != 0,
                      "lifecycle GC retained an expired or LRU session");
    ok = ok && expect(::access(protected_paths.namespace_dir.c_str(), F_OK) == 0 &&
                              ::access(new_paths.namespace_dir.c_str(), F_OK) == 0 &&
                              ::access((unsafe_path + "/owner-link").c_str(), F_OK) == 0 &&
                              ::access(unknown_root_entry.c_str(), F_OK) == 0 &&
                              ::access(malformed_artifact.c_str(), F_OK) == 0 &&
                              ::access(malformed_tombstone_path.c_str(), F_OK) == 0 &&
                              ::access(permissive_namespace_path.c_str(), F_OK) == 0 &&
                              ::access(leading_zero_artifact.c_str(), F_OK) == 0 &&
                              ::access(leading_zero_tombstone_path.c_str(), F_OK) == 0 &&
                              ::access(hardlink_artifact.c_str(), F_OK) == 0 &&
                              ::access(outside_sentinel.c_str(), F_OK) == 0,
                      "lifecycle GC removed protected, recent, or unsafe state");
    axiom::qwen38::qwen38_session_gc_result retry_result;
    ok = ok && lifecycle_store.prune_stale_sessions(
            lifecycle_policy, &retry_result, &error) == AXIOM_OK;
    ok = ok && expect(retry_result.removed_sessions == 0u &&
                              retry_result.removed_files == 0u,
                      "lifecycle GC retry was not idempotent");
    ok = ok && expect(::chmod(lifecycle_store.root_path().c_str(), 0777) == 0,
                      "could not make lifecycle root permissive");
    axiom::qwen38::qwen38_session_gc_result permissive_root_result;
    ok = ok && expect(lifecycle_store.prune_stale_sessions(
                              lifecycle_policy, &permissive_root_result, &error) ==
                              AXIOM_ERR_IO,
                      "lifecycle GC accepted a non-private root");
    ok = ok && expect(::chmod(lifecycle_store.root_path().c_str(), 0700) == 0,
                      "could not restore private lifecycle root mode");
    axiom::qwen38::qwen38_session_gc_policy budget_policy = lifecycle_policy;
    budget_policy.scan_artifact_budget = 1u;
    const size_t fd_count_before = open_fd_count();
    for (size_t attempt = 0u; attempt < 16u; ++attempt) {
        axiom::qwen38::qwen38_session_gc_result budget_result;
        ok = ok && expect(lifecycle_store.prune_stale_sessions(
                                  budget_policy, &budget_result, &error) ==
                                  AXIOM_ERR_BUDGET,
                          "lifecycle GC did not fail closed on scan budget");
    }
    const size_t fd_count_after = open_fd_count();
    ok = ok && expect(fd_count_before != std::numeric_limits<size_t>::max() &&
                              fd_count_before == fd_count_after,
                      "lifecycle GC scan-budget retries leaked file descriptors");

    const std::string namespace_budget_base =
            std::string(directory) + "/namespace-budget.bin";
    axiom::qwen38::qwen38_persistent_session_store namespace_budget_store(
            namespace_budget_base, 4096u, 32u);
    lifecycle_key.session_id = "session-namespace-budget-a";
    axiom::qwen38::qwen38_session_paths namespace_budget_paths_a;
    ok = ok && expect(create_lifecycle_fixture(
                              &namespace_budget_store, lifecycle_key, 1u, 100,
                              &namespace_budget_paths_a, &error),
                      "could not create first namespace-budget fixture");
    lifecycle_key.session_id = "session-namespace-budget-b";
    axiom::qwen38::qwen38_session_paths namespace_budget_paths_b;
    ok = ok && expect(create_lifecycle_fixture(
                              &namespace_budget_store, lifecycle_key, 1u, 200,
                              &namespace_budget_paths_b, &error),
                      "could not create second namespace-budget fixture");
    axiom::qwen38::qwen38_session_gc_policy namespace_budget_policy;
    namespace_budget_policy.now_unix_seconds = 1000u;
    namespace_budget_policy.scan_namespace_budget = 1u;
    axiom::qwen38::qwen38_session_gc_result namespace_budget_result;
    ok = ok && expect(namespace_budget_store.prune_stale_sessions(
                              namespace_budget_policy,
                              &namespace_budget_result,
                              &error) == AXIOM_ERR_BUDGET,
                      "lifecycle GC did not report a namespace scan budget breach");

    const std::string race_base = std::string(directory) + "/race.bin";
    axiom::qwen38::qwen38_persistent_session_store race_store(
            race_base, 4096u, 32u);
    lifecycle_key.session_id = "session-hardlink-race";
    axiom::qwen38::qwen38_session_paths race_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &race_store, lifecycle_key, 1u, 100,
                              &race_paths, &error),
                      "could not create hardlink race fixture");
    hardlink_race_context race_context;
    race_context.outside_file = std::string(directory) + "/race-owner-sentinel";
    ok = ok && expect(create_empty_file(race_context.outside_file),
                      "could not create hardlink race sentinel");
    axiom::qwen38::qwen38_session_gc_policy race_policy;
    race_policy.now_unix_seconds = 1000u;
    race_policy.ttl_seconds = 1u;
    race_policy.before_unlink = replace_validated_tier_with_hardlink;
    race_policy.before_unlink_context = &race_context;
    axiom::qwen38::qwen38_session_gc_result race_result;
    ok = ok && expect(race_store.prune_stale_sessions(
                              race_policy, &race_result, &error) == AXIOM_ERR_IO,
                      "lifecycle GC followed a hardlink introduced after scan");
    ok = ok && expect(race_context.invoked && race_result.removed_sessions == 1u &&
                              ::access(race_context.outside_file.c_str(), F_OK) == 0 &&
                              ::access(race_context.tombstone_directory.c_str(), F_OK) == 0,
                      "hardlink race did not retain the tombstone and owner file");

    /* Model a crash immediately after the atomic namespace rename. A later
     * pass must finish the tombstone without counting a second session. */
    lifecycle_key.session_id = "session-tombstone";
    axiom::qwen38::qwen38_session_paths tombstone_paths;
    ok = ok && expect(create_lifecycle_fixture(
                              &lifecycle_store, lifecycle_key, 1u, 950,
                              &tombstone_paths, &error),
                      "could not create tombstone lifecycle fixture");
    const std::string tombstone_name = ".gc." + tombstone_paths.namespace_id +
            ".123.1000.1";
    const std::string tombstone_path = lifecycle_store.root_path() + "/" + tombstone_name;
    ok = ok && expect(::rename(
                              tombstone_paths.namespace_dir.c_str(),
                              tombstone_path.c_str()) == 0,
                      "could not create lifecycle tombstone fixture");
    axiom::qwen38::qwen38_session_gc_policy tombstone_policy;
    tombstone_policy.now_unix_seconds = 1001u;
    axiom::qwen38::qwen38_session_gc_result tombstone_result;
    ok = ok && lifecycle_store.prune_stale_sessions(
            tombstone_policy, &tombstone_result, &error) == AXIOM_OK;
    ok = ok && expect(tombstone_result.removed_sessions == 0u &&
                              tombstone_result.removed_files == 2u &&
                              ::access(tombstone_path.c_str(), F_OK) != 0,
                      "lifecycle GC did not finish a crash tombstone");

    const int corrupt_fd = ::open(paths.manifest_path.c_str(), O_RDWR | O_CLOEXEC);
    if (ok) ok = expect(corrupt_fd >= 0, "could not open manifest for corruption gate");
    if (ok) {
        struct stat manifest_status{};
        ok = expect(::fstat(corrupt_fd, &manifest_status) == 0 &&
                            manifest_status.st_size >
                                    static_cast<off_t>(manifest.speculative_state.size()),
                    "could not locate speculative payload for corruption gate");
    }
    if (ok) {
        const uint8_t corrupt_byte = 0xffu;
        struct stat manifest_status{};
        ok = expect(::fstat(corrupt_fd, &manifest_status) == 0 &&
                            ::pwrite(corrupt_fd, &corrupt_byte,
                                     sizeof(corrupt_byte),
                                     manifest_status.st_size - 1) == 1,
                    "could not corrupt speculative manifest payload");
    }
    if (corrupt_fd >= 0) ::close(corrupt_fd);
    bool corrupt_exists = false;
    const int corrupt_rc = restarted_store.load(
            key, &loaded, &loaded_paths, &corrupt_exists, &error);
    ok = ok && expect(corrupt_rc == AXIOM_ERR_IO,
                      "corrupt speculative payload was not rejected fail-closed");

    if (ok) {
        ok = expect(store.save(key, paths, manifest, &error) == AXIOM_OK,
                    "could not restore manifest before truncation gate");
    }
    int truncated_fd = -1;
    if (ok) {
        truncated_fd = ::open(paths.manifest_path.c_str(), O_WRONLY | O_CLOEXEC);
        ok = expect(truncated_fd >= 0,
                    "could not open manifest for truncation gate");
    }
    if (ok) {
        struct stat manifest_status{};
        ok = expect(::fstat(truncated_fd, &manifest_status) == 0 &&
                            manifest_status.st_size > 0 &&
                            ::ftruncate(truncated_fd, manifest_status.st_size - 1) == 0,
                    "could not truncate speculative manifest payload");
    }
    if (truncated_fd >= 0) ::close(truncated_fd);
    bool truncated_exists = false;
    const int truncated_rc = restarted_store.load(
            key, &loaded, &loaded_paths, &truncated_exists, &error);
    ok = ok && expect(truncated_rc == AXIOM_ERR_IO,
                      "truncated speculative payload was not rejected fail-closed");

    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    ok = ok && !cleanup_error;
    return ok ? 0 : 1;
}
