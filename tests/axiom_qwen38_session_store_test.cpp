#include "axiom/qwen38_session_store.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <set>
#include <string>
#include <unistd.h>

namespace {

bool expect(bool condition, const char *message) {
    if (condition) return true;
    std::fprintf(stderr, "qwen38-session-store-test: %s\n", message);
    return false;
}

bool create_empty_file(const std::string &path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return false;
    return ::close(fd) == 0;
}

}  // namespace

int main() {
    char directory_template[] = "/tmp/axiom-qwen38-session-test-XXXXXX";
    char *directory = ::mkdtemp(directory_template);
    if (!expect(directory != nullptr, "mkdtemp failed")) return 1;
    const std::string base = std::string(directory) + "/kv-tier.bin";

    axiom::qwen38::qwen38_persistent_session_store store(base, 4096u, 32u);
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
    ok = ok && store.save(key, paths, manifest, &error) == AXIOM_OK;

    axiom::qwen38::qwen38_session_manifest loaded;
    axiom::qwen38::qwen38_session_paths loaded_paths;
    bool exists = false;
    ok = ok && store.load(key, &loaded, &loaded_paths, &exists, &error) == AXIOM_OK;
    ok = ok && expect(exists, "saved manifest not found");
    ok = ok && expect(loaded.generation == 7u && loaded.committed_tokens == 3u,
                      "manifest scalar fields did not round-trip");
    ok = ok && expect(loaded.token_ids == manifest.token_ids &&
                              loaded.recurrent_state == manifest.recurrent_state,
                      "manifest payload did not round-trip");
    ok = ok && expect(loaded_paths.tier_path == paths.tier_path,
                      "generation-specific tier path changed");

    /* A new store instance models a process restart; no in-memory state is
     * allowed to make this load pass. */
    axiom::qwen38::qwen38_persistent_session_store restarted_store(
            base, 4096u, 32u);
    axiom::qwen38::qwen38_session_manifest restarted;
    axiom::qwen38::qwen38_session_paths restarted_paths;
    bool restarted_exists = false;
    ok = ok && restarted_store.load(
            key, &restarted, &restarted_paths, &restarted_exists, &error) == AXIOM_OK;
    ok = ok && expect(restarted_exists && restarted.generation == manifest.generation,
                      "manifest did not survive a store restart");
    ok = ok && expect(restarted.token_ids == manifest.token_ids &&
                              restarted.recurrent_state == manifest.recurrent_state,
                      "restart restore payload did not round-trip");

    const auto generation5 = store.paths_for(key, 5u);
    const auto generation6 = store.paths_for(key, 6u);
    const auto generation8 = store.paths_for(key, 8u);
    ok = ok && expect(create_empty_file(generation5.tier_path),
                      "could not create generation 5 GC fixture");
    ok = ok && expect(create_empty_file(generation6.tier_path),
                      "could not create generation 6 GC fixture");
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
    ok = ok && expect(removed_files == 2u,
                      "GC did not report exactly the obsolete generations");
    ok = ok && expect(::access(generation5.tier_path.c_str(), F_OK) != 0 &&
                              ::access(generation6.tier_path.c_str(), F_OK) != 0,
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

    const int corrupt_fd = ::open(paths.manifest_path.c_str(), O_WRONLY | O_CLOEXEC);
    if (ok) ok = expect(corrupt_fd >= 0, "could not open manifest for corruption gate");
    if (ok) {
        const uint8_t corrupt_byte = 0xffu;
        ok = expect(::pwrite(corrupt_fd, &corrupt_byte, sizeof(corrupt_byte), 4096) == 1,
                    "could not corrupt manifest payload");
    }
    if (corrupt_fd >= 0) ::close(corrupt_fd);
    bool corrupt_exists = false;
    const int corrupt_rc = restarted_store.load(
            key, &loaded, &loaded_paths, &corrupt_exists, &error);
    ok = ok && expect(corrupt_rc == AXIOM_ERR_IO,
                      "corrupt manifest was not rejected fail-closed");

    std::error_code cleanup_error;
    std::filesystem::remove_all(directory, cleanup_error);
    ok = ok && !cleanup_error;
    return ok ? 0 : 1;
}
