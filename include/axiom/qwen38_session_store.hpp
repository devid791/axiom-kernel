#ifndef AXIOM_QWEN38_SESSION_STORE_HPP
#define AXIOM_QWEN38_SESSION_STORE_HPP

/*
 * Crash-safe metadata store for the native Qwen3.8 paged KV tier.
 *
 * The GPU cache itself remains owned by qwen38_kv_tier. This component owns
 * the durable identity around it: model/checkpoint/config/profile namespace,
 * client session id, token history and the recurrent GDN snapshot. A session
 * generation names the durable tier selected by the manifest. New prompt
 * branches use a new generation file; appends keep the active tier and publish
 * the manifest only after the KV header has been committed. A tier/manifest
 * watermark mismatch is rejected during startup, so recovery fails closed
 * instead of restoring an unverified cache after a power loss.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "axiom/axiom.h"

namespace axiom {
namespace qwen38 {

struct qwen38_session_key {
    std::string session_id;
    std::string model_id;
    std::string target_identity;
    std::string dspark_identity;
    std::string config_signature;
    std::string profile;
};

struct qwen38_session_paths {
    std::string namespace_id;
    std::string namespace_dir;
    std::string session_stem;
    std::string manifest_path;
    std::string tier_path;
    uint64_t generation = 0u;
};

struct qwen38_session_manifest {
    uint64_t generation = 0u;
    uint32_t committed_tokens = 0u;
    uint32_t next_token = 0u;
    uint64_t token_hash = 0u;
    std::string session_id;
    std::string namespace_id;
    std::string namespace_text;
    std::string profile;
    std::vector<uint32_t> token_ids;
    std::vector<uint8_t> recurrent_state;
};

class qwen38_persistent_session_store {
public:
    qwen38_persistent_session_store() = default;
    qwen38_persistent_session_store(
            std::string base_path,
            uint32_t max_context,
            uint64_t recurrent_state_bytes = 0u);

    void configure(
            std::string base_path,
            uint32_t max_context,
            uint64_t recurrent_state_bytes = 0u);

    bool enabled() const { return !base_path_.empty(); }
    const std::string &root_path() const { return root_path_; }

    /* Validates the key, creates the namespace directory and returns the
     * generation-specific paths. Generation must be nonzero. */
    int prepare(
            const qwen38_session_key &key,
            uint64_t generation,
            qwen38_session_paths *out,
            std::string *error) const;

    /* A missing manifest is a normal new-session condition and is reported by
     * exists=false. Any malformed, truncated or checksum-invalid manifest is
     * an IO error and must not be silently rebuilt. */
    int load(
            const qwen38_session_key &key,
            qwen38_session_manifest *out,
            qwen38_session_paths *paths,
            bool *exists,
            std::string *error) const;

    /* Publish only after the corresponding KV tier header has committed. */
    int save(
            const qwen38_session_key &key,
            const qwen38_session_paths &paths,
            const qwen38_session_manifest &manifest,
            std::string *error) const;

    /* Remove only tier files from generations older than the manifest that
     * has already been published and fsynced. The current generation and any
     * future generation are never touched, making the operation idempotent
     * and safe to retry after a crash. */
    int prune_obsolete_generations(
            const qwen38_session_key &key,
            const qwen38_session_paths &committed_paths,
            uint64_t *removed_files,
            std::string *error) const;

    qwen38_session_paths paths_for(
            const qwen38_session_key &key,
            uint64_t generation) const;

    static bool valid_session_id(const std::string &session_id);
    static std::string generate_session_id();
    static std::string namespace_text(const qwen38_session_key &key);
    static std::string namespace_id(const qwen38_session_key &key);
    static uint64_t token_hash(const std::vector<uint32_t> &tokens);

private:
    std::string base_path_;
    std::string root_path_;
    uint32_t max_context_ = 0u;
    uint64_t recurrent_state_bytes_ = 0u;
};

}  // namespace qwen38
}  // namespace axiom

#endif  // AXIOM_QWEN38_SESSION_STORE_HPP
