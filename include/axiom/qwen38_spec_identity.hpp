#ifndef AXIOM_QWEN38_SPEC_IDENTITY_HPP
#define AXIOM_QWEN38_SPEC_IDENTITY_HPP

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace axiom::qwen38::spec_identity {

inline constexpr const char *kFingerprintPrefix = "specfp1:";
inline constexpr const char *kSidecarSchema =
        "axiom-speculative-contract-v1";
inline constexpr const char *kSidecarFilename =
        "axiom-speculative-contract-v1.json";

/*
 * Values which may change when the native target/draft execution ABI changes.
 * Textual execution semantics are deliberately fixed by specfp1 revision 1;
 * changing one requires a new fingerprint revision, not an untracked string.
 */
struct runtime_contract {
    std::uint32_t axiom_abi_version = 1u;
    std::uint32_t target_model_dspark_abi_version = 1u;
    std::uint32_t target_capture_abi_version = 2u;
    std::uint32_t speculative_target_abi_version = 2u;
    std::uint32_t dspark_compute_layout_version = 3u;
    std::uint32_t dspark_compute_device_abi_version = 3u;

    std::array<std::uint32_t, 5> target_tap_layer_ids{{4u, 16u, 28u, 40u, 52u}};
    std::uint32_t target_tap_count = 5u;
    std::uint32_t temporal_verify_width = 8u;
    std::uint32_t draft_block_size = 7u;

    std::uint32_t target_hidden_size = 5120u;
    std::uint32_t target_intermediate_size = 17408u;
    std::uint32_t target_layers = 64u;
    std::uint32_t target_attention_heads = 24u;
    std::uint32_t target_key_value_heads = 4u;
    std::uint32_t target_head_dim = 256u;
    std::uint32_t target_rope_dim = 64u;
    std::uint32_t target_linear_key_heads = 16u;
    std::uint32_t target_linear_key_head_dim = 128u;
    std::uint32_t target_linear_value_heads = 48u;
    std::uint32_t target_linear_value_head_dim = 128u;
    std::uint32_t target_linear_conv_kernel_dim = 4u;
    std::uint32_t target_vocab_size = 248320u;
    std::uint32_t target_native_context = 262144u;
    std::uint64_t target_rope_theta = 10000000u;
    std::uint32_t target_yarn_factor = 4u;
    std::uint32_t target_yarn_original_context = 262144u;
    std::uint32_t target_yarn_beta_fast = 32u;
    std::uint32_t target_yarn_beta_slow = 1u;

    std::uint32_t draft_hidden_size = 5120u;
    std::uint32_t draft_intermediate_size = 10240u;
    std::uint32_t draft_layers = 5u;
    std::uint32_t draft_attention_heads = 40u;
    std::uint32_t draft_key_value_heads = 8u;
    std::uint32_t draft_head_dim = 128u;
    std::uint32_t draft_vocab_size = 248320u;
    std::uint32_t draft_max_context = 262144u;
    std::uint64_t draft_rope_theta = 10000000u;
    std::uint32_t draft_yarn_factor = 32u;
    std::uint32_t draft_yarn_original_context = 8192u;
    std::uint32_t draft_yarn_beta_fast = 32u;
    std::uint32_t draft_yarn_beta_slow = 1u;
    std::uint32_t draft_mask_token_id = 248077u;
    std::uint32_t draft_markov_rank = 256u;
    std::uint32_t draft_confidence_features = 5376u;
};

struct component_hashes {
    std::string target_sha256;
    std::string tokenizer_sha256;
    std::string chat_template_sha256;
    std::string draft_sha256;
    std::string runtime_sha256;
};

struct artifact_evidence {
    std::string component;
    std::string logical_path;
    std::string source_kind;
    std::string sha256;
    std::uint64_t bytes = 0u;
};

struct result {
    std::string fingerprint;
    component_hashes components;
    std::vector<artifact_evidence> artifacts;
    /* Sorted canonical key/value records included in runtime_sha256. */
    std::vector<std::pair<std::string, std::string>> runtime_records;
};

struct qualification_evidence {
    std::string status;
    std::string evidence_sha256;
    std::string source_commit;
};

/*
 * Computes a relocation-independent identity. All required artifacts are
 * opened without following symbolic links and are verified not to change
 * while being hashed. No exception crosses this API boundary.
 */
bool compute(
        const std::string &target_root,
        const std::string &draft_root,
        const runtime_contract &runtime,
        result *out,
        std::string *error) noexcept;

inline bool compute(
        const std::string &target_root,
        const std::string &draft_root,
        result *out,
        std::string *error) noexcept {
    return compute(target_root, draft_root, runtime_contract{}, out, error);
}

/* Strict parser/validator for axiom-speculative-contract-v1.json. */
bool validate_sidecar(
        const std::string &sidecar_path,
        const result &expected,
        qualification_evidence *qualification,
        std::string *error) noexcept;

/* Produces the exact schema accepted by validate_sidecar. */
bool build_sidecar_json(
        const result &identity,
        const std::string &source_commit,
        const std::string &evidence_sha256,
        std::string *json,
        std::string *error) noexcept;

}  // namespace axiom::qwen38::spec_identity

#endif
