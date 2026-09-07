#ifndef AXIOM_QWEN4EXP_ADMISSION_HPP
#define AXIOM_QWEN4EXP_ADMISSION_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace axiom::qwen4exp {

/* Immutable checkpoint identity.  These values describe exactly the
 * RadixArk snapshot qualified by this implementation; a different revision
 * must be admitted as a new contract instead of silently reusing this one. */
inline constexpr char kRepository[] =
        "RadixArk/Qwen3.8-Flash-Next-NVFP4";
inline constexpr char kRevision[] =
        "7b719225242aacd3dbd3f9407468c2ee9a9d2594";
inline constexpr char kConfigSha256[] =
        "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624";
inline constexpr char kQuantConfigSha256[] =
        "7e69ef4b94302ae5b6f453b913621f698d5631a1d023d8b3e9e3b829721b98e8";
inline constexpr char kIndexSha256[] =
        "da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84";
inline constexpr char kTokenizerSha256[] =
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3";
inline constexpr char kWeightManifestSha256[] =
        "e0b8949fee66faf29b656b84c747e7017f1bf4efd1e985e102f57c21b13d9501";
inline constexpr bool kNativeRuntimeQualified = true;

/* Hugging Face index.metadata.total_size for this checkpoint is the physical
 * sum of the 206 Safetensors files, including their JSON headers.  Keep it
 * distinct from the sum of all descriptor data spans used for allocations. */
inline constexpr std::uint64_t kSafetensorsPhysicalBytes = 135195303851ull;
inline constexpr std::uint64_t kTensorPayloadBytes = 135156121594ull;
inline constexpr std::uint64_t kHuggingFaceUsedStorageBytes = 135242371978ull;
inline constexpr std::uint64_t kRepositoryFileBytes = 135253622894ull;
inline constexpr std::uint32_t kTensorCount = 296475u;
inline constexpr std::uint32_t kShardCount = 206u;

struct role_counts {
    std::uint64_t global = 0;
    std::uint64_t hyper_connection = 0;
    std::uint64_t gated_deltanet = 0;
    std::uint64_t qsa = 0;
    std::uint64_t routed_expert = 0;
    std::uint64_t moe_control = 0;
    std::uint64_t ple = 0;
    std::uint64_t vision = 0;
    std::uint64_t mtp = 0;
};

struct admission_report {
    std::string repository = kRepository;
    std::string revision = kRevision;
    std::string model_root;

    bool identity_valid = false;
    bool config_valid = false;
    bool quantization_valid = false;
    bool tensor_contract_valid = false;
    bool metadata_admitted = false;
    bool checkpoint_complete = false;
    bool payload_hashes_valid = false;
    bool checkpoint_admitted = false;

    /* Build/release property, independent of whether one inspected checkpoint
     * directory is complete. The checkpoint_* fields remain the authority for
     * payload readiness. */
    bool native_runtime_ready = kNativeRuntimeQualified;

    std::uint64_t tensor_payload_bytes = 0;
    std::uint64_t safetensors_physical_bytes = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t shard_count = 0;
    std::uint64_t present_shards = 0;
    std::uint64_t present_shard_bytes = 0;
    std::string weight_manifest_sha256;
    role_counts roles;

    std::map<std::string, std::string> metadata_sha256;
    std::vector<std::string> missing_shards;
    std::vector<std::string> incomplete_shards;
    std::vector<std::string> diagnostics;
};

/* Inspect the pinned checkpoint without touching the model files.  When
 * metadata_only is true, missing/in-progress weight shards are reported but
 * do not make the metadata contract itself fail. */
bool inspect_checkpoint(const std::string &model_root,
                        bool metadata_only,
                        admission_report *report,
                        std::string *error) noexcept;

std::string report_json(const admission_report &report);

}  // namespace axiom::qwen4exp

#endif
