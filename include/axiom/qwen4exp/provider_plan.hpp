#ifndef AXIOM_QWEN4EXP_PROVIDER_PLAN_HPP
#define AXIOM_QWEN4EXP_PROVIDER_PLAN_HPP

#include "axiom/qwen4exp/checkpoint.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace axiom::qwen4exp {

inline constexpr char kProviderId[] = "qwen4_exp";
inline constexpr char kProviderModelId[] = "qwen3.8-flash-next-nvfp4";
inline constexpr std::uint32_t kProviderLayers = 48u;
inline constexpr std::uint32_t kProviderQsaLayers = 12u;
inline constexpr std::uint32_t kProviderGdnLayers = 36u;
inline constexpr std::uint32_t kProviderContext = 262144u;

enum class provider_plan_status : std::uint8_t {
    ok = 0,
    invalid_argument,
    invalid_checkpoint,
    arithmetic_overflow,
};

enum class provider_layer_kind : std::uint8_t {
    gated_deltanet = 0,
    qsa = 1,
};

struct provider_layer_plan {
    std::uint32_t index = 0u;
    provider_layer_kind kind = provider_layer_kind::gated_deltanet;
    bool has_ple = false;
    std::uint64_t resident_bytes = 0u;
    std::uint64_t routed_expert_bytes = 0u;
    std::uint64_t ple_paged_bytes = 0u;
    std::uint64_t resident_tensors = 0u;
    std::uint64_t routed_expert_tensors = 0u;
    std::uint64_t ple_paged_tensors = 0u;
};

/* Immutable memory/residency contract derived from the admitted checkpoint.
 * It is an additive provider plan, not a mutation of the existing Axiom
 * registry.  Routed experts are pager-managed, the 51.2 GB PLE table remains
 * range-readable through NVMe/RAM, and every other tensor is an explicit
 * resident or optional resident set. */
struct provider_plan {
    std::string provider_id = kProviderId;
    std::string model_id = kProviderModelId;
    std::string checkpoint_root;
    std::string checkpoint_fingerprint;

    std::uint32_t max_context = kProviderContext;
    std::uint32_t layers = kProviderLayers;
    std::uint32_t gdn_layers = 0u;
    std::uint32_t qsa_layers = 0u;
    std::uint32_t ple_layers = 0u;

    std::uint64_t tensor_count = 0u;
    std::uint64_t payload_bytes = 0u;
    std::uint64_t text_resident_tensors = 0u;
    std::uint64_t text_resident_bytes = 0u;
    std::uint64_t vision_resident_tensors = 0u;
    std::uint64_t vision_resident_bytes = 0u;
    std::uint64_t mtp_resident_tensors = 0u;
    std::uint64_t mtp_resident_bytes = 0u;
    std::uint64_t routed_expert_tensors = 0u;
    std::uint64_t routed_expert_bytes = 0u;
    std::uint64_t ple_paged_tensors = 0u;
    std::uint64_t ple_paged_bytes = 0u;

    bool text_graph_required = true;
    bool vision_graph_optional = true;
    bool mtp_graph_optional = true;
    bool kv_cache_bf16 = true;
    bool kv_cache_quantized = false;
    bool yarn_enabled = false;
    bool existing_axiom_providers_preserved = true;

    std::array<provider_layer_plan, kProviderLayers> layer{};
};

[[nodiscard]] provider_plan_status build_provider_plan(
        const checkpoint_catalog *catalog,
        const std::string &checkpoint_fingerprint,
        provider_plan *plan,
        std::string *error) noexcept;

[[nodiscard]] const char *provider_plan_status_string(
        provider_plan_status status) noexcept;

}  // namespace axiom::qwen4exp

#endif
