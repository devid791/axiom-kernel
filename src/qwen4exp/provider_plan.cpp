#include "axiom/qwen4exp/provider_plan.hpp"

#include "axiom/qwen4exp_admission.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>

namespace axiom::qwen4exp {
namespace {

constexpr std::uint64_t kExpectedExpertTensors = 294912u;
constexpr std::uint64_t kExpectedPlePagedTensors = 129u;
constexpr std::uint64_t kExpectedResidentTensors =
        static_cast<std::uint64_t>(kTensorCount) -
        kExpectedExpertTensors - kExpectedPlePagedTensors;
constexpr std::uint64_t kExpectedExpertTensorsPerLayer = 6144u;

bool starts_with(const std::string &value, const char *prefix) noexcept {
    const std::string head(prefix);
    return value.size() >= head.size() &&
           value.compare(0u, head.size(), head) == 0;
}

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t *out) noexcept {
    if (out == nullptr || right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *out = left + right;
    return true;
}

provider_plan_status reject(std::string *error,
                            const std::string &message) noexcept {
    if (error != nullptr) {
        try {
            *error = "qwen4_exp provider plan: " + message;
        } catch (...) {
        }
    }
    return provider_plan_status::invalid_checkpoint;
}

bool parse_main_layer(const std::string &name,
                      std::uint32_t *layer) noexcept {
    constexpr char prefix[] = "model.language_model.layers.";
    if (!starts_with(name, prefix) || layer == nullptr) return false;
    std::size_t cursor = sizeof(prefix) - 1u;
    if (cursor >= name.size() || name[cursor] < '0' || name[cursor] > '9') return false;
    std::uint32_t value = 0u;
    while (cursor < name.size() && name[cursor] >= '0' && name[cursor] <= '9') {
        value = value * 10u + static_cast<unsigned>(name[cursor] - '0');
        if (value >= kProviderLayers) return false;
        ++cursor;
    }
    if (cursor >= name.size() || name[cursor] != '.') return false;
    *layer = value;
    return true;
}

bool is_main_expert(const std::string &name, std::uint32_t *layer) noexcept {
    return parse_main_layer(name, layer) &&
           name.find(".mlp.experts.") != std::string::npos;
}

bool is_ple_payload(const std::string &name, std::uint32_t *layer) noexcept {
    if (!parse_main_layer(name, layer) || *layer != 1u) return false;
    constexpr char prefix[] =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";
    if (!starts_with(name, prefix)) return false;
    const std::string suffix = name.substr(sizeof(prefix) - 1u);
    return suffix == "weight_scale" ||
           (starts_with(suffix, "shard_") &&
            suffix.size() > 7u &&
            suffix.compare(suffix.size() - 7u, 7u, ".weight") == 0);
}

bool valid_expert_dtype(tensor_dtype dtype) noexcept {
    return dtype == tensor_dtype::u8 || dtype == tensor_dtype::f8_e4m3 ||
           dtype == tensor_dtype::f32;
}

bool valid_resident_dtype(tensor_dtype dtype) noexcept {
    return dtype == tensor_dtype::bf16 || dtype == tensor_dtype::f32 ||
           dtype == tensor_dtype::i64;
}

}  // namespace

provider_plan_status build_provider_plan(
        const checkpoint_catalog *catalog,
        const std::string &checkpoint_fingerprint,
        provider_plan *plan,
        std::string *error) noexcept {
    if (catalog == nullptr || plan == nullptr || checkpoint_fingerprint.empty()) {
        return provider_plan_status::invalid_argument;
    }
    try {
        provider_plan result{};
        result.checkpoint_root = catalog->model_root();
        result.checkpoint_fingerprint = checkpoint_fingerprint;
        for (std::uint32_t index = 0u; index < kProviderLayers; ++index) {
            provider_layer_plan &layer = result.layer[index];
            layer.index = index;
            layer.kind = index % 4u == 3u ? provider_layer_kind::qsa
                                          : provider_layer_kind::gated_deltanet;
            layer.has_ple = index == 1u;
            if (layer.kind == provider_layer_kind::qsa) ++result.qsa_layers;
            else ++result.gdn_layers;
            if (layer.has_ple) ++result.ple_layers;
        }

        for (std::size_t index = 0u; index < catalog->tensor_count(); ++index) {
            const tensor_span *span = catalog->at(index);
            if (span == nullptr || span->name.empty() || span->shard.empty() ||
                span->dtype == tensor_dtype::unknown || span->bytes == 0u) {
                return reject(error, "catalog contains an invalid tensor span");
            }
            if (!checked_add(result.payload_bytes, span->bytes,
                             &result.payload_bytes)) {
                return provider_plan_status::arithmetic_overflow;
            }
            ++result.tensor_count;
            std::uint32_t layer_index = 0u;
            if (is_main_expert(span->name, &layer_index)) {
                if (!valid_expert_dtype(span->dtype)) {
                    return reject(error, "routed expert has a non-ModelOpt dtype: " +
                                         span->name);
                }
                ++result.routed_expert_tensors;
                if (!checked_add(result.routed_expert_bytes, span->bytes,
                                 &result.routed_expert_bytes)) {
                    return provider_plan_status::arithmetic_overflow;
                }
                provider_layer_plan &layer = result.layer[layer_index];
                ++layer.routed_expert_tensors;
                if (!checked_add(layer.routed_expert_bytes, span->bytes,
                                 &layer.routed_expert_bytes)) {
                    return provider_plan_status::arithmetic_overflow;
                }
                continue;
            }
            if (is_ple_payload(span->name, &layer_index)) {
                const bool valid_dtype =
                        span->name.find("weight_scale") != std::string::npos
                                ? span->dtype == tensor_dtype::bf16
                                : span->dtype == tensor_dtype::f8_e4m3;
                if (!valid_dtype) {
                    return reject(error, "PLE payload has an invalid dtype: " + span->name);
                }
                ++result.ple_paged_tensors;
                if (!checked_add(result.ple_paged_bytes, span->bytes,
                                 &result.ple_paged_bytes)) {
                    return provider_plan_status::arithmetic_overflow;
                }
                provider_layer_plan &layer = result.layer[layer_index];
                ++layer.ple_paged_tensors;
                if (!checked_add(layer.ple_paged_bytes, span->bytes,
                                 &layer.ple_paged_bytes)) {
                    return provider_plan_status::arithmetic_overflow;
                }
                continue;
            }
            if (!valid_resident_dtype(span->dtype)) {
                return reject(error, "resident tensor has an unsupported dtype: " + span->name);
            }
            std::uint64_t *bytes = &result.text_resident_bytes;
            std::uint64_t *count = &result.text_resident_tensors;
            if (starts_with(span->name, "model.visual.")) {
                bytes = &result.vision_resident_bytes;
                count = &result.vision_resident_tensors;
            } else if (starts_with(span->name, "mtp.")) {
                bytes = &result.mtp_resident_bytes;
                count = &result.mtp_resident_tensors;
            }
            ++*count;
            if (!checked_add(*bytes, span->bytes, bytes)) {
                return provider_plan_status::arithmetic_overflow;
            }
            if (parse_main_layer(span->name, &layer_index)) {
                provider_layer_plan &layer = result.layer[layer_index];
                ++layer.resident_tensors;
                if (!checked_add(layer.resident_bytes, span->bytes,
                                 &layer.resident_bytes)) {
                    return provider_plan_status::arithmetic_overflow;
                }
            }
        }

        const std::uint64_t resident_tensors =
                result.text_resident_tensors + result.vision_resident_tensors +
                result.mtp_resident_tensors;
        if (result.tensor_count != kTensorCount ||
            result.payload_bytes != kTensorPayloadBytes ||
            result.routed_expert_tensors != kExpectedExpertTensors ||
            result.ple_paged_tensors != kExpectedPlePagedTensors ||
            resident_tensors != kExpectedResidentTensors ||
            result.gdn_layers != kProviderGdnLayers ||
            result.qsa_layers != kProviderQsaLayers || result.ple_layers != 1u) {
            return reject(
                    error,
                    "aggregate residency contract mismatch: tensors=" +
                    std::to_string(result.tensor_count) + " payload=" +
                    std::to_string(result.payload_bytes) + " experts=" +
                    std::to_string(result.routed_expert_tensors) + " ple=" +
                    std::to_string(result.ple_paged_tensors) + " resident=" +
                    std::to_string(resident_tensors) + " gdn=" +
                    std::to_string(result.gdn_layers) + " qsa=" +
                    std::to_string(result.qsa_layers) + " ple_layers=" +
                    std::to_string(result.ple_layers));
        }
        for (const provider_layer_plan &layer : result.layer) {
            if (layer.routed_expert_tensors != kExpectedExpertTensorsPerLayer ||
                (layer.has_ple ? layer.ple_paged_tensors != kExpectedPlePagedTensors
                               : layer.ple_paged_tensors != 0u)) {
                return reject(error, "per-layer residency contract mismatch at layer " +
                                     std::to_string(layer.index));
            }
        }
        *plan = std::move(result);
        if (error != nullptr) error->clear();
        return provider_plan_status::ok;
    } catch (const std::exception &exception) {
        return reject(error, std::string("exception: ") + exception.what());
    } catch (...) {
        return reject(error, "unknown exception");
    }
}

const char *provider_plan_status_string(provider_plan_status status) noexcept {
    switch (status) {
        case provider_plan_status::ok: return "ok";
        case provider_plan_status::invalid_argument: return "invalid_argument";
        case provider_plan_status::invalid_checkpoint: return "invalid_checkpoint";
        case provider_plan_status::arithmetic_overflow: return "arithmetic_overflow";
    }
    return "unknown";
}

}  // namespace axiom::qwen4exp
