#include "axiom/qwen4exp/provider_plan.hpp"
#include "axiom/qwen4exp_admission.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>

namespace q4 = axiom::qwen4exp;

namespace {

bool require(bool condition, const std::string &message) {
    if (!condition) std::fprintf(stderr, "qwen4exp-provider-plan-test: %s\n",
                                 message.c_str());
    return condition;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL_ROOT\n", argv[0]);
        return 2;
    }
    std::unique_ptr<q4::checkpoint_catalog> catalog;
    std::string error;
    if (!q4::checkpoint_catalog::open(argv[1], &catalog, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    q4::provider_plan plan{};
    const q4::provider_plan_status status = q4::build_provider_plan(
            catalog.get(), q4::kWeightManifestSha256, &plan, &error);
    bool ok = require(status == q4::provider_plan_status::ok,
                      "plan failed: " + error);
    ok = require(plan.provider_id == "qwen4_exp" &&
                 plan.model_id == "qwen3.8-flash-next-nvfp4",
                 "provider identity mismatch") && ok;
    ok = require(plan.layers == 48u && plan.gdn_layers == 36u &&
                 plan.qsa_layers == 12u && plan.ple_layers == 1u,
                 "layer graph mismatch") && ok;
    ok = require(plan.tensor_count == q4::kTensorCount &&
                 plan.payload_bytes == q4::kTensorPayloadBytes,
                 "payload totals mismatch") && ok;
    ok = require(plan.routed_expert_tensors == 294912u &&
                 plan.ple_paged_tensors == 129u,
                 "paged tensor totals mismatch") && ok;
    ok = require(plan.kv_cache_bf16 && !plan.kv_cache_quantized &&
                 !plan.yarn_enabled && plan.max_context == 262144u,
                 "runtime truth contract mismatch") && ok;
    ok = require(plan.existing_axiom_providers_preserved,
                 "additive provider guarantee missing") && ok;
    ok = require(plan.layer[1].has_ple &&
                 plan.layer[3].kind == q4::provider_layer_kind::qsa &&
                 plan.layer[2].kind == q4::provider_layer_kind::gated_deltanet,
                 "layer role mapping mismatch") && ok;
    if (!ok) return 1;

    const std::uint64_t resident_bytes =
            plan.text_resident_bytes + plan.vision_resident_bytes +
            plan.mtp_resident_bytes;
    const std::uint64_t resident_tensors =
            plan.text_resident_tensors + plan.vision_resident_tensors +
            plan.mtp_resident_tensors;
    std::printf(
            "qwen4exp-provider-plan-test: pass tensors=%llu payload=%llu "
            "resident=%llu/%llu text=%llu vision=%llu mtp=%llu "
            "experts=%llu/%llu ple=%llu/%llu\n",
            static_cast<unsigned long long>(plan.tensor_count),
            static_cast<unsigned long long>(plan.payload_bytes),
            static_cast<unsigned long long>(resident_tensors),
            static_cast<unsigned long long>(resident_bytes),
            static_cast<unsigned long long>(plan.text_resident_bytes),
            static_cast<unsigned long long>(plan.vision_resident_bytes),
            static_cast<unsigned long long>(plan.mtp_resident_bytes),
            static_cast<unsigned long long>(plan.routed_expert_tensors),
            static_cast<unsigned long long>(plan.routed_expert_bytes),
            static_cast<unsigned long long>(plan.ple_paged_tensors),
            static_cast<unsigned long long>(plan.ple_paged_bytes));
    return 0;
}
