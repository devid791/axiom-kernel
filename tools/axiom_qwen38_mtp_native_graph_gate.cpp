/*
 * Bounded full native-Qwen3.8 MTP CUDA-graph parity gate.
 *
 * The fixed 21-token prompt is the tokenizer-verified `greeting_it` case from
 * benchmarks/qwen38_speculative_real_corpus.json.  The gate deliberately runs
 * reference and device phases serially and releases every reference resource
 * before reloading the graph phase so it fits a single RTX 5090.
 *
 * Coverage is intentionally bounded:
 *   - four exact greedy reference/device parity cycles;
 *   - one resident-graph suffix prefill followed by a second-session replay,
 *     proving restored KV/pending-hidden handoff by matching reference cycle
 *     five without destroying or recapturing the CUDA graph;
 *   - one invalid-anchor replay proving async fail-closed rollback to the
 *     prompt watermark with no committed output.
 *
 * There is no fallback and no performance claim in this correctness gate.
 */

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

#include "axiom/axiom.h"
#include "axiom/qwen38_model.h"
#include "axiom/qwen38_mtp.h"
#include "axiom/qwen38_mtp_compute.h"
#include "axiom/qwen38_mtp_speculative.h"

namespace {

constexpr uint32_t kMaxContext = 64u;
constexpr uint32_t kPrimaryCycles = 4u;
constexpr uint32_t kMaterializationProbeCycles = 1u;
constexpr uint32_t kReferenceCycles =
        kPrimaryCycles + kMaterializationProbeCycles;
constexpr uint32_t kDeliberatelyWrongSeedOffset = 1u;
constexpr uint32_t kDraft = AXIOM_QWEN38_MTP_SPECULATIVE_DRAFT_TOKENS;
constexpr uint32_t kWidth = AXIOM_QWEN38_MTP_SPECULATIVE_VERIFY_WIDTH;
constexpr char kPromptCase[] = "greeting_it";
constexpr char kPromptText[] =
        "Ciao. Presentati in italiano in due frasi concise e dimmi come puoi "
        "aiutarmi oggi.";
constexpr std::array<uint32_t, 21u> kPromptTokens = {
        34u, 21817u, 13u, 25790u, 9031u, 303u, 57786u,
        303u, 4016u, 1374u, 9911u, 61446u, 378u, 4941u,
        7898u, 2445u, 175507u, 186986u, 203007u, 85593u, 13u,
};
constexpr uint32_t kSessionSuffixToken = 198u;
constexpr std::array<uint32_t, 2u> kStopTokens = {151643u, 151645u};
constexpr std::array<uint32_t,
        AXIOM_QWEN38_MODEL_DSPARK_TEMPORAL_VERIFY_WIDTH>
        kTemporalValidationTokens = {
                151643u, 198u, 271u, 646u, 151644u, 8948u, 13u, 151645u,
        };

static_assert(kDraft == 7u, "native MTP graph gate requires gamma seven");
static_assert(kWidth == 8u, "native MTP graph gate requires M8 verification");
static_assert(kPrimaryCycles == 4u, "the initial graph qualification is bounded");
static_assert(kPromptTokens.size() + 1u + kReferenceCycles * kWidth <= kMaxContext,
              "bounded gate history must fit its context");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_embed_f32_device),
                     axiom_qwen38_mtp_target_embed_f32_device_fn>::value,
        "target embedding callback ABI changed");
static_assert(
        std::is_same<decltype(&axiom_qwen38_model_dspark_target_lm_head_f32_device),
                     axiom_qwen38_mtp_target_lm_head_f32_device_fn>::value,
        "target LM-head callback ABI changed");

struct ReferenceCycle {
    uint32_t anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t snapshot_position = 0u;
    uint32_t position_after = 0u;
    uint32_t accepted_prefix = 0u;
    uint32_t commit_count = 0u;
    uint32_t continuation = AXIOM_TOKEN_ID_INVALID;
    std::array<uint32_t, kDraft> draft{};
    std::array<uint32_t, kWidth> verify{};
    std::array<uint32_t, kWidth> emitted{};
};

struct CommittedHistory {
    std::array<uint32_t, kMaxContext> token_ids{};
    uint32_t count = 0u;
};

struct ReferenceEvidence {
    uint32_t initial_anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t accepted_total = 0u;
    uint32_t emitted_total = 0u;
    uint32_t primary_final_position = 0u;
    uint32_t probe_final_position = 0u;
    std::array<ReferenceCycle, kReferenceCycles> cycles{};
    CommittedHistory primary_history{};
    CommittedHistory probe_history{};
};

struct DeviceCycleHistory {
    uint32_t draft[kDraft];
    uint32_t verify[kWidth];
    uint32_t emitted[kWidth];
    uint32_t accepted_prefix;
    uint32_t commit_count;
    uint32_t emitted_count;
    uint32_t continuation;
    uint32_t stop_detected;
    uint32_t matched_stop_token;
    uint32_t next_anchor;
    uint32_t next_position;
    uint32_t async_status;
    float continuation_logit;
};

static_assert(std::is_standard_layout<DeviceCycleHistory>::value,
              "device history must have a stable byte layout");

struct PhaseResources {
    int device = 0;
    axiom_qwen38_model *target = nullptr;
    axiom_runtime *runtime = nullptr;
    axiom_model *checkpoint = nullptr;
    axiom_qwen38_mtp *mtp = nullptr;
    axiom_qwen38_mtp_compute *compute = nullptr;
    axiom_qwen38_mtp_speculative *controller = nullptr;
    uint32_t temporal_m8_validated = 0u;
    uint32_t mtp_tensor_count = 0u;

    void release() {
        if (device >= 0) (void)cudaSetDevice(device);
        (void)cudaDeviceSynchronize();
        axiom_qwen38_mtp_speculative_destroy(controller);
        controller = nullptr;
        axiom_qwen38_mtp_compute_destroy(compute);
        compute = nullptr;
        axiom_qwen38_mtp_destroy(mtp);
        mtp = nullptr;
        if (checkpoint) axiom_model_close(checkpoint);
        checkpoint = nullptr;
        if (runtime) axiom_runtime_destroy(runtime);
        runtime = nullptr;
        axiom_qwen38_model_destroy(target);
        target = nullptr;
    }
};

struct DeviceBuffers {
    int device = 0;
    cudaStream_t stream = nullptr;
    uint32_t *seed_anchor = nullptr;
    uint32_t *seed_position = nullptr;
    DeviceCycleHistory *history_host = nullptr;

    void release() {
        if (device >= 0) (void)cudaSetDevice(device);
        if (stream) (void)cudaStreamSynchronize(stream);
        if (history_host) (void)cudaFreeHost(history_host);
        history_host = nullptr;
        (void)cudaFree(seed_position);
        seed_position = nullptr;
        (void)cudaFree(seed_anchor);
        seed_anchor = nullptr;
        if (stream) (void)cudaStreamDestroy(stream);
        stream = nullptr;
    }
};

int cuda_status(const cudaError_t status) {
    if (status == cudaSuccess) return AXIOM_OK;
    return status == cudaErrorMemoryAllocation ? AXIOM_ERR_BUDGET : AXIOM_ERR_CUDA;
}

bool parse_device(const char *text, int *out) {
    if (!text || !out || text[0] == '\0') return false;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    *out = static_cast<int>(value);
    return true;
}

bool append_history(
        CommittedHistory *history,
        const uint32_t *tokens,
        const uint32_t count) {
    if (!history || (count != 0u && !tokens) ||
        count > history->token_ids.size() - history->count) {
        return false;
    }
    for (uint32_t index = 0u; index < count; ++index) {
        if (tokens[index] >= AXIOM_QWEN38_MODEL_VOCAB) return false;
        history->token_ids[history->count++] = tokens[index];
    }
    return true;
}

int controller_info(
        const axiom_qwen38_mtp_speculative *controller,
        axiom_qwen38_mtp_speculative_info *out) {
    if (!controller || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
    out->struct_size = sizeof(*out);
    return axiom_qwen38_mtp_speculative_info_get(controller, out);
}

int compute_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_info *out) {
    if (!compute || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
    return axiom_qwen38_mtp_compute_info_get(compute, out);
}

int device_info(
        const axiom_qwen38_mtp_compute *compute,
        axiom_qwen38_mtp_compute_device_info_v1 *out) {
    if (!compute || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    *out = {};
    out->abi_version = AXIOM_QWEN38_MTP_COMPUTE_DEVICE_ABI_VERSION;
    out->struct_size = sizeof(*out);
    return axiom_qwen38_mtp_compute_device_info_get_v1(
            compute, out, sizeof(*out));
}

int verify_fixed_tokenization(const char *model_path, const char **stage) {
    if (!model_path || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "tokenizer_open";
    axiom_tokenizer *tokenizer = nullptr;
    axiom_tokenizer_config config{};
    config.abi_version = AXIOM_ABI_VERSION;
    config.path = model_path;
    config.name = "qwen3.8-mtp-native-graph-gate";
    config.format = AXIOM_TOKENIZER_FORMAT_HF_JSON;
    int rc = axiom_tokenizer_open(&tokenizer, &config);
    std::array<uint32_t, 32u> encoded{};
    uint32_t encoded_count = 0u;
    if (rc == AXIOM_OK) {
        *stage = "tokenizer_encode";
        rc = axiom_tokenizer_encode_text(
                tokenizer, kPromptText, encoded.data(),
                static_cast<uint32_t>(encoded.size()), &encoded_count);
    }
    if (rc == AXIOM_OK) {
        *stage = "tokenizer_exact_ids";
        if (encoded_count != kPromptTokens.size() ||
            !std::equal(kPromptTokens.begin(), kPromptTokens.end(), encoded.begin())) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (tokenizer) axiom_tokenizer_close(tokenizer);
    return rc;
}

int load_phase(
        const char *model_path,
        const int device,
        PhaseResources *phase,
        const char **stage) {
    if (!model_path || !phase || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    phase->device = device;
    *stage = "cuda_device";
    int rc = cudaSetDevice(device) == cudaSuccess ? AXIOM_OK : AXIOM_ERR_CUDA;
    if (rc == AXIOM_OK) {
        *stage = "target_create";
        rc = axiom_qwen38_model_create(
                model_path, device, kMaxContext, &phase->target);
    }
    axiom_qwen38_model_dspark_temporal_validation_result validation{};
    validation.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        *stage = "temporal_m8_validate";
        rc = axiom_qwen38_model_dspark_temporal8_validate(
                phase->target, kTemporalValidationTokens.data(), 0.0f,
                &validation);
        if (rc == AXIOM_OK && validation.passed != 1u) rc = AXIOM_ERR_RUNTIME;
        phase->temporal_m8_validated = validation.passed;
    }
    axiom_qwen38_model_dspark_temporal_capabilities capabilities{};
    capabilities.abi_version = AXIOM_QWEN38_MODEL_DSPARK_ABI_VERSION;
    if (rc == AXIOM_OK) {
        *stage = "temporal_capabilities";
        rc = axiom_qwen38_model_dspark_temporal_capabilities_get(
                phase->target, &capabilities);
        if (rc == AXIOM_OK &&
            (capabilities.temporal_m8_available != 1u ||
             capabilities.temporal_m8_implemented != 1u ||
             capabilities.temporal_m8_validated != 1u ||
             capabilities.temporal_verify_width != kWidth ||
             capabilities.draft_block_size != kDraft ||
             capabilities.requires_gdn_causal_rows != 1u ||
             capabilities.requires_attention_triangular_kv != 1u ||
             capabilities.requires_prefix_state_install != 1u)) {
            rc = AXIOM_ERR_NOT_IMPLEMENTED;
        }
    }
    if (rc == AXIOM_OK) {
        *stage = "runtime_create";
        axiom_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.backend = AXIOM_BACKEND_CUDA;
        config.device = static_cast<uint32_t>(device);
        rc = axiom_runtime_create(&phase->runtime, &config);
    }
    if (rc == AXIOM_OK) {
        *stage = "checkpoint_open";
        axiom_model_config config{};
        config.abi_version = AXIOM_ABI_VERSION;
        config.path = model_path;
        config.name = "qwen38-mtp-native-graph-gate";
        config.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
        config.placement.abi_version = AXIOM_ABI_VERSION;
        config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
        config.placement.device_id = static_cast<uint32_t>(device);
        rc = axiom_model_open(phase->runtime, &phase->checkpoint, &config);
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_load";
        rc = axiom_qwen38_mtp_load(
                phase->checkpoint, phase->runtime, device, &phase->mtp);
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_contract";
        axiom_qwen38_mtp_info info{};
        info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_qwen38_mtp_info_get(phase->mtp, &info);
        phase->mtp_tensor_count = info.tensor_count;
        if (rc == AXIOM_OK &&
            (info.tensor_count != AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT ||
             info.resident_dtype != AXIOM_TENSOR_DTYPE_BF16 ||
             info.device_bytes == 0u)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc == AXIOM_OK) {
        *stage = "mtp_compute_create";
        axiom_qwen38_mtp_target_binding binding{};
        binding.abi_version = AXIOM_QWEN38_MTP_COMPUTE_LAYOUT_VERSION;
        binding.user_data = phase->target;
        binding.embed_f32_device =
                axiom_qwen38_model_dspark_target_embed_f32_device;
        binding.lm_head_f32_device =
                axiom_qwen38_model_dspark_target_lm_head_f32_device;
        axiom_qwen38_mtp_compute_config_v2 config{};
        config.abi_version = AXIOM_QWEN38_MTP_COMPUTE_CONFIG_V2_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.max_context = kMaxContext;
        config.cache_dtype = AXIOM_TENSOR_DTYPE_BF16;
        config.rope_profile = axiom_qwen38_model_rope_profile(phase->target);
        rc = axiom_qwen38_mtp_compute_create_v2(
                phase->mtp, phase->runtime, device, &config, sizeof(config),
                &binding, &phase->compute);
    }
    if (rc == AXIOM_OK) {
        *stage = "controller_create";
        axiom_qwen38_mtp_speculative_config config{};
        config.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        config.struct_size = sizeof(config);
        config.device = static_cast<uint32_t>(device);
        rc = axiom_qwen38_mtp_speculative_create_v1(
                phase->target, phase->compute, &config, sizeof(config),
                &phase->controller);
    }
    return rc;
}

int prefill_prompt(
        PhaseResources *phase,
        uint32_t *out_anchor,
        const char **stage) {
    if (!phase || !phase->controller || !out_anchor || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_anchor = AXIOM_TOKEN_ID_INVALID;
    for (uint32_t index = 0u; index < kPromptTokens.size(); ++index) {
        *stage = "prompt_prefill";
        axiom_qwen38_mtp_speculative_prefill_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        int rc = axiom_qwen38_mtp_speculative_prefill_token(
                phase->controller, kPromptTokens[index], &result);
        if (rc != AXIOM_OK) return rc;
        if (result.token_position != index ||
            result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
            !std::isfinite(result.target_logit)) {
            *stage = "prompt_prefill_invariants";
            return AXIOM_ERR_RUNTIME;
        }
        *out_anchor = result.target_token_id;
    }
    *stage = "prompt_position_invariants";
    if (axiom_qwen38_model_position(phase->target) != kPromptTokens.size() ||
        axiom_qwen38_mtp_compute_position(phase->compute) !=
                kPromptTokens.size()) {
        return AXIOM_ERR_RUNTIME;
    }
    return AXIOM_OK;
}

int prefill_session_suffix(
        PhaseResources *phase,
        const uint32_t expected_position,
        uint32_t *out_anchor,
        const char **stage) {
    if (!phase || !phase->controller || !out_anchor || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *stage = "resident_graph_suffix_prefill";
    axiom_qwen38_mtp_speculative_prefill_result result{};
    result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
    result.struct_size = sizeof(result);
    int rc = axiom_qwen38_mtp_speculative_prefill_token(
            phase->controller, kSessionSuffixToken, &result);
    if (rc == AXIOM_OK &&
        (result.token_position != expected_position ||
         result.target_token_id >= AXIOM_QWEN38_MODEL_VOCAB ||
         !std::isfinite(result.target_logit) ||
         axiom_qwen38_model_position(phase->target) != expected_position + 1u ||
         axiom_qwen38_mtp_compute_position(phase->compute) !=
                 expected_position + 1u)) {
        *stage = "resident_graph_suffix_prefill_invariants";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) *out_anchor = result.target_token_id;
    return rc;
}

bool valid_reference_step(
        const axiom_qwen38_mtp_speculative_step_result &step,
        const uint32_t expected_snapshot) {
    const uint32_t accepted = step.accepted_draft_prefix;
    if (step.abi_version != AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION ||
        step.struct_size != sizeof(step) ||
        step.snapshot_position != expected_snapshot || accepted > kDraft ||
        step.emitted_token_count != accepted + 1u ||
        step.position_after_commit != expected_snapshot + accepted + 1u ||
        step.continuation_token_id != step.target_token_ids[accepted] ||
        step.full_block_accept != (accepted == kDraft ? 1u : 0u)) {
        return false;
    }
    for (uint32_t index = 0u; index < kDraft; ++index) {
        if (step.draft_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB) return false;
        if (index < accepted &&
            step.draft_token_ids[index] != step.target_token_ids[index]) {
            return false;
        }
        if (index == accepted && accepted < kDraft &&
            step.draft_token_ids[index] == step.target_token_ids[index]) {
            return false;
        }
    }
    for (uint32_t index = 0u; index < kWidth; ++index) {
        if (step.target_token_ids[index] >= AXIOM_QWEN38_MODEL_VOCAB ||
            !std::isfinite(step.target_logits[index])) {
            return false;
        }
    }
    for (uint32_t index = 0u; index < accepted; ++index) {
        if (step.emitted_token_ids[index] != step.draft_token_ids[index]) {
            return false;
        }
    }
    return step.emitted_token_ids[accepted] == step.continuation_token_id;
}

int run_reference_phase(
        PhaseResources *phase,
        ReferenceEvidence *evidence,
        const char **stage) {
    if (!phase || !evidence || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    evidence->probe_history = {};
    if (!append_history(
            &evidence->probe_history, kPromptTokens.data(),
            static_cast<uint32_t>(kPromptTokens.size()))) {
        *stage = "reference_history_init";
        return AXIOM_ERR_BUDGET;
    }
    uint32_t anchor = AXIOM_TOKEN_ID_INVALID;
    int rc = prefill_prompt(phase, &anchor, stage);
    evidence->initial_anchor = anchor;
    uint32_t expected_position = static_cast<uint32_t>(kPromptTokens.size());
    for (uint32_t cycle = 0u; cycle < kReferenceCycles && rc == AXIOM_OK;
         ++cycle) {
        *stage = "reference_cycle";
        axiom_qwen38_mtp_speculative_step_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_id = anchor;
        axiom_qwen38_mtp_speculative_step_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_step(
                phase->controller, &request, &result);
        if (rc == AXIOM_OK && !valid_reference_step(result, expected_position)) {
            *stage = "reference_cycle_invariants";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc != AXIOM_OK) break;

        ReferenceCycle &expected = evidence->cycles[cycle];
        expected.anchor = anchor;
        expected.snapshot_position = expected_position;
        expected.position_after = result.position_after_commit;
        expected.accepted_prefix = result.accepted_draft_prefix;
        expected.commit_count = result.emitted_token_count;
        expected.continuation = result.continuation_token_id;
        std::copy_n(result.draft_token_ids, kDraft, expected.draft.begin());
        expected.verify[0] = anchor;
        for (uint32_t index = 0u; index < kDraft; ++index) {
            expected.verify[index + 1u] = result.draft_token_ids[index];
        }
        std::copy_n(result.emitted_token_ids, kWidth, expected.emitted.begin());
        if (!append_history(
                &evidence->probe_history, result.emitted_token_ids,
                result.emitted_token_count)) {
            *stage = "reference_history_append";
            rc = AXIOM_ERR_BUDGET;
            break;
        }
        evidence->accepted_total += result.accepted_draft_prefix;
        evidence->emitted_total += result.emitted_token_count;
        expected_position = result.position_after_commit;
        anchor = result.continuation_token_id;
        if (cycle + 1u == kPrimaryCycles) {
            evidence->primary_history = evidence->probe_history;
            evidence->primary_final_position = expected_position;
            rc = prefill_session_suffix(
                    phase, expected_position, &anchor, stage);
            if (rc == AXIOM_OK && !append_history(
                    &evidence->probe_history, &kSessionSuffixToken, 1u)) {
                *stage = "reference_suffix_history_append";
                rc = AXIOM_ERR_BUDGET;
            }
            if (rc == AXIOM_OK) ++expected_position;
        }
    }
    evidence->probe_final_position = expected_position;
    if (rc == AXIOM_OK) {
        *stage = "reference_final_invariants";
        axiom_qwen38_mtp_speculative_info controller{};
        axiom_qwen38_mtp_compute_info compute{};
        rc = controller_info(phase->controller, &controller);
        if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
        if (rc == AXIOM_OK &&
            (evidence->accepted_total == 0u ||
             evidence->primary_history.count != evidence->primary_final_position ||
             evidence->probe_history.count != evidence->probe_final_position ||
             controller.poisoned != 0u || controller.position != expected_position ||
             controller.attempted_steps != kReferenceCycles ||
             controller.committed_steps != kReferenceCycles ||
             controller.proposed_tokens != kReferenceCycles * kDraft ||
             controller.accepted_tokens != evidence->accepted_total ||
             controller.emitted_tokens != evidence->emitted_total ||
             controller.failed_steps != 0u ||
             axiom_qwen38_model_position(phase->target) != expected_position ||
             axiom_qwen38_mtp_compute_position(phase->compute) !=
                     expected_position ||
             compute.committed_position != expected_position ||
             compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE)) {
            rc = AXIOM_ERR_RUNTIME;
        }
    }
    return rc;
}

int allocate_device_buffers(
        const int device,
        DeviceBuffers *buffers,
        const char **stage) {
    if (!buffers || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    buffers->device = device;
    *stage = "device_stream_create";
    int rc = cuda_status(cudaStreamCreateWithFlags(
            &buffers->stream, cudaStreamNonBlocking));
    if (rc == AXIOM_OK) {
        *stage = "device_seed_anchor_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&buffers->seed_anchor),
                sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_seed_position_allocate";
        rc = cuda_status(cudaMalloc(
                reinterpret_cast<void **>(&buffers->seed_position),
                sizeof(uint32_t)));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_pinned_history_allocate";
        rc = cuda_status(cudaHostAlloc(
                reinterpret_cast<void **>(&buffers->history_host),
                static_cast<size_t>(kPrimaryCycles) *
                        sizeof(DeviceCycleHistory), cudaHostAllocPortable));
    }
    return rc;
}

int append_device_to_host_copy(
        void *destination,
        const void *source,
        const size_t bytes,
        const cudaStream_t stream) {
    if (!destination || !source || bytes == 0u || !stream) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return cuda_status(cudaMemcpyAsync(
            destination, source, bytes, cudaMemcpyDeviceToHost, stream));
}

int snapshot_cycle(
        DeviceCycleHistory *history,
        const axiom_qwen38_mtp_speculative_device_step_result &result,
        const cudaStream_t stream) {
    if (!history) return AXIOM_ERR_INVALID_ARGUMENT;
    int rc = append_device_to_host_copy(
            history->draft,
            result.draft_token_ids_device,
            static_cast<size_t>(kDraft) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            history->verify,
            result.verify_token_ids_device,
            static_cast<size_t>(kWidth) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            history->emitted,
            result.emitted_token_ids_device,
            static_cast<size_t>(kWidth) * sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->accepted_prefix,
            result.accepted_prefix_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->commit_count,
            result.target_commit_count_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->emitted_count,
            result.emitted_token_count_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->continuation,
            result.continuation_token_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->stop_detected,
            result.stop_detected_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->matched_stop_token,
            result.matched_stop_token_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->next_anchor,
            result.next_anchor_token_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->next_position,
            result.next_anchor_position_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->async_status,
            result.async_status_device, sizeof(uint32_t), stream);
    if (rc == AXIOM_OK) rc = append_device_to_host_copy(
            &history->continuation_logit,
            result.continuation_logit_device, sizeof(float), stream);
    return rc;
}

int run_device_cycles(
        PhaseResources *phase,
        DeviceBuffers *buffers,
        const uint32_t seed_anchor,
        const uint32_t seed_position,
        const uint32_t cycles,
        DeviceCycleHistory *host_history,
        const char **stage) {
    if (!phase || !phase->controller || !buffers || !buffers->stream ||
        !buffers->seed_anchor || !buffers->seed_position ||
        !buffers->history_host ||
        cycles == 0u || cycles > kPrimaryCycles || !host_history || !stage) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    std::memset(
            buffers->history_host, 0,
            static_cast<size_t>(cycles) * sizeof(DeviceCycleHistory));
    *stage = "device_seed_anchor";
    int rc = cuda_status(cudaMemcpyAsync(
            buffers->seed_anchor, &seed_anchor, sizeof(seed_anchor),
            cudaMemcpyHostToDevice, buffers->stream));
    if (rc == AXIOM_OK) {
        *stage = "device_seed_position";
        rc = cuda_status(cudaMemcpyAsync(
                buffers->seed_position, &seed_position, sizeof(seed_position),
                cudaMemcpyHostToDevice, buffers->stream));
    }
    if (rc == AXIOM_OK) {
        *stage = "device_commit_limit";
        rc = axiom_qwen38_mtp_speculative_device_commit_limit_set(
                phase->controller, kWidth,
                reinterpret_cast<void *>(buffers->stream));
    }

    const uint32_t *anchor_device = buffers->seed_anchor;
    const uint32_t *position_device = buffers->seed_position;
    for (uint32_t cycle = 0u; cycle < cycles && rc == AXIOM_OK; ++cycle) {
        *stage = "device_graph_enqueue";
        axiom_qwen38_mtp_speculative_device_step_request request{};
        request.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
        request.struct_size = sizeof(request);
        request.anchor_token_device = anchor_device;
        request.anchor_position_device = position_device;
        request.stream = reinterpret_cast<void *>(buffers->stream);
        axiom_qwen38_mtp_speculative_device_step_result result{};
        result.abi_version = AXIOM_QWEN38_MTP_SPECULATIVE_DEVICE_ABI_VERSION;
        result.struct_size = sizeof(result);
        rc = axiom_qwen38_mtp_speculative_device_step_enqueue(
                phase->controller, &request, &result);
        if (rc == AXIOM_OK &&
            (result.graph_replayed != 1u || result.draft_tokens != kDraft ||
             result.verify_width != kWidth || !result.draft_token_ids_device ||
             !result.verify_token_ids_device ||
             !result.accepted_prefix_device ||
             !result.target_commit_count_device ||
             !result.emitted_token_count_device ||
             !result.continuation_token_device ||
             !result.continuation_logit_device ||
             !result.emitted_token_ids_device ||
             !result.stop_detected_device ||
             !result.matched_stop_token_device ||
             !result.next_anchor_token_device ||
             !result.next_anchor_position_device ||
             !result.async_status_device)) {
            *stage = "device_result_contract";
            rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) {
            *stage = "device_history_snapshot";
            rc = snapshot_cycle(
                    &buffers->history_host[cycle], result, buffers->stream);
        }
        anchor_device = result.next_anchor_token_device;
        position_device = result.next_anchor_position_device;
    }
    if (rc == AXIOM_OK) {
        *stage = "device_stream_sync";
        rc = cuda_status(cudaStreamSynchronize(buffers->stream));
    }
    if (rc == AXIOM_OK) {
        std::memcpy(
                host_history, buffers->history_host,
                static_cast<size_t>(cycles) * sizeof(DeviceCycleHistory));
    }
    return rc;
}

int verify_prepared_capture_state(
        PhaseResources *phase,
        const uint32_t prompt_position,
        const char **stage) {
    if (!phase || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "capture_preserves_host_watermark";
    axiom_qwen38_mtp_speculative_info controller{};
    axiom_qwen38_mtp_compute_info compute{};
    axiom_qwen38_mtp_compute_device_info_v1 device{};
    int rc = controller_info(phase->controller, &controller);
    if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
    if (rc == AXIOM_OK) rc = device_info(phase->compute, &device);
    if (rc == AXIOM_OK &&
        (controller.poisoned != 0u || controller.position != prompt_position ||
         axiom_qwen38_model_position(phase->target) != prompt_position ||
         axiom_qwen38_mtp_compute_position(phase->compute) != prompt_position ||
         compute.committed_position != prompt_position ||
         compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
         device.graph_capture_active != 0u ||
         device.device_session_active != 0u ||
         device.committed_position != prompt_position)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    return rc;
}

bool device_cycle_matches(
        const DeviceCycleHistory &actual,
        const ReferenceCycle &expected) {
    if (actual.async_status != static_cast<uint32_t>(AXIOM_OK) ||
        actual.accepted_prefix != expected.accepted_prefix ||
        actual.commit_count != expected.commit_count ||
        actual.emitted_count != actual.commit_count ||
        actual.commit_count != actual.accepted_prefix + 1u ||
        actual.stop_detected != 0u ||
        actual.matched_stop_token != AXIOM_TOKEN_ID_INVALID ||
        actual.continuation != expected.continuation ||
        actual.next_anchor != expected.continuation ||
        actual.next_position != expected.position_after ||
        !std::isfinite(actual.continuation_logit)) {
        return false;
    }
    for (uint32_t index = 0u; index < kDraft; ++index) {
        if (actual.draft[index] != expected.draft[index]) return false;
    }
    for (uint32_t index = 0u; index < kWidth; ++index) {
        if (actual.verify[index] != expected.verify[index]) return false;
    }
    for (uint32_t index = 0u; index < actual.commit_count; ++index) {
        if (actual.emitted[index] != expected.emitted[index]) return false;
    }
    return true;
}

int validate_cycles(
        const DeviceCycleHistory *actual,
        const ReferenceEvidence &reference,
        const uint32_t reference_offset,
        const uint32_t cycles,
        CommittedHistory *history,
        uint32_t *parity_tokens,
        const char **stage) {
    if (!actual || !history || !parity_tokens || !stage ||
        reference_offset + cycles > kReferenceCycles) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t cycle = 0u; cycle < cycles; ++cycle) {
        const ReferenceCycle &expected = reference.cycles[reference_offset + cycle];
        if (!device_cycle_matches(actual[cycle], expected)) {
            *stage = "device_reference_parity";
            return AXIOM_ERR_RUNTIME;
        }
        if (!append_history(
                history, actual[cycle].emitted, actual[cycle].commit_count)) {
            *stage = "device_history_append";
            return AXIOM_ERR_BUDGET;
        }
        *parity_tokens += actual[cycle].commit_count;
    }
    return AXIOM_OK;
}

int verify_active_device_session(
        PhaseResources *phase,
        const uint32_t host_position,
        const char **stage) {
    if (!phase || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "device_active_session_invariants";
    axiom_qwen38_mtp_speculative_info controller{};
    axiom_qwen38_mtp_compute_info compute{};
    axiom_qwen38_mtp_compute_device_info_v1 device{};
    int rc = controller_info(phase->controller, &controller);
    if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
    if (rc == AXIOM_OK) rc = device_info(phase->compute, &device);
    if (rc == AXIOM_OK &&
        (controller.poisoned != 0u || controller.position != host_position ||
         axiom_qwen38_model_position(phase->target) != host_position ||
         axiom_qwen38_mtp_compute_position(phase->compute) != host_position ||
         compute.committed_position != host_position ||
         compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
         device.device_session_active != 1u ||
         device.device_session_begin_position != host_position ||
         device.committed_position != host_position)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    return rc;
}

int verify_materialized_session(
        PhaseResources *phase,
        const uint32_t position,
        const uint32_t expected_poisoned,
        const char **stage) {
    if (!phase || !stage) return AXIOM_ERR_INVALID_ARGUMENT;
    *stage = "materialized_session_invariants";
    axiom_qwen38_mtp_speculative_info controller{};
    axiom_qwen38_mtp_compute_info compute{};
    axiom_qwen38_mtp_compute_device_info_v1 device{};
    int rc = controller_info(phase->controller, &controller);
    if (rc == AXIOM_OK) rc = compute_info(phase->compute, &compute);
    if (rc == AXIOM_OK) rc = device_info(phase->compute, &device);
    if (rc == AXIOM_OK &&
        (controller.poisoned != expected_poisoned ||
         controller.position != position ||
         axiom_qwen38_model_position(phase->target) != position ||
         axiom_qwen38_mtp_compute_position(phase->compute) != position ||
         compute.committed_position != position ||
         compute.transaction_state != AXIOM_QWEN38_MTP_TRANSACTION_IDLE ||
         device.device_session_active != 0u ||
         device.committed_position != position)) {
        rc = AXIOM_ERR_RUNTIME;
    }
    return rc;
}

bool rollback_record_valid(
        const DeviceCycleHistory &record,
        const uint32_t prompt_position,
        const uint32_t expected_status) {
    if (record.async_status != expected_status || record.accepted_prefix != 0u ||
        record.commit_count != 0u || record.emitted_count != 0u ||
        record.stop_detected != 0u ||
        record.matched_stop_token != AXIOM_TOKEN_ID_INVALID ||
        record.next_position != prompt_position ||
        record.continuation != AXIOM_TOKEN_ID_INVALID) {
        return false;
    }
    for (uint32_t index = 0u; index < kWidth; ++index) {
        if (record.emitted[index] != AXIOM_TOKEN_ID_INVALID) return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    const char *stage = "arguments";
    int rc = AXIOM_OK;
    int device = 0;
    if (argc < 2 || argc > 3 ||
        (argc == 3 && !parse_device(argv[2], &device))) {
        rc = AXIOM_ERR_INVALID_ARGUMENT;
    }
    if (rc == AXIOM_OK && setenv("AXIOM_MODEL_NO_RESIDENT", "1", 1) != 0) {
        stage = "disable_duplicate_generic_residency";
        rc = AXIOM_ERR_RUNTIME;
    }

    bool tokenizer_verified = false;
    if (rc == AXIOM_OK) {
        rc = verify_fixed_tokenization(argv[1], &stage);
        tokenizer_verified = rc == AXIOM_OK;
    }

    ReferenceEvidence reference{};
    PhaseResources reference_phase{};
    if (rc == AXIOM_OK) rc = load_phase(
            argv[1], device, &reference_phase, &stage);
    if (rc == AXIOM_OK) {
        stage = "reference_phase";
        rc = run_reference_phase(&reference_phase, &reference, &stage);
    }
    const uint32_t reference_temporal_validated =
            reference_phase.temporal_m8_validated;
    const uint32_t reference_mtp_tensors = reference_phase.mtp_tensor_count;
    reference_phase.release();
    if (rc == AXIOM_OK) {
        stage = "reference_release_sync";
        rc = cuda_status(cudaDeviceSynchronize());
    }

    PhaseResources graph_phase{};
    DeviceBuffers buffers{};
    uint32_t graph_temporal_validated = 0u;
    uint32_t graph_mtp_tensors = 0u;
    uint32_t graph_initial_anchor = AXIOM_TOKEN_ID_INVALID;
    uint32_t parity_tokens = 0u;
    uint32_t rollback_status = static_cast<uint32_t>(AXIOM_OK);
    int rollback_session_end_rc = AXIOM_OK;
    bool primary_parity = false;
    bool capture_preserved_host_watermark = false;
    bool trusted_host_watermark = false;
    bool pending_roundtrip = false;
    bool resident_graph_suffix_prefill = false;
    bool rollback_verified = false;
    std::array<DeviceCycleHistory, kPrimaryCycles> primary_history{};
    DeviceCycleHistory probe_history{};
    DeviceCycleHistory rollback_history{};
    CommittedHistory graph_history{};

    if (rc == AXIOM_OK) rc = load_phase(argv[1], device, &graph_phase, &stage);
    graph_temporal_validated = graph_phase.temporal_m8_validated;
    graph_mtp_tensors = graph_phase.mtp_tensor_count;
    if (rc == AXIOM_OK) rc = prefill_prompt(
            &graph_phase, &graph_initial_anchor, &stage);
    if (rc == AXIOM_OK && graph_initial_anchor != reference.initial_anchor) {
        stage = "prefill_anchor_parity";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        stage = "device_stop_contract";
        rc = axiom_qwen38_mtp_speculative_device_stop_tokens_set(
                graph_phase.controller, kStopTokens.data(),
                static_cast<uint32_t>(kStopTokens.size()));
    }
    if (rc == AXIOM_OK) {
        stage = "device_graph_prepare";
        rc = axiom_qwen38_mtp_speculative_device_prepare(
                graph_phase.controller);
    }
    if (rc == AXIOM_OK) rc = verify_prepared_capture_state(
            &graph_phase, static_cast<uint32_t>(kPromptTokens.size()), &stage);
    capture_preserved_host_watermark = rc == AXIOM_OK;
    if (rc == AXIOM_OK) rc = allocate_device_buffers(
            device, &buffers, &stage);
    if (rc == AXIOM_OK) {
        graph_history = {};
        if (!append_history(
                &graph_history, kPromptTokens.data(),
                static_cast<uint32_t>(kPromptTokens.size()))) {
            stage = "graph_history_init";
            rc = AXIOM_ERR_BUDGET;
        }
    }
    if (rc == AXIOM_OK) rc = run_device_cycles(
            &graph_phase, &buffers, graph_initial_anchor,
            static_cast<uint32_t>(kPromptTokens.size()) +
                    kDeliberatelyWrongSeedOffset,
            kPrimaryCycles,
            primary_history.data(), &stage);
    if (rc == AXIOM_OK) {
        stage = "trusted_host_watermark_overrides_caller_seed";
        trusted_host_watermark =
                primary_history[0].next_position ==
                        reference.cycles[0].position_after &&
                primary_history[0].next_position !=
                        static_cast<uint32_t>(kPromptTokens.size()) +
                        kDeliberatelyWrongSeedOffset +
                        primary_history[0].commit_count;
        if (!trusted_host_watermark) rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) rc = validate_cycles(
            primary_history.data(), reference, 0u, kPrimaryCycles,
            &graph_history, &parity_tokens, &stage);
    primary_parity = rc == AXIOM_OK;
    if (rc == AXIOM_OK &&
        (graph_history.count != reference.primary_history.count ||
         !std::equal(
                 graph_history.token_ids.begin(),
                 graph_history.token_ids.begin() + graph_history.count,
                 reference.primary_history.token_ids.begin()))) {
        stage = "primary_output_sequence_parity";
        rc = AXIOM_ERR_RUNTIME;
        primary_parity = false;
    }
    if (rc == AXIOM_OK) rc = verify_active_device_session(
            &graph_phase, static_cast<uint32_t>(kPromptTokens.size()), &stage);
    if (rc == AXIOM_OK) {
        stage = "primary_session_materialize";
        rc = axiom_qwen38_mtp_speculative_device_session_end(
                graph_phase.controller, reinterpret_cast<void *>(buffers.stream),
                graph_history.token_ids.data(), graph_history.count);
    }
    if (rc == AXIOM_OK) rc = verify_materialized_session(
            &graph_phase, reference.primary_final_position, 0u, &stage);

    uint32_t graph_suffix_anchor = AXIOM_TOKEN_ID_INVALID;
    if (rc == AXIOM_OK) rc = prefill_session_suffix(
            &graph_phase, reference.primary_final_position,
            &graph_suffix_anchor, &stage);
    if (rc == AXIOM_OK &&
        graph_suffix_anchor != reference.cycles[kPrimaryCycles].anchor) {
        stage = "resident_graph_suffix_anchor_parity";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK && !append_history(
            &graph_history, &kSessionSuffixToken, 1u)) {
        stage = "graph_suffix_history_append";
        rc = AXIOM_ERR_BUDGET;
    }
    resident_graph_suffix_prefill = rc == AXIOM_OK;

    /* The suffix prefill above runs while both graph executables remain
     * resident. Starting a fresh device session then copies the updated host
     * pending-hidden row back to its graph-owned row. Exact parity with
     * reference cycle five proves the complete restore->suffix->graph handoff. */
    if (rc == AXIOM_OK) rc = run_device_cycles(
            &graph_phase, &buffers, graph_suffix_anchor,
            reference.primary_final_position + 1u,
            kMaterializationProbeCycles,
            &probe_history, &stage);
    if (rc == AXIOM_OK) rc = validate_cycles(
            &probe_history, reference, kPrimaryCycles,
            kMaterializationProbeCycles, &graph_history, &parity_tokens, &stage);
    pending_roundtrip = rc == AXIOM_OK;
    if (rc == AXIOM_OK &&
        (graph_history.count != reference.probe_history.count ||
         !std::equal(
                 graph_history.token_ids.begin(),
                 graph_history.token_ids.begin() + graph_history.count,
                 reference.probe_history.token_ids.begin()))) {
        stage = "probe_output_sequence_parity";
        rc = AXIOM_ERR_RUNTIME;
        pending_roundtrip = false;
    }
    if (rc == AXIOM_OK) rc = verify_active_device_session(
            &graph_phase, reference.primary_final_position + 1u, &stage);
    if (rc == AXIOM_OK) {
        stage = "probe_session_materialize";
        rc = axiom_qwen38_mtp_speculative_device_session_end(
                graph_phase.controller, reinterpret_cast<void *>(buffers.stream),
                graph_history.token_ids.data(), graph_history.count);
    }
    if (rc == AXIOM_OK) rc = verify_materialized_session(
            &graph_phase, reference.probe_final_position, 0u, &stage);

    /* Reuse the loaded checkpoint for one bounded rollback arm.  Reset is a
     * cold operation and destroys the previous graph before prompt replay. */
    if (rc == AXIOM_OK) {
        stage = "rollback_controller_reset";
        rc = axiom_qwen38_mtp_speculative_reset(graph_phase.controller);
    }
    uint32_t rollback_prompt_anchor = AXIOM_TOKEN_ID_INVALID;
    if (rc == AXIOM_OK) rc = prefill_prompt(
            &graph_phase, &rollback_prompt_anchor, &stage);
    if (rc == AXIOM_OK && rollback_prompt_anchor != reference.initial_anchor) {
        stage = "rollback_prefill_anchor_parity";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        stage = "rollback_graph_prepare";
        rc = axiom_qwen38_mtp_speculative_device_prepare(
                graph_phase.controller);
    }
    if (rc == AXIOM_OK) rc = verify_prepared_capture_state(
            &graph_phase, static_cast<uint32_t>(kPromptTokens.size()), &stage);
    const uint32_t invalid_anchor = AXIOM_QWEN38_MODEL_VOCAB;
    if (rc == AXIOM_OK) rc = run_device_cycles(
            &graph_phase, &buffers, invalid_anchor,
            static_cast<uint32_t>(kPromptTokens.size()), 1u,
            &rollback_history, &stage);
    rollback_status = rollback_history.async_status;
    if (rc == AXIOM_OK && !rollback_record_valid(
            rollback_history, static_cast<uint32_t>(kPromptTokens.size()),
            static_cast<uint32_t>(AXIOM_ERR_INVALID_ARGUMENT))) {
        stage = "rollback_device_evidence";
        rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) rc = verify_active_device_session(
            &graph_phase, static_cast<uint32_t>(kPromptTokens.size()), &stage);
    if (rc == AXIOM_OK) {
        stage = "rollback_session_end";
        rollback_session_end_rc =
                axiom_qwen38_mtp_speculative_device_session_end(
                        graph_phase.controller,
                        reinterpret_cast<void *>(buffers.stream),
                        kPromptTokens.data(),
                        static_cast<uint32_t>(kPromptTokens.size()));
        if (rollback_session_end_rc != AXIOM_ERR_INVALID_ARGUMENT) {
            rc = rollback_session_end_rc == AXIOM_OK
                    ? AXIOM_ERR_RUNTIME
                    : rollback_session_end_rc;
        }
    }
    if (rc == AXIOM_OK) rc = verify_materialized_session(
            &graph_phase, static_cast<uint32_t>(kPromptTokens.size()), 1u,
            &stage);
    rollback_verified = rc == AXIOM_OK;
    if (rc == AXIOM_OK) stage = "complete";

    const uint32_t graph_target_position = graph_phase.target
            ? axiom_qwen38_model_position(graph_phase.target) : 0u;
    const uint32_t graph_mtp_position = graph_phase.compute
            ? axiom_qwen38_mtp_compute_position(graph_phase.compute) : 0u;
    axiom_qwen38_mtp_speculative_info final_controller{};
    const bool final_controller_valid = graph_phase.controller &&
            controller_info(graph_phase.controller, &final_controller) == AXIOM_OK;

    buffers.release();
    graph_phase.release();

    const bool passed = rc == AXIOM_OK && tokenizer_verified && primary_parity &&
            capture_preserved_host_watermark && trusted_host_watermark &&
            resident_graph_suffix_prefill && pending_roundtrip &&
            rollback_verified;
    std::printf(
            "{\"status\":\"%s\",\"stage\":\"%s\",\"rc\":%d,"
            "\"device\":%d,\"prompt_case\":\"%s\","
            "\"prompt_tokens\":%zu,\"tokenizer_verified\":%u,"
            "\"reference_resources_released_before_graph\":1,"
            "\"reference_cycles\":%u,\"graph_cycles\":%u,"
            "\"materialization_probe_cycles\":%u,"
            "\"capture_preserved_host_watermark\":%u,"
            "\"caller_seed_position_was_deliberately_wrong\":1,"
            "\"trusted_host_watermark\":%u,"
            "\"per_replay_d2h_before_next_replay\":1,"
            "\"reference_accepted\":%u,\"reference_emitted\":%u,"
            "\"parity_tokens\":%u,\"token_id_parity_100pct\":%u,"
            "\"pending_hidden_roundtrip\":%u,"
            "\"resident_graph_suffix_prefill\":%u,"
            "\"session_materialization\":%u,"
            "\"rollback_verified\":%u,\"rollback_async_status\":%u,"
            "\"rollback_session_end_rc\":%d,"
            "\"reference_temporal_m8_validated\":%u,"
            "\"graph_temporal_m8_validated\":%u,"
            "\"reference_mtp_tensor_count\":%u,"
            "\"graph_mtp_tensor_count\":%u,"
            "\"primary_final_position\":%u,"
            "\"probe_final_position\":%u,"
            "\"final_target_position\":%u,"
            "\"final_mtp_position\":%u,\"final_poisoned\":",
            passed ? "pass" : "fail", stage, rc, device, kPromptCase,
            kPromptTokens.size(), tokenizer_verified ? 1u : 0u,
            kReferenceCycles, kPrimaryCycles, kMaterializationProbeCycles,
            capture_preserved_host_watermark ? 1u : 0u,
            trusted_host_watermark ? 1u : 0u,
            reference.accepted_total, reference.emitted_total, parity_tokens,
            primary_parity && pending_roundtrip ? 1u : 0u,
            pending_roundtrip ? 1u : 0u,
            resident_graph_suffix_prefill ? 1u : 0u,
            pending_roundtrip ? 1u : 0u,
            rollback_verified ? 1u : 0u, rollback_status,
            rollback_session_end_rc, reference_temporal_validated,
            graph_temporal_validated, reference_mtp_tensors,
            graph_mtp_tensors, reference.primary_final_position,
            reference.probe_final_position, graph_target_position,
            graph_mtp_position);
    if (final_controller_valid) std::printf("%u", final_controller.poisoned);
    else std::fputs("null", stdout);
    std::fputs(",\"no_fallback\":1,\"performance_claim\":false}\n", stdout);
    return passed ? 0 : 1;
}
