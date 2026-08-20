/* Native resident loader for Qwen3.8-27B-DSpark. */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_dspark.h"

namespace {

constexpr char kDsparkFile[] = "model.safetensors";
constexpr uint64_t kUploadChunkBytes = 16ull * 1024ull * 1024ull;
constexpr uint32_t kTargetLayerIds[AXIOM_QWEN38_DSPARK_TARGET_FEATURES] = {
    4u, 16u, 28u, 40u, 52u,
};

struct tensor_spec {
    axiom_qwen38_dspark_tensor tensor = AXIOM_QWEN38_DSPARK_TENSOR_INVALID;
    uint32_t layer = AXIOM_QWEN38_DSPARK_GLOBAL_LAYER;
    char name[128]{};
    uint32_t rank = 0u;
    uint64_t shape[2]{};
};

bool checked_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > std::numeric_limits<uint64_t>::max() / a)) return false;
    *out = a * b;
    return true;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool is_global_tensor(axiom_qwen38_dspark_tensor tensor) {
    const uint32_t value = static_cast<uint32_t>(tensor);
    return value >= static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT) &&
           value <= static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W2_WEIGHT);
}

bool is_layer_tensor(axiom_qwen38_dspark_tensor tensor) {
    const uint32_t value = static_cast<uint32_t>(tensor);
    return value >= static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT) &&
           value <= static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_V_PROJ_WEIGHT);
}

int tensor_slot(
        uint32_t layer,
        axiom_qwen38_dspark_tensor tensor,
        uint32_t *out_slot) {
    if (!out_slot) return AXIOM_ERR_INVALID_ARGUMENT;
    if (is_global_tensor(tensor)) {
        if (layer != AXIOM_QWEN38_DSPARK_GLOBAL_LAYER) return AXIOM_ERR_INVALID_ARGUMENT;
        *out_slot = static_cast<uint32_t>(tensor) -
                    static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT);
        return AXIOM_OK;
    }
    if (!is_layer_tensor(tensor) || layer >= AXIOM_QWEN38_DSPARK_LAYERS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out_slot = AXIOM_QWEN38_DSPARK_GLOBAL_TENSOR_COUNT +
                layer * AXIOM_QWEN38_DSPARK_LAYER_TENSOR_COUNT +
                (static_cast<uint32_t>(tensor) -
                 static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT));
    return *out_slot < AXIOM_QWEN38_DSPARK_TENSOR_COUNT ? AXIOM_OK : AXIOM_ERR_RUNTIME;
}

int set_spec(
        tensor_spec *out,
        axiom_qwen38_dspark_tensor tensor,
        uint32_t layer,
        const char *name,
        uint32_t rank,
        uint64_t d0,
        uint64_t d1) {
    if (!out || !name || rank == 0u || rank > 2u || d0 == 0u || (rank == 2u && d1 == 0u)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const size_t length = std::strlen(name);
    if (length >= sizeof(out->name)) return AXIOM_ERR_RUNTIME;
    *out = tensor_spec{};
    out->tensor = tensor;
    out->layer = layer;
    std::memcpy(out->name, name, length + 1u);
    out->rank = rank;
    out->shape[0] = d0;
    out->shape[1] = d1;
    return AXIOM_OK;
}

int make_tensor_spec(
        uint32_t layer,
        axiom_qwen38_dspark_tensor tensor,
        tensor_spec *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    if (is_global_tensor(tensor)) {
        if (layer != AXIOM_QWEN38_DSPARK_GLOBAL_LAYER) return AXIOM_ERR_INVALID_ARGUMENT;
        switch (tensor) {
        case AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT:
            return set_spec(out, tensor, layer, "fc.weight", 2u,
                            AXIOM_QWEN38_DSPARK_HIDDEN, AXIOM_QWEN38_DSPARK_FUSION_INPUT);
        case AXIOM_QWEN38_DSPARK_TENSOR_HIDDEN_NORM_WEIGHT:
            return set_spec(out, tensor, layer, "hidden_norm.weight", 1u,
                            AXIOM_QWEN38_DSPARK_HIDDEN, 0u);
        case AXIOM_QWEN38_DSPARK_TENSOR_FINAL_NORM_WEIGHT:
            return set_spec(out, tensor, layer, "norm.weight", 1u,
                            AXIOM_QWEN38_DSPARK_HIDDEN, 0u);
        case AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_WEIGHT:
            return set_spec(out, tensor, layer, "confidence_head.proj.weight", 2u,
                            1u, AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES);
        case AXIOM_QWEN38_DSPARK_TENSOR_CONFIDENCE_PROJ_BIAS:
            return set_spec(out, tensor, layer, "confidence_head.proj.bias", 1u, 1u, 0u);
        case AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W1_WEIGHT:
            return set_spec(out, tensor, layer, "markov_head.markov_w1.weight", 2u,
                            AXIOM_QWEN38_DSPARK_VOCAB, AXIOM_QWEN38_DSPARK_MARKOV_RANK);
        case AXIOM_QWEN38_DSPARK_TENSOR_MARKOV_W2_WEIGHT:
            return set_spec(out, tensor, layer, "markov_head.markov_w2.weight", 2u,
                            AXIOM_QWEN38_DSPARK_VOCAB, AXIOM_QWEN38_DSPARK_MARKOV_RANK);
        default:
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    if (!is_layer_tensor(tensor) || layer >= AXIOM_QWEN38_DSPARK_LAYERS) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    const char *suffix = nullptr;
    uint32_t rank = 0u;
    uint64_t d0 = 0u;
    uint64_t d1 = 0u;
    switch (tensor) {
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT:
        suffix = "input_layernorm.weight";
        rank = 1u;
        d0 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_DOWN_PROJ_WEIGHT:
        suffix = "mlp.down_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_HIDDEN;
        d1 = AXIOM_QWEN38_DSPARK_INTERMEDIATE;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_GATE_PROJ_WEIGHT:
        suffix = "mlp.gate_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_INTERMEDIATE;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_MLP_UP_PROJ_WEIGHT:
        suffix = "mlp.up_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_INTERMEDIATE;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_POST_ATTENTION_LAYERNORM_WEIGHT:
        suffix = "post_attention_layernorm.weight";
        rank = 1u;
        d0 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_NORM_WEIGHT:
        suffix = "self_attn.k_norm.weight";
        rank = 1u;
        d0 = AXIOM_QWEN38_DSPARK_HEAD_DIM;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_K_PROJ_WEIGHT:
        suffix = "self_attn.k_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_KV_HEADS * AXIOM_QWEN38_DSPARK_HEAD_DIM;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_O_PROJ_WEIGHT:
        suffix = "self_attn.o_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_HIDDEN;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_NORM_WEIGHT:
        suffix = "self_attn.q_norm.weight";
        rank = 1u;
        d0 = AXIOM_QWEN38_DSPARK_HEAD_DIM;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_Q_PROJ_WEIGHT:
        suffix = "self_attn.q_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_HIDDEN;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    case AXIOM_QWEN38_DSPARK_TENSOR_LAYER_SELF_ATTN_V_PROJ_WEIGHT:
        suffix = "self_attn.v_proj.weight";
        rank = 2u;
        d0 = AXIOM_QWEN38_DSPARK_KV_HEADS * AXIOM_QWEN38_DSPARK_HEAD_DIM;
        d1 = AXIOM_QWEN38_DSPARK_HIDDEN;
        break;
    default:
        return AXIOM_ERR_INVALID_ARGUMENT;
    }

    char name[128]{};
    const int written = std::snprintf(name, sizeof(name), "layers.%u.%s", layer, suffix);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) return AXIOM_ERR_RUNTIME;
    return set_spec(out, tensor, layer, name, rank, d0, d1);
}

int expected_bytes(const tensor_spec &spec, uint64_t *out) {
    if (!out || spec.rank == 0u || spec.rank > 2u) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t elements = 1u;
    for (uint32_t dim = 0u; dim < spec.rank; ++dim) {
        if (spec.shape[dim] == 0u || !checked_mul(elements, spec.shape[dim], &elements)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    return checked_mul(elements, sizeof(uint16_t), out) ? AXIOM_OK : AXIOM_ERR_INVALID_ARGUMENT;
}

int validate_model_geometry(axiom_model *model, uint64_t expected_payload_bytes) {
    axiom_model_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_info_get(model, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.hidden_size != AXIOM_QWEN38_DSPARK_HIDDEN ||
        info.intermediate_size != AXIOM_QWEN38_DSPARK_INTERMEDIATE ||
        info.num_hidden_layers != AXIOM_QWEN38_DSPARK_LAYERS ||
        info.num_attention_heads != AXIOM_QWEN38_DSPARK_HEADS ||
        info.num_key_value_heads != AXIOM_QWEN38_DSPARK_KV_HEADS ||
        info.vocab_size != AXIOM_QWEN38_DSPARK_VOCAB ||
        info.max_context != 262144u ||
        info.safetensors_file_count != 1u ||
        info.safetensors_header_tensor_count != AXIOM_QWEN38_DSPARK_TENSOR_COUNT ||
        info.safetensors_header_data_bytes != expected_payload_bytes) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

int inspect_tensor(axiom_model *model, const tensor_spec &spec, axiom_tensor_info *out) {
    if (!model || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, spec.name, &info);
    if (rc != AXIOM_OK) return rc;
    if (std::strcmp(info.file, kDsparkFile) != 0 ||
        info.dtype != AXIOM_TENSOR_DTYPE_BF16 || info.rank != spec.rank) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t expected = 0u;
    rc = expected_bytes(spec, &expected);
    if (rc != AXIOM_OK || info.byte_count != expected) {
        return rc == AXIOM_OK ? AXIOM_ERR_INVALID_ARGUMENT : rc;
    }
    for (uint32_t dim = 0u; dim < spec.rank; ++dim) {
        if (info.shape[dim] != spec.shape[dim]) return AXIOM_ERR_INVALID_ARGUMENT;
    }
    *out = info;
    return AXIOM_OK;
}

int upload_tensor(
        axiom_model *model,
        axiom_runtime *runtime,
        const tensor_spec &spec,
        const axiom_tensor_info &info,
        std::vector<uint8_t> *scratch,
        axiom_device_buffer **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || !scratch || scratch->empty() || !out || info.byte_count == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_device_buffer *buffer = nullptr;
    int rc = axiom_device_buffer_create(runtime, &buffer, info.byte_count);
    if (rc != AXIOM_OK) return rc;
    for (uint64_t offset = 0u; offset < info.byte_count;) {
        const uint64_t remaining = info.byte_count - offset;
        const uint64_t count = remaining < scratch->size() ? remaining : scratch->size();
        rc = axiom_model_tensor_read_slice(model, spec.name, offset, scratch->data(), count);
        if (rc == AXIOM_OK) rc = axiom_device_buffer_upload(buffer, offset, scratch->data(), count);
        if (rc != AXIOM_OK) {
            axiom_device_buffer_destroy(buffer);
            return rc;
        }
        offset += count;
    }
    *out = buffer;
    return AXIOM_OK;
}

int fill_contract(axiom_qwen38_dspark_forward_contract *out) {
    if (!out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->scalar_dtype = AXIOM_TENSOR_DTYPE_F32;
    out->hidden_size = AXIOM_QWEN38_DSPARK_HIDDEN;
    out->target_feature_count = AXIOM_QWEN38_DSPARK_TARGET_FEATURES;
    std::memcpy(out->target_layer_ids, kTargetLayerIds, sizeof(kTargetLayerIds));
    out->draft_layers = AXIOM_QWEN38_DSPARK_LAYERS;
    out->attention_heads = AXIOM_QWEN38_DSPARK_HEADS;
    out->key_value_heads = AXIOM_QWEN38_DSPARK_KV_HEADS;
    out->head_dim = AXIOM_QWEN38_DSPARK_HEAD_DIM;
    out->block_size = AXIOM_QWEN38_DSPARK_BLOCK_SIZE;
    out->verify_width = AXIOM_QWEN38_DSPARK_VERIFY_WIDTH;
    out->mask_token_id = AXIOM_QWEN38_DSPARK_MASK_TOKEN_ID;
    out->markov_rank = AXIOM_QWEN38_DSPARK_MARKOV_RANK;
    out->confidence_features = AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES;
    out->uses_target_embedding = 1u;
    out->uses_target_lm_head = 1u;
    out->requires_draft_kv_injection = 1u;
    out->requires_draft_kv_transaction = 1u;
    out->attention_is_noncausal_over_draft_block = 1u;
    out->rmsnorm_zero_centered = 0u;
    out->max_context = 262144u;
    out->rms_norm_eps = 1.0e-6f;
    out->confidence_head_alpha = 1.0f;
    out->rope_theta = 10000000.0f;
    out->yarn_factor = 4.0f;
    out->yarn_beta_fast = 32.0f;
    out->yarn_beta_slow = 1.0f;
    out->yarn_original_context = 262144u;
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_dspark {
    int device = -1;
    uint64_t checkpoint_bytes = 0u;
    uint64_t device_bytes = 0u;
    axiom_device_buffer *buffers[AXIOM_QWEN38_DSPARK_TENSOR_COUNT]{};
};

extern "C" void axiom_qwen38_dspark_destroy(axiom_qwen38_dspark *dspark) {
    if (!dspark) return;
    for (axiom_device_buffer *buffer : dspark->buffers) axiom_device_buffer_destroy(buffer);
    delete dspark;
}

extern "C" int axiom_qwen38_dspark_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        axiom_qwen38_dspark **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || device < 0 || !out) return AXIOM_ERR_INVALID_ARGUMENT;

    uint32_t runtime_device = 0u;
    int rc = axiom_runtime_device_id(runtime, &runtime_device);
    if (rc != AXIOM_OK) return rc;
    if (runtime_device != static_cast<uint32_t>(device)) return AXIOM_ERR_INVALID_ARGUMENT;

    tensor_spec specs[AXIOM_QWEN38_DSPARK_TENSOR_COUNT]{};
    axiom_tensor_info infos[AXIOM_QWEN38_DSPARK_TENSOR_COUNT]{};
    uint64_t payload_bytes = 0u;
    uint32_t index = 0u;
    for (uint32_t ordinal = 0u; ordinal < AXIOM_QWEN38_DSPARK_GLOBAL_TENSOR_COUNT; ++ordinal) {
        const auto tensor = static_cast<axiom_qwen38_dspark_tensor>(
                static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_FC_WEIGHT) + ordinal);
        rc = make_tensor_spec(AXIOM_QWEN38_DSPARK_GLOBAL_LAYER, tensor, &specs[index]);
        if (rc != AXIOM_OK) return rc;
        uint64_t bytes = 0u;
        rc = expected_bytes(specs[index], &bytes);
        if (rc != AXIOM_OK || !checked_add(payload_bytes, bytes, &payload_bytes)) {
            return rc != AXIOM_OK ? rc : AXIOM_ERR_BUDGET;
        }
        ++index;
    }
    for (uint32_t layer = 0u; layer < AXIOM_QWEN38_DSPARK_LAYERS; ++layer) {
        for (uint32_t ordinal = 0u; ordinal < AXIOM_QWEN38_DSPARK_LAYER_TENSOR_COUNT; ++ordinal) {
            const auto tensor = static_cast<axiom_qwen38_dspark_tensor>(
                    static_cast<uint32_t>(AXIOM_QWEN38_DSPARK_TENSOR_LAYER_INPUT_LAYERNORM_WEIGHT) + ordinal);
            rc = make_tensor_spec(layer, tensor, &specs[index]);
            if (rc != AXIOM_OK) return rc;
            uint64_t bytes = 0u;
            rc = expected_bytes(specs[index], &bytes);
            if (rc != AXIOM_OK || !checked_add(payload_bytes, bytes, &payload_bytes)) {
                return rc != AXIOM_OK ? rc : AXIOM_ERR_BUDGET;
            }
            ++index;
        }
    }
    if (index != AXIOM_QWEN38_DSPARK_TENSOR_COUNT) return AXIOM_ERR_RUNTIME;

    rc = validate_model_geometry(model, payload_bytes);
    if (rc != AXIOM_OK) return rc;
    for (uint32_t slot = 0u; slot < AXIOM_QWEN38_DSPARK_TENSOR_COUNT; ++slot) {
        rc = inspect_tensor(model, specs[slot], &infos[slot]);
        if (rc != AXIOM_OK) return rc;
    }

    std::vector<uint8_t> scratch;
    try {
        scratch.resize(static_cast<size_t>(kUploadChunkBytes));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }
    axiom_qwen38_dspark *dspark = new (std::nothrow) axiom_qwen38_dspark();
    if (!dspark) return AXIOM_ERR_BUDGET;
    dspark->device = device;
    dspark->checkpoint_bytes = payload_bytes;

    for (uint32_t slot = 0u; slot < AXIOM_QWEN38_DSPARK_TENSOR_COUNT; ++slot) {
        rc = upload_tensor(model, runtime, specs[slot], infos[slot], &scratch, &dspark->buffers[slot]);
        if (rc != AXIOM_OK || !dspark->buffers[slot] ||
            !checked_add(dspark->device_bytes, infos[slot].byte_count, &dspark->device_bytes)) {
            axiom_qwen38_dspark_destroy(dspark);
            return rc != AXIOM_OK ? rc : AXIOM_ERR_BUDGET;
        }
    }
    *out = dspark;
    return AXIOM_OK;
}

extern "C" uint64_t axiom_qwen38_dspark_device_bytes(const axiom_qwen38_dspark *dspark) {
    return dspark ? dspark->device_bytes : 0u;
}

extern "C" int axiom_qwen38_dspark_info_get(
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_info *out) {
    if (!dspark || !out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->hidden_size = AXIOM_QWEN38_DSPARK_HIDDEN;
    out->intermediate_size = AXIOM_QWEN38_DSPARK_INTERMEDIATE;
    out->draft_layers = AXIOM_QWEN38_DSPARK_LAYERS;
    out->attention_heads = AXIOM_QWEN38_DSPARK_HEADS;
    out->key_value_heads = AXIOM_QWEN38_DSPARK_KV_HEADS;
    out->head_dim = AXIOM_QWEN38_DSPARK_HEAD_DIM;
    out->vocab_size = AXIOM_QWEN38_DSPARK_VOCAB;
    out->target_model_layers = AXIOM_QWEN38_DSPARK_TARGET_LAYERS;
    out->target_feature_count = AXIOM_QWEN38_DSPARK_TARGET_FEATURES;
    std::memcpy(out->target_layer_ids, kTargetLayerIds, sizeof(kTargetLayerIds));
    out->block_size = AXIOM_QWEN38_DSPARK_BLOCK_SIZE;
    out->verify_width = AXIOM_QWEN38_DSPARK_VERIFY_WIDTH;
    out->mask_token_id = AXIOM_QWEN38_DSPARK_MASK_TOKEN_ID;
    out->markov_rank = AXIOM_QWEN38_DSPARK_MARKOV_RANK;
    out->confidence_features = AXIOM_QWEN38_DSPARK_CONFIDENCE_FEATURES;
    out->tensor_count = AXIOM_QWEN38_DSPARK_TENSOR_COUNT;
    out->resident_dtype = AXIOM_TENSOR_DTYPE_BF16;
    out->checkpoint_bytes = dspark->checkpoint_bytes;
    out->device_bytes = dspark->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_tensor_view_get(
        const axiom_qwen38_dspark *dspark,
        uint32_t layer,
        axiom_qwen38_dspark_tensor tensor,
        axiom_qwen38_dspark_tensor_view *out) {
    if (!dspark || !out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    uint32_t slot = 0u;
    int rc = tensor_slot(layer, tensor, &slot);
    if (rc != AXIOM_OK || !dspark->buffers[slot]) return rc != AXIOM_OK ? rc : AXIOM_ERR_RUNTIME;
    tensor_spec spec{};
    rc = make_tensor_spec(layer, tensor, &spec);
    if (rc != AXIOM_OK) return rc;
    uint64_t bytes = 0u;
    rc = expected_bytes(spec, &bytes);
    if (rc != AXIOM_OK) return rc;

    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->tensor = tensor;
    out->layer = layer;
    out->dtype = AXIOM_TENSOR_DTYPE_BF16;
    out->rank = spec.rank;
    for (uint32_t dim = 0u; dim < spec.rank; ++dim) out->shape[dim] = spec.shape[dim];
    out->byte_count = bytes;
    std::memcpy(out->name, spec.name, std::strlen(spec.name) + 1u);
    out->device_buffer = dspark->buffers[slot];
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_forward_contract_get(
        const axiom_qwen38_dspark *dspark,
        axiom_qwen38_dspark_forward_contract *out) {
    if (!dspark) return AXIOM_ERR_INVALID_ARGUMENT;
    return fill_contract(out);
}

extern "C" int axiom_qwen38_dspark_fusion_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_fusion_request *request) {
    if (!dspark || !request || request->abi_version != AXIOM_ABI_VERSION ||
        !request->target_aux_hidden || !request->fused_hidden || request->batch == 0u ||
        request->stream != nullptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_kv_injection_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_kv_injection_request *request) {
    if (!dspark || !request || request->abi_version != AXIOM_ABI_VERSION ||
        !request->fused_hidden || !request->positions || request->batch == 0u ||
        request->stream != nullptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_draft_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_draft_request *request) {
    if (!dspark || !request || request->abi_version != AXIOM_ABI_VERSION ||
        !request->token_ids || !request->positions || !request->out_hidden || request->batch == 0u ||
        request->token_count == 0u || request->token_count > AXIOM_QWEN38_DSPARK_BLOCK_SIZE ||
        request->stream != nullptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_dspark_markov_request_validate(
        const axiom_qwen38_dspark *dspark,
        const axiom_qwen38_dspark_markov_request *request) {
    if (!dspark || !request || request->abi_version != AXIOM_ABI_VERSION ||
        !request->base_logits || !request->draft_hidden || !request->anchor_tokens ||
        !request->out_logits || !request->out_confidence || request->batch == 0u ||
        request->token_count == 0u || request->token_count > AXIOM_QWEN38_DSPARK_BLOCK_SIZE ||
        request->stream != nullptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}
