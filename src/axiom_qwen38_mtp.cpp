/* Native resident loader for the Qwen3.8 BF16 MTP sidecar. */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "axiom/axiom.h"
#include "axiom/qwen38_mtp.h"

namespace {

constexpr char kMtpFile[] = "model_mtp.safetensors";
constexpr uint64_t kUploadChunkBytes = 16ull * 1024ull * 1024ull;

struct mtp_tensor_spec {
    axiom_qwen38_mtp_tensor tensor;
    const char *name;
    uint32_t rank;
    uint64_t shape[2];
};

constexpr mtp_tensor_spec kTensorSpecs[AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT] = {
    {AXIOM_QWEN38_MTP_TENSOR_FC_WEIGHT, "mtp.fc.weight", 2u,
     {AXIOM_QWEN38_MTP_HIDDEN, AXIOM_QWEN38_MTP_FC_INPUT}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_INPUT_LAYERNORM_WEIGHT,
     "mtp.layers.0.input_layernorm.weight", 1u, {AXIOM_QWEN38_MTP_HIDDEN, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_DOWN_PROJ_WEIGHT,
     "mtp.layers.0.mlp.down_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_HIDDEN, AXIOM_QWEN38_MTP_INTERMEDIATE}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_GATE_PROJ_WEIGHT,
     "mtp.layers.0.mlp.gate_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_INTERMEDIATE, AXIOM_QWEN38_MTP_HIDDEN}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_MLP_UP_PROJ_WEIGHT,
     "mtp.layers.0.mlp.up_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_INTERMEDIATE, AXIOM_QWEN38_MTP_HIDDEN}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_POST_ATTENTION_LAYERNORM_WEIGHT,
     "mtp.layers.0.post_attention_layernorm.weight", 1u,
     {AXIOM_QWEN38_MTP_HIDDEN, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_NORM_WEIGHT,
     "mtp.layers.0.self_attn.k_norm.weight", 1u, {AXIOM_QWEN38_MTP_HEAD_DIM, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_K_PROJ_WEIGHT,
     "mtp.layers.0.self_attn.k_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_KV_HEADS * AXIOM_QWEN38_MTP_HEAD_DIM, AXIOM_QWEN38_MTP_HIDDEN}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_O_PROJ_WEIGHT,
     "mtp.layers.0.self_attn.o_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_HIDDEN, AXIOM_QWEN38_MTP_ATTENTION_HEADS * AXIOM_QWEN38_MTP_HEAD_DIM}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_NORM_WEIGHT,
     "mtp.layers.0.self_attn.q_norm.weight", 1u, {AXIOM_QWEN38_MTP_HEAD_DIM, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_Q_PROJ_WEIGHT,
     "mtp.layers.0.self_attn.q_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_ATTENTION_HEADS * AXIOM_QWEN38_MTP_HEAD_DIM * 2u,
      AXIOM_QWEN38_MTP_HIDDEN}},
    {AXIOM_QWEN38_MTP_TENSOR_LAYER0_SELF_ATTN_V_PROJ_WEIGHT,
     "mtp.layers.0.self_attn.v_proj.weight", 2u,
     {AXIOM_QWEN38_MTP_KV_HEADS * AXIOM_QWEN38_MTP_HEAD_DIM, AXIOM_QWEN38_MTP_HIDDEN}},
    {AXIOM_QWEN38_MTP_TENSOR_NORM_WEIGHT, "mtp.norm.weight", 1u,
     {AXIOM_QWEN38_MTP_HIDDEN, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_EMBEDDING_WEIGHT,
     "mtp.pre_fc_norm_embedding.weight", 1u, {AXIOM_QWEN38_MTP_HIDDEN, 0u}},
    {AXIOM_QWEN38_MTP_TENSOR_PRE_FC_NORM_HIDDEN_WEIGHT,
     "mtp.pre_fc_norm_hidden.weight", 1u, {AXIOM_QWEN38_MTP_HIDDEN, 0u}},
};

static_assert(sizeof(kTensorSpecs) / sizeof(kTensorSpecs[0]) ==
              AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT,
              "Qwen3.8 MTP tensor inventory changed");

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

bool tensor_index_valid(axiom_qwen38_mtp_tensor tensor) {
    return tensor >= AXIOM_QWEN38_MTP_TENSOR_FC_WEIGHT &&
           static_cast<uint32_t>(tensor) < AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT;
}

int expected_bytes(const mtp_tensor_spec &spec, uint64_t *out) {
    if (!out) return AXIOM_ERR_INVALID_ARGUMENT;
    uint64_t elements = 1u;
    for (uint32_t dim = 0u; dim < spec.rank; ++dim) {
        if (spec.shape[dim] == 0u || !checked_mul(elements, spec.shape[dim], &elements)) {
            return AXIOM_ERR_INVALID_ARGUMENT;
        }
    }
    if (!checked_mul(elements, sizeof(uint16_t), out)) return AXIOM_ERR_INVALID_ARGUMENT;
    return AXIOM_OK;
}

int validate_base_geometry(axiom_model *model) {
    axiom_model_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_info_get(model, &info);
    if (rc != AXIOM_OK) return rc;
    if (info.hidden_size != AXIOM_QWEN38_MTP_HIDDEN ||
        info.intermediate_size != AXIOM_QWEN38_MTP_INTERMEDIATE ||
        info.num_hidden_layers != 64u ||
        info.num_attention_heads != AXIOM_QWEN38_MTP_ATTENTION_HEADS ||
        info.num_key_value_heads != AXIOM_QWEN38_MTP_KV_HEADS ||
        info.vocab_size != AXIOM_QWEN38_MTP_VOCAB) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}

int inspect_tensor(
        axiom_model *model,
        const mtp_tensor_spec &spec,
        axiom_tensor_info *out) {
    if (!model || !out) return AXIOM_ERR_INVALID_ARGUMENT;
    axiom_tensor_info info{};
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, spec.name, &info);
    if (rc != AXIOM_OK) return rc;
    if (std::strcmp(info.file, kMtpFile) != 0 ||
        info.dtype != AXIOM_TENSOR_DTYPE_BF16 ||
        info.rank != spec.rank) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    uint64_t bytes = 0u;
    rc = expected_bytes(spec, &bytes);
    if (rc != AXIOM_OK || info.byte_count != bytes) {
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
        const mtp_tensor_spec &spec,
        const axiom_tensor_info &info,
        std::vector<uint8_t> *scratch,
        axiom_device_buffer **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || !scratch || !out || info.byte_count == 0u) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    axiom_device_buffer *buffer = nullptr;
    int rc = axiom_device_buffer_create(runtime, &buffer, info.byte_count);
    if (rc != AXIOM_OK) return rc;

    for (uint64_t offset = 0u; offset < info.byte_count;) {
        const uint64_t remaining = info.byte_count - offset;
        const uint64_t count = remaining < scratch->size() ? remaining : scratch->size();
        if (count == 0u) {
            axiom_device_buffer_destroy(buffer);
            return AXIOM_ERR_BUDGET;
        }
        rc = axiom_model_tensor_read_slice(model, spec.name, offset, scratch->data(), count);
        if (rc == AXIOM_OK) {
            rc = axiom_device_buffer_upload(buffer, offset, scratch->data(), count);
        }
        if (rc != AXIOM_OK) {
            axiom_device_buffer_destroy(buffer);
            return rc;
        }
        offset += count;
    }
    *out = buffer;
    return AXIOM_OK;
}

}  // namespace

struct axiom_qwen38_mtp {
    int device = -1;
    uint64_t device_bytes = 0u;
    uint64_t checkpoint_bytes = 0u;
    axiom_device_buffer *buffers[AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT]{};
};

extern "C" void axiom_qwen38_mtp_destroy(axiom_qwen38_mtp *mtp) {
    if (!mtp) return;
    for (axiom_device_buffer *buffer : mtp->buffers) {
        axiom_device_buffer_destroy(buffer);
    }
    delete mtp;
}

extern "C" int axiom_qwen38_mtp_load(
        axiom_model *model,
        axiom_runtime *runtime,
        int device,
        axiom_qwen38_mtp **out) {
    if (out) *out = nullptr;
    if (!model || !runtime || device < 0 || !out) return AXIOM_ERR_INVALID_ARGUMENT;

    uint32_t runtime_device = 0u;
    int rc = axiom_runtime_device_id(runtime, &runtime_device);
    if (rc != AXIOM_OK) return rc;
    if (runtime_device != static_cast<uint32_t>(device)) return AXIOM_ERR_INVALID_ARGUMENT;

    rc = validate_base_geometry(model);
    if (rc != AXIOM_OK) return rc;

    axiom_tensor_info tensor_infos[AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT]{};
    uint64_t checkpoint_bytes = 0u;
    for (uint32_t index = 0u; index < AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT; ++index) {
        rc = inspect_tensor(model, kTensorSpecs[index], &tensor_infos[index]);
        if (rc != AXIOM_OK) return rc;
        if (!checked_add(checkpoint_bytes, tensor_infos[index].byte_count, &checkpoint_bytes)) {
            return AXIOM_ERR_BUDGET;
        }
    }

    std::vector<uint8_t> scratch;
    try {
        scratch.resize(static_cast<size_t>(kUploadChunkBytes));
    } catch (...) {
        return AXIOM_ERR_BUDGET;
    }

    axiom_qwen38_mtp *mtp = new (std::nothrow) axiom_qwen38_mtp();
    if (!mtp) return AXIOM_ERR_BUDGET;
    mtp->device = device;
    mtp->checkpoint_bytes = checkpoint_bytes;

    for (uint32_t index = 0u; index < AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT; ++index) {
        rc = upload_tensor(model, runtime, kTensorSpecs[index], tensor_infos[index], &scratch,
                           &mtp->buffers[index]);
        if (rc != AXIOM_OK || !mtp->buffers[index] ||
            !checked_add(mtp->device_bytes, tensor_infos[index].byte_count, &mtp->device_bytes)) {
            axiom_qwen38_mtp_destroy(mtp);
            return rc != AXIOM_OK ? rc : AXIOM_ERR_BUDGET;
        }
    }

    *out = mtp;
    return AXIOM_OK;
}

extern "C" uint64_t axiom_qwen38_mtp_device_bytes(const axiom_qwen38_mtp *mtp) {
    return mtp ? mtp->device_bytes : 0u;
}

extern "C" int axiom_qwen38_mtp_info_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_info *out) {
    if (!mtp || !out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->hidden_size = AXIOM_QWEN38_MTP_HIDDEN;
    out->intermediate_size = AXIOM_QWEN38_MTP_INTERMEDIATE;
    out->mtp_layers = AXIOM_QWEN38_MTP_LAYERS;
    out->attention_heads = AXIOM_QWEN38_MTP_ATTENTION_HEADS;
    out->key_value_heads = AXIOM_QWEN38_MTP_KV_HEADS;
    out->head_dim = AXIOM_QWEN38_MTP_HEAD_DIM;
    out->vocab_size = AXIOM_QWEN38_MTP_VOCAB;
    out->tensor_count = AXIOM_QWEN38_MTP_REQUIRED_TENSOR_COUNT;
    out->resident_dtype = AXIOM_TENSOR_DTYPE_BF16;
    out->checkpoint_bytes = mtp->checkpoint_bytes;
    out->device_bytes = mtp->device_bytes;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_tensor_info_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_tensor tensor,
        axiom_qwen38_mtp_tensor_info *out) {
    if (!mtp || !out || out->abi_version != AXIOM_ABI_VERSION || !tensor_index_valid(tensor)) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (!mtp->buffers[index]) return AXIOM_ERR_RUNTIME;
    const mtp_tensor_spec &spec = kTensorSpecs[index];
    uint64_t bytes = 0u;
    int rc = expected_bytes(spec, &bytes);
    if (rc != AXIOM_OK) return rc;

    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->tensor = tensor;
    out->dtype = AXIOM_TENSOR_DTYPE_BF16;
    out->rank = spec.rank;
    for (uint32_t dim = 0u; dim < spec.rank; ++dim) out->shape[dim] = spec.shape[dim];
    out->byte_count = bytes;
    const size_t name_length = std::strlen(spec.name);
    if (name_length >= sizeof(out->name)) return AXIOM_ERR_RUNTIME;
    std::memcpy(out->name, spec.name, name_length + 1u);
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_tensor_buffer_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_tensor tensor,
        const axiom_device_buffer **out) {
    if (out) *out = nullptr;
    if (!mtp || !out || !tensor_index_valid(tensor)) return AXIOM_ERR_INVALID_ARGUMENT;
    const axiom_device_buffer *buffer = mtp->buffers[static_cast<uint32_t>(tensor)];
    if (!buffer) return AXIOM_ERR_RUNTIME;
    *out = buffer;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_forward_contract_get(
        const axiom_qwen38_mtp *mtp,
        axiom_qwen38_mtp_forward_contract *out) {
    if (!mtp || !out || out->abi_version != AXIOM_ABI_VERSION) return AXIOM_ERR_INVALID_ARGUMENT;
    const uint32_t abi = out->abi_version;
    std::memset(out, 0, sizeof(*out));
    out->abi_version = abi;
    out->scalar_dtype = AXIOM_TENSOR_DTYPE_F32;
    out->hidden_size = AXIOM_QWEN38_MTP_HIDDEN;
    out->batch = AXIOM_QWEN38_MTP_BATCH;
    out->mtp_layers = AXIOM_QWEN38_MTP_LAYERS;
    out->attention_heads = AXIOM_QWEN38_MTP_ATTENTION_HEADS;
    out->key_value_heads = AXIOM_QWEN38_MTP_KV_HEADS;
    out->head_dim = AXIOM_QWEN38_MTP_HEAD_DIM;
    out->uses_base_embedding = 1u;
    out->uses_base_lm_head = 1u;
    out->requires_mtp_kv_transaction = 1u;
    out->requires_target_cache_transaction = 1u;
    out->output_is_post_mtp_norm = 1u;
    out->zero_centered_rmsnorm = 1u;
    out->attention_output_gate = 1u;
    return AXIOM_OK;
}

extern "C" int axiom_qwen38_mtp_forward_request_validate(
        const axiom_qwen38_mtp *mtp,
        const axiom_qwen38_mtp_forward_request *request) {
    if (!mtp || !request || request->abi_version != AXIOM_ABI_VERSION ||
        !request->target_hidden || !request->proposed_token_embedding || !request->draft_hidden ||
        request->batch != AXIOM_QWEN38_MTP_BATCH || request->stream != nullptr) {
        return AXIOM_ERR_INVALID_ARGUMENT;
    }
    return AXIOM_OK;
}
