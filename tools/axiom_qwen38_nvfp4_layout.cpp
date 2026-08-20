/*
 * axiom_qwen38_nvfp4_layout.cpp
 *
 * Strict checkpoint-layout gate for Qwen3.8-27B NVFP4.  This is deliberately
 * native-Axiom only: it opens the safetensors directory through axiom_model,
 * validates the Qwen3.8 GDN geometry and the exact compressed-tensors planes
 * used by the Unsloth NVFP4 release, then exits before allocating the model.
 *
 * The resident loader consumes these same names and shapes.  Keeping this
 * executable independent makes a bad or changed checkpoint fail before a
 * partial 20+ GiB upload can reach the RTX 5090.
 */
#include "axiom/axiom.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace {

constexpr uint32_t kLayers = 64u;
constexpr uint32_t kHidden = 5120u;
constexpr uint32_t kFfn = 17408u;
constexpr uint32_t kVocab = 248320u;
constexpr uint32_t kNvfp4Layers = 56u;
constexpr uint32_t kFullAttentionInterval = 4u;

int fail(const char *name, const char *why) {
    fprintf(stderr, "axiom-qwen38-nvfp4-layout: %s: %s\n", name, why);
    return 0;
}

int expect_tensor(axiom_model *model,
                  const char *name,
                  axiom_tensor_dtype dtype,
                  uint32_t rank,
                  uint64_t dim0,
                  uint64_t dim1,
                  uint64_t dim2 = 0u) {
    axiom_tensor_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qwen38-nvfp4-layout: missing tensor %s (%s)\n",
                name, axiom_status_string(rc));
        return 0;
    }
    if (info.dtype != dtype || info.rank != rank ||
        info.shape[0] != dim0 ||
        (rank > 1u && info.shape[1] != dim1) ||
        (rank > 2u && info.shape[2] != dim2)) {
        fprintf(stderr,
                "axiom-qwen38-nvfp4-layout: tensor %s has dtype=%u rank=%u"
                " shape=[%llu,%llu,%llu], expected dtype=%u rank=%u"
                " shape=[%llu,%llu,%llu]\n",
                name,
                (unsigned)info.dtype,
                info.rank,
                (unsigned long long)info.shape[0],
                (unsigned long long)info.shape[1],
                (unsigned long long)info.shape[2],
                (unsigned)dtype,
                rank,
                (unsigned long long)dim0,
                (unsigned long long)dim1,
                (unsigned long long)dim2);
        return 0;
    }
    return 1;
}

int expect_layer(axiom_model *model, uint32_t layer) {
    char n[256];
    const int linear = ((layer + 1u) % kFullAttentionInterval) != 0u;

    snprintf(n, sizeof(n), "model.language_model.layers.%u.input_layernorm.weight", layer);
    if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, kHidden, 0u)) return 0;
    snprintf(n, sizeof(n), "model.language_model.layers.%u.post_attention_layernorm.weight", layer);
    if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, kHidden, 0u)) return 0;

    if (linear) {
        const struct {
            const char *suffix;
            uint32_t rows;
        } fp8[] = {
            {"linear_attn.in_proj_qkv.weight", 10240u},
            {"linear_attn.in_proj_z.weight", 6144u},
            {"linear_attn.out_proj.weight", kHidden},
        };
        for (const auto &entry : fp8) {
            snprintf(n, sizeof(n), "model.language_model.layers.%u.%s", layer, entry.suffix);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F8_E4M3, 2u, entry.rows,
                               entry.rows == kHidden ? 6144u : kHidden)) return 0;
            const size_t used = strlen(n);
            if (used + strlen("_scale") + 1u > sizeof(n)) return fail(entry.suffix, "name overflow");
            memcpy(n + used, "_scale", strlen("_scale") + 1u);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 2u, entry.rows, 1u)) return 0;
        }
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.in_proj_a.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 2u, 48u, kHidden)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.in_proj_b.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 2u, 48u, kHidden)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.A_log", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 48u, 0u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.dt_bias", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 48u, 0u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.conv1d.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 3u, 10240u, 1u, 4u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.linear_attn.norm.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 128u, 0u)) return 0;
    } else {
        const struct {
            const char *suffix;
            uint32_t rows;
            uint32_t cols;
        } fp8[] = {
            {"self_attn.q_proj.weight", 12288u, kHidden},
            {"self_attn.k_proj.weight", 1024u, kHidden},
            {"self_attn.v_proj.weight", 1024u, kHidden},
            {"self_attn.o_proj.weight", kHidden, 6144u},
        };
        for (const auto &entry : fp8) {
            snprintf(n, sizeof(n), "model.language_model.layers.%u.%s", layer, entry.suffix);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F8_E4M3, 2u, entry.rows, entry.cols)) return 0;
            const size_t used = strlen(n);
            if (used + strlen("_scale") + 1u > sizeof(n)) return fail(entry.suffix, "name overflow");
            memcpy(n + used, "_scale", strlen("_scale") + 1u);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 2u, entry.rows, 1u)) return 0;
        }
        snprintf(n, sizeof(n), "model.language_model.layers.%u.self_attn.q_norm.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 256u, 0u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.self_attn.k_norm.weight", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 256u, 0u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.self_attn.k_scale", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 1u, 0u)) return 0;
        snprintf(n, sizeof(n), "model.language_model.layers.%u.self_attn.v_scale", layer);
        if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 1u, 1u, 0u)) return 0;
    }

    if (layer < kNvfp4Layers) {
        const struct {
            const char *projection;
            uint32_t rows;
            uint32_t packed_cols;
            uint32_t scale_cols;
        } nvfp4[] = {
            {"gate_proj", kFfn, kHidden / 2u, kHidden / 16u},
            {"up_proj",   kFfn, kHidden / 2u, kHidden / 16u},
            {"down_proj", kHidden, kFfn / 2u, kFfn / 16u},
        };
        for (const auto &entry : nvfp4) {
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.weight_packed",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_U8, 2u,
                               entry.rows, entry.packed_cols)) return 0;
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.weight_scale",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F8_E4M3, 2u,
                               entry.rows, entry.scale_cols)) return 0;
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.input_global_scale",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F32, 1u, 1u, 0u)) return 0;
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.weight_global_scale",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F32, 1u, 1u, 0u)) return 0;
        }
    } else {
        const struct {
            const char *projection;
            uint32_t rows;
            uint32_t cols;
        } fp8[] = {
            {"gate_proj", kFfn, kHidden},
            {"up_proj",   kFfn, kHidden},
            {"down_proj", kHidden, kFfn},
        };
        for (const auto &entry : fp8) {
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.weight",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_F8_E4M3, 2u,
                               entry.rows, entry.cols)) return 0;
            snprintf(n, sizeof(n), "model.language_model.layers.%u.mlp.%s.weight_scale",
                     layer, entry.projection);
            if (!expect_tensor(model, n, AXIOM_TENSOR_DTYPE_BF16, 2u,
                               entry.rows, 1u)) return 0;
        }
    }
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]);
        return 2;
    }

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;
    runtime_cfg.device = 0;
    axiom_runtime *runtime = nullptr;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qwen38-nvfp4-layout: runtime: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = argv[1];
    model_cfg.name = "qwen3.8-27b-nvfp4";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    /* Layout validation reads only headers, not the 22+ GiB tensors. */
    model_cfg.memory_budget_bytes = 0;
    axiom_model *model = nullptr;
    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qwen38-nvfp4-layout: model open: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    int ok = axiom_model_info_get(model, &info) == AXIOM_OK;
    if (!ok || strcmp(info.model_type, "qwen3_5_text") != 0 ||
        info.hidden_size != kHidden || info.intermediate_size != kFfn ||
        info.num_hidden_layers != kLayers || info.num_attention_heads != 24u ||
        info.num_key_value_heads != 4u || info.vocab_size != kVocab ||
        info.max_context != 262144u || info.safetensors_file_count != 2u) {
        fprintf(stderr, "axiom-qwen38-nvfp4-layout: config geometry mismatch\n");
        ok = 0;
    }

    if (ok) ok = expect_tensor(model, "model.language_model.embed_tokens.weight",
                               AXIOM_TENSOR_DTYPE_BF16, 2u, kVocab, kHidden);
    if (ok) ok = expect_tensor(model, "model.language_model.norm.weight",
                               AXIOM_TENSOR_DTYPE_BF16, 1u, kHidden, 0u);
    if (ok) ok = expect_tensor(model, "lm_head.weight",
                               AXIOM_TENSOR_DTYPE_F8_E4M3, 2u, kVocab, kHidden);
    if (ok) ok = expect_tensor(model, "lm_head.weight_scale",
                               AXIOM_TENSOR_DTYPE_BF16, 2u, kVocab, 1u);
    if (ok) ok = expect_tensor(model, "mtp.fc.weight",
                               AXIOM_TENSOR_DTYPE_BF16, 2u, kHidden, 10240u);
    for (uint32_t layer = 0; ok && layer < kLayers; ++layer) {
        ok = expect_layer(model, layer);
    }

    if (ok) {
        printf("{\"status\":\"pass\",\"family\":\"qwen3_8_gdn_nvfp4\","
               "\"layers\":64,\"gdn_layers\":48,\"full_attention_layers\":16,"
               "\"nvfp4_mlp_layers\":56,\"fp8_mlp_layers\":8,"
               "\"native_context\":262144,\"safetensors_bytes\":%llu}\n",
               (unsigned long long)info.safetensors_bytes);
    }

    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return ok ? 0 : 1;
}
