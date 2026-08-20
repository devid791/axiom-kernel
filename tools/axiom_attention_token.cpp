#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void stats(const char *name, const float *x, uint32_t n) {
    double sum = 0.0;
    double sq = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += x[i];
        sq += (double)x[i] * (double)x[i];
    }
    printf("%s n=%u sum=%.9f l2=%.9f head=", name, n, sum, sqrt(sq));
    const uint32_t head = n < 8 ? n : 8;
    for (uint32_t i = 0; i < head; ++i) printf("%s%.9f", i ? "," : "", x[i]);
    puts("");
}

static int linear(
        axiom_model *model,
        const char *weight,
        const char *bias,
        const float *input,
        uint32_t input_count,
        float **out,
        uint32_t *out_count) {
    axiom_tensor_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, weight, &info);
    if (rc != AXIOM_OK || info.rank != 2) return 0;
    *out_count = (uint32_t)info.shape[0];
    *out = (float *)calloc(*out_count, sizeof(float));
    if (!*out) return 0;
    rc = axiom_model_linear_bf16_f32(
            model, weight, bias, input, input_count, *out, *out_count);
    return rc == AXIOM_OK;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL_DIR TOKEN_ID POSITION\n", argv[0]);
        return 2;
    }
    const uint32_t token_id = (uint32_t)strtoul(argv[2], NULL, 10);
    const uint32_t position = (uint32_t)strtoul(argv[3], NULL, 10);

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-attention-token: runtime failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = argv[1];
    model_cfg.name = "model";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_cfg.memory_budget_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;

    axiom_model *model = NULL;
    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-attention-token: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK || mi.hidden_size == 0 ||
        mi.num_attention_heads == 0 || mi.num_key_value_heads == 0) {
        fprintf(stderr, "axiom-attention-token: model info failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    axiom_tensor_info q_info;
    memset(&q_info, 0, sizeof(q_info));
    q_info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_tensor_info_get(
            model, "model.layers.0.self_attn.q_proj.weight", &q_info);
    if (rc != AXIOM_OK || q_info.rank != 2 || q_info.shape[0] == 0 ||
        (q_info.shape[0] % mi.num_attention_heads) != 0) {
        fprintf(stderr, "axiom-attention-token: q_proj info failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    /* Qwen3 decouples head_dim from hidden_size; derive it from the actual
       q_proj output width (num_attention_heads * head_dim) rather than
       assuming hidden_size == num_attention_heads * head_dim. */
    const uint32_t head_dim = (uint32_t)(q_info.shape[0] / mi.num_attention_heads);

    float *embedding = (float *)calloc(mi.hidden_size, sizeof(float));
    float *normed = (float *)calloc(mi.hidden_size, sizeof(float));
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *q_rope = (float *)calloc(mi.num_attention_heads * head_dim, sizeof(float));
    float *k_rope = (float *)calloc(mi.num_key_value_heads * head_dim, sizeof(float));
    float *attn = (float *)calloc(mi.num_attention_heads * head_dim, sizeof(float));
    float *attn_out = (float *)calloc(mi.hidden_size, sizeof(float));
    uint32_t qn = 0, kn = 0, vn = 0;
    if (!embedding || !normed || !q_rope || !k_rope || !attn || !attn_out) goto fail;

    rc = axiom_model_embed_token_f32(model, token_id, embedding, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_rmsnorm_f32(
            model,
            "model.layers.0.input_layernorm.weight",
            embedding,
            normed,
            mi.hidden_size,
            1.0e-6f);
    if (rc != AXIOM_OK) goto fail;
    if (!linear(model, "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.bias", normed, mi.hidden_size, &q, &qn) ||
        !linear(model, "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.k_proj.bias", normed, mi.hidden_size, &k, &kn) ||
        !linear(model, "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.v_proj.bias", normed, mi.hidden_size, &v, &vn)) {
        goto fail;
    }
    rc = axiom_model_rope_f32(model, q, q_rope, mi.num_attention_heads, head_dim, position, 1000000.0f);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_rope_f32(model, k, k_rope, mi.num_key_value_heads, head_dim, position, 1000000.0f);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_attention_single_f32(
            model,
            q_rope,
            k_rope,
            v,
            attn,
            mi.num_attention_heads,
            mi.num_key_value_heads,
            head_dim);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_linear_bf16_f32(
            model,
            "model.layers.0.self_attn.o_proj.weight",
            NULL,
            attn,
            mi.num_attention_heads * head_dim,
            attn_out,
            mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;

    printf("token_id=%u position=%u heads=%u kv_heads=%u head_dim=%u\n",
            token_id, position, mi.num_attention_heads, mi.num_key_value_heads, head_dim);
    stats("q_rope", q_rope, qn);
    stats("k_rope", k_rope, kn);
    stats("v", v, vn);
    stats("attention_single", attn, mi.num_attention_heads * head_dim);
    stats("attention_o_proj", attn_out, mi.hidden_size);

    free(attn_out);
    free(attn);
    free(k_rope);
    free(q_rope);
    free(v);
    free(k);
    free(q);
    free(normed);
    free(embedding);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 0;

fail:
    fprintf(stderr, "axiom-attention-token: failed: %s\n", axiom_status_string(rc));
    free(attn_out);
    free(attn);
    free(k_rope);
    free(q_rope);
    free(v);
    free(k);
    free(q);
    free(normed);
    free(embedding);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 1;
}
