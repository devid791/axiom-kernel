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
    rc = axiom_model_linear_bf16_f32(model, weight, bias, input, input_count, *out, *out_count);
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
    axiom_model *model = NULL;
    float *x = NULL;
    float *attn_norm = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *q_rope = NULL;
    float *k_rope = NULL;
    float *attn = NULL;
    float *attn_out = NULL;
    float *post_attn = NULL;
    float *mlp_norm = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *mlp_act = NULL;
    float *mlp_down = NULL;
    float *block_out = NULL;
    uint32_t head_dim = 0;
    uint32_t qn = 0;
    uint32_t kn = 0;
    uint32_t vn = 0;
    uint32_t gate_n = 0;
    uint32_t up_n = 0;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) return 1;

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = argv[1];
    model_cfg.name = "model";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_cfg.memory_budget_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;

    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-block0-token: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK || mi.hidden_size == 0 || mi.num_attention_heads == 0) goto fail;
    {
        /* Qwen3 decouples head_dim from hidden_size/num_attention_heads, so derive it
         * from the q_proj output dimension (num_attention_heads * head_dim) instead. */
        axiom_tensor_info q_info;
        memset(&q_info, 0, sizeof(q_info));
        q_info.abi_version = AXIOM_ABI_VERSION;
        rc = axiom_model_tensor_info_get(
                model, "model.layers.0.self_attn.q_proj.weight", &q_info);
        if (rc == AXIOM_OK && (q_info.rank != 2 || q_info.shape[0] == 0 ||
                               q_info.shape[0] % mi.num_attention_heads != 0)) {
            rc = AXIOM_ERR_INVALID_ARGUMENT;
        }
        if (rc != AXIOM_OK) goto fail;
        qn = (uint32_t)q_info.shape[0];
    }
    head_dim = qn / mi.num_attention_heads;

    x = (float *)calloc(mi.hidden_size, sizeof(float));
    attn_norm = (float *)calloc(mi.hidden_size, sizeof(float));
    q_rope = (float *)calloc(qn, sizeof(float));
    k_rope = (float *)calloc(mi.num_key_value_heads * head_dim, sizeof(float));
    attn = (float *)calloc(qn, sizeof(float));
    attn_out = (float *)calloc(mi.hidden_size, sizeof(float));
    post_attn = (float *)calloc(mi.hidden_size, sizeof(float));
    mlp_norm = (float *)calloc(mi.hidden_size, sizeof(float));

    if (!x || !attn_norm || !q_rope || !k_rope || !attn || !attn_out || !post_attn || !mlp_norm) goto fail;
    rc = axiom_model_embed_token_f32(model, token_id, x, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_rmsnorm_f32(model, "model.layers.0.input_layernorm.weight", x, attn_norm, mi.hidden_size, 1.0e-6f);
    if (rc != AXIOM_OK) goto fail;
    if (!linear(model, "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.bias", attn_norm, mi.hidden_size, &q, &qn) ||
        !linear(model, "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.k_proj.bias", attn_norm, mi.hidden_size, &k, &kn) ||
        !linear(model, "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.v_proj.bias", attn_norm, mi.hidden_size, &v, &vn)) {
        goto fail;
    }
    rc = axiom_model_rope_f32(model, q, q_rope, mi.num_attention_heads, head_dim, position, 1000000.0f);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_rope_f32(model, k, k_rope, mi.num_key_value_heads, head_dim, position, 1000000.0f);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_attention_single_f32(model, q_rope, k_rope, v, attn, mi.num_attention_heads, mi.num_key_value_heads, head_dim);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_linear_bf16_f32(model, "model.layers.0.self_attn.o_proj.weight", NULL, attn, qn, attn_out, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_smoke_vector_add(runtime, x, attn_out, post_attn, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_rmsnorm_f32(model, "model.layers.0.post_attention_layernorm.weight", post_attn, mlp_norm, mi.hidden_size, 1.0e-6f);
    if (rc != AXIOM_OK) goto fail;
    if (!linear(model, "model.layers.0.mlp.gate_proj.weight", NULL, mlp_norm, mi.hidden_size, &gate, &gate_n) ||
        !linear(model, "model.layers.0.mlp.up_proj.weight", NULL, mlp_norm, mi.hidden_size, &up, &up_n) ||
        gate_n != up_n) {
        goto fail;
    }
    mlp_act = (float *)calloc(gate_n, sizeof(float));
    mlp_down = (float *)calloc(mi.hidden_size, sizeof(float));
    block_out = (float *)calloc(mi.hidden_size, sizeof(float));
    if (!mlp_act || !mlp_down || !block_out) goto fail;
    rc = axiom_model_silu_mul_f32(model, gate, up, mlp_act, gate_n);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_model_linear_bf16_f32(model, "model.layers.0.mlp.down_proj.weight", NULL, mlp_act, gate_n, mlp_down, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;
    rc = axiom_smoke_vector_add(runtime, post_attn, mlp_down, block_out, mi.hidden_size);
    if (rc != AXIOM_OK) goto fail;

    printf("token_id=%u position=%u hidden=%u intermediate=%u\n", token_id, position, mi.hidden_size, gate_n);
    stats("embedding", x, mi.hidden_size);
    stats("attention_o_proj", attn_out, mi.hidden_size);
    stats("post_attention_residual", post_attn, mi.hidden_size);
    stats("mlp_norm", mlp_norm, mi.hidden_size);
    stats("mlp_act", mlp_act, gate_n);
    stats("mlp_down", mlp_down, mi.hidden_size);
    stats("block0_out", block_out, mi.hidden_size);

    free(block_out); free(mlp_down); free(mlp_act); free(up); free(gate);
    free(mlp_norm); free(post_attn); free(attn_out); free(attn); free(k_rope); free(q_rope);
    free(v); free(k); free(q); free(attn_norm); free(x);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 0;

fail:
    fprintf(stderr, "axiom-block0-token: failed: %s\n", axiom_status_string(rc));
    free(block_out); free(mlp_down); free(mlp_act); free(up); free(gate);
    free(mlp_norm); free(post_attn); free(attn_out); free(attn); free(k_rope); free(q_rope);
    free(v); free(k); free(q); free(attn_norm); free(x);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 1;
}
