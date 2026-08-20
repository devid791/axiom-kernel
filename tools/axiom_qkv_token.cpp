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
    for (uint32_t i = 0; i < head; ++i) {
        printf("%s%.9f", i ? "," : "", x[i]);
    }
    puts("");
}

static int run_linear(
        axiom_model *model,
        const char *name,
        const char *bias,
        const float *input,
        uint32_t input_count) {
    axiom_tensor_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK || info.rank != 2) {
        fprintf(stderr, "axiom-qkv-token: tensor info failed for %s: %s\n", name, axiom_status_string(rc));
        return 0;
    }
    float *out = (float *)calloc((size_t)info.shape[0], sizeof(float));
    if (!out) return 0;
    rc = axiom_model_linear_bf16_f32(
            model,
            name,
            bias,
            input,
            input_count,
            out,
            (uint32_t)info.shape[0]);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qkv-token: linear failed for %s: %s\n", name, axiom_status_string(rc));
        free(out);
        return 0;
    }
    stats(name, out, (uint32_t)info.shape[0]);
    free(out);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TOKEN_ID\n", argv[0]);
        return 2;
    }
    const uint32_t token_id = (uint32_t)strtoul(argv[2], NULL, 10);

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qkv-token: runtime failed: %s\n", axiom_status_string(rc));
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
        fprintf(stderr, "axiom-qkv-token: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK || mi.hidden_size == 0) {
        fprintf(stderr, "axiom-qkv-token: model info failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    float *embedding = (float *)calloc(mi.hidden_size, sizeof(float));
    float *normed = (float *)calloc(mi.hidden_size, sizeof(float));
    if (!embedding || !normed) {
        fprintf(stderr, "axiom-qkv-token: alloc failed\n");
        free(normed);
        free(embedding);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    rc = axiom_model_embed_token_f32(model, token_id, embedding, mi.hidden_size);
    if (rc == AXIOM_OK) {
        rc = axiom_model_rmsnorm_f32(
                model,
                "model.layers.0.input_layernorm.weight",
                embedding,
                normed,
                mi.hidden_size,
                1.0e-6f);
    }
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-qkv-token: embed/rmsnorm failed: %s\n", axiom_status_string(rc));
        free(normed);
        free(embedding);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    printf("token_id=%u hidden=%u\n", token_id, mi.hidden_size);
    stats("embedding", embedding, mi.hidden_size);
    stats("rmsnorm", normed, mi.hidden_size);
    const int ok =
            run_linear(model, "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.q_proj.bias", normed, mi.hidden_size) &&
            run_linear(model, "model.layers.0.self_attn.k_proj.weight",
                    "model.layers.0.self_attn.k_proj.bias", normed, mi.hidden_size) &&
            run_linear(model, "model.layers.0.self_attn.v_proj.weight",
                    "model.layers.0.self_attn.v_proj.bias", normed, mi.hidden_size);

    free(normed);
    free(embedding);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return ok ? 0 : 1;
}
