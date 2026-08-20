#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL_DIR TOKEN_ID WEIGHT_TENSOR BIAS_TENSOR_OR_NONE\n", argv[0]);
        return 2;
    }
    const uint32_t token_id = (uint32_t)strtoul(argv[2], NULL, 10);
    const char *bias_name = strcmp(argv[4], "-") == 0 ? NULL : argv[4];

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-linear-token: runtime failed: %s\n", axiom_status_string(rc));
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
        fprintf(stderr, "axiom-linear-token: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK || mi.hidden_size == 0) {
        fprintf(stderr, "axiom-linear-token: model info failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_tensor_info wi;
    memset(&wi, 0, sizeof(wi));
    wi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_tensor_info_get(model, argv[3], &wi);
    if (rc != AXIOM_OK || wi.rank != 2) {
        fprintf(stderr, "axiom-linear-token: weight info failed: %s\n", axiom_status_string(rc));
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    float *embedding = (float *)calloc(mi.hidden_size, sizeof(float));
    float *out = (float *)calloc((size_t)wi.shape[0], sizeof(float));
    if (!embedding || !out) {
        fprintf(stderr, "axiom-linear-token: alloc failed\n");
        free(out);
        free(embedding);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    rc = axiom_model_embed_token_f32(model, token_id, embedding, mi.hidden_size);
    if (rc == AXIOM_OK) {
        rc = axiom_model_linear_bf16_f32(
                model,
                argv[3],
                bias_name,
                embedding,
                mi.hidden_size,
                out,
                (uint32_t)wi.shape[0]);
    }
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-linear-token: projection failed: %s\n", axiom_status_string(rc));
        free(out);
        free(embedding);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    double sum = 0.0;
    double sq = 0.0;
    for (uint64_t i = 0; i < wi.shape[0]; ++i) {
        sum += out[i];
        sq += (double)out[i] * (double)out[i];
    }
    printf("token_id=%u weight=%s rows=%llu cols=%llu sum=%.9f l2=%.9f head=",
            token_id,
            argv[3],
            (unsigned long long)wi.shape[0],
            (unsigned long long)wi.shape[1],
            sum,
            sqrt(sq));
    const uint64_t head = wi.shape[0] < 8 ? wi.shape[0] : 8;
    for (uint64_t i = 0; i < head; ++i) {
        printf("%s%.9f", i ? "," : "", out[i]);
    }
    puts("");

    free(out);
    free(embedding);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 0;
}
