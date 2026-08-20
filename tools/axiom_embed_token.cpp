#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        fprintf(stderr, "axiom-embed-token: runtime failed: %s\n", axiom_status_string(rc));
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
        fprintf(stderr, "axiom-embed-token: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK || mi.hidden_size == 0) {
        fprintf(stderr, "axiom-embed-token: model info failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    float *embedding = (float *)calloc(mi.hidden_size, sizeof(float));
    if (!embedding) {
        fprintf(stderr, "axiom-embed-token: alloc failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    rc = axiom_model_embed_token_f32(model, token_id, embedding, mi.hidden_size);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-embed-token: embed failed: %s\n", axiom_status_string(rc));
        free(embedding);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    double sum = 0.0;
    double sq = 0.0;
    for (uint32_t i = 0; i < mi.hidden_size; ++i) {
        sum += embedding[i];
        sq += (double)embedding[i] * (double)embedding[i];
    }
    printf("token_id=%u hidden=%u sum=%.9f l2=%.9f head=",
            token_id,
            mi.hidden_size,
            sum,
            sqrt(sq));
    const uint32_t head = mi.hidden_size < 8 ? mi.hidden_size : 8;
    for (uint32_t i = 0; i < head; ++i) {
        printf("%s%.9f", i ? "," : "", embedding[i]);
    }
    puts("");

    free(embedding);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 0;
}
