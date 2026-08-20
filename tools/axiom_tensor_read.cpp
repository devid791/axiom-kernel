#include <cstdlib>
#include "axiom/axiom.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv1a(const unsigned char *data, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= (uint64_t)data[i];
        h *= 1099511628211ull;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TENSOR_NAME\n", argv[0]);
        return 2;
    }

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-read: runtime failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = argv[1];
    model_cfg.name = "model";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    {
        const char *bg = std::getenv("AXIOM_MODEL_BUDGET_GB");
        model_cfg.memory_budget_bytes =
            (bg && bg[0] ? (unsigned long long)strtoull(bg, nullptr, 10) : 256ull)
            * 1024ull * 1024ull * 1024ull;
    }
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;

    axiom_model *model = NULL;
    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-read: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_tensor_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_tensor_info_get(model, argv[2], &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-read: tensor info failed: %s\n", axiom_status_string(rc));
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    unsigned char *data = (unsigned char *)malloc((size_t)info.byte_count);
    if (!data) {
        fprintf(stderr, "axiom-tensor-read: alloc failed\n");
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }
    uint64_t bytes = 0;
    rc = axiom_model_tensor_read(model, argv[2], data, info.byte_count, &bytes);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-read: read failed: %s\n", axiom_status_string(rc));
        free(data);
        axiom_model_close(model);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    printf("name=%s bytes=%llu hash=0x%016llx head=",
            argv[2],
            (unsigned long long)bytes,
            (unsigned long long)fnv1a(data, bytes));
    const uint64_t head = bytes < 16 ? bytes : 16;
    for (uint64_t i = 0; i < head; ++i) printf("%02x", data[i]);
    puts("");

    free(data);
    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return 0;
}
