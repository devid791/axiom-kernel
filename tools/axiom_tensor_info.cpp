#include "axiom/axiom.h"

#include <stdio.h>
#include <string.h>

static const char *dtype_name(axiom_tensor_dtype dtype) {
    switch (dtype) {
    case AXIOM_TENSOR_DTYPE_F32:
        return "F32";
    case AXIOM_TENSOR_DTYPE_F16:
        return "F16";
    case AXIOM_TENSOR_DTYPE_BF16:
        return "BF16";
    case AXIOM_TENSOR_DTYPE_I64:
        return "I64";
    case AXIOM_TENSOR_DTYPE_I32:
        return "I32";
    case AXIOM_TENSOR_DTYPE_U8:
        return "U8";
    case AXIOM_TENSOR_DTYPE_F8_E4M3:
        return "F8_E4M3";
    case AXIOM_TENSOR_DTYPE_F8_E5M2:
        return "F8_E5M2";
    default:
        return "UNKNOWN";
    }
}

static int print_tensor(axiom_model *model, const char *name) {
    axiom_tensor_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_tensor_info_get(model, name, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-info: %s failed: %s\n", name, axiom_status_string(rc));
        return 0;
    }
    printf("name=%s\n", info.name);
    printf("file=%s dtype=%s rank=%u shape=[", info.file, dtype_name(info.dtype), info.rank);
    for (uint32_t i = 0; i < info.rank; ++i) {
        printf("%s%llu", i ? "," : "", (unsigned long long)info.shape[i]);
    }
    printf("]\n");
    printf("data_offsets=[%llu,%llu] file_offsets=[%llu,%llu] bytes=%llu\n",
            (unsigned long long)info.data_offset_begin,
            (unsigned long long)info.data_offset_end,
            (unsigned long long)info.file_offset_begin,
            (unsigned long long)info.file_offset_end,
            (unsigned long long)info.byte_count);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL_DIR TENSOR_NAME [TENSOR_NAME...]\n", argv[0]);
        return 2;
    }

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;
    runtime_cfg.device = 0;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-info: runtime failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = argv[1];
    model_cfg.name = "model";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    model_cfg.memory_budget_bytes = 16ull * 1024ull * 1024ull * 1024ull;

    axiom_model *model = NULL;
    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-tensor-info: open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    int ok = 1;
    for (int i = 2; i < argc; ++i) {
        if (i > 2) puts("---");
        ok = print_tensor(model, argv[i]) && ok;
    }

    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    return ok ? 0 : 1;
}
