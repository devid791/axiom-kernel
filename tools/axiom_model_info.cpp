#include "axiom/axiom.h"

#include <stdio.h>
#include <string.h>

static int open_model(
        axiom_runtime *runtime,
        const char *path,
        const char *name,
        axiom_model **out) {
    axiom_model_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.path = path;
    cfg.name = name;
    cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    cfg.placement.abi_version = AXIOM_ABI_VERSION;
    cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;
    /* Header inspection does not allocate weights.  A fixed 16 GiB ceiling
     * makes the tool reject otherwise valid single-file NVFP4 checkpoints
     * before their tensor schema can be examined.  Admission belongs to the
     * actual resident loader, so keep this inspection-only open unbounded. */
    cfg.memory_budget_bytes = 0;
    const int rc = axiom_model_open(runtime, out, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-model-info: open %s failed: %s\n", path, axiom_status_string(rc));
        return 0;
    }
    return 1;
}

static int print_info(axiom_model *model) {
    axiom_model_info info;
    memset(&info, 0, sizeof(info));
    info.abi_version = AXIOM_ABI_VERSION;
    const int rc = axiom_model_info_get(model, &info);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-model-info: info failed: %s\n", axiom_status_string(rc));
        return 0;
    }
    printf("name=%s\n", info.name);
    printf("path=%s\n", info.path);
    printf("model_type=%s architecture=%s\n", info.model_type, info.architecture);
    printf("bytes=%llu safetensors_bytes=%llu safetensors_files=%u index_total=%llu index_tensors=%u header_data=%llu header_tensors=%u\n",
            (unsigned long long)info.bytes,
            (unsigned long long)info.safetensors_bytes,
            info.safetensors_file_count,
            (unsigned long long)info.weight_index_total_bytes,
            info.weight_tensor_count,
            (unsigned long long)info.safetensors_header_data_bytes,
            info.safetensors_header_tensor_count);
    printf("hidden=%u intermediate=%u layers=%u heads=%u kv_heads=%u vocab=%u ctx=%u\n",
            info.hidden_size,
            info.intermediate_size,
            info.num_hidden_layers,
            info.num_attention_heads,
            info.num_key_value_heads,
            info.vocab_size,
            info.max_context);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "usage: %s MODEL_DIR [MODEL_DIR]\n", argv[0]);
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
        fprintf(stderr, "axiom-model-info: runtime failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model *a = NULL;
    axiom_model *b = NULL;
    if (!open_model(runtime, argv[1], "a", &a) || !print_info(a)) {
        axiom_model_close(a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    if (argc == 3) {
        puts("---");
        if (!open_model(runtime, argv[2], "b", &b) || !print_info(b)) {
            axiom_model_close(b);
            axiom_model_close(a);
            axiom_runtime_destroy(runtime);
            return 1;
        }

        axiom_model_info ia;
        axiom_model_info ib;
        memset(&ia, 0, sizeof(ia));
        memset(&ib, 0, sizeof(ib));
        ia.abi_version = AXIOM_ABI_VERSION;
        ib.abi_version = AXIOM_ABI_VERSION;
        axiom_model_info_get(a, &ia);
        axiom_model_info_get(b, &ib);
        const int same_latent_width = ia.hidden_size != 0 && ia.hidden_size == ib.hidden_size;
        const int same_transformer_shape =
                same_latent_width &&
                ia.num_hidden_layers == ib.num_hidden_layers &&
                ia.num_attention_heads == ib.num_attention_heads &&
                ia.num_key_value_heads == ib.num_key_value_heads;
        printf("---\nsame_latent_width=%d\nsame_transformer_shape=%d\n",
                same_latent_width,
                same_transformer_shape);
    }

    axiom_model_close(b);
    axiom_model_close(a);
    axiom_runtime_destroy(runtime);
    return 0;
}
