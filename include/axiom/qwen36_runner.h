#ifndef AXIOM_QWEN36_RUNNER_H
#define AXIOM_QWEN36_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN36_GENERATE_CONFIG_ABI_VERSION 1u

typedef struct axiom_qwen36_generate_config {
    uint32_t abi_version;
    const char *model_dir;
    const char *ids;
    const char *prompt;
    int max_new;
    const char *trace_json;
    const char *proof_model_id;
    const char *proof_family;
    const char *remote_join;
    const char *remote_transport;
    uint32_t remote_start;
    uint32_t remote_end;
    uint32_t remote_node_id;
    uint32_t remote_node_count;
    uint32_t remote_target_node;
    uint32_t remote_target_device;
    int has_remote_start;
    int has_remote_end;
    int has_remote_node_id;
    int has_remote_node_count;
    int has_remote_target_node;
    int has_remote_target_device;
} axiom_qwen36_generate_config;

int axiom_qwen36_generate(const axiom_qwen36_generate_config *config);
int axiom_qwen36_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif
