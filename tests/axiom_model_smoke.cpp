#include "axiom/axiom.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int expect_ok(int rc, const char *what) {
    if (rc == AXIOM_OK) return 1;
    fprintf(stderr, "axiom-model-smoke: %s failed: %s\n", what, axiom_status_string(rc));
    return 0;
}

static int write_file(const char *path, size_t bytes, unsigned char seed) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    for (size_t i = 0; i < bytes; i++) {
        unsigned char v = (unsigned char)(seed + (unsigned char)i);
        if (fwrite(&v, 1, 1, fp) != 1) {
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}

int main(void) {
    const char *dir = "/tmp/axiom-model-smoke";
    const char *a_path = "/tmp/axiom-model-smoke/generalist.axm";
    const char *b_path = "/tmp/axiom-model-smoke/coder.axm";
    unlink(a_path);
    unlink(b_path);
    rmdir(dir);
    if (mkdir(dir, 0700) != 0) {
        fprintf(stderr, "axiom-model-smoke: could not create %s\n", dir);
        return 1;
    }
    if (!write_file(a_path, 4096, 17) || !write_file(b_path, 8192, 29)) {
        fprintf(stderr, "axiom-model-smoke: could not create test model files in %s\n", dir);
        return 1;
    }

    axiom_config runtime_config;
    memset(&runtime_config, 0, sizeof(runtime_config));
    runtime_config.abi_version = AXIOM_ABI_VERSION;
    runtime_config.backend = AXIOM_BACKEND_CUDA;
    runtime_config.device = 0;

    axiom_runtime *runtime = NULL;
    if (!expect_ok(axiom_runtime_create(&runtime, &runtime_config), "runtime create")) return 1;

    axiom_model_config model_a_config;
    memset(&model_a_config, 0, sizeof(model_a_config));
    model_a_config.abi_version = AXIOM_ABI_VERSION;
    model_a_config.path = a_path;
    model_a_config.name = "generalist";
    model_a_config.format = AXIOM_MODEL_FORMAT_NATIVE;
    model_a_config.memory_budget_bytes = 16384;
    model_a_config.max_context = 4096;
    model_a_config.placement.abi_version = AXIOM_ABI_VERSION;
    model_a_config.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;

    axiom_model_config model_b_config = model_a_config;
    model_b_config.path = b_path;
    model_b_config.name = "coder";
    model_b_config.memory_budget_bytes = 16384;

    axiom_model *model_a = NULL;
    axiom_model *model_b = NULL;
    if (!expect_ok(axiom_model_open(runtime, &model_a, &model_a_config), "model generalist open") ||
        !expect_ok(axiom_model_open(runtime, &model_b, &model_b_config), "model coder open")) {
        axiom_model_close(model_b);
        axiom_model_close(model_a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_entity_config entity_a_config;
    memset(&entity_a_config, 0, sizeof(entity_a_config));
    entity_a_config.abi_version = AXIOM_ABI_VERSION;
    entity_a_config.name = "alice-generalist";
    entity_a_config.role = "planner";
    entity_a_config.memory_budget_bytes = 4096;

    axiom_entity_config entity_b_config = entity_a_config;
    entity_b_config.name = "alice-coder";
    entity_b_config.role = "coder";

    axiom_entity *entity_a = NULL;
    axiom_entity *entity_b = NULL;
    if (!expect_ok(axiom_entity_create(runtime, &entity_a, model_a, &entity_a_config), "entity generalist create") ||
        !expect_ok(axiom_entity_create(runtime, &entity_b, model_b, &entity_b_config), "entity coder create")) {
        axiom_entity_destroy(entity_b);
        axiom_entity_destroy(entity_a);
        axiom_model_close(model_b);
        axiom_model_close(model_a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_session_config session_config;
    memset(&session_config, 0, sizeof(session_config));
    session_config.abi_version = AXIOM_ABI_VERSION;
    session_config.max_context = 2048;
    session_config.kv_budget_bytes = 1024 * 1024;

    axiom_session *session_a = NULL;
    axiom_session *session_b = NULL;
    if (!expect_ok(axiom_session_create(entity_a, &session_a, &session_config), "session generalist create") ||
        !expect_ok(axiom_session_create(entity_b, &session_b, &session_config), "session coder create")) {
        axiom_session_destroy(session_b);
        axiom_session_destroy(session_a);
        axiom_entity_destroy(entity_b);
        axiom_entity_destroy(entity_a);
        axiom_model_close(model_b);
        axiom_model_close(model_a);
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info info_a;
    memset(&info_a, 0, sizeof(info_a));
    info_a.abi_version = AXIOM_ABI_VERSION;
    axiom_model_info info_b = info_a;
    if (!expect_ok(axiom_model_info_get(model_a, &info_a), "model generalist info") ||
        !expect_ok(axiom_model_info_get(model_b, &info_b), "model coder info")) {
        return 1;
    }
    if (info_a.bytes != 4096 || info_b.bytes != 8192 ||
        info_a.entity_count != 1 || info_b.entity_count != 1) {
        fprintf(stderr, "axiom-model-smoke: bad model info\n");
        return 1;
    }

    axiom_entity_info ent_a;
    memset(&ent_a, 0, sizeof(ent_a));
    ent_a.abi_version = AXIOM_ABI_VERSION;
    axiom_entity_info ent_b = ent_a;
    if (!expect_ok(axiom_entity_info_get(entity_a, &ent_a), "entity generalist info") ||
        !expect_ok(axiom_entity_info_get(entity_b, &ent_b), "entity coder info")) {
        return 1;
    }
    if (ent_a.session_count != 1 || ent_b.session_count != 1) {
        fprintf(stderr, "axiom-model-smoke: bad session count\n");
        return 1;
    }

    axiom_session_destroy(session_b);
    axiom_session_destroy(session_a);
    axiom_entity_destroy(entity_b);
    axiom_entity_destroy(entity_a);
    axiom_model_close(model_b);
    axiom_model_close(model_a);
    axiom_runtime_destroy(runtime);

    puts("axiom-model-smoke: OK");
    return 0;
}
