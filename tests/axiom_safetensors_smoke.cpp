#include "axiom/axiom.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_text(const char *path, const char *data) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    const size_t n = strlen(data);
    const int ok = fwrite(data, 1, n, f) == n;
    fclose(f);
    return ok;
}

static int write_u64_le(FILE *f, uint64_t v) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = (unsigned char)((v >> (8u * i)) & 0xffu);
    return fwrite(b, 1, 8, f) == 8;
}

static int write_safetensors(const char *path) {
    const char *header =
            "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]},"
            "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[12,18]},"
            "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[18,30]},"
            "\"model.layers.0.self_attn.q_proj.bias\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[30,34]},"
            "\"model.scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[34,38]}}";
    const unsigned char embed[12] = {
            0x80, 0x3f, 0x00, 0x40, 0x40, 0x40,
            0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40};
    const unsigned char weight[12] = {
            0x80, 0x3f, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3f, 0x80, 0x3f};
    const unsigned char bias[4] = {0x00, 0x3f, 0x80, 0xbf};
    const unsigned char scalar[4] = {0x00, 0x00, 0x80, 0x3e};
    const unsigned char norm[6] = {0x80, 0x3f, 0x80, 0x3f, 0x80, 0x3f};
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    const uint64_t n = (uint64_t)strlen(header);
    int ok = write_u64_le(f, n) && fwrite(header, 1, (size_t)n, f) == n;
    if (ok) ok = fwrite(embed, 1, sizeof(embed), f) == sizeof(embed);
    if (ok) ok = fwrite(norm, 1, sizeof(norm), f) == sizeof(norm);
    if (ok) ok = fwrite(weight, 1, sizeof(weight), f) == sizeof(weight);
    if (ok) ok = fwrite(bias, 1, sizeof(bias), f) == sizeof(bias);
    if (ok) ok = fwrite(scalar, 1, sizeof(scalar), f) == sizeof(scalar);
    fclose(f);
    return ok;
}

int main(void) {
    const char *dir = "/tmp/axiom-safetensors-smoke";
    const char *config_path = "/tmp/axiom-safetensors-smoke/config.json";
    const char *index_path = "/tmp/axiom-safetensors-smoke/model.safetensors.index.json";
    const char *tensor_path = "/tmp/axiom-safetensors-smoke/model-00001-of-00001.safetensors";

    unlink(config_path);
    unlink(index_path);
    unlink(tensor_path);
    rmdir(dir);
    if (mkdir(dir, 0700) != 0) {
        fprintf(stderr, "axiom-safetensors-smoke: mkdir failed\n");
        return 1;
    }
    if (!write_text(config_path,
                "{\"model_type\":\"smoke\",\"architectures\":[\"AxiomSmokeForCausalLM\"],"
                "\"hidden_size\":3,\"intermediate_size\":8,\"num_hidden_layers\":1,"
                "\"num_attention_heads\":2,\"num_key_value_heads\":1,\"vocab_size\":16,"
                "\"max_position_embeddings\":32}") ||
        !write_text(index_path,
                "{\"metadata\":{\"total_size\":38},\"weight_map\":{"
                "\"model.embed_tokens.weight\":\"model-00001-of-00001.safetensors\","
                "\"model.layers.0.input_layernorm.weight\":\"model-00001-of-00001.safetensors\","
                "\"model.layers.0.self_attn.q_proj.weight\":\"model-00001-of-00001.safetensors\","
                "\"model.layers.0.self_attn.q_proj.bias\":\"model-00001-of-00001.safetensors\","
                "\"model.scale\":\"model-00001-of-00001.safetensors\"}}") ||
        !write_safetensors(tensor_path)) {
        fprintf(stderr, "axiom-safetensors-smoke: fixture write failed\n");
        return 1;
    }

    axiom_config runtime_cfg;
    memset(&runtime_cfg, 0, sizeof(runtime_cfg));
    runtime_cfg.abi_version = AXIOM_ABI_VERSION;
    runtime_cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *runtime = NULL;
    int rc = axiom_runtime_create(&runtime, &runtime_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-safetensors-smoke: runtime failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    axiom_model_config model_cfg;
    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.abi_version = AXIOM_ABI_VERSION;
    model_cfg.path = dir;
    model_cfg.name = "fixture";
    model_cfg.format = AXIOM_MODEL_FORMAT_SAFETENSORS;
    model_cfg.memory_budget_bytes = 1024 * 1024;
    model_cfg.placement.abi_version = AXIOM_ABI_VERSION;
    model_cfg.placement.kind = AXIOM_PLACEMENT_LOCAL_DEVICE;

    axiom_model *model = NULL;
    rc = axiom_model_open(runtime, &model, &model_cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-safetensors-smoke: model open failed: %s\n", axiom_status_string(rc));
        axiom_runtime_destroy(runtime);
        return 1;
    }

    axiom_model_info mi;
    memset(&mi, 0, sizeof(mi));
    mi.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_info_get(model, &mi);
    if (rc != AXIOM_OK ||
        mi.weight_tensor_count != 5 ||
        mi.safetensors_header_tensor_count != 5 ||
        mi.safetensors_header_data_bytes != 38 ||
        mi.hidden_size != 3) {
        fprintf(stderr, "axiom-safetensors-smoke: bad model info\n");
        return 1;
    }

    axiom_tensor_info ti;
    memset(&ti, 0, sizeof(ti));
    ti.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_tensor_info_get(model, "model.embed_tokens.weight", &ti);
    if (rc != AXIOM_OK ||
        ti.dtype != AXIOM_TENSOR_DTYPE_BF16 ||
        ti.rank != 2 ||
        ti.shape[0] != 2 ||
        ti.shape[1] != 3 ||
        ti.byte_count != 12 ||
        ti.file_offset_begin <= 8) {
        fprintf(stderr, "axiom-safetensors-smoke: bad tensor info\n");
        return 1;
    }

    unsigned char data[12];
    uint64_t bytes = 0;
    rc = axiom_model_tensor_read(model, "model.embed_tokens.weight", data, sizeof(data), &bytes);
    if (rc != AXIOM_OK || bytes != sizeof(data) ||
        data[0] != 0x80 || data[1] != 0x3f || data[10] != 0xc0 || data[11] != 0x40) {
        fprintf(stderr, "axiom-safetensors-smoke: tensor read failed\n");
        return 1;
    }

    memset(&ti, 0, sizeof(ti));
    ti.abi_version = AXIOM_ABI_VERSION;
    rc = axiom_model_tensor_info_get(model, "model.scale", &ti);
    float scalar = 0.0f;
    bytes = 0;
    if (rc == AXIOM_OK) {
        rc = axiom_model_tensor_read(model, "model.scale", &scalar, sizeof(scalar), &bytes);
    }
    if (rc != AXIOM_OK || ti.dtype != AXIOM_TENSOR_DTYPE_F32 || ti.rank != 0 ||
        ti.byte_count != sizeof(float) || bytes != sizeof(float) ||
        scalar < 0.249f || scalar > 0.251f) {
        fprintf(stderr, "axiom-safetensors-smoke: scalar tensor failed\n");
        return 1;
    }

    float embedding[3] = {0.0f, 0.0f, 0.0f};
    rc = axiom_model_embed_token_f32(model, 1, embedding, 3);
    if (rc != AXIOM_OK ||
        embedding[0] < 3.99f || embedding[0] > 4.01f ||
        embedding[1] < 4.99f || embedding[1] > 5.01f ||
        embedding[2] < 5.99f || embedding[2] > 6.01f) {
        fprintf(stderr, "axiom-safetensors-smoke: embedding lookup failed\n");
        return 1;
    }

    float projected[2] = {0.0f, 0.0f};
    float normed[3] = {0.0f, 0.0f, 0.0f};
    rc = axiom_model_rmsnorm_f32(
            model,
            "model.layers.0.input_layernorm.weight",
        embedding,
        normed,
        3,
        1.0e-6f);
    if (rc != AXIOM_OK ||
        normed[0] < 0.789f || normed[0] > 0.790f ||
        normed[1] < 0.986f || normed[1] > 0.987f ||
        normed[2] < 1.184f || normed[2] > 1.185f) {
        fprintf(stderr, "axiom-safetensors-smoke: rmsnorm failed\n");
        return 1;
    }

    rc = axiom_model_linear_bf16_f32(
            model,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            normed,
            3,
            projected,
            2);
    if (rc != AXIOM_OK ||
        projected[0] < 1.28f || projected[0] > 1.30f ||
        projected[1] < 1.16f || projected[1] > 1.18f) {
        fprintf(stderr, "axiom-safetensors-smoke: bf16 matvec failed\n");
        return 1;
    }

    float rope_in[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float rope_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    rc = axiom_model_rope_f32(model, rope_in, rope_out, 1, 4, 1, 10000.0f);
    if (rc != AXIOM_OK ||
        rope_out[0] < 0.540f || rope_out[0] > 0.541f ||
        rope_out[1] < 0.841f || rope_out[1] > 0.842f ||
        rope_out[2] < -0.0101f || rope_out[2] > -0.0099f ||
        rope_out[3] < 0.9999f || rope_out[3] > 1.0001f) {
        fprintf(stderr, "axiom-safetensors-smoke: rope failed\n");
        return 1;
    }

    float q[8] = {0};
    float k[4] = {0};
    float v[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float attn[8] = {0};
    rc = axiom_model_attention_single_f32(model, q, k, v, attn, 4, 2, 2);
    if (rc != AXIOM_OK ||
        attn[0] != 1.0f || attn[1] != 2.0f ||
        attn[2] != 1.0f || attn[3] != 2.0f ||
        attn[4] != 3.0f || attn[5] != 4.0f ||
        attn[6] != 3.0f || attn[7] != 4.0f) {
        fprintf(stderr, "axiom-safetensors-smoke: single attention failed\n");
        return 1;
    }

    float gate[3] = {0.0f, 1.0f, -1.0f};
    float up[3] = {2.0f, 2.0f, 2.0f};
    float act[3] = {0.0f, 0.0f, 0.0f};
    rc = axiom_model_silu_mul_f32(model, gate, up, act, 3);
    if (rc != AXIOM_OK ||
        act[0] != 0.0f ||
        act[1] < 1.462f || act[1] > 1.463f ||
        act[2] < -0.538f || act[2] > -0.537f) {
        fprintf(stderr, "axiom-safetensors-smoke: silu mul failed\n");
        return 1;
    }

    axiom_model_close(model);
    axiom_runtime_destroy(runtime);
    puts("axiom-safetensors-smoke: OK");
    return 0;
}
