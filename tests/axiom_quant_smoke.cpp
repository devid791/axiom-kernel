#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8u);
}

static int close_enough(float got, float want) {
    return fabsf(got - want) < 1.0e-3f;
}

int main(void) {
    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;

    axiom_runtime *rt = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "runtime_create failed: %s\n", axiom_status_string(rc));
        return 1;
    }

    float q8_input[32];
    for (uint32_t i = 0; i < 32; ++i) q8_input[i] = 1.0f;
    uint8_t q8[2 * 34];
    memset(q8, 0, sizeof(q8));
    put_f16(q8, 0x3c00u);
    memset(q8 + 2, 1, 32);
    put_f16(q8 + 34, 0x3c00u);
    memset(q8 + 34 + 2, 2, 32);
    float q8_out[2] = {0.0f, 0.0f};
    rc = axiom_runtime_q8_0_matvec_f32(rt, q8, q8_input, q8_out, 2, 32);
    if (rc != AXIOM_OK || !close_enough(q8_out[0], 32.0f) || !close_enough(q8_out[1], 64.0f)) {
        fprintf(stderr, "q8_0 failed rc=%d out=%f,%f\n", rc, q8_out[0], q8_out[1]);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer *q8_w = NULL;
    axiom_device_buffer *q8_x = NULL;
    axiom_device_buffer *q8_y = NULL;
    float q8_dev_out[2] = {0.0f, 0.0f};
    rc = axiom_device_buffer_create(rt, &q8_w, sizeof(q8));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q8_x, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q8_y, sizeof(q8_dev_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(q8_w, 0, q8, sizeof(q8)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(q8_x, 0, q8_input, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, q8_w, 0, q8_x, 0, q8_y, 0, 2, 32) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q8_y, 0, q8_dev_out, sizeof(q8_dev_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(q8_dev_out[0], 32.0f) || !close_enough(q8_dev_out[1], 64.0f)) {
        fprintf(stderr, "q8_0 resident failed rc=%d out=%f,%f\n", rc, q8_dev_out[0], q8_dev_out[1]);
        axiom_device_buffer_destroy(q8_y);
        axiom_device_buffer_destroy(q8_x);
        axiom_device_buffer_destroy(q8_w);
        axiom_runtime_destroy(rt);
        return 1;
    }
    float q8_gather_out[32];
    float q8_logits[2] = {-1.0f, 3.0f};
    axiom_device_buffer *q8_embed_out = NULL;
    axiom_device_buffer *q8_token = NULL;
    axiom_device_buffer *q8_logits_dev = NULL;
    memset(q8_gather_out, 0, sizeof(q8_gather_out));
    rc = axiom_device_buffer_create(rt, &q8_embed_out, sizeof(q8_gather_out));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q8_token, sizeof(uint32_t)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q8_logits_dev, sizeof(q8_logits)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(q8_logits_dev, 0, q8_logits, sizeof(q8_logits)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_f32_argmax_to_buffer_device(rt, q8_logits_dev, 0, 2, q8_token, 0) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_embedding_gather_token_f32_device(
            rt, q8_w, 0, q8_token, 0, q8_embed_out, 0, 2, 32) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q8_embed_out, 0, q8_gather_out, sizeof(q8_gather_out)) : rc;
    for (uint32_t i = 0; rc == AXIOM_OK && i < 32u; ++i) {
        if (!close_enough(q8_gather_out[i], 2.0f)) rc = AXIOM_ERR_RUNTIME;
    }
    if (rc == AXIOM_OK) {
        rc = axiom_runtime_q8_0_embedding_gather_f32_device(rt, q8_w, 0, q8_embed_out, 0, 0, 2, 32);
        rc = rc == AXIOM_OK ? axiom_device_buffer_download(q8_embed_out, 0, q8_gather_out, sizeof(q8_gather_out)) : rc;
        for (uint32_t i = 0; rc == AXIOM_OK && i < 32u; ++i) {
            if (!close_enough(q8_gather_out[i], 1.0f)) rc = AXIOM_ERR_RUNTIME;
        }
    }
    if (rc != AXIOM_OK) {
        fprintf(stderr, "q8_0 embedding gather/token chain failed rc=%d first=%f\n", rc, q8_gather_out[0]);
        axiom_device_buffer_destroy(q8_logits_dev);
        axiom_device_buffer_destroy(q8_token);
        axiom_device_buffer_destroy(q8_embed_out);
        axiom_device_buffer_destroy(q8_y);
        axiom_device_buffer_destroy(q8_x);
        axiom_device_buffer_destroy(q8_w);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(q8_logits_dev);
    axiom_device_buffer_destroy(q8_token);
    axiom_device_buffer_destroy(q8_embed_out);
    axiom_device_buffer_destroy(q8_y);
    axiom_device_buffer_destroy(q8_x);
    axiom_device_buffer_destroy(q8_w);

    float norm_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float norm_w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float norm_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    axiom_device_buffer *norm_x_dev = NULL;
    axiom_device_buffer *norm_w_dev = NULL;
    axiom_device_buffer *norm_y_dev = NULL;
    rc = axiom_device_buffer_create(rt, &norm_x_dev, sizeof(norm_x));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &norm_w_dev, sizeof(norm_w)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &norm_y_dev, sizeof(norm_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(norm_x_dev, 0, norm_x, sizeof(norm_x)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(norm_w_dev, 0, norm_w, sizeof(norm_w)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_rmsnorm_f32_device(rt, norm_w_dev, 0, norm_x_dev, 0, norm_y_dev, 0, 4, 1.0e-6f) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(norm_y_dev, 0, norm_out, sizeof(norm_out)) : rc;
    const float norm_inv = 1.0f / sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1.0e-6f);
    if (rc != AXIOM_OK || !close_enough(norm_out[0], norm_x[0] * norm_inv) ||
        !close_enough(norm_out[3], norm_x[3] * norm_inv)) {
        fprintf(stderr, "rmsnorm resident failed rc=%d out=%f,%f\n", rc, norm_out[0], norm_out[3]);
        axiom_device_buffer_destroy(norm_y_dev);
        axiom_device_buffer_destroy(norm_w_dev);
        axiom_device_buffer_destroy(norm_x_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(norm_y_dev);
    axiom_device_buffer_destroy(norm_w_dev);
    axiom_device_buffer_destroy(norm_x_dev);

    float multi_q[2] = {0.0f, 0.0f};
    float multi_kv[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float multi_sink[1] = {-100.0f};
    float multi_out[2] = {0.0f, 0.0f};
    axiom_device_buffer *multi_q_dev = NULL;
    axiom_device_buffer *multi_kv_dev = NULL;
    axiom_device_buffer *multi_sink_dev = NULL;
    axiom_device_buffer *multi_out_dev = NULL;
    rc = axiom_device_buffer_create(rt, &multi_q_dev, sizeof(multi_q));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &multi_kv_dev, sizeof(multi_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &multi_sink_dev, sizeof(multi_sink)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &multi_out_dev, sizeof(multi_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(multi_q_dev, 0, multi_q, sizeof(multi_q)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(multi_kv_dev, 0, multi_kv, sizeof(multi_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(multi_sink_dev, 0, multi_sink, sizeof(multi_sink)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_attention_multi_kv_single_f32_device(
            rt, multi_q_dev, 0, multi_kv_dev, 0, multi_sink_dev, 0, multi_out_dev, 0, 1, 2, 2, 1.0e-6f) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(multi_out_dev, 0, multi_out, sizeof(multi_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(multi_out[0], 4.0f) || !close_enough(multi_out[1], 6.0f)) {
        fprintf(stderr, "multi-kv attention failed rc=%d out=%f,%f\n", rc, multi_out[0], multi_out[1]);
        axiom_device_buffer_destroy(multi_out_dev);
        axiom_device_buffer_destroy(multi_sink_dev);
        axiom_device_buffer_destroy(multi_kv_dev);
        axiom_device_buffer_destroy(multi_q_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(multi_out_dev);
    axiom_device_buffer_destroy(multi_sink_dev);
    axiom_device_buffer_destroy(multi_kv_dev);
    axiom_device_buffer_destroy(multi_q_dev);

    float sliding_q[2] = {1.0f, 0.0f};
    float sliding_current_kv[2] = {0.0f, 50.0f};
    float sliding_ring_kv[4 * 2] = {
        20.0f, 30.0f,
        0.0f, 40.0f,
        5.0f, 60.0f,
        7.0f, 80.0f,
    };
    float sliding_sink[1] = {-100.0f};
    float sliding_out[2] = {0.0f, 0.0f};
    axiom_device_buffer *sliding_q_dev = NULL;
    axiom_device_buffer *sliding_current_dev = NULL;
    axiom_device_buffer *sliding_ring_dev = NULL;
    axiom_device_buffer *sliding_sink_dev = NULL;
    axiom_device_buffer *sliding_out_dev = NULL;
    rc = axiom_device_buffer_create(rt, &sliding_q_dev, sizeof(sliding_q));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &sliding_current_dev, sizeof(sliding_current_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &sliding_ring_dev, sizeof(sliding_ring_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &sliding_sink_dev, sizeof(sliding_sink)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &sliding_out_dev, sizeof(sliding_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(sliding_q_dev, 0, sliding_q, sizeof(sliding_q)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(sliding_current_dev, 0, sliding_current_kv, sizeof(sliding_current_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(sliding_ring_dev, 0, sliding_ring_kv, sizeof(sliding_ring_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(sliding_sink_dev, 0, sliding_sink, sizeof(sliding_sink)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_sliding_attention_ring_f32_device(
            rt, sliding_q_dev, 0, sliding_current_dev, 0, sliding_ring_dev, 0,
            sliding_sink_dev, 0, sliding_out_dev, 0, 1, 2, 4, 2, 2, 1.0e-6f) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(sliding_out_dev, 0, sliding_out, sizeof(sliding_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(sliding_out[0], 20.0f) || !close_enough(sliding_out[1], 30.0f)) {
        fprintf(stderr, "sliding ring attention failed rc=%d out=%f,%f\n", rc, sliding_out[0], sliding_out[1]);
        axiom_device_buffer_destroy(sliding_out_dev);
        axiom_device_buffer_destroy(sliding_sink_dev);
        axiom_device_buffer_destroy(sliding_ring_dev);
        axiom_device_buffer_destroy(sliding_current_dev);
        axiom_device_buffer_destroy(sliding_q_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(sliding_out_dev);
    axiom_device_buffer_destroy(sliding_sink_dev);
    axiom_device_buffer_destroy(sliding_ring_dev);
    axiom_device_buffer_destroy(sliding_current_dev);
    axiom_device_buffer_destroy(sliding_q_dev);

    uint8_t f16_raw[2 * 4 * 2];
    float f16_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float f16_out[2] = {0.0f, 0.0f};
    for (uint32_t i = 0; i < 4; ++i) put_f16(f16_raw + i * 2u, 0x3c00u);
    for (uint32_t i = 0; i < 4; ++i) put_f16(f16_raw + (4u + i) * 2u, 0x4000u);
    axiom_device_buffer *f16_w_dev = NULL;
    axiom_device_buffer *f16_x_dev = NULL;
    axiom_device_buffer *f16_y_dev = NULL;
    rc = axiom_device_buffer_create(rt, &f16_w_dev, sizeof(f16_raw));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &f16_x_dev, sizeof(f16_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &f16_y_dev, sizeof(f16_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(f16_w_dev, 0, f16_raw, sizeof(f16_raw)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(f16_x_dev, 0, f16_input, sizeof(f16_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_f16_matvec_f32_device(
            rt, f16_w_dev, 0, f16_x_dev, 0, f16_y_dev, 0, 2, 4) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(f16_y_dev, 0, f16_out, sizeof(f16_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(f16_out[0], 10.0f) || !close_enough(f16_out[1], 20.0f)) {
        fprintf(stderr, "f16 resident failed rc=%d out=%f,%f\n", rc, f16_out[0], f16_out[1]);
        axiom_device_buffer_destroy(f16_y_dev);
        axiom_device_buffer_destroy(f16_x_dev);
        axiom_device_buffer_destroy(f16_w_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(f16_y_dev);
    axiom_device_buffer_destroy(f16_x_dev);
    axiom_device_buffer_destroy(f16_w_dev);

    float csa_kv[4 * 4] = {0.0f};
    float csa_gate[4 * 4] = {0.0f};
    float csa_bias[4 * 4] = {0.0f};
    float csa_out[2] = {0.0f, 0.0f};
    for (uint32_t i = 0; i < 4; ++i) {
        csa_kv[i * 4u + 2u] = 1.0f + 2.0f * (float)i;
        csa_kv[i * 4u + 3u] = 2.0f + 2.0f * (float)i;
    }
    axiom_device_buffer *csa_kv_dev = NULL;
    axiom_device_buffer *csa_gate_dev = NULL;
    axiom_device_buffer *csa_bias_dev = NULL;
    axiom_device_buffer *csa_out_dev = NULL;
    rc = axiom_device_buffer_create(rt, &csa_kv_dev, sizeof(csa_kv));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_gate_dev, sizeof(csa_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_bias_dev, sizeof(csa_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_out_dev, sizeof(csa_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_kv_dev, 0, csa_kv, sizeof(csa_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_gate_dev, 0, csa_gate, sizeof(csa_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_bias_dev, 0, csa_bias, sizeof(csa_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_csa_cold_window_f32_device(
            rt, csa_kv_dev, 0, csa_gate_dev, 0, csa_bias_dev, 0, csa_out_dev, 0, 4, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(csa_out_dev, 0, csa_out, sizeof(csa_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(csa_out[0], 4.0f) || !close_enough(csa_out[1], 5.0f)) {
        fprintf(stderr, "csa cold window failed rc=%d out=%f,%f\n", rc, csa_out[0], csa_out[1]);
        axiom_device_buffer_destroy(csa_out_dev);
        axiom_device_buffer_destroy(csa_bias_dev);
        axiom_device_buffer_destroy(csa_gate_dev);
        axiom_device_buffer_destroy(csa_kv_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(csa_out_dev);
    axiom_device_buffer_destroy(csa_bias_dev);
    axiom_device_buffer_destroy(csa_gate_dev);
    axiom_device_buffer_destroy(csa_kv_dev);

    float csa_ring_bias[4 * 4] = {0.0f};
    float csa_ring_out[2] = {0.0f, 0.0f};
    for (uint32_t i = 0; i < 4; ++i) {
        csa_ring_bias[i * 4u + 2u] = i == 0 ? 50.0f : -50.0f;
        csa_ring_bias[i * 4u + 3u] = i == 0 ? 50.0f : -50.0f;
    }
    csa_kv_dev = NULL;
    csa_gate_dev = NULL;
    csa_bias_dev = NULL;
    csa_out_dev = NULL;
    rc = axiom_device_buffer_create(rt, &csa_kv_dev, sizeof(csa_kv));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_gate_dev, sizeof(csa_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_bias_dev, sizeof(csa_ring_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &csa_out_dev, sizeof(csa_ring_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_kv_dev, 0, csa_kv, sizeof(csa_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_gate_dev, 0, csa_gate, sizeof(csa_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(csa_bias_dev, 0, csa_ring_bias, sizeof(csa_ring_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_csa_ring_window_f32_device(
            rt, csa_kv_dev, 0, csa_gate_dev, 0, csa_bias_dev, 0, csa_out_dev, 0, 4, 2, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(csa_out_dev, 0, csa_ring_out, sizeof(csa_ring_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(csa_ring_out[0], 5.0f) || !close_enough(csa_ring_out[1], 6.0f)) {
        fprintf(stderr, "csa ring window failed rc=%d out=%f,%f\n", rc, csa_ring_out[0], csa_ring_out[1]);
        axiom_device_buffer_destroy(csa_out_dev);
        axiom_device_buffer_destroy(csa_bias_dev);
        axiom_device_buffer_destroy(csa_gate_dev);
        axiom_device_buffer_destroy(csa_kv_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    memset(csa_ring_bias, 0, sizeof(csa_ring_bias));
    csa_ring_bias[1u * 4u + 2u] = 50.0f;
    csa_ring_bias[1u * 4u + 3u] = 50.0f;
    rc = axiom_device_buffer_upload(csa_bias_dev, 0, csa_ring_bias, sizeof(csa_ring_bias));
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_csa_ring_window_count_f32_device(
            rt, csa_kv_dev, 0, csa_gate_dev, 0, csa_bias_dev, 0, csa_out_dev, 0, 4, 2, 2, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(csa_out_dev, 0, csa_ring_out, sizeof(csa_ring_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(csa_ring_out[0], 3.0f) || !close_enough(csa_ring_out[1], 4.0f)) {
        fprintf(stderr, "csa ring count window failed rc=%d out=%f,%f\n", rc, csa_ring_out[0], csa_ring_out[1]);
        axiom_device_buffer_destroy(csa_out_dev);
        axiom_device_buffer_destroy(csa_bias_dev);
        axiom_device_buffer_destroy(csa_gate_dev);
        axiom_device_buffer_destroy(csa_kv_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(csa_out_dev);
    axiom_device_buffer_destroy(csa_bias_dev);
    axiom_device_buffer_destroy(csa_gate_dev);
    axiom_device_buffer_destroy(csa_kv_dev);

    float hca_kv[4 * 2] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float hca_gate[4 * 2] = {0.0f};
    float hca_bias[4 * 2] = {0.0f};
    float hca_out[2] = {0.0f, 0.0f};
    axiom_device_buffer *hca_kv_dev = NULL;
    axiom_device_buffer *hca_gate_dev = NULL;
    axiom_device_buffer *hca_bias_dev = NULL;
    axiom_device_buffer *hca_out_dev = NULL;
    rc = axiom_device_buffer_create(rt, &hca_kv_dev, sizeof(hca_kv));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &hca_gate_dev, sizeof(hca_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &hca_bias_dev, sizeof(hca_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &hca_out_dev, sizeof(hca_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(hca_kv_dev, 0, hca_kv, sizeof(hca_kv)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(hca_gate_dev, 0, hca_gate, sizeof(hca_gate)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(hca_bias_dev, 0, hca_bias, sizeof(hca_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_hca_window_f32_device(
            rt, hca_kv_dev, 0, hca_gate_dev, 0, hca_bias_dev, 0, hca_out_dev, 0, 4, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(hca_out_dev, 0, hca_out, sizeof(hca_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(hca_out[0], 4.0f) || !close_enough(hca_out[1], 5.0f)) {
        fprintf(stderr, "hca window failed rc=%d out=%f,%f\n", rc, hca_out[0], hca_out[1]);
        axiom_device_buffer_destroy(hca_out_dev);
        axiom_device_buffer_destroy(hca_bias_dev);
        axiom_device_buffer_destroy(hca_gate_dev);
        axiom_device_buffer_destroy(hca_kv_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    memset(hca_bias, 0, sizeof(hca_bias));
    hca_bias[1u * 2u + 0u] = 50.0f;
    hca_bias[1u * 2u + 1u] = 50.0f;
    rc = axiom_device_buffer_upload(hca_bias_dev, 0, hca_bias, sizeof(hca_bias));
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_hca_ring_window_count_f32_device(
            rt, hca_kv_dev, 0, hca_gate_dev, 0, hca_bias_dev, 0, hca_out_dev, 0, 4, 2, 2, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(hca_out_dev, 0, hca_out, sizeof(hca_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(hca_out[0], 3.0f) || !close_enough(hca_out[1], 4.0f)) {
        fprintf(stderr, "hca ring count window failed rc=%d out=%f,%f\n", rc, hca_out[0], hca_out[1]);
        axiom_device_buffer_destroy(hca_out_dev);
        axiom_device_buffer_destroy(hca_bias_dev);
        axiom_device_buffer_destroy(hca_gate_dev);
        axiom_device_buffer_destroy(hca_kv_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(hca_out_dev);
    axiom_device_buffer_destroy(hca_bias_dev);
    axiom_device_buffer_destroy(hca_gate_dev);
    axiom_device_buffer_destroy(hca_kv_dev);

    uint8_t router_raw[4 * 32 * 2];
    for (uint32_t expert = 0; expert < 4; ++expert) {
        const uint16_t h = expert == 0 ? 0x3c00u : expert == 1 ? 0x4000u : expert == 2 ? 0x4200u : 0x4400u;
        for (uint32_t i = 0; i < 32; ++i) put_f16(router_raw + ((expert * 32u + i) * 2u), h);
    }
    axiom_device_buffer *router_dev = NULL;
    axiom_device_buffer *router_x_dev = NULL;
    axiom_device_buffer *router_idx_dev = NULL;
    axiom_device_buffer *router_w_dev = NULL;
    uint32_t router_idx[2] = {0, 0};
    float router_w[2] = {0.0f, 0.0f};
    rc = axiom_device_buffer_create(rt, &router_dev, sizeof(router_raw));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_x_dev, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_idx_dev, sizeof(router_idx)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_w_dev, sizeof(router_w)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(router_dev, 0, router_raw, sizeof(router_raw)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(router_x_dev, 0, q8_input, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_router_topk_f16_f32_device(
            rt, router_dev, 0, router_x_dev, 0, router_idx_dev, 0, router_w_dev, 0, 4, 32, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(router_idx_dev, 0, router_idx, sizeof(router_idx)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(router_w_dev, 0, router_w, sizeof(router_w)) : rc;
    if (rc != AXIOM_OK || router_idx[0] != 3 || router_idx[1] != 2 ||
        fabsf((router_w[0] + router_w[1]) - 1.5f) > 1.0e-4f) {
        fprintf(stderr, "router resident failed rc=%d idx=%u,%u w=%f,%f\n",
                rc, router_idx[0], router_idx[1], router_w[0], router_w[1]);
        axiom_device_buffer_destroy(router_w_dev);
        axiom_device_buffer_destroy(router_idx_dev);
        axiom_device_buffer_destroy(router_x_dev);
        axiom_device_buffer_destroy(router_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(router_w_dev);
    axiom_device_buffer_destroy(router_idx_dev);
    axiom_device_buffer_destroy(router_x_dev);
    axiom_device_buffer_destroy(router_dev);
    router_w_dev = NULL;
    router_idx_dev = NULL;
    router_x_dev = NULL;
    router_dev = NULL;

    float router_bias[4] = {100.0f, 0.0f, 0.0f, 0.0f};
    rc = axiom_device_buffer_create(rt, &router_dev, sizeof(router_raw));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_x_dev, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_idx_dev, sizeof(router_idx)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_w_dev, sizeof(router_w)) : rc;
    axiom_device_buffer *router_bias_dev = NULL;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &router_bias_dev, sizeof(router_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(router_dev, 0, router_raw, sizeof(router_raw)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(router_x_dev, 0, q8_input, sizeof(q8_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(router_bias_dev, 0, router_bias, sizeof(router_bias)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_router_topk_biased_f16_f32_device(
            rt, router_dev, 0, router_bias_dev, 0, router_x_dev, 0,
            router_idx_dev, 0, router_w_dev, 0, 4, 32, 2) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(router_idx_dev, 0, router_idx, sizeof(router_idx)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(router_w_dev, 0, router_w, sizeof(router_w)) : rc;
    if (rc != AXIOM_OK || router_idx[0] != 0 || router_idx[1] != 3 ||
        fabsf((router_w[0] + router_w[1]) - 1.5f) > 1.0e-4f) {
        fprintf(stderr, "biased router resident failed rc=%d idx=%u,%u w=%f,%f\n",
                rc, router_idx[0], router_idx[1], router_w[0], router_w[1]);
        axiom_device_buffer_destroy(router_bias_dev);
        axiom_device_buffer_destroy(router_w_dev);
        axiom_device_buffer_destroy(router_idx_dev);
        axiom_device_buffer_destroy(router_x_dev);
        axiom_device_buffer_destroy(router_dev);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(router_bias_dev);
    axiom_device_buffer_destroy(router_w_dev);
    axiom_device_buffer_destroy(router_idx_dev);
    axiom_device_buffer_destroy(router_x_dev);
    axiom_device_buffer_destroy(router_dev);

    float k_input[256];
    for (uint32_t i = 0; i < 256; ++i) k_input[i] = 1.0f;
    uint8_t q2[2 * 84];
    memset(q2, 0, sizeof(q2));
    memset(q2, 0x01, 16);
    memset(q2 + 16, 0x55, 64);
    put_f16(q2 + 80, 0x3c00u);
    put_f16(q2 + 82, 0x0000u);
    memset(q2 + 84, 0x01, 16);
    memset(q2 + 84 + 16, 0xaa, 64);
    put_f16(q2 + 84 + 80, 0x3c00u);
    put_f16(q2 + 84 + 82, 0x0000u);
    float q2_out[2] = {0.0f, 0.0f};
    rc = axiom_runtime_q2_k_matvec_f32(rt, q2, k_input, q2_out, 2, 256);
    if (rc != AXIOM_OK || !close_enough(q2_out[0], 256.0f) || !close_enough(q2_out[1], 512.0f)) {
        fprintf(stderr, "q2_k failed rc=%d out=%f,%f\n", rc, q2_out[0], q2_out[1]);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer *q2_w = NULL;
    axiom_device_buffer *q2_x = NULL;
    axiom_device_buffer *q2_y = NULL;
    float q2_dev_out[2] = {0.0f, 0.0f};
    rc = axiom_device_buffer_create(rt, &q2_w, sizeof(q2));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q2_x, sizeof(k_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q2_y, sizeof(q2_dev_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(q2_w, 0, q2, sizeof(q2)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(q2_x, 0, k_input, sizeof(k_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q2_k_matvec_f32_device(rt, q2_w, 0, q2_x, 0, q2_y, 0, 2, 256) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q2_y, 0, q2_dev_out, sizeof(q2_dev_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(q2_dev_out[0], 256.0f) || !close_enough(q2_dev_out[1], 512.0f)) {
        fprintf(stderr, "q2_k resident failed rc=%d out=%f,%f\n", rc, q2_dev_out[0], q2_dev_out[1]);
        axiom_device_buffer_destroy(q2_y);
        axiom_device_buffer_destroy(q2_x);
        axiom_device_buffer_destroy(q2_w);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(q2_y);
    axiom_device_buffer_destroy(q2_x);
    axiom_device_buffer_destroy(q2_w);

    uint8_t iq2[2 * 66];
    memset(iq2, 0, sizeof(iq2));
    put_f16(iq2, 0x3c00u);
    put_f16(iq2 + 66, 0x4000u);
    float iq2_out[2] = {0.0f, 0.0f};
    rc = axiom_runtime_iq2_xxs_matvec_f32(rt, iq2, k_input, iq2_out, 2, 256);
    if (rc != AXIOM_OK || !close_enough(iq2_out[0], 256.0f) || !close_enough(iq2_out[1], 512.0f)) {
        fprintf(stderr, "iq2_xxs failed rc=%d out=%f,%f\n", rc, iq2_out[0], iq2_out[1]);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer *iq2_w = NULL;
    axiom_device_buffer *iq2_x = NULL;
    axiom_device_buffer *iq2_y = NULL;
    float iq2_dev_out[2] = {0.0f, 0.0f};
    rc = axiom_device_buffer_create(rt, &iq2_w, sizeof(iq2));
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &iq2_x, sizeof(k_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &iq2_y, sizeof(iq2_dev_out)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(iq2_w, 0, iq2, sizeof(iq2)) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(iq2_x, 0, k_input, sizeof(k_input)) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_iq2_xxs_matvec_f32_device(rt, iq2_w, 0, iq2_x, 0, iq2_y, 0, 2, 256) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(iq2_y, 0, iq2_dev_out, sizeof(iq2_dev_out)) : rc;
    if (rc != AXIOM_OK || !close_enough(iq2_dev_out[0], 256.0f) || !close_enough(iq2_dev_out[1], 512.0f)) {
        fprintf(stderr, "iq2_xxs resident failed rc=%d out=%f,%f\n", rc, iq2_dev_out[0], iq2_dev_out[1]);
        axiom_device_buffer_destroy(iq2_y);
        axiom_device_buffer_destroy(iq2_x);
        axiom_device_buffer_destroy(iq2_w);
        axiom_runtime_destroy(rt);
        return 1;
    }
    axiom_device_buffer_destroy(iq2_y);
    axiom_device_buffer_destroy(iq2_x);
    axiom_device_buffer_destroy(iq2_w);

    axiom_runtime_destroy(rt);
    printf("axiom-quant-smoke: OK q8=%g,%g q2=%g,%g iq2=%g,%g rmsnorm=pass multi_kv=pass sliding_ring=pass f16=pass csa=pass csa_ring=pass hca=pass router=pass biased_router=pass resident=pass\n",
            q8_out[0], q8_out[1], q2_out[0], q2_out[1], iq2_out[0], iq2_out[1]);
    return 0;
}
