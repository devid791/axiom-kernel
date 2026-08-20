#define _POSIX_C_SOURCE 200809L

#include "axiom/axiom.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8u);
}

static float input_value(uint32_t i) {
    const int32_t v = (int32_t)((i * 41u + 17u) % 251u) - 125;
    return (float)v / 13.0f;
}

static float residual_value(uint32_t i) {
    const int32_t v = (int32_t)((i * 29u + 7u) % 191u) - 95;
    return (float)v / 17.0f;
}

static int8_t weight_value(uint32_t row, uint32_t col, uint32_t salt) {
    const int32_t v = (int32_t)((row * 19u + col * 31u + salt * 47u + 13u) % 255u) - 127;
    return (int8_t)v;
}

static void fill_q8_0(uint8_t *weight, uint32_t rows, uint32_t cols, uint32_t salt) {
    const uint32_t blocks = cols / 32u;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *blk = weight + ((uint64_t)r * blocks + b) * 34u;
            put_u16(blk, 0x3c00u);
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                blk[2u + lane] = (uint8_t)weight_value(r, b * 32u + lane, salt);
            }
        }
    }
}

static int byte_equal(const float *a, const float *b, uint32_t n, uint32_t *first_bad) {
    if (memcmp(a, b, (size_t)n * sizeof(float)) == 0) return 1;
    for (uint32_t i = 0; i < n; ++i) {
        if (memcmp(a + i, b + i, sizeof(float)) != 0) {
            *first_bad = i;
            return 0;
        }
    }
    *first_bad = n;
    return 0;
}

static uint32_t f32_bits(float v) {
    uint32_t bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return bits;
}

int main(int argc, char **argv) {
    uint32_t rows_q = 257u;
    uint32_t rows_k = 131u;
    uint32_t rows_v = 133u;
    uint32_t rows_dual = 257u;
    uint32_t cols = 4096u;
    if (argc > 1) rows_q = (uint32_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) rows_k = (uint32_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) rows_v = (uint32_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) rows_dual = (uint32_t)strtoul(argv[4], NULL, 10);
    if (argc > 5) cols = (uint32_t)strtoul(argv[5], NULL, 10);
    if (argc > 6 || rows_q == 0 || rows_k == 0 || rows_v == 0 ||
        rows_dual == 0 || cols == 0 || (cols % 32u) != 0u) {
        fprintf(stderr, "usage: axiom-q8-pack-reuse-smoke [rows_q rows_k rows_v rows_dual cols]\n");
        return 1;
    }

    setenv("AXIOM_DS4_Q8_DP4A", "0", 1);
    setenv("AXIOM_DS4_Q8_PREQ_DISABLE", "0", 1);
    setenv("AXIOM_DS4_Q8_PREQ_SXQ", "0", 1);

    const uint32_t blocks = cols / 32u;
    const uint64_t x_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t wq_bytes = (uint64_t)rows_q * blocks * 34u;
    const uint64_t wk_bytes = (uint64_t)rows_k * blocks * 34u;
    const uint64_t wv_bytes = (uint64_t)rows_v * blocks * 34u;
    const uint64_t wd_bytes = (uint64_t)rows_dual * blocks * 34u;
    const uint64_t oq_bytes = (uint64_t)rows_q * sizeof(float);
    const uint64_t ok_bytes = (uint64_t)rows_k * sizeof(float);
    const uint64_t ov_bytes = (uint64_t)rows_v * sizeof(float);
    const uint64_t od_bytes = (uint64_t)rows_dual * sizeof(float);

    float *x = (float *)malloc((size_t)x_bytes);
    uint8_t *wq = (uint8_t *)malloc((size_t)wq_bytes);
    uint8_t *wk = (uint8_t *)malloc((size_t)wk_bytes);
    uint8_t *wv = (uint8_t *)malloc((size_t)wv_bytes);
    uint8_t *wa = (uint8_t *)malloc((size_t)wd_bytes);
    uint8_t *wb = (uint8_t *)malloc((size_t)wd_bytes);
    float *q_ref = (float *)malloc((size_t)oq_bytes);
    float *k_ref = (float *)malloc((size_t)ok_bytes);
    float *v_ref = (float *)malloc((size_t)ov_bytes);
    float *q_got = (float *)malloc((size_t)oq_bytes);
    float *k_got = (float *)malloc((size_t)ok_bytes);
    float *v_got = (float *)malloc((size_t)ov_bytes);
    float *a_ref = (float *)malloc((size_t)od_bytes);
    float *b_ref = (float *)malloc((size_t)od_bytes);
    float *a_got = (float *)malloc((size_t)od_bytes);
    float *b_got = (float *)malloc((size_t)od_bytes);
    float *residual = (float *)malloc((size_t)oq_bytes);
    float *add_ref = (float *)malloc((size_t)oq_bytes);
    float *add_got = (float *)malloc((size_t)oq_bytes);
    float *mid_ref = (float *)malloc((size_t)od_bytes);
    float *mid_got = (float *)malloc((size_t)od_bytes);
    if (!x || !wq || !wk || !wv || !wa || !wb || !q_ref || !k_ref || !v_ref ||
        !q_got || !k_got || !v_got || !a_ref || !b_ref || !a_got || !b_got ||
        !residual || !add_ref || !add_got || !mid_ref || !mid_got) {
        fprintf(stderr, "alloc_fail\n");
        return 1;
    }
    for (uint32_t i = 0; i < cols; ++i) x[i] = input_value(i);
    for (uint32_t i = 0; i < rows_q; ++i) residual[i] = residual_value(i);
    fill_q8_0(wq, rows_q, cols, 1u);
    fill_q8_0(wk, rows_k, cols, 2u);
    fill_q8_0(wv, rows_v, cols, 3u);
    fill_q8_0(wa, rows_dual, cols, 4u);
    fill_q8_0(wb, rows_dual, cols, 5u);

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;

    axiom_runtime *rt = NULL;
    axiom_device_buffer *x_dev = NULL;
    axiom_device_buffer *wq_dev = NULL;
    axiom_device_buffer *wk_dev = NULL;
    axiom_device_buffer *wv_dev = NULL;
    axiom_device_buffer *wa_dev = NULL;
    axiom_device_buffer *wb_dev = NULL;
    axiom_device_buffer *q_dev = NULL;
    axiom_device_buffer *k_dev = NULL;
    axiom_device_buffer *v_dev = NULL;
    axiom_device_buffer *a_dev = NULL;
    axiom_device_buffer *b_dev = NULL;
    axiom_device_buffer *r_dev = NULL;

    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &x_dev, x_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &wq_dev, wq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &wk_dev, wk_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &wv_dev, wv_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &wa_dev, wd_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &wb_dev, wd_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q_dev, oq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &k_dev, ok_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &v_dev, ov_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &a_dev, od_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &b_dev, od_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &r_dev, oq_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(x_dev, 0, x, x_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(wq_dev, 0, wq, wq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(wk_dev, 0, wk, wk_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(wv_dev, 0, wv, wv_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(wa_dev, 0, wa, wd_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(wb_dev, 0, wb, wd_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(r_dev, 0, residual, oq_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wq_dev, 0, x_dev, 0, q_dev, 0, rows_q, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q_dev, 0, q_ref, oq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wk_dev, 0, x_dev, 0, k_dev, 0, rows_k, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(k_dev, 0, k_ref, ok_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wv_dev, 0, x_dev, 0, v_dev, 0, rows_v, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(v_dev, 0, v_ref, ov_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wa_dev, 0, x_dev, 0, a_dev, 0, rows_dual, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(a_dev, 0, a_ref, od_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wb_dev, 0, x_dev, 0, b_dev, 0, rows_dual, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(b_dev, 0, b_ref, od_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_qkv_matvec_f32_device(
            rt, wq_dev, 0, wk_dev, 0, wv_dev, 0, x_dev, 0,
            q_dev, 0, k_dev, 0, v_dev, 0, rows_q, rows_k, rows_v, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q_dev, 0, q_got, oq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(k_dev, 0, k_got, ok_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(v_dev, 0, v_got, ov_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_dual_matvec_f32_device(
            rt, wa_dev, 0, wb_dev, 0, x_dev, 0,
            a_dev, 0, b_dev, 0, rows_dual, rows_dual, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(a_dev, 0, a_got, od_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(b_dev, 0, b_got, od_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(rt, wq_dev, 0, x_dev, 0, q_dev, 0, rows_q, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_add_f32_device(rt, r_dev, 0, q_dev, 0, q_dev, 0, rows_q) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(q_dev, 0, add_ref, oq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(r_dev, 0, residual, oq_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_add_f32_device(
            rt, wq_dev, 0, x_dev, 0, r_dev, 0, r_dev, 0, rows_q, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(r_dev, 0, add_got, oq_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_dual_matvec_f32_device(
            rt, wa_dev, 0, wb_dev, 0, x_dev, 0,
            a_dev, 0, b_dev, 0, rows_dual, rows_dual, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_silu_mul_f32_device(rt, a_dev, 0, b_dev, 0, a_dev, 0, rows_dual) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(a_dev, 0, mid_ref, od_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_dual_matvec_silu_f32_device(
            rt, wa_dev, 0, wb_dev, 0, x_dev, 0, a_dev, 0, rows_dual, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(a_dev, 0, mid_got, od_bytes) : rc;

    uint32_t bad_q = 0, bad_k = 0, bad_v = 0, bad_a = 0, bad_b = 0, bad_add = 0, bad_mid = 0;
    const int q_ok = rc == AXIOM_OK && byte_equal(q_ref, q_got, rows_q, &bad_q);
    const int k_ok = rc == AXIOM_OK && byte_equal(k_ref, k_got, rows_k, &bad_k);
    const int v_ok = rc == AXIOM_OK && byte_equal(v_ref, v_got, rows_v, &bad_v);
    const int a_ok = rc == AXIOM_OK && byte_equal(a_ref, a_got, rows_dual, &bad_a);
    const int b_ok = rc == AXIOM_OK && byte_equal(b_ref, b_got, rows_dual, &bad_b);
    const int add_ok = rc == AXIOM_OK && byte_equal(add_ref, add_got, rows_q, &bad_add);
    const int mid_ok = rc == AXIOM_OK && byte_equal(mid_ref, mid_got, rows_dual, &bad_mid);
    const int pass = q_ok && k_ok && v_ok && a_ok && b_ok && add_ok && mid_ok;

    printf("{\"status\":\"%s\",\"rc\":%d,\"rows_q\":%u,\"rows_k\":%u,"
           "\"rows_v\":%u,\"rows_dual\":%u,\"cols\":%u,"
           "\"q_ok\":%d,\"k_ok\":%d,\"v_ok\":%d,\"dual_a_ok\":%d,\"dual_b_ok\":%d,"
           "\"matvec_add_ok\":%d,\"dual_silu_ok\":%d",
           pass ? "pass" : "fail", rc, rows_q, rows_k, rows_v, rows_dual, cols,
           q_ok, k_ok, v_ok, a_ok, b_ok, add_ok, mid_ok);
    if (!pass && rc == AXIOM_OK) {
        const float *ref = !q_ok ? q_ref : (!k_ok ? k_ref : (!v_ok ? v_ref : (!a_ok ? a_ref : (!b_ok ? b_ref : (!add_ok ? add_ref : mid_ref)))));
        const float *got = !q_ok ? q_got : (!k_ok ? k_got : (!v_ok ? v_got : (!a_ok ? a_got : (!b_ok ? b_got : (!add_ok ? add_got : mid_got)))));
        const uint32_t bad = !q_ok ? bad_q : (!k_ok ? bad_k : (!v_ok ? bad_v : (!a_ok ? bad_a : (!b_ok ? bad_b : (!add_ok ? bad_add : bad_mid)))));
        printf(",\"first_bad\":%u,\"ref_bits\":\"0x%08x\",\"got_bits\":\"0x%08x\"",
               bad, f32_bits(ref[bad]), f32_bits(got[bad]));
    }
    printf("}\n");

    axiom_device_buffer_destroy(r_dev);
    axiom_device_buffer_destroy(b_dev);
    axiom_device_buffer_destroy(a_dev);
    axiom_device_buffer_destroy(v_dev);
    axiom_device_buffer_destroy(k_dev);
    axiom_device_buffer_destroy(q_dev);
    axiom_device_buffer_destroy(wb_dev);
    axiom_device_buffer_destroy(wa_dev);
    axiom_device_buffer_destroy(wv_dev);
    axiom_device_buffer_destroy(wk_dev);
    axiom_device_buffer_destroy(wq_dev);
    axiom_device_buffer_destroy(x_dev);
    axiom_runtime_destroy(rt);
    free(mid_got);
    free(mid_ref);
    free(add_got);
    free(add_ref);
    free(residual);
    free(b_got);
    free(a_got);
    free(b_ref);
    free(a_ref);
    free(v_got);
    free(k_got);
    free(q_got);
    free(v_ref);
    free(k_ref);
    free(q_ref);
    free(wb);
    free(wa);
    free(wv);
    free(wk);
    free(wq);
    free(x);
    return pass ? 0 : 1;
}
