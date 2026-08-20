#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float value_at(uint32_t i, uint32_t salt) {
    return 0.25f * sinf((float)(i + 1u + salt) * 0.071f) +
           0.13f * cosf((float)(i + 3u * salt) * 0.037f);
}

static void fill(float *x, uint32_t n, uint32_t salt) {
    for (uint32_t i = 0; i < n; i++) x[i] = value_at(i, salt);
}

static void metrics(
        const float *a,
        const float *b,
        uint32_t n,
        double *max_abs,
        double *cos_sim,
        uint64_t *nonfinite) {
    double dot = 0.0, n1 = 0.0, n2 = 0.0, ma = 0.0;
    uint64_t nf = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) {
            nf++;
            continue;
        }
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > ma) ma = d;
        dot += (double)a[i] * (double)b[i];
        n1 += (double)a[i] * (double)a[i];
        n2 += (double)b[i] * (double)b[i];
    }
    *max_abs = ma;
    *cos_sim = (n1 > 0.0 && n2 > 0.0) ? dot / (sqrt(n1) * sqrt(n2)) : 0.0;
    *nonfinite = nf;
}

static int run_case(
        axiom_runtime *rt,
        uint32_t heads,
        uint32_t head_dim,
        uint32_t ring_slots,
        uint32_t ring_head,
        uint32_t ring_count,
        const char **out_stage,
        double *out_max_abs0,
        double *out_cos0,
        double *out_max_abs1,
        double *out_cos1,
        uint64_t *out_nonfinite) {
    const char *stage = "alloc_host";
    if (out_stage) *out_stage = stage;
    const uint32_t q_count = heads * head_dim;
    const uint32_t ring_count_f = ring_slots * head_dim;
    const uint64_t q_bytes = (uint64_t)q_count * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)head_dim * sizeof(float);
    const uint64_t ring_bytes = (uint64_t)ring_count_f * sizeof(float);
    const uint64_t sink_bytes = (uint64_t)heads * sizeof(float);
    float *q0 = (float *)malloc((size_t)q_bytes);
    float *q1 = (float *)malloc((size_t)q_bytes);
    float *kv0 = (float *)malloc((size_t)kv_bytes);
    float *kv1 = (float *)malloc((size_t)kv_bytes);
    float *ring = (float *)malloc((size_t)ring_bytes);
    float *ring_after = (float *)malloc((size_t)ring_bytes);
    float *sink = (float *)malloc((size_t)sink_bytes);
    float *ref0 = (float *)calloc(q_count, sizeof(float));
    float *ref1 = (float *)calloc(q_count, sizeof(float));
    float *got0 = (float *)calloc(q_count, sizeof(float));
    float *got1 = (float *)calloc(q_count, sizeof(float));
    if (!q0 || !q1 || !kv0 || !kv1 || !ring || !ring_after || !sink ||
        !ref0 || !ref1 || !got0 || !got1) {
        free(got1); free(got0); free(ref1); free(ref0); free(sink);
        free(ring_after); free(ring); free(kv1); free(kv0); free(q1); free(q0);
        return AXIOM_ERR_RUNTIME;
    }
    fill(q0, q_count, 1u + ring_count);
    fill(q1, q_count, 7u + ring_count);
    fill(kv0, head_dim, 13u + ring_count);
    fill(kv1, head_dim, 19u + ring_count);
    fill(ring, ring_count_f, 23u + ring_count);
    fill(sink, heads, 29u + ring_count);
    memcpy(ring_after, ring, (size_t)ring_bytes);
    memcpy(ring_after + (uint64_t)ring_head * head_dim, kv0, (size_t)kv_bytes);
    const uint32_t ring_head1 = (ring_head + 1u) % ring_slots;
    const uint32_t ring_count1 = ring_count + 1u > ring_slots ? ring_slots : ring_count + 1u;

    axiom_device_buffer *dq0 = NULL, *dq1 = NULL, *dkv0 = NULL, *dkv1 = NULL;
    axiom_device_buffer *dring = NULL, *dring_after = NULL, *dsink = NULL;
    axiom_device_buffer *dref0 = NULL, *dref1 = NULL, *dgot0 = NULL, *dgot1 = NULL;
    int rc = AXIOM_OK;
#define CREATE(buf, bytes) do { rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &(buf), (bytes)) : rc; } while (0)
    stage = "create_buffers"; if (out_stage) *out_stage = stage;
    CREATE(dq0, q_bytes); CREATE(dq1, q_bytes);
    CREATE(dkv0, kv_bytes); CREATE(dkv1, kv_bytes);
    CREATE(dring, ring_bytes); CREATE(dring_after, ring_bytes);
    CREATE(dsink, sink_bytes);
    CREATE(dref0, q_bytes); CREATE(dref1, q_bytes);
    CREATE(dgot0, q_bytes); CREATE(dgot1, q_bytes);
#undef CREATE
    stage = "upload_q0"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dq0, 0, q0, q_bytes) : rc;
    stage = "upload_q1"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dq1, 0, q1, q_bytes) : rc;
    stage = "upload_kv0"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dkv0, 0, kv0, kv_bytes) : rc;
    stage = "upload_kv1"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dkv1, 0, kv1, kv_bytes) : rc;
    stage = "upload_ring"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dring, 0, ring, ring_bytes) : rc;
    stage = "upload_ring_after"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dring_after, 0, ring_after, ring_bytes) : rc;
    stage = "upload_sink"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(dsink, 0, sink, sink_bytes) : rc;
    stage = "serial_ref0"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_sliding_attention_ring_f32_device(
            rt, dq0, 0, dkv0, 0, dring, 0, dsink, 0, dref0, 0,
            heads, head_dim, ring_slots, ring_head, ring_count, 1.0e-6f) : rc;
    stage = "serial_ref1"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_sliding_attention_ring_f32_device(
            rt, dq1, 0, dkv1, 0, dring_after, 0, dsink, 0, dref1, 0,
            heads, head_dim, ring_slots, ring_head1, ring_count1, 1.0e-6f) : rc;
    stage = "ring2_causal"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_sliding_attention_ring2_causal_f32_device(
            rt, dq0, 0, dq1, 0, dkv0, 0, dkv1, 0, dring, 0, dsink, 0,
            dgot0, 0, dgot1, 0, heads, head_dim, ring_slots, ring_head,
            ring_count, 1.0e-6f) : rc;
    stage = "download_ref0"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(dref0, 0, ref0, q_bytes) : rc;
    stage = "download_ref1"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(dref1, 0, ref1, q_bytes) : rc;
    stage = "download_got0"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(dgot0, 0, got0, q_bytes) : rc;
    stage = "download_got1"; if (out_stage) *out_stage = stage;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(dgot1, 0, got1, q_bytes) : rc;
    if (rc == AXIOM_OK) {
        uint64_t nf0 = 0, nf1 = 0;
        metrics(ref0, got0, q_count, out_max_abs0, out_cos0, &nf0);
        metrics(ref1, got1, q_count, out_max_abs1, out_cos1, &nf1);
        *out_nonfinite = nf0 + nf1;
    }
    axiom_device_buffer_destroy(dgot1); axiom_device_buffer_destroy(dgot0);
    axiom_device_buffer_destroy(dref1); axiom_device_buffer_destroy(dref0);
    axiom_device_buffer_destroy(dsink); axiom_device_buffer_destroy(dring_after);
    axiom_device_buffer_destroy(dring); axiom_device_buffer_destroy(dkv1);
    axiom_device_buffer_destroy(dkv0); axiom_device_buffer_destroy(dq1);
    axiom_device_buffer_destroy(dq0);
    free(got1); free(got0); free(ref1); free(ref0); free(sink);
    free(ring_after); free(ring); free(kv1); free(kv0); free(q1); free(q0);
    return rc;
}

int main(void) {
    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;
    axiom_runtime *rt = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    const uint32_t heads = 4u;
    const uint32_t head_dim = 32u;
    const uint32_t slots = 128u;
    struct {
        uint32_t head;
        uint32_t count;
    } cases[] = {
        {0u, 0u},
        {7u, 7u},
        {37u, 128u},
    };
    double worst_abs = 0.0;
    double worst_cos = 1.0;
    uint64_t nonfinite = 0;
    uint32_t pass_count = 0;
    const char *stage = "init";
    for (uint32_t i = 0; rc == AXIOM_OK && i < (uint32_t)(sizeof(cases) / sizeof(cases[0])); i++) {
        double a0 = 0.0, c0 = 0.0, a1 = 0.0, c1 = 0.0;
        uint64_t nf = 0;
        rc = run_case(rt, heads, head_dim, slots, cases[i].head, cases[i].count,
                &stage, &a0, &c0, &a1, &c1, &nf);
        if (rc == AXIOM_OK) {
            if (a0 > worst_abs) worst_abs = a0;
            if (a1 > worst_abs) worst_abs = a1;
            if (c0 < worst_cos) worst_cos = c0;
            if (c1 < worst_cos) worst_cos = c1;
            nonfinite += nf;
            if (a0 < 1.0e-7 && a1 < 1.0e-7 && c0 > 0.999999999 && c1 > 0.999999999 && nf == 0) {
                pass_count++;
            }
        }
    }
    const int pass = rc == AXIOM_OK && pass_count == (uint32_t)(sizeof(cases) / sizeof(cases[0]));
    printf("{\"status\":\"%s\",\"rc\":%d,\"cases\":%u,\"pass_count\":%u,"
           "\"stage\":\"%s\",\"max_abs\":%.9g,\"cos_sim\":%.12f,\"nonfinite\":%llu,"
           "\"primitive\":\"sliding_attention_ring2_causal\"}\n",
           pass ? "pass" : "fail",
           rc,
           (uint32_t)(sizeof(cases) / sizeof(cases[0])),
           pass_count,
           stage,
           worst_abs,
           worst_cos,
           (unsigned long long)nonfinite);
    axiom_runtime_destroy(rt);
    return pass ? 0 : 1;
}
