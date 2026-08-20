#define _POSIX_C_SOURCE 200809L

#include "axiom/axiom.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define IQ2_XXS_BLOCK_BYTES 66u
#define QK_K 256u

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int parse_u32(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v == 0 || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int parse_u32_zero_ok(const char *s, uint32_t *out) {
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT32_MAX) return 0;
    *out = (uint32_t)v;
    return 1;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median_double(double *values, uint32_t n) {
    qsort(values, n, sizeof(double), cmp_double);
    return (n & 1u) ? values[n / 2u] : (values[n / 2u - 1u] + values[n / 2u]) * 0.5;
}

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8u);
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)((v >> 8u) & 0xffu);
    p[2] = (uint8_t)((v >> 16u) & 0xffu);
    p[3] = (uint8_t)(v >> 24u);
}

static void fill_iq2(uint8_t *buf, uint32_t experts, uint32_t rows, uint32_t cols, uint32_t salt) {
    const uint64_t blocks = cols / QK_K;
    const uint64_t row_bytes = blocks * IQ2_XXS_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    for (uint32_t e = 0; e < experts; ++e) {
        uint8_t *expert = buf + (uint64_t)e * expert_bytes;
        for (uint32_t r = 0; r < rows; ++r) {
            uint8_t *row = expert + (uint64_t)r * row_bytes;
            for (uint32_t b = 0; b < blocks; ++b) {
                uint8_t *blk = row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES;
                put_f16(blk, 0x3c00u);
                for (uint32_t g = 0; g < 8u; ++g) {
                    const uint32_t base = e * 131u + r * 17u + b * 29u + g * 37u + salt;
                    const uint32_t aux_g =
                            ((base + 3u) & 255u) |
                            (((base + 19u) & 255u) << 8u) |
                            (((base + 47u) & 255u) << 16u) |
                            (((base + 101u) & 255u) << 24u);
                    const uint32_t s0 = (base + 5u) & 127u;
                    const uint32_t s1 = (base + 11u) & 127u;
                    const uint32_t s2 = (base + 23u) & 127u;
                    const uint32_t s3 = (base + 41u) & 127u;
                    const uint32_t scale = (base >> 3u) & 7u;
                    const uint32_t aux_s = s0 | (s1 << 7u) | (s2 << 14u) | (s3 << 21u) | (scale << 28u);
                    put_u32(blk + 2u + g * 8u, aux_g);
                    put_u32(blk + 2u + g * 8u + 4u, aux_s);
                }
            }
        }
    }
}

static void fill_input(float *x, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        const int a = (int)(i % 67u) - 33;
        const int b = (int)((i * 17u + 5u) % 19u) - 9;
        x[i] = (float)a * 0.0015f + (float)b * 0.0002f;
    }
}

static void compare_outputs(
        const float *base,
        const float *cand,
        uint32_t n,
        double *mean_abs,
        double *rmse,
        double *rel_l2,
        double *cosine,
        float *max_abs,
        uint32_t *max_idx) {
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double base_sq = 0.0;
    double cand_sq = 0.0;
    double dot = 0.0;
    *max_abs = 0.0f;
    *max_idx = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = (double)cand[i] - (double)base[i];
        const double ad = fabs(d);
        if (ad > (double)*max_abs) {
            *max_abs = (float)ad;
            *max_idx = i;
        }
        sum_abs += ad;
        sum_sq += d * d;
        base_sq += (double)base[i] * (double)base[i];
        cand_sq += (double)cand[i] * (double)cand[i];
        dot += (double)base[i] * (double)cand[i];
    }
    *mean_abs = sum_abs / (double)n;
    *rmse = sqrt(sum_sq / (double)n);
    *rel_l2 = base_sq > 0.0 ? sqrt(sum_sq / base_sq) : 0.0;
    *cosine = (base_sq > 0.0 && cand_sq > 0.0) ? dot / (sqrt(base_sq) * sqrt(cand_sq)) : 0.0;
}

int main(int argc, char **argv) {
    uint32_t experts = 256;
    uint32_t topk = 6;
    uint32_t hidden = 4096;
    uint32_t expert_hidden = 2048;
    uint32_t reps = 100;
    uint32_t warmup = 5;
    uint32_t samples = 5;
    uint32_t reps_per_sample = 0;
    if (argc > 1 && !parse_u32(argv[1], &experts)) return 2;
    if (argc > 2 && !parse_u32(argv[2], &topk)) return 2;
    if (argc > 3 && !parse_u32(argv[3], &hidden)) return 2;
    if (argc > 4 && !parse_u32(argv[4], &expert_hidden)) return 2;
    if (argc > 5 && !parse_u32(argv[5], &reps)) return 2;
    if (argc > 6 && !parse_u32_zero_ok(argv[6], &warmup)) return 2;
    if (argc > 7 && !parse_u32(argv[7], &samples)) return 2;
    if (argc > 8 && !parse_u32(argv[8], &reps_per_sample)) return 2;
    if (argc > 9 || topk > experts || topk > 16u ||
        (hidden % QK_K) != 0 || (expert_hidden % QK_K) != 0 || samples == 0) {
        return 2;
    }
    if (reps_per_sample == 0) reps_per_sample = reps;

    const uint64_t blocks = hidden / QK_K;
    const uint64_t row_bytes = blocks * IQ2_XXS_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)expert_hidden * row_bytes;
    const uint64_t weight_bytes = (uint64_t)experts * expert_bytes;
    const uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    const uint64_t mid_count = (uint64_t)topk * expert_hidden;
    const uint64_t mid_bytes = mid_count * sizeof(float);
    const uint64_t q8k_bytes = blocks * AXIOM_Q8_K_BLOCK_BYTES;

    uint8_t *gate = (uint8_t *)malloc((size_t)weight_bytes);
    uint8_t *up = (uint8_t *)malloc((size_t)weight_bytes);
    float *input = (float *)malloc((size_t)input_bytes);
    uint32_t *indices = (uint32_t *)malloc((size_t)indices_bytes);
    float *base = (float *)malloc((size_t)mid_bytes);
    float *cand = (float *)malloc((size_t)mid_bytes);
    double *base_samples = (double *)malloc((size_t)samples * sizeof(double));
    double *cand_samples = (double *)malloc((size_t)samples * sizeof(double));
    if (!gate || !up || !input || !indices || !base || !cand || !base_samples || !cand_samples) return 1;

    fill_iq2(gate, experts, expert_hidden, hidden, 7u);
    fill_iq2(up, experts, expert_hidden, hidden, 113u);
    fill_input(input, hidden);
    for (uint32_t i = 0; i < topk; ++i) indices[i] = i;

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    axiom_runtime *rt = NULL;
    axiom_device_buffer *gate_dev = NULL;
    axiom_device_buffer *up_dev = NULL;
    axiom_device_buffer *input_dev = NULL;
    axiom_device_buffer *indices_dev = NULL;
    axiom_device_buffer *base_dev = NULL;
    axiom_device_buffer *cand_dev = NULL;
    axiom_device_buffer *q8k_dev = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &gate_dev, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &up_dev, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &input_dev, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &indices_dev, indices_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &base_dev, mid_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &cand_dev, mid_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &q8k_dev, q8k_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(gate_dev, 0, gate, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(up_dev, 0, up, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(input_dev, 0, input, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(indices_dev, 0, indices, indices_bytes) : rc;
    if (rc != AXIOM_OK) {
        fprintf(stderr, "setup failed rc=%d %s\n", rc, axiom_status_string(rc));
        return 1;
    }

    for (uint32_t i = 0; i < warmup; ++i) {
        rc = axiom_runtime_deepseek_moe_gate_up_indexed_f32_scratch_device(
                rt, gate_dev, 0, up_dev, 0, input_dev, 0, indices_dev, 0,
                base_dev, 0, experts, topk, hidden, expert_hidden);
        rc = rc == AXIOM_OK ? axiom_runtime_deepseek_moe_gate_up_indexed_q8k_scratch_device(
                rt, gate_dev, 0, up_dev, 0, input_dev, 0, indices_dev, 0,
                q8k_dev, 0, cand_dev, 0, experts, topk, hidden, expert_hidden) : rc;
        if (rc != AXIOM_OK) return 1;
    }

    for (uint32_t s = 0; s < samples; ++s) {
        double t0 = now_ms();
        for (uint32_t i = 0; i < reps_per_sample; ++i) {
            rc = axiom_runtime_deepseek_moe_gate_up_indexed_f32_scratch_device(
                    rt, gate_dev, 0, up_dev, 0, input_dev, 0, indices_dev, 0,
                    base_dev, 0, experts, topk, hidden, expert_hidden);
            if (rc != AXIOM_OK) return 1;
        }
        double t1 = now_ms();
        base_samples[s] = ((double)reps_per_sample * 1000.0) / (t1 - t0);

        t0 = now_ms();
        for (uint32_t i = 0; i < reps_per_sample; ++i) {
            rc = axiom_runtime_deepseek_moe_gate_up_indexed_q8k_scratch_device(
                    rt, gate_dev, 0, up_dev, 0, input_dev, 0, indices_dev, 0,
                    q8k_dev, 0, cand_dev, 0, experts, topk, hidden, expert_hidden);
            if (rc != AXIOM_OK) return 1;
        }
        t1 = now_ms();
        cand_samples[s] = ((double)reps_per_sample * 1000.0) / (t1 - t0);
    }

    rc = axiom_device_buffer_download(base_dev, 0, base, mid_bytes);
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(cand_dev, 0, cand, mid_bytes) : rc;
    if (rc != AXIOM_OK) return 1;

    double mean_abs = 0.0;
    double rmse = 0.0;
    double rel_l2 = 0.0;
    double cosine = 0.0;
    float max_abs = 0.0f;
    uint32_t max_idx = 0;
    compare_outputs(base, cand, (uint32_t)mid_count, &mean_abs, &rmse, &rel_l2, &cosine, &max_abs, &max_idx);
    const double base_med = median_double(base_samples, samples);
    const double cand_med = median_double(cand_samples, samples);
    const int ok = isfinite(cosine) && rel_l2 <= 0.01 && cosine >= 0.9999;

    printf("moe_gateup_q8k_bench experts=%u topk=%u hidden=%u expert_hidden=%u reps=%u warmup=%u samples=%u reps_per_sample=%u f32_iter_s=%.3f q8k_iter_s=%.3f speedup=%.3f max_abs=%.9f mean_abs=%.9f rmse=%.9f rel_l2=%.9f cosine=%.9f max_idx=%u base=%.9f cand=%.9f status=%s\n",
            experts, topk, hidden, expert_hidden, reps, warmup, samples, reps_per_sample,
            base_med, cand_med, base_med > 0.0 ? cand_med / base_med : 0.0,
            max_abs, mean_abs, rmse, rel_l2, cosine, max_idx, base[max_idx], cand[max_idx],
            ok ? "pass" : "warn");

    axiom_device_buffer_destroy(q8k_dev);
    axiom_device_buffer_destroy(cand_dev);
    axiom_device_buffer_destroy(base_dev);
    axiom_device_buffer_destroy(indices_dev);
    axiom_device_buffer_destroy(input_dev);
    axiom_device_buffer_destroy(up_dev);
    axiom_device_buffer_destroy(gate_dev);
    axiom_runtime_destroy(rt);
    free(cand_samples);
    free(base_samples);
    free(cand);
    free(base);
    free(indices);
    free(input);
    free(up);
    free(gate);
    return ok ? 0 : 3;
}
