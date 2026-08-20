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
#define Q2_K_BLOCK_BYTES 84u
#define QK_K 256u

static void put_f16(uint8_t *p, uint16_t h) {
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8u);
}

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

static double median_double(const double *values, uint32_t n) {
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return 0.0;
    memcpy(sorted, values, (size_t)n * sizeof(double));
    qsort(sorted, n, sizeof(double), cmp_double);
    const double median = (n & 1u) != 0
        ? sorted[n / 2u]
        : (sorted[n / 2u - 1u] + sorted[n / 2u]) * 0.5;
    free(sorted);
    return median;
}

static void print_result(
        uint32_t experts,
        uint32_t topk,
        uint32_t hidden,
        uint32_t expert_hidden,
        uint32_t reps,
        uint32_t warmup,
        uint32_t samples,
        uint32_t reps_per_sample,
        double elapsed_ms,
        double iter_s_median,
        double iter_s_best,
        double iter_s_min,
        double iter_s_max,
        float first,
        float expected,
        float max_abs,
        int ok) {
    printf("moe_indexed_bench experts=%u topk=%u hidden=%u expert_hidden=%u reps=%u warmup=%u samples=%u reps_per_sample=%u elapsed_ms=%.3f iter_s=%.3f iter_s_median=%.3f iter_s_best=%.3f iter_s_min=%.3f iter_s_max=%.3f first=%.6f expected=%.6f max_abs=%.6f status=%s\n",
            experts,
            topk,
            hidden,
            expert_hidden,
            reps,
            warmup,
            samples,
            reps_per_sample,
            elapsed_ms,
            iter_s_median,
            iter_s_median,
            iter_s_best,
            iter_s_min,
            iter_s_max,
            first,
            expected,
            max_abs,
            ok ? "ok" : "fail");
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > UINT64_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static void fill_iq2(uint8_t *buf, uint32_t experts, uint32_t rows, uint32_t cols) {
    const uint64_t blocks = cols / QK_K;
    const uint64_t row_bytes = blocks * IQ2_XXS_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    memset(buf, 0, (size_t)experts * (size_t)expert_bytes);
    for (uint32_t e = 0; e < experts; e++) {
        uint8_t *expert = buf + (uint64_t)e * expert_bytes;
        for (uint32_t r = 0; r < rows; r++) {
            uint8_t *row = expert + (uint64_t)r * row_bytes;
            for (uint32_t b = 0; b < blocks; b++) {
                put_f16(row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES, 0x3c00u);
            }
        }
    }
}

static void fill_q2(uint8_t *buf, uint32_t experts, uint32_t rows, uint32_t cols) {
    const uint64_t blocks = cols / QK_K;
    const uint64_t row_bytes = blocks * Q2_K_BLOCK_BYTES;
    const uint64_t expert_bytes = (uint64_t)rows * row_bytes;
    memset(buf, 0, (size_t)experts * (size_t)expert_bytes);
    for (uint32_t e = 0; e < experts; e++) {
        uint8_t *expert = buf + (uint64_t)e * expert_bytes;
        for (uint32_t r = 0; r < rows; r++) {
            uint8_t *row = expert + (uint64_t)r * row_bytes;
            for (uint32_t b = 0; b < blocks; b++) {
                uint8_t *blk = row + (uint64_t)b * Q2_K_BLOCK_BYTES;
                memset(blk, 0x01, 16);
                memset(blk + 16, 0x55, 64);
                put_f16(blk + 80, 0x3c00u);
                put_f16(blk + 82, 0x0000u);
            }
        }
    }
}

int main(int argc, char **argv) {
    uint32_t experts = 256;
    uint32_t topk = 6;
    uint32_t hidden = 4096;
    uint32_t expert_hidden = 2048;
    uint32_t reps = 10;
    uint32_t warmup = 1;
    uint32_t samples = 1;
    uint32_t reps_per_sample = 0;
    if (argc > 1 && !parse_u32(argv[1], &experts)) return 2;
    if (argc > 2 && !parse_u32(argv[2], &topk)) return 2;
    if (argc > 3 && !parse_u32(argv[3], &hidden)) return 2;
    if (argc > 4 && !parse_u32(argv[4], &expert_hidden)) return 2;
    if (argc > 5 && !parse_u32(argv[5], &reps)) return 2;
    if (argc > 6 && !parse_u32_zero_ok(argv[6], &warmup)) return 2;
    if (argc > 7 && !parse_u32(argv[7], &samples)) return 2;
    if (argc > 8 && !parse_u32(argv[8], &reps_per_sample)) return 2;
    if (reps_per_sample == 0) reps_per_sample = reps;
    if (argc > 9 || topk > experts || topk > 16 ||
        (hidden % QK_K) != 0 || (expert_hidden % QK_K) != 0) {
        return 2;
    }

    uint64_t iq2_row_bytes = (uint64_t)(hidden / QK_K) * IQ2_XXS_BLOCK_BYTES;
    uint64_t q2_row_bytes = (uint64_t)(expert_hidden / QK_K) * Q2_K_BLOCK_BYTES;
    uint64_t iq2_expert_bytes = 0;
    uint64_t q2_expert_bytes = 0;
    uint64_t iq2_bytes = 0;
    uint64_t q2_bytes = 0;
    uint64_t input_bytes = (uint64_t)hidden * sizeof(float);
    uint64_t indices_bytes = (uint64_t)topk * sizeof(uint32_t);
    uint64_t weights_bytes = (uint64_t)topk * sizeof(float);
    uint64_t scratch_bytes = (uint64_t)topk * expert_hidden * sizeof(float);
    uint64_t out_bytes = (uint64_t)hidden * sizeof(float);
    if (!checked_mul_u64(expert_hidden, iq2_row_bytes, &iq2_expert_bytes) ||
        !checked_mul_u64(experts, iq2_expert_bytes, &iq2_bytes) ||
        !checked_mul_u64(hidden, q2_row_bytes, &q2_expert_bytes) ||
        !checked_mul_u64(experts, q2_expert_bytes, &q2_bytes)) {
        return 2;
    }
    if (iq2_bytes > SIZE_MAX || q2_bytes > SIZE_MAX ||
        input_bytes > SIZE_MAX || out_bytes > SIZE_MAX || scratch_bytes > SIZE_MAX) {
        return 2;
    }

    uint8_t *gate = (uint8_t *)malloc((size_t)iq2_bytes);
    uint8_t *up = (uint8_t *)malloc((size_t)iq2_bytes);
    uint8_t *down = (uint8_t *)malloc((size_t)q2_bytes);
    float *input = (float *)malloc((size_t)input_bytes);
    uint32_t *indices = (uint32_t *)malloc((size_t)indices_bytes);
    float *router_weights = (float *)malloc((size_t)weights_bytes);
    float *out = (float *)malloc((size_t)out_bytes);
    double *sample_iter_s = (double *)malloc((size_t)samples * sizeof(double));
    if (!gate || !up || !down || !input || !indices || !router_weights || !out || !sample_iter_s) {
        free(sample_iter_s);
        free(out);
        free(router_weights);
        free(indices);
        free(input);
        free(down);
        free(up);
        free(gate);
        return 1;
    }

    fill_iq2(gate, experts, expert_hidden, hidden);
    fill_iq2(up, experts, expert_hidden, hidden);
    fill_q2(down, experts, hidden, expert_hidden);
    for (uint32_t i = 0; i < hidden; i++) input[i] = 1.0f;
    for (uint32_t i = 0; i < topk; i++) {
        indices[i] = i;
        router_weights[i] = 1.0f / (float)topk;
    }

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *rt = NULL;
    axiom_device_buffer *gate_dev = NULL;
    axiom_device_buffer *up_dev = NULL;
    axiom_device_buffer *down_dev = NULL;
    axiom_device_buffer *input_dev = NULL;
    axiom_device_buffer *indices_dev = NULL;
    axiom_device_buffer *weights_dev = NULL;
    axiom_device_buffer *scratch_dev = NULL;
    axiom_device_buffer *out_dev = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &gate_dev, iq2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &up_dev, iq2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &down_dev, q2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &input_dev, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &indices_dev, indices_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &weights_dev, weights_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &scratch_dev, scratch_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &out_dev, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(gate_dev, 0, gate, iq2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(up_dev, 0, up, iq2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(down_dev, 0, down, q2_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(input_dev, 0, input, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(indices_dev, 0, indices, indices_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(weights_dev, 0, router_weights, weights_bytes) : rc;

    double elapsed_ms = 0.0;
    double iter_s_median = 0.0;
    double iter_s_best = 0.0;
    double iter_s_min = 0.0;
    double iter_s_max = 0.0;
    if (rc == AXIOM_OK) {
        for (uint32_t i = 0; i < warmup && rc == AXIOM_OK; i++) {
            rc = axiom_runtime_deepseek_moe_indexed_f32_scratch_device(
                    rt, gate_dev, 0, up_dev, 0, down_dev, 0, input_dev, 0,
                    indices_dev, 0, weights_dev, 0, scratch_dev, 0, out_dev, 0,
                    experts, topk, hidden, expert_hidden);
        }
    }
    if (rc == AXIOM_OK) {
        for (uint32_t sample = 0; sample < samples && rc == AXIOM_OK; sample++) {
            const double t0 = now_ms();
            for (uint32_t i = 0; i < reps_per_sample && rc == AXIOM_OK; i++) {
                rc = axiom_runtime_deepseek_moe_indexed_f32_scratch_device(
                        rt, gate_dev, 0, up_dev, 0, down_dev, 0, input_dev, 0,
                        indices_dev, 0, weights_dev, 0, scratch_dev, 0, out_dev, 0,
                        experts, topk, hidden, expert_hidden);
            }
            const double t1 = now_ms();
            const double sample_ms = t1 - t0;
            elapsed_ms += sample_ms;
            sample_iter_s[sample] = sample_ms > 0.0 ? ((double)reps_per_sample * 1000.0) / sample_ms : 0.0;
            if (sample == 0 || sample_iter_s[sample] > iter_s_best) iter_s_best = sample_iter_s[sample];
            if (sample == 0 || sample_iter_s[sample] < iter_s_min) iter_s_min = sample_iter_s[sample];
            if (sample == 0 || sample_iter_s[sample] > iter_s_max) iter_s_max = sample_iter_s[sample];
        }
        if (rc == AXIOM_OK) {
            iter_s_median = median_double(sample_iter_s, samples);
        }
    }
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(out_dev, 0, out, out_bytes) : rc;

    const float mid = (10.0f / (1.0f + expf(-10.0f))) * 10.0f;
    const float expected = (float)expert_hidden * mid;
    float max_abs = -1.0f;
    int ok = rc == AXIOM_OK;
    const float abs_limit = fmaxf(4.0f, fabsf(expected) * 0.00002f);
    for (uint32_t i = 0; ok && i < hidden; i++) {
        const float d = fabsf(out[i] - expected);
        if (d > max_abs) max_abs = d;
        if (!isfinite(out[i]) || d > abs_limit) ok = 0;
    }
    print_result(experts, topk, hidden, expert_hidden, reps, warmup, samples, reps_per_sample,
            elapsed_ms, iter_s_median, iter_s_best, iter_s_min, iter_s_max,
            out ? out[0] : 0.0f, expected, max_abs, ok);

    axiom_device_buffer_destroy(out_dev);
    axiom_device_buffer_destroy(scratch_dev);
    axiom_device_buffer_destroy(weights_dev);
    axiom_device_buffer_destroy(indices_dev);
    axiom_device_buffer_destroy(input_dev);
    axiom_device_buffer_destroy(down_dev);
    axiom_device_buffer_destroy(up_dev);
    axiom_device_buffer_destroy(gate_dev);
    axiom_runtime_destroy(rt);
    free(out);
    free(router_weights);
    free(indices);
    free(input);
    free(down);
    free(up);
    free(gate);
    free(sample_iter_s);
    return ok ? 0 : 1;
}
