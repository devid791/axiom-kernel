#define _POSIX_C_SOURCE 200809L

#include "axiom/axiom.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8u);
}

static float input_value(uint32_t i, uint32_t salt) {
    const int32_t v = (int32_t)((i * 37u + salt * 53u) % 101u) - 50;
    return (float)v / (float)(3u + (salt & 3u));
}

static int8_t weight_value(uint32_t row, uint32_t col) {
    const int32_t v = (int32_t)((row * 17u + col * 29u + 11u) % 255u) - 127;
    return (int8_t)v;
}

static double max_abs_diff(const float *a, const float *b, uint32_t n) {
    double m = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    uint32_t rows = 257u;
    uint32_t cols = 4096u;
    if (argc > 1) rows = (uint32_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) cols = (uint32_t)strtoul(argv[2], NULL, 10);
    if (argc > 3 || rows == 0 || cols == 0 || (cols % 32u) != 0u) {
        fprintf(stderr, "usage: axiom-q8-soa-matvec2-smoke [rows] [cols]\n");
        return 1;
    }

    const uint32_t blocks = cols / 32u;
    const uint64_t qs_bytes = (uint64_t)rows * cols;
    const uint64_t scale_bytes = (uint64_t)rows * blocks * sizeof(uint16_t);
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);

    int8_t *qs = (int8_t *)malloc((size_t)qs_bytes);
    uint8_t *scales = (uint8_t *)malloc((size_t)scale_bytes);
    float *x0 = (float *)malloc((size_t)input_bytes);
    float *x1 = (float *)malloc((size_t)input_bytes);
    float *ref0 = (float *)malloc((size_t)out_bytes);
    float *ref1 = (float *)malloc((size_t)out_bytes);
    float *got0 = (float *)malloc((size_t)out_bytes);
    float *got1 = (float *)malloc((size_t)out_bytes);
    if (!qs || !scales || !x0 || !x1 || !ref0 || !ref1 || !got0 || !got1) {
        fprintf(stderr, "alloc_fail\n");
        free(got1); free(got0); free(ref1); free(ref0);
        free(x1); free(x0); free(scales); free(qs);
        return 1;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            qs[(uint64_t)r * cols + c] = weight_value(r, c);
        }
        for (uint32_t b = 0; b < blocks; ++b) {
            put_u16(scales + ((uint64_t)r * blocks + b) * sizeof(uint16_t), 0x3c00u);
        }
    }
    for (uint32_t c = 0; c < cols; ++c) {
        x0[c] = input_value(c, 1u);
        x1[c] = input_value(c, 7u);
    }

    setenv("AXIOM_DS4_Q8_SOA_PREQ", "1", 1);

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;

    axiom_runtime *rt = NULL;
    axiom_device_buffer *qs_dev = NULL;
    axiom_device_buffer *scales_dev = NULL;
    axiom_device_buffer *x0_dev = NULL;
    axiom_device_buffer *x1_dev = NULL;
    axiom_device_buffer *y0_dev = NULL;
    axiom_device_buffer *y1_dev = NULL;

    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &qs_dev, qs_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &scales_dev, scale_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &x0_dev, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &x1_dev, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &y0_dev, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &y1_dev, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(qs_dev, 0, qs, qs_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(scales_dev, 0, scales, scale_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(x0_dev, 0, x0, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(x1_dev, 0, x1, input_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_soa_matvec_f32_device(
            rt, qs_dev, scales_dev, x0_dev, 0, y0_dev, 0, rows, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_soa_matvec_f32_device(
            rt, qs_dev, scales_dev, x1_dev, 0, y1_dev, 0, rows, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(y0_dev, 0, ref0, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(y1_dev, 0, ref1, out_bytes) : rc;

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_soa_matvec2_f32_device(
            rt, qs_dev, scales_dev,
            x0_dev, 0, x1_dev, 0,
            y0_dev, 0, y1_dev, 0,
            rows, cols) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(y0_dev, 0, got0, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(y1_dev, 0, got1, out_bytes) : rc;

    const double max0 = rc == AXIOM_OK ? max_abs_diff(ref0, got0, rows) : INFINITY;
    const double max1 = rc == AXIOM_OK ? max_abs_diff(ref1, got1, rows) : INFINITY;
    const int pass = rc == AXIOM_OK && max0 == 0.0 && max1 == 0.0;
    printf("{\"status\":\"%s\",\"rc\":%d,\"rows\":%u,\"cols\":%u,"
           "\"max_abs0\":%.9g,\"max_abs1\":%.9g}\n",
           pass ? "pass" : "fail", rc, rows, cols, max0, max1);

    axiom_device_buffer_destroy(y1_dev);
    axiom_device_buffer_destroy(y0_dev);
    axiom_device_buffer_destroy(x1_dev);
    axiom_device_buffer_destroy(x0_dev);
    axiom_device_buffer_destroy(scales_dev);
    axiom_device_buffer_destroy(qs_dev);
    axiom_runtime_destroy(rt);
    free(got1); free(got0); free(ref1); free(ref0);
    free(x1); free(x0); free(scales); free(qs);
    return pass ? 0 : 1;
}
