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
    const int32_t v = (int32_t)((i * 41u + salt * 67u) % 131u) - 65;
    return (float)v / (float)(5u + (salt & 7u));
}

static int8_t weight_value(uint32_t row, uint32_t col) {
    const int32_t v = (int32_t)((row * 19u + col * 31u + 13u) % 255u) - 127;
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
        fprintf(stderr, "usage: axiom-q8-matvec4-smoke [rows] [cols]\n");
        return 1;
    }

    const uint32_t blocks = cols / 32u;
    const uint64_t weight_bytes = (uint64_t)rows * blocks * 34u;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);

    uint8_t *weight = (uint8_t *)malloc((size_t)weight_bytes);
    float *x[4] = {NULL, NULL, NULL, NULL};
    float *ref[4] = {NULL, NULL, NULL, NULL};
    float *got[4] = {NULL, NULL, NULL, NULL};
    int alloc_ok = weight != NULL;
    for (uint32_t s = 0; s < 4u; ++s) {
        x[s] = (float *)malloc((size_t)input_bytes);
        ref[s] = (float *)malloc((size_t)out_bytes);
        got[s] = (float *)malloc((size_t)out_bytes);
        alloc_ok = alloc_ok && x[s] && ref[s] && got[s];
    }
    if (!alloc_ok) {
        fprintf(stderr, "alloc_fail\n");
        for (uint32_t s = 0; s < 4u; ++s) {
            free(got[s]);
            free(ref[s]);
            free(x[s]);
        }
        free(weight);
        return 1;
    }

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t b = 0; b < blocks; ++b) {
            uint8_t *blk = weight + ((uint64_t)r * blocks + b) * 34u;
            put_u16(blk, 0x3c00u);
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                blk[2u + lane] = (uint8_t)weight_value(r, b * 32u + lane);
            }
        }
    }
    for (uint32_t s = 0; s < 4u; ++s) {
        for (uint32_t c = 0; c < cols; ++c) {
            x[s][c] = input_value(c, 3u + s * 5u);
        }
    }

    setenv("AXIOM_DS4_Q8_DP4A", "0", 1);
    setenv("AXIOM_DS4_Q8_PREQ_DISABLE", "0", 1);
    setenv("AXIOM_DS4_Q8_PREQ_SXQ", "0", 1);

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;

    axiom_runtime *rt = NULL;
    axiom_device_buffer *weight_dev = NULL;
    axiom_device_buffer *x_dev[4] = {NULL, NULL, NULL, NULL};
    axiom_device_buffer *y_dev[4] = {NULL, NULL, NULL, NULL};

    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &weight_dev, weight_bytes) : rc;
    for (uint32_t s = 0; s < 4u; ++s) {
        rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &x_dev[s], input_bytes) : rc;
        rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &y_dev[s], out_bytes) : rc;
    }
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(weight_dev, 0, weight, weight_bytes) : rc;
    for (uint32_t s = 0; s < 4u; ++s) {
        rc = rc == AXIOM_OK ? axiom_device_buffer_upload(x_dev[s], 0, x[s], input_bytes) : rc;
    }

    for (uint32_t s = 0; s < 4u; ++s) {
        rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec_f32_device(
                rt, weight_dev, 0, x_dev[s], 0, y_dev[s], 0, rows, cols) : rc;
        rc = rc == AXIOM_OK ? axiom_device_buffer_download(y_dev[s], 0, ref[s], out_bytes) : rc;
    }

    rc = rc == AXIOM_OK ? axiom_runtime_q8_0_matvec4_f32_device(
            rt, weight_dev, 0,
            x_dev[0], 0, x_dev[1], 0, x_dev[2], 0, x_dev[3], 0,
            y_dev[0], 0, y_dev[1], 0, y_dev[2], 0, y_dev[3], 0,
            rows, cols) : rc;
    for (uint32_t s = 0; s < 4u; ++s) {
        rc = rc == AXIOM_OK ? axiom_device_buffer_download(y_dev[s], 0, got[s], out_bytes) : rc;
    }

    double max_abs[4] = {INFINITY, INFINITY, INFINITY, INFINITY};
    if (rc == AXIOM_OK) {
        for (uint32_t s = 0; s < 4u; ++s) max_abs[s] = max_abs_diff(ref[s], got[s], rows);
    }
    const double tol = 1.0e-5;
    const int pass = rc == AXIOM_OK &&
            max_abs[0] <= tol && max_abs[1] <= tol &&
            max_abs[2] <= tol && max_abs[3] <= tol;
    printf("{\"status\":\"%s\",\"rc\":%d,\"rows\":%u,\"cols\":%u,"
           "\"max_abs0\":%.9g,\"max_abs1\":%.9g,"
           "\"max_abs2\":%.9g,\"max_abs3\":%.9g}\n",
           pass ? "pass" : "fail", rc, rows, cols,
           max_abs[0], max_abs[1], max_abs[2], max_abs[3]);

    for (uint32_t s = 0; s < 4u; ++s) {
        axiom_device_buffer_destroy(y_dev[s]);
        axiom_device_buffer_destroy(x_dev[s]);
    }
    axiom_device_buffer_destroy(weight_dev);
    axiom_runtime_destroy(rt);
    for (uint32_t s = 0; s < 4u; ++s) {
        free(got[s]);
        free(ref[s]);
        free(x[s]);
    }
    free(weight);
    return pass ? 0 : 1;
}
