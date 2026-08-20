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
#define IQ2_XXS_BLOCK_COLS 256u

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

static void print_result(const char *kernel, uint32_t rows, uint32_t cols, uint32_t reps, double elapsed_ms, int ok) {
    const double matvec_s = elapsed_ms > 0.0 ? ((double)reps * 1000.0) / elapsed_ms : 0.0;
    const double throughput = elapsed_ms > 0.0 ?
            ((double)rows * (double)cols * (double)reps) / (elapsed_ms * 1000000.0) : 0.0;
    printf("kernel=%s rows=%u cols=%u reps=%u elapsed_ms=%.3f matvec_s=%.3f throughput=%.3f status=%s\n",
            kernel,
            rows, cols, reps, elapsed_ms, matvec_s, throughput, ok ? "ok" : "fail");
}

typedef int (*iq2_kernel_fn)(
        axiom_runtime *,
        const axiom_device_buffer *,
        uint64_t,
        const axiom_device_buffer *,
        uint64_t,
        axiom_device_buffer *,
        uint64_t,
        uint32_t,
        uint32_t);

static int run_kernel(
        const char *name,
        iq2_kernel_fn fn,
        axiom_runtime *rt,
        axiom_device_buffer *w_dev,
        axiom_device_buffer *x_dev,
        axiom_device_buffer *y_dev,
        float *out,
        uint64_t out_bytes,
        uint32_t rows,
        uint32_t cols,
        uint32_t reps) {
    memset(out, 0, (size_t)out_bytes);
    int rc = fn(rt, w_dev, 0, x_dev, 0, y_dev, 0, rows, cols);
    double elapsed_ms = 0.0;
    if (rc == AXIOM_OK) {
        const double t0 = now_ms();
        for (uint32_t i = 0; i < reps && rc == AXIOM_OK; ++i) {
            rc = fn(rt, w_dev, 0, x_dev, 0, y_dev, 0, rows, cols);
        }
        const double t1 = now_ms();
        elapsed_ms = t1 - t0;
    }
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(y_dev, 0, out, out_bytes) : rc;

    int ok = rc == AXIOM_OK;
    const float expected = (float)cols;
    for (uint32_t r = 0; ok && r < rows; ++r) {
        if (fabsf(out[r] - expected) > 1.0e-3f) ok = 0;
    }
    print_result(name, rows, cols, reps, elapsed_ms, ok);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    uint32_t rows = 4096;
    uint32_t cols = 4096;
    uint32_t reps = 100;
    if (argc > 1 && !parse_u32(argv[1], &rows)) {
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }
    if (argc > 2 && !parse_u32(argv[2], &cols)) {
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }
    if (argc > 3 && !parse_u32(argv[3], &reps)) {
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }
    if (argc > 4 || (cols % IQ2_XXS_BLOCK_COLS) != 0u) {
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }

    const uint64_t blocks = (uint64_t)cols / IQ2_XXS_BLOCK_COLS;
    const uint64_t row_bytes = blocks * IQ2_XXS_BLOCK_BYTES;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    const uint64_t input_bytes = (uint64_t)cols * sizeof(float);
    const uint64_t out_bytes = (uint64_t)rows * sizeof(float);
    if (row_bytes == 0 || weight_bytes / row_bytes != rows ||
        input_bytes / sizeof(float) != cols || out_bytes / sizeof(float) != rows) {
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }

    uint8_t *weights = (uint8_t *)calloc(1, (size_t)weight_bytes);
    float *input = (float *)malloc((size_t)input_bytes);
    float *out = (float *)malloc((size_t)out_bytes);
    if (!weights || !input || !out) {
        free(out);
        free(input);
        free(weights);
        print_result("none", rows, cols, reps, 0.0, 0);
        return 1;
    }

    for (uint32_t i = 0; i < cols; ++i) input[i] = 1.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        uint8_t *row = weights + (uint64_t)r * row_bytes;
        for (uint32_t b = 0; b < (uint32_t)blocks; ++b) {
            put_f16(row + (uint64_t)b * IQ2_XXS_BLOCK_BYTES, 0x3c00u);
        }
    }

    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;

    axiom_runtime *rt = NULL;
    axiom_device_buffer *w_dev = NULL;
    axiom_device_buffer *x_dev = NULL;
    axiom_device_buffer *y_dev = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &w_dev, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &x_dev, input_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_create(rt, &y_dev, out_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(w_dev, 0, weights, weight_bytes) : rc;
    rc = rc == AXIOM_OK ? axiom_device_buffer_upload(x_dev, 0, input, input_bytes) : rc;

    int ok = 0;
    if (rc == AXIOM_OK) {
        ok = run_kernel("scalar", axiom_runtime_iq2_xxs_matvec_f32_device,
                        rt, w_dev, x_dev, y_dev, out, out_bytes, rows, cols, reps);
        ok |= run_kernel("warp", axiom_runtime_iq2_xxs_matvec_f32_warp_device,
                         rt, w_dev, x_dev, y_dev, out, out_bytes, rows, cols, reps);
    } else {
        print_result("setup", rows, cols, reps, 0.0, 0);
        ok = 1;
    }

    axiom_device_buffer_destroy(y_dev);
    axiom_device_buffer_destroy(x_dev);
    axiom_device_buffer_destroy(w_dev);
    axiom_runtime_destroy(rt);
    free(out);
    free(input);
    free(weights);

    return ok ? 1 : 0;
}
