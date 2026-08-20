/* axiom_fp8_probe.cpp — cross-check the libaxiom FP8 (E4M3 + E8M0 128x128 block) matvec
 * against a numpy f64 reference on a REAL DeepSeek attn/shared weight. GREEN = the FP8
 * compute path (needed for approach (i): keep attn/shared native, no requant) is correct
 * on real weights on the GB10.
 * Build: g++ -O3 -Iinclude tools/axiom_fp8_probe.cpp -o bin/axiom-fp8 -Llib -laxiom \
 *        -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lm
 * Run:  LD_LIBRARY_PATH=lib ./bin/axiom-fp8 <dir-with-fp8_*.bin>
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "axiom/axiom.h"

extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

static void *slurp(const char *p, size_t want) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(2); }
    void *b = malloc(want); if (fread(b, 1, want, f) != want) { fprintf(stderr, "read %s\n", p); exit(2); }
    fclose(f); return b;
}

int main(int argc, char **argv) {
    const char *d = argc > 1 ? argv[1] : ".";
    char pm[512]; snprintf(pm, sizeof pm, "%s/fp8_meta.txt", d);
    FILE *mf = fopen(pm, "r"); if (!mf) { fprintf(stderr, "no meta\n"); return 2; }
    unsigned rows = 0, cols = 0; if (fscanf(mf, "%u %u", &rows, &cols) != 2) return 2; fclose(mf);
    printf("axiom-fp8: rows=%u cols=%u\n", rows, cols);
    char p[512];
#define LD(nm, want) ( snprintf(p, sizeof p, "%s/" nm, d), slurp(p, want) )
    uint8_t *w = (uint8_t *)LD("fp8_w.bin", (size_t)rows * cols);
    uint8_t *s = (uint8_t *)LD("fp8_s.bin", (size_t)(rows / 128) * (cols / 128));
    float *x = (float *)LD("fp8_x.bin", (size_t)cols * 4), *yref = (float *)LD("fp8_yref.bin", (size_t)rows * 4);
    float *y = (float *)malloc((size_t)rows * 4);

    uint8_t *dw, *ds; float *dx, *dy;
    cudaMalloc(&dw, (size_t)rows * cols); cudaMalloc(&ds, (size_t)(rows / 128) * (cols / 128));
    cudaMalloc(&dx, (size_t)cols * 4); cudaMalloc(&dy, (size_t)rows * 4);
    cudaMemcpy(dw, w, (size_t)rows * cols, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s, (size_t)(rows / 128) * (cols / 128), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)cols * 4, cudaMemcpyHostToDevice);
    int rc = axiom_cuda_fp8_e4m3_e8m0_matvec_f32(0, dw, ds, dx, dy, rows, cols);
    if (rc != AXIOM_OK) { printf("axiom-fp8: matvec rc=%d FAIL\n", rc); return 1; }
    cudaMemcpy(y, dy, (size_t)rows * 4, cudaMemcpyDeviceToHost);

    double ss = 0, maxa = 0;
    for (unsigned i = 0; i < rows; ++i) ss += (double)yref[i] * yref[i];
    double rms = sqrt(ss / rows);
    for (unsigned i = 0; i < rows; ++i) { double a = fabs((double)y[i] - (double)yref[i]); if (a > maxa) maxa = a; }
    int ok = maxa < 1e-2 * (rms + 1e-6);
    printf("axiom-fp8: |y|rms=%.5f max_abs=%.6g %s\n", rms, maxa, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
