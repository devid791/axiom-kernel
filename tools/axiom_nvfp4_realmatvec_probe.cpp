/* axiom_nvfp4_realmatvec_probe.cpp — run the libaxiom NVFP4 matvec on a REAL DeepSeek
 * expert weight and cross-check vs a numpy f64 reference (tools/axiom_nvfp4_real_decode.py
 * ... dump). Inputs are the dumped raw binaries. GREEN = the production kernel is correct
 * on real weights, cross-validated by an independent (Python) implementation.
 *
 * Build (links libaxiom):
 *   g++ -O3 -Iinclude tools/axiom_nvfp4_realmatvec_probe.cpp -o bin/axiom-nvfp4-realmatvec \
 *       -Llib -laxiom -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lm -Wl,-rpath,'$ORIGIN/../lib'
 * Run:  ./bin/axiom-nvfp4-realmatvec <dir-with-nvfp4_*.bin>
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "axiom/axiom.h"

extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

static void *slurp(const char *path, size_t want) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "open %s failed\n", path); exit(2); }
    void *buf = malloc(want);
    size_t got = fread(buf, 1, want, fp); fclose(fp);
    if (got != want) { fprintf(stderr, "%s: read %zu != %zu\n", path, got, want); exit(2); }
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <dir-with-nvfp4_*.bin>\n", argv[0]);
        return 2;
    }
    const char *d = argv[1];
    char meta[512]; snprintf(meta, sizeof meta, "%s/nvfp4_meta.txt", d);
    FILE *mf = fopen(meta, "r"); if (!mf) { fprintf(stderr, "no meta\n"); return 2; }
    unsigned rows = 0, cols = 0; float global = 0.0f;
    if (fscanf(mf, "%u %u %f", &rows, &cols, &global) != 3) { fprintf(stderr, "meta parse\n"); return 2; }
    fclose(mf);
    printf("axiom-nvfp4-realmatvec: rows=%u cols=%u global=%.6g\n", rows, cols, global);

    char pw[512], ps[512], px[512], py[512];
    snprintf(pw, sizeof pw, "%s/nvfp4_w.bin", d); snprintf(ps, sizeof ps, "%s/nvfp4_s.bin", d);
    snprintf(px, sizeof px, "%s/nvfp4_x.bin", d); snprintf(py, sizeof py, "%s/nvfp4_yref.bin", d);
    size_t nw = (size_t)rows * cols / 2, ns = (size_t)rows * (cols / 16);
    uint8_t *w = (uint8_t *)slurp(pw, nw), *s = (uint8_t *)slurp(ps, ns);
    float *x = (float *)slurp(px, (size_t)cols * 4), *yref = (float *)slurp(py, (size_t)rows * 4);
    float *yd = (float *)malloc((size_t)rows * 4);

    uint8_t *dw, *ds; float *dx, *dy;
    if (cudaMalloc(&dw, nw) || cudaMalloc(&ds, ns) || cudaMalloc(&dx, (size_t)cols * 4) ||
        cudaMalloc(&dy, (size_t)rows * 4)) { fprintf(stderr, "cudaMalloc\n"); return 2; }
    cudaMemcpy(dw, w, nw, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s, ns, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)cols * 4, cudaMemcpyHostToDevice);
    int rc = axiom_cuda_e2m1_nvfp4_matvec_f32(0, dw, ds, global, dx, dy, rows, cols);
    if (rc != AXIOM_OK) { printf("axiom-nvfp4-realmatvec: matvec rc=%d FAIL\n", rc); return 1; }
    cudaMemcpy(yd, dy, (size_t)rows * 4, cudaMemcpyDeviceToHost);

    double max_abs = 0, max_rel = 0, ynorm = 0;
    for (unsigned r = 0; r < rows; ++r) ynorm += (double)yref[r] * yref[r];
    ynorm = sqrt(ynorm / rows);
    for (unsigned r = 0; r < rows; ++r) {
        double a = fabs((double)yd[r] - (double)yref[r]);
        if (a > max_abs) max_abs = a;
        double rel = a / (fabs((double)yref[r]) + 1e-6);
        if (rel > max_rel) max_rel = rel;
    }
    int ok = max_abs < 1e-2 * (ynorm + 1e-6);   /* abs error small vs typical |y| */
    printf("axiom-nvfp4-realmatvec: |y|rms=%.5f max_abs=%.6g max_rel=%.4g %s\n",
           ynorm, max_abs, max_rel, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
