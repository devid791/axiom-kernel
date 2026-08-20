/* axiom_nvfp4_ffn_probe.cpp — run a full real NVFP4 routed-expert FFN (gate w1 -> silu ->
 * * up w3 -> down w2) via the libaxiom NVFP4 matvec on the GB10, cross-checked vs the numpy
 * f64 reference (tools/nvfp4_ffn_ref.py). GREEN = the expert FFN compute path is correct on
 * real DeepSeek weights (routed-expert path; independent of the attn/shared dtype decision).
 * Build: g++ -O3 -Iinclude tools/axiom_nvfp4_ffn_probe.cpp -o bin/axiom-nvfp4-ffn \
 *        -Llib -laxiom -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lm
 * Run:  LD_LIBRARY_PATH=lib ./bin/axiom-nvfp4-ffn <dir-with-*.bin>
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "axiom/axiom.h"

extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

static void *slurp(const char *p, size_t want) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(2); }
    void *b = malloc(want); if (fread(b, 1, want, f) != want) { fprintf(stderr, "read %s\n", p); exit(2); }
    fclose(f); return b;
}
/* device NVFP4 matvec on host-provided host arrays: uploads w/scale/x, returns out (host). */
static void nvfp4_matvec(const uint8_t *w, const uint8_t *s, float g, const float *x, float *y,
                         uint32_t rows, uint32_t cols) {
    uint8_t *dw, *ds; float *dx, *dy;
    cudaMalloc(&dw, (size_t)rows * cols / 2); cudaMalloc(&ds, (size_t)rows * (cols / 16));
    cudaMalloc(&dx, (size_t)cols * 4); cudaMalloc(&dy, (size_t)rows * 4);
    cudaMemcpy(dw, w, (size_t)rows * cols / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, s, (size_t)rows * (cols / 16), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t)cols * 4, cudaMemcpyHostToDevice);
    int rc = axiom_cuda_e2m1_nvfp4_matvec_f32(0, dw, ds, g, dx, dy, rows, cols);
    if (rc != AXIOM_OK) { fprintf(stderr, "matvec rc=%d\n", rc); exit(1); }
    cudaMemcpy(y, dy, (size_t)rows * 4, cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(ds); cudaFree(dx); cudaFree(dy);
}

int main(int argc, char **argv) {
    const char *d = argc > 1 ? argv[1] : ".";
    char pm[512]; snprintf(pm, sizeof pm, "%s/ffn_meta.txt", d);
    FILE *mf = fopen(pm, "r"); if (!mf) { fprintf(stderr, "no meta\n"); return 2; }
    unsigned inter = 0, hidden = 0; float g1 = 0, g3 = 0, g2 = 0;
    if (fscanf(mf, "%u %u %f %f %f", &inter, &hidden, &g1, &g3, &g2) != 5) return 2;
    fclose(mf);
    printf("axiom-nvfp4-ffn: inter=%u hidden=%u\n", inter, hidden);
    char p[512];
#define LD(nm, want) ( snprintf(p, sizeof p, "%s/" nm, d), slurp(p, want) )
    uint8_t *w1 = (uint8_t *)LD("w1_w.bin", (size_t)inter * hidden / 2), *s1 = (uint8_t *)LD("w1_s.bin", (size_t)inter * (hidden / 16));
    uint8_t *w3 = (uint8_t *)LD("w3_w.bin", (size_t)inter * hidden / 2), *s3 = (uint8_t *)LD("w3_s.bin", (size_t)inter * (hidden / 16));
    uint8_t *w2 = (uint8_t *)LD("w2_w.bin", (size_t)hidden * inter / 2), *s2 = (uint8_t *)LD("w2_s.bin", (size_t)hidden * (inter / 16));
    float *x = (float *)LD("ffn_x.bin", (size_t)hidden * 4), *yref = (float *)LD("ffn_yref.bin", (size_t)hidden * 4);
    float *g = (float *)malloc((size_t)inter * 4), *u = (float *)malloc((size_t)inter * 4);
    float *hbuf = (float *)malloc((size_t)inter * 4), *y = (float *)malloc((size_t)hidden * 4);

    nvfp4_matvec(w1, s1, g1, x, g, inter, hidden);   /* gate -> [inter] */
    nvfp4_matvec(w3, s3, g3, x, u, inter, hidden);   /* up   -> [inter] */
    for (unsigned i = 0; i < inter; ++i) { float v = g[i]; hbuf[i] = (v / (1.0f + expf(-v))) * u[i]; }
    nvfp4_matvec(w2, s2, g2, hbuf, y, hidden, inter);/* down -> [hidden] */

    double ss = 0, maxa = 0;
    for (unsigned i = 0; i < hidden; ++i) ss += (double)yref[i] * yref[i];
    double rms = sqrt(ss / hidden);
    for (unsigned i = 0; i < hidden; ++i) { double a = fabs((double)y[i] - (double)yref[i]); if (a > maxa) maxa = a; }
    int ok = maxa < 1e-2 * (rms + 1e-6);
    printf("axiom-nvfp4-ffn: |y|rms=%.5f max_abs=%.6g %s\n", rms, maxa, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
