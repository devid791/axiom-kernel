/* axiom_nvfp4_smoke.cpp — oracle-parity smoke for the libaxiom NVFP4 expert matvec.
 * Links libaxiom; allocates device memory; compares axiom_cuda_e2m1_nvfp4_matvec_f32
 * against a host reference decode. GREEN on the GB10 = the production NVFP4 kernel is
 * compiled into libaxiom and numerically correct on Blackwell.
 *
 * FUSED MoE SECTION (compiled only with -DAXIOM_NVFP4_SMOKE_FUSED, target
 * `make nvfp4-moe-fused-smoke` / bin/axiom-nvfp4-fused-smoke <vectors_dir>): loads the
 * tools/nvfp4_moe_fused_ref.py vectors and runs the SAME device tensors through
 *   (a) the host-loop oracle  axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device
 *   (b) the fused chain       axiom_runtime_deepseek_moe_nvfp4_indexed_fused_f32_scratch_device
 * then checks (a) vs numpy-f64, (b) vs numpy-f64, (b) vs (a), and the fused mid plane
 * vs the numpy mid. BIT-EXACTNESS BETWEEN (a) AND (b) IS NOT CLAIMED: the host loop
 * accumulates e2m1*bs*global*x per element in strict column order; the fused kernels
 * accumulate per-group-16 partials times the block scale, in lane-strided order, with
 * the global scale hoisted to the end — same real value, different fp32 association.
 * The parity bar is therefore relative error: |d| <= rtol*|ref| + atol_rel*rms(ref)
 * with rtol = 2e-3 and atol_rel = 1e-4 (roughly 10-30x the expected fp32-reassociation
 * drift at these dims, ~100x below any layout/indexing bug, which shows as order-1
 * relative error). The guard keeps the legacy axiom-nvfp4-smoke link (libaxiom only)
 * free of fused symbols; the fused binary links build/axiom_cuda_nvfp4_moe.o
 * explicitly and exports it (-Wl,--export-dynamic) so libaxiom's runtime wrapper can
 * resolve the fused device entry before the .o joins AXIOM_OBJS. */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "axiom/axiom.h"

extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

static float e2m1(uint8_t n) { static const float t[8] = {0,0.5f,1,1.5f,2,3,4,6}; float v = t[n & 7]; return (n & 8) ? -v : v; }
static float e4m3(uint8_t b) {
    uint32_t s = (b >> 7) & 1, e = (b >> 3) & 0xf, m = b & 7;
    float v = (e == 0) ? (float)m * (0.015625f / 8.0f) : (1.0f + (float)m / 8.0f) * exp2f((float)((int)e - 7));
    return s ? -v : v;
}

#ifdef AXIOM_NVFP4_SMOKE_FUSED
#include <cstring>

static void *fused_load(const char *dir, const char *name, uint64_t bytes) {
    char p[1024];
    snprintf(p, sizeof p, "%s/%s", dir, name);
    FILE *f = fopen(p, "rb");
    if (!f) { printf("axiom-nvfp4-smoke[fused]: missing %s FAIL\n", p); return NULL; }
    void *buf = malloc(bytes ? (size_t)bytes : 1u);
    size_t got = buf ? fread(buf, 1, (size_t)bytes, f) : 0;
    int extra = fgetc(f);
    fclose(f);
    if (!buf || got != bytes || extra != EOF) {
        printf("axiom-nvfp4-smoke[fused]: bad size for %s (want %llu) FAIL\n",
               p, (unsigned long long)bytes);
        free(buf);
        return NULL;
    }
    return buf;
}

static axiom_device_buffer *fused_dev(axiom_runtime *rt, const void *src, uint64_t bytes) {
    axiom_device_buffer *b = NULL;
    if (axiom_device_buffer_create(rt, &b, bytes) != AXIOM_OK) return NULL;
    if (src && axiom_device_buffer_upload(b, 0, src, bytes) != AXIOM_OK) {
        axiom_device_buffer_destroy(b);
        return NULL;
    }
    return b;
}

/* pass criterion: |got-ref| <= rtol*|ref| + atol_rel*rms(ref)  (see file header) */
static int fused_cmp(const char *tag, const float *got, const float *ref, uint32_t n) {
    const float rtol = 2.0e-3f, atol_rel = 1.0e-4f;
    double sq = 0.0;
    for (uint32_t i = 0; i < n; i++) sq += (double)ref[i] * (double)ref[i];
    const float atol = atol_rel * (float)sqrt(sq / (double)n);
    float mad = 0.0f, mrel = 0.0f;
    int ok = 1;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(got[i] - ref[i]);
        if (d > rtol * fabsf(ref[i]) + atol) ok = 0;
        if (d > mad) mad = d;
        const float r = d / fmaxf(fabsf(ref[i]), 1.0e-6f);
        if (r > mrel) mrel = r;
    }
    printf("axiom-nvfp4-smoke[fused]: %-14s n=%u max_abs=%.3g max_rel=%.3g %s\n",
           tag, n, mad, mrel, ok ? "OK" : "FAIL");
    return ok;
}

/* host-loop oracle vs fused chain vs numpy f64, all on the same device tensors. */
static int fused_moe_section(const char *dir) {
    if (!dir) {
        printf("axiom-nvfp4-smoke[fused]: usage: axiom-nvfp4-fused-smoke <vectors_dir> FAIL\n");
        return 2;
    }
    char p[1024];
    snprintf(p, sizeof p, "%s/meta.txt", dir);
    FILE *mf = fopen(p, "r");
    unsigned E = 0, K = 0, H = 0, EH = 0;
    if (!mf || fscanf(mf, "%u %u %u %u", &E, &K, &H, &EH) != 4) {
        if (mf) fclose(mf);
        printf("axiom-nvfp4-smoke[fused]: bad %s FAIL\n", p);
        return 2;
    }
    fclose(mf);
    if (E == 0 || K == 0 || K > E || H == 0 || (H % 32u) != 0u || EH == 0 || (EH % 32u) != 0u) {
        printf("axiom-nvfp4-smoke[fused]: bad dims E=%u K=%u H=%u EH=%u FAIL\n", E, K, H, EH);
        return 2;
    }
    const uint64_t gu_w = (uint64_t)E * EH * (H / 2u), gu_bs = (uint64_t)E * EH * (H / 16u);
    const uint64_t dn_w = (uint64_t)E * H * (EH / 2u), dn_bs = (uint64_t)E * H * (EH / 16u);
    const uint64_t gs = (uint64_t)E * 4u, xb = (uint64_t)H * 4u, ib = (uint64_t)K * 4u;
    const uint64_t midb = (uint64_t)K * EH * 4u;
    void *gw = fused_load(dir, "gate_w.bin", gu_w), *gbs = fused_load(dir, "gate_bs.bin", gu_bs);
    void *gg = fused_load(dir, "gate_gs.bin", gs);
    void *uw = fused_load(dir, "up_w.bin", gu_w), *ubs = fused_load(dir, "up_bs.bin", gu_bs);
    void *ug = fused_load(dir, "up_gs.bin", gs);
    void *dw = fused_load(dir, "down_w.bin", dn_w), *dbs = fused_load(dir, "down_bs.bin", dn_bs);
    void *dg = fused_load(dir, "down_gs.bin", gs);
    void *x = fused_load(dir, "x.bin", xb), *idx = fused_load(dir, "idx.bin", ib);
    void *rw = fused_load(dir, "rw.bin", ib);
    float *mid_ref = (float *)fused_load(dir, "mid_ref.bin", midb);
    float *y_ref = (float *)fused_load(dir, "y_ref.bin", xb);
    if (!gw || !gbs || !gg || !uw || !ubs || !ug || !dw || !dbs || !dg ||
        !x || !idx || !rw || !mid_ref || !y_ref) return 2;

    axiom_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;
    cfg.device = 0;
    axiom_runtime *rt = NULL;
    int rc = axiom_runtime_create(&rt, &cfg);
    if (rc != AXIOM_OK) {
        printf("axiom-nvfp4-smoke[fused]: runtime_create rc=%d FAIL\n", rc);
        return 2;
    }
    axiom_device_buffer *d_gw = fused_dev(rt, gw, gu_w), *d_gbs = fused_dev(rt, gbs, gu_bs);
    axiom_device_buffer *d_gg = fused_dev(rt, gg, gs);
    axiom_device_buffer *d_uw = fused_dev(rt, uw, gu_w), *d_ubs = fused_dev(rt, ubs, gu_bs);
    axiom_device_buffer *d_ug = fused_dev(rt, ug, gs);
    axiom_device_buffer *d_dw = fused_dev(rt, dw, dn_w), *d_dbs = fused_dev(rt, dbs, dn_bs);
    axiom_device_buffer *d_dg = fused_dev(rt, dg, gs);
    axiom_device_buffer *d_x = fused_dev(rt, x, xb);
    axiom_device_buffer *d_idx = fused_dev(rt, idx, ib), *d_rw = fused_dev(rt, rw, ib);
    /* scratch_gate sized as the FUSED mid plane [K, EH]; the host loop only touches its
     * first EH floats, so both entries can share it. */
    axiom_device_buffer *d_mid = fused_dev(rt, NULL, midb);
    axiom_device_buffer *d_up = fused_dev(rt, NULL, (uint64_t)EH * 4u);
    axiom_device_buffer *d_eout = fused_dev(rt, NULL, xb);
    axiom_device_buffer *d_yh = fused_dev(rt, NULL, xb), *d_yf = fused_dev(rt, NULL, xb);
    float *yh = (float *)malloc((size_t)xb), *yf = (float *)malloc((size_t)xb);
    float *mid_got = (float *)malloc((size_t)midb);
    if (!d_gw || !d_gbs || !d_gg || !d_uw || !d_ubs || !d_ug || !d_dw || !d_dbs || !d_dg ||
        !d_x || !d_idx || !d_rw || !d_mid || !d_up || !d_eout || !d_yh || !d_yf ||
        !yh || !yf || !mid_got) {
        printf("axiom-nvfp4-smoke[fused]: device alloc/upload FAIL\n");
        return 2;
    }

    /* (a) host-loop oracle */
    rc = axiom_runtime_deepseek_moe_nvfp4_indexed_f32_scratch_device(
            rt, d_gw, d_gbs, d_gg, d_uw, d_ubs, d_ug, d_dw, d_dbs, d_dg,
            d_x, 0, d_idx, 0, d_rw, 0, d_mid, d_up, d_eout, d_yh, 0, E, K, H, EH);
    if (rc != AXIOM_OK) {
        printf("axiom-nvfp4-smoke[fused]: host-loop rc=%d FAIL\n", rc);
        return 1;
    }
    rc = axiom_device_buffer_download(d_yh, 0, yh, xb);
    /* (b) fused chain (same buffers; overwrites d_mid with the fused mid plane) */
    rc = rc == AXIOM_OK ? axiom_runtime_deepseek_moe_nvfp4_indexed_fused_f32_scratch_device(
            rt, d_gw, d_gbs, d_gg, d_uw, d_ubs, d_ug, d_dw, d_dbs, d_dg,
            d_x, 0, d_idx, 0, d_rw, 0, d_mid, d_up, d_eout, d_yf, 0, E, K, H, EH) : rc;
    if (rc != AXIOM_OK) {
        printf("axiom-nvfp4-smoke[fused]: fused rc=%d FAIL\n", rc);
        return 1;
    }
    rc = axiom_device_buffer_download(d_yf, 0, yf, xb);
    rc = rc == AXIOM_OK ? axiom_device_buffer_download(d_mid, 0, mid_got, midb) : rc;
    if (rc != AXIOM_OK) {
        printf("axiom-nvfp4-smoke[fused]: download rc=%d FAIL\n", rc);
        return 1;
    }

    int ok = 1;
    ok &= fused_cmp("mid_vs_numpy", mid_got, mid_ref, K * EH);
    ok &= fused_cmp("host_vs_numpy", yh, y_ref, H);
    ok &= fused_cmp("fused_vs_numpy", yf, y_ref, H);
    ok &= fused_cmp("fused_vs_host", yf, yh, H);
    printf("axiom-nvfp4-smoke[fused]: E=%u K=%u H=%u EH=%u %s\n", E, K, H, EH, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
#endif /* AXIOM_NVFP4_SMOKE_FUSED */

int main(int argc, char **argv) {
    const uint32_t rows = 64, cols = 256, G = 16, groups = cols / G; const float global = 0.5f;
    uint64_t nq = (uint64_t)rows * cols / 2;
    uint8_t *w = (uint8_t *)malloc(nq), *bs = (uint8_t *)malloc((size_t)rows * groups);
    float *x = (float *)malloc(cols * 4), *oh = (float *)malloc(rows * 4), *od = (float *)malloc(rows * 4);
    static const uint8_t codes[4] = {0x38, 0x30, 0x40, 0x3c}; /* e4m3 1.0,0.5,2.0,1.5 */
    for (uint32_t r = 0; r < rows; r++) for (uint32_t k = 0; k < cols; k++) {
        uint8_t n = (uint8_t)(((r * 131u + k * 17u + 3u) >> 1) & 0xf);
        uint64_t i = (uint64_t)r * cols + k;
        if (i & 1) w[i >> 1] = (uint8_t)((w[i >> 1] & 0x0f) | (n << 4));
        else       w[i >> 1] = (uint8_t)((w[i >> 1] & 0xf0) | n);
    }
    for (uint32_t r = 0; r < rows; r++) for (uint32_t g = 0; g < groups; g++) bs[r * groups + g] = codes[(r + g) % 4];
    for (uint32_t k = 0; k < cols; k++) x[k] = (float)((int)(k % 7u) - 3) * 0.5f;
    for (uint32_t r = 0; r < rows; r++) {
        double a = 0;
        for (uint32_t k = 0; k < cols; k++) {
            uint64_t i = (uint64_t)r * cols + k; uint8_t b = w[i >> 1];
            uint8_t n = (i & 1) ? (b >> 4) : (b & 0x0f);
            a += (double)e2m1(n) * (double)e4m3(bs[r * groups + (k / G)]) * (double)global * (double)x[k];
        }
        oh[r] = (float)a;
    }
    uint8_t *dw, *dbs; float *dx, *doo;
    if (cudaMalloc(&dw, nq) || cudaMalloc(&dbs, (size_t)rows * groups) ||
        cudaMalloc(&dx, cols * 4) || cudaMalloc(&doo, rows * 4)) { printf("axiom-nvfp4-smoke: cudaMalloc FAIL\n"); return 2; }
    cudaMemcpy(dw, w, nq, cudaMemcpyHostToDevice);
    cudaMemcpy(dbs, bs, (size_t)rows * groups, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, cols * 4, cudaMemcpyHostToDevice);
    int rc = axiom_cuda_e2m1_nvfp4_matvec_f32(0, dw, dbs, global, dx, doo, rows, cols);
    if (rc != AXIOM_OK) { printf("axiom-nvfp4-smoke: matvec rc=%d FAIL\n", rc); return 1; }
    cudaMemcpy(od, doo, rows * 4, cudaMemcpyDeviceToHost);
    float mx = 0; for (uint32_t r = 0; r < rows; r++) { float d = fabsf(od[r] - oh[r]); if (d > mx) mx = d; }
    int ok = mx < 1.0e-3f;
    printf("axiom-nvfp4-smoke: rows=%u cols=%u group=%u max_abs=%.9g %s\n", rows, cols, G, mx, ok ? "OK" : "FAIL");
    cudaFree(dw); cudaFree(dbs); cudaFree(dx); cudaFree(doo);
#ifdef AXIOM_NVFP4_SMOKE_FUSED
    if (!ok) return 1;
    return fused_moe_section(argc >= 2 ? argv[1] : NULL);
#else
    (void)argc; (void)argv;
    return ok ? 0 : 1;
#endif
}
