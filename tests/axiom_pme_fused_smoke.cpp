/* axiom_pme_fused_smoke.cpp — oracle-parity + OFF-bit-identity smoke for the AXIOM PME
 * fused serving kernel (src/axiom_cuda_pme.cu).
 *
 * Loads the vectors dumped by tools/pme_fused_ref.py (argv[1] = vector dir) and gates:
 *   G1  NVFP4: pme(pack=NULL) output MEMCMP-EQUAL to the plain production kernel
 *       axiom_cuda_e2m1_nvfp4_matvec_f32 (the house OFF==identical discipline — this is
 *       the proof, on real silicon, of the same-fp-ops argument in the kernel header).
 *   G2  NVFP4: pme(pack=NULL) vs f64 oracle y_off within TOL (base decode sanity).
 *   G3  NVFP4: pme(pack on) vs f64 oracle y_on within TOL (fused LoRA correctness).
 *   G4  indexed variant, expert slice 2/4 (other slots poisoned with 1e6): MEMCMP-EQUAL
 *       to G3's output (indexing picks exactly the right slice).
 *   G5  indexed variant, pack=NULL: MEMCMP-EQUAL to the plain kernel.
 *   G6  _device adapter (mirrored-buffer ABI) with a nonzero input offset: MEMCMP-EQUAL
 *       to G3's output.
 *   G7  _device adapter bounds: undersized out span MUST return AXIOM_ERR_INVALID_ARGUMENT.
 *   G8  raw entry pack-pointer mismatch (A set, B NULL) MUST return AXIOM_ERR_INVALID_ARGUMENT.
 *   G9  FP8: pme(pack=NULL) MEMCMP-EQUAL to plain axiom_cuda_fp8_e4m3_e8m0_matvec_f32.
 *   G10 FP8: pme(pack=NULL) vs y_off within TOL.
 *   G11 FP8: pme(pack on) vs y_on within TOL.
 * Every input file size is validated byte-exactly against the meta before upload; every
 * CUDA allocation/copy return code is checked. Exit 0 = all gates pass; 1 = a gate
 * failed; 2 = setup failure (files/CUDA). Build: `make pme-fused-smoke` (links
 * build/axiom_cuda_pme.o next to libaxiom.so — see the Makefile append-block). */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "axiom/axiom.h"

/* Plain production kernels (compiled into libaxiom from src/axiom_cuda_nvfp4.cu; raw
 * entries are not declared in axiom.h — local decls, same pattern as axiom_nvfp4_smoke). */
extern "C" int axiom_cuda_e2m1_nvfp4_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale, float global_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);
extern "C" int axiom_cuda_fp8_e4m3_e8m0_matvec_f32(
        int device, const uint8_t *weight, const uint8_t *block_scale,
        const float *input, float *out, uint32_t rows, uint32_t cols);

/* Mirror of the opaque backend buffer handle consumed by the _device adapters (layout
 * contract documented in src/axiom_cuda_pme.cu / src/axiom_cuda.cu:33-37). */
struct pme_buf_view { int device; void *ptr; uint64_t bytes; };

#define TOL 2.0e-3f
#define EXPERTS 4u
#define SLOT 2u

static int g_fail = 0;
static void gate(const char *name, int ok) {
    printf("axiom-pme-fused-smoke: %-34s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) g_fail = 1;
}

static void *load_exact(const char *dir, const char *name, uint64_t expect) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { printf("axiom-pme-fused-smoke: open %s FAIL\n", path); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { printf("axiom-pme-fused-smoke: seek %s FAIL\n", path); exit(2); }
    long sz = ftell(f);
    if (sz < 0 || (uint64_t)sz != expect) {
        printf("axiom-pme-fused-smoke: %s size=%ld expect=%llu FAIL\n",
               path, sz, (unsigned long long)expect);
        exit(2);
    }
    rewind(f);
    void *buf = malloc(expect);
    if (!buf || fread(buf, 1, expect, f) != expect) {
        printf("axiom-pme-fused-smoke: read %s FAIL\n", path); exit(2);
    }
    fclose(f);
    return buf;
}

static void *dupload(const void *host, uint64_t bytes) {
    void *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess ||
        cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("axiom-pme-fused-smoke: cudaMalloc/Memcpy(%llu) FAIL\n", (unsigned long long)bytes);
        exit(2);
    }
    return d;
}

static float *dout(uint64_t bytes) {
    void *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess ||
        cudaMemset(d, 0xCD, bytes) != cudaSuccess) { /* poison so a non-writing kernel fails loudly */
        printf("axiom-pme-fused-smoke: cudaMalloc out(%llu) FAIL\n", (unsigned long long)bytes);
        exit(2);
    }
    return (float *)d;
}

static void ddownload(float *host, const float *dev, uint64_t bytes) {
    if (cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("axiom-pme-fused-smoke: download FAIL\n"); exit(2);
    }
}

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
    float mx = 0.0f;
    for (uint32_t i = 0; i < n; ++i) { float d = fabsf(a[i] - b[i]); if (d > mx) mx = d; }
    return mx;
}

int main(int argc, char **argv) {
    if (argc != 2) { printf("usage: axiom-pme-fused-smoke <vector_dir>\n"); return 2; }
    const char *dir = argv[1];

    uint32_t nv_rows, nv_cols, nv_rank, f8_rows, f8_cols, f8_rank;
    float nv_global, nv_pscale, f8_pscale;
    {
        char path[1024]; snprintf(path, sizeof(path), "%s/pme_meta.txt", dir);
        FILE *f = fopen(path, "rb");
        if (!f || fscanf(f, "%u %u %u %f %f %u %u %u %f",
                         &nv_rows, &nv_cols, &nv_rank, &nv_global, &nv_pscale,
                         &f8_rows, &f8_cols, &f8_rank, &f8_pscale) != 9) {
            printf("axiom-pme-fused-smoke: meta %s FAIL\n", path); return 2;
        }
        fclose(f);
    }
    if (nv_rows == 0 || nv_cols == 0 || (nv_cols % 16u) != 0u ||
        nv_rank == 0 || nv_rank > AXIOM_PME_MAX_RANK ||
        f8_rows == 0 || f8_cols == 0 || (f8_rows % 128u) != 0u || (f8_cols % 128u) != 0u ||
        f8_rank == 0 || f8_rank > AXIOM_PME_MAX_RANK) {
        printf("axiom-pme-fused-smoke: bad meta values FAIL\n"); return 2;
    }
    printf("axiom-pme-fused-smoke: nv %ux%u rank=%u gs=%g ps=%g | fp8 %ux%u rank=%u ps=%g tol=%g\n",
           nv_rows, nv_cols, nv_rank, nv_global, nv_pscale,
           f8_rows, f8_cols, f8_rank, f8_pscale, (double)TOL);

    /* ---- load host vectors (byte-exact size validation) ---- */
    const uint64_t nv_w_b = (uint64_t)nv_rows * (nv_cols / 2u);
    const uint64_t nv_bs_b = (uint64_t)nv_rows * (nv_cols / 16u);
    const uint64_t nv_x_b = (uint64_t)nv_cols * 4u, nv_y_b = (uint64_t)nv_rows * 4u;
    const uint64_t nv_a_b = (uint64_t)nv_rank * nv_cols * 4u;
    const uint64_t nv_b_b = (uint64_t)nv_rows * nv_rank * 4u;
    uint8_t *h_nv_w = (uint8_t *)load_exact(dir, "nv_w.bin", nv_w_b);
    uint8_t *h_nv_bs = (uint8_t *)load_exact(dir, "nv_bs.bin", nv_bs_b);
    float *h_nv_x = (float *)load_exact(dir, "nv_x.bin", nv_x_b);
    float *h_nv_a = (float *)load_exact(dir, "nv_a.bin", nv_a_b);
    float *h_nv_b = (float *)load_exact(dir, "nv_b.bin", nv_b_b);
    float *h_nv_yoff = (float *)load_exact(dir, "nv_yoff.bin", nv_y_b);
    float *h_nv_yon = (float *)load_exact(dir, "nv_yon.bin", nv_y_b);

    const uint64_t f8_w_b = (uint64_t)f8_rows * f8_cols;
    const uint64_t f8_s_b = (uint64_t)(f8_rows / 128u) * (f8_cols / 128u);
    const uint64_t f8_x_b = (uint64_t)f8_cols * 4u, f8_y_b = (uint64_t)f8_rows * 4u;
    const uint64_t f8_a_b = (uint64_t)f8_rank * f8_cols * 4u;
    const uint64_t f8_b_b = (uint64_t)f8_rows * f8_rank * 4u;
    uint8_t *h_f8_w = (uint8_t *)load_exact(dir, "f8_w.bin", f8_w_b);
    uint8_t *h_f8_s = (uint8_t *)load_exact(dir, "f8_s.bin", f8_s_b);
    float *h_f8_x = (float *)load_exact(dir, "f8_x.bin", f8_x_b);
    float *h_f8_a = (float *)load_exact(dir, "f8_a.bin", f8_a_b);
    float *h_f8_b = (float *)load_exact(dir, "f8_b.bin", f8_b_b);
    float *h_f8_yoff = (float *)load_exact(dir, "f8_yoff.bin", f8_y_b);
    float *h_f8_yon = (float *)load_exact(dir, "f8_yon.bin", f8_y_b);

    /* ---- device uploads ---- */
    uint8_t *d_nv_w = (uint8_t *)dupload(h_nv_w, nv_w_b);
    uint8_t *d_nv_bs = (uint8_t *)dupload(h_nv_bs, nv_bs_b);
    float *d_nv_x = (float *)dupload(h_nv_x, nv_x_b);
    float *d_nv_a = (float *)dupload(h_nv_a, nv_a_b);
    float *d_nv_b = (float *)dupload(h_nv_b, nv_b_b);
    float *d_o_plain = dout(nv_y_b), *d_o_off = dout(nv_y_b), *d_o_on = dout(nv_y_b);
    float *d_o_idx = dout(nv_y_b), *d_o_idxoff = dout(nv_y_b), *d_o_dev = dout(nv_y_b);

    float *h_plain = (float *)malloc(nv_y_b), *h_off = (float *)malloc(nv_y_b);
    float *h_on = (float *)malloc(nv_y_b), *h_idx = (float *)malloc(nv_y_b);
    float *h_idxoff = (float *)malloc(nv_y_b), *h_dev = (float *)malloc(nv_y_b);
    if (!h_plain || !h_off || !h_on || !h_idx || !h_idxoff || !h_dev) {
        printf("axiom-pme-fused-smoke: malloc FAIL\n"); return 2;
    }

    int rc = axiom_cuda_e2m1_nvfp4_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
                                              d_nv_x, d_o_plain, nv_rows, nv_cols);
    if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: plain nvfp4 rc=%d FAIL\n", rc); return 2; }
    ddownload(h_plain, d_o_plain, nv_y_b);

    /* G1: pack=NULL bit-identity vs plain kernel */
    rc = axiom_cuda_e2m1_nvfp4_pme_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
            NULL, NULL, 0u, 0.0f, d_nv_x, d_o_off, nv_rows, nv_cols);
    if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: pme off rc=%d FAIL\n", rc); return 2; }
    ddownload(h_off, d_o_off, nv_y_b);
    gate("G1 nvfp4 off bit-identity", memcmp(h_plain, h_off, nv_y_b) == 0);

    /* G2: base decode sanity vs f64 oracle */
    float mx = max_abs_diff(h_off, h_nv_yoff, nv_rows);
    printf("axiom-pme-fused-smoke:   nvfp4 off max_abs=%.9g\n", mx);
    gate("G2 nvfp4 off vs oracle", mx < TOL);

    /* G3: fused pack correctness vs f64 oracle */
    rc = axiom_cuda_e2m1_nvfp4_pme_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
            d_nv_a, d_nv_b, nv_rank, nv_pscale, d_nv_x, d_o_on, nv_rows, nv_cols);
    if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: pme on rc=%d FAIL\n", rc); return 2; }
    ddownload(h_on, d_o_on, nv_y_b);
    mx = max_abs_diff(h_on, h_nv_yon, nv_rows);
    printf("axiom-pme-fused-smoke:   nvfp4 on  max_abs=%.9g\n", mx);
    gate("G3 nvfp4 fused pack vs oracle", mx < TOL);
    gate("G3b nvfp4 pack actually fires", memcmp(h_on, h_off, nv_y_b) != 0);

    /* G4/G5: indexed all-experts layout, slot SLOT of EXPERTS; other slots poisoned. */
    {
        const uint64_t aall_b = (uint64_t)EXPERTS * nv_a_b, ball_b = (uint64_t)EXPERTS * nv_b_b;
        float *h_aall = (float *)malloc(aall_b), *h_ball = (float *)malloc(ball_b);
        if (!h_aall || !h_ball) { printf("axiom-pme-fused-smoke: malloc all FAIL\n"); return 2; }
        for (uint64_t i = 0; i < aall_b / 4u; ++i) h_aall[i] = 1.0e6f;
        for (uint64_t i = 0; i < ball_b / 4u; ++i) h_ball[i] = 1.0e6f;
        memcpy(h_aall + (uint64_t)SLOT * nv_rank * nv_cols, h_nv_a, nv_a_b);
        memcpy(h_ball + (uint64_t)SLOT * nv_rows * nv_rank, h_nv_b, nv_b_b);
        float *d_aall = (float *)dupload(h_aall, aall_b);
        float *d_ball = (float *)dupload(h_ball, ball_b);
        rc = axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
                d_aall, d_ball, EXPERTS, SLOT, nv_rank, nv_pscale, d_nv_x, d_o_idx, nv_rows, nv_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: pme indexed rc=%d FAIL\n", rc); return 2; }
        ddownload(h_idx, d_o_idx, nv_y_b);
        gate("G4 indexed slice == direct pack", memcmp(h_idx, h_on, nv_y_b) == 0);

        rc = axiom_cuda_e2m1_nvfp4_pme_indexed_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
                NULL, NULL, 0u, 0u, 0u, 0.0f, d_nv_x, d_o_idxoff, nv_rows, nv_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: pme indexed off rc=%d FAIL\n", rc); return 2; }
        ddownload(h_idxoff, d_o_idxoff, nv_y_b);
        gate("G5 indexed off bit-identity", memcmp(h_idxoff, h_plain, nv_y_b) == 0);
        cudaFree(d_aall); cudaFree(d_ball); free(h_aall); free(h_ball);
    }

    /* G6/G7: _device adapters over the mirrored-buffer ABI. Input placed at a nonzero
     * (64-byte) offset inside a padded allocation to exercise ptr+offset plumbing. */
    {
        uint8_t *d_x_pad = NULL;
        if (cudaMalloc(&d_x_pad, 64u + nv_x_b) != cudaSuccess ||
            cudaMemcpy(d_x_pad + 64, h_nv_x, nv_x_b, cudaMemcpyHostToDevice) != cudaSuccess) {
            printf("axiom-pme-fused-smoke: pad alloc FAIL\n"); return 2;
        }
        pme_buf_view bw = { 0, d_nv_w, nv_w_b };
        pme_buf_view bs = { 0, d_nv_bs, nv_bs_b };
        pme_buf_view ba = { 0, d_nv_a, nv_a_b };
        pme_buf_view bb = { 0, d_nv_b, nv_b_b };
        pme_buf_view bx = { 0, d_x_pad, 64u + nv_x_b };
        pme_buf_view bo = { 0, d_o_dev, nv_y_b };
        rc = axiom_cuda_e2m1_nvfp4_pme_matvec_f32_device(NULL,
                &bw, 0, &bs, 0, nv_global, &ba, 0, &bb, 0, nv_rank, nv_pscale,
                &bx, 64u, &bo, 0, nv_rows, nv_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: device adapter rc=%d FAIL\n", rc); return 2; }
        ddownload(h_dev, d_o_dev, nv_y_b);
        gate("G6 device adapter offset parity", memcmp(h_dev, h_on, nv_y_b) == 0);

        pme_buf_view bo_small = { 0, d_o_dev, nv_y_b - 4u }; /* one float short */
        rc = axiom_cuda_e2m1_nvfp4_pme_matvec_f32_device(NULL,
                &bw, 0, &bs, 0, nv_global, &ba, 0, &bb, 0, nv_rank, nv_pscale,
                &bx, 64u, &bo_small, 0, nv_rows, nv_cols);
        gate("G7 device adapter bounds reject", rc == AXIOM_ERR_INVALID_ARGUMENT);
        cudaFree(d_x_pad);
    }

    /* G8: pack pointer mismatch must be rejected (raw entry). */
    rc = axiom_cuda_e2m1_nvfp4_pme_matvec_f32(0, d_nv_w, d_nv_bs, nv_global,
            d_nv_a, NULL, nv_rank, nv_pscale, d_nv_x, d_o_on, nv_rows, nv_cols);
    gate("G8 pack mismatch reject", rc == AXIOM_ERR_INVALID_ARGUMENT);

    /* ---- FP8 base ---- */
    {
        uint8_t *d_w = (uint8_t *)dupload(h_f8_w, f8_w_b);
        uint8_t *d_s = (uint8_t *)dupload(h_f8_s, f8_s_b);
        float *d_x = (float *)dupload(h_f8_x, f8_x_b);
        float *d_a = (float *)dupload(h_f8_a, f8_a_b);
        float *d_b = (float *)dupload(h_f8_b, f8_b_b);
        float *d_p = dout(f8_y_b), *d_off = dout(f8_y_b), *d_on = dout(f8_y_b);
        float *hp = (float *)malloc(f8_y_b), *ho = (float *)malloc(f8_y_b), *hn = (float *)malloc(f8_y_b);
        if (!hp || !ho || !hn) { printf("axiom-pme-fused-smoke: malloc f8 FAIL\n"); return 2; }

        rc = axiom_cuda_fp8_e4m3_e8m0_matvec_f32(0, d_w, d_s, d_x, d_p, f8_rows, f8_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: plain fp8 rc=%d FAIL\n", rc); return 2; }
        ddownload(hp, d_p, f8_y_b);

        rc = axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32(0, d_w, d_s,
                NULL, NULL, 0u, 0.0f, d_x, d_off, f8_rows, f8_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: fp8 pme off rc=%d FAIL\n", rc); return 2; }
        ddownload(ho, d_off, f8_y_b);
        gate("G9 fp8 off bit-identity", memcmp(hp, ho, f8_y_b) == 0);

        mx = max_abs_diff(ho, h_f8_yoff, f8_rows);
        printf("axiom-pme-fused-smoke:   fp8 off max_abs=%.9g\n", mx);
        gate("G10 fp8 off vs oracle", mx < TOL);

        rc = axiom_cuda_fp8_e4m3_e8m0_pme_matvec_f32(0, d_w, d_s,
                d_a, d_b, f8_rank, f8_pscale, d_x, d_on, f8_rows, f8_cols);
        if (rc != AXIOM_OK) { printf("axiom-pme-fused-smoke: fp8 pme on rc=%d FAIL\n", rc); return 2; }
        ddownload(hn, d_on, f8_y_b);
        mx = max_abs_diff(hn, h_f8_yon, f8_rows);
        printf("axiom-pme-fused-smoke:   fp8 on  max_abs=%.9g\n", mx);
        gate("G11 fp8 fused pack vs oracle", mx < TOL);

        cudaFree(d_w); cudaFree(d_s); cudaFree(d_x); cudaFree(d_a); cudaFree(d_b);
        cudaFree(d_p); cudaFree(d_off); cudaFree(d_on);
        free(hp); free(ho); free(hn);
    }

    cudaFree(d_nv_w); cudaFree(d_nv_bs); cudaFree(d_nv_x); cudaFree(d_nv_a); cudaFree(d_nv_b);
    cudaFree(d_o_plain); cudaFree(d_o_off); cudaFree(d_o_on);
    cudaFree(d_o_idx); cudaFree(d_o_idxoff); cudaFree(d_o_dev);
    free(h_nv_w); free(h_nv_bs); free(h_nv_x); free(h_nv_a); free(h_nv_b);
    free(h_nv_yoff); free(h_nv_yon);
    free(h_f8_w); free(h_f8_s); free(h_f8_x); free(h_f8_a); free(h_f8_b);
    free(h_f8_yoff); free(h_f8_yon);
    free(h_plain); free(h_off); free(h_on); free(h_idx); free(h_idxoff); free(h_dev);

    printf("axiom-pme-fused-smoke: %s\n", g_fail ? "FAIL" : "ALL GATES PASS");
    return g_fail ? 1 : 0;
}
