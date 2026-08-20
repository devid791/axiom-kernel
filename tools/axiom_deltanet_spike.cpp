/*
 * axiom_deltanet_spike.cpp - host validation of the Gated DeltaNet decode step
 * (causal depthwise conv1d + SiLU, and the gated delta-rule recurrence) against
 * the numpy reference in tools/deltanet_ref.py. Model-free, GPU-free: proves the
 * #1-risk recurrence/conv math is correct before touching the real 35B model.
 *
 *   python3 tools/deltanet_ref.py /tmp/deltanet_vec.bin
 *   c++ -O2 -o bin/axiom-deltanet-spike tools/axiom_deltanet_spike.cpp -lm
 *   ./bin/axiom-deltanet-spike /tmp/deltanet_vec.bin
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static float *rd(FILE *f, size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (fread(p, sizeof(float), n, f) != n) { fprintf(stderr, "short read\n"); exit(1); }
    return p;
}
static float siluf(float x) { return x / (1.0f + expf(-x)); }
static float maxabs(const float *a, const float *b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/tmp/deltanet_vec.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open vec"); return 1; }
    int32_t hdr[5];
    if (fread(hdr, sizeof(int32_t), 5, f) != 5) { fprintf(stderr, "bad header\n"); return 1; }
    int seq = hdr[0], H = hdr[1], hd = hdr[2], C = hdr[3], K = hdr[4];
    printf("[spike] seq=%d vheads=%d hd=%d conv_dim=%d kernel=%d\n", seq, H, hd, C, K);

    float *conv_w   = rd(f, (size_t)C * K);
    float *conv_in  = rd(f, (size_t)seq * C);
    float *conv_ref = rd(f, (size_t)seq * C);
    float *rq    = rd(f, (size_t)seq * H * hd);
    float *rk    = rd(f, (size_t)seq * H * hd);
    float *rv    = rd(f, (size_t)seq * H * hd);
    float *rbeta = rd(f, (size_t)seq * H);
    float *rg    = rd(f, (size_t)seq * H);
    float *S0    = rd(f, (size_t)H * hd * hd);
    float *outref= rd(f, (size_t)seq * H * hd);
    float *Sref  = rd(f, (size_t)H * hd * hd);
    fclose(f);

    /* ---- causal depthwise conv1d (k=K, no bias) + SiLU ---- */
    float *conv_c = (float *)calloc((size_t)seq * C, sizeof(float));
    for (int t = 0; t < seq; t++)
        for (int c = 0; c < C; c++) {
            float acc = 0.0f;
            for (int j = 0; j < K; j++) {
                int ti = t - (K - 1) + j;            /* causal: x[t-3+j] */
                float x = (ti >= 0) ? conv_in[(size_t)ti * C + c] : 0.0f;
                acc += conv_w[(size_t)c * K + j] * x;
            }
            conv_c[(size_t)t * C + c] = siluf(acc);
        }

    /* ---- gated delta-rule recurrence (per v-head) ---- */
    float *out_c = (float *)calloc((size_t)seq * H * hd, sizeof(float));
    float *S = (float *)malloc((size_t)H * hd * hd * sizeof(float));
    for (size_t i = 0; i < (size_t)H * hd * hd; i++) S[i] = S0[i];
    float *qq = (float *)malloc(hd * 4), *kk = (float *)malloc(hd * 4), *kvm = (float *)malloc(hd * 4), *dl = (float *)malloc(hd * 4);
    const float scale = 1.0f / sqrtf((float)hd);
    for (int t = 0; t < seq; t++)
        for (int h = 0; h < H; h++) {
            const float *q = rq + ((size_t)t * H + h) * hd;
            const float *k = rk + ((size_t)t * H + h) * hd;
            const float *v = rv + ((size_t)t * H + h) * hd;
            /* l2norm q,k (eps inside rsqrt); q *= 1/sqrt(hd) */
            float sq = 0.0f, sk = 0.0f;
            for (int i = 0; i < hd; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
            float iq = 1.0f / sqrtf(sq + 1e-6f), ik = 1.0f / sqrtf(sk + 1e-6f);
            for (int i = 0; i < hd; i++) { qq[i] = q[i] * iq * scale; kk[i] = k[i] * ik; }
            float gt = expf(rg[(size_t)t * H + h]);
            float bt = rbeta[(size_t)t * H + h];
            float *Sh = S + (size_t)h * hd * hd;     /* [k*hd + v] */
            for (size_t i = 0; i < (size_t)hd * hd; i++) Sh[i] *= gt;          /* (a) decay */
            for (int vv = 0; vv < hd; vv++) {                                  /* (b) read key */
                float acc = 0.0f;
                for (int kk2 = 0; kk2 < hd; kk2++) acc += Sh[(size_t)kk2 * hd + vv] * kk[kk2];
                kvm[vv] = acc;
            }
            for (int vv = 0; vv < hd; vv++) dl[vv] = (v[vv] - kvm[vv]) * bt;   /* (c) delta */
            for (int kk2 = 0; kk2 < hd; kk2++) {                              /* (d) write outer */
                float kv = kk[kk2];
                float *row = Sh + (size_t)kk2 * hd;
                for (int vv = 0; vv < hd; vv++) row[vv] += kv * dl[vv];
            }
            float *o = out_c + ((size_t)t * H + h) * hd;                       /* (e) read query */
            for (int vv = 0; vv < hd; vv++) {
                float acc = 0.0f;
                for (int kk2 = 0; kk2 < hd; kk2++) acc += Sh[(size_t)kk2 * hd + vv] * qq[kk2];
                o[vv] = acc;
            }
        }

    float ec = maxabs(conv_c, conv_ref, (size_t)seq * C);
    float eo = maxabs(out_c, outref, (size_t)seq * H * hd);
    float es = maxabs(S, Sref, (size_t)H * hd * hd);
    printf("[spike] max|conv - ref|      = %.3e\n", ec);
    printf("[spike] max|out  - ref|      = %.3e\n", eo);
    printf("[spike] max|state- ref|      = %.3e\n", es);
    float tol = 2e-3f;
    int pass = (ec < tol && eo < tol && es < tol);
    printf("[spike] %s (tol=%.0e)\n", pass ? "PASS" : "FAIL", tol);
    return pass ? 0 : 1;
}
