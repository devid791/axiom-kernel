/*
 * axiom_deltanet_kernels.cu — Gated DeltaNet CUDA kernels (v1, fp32 state in global),
 * developed standalone and validated against the golden vectors from tools/deltanet_ref.py
 * (which is itself bit-validated vs transformers + llama.cpp). Once green here, the kernel
 * bodies + extern-C wrappers move into src/axiom_cuda.cu (the Axiom runtime).
 *
 *   python3 tools/deltanet_ref.py /tmp/deltanet_vec.bin
 *   nvcc -O3 -arch=sm_120 -o bin/axiom-deltanet-kernels tools/axiom_deltanet_kernels.cu
 *   ./bin/axiom-deltanet-kernels /tmp/deltanet_vec.bin
 *
 * Three kernels (match tools/axiom_qwen3next.cpp deltanet() line-for-line):
 *   1) conv1d depthwise k=4 + SiLU, persistent ring [conv_dim][3]
 *   2) gated delta recurrence (decay->read-key->delta->write-outer->read-query), S[k*HD+v]
 *   3) gated RMSNorm: rmsnorm_per_head(o) * wnorm * silu(z)
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

/* ---- Kernel 1: causal depthwise conv1d (k=4, no bias) + SiLU, persistent ring ---- */
__global__ static void deltanet_conv1d_silu_kernel(
        const float *__restrict__ in, const float *__restrict__ w,
        float *__restrict__ ring, float *__restrict__ out, uint32_t conv_dim) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float acc = w[c*4+0]*ring[c*3+0] + w[c*4+1]*ring[c*3+1] + w[c*4+2]*ring[c*3+2] + w[c*4+3]*in[c];
    out[c] = acc / (1.0f + expf(-acc));               /* SiLU */
    ring[c*3+0] = ring[c*3+1]; ring[c*3+1] = ring[c*3+2]; ring[c*3+2] = in[c];   /* shift ring */
}

/* ---- Kernel 2: gated delta recurrence (T=1 decode). grid=NVH, block=HD. thread=v-column ---- */
__global__ static void deltanet_recurrence_kernel(
        const float *__restrict__ qc, const float *__restrict__ kc, const float *__restrict__ vc,
        const float *__restrict__ g, const float *__restrict__ beta,
        float *__restrict__ state, float *__restrict__ out,
        uint32_t nkh, uint32_t hd) {
    const uint32_t h = blockIdx.x, v = threadIdx.x;
    if (v >= hd) return;
    const uint32_t kh = h % nkh;                       /* repeat = TILE (host-C h%16) */
    const float *qv = qc + (size_t)kh*hd;
    const float *kv = kc + (size_t)kh*hd;
    const float *vin = vc + (size_t)h*hd;
    float *Sh = state + (size_t)h*hd*hd;               /* S[k*hd + v], thread owns column v */
    extern __shared__ float smem[];
    float *qq = smem, *kk = smem + hd, *sums = smem + 2*hd;
    const float qval = qv[v], kval = kv[v], vreg = vin[v];
    /* L2-norm of q over hd (block reduce) */
    sums[v] = qval*qval; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) sums[v] += sums[v+s]; __syncthreads(); }
    const float iq = rsqrtf(sums[0] + 1e-6f); __syncthreads();
    sums[v] = kval*kval; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) sums[v] += sums[v+s]; __syncthreads(); }
    const float ik = rsqrtf(sums[0] + 1e-6f); __syncthreads();
    const float qscale = rsqrtf((float)hd);
    qq[v] = qval * iq * qscale;                        /* q l2-normed AND scaled */
    kk[v] = kval * ik;
    __syncthreads();
    const float gt = expf(g[h]), bt = beta[h];         /* kernel applies exp() only; g precomputed */
    /* each thread owns column v of S -> no cross-thread S dependency, no sync needed below */
    for (uint32_t k = 0; k < hd; ++k) Sh[k*hd+v] *= gt;                 /* (a) decay */
    float kvm = 0.0f;
    for (uint32_t k = 0; k < hd; ++k) kvm += Sh[k*hd+v] * kk[k];        /* (b) read-key */
    const float dl = (vreg - kvm) * bt;                                /* (c) delta */
    for (uint32_t k = 0; k < hd; ++k) Sh[k*hd+v] += kk[k] * dl;         /* (d) write-outer */
    float o = 0.0f;
    for (uint32_t k = 0; k < hd; ++k) o += Sh[k*hd+v] * qq[k];          /* (e) read-query */
    out[(size_t)h*hd+v] = o;
}

/* ---- Kernel 3: gated RMSNorm: rmsnorm_per_head(o) * wnorm * silu(z) ---- */
__global__ static void deltanet_gated_norm_kernel(
        const float *__restrict__ o, const float *__restrict__ wnorm, const float *__restrict__ z,
        float *__restrict__ out, uint32_t hd, float eps) {
    const uint32_t h = blockIdx.x, v = threadIdx.x;
    if (v >= hd) return;
    const float *oh = o + (size_t)h*hd;
    extern __shared__ float sums[];
    const float ov = oh[v];
    sums[v] = ov*ov; __syncthreads();
    for (uint32_t s = hd>>1; s; s >>= 1) { if (v < s) sums[v] += sums[v+s]; __syncthreads(); }
    const float inv = rsqrtf(sums[0]/(float)hd + eps);
    const float zv = z[(size_t)h*hd+v];
    out[(size_t)h*hd+v] = ov * inv * wnorm[v] * (zv / (1.0f + expf(-zv)));
}

/* ---------------- test harness ---------------- */
static float *rd(FILE *f, size_t n){ float *p=(float*)malloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} return p; }
static float *dev(const float *h, size_t n){ float *d; CK(cudaMalloc(&d,n*4)); if(h)CK(cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice)); else CK(cudaMemset(d,0,n*4)); return d; }
static float maxabs(const float *a, const float *b, size_t n){ float m=0; for(size_t i=0;i<n;i++){ float d=fabsf(a[i]-b[i]); if(d>m)m=d; } return m; }

int main(int argc, char **argv){
    const char *path = argc>1?argv[1]:"/tmp/deltanet_vec.bin";
    FILE *f = fopen(path,"rb"); if(!f){perror("open");return 1;}
    int32_t hdr[5]; if(fread(hdr,4,5,f)!=5){fprintf(stderr,"hdr\n");return 1;}
    int seq=hdr[0], H=hdr[1], hd=hdr[2], C=hdr[3], K=hdr[4];
    printf("[dnk] seq=%d vheads=%d hd=%d conv_dim=%d kernel=%d\n", seq,H,hd,C,K);
    float *conv_w=rd(f,(size_t)C*K), *conv_in=rd(f,(size_t)seq*C), *conv_ref=rd(f,(size_t)seq*C);
    float *rq=rd(f,(size_t)seq*H*hd), *rk=rd(f,(size_t)seq*H*hd), *rv=rd(f,(size_t)seq*H*hd);
    float *rbeta=rd(f,(size_t)seq*H), *rg=rd(f,(size_t)seq*H);
    float *S0=rd(f,(size_t)H*hd*hd), *out_ref=rd(f,(size_t)seq*H*hd), *Sfin_ref=rd(f,(size_t)H*hd*hd);
    float *znorm=rd(f,(size_t)hd), *zg=rd(f,(size_t)seq*H*hd), *gated_ref=rd(f,(size_t)seq*H*hd);
    fclose(f);

    /* --- conv1d --- */
    float *d_cw=dev(conv_w,(size_t)C*K), *d_cin=dev(conv_in,(size_t)seq*C), *d_cout=dev(0,(size_t)seq*C), *d_ring=dev(0,(size_t)C*3);
    for(int t=0;t<seq;t++) deltanet_conv1d_silu_kernel<<<(C+255)/256,256>>>(d_cin+(size_t)t*C, d_cw, d_ring, d_cout+(size_t)t*C, C);
    CK(cudaDeviceSynchronize());
    float *h_cout=(float*)malloc((size_t)seq*C*4); CK(cudaMemcpy(h_cout,d_cout,(size_t)seq*C*4,cudaMemcpyDeviceToHost));
    float e_conv=maxabs(h_cout,conv_ref,(size_t)seq*C);

    /* --- recurrence (nkh = vheads for this synthetic test) --- */
    float *d_q=dev(rq,(size_t)seq*H*hd), *d_k=dev(rk,(size_t)seq*H*hd), *d_v=dev(rv,(size_t)seq*H*hd);
    float *d_g=dev(rg,(size_t)seq*H), *d_b=dev(rbeta,(size_t)seq*H);
    float *d_S=dev(S0,(size_t)H*hd*hd), *d_out=dev(0,(size_t)seq*H*hd);
    size_t shmem = (size_t)3*hd*sizeof(float);
    for(int t=0;t<seq;t++)
        deltanet_recurrence_kernel<<<H, hd, shmem>>>(d_q+(size_t)t*H*hd, d_k+(size_t)t*H*hd, d_v+(size_t)t*H*hd,
            d_g+(size_t)t*H, d_b+(size_t)t*H, d_S, d_out+(size_t)t*H*hd, (uint32_t)H, (uint32_t)hd);
    CK(cudaDeviceSynchronize());
    float *h_out=(float*)malloc((size_t)seq*H*hd*4); CK(cudaMemcpy(h_out,d_out,(size_t)seq*H*hd*4,cudaMemcpyDeviceToHost));
    float *h_S=(float*)malloc((size_t)H*hd*hd*4); CK(cudaMemcpy(h_S,d_S,(size_t)H*hd*hd*4,cudaMemcpyDeviceToHost));
    float e_out=maxabs(h_out,out_ref,(size_t)seq*H*hd);
    float e_S=maxabs(h_S,Sfin_ref,(size_t)H*hd*hd);

    /* --- gated norm (feed the reference recurrence output) --- */
    float *d_oref=dev(out_ref,(size_t)seq*H*hd), *d_zn=dev(znorm,(size_t)hd), *d_z=dev(zg,(size_t)seq*H*hd), *d_gn=dev(0,(size_t)seq*H*hd);
    for(int t=0;t<seq;t++)
        deltanet_gated_norm_kernel<<<H, hd, (size_t)hd*sizeof(float)>>>(d_oref+(size_t)t*H*hd, d_zn, d_z+(size_t)t*H*hd, d_gn+(size_t)t*H*hd, (uint32_t)hd, 1e-6f);
    CK(cudaDeviceSynchronize());
    float *h_gn=(float*)malloc((size_t)seq*H*hd*4); CK(cudaMemcpy(h_gn,d_gn,(size_t)seq*H*hd*4,cudaMemcpyDeviceToHost));
    float e_gn=maxabs(h_gn,gated_ref,(size_t)seq*H*hd);

    printf("[dnk] max|conv  - ref| = %.3e\n", e_conv);
    printf("[dnk] max|recur - ref| = %.3e\n", e_out);
    printf("[dnk] max|state - ref| = %.3e\n", e_S);
    printf("[dnk] max|gnorm - ref| = %.3e\n", e_gn);
    float tol=2e-3f; int pass = e_conv<tol && e_out<tol && e_S<tol && e_gn<tol;
    printf("[dnk] %s (tol=%.0e)\n", pass?"PASS":"FAIL", tol);
    return pass?0:1;
}
