/*
 * axiom_deltanet_device_smoke.cpp - validate the integrated Axiom DeltaNet device
 * path (axiom_runtime_deltanet_*_device via libaxiom + axiom_device_buffer) against the
 * golden vectors from tools/deltanet_ref.py. Same checks as the standalone kernel test
 * but exercising the real library API end of the wrappers/thunks.
 *
 *   python3 tools/deltanet_ref.py /tmp/deltanet_vec.bin
 *   make bin/axiom-deltanet-device-smoke CUDA_ARCH=sm_120
 *   ./bin/axiom-deltanet-device-smoke /tmp/deltanet_vec.bin
 */
#include "axiom/axiom.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *rd(FILE *f, size_t n){ float *p=(float*)malloc(n*4); if(fread(p,4,n,f)!=n){fprintf(stderr,"short read\n");exit(1);} return p; }
static float maxabs(const float *a, const float *b, size_t n){ float m=0; for(size_t i=0;i<n;i++){float d=fabsf(a[i]-b[i]); if(d>m)m=d;} return m; }
#define CHK(x) do{ int rc_=(x); if(rc_!=AXIOM_OK){ fprintf(stderr,"%s -> rc=%d (%s)\n",#x,rc_,axiom_status_string(rc_)); return 1;} }while(0)

static int expect_invalid(const char *name, int rc){
    if(rc != AXIOM_ERR_INVALID_ARGUMENT){
        fprintf(stderr,"[guard] %s -> rc=%d (%s), expected invalid_argument\n",name,rc,axiom_status_string(rc));
        return 0;
    }
    return 1;
}

static axiom_device_buffer *mk(axiom_runtime *rt, uint64_t bytes, const float *host){
    axiom_device_buffer *b=NULL;
    if(axiom_device_buffer_create(rt,&b,bytes)!=AXIOM_OK) return NULL;
    if(host) axiom_device_buffer_upload(b,0,host,bytes);
    else { void *z=calloc(1,bytes); axiom_device_buffer_upload(b,0,z,bytes); free(z); }
    return b;
}

int main(int argc, char **argv){
    const char *path = argc>1?argv[1]:"/tmp/deltanet_vec.bin";
    FILE *f=fopen(path,"rb"); if(!f){perror("open");return 1;}
    int32_t hdr[5]; if(fread(hdr,4,5,f)!=5){fprintf(stderr,"hdr\n");return 1;}
    int seq=hdr[0],H=hdr[1],hd=hdr[2],C=hdr[3],K=hdr[4];
    printf("[dev] seq=%d vheads=%d hd=%d conv_dim=%d kernel=%d\n",seq,H,hd,C,K);
    float *conv_w=rd(f,(size_t)C*K),*conv_in=rd(f,(size_t)seq*C),*conv_ref=rd(f,(size_t)seq*C);
    float *rq=rd(f,(size_t)seq*H*hd),*rk=rd(f,(size_t)seq*H*hd),*rv=rd(f,(size_t)seq*H*hd);
    float *rbeta=rd(f,(size_t)seq*H),*rg=rd(f,(size_t)seq*H);
    float *S0=rd(f,(size_t)H*hd*hd),*out_ref=rd(f,(size_t)seq*H*hd),*Sfin_ref=rd(f,(size_t)H*hd*hd);
    float *znorm=rd(f,(size_t)hd),*zg=rd(f,(size_t)seq*H*hd),*gated_ref=rd(f,(size_t)seq*H*hd);
    fclose(f);

    axiom_config cfg; memset(&cfg,0,sizeof cfg); cfg.abi_version=AXIOM_ABI_VERSION; cfg.backend=AXIOM_BACKEND_CUDA;
    axiom_runtime *rt=NULL; CHK(axiom_runtime_create(&rt,&cfg));

    /* conv1d */
    axiom_device_buffer *d_in=mk(rt,(uint64_t)C*4,NULL),*d_w=mk(rt,(uint64_t)C*K*4,conv_w),*d_ring=mk(rt,(uint64_t)C*3*4,NULL),*d_co=mk(rt,(uint64_t)C*4,NULL);
    float *h_co=(float*)malloc((size_t)seq*C*4);
    for(int t=0;t<seq;t++){ CHK(axiom_device_buffer_upload(d_in,0,conv_in+(size_t)t*C,(uint64_t)C*4));
        CHK(axiom_runtime_deltanet_conv1d_silu_f32_device(rt,d_in,0,d_w,0,d_ring,0,d_co,0,(uint32_t)C));
        CHK(axiom_device_buffer_download(d_co,0,h_co+(size_t)t*C,(uint64_t)C*4)); }
    float e_conv=maxabs(h_co,conv_ref,(size_t)seq*C);

    /* recurrence (nkh=vheads here) */
    uint64_t reg=(uint64_t)H*hd*4;
    axiom_device_buffer *d_qkv=mk(rt,3*reg,NULL),*d_g=mk(rt,(uint64_t)H*4,NULL),*d_b=mk(rt,(uint64_t)H*4,NULL),*d_S=mk(rt,(uint64_t)H*hd*hd*4,S0),*d_o=mk(rt,reg,NULL);
    float *h_out=(float*)malloc((size_t)seq*H*hd*4);
    for(int t=0;t<seq;t++){
        CHK(axiom_device_buffer_upload(d_qkv,0,    rq+(size_t)t*H*hd,reg));
        CHK(axiom_device_buffer_upload(d_qkv,reg,  rk+(size_t)t*H*hd,reg));
        CHK(axiom_device_buffer_upload(d_qkv,2*reg,rv+(size_t)t*H*hd,reg));
        CHK(axiom_device_buffer_upload(d_g,0,rg+(size_t)t*H,(uint64_t)H*4));
        CHK(axiom_device_buffer_upload(d_b,0,rbeta+(size_t)t*H,(uint64_t)H*4));
        CHK(axiom_runtime_deltanet_recurrence_f32_device(rt,d_qkv,0,reg,2*reg,d_g,0,d_b,0,d_S,0,d_o,0,(uint32_t)H,(uint32_t)H,(uint32_t)hd));
        CHK(axiom_device_buffer_download(d_o,0,h_out+(size_t)t*H*hd,reg)); }
    float *h_S=(float*)malloc((size_t)H*hd*hd*4); CHK(axiom_device_buffer_download(d_S,0,h_S,(uint64_t)H*hd*hd*4));
    float e_out=maxabs(h_out,out_ref,(size_t)seq*H*hd), e_S=maxabs(h_S,Sfin_ref,(size_t)H*hd*hd);

    /* gated norm (feed reference recurrence output) */
    axiom_device_buffer *d_or=mk(rt,reg,NULL),*d_zn=mk(rt,(uint64_t)hd*4,znorm),*d_z=mk(rt,reg,NULL),*d_gn=mk(rt,reg,NULL);
    float *h_gn=(float*)malloc((size_t)seq*H*hd*4);
    for(int t=0;t<seq;t++){ CHK(axiom_device_buffer_upload(d_or,0,out_ref+(size_t)t*H*hd,reg));
        CHK(axiom_device_buffer_upload(d_z,0,zg+(size_t)t*H*hd,reg));
        CHK(axiom_runtime_deltanet_gated_norm_f32_device(rt,d_or,0,d_zn,0,d_z,0,d_gn,0,(uint32_t)H,(uint32_t)hd,1e-6f));
        CHK(axiom_device_buffer_download(d_gn,0,h_gn+(size_t)t*H*hd,reg)); }
    float e_gn=maxabs(h_gn,gated_ref,(size_t)seq*H*hd);

    printf("[dev] max|conv  - ref| = %.3e\n",e_conv);
    printf("[dev] max|recur - ref| = %.3e\n",e_out);
    printf("[dev] max|state - ref| = %.3e\n",e_S);
    printf("[dev] max|gnorm - ref| = %.3e\n",e_gn);
    int guard_pass = 1;
    guard_pass &= expect_invalid("conv short span",
        axiom_runtime_deltanet_conv1d_silu_f32_device(rt,d_in,0,d_w,0,d_ring,0,d_co,0,(uint32_t)C+1u));
    guard_pass &= expect_invalid("recurrence incompatible heads",
        axiom_runtime_deltanet_recurrence_f32_device(rt,d_qkv,0,reg,2*reg,d_g,0,d_b,0,d_S,0,d_o,0,(uint32_t)H,(uint32_t)H+1u,(uint32_t)hd));
    guard_pass &= expect_invalid("gated norm non power-of-two head_dim",
        axiom_runtime_deltanet_gated_norm_f32_device(rt,d_or,0,d_zn,0,d_z,0,d_gn,0,(uint32_t)H,(uint32_t)hd-1u,1e-6f));
    printf("[guard] %s invalid-argument checks\n", guard_pass?"PASS":"FAIL");
    float tol=2e-3f; int pass=e_conv<tol&&e_out<tol&&e_S<tol&&e_gn<tol;
    printf("[dev] %s (tol=%.0e) - Axiom lib path (axiom_runtime_deltanet_*_device)\n",pass?"PASS":"FAIL",tol);
    axiom_runtime_destroy(rt);
    return (pass && guard_pass)?0:1;
}
