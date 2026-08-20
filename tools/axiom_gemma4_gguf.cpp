/*
 * axiom-gemma4-gguf — CPU reference Gemma-4 12B text-decoder runner (Q8_0 GGUF).
 *
 * Bit-faithful CPU forward for the gemma-4 text decoder, reading a caller-
 * supplied Q8_0 GGUF, so a greedy-token-parity gate against an independent
 * reference runner is bit-for-bit comparable (no
 * tolerance — Q8_0 is identical on both sides).
 *
 * Skeleton = tools/axiom_gemma.cpp (the gemma-3 GGUF runner): gguf_open/find/
 * tensor_ptr/load_q8/load_f32/f16_to_f32/skip_value, matvec_q8, embed_row,
 * rmsnorm(PLAIN weight, no +1), qk_norm, geglu_mul, SPM detok and the greedy
 * GFWD-macro driver are reused verbatim. Blueprint for the gemma-4 layer logic =
 * tools/axiom_gemma4.cpp (per-layer-type hd/n_kv/theta select, KVMAX=2048 strided
 * cache, V=memcpy(K) on global k==v layers, the 30*tanh(x/30) final soft-cap),
 * ported here from BF16 safetensors to Q8_0 GGUF. The independent reference
 * implementation is an external test input and is not part of this repository.
 *
 * GEMMA-4 SPECIFICS (vs gemma-3 / qwen):
 *   - attention SCALE = 1.0 (NOT 1/sqrt(head_dim)) — f_attention_scale=1.0.
 *   - per-layer type: GLOBAL (5,11,17,23,29,35,41,47) hd=512 n_kv=1(MQA) n_rot=512
 *     theta=1e6 wv ABSENT (V=raw K-projection, k==v sharing); SLIDING (rest)
 *     hd=256 n_kv=8(GQA group=2) n_rot=256 theta=1e4 wv present. head_count=16.
 *   - Q,K get per-head RMSNorm WITH weight; V gets BARE per-head RMSNorm (no
 *     weight) and NO rope.
 *   - RoPE NeoX: GLOBAL uses partial/proportional rope via the shared
 *     rope_freqs.weight (ff[0..63]=1 rotate first 64 pairs, ff[64..255]=1e30 pass
 *     through); SLIDING uses standard NeoX (ff=NULL, all n_rot/2 pairs rotate).
 *   - 4-norm sandwich: post_attention_norm on the ATTN OUTPUT then +resid;
 *     post_ffw_norm on the FFN OUTPUT then +resid. RMSNorm plain weight, eps=1e-6.
 *   - layer_output_scale: multiply the hidden by blk.L.layer_output_scale.weight[0]
 *     AFTER both residual adds.
 *   - embedding scale sqrt(3840); GeGLU gelu_tanh(gate)*up; tied lm_head; final
 *     logit soft-cap 30*tanh(x/30) over the whole vocab; NO attention soft-cap.
 *   - SWA window 1024 (inert for <1024-token prompts); each layer reads/writes its
 *     OWN KVMAX=2048-strided cache (shared_kv_layers=0).
 *
 * Two entry points:
 *   - axiom_gemma4_gguf_generate_ids(): callable from the manifest-runner/service.
 *   - main(): CLI (--ids a,b,c [--max N] [--score]), built unless AXIOM_GEMMA4_GGUF_NO_CLI_MAIN.
 * Pure CPU, self-contained (no libaxiom).
 *
 * --score (criterion #2, APPLES-TO-APPLES): teacher-forced perplexity over the
 * given ids, MIRRORING tools-oracle/llama_logprob.cpp bit-for-bit (per-position
 * float max/argmax, double sum of expf(logit-max), nll += -(logits[tok]-max-
 * log(sum_exp)), target = ids[t+1]). Because THIS runner reads the SAME Q8_0
 * GGUF the oracle loads, the diff is matched-quant — no bf16-vs-Q8 excuse.
 * One stdout line:  SCORE_NLL=.. SCORE_COUNT=.. SCORE_PPL=.. TF_TOP1_MATCH=k/N
 * In score mode the end-of-image/audio logit suppression is SKIPPED (the oracle
 * does not suppress; generation behavior is unchanged when --score is absent).
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { GGUF_U8=0, GGUF_I8=1, GGUF_U16=2, GGUF_I16=3, GGUF_U32=4, GGUF_I32=5,
       GGUF_F32=6, GGUF_BOOL=7, GGUF_STRING=8, GGUF_ARRAY=9, GGUF_U64=10,
       GGUF_I64=11, GGUF_F64=12 };
enum { GGML_F32=0, GGML_F16=1, GGML_Q8_0=8 };

#define G4_MAXLAYERS 96u

typedef struct { const char *p; uint64_t len; } strref;
typedef struct { strref name; uint32_t n_dims; uint64_t dims[4]; uint32_t type; uint64_t offset; } gtensor;
typedef struct {
    int fd; const uint8_t *base; size_t size; uint64_t data_start; uint32_t alignment;
    gtensor *tensors; uint32_t n_tensors;
    uint32_t n_layers, hidden, n_heads, ffn;
    uint32_t key_length, key_length_swa, n_rot_full, n_rot_swa, sliding_window, shared_kv_layers;
    float rope_theta, rope_theta_swa, rms_eps, final_softcap;
    uint32_t head_count_kv[G4_MAXLAYERS]; /* per-layer n_kv (INT32 array gemma4.attention.head_count_kv) */
    int      is_swa[G4_MAXLAYERS];        /* per-layer (BOOL array gemma4.attention.sliding_window_pattern; True=sliding) */
    int      have_kv_arr, have_swa_arr;
    uint32_t bos_id, eos_id, eot_id;
    strref *tokens; uint64_t n_tokens;
} gguf;

static uint32_t rd_u32(const uint8_t **c){ uint32_t v; memcpy(&v,*c,4); *c+=4; return v; }
static int32_t  rd_i32(const uint8_t **c){ int32_t v; memcpy(&v,*c,4); *c+=4; return v; }
static uint64_t rd_u64(const uint8_t **c){ uint64_t v; memcpy(&v,*c,8); *c+=8; return v; }
static float    rd_f32(const uint8_t **c){ float v; memcpy(&v,*c,4); *c+=4; return v; }
static strref   rd_str(const uint8_t **c){ strref s; s.len=rd_u64(c); s.p=(const char*)*c; *c+=s.len; return s; }
static int sref_eq(strref s, const char *lit){ size_t n=strlen(lit); return s.len==n && memcmp(s.p,lit,n)==0; }

static void skip_value(const uint8_t **c, uint32_t type){
    switch(type){
        case GGUF_U8: case GGUF_I8: case GGUF_BOOL: *c+=1; break;
        case GGUF_U16: case GGUF_I16: *c+=2; break;
        case GGUF_U32: case GGUF_I32: case GGUF_F32: *c+=4; break;
        case GGUF_U64: case GGUF_I64: case GGUF_F64: *c+=8; break;
        case GGUF_STRING: { uint64_t n=rd_u64(c); *c+=n; } break;
        case GGUF_ARRAY: { uint32_t et=rd_u32(c); uint64_t cnt=rd_u64(c); for(uint64_t i=0;i<cnt;i++) skip_value(c,et); } break;
        default: fprintf(stderr,"gguf: unknown value type %u\n",type); exit(1);
    }
}
static float f16_to_f32(uint16_t h){
    uint32_t sign=(uint32_t)(h&0x8000u)<<16, exp=(h>>10)&0x1Fu, man=h&0x3FFu, bits;
    if(exp==0){ if(man==0) bits=sign; else { exp=127-15+1; while((man&0x400u)==0){man<<=1;exp--;} man&=0x3FFu; bits=sign|(exp<<23)|(man<<13);} }
    else if(exp==0x1Fu) bits=sign|0x7F800000u|(man<<13);
    else bits=sign|((exp+(127-15))<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
static uint64_t align_up(uint64_t x, uint64_t a){ return (x+a-1)/a*a; }

static void gguf_open(gguf *g, const char *path){
    memset(g,0,sizeof(*g)); g->alignment=32;
    g->rope_theta=1000000.0f; g->rope_theta_swa=10000.0f; g->rms_eps=1e-6f; g->final_softcap=30.0f;
    g->key_length=512; g->key_length_swa=256; g->n_rot_full=512; g->n_rot_swa=256;
    g->sliding_window=1024; g->shared_kv_layers=0; g->n_heads=16;
    g->bos_id=2; g->eos_id=1; g->eot_id=106;
    g->fd=open(path,O_RDONLY); if(g->fd<0){perror("open gguf");exit(1);}
    struct stat st; fstat(g->fd,&st); g->size=(size_t)st.st_size;
    g->base=(const uint8_t*)mmap(NULL,g->size,PROT_READ,MAP_PRIVATE,g->fd,0);
    if(g->base==MAP_FAILED){perror("mmap gguf");exit(1);}
    const uint8_t *c=g->base;
    if(rd_u32(&c)!=0x46554747u){fprintf(stderr,"bad gguf magic\n");exit(1);}
    rd_u32(&c);
    uint64_t n_tensors=rd_u64(&c), n_kv=rd_u64(&c);
    for(uint64_t i=0;i<n_kv;i++){
        strref key=rd_str(&c); uint32_t type=rd_u32(&c);
        if(sref_eq(key,"general.alignment")&&type==GGUF_U32) g->alignment=rd_u32(&c);
        else if(sref_eq(key,"gemma4.block_count")&&type==GGUF_U32) g->n_layers=rd_u32(&c);
        else if(sref_eq(key,"gemma4.embedding_length")&&type==GGUF_U32) g->hidden=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.head_count")&&type==GGUF_U32) g->n_heads=rd_u32(&c);
        else if(sref_eq(key,"gemma4.feed_forward_length")&&type==GGUF_U32) g->ffn=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.key_length")&&type==GGUF_U32) g->key_length=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.value_length")&&type==GGUF_U32) rd_u32(&c); /* == key_length */
        else if(sref_eq(key,"gemma4.attention.key_length_swa")&&type==GGUF_U32) g->key_length_swa=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.value_length_swa")&&type==GGUF_U32) rd_u32(&c);
        else if(sref_eq(key,"gemma4.rope.dimension_count")&&type==GGUF_U32) g->n_rot_full=rd_u32(&c);
        else if(sref_eq(key,"gemma4.rope.dimension_count_swa")&&type==GGUF_U32) g->n_rot_swa=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.sliding_window")&&type==GGUF_U32) g->sliding_window=rd_u32(&c);
        else if(sref_eq(key,"gemma4.attention.shared_kv_layers")&&type==GGUF_U32) g->shared_kv_layers=rd_u32(&c);
        else if(sref_eq(key,"gemma4.rope.freq_base")&&type==GGUF_F32) g->rope_theta=rd_f32(&c);
        else if(sref_eq(key,"gemma4.rope.freq_base_swa")&&type==GGUF_F32) g->rope_theta_swa=rd_f32(&c);
        else if(sref_eq(key,"gemma4.attention.layer_norm_rms_epsilon")&&type==GGUF_F32) g->rms_eps=rd_f32(&c);
        else if(sref_eq(key,"gemma4.final_logit_softcapping")&&type==GGUF_F32) g->final_softcap=rd_f32(&c);
        else if(sref_eq(key,"gemma4.attention.head_count_kv")&&type==GGUF_ARRAY){
            uint32_t et=rd_u32(&c); uint64_t cnt=rd_u64(&c);
            for(uint64_t k=0;k<cnt;k++){
                int32_t v=0;
                if(et==GGUF_I32||et==GGUF_U32) v=rd_i32(&c);
                else { skip_value(&c,et); }
                if(k<G4_MAXLAYERS) g->head_count_kv[k]=(uint32_t)v;
            }
            g->have_kv_arr=1;
        }
        else if(sref_eq(key,"gemma4.attention.sliding_window_pattern")&&type==GGUF_ARRAY){
            uint32_t et=rd_u32(&c); uint64_t cnt=rd_u64(&c);
            for(uint64_t k=0;k<cnt;k++){
                int v=0;
                if(et==GGUF_BOOL||et==GGUF_U8||et==GGUF_I8){ v=(int)(*c); c+=1; }
                else { skip_value(&c,et); }
                if(k<G4_MAXLAYERS) g->is_swa[k]=v; /* True=sliding */
            }
            g->have_swa_arr=1;
        }
        else if(sref_eq(key,"tokenizer.ggml.bos_token_id")&&type==GGUF_U32) g->bos_id=rd_u32(&c);
        else if(sref_eq(key,"tokenizer.ggml.eos_token_id")&&type==GGUF_U32) g->eos_id=rd_u32(&c);
        else if(sref_eq(key,"tokenizer.ggml.tokens")&&type==GGUF_ARRAY){
            uint32_t et=rd_u32(&c); uint64_t cnt=rd_u64(&c);
            if(et!=GGUF_STRING){fprintf(stderr,"tokens not string\n");exit(1);}
            g->tokens=(strref*)malloc(cnt*sizeof(strref)); g->n_tokens=cnt;
            for(uint64_t k=0;k<cnt;k++) g->tokens[k]=rd_str(&c);
        }
        else skip_value(&c,type);
    }
    if(!g->have_swa_arr || !g->have_kv_arr){
        /* fallback to the [5 sliding + 1 global] pattern (global every 6th, L%6==5) */
        for(uint32_t L=0;L<g->n_layers && L<G4_MAXLAYERS;L++){
            int global = ((L%6u)==5u);
            if(!g->have_swa_arr) g->is_swa[L]= global?0:1;
            if(!g->have_kv_arr)  g->head_count_kv[L]= global?1u:8u;
        }
    }
    g->n_tensors=(uint32_t)n_tensors;
    g->tensors=(gtensor*)calloc(n_tensors,sizeof(gtensor));
    for(uint64_t i=0;i<n_tensors;i++){
        gtensor *t=&g->tensors[i];
        t->name=rd_str(&c); t->n_dims=rd_u32(&c);
        for(uint32_t d=0;d<t->n_dims;d++) t->dims[d]=rd_u64(&c);
        t->type=rd_u32(&c); t->offset=rd_u64(&c);
    }
    g->data_start=align_up((uint64_t)(c-g->base), g->alignment);
}
static const gtensor *gguf_find_opt(const gguf *g, const char *name){
    for(uint32_t i=0;i<g->n_tensors;i++) if(sref_eq(g->tensors[i].name,name)) return &g->tensors[i];
    return NULL;
}
static const gtensor *gguf_find(const gguf *g, const char *name){
    const gtensor *t=gguf_find_opt(g,name);
    if(!t){ fprintf(stderr,"gguf: tensor '%s' not found\n",name); exit(1);} return t;
}
static const uint8_t *tensor_ptr(const gguf *g, const gtensor *t){ return g->base+g->data_start+t->offset; }

/* ---- math ---- */
static void matvec_q8(const uint8_t *w, const float *in, uint32_t cols, float *out, uint32_t rows){
    uint32_t blocks=cols/32u;
    #pragma omp parallel for schedule(static)
    for(uint32_t r=0;r<rows;r++){
        const uint8_t *row=w+(uint64_t)r*blocks*34u; float acc=0.0f;
        for(uint32_t b=0;b<blocks;b++){
            uint16_t sh; memcpy(&sh,row,2); row+=2; float scale=f16_to_f32(sh);
            const int8_t *q=(const int8_t*)row; const float *iv=in+b*32u;
            float s=0.0f; for(int j=0;j<32;j++) s+=(float)q[j]*iv[j];
            acc+=scale*s; row+=32;
        }
        out[r]=acc;
    }
}
static void embed_row(const uint8_t *embd, uint32_t hidden, uint32_t token, float *out){
    uint32_t blocks=hidden/32u; const uint8_t *row=embd+(uint64_t)token*blocks*34u;
    for(uint32_t b=0;b<blocks;b++){
        uint16_t sh; memcpy(&sh,row,2); row+=2; float scale=f16_to_f32(sh);
        for(uint32_t j=0;j<32u;j++) out[b*32u+j]=scale*(float)(int8_t)row[j]; row+=32;
    }
}
/* RMSNorm with PLAIN learned weight (the gemma +1 is baked into the GGUF f32 at
 * convert time, so a plain multiply matches llama.cpp build_norm exactly). */
static void rmsnorm(float *out, const float *in, const float *w, uint32_t n, float eps){
    double ss=0.0; for(uint32_t i=0;i<n;i++) ss+=(double)in[i]*in[i];
    float inv=(float)(1.0/sqrt(ss/(double)n+(double)eps));
    for(uint32_t i=0;i<n;i++) out[i]=in[i]*inv*w[i];
}
/* per-head RMSNorm WITH learned weight (Q/K norm). */
static void qk_norm(float *x, uint32_t heads, uint32_t hd, const float *w, float eps){
    for(uint32_t h=0;h<heads;h++){
        float *row=x+(uint64_t)h*hd; double ss=0.0;
        for(uint32_t i=0;i<hd;i++) ss+=(double)row[i]*row[i];
        float inv=(float)(1.0/sqrt(ss/(double)hd+(double)eps));
        for(uint32_t i=0;i<hd;i++) row[i]=row[i]*inv*w[i];
    }
}
/* per-head BARE RMSNorm, NO learned weight (V norm = ggml_rms_norm, with_scale=False). */
static void vnorm_bare(float *x, uint32_t heads, uint32_t hd, float eps){
    for(uint32_t h=0;h<heads;h++){
        float *row=x+(uint64_t)h*hd; double ss=0.0;
        for(uint32_t i=0;i<hd;i++) ss+=(double)row[i]*row[i];
        float inv=(float)(1.0/sqrt(ss/(double)hd+(double)eps));
        for(uint32_t i=0;i<hd;i++) row[i]=row[i]*inv;
    }
}
/* proportional/partial NeoX rope with optional freq_factors (ff). Pair j in
 * [0,n_rot/2): angle = pos*theta^(-2j/n_rot) / ff[j] (ff=rope_freqs for GLOBAL,
 * else NULL=1.0); rotate (x[j], x[j+hd/2]). With hd==n_rot here, dims>=n_rot do
 * not exist at head level; GLOBAL's ff[64..255]=1e30 makes those pairs pass
 * through. Mirrors ggml_rope_cache_init: theta_scale=freq_base^(-2/n_dims),
 * theta/ff, neox offset n_dims/2. */
static void rope_ff(float *x, uint32_t heads, uint32_t hd, uint32_t n_rot, uint32_t pos, float theta, const float *ff){
    uint32_t half=hd/2u, rot_pairs=n_rot/2u; if(rot_pairs>half) rot_pairs=half;
    for(uint32_t h=0;h<heads;h++){
        float *row=x+(uint64_t)h*hd;
        for(uint32_t j=0;j<rot_pairs;j++){
            float freq=powf(theta,-2.0f*(float)j/(float)n_rot);
            float ang=(float)pos*freq; if(ff) ang/=ff[j];
            float c=cosf(ang), s=sinf(ang);
            float x0=row[j], x1=row[j+half];
            row[j]=x0*c-x1*s; row[j+half]=x0*s+x1*c;
        }
    }
}
/* GQA/MQA softmax attention, SCALE = 1.0 (no 1/sqrt(hd)). Keys live in a
 * KVMAX-strided cache; attends t in [t_start, pos] (t_start>0 only for the
 * sliding-window boundary at pos>=1024). */
static void attention_g4(float *out, const float *q, const float *kc, const float *vc,
                         uint32_t pos, uint32_t q_heads, uint32_t kv_heads, uint32_t hd,
                         uint32_t kvmax, uint32_t t_start){
    uint32_t group=q_heads/kv_heads, n=pos-t_start+1u;
    float *score=(float*)malloc((size_t)n*sizeof(float));
    for(uint32_t qh=0;qh<q_heads;qh++){
        const float *qv=q+(uint64_t)qh*hd; uint32_t kvh=qh/group;
        float maxs=-3.4028234663852886e+38f;
        for(uint32_t t=t_start;t<=pos;t++){
            const float *kv=kc+(uint64_t)t*kvmax+(uint64_t)kvh*hd;
            float dot=0.0f; for(uint32_t d=0;d<hd;d++) dot+=qv[d]*kv[d];
            score[t-t_start]=dot; if(dot>maxs) maxs=dot;   /* scale = 1.0 */
        }
        float denom=0.0f; for(uint32_t i=0;i<n;i++){ score[i]=expf(score[i]-maxs); denom+=score[i]; }
        float *ov=out+(uint64_t)qh*hd; for(uint32_t d=0;d<hd;d++) ov[d]=0.0f;
        for(uint32_t t=t_start;t<=pos;t++){
            const float *vv=vc+(uint64_t)t*kvmax+(uint64_t)kvh*hd; float w=score[t-t_start]/denom;
            for(uint32_t d=0;d<hd;d++) ov[d]+=w*vv[d];
        }
    }
    free(score);
}
static void geglu_mul(float *out, const float *gate, const float *up, uint32_t n){
    const float k=0.7978845608028654f;
    for(uint32_t i=0;i<n;i++){ float x=gate[i]; float g=0.5f*x*(1.0f+tanhf(k*(x+0.044715f*x*x*x))); out[i]=g*up[i]; }
}

/* ---- model ---- */
typedef struct {
    const uint8_t *wq,*wk,*wv,*wo,*wgate,*wup,*wdown;   /* wv NULL on GLOBAL (k==v) layers */
    const float *attn_norm,*post_attn_norm,*ffn_norm,*post_ffw_norm,*q_norm,*k_norm,*out_scale;
    int is_swa; uint32_t hd,n_kv,n_rot,QD,KVD; float theta;
} glayer;
typedef struct {
    glayer *layers; const uint8_t *embd; const uint8_t *lm_head;
    const float *output_norm; const float *rope_freqs;
} gmodel;

static const float *load_f32(const gguf *g, const char *name){
    const gtensor *t=gguf_find(g,name);
    if(t->type!=GGML_F32){ fprintf(stderr,"%s not F32\n",name); exit(1);} return (const float*)tensor_ptr(g,t);
}
static const uint8_t *load_q8(const gguf *g, const char *name){
    const gtensor *t=gguf_find(g,name);
    if(t->type!=GGML_Q8_0){ fprintf(stderr,"%s not Q8_0\n",name); exit(1);} return tensor_ptr(g,t);
}
static const uint8_t *load_q8_opt(const gguf *g, const char *name){ /* NULL on miss (GLOBAL attn_v) */
    const gtensor *t=gguf_find_opt(g,name); if(!t) return NULL;
    if(t->type!=GGML_Q8_0){ fprintf(stderr,"%s not Q8_0\n",name); exit(1);} return tensor_ptr(g,t);
}
static void model_load(gmodel *m, const gguf *g){
    m->layers=(glayer*)calloc(g->n_layers,sizeof(glayer)); char nm[80];
    for(uint32_t L=0;L<g->n_layers;L++){
        glayer *ly=&m->layers[L];
        ly->is_swa = g->is_swa[L];
        ly->hd     = ly->is_swa ? g->key_length_swa : g->key_length;   /* 256 : 512 */
        ly->n_kv   = g->head_count_kv[L];                              /* 8 : 1     */
        ly->n_rot  = ly->is_swa ? g->n_rot_swa : g->n_rot_full;        /* 256 : 512 */
        ly->theta  = ly->is_swa ? g->rope_theta_swa : g->rope_theta;   /* 1e4 : 1e6 */
        ly->QD     = g->n_heads * ly->hd;
        ly->KVD    = ly->n_kv   * ly->hd;
        snprintf(nm,sizeof nm,"blk.%u.attn_q.weight",L);              ly->wq=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.attn_k.weight",L);              ly->wk=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.attn_v.weight",L);              ly->wv=load_q8_opt(g,nm); /* NULL on GLOBAL */
        snprintf(nm,sizeof nm,"blk.%u.attn_output.weight",L);         ly->wo=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.ffn_gate.weight",L);            ly->wgate=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.ffn_up.weight",L);              ly->wup=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.ffn_down.weight",L);            ly->wdown=load_q8(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.attn_norm.weight",L);           ly->attn_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.post_attention_norm.weight",L); ly->post_attn_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.ffn_norm.weight",L);            ly->ffn_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.post_ffw_norm.weight",L);       ly->post_ffw_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.attn_q_norm.weight",L);         ly->q_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.attn_k_norm.weight",L);         ly->k_norm=load_f32(g,nm);
        snprintf(nm,sizeof nm,"blk.%u.layer_output_scale.weight",L);  ly->out_scale=load_f32(g,nm);
    }
    m->embd=load_q8(g,"token_embd.weight");
    const uint8_t *lm=load_q8_opt(g,"output.weight"); /* tied: absent in this GGUF -> token_embd */
    m->lm_head = lm ? lm : m->embd;
    m->output_norm=load_f32(g,"output_norm.weight");
    m->rope_freqs=load_f32(g,"rope_freqs.weight"); /* single shared [n_rot_full/2]=256 */
}

static int hexval(char c){ if(c>='0'&&c<='9')return c-'0'; if(c>='a'&&c<='f')return c-'a'+10; if(c>='A'&&c<='F')return c-'A'+10; return 0; }
/* SPM detok: U+2581 -> space, <0xHH> -> byte, skip control tokens. */
static void detok_append(char *buf, int sz, int *len, const gguf *g, uint32_t id){
    if(id>=g->n_tokens) return;
    if(id==g->bos_id||id==g->eos_id||id==g->eot_id||id==105u||id==0u||id==100u) return;
    strref t=g->tokens[id];
    if(t.len==6 && t.p[0]=='<'&&t.p[1]=='0'&&t.p[2]=='x'&&t.p[5]=='>'){
        char c=(char)((hexval(t.p[3])<<4)|hexval(t.p[4])); if(*len<sz-1) buf[(*len)++]=c; return;
    }
    for(uint64_t i=0;i<t.len;i++){
        if(i+2<t.len && (unsigned char)t.p[i]==0xE2 && (unsigned char)t.p[i+1]==0x96 && (unsigned char)t.p[i+2]==0x81){
            if(*len<sz-1) buf[(*len)++]=' '; i+=2; continue;
        }
        if(*len<sz-1) buf[(*len)++]=t.p[i];
    }
}

/* OFF-PATH criterion #2: when set (CLI --score), generate runs a teacher-forced
 * scoring pass over the input ids and returns BEFORE generation (mirrors the
 * g_g4_score pattern in tools/axiom_gemma4.cpp, ec38a4e). Default 0 = greedy
 * generation, bit-identical to the pre---score binary. */
static int g_g4gg_score = 0;

/* Forward + greedy decode from explicit input ids. Fills out_ids/n_out and (optional) detok text_buf. */
extern "C" int axiom_gemma4_gguf_generate_ids(const char *gguf_path, const uint32_t *in_ids, int n_in,
                                              int max_new, int use_eos, uint32_t *out_ids, int out_cap,
                                              int *n_out_p, char *text_buf, int text_buf_sz){
    if(!gguf_path||!in_ids||n_in<=0||!out_ids||out_cap<=0||!n_out_p) return 2;
    gguf g; gguf_open(&g,gguf_path);
    gmodel m; model_load(&m,&g);
    const uint32_t H=g.hidden, NH=g.n_heads, FF=g.ffn, NL=g.n_layers;
    const uint32_t VOCAB=(uint32_t)gguf_find(&g,"token_embd.weight")->dims[1];
    const uint32_t WINDOW=g.sliding_window;
    const float eps=g.rms_eps, embed_scale=sqrtf((float)H), softcap=g.final_softcap;
    /* KVMAX = max KVD over layers = max(1*512, 8*256) = 2048 */
    uint32_t KVMAX=0, QDMAX=0;
    for(uint32_t L=0;L<NL;L++){ if(m.layers[L].KVD>KVMAX)KVMAX=m.layers[L].KVD; if(m.layers[L].QD>QDMAX)QDMAX=m.layers[L].QD; }
    if(KVMAX==0) KVMAX=2048; if(QDMAX==0) QDMAX=NH*512u;
    const uint32_t MAXSEQ=(uint32_t)(n_in+max_new+4);
    float *kc=(float*)malloc((size_t)NL*MAXSEQ*KVMAX*4);
    float *vc=(float*)malloc((size_t)NL*MAXSEQ*KVMAX*4);
    float *hid=(float*)malloc(H*4),*nrm=(float*)malloc(H*4),*tmp=(float*)malloc(H*4),*resid=(float*)malloc(H*4);
    float *q=(float*)malloc((size_t)QDMAX*4),*k=(float*)malloc((size_t)KVMAX*4),*v=(float*)malloc((size_t)KVMAX*4);
    float *att=(float*)malloc((size_t)QDMAX*4),*o=(float*)malloc(H*4);
    float *gate=(float*)malloc((size_t)FF*4),*up=(float*)malloc((size_t)FF*4),*mid=(float*)malloc((size_t)FF*4),*dn=(float*)malloc(H*4);
    float *logits=(float*)malloc((size_t)VOCAB*4);

    #define GFWD(token,pos,want_logits) do {                                                  \
        embed_row(m.embd,H,(token),hid);                                                      \
        for(uint32_t i=0;i<H;i++) hid[i]*=embed_scale;                                        \
        for(uint32_t L=0;L<NL;L++){                                                           \
            glayer *ly=&m.layers[L];                                                          \
            uint32_t hd=ly->hd, n_kv=ly->n_kv, n_rot=ly->n_rot, QD=ly->QD, KVD=ly->KVD;       \
            float theta=ly->theta; const float *ff = ly->is_swa ? (const float*)0 : m.rope_freqs; \
            float *kb=kc+((size_t)L*MAXSEQ+(pos))*KVMAX, *vb=vc+((size_t)L*MAXSEQ+(pos))*KVMAX;\
            memcpy(resid,hid,H*4);                                                            \
            rmsnorm(nrm,hid,ly->attn_norm,H,eps);                                             \
            matvec_q8(ly->wq,nrm,H,q,QD); matvec_q8(ly->wk,nrm,H,k,KVD);                      \
            if(ly->wv) matvec_q8(ly->wv,nrm,H,v,KVD); else memcpy(v,k,KVD*4); /* k==v: V=raw k-proj */ \
            qk_norm(q,NH,hd,ly->q_norm,eps); qk_norm(k,n_kv,hd,ly->k_norm,eps);               \
            vnorm_bare(v,n_kv,hd,eps);                  /* V: bare RMSNorm, no weight, no rope */ \
            rope_ff(q,NH,hd,n_rot,(pos),theta,ff); rope_ff(k,n_kv,hd,n_rot,(pos),theta,ff);   \
            memcpy(kb,k,KVD*4); memcpy(vb,v,KVD*4);                                           \
            { uint32_t t_start = (ly->is_swa && (pos)+1u>WINDOW) ? (pos)+1u-WINDOW : 0u;      \
              attention_g4(att,q,kc+(size_t)L*MAXSEQ*KVMAX,vc+(size_t)L*MAXSEQ*KVMAX,(pos),NH,n_kv,hd,KVMAX,t_start); } \
            matvec_q8(ly->wo,att,QD,o,H);                                                     \
            rmsnorm(tmp,o,ly->post_attn_norm,H,eps); for(uint32_t z=0;z<H;z++) hid[z]=resid[z]+tmp[z]; \
            memcpy(resid,hid,H*4);                                                            \
            rmsnorm(nrm,hid,ly->ffn_norm,H,eps);                                              \
            matvec_q8(ly->wgate,nrm,H,gate,FF); matvec_q8(ly->wup,nrm,H,up,FF);               \
            geglu_mul(mid,gate,up,FF); matvec_q8(ly->wdown,mid,FF,dn,H);                      \
            rmsnorm(tmp,dn,ly->post_ffw_norm,H,eps); for(uint32_t z=0;z<H;z++) hid[z]=resid[z]+tmp[z]; \
            float lsv=ly->out_scale[0]; for(uint32_t z=0;z<H;z++) hid[z]*=lsv; /* layer_output_scale */ \
        }                                                                                     \
        if(want_logits){ rmsnorm(nrm,hid,m.output_norm,H,eps); matvec_q8(m.lm_head,nrm,H,logits,VOCAB); \
            for(uint32_t z=0;z<VOCAB;z++) logits[z]=softcap*tanhf(logits[z]/softcap);         \
            if(!g_g4gg_score){ /* oracle scoring does NOT suppress; generation does */        \
                if(258882u<VOCAB) logits[258882]=-INFINITY; /* suppress end-of-image */       \
                if(258883u<VOCAB) logits[258883]=-INFINITY; /* suppress end-of-audio */ } }   \
    } while(0)

    if(g_g4gg_score){
        /* OFF-PATH teacher-forced perplexity (criterion #2, matched quant): want_logits at
         * EVERY position, math mirroring tools-oracle/llama_logprob.cpp BIT-FOR-BIT
         * (float row max/argmax, double sum_exp of expf(logit-max), double nll, target =
         * in_ids[tp+1]). One SCORE line on stdout; returns BEFORE generation. Caller-loop
         * locals are NOT named like any GFWD-internal (i,L,z,...) — no macro capture. */
        double nll=0.0; int count=0, tfm=0;
        /* AXIOM_G4GG_SCORE_DEBUG=1: per-position argmax + top-2 margin on stderr (near-tie
         * forensics for TF_TOP1 diffs vs the oracle); OFF by default, stdout unchanged. */
        const char *_dbge=getenv("AXIOM_G4GG_SCORE_DEBUG"); const int dbg=(_dbge&&*_dbge=='1')?1:0;
        struct timespec _t0,_t1; clock_gettime(CLOCK_MONOTONIC,&_t0);
        fprintf(stderr,"[gemma4-gguf] score: teacher-forcing %d positions (%u-vocab lm_head every step)\n",n_in-1,VOCAB);
        for(int tp=0;tp<n_in-1;tp++){
            GFWD(in_ids[tp],(uint32_t)tp,1);
            float max_logit=logits[0]; uint32_t amax=0;
            for(uint32_t vi=1;vi<VOCAB;vi++) if(logits[vi]>max_logit){ max_logit=logits[vi]; amax=vi; }
            double sum_exp=0.0;
            for(uint32_t vi=0;vi<VOCAB;vi++) sum_exp+=expf(logits[vi]-max_logit);
            const uint32_t tok=in_ids[tp+1];
            nll+=-((double)logits[tok]-(double)max_logit-log(sum_exp));
            if(amax==tok) tfm++;
            count++;
            if(dbg){
                float second=-3.4028234663852886e+38f; uint32_t a2=0;
                for(uint32_t vi=0;vi<VOCAB;vi++) if(vi!=amax && logits[vi]>second){ second=logits[vi]; a2=vi; }
                fprintf(stderr,"SCOREDBG pos=%d argmax=%u top1=%.6f top2id=%u top2=%.6f margin=%.6f tok=%u ltok=%.6f\n",
                        tp,amax,max_logit,a2,second,max_logit-second,tok,logits[tok]);
            }
            if(tp<2 || (tp+1)%32==0 || tp==n_in-2){
                clock_gettime(CLOCK_MONOTONIC,&_t1);
                double el=(_t1.tv_sec-_t0.tv_sec)+(_t1.tv_nsec-_t0.tv_nsec)/1e9;
                double per=el/(double)(tp+1), eta=per*(double)((n_in-1)-(tp+1));
                fprintf(stderr,"[gemma4-gguf] score %d/%d  %.1fs  (%.2f s/pos, eta %.0fs)\n",tp+1,n_in-1,el,per,eta);
            }
        }
        const double ppl=count>0?exp(nll/(double)count):0.0;
        printf("SCORE_NLL=%.6f SCORE_COUNT=%d SCORE_PPL=%.6f TF_TOP1_MATCH=%d/%d\n",nll,count,ppl,tfm,count);
        *n_out_p=0;
        /* NOTE: no #undef here — GFWD is still expanded by the generation path below;
         * the function-end #undef is the single point of retirement. */
        free(kc);free(vc);free(hid);free(nrm);free(tmp);free(resid);free(q);free(k);free(v);
        free(att);free(o);free(gate);free(up);free(mid);free(dn);free(logits);
        free(m.layers);
        if(g.tokens) free(g.tokens); free(g.tensors);
        if(g.base&&g.base!=MAP_FAILED) munmap((void*)g.base,g.size); if(g.fd>=0) close(g.fd);
        return 0;
    }

    for(int i=0;i<n_in;i++) GFWD(in_ids[i],(uint32_t)i,(i==n_in-1));
    int pos=n_in, n_out=0, tlen=0;
    for(int step=0; step<max_new && n_out<out_cap; step++){
        uint32_t best=0; float bestv=logits[0];
        for(uint32_t j=1;j<VOCAB;j++) if(logits[j]>bestv){ bestv=logits[j]; best=j; }
        out_ids[n_out++]=best;
        if(text_buf) detok_append(text_buf,text_buf_sz,&tlen,&g,best);
        if(use_eos && (best==g.eos_id || best==g.eot_id)) break;
        GFWD(best,(uint32_t)pos,1); pos++;
    }
    if(text_buf && text_buf_sz>0) text_buf[tlen<text_buf_sz?tlen:text_buf_sz-1]='\0';
    *n_out_p=n_out;
    #undef GFWD
    free(kc);free(vc);free(hid);free(nrm);free(tmp);free(resid);free(q);free(k);free(v);
    free(att);free(o);free(gate);free(up);free(mid);free(dn);free(logits);
    free(m.layers);
    if(g.tokens) free(g.tokens); free(g.tensors);
    if(g.base&&g.base!=MAP_FAILED) munmap((void*)g.base,g.size); if(g.fd>=0) close(g.fd);
    return 0;
}

#ifndef AXIOM_GEMMA4_GGUF_NO_CLI_MAIN
int main(int argc, char **argv){
    if(argc<2){ fprintf(stderr,"usage: %s <model.gguf> --ids a,b,c [--max N] [--no-eos] [--show-ids] [--score]\n",argv[0]); return 2; }
    const char *path=argv[1], *ids_csv=NULL; int max_new=8, use_eos=1, show_ids=0;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--ids")&&i+1<argc) ids_csv=argv[++i];
        else if(!strcmp(argv[i],"--max")&&i+1<argc) max_new=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--no-eos")) use_eos=0;
        else if(!strcmp(argv[i],"--show-ids")) show_ids=1;
        else if(!strcmp(argv[i],"--score")) g_g4gg_score=1;   /* OFF-PATH teacher-forced perplexity (criterion #2) */
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 2; }
    }
    if(!ids_csv){ fprintf(stderr,"need --ids (BOS already included)\n"); return 2; }
    uint32_t in_ids[8192]; int n_in=0;
    { char *s=strdup(ids_csv); for(char *t=strtok(s,",");t;t=strtok(NULL,",")) if(n_in<8192) in_ids[n_in++]=(uint32_t)strtoul(t,NULL,10); free(s); }
    if(n_in==0){ fprintf(stderr,"no input ids\n"); return 2; }
    if(show_ids){ fprintf(stderr,"[gemma4-gguf] in:"); for(int i=0;i<n_in;i++) fprintf(stderr," %u",in_ids[i]); fprintf(stderr,"\n"); }
    uint32_t out_ids[4096]; int n_out=0; static char text[16384];
    int rc=axiom_gemma4_gguf_generate_ids(path,in_ids,n_in,max_new,use_eos,out_ids,4096,&n_out,text,sizeof text);
    if(rc!=0) return rc;
    if(g_g4gg_score) return 0;   /* SCORE line already printed by the generate path */
    printf("GEN_IDS:"); for(int i=0;i<n_out;i++) printf(" %u",out_ids[i]); printf("\n");
    printf("TEXT: %s\n", text);
    fprintf(stderr,"[gemma4-gguf] generated=%d\n",n_out);
    return 0;
}
#endif
