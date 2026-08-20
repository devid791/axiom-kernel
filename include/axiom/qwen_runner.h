#ifndef AXIOM_QWEN_RUNNER_H
#define AXIOM_QWEN_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_QWEN_GENERATE_CONFIG_ABI_VERSION 1u
#define AXIOM_QWEN_RESIDENT_CONFIG_ABI_VERSION 3u

#define AXIOM_QWEN_PLACEMENT_GPU_RESIDENT "gpu_resident"
#define AXIOM_QWEN_PLACEMENT_GPU_HOST_OFFLOAD "gpu_with_host_offload"

typedef struct axiom_qwen_generate_config {
    uint32_t abi_version;
    const char *model_path;
    const char *ids;
    const char *prompt;
    int max_new;
    const char *trace_json;
    const char *proof_model_id;
    const char *proof_family;
    int use_eos;
    int show_ids;
    int chat;
    int no_think;
    float temperature;          /* <=0 => greedy argmax (default) */
    int top_k;                  /* >0 => restrict to top-k (clamped <=256) */
    float top_p;                /* (0,1) => nucleus cutoff */
    unsigned int sampling_seed; /* 0 => default deterministic stream */
    void *stream_ctx;           /* passed to stream_cb */
    void (*stream_cb)(void *ctx, const char *piece, int len); /* called per generated token (detok'd text) */
    /* optional in-process capture (NULL/0 => behave as before) so the generic
     * axiom_resident interface can treat qwen as just another family. */
    uint32_t *capture_out_ids;  /* if set, generated ids are copied here */
    int capture_out_cap;        /* capacity of capture_out_ids */
    int *capture_n_out;         /* if set, receives the generated id count */
    char *capture_text;         /* if set, receives the detokenized answer */
    int capture_text_sz;        /* capacity of capture_text */
    int quiet;                  /* 1 => suppress GEN_IDS/TEXT stdout + tok/s stderr */
} axiom_qwen_generate_config;

typedef struct axiom_qwen_resident axiom_qwen_resident;

typedef struct axiom_qwen_resident_config {
    uint32_t abi_version;
    const char *model_path;
    const char *placement_mode;
    uint64_t host_offload_device_live_bytes;
    uint64_t host_memory_budget_bytes;
    uint64_t flags;
    const char *split_join;
    const char *split_span;
    const char *split_model_sha;
} axiom_qwen_resident_config;

int axiom_qwen_resident_create(axiom_qwen_resident **out, const axiom_qwen_resident_config *config);
int axiom_qwen_resident_generate(axiom_qwen_resident *resident, const axiom_qwen_generate_config *config);
void axiom_qwen_resident_destroy(axiom_qwen_resident *resident);
void axiom_qwen_gpu_set_cancel(axiom_qwen_resident *h, const volatile int *flag);
uint64_t axiom_qwen_gpu_vram_bytes(const axiom_qwen_resident *h);
uint64_t axiom_qwen_span_worker_resident_bytes(const char *model_path, const char *span);
int axiom_qwen_split_worker_heartbeat(const char *join);

int axiom_qwen_generate(const axiom_qwen_generate_config *config);
int axiom_qwen_main(int argc, char **argv);

/* Clean handle so qwen plugs into the generic axiom_resident interface like any
 * other family. create loads weights into VRAM once; generate self-tokenizes the
 * prompt (or uses in_ids) and fills out_ids/text; destroy frees. */
axiom_qwen_resident *axiom_qwen_gpu_create(const char *gguf_path);
int axiom_qwen_gpu_generate(axiom_qwen_resident *h, const char *prompt, int chat, int no_think,
                            const uint32_t *in_ids, int n_in, int max_new,
                            float temperature, int top_k, float top_p, unsigned int seed,
                            void *stream_ctx, void (*stream_cb)(void *ctx, const char *piece, int len),
                            uint32_t *out_ids, int out_cap, int *n_out, char *text, int text_sz);
void axiom_qwen_gpu_destroy(axiom_qwen_resident *h);

/* Build the exact input token ids the self-tokenizing generate path produces for
 * (prompt, chat, no_think): ChatML template + byte-level BPE when chat, else raw
 * BPE. Lets a caller (the daemon) pre-tokenize qwen for the static-batch decode
 * (axiom_qwen_gpu_generate_batch takes in_ids) and obtain BIT-IDENTICAL ids to
 * what axiom_qwen_gpu_generate would self-tokenize from the same prompt. Writes
 * up to `cap` ids into out_ids; returns the id count, or -1 on bad args. NOT
 * thread-safe (the tokenizer uses shared static scratch) — callers serialize. */
int axiom_qwen_gpu_tokenize(axiom_qwen_resident *h, const char *prompt, int chat, int no_think,
                            uint32_t *out_ids, int cap);

/* Static-batch lockstep decode: runs B sequences together, every matvec batched
 * over the active sequences via the brick batched kernel, BIT-IDENTICAL per
 * sequence to axiom_qwen_gpu_generate. Prefill is serial. Additive — the
 * single-seq generate paths are unchanged.
 *   in_ids[s]/n_in[s] : prompt token ids per sequence (s in 0..B).
 *   out_ids[s]        : buffer (capacity out_cap) receiving generated ids.
 *   n_out[s]          : generated id count per sequence.
 *   text[s]           : buffer (capacity text_sz) receiving detok'd answer (optional).
 *   seeds             : per-seq RNG seeds (optional; only used when temperature>0).
 * Returns 0 on success. */
int axiom_qwen_gpu_generate_batch(axiom_qwen_resident *h,
                                  const uint32_t *const *in_ids, const int *n_in, int B,
                                  int max_new, int use_eos,
                                  float temperature, int top_k, float top_p,
                                  const unsigned int *seeds,
                                  uint32_t **out_ids, int out_cap, int *n_out,
                                  char **text, int text_sz);

/* WAVE 1 (continuous batching, step 1): STATEFUL STEP ENGINE.
 * Hoists the static-batch lockstep decode loop into a create/admit/step/collect/
 * retire/destroy engine so the service can drive continuous admission (W2).
 * BYTE-IDENTICAL to axiom_qwen_gpu_generate_batch: that function is now a thin
 * wrapper (create -> admit all B -> while num_active>0 step -> collect -> destroy).
 *
 * W5b: the engine core is FAMILY-AGNOSTIC (include/axiom/axiom_engine.h,
 * tools/axiom_engine.cpp); qwen is its first axiom_engine_family_ops provider.
 * axiom_qwen_engine_create below is the family factory returning the GENERIC
 * axiom_engine* (this is what the family-agnostic resident dispatch calls); the
 * axiom_qwen_batch_* functions remain ONLY as thin qwen-internal wrappers over
 * the generic ABI, used by the axiom-qwen CLI batch path — the service drives
 * axiom_engine_* directly and never references a family-named engine symbol.
 *
 *   create  : allocate the engine bound to a resident (weights stay shared).
 *   admit   : PREFILL one seq (serial, exact single-seq math) + alloc its KV +
 *             join the running set; sid identifies it for collect/retire.
 *   step    : ONE lockstep decode step over the current running set (gather the
 *             not-done seqs in admission order -> per-seq sample (own rng) ->
 *             ONE batched forward with per-seq cache length pos[s]+1 -> append).
 *   collect : copy a seq's generated ids + detok'd text out (by sid).
 *   retire  : drop a finished seq, freeing its KV.
 *   destroy : free the engine + all per-seq KV.
 * The engine owns the running set + per-seq KV; retire/destroy free all KV.
 * NOT thread-safe — the caller serializes (the daemon holds the GPU mutex). */
struct axiom_engine;   /* the generic step engine (include/axiom/axiom_engine.h) */
/* kv_pool_blocks / kv_max_context: PER-MODEL paged-KV arena budget plumbed from
 * the model manifest (deploy fix; 0/0 = AXIOM_KV_POOL_BLOCKS env / computed
 * default, unchanged). First engine create sizes the resident's arena. */
struct axiom_engine *axiom_qwen_engine_create(axiom_qwen_resident *h,
                            const char *model_id, const char *variant_key,
                            uint64_t variant_version,
                            int kv_pool_blocks, int kv_max_context);
typedef struct axiom_qwen_batch_engine axiom_qwen_batch_engine;
/* WAVE 4 (prefix cache): create a step engine BORROWING the resident's persistent
 * paged-KV + prefix-cache arena (created lazily on first call, freed in resident
 * destroy). The cache key namespace = hash(model_id, variant_key[, variant_version]);
 * callers use the opaque variant fields to isolate model/profile/config variants.
 * Pass "" / 0 when no additional namespace is required. */
axiom_qwen_batch_engine *axiom_qwen_batch_create(axiom_qwen_resident *h,
                            const char *model_id, const char *variant_key,
                            uint64_t variant_version);
/* admit carries use_eos PER-SEQUENCE (WAVE 2: promoted from the W1 engine-global
 * e->use_eos to per-SeqState so continuous admission can mix per-request eos
 * policy; the static wrapper passes its scalar use_eos to every seq). */
int  axiom_qwen_batch_admit(axiom_qwen_batch_engine *engine, uint64_t sid,
                            const uint32_t *ids, int n_in, float temp, int top_k, float top_p,
                            unsigned int seed, int max_new, int use_eos);
int  axiom_qwen_batch_step(axiom_qwen_batch_engine *engine);
int  axiom_qwen_batch_collect(axiom_qwen_batch_engine *engine, uint64_t sid,
                              uint32_t *out_ids, int *n_out, char *text, int text_cap);
int  axiom_qwen_batch_num_active(axiom_qwen_batch_engine *engine);
/* WAVE 2 (continuous admission): report a seq that finished (done) this step but is
 * still in the running set. Returns 1 and writes its sid to *sid_out if such a seq
 * exists (caller collect -> fulfill -> retire it), else 0. Lets the driver retire
 * finished seqs mid-flight so freed slots can admit newcomers for the next step. */
int  axiom_qwen_batch_next_finished(axiom_qwen_batch_engine *engine, uint64_t *sid_out);
void axiom_qwen_batch_retire(axiom_qwen_batch_engine *engine, uint64_t sid);
void axiom_qwen_batch_destroy(axiom_qwen_batch_engine *engine);
/* WAVE 4 (observability): cumulative prefix-cache counters (arena lifetime) + the
 * most-recent admit's reuse (blocks reused from cache, first prefilled token index).
 * Any out pointer may be NULL. */
void axiom_qwen_batch_cache_stats(axiom_qwen_batch_engine *engine,
                                  uint64_t *reused_total, uint64_t *registered_total,
                                  int *last_reuse_blocks, int *last_prefill_start);

#ifdef __cplusplus
}
#endif

#endif
