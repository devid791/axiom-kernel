#ifndef AXIOM_RESIDENT_H
#define AXIOM_RESIDENT_H
/*
 * Axiom resident-model interface — FAMILY-AGNOSTIC.
 *
 * Axiom is a kernel for many model families, not a single-model runner. This is
 * the one generic handle the service/daemon uses: weights are loaded into VRAM
 * ONCE at create() and reused per generate() (no per-request reload). Each
 * family (qwen3_dense, gemma3, nemotron, gemma4_unified) is just an
 * implementation registered behind this interface — qwen is one family like the
 * others, never "the engine".
 *
 * Family implementations live in tools/axiom_<family>.cpp and expose their own
 * <fam>_gpu_create/generate/destroy; tools/axiom_resident.cpp dispatches by the
 * `family` string. The service never calls a family-specific symbol directly.
 */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_resident_model axiom_resident_model;

/* Recognized family strings (match the manifest [model].family). */
#define AXIOM_FAMILY_QWEN3_DENSE   "qwen3_dense"
#define AXIOM_FAMILY_GEMMA3        "gemma3"
#define AXIOM_FAMILY_NEMOTRON      "nemotron"
#define AXIOM_FAMILY_GEMMA4        "gemma4_unified"
#define AXIOM_FAMILY_QWEN35MOE     "qwen35moe"      /* Qwen3.6/Qwen3-Next hybrid (DeltaNet+MoE) */
#define AXIOM_FAMILY_LLAMA         "rope_swiglu_decoder" /* Llama-3.x-like dense GGUF (P3-N1/N2 onboarding proof; family-level, not one model) */

/*
 * Create a GPU-resident model. Loads + (if needed) quantizes weights into VRAM
 * once. `weights_path` is a GGUF (qwen/gemma3/nemotron) or safetensors
 * (gemma4). `tok_path` is a tokenizer GGUF used for detokenization where the
 * weights file has no vocab (gemma4 -> the shared gemma vocab); pass NULL when
 * the weights file already carries the tokenizer. Returns NULL on failure.
 */
axiom_resident_model *axiom_resident_model_create(const char *family,
                                                  const char *weights_path,
                                                  const char *tok_path);

/*
 * Generate an answer. Two input modes, picked per family:
 *   - qwen3_dense self-tokenizes: pass `prompt` (+ chat/no_think); ChatML is
 *     applied internally; in_ids is ignored.
 *   - gemma3 / nemotron / gemma4_unified take pre-tokenized `in_ids` (the
 *     caller applied the family chat template + SPM encode); prompt is ignored.
 * Fills out_ids[0..*n_out) (<= out_cap) and, if text != NULL, the detokenized
 * answer into text[0..text_sz). temperature<=0 => greedy. NOT reentrant per
 * handle — the caller serializes concurrent calls on one model (one GPU).
 *
 * Cooperative cancellation: callers may arm a per-request flag with
 * axiom_resident_model_set_cancel() immediately before generate and MUST clear
 * it afterwards. The handle is already non-reentrant, so this is handle-local
 * request state, not a cross-thread ownership transfer. NULL (the default) means
 * never cancelled and preserves legacy behavior.
 *
 * Returns 0 on success, 2 when cancelled after committing a partial output
 * (out_ids/n_out/text contain the coherent prefix generated so far), and other
 * nonzero values for errors.
 */
int axiom_resident_model_generate(axiom_resident_model *m,
                                  const char *prompt, int chat, int no_think,
                                  const uint32_t *in_ids, int n_in,
                                  int max_new, int use_eos,
                                  float temperature, int top_k, float top_p, unsigned int seed,
                                  void *stream_ctx,
                                  void (*stream_cb)(void *ctx, const char *piece, int len),
                                  uint32_t *out_ids, int out_cap, int *n_out,
                                  char *text, int text_sz);

void axiom_resident_model_set_cancel(axiom_resident_model *m,
                                     const volatile int *flag);

/*
 * Static-batch generation — runs B sequences together (daemon-side batching).
 * Takes PRE-TOKENIZED input ids per sequence (in_ids[s]/n_in[s]) for EVERY
 * family (use axiom_resident_model_tokenize for qwen, the GGUF probe for the
 * others). Each sequence shares the scalar sampling params (max_new, use_eos,
 * temperature, top_k, top_p); seeds are per-sequence (only used when
 * temperature>0). out_ids[s]/n_out[s]/text[s] receive each sequence's result.
 *
 * Dispatch: qwen3_dense -> axiom_qwen_gpu_generate_batch (true kernel-batched
 * lockstep decode, bit-identical per sequence to a solo run). Other families
 * have no batched kernel yet, so each sequence is run through its single
 * per-family generate (still correct, just not GPU-batched). Returns 0 on
 * success (first non-zero per-seq rc on partial failure). NOT reentrant per
 * handle — the caller serializes concurrent calls on one model (one GPU).
 */
int axiom_resident_model_generate_batch(axiom_resident_model *m,
                                        const uint32_t *const *in_ids, const int *n_in, int B,
                                        int max_new, int use_eos,
                                        float temperature, int top_k, float top_p,
                                        const unsigned int *seeds,
                                        uint32_t **out_ids, int out_cap, int *n_out,
                                        char **text, int text_sz);

/*
 * Tokenize a prompt into family input ids for the batch path. Only families
 * that self-tokenize in-process implement this (qwen3_dense -> ChatML + BPE);
 * others return -1 (the caller tokenizes them via the GGUF vocab probe). Writes
 * up to `cap` ids into out_ids; returns the id count, or -1 when unsupported /
 * on bad args. For qwen the ids are bit-identical to the self-tokenizing
 * single-sequence generate path.
 */
int axiom_resident_model_tokenize(axiom_resident_model *m,
                                  const char *prompt, int chat, int no_think,
                                  uint32_t *out_ids, int cap);

void axiom_resident_model_destroy(axiom_resident_model *m);

/* Returns the family string the handle was created with (for /health, honesty). */
const char *axiom_resident_model_family(const axiom_resident_model *m);

/*
 * Creation-time device-memory accounting for the resident handle, in bytes.
 * Includes only family accounting the implementation knows (resident weights and
 * device pools/scratch it can size honestly). 0 means "unaccounted", never a
 * synthetic estimate.
 */
uint64_t axiom_resident_model_vram_bytes(const axiom_resident_model *m);

/*
 * ONE TRUTH (P3-R1): the engine-implemented family list, exported by the
 * kernel itself. Returns a NULL-terminated array of the family strings this
 * build's resident dispatcher actually implements (exactly the set the
 * dispatcher in tools/axiom_resident.cpp accepts). Consumers (the serving
 * daemon's readiness computation and /health, the catalog's runtime_status,
 * the reconcile gate) derive "implemented" FROM THIS accessor instead of
 * keeping hand-maintained claims that can drift from the kernel.
 * Family-agnostic by construction: it is a list, not a per-family symbol.
 */
const char * const *axiom_resident_implemented_families(void);

/*
 * WAVE 2 / W5b (continuous batching): create a GENERIC stateful step engine
 * bound to this resident handle for CONTINUOUS admission. Returns NULL when the
 * model's family has no engine provider yet — that IS the capability check: the
 * caller keeps such models on the static generate_batch path. The returned
 * engine is driven ONLY via the family-agnostic axiom_engine_* ABI
 * (include/axiom/axiom_engine.h); no family-named symbol is involved on the
 * caller's side. The caller owns it and MUST destroy it (axiom_engine_destroy);
 * weights stay shared with the handle and the caller MUST hold the same GPU
 * lock that serializes generation for the engine's whole lifetime (one GPU).
 *
 * WAVE 4 (prefix cache): the caller passes the cache-key inputs through to the
 * family engine — model_id (cache namespace), a caller-owned variant key and a
 * variant version. The key is opaque to the kernel and isolates model/profile
 * or configuration variants in the persistent cache.
 *
 * PER-MODEL KV POOL BUDGET (deploy fix): kv_pool_blocks / kv_max_context are
 * the manifest-plumbed sizing overrides for the family's persistent paged-KV
 * arena (see axiom_engine_kv_arena_create). One process serves models whose
 * per-block KV bytes differ by ~20x, and the AXIOM_KV_POOL_BLOCKS env is one
 * GLOBAL knob — these reach the per-model arena create instead. Pass 0/0 to
 * keep the env/default sizing (the env stays the fallback). The arena is
 * resident-lifetime: the FIRST engine create for a model sizes it.
 */
struct axiom_engine;
struct axiom_engine *axiom_resident_model_engine_create(
    axiom_resident_model *m, const char *model_id, const char *variant_key,
    uint64_t variant_version, int kv_pool_blocks, int kv_max_context);

/*
 * P3-R2: does this handle's FAMILY ship a step-engine provider at all?
 * Returns 1 for engine-capable families, 0 for provider-less ones (qwen35moe —
 * served via the daemon's static path) and for NULL. This is the capability
 * PREDICATE the caller consults BEFORE a session, so it can distinguish
 * "family has no provider -> static path" (route choice) from "provider's
 * create returned NULL -> real failure" (typed error). One truth: the same
 * dispatcher switch that owns engine_create owns this answer.
 */
int axiom_resident_model_engine_capable(const axiom_resident_model *m);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_RESIDENT_H */
