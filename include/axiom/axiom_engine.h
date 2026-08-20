#ifndef AXIOM_ENGINE_H
#define AXIOM_ENGINE_H
/*
 * Axiom continuous-batching STEP ENGINE — the FAMILY-AGNOSTIC core.
 *
 * Axiom serves many model families behind one resident interface; this engine is
 * the family-independent half of continuous batching, hoisted P2-W5b from the
 * first family runner that grew it. It owns ONLY machinery whose behavior does
 * not depend on any family's math:
 *
 *   - per-sequence state (axiom_engine_seq) and the running/active sets,
 *   - the paged-KV block pool: free-list, refcounts, lazy per-seq block tables,
 *     graceful out-of-blocks back-pressure (admit) / clean hard error (decode),
 *   - admission, retire, collect, next_finished and drain bookkeeping,
 *   - the per-seq rng/sampling DRIVER (greedy argmax inline; temperature
 *     sampling delegated to the family with the seq's own rng stream),
 *   - the device block-table/cache-length upload bookkeeping (d_tbl/d_meta),
 *   - the lockstep step() skeleton: selection -> ensure blocks -> device table
 *     upload -> ONE family forward over the active subset -> stash bookkeeping.
 *
 * Everything with model math in it stays family-side, reached through the
 * axiom_engine_family_ops vtable: the prefill-token forward, the batched
 * decode forward, temperature sampling, detokenization, and the ENTIRE prefix
 * cache (lookup/register/evict and its key + LRU metadata — family-side per the
 * fire-board decision; the engine sees it only as four optional hooks).
 *
 * The hoist is a PURE refactor: with the first family's ops plugged in, every
 * byte of stdout/stderr/exit-code of the pre-hoist engine is reproduced.
 *
 * The header is split in two layers:
 *   1) the DRIVER ABI — the opaque engine handle + the generic session calls a
 *      scheduler/daemon needs. It depends on NOTHING but <stdint.h>, so driver
 *      TUs that cannot see the Axiom C runtime type names (e.g. a service tree
 *      that uses them for its own identifiers) define AXIOM_ENGINE_DRIVER_ONLY
 *      before including and still drive engines.
 *   2) the PROVIDER SPI — the seq/arena/device views and the family-ops vtable
 *      an engine provider implements. Needs the Axiom C-API (axiom/axiom.h).
 *
 * Thread-safety: NONE of these calls are thread-safe; the caller serializes
 * (the daemon holds its GPU mutex for an engine session's whole lifetime).
 */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 1) DRIVER ABI — generic engine sessions (no other Axiom header required).
 * ========================================================================= */

typedef struct axiom_engine axiom_engine;
typedef void (*axiom_engine_on_piece)(void *ctx, int seq,
                                      const char *piece, int len);

/*
 * Engines are created per family through a factory (the family-agnostic
 * dispatch is axiom_resident_model_engine_create in axiom_resident.h; NULL =>
 * the model's family has no engine provider and the caller keeps its existing
 * path). The driver then runs the session:
 *   admit   : init one seq + prefix lookup + (suffix-only) serial prefill +
 *             join the running set. out_cap is the top-check generation cap
 *             (callers normally pass max_new). Non-zero rc => seq NOT joined;
 *             pool exhaustion returns a graceful back-pressure error code.
 *   step    : ONE lockstep decode step over the running set (selection ->
 *             ensure blocks -> device table upload -> family forward).
 *   collect : copy a seq's generated ids + detok'd text out (by sid).
 *   next_finished : report a finished-but-still-running seq for the driver to
 *             collect/fulfill/retire (frees its slot for newcomers).
 *   on_piece : optional live observer for committed detok pieces; NULL is a
 *             byte-identical no-op for the batch path.
 *   retire  : drop a seq, releasing its block references.
 *   cache_stats : prefix-cache counters + most-recent-admit reuse.
 *   destroy : release all seqs + engine scratch, then the family ctx.
 */
int  axiom_engine_admit(axiom_engine *e, uint64_t sid, const uint32_t *ids, int n_in,
                        float temp, int top_k, float top_p, unsigned int seed,
                        int max_new, int out_cap, int use_eos,
                        void *piece_ctx, axiom_engine_on_piece on_piece,
                        int piece_seq);
int  axiom_engine_step(axiom_engine *e);
int  axiom_engine_collect(axiom_engine *e, uint64_t sid,
                          uint32_t *out_ids, int *n_out, char *text, int text_cap);
int  axiom_engine_num_active(axiom_engine *e);
int  axiom_engine_next_finished(axiom_engine *e, uint64_t *sid_out);
void axiom_engine_retire(axiom_engine *e, uint64_t sid);
void axiom_engine_cache_stats(axiom_engine *e,
                              uint64_t *reused_total, uint64_t *registered_total,
                              int *last_reuse_blocks, int *last_prefill_start);
void axiom_engine_destroy(axiom_engine *e);

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * 2) PROVIDER SPI — what a family implements to plug its math into the engine.
 * ========================================================================= */
#ifndef AXIOM_ENGINE_DRIVER_ONLY

#include "axiom/axiom.h"   /* axiom_runtime, axiom_device_buffer */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Per-sequence state. PUBLIC (not opaque) because the family forward hooks
 * address it directly: prefill/forward read next_tok / pos / block_table and
 * write the next-step logits into seq_logits. The engine owns allocation,
 * lifetime and every counter; family hooks must treat all bookkeeping fields
 * as read-only.
 */
typedef struct axiom_engine_seq {
    uint64_t sid;
    uint32_t *ids;       /* in+out ids, contiguous; capacity ids_cap */
    int ids_cap;
    int n_in;            /* prompt id count */
    int n_out;           /* generated id count (out portion starts at ids+n_in) */
    int pos;             /* current cache length (= n_in after prefill, ++ per decode) */
    int generated;       /* same counter as n_out; kept distinct to mirror the original */
    int max_new;
    int out_cap;         /* top-check generation cap (may differ from max_new) */
    uint64_t rng;        /* per-seq sampler stream */
    float *seq_logits;   /* vocab f32 — CURRENT logits to sample from */
    int *block_table;    /* logical block lb -> physical block id in the arena pool
                          * (ONE table shared across all layers; grows with pos) */
    int n_blocks;        /* logical blocks currently allocated (== valid block_table entries) */
    int blocks_cap;      /* capacity of block_table (= ceil(maxseq/block_tokens)) */
    uint32_t maxseq;
    int done;
    int use_eos;         /* PER-SEQUENCE eos policy: stop on EOS only if set */
    uint32_t next_tok;   /* token to embed this step (set in selection) */
    uint32_t device_greedy_tok; /* provider-staged greedy token from device logits */
    int device_greedy_tok_valid;
    int device_greedy_tok_resident; /* staged token is also present in provider token buffer */
    int device_greedy_tok_pending;  /* provider has async device work producing the token */
    int (*device_greedy_tok_materialize)(void *ctx, struct axiom_engine_seq *seq);
    void *device_greedy_tok_materialize_ctx;
    int next_tok_device_resident; /* next_tok is still resident in provider token buffer */
    float temp; int top_k; float top_p; unsigned int seed;
    void *piece_ctx;     /* optional per-seq live observer; NULL => no-op */
    axiom_engine_on_piece on_piece;
    int piece_seq;
    int stream_after_reset; /* provider-side streaming gate for reset-marker families */
} axiom_engine_seq;

/*
 * PERSISTENT paged-KV arena (resident/model lifetime). Created lazily by the
 * family (via the ops kv_arena accessor) so it outlives any one engine session;
 * the engine only BORROWS it. Single owner = the family resident handle, which
 * frees it (and its own `prefix` state) exactly once at resident destroy.
 *
 * The arena holds the family-independent pool substrate: payload pools (host
 * or device), the free-list and the per-physical-block refcounts. `prefix` is
 * an OPAQUE family pointer for the family-side prefix-cache metadata (keys,
 * id-spans, LRU); the engine never dereferences it.
 *
 * Block lifecycle (per physical block pb):
 *   free-list -> alloc -> live (refcount>0) -> last release ->
 *     family keeps it idle (prefix_block_idle returns 1; evictable later via
 *     prefix_evict_idle) OR back to the free-list.
 */
typedef struct axiom_engine_kv_arena {
    axiom_runtime *rt;       /* runtime the device pools (if any) live on */
    uint32_t kv_bt;          /* BLOCK_TOKENS (env AXIOM_KV_BLOCK, default 16) */
    int kv_nphys;            /* total physical blocks in the pool */
    float *kpool, *vpool;    /* kv_nphys*n_layers*BT*kv_dim floats each (HOST mode; NULL in device mode) */
    /* When device==1 the pool PAYLOAD is device-resident (d_kpool/d_vpool, same
     * element count/addressing) and kpool/vpool stay NULL. Decided ONCE at arena
     * create: AXIOM_ENGINE_DEVICE_ATTN=0 forces host, otherwise device when both
     * allocations succeed — on failure the arena logs once and falls back to the
     * HOST pool path unchanged. All metadata below is host-resident in BOTH modes. */
    int device;
    axiom_device_buffer *d_kpool, *d_vpool;
    int *free_list;          /* free (idle, refcount==0, not family-kept) physical block ids; n_free on top */
    int n_free;
    int *refcount;           /* [kv_nphys] live block_table references across all running seqs */
    void *prefix;            /* OPAQUE family prefix-cache state (the engine never reads it) */
} axiom_engine_kv_arena;

/*
 * Engine-owned DEVICE view: the block-table/cache-length upload bookkeeping for
 * device-mode (arena->device) stepping, plus borrows of the arena pools. The
 * engine sizes/uploads d_tbl and d_meta; family forward hooks READ this struct
 * to address the paged kernels (cap fixes the d_meta byte offset of the pos
 * row: cache_tokens ride at offset 0, pos at offset cap*4).
 */
typedef struct axiom_engine_dev {
    axiom_runtime *rt;
    axiom_device_buffer *d_kpool, *d_vpool;   /* arena-owned device pools (borrowed) */
    axiom_device_buffer *d_tbl;    /* [cap][tbl_stride] int32 block-table matrix */
    axiom_device_buffer *d_meta;   /* [2*cap] u32: cache_tokens at 0, pos at cap*4 */
    int cap;                       /* seq-slots the bookkeeping holds */
    uint32_t tbl_stride;           /* int32 entries per d_tbl row */
    int32_t  *htbl;                /* host staging [cap][tbl_stride] */
    uint32_t *hmeta;               /* host staging [2*cap] (ct, then pos at index cap) */
} axiom_engine_dev;

/*
 * Family vtable. The engine drives these; the family supplies the math.
 * Mandatory: prefill_token, forward_step, sample_token, detok,
 * ensure_batch_scratch, kv_arena, destroy. Optional (may be NULL):
 * bind, dev_scratch_ensure (device mode only), the four prefix hooks and
 * cache_totals (no prefix cache -> cold prefill always, idle blocks free-listed).
 */
typedef struct axiom_engine_family_ops {
    /* Make the family's runtime globals current. Called at admit/step entry,
     * mirroring the pre-hoist call points. Optional. */
    void (*bind)(void *fam);
    /* PREFILL one token of one seq at absolute position `pos` (serial, exact
     * single-seq math), writing this token's KV through seq->block_table and,
     * when want_logits, the logits into seq->seq_logits. In device mode the
     * engine has ALREADY uploaded the seq's table + (ct,pos) metadata for this
     * token; `dev` is non-NULL with the kernel addressing view. Returns 0 ok. */
    int (*prefill_token)(void *fam, axiom_engine_seq *seq, uint32_t token,
                         uint32_t pos, int want_logits, const axiom_engine_dev *dev);
    /* ONE lockstep batched decode forward over the active subset (admission
     * order): embed each seq's next_tok, run the layers with per-seq cache
     * length seq->pos+1, write each seq's next logits into seq->seq_logits.
     * The engine increments pos AFTER this returns 0. In device mode the
     * engine has ALREADY uploaded all A block tables + lengths; `dev` is
     * non-NULL. Returns 0 ok (engine aborts the step on rc). */
    int (*forward_step)(void *fam, axiom_engine_seq *const *active, int n_active,
                        const axiom_engine_dev *dev);
    /* Temperature sampling from `logits` using the seq's OWN rng stream
     * (read+advance *rng). Only called when temp > 0 (greedy argmax is inline
     * in the engine). Returns the sampled token id. */
    uint32_t (*sample_token)(void *fam, const float *logits, uint32_t vocab,
                             float temp, int top_k, float top_p, uint64_t *rng);
    /* Detokenize one id into buf (cap bytes incl. NUL); returns byte count.
     * MAY return a NEGATIVE count: collect then DISCARDS the text assembled so
     * far for that seq and appends nothing for this id. This keeps a family
     * text contract that defines a RESET MARKER (an id whose solo-path
     * semantic is "clear the accumulated display text", e.g. an
     * end-of-hidden-segment marker) intact under the engine's per-id text
     * assembly. Generated ids are never affected — text only. */
    int (*detok)(void *fam, uint32_t id, char *buf, int cap);
    /* Optional streaming policy: when nonzero, committed pieces are held until
     * detok returns the generic RESET MARKER once for that seq. This preserves
     * families whose solo stream intentionally hides a pre-reset thinking span. */
    int stream_after_reset_marker;
    /* Grow the family's batched gather/scatter scratch to >= cap seq-slots
     * (no-op when already large enough). Contents are pure per-step scratch. */
    int (*ensure_batch_scratch)(void *fam, int cap);
    /* Device mode only (optional): grow the family's per-step staging rows
     * (Q/K/V/att) to >= cap seq-slots. Called right after the engine grows its
     * own d_tbl/d_meta bookkeeping, at the same pre-hoist trigger points. */
    int (*dev_scratch_ensure)(void *fam, int cap);
    /* ---- prefix cache (ALL optional; family-side per the fire-board decision) ----
     * lookup: at admit, BEFORE prefill — reuse shared prefix blocks into
     *   seq->block_table[0..reuse) (bump arena refcounts, set seq->n_blocks);
     *   returns the number of blocks reused (prefill starts at reuse*kv_bt).
     * register: at admit, AFTER prefill — register the newly-written full
     *   blocks (first-writer-wins).
     * evict_idle: the free-list is empty — evict one idle cached block and
     *   return its physical id, or -1 (engine back-pressure).
     * block_idle: a block's refcount just hit 0 — return 1 to keep it
     *   allocated (cached, evictable later), 0 to return it to the free-list.
     * cache_totals: cumulative arena-lifetime counters for observability. */
    int (*prefix_lookup)(void *fam, axiom_engine_kv_arena *ar, axiom_engine_seq *seq);
    void (*prefix_register)(void *fam, axiom_engine_kv_arena *ar,
                            axiom_engine_seq *seq, int reuse_blocks);
    int (*prefix_evict_idle)(void *fam, axiom_engine_kv_arena *ar);
    int (*prefix_block_idle)(void *fam, axiom_engine_kv_arena *ar, int pb);
    void (*cache_totals)(void *fam, axiom_engine_kv_arena *ar,
                         uint64_t *reused_total, uint64_t *registered_total);
    /* Get (or lazily create) the resident's PERSISTENT arena. Called once at
     * engine create. NULL => engine create fails cleanly (caller falls back). */
    axiom_engine_kv_arena *(*kv_arena)(void *fam);
    /* Free the family ctx (its scratch, staging and the ctx itself). Called
     * LAST in axiom_engine_destroy. */
    void (*destroy)(void *fam);
} axiom_engine_family_ops;

/*
 * Create/destroy the family-independent half of the persistent arena (pool
 * payload + free-list + refcounts; `prefix` starts NULL — the family owns and
 * frees whatever it hangs there BEFORE calling destroy). Sizing/placement come
 * from the env exactly as before: AXIOM_KV_BLOCK, AXIOM_KV_POOL_BLOCKS (default
 * AXIOM_MAX_BATCH * ceil(AXIOM_KV_MAX_CONTEXT/BT)), AXIOM_ENGINE_DEVICE_ATTN.
 * n_layers/kv_dim are the family's pool geometry (element offset of (physical
 * block pb, layer L, slot, feature 0) = (((pb*n_layers+L)*BT+slot)*kv_dim)).
 *
 * PER-MODEL sizing override (deploy fix: one process serves models whose
 * per-block bytes differ by ~20x, while AXIOM_KV_POOL_BLOCKS is one global
 * env): pool_blocks > 0 fixes kv_nphys for THIS arena (beats the env);
 * max_context > 0 replaces AXIOM_KV_MAX_CONTEXT in the default-size formula
 * (only consulted when pool_blocks <= 0 and the env is unset). Pass 0/0 for
 * the historical env/default behavior, byte-identical. The arena is created
 * once per resident (model lifetime), so the FIRST create sizes it.
 */
axiom_engine_kv_arena *axiom_engine_kv_arena_create(axiom_runtime *rt,
                                                    uint32_t n_layers, uint32_t kv_dim,
                                                    int pool_blocks, int max_context);
void axiom_engine_kv_arena_destroy(axiom_engine_kv_arena *arena);

/*
 * Bind the vtable + family ctx, borrow the family's persistent arena
 * (ops->kv_arena), and record the per-model dims: vocab (the logits width) and
 * the EOS id list (copied; up to 8 ids). Returns NULL on allocation failure or
 * when the arena accessor returns NULL — the caller keeps its existing
 * non-engine path. Family factories wrap this (one per provider).
 */
axiom_engine *axiom_engine_create(const axiom_engine_family_ops *ops, void *fam,
                                  uint32_t vocab, const uint32_t *eos_ids, uint32_t n_eos);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_ENGINE_DRIVER_ONLY */

#endif /* AXIOM_ENGINE_H */
