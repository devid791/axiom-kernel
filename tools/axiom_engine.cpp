/*
 * axiom_engine.cpp — the family-agnostic continuous-batching step engine.
 *
 * P2-W5b hoist: the engine machinery below is a VERBATIM move of the step
 * engine that grew inside the first family runner — per-seq state, running/
 * active sets, the paged-KV block pool (free-list, refcounts, back-pressure),
 * admission/retire/collect/drain bookkeeping, the per-seq rng/sampling driver,
 * the device block-table/cache-length upload bookkeeping and the lockstep
 * step() skeleton — re-pointed at the axiom_engine_family_ops vtable for every
 * piece of family math (prefill forward, batched decode forward, temperature
 * sampling, detok, prefix cache). PURE refactor: with the original family's
 * ops plugged in, stdout/stderr/exit codes are byte-for-byte unchanged.
 *
 * See include/axiom/axiom_engine.h for the contract. NOT thread-safe — the
 * caller serializes (the daemon holds its GPU mutex per engine session).
 */
#include "axiom/axiom_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct axiom_engine {
    axiom_engine_family_ops ops;   /* copied vtable */
    void *fam;                     /* family ctx (owned by the family; freed via ops.destroy) */
    axiom_engine_kv_arena *arena;  /* BORROWED persistent arena (family/resident owns it) */
    uint32_t kv_bt;                /* BLOCK_TOKENS (= arena->kv_bt) */
    int kv_nphys;                  /* total physical blocks (= arena->kv_nphys) */
    uint32_t vocab;                /* logits width (per-model) */
    uint32_t eos_ids[8];           /* EOS id list (copied at create) */
    uint32_t n_eos;
    /* Device mode mirrors arena->device (decided once at arena create). dev
     * borrows the arena's device pools and owns the per-engine block-table/
     * cache-length bookkeeping (lazily grown, freed in destroy). When
     * dev_attn==0 every host code path runs byte-for-byte unchanged. */
    int dev_attn;
    axiom_engine_dev dev;
    /* most-recent-admit prefix reuse (observability) */
    int last_reuse_blocks;   /* blocks reused on the most recent admit */
    int last_prefill_start;  /* first prefill token index on the most recent admit */
    /* running set — admission order preserved (selection iterates this order) */
    axiom_engine_seq **running; int n_running; int running_cap;
    /* per-step active list (seq* in admission order) */
    axiom_engine_seq **active; int active_cap;
};

static int engine_env_int(const char *name, int def) {
    const char *v = getenv(name);
    if (!v || !*v) return def;
    return (int)strtol(v, NULL, 10);
}

/* ----------------------------------------------------------------------------
 * PERSISTENT paged-KV arena (resident/model lifetime) — pool payload, free-list
 * and refcounts. The family-side prefix cache hangs off arena->prefix (opaque
 * here). Created lazily by the family's ops->kv_arena accessor on the first
 * engine create; freed exactly once by the resident (single owner -> no UAF /
 * double-free). Across sessions the pool (and the family's cache) survives.
 * ------------------------------------------------------------------------- */
extern "C" axiom_engine_kv_arena *axiom_engine_kv_arena_create(axiom_runtime *rt,
                                                               uint32_t n_layers,
                                                               uint32_t kv_dim,
                                                               int pool_blocks,
                                                               int max_context) {
    const uint32_t NL = n_layers, KVD = kv_dim;
    axiom_engine_kv_arena *ar =
        (axiom_engine_kv_arena *)calloc(1, sizeof(axiom_engine_kv_arena));
    if (!ar) return NULL;
    ar->rt = rt;
    ar->kv_bt = (uint32_t)engine_env_int("AXIOM_KV_BLOCK", 16);
    if (ar->kv_bt == 0) ar->kv_bt = 16;
    {
        /* Pool size precedence: per-model override (manifest-plumbed by the
         * caller, > 0) -> AXIOM_KV_POOL_BLOCKS env (one GLOBAL knob across all
         * arenas in the process — kept as default/fallback) -> computed default
         * AXIOM_MAX_BATCH * ceil(max_context/BT). max_context > 0 likewise
         * overrides AXIOM_KV_MAX_CONTEXT inside that default formula. 0/0 =>
         * the historical env/default behavior, byte-identical. */
        int max_batch = engine_env_int("AXIOM_MAX_BATCH", 8); if (max_batch < 1) max_batch = 1;
        int max_ctx = (max_context > 0) ? max_context
                                        : engine_env_int("AXIOM_KV_MAX_CONTEXT", 4096);
        if (max_ctx < (int)ar->kv_bt) max_ctx = (int)ar->kv_bt;
        int per_seq = (max_ctx + (int)ar->kv_bt - 1) / (int)ar->kv_bt;
        ar->kv_nphys = (pool_blocks > 0)
                           ? pool_blocks
                           : engine_env_int("AXIOM_KV_POOL_BLOCKS", max_batch * per_seq);
        if (ar->kv_nphys < 1) ar->kv_nphys = 1;
    }
    size_t pool_elems = (size_t)ar->kv_nphys * NL * ar->kv_bt * KVD;
    /* Pool payload placement. Default = DEVICE: same sizing, same
     * (((pb*NL+L)*BT+slot)*KVD) addressing, just device-allocated. Kill-switch
     * AXIOM_ENGINE_DEVICE_ATTN=0 forces the host pool; an allocation failure
     * (VRAM pressure) logs once and falls back to the host pool + host
     * attention path (graceful, never fatal). The arena is resident-lifetime
     * either way, so a device pool persists across engine sessions exactly
     * like the host one (prefix-cached blocks keep their physical ids; warm
     * reuse = zero copies). */
    ar->device = 0;
    ar->d_kpool = ar->d_vpool = NULL;
    ar->kpool = ar->vpool = NULL;
    {
        const char *sw = getenv("AXIOM_ENGINE_DEVICE_ATTN");
        int want_device = !(sw && *sw && strtol(sw, NULL, 10) == 0);
        if (want_device && rt) {
            uint64_t pool_bytes = (uint64_t)pool_elems * 4u;
            int drc = axiom_device_buffer_create(rt, &ar->d_kpool, pool_bytes);
            if (drc == AXIOM_OK) drc = axiom_device_buffer_create(rt, &ar->d_vpool, pool_bytes);
            if (drc == AXIOM_OK) {
                ar->device = 1;
                fprintf(stderr, "axiom-kv: DEVICE paged-KV arena (blocks=%d bt=%u %.1f MiB x2)\n",
                        ar->kv_nphys, ar->kv_bt, (double)pool_bytes / (1024.0 * 1024.0));
            } else {
                axiom_device_buffer_destroy(ar->d_kpool); ar->d_kpool = NULL;
                axiom_device_buffer_destroy(ar->d_vpool); ar->d_vpool = NULL;
                fprintf(stderr, "axiom-kv: device pool alloc failed (rc=%d, %.1f MiB x2) — "
                                "falling back to HOST paged-KV\n",
                        drc, (double)pool_bytes / (1024.0 * 1024.0));
            }
        }
    }
    if (!ar->device) {
        ar->kpool = (float *)malloc(pool_elems * 4);
        ar->vpool = (float *)malloc(pool_elems * 4);
    }
    ar->free_list = (int *)malloc((size_t)ar->kv_nphys * sizeof(int));
    ar->refcount = (int *)calloc((size_t)ar->kv_nphys, sizeof(int));
    if ((!ar->device && (!ar->kpool || !ar->vpool)) || !ar->free_list || !ar->refcount) {
        free(ar->kpool); free(ar->vpool); free(ar->free_list); free(ar->refcount);
        axiom_device_buffer_destroy(ar->d_kpool);
        axiom_device_buffer_destroy(ar->d_vpool);
        free(ar); return NULL;
    }
    for (int b = 0; b < ar->kv_nphys; b++) ar->free_list[b] = ar->kv_nphys - 1 - b; /* block 0 on top */
    ar->n_free = ar->kv_nphys;
    ar->prefix = NULL;   /* the family hangs (and owns) its prefix-cache state here */
    return ar;
}

extern "C" void axiom_engine_kv_arena_destroy(axiom_engine_kv_arena *ar) {
    if (!ar) return;
    /* The family must have freed its `prefix` state already (it owns it). */
    free(ar->kpool); free(ar->vpool); free(ar->free_list); free(ar->refcount);
    /* device pools (NULL-safe; the resident destroys the arena BEFORE its runtime). */
    axiom_device_buffer_destroy(ar->d_kpool);
    axiom_device_buffer_destroy(ar->d_vpool);
    free(ar);
}

/* Allocate one physical block. Prefer the free-list; when exhausted, ask the
 * family to EVICT an idle cached block (dropping its cache entry) and reuse it.
 * Returns -1 only when nothing is free AND nothing is evictable (every block is
 * refcount>0): graceful back-pressure (admit) / clean hard error (decode). The
 * returned block has refcount 0; the caller increments it on assignment. */
static int engine_block_alloc(axiom_engine *e) {
    axiom_engine_kv_arena *ar = e->arena;
    if (ar->n_free > 0) return ar->free_list[--ar->n_free];
    if (e->ops.prefix_evict_idle) return e->ops.prefix_evict_idle(e->fam, ar);
    return -1;                                   /* exhausted, nothing evictable -> back-pressure */
}

/* Ensure the seq's block_table covers logical block `lb` (alloc fresh physical blocks
 * for any missing entries up to lb). Each freshly-assigned block gets refcount++ for
 * this seq. Returns 0 on success, -1 on pool exhaustion (out-of-blocks). Never overruns
 * block_table (cap = ceil(maxseq/BT) > any reachable lb). */
static int engine_seq_ensure_block(axiom_engine *e, axiom_engine_seq *st, int lb) {
    while (st->n_blocks <= lb) {
        if (st->n_blocks >= st->blocks_cap) return -1;   /* defensive: never exceed the table */
        int pb = engine_block_alloc(e);
        if (pb < 0) return -1;
        st->block_table[st->n_blocks++] = pb;
        e->arena->refcount[pb]++;                        /* this seq now references pb */
    }
    return 0;
}

static int engine_materialize_device_greedy(axiom_engine_seq *st) {
    if (!st || !st->device_greedy_tok_pending) return 0;
    if (!st->device_greedy_tok_materialize) {
        st->device_greedy_tok_pending = 0;
        st->device_greedy_tok_valid = 0;
        st->device_greedy_tok_resident = 0;
        return AXIOM_ERR_RUNTIME;
    }
    int rc = st->device_greedy_tok_materialize(st->device_greedy_tok_materialize_ctx, st);
    if (rc == AXIOM_OK) st->device_greedy_tok_pending = 0;
    return rc;
}

/* Drop ALL of a seq's block references (refcount-- each, exactly once) then free the
 * seq. A block whose refcount hits 0 is either KEPT allocated by the family (a cached
 * shared prefix block, evictable later) or returned to the free-list. Shared prefix
 * blocks are NEVER freed while another seq still references them. */
static void engine_seq_release(axiom_engine *e, axiom_engine_seq *st) {
    if (!st) return;
    (void)engine_materialize_device_greedy(st);
    if (e && e->arena && st->block_table) {
        axiom_engine_kv_arena *ar = e->arena;
        for (int b = 0; b < st->n_blocks; b++) {
            int pb = st->block_table[b];
            if (pb < 0 || pb >= ar->kv_nphys) continue;
            if (--ar->refcount[pb] == 0) {
                if (!(e->ops.prefix_block_idle && e->ops.prefix_block_idle(e->fam, ar, pb)))
                    ar->free_list[ar->n_free++] = pb;    /* return to free-list */
            }
        }
    }
    free(st->ids); free(st->block_table); free(st->seq_logits); free(st);
}

/* Grow the engine's active list + the family's batched scratch to hold `cap`
 * seq-slots. Contents are pure scratch (recomputed each step), so free+malloc
 * is safe and numerically inert. */
static int engine_ensure_scratch(axiom_engine *e, int cap) {
    if (cap <= e->active_cap) return 0;
    free(e->active);
    e->active = (axiom_engine_seq **)malloc((size_t)cap * sizeof(axiom_engine_seq *));
    e->active_cap = cap;
    e->ops.ensure_batch_scratch(e->fam, cap);
    return 0;
}

/* Free the engine's device bookkeeping (NOT the arena pools — those are
 * resident-lifetime). NULL-safe; resets cap/stride so a later ensure re-creates. */
static void engine_dev_scratch_destroy(axiom_engine *e) {
    axiom_engine_dev *d = &e->dev;
    axiom_device_buffer_destroy(d->d_tbl);  d->d_tbl  = NULL;
    axiom_device_buffer_destroy(d->d_meta); d->d_meta = NULL;
    free(d->htbl);  d->htbl  = NULL;
    free(d->hmeta); d->hmeta = NULL;
    d->cap = 0; d->tbl_stride = 0;
}

/* Grow the engine's DEVICE bookkeeping to >= cap seq-slots with a block-table
 * row of >= min_stride int32 entries — d_tbl [cap][stride] i32, d_meta [2*cap]
 * u32 (cache_tokens then pos at offset cap*4) + mirrored host staging — and the
 * family's staging rows (ops->dev_scratch_ensure) at the same trigger points.
 * Contents are PURE per-step scratch (tables/lengths/rows re-uploaded before
 * every use), so destroy+recreate on growth is numerically inert. On allocation
 * failure everything engine-side is torn down and a graceful rc is returned
 * (the step/admit fails cleanly; a later call retries). */
static int engine_ensure_dev_scratch(axiom_engine *e, int cap, int min_stride) {
    axiom_engine_dev *d = &e->dev;
    if (min_stride < 1) min_stride = 1;
    if (cap < 1) cap = 1;
    int ncap = cap > d->cap ? cap : d->cap;
    uint32_t nstride = (uint32_t)min_stride > d->tbl_stride ? (uint32_t)min_stride : d->tbl_stride;
    int rc = AXIOM_OK;
    if (!(cap <= d->cap && (uint32_t)min_stride <= d->tbl_stride)) {
        engine_dev_scratch_destroy(e);
        axiom_runtime *rt = d->rt;
        rc = axiom_device_buffer_create(rt, &d->d_tbl, (uint64_t)ncap * nstride * sizeof(int32_t));
        if (rc == AXIOM_OK) rc = axiom_device_buffer_create(rt, &d->d_meta, (uint64_t)2 * ncap * sizeof(uint32_t));
        if (rc == AXIOM_OK) {
            d->htbl  = (int32_t *)calloc((size_t)ncap * nstride, sizeof(int32_t));
            d->hmeta = (uint32_t *)calloc((size_t)2 * ncap, sizeof(uint32_t));
            if (!d->htbl || !d->hmeta) rc = AXIOM_ERR_RUNTIME;
        }
        if (rc == AXIOM_OK) {
            d->cap = ncap;
            d->tbl_stride = nstride;
        }
    }
    /* family staging rows grow in lockstep (no-op when already large enough) */
    if (rc == AXIOM_OK && e->ops.dev_scratch_ensure)
        rc = e->ops.dev_scratch_ensure(e->fam, ncap);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "axiom-kv: device scratch alloc failed rc=%d (cap=%d stride=%u)\n",
                rc, ncap, nstride);
        engine_dev_scratch_destroy(e);
        return rc;
    }
    return AXIOM_OK;
}

static int engine_push_running(axiom_engine *e, axiom_engine_seq *st) {
    if (e->n_running >= e->running_cap) {
        int nc = e->running_cap ? e->running_cap * 2 : 8;
        axiom_engine_seq **nr =
            (axiom_engine_seq **)realloc(e->running, (size_t)nc * sizeof(axiom_engine_seq *));
        if (!nr) return 1;
        e->running = nr; e->running_cap = nc;
    }
    e->running[e->n_running++] = st;
    return 0;
}

static void engine_emit_piece(axiom_engine *e, axiom_engine_seq *st,
                              uint32_t id, int is_eos) {
    if (!e || !st || !st->on_piece || is_eos || id >= e->vocab) return;
    char tb[96];
    int tl = e->ops.detok(e->fam, id, tb, (int)sizeof tb);
    if (tl < 0) {
        /* Live streaming assumes a single reset marker: gemma-4 emits one
         * end-of-thinking; more would diverge from collect's repeated clears. */
        st->stream_after_reset = 1;
        return;
    }
    if (tl <= 0) return;
    if (e->ops.stream_after_reset_marker && !st->stream_after_reset) return;
    st->on_piece(st->piece_ctx, st->piece_seq, tb, tl);
}

/* Create a step engine BORROWING the family's persistent KV arena (lazily
 * created by ops->kv_arena, resident lifetime — the cache spans sessions). */
extern "C" axiom_engine *axiom_engine_create(const axiom_engine_family_ops *ops, void *fam,
                                             uint32_t vocab, const uint32_t *eos_ids,
                                             uint32_t n_eos) {
    if (!ops || !ops->prefill_token || !ops->forward_step || !ops->sample_token ||
        !ops->detok || !ops->ensure_batch_scratch || !ops->kv_arena || !ops->destroy)
        return NULL;
    axiom_engine *e = (axiom_engine *)calloc(1, sizeof *e);
    if (!e) return NULL;
    e->ops = *ops;
    e->fam = fam;
    e->vocab = vocab;
    e->n_eos = n_eos > 8 ? 8 : n_eos;
    for (uint32_t i = 0; i < e->n_eos; i++) e->eos_ids[i] = eos_ids[i];
    /* Lazily create the resident-lifetime arena; the engine BORROWS its pool
     * (destroy never frees it; the resident does). On OOM the engine creation
     * fails cleanly (NULL) -> the caller keeps its existing non-engine path. */
    e->arena = e->ops.kv_arena(fam);
    if (!e->arena) { free(e); return NULL; }
    e->kv_bt = e->arena->kv_bt;
    e->kv_nphys = e->arena->kv_nphys;
    /* device view — borrow the arena's device pools; the table/length
     * bookkeeping is allocated lazily (engine_ensure_dev_scratch) on first
     * admit/step. */
    e->dev_attn = e->arena->device;
    if (e->dev_attn) {
        e->dev.rt = e->arena->rt;
        e->dev.d_kpool = e->arena->d_kpool;
        e->dev.d_vpool = e->arena->d_vpool;
    }
    e->last_reuse_blocks = 0;
    e->last_prefill_start = 0;
    return e;
}

/* PREFILL + alloc + join running set: init one seq (same rng seed mix, same
 * maxseq), family prefix lookup, then suffix-only serial prefill via
 * ops->prefill_token (want_logits only on the last prompt token), then family
 * prefix registration. On prefill fault the seq is freed and NOT joined; the
 * caller aborts and destroys. Returns 0 on success. */
extern "C" int axiom_engine_admit(axiom_engine *e, uint64_t sid,
                                  const uint32_t *ids, int n_in, float temp, int top_k,
                                  float top_p, unsigned int seed, int max_new,
                                  int out_cap, int use_eos,
                                  void *piece_ctx, axiom_engine_on_piece on_piece,
                                  int piece_seq) {
    if (!e) return 2;
    if (max_new < 0) max_new = 0;
    int ni = n_in > 0 ? n_in : 0;
    if (ni > 0 && !ids) return 2;
    const uint32_t VOCAB = e->vocab;
    if (e->ops.bind) e->ops.bind(e->fam);

    axiom_engine_seq *st = (axiom_engine_seq *)calloc(1, sizeof *st);
    if (!st) return 1;
    st->sid = sid;
    st->n_in = ni;
    st->max_new = max_new;
    st->out_cap = out_cap;
    st->maxseq = (uint32_t)(ni + max_new + 4);
    st->ids_cap = ni + max_new + 4;
    st->ids = (uint32_t *)malloc((size_t)(st->ids_cap > 0 ? st->ids_cap : 1) * sizeof(uint32_t));
    for (int i = 0; i < ni; i++) st->ids[i] = ids[i];
    /* per-seq block table (physical blocks alloc'd lazily as pos grows). */
    st->blocks_cap = (int)(((size_t)st->maxseq + e->kv_bt - 1) / e->kv_bt);
    if (st->blocks_cap < 1) st->blocks_cap = 1;
    st->block_table = (int *)malloc((size_t)st->blocks_cap * sizeof(int));
    st->n_blocks = 0;
    st->seq_logits = (float *)malloc((size_t)VOCAB * 4);
    if (!st->ids || !st->block_table || !st->seq_logits) { engine_seq_release(e, st); return 1; }
    st->temp = temp; st->top_k = top_k; st->top_p = top_p; st->seed = seed;
    st->use_eos = use_eos;
    st->piece_ctx = piece_ctx;
    st->on_piece = on_piece;
    st->piece_seq = piece_seq;
    st->stream_after_reset = 0;
    st->rng = seed ? ((uint64_t)seed * 2654435761ull + 0x9E3779B97F4A7C15ull)
                   : 0x9E3779B97F4A7C15ull;
    st->pos = 0; st->n_out = 0; st->generated = 0;
    st->done = (ni == 0);

    int reuse = 0;   /* prefix blocks reused from the cache (observability + prefill start) */
    if (!st->done && ni > 0) {
        const int BT = (int)e->kv_bt;
        /* WARM LOOKUP (family prefix cache): reuse shared prefix blocks into
         * block_table[0..reuse); prefill starts at reuse*BT (BT-aligned =>
         * never a slot inside a shared block => shared KV is immutable). */
        if (e->ops.prefix_lookup)
            reuse = e->ops.prefix_lookup(e->fam, e->arena, st);
        /* SUFFIX-ONLY PREFILL, want_logits on the last token. In device mode
         * the SAME loop runs with the paged kernels at active=1 per token —
         * reused blocks' payload is already in the device pool, the suffix
         * scatters/attends past them through the same table. */
        if (e->dev_attn) {
            int rc = engine_ensure_dev_scratch(e, 1, st->blocks_cap);
            if (rc) { engine_seq_release(e, st); return rc; }
        }
        for (int i = reuse * BT; i < ni; i++) {
            /* alloc the physical block for this prefill token on a block-boundary cross;
             * pool exhaustion here is graceful admission back-pressure (seq not joined). */
            if (engine_seq_ensure_block(e, st, i / BT)) { engine_seq_release(e, st); return AXIOM_ERR_RUNTIME; }
            if (e->dev_attn) {
                /* per-token metadata upload: the seq's table prefix (entries 0..lb
                 * cover every token t <= i), its store position and its attention
                 * length, all at row 0 (active=1). The table can grow one entry per
                 * BT-crossing during prefill, so this re-upload per token keeps the
                 * device view exact. ct rides at d_meta offset 0, pos at the fixed
                 * offset cap*4 (one packed upload). */
                axiom_engine_dev *d = &e->dev;
                const uint32_t lb = (uint32_t)i / e->kv_bt;
                d->hmeta[0] = (uint32_t)i + 1u;      /* cache_tokens[0] */
                d->hmeta[d->cap] = (uint32_t)i;      /* pos[0] */
                int rc = axiom_device_buffer_upload(d->d_tbl, 0, st->block_table,
                                                    (uint64_t)(lb + 1u) * sizeof(int32_t));
                if (rc == AXIOM_OK)
                    rc = axiom_device_buffer_upload(d->d_meta, 0, d->hmeta,
                                                    ((uint64_t)d->cap + 1u) * sizeof(uint32_t));
                if (rc != AXIOM_OK) { engine_seq_release(e, st); return rc; }
            }
            int rc = e->ops.prefill_token(e->fam, st, st->ids[i], (uint32_t)i,
                                          (i == ni - 1), e->dev_attn ? &e->dev : NULL);
            if (rc) { engine_seq_release(e, st); return rc; }
        }
        st->pos = ni;
        /* REGISTER the newly-written FULL blocks (family prefix cache,
         * first-writer-wins); never the partial last block (mutable — decode
         * appends into it). */
        if (e->ops.prefix_register)
            e->ops.prefix_register(e->fam, e->arena, st, reuse);
    }
    e->last_reuse_blocks = reuse;
    e->last_prefill_start = reuse * (int)e->kv_bt;
    if (engine_push_running(e, st)) { engine_seq_release(e, st); return 1; }
    engine_ensure_scratch(e, e->n_running);
    return 0;
}

extern "C" int axiom_engine_num_active(axiom_engine *e) {
    if (!e) return 0;
    int n = 0;
    for (int i = 0; i < e->n_running; i++) if (!e->running[i]->done) n++;
    return n;
}

/* ONE lockstep decode step over the current running set: selection (per-seq
 * sample from current logits, own rng) -> active subset -> ensure blocks ->
 * device table/length upload -> ONE family batched forward -> advance pos. */
extern "C" int axiom_engine_step(axiom_engine *e) {
    if (!e) return 2;
    const uint32_t VOCAB = e->vocab;
    int rc = 0;
    if (e->ops.bind) e->ops.bind(e->fam);
    engine_ensure_scratch(e, e->n_running);

    /* selection: each !done seq picks next token from its current logits */
    int A = 0;
    for (int i = 0; i < e->n_running; i++) {
        axiom_engine_seq *st = e->running[i];
        if (st->done) continue;
        st->next_tok_device_resident = 0;
        if ((rc = engine_materialize_device_greedy(st))) return rc;
        if (st->n_out >= st->out_cap) {
            st->device_greedy_tok_valid = 0;
            st->device_greedy_tok_resident = 0;
            st->done = 1;
            continue;
        }
        float *lg = st->seq_logits;
        uint32_t best;
        if (st->temp > 0.0f) {
            uint32_t amax = 0; float amaxv = lg[0];
            for (uint32_t j = 1; j < VOCAB; j++) if (lg[j] > amaxv) { amaxv = lg[j]; amax = j; }
            (void)amax;
            best = e->ops.sample_token(e->fam, lg, VOCAB, st->temp, st->top_k, st->top_p,
                                       &st->rng);
            st->device_greedy_tok_valid = 0;
            st->device_greedy_tok_resident = 0;
            st->device_greedy_tok_pending = 0;
        } else if (st->device_greedy_tok_valid) {
            best = st->device_greedy_tok;
            st->next_tok_device_resident = st->device_greedy_tok_resident;
            st->device_greedy_tok_valid = 0;
            st->device_greedy_tok_resident = 0;
        } else {
            uint32_t amax = 0; float amaxv = lg[0];
            for (uint32_t j = 1; j < VOCAB; j++) if (lg[j] > amaxv) { amaxv = lg[j]; amax = j; }
            best = amax;
            st->device_greedy_tok_resident = 0;
            st->device_greedy_tok_pending = 0;
        }
        st->ids[st->n_in + st->n_out] = best;
        st->n_out++;
        st->generated++;
        int is_eos = 0;
        if (st->use_eos)
            for (uint32_t k = 0; k < e->n_eos; k++) if (best == e->eos_ids[k]) { is_eos = 1; break; }
        engine_emit_piece(e, st, best, is_eos);
        if (is_eos) { st->done = 1; continue; }
        if (st->generated >= st->max_new) { st->done = 1; continue; }
        st->next_tok = best;
        e->active[A++] = st;
    }
    if (A == 0) return 0;

    /* ensure the physical block for each active seq's write position (st->pos)
     * exists BEFORE the forward (one block spans all layers). Pool exhaustion for
     * an in-flight seq is a CLEAN hard error: return rc, the caller fails the group
     * gracefully — no other seq is touched, no overrun. */
    for (int a = 0; a < A; a++) {
        axiom_engine_seq *st = e->active[a];
        if (engine_seq_ensure_block(e, st, st->pos / (int)e->kv_bt)) return AXIOM_ERR_RUNTIME;
    }

    /* device mode — upload the active seqs' block tables + lengths ONCE per step
     * (~8 KiB), AFTER the ensure-block loop so any block allocated this step is in
     * the device view. The kernels only ever read table entries for t <
     * cache_tokens[a] (= pos+1 <= n_blocks*BT), so rows are valid by construction. */
    if (e->dev_attn) {
        int need_stride = 1;
        for (int a = 0; a < A; a++)
            if (e->active[a]->blocks_cap > need_stride) need_stride = e->active[a]->blocks_cap;
        if ((rc = engine_ensure_dev_scratch(e, A, need_stride))) return rc;
        axiom_engine_dev *d = &e->dev;
        for (int a = 0; a < A; a++) {
            axiom_engine_seq *st = e->active[a];
            memcpy(d->htbl + (size_t)a * d->tbl_stride, st->block_table,
                   (size_t)st->n_blocks * sizeof(int32_t));
            d->hmeta[a]          = (uint32_t)st->pos + 1u;   /* cache_tokens */
            d->hmeta[d->cap + a] = (uint32_t)st->pos;        /* store pos */
        }
        rc = axiom_device_buffer_upload(d->d_tbl, 0, d->htbl,
                                        (uint64_t)A * d->tbl_stride * sizeof(int32_t));
        if (rc == AXIOM_OK)
            rc = axiom_device_buffer_upload(d->d_meta, 0, d->hmeta,
                                            ((uint64_t)d->cap + A) * sizeof(uint32_t));
        if (rc != AXIOM_OK) return rc;
    }

    /* forward phase: ONE family batched forward over the active subset; each
     * seq's next logits land in seq_logits, then the engine advances pos. */
    if ((rc = e->ops.forward_step(e->fam, e->active, A, e->dev_attn ? &e->dev : NULL)))
        return rc;
    for (int a = 0; a < A; a++) e->active[a]->pos++;
    return 0;
}

/* Copy a (finished) seq's generated ids + detok'd text out. Detok skips
 * eos/out-of-vocab ids, appends family-decoded bytes, caps at text_cap-1.
 * A NEGATIVE detok return is the family's RESET MARKER (see axiom_engine.h):
 * the text assembled so far for this seq is discarded and assembly restarts
 * from the next id — reproducing the family solo path's clear-on-marker text
 * semantics. Ids are returned in full either way. */
extern "C" int axiom_engine_collect(axiom_engine *e, uint64_t sid,
                                    uint32_t *out_ids, int *n_out, char *text, int text_cap) {
    if (!e) return 2;
    axiom_engine_seq *st = NULL;
    for (int i = 0; i < e->n_running; i++) if (e->running[i]->sid == sid) { st = e->running[i]; break; }
    if (!st) return 2;
    if (out_ids) for (int i = 0; i < st->n_out; i++) out_ids[i] = st->ids[st->n_in + i];
    if (n_out) *n_out = st->n_out;
    if (text && text_cap > 0) {
        int cap_len = 0;
        for (int i = 0; i < st->n_out; i++) {
            uint32_t id = st->ids[st->n_in + i];
            int is_eos = 0;
            for (uint32_t k = 0; k < e->n_eos; k++) if (id == e->eos_ids[k]) { is_eos = 1; break; }
            if (id >= e->vocab || is_eos) continue;
            char tb[96]; int tl = e->ops.detok(e->fam, id, tb, (int)sizeof tb);
            if (tl < 0) { cap_len = 0; continue; }   /* family reset marker: drop assembled text */
            for (int z = 0; z < tl && cap_len < text_cap - 1; z++) text[cap_len++] = tb[z];
        }
        text[cap_len] = '\0';
    }
    return 0;
}

/* Report the first seq that finished (done) this step but is still in the running
 * set, so the continuous driver can collect/fulfill/retire it and free its slot.
 * Returns 1 (writes *sid_out) when such a seq exists, else 0. Pure read of the
 * per-seq done flag — numerically inert. */
extern "C" int axiom_engine_next_finished(axiom_engine *e, uint64_t *sid_out) {
    if (!e) return 0;
    for (int i = 0; i < e->n_running; i++) {
        if (e->running[i]->done) {
            if (sid_out) *sid_out = e->running[i]->sid;
            return 1;
        }
    }
    return 0;
}

extern "C" void axiom_engine_retire(axiom_engine *e, uint64_t sid) {
    if (!e) return;
    for (int i = 0; i < e->n_running; i++) {
        if (e->running[i]->sid == sid) {
            engine_seq_release(e, e->running[i]);   /* returns its blocks to the free-list */
            for (int j = i + 1; j < e->n_running; j++) e->running[j - 1] = e->running[j];
            e->n_running--;
            return;
        }
    }
}

/* Observability — cumulative family prefix-cache counters (arena-lifetime) +
 * the per-engine most-recent-admit reuse (blocks reused, first prefilled
 * token). Lets the daemon log warmth so reuse is confirmable. */
extern "C" void axiom_engine_cache_stats(axiom_engine *e,
        uint64_t *reused_total, uint64_t *registered_total,
        int *last_reuse_blocks, int *last_prefill_start) {
    uint64_t ru = 0, rg = 0;
    if (e && e->arena && e->ops.cache_totals)
        e->ops.cache_totals(e->fam, e->arena, &ru, &rg);
    if (reused_total)     *reused_total     = ru;
    if (registered_total) *registered_total = rg;
    if (last_reuse_blocks)  *last_reuse_blocks  = e ? e->last_reuse_blocks  : 0;
    if (last_prefill_start) *last_prefill_start = e ? e->last_prefill_start : 0;
}

extern "C" void axiom_engine_destroy(axiom_engine *e) {
    if (!e) return;
    /* Refcount-release each running seq (cached prefix blocks survive in the arena
     * for the next session; uncached blocks return to the free-list). The POOL +
     * the family's prefix cache are arena-owned (resident lifetime), NOT freed here. */
    for (int i = 0; i < e->n_running; i++) engine_seq_release(e, e->running[i]);
    /* Free the engine's device bookkeeping; the arena's DEVICE pools (like its
     * host pool) are resident-lifetime and survive for the next session's warm
     * reuse. */
    engine_dev_scratch_destroy(e);
    free(e->running);
    free(e->active);
    /* Family ctx LAST (its scratch, staging rows and the ctx itself). */
    e->ops.destroy(e->fam);
    free(e);
}
