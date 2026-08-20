/* W5a STAGE-K gate: paged attention cores + paged KV store vs the oracle-blessed CONTIGUOUS
 * attention cores — BYTE-EQUAL.
 *
 * For every config in the matrix
 *   ct {1,15,16,17,31,100,2048} x BT {1,16,17,128} x tables {identity,scrambled}
 *   x window {0,512} x pool_stride {kv_dim, 2048 (wider-than-kv_dim)} x kv_heads {8 GQA, 1 MQA}
 *   x variants {blk (1/sqrt(hd) scale), scale (caller scale 0.7071)}
 * the harness:
 *   1. builds TWO active seqs (proves blockIdx.y / per-seq tables / per-seq cache_tokens):
 *      seq0 = the config's sequence (table mode applies); seq1 = a decoy with its OWN random
 *      KV, an always-scrambled DISJOINT block set, and a RAGGED attention length
 *      ct1 = ct0/2+1 < ct0 (stored ct0 tokens, attends over ct1 — proves the t<cache_tokens[a]
 *      bound in one launch);
 *   2. poisons the ENTIRE pools with 0xFF bytes (= -NaN f32) and points every UNUSED block-table
 *      slot at a never-stored physical block — any read outside the table/layer/stride contract
 *      turns the online softmax into NaN and fails the byte-compare;
 *   3. scatters the KV through axiom_runtime_kv_pool_store_paged_f32_device, one launch per
 *      token with active=2 and PER-SEQ positions (seq1 stores in REVERSE order — proves pos[a]);
 *   4. for small cts (<=31, all BT boundary cases) downloads the pools and bit-verifies every
 *      element against the host-computed addressing (anti-collusion: a store bug cannot be
 *      cancelled by a matching gather bug) — stored rows exact, everything else still 0xFF;
 *   5. runs ONE paged attention launch (active=2) and, per seq, the CONTIGUOUS oracle-blessed
 *      kernel on a linear cache holding the same logical KV; the outputs must be BYTE-EQUAL
 *      (paging is pure storage relocation: same bytes, same strictly ascending-t order).
 * Pool geometry uses NL=3 layers with the data in the MIDDLE layer L=1 (proves the
 * (pb*NL+L)*BT+slot term); tbl_stride = n_logical+3 (proves the a*tbl_stride row stride);
 * pool_stride 2048 > kv_dim (proves the row stride is decoupled from kv_heads*head_dim).
 * Prints one PASS line per config; exits non-zero on ANY mismatch. */
#include "axiom/axiom.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kQHeads = 16u;
constexpr uint32_t kHeadDim = 128u;
constexpr uint32_t kNL = 3u;  /* pool layers; data lives in the middle one */
constexpr uint32_t kLayer = 1u;
constexpr float kCallerScale = 0.70710678f;
constexpr uint32_t kPoisonBits = 0xFFFFFFFFu; /* all-FF f32 = negative quiet NaN */

struct Cfg {
    uint32_t ct;
    uint32_t bt;
    bool scrambled;
    uint32_t window;
    uint32_t pool_stride; /* 0 = pack exactly kv_dim */
    uint32_t kv_heads;
    bool scale_variant;
};

struct DevBuf {
    axiom_device_buffer *b = nullptr;
    ~DevBuf() {
        if (b) axiom_device_buffer_destroy(b);
    }
};

const char *cfg_tbl(const Cfg &c) { return c.scrambled ? "scrambled" : "identity"; }
const char *cfg_var(const Cfg &c) { return c.scale_variant ? "scale" : "blk"; }

void print_cfg(const char *status, const Cfg &c, uint32_t ps, const char *detail) {
    printf("%s var=%s ct=%u bt=%u tbl=%s win=%u ps=%u kvh=%u%s%s\n",
           status, cfg_var(c), c.ct, c.bt, cfg_tbl(c), c.window, ps, c.kv_heads,
           detail[0] ? " " : "", detail);
    fflush(stdout);
}

bool run_cfg(axiom_runtime *rt, const Cfg &c) {
    const uint32_t kv_dim = c.kv_heads * kHeadDim;
    const uint32_t ps = c.pool_stride ? c.pool_stride : kv_dim;
    const uint32_t ct0 = c.ct;
    const uint32_t ct1 = ct0 / 2u + 1u; /* ragged decoy length (<= ct0) */
    const uint32_t n_log = (ct0 + c.bt - 1u) / c.bt; /* logical blocks per seq (both store ct0) */
    const uint32_t n_phys = 2u * n_log + 5u; /* + spares; last one = table-poison block */
    const uint32_t tbl_stride = n_log + 3u;
    const uint32_t qdim = kQHeads * kHeadDim;
    char detail[160];
    detail[0] = '\0';

#define FAILCFG(...)                                  \
    do {                                              \
        snprintf(detail, sizeof(detail), __VA_ARGS__); \
        print_cfg("FAIL", c, ps, detail);             \
        return false;                                 \
    } while (0)
#define RC_CHECK(expr, what)                                        \
    do {                                                            \
        const int rc_ = (expr);                                     \
        if (rc_ != AXIOM_OK) FAILCFG("%s rc=%d", (what), rc_);      \
    } while (0)

    std::mt19937 rng(0x5EEDu ^ (ct0 * 2654435761u) ^ (c.bt * 40503u) ^ (c.window * 977u) ^
                     (ps * 31u) ^ (c.kv_heads * 7u) ^ (c.scrambled ? 0x10000u : 0u) ^
                     (c.scale_variant ? 0x20000u : 0u));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    /* ---- host data: q for both seqs + independent logical KV per seq ---- */
    std::vector<float> hq(2u * (size_t)qdim);
    for (auto &v : hq) v = dist(rng);
    std::vector<float> hk[2], hv[2];
    for (int s = 0; s < 2; ++s) {
        hk[s].resize((size_t)ct0 * kv_dim);
        hv[s].resize((size_t)ct0 * kv_dim);
        for (auto &v : hk[s]) v = dist(rng);
        for (auto &v : hv[s]) v = dist(rng);
    }

    /* ---- block tables: seq0 identity|scrambled, seq1 ALWAYS scrambled, disjoint sets;
       every unused table slot points at the never-stored poison block ---- */
    std::vector<int32_t> ids(n_phys - 1u);
    for (uint32_t i = 0; i < n_phys - 1u; ++i) ids[i] = (int32_t)i;
    const int32_t poison_block = (int32_t)(n_phys - 1u);
    if (c.scrambled) {
        std::shuffle(ids.begin(), ids.end(), rng); /* both seqs interleaved over the pool */
    } else {
        std::shuffle(ids.begin() + n_log, ids.end(), rng); /* seq0 stays identity 0..n_log-1 */
    }
    std::vector<int32_t> tbl(2u * (size_t)tbl_stride, poison_block);
    std::vector<int32_t> seq_of(n_phys, -1), lb_of(n_phys, -1);
    for (uint32_t lb = 0; lb < n_log; ++lb) {
        tbl[lb] = ids[lb];
        tbl[(size_t)tbl_stride + lb] = ids[(size_t)n_log + lb];
        seq_of[(uint32_t)ids[lb]] = 0;
        lb_of[(uint32_t)ids[lb]] = (int32_t)lb;
        seq_of[(uint32_t)ids[(size_t)n_log + lb]] = 1;
        lb_of[(uint32_t)ids[(size_t)n_log + lb]] = (int32_t)lb;
    }

    /* ---- staging + per-seq positions: launch t stores seq0 token t AND seq1 token ct0-1-t
       (reverse order for seq1 — proves pos[a] is per-seq); end state is position-correct ---- */
    std::vector<float> hsk((size_t)ct0 * 2u * kv_dim), hsv((size_t)ct0 * 2u * kv_dim);
    std::vector<uint32_t> hpos((size_t)ct0 * 2u);
    for (uint32_t t = 0; t < ct0; ++t) {
        const uint32_t t1 = ct0 - 1u - t;
        memcpy(&hsk[((size_t)t * 2u + 0u) * kv_dim], &hk[0][(size_t)t * kv_dim],
               (size_t)kv_dim * sizeof(float));
        memcpy(&hsk[((size_t)t * 2u + 1u) * kv_dim], &hk[1][(size_t)t1 * kv_dim],
               (size_t)kv_dim * sizeof(float));
        memcpy(&hsv[((size_t)t * 2u + 0u) * kv_dim], &hv[0][(size_t)t * kv_dim],
               (size_t)kv_dim * sizeof(float));
        memcpy(&hsv[((size_t)t * 2u + 1u) * kv_dim], &hv[1][(size_t)t1 * kv_dim],
               (size_t)kv_dim * sizeof(float));
        hpos[(size_t)t * 2u + 0u] = t;
        hpos[(size_t)t * 2u + 1u] = t1;
    }
    const uint32_t hct[2] = {ct0, ct1};

    /* ---- device buffers ---- */
    const uint64_t pool_elems = (uint64_t)n_phys * kNL * c.bt * ps;
    const uint64_t pool_bytes = pool_elems * sizeof(float);
    std::vector<uint32_t> hpoison(pool_elems, kPoisonBits);

    DevBuf dq, dkpool, dvpool, dtbl, dct, dpos, dsk, dsv, dout, dlink, dlinv, dref;
    RC_CHECK(axiom_device_buffer_create(rt, &dq.b, hq.size() * sizeof(float)), "create q");
    RC_CHECK(axiom_device_buffer_create(rt, &dkpool.b, pool_bytes), "create kpool");
    RC_CHECK(axiom_device_buffer_create(rt, &dvpool.b, pool_bytes), "create vpool");
    RC_CHECK(axiom_device_buffer_create(rt, &dtbl.b, tbl.size() * sizeof(int32_t)), "create tbl");
    RC_CHECK(axiom_device_buffer_create(rt, &dct.b, sizeof(hct)), "create ct");
    RC_CHECK(axiom_device_buffer_create(rt, &dpos.b, hpos.size() * sizeof(uint32_t)), "create pos");
    RC_CHECK(axiom_device_buffer_create(rt, &dsk.b, hsk.size() * sizeof(float)), "create sk");
    RC_CHECK(axiom_device_buffer_create(rt, &dsv.b, hsv.size() * sizeof(float)), "create sv");
    RC_CHECK(axiom_device_buffer_create(rt, &dout.b, 2u * (uint64_t)qdim * sizeof(float)),
             "create out");
    RC_CHECK(axiom_device_buffer_create(rt, &dlink.b, (uint64_t)ct0 * kv_dim * sizeof(float)),
             "create link");
    RC_CHECK(axiom_device_buffer_create(rt, &dlinv.b, (uint64_t)ct0 * kv_dim * sizeof(float)),
             "create linv");
    RC_CHECK(axiom_device_buffer_create(rt, &dref.b, (uint64_t)qdim * sizeof(float)),
             "create ref");

    RC_CHECK(axiom_device_buffer_upload(dq.b, 0, hq.data(), hq.size() * sizeof(float)), "up q");
    RC_CHECK(axiom_device_buffer_upload(dkpool.b, 0, hpoison.data(), pool_bytes), "poison kpool");
    RC_CHECK(axiom_device_buffer_upload(dvpool.b, 0, hpoison.data(), pool_bytes), "poison vpool");
    RC_CHECK(axiom_device_buffer_upload(dtbl.b, 0, tbl.data(), tbl.size() * sizeof(int32_t)),
             "up tbl");
    RC_CHECK(axiom_device_buffer_upload(dct.b, 0, hct, sizeof(hct)), "up ct");
    RC_CHECK(axiom_device_buffer_upload(dpos.b, 0, hpos.data(), hpos.size() * sizeof(uint32_t)),
             "up pos");
    RC_CHECK(axiom_device_buffer_upload(dsk.b, 0, hsk.data(), hsk.size() * sizeof(float)),
             "up sk");
    RC_CHECK(axiom_device_buffer_upload(dsv.b, 0, hsv.data(), hsv.size() * sizeof(float)),
             "up sv");
    /* poison the paged output too: a launch that fails to cover a (head, seq) cannot pass */
    std::vector<uint32_t> hout_poison(2u * (size_t)qdim, kPoisonBits);
    RC_CHECK(axiom_device_buffer_upload(dout.b, 0, hout_poison.data(),
                                        hout_poison.size() * sizeof(uint32_t)),
             "poison out");

    /* ---- scatter the KV through the paged store kernel (active=2, per-seq positions) ---- */
    for (uint32_t t = 0; t < ct0; ++t) {
        RC_CHECK(axiom_runtime_kv_pool_store_paged_f32_device(
                         rt, dsk.b, (uint64_t)t * 2u * kv_dim * sizeof(float), dsv.b,
                         (uint64_t)t * 2u * kv_dim * sizeof(float), dkpool.b, 0, dvpool.b, 0,
                         dtbl.b, 0, dpos.b, (uint64_t)t * 2u * sizeof(uint32_t), 2u, tbl_stride,
                         kNL, kLayer, c.bt, ps, kv_dim),
                 "kv_pool_store");
    }

    /* ---- anti-collusion: bit-verify the WHOLE pool layout on host (small cts cover every
       BT boundary case incl. partial last blocks); stored rows exact, all else still 0xFF ---- */
    if (ct0 <= 31u) {
        std::vector<uint32_t> got(pool_elems);
        const std::vector<float> *src[2][2] = {{&hk[0], &hk[1]}, {&hv[0], &hv[1]}};
        DevBuf *pools[2] = {&dkpool, &dvpool};
        for (int which = 0; which < 2; ++which) {
            RC_CHECK(axiom_device_buffer_download(pools[which]->b, 0, got.data(), pool_bytes),
                     "down pool");
            for (uint32_t pb = 0; pb < n_phys; ++pb) {
                for (uint32_t l = 0; l < kNL; ++l) {
                    for (uint32_t slot = 0; slot < c.bt; ++slot) {
                        const uint64_t row = (((uint64_t)pb * kNL + l) * c.bt + slot) * ps;
                        for (uint32_t lane = 0; lane < ps; ++lane) {
                            uint32_t want = kPoisonBits;
                            if (l == kLayer && seq_of[pb] >= 0 && lane < kv_dim) {
                                const uint32_t t = (uint32_t)lb_of[pb] * c.bt + slot;
                                if (t < ct0) {
                                    memcpy(&want,
                                           &(*src[which][seq_of[pb]])[(size_t)t * kv_dim + lane],
                                           sizeof(uint32_t));
                                }
                            }
                            if (got[row + lane] != want) {
                                FAILCFG("pool[%s] pb=%u l=%u slot=%u lane=%u got=0x%08x "
                                        "want=0x%08x",
                                        which ? "v" : "k", pb, l, slot, lane, got[row + lane],
                                        want);
                            }
                        }
                    }
                }
            }
        }
    }

    /* ---- ONE paged attention launch over both (ragged) seqs ---- */
    if (c.scale_variant) {
        RC_CHECK(axiom_runtime_attention_core_scale_paged_f32_device(
                         rt, dq.b, 0, dkpool.b, 0, dvpool.b, 0, dout.b, 0, dtbl.b, 0, dct.b, 0,
                         2u, tbl_stride, kNL, kLayer, c.bt, ps, kQHeads, c.kv_heads, kHeadDim,
                         kCallerScale, c.window),
                 "paged attn (scale)");
    } else {
        RC_CHECK(axiom_runtime_attention_core_paged_f32_device(
                         rt, dq.b, 0, dkpool.b, 0, dvpool.b, 0, dout.b, 0, dtbl.b, 0, dct.b, 0,
                         2u, tbl_stride, kNL, kLayer, c.bt, ps, kQHeads, c.kv_heads, kHeadDim,
                         c.window),
                 "paged attn (blk)");
    }
    std::vector<float> hout(2u * (size_t)qdim);
    RC_CHECK(axiom_device_buffer_download(dout.b, 0, hout.data(), hout.size() * sizeof(float)),
             "down out");

    /* ---- per-seq contiguous reference on a LINEAR cache: must be BYTE-EQUAL ---- */
    for (int s = 0; s < 2; ++s) {
        const uint32_t ct_s = (s == 0) ? ct0 : ct1;
        RC_CHECK(axiom_device_buffer_upload(dlink.b, 0, hk[s].data(),
                                            (uint64_t)ct_s * kv_dim * sizeof(float)),
                 "up lin k");
        RC_CHECK(axiom_device_buffer_upload(dlinv.b, 0, hv[s].data(),
                                            (uint64_t)ct_s * kv_dim * sizeof(float)),
                 "up lin v");
        if (c.scale_variant) {
            RC_CHECK(axiom_runtime_attention_core_scale_f32_device(
                             rt, dq.b, (uint64_t)s * qdim * sizeof(float), dlink.b, 0, dlinv.b, 0,
                             dref.b, 0, kQHeads, c.kv_heads, kHeadDim, ct_s, kCallerScale,
                             c.window),
                     "contiguous attn (scale)");
        } else {
            RC_CHECK(axiom_runtime_attention_core_fast_f32_device(
                             rt, dq.b, (uint64_t)s * qdim * sizeof(float), dlink.b, 0, dlinv.b, 0,
                             dref.b, 0, kQHeads, c.kv_heads, kHeadDim, ct_s, c.window),
                     "contiguous attn (blk)");
        }
        std::vector<float> href(qdim);
        RC_CHECK(axiom_device_buffer_download(dref.b, 0, href.data(), href.size() * sizeof(float)),
                 "down ref");
        if (memcmp(&hout[(size_t)s * qdim], href.data(), (size_t)qdim * sizeof(float)) != 0) {
            uint32_t bad = 0;
            uint32_t gb = 0, wb = 0;
            for (uint32_t i = 0; i < qdim; ++i) {
                memcpy(&gb, &hout[(size_t)s * qdim + i], 4);
                memcpy(&wb, &href[i], 4);
                if (gb != wb) {
                    bad = i;
                    break;
                }
            }
            FAILCFG("seq=%d ct_s=%u first_diff=%u got=0x%08x(%.9g) want=0x%08x(%.9g)", s, ct_s,
                    bad, gb, hout[(size_t)s * qdim + bad], wb, href[bad]);
        }
    }

    snprintf(detail, sizeof(detail), "(ct1=%u n_phys=%u pool=%.1fMB)", ct1, n_phys,
             (double)pool_bytes / (1024.0 * 1024.0));
    print_cfg("PASS", c, ps, detail);
    return true;
#undef RC_CHECK
#undef FAILCFG
}

} // namespace

int main(void) {
    axiom_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = AXIOM_ABI_VERSION;
    cfg.backend = AXIOM_BACKEND_CUDA;

    axiom_runtime *rt = nullptr;
    int rc = axiom_runtime_create(&rt, &cfg);
    if (rc != AXIOM_OK) {
        fprintf(stderr, "paged-attn-smoke: runtime rc=%d\n", rc);
        return 1;
    }

    const uint32_t cts[] = {1u, 15u, 16u, 17u, 31u, 100u, 2048u};
    const uint32_t bts[] = {1u, 16u, 17u, 128u};
    const bool tbls[] = {false, true};
    const uint32_t wins[] = {0u, 512u};
    const uint32_t pss[] = {0u, 2048u}; /* 0 = exactly kv_dim */
    const uint32_t kvhs[] = {8u, 1u};
    const bool vars[] = {false, true};

    uint32_t total = 0, passed = 0;
    for (bool var : vars)
        for (uint32_t kvh : kvhs)
            for (uint32_t ps : pss)
                for (uint32_t win : wins)
                    for (bool scram : tbls)
                        for (uint32_t bt : bts)
                            for (uint32_t ct : cts) {
                                Cfg c;
                                c.ct = ct;
                                c.bt = bt;
                                c.scrambled = scram;
                                c.window = win;
                                c.pool_stride = ps;
                                c.kv_heads = kvh;
                                c.scale_variant = var;
                                ++total;
                                if (run_cfg(rt, c)) ++passed;
                            }

    printf("paged-attn-smoke: %u/%u PASS%s\n", passed, total,
           passed == total ? "" : " — FAILURES PRESENT");
    axiom_runtime_destroy(rt);
    return passed == total ? 0 : 1;
}
