#!/usr/bin/env python3
"""GLM-5.2-NVFP4 safetensors -> Axiom glm52 GGUF converter — FAIL-CLOSED SCAFFOLDING.

Extends the PROVEN full-model converter pattern of tools/nvfp4_to_gguf_full.py
(imported, NOT modified) but is parameterized by the glm52 family descriptor
facts (include/axiom/glm52_state.h, src/axiom_glm52_family.cpp,
docs/glm52/GLM52_ONBOARDING.md). Same architecture-as-data idea on the tooling
side: the model-specific part is the DESC + ROLES tables below, not new code.

Encoding families (== ds4 GGML type codes, src/axiom_ds4_weights.cpp:19-29):
  * routed experts -> NVFP4 (type 17): 256 experts stacked into one 3D tensor
        [in, out, NEXP] of 9B/16 blocks (8 E2M1-nibble bytes + 1 F8_E4M3
        group-16 scale) + an F32 per-expert ".gscale" companion [NEXP].
        Byte-identical repack contract to the ds4 loader's
        read_tensor_nvfp4_deblock_upload (src/axiom_ds4_weights.cpp:734-813).
        VERIFIED class for GLM-5.2 (nvidia/GLM-5.2-NVFP4: MoE-expert linears
        NVFP4; tensor types U8 + F8_E4M3 + F32; modelopt 0.46.0).
  * fp8 class (type 18): raw E4M3 weight bytes + E8M0 ".scale" companion.
  * bf16 class (type 30 = standard GGML BF16 id): raw BF16 passthrough,
        byte-lossless. NOTE: type 30 is NOT in the ds4 loader enum — the glm52
        loader must add it (GLM52_ONBOARDING.md section 4.3).
  * f16 / f32 / i32: as in the ds4 converter.

FAIL-CLOSED: every GLM-5.2 source tensor name and several structural facts are
UNRESOLVED until the real model is downloaded (config.json +
model.safetensors.index.json). Those are `None` sentinels below, each marked
TODO(verify-from-index.json). A real conversion REFUSES to run and prints the
unresolved list instead. Nothing is invented.

Usage:
  glm52_to_gguf_full.py <model_dir> <out.gguf> [--layers N]   # refuses while UNRESOLVED
  glm52_to_gguf_full.py --probe <model_dir>    # header-only survey to RESOLVE the tables
  glm52_to_gguf_full.py --self-test [<scratch_dir>]

--probe mirrors tools/axiom_nvfp4_validate.py: parses only shard headers,
prints the dtype histogram + representative tensor names per layer so the
ROLES src templates can be filled from evidence.

--self-test builds a tiny synthetic GLM-shaped model (with clearly-synthetic
stand-ins for the UNRESOLVED facts) and proves the machinery end-to-end:
expert stacking [in,out,NEXP]+gscale, FP8+scale, BF16 passthrough, GGUF
write, and full byte-equal reparse — exactly the ds4 converter's proof shape.
Safe on this Windows host (never touches a real model).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvfp4_to_gguf_full as ds4conv  # the PROVEN base — do not modify that file
from nvfp4_to_gguf_full import (
    SafetensorsModel,
    PlanEntry,
    elems,
    gguf_dims,
    to_f16_bytes,
    to_f32_bytes,
    to_i32_bytes,
    read_gguf_index,
    read_gguf_bytes,
    _gs,
    _compute_offsets,
    _pack_shard,
    _bf16,
)

# ---- GGML type codes ---------------------------------------------------------
T_F32 = 0
T_F16 = 1
T_I32 = 26
T_NVFP4 = 17  # == DS4_GGML_TYPE_NVFP4 (src/axiom_ds4_weights.cpp:26)
T_FP8 = 18    # == DS4_GGML_TYPE_FP8_E4M3 (src/axiom_ds4_weights.cpp:27)
T_BF16 = 30   # standard GGML BF16 id — NOT yet in the ds4/glm52 loader enum (new code)

ALIGN = 32

UNKNOWN = None  # TODO(verify-from-index.json) sentinel — mirrors AXIOM_GLM52_UNKNOWN_*

# ==============================================================================
# Family descriptor facts (python mirror of src/axiom_glm52_family.cpp —
# keep the two in sync; the C side is authoritative).
# VERIFIED [HF] = zai-org/GLM-5.2 + nvidia/GLM-5.2-NVFP4 model cards, July 2026.
# ==============================================================================
DESC = {
    "family": "glm52",
    "gguf_architecture": "glm52",     # [design] written to general.architecture
    "hidden": 6144,                   # VERIFIED [HF]
    "layers": 78,                     # VERIFIED [HF]
    "vocab": 154880,                  # VERIFIED [HF]
    "max_context": 1048576,           # VERIFIED [HF] 1M
    "routed_experts": 256,            # VERIFIED [HF]
    "shared_experts": 1,              # VERIFIED [HF]
    "topk": 8,                        # VERIFIED [HF]
    "indexer_share_period": 4,        # VERIFIED [HF] IndexShare
    # ---- UNRESOLVED (numbered as in GLM52_ONBOARDING.md section 6) ----
    "expert_hidden": UNKNOWN,         # U-7 moe_intermediate_size
    "dense_layers": UNKNOWN,          # U-7 first_k_dense_replace
    "indexer_anchor_phase": UNKNOWN,  # U-5 which layer inside each 4-group owns the indexer
    "indexer_first_layer": UNKNOWN,   # U-4 first sparse layer
    "shared_expert_dtype": UNKNOWN,   # U-6 "bf16" | "fp8" (unquantized per [HF], flavor TBD)
    "attn_dtype": UNKNOWN,            # U-6 expected bf16 (only MoE experts NVFP4 per [HF])
    "embed_dtype": UNKNOWN,           # U-6
    "head_dtype": UNKNOWN,            # U-6
}

# ==============================================================================
# ROLES: glm -> gguf name-template mapping (python mirror of GLM52_ROLES in
# src/axiom_glm52_family.cpp). Target GGUF names are [design] and FINAL (they
# keep ds4 "blk.N.<suffix>" conventions so glm52 loader code can reuse the ds4
# collect/upload machinery). EVERY src is UNKNOWN =
# TODO(verify-from-index.json); `guess` is a NON-OPERATIVE hint for whoever
# runs --probe (never used by the code).
#
# Presence rule per layer L (== axiom_fam_layer_role_present):
#   L >= min_layer  AND  (max_layer_excl==0 or L < max_layer_excl)
#   AND (mod==0 or L % mod == phase); any UNKNOWN field -> not present.
#
# "cls" producer classes: f32 f16 bf16 i32 fp8 nvfp4_experts
# ==============================================================================
ROLES = [
    # --- norms / small params (every layer) ---
    dict(gguf="attn_norm.weight", cls="f32", src=UNKNOWN,
         guess="model.layers.{L}.input_layernorm.weight"),
    dict(gguf="ffn_norm.weight", cls="f32", src=UNKNOWN,
         guess="model.layers.{L}.post_attention_layernorm.weight"),
    # --- MLA attention projections — PLACEHOLDER SET (U-3): DS-V3.2-canonical
    #     q_a/q_b/kv_a/kv_b/output split; ds4-flash instead has single wkv +
    #     factored wo_a/wo_b. The role LIST must be reconciled with index.json.
    dict(gguf="attn_q_a.weight", cls="bf16", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.q_a_proj.weight"),
    dict(gguf="attn_q_a_norm.weight", cls="f32", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.q_a_layernorm.weight"),
    dict(gguf="attn_q_b.weight", cls="bf16", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.q_b_proj.weight"),
    dict(gguf="attn_kv_a.weight", cls="bf16", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight"),
    dict(gguf="attn_kv_a_norm.weight", cls="f32", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.kv_a_layernorm.weight"),
    dict(gguf="attn_kv_b.weight", cls="bf16", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.kv_b_proj.weight"),
    dict(gguf="attn_output.weight", cls="bf16", src=UNKNOWN,
         guess="model.layers.{L}.self_attn.o_proj.weight"),
    # --- DSA lightning indexer: OWNER layers only. mod=4 is the VERIFIED
    #     IndexShare period [HF]; phase + first layer UNKNOWN (U-4/U-5).
    #     Component set (wq_b/wk/k_norm/weights_proj) is a PLACEHOLDER modeled
    #     on the DeepSeek-V3.2 DSA indexer module (U-5).
    dict(gguf="indexer.wq_b.weight", cls="bf16", src=UNKNOWN,
         min_layer=DESC["indexer_first_layer"],
         mod=(DESC["indexer_share_period"], DESC["indexer_anchor_phase"]),
         guess="model.layers.{L}.self_attn.indexer.wq_b.weight"),
    dict(gguf="indexer.wk.weight", cls="bf16", src=UNKNOWN,
         min_layer=DESC["indexer_first_layer"],
         mod=(DESC["indexer_share_period"], DESC["indexer_anchor_phase"]),
         guess="model.layers.{L}.self_attn.indexer.wk.weight"),
    dict(gguf="indexer.k_norm.weight", cls="f32", src=UNKNOWN,
         min_layer=DESC["indexer_first_layer"],
         mod=(DESC["indexer_share_period"], DESC["indexer_anchor_phase"]),
         guess="model.layers.{L}.self_attn.indexer.k_norm.weight"),
    dict(gguf="indexer.weights_proj.weight", cls="bf16", src=UNKNOWN,
         min_layer=DESC["indexer_first_layer"],
         mod=(DESC["indexer_share_period"], DESC["indexer_anchor_phase"]),
         guess="model.layers.{L}.self_attn.indexer.weights_proj.weight"),
    # --- dense FFN (leading dense layers only; cap == dense_layers, U-7) ---
    dict(gguf="ffn_gate.weight", cls="bf16", src=UNKNOWN,
         max_layer_excl=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.gate_proj.weight"),
    dict(gguf="ffn_up.weight", cls="bf16", src=UNKNOWN,
         max_layer_excl=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.up_proj.weight"),
    dict(gguf="ffn_down.weight", cls="bf16", src=UNKNOWN,
         max_layer_excl=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.down_proj.weight"),
    # --- MoE (layers >= dense_layers, U-7) ---
    dict(gguf="ffn_gate_inp.weight", cls="f16", src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.gate.weight"),
    dict(gguf="exp_probs_b.bias", cls="f32", src=UNKNOWN, optional=True,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.gate.e_score_correction_bias"),
    # routed experts: stacked [in,out,NEXP] NVFP4 + .gscale — VERIFIED class [HF].
    # src is the PER-EXPERT base template ({L}=layer, {E}=expert); the producer
    # reads base+".weight"(U8) + ".weight_scale"(F8_E4M3) + ".weight_scale_2"(F32),
    # identical to the proven ds4 flow.
    dict(gguf="ffn_gate_exps.weight", cls="nvfp4_experts", src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.experts.{E}.gate_proj"),
    dict(gguf="ffn_up_exps.weight", cls="nvfp4_experts", src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.experts.{E}.up_proj"),
    dict(gguf="ffn_down_exps.weight", cls="nvfp4_experts", src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.experts.{E}.down_proj"),
    # shared expert — UNQUANTIZED per [HF]; bf16 vs fp8 is U-6, so even the
    # producer CLASS is unresolved (cls=UNKNOWN -> counted unresolved).
    dict(gguf="ffn_gate_shexp.weight", cls=UNKNOWN, src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.shared_experts.gate_proj.weight"),
    dict(gguf="ffn_up_shexp.weight", cls=UNKNOWN, src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.shared_experts.up_proj.weight"),
    dict(gguf="ffn_down_shexp.weight", cls=UNKNOWN, src=UNKNOWN,
         min_layer=DESC["dense_layers"],
         guess="model.layers.{L}.mlp.shared_experts.down_proj.weight"),
]

# Common/tail (family-invariant target names, ds4 convention). MTP tensors are
# NOT emitted yet: the GLM-5.2 MTP inventory is U-11.
COMMON_ROLES = [
    dict(gguf="token_embd.weight", cls=UNKNOWN, src=UNKNOWN,  # dtype U-6
         guess="model.embed_tokens.weight"),
]
TAIL_ROLES = [
    dict(gguf="output_norm.weight", cls="f32", src=UNKNOWN, guess="model.norm.weight"),
    dict(gguf="output.weight", cls=UNKNOWN, src=UNKNOWN,      # dtype U-6
         guess="lm_head.weight"),
]


def role_present(role, layer):
    """Mirror of axiom_fam_layer_role_present (fail-closed on UNKNOWN)."""
    mn = role.get("min_layer", 0)
    mxe = role.get("max_layer_excl", 0)
    mod, phase = role.get("mod", (0, 0))
    if mn is UNKNOWN or mxe is UNKNOWN or mod is UNKNOWN:
        return False
    if layer < mn:
        return False
    if mxe and layer >= mxe:
        return False
    if mod:
        if phase is UNKNOWN:
            return False
        if (layer % mod) != phase:
            return False
    return True


# ==============================================================================
# Producers. nvfp4/gscale/fp8/f16/f32/i32 reuse the PROVEN ds4 implementations;
# bf16 is the one genuinely new class (raw byte passthrough, lossless).
# ==============================================================================
def to_bf16_bytes(raw, dt):
    if dt == "BF16":
        return raw  # byte-lossless passthrough
    if dt == "F32":
        # exact truncation is NOT performed: refuse silent precision decisions.
        raise ValueError("bf16 class expects BF16 source, got %s (decide explicitly)" % dt)
    raise ValueError("cannot pass dtype %s through as BF16" % dt)


def produce(model, entry):
    kind = entry.producer[0]
    if kind == "nvfp4":
        _, bases, out, groups = entry.producer
        return b"".join(
            ds4conv._nvfp4_expert_blocks(model, b, out, groups) for b in bases)
    if kind == "gscale":
        _, bases = entry.producer
        return np.array(
            [ds4conv._expert_global(model, b) for b in bases], np.float32).tobytes()
    src = entry.producer[1]
    raw, dt, sh = model.read(src)
    if kind in ("fp8w", "fp8s"):
        return raw
    if kind == "bf16":
        return to_bf16_bytes(raw, dt)
    if kind == "f16":
        return to_f16_bytes(raw, dt)
    if kind == "f32":
        return to_f32_bytes(raw, dt)
    if kind == "i32":
        return to_i32_bytes(raw, dt)
    raise ValueError(kind)


# ==============================================================================
# Plan building — role-table-driven (the glm52 analog of ds4 build_plan, but
# the per-tensor list is DATA, not code).
# ==============================================================================
def _fmt(template, layer=None, expert=None):
    s = template
    if layer is not None:
        s = s.replace("{L}", str(layer))
    if expert is not None:
        s = s.replace("{E}", str(expert))
    return s


def _plan_simple(plan, model, target, src, cls, unresolved, optional=False):
    if src is UNKNOWN or cls is UNKNOWN:
        unresolved.append(target)
        return
    if not model.has(src):
        if optional:
            return
        unresolved.append("%s (source '%s' absent)" % (target, src))
        return
    dt, sh = model.info(src)
    if cls == "bf16":
        plan.append(PlanEntry(target, T_BF16, gguf_dims(sh), elems(sh) * 2, ("bf16", src)))
    elif cls == "f16":
        plan.append(PlanEntry(target, T_F16, gguf_dims(sh), elems(sh) * 2, ("f16", src)))
    elif cls == "f32":
        plan.append(PlanEntry(target, T_F32, gguf_dims(sh), elems(sh) * 4, ("f32", src)))
    elif cls == "i32":
        plan.append(PlanEntry(target, T_I32, gguf_dims(sh), elems(sh) * 4, ("i32", src)))
    elif cls == "fp8":
        plan.append(PlanEntry(target, T_FP8, gguf_dims(sh), elems(sh), ("fp8w", src)))
        base = src[:-len(".weight")] if src.endswith(".weight") else src
        scale = base + ".scale"
        if model.has(scale):
            _, ssh = model.info(scale)
            tgt = (target[:-len(".weight")] + ".scale"
                   if target.endswith(".weight") else target + ".scale")
            plan.append(PlanEntry(tgt, T_FP8, gguf_dims(ssh), elems(ssh), ("fp8s", scale)))
    else:
        raise ValueError(cls)


def _plan_nvfp4_experts(plan, model, target, base_template, layer, nexp, unresolved):
    """Stacked routed experts [in, out, NEXP] + F32 .gscale — the ds4 contract."""
    if base_template is UNKNOWN:
        unresolved.append(target)
        return
    bases = [_fmt(base_template, layer=layer, expert=e) for e in range(nexp)]
    if nexp == 0 or not model.has(bases[0] + ".weight"):
        unresolved.append("%s (expert-0 source '%s.weight' absent)" % (target, bases[0]))
        return
    out, in_cols, groups = ds4conv._nvfp4_geom(model, bases[0])
    per_expert = out * groups * 9
    plan.append(PlanEntry(target, T_NVFP4, [in_cols, out, nexp],
                          nexp * per_expert, ("nvfp4", bases, out, groups)))
    gname = (target[:-len(".weight")] + ".gscale"
             if target.endswith(".weight") else target + ".gscale")
    plan.append(PlanEntry(gname, T_F32, [nexp], nexp * 4, ("gscale", bases)))


def build_plan(model, desc, roles, common_roles, tail_roles, n_layers):
    plan = []
    unresolved = []
    for role in common_roles:
        _plan_simple(plan, model, role["gguf"], role["src"], role["cls"], unresolved)
    for L in range(n_layers):
        for role in roles:
            if not role_present(role, L):
                # distinguish "predicate says absent" from "predicate UNRESOLVED"
                mn = role.get("min_layer", 0)
                mxe = role.get("max_layer_excl", 0)
                mod, phase = role.get("mod", (0, 0))
                if mn is UNKNOWN or mxe is UNKNOWN or mod is UNKNOWN or \
                   (mod not in (0, UNKNOWN) and phase is UNKNOWN):
                    if L == 0:  # report once, not per-layer
                        unresolved.append("blk.*.%s (presence predicate UNRESOLVED)"
                                          % role["gguf"])
                continue
            target = "blk.%d.%s" % (L, role["gguf"])
            if role["cls"] == "nvfp4_experts":
                _plan_nvfp4_experts(plan, model, target, role["src"], L,
                                    desc["routed_experts"], unresolved)
            else:
                src = (_fmt(role["src"], layer=L)
                       if role["src"] is not UNKNOWN else UNKNOWN)
                _plan_simple(plan, model, target, src, role["cls"], unresolved,
                             optional=role.get("optional", False))
    for role in tail_roles:
        _plan_simple(plan, model, role["gguf"], role["src"], role["cls"], unresolved)
    return plan, unresolved


# ==============================================================================
# GGUF writer + verification (adapted from the proven ds4 versions; local
# copies only because they must call THIS module's produce(), which adds bf16).
# ==============================================================================
def write_gguf(model, plan, out_path, architecture):
    import struct
    offs = _compute_offsets(plan)
    with open(out_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(plan)))
        f.write(struct.pack("<Q", 1))
        f.write(_gs("general.architecture") + struct.pack("<I", 8) + _gs(architecture))
        for e, off in zip(plan, offs):
            f.write(_gs(e.name))
            f.write(struct.pack("<I", len(e.dims)))
            for d in e.dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", e.type))
            f.write(struct.pack("<Q", off))
        pos = f.tell()
        f.write(b"\x00" * ((-pos) % ALIGN))
        dstart = f.tell()
        for e, off in zip(plan, offs):
            pad = (dstart + off) - f.tell()
            if pad:
                f.write(b"\x00" * pad)
            data = produce(model, e)
            if len(data) != e.nbytes:
                raise RuntimeError("%s: produced %d bytes, planned %d"
                                   % (e.name, len(data), e.nbytes))
            f.write(data)
    return dstart


def full_verify(model, plan, out_path):
    """Byte-verify EVERY tensor against independent recomputation."""
    ver, dstart, tensors = read_gguf_index(out_path)
    assert ver == 3, "bad version"
    assert len(tensors) == len(plan), "tensor count mismatch"
    for e in plan:
        typ, dims, off = tensors[e.name]
        assert typ == e.type, "%s type %d != %d" % (e.name, typ, e.type)
        assert dims == e.dims, "%s dims %s != %s" % (e.name, dims, e.dims)
        got = read_gguf_bytes(out_path, dstart, off, e.nbytes)
        assert got == produce(model, e), "%s round-trip BYTE MISMATCH" % e.name
    return len(plan)


# ==============================================================================
# Real conversion (fail-closed) + probe
# ==============================================================================
def convert(model_dir, out_path, n_layers, desc=None, roles=None,
            common_roles=None, tail_roles=None):
    desc = desc or DESC
    roles = roles if roles is not None else ROLES
    common_roles = common_roles if common_roles is not None else COMMON_ROLES
    tail_roles = tail_roles if tail_roles is not None else TAIL_ROLES

    model = SafetensorsModel(model_dir)
    plan, unresolved = build_plan(model, desc, roles, common_roles, tail_roles, n_layers)
    if unresolved:
        print("REFUSING to convert: %d UNRESOLVED target(s) — the glm52 name-template"
              " table still contains TODO(verify-from-index.json) entries." % len(unresolved),
              file=sys.stderr)
        seen = set()
        for u in unresolved:
            if u in seen:
                continue
            seen.add(u)
            print("  UNRESOLVED:", u, file=sys.stderr)
        print("Run `%s --probe %s` and fill DESC/ROLES from the real headers "
              "(see docs/glm52/GLM52_ONBOARDING.md section 6)."
              % (os.path.basename(sys.argv[0]), model_dir), file=sys.stderr)
        raise SystemExit(2)

    from collections import Counter, defaultdict
    tc = Counter(e.type for e in plan)
    tb = defaultdict(int)
    for e in plan:
        tb[e.type] += e.nbytes
    total = sum(e.nbytes for e in plan)
    names = {T_F32: "F32", T_F16: "F16", T_I32: "I32",
             T_NVFP4: "NVFP4", T_FP8: "FP8", T_BF16: "BF16"}
    print("layers=%d  tensors=%d  data_bytes=%d (%.2f GiB)"
          % (n_layers, len(plan), total, total / (1 << 30)))
    for t in sorted(tc):
        print("  type %-6s: %6d tensors  %14d bytes" % (names.get(t, t), tc[t], tb[t]))
    dstart = write_gguf(model, plan, out_path, desc["gguf_architecture"])
    print("wrote %s  (data_start=%d)" % (out_path, dstart))
    n = full_verify(model, plan, out_path)
    print("full verify: all %d tensors byte-equal on reparse." % n)


def probe(model_dir):
    """Header-only survey (mirrors tools/axiom_nvfp4_validate.py) to resolve
    the TODO(verify-from-index.json) entries with evidence."""
    import collections
    import glob as globmod
    import json
    import re
    import struct

    files = sorted(globmod.glob(os.path.join(model_dir, "*.safetensors")))
    dt = collections.Counter()
    layers = set()
    layer0 = []
    nonlayer = []
    total = 0
    for p in files:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            dt[v["dtype"]] += 1
            total += 1
            m = re.search(r"layers\.(\d+)\.", k)
            if m:
                layers.add(int(m.group(1)))
                if int(m.group(1)) == 0:
                    layer0.append((k, v["dtype"], v["shape"]))
            else:
                nonlayer.append((k, v["dtype"], v["shape"]))
    print("shards=%d tensors=%d" % (len(files), total))
    print("dtypes:", dict(dt))
    if layers:
        print("layers seen: %d  range=%d..%d  (descriptor says %d)"
              % (len(layers), min(layers), max(layers), DESC["layers"]))
    print("--- non-layer tensors (embed/head/norm/mtp?) ---")
    for k, d, s in sorted(nonlayer):
        print("  %-70s %-8s %s" % (k, d, s))
    print("--- layer-0 tensors (fill ROLES src templates from these) ---")
    for k, d, s in sorted(layer0):
        print("  %-70s %-8s %s" % (k, d, s))


# ==============================================================================
# Self-test: synthetic GLM-shaped model. The UNRESOLVED facts get
# clearly-synthetic stand-ins (SYN_*) so the machinery — stacked NVFP4 experts,
# FP8+scale, BF16 passthrough, role predicates, GGUF write, byte-equal reparse
# — is proven end-to-end without inventing anything about the real model.
# ==============================================================================
def self_test(scratch):
    rng = np.random.default_rng(0)
    md = os.path.join(scratch, "glm52_synthetic_model")
    os.makedirs(md, exist_ok=True)

    NL = 6        # exercises: dense layer 0, MoE layers 1.., indexer anchors
    NEXP = 4
    HID = 32      # expert `in` (must be %16); tiny on purpose
    INTER = 16
    SYN = {       # synthetic stand-ins for the UNRESOLVED descriptor facts
        "dense_layers": 1,
        "indexer_first_layer": 1,
        "indexer_anchor_phase": 1,   # owners at layers 1 and 5 (period 4)
        "shared_expert_cls": "fp8",  # exercise the fp8 branch for shexp
        "embed_cls": "bf16",
        "head_cls": "bf16",
    }

    weight_map = {}
    shard_tensors = {}

    def put(name, dt, arr, shard="model-0001.safetensors"):
        weight_map[name] = shard
        shard_tensors.setdefault(shard, []).append((name, dt, arr))

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    def bf16(*shape):
        return _bf16(rng.standard_normal(shape).astype(np.float32))

    def u8(*shape):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)

    put("syn.embed.weight", "BF16", bf16(48, HID))
    put("syn.norm.weight", "F32", f32(HID))
    put("syn.head.weight", "BF16", bf16(48, HID))
    for L in range(NL):
        put("syn.layers.%d.attn_norm.weight" % L, "F32", f32(HID))
        put("syn.layers.%d.ffn_norm.weight" % L, "F32", f32(HID))
        for proj, oc in (("q_a", 16), ("q_b", 16), ("kv_a", 16), ("kv_b", 16), ("o", HID)):
            put("syn.layers.%d.attn.%s.weight" % (L, proj), "BF16", bf16(oc, HID))
        put("syn.layers.%d.attn.q_a_norm.weight" % L, "F32", f32(16))
        put("syn.layers.%d.attn.kv_a_norm.weight" % L, "F32", f32(16))
        is_owner = (L >= SYN["indexer_first_layer"] and
                    L % DESC["indexer_share_period"] == SYN["indexer_anchor_phase"])
        if is_owner:
            put("syn.layers.%d.indexer.wq_b.weight" % L, "BF16", bf16(16, HID))
            put("syn.layers.%d.indexer.wk.weight" % L, "BF16", bf16(16, HID))
            put("syn.layers.%d.indexer.k_norm.weight" % L, "F32", f32(16))
            put("syn.layers.%d.indexer.weights_proj.weight" % L, "BF16", bf16(8, HID))
        if L < SYN["dense_layers"]:
            put("syn.layers.%d.dense.gate.weight" % L, "BF16", bf16(INTER, HID))
            put("syn.layers.%d.dense.up.weight" % L, "BF16", bf16(INTER, HID))
            put("syn.layers.%d.dense.down.weight" % L, "BF16", bf16(HID, INTER))
        else:
            put("syn.layers.%d.gate_inp.weight" % L, "BF16", bf16(NEXP, HID))
            put("syn.layers.%d.gate_inp.bias" % L, "F32", f32(NEXP))
            for e in range(NEXP):
                for w, oc, ic in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
                    b = "syn.layers.%d.experts.%d.%s" % (L, e, w)
                    put(b + ".weight", "U8", u8(oc, ic // 2))
                    put(b + ".weight_scale", "F8_E4M3", u8(oc, ic // 16))
                    put(b + ".weight_scale_2", "F32",
                        np.array([rng.standard_normal()], np.float32))
            for w, oc, ic in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
                b = "syn.layers.%d.shexp.%s" % (L, w)
                put(b + ".weight", "F8_E4M3", u8(oc, ic))
                put(b + ".scale", "F8_E8M0", u8(max(1, oc // 16), max(1, ic // 16)))

    import json
    for shard, tl in shard_tensors.items():
        _pack_shard(os.path.join(md, shard), tl)
    with open(os.path.join(md, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)

    # RESOLVED role tables for the synthetic model — same shapes as the real
    # (still-TODO) tables; only src templates + the SYN facts differ.
    syn_desc = dict(DESC, routed_experts=NEXP, layers=NL,
                    dense_layers=SYN["dense_layers"],
                    indexer_first_layer=SYN["indexer_first_layer"],
                    indexer_anchor_phase=SYN["indexer_anchor_phase"])
    syn_roles = [
        dict(gguf="attn_norm.weight", cls="f32", src="syn.layers.{L}.attn_norm.weight"),
        dict(gguf="ffn_norm.weight", cls="f32", src="syn.layers.{L}.ffn_norm.weight"),
        dict(gguf="attn_q_a.weight", cls="bf16", src="syn.layers.{L}.attn.q_a.weight"),
        dict(gguf="attn_q_a_norm.weight", cls="f32", src="syn.layers.{L}.attn.q_a_norm.weight"),
        dict(gguf="attn_q_b.weight", cls="bf16", src="syn.layers.{L}.attn.q_b.weight"),
        dict(gguf="attn_kv_a.weight", cls="bf16", src="syn.layers.{L}.attn.kv_a.weight"),
        dict(gguf="attn_kv_a_norm.weight", cls="f32", src="syn.layers.{L}.attn.kv_a_norm.weight"),
        dict(gguf="attn_kv_b.weight", cls="bf16", src="syn.layers.{L}.attn.kv_b.weight"),
        dict(gguf="attn_output.weight", cls="bf16", src="syn.layers.{L}.attn.o.weight"),
        dict(gguf="indexer.wq_b.weight", cls="bf16", src="syn.layers.{L}.indexer.wq_b.weight",
             min_layer=SYN["indexer_first_layer"],
             mod=(DESC["indexer_share_period"], SYN["indexer_anchor_phase"])),
        dict(gguf="indexer.wk.weight", cls="bf16", src="syn.layers.{L}.indexer.wk.weight",
             min_layer=SYN["indexer_first_layer"],
             mod=(DESC["indexer_share_period"], SYN["indexer_anchor_phase"])),
        dict(gguf="indexer.k_norm.weight", cls="f32", src="syn.layers.{L}.indexer.k_norm.weight",
             min_layer=SYN["indexer_first_layer"],
             mod=(DESC["indexer_share_period"], SYN["indexer_anchor_phase"])),
        dict(gguf="indexer.weights_proj.weight", cls="bf16",
             src="syn.layers.{L}.indexer.weights_proj.weight",
             min_layer=SYN["indexer_first_layer"],
             mod=(DESC["indexer_share_period"], SYN["indexer_anchor_phase"])),
        dict(gguf="ffn_gate.weight", cls="bf16", src="syn.layers.{L}.dense.gate.weight",
             max_layer_excl=SYN["dense_layers"]),
        dict(gguf="ffn_up.weight", cls="bf16", src="syn.layers.{L}.dense.up.weight",
             max_layer_excl=SYN["dense_layers"]),
        dict(gguf="ffn_down.weight", cls="bf16", src="syn.layers.{L}.dense.down.weight",
             max_layer_excl=SYN["dense_layers"]),
        dict(gguf="ffn_gate_inp.weight", cls="f16", src="syn.layers.{L}.gate_inp.weight",
             min_layer=SYN["dense_layers"]),
        dict(gguf="exp_probs_b.bias", cls="f32", src="syn.layers.{L}.gate_inp.bias",
             optional=True, min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_gate_exps.weight", cls="nvfp4_experts",
             src="syn.layers.{L}.experts.{E}.w1", min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_up_exps.weight", cls="nvfp4_experts",
             src="syn.layers.{L}.experts.{E}.w3", min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_down_exps.weight", cls="nvfp4_experts",
             src="syn.layers.{L}.experts.{E}.w2", min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_gate_shexp.weight", cls=SYN["shared_expert_cls"],
             src="syn.layers.{L}.shexp.w1.weight", min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_up_shexp.weight", cls=SYN["shared_expert_cls"],
             src="syn.layers.{L}.shexp.w3.weight", min_layer=SYN["dense_layers"]),
        dict(gguf="ffn_down_shexp.weight", cls=SYN["shared_expert_cls"],
             src="syn.layers.{L}.shexp.w2.weight", min_layer=SYN["dense_layers"]),
    ]
    syn_common = [dict(gguf="token_embd.weight", cls=SYN["embed_cls"], src="syn.embed.weight")]
    syn_tail = [dict(gguf="output_norm.weight", cls="f32", src="syn.norm.weight"),
                dict(gguf="output.weight", cls=SYN["head_cls"], src="syn.head.weight")]

    out = os.path.join(scratch, "glm52_synthetic.gguf")
    print("=== glm52 self-test: synthetic model (%d layers, %d experts) ===" % (NL, NEXP))
    convert(md, out, NL, desc=syn_desc, roles=syn_roles,
            common_roles=syn_common, tail_roles=syn_tail)

    # Structural spot-checks beyond byte-equality:
    ver, dstart, tensors = read_gguf_index(out)
    typ, dims, _ = tensors["blk.1.ffn_gate_exps.weight"]
    assert typ == T_NVFP4 and dims == [HID, INTER, NEXP], \
        "stacked expert dims wrong: %s" % dims
    typ, dims, _ = tensors["blk.1.ffn_gate_exps.gscale"]
    assert typ == T_F32 and dims == [NEXP]
    assert tensors["token_embd.weight"][0] == T_BF16, "bf16 class not exercised"
    assert tensors["blk.1.ffn_gate_shexp.weight"][0] == T_FP8
    assert "blk.1.ffn_gate_shexp.scale" in tensors, "fp8 scale companion missing"
    assert "blk.0.ffn_gate_exps.weight" not in tensors, "dense layer 0 must have no experts"
    assert "blk.0.ffn_gate.weight" in tensors, "dense layer 0 FFN missing"
    assert "blk.1.indexer.wq_b.weight" in tensors and \
           "blk.5.indexer.wq_b.weight" in tensors, "indexer owners (1,5) missing"
    assert "blk.2.indexer.wq_b.weight" not in tensors and \
           "blk.4.indexer.wq_b.weight" not in tensors, \
        "IndexShare predicate leaked indexer weights onto non-owner layers"
    print("structural checks OK: stacking [in,out,%d]+gscale, BF16/FP8 classes,"
          " dense/MoE split, IndexShare owner predicate. SELF-TEST PASS." % NEXP)

    # And the real (production) tables must still be FAIL-CLOSED:
    model = SafetensorsModel(md)
    _, unresolved = build_plan(model, DESC, ROLES, COMMON_ROLES, TAIL_ROLES, 2)
    assert unresolved, "production ROLES unexpectedly resolved — update the C descriptor too"
    print("fail-closed check OK: production tables still report %d unresolved target(s)."
          % len(unresolved))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", type=int, default=DESC["layers"])
    ap.add_argument("--probe", metavar="MODEL_DIR")
    ap.add_argument("--self-test", nargs="?", const=".", metavar="SCRATCH_DIR")
    args = ap.parse_args()

    if args.self_test is not None:
        self_test(args.self_test)
        return
    if args.probe:
        probe(args.probe)
        return
    if not args.model_dir or not args.out:
        ap.error("model_dir and out are required (or use --probe / --self-test)")
    n = max(1, min(args.layers, DESC["layers"]))
    convert(args.model_dir, args.out, n)


if __name__ == "__main__":
    main()
