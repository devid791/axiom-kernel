#!/usr/bin/env python3
"""Full-model NVFP4/FP8 safetensors -> ds4-parseable GGUF converter.

Scales the proven spike (tools/nvfp4_to_gguf.py) to the WHOLE
`nvidia/DeepSeek-V4-Flash-NVFP4` model (46 shards indexed by
model.safetensors.index.json) into a single GGUF that the Axiom ds4 loader
(`src/axiom_ds4_weights.cpp`, `axiom_ds4_resident_weights_load`) can read.

Encoding families (see docs/deepseek_v4_flash_cluster/converter_mapping.md):
  * routed experts (ffn.experts.E.w1/w3/w2)  -> NVFP4  (ds4 type 17):
        combined 3D tensor [in, out, NEXP] of 9B/16 blocks (8 E2M1-nibble
        bytes + 1 F8_E4M3 group scale) + an F32 per-expert `.gscale` companion.
        Pure byte-repack of source .weight(U8) + .weight_scale(F8_E4M3);
        globals from .weight_scale_2(F32). LOSSLESS.
  * attn/shared/indexer.wq_b weights (source .weight F8_E4M3 + .scale F8_E8M0)
        -> FP8 (ds4 type 18): raw E4M3 weight bytes + an E8M0 `.scale`
        companion (1 byte/elem). LOSSLESS.  NOTE: the current ds4 resident
        loader reads these into `*_q8` fields and the forward runs a Q8_0
        matvec on them -> a Q8_0-vs-FP8 dtype mismatch that needs forward
        wiring. Flagged in the mapping doc.  (The `.scale` companions are not
        yet read by the loader; they are written so the file is forward-ready.)
  * everything named `_f16` w/ no source scale (compressor/indexer projections,
        router gate, embed, hc_*_fn, ape) -> F16 (ds4 type 1). BF16/F32 source
        is converted to F16 (standard, effectively lossless).
  * everything named `_f32` (norms, sinks, hc_*_base/scale, router bias)
        -> F32 (ds4 type 0). BF16 source upcast to F32 (lossless).
  * router tid2eid (layers 0..2) -> I32 (ds4 type 26).

MTP support block (`--mtp-out`): the source also carries the `mtp.0.*`
multi-token-prediction module (the 35-40 tok/s speculative-decode lever, see
docs/axiom_mtp_speculative_decode_recipe_20260601.md). It is emitted as a
SEPARATE GGUF (matching the ds4 MTP loader's separate-file architecture,
`axiom_ds4_mtp_resident_weights_load`), LOSSLESSLY in native dtypes:
  * the 32 loader-required names (src/axiom_ds4_weights.cpp:48-81) mapped from
    the mtp.0.* source names; FP8 attn/shared/e_proj/h_proj as type 18 pairs,
    norms/hc/router as F32, routed experts in whichever native family the
    shard headers really carry (probed: FP8 pair OR NVFP4 triplet — the tool
    REFUSES on anything else rather than guess).
  * every mtp.* source tensor must be consumed (or be a recognized
    `.input_scale` activation hint) — any unknown name is a HARD ERROR. Silent
    drops of the MTP block are the bug this flag fixes; a main conversion now
    also refuses when mtp.* tensors exist and neither --mtp-out nor
    --skip-mtp was given.
NOTE: the current MTP loader expects Q8_0/Q4_K types and exactly 32 tensors;
loading this native MTP GGUF requires the loader native branches described in
docs/deepseek_v4_flash_cluster/mtp_native_plan.md.

Streams shards (never loads all ~157G at once): pass 1 reads only shard
*headers* to build the tensor table + offsets; pass 2 seeks each source
tensor's bytes on demand and writes them out.  Peak RAM ~ one expert block
slice (a few MB).

Usage:
  nvfp4_to_gguf_full.py <model_dir> <out.gguf> [--layers N] [--mtp-out <mtp.gguf>]
  nvfp4_to_gguf_full.py <model_dir> --mtp-out <mtp.gguf>      # MTP block only
  nvfp4_to_gguf_full.py --self-test [<scratch_dir>]

`--layers N` limits emitted layers to 0..N-1 (converter testing without the
full model). A production run for the ds4 resident loader needs N=43 (all
layers) + common + tail; anything less is only loadable via the span loader.
`--layers` does not affect --mtp-out (the MTP module is a single extra layer).

Do NOT run on the real 157G model on this Windows host (model lives only on the
remote GX10 node). Use --self-test here to validate logic + round-trip.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# ---- ds4 GGML type codes (must match src/axiom_ds4_weights.cpp enum) --------
T_F32 = 0
T_F16 = 1
T_I32 = 26
T_NVFP4 = 17          # DS4_GGML_TYPE_NVFP4  (9B/16 block)
T_FP8 = 18            # DS4_GGML_TYPE_FP8_E4M3 (1 byte/elem; weight or scale)

ALIGN = 32
DS4_LAYER_COUNT = 43

# =============================================================================
# safetensors streaming reader (headers cached; data read on demand)
# =============================================================================
class SafetensorsModel:
    def __init__(self, model_dir):
        self.dir = model_dir
        idx_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(idx_path, "r") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._headers = {}   # shard filename -> parsed header dict

    def _header(self, shard):
        if shard not in self._headers:
            p = os.path.join(self.dir, shard)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            h["__base__"] = 8 + n
            self._headers[shard] = h
        return self._headers[shard]

    def has(self, name):
        return name in self.weight_map

    def info(self, name):
        """Return (dtype_str, shape) reading only the shard header (cheap)."""
        shard = self.weight_map[name]
        e = self._header(shard)[name]
        return e["dtype"], list(e["shape"])

    def read(self, name):
        """Return (raw_bytes, dtype_str, shape) — reads the tensor payload."""
        shard = self.weight_map[name]
        h = self._header(shard)
        e = h[name]
        a, b = e["data_offsets"]
        p = os.path.join(self.dir, shard)
        with open(p, "rb") as f:
            f.seek(h["__base__"] + a)
            raw = f.read(b - a)
        return raw, e["dtype"], list(e["shape"])


# =============================================================================
# dtype conversions
# =============================================================================
def _decode_f32(raw, dt):
    if dt == "F32":
        return np.frombuffer(raw, "<f4")
    if dt == "F16":
        return np.frombuffer(raw, "<f2").astype(np.float32)
    if dt == "BF16":
        u = np.frombuffer(raw, "<u2").astype(np.uint32) << 16
        return u.view(np.float32)
    raise ValueError("cannot decode dtype %s to f32" % dt)


def to_f16_bytes(raw, dt):
    if dt == "F16":
        return raw                       # already f16 payload, byte-lossless
    return _decode_f32(raw, dt).astype(np.float16).tobytes()


def to_f32_bytes(raw, dt):
    if dt == "F32":
        return raw
    return _decode_f32(raw, dt).astype(np.float32).tobytes()


def to_i32_bytes(raw, dt):
    m = {"I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "|i1",
         "U8": "|u1", "U32": "<u4", "U64": "<u8"}
    if dt not in m:
        raise ValueError("cannot decode dtype %s to i32" % dt)
    return np.frombuffer(raw, m[dt]).astype(np.int32).tobytes()


def elems(shape):
    n = 1
    for s in shape:
        n *= s
    return n


# =============================================================================
# GGUF dims: ne0-first == reversed(safetensors row-major shape).
# Byte order of the payload is UNCHANGED (C-contiguous); only metadata reverses.
# =============================================================================
def gguf_dims(shape):
    return list(reversed(shape)) if shape else [1]


# =============================================================================
# Source-name resolution.  The task's template lists some names without a
# `.weight` suffix (hc_*, attn_sink, gate.tid2eid, compressor.ape ...).  Try a
# few candidate forms and use the first present; record what was used.
# =============================================================================
def resolve(model, *candidates):
    for c in candidates:
        if model.has(c):
            return c
    return None


# =============================================================================
# Build the target-tensor plan (name, ggml_type, dims, nbytes, producer).
# producer is a tuple describing how pass-2 fills the bytes.
# =============================================================================
def _expert_count(model, layer):
    e = 0
    while model.has("layers.%d.ffn.experts.%d.w1.weight" % (layer, e)):
        e += 1
    return e


def _nvfp4_geom(model, base0):
    """From expert-0 .weight [out, in/2] -> (out, in, groups)."""
    _, wsh = model.info(base0 + ".weight")
    out, in_half = wsh
    in_cols = in_half * 2
    if in_cols % 16 != 0:
        raise ValueError("%s in=%d not %%16" % (base0, in_cols))
    return out, in_cols, in_cols // 16


class PlanEntry:
    __slots__ = ("name", "type", "dims", "nbytes", "producer")

    def __init__(self, name, type_, dims, nbytes, producer):
        self.name = name
        self.type = type_
        self.dims = dims
        self.nbytes = nbytes
        self.producer = producer


def _add_simple(plan, model, target, src, kind, missing):
    """kind in {f16, f32, i32}."""
    if src is None or not model.has(src):
        missing.append(target)
        return
    dt, sh = model.info(src)
    if kind == "f16":
        nb, typ = elems(sh) * 2, T_F16
    elif kind == "f32":
        nb, typ = elems(sh) * 4, T_F32
    elif kind == "i32":
        nb, typ = elems(sh) * 4, T_I32
    else:
        raise ValueError(kind)
    plan.append(PlanEntry(target, typ, gguf_dims(sh), nb, (kind, src)))


def _add_fp8(plan, model, target, src, missing):
    """FP8 weight (type 18) + E8M0 .scale companion if present."""
    if src is None or not model.has(src):
        missing.append(target)
        return
    dt, sh = model.info(src)               # E4M3, 1 byte/elem
    plan.append(PlanEntry(target, T_FP8, gguf_dims(sh), elems(sh), ("fp8w", src)))
    # source scale: replace trailing `.weight` with `.scale`
    base = src[:-len(".weight")] if src.endswith(".weight") else src
    scale = base + ".scale"
    if model.has(scale):
        _, ssh = model.info(scale)
        tgt_scale = target[:-len(".weight")] + ".scale" if target.endswith(".weight") else target + ".scale"
        plan.append(PlanEntry(tgt_scale, T_FP8, gguf_dims(ssh), elems(ssh), ("fp8s", scale)))


def _add_nvfp4_experts(plan, model, layer, target, src_suffix, nexp, missing):
    """Combined routed-expert tensor + F32 .gscale companion."""
    bases = ["layers.%d.ffn.experts.%d.%s" % (layer, e, src_suffix) for e in range(nexp)]
    if nexp == 0 or not model.has(bases[0] + ".weight"):
        missing.append(target)
        return
    out, in_cols, groups = _nvfp4_geom(model, bases[0])
    per_expert = out * groups * 9
    plan.append(PlanEntry(target, T_NVFP4, [in_cols, out, nexp],
                          nexp * per_expert, ("nvfp4", bases, out, groups)))
    gname = target[:-len(".weight")] + ".gscale" if target.endswith(".weight") else target + ".gscale"
    plan.append(PlanEntry(gname, T_F32, [nexp], nexp * 4, ("gscale", bases)))


def build_plan(model, n_layers):
    plan = []
    missing = []       # required targets whose source is absent

    # ---- common ----
    _add_simple(plan, model, "token_embd.weight",
                resolve(model, "embed.weight", "embed_tokens.weight",
                        "model.embed_tokens.weight"), "f16", missing)

    for L in range(n_layers):
        odd = (L & 1) == 1

        # norms / sinks (F32)
        _add_simple(plan, model, "blk.%d.attn_norm.weight" % L,
                    resolve(model, "layers.%d.attn_norm.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.attn_sinks.weight" % L,
                    resolve(model, "layers.%d.attn.attn_sink" % L,
                            "layers.%d.attn.attn_sink.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.attn_kv_a_norm.weight" % L,
                    resolve(model, "layers.%d.attn.kv_norm.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.attn_q_a_norm.weight" % L,
                    resolve(model, "layers.%d.attn.q_norm.weight" % L), "f32", missing)

        # attn projections (FP8 weight + E8M0 scale companion)
        _add_fp8(plan, model, "blk.%d.attn_kv.weight" % L,
                 resolve(model, "layers.%d.attn.wkv.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.attn_output_a.weight" % L,
                 resolve(model, "layers.%d.attn.wo_a.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.attn_output_b.weight" % L,
                 resolve(model, "layers.%d.attn.wo_b.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.attn_q_a.weight" % L,
                 resolve(model, "layers.%d.attn.wq_a.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.attn_q_b.weight" % L,
                 resolve(model, "layers.%d.attn.wq_b.weight" % L), missing)

        # compressor (layer >= 2)
        if L >= 2:
            _add_simple(plan, model, "blk.%d.attn_compressor_kv.weight" % L,
                        resolve(model, "layers.%d.attn.compressor.wkv.weight" % L), "f16", missing)
            _add_simple(plan, model, "blk.%d.attn_compressor_gate.weight" % L,
                        resolve(model, "layers.%d.attn.compressor.wgate.weight" % L), "f16", missing)
            _add_simple(plan, model, "blk.%d.attn_compressor_norm.weight" % L,
                        resolve(model, "layers.%d.attn.compressor.norm.weight" % L), "f32", missing)
            # ape MUST be F16 (loader hard-checks type==F16)
            _add_simple(plan, model, "blk.%d.attn_compressor_ape.weight" % L,
                        resolve(model, "layers.%d.attn.compressor.ape" % L,
                                "layers.%d.attn.compressor.ape.weight" % L), "f16", missing)

        # indexer (layer >= 2 and even)
        if L >= 2 and not odd:
            _add_fp8(plan, model, "blk.%d.indexer.attn_q_b.weight" % L,
                     resolve(model, "layers.%d.attn.indexer.wq_b.weight" % L), missing)
            _add_simple(plan, model, "blk.%d.indexer.proj.weight" % L,
                        resolve(model, "layers.%d.attn.indexer.weights_proj.weight" % L), "f16", missing)
            _add_simple(plan, model, "blk.%d.indexer_compressor_kv.weight" % L,
                        resolve(model, "layers.%d.attn.indexer.compressor.wkv.weight" % L), "f16", missing)
            _add_simple(plan, model, "blk.%d.indexer_compressor_gate.weight" % L,
                        resolve(model, "layers.%d.attn.indexer.compressor.wgate.weight" % L), "f16", missing)
            _add_simple(plan, model, "blk.%d.indexer_compressor_norm.weight" % L,
                        resolve(model, "layers.%d.attn.indexer.compressor.norm.weight" % L), "f32", missing)
            _add_simple(plan, model, "blk.%d.indexer_compressor_ape.weight" % L,
                        resolve(model, "layers.%d.attn.indexer.compressor.ape" % L,
                                "layers.%d.attn.indexer.compressor.ape.weight" % L), "f16", missing)

        # ffn norm + router
        _add_simple(plan, model, "blk.%d.ffn_norm.weight" % L,
                    resolve(model, "layers.%d.ffn_norm.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.ffn_gate_inp.weight" % L,
                    resolve(model, "layers.%d.ffn.gate.weight" % L), "f16", missing)
        # exp_probs_b.bias is OPTIONAL in the loader
        bsrc = resolve(model, "layers.%d.ffn.gate.bias" % L)
        if bsrc is not None:
            _add_simple(plan, model, "blk.%d.exp_probs_b.bias" % L, bsrc, "f32", missing)
        if L < 3:
            _add_simple(plan, model, "blk.%d.ffn_gate_tid2eid.weight" % L,
                        resolve(model, "layers.%d.ffn.gate.tid2eid" % L,
                                "layers.%d.ffn.gate.tid2eid.weight" % L), "i32", missing)

        # routed experts (NVFP4 combined)  w1->gate  w3->up  w2->down
        nexp = _expert_count(model, L)
        _add_nvfp4_experts(plan, model, L, "blk.%d.ffn_gate_exps.weight" % L, "w1", nexp, missing)
        _add_nvfp4_experts(plan, model, L, "blk.%d.ffn_up_exps.weight" % L, "w3", nexp, missing)
        _add_nvfp4_experts(plan, model, L, "blk.%d.ffn_down_exps.weight" % L, "w2", nexp, missing)

        # shared experts (FP8)  w1->gate  w3->up  w2->down
        _add_fp8(plan, model, "blk.%d.ffn_gate_shexp.weight" % L,
                 resolve(model, "layers.%d.ffn.shared_experts.w1.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.ffn_up_shexp.weight" % L,
                 resolve(model, "layers.%d.ffn.shared_experts.w3.weight" % L), missing)
        _add_fp8(plan, model, "blk.%d.ffn_down_shexp.weight" % L,
                 resolve(model, "layers.%d.ffn.shared_experts.w2.weight" % L), missing)

        # hierarchical-context (hc) coefficients
        _add_simple(plan, model, "blk.%d.hc_attn_base.weight" % L,
                    resolve(model, "layers.%d.hc_attn_base" % L, "layers.%d.hc_attn_base.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.hc_attn_fn.weight" % L,
                    resolve(model, "layers.%d.hc_attn_fn" % L, "layers.%d.hc_attn_fn.weight" % L), "f16", missing)
        _add_simple(plan, model, "blk.%d.hc_attn_scale.weight" % L,
                    resolve(model, "layers.%d.hc_attn_scale" % L, "layers.%d.hc_attn_scale.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.hc_ffn_base.weight" % L,
                    resolve(model, "layers.%d.hc_ffn_base" % L, "layers.%d.hc_ffn_base.weight" % L), "f32", missing)
        _add_simple(plan, model, "blk.%d.hc_ffn_fn.weight" % L,
                    resolve(model, "layers.%d.hc_ffn_fn" % L, "layers.%d.hc_ffn_fn.weight" % L), "f16", missing)
        _add_simple(plan, model, "blk.%d.hc_ffn_scale.weight" % L,
                    resolve(model, "layers.%d.hc_ffn_scale" % L, "layers.%d.hc_ffn_scale.weight" % L), "f32", missing)

    # ---- tail ----
    _add_simple(plan, model, "output_norm.weight",
                resolve(model, "norm.weight", "model.norm.weight"), "f32", missing)
    _add_simple(plan, model, "output_hc_base.weight",
                resolve(model, "hc_head_base", "hc_head_base.weight"), "f32", missing)
    _add_simple(plan, model, "output_hc_fn.weight",
                resolve(model, "hc_head_fn", "hc_head_fn.weight"), "f16", missing)
    _add_simple(plan, model, "output_hc_scale.weight",
                resolve(model, "hc_head_scale", "hc_head_scale.weight"), "f32", missing)
    _add_simple(plan, model, "output.weight",
                resolve(model, "head.weight", "lm_head.weight"), "f16", missing)

    return plan, missing


# =============================================================================
# MTP support block (`mtp.0.*`) -> SEPARATE GGUF for the ds4 MTP loader.
#
# Target names below are EXACTLY the 32 required by
# `axiom_ds4_mtp_resident_weights_load` / DS4_MTP_REQUIRED_TENSORS
# (src/axiom_ds4_weights.cpp:48-81).  The loader's type column there is the
# LEGACY requant pack (Q8_0/Q4_K); this converter emits the NATIVE dtypes
# instead (FP8 pairs / NVFP4 / F32) — the loader needs the native type-detect
# branches (see docs/deepseek_v4_flash_cluster/mtp_native_plan.md) before it
# can consume this file.  All conversions are LOSSLESS (raw byte copy for
# FP8/NVFP4; BF16->F32 exact upcast for norms/hc/router).
# =============================================================================
MTP_PREFIX = "mtp."

MTP_LOADER_REQUIRED = [
    "mtp.0.hc_head_base.weight",
    "mtp.0.hc_head_fn.weight",
    "mtp.0.hc_head_scale.weight",
    "mtp.0.hc_attn_base.weight",
    "mtp.0.hc_ffn_base.weight",
    "mtp.0.hc_attn_fn.weight",
    "mtp.0.hc_attn_scale.weight",
    "mtp.0.hc_ffn_fn.weight",
    "mtp.0.hc_ffn_scale.weight",
    "mtp.0.attn_sinks.weight",
    "mtp.0.attn_q_a.weight",
    "mtp.0.attn_q_b.weight",
    "mtp.0.attn_q_a_norm.weight",
    "mtp.0.attn_output_a.weight",
    "mtp.0.attn_kv.weight",
    "mtp.0.attn_kv_a_norm.weight",
    "mtp.0.attn_output_b.weight",
    "mtp.0.attn_norm.weight",
    "mtp.0.ffn_norm.weight",
    "mtp.0.ffn_gate_shexp.weight",
    "mtp.0.ffn_up_shexp.weight",
    "mtp.0.ffn_down_shexp.weight",
    "mtp.0.ffn_gate_exps.weight",
    "mtp.0.ffn_up_exps.weight",
    "mtp.0.ffn_down_exps.weight",
    "mtp.0.ffn_gate_inp.weight",
    "mtp.0.exp_probs_b.bias",
    "mtp.0.e_proj.weight",
    "mtp.0.h_proj.weight",
    "mtp.0.enorm.weight",
    "mtp.0.hnorm.weight",
    "mtp.0.norm.weight",
]

# FP8 projections: source base (.weight F8_E4M3 + .scale) -> loader name.
MTP_FP8_MAP = [
    ("mtp.0.attn.wq_a", "mtp.0.attn_q_a.weight"),
    ("mtp.0.attn.wq_b", "mtp.0.attn_q_b.weight"),
    ("mtp.0.attn.wkv", "mtp.0.attn_kv.weight"),
    ("mtp.0.attn.wo_a", "mtp.0.attn_output_a.weight"),
    ("mtp.0.attn.wo_b", "mtp.0.attn_output_b.weight"),
    ("mtp.0.e_proj", "mtp.0.e_proj.weight"),
    ("mtp.0.h_proj", "mtp.0.h_proj.weight"),
    ("mtp.0.ffn.shared_experts.w1", "mtp.0.ffn_gate_shexp.weight"),
    ("mtp.0.ffn.shared_experts.w3", "mtp.0.ffn_up_shexp.weight"),
    ("mtp.0.ffn.shared_experts.w2", "mtp.0.ffn_down_shexp.weight"),
]

# F32 tensors (norms/sinks/router/hc). NOTE: the MTP loader wants F32 for the
# hc_*_fn and ffn_gate_inp tensors (unlike the main model, F16 there) — BF16
# source upcasts to F32 exactly, so this stays lossless.
MTP_F32_MAP = [
    (("mtp.0.attn.q_norm.weight",), "mtp.0.attn_q_a_norm.weight"),
    (("mtp.0.attn.kv_norm.weight",), "mtp.0.attn_kv_a_norm.weight"),
    (("mtp.0.attn.attn_sink", "mtp.0.attn.attn_sink.weight"), "mtp.0.attn_sinks.weight"),
    (("mtp.0.attn_norm.weight",), "mtp.0.attn_norm.weight"),
    (("mtp.0.ffn_norm.weight",), "mtp.0.ffn_norm.weight"),
    (("mtp.0.enorm.weight", "mtp.0.enorm"), "mtp.0.enorm.weight"),
    (("mtp.0.hnorm.weight", "mtp.0.hnorm"), "mtp.0.hnorm.weight"),
    (("mtp.0.norm.weight", "mtp.0.norm"), "mtp.0.norm.weight"),
    (("mtp.0.ffn.gate.weight",), "mtp.0.ffn_gate_inp.weight"),
    (("mtp.0.ffn.gate.bias",), "mtp.0.exp_probs_b.bias"),
] + [
    (("mtp.0.%s" % hc, "mtp.0.%s.weight" % hc), "mtp.0.%s.weight" % hc)
    for hc in ("hc_attn_base", "hc_attn_fn", "hc_attn_scale",
               "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
               "hc_head_base", "hc_head_fn", "hc_head_scale")
]

MTP_FP8_SCALE_DTYPES = ("F8_E8M0", "F8_E4M3")


def _mtp_add_f32(plan, model, target, candidates, missing, problems):
    src = resolve(model, *candidates)
    if src is None:
        missing.append("%s  (tried source: %s)" % (target, ", ".join(candidates)))
        return
    dt, sh = model.info(src)
    if dt not in ("F32", "BF16", "F16"):
        problems.append("%s: dtype %s cannot go to F32 losslessly (want F32/BF16/F16)" % (src, dt))
        return
    plan.append(PlanEntry(target, T_F32, gguf_dims(sh), elems(sh) * 4, ("f32", src)))


def _mtp_add_fp8(plan, model, target, src_base, missing, problems, scale_dts):
    w = src_base + ".weight"
    if not model.has(w):
        missing.append("%s  (source %s absent)" % (target, w))
        return
    dt, sh = model.info(w)
    if dt != "F8_E4M3":
        problems.append("%s: expected F8_E4M3 FP8 weight, got %s — refusing to guess" % (w, dt))
        return
    plan.append(PlanEntry(target, T_FP8, gguf_dims(sh), elems(sh), ("fp8w", w)))
    s = src_base + ".scale"
    if not model.has(s):
        problems.append("%s: FP8 weight has no %s companion" % (w, s))
        return
    sdt, ssh = model.info(s)
    if sdt not in MTP_FP8_SCALE_DTYPES:
        problems.append("%s: unexpected scale dtype %s (want one of %s)"
                        % (s, sdt, "/".join(MTP_FP8_SCALE_DTYPES)))
        return
    scale_dts.add(sdt)
    plan.append(PlanEntry(target[:-len(".weight")] + ".scale", T_FP8,
                          gguf_dims(ssh), elems(ssh), ("fp8s", s)))


def _mtp_expert_count(model, w):
    e = 0
    while model.has("mtp.0.ffn.experts.%d.%s.weight" % (e, w)):
        e += 1
    return e


def _mtp_add_experts(plan, model, missing, problems, notes, scale_dts):
    """Combined routed-expert tensors for the MTP layer.

    The mtp experts' companion is named `.scale` in the source index — NOT the
    `.weight_scale`/`.weight_scale_2` pair of the main layers' NVFP4 experts —
    which suggests FP8 (E4M3+scale, like attn/shared), not NVFP4.  This is
    resolved by PROBING the shard headers, never assumed:
      * .weight F8_E4M3 + .scale             -> FP8 family (type 18 pair, 3D-stacked)
      * .weight U8 + .weight_scale(F8_E4M3)
        + .weight_scale_2(F32)               -> NVFP4 family (type 17 + .gscale)
    Anything else is a hard error."""
    targets = (("w1", "mtp.0.ffn_gate_exps"),
               ("w3", "mtp.0.ffn_up_exps"),
               ("w2", "mtp.0.ffn_down_exps"))
    counts = dict((w, _mtp_expert_count(model, w)) for w, _ in targets)
    nexp = counts["w1"]
    if nexp == 0:
        missing.append("mtp.0.ffn_gate_exps.weight  (no mtp.0.ffn.experts.0.w1.weight in index)")
        return
    if counts["w3"] != nexp or counts["w2"] != nexp:
        problems.append("mtp expert count mismatch: w1=%d w3=%d w2=%d"
                        % (counts["w1"], counts["w3"], counts["w2"]))
        return

    base0 = "mtp.0.ffn.experts.0.w1"
    dt0, _ = model.info(base0 + ".weight")
    if dt0 == "F8_E4M3":
        family = "fp8"
    elif dt0 == "U8" and model.has(base0 + ".weight_scale") and model.has(base0 + ".weight_scale_2"):
        family = "nvfp4"
    else:
        problems.append(
            "%s.weight: dtype %s matches NEITHER the FP8 pair family (F8_E4M3 + .scale) "
            "NOR the NVFP4 family (U8 + .weight_scale + .weight_scale_2) — refusing to guess"
            % (base0, dt0))
        return
    notes.append("mtp routed experts: %d experts x 3 proj, native family=%s (probed from shard headers)"
                 % (nexp, family.upper()))

    for w, tgt in targets:
        bases = ["mtp.0.ffn.experts.%d.%s" % (e, w) for e in range(nexp)]
        if family == "fp8":
            wdt0, wsh0 = model.info(bases[0] + ".weight")
            if not model.has(bases[0] + ".scale"):
                problems.append("%s.weight: FP8 expert has no .scale companion" % bases[0])
                continue
            sdt0, ssh0 = model.info(bases[0] + ".scale")
            if sdt0 not in MTP_FP8_SCALE_DTYPES:
                problems.append("%s.scale: unexpected scale dtype %s (want one of %s)"
                                % (bases[0], sdt0, "/".join(MTP_FP8_SCALE_DTYPES)))
                continue
            scale_dts.add(sdt0)
            bad = None
            for b in bases:
                wdt, wsh = model.info(b + ".weight")
                if wdt != wdt0 or wsh != wsh0:
                    bad = "%s.weight: dtype/shape %s%s != expert-0 %s%s" % (b, wdt, wsh, wdt0, wsh0)
                    break
                if not model.has(b + ".scale"):
                    bad = "%s.scale: absent" % b
                    break
                sdt, ssh = model.info(b + ".scale")
                if sdt != sdt0 or ssh != ssh0:
                    bad = "%s.scale: dtype/shape %s%s != expert-0 %s%s" % (b, sdt, ssh, sdt0, ssh0)
                    break
            if bad:
                problems.append("mtp expert family FP8 not uniform: " + bad)
                continue
            plan.append(PlanEntry(tgt + ".weight", T_FP8, gguf_dims(wsh0) + [nexp],
                                  nexp * elems(wsh0),
                                  ("fp8xw", [b + ".weight" for b in bases])))
            plan.append(PlanEntry(tgt + ".scale", T_FP8, gguf_dims(ssh0) + [nexp],
                                  nexp * elems(ssh0),
                                  ("fp8xs", [b + ".scale" for b in bases])))
        else:
            out, in_cols, groups = _nvfp4_geom(model, bases[0])
            bad = None
            for b in bases:
                wdt, wsh = model.info(b + ".weight")
                if wdt != "U8" or wsh != [out, in_cols // 2]:
                    bad = "%s.weight: dtype/shape %s%s != U8[%d,%d]" % (b, wdt, wsh, out, in_cols // 2)
                    break
                if not model.has(b + ".weight_scale") or not model.has(b + ".weight_scale_2"):
                    bad = "%s: missing .weight_scale/.weight_scale_2" % b
                    break
                sdt, ssh = model.info(b + ".weight_scale")
                if sdt != "F8_E4M3" or ssh != [out, groups]:
                    bad = "%s.weight_scale: dtype/shape %s%s != F8_E4M3[%d,%d]" % (b, sdt, ssh, out, groups)
                    break
                gdt, _gsh = model.info(b + ".weight_scale_2")
                if gdt not in ("F32", "BF16", "F16"):
                    bad = "%s.weight_scale_2: dtype %s not decodable to F32" % (b, gdt)
                    break
            if bad:
                problems.append("mtp expert family NVFP4 not uniform: " + bad)
                continue
            per_expert = out * groups * 9
            plan.append(PlanEntry(tgt + ".weight", T_NVFP4, [in_cols, out, nexp],
                                  nexp * per_expert, ("nvfp4", bases, out, groups)))
            plan.append(PlanEntry(tgt + ".gscale", T_F32, [nexp], nexp * 4, ("gscale", bases)))


def build_mtp_plan(model):
    """Returns (plan, missing, problems, notes).  Every entry in `missing` /
    `problems` is fatal for the MTP conversion (no silent drops, no dtype
    guessing)."""
    plan = []
    missing = []
    problems = []
    notes = []
    scale_dts = set()
    for src_base, target in MTP_FP8_MAP:
        _mtp_add_fp8(plan, model, target, src_base, missing, problems, scale_dts)
    for candidates, target in MTP_F32_MAP:
        _mtp_add_f32(plan, model, target, candidates, missing, problems)
    _mtp_add_experts(plan, model, missing, problems, notes, scale_dts)
    if scale_dts:
        notes.append("FP8 .scale companion dtype(s) observed: %s" % ", ".join(sorted(scale_dts)))
    return plan, missing, problems, notes


def _plan_consumed_sources(plan):
    """All source-tensor names a plan reads (for the no-silent-drop audit)."""
    consumed = set()
    for e in plan:
        kind = e.producer[0]
        if kind in ("f16", "f32", "i32", "fp8w", "fp8s"):
            consumed.add(e.producer[1])
        elif kind in ("fp8xw", "fp8xs"):
            consumed.update(e.producer[1])
        elif kind == "nvfp4":
            for b in e.producer[1]:
                consumed.add(b + ".weight")
                consumed.add(b + ".weight_scale")
        elif kind == "gscale":
            for b in e.producer[1]:
                consumed.add(b + ".weight_scale_2")
        else:
            raise ValueError(kind)
    return consumed


# =============================================================================
# Pass-2 data producers -> yield bytes for one target tensor.
# =============================================================================
def _nvfp4_expert_blocks(model, base, out, groups):
    W, wdt, wsh = model.read(base + ".weight")
    S, sdt, ssh = model.read(base + ".weight_scale")
    W = np.frombuffer(W, np.uint8).reshape(wsh)            # [out, in/2]
    S = np.frombuffer(S, np.uint8).reshape(ssh)            # [out, groups] E4M3
    blk = np.empty((out, groups, 9), np.uint8)
    blk[:, :, :8] = W.reshape(out, groups, 8)
    blk[:, :, 8] = S
    return blk.tobytes()


def _expert_global(model, base):
    G, dt, _ = model.read(base + ".weight_scale_2")
    return _decode_f32(G, dt).astype(np.float32).reshape(-1)[0]


def produce(model, entry):
    kind = entry.producer[0]
    if kind == "nvfp4":
        _, bases, out, groups = entry.producer
        return b"".join(_nvfp4_expert_blocks(model, b, out, groups) for b in bases)
    if kind == "gscale":
        _, bases = entry.producer
        return np.array([_expert_global(model, b) for b in bases], np.float32).tobytes()
    if kind == "fp8xw" or kind == "fp8xs":
        # stacked per-expert FP8 raw bytes (weight E4M3 / scale companion), expert-major
        return b"".join(model.read(s)[0] for s in entry.producer[1])
    src = entry.producer[1]
    raw, dt, sh = model.read(src)
    if kind == "fp8w" or kind == "fp8s":
        return raw                          # raw E4M3 / E8M0 bytes
    if kind == "f16":
        return to_f16_bytes(raw, dt)
    if kind == "f32":
        return to_f32_bytes(raw, dt)
    if kind == "i32":
        return to_i32_bytes(raw, dt)
    raise ValueError(kind)


# =============================================================================
# GGUF writer
# =============================================================================
def _gs(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _compute_offsets(plan):
    offs = []
    cur = 0
    for e in plan:
        cur = (cur + ALIGN - 1) // ALIGN * ALIGN
        offs.append(cur)
        cur += e.nbytes
    return offs


def write_gguf(model, plan, out_path, arch="deepseek4"):
    offs = _compute_offsets(plan)
    with open(out_path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))                 # version
        f.write(struct.pack("<Q", len(plan)))         # tensor_count
        f.write(struct.pack("<Q", 1))                 # kv_count
        # one KV: general.architecture (the ds4 loader skips it; informational)
        f.write(_gs("general.architecture") + struct.pack("<I", 8) + _gs(arch))
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
    return dstart, offs


# =============================================================================
# Minimal GGUF reader mirroring the C loader (for round-trip verification)
# =============================================================================
def read_gguf_index(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        tcount = struct.unpack("<Q", f.read(8))[0]
        kvcount = struct.unpack("<Q", f.read(8))[0]

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n)

        def skip_value(t):
            if t in (0, 1, 7):
                f.read(1)
            elif t in (2, 3):
                f.read(2)
            elif t in (4, 5, 6):
                f.read(4)
            elif t in (10, 11, 12):
                f.read(8)
            elif t == 8:
                n = struct.unpack("<Q", f.read(8))[0]
                f.read(n)
            elif t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                cnt = struct.unpack("<Q", f.read(8))[0]
                for _ in range(cnt):
                    skip_value(et)
            else:
                raise ValueError("bad kv type %d" % t)

        for _ in range(kvcount):
            rstr()
            t = struct.unpack("<I", f.read(4))[0]
            skip_value(t)
        tensors = {}
        for _ in range(tcount):
            name = rstr().decode()
            ndim = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndim)]
            typ = struct.unpack("<I", f.read(4))[0]
            off = struct.unpack("<Q", f.read(8))[0]
            tensors[name] = (typ, dims, off)
        pos = f.tell()
        dstart = (pos + ALIGN - 1) // ALIGN * ALIGN
    return ver, dstart, tensors


def read_gguf_bytes(path, dstart, off, nbytes):
    with open(path, "rb") as f:
        f.seek(dstart + off)
        return f.read(nbytes)


# =============================================================================
# Round-trip verification: re-parse the file and assert byte-equality of a
# representative sample of tensors against independent recomputation.
# =============================================================================
def roundtrip_verify(model, plan, out_path):
    ver, dstart, tensors = read_gguf_index(out_path)
    assert ver == 3, "bad version"
    assert len(tensors) == len(plan), "tensor count mismatch"

    # pick representative samples: first of each producer-kind.
    by_kind = {}
    for e in plan:
        by_kind.setdefault(e.producer[0], e)
    checked = 0
    for kind, e in sorted(by_kind.items()):
        typ, dims, off = tensors[e.name]
        assert typ == e.type, "%s type %d != %d" % (e.name, typ, e.type)
        assert dims == e.dims, "%s dims %s != %s" % (e.name, dims, e.dims)
        got = read_gguf_bytes(out_path, dstart, off, e.nbytes)
        exp = produce(model, e)
        assert got == exp, "%s (%s) round-trip BYTE MISMATCH" % (e.name, kind)
        print("  roundtrip OK  %-34s kind=%-6s type=%-2d nbytes=%d"
              % (e.name, kind, typ, e.nbytes))
        checked += 1
    return checked


# =============================================================================
# main
# =============================================================================
def _print_plan_summary(plan, header):
    from collections import Counter, defaultdict
    tc = Counter(e.type for e in plan)
    tb = defaultdict(int)
    for e in plan:
        tb[e.type] += e.nbytes
    total = sum(e.nbytes for e in plan)
    names = {T_F32: "F32", T_F16: "F16", T_I32: "I32", T_NVFP4: "NVFP4", T_FP8: "FP8"}
    print("%s  tensors=%d  data_bytes=%d (%.2f GiB)"
          % (header, len(plan), total, total / (1 << 30)))
    for t in sorted(tc):
        print("  type %-6s: %6d tensors  %14d bytes" % (names.get(t, t), tc[t], tb[t]))


def convert(model_dir, out_path, n_layers, mtp_handled=False):
    model = SafetensorsModel(model_dir)

    # Guard against the historical bug: the mtp.0.* block silently dropped at
    # conversion (foreclosing MTP speculative decode on the native artifact).
    mtp_srcs = [n for n in model.weight_map if n.startswith(MTP_PREFIX)]
    if mtp_srcs and not mtp_handled:
        print("ERROR: the source index carries %d `mtp.*` tensors which this run would "
              "silently DROP.\n"
              "The MTP block is the 35-40 tok/s speculative-decode lever "
              "(docs/axiom_mtp_speculative_decode_recipe_20260601.md) and dropping it at\n"
              "conversion forecloses it on the native artifact. Re-run with "
              "`--mtp-out <mtp.gguf>` to emit the separate MTP support GGUF as well,\n"
              "or pass `--skip-mtp` to drop it DELIBERATELY." % len(mtp_srcs),
              file=sys.stderr)
        raise SystemExit(2)

    plan, missing = build_plan(model, n_layers)
    if missing:
        print("ERROR: %d REQUIRED target tensors have no source in the index:"
              % len(missing), file=sys.stderr)
        for m in missing[:50]:
            print("   MISSING:", m, file=sys.stderr)
        raise SystemExit(2)

    _print_plan_summary(plan, "layers=%d" % n_layers)

    dstart, offs = write_gguf(model, plan, out_path)
    print("wrote %s  (data_start=%d)" % (out_path, dstart))
    n = roundtrip_verify(model, plan, out_path)
    print("round-trip verified %d representative tensors: ALL BYTE-EQUAL" % n)


def convert_mtp(model_dir, out_path):
    """Emit the separate MTP support GGUF (native dtypes, lossless).

    Hard-fails (exit 2) on: any of the 32 loader-required targets missing, any
    unexpected source dtype (probed from shard headers — never guessed), any
    mtp.* source tensor not consumed by the mapping (silent drops forbidden).
    Only `.input_scale` activation-calibration scalars are recognized as
    deliberately unused, and those are reported loudly."""
    model = SafetensorsModel(model_dir)
    all_mtp = sorted(n for n in model.weight_map if n.startswith(MTP_PREFIX))
    if not all_mtp:
        print("ERROR: --mtp-out given but the source index has no `mtp.*` tensors",
              file=sys.stderr)
        raise SystemExit(2)

    plan, missing, problems, notes = build_mtp_plan(model)
    for n in notes:
        print("NOTE:", n)

    # No-silent-drop audit over the WHOLE mtp.* namespace.
    consumed = _plan_consumed_sources(plan)
    ignored = [n for n in all_mtp if n.endswith(".input_scale")]
    unknown = [n for n in all_mtp if n not in consumed and not n.endswith(".input_scale")]
    if ignored:
        print("NOTE: ignoring %d `.input_scale` activation-calibration scalars "
              "(not weights; e.g. %s)" % (len(ignored), ignored[0]))

    # The non-companion target set must be EXACTLY the loader's 32 names.
    targets = [e.name for e in plan
               if not (e.name.endswith(".scale") or e.name.endswith(".gscale"))]
    extra_targets = sorted(set(targets) - set(MTP_LOADER_REQUIRED))
    if len(set(e.name for e in plan)) != len(plan):
        problems.append("duplicate target tensor names in the mtp plan")
    if extra_targets:
        problems.append("targets not in the loader's 32-name table: %s" % ", ".join(extra_targets))
    # (targets absent from the table are already in `missing` — every add is required)

    fatal = False
    if missing:
        fatal = True
        print("ERROR: %d loader-required MTP targets have no source:" % len(missing), file=sys.stderr)
        for m in missing:
            print("   MISSING:", m, file=sys.stderr)
    if problems:
        fatal = True
        print("ERROR: %d MTP dtype/shape probe failures (refusing to guess):" % len(problems),
              file=sys.stderr)
        for p in problems:
            print("   PROBLEM:", p, file=sys.stderr)
    if unknown:
        fatal = True
        print("ERROR: %d `mtp.*` source tensors are NOT consumed by the mapping "
              "(silent drops are the bug this converter fixes):" % len(unknown), file=sys.stderr)
        for u in unknown[:100]:
            print("   UNMAPPED:", u, file=sys.stderr)
        if len(unknown) > 100:
            print("   ... and %d more" % (len(unknown) - 100), file=sys.stderr)
    if fatal:
        print("---- observed mtp.* source names (%d total, first 200) ----" % len(all_mtp),
              file=sys.stderr)
        for n in all_mtp[:200]:
            print("   ", n, file=sys.stderr)
        raise SystemExit(2)

    _print_plan_summary(plan, "mtp")
    dstart, offs = write_gguf(model, plan, out_path, arch="deepseek4_mtp_support")
    print("wrote %s  (data_start=%d)" % (out_path, dstart))
    n = roundtrip_verify(model, plan, out_path)
    print("mtp round-trip verified %d representative tensors: ALL BYTE-EQUAL" % n)


# ------------------------------------------------------------------ self test
def _pack_shard(path, tensors):
    """tensors: list of (name, dtype_str, np.ndarray). Writes a .safetensors."""
    header = {}
    blob = bytearray()
    for name, dt, arr in tensors:
        b = arr.tobytes()
        a = len(blob)
        header[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [a, a + len(b)]}
        blob += b
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(bytes(blob))


def _bf16(arr):
    """f32 ndarray -> BF16 raw stored as uint16 ndarray (top 16 bits)."""
    u = arr.astype(np.float32).view(np.uint32)
    return (u >> 16).astype(np.uint16)


def _put_synthetic_mtp(put, rng, HID, INTER, NEXP, expert_family):
    """Emit a synthetic mtp.0.* block into a synthetic model (via `put`).
    expert_family: 'fp8' (.weight F8_E4M3 + .scale — the naming the real index
    shows) or 'nvfp4' (U8 + .weight_scale + .weight_scale_2)."""
    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    def bf16(*shape):
        return _bf16(rng.standard_normal(shape).astype(np.float32))

    def e4m3(*shape):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)

    put("mtp.0.attn_norm.weight", "F32", f32(HID))
    put("mtp.0.ffn_norm.weight", "F32", f32(HID))
    put("mtp.0.attn.q_norm.weight", "BF16", bf16(HID))       # exercise BF16->F32
    put("mtp.0.attn.kv_norm.weight", "F32", f32(HID))
    put("mtp.0.attn.attn_sink", "F32", f32(4))
    for proj, oc in (("wkv", 16), ("wo_a", 16), ("wo_b", 16), ("wq_a", 16), ("wq_b", 16)):
        put("mtp.0.attn.%s.weight" % proj, "F8_E4M3", e4m3(oc, HID))
        put("mtp.0.attn.%s.scale" % proj, "F8_E8M0", e4m3(1, 2))
    for w, oc, ic in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
        b = "mtp.0.ffn.shared_experts." + w
        put(b + ".weight", "F8_E4M3", e4m3(oc, ic))
        put(b + ".scale", "F8_E8M0", e4m3(max(1, oc // 16), max(1, ic // 16)))
    for e in range(NEXP):
        for w, oc, ic in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
            b = "mtp.0.ffn.experts.%d.%s" % (e, w)
            if expert_family == "fp8":
                put(b + ".weight", "F8_E4M3", e4m3(oc, ic))
                put(b + ".scale", "F8_E8M0", e4m3(max(1, oc // 16), max(1, ic // 16)))
            else:
                put(b + ".weight", "U8", rng.integers(0, 256, size=(oc, ic // 2), dtype=np.uint8))
                put(b + ".weight_scale", "F8_E4M3", e4m3(oc, ic // 16))
                put(b + ".weight_scale_2", "F32", np.array([rng.standard_normal()], np.float32))
            put(b + ".input_scale", "F32", np.array([1.0], np.float32))
    put("mtp.0.ffn.gate.weight", "BF16", bf16(NEXP, HID))
    put("mtp.0.ffn.gate.bias", "F32", f32(NEXP))
    for p in ("e_proj", "h_proj"):
        put("mtp.0.%s.weight" % p, "F8_E4M3", e4m3(HID, 2 * HID))
        put("mtp.0.%s.scale" % p, "F8_E8M0", e4m3(1, 4))
    for nrm in ("enorm", "hnorm", "norm"):
        put("mtp.0.%s.weight" % nrm, "F32", f32(HID))
    for hc in ("hc_attn_base", "hc_attn_scale", "hc_ffn_base", "hc_ffn_scale",
               "hc_head_base", "hc_head_scale"):
        put("mtp.0." + hc, "F32", f32(4))
    for hc in ("hc_attn_fn", "hc_ffn_fn", "hc_head_fn"):
        put("mtp.0." + hc, "BF16", bf16(4))                   # BF16 source -> F32 target


def _build_mtp_only_model(scratch, dirname, seed, expert_family, extra=None):
    """Synthetic model dir containing ONLY an mtp.0.* block (+optional extras)."""
    rng = np.random.default_rng(seed)
    md = os.path.join(scratch, dirname)
    os.makedirs(md, exist_ok=True)
    weight_map = {}
    tensors = []

    def put(name, dt, arr, shard="model-0001.safetensors"):
        weight_map[name] = shard
        tensors.append((name, dt, arr))

    _put_synthetic_mtp(put, rng, 32, 16, 4, expert_family)
    if extra:
        for name, dt, arr in extra:
            put(name, dt, arr)
    _pack_shard(os.path.join(md, "model-0001.safetensors"), tensors)
    with open(os.path.join(md, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)
    return md


def _mtp_full_verify(md, out):
    """Exhaustive byte-equality of EVERY mtp tensor on reparse + 32-name check."""
    model = SafetensorsModel(md)
    plan, missing, problems, _notes = build_mtp_plan(model)
    assert not missing and not problems, (missing, problems)
    targets = set(e.name for e in plan
                  if not (e.name.endswith(".scale") or e.name.endswith(".gscale")))
    assert targets == set(MTP_LOADER_REQUIRED), (
        "target set != loader 32-name table",
        sorted(targets ^ set(MTP_LOADER_REQUIRED)))
    _, dstart, tensors = read_gguf_index(out)
    assert len(tensors) == len(plan), "mtp tensor count mismatch"
    for e in plan:
        typ, dims, off = tensors[e.name]
        assert typ == e.type, "%s type %d != %d" % (e.name, typ, e.type)
        assert dims == e.dims, "%s dims %s != %s" % (e.name, dims, e.dims)
        got = read_gguf_bytes(out, dstart, off, e.nbytes)
        assert got == produce(model, e), "MTP FULL verify mismatch: " + e.name
    print("MTP FULL verify: all %d tensors byte-equal on reparse." % len(plan))


def self_test(scratch):
    rng = np.random.default_rng(0)
    md = os.path.join(scratch, "synthetic_model")
    os.makedirs(md, exist_ok=True)
    NEXP = 4
    NL = 3                       # layers 0,1,2 -> exercises tid2eid, compressor, indexer
    HID = 32                     # expert `in` (must be %16); keep tiny
    INTER = 16
    weight_map = {}
    shard_tensors = {}

    def put(name, dt, arr, shard="model-0001.safetensors"):
        weight_map[name] = shard
        shard_tensors.setdefault(shard, []).append((name, dt, arr))

    def f32(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    def bf16(*shape):
        return _bf16(rng.standard_normal(shape).astype(np.float32))

    def e4m3(*shape):
        return rng.integers(0, 256, size=shape, dtype=np.uint8)

    # global
    put("embed.weight", "BF16", bf16(48, HID))
    put("norm.weight", "F32", f32(HID))
    put("head.weight", "BF16", bf16(48, HID))
    put("hc_head_base", "F32", f32(4))
    put("hc_head_fn", "BF16", bf16(4))
    put("hc_head_scale", "F32", f32(4))

    for L in range(NL):
        odd = (L & 1) == 1
        put("layers.%d.attn_norm.weight" % L, "F32", f32(HID))
        put("layers.%d.attn.attn_sink" % L, "F32", f32(4))
        put("layers.%d.attn.kv_norm.weight" % L, "F32", f32(HID))
        put("layers.%d.attn.q_norm.weight" % L, "F32", f32(HID))
        for proj, oc in (("wkv", 16), ("wo_a", 16), ("wo_b", 16), ("wq_a", 16), ("wq_b", 16)):
            put("layers.%d.attn.%s.weight" % (L, proj), "F8_E4M3", e4m3(oc, HID))
            put("layers.%d.attn.%s.scale" % (L, proj), "F8_E8M0", e4m3(1, 1))
        if L >= 2:
            put("layers.%d.attn.compressor.wkv.weight" % L, "BF16", bf16(16, HID))
            put("layers.%d.attn.compressor.wgate.weight" % L, "BF16", bf16(16, HID))
            put("layers.%d.attn.compressor.norm.weight" % L, "F32", f32(HID))
            put("layers.%d.attn.compressor.ape" % L, "F32", f32(4, 8))
        if L >= 2 and not odd:
            put("layers.%d.attn.indexer.wq_b.weight" % L, "F8_E4M3", e4m3(16, HID))
            put("layers.%d.attn.indexer.wq_b.scale" % L, "F8_E8M0", e4m3(1, 1))
            put("layers.%d.attn.indexer.weights_proj.weight" % L, "BF16", bf16(8, HID))
            put("layers.%d.attn.indexer.compressor.wkv.weight" % L, "BF16", bf16(16, HID))
            put("layers.%d.attn.indexer.compressor.wgate.weight" % L, "BF16", bf16(16, HID))
            put("layers.%d.attn.indexer.compressor.norm.weight" % L, "F32", f32(HID))
            put("layers.%d.attn.indexer.compressor.ape" % L, "F32", f32(4, 4))
        put("layers.%d.ffn_norm.weight" % L, "F32", f32(HID))
        put("layers.%d.ffn.gate.weight" % L, "BF16", bf16(NEXP, HID))
        put("layers.%d.ffn.gate.bias" % L, "F32", f32(NEXP))
        if L < 3:
            put("layers.%d.ffn.gate.tid2eid" % L, "I64", rng.integers(0, NEXP, size=(NEXP,), dtype=np.int64))
        for e in range(NEXP):
            for w, out_c, in_c in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
                b = "layers.%d.ffn.experts.%d.%s" % (L, e, w)
                put(b + ".weight", "U8", rng.integers(0, 256, size=(out_c, in_c // 2), dtype=np.uint8))
                put(b + ".weight_scale", "F8_E4M3", e4m3(out_c, in_c // 16))
                put(b + ".weight_scale_2", "F32", np.array([rng.standard_normal()], np.float32))
                put(b + ".input_scale", "F32", np.array([1.0], np.float32))
        for w, out_c, in_c in (("w1", INTER, HID), ("w3", INTER, HID), ("w2", HID, INTER)):
            b = "layers.%d.ffn.shared_experts.%s" % (L, w)
            put(b + ".weight", "F8_E4M3", e4m3(out_c, in_c))
            put(b + ".scale", "F8_E8M0", e4m3(max(1, out_c // 16), max(1, in_c // 16)))
        put("layers.%d.hc_attn_base" % L, "F32", f32(4))
        put("layers.%d.hc_attn_fn" % L, "BF16", bf16(4))
        put("layers.%d.hc_attn_scale" % L, "F32", f32(4))
        put("layers.%d.hc_ffn_base" % L, "F32", f32(4))
        put("layers.%d.hc_ffn_fn" % L, "BF16", bf16(4))
        put("layers.%d.hc_ffn_scale" % L, "F32", f32(4))

    # mtp.0.* block lives in the SAME shard set as the main model (like the
    # real index). FP8-pair experts here — the family the real `.scale` naming
    # suggests; the NVFP4-family branch is exercised separately below.
    _put_synthetic_mtp(put, rng, HID, INTER, NEXP, "fp8")

    for shard, tl in shard_tensors.items():
        _pack_shard(os.path.join(md, shard), tl)
    with open(os.path.join(md, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)

    print("=== self-test: synthetic model (%d layers, %d experts) ===" % (NL, NEXP))

    # ---- guard: a main conversion that would DROP the mtp block must refuse
    try:
        convert(md, os.path.join(scratch, "must_not_exist.gguf"), NL)
    except SystemExit:
        print("guard OK: main convert refuses to drop mtp.* without --mtp-out/--skip-mtp")
    else:
        raise AssertionError("main convert silently dropped the mtp.* block")

    # ---- MTP conversion, family A: FP8-pair experts
    mtp_out = os.path.join(scratch, "synthetic_mtp_fp8.gguf")
    print("=== self-test: mtp block, FP8-pair experts ===")
    convert_mtp(md, mtp_out)
    _mtp_full_verify(md, mtp_out)

    # ---- main conversion (mtp handled above)
    out = os.path.join(scratch, "synthetic.gguf")
    convert(md, out, NL, mtp_handled=True)

    # extra: exhaustively byte-verify EVERY tensor (not just samples)
    model = SafetensorsModel(md)
    plan, missing = build_plan(model, NL)
    assert not missing, missing
    _, dstart, tensors = read_gguf_index(out)
    for e in plan:
        typ, dims, off = tensors[e.name]
        got = read_gguf_bytes(out, dstart, off, e.nbytes)
        assert got == produce(model, e), "FULL verify mismatch: " + e.name
    print("FULL verify: all %d tensors byte-equal on reparse." % len(plan))

    # ---- MTP conversion, family B: NVFP4 experts (probed, not assumed)
    md_nv = _build_mtp_only_model(scratch, "synthetic_model_mtp_nvfp4", 1, "nvfp4")
    mtp_out_nv = os.path.join(scratch, "synthetic_mtp_nvfp4.gguf")
    print("=== self-test: mtp block, NVFP4 experts ===")
    convert_mtp(md_nv, mtp_out_nv)
    _mtp_full_verify(md_nv, mtp_out_nv)

    # ---- an UNKNOWN mtp.* tensor must be a hard error (no silent drops)
    md_bad = _build_mtp_only_model(
        scratch, "synthetic_model_mtp_unknown", 2, "fp8",
        extra=[("mtp.0.mystery_block.weight", "F32",
                np.zeros((4, 4), np.float32))])
    try:
        convert_mtp(md_bad, os.path.join(scratch, "must_not_exist_mtp.gguf"))
    except SystemExit:
        print("guard OK: convert_mtp hard-errors on an unmapped mtp.* tensor")
    else:
        raise AssertionError("convert_mtp silently dropped an unknown mtp.* tensor")

    # ---- an unexpected expert dtype must be a hard error (probe, don't guess)
    md_odd = _build_mtp_only_model(scratch, "synthetic_model_mtp_odddtype", 3, "fp8")
    # overwrite expert-0 w1 with a BF16 weight (neither FP8 pair nor NVFP4)
    idx_path = os.path.join(md_odd, "model.safetensors.index.json")
    with open(idx_path, "r") as f:
        wm = json.load(f)
    wm["weight_map"]["mtp.0.ffn.experts.0.w1.weight"] = "model-0002.safetensors"
    _pack_shard(os.path.join(md_odd, "model-0002.safetensors"),
                [("mtp.0.ffn.experts.0.w1.weight", "BF16", _bf16(np.zeros((16, 32), np.float32)))])
    with open(idx_path, "w") as f:
        json.dump(wm, f)
    try:
        convert_mtp(md_odd, os.path.join(scratch, "must_not_exist_mtp2.gguf"))
    except SystemExit:
        print("guard OK: convert_mtp refuses an unexpected expert dtype")
    else:
        raise AssertionError("convert_mtp guessed on an unexpected expert dtype")

    print("SELF-TEST PASS.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--layers", type=int, default=DS4_LAYER_COUNT)
    ap.add_argument("--mtp-out", metavar="MTP_GGUF",
                    help="also emit the separate mtp.0.* support GGUF (native dtypes, "
                         "lossless). Runs BEFORE the main conversion (fails fast).")
    ap.add_argument("--skip-mtp", action="store_true",
                    help="DELIBERATELY drop the mtp.0.* block (old behavior; "
                         "forecloses MTP speculative decode on this artifact)")
    ap.add_argument("--self-test", nargs="?", const=".", metavar="SCRATCH_DIR")
    args = ap.parse_args()

    if args.self_test is not None:
        self_test(args.self_test)
        return
    if not args.model_dir or not (args.out or args.mtp_out):
        ap.error("model_dir plus out and/or --mtp-out are required (or use --self-test)")
    if args.mtp_out:
        # MTP first: it is tiny compared to the 164G main pass, so a bad MTP
        # mapping fails BEFORE hours of main-conversion I/O are spent.
        convert_mtp(args.model_dir, args.mtp_out)
    if args.out:
        n = max(1, min(args.layers, DS4_LAYER_COUNT))
        convert(args.model_dir, args.out, n,
                mtp_handled=bool(args.mtp_out) or args.skip_mtp)


if __name__ == "__main__":
    main()
