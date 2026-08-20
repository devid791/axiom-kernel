#!/usr/bin/env python3
"""NVFP4/FP8 safetensors -> ds4-GGUF converter (spike, multi-tensor). Encodes real DeepSeek
weights into the ds4 GGUF types and writes a standard GGUF ds4's parser reads by name:
  - routed expert (gate w1)  -> NVFP4 (type 17): 9B/16 block = 8 E2M1-nibble bytes + 1 e4m3 scale
  - attn weight              -> FP8 (type 18): raw E4M3 bytes
  - attn scale companion     -> FP8 (type 18): raw E8M0 bytes
Round-trips each vs the direct safetensors decode. Usage: nvfp4_to_gguf.py <model_dir> <layer> <expert> <out.gguf>
"""
import json, struct, sys
import numpy as np

d = sys.argv[1]; L = int(sys.argv[2]); E = int(sys.argv[3]); outp = sys.argv[4]
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
def read(name):
    p = d + "/" + idx[name]; f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n)); base = 8 + n; e = h[name]; a, b = e["data_offsets"]
    f.seek(base + a); raw = f.read(b - a); f.close(); return raw, e["shape"]

NVFP4, FP8 = 17, 18
mag = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def e2m1(x): v = mag[x & 7]; return np.where((x & 8) > 0, -v, v).astype(np.float32)
def e4m3(b):
    b = b.astype(np.uint32); s = (b >> 7) & 1; e = (b >> 3) & 0xf; m = b & 7
    v = np.where(e == 0, m * (0.015625 / 8), (1 + m / 8.) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s > 0, -v, v).astype(np.float32)

tensors = []   # (name, ggml_type, dims_ne0_first, data_bytes, checker)

# --- NVFP4 routed experts (gate), COMBINED N experts + per-expert global scales ---
import os
NEXP = 4          # subset for the test (real model = 256); combined [in, out, NEXP]
TE = 2            # expert index to reference for the loader probe
blocks_all = []; globals_all = []; Wt = St = None
def decode_direct(W, S, G, out_rows, in_cols):
    lo = W & 0xf; hi = (W >> 4) & 0xf
    d = np.empty((out_rows, in_cols), np.float32); d[:, 0::2] = e2m1(lo); d[:, 1::2] = e2m1(hi)
    return d * np.repeat(e4m3(S), 16, axis=1) * G
for e in range(NEXP):
    eb = "layers.%d.ffn.experts.%d.w1" % (L, e)
    W, wsh = read(eb + ".weight"); S, ssh = read(eb + ".weight_scale"); G, _ = read(eb + ".weight_scale_2")
    W = np.frombuffer(W, np.uint8).reshape(wsh); S = np.frombuffer(S, np.uint8).reshape(ssh)
    G = float(np.frombuffer(G, np.float32)[0]); out_rows, in_half = wsh; in_cols = in_half * 2; groups = in_cols // 16
    blk = np.empty((out_rows, groups, 9), np.uint8); blk[:, :, :8] = W.reshape(out_rows, groups, 8); blk[:, :, 8] = S
    blocks_all.append(blk.tobytes()); globals_all.append(G)
    if e == TE: Wt, St, Gt = W, S, G
combined = b"".join(blocks_all)          # experts contiguous: expert e at e*(out_rows*groups*9)
tensors.append(("blk.%d.ffn_gate_exps.weight" % L, NVFP4, [in_cols, out_rows, NEXP], combined, None))
tensors.append(("blk.%d.ffn_gate_exps.gscale" % L, 0, [NEXP], np.array(globals_all, np.float32).tobytes(), None))  # F32=0
# loader-probe reference for expert TE
_Wd = decode_direct(Wt, St, Gt, out_rows, in_cols)
_x = np.sin(np.arange(in_cols, dtype=np.float32) * 0.01).astype(np.float32)
_y = (_Wd.astype(np.float64) @ _x).astype(np.float32)
_od = os.path.dirname(outp) or "."
_x.tofile(_od + "/ldr_x.bin"); _y.tofile(_od + "/ldr_yref.bin")
open(_od + "/ldr_meta.txt", "w").write("%d %d %d %d\n" % (out_rows, in_cols, NEXP, TE))

# --- FP8 attn (weight + scale companion) ---
AW, awsh = read("layers.%d.attn.wkv.weight" % L)     # E4M3 [out,in]
AS, assh = read("layers.%d.attn.wkv.scale" % L)      # E8M0 [out/128,in/128]
tensors.append(("blk.%d.attn_kv.weight" % L, FP8, [awsh[1], awsh[0]], AW, lambda raw, AW=AW: float((np.frombuffer(raw, np.uint8) != np.frombuffer(AW, np.uint8)).sum())))
tensors.append(("blk.%d.attn_kv.scale" % L, FP8, [assh[1], assh[0]], AS, lambda raw, AS=AS: float((np.frombuffer(raw, np.uint8) != np.frombuffer(AS, np.uint8)).sum())))

# --- write GGUF (multi-tensor, aligned) ---
def gs(s): return struct.pack("<Q", len(s)) + s.encode()
ALIGN = 32
# compute data-section offsets
offs = []; cur = 0
for (_, _, _, data, _) in tensors:
    cur = (cur + ALIGN - 1) // ALIGN * ALIGN; offs.append(cur); cur += len(data)
with open(outp, "wb") as f:
    f.write(b"GGUF"); f.write(struct.pack("<I", 3))
    f.write(struct.pack("<Q", len(tensors))); f.write(struct.pack("<Q", 1))
    f.write(gs("general.architecture") + struct.pack("<I", 8) + gs("deepseek4"))
    for (name, typ, dims, data, _), off in zip(tensors, offs):
        f.write(gs(name)); f.write(struct.pack("<I", len(dims)))
        for dv in dims: f.write(struct.pack("<Q", dv))
        f.write(struct.pack("<I", typ)); f.write(struct.pack("<Q", off))
    pos = f.tell(); f.write(b"\x00" * ((-pos) % ALIGN)); dstart = f.tell()
    for (_, _, _, data, _), off in zip(tensors, offs):
        pad = (dstart + off) - f.tell(); f.write(b"\x00" * pad); f.write(data)
print("wrote %s: %d tensors" % (outp, len(tensors)))

# --- round-trip each tensor from the written file ---
raw = open(outp, "rb").read()
for (name, typ, dims, data, chk), off in zip(tensors, offs):
    got = raw[dstart + off: dstart + off + len(data)]
    err = chk(got)
    print("  %-32s type=%d nbytes=%d roundtrip_err=%.9g %s" % (name, typ, len(data), err, "OK" if err == 0.0 else "FAIL"))
