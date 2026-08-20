#!/usr/bin/env python3
"""Dump a real DeepSeek-V4-Flash FP8 weight (F8_E4M3 + F8_E8M0 128x128 block scale) +
a deterministic input + an f64 reference matvec, to cross-check the libaxiom FP8 kernel.
Usage: fp8_ref.py <model_dir> <tensor_base e.g. layers.1.attn.wkv> <out_dir>
"""
import json, struct, sys
import numpy as np

d = sys.argv[1]; base = sys.argv[2]; out = sys.argv[3]
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]
def read(name):
    p = d + "/" + idx[name]
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); b = 8 + n
    e = h[name]; a, z = e["data_offsets"]; f.seek(b + a); raw = f.read(z - a); f.close()
    return raw, e["shape"]
wb, wsh = read(base + ".weight"); sb, ssh = read(base + ".scale")
W = np.frombuffer(wb, np.uint8).reshape(wsh)   # [rows, cols]
S = np.frombuffer(sb, np.uint8).reshape(ssh)   # [rows/128, cols/128]
rows, cols = wsh
def e4m3(b):
    b = b.astype(np.uint32); s = (b >> 7) & 1; e = (b >> 3) & 0xf; m = b & 7
    v = np.where(e == 0, m * (0.015625 / 8), (1 + m / 8.) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s > 0, -v, v).astype(np.float32)
def e8m0(b): return np.exp2(b.astype(np.float32) - 127.0).astype(np.float32)
Wf = e4m3(W)
Sx = np.repeat(np.repeat(e8m0(S), 128, axis=0), 128, axis=1)[:rows, :cols]
Wd = Wf * Sx
x = np.sin(np.arange(cols, dtype=np.float32) * 0.01).astype(np.float32)
y = (Wd.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
W.tofile(out + "/fp8_w.bin"); S.tofile(out + "/fp8_s.bin")
x.tofile(out + "/fp8_x.bin"); y.tofile(out + "/fp8_yref.bin")
open(out + "/fp8_meta.txt", "w").write("%d %d\n" % (rows, cols))
print("%s rows=%d cols=%d scale=%s Wd[absmean]=%.5f yref[min/max]=%.4f/%.4f"
      % (base, rows, cols, ssh, np.abs(Wd).mean(), y.min(), y.max()))
