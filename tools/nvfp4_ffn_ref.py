#!/usr/bin/env python3
"""Dump a real DeepSeek-V4-Flash-NVFP4 routed-expert FFN (gate w1, up w3, down w2) +
a deterministic input + an f64 reference output, so the libaxiom NVFP4 kernel can be
cross-checked on a full expert FFN on real weights. Uses the safetensors index to
locate each tensor's shard. Usage: nvfp4_ffn_ref.py <model_dir> <layer> <expert> <out_dir>
"""
import json, struct, sys
import numpy as np

d = sys.argv[1]; L = int(sys.argv[2]); E = int(sys.argv[3]); out = sys.argv[4]
idx = json.load(open(d + "/model.safetensors.index.json"))["weight_map"]

def read(name):
    p = d + "/" + idx[name]
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
    e = h[name]; a, b = e["data_offsets"]; f.seek(base + a); raw = f.read(b - a); f.close()
    return raw, e["shape"]

def load(base):
    wb, wsh = read(base + ".weight"); sb, ssh = read(base + ".weight_scale"); gb, _ = read(base + ".weight_scale_2")
    W = np.frombuffer(wb, np.uint8).reshape(wsh); S = np.frombuffer(sb, np.uint8).reshape(ssh)
    G = float(np.frombuffer(gb, np.float32)[0]); return W, S, G

mag = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def e2m1(x): v = mag[x & 7]; return np.where((x & 8) > 0, -v, v).astype(np.float32)
def e4m3(b):
    b = b.astype(np.uint32); s = (b >> 7) & 1; e = (b >> 3) & 0xf; m = b & 7
    v = np.where(e == 0, m * (0.015625 / 8), (1 + m / 8.) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s > 0, -v, v).astype(np.float32)
def decode(W, S, G):
    low = W & 0xf; high = (W >> 4) & 0xf
    dec = np.empty((W.shape[0], W.shape[1] * 2), np.float32); dec[:, 0::2] = e2m1(low); dec[:, 1::2] = e2m1(high)
    return dec * np.repeat(e4m3(S), 16, axis=1) * G

base = "layers.%d.ffn.experts.%d" % (L, E)
W1, S1, G1 = load(base + ".w1"); W3, S3, G3 = load(base + ".w3"); W2, S2, G2 = load(base + ".w2")
Wf1 = decode(W1, S1, G1); Wf3 = decode(W3, S3, G3); Wf2 = decode(W2, S2, G2)
inter, hidden = Wf1.shape          # w1: [inter, hidden]
x = np.sin(np.arange(hidden, dtype=np.float32) * 0.01).astype(np.float32)
g = Wf1.astype(np.float64) @ x; u = Wf3.astype(np.float64) @ x
h = (g / (1.0 + np.exp(-g))) * u   # silu(g) * u   -> [inter]
y = (Wf2.astype(np.float64) @ h).astype(np.float32)  # [hidden]

W1.tofile(out + "/w1_w.bin"); S1.tofile(out + "/w1_s.bin")
W3.tofile(out + "/w3_w.bin"); S3.tofile(out + "/w3_s.bin")
W2.tofile(out + "/w2_w.bin"); S2.tofile(out + "/w2_s.bin")
x.tofile(out + "/ffn_x.bin"); y.tofile(out + "/ffn_yref.bin")
open(out + "/ffn_meta.txt", "w").write("%d %d %.9g %.9g %.9g\n" % (inter, hidden, G1, G3, G2))
print("layer=%d expert=%d inter=%d hidden=%d yref[min/max]=%.4f/%.4f" % (L, E, inter, hidden, y.min(), y.max()))
