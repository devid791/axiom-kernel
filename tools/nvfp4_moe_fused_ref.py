#!/usr/bin/env python3
"""Synthetic full-MoE oracle for the fused NVFP4 routed-expert chain
(src/axiom_cuda_nvfp4_moe.cu). Builds layout-faithful de-blocked combined planes —
weight U8 [experts, rows, cols/2] (2x signed-E2M1 nibbles/byte, low nibble = even col),
block_scale U8 e4m3 [experts, rows, cols/16] (group 16), global_scale f32 [experts],
expert e at e*stride (the exact per-expert offset math of the host loop,
src/axiom_runtime.cpp:7374-7397) — plus router indices/weights, and computes the f64
reference  y = sum_j rw[j] * (Wd[e_j] . silu(Wg[e_j].x) * (Wu[e_j].x))  and the
per-slot mid plane. Small but layout-faithful: experts=8, topk=6, hidden=256,
expert_hidden=128 (both %32 as the fused kernels require). e2m1/e4m3 decode reused
verbatim from the proven tools/nvfp4_ffn_ref.py. Deterministic (fixed seed).
Usage: nvfp4_moe_fused_ref.py <out_dir>
"""
import sys
import numpy as np

out = sys.argv[1]
E, K, H, EH = 8, 6, 256, 128
rng = np.random.default_rng(67)

# ---- decode: copied from tools/nvfp4_ffn_ref.py (proven against the GB10 kernel) ----
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

def planes(rows, cols):
    """One combined [experts, rows, cols/2] nibble plane + [experts, rows, cols/16]
    e4m3 scale plane (exponent band 4..9 => scales 2^-3..2^2*1.875, keeps the dynamic
    range moderate so the f32-vs-f64 tolerance stays tight) + f32 [experts] globals."""
    w = rng.integers(0, 256, size=(E, rows, cols // 2), dtype=np.uint8)
    sc = ((rng.integers(4, 10, size=(E, rows, cols // 16)) << 3) |
          rng.integers(0, 8, size=(E, rows, cols // 16))).astype(np.uint8)
    gs = rng.uniform(0.01, 0.05, size=E).astype(np.float32)
    return w, sc, gs

GW, GS, GG = planes(EH, H)   # gate: rows=expert_hidden, cols=hidden
UW, US, UG = planes(EH, H)   # up
DW, DS, DG = planes(H, EH)   # down: rows=hidden, cols=expert_hidden

x = (np.sin(np.arange(H, dtype=np.float32) * 0.37) * 0.5).astype(np.float32)
idx = rng.choice(E, size=K, replace=False).astype(np.uint32)
rw = rng.uniform(0.05, 1.0, size=K); rw = (rw / rw.sum()).astype(np.float32)

xd = x.astype(np.float64)
mid = np.zeros((K, EH), np.float64)
y = np.zeros(H, np.float64)
for j in range(K):
    e = int(idx[j])
    g = decode(GW[e], GS[e], GG[e]).astype(np.float64) @ xd
    u = decode(UW[e], US[e], UG[e]).astype(np.float64) @ xd
    h = (g / (1.0 + np.exp(-g))) * u                      # silu(g) * u
    mid[j] = h
    y += float(rw[j]) * (decode(DW[e], DS[e], DG[e]).astype(np.float64) @ h)

GW.tofile(out + "/gate_w.bin"); GS.tofile(out + "/gate_bs.bin"); GG.tofile(out + "/gate_gs.bin")
UW.tofile(out + "/up_w.bin");   US.tofile(out + "/up_bs.bin");   UG.tofile(out + "/up_gs.bin")
DW.tofile(out + "/down_w.bin"); DS.tofile(out + "/down_bs.bin"); DG.tofile(out + "/down_gs.bin")
x.tofile(out + "/x.bin"); idx.tofile(out + "/idx.bin"); rw.tofile(out + "/rw.bin")
mid.astype(np.float32).tofile(out + "/mid_ref.bin")
y.astype(np.float32).tofile(out + "/y_ref.bin")
open(out + "/meta.txt", "w").write("%d %d %d %d\n" % (E, K, H, EH))
print("nvfp4-moe-fused-ref: experts=%d topk=%d hidden=%d expert_hidden=%d idx=%s y[min/max]=%.4f/%.4f"
      % (E, K, H, EH, idx.tolist(), y.min(), y.max()))
