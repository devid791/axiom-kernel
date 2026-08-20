#!/usr/bin/env python3
"""Dump deterministic synthetic test vectors + f64 reference outputs for the AXIOM PME
fused serving kernel (src/axiom_cuda_pme.cu): base NVFP4/FP8 GEMV + fused rank-r memory
pack  y = dequant(W)@x + (B@(A@x))*pack_scale. No model files needed — self-contained,
so the smoke runs on GB10 in minutes. e2m1/e4m3 decode copied from the proven
tools/nvfp4_ffn_ref.py, e8m0 from tools/fp8_ref.py (must stay bit-equal to the device
decode in src/axiom_cuda_nvfp4.cu). Consumed by tests/axiom_pme_fused_smoke.cpp.
Usage: pme_fused_ref.py <out_dir> [seed=20260706]
"""
import os, sys
import numpy as np

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260706
rng = np.random.default_rng(seed)
os.makedirs(out, exist_ok=True)

# ---- decode (copied from nvfp4_ffn_ref.py / fp8_ref.py — proven vs the device kernels) ----
mag = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def e2m1(x): v = mag[x & 7]; return np.where((x & 8) > 0, -v, v).astype(np.float32)
def e4m3(b):
    b = b.astype(np.uint32); s = (b >> 7) & 1; e = (b >> 3) & 0xf; m = b & 7
    v = np.where(e == 0, m * (0.015625 / 8), (1 + m / 8.) * np.exp2(e.astype(np.float32) - 7))
    return np.where(s > 0, -v, v).astype(np.float32)
def e8m0(b): return np.exp2(b.astype(np.float32) - 127.0).astype(np.float32)
def nv_decode(W, S, G):     # W u8 [rows, cols/2] packed nibbles (low = even col), S u8 [rows, cols/16]
    low = W & 0xf; high = (W >> 4) & 0xf
    dec = np.empty((W.shape[0], W.shape[1] * 2), np.float32); dec[:, 0::2] = e2m1(low); dec[:, 1::2] = e2m1(high)
    return dec * np.repeat(e4m3(S), 16, axis=1) * G

def lora(A, B, x, s):       # f64 reference of the fused pack update (B @ (A @ x)) * s
    return (B.astype(np.float64) @ (A.astype(np.float64) @ x.astype(np.float64))) * s

# ---- case 1: NVFP4 base (routed-expert layout), rank at the AXIOM_PME_MAX_RANK boundary.
# rows deliberately NOT a multiple of blockDim 64 (partial last block exercises the
# r>=rows guard + staging participation of inactive rows).
NV_ROWS, NV_COLS, NV_RANK = 112, 512, 256
NV_GLOBAL, NV_PACK_SCALE = 0.75, 0.125           # pack_scale = alpha/rank (e.g. 32/256)
nv_w = rng.integers(0, 256, size=(NV_ROWS, NV_COLS // 2), dtype=np.uint8)
nv_e = rng.integers(5, 9, size=(NV_ROWS, NV_COLS // 16)).astype(np.uint32)   # e4m3 codes, 0.25..3.75
nv_m = rng.integers(0, 8, size=(NV_ROWS, NV_COLS // 16)).astype(np.uint32)
nv_bs = ((nv_e << 3) | nv_m).astype(np.uint8)
nv_x = (rng.standard_normal(NV_COLS) * 0.1).astype(np.float32)
nv_a = (rng.standard_normal((NV_RANK, NV_COLS)) * 0.05).astype(np.float32)
nv_b = (rng.standard_normal((NV_ROWS, NV_RANK)) * 0.05).astype(np.float32)
nv_yoff = (nv_decode(nv_w, nv_bs, NV_GLOBAL).astype(np.float64) @ nv_x.astype(np.float64))
nv_yon = (nv_yoff + lora(nv_a, nv_b, nv_x, NV_PACK_SCALE)).astype(np.float32)
nv_yoff = nv_yoff.astype(np.float32)

# ---- case 2: FP8 base (attn/shared layout: e4m3 weight + e8m0 128x128 block scale),
# small rank (staging loop with idle threads: rank < blockDim).
F8_ROWS, F8_COLS, F8_RANK = 128, 256, 16
F8_PACK_SCALE = 0.5
f8_s_bits = rng.integers(0, 2, size=(F8_ROWS, F8_COLS)).astype(np.uint32)
f8_e = rng.integers(4, 9, size=(F8_ROWS, F8_COLS)).astype(np.uint32)         # no NaN codes
f8_m = rng.integers(0, 8, size=(F8_ROWS, F8_COLS)).astype(np.uint32)
f8_w = ((f8_s_bits << 7) | (f8_e << 3) | f8_m).astype(np.uint8)
f8_s = rng.integers(125, 130, size=(F8_ROWS // 128, F8_COLS // 128), dtype=np.uint8)  # scales 0.25..8
f8_x = (rng.standard_normal(F8_COLS) * 0.1).astype(np.float32)
f8_a = (rng.standard_normal((F8_RANK, F8_COLS)) * 0.05).astype(np.float32)
f8_b = (rng.standard_normal((F8_ROWS, F8_RANK)) * 0.05).astype(np.float32)
f8_wd = e4m3(f8_w) * np.repeat(np.repeat(e8m0(f8_s), 128, axis=0), 128, axis=1)[:F8_ROWS, :F8_COLS]
f8_yoff = (f8_wd.astype(np.float64) @ f8_x.astype(np.float64))
f8_yon = (f8_yoff + lora(f8_a, f8_b, f8_x, F8_PACK_SCALE)).astype(np.float32)
f8_yoff = f8_yoff.astype(np.float32)

nv_w.tofile(out + "/nv_w.bin"); nv_bs.tofile(out + "/nv_bs.bin")
nv_x.tofile(out + "/nv_x.bin"); nv_a.tofile(out + "/nv_a.bin"); nv_b.tofile(out + "/nv_b.bin")
nv_yoff.tofile(out + "/nv_yoff.bin"); nv_yon.tofile(out + "/nv_yon.bin")
f8_w.tofile(out + "/f8_w.bin"); f8_s.tofile(out + "/f8_s.bin")
f8_x.tofile(out + "/f8_x.bin"); f8_a.tofile(out + "/f8_a.bin"); f8_b.tofile(out + "/f8_b.bin")
f8_yoff.tofile(out + "/f8_yoff.bin"); f8_yon.tofile(out + "/f8_yon.bin")
open(out + "/pme_meta.txt", "w").write("%d %d %d %.9g %.9g %d %d %d %.9g\n" % (
    NV_ROWS, NV_COLS, NV_RANK, NV_GLOBAL, NV_PACK_SCALE, F8_ROWS, F8_COLS, F8_RANK, F8_PACK_SCALE))
print("pme_fused_ref: seed=%d nv[%dx%d r=%d] yoff|max|=%.4f yon|max|=%.4f  f8[%dx%d r=%d] yoff|max|=%.4f yon|max|=%.4f"
      % (seed, NV_ROWS, NV_COLS, NV_RANK, np.abs(nv_yoff).max(), np.abs(nv_yon).max(),
         F8_ROWS, F8_COLS, F8_RANK, np.abs(f8_yoff).max(), np.abs(f8_yon).max()))
