#!/usr/bin/env python3
"""Decode a real NVFP4 routed-expert weight from a DeepSeek-V4-Flash-NVFP4 shard and
sanity-check the magnitudes. Independent (numpy) reference for the Axiom NVFP4 decode
(src/axiom_cuda_nvfp4.cu). Reads only the header + the 3 tensors, not the whole file.

Usage: python axiom_nvfp4_real_decode.py <shard.safetensors> [tensor_base]
  tensor_base default: layers.1.ffn.experts.0.w1
Layout (confirmed): .weight U8[out,in/2] (2x E2M1/byte), .weight_scale F8_E4M3[out,in/16]
(group 16), .weight_scale_2 F32 global. value = e2m1(nib) * e4m3(block) * global.
"""
import json, struct, sys
import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: axiom_nvfp4_real_decode.py <shard.safetensors> [tensor_base] [dump <output_dir>]")
path = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "layers.1.ffn.experts.0.w1"

f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); off = 8 + n
def raw(name):
    e = hdr[name]; a, b = e["data_offsets"]; f.seek(off + a); return f.read(b - a), e["dtype"], e["shape"]
wb, wdt, wsh = raw(base + ".weight")
sb, sdt, ssh = raw(base + ".weight_scale")
gb, gdt, gsh = raw(base + ".weight_scale_2")
print("weight", wdt, wsh, "| scale", sdt, ssh, "| global", gdt, gsh)

W = np.frombuffer(wb, dtype=np.uint8).reshape(wsh)
S = np.frombuffer(sb, dtype=np.uint8).reshape(ssh)
G = float(np.frombuffer(gb, dtype=np.float32)[0])
mag = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
def e2m1(x): v = mag[x & 7]; return np.where((x & 8) > 0, -v, v).astype(np.float32)
low = W & 0x0f; high = (W >> 4) & 0x0f
dec = np.empty((wsh[0], wsh[1] * 2), np.float32); dec[:, 0::2] = e2m1(low); dec[:, 1::2] = e2m1(high)
def e4m3(b):
    b = b.astype(np.uint32); s = (b >> 7) & 1; e = (b >> 3) & 0xf; m = b & 7
    v = np.where(e == 0, m * (0.015625 / 8.0), (1.0 + m / 8.0) * np.exp2(e.astype(np.float32) - 7.0))
    return np.where(s > 0, -v, v).astype(np.float32)
Sx = np.repeat(e4m3(S), 16, axis=1)
Wf = dec * Sx * G
print("global=%.6g block_scale[min/max]=%.4g/%.4g" % (G, e4m3(S).min(), e4m3(S).max()))
print("DECODED WEIGHT: min=%.5f max=%.5f mean=%.6f absmean=%.6f std=%.6f"
      % (Wf.min(), Wf.max(), Wf.mean(), np.abs(Wf).mean(), Wf.std()))
plausible = (abs(Wf).max() < 50) and (np.abs(Wf).mean() < 2)
print("PLAUSIBILITY:", "OK" if plausible else "SUSPECT (wrong layout/scale?)")

# Optional: dump raw tensors + a deterministic input + f64-accumulated reference
# matvec so the libaxiom CUDA kernel can be cross-checked on REAL weights.
if len(sys.argv) > 3 and sys.argv[3] == "dump":
    d = sys.argv[4] if len(sys.argv) > 4 else "."
    rows, cols = Wf.shape
    x = np.sin(np.arange(cols, dtype=np.float32) * 0.01).astype(np.float32)
    yref = (Wf.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
    W.tofile(d + "/nvfp4_w.bin"); S.tofile(d + "/nvfp4_s.bin")
    x.tofile(d + "/nvfp4_x.bin"); yref.tofile(d + "/nvfp4_yref.bin")
    open(d + "/nvfp4_meta.txt", "w").write("%d %d %.9g\n" % (rows, cols, G))
    print("DUMP rows=%d cols=%d yref[min/max]=%.4f/%.4f -> %s" % (rows, cols, yref.min(), yref.max(), d))
