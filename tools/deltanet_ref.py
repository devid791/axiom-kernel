#!/usr/bin/env python3
# Gated DeltaNet decode-step reference (numpy), mirroring transformers
# qwen3_next: torch_recurrent_gated_delta_rule + causal depthwise conv1d + SiLU.
# Emits a binary vector file consumed by tools/axiom_deltanet_spike.cpp for
# byte-free, model-free validation of the Axiom C implementation.
#
# usage: python3 tools/deltanet_ref.py /tmp/deltanet_vec.bin
import sys, struct
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/deltanet_vec.bin"
rng = np.random.default_rng(1234)
f32 = np.float32

# small but representative dims (math is dimension-general)
seq, vheads, hd = 8, 4, 128
conv_dim, kernel = 64, 4

def silu(x): return x / (1.0 + np.exp(-x))

# ---------- causal depthwise conv1d (k=4, no bias) + SiLU ----------
conv_w  = rng.standard_normal((conv_dim, kernel)).astype(f32)        # [C][K]
conv_in = rng.standard_normal((seq, conv_dim)).astype(f32)           # [T][C]
conv_out = np.zeros((seq, conv_dim), dtype=f32)
# pad 3 zeros at the front (causal): out[t][c] = silu(sum_j w[c][j]*x[t-3+j][c])
xpad = np.concatenate([np.zeros((kernel - 1, conv_dim), f32), conv_in], axis=0)
for t in range(seq):
    win = xpad[t:t + kernel]                      # [K][C]
    acc = (win.T * conv_w).sum(axis=1)            # [C]
    conv_out[t] = silu(acc.astype(f32)).astype(f32)

# ---------- gated delta-rule recurrence (per v-head) ----------
rq    = rng.standard_normal((seq, vheads, hd)).astype(f32)
rk    = rng.standard_normal((seq, vheads, hd)).astype(f32)
rv    = rng.standard_normal((seq, vheads, hd)).astype(f32)
# realistic gates: beta in (0,1), g = -softplus(.) <= 0
rbeta = (1.0 / (1.0 + np.exp(-rng.standard_normal((seq, vheads))))).astype(f32)
rg    = (-np.log1p(np.exp(rng.standard_normal((seq, vheads))))).astype(f32)
S0    = (0.01 * rng.standard_normal((vheads, hd, hd))).astype(f32)   # [H][K][V]

def l2norm(x, eps=1e-6):
    return (x * (1.0 / np.sqrt((x * x).sum() + eps))).astype(f32)

out = np.zeros((seq, vheads, hd), dtype=f32)
S = S0.copy()
scale = f32(hd ** -0.5)
for t in range(seq):
    for h in range(vheads):
        q = l2norm(rq[t, h]) * scale
        k = l2norm(rk[t, h])
        v = rv[t, h]
        gt = f32(np.exp(rg[t, h]))
        bt = f32(rbeta[t, h])
        Sh = S[h]
        Sh *= gt                                   # (a) decay
        kv_mem = (Sh * k[:, None]).sum(axis=0)     # (b) read with key  -> [V]
        delta = ((v - kv_mem) * bt).astype(f32)    # (c) delta
        Sh += (k[:, None] * delta[None, :])        # (d) write outer
        out[t, h] = (Sh * q[:, None]).sum(axis=0)  # (e) read with query -> [V]
Sfin = S

# ---------- gated RMSNorm (kernel 3 ref): rmsnorm_per_head(out) * znorm * silu(z) ----------
znorm = rng.standard_normal(hd).astype(f32)
zg    = rng.standard_normal((seq, vheads, hd)).astype(f32)
out_gated = np.zeros((seq, vheads, hd), dtype=f32)
for t in range(seq):
    for h in range(vheads):
        o = out[t, h]
        inv = f32(1.0 / np.sqrt((o * o).mean() + 1e-6))
        out_gated[t, h] = (o * inv * znorm * silu(zg[t, h])).astype(f32)

# ---------- dump ----------
def w(fh, a): fh.write(np.ascontiguousarray(a, dtype=f32).tobytes())
with open(OUT, "wb") as fh:
    fh.write(struct.pack("<5i", seq, vheads, hd, conv_dim, kernel))
    w(fh, conv_w); w(fh, conv_in); w(fh, conv_out)
    w(fh, rq); w(fh, rk); w(fh, rv); w(fh, rbeta); w(fh, rg)
    w(fh, S0); w(fh, out); w(fh, Sfin)
    w(fh, znorm); w(fh, zg); w(fh, out_gated)
print("wrote %s  (seq=%d vheads=%d hd=%d conv_dim=%d k=%d)" % (OUT, seq, vheads, hd, conv_dim, kernel))
print("ref conv_out[0,:4]=", conv_out[0, :4])
print("ref out[seq-1,0,:4]=", out[seq - 1, 0, :4])
