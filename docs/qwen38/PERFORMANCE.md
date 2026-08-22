# Qwen3.8 performance methodology

Performance claims must separate prompt prefill, time to first token, decode
loop time, visible transport time and post-generation persistence. At minimum,
record prompt/output token counts, sampling mode, speculative cycles, accepted
draft tokens, GPU architecture, driver/CUDA/compiler versions and hashes for
the executable, shared library, checkpoints and output.

## Verified RTX 5090 fixture — 2026-08-22

A separate private production fixture ran four consecutive requests through
the real OpenAI-compatible HTTP path. Each request used:

- one RTX 5090;
- a fixed 1,002-token prompt;
- greedy/no-thinking generation;
- 256 generated tokens;
- 33 speculative cycles;
- the same kernel, model artifacts and configuration;
- 223 accepted proposals out of 224 eligible proposals;
- output SHA-256
  `b0d7b6f4bad084c8bdba57559fccbbb3b4acef5d085274123c829aa0eb793b5b`.

| Run | Decode tokens/s | Visible tokens/s | HTTP wall time |
| ---: | ---: | ---: | ---: |
| 1 | 382.9717 | 384.4735 | 4.232 s |
| 2 | 384.2492 | 385.7560 | 3.901 s |
| 3 | 384.2563 | 385.7632 | 4.945 s |
| 4 | 384.0674 | 385.5736 | 4.874 s |
| **Median** | **384.1583** | **385.6648** | — |

The preceding public record was approximately 304 internal decode tokens/s
(298 visible tokens/s) on the same GPU class and prompt/output sizes. The new
record therefore represents about a 26% increase in internal decode rate.
Because the old and new records were produced by private production fixtures,
this comparison is evidence for that deployment, not a portable source-only
guarantee.

The standalone exact graph gate also produced 382.374–383.558 tokens/s over
four confirmations with stable token IDs. It is useful for isolating CUDA graph
replay from HTTP, prefill and persistence overhead.

## Changes in this release

The public kernel update includes the same exact-path improvements used by the
verified fixture:

- validated per-shape cuBLASLt candidate selection for temporal FP8/NVFP4
  matrix multiplication;
- 16-lane cooperative NVFP4 activation quantization;
- shared normalized FP8 input for Q/K/V projections;
- parallel independent projection streams;
- exact split-K temporal attention and FlashInfer split-KV support;
- hierarchical target and DSpark vocabulary top-1 reduction;
- exact eight-token prefill blocks;
- device-authoritative graph commit without replaying the complete token
  history;
- explicit host-history installation after a synchronized device commit.

All optimized paths retain a direct rollback control:

| Variable | Default | Rollback value |
| --- | --- | --- |
| `AXIOM_QWEN38_MATMUL_AUTOTUNE` | off at library level | `0` |
| `AXIOM_QWEN38_FP8_AUTOTUNE` | inherits global | `0` |
| `AXIOM_QWEN38_NVFP4_AUTOTUNE` | inherits global | `0` |
| `AXIOM_QWEN38_NVFP4_WARP_QUANT` | on | `0` |
| `AXIOM_QWEN38_DSPARK_FUSED_TOP1` | on | `0` |
| `AXIOM_QWEN38_TARGET_HIERARCHICAL_TOP1` | on | `0` |
| `AXIOM_QWEN38_DSPARK_ATTENTION_SPLIT_K` | `4` | `1` |
| `AXIOM_QWEN38_FLASHINFER_SPLIT_KV` | on | `0` |
| `AXIOM_QWEN38_GDN_TEMPORAL_VALUE_TILE` | `8` | `16`, `32` or `64` |
| `AXIOM_QWEN38_ATTENTION_SHARED_FP8_INPUT` | on | `0` |
| `AXIOM_QWEN38_PARALLEL_PROJECTIONS` | on | `0` |

Autotuning occurs during model initialization, never during graph replay. A
candidate is eligible only after exact validation against deterministic input
patterns. Deployments that enable autotuning should record the selected
algorithms and keep the rollback values available.

## Reproduction boundary

The public repository intentionally omits the private HTTP daemon, production
logs, model weights and checkpoint hashes. Rebuild the public kernel, supply a
legally obtained compatible checkpoint, and record the complete fixture
described in `docs/BUILD_AND_VERIFY.md`. Do not compare only a peak sample:
report repeated runs, output identity and acceptance counters.

The verified result does **not** claim more than 400 tokens/s and does not prove
performance on a different GPU, model artifact, context length or serving
adapter.
