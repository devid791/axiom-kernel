# Qwen3.8 performance methodology

Performance claims must separate prompt prefill, time to first token, decode
loop time, visible transport time and post-generation persistence. At minimum,
record prompt/output token counts, sampling mode, speculative cycles, accepted
draft tokens, GPU architecture, driver/CUDA/compiler versions and hashes for
the executable, shared library, checkpoints and output.

## Current record: RTX 5090, DSpark — 2026-09-05

The selected R4 production profile completed three post-restart requests
through the real private OpenAI-compatible HTTP API using the **normal
Qwen3.8-27B NVFP4 checkpoint**, native DSpark speculative generation and one
RTX 5090 on a dedicated physical host. The fixture used a 1,002-token prompt,
256 completion tokens and greedy/no-thinking generation.

| Run | Decode tokens/s |
| ---: | ---: |
| 1 | 409.111724 |
| 2 | 409.410438 |
| 3 | 409.323798 |
| **Median** | **409.323798** |

The decode numerator is **255 graph-emitted tokens**, excluding the first
anchor token. The denominator is native decode time, not total HTTP time:
prefill, time to first token, network delivery and session persistence are
separate measurements. All responses matched the golden text SHA-256
`b0d7b6f4bad084c8bdba57559fccbbb3b4acef5d085274123c829aa0eb793b5b`.
The saved strict agentic gate also passed SSE, tool continuation, speculative
path and exact-prefix session restore checks.

This supersedes the historical approximately 384 tokens/s record for this
deployment. It does not imply continuous end-to-end 409 tokens/s on arbitrary
prompts, the uncensored checkpoint or Flash-Next. The selected production
profile, rather than an isolated microbenchmark or faster experimental peak,
is the record published here. See [the sanitized record](RECORD_409.json).

Publication on September 7 adds source and host/build verification, not a new
GPU benchmark. The historical private build included a source snapshot beyond
its recorded commit; its commit alone is not a byte-identical build recipe.
Public artifact hashes describe the newly compiled sanitized source separately.

## Historical RTX 5090 fixture — 2026-08-22

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

The August fixture did not establish more than 400 tokens/s; the September
record above subsequently did. Neither proves performance on a different GPU,
model artifact, context length or serving adapter.

## Persistent-session lifecycle confirmation — 2026-08-22

After adding bounded session GC and lossless stateful resume, the same private
production fixture completed four more byte-identical HTTP runs:

| Run | Decode tokens/s | Visible tokens/s |
| ---: | ---: | ---: |
| 1 | 383.6865 | 385.191 |
| 2 | 383.1897 | 384.692 |
| 3 | 382.5082 | 384.008 |
| 4 | 383.1387 | 384.641 |
| **Median** | **383.1642** | **384.6665** |

The output SHA-256 remained
`b0d7b6f4bad084c8bdba57559fccbbb3b4acef5d085274123c829aa0eb793b5b`.
The 0.26% difference from the preceding 384.1583 tokens/s median is within
normal run variation and does not indicate a material decode regression. The
lifecycle work runs outside graph replay and persistence quiescence is checked
separately from decode throughput.

## Resident graph correctness update — 2026-08-24

Version 0.4 adds the kernel substrate required to reuse one captured DSpark/M8
executor safely across requests:

- layout ABI revision 2 with size-checked configuration, state and history;
- device-owned stop-token state, including terminal no-op graph cycles;
- an exact committed-token count in the compact device history;
- explicit target-model device-session ownership renewal per request;
- a startup-selected 2,048–8,192-token speculative hot window with matching
  FlashInfer planning and workspace allocation.

The model-backed qualification recorded temporal, paged-KV, vision and
8,192-token speculative-graph runtime gates as passing. Serving-contract,
multi-session, cancellation and swarm gates also passed in the private
application layer, which remains outside this repository.

That v0.4 publication made no new speed claim. Two later qualification attempts on
the same kernel code observed median visible throughput of approximately
374.154 and 362.949 tokens/s; the latter missed the release policy threshold of
370 tokens/s. Correctness evidence therefore ships independently from the
historical v0.3 performance record above. Do not present 384 tokens/s—or any
single peak—as continuous throughput without reproducing the exact fixture.
