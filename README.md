# Axiom Kernel

**Model-agnostic GPU inference core with native family backends.**

> **Axiom is the kernel. Qwen3.8 is the current production reference backend,
> not the name or architectural limit of the project.** File and symbol names
> prefixed with `qwen38` are intentionally scoped to that backend.

Axiom is a native C++17/CUDA inference kernel for quantized and mixed-precision
large language model execution. The public tree is a sanitized source release:
it contains kernel code, public headers, tests, performance probes and design
documentation; it does not contain model weights, private service daemons,
deployment files, credentials, internal endpoints, or production machines.

Axiom is an owned GPU inference kernel and a multi-family, OpenAI-compatible inference daemon, written and owned by Davide Zenati. It is not vLLM, llama.cpp, GGML, or a fork of any of them. This public repository is the sanitized kernel source release; the private serving daemon and its adapters are intentionally maintained outside this repository.

The runtime is multi-family by design. The stable C ABI and family-neutral
step-engine ABI are separate from model-specific math. The public snapshot
includes the native Qwen3.8 path as its first complete reference integration,
reusable NVFP4/FP8/BF16 primitives, generic engine contracts and additional
family onboarding work. Private service adapters remain outside this public
kernel snapshot.

## Architecture at a glance

| Layer | Responsibility | Public examples |
| --- | --- | --- |
| Axiom core | Stable C ABI, tensor/model I/O, scheduling and family-neutral execution contracts | `axiom.h`, `axiom_engine.h`, `axiom_runtime.cpp` |
| Device primitives | Quantized linear algebra, attention, RoPE, MoE and paged KV building blocks | `axiom_cuda*.cu` |
| Model backends | Architecture-specific topology, weights, token flow and verification | `qwen38_*` reference backend; Gemma and DeepSeek integration components |
| Serving layer | HTTP protocols, authentication, tools and deployment policy | Deliberately outside this public repository |

See [Architecture and model backends](docs/ARCHITECTURE.md) for the exact
boundary and maturity of each public integration.

## Included capabilities

| Area | Public implementation |
| --- | --- |
| Native execution | C++17 host ABI, CUDA device runtime, Blackwell-oriented kernels |
| Quantization | NVIDIA NVFP4/E2M1, FP8 E4M3, BF16 and fused MoE primitives |
| Reference backend | Qwen3.8 native 27B decoder, GDN, attention, MLP banks and NVFP4/FP8 LM head |
| Decode path | Resident DSpark/M8 temporal speculative path, device-side verification and size-checked layout ABI v2 |
| Attention | FlashInfer headers vendored under their upstream license |
| KV | Paged KV, persistent sessions, hot GPU pages, cold NVMe pages and bounded crash-safe lifecycle GC |
| Context | YaRN-compatible long-context plumbing and deterministic compaction hooks |
| Vision | Native image preprocessing/vision tower; unsupported video containers return an explicit error |
| Agents | C++ swarm scheduler and generic continuous-batching engine contracts |
| Reasoning | Seven explicit profiles: ultra-fast, minimal, low, medium, high, xhigh, max |

The public source does not include a serving API. OpenAI/Anthropic adapters,
authentication, tool policy and network exposure belong in a separately audited
application layer.

## Deliberate security scope

Activation-steering code, steering packs and their legacy provider adapters were
deliberately removed from this release. This reduces the exposed model-control
surface and avoids publishing a direct activation-manipulation mechanism. The
choice is defense-in-depth, not a claim that a model is incapable of unsafe
output: deployment policy, tool authorization, filesystem permissions and
network controls remain application responsibilities.

DFlash2-specific executor code is also not part of this snapshot. The public
Qwen path is the DSpark/M8 path used by the current Axiom kernel integration.

## Build prerequisites

- Linux or another CUDA-capable platform
- CUDA toolkit with `nvcc` and a C++17 compiler
- CUDA architecture selected explicitly, for example `CUDA_ARCH=sm_120`
- Optional image libraries for the vision preprocessor: JPEG, PNG and WebP
- Optional FFmpeg development libraries for the video-container decoder

No weights or tokenizer assets are included. A model-specific integration must
provide them separately and must verify their format and checksum before use.

Examples:

```sh
make CUDA_ARCH=sm_120
make check
make public-scan
make CUDA_ARCH=sm_120 qwen38-dspark-abi-gate
```

`MEDIA=0` (the default) keeps the optional FFmpeg object out of the core build;
use `make MEDIA=1` only on a machine where the decoder toolchain is verified.

## Performance record

The lifecycle update sustained a median **383.164 decoded tokens/s** across
four consecutive requests to a separate private production fixture on one RTX
5090. The fixed test used a 1,002-token prompt, generated 256 tokens through 32
speculative cycles and produced the same output SHA-256 on every run. Individual
decode rates were 383.686, 383.190, 382.508 and 383.139 tokens/s. This is within
0.26% of the preceding 384.158 tokens/s median and is not a material regression.

This is a measured result for one exact fixture, not a universal guarantee for
this source tree, another GPU or another checkpoint. The private HTTP daemon,
weights and production configuration are not published here. See
[the complete methodology](docs/qwen38/PERFORMANCE.md) and reproduce the result
with recorded artifact hashes, toolchain, CUDA architecture, sampling policy
and benchmark protocol.

Version 0.4 publishes the resident-graph correctness substrate used by the
later private serving integration: a size-checked DSpark ABI, device-owned
terminal state, exact committed-token accounting, per-request resident-session
ownership and a configurable 2K–8K speculative hot window. It deliberately
does not replace the measured v0.3 performance record with an unverified peak.

## Community

Please read [CONTRIBUTING](CONTRIBUTING.md), [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md)
and [SECURITY](SECURITY.md) before opening an issue or pull request. Use the
repository templates so reports include reproducible environment details
without exposing private data.

## Licensing

Axiom-owned source is released under the MIT License. Vendored dependencies
retain their own licenses; in particular, `third_party/flashinfer` includes its
Apache-2.0 license and notice. See [NOTICE](NOTICE) and the relevant files
before redistributing a binary.

See [PUBLIC_SCOPE.md](docs/PUBLIC_SCOPE.md) for the release boundary and
[BUILD_AND_VERIFY.md](docs/BUILD_AND_VERIFY.md) for the reproducibility gates.
