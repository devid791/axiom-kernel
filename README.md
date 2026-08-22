# Axiom Kernel

Axiom is a native C++17/CUDA inference kernel for quantized and mixed-precision
large language model execution. The public tree is a sanitized source release:
it contains kernel code, public headers, tests, performance probes and design
documentation; it does not contain model weights, private service daemons,
deployment files, credentials, internal endpoints, or production machines.

Axiom is an owned GPU inference kernel and a multi-family, OpenAI-compatible inference daemon, written and owned by Davide Zenati. It is not vLLM, llama.cpp, GGML, or a fork of any of them. This public repository is the sanitized kernel source release; the private serving daemon and its adapters are intentionally maintained outside this repository.

The runtime is multi-family by design. The stable C ABI and the family-agnostic
step-engine ABI are separate from model-specific math. The public snapshot
includes the native Qwen3.8 path as the primary complete integration, reusable
NVFP4/FP8 primitives, the generic engine contracts and a standalone Gemma-4
reference adapter. Private/legacy service adapters are intentionally outside
this public kernel snapshot.

## Included capabilities

| Area | Public implementation |
| --- | --- |
| Native execution | C++17 host ABI, CUDA device runtime, Blackwell-oriented kernels |
| Quantization | NVIDIA NVFP4/E2M1, FP8 E4M3, BF16 and fused MoE primitives |
| Qwen3.8 | Native 27B decoder, GDN, attention, MLP banks, NVFP4/FP8 LM head |
| Decode path | DSpark/M8 temporal speculative path and device-side verification ABI |
| Attention | FlashInfer headers vendored under their upstream license |
| KV | Paged KV, persistent session store, hot device pages and cold NVMe pages |
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
```

`MEDIA=0` (the default) keeps the optional FFmpeg object out of the core build;
use `make MEDIA=1` only on a machine where the decoder toolchain is verified.

## Performance record

The 2026-08-22 kernel update sustained a median **384.158 decoded tokens/s**
across four consecutive requests to a separate private production fixture on
one RTX 5090. The fixed test used a 1,002-token prompt, generated 256 tokens
through 33 speculative cycles and produced the same output SHA-256 on every
run. Individual decode rates were 382.972, 384.249, 384.256 and 384.067
tokens/s; visible transport rates were 384.474, 385.756, 385.763 and 385.574
tokens/s.

This is a measured result for one exact fixture, not a universal guarantee for
this source tree, another GPU or another checkpoint. The private HTTP daemon,
weights and production configuration are not published here. See
[the complete methodology](docs/qwen38/PERFORMANCE.md) and reproduce the result
with recorded artifact hashes, toolchain, CUDA architecture, sampling policy
and benchmark protocol.

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
