# Architecture and model backends

Axiom is the inference kernel. A model name identifies a backend registered
behind Axiom's stable interfaces; it does not identify the project itself.

## Layer boundary

1. **Core ABI** — `include/axiom/axiom.h`, `axiom_engine.h` and
   `axiom_resident.h` define model loading, tensor access, execution and
   scheduling contracts without requiring a Qwen-specific caller.
2. **Reusable GPU primitives** — the generic CUDA translation units implement
   quantization, matrix operations, attention, RoPE, MoE and KV operations that
   model backends compose according to their architecture.
3. **Family backends** — architecture-specific files own topology, tensor-name
   mapping, recurrent state, token flow and parity gates. Their names must stay
   explicit: a Qwen result must not silently become a claim about another
   family.
4. **Serving applications** — OpenAI/Anthropic protocol adapters, tools,
   authentication and deployment policy are separate from this sanitized
   kernel repository.

## Public backend maturity

| Family or integration | Evidence in this repository | Public status |
| --- | --- | --- |
| Qwen3.8 27B NVFP4 | Native decoder, GDN, attention, MLP banks, vision, DSpark/M8, paged persistent KV and model-backed production evidence | Complete reference backend |
| Gemma-4 | Standalone reference runner plus family-specific attention, RoPE and normalization primitives | Reference/onboarding implementation; not presented as production parity |
| DeepSeek V4 Flash | NVFP4/FP8/MoE primitives, converters and validation probes | Kernel integration components; the DFlash2 executor is intentionally excluded |
| Llama-like dense GGUF and other registered families | Family identifiers, generic resident/engine contracts and shared primitives | ABI/onboarding surface; no complete production backend claimed here |

“Multi-family” therefore describes the architecture and reusable execution
surface. “Complete reference backend” describes the stronger, separately
verified state currently published for Qwen3.8.

## Naming policy

- `axiom_*` denotes family-neutral core code or reusable primitives.
- `axiom_<family>_*` denotes a model-family backend, probe or gate.
- Performance and parity documents remain under `docs/<family>/` because their
  fixtures, tokenizer, topology and checkpoint assumptions are not portable.

This explicit namespace is intentional. Replacing every `qwen38` symbol with
`axiom` would blur the backend boundary and could break the public ABI without
making the underlying implementation more general.
