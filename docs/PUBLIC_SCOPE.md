# Public release scope

This repository is a fresh public source snapshot derived from the internal
source commit `7945f8bf595039c5e3be50896b16bdf86ca3fa1c`. The internal history,
production branch names and deployment evidence are not copied into the public
Git history.

Included:

- Axiom C++17/CUDA runtime and public ABI;
- native Qwen3.8 decoder, NVFP4/FP8/BF16 primitives, DSpark/M8 execution,
  vision, optional media decoding and paged/persistent KV components;
- generic engine contracts, reasoning profiles and swarm scheduler;
- public tests, probes, FlashInfer source and its license/notice;
- build and reproducibility documentation.

Excluded deliberately:

- model weights, tokenizer files, checkpoint hashes and private release
  manifests;
- HTTP daemons, OpenAI/Anthropic adapters, operator consoles, systemd units,
  deployment scripts and internal network configuration;
- private submodules, production logs, benchmark result archives and local
  filesystem assumptions;
- activation-steering implementation, steering packs and legacy adapters that
  depended on them;
- DFlash2-specific executor code. The public speculative path is DSpark/M8.

The exclusion of adapters is a publication boundary, not a claim that the
internal runtime supports only one model family. The public ABI is designed for
multiple families, while each adapter must be independently audited and
licensed before it is added to a public build.
