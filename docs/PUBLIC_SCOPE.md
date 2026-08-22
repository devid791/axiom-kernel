# Public release scope

This repository is a fresh public source snapshot. Its Qwen3.8 performance
update is derived from the reviewed kernel commit
`41403d5fed91f3481d92ae111e83fa75f544dcbb`; the initial public snapshot was
derived from `7945f8bf595039c5e3be50896b16bdf86ca3fa1c`. Internal history,
production branch names and private deployment artifacts are not copied into
the public Git history.

Included:

- Axiom C++17/CUDA runtime and public ABI;
- native Qwen3.8 decoder, NVFP4/FP8/BF16 primitives, DSpark/M8 execution,
  vision, optional media decoding and paged/persistent KV components;
- generic engine contracts, reasoning profiles and swarm scheduler;
- public tests, probes, FlashInfer source and its license/notice;
- build and reproducibility documentation.

The performance table publishes aggregate measurements and an output digest
from the production fixture. It does not publish network identities, local
paths, service configuration, model artifacts or credentials.

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
