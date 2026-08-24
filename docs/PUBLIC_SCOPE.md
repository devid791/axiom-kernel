# Public release scope

This repository is a sanitized public source snapshot. The resident-graph ABI
update is derived from reviewed source tree
`aa4aa9ff708d3476d3959945e70a8cac1a50647b`, with qualification evidence
recorded by `2bd1c8cb0eaa038859044c727be1e545f45fba2d`. The persistent-session
lifecycle update was derived from
`3ae57422ff8fcb7ebcf8f12074a2c16d921eed53`, and the preceding performance
update from `41403d5fed91f3481d92ae111e83fa75f544dcbb`. Internal history,
production branch names and private deployment artifacts are not copied into
the public Git history.

Included:

- Axiom C++17/CUDA runtime and public ABI;
- family-neutral engine contracts and reusable GPU primitives, documented
  independently from any one model backend;
- native Qwen3.8 decoder, NVFP4/FP8/BF16 primitives, DSpark/M8 execution,
  vision, optional media decoding and paged/persistent KV components;
- DSpark layout ABI v2, device-owned graph termination, exact committed-token
  history and resident target-session ownership renewal;
- generic engine contracts, reasoning profiles and swarm scheduler;
- crash-safe TTL/LRU persistent-session cleanup and stateful-resume helpers;
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

The `qwen38` prefix in source files, public symbols and the `docs/qwen38`
directory identifies the current reference backend. It does not rename Axiom
or make the core ABI Qwen-specific. Backend-specific evidence remains namespaced
so that results from one architecture are not accidentally presented as
portable to every model family.
