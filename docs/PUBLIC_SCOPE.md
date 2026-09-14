# Public release scope

## September 14 reasoning and compaction update

Selectively imported the provider/test delta through
`543e93667637a52a743be994b2c3d87a139fd250` without importing private history.
Public build targets and documentation are maintained separately. Loopback
defaults, request serialization, MIT licensing and all existing publication
exclusions remain unchanged. Aggregate QA and independent attribution findings
are documented in [REASONING_AND_COMPACTION.md](REASONING_AND_COMPACTION.md);
private transcripts, operational topology and checkpoint assets are excluded.

## September 13 conversation reliability update

Selectively imported continuity, progress, image-replay and mixed-attention
fixes from `fffdaf0588b05589fdbeb5bdf517031d3f68f8b6`, plus regression tests and
loopback-default, explicitly invoked live QA runners. The public build retains
the exclusions and transport protections below. Only aggregate qualification
evidence is published; raw conversations and internal deployment reports stay
private. See [CONVERSATION_RELIABILITY.md](CONVERSATION_RELIABILITY.md).

## September 11 Codex integration update

The follow-up selectively imports source through `711318c` onto the existing
public history. It adds the native Qwen3.8 HTTP provider, `/codex/v1` bridge,
known-token/paged prefill and dynamic vision memory management. This explicitly
extends the historical v0.5 serving exclusion below for this audited provider
only. See [CODEX_APP_SERVER.md](CODEX_APP_SERVER.md).

Private credentials, network identities, deployment scripts, raw operational
reports, activation-control implementations and DFlash2 are still excluded.
The public provider defaults to loopback, preserves request serialization and
does not provide authentication or TLS. Host/build checks are not model-backed
or production qualification of the independently sanitized executable.

## Historical v0.5 boundary

The v0.5 update includes the subsequent production-reference kernel changes
through `d29b033c6c1593ed31ae6c4e15431a1d11ca4ed1`, plus the separate native
Flash-Next backend at `a4aeda07b332471cb8aed1320b231c03521f01b2`.
Public changes are merged onto the sanitized tree, not copied with internal
history. HTTP daemons, deployment profiles, credentials and activation-control
code remain outside the public boundary. The later unqualified HTTP width2,
persistence-overlap and adaptive-proposal candidate is not included as a
production feature. Public build qualification does not imply a new GPU
inference performance result.

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
- DSpark layout ABI v3, device-owned graph termination, exact committed-token
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
