# Changelog

## Reasoning and compaction source update — 2026-09-14

- Preserve the full current user/tool exchange when compacting old history;
  retain pinned instructions and reject an oversized current task explicitly.
- Keep archived tool calls inert and use the current effective tool catalog,
  including discovered tools, as the authority for new calls.
- Match native text/multimodal ChatML framing, effort conditioning, assistant
  history and grouped tool results; version durable prompt compatibility.
- Enforce visible-output caps before committing tokens, keep hidden planning
  out of tool execution, and forward the visible first-pass prefix through SSE.
- Remove only the native two-newline reasoning separator; preserve actual
  answer whitespace and literal closing tags in reasoning-disabled mode.
- Add reproducible provider, template, streaming and compaction regressions.
  Publish [QA results and remaining limitations](docs/REASONING_AND_COMPACTION.md),
  including independently observed model-answer errors. No universal GO claim.
- Source only, MIT: no private history, credentials, model assets, new app
  packages, production restart or new inference campaign for publication.

## Conversation reliability source update — 2026-09-13

- Reuse byte/control-equivalent committed text prefixes and exact Codex
  parallel-tool history projections without ignoring authoritative edits.
- Add opt-in mixed resident/cold split-K attention with checked ring identities;
  preserve complete history and existing speculation/precision boundaries.
- Report real active-request progress, incomplete output budgets and empty
  response failures; preserve explicit deadlines without an invented default.
- Fix image-history follow-up normalization of assistant `output_text`.
- Include CPU/GPU regression gates and explicit live QA runners. Publish
  aggregate evidence and limitations in [Conversation reliability](docs/CONVERSATION_RELIABILITY.md),
  including the still-disabled multimodal persistent KV reuse.
- Source-only publication: no new app packages, weights or production deployment.

## Codex provider source update — 2026-09-11

- Added the native `/codex/v1` provider and its schema/name/custom-tool boundary,
  discovery, original call/result identity and incremental Responses stream.
- Added original-schema output validation, explicit unsupported-tool errors,
  deadline/cancellation handling and long-request stream progress.
- Added known-token scalar/paged prefill without extending speculative decode
  boundaries, and checked dynamic vision scratch-memory growth.
- Preserved the already published DSpark/M8 update and historical performance
  record. No new model-backed performance or public-build inference claim.
- Included native host-side contract tests; removed private activation controls
  from the public provider and changed its default listener to loopback only.
- No production deployment, installed-app replacement, private history or
  credentials are included in source publication.

## v0.5.0-public — 2026-09-07

- Published the saved September 5 RTX 5090 record: **409.324 decoded tokens/s
  median** across three post-restart HTTP requests, normal 27B NVFP4 + DSpark,
  1,002 prompt / 256 completion tokens, golden-output identical. This is a
  decode measurement, not end-to-end throughput or an uncensored-model claim.
- Added the native Qwen3.8 Flash-Next NVFP4 backend (`qwen4_exp`) in a separate
  opt-in library: checkpoint admission, QSA/GDN, multi-hyperconnection, PLE,
  routed/shared experts and bounded GPU/RAM/NVMe expert residency.
- Updated 27B DSpark feature capture to post-decoder-block hidden states and
  published stronger scalar/temporal KV, logits and tap validation.
- Updated the size-checked DSpark device ABI to revision 3, including history
  and target verification-policy identity; older symbol layouts reject safely.
- Added exact attention overlap kernels, register-state GDN, BF16 paired
  projections and expanded matmul tuning candidates. Native reference CUDA
  objects compile without global fast-math to preserve the exact math contract.
- Added transactional KV recovery and SHA-256 protected version-2 session
  manifests, while preserving legacy session loading and cleanup semantics.
- Added checkpoint/speculator fingerprint validation and native MTP library
  primitives. MTP availability is distinct from accelerated qualification for
  an individual checkpoint; DSpark remains the production reference path.
- Added build/host checks for both backends. No production restart, new
  inference benchmark, public daemon or model weights are part of this release.
- Excluded the unqualified HTTP width2/adaptive-proposal candidate. No new
  universal throughput, vision HTTP or full-context claim is made.

## v0.4.0-public — 2026-08-24

- Published DSpark device-layout ABI revision 2 with size-checked entry points.
  Legacy revision-1 binary symbols now reject incompatible buffers before
  dereferencing or clearing them.
- Added device-owned stop-token and terminal-cycle state. CUDA graph cycles
  already queued after EOS become deterministic no-ops instead of advancing
  recurrent or KV state.
- Added an explicit committed-token authority field to the compact device
  history record and renewed target-model device ownership for every resident
  executor request.
- Made the DSpark/M8 hot speculative context configurable from 2,048 through
  8,192 tokens in 256-token pages, with matching FlashInfer planning and
  workspace sizing.
- Added a public DSpark ABI gate covering layout revision, undersized-buffer
  rejection and fail-closed legacy symbols.
- Retained the v0.3 measured performance record. This release publishes
  correctness and resident-executor hardening; it does not claim a new
  throughput record.

## v0.3.1-public — 2026-08-22

- Clarified that Axiom is the model-agnostic kernel and Qwen3.8 is its current
  complete production reference backend.
- Documented the boundary between the generic ABI, reusable CUDA primitives,
  model-family backends and the separately maintained serving layer.
- Added an evidence-based model-backend maturity matrix. No runtime code or
  benchmark result changed in this documentation release.

## v0.3.0-public — 2026-08-22

- Added configurable TTL/LRU retention for complete persistent-session
  namespaces, with explicit protection for active sessions.
- Added atomic GC tombstones and idempotent crash recovery. Cleanup uses
  directory-relative file operations, rejects symlinks and hard links, and
  fails closed on malformed namespaces or scan-budget exhaustion.
- Added exact stateful-resume prompt construction for clients that can resend
  only normalized visible history while the durable state contains hidden
  thinking or tool-call tokens.
- Added deterministic lifecycle, race, descriptor-leak, malformed-artifact,
  hard-link and resume-watermark tests.
- Verified the production path at a 383.164 tokens/s median across four
  byte-identical RTX 5090 runs after enabling the lifecycle implementation.
- Preserved the public boundary: the serving daemon, internal endpoints,
  deployment configuration and model artifacts remain private.

## v0.2.0-public — 2026-08-22

- Added the production-verified exact Qwen3.8 speculative graph
  optimizations for FP8/NVFP4 matrix multiplication, attention, GDN,
  vocabulary reduction and device-side commit.
- Added direct rollback controls for every optimized kernel family.
- Added an explicit committed-history bridge for restored device-authoritative
  state.
- Preserved the public release boundary: no serving daemon, credentials,
  internal endpoints, model weights, activation steering or DFlash2 executor.
- Recorded the four-run RTX 5090 production fixture with a median decode rate
  of 384.158 tokens/s and byte-identical output.

## v0.1.1-public — 2026-08-20

- Clarified project ownership and independence.
- Added community standards and contribution templates.

## v0.1.0-public — 2026-08-20

- Initial sanitized Axiom Kernel source release.
