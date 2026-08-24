# Changelog

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
