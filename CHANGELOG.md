# Changelog

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
