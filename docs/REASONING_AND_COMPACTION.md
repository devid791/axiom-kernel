# Reasoning, streaming and compaction reliability

## September 14, 2026 source publication

This update selectively carries the reviewed provider/test changes through
source commit `543e93667637a52a743be994b2c3d87a139fd250` onto public commit
`bf597857ba6cef8ee680cda5f128af28bc738068`. It does not import private Git
history, deployment configuration, raw conversations, credentials or model
assets. Axiom-owned code remains MIT. No app package or release tag is created.

## Corrections

- Compaction preserves the complete latest user exchange, including its
  assistant/tool replies. The recent-history allowance applies to older
  exchanges. Historical tokenization is bounded by the loaded model ceiling;
  the final prompt must fit the selected context. An oversized current task
  rejects explicitly instead of silently losing its content.
- Historical tool calls/results are escaped observations, not fresh executable
  examples. Only the effective current catalog, including discovered tools,
  authorizes new calls. Pinned instructions and current tool turns remain intact.
- Native text and multimodal prompts now use the correct turn newlines,
  reasoning prefill, system/conditioning order, assistant-history wrappers and
  grouped tool replies. Durable template versioning prevents old malformed KV
  prompts from being extended through the stateful-tail fallback.
- Visible-output budgets are enforced before scalar/speculative token commits,
  avoiding truncation that would leave native state ahead of returned history.
  Hidden planning cannot authorize a tool call. The selected seven effort IDs,
  reasoning caps and sampling defaults are unchanged.
- Streaming forwards any already-visible first-pass prefix before the second
  pass. Only the protocol's exact two-newline separator is removed. Actual
  whitespace and literal closing tags in Off-mode output are preserved.

Public loopback defaults, request serialization and the existing source
exclusions remain intact. No production configuration or feature was changed
for publication. See the [profile contract](qwen38/REASONING_PROFILES.md).

## Prior source-deployment qualification

These are aggregate results from the qualified source-derived deployment,
not new model inference on the independently sanitized public executable.

The final reasoning campaign comprised 51 matrix requests, one exact-copy
phase probe and two image requests: **54/54 backend checks pass**. It includes
14 real tool-call/result requests and eight same-session effort transitions.
Backend correctness is distinct from answer correctness:

| Normal objective fixtures | Passed / tested |
| --- | ---: |
| Ultra-fast / Off | 0 / 3 |
| Minimal | 2 / 3 |
| Low | 1 / 3 |
| Medium | 2 / 3 |
| High | 3 / 3 |
| XHigh | 3 / 3 |
| Max | 3 / 3 |

The normal score remains **14/21**, with three incorrect Off answers and four
enabled answers whose correct JSON values were inside forbidden Markdown
fences. Three separate target-only Off controls reproduced the original wrong
bytes without speculative decoding; these are controls, not three new unique
defects. All failures remain recorded. These single sampled runs do not prove
a stable profile ranking or a universal accuracy rate.

The dedicated 400-line copy probe produced exactly 4,299 expected bytes. Its
957-token already-visible first-pass prefix survived the handoff, and the
next pass prefilled only one new token. A different generic probe did not
exercise that strict handoff condition; its generic backend pass is not used
as a substitute. The two real image answers were correct. An initial checker
incorrectly compared total generated tokens with a visible-only cap; offline
rescoring used the same saved requests, preserving the original report. The
enabled image had 81 reasoning + 7 visible tokens against a visible cap of 16.
No extra image inference was used to hide that checker error.

Earlier compaction qualification reproduced the original unavailable-tool
history with unchanged instructions and output allowance. Current marker
retention, dependent cache continuation, discovered-tool execution and explicit
no-tools behavior passed. A distinct 96-token recall control remained incomplete
and its dependent handler was not run; a separately labelled 256-token action
control passed. These do not imply that the original incomplete control passed
or that arbitrary historical summarization is lossless.

## Independent answer attribution

A subsequent diagnostic checked the same quantized checkpoint outside Axiom
using CPU-only PyTorch 2.8.0, the official Transformers 5.12.1 Qwen3_5 model
implementation and NVIDIA ModelOpt 0.46.1 weight/activation emulation. Frozen
prompts, token identities and independently computed answer oracles were used.

- JavaScript trace: every one of 25 answer positions selected the same wrong
  next-token argmax under the verified causal prefix. This is a causal token
  comparison, not a separate free-generation run; stop tokens were excluded.
- Two other tasks had non-identical causal predictions, so each was also
  generated freely once from its original prompt, without supplying an answer.
  Both remained wrong: a pack combination summed to 75 instead of 83, and a
  filtered-record total was 32 instead of 40. Their incorrect values differ
  from Axiom's and must not be described as exact output parity.

Wrong-answer behavior therefore also exists outside Axiom; it cannot be
explained solely by its wrapper or speculative path. This does **not** prove
every Axiom numerical path correct. CPU float32 attention/recurrent execution
is not bitwise CUDA/FP8-KV parity, and one checkpoint cannot separate base-model,
abliteration and quantization effects. The four enabled format failures were
not independently regenerated in that diagnostic. Raw operational evidence
and model assets remain private.

## Public-source verification

The independently sanitized tree is checked with compile/link, provider host
tests and real-tokenizer CPU regressions, not a new GPU inference campaign:

```sh
make -j3 MEDIA=1 VISION=1 codex-host-tests
make -j3 MEDIA=1 VISION=1 qwen38-tokenizer-tests QWEN38_TOKENIZER_PATH=/path/to/tokenizer.json
make public-scan
```

The tokenizer must be supplied by the operator outside the repository. Tests
exercise the actual provider translation unit, template encoding, socketpair
SSE, exact prefix replay and large synthetic-history compaction. Build hashes
and final gate outcomes are recorded in [the manifest](../RELEASE_MANIFEST.json).

Public compile/link and host tests passed. Tokenizer-backed checks passed:
108 template/text-slot checks, 87 streaming handoff checks, 77 reasoning/parser/
budget checks, 120 exact prefix-replay checks and 16 compaction checks. All five
API self-tests also passed. The public-scope scan passed. The staged secret
scan reported four executable SHA-256 fields in the current/previous manifests;
all were reviewed as checksums, not credentials. No detection rules were disabled.

## Remaining limits

There is no blanket all-profile quality or general production GO. Multimodal
persistent KV reuse remains disabled pending safe persisted visual identity;
this update does not disable it anew. Cold long-context prefill can remain
slow and compaction summaries remain lossy. Full-model million-token use,
arbitrary histories, every model task, all desktop actions and all live
tutor/plugin/bot workflows are not qualified by this campaign. Correctly
reported output-budget exhaustion is not a completed answer.
