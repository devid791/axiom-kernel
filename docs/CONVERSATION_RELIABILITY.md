# Conversation continuity and long-context reliability

For the subsequent current-exchange/tool-authority correction and native
reasoning-template fixes, see the
[September 14 update](REASONING_AND_COMPACTION.md). The evidence below remains
the original September 13 publication record, not a rerun of all these cases.

## September 13, 2026 source update

This update imports reviewed fixes from source
`fffdaf0588b05589fdbeb5bdf517031d3f68f8b6` onto the existing sanitized public
history. It does not publish private conversations, operational reports,
credentials, model assets or deployment profiles. The license remains MIT.

## What changed

- **Equivalent text prefixes:** generated text can be split into different BPE
  tokens when a client resends it. Ordinary-token bytes must match exactly,
  while control tokens remain indivisible identity boundaries. Successful reuse
  keeps the committed native token IDs and KV/recurrent state, processing only
  the new suffix. Changed history, controls, invalid decoding and unsafe bounds
  reject reuse; matching visible text does not permit rewriting cached state.
- **Tool-result continuation:** a native assistant turn with multiple calls can
  be replayed as separate client items. The saved history is projected through
  the existing tool parser/renderer and compared against the authoritative
  incoming prefix. Instructions, schemas, arguments, results and ordering are
  not ignored. This projection is restricted to Codex text/no-thinking replay;
  unknown representations fail closed. The final EOS is consumed exactly once.
- **Mixed resident/cold attention:** the opt-in
  `AXIOM_QWEN38_KV_MIXED_SPLIT64=1` path validates logical page identities,
  processes the resident suffix with one split-K pass, and combines its FP32
  softmax state with the cold-page states before final normalization. It does
  not drop history, change KV precision or extend speculative decoding windows.
  Unsetting the flag retains the previous attention path.
- **Observed request progress:** `/ops/runtime` and Responses progress carry
  actual prefill/decode counters. A heartbeat does not invent token progress.
  Active state is isolated by request ownership and cleared after completion.
- **Explicit terminal outcomes:** exhausted output budgets report
  `incomplete/max_output_tokens`; empty non-tool responses report failure.
  Neither is incorrectly reported as successful completion. An absent operator
  deadline no longer invents a 600-second total cap. Explicit deadlines,
  cancellation, token/context bounds and queue controls remain authoritative.
- **Image-history replay:** assistant `output_text` is normalized to internal
  `text` when replayed alongside media. Images and annotations are preserved;
  malformed text is rejected. This fixes a second-turn HTTP 400, not image KV
  persistence.

## Evidence and its boundaries

The following measurements were obtained on the source-derived private native
deployment, **not** by deploying this independently sanitized public build:

| Check | Result and scope |
| --- | --- |
| Ring/address properties | 10,108 CPU checks passed |
| GPU attention equivalence | 59 cases passed, 1 through 1,048,576 tokens; random, analytic marker, zero, gaps and ring capacities |
| Numerical thresholds | Max absolute error <0.002 and relative L2 <1%; largest observed errors 0.000244141 and 0.000425691 |
| CUDA memory checking | Two small gap/ring cases, zero memcheck errors |
| Real conversation matrix | 23/23 final checks passed: text, Unicode, JSON/SSE, two reasoning profiles, session isolation, images, long-context tools |
| Real control matrix | 9/9 passed: cancellation/recovery, concurrency, invalid budget/recovery, backend compaction and continuation |
| Long text reuse | All ten approximately 69K-token follow-ups restored their prefix; 4.196–10.441 seconds, median 4.996 seconds |
| Real Core tool execution | Linux Core 0.153.4, two owned QA file reads, subsequent cache restoration and exact recall passed |

The GPU matrix is an attention-only numerical test, not a full-model 1M-token
conversation or NVMe transfer benchmark. Long-turn timings involve different
prompts and output sizes and are not a universal speedup claim. An initial
copy-prompt punctuation omission was recorded; clarifying that prompt is not
a model-quality fix. Backend compaction used repetitive synthetic 260K history:
it proves activation/current-request preservation/continuation, not arbitrary
summary fidelity or a separate desktop compaction workflow.

**Remaining gap:** multimodal persistent KV reuse is still disabled. Image
follow-ups now succeed but rebuild their context. Enabling reuse safely requires
persisted visual identity/state: placeholder token IDs alone cannot distinguish
different images. Long-context vision, every reasoning profile, every plugin or
bot, full-model million-token use and all desktop UI actions are not qualified
by this matrix. No universal production GO or model-correctness guarantee.

## Reproduce checks

Publication verification on the independent public build passed compile/link,
the provider host suite (including image replay and terminal/deadline handling),
13 SSE progress cases, 39 request-progress checks and 10,108 ring checks. The
tokenizer-replay and GPU-matrix executables were built but not run against a
model/tokenizer or GPU for this publication. Both JavaScript runners passed
syntax checks. The public-scope scan passed. The staged secret scan flagged
four executable SHA-256 fields in the current/archived manifests; each was
reviewed as a checksum, not a credential. No detection rules were suppressed.

Model-free checks on the public source:

```sh
make -j3 MEDIA=1 VISION=1 codex-host-tests
make public-scan
```

Tokenizer-backed CPU replay tests require a compatible tokenizer supplied by
the operator; no model is loaded and no GPU inference is started:

```sh
make MEDIA=1 VISION=1 codex-prefix-replay-test QWEN38_TOKENIZER_PATH=/path/to/tokenizer.json
```

Attention-only GPU checks require spare GPU memory. The full matrix allocates
large synthetic fixtures; schedule it on an idle, isolated GPU, not beside an
active production model:

```sh
make MEDIA=1 VISION=1 bin/axiom-qwen38-mixed-kv-microbench
bin/axiom-qwen38-mixed-kv-microbench --matrix
compute-sanitizer --tool memcheck --error-exitcode 99 bin/axiom-qwen38-mixed-kv-microbench --sanity
```

The live runners deliberately require `--allow-live` and an output directory.
They submit real requests, consume model resources and cancel QA work. Use an
isolated test provider/session store. Output includes private request/response
content and must not be committed or shared without review. The default target
is loopback, with no embedded credentials or remote deployment assumptions:

```sh
node tools/axiom_qwen38_live_matrix.mjs --allow-live --out=qa-conversations --endpoint=http://127.0.0.1:8015
node tools/axiom_qwen38_control_matrix.mjs --allow-live --out=qa-controls --endpoint=http://127.0.0.1:8015
```

The first runner covers 13 scenarios by default. Its optional
`--long-state=/path/to/owned-qa-state.json` adds ten long-context scenarios;
the seed must be an independently prepared QA session with matching `qa-`
thread/cache identities, never a live user's session. Its tool adapter reads
only its newly owned fixture after checking the exact proposed call; it never
executes arbitrary model-generated shell commands. Reports distinguish failure
and incomplete runs from success.

The public provider retains its loopback default, request serialization and
exclusion of private activation controls. It has no built-in authentication/TLS;
see [provider boundaries](CODEX_APP_SERVER.md) before any remote deployment.
