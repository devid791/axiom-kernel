# Build and verification

## Reasoning and compaction regressions

`codex-host-tests` includes the model-free reasoning/parser/output-budget
regression and all five API self-tests. Additional CPU tests use an
operator-supplied compatible tokenizer, without loading weights, starting an
HTTP listener or running GPU inference:

```sh
make -j3 MEDIA=1 VISION=1 codex-host-tests
make -j3 MEDIA=1 VISION=1 qwen38-tokenizer-tests QWEN38_TOKENIZER_PATH=/path/to/tokenizer.json
make public-scan
```

The tokenizer group includes exact prefix replay, native text/multimodal
text-slot templates, socketpair SSE phase handoff and large synthetic history
compaction. Supplying a tokenizer does not qualify actual image inference or
model-answer quality. See [results and limits](REASONING_AND_COMPACTION.md).

## Codex provider follow-up

The optional public HTTP integration requires the image and FFmpeg development
libraries. Build and run the model-free native contract gates with:

```sh
make -j4 CUDA_ARCH=sm_120 MEDIA=1 VISION=1 all extended-host-tests codex-host-tests
make public-scan
```

An initial provider link with the kernel's default `MEDIA=0` exposed its required
video-decoder dependency. Provider targets now report that prerequisite explicitly;
the `MEDIA=1 VISION=1` build links the existing native decoder. This is not a
change to standalone kernel defaults. See [CODEX_APP_SERVER.md](CODEX_APP_SERVER.md)
for listener, authentication and real-inference qualification boundaries.

## September source update

The public source builds both the updated 27B backend and the opt-in Flash-Next
library. OpenSSL development headers/libraries are required for manifest v2.
The native reference CUDA units deliberately exclude global fast-math.

```sh
make -j4 CUDA_ARCH=sm_120 all extended-host-tests qwen38-dspark-abi-gate
make -j4 CUDA_ARCH=sm_120 qwen4exp-build qwen4exp-text-model-build
make qwen4exp-host-tests
```

These commands compile/link the native GPU code but do not load model weights
or run new GPU inference. Host tests cover manifests, KV transaction recovery,
SHA-256, checkpoint/speculator identities, scheduler, PLE and expert paging.
They do not replace checkpoint-backed CUDA correctness or performance gates.
`bin/qwen4exp-text-model-test MODEL_ROOT` is a separate explicit GPU operation.
Run only with adequate hardware and the compatible externally supplied weights.

The publication scan uses Gitleaks default rules. Eight findings were reviewed:
attention-dimension identifiers in the two GDN implementations, MTP key/value
head constants, and a public model-id string in the session-store fixture.
They are C++ symbols/test data, not credentials. The raw scan is not described
as zero findings, and no files or rules are disabled to suppress it.
Model/control/deployment exclusion checks are also provided by `public-scan`.

## Source checks

Run these checks from the repository root:

```sh
make public-scan
make host-tests
```

`public-scan` fails if code or build metadata contains private paths, private
network identities, credential-shaped strings, activation-steering symbols or
DFlash2 symbols. Documentation may describe excluded features so that the
release boundary is explicit.

The host session-store test exercises manifest durability, generation cleanup,
stateful resume, TTL/LRU selection, protected namespaces, atomic tombstones,
crash recovery, malformed artifacts, symlink/hard-link rejection, scan budgets
and file-descriptor stability. For sanitizer verification run the same source
with AddressSanitizer and UndefinedBehaviorSanitizer enabled:

```sh
c++ -O1 -g -Wall -Wextra -Werror -std=c++17 \
  -fsanitize=address,undefined -Iinclude \
  tests/axiom_qwen38_session_store_test.cpp \
  src/axiom_qwen38_session_store.cpp src/axiom_sha256.cpp \
  -lcrypto -o /tmp/axiom-session-store-asan
/tmp/axiom-session-store-asan
```

## CUDA build

The full library requires a CUDA toolkit and a C++17 compiler:

```sh
make CUDA_ARCH=sm_120
```

The architecture must match the installed GPU. `sm_120` is the RTX 5090/
Blackwell-oriented default; use the architecture supported by the actual
device and CUDA toolkit. The build never downloads weights or contacts a
service.

The image preprocessor is enabled by default and needs JPEG, PNG and WebP
development libraries. The FFmpeg container decoder is opt-in:

```sh
make CUDA_ARCH=sm_120 MEDIA=1
```

If the decoder libraries are absent, keep `MEDIA=0`; unsupported containers
must remain an explicit runtime error rather than silently falling back to an
unverified path.

## Runtime gates

For a model-backed verification run, record all of the following together:

1. source commit and compiler/CUDA versions;
2. GPU name, driver and selected `CUDA_ARCH`;
3. model and auxiliary checkpoint hashes;
4. tokenizer/template identity;
5. context/profile/sampling configuration;
6. cold and warm KV runs, including restored session identity;
7. output parity, acceptance statistics, TTFT and decode throughput.

No performance or long-context statement is portable without this fixture.

## Exact speculative-path verification

The optimized path is enabled by default except for load-time matrix
autotuning. For a performance candidate, enable autotuning before model
creation:

```sh
export AXIOM_QWEN38_MATMUL_AUTOTUNE=1
```

Use `tools/axiom_qwen38_temporal_gate.cpp` to compare temporal M8 output with
the scalar causal path, then use
`tools/axiom_qwen38_speculative_graph_gate.cpp` for repeated device-graph
decode. Build the three relevant gates with:

```sh
make CUDA_ARCH=sm_120 qwen38-temporal-gate
make CUDA_ARCH=sm_120 qwen38-speculative-graph-gate
make CUDA_ARCH=sm_120 qwen38-paged-runtime-gate
make CUDA_ARCH=sm_120 qwen38-dspark-abi-gate
```

The binaries require compatible model artifacts supplied outside this
repository. A release result is valid only if:

- temporal token IDs, logits and all five target taps pass parity;
- repeated graph runs produce identical token IDs;
- speculative acceptance counters remain stable;
- a paged-runtime run crosses at least one page boundary without token or
  logit mismatch;
- the output digest and artifact hashes are recorded with throughput.

The DSpark ABI gate is model-independent and runs after linking. It verifies
that the public header and library agree on layout revision 3, that undersized
configuration/state/history buffers are rejected without being touched, and
that revision-1 binary symbols fail closed. Passing this gate proves the ABI
guards; it does not replace model-backed token/logit parity.

For a resident graph executor, also verify request boundaries: begin a fresh
device session for every borrowed executor, stop at the first configured EOS,
record the exact committed-token count, and confirm that graph cycles already
queued after termination are no-ops. The target model must not retain ownership
from the preceding request.

The optimization controls and their rollback values are listed in
`docs/qwen38/PERFORMANCE.md`. Change one control at a time when diagnosing a
regression. Load-time autotuning must complete before graph capture; timing or
algorithm selection inside replay is a release failure.
