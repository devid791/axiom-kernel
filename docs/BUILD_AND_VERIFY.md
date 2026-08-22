# Build and verification

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
  src/axiom_qwen38_session_store.cpp -o /tmp/axiom-session-store-asan
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
```

The binaries require compatible model artifacts supplied outside this
repository. A release result is valid only if:

- temporal token IDs, logits and all five target taps pass parity;
- repeated graph runs produce identical token IDs;
- speculative acceptance counters remain stable;
- a paged-runtime run crosses at least one page boundary without token or
  logit mismatch;
- the output digest and artifact hashes are recorded with throughput.

The optimization controls and their rollback values are listed in
`docs/qwen38/PERFORMANCE.md`. Change one control at a time when diagnosing a
regression. Load-time autotuning must complete before graph capture; timing or
algorithm selection inside replay is a release failure.
