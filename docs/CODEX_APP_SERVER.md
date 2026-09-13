# Codex App Server and the native provider

This public source includes the native Qwen3.8 HTTP provider and its
`/codex/v1` integration, initially derived from source `711318c` (11 September
2026), with [conversation reliability fixes](CONVERSATION_RELIABILITY.md)
from `fffdaf0` (13 September 2026).
It complements the existing DSpark/M8 kernel update; it does not replace Codex
App Server or move tool execution into the inference server.

## Boundaries

| Surface | Contract |
| --- | --- |
| `/codex/v1/responses` | Request-local Codex name/schema/custom-tool bridge and incremental Responses SSE |
| `/codex/v1/models` | Provider catalog and Codex capability metadata |
| `/v1/responses` | Existing strict native Responses route |
| `/v1/chat/completions` | Existing native Chat Completions route |
| Core stdio JSON-RPC | Client/App Server thread, turn, tool and approval protocol; not one of the HTTP routes above |

The bridge handles namespaces, readable reversible tool identities, deferred
discovery, custom input descriptions and supported exact schema alternatives.
It validates generated arguments against the original retained schema before
emitting a tool call. Unsupported definitions produce explicit rejection data;
the strict `/v1` route is not silently relaxed.

Call IDs, item identities, original names/namespaces and opaque tool outputs
survive the bridge. Incremental output, terminal events, errors, cancellation
and stream-progress comments use the native writer. Core and its configured
MCP/client handlers still own dispatch, permissions and actual execution.

## Build and host-side checks

Use a CUDA toolkit, C++17, OpenSSL, JPEG/PNG/WebP and FFmpeg development
libraries. The HTTP provider supports multimodal input and requires both
`VISION=1` and `MEDIA=1`. The standalone kernel's optional-media build remains
unchanged. The explicit provider build is:

```sh
make -j4 CUDA_ARCH=sm_120 MEDIA=1 VISION=1 qwen38-api codex-host-tests
make public-scan
```

The host-side contract checks compile/link the actual daemon and exercise
normalization, schemas, tool identities, call/result history, SSE, stream
progress, UTF-8, cancellation/deadline logic and memory-budget boundaries.
They do not load weights, launch GPU inference or prove a real App Server turn.

The `codex-provider-test` header test and `vision-memory-budget-test` can run
without the full CUDA provider. Full `codex-provider-api-test` links the actual
native library, but its tests remain model-free.

## Run a local provider

The executable accepts:

```text
axiom-qwen38-api TARGET_DIR DSPARK_DIR [IP:PORT[,IP:PORT...]] [MAX_CONTEXT]
```

Supply compatible target, tokenizer and speculator artifacts yourself. The
checkpoint/speculator identity sidecar must match the actual artifacts; the
server fails closed on a missing or incompatible contract. Use an application-
owned writable KV directory via `AXIOM_QWEN38_KV_TIER_PATH`. No weights, sidecars
containing private artifact identities, service units or deployment profiles
are provided by this source publication.

The public default listener is **127.0.0.1:8015 only**. The daemon has **no
authentication or TLS layer**. Do not expose it directly to the internet or
untrusted users, including its operations endpoints. Remote use requires an
independently configured authenticated transport and access policy. A supplied
Bearer header is not authentication unless the receiving transport enforces it.

Configure a compatible client to use base URL
`http://127.0.0.1:8015/codex/v1` only when it runs on the same host. In Synora,
use Models & accounts to select Axiom and configure the endpoint. Do not point
App Server's JSON-RPC transport at an HTTP provider endpoint.

## Core versions and discovery

The integration was developed against the pinned Core 0.153.4 protocol.
Synora carries its own versioned adapter and platform-specific update gates.
Use Core's `model/list` result and the actual provider catalog; do not hardcode
model entitlements or infer reasoning effort from a model's friendly name.
Upgrading an upstream Core package does not qualify that package automatically.

See [official App Server documentation](https://learn.chatgpt.com/docs/app-server)
and [Synora](https://github.com/devid791/Synora). Available reasoning content
is provider output, not a guarantee of access to hidden internal reasoning.

## Subsequent kernel improvements

Known-token scalar prefill can omit an unnecessary vocabulary projection for
intermediate prompt tokens. Paged eight-token prefill has its own causal,
page-contained path and does not enlarge the speculative hot window. The final
prompt prediction and generated tokens retain their normal decoding contract.

Vision scratch allocations grow against checked runtime memory budgets rather
than a fixed combined patch-token ceiling. Stream progress preserves liveness
during long preprocessing/prefill. Dynamic capacity still has real VRAM, context,
integer-overflow and allocation-error boundaries; it is not unlimited vision.

The historical 409 tok/s DSpark record is unchanged. It is not a benchmark of
this public provider build, a new prefill performance claim or proof of all
large-context/large-image workloads. No production service was changed or
restarted for this publication.

Activation-control implementations and DFlash2 remain excluded. The provider
was adapted to the public kernel by removing those controls, preserving request
serialization, using a loopback default and rejecting activation overrides.
The independent sanitized build needs its own model-backed qualification before
deployment; private-runtime qualification does not automatically transfer.
