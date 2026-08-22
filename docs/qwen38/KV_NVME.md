# Paged persistent KV and NVMe tier

The Qwen3.8 path separates logical context from physical residency:

- the logical address space is paged in 256-token units;
- a small hot window remains resident on the GPU for the active decode;
- cold, immutable pages can be checkpointed to an NVMe-backed session store;
- a session manifest records the model identity, auxiliary identity, profile and
  configuration signature alongside the committed token watermark;
- restore validates the manifest and page checksums before making a page
  visible to the decoder;
- separate sessions and model/profile namespaces cannot alias the same prefix.

The tier is a storage mechanism, not a claim that every request will use NVMe.
Short prompts can remain entirely in device memory. A reproducible KV test must
show a cold run, a committed snapshot, a new process or session restore, a warm
run and output/parity checks across the page boundary.

The implementation keeps page movement outside the native decode graph. That
preserves the graph's hot path while allowing the session store to evict and
restore cold pages. A deployment may choose its NVMe directory and capacity;
the public kernel contains no machine-specific path or endpoint.

## Bounded session lifecycle

Generation cleanup and whole-session retention are separate operations. The
caller supplies a TTL, an optional maximum namespace count and the namespace
identifiers that are still active. A zero TTL disables age-based removal and a
zero namespace cap disables LRU removal.

Each candidate namespace is validated before removal. A valid namespace:

- is a private directory owned by the current runtime user;
- contains files for exactly one validated session stem;
- contains only recognized manifest, temporary-manifest and generation-page
  names;
- contains no symlink, hard link, nested directory or unknown artifact.

Removal first atomically renames the namespace to a private GC tombstone. It
then reopens that directory without following symlinks and removes only the
files validated during the scan. A crash may therefore leave a complete
tombstone, never a half-visible live namespace. The next lifecycle pass finishes
valid tombstones idempotently. Malformed or raced state is retained and reported
instead of being followed or deleted.

The scan has configurable namespace and artifact budgets. Exhausting either
budget returns an explicit error and does not select additional sessions for
removal. Deployments must serialize lifecycle GC with native generation and
persistence, and must protect every namespace with an active request, lease or
pending asynchronous commit.

## Lossless native resume

The session manifest records the exact committed token watermark separately
from the pending predicted token. `qwen38_build_stateful_resume_prompt` can
combine that durable raw prefix with a visible-history suffix when a client
normalizes hidden thinking or tool-call markup. It rejects EOS cursors,
malformed watermarks, invalid token IDs and candidates that exceed the caller's
prompt budget. A rejection is a safe cache miss; it never reinterprets an older
manifest namespace.
