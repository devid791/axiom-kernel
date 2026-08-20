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
