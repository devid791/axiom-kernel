# Performance methodology

Report prefill and decode separately. At minimum record prompt tokens, output
tokens, time to first token, total latency, sampling mode, speculative cycles,
accepted draft tokens, GPU architecture, CUDA/toolchain versions and model
artifact hashes.

A private RTX 5090 production fixture recorded roughly 298 visible decoded
tokens/s and roughly 304 internal decode tokens/s for a 1,002-token prompt and
256 generated tokens over 33 speculative cycles. This is an observation for
one exact fixture; it is not a source-only certification, a long-context proof
or a promise for another GPU, model artifact or serving adapter.

The public repository intentionally omits the private HTTP daemon and its
production logs. Reproduce the number only after rebuilding the kernel and
recording the complete fixture described in `docs/BUILD_AND_VERIFY.md`.
