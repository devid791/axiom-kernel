# Contributing to Axiom Kernel

Keep the kernel portable, reproducible and free of deployment assumptions.

- Do not commit model weights, credentials, private keys, internal hostnames,
  private IP addresses, production manifests or service logs.
- Keep model-specific code behind public headers and the generic ABI where
  practical.
- Preserve upstream license and notice files for vendored dependencies.
- Add a deterministic host test or GPU parity test for changes to numerical
  kernels, quantization, KV paging or speculative verification.
- Report the CUDA toolkit, driver, GPU architecture, compiler and model
  artifact hashes when publishing performance numbers.
- Do not reintroduce activation-steering hooks or steering packs into the public
  tree without an explicit security and scope review.

Run `make public-scan` before opening a pull request. Full CUDA tests require a
CUDA-capable runner and model artifacts supplied outside the repository.
