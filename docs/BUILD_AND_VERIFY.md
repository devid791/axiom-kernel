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
