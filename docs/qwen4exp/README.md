# Qwen3.8 Flash-Next NVFP4 native backend

Axiom includes a separate `qwen4_exp` backend for
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, checkpoint revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
This is a usable native C++17/CUDA backend with recorded real-model execution,
not just an architectural plan or a routing alias to another engine.
Model weights and their upstream license remain separate from Axiom's MIT code.

## Included source

- ModelOpt checkpoint/tensor admission and payload verification.
- A 48-layer native decoder: 36 Gated DeltaNet and 12 QSA layers.
- Multi-hyperconnection (mHC), PLE, rotary embedding and output head.
- Routed/shared expert compute with bounded exclusive GPU/RAM residency,
  NVMe direct reads, eviction and exact PLE row fetching.
- Independent session state, transactions and decoder residency adapters.
- Native vision/MTP components and their qualification probes, with the
  limitations below. Their presence is not evidence of end-to-end support.

## Support and evidence boundaries

| Surface | Status |
| --- | --- |
| Native short-context text execution | Implemented; historically qualified in an isolated private deployment |
| GPU/RAM/NVMe expert paging | Implemented; exact routing and cache gates recorded in the private qualification |
| Public library and text-model probe | Separate build targets; no model weights are bundled |
| HTTP/OpenAI/Anthropic serving | Private integration exists; no serving daemon is included in this repository |
| Native vision tower | Historical synthetic-patch determinism/device-handoff gate; not image QA |
| Vision HTTP | Not implemented in the qualified Flash-Next integration |
| MTP accelerated decoding | Not qualified; disabled in the qualified private text profile |
| 262,144-token logical ceiling | Configuration/model contract, not a completed full-context runtime test |
| 1M context | Not qualified |
| Concurrent GPU serving | Not claimed by this release |

The prior isolated qualification used an RTX 5090 with GPU/RAM expert caching.
It recorded **approximately 41 decoded tokens/s on a warm short greeting**:
8 visible tokens after the first token over 195.196 ms, with 9 total output
tokens and 582.542 ms complete HTTP time. An earlier 4,096-GPU-slot profile
recorded approximately 44 tok/s on a greeting. These are real historical
private-runtime measurements, not benchmarks rerun for this source publication.
Twelve fixed text smoke cases produced 10 passes and 2 failures; this is not a
general accuracy benchmark. Small fully warm greetings are not representative
of general offloaded conversations. No throughput promise is made here.
The qualified private profile kept MTP off and did not replace 27B production.

## Build and reproduce

```sh
make CUDA_ARCH=sm_120 qwen4exp-build qwen4exp-text-model-build
make qwen4exp-host-tests
```

The outputs are `lib/libaxiom-qwen4exp.so`, linked against Axiom core, and
`bin/qwen4exp-text-model-test`. Run the latter with an explicit `MODEL_ROOT`
containing the compatible checkpoint, on an otherwise appropriately provisioned
CUDA machine. It verifies the model payload by default. Do not skip payload
verification when qualifying a new deployment.

GPU probes are not run by these build targets. `qwen4exp-host-tests` runs only
synthetic PLE and expert-pager CPU tests, including bounded temporary-file I/O.
Hardware memory requirements depend on the explicit cache and session options;
the whole checkpoint does not fit in a 32 GB GPU. Inspect the public option
structures before sizing a deployment. No internal machine paths, deployment
profiles, credentials or checkpoint files are part of this export.

Source provenance: `a4aeda07b332471cb8aed1320b231c03521f01b2`; historical
qualification source: `d3633f924e036fa444506f3f155ac4eb0841dfc0`.
