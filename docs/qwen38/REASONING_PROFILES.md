# Qwen3.8 reasoning profiles

Axiom exposes seven model-independent profiles. The budget is an exact ceiling
for hidden reasoning tokens; it is not a promise that every request spends the
whole allowance.

| Profile | Thinking | Budget cap | Intended use |
| --- | --- | ---: | --- |
| `ultra-fast` | off | 0 | lowest latency, direct answers |
| `minimal` | on | 1,024 | short structured tasks |
| `low` | on | 2,048 | light analysis |
| `medium` | on | 4,096 | default balanced reasoning |
| `high` | on | 8,192 | complex coding and planning |
| `xhigh` | on | 16,384 | difficult multi-step work |
| `max` | on | 32,768 | maximum reasoning allowance |

Sampling parameters are selected by the serving layer and are not encoded as
activation steering. A profile change must be included in the persistent KV
namespace so a cache created under one profile cannot be reused under another.
