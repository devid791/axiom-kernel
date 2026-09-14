# Axiom reasoning profiles: Qwen reference adapter

Axiom exposes seven stable profile identifiers. `ultra-fast` explicitly
disables reasoning; it is not a faster enabled-thinking level or an omitted
override. The budget is a ceiling, not a required token spend or a guarantee
of accuracy. These numerical caps are Axiom/Qwen policy, not OpenAI budgets
or a claim of equivalent intelligence between providers.

| Wire profile | Thinking | Budget cap | Native instruction family |
| --- | --- | ---: | --- |
| `ultra-fast` | off | 0 | off, empty thinking block |
| `minimal` | on | 1,024 | low |
| `low` | on | 2,048 | low |
| `medium` | on | 4,096 | medium, no added effort instruction |
| `high` | on | 8,192 | xhigh |
| `xhigh` | on | 16,384 | xhigh |
| `max` | on | 32,768 | xhigh |

The catalog exposes `native_conditioning` separately from the selected profile
and its budget. The Qwen template's three enabled instruction families do not
collapse the six enabled wire IDs. Other model adapters must honor their actual
advertised choices rather than silently inventing support or downgrading one
profile to another. Tutor/worker selections must be preserved independently;
this API publication does not qualify all live delegation workflows.

Default sampling is temperature 0, top-p 1, top-k 1 with reasoning disabled,
and 0.6 / 0.95 / 20 when enabled. Actual `thinking_tokens` are distinct from
the selected `thinking_budget` and visible-output allowance. Total generated
tokens may legitimately exceed a visible-only cap when reasoning is enabled.

HTTP clients select `reasoning_effort` or `reasoning.effort` using one of the
seven IDs. An explicit effort overrides the legacy boolean `think`; unknown
IDs reject rather than becoming a default. `none` is not an additional Axiom
wire ID. The operator setting `AXIOM_QWEN38_NO_THINK=1` supplies an Off default
only when no explicit effort is supplied.

## Native prompt and output contract

Text and multimodal encoders use native ChatML turn-ending newlines, enabled
`<think>\n` prefill, conditioning/tools/initial-system ordering, assistant
reasoning wrappers and grouped consecutive tool replies. Original content
bytes and multimodal slots are preserved. Only the exact two-newline separator
after a natural reasoning close is removed before visible streaming and
finalization; additional answer whitespace is retained.

The prompt contract is versioned as `qwen38-chatml-v1` in the durable session
namespace, alongside the reasoning profile. Old KV files are not deleted or
treated as compatible state. An existing conversation must supply its history
once for reconstruction into the new namespace; subsequent compatible turns
retain exact prefix reuse. This is a one-time compatibility boundary, not a
requirement to prefill all history on every turn.

See [reasoning/compaction QA](../REASONING_AND_COMPACTION.md) for actual results
and remaining model-quality failures. More budget does not guarantee a correct
answer, and single stochastic runs do not establish an accuracy ranking.
