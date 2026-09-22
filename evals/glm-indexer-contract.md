# GLM indexer contract: evidence before implementation

Written before the production fix, 2026-09-22. Run with:

```sh
uv run pytest tests/test_glm_evidence.py
```

## Independent anchors

The JSON fixtures preserve published model configurations at immutable Hugging
Face revisions. Only the explicitly listed quantization configurations are
omitted; these evaluations pass cache precision explicitly and do not qualify
weight loading. Each fixture records its source and revision.

For GLM-5.3, a separate extraction of the checkpoint tensor index identifies
21 base-model layers with `self_attn.indexer.*` weights, plus the MTP layer 78.
This observation is independent of kvfit's formula and frequency arithmetic.
The base-model index payload at 1,048,576 tokens and one byte per element is
2.625 GiB. MLA latent/RoPE state is 43.875 GiB: total 46.5 GiB before runtime
overhead. The MTP layer is outside this estimate.

The interpretation of `full`/`shared` and the frequency/pattern defaults is
grounded in Transformers revision `14793d45af28336310a0013a89f1488d8ba1cc51`:

- [Configuration rules](https://github.com/huggingface/transformers/blob/14793d45af28336310a0013a89f1488d8ba1cc51/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py).
- [Only full layers instantiate and update an indexer](https://github.com/huggingface/transformers/blob/14793d45af28336310a0013a89f1488d8ba1cc51/src/transformers/models/glm_moe_dsa/modular_glm_moe_dsa.py).

## Orthogonal checks and acceptance criteria

| Dimension | Observation required |
| --- | --- |
| Model generations | GLM-5.1 keeps all indexers; 5.2/5.3 count their shared schedule. |
| Other architectures | Pinned DeepSeek V3.2/V4, Qwen3 and Qwen3.6 preserve distinct cache semantics; GLM-5.3-Flash remains unsupported. |
| Context and precision | Boundary lengths and independently varied MLA/index widths affect only the corresponding state. Sub-byte cases test arithmetic, not kernel support. |
| Schedule representation | Explicit modes, string/list patterns and frequency/offset rules agree when equivalent. |
| Uncertainty | Invalid lengths/types, orphan shared layers, contradictory declarations and unmodeled state must raise `UnsupportedArchitecture`, not return a plausible fit. |
| Model identity | New types in the GLM, DeepSeek and Qwen cases must not inherit support from older class names; fallback aliases must match exactly. |
| TP/DP | MLA cache remains replicated under the planner's TP assumption; weights and replicas follow their own rules. Custom memory isolates this check from GPU specifications. |
| Public entry points | CLI JSON and audit report use the corrected payload and preserve memory-only qualification. |
| Oracle independence | Feeding the original incorrect 78-indexer estimate must fail, even when its arithmetic is internally consistent. |

The arithmetic oracle is a separate implementation, not independent empirical
validation. These checks cannot discover arbitrary new cache fields automatically.
New architecture support requires a pinned configuration, a source describing its
state, a positive case, an adversarial/changed-state case, and a regression for
existing families. Target-host measurement remains necessary for actual memory,
throughput, latency and serving capacity. GPU specifications are outside this fix.

## Verification record

Before production changes, the 119-case evaluation produced 106 failures and
13 passes; the pre-existing 150-test suite passed. The failures exposed the GLM
overcount, ignored/invalid schedules, unknown architecture aliases, and a separate
oracle crash on Qwen3-32B's published `sliding_window: null`.

A second dimension-validation evaluation was also run before its guard was
changed: five fractional-dimension cases failed and ten bool/string cases were
already rejected. This records detection, not 111 independent product defects:
many cases deliberately exercise the same defect across orthogonal inputs.

After the changes: 284 offline tests pass on Python 3.14, including all 134 new
model-evidence cases; Ruff passes. The 0.2.4 wheel was built through its source
distribution, installed in a separate environment, and exercised on GLM-5.1,
GLM-5.3 and explicit Flash rejection. A live metadata-only CLI request at the
pinned GLM-5.3 revision reports 43.875 GiB MLA plus 2.625 GiB index keys and
retains `memory-only` qualification. No weights, engine or GPU were loaded.
