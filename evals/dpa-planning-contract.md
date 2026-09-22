# SGLang DPA placement: evaluation before implementation

Outcome: an installed kvfit can compare local TP replicas with qualified local
SGLang DP-Attention + EP layouts, without conflating full replicas and attention
groups. Existing TP calculations remain unchanged when the profile is not chosen.

Evidence: SGLang 20a491d1d311553bbab3f22e19bbafb86ef3c0cc, specifically
`layers/dp_attention.py:compute_dp_attention_world_info`,
`models/glm4_moe.py:GlmMoeDsaForCausalLM` and `models/deepseek_v2.py`.
GLM inherits the DeepSeek implementation. Attention TP is group_width / DPA.
MLA caches are replicated within that attention group, not across independent
attention groups. Routed experts are distributed across EP ranks. Other checkpoint
tensors are conservatively counted fully replicated; this is a checkpoint
placement envelope, not a bound on runtime transformations or allocations.

GLM-5.3 aca966e4e02791568aa6a4ced368624b3d897f42 headers, observed independently
before this change: 734618714112 routed-expert bytes and 20998426304 other bytes;
the sum equals index metadata.total_size. 141 bounded header reads; no weights.
Those constants belong in evidence fixtures, never in production logic.

Required challenges:

- Ordinary TP unchanged; full and partial DPA maintain distinct replica counts.
- Per-rank bottlenecks, integer placement, one-time padding and reserves.
- Expert count divides EP; query heads divide attention TP (not the whole group).
- No cross-node expert groups silently substituted in whole-system search.
- Complete checkpoint tensor coverage and totals; missing/malformed headers,
  mismatched shapes/dtypes, unequal experts, unsupported quantization rejected.
- Different layer/expert counts, BF16 versus FP8 and quantization scales.
- No GLM-name matching; neighboring architectures explicitly unqualified.
- HTTP range honored and bounded; pin resolved revision; no weight download.
- CLI/JSON/TOML expose the profile, evidence, exclusions and DPA versus full DP.
- A memory profile must not silently launch or probe an ordinary TP server and
  attribute that result to DPA; calibration needs explicit matching deployment.
- Complete offline suite, lint, built-wheel install, real pinned metadata run.

Initial failing evaluation is recorded before production changes. GPU allocations,
kernel support, throughput, latency and SLO are outside this offline evidence.
