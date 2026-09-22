---
name: kvfit
description: Plan and qualify LLM inference memory with kvfit. Use when Codex needs to determine whether an exact Hugging Face model or checkpoint fits one or more GPUs or DGX systems; size KV cache or recurrent state for long contexts; compare TP/DP layouts; audit vLLM or SGLang support; or design target-host calibration. Trigger on requests involving LLM GPU memory, VRAM, KV-cache calculators, DGX Spark or GB10, model fit/OOM, tensor parallelism, context length, active serving sequences, or inference capacity.
---

# Use kvfit

Use the CLI as an evidence ladder: static topology planning first, installed-engine
preflight second when available, and measured target-host calibration for runtime
or SLO claims.

## 1. Establish the exact question

Identify:

- Hugging Face repository, checkpoint URL, and optional revision;
- context tokens per active sequence;
- GPU preset/count or DGX system/node count;
- requested TP size, or use `--tp auto` to compare valid TP/DP layouts;
- cache precision and any separate sparse-index precision;
- active sequences per user when agent branches can coexist.

Do not silently substitute total users for simultaneously resident active
sequences. Do not treat multiple GPUs or systems as pooled memory.

Inspect available presets when needed:

```bash
uvx kvfit --list-hardware
uvx kvfit --list-systems
```

Inside a kvfit source checkout, replace `uvx kvfit` with `uv run kvfit`.

## 2. Run the static plan

To find minimum whole-system counts per family for a concurrent-user target,
use native search (kvfit >= 0.2.5):

```bash
uvx kvfit OWNER/MODEL --systems dgx-h200 dgx-b200 dgx-b300 \
  --max-nodes 8 --concurrent-users 25 --context 128k
```

Default search uses whole systems with scale-up-local TP and full DP replicas.
For qualified native GLM DSA checkpoints (kvfit >= 0.3.0), compare the explicit
`--parallelism sglang-dpa` profile with a SGLang storage profile. It reads tensor
headers and adds local DPA/EP candidates. Distinguish full model groups, attention
DP/TP and expert ranks in the result. Never divide the cache by DPA while keeping
an unchanged ideal weight shard. All non-routed tensors are counted replicated;
this is a checkpoint envelope, not a bound on loaded runtime memory.

Report an empty search as no candidate in scope, not a universal impossibility.
Cross-node EP, PP/CP, offload and cache sharing remain unmodeled. Use
`--device-memory-gib` for explicit target memory and `--runtime-reserve-gib` for
an allowance inside the utilization budget. Neither is automatically measured.
Read [the evidence and exclusions](../../docs/capacity-search.md) before applying
`--cache-layout sglang-dsa-scaled --mtp`. Planning flags do not configure a server;
DPA calibration needs an explicit matching deployment. Unknown weight or cache
placement fails closed. Preserve evaluations before adapter changes.

Prefer JSON for agent use:

```bash
uvx kvfit OWNER/MODEL \
  --hardware h100-80 \
  --gpus 8 \
  --context 128k \
  --tp auto \
  --json
```

For complete DGX systems, use `--system` and `--nodes` instead of manually
multiplying GPU memory:

```bash
uvx kvfit nvidia/MiniMax-M3-NVFP4 \
  --system dgx-spark \
  --nodes 2 \
  --tp auto \
  --context 128k \
  --json
```

Use `--tp 2` only when the deployment will actually shard the model across the
two hosts. Compare TP=1/DP=2 and TP=2/DP=1 before choosing.

Checked integrated DeepSeek V4 DSpark checkpoints are detected automatically
from their config and draft-layer schedule:

```bash
uvx kvfit deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --system dgx-spark \
  --nodes 2 \
  --tp 2 \
  --context 1m \
  --json
```

The static report counts the integrated artifact weights and logical draft KV
component. It does not infer standalone speculator/target pairings, DSpark
hidden-state buffers, CUDA graphs, acceptance, or speed.

## 3. Interpret without overclaiming

Report:

- resolved model revision and weight source;
- architecture-specific cache/recurrent components;
- detected speculative-decoding method and modeled draft cache, when present;
- per-rank weights and cache for each TP/DP layout;
- static counted-state upper bound before runtime overhead;
- explicit fit/OOM reason and warnings.

Call a static `fits` verdict **memory-only**. It does not prove engine support,
runtime packing, allocator/workspace headroom, OOM safety, cross-host orchestration,
throughput, latency, long-context quality, or stability.

If kvfit rejects an unknown architecture, preserve the failure. Do not replace
it with the standard Transformer KV formula unless the architecture state has
been independently verified and implemented.

## 4. Check an installed engine when relevant

On the target environment, probe vLLM or SGLang:

```bash
uvx kvfit OWNER/MODEL \
  --check-engine all \
  --engine-python /path/to/serving-env/bin/python \
  --engine-probe config \
  --require-engine-pass \
  --json
```

Always point `--engine-python` at the Python executable where vLLM or SGLang is
installed; `uvx` itself runs kvfit in an isolated environment. Use
`--engine-probe load` only on the intended GPU host when downloading and
allocating the checkpoint is acceptable. A one-token load probe closes a launch
gap but still does not establish production capacity.

## 5. Require calibration for measured claims

Use a deployment TOML and run on the target CUDA host:

```bash
uvx --from kvfit kvfit-calibrate deployment.toml --output calibration.json
```

Use `--dry-run` first to resolve metadata and inspect launch/benchmark commands.
For multi-host systems, attach to an already orchestrated OpenAI-compatible
endpoint or provide explicit cluster launch commands. Never benchmark one host
and label it as a multi-host result.

Keep offered concurrency, measured resident requests, queueing, successful
completion, SLO qualification, and direct OOM distinct in the conclusion.
