# kvfit — LLM KV-cache and GPU memory capacity planner

[![PyPI](https://img.shields.io/pypi/v/kvfit)](https://pypi.org/project/kvfit/)
[![Python](https://img.shields.io/pypi/pyversions/kvfit)](https://pypi.org/project/kvfit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/teilomillet/kvfit/blob/main/LICENSE)

**Will this exact Hugging Face model fit my GPU or DGX system at the context
length and concurrency I need?**

`kvfit` is a Python CLI for LLM inference-memory and KV-cache capacity planning.
It reads the requested checkpoint metadata, records the resolved revision,
counts model weights and architecture-specific inference state, evaluates
explicit tensor-parallel (TP) and data-parallel (DP) layouts, and explains why
a deployment fits or runs out of memory. It can also check an installed vLLM or
SGLang build and calibrate the estimate against a real target-host serving
sweep.

Use it when you are searching for a **KV-cache calculator**, **LLM GPU-memory or
VRAM estimator**, **Hugging Face model fit checker**, or **DGX capacity planner**
that keeps static estimates separate from measured runtime evidence.

[Install](#install) · [Quick start](#quick-start) ·
[Target-host calibration](#calibrate-on-the-target-serving-host) ·
[Engine checks](#check-the-installed-serving-engine) ·
[DSpark detection](#deepseek-v4-dspark-detection) ·
[DGX Spark accuracy](https://github.com/teilomillet/kvfit/blob/main/docs/dgx-spark.md) ·
[Supported architectures](#what-is-supported-now) ·
[Compare similar tools](#how-kvfit-differs-from-similar-tools) ·
[Method and limits](#what-the-number-means)

## Install

Run it without modifying a project:

```bash
uvx kvfit --version
```

Or install it:

```bash
pip install kvfit
kvfit --version
```

Inside a source checkout, use `uv sync` followed by `uv run kvfit`.

## Quick start

```bash
uvx kvfit Qwen/Qwen3-32B \
  --hardware h100-80 \
  --gpus 8 \
  --context 128k \
  --kv-dtype fp8
```

The report gives you:

- the resolved Hugging Face revision and checkpoint weight footprint;
- KV or recurrent state per full-context active sequence;
- per-rank weight and cache costs for each valid TP/DP layout;
- a static counted-state upper bound and explicit fit/OOM reason that never
  masquerade as a runtime guarantee;
- warnings when runtime packing, cross-node fabric, or engine behavior is still
  unmeasured.

`kvfit` fails closed on cache architectures it does not understand. It does not
silently apply the standard Transformer KV formula to a new hybrid, recurrent,
compressed, or sparse-attention model.

### For research agents and automation

Use `--json` for a machine-readable report:

```bash
uvx kvfit nvidia/MiniMax-M3-NVFP4 \
  --system dgx-spark \
  --nodes 2 \
  --tp 2 \
  --context 128k \
  --json
```

An agent can inspect the resolved revision, weight source, cache components,
topology fields and reasons, utilization, concurrent active sequences, and
warnings without parsing prose. A static `fits` result is intentionally labeled
memory-only; use `kvfit calibrate` on the target CUDA hosts for measured capacity
and SLO evidence.

The repository also ships a portable
[kvfit Agent Skill](https://github.com/teilomillet/kvfit/blob/main/skills/kvfit/SKILL.md)
with the selection workflow and evidence guardrails used by Codex-compatible
agents. It helps an agent invoke a tool it has already discovered; it does not
by itself make an unpublished repository searchable.

### How kvfit differs from similar tools

`kvfit` is the narrow, automation-first choice when the question starts with an
exact Hugging Face checkpoint, context, and GPU or DGX topology. It complements
rather than replaces broader tools:

- Hugging Face `accelerate estimate-memory` estimates model loading, not inference.
- ModelInfo CLI inspects checkpoint tensors and simulates VRAM/vLLM capacity.
- hf-mem is a lightweight Hugging Face weights and experimental-KV estimator.
- FitLLM is a browser-first GPU/Mac fit calculator.
- llmfit recommends and benchmarks local models against detected hardware.
- llm-mem-planner explores TP/DP/SP/EP memory and approximate performance layouts.

The distinguishing boundary is architecture-specific inference state plus
explicit topology and target-host qualification. See the
[source-backed comparison](https://github.com/teilomillet/kvfit/blob/main/docs/alternatives.md)
or the dedicated
[DGX Spark LLM memory and accuracy guide](https://github.com/teilomillet/kvfit/blob/main/docs/dgx-spark.md) and
[MiniMax M3 on two DGX Sparks guide](https://github.com/teilomillet/kvfit/blob/main/docs/minimax-m3-dgx-spark.md).

### More model and topology examples

DeepSeek V4 keeps its sparse indexer at a different precision in common vLLM
deployments, so state that explicitly:

```bash
uvx kvfit https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash \
  --hardware b200-180 \
  --gpus 4 \
  --context 1m \
  --kv-dtype fp8 \
  --index-dtype fp4
```

Use `--list-hardware` to see built-in presets; `--hardware custom:96` handles an
accelerator with 96 GiB. Contexts beyond the checkpoint's declared maximum are
rejected unless you add `--allow-context-overflow`, which labels the result as
hypothetical.

### DeepSeek V4 DSpark detection

Integrated DeepSeek V4 DSpark checkpoints are detected from their checked
`dspark_*` config fields and post-model cache schedule, not from a repository
name. No extra flag is required:

```bash
uvx kvfit deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --system dgx-spark \
  --nodes 2 \
  --tp 2 \
  --context 1m \
  --json
```

The detector is regression-checked against two differently sized official
configs: [DeepSeek-V4-Flash-DSpark at `62af8ff`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/config.json)
and [DeepSeek-V4-Pro-DSpark at `7c09739`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/blob/7c09739fd136abfb70a49ec334157f65f45b52cd/config.json).
Both reports identify integrated `dspark`, record three draft layers, and add a
`dspark-draft-kv` component:

```text
3 draft layers × min(context, 128) entries × 512 shared K=V width × KV bytes
```

At 1M context with BF16 cache values, that component is 393,216 bytes
(0.000366 GiB) per active sequence and rank. The target model's logical state is
unchanged; the draft state is additional and replicated because the checked
DeepSeek layout has one shared K=V head. Repository artifact accounting already
includes the integrated draft weights.

This is not a complete DSpark runtime-memory or speed prediction. vLLM also
allocates hidden-state buffers, CUDA graphs, allocator workspace, and other
engine state based on serving settings. `kvfit` reports those as unmodelled and
leaves acceptance, latency, and throughput to `kvfit calibrate`. A checkpoint
can declare DSpark even when a particular launch does not enable it; the static
planner conservatively includes its declared draft cache. JSON therefore labels
the declaration source as `model-config` and leaves `runtime_enabled` as `null`.
Standalone DSpark
speculator repositories still require their target model and are not inferred
as standalone serving targets.

For complete DGX systems, specify systems rather than manually multiplying GPU
counts. This preserves the scale-up boundary and reports a static counted-state
upper bound:

```bash
uvx kvfit openai/gpt-oss-120b \
  --system dgx-h100 \
  --nodes 2 \
  --context 90000 \
  --active-sequences-per-user 1
```

`context = 90000` means exactly 90,000 tokens; the shorthand `90k` means 92,160
tokens because suffixes use powers of 1024. One concurrent user means one
resident full-context sequence by default. Set `--active-sequences-per-user 2`
for an agent that may keep two branches resident; the reported user count is
then divided by two. This remains a static counted-state upper bound, not a
runtime OOM guarantee or throughput or latency SLO.

Built-in systems include DGX Spark, DGX H100, H200, B200, B300, and the DGX
GB200 NVL72 rack. `--list-systems` prints their accelerator counts and
scale-up/scale-out boundaries. `--nodes` multiplies complete systems; TP within
a system's NVLink domain is labeled node-local, while larger TP is explicitly
reported as cross-domain and unqualified until tested on the target fabric.

## Reusable TOML presets

Pass a TOML file as the positional argument, or use `--config`. Command-line
values override the file, so the preset can hold the stable deployment while a
single run changes one dimension:

```toml
# inkling-b200.toml
model = "thinkingmachines/Inkling-NVFP4"
hardware = "b200-180"
gpus = 8
context = "1m"
kv_dtype = "bf16"
utilization = 0.90
tp = 8

[engine]
check = ["vllm"]
python = ".engine-envs/vllm/bin/python"
probe = "config"
tp = 8
timeout = 300
require_pass = false

[output]
json = true
```

```bash
uv run kvfit inkling-b200.toml
uv run kvfit --config inkling-b200.toml --context 256k --gpus 4 --tp 4
```

Supported root keys are `model`, `revision`, `context`, `kv_dtype`,
`index_dtype`, `weight_gib`, `allow_context_overflow`, `hardware`, `gpus`,
`system`, `nodes`, `active_sequences_per_user`, `utilization`, `tp`, and the
Hugging Face request `timeout`. Use either `hardware` plus `gpus`, or `system`
plus `nodes`; the strict parser rejects a mixture. The `[engine]`
table accepts `check`, `python`, `probe`, `tp`, `timeout`, and `require_pass`;
`[output]` currently accepts `json`. A relative `engine.python` path is resolved
from the preset's directory. Unknown keys, invalid enum values, malformed TOML,
and wrong value types are errors rather than ignored settings. JSON reports
include the resolved `input_config` path.

## Calibrate on the target serving host

The memory result is a hypothesis until the installed engine serves the real
checkpoint. Add a strict `[calibration]` table and run the target-host sweep:

```toml
model = "openai/gpt-oss-120b"
system = "dgx-spark"
nodes = 1
context = 90000
kv_dtype = "bf16"
tp = 1

[engine]
check = "vllm"
python = "/opt/vllm/bin/python"
probe = "config"

[calibration]
engine = "vllm"
mode = "launch"
python = "/opt/vllm/bin/python"
output_tokens = 1
max_concurrency = 32
requests_per_concurrency = 1
startup_timeout = 1800
request_timeout = 7200
metrics_interval = 1.0
stop_on_failure = true
```

```bash
uv run kvfit calibrate deployment.toml --dry-run
uv run kvfit calibrate deployment.toml --output calibration.json
# Equivalent standalone entry point:
uv run kvfit-calibrate deployment.toml --output calibration.json
```

`launch` first runs the installed-version config preflight, rejects a default
managed launch without a CUDA accelerator, records the exact server command,
waits for readiness, and then runs the engine project's own serving benchmark.
For each concurrency point, the random workload uses exactly
`context - output_tokens` input tokens, zero length variation, and the requested
output length. A run qualifies only when the benchmark reports an aggregate
input-token count that matches the requested workload; missing or mismatched
token evidence fails closed.

The sweep samples `/metrics` concurrently. It understands current and legacy
vLLM running/waiting/KV-use names plus SGLang's running, queue, and token-use
gauges. The result keeps offered concurrency, measured resident requests,
queueing, successful completions, SLO qualification, and direct OOM evidence as
separate facts. A scheduler queue is never labeled an OOM.

By default, the sweep tests powers of two and the predicted boundary, capped by
`max_concurrency`. Set `concurrency = [1, 2, 4, 8]` for exact points. One request
per point is an intentionally cheap long-context capacity boundary. Set
`requests_per_concurrency = 5` or more for a steady-state run; this follows the
SGLang serving-benchmark guidance and can be very expensive at 90k+ tokens.
Optional `max_ttft_ms`, `max_tpot_ms`, and `max_e2e_ms` produce a separate
SLO-qualified ceiling.

For one host, the generated vLLM/SGLang command preserves the selected TP and
DP layout. For two Sparks or multiple DGX systems, `kvfit` refuses to pretend it
can provision the remote workers and router. Use `mode = "attach"` with the
cluster's OpenAI-compatible endpoint, or provide a version-specific
`server_command` array. `examples/calibration-dual-spark.toml` shows two
GPT-OSS replicas behind one dual-Spark endpoint (`tp = 1`, planner DP = 2). For
a model sharded across both Sparks, select `tp = 2` and attach to that TP=2
deployment instead.

Custom `server_command` and `benchmark_command` arrays support placeholders:
`{python}`, `{engine}`, `{base_url}`, `{host}`, `{port}`, `{model}`,
`{served_model}`, `{tokenizer}`, `{revision}`, `{context}`, `{input_tokens}`,
`{output_tokens}`, `{concurrency}`, `{num_prompts}`, `{result_file}`, `{seed}`,
`{tensor_parallel}`, `{data_parallel}`, `{utilization}`, and
`{max_concurrency}`. This is the escape hatch for Ray, Slurm, SSH, containers,
SGLang Model Gateway, or an engine CLI whose installed version differs from the
checked defaults.

The complete `[calibration]` schema also accepts `base_url`, `host`, `port`,
`server_args`, `startup_timeout`, `request_timeout`, `api_key_env`,
`stop_on_failure`, `keep_server`, and `allow_cpu`. Secrets are read from the
named environment variable and redacted from reports. `allow_cpu` exists for
deliberate tests, not production qualification. Attach mode records endpoint
metadata when exposed, but labels cache precision and topology unverified when
the server cannot report them. Exit code 4 means the sweep ran but did not fully
calibrate; configuration and launch errors use exit code 2.

Weight quantization still comes from the exact checkpoint and is independently
checked against the installed engine. Managed launch separately passes the
resolved KV-cache dtype using that engine's spelling. For DeepSeek V4 with an
FP4 sparse-index cache, current vLLM receives
`--attention-config '{"use_fp4_indexer_cache": true}'`. No equivalent SGLang
flag is assumed: use an explicitly checked `server_args`/`server_command` or
attach to a qualified server when the installed SGLang version supports that
path.

For a fleet/model decision rather than one deployment, use the strict audit
schema. The bundled frontier has 28 checkpoint/quantization variants from 13
families, 13 named accelerator presets, two context points per checkpoint, and
50 one-/multi-device fleet shapes:

```bash
uv run kvfit-audit examples/enterprise-frontier-2026.toml
uv run kvfit-audit examples/enterprise-frontier-2026.toml --json
```

The audit fetches each model live, records its resolved Hugging Face commit,
runs every model/context against every hardware/count, and exits 1 when the TOML
coverage thresholds are missed. A family counts as supported only when every
required variant and context succeeds. Every cache result must also match a
separate formula implementation; topology arithmetic and monotonicity across
context, dtype width, and accelerator memory are checked over the final matrix.
The reported percentage is equal-weight coverage of the explicitly named TOML
frontier, not an unsupported market-share claim.

Add `[[engines]]` tables to the same audit TOML to check an installed vLLM or
SGLang version for every successful model scenario. The fields are `name`,
`python`, `probe`, `timeout`, and `required`. Relative Python paths resolve from
the TOML directory. Keep `required = false` when surveying an old or
platform-incomplete environment; set it to true for a deployment gate.

The audit schema also accepts `[[systems]]` with `preset` and `node_counts`.
For example, `examples/dgx-long-horizon.toml` evaluates exact 90,000- and
131,072-token contexts across one, two, and eight complete DGX systems. Each
row includes `concurrency.recommended`, `best_scale_up_local`, and
`best_any_fabric`, so a cross-node arithmetic result cannot masquerade as a
validated local layout.

## Check the installed serving engine

`kvfit` can ask the exact vLLM or SGLang installation in a Python environment
whether it recognizes the checkpoint architecture, weight quantization, and
requested KV/index-cache formats on the local hardware backend:

```bash
uv run kvfit openai/gpt-oss-120b \
  --check-engine all \
  --engine-python /path/to/serving-env/bin/python \
  --context 128k
```

No `--hardware` is needed for an engine-only check. Add it when you want the
engine result and memory layouts in the same report:

```bash
uv run kvfit openai/gpt-oss-120b \
  --check-engine vllm \
  --engine-python /path/to/vllm-env/bin/python \
  --hardware h100-80 \
  --gpus 2 \
  --context 128k
```

The default `--engine-probe config` has two levels. It first reads the installed
package's own model and quantization registries, then asks that version to parse
the model config and resolve its implementation. It sets
`trust_remote_code=False`, does not download weight shards, does not allocate the
model, and does not generate a token. Therefore `preflight-pass` means
**config-compatible on the detected backend**, not load-tested.

The probe also reads that installed version's cache-dtype choices. The report
shows both the portable spelling requested from `kvfit` and the exact engine
spelling (for example, BF16 becomes `bfloat16`, SGLang MXFP4 KV becomes
`fp4_e2m1`, and vLLM NVFP4 KV remains `nvfp4`). Prefer the explicit `mxfp4` or
`nvfp4` choices over generic `fp4`: the two formats are not interchangeable.
Config/registry mode proves that the option exists, while `load` passes it into
the engine and is the only mode that exercises the model-specific kernel path.

When you are on the target GPU host and want actual proof, opt into the
destructive smoke test:

```bash
uv run kvfit openai/gpt-oss-120b \
  --check-engine vllm \
  --engine-python /path/to/vllm-env/bin/python \
  --engine-probe load \
  --engine-tp 2 \
  --engine-timeout 1800 \
  --context 128k
```

`load` downloads missing weights, initializes the engine at the requested
context and tensor-parallel size, and requests one generated token. A
`smoke-pass` is therefore much stronger than `preflight-pass`. It can consume
the full accelerator allocation and should not be run beside production work.
The default tensor parallelism is `--engine-tp`, then an explicit `--tp`, then
`--gpus`.

Use `--engine-probe registry` for a faster offline-ish check that does not ask
the engine to fetch and parse the config again. This weaker mode can return
`unknown` when an engine advertises a generic Transformers fallback. For CI,
`--require-engine-pass` exits with status 3 unless every requested engine gets
`preflight-pass`; ordinary reporting still exits successfully for unsupported or
missing engines.

vLLM and SGLang are commonly installed in separate environments. Run the command
once per `--engine-python` in that case; `--check-engine all` checks both packages
only inside the one selected interpreter.

GPT-OSS weight quantization and KV precision are separate. `--kv-dtype auto`
(the default) keeps the official MXFP4 checkpoints at BF16 KV, but detects an
explicit KVFP8 declaration when a checkpoint provides one. Override it only
when the serving backend is configured differently.

Standalone `hf_quant_config.json` files are read when a repository publishes
one. This matters for `thinkingmachines/Inkling-NVFP4`: only routed experts are
NVFP4, hundreds of excluded module patterns remain BF16, and
`kv_cache_quant_algo: none` leaves the attention cache at the model dtype. The
JSON report exposes the weight algorithm, mixed scope, exclusion count, config
source, and independent KV-cache declaration instead of calling the checkpoint
simply BF16.

For a repository with multiple GGUF quantizations, pass the exact Hugging Face
blob URL. Split GGUF variants are summed automatically:

```bash
uv run kvfit \
  https://huggingface.co/unsloth/gpt-oss-120b-GGUF/blob/main/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf \
  --hardware h100-80 \
  --context 128k
```

## What is supported now

- Standard MHA, GQA, and MQA.
- Explicit full/sliding-window layer schedules.
- DeepSeek V2/V3 MLA.
- DeepSeek V3.2 DSA: MLA plus Lightning Indexer state.
- DeepSeek V4: shared K=V, local sliding state, C4 compressed sparse state,
  C128 heavily compressed state, and the C4 indexer. Checked integrated DSpark
  configs add their sliding-window draft KV state automatically.
- GPT-OSS 20B and 120B: checked alternating 128-token sliding/full GQA schedule.
- Thinking Machines Inkling BF16 and NVFP4: 11 full-attention layers, 55
  512-token sliding layers, and all four short-convolution history streams.
- Qwen3-Next and Qwen3.6: full-attention KV plus gated-delta short-convolution
  and FP32 recurrent matrices.
- Nemotron-H / Nemotron 3: explicit attention/Mamba-2/expert schedule, attention
  KV, Mamba convolution history, and FP32 temporal state.
- Gemma 4: distinct global/local KV head counts and head dimensions, sliding
  windows, and cross-layer KV sharing when declared.
- GLM DSA: latent MLA state plus its index-key cache.
- MiniMax M3: full GQA history plus the sparse-attention index-key side cache;
  sparse selection does not imply KV eviction.
- Replicated tensor-parallel groups on identical accelerators.

Unadapted hybrid/recurrent state such as generic Mamba or Jamba is detected but
rejected until its exact state shapes and schedule are checked.

Unknown model types and configs that require custom remote model code are also
rejected by default. This is intentional: a new architecture can expose familiar
field names while storing very different inference state.

## What the number means

`kvfit` reports a **static counted-state upper bound**, not production
concurrency or a runtime OOM guarantee. A result of
"four full-context sequences fit" does not promise that four simultaneous
requests satisfy a latency or throughput objective.

Multi-GPU estimates currently assume ideal weight sharding inside each
tensor-parallel group and full model replication across data-parallel groups.
Cache components shard independently when an architecture has different global
and local KV-head counts (Inkling uses 8 and 16 respectively). Inkling's sconv
history stays BF16 even when its attention cache is explicitly sized at a lower
precision.
Qwen gated-delta and Nemotron-H recurrent matrices stay FP32 in the checked
reference configurations. The capacity estimate counts one live recurrent
state per sequence. Engine prefix caching can retain additional recurrent-state
checkpoints per cache block; that engine mode is not inferred and must be
measured on the serving host.
Expert parallelism, pipeline parallelism, context parallelism, offload, runtime
allocator page rounding, CUDA graphs, and framework-specific workspaces are not
silently inferred. The chosen `--utilization` is the explicit budget for those
unmodeled costs, but it remains an assumption until checked against target-host
memory state. This matters especially on DGX Spark, where CPU, GPU, and other
engines share unified memory and allocatable memory changes with host and swap
state. See the
[pinned external DGX Spark validation](https://github.com/teilomillet/kvfit/blob/main/docs/dgx-spark.md#external-measured-cross-check).

The installed-engine preflight is deliberately separate from this topology
arithmetic. An engine can accept an architecture and quantization while still
OOMing during weight load, lacking a required kernel at runtime, or failing a
real request. `--engine-probe load` closes that launch-time gap on the target GPU
host, but one successful token still does not validate production throughput,
latency, long-context accuracy, or stability under concurrency.

Repository artifact sizes are used as a weight-footprint proxy unless
`--weight-gib` is supplied. Runtime packing can differ, especially for
quantized checkpoints.

For GPT-OSS, the planner converts OpenAI's approximate decimal-GB runtime
guidance into conservative binary planning floors: 16 GiB for compact 20B
checkpoints, 48 GiB for a 20B BF16 checkpoint, and 60 GiB for compact 120B
checkpoints. These are planning floors, not measured guarantees. Safetensors
GPT-OSS checkpoints are rejected when their index does not cover every declared
layer. GGUF tensor coverage is not yet parsed, so that limitation is shown as a
warning.

## References used as test oracles

- [Hugging Face cache documentation](https://huggingface.co/docs/transformers/kv_cache)
- [Hugging Face DeepSeek V4 architecture](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/deepseek_v4.md)
- [vLLM DeepSeek V4 cache arithmetic](https://vllm.ai/blog/2026/04/24/deepseek-v4.html#the-math-behind-deepseek-v4s-attention-mechanism)
- [Pinned vLLM DeepSeek V4 DSpark cache implementation](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/models/deepseek_v4/nvidia/dspark.py)
- [DeepSeek V4 Flash DSpark checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
- [OpenAI GPT-OSS reference implementation](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py)
- [OpenAI GPT-OSS runtime memory guidance](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers#pick-your-model)
- [vLLM Inkling architecture and sconv cache design](https://vllm.ai/blog/2026/07/15/inkling)
- [vLLM Inkling sconv cache implementation](https://github.com/vllm-project/vllm/blob/main/vllm/models/inkling/nvidia/sconv_swa_attn.py)
- [vLLM serving benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/)
- [vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [SGLang serving benchmark](https://github.com/sgl-project/sglang/blob/main/docs/developer_guide/bench_serving.md)
- [SGLang server arguments](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/server_arguments.md)
- [Thinking Machines Inkling release](https://thinkingmachines.ai/news/introducing-inkling/)
- [vLLM recurrent and gated-delta state shapes](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py)
- [vLLM Nemotron-H state integration](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h.py)
- [Transformers Gemma 4 cache geometry](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
- [vLLM MiniMax M3 indexed cache](https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/nvidia/model.py)
- [NVIDIA DGX Spark specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark unified-memory reporting guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture)
- [NVIDIA connecting two DGX Sparks](https://build.nvidia.com/spark/connect-two-sparks)
- [NVIDIA vLLM on two DGX Sparks](https://build.nvidia.com/spark/vllm/stacked-sparks)
- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
- [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
- [NVIDIA accelerator memory table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html)
- [NVIDIA HGX H200/B200/B300 memory](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
- [AMD Instinct MI300/MI325 memory](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
- [AMD Instinct MI350/MI355 memory](https://www.amd.com/en/products/accelerators/instinct/mi350.html)
