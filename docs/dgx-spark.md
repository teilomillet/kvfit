# DGX Spark LLM memory calculator: will a Hugging Face model fit GB10?

Use kvfit to check the counted weights and architecture-specific inference state
for an arbitrary Hugging Face checkpoint on one or more NVIDIA DGX Spark / GB10
systems. A static check can prove that counted state is too large, or establish a
reproducible preflight hypothesis. Only a target-host launch can prove runtime fit.

```bash
uvx kvfit OWNER/MODEL \
  --system dgx-spark \
  --nodes 1 \
  --tp auto \
  --context 256k \
  --json
```

For two Sparks, use `--nodes 2`. `--tp auto` compares TP=1/DP=2 with
cross-host TP=2/DP=1 instead of pretending that the two systems form one freely
addressable 256 GB pool.

## Accuracy contract

| Claim | Evidence required | What kvfit may say |
| --- | --- | --- |
| Spark hardware | Current NVIDIA specification | 128 GB unified system memory; CPU, GPU, and other engines share it |
| Model inputs | Exact Hugging Face repository and resolved commit | Artifact bytes and state-shape fields at that revision |
| Cache or recurrent state | Architecture implementation plus an independent formula oracle | Logical bytes per active sequence |
| Static topology | Per-rank TP/DP arithmetic and an explicit memory budget | Counted state fits or does not fit before unmodeled runtime overhead |
| Runtime fit | Weight-load/request evidence on the target Spark(s) | This pinned engine and deployment launched successfully |
| Throughput or latency | Warm repeated target-host measurements | Results only for the measured versions, context, concurrency, and topology |

The first four rows are reproducible static evidence. They are not allowed to
silently become the last two. NVIDIA documents that DGX Spark uses unified
memory and that allocatable memory varies with host and swap state, so a fixed
VRAM-style budget is not an OOM guarantee.

Every Spark hardware fact needs an official hardware source; community results
are independent measured checks, not substitutes for the platform specification.

Primary platform sources:

- [NVIDIA DGX Spark hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA unified-memory reporting guidance](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html#guidance-for-reporting-memory-resources-with-unified-memory-architecture)
- [NVIDIA Spark stacking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)

## External measured cross-check

The repository keeps one external hardware regression against
[HowToSpark's Qwen3.6 35B-A3B NVFP4 recipe](https://howtospark.com/recipes/qwen3-6-35b-a3b-nvfp4-fast).
The comparison is pinned to Hugging Face revision
[`1c3f884bc99aac2524f6d49bcbac8c88401afd66`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast/tree/1c3f884bc99aac2524f6d49bcbac8c88401afd66)
and was rerun on 2026-07-21.

| Quantity | kvfit static result | HowToSpark measured result | Difference |
| --- | ---: | ---: | ---: |
| Checkpoint artifacts vs loaded weights | 22.0248 GiB | 22.15 GiB | -0.57% |
| No-draft state at 262,144 tokens | 2.5593 GiB | 2.5800 GiB implied by 4 GiB / 406,424 tokens | -0.80% |

The sub-1% agreement validates this pinned state calculation. It does **not**
validate every checkpoint or every vLLM version. The same recipe reports a
measured 29.1 full-context multiples in its uncapped KV pool, while kvfit's
default static budget reports a larger counted-state upper bound because it
does not model that run's allocator, graphs, draft layer, OS use, and other
runtime overhead. That disagreement is expected and is why kvfit no longer
calls the static number an OOM ceiling.

The offline regression is `tests/test_dgx_spark_evidence.py`. Re-run the live,
revision-pinned formula and topology audit with:

```bash
uvx --from kvfit kvfit-audit examples/dgx-spark-evidence.toml --json
```

The public
[`DGX Spark evidence` workflow](../.github/workflows/dgx-spark-evidence.yml)
runs the offline regression on every change and reruns the live pinned audit
weekly. Its JSON report is uploaded as a workflow artifact. A green workflow
means the pinned evidence contract passed; it still does not claim that a
different checkpoint, engine build, or Spark host will behave identically.

## kvfit and HowToSpark answer different halves

[HowToSpark](https://howtospark.com/) publishes measured DGX Spark recipes,
commands, memory observations, TTFT, prefill, and decode results. kvfit handles
the earlier arbitrary-checkpoint question: resolve a Hugging Face revision,
count its architecture-specific state, and compare explicit one- or two-Spark
layouts. Use kvfit to form the deployment hypothesis, then use a HowToSpark
recipe when one matches exactly or run `kvfit-calibrate` yourself.

## Runtime qualification

Before trusting a deployment answer:

1. Run the static command with the exact model revision and context.
2. Run `kvfit-audit` and require the independent oracle to pass.
3. Probe the installed vLLM or SGLang build on the intended Spark.
4. Launch the exact checkpoint and record loaded weight and cache allocation.
5. Warm the server, sweep concurrency, and record OOM, TTFT, TPOT, throughput,
   completion, and queueing separately.
6. For two Sparks, record the DGX OS, driver, engine, NCCL, network interface,
   cable/fabric, and TP orchestration on both systems.

`examples/calibration-dgx-spark.toml` and
`examples/calibration-dual-spark.toml` are starting points. A result from one
Spark must never be relabeled as a two-Spark measurement.

## Discoverability is also tested, not assumed

The neutral, brand-blind query corpus and acceptance threshold live in
[`evals/dgx-spark-discovery.json`](../evals/dgx-spark-discovery.json). Publication
is not counted as organic discovery until fresh research agents find kvfit in at
least four of five queries, within the first ten results, on three different
dates, without searching for `kvfit` or its owner.

Baseline on 2026-07-21: a fresh agent ran 24 neutral searches and found kvfit in
zero. It did validate the product gap, but the discoverability gate remains a
hard fail until future blinded runs meet the threshold.
