# Can NVIDIA MiniMax M3 NVFP4 fit on two DGX Spark / GB10 systems?

Use kvfit to answer this for the exact checkpoint revision, requested context,
cache precision, and TP/DP layout. This guide intentionally does not hard-code a
yes/no answer: repository artifacts can change, and a static model-fit result
does not prove that the serving engine launches or meets a latency objective.

## Compare both two-node layouts

```bash
uvx kvfit nvidia/MiniMax-M3-NVFP4 \
  --system dgx-spark \
  --nodes 2 \
  --tp auto \
  --context 128k \
  --json
```

`--tp auto` compares the valid layouts:

- **TP=1, DP=2:** one complete model replica per Spark.
- **TP=2, DP=1:** one model sharded across both Sparks.

Two DGX Sparks are separate memory domains. Their two 128 GiB unified-memory
budgets do not become one freely addressable 256 GiB pool. TP=2 is meaningful
only when the serving stack actually shards and communicates across both hosts.

To inspect only the cross-node TP layout, replace `--tp auto` with `--tp 2`.

## What the report checks

For MiniMax M3, kvfit counts:

- the resolved Hugging Face checkpoint artifacts;
- full GQA key/value history;
- the sparse-attention index-key side cache;
- per-rank weights and cache under each TP/DP layout;
- the static counted-state upper bound and explicit fit/OOM reason.

Sparse token selection does not imply that old KV entries are evicted. kvfit
does not subtract KV history merely because the attention index selects a subset
of tokens.

## What a static fit does not prove

A static `fits` verdict does not prove:

- that the installed vLLM or SGLang version supports this exact checkpoint;
- that runtime packing matches repository artifact size;
- that CUDA graphs, allocator pages, workspaces, and communication fit;
- that two-host orchestration and networking are configured;
- that the deployment meets throughput, TTFT, TPOT, or end-to-end SLOs.

Run `kvfit-audit` to inspect the pinned evidence and engine compatibility. Then
run `kvfit-calibrate` on the target CUDA hosts. For two Sparks, calibration must
attach to an already orchestrated OpenAI-compatible endpoint or use explicit
cluster-specific launch commands; kvfit will not benchmark one host and label it
as a two-host measurement.

The repository includes [`examples/calibration-dual-spark.toml`](../examples/calibration-dual-spark.toml)
as the starting point for that measured qualification.
