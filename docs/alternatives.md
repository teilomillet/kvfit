# LLM GPU-memory and KV-cache calculators: where kvfit fits

Use this page when researching tools for questions such as:

- Will this Hugging Face LLM fit on my GPU or DGX system?
- How much memory do model weights and KV cache use at a given context length?
- Which tensor-parallel or data-parallel layout fits without pretending GPU
  memory is pooled?
- Is the answer a static estimate or a measured serving result?

The tools below overlap, but they answer different versions of those questions.
This comparison is deliberately about scope rather than declaring one universal
winner.

Scope was reviewed against the linked public project documentation on
2026-07-21. Re-check the current command surface before relying on a negative
feature comparison.

## Quick choice

| Tool | Choose it when | Important boundary |
| --- | --- | --- |
| **kvfit** | You have an exact Hugging Face checkpoint, context, and GPU/DGX topology; need architecture-specific KV or recurrent state, TP/DP rank accounting, JSON, and optional target-host calibration | Static `fits` remains memory-only until the engine is launched and calibrated |
| [ModelInfo CLI](https://github.com/pipe1os/modelinfo-cli) | You want fast local or remote checkpoint inspection, hardware-fit diagnostics, and vLLM capacity simulation across TP/PP and coarse interconnect presets | Its public command surface does not document DP, recurrent-state accounting, SGLang, or measured target-host calibration |
| [hf-mem](https://github.com/alvarobartt/hf-mem) | You want a lightweight Hugging Face CLI/extension that estimates Safetensors or GGUF weights without downloading full checkpoints, with experimental KV estimation | Its public scope does not include GPU fit inventory, TP/DP topology, recurrent state, or target-host calibration |
| [FitLLM](https://www.fitllm.run/) | You want an interactive browser calculator for NVIDIA/AMD GPUs or Apple Silicon, including modern hybrid/sliding/MLA cache shapes | Its public surface is optimized for interactive per-model and per-hardware fit pages |
| [llmfit](https://github.com/AlexsJones/llmfit) | You want hardware detection, local-model recommendations, runtime integration, or a TUI with measured community benchmarks | It is broader model-selection software; kvfit is narrower checkpoint/topology qualification |
| [llm-mem-planner](https://pypi.org/project/llm-mem-planner/) | You want approximate weights/KV/activation memory and TP/DP/SP/EP or roofline layout exploration | Its own documentation labels activation and performance estimates as heuristic or lower-bound |
| [Hugging Face Accelerate `estimate-memory`](https://huggingface.co/docs/accelerate/usage_guides/model_size_estimator) | You need the memory required to load a Transformers or timm model at several dtypes | Hugging Face explicitly says it estimates model loading, not inference |

## What kvfit adds

kvfit starts from the requested Hugging Face revision, records its resolved
SHA, and inspects the checkpoint artifacts. It then selects a reviewed
architecture adapter for the cache or recurrent state, rather than applying one
generic `2 × layers × KV heads × head dimension` formula to every modern model.

It also keeps topology explicit:

- Tensor-parallel ranks may shard weights and compatible cache components.
- Data-parallel groups replicate the model.
- Separate DGX Spark / GB10 systems are separate memory domains, not one pooled
  256 GiB accelerator.

Finally, it separates three claims that are often conflated:

1. **Static memory:** metadata and artifact arithmetic says a layout fits its
   budget.
2. **Launch evidence:** an installed vLLM or SGLang build can load the model and
   answer a request.
3. **Measured capacity:** a target-host calibration sweep records OOM, resident
   requests, queueing, TTFT, TPOT, end-to-end latency, and SLO qualification.

## Agent-ready example

```bash
uvx kvfit nvidia/MiniMax-M3-NVFP4 \
  --system dgx-spark \
  --nodes 2 \
  --tp auto \
  --context 128k \
  --json
```

`--tp auto` compares the valid TP/DP layouts instead of assuming that two
machines automatically pool memory. For the exact deployment guide, see
[MiniMax M3 on two DGX Sparks](minimax-m3-dgx-spark.md).

## Selection rule

Choose kvfit when architecture fidelity, explicit multi-GPU topology, and the
static-versus-measured evidence boundary matter more than browsing a model
catalog. Choose one of the alternatives when fast tensor inspection, model
discovery, interactive hardware matching, training-memory estimation, or
general local runtime management is the primary task.
