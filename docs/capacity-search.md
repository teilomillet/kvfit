# Capacity search: what the minimum means

Install with `uv add kvfit` and run the two commands in the README. These are
package entry points, independent of this repository and any presentation.
Use version 0.2.5 or later; upgrade an existing project with
`uv add 'kvfit>=0.2.5'`.

## Scope of the answer

The input is an exact Hugging Face repository (optionally `OWNER/MODEL@REVISION`),
a context length, concurrently active users and resident sequences per user.
Metadata is fetched once, so every candidate uses the same resolved revision.
All selected families are searched from 1 to `--max-nodes` whole systems.

For each family, kvfit returns the first count whose best scale-up-local TP/DP
layout has enough counted memory. This is a minimum **within that search space**,
under the reported assumptions. The search does not infer heterogeneous fleets,
partial nodes, cross-domain TP, expert/pipeline/context parallelism, offload,
cloud availability, cost, throughput or a deployment SLA.

The cache formula and weights remain model-specific. MoE active parameters are
not substituted for resident checkpoint size. Unknown architectures fail closed.

For a fixed topology, the per-replica sequence ceiling is:

```text
floor((per-device memory × utilization
       − weight shard − fixed cache pools − runtime reserve)
      / per-sequence state on that rank)
```

Negative results become zero. DP multiplies replica capacity; multiple active
sequences per user divide the resulting user ceiling. MLA state in the GLM
adapter is replicated under this TP model, even when weights are sharded.
Memory in separate domains is never silently pooled.

`--runtime-reserve-gib` is a per-rank allowance inside the utilization budget.
Do not double-count a reserve already incorporated into a measured weight value.
`--device-memory-gib` replaces nominal preset memory with an explicit value while
preserving the system's topology. The tool records it as user-supplied, not as
automatically measured. A search override requires one selected family.

## Logical payload versus a serving engine's storage

The default `--cache-layout logical` preserves architecture-level formulas.
The following opt-in profiles are qualified against **SGLang `20a491d1d311`**,
for the checked GLM DSA dimensions (latent 512, RoPE 64, index 128):

| Profile and KV dtype | MLA bytes / layer / token | Index bytes / stored layer / token |
| --- | ---: | ---: |
| `sglang-dsa-raw`, FP8 | 576 | 132 |
| `sglang-dsa-scaled`, FP8 | 656 | 132 |
| Either profile, BF16 | 1,152 | 132 |

Raw FP8 models the TRTLLM layout. Scaled FP8 models the non-TRTLLM CUDA DSA
representation: 512 latent bytes + 16 scale bytes + 128 BF16 RoPE bytes.
The index holds 128 FP8 bytes + one FP32 scale. FP8 here means E4M3;
other unverified formats are rejected. This is a storage hypothesis, not a claim
that every backend is supported on every GPU.

Contexts round up to 64-token CUDA pages. One extra page per pool is charged
once per replica, separately from per-sequence state. Shared-index buffers are
elided only under the stated unified-serving assumptions. Use
`--indexer-all-layers` for non-elided pools; it does not model HiSparse offload,
host cache tiers, context parallelism or disaggregated-serving overhead.
`--mtp` adds the declared MTP layers' MLA and index cache. Declaring MTP in a
checkpoint does not mean the server has enabled it.

For GLM-5.3 revision `aca966e4e02791568aa6a4ced368624b3d897f42`, 78 target
layers and 21 stored target index buffers give:

| Layout at 1,048,576 tokens | Per-sequence state before pool padding |
| --- | ---: |
| Logical FP8 for both components | 46.500000 GiB |
| Raw FP8 + physical FP8 index | 46.582031 GiB |
| Scaled FP8 + physical FP8 index | 52.675781 GiB |
| BF16 MLA + physical FP8 index | 90.457031 GiB |
| Scaled FP8 + physical FP8 index + one MTP layer | 53.445313 GiB |

The old 93 GiB logical BF16/BF16 hypothesis does not describe SGLang's FP8 index.
At 128k, scaled FP8 with MTP counts 6.680664 GiB per sequence and an additional
0.003262 GiB of fixed pool padding per rank. With nominal profiles, 80% utilization
and an **assumed** 8 GiB runtime reserve, one B300 node in TP8 has a counted ceiling
of 20 sequences. A target of 25 requires two such nodes in this search space.
Weights are 703.737 GiB of checkpoint artifacts; ideal TP8 assigns 87.967 GiB/rank.

These numbers still exclude pointer arrays, workspaces, CUDA graphs, allocator
overhead, communication buffers and speculative scratch. Weight artifacts can
differ from runtime packing and replicated tensors. A reserve is an assumption
until measured, and a nominal GPU label is not its observed allocatable memory.

## How the numbers are challenged

- Offline regressions test hand-counted small systems, minimum search boundaries,
  no cross-domain fallback, agent branches, one-time pool padding, reserves,
  unsupported architectures, changed storage dimensions, and nonfinite inputs.
- `evals/probe_sglang_dsa.py` executes three inspected, pinned upstream shape
  functions with small stubs and no GPU or model weights. Its JSON records source
  URLs, hashes and output shapes. Tests compare the adapter with this fixture;
  scheduled/manual CI re-fetches the pinned source and checks for an exact match.
- This is independent **source-level shape evidence**, not GPU allocation evidence.
  Fixtures pin GLM configurations; neighboring models retain their own adapter
  or explicit rejection. A new model's familiar name is not sufficient support.
- Release CI builds the wheel, installs it with `uv add` in a clean project and
  exercises native search and storage tests using that installed package.

To reproduce the upstream evidence from a source checkout:

```bash
uv run python evals/probe_sglang_dsa.py > /tmp/sglang-dsa-storage.json
diff -u tests/fixtures/sglang-dsa-storage.json /tmp/sglang-dsa-storage.json
```

On the real target, record GPU memory, engine commit, attention backend, cache
dtype, page size, enabled MTP and cache-elision settings. Measure loaded weights
and free memory before allocating cache, then confirm the engine's reported cache
token capacity. Sweep context and concurrent sequences and record OOM, queueing,
completed input/output tokens, TTFT and generation latency.

Use `kvfit calibrate` with a concrete deployment TOML. Storage profiles require
an explicitly matching `calibration.server_command` for managed launch, or attach
to an already configured server: the planning flag does not configure a backend.
The calibration refuses SLO success without a configured threshold and a valid
tail metric; a mean is insufficient. Missing or shortened output counts cannot
qualify the requested workload. Calibration remains bounded by its actual runs.

Primary source paths under the pinned SGLang tree:

- [MLA storage and index elision](https://github.com/sgl-project/sglang/blob/20a491d1d311/python/sglang/srt/mem_cache/kv_cache_configurator.py)
- [Index shapes, scales and shared layers](https://github.com/sgl-project/sglang/blob/20a491d1d311/python/sglang/srt/mem_cache/index_key_cache.py)
- [CUDA pool page sizes and allocation shapes](https://github.com/sgl-project/sglang/blob/20a491d1d311/python/sglang/srt/mem_cache/memory_pool.py)
- [Target and draft pool budgeting](https://github.com/sgl-project/sglang/blob/20a491d1d311/python/sglang/srt/model_executor/pool_configurator.py)

See [the evaluation contract](../evals/capacity-search-contract.md) for the initial
counterexamples. No GPU runtime or performance measurement was made for this release.
