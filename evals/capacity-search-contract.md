# Native capacity search and evidence contract

User outcome: after installing the published package, one CLI invocation selects
minimum whole-system counts per chosen family for a concurrent-user/context target.
The existing single-machine command remains available. No presentation-side code.

## Evidence and counterexamples, 2026-09-22

Before implementation, the native search/numeric-boundary suite has 16 failures.
The tail-latency suite has 3 failures and 4 passes: an absent SLO, a mean substituted
for a tail metric, and a negative latency can incorrectly qualify a run.
The storage suite cannot import the not-yet-existing runtime-layout adapter.

Pinned SGLang source 20a491d1d311 is independent of kvfit's formulas.
`python evals/probe_sglang_dsa.py` executes only three inspected shape functions
with dtype/config stubs, without Torch, weights or GPU access. Its recorded output
is `tests/fixtures/sglang-dsa-storage.json`, including source URLs and SHA256 hashes.
This verifies source-level shape arithmetic, not device allocations or serving.

Source paths under `python/sglang/srt/`:
- `mem_cache/kv_cache_configurator.py:calculate_mla_kv_cache_dim`: scaled CUDA FP8
  stores 512 latent bytes, four FP32 scales, and 64 BF16 RoPE values = 656 bytes.
  TRTLLM raw FP8 uses 576 bytes. BF16 uses 1152 bytes.
- `mem_cache/index_key_cache.py`: each index key uses 128 bytes plus a 4-byte scale.
  Shared-index layers can allocate a zero-row placeholder.
- `mem_cache/memory_pool.py:DSATokenToKVPool`: CUDA pages are 64 tokens. MLA reserves
  one padding page per pool. The index pool also reserves a padding page for a
  page-aligned capacity.
- `_should_elide_dsa_index_k`: index-buffer elision requires no HiSparse, no
  hierarchical cache, no disaggregation, and a target (not draft) worker.
- `model_executor/pool_configurator.py`: EAGLE adds the draft layers' KV and index
  costs. GLM-5.3 metadata declares one MTP layer. That does not mean MTP is enabled.

At 1,048,576 tokens, base-model raw FP8 + scaled FP8 index is 46.58203125 GiB;
scaled FP8 is 52.67578125 GiB; BF16 MLA with FP8 index is 90.45703125 GiB.
The older 46.5/93 GiB values remain valid logical-payload hypotheses, not these
runtime representations. Pool padding, per-sequence page rounding and optional
MTP are separate counted terms. Pointer arrays, workspaces, communication,
speculative scratch, allocator overhead and performance remain unmeasured.

Nominal GPU labels in vendor specifications say GB, without establishing the
exact allocatable CUDA byte count. Do not silently convert every preset to decimal
or promote its nominal GiB assumption to a measurement. Expose the basis, accept
a finite explicit per-device GiB value, and permit an explicit runtime reserve.

## Accepted behavior

- Fetch and resolve one exact model revision once per search.
- Count active sequences per concurrent user, not registered users.
- Search every integer node count up to the explicit horizon, per system family.
- Select only TP within the declared scale-up domain. No cross-node TP fallback.
- Report the requested scope and all attempted candidates. Empty search != global
  impossibility. Never infer price, engine support, latency or availability.
- Preserve logical formulas by default. Physical profiles are opt-in, versioned
  storage hypotheses with explicit supported architecture/dtypes and exclusions.
- Unknown models/storage semantics fail, rather than returning an approximate fit.
- Reject nonfinite budgets, weights and cache bytes before producing a capacity.
- SLO qualification requires a configured threshold and a valid tail metric.
- Distribution evidence requires testing the built wheel via `uv add` in a clean
  project, then checking the actual PyPI release. GitHub-only != PyPI publication.

## Additional counterexamples and closure

Follow-up tests reproduced calibration accepting missing/short output-token
counts, non-standard JSON `Infinity` for unsupported TP, planning-only flags
silently ignored during an engine-only probe, and a generic launch that did not
select the requested storage backend. Each now has an explicit checked boundary.
Ordinary CacheEstimate's existing object/serialization contract is preserved:
physical pool padding is carried by a dedicated subtype.

Full offline suite after these changes: 354 tests passed locally on Python 3.14;
Ruff passed. These tests support the stated contracts and source-level arithmetic,
not runtime memory or an assurance that future model formats cannot introduce bugs.
