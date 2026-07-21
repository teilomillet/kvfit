# kvfit agent guidance

When a user asks whether a Hugging Face model fits a GPU or DGX topology, how
much KV cache or recurrent state it needs, or which TP/DP layout to use, read and
follow [`skills/kvfit/SKILL.md`](skills/kvfit/SKILL.md).

Preserve kvfit's evidence boundary in code and documentation:

- Verify architecture-specific state instead of assuming the standard KV formula.
- Keep separate GPUs and systems as separate memory domains unless the declared
  serving topology actually shards across them.
- Label static fit as memory-only; require target-host measurement for runtime,
  latency, throughput, or SLO claims.
