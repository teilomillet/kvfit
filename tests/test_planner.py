from __future__ import annotations

from kvfit.architectures import estimate_cache
from kvfit.hardware import parse_hardware
from kvfit.models import GIB, CacheComponent, CacheEstimate
from kvfit.planner import plan_topologies, summarize_concurrency


def test_gqa_cache_shards_across_tp_heads() -> None:
    cache = estimate_cache(
        {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
        },
        context_tokens=128 * 1024,
    )

    layouts = plan_topologies(
        cache,
        weight_bytes=16 * GIB,
        hardware=parse_hardware("h100"),
        gpus=2,
        utilization=0.9,
    )

    tp1, tp2 = layouts
    assert (tp1.tensor_parallel, tp1.data_parallel, tp1.total_sequences) == (1, 2, 6)
    assert (tp2.tensor_parallel, tp2.data_parallel, tp2.total_sequences) == (2, 1, 8)
    assert tp2.kv_per_sequence_per_rank_bytes == cache.total_bytes / 2


def test_mqa_cache_is_replicated_when_tp_exceeds_one_head() -> None:
    cache = estimate_cache(
        {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "head_dim": 128,
        },
        context_tokens=128 * 1024,
    )

    layout = plan_topologies(
        cache,
        weight_bytes=16 * GIB,
        hardware=parse_hardware("h100"),
        gpus=2,
        utilization=0.9,
        tensor_parallel=2,
    )[0]

    assert layout.kv_per_sequence_per_rank_bytes == cache.total_bytes


def test_query_head_divisibility_is_reported() -> None:
    cache = estimate_cache(
        {
            "model_type": "llama",
            "num_hidden_layers": 4,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "head_dim": 64,
        },
        context_tokens=4096,
    )

    layout = plan_topologies(
        cache,
        weight_bytes=GIB,
        hardware=parse_hardware("custom:8"),
        gpus=4,
        utilization=0.9,
        tensor_parallel=4,
    )[0]

    assert layout.verdict == "unsupported"
    assert "not divisible" in layout.reason


def test_components_use_their_own_tensor_parallel_width() -> None:
    cache = CacheEstimate(
        architecture="mixed",
        context_tokens=1,
        components=(
            CacheComponent("global", 80, "test", tp_parallel_units=8),
            CacheComponent("local", 160, "test", tp_parallel_units=16),
        ),
        kv_parallel_heads=8,
        query_heads=64,
        confidence="test",
        reference="test",
    )

    layout = plan_topologies(
        cache,
        weight_bytes=0,
        hardware=parse_hardware("custom:1"),
        gpus=16,
        utilization=1,
        tensor_parallel=16,
    )[0]

    assert layout.kv_per_sequence_per_rank_bytes == 20


def test_multi_dgx_auto_topologies_respect_scale_up_domain_boundaries() -> None:
    cache = estimate_cache(
        {
            "model_type": "llama",
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 64,
        },
        context_tokens=90_000,
    )

    layouts = plan_topologies(
        cache,
        weight_bytes=16 * GIB,
        hardware=parse_hardware("h100"),
        gpus=16,
        utilization=0.9,
        scale_up_domain_accelerators=8,
    )

    assert [layout.tensor_parallel for layout in layouts] == [1, 2, 4, 8, 16]
    assert all(not layout.cross_domain_tensor_parallel for layout in layouts[:-1])
    assert layouts[-1].cross_domain_tensor_parallel is True
    assert layouts[-1].tensor_parallel_domains == 2


def test_concurrent_users_account_for_agent_parallel_sequences() -> None:
    cache = CacheEstimate(
        architecture="test",
        context_tokens=90_000,
        components=(CacheComponent("kv", GIB, "test", tp_parallel_units=8),),
        kv_parallel_heads=8,
        query_heads=8,
        confidence="test",
        reference="test",
    )
    layouts = plan_topologies(
        cache,
        weight_bytes=8 * GIB,
        hardware=parse_hardware("h100"),
        gpus=16,
        utilization=0.9,
        scale_up_domain_accelerators=8,
    )

    one_sequence = summarize_concurrency(layouts, active_sequences_per_user=1)
    two_sequences = summarize_concurrency(layouts, active_sequences_per_user=2)

    assert one_sequence["qualification"] == "memory-only"
    assert two_sequences["recommended"]["concurrent_users"] == (
        one_sequence["recommended"]["active_sequences"] // 2
    )
    assert one_sequence["best_scale_up_local"]["layout"]["cross_domain_tensor_parallel"] is False
    assert "static counted-state upper bound" in one_sequence["definition"]
    assert "not an OOM guarantee" in one_sequence["definition"]
