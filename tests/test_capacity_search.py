from __future__ import annotations

import json
from dataclasses import replace

import pytest

from kvfit.cli import main
from kvfit.hardware import Hardware, parse_hardware
from kvfit.models import GIB, CacheComponent, CacheEstimate, ModelMetadata
from kvfit.planner import plan_topologies


def tiny_metadata():
    # 1 layer * 1 KV head * (K+V) * dimension 1 * BF16 * 2**28 tokens = 1 GiB.
    return ModelMetadata(
        repo_id="test/exact",
        requested_revision="main",
        resolved_revision="pinned",
        config={
            "model_type": "llama",
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 1,
            "hidden_size": 8,
            "torch_dtype": "bfloat16",
        },
        weight_bytes=8 * GIB,
        weight_source="fixture",
    )


def test_native_search_uses_one_metadata_fetch_and_first_sufficient_node(monkeypatch, capsys):
    from kvfit.systems import System

    calls = []
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *a, **k: calls.append(a) or tiny_metadata()
    )
    monkeypatch.setattr("kvfit.cli.parse_hardware", lambda _: Hardware("test", "test", 5))
    monkeypatch.setattr(
        "kvfit.cli.parse_system",
        lambda name: System(name, name, "test", 2, 2, "node", "local", "remote", "fixture", ""),
    )
    code = main(
        [
            "test/exact",
            "--systems",
            "test",
            "--max-nodes",
            "3",
            "--concurrent-users",
            "3",
            "--context",
            str(2**28),
            "--utilization",
            "1",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    # TP2: each GPU has 4 GiB weights + 1 GiB/cache, one sequence per replica.
    assert len(calls) == 1
    assert payload["search"]["minimums"][0]["nodes"] == 3
    assert payload["search"]["minimums"][0]["capacity"]["concurrent_users"] == 3
    assert [r["nodes"] for r in payload["search"]["attempts"]] == [1, 2, 3]
    assert payload["search"]["qualification"] == "memory-only"


def test_search_does_not_substitute_cross_node_tp(monkeypatch, capsys):
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata",
        lambda *a, **k: replace(tiny_metadata(), weight_bytes=1000 * GIB),
    )
    code = main(
        [
            "test/exact",
            "--systems",
            "dgx-h100",
            "--max-nodes",
            "2",
            "--concurrent-users",
            "1",
            "--context",
            "128k",
            "--json",
        ]
    )
    assert code == 1
    p = json.loads(capsys.readouterr().out)
    assert p["search"]["minimums"] == []
    assert p["search"]["not_found"] == ["dgx-h100"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--concurrent-users", "0"],
        ["--max-nodes", "0"],
        ["--system", "dgx-b300"],
        ["--hardware", "h200"],
        ["--gpus", "8"],
        ["--nodes", "2"],
        ["--tp", "8"],
        ["--check-engine", "sglang"],
    ],
)
def test_invalid_search_is_rejected_before_network(monkeypatch, capsys, extra):
    def no_network(*a, **k):
        pytest.fail("invalid search reached metadata network boundary")

    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", no_network)
    assert main(["test/exact", "--systems", "dgx-b300", "--concurrent-users", "1", *extra]) == 2


def test_runtime_reserve_and_measured_memory_override_change_capacity(monkeypatch, capsys):
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *a, **k: tiny_metadata())
    base = [
        "test/exact",
        "--system",
        "dgx-h200",
        "--tp",
        "8",
        "--context",
        str(2**28),
        "--device-memory-gib",
        "5",
        "--utilization",
        "1",
        "--runtime-reserve-gib",
        "1",
        "--json",
    ]
    assert main(base) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["layouts"][0]["sequences_per_replica"] == 3  # 5 - 1 weight - 1 reserve
    assert p["hardware"]["memory_basis"] == "user-supplied-gib"
    assert p["layouts"][0]["runtime_reserve_per_rank_gib"] == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_custom_memory_is_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        parse_hardware(f"custom:{value}")


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_nonfinite_planner_weights_are_rejected(weight):
    cache = CacheEstimate("test", 1, (CacheComponent("kv", 1, "fixture"),), 1, 8, "test", "fixture")
    with pytest.raises(ValueError, match="finite"):
        plan_topologies(
            cache, weight_bytes=weight, hardware=Hardware("test", "test", 8), gpus=1, utilization=1
        )
