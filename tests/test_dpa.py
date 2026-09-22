from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kvfit.dpa import DpaWeights, classify_dpa_weights
from kvfit.hardware import Hardware, parse_hardware
from kvfit.models import GIB, CacheComponent, StorageCacheEstimate
from kvfit.planner import plan_topologies, search_systems
from kvfit.systems import parse_system


def tiny_checkpoint(*, experts=4, layers=2, quantized=False):
    config = {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": layers,
        "num_nextn_predict_layers": 0,
        "n_routed_experts": experts,
        "first_k_dense_replace": 0,
        "hidden_size": 128,
        "moe_intermediate_size": 128,
        "dtype": "bfloat16",
    }
    if quantized:
        config["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
    tensors = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 128]}}
    for layer in range(layers):
        for expert in range(experts):
            for proj in ["gate", "up", "down"]:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
                tensors[key + ".weight"] = {
                    "dtype": "F8_E4M3" if quantized else "BF16",
                    "shape": [128, 128],
                }
                if quantized:
                    tensors[key + ".weight_scale_inv"] = {"dtype": "F32", "shape": [1, 1]}
    offset = 0
    for tensor in tensors.values():
        size = (
            tensor["shape"][0]
            * tensor["shape"][1]
            * {
                "BF16": 2,
                "F32": 4,
                "F8_E4M3": 1,
            }[tensor["dtype"]]
        )
        tensor["data_offsets"] = [offset, offset + size]
        offset += size
    return config, tensors


@pytest.mark.parametrize("experts,layers", [(2, 1), (4, 2), (8, 3)])
@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_split_counts_scales_and_replicated_tensors(experts, layers, quantized):
    config, tensors = tiny_checkpoint(experts=experts, layers=layers, quantized=quantized)
    weights = classify_dpa_weights(config, tensors, source="independent tiny checkpoint")
    per_matrix = 128 * 128 * (1 if quantized else 2) + (4 if quantized else 0)
    assert weights.routed_expert_bytes == layers * experts * 3 * per_matrix
    assert weights.replicated_bytes == 512
    assert weights.expert_count == experts


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_weight",
        "extra_expert",
        "bad_shape",
        "bad_dtype",
        "bad_offsets",
        "missing_scale",
        "unequal_experts",
        "unknown_expert_tensor",
        "unqualified_model",
        "unqualified_quantization",
        "wrong_layer_schedule",
        "noninteger_experts",
    ],
)
def test_unknown_or_incomplete_weight_placement_fails_closed(mutation):
    config, tensors = tiny_checkpoint(quantized=True)
    key = "model.layers.0.mlp.experts.0.down_proj.weight"
    if mutation == "missing_weight":
        del tensors[key]
    elif mutation == "extra_expert":
        tensors[key.replace("experts.0", "experts.4")] = tensors[key]
    elif mutation == "bad_shape":
        tensors[key]["shape"] = [64, 256]
    elif mutation == "bad_dtype":
        tensors[key]["dtype"] = "F4"
    elif mutation == "bad_offsets":
        tensors[key]["data_offsets"] = [9, 0]
    elif mutation == "missing_scale":
        del tensors[key + "_scale_inv"]
    elif mutation == "unequal_experts":
        tensors[key] = {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 32768]}
        del tensors[key + "_scale_inv"]
    elif mutation == "unknown_expert_tensor":
        tensors[key.replace("down_proj.weight", "new_state")] = tensors[key]
    elif mutation == "unqualified_model":
        config["model_type"] = "deepseek_v4"
    elif mutation == "unqualified_quantization":
        config["quantization_config"]["quant_method"] = "modelopt"
    elif mutation == "wrong_layer_schedule":
        config["first_k_dense_replace"] = 1
    else:
        config["n_routed_experts"] = 4.5
    with pytest.raises(ValueError):
        classify_dpa_weights(config, tensors, source="mutated checkpoint")


def cache(state_gib=1, padding_gib=0):
    return StorageCacheEstimate(
        "glm-dsa-mla",
        1024,
        (CacheComponent("state", state_gib * GIB, "fixture"),),
        1,
        8,
        "upstream-storage-formula",
        "fixture",
        fixed_components=(CacheComponent("padding", padding_gib * GIB, "fixture"),),
    )


def test_dpa_distinguishes_attention_groups_from_full_replicas():
    kwargs = dict(
        weight_bytes=10 * GIB,
        hardware=Hardware("test", "test", 7),
        gpus=16,
        utilization=1,
        tensor_parallel=8,
        runtime_reserve_bytes=GIB,
    )
    ordinary = plan_topologies(cache(padding_gib=1), **kwargs)
    layouts = plan_topologies(
        cache(padding_gib=1), dpa_weights=DpaWeights(8 * GIB, 2 * GIB, 8, "fixture"), **kwargs
    )
    assert ordinary == tuple(x for x in layouts if x.strategy == "tp")
    by_dpa = {x.attention_data_parallel: x for x in layouts if x.strategy == "sglang-dpa"}
    for dpa in [2, 4, 8]:
        layout = by_dpa[dpa]
        # Per rank: 1 GiB experts + 2 GiB other weights + 1 padding + 1 reserve.
        assert layout.weights_per_rank_bytes == 3 * GIB
        assert layout.kv_per_sequence_per_rank_bytes == GIB  # Never divide MLA by DPA.
        assert layout.sequences_per_attention_replica == 2
        assert layout.sequences_per_replica == 2 * dpa
        assert layout.total_sequences == 2 * dpa * 2
        assert layout.data_parallel == 2  # Full model groups, not DPA.
        assert layout.attention_tensor_parallel == 8 // dpa
        assert layout.expert_parallel == 8


def test_dpa_uses_attention_head_divisibility_and_expert_ownership():
    c = replace(cache(), query_heads=2)
    layouts = plan_topologies(
        c,
        weight_bytes=8 * GIB,
        hardware=Hardware("x", "x", 8),
        gpus=8,
        tensor_parallel=8,
        utilization=1,
        dpa_weights=DpaWeights(6 * GIB, 2 * GIB, 8, "fixture"),
    )
    assert layouts[0].verdict == "unsupported"  # TP8 cannot partition two heads.
    assert {
        x.attention_data_parallel
        for x in layouts
        if x.strategy == "sglang-dpa" and x.verdict == "fits"
    } == {4, 8}
    layouts = plan_topologies(
        cache(),
        weight_bytes=8 * GIB,
        hardware=Hardware("x", "x", 8),
        gpus=8,
        tensor_parallel=8,
        utilization=1,
        dpa_weights=DpaWeights(6 * GIB, 2 * GIB, 6, "fixture"),
    )
    assert not any(x.strategy == "sglang-dpa" and x.verdict == "fits" for x in layouts)


def test_glm53_pinned_evidence_yields_four_local_b300_groups():
    evidence = json.loads((Path(__file__).parent / "fixtures" / "sglang-dpa.json").read_text())
    weights = DpaWeights(**evidence["weights"])
    c = cache(54728 / 1024, 54728 * 64 / GIB)
    result = search_systems(
        c,
        weight_bytes=755632050320,
        targets=[(parse_system("dgx-b300"), parse_hardware("b300"))],
        max_nodes=4,
        concurrent_users=25,
        active_sequences_per_user=2,
        utilization=0.8,
        runtime_reserve_bytes=8 * GIB,
        dpa_weights=weights,
    )
    row = result["minimums"][0]
    assert row["nodes"] == 4
    assert row["capacity"]["layout"]["strategy"] == "sglang-dpa"
    assert row["capacity"]["active_sequences"] == 64
    assert result["attempts"][-2]["capacity"]["active_sequences"] == 48
    assert "DP-Attention" in result["scope"]


def test_dpa_rejects_unqualified_cache():
    with pytest.raises(ValueError, match="GLM"):
        plan_topologies(
            replace(cache(), architecture="deepseek-v4-compressed-hybrid"),
            weight_bytes=8 * GIB,
            hardware=Hardware("x", "x", 8),
            gpus=8,
            utilization=1,
            dpa_weights=DpaWeights(6 * GIB, 2 * GIB, 8, "fixture"),
        )


@pytest.mark.parametrize("values", [(float("nan"), 1, 8), (1, -1, 8), (1, 1, 0), (1, 1, True)])
def test_invalid_weight_evidence_is_rejected(values):
    with pytest.raises(ValueError):
        DpaWeights(*values, "invalid")


@pytest.mark.parametrize(
    "extra",
    [
        ["--weight-gib", "100"],
        ["--check-engine", "sglang"],
    ],
)
def test_dpa_rejects_ambiguous_weight_override_or_unconfigured_probe(monkeypatch, capsys, extra):
    from kvfit.cli import main

    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *a, **k: pytest.fail("network"))
    assert (
        main(
            [
                "test/glm",
                "--system",
                "dgx-b300",
                "--parallelism",
                "sglang-dpa",
                "--cache-layout",
                "sglang-dsa-scaled",
                *extra,
            ]
        )
        == 2
    )


def test_dpa_calibration_cannot_silently_launch_tp(tmp_path):
    from test_calibrate import _prediction

    from kvfit.calibrate import _default_server_command
    from kvfit.calibration_config import load_calibration_config

    path = tmp_path / "deployment.toml"
    path.write_text('model="test/glm"\nhardware="b300"\n[calibration]\nengine="sglang"\n')
    config = load_calibration_config(path)
    prediction = copy.deepcopy(_prediction())
    prediction["concurrency"]["recommended"]["layout"]["strategy"] = "sglang-dpa"
    with pytest.raises(ValueError, match=r"DPA|DP-Attention"):
        _default_server_command(config, prediction)


def test_attention_groups_match_recorded_upstream_execution():
    evidence = json.loads((Path(__file__).parent / "fixtures/sglang-dpa.json").read_text())
    layouts = plan_topologies(
        cache(),
        weight_bytes=10 * GIB,
        hardware=Hardware("test", "test", 20),
        gpus=8,
        utilization=1,
        tensor_parallel=8,
        dpa_weights=DpaWeights(8 * GIB, 2 * GIB, 8, "fixture"),
    )
    for layout in layouts:
        observed = evidence["world_info"][str(layout.attention_data_parallel)]
        assert {row[1] for row in observed} == {layout.as_dict()["attention_tensor_parallel"]}
        assert len({row[2] for row in observed}) == layout.attention_data_parallel


def test_local_search_does_not_infer_cross_node_expert_groups():
    weights = DpaWeights(684 * GIB, 20 * GIB, 256, "fixture")
    result = search_systems(
        cache(54),
        weight_bytes=704 * GIB,
        targets=[(parse_system("dgx-h200"), parse_hardware("h200"))],
        max_nodes=4,
        concurrent_users=25,
        active_sequences_per_user=2,
        utilization=0.8,
        runtime_reserve_bytes=8 * GIB,
        dpa_weights=weights,
    )
    assert result["minimums"] == []
    assert not any(
        layout["expert_parallel"] > 8
        for row in result["attempts"]
        for layout in row["layouts"]
        if layout["strategy"] == "sglang-dpa"
    )


@pytest.mark.parametrize("neighbor", ["zai-org--GLM-5.1", "zai-org--GLM-5.2"])
def test_qualification_uses_architecture_fields_not_glm53_constants(neighbor):
    from kvfit.dpa import qualify_dpa_config

    data = json.loads(
        (Path(__file__).parent / f"fixtures/model_configs/{neighbor}.json").read_text()
    )
    config = data["config"]
    count, hidden, width, layers = qualify_dpa_config(config)
    assert count == config["n_routed_experts"]
    assert hidden == config["hidden_size"]
    assert width == config["moe_intermediate_size"]
    assert len(layers) == config["num_hidden_layers"] - config[
        "first_k_dense_replace"
    ] + config.get("num_nextn_predict_layers", 0)


@pytest.mark.parametrize("profile", ["sglang-dsa-scaled", "sglang-dsa-raw"])
@pytest.mark.parametrize(
    "context,dtype", [(64, "fp8"), (65, "fp8"), (131072, "bf16"), (1048576, "fp8")]
)
def test_dpa_consumes_actual_cache_profile_without_a_fixed_context_multiplier(
    profile, context, dtype
):
    from kvfit.architectures import estimate_cache
    from kvfit.cache_layout import apply_cache_layout

    cfg = json.loads(
        (Path(__file__).parent / "fixtures/model_configs/zai-org--GLM-5.3.json").read_text()
    )["config"]
    base = estimate_cache(cfg, context_tokens=context, kv_bytes=1 if dtype == "fp8" else 2)
    stored = apply_cache_layout(base, cfg, profile=profile, kv_dtype=dtype, mtp=True)
    layouts = plan_topologies(
        stored,
        weight_bytes=755632050320,
        hardware=parse_hardware("b300"),
        gpus=8,
        tensor_parallel=8,
        utilization=0.8,
        runtime_reserve_bytes=8 * GIB,
        dpa_weights=DpaWeights(734618714112, 20998426304, 256, "headers"),
    )
    selected = next(x for x in layouts if x.attention_data_parallel == 8)
    assert selected.kv_per_sequence_per_rank_bytes == stored.total_bytes
    assert selected.fixed_cache_per_rank_bytes == sum(c.bytes for c in stored.fixed_components)
