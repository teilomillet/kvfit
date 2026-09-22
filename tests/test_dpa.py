from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kvfit.dpa import DpaWeights, classify_dpa_weights
from kvfit.hardware import Hardware, parse_hardware
from kvfit.models import GIB, CacheComponent, ModelMetadata, StorageCacheEstimate
from kvfit.planner import plan_topologies, search_systems
from kvfit.systems import parse_system


def tiny_checkpoint(*, experts=4, layers=2, quantized=False):
    config = {
        "model_type": "glm_moe_dsa", "num_hidden_layers": layers,
        "num_nextn_predict_layers": 0, "n_routed_experts": experts,
        "first_k_dense_replace": 0, "hidden_size": 128, "moe_intermediate_size": 128,
        "dtype": "bfloat16",
    }
    if quantized:
        config["quantization_config"] = {
            "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
        }
    tensors = {"model.embed_tokens.weight": {"dtype": "BF16", "shape": [2, 128]}}
    for layer in range(layers):
        for expert in range(experts):
            for proj in ["gate", "up", "down"]:
                key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj"
                tensors[key + ".weight"] = {
                    "dtype": "F8_E4M3" if quantized else "BF16", "shape": [128, 128],
                }
                if quantized:
                    tensors[key + ".weight_scale_inv"] = {"dtype": "F32", "shape": [1, 1]}
    offset = 0
    for tensor in tensors.values():
        size = tensor["shape"][0] * tensor["shape"][1] * {
            "BF16": 2, "F32": 4, "F8_E4M3": 1,
        }[tensor["dtype"]]
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


@pytest.mark.parametrize("mutation", [
    "missing_weight", "extra_expert", "bad_shape", "bad_dtype", "bad_offsets",
    "missing_scale", "unequal_experts", "unknown_expert_tensor", "unqualified_model",
    "unqualified_quantization", "wrong_layer_schedule", "noninteger_experts",
])
def test_unknown_or_incomplete_weight_placement_fails_closed(mutation):
    config, tensors = tiny_checkpoint(quantized=True)
    key = "model.layers.0.mlp.experts.0.down_proj.weight"
    if mutation == "missing_weight": del tensors[key]
    elif mutation == "extra_expert":
        tensors[key.replace("experts.0", "experts.4")] = tensors[key]
    elif mutation == "bad_shape": tensors[key]["shape"] = [64, 256]
    elif mutation == "bad_dtype": tensors[key]["dtype"] = "F4"
    elif mutation == "bad_offsets": tensors[key]["data_offsets"] = [9, 0]
    elif mutation == "missing_scale": del tensors[key + "_scale_inv"]
    elif mutation == "unequal_experts":
        tensors[key] = {"dtype": "BF16", "shape": [128, 128], "data_offsets": [0, 32768]}
        del tensors[key + "_scale_inv"]
    elif mutation == "unknown_expert_tensor":
        tensors[key.replace("down_proj.weight", "new_state")] = tensors[key]
    elif mutation == "unqualified_model": config["model_type"] = "deepseek_v4"
    elif mutation == "unqualified_quantization":
        config["quantization_config"]["quant_method"] = "modelopt"
    elif mutation == "wrong_layer_schedule": config["first_k_dense_replace"] = 1
    else: config["n_routed_experts"] = 4.5
    with pytest.raises(ValueError):
        classify_dpa_weights(config, tensors, source="mutated checkpoint")


def cache(state_gib=1, padding_gib=0):
    return StorageCacheEstimate(
        "glm-dsa-mla", 1024, (CacheComponent("state", state_gib * GIB, "fixture"),),
        1, 8, "upstream-storage-formula", "fixture",
        fixed_components=(CacheComponent("padding", padding_gib * GIB, "fixture"),),
    )


def test_dpa_distinguishes_attention_groups_from_full_replicas():
    kwargs = dict(weight_bytes=10 * GIB, hardware=Hardware("test", "test", 7),
                  gpus=16, utilization=1, tensor_parallel=8, runtime_reserve_bytes=GIB)
    ordinary = plan_topologies(cache(padding_gib=1), **kwargs)
    layouts = plan_topologies(cache(padding_gib=1),
        dpa_weights=DpaWeights(8 * GIB, 2 * GIB, 8, "fixture"), **kwargs)
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
    layouts = plan_topologies(c, weight_bytes=8*GIB, hardware=Hardware("x", "x", 8),
        gpus=8, tensor_parallel=8, utilization=1,
        dpa_weights=DpaWeights(6*GIB, 2*GIB, 8, "fixture"))
    assert layouts[0].verdict == "unsupported"  # TP8 cannot partition two heads.
    assert {x.attention_data_parallel for x in layouts if x.strategy == "sglang-dpa"
            and x.verdict == "fits"} == {4, 8}
    layouts = plan_topologies(cache(), weight_bytes=8*GIB, hardware=Hardware("x", "x", 8),
        gpus=8, tensor_parallel=8, utilization=1,
        dpa_weights=DpaWeights(6*GIB, 2*GIB, 6, "fixture"))
    assert not any(x.strategy == "sglang-dpa" and x.verdict == "fits" for x in layouts)


def test_glm53_pinned_evidence_yields_four_local_b300_groups():
    evidence = json.loads((Path(__file__).parent / "fixtures" / "sglang-dpa.json").read_text())
    weights = DpaWeights(**evidence["weights"])
    c = cache(54728 / 1024, 54728 * 64 / GIB)
    result = search_systems(c, weight_bytes=755632050320,
        targets=[(parse_system("dgx-b300"), parse_hardware("b300"))],
        max_nodes=4, concurrent_users=25, active_sequences_per_user=2,
        utilization=.8, runtime_reserve_bytes=8*GIB, dpa_weights=weights)
    row = result["minimums"][0]
    assert row["nodes"] == 4
    assert row["capacity"]["layout"]["strategy"] == "sglang-dpa"
    assert row["capacity"]["active_sequences"] == 64
    assert result["attempts"][-2]["capacity"]["active_sequences"] == 48
    assert "DP-Attention" in result["scope"]


def test_dpa_rejects_unqualified_cache():
    with pytest.raises(ValueError, match="GLM"):
        plan_topologies(replace(cache(), architecture="deepseek-v4-compressed-hybrid"),
            weight_bytes=8*GIB, hardware=Hardware("x", "x", 8), gpus=8, utilization=1,
            dpa_weights=DpaWeights(6*GIB, 2*GIB, 8, "fixture"))


@pytest.mark.parametrize("values", [(float("nan"), 1, 8), (1, -1, 8), (1, 1, 0), (1, 1, True)])
def test_invalid_weight_evidence_is_rejected(values):
    with pytest.raises(ValueError):
        DpaWeights(*values, "invalid")


def test_cli_search_and_toml_use_the_same_profile(monkeypatch, capsys, tmp_path):
    from kvfit.cli import main
    config = json.loads((Path(__file__).parent / "fixtures/model_configs/zai-org--GLM-5.3.json")
                        .read_text())["config"]
    metadata = ModelMetadata("test/glm", "main", "fixed", config, 755632050320, "fixture")
    calls = []
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *a, **k: metadata)
    monkeypatch.setattr("kvfit.cli.fetch_dpa_weights", lambda *a, **k: calls.append(a) or
        DpaWeights(734618714112, 20998426304, 256, "observed headers"))
    path = tmp_path / "test.toml"
    path.write_text('model="test/glm"\nsystems=["dgx-b300"]\nmax_nodes=4\n'
        'concurrent_users=25\nactive_sequences_per_user=2\ncontext="1m"\n'
        'kv_dtype="fp8"\ncache_layout="sglang-dsa-scaled"\nmtp=true\n'
        'parallelism="sglang-dpa"\nruntime_reserve_gib=8\nutilization=0.8\n'
        '[output]\njson=true\n')
    assert main(["--config", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["search"]["minimums"][0]["nodes"] == 4
    assert output["weight_placement"]["routed_expert_bytes"] == 734618714112
    assert len(calls) == 1


@pytest.mark.parametrize("extra", [
    ["--weight-gib", "100"], ["--check-engine", "sglang"],
])
def test_dpa_rejects_ambiguous_weight_override_or_unconfigured_probe(monkeypatch, capsys, extra):
    from kvfit.cli import main
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *a, **k: pytest.fail("network"))
    assert main(["test/glm", "--system", "dgx-b300", "--parallelism", "sglang-dpa",
                 "--cache-layout", "sglang-dsa-scaled", *extra]) == 2


def test_dpa_calibration_cannot_silently_launch_tp(tmp_path):
    from kvfit.calibrate import _default_server_command
    from kvfit.calibration_config import load_calibration_config
    from test_calibrate import _prediction
    path = tmp_path / "deployment.toml"
    path.write_text('model="test/glm"\nhardware="b300"\n[calibration]\nengine="sglang"\n')
    config = load_calibration_config(path)
    prediction = copy.deepcopy(_prediction())
    prediction["concurrency"]["recommended"]["layout"]["strategy"] = "sglang-dpa"
    with pytest.raises(ValueError, match="DPA|DP-Attention"):
        _default_server_command(config, prediction)
