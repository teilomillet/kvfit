from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kvfit.architectures import estimate_cache
from kvfit.audit import load_audit_config, run_audit
from kvfit.cli import main, resolve_cache_dtype
from kvfit.hardware import parse_hardware
from kvfit.models import GIB, ModelMetadata, UnsupportedArchitecture
from kvfit.oracle import check_cache_estimate
from kvfit.planner import plan_topologies

FIXTURES = Path(__file__).parent / "fixtures" / "model_configs"


def _fixture(name: str = "zai-org--GLM-5.3") -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _small_glm(**overrides: object) -> dict:
    return {
        "model_type": "glm_moe_dsa",
        "architectures": ["GlmMoeDsaForCausalLM"],
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        **overrides,
    }


@pytest.mark.parametrize("context", [1, 127, 128, 129, 4096, 1048576])
@pytest.mark.parametrize("kv_bytes", [0.5, 1, 2])
@pytest.mark.parametrize("index_bytes", [0.5, 1, 2])
def test_glm53_index_keys_follow_checkpoint_tensor_ownership(
    context: int, kv_bytes: float, index_bytes: float
) -> None:
    # These layer IDs come from the checkpoint's tensor index, not from kvfit
    # or from re-implementing its frequency rule. See the fixture provenance.
    evidence = _fixture("glm53-indexer-evidence")
    config = _fixture()["config"]
    owners = evidence["target_layers"]
    assert len(owners) == 21
    assert evidence["mtp_layers"] == [78]  # Outside this base-model estimate.
    assert [i for i, kind in enumerate(config["indexer_types"]) if kind == "full"] == owners
    estimate = estimate_cache(
        config, context_tokens=context, kv_bytes=kv_bytes, index_bytes=index_bytes
    )
    mla, index = estimate.components
    assert mla.bytes == 78 * context * 576 * kv_bytes
    assert index.bytes == len(owners) * context * 128 * index_bytes
    assert "21" in index.detail
    assert "14793d45af28336310a0013a89f1488d8ba1cc51" in estimate.reference
    assert estimate.confidence == "architecture-formula"


@pytest.mark.parametrize(
    ("name", "context", "expected_gib", "architecture"),
    [
        ("zai-org--GLM-5.1", 1048576, 53.625, "glm-dsa-mla"),
        ("zai-org--GLM-5.2", 1048576, 46.5, "glm-dsa-mla"),
        ("zai-org--GLM-5.3", 1048576, 46.5, "glm-dsa-mla"),
        ("deepseek-ai--DeepSeek-V3.2", 1048576, 41.9375, "deepseek-v3.2-dsa"),
        (
            "deepseek-ai--DeepSeek-V4-Flash",
            1048576,
            3.36199951171875,
            "deepseek-v4-compressed-hybrid",
        ),
        ("Qwen--Qwen3-32B", 40960, 5.0, "standard-gqa-mqa"),
    ],
)
def test_pinned_models_keep_architecture_specific_payloads(
    name: str, context: int, expected_gib: float, architecture: str
) -> None:
    config = _fixture(name)["config"]
    estimate = estimate_cache(config, context_tokens=context, kv_bytes=1, index_bytes=1)
    assert estimate.architecture == architecture
    assert estimate.total_gib == expected_gib
    assert check_cache_estimate(config, estimate, kv_bytes=1, index_bytes=1).passed


def test_recent_qwen_hybrid_retains_fixed_recurrent_state() -> None:
    config = _fixture("Qwen--Qwen3.6-35B-A3B")["config"]
    small = estimate_cache(config, context_tokens=4096, kv_bytes=1)
    large = estimate_cache(config, context_tokens=8192, kv_bytes=1)
    # Ten full-attention layers, two KV heads, width 256. The other thirty
    # layers carry fixed recurrent/conv state; doubling context must not double it.
    assert large.total_bytes - small.total_bytes == 2 * 10 * 4096 * 2 * 256
    assert large.total_bytes < 2 * small.total_bytes
    assert check_cache_estimate(config, large, kv_bytes=1, index_bytes=1).passed


def test_recent_glm_flash_is_not_approximated_by_glm_mla() -> None:
    with pytest.raises(UnsupportedArchitecture):
        estimate_cache(_fixture("zai-org--GLM-5.3-Flash")["config"], context_tokens=4096)


@pytest.mark.parametrize(
    "name",
    [
        "zai-org--GLM-5.3",
        "deepseek-ai--DeepSeek-V3.2",
        "deepseek-ai--DeepSeek-V4-Flash",
        "Qwen--Qwen3-32B",
        "Qwen--Qwen3.6-35B-A3B",
    ],
)
def test_new_model_type_cannot_inherit_support_from_an_old_class(name: str) -> None:
    config = _fixture(name)["config"]
    text = config.get("text_config", config)
    text["model_type"] = "future_cache_architecture"
    with pytest.raises(UnsupportedArchitecture):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize("name", ["zai-org--GLM-5.3", "deepseek-ai--DeepSeek-V3.2"])
def test_exact_known_class_still_works_without_model_type(name: str) -> None:
    config = _fixture(name)["config"]
    expected = estimate_cache(config, context_tokens=4096)
    del config["model_type"]
    assert estimate_cache(config, context_tokens=4096).total_bytes == expected.total_bytes


@pytest.mark.parametrize(
    "name",
    ["zai-org--GLM-5.3", "deepseek-ai--DeepSeek-V3.2", "deepseek-ai--DeepSeek-V4-Flash"],
)
def test_future_class_name_does_not_match_by_substring(name: str) -> None:
    config = _fixture(name)["config"]
    del config["model_type"]
    config["architectures"] = [
        config["architectures"][0].replace("ForCausalLM", "FutureForCausalLM")
    ]
    with pytest.raises(UnsupportedArchitecture):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize(
    ("declaration", "owners"),
    [
        ({}, 8),  # Documented GLM default, exercised by the older 5.1 checkpoint.
        ({"indexer_types": ["full", "shared"] * 4}, 4),
        ({"index_topk_pattern": "FSSSFSSS"}, 2),
        ({"index_topk_pattern": ["full"] + ["shared"] * 7}, 1),
        ({"index_topk_freq": 2, "index_skip_topk_offset": 2}, 5),
        ({"index_topk_freq": 3, "index_skip_topk_offset": 2}, 4),
        ({"index_topk_freq": 4, "index_skip_topk_offset": 3}, 4),
        ({"index_topk_freq": 1, "index_skip_topk_offset": 0}, 8),
        ({"index_topk_freq": 3}, 4),
        ({"index_skip_topk_offset": 3}, 8),
    ],
)
def test_schedule_representations_change_only_index_state(declaration: dict, owners: int) -> None:
    config = _small_glm(**declaration)
    base = estimate_cache(_small_glm(), context_tokens=4096, kv_bytes=2, index_bytes=1)
    actual = estimate_cache(config, context_tokens=4096, kv_bytes=2, index_bytes=1)
    assert actual.components[0] == base.components[0]
    assert actual.components[1].bytes == owners * 4096 * 128
    assert check_cache_estimate(config, actual, kv_bytes=2, index_bytes=1).passed


def test_equivalent_schedule_declarations_agree() -> None:
    estimate = estimate_cache(
        _small_glm(indexer_types=["full", "shared"] * 4, index_topk_pattern="FSFSFSFS"),
        context_tokens=4096,
    )
    assert estimate.components[1].bytes == 4 * 4096 * 128 * 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"indexer_types": []},
        {"indexer_types": "FFFFFFFF"},
        {"indexer_types": ["full"] * 7},
        {"indexer_types": ["shared"] * 8},
        {"indexer_types": ["full"] + ["future_mode"] * 7},
        {"indexer_types": ["full"] + [{}] * 7},
        {"index_topk_pattern": "FSSF"},
        {"index_topk_pattern": "FXXXXXXX"},
        {"index_topk_pattern": 8},
        {"index_topk_freq": 0},
        {"index_topk_freq": -1},
        {"index_topk_freq": True},
        {"index_topk_freq": 2.5},
        {"index_skip_topk_offset": -1},
        {"index_skip_topk_offset": True},
        {"index_topk_freq": 2, "index_skip_topk_offset": 0},
        {"indexer_types": ["full"] * 8, "index_topk_pattern": "FSFSFSFS"},
        {"indexer_types": ["full"] * 8, "index_topk_freq": 4},
        {"index_topk_pattern": "FFFFFFFF", "index_topk_freq": 4},
        {"layer_types": ["linear_attention"] * 8},
        {"layer_types": ["indexed_attention"] * 7},
        {"linear_attn_config": {}},
        {"compress_ratios": [4] * 8},
        {"index_kpool": 4},
        {"sliding_window": 128},
        {"auto_map": {"AutoModelForCausalLM": "custom.CustomModel"}},
        {"model_type": "glm_moe_dsa_future"},
        {"architectures": ["GlmMoeDsaFutureForCausalLM"]},
    ],
)
def test_ambiguous_or_unmodeled_glm_state_fails_closed(overrides: dict) -> None:
    with pytest.raises(UnsupportedArchitecture):
        estimate_cache(_small_glm(**overrides), context_tokens=4096)


@pytest.mark.parametrize("layer_type", ["indexed_attention", "deepseek_sparse_attention"])
def test_declared_dsa_layer_schedule_is_accepted(layer_type: str) -> None:
    result = estimate_cache(_small_glm(layer_types=[layer_type] * 8), context_tokens=4096)
    assert result.architecture == "glm-dsa-mla"


def test_oracle_rejects_original_all_layer_indexer_bug() -> None:
    config = _fixture()["config"]
    correct = estimate_cache(config, context_tokens=1048576, kv_bytes=1, index_bytes=1)
    legacy = replace(
        correct,
        components=(correct.components[0], replace(correct.components[1], bytes=9.75 * GIB)),
    )
    assert not check_cache_estimate(config, legacy, kv_bytes=1, index_bytes=1).passed


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_tensor_parallel_does_not_divide_glm_cache(tp: int) -> None:
    cache = estimate_cache(_fixture()["config"], context_tokens=1048576, kv_bytes=1)
    (layout,) = plan_topologies(
        cache,
        weight_bytes=64 * GIB,
        hardware=parse_hardware("custom:256"),
        gpus=8,
        utilization=0.9,
        tensor_parallel=tp,
    )
    assert layout.kv_per_sequence_per_rank_bytes == 46.5 * GIB
    assert layout.weights_per_rank_bytes == 64 * GIB / tp
    assert layout.data_parallel == 8 // tp
    assert layout.total_sequences == layout.sequences_per_replica * layout.data_parallel


def test_cli_and_audit_use_corrected_pinned_model(monkeypatch, capsys, tmp_path) -> None:
    fixture = _fixture()
    metadata = ModelMetadata(
        repo_id=fixture["repo"],
        requested_revision=fixture["revision"],
        resolved_revision=fixture["revision"],
        config=fixture["config"],
        weight_bytes=64 * GIB,
        weight_source="synthetic weights; real pinned cache config",
    )
    assert resolve_cache_dtype(metadata.config, "auto")[0] == "bf16"
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *a, **kw: metadata)
    assert (
        main(
            [
                metadata.repo_id,
                "--hardware",
                "custom:256",
                "--context",
                "1m",
                "--kv-dtype",
                "fp8",
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["cache"]["gib_per_sequence"] == 46.5
    assert output["concurrency"]["qualification"] == "memory-only"
    monkeypatch.setattr("kvfit.audit.fetch_model_metadata", lambda *a, **kw: metadata)
    path = tmp_path / "audit.toml"
    path.write_text("""contexts = ["1m"]
kv_dtypes = ["fp8"]
gpu_counts = [1]
[[models]]
repo = "zai-org/GLM-5.3"
expected_architecture = "glm-dsa-mla"
[[hardware]]
preset = "h100-80"
""")
    report = run_audit(load_audit_config(path))
    assert report["passed"]
    assert report["rows"][0]["oracle"]["expected_bytes"] == 46.5 * GIB
