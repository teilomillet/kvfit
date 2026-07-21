from __future__ import annotations

import json
from dataclasses import fields

import pytest

from kvfit.architectures import estimate_cache
from kvfit.cli import main
from kvfit.hardware import parse_hardware
from kvfit.models import (
    GIB,
    CacheEstimate,
    ModelMetadata,
    SpeculativeCacheEstimate,
    UnsupportedArchitecture,
)
from kvfit.planner import plan_topologies

PINNED_FLASH_MODEL_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
PINNED_PRO_MODEL_REVISION = "7c09739fd136abfb70a49ec334157f65f45b52cd"
PINNED_VLLM_REVISION = "752a3a504485790a2e8491cacbb35c137339ad34"


def _deepseek_v4_flash_config(*, dspark: bool) -> dict[str, object]:
    # State-shape fields from deepseek-ai/DeepSeek-V4-Flash[-DSpark]. The
    # first 43 entries are the target schedule. Standard V4 Flash appends one
    # MTP entry; the checked DSpark revision appends three draft-layer entries.
    target_ratios = [0, 0] + [4, 128] * 20 + [4]
    config: dict[str, object] = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": target_ratios + ([0, 0, 0] if dspark else [0]),
        "max_position_embeddings": 1024**2,
        "torch_dtype": "bfloat16",
    }
    if dspark:
        config.update(
            {
                "dspark_block_size": 5,
                "dspark_noise_token_id": 128799,
                "dspark_target_layer_ids": [40, 41, 42],
                "dspark_markov_rank": 256,
            }
        )
    return config


def _deepseek_v4_pro_dspark_config() -> dict[str, object]:
    # Exact cache-shape fields from the pinned official Pro DSpark config. This
    # is deliberately separate from the Flash fixture: Pro has 61 target layers,
    # 128 query heads, a different compression schedule, and Markov rank 512.
    target_ratios = [128, 128] + [4, 128] * 29 + [4]
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [*target_ratios, 0, 0, 0],
        "max_position_embeddings": 1024**2,
        "torch_dtype": "bfloat16",
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [58, 59, 60],
        "dspark_markov_rank": 512,
    }


def test_integrated_dspark_is_detected_and_adds_verified_draft_kv() -> None:
    integrated_config = _deepseek_v4_flash_config(dspark=True)
    target_config = {
        key: value for key, value in integrated_config.items() if not key.startswith("dspark_")
    }
    target = estimate_cache(
        target_config,
        context_tokens=1024**2,
    )
    dspark = estimate_cache(
        integrated_config,
        context_tokens=1024**2,
    )

    expected_draft_bytes = 3 * 128 * 512 * 2
    assert dspark.architecture == "deepseek-v4-compressed-hybrid+dspark"
    assert type(dspark) is SpeculativeCacheEstimate
    assert dspark.total_bytes == target.total_bytes + expected_draft_bytes
    assert dspark.components[:-1] == target.components
    assert dspark.reference == (
        "https://vllm.ai/blog/2026/04/24/deepseek-v4.html#the-math-behind-"
        "deepseek-v4s-attention-mechanism"
    )
    assert dspark.components[-1].name == "dspark-draft-kv"
    assert dspark.components[-1].bytes == expected_draft_bytes
    assert dspark.components[-1].tp_parallel_units == 1
    assert dspark.speculative_decoding is not None
    assert dspark.speculative_decoding.as_dict() == {
        "method": "dspark",
        "packaging": "integrated",
        "declaration_source": "model-config",
        "runtime_enabled": None,
        "draft_layers": 3,
        "checkpoint_block_size": 5,
        "target_layer_ids": [40, 41, 42],
        "cache_component": "dspark-draft-kv",
        "cache_modeled": True,
        "runtime_buffers_modeled": False,
        "performance_modeled": False,
        "reference": (
            "https://github.com/vllm-project/vllm/blob/"
            f"{PINNED_VLLM_REVISION}/vllm/models/deepseek_v4/nvidia/dspark.py"
        ),
    }
    assert any("logical sliding-window KV state is counted" in note for note in dspark.notes)
    assert any("CUDA graphs" in note and "runtime-only" in note for note in dspark.notes)


def test_ordinary_cache_object_and_serialization_keep_the_pre_dspark_shape() -> None:
    estimate = estimate_cache(
        _deepseek_v4_flash_config(dspark=False),
        context_tokens=4096,
    )
    payload = estimate.as_dict()

    assert type(estimate) is CacheEstimate
    assert not hasattr(estimate, "speculative_decoding")
    assert [field.name for field in fields(estimate)] == [
        "architecture",
        "context_tokens",
        "components",
        "kv_parallel_heads",
        "query_heads",
        "confidence",
        "reference",
        "notes",
    ]
    assert set(payload) == {
        "architecture",
        "context_tokens",
        "bytes_per_sequence",
        "gib_per_sequence",
        "kv_parallel_heads",
        "query_heads",
        "confidence",
        "reference",
        "components",
        "notes",
    }


def test_official_pro_dspark_config_confirms_detection_on_different_geometry() -> None:
    estimate = estimate_cache(
        _deepseek_v4_pro_dspark_config(),
        context_tokens=1024**2,
    )

    draft = estimate.components[-1]
    assert estimate.architecture == "deepseek-v4-compressed-hybrid+dspark"
    assert estimate.query_heads == 128
    assert estimate.total_bytes == 10_334_765_056, PINNED_PRO_MODEL_REVISION
    assert draft.name == "dspark-draft-kv"
    assert draft.bytes == 3 * 128 * 512 * 2
    assert estimate.speculative_decoding is not None
    assert estimate.speculative_decoding.draft_layers == 3
    assert estimate.speculative_decoding.target_layer_ids == (58, 59, 60)


def test_dspark_draft_cache_uses_context_below_sliding_window_and_selected_dtype() -> None:
    estimate = estimate_cache(
        _deepseek_v4_flash_config(dspark=True),
        context_tokens=64,
        kv_bytes=0.5,
    )

    draft = next(
        component for component in estimate.components if component.name == "dspark-draft-kv"
    )
    assert draft.bytes == 3 * 64 * 512 * 0.5
    assert "64 sliding-window entries" in draft.detail


@pytest.mark.parametrize("tensor_parallel", [1, 2, 4, 8])
def test_dspark_addition_commutes_with_tp_placement(tensor_parallel: int) -> None:
    target = estimate_cache(
        _deepseek_v4_flash_config(dspark=False),
        context_tokens=1024**2,
    )
    combined = estimate_cache(
        _deepseek_v4_flash_config(dspark=True),
        context_tokens=1024**2,
    )
    target_layout = plan_topologies(
        target,
        weight_bytes=0,
        hardware=parse_hardware("custom:32"),
        gpus=tensor_parallel,
        utilization=1,
        tensor_parallel=tensor_parallel,
    )[0]
    combined_layout = plan_topologies(
        combined,
        weight_bytes=0,
        hardware=parse_hardware("custom:32"),
        gpus=tensor_parallel,
        utilization=1,
        tensor_parallel=tensor_parallel,
    )[0]

    assert combined_layout.kv_per_sequence_per_rank_bytes == (
        target_layout.kv_per_sequence_per_rank_bytes + combined.components[-1].bytes
    )


def test_partial_dspark_config_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    del config["dspark_markov_rank"]

    with pytest.raises(UnsupportedArchitecture, match="partial DSpark configuration"):
        estimate_cache(config, context_tokens=4096)


def test_unknown_dspark_marker_does_not_fall_back_to_base_model() -> None:
    config = _deepseek_v4_flash_config(dspark=False)
    config["dspark_future_layout"] = "unknown"

    with pytest.raises(UnsupportedArchitecture, match="unverified DSpark configuration fields"):
        estimate_cache(config, context_tokens=4096)


def test_new_dspark_field_fails_closed_until_its_semantics_are_verified() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_future_layout"] = "unknown"

    with pytest.raises(UnsupportedArchitecture, match="unverified DSpark configuration fields"):
        estimate_cache(config, context_tokens=4096)


def test_changed_dspark_draft_cache_schedule_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    assert isinstance(ratios, list)
    ratios[-1] = 4

    with pytest.raises(UnsupportedArchitecture, match="sliding-window DSpark"):
        estimate_cache(config, context_tokens=4096)


def test_fractional_dspark_draft_cache_schedule_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    assert isinstance(ratios, list)
    ratios[-1] = 0.5

    with pytest.raises(UnsupportedArchitecture, match="sliding-window DSpark"):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dspark_block_size", 5.5),
        ("dspark_noise_token_id", 128799.5),
        ("dspark_markov_rank", 256.5),
        ("n_mtp_layers", 3.5),
    ],
)
def test_fractional_dspark_integer_field_fails_closed(field: str, value: float) -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config[field] = value

    with pytest.raises(UnsupportedArchitecture, match=f"non-integer '{field}'"):
        estimate_cache(config, context_tokens=4096)


def test_dspark_target_layer_zero_is_a_valid_lower_boundary() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [0, 1, 2]

    estimate = estimate_cache(config, context_tokens=4096)

    assert estimate.speculative_decoding.target_layer_ids == (0, 1, 2)


def test_dspark_target_layer_equal_to_layer_count_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [40, 41, 43]

    with pytest.raises(UnsupportedArchitecture, match="strictly increasing base-layer indices"):
        estimate_cache(config, context_tokens=4096)


def test_negative_dspark_target_layer_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [-1, 0, 1]

    with pytest.raises(UnsupportedArchitecture, match="strictly increasing base-layer indices"):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize("target_ids", [40, "40,41,42"])
def test_non_sequence_dspark_target_layers_fail_closed(target_ids: object) -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = target_ids

    with pytest.raises(UnsupportedArchitecture, match="explicit sequence"):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize("target_ids", [[], [40, 41, True], [40, 41, 42.0]])
def test_non_integer_dspark_target_layers_fail_closed(target_ids: list[object]) -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = target_ids

    with pytest.raises(UnsupportedArchitecture, match="integer target-layer indices"):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize("post_model_ratios", [[], [0, 0, False], [0, 0, "0"]])
def test_non_numeric_dspark_draft_schedule_fails_closed(
    post_model_ratios: list[object],
) -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    assert isinstance(ratios, list)
    config["compress_ratios"] = [*ratios[:43], *post_model_ratios]

    with pytest.raises(UnsupportedArchitecture, match="numeric post-model cache entry"):
        estimate_cache(config, context_tokens=4096)


def test_dspark_noise_token_zero_is_valid_and_negative_is_rejected() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_noise_token_id"] = 0
    assert estimate_cache(config, context_tokens=4096).speculative_decoding is not None

    config["dspark_noise_token_id"] = -1
    with pytest.raises(UnsupportedArchitecture, match="invalid 'dspark_noise_token_id'=-1"):
        estimate_cache(config, context_tokens=4096)


def test_matching_explicit_dspark_layer_count_is_accepted() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["n_mtp_layers"] = 3

    estimate = estimate_cache(config, context_tokens=4096)

    assert estimate.speculative_decoding.draft_layers == 3


def test_mismatched_explicit_dspark_layer_count_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["n_mtp_layers"] = 4

    with pytest.raises(UnsupportedArchitecture, match="post-model cache schedule"):
        estimate_cache(config, context_tokens=4096)


def test_dspark_target_layer_count_must_match_draft_layer_count() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [40, 41]

    with pytest.raises(UnsupportedArchitecture, match="target-layer count differs"):
        estimate_cache(config, context_tokens=4096)


def test_dspark_rejections_preserve_structured_model_and_reason_fields() -> None:
    cases: list[tuple[dict[str, object], str]] = []

    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_future_layout"] = "unknown"
    cases.append((config, "unverified DSpark configuration fields cannot be sized safely"))

    config = _deepseek_v4_flash_config(dspark=True)
    del config["dspark_markov_rank"]
    cases.append((config, "partial DSpark configuration cannot be sized safely"))

    for target_ids in ("40,41,42", 40):
        config = _deepseek_v4_flash_config(dspark=True)
        config["dspark_target_layer_ids"] = target_ids
        cases.append((config, "dspark_target_layer_ids must be an explicit sequence"))

    for target_ids in ([], [40, 41, 42.0]):
        config = _deepseek_v4_flash_config(dspark=True)
        config["dspark_target_layer_ids"] = target_ids
        cases.append((config, "dspark_target_layer_ids must contain integer target-layer indices"))

    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [-1, 0, 1]
    cases.append((config, "dspark_target_layer_ids must be strictly increasing base-layer indices"))

    for post_model_ratios in ([], [0, 0, "0"]):
        config = _deepseek_v4_flash_config(dspark=True)
        ratios = config["compress_ratios"]
        assert isinstance(ratios, list)
        config["compress_ratios"] = [*ratios[:43], *post_model_ratios]
        cases.append((config, "DSpark requires one numeric post-model cache entry per draft layer"))

    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    assert isinstance(ratios, list)
    ratios[-1] = 4
    cases.append((config, "only sliding-window DSpark draft layers are verified"))

    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    assert isinstance(ratios, list)
    ratios.pop()
    cases.append((config, "DSpark without n_mtp_layers must match the verified three-layer layout"))

    config = _deepseek_v4_flash_config(dspark=True)
    config["n_mtp_layers"] = 4
    cases.append((config, "n_mtp_layers differs from the post-model cache schedule"))

    config = _deepseek_v4_flash_config(dspark=True)
    config["dspark_target_layer_ids"] = [40, 41]
    cases.append((config, "DSpark target-layer count differs from its draft-layer count"))

    for config, expected_reason in cases:
        with pytest.raises(UnsupportedArchitecture) as caught:
            estimate_cache(config, context_tokens=4096)
        assert caught.value.model_type == "deepseek_v4"
        assert caught.value.reason == expected_reason


def test_unversioned_nonstandard_dspark_layer_count_fails_closed() -> None:
    config = _deepseek_v4_flash_config(dspark=True)
    ratios = config["compress_ratios"]
    target_ids = config["dspark_target_layer_ids"]
    assert isinstance(ratios, list)
    assert isinstance(target_ids, list)
    ratios.pop()
    target_ids.pop()

    with pytest.raises(UnsupportedArchitecture, match="verified three-layer layout"):
        estimate_cache(config, context_tokens=4096)


def test_cli_json_exposes_integrated_dspark_and_runtime_boundary(monkeypatch, capsys) -> None:
    metadata = ModelMetadata(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        requested_revision=PINNED_FLASH_MODEL_REVISION,
        resolved_revision=PINNED_FLASH_MODEL_REVISION,
        config=_deepseek_v4_flash_config(dspark=True),
        weight_bytes=155 * GIB,
        weight_source="pinned test artifact",
    )
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)

    assert (
        main(
            [
                metadata.repo_id,
                "--hardware",
                "custom:256",
                "--gpus",
                "2",
                "--tp",
                "2",
                "--context",
                "1m",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    speculative = payload["cache"]["speculative_decoding"]
    assert speculative["method"] == "dspark"
    assert speculative["declaration_source"] == "model-config"
    assert speculative["runtime_enabled"] is None
    assert speculative["draft_layers"] == 3
    assert speculative["cache_modeled"] is True
    assert speculative["runtime_buffers_modeled"] is False
    assert payload["cache"]["components"][-1]["name"] == "dspark-draft-kv"
    assert payload["layouts"][0]["kv_per_sequence_per_rank_gib"] == pytest.approx(
        payload["cache"]["gib_per_sequence"]
    )


def test_human_output_names_dspark_and_counted_draft_kv(monkeypatch, capsys) -> None:
    metadata = ModelMetadata(
        repo_id="deepseek-ai/DeepSeek-V4-Flash-DSpark",
        requested_revision=PINNED_FLASH_MODEL_REVISION,
        resolved_revision=PINNED_FLASH_MODEL_REVISION,
        config=_deepseek_v4_flash_config(dspark=True),
        weight_bytes=155 * GIB,
        weight_source="pinned test artifact",
    )
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)

    assert main([metadata.repo_id, "--hardware", "custom:256", "--context", "1m"]) == 0
    output = capsys.readouterr().out

    assert (
        "Speculative:  DSPARK (integrated; checkpoint-declared, runtime enablement unverified), "
        "3 draft layers, checkpoint block 5; draft KV counted"
    ) in output
    assert "dspark-draft-kv" in output
    assert "CUDA graphs" in output
    assert "Formula reference: https://vllm.ai/blog/2026/04/24/deepseek-v4.html" in output
    assert (
        f"Speculative reference: https://github.com/vllm-project/vllm/blob/{PINNED_VLLM_REVISION}"
        in output
    )
