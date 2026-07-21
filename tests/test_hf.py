from __future__ import annotations

import pytest

from kvfit.hf import (
    HuggingFaceError,
    _gpt_oss_runtime_weight_floor,
    _parse_model_reference,
    _validate_gpt_oss_index,
    _weight_artifact_bytes,
    fetch_model_metadata,
    parse_model_reference,
)
from kvfit.models import GIB


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Qwen/Qwen3-32B", ("Qwen/Qwen3-32B", "main")),
        ("Qwen/Qwen3-32B@abc123", ("Qwen/Qwen3-32B", "abc123")),
        (
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
            ("deepseek-ai/DeepSeek-V4-Flash", "main"),
        ),
        (
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/dev",
            ("deepseek-ai/DeepSeek-V4-Flash", "dev"),
        ),
        ("hf://meta-llama/Llama-3.1-8B", ("meta-llama/Llama-3.1-8B", "main")),
    ],
)
def test_parse_model_reference(value: str, expected: tuple[str, str]) -> None:
    assert parse_model_reference(value) == expected


def test_parse_gguf_blob_reference_preserves_artifact() -> None:
    parsed = _parse_model_reference(
        "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/blob/main/gpt-oss-20b-Q4_K_M.gguf"
    )

    assert parsed.repo_id == "unsloth/gpt-oss-20b-GGUF"
    assert parsed.revision == "main"
    assert parsed.artifact_path == "gpt-oss-20b-Q4_K_M.gguf"


def test_weight_artifacts_prefer_transformers_shards() -> None:
    metadata = {
        "siblings": [
            {"rfilename": "model-00001-of-00002.safetensors", "size": 100},
            {"rfilename": "model-00002-of-00002.safetensors", "size": 200},
            {"rfilename": "optimizer.safetensors", "size": 999},
            {"rfilename": "original/model.safetensors", "size": 500},
        ]
    }

    size, source = _weight_artifact_bytes(metadata)

    assert size == 300
    assert source == "HF root safetensors (2 file(s))"


def test_weight_artifacts_fall_back_to_dtype_counts() -> None:
    metadata = {
        "safetensors": {
            "parameters": {
                "BF16": 100,
                "F32": 10,
            }
        }
    }

    size, source = _weight_artifact_bytes(metadata)

    assert size == 240
    assert source == "HF safetensors dtype counts (runtime overhead excluded)"


def test_selected_split_gguf_sums_every_shard() -> None:
    metadata = {
        "siblings": [
            {"rfilename": "Q4/model-00001-of-00002.gguf", "size": 100},
            {"rfilename": "Q4/model-00002-of-00002.gguf", "size": 200},
            {"rfilename": "Q8/model-00001-of-00002.gguf", "size": 999},
        ]
    }

    size, source = _weight_artifact_bytes(metadata, "Q4/model-00001-of-00002.gguf")

    assert size == 300
    assert source == "HF GGUF (2 file(s)): Q4/model-00001-of-00002.gguf"


def test_selected_zero_based_split_gguf_sums_every_shard() -> None:
    metadata = {
        "siblings": [
            {"rfilename": "Q4/model-00000-of-00002.gguf", "size": 100},
            {"rfilename": "Q4/model-00001-of-00002.gguf", "size": 200},
        ]
    }

    size, source = _weight_artifact_bytes(metadata, "Q4/model-00000-of-00002.gguf")

    assert size == 300
    assert source == "HF GGUF (2 file(s)): Q4/model-00000-of-00002.gguf"


def test_multiple_gguf_variants_require_a_blob_url() -> None:
    metadata = {
        "siblings": [
            {"rfilename": "model-Q4.gguf", "size": 100},
            {"rfilename": "model-Q8.gguf", "size": 200},
        ]
    }

    with pytest.raises(HuggingFaceError, match="multiple GGUF variants"):
        _weight_artifact_bytes(metadata)


def test_gpt_oss_index_rejects_missing_layers() -> None:
    config = {"model_type": "gpt_oss", "num_hidden_layers": 4}
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-1.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-1.safetensors",
        }
    }

    with pytest.raises(HuggingFaceError, match=r"missing layers \[2, 3\]"):
        _validate_gpt_oss_index(config, index)


def test_gpt_oss_index_accepts_complete_layer_coverage() -> None:
    config = {"model_type": "gpt_oss", "num_hidden_layers": 2}
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-1.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-2.safetensors",
        }
    }

    _validate_gpt_oss_index(config, index)


@pytest.mark.parametrize(
    ("layers", "dtypes", "expected_gib"),
    [
        (24, {"BF16": 1, "U8": 1}, 16),
        (24, {"BF16": 1}, 48),
        (36, {"BF16": 1, "U8": 1}, 60),
    ],
)
def test_gpt_oss_runtime_weight_floors(
    layers: int,
    dtypes: dict[str, int],
    expected_gib: int,
) -> None:
    floor, source = _gpt_oss_runtime_weight_floor(
        {"model_type": "gpt_oss", "num_hidden_layers": layers},
        artifact_bytes=10 * GIB,
        weight_dtypes=dtypes,
    )

    assert floor == expected_gib * GIB
    assert source and "OpenAI approximate" in source


def test_fetches_standalone_hf_quant_config(monkeypatch) -> None:
    config = {
        "model_type": "inkling_mm_model",
        "text_config": {"num_hidden_layers": 66, "torch_dtype": "bfloat16"},
    }
    repository = {
        "sha": "abc123",
        "siblings": [
            {"rfilename": "model.safetensors", "size": 100},
            {"rfilename": "hf_quant_config.json", "size": 100},
        ],
        "safetensors": {"parameters": {"BF16": 10, "U8": 80}},
    }
    external = {
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "none",
            "group_size": 16,
            "exclude_modules": ["*shared_experts*", "*qkvr*"],
        }
    }

    def fake_fetch(url, **kwargs):
        if url.endswith("hf_quant_config.json"):
            return external
        if url.endswith("config.json"):
            return config
        return repository

    monkeypatch.setattr("kvfit.hf._fetch_json", fake_fetch)

    metadata = fetch_model_metadata("thinkingmachines/Inkling-NVFP4")

    assert metadata.quantization == "nvfp4"
    assert metadata.quantization_source == "hf_quant_config.json: quantization.quant_algo"
    assert metadata.quantization_scope == "mixed"
    assert metadata.quantization_excluded_modules == 2
    assert metadata.kv_cache_quantization == "none"
    assert metadata.quantization_config_path == "hf_quant_config.json"
    assert metadata.quantization_config == external
    assert any("mixed nvfp4" in warning for warning in metadata.warnings)
