from __future__ import annotations

import json
from pathlib import Path

import pytest
from kvfit.cache_layout import apply_cache_layout

from kvfit.architectures import estimate_cache
from kvfit.models import UnsupportedArchitecture

FIXTURES = Path(__file__).parent / "fixtures" / "model_configs"


def config():
    data = json.loads((FIXTURES / "zai-org--GLM-5.3.json").read_text())
    return data.get("config", data)


@pytest.mark.parametrize(
    "profile,dtype,cell",
    [
        ("sglang-dsa-raw", "fp8", 47700),
        ("sglang-dsa-scaled", "fp8", 53940),
        ("sglang-dsa-scaled", "bf16", 92628),
    ],
)
@pytest.mark.parametrize(
    "context,rounded", [(1, 64), (63, 64), (64, 64), (65, 128), (1048576, 1048576)]
)
def test_physical_cache_shapes_match_upstream(profile, dtype, cell, context, rounded):
    cfg = config()
    base = estimate_cache(cfg, context_tokens=context, kv_bytes=1 if dtype == "fp8" else 2)
    result = apply_cache_layout(base, cfg, profile=profile, kv_dtype=dtype)
    assert result.total_bytes == rounded * cell
    assert sum(c.bytes for c in result.fixed_components) == 64 * cell
    assert result.confidence == "upstream-storage-formula"


def test_mtp_and_non_elided_index_change_physical_storage():
    cfg = config()
    base = estimate_cache(cfg, context_tokens=1048576, kv_bytes=1)
    draft = apply_cache_layout(base, cfg, profile="sglang-dsa-scaled", kv_dtype="fp8", mtp=True)
    all_layers = apply_cache_layout(
        base, cfg, profile="sglang-dsa-scaled", kv_dtype="fp8", indexer_all_layers=True
    )
    assert draft.total_gib == 54728 / 1024
    assert all_layers.total_gib == (78 * (656 + 132)) / 1024


def test_logical_estimate_remains_logical():
    cfg = config()
    base = estimate_cache(cfg, context_tokens=1048576, kv_bytes=1)
    assert apply_cache_layout(base, cfg, profile="logical", kv_dtype="fp8") is base
    assert base.total_gib == 46.5


def test_physical_profile_refuses_unsupported_architecture():
    cfg = {
        "model_type": "llama",
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 1,
        "hidden_size": 8,
    }
    base = estimate_cache(cfg, context_tokens=128, kv_bytes=2)
    with pytest.raises(UnsupportedArchitecture):
        apply_cache_layout(base, cfg, profile="sglang-dsa-raw", kv_dtype="bf16")


def test_sglang_index_cache_cannot_silently_be_bf16():
    cfg = config()
    base = estimate_cache(cfg, context_tokens=128, kv_bytes=2)
    with pytest.raises(ValueError, match="index"):
        apply_cache_layout(base, cfg, profile="sglang-dsa-raw", kv_dtype="bf16", index_dtype="bf16")
