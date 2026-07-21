from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from kvfit.models import CacheEstimate


@dataclass(frozen=True)
class OracleResult:
    passed: bool
    expected_bytes: float
    actual_bytes: float
    relative_error: float
    formula: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "expected_bytes": self.expected_bytes,
            "actual_bytes": self.actual_bytes,
            "relative_error": self.relative_error,
            "formula": self.formula,
        }


def _text_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = raw.get("text_config")
    if isinstance(nested, Mapping) and nested.get("num_hidden_layers") is not None:
        return nested
    return raw


def _number(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"oracle is missing numeric {key}")
    return int(value)


def _head_dim(config: Mapping[str, Any]) -> int:
    value = config.get("head_dim")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return _number(config, "hidden_size") // _number(config, "num_attention_heads")


def _standard_bytes(config: Mapping[str, Any], context: int, kv_bytes: float) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    query_heads = _number(config, "num_attention_heads")
    kv_heads = int(config.get("num_key_value_heads", config.get("num_kv_heads", query_heads)))
    head_dim = _head_dim(config)
    raw_types = config.get("layer_types")
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        schedule = [str(value).lower() for value in raw_types[:layers]]
    elif config.get("sliding_window") and config.get("use_sliding_window") is not False:
        bottom = config.get("max_window_layers")
        if isinstance(bottom, int):
            schedule = ["sliding_attention"] * bottom + ["full_attention"] * (layers - bottom)
        else:
            schedule = ["sliding_attention"] * layers
    else:
        schedule = ["full_attention"] * layers
    window = int(config.get("sliding_window", context))
    token_layer_sum = sum(
        min(context, window) if "sliding" in kind else context for kind in schedule
    )
    expected = 2 * token_layer_sum * kv_heads * head_dim * kv_bytes
    return expected, "independent per-layer K+V sum"


def _mla_bytes(
    config: Mapping[str, Any], context: int, kv_bytes: float, index_bytes: float, indexed: bool
) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    expected = (
        layers
        * context
        * (_number(config, "kv_lora_rank") + _number(config, "qk_rope_head_dim"))
        * kv_bytes
    )
    if indexed:
        expected += layers * context * _number(config, "index_head_dim") * index_bytes
    return expected, "independent MLA latent/RoPE plus index-key sum"


def _deepseek_v4_bytes(
    config: Mapping[str, Any], context: int, kv_bytes: float, index_bytes: float
) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    head_dim = _number(config, "head_dim")
    index_dim = _number(config, "index_head_dim")
    window = _number(config, "sliding_window")
    ratios = [int(value) for value in config["compress_ratios"][:layers]]
    csa = int(config.get("compress_rate_csa", 4))
    hca = int(config.get("compress_rate_hca", 128))
    expected = layers * min(context, window) * head_dim * kv_bytes
    for ratio in ratios:
        entries = math.ceil(context / ratio) if ratio else 0
        if ratio == csa:
            expected += entries * (head_dim * kv_bytes + index_dim * index_bytes)
        elif ratio == hca:
            expected += entries * head_dim * kv_bytes
    return expected, "independent V4 layer-by-layer compression sum"


def _qwen_gdn_bytes(config: Mapping[str, Any], context: int, kv_bytes: float) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    raw_types = config.get("layer_types")
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        schedule = [str(value).lower() for value in raw_types]
    else:
        interval = _number(config, "full_attention_interval")
        schedule = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layers)
        ]
    full = schedule.count("full_attention")
    linear = schedule.count("linear_attention")
    kv_heads = _number(config, "num_key_value_heads")
    head_dim = _head_dim(config)
    k_heads = _number(config, "linear_num_key_heads")
    v_heads = _number(config, "linear_num_value_heads")
    k_dim = _number(config, "linear_key_head_dim")
    v_dim = _number(config, "linear_value_head_dim")
    kernel = _number(config, "linear_conv_kernel_dim")
    full_state = 2 * full * context * kv_heads * head_dim * kv_bytes
    conv_state = linear * (2 * k_heads * k_dim + v_heads * v_dim) * (kernel - 1) * kv_bytes
    recurrent_state = linear * v_heads * v_dim * k_dim * 4
    return full_state + conv_state + recurrent_state, "independent GDN state-shape product"


def _nemotron_bytes(config: Mapping[str, Any], context: int, kv_bytes: float) -> tuple[float, str]:
    pattern = str(config["hybrid_override_pattern"])
    attention_layers = pattern.count("*")
    mamba_layers = pattern.count("M")
    attention = (
        2
        * attention_layers
        * context
        * _number(config, "num_key_value_heads")
        * _head_dim(config)
        * kv_bytes
    )
    mamba_heads = _number(config, "mamba_num_heads")
    mamba_head_dim = _number(config, "mamba_head_dim")
    state_size = _number(config, "ssm_state_size")
    groups = _number(config, "n_groups")
    kernel = _number(config, "conv_kernel")
    conv_width = mamba_heads * mamba_head_dim + 2 * groups * state_size
    conv = mamba_layers * conv_width * (kernel - 1) * kv_bytes
    temporal = mamba_layers * mamba_heads * mamba_head_dim * state_size * 4
    return attention + conv + temporal, "independent Mamba-2 and attention state-shape product"


def _gemma4_bytes(config: Mapping[str, Any], context: int, kv_bytes: float) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    shared = int(config.get("num_kv_shared_layers", 0))
    schedule = [str(value).lower() for value in config["layer_types"][: layers - shared]]
    full = schedule.count("full_attention")
    sliding = schedule.count("sliding_attention")
    local_heads = _number(config, "num_key_value_heads")
    local_dim = _number(config, "head_dim")
    global_heads = int(config.get("num_global_key_value_heads", local_heads))
    global_dim = int(config.get("global_head_dim", local_dim))
    window = _number(config, "sliding_window")
    expected = 2 * full * context * global_heads * global_dim * kv_bytes
    expected += 2 * sliding * min(context, window) * local_heads * local_dim * kv_bytes
    return expected, "independent Gemma 4 local/global per-layer sum"


def _minimax_bytes(
    config: Mapping[str, Any], context: int, kv_bytes: float, index_bytes: float
) -> tuple[float, str]:
    layers = _number(config, "num_hidden_layers")
    main = 2 * layers * context * _number(config, "num_key_value_heads") * _head_dim(config)
    main *= kv_bytes
    sparse = config["sparse_attention_config"]
    frequency = sparse["sparse_attention_freq"]
    sparse_layers = sum(value != 0 for value in frequency)
    index = sparse_layers * context * int(sparse["sparse_num_index_heads"])
    index *= int(sparse["sparse_index_dim"]) * index_bytes
    return main + index, "independent MiniMax main-KV plus index-key sum"


def check_cache_estimate(
    raw_config: Mapping[str, Any],
    estimate: CacheEstimate,
    *,
    kv_bytes: float,
    index_bytes: float,
) -> OracleResult:
    """Cross-check an estimate without calling the production formula helpers."""
    config = _text_config(raw_config)
    architecture = estimate.architecture
    context = estimate.context_tokens
    if architecture in {
        "standard-mha",
        "standard-gqa-mqa",
        "standard-full-sliding-hybrid",
        "standard-pure-sliding",
        "gpt-oss-alternating-gqa",
    }:
        expected, formula = _standard_bytes(config, context, kv_bytes)
    elif architecture == "deepseek-mla":
        expected, formula = _mla_bytes(config, context, kv_bytes, index_bytes, False)
    elif architecture in {"deepseek-v3.2-dsa", "glm-dsa-mla"}:
        expected, formula = _mla_bytes(config, context, kv_bytes, index_bytes, True)
    elif architecture == "deepseek-v4-compressed-hybrid":
        expected, formula = _deepseek_v4_bytes(config, context, kv_bytes, index_bytes)
    elif architecture == "qwen-gated-delta-hybrid":
        expected, formula = _qwen_gdn_bytes(config, context, kv_bytes)
    elif architecture == "nemotron-h-mamba2-attention":
        expected, formula = _nemotron_bytes(config, context, kv_bytes)
    elif architecture == "gemma4-global-sliding-hybrid":
        expected, formula = _gemma4_bytes(config, context, kv_bytes)
    elif architecture == "minimax-m3-sparse-indexed-gqa":
        expected, formula = _minimax_bytes(config, context, kv_bytes, index_bytes)
    elif architecture == "inkling-full-sliding-sconv":
        layers = 66
        local_layers = 55
        global_layers = layers - local_layers
        global_kv = 2 * global_layers * context * 8 * 128 * kv_bytes
        local_kv = 2 * local_layers * min(context, 512) * 16 * 128 * kv_bytes
        sconv = global_layers * 3 * (2 * 8 * 128 + 2 * 6144) * 2
        sconv += local_layers * 3 * (2 * 16 * 128 + 2 * 6144) * 2
        expected = global_kv + local_kv + sconv
        formula = "independent Inkling attention and four-stream sconv sum"
    else:
        raise ValueError(f"oracle has no formula for {architecture}")
    actual = estimate.total_bytes
    relative_error = abs(actual - expected) / max(abs(expected), 1.0)
    return OracleResult(
        passed=relative_error <= 1e-12,
        expected_bytes=expected,
        actual_bytes=actual,
        relative_error=relative_error,
        formula=formula,
    )
