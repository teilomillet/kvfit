from __future__ import annotations

import pytest

from kvfit.architectures import estimate_cache
from kvfit.models import GIB, UnsupportedArchitecture


def test_standard_gqa_formula() -> None:
    config = {
        "model_type": "llama",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }

    estimate = estimate_cache(config, context_tokens=128 * 1024, kv_bytes=2)

    assert estimate.architecture == "standard-gqa-mqa"
    assert estimate.total_bytes == 16 * GIB
    assert estimate.kv_parallel_heads == 8


def test_explicit_sliding_full_layer_schedule() -> None:
    config = {
        "model_type": "gemma",
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sliding_window": 1024,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
    }

    estimate = estimate_cache(config, context_tokens=8192, kv_bytes=2)

    expected_tokens_across_layers = 3 * 1024 + 8192
    expected = 2 * expected_tokens_across_layers * 4 * 128 * 2
    assert estimate.architecture == "standard-full-sliding-hybrid"
    assert estimate.total_bytes == expected


def test_qwen_style_bottom_sliding_layers() -> None:
    config = {
        "model_type": "qwen2",
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sliding_window": 1024,
        "use_sliding_window": True,
        "max_window_layers": 3,
    }

    estimate = estimate_cache(config, context_tokens=8192)

    expected_tokens_across_layers = 3 * 1024 + 8192
    expected = 2 * expected_tokens_across_layers * 4 * 128 * 2
    assert estimate.architecture == "standard-full-sliding-hybrid"
    assert estimate.total_bytes == expected


@pytest.mark.parametrize(
    ("layers", "expected_gib"),
    [
        (24, 3.0029296875),
        (36, 4.50439453125),
    ],
)
def test_gpt_oss_128k_reference_formula(layers: int, expected_gib: float) -> None:
    config = {
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "num_hidden_layers": layers,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "layer_types": [
            "sliding_attention" if index % 2 == 0 else "full_attention" for index in range(layers)
        ],
    }

    estimate = estimate_cache(config, context_tokens=128 * 1024, kv_bytes=2)

    assert estimate.architecture == "gpt-oss-alternating-gqa"
    assert estimate.total_gib == pytest.approx(expected_gib)
    assert estimate.kv_parallel_heads == 8


def test_gpt_oss_changed_schedule_fails_closed() -> None:
    config = {
        "model_type": "gpt_oss",
        "num_hidden_layers": 2,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "layer_types": ["full_attention", "full_attention"],
    }

    with pytest.raises(UnsupportedArchitecture, match="layer schedule differs"):
        estimate_cache(config, context_tokens=4096)


def _inkling_config() -> dict[str, object]:
    return {
        "model_type": "inkling_mm_model",
        "architectures": ["InklingForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 66,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 6144,
            "local_layer_ids": [index for index in range(66) if (index + 1) % 6],
            "sliding_window_size": 512,
            "swa_num_attention_heads": 64,
            "swa_num_key_value_heads": 16,
            "swa_head_dim": 128,
            "use_sconv": True,
            "sconv_kernel_size": 4,
        },
    }


def test_inkling_one_million_context_includes_attention_and_sconv() -> None:
    estimate = estimate_cache(_inkling_config(), context_tokens=1024**2, kv_bytes=2)

    expected_global = 2 * 11 * 1024**2 * 8 * 128 * 2
    expected_local = 2 * 55 * 512 * 16 * 128 * 2
    expected_global_sconv = 11 * 3 * (2 * 8 * 128 + 2 * 6144) * 2
    expected_local_sconv = 55 * 3 * (2 * 16 * 128 + 2 * 6144) * 2
    assert estimate.architecture == "inkling-full-sliding-sconv"
    assert estimate.confidence == "reference-checked"
    assert estimate.total_bytes == (
        expected_global + expected_local + expected_global_sconv + expected_local_sconv
    )
    assert [component.name for component in estimate.components] == [
        "global-kv",
        "sliding-kv",
        "global-sconv-history",
        "local-sconv-history",
    ]
    assert [component.tp_parallel_units for component in estimate.components] == [8, 16, 8, 16]


def test_inkling_sconv_remains_bf16_when_attention_uses_fp4() -> None:
    bf16 = estimate_cache(_inkling_config(), context_tokens=4096, kv_bytes=2)
    fp4 = estimate_cache(_inkling_config(), context_tokens=4096, kv_bytes=0.5)

    assert fp4.components[0].bytes == bf16.components[0].bytes / 4
    assert fp4.components[1].bytes == bf16.components[1].bytes / 4
    assert fp4.components[2].bytes == bf16.components[2].bytes
    assert fp4.components[3].bytes == bf16.components[3].bytes


def test_inkling_changed_layer_schedule_fails_closed() -> None:
    config = _inkling_config()
    text_config = config["text_config"]
    assert isinstance(text_config, dict)
    text_config["local_layer_ids"] = list(range(55))

    with pytest.raises(UnsupportedArchitecture, match="layer schedule differs"):
        estimate_cache(config, context_tokens=4096)


@pytest.mark.parametrize("layers", [24, 36])
@pytest.mark.parametrize("context_tokens", [64, 128, 4096, 128 * 1024])
@pytest.mark.parametrize("kv_bytes", [1, 2])
def test_gpt_oss_matches_independent_formula_across_contexts(
    layers: int,
    context_tokens: int,
    kv_bytes: int,
) -> None:
    config = {
        "model_type": "gpt_oss",
        "num_hidden_layers": layers,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "sliding_window": 128,
        "layer_types": [
            "sliding_attention" if index % 2 == 0 else "full_attention" for index in range(layers)
        ],
    }

    estimate = estimate_cache(config, context_tokens=context_tokens, kv_bytes=kv_bytes)

    full_layers = layers // 2
    sliding_layers = layers // 2
    cached_token_layers = full_layers * context_tokens + sliding_layers * min(context_tokens, 128)
    expected = 2 * cached_token_layers * 8 * 64 * kv_bytes
    assert estimate.total_bytes == expected


def test_deepseek_v32_includes_indexer_reference_case() -> None:
    config = {
        "model_type": "deepseek_v32",
        "architectures": ["DeepseekV32ForCausalLM"],
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
    }

    estimate = estimate_cache(
        config,
        context_tokens=1024**2,
        kv_bytes=2,
        index_bytes=2,
    )

    assert estimate.architecture == "deepseek-v3.2-dsa"
    assert estimate.total_gib == pytest.approx(83.875)
    assert [component.name for component in estimate.components] == [
        "mla",
        "lightning-indexer",
    ]


def test_deepseek_v4_matches_vllm_published_reference() -> None:
    config = {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [4] * 30 + [128] * 31,
    }

    estimate = estimate_cache(
        config,
        context_tokens=1024**2,
        kv_bytes=2,
        index_bytes=2,
    )

    assert estimate.architecture == "deepseek-v4-compressed-hybrid"
    assert estimate.total_gib == pytest.approx(9.623, abs=0.01)
    assert estimate.kv_parallel_heads == 1


def test_deepseek_v4_flash_live_config_snapshot() -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0],
    }

    estimate = estimate_cache(config, context_tokens=1024**2)

    assert estimate.total_gib == pytest.approx(6.723999)
    assert any("2 sliding-only, 21 C4 CSA, 20 C128 HCA" in note for note in estimate.notes)


def test_deepseek_v4_ignores_only_declared_post_model_entries() -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [4, 128, 0],
    }

    estimate = estimate_cache(config, context_tokens=4096)

    assert any("Ignored 1 post-model" in note for note in estimate.notes)
    assert any("0 sliding-only, 1 C4 CSA, 1 C128 HCA" in note for note in estimate.notes)


def test_unknown_deepseek_v4_compression_ratio_fails_closed() -> None:
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [16],
    }

    with pytest.raises(UnsupportedArchitecture, match="unverified compression ratios"):
        estimate_cache(config, context_tokens=4096)


def test_incomplete_recurrent_hybrid_fails_closed() -> None:
    config = {
        "model_type": "qwen3_next",
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_attn_config": {"linear_attn_period": 4},
    }

    with pytest.raises(UnsupportedArchitecture, match="explicit layer_types"):
        estimate_cache(config, context_tokens=128 * 1024)


def test_qwen_gated_delta_hybrid_matches_independent_state_shapes() -> None:
    config = {
        "model_type": "qwen3_next",
        "num_hidden_layers": 8,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }

    estimate = estimate_cache(config, context_tokens=8192, kv_bytes=2)

    full_kv = 2 * 2 * 8192 * 2 * 256 * 2
    conv_state = 6 * (2 * 16 * 128 + 32 * 128) * 3 * 2
    recurrent_state = 6 * 32 * 128 * 128 * 4
    assert estimate.architecture == "qwen-gated-delta-hybrid"
    assert estimate.total_bytes == full_kv + conv_state + recurrent_state
    assert [component.name for component in estimate.components] == [
        "full-kv",
        "gdn-conv-history",
        "gdn-recurrent-state",
    ]


def test_nemotron_h_matches_mamba2_and_attention_state_shapes() -> None:
    config = {
        "model_type": "nemotron_h",
        "num_hidden_layers": 4,
        "hybrid_override_pattern": "M*E-",
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 16,
        "mamba_head_dim": 64,
        "ssm_state_size": 32,
        "n_groups": 4,
        "conv_kernel": 4,
    }

    estimate = estimate_cache(config, context_tokens=4096, kv_bytes=2)

    attention = 2 * 1 * 4096 * 2 * 128 * 2
    conv = 1 * (16 * 64 + 2 * 4 * 32) * 3 * 2
    temporal = 1 * 16 * 64 * 32 * 4
    assert estimate.architecture == "nemotron-h-mamba2-attention"
    assert estimate.total_bytes == attention + conv + temporal


def test_gemma4_uses_distinct_global_and_sliding_geometry() -> None:
    config = {
        "model_type": "gemma4_text",
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "num_global_key_value_heads": 2,
        "global_head_dim": 512,
        "sliding_window": 1024,
        "num_kv_shared_layers": 0,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    }

    estimate = estimate_cache(config, context_tokens=8192, kv_bytes=1)

    global_kv = 2 * 1 * 8192 * 2 * 512
    local_kv = 2 * 5 * 1024 * 8 * 256
    assert estimate.architecture == "gemma4-global-sliding-hybrid"
    assert estimate.total_bytes == global_kv + local_kv


def test_minimax_m3_includes_sparse_index_key_cache() -> None:
    config = {
        "model_type": "minimax_m3",
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_attention_freq": [0, 1, 0, 1],
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 64,
        },
    }

    estimate = estimate_cache(config, context_tokens=4096, kv_bytes=1, index_bytes=2)

    main_kv = 2 * 4 * 4096 * 4 * 128
    index_keys = 2 * 4096 * 4 * 64 * 2
    assert estimate.architecture == "minimax-m3-sparse-indexed-gqa"
    assert estimate.total_bytes == main_kv + index_keys


def test_unknown_model_type_fails_closed() -> None:
    config = {
        "model_type": "brand_new_attention",
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }

    with pytest.raises(UnsupportedArchitecture, match="checked standard-attention set"):
        estimate_cache(config, context_tokens=4096)


def test_implicit_alternating_window_pattern_fails_closed() -> None:
    config = {
        "model_type": "gemma2",
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "sliding_window": 4096,
        "sliding_window_pattern": 2,
    }

    with pytest.raises(UnsupportedArchitecture, match="alternating sliding/global"):
        estimate_cache(config, context_tokens=32768)
