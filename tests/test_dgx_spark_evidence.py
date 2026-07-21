from __future__ import annotations

from kvfit.architectures import estimate_cache

HOWTOSPARK_QWEN_RECIPE = "https://howtospark.com/recipes/qwen3-6-35b-a3b-nvfp4-fast"
PINNED_REVISION = "1c3f884bc99aac2524f6d49bcbac8c88401afd66"
HOWTOSPARK_DSPARK_RECIPE = "https://howtospark.com/recipes/deepseek-v4-flash-dspark-dual-spark-1m"
PINNED_DSPARK_REVISION = "62af8fffb2f7030cac4de2f0169f5b8d1101b646"


def test_qwen36_cache_formula_matches_measured_dgx_spark_pool_within_one_percent() -> None:
    # Minimal state-shape fields from the pinned Hugging Face config. HowToSpark
    # reports that a measured 4 GiB no-draft vLLM pool holds 406,424 tokens.
    # This is an external hardware regression, not a second algebraic oracle.
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": [
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(40)
            ],
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
        }
    }
    context_tokens = 262_144
    estimate = estimate_cache(config, context_tokens=context_tokens, kv_bytes=1)
    measured_gib_per_context = 4 / (406_424 / context_tokens)
    relative_error = abs(estimate.total_gib - measured_gib_per_context) / measured_gib_per_context

    assert estimate.architecture == "qwen-gated-delta-hybrid"
    assert relative_error < 0.01, (
        f"{PINNED_REVISION=} {HOWTOSPARK_QWEN_RECIPE=} {relative_error=:.3%}"
    )


def test_deepseek_v4_dspark_logical_cache_tracks_measured_pool_within_six_percent() -> None:
    # The pinned recipe measured a 17.21 GiB/rank runtime KV allocation holding
    # 2,555,830 tokens. Unlike the Qwen no-draft oracle above, this pool includes
    # backend layout and allocation overhead. The logical result must remain
    # below it and close enough to catch a missing target-cache term. The tiny
    # draft term is below this measurement's resolving power and is checked
    # separately against the pinned implementation in test_architectures_dspark.py.
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 128,
        "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0, 0, 0],
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    }
    context_tokens = 1024**2
    estimate = estimate_cache(config, context_tokens=context_tokens, kv_bytes=2)
    measured_gib_per_context = 17.21 / (2_555_830 / context_tokens)
    relative_gap = (measured_gib_per_context - estimate.total_gib) / measured_gib_per_context

    assert estimate.architecture == "deepseek-v4-compressed-hybrid+dspark"
    assert estimate.components[-1].name == "dspark-draft-kv"
    assert 0 < relative_gap < 0.06, (
        f"{PINNED_DSPARK_REVISION=} {HOWTOSPARK_DSPARK_RECIPE=} {relative_gap=:.3%}"
    )
