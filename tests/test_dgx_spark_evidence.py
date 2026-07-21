from __future__ import annotations

from kvfit.architectures import estimate_cache

HOWTOSPARK_QWEN_RECIPE = "https://howtospark.com/recipes/qwen3-6-35b-a3b-nvfp4-fast"
PINNED_REVISION = "1c3f884bc99aac2524f6d49bcbac8c88401afd66"


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
