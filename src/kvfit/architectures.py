from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from kvfit.models import (
    CacheComponent,
    CacheEstimate,
    SpeculativeCacheEstimate,
    SpeculativeDecodingEstimate,
    UnsupportedArchitecture,
)

VLLM_DEEPSEEK_V4_REFERENCE = (
    "https://vllm.ai/blog/2026/04/24/deepseek-v4.html#the-math-behind-"
    "deepseek-v4s-attention-mechanism"
)
VLLM_DEEPSEEK_V4_DSPARK_REFERENCE = (
    "https://github.com/vllm-project/vllm/blob/"
    "752a3a504485790a2e8491cacbb35c137339ad34/"
    "vllm/models/deepseek_v4/nvidia/dspark.py"
)
HF_KV_REFERENCE = "https://huggingface.co/docs/transformers/kv_cache"
GPT_OSS_REFERENCE = "https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py"
INKLING_REFERENCE = "https://vllm.ai/blog/2026/07/15/inkling"
QWEN_GDN_REFERENCE = (
    "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/mamba_utils.py"
)
NEMOTRON_H_REFERENCE = (
    "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_h.py"
)
GEMMA4_REFERENCE = (
    "https://github.com/huggingface/transformers/blob/main/src/transformers/"
    "models/gemma4/modeling_gemma4.py"
)
MINIMAX_M3_REFERENCE = (
    "https://github.com/vllm-project/vllm/blob/main/vllm/models/minimax_m3/nvidia/model.py"
)

# These families use the conventional per-layer K/V state represented by the
# Transformers config fields consumed below. New model types are deliberately
# rejected until their cache behavior is checked.
STANDARD_ATTENTION_MODEL_TYPES = frozenset(
    {
        "cohere",
        "cohere2",
        "cohere2_moe",
        "falcon",
        "gemma",
        "gemma2",
        "gemma3_text",
        "granite",
        "granite_moe",
        "internlm2",
        "internlm3",
        "llama",
        "mistral",
        "mixtral",
        "olmo",
        "olmo2",
        "olmoe",
        "opt",
        "phi3",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "stablelm",
        "starcoder2",
    }
)


def _layer_schedule(
    config: Mapping[str, Any],
    *,
    allowed: set[str],
    fallback_interval: int | None = None,
) -> list[str]:
    """Return a checked, explicit per-layer state schedule."""
    layers = _int(config, "num_hidden_layers")
    raw_types = config.get("layer_types")
    if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
        if len(raw_types) != layers:
            raise UnsupportedArchitecture(
                str(config.get("model_type", "unknown")),
                "layer_types length differs from num_hidden_layers",
                relevant_fields=("layer_types", "num_hidden_layers"),
            )
        selected = [str(value).lower() for value in raw_types]
    elif fallback_interval is not None:
        if fallback_interval < 1:
            raise UnsupportedArchitecture(
                str(config.get("model_type", "unknown")),
                "full_attention_interval must be positive",
                relevant_fields=("full_attention_interval",),
            )
        selected = [
            "full_attention" if (index + 1) % fallback_interval == 0 else "linear_attention"
            for index in range(layers)
        ]
    else:
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            "an explicit layer_types schedule is required",
            relevant_fields=("layer_types",),
        )
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            f"unverified layer types {unknown!r}",
            relevant_fields=("layer_types",),
        )
    return selected


def _int(config: Mapping[str, Any], key: str, *, minimum: int = 1) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            f"missing numeric {key!r}",
            relevant_fields=(key,),
        )
    result = int(value)
    if result < minimum:
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            f"invalid {key!r}={value!r}",
            relevant_fields=(key,),
        )
    return result


def _is_exact_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_exact_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _strict_int(config: Mapping[str, Any], key: str, *, minimum: int = 1) -> int:
    """Read an integer without accepting lossy numeric coercion."""
    value = config.get(key)
    if not _is_exact_integer(value):
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            f"missing or non-integer {key!r}",
            relevant_fields=(key,),
        )
    if value < minimum:
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            f"invalid {key!r}={value!r}",
            relevant_fields=(key,),
        )
    return value


def _text_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = raw.get("text_config")
    if isinstance(nested, Mapping) and nested.get("num_hidden_layers") is not None:
        return nested
    return raw


def _architecture_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = config.get("architectures", ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(name) for name in value)


def _head_dim(config: Mapping[str, Any], query_heads: int) -> int:
    explicit = config.get("head_dim")
    if isinstance(explicit, (int, float)) and not isinstance(explicit, bool) and explicit > 0:
        return int(explicit)
    hidden_size = _int(config, "hidden_size")
    if hidden_size % query_heads:
        raise UnsupportedArchitecture(
            str(config.get("model_type", "unknown")),
            "hidden_size is not divisible by num_attention_heads and head_dim is absent",
            relevant_fields=("hidden_size", "num_attention_heads", "head_dim"),
        )
    return hidden_size // query_heads


def _estimate_deepseek_mla(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
    index_bytes: float,
    with_indexer: bool,
) -> CacheEstimate:
    layers = _int(config, "num_hidden_layers")
    query_heads = _int(config, "num_attention_heads")
    kv_lora_rank = _int(config, "kv_lora_rank")
    rope_dim = _int(config, "qk_rope_head_dim", minimum=0)
    mla_bytes = layers * context_tokens * (kv_lora_rank + rope_dim) * kv_bytes
    components = [
        CacheComponent(
            name="mla",
            bytes=mla_bytes,
            detail=(
                f"{layers} layers × {context_tokens} tokens × "
                f"({kv_lora_rank} latent + {rope_dim} RoPE) × {kv_bytes:g} bytes"
            ),
        )
    ]
    architecture = "deepseek-mla"
    confidence = "architecture-formula"
    notes: list[str] = []
    if with_indexer:
        index_dim = _int(config, "index_head_dim")
        indexer_bytes = layers * context_tokens * index_dim * index_bytes
        components.append(
            CacheComponent(
                name="lightning-indexer",
                bytes=indexer_bytes,
                detail=(
                    f"{layers} layers × {context_tokens} tokens × "
                    f"{index_dim} index state × {index_bytes:g} bytes"
                ),
            )
        )
        architecture = "deepseek-v3.2-dsa"
        confidence = "reference-checked"
        notes.append("Includes the Lightning Indexer state omitted by MLA-only calculators.")
    return CacheEstimate(
        architecture=architecture,
        context_tokens=context_tokens,
        components=tuple(components),
        kv_parallel_heads=1,
        query_heads=query_heads,
        confidence=confidence,
        reference=VLLM_DEEPSEEK_V4_REFERENCE,
        notes=tuple(notes),
    )


def _deepseek_v4_dspark(
    config: Mapping[str, Any],
    *,
    base_layers: int,
    post_model_ratios: Sequence[Any],
    context_tokens: int,
    window: int,
    head_dim: int,
    kv_bytes: float,
) -> tuple[CacheComponent, SpeculativeDecodingEstimate] | None:
    required_fields = (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    )
    declared_fields = tuple(
        str(key) for key in config if isinstance(key, str) and key.startswith("dspark_")
    )
    if not declared_fields:
        return None
    unknown_fields = tuple(sorted(set(declared_fields) - set(required_fields)))
    if unknown_fields:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "unverified DSpark configuration fields cannot be sized safely",
            relevant_fields=tuple(dict.fromkeys((*required_fields, *unknown_fields))),
        )
    present_fields = tuple(key for key in required_fields if config.get(key) is not None)
    missing_fields = tuple(key for key in required_fields if config.get(key) is None)
    if missing_fields:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "partial DSpark configuration cannot be sized safely",
            relevant_fields=tuple(
                dict.fromkeys((*declared_fields, *present_fields, *missing_fields))
            ),
        )

    block_size = _strict_int(config, "dspark_block_size")
    _strict_int(config, "dspark_noise_token_id", minimum=0)
    _strict_int(config, "dspark_markov_rank")

    raw_target_ids = config.get("dspark_target_layer_ids")
    if isinstance(raw_target_ids, (str, bytes)):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "dspark_target_layer_ids must be an explicit sequence",
            relevant_fields=("dspark_target_layer_ids",),
        )
    if not isinstance(raw_target_ids, Sequence):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "dspark_target_layer_ids must be an explicit sequence",
            relevant_fields=("dspark_target_layer_ids",),
        )
    if not raw_target_ids:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "dspark_target_layer_ids must contain integer target-layer indices",
            relevant_fields=("dspark_target_layer_ids",),
        )
    if any(not _is_exact_integer(value) for value in raw_target_ids):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "dspark_target_layer_ids must contain integer target-layer indices",
            relevant_fields=("dspark_target_layer_ids",),
        )
    target_layer_ids = tuple(int(value) for value in raw_target_ids)
    if target_layer_ids != tuple(sorted(set(target_layer_ids))) or any(
        value < 0 or value >= base_layers for value in target_layer_ids
    ):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "dspark_target_layer_ids must be strictly increasing base-layer indices",
            relevant_fields=("dspark_target_layer_ids", "num_hidden_layers"),
        )

    if not post_model_ratios:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "DSpark requires one numeric post-model cache entry per draft layer",
            relevant_fields=("compress_ratios",),
        )
    if any(not _is_exact_number(value) for value in post_model_ratios):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "DSpark requires one numeric post-model cache entry per draft layer",
            relevant_fields=("compress_ratios",),
        )
    if any(value != 0 for value in post_model_ratios):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "only sliding-window DSpark draft layers are verified",
            relevant_fields=("compress_ratios", "sliding_window"),
        )

    declared_layers = config.get("n_mtp_layers")
    if declared_layers is None:
        # The pinned DeepSeek-V4 vLLM implementation defaults to three draft
        # layers when n_mtp_layers is absent. Do not generalize another count.
        if len(post_model_ratios) != 3:
            raise UnsupportedArchitecture(
                "deepseek_v4",
                "DSpark without n_mtp_layers must match the verified three-layer layout",
                relevant_fields=("n_mtp_layers", "compress_ratios"),
            )
        draft_layers = 3
    else:
        draft_layers = _strict_int(config, "n_mtp_layers")
        if draft_layers != len(post_model_ratios):
            raise UnsupportedArchitecture(
                "deepseek_v4",
                "n_mtp_layers differs from the post-model cache schedule",
                relevant_fields=("n_mtp_layers", "compress_ratios"),
            )
    if len(target_layer_ids) != draft_layers:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "DSpark target-layer count differs from its draft-layer count",
            relevant_fields=("dspark_target_layer_ids", "n_mtp_layers", "compress_ratios"),
        )

    entries = min(context_tokens, window)
    component = CacheComponent(
        name="dspark-draft-kv",
        bytes=draft_layers * entries * head_dim * kv_bytes,
        detail=(
            f"{draft_layers} DSpark draft layers × {entries} sliding-window entries × "
            f"{head_dim} shared K=V width × {kv_bytes:g} bytes"
        ),
        tp_parallel_units=1,
    )
    speculative = SpeculativeDecodingEstimate(
        method="dspark",
        packaging="integrated",
        declaration_source="model-config",
        runtime_enabled=None,
        draft_layers=draft_layers,
        checkpoint_block_size=block_size,
        target_layer_ids=target_layer_ids,
        cache_component=component.name,
        cache_modeled=True,
        runtime_buffers_modeled=False,
        performance_modeled=False,
        reference=VLLM_DEEPSEEK_V4_DSPARK_REFERENCE,
    )
    return component, speculative


def _estimate_deepseek_v4(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
    index_bytes: float,
) -> CacheEstimate:
    layers = _int(config, "num_hidden_layers")
    query_heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads")
    if kv_heads != 1:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "the verified V4 formula requires shared single-head K=V state",
            relevant_fields=("num_key_value_heads",),
        )
    head_dim = _int(config, "head_dim")
    index_dim = _int(config, "index_head_dim")
    window = _int(config, "sliding_window")
    raw_ratios = config.get("compress_ratios")
    if not isinstance(raw_ratios, Sequence) or isinstance(raw_ratios, (str, bytes)):
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "compress_ratios is required for layer-by-layer cache sizing",
            relevant_fields=("compress_ratios",),
        )
    if len(raw_ratios) < layers:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            "compress_ratios has fewer entries than num_hidden_layers",
            relevant_fields=("compress_ratios", "num_hidden_layers"),
        )
    # Configs may append one or more MTP draft-layer entries. The base model's
    # KV cache is determined by the first num_hidden_layers entries.
    ratios: list[int] = []
    for value in raw_ratios[:layers]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UnsupportedArchitecture(
                "deepseek_v4",
                "compress_ratios contains a non-numeric entry",
                relevant_fields=("compress_ratios",),
            )
        ratios.append(int(value))

    csa_rate = int(config.get("compress_rate_csa", 4))
    hca_rate = int(config.get("compress_rate_hca", 128))
    expected = {0, csa_rate, hca_rate}
    unknown = sorted(set(ratios) - expected)
    if unknown:
        raise UnsupportedArchitecture(
            "deepseek_v4",
            f"unverified compression ratios {unknown!r}",
            relevant_fields=(
                "compress_ratios",
                "compress_rate_csa",
                "compress_rate_hca",
            ),
        )

    counts = Counter(ratios)
    local_entries = min(context_tokens, window)
    local_bytes = layers * local_entries * head_dim * kv_bytes
    components = [
        CacheComponent(
            name="shared-local-kv",
            bytes=local_bytes,
            detail=(
                f"{layers} layers × {local_entries} local entries × "
                f"{head_dim} shared K=V width × {kv_bytes:g} bytes"
            ),
        )
    ]

    csa_layers = counts[csa_rate]
    if csa_layers:
        csa_entries = math.ceil(context_tokens / csa_rate)
        csa_kv_bytes = csa_layers * csa_entries * head_dim * kv_bytes
        csa_index_bytes = csa_layers * csa_entries * index_dim * index_bytes
        components.extend(
            (
                CacheComponent(
                    name=f"c{csa_rate}-compressed-kv",
                    bytes=csa_kv_bytes,
                    detail=(
                        f"{csa_layers} CSA layers × {csa_entries} compressed entries × "
                        f"{head_dim} shared K=V width × {kv_bytes:g} bytes"
                    ),
                ),
                CacheComponent(
                    name=f"c{csa_rate}-indexer",
                    bytes=csa_index_bytes,
                    detail=(
                        f"{csa_layers} CSA layers × {csa_entries} index entries × "
                        f"{index_dim} width × {index_bytes:g} bytes"
                    ),
                ),
            )
        )

    hca_layers = counts[hca_rate]
    if hca_layers:
        hca_entries = math.ceil(context_tokens / hca_rate)
        hca_bytes = hca_layers * hca_entries * head_dim * kv_bytes
        components.append(
            CacheComponent(
                name=f"c{hca_rate}-compressed-kv",
                bytes=hca_bytes,
                detail=(
                    f"{hca_layers} HCA layers × {hca_entries} compressed entries × "
                    f"{head_dim} shared K=V width × {kv_bytes:g} bytes"
                ),
            )
        )

    dspark = _deepseek_v4_dspark(
        config,
        base_layers=layers,
        post_model_ratios=raw_ratios[layers:],
        context_tokens=context_tokens,
        window=window,
        head_dim=head_dim,
        kv_bytes=kv_bytes,
    )
    speculative_decoding: SpeculativeDecodingEstimate | None = None
    if dspark is not None:
        draft_component, speculative_decoding = dspark
        components.append(draft_component)

    notes = [
        (
            f"Layer schedule: {counts[0]} sliding-only, {csa_layers} C{csa_rate} CSA, "
            f"{hca_layers} C{hca_rate} HCA."
        ),
        (
            "This is logical cache state; allocator page rounding and compressor residuals "
            "are runtime overhead."
        ),
    ]
    if speculative_decoding is not None:
        notes.extend(
            (
                (
                    f"Detected integrated DSpark with {speculative_decoding.draft_layers} "
                    f"draft layers; their logical sliding-window KV state is counted."
                ),
                (
                    "DSpark runtime speculative-token count, hidden-state buffers, CUDA graphs, "
                    "allocator workspace, acceptance, and speed remain runtime-only and require "
                    "target-engine calibration."
                ),
            )
        )
    elif len(raw_ratios) > layers:
        notes.append(
            f"Ignored {len(raw_ratios) - layers} post-model compression entry/entries, usually MTP."
        )
    estimate_fields: dict[str, Any] = {
        "architecture": (
            "deepseek-v4-compressed-hybrid+dspark"
            if speculative_decoding is not None
            else "deepseek-v4-compressed-hybrid"
        ),
        "context_tokens": context_tokens,
        "components": tuple(components),
        "kv_parallel_heads": 1,
        "query_heads": query_heads,
        "confidence": "reference-checked",
        "reference": VLLM_DEEPSEEK_V4_REFERENCE,
        "notes": tuple(notes),
    }
    if speculative_decoding is not None:
        return SpeculativeCacheEstimate(
            **estimate_fields,
            speculative_decoding=speculative_decoding,
        )
    return CacheEstimate(**estimate_fields)


def _estimate_standard(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    model_type = str(config.get("model_type", "unknown"))
    layers = _int(config, "num_hidden_layers")
    query_heads = _int(config, "num_attention_heads")
    kv_heads = int(config.get("num_key_value_heads", config.get("num_kv_heads", query_heads)))
    if kv_heads < 1:
        raise UnsupportedArchitecture(
            model_type,
            "invalid KV-head count",
            relevant_fields=("num_key_value_heads", "num_kv_heads"),
        )
    head_dim = _head_dim(config, query_heads)

    layer_types = config.get("layer_types")
    token_layers: list[tuple[str, int]]
    notes: list[str] = []
    architecture = "standard-mha" if kv_heads == query_heads else "standard-gqa-mqa"
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, (str, bytes)):
        if len(layer_types) < layers:
            raise UnsupportedArchitecture(
                model_type,
                "layer_types has fewer entries than num_hidden_layers",
                relevant_fields=("layer_types", "num_hidden_layers"),
            )
        selected = [str(value).lower() for value in layer_types[:layers]]
        allowed = {"full_attention", "attention", "sliding_attention"}
        unknown = sorted(set(selected) - allowed)
        if unknown:
            raise UnsupportedArchitecture(
                model_type,
                f"stateful nonstandard layer types {unknown!r} require an adapter",
                relevant_fields=("layer_types",),
            )
        if "sliding_attention" in selected:
            window = _int(config, "sliding_window")
            token_layers = [
                (
                    kind,
                    min(context_tokens, window) if kind == "sliding_attention" else context_tokens,
                )
                for kind in selected
            ]
            architecture = "standard-full-sliding-hybrid"
            notes.append(f"Uses explicit layer_types with a {window}-token sliding window.")
        else:
            token_layers = [(kind, context_tokens) for kind in selected]
    elif config.get("sliding_window_pattern") is not None:
        raise UnsupportedArchitecture(
            model_type,
            "implicit alternating sliding/global attention requires a checked adapter",
            relevant_fields=("sliding_window_pattern", "sliding_window"),
        )
    elif (
        config.get("sliding_window")
        and config.get("use_sliding_window") is True
        and config.get("max_window_layers") is not None
    ):
        window = _int(config, "sliding_window")
        sliding_layers = _int(config, "max_window_layers", minimum=0)
        if sliding_layers > layers:
            raise UnsupportedArchitecture(
                model_type,
                "max_window_layers exceeds num_hidden_layers",
                relevant_fields=("max_window_layers", "num_hidden_layers"),
            )
        token_layers = [("sliding_attention", min(context_tokens, window))] * sliding_layers + [
            ("full_attention", context_tokens)
        ] * (layers - sliding_layers)
        architecture = "standard-full-sliding-hybrid"
        notes.append(f"Uses {sliding_layers} bottom sliding layers with a {window}-token window.")
    elif config.get("sliding_window") and config.get("use_sliding_window") is not False:
        window = _int(config, "sliding_window")
        token_layers = [("sliding_attention", min(context_tokens, window))] * layers
        architecture = "standard-pure-sliding"
        notes.append("Treats every layer as sliding because the config explicitly enables it.")
    else:
        token_layers = [("full_attention", context_tokens)] * layers

    counts: Counter[tuple[str, int]] = Counter(token_layers)
    components = []
    for (kind, cached_tokens), count in sorted(counts.items()):
        size = 2 * count * cached_tokens * kv_heads * head_dim * kv_bytes
        components.append(
            CacheComponent(
                name=kind.replace("_attention", "") + "-kv",
                bytes=size,
                detail=(
                    f"K+V × {count} layers × {cached_tokens} tokens × {kv_heads} KV heads "
                    f"× {head_dim} head dim × {kv_bytes:g} bytes"
                ),
            )
        )
    return CacheEstimate(
        architecture=architecture,
        context_tokens=context_tokens,
        components=tuple(components),
        kv_parallel_heads=kv_heads,
        query_heads=query_heads,
        confidence="architecture-formula",
        reference=HF_KV_REFERENCE,
        notes=tuple(notes),
    )


def _estimate_gpt_oss(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    layers = _int(config, "num_hidden_layers")
    query_heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads")
    head_dim = _int(config, "head_dim")
    window = _int(config, "sliding_window")
    if (query_heads, kv_heads, head_dim, window) != (64, 8, 64, 128):
        raise UnsupportedArchitecture(
            "gpt_oss",
            "attention dimensions differ from the checked OpenAI reference",
            relevant_fields=(
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "sliding_window",
            ),
        )
    layer_types = config.get("layer_types")
    expected = [
        "sliding_attention" if index % 2 == 0 else "full_attention" for index in range(layers)
    ]
    if not isinstance(layer_types, Sequence) or isinstance(layer_types, (str, bytes)):
        raise UnsupportedArchitecture(
            "gpt_oss",
            "explicit layer_types is required",
            relevant_fields=("layer_types",),
        )
    selected = [str(value).lower() for value in layer_types[:layers]]
    if len(layer_types) != layers or selected != expected:
        raise UnsupportedArchitecture(
            "gpt_oss",
            "layer schedule differs from alternating sliding/full reference attention",
            relevant_fields=("layer_types", "num_hidden_layers"),
        )
    estimate = _estimate_standard(
        config,
        context_tokens=context_tokens,
        kv_bytes=kv_bytes,
    )
    return replace(
        estimate,
        architecture="gpt-oss-alternating-gqa",
        confidence="reference-checked",
        reference=GPT_OSS_REFERENCE,
        notes=(
            *estimate.notes,
            "Weight quantization does not change KV precision; cache dtype is resolved separately.",
        ),
    )


def _estimate_inkling(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    """Model Inkling attention plus its four BF16 short-convolution streams."""
    expected = {
        "num_hidden_layers": 66,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "swa_num_attention_heads": 64,
        "swa_num_key_value_heads": 16,
        "swa_head_dim": 128,
        "sliding_window_size": 512,
        "sconv_kernel_size": 4,
        "hidden_size": 6144,
    }
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if config.get("use_sconv") is not True:
        mismatches.append("use_sconv")
    if mismatches:
        raise UnsupportedArchitecture(
            "inkling_mm_model",
            "attention or sconv dimensions differ from the checked Inkling release",
            relevant_fields=tuple(dict.fromkeys(mismatches)),
        )

    layers = expected["num_hidden_layers"]
    raw_local_ids = config.get("local_layer_ids")
    if not isinstance(raw_local_ids, Sequence) or isinstance(raw_local_ids, (str, bytes)):
        raise UnsupportedArchitecture(
            "inkling_mm_model",
            "explicit local_layer_ids is required",
            relevant_fields=("local_layer_ids",),
        )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_local_ids):
        raise UnsupportedArchitecture(
            "inkling_mm_model",
            "local_layer_ids contains a non-integer entry",
            relevant_fields=("local_layer_ids",),
        )
    local_ids = [int(value) for value in raw_local_ids]
    expected_local_ids = [index for index in range(layers) if (index + 1) % 6]
    if local_ids != expected_local_ids:
        raise UnsupportedArchitecture(
            "inkling_mm_model",
            "local/global layer schedule differs from the checked 55-local/11-global release",
            relevant_fields=("local_layer_ids", "num_hidden_layers"),
        )

    local_layers = len(local_ids)
    global_layers = layers - local_layers
    full_kv_heads = expected["num_key_value_heads"]
    local_kv_heads = expected["swa_num_key_value_heads"]
    head_dim = expected["head_dim"]
    hidden_size = expected["hidden_size"]
    local_entries = min(context_tokens, expected["sliding_window_size"])
    history = expected["sconv_kernel_size"] - 1

    # Attention K/V follows --kv-dtype. vLLM's Inkling short-convolution
    # implementation fixes its persistent history state to BF16.
    components = (
        CacheComponent(
            name="global-kv",
            bytes=2 * global_layers * context_tokens * full_kv_heads * head_dim * kv_bytes,
            detail=(
                f"K+V × {global_layers} layers × {context_tokens} tokens × "
                f"{full_kv_heads} KV heads × {head_dim} head dim × {kv_bytes:g} bytes"
            ),
            tp_parallel_units=full_kv_heads,
        ),
        CacheComponent(
            name="sliding-kv",
            bytes=2 * local_layers * local_entries * local_kv_heads * head_dim * kv_bytes,
            detail=(
                f"K+V × {local_layers} layers × {local_entries} tokens × "
                f"{local_kv_heads} KV heads × {head_dim} head dim × {kv_bytes:g} bytes"
            ),
            tp_parallel_units=local_kv_heads,
        ),
        CacheComponent(
            name="global-sconv-history",
            bytes=(global_layers * history * (2 * full_kv_heads * head_dim + 2 * hidden_size) * 2),
            detail=(
                f"{global_layers} layers × {history} prior states × "
                f"(global K+V + attention-output + MoE-output streams) × 2 BF16 bytes"
            ),
            tp_parallel_units=full_kv_heads,
        ),
        CacheComponent(
            name="local-sconv-history",
            bytes=(local_layers * history * (2 * local_kv_heads * head_dim + 2 * hidden_size) * 2),
            detail=(
                f"{local_layers} layers × {history} prior states × "
                f"(local K+V + attention-output + MoE-output streams) × 2 BF16 bytes"
            ),
            tp_parallel_units=local_kv_heads,
        ),
    )
    return CacheEstimate(
        architecture="inkling-full-sliding-sconv",
        context_tokens=context_tokens,
        components=components,
        kv_parallel_heads=full_kv_heads,
        query_heads=expected["num_attention_heads"],
        confidence="reference-checked",
        reference=INKLING_REFERENCE,
        notes=(
            "Layer schedule: 11 full-attention and 55 512-token sliding-attention layers.",
            "Includes the four intrinsic sconv histories per layer; those histories stay BF16.",
            "sconv uses three semantic history states; serving allocators can pad its four-token "
            "pages and packed head width.",
        ),
    )


def _estimate_qwen_gdn(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    """Size Qwen's full-attention layers plus fixed gated-delta state."""
    layers = _int(config, "num_hidden_layers")
    interval_value = config.get("full_attention_interval")
    interval = (
        int(interval_value)
        if isinstance(interval_value, (int, float)) and not isinstance(interval_value, bool)
        else None
    )
    schedule = _layer_schedule(
        config,
        allowed={"full_attention", "linear_attention"},
        fallback_interval=interval,
    )
    if interval is not None:
        expected = [
            "full_attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(layers)
        ]
        if schedule != expected:
            raise UnsupportedArchitecture(
                str(config.get("model_type", "unknown")),
                "layer_types differs from full_attention_interval",
                relevant_fields=("layer_types", "full_attention_interval"),
            )

    query_heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads")
    head_dim = _head_dim(config, query_heads)
    key_heads = _int(config, "linear_num_key_heads")
    value_heads = _int(config, "linear_num_value_heads")
    key_dim = _int(config, "linear_key_head_dim")
    value_dim = _int(config, "linear_value_head_dim")
    conv_kernel = _int(config, "linear_conv_kernel_dim")
    full_layers = schedule.count("full_attention")
    linear_layers = schedule.count("linear_attention")
    conv_width = 2 * key_heads * key_dim + value_heads * value_dim

    components: list[CacheComponent] = []
    if full_layers:
        components.append(
            CacheComponent(
                name="full-kv",
                bytes=2 * full_layers * context_tokens * kv_heads * head_dim * kv_bytes,
                detail=(
                    f"K+V × {full_layers} full layers × {context_tokens} tokens × "
                    f"{kv_heads} KV heads × {head_dim} head dim × {kv_bytes:g} bytes"
                ),
                tp_parallel_units=kv_heads,
            )
        )
    components.extend(
        (
            CacheComponent(
                name="gdn-conv-history",
                bytes=linear_layers * conv_width * (conv_kernel - 1) * kv_bytes,
                detail=(
                    f"{linear_layers} GDN layers × {conv_width} QKV channels × "
                    f"{conv_kernel - 1} prior states × {kv_bytes:g} bytes"
                ),
                tp_parallel_units=math.gcd(key_heads, value_heads),
            ),
            CacheComponent(
                name="gdn-recurrent-state",
                bytes=linear_layers * value_heads * value_dim * key_dim * 4,
                detail=(
                    f"{linear_layers} GDN layers × {value_heads} value heads × "
                    f"{value_dim} value dim × {key_dim} key dim × 4 FP32 bytes"
                ),
                tp_parallel_units=value_heads,
            ),
        )
    )
    return CacheEstimate(
        architecture="qwen-gated-delta-hybrid",
        context_tokens=context_tokens,
        components=tuple(components),
        kv_parallel_heads=math.gcd(kv_heads, key_heads, value_heads),
        query_heads=query_heads,
        confidence="reference-checked",
        reference=QWEN_GDN_REFERENCE,
        notes=(
            f"Layer schedule: {full_layers} full-attention and {linear_layers} gated-delta layers.",
            "Gated-delta recurrent matrices stay FP32; KV dtype only changes attention and "
            "short-convolution state.",
            "Counts one recurrent state per active sequence; prefix-cache checkpoints can "
            "multiply recurrent state allocation in an engine-specific way.",
        ),
    )


def _estimate_nemotron_h(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    """Size Nemotron-H's attention and Mamba-2 persistent states."""
    layers = _int(config, "num_hidden_layers")
    raw_pattern = config.get("hybrid_override_pattern")
    if not isinstance(raw_pattern, str) or len(raw_pattern) != layers:
        raise UnsupportedArchitecture(
            "nemotron_h",
            "hybrid_override_pattern length differs from num_hidden_layers",
            relevant_fields=("hybrid_override_pattern", "num_hidden_layers"),
        )
    unknown = sorted(set(raw_pattern) - {"M", "*", "E", "-"})
    if unknown:
        raise UnsupportedArchitecture(
            "nemotron_h",
            f"unverified hybrid layer markers {unknown!r}",
            relevant_fields=("hybrid_override_pattern",),
        )
    attention_layers = raw_pattern.count("*")
    mamba_layers = raw_pattern.count("M")
    if not attention_layers or not mamba_layers:
        raise UnsupportedArchitecture(
            "nemotron_h",
            "checked hybrid requires both attention and Mamba layers",
            relevant_fields=("hybrid_override_pattern",),
        )

    query_heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads")
    head_dim = _head_dim(config, query_heads)
    mamba_heads = _int(config, "mamba_num_heads")
    mamba_head_dim = _int(config, "mamba_head_dim")
    state_size = _int(config, "ssm_state_size")
    groups = _int(config, "n_groups")
    conv_kernel = _int(config, "conv_kernel")
    intermediate = mamba_heads * mamba_head_dim
    conv_width = intermediate + 2 * groups * state_size

    components = (
        CacheComponent(
            name="attention-kv",
            bytes=2 * attention_layers * context_tokens * kv_heads * head_dim * kv_bytes,
            detail=(
                f"K+V × {attention_layers} attention layers × {context_tokens} tokens × "
                f"{kv_heads} KV heads × {head_dim} head dim × {kv_bytes:g} bytes"
            ),
            tp_parallel_units=kv_heads,
        ),
        CacheComponent(
            name="mamba-conv-history",
            bytes=mamba_layers * conv_width * (conv_kernel - 1) * kv_bytes,
            detail=(
                f"{mamba_layers} Mamba-2 layers × {conv_width} channels × "
                f"{conv_kernel - 1} prior states × {kv_bytes:g} bytes"
            ),
            # Group state is replicated beyond n_groups. Dividing the whole
            # component only up to this point is conservative for larger TP.
            tp_parallel_units=groups,
        ),
        CacheComponent(
            name="mamba-temporal-state",
            bytes=mamba_layers * mamba_heads * mamba_head_dim * state_size * 4,
            detail=(
                f"{mamba_layers} Mamba-2 layers × {mamba_heads} heads × "
                f"{mamba_head_dim} head dim × {state_size} state size × 4 FP32 bytes"
            ),
            tp_parallel_units=mamba_heads,
        ),
    )
    return CacheEstimate(
        architecture="nemotron-h-mamba2-attention",
        context_tokens=context_tokens,
        components=components,
        kv_parallel_heads=math.gcd(kv_heads, groups),
        query_heads=query_heads,
        confidence="reference-checked",
        reference=NEMOTRON_H_REFERENCE,
        notes=(
            f"Layer schedule: {attention_layers} attention, {mamba_layers} Mamba-2, and "
            f"{layers - attention_layers - mamba_layers} expert/MLP-only layers.",
            "Mamba temporal state follows the checkpoint's FP32 serving setting.",
            "Counts one recurrent state per active sequence; vLLM Mamba prefix caching stores "
            "additional block checkpoints and must be measured separately.",
        ),
    )


def _estimate_gemma4(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
) -> CacheEstimate:
    """Size Gemma 4's distinct local and global KV geometries."""
    layers = _int(config, "num_hidden_layers")
    schedule = _layer_schedule(
        config,
        allowed={"full_attention", "sliding_attention"},
    )
    shared_layers = _int(config, "num_kv_shared_layers", minimum=0)
    if shared_layers > layers:
        raise UnsupportedArchitecture(
            "gemma4_text",
            "num_kv_shared_layers exceeds num_hidden_layers",
            relevant_fields=("num_kv_shared_layers", "num_hidden_layers"),
        )
    cached_schedule = schedule[: layers - shared_layers]
    full_layers = cached_schedule.count("full_attention")
    sliding_layers = cached_schedule.count("sliding_attention")
    query_heads = _int(config, "num_attention_heads")
    local_kv_heads = _int(config, "num_key_value_heads")
    local_head_dim = _int(config, "head_dim")
    global_kv_heads = int(config.get("num_global_key_value_heads", local_kv_heads))
    global_head_dim = int(config.get("global_head_dim", local_head_dim))
    if global_kv_heads < 1 or global_head_dim < 1:
        raise UnsupportedArchitecture(
            "gemma4_text",
            "invalid global KV geometry",
            relevant_fields=("num_global_key_value_heads", "global_head_dim"),
        )
    window = _int(config, "sliding_window")
    local_entries = min(context_tokens, window)
    components = (
        CacheComponent(
            name="global-kv",
            bytes=2 * full_layers * context_tokens * global_kv_heads * global_head_dim * kv_bytes,
            detail=(
                f"K+V × {full_layers} full layers × {context_tokens} tokens × "
                f"{global_kv_heads} KV heads × {global_head_dim} head dim × {kv_bytes:g} bytes"
            ),
            tp_parallel_units=global_kv_heads,
        ),
        CacheComponent(
            name="sliding-kv",
            bytes=(2 * sliding_layers * local_entries * local_kv_heads * local_head_dim * kv_bytes),
            detail=(
                f"K+V × {sliding_layers} sliding layers × {local_entries} tokens × "
                f"{local_kv_heads} KV heads × {local_head_dim} head dim × {kv_bytes:g} bytes"
            ),
            tp_parallel_units=local_kv_heads,
        ),
    )
    return CacheEstimate(
        architecture="gemma4-global-sliding-hybrid",
        context_tokens=context_tokens,
        components=components,
        kv_parallel_heads=math.gcd(global_kv_heads, local_kv_heads),
        query_heads=query_heads,
        confidence="reference-checked",
        reference=GEMMA4_REFERENCE,
        notes=(
            (
                f"Layer schedule: {full_layers} cached global and "
                f"{sliding_layers} cached local layers."
            ),
            "Full-attention K=V checkpoints are still stored in the serving engine's separate "
            "K and V cache slots.",
            f"Excluded {shared_layers} tail layer(s) that reuse earlier KV state."
            if shared_layers
            else "The checkpoint declares no cross-layer KV sharing.",
        ),
    )


def _estimate_minimax_m3(
    config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float,
    index_bytes: float,
) -> CacheEstimate:
    """Size MiniMax M3 GQA state plus its sparse index-key side cache."""
    layers = _int(config, "num_hidden_layers")
    query_heads = _int(config, "num_attention_heads")
    kv_heads = _int(config, "num_key_value_heads")
    head_dim = _head_dim(config, query_heads)
    sparse = config.get("sparse_attention_config")
    if not isinstance(sparse, Mapping) or sparse.get("use_sparse_attention") is not True:
        raise UnsupportedArchitecture(
            "minimax_m3",
            "checked M3 adapter requires sparse_attention_config",
            relevant_fields=("sparse_attention_config",),
        )
    raw_frequency = sparse.get("sparse_attention_freq")
    if not isinstance(raw_frequency, Sequence) or isinstance(raw_frequency, (str, bytes)):
        raise UnsupportedArchitecture(
            "minimax_m3",
            "sparse_attention_freq is required",
            relevant_fields=("sparse_attention_config.sparse_attention_freq",),
        )
    if len(raw_frequency) != layers or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw_frequency
    ):
        raise UnsupportedArchitecture(
            "minimax_m3",
            "sparse_attention_freq must have one numeric entry per layer",
            relevant_fields=("sparse_attention_config.sparse_attention_freq",),
        )
    sparse_layers = sum(value != 0 for value in raw_frequency)
    index_heads = sparse.get("sparse_num_index_heads")
    index_dim = sparse.get("sparse_index_dim")
    if (
        isinstance(index_heads, bool)
        or not isinstance(index_heads, (int, float))
        or index_heads < 1
        or isinstance(index_dim, bool)
        or not isinstance(index_dim, (int, float))
        or index_dim < 1
    ):
        raise UnsupportedArchitecture(
            "minimax_m3",
            "invalid sparse index geometry",
            relevant_fields=(
                "sparse_attention_config.sparse_num_index_heads",
                "sparse_attention_config.sparse_index_dim",
            ),
        )
    index_heads = int(index_heads)
    index_dim = int(index_dim)
    return CacheEstimate(
        architecture="minimax-m3-sparse-indexed-gqa",
        context_tokens=context_tokens,
        components=(
            CacheComponent(
                name="attention-kv",
                bytes=2 * layers * context_tokens * kv_heads * head_dim * kv_bytes,
                detail=(
                    f"K+V × {layers} layers × {context_tokens} tokens × {kv_heads} KV heads × "
                    f"{head_dim} head dim × {kv_bytes:g} bytes"
                ),
                tp_parallel_units=kv_heads,
            ),
            CacheComponent(
                name="sparse-index-keys",
                bytes=sparse_layers * context_tokens * index_heads * index_dim * index_bytes,
                detail=(
                    f"{sparse_layers} sparse layers × {context_tokens} tokens × "
                    f"{index_heads} index heads × {index_dim} dim × {index_bytes:g} bytes"
                ),
                tp_parallel_units=index_heads,
            ),
        ),
        kv_parallel_heads=math.gcd(kv_heads, index_heads),
        query_heads=query_heads,
        confidence="reference-checked",
        reference=MINIMAX_M3_REFERENCE,
        notes=(
            f"Includes one index-key cache for each of {sparse_layers} sparse-attention layers.",
            "Sparse top-k attention does not evict the full K/V history; all tokens remain cached.",
        ),
    )


def estimate_cache(
    raw_config: Mapping[str, Any],
    *,
    context_tokens: int,
    kv_bytes: float = 2.0,
    index_bytes: float | None = None,
) -> CacheEstimate:
    """Estimate logical per-sequence cache state from a Hugging Face config.

    Unknown stateful architectures fail closed instead of falling back to the
    standard K/V formula.
    """
    if context_tokens < 1:
        raise ValueError("context_tokens must be positive")
    if kv_bytes <= 0:
        raise ValueError("kv_bytes must be positive")
    if index_bytes is None:
        index_bytes = kv_bytes
    if index_bytes <= 0:
        raise ValueError("index_bytes must be positive")

    config = _text_config(raw_config)
    model_type = str(config.get("model_type", raw_config.get("model_type", "unknown"))).lower()
    architecture_names = tuple(name.lower() for name in _architecture_names(config))
    if not architecture_names:
        architecture_names = tuple(name.lower() for name in _architecture_names(raw_config))

    is_v32 = model_type in {"deepseek_v32", "deepseek_v3_2"} or any(
        "deepseekv32" in name or "deepseekv3_2" in name for name in architecture_names
    )
    if is_v32:
        return _estimate_deepseek_mla(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            index_bytes=index_bytes,
            with_indexer=True,
        )

    if model_type == "deepseek_v4" or any("deepseekv4" in name for name in architecture_names):
        return _estimate_deepseek_v4(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            index_bytes=index_bytes,
        )

    is_deepseek_mla = model_type in {"deepseek_v2", "deepseek_v3"} or any(
        name.startswith(("deepseekv2", "deepseekv3")) for name in architecture_names
    )
    if is_deepseek_mla and config.get("kv_lora_rank") is not None:
        return _estimate_deepseek_mla(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            index_bytes=index_bytes,
            with_indexer=False,
        )

    if model_type == "gpt_oss" or any("gptoss" in name for name in architecture_names):
        return _estimate_gpt_oss(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
        )

    is_inkling = model_type == "inkling_mm_model" or any(
        "inkling" in name for name in architecture_names
    )
    if is_inkling:
        return _estimate_inkling(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
        )

    is_qwen_gdn = model_type in {
        "qwen3_next",
        "qwen3_5_text",
        "qwen3_5_moe_text",
    } or any("qwen3next" in name or "qwen3_5" in name for name in architecture_names)
    if is_qwen_gdn:
        return _estimate_qwen_gdn(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
        )

    if model_type == "nemotron_h" or any("nemotronh" in name for name in architecture_names):
        return _estimate_nemotron_h(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
        )

    if model_type == "gemma4_text" or any("gemma4" in name for name in architecture_names):
        return _estimate_gemma4(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
        )

    is_minimax_m3 = model_type in {"minimax_m3", "minimax_m3_sparse"} or any(
        "minimaxm3" in name for name in architecture_names
    )
    if is_minimax_m3:
        return _estimate_minimax_m3(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            index_bytes=index_bytes,
        )

    if model_type == "glm_moe_dsa" or any("glmmoedsa" in name for name in architecture_names):
        estimate = _estimate_deepseek_mla(
            config,
            context_tokens=context_tokens,
            kv_bytes=kv_bytes,
            index_bytes=index_bytes,
            with_indexer=True,
        )
        return replace(
            estimate,
            architecture="glm-dsa-mla",
            confidence="architecture-formula",
            notes=(
                *estimate.notes,
                "GLM's DSA config exposes the same latent-KV, RoPE, and index-key widths; "
                "a live engine load remains the stronger verification level.",
            ),
        )

    unsupported_markers = tuple(
        key
        for key in (
            "linear_attn_config",
            "linear_layer_indices",
            "mamba_d_state",
            "mamba_d_conv",
            "ssm_cfg",
            "attention_chunk_size",
        )
        if config.get(key) is not None
    )
    if unsupported_markers:
        raise UnsupportedArchitecture(
            model_type,
            "hybrid/recurrent cache state is not yet modeled",
            relevant_fields=unsupported_markers,
        )
    if config.get("compress_ratios") is not None:
        raise UnsupportedArchitecture(
            model_type,
            "compressed attention requires an architecture-specific adapter",
            relevant_fields=("compress_ratios",),
        )

    auto_map = config.get("auto_map", raw_config.get("auto_map"))
    if isinstance(auto_map, Mapping) and auto_map.get("AutoModelForCausalLM") is not None:
        raise UnsupportedArchitecture(
            model_type,
            "custom remote model code can change cache semantics",
            relevant_fields=("auto_map",),
        )
    if model_type not in STANDARD_ATTENTION_MODEL_TYPES:
        raise UnsupportedArchitecture(
            model_type,
            "model type is not in the checked standard-attention set",
            relevant_fields=("model_type", "architectures"),
        )

    return _estimate_standard(config, context_tokens=context_tokens, kv_bytes=kv_bytes)
