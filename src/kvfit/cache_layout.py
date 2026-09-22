"""Opt-in storage layouts, separate from architecture payload estimates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kvfit.architectures import _glm_indexer_schedule
from kvfit.models import (
    CacheComponent,
    CacheEstimate,
    StorageCacheEstimate,
    UnsupportedArchitecture,
)

SGLANG_REVISION = "20a491d1d311"
SGLANG_SOURCE = (
    f"https://github.com/sgl-project/sglang/blob/{SGLANG_REVISION}/"
    "python/sglang/srt/mem_cache/kv_cache_configurator.py"
)
CACHE_LAYOUTS = ("logical", "sglang-dsa-raw", "sglang-dsa-scaled")


def apply_cache_layout(
    cache: CacheEstimate,
    config: Mapping[str, Any],
    *,
    profile: str,
    kv_dtype: str,
    index_dtype: str | None = None,
    mtp: bool = False,
    indexer_all_layers: bool = False,
) -> CacheEstimate:
    if profile == "logical":
        if mtp or indexer_all_layers:
            raise ValueError("--mtp and --indexer-all-layers require a SGLang storage profile")
        return cache
    if profile not in CACHE_LAYOUTS:
        raise ValueError(f"unknown cache layout {profile!r}")
    if cache.architecture != "glm-dsa-mla":
        raise UnsupportedArchitecture(
            cache.architecture, "SGLang storage profile is qualified only for GLM DSA"
        )
    if kv_dtype not in {"fp8", "fp8_e4m3", "bf16"}:
        raise ValueError("SGLang DSA storage profiles support only bf16 or fp8_e4m3 KV")
    if index_dtype not in {None, "fp8", "fp8_e4m3"}:
        raise ValueError(
            "SGLang DSA index cache is FP8 + FP32 scales, not the requested index dtype"
        )
    layers = config["num_hidden_layers"]
    latent = config["kv_lora_rank"]
    rope = config["qk_rope_head_dim"]
    index_dim = config["index_head_dim"]
    if (latent, rope, index_dim) != (512, 64, 128):
        raise UnsupportedArchitecture(
            cache.architecture, "unverified SGLang DSA storage dimensions"
        )
    schedule, _ = _glm_indexer_schedule(config)
    index_layers = layers if indexer_all_layers else schedule.count("full")
    if kv_dtype == "bf16":
        mla_cell = (latent + rope) * 2
    elif profile == "sglang-dsa-scaled":
        mla_cell = latent + (latent // 128) * 4 + rope * 2
    else:
        mla_cell = latent + rope
    index_cell = index_dim + (index_dim // 128) * 4
    draft_layers = 0
    if mtp:
        draft_layers = config.get("num_nextn_predict_layers", 0)
        if isinstance(draft_layers, bool) or not isinstance(draft_layers, int) or draft_layers < 1:
            raise ValueError("--mtp requires a positive declared num_nextn_predict_layers")
    page_size = 64
    rounded = ((cache.context_tokens + page_size - 1) // page_size) * page_size
    cells = [("mla-storage", layers * mla_cell), ("index-storage", index_layers * index_cell)]
    if draft_layers:
        cells.append(("mtp-storage", draft_layers * (mla_cell + index_cell)))
    components = tuple(
        CacheComponent(
            name,
            cell * rounded,
            f"{rounded} page-rounded tokens × {cell} bytes/token; {profile}",
        )
        for name, cell in cells
    )
    fixed = tuple(
        CacheComponent(
            f"{name}-pool-padding",
            cell * page_size,
            "one CUDA padding page per pool",
        )
        for name, cell in cells
    )
    return StorageCacheEstimate(
        architecture=cache.architecture,
        context_tokens=cache.context_tokens,
        components=components,
        fixed_components=fixed,
        kv_parallel_heads=cache.kv_parallel_heads,
        query_heads=cache.query_heads,
        confidence="upstream-storage-formula",
        reference=SGLANG_SOURCE,
        notes=(
            f"Storage profile {profile}, SGLang source {SGLANG_REVISION}; not a GPU measurement.",
            f"CUDA 64-token pages; {index_layers} index buffers; {draft_layers} MTP layer(s).",
            "Raw FP8 models TRTLLM layout; scaled FP8 models non-TRTLLM CUDA DSA layout.",
            "Index elision assumes unified serving without HiCache/HiSparse; use "
            "--indexer-all-layers for non-elided pools. HiSparse/offload/CP are not modeled.",
            "Counts KV/index storage, quantization scales and pool padding. Excludes pointer "
            "tables, scratch, speculative buffers, CUDA graphs, communication "
            "and allocator overhead.",
            "Weights remain a separate checkpoint proxy; runtime packing and "
            "kernel support require validation.",
        ),
    )
