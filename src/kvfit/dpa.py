"""Qualified SGLang DPA/EP checkpoint placement, separate from runtime memory."""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import quote

from kvfit.hf import (
    HF_BASE,
    MAX_CONFIG_BYTES,
    MAX_REPOSITORY_METADATA_BYTES,
    _fetch_json,
    _read_safetensors_header,
)
from kvfit.models import ModelMetadata

REFERENCE = (
    "https://github.com/sgl-project/sglang/blob/20a491d1d311553bbab3f22e19bbafb86ef3c0cc/"
    "python/sglang/srt/layers/dp_attention.py"
)
EXPERT_TENSOR = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.(weight|weight_scale_inv)"
)
TENSOR_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "I64": 8,
    "I32": 4,
    "U8": 1,
    "I8": 1,
}


def _integer(value, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"DPA {name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DpaWeights:
    routed_expert_bytes: int
    replicated_bytes: int
    expert_count: int
    source: str

    def __post_init__(self):
        _integer(self.routed_expert_bytes, "routed expert bytes")
        _integer(self.replicated_bytes, "replicated bytes", minimum=0)
        _integer(self.expert_count, "expert count")

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "reference": REFERENCE,
            "qualification": "checkpoint-placement-envelope",
            "assumptions": [
                "Routed experts distributed evenly across EP ranks, no redundant experts.",
                "All other checkpoint tensors counted fully replicated on every rank.",
                "Runtime weight transformations, communication and scratch are unmeasured.",
            ],
        }


def qualify_dpa_config(config: dict[str, Any]) -> tuple[int, int, int, set[int]]:
    if config.get("model_type") != "glm_moe_dsa":
        raise ValueError("SGLang DPA weight placement is only qualified for GLM DSA checkpoints")
    count = _integer(config.get("n_routed_experts"), "expert count")
    hidden = _integer(config.get("hidden_size"), "hidden size")
    width = _integer(config.get("moe_intermediate_size"), "expert intermediate size")
    layers = _integer(config.get("num_hidden_layers"), "layer count")
    draft = _integer(config.get("num_nextn_predict_layers", 0), "draft layers", minimum=0)
    dense = _integer(config.get("first_k_dense_replace"), "dense prefix", minimum=0)
    if dense >= layers or draft > 1 or count > 65536 or layers > 4096:
        raise ValueError("unqualified DPA expert/layer schedule")
    if config.get("moe_layer_freq", 1) != 1:
        raise ValueError("unqualified DPA MoE layer frequency")
    expected = ["dense"] * dense + ["sparse"] * (layers - dense)
    if "mlp_layer_types" in config and config["mlp_layer_types"] != expected:
        raise ValueError("DPA layer schedule conflicts with dense prefix")
    quant = config.get("quantization_config") or {}
    if not isinstance(quant, dict) or (
        quant
        and (
            quant.get("quant_method") != "fp8"
            or quant.get("fmt") != "e4m3"
            or quant.get("weight_block_size") != [128, 128]
        )
    ):
        raise ValueError("DPA supports native BF16 or block-128 FP8 E4M3 checkpoints only")
    return count, hidden, width, set(range(dense, layers + draft))


def _tensor_size(tensor: Any) -> int:
    if not isinstance(tensor, dict):
        raise ValueError("invalid DPA tensor metadata")
    shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
    dtype = tensor.get("dtype")
    if not isinstance(dtype, str) or dtype not in TENSOR_BYTES:
        raise ValueError("unqualified tensor dtype in DPA checkpoint")
    if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
        raise ValueError("invalid DPA tensor shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(n) is int and n >= 0 for n in offsets)
    ):
        raise ValueError("invalid DPA tensor offsets")
    size = math.prod(shape) * TENSOR_BYTES[dtype]
    if offsets[1] - offsets[0] != size:
        raise ValueError("DPA tensor shape/dtype does not match its byte span")
    return size


def classify_dpa_weights(
    config: dict[str, Any],
    tensors: dict[str, Any],
    *,
    source: str,
) -> DpaWeights:
    count, hidden, width, layers = qualify_dpa_config(config)
    routed, replicated = 0, 0
    owned = {}
    per_expert = defaultdict(int)
    quantized = bool(config.get("quantization_config"))
    for name, tensor in tensors.items():
        size = _tensor_size(tensor)
        if ".experts." not in name:
            replicated += size
            continue
        match = EXPERT_TENSOR.fullmatch(name)
        if not match:
            raise ValueError(f"unqualified routed-expert tensor: {name}")
        layer, expert = map(int, match.group(1, 2))
        projection, kind = match.group(3, 4)
        if layer not in layers or not 0 <= expert < count:
            raise ValueError("DPA expert tensor conflicts with checkpoint layer/expert count")
        shape = [hidden, width] if projection == "down" else [width, hidden]
        if kind == "weight_scale_inv":
            shape = [(n + 127) // 128 for n in shape]
            expected_dtype = "F32"
        else:
            expected_dtype = "F8_E4M3" if quantized else "BF16"
        if tensor["shape"] != shape or tensor["dtype"] != expected_dtype:
            raise ValueError("unqualified expert tensor shape, dtype or mixed quantization")
        if kind == "weight_scale_inv" and not quantized:
            raise ValueError("unexpected expert quantization scale")
        owned[(layer, expert, projection, kind)] = size
        per_expert[(layer, expert)] += size
        routed += size
    kinds = ["weight", "weight_scale_inv"] if quantized else ["weight"]
    if len(owned) != len(layers) * count * 3 * len(kinds):
        raise ValueError("incomplete routed-expert tensor coverage")
    if len(set(per_expert.values())) != 1:
        raise ValueError("unequal routed-expert payloads are not qualified")
    return DpaWeights(routed, replicated, count, source)


def fetch_dpa_weights(metadata: ModelMetadata, *, timeout: float = 20) -> DpaWeights:
    """Resolve all weight ownership from pinned headers; no tensor payload reads."""
    qualify_dpa_config(metadata.config)
    if metadata.selected_artifact or metadata.quantization_config:
        raise ValueError(
            "DPA requires a native complete root checkpoint, not an artifact/quant plan"
        )
    if not metadata.resolved_revision or not (metadata.weight_source or "").startswith(
        "HF root safetensors"
    ):
        raise ValueError("DPA requires pinned root safetensors artifact sizes")
    base = (
        f"{HF_BASE}/{quote(metadata.repo_id, safe='/')}/resolve/"
        f"{quote(metadata.resolved_revision, safe='')}/"
    )
    token = os.environ.get("HF_TOKEN")
    pinned_config = _fetch_json(
        base + "config.json", token=token, max_bytes=MAX_CONFIG_BYTES, timeout=timeout
    )
    if pinned_config != metadata.config:
        raise ValueError("DPA config changed during resolution; rerun with the resolved revision")
    index = _fetch_json(
        base + "model.safetensors.index.json",
        token=token,
        max_bytes=MAX_REPOSITORY_METADATA_BYTES,
        timeout=timeout,
    )
    mapping = index.get("weight_map")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(
            isinstance(k, str)
            and isinstance(v, str)
            and re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", v)
            for k, v in mapping.items()
        )
    ):
        raise ValueError("DPA requires a complete root safetensors index")
    files = sorted(set(mapping.values()))
    if len(files) > 2048:
        raise ValueError("DPA checkpoint exceeds the header file-count limit")

    def read(name):
        header, header_bytes, file_bytes = _read_safetensors_header(
            base + name, token=token, timeout=timeout
        )
        return name, header, header_bytes, file_bytes

    tensors: dict[str, Any] = {}
    total_artifacts, total_headers, total_payload = 0, 0, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        # Bound completed-but-unconsumed headers as well as concurrent requests.
        for batch_start in range(0, len(files), 8):
            for name, header, header_bytes, file_bytes in pool.map(
                read, files[batch_start : batch_start + 8]
            ):
                total_artifacts += file_bytes
                total_headers += header_bytes
                if total_headers > 256 * 1024**2:
                    raise ValueError("DPA checkpoint exceeds aggregate header-size limit")
                ranges = []
                for key, tensor in header.items():
                    if key == "__metadata__":
                        continue
                    if key in tensors or mapping.get(key) != name:
                        raise ValueError("DPA header tensor ownership conflicts with index")
                    total_payload += _tensor_size(tensor)
                    ranges.append(tuple(tensor["data_offsets"]))
                    tensors[key] = tensor
                end = 0
                for start, stop in sorted(ranges):
                    if start != end:
                        raise ValueError("overlapping or incomplete safetensors payload spans")
                    end = stop
                if end + header_bytes != file_bytes:
                    raise ValueError("safetensors payload span does not match artifact size")
    index_metadata = index.get("metadata")
    total_size = index_metadata.get("total_size") if isinstance(index_metadata, dict) else None
    if (
        tensors.keys() != mapping.keys()
        or total_payload != total_size
        or total_artifacts != metadata.weight_bytes
    ):
        raise ValueError("DPA checkpoint headers, index and artifact totals do not reconcile")
    return classify_dpa_weights(
        metadata.config,
        tensors,
        source=(
            f"{len(files)} safetensors headers at {metadata.repo_id}@{metadata.resolved_revision}; "
            "all tensor spans, index coverage and artifact totals reconciled"
        ),
    )
