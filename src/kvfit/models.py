from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GIB = 1024**3


@dataclass(frozen=True)
class CacheComponent:
    name: str
    bytes: float
    detail: str
    tp_parallel_units: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.bytes,
            "gib": self.bytes / GIB,
            "detail": self.detail,
            "tp_parallel_units": self.tp_parallel_units,
        }


@dataclass(frozen=True)
class CacheEstimate:
    architecture: str
    context_tokens: int
    components: tuple[CacheComponent, ...]
    kv_parallel_heads: int
    query_heads: int
    confidence: str
    reference: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_bytes(self) -> float:
        return sum(component.bytes for component in self.components)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    def as_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "context_tokens": self.context_tokens,
            "bytes_per_sequence": self.total_bytes,
            "gib_per_sequence": self.total_gib,
            "kv_parallel_heads": self.kv_parallel_heads,
            "query_heads": self.query_heads,
            "confidence": self.confidence,
            "reference": self.reference,
            "components": [component.as_dict() for component in self.components],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ModelMetadata:
    repo_id: str
    requested_revision: str
    resolved_revision: str | None
    config: dict[str, Any]
    weight_bytes: int | None
    weight_source: str | None
    selected_artifact: str | None = None
    quantization: str | None = None
    quantization_source: str | None = None
    quantization_scope: str | None = None
    quantization_excluded_modules: int | None = None
    kv_cache_quantization: str | None = None
    quantization_config_path: str | None = None
    quantization_config: dict[str, Any] = field(default_factory=dict, repr=False)
    weight_dtypes: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    runtime_weight_floor_bytes: int | None = None
    runtime_weight_floor_source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "weight_bytes": self.weight_bytes,
            "weight_gib": self.weight_bytes / GIB if self.weight_bytes is not None else None,
            "weight_source": self.weight_source,
            "selected_artifact": self.selected_artifact,
            "quantization": self.quantization,
            "quantization_source": self.quantization_source,
            "quantization_scope": self.quantization_scope,
            "quantization_excluded_modules": self.quantization_excluded_modules,
            "kv_cache_quantization": self.kv_cache_quantization,
            "quantization_config_path": self.quantization_config_path,
            "weight_dtypes": dict(self.weight_dtypes),
            "warnings": list(self.warnings),
            "runtime_weight_floor_bytes": self.runtime_weight_floor_bytes,
            "runtime_weight_floor_gib": (
                self.runtime_weight_floor_bytes / GIB
                if self.runtime_weight_floor_bytes is not None
                else None
            ),
            "runtime_weight_floor_source": self.runtime_weight_floor_source,
        }


class UnsupportedArchitecture(ValueError):
    def __init__(
        self,
        model_type: str,
        reason: str,
        *,
        relevant_fields: tuple[str, ...] = (),
    ) -> None:
        self.model_type = model_type
        self.reason = reason
        self.relevant_fields = relevant_fields
        fields = f"; relevant fields: {', '.join(relevant_fields)}" if relevant_fields else ""
        super().__init__(f"unsupported architecture {model_type!r}: {reason}{fields}")
