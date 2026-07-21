from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from kvfit.architectures import estimate_cache
from kvfit.cli import DTYPE_BYTES, resolve_cache_dtype
from kvfit.engines import EngineName, ProbeMode, check_engine
from kvfit.hardware import Hardware, parse_hardware
from kvfit.hf import HuggingFaceError, fetch_model_metadata
from kvfit.models import GIB, ModelMetadata, UnsupportedArchitecture
from kvfit.oracle import check_cache_estimate
from kvfit.planner import TopologyEstimate, plan_topologies, summarize_concurrency
from kvfit.systems import System, parse_system
from kvfit.toml_config import DTYPES, parse_token_count


@dataclass(frozen=True)
class ModelCase:
    id: str
    family: str
    repo: str
    revision: str | None
    contexts: tuple[int, ...]
    kv_dtypes: tuple[str, ...]
    index_dtype: str | None
    weight_gib: float | None
    expected_architecture: str | None
    required: bool


@dataclass(frozen=True)
class HardwareCase:
    hardware: Hardware
    gpu_counts: tuple[int, ...]
    required: bool


@dataclass(frozen=True)
class SystemCase:
    system: System
    hardware: Hardware
    node_counts: tuple[int, ...]
    required: bool


@dataclass(frozen=True)
class DeploymentTarget:
    id: str
    hardware: Hardware
    configurations: tuple[tuple[int, int | None], ...]
    required: bool
    system: System | None = None


@dataclass(frozen=True)
class EngineCase:
    name: EngineName
    python: str
    probe: ProbeMode
    timeout: float
    required: bool


@dataclass(frozen=True)
class AuditConfig:
    path: Path
    models: tuple[ModelCase, ...]
    hardware: tuple[HardwareCase, ...]
    systems: tuple[SystemCase, ...]
    engines: tuple[EngineCase, ...]
    utilization: float
    active_sequences_per_user: int
    min_model_coverage: float
    min_hardware_coverage: float
    min_system_coverage: float
    timeout: float
    allow_context_overflow: bool


ROOT_KEYS = {
    "version",
    "contexts",
    "kv_dtypes",
    "index_dtype",
    "gpu_counts",
    "node_counts",
    "utilization",
    "active_sequences_per_user",
    "min_model_coverage",
    "min_hardware_coverage",
    "min_system_coverage",
    "timeout",
    "allow_context_overflow",
    "models",
    "hardware",
    "systems",
    "engines",
}
MODEL_KEYS = {
    "id",
    "family",
    "repo",
    "revision",
    "contexts",
    "kv_dtypes",
    "index_dtype",
    "weight_gib",
    "expected_architecture",
    "required",
}
HARDWARE_KEYS = {"preset", "gpu_counts", "required"}
SYSTEM_KEYS = {"preset", "node_counts", "required"}
ENGINE_KEYS = {"name", "python", "probe", "timeout", "required"}


def _error(path: Path, field: str, message: str) -> ValueError:
    return ValueError(f"invalid audit TOML {path}: {field} {message}")


def _string(path: Path, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, field, "must be a non-empty string")
    return value.strip()


def _bool(path: Path, field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _error(path, field, "must be true or false")
    return value


def _positive_float(path: Path, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise _error(path, field, "must be a positive number")
    return float(value)


def _positive_int(path: Path, field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error(path, field, "must be a positive integer")
    return value


def _coverage(path: Path, field: str, value: Any) -> float:
    result = _positive_float(path, field, value)
    if result > 1:
        raise _error(path, field, "must be at most 1")
    return result


def _array(path: Path, field: str, value: Any) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise _error(path, field, "must be a non-empty array")
    return value


def _contexts(path: Path, field: str, value: Any) -> tuple[int, ...]:
    result: list[int] = []
    for item in _array(path, field, value):
        if isinstance(item, bool):
            raise _error(path, field, "entries must be positive integers or strings like '128k'")
        if isinstance(item, int) and item > 0:
            parsed = item
        elif isinstance(item, str):
            try:
                parsed = parse_token_count(item)
            except ValueError as error:
                raise _error(path, field, str(error)) from error
        else:
            raise _error(path, field, "entries must be positive integers or strings like '128k'")
        if parsed not in result:
            result.append(parsed)
    return tuple(result)


def _dtypes(path: Path, field: str, value: Any, *, allow_auto: bool) -> tuple[str, ...]:
    allowed = DTYPES if allow_auto else DTYPES - {"auto"}
    result: list[str] = []
    for item in _array(path, field, value):
        dtype = _string(path, field, item).lower()
        if dtype not in allowed:
            raise _error(path, field, f"entries must be one of {', '.join(sorted(allowed))}")
        if dtype not in result:
            result.append(dtype)
    return tuple(result)


def _gpu_counts(path: Path, field: str, value: Any) -> tuple[int, ...]:
    result: list[int] = []
    for item in _array(path, field, value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise _error(path, field, "entries must be positive integers")
        if item not in result:
            result.append(item)
    return tuple(result)


def _tables(
    path: Path, field: str, value: Any, *, allow_empty: bool = False
) -> list[Mapping[str, Any]]:
    if allow_empty and value == []:
        return []
    rows = _array(path, field, value)
    if not all(isinstance(row, Mapping) for row in rows):
        raise _error(path, field, "must contain only tables")
    return cast(list[Mapping[str, Any]], rows)


def _check_keys(path: Path, field: str, row: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise _error(path, field, f"contains unknown key(s): {', '.join(unknown)}")


def load_audit_config(value: str | Path) -> AuditConfig:
    path = Path(value).expanduser().resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ValueError(f"audit TOML was not found: {path}") from error
    except (PermissionError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"could not read audit TOML {path}: {error}") from error
    _check_keys(path, ".", data, ROOT_KEYS)
    if data.get("version", 1) != 1:
        raise _error(path, "version", "must be 1")

    global_contexts = _contexts(path, "contexts", data.get("contexts", [128 * 1024]))
    global_dtypes = _dtypes(path, "kv_dtypes", data.get("kv_dtypes", ["auto"]), allow_auto=True)
    global_counts = _gpu_counts(path, "gpu_counts", data.get("gpu_counts", [1, 2, 4, 8]))
    global_node_counts = _gpu_counts(path, "node_counts", data.get("node_counts", [1, 2, 8]))
    global_index = data.get("index_dtype")
    if global_index is not None:
        global_index = _dtypes(path, "index_dtype", [global_index], allow_auto=False)[0]

    models: list[ModelCase] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(_tables(path, "models", data.get("models", []))):
        field = f"models[{index}]"
        _check_keys(path, field, row, MODEL_KEYS)
        repo = _string(path, f"{field}.repo", row.get("repo"))
        model_id = _string(path, f"{field}.id", row.get("id", repo.replace("/", "-")))
        if model_id in seen_ids:
            raise _error(path, f"{field}.id", f"duplicates {model_id!r}")
        seen_ids.add(model_id)
        family = _string(path, f"{field}.family", row.get("family", model_id))
        revision = row.get("revision")
        models.append(
            ModelCase(
                id=model_id,
                family=family,
                repo=repo,
                revision=(
                    _string(path, f"{field}.revision", revision) if revision is not None else None
                ),
                contexts=(
                    _contexts(path, f"{field}.contexts", row["contexts"])
                    if "contexts" in row
                    else global_contexts
                ),
                kv_dtypes=(
                    _dtypes(
                        path,
                        f"{field}.kv_dtypes",
                        row["kv_dtypes"],
                        allow_auto=True,
                    )
                    if "kv_dtypes" in row
                    else global_dtypes
                ),
                index_dtype=(
                    _dtypes(
                        path,
                        f"{field}.index_dtype",
                        [row["index_dtype"]],
                        allow_auto=False,
                    )[0]
                    if "index_dtype" in row
                    else cast(str | None, global_index)
                ),
                weight_gib=(
                    _positive_float(path, f"{field}.weight_gib", row["weight_gib"])
                    if "weight_gib" in row
                    else None
                ),
                expected_architecture=(
                    _string(
                        path,
                        f"{field}.expected_architecture",
                        row["expected_architecture"],
                    )
                    if "expected_architecture" in row
                    else None
                ),
                required=_bool(path, f"{field}.required", row.get("required", True)),
            )
        )

    hardware_cases: list[HardwareCase] = []
    seen_hardware: set[str] = set()
    for index, row in enumerate(
        _tables(path, "hardware", data.get("hardware", []), allow_empty=True)
    ):
        field = f"hardware[{index}]"
        _check_keys(path, field, row, HARDWARE_KEYS)
        preset = _string(path, f"{field}.preset", row.get("preset"))
        hardware = parse_hardware(preset)
        if hardware.id in seen_hardware:
            raise _error(path, f"{field}.preset", f"duplicates {hardware.id!r}")
        seen_hardware.add(hardware.id)
        hardware_cases.append(
            HardwareCase(
                hardware=hardware,
                gpu_counts=(
                    _gpu_counts(path, f"{field}.gpu_counts", row["gpu_counts"])
                    if "gpu_counts" in row
                    else global_counts
                ),
                required=_bool(path, f"{field}.required", row.get("required", True)),
            )
        )

    system_cases: list[SystemCase] = []
    seen_systems: set[str] = set()
    for index, row in enumerate(
        _tables(path, "systems", data.get("systems", []), allow_empty=True)
    ):
        field = f"systems[{index}]"
        _check_keys(path, field, row, SYSTEM_KEYS)
        preset = _string(path, f"{field}.preset", row.get("preset"))
        system = parse_system(preset)
        if system.id in seen_systems:
            raise _error(path, f"{field}.preset", f"duplicates {system.id!r}")
        seen_systems.add(system.id)
        system_cases.append(
            SystemCase(
                system=system,
                hardware=parse_hardware(system.hardware_id),
                node_counts=(
                    _gpu_counts(path, f"{field}.node_counts", row["node_counts"])
                    if "node_counts" in row
                    else global_node_counts
                ),
                required=_bool(path, f"{field}.required", row.get("required", True)),
            )
        )

    engines: list[EngineCase] = []
    for index, row in enumerate(
        _tables(path, "engines", data.get("engines", []), allow_empty=True)
    ):
        field = f"engines[{index}]"
        _check_keys(path, field, row, ENGINE_KEYS)
        name = _string(path, f"{field}.name", row.get("name")).lower()
        if name not in {"vllm", "sglang"}:
            raise _error(path, f"{field}.name", "must be vllm or sglang")
        probe = _string(path, f"{field}.probe", row.get("probe", "config")).lower()
        if probe not in {"registry", "config", "load"}:
            raise _error(path, f"{field}.probe", "must be registry, config, or load")
        raw_python = _string(path, f"{field}.python", row.get("python", sys.executable))
        python_path = Path(raw_python).expanduser()
        if not python_path.is_absolute() and "/" in raw_python:
            # Preserve a venv's bin/python symlink instead of resolving it to
            # the base interpreter, which would inspect the wrong packages.
            python_path = Path(os.path.abspath(path.parent / python_path))
        engines.append(
            EngineCase(
                name=cast(EngineName, name),
                python=str(python_path),
                probe=cast(ProbeMode, probe),
                timeout=_positive_float(path, f"{field}.timeout", row.get("timeout", 90)),
                required=_bool(path, f"{field}.required", row.get("required", False)),
            )
        )

    if not models:
        raise _error(path, "models", "must contain at least one model")
    if not hardware_cases and not system_cases:
        raise _error(path, ".", "must contain at least one hardware or system preset")
    return AuditConfig(
        path=path,
        models=tuple(models),
        hardware=tuple(hardware_cases),
        systems=tuple(system_cases),
        engines=tuple(engines),
        utilization=_coverage(path, "utilization", data.get("utilization", 0.9)),
        active_sequences_per_user=_positive_int(
            path,
            "active_sequences_per_user",
            data.get("active_sequences_per_user", 1),
        ),
        min_model_coverage=_coverage(
            path, "min_model_coverage", data.get("min_model_coverage", 0.8)
        ),
        min_hardware_coverage=_coverage(
            path, "min_hardware_coverage", data.get("min_hardware_coverage", 0.8)
        ),
        min_system_coverage=_coverage(
            path,
            "min_system_coverage",
            data.get("min_system_coverage", data.get("min_hardware_coverage", 0.8)),
        ),
        timeout=_positive_float(path, "timeout", data.get("timeout", 30)),
        allow_context_overflow=_bool(
            path,
            "allow_context_overflow",
            data.get("allow_context_overflow", False),
        ),
    )


def _max_context(config: Mapping[str, Any]) -> int | None:
    source = config.get("text_config")
    source = source if isinstance(source, Mapping) else config
    value = source.get("max_position_embeddings", source.get("model_max_length"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _weight_footprint(metadata: ModelMetadata, case: ModelCase) -> tuple[int, str, list[str]]:
    warnings: list[str] = []
    if case.weight_gib is not None:
        return int(case.weight_gib * GIB), "audit TOML override", warnings
    if metadata.weight_bytes is None:
        raise ValueError("HF metadata exposes no weight artifact size; set weight_gib")
    weight_bytes = metadata.weight_bytes
    source = metadata.weight_source or "HF metadata"
    if (
        metadata.runtime_weight_floor_bytes is not None
        and metadata.runtime_weight_floor_bytes > weight_bytes
    ):
        warnings.append("published runtime floor replaced the smaller artifact-byte proxy")
        weight_bytes = metadata.runtime_weight_floor_bytes
        source = metadata.runtime_weight_floor_source or "published runtime floor"
    else:
        warnings.append("artifact bytes are a runtime weight-footprint proxy")
    return weight_bytes, source, warnings


def _best_layout(layouts: Sequence[TopologyEstimate]) -> TopologyEstimate:
    return max(
        layouts,
        key=lambda layout: (
            layout.verdict == "fits",
            layout.total_sequences,
            layout.tensor_parallel,
        ),
    )


def _matrix_invariants(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    failed_checks: set[str] = set()
    context_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    memory_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    dtype_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    system_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)

    def violate(check: str, message: str) -> None:
        failed_checks.add(check)
        violations.append(message)

    for row in rows:
        if not row.get("resolved_revision"):
            violate("resolved_live_revisions", f"{row['model_id']}: live revision was not resolved")
        oracle = row.get("oracle")
        if not isinstance(oracle, Mapping) or oracle.get("passed") is not True:
            violate(
                "independent_formula_match",
                f"{row['model_id']}: independent oracle did not pass",
            )
        system = row.get("system")
        if isinstance(system, Mapping):
            if system["systems"] * system["accelerators_per_system"] != row["gpus"]:
                violate(
                    "system_device_arithmetic",
                    f"{row['model_id']}/{row['deployment_id']}: system×accelerator arithmetic",
                )
            system_groups[
                (
                    row["model_id"],
                    row["context_tokens"],
                    row["requested_kv_dtype"],
                    row["deployment_id"],
                )
            ].append(row)
        for layout in row["layouts"]:
            if layout["tensor_parallel"] * layout["data_parallel"] != row["gpus"]:
                violate(
                    "topology_arithmetic",
                    f"{row['model_id']}/{row['deployment_id']}: TP×DP differs from GPU count",
                )
            if (
                layout["total_sequences"]
                != layout["sequences_per_replica"] * layout["data_parallel"]
            ):
                violate(
                    "topology_arithmetic",
                    f"{row['model_id']}/{row['deployment_id']}: total sequence arithmetic",
                )
            if layout["verdict"] == "fits" and layout["sequences_per_replica"] < 1:
                violate(
                    "topology_arithmetic",
                    f"{row['model_id']}/{row['deployment_id']}: fits with zero sequences",
                )
            domain = layout["scale_up_domain_accelerators"]
            if domain is not None:
                expected_cross = (
                    layout["tensor_parallel"] > domain or domain % layout["tensor_parallel"] != 0
                )
                if layout["cross_domain_tensor_parallel"] != expected_cross:
                    violate(
                        "fabric_boundary_labels",
                        f"{row['model_id']}/{row['deployment_id']}: incorrect fabric label",
                    )

        concurrency = row["concurrency"]
        sequences_per_user = concurrency["active_sequences_per_user"]
        for name in ("recommended", "best_scale_up_local", "best_any_fabric"):
            capacity = concurrency[name]
            if capacity is not None and capacity["concurrent_users"] != (
                capacity["active_sequences"] // sequences_per_user
            ):
                violate(
                    "user_arithmetic",
                    f"{row['model_id']}/{row['deployment_id']}: {name} user arithmetic",
                )

        context_groups[
            (
                row["model_id"],
                row["requested_kv_dtype"],
                row["deployment_id"],
                row["gpus"],
            )
        ].append(row)
        memory_groups[
            (
                row["model_id"],
                row["context_tokens"],
                row["requested_kv_dtype"],
                row["gpus"],
            )
        ].append(row)
        dtype_groups[
            (
                row["model_id"],
                row["context_tokens"],
                row["deployment_id"],
                row["gpus"],
            )
        ].append(row)

    for key, group in context_groups.items():
        ordered = sorted(group, key=lambda row: row["context_tokens"])
        cache_sizes = [row["cache_bytes_per_sequence"] for row in ordered]
        if any(right < left for left, right in pairwise(cache_sizes)):
            violate(
                "context_monotonicity",
                f"{key}: cache bytes decreased at a longer context",
            )
        capacities = [
            (row["concurrency"]["recommended"] or {"concurrent_users": 0})["concurrent_users"]
            for row in ordered
        ]
        if any(right > left for left, right in pairwise(capacities)):
            violate(
                "capacity_context_monotonicity",
                f"{key}: user capacity increased at a longer context",
            )

    for key, group in memory_groups.items():
        ordered = sorted(group, key=lambda row: row["hardware"]["memory_gib"])
        capacities = [row["best_layout"]["total_sequences"] for row in ordered]
        if any(right < left for left, right in pairwise(capacities)):
            violate(
                "hardware_memory_monotonicity",
                f"{key}: best capacity decreased on a larger-memory preset",
            )

    for key, group in dtype_groups.items():
        unique = {row["kv_dtype"]: row for row in group}
        ordered = sorted(unique.values(), key=lambda row: DTYPE_BYTES[row["kv_dtype"]])
        cache_sizes = [row["cache_bytes_per_sequence"] for row in ordered]
        if any(right < left for left, right in pairwise(cache_sizes)):
            violate(
                "dtype_width_monotonicity",
                f"{key}: cache bytes decreased at a wider dtype",
            )

    for key, group in system_groups.items():
        ordered = sorted(group, key=lambda row: row["system"]["systems"])
        capacities = [
            (row["concurrency"]["recommended"] or {"concurrent_users": 0})["concurrent_users"]
            for row in ordered
        ]
        if any(right < left for left, right in pairwise(capacities)):
            violate(
                "system_scale_monotonicity",
                f"{key}: user capacity decreased with more complete systems",
            )

    checks = (
        "resolved_live_revisions",
        "independent_formula_match",
        "topology_arithmetic",
        "system_device_arithmetic",
        "user_arithmetic",
        "fabric_boundary_labels",
        "context_monotonicity",
        "capacity_context_monotonicity",
        "hardware_memory_monotonicity",
        "dtype_width_monotonicity",
        "system_scale_monotonicity",
    )
    return {
        "passed": not violations,
        "checks": {check: check not in failed_checks for check in checks},
        "violations": violations,
    }


def _deployment_targets(config: AuditConfig) -> tuple[DeploymentTarget, ...]:
    hardware = [
        DeploymentTarget(
            id=case.hardware.id,
            hardware=case.hardware,
            configurations=tuple((count, None) for count in case.gpu_counts),
            required=case.required,
        )
        for case in config.hardware
    ]
    systems = [
        DeploymentTarget(
            id=case.system.id,
            hardware=case.hardware,
            configurations=tuple(
                (case.system.accelerators_per_system * nodes, nodes) for nodes in case.node_counts
            ),
            required=case.required,
            system=case.system,
        )
        for case in config.systems
    ]
    return tuple([*hardware, *systems])


def run_audit(config: AuditConfig) -> dict[str, Any]:
    targets = _deployment_targets(config)
    rows: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    successful_scenarios = 0
    family_cases: dict[str, list[bool]] = defaultdict(list)
    deployment_rows: dict[str, int] = defaultdict(int)
    deployment_expected_rows: dict[str, int] = defaultdict(int)
    oracle_failures = 0
    required_engine_failures = 0

    for case in config.models:
        case_errors: list[str] = []
        case_warnings: list[str] = []
        case_scenarios = 0
        case_successes = 0
        metadata: ModelMetadata | None = None
        try:
            metadata = fetch_model_metadata(
                case.repo,
                revision=case.revision,
                timeout=config.timeout,
            )
            weight_bytes, weight_source, weight_warnings = _weight_footprint(metadata, case)
            case_warnings.extend(metadata.warnings)
            case_warnings.extend(weight_warnings)
            max_context = _max_context(metadata.config)
        except (HuggingFaceError, ValueError) as error:
            case_errors.append(str(error))
            metadata = None
            weight_bytes = 0
            weight_source = "unavailable"
            max_context = None

        if metadata is not None:
            for context in case.contexts:
                for requested_dtype in case.kv_dtypes:
                    case_scenarios += 1
                    scenario_error: str | None = None
                    if (
                        max_context is not None
                        and context > max_context
                        and not config.allow_context_overflow
                    ):
                        scenario_error = f"context {context} exceeds declared maximum {max_context}"
                    try:
                        kv_dtype, dtype_source, dtype_warnings = resolve_cache_dtype(
                            metadata.config,
                            requested_dtype,
                            metadata.quantization_config,
                        )
                        index_dtype = case.index_dtype or kv_dtype
                        cache = estimate_cache(
                            metadata.config,
                            context_tokens=context,
                            kv_bytes=DTYPE_BYTES[kv_dtype],
                            index_bytes=DTYPE_BYTES[index_dtype],
                        )
                        oracle = check_cache_estimate(
                            metadata.config,
                            cache,
                            kv_bytes=DTYPE_BYTES[kv_dtype],
                            index_bytes=DTYPE_BYTES[index_dtype],
                        )
                        if not oracle.passed:
                            oracle_failures += 1
                            scenario_error = (
                                f"independent oracle differs by {oracle.relative_error:.3e}"
                            )
                        if (
                            case.expected_architecture is not None
                            and cache.architecture != case.expected_architecture
                        ):
                            scenario_error = (
                                f"architecture {cache.architecture!r} differs from expected "
                                f"{case.expected_architecture!r}"
                            )
                    except (UnsupportedArchitecture, ValueError, KeyError) as error:
                        scenario_error = str(error)
                        cache = None
                        oracle = None
                        kv_dtype = requested_dtype
                        index_dtype = case.index_dtype or requested_dtype
                        dtype_source = "unresolved"
                        dtype_warnings = ()

                    engine_checks = []
                    if scenario_error is None and cache is not None:
                        for engine in config.engines:
                            check = check_engine(
                                engine.name,
                                metadata,
                                python=engine.python,
                                mode=engine.probe,
                                context_tokens=context,
                                timeout=engine.timeout,
                                tensor_parallel=max(
                                    gpu_count
                                    for target in targets
                                    for gpu_count, _ in target.configurations
                                ),
                                utilization=config.utilization,
                                kv_dtype=kv_dtype,
                                index_dtype=case.index_dtype,
                            )
                            engine_checks.append({**check.as_dict(), "required": engine.required})
                            if engine.required and not check.passed:
                                required_engine_failures += 1
                                scenario_error = (
                                    f"required {engine.name} probe returned {check.overall}"
                                )

                    scenario_ok = scenario_error is None and cache is not None
                    if scenario_ok:
                        case_successes += 1
                        successful_scenarios += 1
                    else:
                        case_errors.append(
                            f"{context} tokens/{requested_dtype}: "
                            f"{scenario_error or 'unknown error'}"
                        )

                    for target in targets:
                        for gpu_count, node_count in target.configurations:
                            deployment_expected_rows[target.id] += int(scenario_ok)
                            if not scenario_ok or cache is None:
                                continue
                            layouts = plan_topologies(
                                cache,
                                weight_bytes=weight_bytes,
                                hardware=target.hardware,
                                gpus=gpu_count,
                                utilization=config.utilization,
                                scale_up_domain_accelerators=(
                                    target.system.scale_up_domain_accelerators
                                    if target.system
                                    else None
                                ),
                            )
                            best = _best_layout(layouts)
                            concurrency = summarize_concurrency(
                                layouts,
                                active_sequences_per_user=(config.active_sequences_per_user),
                            )
                            deployment_rows[target.id] += 1
                            rows.append(
                                {
                                    "model_id": case.id,
                                    "family": case.family,
                                    "repo": case.repo,
                                    "resolved_revision": metadata.resolved_revision,
                                    "weight_quantization": metadata.quantization,
                                    "weight_bytes": weight_bytes,
                                    "weight_source": weight_source,
                                    "context_tokens": context,
                                    "requested_kv_dtype": requested_dtype,
                                    "kv_dtype": kv_dtype,
                                    "kv_dtype_source": dtype_source,
                                    "index_dtype": index_dtype,
                                    "architecture": cache.architecture,
                                    "confidence": cache.confidence,
                                    "cache_bytes_per_sequence": cache.total_bytes,
                                    "oracle": oracle.as_dict() if oracle else None,
                                    "deployment_id": target.id,
                                    "hardware": target.hardware.as_dict(),
                                    "system": (
                                        {
                                            **target.system.as_dict(),
                                            "systems": node_count,
                                            "total_accelerators": gpu_count,
                                        }
                                        if target.system
                                        else None
                                    ),
                                    "gpus": gpu_count,
                                    "best_layout": best.as_dict(),
                                    "layouts": [layout.as_dict() for layout in layouts],
                                    "concurrency": concurrency,
                                    "engine_checks": engine_checks,
                                    "warnings": [*dtype_warnings],
                                }
                            )

        case_passed = case_scenarios > 0 and case_successes == case_scenarios
        family_cases[case.family].append(case_passed or not case.required)
        model_results.append(
            {
                "id": case.id,
                "family": case.family,
                "repo": case.repo,
                "required": case.required,
                "resolved_revision": metadata.resolved_revision if metadata else None,
                "quantization": metadata.quantization if metadata else None,
                "scenarios": case_scenarios,
                "successful_scenarios": case_successes,
                "passed": case_passed,
                "errors": case_errors,
                "warnings": sorted(set(case_warnings)),
            }
        )

    family_results = {family: all(results) for family, results in sorted(family_cases.items())}
    deployment_results: dict[str, bool] = {}
    for target in targets:
        expected = deployment_expected_rows[target.id]
        deployment_results[target.id] = expected > 0 and deployment_rows[target.id] == expected
    required_hardware = [target for target in config.hardware if target.required]
    required_systems = [target for target in config.systems if target.required]
    hardware_results = {
        target.hardware.id: deployment_results[target.hardware.id] for target in config.hardware
    }
    system_results = {
        target.system.id: deployment_results[target.system.id] for target in config.systems
    }

    required_families = {case.family for case in config.models if case.required}
    supported_families = sum(family_results[family] for family in required_families)
    model_coverage = supported_families / len(required_families) if required_families else 1.0
    supported_hardware = sum(hardware_results[target.hardware.id] for target in required_hardware)
    hardware_coverage = supported_hardware / len(required_hardware) if required_hardware else 1.0
    supported_systems = sum(system_results[target.system.id] for target in required_systems)
    system_coverage = supported_systems / len(required_systems) if required_systems else 1.0
    invariants = _matrix_invariants(rows)
    passed = (
        model_coverage >= config.min_model_coverage
        and hardware_coverage >= config.min_hardware_coverage
        and system_coverage >= config.min_system_coverage
        and oracle_failures == 0
        and required_engine_failures == 0
        and invariants["passed"]
    )
    return {
        "schema_version": 1,
        "input_config": str(config.path),
        "passed": passed,
        "coverage": {
            "definition": (
                "Equal-weight representative families, accelerators, and complete systems in "
                "this TOML; this is not market-share coverage. A family passes only when every "
                "required variant/context/dtype has a formula and independent-oracle match."
            ),
            "model_families": {
                "supported": supported_families,
                "total": len(required_families),
                "ratio": model_coverage,
                "minimum": config.min_model_coverage,
                "results": family_results,
            },
            "hardware_presets": {
                "supported": supported_hardware,
                "total": len(required_hardware),
                "ratio": hardware_coverage,
                "minimum": config.min_hardware_coverage,
                "results": hardware_results,
            },
            "system_presets": {
                "supported": supported_systems,
                "total": len(required_systems),
                "ratio": system_coverage,
                "minimum": config.min_system_coverage,
                "results": system_results,
            },
            "successful_model_scenarios": successful_scenarios,
            "decision_rows": len(rows),
            "oracle_failures": oracle_failures,
            "required_engine_failures": required_engine_failures,
        },
        "models": model_results,
        "invariants": invariants,
        "hardware": [
            {
                **target.hardware.as_dict(),
                "gpu_counts": list(target.gpu_counts),
                "required": target.required,
                "passed": hardware_results[target.hardware.id],
            }
            for target in config.hardware
        ],
        "systems": [
            {
                **target.system.as_dict(),
                "node_counts": list(target.node_counts),
                "required": target.required,
                "passed": system_results[target.system.id],
            }
            for target in config.systems
        ],
        "rows": rows,
        "boundaries": [
            "Capacity rows assume ideal weight sharding and report memory, not latency or SLOs.",
            "Artifact bytes are only a runtime-weight proxy unless the TOML overrides weight_gib.",
            "Recurrent models count one live state per sequence; engine prefix-cache checkpoints "
            "can require additional state memory.",
            "A registry/config engine pass is not a weight-load test; only probe='load' is a "
            "one-token runtime smoke test.",
            "Cross-domain TP is an arithmetic memory hypothesis until the selected engine and "
            "cluster fabric are load-tested.",
        ],
    }


def _human_report(report: Mapping[str, Any]) -> None:
    coverage = report["coverage"]
    model = coverage["model_families"]
    hardware = coverage["hardware_presets"]
    systems = coverage["system_presets"]
    verdict = "PASS" if report["passed"] else "FAIL"
    print(f"Enterprise matrix: {verdict}")
    print(
        f"Model families:    {model['supported']}/{model['total']} "
        f"({model['ratio']:.0%}; required {model['minimum']:.0%})"
    )
    print(
        f"Hardware presets:  {hardware['supported']}/{hardware['total']} "
        f"({hardware['ratio']:.0%}; required {hardware['minimum']:.0%})"
    )
    if systems["total"]:
        print(
            f"System presets:    {systems['supported']}/{systems['total']} "
            f"({systems['ratio']:.0%}; required {systems['minimum']:.0%})"
        )
    print(f"Decision rows:     {coverage['decision_rows']}")
    print(f"Oracle failures:   {coverage['oracle_failures']}")
    print(f"Matrix invariants: {'pass' if report['invariants']['passed'] else 'FAIL'}")
    if coverage["required_engine_failures"]:
        print(f"Engine failures:   {coverage['required_engine_failures']}")
    failed = [model for model in report["models"] if model["required"] and not model["passed"]]
    if failed:
        print("\nUnsupported or failed model variants:")
        for model_result in failed:
            print(f"  {model_result['id']}: {model_result['errors'][0]}")
    print("\nBoundary: memory-capacity decisions are not throughput or latency qualification.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kvfit-audit",
        description=(
            "Run a live, TOML-defined model × accelerator/system KV-capacity decision matrix."
        ),
    )
    parser.add_argument("config", help="enterprise matrix TOML")
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    parser.add_argument("--output", help="also write the complete JSON report to this path")
    args = parser.parse_args(argv)
    try:
        config = load_audit_config(args.config)
        report = run_audit(config)
    except (ValueError, HuggingFaceError) as error:
        print(f"kvfit-audit: {error}", file=sys.stderr)
        return 2
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        try:
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        except OSError as error:
            print(f"kvfit-audit: could not write {output_path}: {error}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _human_report(report)
    return 0 if report["passed"] else 1
