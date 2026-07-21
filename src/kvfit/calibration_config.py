from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from kvfit.toml_config import load_toml_defaults

CalibrationEngine = Literal["vllm", "sglang"]
CalibrationMode = Literal["attach", "launch"]


@dataclass(frozen=True)
class CalibrationConfig:
    path: Path
    engine: CalibrationEngine
    mode: CalibrationMode
    base_url: str
    host: str
    port: int
    python: str
    server_command: tuple[str, ...] | None
    server_args: tuple[str, ...]
    benchmark_command: tuple[str, ...] | None
    startup_timeout: float
    request_timeout: float
    output_tokens: int
    concurrency: tuple[int, ...] | None
    max_concurrency: int
    requests_per_concurrency: int
    seed: int
    metrics_interval: float
    api_key_env: str | None
    max_ttft_ms: float | None
    max_tpot_ms: float | None
    max_e2e_ms: float | None
    stop_on_failure: bool
    keep_server: bool
    allow_cpu: bool


def _error(path: Path, field: str, message: str) -> ValueError:
    return ValueError(f"invalid TOML config {path}: {field} {message}")


def _string(path: Path, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, field, "must be a non-empty string")
    return value.strip()


def _bool(path: Path, field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _error(path, field, "must be true or false")
    return value


def _int(path: Path, field: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, field, f"must be an integer >= {minimum}")
    return value


def _number(path: Path, field: str, value: Any, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, field, "must be a number")
    result = float(value)
    if positive and result <= 0:
        raise _error(path, field, "must be greater than 0")
    return result


def _command(path: Path, field: str, value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _error(path, field, "must be a non-empty array of strings")
    if not all(isinstance(item, str) and item for item in value):
        raise _error(path, field, "must contain only non-empty strings")
    return tuple(value)


def _python(path: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() or "/" not in value:
        return str(candidate)
    # Preserve a virtualenv's bin/python symlink: resolving it would lose the
    # environment whose vLLM/SGLang installation is being calibrated.
    return os.path.abspath(path.parent / candidate)


def _engine_from_parent(path: Path, parent: Mapping[str, Any]) -> str | None:
    raw = parent.get("check")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return None
    expanded: list[str] = []
    for value in values:
        if value == "all":
            expanded.extend(("vllm", "sglang"))
        elif value in {"vllm", "sglang"}:
            expanded.append(value)
    unique = set(expanded)
    if len(unique) == 1:
        return unique.pop()
    if unique:
        raise _error(
            path,
            "calibration.engine",
            "is required when engine.check names more than one engine",
        )
    return None


def load_calibration_config(value: str | Path) -> CalibrationConfig:
    # Reuse the deployment parser first. This validates all ordinary planning
    # fields and every allowed calibration key before we interpret the table.
    path, defaults = load_toml_defaults(value)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_calibration = data.get("calibration")
    if not isinstance(raw_calibration, Mapping):
        raise _error(path, "calibration", "must be a table")
    parent_engine = data.get("engine", {})
    if not isinstance(parent_engine, Mapping):  # already checked by deployment parser
        parent_engine = {}

    raw_engine = raw_calibration.get("engine")
    engine = (
        _string(path, "calibration.engine", raw_engine).lower()
        if raw_engine is not None
        else _engine_from_parent(path, parent_engine)
    )
    if engine not in {"vllm", "sglang"}:
        raise _error(path, "calibration.engine", "must be vllm or sglang")

    raw_mode = raw_calibration.get("mode", "attach")
    mode = _string(path, "calibration.mode", raw_mode).lower()
    if mode not in {"attach", "launch"}:
        raise _error(path, "calibration.mode", "must be attach or launch")

    host = _string(path, "calibration.host", raw_calibration.get("host", "127.0.0.1"))
    default_port = 8000 if engine == "vllm" else 30000
    port = _int(path, "calibration.port", raw_calibration.get("port", default_port))
    if port > 65535:
        raise _error(path, "calibration.port", "must be at most 65535")
    base_url = _string(
        path,
        "calibration.base_url",
        raw_calibration.get("base_url", f"http://{host}:{port}"),
    ).rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise _error(path, "calibration.base_url", "must start with http:// or https://")

    raw_python = raw_calibration.get("python", defaults.get("engine_python", sys.executable))
    python = _python(path, _string(path, "calibration.python", raw_python))
    server_command = (
        _command(path, "calibration.server_command", raw_calibration["server_command"])
        if "server_command" in raw_calibration
        else None
    )
    server_args = (
        _command(path, "calibration.server_args", raw_calibration["server_args"])
        if "server_args" in raw_calibration
        else ()
    )
    benchmark_command = (
        _command(path, "calibration.benchmark_command", raw_calibration["benchmark_command"])
        if "benchmark_command" in raw_calibration
        else None
    )

    raw_concurrency = raw_calibration.get("concurrency")
    concurrency: tuple[int, ...] | None = None
    if raw_concurrency is not None:
        if not isinstance(raw_concurrency, list) or not raw_concurrency:
            raise _error(
                path,
                "calibration.concurrency",
                "must be a non-empty array of positive integers",
            )
        values = tuple(
            _int(path, f"calibration.concurrency[{index}]", item)
            for index, item in enumerate(raw_concurrency)
        )
        if len(set(values)) != len(values):
            raise _error(path, "calibration.concurrency", "must not contain duplicates")
        concurrency = tuple(sorted(values))

    def optional_slo(name: str) -> float | None:
        value = raw_calibration.get(name)
        return None if value is None else _number(path, f"calibration.{name}", value)

    api_key_env = raw_calibration.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _string(path, "calibration.api_key_env", api_key_env)

    config = CalibrationConfig(
        path=path,
        engine=engine,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        base_url=base_url,
        host=host,
        port=port,
        python=python,
        server_command=server_command,
        server_args=server_args,
        benchmark_command=benchmark_command,
        startup_timeout=_number(
            path,
            "calibration.startup_timeout",
            raw_calibration.get("startup_timeout", 1800),
        ),
        request_timeout=_number(
            path,
            "calibration.request_timeout",
            raw_calibration.get("request_timeout", 7200),
        ),
        output_tokens=_int(
            path,
            "calibration.output_tokens",
            raw_calibration.get("output_tokens", 1),
        ),
        concurrency=concurrency,
        max_concurrency=_int(
            path,
            "calibration.max_concurrency",
            raw_calibration.get("max_concurrency", 32),
        ),
        requests_per_concurrency=_int(
            path,
            "calibration.requests_per_concurrency",
            raw_calibration.get("requests_per_concurrency", 1),
        ),
        seed=_int(path, "calibration.seed", raw_calibration.get("seed", 0), minimum=0),
        metrics_interval=_number(
            path,
            "calibration.metrics_interval",
            raw_calibration.get("metrics_interval", 1.0),
        ),
        api_key_env=api_key_env,
        max_ttft_ms=optional_slo("max_ttft_ms"),
        max_tpot_ms=optional_slo("max_tpot_ms"),
        max_e2e_ms=optional_slo("max_e2e_ms"),
        stop_on_failure=_bool(
            path,
            "calibration.stop_on_failure",
            raw_calibration.get("stop_on_failure", True),
        ),
        keep_server=_bool(
            path,
            "calibration.keep_server",
            raw_calibration.get("keep_server", False),
        ),
        allow_cpu=_bool(
            path,
            "calibration.allow_cpu",
            raw_calibration.get("allow_cpu", False),
        ),
    )
    if config.concurrency and max(config.concurrency) > config.max_concurrency:
        raise _error(
            path,
            "calibration.concurrency",
            "cannot exceed calibration.max_concurrency",
        )
    return config
