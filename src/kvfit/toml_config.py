from __future__ import annotations

import argparse
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT_KEYS = {
    "model",
    "revision",
    "context",
    "kv_dtype",
    "index_dtype",
    "weight_gib",
    "allow_context_overflow",
    "hardware",
    "system",
    "gpus",
    "nodes",
    "utilization",
    "tp",
    "active_sequences_per_user",
    "timeout",
}
ENGINE_KEYS = {"check", "python", "probe", "tp", "timeout", "require_pass"}
OUTPUT_KEYS = {"json"}
CALIBRATION_KEYS = {
    "engine",
    "mode",
    "base_url",
    "host",
    "port",
    "python",
    "server_command",
    "server_args",
    "benchmark_command",
    "startup_timeout",
    "request_timeout",
    "output_tokens",
    "concurrency",
    "max_concurrency",
    "requests_per_concurrency",
    "seed",
    "metrics_interval",
    "api_key_env",
    "max_ttft_ms",
    "max_tpot_ms",
    "max_e2e_ms",
    "stop_on_failure",
    "keep_server",
    "allow_cpu",
}
DTYPES = {
    "auto",
    "bf16",
    "fp16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
    "fp4",
    "mxfp4",
    "nvfp4",
    "int4",
}
TOKEN_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([km]?)$", re.IGNORECASE)


def _error(path: Path, field: str, message: str) -> ValueError:
    return ValueError(f"invalid TOML config {path}: {field} {message}")


def _mapping(path: Path, data: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = data.get(section, {})
    if not isinstance(value, Mapping):
        raise _error(path, section, "must be a table")
    return value


def _check_keys(path: Path, data: Mapping[str, Any]) -> None:
    allowed_root = ROOT_KEYS | {"engine", "output", "calibration"}
    unknown_root = sorted(set(data) - allowed_root)
    if unknown_root:
        raise _error(path, ".", f"contains unknown key(s): {', '.join(unknown_root)}")
    for section, allowed in (
        ("engine", ENGINE_KEYS),
        ("output", OUTPUT_KEYS),
        ("calibration", CALIBRATION_KEYS),
    ):
        values = _mapping(path, data, section)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise _error(
                path,
                f"{section}.",
                f"contains unknown key(s): {', '.join(unknown)}",
            )


def _string(path: Path, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(path, field, "must be a non-empty string")
    return value


def _bool(path: Path, field: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise _error(path, field, "must be true or false")
    return value


def _int(path: Path, field: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, field, f"must be an integer >= {minimum}")
    return value


def _float(path: Path, field: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, field, "must be a number")
    return float(value)


def _positive_float(path: Path, field: str, value: Any) -> float:
    result = _float(path, field, value)
    if result <= 0:
        raise _error(path, field, "must be greater than 0")
    return result


def parse_token_count(value: str) -> int:
    match = TOKEN_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("expected tokens like 32768, 128k, or 1m")
    multiplier = {"": 1, "k": 1024, "m": 1024**2}[match.group(2).lower()]
    result = int(float(match.group(1)) * multiplier)
    if result < 1:
        raise ValueError("context must be positive")
    return result


def _context(path: Path, value: Any) -> int:
    if isinstance(value, bool):
        raise _error(path, "context", "must be a positive integer or a string like '128k'")
    if isinstance(value, int):
        return _int(path, "context", value)
    if isinstance(value, str):
        try:
            return parse_token_count(value)
        except ValueError:
            pass
    raise _error(path, "context", "must be a positive integer or a string like '128k'")


def _tp(path: Path, field: str, value: Any) -> int | None:
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return _int(path, field, value)


def _dtype(path: Path, field: str, value: Any, *, allow_auto: bool) -> str:
    dtype = _string(path, field, value).lower()
    choices = DTYPES if allow_auto else DTYPES - {"auto"}
    if dtype not in choices:
        raise _error(path, field, f"must be one of: {', '.join(sorted(choices))}")
    return dtype


def _engine_checks(path: Path, value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise _error(path, "engine.check", "must be a string or non-empty array")
    if not all(isinstance(item, str) and item in {"vllm", "sglang", "all"} for item in values):
        raise _error(path, "engine.check", "entries must be vllm, sglang, or all")
    return list(values)


def _engine_python(path: Path, value: Any) -> str:
    executable = _string(path, "engine.python", value)
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or "/" not in executable:
        return str(candidate)
    # Do not use Path.resolve(): venv bin/python is normally a symlink, and
    # dereferencing it loses the environment whose packages we need to probe.
    return os.path.abspath(path.parent / candidate)


def load_toml_defaults(value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(value).expanduser().resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ValueError(f"TOML config was not found: {path}") from error
    except PermissionError as error:
        raise ValueError(f"TOML config is not readable: {path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"invalid TOML config {path}: {error}") from error
    if not isinstance(data, Mapping):  # pragma: no cover - tomllib guarantees a dict
        raise ValueError(f"invalid TOML config {path}: root must be a table")
    _check_keys(path, data)
    if "system" in data and "hardware" in data:
        raise _error(path, ".", "cannot contain both system and hardware")
    if "system" in data and "gpus" in data:
        raise _error(path, ".", "system derives accelerator count; use nodes instead of gpus")
    if "nodes" in data and "system" not in data:
        raise _error(path, "nodes", "requires system")

    defaults: dict[str, Any] = {}
    for field in ("model", "revision", "hardware", "system"):
        if field in data:
            defaults[field] = _string(path, field, data[field])
    if "context" in data:
        defaults["context"] = _context(path, data["context"])
    if "kv_dtype" in data:
        defaults["kv_dtype"] = _dtype(path, "kv_dtype", data["kv_dtype"], allow_auto=True)
    if "index_dtype" in data:
        defaults["index_dtype"] = _dtype(path, "index_dtype", data["index_dtype"], allow_auto=False)
    if "weight_gib" in data:
        defaults["weight_gib"] = _positive_float(path, "weight_gib", data["weight_gib"])
    if "allow_context_overflow" in data:
        defaults["allow_context_overflow"] = _bool(
            path, "allow_context_overflow", data["allow_context_overflow"]
        )
    if "gpus" in data:
        defaults["gpus"] = _int(path, "gpus", data["gpus"])
    if "nodes" in data:
        defaults["nodes"] = _int(path, "nodes", data["nodes"])
    if "utilization" in data:
        utilization = _float(path, "utilization", data["utilization"])
        if not 0 < utilization <= 1:
            raise _error(path, "utilization", "must be greater than 0 and at most 1")
        defaults["utilization"] = utilization
    if "tp" in data:
        defaults["tp"] = _tp(path, "tp", data["tp"])
    if "active_sequences_per_user" in data:
        defaults["active_sequences_per_user"] = _int(
            path,
            "active_sequences_per_user",
            data["active_sequences_per_user"],
        )
    if "timeout" in data:
        defaults["timeout"] = _positive_float(path, "timeout", data["timeout"])

    engine = _mapping(path, data, "engine")
    if "check" in engine:
        defaults["check_engine"] = _engine_checks(path, engine["check"])
    if "python" in engine:
        defaults["engine_python"] = _engine_python(path, engine["python"])
    if "probe" in engine:
        probe = _string(path, "engine.probe", engine["probe"]).lower()
        if probe not in {"registry", "config", "load"}:
            raise _error(path, "engine.probe", "must be registry, config, or load")
        defaults["engine_probe"] = probe
    if "tp" in engine:
        defaults["engine_tp"] = _int(path, "engine.tp", engine["tp"])
    if "timeout" in engine:
        defaults["engine_timeout"] = _positive_float(path, "engine.timeout", engine["timeout"])
    if "require_pass" in engine:
        defaults["require_engine_pass"] = _bool(path, "engine.require_pass", engine["require_pass"])

    output = _mapping(path, data, "output")
    if "json" in output:
        defaults["json"] = _bool(path, "output.json", output["json"])
    return path, defaults


def toml_path_from_argv(argv: list[str]) -> tuple[list[str], str | None]:
    normalized = list(argv)
    if normalized and normalized[0].lower().endswith(".toml"):
        normalized = ["--config", normalized[0], *normalized[1:]]
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    known, _ = bootstrap.parse_known_args(normalized)
    return normalized, known.config
