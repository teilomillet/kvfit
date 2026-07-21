from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from kvfit.models import ModelMetadata

EngineName = Literal["vllm", "sglang"]
ProbeMode = Literal["registry", "config", "load"]
MARKER = "KVFIT_ENGINE_CHECK_JSON="
VALID_STEP_STATUSES = {"pass", "fail", "unknown", "not-applicable"}
VALID_OVERALL_STATUSES = {
    "preflight-pass",
    "smoke-pass",
    "runtime-fail",
    "unsupported",
    "unknown",
    "unavailable",
    "error",
}


@dataclass(frozen=True)
class EngineStep:
    status: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "detail": self.detail}


@dataclass(frozen=True)
class EngineCheck:
    engine: str
    python: str
    installed: bool
    version: str | None
    mode: str
    verification_level: str
    load_tested: bool
    architectures: tuple[str, ...]
    matched_architecture: str | None
    quantization: str | None
    resolved_quantization: str | None
    platform: Mapping[str, str | None]
    steps: Mapping[str, EngineStep]
    overall: str
    summary: str
    kv_dtype: str | None = None
    resolved_kv_dtype: str | None = None
    index_dtype: str | None = None
    resolved_index_dtype: str | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.overall in {"preflight-pass", "smoke-pass"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "python": self.python,
            "installed": self.installed,
            "version": self.version,
            "mode": self.mode,
            "verification_level": self.verification_level,
            "load_tested": self.load_tested,
            "architectures": list(self.architectures),
            "matched_architecture": self.matched_architecture,
            "quantization": self.quantization,
            "resolved_quantization": self.resolved_quantization,
            "kv_dtype": self.kv_dtype,
            "resolved_kv_dtype": self.resolved_kv_dtype,
            "index_dtype": self.index_dtype,
            "resolved_index_dtype": self.resolved_index_dtype,
            "platform": dict(self.platform),
            "steps": {name: step.as_dict() for name, step in self.steps.items()},
            "overall": self.overall,
            "summary": self.summary,
            "diagnostics": list(self.diagnostics),
        }


def _architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = config.get("architectures")
    if not candidates:
        nested = config.get("text_config")
        if isinstance(nested, Mapping):
            candidates = nested.get("architectures")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    return tuple(str(value) for value in candidates if isinstance(value, str) and value)


def _error_check(
    engine: str,
    python: str,
    mode: str,
    summary: str,
    *,
    diagnostics: Sequence[str] = (),
) -> EngineCheck:
    return EngineCheck(
        engine=engine,
        python=python,
        installed=False,
        version=None,
        mode=mode,
        verification_level={
            "registry": "registry-only",
            "config": "config-only",
            "load": "weight-load-and-one-token",
        }.get(mode, "unknown"),
        load_tested=False,
        architectures=(),
        matched_architecture=None,
        quantization=None,
        resolved_quantization=None,
        platform={"device_type": None, "device_name": None, "enum": None},
        steps={
            "package": EngineStep("unknown", summary),
            "architecture": EngineStep("unknown", "probe did not run"),
            "quantization": EngineStep("unknown", "probe did not run"),
            "kv_cache": EngineStep("unknown", "probe did not run"),
            "index_cache": EngineStep("unknown", "probe did not run"),
            "platform": EngineStep("unknown", "probe did not run"),
            "config": EngineStep("unknown", "probe did not run"),
            "smoke": EngineStep("unknown", "probe did not run"),
        },
        overall="error",
        summary=summary,
        diagnostics=tuple(diagnostics),
    )


def _parse_result(
    payload: Mapping[str, Any],
    *,
    engine: str,
    requested_python: str,
    mode: str,
    diagnostics: Sequence[str],
) -> EngineCheck:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Mapping):
        raise ValueError("engine probe returned no step results")
    steps: dict[str, EngineStep] = {}
    for name in (
        "package",
        "architecture",
        "quantization",
        "kv_cache",
        "index_cache",
        "platform",
        "config",
        "smoke",
    ):
        raw_step = raw_steps.get(name)
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"engine probe returned no {name} step")
        status = str(raw_step.get("status", "unknown"))
        if status not in VALID_STEP_STATUSES:
            raise ValueError(f"engine probe returned invalid {name} status {status!r}")
        steps[name] = EngineStep(status, str(raw_step.get("detail", "")))
    overall = str(payload.get("overall", "unknown"))
    if overall not in VALID_OVERALL_STATUSES:
        raise ValueError(f"engine probe returned invalid overall status {overall!r}")
    platform = payload.get("platform")
    if not isinstance(platform, Mapping):
        platform = {"device_type": None, "device_name": None, "enum": None}
    architectures = payload.get("architectures")
    if not isinstance(architectures, list):
        architectures = []
    return EngineCheck(
        engine=str(payload.get("engine", engine)),
        python=str(payload.get("python", requested_python)),
        installed=bool(payload.get("installed", False)),
        version=str(payload["version"]) if payload.get("version") is not None else None,
        mode=str(payload.get("mode", mode)),
        verification_level=str(payload.get("verification_level", "unknown")),
        load_tested=bool(payload.get("load_tested", False)),
        architectures=tuple(str(value) for value in architectures),
        matched_architecture=(
            str(payload["matched_architecture"])
            if payload.get("matched_architecture") is not None
            else None
        ),
        quantization=(
            str(payload["quantization"]) if payload.get("quantization") is not None else None
        ),
        resolved_quantization=(
            str(payload["resolved_quantization"])
            if payload.get("resolved_quantization") is not None
            else None
        ),
        platform={
            "device_type": (
                str(platform["device_type"]) if platform.get("device_type") is not None else None
            ),
            "device_name": (
                str(platform["device_name"]) if platform.get("device_name") is not None else None
            ),
            "enum": str(platform["enum"]) if platform.get("enum") is not None else None,
        },
        steps=steps,
        overall=overall,
        summary=str(payload.get("summary", "")),
        kv_dtype=str(payload["kv_dtype"]) if payload.get("kv_dtype") is not None else None,
        resolved_kv_dtype=(
            str(payload["resolved_kv_dtype"])
            if payload.get("resolved_kv_dtype") is not None
            else None
        ),
        index_dtype=(
            str(payload["index_dtype"]) if payload.get("index_dtype") is not None else None
        ),
        resolved_index_dtype=(
            str(payload["resolved_index_dtype"])
            if payload.get("resolved_index_dtype") is not None
            else None
        ),
        diagnostics=tuple(diagnostics),
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if hasattr(os, "killpg"):
        os.killpg(process.pid, signal.SIGTERM)
    else:  # pragma: no cover - Windows fallback
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    if hasattr(os, "killpg"):
        os.killpg(process.pid, signal.SIGKILL)
    else:  # pragma: no cover - Windows fallback
        process.kill()
    process.wait(timeout=5)


def _run_probe_process(
    executable: str,
    probe_path: Path,
    payload: Mapping[str, Any],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [executable, str(probe_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        raise
    return subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def check_engine(
    engine: EngineName,
    metadata: ModelMetadata,
    *,
    python: str = sys.executable,
    mode: ProbeMode = "config",
    context_tokens: int = 128 * 1024,
    timeout: float = 60.0,
    tensor_parallel: int = 1,
    utilization: float = 0.9,
    kv_dtype: str = "bf16",
    index_dtype: str | None = None,
) -> EngineCheck:
    executable = shutil.which(python) if not Path(python).is_file() else str(Path(python))
    if executable is None:
        return _error_check(engine, python, mode, f"Python interpreter was not found: {python}")
    probe_path = Path(__file__).with_name("_engine_probe.py")
    payload = {
        "engine": engine,
        "mode": mode,
        "repo_id": metadata.repo_id,
        "revision": metadata.resolved_revision or metadata.requested_revision,
        "architectures": list(_architectures(metadata.config)),
        "quantization": metadata.quantization,
        "model_type": str(
            (
                metadata.config.get("text_config")
                if isinstance(metadata.config.get("text_config"), Mapping)
                else metadata.config
            ).get("model_type", metadata.config.get("model_type", ""))
        ),
        "kv_dtype": kv_dtype,
        "index_dtype": index_dtype,
        "selected_artifact": metadata.selected_artifact,
        "context_tokens": context_tokens,
        "tensor_parallel": tensor_parallel,
        "utilization": utilization,
    }
    try:
        completed = _run_probe_process(executable, probe_path, payload, timeout)
    except subprocess.TimeoutExpired as error:
        return _error_check(
            engine,
            executable,
            mode,
            f"engine probe timed out after {timeout:g} seconds",
            diagnostics=(str(error),),
        )
    diagnostics = tuple(
        value
        for value in (
            completed.stderr.strip()[-4000:],
            f"probe exited with status {completed.returncode}" if completed.returncode else "",
        )
        if value
    )
    marked = [
        line[len(MARKER) :] for line in completed.stdout.splitlines() if line.startswith(MARKER)
    ]
    if not marked:
        return _error_check(
            engine,
            executable,
            mode,
            "engine probe returned no machine-readable result",
            diagnostics=(*diagnostics, completed.stdout.strip()[-4000:]),
        )
    try:
        raw_result = json.loads(marked[-1])
        if not isinstance(raw_result, dict):
            raise ValueError("engine probe result is not a JSON object")
        if "steps" not in raw_result:
            raise ValueError(str(raw_result.get("summary", "engine probe failed")))
        return _parse_result(
            raw_result,
            engine=engine,
            requested_python=executable,
            mode=mode,
            diagnostics=diagnostics,
        )
    except (json.JSONDecodeError, ValueError) as error:
        return _error_check(
            engine,
            executable,
            mode,
            f"invalid engine probe result: {error}",
            diagnostics=diagnostics,
        )


def check_engines(
    engines: Sequence[EngineName],
    metadata: ModelMetadata,
    *,
    python: str = sys.executable,
    mode: ProbeMode = "config",
    context_tokens: int = 128 * 1024,
    timeout: float = 60.0,
    tensor_parallel: int = 1,
    utilization: float = 0.9,
    kv_dtype: str = "bf16",
    index_dtype: str | None = None,
) -> tuple[EngineCheck, ...]:
    return tuple(
        check_engine(
            engine,
            metadata,
            python=python,
            mode=mode,
            context_tokens=context_tokens,
            timeout=timeout,
            tensor_parallel=tensor_parallel,
            utilization=utilization,
            kv_dtype=kv_dtype,
            index_dtype=index_dtype,
        )
        for engine in engines
    )
