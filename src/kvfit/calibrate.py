from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kvfit import __version__
from kvfit.calibration_config import CalibrationConfig, load_calibration_config

OOM_RE = re.compile(
    r"(?:torch\.OutOfMemoryError|OutOfMemoryError|CUDA out of memory|CUDA OOM|"
    r"CUDA error: out of memory|HIP out of memory|MemoryError:|cannot allocate memory|"
    r"failed to allocate)",
    re.IGNORECASE,
)
PROMETHEUS_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]Inf|NaN)"
)
RUNNING_METRICS = (
    "vllm:num_requests_running",
    "vllm_num_requests_running",
    "sglang:num_running_reqs",
    "sglang_num_running_reqs",
    "sglang:num_running_requests",
)
WAITING_METRICS = (
    "vllm:num_requests_waiting",
    "vllm_num_requests_waiting",
    "sglang:num_queue_reqs",
    "sglang_num_queue_reqs",
    "sglang:num_requests_waiting",
)
KV_USAGE_METRICS = (
    "vllm:kv_cache_usage_perc",
    "vllm_kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
    "vllm_gpu_cache_usage_perc",
    "sglang:token_usage",
    "sglang_token_usage",
)
INPUT_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm_prompt_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang_prompt_tokens_total",
)
OUTPUT_COUNTERS = (
    "vllm:generation_tokens_total",
    "vllm_generation_tokens_total",
    "sglang:generation_tokens_total",
    "sglang_generation_tokens_total",
)
KV_DTYPE_ALIASES = {
    "vllm": {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp8": "fp8",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "int8": "int8_per_token_head",
        "int4": "int4_per_token_head",
        "fp4": "nvfp4",
        "nvfp4": "nvfp4",
    },
    "sglang": {
        "bf16": "bfloat16",
        "fp8": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e5m2": "fp8_e5m2",
        "fp4": "fp4_e2m1",
        "mxfp4": "fp4_e2m1",
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _tail(value: str, lines: int = 120) -> str:
    return "\n".join(value.splitlines()[-lines:])


def _json_request(
    url: str,
    *,
    timeout: float,
    api_key: str | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else None


def _text_request(
    url: str,
    *,
    timeout: float,
    api_key: str | None = None,
) -> str:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_prometheus(value: str) -> dict[str, list[float]]:
    parsed: dict[str, list[float]] = {}
    for line in value.splitlines():
        match = PROMETHEUS_RE.match(line.strip())
        if not match:
            continue
        try:
            number = float(match.group("value"))
        except ValueError:
            continue
        if not math.isfinite(number):
            continue
        parsed.setdefault(match.group("name"), []).append(number)
    return parsed


def _metric_total(metrics: dict[str, list[float]], names: tuple[str, ...]) -> float | None:
    for name in names:
        values = metrics.get(name)
        if values:
            return sum(values)
    return None


def _metric_max(metrics: dict[str, list[float]], names: tuple[str, ...]) -> float | None:
    for name in names:
        values = metrics.get(name)
        if values:
            return max(values)
    return None


@dataclass
class MetricsSampler:
    url: str
    interval: float
    timeout: float
    api_key: str | None
    samples: list[dict[str, float | str | None]] = field(default_factory=list)
    metric_names: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.timeout + 1.0))
        return self.summary()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = _text_request(self.url, timeout=self.timeout, api_key=self.api_key)
                metrics = parse_prometheus(raw)
                self.metric_names.update(metrics)
                self.samples.append(
                    {
                        "timestamp": _utc_now(),
                        "running": _metric_total(metrics, RUNNING_METRICS),
                        "waiting": _metric_total(metrics, WAITING_METRICS),
                        "kv_usage": _metric_max(metrics, KV_USAGE_METRICS),
                        "input_tokens_total": _metric_total(metrics, INPUT_COUNTERS),
                        "output_tokens_total": _metric_total(metrics, OUTPUT_COUNTERS),
                    }
                )
            except Exception as error:  # metrics are auxiliary evidence
                detail = f"{type(error).__name__}: {error}"
                if not self.errors or self.errors[-1] != detail:
                    self.errors.append(detail)
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        def finite_values(name: str) -> list[float]:
            return [
                float(sample[name])
                for sample in self.samples
                if isinstance(sample.get(name), (int, float))
            ]

        running = finite_values("running")
        waiting = finite_values("waiting")
        usage = finite_values("kv_usage")
        input_tokens = finite_values("input_tokens_total")
        output_tokens = finite_values("output_tokens_total")
        return {
            "endpoint": self.url,
            "sample_count": len(self.samples),
            "max_running": max(running) if running else None,
            "max_waiting": max(waiting) if waiting else None,
            "max_kv_usage": max(usage) if usage else None,
            "input_token_delta": (
                max(input_tokens) - min(input_tokens) if len(input_tokens) >= 2 else None
            ),
            "output_token_delta": (
                max(output_tokens) - min(output_tokens) if len(output_tokens) >= 2 else None
            ),
            "recognized": {
                "running": any(name in self.metric_names for name in RUNNING_METRICS),
                "waiting": any(name in self.metric_names for name in WAITING_METRICS),
                "kv_usage": any(name in self.metric_names for name in KV_USAGE_METRICS),
            },
            "metric_names": sorted(self.metric_names),
            "errors": self.errors[-5:],
        }


def _api_key(config: CalibrationConfig) -> str | None:
    if config.api_key_env is None:
        return None
    value = os.environ.get(config.api_key_env)
    if not value:
        raise ValueError(
            f"environment variable {config.api_key_env!r} is required by calibration.api_key_env"
        )
    return value


def _prediction(config: CalibrationConfig) -> dict[str, Any]:
    # Run the normal public planning path in-process so calibration and the CLI
    # cannot silently drift to separate formulas.
    from kvfit.cli import main as planning_main

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = planning_main(
            [
                str(config.path),
                "--json",
                "--no-require-engine-pass",
                "--check-engine",
                config.engine,
                "--engine-python",
                config.python,
                "--engine-probe",
                "config",
                "--engine-timeout",
                str(config.startup_timeout),
            ]
        )
    if status != 0:
        detail = stderr.getvalue().strip() or stdout.getvalue().strip()
        raise ValueError(f"deployment prediction failed before calibration: {detail}")
    try:
        payload = json.loads(stdout.getvalue())
    except json.JSONDecodeError as error:
        raise ValueError("deployment prediction did not produce valid JSON") from error
    if not isinstance(payload.get("cache"), dict) or not isinstance(
        payload.get("concurrency"), dict
    ):
        raise ValueError("calibration requires a deployment with hardware capacity results")
    return payload


def _environment(config: CalibrationConfig) -> dict[str, Any]:
    script = (
        "import importlib.metadata as m,json,platform;"
        f"name={config.engine!r};"
        "p={'python':platform.python_version(),'platform':platform.platform()};"
        "\ntry:p['engine_version']=m.version(name)\nexcept m.PackageNotFoundError:"
        "p['engine_version']=None\n"
        "try:\n import torch\n p['torch_version']=torch.__version__;"
        "p['cuda_available']=torch.cuda.is_available();"
        "p['device_count']=torch.cuda.device_count();"
        "p['device_names']=[torch.cuda.get_device_name(i) "
        "for i in range(torch.cuda.device_count())]"
        "\nexcept Exception as e:p['torch_error']=type(e).__name__+': '+str(e)\n"
        "print(json.dumps(p))"
    )
    try:
        completed = subprocess.run(
            [config.python, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"python_executable": config.python, "probe_error": str(error)}
    try:
        result = json.loads(completed.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result = {
            "probe_error": _tail(completed.stderr or completed.stdout, 20),
            "returncode": completed.returncode,
        }
    result["python_executable"] = config.python
    return result


def _model_details(prediction: dict[str, Any]) -> tuple[str, str | None, int]:
    model = prediction.get("model", {})
    cache = prediction.get("cache", {})
    repo_id = model.get("repo_id")
    revision = model.get("resolved_revision") or model.get("requested_revision")
    context = cache.get("context_tokens")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("prediction is missing the Hugging Face repository ID")
    if not isinstance(context, int) or context < 1:
        raise ValueError("prediction is missing the context length")
    return repo_id, revision if isinstance(revision, str) else None, context


def _recommended_layout(prediction: dict[str, Any]) -> dict[str, Any]:
    recommended = prediction["concurrency"].get("recommended")
    if not isinstance(recommended, dict) or not isinstance(recommended.get("layout"), dict):
        raise ValueError("memory planner found no deployment layout that fits")
    return recommended["layout"]


def _format_command(command: tuple[str, ...], values: dict[str, Any]) -> list[str]:
    formatted: list[str] = []
    for item in command:
        try:
            formatted.append(item.format_map(values))
        except KeyError as error:
            raise ValueError(f"unknown command placeholder {{{error.args[0]}}}") from error
    return formatted


def _command_values(
    config: CalibrationConfig,
    prediction: dict[str, Any],
    *,
    served_model: str | None = None,
    concurrency: int | None = None,
    num_prompts: int | None = None,
    result_file: Path | None = None,
) -> dict[str, Any]:
    model, revision, context = _model_details(prediction)
    layout = _recommended_layout(prediction)
    prompt_tokens = context - config.output_tokens
    return {
        "python": config.python,
        "engine": config.engine,
        "base_url": config.base_url,
        "host": config.host,
        "port": config.port,
        "model": model,
        "served_model": served_model or model,
        "tokenizer": model,
        "revision": revision or "",
        "context": context,
        "input_tokens": prompt_tokens,
        "output_tokens": config.output_tokens,
        "concurrency": concurrency or 1,
        "num_prompts": num_prompts or 1,
        "result_file": str(result_file) if result_file else "",
        "seed": config.seed,
        "tensor_parallel": int(layout["tensor_parallel"]),
        "data_parallel": int(layout["data_parallel"]),
        "utilization": float(prediction["hardware"]["utilization"]),
        "max_concurrency": config.max_concurrency,
    }


def _default_server_command(
    config: CalibrationConfig,
    prediction: dict[str, Any],
) -> list[str]:
    if prediction.get("cache_layout", "logical") != "logical":
        raise ValueError(
            "a storage profile does not configure an engine backend or MTP; "
            "provide calibration.server_command with matching settings, or attach "
            "to an explicitly configured server"
        )
    values = _command_values(config, prediction)
    model, revision, _ = _model_details(prediction)
    nodes = int((prediction.get("system") or {}).get("systems", 1))
    if nodes > 1:
        raise ValueError(
            "default managed launch is single-host only; for a multi-system deployment, "
            "attach to its serving endpoint or provide calibration.server_command"
        )
    if config.engine == "vllm":
        command = [
            config.python,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(values["tensor_parallel"]),
            "--data-parallel-size",
            str(values["data_parallel"]),
            "--max-model-len",
            str(values["context"]),
            "--gpu-memory-utilization",
            str(values["utilization"]),
            "--max-num-seqs",
            str(config.max_concurrency),
        ]
        if revision:
            command.extend(["--revision", revision, "--tokenizer-revision", revision])
    else:
        command = [
            config.python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tp",
            str(values["tensor_parallel"]),
            "--dp",
            str(values["data_parallel"]),
            "--context-length",
            str(values["context"]),
            "--mem-fraction-static",
            str(values["utilization"]),
            "--enable-metrics",
        ]
        if revision:
            command.extend(["--revision", revision])

    precision = prediction.get("precision", {})
    kv_dtype = precision.get("kv_dtype")
    resolved = KV_DTYPE_ALIASES[config.engine].get(str(kv_dtype))
    if resolved:
        command.extend(["--kv-cache-dtype", resolved])
    index_dtype = precision.get("index_dtype")
    if index_dtype and index_dtype != kv_dtype:
        if config.engine == "vllm" and index_dtype in {"fp4", "mxfp4"}:
            command.extend(["--attention-config", '{"use_fp4_indexer_cache": true}'])
        elif not config.server_args:
            raise ValueError(
                f"{config.engine} has no stable generic launch flag for independent index "
                f"dtype {index_dtype}; provide the installed version's flag in "
                "calibration.server_args or attach to an already configured server"
            )
    command.extend(config.server_args)
    return command


def _server_command(config: CalibrationConfig, prediction: dict[str, Any]) -> list[str]:
    if config.server_command:
        return _format_command(config.server_command, _command_values(config, prediction))
    return _default_server_command(config, prediction)


def _sanitize_command(command: list[str], secret: str | None = None) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for item in command:
        lowered = item.lower()
        if redact_next or (secret and secret in item):
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if any(marker in lowered for marker in ("api-key", "api_key", "password", "token=")):
            if "=" in item:
                sanitized.append(item.split("=", 1)[0] + "=<redacted>")
            else:
                sanitized.append(item)
                redact_next = True
            continue
        if lowered.startswith("authorization="):
            sanitized.append("Authorization=<redacted>")
            continue
        sanitized.append(item)
    return sanitized


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


def _endpoint_evidence(config: CalibrationConfig, api_key: str | None) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for endpoint in ("/v1/models", "/version", "/get_model_info", "/get_server_info"):
        try:
            evidence[endpoint] = _json_request(
                config.base_url + endpoint,
                timeout=min(10.0, config.request_timeout),
                api_key=api_key,
            )
        except Exception as error:
            evidence[endpoint] = {"unavailable": f"{type(error).__name__}: {error}"}
    return evidence


def _served_model(endpoint_evidence: dict[str, Any], fallback: str) -> str:
    models = endpoint_evidence.get("/v1/models")
    if isinstance(models, dict) and isinstance(models.get("data"), list) and models["data"]:
        first = models["data"][0]
        if isinstance(first, dict) and isinstance(first.get("id"), str):
            return first["id"]
    return fallback


def _wait_ready(
    config: CalibrationConfig,
    process: subprocess.Popen[Any] | None,
    api_key: str | None,
) -> dict[str, Any]:
    deadline = time.monotonic() + config.startup_timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        for endpoint in ("/health", "/v1/models"):
            try:
                if endpoint == "/health":
                    _text_request(
                        config.base_url + endpoint,
                        timeout=5,
                        api_key=api_key,
                    )
                else:
                    _json_request(
                        config.base_url + endpoint,
                        timeout=5,
                        api_key=api_key,
                    )
                return _endpoint_evidence(config, api_key)
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
        time.sleep(1)
    raise TimeoutError(
        f"server at {config.base_url} was not ready after {config.startup_timeout:g}s: {last_error}"
    )


def _concurrency_points(config: CalibrationConfig, prediction: dict[str, Any]) -> list[int]:
    if config.concurrency:
        return list(config.concurrency)
    recommended = prediction["concurrency"].get("recommended") or {}
    predicted = int(recommended.get("active_sequences", 0))
    cap = min(max(predicted, 1), config.max_concurrency)
    points: list[int] = []
    value = 1
    while value <= cap:
        points.append(value)
        value *= 2
    if cap not in points:
        points.append(cap)
    return sorted(set(points))


def _default_benchmark_command(
    config: CalibrationConfig,
    prediction: dict[str, Any],
    *,
    served_model: str,
    concurrency: int,
    num_prompts: int,
    result_file: Path,
    api_key: str | None,
) -> list[str]:
    model, _, context = _model_details(prediction)
    input_tokens = context - config.output_tokens
    if config.engine == "vllm":
        command = [
            config.python,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            config.base_url,
            "--endpoint",
            "/v1/completions",
            "--model",
            model,
            "--served-model-name",
            served_model,
            "--tokenizer",
            model,
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_tokens),
            "--random-output-len",
            str(config.output_tokens),
            "--random-range-ratio",
            "0",
            "--num-prompts",
            str(num_prompts),
            "--request-rate",
            "inf",
            "--max-concurrency",
            str(concurrency),
            "--seed",
            str(config.seed),
            "--ignore-eos",
            "--percentile-metrics",
            "ttft,tpot,itl,e2el",
            "--metric-percentiles",
            "50,95,99",
            "--save-result",
            "--result-dir",
            str(result_file.parent),
            "--result-filename",
            result_file.name,
            "--disable-tqdm",
        ]
        if api_key:
            command.extend(["--header", f"Authorization=Bearer {api_key}"])
        return command
    command = [
        config.python,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-oai",
        "--base-url",
        config.base_url,
        "--model",
        served_model,
        "--tokenizer",
        model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_tokens),
        "--random-output-len",
        str(config.output_tokens),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(config.seed),
        "--warmup-requests",
        "0",
        "--disable-tqdm",
        "--output-file",
        str(result_file),
        "--output-details",
    ]
    return command


def _load_benchmark_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"raw": value}
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line in raw.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                rows.append(json.loads(line))
        for row in reversed(rows):
            if isinstance(row, dict) and any(
                key in row for key in ("completed", "request_throughput", "mean_ttft_ms")
            ):
                return row
        return {"jsonl": rows}


def _metric_value(result: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = result.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            return float(value)
    return None


def _slo_result(config: CalibrationConfig, result: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    specifications = (
        (
            "ttft_ms",
            config.max_ttft_ms,
            ("p95_ttft_ms", "p99_ttft_ms"),
        ),
        (
            "tpot_ms",
            config.max_tpot_ms,
            ("p95_tpot_ms", "p99_tpot_ms"),
        ),
        (
            "e2e_ms",
            config.max_e2e_ms,
            (
                "p95_e2el_ms",
                "p95_e2e_latency_ms",
                "p99_e2e_latency_ms",
                "p95_e2e_ms",
            ),
        ),
    )
    for label, limit, names in specifications:
        if limit is None:
            continue
        measured = _metric_value(result, *names)
        checks[label] = {
            "limit": limit,
            "measured": measured,
            "source": next(
                (name for name in names if _metric_value(result, name) is not None), None
            ),
            "passed": measured is not None and measured <= limit,
        }
    return {
        "passed": bool(checks) and all(value["passed"] for value in checks.values()),
        "configured": bool(checks),
        "checks": checks,
    }


def _run_one(
    config: CalibrationConfig,
    prediction: dict[str, Any],
    *,
    served_model: str,
    concurrency: int,
    api_key: str | None,
    directory: Path,
) -> dict[str, Any]:
    num_prompts = concurrency * config.requests_per_concurrency
    result_file = directory / f"concurrency-{concurrency}.json"
    if config.benchmark_command:
        command = _format_command(
            config.benchmark_command,
            _command_values(
                config,
                prediction,
                served_model=served_model,
                concurrency=concurrency,
                num_prompts=num_prompts,
                result_file=result_file,
            ),
        )
    else:
        command = _default_benchmark_command(
            config,
            prediction,
            served_model=served_model,
            concurrency=concurrency,
            num_prompts=num_prompts,
            result_file=result_file,
            api_key=api_key,
        )
    sampler = MetricsSampler(
        config.base_url + "/metrics",
        interval=config.metrics_interval,
        timeout=min(5.0, config.metrics_interval),
        api_key=api_key,
    )
    started_at = _utc_now()
    start = time.monotonic()
    sampler.start()
    environment = os.environ.copy()
    if api_key:
        environment["OPENAI_API_KEY"] = api_key
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=config.request_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate(process)
        stdout, stderr = process.communicate()
    metrics = sampler.stop()
    duration = time.monotonic() - start
    raw_result = _load_benchmark_result(result_file)
    combined_output = stdout + "\n" + stderr
    oom_observed = bool(OOM_RE.search(combined_output))
    completed = raw_result.get("completed")
    exact_input = raw_result.get("total_input_tokens")
    _, _, context = _model_details(prediction)
    input_tokens = context - config.output_tokens
    exact_tokens_verified = (
        type(completed) is int
        and type(exact_input) is int
        and exact_input == completed * input_tokens
    )
    exact_output = raw_result.get("total_output_tokens")
    exact_output_verified = (
        type(completed) is int
        and type(exact_output) is int
        and exact_output == completed * config.output_tokens
    )
    completed_all = type(completed) is int and completed == num_prompts
    success = (
        process.returncode == 0
        and completed_all
        and exact_tokens_verified
        and exact_output_verified
        and not timed_out
        and not oom_observed
    )
    slo = _slo_result(config, raw_result)
    return {
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "started_at": started_at,
        "wall_duration_seconds": duration,
        "command": _sanitize_command(command, api_key),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "oom_observed": oom_observed,
        "completed_all": completed_all,
        "success": success,
        "slo": slo,
        "slo_qualified": success and slo["passed"],
        "exact_input_tokens_verified": exact_tokens_verified,
        "exact_output_tokens_verified": exact_output_verified,
        "expected_input_tokens_per_request": input_tokens,
        "benchmark": raw_result,
        "metrics": metrics,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
    }


def _startup_evidence(log: str) -> dict[str, Any]:
    patterns = {
        "max_total_num_tokens": r"max_total_num_tokens[=:]\s*([0-9,]+)",
        "max_running_requests": r"max_running_requests[=:]\s*([0-9,]+)",
        "gpu_kv_cache_tokens": r"GPU KV cache size:\s*([0-9,]+) tokens",
        "available_gpu_memory_gib": r"available_gpu_mem[=:]\s*([0-9.]+)",
    }
    values: dict[str, int | float] = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, log, re.IGNORECASE)
        if not matches:
            continue
        raw = matches[-1].replace(",", "")
        values[name] = float(raw) if "." in raw else int(raw)
    return values


def _comparison(
    config: CalibrationConfig,
    prediction: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    recommended = prediction["concurrency"]["recommended"]
    predicted_sequences = int(recommended["active_sequences"])
    per_user = int(prediction["concurrency"]["active_sequences_per_user"])
    successful = [run for run in runs if run["success"]]
    qualified = [run for run in runs if run["slo_qualified"]]
    max_success = max((run["concurrency"] for run in successful), default=None)
    max_qualified = max((run["concurrency"] for run in qualified), default=None)
    max_running_values = [
        run["metrics"]["max_running"]
        for run in runs
        if run["metrics"].get("max_running") is not None
    ]
    first_queue = next(
        (run["concurrency"] for run in runs if (run["metrics"].get("max_waiting") or 0) > 0),
        None,
    )
    first_failure = next((run["concurrency"] for run in runs if not run["success"]), None)
    first_oom = next((run["concurrency"] for run in runs if run["oom_observed"]), None)
    highest_requested = max((run["concurrency"] for run in runs), default=0)
    explicit = config.concurrency is not None
    truncated = not explicit and highest_requested < predicted_sequences
    return {
        "predicted_memory_only_active_sequences": predicted_sequences,
        "predicted_memory_only_users": int(recommended["concurrent_users"]),
        "active_sequences_per_user": per_user,
        "max_successful_offered_concurrency": max_success,
        "max_successful_users": max_success // per_user if max_success is not None else None,
        "max_slo_qualified_offered_concurrency": max_qualified,
        "max_slo_qualified_users": (
            max_qualified // per_user if max_qualified is not None else None
        ),
        "max_observed_running_requests": max(max_running_values) if max_running_values else None,
        "first_observed_queueing_concurrency": first_queue,
        "first_failed_concurrency": first_failure,
        "first_actual_oom_concurrency": first_oom,
        "successful_to_predicted_ratio": (
            max_success / predicted_sequences if max_success is not None else None
        ),
        "search_truncated_below_prediction": truncated,
        "successful_result_is_lower_bound": bool(max_success is not None and first_failure is None),
    }


def run_calibration(
    config: CalibrationConfig,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    prediction = _prediction(config)
    model, _, context = _model_details(prediction)
    if config.output_tokens >= context:
        raise ValueError("calibration.output_tokens must be smaller than the deployment context")
    api_key = _api_key(config)
    environment = _environment(config)
    server_command = _server_command(config, prediction) if config.mode == "launch" else None
    checks = [
        check
        for check in prediction.get("engine_checks", [])
        if check.get("engine") == config.engine
    ]
    if (
        not dry_run
        and config.mode == "launch"
        and config.server_command is None
        and (not checks or checks[-1].get("overall") != "preflight-pass")
    ):
        verdict = checks[-1].get("overall") if checks else "missing"
        raise ValueError(
            f"installed {config.engine} preflight is {verdict}, so managed launch is refused; "
            "fix the engine environment, attach to a separately qualified endpoint, or provide "
            "an explicit server_command"
        )
    if (
        not dry_run
        and config.mode == "launch"
        and config.server_command is None
        and not config.allow_cpu
        and not environment.get("cuda_available")
    ):
        raise ValueError(
            "managed launch found no CUDA accelerator in calibration.python; run on the target "
            "host, use mode='attach' for a remote endpoint, or set allow_cpu=true only for a "
            "deliberate non-production test"
        )

    points = _concurrency_points(config, prediction)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "kvfit_version": __version__,
        "created_at": _utc_now(),
        "input_config": str(config.path),
        "qualification": "not-run" if dry_run else "in-progress",
        "engine": {
            "name": config.engine,
            "mode": config.mode,
            "base_url": config.base_url,
            "environment": environment,
            "server_command": _sanitize_command(server_command or [], api_key) or None,
            "server_configuration_verification": (
                "managed-command-recorded"
                if config.mode == "launch"
                else "attached-endpoint-reported-only"
            ),
        },
        "workload": {
            "model": model,
            "context_tokens": context,
            "input_tokens": context - config.output_tokens,
            "output_tokens": config.output_tokens,
            "concurrency_points": points,
            "requests_per_concurrency": config.requests_per_concurrency,
            "seed": config.seed,
            "steady_state_qualified": config.requests_per_concurrency >= 5,
        },
        "prediction": prediction,
        "runs": [],
        "boundaries": [
            "A successful offered concurrency is not proof that every request was resident "
            "at once; "
            "server running/queue metrics are reported separately when exported.",
            "Queueing is scheduler behavior, not an OOM. OOM is reported only when an engine or "
            "benchmark process emits direct allocation-failure evidence.",
            "requests_per_concurrency below 5 is a boundary search, not a steady-state throughput "
            "benchmark; use at least 5 for the engine projects' recommended sustained load shape.",
            "Attached-server cache dtype and topology remain assertions unless the endpoint "
            "exposes "
            "them; managed launches preserve the exact sanitized command as evidence.",
        ],
    }
    if dry_run:
        base_report["planned_benchmarks"] = [
            _sanitize_command(
                (
                    _format_command(
                        config.benchmark_command,
                        _command_values(
                            config,
                            prediction,
                            served_model=model,
                            concurrency=point,
                            num_prompts=point * config.requests_per_concurrency,
                            result_file=Path(f"concurrency-{point}.json"),
                        ),
                    )
                    if config.benchmark_command
                    else _default_benchmark_command(
                        config,
                        prediction,
                        served_model=model,
                        concurrency=point,
                        num_prompts=point * config.requests_per_concurrency,
                        result_file=Path(f"concurrency-{point}.json"),
                        api_key=api_key,
                    )
                ),
                api_key,
            )
            for point in points
        ]
        return base_report

    process: subprocess.Popen[Any] | None = None
    server_log_handle: Any = None
    try:
        if server_command is not None:
            # This handle intentionally spans launch, readiness, and the full sweep.
            server_log_handle = tempfile.TemporaryFile(  # noqa: SIM115
                mode="w+", encoding="utf-8"
            )
            process = subprocess.Popen(
                server_command,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        endpoint_evidence = _wait_ready(config, process, api_key)
        base_report["engine"]["endpoint_evidence"] = endpoint_evidence
        served_model = _served_model(endpoint_evidence, model)
        base_report["workload"]["served_model"] = served_model
        with tempfile.TemporaryDirectory(prefix="kvfit-calibrate-") as raw_directory:
            directory = Path(raw_directory)
            for point in points:
                run = _run_one(
                    config,
                    prediction,
                    served_model=served_model,
                    concurrency=point,
                    api_key=api_key,
                    directory=directory,
                )
                base_report["runs"].append(run)
                if config.stop_on_failure and not run["success"]:
                    break
        base_report["comparison"] = _comparison(config, prediction, base_report["runs"])
        base_report["qualification"] = (
            "calibrated"
            if base_report["runs"] and all(run["success"] for run in base_report["runs"])
            else "partial"
            if any(run["success"] for run in base_report["runs"])
            else "failed"
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        base_report["qualification"] = "failed"
        base_report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None and (not config.keep_server or process.poll() is not None):
            _terminate(process)
            base_report["engine"]["managed_server_stopped"] = True
        elif process is not None:
            base_report["engine"]["managed_server_pid"] = process.pid
            base_report["engine"]["managed_server_stopped"] = False
        if server_log_handle is not None:
            server_log_handle.flush()
            server_log_handle.seek(0)
            log = server_log_handle.read()
            base_report["engine"]["server_log_tail"] = _tail(log, 200)
            base_report["engine"]["startup_evidence"] = _startup_evidence(log)
            base_report["engine"]["oom_observed_in_server"] = bool(OOM_RE.search(log))
            server_log_handle.close()
    if base_report["engine"].get("oom_observed_in_server") and base_report["runs"]:
        # The server log is the authoritative OOM surface when the client only
        # sees a disconnected request. Associate it with the final in-flight sweep.
        final_run = base_report["runs"][-1]
        final_run["oom_observed_in_server_log"] = True
        final_run["oom_observed"] = True
        final_run["success"] = False
        final_run["slo_qualified"] = False
        base_report["comparison"] = _comparison(config, prediction, base_report["runs"])
        base_report["qualification"] = (
            "partial" if any(run["success"] for run in base_report["runs"]) else "failed"
        )
    return base_report


def _human(report: dict[str, Any], output: Path | None) -> None:
    workload = report["workload"]
    print(
        f"Calibration: {report['qualification']} — {report['engine']['name']} at "
        f"{report['engine']['base_url']}"
    )
    print(
        f"Workload:    {workload['input_tokens']:,} input + {workload['output_tokens']:,} "
        f"output = {workload['context_tokens']:,} tokens"
    )
    comparison = report.get("comparison")
    if comparison:
        print(
            "Prediction:  "
            f"{comparison['predicted_memory_only_active_sequences']} active sequences; "
            f"{comparison['predicted_memory_only_users']} users"
        )
        print(
            "Observed:    "
            f"{comparison['max_successful_offered_concurrency']} max successful offered; "
            f"{comparison['max_observed_running_requests']} max measured running; "
            f"OOM at {comparison['first_actual_oom_concurrency']}"
        )
        if comparison["search_truncated_below_prediction"]:
            print("Boundary:    search cap is below the memory-only prediction")
    else:
        print(f"Planned:     concurrency {workload['concurrency_points']}")
    if output:
        print(f"Report:      {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvfit calibrate",
        description=(
            "Launch or attach to vLLM/SGLang on the target host, run exact-token "
            "concurrency sweeps, and compare observed evidence with kvfit's memory prediction."
        ),
    )
    parser.add_argument("config", help="deployment TOML containing [calibration]")
    parser.add_argument("--output", type=Path, help="write the full JSON evidence report")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve metadata and print commands without contacting or launching a server",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_calibration_config(args.config)
        report = run_calibration(config, dry_run=args.dry_run)
        output = args.output.expanduser().resolve() if args.output else None
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _human(report, output)
        return 0 if report["qualification"] in {"calibrated", "not-run"} else 4
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"kvfit calibrate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
