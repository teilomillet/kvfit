from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from kvfit.calibrate import (
    _concurrency_points,
    _default_benchmark_command,
    parse_prometheus,
    run_calibration,
)
from kvfit.calibration_config import load_calibration_config


def _prediction(*, nodes: int = 1, sequences: int = 3) -> dict:
    layout = {
        "tensor_parallel": 1,
        "data_parallel": nodes,
        "total_sequences": sequences,
        "sequences_per_replica": sequences // nodes if nodes else sequences,
        "cross_domain_tensor_parallel": False,
    }
    return {
        "model": {
            "repo_id": "openai/gpt-oss-20b",
            "resolved_revision": "abc123",
        },
        "cache": {"context_tokens": 90_000},
        "precision": {"kv_dtype": "bf16", "index_dtype": "bf16"},
        "hardware": {"utilization": 0.9, "gpus": nodes},
        "system": {"systems": nodes} if nodes else None,
        "concurrency": {
            "active_sequences_per_user": 1,
            "recommended": {
                "active_sequences": sequences,
                "concurrent_users": sequences,
                "layout": layout,
            },
        },
        "engine_checks": [],
        "warnings": [],
    }


def _write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_calibration_toml_is_strict_and_resolves_python(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "deployment.toml",
        """
model = "openai/gpt-oss-20b"
hardware = "h100-80"
context = 90000

[calibration]
engine = "vllm"
mode = "attach"
python = "env/bin/python"
concurrency = [4, 1, 2]
max_concurrency = 4
""",
    )
    config = load_calibration_config(config_path)
    assert config.python == str(tmp_path / "env/bin/python")
    assert config.concurrency == (1, 2, 4)
    assert config.output_tokens == 1

    bad = _write_config(
        tmp_path / "bad.toml",
        config_path.read_text(encoding="utf-8").replace(
            'mode = "attach"', 'mode = "attach"\nmade_up = true'
        ),
    )
    with pytest.raises(ValueError, match="unknown key"):
        load_calibration_config(bad)


def test_prometheus_parser_understands_old_new_vllm_and_sglang_names() -> None:
    metrics = parse_prometheus(
        """
# HELP ignored ignored
vllm:num_requests_running{model_name="a"} 2
vllm:gpu_cache_usage_perc{model_name="a"} 0.75
sglang:num_queue_reqs 3
sglang:token_usage 0.8
"""
    )
    assert metrics["vllm:num_requests_running"] == [2.0]
    assert metrics["vllm:gpu_cache_usage_perc"] == [0.75]
    assert metrics["sglang:num_queue_reqs"] == [3.0]
    assert metrics["sglang:token_usage"] == [0.8]


def test_default_sweep_includes_prediction_or_reports_cap(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "deployment.toml",
        """
model = "openai/gpt-oss-20b"
hardware = "h100-80"
context = 90000

[calibration]
engine = "vllm"
mode = "attach"
max_concurrency = 8
""",
    )
    config = load_calibration_config(config_path)
    assert _concurrency_points(config, _prediction(sequences=7)) == [1, 2, 4, 7]
    assert _concurrency_points(config, _prediction(sequences=17)) == [1, 2, 4, 8]


def test_official_vllm_command_uses_exact_random_lengths(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "deployment.toml",
        f"""
model = "openai/gpt-oss-20b"
hardware = "h100-80"
context = 90000

[calibration]
engine = "vllm"
mode = "attach"
python = {json.dumps(sys.executable)}
output_tokens = 8
""",
    )
    config = load_calibration_config(config_path)
    command = _default_benchmark_command(
        config,
        _prediction(),
        served_model="served",
        concurrency=2,
        num_prompts=10,
        result_file=tmp_path / "result.json",
        api_key=None,
    )
    assert command[command.index("--random-input-len") + 1] == "89992"
    assert command[command.index("--random-output-len") + 1] == "8"
    assert command[command.index("--random-range-ratio") + 1] == "0"
    assert command[command.index("--max-concurrency") + 1] == "2"
    assert command[command.index("--num-prompts") + 1] == "10"


class _FakeEngineHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "served-gpt-oss"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/version":
            body = json.dumps({"version": "test"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = (
                b"vllm:num_requests_running 2\n"
                b"vllm:num_requests_waiting 0\n"
                b"vllm:kv_cache_usage_perc 0.5\n"
                b"vllm:prompt_tokens_total 100\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def test_attach_calibration_runs_live_sweep_and_verifies_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEngineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    benchmark = tmp_path / "fake_benchmark.py"
    benchmark.write_text(
        """
import json, pathlib, sys, time
result, input_tokens, output_tokens, concurrency, num_prompts = sys.argv[1:]
time.sleep(0.08)
n = int(num_prompts)
pathlib.Path(result).write_text(json.dumps({
    "completed": n,
    "total_input_tokens": n * int(input_tokens),
    "total_output_tokens": n * int(output_tokens),
    "request_throughput": float(concurrency),
    "mean_ttft_ms": 10.0,
    "p95_ttft_ms": 12.0,
    "mean_tpot_ms": 2.0,
    "p95_tpot_ms": 3.0,
    "mean_e2el_ms": 15.0,
}))
""",
        encoding="utf-8",
    )
    port = server.server_address[1]
    config_path = _write_config(
        tmp_path / "deployment.toml",
        f"""
model = "openai/gpt-oss-20b"
hardware = "h100-80"
context = 90000

[calibration]
engine = "vllm"
mode = "attach"
base_url = "http://127.0.0.1:{port}"
python = {json.dumps(sys.executable)}
benchmark_command = [
  "{{python}}", {json.dumps(str(benchmark))}, "{{result_file}}",
  "{{input_tokens}}", "{{output_tokens}}", "{{concurrency}}", "{{num_prompts}}"
]
concurrency = [1, 2]
max_concurrency = 2
requests_per_concurrency = 1
metrics_interval = 0.01
max_ttft_ms = 20
""",
    )
    monkeypatch.setattr("kvfit.calibrate._prediction", lambda _config: _prediction())
    try:
        report = run_calibration(load_calibration_config(config_path))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert report["qualification"] == "calibrated"
    assert [run["success"] for run in report["runs"]] == [True, True]
    assert all(run["exact_input_tokens_verified"] for run in report["runs"])
    assert report["comparison"]["max_successful_offered_concurrency"] == 2
    assert report["comparison"]["max_observed_running_requests"] == 2
    assert report["comparison"]["first_actual_oom_concurrency"] is None
    assert report["engine"]["server_configuration_verification"] == (
        "attached-endpoint-reported-only"
    )


@pytest.mark.parametrize("token_evidence", ["missing", "mismatched"])
def test_attach_calibration_requires_exact_input_token_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token_evidence: str,
) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEngineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    benchmark = tmp_path / "fake_benchmark.py"
    benchmark.write_text(
        """
import json, pathlib, sys
result, input_tokens, _output_tokens, _concurrency, num_prompts, evidence = sys.argv[1:]
n = int(num_prompts)
payload = {"completed": n, "request_throughput": 1.0}
if evidence == "mismatched":
    payload["total_input_tokens"] = n * int(input_tokens) - 1
pathlib.Path(result).write_text(json.dumps(payload))
""",
        encoding="utf-8",
    )
    port = server.server_address[1]
    config_path = _write_config(
        tmp_path / "deployment.toml",
        f"""
model = "openai/gpt-oss-20b"
hardware = "h100-80"
context = 90000

[calibration]
engine = "vllm"
mode = "attach"
base_url = "http://127.0.0.1:{port}"
python = {json.dumps(sys.executable)}
benchmark_command = [
  "{{python}}", {json.dumps(str(benchmark))}, "{{result_file}}",
  "{{input_tokens}}", "{{output_tokens}}", "{{concurrency}}", "{{num_prompts}}",
  {json.dumps(token_evidence)}
]
concurrency = [1, 2]
max_concurrency = 2
requests_per_concurrency = 1
metrics_interval = 0.01
""",
    )
    monkeypatch.setattr("kvfit.calibrate._prediction", lambda _config: _prediction())
    try:
        report = run_calibration(load_calibration_config(config_path))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert report["qualification"] == "failed"
    assert len(report["runs"]) == 1
    assert report["runs"][0]["completed_all"] is True
    assert report["runs"][0]["exact_input_tokens_verified"] is False
    assert report["runs"][0]["success"] is False
    assert report["comparison"]["max_successful_offered_concurrency"] is None
    assert report["comparison"]["first_failed_concurrency"] == 1


def test_multi_node_default_launch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "deployment.toml",
        """
model = "openai/gpt-oss-20b"
system = "dgx-spark"
nodes = 2
context = 90000
tp = 1

[calibration]
engine = "vllm"
mode = "launch"
allow_cpu = true
""",
    )
    monkeypatch.setattr(
        "kvfit.calibrate._prediction", lambda _config: _prediction(nodes=2, sequences=4)
    )
    with pytest.raises(ValueError, match="multi-system deployment"):
        run_calibration(load_calibration_config(config_path), dry_run=True)
