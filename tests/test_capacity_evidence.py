from types import SimpleNamespace

import pytest

from kvfit.calibrate import _slo_result


def limits(**kwargs):
    return SimpleNamespace(max_ttft_ms=kwargs.get("ttft"), max_tpot_ms=None, max_e2e_ms=None)


def test_no_slo_is_unknown_not_passed():
    assert _slo_result(limits(), {"mean_ttft_ms": 1})["passed"] is False


def test_mean_does_not_establish_tail_latency():
    assert _slo_result(limits(ttft=20), {"mean_ttft_ms": 1})["passed"] is False


@pytest.mark.parametrize("metric", [-1, float("nan"), float("inf"), True])
def test_invalid_latency_is_not_evidence(metric):
    assert _slo_result(limits(ttft=20), {"p95_ttft_ms": metric})["passed"] is False


def test_report_names_the_usable_metric():
    result = _slo_result(limits(ttft=20), {"p95_ttft_ms": None, "p99_ttft_ms": 19})
    assert result["passed"] is True
    assert result["checks"]["ttft_ms"]["source"] == "p99_ttft_ms"


def test_physical_profile_requires_an_explicit_launch_configuration(tmp_path):
    from kvfit.calibrate import _default_server_command
    from kvfit.calibration_config import load_calibration_config

    path = tmp_path / "physical.toml"
    path.write_text(
        'model="test/model"\nhardware="h200"\ncontext=128\n'
        '[calibration]\nengine="sglang"\nmode="launch"\n'
    )
    with pytest.raises(ValueError, match="storage profile"):
        _default_server_command(
            load_calibration_config(path),
            {
                "cache_layout": "sglang-dsa-scaled",
                "model": {"repo_id": "test/model"},
                "cache": {"context_tokens": 128},
                "hardware": {"utilization": 0.8},
                "concurrency": {
                    "recommended": {
                        "layout": {"tensor_parallel": 1, "data_parallel": 1},
                    }
                },
            },
        )


@pytest.mark.parametrize("metric", ["nan", "inf", "-1"])
def test_calibration_rejects_invalid_slo_thresholds(tmp_path, metric):
    from kvfit.calibration_config import load_calibration_config

    path = tmp_path / "invalid.toml"
    path.write_text(
        'model="test/model"\nhardware="h200"\n'
        f'[calibration]\nengine="sglang"\nmax_ttft_ms={metric}\n'
    )
    with pytest.raises(ValueError):
        load_calibration_config(path)


@pytest.mark.parametrize("reported_output", [None, 1, True])
def test_calibration_requires_the_output_tokens_it_claims(monkeypatch, tmp_path, reported_output):
    import json
    import sys

    from kvfit.calibrate import _run_one
    from kvfit.calibration_config import load_calibration_config

    script = tmp_path / "benchmark.py"
    script.write_text(
        "import json,pathlib,sys\n"
        f"pathlib.Path(sys.argv[1]).write_text(json.dumps({{'completed': 1, "
        f"'total_input_tokens': 8, 'total_output_tokens': {reported_output!r}}}))\n"
    )
    path = tmp_path / "calibration.toml"
    path.write_text(
        'model="test/model"\nhardware="h100"\ncontext=10\n[calibration]\n'
        'engine="vllm"\nmode="attach"\noutput_tokens=2\nrequests_per_concurrency=1\n'
        f"benchmark_command=[{json.dumps(sys.executable)}, "
        f'{json.dumps(str(script))}, "{{result_file}}"]\n'
    )

    # The process reports successful exit and exact input, but incomplete output.
    class Sampler:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def stop(self):
            return {}

    monkeypatch.setattr("kvfit.calibrate.MetricsSampler", Sampler)
    prediction = {
        "model": {"repo_id": "test/model"},
        "cache": {"context_tokens": 10},
        "hardware": {"utilization": 0.8},
        "concurrency": {"recommended": {"layout": {"tensor_parallel": 1, "data_parallel": 1}}},
    }
    result = _run_one(
        load_calibration_config(path),
        prediction,
        served_model="test/model",
        concurrency=1,
        api_key=None,
        directory=tmp_path,
    )
    assert result["success"] is False
    assert result["exact_output_tokens_verified"] is False
