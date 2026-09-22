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
