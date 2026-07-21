from __future__ import annotations

import pytest

from kvfit.systems import parse_system


def test_current_dgx_system_shapes() -> None:
    assert parse_system("dgx-h100").accelerators_per_system == 8
    assert parse_system("dgx-h200").hardware_id == "h200-141"
    assert parse_system("dgx-b200").accelerators_per_system == 8
    assert parse_system("dgx-b300").hardware_id == "b300-288"
    assert parse_system("dgx-gb200-nvl72").scale_up_domain_accelerators == 72
    assert parse_system("dgx-spark").accelerators_per_system == 1


def test_system_alias_and_unknown_value() -> None:
    assert parse_system("H100").id == "dgx-h100"
    with pytest.raises(ValueError, match="unknown system"):
        parse_system("dgx-imaginary")
