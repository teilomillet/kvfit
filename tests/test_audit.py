from __future__ import annotations

import sys

from kvfit.audit import load_audit_config, run_audit
from kvfit.models import GIB, ModelMetadata


def test_toml_audit_runs_complete_matrix_with_oracle(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "matrix.toml"
    config_path.write_text(
        """
version = 1
contexts = ["4k", "8k"]
kv_dtypes = ["auto"]
gpu_counts = [1, 2]
min_model_coverage = 1.0
min_hardware_coverage = 1.0

[[models]]
id = "test-llama"
family = "llama"
repo = "owner/model"
expected_architecture = "standard-gqa-mqa"

[[hardware]]
preset = "h100-80"
""".strip()
    )
    metadata = ModelMetadata(
        repo_id="owner/model",
        requested_revision="main",
        resolved_revision="abc123",
        config={
            "model_type": "llama",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
        weight_bytes=GIB,
        weight_source="test fixture",
    )
    monkeypatch.setattr("kvfit.audit.fetch_model_metadata", lambda *args, **kwargs: metadata)

    report = run_audit(load_audit_config(config_path))

    assert report["passed"] is True
    assert report["coverage"]["model_families"]["ratio"] == 1
    assert report["coverage"]["hardware_presets"]["ratio"] == 1
    assert report["coverage"]["decision_rows"] == 4
    assert report["invariants"]["passed"] is True
    assert all(row["oracle"]["passed"] for row in report["rows"])


def test_audit_toml_rejects_unknown_model_fields(tmp_path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        """
[[models]]
repo = "owner/model"
typo = true

[[hardware]]
preset = "h100"
""".strip()
    )

    try:
        load_audit_config(config_path)
    except ValueError as error:
        assert "unknown key(s): typo" in str(error)
    else:  # pragma: no cover - explicit failure keeps the error text visible
        raise AssertionError("unknown TOML field was accepted")


def test_audit_engine_python_preserves_virtualenv_symlink(tmp_path) -> None:
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    config_path = tmp_path / "matrix.toml"
    config_path.write_text(
        """
[[models]]
repo = "owner/model"

[[hardware]]
preset = "h100-80"

[[engines]]
name = "vllm"
python = "venv/bin/python"
""".strip()
    )

    config = load_audit_config(config_path)

    assert config.engines[0].python == str(executable)


def test_audit_toml_runs_single_and_multi_dgx_systems(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "dgx-matrix.toml"
    config_path.write_text(
        """
version = 1
contexts = ["4k"]
active_sequences_per_user = 2
min_model_coverage = 1.0
min_system_coverage = 1.0

[[models]]
id = "test-llama"
family = "llama"
repo = "owner/model"
expected_architecture = "standard-gqa-mqa"

[[systems]]
preset = "dgx-h100"
node_counts = [1, 2]
""".strip()
    )
    metadata = ModelMetadata(
        repo_id="owner/model",
        requested_revision="main",
        resolved_revision="abc123",
        config={
            "model_type": "llama",
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
        weight_bytes=GIB,
        weight_source="test fixture",
    )
    monkeypatch.setattr("kvfit.audit.fetch_model_metadata", lambda *args, **kwargs: metadata)

    report = run_audit(load_audit_config(config_path))

    assert report["passed"] is True
    assert report["coverage"]["system_presets"]["results"] == {"dgx-h100": True}
    assert report["coverage"]["decision_rows"] == 2
    assert {row["system"]["systems"] for row in report["rows"]} == {1, 2}
    assert all(row["deployment_id"] == "dgx-h100" for row in report["rows"])
    assert all(row["concurrency"]["active_sequences_per_user"] == 2 for row in report["rows"])
