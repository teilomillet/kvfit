from __future__ import annotations

import json
import sys

from kvfit.cli import main, parse_tokens, resolve_cache_dtype
from kvfit.engines import EngineCheck, EngineStep
from kvfit.models import GIB, ModelMetadata
from kvfit.toml_config import load_toml_defaults


def test_parse_tokens_uses_binary_suffixes() -> None:
    assert parse_tokens("128k") == 128 * 1024
    assert parse_tokens("1m") == 1024**2


def test_list_hardware_json(capsys) -> None:
    assert main(["--list-hardware", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert any(row["id"] == "dgx-spark" and row["unified_memory"] for row in payload)
    assert any(row["id"] == "b200-180" and row["memory_gib"] == 180 for row in payload)


def test_list_systems_json(capsys) -> None:
    assert main(["--list-systems", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)

    assert any(row["id"] == "dgx-h100" and row["accelerators_per_system"] == 8 for row in payload)
    assert any(
        row["id"] == "dgx-gb200-nvl72" and row["scale_up_domain_accelerators"] == 72
        for row in payload
    )


def test_gpt_oss_mxfp4_weights_do_not_imply_four_bit_kv() -> None:
    config = {
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "mxfp4"},
    }

    dtype, source, warnings = resolve_cache_dtype(config, "auto")

    assert (dtype, source, warnings) == ("bf16", "model torch_dtype", ())


def test_gpt_oss_reference_defaults_kv_to_bf16() -> None:
    dtype, source, warnings = resolve_cache_dtype(
        {"model_type": "gpt_oss", "quantization_config": {"quant_method": "mxfp4"}},
        "auto",
    )

    assert dtype == "bf16"
    assert source == "OpenAI GPT-OSS reference implementation"
    assert warnings == ()


def test_explicit_fp8_kv_metadata_is_detected() -> None:
    config = {
        "model_type": "gpt_oss",
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "kv_cache_quant_config": {
                "*k_proj": {"output_tensors": {"dtype": "fp8_e4m3"}},
                "*v_proj": {"output_tensors": {"dtype": "fp8_e4m3"}},
            }
        },
    }

    dtype, source, warnings = resolve_cache_dtype(config, "auto")

    assert dtype == "fp8_e4m3"
    assert source == "quantization_config.kv_cache_quant_config"
    assert warnings == ()


def test_modelopt_structured_fp8_kv_scheme_is_detected() -> None:
    config = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "modelopt",
            "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        },
    }

    dtype, source, warnings = resolve_cache_dtype(config, "auto")

    assert dtype == "fp8"
    assert source == "quantization_config.kv_cache_scheme"
    assert warnings == ()


def test_bitsandbytes_compute_dtype_drives_kv_storage() -> None:
    config = {
        "model_type": "gpt_oss",
        "torch_dtype": "float16",
        "quantization_config": {
            "quant_method": "bitsandbytes",
            "bnb_4bit_compute_dtype": "bfloat16",
        },
    }

    dtype, source, warnings = resolve_cache_dtype(config, "auto")

    assert dtype == "bf16"
    assert source == "quantization_config.bnb_4bit_compute_dtype"
    assert warnings == ()


def test_context_beyond_config_max_fails_without_opt_in(monkeypatch, capsys) -> None:
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
            "max_position_embeddings": 4096,
        },
        weight_bytes=GIB,
        weight_source="test",
    )
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)

    assert main(["owner/model", "--hardware", "custom:8", "--context", "8k"]) == 2

    assert "--allow-context-overflow" in capsys.readouterr().err


def test_runtime_weight_floor_replaces_smaller_artifact(monkeypatch, capsys) -> None:
    metadata = ModelMetadata(
        repo_id="openai/gpt-oss-test",
        requested_revision="main",
        resolved_revision="abc123",
        config={
            "model_type": "gpt_oss",
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 64,
            "sliding_window": 128,
            "layer_types": ["sliding_attention", "full_attention"],
            "max_position_embeddings": 4096,
        },
        weight_bytes=GIB,
        weight_source="test artifact",
        runtime_weight_floor_bytes=2 * GIB,
        runtime_weight_floor_source="test runtime floor",
    )
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)

    assert (
        main(
            [
                "openai/gpt-oss-test",
                "--hardware",
                "custom:4",
                "--context",
                "128",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["weights"]["gib"] == 2
    assert payload["weights"]["source"] == "test runtime floor"
    assert any("raised the planning weight footprint" in warning for warning in payload["warnings"])


def test_engine_check_can_run_without_hardware(monkeypatch, capsys) -> None:
    metadata = ModelMetadata(
        repo_id="openai/gpt-oss-test",
        requested_revision="main",
        resolved_revision="abc123",
        config={
            "architectures": ["GptOssForCausalLM"],
            "model_type": "gpt_oss",
            "max_position_embeddings": 4096,
        },
        weight_bytes=GIB,
        weight_source="test",
        quantization="mxfp4",
    )
    steps = {
        name: EngineStep("pass", "test")
        for name in ("package", "architecture", "quantization", "platform", "config", "smoke")
    }
    check = EngineCheck(
        engine="vllm",
        python="/test/python",
        installed=True,
        version="1.2.3",
        mode="config",
        verification_level="config-only",
        load_tested=False,
        architectures=("GptOssForCausalLM",),
        matched_architecture="GptOssForCausalLM",
        quantization="mxfp4",
        resolved_quantization="mxfp4",
        platform={"device_type": "cuda", "device_name": "cuda", "enum": "cuda"},
        steps=steps,
        overall="preflight-pass",
        summary="test preflight passed",
    )
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr("kvfit.cli.check_engines", lambda *args, **kwargs: (check,))

    assert (
        main(
            [
                "openai/gpt-oss-test",
                "--check-engine",
                "vllm",
                "--context",
                "4096",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert "hardware" not in payload
    assert payload["engine_checks"][0]["overall"] == "preflight-pass"
    assert payload["engine_checks"][0]["load_tested"] is False


def _toml_test_metadata() -> ModelMetadata:
    return ModelMetadata(
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
        weight_source="test",
    )


def test_toml_positional_preset_runs_full_report(monkeypatch, tmp_path, capsys) -> None:
    preset = tmp_path / "inkling-lab.toml"
    preset.write_text(
        """
model = "owner/model"
hardware = "custom:4"
gpus = 2
context = "4k"
kv_dtype = "fp8"
utilization = 0.8
tp = 2

[output]
json = true
"""
    )
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: _toml_test_metadata()
    )

    assert main([str(preset)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["input_config"] == str(preset)
    assert payload["model"]["repo_id"] == "owner/model"
    assert payload["cache"]["context_tokens"] == 4096
    assert payload["precision"]["kv_dtype"] == "fp8"
    assert payload["hardware"]["gpus"] == 2
    assert [layout["tensor_parallel"] for layout in payload["layouts"]] == [2]


def test_toml_multi_dgx_reports_memory_only_concurrent_users(monkeypatch, tmp_path, capsys) -> None:
    preset = tmp_path / "multi-dgx.toml"
    preset.write_text(
        """
model = "owner/model"
system = "dgx-h100"
nodes = 2
context = 90000
active_sequences_per_user = 2

[output]
json = true
""".strip()
    )
    metadata = _toml_test_metadata()
    metadata.config["max_position_embeddings"] = 100_000
    monkeypatch.setattr("kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: metadata)

    assert main([str(preset)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["system"]["id"] == "dgx-h100"
    assert payload["system"]["systems"] == 2
    assert payload["hardware"]["gpus"] == 16
    assert payload["cache"]["context_tokens"] == 90_000
    assert payload["concurrency"]["qualification"] == "memory-only"
    assert payload["concurrency"]["active_sequences_per_user"] == 2
    assert payload["concurrency"]["best_scale_up_local"] is not None


def test_system_rejects_manual_gpu_count(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: _toml_test_metadata()
    )

    assert main(["owner/model", "--system", "dgx-h100", "--gpus", "8"]) == 2
    assert "use --nodes instead of --gpus" in capsys.readouterr().err


def test_cli_values_override_toml_defaults(monkeypatch, tmp_path, capsys) -> None:
    preset = tmp_path / "preset.toml"
    preset.write_text(
        """
model = "owner/from-file"
hardware = "custom:4"
context = "4k"

[output]
json = true
"""
    )
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: _toml_test_metadata()
    )

    assert main([str(preset), "owner/from-cli", "--context", "2k", "--gpus", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["cache"]["context_tokens"] == 2048
    assert payload["hardware"]["gpus"] == 2


def test_cli_can_disable_toml_boolean(monkeypatch, tmp_path, capsys) -> None:
    preset = tmp_path / "preset.toml"
    preset.write_text(
        """
model = "owner/model"
hardware = "custom:4"
context = "4k"

[output]
json = true
"""
    )
    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: _toml_test_metadata()
    )

    assert main([str(preset), "--no-json"]) == 0
    output = capsys.readouterr().out

    assert output.startswith("Preset:")
    assert "Model:        owner/model@abc123" in output


def test_toml_engine_python_is_relative_to_preset(monkeypatch, tmp_path) -> None:
    preset = tmp_path / "presets" / "engine.toml"
    preset.parent.mkdir()
    preset.write_text(
        """
model = "owner/model"

[engine]
check = ["vllm"]
python = "../engines/vllm/bin/python"
probe = "registry"
tp = 2
require_pass = true
"""
    )

    path, defaults = load_toml_defaults(preset)

    assert path == preset.resolve()
    assert defaults["check_engine"] == ["vllm"]
    assert defaults["engine_python"] == str((tmp_path / "engines/vllm/bin/python").resolve())


def test_toml_engine_python_preserves_virtualenv_symlink(tmp_path) -> None:
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    preset = tmp_path / "engine.toml"
    preset.write_text(
        """
model = "owner/model"

[engine]
check = "vllm"
python = "venv/bin/python"
""".strip()
    )

    _, defaults = load_toml_defaults(preset)

    assert defaults["engine_python"] == str(executable)


def test_toml_engine_settings_reach_probe_and_cli_check_overrides(
    monkeypatch, tmp_path, capsys
) -> None:
    preset = tmp_path / "engine.toml"
    preset.write_text(
        """
model = "owner/model"
context = "4k"

[engine]
check = ["vllm"]
python = "engine/bin/python"
probe = "registry"
tp = 2
timeout = 12

[output]
json = true
"""
    )
    captured: dict[str, object] = {}

    def fake_checks(engines, metadata, **kwargs):
        captured["engines"] = engines
        captured.update(kwargs)
        return ()

    monkeypatch.setattr(
        "kvfit.cli.fetch_model_metadata", lambda *args, **kwargs: _toml_test_metadata()
    )
    monkeypatch.setattr("kvfit.cli.check_engines", fake_checks)

    assert main([str(preset), "--check-engine", "sglang"]) == 0
    json.loads(capsys.readouterr().out)

    assert captured["engines"] == ("sglang",)
    assert captured["python"] == str((tmp_path / "engine/bin/python").resolve())
    assert captured["mode"] == "registry"
    assert captured["tensor_parallel"] == 2
    assert captured["timeout"] == 12


def test_toml_unknown_key_returns_specific_error(tmp_path, capsys) -> None:
    preset = tmp_path / "broken.toml"
    preset.write_text('modle = "owner/model"\n')

    assert main(["--config", str(preset)]) == 2

    error = capsys.readouterr().err
    assert "invalid TOML config" in error
    assert "unknown key(s): modle" in error


def test_toml_wrong_type_returns_specific_error(tmp_path, capsys) -> None:
    preset = tmp_path / "broken.toml"
    preset.write_text('model = "owner/model"\ngpus = "eight"\n')

    assert main([str(preset)]) == 2

    assert "gpus must be an integer" in capsys.readouterr().err


def test_toml_rejects_mixed_system_and_raw_hardware(tmp_path, capsys) -> None:
    preset = tmp_path / "broken.toml"
    preset.write_text('model = "owner/model"\nsystem = "dgx-h100"\nhardware = "h100"\n')

    assert main([str(preset)]) == 2

    assert "cannot contain both system and hardware" in capsys.readouterr().err
