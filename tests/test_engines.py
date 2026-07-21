from __future__ import annotations

import sys
from pathlib import Path

from kvfit.engines import check_engine
from kvfit.models import GIB, ModelMetadata


def _write(root: Path, relative: str, contents: str = "") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def _metadata(
    *,
    architecture: str = "GptOssForCausalLM",
    model_type: str = "gpt_oss",
    quantization: str | None = "mxfp4",
) -> ModelMetadata:
    return ModelMetadata(
        repo_id="openai/gpt-oss-test",
        requested_revision="main",
        resolved_revision="abc123",
        config={
            "architectures": [architecture],
            "model_type": model_type,
            "max_position_embeddings": 4096,
        },
        weight_bytes=GIB,
        weight_source="test",
        quantization=quantization,
    )


def _fake_vllm(root: Path, *, supported_architecture: str = "GptOssForCausalLM") -> None:
    for package in (
        "vllm/model_executor/__init__.py",
        "vllm/model_executor/models/__init__.py",
        "vllm/model_executor/layers/__init__.py",
    ):
        _write(root, package)
    _write(
        root,
        "vllm/__init__.py",
        """
class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class LLM:
    def __init__(self, **kwargs):
        if kwargs.get("kv_cache_dtype") != "bfloat16":
            raise ValueError("KV dtype was not propagated")
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params, use_tqdm=False):
        return [{"token_ids": [1]}]
""",
    )
    _write(
        root,
        "vllm/model_executor/models/registry.py",
        f"""
class Registry:
    def get_supported_archs(self):
        return {{{supported_architecture!r}, "TransformersForCausalLM"}}

    def inspect_model_cls(self, architectures, model_config):
        if {supported_architecture!r} not in architectures:
            raise ValueError("architecture unsupported")
        return object(), {supported_architecture!r}

ModelRegistry = Registry()
""",
    )
    _write(
        root,
        "vllm/model_executor/layers/quantization/__init__.py",
        'QUANTIZATION_METHODS = ["mxfp4", "bitsandbytes"]\n',
    )
    _write(
        root,
        "vllm/platforms/__init__.py",
        """
class Enum:
    name = "CUDA"

class Platform:
    _enum = Enum()
    device_type = "cuda"
    device_name = "cuda"

    def verify_quantization(self, quantization):
        if quantization != "mxfp4":
            raise ValueError("quantization unsupported")

current_platform = Platform()
""",
    )
    _write(
        root,
        "vllm/config.py",
        f"""
from typing import Literal

class HFConfig:
    architectures = [{supported_architecture!r}]

class ModelConfig:
    def __init__(self, model, revision=None, max_model_len=None, **kwargs):
        self.architectures = [{supported_architecture!r}]
        self.hf_config = HFConfig()
        self.quantization = "mxfp4"

class CacheConfig:
    cache_dtype: Literal["auto", "bfloat16", "float16", "fp8", "nvfp4"] = "auto"

    def __init__(self, cache_dtype="auto", **kwargs):
        if cache_dtype not in ("auto", "bfloat16", "float16", "fp8", "nvfp4"):
            raise ValueError("unsupported cache dtype")

class AttentionConfig:
    indexer_kv_dtype: Literal["bf16", "fp8", "mxfp4", "nvfp4"] = "bf16"
""",
    )
    _write(
        root,
        "vllm-9.9.0.dist-info/METADATA",
        "Metadata-Version: 2.1\nName: vllm\nVersion: 9.9.0\n",
    )


def _fake_sglang(root: Path) -> None:
    for package in (
        "sglang/srt/__init__.py",
        "sglang/srt/models/__init__.py",
        "sglang/srt/layers/__init__.py",
        "sglang/srt/configs/__init__.py",
    ):
        _write(root, package)
    _write(
        root,
        "sglang/__init__.py",
        """
class Engine:
    def __init__(self, **kwargs):
        if kwargs.get("kv_cache_dtype") != "bfloat16":
            raise ValueError("KV dtype was not propagated")
        self.kwargs = kwargs

    def generate(self, prompt=None, sampling_params=None):
        return {"text": "x"}

    def shutdown(self):
        pass
""",
    )
    _write(
        root,
        "sglang/srt/server_args.py",
        """
class ServerArgs:
    enable_deepseek_v4_fp4_indexer: bool = False

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument(
            "--kv-cache-dtype",
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
        )
""",
    )
    _write(
        root,
        "sglang/srt/models/registry.py",
        """
class Registry:
    def get_supported_archs(self):
        return {"GptOssForCausalLM", "TransformersForCausalLM"}

    def resolve_model_cls(self, architectures):
        if "GptOssForCausalLM" not in architectures:
            raise ValueError("architecture unsupported")
        return object(), "GptOssForCausalLM"

ModelRegistry = Registry()
""",
    )
    _write(
        root,
        "sglang/srt/layers/quantization/__init__.py",
        """
QUANTIZATION_METHODS = {"mxfp4": object}

def get_quantization_config(quantization):
    return QUANTIZATION_METHODS[quantization]
""",
    )
    _write(
        root,
        "sglang/srt/platforms/__init__.py",
        """
class Enum:
    name = "CUDA"

class Platform:
    _enum = Enum()
    device_type = "cuda"
    device_name = "cuda"

current_platform = Platform()
""",
    )
    _write(
        root,
        "sglang/srt/configs/model_config.py",
        """
class HFConfig:
    architectures = ["GptOssForCausalLM"]

class ModelConfig:
    def __init__(self, model_path, trust_remote_code=False, **kwargs):
        self.hf_config = HFConfig()
        self.quantization = "mxfp4"
""",
    )
    _write(
        root,
        "sglang-9.8.0.dist-info/METADATA",
        "Metadata-Version: 2.1\nName: sglang\nVersion: 9.8.0\n",
    )


def test_vllm_config_preflight_uses_target_environment(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(),
        python=sys.executable,
        mode="config",
        context_tokens=4096,
    )

    assert check.overall == "preflight-pass"
    assert check.version == "9.9.0"
    assert check.matched_architecture == "GptOssForCausalLM"
    assert check.resolved_quantization == "mxfp4"
    assert check.platform["device_type"] == "cuda"
    assert not check.load_tested


def test_sglang_config_preflight_uses_target_environment(monkeypatch, tmp_path) -> None:
    _fake_sglang(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "sglang",
        _metadata(),
        python=sys.executable,
        mode="config",
        context_tokens=4096,
    )

    assert check.overall == "preflight-pass"
    assert check.version == "9.8.0"
    assert check.steps["architecture"].status == "pass"
    assert check.steps["quantization"].status == "pass"
    assert check.steps["kv_cache"].status == "pass"
    assert check.steps["config"].status == "pass"


def test_vllm_load_probe_generates_one_token(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(),
        python=sys.executable,
        mode="load",
        context_tokens=4096,
        tensor_parallel=2,
    )

    assert check.overall == "smoke-pass"
    assert check.steps["smoke"].status == "pass"
    assert check.resolved_kv_dtype == "bfloat16"
    assert check.load_tested


def test_sglang_load_probe_generates_one_token(monkeypatch, tmp_path) -> None:
    _fake_sglang(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "sglang",
        _metadata(),
        python=sys.executable,
        mode="load",
        context_tokens=4096,
        tensor_parallel=2,
    )

    assert check.overall == "smoke-pass"
    assert check.steps["smoke"].status == "pass"
    assert check.resolved_kv_dtype == "bfloat16"
    assert check.load_tested


def test_registry_preflight_rejects_missing_native_architecture(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path, supported_architecture="LlamaForCausalLM")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(),
        python=sys.executable,
        mode="registry",
    )

    assert check.overall == "unknown"
    assert check.steps["architecture"].status == "unknown"
    assert "Transformers fallback" in check.steps["architecture"].detail


def test_broken_quantization_import_is_unknown_not_unsupported(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path)
    _write(
        tmp_path,
        "vllm/model_executor/layers/quantization/__init__.py",
        'raise ImportError("missing compiled extension")\n',
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(),
        python=sys.executable,
        mode="registry",
    )

    assert check.overall == "unknown"
    assert check.steps["quantization"].status == "unknown"
    assert "missing compiled extension" in check.steps["quantization"].detail


def test_missing_python_is_an_error() -> None:
    check = check_engine(
        "vllm",
        _metadata(),
        python="/definitely/missing/kvfit-python",
    )

    assert check.overall == "error"
    assert not check.installed
    assert "not found" in check.summary


def test_vllm_accepts_named_nvfp4_kv_dtype(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(),
        python=sys.executable,
        mode="registry",
        kv_dtype="nvfp4",
    )

    assert check.steps["kv_cache"].status == "pass"
    assert check.resolved_kv_dtype == "nvfp4"


def test_sglang_rejects_nvfp4_but_accepts_mxfp4_kv(monkeypatch, tmp_path) -> None:
    _fake_sglang(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    nvfp4 = check_engine(
        "sglang",
        _metadata(),
        python=sys.executable,
        mode="registry",
        kv_dtype="nvfp4",
    )
    mxfp4 = check_engine(
        "sglang",
        _metadata(),
        python=sys.executable,
        mode="registry",
        kv_dtype="mxfp4",
    )

    assert nvfp4.steps["kv_cache"].status == "fail"
    assert mxfp4.steps["kv_cache"].status == "pass"
    assert mxfp4.resolved_kv_dtype == "fp4_e2m1"


def test_vllm_checks_deepseek_v4_index_dtype_option(monkeypatch, tmp_path) -> None:
    _fake_vllm(tmp_path, supported_architecture="DeepseekV4ForCausalLM")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    check = check_engine(
        "vllm",
        _metadata(
            architecture="DeepseekV4ForCausalLM",
            model_type="deepseek_v4",
            quantization=None,
        ),
        python=sys.executable,
        mode="config",
        kv_dtype="fp8",
        index_dtype="mxfp4",
    )

    assert check.steps["kv_cache"].status == "pass"
    assert check.steps["index_cache"].status == "pass"
    assert check.resolved_index_dtype == "mxfp4"
