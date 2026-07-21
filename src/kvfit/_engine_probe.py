"""Standalone probe executed by the Python environment that owns an inference engine.

This file intentionally imports no other kvfit modules: ``--engine-python`` may point
at an environment where kvfit itself is not installed.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from contextlib import suppress
from importlib import metadata
from typing import Any, get_args, get_type_hints

MARKER = "KVFIT_ENGINE_CHECK_JSON="
VALID_ENGINES = {"vllm", "sglang"}
FULL_PRECISION_LABELS = {
    "bf16",
    "bfloat16",
    "f16",
    "float16",
    "fp16",
    "f32",
    "float32",
    "fp32",
    "none",
    "unquantized",
}
QUANTIZATION_ALIASES = {
    "compressed_tensors": ("compressed_tensors", "compressed-tensors"),
    "compressed-tensors": ("compressed-tensors", "compressed_tensors"),
    "bnb": ("bitsandbytes",),
    "bits_and_bytes": ("bitsandbytes",),
    "modelopt_nvfp4": ("modelopt_fp4", "modelopt"),
    "nvfp4": ("modelopt_fp4", "nvfp4_online", "petit_nvfp4"),
}
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
INDEX_DTYPE_ALIASES = {
    "bf16": "bf16",
    "fp8": "fp8",
    "fp4": "mxfp4",
    "mxfp4": "mxfp4",
    "nvfp4": "nvfp4",
}


def _step(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def _exception_detail(error: BaseException) -> str:
    detail = f"{type(error).__name__}: {error}".replace("\n", " ").strip()
    return detail[:1200]


def _distribution_version(engine: str) -> str | None:
    try:
        return metadata.version(engine)
    except metadata.PackageNotFoundError:
        return None


def _registry(engine: str) -> Any:
    if engine == "vllm":
        from vllm.model_executor.models.registry import ModelRegistry

        return ModelRegistry
    from sglang.srt.models.registry import ModelRegistry

    return ModelRegistry


def _platform(engine: str) -> Any:
    if engine == "vllm":
        from vllm.platforms import current_platform

        return current_platform
    from sglang.srt.platforms import current_platform

    return current_platform


def _platform_identity(platform: Any) -> tuple[str | None, str | None, str | None]:
    device_type = getattr(platform, "device_type", None)
    device_name = getattr(platform, "device_name", None)
    enum_value = getattr(platform, "_enum", None)
    enum_name = getattr(enum_value, "name", None)
    return (
        str(device_type) if device_type else None,
        str(device_name) if device_name else None,
        str(enum_name).lower() if enum_name else None,
    )


def _probe_platform(engine: str, result: dict[str, Any]) -> Any | None:
    try:
        platform = _platform(engine)
        device_type, device_name, enum_name = _platform_identity(platform)
        result["platform"] = {
            "device_type": device_type,
            "device_name": device_name,
            "enum": enum_name,
        }
        if enum_name in {None, "unspecified"} and device_name in {None, "unknown"}:
            result["steps"]["platform"] = _step(
                "unknown", "the installed engine did not detect an active backend"
            )
        elif not device_type:
            result["steps"]["platform"] = _step(
                "unknown", f"detected platform {device_name or enum_name!r} without a device type"
            )
        else:
            result["steps"]["platform"] = _step(
                "pass", f"detected {device_name or enum_name or device_type} ({device_type})"
            )
        return platform
    except Exception as error:  # Engine imports fail in partially installed environments.
        result["steps"]["platform"] = _step("unknown", _exception_detail(error))
        return None


def _probe_registry(
    engine: str,
    architectures: list[str],
    result: dict[str, Any],
) -> Any | None:
    try:
        registry = _registry(engine)
        supported = {str(value) for value in registry.get_supported_archs()}
        matched = [architecture for architecture in architectures if architecture in supported]
        result["matched_architecture"] = matched[0] if matched else None
        if not architectures:
            result["steps"]["architecture"] = _step(
                "unknown", "the Hugging Face config declares no architectures"
            )
        elif matched:
            result["steps"]["architecture"] = _step(
                "pass", f"native registry contains {matched[0]}"
            )
        elif "TransformersForCausalLM" in supported:
            result["steps"]["architecture"] = _step(
                "unknown",
                "no native architecture match; a Transformers fallback is registered but was "
                "not resolved yet",
            )
        else:
            result["steps"]["architecture"] = _step(
                "fail", f"installed registry does not contain {', '.join(architectures)}"
            )
        return registry
    except Exception as error:
        result["steps"]["architecture"] = _step("unknown", _exception_detail(error))
        return None


def _quantization_candidates(label: str) -> tuple[str, ...]:
    normalized = label.strip().lower()
    return QUANTIZATION_ALIASES.get(normalized, (normalized,))


def _probe_quantization(
    engine: str,
    quantization: str | None,
    platform: Any | None,
    result: dict[str, Any],
) -> None:
    if not quantization or quantization.strip().lower() in FULL_PRECISION_LABELS:
        result["steps"]["quantization"] = _step(
            "not-applicable", "checkpoint does not declare a weight-quantization method"
        )
        return
    normalized = quantization.strip().lower()
    if normalized.endswith("-bit") and normalized[:-4].isdigit():
        result["steps"]["quantization"] = _step(
            "unknown", f"{normalized} gives a bit width but not an engine format"
        )
        return
    getter = None
    try:
        if engine == "vllm":
            from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

            supported = {str(value).lower() for value in QUANTIZATION_METHODS}
        else:
            from sglang.srt.layers.quantization import (
                QUANTIZATION_METHODS,
                get_quantization_config,
            )

            values = (
                QUANTIZATION_METHODS.keys()
                if hasattr(QUANTIZATION_METHODS, "keys")
                else QUANTIZATION_METHODS
            )
            supported = {str(value).lower() for value in values}
            getter = get_quantization_config
    except Exception as error:
        result["steps"]["quantization"] = _step(
            "unknown", f"could not inspect the quantization registry: {_exception_detail(error)}"
        )
        return
    matched = next(
        (candidate for candidate in _quantization_candidates(normalized) if candidate in supported),
        None,
    )
    if matched is None:
        result["steps"]["quantization"] = _step(
            "fail", f"installed {engine} does not register {normalized}"
        )
        return
    try:
        if engine == "vllm" and platform is not None:
            verifier = getattr(platform, "verify_quantization", None)
            if callable(verifier):
                verifier(matched)
        elif getter is not None:
            getter(matched)
        result["resolved_quantization"] = matched
        result["steps"]["quantization"] = _step(
            "pass", f"installed {engine} registers {matched} on the detected platform"
        )
    except Exception as error:
        result["steps"]["quantization"] = _step(
            "fail", f"registered format was rejected on this platform: {_exception_detail(error)}"
        )


def _literal_strings(annotation: Any) -> set[str]:
    values: set[str] = set()
    for value in get_args(annotation):
        if isinstance(value, str):
            values.add(value)
        else:
            values.update(_literal_strings(value))
    return values


def _vllm_config_choices(config_class: Any, field_name: str) -> set[str]:
    for attribute in ("model_fields", "__pydantic_fields__", "__dataclass_fields__"):
        fields = getattr(config_class, attribute, None)
        if isinstance(fields, dict) and field_name in fields:
            field = fields[field_name]
            annotation = getattr(field, "annotation", None) or getattr(field, "type", None)
            choices = _literal_strings(annotation)
            if choices:
                return choices
    try:
        annotation = get_type_hints(config_class, include_extras=True).get(field_name)
    except Exception:
        annotation = None
    return _literal_strings(annotation)


def _sglang_kv_choices() -> set[str]:
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser(add_help=False)
    ServerArgs.add_cli_args(parser)
    for action in parser._actions:
        if "--kv-cache-dtype" in action.option_strings and action.choices is not None:
            return {str(value) for value in action.choices}
    return set()


def _probe_kv_cache_dtype(engine: str, requested: str | None, result: dict[str, Any]) -> None:
    if not requested:
        result["steps"]["kv_cache"] = _step("unknown", "no KV cache dtype was requested")
        return
    resolved = KV_DTYPE_ALIASES[engine].get(requested)
    if resolved is None:
        result["steps"]["kv_cache"] = _step(
            "fail", f"{engine} has no checked mapping for KV cache dtype {requested}"
        )
        return
    try:
        if engine == "vllm":
            from vllm.config import CacheConfig

            choices = _vllm_config_choices(CacheConfig, "cache_dtype")
            if not choices:
                # Construction is a stronger fallback for versions whose
                # generated config class does not preserve Literal metadata.
                CacheConfig(**_filtered_kwargs(CacheConfig, {"cache_dtype": resolved}))
        else:
            choices = _sglang_kv_choices()
        if choices and resolved not in choices:
            result["steps"]["kv_cache"] = _step(
                "fail",
                f"installed {engine} does not accept {resolved}; choices are {sorted(choices)}",
            )
            return
        result["resolved_kv_dtype"] = resolved
        mapping = f" (mapped from {requested})" if resolved != requested else ""
        result["steps"]["kv_cache"] = _step(
            "pass", f"installed {engine} accepts KV cache dtype {resolved}{mapping}"
        )
    except Exception as error:
        result["steps"]["kv_cache"] = _step(
            "unknown", f"could not inspect KV dtype support: {_exception_detail(error)}"
        )


def _is_deepseek_v4(payload: dict[str, Any]) -> bool:
    model_type = str(payload.get("model_type", "")).lower()
    architectures = [str(value).lower() for value in payload.get("architectures") or ()]
    return model_type == "deepseek_v4" or any("deepseekv4" in value for value in architectures)


def _class_has_field(config_class: Any, field_name: str) -> bool:
    if field_name in getattr(config_class, "__annotations__", {}):
        return True
    return any(
        isinstance(getattr(config_class, attribute, None), dict)
        and field_name in getattr(config_class, attribute)
        for attribute in ("model_fields", "__pydantic_fields__", "__dataclass_fields__")
    )


def _probe_index_cache_dtype(engine: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    requested = payload.get("index_dtype")
    if not requested:
        result["steps"]["index_cache"] = _step(
            "not-applicable", "no separate sparse-index cache dtype was requested"
        )
        return
    if not _is_deepseek_v4(payload):
        result["steps"]["index_cache"] = _step(
            "fail", "a separate index cache dtype is only modeled for DeepSeek V4"
        )
        return
    resolved = INDEX_DTYPE_ALIASES.get(str(requested))
    if resolved is None:
        result["steps"]["index_cache"] = _step(
            "fail", f"no checked DeepSeek V4 index-cache mapping exists for {requested}"
        )
        return
    try:
        if engine == "vllm":
            from vllm.config import AttentionConfig

            # DeepSeek V4 currently selects its FP8/MXFP4 indexer path with
            # use_fp4_indexer_cache. Prefer that architecture-specific switch
            # over the newer generic indexer_kv_dtype field when both exist.
            if _class_has_field(AttentionConfig, "use_fp4_indexer_cache"):
                if resolved not in {"fp8", "mxfp4"}:
                    raise ValueError(
                        f"installed vllm DeepSeek V4 indexer supports fp8 or mxfp4, not {resolved}"
                    )
            else:
                choices = _vllm_config_choices(AttentionConfig, "indexer_kv_dtype")
                if choices and resolved not in choices:
                    raise ValueError(
                        f"installed vllm choices are {sorted(choices)}, not {resolved}"
                    )
                if not choices:
                    result["steps"]["index_cache"] = _step(
                        "unknown", "installed vllm exposes no inspectable index-cache dtype option"
                    )
                    return
        else:
            from sglang.srt.server_args import ServerArgs

            if not _class_has_field(ServerArgs, "enable_deepseek_v4_fp4_indexer"):
                result["steps"]["index_cache"] = _step(
                    "unknown", "installed sglang exposes no inspectable DeepSeek V4 index option"
                )
                return
            if resolved not in {"fp8", "mxfp4"}:
                raise ValueError(
                    "installed sglang DeepSeek V4 indexer supports fp8 or experimental mxfp4, "
                    f"not {resolved}"
                )
        result["resolved_index_dtype"] = resolved
        result["steps"]["index_cache"] = _step(
            "pass", f"installed {engine} accepts DeepSeek V4 index cache dtype {resolved}"
        )
    except ValueError as error:
        result["steps"]["index_cache"] = _step("fail", str(error))
    except Exception as error:
        result["steps"]["index_cache"] = _step(
            "unknown", f"could not inspect index dtype support: {_exception_detail(error)}"
        )


def _filtered_kwargs(callable_object: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return candidates
    accepts_extra = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_extra:
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


def _call_registry_resolver(
    engine: str,
    registry: Any,
    architectures: list[str],
    model_config: Any,
) -> str | None:
    if engine == "vllm":
        resolver = getattr(registry, "inspect_model_cls", None)
        if resolver is None:
            resolver = registry.resolve_model_cls
        try:
            parameters = inspect.signature(resolver).parameters
        except (TypeError, ValueError):
            parameters = {"architectures": None, "model_config": None}
        resolved = (
            resolver(architectures, model_config)
            if "model_config" in parameters or len(parameters) >= 2
            else resolver(architectures)
        )
    else:
        resolved = registry.resolve_model_cls(architectures)
    if isinstance(resolved, tuple) and len(resolved) >= 2:
        return str(resolved[1])
    return None


def _probe_config(
    engine: str,
    payload: dict[str, Any],
    registry: Any | None,
    result: dict[str, Any],
) -> None:
    if payload.get("selected_artifact"):
        result["steps"]["config"] = _step(
            "unknown",
            "an exact weight artifact was selected; config-only engine probing cannot verify "
            "that local artifact",
        )
        return
    try:
        if engine == "vllm":
            from vllm.config import ModelConfig

            candidates = {
                "model": payload["repo_id"],
                "tokenizer": payload["repo_id"],
                "tokenizer_mode": "auto",
                "trust_remote_code": False,
                "dtype": "auto",
                "seed": 0,
                "revision": payload.get("revision"),
                "tokenizer_revision": payload.get("revision"),
                "max_model_len": payload.get("context_tokens"),
                "skip_tokenizer_init": True,
                "enforce_eager": True,
            }
        else:
            from sglang.srt.configs.model_config import ModelConfig

            candidates = {
                "model_path": payload["repo_id"],
                "trust_remote_code": False,
                "revision": payload.get("revision"),
                "context_length": payload.get("context_tokens"),
            }
        model_config = ModelConfig(**_filtered_kwargs(ModelConfig, candidates))
        configured_architectures = getattr(model_config, "architectures", None)
        if configured_architectures is None:
            hf_config = getattr(model_config, "hf_config", None)
            configured_architectures = getattr(hf_config, "architectures", None)
        architectures = [
            str(value) for value in (configured_architectures or payload.get("architectures") or ())
        ]
        if registry is None:
            registry = _registry(engine)
        resolved = _call_registry_resolver(engine, registry, architectures, model_config)
        if resolved:
            result["matched_architecture"] = resolved
        result["steps"]["architecture"] = _step(
            "pass", f"engine config resolved {resolved or architectures[0]}"
        )
        resolved_quantization = getattr(model_config, "quantization", None)
        if resolved_quantization:
            result["resolved_quantization"] = str(resolved_quantization)
            result["steps"]["quantization"] = _step(
                "pass", f"engine config accepted {resolved_quantization}"
            )
        elif result["steps"]["quantization"]["status"] != "pass":
            result["steps"]["quantization"] = _step(
                "not-applicable", "engine config resolved no weight quantization"
            )
        result["steps"]["config"] = _step(
            "pass", "engine parsed the checkpoint config and resolved a model implementation"
        )
    except Exception as error:
        result["steps"]["config"] = _step("fail", _exception_detail(error))


def _probe_smoke(engine: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    if payload.get("selected_artifact"):
        result["steps"]["smoke"] = _step(
            "unknown",
            "exact artifact smoke tests require a local engine-readable weight path",
        )
        return
    result["load_tested"] = True
    runtime = None
    try:
        if engine == "vllm":
            from vllm import LLM, SamplingParams

            candidates = {
                "model": payload["repo_id"],
                "revision": payload.get("revision"),
                "tokenizer_revision": payload.get("revision"),
                "trust_remote_code": False,
                "dtype": "auto",
                "tensor_parallel_size": payload.get("tensor_parallel", 1),
                "max_model_len": payload.get("context_tokens"),
                "gpu_memory_utilization": payload.get("utilization", 0.9),
                "enforce_eager": True,
                "disable_log_stats": True,
                "kv_cache_dtype": result.get("resolved_kv_dtype"),
            }
            if result.get("resolved_index_dtype"):
                from vllm.config import AttentionConfig

                if _class_has_field(AttentionConfig, "use_fp4_indexer_cache"):
                    candidates["attention_config"] = {
                        "use_fp4_indexer_cache": result["resolved_index_dtype"] == "mxfp4"
                    }
                elif _class_has_field(AttentionConfig, "indexer_kv_dtype"):
                    candidates["attention_config"] = {
                        "indexer_kv_dtype": result["resolved_index_dtype"]
                    }
            runtime = LLM(**_filtered_kwargs(LLM, candidates))
            sampling = SamplingParams(max_tokens=1, temperature=0)
            generate_kwargs = _filtered_kwargs(
                runtime.generate,
                {"prompts": ["Hello"], "sampling_params": sampling, "use_tqdm": False},
            )
            # Older vLLM versions use the first two arguments positionally even though the
            # current API names them. Keep a narrow fallback for that public API shape.
            if "prompts" in generate_kwargs:
                outputs = runtime.generate(**generate_kwargs)
            else:
                outputs = runtime.generate(["Hello"], sampling, use_tqdm=False)
        else:
            from sglang import Engine

            candidates = {
                "model_path": payload["repo_id"],
                "revision": payload.get("revision"),
                "trust_remote_code": False,
                "tp_size": payload.get("tensor_parallel", 1),
                "context_length": payload.get("context_tokens"),
                "mem_fraction_static": payload.get("utilization", 0.9),
                "disable_cuda_graph": True,
                "log_level": "error",
                "kv_cache_dtype": result.get("resolved_kv_dtype"),
            }
            if result.get("resolved_index_dtype") == "mxfp4":
                candidates["enable_deepseek_v4_fp4_indexer"] = True
            runtime = Engine(**_filtered_kwargs(Engine, candidates))
            outputs = runtime.generate(
                prompt="Hello",
                sampling_params={"temperature": 0, "max_new_tokens": 1},
            )
        if outputs is None or (hasattr(outputs, "__len__") and len(outputs) == 0):
            raise RuntimeError("engine returned no generation result")
        result["steps"]["smoke"] = _step(
            "pass", "loaded weights and completed a one-token generation"
        )
    except Exception as error:
        result["steps"]["smoke"] = _step("fail", _exception_detail(error))
    finally:
        if runtime is not None:
            shutdown = getattr(runtime, "shutdown", None)
            if callable(shutdown):
                with suppress(Exception):
                    shutdown()


def _finalize(result: dict[str, Any]) -> None:
    steps = result["steps"]
    if not result["installed"]:
        result["overall"] = "unavailable"
        result["summary"] = "package is not installed in the selected Python environment"
        return
    if any(
        steps[name]["status"] == "fail"
        for name in ("architecture", "quantization", "kv_cache", "index_cache")
    ):
        result["overall"] = "unsupported"
        result["summary"] = "the installed engine rejects a required model capability"
        return
    if result["mode"] == "config" and steps["config"]["status"] == "fail":
        result["overall"] = "error"
        result["summary"] = "the installed engine failed its config-level preflight"
        return
    if result["mode"] == "load":
        if steps["config"]["status"] == "fail":
            result["overall"] = "error"
            result["summary"] = "the installed engine failed before weight loading"
            return
        if steps["smoke"]["status"] == "fail":
            result["overall"] = "runtime-fail"
            result["summary"] = "the engine failed its weight-load or one-token smoke test"
            return
        if steps["smoke"]["status"] == "pass":
            result["overall"] = "smoke-pass"
            result["summary"] = "the engine loaded the model and generated one token"
            return
    required = [
        steps["architecture"]["status"],
        steps["platform"]["status"],
        steps["kv_cache"]["status"],
    ]
    quantization = steps["quantization"]["status"]
    config = steps["config"]["status"]
    quantization_passed = quantization in {"pass", "not-applicable"}
    config_passed = result["mode"] == "registry" or config == "pass"
    index_passed = steps["index_cache"]["status"] in {"pass", "not-applicable"}
    if (
        all(status == "pass" for status in required)
        and quantization_passed
        and index_passed
        and config_passed
    ):
        result["overall"] = "preflight-pass"
        result["summary"] = "installed engine passed the non-loading compatibility preflight"
        return
    result["overall"] = "unknown"
    result["summary"] = "the available evidence is insufficient for a support verdict"


def probe(payload: dict[str, Any]) -> dict[str, Any]:
    engine = str(payload.get("engine", "")).lower()
    if engine not in VALID_ENGINES:
        raise ValueError(f"unsupported engine {engine!r}")
    mode = str(payload.get("mode", "config"))
    if mode not in {"registry", "config", "load"}:
        raise ValueError(f"unsupported probe mode {mode!r}")
    architectures = [str(value) for value in payload.get("architectures") or ()]
    version = _distribution_version(engine)
    result: dict[str, Any] = {
        "engine": engine,
        "python": sys.executable,
        "installed": version is not None,
        "version": version,
        "mode": mode,
        "verification_level": {
            "registry": "registry-only",
            "config": "config-only",
            "load": "weight-load-and-one-token",
        }[mode],
        "load_tested": False,
        "architectures": architectures,
        "matched_architecture": None,
        "quantization": payload.get("quantization"),
        "resolved_quantization": None,
        "kv_dtype": payload.get("kv_dtype"),
        "resolved_kv_dtype": None,
        "index_dtype": payload.get("index_dtype"),
        "resolved_index_dtype": None,
        "platform": {"device_type": None, "device_name": None, "enum": None},
        "steps": {
            "package": _step(
                "pass" if version else "fail",
                f"installed version {version}" if version else "distribution not found",
            ),
            "architecture": _step("unknown", "not checked"),
            "quantization": _step("unknown", "not checked"),
            "kv_cache": _step("unknown", "not checked"),
            "index_cache": _step("unknown", "not checked"),
            "platform": _step("unknown", "not checked"),
            "config": (
                _step("unknown", "not checked")
                if mode in {"config", "load"}
                else _step("not-applicable", "registry mode does not parse engine config")
            ),
            "smoke": (
                _step("unknown", "not checked")
                if mode == "load"
                else _step("not-applicable", "weight loading was not requested")
            ),
        },
        "overall": "unknown",
        "summary": "probe did not finish",
    }
    if version is None:
        _finalize(result)
        return result
    registry = _probe_registry(engine, architectures, result)
    platform = _probe_platform(engine, result)
    _probe_quantization(engine, payload.get("quantization"), platform, result)
    _probe_kv_cache_dtype(engine, payload.get("kv_dtype"), result)
    _probe_index_cache_dtype(engine, payload, result)
    if mode in {"config", "load"}:
        _probe_config(engine, payload, registry, result)
    if (
        mode == "load"
        and result["steps"]["config"]["status"] == "pass"
        and result["steps"]["kv_cache"]["status"] == "pass"
        and result["steps"]["index_cache"]["status"] in {"pass", "not-applicable"}
    ):
        _probe_smoke(engine, payload, result)
    _finalize(result)
    return result


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("probe input must be a JSON object")
        result = probe(payload)
    except Exception as error:
        result = {
            "overall": "error",
            "summary": _exception_detail(error),
            "load_tested": False,
        }
    print(MARKER + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
