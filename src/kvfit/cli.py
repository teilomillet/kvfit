from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from kvfit import __version__
from kvfit.architectures import estimate_cache
from kvfit.engines import EngineCheck, EngineName, check_engines
from kvfit.hardware import HARDWARE, parse_hardware
from kvfit.hf import HuggingFaceError, fetch_model_metadata
from kvfit.models import GIB, UnsupportedArchitecture
from kvfit.planner import TopologyEstimate, plan_topologies, summarize_concurrency
from kvfit.systems import SYSTEMS, System, parse_system
from kvfit.toml_config import load_toml_defaults, parse_token_count, toml_path_from_argv

DTYPE_BYTES = {
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "mxfp4": 0.5,
    "nvfp4": 0.5,
    "int4": 0.5,
}


def _normalize_cache_dtype(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("torch.", "")
    if normalized in {"bfloat16", "bf16"}:
        return "bf16"
    if normalized in {"float16", "fp16", "half"}:
        return "fp16"
    if "e4m3" in normalized:
        return "fp8_e4m3"
    if "e5m2" in normalized:
        return "fp8_e5m2"
    if "fp8" in normalized or "float8" in normalized:
        return "fp8"
    if normalized in {"int8", "i8"}:
        return "int8"
    if "nvfp4" in normalized:
        return "nvfp4"
    if "mxfp4" in normalized or "fp4_e2m1" in normalized:
        return "mxfp4"
    if "fp4" in normalized:
        return "fp4"
    if normalized in {"int4", "i4"}:
        return "int4"
    return None


def _output_cache_dtypes(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        output = value.get("output_tensors")
        if isinstance(output, Mapping):
            normalized = _normalize_cache_dtype(output.get("dtype"))
            if normalized:
                found.add(normalized)
        for nested in value.values():
            found.update(_output_cache_dtypes(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            found.update(_output_cache_dtypes(nested))
    return found


def _structured_cache_dtype(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    bits = value.get("num_bits")
    kind = value.get("type")
    if isinstance(bits, bool) or not isinstance(bits, int):
        return None
    normalized_kind = str(kind).lower() if kind is not None else "float"
    if bits == 8 and normalized_kind in {"float", "fp", "floating"}:
        return "fp8"
    if bits == 8 and normalized_kind in {"int", "integer"}:
        return "int8"
    if bits == 4 and normalized_kind in {"float", "fp", "floating"}:
        return "fp4"
    if bits == 4 and normalized_kind in {"int", "integer"}:
        return "int4"
    return None


def resolve_cache_dtype(
    config: Mapping[str, Any],
    requested: str,
    external_quantization_config: Mapping[str, Any] | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    if requested != "auto":
        return requested, "user override", ()
    nested = config.get("text_config")
    source = nested if isinstance(nested, Mapping) else config
    direct = _normalize_cache_dtype(source.get("kv_cache_dtype"))
    if direct:
        return direct, "config kv_cache_dtype", ()
    if external_quantization_config:
        external = external_quantization_config.get("quantization")
        if isinstance(external, Mapping):
            direct = _normalize_cache_dtype(external.get("kv_cache_quant_algo"))
            if direct:
                return direct, "hf_quant_config.json kv_cache_quant_algo", ()
    quantization = source.get("quantization_config", config.get("quantization_config"))
    if isinstance(quantization, Mapping):
        for key in ("kv_cache_dtype", "kv_cache_quant_method", "kv_cache_scheme"):
            direct = _normalize_cache_dtype(quantization.get(key))
            if direct:
                return direct, f"quantization_config.{key}", ()
            structured = _structured_cache_dtype(quantization.get(key))
            if structured:
                return structured, f"quantization_config.{key}", ()
        kv_config = quantization.get("kv_cache_quant_config")
        discovered = _output_cache_dtypes(kv_config)
        if len(discovered) == 1:
            return discovered.pop(), "quantization_config.kv_cache_quant_config", ()
        if len(discovered) > 1:
            raise ValueError(
                f"conflicting KV cache dtypes in quantization_config: {sorted(discovered)!r}"
            )
        for key in ("bnb_4bit_compute_dtype", "compute_dtype", "activation_dtype"):
            compute_dtype = _normalize_cache_dtype(quantization.get(key))
            if compute_dtype:
                return compute_dtype, f"quantization_config.{key}", ()
    torch_dtype = _normalize_cache_dtype(source.get("torch_dtype", source.get("dtype")))
    if torch_dtype:
        return torch_dtype, "model torch_dtype", ()
    model_type = str(source.get("model_type", config.get("model_type", ""))).lower()
    if model_type == "gpt_oss":
        return "bf16", "OpenAI GPT-OSS reference implementation", ()
    return (
        "bf16",
        "conservative default",
        ("KV dtype was not declared by the checkpoint; defaulted to bf16",),
    )


def parse_tokens(value: str) -> int:
    try:
        return parse_token_count(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_tp(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("TP must be 'auto' or a positive integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("TP must be positive")
    return result


def _parser(defaults: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvfit",
        description=(
            "Estimate per-sequence KV state and full-context memory capacity from "
            "Hugging Face metadata. Unknown stateful architectures fail closed."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="HF owner/model, model or weight-file URL, or owner/model@revision",
    )
    parser.add_argument(
        "--config",
        help="TOML preset; command-line values override fields from the file",
    )
    parser.add_argument("--hardware", help="hardware preset, alias, or custom:<GiB>")
    parser.add_argument(
        "--system",
        help="DGX system preset; use with --nodes instead of --hardware/--gpus",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="accelerator count (default: 1; derived automatically for --system)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="number of complete --system units (default: 1)",
    )
    parser.add_argument(
        "--context",
        type=parse_tokens,
        default=128 * 1024,
        help="tokens per active sequence; k/m use powers of 1024 (default: 128k)",
    )
    parser.add_argument(
        "--kv-dtype",
        choices=["auto", *sorted(DTYPE_BYTES)],
        default="auto",
        help="cache dtype; auto reads explicit KV metadata then model dtype (default: auto)",
    )
    parser.add_argument(
        "--index-dtype",
        choices=sorted(DTYPE_BYTES),
        help="separate sparse-index cache dtype (default: same as --kv-dtype)",
    )
    parser.add_argument(
        "--utilization",
        type=float,
        default=0.90,
        help="fraction of each device treated as the model-executor budget (default: 0.90)",
    )
    parser.add_argument(
        "--tp",
        type=parse_tp,
        default=None,
        help="tensor-parallel size or auto to compare every divisor (default: auto)",
    )
    parser.add_argument(
        "--active-sequences-per-user",
        type=int,
        default=1,
        help=(
            "simultaneously resident sequences consumed by one user; agent branches may need "
            "more than one (default: 1)"
        ),
    )
    parser.add_argument(
        "--weight-gib",
        type=float,
        help="override repository artifact size with an expected runtime weight footprint",
    )
    parser.add_argument(
        "--revision",
        help="HF branch, tag, or commit (default: URL revision or main)",
    )
    parser.add_argument(
        "--allow-context-overflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="size a hypothetical context beyond the checkpoint's declared maximum",
    )
    parser.add_argument(
        "--check-engine",
        action="append",
        choices=["vllm", "sglang", "all"],
        help=(
            "probe an installed inference engine; repeat for both, or use all "
            "(hardware is optional when this is set)"
        ),
    )
    parser.add_argument(
        "--engine-python",
        default=sys.executable,
        help="Python executable containing vLLM/SGLang (default: current interpreter)",
    )
    parser.add_argument(
        "--engine-probe",
        choices=["registry", "config", "load"],
        default="config",
        help=(
            "registry checks installed capability tables; config also asks the engine to parse "
            "the checkpoint config; load downloads/allocates weights and generates one token "
            "(default: config)"
        ),
    )
    parser.add_argument(
        "--engine-tp",
        type=int,
        help="tensor-parallel size for --engine-probe load (default: --tp, then --gpus)",
    )
    parser.add_argument(
        "--engine-timeout",
        type=float,
        default=60.0,
        help="timeout per installed-engine probe in seconds (default: 60)",
    )
    parser.add_argument(
        "--require-engine-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exit 3 unless every requested engine returns a passing verdict",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HF request timeout in seconds")
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit machine-readable JSON",
    )
    parser.add_argument(
        "--list-hardware",
        action="store_true",
        help="list hardware presets and exit",
    )
    parser.add_argument(
        "--list-systems",
        action="store_true",
        help="list DGX system presets and exit",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def _max_context(config: dict[str, Any]) -> int | None:
    nested = config.get("text_config")
    source = nested if isinstance(nested, dict) else config
    value = source.get("max_position_embeddings", source.get("model_max_length"))
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _list_hardware(*, as_json: bool) -> None:
    rows = [hardware.as_dict() for hardware in HARDWARE.values()]
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        kind = " unified" if row["unified_memory"] else ""
        print(f"{row['id']:<20} {row['memory_gib']:>6g} GiB{kind}  {row['label']}")
        if row["note"]:
            print(f"{'':28}{row['note']}")


def _list_systems(*, as_json: bool) -> None:
    rows = [system.as_dict() for system in SYSTEMS.values()]
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    for row in rows:
        print(
            f"{row['id']:<20} {row['accelerators_per_system']:>3} accelerators/"
            f"{row['unit_name']:<6}  {row['label']}"
        )
        print(f"{'':24}scale-up: {row['scale_up_fabric']}")
        print(f"{'':24}scale-out: {row['scale_out_fabric']}")


def _fmt_gib(value: float) -> str:
    if value == float("inf"):
        return "—"
    return f"{value / GIB:.2f}"


def _selected_engines(values: Sequence[str] | None) -> tuple[EngineName, ...]:
    selected: list[EngineName] = []
    for value in values or ():
        if value == "all":
            candidates: tuple[EngineName, ...] = ("vllm", "sglang")
        elif value == "vllm":
            candidates = ("vllm",)
        else:
            candidates = ("sglang",)
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
    return tuple(selected)


def _print_engine_checks(checks: Sequence[EngineCheck]) -> None:
    if not checks:
        return
    loading = any(check.mode == "load" for check in checks)
    heading = (
        "Installed-engine smoke test (weights loaded):"
        if loading
        else "Installed-engine preflight (no weights loaded):"
    )
    print(f"\n{heading}")
    for check in checks:
        version = check.version or "not installed"
        print(f"  {check.engine:<7} {version:<16} {check.overall}: {check.summary}")
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
            step = check.steps[name]
            print(f"    {name:<14} {step.status:<14} {step.detail}")
        if check.diagnostics and check.overall in {"error", "unknown"}:
            print(f"    diagnostics    {check.diagnostics[-1]}")
    if loading:
        print(
            "  Boundary: smoke-pass proves one launch/request, not production throughput or SLOs."
        )
    else:
        print(
            "  Boundary: preflight-pass is config compatibility, not a weight-load or token test."
        )


def _human_engine_only(
    *,
    metadata: Any,
    context_tokens: int,
    checks: Sequence[EngineCheck],
    warnings: Sequence[str],
    kv_dtype: str,
    kv_dtype_source: str,
    index_dtype: str | None,
) -> None:
    revision = metadata.resolved_revision or metadata.requested_revision
    print(f"Model:        {metadata.repo_id}@{revision}")
    architectures = checks[0].architectures if checks else ()
    print(f"Architecture: {', '.join(architectures) if architectures else 'not declared'}")
    print(f"Context:      {context_tokens:,} tokens")
    index_detail = f"; index dtype: {index_dtype}" if index_dtype else ""
    print(f"KV dtype:     {kv_dtype} ({kv_dtype_source}){index_detail}")
    print(f"Weights:      {metadata.quantization or 'unquantized/unspecified'}")
    _print_engine_checks(checks)
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


def _print_layouts(layouts: Sequence[TopologyEstimate]) -> None:
    print("\nMemory-capacity layouts (DP means full model replicas; EP is disabled):")
    print("  TP  DP  weights/rank  KV/sequence/rank  seq/replica  total seq  fabric    verdict")
    for layout in layouts:
        if layout.scale_up_domain_accelerators is None:
            fabric = "—"
        elif layout.cross_domain_tensor_parallel:
            fabric = f"cross×{layout.tensor_parallel_domains}"
        else:
            fabric = "local"
        print(
            f"  {layout.tensor_parallel:>2}  {layout.data_parallel:>2}  "
            f"{_fmt_gib(layout.weights_per_rank_bytes):>9} GiB  "
            f"{_fmt_gib(layout.kv_per_sequence_per_rank_bytes):>14} GiB  "
            f"{layout.sequences_per_replica:>11}  {layout.total_sequences:>9}  "
            f"{fabric:<8}  {layout.verdict}"
        )
        if layout.verdict != "fits":
            print(f"      {layout.reason}")


def _human_output(
    *,
    metadata: Any,
    cache: Any,
    hardware: Any,
    gpus: int,
    system: System | None,
    nodes: int,
    utilization: float,
    weight_bytes: int,
    weight_source: str,
    layouts: Sequence[TopologyEstimate],
    warnings: Sequence[str],
    kv_dtype: str,
    kv_dtype_source: str,
    index_dtype: str,
    engine_checks: Sequence[EngineCheck],
    concurrency: Mapping[str, Any],
) -> None:
    revision = metadata.resolved_revision or metadata.requested_revision
    print(f"Model:        {metadata.repo_id}@{revision}")
    print(f"Architecture: {cache.architecture} ({cache.confidence})")
    print(f"Context:      {cache.context_tokens:,} tokens")
    print(f"KV dtype:     {kv_dtype} ({kv_dtype_source}); index dtype: {index_dtype}")
    print(f"KV/sequence:  {cache.total_gib:.2f} GiB logical state")
    for component in cache.components:
        print(f"  {component.name:<24} {component.bytes / GIB:>8.3f} GiB")
    print(f"Weights:      {weight_bytes / GIB:.2f} GiB ({weight_source})")
    if system:
        unit = system.unit_name if nodes == 1 else f"{system.unit_name}s"
        print(
            f"System:       {nodes}× {system.label} {unit}; {gpus}× {hardware.label}; "
            f"{utilization:.0%} ({hardware.memory_gib * utilization:.2f} GiB/rank) budgeted"
        )
        print(f"Scale-up:     {system.scale_up_fabric}")
        if nodes > 1:
            print(f"Scale-out:    {system.scale_out_fabric}")
    else:
        print(
            f"Hardware:     {gpus}× {hardware.label}; "
            f"{utilization:.0%} ({hardware.memory_gib * utilization:.2f} GiB/rank) budgeted"
        )
    _print_layouts(layouts)
    recommended = concurrency["recommended"]
    if recommended:
        layout = recommended["layout"]
        print(
            "\nStatic counted-state upper bound: "
            f"{recommended['concurrent_users']} users from "
            f"{recommended['active_sequences']} full-context sequences "
            f"({concurrency['active_sequences_per_user']} active sequence(s)/user), "
            f"TP={layout['tensor_parallel']}, DP={layout['data_parallel']}"
        )
        if layout["cross_domain_tensor_parallel"]:
            print("  Warning: the selected TP layout crosses scale-up domains.")
    else:
        print("\nStatic counted-state upper bound: 0 (no full-context layout fits)")
    best_any = concurrency["best_any_fabric"]
    if best_any and recommended and best_any != recommended:
        layout = best_any["layout"]
        print(
            "Cross-domain arithmetic ceiling: "
            f"{best_any['concurrent_users']} users, TP={layout['tensor_parallel']} across "
            f"{layout['tensor_parallel_domains']} scale-up domains (not runtime-qualified)"
        )
    _print_engine_checks(engine_checks)
    print("\nAssumptions:")
    print("  - Weight shards are ideally balanced across TP ranks.")
    print("  - Each cache component shards only across its own available parallel units.")
    print("  - DP entries are independent full replicas; no expert/pipeline/context parallelism.")
    print("  - This excludes unmeasured runtime overhead and is not an OOM guarantee or SLO.")
    for note in cache.notes:
        print(f"  - {note}")
    if hardware.note:
        print(f"  - {hardware.note}")
    if system:
        print(f"  - {system.note}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"\nFormula reference: {cache.reference}")


def run(args: argparse.Namespace) -> int:
    if args.list_hardware:
        _list_hardware(as_json=args.json)
        return 0
    if args.list_systems:
        _list_systems(as_json=args.json)
        return 0
    if not args.model:
        raise ValueError("MODEL is required unless a --list-* option is used")
    selected_engines = _selected_engines(args.check_engine)
    if not args.hardware and not args.system and not selected_engines:
        raise ValueError("--hardware, --system, or --check-engine is required")
    if args.require_engine_pass and not selected_engines:
        raise ValueError("--require-engine-pass requires --check-engine")
    if args.gpus is not None and args.gpus < 1:
        raise ValueError("--gpus must be positive")
    if args.nodes < 1:
        raise ValueError("--nodes must be positive")
    if args.active_sequences_per_user < 1:
        raise ValueError("--active-sequences-per-user must be positive")
    if args.system and args.hardware:
        raise ValueError("--system and --hardware are mutually exclusive")
    if args.system and args.gpus is not None:
        raise ValueError("--system derives accelerator count; use --nodes instead of --gpus")
    if not args.system and args.nodes != 1:
        raise ValueError("--nodes requires --system")
    if not 0 < args.utilization <= 1:
        raise ValueError("--utilization must be greater than 0 and at most 1")
    if args.weight_gib is not None and args.weight_gib <= 0:
        raise ValueError("--weight-gib must be positive")
    if args.engine_timeout <= 0:
        raise ValueError("--engine-timeout must be positive")
    if args.engine_tp is not None and args.engine_tp < 1:
        raise ValueError("--engine-tp must be positive")

    system = parse_system(args.system) if args.system else None
    if system:
        hardware = parse_hardware(system.hardware_id)
        gpus = system.accelerators_per_system * args.nodes
        scale_up_domain_accelerators = system.scale_up_domain_accelerators
    else:
        hardware = parse_hardware(args.hardware) if args.hardware else None
        gpus = args.gpus or 1
        scale_up_domain_accelerators = None

    metadata = fetch_model_metadata(
        args.model,
        revision=args.revision,
        timeout=args.timeout,
    )
    warnings = list(metadata.warnings)
    max_context = _max_context(metadata.config)
    if max_context is not None and args.context > max_context:
        if not args.allow_context_overflow:
            raise ValueError(
                f"requested context {args.context:,} exceeds config max_position_embeddings "
                f"{max_context:,}; pass --allow-context-overflow for a hypothetical memory-only "
                "estimate"
            )
        warnings.append(
            f"requested context {args.context:,} exceeds config max_position_embeddings "
            f"{max_context:,}; result is hypothetical"
        )
    kv_dtype, kv_dtype_source, dtype_warnings = resolve_cache_dtype(
        metadata.config,
        args.kv_dtype,
        metadata.quantization_config,
    )
    warnings.extend(dtype_warnings)
    index_dtype = args.index_dtype or kv_dtype
    engine_checks = check_engines(
        selected_engines,
        metadata,
        python=args.engine_python,
        mode=args.engine_probe,
        context_tokens=args.context,
        timeout=args.engine_timeout,
        tensor_parallel=args.engine_tp or args.tp or gpus,
        utilization=args.utilization,
        kv_dtype=kv_dtype,
        index_dtype=args.index_dtype,
    )
    engine_exit = (
        3 if args.require_engine_pass and not all(check.passed for check in engine_checks) else 0
    )
    if hardware is None:
        if args.json:
            print(
                json.dumps(
                    {
                        "input_config": args.config,
                        "model": metadata.as_dict(),
                        "context_tokens": args.context,
                        "precision": {
                            "kv_dtype": kv_dtype,
                            "kv_dtype_source": kv_dtype_source,
                            "index_dtype": args.index_dtype,
                        },
                        "engine_checks": [check.as_dict() for check in engine_checks],
                        "warnings": warnings,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            if args.config:
                print(f"Preset:       {args.config}")
            _human_engine_only(
                metadata=metadata,
                context_tokens=args.context,
                checks=engine_checks,
                warnings=warnings,
                kv_dtype=kv_dtype,
                kv_dtype_source=kv_dtype_source,
                index_dtype=args.index_dtype,
            )
        return engine_exit

    if hardware.unified_memory:
        warnings.append(
            "unified-memory availability changes with host and swap state; the selected "
            "utilization is a static counted-state budget, not measured allocatable memory"
        )

    cache = estimate_cache(
        metadata.config,
        context_tokens=args.context,
        kv_bytes=DTYPE_BYTES[kv_dtype],
        index_bytes=DTYPE_BYTES[index_dtype],
    )

    if DTYPE_BYTES[kv_dtype] < 1 or DTYPE_BYTES[index_dtype] < 1:
        warnings.append(
            "4-bit cache values are packed-payload ideals; runtime scales, alignment, and "
            "backend support can increase allocation"
        )
    if args.weight_gib is not None:
        weight_bytes = int(args.weight_gib * GIB)
        weight_source = "user override"
    elif metadata.weight_bytes is not None:
        weight_bytes = metadata.weight_bytes
        weight_source = metadata.weight_source or "HF metadata"
        if (
            metadata.runtime_weight_floor_bytes is not None
            and metadata.runtime_weight_floor_bytes > weight_bytes
        ):
            artifact_gib = weight_bytes / GIB
            weight_bytes = metadata.runtime_weight_floor_bytes
            weight_source = metadata.runtime_weight_floor_source or "published runtime floor"
            warnings.append(
                f"raised the planning weight footprint from {artifact_gib:.2f} GiB of artifacts "
                f"to a {weight_bytes / GIB:.2f} GiB published runtime floor"
            )
            warnings.append(
                "published runtime memory is approximate and backend-specific; measured loading "
                "still wins over this floor"
            )
        else:
            warnings.append(
                "repository artifact bytes are a weight-footprint proxy; runtime packing can differ"
            )
    else:
        raise ValueError(
            "HF metadata did not expose weight artifact sizes; pass --weight-gib with a "
            "runtime-specific footprint"
        )

    layouts = plan_topologies(
        cache,
        weight_bytes=weight_bytes,
        hardware=hardware,
        gpus=gpus,
        utilization=args.utilization,
        tensor_parallel=args.tp,
        scale_up_domain_accelerators=scale_up_domain_accelerators,
    )
    concurrency = summarize_concurrency(
        layouts,
        active_sequences_per_user=args.active_sequences_per_user,
    )
    if args.json:
        payload = {
            "input_config": args.config,
            "model": metadata.as_dict(),
            "cache": cache.as_dict(),
            "precision": {
                "kv_dtype": kv_dtype,
                "kv_dtype_source": kv_dtype_source,
                "index_dtype": index_dtype,
            },
            "weights": {
                "bytes": weight_bytes,
                "gib": weight_bytes / GIB,
                "source": weight_source,
            },
            "hardware": {
                **hardware.as_dict(),
                "gpus": gpus,
                "memory_gib_per_device": hardware.memory_gib,
                "utilization": args.utilization,
            },
            "system": (
                {
                    **system.as_dict(),
                    "systems": args.nodes,
                    "total_accelerators": gpus,
                }
                if system
                else None
            ),
            "layouts": [layout.as_dict() for layout in layouts],
            "concurrency": concurrency,
            "engine_checks": [check.as_dict() for check in engine_checks],
            "warnings": warnings,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if args.config:
            print(f"Preset:       {args.config}")
        _human_output(
            metadata=metadata,
            cache=cache,
            hardware=hardware,
            gpus=gpus,
            system=system,
            nodes=args.nodes,
            utilization=args.utilization,
            weight_bytes=weight_bytes,
            weight_source=weight_source,
            layouts=layouts,
            warnings=warnings,
            kv_dtype=kv_dtype,
            kv_dtype_source=kv_dtype_source,
            index_dtype=index_dtype,
            engine_checks=engine_checks,
            concurrency=concurrency,
        )
    return engine_exit


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] == "calibrate":
        from kvfit.calibrate import main as calibrate_main

        return calibrate_main(raw_argv[1:])
    normalized_argv, config_value = toml_path_from_argv(raw_argv)
    defaults: dict[str, Any] = {}
    config_checks: list[str] | None = None
    resolved_config: str | None = None
    if config_value:
        try:
            config_path, defaults = load_toml_defaults(config_value)
        except ValueError as error:
            print(f"kvfit: {error}", file=sys.stderr)
            return 2
        resolved_config = str(config_path)
        raw_checks = defaults.pop("check_engine", None)
        config_checks = list(raw_checks) if isinstance(raw_checks, list) else None
    has_cli_system = any(
        value == "--system" or value.startswith("--system=") for value in normalized_argv
    )
    has_cli_hardware = any(
        value == "--hardware" or value.startswith("--hardware=") for value in normalized_argv
    )
    if has_cli_system:
        defaults.pop("hardware", None)
        defaults.pop("gpus", None)
    if has_cli_hardware:
        defaults.pop("system", None)
        defaults.pop("nodes", None)
    parser = _parser(defaults)
    args = parser.parse_args(normalized_argv)
    args.config = resolved_config
    has_cli_engine_check = any(
        value == "--check-engine" or value.startswith("--check-engine=")
        for value in normalized_argv
    )
    if config_checks is not None and not has_cli_engine_check:
        args.check_engine = config_checks
    try:
        return run(args)
    except UnsupportedArchitecture as error:
        print(f"kvfit: {error}", file=sys.stderr)
        print("kvfit: refusing to produce a potentially wrong capacity verdict", file=sys.stderr)
        return 2
    except (HuggingFaceError, ValueError) as error:
        print(f"kvfit: {error}", file=sys.stderr)
        return 2
