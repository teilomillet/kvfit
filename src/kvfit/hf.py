from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener, urlopen

from kvfit import __version__
from kvfit.models import GIB, ModelMetadata

HF_BASE = "https://huggingface.co"
USER_AGENT = f"kvfit/{__version__}"
MAX_CONFIG_BYTES = 16 * 1024**2
MAX_REPOSITORY_METADATA_BYTES = 32 * 1024**2
MAX_QUANTIZATION_CONFIG_BYTES = 16 * 1024**2
OPENAI_GPT_OSS_MEMORY_REFERENCE = (
    "https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers#pick-your-model"
)
SHARD_RE = re.compile(r"^model-\d+-of-\d+\.safetensors$")
SPLIT_ARTIFACT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})(?P<suffix>\.[^.]+)$"
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
DTYPE_BYTES = {
    "BOOL": 1 / 8,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


class HuggingFaceError(RuntimeError):
    pass


class _SafeMetadataRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise HuggingFaceError("metadata redirect must remain HTTPS")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


_range_opener = build_opener(_SafeMetadataRedirect())
MAX_SAFETENSORS_HEADER_BYTES = 8 * 1024**2


def _read_safetensors_header(
    url: str,
    *,
    token: str | None,
    timeout: float,
) -> tuple[dict[str, Any], int, int]:
    """Read only bounded metadata ranges; never fall back to a weight download."""

    def read_range(start: int, end: int, part: str) -> tuple[bytes, int]:
        headers = {"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        # Distinct URLs avoid CDNs reusing the eight-byte response for the header.
        request = Request(f"{url}?kvfit_header={part}", headers=headers)
        try:
            with _range_opener.open(request, timeout=timeout) as response:
                match = re.fullmatch(
                    r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
                )
                if response.status != 206 or not match:
                    raise HuggingFaceError("server did not honor the metadata byte range")
                first, last, total = map(int, match.groups())
                if (first, last) != (start, end) or total <= end:
                    raise HuggingFaceError("incorrect Content-Range in checkpoint metadata")
                raw = response.read(end - start + 2)
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise HuggingFaceError(f"failed to read checkpoint header: {error}") from error
        if len(raw) != end - start + 1:
            raise HuggingFaceError("truncated or oversized checkpoint header range")
        return raw, total

    length, total = read_range(0, 7, "length")
    size = struct.unpack("<Q", length)[0]
    if not 2 <= size <= MAX_SAFETENSORS_HEADER_BYTES or size + 8 > total:
        raise HuggingFaceError("invalid or oversized safetensors header")
    raw, second_total = read_range(8, 7 + size, "json")
    if total != second_total:
        raise HuggingFaceError("checkpoint size changed between metadata reads")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        header = json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError) as error:
        raise HuggingFaceError("invalid safetensors JSON header") from error
    if not isinstance(header, dict):
        raise HuggingFaceError("safetensors header must be an object")
    return header, size + 8, total


@dataclass(frozen=True)
class ParsedModelReference:
    repo_id: str
    revision: str
    artifact_path: str | None = None


def _parse_model_reference(value: str, revision: str | None = None) -> ParsedModelReference:
    text = value.strip()
    if not text:
        raise ValueError("model reference is empty")

    parsed = urlparse(text)
    segments: list[str]
    inferred_revision: str | None = None
    artifact_path: str | None = None
    if parsed.scheme in {"http", "https"}:
        if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("only huggingface.co model URLs are supported")
        segments = [segment for segment in parsed.path.split("/") if segment]
    elif parsed.scheme == "hf":
        segments = [parsed.netloc, *[segment for segment in parsed.path.split("/") if segment]]
    elif "://" in text:
        raise ValueError("unsupported model URL scheme")
    else:
        bare, separator, suffix = text.partition("@")
        if separator:
            inferred_revision = suffix
        segments = [segment for segment in bare.strip("/").split("/") if segment]

    if len(segments) < 2:
        raise ValueError("expected an HF model ID like owner/model")
    repo_id = "/".join(segments[:2])
    if len(segments) >= 4 and segments[2] in {"tree", "blob", "resolve"}:
        inferred_revision = unquote(segments[3])
        if segments[2] in {"blob", "resolve"} and len(segments) >= 5:
            artifact_path = unquote("/".join(segments[4:]))
    return ParsedModelReference(
        repo_id=repo_id,
        revision=revision or inferred_revision or "main",
        artifact_path=artifact_path,
    )


def parse_model_reference(value: str, revision: str | None = None) -> tuple[str, str]:
    """Parse an HF repo ID or URL into (repo_id, revision)."""
    parsed = _parse_model_reference(value, revision)
    return parsed.repo_id, parsed.revision


def _fetch_json(
    url: str,
    *,
    token: str | None,
    max_bytes: int,
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise HuggingFaceError(f"response exceeded {max_bytes} bytes: {url}")
            raw = response.read(max_bytes + 1)
    except HTTPError as error:
        if error.code in {401, 403}:
            raise HuggingFaceError(
                f"Hugging Face denied access to the model metadata (HTTP {error.code}); "
                "set HF_TOKEN for gated repositories"
            ) from error
        if error.code == 404:
            raise HuggingFaceError(f"model metadata was not found: {url}") from error
        raise HuggingFaceError(f"Hugging Face returned HTTP {error.code}: {url}") from error
    except (URLError, TimeoutError) as error:
        raise HuggingFaceError(f"failed to fetch Hugging Face metadata: {error}") from error
    if len(raw) > max_bytes:
        raise HuggingFaceError(f"response exceeded {max_bytes} bytes: {url}")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HuggingFaceError(f"Hugging Face returned invalid JSON: {url}") from error
    if not isinstance(result, dict):
        raise HuggingFaceError(f"expected a JSON object from Hugging Face: {url}")
    return result


def _sibling_size(sibling: Mapping[str, Any]) -> int | None:
    size = sibling.get("size")
    if isinstance(size, int) and size >= 0:
        return size
    lfs = sibling.get("lfs")
    if isinstance(lfs, Mapping):
        lfs_size = lfs.get("size")
        if isinstance(lfs_size, int) and lfs_size >= 0:
            return lfs_size
    return None


def _artifact_group(
    siblings: list[Mapping[str, Any]],
    artifact_path: str,
) -> list[tuple[str, int]]:
    available: dict[str, int] = {}
    for sibling in siblings:
        name = sibling.get("rfilename")
        size = _sibling_size(sibling)
        if isinstance(name, str) and size is not None:
            available[name] = size
    if artifact_path not in available:
        raise HuggingFaceError(
            f"selected artifact was not found in the repository: {artifact_path}"
        )
    match = SPLIT_ARTIFACT_RE.match(artifact_path)
    if not match:
        return [(artifact_path, available[artifact_path])]
    prefix = match.group("prefix")
    count = int(match.group("count"))
    suffix = match.group("suffix")
    members: list[tuple[int, str, int]] = []
    for name, size in available.items():
        candidate = SPLIT_ARTIFACT_RE.match(name)
        if (
            candidate
            and candidate.group("prefix") == prefix
            and candidate.group("suffix") == suffix
            and int(candidate.group("count")) == count
        ):
            members.append((int(candidate.group("index")), name, size))
    indices = sorted(index for index, _, _ in members)
    if indices not in (list(range(1, count + 1)), list(range(count))):
        raise HuggingFaceError(
            f"selected split artifact is incomplete: {artifact_path}; found shard indices {indices}"
        )
    return [(name, size) for _, name, size in sorted(members)]


def _weight_artifact_bytes(
    metadata: Mapping[str, Any],
    artifact_path: str | None = None,
) -> tuple[int | None, str | None]:
    siblings = metadata.get("siblings", ())
    if not isinstance(siblings, list):
        siblings = []
    typed_siblings = [sibling for sibling in siblings if isinstance(sibling, Mapping)]

    if artifact_path is not None:
        chosen = _artifact_group(typed_siblings, artifact_path)
        kind = "GGUF" if artifact_path.lower().endswith(".gguf") else "selected artifact"
        return sum(
            size for _, size in chosen
        ), f"HF {kind} ({len(chosen)} file(s)): {artifact_path}"

    candidates: list[tuple[str, int]] = []
    fallback: list[tuple[str, int]] = []
    gguf: list[str] = []
    for sibling in typed_siblings:
        name = sibling.get("rfilename")
        size = _sibling_size(sibling)
        if not isinstance(name, str) or size is None:
            continue
        if "/" not in name and (name == "model.safetensors" or SHARD_RE.match(name)):
            candidates.append((name, size))
        elif "/" not in name and name.endswith(".safetensors"):
            fallback.append((name, size))
        elif name.lower().endswith(".gguf"):
            gguf.append(name)
    chosen = candidates or fallback
    if chosen:
        return sum(size for _, size in chosen), f"HF root safetensors ({len(chosen)} file(s))"
    if len(gguf) == 1:
        selected = _artifact_group(typed_siblings, gguf[0])
        return sum(size for _, size in selected), f"HF GGUF ({len(selected)} file(s)): {gguf[0]}"
    if len(gguf) > 1:
        examples = ", ".join(gguf[:3])
        raise HuggingFaceError(
            "repository contains multiple GGUF variants; pass a Hugging Face blob URL for the "
            f"exact file (for example: {examples})"
        )

    safetensors = metadata.get("safetensors")
    if not isinstance(safetensors, Mapping):
        return None, None
    parameters = safetensors.get("parameters")
    if isinstance(parameters, Mapping):
        total = 0.0
        seen = False
        for dtype, count in parameters.items():
            if str(dtype).upper() not in DTYPE_BYTES or not isinstance(count, int):
                continue
            total += count * DTYPE_BYTES[str(dtype).upper()]
            seen = True
        if seen:
            return int(total), "HF safetensors dtype counts (runtime overhead excluded)"
    return None, None


def _text_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = config.get("text_config")
    if isinstance(nested, Mapping) and nested.get("num_hidden_layers") is not None:
        return nested
    return config


def _validate_gpt_oss_index(
    config: Mapping[str, Any],
    index: Mapping[str, Any],
) -> None:
    text_config = _text_config(config)
    layers = text_config.get("num_hidden_layers")
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise HuggingFaceError("GPT-OSS config has no valid num_hidden_layers")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise HuggingFaceError("GPT-OSS safetensors index has no weight_map")
    found = {
        int(match.group(1))
        for name in weight_map
        if isinstance(name, str) and (match := LAYER_RE.search(name))
    }
    expected = set(range(layers))
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        detail = f"missing layers {missing}" if missing else ""
        if extra:
            detail += ("; " if detail else "") + f"unexpected layers {extra}"
        raise HuggingFaceError(
            f"checkpoint index does not cover the declared {layers}-layer GPT-OSS model: {detail}"
        )


def _weight_dtypes(metadata: Mapping[str, Any]) -> dict[str, int]:
    safetensors = metadata.get("safetensors")
    if not isinstance(safetensors, Mapping):
        return {}
    parameters = safetensors.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    return {
        str(dtype).upper(): count
        for dtype, count in parameters.items()
        if isinstance(count, int) and count >= 0
    }


def _external_quantization(
    external_config: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, int | None]:
    quantization = external_config.get("quantization")
    if not isinstance(quantization, Mapping):
        return None, None, None, None
    raw_method = quantization.get("quant_algo") or quantization.get("quant_method")
    method = raw_method.strip().lower() if isinstance(raw_method, str) else None
    raw_kv = quantization.get("kv_cache_quant_algo")
    kv_method = raw_kv.strip().lower() if isinstance(raw_kv, str) else None
    excluded = quantization.get("exclude_modules")
    excluded_count = (
        len(excluded)
        if isinstance(excluded, list) and all(isinstance(value, str) for value in excluded)
        else None
    )
    scope = "mixed" if excluded_count else ("whole-model" if method else None)
    return method, kv_method, scope, excluded_count


def _quantization_label(
    config: Mapping[str, Any],
    artifact_path: str | None,
    external_config: Mapping[str, Any] | None = None,
) -> str | None:
    if artifact_path and artifact_path.lower().endswith(".gguf"):
        return "gguf"
    if external_config:
        method, _, _, _ = _external_quantization(external_config)
        if method:
            return method
    text_config = _text_config(config)
    quantization = text_config.get("quantization_config", config.get("quantization_config"))
    if isinstance(quantization, Mapping):
        method = quantization.get("quant_method") or quantization.get("mode")
        if isinstance(method, str):
            return method
        bits = quantization.get("bits")
        if isinstance(bits, int):
            return f"{bits}-bit"
    dtype = text_config.get("torch_dtype", text_config.get("dtype"))
    return str(dtype) if isinstance(dtype, str) else None


def _metadata_warnings(
    config: Mapping[str, Any],
    weight_dtypes: Mapping[str, int],
    artifact_path: str | None,
    external_config: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if artifact_path and artifact_path.lower().endswith(".gguf"):
        warnings.append(
            "GGUF shard completeness is inferred from filenames; tensor coverage is not yet "
            "validated"
        )
    quantization = _text_config(config).get(
        "quantization_config", config.get("quantization_config")
    )
    method = quantization.get("quant_method") if isinstance(quantization, Mapping) else None
    full_precision = {"BF16", "F16", "F32", "F64"}
    if method and weight_dtypes and set(weight_dtypes) <= full_precision:
        warnings.append(
            f"config declares {method!s} quantization, but repository tensor metadata is "
            "full-precision only; runtime conversion is backend-dependent"
        )
    if external_config:
        external_method, kv_method, scope, excluded_count = _external_quantization(external_config)
        if external_method and scope == "mixed":
            warnings.append(
                f"hf_quant_config.json declares mixed {external_method} weight quantization "
                f"with {excluded_count} excluded module patterns"
            )
        if kv_method == "none":
            warnings.append(
                "hf_quant_config.json explicitly leaves the KV cache unquantized; cache "
                "precision still follows the model dtype or --kv-dtype"
            )
    return tuple(warnings)


def _gpt_oss_runtime_weight_floor(
    config: Mapping[str, Any],
    artifact_bytes: int | None,
    weight_dtypes: Mapping[str, int],
) -> tuple[int | None, str | None]:
    text_config = _text_config(config)
    if str(text_config.get("model_type", config.get("model_type", ""))).lower() != "gpt_oss":
        return None, None
    layers = text_config.get("num_hidden_layers")
    if layers == 24:
        full_precision = {"BF16", "F16", "F32", "F64"}
        floor_gib = 48 if weight_dtypes and set(weight_dtypes) <= full_precision else 16
    elif layers == 36:
        floor_gib = 60
    else:
        return None, None
    floor_bytes = floor_gib * GIB
    if artifact_bytes is not None and artifact_bytes >= floor_bytes:
        return None, None
    return (
        floor_bytes,
        f"OpenAI approximate GPT-OSS runtime guidance: {OPENAI_GPT_OSS_MEMORY_REFERENCE}",
    )


def fetch_model_metadata(
    model: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    timeout: float = 20.0,
) -> ModelMetadata:
    parsed = _parse_model_reference(model, revision)
    repo_id = parsed.repo_id
    requested_revision = parsed.revision
    selected_artifact = (
        parsed.artifact_path
        if parsed.artifact_path and parsed.artifact_path.lower().endswith((".safetensors", ".gguf"))
        else None
    )
    token = token or os.environ.get("HF_TOKEN")
    quoted_repo = quote(repo_id, safe="/")
    quoted_revision = quote(requested_revision, safe="")
    config_url = f"{HF_BASE}/{quoted_repo}/resolve/{quoted_revision}/config.json"
    metadata_url = f"{HF_BASE}/api/models/{quoted_repo}/revision/{quoted_revision}?" + urlencode(
        {"blobs": "true"}
    )
    # Publisher checkpoints can embed per-module mixed-precision plans with
    # thousands of target names. Keep a finite bound, but do not reject valid
    # ModelOpt/Quark configs solely because they exceed the older 2 MiB cap.
    config = _fetch_json(config_url, token=token, max_bytes=MAX_CONFIG_BYTES, timeout=timeout)
    try:
        metadata = _fetch_json(
            metadata_url,
            token=token,
            max_bytes=MAX_REPOSITORY_METADATA_BYTES,
            timeout=timeout,
        )
    except HuggingFaceError:
        if requested_revision != "main":
            raise
        fallback_url = f"{HF_BASE}/api/models/{quoted_repo}?{urlencode({'blobs': 'true'})}"
        metadata = _fetch_json(
            fallback_url,
            token=token,
            max_bytes=MAX_REPOSITORY_METADATA_BYTES,
            timeout=timeout,
        )
    weight_bytes, weight_source = _weight_artifact_bytes(metadata, selected_artifact)
    siblings = metadata.get("siblings")
    sibling_rows = siblings if isinstance(siblings, list) else []
    sibling_names = {
        sibling.get("rfilename") for sibling in sibling_rows if isinstance(sibling, Mapping)
    }
    external_quantization_config: dict[str, Any] = {}
    quantization_config_path: str | None = None
    if "hf_quant_config.json" in sibling_names:
        quantization_config_path = "hf_quant_config.json"
        quantization_url = (
            f"{HF_BASE}/{quoted_repo}/resolve/{quoted_revision}/{quantization_config_path}"
        )
        external_quantization_config = _fetch_json(
            quantization_url,
            token=token,
            max_bytes=MAX_QUANTIZATION_CONFIG_BYTES,
            timeout=timeout,
        )
    text_config = _text_config(config)
    model_type = str(text_config.get("model_type", config.get("model_type", ""))).lower()
    if model_type == "gpt_oss" and not (
        selected_artifact and selected_artifact.lower().endswith(".gguf")
    ):
        names = {
            sibling.get("rfilename") for sibling in sibling_rows if isinstance(sibling, Mapping)
        }
        index_name = "model.safetensors.index.json"
        if index_name not in names:
            raise HuggingFaceError(
                "GPT-OSS safetensors checkpoint has no model.safetensors.index.json; "
                "layer completeness cannot be verified"
            )
        index_url = f"{HF_BASE}/{quoted_repo}/resolve/{quoted_revision}/{index_name}"
        index = _fetch_json(
            index_url,
            token=token,
            max_bytes=MAX_REPOSITORY_METADATA_BYTES,
            timeout=timeout,
        )
        _validate_gpt_oss_index(config, index)
    weight_dtypes = _weight_dtypes(metadata)
    warnings = _metadata_warnings(
        config,
        weight_dtypes,
        selected_artifact,
        external_quantization_config,
    )
    external_method, external_kv, quantization_scope, excluded_count = _external_quantization(
        external_quantization_config
    )
    embedded_quantization = _text_config(config).get(
        "quantization_config", config.get("quantization_config")
    )
    embedded_method = (
        embedded_quantization.get("quant_method")
        if isinstance(embedded_quantization, Mapping)
        else None
    )
    runtime_floor, runtime_floor_source = _gpt_oss_runtime_weight_floor(
        config,
        weight_bytes,
        weight_dtypes,
    )
    sha = metadata.get("sha")
    return ModelMetadata(
        repo_id=repo_id,
        requested_revision=requested_revision,
        resolved_revision=sha if isinstance(sha, str) else None,
        config=config,
        weight_bytes=weight_bytes,
        weight_source=weight_source,
        selected_artifact=selected_artifact,
        quantization=_quantization_label(
            config,
            selected_artifact,
            external_quantization_config,
        ),
        quantization_source=(
            "hf_quant_config.json: quantization.quant_algo"
            if external_method
            else "config.json: quantization_config.quant_method"
            if embedded_method
            else "selected GGUF artifact"
            if selected_artifact and selected_artifact.lower().endswith(".gguf")
            else "config.json: model dtype"
        ),
        quantization_scope=quantization_scope,
        quantization_excluded_modules=excluded_count,
        kv_cache_quantization=external_kv,
        quantization_config_path=quantization_config_path,
        quantization_config=external_quantization_config,
        weight_dtypes=weight_dtypes,
        warnings=warnings,
        runtime_weight_floor_bytes=runtime_floor,
        runtime_weight_floor_source=runtime_floor_source,
    )
