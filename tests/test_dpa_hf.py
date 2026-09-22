from __future__ import annotations

import io
import json
import struct

import pytest

from kvfit.hf import HuggingFaceError, _read_safetensors_header


class Response(io.BytesIO):
    def __init__(self, body, start, end, *, status=206, total=999):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}


@pytest.mark.parametrize("failure", [None, "ignored", "short", "wrong_range", "huge", "duplicate"])
def test_header_reads_are_bounded_and_reject_invalid_responses(monkeypatch, failure):
    header = json.dumps({"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    if failure == "duplicate":
        header = b'{"x": {}, "x": {}}'
    requests = []

    def open_request(request, **kwargs):
        value = request.get_header("Range")
        requests.append(value)
        if len(requests) == 1:
            n = 1 << 40 if failure == "huge" else len(header)
            return Response(struct.pack("<Q", n), 0, 7, status=200 if failure == "ignored" else 206)
        return Response(
            header[:-1] if failure == "short" else header,
            9 if failure == "wrong_range" else 8,
            7 + len(header),
        )

    monkeypatch.setattr("kvfit.hf._range_opener.open", open_request)
    if failure:
        with pytest.raises(HuggingFaceError):
            _read_safetensors_header("https://huggingface.co/test/file", token=None, timeout=1)
    else:
        result, header_bytes, file_bytes = _read_safetensors_header(
            "https://huggingface.co/test/file", token=None, timeout=1
        )
        assert result["x"]["shape"] == [1]
        assert header_bytes == len(header) + 8
        assert file_bytes == 999
    assert len(requests) == (1 if failure in {"huge", "ignored"} else 2)


def test_cross_host_range_redirect_strips_auth():
    from urllib.request import Request

    from kvfit.hf import _SafeMetadataRedirect

    handler = _SafeMetadataRedirect()
    request = Request("https://huggingface.co/model", headers={"Authorization": "Bearer private"})
    redirected = handler.redirect_request(
        request, None, 302, "Found", {}, "https://cdn.example/signed-object"
    )
    assert not redirected.has_header("Authorization")


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "config_drift",
        "missing_tensor",
        "wrong_index_total",
        "wrong_artifact_total",
        "unindexed_tensor",
        "overlap",
    ],
)
def test_fetch_reconciles_pinned_config_headers_index_and_artifacts(monkeypatch, mutation):
    from test_dpa import tiny_checkpoint

    from kvfit.dpa import fetch_dpa_weights
    from kvfit.models import ModelMetadata

    config, tensors = tiny_checkpoint(quantized=True)
    size = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values())
    index = {
        "metadata": {"total_size": size},
        "weight_map": {k: "model.safetensors" for k in tensors},
    }
    metadata = ModelMetadata(
        "test/model", "moving", "fixed", config, size + 100, "HF root safetensors (1 file(s))"
    )
    if mutation == "missing_tensor":
        del tensors[next(iter(tensors))]
    elif mutation == "wrong_index_total":
        index["metadata"]["total_size"] += 1
    elif mutation == "wrong_artifact_total":
        from dataclasses import replace

        metadata = replace(metadata, weight_bytes=metadata.weight_bytes + 1)
    elif mutation == "unindexed_tensor":
        tensors["extra.weight"] = {"dtype": "BF16", "shape": [1], "data_offsets": [size, size + 2]}
    elif mutation == "overlap":
        key = list(tensors)[1]
        length = tensors[key]["data_offsets"][1] - tensors[key]["data_offsets"][0]
        tensors[key]["data_offsets"] = [0, length]
    requests = []

    def fetch(url, **kwargs):
        requests.append(url)
        assert "/fixed/" in url
        if url.endswith("config.json"):
            return {**config, "hidden_size": 256} if mutation == "config_drift" else config
        return index

    monkeypatch.setattr("kvfit.dpa._fetch_json", fetch)
    monkeypatch.setattr(
        "kvfit.dpa._read_safetensors_header", lambda *a, **k: (tensors, 100, size + 100)
    )
    if mutation:
        with pytest.raises(ValueError):
            fetch_dpa_weights(metadata)
    else:
        result = fetch_dpa_weights(metadata)
        assert result.routed_expert_bytes + result.replicated_bytes == size
        assert len(requests) == 2
