"""Recheck the inspected, pinned upstream DPA rank function, without torch/GPU."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def verify_dpa():
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/sglang-dpa.json").read_text()
    )
    url = (
        fixture["source"]
        .replace("github.com/", "raw.githubusercontent.com/")
        .replace("/blob/", "/")
    )
    with urlopen(url, timeout=30) as response:
        source = response.read(1024 * 1024 + 1)
    if hashlib.sha256(source).hexdigest() != fixture["source_sha256"]:
        raise ValueError("pinned source differs from the inspected DPA function source")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_dp_attention_world_info"
    )
    namespace = {"__builtins__": {}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), url, "exec"), namespace)
    observed = {
        str(dp): [list(namespace[function.name](dp > 1, rank, 8, dp)) for rank in range(8)]
        for dp in [1, 2, 4, 8]
    }
    if observed != fixture["world_info"]:
        raise ValueError("upstream attention rank ownership differs from the recorded fixture")
    return {
        "source": url,
        "sha256": fixture["source_sha256"],
        "world_info": observed,
        "qualification": "isolated upstream function; no GPU or engine launch",
    }


if __name__ == "__main__":
    print(json.dumps(verify_dpa(), indent=2))
