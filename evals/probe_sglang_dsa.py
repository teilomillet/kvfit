"""Read pinned SGLang shape functions in isolation; no torch, weights or GPU.

Executes only the three named, inspected arithmetic functions, with a closed
namespace of dtype/config stubs. This observes allocation shapes from source,
not allocator behavior or performance. Run from the repository root.
"""

from __future__ import annotations
import __future__

import ast
import hashlib
import json
import urllib.request
from types import SimpleNamespace as NS

REVISION = "20a491d1d311"
BASE = f"https://raw.githubusercontent.com/sgl-project/sglang/{REVISION}/python/sglang/srt/"


def selected_function(source, name):
    matches = [
        n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name}")
    module = ast.Module(body=[matches[0]], type_ignores=[])
    return compile(
        ast.fix_missing_locations(module),
        f"upstream:{name}",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )


def main():
    paths = ["mem_cache/kv_cache_configurator.py", "mem_cache/index_key_cache.py"]
    sources = {p: urllib.request.urlopen(BASE + p, timeout=30).read() for p in paths}
    namespace = {
        "__builtins__": {},
        "is_deepseek_dsa": lambda config: True,
        "_is_hip": False,
        "torch": NS(float8_e4m3fn="fp8"),
        "DSATokenToKVPool": NS(quant_block_size=128, rope_storage_dtype=NS(itemsize=2)),
    }
    exec(selected_function(sources[paths[0]], "calculate_mla_kv_cache_dim"), namespace)
    model = NS(hf_config={}, kv_lora_rank=512, qk_rope_head_dim=64)
    cells = {}
    for backend in ["trtllm", "flashmla_sparse_q8"]:
        namespace["get_exec"] = lambda b=backend: NS(
            kernel=NS(dsa_prefill_backend=b, dsa_decode_backend=b)
        )
        for dtype, width in [("fp8", 1), ("bf16", 2)]:
            cells[f"{backend}:{dtype}"] = (
                namespace["calculate_mla_kv_cache_dim"](model_config=model, kv_cache_dtype=dtype)
                * width
            )
    shapes = {"__builtins__": {}}
    for name in ["_buffer_shape", "_layer_num_pages"]:
        exec(selected_function(sources[paths[1]], name), shapes)
    pool = NS(
        page_size=64, index_head_dim=128, quant_block_size=128, skip_topk_layers=[False, True]
    )
    obj = NS(pool=pool)
    print(
        json.dumps(
            {
                "revision": REVISION,
                "sources": [
                    {"url": BASE + p, "sha256": hashlib.sha256(b).hexdigest()}
                    for p, b in sources.items()
                ],
                "evidence": (
                    "isolated execution of pinned upstream shape functions; no GPU allocation"
                ),
                "mla_bytes_per_layer_token": cells,
                "index_shape_one_page": shapes["_buffer_shape"](obj, 1),
                "index_pages_full_shared": [
                    shapes["_layer_num_pages"](obj, i, 17) for i in range(2)
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
