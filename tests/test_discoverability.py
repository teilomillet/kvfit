from __future__ import annotations

import tomllib
from pathlib import Path

from kvfit import __version__

ROOT = Path(__file__).parents[1]


def test_readme_first_screen_exposes_installation_and_retrieval_terms() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:5_000].lower()

    for phrase in (
        "pip install kvfit",
        "uvx kvfit",
        "hugging face model fit",
        "kv-cache calculator",
        "gpu-memory",
        "dgx capacity planner",
        "vllm",
        "sglang",
        "--json",
    ):
        assert phrase in first_screen


def test_package_metadata_routes_people_to_public_project_surfaces() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["version"] == __version__
    assert "Hugging Face LLM" in project["description"]
    assert "KV cache" in project["description"]
    assert project["urls"] == {
        "Homepage": "https://github.com/teilomillet/kvfit",
        "Documentation": "https://github.com/teilomillet/kvfit#readme",
        "Repository": "https://github.com/teilomillet/kvfit.git",
        "Issues": "https://github.com/teilomillet/kvfit/issues",
    }
    assert {
        "kv-cache",
        "gpu-memory",
        "vram-calculator",
        "capacity-planning",
        "huggingface",
        "vllm",
        "sglang",
        "dgx",
    } <= set(project["keywords"])
