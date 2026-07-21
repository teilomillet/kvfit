from __future__ import annotations

import tomllib
from pathlib import Path

from kvfit import __version__

ROOT = Path(__file__).parents[1]


def normalized_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").lower().replace("`", "").replace("*", "")
    return " ".join(text.split())


def test_readme_first_screen_exposes_installation_and_retrieval_terms() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    first_screen = readme[:5_000].lower()

    for phrase in (
        "pip install kvfit",
        "uvx kvfit",
        "hugging face model fit",
        "kv-cache calculator",
        "gpu-memory",
        "similar tools",
        "dgx capacity planner",
        "vllm",
        "sglang",
        "--json",
    ):
        assert phrase in first_screen

    assert "https://github.com/teilomillet/kvfit/blob/main/docs/alternatives.md" in readme
    assert "https://github.com/teilomillet/kvfit/blob/main/docs/minimax-m3-dgx-spark.md" in readme
    assert "https://github.com/teilomillet/kvfit/blob/main/skills/kvfit/SKILL.md" in readme


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
        "llm-memory-calculator",
        "gpu-memory-calculator",
        "model-fit",
        "inference-memory",
        "recurrent-state",
        "huggingface",
        "vllm",
        "sglang",
        "dgx",
        "dgx-spark",
        "gb10",
    } <= set(project["keywords"])


def test_agent_and_comparison_surfaces_explain_selection_and_evidence_boundaries() -> None:
    llms = normalized_text(ROOT / "llms.txt")
    comparison = normalized_text(ROOT / "docs" / "alternatives.md")
    minimax = normalized_text(ROOT / "docs" / "minimax-m3-dgx-spark.md")

    for phrase in (
        "hugging face llm",
        "kv-cache calculator",
        "gpu-memory calculator",
        "target-host calibration",
        "static",
        "measured",
    ):
        assert phrase in llms

    for name in (
        "accelerate estimate-memory",
        "modelinfo cli",
        "hf-mem",
        "fitllm",
        "llmfit",
        "llm-mem-planner",
    ):
        assert name in comparison

    for phrase in (
        "nvidia/minimax-m3-nvfp4",
        "--system dgx-spark",
        "--nodes 2",
        "--tp auto",
        "separate memory domains",
        "does not prove",
    ):
        assert phrase in minimax


def test_portable_agent_skill_has_search_triggers_and_no_template_placeholders() -> None:
    skill = normalized_text(ROOT / "skills" / "kvfit" / "SKILL.md")
    agents = normalized_text(ROOT / "AGENTS.md")

    for phrase in (
        "hugging face model",
        "kv-cache calculators",
        "dgx spark",
        "model fit/oom",
        "tensor parallelism",
        "target-host calibration",
        "--engine-python /path/to/serving-env/bin/python",
        "uvx --from kvfit kvfit-calibrate",
    ):
        assert phrase in skill

    assert "todo" not in skill
    assert "skills/kvfit/skill.md" in agents
