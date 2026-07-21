from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Hardware:
    id: str
    label: str
    memory_gib: float
    unified_memory: bool = False
    note: str | None = None
    vendor: str | None = None
    architecture: str | None = None
    memory_type: str | None = None
    source_url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "memory_gib": self.memory_gib,
            "unified_memory": self.unified_memory,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "memory_type": self.memory_type,
            "source_url": self.source_url,
            "note": self.note,
        }


NVIDIA_MIG_SPECS = "https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-gpus.html"
NVIDIA_HGX_SPECS = (
    "https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/"
    "latest/components.html"
)
AMD_MI300_SPECS = "https://www.amd.com/en/products/accelerators/instinct/mi300.html"
AMD_MI350_SPECS = "https://www.amd.com/en/products/accelerators/instinct/mi350.html"


HARDWARE: dict[str, Hardware] = {
    "dgx-spark": Hardware(
        id="dgx-spark",
        label="NVIDIA DGX Spark / GB10",
        memory_gib=128,
        unified_memory=True,
        vendor="NVIDIA",
        architecture="Grace Blackwell / GB10",
        memory_type="LPDDR5x unified",
        source_url="https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
        note=(
            "128 GB is unified system memory shared with the OS and applications, and "
            "DGX OS reserves memory for display; validate the utilization assumption on "
            "the target machine."
        ),
    ),
    "l40s-48": Hardware(
        id="l40s-48",
        label="NVIDIA L40S 48 GB",
        memory_gib=48,
        vendor="NVIDIA",
        architecture="Ada Lovelace",
        memory_type="GDDR6 ECC",
        source_url="https://www.nvidia.com/en-us/data-center/l40s/",
        note="PCIe accelerator without NVLink; multi-GPU TP depends on host fabric and NCCL.",
    ),
    "rtx-pro-6000-96": Hardware(
        id="rtx-pro-6000-96",
        label="NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB",
        memory_gib=96,
        vendor="NVIDIA",
        architecture="Blackwell / GB202",
        memory_type="GDDR7 ECC",
        source_url=NVIDIA_MIG_SPECS,
        note="PCIe server GPU; this memory planner does not qualify its scale-up fabric.",
    ),
    "h100-80": Hardware(
        id="h100-80",
        label="NVIDIA H100 80 GB",
        memory_gib=80,
        vendor="NVIDIA",
        architecture="Hopper / GH100",
        memory_type="HBM3",
        source_url=NVIDIA_MIG_SPECS,
    ),
    "h100-nvl-94": Hardware(
        id="h100-nvl-94",
        label="NVIDIA H100 NVL 94 GB",
        memory_gib=94,
        vendor="NVIDIA",
        architecture="Hopper / GH100",
        memory_type="HBM3",
        source_url=NVIDIA_MIG_SPECS,
    ),
    "h200-141": Hardware(
        id="h200-141",
        label="NVIDIA H200 141 GB",
        memory_gib=141,
        vendor="NVIDIA",
        architecture="Hopper / GH100",
        memory_type="HBM3e",
        source_url=NVIDIA_MIG_SPECS,
    ),
    "b200-180": Hardware(
        id="b200-180",
        label="NVIDIA B200 180 GB",
        memory_gib=180,
        vendor="NVIDIA",
        architecture="Blackwell / GB100",
        memory_type="HBM3e",
        source_url=NVIDIA_HGX_SPECS,
    ),
    "gb200-186": Hardware(
        id="gb200-186",
        label="NVIDIA GB200 186 GB GPU",
        memory_gib=186,
        vendor="NVIDIA",
        architecture="Grace Blackwell / GB100",
        memory_type="HBM3e",
        source_url=NVIDIA_MIG_SPECS,
        note=(
            "Per-GPU memory in a GB200 system; topology assumptions still require the target rack."
        ),
    ),
    "b300-288": Hardware(
        id="b300-288",
        label="NVIDIA B300 288 GB",
        memory_gib=288,
        vendor="NVIDIA",
        architecture="Blackwell Ultra",
        memory_type="HBM3e",
        source_url=NVIDIA_HGX_SPECS,
    ),
    "mi300x-192": Hardware(
        id="mi300x-192",
        label="AMD Instinct MI300X 192 GB",
        memory_gib=192,
        vendor="AMD",
        architecture="CDNA 3",
        memory_type="HBM3",
        source_url=AMD_MI300_SPECS,
    ),
    "mi325x-256": Hardware(
        id="mi325x-256",
        label="AMD Instinct MI325X 256 GB",
        memory_gib=256,
        vendor="AMD",
        architecture="CDNA 3",
        memory_type="HBM3e",
        source_url=AMD_MI300_SPECS,
    ),
    "mi350x-288": Hardware(
        id="mi350x-288",
        label="AMD Instinct MI350X 288 GB",
        memory_gib=288,
        vendor="AMD",
        architecture="CDNA 4",
        memory_type="HBM3e",
        source_url="https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html",
    ),
    "mi355x-288": Hardware(
        id="mi355x-288",
        label="AMD Instinct MI355X 288 GB",
        memory_gib=288,
        vendor="AMD",
        architecture="CDNA 4",
        memory_type="HBM3e",
        source_url="https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html",
    ),
}

ALIASES = {
    "gb10": "dgx-spark",
    "spark": "dgx-spark",
    "l40s": "l40s-48",
    "rtx-pro-6000": "rtx-pro-6000-96",
    "h100": "h100-80",
    "h100-80gb": "h100-80",
    "h100-nvl": "h100-nvl-94",
    "h200": "h200-141",
    "b200": "b200-180",
    "gb200": "gb200-186",
    "b300": "b300-288",
    "mi300x": "mi300x-192",
    "mi325x": "mi325x-256",
    "mi350x": "mi350x-288",
    "mi355x": "mi355x-288",
}


def parse_hardware(value: str) -> Hardware:
    normalized = value.strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized in HARDWARE:
        return HARDWARE[normalized]
    if normalized.startswith("custom:"):
        raw_memory = normalized.partition(":")[2]
        try:
            memory_gib = float(raw_memory)
        except ValueError as error:
            raise ValueError("custom hardware must look like custom:80") from error
        if memory_gib <= 0:
            raise ValueError("custom hardware memory must be positive")
        return Hardware(
            id=f"custom:{memory_gib:g}",
            label=f"Custom {memory_gib:g} GiB accelerator",
            memory_gib=memory_gib,
        )
    choices = ", ".join(sorted(HARDWARE))
    raise ValueError(f"unknown hardware {value!r}; choose {choices}, or custom:<GiB>")
