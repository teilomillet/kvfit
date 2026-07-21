from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class System:
    id: str
    label: str
    hardware_id: str
    accelerators_per_system: int
    scale_up_domain_accelerators: int
    unit_name: str
    scale_up_fabric: str
    scale_out_fabric: str
    source_url: str
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "hardware_id": self.hardware_id,
            "accelerators_per_system": self.accelerators_per_system,
            "scale_up_domain_accelerators": self.scale_up_domain_accelerators,
            "unit_name": self.unit_name,
            "scale_up_fabric": self.scale_up_fabric,
            "scale_out_fabric": self.scale_out_fabric,
            "source_url": self.source_url,
            "note": self.note,
        }


DGX_HOPPER_SPECS = "https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html"
DGX_B200_SPECS = "https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html"
DGX_B300_SPECS = "https://docs.nvidia.com/dgx/dgxb300-user-guide/index.html"
DGX_GB_RACK_SPECS = "https://docs.nvidia.com/dgx/dgxgb200-user-guide/hardware.html"
DGX_SPARK_SPECS = "https://docs.nvidia.com/dgx/dgx-spark/hardware.html"


SYSTEMS: dict[str, System] = {
    "dgx-spark": System(
        id="dgx-spark",
        label="NVIDIA DGX Spark",
        hardware_id="dgx-spark",
        accelerators_per_system=1,
        scale_up_domain_accelerators=1,
        unit_name="system",
        scale_up_fabric="single GB10 unified-memory system",
        scale_out_fabric="ConnectX-7 / NCCL between systems",
        source_url=DGX_SPARK_SPECS,
        note=(
            "Additional Sparks are separate memory domains. Capacity scales only when the "
            "serving engine supports the selected distributed layout; it is not one pooled "
            "unified-memory address space."
        ),
    ),
    "dgx-h100": System(
        id="dgx-h100",
        label="NVIDIA DGX H100",
        hardware_id="h100-80",
        accelerators_per_system=8,
        scale_up_domain_accelerators=8,
        unit_name="node",
        scale_up_fabric="8-GPU fourth-generation NVLink/NVSwitch domain",
        scale_out_fabric="ConnectX-7 InfiniBand/Ethernet between nodes",
        source_url=DGX_HOPPER_SPECS,
        note=(
            "Tensor parallelism up to eight GPUs stays inside one DGX. Larger TP crosses the "
            "cluster fabric and is a memory-only hypothesis until measured on the target fabric."
        ),
    ),
    "dgx-h200": System(
        id="dgx-h200",
        label="NVIDIA DGX H200",
        hardware_id="h200-141",
        accelerators_per_system=8,
        scale_up_domain_accelerators=8,
        unit_name="node",
        scale_up_fabric="8-GPU fourth-generation NVLink/NVSwitch domain",
        scale_out_fabric="ConnectX-7 InfiniBand/Ethernet between nodes",
        source_url=DGX_HOPPER_SPECS,
        note=(
            "Tensor parallelism up to eight GPUs stays inside one DGX. Larger TP crosses the "
            "cluster fabric and is a memory-only hypothesis until measured on the target fabric."
        ),
    ),
    "dgx-b200": System(
        id="dgx-b200",
        label="NVIDIA DGX B200",
        hardware_id="b200-180",
        accelerators_per_system=8,
        scale_up_domain_accelerators=8,
        unit_name="node",
        scale_up_fabric="8-GPU fifth-generation NVLink/NVSwitch domain",
        scale_out_fabric="ConnectX-7 InfiniBand/Ethernet between nodes",
        source_url=DGX_B200_SPECS,
        note=(
            "Tensor parallelism up to eight GPUs stays inside one DGX. Larger TP crosses the "
            "cluster fabric and is a memory-only hypothesis until measured on the target fabric."
        ),
    ),
    "dgx-b300": System(
        id="dgx-b300",
        label="NVIDIA DGX B300",
        hardware_id="b300-288",
        accelerators_per_system=8,
        scale_up_domain_accelerators=8,
        unit_name="node",
        scale_up_fabric="8-GPU fifth-generation NVLink/NVSwitch domain",
        scale_out_fabric="ConnectX-8 InfiniBand/Ethernet between nodes",
        source_url=DGX_B300_SPECS,
        note=(
            "Tensor parallelism up to eight GPUs stays inside one DGX. Larger TP crosses the "
            "cluster fabric and is a memory-only hypothesis until measured on the target fabric."
        ),
    ),
    "dgx-gb200-nvl72": System(
        id="dgx-gb200-nvl72",
        label="NVIDIA DGX GB200 NVL72 rack",
        hardware_id="gb200-186",
        accelerators_per_system=72,
        scale_up_domain_accelerators=72,
        unit_name="rack",
        scale_up_fabric="72-GPU fifth-generation NVLink/NVSwitch domain",
        scale_out_fabric="InfiniBand/Ethernet between racks",
        source_url=DGX_GB_RACK_SPECS,
        note=(
            "A rack is one 72-GPU NVLink domain spanning multiple OS nodes. TP beyond 72 GPUs "
            "crosses the rack-scale fabric and requires target-engine validation."
        ),
    ),
}

ALIASES = {
    "spark": "dgx-spark",
    "h100": "dgx-h100",
    "h200": "dgx-h200",
    "b200": "dgx-b200",
    "b300": "dgx-b300",
    "gb200": "dgx-gb200-nvl72",
    "gb200-nvl72": "dgx-gb200-nvl72",
}


def parse_system(value: str) -> System:
    normalized = value.strip().lower()
    normalized = ALIASES.get(normalized, normalized)
    if normalized in SYSTEMS:
        return SYSTEMS[normalized]
    choices = ", ".join(sorted(SYSTEMS))
    raise ValueError(f"unknown system {value!r}; choose {choices}")
