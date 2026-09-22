from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kvfit.dpa import DpaWeights
from kvfit.hardware import Hardware
from kvfit.models import GIB, CacheEstimate
from kvfit.systems import System


@dataclass(frozen=True)
class TopologyEstimate:
    tensor_parallel: int
    data_parallel: int
    budget_per_rank_bytes: float
    weights_per_rank_bytes: float
    kv_per_sequence_per_rank_bytes: float
    sequences_per_replica: int
    total_sequences: int
    verdict: str
    reason: str
    scale_up_domain_accelerators: int | None = None
    tensor_parallel_domains: int = 1
    cross_domain_tensor_parallel: bool = False
    fixed_cache_per_rank_bytes: float = 0
    runtime_reserve_per_rank_bytes: float = 0
    strategy: str = "tp"
    attention_data_parallel: int = 1
    attention_tensor_parallel: int | None = None
    expert_parallel: int = 1
    sequences_per_attention_replica: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "attention_data_parallel": self.attention_data_parallel,
            "attention_tensor_parallel": self.attention_tensor_parallel or self.tensor_parallel,
            "expert_parallel": self.expert_parallel,
            "sequences_per_attention_replica": (
                self.sequences_per_replica
                if self.sequences_per_attention_replica is None
                else self.sequences_per_attention_replica
            ),
            "tensor_parallel": self.tensor_parallel,
            "data_parallel": self.data_parallel,
            "budget_per_rank_gib": self.budget_per_rank_bytes / GIB,
            "weights_per_rank_gib": self.weights_per_rank_bytes / GIB,
            "kv_per_sequence_per_rank_gib": (
                self.kv_per_sequence_per_rank_bytes / GIB
                if math.isfinite(self.kv_per_sequence_per_rank_bytes)
                else None
            ),
            "sequences_per_replica": self.sequences_per_replica,
            "total_sequences": self.total_sequences,
            "verdict": self.verdict,
            "reason": self.reason,
            "fixed_cache_per_rank_gib": self.fixed_cache_per_rank_bytes / GIB,
            "runtime_reserve_per_rank_gib": self.runtime_reserve_per_rank_bytes / GIB,
            "scale_up_domain_accelerators": self.scale_up_domain_accelerators,
            "tensor_parallel_domains": self.tensor_parallel_domains,
            "cross_domain_tensor_parallel": self.cross_domain_tensor_parallel,
        }


def _candidate_tp_sizes(
    gpus: int,
    requested_tp: int | None,
    scale_up_domain_accelerators: int | None,
) -> list[int]:
    if gpus < 1:
        raise ValueError("gpus must be positive")
    if requested_tp is not None:
        if requested_tp < 1 or gpus % requested_tp:
            raise ValueError("tensor parallel size must be a positive divisor of the GPU count")
        return [requested_tp]
    candidates = [size for size in range(1, gpus + 1) if gpus % size == 0]
    if scale_up_domain_accelerators is None:
        return candidates
    # Avoid auto-suggesting awkward TP groups that cannot be packed cleanly
    # within a scale-up domain or as whole scale-up domains. Explicit TP still
    # permits such layouts and labels them as cross-domain.
    return [
        size
        for size in candidates
        if (size <= scale_up_domain_accelerators and scale_up_domain_accelerators % size == 0)
        or size % scale_up_domain_accelerators == 0
    ]


def plan_topologies(
    cache: CacheEstimate,
    *,
    weight_bytes: int,
    hardware: Hardware,
    gpus: int,
    utilization: float,
    tensor_parallel: int | None = None,
    scale_up_domain_accelerators: int | None = None,
    runtime_reserve_bytes: float = 0,
    dpa_weights: DpaWeights | None = None,
) -> tuple[TopologyEstimate, ...]:
    """Plan replicated TP groups using ideal weight sharding.

    Optional qualified DPA adds local attention groups with distributed experts.
    Data-parallel groups here remain full replicas. No PP/CP/offload is inferred.
    """
    quantities = [
        weight_bytes,
        hardware.memory_gib,
        runtime_reserve_bytes,
        *(c.bytes for c in (*cache.components, *getattr(cache, "fixed_components", ()))),
    ]
    if not all(math.isfinite(value) and value >= 0 for value in quantities):
        raise ValueError("memory, weights, cache and reserve must be finite and non-negative")
    if hardware.memory_gib <= 0:
        raise ValueError("hardware memory must be positive")
    if not 0 < utilization <= 1:
        raise ValueError("utilization must be in (0, 1]")
    if scale_up_domain_accelerators is not None:
        if scale_up_domain_accelerators < 1:
            raise ValueError("scale-up domain accelerator count must be positive")
        if gpus % scale_up_domain_accelerators:
            raise ValueError("GPU count must be a multiple of the scale-up domain size")

    budget = hardware.memory_gib * GIB * utilization
    layouts: list[TopologyEstimate] = []
    for tp in _candidate_tp_sizes(gpus, tensor_parallel, scale_up_domain_accelerators):
        dp = gpus // tp
        crosses_domain = bool(
            scale_up_domain_accelerators
            and (tp > scale_up_domain_accelerators or scale_up_domain_accelerators % tp != 0)
        )
        domains = (
            math.ceil(tp / scale_up_domain_accelerators) if scale_up_domain_accelerators else 1
        )
        if cache.query_heads % tp:
            layouts.append(
                TopologyEstimate(
                    tensor_parallel=tp,
                    data_parallel=dp,
                    budget_per_rank_bytes=budget,
                    weights_per_rank_bytes=weight_bytes / tp,
                    kv_per_sequence_per_rank_bytes=math.inf,
                    sequences_per_replica=0,
                    total_sequences=0,
                    verdict="unsupported",
                    reason=f"{cache.query_heads} query heads are not divisible by TP={tp}",
                    scale_up_domain_accelerators=scale_up_domain_accelerators,
                    tensor_parallel_domains=domains,
                    cross_domain_tensor_parallel=crosses_domain,
                )
            )
            continue

        weights_per_rank = weight_bytes / tp
        # Cache components can expose different sharding widths (for example,
        # Inkling has 8 global KV heads but 16 local KV heads). Components that
        # do not opt in retain the architecture-wide compatibility value.
        kv_per_rank = sum(
            component.bytes / min(tp, component.tp_parallel_units or cache.kv_parallel_heads)
            for component in cache.components
        )
        fixed_per_rank = sum(
            c.bytes / min(tp, c.tp_parallel_units or cache.kv_parallel_heads)
            for c in getattr(cache, "fixed_components", ())
        )
        available_for_kv = budget - weights_per_rank - fixed_per_rank - runtime_reserve_bytes
        if weights_per_rank > budget:
            sequences = 0
            verdict = "weights-oom"
            reason = "ideal weight shard exceeds the per-rank planning budget"
        else:
            sequences = max(0, math.floor(available_for_kv / kv_per_rank)) if kv_per_rank else 0
            verdict = "fits" if sequences >= 1 else "context-oom"
            reason = (
                "counted weights and full-context state fit before unmodeled runtime overhead"
                if sequences >= 1
                else "weights fit, but cache plus fixed state and runtime reserve exceed budget"
            )
        layouts.append(
            TopologyEstimate(
                tensor_parallel=tp,
                data_parallel=dp,
                budget_per_rank_bytes=budget,
                weights_per_rank_bytes=weights_per_rank,
                kv_per_sequence_per_rank_bytes=kv_per_rank,
                sequences_per_replica=sequences,
                total_sequences=sequences * dp,
                verdict=verdict,
                reason=reason,
                scale_up_domain_accelerators=scale_up_domain_accelerators,
                tensor_parallel_domains=domains,
                cross_domain_tensor_parallel=crosses_domain,
                fixed_cache_per_rank_bytes=fixed_per_rank,
                runtime_reserve_per_rank_bytes=runtime_reserve_bytes,
            )
        )
    if dpa_weights is not None:
        if cache.architecture != "glm-dsa-mla" or cache.confidence != "upstream-storage-formula":
            raise ValueError("DPA requires a qualified GLM DSA storage profile")
        for group in _candidate_tp_sizes(gpus, tensor_parallel, scale_up_domain_accelerators):
            if scale_up_domain_accelerators and (
                group > scale_up_domain_accelerators or scale_up_domain_accelerators % group
            ):
                continue
            if dpa_weights.expert_count % group:
                continue
            for dpa in (d for d in range(2, group + 1) if group % d == 0):
                attn_tp = group // dpa
                if cache.query_heads % attn_tp:
                    continue
                weights = dpa_weights.routed_expert_bytes / group + dpa_weights.replicated_bytes
                state = sum(
                    c.bytes / min(attn_tp, c.tp_parallel_units or cache.kv_parallel_heads)
                    for c in cache.components
                )
                fixed = sum(
                    c.bytes / min(attn_tp, c.tp_parallel_units or cache.kv_parallel_heads)
                    for c in getattr(cache, "fixed_components", ())
                )
                available = budget - weights - fixed - runtime_reserve_bytes
                sequences = max(0, math.floor(available / state)) if state else 0
                verdict = (
                    "fits" if sequences else "weights-oom" if weights > budget else "context-oom"
                )
                layouts.append(
                    TopologyEstimate(
                        tensor_parallel=group,
                        data_parallel=gpus // group,
                        budget_per_rank_bytes=budget,
                        weights_per_rank_bytes=weights,
                        kv_per_sequence_per_rank_bytes=state,
                        sequences_per_replica=sequences * dpa,
                        total_sequences=sequences * dpa * (gpus // group),
                        verdict=verdict,
                        reason=(
                            "SGLang DPA checkpoint-placement envelope; all non-routed tensors "
                            "replicated; runtime allocations and performance unmeasured"
                        ),
                        scale_up_domain_accelerators=scale_up_domain_accelerators,
                        fixed_cache_per_rank_bytes=fixed,
                        runtime_reserve_per_rank_bytes=runtime_reserve_bytes,
                        strategy="sglang-dpa",
                        attention_data_parallel=dpa,
                        attention_tensor_parallel=attn_tp,
                        expert_parallel=group,
                        sequences_per_attention_replica=sequences,
                    )
                )
    return tuple(layouts)


def summarize_concurrency(
    layouts: tuple[TopologyEstimate, ...],
    *,
    active_sequences_per_user: int,
) -> dict[str, Any]:
    """Summarize the memory ceiling for one or more active sequences per user."""
    if active_sequences_per_user < 1:
        raise ValueError("active sequences per user must be positive")
    fitting = [layout for layout in layouts if layout.verdict == "fits"]
    node_local = [layout for layout in fitting if not layout.cross_domain_tensor_parallel]

    def best(candidates: list[TopologyEstimate]) -> TopologyEstimate | None:
        return max(
            candidates,
            key=lambda layout: (
                layout.total_sequences,
                -layout.tensor_parallel_domains,
                -layout.tensor_parallel,
            ),
            default=None,
        )

    best_any = best(fitting)
    best_local = best(node_local)
    recommended = best_local or best_any

    def capacity(layout: TopologyEstimate | None) -> dict[str, Any] | None:
        if layout is None:
            return None
        return {
            "active_sequences": layout.total_sequences,
            "concurrent_users": layout.total_sequences // active_sequences_per_user,
            "layout": layout.as_dict(),
        }

    return {
        "qualification": "memory-only",
        "definition": (
            "One concurrent user means the configured number of simultaneously resident "
            "full-context sequences. This is a static counted-state upper bound before "
            "runtime overhead, not an OOM guarantee or a throughput or latency SLO."
        ),
        "active_sequences_per_user": active_sequences_per_user,
        "recommended": capacity(recommended),
        "best_scale_up_local": capacity(best_local),
        "best_any_fabric": capacity(best_any),
    }


def search_systems(
    cache: CacheEstimate,
    *,
    weight_bytes: int,
    targets: list[tuple[System, Hardware]],
    max_nodes: int,
    concurrent_users: int,
    active_sequences_per_user: int,
    utilization: float,
    runtime_reserve_bytes: float = 0,
    dpa_weights: DpaWeights | None = None,
) -> dict[str, Any]:
    """First sufficient whole-system count per family, with node-local TP only."""
    if max_nodes < 1 or concurrent_users < 1 or active_sequences_per_user < 1:
        raise ValueError("search horizon and user/sequence counts must be positive")
    minimums, attempts, not_found = [], [], []
    for system, hardware in targets:
        for nodes in range(1, max_nodes + 1):
            layouts = plan_topologies(
                cache,
                weight_bytes=weight_bytes,
                hardware=hardware,
                gpus=nodes * system.accelerators_per_system,
                utilization=utilization,
                scale_up_domain_accelerators=system.scale_up_domain_accelerators,
                runtime_reserve_bytes=runtime_reserve_bytes,
                dpa_weights=dpa_weights,
            )
            local = summarize_concurrency(
                layouts,
                active_sequences_per_user=active_sequences_per_user,
            )["best_scale_up_local"]
            row = {
                "system": system.as_dict(),
                "hardware": hardware.as_dict(),
                "nodes": nodes,
                "gpus": nodes * system.accelerators_per_system,
                "capacity": local,
                "meets_target": local is not None and local["concurrent_users"] >= concurrent_users,
                "layouts": [layout.as_dict() for layout in layouts],
            }
            attempts.append(row)
            if row["meets_target"]:
                minimums.append(row)
                break
        else:
            not_found.append(system.id)
    return {
        "qualification": "memory-only",
        "scope": (
            "best whole-system count among evaluated local TP and SGLang DP-Attention/EP layouts"
            if dpa_weights
            else "minimum whole-system count per selected family with scale-up-local TP"
        ),
        "evaluated_strategies": ["tp", "sglang-dpa"] if dpa_weights else ["tp"],
        "max_nodes": max_nodes,
        "concurrent_users": concurrent_users,
        "active_sequences_per_user": active_sequences_per_user,
        "resident_sequences": concurrent_users * active_sequences_per_user,
        "minimums": minimums,
        "attempts": attempts,
        "not_found": not_found,
        "exclusions": [
            *([] if dpa_weights else ["DP-Attention and expert parallelism"]),
            "cross-domain TP",
            "cross-domain EP, pipeline/context parallelism",
            "CPU offload",
            "heterogeneous or partial servers",
            "cost",
            "runtime performance",
        ],
    }
