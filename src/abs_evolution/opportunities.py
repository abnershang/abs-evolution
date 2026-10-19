"""Bounded, evidence-linked summaries of opt-in task receipts."""

from collections import defaultdict
from typing import Any

from .errors import EvolutionError


def summarize(receipts: list[dict[str, Any]], *, min_occurrences: int = 2) -> dict[str, Any]:
    """Group repeated observations as comparison hypotheses, never as causal claims."""
    if isinstance(min_occurrences, bool) or not isinstance(min_occurrences, int) or min_occurrences < 1:
        raise EvolutionError("invalid_threshold", "min_occurrences must be a positive integer")
    groups: dict[tuple[str, str | None, str | None, str], dict[str, set[str]]] = defaultdict(lambda: {
        "tasks": set(), "versions": set(), "receipts": set(), "evidence_refs": set(), "outcomes": set()
    })
    counts = {"accepted": 0, "failed": 0, "unknown": 0}
    for receipt in receipts:
        outcome = receipt["outcome"]
        counts[outcome] += 1
        family = receipt.get("task_family")

        def observe(signal_type: str, dimension: str | None, tag: str) -> None:
            key = (signal_type, family, dimension, tag)
            group = groups[key]
            group["tasks"].add(receipt["task_id"])
            group["versions"].add(receipt["snapshot_id"])
            group["receipts"].add(receipt["id"])
            group["evidence_refs"].update(receipt["evidence_refs"])
            group["outcomes"].add(outcome)

        if outcome == "failed":
            for tag in receipt["failure_tags"]:
                observe("failure", None, tag)
        for signal in receipt.get("optimization_signals", []):
            observe("optimization", signal["dimension"], signal["tag"])

    opportunities = []
    for (signal_type, family, dimension, tag), group in groups.items():
        if len(group["tasks"]) < min_occurrences:
            continue
        item = {
            "signal_type": signal_type,
            "task_family": family,
            "dimension": dimension,
            "tag": tag,
            "distinct_tasks": len(group["tasks"]),
            "snapshot_ids": sorted(group["versions"]),
            "receipt_ids": sorted(group["receipts"]),
            "evidence_refs": sorted(group["evidence_refs"]),
            "observed_outcomes": sorted(group["outcomes"]),
            "status": "needs_comparison" if signal_type == "optimization" else "needs_investigation",
        }
        if signal_type == "failure":
            item["failure_tag"] = tag
        opportunities.append(item)
    opportunities.sort(key=lambda item: (
        -item["distinct_tasks"], item["task_family"] or "", item["signal_type"],
        item["dimension"] or "", item["tag"],
    ))
    return {"outcome_counts": counts, "opportunities": opportunities,
            "interpretation": (
                "Observed signals are hypotheses requiring baseline/candidate comparisons; they do not "
                "establish gains or causes."
            )}
