"""Bounded, evidence-linked summaries of opt-in task receipts."""

from collections import defaultdict
from typing import Any

from .errors import EvolutionError


def summarize(receipts: list[dict[str, Any]], *, min_occurrences: int = 2) -> dict[str, Any]:
    """Group repeated observations without claiming they prove a root cause."""
    if isinstance(min_occurrences, bool) or not isinstance(min_occurrences, int) or min_occurrences < 1:
        raise EvolutionError("invalid_threshold", "min_occurrences must be a positive integer")
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: {
        "tasks": set(), "versions": set(), "receipts": set(), "evidence_refs": set()
    })
    counts = {"accepted": 0, "failed": 0, "unknown": 0}
    for receipt in receipts:
        outcome = receipt["outcome"]
        counts[outcome] += 1
        if outcome != "failed":
            continue
        for tag in receipt["failure_tags"]:
            group = groups[tag]
            group["tasks"].add(receipt["task_id"])
            group["versions"].add(receipt["snapshot_id"])
            group["receipts"].add(receipt["id"])
            group["evidence_refs"].update(receipt["evidence_refs"])
    opportunities = [
        {
            "failure_tag": tag,
            "distinct_tasks": len(group["tasks"]),
            "snapshot_ids": sorted(group["versions"]),
            "receipt_ids": sorted(group["receipts"]),
            "evidence_refs": sorted(group["evidence_refs"]),
            "status": "needs_investigation",
        }
        for tag, group in groups.items()
        if len(group["tasks"]) >= min_occurrences
    ]
    opportunities.sort(key=lambda item: (-item["distinct_tasks"], item["failure_tag"]))
    return {"outcome_counts": counts, "opportunities": opportunities,
            "interpretation": "Repeated failure tags are research candidates, not verified causes."}
