"""Receipt analysis: calibration, flip-rate, and per-node audit."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .receipt import Receipt


def cost_report(
    receipts: Sequence[Receipt],
    *,
    group_by: str = "node",
    thread_id: str | None = None,
    run_id: str | None = None,
    input_per_million: float = 0,
    output_per_million: float = 0,
) -> list[dict]:
    """Aggregate measured requests, deduplicating repeated per-question usage.

    A caller supplies prices in one currency. Unknown usage produces unknown cost.
    run_id is the caller's configurable.jev_run_id and survives resume.
    """
    if group_by not in ("node", "thread_id", "run_id", "model"):
        raise ValueError("group_by must be node, thread_id, run_id, or model")
    if any(not math.isfinite(p) or p < 0 for p in (input_per_million, output_per_million)):
        raise ValueError("Prices must be finite and nonnegative")
    groups = {}
    for r in receipts:
        if thread_id is not None and r.thread_id != thread_id:
            continue
        if run_id is not None and r.run_id != run_id:
            continue
        key = (r.metering_node or r.node) if group_by == "node" else getattr(r, group_by)
        calls = groups.setdefault(key, {})
        call_id = r.call_id or r.invocation_id or r.id
        counts = (r.input_tokens, r.output_tokens)
        if call_id in calls and calls[call_id] != counts:
            raise ValueError(f"Inconsistent token usage for call {call_id}")
        calls[call_id] = counts
    rows = []
    for key, calls in groups.items():
        inputs = [c[0] for c in calls.values()]
        outputs = [c[1] for c in calls.values()]
        in_total = sum(inputs) if all(c is not None for c in inputs) else None
        out_total = sum(outputs) if all(c is not None for c in outputs) else None
        rows.append(
            {
                group_by: key,
                "requests": len(calls),
                "input_tokens": in_total,
                "output_tokens": out_total,
                "unmetered_requests": sum(i is None or o is None for i, o in calls.values()),
                "cost": (in_total * input_per_million + out_total * output_per_million) / 1_000_000
                if in_total is not None and out_total is not None
                else None,
            }
        )
    return rows


def calibration_by_model(
    receipts: Sequence[Receipt], outcomes: dict[str, bool], bins: int = 5
) -> list[dict]:
    """Compare selected-label calibration for observed model versions."""
    groups = defaultdict(list)
    for r in receipts:
        if r.id in outcomes:
            groups[r.model].append(r)
    return [
        {
            "model": model,
            "labeled_decisions": len(records),
            "ece": expected_calibration_error(records, outcomes, bins),
            "table": calibration_table(records, outcomes, bins),
        }
        for model, records in groups.items()
    ]


def flip_rate(receipts: Sequence[Receipt], margin_threshold: float = 0.1) -> float:
    """Fraction of decisions that were effectively coin flips.

    A high flip rate at a decision node means the node is doing little
    useful discrimination; consider removing it or retraining.
    """
    if not receipts:
        return 0.0
    flips = sum(1 for r in receipts if r.margin < margin_threshold)
    return flips / len(receipts)


def policy_histogram(receipts: Sequence[Receipt]) -> dict[str, int]:
    """How often each policy band fired."""
    h: dict[str, int] = defaultdict(int)
    for r in receipts:
        h[r.policy] += 1
    return dict(h)


def calibration_table(
    receipts: Sequence[Receipt],
    outcomes: dict[str, bool],
    bins: int = 5,
) -> list[dict]:
    """Reliability table: selected-label probability vs observed accuracy.

    `outcomes` maps receipt id -> whether the chosen route turned out to
    be correct (as judged downstream, e.g. by a human or a later JEV
    verification pass). Rows with no outcome are skipped.
    """
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for r in receipts:
        if r.id not in outcomes:
            continue
        p = r.chosen_probability
        i = min(int(p * bins), bins - 1)
        buckets[i].append((p, int(outcomes[r.id])))
    table = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        table.append(
            {
                "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                "n": len(b),
                "mean_probability": sum(c for c, _ in b) / len(b),
                "accuracy": sum(a for _, a in b) / len(b),
            }
        )
    return table


def expected_calibration_error(
    receipts: Sequence[Receipt], outcomes: dict[str, bool], bins: int = 5
) -> float:
    """Scalar ECE over the calibration table."""
    table = calibration_table(receipts, outcomes, bins)
    total = sum(row["n"] for row in table)
    if total == 0:
        return 0.0
    return sum(row["n"] * abs(row["mean_probability"] - row["accuracy"]) for row in table) / total
