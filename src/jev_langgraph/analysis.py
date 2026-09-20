"""Receipt analysis: calibration, flip-rate, and per-node audit."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .receipt import Receipt


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
