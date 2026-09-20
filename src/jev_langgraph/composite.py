"""Deterministic, normalized aggregation of ordinal question distributions."""

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .receipt import Receipt


@dataclass(frozen=True)
class CompositeScore:
    weights: Mapping[str, float]

    def __post_init__(self):
        if (
            not self.weights
            or any(not math.isfinite(w) or w < 0 for w in self.weights.values())
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError("Composite weights must be nonnegative, finite, with positive total")

    def evaluate(self, receipts: Sequence[Receipt]) -> float:
        """Weighted mean of expected level / (number of levels - 1), in [0, 1]."""
        indexed = {r.question_id: r for r in receipts}
        if len(indexed) != len(receipts):
            raise ValueError("Composite requires one receipt per question from one batch")
        if len({r.invocation_id for r in receipts}) > 1:
            raise ValueError("Composite requires a single decision batch")
        total = 0.0
        for question_id, weight in self.weights.items():
            r = indexed[question_id]
            if r.kind != "score" or r.score is None or len(r.distribution) < 2:
                raise ValueError(f"Composite requires ordinal Score receipt: {question_id}")
            normalized = r.score / (len(r.distribution) - 1)
            if not math.isfinite(normalized) or not 0 <= normalized <= 1:
                raise ValueError(f"Score out of range for {question_id}")
            total += weight * normalized
        return total / sum(self.weights.values())
