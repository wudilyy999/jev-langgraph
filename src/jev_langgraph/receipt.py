"""Decision receipts: probabilistic provenance stored in graph state."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Receipt(BaseModel):
    """One JEV decision, recorded with the full distribution.

    The full distribution (not just the winner) is what makes post-hoc
    calibration analysis possible: a decision won 0.51 vs 0.49 is a coin
    flip and should be auditable as such.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    invocation_id: str = ""
    node: str
    question_id: str
    kind: Literal["choice", "score", "noul"]
    distribution: dict[str, float]
    chosen: str
    confidence: Optional[float]
    policy: str
    latency_ms: int
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    policy_version: str = "default-v1"
    policy_metric: str = "confidence"
    selected: list[str] = Field(default_factory=list)
    destinations: list[str] = Field(default_factory=list)
    status: Literal["pending", "dispatched"] = "pending"
    human_choice: Optional[str] = None
    reviewer: Optional[str] = None
    reason: Optional[str] = None
    score: Optional[float] = None

    @property
    def chosen_probability(self) -> float:
        return self.distribution[self.chosen]

    @property
    def margin(self) -> float:
        """Gap between top-2 probability mass. Near zero means a coin flip."""
        vals = sorted(self.distribution.values(), reverse=True)
        if len(vals) < 2:
            return 1.0
        return vals[0] - vals[1]


def add_receipts(left: list[dict], right: list[dict]) -> list[dict]:
    """Merge stable IDs so dispatch and resume update the original decision."""
    merged = {r["id"]: r for r in left}
    merged.update({r["id"]: r for r in right})
    return list(merged.values())
