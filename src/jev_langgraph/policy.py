"""Confidence-band policies: what the runtime does with a JEV answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PolicyAction = Literal["auto", "provisional", "review", "fallback"]


@dataclass(frozen=True)
class Policy:
    """Confidence bands, evaluated top-down.

    >= auto       : take argmax, receipt tagged `auto`
    >= provisional: take argmax, receipt tagged `provisional` (downstream
                    nodes may read the tag and add verification)
    >= review     : interrupt for a human; human sees the full top-k
    <  review     : fall back (route to `fallback_node`, e.g. an LLM)
    """

    auto: float = 0.95
    provisional: float = 0.7
    review: float = 0.4
    version: str = "default-v1"
    metric: Literal["confidence", "probability"] = "confidence"

    def __post_init__(self):
        if not 0 <= self.review <= self.provisional <= self.auto <= 1:
            raise ValueError("Require 0 <= review <= provisional <= auto <= 1")
        if not self.version or self.metric not in ("confidence", "probability"):
            raise ValueError("Policy requires a version and a supported metric")

    def action_for(self, confidence: float) -> PolicyAction:
        if confidence >= self.auto:
            return "auto"
        if confidence >= self.provisional:
            return "provisional"
        if confidence >= self.review:
            return "review"
        return "fallback"
