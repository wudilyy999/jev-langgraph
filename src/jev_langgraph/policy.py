"""Confidence-band policies: what the runtime does with a JEV answer."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    auto_enabled: bool = True
    adaptation: dict = field(default_factory=dict)
    token_budget: int | None = None
    budget_auto: float = 1.0

    def __post_init__(self):
        if not 0 <= self.review <= self.provisional <= self.auto <= 1:
            raise ValueError("Require 0 <= review <= provisional <= auto <= 1")
        if not self.version or self.metric not in ("confidence", "probability"):
            raise ValueError("Policy requires a version and a supported metric")
        if self.token_budget is not None and (
            type(self.token_budget) is not int or self.token_budget < 0
        ):
            raise ValueError("token_budget must be a nonnegative integer")
        if not self.auto <= self.budget_auto <= 1:
            raise ValueError("budget_auto must be between auto and one")

    def action_for(self, confidence: float, *, budget_exceeded: bool = False) -> PolicyAction:
        threshold = self.budget_auto if budget_exceeded else self.auto
        if self.auto_enabled and confidence >= threshold:
            return "auto"
        if budget_exceeded or not self.auto_enabled:
            return "review" if confidence >= self.review else "fallback"
        if confidence >= self.provisional:
            return "provisional"
        if confidence >= self.review:
            return "review"
        return "fallback"

    def adapt(
        self,
        receipts,
        outcomes: dict[str, bool],
        *,
        node: str,
        question_id: str,
        model: str,
        version: str,
        target_accuracy: float = 0.95,
        min_samples: int = 30,
    ):
        """Derive a new auto threshold from labeled tail accuracy (empirical, not a guarantee).

        Only raises thresholds. Refit explicitly between runs; never mutates a live policy.
        No supported threshold disables autonomous execution for this policy.
        """
        if not 0 < target_accuracy <= 1 or type(min_samples) is not int or min_samples < 1:
            raise ValueError("Require accuracy in (0, 1] and positive min_samples")
        if not version or version == self.version or not node or not question_id or not model:
            raise ValueError("Adaptation needs a new version and explicit domain/question/model")
        records = {r.id: r for r in receipts}
        labeled = []
        for r in records.values():
            if (r.node, r.question_id, r.model) != (
                node,
                question_id,
                model,
            ) or r.id not in outcomes:
                continue
            if type(outcomes[r.id]) is not bool:
                raise ValueError("Outcomes must label correctness of the original model prediction")
            value = r.confidence if self.metric == "confidence" else r.chosen_probability
            if value is None:
                raise ValueError("Use probability metric to adapt Noul policies")
            labeled.append((value, outcomes[r.id]))
        candidates = sorted({self.auto} | {v for v, _ in labeled if v >= self.auto})
        chosen = None
        support = []
        for threshold in candidates:
            tail = [correct for value, correct in labeled if value >= threshold]
            if len(tail) >= min_samples and sum(tail) / len(tail) >= target_accuracy:
                chosen, support = threshold, tail
                break
        evidence = {
            "node": node,
            "question_id": question_id,
            "model": model,
            "labeled": len(labeled),
            "target_accuracy": target_accuracy,
            "min_samples": min_samples,
            "support": len(support),
            "observed_accuracy": sum(support) / len(support) if support else None,
            "previous_version": self.version,
        }
        return replace(
            self,
            auto=chosen if chosen is not None else self.auto,
            auto_enabled=chosen is not None,
            version=version,
            adaptation=evidence,
            budget_auto=max(self.budget_auto, chosen or self.auto),
        )
