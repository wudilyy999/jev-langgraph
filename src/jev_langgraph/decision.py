"""DecisionNode: one decision domain, many JEV questions, one request."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .client import JevClient, Question, options_for, state_to_text, validate_answer
from .policy import Policy
from .receipt import Receipt

FALLBACK = "__jev_fallback__"
REVIEW = "__jev_review__"


@dataclass(frozen=True)
class ProbEdge:
    routes: Mapping[str, str]
    top_k: int = 1
    policy: Policy | None = None

    def __post_init__(self):
        if not self.routes or type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("ProbEdge requires routes and a positive integer top_k")


@dataclass
class DecisionNode:
    """A JEV-native decision point.

    All questions are evaluated in ONE JEV request (JEV parallelizes
    internally), then each question's answer drives its own routing key.
    `routes` maps a routing key (usually the choice/score value) to a
    node name in the graph. `fallback` receives traffic whose confidence
    is below the policy's review band.
    """

    questions: Sequence[Question]
    routes: Mapping[str, str] = field(default_factory=dict)
    policy: Policy = field(default_factory=Policy)
    fallback: str | None = None
    review: str | None = None
    edges: Mapping[str, ProbEdge | None] | None = None
    input_selector: Callable = state_to_text

    def __post_init__(self):
        ids = [q.id for q in self.questions]
        if not ids or any(not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be nonempty and unique")
        if self.edges is not None:
            if self.routes or set(self.edges) != set(ids):
                raise ValueError(
                    "Use routes or edges; edges must include each question (None observes)"
                )
        elif not self.routes:
            raise ValueError("Decision requires routes or explicit edges")
        for q in self.questions:
            options = options_for(q)
            if (
                not options
                or len(set(options)) != len(options)
                or any(not isinstance(o, str) or not o for o in options)
            ):
                raise ValueError(f"Invalid options for {q.id}")
            if q.kind == "score" and not 2 <= len(options) <= 10:
                raise ValueError("Score requires 2 to 10 levels")
            if q.kind == "choice" and len(options) > 255:
                raise ValueError("Choice accepts at most 255 options")
            edge = self.edge_for(q.id)
            if edge and edge.top_k > len(options):
                raise ValueError(f"top_k exceeds option count for {q.id}")

    def edge_for(self, question_id: str) -> ProbEdge | None:
        return self.edges[question_id] if self.edges is not None else ProbEdge(self.routes)

    def targets(self) -> list[str]:
        targets = []
        for q in self.questions:
            edge = self.edge_for(q.id)
            if edge:
                targets.extend(edge.routes.values())
        return list(dict.fromkeys(targets + [t for t in (self.fallback, self.review) if t]))

    def evaluate(self, name: str, state, client: JevClient) -> tuple[dict[str, str], list[Receipt]]:
        """Returns ({question_id: route_key}, receipts)."""
        resp = client.decide(self.input_selector(state), self.questions)
        if set(resp.answers) != {q.id for q in self.questions}:
            raise ValueError("Provider must answer exactly the requested questions")
        invocation_id = uuid.uuid4().hex
        keys: dict[str, str] = {}
        receipts: list[Receipt] = []
        for q in self.questions:
            a = resp.answers[q.id]
            validate_answer(q, a)
            edge = self.edge_for(q.id)
            policy = (edge.policy if edge else None) or self.policy
            metric = policy.metric if a.confidence is not None else "probability"
            value = a.confidence if metric == "confidence" else a.distribution[a.value]
            action = policy.action_for(value) if edge else "observe"
            if action == "fallback" and self.fallback is None:
                raise ValueError(f"{name}/{q.id} requires a fallback target")
            selected = []
            if edge and action in ("auto", "provisional"):
                selected = sorted(a.distribution, key=lambda k: (-a.distribution[k], k != a.value))[
                    : edge.top_k
                ]
                if any(label not in edge.routes for label in selected):
                    raise ValueError(f"Unmapped answer for {q.id}: {selected}")
            if action == "fallback" and self.fallback is not None:
                key = FALLBACK
            elif action == "review" and self.review is not None:
                key = REVIEW
            else:
                key = a.value
            keys[q.id] = key
            receipts.append(
                Receipt(
                    node=name,
                    invocation_id=invocation_id,
                    question_id=q.id,
                    kind=a.kind,  # type: ignore[arg-type]
                    distribution=a.distribution,
                    chosen=a.value,
                    confidence=a.confidence,
                    policy=action,
                    latency_ms=resp.latency_ms,
                    selected=selected,
                    policy_version=policy.version,
                    policy_metric=metric,
                    score=a.score,
                    metadata={
                        k: resp.raw[k] for k in ("model", "request_id", "usage") if k in resp.raw
                    },
                )
            )
        return keys, receipts

    def mapping(self) -> dict[str, str]:
        """Static route table for LangGraph's path_map (enables graphviz)."""
        m = dict(self.routes)
        if self.fallback is not None:
            m[FALLBACK] = self.fallback
        if self.review is not None:
            m[REVIEW] = self.review
        return m
