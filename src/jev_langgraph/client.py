"""Typed requests for JEV and an offline client."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, Union, runtime_checkable

from langchain_core.messages import BaseMessage

StructuredValue = Union[str, dict[str, Any], list[Any]]


def json_value(value: Any) -> Any:
    """Convert messages recursively, preserving structured state and JSON types."""
    if isinstance(value, BaseMessage):
        role = {"human": "user", "ai": "assistant"}.get(value.type, value.type)
        return {"role": role, "content": json_value(value.content)}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("JSON object keys must be strings")
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported JSON value: {type(value).__name__}")


def structured_state(state: Any) -> StructuredValue:
    """Select user state without framework bookkeeping; keep its JSON structure."""
    if isinstance(state, dict):
        if any(not isinstance(k, str) for k in state):
            raise ValueError("JSON object keys must be strings")
        state = {k: v for k, v in state.items() if k != "receipts" and not k.startswith("__jev_")}
    result = json_value(state)
    if not isinstance(result, (str, dict, list)):
        raise ValueError("JEV state must be a string, object, or array")
    return result


@dataclass(frozen=True)
class Choice:
    """Pick one from a fixed set of options."""

    id: str
    options: tuple[str, ...]
    prompt: StructuredValue = ""
    kind: Literal["choice"] = "choice"
    criteria: dict[str, Any] | None = None


@dataclass(frozen=True)
class Score:
    """Rate the state against ordered levels."""

    id: str
    levels: tuple[str, ...]
    prompt: StructuredValue = ""
    kind: Literal["score"] = "score"
    criteria: list[Any] | None = None


@dataclass(frozen=True)
class Noul:
    """Yes/no question about the state."""

    id: str
    prompt: StructuredValue = ""
    kind: Literal["noul"] = "noul"
    criteria: dict[str, Any] | None = None


Question = Union[Choice, Score, Noul]


@dataclass
class Answer:
    question_id: str
    kind: str
    distribution: dict[str, float]
    value: str
    confidence: float | None
    score: float | None = None


def options_for(q: Question) -> tuple[str, ...]:
    if isinstance(q, Choice):
        return tuple(q.options)
    if isinstance(q, Score):
        return tuple(q.levels)
    return ("yes", "no")


def validate_answer(q: Question, a: Answer) -> None:
    """Validate the provider boundary before scheduling business work."""
    if a.question_id != q.id or a.kind != q.kind:
        raise ValueError(f"Answer id/type mismatch for {q.id}")
    if set(a.distribution) != set(options_for(q)):
        raise ValueError(f"Distribution options mismatch for {q.id}")
    values = a.distribution.values()
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError(f"Invalid probability for {q.id}")
    if not math.isclose(sum(values), 1, abs_tol=1e-5):
        raise ValueError(f"Probabilities must sum to one for {q.id}")
    if a.value not in a.distribution or a.distribution[a.value] != max(values):
        raise ValueError(f"Answer must select a highest-probability option for {q.id}")
    if a.confidence is not None and not 0 <= a.confidence <= 1:
        raise ValueError(f"Invalid confidence for {q.id}")
    if a.kind != "noul" and a.confidence is None:
        raise ValueError(f"Missing confidence for {q.id}")


@dataclass
class DecideResponse:
    answers: dict[str, Answer]
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self):
        # Accept older custom clients' raw metadata while exposing one canonical contract.
        if self.model is None:
            self.model = self.raw.get("model")
        if self.request_id is None:
            self.request_id = self.raw.get("request_id")
        usage = self.raw.get("usage", {})
        if self.input_tokens is None:
            self.input_tokens = usage.get("input_tokens")
        if self.output_tokens is None:
            self.output_tokens = usage.get("output_tokens")
        for value in (self.input_tokens, self.output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Token counts must be nonnegative integers or None")
        for value in (self.model, self.request_id):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("Model and request_id must be nonempty strings or None")


@runtime_checkable
class JevClient(Protocol):
    """All JEV calls in the library go through this single seam."""

    def decide(
        self, state_text: StructuredValue, questions: Sequence[Question]
    ) -> DecideResponse: ...


def state_to_text(state: Any) -> str:
    """Explicit legacy text selector; new decisions default to structured_state."""
    value = structured_state(state)
    return (
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


class MockJevClient:
    """Deterministic client for tests: rule table keyed by question id.

    rules[question_id] = (distribution, confidence). Missing rules fail fast.
    """

    def __init__(self, rules: dict[str, tuple[dict[str, float], float]] | None = None):
        self.rules = rules or {}
        self.calls: list[tuple[StructuredValue, list[str]]] = []

    def decide(self, state_text: StructuredValue, questions: Sequence[Question]) -> DecideResponse:
        self.calls.append((state_text, [q.id for q in questions]))
        answers: dict[str, Answer] = {}
        for q in questions:
            dist, conf = self.rules[q.id]
            top = max(dist, key=dist.get)
            answers[q.id] = Answer(
                question_id=q.id,
                kind=q.kind,
                distribution=dict(dist),
                value=top,
                confidence=conf,
            )
        return DecideResponse(answers=answers, latency_ms=1)
