"""Direct TypeSafe /v1/systemone client. HTTP errors propagate to the caller."""

from __future__ import annotations

import json
import math
import time
import urllib.request
from typing import Sequence
from urllib.parse import urlsplit

from .client import (
    Answer,
    Choice,
    DecideResponse,
    Noul,
    Question,
    Score,
    StructuredValue,
    json_value,
    validate_answer,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpJevClient:
    def __init__(
        self, base_url: str, api_key: str, timeout: float = 10.0, model: str = "jev-latest"
    ):
        url = urlsplit(base_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("JEV base_url must be an HTTPS URL without credentials/query/fragment")
        if not api_key.strip() or not model.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("JEV requires credentials, model, and a positive finite timeout")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.model = model
        self._opener = urllib.request.build_opener(_NoRedirect())

    def decide(self, state_text: StructuredValue, questions: Sequence[Question]) -> DecideResponse:
        if not isinstance(state_text, (str, dict, list)):
            raise ValueError("JEV state must be a string, object, or array")
        state_text = json_value(state_text)
        if not questions or len({q.id for q in questions}) != len(questions):
            raise ValueError("Questions must be nonempty with unique IDs")
        wire_questions = {}
        for q in questions:
            if (
                not isinstance(q.prompt, (str, dict, list))
                or not q.prompt
                or (isinstance(q.prompt, str) and not q.prompt.strip())
            ):
                raise ValueError(f"Question {q.id} requires explicit instructions")
            wire = {"type": q.kind, "instructions": json_value(q.prompt)}
            if isinstance(q, Choice):
                if q.criteria is not None and set(q.criteria) != set(q.options):
                    raise ValueError(f"Choice criteria must match options for {q.id}")
                wire["criteria"] = json_value(
                    q.criteria if q.criteria is not None else {label: label for label in q.options}
                )
            elif isinstance(q, Score):
                if q.criteria is not None and len(q.criteria) != len(q.levels):
                    raise ValueError(f"Score criteria must match levels for {q.id}")
                wire["criteria"] = json_value(
                    q.criteria if q.criteria is not None else list(q.levels)
                )
            elif q.criteria is not None:
                if set(q.criteria) - {"true", "false"}:
                    raise ValueError("Noul criteria supports true and false")
                wire["criteria"] = json_value(q.criteria)
            wire_questions[q.id] = wire
        body = {"state": state_text, "model": self.model, "questions": wire_questions}
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(body, allow_nan=False).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.perf_counter()
        with self._opener.open(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
            request_id = getattr(resp, "headers", {}).get("x-request-id")
        if set(payload["answers"]) != {q.id for q in questions}:
            raise ValueError("Provider answer IDs do not match questions")
        answers = {}
        for q in questions:
            raw = payload["answers"][q.id]
            score = None
            if raw["type"] != q.kind:
                raise ValueError(f"Provider answer type mismatch for {q.id}")
            if isinstance(q, Noul):
                dist = {"yes": raw["noul"], "no": 1 - raw["noul"]}
                confidence = None
            elif isinstance(q, Score):
                expected_keys = {str(i) for i in range(len(q.levels))}
                if set(raw["probabilities"]) != expected_keys:
                    raise ValueError(f"Score levels mismatch for {q.id}")
                dist = {label: raw["probabilities"][str(i)] for i, label in enumerate(q.levels)}
                score = raw["score"]
                expected_score = sum(i * dist[label] for i, label in enumerate(q.levels))
                if not math.isclose(score, expected_score, abs_tol=1e-5):
                    raise ValueError(f"Score expectation mismatch for {q.id}")
                confidence = raw["confidence"]
            else:
                dist, confidence = raw["probabilities"], raw["confidence"]
            value = raw["choice"] if isinstance(q, Choice) else max(dist, key=dist.get)
            answer = Answer(q.id, q.kind, dist, value, confidence, score)
            validate_answer(q, answer)
            answers[q.id] = answer
        return DecideResponse(
            answers=answers,
            latency_ms=round((time.perf_counter() - started) * 1000),
            raw=payload,
            request_id=payload.get("request_id") or request_id,
        )
