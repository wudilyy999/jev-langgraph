import io
import json
from urllib.error import HTTPError

import pytest

from jev_langgraph import Choice, Noul, Score
from jev_langgraph.http_client import HttpJevClient


def test_official_wire_contract(monkeypatch):
    client = HttpJevClient("https://api.typesafe.ai", "test-key")

    def respond(request, timeout):
        assert request.full_url == "https://api.typesafe.ai/v1/systemone"
        assert request.get_header("Authorization") == "Bearer test-key"
        body = json.loads(request.data)
        assert body == {
            "state": "ticket",
            "model": "jev-latest",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": {"a": "a", "b": "b"},
                },
                "level": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "high"],
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
            },
        }
        return io.BytesIO(
            json.dumps(
                {
                    "model": "jev-1.13.0",
                    "answers": {
                        "route": {
                            "type": "choice",
                            "choice": "a",
                            "probabilities": {"a": 0.9, "b": 0.1},
                            "confidence": 0.8,
                        },
                        "level": {
                            "type": "score",
                            "score": 0.8,
                            "legend": {"0": "low", "1": "high"},
                            "probabilities": {"0": 0.2, "1": 0.8},
                            "confidence": 0.6,
                        },
                        "urgent": {"type": "noul", "noul": 0.1},
                    },
                    "usage": {"input_tokens": 123},
                }
            ).encode()
        )

    monkeypatch.setattr(client._opener, "open", respond)
    result = client.decide(
        "ticket",
        [
            Choice("route", ("a", "b"), "Which?"),
            Score("level", ("low", "high"), "How severe?"),
            Noul("urgent", "Urgent?"),
        ],
    )
    assert result.answers["level"].score == 0.8
    assert result.answers["urgent"].confidence is None
    assert result.answers["urgent"].value == "no"
    assert result.answers["route"].confidence == 0.8


@pytest.mark.parametrize("status", [401, 429, 529])
def test_http_errors_propagate_without_retry(monkeypatch, status):
    client = HttpJevClient("https://api.typesafe.ai", "test-key")

    def fail(*args, **kwargs):
        raise HTTPError("https://api.typesafe.ai", status, "failure", {}, None)

    monkeypatch.setattr(client._opener, "open", fail)
    with pytest.raises(HTTPError) as error:
        client.decide("input", [Noul("q", "Yes?")])
    assert error.value.code == status


@pytest.mark.parametrize("dist", [{"a": 0.8, "b": 0.8}, {"a": float("nan"), "b": 0}, {"a": 1}])
def test_malformed_distribution_rejected(monkeypatch, dist):
    client = HttpJevClient("https://api.typesafe.ai", "test-key")
    payload = {
        "answers": {
            "q": {"type": "choice", "choice": "a", "probabilities": dist, "confidence": 0.9}
        }
    }
    monkeypatch.setattr(
        client._opener, "open", lambda *a, **k: io.BytesIO(json.dumps(payload).encode())
    )
    with pytest.raises(ValueError):
        client.decide("input", [Choice("q", ("a", "b"), "Which?")])


def test_explicit_instructions_required():
    with pytest.raises(ValueError, match="instructions"):
        HttpJevClient("https://api.typesafe.ai", "test-key").decide("text", [Noul("q")])
