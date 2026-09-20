import io
import json
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from jev_langgraph import (
    Choice,
    DecideResponse,
    DecisionNode,
    JevGraph,
    MockJevClient,
    Noul,
    Receipt,
    add_receipts,
)
from jev_langgraph.analysis import calibration_by_model, cost_report
from jev_langgraph.client import structured_state
from jev_langgraph.http_client import HttpJevClient


class State(TypedDict):
    payload: dict
    receipts: Annotated[list, add_receipts]


def test_default_selector_preserves_nested_types_and_roles():
    state = {
        "payload": {
            "amount": 5,
            "enabled": True,
            "missing": None,
            "messages": [SystemMessage("Rules"), HumanMessage("Help")],
        },
        "receipts": [],
        "__jev_internal": "private",
    }
    result = structured_state(state)
    assert result == {
        "payload": {
            "amount": 5,
            "enabled": True,
            "missing": None,
            "messages": [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Help"},
            ],
        }
    }
    client = MockJevClient({"q": ({"yes": 1.0, "no": 0.0}, None)})
    g = JevGraph(State, client)
    g.add_decision("d", DecisionNode([Noul("q")], {"yes": END, "no": END}))
    g.set_entry_point("d")
    g.compile().invoke(state)
    assert client.calls[0][0] == result


@pytest.mark.parametrize("state", [float("nan"), {"x": object()}, {1: "bad"}])
def test_non_json_state_fails_explicitly(state):
    with pytest.raises(ValueError):
        structured_state(state)


def test_structured_http_request_and_first_class_response(monkeypatch):
    client = HttpJevClient("https://api.typesafe.ai", "test-key")
    prompt = {"question": "Check `account.active`", "context_field": "account"}
    criteria = {"a": {"meaning": "active"}, "b": None}

    class Response(io.BytesIO):
        headers = {"x-request-id": "req-123"}

    def respond(request, timeout):
        body = json.loads(request.data)
        assert body["state"] == {"account": {"active": True}, "items": [1, None]}
        assert body["questions"]["q"]["instructions"] == prompt
        assert body["questions"]["q"]["criteria"] == criteria
        return Response(
            json.dumps(
                {
                    "model": "jev-1.13.0",
                    "usage": {"input_tokens": 23, "output_tokens": 4},
                    "answers": {
                        "q": {
                            "type": "choice",
                            "choice": "a",
                            "probabilities": {"a": 1, "b": 0},
                            "confidence": 1,
                        }
                    },
                }
            ).encode()
        )

    monkeypatch.setattr(client._opener, "open", respond)
    result = client.decide(
        {"account": {"active": True}, "items": [1, None]},
        [Choice("q", ("a", "b"), prompt, criteria=criteria)],
    )
    assert (result.model, result.request_id, result.input_tokens, result.output_tokens) == (
        "jev-1.13.0",
        "req-123",
        23,
        4,
    )


def test_multiquestion_usage_is_counted_once_and_run_ids_are_recorded():
    class Metered(MockJevClient):
        def decide(self, state, questions):
            response = super().decide(state, questions)
            response.model = "jev-v1"
            response.request_id = "provider-id"
            response.input_tokens, response.output_tokens = 100, 20
            return response

    client = Metered({"q": ({"a": 1.0}, 1.0), "r": ({"a": 1.0}, 1.0)})
    g = JevGraph(State, client)
    g.add_decision("d", DecisionNode([Choice("q", ("a",)), Choice("r", ("a",))], {"a": END}))
    g.set_entry_point("d")
    out = g.compile().invoke(
        {"payload": {}, "receipts": []}, {"configurable": {"thread_id": "t1", "jev_run_id": "run1"}}
    )
    records = [Receipt(**r) for r in out["receipts"]]
    assert all(r.model == "jev-v1" and r.request_id == "provider-id" for r in records)
    assert all(r.thread_id == "t1" and r.run_id == "run1" for r in records)
    report = cost_report(records, run_id="run1", input_per_million=1, output_per_million=2)[0]
    assert report == {
        "node": "d",
        "requests": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "unmetered_requests": 0,
        "cost": 0.00014,
    }
    assert cost_report(records, group_by="thread_id")[0]["thread_id"] == "t1"
    assert cost_report(records, group_by="run_id")[0]["run_id"] == "run1"
    assert cost_report(records, run_id="missing") == []
    model_report = calibration_by_model(records, {r.id: True for r in records})
    assert model_report[0]["model"] == "jev-v1"
    assert model_report[0]["ece"] == 0


def test_unknown_usage_is_not_free_usage():
    r = Receipt(
        node="d",
        question_id="q",
        kind="noul",
        distribution={"yes": 1},
        chosen="yes",
        confidence=None,
        policy="auto",
        latency_ms=0,
    )
    row = cost_report([r])[0]
    assert row["cost"] is None and row["input_tokens"] is None
    assert row["unmetered_requests"] == 1


@pytest.mark.parametrize(
    "counts", [{"input_tokens": -1}, {"output_tokens": True}, {"input_tokens": 2.5}]
)
def test_invalid_provider_usage_rejected(counts):
    with pytest.raises(ValueError):
        DecideResponse({}, **counts)
