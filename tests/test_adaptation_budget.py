import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from test_acceptance import State, initial

from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, Policy, Receipt


def records():
    rows = []
    outcomes = {}
    for i in range(100):
        score = 0.8 if i < 50 else 0.95
        r = Receipt(
            node="d",
            question_id="q",
            kind="choice",
            chosen="a",
            distribution={"a": score, "b": 1 - score},
            confidence=score,
            policy="auto",
            model="v1",
            latency_ms=0,
        )
        rows.append(r)
        outcomes[r.id] = i >= 50 or i < 30
    return rows, outcomes


def test_adaptation_raises_threshold_and_audits_evidence():
    rows, outcomes = records()
    policy = Policy(auto=0.8)
    adapted = policy.adapt(
        rows, outcomes, node="d", question_id="q", model="v1", version="v2", target_accuracy=0.95
    )
    assert adapted.auto == 0.95
    assert policy.auto == 0.8
    assert adapted.action_for(0.8) == "provisional"
    assert adapted.adaptation["support"] == 50
    assert adapted.adaptation["observed_accuracy"] == 1


def test_adaptation_insufficient_or_wrong_model_disables_auto():
    rows, outcomes = records()
    adapted = Policy().adapt(
        rows, outcomes, node="d", question_id="q", model="unseen", version="v2"
    )
    assert not adapted.auto_enabled
    assert adapted.action_for(1) == "review"


def test_budget_changes_runtime_action_before_business_execution():
    class Metered(MockJevClient):
        def decide(self, state, questions):
            response = super().decide(state, questions)
            response.input_tokens, response.output_tokens = 80, 30
            return response

    client = Metered({"q": ({"a": 0.99, "b": 0.01}, 0.99)})
    g = JevGraph(State, client)
    g.add_decision(
        "d",
        DecisionNode(
            [Choice("q", ("a", "b"))], {"a": "write", "b": END}, policy=Policy(token_budget=100)
        ),
    )
    g.add_node("write", lambda s: {"visited": ["write"]})
    g.add_edge("write", END)
    g.set_entry_point("d")
    out = g.compile(checkpointer=InMemorySaver()).invoke(
        initial(), {"configurable": {"thread_id": "budget", "jev_run_id": "r1"}}
    )
    assert out["visited"] == []
    assert out["__interrupt__"]
    assert out["receipts"][0]["budget_exceeded"]
    assert out["receipts"][0]["policy"] == "review"


def test_unmetered_budget_fails_explicitly():
    client = MockJevClient({"q": ({"a": 1.0}, 1.0)})
    d = DecisionNode([Choice("q", ("a",))], {"a": END}, policy=Policy(token_budget=1))
    with pytest.raises(ValueError, match="measured usage"):
        d.evaluate("d", initial(), client)


def test_budget_accumulates_once_per_call_and_isolates_runs():
    class Metered(MockJevClient):
        def decide(self, state, questions):
            r = super().decide(state, questions)
            r.input_tokens, r.output_tokens = 40, 20
            return r

    client = Metered({"q": ({"a": 0.99, "b": 0.01}, 0.99)})
    d = DecisionNode(
        [Choice("q", ("a", "b"))], {"a": END, "b": END}, policy=Policy(token_budget=100)
    )
    _, first = d.evaluate("d", initial(), client, run_id="r1")
    state = {**initial(), "receipts": [r.model_dump() for r in first]}
    _, second = d.evaluate("d", state, client, run_id="r1")
    assert second[0].policy == "review" and second[0].budget_exceeded
    _, independent = d.evaluate("d", state, client, run_id="r2")
    assert independent[0].policy == "auto"
    _, cached = d.evaluate("d", state, client, run_id="r1", call_id=first[0].call_id)
    assert cached[0].policy == "auto"


def test_adapted_policy_changes_graph_execution_and_receipt_version():
    rows, outcomes = records()
    adapted = Policy(auto=0.8).adapt(
        rows, outcomes, node="d", question_id="q", model="v1", version="calibrated-v2"
    )
    client = MockJevClient({"q": ({"a": 0.8, "b": 0.2}, 0.8)})
    d = DecisionNode([Choice("q", ("a", "b"))], {"a": END, "b": END}, policy=adapted)
    _, receipts = d.evaluate("d", initial(), client)
    assert receipts[0].policy == "provisional"
    assert receipts[0].policy_version == "calibrated-v2"
    assert receipts[0].metadata["adaptation"]["support"] == 50
