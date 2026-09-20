"""Acceptance contracts for the proposed JEV-native LangGraph behavior."""

import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command, interrupt

from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, Noul, add_receipts
from jev_langgraph.harness import build_harness


class State(TypedDict):
    text: str
    visited: Annotated[list, operator.add]
    receipts: Annotated[list, add_receipts]


def initial():
    return {"text": "task", "visited": [], "receipts": []}


def graph(client, questions=None, routes=None, **kwargs):
    g = JevGraph(State, client)
    for name in ("a", "b", "human"):
        g.add_node(name, lambda state, name=name: {"visited": [name]})
        g.add_edge(name, END)
    g.add_decision(
        "decision",
        DecisionNode(
            questions=questions or [Choice("route", options=("a", "b"))],
            routes=routes or {"a": "a", "b": "b"},
            **kwargs,
        ),
    )
    g.set_entry_point("decision")
    return g


def test_one_request_per_decision_domain():
    client = MockJevClient({"route": ({"a": 0.99, "b": 0.01}, 0.99)})
    graph(client).compile().invoke(initial())
    assert len(client.calls) == 1


def test_receipt_matches_executed_route_when_provider_changes_answer():
    class ChangingClient(MockJevClient):
        def decide(self, text, questions):
            dist = {"a": 0.99, "b": 0.01} if not self.calls else {"a": 0.01, "b": 0.99}
            self.rules["route"] = (dist, 0.99)
            return super().decide(text, questions)

    out = graph(ChangingClient()).compile().invoke(initial())
    assert out["visited"] == [out["receipts"][0]["chosen"]]


def test_independent_questions_execute_both_route_groups():
    client = MockJevClient(
        {
            "first": ({"a": 0.99, "skip_a": 0.01}, 0.99),
            "second": ({"b": 0.99, "skip_b": 0.01}, 0.99),
        }
    )
    questions = [
        Choice("first", options=("a", "skip_a")),
        Choice("second", options=("b", "skip_b")),
    ]
    out = graph(client, questions=questions).compile().invoke(initial())
    assert sorted(out["visited"]) == ["a", "b"]


def test_review_band_provides_interrupt_without_business_execution():
    client = MockJevClient({"route": ({"a": 0.6, "b": 0.4}, 0.6)})
    app = graph(client, review="human").compile(checkpointer=InMemorySaver())
    out = app.invoke(initial(), {"configurable": {"thread_id": "review"}})
    assert out.get("__interrupt__"), "Review policy must pause for a human decision"
    assert out["visited"] == []


def test_unconfigured_fallback_does_not_execute_low_confidence_winner():
    client = MockJevClient({"route": ({"a": 0.34, "b": 0.33, "c": 0.33}, 0.34)})
    with pytest.raises(ValueError):
        graph(client, questions=[Choice("route", options=("a", "b", "c"))]).compile().invoke(
            initial()
        )


def test_unmapped_answer_does_not_silently_select_first_route():
    client = MockJevClient({"route": ({"unmapped": 0.99, "a": 0.01}, 0.99)})
    with pytest.raises((ValueError, KeyError)):
        graph(client, questions=[Choice("route", options=("unmapped", "a"))]).compile().invoke(
            initial()
        )


def test_harness_respects_max_fix_rounds():
    client = MockJevClient(
        {
            "model": ({"fast": 1.0}, 1.0),
            "ok": ({"yes": 0.01, "no": 0.99}, 0.99),
            "why": ({"wrong": 1.0}, 1.0),
        }
    )
    app = build_harness(
        State,
        client,
        models={"fast": lambda state: {"visited": ["generate"]}},
        verify=Noul("ok"),
        diagnose=Choice("why", options=("wrong",)),
        fixer=lambda state: {"visited": ["fix"]},
        max_fix_rounds=1,
    )
    out = app.invoke(initial(), {"recursion_limit": 20})
    assert out["visited"].count("fix") <= 1


def test_native_interrupt_resume_and_checkpoint_after_jev_decision():
    client = MockJevClient({"route": ({"a": 0.99, "b": 0.01}, 0.99)})
    g = JevGraph(State, client)
    g.add_decision(
        "decision",
        DecisionNode(
            questions=[Choice("route", options=("a", "b"))],
            routes={"a": "human", "b": END},
        ),
    )

    def human(state):
        choice = interrupt({"distribution": state["receipts"][-1]["distribution"]})
        return {"text": choice}

    g.add_node("human", human)
    g.add_conditional_edges("human", lambda state: state["text"], {"approve": "done"})
    g.add_node("done", lambda state: {"visited": ["done"]})
    g.add_edge("done", END)
    g.set_entry_point("decision")
    app = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "native"}}
    paused = app.invoke(initial(), config)
    assert paused["__interrupt__"][0].value["distribution"]["a"] == 0.99
    assert len(app.get_state(config).values["receipts"]) == 1
    out = app.invoke(Command(resume="approve"), config)
    assert out["visited"] == ["done"]
    assert len(out["receipts"]) == 1


def test_native_async_and_stream_execution():
    client = MockJevClient({"route": ({"a": 0.99, "b": 0.01}, 0.99)})
    app = graph(client).compile()
    assert asyncio.run(app.ainvoke(initial()))["visited"] == ["a"]
    updates = list(app.stream(initial(), stream_mode="updates"))
    assert any("decision" in update for update in updates)
    assert any("a" in update for update in updates)
