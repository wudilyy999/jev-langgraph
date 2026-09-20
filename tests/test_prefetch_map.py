import asyncio
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from test_acceptance import State, initial

from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, Receipt, add_receipts
from jev_langgraph.analysis import cost_report


def prefetch_graph(client, saver=None, change_input=False):
    def selector(s):
        return {"text": s["text"]}

    d1 = DecisionNode(
        [Choice("q", ("a", "b"))], {"a": "after", "b": "after"}, input_selector=selector
    )
    d2 = DecisionNode(
        [Choice("r", ("a", "b"))], {"a": "done", "b": "done"}, input_selector=selector
    )
    g = JevGraph(State, client)
    g.add_prefetch("hub", {"first": d1, "second": d2}, input_selector=selector)
    g.add_decision("first", d1, prefetch="hub")
    g.add_decision("second", d2, prefetch="hub")
    g.add_node("after", lambda s: {"text": "changed"} if change_input else {"visited": ["after"]})
    g.add_node("done", lambda s: {"visited": ["done"]})
    g.add_edge("hub", "first")
    g.add_edge("after", "second")
    g.add_edge("done", END)
    g.set_entry_point("hub")
    return g.compile(checkpointer=saver)


def test_prefetch_two_domains_one_call_and_no_double_metering():
    class Metered(MockJevClient):
        def decide(self, state, questions):
            r = super().decide(state, questions)
            r.input_tokens, r.output_tokens = 100, 20
            return r

    client = Metered({"q": ({"a": 1.0, "b": 0}, 1), "r": ({"a": 1.0, "b": 0}, 1)})
    out = prefetch_graph(client).invoke(initial(), {"configurable": {"jev_run_id": "run"}})
    assert out["visited"] == ["after", "done"]
    assert len(client.calls) == 1
    assert client.calls[0][1] == ["q", "r"]
    records = [Receipt(**r) for r in out["receipts"]]
    assert len(records) == 4
    assert cost_report(records) == [
        {
            "node": "hub",
            "requests": 1,
            "input_tokens": 100,
            "output_tokens": 20,
            "unmetered_requests": 0,
            "cost": 0,
        }
    ]
    assert all(r.source_receipt_id for r in records if r.node != "hub")


def test_prefetch_detects_changed_snapshot():
    client = MockJevClient({"q": ({"a": 1.0, "b": 0}, 1), "r": ({"a": 1.0, "b": 0}, 1)})
    with pytest.raises(ValueError, match="stale"):
        prefetch_graph(client, change_input=True).invoke(
            initial(), {"configurable": {"jev_run_id": "run"}}
        )
    assert len(client.calls) == 1


def test_prefetch_review_resume_uses_saved_answers():
    client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6), "r": ({"a": 1.0, "b": 0}, 1)})
    app = prefetch_graph(client, InMemorySaver())
    config = {"configurable": {"thread_id": "prefetch", "jev_run_id": "run"}}
    assert app.invoke(initial(), config)["__interrupt__"]
    out = app.invoke(Command(resume={"choices": {"q": "b"}, "reviewer": "alice"}), config)
    assert out["visited"] == ["after", "done"]
    assert len(client.calls) == 1


class BatchState(TypedDict):
    items: list
    map_results: Annotated[list, operator.add]
    receipts: Annotated[list, add_receipts]


def map_graph(client, saver=None):
    g = JevGraph(BatchState, client)
    d = DecisionNode(
        [Choice("q", ("keep", "drop"))],
        {"keep": "keep", "drop": "drop"},
        input_selector=lambda s: s["item"],
        fallback="drop",
    )
    g.map_decision(
        "classify",
        d,
        items=lambda s: s["items"],
        handlers={
            "keep": lambda s: {"result": {"value": s["item"], "kept": True}},
            "drop": lambda s: {"result": {"value": s["item"], "kept": False}},
        },
    )
    g.set_entry_point("classify")
    return g.compile(checkpointer=saver)


def test_map_send_fanout_independent_receipts_and_results():
    class ByItem(MockJevClient):
        def decide(self, state, questions):
            self.calls.append(state)
            return MockJevClient(
                {"q": ({"keep": float(state["keep"]), "drop": float(not state["keep"])}, 1)}
            ).decide(state, questions)

    client = ByItem()
    out = map_graph(client).invoke(
        {"items": [{"keep": True}, {"keep": False}], "receipts": [], "map_results": []}
    )
    assert len(out["receipts"]) == len(client.calls) == len(out["map_results"]) == 2
    assert {r["item_id"] for r in out["receipts"]} == {"0", "1"}
    assert {r["result"]["kept"] for r in out["map_results"]} == {True, False}


def test_map_empty_input_calls_nothing():
    client = MockJevClient()
    out = map_graph(client).invoke({"items": [], "receipts": [], "map_results": []})
    assert out["map_results"] == [] and client.calls == []


def test_map_review_resume_does_not_repeat_model_call():
    client = MockJevClient({"q": ({"keep": 0.6, "drop": 0.4}, 0.6)})
    app = map_graph(client, InMemorySaver())
    config = {"configurable": {"thread_id": "batch"}}
    paused = app.invoke({"items": [{"x": 1}], "receipts": [], "map_results": []}, config)
    assert paused["__interrupt__"]
    out = app.invoke(Command(resume={"choices": {"q": "drop"}, "reviewer": "alice"}), config)
    assert len(client.calls) == 1
    assert out["map_results"][0]["result"]["kept"] is False
    assert out["receipts"][0]["human_choice"] == "drop"


def test_multiple_map_interrupts_resume_independently():
    client = MockJevClient({"q": ({"keep": 0.6, "drop": 0.4}, 0.6)})
    app = map_graph(client, InMemorySaver())
    config = {"configurable": {"thread_id": "multi-review"}}
    paused = app.invoke({"items": [{"x": 1}, {"x": 2}], "receipts": [], "map_results": []}, config)
    interrupts = paused["__interrupt__"]
    assert len(interrupts) == 2
    responses = {
        event.id: {"choices": {"q": "keep" if i == 0 else "drop"}, "reviewer": "alice"}
        for i, event in enumerate(interrupts)
    }
    out = app.invoke(Command(resume=responses), config)
    assert len(client.calls) == 2
    assert len(out["map_results"]) == len(out["receipts"]) == 2
    assert {r["human_choice"] for r in out["receipts"]} == {"keep", "drop"}


def test_async_map_review_and_async_handler():
    async def run():
        client = MockJevClient({"q": ({"keep": 0.6, "drop": 0.4}, 0.6)})
        g = JevGraph(BatchState, client)

        async def handler(state):
            return {"result": state["item"]["x"] * 2}

        g.map_decision(
            "batch",
            DecisionNode(
                [Choice("q", ("keep", "drop"))],
                {"keep": "handle", "drop": "handle"},
                input_selector=lambda s: s["item"],
            ),
            items=lambda s: s["items"],
            handlers={"handle": handler},
        )
        g.set_entry_point("batch")
        app = g.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "async-map"}}
        paused = await app.ainvoke({"items": [{"x": 3}], "receipts": [], "map_results": []}, config)
        assert paused["__interrupt__"]
        out = await app.ainvoke(
            Command(resume={"choices": {"q": "keep"}, "reviewer": "alice"}), config
        )
        assert out["map_results"][0]["result"] == 6
        assert len(client.calls) == 1

    asyncio.run(run())


def test_prefetch_requires_explicit_run_identity():
    client = MockJevClient()
    with pytest.raises(ValueError, match="jev_run_id"):
        prefetch_graph(client).invoke(initial())
    assert client.calls == []


def test_prefetch_sqlite_reopen_zero_calls(tmp_path):
    from langgraph.checkpoint.sqlite import SqliteSaver

    config = {"configurable": {"thread_id": "persisted", "jev_run_id": "one-run"}}
    db = str(tmp_path / "prefetch.sqlite")
    with SqliteSaver.from_conn_string(db) as saver:
        client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6), "r": ({"a": 1.0, "b": 0}, 1)})
        assert prefetch_graph(client, saver).invoke(initial(), config)["__interrupt__"]
    with SqliteSaver.from_conn_string(db) as saver:
        client = MockJevClient()
        out = prefetch_graph(client, saver).invoke(
            Command(resume={"choices": {"q": "a"}, "reviewer": "alice"}), config
        )
        assert out["visited"] == ["after", "done"]
        assert client.calls == []


def test_changed_question_definition_rejected():
    from dataclasses import asdict

    from jev_langgraph.client import json_value
    from jev_langgraph.graph import _CachedClient

    q = Choice("q", ("a", "b"), "original")
    r = Receipt(
        node="hub",
        question_id="q",
        kind="choice",
        distribution={"a": 1, "b": 0},
        chosen="a",
        confidence=1,
        policy="observe",
        latency_ms=0,
        snapshot={"x": True},
        question_spec=json_value(asdict(q)),
    )
    cache = _CachedClient([r])
    with pytest.raises(ValueError, match="definition changed"):
        cache.decide({"x": True}, [Choice("q", ("a", "b"), "changed")])
    with pytest.raises(ValueError, match="stale"):
        cache.decide({"x": 1}, [q])


def test_map_join_runs_once_after_all_items():
    class JoinedState(BatchState):
        joined: int

    g = JevGraph(JoinedState, MockJevClient({"q": ({"a": 1}, 1)}))
    g.map_decision(
        "batch",
        DecisionNode([Choice("q", ("a",))], {"a": "handle"}),
        items=lambda s: s["items"],
        handlers={"handle": lambda s: {"result": s["item"]}},
        then="join",
    )
    g.add_node("join", lambda s: {"joined": len(s["map_results"])})
    g.add_edge("join", END)
    g.set_entry_point("batch")
    out = g.compile().invoke(
        {"items": [{"v": 1}, {"v": 2}, {"v": 3}], "receipts": [], "map_results": [], "joined": 0}
    )
    assert out["joined"] == 3


def test_map_completed_item_is_not_reexecuted_when_other_item_resumes():
    class Mixed(MockJevClient):
        def decide(self, state, questions):
            self.calls.append(state)
            p = state["p"]
            return MockJevClient({"q": ({"keep": p, "drop": 1 - p}, p)}).decide(state, questions)

    client = Mixed()
    app = map_graph(client, InMemorySaver())
    config = {"configurable": {"thread_id": "mixed"}}
    paused = app.invoke(
        {"items": [{"p": 1}, {"p": 0.6}], "receipts": [], "map_results": []}, config
    )
    assert len(paused["__interrupt__"]) == 1
    assert paused["__interrupt__"][0].value["questions"][0]["item_id"] == "1"
    out = app.invoke(Command(resume={"choices": {"q": "drop"}, "reviewer": "alice"}), config)
    assert len(client.calls) == len(out["map_results"]) == 2
