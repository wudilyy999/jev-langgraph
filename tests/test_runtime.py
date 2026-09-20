import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command
from test_acceptance import State, initial

from jev_langgraph import (
    Choice,
    DecisionNode,
    JevGraph,
    MockJevClient,
    Noul,
    Policy,
    ProbEdge,
    Receipt,
)
from jev_langgraph.analysis import expected_calibration_error
from jev_langgraph.harness import build_harness


def build(client, decision, checkpointer=None):
    g = JevGraph(State, client)
    for name in ("a", "b", "fallback"):
        g.add_node(name, lambda s, name=name: {"visited": [name]})
        g.add_edge(name, END)
    g.add_decision("choose", decision)
    g.set_entry_point("choose")
    return g.compile(checkpointer=checkpointer)


def decision(**kwargs):
    return DecisionNode([Choice("q", ("a", "b"))], {"a": "a", "b": "b"}, **kwargs)


def test_review_override_is_persisted_without_second_call():
    client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6)})
    app = build(client, decision(), InMemorySaver())
    config = {"configurable": {"thread_id": "override"}}
    out = app.invoke(initial(), config)
    assert out["visited"] == []
    assert out["receipts"][0]["status"] == "pending"
    out = app.invoke(Command(resume={"choices": {"q": "b"}, "reviewer": "alice"}), config)
    assert out["visited"] == ["b"]
    assert len(client.calls) == len(out["receipts"]) == 1
    r = out["receipts"][0]
    assert (r["chosen"], r["human_choice"], r["destinations"]) == ("a", "b", ["b"])
    assert r["reviewer"] == "alice"


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"choices": []},
        {"choices": {"q": "a"}},
        {"choices": {"q": "unknown"}, "reviewer": "alice"},
        {"choices": {"q": "a", "extra": "a"}, "reviewer": "alice"},
    ],
)
def test_invalid_review_never_executes_business(response):
    client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6)})
    app = build(client, decision(), InMemorySaver())
    config = {"configurable": {"thread_id": "invalid"}}
    app.invoke(initial(), config)
    # None is LangGraph's continue command; use a non-dict scalar for that case.
    with pytest.raises(ValueError):
        app.invoke(Command(resume=response if response is not None else "invalid"), config)
    assert app.get_state(config).values["visited"] == []
    assert len(client.calls) == 1


def test_empty_resume_keeps_review_pending():
    client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6)})
    app = build(client, decision(), InMemorySaver())
    config = {"configurable": {"thread_id": "empty"}}
    app.invoke(initial(), config)
    out = app.invoke(Command(resume={}), config)
    assert out["__interrupt__"]
    assert out["visited"] == []
    assert len(client.calls) == 1


def test_top_k_and_shared_targets_are_deduplicated():
    client = MockJevClient({"q": ({"a": 0.9, "b": 0.1}, 0.9), "r": ({"a": 1.0}, 1.0)})
    d = DecisionNode(
        [Choice("q", ("a", "b")), Choice("r", ("a",))],
        edges={
            "q": ProbEdge({"a": "a", "b": "b"}, top_k=2),
            "r": ProbEdge({"a": "a"}),
        },
    )
    out = build(client, d).invoke(initial())
    assert sorted(out["visited"]) == ["a", "b"]
    assert len(client.calls) == 1
    assert len(out["receipts"]) == 2


def test_fallback_suppresses_high_confidence_branch_and_review():
    client = MockJevClient({"q": ({"a": 0.9, "b": 0.1}, 0.9), "r": ({"a": 0.5, "b": 0.5}, 0.1)})
    d = DecisionNode(
        [Choice("q", ("a", "b")), Choice("r", ("a", "b"))],
        {"a": "a", "b": "b"},
        fallback="fallback",
    )
    out = build(client, d).invoke(initial())
    assert out["visited"] == ["fallback"]
    assert all(r["destinations"] == ["fallback"] for r in out["receipts"])


def test_observation_does_not_trigger_fallback():
    client = MockJevClient({"q": ({"a": 0.5, "b": 0.5}, 0.0)})
    out = build(client, DecisionNode([Choice("q", ("a", "b"))], edges={"q": None})).invoke(
        initial()
    )
    assert out["visited"] == []
    assert out["receipts"][0]["policy"] == "observe"


def test_concurrent_threads_do_not_share_decisions():
    class ByInput(MockJevClient):
        def decide(self, text, questions):
            label = "a" if text["text"] == "pick_a" else "b"
            return MockJevClient(
                {"q": ({"a": float(label == "a"), "b": float(label == "b")}, 1.0)}
            ).decide(text, questions)

    app = build(ByInput(), decision(), InMemorySaver())

    def run(i):
        label = "a" if i % 2 else "b"
        out = app.invoke(
            {**initial(), "text": f"pick_{label}"}, {"configurable": {"thread_id": str(i)}}
        )
        assert out["visited"] == [label]
        assert len(out["receipts"]) == 1

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))


def test_async_review_resume():
    async def run():
        client = MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6)})
        app = build(client, decision(), InMemorySaver())
        config = {"configurable": {"thread_id": "async"}}
        assert (await app.ainvoke(initial(), config))["__interrupt__"]
        out = await app.ainvoke(
            Command(resume={"choices": {"q": "b"}, "reviewer": "alice"}), config
        )
        assert out["visited"] == ["b"]
        assert len(client.calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_repair_limit_and_failure_status(limit):
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
        {"fast": lambda s: {}},
        Noul("ok"),
        Choice("why", ("wrong",)),
        lambda s: {"visited": ["fix"]},
        max_fix_rounds=limit,
    )
    out = app.invoke(initial(), {"recursion_limit": 100})
    assert out["visited"].count("fix") == out["fix_rounds"] == limit
    assert out["harness_status"] == "failed"


def test_calibration_uses_probability_not_provider_confidence():
    r = Receipt(
        node="n",
        question_id="q",
        kind="choice",
        distribution={"a": 0.9, "b": 0.1},
        chosen="a",
        confidence=0.2,
        policy="auto",
        latency_ms=0,
    )
    assert expected_calibration_error([r], {r.id: True}) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "kwargs", [{"auto": 0.2}, {"review": -0.1}, {"metric": "bad"}, {"auto": float("nan")}]
)
def test_invalid_policy_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        Policy(**kwargs)


def test_sqlite_reopen_resumes_without_model_call(tmp_path):
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite")
    db = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "disk"}}
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        app = build(MockJevClient({"q": ({"a": 0.6, "b": 0.4}, 0.6)}), decision(), saver)
        assert app.invoke(initial(), config)["__interrupt__"]
    with sqlite.SqliteSaver.from_conn_string(db) as saver:
        client = MockJevClient({})
        app = build(client, decision(), saver)
        out = app.invoke(Command(resume={"choices": {"q": "b"}, "reviewer": "alice"}), config)
        assert out["visited"] == ["b"]
        assert client.calls == []


def test_loop_selects_current_invocation_and_preserves_history():
    class SequenceClient(MockJevClient):
        def decide(self, text, questions):
            self.rules["q"] = (
                ({"a": 1.0, "b": 0.0}, 1.0) if not self.calls else ({"a": 0.0, "b": 1.0}, 1.0)
            )
            return super().decide(text, questions)

    client = SequenceClient()
    g = JevGraph(State, client)
    g.add_decision("choose", decision())
    g.add_node("a", lambda s: {"visited": ["a"]})
    g.add_node("b", lambda s: {"visited": ["b"]})
    g.add_edge("a", "choose")
    g.add_edge("b", END)
    g.set_entry_point("choose")
    out = g.compile().invoke(initial())
    assert out["visited"] == ["a", "b"]
    assert [r["chosen"] for r in out["receipts"]] == ["a", "b"]
    assert len(client.calls) == 2


def test_parallel_decision_domains_keep_both_receipt_updates():
    from langgraph.graph import START

    client = MockJevClient({"q": ({"a": 1.0, "b": 0.0}, 1.0)})
    g = JevGraph(State, client)
    for name in ("left", "right"):
        g.add_decision(
            name, DecisionNode([Choice("q", ("a", "b"))], {"a": name + "_done", "b": END})
        )
        g.add_node(name + "_done", lambda s, name=name: {"visited": [name]})
        g.add_edge(START, name)
        g.add_edge(name + "_done", END)
    out = g.compile().invoke(initial())
    assert sorted(out["visited"]) == ["left", "right"]
    assert len(out["receipts"]) == 2
    assert all(r["status"] == "dispatched" for r in out["receipts"])


def test_noul_uses_selected_probability_and_preserves_missing_confidence():
    client = MockJevClient({"q": ({"yes": 0.01, "no": 0.99}, None)})
    d = DecisionNode([Noul("q")], {"yes": "a", "no": "b"})
    out = build(client, d).invoke(initial())
    assert out["visited"] == ["b"]
    assert out["receipts"][0]["confidence"] is None
    assert out["receipts"][0]["policy_metric"] == "probability"


def test_native_fanout_join_after_decision():
    client = MockJevClient({"q": ({"a": 0.9, "b": 0.1}, 0.9)})
    g = JevGraph(State, client)
    g.add_decision(
        "choose",
        DecisionNode(
            [Choice("q", ("a", "b"))], edges={"q": ProbEdge({"a": "a", "b": "b"}, top_k=2)}
        ),
    )
    g.add_node("a", lambda s: {"visited": ["a"]})
    g.add_node("b", lambda s: {"visited": ["b"]})
    g.add_node("join", lambda s: {"visited": ["join"]})
    g.add_edge(["a", "b"], "join")
    g.add_edge("join", END)
    g.set_entry_point("choose")
    out = g.compile().invoke(initial())
    assert sorted(out["visited"][:-1]) == ["a", "b"]
    assert out["visited"][-1] == "join"
