from typing import Annotated, TypedDict

from jev_langgraph import (
    Choice,
    DecisionNode,
    JevGraph,
    MockJevClient,
    Policy,
    add_receipts,
)


class S(TypedDict):
    text: str
    receipts: Annotated[list, add_receipts]


def build(rules):
    client = MockJevClient(rules)
    g = JevGraph(S, client)
    g.add_node("billing", lambda s: {"text": s["text"] + " -> billing"})
    g.add_node("tech", lambda s: {"text": s["text"] + " -> tech"})
    g.add_node(
        "human",
        lambda s: {"text": s["text"] + " -> human"},
    )
    g.add_decision(
        "triage",
        DecisionNode(
            questions=[Choice("intent", options=("billing", "tech"))],
            routes={"billing": "billing", "tech": "tech"},
            policy=Policy(auto=0.95, provisional=0.7, review=0.4),
            fallback="human",
        ),
    )
    g.set_entry_point("triage")
    return g.compile(), client


def test_high_confidence_routes_and_receipts():
    app, client = build({"intent": ({"billing": 0.99, "tech": 0.01}, 0.99)})
    out = app.invoke({"text": "refund please", "receipts": []})
    assert out["text"].endswith("billing")
    # one JEV request covering both node eval + path eval
    assert len(client.calls) == 1
    r = out["receipts"][0]
    assert r["chosen"] == "billing"
    assert r["policy"] == "auto"
    assert r["distribution"]["billing"] == 0.99


def test_low_confidence_falls_back():
    app, _ = build({"intent": ({"billing": 0.51, "tech": 0.49}, 0.2)})
    out = app.invoke({"text": "??", "receipts": []})
    assert out["text"].endswith("human")
    assert out["receipts"][0]["policy"] == "fallback"


def test_provisional_band_tags_receipt():
    app, _ = build({"intent": ({"billing": 0.8, "tech": 0.2}, 0.8)})
    out = app.invoke({"text": "x", "receipts": []})
    assert out["text"].endswith("billing")
    assert out["receipts"][0]["policy"] == "provisional"


def test_native_nodes_alongside():
    # native conditional edge still works in the same graph
    client = MockJevClient({"intent": ({"billing": 0.99, "tech": 0.01}, 0.99)})
    g = JevGraph(S, client)
    g.add_node("start", lambda s: {"text": s["text"]})
    g.add_node("billing", lambda s: {"text": s["text"] + " -> billing"})
    g.add_node("tech", lambda s: {"text": s["text"] + " -> tech"})
    g.add_conditional_edges("start", lambda s: "billing")
    g.set_entry_point("start")
    app = g.compile()
    out = app.invoke({"text": "hi", "receipts": []})
    assert out["text"].endswith("billing")
