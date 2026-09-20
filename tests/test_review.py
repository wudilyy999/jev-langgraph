"""Human-in-the-loop: confidence in the review band routes to a review node."""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, Policy, add_receipts


class S(TypedDict):
    text: str
    receipts: Annotated[list, add_receipts]


def test_review_band_routes_to_human_node():
    client = MockJevClient({"intent": ({"billing": 0.6, "tech": 0.4}, 0.5)})
    g = JevGraph(S, client)
    g.add_node("billing", lambda s: {"text": s["text"] + " -> billing"})
    g.add_node("tech", lambda s: {"text": s["text"] + " -> tech"})
    g.add_node("human_review", lambda s: {"text": s["text"] + " -> human_review"})
    g.add_decision(
        "triage",
        DecisionNode(
            questions=[Choice("intent", options=("billing", "tech"))],
            routes={"billing": "billing", "tech": "tech"},
            policy=Policy(auto=0.95, provisional=0.7, review=0.4),
            review="human_review",
        ),
    )
    g.set_entry_point("triage")
    app = g.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "review"}}
    paused = app.invoke({"text": "unclear", "receipts": []}, config)
    assert paused["__interrupt__"]
    out = app.invoke(Command(resume={"choices": {"intent": "tech"}, "reviewer": "tester"}), config)
    assert len(client.calls) == 1
    assert out["receipts"][0]["human_choice"] == "tech"
    assert out["text"].endswith("human_review")
    assert out["receipts"][0]["policy"] == "review"
    # full distribution survives for the human to inspect
    assert out["receipts"][0]["distribution"] == {"billing": 0.6, "tech": 0.4}
