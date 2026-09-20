"""Support-ticket triage with JEV-native routing.

Run offline (mock client):  python3 examples/support_triage.py
Run against real JEV:       set TYPESAFE_API_KEY (optional JEV_BASE_URL)
"""

import os
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver

from jev_langgraph import (
    Choice,
    DecisionNode,
    JevGraph,
    MockJevClient,
    Noul,
    Policy,
    ProbEdge,
    Score,
    add_receipts,
)


class Ticket(TypedDict):
    text: str
    receipts: Annotated[list, add_receipts]


def billing(s):
    return {"text": s["text"] + "\n[billing handled]"}


def tech(s):
    return {"text": s["text"] + "\n[tech handled]"}


def human(s):
    return {"text": s["text"] + "\n[escalated to human]"}


def build(client, checkpointer=None):
    g = JevGraph(Ticket, client)
    g.add_node("billing", billing)
    g.add_node("tech", tech)
    g.add_node("human", human)
    g.add_decision(
        "triage",
        DecisionNode(
            questions=[
                Choice("intent", options=("billing", "tech"), prompt="What does this ticket need?"),
                Score("severity", levels=("low", "high"), prompt="How severe is it?"),
                Noul("angry", prompt="Is the customer angry?"),
            ],
            edges={
                "intent": ProbEdge({"billing": "billing", "tech": "tech"}),
                "severity": None,
                "angry": None,
            },
            policy=Policy(auto=0.95, provisional=0.7, review=0.4),
            fallback="human",
        ),
    )
    g.set_entry_point("triage")
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    if os.getenv("TYPESAFE_API_KEY"):
        from jev_langgraph.http_client import HttpJevClient

        client = HttpJevClient(
            os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"), os.environ["TYPESAFE_API_KEY"]
        )
    else:
        client = MockJevClient(
            {
                "intent": ({"billing": 0.97, "tech": 0.03}, 0.96),
                "severity": ({"low": 0.2, "high": 0.8}, 0.75),
                "angry": ({"yes": 0.12, "no": 0.88}, 0.9),
            }
        )
    app = build(client, InMemorySaver())
    config = {"configurable": {"thread_id": "ticket-demo"}}
    out = app.invoke({"text": "I was charged twice and want my money back", "receipts": []}, config)
    if out.get("__interrupt__"):
        print("Awaiting review:", out["__interrupt__"][0].value)
    print(out["text"])
    print("\nreceipts:")
    for r in out["receipts"]:
        print(
            f"  {r['question_id']:10s} -> {r['chosen']:8s} confidence={r['confidence']} policy={r['policy']}"
        )
