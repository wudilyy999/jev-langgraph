"""Run: python examples/review_resume.py. Offline review and override."""

from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END
from langgraph.types import Command

from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, add_receipts


class State(TypedDict):
    text: str
    result: str
    receipts: Annotated[list, add_receipts]


client = MockJevClient({"intent": ({"billing": 0.6, "tech": 0.4}, 0.6)})
g = JevGraph(State, client)
g.add_node("billing", lambda s: {"result": "billing handled"})
g.add_node("tech", lambda s: {"result": "tech handled"})
g.add_decision(
    "triage",
    DecisionNode(
        questions=[Choice("intent", ("billing", "tech"), "Which team should handle this ticket?")],
        routes={"billing": "billing", "tech": "tech"},
    ),
)
g.set_entry_point("triage")
g.add_edge("billing", END)
g.add_edge("tech", END)
app = g.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-001"}}
paused = app.invoke({"text": "Cannot access my invoice", "result": "", "receipts": []}, config)
print("Review payload:", paused["__interrupt__"][0].value)

# In an application, obtain these values from an authenticated human review endpoint.
out = app.invoke(
    Command(resume={"choices": {"intent": "tech"}, "reviewer": "demo-reviewer"}), config
)
print("Result:", out["result"])
print("Receipt:", out["receipts"][0])
print("JEV requests:", len(client.calls))
assert out["result"] == "tech handled"
assert len(client.calls) == 1
