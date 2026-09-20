import pytest
from langgraph.graph import END
from test_acceptance import State, initial

from jev_langgraph import CompositeScore, DecisionNode, JevGraph, MockJevClient, Noul, Score


@pytest.mark.parametrize(
    "probability,allowed", [(0.69, False), (0.7, True), (0.99, True), (0, False)]
)
def test_gate_uses_yes_probability_at_inclusive_threshold(probability, allowed):
    client = MockJevClient({"permitted": ({"yes": probability, "no": 1 - probability}, None)})
    g = JevGraph(State, client)
    g.add_gate(
        "guard",
        Noul("permitted", "May this tool call proceed?"),
        0.7,
        allow="tool",
        block="blocked",
    )
    g.add_node("tool", lambda s: {"visited": ["tool"]})
    g.add_node("blocked", lambda s: {"visited": ["blocked"]})
    g.add_edge("tool", END)
    g.add_edge("blocked", END)
    g.set_entry_point("guard")
    out = g.compile().invoke(initial())
    assert out["visited"] == (["tool"] if allowed else ["blocked"])
    r = out["receipts"][0]
    assert r["gate_probability"] == probability
    assert r["gate_allowed"] == allowed
    assert len(client.calls) == 1


def test_composite_normalizes_different_rubrics_and_routes():
    client = MockJevClient(
        {
            "quality": ({"low": 0.2, "high": 0.8}, 0.9),
            "risk": ({"low": 0.0, "medium": 1.0, "high": 0.0}, 1.0),
        }
    )
    g = JevGraph(State, client)
    g.add_decision(
        "judge",
        DecisionNode(
            [Score("quality", ("low", "high")), Score("risk", ("low", "medium", "high"))],
            edges={"quality": None, "risk": None},
            composite=CompositeScore({"quality": 3, "risk": 1}),
            composite_routes={0.0: "reject", 0.7: "accept"},
        ),
    )
    for name in ("reject", "accept"):
        g.add_node(name, lambda s, name=name: {"visited": [name]})
        g.add_edge(name, END)
    g.set_entry_point("judge")
    out = g.compile().invoke(initial())
    assert out["visited"] == ["accept"]
    assert out["receipts"][0]["composite_value"] == pytest.approx(0.725)
    assert all(r["reason"] == "composite_score" for r in out["receipts"])
    assert len(client.calls) == 1


@pytest.mark.parametrize("weights", [{}, {"q": 0}, {"q": -1}, {"q": float("nan")}])
def test_invalid_composite_weights(weights):
    with pytest.raises(ValueError):
        CompositeScore(weights)
