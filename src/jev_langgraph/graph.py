"""Compile decisions into checkpointed evaluation and dispatch steps."""

from __future__ import annotations

from typing import Any, get_type_hints

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .client import JevClient
from .decision import DecisionNode
from .receipt import Receipt, add_receipts


def _batch(state, name):
    records = [r for r in state["receipts"] if r["node"] == name]
    invocation = records[-1]["invocation_id"]
    return [Receipt(**r) for r in records if r["invocation_id"] == invocation]


class JevGraph:
    def __init__(self, state_schema, client: JevClient, **kwargs):
        self._client = client
        self._graph = StateGraph(state_schema, **kwargs)
        self._schema = state_schema

    def __getattr__(self, name: str) -> Any:
        # add_node, add_edge, add_conditional_edges, set_entry_point, etc.
        # all forward to the wrapped StateGraph.
        return getattr(self._graph, name)

    def compile(self, **kwargs):
        return self._graph.compile(**kwargs)

    def add_decision(self, name: str, decision: DecisionNode) -> None:
        """Review pauses after evaluation, so resume reuses the stored answer."""
        channel = get_type_hints(self._schema, include_extras=True).get("receipts")
        if add_receipts not in getattr(channel, "__metadata__", ()):
            raise ValueError("State requires receipts: Annotated[list, add_receipts]")
        client = self._client
        dispatch_name = f"{name}__dispatch"

        def _node(state):
            _, receipts = decision.evaluate(name, state, client)
            return {"receipts": [r.model_dump() for r in receipts]}

        def _dispatch(state):
            receipts = _batch(state, name)
            fallback = any(r.policy == "fallback" for r in receipts)
            reviews = [r for r in receipts if r.policy == "review"]
            # Gate the whole decision domain before any business branch is scheduled.
            if reviews and not fallback:
                response = interrupt(
                    {
                        "type": "jev_review",
                        "node": name,
                        "questions": [
                            {
                                "receipt_id": r.id,
                                "question_id": r.question_id,
                                "suggested": r.chosen,
                                "distribution": r.distribution,
                                "candidates": sorted(r.distribution.items(), key=lambda p: -p[1])[
                                    :3
                                ],
                                "allowed": list(decision.edge_for(r.question_id).routes),
                            }
                            for r in reviews
                        ],
                    }
                )
                if not isinstance(response, dict) or not isinstance(response.get("choices"), dict):
                    raise ValueError("Resume requires a choices mapping and reviewer")
                if set(response["choices"]) != {r.question_id for r in reviews}:
                    raise ValueError("Resume must answer exactly the reviewed questions")
                if (
                    not isinstance(response.get("reviewer"), str)
                    or not response["reviewer"].strip()
                ):
                    raise ValueError("Resume requires a reviewer identifier")
                for r in reviews:
                    choice = response["choices"][r.question_id]
                    if (
                        not isinstance(choice, str)
                        or choice not in r.distribution
                        or choice not in decision.edge_for(r.question_id).routes
                    ):
                        raise ValueError(f"Invalid human choice for {r.question_id}")
                    r.selected = [choice]
                    r.human_choice = choice
                    r.reviewer = response["reviewer"]
            for r in receipts:
                r.status = "dispatched"
                if fallback:
                    r.destinations = [decision.fallback]
                    r.selected = []
                    r.reason = "domain_fallback"
                elif reviews and decision.review:
                    r.destinations = [decision.review]
                    r.reason = "post_review_handler"
                elif r.policy != "observe":
                    edge = decision.edge_for(r.question_id)
                    r.destinations = list(dict.fromkeys(edge.routes[k] for k in r.selected))
            return {"receipts": [r.model_dump() for r in receipts]}

        def _path(state):
            return list(dict.fromkeys(t for r in _batch(state, name) for t in r.destinations)) or [
                END
            ]

        self._graph.add_node(name, _node)
        self._graph.add_node(dispatch_name, _dispatch)
        self._graph.add_edge(name, dispatch_name)
        self._graph.add_conditional_edges(
            dispatch_name, _path, list(dict.fromkeys(decision.targets() + [END]))
        )
