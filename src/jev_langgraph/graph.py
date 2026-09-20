"""Compile decisions into checkpointed evaluation and dispatch steps."""

from __future__ import annotations

import json
import operator
from dataclasses import asdict, replace
from typing import Annotated, Any, Callable, Mapping, TypedDict, get_type_hints

from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.types import Send, interrupt

from .client import Answer, DecideResponse, JevClient, Noul, json_value, structured_state
from .decision import DecisionNode
from .receipt import Receipt, add_receipts


def _batch(state, name):
    records = [r for r in state["receipts"] if r["node"] == name]
    invocation = records[-1]["invocation_id"]
    return [Receipt(**r) for r in records if r["invocation_id"] == invocation]


class _ItemState(TypedDict):
    item: Any
    item_id: str
    result: Any
    receipts: Annotated[list, add_receipts]


class _CachedClient:
    def __init__(self, receipts):
        self.receipts = receipts

    def decide(self, state, questions):
        indexed = {r.question_id: r for r in self.receipts}
        first = self.receipts[0]
        answers = {}
        for q in questions:
            if q.id not in indexed:
                raise ValueError(f"Question {q.id} was not prefetched")
            r = indexed[q.id]
            if r.question_spec != json_value(asdict(q)):
                raise ValueError(f"Prefetched question definition changed: {q.id}")
            if json.dumps(r.snapshot, sort_keys=True, allow_nan=False) != json.dumps(
                json_value(state), sort_keys=True, allow_nan=False
            ):
                raise ValueError(
                    "Prefetched input snapshot is stale; refresh the hub or select stable inputs"
                )
            answers[q.id] = Answer(q.id, r.kind, r.distribution, r.chosen, r.confidence, r.score)
        return DecideResponse(
            answers,
            model=first.model,
            request_id=first.request_id,
            input_tokens=first.input_tokens,
            output_tokens=first.output_tokens,
        )


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

    def add_gate(
        self,
        name: str,
        noul: Noul,
        threshold: float,
        *,
        allow: str,
        block: str = END,
        input_selector=structured_state,
    ) -> None:
        """Allow when P(yes) >= threshold; ask the Noul as a permission question."""
        self.add_decision(
            name,
            DecisionNode(
                [noul],
                {"yes": allow, "no": block},
                gate_threshold=threshold,
                input_selector=input_selector,
            ),
        )

    def add_prefetch(
        self, name: str, domains: Mapping[str, DecisionNode], *, input_selector=structured_state
    ) -> None:
        """Evaluate the union of future domains' questions against one explicit snapshot."""
        questions = {}
        for decision in domains.values():
            for q in decision.questions:
                if q.id in questions and questions[q.id] != q:
                    raise ValueError(f"Prefetch question ID collision: {q.id}")
                questions[q.id] = q
        if not questions:
            raise ValueError("Prefetch requires at least one domain")
        observation = DecisionNode(
            list(questions.values()),
            edges={q: None for q in questions},
            input_selector=input_selector,
        )
        self._require_receipts()

        def prefetch(state, config: RunnableConfig):
            context = config.get("configurable", {})
            if not context.get("jev_run_id"):
                raise ValueError(
                    "Prefetch requires configurable.jev_run_id; reuse it only when resuming that run"
                )
            snapshot = json_value(input_selector(state))
            _, receipts = replace(observation, input_selector=lambda _: snapshot).evaluate(
                name,
                state,
                self._client,
                thread_id=context.get("thread_id"),
                run_id=context.get("jev_run_id"),
            )
            for r in receipts:
                r.snapshot = snapshot
                r.question_spec = json_value(asdict(questions[r.question_id]))
                r.metadata["prefetch_domains"] = list(domains)
            return {"receipts": [r.model_dump() for r in receipts]}

        self._graph.add_node(name, prefetch)

    def _require_receipts(self):
        channel = get_type_hints(self._schema, include_extras=True).get("receipts")
        if add_receipts not in getattr(channel, "__metadata__", ()):
            raise ValueError("State requires receipts: Annotated[list, add_receipts]")

    def map_decision(
        self,
        name: str,
        decision: DecisionNode,
        *,
        items: Callable,
        handlers: Mapping[str, Callable],
        then: str = END,
    ) -> None:
        """Send each item to its own checkpointed decision subgraph.

        Parent state requires map_results: Annotated[list, operator.add].
        Handler receives {item, item_id, receipts, result} and returns {result: ...}.
        Results carry batch_id and item_id; execution order is not an API guarantee.
        """
        import uuid

        self._require_receipts()
        channel = get_type_hints(self._schema, include_extras=True).get("map_results")
        if operator.add not in getattr(channel, "__metadata__", ()):
            raise ValueError("map_decision requires map_results: Annotated[list, operator.add]")
        if set(decision.targets()) - {END} != set(handlers):
            raise ValueError("Map handlers must match decision targets")
        if len(decision.questions) != 1:
            raise ValueError("map_decision evaluates one question independently per item")
        if any(
            decision.edge_for(q.id) and decision.edge_for(q.id).top_k != 1
            for q in decision.questions
        ):
            raise ValueError(
                "Map supports one target per item; top-k belongs to ordinary decision graphs"
            )
        child = JevGraph(_ItemState, self._client)
        child.add_decision(name, decision)
        for target, handler in handlers.items():
            child.add_node(target, handler)
            child.add_edge(target, END)
        child.set_entry_point(name)
        app = child.compile()
        worker_name = name + "__item"

        def fanout(state):
            values = items(state)
            if not isinstance(values, list):
                raise ValueError("Map items selector must return a list")
            batch_id = uuid.uuid4().hex
            return [
                Send(worker_name, {"item": value, "item_id": str(i), "batch_id": batch_id})
                for i, value in enumerate(values)
            ] or [then]

        def worker(state, config: RunnableConfig):
            out = app.invoke(
                {
                    "item": state["item"],
                    "item_id": state["item_id"],
                    "result": None,
                    "receipts": [],
                },
                config,
            )
            return collect(state, out)

        async def aworker(state, config: RunnableConfig):
            out = await app.ainvoke(
                {
                    "item": state["item"],
                    "item_id": state["item_id"],
                    "result": None,
                    "receipts": [],
                },
                config,
            )
            return collect(state, out)

        def collect(state, out):
            receipts = [Receipt(**r) for r in out["receipts"]]
            for r in receipts:
                r.item_id = state["item_id"]
                r.metadata["map_batch_id"] = state["batch_id"]
            return {
                "receipts": [r.model_dump() for r in receipts],
                "map_results": [
                    {
                        "batch_id": state["batch_id"],
                        "item_id": state["item_id"],
                        "result": out["result"],
                    }
                ],
            }

        self._graph.add_node(name, lambda state: {})
        self._graph.add_node(worker_name, RunnableLambda(worker, afunc=aworker))
        self._graph.add_conditional_edges(name, fanout, [worker_name, then])
        self._graph.add_edge(worker_name, then)

    def add_decision(
        self, name: str, decision: DecisionNode, *, prefetch: str | None = None
    ) -> None:
        """Review pauses after evaluation, so resume reuses the stored answer."""
        self._require_receipts()
        client = self._client
        dispatch_name = f"{name}__dispatch"

        def _node(state, config: RunnableConfig):
            context = config.get("configurable", {})
            selected_client = client
            cached = None
            if prefetch is not None:
                if not any(r["node"] == prefetch for r in state["receipts"]):
                    raise ValueError(f"Missing prefetch batch: {prefetch}")
                cached = _batch(state, prefetch)
                if not context.get("jev_run_id"):
                    raise ValueError("Consuming prefetch requires configurable.jev_run_id")
                if any(
                    r.run_id != context.get("jev_run_id") or r.thread_id != context.get("thread_id")
                    for r in cached
                ):
                    raise ValueError("Prefetch batch belongs to another run or thread")
                if name not in cached[0].metadata["prefetch_domains"]:
                    raise ValueError(f"Domain {name} was not registered in prefetch {prefetch}")
                selected_client = _CachedClient(cached)
            _, receipts = decision.evaluate(
                name,
                state,
                selected_client,
                thread_id=context.get("thread_id"),
                run_id=context.get("jev_run_id"),
                call_id=cached[0].call_id if cached else None,
                metering_node=cached[0].metering_node if cached else None,
            )
            if cached is not None:
                indexed = {r.question_id: r for r in cached}
                for r in receipts:
                    source = indexed[r.question_id]
                    r.call_id = source.call_id
                    r.metering_node = source.metering_node
                    r.source_receipt_id = source.id
            if "item_id" in state:
                for r in receipts:
                    r.item_id = state["item_id"]
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
                                "item_id": r.item_id,
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
            if decision.composite and not fallback and not reviews:
                value = decision.composite.evaluate(receipts)
                floor = max(t for t in decision.composite_routes if value >= t)
                for r in receipts:
                    r.composite_value = value
                    r.destinations = [decision.composite_routes[floor]]
                    r.reason = "composite_score"
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
