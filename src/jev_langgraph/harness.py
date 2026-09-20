"""Multi-model harness template: JEV as the nervous system, LLMs as muscle.

Topology:
    input
      -> JEV route (which model?)          ~100ms, one request
      -> LLM-A / LLM-B (generate)
      -> JEV verify (does it pass?)        ~100ms, one request
      -> pass: end | fail: JEV diagnose -> targeted fix -> verify again

All JEV hops are decisions; only the middle generate/fix nodes are LLMs.
"""

from __future__ import annotations

from typing import Callable, Mapping, TypedDict, get_type_hints

from .client import Choice, JevClient, Noul, Question, options_for
from .decision import DecisionNode
from .graph import JevGraph
from .policy import Policy


def build_harness(
    state_schema,
    client: JevClient,
    models: Mapping[str, Callable],
    verify: Noul,
    diagnose: Question | None = None,
    fixer: Callable | None = None,
    route_question: Choice | None = None,
    policy: Policy | None = None,
    max_fix_rounds: int = 2,
    checkpointer=None,
):
    """Assemble a JEV-gated generate-verify-fix graph.

    `models` maps route value -> LLM node fn. `verify` is the quality gate.
    `diagnose`+`fixer` add a repair loop; without them failures end the run.
    """
    if type(max_fix_rounds) is not int or max_fix_rounds < 0:
        raise ValueError("max_fix_rounds must be a nonnegative integer")
    if not models or (diagnose is None) != (fixer is None):
        raise ValueError("Provide models and either both diagnose/fixer or neither")
    schema = TypedDict(
        "HarnessState",
        {
            **get_type_hints(state_schema, include_extras=True),
            "fix_rounds": int,
            "harness_status": str,
        },
    )
    g = JevGraph(schema, client)
    policy = policy or Policy()
    route_question = route_question or Choice("model", options=tuple(models))
    if set(route_question.options) != set(models):
        raise ValueError("Model routing options must match model names")
    g.add_node("initialize", lambda s: {"fix_rounds": 0, "harness_status": "running"})
    g.add_edge("initialize", "route")
    g.add_node("passed", lambda s: {"harness_status": "passed"})
    g.add_node("failed", lambda s: {"harness_status": "failed"})
    g.add_edge("passed", "__end__")
    g.add_edge("failed", "__end__")

    g.add_decision(
        "route",
        DecisionNode(
            questions=[route_question],
            routes={k: f"gen_{k}" for k in models},
            policy=policy,
            fallback="failed",
        ),
    )
    for name, fn in models.items():
        g.add_node(f"gen_{name}", fn)
        g.add_edge(f"gen_{name}", "verify")

    verify_routes: dict[str, str] = {"yes": "passed"}
    if diagnose is not None and fixer is not None:
        verify_routes["no"] = "repair_gate"
        g.add_node("repair_gate", lambda s: {})
        g.add_conditional_edges(
            "repair_gate",
            lambda s: "repair" if s["fix_rounds"] < max_fix_rounds else "exhausted",
            {"repair": "diagnose", "exhausted": "failed"},
        )
        g.add_decision(
            "diagnose",
            DecisionNode(
                questions=[diagnose],
                routes={o: "fix" for o in options_for(diagnose)},
                policy=policy,
                fallback="failed",
            ),
        )
        g.add_node("fix", fixer)
        g.add_node("count_fix", lambda s: {"fix_rounds": s["fix_rounds"] + 1})
        g.add_edge("fix", "count_fix")
        g.add_edge("count_fix", "verify")
    else:
        verify_routes["no"] = "failed"
    g.add_decision(
        "verify",
        DecisionNode(questions=[verify], routes=verify_routes, policy=policy, fallback="failed"),
    )

    g.set_entry_point("initialize")
    return g.compile(checkpointer=checkpointer)
