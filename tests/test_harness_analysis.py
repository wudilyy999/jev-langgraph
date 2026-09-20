from typing import Annotated, TypedDict

from jev_langgraph import Choice, MockJevClient, Noul, Receipt, add_receipts
from jev_langgraph.analysis import (
    calibration_table,
    expected_calibration_error,
    flip_rate,
    policy_histogram,
)
from jev_langgraph.harness import build_harness


class S(TypedDict):
    text: str
    receipts: Annotated[list, add_receipts]


def test_harness_pass_path():
    client = MockJevClient(
        {
            "model": ({"fast": 0.9, "strong": 0.1}, 0.9),
            "ok": ({"yes": 0.99, "no": 0.01}, 0.99),
        }
    )
    app = build_harness(
        S,
        client,
        models={
            "fast": lambda s: {"text": s["text"] + " [fast]"},
            "strong": lambda s: {"text": s["text"] + " [strong]"},
        },
        verify=Noul("ok", prompt="Is the answer good?"),
    )
    out = app.invoke({"text": "q", "receipts": []})
    assert "[fast]" in out["text"]
    kinds = {(r["node"], r["question_id"]) for r in out["receipts"]}
    assert ("route", "model") in kinds
    assert ("verify", "ok") in kinds


def test_harness_fix_loop():
    # first verify says no, diagnose picks fix, second verify says yes
    verdicts = [({"yes": 0.1, "no": 0.9}, 0.9), ({"yes": 0.9, "no": 0.1}, 0.9)]
    state = {"i": 0}

    class SeqClient(MockJevClient):
        def decide(self, state_text, questions):
            if questions[0].id == "ok":
                dist, conf = verdicts[min(state["i"], len(verdicts) - 1)]
                from jev_langgraph.client import Answer, DecideResponse

                return DecideResponse(
                    answers={"ok": Answer("ok", "noul", dist, max(dist, key=dist.get), conf)}
                )
            return super().decide(state_text, questions)

    def fix(s):
        state["i"] += 1
        return {"text": s["text"] + " [fixed]"}

    client = SeqClient(
        {
            "model": ({"fast": 1.0}, 1.0),
            "why": ({"shallow": 0.8, "wrong": 0.2}, 0.8),
        }
    )
    app = build_harness(
        S,
        client,
        models={"fast": lambda s: {"text": s["text"] + " [fast]"}},
        verify=Noul("ok"),
        diagnose=Choice("why", options=("shallow", "wrong")),
        fixer=fix,
    )
    out = app.invoke({"text": "q", "receipts": []})
    assert "[fixed]" in out["text"]


def _r(id_, conf, margin_pair=(0.9, 0.1), policy="auto"):
    a, b = margin_pair
    return Receipt(
        id=id_,
        node="n",
        question_id="q",
        kind="choice",
        distribution={"x": a, "y": b},
        chosen="x",
        confidence=conf,
        policy=policy,
        latency_ms=1,
    )


def test_flip_rate():
    rs = [_r("1", 0.9, (0.55, 0.45)), _r("2", 0.9), _r("3", 0.9, (0.51, 0.49))]
    # margins: 0.10 (not < 0.1), 0.80, 0.02 -> only id 3 flips
    assert abs(flip_rate(rs, 0.1) - 1 / 3) < 1e-9
    assert abs(flip_rate(rs, 0.11) - 2 / 3) < 1e-9


def test_policy_histogram():
    rs = [
        _r("1", 0.9, policy="auto"),
        _r("2", 0.8, policy="provisional"),
        _r("3", 0.2, policy="fallback"),
    ]
    h = policy_histogram(rs)
    assert h == {"auto": 1, "provisional": 1, "fallback": 1}


def test_calibration_and_ece():
    rs = [_r(str(i), 0.9) for i in range(10)]
    outcomes = {str(i): i < 9 for i in range(10)}  # 9/10 correct at 0.9 conf
    table = calibration_table(rs, outcomes)
    assert len(table) == 1
    assert abs(table[0]["accuracy"] - 0.9) < 1e-9
    assert expected_calibration_error(rs, outcomes) < 0.05
