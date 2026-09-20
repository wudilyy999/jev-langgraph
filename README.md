# jev-langgraph

JEV-native probabilistic decisions compiled to standard LangGraph graphs.
Native nodes and edges remain available. No LangGraph fork or LangChain agent
middleware is required.

## Install

Python 3.11+:

```bash
python -m pip install -e '.[dev,sqlite]'
python -m pytest -q
python examples/support_triage.py
python examples/review_resume.py
```

See [使用指南](USAGE.md) for complete runnable examples, native API integration,
multi-question/top-k routing, review/resume, direct JEV access and deployment checks.

## Execution Contract

- One client request per decision evaluation; conditional routing reads the saved result.
- Evaluation and dispatch are separate checkpointed nodes. Human resume reuses the evaluation.
- `ProbEdge` supports per-question routes, top-k, and per-edge policies. `None` observes only.
- A domain-level fallback suppresses all business branches; review pauses before dispatch.
- Missing fallback targets, unmapped selections and invalid provider responses fail explicitly.
- Receipts retain the original prediction, distribution, policy version, human override,
  and actual dispatch targets. Their reducer merges by stable ID.
- The harness enforces its repair limit and returns an explicit passed/failed status.
- Probability calibration uses selected-label probability, separate from provider confidence.

The direct client implements the documented [TypeSafe API](https://docs.typesafe.ai/api.md)
at `https://api.typesafe.ai/v1/systemone`. Tests use controlled responses; live provider
verification requires deployment credentials. No automatic HTTP retries are performed.

## Verification and Limits

Tests cover routing consistency, batching, fan-out, review override, invalid resume,
thread isolation, async execution, streaming, bounded repair, HTTP contracts and
SQLite checkpoint reopening. CI defines Python 3.11/3.12 and LangGraph 0.6.11/1.2.11 jobs.
Local execution and remote CI results should be reported separately.

Local verification: all 52 tests passed on Python 3.12 with both LangGraph 0.6.11
and 1.2.11, including SQLite persistence. Python 3.11 is configured in CI but was
not run locally. Both offline examples were executed successfully.

Persistence does not make external business side effects exactly-once. A crash before
an evaluation checkpoint commits can repeat the model request. Applications own
authentication, thread authorization, idempotent business actions and operational retries.
Use a durable production checkpointer; `InMemorySaver` is for local examples.

Version 0.2 introduces review pauses, multi-branch routing, receipt upserts and the
`mean_probability` calibration column. See the migration notes in the usage guide.
