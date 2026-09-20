# jev-langgraph

[中文](#中文) | [English](#english)

## 中文

**在标准 LangGraph 执行引擎上，用 JEV 的概率分布驱动决策、审核与可追溯执行。**

`jev-langgraph` 是独立的 Python 扩展包：把 `DecisionNode`、`ProbEdge` 和策略编译为
标准 LangGraph 节点与条件边。业务可以保留普通函数、LLM、工具节点和原生条件边，
只在需要概率判断的位置引入 JEV。无需 fork LangGraph，也无需 LangChain agent 中间件。

```text
业务输入 -> JEV 决策域（一次请求评估多个问题）
              -> auto / provisional -> 普通节点、工具或 LLM
              -> review             -> interrupt -> 人工恢复
              -> fallback           -> 配置的兜底节点
           Receipt 记录概率、策略、审核选择和实际调度目标
```

### 功能

| 能力 | 当前实现 |
|---|---|
| 多问题概率路由 | `Choice` 分类、`Score` 有序评分、`Noul` 是非概率；一个决策域合并一次请求，`ProbEdge` 支持逐问题路由和 top-k 分发 |
| 可配置执行策略 | `Policy` 配置自动执行、暂定执行、人工审核和兜底；概率与提供商 confidence 可分别作为策略指标 |
| 原生 LangGraph 混用 | `add_node`、`add_edge`、`add_conditional_edges` 转发给 `StateGraph`；编译后使用原生 `invoke`、`ainvoke`、`stream` 与 checkpointer |
| 可追溯 Receipt | 保存完整分布、原预测、策略版本、模型版本、请求 ID、人工改选和实际分发目标；恢复时按稳定 ID 更新 |
| 结构化输入 | 保留 JSON 对象、数组与值类型；支持结构化指令及 criteria，`input_selector` 限定发给模型的数据 |
| 用量与预算 | 按请求去重的 token/成本报表；按节点、thread、run 或模型汇总；超预算后提高自动执行门槛 |
| 概率守卫 | `add_gate` 直接按 `P(yes)` 与阈值比较，决定放行或阻止 |
| 决策预取 | `add_prefetch` 合并同一输入快照上的多个决策域；下游复用答案并校验输入、问题定义和运行身份 |
| 复合评分 | `CompositeScore` 对多个有序评分的归一化期望值做加权组合，再按阈值路由 |
| 校准与策略适配 | 按实际模型版本计算校准表/ECE；`Policy.adapt` 从已标注结果生成带证据的新策略版本 |
| 条目级并行 | `map_decision` 使用 LangGraph `Send` 为每项建立独立决策子图，支持逐项审核和批量汇合 |
| 多模型质检循环 | `build_harness` 组织模型选择、生成、质检、诊断与有界修复，返回明确的通过/失败状态 |

### 安装

Python 3.11+。从仓库根目录安装；此方式不依赖 PyPI 发布状态：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Windows 激活命令为 `.venv\Scripts\activate`。需要 SQLite 时使用
`python -m pip install '.[sqlite]'`；开发验证使用 `python -m pip install -e '.[dev,sqlite]'`。

名称约定：`jev-langgraph` 是仓库和发行包名，`jev_langgraph` 是 Python 导入名，
`JevGraph` 是图构建类。当前项目版本为 **0.3.0**。

### 快速开始

以下示例无需 API key。它在真实 LangGraph 引擎上运行，JEV 响应由离线客户端提供。
`confidence=0.6` 触发审核，人工选择技术支持后恢复；模型求值只发生一次。

```python
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
graph = JevGraph(State, client)
graph.add_node("billing", lambda state: {"result": "Billing handled"})
graph.add_node("tech", lambda state: {"result": "Technical support"})
graph.add_node("fallback", lambda state: {"result": "Escalated"})
graph.add_decision("triage", DecisionNode(
    questions=[Choice("intent", ("billing", "tech"), "Which team should handle this ticket?")],
    routes={"billing": "billing", "tech": "tech"},
    fallback="fallback",
    input_selector=lambda state: {"ticket": state["text"]},
))
graph.set_entry_point("triage")
for node in ("billing", "tech", "fallback"):
    graph.add_edge(node, END)

app = graph.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-001", "jev_run_id": "run-001"}}
paused = app.invoke({"text": "I cannot open my invoice", "result": "", "receipts": []}, config)
print(paused["__interrupt__"][0].value)

out = app.invoke(Command(resume={
    "choices": {"intent": "tech"}, "reviewer": "operator-001",
}), config)
assert out["result"] == "Technical support"
assert len(client.calls) == 1
assert out["receipts"][0]["human_choice"] == "tech"
print(out["result"])
```

普通业务节点保持原生语义。JEV 决策的出边通过 `routes` / `edges` 配置，
由审核后的调度节点执行；直接从求值节点添加业务出边会绕过审核。
top-k 未选中的节点仍保留在编译图中，只在该轮不执行。

### 接入真实 JEV

在构建图前，用以下客户端替换离线客户端，并移除示例中对 `client.calls` 的断言：

```python
import os
from jev_langgraph.http_client import HttpJevClient

client = HttpJevClient(
    "https://api.typesafe.ai",
    api_key=os.environ["TYPESAFE_API_KEY"],
    model="jev-latest",
    timeout=10,
)
```

客户端直接调用 [TypeSafe API](https://docs.typesafe.ai/api.md) 的 `POST /v1/systemone`。
真实输出决定执行分支，可能直接执行或兜底；仅在收到 `__interrupt__` 时提交人工恢复。
生产环境应固定经过验证的具体模型版本。密钥由环境或密钥服务注入，不提交到仓库。

### 执行约定与边界

- 默认策略为 `auto >= 0.95`、`provisional >= 0.7`、`review >= 0.4`，更低值进入兜底。阈值可配置。
- 提供商 `confidence` 与选中标签概率分开保存。Noul 使用概率指标；守卫门直接比较 `P(yes)`。
- 求值与调度分别 checkpoint；已保存求值后的人工恢复复用答案。模型错误、未映射选项和缺失兜底显式报错。
- 多问题中兜底优先于审核，审核完成前不执行业务分支；暂定执行只添加标记，下游验证由业务节点实现。
- 预算在模型返回后按单决策域、单运行累计用量执行升级策略。它不能阻止当前请求超额，也不提供全图并发硬预算。
- 预取要求同一输入快照及明确的 `jev_run_id`；新运行使用新 ID，审核恢复沿用原 ID。
- `Policy.adapt` 在运行之间显式拟合和替换策略。当前不包含自动 A/B 分流、训练数据上传或图结构自修改。
- Receipt 的 `dispatched` 表示已调度，业务成功需业务自身确认。应用负责审核认证、thread 授权、外部操作幂等与重试。
- `InMemorySaver` 用于示例；部署时配置持久化 checkpointer。求值 checkpoint 提交前的崩溃可能导致再次请求模型。
- 当前 HTTP 客户端为同步实现，在异步图中由 LangGraph 执行器调用；没有自动 HTTP 重试。

### 测试与文档

```bash
python -m pip install -e '.[dev,sqlite]' build
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
python examples/support_triage.py
python examples/review_resume.py
python -m build
```

集成测试使用正式安装的 LangGraph，覆盖原生节点/条件边混用、标准编译图类型、
异步与流式执行、审核恢复、SQLite 关闭重开、预取和 Send 子图。
CI 配置 Python 3.11 / 3.12 与 LangGraph 0.6.11 / 1.2.11 的四组合矩阵，
并在源码目录之外验证构建后的 wheel。无需克隆上游源码。
JEV HTTP 测试使用可控响应；真实服务鉴权、模型效果和生产数据库需在部署环境验证。

- [使用指南](USAGE.md)：原生 API 混用、路由、审核恢复、部署和迁移。
- [JEV-native 能力](NATIVE.md)：八项新增能力的接口、示例与精确语义。
- [可运行示例](examples/) 与 [测试](tests/)。

许可证：[MIT](LICENSE)。独立集成项目，与 LangGraph / TypeSafe 的官方项目身份分开。

## English

**Use JEV probability distributions to drive decisions, human review, and auditable execution on the standard LangGraph engine.**

`jev-langgraph` is a standalone Python extension. It compiles `DecisionNode`, `ProbEdge`,
and execution policies into ordinary LangGraph nodes and conditional edges. Keep your
existing functions, LLMs, tools, and native edges; introduce JEV where probabilistic
decisions are useful. No LangGraph fork or LangChain agent middleware is required.

### Features

| Capability | Implementation |
|---|---|
| Multi-question routing | `Choice`, ordinal `Score`, and yes/no `Noul`; one request per decision domain, per-question routes and top-k dispatch through `ProbEdge` |
| Execution policies | Configurable auto, provisional, review, and fallback bands; choose provider confidence or selected-label probability as the policy metric |
| Native integration | Forward `add_node`, `add_edge`, and `add_conditional_edges` to `StateGraph`; compile to a standard graph with `invoke`, `ainvoke`, `stream`, and checkpointing |
| Decision receipts | Full distributions, original predictions, policy/model versions, request IDs, human overrides, and actual dispatch targets, merged by stable receipt ID |
| Structured inputs | JSON state, structured instructions and criteria; restrict provider-visible data with `input_selector` |
| Metering and budgets | Deduplicate token usage by request; group reports by node, thread, run, or model; raise the auto threshold after observed budget overrun |
| Probability gates | `add_gate` compares `P(yes)` directly against an allow/block threshold |
| Speculative prefetch | Batch future domains on one input snapshot; downstream decisions reuse answers with snapshot, question, and run checks |
| Composite scores | Weighted normalized expectations from multiple ordinal questions, followed by threshold-based routing |
| Calibration and adaptation | Per-model calibration tables/ECE; `Policy.adapt` produces a new versioned policy from labeled outcomes |
| Per-item parallelism | `map_decision` uses LangGraph `Send` and independent decision subgraphs, with per-item review and batch joining |
| Multi-model quality loop | `build_harness` assembles model routing, generation, verification, diagnosis, and bounded repair, with explicit passed/failed status |

### Installation and Quick Start

Requires Python 3.11+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

On Windows, activate with `.venv\Scripts\activate`. Use `'.[sqlite]'` for SQLite
checkpoint support or `-e '.[dev,sqlite]'` for development. This installation method
does not depend on a PyPI release.

The distribution/repository is `jev-langgraph`, the import is `jev_langgraph`, and
the graph builder is `JevGraph`. The current project version is **0.3.0**.

Run the complete [quick-start code above](#快速开始), which uses English identifiers
and output. It requires no API key: the real LangGraph engine runs a controlled JEV
response, pauses for review, and resumes into technical support after a human override.
The assertions verify one model evaluation and the recorded human choice.

The [live-client snippet](#接入真实-jev) replaces `MockJevClient` with `HttpJevClient`.
Set `TYPESAFE_API_KEY` through your environment or secret manager, construct the graph
with that client, and remove the mock-only `client.calls` assertion. Live outputs may
execute or fall back immediately; submit a resume only when `__interrupt__` is present.
The client calls the documented [TypeSafe API](https://docs.typesafe.ai/api.md) at
`POST https://api.typesafe.ai/v1/systemone`. Pin a validated model version for deployment.

### Execution Semantics

- Default thresholds: auto at 0.95, provisional at 0.7, review at 0.4, fallback below 0.4. All are configurable.
- Provider confidence is distinct from selected-label probability. Noul policies use probability; gates compare `P(yes)` directly.
- Evaluation and dispatch are separate checkpointed steps. Human resume reuses saved evaluation results. Invalid responses, unmapped selections, and missing fallback targets fail explicitly.
- Domain fallback takes priority over review; review pauses business dispatch. Provisional execution records a flag for downstream checks implemented by the application.
- Configure JEV outgoing routes through `routes` / `edges`. Adding a native business edge directly from the evaluation node bypasses review. Ordinary business nodes retain native semantics.
- Top-k changes which nodes execute on a run; unselected nodes remain in the compiled graph. Concurrent writes need appropriate LangGraph reducers.
- Budgets escalate after a response, based on measured cumulative usage within a domain and run. They cannot prevent the current request from exceeding budget and are not shared parallel hard limits.
- Prefetch requires the same input snapshot and an explicit `jev_run_id`. Use a new ID for a new run and retain it when resuming.
- Policy adaptation is an explicit between-run empirical fit. Automatic A/B allocation, training-data upload, and self-modifying topology are outside the current implementation.
- A dispatched Receipt records scheduling, not successful business execution. Applications own reviewer authentication, thread authorization, idempotent side effects, and operational retries.
- Use a durable checkpointer in deployment. `InMemorySaver` is for examples. A crash before evaluation is checkpointed may repeat a model request.
- The HTTP client is synchronous and runs through LangGraph's executor in asynchronous graphs. It does not retry HTTP requests automatically.

### Verification and Documentation

```bash
python -m pip install -e '.[dev,sqlite]' build
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
python examples/support_triage.py
python examples/review_resume.py
python -m build
```

Integration tests run against installed LangGraph releases, covering native node/edge
mixing, standard compiled graph types, async/stream execution, review/resume, SQLite
reopening, prefetch, and Send subgraphs. CI defines a four-job matrix for Python
3.11/3.12 and LangGraph 0.6.11/1.2.11 and verifies the built wheel outside the source
directory. Cloning LangGraph's source is unnecessary for these checks.

Provider HTTP tests use controlled responses. Live authentication, model quality, and
the production database remain deployment checks. CI configuration describes the
intended checks; consult the actual Actions run for their remote status.

Version 0.3 defaults to structured JSON inputs; custom clients must accept objects
and arrays or explicitly select text input. Receipt calibration uses the
`mean_probability` column. Detailed guides are currently in Chinese; the runnable
Python examples use the same public API:

- [Usage guide](USAGE.md): routing, native APIs, review/resume, deployment, and migration.
- [Native capabilities](NATIVE.md): structured input, metering, gates, prefetch, scoring, model tracing, adaptation, and batch decisions.
- [Runnable examples](examples/) and [tests](tests/).

License: [MIT](LICENSE). An independent integration project, with no claim of official
LangGraph or TypeSafe affiliation.
