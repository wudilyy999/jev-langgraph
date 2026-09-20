# 使用指南

## 安装与验证

项目是嵌入业务的 Python 包，执行引擎为 LangGraph，无需 fork。
正式支持 Python 3.11+。本机已准备 Python 3.12 的 `.venv`。

```bash
cd /Users/liyuyang/Downloads/jev-langgraph
.venv/bin/python -m pip install -e '.[dev,sqlite]'
.venv/bin/python -m pytest -q
.venv/bin/python examples/support_triage.py
.venv/bin/python examples/review_resume.py
```

## 原生节点与 JEV 决策混用

以下示例可直接运行，使用离线结果演示人工审核：

```python
from typing import Annotated, TypedDict
from langgraph.graph import END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from jev_langgraph import Choice, DecisionNode, JevGraph, MockJevClient, add_receipts

class State(TypedDict):
    text: str
    result: str
    receipts: Annotated[list, add_receipts]

client = MockJevClient({"intent": ({"billing": 0.6, "tech": 0.4}, 0.6)})
g = JevGraph(State, client)
g.add_node("billing", lambda s: {"result": "账单已处理"})
g.add_node("tech", lambda s: {"result": "已转技术支持"})
g.add_decision("triage", DecisionNode(
    questions=[Choice("intent", ("billing", "tech"), "Which team should handle this ticket?")],
    routes={"billing": "billing", "tech": "tech"},
))
g.set_entry_point("triage")
g.add_edge("billing", END)
g.add_edge("tech", END)
app = g.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-001"}}
paused = app.invoke({"text": "无法查看账单", "result": "", "receipts": []}, config)
print(paused["__interrupt__"][0].value)

# choices 来自人工表单；reviewer 应由业务服务的已认证身份生成。
out = app.invoke(Command(resume={
    "choices": {"intent": "tech"}, "reviewer": "operator-001",
}), config)
print(out["result"])
assert len(client.calls) == 1
```

`add_node`、`add_edge`、`add_conditional_edges` 转发给标准 `StateGraph`。
`compile()` 返回标准 `CompiledStateGraph`，支持 `invoke`、`ainvoke`、`stream`。
同步客户端在异步图中由 LangGraph 线程执行器调用；当前没有原生异步 HTTP 客户端。
内部节点名使用 `<决策名>__dispatch`，业务节点应避免占用这些名字。
JEV 决策的出边由 `routes/edges` 管理。不要再从求值节点添加原生出边，
这会创建绕过审核步骤的独立路径；原生出边应连接业务节点。

## 决策与概率边

每次进入决策域，所有问题合并为一次客户端调用。求值节点持久化结果，
随后调度节点处理审核和条件路由。条件边与人工恢复均读取已保存结果。

```python
from jev_langgraph import Choice, Score, DecisionNode, ProbEdge, Policy

triage = DecisionNode(
    questions=[
        Choice("intent", ("billing", "tech"), "Which teams can handle this ticket?"),
        Score("severity", ("low", "high"), "How severe is this issue?"),
    ],
    edges={
        "intent": ProbEdge({"billing": "billing", "tech": "tech"}, top_k=2),
        "severity": None,
    },
    policy=Policy(auto=0.95, provisional=0.7, review=0.4, version="support-v1"),
    fallback="fallback_node",
)
```

- `routes` 为简写共享路由表；`edges` 按问题 ID 配置，可解决多个问题出现相同标签的歧义。两者互斥。
- `edges` 必须列出每个问题，值为 `None` 表示只记录观察结果，不参与策略和路由。
- `ProbEdge.top_k` 选择概率最大的 k 个标签，允许单独指定 `policy`。同一目标节点去重执行。
- 未选节点保留在编译图中，该轮不调度。多个分支写同一状态键时，应配置 LangGraph reducer。
- 分发后的汇合使用原生 `add_edge(["a", "b"], "join")`，仅在 a、b 均确定会执行时使用。
- 未映射的选项与畸形模型响应会报错，业务分支不会被默认放行。
- 默认模型输入为除 Receipt 外的状态文本。可用 `input_selector=lambda s: s["text"]` 限定模型可见信息。

## 策略与审核

默认阈值为 `0.95 / 0.7 / 0.4`，比较使用 `>=`。

| 区间 | 行为 |
|---|---|
| auto | 调度选中分支 |
| provisional | 调度分支并记录标记，下游可自行追加验证 |
| review | 调用 `interrupt()`，等待人工选择 |
| fallback | 整个决策域转向 `fallback`，不执行其他业务分支 |

多问题中 fallback 优先，其次暂停所有业务分支等待 review。恢复后统一分发。
触发 fallback 且没有配置目标时抛出 `ValueError`。

默认不传 `review`：人工选择直接映射到对应业务节点。
若传 `review="human_handler"`：仍先暂停，恢复后统一进入该处理节点，
该节点从 Receipt 的 `human_choice` 读取人工选择，自行决定后续流程。
这与旧版直接跳转审核节点的行为不同。

审核需要 checkpointer 和稳定 `thread_id`。恢复数据必须包含准确的问题集合、有效标签、非空 reviewer。
框架校验结构；身份认证和对该 thread 的操作权限由业务服务负责。
不要在同一 thread 上并发 invoke/resume。

`confidence` 是官方提供的分布集中程度统计量，独立于选中标签概率。
`Policy(metric="probability")` 可直接按选中标签概率判断。
Noul 没有官方 confidence，策略使用 `max(p_yes, p_no)`，Receipt 记录 `policy_metric="probability"`。
Score 保留期望值 `score`，路由使用最大概率的等级标签。

## 连接真实 JEV

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

直接调用官方 `POST /v1/systemone`，无需 `langchain-typesafe`。
每个问题必须提供明确的 `prompt`；问题 ID 不会被模型用作指令。
客户端强制 HTTPS，拒绝重定向，设置超时；401、429、529、网络错误直接向调用方传播。
本版没有自动网络重试，调用方可为只读模型求值配置有界重试策略。
不要给包含外部写入的整个业务图盲目重试。

协议依据：<https://docs.typesafe.ai/api.md>、<https://docs.typesafe.ai/confidence.md>。
本地协议测试使用模拟 HTTP 响应，真实凭据请求仍需在部署环境执行。

## Receipt、校准与修复循环

Receipt 保留 `chosen`（原始模型建议）、完整分布、策略版本、选中标签、
实际调度目标、人工选择及 reviewer。稳定 ID 的 reducer 支持原记录更新；
checkpoint 保存各步骤历史。`status="dispatched"` 只表示调度，不代表业务成功。

`calibration_table` 与 `expected_calibration_error` 使用 `distribution[chosen]`，
`outcomes` 的键为 Receipt ID、值为原模型建议是否正确。人工选择可作为标签来源，
业务最终结果也可以作为标签来源。`mean_probability` 替代旧字段 `mean_confidence`。
当前没有自动 A/B 流量分配或向提供商上传训练数据。

`build_harness` 返回的状态增加 `fix_rounds` 与 `harness_status`：
`running`、`passed`、`failed`。达到 `max_fix_rounds` 或没有修复器时，
质检失败进入 `failed`。支持传入 `checkpointer`；LLM 节点由使用者提供。
真实客户端使用 harness 时应给 `route_question`、`verify`、`diagnose` 显式填写 prompt。
大循环需适当设置 LangGraph `recursion_limit`，例如 `100`；该限制独立于业务修复次数。

## 部署边界

1. 使用 Python 3.11+，固定实际验证的依赖版本和具体模型版本。
2. 使用持久化 checkpointer，生产多进程部署推荐 PostgreSQL；SQLite 适合单机。
3. API key 从环境或密钥服务读取，限制 Receipt 中的敏感业务数据。
4. 审核接口认证、thread 授权、外部业务操作幂等性由业务应用实现。
5. 评估完成但 checkpoint 尚未提交时若进程崩溃，恢复可能再次调用模型。
   checkpoint 已提交后的审核恢复不会再次调用模型；框架不承诺跨网络 exactly-once。
6. 部署前用真实凭据验证 Choice、Score、Noul、低置信度处理与审核恢复。
7. 时间旅行、子图与观测平台的所有组合尚未完整验证，应按实际部署拓扑增加测试。

迁移到 0.2：Python 最低版本调整至 3.11；移除无实现的 typesafe extra；
Receipt reducer 从追加改为按 ID 合并；审核会真正暂停；多问题会分发所有选中分支，
原先仅做旁路观察的问题需显式设置 `edges[id]=None`。
