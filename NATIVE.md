# JEV-native 能力（0.3）

以下接口直接扩展 LangGraph，编译产物仍为标准图。默认输入由文本改为结构化 JSON。
真实 HTTP 协议依据 https://docs.typesafe.ai/api.md；本地测试通过模拟响应验证协议。

## 1. 结构化输入、指令与 criteria

```python
from jev_langgraph import Choice, DecisionNode

decision = DecisionNode(
    [Choice("intent", ("billing", "tech"),
            prompt={"question": "Read `ticket` and select a team", "context_field": "ticket"},
            criteria={"billing": {"handles": ["invoices", "refunds"]}, "tech": "Product issues"})],
    routes={"billing": "billing", "tech": "tech"},
    input_selector=lambda state: {"ticket": state["ticket"]},
)
```

默认 `structured_state` 保留嵌套 dict/list、数值、布尔值、null，递归把 LangChain 消息
转成 role/content；不传 Receipt 和 `__jev_` 内部字段。非 JSON 对象明确报错。
`prompt` 接受字符串、对象、数组；Choice criteria 按标签配置，Score criteria 按等级顺序配置，
Noul criteria 使用 true/false 键。需要旧文本方式时显式传 `input_selector=state_to_text`。

## 2. 请求计量、运行报表与预算

Receipt 正式字段：`call_id`、`input_tokens`、`output_tokens`、`thread_id`、`run_id`、
`model`、`request_id`。同一模型请求回答多个问题时共享 call_id，报表只计算一次。
预取消费记录沿用源 call_id，成本归属于发起调用的枢纽域 `metering_node`。

```python
from jev_langgraph import Receipt, cost_report, Policy

config = {"configurable": {"thread_id": "ticket-001", "jev_run_id": "run-001"}}
out = app.invoke(input_state, config)
records = [Receipt(**r) for r in out["receipts"]]
report = cost_report(records, group_by="node", run_id="run-001",
                     input_per_million=0.1, output_per_million=0)
thread_report = cost_report(records, group_by="thread_id")
policy = Policy(token_budget=1000, budget_auto=1.0, version="budget-v1")
```

价格由调用方提供，单位为每百万 token，同一货币。未报告用量时计量和成本为 None，
同时记录 unmetered_requests。本地成本报表不会猜测价格或将未知请求当成免费。

预算是**单域、单运行已观测总 token 的事后策略门槛**，包含当前请求。
超预算时 auto 提升到 budget_auto，未达到该阈值的结果升级到 review/fallback；
不会退回 provisional 自动执行。预算策略遇到未计量用量明确报错。
这无法阻止当前请求自身超额，也不是供应商的硬 token 限流。
并行 map 条目各自有独立域上下文；本版不提供全图并发共享的硬预算。
每次新业务运行生成新的 jev_run_id，审核恢复继续使用原 ID。

## 3. 图守卫门

```python
from jev_langgraph import Noul
g.add_gate("guard", Noul("allowed", "Is this write operation permitted?"),
           threshold=0.95, allow="write_tool", block="deny")
```

P(yes) >= threshold 时执行 allow，否则执行 block（默认 END）。问题必须表述为
“允许执行吗”；如果问的是“有风险吗”，需自行转换问题或交换语义，不能直接套用。
Receipt 保留 gate_probability、gate_threshold、gate_allowed。阈值独立于 argmax/confidence。
门控只约束经过该守卫的图路径，业务授权和其他路径由应用控制。

## 4. 投机预取

```python
selector = lambda s: {"ticket": s["ticket"]}
first.input_selector = selector
second.input_selector = selector
g.add_prefetch("hub", {"first": first, "second": second}, input_selector=selector)
g.add_decision("first", first, prefetch="hub")
g.add_decision("second", second, prefetch="hub")
g.add_edge("hub", "first")
# first 的路由可以继续到 second，也可以跳过 second。
```

枢纽合并问题一次调用；相同 ID 必须拥有相同定义。下游只消费被注册域的问题，
输入快照、问题定义、thread_id、jev_run_id 必须一致；不一致会报错，不会暗中重新请求。
必须显式设置 jev_run_id。重新运行枢纽生成新批次；按最近批次消费。
多个预取域必须针对同一个快照，不能预取依赖尚未生成内容的决策。
快照保存在 Receipt 中用于校验，可能包含敏感数据，应限制 input_selector 和持久化访问。
下游 source_receipt_id 指向源记录，审核恢复和 SQLite 重启恢复均复用已有结果。

## 5. 复合评分

```python
from jev_langgraph import CompositeScore, Score, DecisionNode
scoring = DecisionNode(
    [Score("quality", ("low", "high"), "Assess quality"),
     Score("coverage", ("none", "partial", "full"), "Assess coverage")],
    edges={"quality": None, "coverage": None},
    composite=CompositeScore({"quality": 3, "coverage": 1}),
    composite_routes={0.0: "reject", 0.7: "accept"},
)
```

每个分数为等级索引的期望值；先除以最大等级索引归一化到 [0,1]，再做权重平均。
权重必须有限、非负且总和为正。路由选不大于结果的最大阈值，必须提供 0 阈值。
结果写入各 Receipt 的 composite_value 和实际 destinations。
若同时配置参与路由的边，fallback/review 优先；无审核时复合路由替代各问题出边。
需要对评分整体进行人工审核时，可在复合结果之后连接原生审核节点。

## 6. 模型版本追踪与漂移对比

```python
from jev_langgraph import calibration_by_model
comparison = calibration_by_model(records, outcomes)
```

返回每个实际 model 版本的已标注数、ECE 和校准表；没有标签的模型不产生虚构结果。
outcomes 为 Receipt ID 到“原模型建议是否正确”的映射。request_id 优先读响应体，
缺失时读 x-request-id 响应头；未提供就保留 None。

## 7. 校准驱动阈值适配

```python
updated = old_policy.adapt(records, outcomes,
    node="triage", question_id="intent", model="jev-1.13.0",
    version="support-calibrated-v2", target_accuracy=0.95, min_samples=30)
decision.policy = updated
```

按域、问题、实际模型版本筛选标签；选满足最小样本量和目标经验准确率的最低尾部阈值。
auto 只上调，返回全新的带版本 Policy，原策略保持不变。无支持阈值时关闭 auto，
即使值为 1 也进入审核；低于 review 的值仍走 fallback。
`adaptation` 保存来源版本、筛选范围、样本量、目标和实际准确率，随 Receipt 写入。
这是确定性的经验拟合，不是未来准确率保证；应使用独立验证集后再部署策略。
Noul 应使用 `Policy(metric="probability")`。在运行之间替换配置，避免中途更改运行图。

## 8. 条目级 map_decision

父状态增加 `map_results: Annotated[list, operator.add]`。

```python
g.map_decision("passages", decision,
    items=lambda s: s["passages"],
    handlers={
        "keep": lambda s: {"result": {"item": s["item"], "keep": True}},
        "drop": lambda s: {"result": {"item": s["item"], "keep": False}},
    }, then="collect")
```

decision 包含一个问题、top_k=1，input_selector 通常为 `lambda s: s["item"]`。
每项通过 Send 启动独立决策子图；处理器收到 item、item_id、receipts、result。
处理器更新 result；父图合并 `{batch_id, item_id, result}` 和每项 Receipt。
item_id 为输入索引字符串，结果顺序不保证；空列表不调用模型，直接到 then。
handlers 必须覆盖所有非 END 目标，包括 fallback/review 处理器。
async 处理器使用 `ainvoke`。正常条目完成后，其他条目仍可暂停等待人工审核。
多个审核使用 `Command(resume={interrupt_id: {choices: ..., reviewer: ...}, ...})`，
审核 payload 含 item_id。每个条目的恢复复用自己的模型结果。
`then` 接收整批结果；需要控制并发时传 LangGraph `max_concurrency`。

## 验证入口

`tests/test_native_data.py`、`test_native_routing.py`、`test_adaptation_budget.py`、
`test_prefetch_map.py` 覆盖上述新增语义；原有执行层测试继续运行。
真实凭据调用、生产数据库和实际业务授权链路仍由部署环境验证。
