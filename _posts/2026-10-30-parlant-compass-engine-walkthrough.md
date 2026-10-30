---
title:      Parlant 的新方向：从 guideline 到统一 Rule，Compass 引擎源码解读
subtitle:   Rule 抽象 · 三档匹配 · Jev 批量判决 · 加速手段
date:       2026-10-30 11:45:00 +0800
author:     张弦
tags:
    - Parlant
    - Agent
    - 规则引擎
    - 源码解读
---

## 摘要

Parlant 是一个面向客户对话的 Agent 行为控制框架。它的新文档站（next.parlant.io）里出现了一批 PyPI 上还没有的概念：`create_policy`、`Effort`、`Weight`、`signals`、`prioritize_over`……我顺着这些名字找到了尚未发布的 `compass-wip` 分支，读了里面的 Compass 引擎。

读完后的几个结论：

- **文档里的新 API 还没发布**。PyPI 最新是 3.3.2（2026-04-28），里面没有 `create_policy`。新概念只存在于 `compass-wip` 分支的一次 WIP 提交中，目前只能算原型。
- **核心设计是统一**：policy、observation 是同一个 `Rule`，区别只是有没有 `condition`；优先级、依赖、蕴含这些关系不放进规则里，单独存关系表。
- **每轮的判断成本被分了三档**：有的规则不判断，有的只用 embedding，只有必须拿到"本轮是否命中"的规则才交给模型。
- **加速靠缓存预热、并发、批量判决和 hedging**，收益依赖提示词前缀的字节级稳定。

本文只记录代码里能核实的部分。所有结论来自静态阅读，没有运行过。

---

## 1. 读的是什么

| 项目 | 内容 |
|---|---|
| 仓库 | [emcie-co/parlant](https://github.com/emcie-co/parlant)，Apache 2.0 |
| 已发布版本 | PyPI 3.3.2（2026-04-28），API 为 guideline、observation、journey |
| 本文读的分支 | `compass-wip`，提交 c30226c（2026-10-28），标题 "WIP compass drop" |
| 默认分支 | `develop` 自 2026-06-26 起没有新提交 |

也就是说，下面讲的 Compass 引擎是一个还没发布的方向。它会不会合并、什么时候发布，我没有办法判断。

---

## 2. 概念演进

![图 1：3.3.2 与 compass-wip 的概念对照](/img/posts/parlant-compass/01-concepts.svg)

| 对象 | 3.3.2（已发布） | compass-wip（未发布） |
|---|---|---|
| 行为规则 | `create_guideline` | `create_policy`：只有 `content`，没有 condition |
| 触发观察 | `create_observation` | `create_observation`：带 condition，和 policy 同为 `Rule` |
| 流程 | `create_journey` | 在 `relational_resolver.py` 的文档字符串里标为范围之外 |
| 工作量控制 | 未找到对应项 | `Effort` 五档，规则可用 `trigger_effort` 抬高当轮档位 |
| 重要程度 | 未找到对应项 | `Weight` 三档 |
| 关系 | 有基础关系（我没有逐项核对具体类型） | `prioritize_over`、`depend_on`、`entail`、`reevaluate_after` |
| 判断辅助 | 未找到对应项 | `signals` / `anti_signals`、`evaluation_func` |
| 命中后动作 | 未找到对应项 | `on_matched`、`on_discovered`、`labels`、复核开关 |

表里"未找到对应项"的意思是我在 3.3.2 的 `sdk.py` 里没搜到，不代表之前完全没有类似能力。

官方文档给出的理由：

- 真实对话不是线性的，节点切换和 agent 交接会丢上下文，所以保持一个 agent，每轮动态拼上下文。
- policy 有三种写法：policy document（一页手册）、procedure（编号步骤，近似 SOP）、steering guideline（一句话微调）。
- 一个 procedure 建议整体写成一条 policy，不要拆成"当 X 时问订单号"这类碎片。

代码里与此对应的事实是：policy 和 observation 是同一个 `Rule` dataclass，区别只是有没有 `condition`。

---

## 3. 核心架构

### 3.1 统一的 Rule

`core/rules.py:65` 的 `Rule` 包含 `content`、`condition`、`title`、`weight`、`effort`、`priority`、`signals`、`anti_signals`、`labels`、`message_review`、`tool_review`、`groups`、`metadata`。至少要有 `content` 或 `condition` 之一。

关系（优先级、依赖、蕴含、重新评估）不放在 Rule 里，存在单独的关系表，类型有 entailment、priority、dependency、dependency_any（组内 OR）、reevaluation、overlap。规则 id 在不指定时由内容哈希生成。

### 3.2 一轮对话的流程

1. `match_early`：只跑确定性的早期代码判断，不调模型。
2. 成本闸门：有成本控制策略时，先判断这一轮是否允许。
3. `preload`：加载可用规则、上下文变量、已发现集。
4. `fill`：并行跑规则匹配、工具打分、术语召回，然后选工具。
5. 响应循环：模型生成和工具调用，每步工具执行后重判受影响的规则。
6. 回复发出后，在后台压缩、修剪已发现规则，并为下一轮预热缓存。

### 3.3 三档匹配

![图 2：规则筛选的三档分流与关系解析](/img/posts/parlant-compass/02-matching-ladder.svg)

| 档位 | 怎么处理 | 有本轮判决吗 |
|---|---|---|
| NONE | 直接准入已发现集，不判断 | 没有 |
| RECALL | embedding 加逐规则逻辑回归，只决定是否进发现集 | 没有 |
| RANK | 不在已发现集时仍走召回；再由排序模型判本轮是否命中 | 有 |

起始档位由 effort 与 weight 的矩阵决定，规则很少（≤ 5 条）时直接用 NONE。以下情况会被强制升到 RANK：带 `labels`、被其他规则依赖、有 priority 关系、关联工具（effort ≥ MEDIUM）、声明了 `trigger_effort`、要求复核、有 `on_matched` 回调、显式指定 rank 或 distill 评估。原因是这些附件只有在拿到"本轮官方判决"后才会触发。

### 3.4 两类判决

- **discovery**：回答"对话是否让这条规则对本会话持续相关"，通过的规则进入已发现集并保留到被修剪（粘性）。
- **turn 评估**：回答"这条规则本轮是否匹配"，结果只对本轮有效。

系统提示里放完整的已发现集，每轮再单独列出本轮匹配。

### 3.5 三种判断方式

- **召回**（`rule_recaller.py`）：对话做 embedding，每条规则一个离线训练的逻辑回归。`signals` 是正例，`anti_signals` 和其他规则的正例是负例。对历史按最大值池化，并有防止规则来回进出的准入记录。
- **Jev 批量判决**（`rule_ranker.py`）：effort 为 MIN、LOW、MEDIUM 且配了 decision 模型时，把对话当作状态，每条规则当作一个 Proposition，一次请求返回所有规则的概率，≥ 0.5 算命中。提示里要求"拿不准时偏 yes"，因为这只是第一遍过滤。
- **逐条打分**：effort 为 HIGH、MAX，或没有 decision 模型时，小模型对每条规则单独发请求打 1 到 5 分，≥ 4 算命中，并带解释。

批量判决的请求结构大致是：

```
state     = 系统角色 + 对话历史 + "# CURRENT TURN" + 工具事件
questions = { rule_id: Proposition(
                instructions = { rule: "Title/Policy 文本 + 术语 + 依赖上下文",
                                 question: "该规则是否适用于对话当前状态，尤其是最近一句？" },
                criteria     = { true:  signals 作为示例,
                                 false: anti_signals 作为示例 } ) }
→ answers = { rule_id: { probability } }
```

`signals` 在两处用法不同：召回拿它们训练分类器，只能度量相似度；Jev 把它们当作判据的示例，并在提示里明说"这些是示例，不是要匹配的模式"。

代码里写的 OpenRouter 默认模型是：排序用 `openai/gpt-oss-20b:nitro`，判决用 `~typesafe/jev-latest`。这是代码中的字符串，我没有独立验证这些模型。

### 3.6 分轮评估与关系解析

一轮里候选规则数 ≥ 8 才启用分轮，最多 2 轮：非末轮只评估前置条件已满足的规则，末轮评估所有"未被判为不满足"的规则。不足 8 条时一轮跑完。

关系解析是纯代码，不调模型：

- 先展开 entailment，再过滤未满足的依赖。
- 依赖只能由"本会话曾经匹配过"和"同一轮的匹配"满足，发现准入不算。
- priority 只作用于本轮匹配集，先关系型再数值型；输的规则降为只准入，并记入"本轮撤销"列表，提示词里会标出。

### 3.7 数据模型

规则、规则向量、规则与分组关联、规则与工具关联、关系各一张表。运行时状态（已发现集、曾匹配规则）存在 session 的 metadata 里，不在规则表中。

---

## 4. 关键机制

### 4.1 工具

工具通过规则与工具的关联表挂在规则上，而不是挂在 agent 上。

- **选择**：`ToolRecaller` 把工具分成三个桶（本轮命中规则的工具、已发现规则的工具、其余工具），每桶按 embedding 相关度排序，最多开放 16 个。工具没有规则那样的准入与淘汰生命周期。
- **参数**：`ToolParameterOptions.source` 取 any、context、customer，分别表示可从任何地方提取、只能从上下文提取、必须由客户在对话里提供且不能猜。
- **高风险工具**：`consequential=True` 的工具在提示里标 CONSEQUENTIAL，系统提示要求先向用户确认；提议的调用里只要有这类工具，就强制触发 tool review。但 `Effort.MIN` 时一律不做 review，这时只剩提示词层面的约束。
- **调用**：模型在响应循环里自己发起调用，一步内的多个调用并发执行；失败和超时（默认 180 秒）收成错误结果回给模型。
- **重判与临时规则**：`reevaluate_after(tool)` 在工具执行后重判相关规则；工具还能通过 `ToolResult(rules=[...])` 返回只在本轮生效的临时规则。

### 4.2 两个档位机制

- **Effort**：MIN、LOW、MEDIUM、HIGH、MAX 五档，决定引擎做多少工作。当轮有效档位取 agent 默认值与本轮命中规则 `trigger_effort` 的最大值，也参与模型路由。
- **service_tier**：priority、standard、flex 三档，决定模型请求向提供商买哪一类服务（更快、普通、更便宜但更慢）。我只核对了 OpenAI 与 Gemini 适配器的映射。

两者独立、可以叠加。

### 4.3 加速

![图 3：加速手段按时间点分布](/img/posts/parlant-compass/03-speedups.svg)

- **缓存预热**：会话一创建就预热稳定的系统提示前缀，每轮回复发出后为下一轮预热。预热对象是排序器和响应循环的稳定前缀，每轮变化的内容放在缓存断点之后。扇出规模 ≥ 5 才值得预热，因为预热本身是一次请求，缓存写入还比普通输入贵。
- **并发与批量**：匹配、工具打分、术语召回并行；一轮内多个工具调用并发；Jev 一次请求判断多条规则。
- **跳过工作**：NONE 档不判断，RECALL 档不调 LLM；`Effort.MIN` 不做 review。
- **hedging**：请求超时未返回，就再发一个相同请求，谁先返回用谁。默认排序 5 秒、主回复 10 秒、review 10 秒（HIGH、MAX 时 15 秒）。
- **连发消息延迟**：同一轮里第 2 条客户消息等 1 秒，第 3 条起等 3 秒，避免反复启动并取消昂贵的处理。

### 4.4 修剪

已发现规则超过 30 条时，回复后在后台清理到 20 条，至少间隔 10 条客户消息；每条规则一个小模型请求；HIGH 权重和本轮匹配的规则不清理。清理造成的缓存失效被安排在两轮之间。

### 4.5 labels

规则上的 `labels` 在规则确认命中后，以集合并集的方式写入会话标签，用于分析。只认 rank 或 code 确认过的匹配，所以带 labels 的规则会被强制走 RANK。

---

## 5. 代码里的限制与前提

下面每一条都能在 `compass-wip` 的代码或注释里找到依据。

- **未发布**：新概念只在 `compass-wip` 分支的一次 WIP 提交里，PyPI 3.3.2 的 `sdk.py` 没有 `create_policy`。
- **缓存前提**：排序器和响应循环的预热依赖提示词前缀在各轮之间字节级一致，每轮变化的内容放在 `# CURRENT TURN` 这个缓存断点之后（`rule_ranker.py`）。
- **预热有门槛**：扇出规模 ≥ 5 才预热（`optimization_policy.py`），因为预热本身是一次请求，缓存写入比普通输入贵。
- **Effort.MIN 不做复核**：此档位下 tool review 一律跳过，`consequential` 工具只剩提示词层面的确认要求（`loop/base_loop.py`）。
- **批量判决依赖配置**：只有 effort 为 MIN、LOW、MEDIUM 且路由里配置了 decision 模型时才走 Jev 批量判决，否则回到逐条打分（`rule_ranker.py`）。
- **默认模型是代码里的字符串**：OpenRouter 路径下排序用 `openai/gpt-oss-20b:nitro`，判决用 `~typesafe/jev-latest`（`adapters/nlp/openrouter_service.py`）。

---

## 参考资料

- 代码：[emcie-co/parlant](https://github.com/emcie-co/parlant)，`compass-wip` 分支，提交 c30226c
- 发布包：[parlant 3.3.2（PyPI）](https://pypi.org/project/parlant/)
- 新版文档：[Which construct for which job](https://next.parlant.io/docs/core-concepts/which-construct)、[Agent architecture](https://next.parlant.io/docs/core-concepts/agent-architecture)、[Observations and policies](https://next.parlant.io/docs/core-concepts/observations-and-policies)、[Designing policies](https://next.parlant.io/docs/customizing-behavior/policies)、[Consequential tools](https://next.parlant.io/docs/playbooks/consequential-tools)、[Comparisons overview](https://next.parlant.io/docs/comparisons/overview)、[Agents](https://next.parlant.io/docs/concepts/agents)、[Perceived performance](https://next.parlant.io/docs/advanced/perceived-performance)
- 旧版文档：[Optimization](https://www.parlant.io/docs/advanced/optimization)、[Agents](https://www.parlant.io/docs/concepts/entities/agents/)

*附注：文中所有代码结论来自对 `compass-wip` 分支的静态阅读，没有运行；默认模型名是代码里的字符串，未独立验证；`compass-wip` 是未发布的 WIP 分支，后续可能大幅变化。*
