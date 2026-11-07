---
title:      Agent 的 SOP 合规评测综述：基准、规则验证、评审模型与工程闭环
subtitle:   读 28 篇公开论文和 Langfuse 文档：合规该怎么定义、怎么判、怎么量可靠性，以及评测本身有多不可靠
title_en:   "Evaluating SOP and Policy Compliance of LLM Agents: A Survey of Benchmarks, Rule Verification, Judges and an Engineering Loop"
subtitle_en: "Notes from 28 public papers and the Langfuse docs: how to define compliance, how to check it, how to measure reliability, and how unreliable the evaluation itself can be"
en_url:     /en/sop-compliance-evaluation/
date:       2026-11-07 09:00:00 +0800
author:     Zhang Xian
tags:
    - Agent
    - 评测
    - SOP 合规
    - 综述
---

## 摘要

SOP（标准作业流程）合规评测要回答的问题是：Agent 有没有按规定的前提和顺序做该做的事，没做不该做的事。"任务完成了没有"回答不了它：Agent 可以走捷径得到正确结果，也可以流程全对却在最后一步越权。

本文梳理 2021 年到 2026 年的公开工作，覆盖四类材料：**基准**（ABCD、FlowBench、GuideBench、τ-bench、τ²-bench、SOPBench、SOP-Bench、JourneyBench、SAGE、ST-WebAgentBench），**规则验证与运行时执行**（AgentSpec、ShieldAgent、Agent-C、ContrAgent、AgentLTL、MonPoly 研究、COVENANT、STAGE 与一篇综述），**评审模型**（LLM-as-a-Judge 综述、LiveMCP-101、TED、Agent-as-a-Judge、False Success 研究），以及**评测基础设施本身的可靠性**（用户模拟器、策略歧义），最后是 Langfuse 文档里的工程积木。

我的阅读可以概括成三句话。第一，合规不是一个指标，而是五个层次：最终状态、动作许可、顺序与完整、参数与话术、稳健与可靠，各层需要不同的判定方式。第二，只看最终结果会系统性地漏掉流程违规：在 τ²-bench 的重放分析里，一个模型的结果通过率有 25% 到 60%，流程干净的比例却是 0。第三，评测自己也不可靠：用户模拟器有 16% 到 47% 的对话带错误，换一个模拟器模型就能让总成绩变 9 个点，约 14% 的 τ²-bench 航空任务因策略歧义而对"错误"的认定存疑。文末给出一份分层方案，那是我的设计，不是任何一篇论文的结论。

## 核心结论

- **合规至少有五层**：最终状态、动作许可、顺序与完整、参数与话术、稳健与可靠（图 1）。每个基准通常只覆盖其中一两层，没有一个同时覆盖五层。
- **结果通过不等于合规**：τ-bench 的作者自己写明奖励为 1 只是必要而非充分条件。ContrAgent 的重放分析里，Claude 3.7 在 τ²-bench 零售上结果 pass4 为 59.6%，流程干净的 joint4 为 0.0%（图 2）。
- **顺序是最薄弱的环节**：AgentLTL 的合成基准里序列顺序层通过率最低只有 0.111；FlowBench 的失败原因里"漏步骤"占 56.9%；SOPBench 里较弱模型主要败在跳过验证。最强的几个模型则主要败在执行了未授权动作。
- **施压很有效**：SOPBench 上追加一句"请尽快用最合适的工具完成我的请求"，在 Healthcare 与 University 上通过率相对下降 60% 到 86%（图 3）。
- **形式化规则能给出强保证，但保证是相对于规则的**：Agent-C 的 100% 合规是对作者手写规范而言的；把自然语言 SOP 翻译成规则这一步，没有论文给出干净的错误率。
- **评审模型适合判话术，不适合判动作**：在 False Success 研究里，只看对话文本的评审识别"假成功"接近随机（AUROC 不超过 0.65）；AgentLTL 里评审与符号谓词的一致性 Cohen's κ 只有 0.09。
- **评测基础设施需要先被评测**：τ²-bench 的用户模拟器在航空、零售、电信域分别有 47%、40%、16% 的对话带错误；Policy Loopholes 的审计发现 7 个航空任务受策略歧义影响。
- **证据质量整体偏低**：一篇 2026 年的综述用 GRADE 给 9 条主要结论评级，7 条是 Very Low，没有一条是 High。
- 所有数字来自各论文，我没有复现；评测的模型多为上一代，绝对值只能参考，评测设计的启发更耐用。

## 范围

本文纳入的工作需要满足两点：评的是 Agent 在工具调用或多轮对话里遵循流程、策略或指南的程度；有可核对的方法与数字。覆盖的主要是客服、订单、金融、网页操作等有明确流程的场景。不覆盖代码补全类的指令遵循、纯安全对齐类的有害内容评测，也不覆盖多智能体协作。语音场景有一小节单独讨论。

文献取自 arXiv、ACL Anthology 与 Langfuse 官方文档，截止到 2026 年 11 月初。后面的每个数字都标注出处；凡是在论文正文与表格之间互相矛盾的地方，我在第 7 节集中列出。

## 先看懂几个概念

- **SOP、策略、指南、工作流**：这几个词在文献里经常混用。SOP 通常指一份有步骤和前提的流程（"先验证身份，再查订单，才能退款"）；策略（policy）偏向规则文档（"订单超过 30 天不得退款"）；指南（guideline）是更宽泛的领域规则；工作流（workflow）强调步骤图。本文统称"流程规则"，必要时指明具体来源怎么叫。
- **Oracle（标准答案程序）**：把流程规则写成可执行代码，同样的输入跑一遍，得到"应该发生什么"。SOPBench 就是这么做的。
- **轨迹（trajectory / trace）**：Agent 一次对话里依次发出的工具调用，加上最终的数据库状态。
- **结果型与过程型判定**：结果型只比较最终状态；过程型检查中间的调用顺序、前提和参数。
- **pass@k 与 pass^k**：每个任务跑 n 次，成功 c 次。τ-bench 把 pass^k 定义为 k 次独立试验全部成功的概率，按任务估计为 C(c, k) / C(n, k) 再取平均；pass@k 则是 1 − C(n−c, k) / C(n, k) 的平均，即至少成功一次。前者衡量可靠性，后者衡量上限。
- **用户模拟器**：用另一个 LLM 扮演用户，驱动多轮对话。它本身会出错，也会因为太合作而放过 Agent 的弱点。
- **LLM 评审**：用另一个模型按评分细则给对话或轨迹打分。适合自然语言内容，需要用人工标注校准。

![图 1：合规的五个层次](/img/posts/sop-compliance-eval/01-layers.svg)

*图 1：示意图。五个层次与代表工作的对应是我的归类，不是任何论文的原话。*

## 1. 基准全景

**结论**：十个基准在"合规是什么、怎么判"上做了不同的选择。

| 基准 | 年份 | 规模与领域 | 合规的定义 | 判定方式 | 代表性结果 |
|---|---|---|---|---|---|
| ABCD | 2021 | 10,042 段对话，55 个意图，一家虚构零售商 | 按内部指南预测下一步动作（按钮、槽位、值） | 确定性，级联评分 | 最好的模型级联得分 31.9%，人类 82.7% |
| FlowBench | 2024 | 6 个领域、51 个场景、536 段会话 | 在文本、代码、流程图三种形式的工作流指导下规划 | 工具调用确定性匹配 + GPT-4 评审会话成功率 | 最好的会话成功率约 43% |
| GuideBench | 2025 | 1,272 条，7 类，537 条规则，以中文为主 | 在给定的领域指南下答对 | 与标注答案比对 | 去掉指南后 GPT-4o 仍有 80.90%，有指南 86.48% |
| τ-bench | 2024 | 零售 115 任务、航空 50 任务 | 最终数据库状态等于标注目标，且所需信息出现在回复里 | 确定性（状态对比 + 子串） | GPT-4o 零售 61.2%、航空 35.2%，零售 pass^8 低于 25% |
| τ²-bench | 2025 | 在 τ-bench 之上加电信 114 任务，用户也能操作设备 | 最终环境状态满足断言 | 确定性断言 | GPT-4.1：零售 74%、航空 56%、电信 34% |
| SOPBench | 2025 | 7 个服务域、903 个用例、167 个工具、165 个约束 | 动作许可、结果匹配、流程完整三项同时满足 | 可执行的规则校验器 | 最好 o4-mini-high 76.08% |
| SOP-Bench | 2025 | 12 个业务领域的工业 SOP，2,411 个任务 | 最终结构化输出等于标准答案 | 确定性比对 | Claude 4 Opus 的 ReAct 平均 TSR 0.724 |
| JourneyBench | 2026 | 3 个领域、703 段对话，SOP 建成有向无环图 | 工具调用序列与期望完全一致 | 确定性（序列 + 参数） | GPT-4o 的 SPA 0.564，DPA 0.717 |
| SAGE | 2026 | 6 个场景、27 个模型，SOP 建成图 | 分类、路径、动作三项加权 | 图引擎确定性 + LLM 分类标签 | 最高 72.62，动作得分远低于分类 |
| ST-WebAgentBench | 2024 起，2026 更新 | 375 个网页任务、3,057 个策略实例 | 完成任务且零策略违规（CuP） | 规则函数 | 完成率均值 24.3%，CuP 均值 15.0% |

*注：表中数字出自各论文，口径不同，不能横向比较。ST-WebAgentBench 的早期版本写的是 234 个任务，这里取 2026 年版本。*

下面按判定方式分四条路线讲。

### 1.1 结果状态型：τ-bench、τ²-bench、SOP-Bench

- **τ-bench**（Yao 等，Sierra）：奖励 r 等于 r_action 乘 r_output。r_action 要求最终数据库状态与唯一的标注目标一致；r_output 要求必要的信息字符串出现在 Agent 的回复里（子串检查）。评审没有 LLM，用户由 GPT-4 扮演。作者自己指出 r = 1 对合规是必要而非充分条件，例如 Agent 可能没经用户明确确认就操作，只要最后状态对，仍得 1。
- **策略是否真的在起作用**：τ-bench 做了去掉策略的消融。GPT-4o 零售从 61.2% 降到 56.8%，航空从 33.2% 降到 10.8%。零售的策略大多是常识，航空的策略才是真约束。这个对照可以直接借用，用来检验一份 SOP 是否真的"承重"。
- **τ²-bench**：在零售、航空之上加了电信域。电信域让用户也能操作设备（双控），任务由原子子任务程序化组合，用断言函数检查最终环境状态。它的消融把"策略推理"与"和用户沟通"拆开：默认模式、无用户模式、给出标准计划三档。无用户模式比默认模式高 18 个点（GPT-4.1）和 25 个点（o4-mini）。
- **SOP-Bench**（Nandi 等，Amazon）：用 LLM 辅助加专家校验生成 SOP、工具和数据，最终比较结构化输出。它把"跑完"和"跑对"拆开：ECR（产出了完成结果的比例）、C-TSR（完成的里面正确的比例）、TSR（总体正确率）。例如 Aircraft Inspection 在函数调用方式下 ECR 为 1.00，C-TSR 只有 0.45，即几乎每次都做完，但一半是错的。

这条路线的共同点是确定性、好复现。共同的盲区是：它看不见中间步骤，所以"侥幸答对"和"跳步答对"都算成功。SOP-Bench 自己也承认没有评估规划与工具调用的准确性。

### 1.2 规则与 Oracle 型：SOPBench、ST-WebAgentBench、ABCD、SAGE

- **SOPBench**（Li 等，arXiv 2503.08669）：每个服务函数有若干前提（约束），每个前提有一个返回真假的校验器。约束的组合方式有 Single、And、Or、Chain（必须按顺序）四种，可以嵌套。评估有三个维度：动作许可（被调用的服务函数按约束本不该被允许就算违规）、结果匹配（最终数据库状态等于 Oracle 的状态）、流程完整（每个服务调用前必需的辅助调用都做过）。三项都满足才通过。用例由约束的满足与否排列生成：And 关系下只有一种全满足的情况和很多失败的情况，Or 关系下只有一种失败情况和很多成功情况。
- **ST-WebAgentBench**（Levy 等，IBM Research）：策略分三级，组织策略优先于用户偏好优先于任务指令，覆盖六个维度（用户同意、边界、严格执行、层级遵循、稳健与安全、错误处理）。核心指标 CuP（Completion under Policy）等于完成率乘以"零违规"，所以是严格的合取。它还给出 Risk Ratio 分维度统计违规，以及 all-pass@k。论文报告完成率均值 24.3%，CuP 均值 15.0%，相对下降约 38%；有效策略越多，CuP 越低（只有 1 条策略时 18.2%，5 条及以上时 7.1%，相关系数 −0.71），完成率却基本不变。
- **ABCD**（Chen 等，ASAPP 与 Columbia）：55 个客户意图各对应一条固定的动作序列。它提出的"级联评分"很有意思：每个轮次的得分是该轮之后在第一次出错前连续预测对的比例，再对轮次取平均，所以早期出错比晚期出错损失更大。这是 2021 年的 BERT 时代数据集，但这个评分思路可以用在任何长流程上。
- **SAGE**（Shi 等，KDD '26）：把 SOP 建成含决策节点和动作节点的有向图，一个确定性的规则引擎在分类标签确定后给出参考路径和动作。逻辑合规分等于 0.4 乘分类准确率、0.4 乘路径正确率、0.2 乘动作正确率，总分再与对话质量按 0.8 与 0.2 加权。平均下来分类得分 72.9、路径 67.3、动作 41.7：分类对了但动作做不对，是普遍现象。

### 1.3 轨迹序列型：JourneyBench、AgentLTL

- **JourneyBench**（Balaji 等，Observe.AI 团队）：每个 SOP 是一个有向无环图，节点是任务，边是条件转移；用广度优先搜索枚举全部"用户旅程"。每条正确旅程再衍生两种更难的变体：缺参数、工具调用失败。得分 UJCS：只要工具调用序列有任何缺失、多余或顺序错就是 0，序列一致才按参数值的正确比例计分。这是"先过序列、再看参数"的两段式。
- **生成的 SOP 要过关**：论文让 5 位联络中心专家对 LLM 生成的 10 个 SOP 图按三项标准打分，只有全部通过的才保留，结果只有 4 个全票通过，最终用了 3 个领域。这个"全票门槛的通过率"本身就值得报告。
- **AgentLTL**（Elkoussy 与 Perez）：用有限迹上的一阶 LTL 的一个片段表达流程规则，原子谓词包括 Called、CalledWith、InOrder、CalledN 等。基准里的约束分六层：分支、字面参数、序列顺序、成对顺序、调用次数、精确参数。顺序层是弱点：序列顺序层通过率低至 0.111，成对顺序层低至 0.144，而分支、参数、次数层对强模型大多在 0.85 到 1.0。

### 1.4 工作流与指南的呈现形式：FlowBench、GuideBench

- **FlowBench**（Xiao 等，EMNLP 2024 Findings）把同一份工作流用文本、代码（伪代码）、流程图三种形式交给模型。结果是：流程图在回合级的工具 F1 上最好，GPT-4o 的单场景回合级工具 F1 从无工作流的 66.3 提到 75.5；但会话成功率上文本或代码常常更好（GPT-4o 代码 43.2 对流程图 42.7）。三者合起来的集成在 GPT-4-Turbo 上最高：单场景 45.3，跨场景 43.4。失败原因里漏步骤占 56.9%（流程图、单场景）。
- **GuideBench**（Diao 等）：1,272 条，7 类，规则分条件、嵌套、与常识相反、可更新几种。它的无指南对照很有价值：GPT-4o 没有指南也有 80.90%，有指南 86.48%，指南的增益集中在价格匹配和摘要类。数学类（规则组合最重的一类）多数模型只有 0 到 13，DeepSeek-R1 达到 65.38。

## 2. 结果综合：各路线看到了什么

把上面的数字放在一起，有几条跨基准的规律。

### 2.1 结果通过掩盖流程违规

![图 2：结果通过与流程干净](/img/posts/sop-compliance-eval/02-outcome-vs-procedure.svg)

*图 2：依据 ContrAgent 论文表 9 重绘（τ²-bench，k = 4）。"流程干净"指四次试验里没有触发任何 ContrAgent 的规则，因此是相对于它的合约库而言的。*

- Claude 3.7 在零售的 pass4 为 59.6%，joint4 为 0.0%；在航空 36.0% 对 0.0%；在电信 25.4% 对 0.0%。论文说它在每个域的"流程干净"都是 0%。
- GPT-4.1 在零售 52.6% 对 43.9%，在航空 40.0% 对 24.0%：同样是结果通过，流程违规的损失差别很大。
- ContrAgent 引用了一项研究（AgentPex），称 Claude 的奖励为 1 的轨迹里有 83% 仍违反流程规则；我没有打开那篇原文，这里只作转述。

两点保留：流程干净是用 ContrAgent 自己的合约库衡量的，合约库没覆盖的违规看不到；而且合约是从与基准同源的策略文档生成的，有一定循环成分（第 4 节再讲）。

### 2.2 顺序与步骤遗漏是最常见的失败

- AgentLTL：顺序层最差，强模型在分支、参数、次数层上大多达到 0.85 到 1.0。
- FlowBench：漏步骤占失败原因的 56.9%（流程图、单场景），无工作流时跨场景的失败主要是工具使用（54.3%）。
- τ-bench（GPT-4o，零售，36 个 Agent 失败案例）：错误信息 33.3%，错误参数 22.2%，错误决策（对规则的误解）25.0%，对复合请求只解决了一部分 19.4%。
- SOPBench：较弱的模型主要败在流程（跳过验证），最强的几个（o4-mini-high、DeepSeek-R1、GPT-4.1）主要败在执行了未授权的动作。

所以"漏步骤"与"越权"是两种失败，需要分别统计，不能只报总通过率。

### 2.3 施压、约束数量和工具数量都会拉低合规

![图 3：一句催促对 SOPBench 通过率的影响](/img/posts/sop-compliance-eval/03-pressure.svg)

*图 3：论文图 6 的标注值（Healthcare + University）。模型与数值的对应由横轴顺序推断；"无催促"一列我用主表的两个域按用例数加权复算，五个模型完全吻合，另两个相差一个用例。*

- 施压：追加一句催促语后，相对降幅在 60.1%（o4-mini-high，93.68 到 37.36）到 86.2%（Claude-3.7-Sonnet，70.69 到 9.77）之间。论文正文写的催促语与附录写的措辞不同，且图里同时给了绝对值与相对降幅，容易读混。
- 约束数量：SOPBench 里推理模型对约束数量比较稳定，Gemini-2.0-Flash 在超过 5 个约束时降到约 20%。ContrAgent 的重放里，基础模型的安全率随约束数从 79% 一路降到 0%（约束数分别为 1–2、3–4、5–6、7–9、10 以上）。
- 工具数量：SOPBench 给全部工具时，除 GPT-4o 外都显著下降；SOP-Bench 的 Video Annotation 上，工具从 6 个增加到 26 个，Claude 4.5 Sonnet 的 ReAct 成功率从 37% 降到 20.8%。
- 策略负载：ST-WebAgentBench 里有效策略越多，CuP 越低。
- 对抗型用户：StressSim 在不改变用户意图的前提下让模拟用户施压，pass^4 最多下降 69%；这篇我读到的是论文自己的汇报。

### 2.4 加结构有帮助，但要看比较的是什么

- JourneyBench：DPA（逐节点切换提示和工具）对 SPA（整个 SOP 放一个提示）：GPT-4o 平均 UJCS 0.717 对 0.564，Claude 3.5 Haiku 0.578 对 0.253，Llama 3.3 0.330 对 0.252，GPT-4o-mini 的 DPA 为 0.649。在缺参数和工具失败的场景里，SPA 掉得更多。要注意 DPA 是按构造拿到 SOP 图的，SPA 没有拿到同样的结构化信息，所以比较的是脚手架，不是模型本身的遵循能力。
- STAGE（Luo 等）：把策略编译成图，每一步只给局部策略上下文，并由代码决定路由。在 18 个"模型 × 基准"配对里全部提升，Pass^3 的增益 7.5 到 66.0 个点。例如电信域 GPT-5.6 Luna 从 Pass^1 45.8% 到 95.8%，Pass^3 从 32.5% 到 92.5%。代价是成本上升（Claude Haiku 4.5 为基线的 5.3 到 14.0 倍）。
- COVENANT（Wang 等）：把自然语言工作流编译成图再逐步校验，在 120 个案例上成功率由 43.20% 升到 73.10%，工作流失配率由 51.03% 降到 22.60%。但航空这个最像"策略文档"的场景反而变差了 4.0 个点，成本是调用次数 1.26 倍、时间 2.48 倍。

## 3. 可靠性：pass^k 与它的近亲

**结论**：几乎所有流程合规的现实问题都发生在"多次里偶尔出错"，而不是"能不能做对一次"。

| 工作 | 可靠性怎么量 | 要点 |
|---|---|---|
| τ-bench | pass^k = C(c, k) / C(n, k) 的平均 | 最先给出，GPT-4o 零售 pass^8 低于 25% |
| STAGE | Pass^1 与 Pass^3 并报 | Pass^3 是同一案例三次全部成功的案例比例 |
| ContrAgent | joint4 = 结果通过且流程干净，且四次全部 | 把结果与流程合到一个可靠性指标里 |
| ST-WebAgentBench | CuP、all-pass@k | CuP 是完成且零违规的合取 |
| ABCD | 级联评分 | 对长流程按"首次出错前的连续正确"计分 |
| SOPBench、SOP-Bench | 单次运行 | 没有方差，也没有 pass^k |
| SAGE | 在第 1、5、10、15 轮和最终轮打分 | 看合规随对话变长的趋势，多数场景在第 5 轮左右达到峰值后下降 |

一个手算例子：某任务 4 次试验里成功 2 次，pass^4 的贡献是 C(2,4) / C(4,4) = 0，pass@4 的贡献是 1 − C(2,4) / C(4,4) = 1。下面是这个估计器的最小实现（我写的，已运行验证）：

```python
from math import comb

def pass_hat_k(successes, n, k):
    """tau-bench style: mean over tasks of C(c, k) / C(n, k)."""
    return sum(comb(c, k) / comb(n, k) for c in successes) / len(successes)

def pass_at_k(successes, n, k):
    return sum(1 - comb(n - c, k) / comb(n, k) for c in successes) / len(successes)

# 4 tasks, 4 trials each, successes = [4, 4, 0, 2]
# pass^1 = 0.625, pass^4 = 0.5, pass@4 = 0.75
```

几点提醒：

- 单次运行的基准（SOPBench 主表在温度 0 下，SOP-Bench 的主表）没有方差，差 1 到 3 个点的排名不应当当真。FlowBench 的会话成功率差异也多在 1 到 3 个点，用的是 3 次运行加 LLM 评审，没有置信区间。
- 小样本放大不确定性：STAGE 的 Paystub 域只有 51 个案例，一个案例约等于 2 个点。
- SAGE 的结果说明合规会随对话变长而下降，所以评测要覆盖不同长度。

### 3.1 语音场景的补充

语音 Agent 还要多看一层：说出来的实体是否正确。EVA-Bench（Bogavelli 等，arXiv 2605.13841）在 213 个场景、12 个系统上把准确（EVA-A）与体验（EVA-X）分开，并报告 pass@1、pass@k、pass^k；EVA-A 上 pass@k 减 pass^k 的中位差距是 0.44（我用论文表 4 复算一致）。它还给出级联系统里关键实体转写准确率与任务完成率的相关系数 0.93，说明 WER 看不出一个字符之差的确认码。关于打断环节的评测，见 [如何评测打断：SID-Bench、EchoChain、IHBench 与 Instruct-FD]({% post_url 2026-11-05-barge-in-benchmarks-2026 %})。τ-Voice（arXiv 2603.13686）把 τ²-bench 的文本任务搬到实时语音上，我只确认了它存在，没有读，不引用数字。

## 4. 把 SOP 变成可检查的规则：形式化路线

**结论**：规则验证能给出强保证，但保证只相对于规则，而且从自然语言到规则这一步没有被干净地度量。

### 4.1 几种规范语言

| 工作 | 规范形式 | 判定时机 | 一个例子 |
|---|---|---|---|
| AgentSpec（ICSE 2026） | 触发 + 检查 + 执行三元组，单步规则 | 在线 | 触发"转账"，检查"不是转给家人"，执行"请用户确认" |
| ShieldAgent（ICML 2025） | LLM 从政策文档提取规则，每条含 LTLf 公式，组织成概率规则电路 | 在线 | 若未授权则不得删除数据 |
| Agent-C | 面向前后顺序的 DSL（Before、After、Seq、Exists、Forall），可引用先前工具输出与后端状态，转成一阶逻辑交给 Z3 | 在线，生成时拦截 | 取订单详情之前必须有一次邮箱查找用户，且订单属于该用户 |
| ContrAgent | 有限迹 LTL 加算术谓词的假设-保证合约，编译为自动机 | 在线与离线 | 先批准再退款，且退款金额不超过批准时记录的金额 |
| AgentLTL | 有限迹上的一阶 LTL 片段，加权约束的合规得分 | 离线打分与在线拦截 | 每次搜索之后最终要有一次引用 |
| MonPoly 研究 | 度量一阶时序逻辑（MFOTL） | 离线重放 | 破坏性操作之前必须有批准 |

### 4.2 代价与误报

- **开销**：ContrAgent 报告每次调用的延迟：SOPBench 约 0.097 ms（p50），τ²-bench 约 0.828 ms，在 SOPBench 上平均每个任务增加 0.135 秒，对比提示词方案 +0.905 秒、LLM 护栏 +1.34 秒（基础 1.96 秒）。AgentSpec 报告规则解析约 1.42 ms、谓词评估数毫秒。Agent-C 因要调用 SMT 求解，Qwen3-32B 的整体运行时间比不加约束高 44%（480.23 秒对 333.05 秒）。ShieldAgent 比一次直接评审慢得多（31.1 秒、9.5 次查询，对 6.3 秒、1 次）。
- **漏报与误报**：ContrAgent 在 23 个骨干模型的记录轨迹上，对不安全轨迹的召回在 54.7% 到 100% 之间，对安全轨迹的误报率在 19 个模型上低于 1%、最高 4.5%。MonPoly 研究的结果不太乐观：GPT-4o 的 AgentDojo 里对成功攻击的检测率 70.1%，对良性运行的误触发率 29.3%，似然比 LR+ 只有 2.40。作者的结论是通用规则更适合当分诊过滤器，而不是闸门。
- **日志里缺字段**：同一项研究审计了现成轨迹里是否有规则要用的字段：显式批准字段在 AgentDojo 里是 0%，STAC 里 1.2%，R-Judge 里 0.4%，时间戳和结构化状态都没有。结论是：想让 SOP 可被监控，要先在 Agent 运行框架里把批准编号、时间、前后状态、来源可信域这类字段记下来。
- **来源要绑定**：同一研究里，一个"收款方已知"的朴素谓词，只要在上下文里放一行伪造的结构化文本就会被翻转，在银行和办公两个套件里分别有 94.0% 和 99.0% 的检测被翻转；把谓词改成"值必须来自 Agent 自己发起的查询的返回"之后翻转为 0.0%，检测率与良性触发率都不变（仅对测试过的攻击族成立）。这对"必须先核实某事"的 SOP 步骤是一条具体的设计规则。

### 4.3 最大的缺口：翻译

- 在 AgentSpec 里，LLM（o1）生成的规则在代码任务上执行率 87.26%，具身任务 95.56%，自动驾驶 62.50%；错误来自对示例过拟合、规则过僵（禁止所有倒水，包括给植物浇水）、漏条件、阈值错误。
- Agent-C 让 Claude Sonnet 4.5 根据策略文档、工具列表和一个示例规范生成规范，在零售良性任务上与手写规范给出相同结果（100% 合规，42.11% 效用）；但论文说它有多余的重复检查，也没有做语义等价检查。
- ContrAgent 用"提取模板再落地"的流程：LLM 在固定谓词词汇上填模板，确定性步骤落地占位符，每条公式回译成自然语言交给人审。没有报告翻译错误率，也没有说明各基准的合约是手写还是生成的。
- ShieldAgent 的规则提取没有报告准确率。
- 综述（Dantas 等）引用的其他工作：自然语言到 LTL 的语法正确率超过 90%，语义正确率却很低；NL2Spec 在没有交互消歧时语义准确率 44.4%，有消歧时 86.11%；GPT-4o 零样本生成 PDDL 语法有效 96.1%，语义正确只有 24.8%。

我的归纳：七项形式化工作里，没有一项给出了"自然语言 SOP 到时序规则"这一步的干净错误率。翻译质量需要自己测：随机抽一批规则，回译成自然语言让人对照原文，并用正反例轨迹测试规则的拒绝与放行。

### 4.4 循环论证的风险

- Agent-C 的 100% 合规是对作者手写规范而言的，规范与合规检查器出自同一批作者；摘要里的效用提升数字（Sonnet 71.8 到 75.2，GPT-5 66.1 到 70.6）我在论文的表 5、6 里没有找到，表里是零售 76.52 到 80.46 与 71.01 到 73.62，航空 42.00 到 43.54 与 38.00 到 42.84。
- ContrAgent 在 SOPBench 上的合约来自同一份公开 SOP，而 SOPBench 的标准答案也由这份 SOP 定义，近 100% 的安全率部分是循环的。
- ContrAgent 的 AgentDojo 合约有一部分从同一基准的轨迹里学来，推广性未知。
- 综述对 Agent-C 与 AgentSpec 的 100% 合规用 GRADE 评为 Very Low。

### 4.5 更强的执行不等于更好的结果

综述引用的"verifier tax"研究（Sah 等，2026，τ-bench，两个小型开源模型）称：验证器能拦截最多 94% 的不合规单步动作，但"既安全又成功"的完成率在多数设定下仍低于 5%，原因是 Agent 编造用户标识符。这是单项研究，没有独立复现，我也没有读原文，只作转述。它提示的方向是：拦下违规动作不等于让任务完成，拦截之后需要有重规划。COVENANT 里航空任务变差、ContrAgent 的重放无法衡量被拦后的重规划，也指向同一个问题。

## 5. 评审模型：用在哪、怎么校准

**结论**：评审适合判话术，动作与顺序交给代码；评审必须在人工标注的小样本上校准，并且给它证据，不只给对话文本。

### 5.1 评审模型的已知问题

- 综述（Gu 等）的位置一致性数据：GPT-4-turbo 80.31%，GPT-3.5 68.78%，LLaMA3-8B 38.85%。与人工的一致性在 LLMEval2 上只有约 51% 到 62%；对人工标注为"平局"的样本，GPT-4-turbo 的一致率只有 29.73%。偏差子集的样本数只有 24 到 34 条。
- 唯一有帮助的多次运行缓解手段是多数投票；均值与最优 N 选一都没有帮助（在较弱的评审上）。
- 综述建议：交换位置取平均、结构化输出、拆解评分维度、不要让评审同时生成解释与分数、避免同一模型自评、先用小的元评测找出评审的偏差轮廓。

### 5.2 轨迹与过程的评审：校准证据

| 工作 | 评什么 | 一致性 | 要点与保留 |
|---|---|---|---|
| LiveMCP-101 | 结果分与轨迹分 | 加权 κ：结果 > 0.85，轨迹 > 0.78 | 只在 30 个任务上，只给了阈值；GPT-4.1 既是参考执行者又是评审之一；三次运行的成功率标准差最高 1.93 个点，三个评审模型之间不超过 1.14 个点 |
| TED | 子目标级评审 | κ 在 0.60 到 0.94 | 最低的是 gpt-4.1 的 Agent：航空 0.71，ToolSandbox 0.60；评审、用户代理与其中一个 Agent 都是 gpt-4.1，人工标注者人数未说明；"模糊"算作一致，会抬高 κ |
| Agent-as-a-Judge | 对开发 Agent 的需求逐条判定 | 与人工共识的一致率 84% 到 92% | 对比 LLM 评审基线 60% 到 84%；三位人工多数表决达到 94% 到 95%；评审是能查看工作区的 Agent，等价于能看工具调用日志 |
| JourneyBench | 对话质量（生产 QA 评分细则） | 合成对话得分 84.37% | 这是厂商自己的验证，没有给出评审与人工的一致性 |

### 5.3 对话文本的评审会被"自信的结尾"骗

- False Success 研究（Advani，τ²-bench，9,876 条轨迹、1,730 条失败）：35.6% 的失败是"假成功"（Agent 声称完成，状态不一致）；航空与零售域里假成功占失败的 45% 到 47%，电信域只有 3%。五个评审模型（GPT-4o、Sonnet 4.5、Llama 3.3-70B、DeepSeek-R1、o3-mini）区分假成功与真成功的 AUROC 最高 0.640，接近随机；评审把听上去自信的轨迹评为更完整。一个小的训练型检测器能做到 AUROC 0.825。
- AgentLTL：评审（DeepSeek-V4-Pro）与符号化的"落地"谓词的一致率 77.6%，Cohen's κ 只有 0.09（论文把它归因于类别不平衡，评审判定落地的比例 6.4%，谓词是 20.9%）。

两项合起来的意思是：能用环境状态或工具日志直接判的，不要让评审只看文本。

### 5.4 可借鉴的校准做法

1. 把评审的条目拆成少数几条，每条一个布尔或分类结果和一份细则。
2. 给评审证据（工具调用日志、数据库差异），不只是对话。
3. 抽一小批对话让人按同一细则独立标注，先算人工之间的一致性作为上限，再算评审与人工的一致性。
4. 报告 κ 与混淆矩阵，不只报一致率；样本的难度要分层。
5. 把评审运行三次取多数，并换一个不同家族的评审模型对照，报告核心指标的标准差。
6. 评审与被测 Agent 最好不是同一个模型家族。

## 6. 评测基础设施本身有多可靠

### 6.1 用户模拟器

- τ²-bench 对模拟器做了人工审计（每段对话两位标注者）：航空 100 段，47% 带错误，其中关键错误 13 段；零售 50 段，40%，关键错误 6 段；电信 50 段，16%，关键错误 3 段。论文把电信域做成让用户通过工具操作，是为了约束模拟器、降低错误。
- Lost in Simulation（Seshadri 等）：在 τ-bench 零售上，只换用户模拟器的 LLM，GPT-4o 作 Agent 的成功率就从 67.0 到 75.9。真人参与者下的成功率是 45.2%，模拟用户提问比例（18.8%）和礼貌度（39.2%）都高于真人（9.8% 与 19.9%）；失败归因里，模拟用户下 Agent 负责的占 48.9%，真人下只有 24.5%。
- StressSim：对抗但合法的模拟用户把 pass^4 最多降低 69%；用来检查模拟用户有效性的评审，在"目标信息覆盖"一项上与人工的 κ 只有 0.34。
- τ-bench 的作者承认用户指令的错别字与歧义会造成假失败（40 个失败里有 4 个是这个原因），并通过多次试验调试任务来消除。
- EVA-Bench 在打分前加一道"模拟校验"，不通过就重新生成。

### 6.2 策略歧义

Policy Loopholes（Cao）审计了 τ²-bench 的航空（50 任务）和零售（114 任务）：航空有 7 个任务受策略漏洞影响（14%），零售 0 个，合计 164 个任务里 7 个（4.3%，我的算术）。这 7 个任务上，GPT-5、Kimi-K2-Thinking、Mistral-Large-3 的平均成功率为 32.1%、42.9%、35.7%，非漏洞任务是 64.5%、48.3%、40.7%。去掉漏洞任务后 GPT-5 的总成绩从 60.0 变到 64.5。一项后续版本（τ³-bench）修订了 50 个航空任务里的 29 个，其中包含这 7 个。论文没有给出"Agent 错误里多大比例其实是策略歧义"，它给的是受影响任务的比例。样本很小（7 个任务，三个模型各四次），且任务是依据观察到的失败选出来的，所以只能算是一个提醒：先审计 SOP 的歧义，再把"没按流程做"算作 Agent 的错。

### 6.3 数字对不上的地方

下面是我在阅读中记录的论文自身的不一致，引用相应数字前请对原文核对。

- τ-bench：航空 GPT-4o 在表 2 是 35.2，在表 3 是 33.2；失败分析样本里的 pass^1 是 65.2%，表 2 是 61.2%。
- τ²-bench：表 2 的说明文字写电信域没有关键错误，表里却列了 3 个（6%）。
- SOP-Bench：摘要说 12 个 SOP，表里列了 13 个；正文里若干数字与表 6 不一致。
- SOPBench：正文说每个用例最多运行 5 次，附录说单次运行，温度都是 0；正文与附录的催促语措辞不同。
- ABCD：引言写最好 31.9%，消融表里有 32.7%。
- AgentSpec：摘要说代码类 Agent "超过 90%"，表 3 的合计是 422 次调用里标记了 333 次，约 79%。
- ShieldAgent：摘要里的 11.3% 和 7.4% 增益无法由表 2、3 复现；基准规模在正文里出现 2K、3K、960 条、3,110 条轨迹等不同说法。
- AgentLTL：存在若干内部不一致（试验次数、层权重占比、哪些模型在 soft-block 下变差）。
- GuideBench：正文写 DeepSeek-R1 总体 87.11，表 1 是 87.26。

## 7. 工程闭环：以 Langfuse 为例

**结论**：Langfuse 给的是积木，不替你决定检查什么；顺序类检查要自己写代码评估器。

![图 4：离线实验与线上评测的闭环](/img/posts/sop-compliance-eval/04-review-loop.svg)

*图 4：示意图，结构依据 Langfuse 文档，做了简化。*

- **评分的数据模型**：Score 有名称、值和四种数据类型（NUMERIC、CATEGORICAL、BOOLEAN、TEXT），每个评分挂在一个 trace、observation、session 或数据集运行上；来源有 API、EVAL、ANNOTATION 三种。布尔值存成 0/1。评分配置（名称、类型、范围、类别）一旦创建不可修改，只能归档。
- **数据集与实验**：数据集条目有输入、可选的期望输出和元数据，可以从 trace 或 observation 加入；实验可以通过界面、SDK 或 OpenTelemetry 运行。SDK 的实验里，条目级评估器收到输入、输出、期望输出和元数据，返回带名称、值和备注的评估结果，运行级评估器收到全部条目结果。CI 里可以用 `langfuse/experiment-action`，实验脚本在分数低于阈值时抛 `RegressionError`。实验默认每个条目只有一次，没有重复运行的概念，所以要算 pass^k，需要自己跑多次并聚合。
- **线上评测**：规则用过滤条件（观测类型、名称、元数据、用户、会话、标签、版本、是否根观测）加抽样率选出观测，触发挂在它上面的评估器。一个评估器只看自己那个观测的数据，不看子级和兄弟观测，所以评整段对话要选记录整体输入输出的根观测。
- **LLM 评审**：提示里用 `{% raw %}{{input}}{% endraw %}`、`{% raw %}{{output}}{% endraw %}`、`{% raw %}{{ground_truth}}{% endraw %}` 变量，映射到观测的输入、输出、元数据或工具调用；返回分数和理由。文档估计每次评估 0.01 到 0.10 美元。
- **代码评估器**：文档把它们明确指向"工具调用检查"。它能读到观测的 `tool_calls`（含名称、参数），运行时间上限 2 秒，只能用标准库、不能联网，返回四种类型的分数。我在文档里没有找到现成的"工具序列评估器"，所以顺序、前提这类检查要写成代码评估器，或者放在自己进程里的 SDK 实验评估器里。
- **人工标注与一致性**：标注队列让评审人给 trace、observation 或 session 打分；Score Analytics（Beta）可以比较两个分数，数值型给 Pearson、Spearman、MAE、RMSE，分类与布尔型给 Cohen's Kappa、F1、总体一致率与混淆矩阵。限制：一次只能比较两个分数，总体一致率没有做偶然一致校正。
- **弃用提醒**：文档写明 trace 级评估器已弃用，在 Langfuse Cloud 上 2026 年 11 月 16 日（v4 切换）之后停止产出结果。写这篇时我读到的就是这样，以官方页面为准。

## 8. 一份分层方案（我的设计）

下面把以上文献拼成一条做法，每一层都标出依据。这是我的设计，不是任何论文的结论。

| 层 | 做什么 | 依据 |
|---|---|---|
| 0 条目化与审计 | 把 SOP 拆成必做、禁做、顺序、分支、话术五类条目；先审计 SOP 有没有歧义、冲突与沉默的地方 | SOPBench 的约束组合；Policy Loopholes 的两阶段审计 |
| 1 结果 | 对比最终状态与标准状态，分别记录"跑完"和"跑对" | τ-bench、SOP-Bench 的 ECR 与 C-TSR |
| 2 规则与顺序 | 动作许可、必需步骤、顺序、次数用代码判，序列不一致直接记失败，再看参数 | SOPBench 三项检查；JourneyBench 的 UJCS；AgentLTL 的分层 |
| 3 话术 | 披露、确认、承诺这类代码判不了的条目，用布尔评审加细则 | SAGE、TED；校准做法见第 5.4 节 |
| 4 可靠性 | 每个任务重复运行，报告 pass^k 与置信区间，并分别报告结果与流程两类的 k 次全过 | τ-bench、STAGE、ContrAgent 的 joint4 |
| 5 施压 | 在数据集里放催促、施压、声称有权限、缺参数、工具失败的用例 | SOPBench 的催促实验；JourneyBench 的变体；StressSim |
| 6 评测设施 | 单独报告模拟器的错误率和重生成率，固定并公布模拟器模型 | τ²-bench 审计；Lost in Simulation |
| 7 线上与回流 | 线上抽样评估，失败案例人工复核后回流进数据集 | Langfuse 的线上规则与标注队列 |

**一个最小的规则检查器**（我写的示意，用一份虚构的退款流程，已运行验证三个小例子）：

```python
from dataclasses import dataclass

REQUIRED_BEFORE = {"issue_refund": ["verify_identity", "get_order"]}
FORBIDDEN_WHEN = {"issue_refund": lambda state: state.get("order_age_days", 0) > 30}

@dataclass
class Result:
    ok: bool
    reason: str = ""

def check_trace(calls, state):
    """calls: tool names in order; state: facts the tools returned."""
    seen = []
    for name in calls:
        for req in REQUIRED_BEFORE.get(name, []):
            if req not in seen:
                return Result(False, f"{name} called before {req}")
        rule = FORBIDDEN_WHEN.get(name)
        if rule and rule(state):
            return Result(False, f"{name} not permitted in this state")
        seen.append(name)
    return Result(True)
```

## 9. 值不值得：从工程角度看

- **适合的前提**：流程能被显式写成前提与顺序，工具调用是主要的动作载体，错一次代价高（验证、退款、变更、授权）。这时规则验证与 Oracle 收益最大，也最容易复现。
- **不适合直接套用**：
  - 流程里有大量条件分支或自由裁量（SOPBench 自己说明没有覆盖 IF-THEN-ELSE 类条件流程，且只评动作前的校验约束）。
  - SOP 本身写得含糊，失败可能来自 SOP 而不是 Agent。
  - 动作发生后没法回滚时，在线拦截要特别小心（AgentLTL 自己提到，拦下不可逆的动作可能留下不一致的状态）。
- **比较口径要小心**：
  - 各基准的指标不同，数字不能横向比；评测的模型多为上一代。
  - 厂商报告的结果（JourneyBench 的 DPA、ST-WebAgentBench、各护栏框架）一般在自己设计的基准上，且基线实现可能偏弱。
  - 许多工作用同一家族的模型同时扮演评审、用户和被测 Agent（τ²-bench、TED、LiveMCP-101），同家族效应没有被隔离。
  - 形式化路线的 100% 通常是相对作者手写规范而言的。
- **可以直接借走的**：
  1. 约束写成 Single / And / Or / Chain，并让每条约束有独立的校验器和失败归因。
  2. "跑完"和"跑对"分开记（ECR 与 C-TSR）。
  3. 去掉策略的消融，检验策略是否承重（τ-bench）。
  4. 序列不一致直接判失败，之后再看参数（JourneyBench）。
  5. 报告结果通过与流程干净的联合可靠性（ContrAgent 的 joint4）。
  6. 在数据集里放催促类对抗用例（SOPBench）。
  7. 先审计 SOP 歧义、再认定 Agent 违规（Policy Loopholes）。
  8. 模拟器的错误率作为一个被测量的量，并固定模拟器（τ²-bench、Lost in Simulation）。
  9. 评审只管话术，给它证据，并在人工样本上校准（TED、Agent-as-a-Judge、LiveMCP-101）。
  10. 先在日志里记下批准编号、时间、前后状态、来源可信域，再谈规则监控（MonPoly 研究）。

## 10. 开放问题

以下是我读完之后认为文献里的空白，属于我的判断：

- **翻译缺口**：自然语言 SOP 到可执行规则的错误率，没有人给出干净的数字。
- **没有五层同时覆盖的基准**：结果、动作、顺序、话术与可靠性很少放在同一个基准里；语音场景的合规评测更少。
- **分支与条件流程**：SOPBench 明说没覆盖 IF-THEN-ELSE；JourneyBench 的图能表达条件转移，但只有三个域。
- **拦截之后怎么办**：重放式评估无法度量被拦后的重规划；"verifier tax"一项单研究提示拦截会降低完成率。
- **同家族效应**：评审、用户与被测模型是同一家族时的偏差，没有被隔离。
- **证据质量**：综述自己的 GRADE 评级里 7 条主要结论是 Very Low。
- **评测设施的标准化**：模拟器错误率、策略歧义审计、试验次数，目前没有统一报告的规范。

## 总结

SOP 合规评测的核心，是承认合规有层次，并对每一层选对判定方式：状态与动作用代码，顺序用轨迹逻辑，话术才交给校准过的评审，再用多次试验和对抗用例量可靠性。读下来最重要的三条经验是：结果通过会掩盖流程违规；评测设施自己也会错，需要先被评测；形式化保证只相对于规则，翻译这一步还没有被量过。工程上，Langfuse 一类工具提供了数据集、实验、线上规则、标注队列和一致性分析这些积木，检查什么则取决于你的 SOP 能被写到多具体。

## 待核对

- 综述类与护栏类论文（AgentSpec、ShieldAgent、AgentLTL、MonPoly 研究、ContrAgent）中个别数字来自对全文的分段阅读，我对 ContrAgent、AgentLTL、STAGE、COVENANT、FlowBench、Policy Loopholes 与 ST-WebAgentBench 的关键数字对照了论文 PDF，其余数字请在引用时回到原文。
- "verifier tax"和 AgentPex 的 83% 都是经由其他论文转述，我没有打开原文。
- SOPBench 图 6 的模型与数值对应由横轴顺序推断，第 2.3 节已说明复算情况。
- CompliBench（arXiv 2604.12312）、SOP-Maze（arXiv 2510.08942）和 τ-Voice 只在检索中见到，没有打开，不引用数字。

## 参考资料

基准：

- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (arXiv 2406.12045)](https://arxiv.org/abs/2406.12045)
- [τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment (arXiv 2506.07982)](https://arxiv.org/abs/2506.07982)
- [Action-Based Conversations Dataset (ABCD) (arXiv 2104.00783)](https://arxiv.org/abs/2104.00783)
- [SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints (arXiv 2503.08669)](https://arxiv.org/abs/2503.08669)
- [SOP-Bench: Complex Industrial SOPs for Evaluating LLM Agents (arXiv 2506.08119)](https://arxiv.org/abs/2506.08119)
- [Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence (JourneyBench, arXiv 2601.00596)](https://arxiv.org/abs/2601.00596)
- [FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents (arXiv 2406.14884)](https://arxiv.org/abs/2406.14884)
- [GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents (arXiv 2505.11368)](https://arxiv.org/abs/2505.11368)
- [SAGE: A Service Agent Graph-guided Evaluation Benchmark (arXiv 2604.09285)](https://arxiv.org/abs/2604.09285)
- [ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents (arXiv 2410.06703)](https://arxiv.org/abs/2410.06703)
- [EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents (arXiv 2605.13841)](https://arxiv.org/abs/2605.13841)

规则验证、执行与工作流编译：

- [AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents (arXiv 2503.18666)](https://arxiv.org/abs/2503.18666)
- [ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning (arXiv 2503.22738)](https://arxiv.org/abs/2503.22738)
- [Enforcing Temporal Constraints for LLM Agents (Agent-C, arXiv 2512.23738)](https://arxiv.org/abs/2512.23738)
- [Symbolic Temporal Supervision of LLM Agents Using Contracts (ContrAgent, arXiv 2609.18128)](https://arxiv.org/abs/2609.18128)
- [AgentLTL (arXiv 2607.02599)](https://arxiv.org/abs/2607.02599)
- [Formal Runtime Verification for Tool-Using LLM Agents (arXiv 2610.09793)](https://arxiv.org/abs/2610.09793)
- [Toward Safe LLM Agents: A Survey of Specification, Verification, and Enforcement (arXiv 2608.14590)](https://arxiv.org/abs/2608.14590)
- [COVENANT: Natural-Language Workflow Compilation for Aligned Agent Execution (arXiv 2607.25400)](https://arxiv.org/abs/2607.25400)
- [STAGE: Stateful Translation to Agentic Graph Execution (arXiv 2608.22538)](https://arxiv.org/abs/2608.22538)

评审模型与评测设施：

- [A Survey on LLM-as-a-Judge (arXiv 2411.15594)](https://arxiv.org/abs/2411.15594)
- [LiveMCP-101 (arXiv 2508.15760)](https://arxiv.org/abs/2508.15760)
- [Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis (arXiv 2603.15483)](https://arxiv.org/abs/2603.15483)
- [Agent-as-a-Judge: Evaluate Agents with Agents (arXiv 2410.10934)](https://arxiv.org/abs/2410.10934)
- [Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations (arXiv 2601.17087)](https://arxiv.org/abs/2601.17087)
- [From Confident Closing to Silent Failure: Characterizing False Success in LLM Agents (arXiv 2606.09863)](https://arxiv.org/abs/2606.09863)
- [Intent-Preserving Adversarial User Simulation for Multi-Turn Agent Evaluation (StressSim, arXiv 2610.35426)](https://arxiv.org/abs/2610.35426)
- [Policy Loopholes in Agent Evaluation: When Policy Ambiguity Masquerades as Agent Error (arXiv 2609.14400)](https://arxiv.org/abs/2609.14400)

工具文档：

- [Langfuse：Evaluation overview](https://langfuse.com/docs/evaluation/overview)
- [Langfuse：Core concepts](https://langfuse.com/docs/evaluation/core-concepts)
- [Langfuse：LLM-as-a-Judge](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge)

相关阅读：[如何评测打断：SID-Bench、EchoChain、IHBench 与 Instruct-FD]({% post_url 2026-11-05-barge-in-benchmarks-2026 %})。

*说明：所有论文数字为作者报告，未经我复现；"我的归纳""我的设计""我的推断"标出的内容是我自己的判断，不是论文的结论。*
