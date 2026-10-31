---
layout:     post
title:      从 SM100 到 SM90：DeepSeek-V4.1 推理栈逐层拆解
subtitle:   A Technical Walkthrough —— 张量核心、FP4、在线 Softmax、FlashMLA 与 MoE
title_en:   "From SM100 to SM90: A Layer-by-Layer Breakdown of the DeepSeek-V4.1 Inference Stack"
subtitle_en: "A technical walkthrough: tensor cores, FP4, online softmax, FlashMLA and MoE"
date:       2026-10-28
author:     张弦
tags:
    - 大模型推理
    - 注意力机制
    - GPU
    - MoE
---

## 摘要

DeepSeek 开源的推理引擎 DeepInfer 官方只支持 NVIDIA SM100（Blackwell，CUDA 13）和华为 Ascend 950。大量存量算力是 Hopper（H100/H200/H20，SM90），社区已经出现了实验性的 SM90 移植。一个自然的问题是：**把一个为 Blackwell 写的推理栈搬到 Hopper 上，究竟难在哪里？**

回答这个问题需要把推理栈从上到下拆开：模型结构如何映射到算子，算子如何映射到硬件指令，以及每一层的计算为什么必须那样组织。本文围绕七个核心技术问题展开：

| # | 问题 | 涉及组件 |
|---|---|---|
| Q1 | 为什么为 SM100 编写的 kernel 不能直接在 SM90 上运行？ | 指令集、张量核心、编译目标 |
| Q2 | 没有 FP4 张量核心，FP4 权重在 Hopper 上如何计算？代价在哪？ | DeepGEMM、低精度量化 |
| Q3 | 注意力的计算结构是什么？它为什么允许分块流式计算？ | FlashAttention、在线 softmax |
| Q4 | decode 阶段如何在压缩 KV cache 上高效计算注意力？ | FlashMLA、split-KV |
| Q5 | MoE 层的数据如何在多卡之间流动？ | DeepSelect、DeepEP、DeepGEMM |
| Q6 | 为什么每层同时需要注意力和前馈两种子层？ | SwiGLU、残差流 |
| Q7 | 5120 维的隐藏状态为什么足以承载语义？ | 叠加、规模定律 |

最后一节把这些分析落到一份可执行的 Hopper 移植验证方案上。

## 核心结论

- **编译通过不等于可用。** SM90 与 SM100 的张量核心指令（`wgmma` 对 `tcgen05`）、片上存储（无 TMEM）和低精度格式（无 FP4）都不同，FlashMLA 上游已移除 Hopper 支持，DeepGEMM 的 Hopper 版本缺少 FP4 与 UE8M0 分支，这些都需要重写，而非重新编译。
- **FP4 在 Hopper 上可以"存而不算"。** 权重以 FP4 存储、在片上反量化为 FP8 后计算，转换无精度损失；decode 阶段的带宽收益得以保留，prefill 阶段损失 FP4 的算力优势。
- **在线 softmax 是注意力高效实现的数学基础。** softmax 加权平均可分解为分子、分母两个求和，配合最大值的事后修正，使得注意力可以在 SM 内分块、在 SM 间分段、在卡间分片计算。
- **MLA 让 K 与 V 共享同一份压缩向量。** 通过把还原矩阵吸收进查询和输出投影，decode 时只需读一份压缩 KV，FlashMLA 围绕这一形状设计。
- **注意力负责搬运，前馈负责加工。** 注意力的输出是值向量的凸组合，本身不产生新特征；非线性前馈层承担特征交叉与参数化知识检索。
- **表示容量来自方向而非维度。** 高维空间中近似正交的方向数随维数指数增长，在特征稀疏的前提下，5120 维可以叠加承载远多于 5120 个特征。

## 全景：模型结构与算子的对应关系

![图 1：DeepSeek-V4.1 一次前向与底层算子的对应关系](/img/posts/h200-attention/01-model-kernel-map.svg)

*图 1：左列是模型结构，由推理引擎（vLLM、DeepInfer 等）的代码定义；右列是各步骤依赖的 GPU 算子库。从 SM100 迁移到 SM90 的工作量几乎全部集中在右列。*

DeepSeek-V4.1-Flash 的主要结构参数如下（来自 vLLM recipe 与公开模型规格，不同来源的总参数量说法不一）：

| 项目 | 取值 |
|---|---|
| 层数 / 隐藏维度 | 40 / 5120 |
| 激活参数 | prefill 每 token 约 8B，decode 每 token 约 16B |
| 注意力 | 滑动窗口 + 压缩稀疏注意力，两级 indexer，每个 query 保留 512 个条目 |
| 注意力头 | 64 个 query 头，1 个 KV 头 |
| MoE | 384 个路由专家选 6 个，外加 1 个共享专家 |
| 词表 | 129,280 |
| 投机解码 | DSpark 多 token 草稿头 |

---

## Q1：为什么为 SM100 编写的 kernel 不能直接在 SM90 上运行？

### 结论

kernel 编译出的机器码绑定特定架构。DeepInfer 的关键算子直接使用了 Blackwell 专有的张量核心指令、片上张量内存和 FP4 数据通路，这些在 Hopper 上不存在对应物，只能重写。

### 机制

Hopper（2022，SM90）与 Blackwell（2024，数据中心型号为 SM100/SM103）是 NVIDIA 两代数据中心 GPU 架构。一个高性能矩阵乘 kernel 的主循环大致是：

1. 将数据块从 HBM 搬入 SM 的 shared memory（Hopper 由 TMA 硬件完成）；
2. 由张量核心执行矩阵乘加（Hopper 为 `wgmma`，Blackwell 为 `tcgen05`，并引入 TMEM 存放累加器）；
3. 累加完成后写回 HBM。

`nvcc` 按 `sm_90a` 或 `sm_100a` 生成机器码，产物只能在对应架构上运行。迁移时可能遇到三类失败：

| 失败类型 | 表现 | 风险 |
|---|---|---|
| 构建脚本未包含 `sm_90a` | 编译无报错，运行时报"无可用 kernel 镜像" | 低，容易发现 |
| 源码使用 SM100 专有指令 | `nvcc` 编译失败或模板断言失败 | 低，容易发现 |
| 编译运行均通过，但落入未验证的回退路径 | 数值错误或性能异常 | **高**，只能靠正确性测试发现 |

![图 2：H200 与 B200 从软件到硬件的对比](/img/posts/h200-attention/02-h200-vs-b200.svg)

*图 2：上半部分决定 kernel 能否编译，下半部分决定编译通过后的性能上限。*

### 对 Hopper 移植的含义

根据社区公开的 PR 与跟踪 issue，已知卡点包括：

- **FlashMLA**：上游于 2026 年 9 月 30 日移除 Hopper 支持，SM90 注意力 kernel 需自行补齐；
- **DeepGEMM**：Hopper 版本仅覆盖 FP8 NT 布局与 FP32 缩放系数，缺少 FP4 分支与 UE8M0 缩放格式；融合的 mega-kernel（如 `fp8_fp4_mega_moe`）在 Hopper 上无对应实现；
- **DeepSelect**：需额外加入 `sm_90a` 编译目标，top-k 才能在 Hopper 上运行；
- **依赖版本**：有验证者在 H200 上发现社区分支锁定的 Humming 依赖存在问题。

社区 SM90 分支在 8 卡与 4 卡 H20-3e 上实现了端到端运行，但 FP8 GEMM 的分块配置只针对 H20 调优，H200 走启发式默认值，全模型在 H200 上的结果尚无公开报告。

---

## Q2：没有 FP4 张量核心，FP4 权重在 Hopper 上如何计算？

### 结论

采用"FP4 存储、片上反量化为 FP8 计算"的融合方案：显存占用与 decode 带宽与 Blackwell 一致，转换无精度损失，代价是 prefill 阶段无法获得 FP4 的算力倍增。

### 机制

FP4（E2M1）仅有 8 个非负取值：0、0.5、1、1.5、2、3、4、6，依靠按组共享的缩放系数表示更大的动态范围，真实值 = FP4 值 × scale。DeepSeek V4 系列的 MoE 专家权重以 FP4 存储，其余参数为 FP8。

张量核心只能对其电路支持的格式执行乘加。Blackwell 有 FP4 乘法电路，Hopper 只支持 FP8、FP16/BF16、TF32、INT8，因此 FP4 权重必须先转为 FP8。由于 FP4 的全部取值均可被 FP8 精确表示，该转换是无损的。

![图 3：FP4 权重在 B200 和 H200 上的三条数据通路](/img/posts/h200-attention/03-fp4-fp8-path.svg)

*图 3：左侧框宽度代表权重的显存占用。*

| 方案 | 显存占用 | 计算速率 | 代价 |
|---|---|---|---|
| B200：直接计算 FP4 | 1× | FP4 峰值约为 FP8 的 2 倍 | 无 |
| H200 方案一：FP4 存储，片上转 FP8 | 1× | FP8 | 每次计算增加一次反量化 |
| H200 方案二：直接存 FP8 | 2× | FP8 | 显存容量与带宽开销翻倍 |

### 对 Hopper 移植的含义

社区选择方案一，并将反量化融合进 grouped GEMM（dequant-fused FP8 grouped GEMM）：权重以 4 位体积从 HBM 读入，在片上转换后直接送入张量核心。

这一选择的依据是推理两阶段的瓶颈不同。decode 阶段受限于显存带宽，方案一保留了 4 位的搬运量，基本不受影响；prefill 阶段受限于算力，H200 只能以 FP8 速率计算，这一差距由硬件决定。方案二则使专家权重体积翻倍，原本可容纳于 4 张 141GB 显卡的模型可能无法放下。

---

## Q3：注意力的计算结构是什么？它为什么允许分块流式计算？

### 结论

注意力对每个查询计算的是值向量的 softmax 加权平均，其分子与分母都是对键的求和，因而可以分批累加；数值稳定所需的最大值可以在发现更大值时对已有累加量整体修正。这是 FlashAttention 及其所有变体的数学基础。

### 3.1 计算结构

注意力可以理解为一次可微分的软检索：

| 检索系统 | 注意力 |
|---|---|
| 查询 | 当前 token 的 `q` |
| 文档的匹配字段 | 每个 token 的 `k` |
| 文档的返回内容 | 每个 token 的 `v` |
| 相关度 | `q · k` |
| 返回 top-1 | 对所有 `v` 按 softmax 权重加权平均 |

使用软选择而非 top-1，是为了让输出随分数连续变化，从而使梯度能够传回投影参数。

每层的输入是各 token 的隐藏状态 `x`（V4.1 为 5120 维），由三个可学习矩阵投影得到：`q = x·W_Q`，`k = x·W_K`，`v = x·W_V`。在自注意力中，token i 的查询 `q_i` 来自 `x_i`，被检索的 `k_j`、`v_j` 来自序列中其他 token 的 `x_j`。对第 i 个 token：

```
s_ij = q_i · k_j / sqrt(d)                 打分
p_ij = exp(s_ij) / Σ_j exp(s_ij)           归一化
o_i  = Σ_j p_ij · v_j                      加权平均
```

除以 `sqrt(d)` 是为了抵消点积方差随维度增长，避免 softmax 过尖导致梯度消失。

![图 4：注意力的矩阵形状](/img/posts/h200-attention/04-attention-shapes.svg)

*图 4：N 个 token 并行时，分数矩阵 S = QKᵀ 为 N×N。*

几个与推理实现直接相关的事实：

- **因果遮挡**：生成模型中 `j > i` 的分数被置为 −∞；
- **多头与 KV 头**：V4.1 的 64 个查询头共享 1 个 KV 头；
- **KV cache**：历史 token 的 `k`、`v` 不随后续 token 变化，可缓存复用；
- **prefill 与 decode**：prefill 一次处理全部输入，是完整的 N×N 计算；decode 每步只有一个查询，却要读取整个 KV cache，瓶颈在显存带宽。

### 3.2 问题：中间矩阵的显存流量

GPU 存储分为容量大但慢的 HBM，与每个 SM 仅数百 KB 但快得多的片上 SRAM。朴素实现将打分、softmax、乘 V 作为三个独立算子，N×N 的 S 和 P 都要往返显存。上下文为 10 万 token 时，单个头、单层的 S 在 BF16 下即达 20GB。

![图 5：普通注意力与 FlashAttention 的显存读写对比](/img/posts/h200-attention/05-memory-traffic.svg)

*图 5：FlashAttention 将三步融合在片上完成，S 与 P 从不完整地写入显存。*

### 3.3 在线 softmax 的推导

将输出写成：

```
o_i = ( Σ_j exp(s_ij) · v_j ) / ( Σ_j exp(s_ij) )
```

分子与分母都是对 j 的求和，可以分批累加。对每一行维护三个量：

- `m`：已见分数的最大值，用于数值稳定，实际计算 `exp(s − m)`；
- `l`：分母累加量 `Σ exp(s − m)`；
- `O`：分子累加量 `Σ exp(s − m) · v`。

读入新的一块 K、V 时：若出现更大分数，令 `m_new = max(m_old, 块内最大分数)`，已有的 `l` 与 `O` 整体乘以 `exp(m_old − m_new)`，即把每一项从 `exp(s − m_old)` 换算为 `exp(s − m_new)`；随后加入本块的贡献，并丢弃本块分数。全部处理完后输出 `O / l`，`exp(−m)` 在分子分母间相消。

以一个查询对 8 个键的分数为例，每 2 个键一块：

| key | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 分数 s | 1.2 | 0.3 | 2.1 | −0.5 | 0.8 | 3.0 | 0.1 | 1.5 |
| v | 2 | −1 | 4 | 0 | 3 | 1 | −2 | 5 |

![图 6：在线 softmax 的逐块累加](/img/posts/h200-attention/06-online-softmax.svg)

*图 6：横条总长即为 l。块 1、块 2 出现更大分数，旧项整体乘 0.41（三次恰好都相差 0.9）；块 3 最大值不变，无需修正。最终 O / l = 2.054，与一次性计算 softmax 的结果完全一致。*

作为交叉验证：不做最大值修正、直接以 `exp(s)` 累加，最终分母为 41.34、分子为 84.92，比值同为 2.054；在线版本每一步的 `l` 与 `O` 恰为这两个累加量除以 `exp(当前 m)`。

### 3.4 可分解性的三个来源

1. **行间独立**：第 i 行只依赖 `q_i` 与全部 K、V，可将 Q 按行分块并行于不同 SM；
2. **行内可加**：分子、分母都是求和，K、V 可按块流式处理；
3. **溢出可修正**：最大值变化时，已有累加量乘一个标量即可换算到新基准。

需要指出，N×N 中间矩阵只出现在 prefill。decode 时分数仅为 1×N，主要矛盾转为读取 KV cache 的带宽，这正是 Q4 的问题。

---

## Q4：decode 阶段如何在压缩 KV cache 上高效计算注意力？

### 结论

MLA 通过矩阵吸收使注意力直接在压缩向量上进行，K 与 V 共享同一份数据，所有头共享同一份 KV；FlashMLA 围绕这一形状组织片上循环，并通过 split-KV 把单个请求的计算分散到多个 SM，再用在线 softmax 的合并规则汇总。

### 4.1 MLA 的矩阵吸收

以 DeepSeek V3 公开的 MLA 设计为例（V4 系列在此基础上有改动）：KV cache 中每个 token 仅存储压缩向量 `c_j`（512 维，另加 64 维位置编码部分），`k_j = W_UK · c_j`，`v_j = W_UV · c_j`。若每次先还原再计算，压缩便失去意义。利用结合律：

```
q_i · k_j          = (W_UKᵀ · q_i) · c_j
Σ_j p_ij · v_j     = W_UV · ( Σ_j p_ij · c_j )
```

查询先做一次变换即可直接与 `c_j` 点积；输出先在压缩空间加权平均，最后统一乘以 `W_UV`。于是 `c_j` 同时充当 K 与 V，且被所有头共享。

### 4.2 SM 内的计算循环

decode 时一个请求的计算形状是 64 个头的查询（可视为 64 行）对 N 个压缩向量：

1. 由 TMA 将一页压缩向量从 HBM 搬入片上缓存，页位置由页表给出（分页 KV cache）；
2. 张量核心计算 64 个头对该页的分数；
3. 每个头按在线 softmax 更新 `m`、`l`、`O`；
4. 同一页数据作为 V 参与加权累加，一次搬运、两次使用；
5. 处理下一页，直至结束。

V4.1 在此之上增加了 indexer：一个轻量的辅助注意力先为压缩条目打分，每个查询仅保留最相关的 512 个（之前还有一级候选筛选），kernel 因此需要支持稀疏的按块读取。

### 4.3 SM 间并行：split-KV

单个 decode 请求只有一行查询，若由一个 SM 完成，其余上百个 SM 将闲置。FlashMLA 将请求的 KV cache 切分为若干段：

![图 7：FlashMLA 的 split-KV 与合并](/img/posts/h200-attention/07-flashmla-split-kv.svg)

*图 7：各 SM 计算局部 m、l、O，合并 kernel 以全局最大值统一基准后相加。*

这与 3.3 节是同一数学性质：顺序累加改为并行计算后合并。FlashMLA 的接口在计算前调用 `get_mla_metadata`，依据各请求长度预先规划切分，以平衡各 SM 负载。

### 4.4 卡间并行

decode 阶段注意力通常采用数据并行：每张卡负责一部分请求，并持有这些请求完整的 KV cache，注意力计算不跨卡交换 K、V。当单个请求的上下文超出单卡容量时，采用上下文并行（CP），将 KV cache 按 token 切分到多卡，各卡计算局部 `m`、`l`、`O` 后跨卡合并，规则不变。

---

## Q5：MoE 层的数据如何在多卡之间流动？

### 结论

每层 MoE 包含路由、分发、专家计算、汇总四步，分别由 top-k、all-to-all 通信、grouped GEMM 与反向通信完成。专家分工由训练自发形成，需要负载均衡机制维持。

### 机制

![图 8：一个 token 在一层 MoE 里的分发与汇总](/img/posts/h200-attention/08-moe-dispatch.svg)

*图 8：示意图，专家编号与所在 GPU 为假设。*

| 步骤 | 内容 | 算子 |
|---|---|---|
| 路由 | 线性层为 384 个专家打分，取 top-6 并计算权重 | top-k kernel（DeepSelect，推测） |
| 分发 | batch 内所有 GPU 之间的 all-to-all，机内 NVLink，跨机 RDMA | DeepEP |
| 专家计算 | 各卡按专家分组，执行大小不一的矩阵乘 | DeepGEMM（grouped GEMM） |
| 汇总 | 结果沿原路返回，按路由权重加和，并加上共享专家输出 | DeepEP |

DeepGEMM 之所以不能用通用 BLAS 替代，在于三点：kernel 内需处理 FP8/FP4 及分组缩放；各专家 token 数不同，需在一次启动内完成所有分组矩阵乘；以及将多步融合为 mega-kernel 以减少显存往返。

专家并非人为分配。结构上只规定若干同构前馈网络与一个路由器，参数随机初始化；训练中仅被选中的专家接收梯度，路由与专家能力相互强化，分工逐渐形成。为避免少数专家被过度选择，DeepSeek V3 为每个专家的路由分数引入可调偏置，以无辅助损失的方式实现负载均衡。实际形成的分工通常偏向词法与句法层面，而非清晰的主题划分。

---

## Q6：为什么每层同时需要注意力和前馈两种子层？

### 结论

注意力的输出是值向量的凸组合，只能在 token 之间搬运和混合信息；特征交叉与参数化知识的检索依赖前馈层的非线性。两者通过残差流交替写入同一隐藏状态。

### 6.1 注意力不产生新特征

![图 9：一层里注意力与 MoE 的分工](/img/posts/h200-attention/09-attention-vs-moe.svg)

*图 9：注意力是模型中唯一在 token 之间交换信息的部件；前馈层逐 token 独立计算。*

`o_i = Σ p_ij · v_j` 始终落在各 `v_j` 张成的凸包内。这带来两个局限：

1. **无法表达交互**：例如"不好"，注意力能让"好"的位置取到"不"的信息，但得到的是二者的混合，而非"负面"这一新特征；若各层均为线性混合，深度将失去意义；
2. **无法获取上下文之外的知识**：上下文未出现的事实无从检索，这类知识存储在参数中，主要位于前馈层。

### 6.2 非线性提供的能力

以情感的极性翻转为例，p 表示是否含正面词，n 表示是否含否定词：

| 词组 | n | p | 情感 |
|---|---|---|---|
| 坏 | 0 | 0 | 负 |
| 好 | 0 | 1 | 正 |
| 不坏 | 1 | 0 | 正 |
| 不好 | 1 | 1 | 负 |

线性函数 `a·p + b·n + c` 无法拟合：由前两行需 `a > 0`，由后两行需 `a < 0`。这是异或问题，与 CTR 预估中逻辑回归无法自动学习特征交叉同理。

引入 ReLU，令 `h1 = ReLU(p − n)`、`h2 = ReLU(n − p)`，输出 `h1 + h2`，四种情况全部正确。截断操作提供了条件分支的能力。

![图 10：线性分不开“不好/不坏”，ReLU 的两个开关可以](/img/posts/h200-attention/10-xor-relu.svg)

*图 10：任一直线都无法分开对角的同类点；两个 ReLU 单元分别覆盖两个正类区域。*

现代模型采用 SwiGLU：

```
FFN(x) = W2 · ( SiLU(W1·x) ⊙ (W3·x) )
```

SiLU 提供平滑的开关，逐元素乘积则直接实现"A 且 B"：两个检测器同时激活时乘积才显著。vLLM 在加载 DeepSeek V4 权重时将 `w1`、`w3` 合并为 `gate_up_proj`，以一次矩阵乘同时计算两路。

### 6.3 残差流

```
x ← x + attention(x)
x ← x + ffn(x)
```

隐藏状态 `x` 是所有层共享的读写通道：注意力写入从上下文汇集的信息，前馈层写入加工后的特征与从参数中取出的知识。`o_i` 是本层新增的上下文信息，`x_i` 是截至本层的累积表示。

### 6.4 从隐藏状态到下一个 token

![图 11：从最后一个 token 的向量到下一个 token 的生成循环](/img/posts/h200-attention/12-next-token-loop.svg)

*图 11：LM head 只是一个 5120 × 129,280 的线性层，生成循环依赖 KV cache 只计算新位置。*

最后一个位置的隐藏状态经归一化后与输出矩阵相乘，得到词表上的 logits，softmax 后按贪心、温度或 top-p 采样。新 token 经 embedding 再次通过 40 层，只计算这一个位置，其查询读取全部 KV cache 并追加自身的 K、V。

DSpark 是附加于模型的多 token 草稿头：用最后的隐藏状态一次预测后续多个 token，主模型将其作为新 token 一次性验证，保留一致的部分。验证机制保证输出与不启用时一致。DSpark 参数与主模型存放在同一 checkpoint 中，V4 Flash 加入 DSpark 后仓库总参数由 284B 增至 304B。

---

## Q7：5120 维的隐藏状态为什么足以承载语义？

### 结论

特征由方向而非坐标轴表示；高维空间中近似正交的方向数随维数指数增长，在特征稀疏激活的前提下可叠加承载远多于维数的特征。同时，信息分布于各位置、各层的隐藏状态、模型参数与 KV cache 之中，单个向量只需承担工作记忆。具体维度由算力预算与工程约束决定，而非理论推导。

### 7.1 高维空间的近似正交性

对每个维度随机采样 20,000 对方向并统计夹角余弦：

![图 12：随机方向两两夹角的余弦分布随维度变化](/img/posts/h200-attention/11-cosine-dimension.svg)

*图 12：维度越高，随机方向的余弦越集中于 0。*

| 维度 | 典型 \|cos\| | 对应夹角 | \|cos\| > 0.1 的比例 |
|---|---|---|---|
| 2 | 0.634 | 约 51° | 94% |
| 10 | 0.259 | 约 75° | 77% |
| 100 | 0.080 | 约 85° | 32% |
| 5120 | 0.011 | 约 89° | 0% |

允许少量重叠时，可容纳的方向数随维数指数增长。模型借此将大量特征叠加于同一空间（superposition），可解释性研究使用稀疏自编码器可从大模型中分解出数百万个可解释特征。叠加成立的前提是稀疏性：任一时刻只有少数特征激活，而自然语言恰好满足这一条件。

### 7.2 信息的分布式存储

| 位置 | 内容 |
|---|---|
| 各位置、各层的隐藏状态 | 对当前上下文的理解（工作记忆） |
| 模型参数 | 世界知识与语言规律（长期记忆） |
| KV cache | 历史 token 的 K、V，可随时检索 |

一个 5120 维 BF16 向量的原始容量为 81,920 比特，而从 129,280 个词中指定一个仅需约 17 比特。这只是上限：后续层只能通过线性投影读取信息，且需容忍叠加带来的串扰，有效容量远低于此。该对比仅说明容量在量级上并非瓶颈。

### 7.3 维度如何确定

规模定律（Kaplan 等 2020；Hoffmann 等 2022）表明，模型性能主要由参数量、数据量与算力决定，在较宽范围内对宽度与深度的分配不敏感。实际设计在算力预算下权衡：

| | 加宽 | 加深 |
|---|---|---|
| 单层计算量 | 约随 d² 增长 | 不变 |
| 延迟 | 并行度高 | 层间串行，线性增加 |
| 多卡切分 | 更易切分 | 流水线更长 |

维度通常取 128 的整数倍以匹配张量核心分块，5120 = 128 × 40。DeepSeek-V3 的隐藏维度为 7168，V4.1 Flash 作为新架构族中最小的型号采用 5120。宽度不足的代价体现在冷门知识的保留、叠加串扰导致的幻觉，以及长上下文细节的丢失。

---

## Hopper 移植的验证方案

基于以上分析，一份对上游维护者有价值的 H200 测试报告应分层组织：

**环境与可复现性**

- 驱动、CUDA、GCC、Python 版本；
- DeepInfer 分支与 commit，以及 FlashMLA、DeepGEMM、DeepEP、DeepSelect、Humming 等依赖的 commit SHA；
- 卡数与 NVLink 拓扑。

**第一层：kernel 编译与单元测试**

- 各子模块独立编译并运行自带单元测试；
- 按 Q1 中的三类失败分别记录。

**第二层：离线推理正确性**

- 固定输入与贪心解码，与 H20 或 SM100 的输出逐 token 比对；
- 在小规模评测集（如 GSM8K 子集）上检查分数。

**第三层：DSpark 开关对比**

- 贪心解码下两种模式的输出应完全一致；
- 记录草稿接受率、吞吐与首 token 延迟。由于验证机制保证一致性，输出不一致或接受率异常偏低，是定位 kernel 数值问题的低成本探针。

**性能（可选）**

- 不同 batch 与上下文长度下的 tokens/s；
- 确认 FP8 GEMM 是否使用启发式配置，评估针对 H200 调优的收益。

## 总结

将 DeepInfer 从 SM100 迁移到 SM90，表面上是"加一个编译目标"，实际上涉及张量核心指令、片上存储层次与低精度数据通路的整体重写。理解这些改动为何必要、如何正确，需要回到更底层的问题：注意力为何能被分块计算，MLA 为何能让 K、V 共享一份数据，MoE 的通信与计算如何组织，以及隐藏状态的容量从何而来。这些问题的答案共同决定了一个推理栈在新硬件上的正确性边界与性能上限。

---

## 参考资料

- [DeepInfer 仓库与 README](https://github.com/deepseek-ai/DeepInfer)
- [DeepInfer PR #92：SM90（Hopper）在 H20-3e 上端到端运行](https://github.com/deepseek-ai/DeepInfer/pull/92)
- [DeepInfer PR #148：添加 Hopper（SM90）支持](https://github.com/deepseek-ai/DeepInfer/pull/148)
- [DeepInfer RFC #47：H20-3E 的 SM90 支持](https://github.com/deepseek-ai/DeepInfer/pull/47)
- [SM90 平台支持跟踪 issue（JianWang97/DeepInfer #4）](https://github.com/JianWang97/DeepInfer/issues/4)
- [vLLM recipe：DeepSeek-V4.1-Flash](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM 文档：`deepseek_v4` 模型实现](https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/nvidia/model/)
- [ApX：DeepSeek V4.1 Flash 规格](https://apxml.com/models/deepseek-v4-1-flash)
- Dao 等，*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*，2022
- DeepSeek-AI，*DeepSeek-V2* 与 *DeepSeek-V3* 技术报告
- Kaplan 等，*Scaling Laws for Neural Language Models*，2020；Hoffmann 等，*Training Compute-Optimal Large Language Models*，2022
- Elhage 等，*Toy Models of Superposition*，2022

*附注：FlashMLA 内部实现的描述基于 DeepSeek V3 时期的公开设计，未对照当前源码；V4.1 的结构参数以官方模型卡为准。*
