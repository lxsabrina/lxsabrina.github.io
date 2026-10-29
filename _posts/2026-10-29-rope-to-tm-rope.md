---
title:      从第一原理推导 RoPE，再到 Qwen-Omni 的 TM-RoPE
subtitle:   旋转、相对位置、手算验证与 PyTorch 实现
date:       2026-10-29 10:30:00 +0800
author:     张弦
tags:
    - 位置编码
    - RoPE
    - 多模态
    - Qwen-Omni
---

## 摘要

Qwen-Omni 能把文本、图像、音频、视频放进同一个序列，位置编码上靠的是对 RoPE 的一个扩展：TM-RoPE。要读懂它，先要把 RoPE 本身推明白。

本文从"注意力为什么不认识顺序"出发，推导 RoPE，用 d = 4 的例子手算验证，再用 PyTorch 实现并跑通；然后讲它怎样扩展成 (时间, 高, 宽) 三个分量，以及音频和视频怎样在时间轴上对齐。文中所有数值都是代码实际运行的结果。

| # | 问题 |
|---|---|
| Q1 | 位置信息为什么必须是"相对"的？公式里的 q、k、f、g 是什么？ |
| Q2 | 为什么旋转能让绝对位置抵消，只剩相对距离？ |
| Q3 | RoPE 的完整公式是什么？θ 从哪来，要训练吗？ |
| Q4 | 怎么用 PyTorch 实现和验证？训练时怎么用？ |
| Q5 | TM-RoPE 在 RoPE 上改了什么？音频和视频怎样对齐？ |

## 核心结论

- **注意力本身不认识顺序**。要注入位置，并且希望打分只依赖两个 token 的相对距离 m − n。
- **RoPE 的做法是旋转**：把 q、k 的相邻两维配成一对，按"位置 × θ"的角度旋转。点积时两个角度相减，m 和 n 各自消失，只剩 n − m。
- **θ 是常数，不是训练参数**。θᵢ 由 base 和 head_dim 算出，多个频率像钟表的秒针和分针，分别负责近距离和远距离。
- **RoPE 只要求"位置是一个数"**。M-RoPE 和 TM-RoPE 把这个数换成 (t, h, w)，三个分量各占一部分频率，旋转公式不变。
- **音视频对齐靠真实时间**。音频每个 token 对应固定时长（Qwen2.5-Omni 约 40 ms，Qwen3-Omni 约 80 ms），时间 ID 绑定真实时间；两篇论文都没有描述显式的时间戳 token。

---

## 1. 问题：注意力不认识顺序

注意力里，每个 token 的隐藏状态经过两个线性变换，得到两个向量：

- **q（query，查询）**：这个 token 在找什么。
- **k（key，键）**：这个 token 能提供什么。

token i 对 token j 的关注程度，就是点积 `qᵢ · kⱼ`。q 和 k 只由内容算出，把序列里的 token 打乱，每对 token 的分数不变，所以模型分不出"我爱你"和"你爱我"。位置必须另外注入。

设提问的 token 在位置 m，被查询的 token 在位置 n。我们要设计一个操作 f，满足：

```
⟨ f(q, m), f(k, n) ⟩ = g(q, k, m − n)
```

- **f**：待设计的"注入位置"操作。输入内容向量和位置，输出带位置的新向量。f(q, m) 是新 query，f(k, n) 是新 key。
- **g**：只是一个函数名，意思是"结果只能由 q、k 的内容和差值 m − n 决定"，不能单独依赖 m 或 n。

为什么要相对位置？语言里"主语和谓语隔了几个词"比"它们在第几个位置"更本质，而且相对位置天然能泛化到不同长度的句子。

最早的做法是把位置向量 pₘ 直接加到输入上，打分展开成四项：

```
(q + pₘ)·(k + pₙ) = q·k + q·pₙ + pₘ·k + pₘ·pₙ
```

内容和位置缠在一起，没有一项显式地只依赖 m − n，相对关系得靠模型自己学。RoPE 换了一个思路。

## 2. 第一原理：旋转让位置相减

二维平面里的旋转有两个性质：

- 旋转不改变向量长度；
- 旋转可以叠加：先转 a 再转 b，等价于转 a + b；反过来 `R(a)ᵀ · R(b) = R(b − a)`。

于是定义 f：位置 m 的向量，旋转 m·θ 的角度。

```
f(x, m) = R(mθ) · x
⟨R(mθ)q, R(nθ)k⟩ = qᵀ R(mθ)ᵀ R(nθ) k = qᵀ R((n − m)θ) k
```

m 和 n 各自消失，只剩 n − m，正好是上一节要的 g。

![图 1：旋转抵消](/img/posts/rope-to-tm-rope/01-rotation-cancel.svg)

*图 1：取 θ = 1 rad，q = k。左右两例的位置不同（3 和 1，10 和 8），旋转后的绝对角度完全不同，但夹角都是 2 rad，点积相同。*

用角度再说一遍。设原始 q 的方向角是 α，k 是 β，旋转后：

```
q' 的方向角 = α + mθ
k' 的方向角 = β + nθ
夹角       = (β − α) + (n − m)θ
```

点积由夹角决定。第一项 β − α 只与内容有关，第二项只与位置差有关，m 和 n 相减时抵消。

**关于 rad（弧度）**：圆上一段弧，弧长等于半径时对应的圆心角是 1 rad，约 57.3°；整圈 360° = 2π rad。`cos`、`sin` 在程序里默认吃弧度，本文所有角度都是弧度。

## 3. RoPE 的完整公式

真实的 q、k 有 d 维（d 即 head_dim，每个注意力头的向量长度，常见 64 或 128）。做法是把相邻两维配成一对，共 d/2 对，每对各转各的，每对用不同的角速度：

```
θᵢ = base^(−2i/d)，i = 0, 1, …, d/2 − 1，常见 base = 10000
```

位置 m 的完整旋转矩阵是块对角的，第 i 个 2×2 块是：

```
[ cos(m·θᵢ)  −sin(m·θᵢ) ]
[ sin(m·θᵢ)   cos(m·θᵢ) ]
```

也可以把每一对看成复数 `x₂ᵢ + i·x₂ᵢ₊₁`，旋转就是乘以 `e^(i·m·θᵢ)`。

**θ 要求解吗？不用。** 它是 base 和 d 的确定函数，所有层、所有头、所有位置共用，训练中不更新。d = 4 时 θ = (1, 0.01)；d = 128 时从 1 一路降到约 0.000115。

**为什么要多个频率？** 一个频率转满一圈就回到原点，距离相差一圈的两个位置分不开。多个频率像钟表：秒针（高频）分辨相邻几个 token，分针、时针（低频）分辨很远的距离。

![图 2：不同频率的旋转块看到的距离](/img/posts/rope-to-tm-rope/02-multi-frequency.svg)

*图 2：横轴是相对距离 Δ，纵轴是 cos(Δ·θ)。θ = 1 的块随距离快速振荡，θ = 0.01 的块在 64 个位置内几乎不变。*

RoPE 作用在 Q 和 K 上，不作用在 V：位置只决定"谁关注谁"，不应该改变被取走的内容。

## 4. 手算验证：d = 4

d = 4 只有两个块，θ₀ = 1，θ₁ = 0.01。位置 m 的矩阵 R(m) 是 4×4 的块对角阵，每个位置格式完全相同，只是块里的数值随 m 变化。

m = 0（所有角度为 0，单位矩阵，不旋转）：

```
[ 1  0  0  0 ]
[ 0  1  0  0 ]
[ 0  0  1  0 ]
[ 0  0  0  1 ]
```

m = 1（角度 1 rad 和 0.01 rad）：

```
[ 0.54030  −0.84147   0         0       ]
[ 0.84147   0.54030   0         0       ]
[ 0         0         0.99995  −0.01000 ]
[ 0         0         0.01000   0.99995 ]
```

m = 2（角度 2 rad 和 0.02 rad）：

```
[ −0.41615  −0.90930   0         0       ]
[  0.90930  −0.41615   0         0       ]
[  0         0         0.99980  −0.02000 ]
[  0         0         0.02000   0.99980 ]
```

取 q = k = (1, 0, 1, 0)，不旋转时 q·k = 2，没有位置信息。看两组位置差相同的例子：

| 位置 | q' = R(m)q | k' = R(n)k | q'·k' |
|---|---|---|---|
| m = 3，n = 1 | (−0.98999, 0.14112, 0.99955, 0.03000) | (0.54030, 0.84147, 0.99995, 0.01000) | 0.58365 |
| m = 10，n = 8 | (−0.83907, −0.54402, 0.99500, 0.09983) | (−0.14550, 0.98936, 0.99680, 0.07991) | 0.58365 |

两个例子里 q'、k' 的每个分量都不同，但点积一致。理论值是 `cos(2θ₀) + cos(2θ₁) = cos 2 + cos 0.02 = 0.58365`，与手算吻合。

## 5. PyTorch 实现

先给出四个函数。验证时用 float64，实际训练的精度选择与 RoPE 的原理无关，这里不展开。

```python
import torch

torch.set_default_dtype(torch.float64)


def rope_freqs(d: int, base: float = 10000.0) -> torch.Tensor:
    """θ_i = base^(-2i/d)，i = 0 .. d/2-1，形状 [d/2]。常量，没有可训练参数"""
    return base ** (-torch.arange(0, d, 2) / d)


def rotation_matrix(m: int, d: int, base: float = 10000.0) -> torch.Tensor:
    """教学版：显式构造 d×d 的块对角旋转矩阵（相邻两维配对）"""
    theta = rope_freqs(d, base)
    R = torch.zeros(d, d)
    for i, th in enumerate(theta):
        c, s = torch.cos(m * th), torch.sin(m * th)
        R[2 * i, 2 * i], R[2 * i, 2 * i + 1] = c, -s
        R[2 * i + 1, 2 * i], R[2 * i + 1, 2 * i + 1] = s, c
    return R


def apply_rope_interleaved(x, pos, base=10000.0):
    """交错排布（原论文）。x: [L, d]，pos: [L]，每个 token 的位置"""
    ang = pos[:, None] * rope_freqs(x.shape[-1], base)[None, :]  # [L, d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]  # 偶数维、奇数维配对
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)  # 前一半、后一半配对
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_half(x, pos, base=10000.0):
    """半切排布（LLaMA / HuggingFace 常用）"""
    ang = pos[:, None] * rope_freqs(x.shape[-1], base)[None, :]  # [L, d/2]
    ang = torch.cat([ang, ang], dim=-1)  # [L, d]
    return x * ang.cos() + rotate_half(x) * ang.sin()
```

逐个解释：

- **`rope_freqs`**：直接实现 θᵢ 的公式，返回 d/2 个角速度。
- **`rotation_matrix`**：把每个 2×2 块填到对角线上，和第 4 节的矩阵一一对应，只用来教学和验证，实际不会建矩阵。
- **`apply_rope_interleaved`**：`pos[:, None] * freqs[None, :]` 得到 `[L, d/2]` 的角度表，每行是一个 token，每列是一个频率块。`x[..., 0::2]` 和 `x[..., 1::2]` 取出每对的两个分量，按旋转公式逐元素计算，复杂度 O(d)，而不是矩阵乘法的 O(d²)。
- **`apply_rope_half`**：把第 i 维和第 i + d/2 维配对，写成 `x·cos + rotate_half(x)·sin`。它和交错排布数学上等价，只差一次维度重排。**加载别人训练好的权重时，配对方式必须和对方一致**，否则注意力会算错。

接下来验证第 4 节的数字，以及三个性质：

```python
# 1) d=4 的两个例子
q = k = torch.tensor([1.0, 0.0, 1.0, 0.0])
for m, n in ((3, 1), (10, 8)):
    qm, kn = rotation_matrix(m, 4) @ q, rotation_matrix(n, 4) @ k
    print(f"m={m}, n={n}: q'·k' = {float(qm @ kn):.5f}")

# 2) 矩阵乘法 == 逐元素 == 半切（重排后）
torch.manual_seed(0)
x, pos = torch.randn(5, 8), torch.arange(5.0)
A = torch.stack([rotation_matrix(int(p), 8) @ x[i] for i, p in enumerate(pos)])
B = apply_rope_interleaved(x, pos)
perm = torch.cat([torch.arange(0, 8, 2), torch.arange(1, 8, 2)])  # 偶数维在前，奇数维在后
C = apply_rope_half(x[:, perm], pos)
print(torch.allclose(A, B), torch.allclose(C, B[:, perm]))

# 3) 相对位置不变性（d=128）
q, k = torch.randn(1, 128), torch.randn(1, 128)
def score(m, n):
    return float(apply_rope_interleaved(q, torch.tensor([float(m)])) @
                 apply_rope_interleaved(k, torch.tensor([float(n)])).T)
for m, n in ((5, 3), (105, 103), (5005, 5003), (7, 3)):
    print(f"m={m}, n={n}, m-n={m - n}: score={score(m, n):.6f}")
```

实际输出：

```
m=3, n=1: q'·k' = 0.58365
m=10, n=8: q'·k' = 0.58365
True True
m=5, n=3, m-n=2: score=7.629485
m=105, n=103, m-n=2: score=7.629485
m=5005, n=5003, m-n=2: score=7.629485
m=7, n=3, m-n=4: score=1.773059
```

位置差都是 2 的三组，不管绝对位置是 5 还是 5005，分数完全相同；位置差变成 4，分数才变。

## 6. 训练中怎么用

RoPE 放在线性投影之后、点积之前，只旋转 q 和 k：

```python
import torch.nn as nn
import torch.nn.functional as F


class RoPEAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, pos):  # x: [B, L, d_model]，pos: [L]
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # [B, H, L, dh]
        q, k = apply_rope_half(q, pos), apply_rope_half(k, pos)   # 只旋转 q、k，不动 v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, L, -1))
```

用同一组输入检验：

```python
attn = RoPEAttention(d_model=32, n_heads=4)
x, pos = torch.randn(2, 6, 32), torch.arange(6.0)
print(torch.allclose(attn(x, pos), attn(x, pos + 1000), atol=1e-8))      # 整体平移
print(torch.allclose(attn(x, pos), attn(x, torch.tensor([0., 1, 2, 3, 4, 9]))))  # 改一个位置
print(sum(p.numel() for p in attn.parameters()))
```

输出 `True`、`False`、`4096`：所有位置整体平移 1000，输出不变；只挪动最后一个 token 的位置，输出改变；参数量只有 QKV 与输出投影的 3·32·32 + 32·32 = 4096，RoPE 本身没有参数。

训练时有几点值得记住：

- **没有可训练参数**。θ 固定，cos/sin 表可以预先算好缓存。学习的是 W_q 和 W_k：它们决定内容落在哪些频率块上，也就决定了模型对什么距离敏感。
- **反向传播照常穿过旋转**。旋转是固定的正交线性变换，梯度只是乘上它的转置。
- **位置由数据管线给出**。普通训练就是 0, 1, 2, …；把多条短样本打包成一条长序列时，通常每条样本的位置从 0 重新开始，同时用 attention mask 防止互相看见。
- **推理时 KV cache 存的是旋转后的 k**。新 token 只需要用自己的位置旋转自己的 q 和 k，旧 token 不用重算。
- **训练长度之外会有问题**。大于训练长度的位置，部分低频块的角度落在没见过的范围。位置插值、调大 base、YaRN 等方法就是在处理这个问题，本文不展开。

## 7. 从 RoPE 到 M-RoPE 与 TM-RoPE

回头看第 2 节的推导：RoPE 对位置的唯一要求，是 m 是一个数，旋转角度是 m·θᵢ。它不关心这个数是"第几个 token"，还是别的含义。

**M-RoPE**（Qwen2-VL 引入）把这个数换成三个：(t, h, w)，即时间、高度（height）、宽度（width）。**TM-RoPE**（Qwen2.5-Omni）在此基础上加入音频，并把时间分量绑到真实时间上。

### h 和 w 是什么

图像进模型前被切成小方块（patch），每个 patch 一个 token。这些 token 在序列里排成一列，但在画面上是二维的。h 是 patch 的行号，w 是列号。以 3 行 4 列为例：

```
        w=0    w=1    w=2    w=3
h=0   (0,0)  (0,1)  (0,2)  (0,3)
h=1   (1,0)  (1,1)  (1,2)  (1,3)
h=2   (2,0)  (2,1)  (2,2)  (2,3)
```

序列里 (0,3) 和 (1,0) 序号相邻，画面上却一个在第一行最右、一个在第二行最左，距离很远。有了 (h, w)，注意力才能分清上下左右的邻居关系。

### 各模态怎么赋值

| 模态 | t | h | w |
|---|---|---|---|
| 文本 | 序号 p | p | p |
| 音频 | 时间 ID（真实时间 ÷ 一个 token 的时长） | 同 t | 同 t |
| 图像 | 常数（同一时刻） | 行 | 列 |
| 视频 | 帧的时间 ID | 行 | 列 |

- 文本三个分量相同，旋转的效果与普通一维 RoPE 完全一致，纯文本能力不受损。
- Qwen2-VL 的视频 t 按帧序号递增（每秒采 2 帧）；Qwen2.5-Omni 改成按真实时间换算，帧率变了，时间 ID 仍对应真实时间。
- 不同模态拼接时，后一个模态的编号接在前一个模态的最大编号之后。

### 三个分量怎么分配频率

d/2 个旋转块要分给 t、h、w。每个块只用一个分量的位置去旋转，其余公式完全不变。

![图 3：t、h、w 的频率分配](/img/posts/rope-to-tm-rope/03-mrope-frequency-layout.svg)

*图 3：上图是 Qwen2-VL 的连续分块，时间分量拿到最前面（最快）的 16 个频率；下图是 Qwen3-Omni 的交错分配，三个分量都覆盖从高频到低频的整个范围。*

- **Qwen2-VL**：配置里 `mrope_section = [16, 24, 24]`（以基于 Qwen2-VL-2B 的开源 checkpoint 的配置核对，head_dim = 128，16 + 24 + 24 = 64），按连续分块分配，时间分量拿到最快的一段。
- **Qwen3-Omni**：论文说明旋转角度按 24、20、20 的数量交错分配给时间、高度、宽度，并指出前一代 M-RoPE 把最前面的 16 个角度给了时间；改成交错的目的，是兼顾局部语义和长程依赖。

用代码表达这件事，只需要回答"每个频率归哪个分量"：

```python
def freq_owner(section, interleaved: bool) -> torch.Tensor:
    """返回长度 d/2 的张量，第 i 个频率使用哪个位置分量 (t=0, h=1, w=2)"""
    n = sum(section)
    if not interleaved:  # Qwen2-VL：连续分块 [TTT..HHH..WWW]
        return torch.repeat_interleave(torch.arange(3), torch.tensor(section))
    owner = torch.zeros(n, dtype=torch.long)  # Qwen3 系：交错 [THWTHW...TT]
    owner[1 : 3 * section[1] : 3] = 1
    owner[2 : 3 * section[2] : 3] = 2
    return owner


def apply_mrope(x, pos3, section, interleaved=False, base=10000.0):
    """x: [L, d]，pos3: [3, L]，三行依次是 t、h、w 的位置"""
    L, d = x.shape
    freqs = rope_freqs(d, base)  # [d/2]
    ang = pos3[:, :, None] * freqs  # [3, L, d/2]：三个分量各算一遍
    owner = freq_owner(section, interleaved).view(1, 1, -1).expand(1, L, -1)
    ang = torch.gather(ang, 0, owner)[0]  # [L, d/2]：每个频率只取归属分量的角度
    ang = torch.cat([ang, ang], dim=-1)
    return x * ang.cos() + rotate_half(x) * ang.sin()
```

- `ang` 先把 t、h、w 三个位置各自乘上全部频率，得到三份角度表。
- `freq_owner` 决定每个频率该用哪一份：连续分块是前 16 个给 t，接着 24 个给 h，最后 24 个给 w；交错是 t、h、w 轮流，余下的频率归 t（24 = 20 + 4，所以末尾多出 4 个 t）。
- `torch.gather` 按归属表把每个频率挑出对应分量的角度，之后和普通 RoPE 完全相同。

验证：

```python
for sec, il in (([16, 24, 24], False), ([24, 20, 20], True)):
    o = freq_owner(sec, il)
    print("".join("THW"[i] for i in o.tolist()))

# 文本三个分量相同 → 与普通 RoPE 完全一致
xx, p = torch.randn(6, 128), torch.arange(6.0)
print(torch.allclose(apply_mrope(xx, torch.stack([p, p, p]), [24, 20, 20], True),
                     apply_rope_half(xx, p)))
```

```
TTTTTTTTTTTTTTTTHHHHHHHHHHHHHHHHHHHHHHHHWWWWWWWWWWWWWWWWWWWWWWWW
THWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTHWTTTT
True
```

再验证"三个分量各管各的"：只给两个 token 的 t 同时加 25，分数不变；让 t 相差 25，分数才改变。

```python
torch.manual_seed(1)
q1, k1 = torch.randn(1, 128), torch.randn(1, 128)

def sc(pq, pk):  # pq、pk 是 (t, h, w)
    a = apply_mrope(q1, torch.tensor(pq).view(3, 1).double(), [24, 20, 20], True)
    b = apply_mrope(k1, torch.tensor(pk).view(3, 1).double(), [24, 20, 20], True)
    return float(a @ b.T)

print(f"{sc([10, 2, 3], [4, 1, 1]):.6f}")   # t 相差 6
print(f"{sc([35, 2, 3], [29, 1, 1]):.6f}")  # t 同时 +25，差仍是 6
print(f"{sc([35, 2, 3], [4, 1, 1]):.6f}")   # t 相差 31
```

```
-7.656143
-7.656143
-24.134090
```

两点实现说明：交错部分按 NVIDIA Megatron-Bridge 文档对 Qwen3-VL 的描述实现（T、H、W 交替，余下的 T 排在末尾），具体下标写法以官方仓库为准；跨模态拼接时 h、w 的偏移在本文代码里省略。

## 8. 音频与音视频对齐

### 音频里的三种 Hz

音频涉及几个容易混淆的"频率"：

| 阶段 | 时间粒度 | 频率 | 来源 |
|---|---|---|---|
| 梅尔频谱帧 | 25 ms 窗，10 ms 帧移 | 100 Hz | Qwen2.5-Omni 论文 |
| 进 LLM 的音频 token（Qwen2.5-Omni） | 约 40 ms | 25 Hz | 同上，论文未展开中间的下采样细节 |
| 进 LLM 的音频 token（Qwen3-Omni，AuT 编码器） | 约 80 ms | 12.5 Hz | Qwen3-Omni 论文，编码器 8 倍下采样 |

token 率直接决定序列长度：25 Hz 时 1 分钟音频是 1500 个 token，12.5 Hz 时是 750 个。

另外两个"频率"和它们无关：声音本身的频率（赫兹，音高）是波形里的振动快慢；RoPE 的 θ 是"每移动一个位置转多少弧度"，只是名字里都带"频率"。

### 时间 ID 绑定真实时间

TM-RoPE 里音频 token 的时间 ID，由真实时间换算：Qwen2.5-Omni 里 1 个时间 ID 对应 40 ms，Qwen3-Omni 里对应 80 ms。视频帧的时间 ID 也按它的真实时间换算，抽帧率变了，帧与帧之间的 ID 差随之变化，但每个 ID 始终代表同样长的真实时间。

![图 4：音频与视频在同一条时间轴上](/img/posts/rope-to-tm-rope/04-av-timeline.svg)

*图 4：以 1 个 ID = 40 ms 为例，视频帧落在 ID 0、25、50、75，音频 token 连续占用 ID 0 到 99。2 秒处的视频帧和第 50 个音频 token 的时间 ID 相同，在注意力里的时间相对位置为零。*

### 交错排列

Qwen2.5-Omni 把带音频的视频每 2 秒切成一块，块内视频在前、音频在后，块与块依次排列。Qwen3-Omni 沿用分块流式处理的机制，论文没有给出具体的块长，对齐主要靠绝对时间的 ID。

两层机制各有分工：交错排列让时间上相邻的内容在序列里物理上也相邻；时间 ID 提供精确的时间标记。

![图 5：交错序列与各模态的位置 ID](/img/posts/rope-to-tm-rope/05-sequence-position-ids.svg)

*图 5：序列按 2 秒块排列；文本、音频、视频 token 分别携带 (p, p, p)、(t, t, t)、(t, h, w)。*

用代码把这个玩具例子生成出来（1 ID = 40 ms，每 2 秒一块，视频 1 帧/秒，每帧 2×2 个 patch）：

```python
def build_av_position_ids(num_chunks=2, chunk_s=2.0, fps=1.0, grid=(2, 2), ms_per_id=40):
    """返回 [(模态, t, h, w), ...]；每个 2 秒块内视频在前、音频在后"""
    toks = []
    for c in range(num_chunks):
        t0 = round(c * chunk_s * 1000 / ms_per_id)  # 块起点对应的时间 ID
        for k in range(int(chunk_s * fps)):
            t = t0 + round(k / fps * 1000 / ms_per_id)  # 帧的真实时间 → 时间 ID
            for r in range(grid[0]):
                for col in range(grid[1]):
                    toks.append(("V", t, r, col))
        for j in range(int(chunk_s * 1000 / ms_per_id)):
            toks.append(("A", t0 + j, t0 + j, t0 + j))  # 音频：三个分量相同
    return toks
```

- 视频帧的时间 ID 来自"真实时间 ÷ 40 ms"，而不是帧序号，改 `fps` 后 ID 间隔自动变化。
- 音频 token 每个占 1 个 ID，所以一个 2 秒块有 50 个音频 token。
- 这里 h、w 从 0 开始，省略了跨模态拼接的偏移，真实实现以论文和官方代码为准。

检查一下生成结果：

```python
toks = build_av_position_ids()
print(len(toks), sum(t[0] == "V" for t in toks), sum(t[0] == "A" for t in toks))
print(toks[:6])
v = next(t for t in toks if t[0] == "V" and t[1] == 50)  # 2.0 s 处的第一个视频 patch
a = next(t for t in toks if t[0] == "A" and t[1] == 50)  # 2.0 s 处的音频 token
print(v, a)
print(sorted({t[1] for t in toks if t[0] == "V"}))      # 视频帧的时间 ID
print(min(t[1] for t in toks if t[0] == "A"), max(t[1] for t in toks if t[0] == "A"))
```

```
116 16 100
[('V', 0, 0, 0), ('V', 0, 0, 1), ('V', 0, 1, 0), ('V', 0, 1, 1), ('V', 25, 0, 0), ('V', 25, 0, 1)]
('V', 50, 0, 0) ('A', 50, 50, 50)
[0, 25, 50, 75]
0 99
```

2.0 秒处的视频 patch 和音频 token，时间 ID 都是 50。在注意力里，t 对应的那组频率块上，它们的相位差为零，模型因此能把"这句话"和"这个画面"对上。

关于"时间戳"：两篇 Omni 论文我核对过的部分，都没有描述往序列里插入显式的时间戳 token，时间对齐只通过位置 ID 实现。

## 9. 小结与常见误区

整条线索只有一句话：**RoPE 用旋转让注意力只看相对位置，TM-RoPE 把"位置"换成 (时间, 高, 宽)，并让时间绑定真实时间。**

| 常见误区 | 实际情况 |
|---|---|
| θ 是训练出来的 | θ 由 base 和 head_dim 算出，是常数，训练中不更新 |
| RoPE 作用在 Q、K、V 上 | 只作用在 Q 和 K，V 不旋转 |
| h、w 是像素坐标 | 是 patch 在画面里的行号、列号 |
| 时间 ID 就是 token 序号 | 对音视频，是真实时间换算出来的 ID；文本才是序号 |
| 三个分量各占连续一段频率 | Qwen2-VL 是连续分块（时间拿最快的一段）；Qwen3-Omni 是交错分配，三者都覆盖整个频率范围 |
| 音频 Hz 和 RoPE 的 θ 是一回事 | 前者是时间分辨率（每秒多少 token），后者是每个位置转多少弧度 |

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)（RoPE 原论文）
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)（M-RoPE）
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)（TMRoPE、40 ms 音频 token、2 秒交错）
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)（AuT 12.5 Hz、80 ms 时间 ID、24/20/20 交错分配）
- [NVIDIA Megatron-Bridge：Qwen3-VL RoPE 文档](https://docs.nvidia.com/nemo/megatron-bridge/nightly/apidocs/bridge/bridge.models.qwen_vl.modelling_qwen3_vl.rope.html)（交错 MRoPE 的布局说明）
