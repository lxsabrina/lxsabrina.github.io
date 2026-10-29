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

本文从"注意力为什么不认识顺序"出发，推导 RoPE，用 d = 4 的例子手算验证，再用 PyTorch 实现并跑通；然后讲它怎样扩展成 (时间, 高, 宽) 三个分量。音频这条线则从最底层讲起：声音怎样变成采样点、窗口、FFT、Mel 频带和特征帧，最后讲音频和视频怎样在时间轴上对齐。文中所有数值都是代码实际运行的结果。

| # | 问题 |
|---|---|
| Q1 | 位置信息为什么必须是"相对"的？公式里的 q、k、f、g 是什么？ |
| Q2 | 为什么旋转能让绝对位置抵消，只剩相对距离？ |
| Q3 | RoPE 的完整公式是什么？θ 从哪来，要训练吗？ |
| Q4 | 怎么用 PyTorch 实现和验证？训练时怎么用？ |
| Q5 | TM-RoPE 在 RoPE 上改了什么？ |
| Q6 | 声音怎样变成梅尔频谱的特征帧？采样点、窗口、FFT、Mel 频带各是什么？ |
| Q7 | 音频和视频怎样在时间轴上对齐？ |

## 核心结论

- **注意力本身不认识顺序**。要注入位置，并且希望打分只依赖两个 token 的相对距离 m − n。
- **RoPE 的做法是旋转**：把 q、k 的相邻两维配成一对，按"位置 × θ"的角度旋转。点积时两个角度相减，m 和 n 各自消失，只剩 n − m。
- **θ 是常数，不是训练参数**。θᵢ 由 base 和 head_dim 算出，多个频率像钟表的秒针和分针，分别负责近距离和远距离。
- **RoPE 只要求"位置是一个数"**。M-RoPE 和 TM-RoPE 把这个数换成 (t, h, w)，三个分量各占一部分频率，旋转公式不变。
- **音频先变成梅尔频谱，再变成 token**。声音被量成采样点（16 kHz 下每秒 16000 个数）；每 10 ms 取一个 25 ms 的窗口；FFT 用"和各频率的正弦做点积"算出每个频率的强度；再按人耳的尺度合并成 128 个频带，得到一个 128 维的特征帧。
- **音视频对齐靠真实时间**。音频每个 token 对应固定时长（Qwen2.5-Omni 约 40 ms，Qwen3-Omni 约 80 ms，Qwen3.5-Omni 160 ms），时间 ID 绑定真实时间；视频帧按真实时间戳换算时间 ID，同一帧的 patch 共用 t，用 (h, w) 区分位置。
- **Qwen3.5-Omni 额外加了显式时间戳**：在视频或音视频时间块前插入以秒为单位的文本时间戳，音频里随机间隔插入。前两代论文里我核对过的部分没有这一步。

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

## 8. 声音是怎样变成特征帧的

TM-RoPE 里的"音频 token"不是凭空出现的。进 LLM 之前，声音要经过四步：**采样点 → 窗口 → FFT → Mel 频带**，最后得到一串特征帧，再由编码器压缩成 token。

### 采样点：声音是一串数

麦克风把空气振动量成电压，每隔固定时间记一个数，这个数就是一个**采样点**。每秒记多少个，叫采样率。Qwen-Omni 把音频统一重采样到 16 kHz（Qwen3.5-Omni 报告写明），即每秒 16000 个数。

- 单个采样点只是某一瞬间的振幅，没有"频率"可言。频率是一段波形里振动的快慢，至少要看一小段才能谈。
- 采样率 16 kHz 能表示的最高频率是它的一半，即 8 kHz（奈奎斯特频率）。

### 窗口与帧移：每 10 ms 看一眼 25 ms

整段音频不能一次做分析，否则只能知道"整段里有哪些频率"，不知道它们何时出现。做法是用一个 25 ms 的窗口截一小段，每次向后滑 10 ms：

| 量 | 时间 | 采样点数（16 kHz） |
|---|---|---|
| 窗口长度 | 25 ms | 400 |
| 帧移（步长） | 10 ms | 160 |
| 相邻窗口重叠 | 15 ms | 240 |

![图 4：采样点与滑动窗口](/img/posts/rope-to-tm-rope/04-sampling-and-windows.svg)

*图 4：上图是波形上的采样点；下方是依次滑动的 25 ms 窗口，每个窗口产出一个特征帧。窗口之间重叠 15 ms，所以特征帧是每秒 100 个。*

1 秒音频（16000 个点）共能取到 (16000 − 400) / 160 + 1 = 98 个完整窗口。

### FFT 的原理：用点积探测频率

每个窗口里是 400 个数，要把它变成"各个频率上有多强"。核心想法是**点积探测**：想知道信号里有没有 5 个周期的成分，就拿一个 5 个周期的余弦和正弦去和信号逐点相乘再求和。

- 信号里确实有这个频率：同相位相乘，结果恒为正，累加后很大。
- 信号里没有：正负相互抵消，累加后接近 0。
- 不同的整数频率之间互相正交，所以每个频率的探测互不干扰。

![图 5：点积探测频率](/img/posts/rope-to-tm-rope/05-dot-product-probe.svg)

*图 5：左边探测频率与信号一致，乘积全为正，均值 0.50；右边探测频率不一致，乘积正负抵消，均值 0.00。*

用 PyTorch 直接实现这个"点积"版本的傅里叶变换，并和 `torch.fft.rfft` 对比：

```python
import math
import torch

torch.set_default_dtype(torch.float64)

def dft_by_dot(x):
    N = len(x); n = torch.arange(N); out = []
    for k in range(N // 2 + 1):                                  # 探测 0 .. N/2 个周期
        c = (x * torch.cos(2 * math.pi * k * n / N)).sum()       # 和余弦做点积
        s = (x * torch.sin(2 * math.pi * k * n / N)).sum()       # 和正弦做点积
        out.append(complex(c, -s))                               # 两个点积合成一个复数
    return torch.tensor(out)
```

- `k` 是整段窗口里的周期数，对应频率 k·sr/N。
- 余弦和正弦各做一次点积，是因为信号的相位未知：同一个频率，相位不同时，只和余弦的点积会漏掉信号。两个结果合成复数，模长才是这个频率的强度。

先用 8 个点的小例子检查：信号由 1 个周期的余弦（幅度 2）和 3 个周期的正弦（幅度 1）叠成。

```python
n = torch.arange(8)
x = 2 * torch.cos(2 * math.pi * 1 * n / 8) + torch.sin(2 * math.pi * 3 * n / 8)
X = dft_by_dot(x)
print(x.numpy().round(3))
print(X.abs().numpy().round(3), (2 * X.abs() / 8).numpy().round(3))
print(torch.allclose(X, torch.fft.rfft(x)))
```

```
[ 2.     2.121 -1.    -0.707 -2.    -2.121  1.     0.707]
[0. 8. 0. 4. 0.] [0. 2. 0. 1. 0.]
True
```

第 1 个和第 3 个频率点有强度，模长 8 和 4，乘 2/N 后恰好还原出幅度 2 和 1，其余为 0。结果与 `torch.fft.rfft` 完全一致。

再换成真实的窗口规模：400 个点，里面是 200 Hz（幅度 1.0）加 1000 Hz（幅度 0.5）。

```python
sr, N = 16000, 400
t = torch.arange(N) / sr
w = 1.0 * torch.sin(2 * math.pi * 200 * t) + 0.5 * torch.sin(2 * math.pi * 1000 * t)
amp = 2 * torch.fft.rfft(w).abs() / N
for k in amp.topk(2).indices.sort().values.tolist():
    print(f"bin {k}: {k * sr / N:.0f} Hz, 幅度 {amp[k]:.3f}")
print(len(amp), sr / N)
```

```
bin 5: 200 Hz, 幅度 1.000
bin 25: 1000 Hz, 幅度 0.500
201 40.0
```

400 个点的实数 FFT 得到 201 个频率点，间隔 sr/N = 40 Hz。两个频率分别落在第 5 和第 25 个点上，强度正好是 1.0 和 0.5。

- **"FFT"只是更快的算法**：上面的写法每个频率都要扫一遍信号，总计算量约 N² = 160000 次乘加；FFT 利用对称性，把它降到约 N·log N，几千次。结果是同一个。
- **窗函数**：如果信号的周期数不是整数（比如 210 Hz 在 400 点里是 5.25 个周期），切出来的片段首尾对不上，能量会泄漏到别的频率点。窗口乘上一条两头为 0 的 Hann 曲线，就能让泄漏小得多。在这个 210 Hz 的例子里，远离主峰的频率点相对强度，不加窗约为 0.4%，加 Hann 窗后降到约 3×10⁻⁷。

### Mel：按人耳的尺度合并频率

FFT 给出 201 个均匀分布的频率点，但人耳对低频更敏感：100 Hz 和 200 Hz 听起来差很多，7000 Hz 和 7100 Hz 几乎一样。所以要把 201 个点合并成 128 个**Mel 频带**：低频处频带窄、分得细，高频处频带宽、分得粗。

每个频带是一个三角形的权重：中心权重最大，向两侧线性降到 0。三角形的边界在 Mel 刻度上等间距，换回赫兹后就变成了"低频窄、高频宽"。这里用的是 Slaney 刻度（1000 Hz 以下线性，以上取对数），也是 librosa 的默认值。

![图 6：128 个 Mel 频带](/img/posts/rope-to-tm-rope/06-mel-filterbank.svg)

*图 6：128 个三角形滤波器。低频处三角形窄而密，高频处宽而疏；下方是各频带的宽度随频率的变化。*

在 16 kHz、400 点窗口下，几个频带的宽度是：

| 频带序号 | 中心频率 | 宽度 |
|---|---|---|
| 第 1–32 个 | 23 Hz–748 Hz | 46.8 Hz（相同） |
| 第 45 个 | 1055 Hz | 50.9 Hz |
| 第 64 个 | 1669 Hz | 80.5 Hz |
| 第 96 个 | 3610 Hz | 174.1 Hz |
| 第 128 个 | 7809 Hz | 376.7 Hz |

中心频率低于 1000 Hz 的有 42 个，1000–4000 Hz 的有 58 个，4000 Hz 以上的有 28 个。

完整的 Mel 频谱实现只需要两个函数：

```python
def mel_filterbank(sr=16000, n_fft=400, n_mels=128):
    """Slaney 刻度的三角形 Mel 滤波器组，返回 [n_mels, n_fft//2+1]"""
    f_sp, logstep = 200.0 / 3, math.log(6.4) / 27
    to_mel = lambda f: torch.where(f < 1000, f / f_sp, 15 + torch.log(f.clamp_min(1000) / 1000) / logstep)
    to_hz = lambda m: torch.where(m < 15, f_sp * m, 1000 * torch.exp(logstep * (m - 15)))
    bins = torch.linspace(0, sr / 2, n_fft // 2 + 1)                      # 201 个 FFT 频率点
    top = float(to_mel(torch.tensor(sr / 2)))
    edges = to_hz(torch.linspace(0.0, top, n_mels + 2))                   # n_mels+2 个边界（Mel 刻度上等距）
    l, c, r = edges[:-2, None], edges[1:-1, None], edges[2:, None]        # 每个频带的左边界、中心、右边界
    tri = torch.minimum((bins - l) / (c - l), (r - bins) / (r - c)).clamp_min(0)
    return tri * (2.0 / (r - l))                                          # Slaney 归一化：各频带面积相同


def mel_spectrogram(wave, sr=16000, win=400, hop=160, n_mels=128):
    """wave: [num_samples] → [T, n_mels]，每一行是一个特征帧"""
    window = torch.hann_window(win, periodic=True)
    frames = wave.unfold(0, win, hop)                                     # [T, 400]：窗口 25 ms，步长 10 ms
    spec = torch.fft.rfft(frames * window, dim=-1).abs() ** 2             # [T, 201]：每个频率点的能量
    return spec @ mel_filterbank(sr, win, n_mels).T                       # [T, 128]：合并成 128 个 Mel 频带
```

- `unfold(0, win, hop)` 一步完成"滑动窗口"：每隔 160 个点取 400 个点，得到 `[T, 400]`。
- 每一帧先乘 Hann 窗，再做 `rfft`，取模的平方得到每个频率点的能量，`[T, 201]`。
- 最后一步是一个矩阵乘法：`[T, 201] @ [201, 128]`，把 201 个频率点按三角形权重合并成 128 个频带。

把前面那个 200 Hz + 1000 Hz 的信号持续 1 秒，送进去看看，再和 librosa 对照：

```python
t = torch.arange(16000) / 16000
wave = 1.0 * torch.sin(2 * math.pi * 200 * t) + 0.5 * torch.sin(2 * math.pi * 1000 * t)
M = mel_spectrogram(wave)
print(tuple(M.shape), tuple(mel_filterbank().shape))
print(M.mean(0).topk(2).indices.sort().values + 1)   # 平均能量最大的两个频带（序号从 1 起）
```

```
(98, 128) (128, 201)
tensor([8, 9])
```

1 秒音频得到 98 帧、每帧 128 维。200 Hz 恰好落在第 8 个（中心 187 Hz）和第 9 个（中心 210 Hz）频带之间，能量被这两个频带分摊。我另外拿 librosa（`n_fft=400, hop_length=160, center=False, n_mels=128`）做了对照，两者最大相对误差约 8×10⁻⁸，滤波器组形状为 (128, 201)，没有全零频带。

### 梅尔频谱帧：一帧就是一个 128 维向量

把所有窗口的结果竖着叠起来，就是**梅尔频谱**，形状 `[T, 128]`：横轴是时间，纵轴是 128 个频带，颜色是能量。其中**一列、也就是一帧，是一个 128 维的向量**，描述这 10 ms 左右的声音里各个频带有多强。

![图 7：梅尔频谱与特征帧](/img/posts/rope-to-tm-rope/07-mel-spectrogram-frames.svg)

*图 7：左边是梅尔频谱，竖直的一列就是一帧；右边把其中一帧展开成 128 维向量。*

每秒的数据量是这样变化的：

| 阶段 | 每秒的量 |
|---|---|
| 采样点 | 16000 个数 |
| 梅尔频谱帧 | 100 帧 × 128 = 12800 个数 |
| 进 LLM 的音频 token（Qwen2.5-Omni，25 Hz） | 25 个 |
| 进 LLM 的音频 token（Qwen3-Omni，12.5 Hz） | 12.5 个 |
| 进 LLM 的音频 token（Qwen3.5-Omni，6.25 Hz） | 6.25 个 |

要注意：**特征帧不是 LLM 的 token**。100 Hz 的帧要再经过音频编码器（Qwen3-Omni 的 AuT 里有 8 倍下采样的卷积）压缩成更少的向量，才是 TM-RoPE 里那个"音频 token"。实际模型在取对数、归一化上还有细节，这里不展开，也没有逐项核对论文。

> 一个类比，仅用于帮助理解，不是论文里的说法：DFT 里的 e^(−iωn) 和 RoPE 里的 e^(imθ) 是同一种数学对象，都是复平面上的旋转。但它们用途完全不同：前者在探测信号里有什么频率，后者在给 token 编位置，RoPE 的 θ 和音频的赫兹没有关系。

## 9. 音视频对齐与视觉的处理

### 先分清几个"频率"

| 名称 | 含义 | 例子 |
|---|---|---|
| 声音频率 | 波形振动的快慢（赫兹），决定音高 | 200 Hz、1000 Hz |
| 特征帧率 | 每秒多少个梅尔频谱帧 | 100 Hz（25 ms 窗，10 ms 帧移） |
| token 率 | 每秒多少个音频 token，决定序列长度 | 25 Hz（Qwen2.5-Omni）、12.5 Hz（Qwen3-Omni）、6.25 Hz（Qwen3.5-Omni） |
| RoPE 的 θ | 每移动一个位置转多少弧度 | 与前三者无关 |

token 率直接决定序列长度：25 Hz 时 1 分钟音频是 1500 个 token，12.5 Hz 时是 750 个，6.25 Hz 时是 375 个。

### 时间 ID 绑定真实时间

TM-RoPE 里音频 token 的时间 ID，由真实时间换算：Qwen2.5-Omni 里 1 个时间 ID 对应 40 ms，Qwen3-Omni 里对应 80 ms，Qwen3.5-Omni 里对应 160 ms。视频帧的时间 ID 也按它的真实时间换算，抽帧率变了，帧与帧之间的 ID 差随之变化，但每个 ID 始终代表同样长的真实时间。

![图 8：音频与视频在同一条时间轴上](/img/posts/rope-to-tm-rope/08-av-timeline.svg)

*图 8：以 1 个 ID = 40 ms 为例，视频帧落在 ID 0、25、50、75，音频 token 连续占用 ID 0 到 99。2 秒处的视频帧和第 50 个音频 token 的时间 ID 相同，在注意力里的时间相对位置为零。*

### 交错排列

Qwen2.5-Omni 把带音频的视频每 2 秒切成一块，块内视频在前、音频在后，块与块依次排列。Qwen3-Omni 沿用分块流式处理的机制，论文没有给出具体的块长，对齐主要靠绝对时间的 ID。

两层机制各有分工：交错排列让时间上相邻的内容在序列里物理上也相邻；时间 ID 提供精确的时间标记。

![图 9：交错序列与各模态的位置 ID](/img/posts/rope-to-tm-rope/09-sequence-position-ids.svg)

*图 9：序列按 2 秒块排列；文本、音频、视频 token 分别携带 (p, p, p)、(t, t, t)、(t, h, w)。*

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

Qwen2.5-Omni 和 Qwen3-Omni 的论文里，我核对过的部分都没有往序列里插入显式的时间戳，对齐只靠位置 ID。Qwen3.5-Omni 在这一点上做了改动，见下一节。

### 视觉是怎样跑进来的

图像和视频先由视觉编码器切成 patch 并编码，再给每个 patch 编上 (t, h, w) 三个位置，一路与音频、文本拼进同一条序列。以视频为例：

1. **抽帧**：按一定帧率从视频里取帧，每帧记下它在视频里的真实时间（秒）。Qwen3.5-Omni 报告说明抽帧采用动态帧率，目的是尽量保留视频信息，同时与音频流保持对齐。
2. **编码成 patch**：视觉编码器（Qwen3.5-Omni 报告写的是沿用 Qwen3.5 的视觉编码器，表 1 里标为 SigLIP2）把每帧切成网格，每个格子是一个 patch 向量。patch 的边长、相邻 patch 的合并方式，报告里没有给出；Qwen2-VL 的做法是 14 像素的 patch，再把相邻 2×2 个合并成 1 个 token，视频里每 2 帧合成 1 个时间单元。这些数字属于 Qwen2-VL，不要当成 Qwen3.5-Omni 的设置。
3. **编位置**：同一帧的所有 patch 共用同一个 t，t 由这帧的真实时间换算；h、w 是 patch 的行号和列号，规则和静态图片相同。
4. **拼接**：多个模态依次拼接时，位置编号保持连续，后一个模态从前一个模态的最大位置 ID 加 1 开始。

用一段玩具代码把第 3、4 步写出来（1 个 ID = 160 ms，网格 2×2；行号列号的起点和拼接偏移是我为演示做的简化，真实实现以官方代码为准）：

```python
def build_video_position_ids(timestamps, grid=(2, 2), ms_per_id=160, start=0):
    """视频每帧的 (t, h, w)：t 由真实时间换算，h/w 是 patch 行列号；返回 (位置列表, 下一模态的起点)"""
    pos = []
    for ts in timestamps:
        t = start + round(ts * 1000 / ms_per_id)
        for r in range(grid[0]):
            for c in range(grid[1]):
                pos.append((t, start + r, start + c))
    nxt = max(max(p) for p in pos) + 1
    return pos, nxt

pos, nxt = build_video_position_ids([0.0, 0.5, 1.5, 3.0])     # 动态抽帧：间隔不均匀
print([p for p in pos if p[1:] == (0, 0)])
print(sorted({p[0] for p in pos}), nxt)
pos2, nxt2 = build_video_position_ids([0.0, 0.5], start=nxt)   # 接在后面的下一个模态
print(pos2[0], pos2[-1], nxt2)
```

```
[(0, 0, 0), (3, 0, 0), (9, 0, 0), (19, 0, 0)]
[0, 3, 9, 19] 20
(20, 20, 20) (23, 21, 21) 24
```

- 帧在 0、0.5、1.5、3.0 秒，换算成时间 ID 是 0、3、9、19。**间隔不均匀的抽帧，ID 间隔跟着真实时间走**，所以不要求训练数据在各种帧率上都均匀分布。
- 同一帧的 4 个 patch 共用 t，用 h、w 区分位置。
- 前一段视频最大的 ID 是 19，所以下一段从 20 开始，序号是连续的，没有重叠。

### Qwen3.5-Omni：加上显式时间戳

Qwen3.5-Omni 报告指出，纯靠绝对时间的 ID 有两个问题：长视频带音频时，视觉 patch 的时间 ID 会变得"过于稀疏"，削弱长程时间建模；并且需要在各种帧率上都有大规模、均匀分布的训练数据，数据构造成本高。

所以它在 TM-RoPE 之外，**给每个视频或音视频的时间块前面，加一段以秒为单位的文本时间戳**（格式化的文本字符串）；音频序列里也按随机间隔插入时间戳。代价是上下文略长，好处是时间感知更精确，长上下文外推更稳。

和位置 ID 部分合起来看：

| 项 | 做法（来源：Qwen3.5-Omni 报告） |
|---|---|
| 音频时间 ID | 每 160 ms 一个 ID |
| 视频时间 ID | 按真实时间戳换算，统一到每 ID 160 ms |
| 视频 h、w | 与静态图片相同 |
| 多模态拼接 | 位置连续，后一模态从前一模态最大 ID + 1 起 |
| 显式时间戳 | 视频或音视频时间块前加文本时间戳；音频随机间隔插入 |
| 对齐方式 | 靠时间 ID 锚定到绝对时间，因此可以支持任意时长的流式输入 |

可以这样理解两层分工：**位置 ID 让注意力在相位上"知道"两个 token 差多少时间；文本时间戳让模型能直接"读到"现在是第几秒**，两者互补。


## 10. 小结与常见误区

整条线索只有一句话：**RoPE 用旋转让注意力只看相对位置，TM-RoPE 把"位置"换成 (时间, 高, 宽)，并让时间绑定真实时间。**

| 常见误区 | 实际情况 |
|---|---|
| θ 是训练出来的 | θ 由 base 和 head_dim 算出，是常数，训练中不更新 |
| RoPE 作用在 Q、K、V 上 | 只作用在 Q 和 K，V 不旋转 |
| h、w 是像素坐标 | 是 patch 在画面里的行号、列号 |
| 时间 ID 就是 token 序号 | 对音视频，是真实时间换算出来的 ID；文本才是序号 |
| 三个分量各占连续一段频率 | Qwen2-VL 是连续分块（时间拿最快的一段）；Qwen3-Omni 是交错分配，三者都覆盖整个频率范围 |
| 音频 Hz 和 RoPE 的 θ 是一回事 | 前者是时间分辨率（每秒多少 token），后者是每个位置转多少弧度 |
| 单个采样点有频率 | 频率是一段波形的振动快慢，至少要看一个窗口 |
| Mel 频带是等宽的 | 低频窄、高频宽：16 kHz 下约 47 Hz 到 377 Hz 不等（Slaney 刻度） |
| 一个特征帧就是一个 LLM token | 特征帧 100 Hz，要经过音频编码器压缩成 25、12.5 或 6.25 Hz 的 token |
| 带音频的视频只靠位置 ID 对齐 | Qwen2.5/3-Omni 是这样；Qwen3.5-Omni 还加了文本时间戳 |

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)（RoPE 原论文）
- [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)（M-RoPE）
- [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215)（TMRoPE、40 ms 音频 token、2 秒交错）
- [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)（AuT 12.5 Hz、80 ms 时间 ID、24/20/20 交错分配）
- [NVIDIA Megatron-Bridge：Qwen3-VL RoPE 文档](https://docs.nvidia.com/nemo/megatron-bridge/nightly/apidocs/bridge/bridge.models.qwen_vl.modelling_qwen3_vl.rope.html)（交错 MRoPE 的布局说明）
- [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)（160 ms 时间 ID、显式文本时间戳、位置连续拼接、16 kHz / 128 通道梅尔 / 6.25 Hz 音频 token）
