"""Transformer 编码器（自实现）：多头注意力 + 前馈网络 + 层归一化。

注意力通过 reshape/transpose 把 (B, T, D) 重排为 (B*H, T, d_k)，
复用批量矩阵乘与 softmax；位置编码为无参数的正弦编码。整体复用了
``Embedding`` / ``Linear`` / ``LayerNorm`` / ``Dropout`` / ``ModuleList``。
"""

import math

from .core import nn
from .core.nn import Module, Linear, Embedding, LayerNorm, Dropout, ModuleList
from .core.tensor import Tensor, zeros


class MultiHeadAttention(Module):
    """缩放点积多头注意力。支持可选的加性掩码（掩码为 0 / -inf 的 list）。"""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被头数整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, D = q.shape
        Tk = k.shape[1]                          # 交叉注意力时 k/v 长度可与 q 不同
        H, dk = self.n_heads, self.d_k
        # 分头：reshape 出 (B, T, H, dk) 后交换 T 与 H 维度
        q = self.Wq(q).reshape(B, Tq, H, dk).transpose(1, 2)   # (B, H, Tq, dk)
        k = self.Wk(k).reshape(B, Tk, H, dk).transpose(1, 2)
        v = self.Wv(v).reshape(B, Tk, H, dk).transpose(1, 2)

        q2 = q.reshape(B * H, Tq, dk)                          # 合并 batch 与 head
        k2 = k.reshape(B * H, Tk, dk)
        v2 = v.reshape(B * H, Tk, dk)

        scores = q2.matmul(k2.transpose(1, 2)) / math.sqrt(dk)  # (B*H, Tq, Tk)
        if mask is not None:
            scores = scores + mask
        attn = self.dropout(scores.softmax(-1))
        out = attn.matmul(v2)                                   # (B*H, Tq, dk)
        out = out.reshape(B, H, Tq, dk).transpose(1, 2).reshape(B, Tq, D)
        return self.Wo(out)


class FeedForward(Module):
    """逐位置前馈网络：Linear -> ReLU -> Dropout -> Linear。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x).relu()))


class TransformerEncoderBlock(Module):
    """一个编码器块：残差 + 层归一化 包裹 注意力 与 前馈 两个子层。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop1(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class PositionalEncoding(Module):
    """正弦位置编码（无参数），直接加到嵌入向量上。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = zeros([max_len, d_model])
        for pos in range(max_len):
            for j in range(0, d_model, 2):
                w = 10000.0 ** (j / d_model)
                pe[pos][j] = math.sin(pos / w)
                if j + 1 < d_model:
                    pe[pos][j + 1] = math.cos(pos / w)
        self.pe = pe

    def forward(self, x):
        T = x.shape[1]
        return x + Tensor(self.pe[:T])


class Transformer(Module):
    """用于序列分类的 Transformer 编码器（均值池化后接分类头）。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.blocks = ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                                  for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        self.fc = Linear(d_model, num_classes)

    def forward(self, idx, mask=None):
        x = self.embed(idx)                # (B, T, D)
        x = self.pe(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        x = x.mean(1)                      # 时间维均值池化 -> (B, D)
        return self.fc(x)
