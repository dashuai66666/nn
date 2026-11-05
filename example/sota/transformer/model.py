"""Transformer（Vaswani et al., 2017, "Attention Is All You Need"）复现。

论文提出的完整编码器-解码器 Transformer 是 Transformer 的"封神之作"，
完全基于注意力、抛弃循环与卷积。这里用 ``self`` 复现其核心结构：

- 缩放点积注意力 + 多头注意力（复用 ``self.transformer.MultiHeadAttention``）；
- 编码器：N 层（多头自注意力 + 前馈，残差 + 层归一化）；
- 解码器：N 层（掩码自注意力 + 编码器-解码器交叉注意力 + 前馈）；
- 正弦位置编码与共享的词嵌入结构。

训练采用 teacher forcing；推理采用贪心解码。
"""

import math

from NN.self.core.nn import Module, Linear, Embedding, LayerNorm, Dropout, ModuleList
from NN.self.transformer import (
    MultiHeadAttention,
    FeedForward,
    PositionalEncoding,
    TransformerEncoderBlock,
)


class TransformerDecoderBlock(Module):
    """解码器块：掩码自注意力 + 交叉注意力 + 前馈，各自残差 + 层归一化。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)
        self.drop3 = Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        # 1) 掩码自注意力（只看当前及之前的位置）
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        # 2) 交叉注意力：query 来自解码器，key/value 来自编码器输出
        x = self.norm2(x + self.drop2(self.cross_attn(x, memory, memory)))
        # 3) 前馈网络
        x = self.norm3(x + self.drop3(self.ff(x)))
        return x


class Transformer(Module):
    """完整编码器-解码器 Transformer，用于序列到序列任务。"""

    def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, d_ff, num_layers,
                 max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.enc_embed = Embedding(src_vocab, d_model)
        self.dec_embed = Embedding(tgt_vocab, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.encoder = ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                                   for _ in range(num_layers)])
        self.decoder = ModuleList([TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
                                   for _ in range(num_layers)])
        self.fc = Linear(d_model, tgt_vocab)     # 输出到目标词表

    def encode(self, src):
        x = self.enc_embed(src)
        x = self.pe(x)
        for block in self.encoder:
            x = block(x)
        return x

    def decode(self, tgt, memory, tgt_mask=None):
        x = self.dec_embed(tgt)
        x = self.pe(x)
        for block in self.decoder:
            x = block(x, memory, tgt_mask)
        return self.fc(x)

    def forward(self, src, tgt, tgt_mask=None):
        memory = self.encode(src)
        return self.decode(tgt, memory, tgt_mask)


def causal_mask(T):
    """生成 (T, T) 因果掩码：位置 j > i 处填 -inf，屏蔽未来信息。"""
    m = [[0.0] * T for _ in range(T)]
    for i in range(T):
        for j in range(i + 1, T):
            m[i][j] = -1e9
    return m
