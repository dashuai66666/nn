"""Transformer 编码器：基于 PyTorch ``nn.TransformerEncoder`` 构建。

结构上与 ``self.transformer`` 对应：词嵌入 + 正弦位置编码 + 多层
``TransformerEncoderLayer`` + 层归一化 + 均值池化 + 分类头。
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码，作为 buffer 注册（不参与优化）。"""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class Transformer(nn.Module):
    """用于序列分类的 Transformer 编码器。"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers,
                 num_classes, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, idx, mask=None):
        # idx: (B, T) 的整数索引；mask 为可选 padding 掩码 (B, T)
        x = self.embed(idx) * math.sqrt(self.d_model)   # 缩放嵌入
        x = self.pe(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)
        x = x.mean(dim=1)                               # 时间维均值池化 -> (B, D)
        return self.fc(x)
