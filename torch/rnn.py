"""循环神经网络（RNN）：基于 PyTorch ``nn.RNN`` 构建。

与 ``self.rnn`` 保持相同的接口（输入 (B, T, D)，返回最后时刻的隐状态或分类
logits），方便两边对照。
"""

import torch
import torch.nn as nn


class RNN(nn.Module):
    """多层 RNN。``nn.RNN`` 内置了逐时间步与逐层的循环，无需手动展开。"""

    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True 使输入/输出为 (B, T, D)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) if num_classes else None

    def forward(self, x, h0=None):
        # out: (B, T, H)，h_n: (num_layers, B, H)
        out, _ = self.rnn(x, h0)
        last = out[:, -1, :]                      # 取最后时间步
        return self.fc(last) if self.fc is not None else last
