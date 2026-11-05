"""长短期记忆网络（LSTM）：基于 PyTorch ``nn.LSTM`` 构建。

与 ``self.lstm`` 保持相同的接口（输入 (B, T, D)，返回最后时刻的隐状态或分类
logits）。``nn.LSTM`` 内部已维护隐状态 h 与细胞状态 c 的递推。
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """多层 LSTM。"""

    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) if num_classes else None

    def forward(self, x, state=None):
        # out: (B, T, H)，state: (h_n, c_n)
        out, _ = self.lstm(x, state)
        last = out[:, -1, :]                      # 取最后时间步
        return self.fc(last) if self.fc is not None else last
