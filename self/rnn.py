"""循环神经网络（RNN）：自实现的 RNNCell + 多层 RNN。

完全复用 ``core`` 中的 ``Linear`` / ``ModuleList`` / ``tanh`` 与自动求导，
逐时间步、逐层展开计算。
"""

from .core import nn
from .core.nn import Module, Linear, ModuleList
from .core.tensor import Tensor, zeros


class RNNCell(Module):
    """单步 RNN 单元：h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)。"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.ih = Linear(input_size, hidden_size)   # 当前输入 -> 隐层
        self.hh = Linear(hidden_size, hidden_size)  # 上一隐层 -> 隐层

    def forward(self, x, h):
        return (self.ih(x) + self.hh(h)).tanh()


class RNN(Module):
    """多层 RNN。输入 (B, T, D)，返回最后一层最终时刻的隐状态（或分类 logits）。"""

    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = ModuleList([RNNCell(sizes[i], hidden_size)
                                 for i in range(num_layers)])
        self.fc = Linear(hidden_size, num_classes) if num_classes else None

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        if h0 is None:
            h = [Tensor(zeros([B, self.hidden_size]))
                 for _ in range(self.num_layers)]
        else:
            h = h0
        for t in range(T):
            xt = x.select(1, t)                 # 取出第 t 个时间步 (B, D)
            for l in range(self.num_layers):
                h[l] = self.cells[l](xt if l == 0 else h[l - 1], h[l])
        out = h[-1]
        if self.fc is not None:
            out = self.fc(out)
        return out
