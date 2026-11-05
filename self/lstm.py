"""长短期记忆网络（LSTM）：自实现的 LSTMCell + 多层 LSTM。

门控计算复用两个 ``Linear``（分别作用在输入与隐状态上），再切分成
输入门 / 遗忘门 / 候选门 / 输出门，其余全部交给 ``core`` 的自动求导。
"""

from .core import nn
from .core.nn import Module, Linear, ModuleList
from .core.tensor import Tensor, zeros


class LSTMCell(Module):
    """单步 LSTM 单元，维护隐状态 h 与细胞状态 c。

    forget_bias：遗忘门偏置的初始值。置 1 有助于网络一开始就"记得住"，
    是训练长时记忆任务时的标准做法。
    """

    def __init__(self, input_size, hidden_size, forget_bias=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.W = Linear(input_size, 4 * hidden_size)   # 输入贡献
        self.U = Linear(hidden_size, 4 * hidden_size)  # 隐状态贡献
        if forget_bias:
            H = hidden_size
            self.W.bias.data[H:2 * H] = [forget_bias] * H   # 遗忘门在第 2 段
            self.U.bias.data[H:2 * H] = [forget_bias] * H

    def forward(self, x, h, c):
        H = self.hidden_size
        gates = self.W(x) + self.U(h)        # (B, 4H)
        i = gates.slice(1, 0, H).sigmoid()          # 输入门
        f = gates.slice(1, H, 2 * H).sigmoid()      # 遗忘门
        g = gates.slice(1, 2 * H, 3 * H).tanh()     # 候选值
        o = gates.slice(1, 3 * H, 4 * H).sigmoid()  # 输出门
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new


class LSTM(Module):
    """多层 LSTM。输入 (B, T, D)，返回最后一层最终隐状态（或分类 logits）。"""

    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=None,
                 forget_bias=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.cells = ModuleList([LSTMCell(sizes[i], hidden_size, forget_bias)
                                 for i in range(num_layers)])
        self.fc = Linear(hidden_size, num_classes) if num_classes else None

    def forward(self, x):
        B, T, _ = x.shape
        H = self.hidden_size
        h = [Tensor(zeros([B, H])) for _ in range(self.num_layers)]
        c = [Tensor(zeros([B, H])) for _ in range(self.num_layers)]
        for t in range(T):
            xt = x.select(1, t)
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](xt if l == 0 else h[l - 1], h[l], c[l])
        out = h[-1]
        if self.fc is not None:
            out = self.fc(out)
        return out
