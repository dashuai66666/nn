"""字符级 RNN 语言模型（Karpathy, 2015）复现。

论文/博客《The Unreasonable Effectiveness of Recurrent Neural Networks》让
"训练一个字符级 RNN，让它预测下一个字符、进而逐字生成文本"成为 RNN 的
标志性用法。这里用 ``self`` 复现：词嵌入 + 多层 RNNCell + 输出到词表。

训练时逐时间步预测下一字符；采样时以温度为参数逐字续写。
"""

from NN.self.core.nn import Module, Linear, Embedding, ModuleList
from NN.self.core.tensor import Tensor, zeros
from NN.self.rnn import RNNCell


class CharRNN(Module):
    """字符级语言模型。forward 返回每个时间步的 logits 列表。"""

    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = Embedding(vocab_size, hidden_size)
        self.cells = ModuleList([RNNCell(hidden_size, hidden_size)
                                 for _ in range(num_layers)])
        self.fc = Linear(hidden_size, vocab_size)   # 输出到词表（logits）

    def _init_state(self, batch):
        H = self.hidden_size
        return [Tensor(zeros([batch, H])) for _ in range(self.num_layers)]

    def step(self, x, h):
        """单时间步：输入字符索引 x (B,)，返回 (logits (B, vocab), 新隐状态)。"""
        xt = self.embed(x)
        for l in range(self.num_layers):
            h[l] = self.cells[l](xt if l == 0 else h[l - 1], h[l])
            xt = h[l]
        return self.fc(h[-1]), h

    def forward(self, idx):
        """idx: (B, T) 整数列表 -> [logits_t (B, vocab)] 列表，长度 T。"""
        B, T = len(idx), len(idx[0])
        h = self._init_state(B)
        logits = []
        for t in range(T):
            xt = [row[t] for row in idx]           # 第 t 列的字符索引 (B,)
            lg, h = self.step(xt, h)
            logits.append(lg)
        return logits
