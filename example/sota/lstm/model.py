"""长短期记忆（LSTM, Hochreiter & Schmidhuber, 1997）复现。

LSTM 的"封神"之处在于用门控 + 细胞状态解决 RNN 的长程依赖/梯度消失问题。
本示例复现原论文关心的「长时滞（long time lag）」任务：序列开头有一个关键
token，之后全是填充 token，网络要在读完整个序列后回忆出开头的 token。

注意：纯 Python 训练速度有限，示例用较短时滞（T=20）。在这一长度下，配合
自适应优化器，普通 RNN 也能学会；LSTM 门控的真正优势体现在更长的时滞上
（原论文实验了 T=50~1000）。这里的重点是忠实复现 LSTM 架构本身。

模型复用 ``self`` 的 ``Embedding`` + ``RNN``/``LSTM``，仅做薄封装。
"""

from NN.self.core.nn import Module, Embedding
from NN.self.rnn import RNN
from NN.self.lstm import LSTM


class SeqRecall(Module):
    """读入 token 序列 (B, T)，在末尾输出对「首 token」的分类 logits。

    rnn_type 传 ``RNN`` 或 ``LSTM``，其余关键字参数透传给 RNN 模块。
    """

    def __init__(self, rnn_type, vocab_size, hidden_size, num_layers=1, **rnn_kwargs):
        super().__init__()
        self.embed = Embedding(vocab_size, hidden_size)
        self.rnn = rnn_type(hidden_size, hidden_size, num_layers,
                            num_classes=vocab_size, **rnn_kwargs)

    def forward(self, idx):
        # idx: (B, T) 整数列表 -> (B, vocab)
        return self.rnn(self.embed(idx))
