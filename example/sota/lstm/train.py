"""LSTM 长时滞任务演示：训练 LSTM 回忆序列开头的关键 token，并与普通 RNN 对比。

运行（在仓库根目录）：
    python3 -m NN.example.sota.lstm.train

说明：纯 Python 实现较慢，默认 300 步约需 1 分钟。当前时滞 T=20 较短，普通
RNN 也可能学会；把 ``T`` 调大（如 40~100）后，RNN 的梯度消失会逐渐显现，
LSTM 门控的优势才更明显——但训练时间也会成倍增加。
"""

import random

from NN.self.core import set_seed, F
from NN.example.sota._common import Adam
from .model import SeqRecall
from NN.self.rnn import RNN
from NN.self.lstm import LSTM

# 任务参数
T = 20          # 序列长度（越大越考验长期记忆）
VOCAB = 6       # 0..4 为关键 token，5 为填充 token
FILL = 5
HIDDEN = 12
STEPS = 300     # 每个网络的训练步数


def make_batch(batch, T):
    """序列 = [关键token] + [填充token]*(T-1)，标签为关键 token。"""
    xs, ys = [], []
    for _ in range(batch):
        key = random.randrange(FILL)
        xs.append([key] + [FILL] * (T - 1))
        ys.append(key)
    return xs, ys


def accuracy(model, xs, ys):
    logits = model(xs)
    correct = 0
    for b in range(len(ys)):
        pred = max(range(len(logits.data[b])),
                   key=lambda j: logits.data[b][j])
        correct += (pred == ys[b])
    return correct / len(ys)


def train_one(model, name, steps):
    opt = Adam(model.parameters(), lr=0.003)
    loss = None
    for step in range(steps):
        xs, ys = make_batch(batch=8, T=T)
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print("    %s step %3d/%d, 损失 %.4f" % (name, step + 1, steps, loss.item()))
    return loss.item()


def main():
    set_seed(0)
    random.seed(0)
    print("长时滞任务：序列长度 %d，开头是关键 token，其余为填充 token" % T)
    print("训练 %d 步（隐层 %d），对比 RNN 与 LSTM\n" % (STEPS, HIDDEN))

    results = {}
    for name, rnn_type in [("RNN", RNN), ("LSTM", LSTM)]:
        kwargs = {"forget_bias": 1.0} if rnn_type is LSTM else {}
        model = SeqRecall(rnn_type, vocab_size=VOCAB, hidden_size=HIDDEN,
                          num_layers=1, **kwargs)
        loss = train_one(model, name, STEPS)
        xs, ys = make_batch(batch=128, T=T)
        acc = accuracy(model, xs, ys)
        results[name] = acc
        print("  %-5s: 最终损失 %.4f, 回忆准确率 %.1f%%" % (name, loss, 100 * acc))

    print("\n结论：LSTM %.1f%% vs RNN %.1f%%（随机猜为 %.1f%%）。"
          % (100 * results["LSTM"], 100 * results["RNN"], 100 / FILL))
    print("      当前时滞 T=%d 较短，RNN 也可能学会；更长的时滞下 LSTM 门控优势更明显。"
          % T)


if __name__ == "__main__":
    main()
