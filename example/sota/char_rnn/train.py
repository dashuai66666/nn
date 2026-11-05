"""字符级 RNN 训练与文本生成演示。

在小段英文语料上训练 CharRNN，随后按温度采样逐字续写，展示 RNN 逐字预测
下一个字符的能力（Karpathy 2015 的标志性演示）。

运行（在仓库根目录）：
    python3 -m NN.example.sota.char_rnn.train
"""

import math
import random

from NN.self.core import set_seed, F, Tensor
from NN.self.core.tensor import softmax_last
from NN.example.sota._common import SGD
from .model import CharRNN


TEXT = (
    "deep learning is a subset of machine learning. "
    "a neural network learns from data. "
    "recurrent networks process sequences step by step. "
    "long short term memory remembers over time. "
    "attention lets a model focus on relevant parts. "
    "the transformer is built entirely from attention. "
    "language models predict the next word or character. "
)


def build_vocab(text):
    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    return chars, c2i, i2c


def make_batches(encoded, seq_len, batch_size):
    """把整段文本切成 (batch, seq_len, seq_len+1) 的输入/目标块。"""
    xs, ys = [], []
    for _ in range(batch_size):
        start = random.randrange(0, len(encoded) - seq_len - 1)
        xs.append(encoded[start:start + seq_len])
        ys.append(encoded[start + 1:start + seq_len + 1])
    return xs, ys


def compute_loss(model, xs, ys):
    """逐时间步求交叉熵并求和（等价于对整个序列求平均损失）。"""
    logits = model(xs)                              # [ (B, vocab) ] * T
    total = None
    B = len(xs)
    for t, lg in enumerate(logits):
        tgt = [ys[b][t] for b in range(B)]
        loss_t = F.cross_entropy(lg, tgt)
        total = loss_t if total is None else total + loss_t
    return total


def sample_dist(probs, temp):
    probs = [p ** (1.0 / temp) for p in probs]
    s = sum(probs)
    r = random.random() * s
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def generate(model, seed_chars, length, temp=0.8):
    """给定种子字符索引列表，逐字采样生成。"""
    h = model._init_state(1)
    generated = list(seed_chars)
    for _ in range(length):
        lg, h = model.step([generated[-1]], h)
        probs = softmax_last(lg.data[0])
        generated.append(sample_dist(probs, temp))
    return generated


def main():
    set_seed(0)
    random.seed(0)
    chars, c2i, i2c = build_vocab(TEXT)
    encoded = [c2i[c] for c in TEXT]

    model = CharRNN(vocab_size=len(chars), hidden_size=32, num_layers=1)
    opt = SGD(model.parameters(), lr=0.1)

    print("字符级 RNN（词表 %d，隐层 32）训练 %d 步" % (len(chars), 120))
    steps = 120
    for step in range(steps):
        xs, ys = make_batches(encoded, seq_len=20, batch_size=8)
        loss = compute_loss(model, xs, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            print("  step %3d: 损失 %.4f" % (step + 1, loss.item()))

    # 以文本开头为种子，续写一段
    seed = encoded[:25]
    out = generate(model, seed, length=80, temp=0.7)
    print("\n续写结果：")
    print('"' + "".join(i2c[i] for i in out) + '"')


if __name__ == "__main__":
    main()
