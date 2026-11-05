"""LeNet-5 训练演示：在合成"手写数字"上训练并评估准确率。

因本机无 MNIST 数据集，这里为每个数字类生成一张固定原型图（若干随机亮斑）并
叠加高斯噪声，构造一个可学习的 32x32 单通道分类任务，验证 LeNet-5 端到端可训练。

运行（在仓库根目录）：
    python3 -m NN.example.sota.lenet5.train
"""

import random

from NN.self.core import set_seed, F, Tensor
from NN.example.sota._common import SGD
from .model import LeNet5


def make_prototypes():
    """为 0~9 每个类生成一张 32x32 原型图（类内固定、类间不同）。"""
    protos = []
    for c in range(10):
        img = [[0.0] * 32 for _ in range(32)]
        rr = random.Random(1000 + c)
        for _ in range(4):                     # 每类 4 个随机亮斑
            r, col = rr.randrange(24), rr.randrange(24)
            h, w = rr.randrange(3, 8), rr.randrange(3, 8)
            v = rr.random() * 0.8 + 0.2
            for i in range(h):
                for j in range(w):
                    img[r + i][col + j] += v
        protos.append(img)
    return protos


def make_dataset(samples_per_class=10, noise=0.35):
    """返回 (train_x, train_y, val_x, val_y)。x 形状 (N, 1, 32, 32)。"""
    protos = make_prototypes()
    rng = random.Random(7)
    train_x, train_y, val_x, val_y = [], [], [], []
    for c in range(10):
        for k in range(samples_per_class):
            img = [[max(0.0, min(1.0, protos[c][i][j] + rng.gauss(0.0, noise)))
                    for j in range(32)] for i in range(32)]
            sample = [img]
            if k % 2 == 0:
                train_x.append(sample)
                train_y.append(c)
            else:
                val_x.append(sample)
                val_y.append(c)
    return train_x, train_y, val_x, val_y


def accuracy(model, xs, ys):
    correct = 0
    for x, y in zip(xs, ys):
        logits = model(Tensor([x]))               # (1, 1, 32, 32) -> (1, 10)
        pred = max(range(len(logits.data[0])),
                   key=lambda j: logits.data[0][j])
        correct += (pred == y)
    return correct / len(ys)


def main():
    set_seed(0)
    train_x, train_y, val_x, val_y = make_dataset(samples_per_class=6)
    model = LeNet5(num_classes=10)
    opt = SGD(model.parameters(), lr=0.15)

    print("LeNet-5 训练（合成数字，训练集 %d / 验证集 %d）"
          % (len(train_x), len(val_x)))
    epochs = 4
    for epoch in range(epochs):
        # 全批量前向 + 反向
        logits = model(Tensor(train_x))          # (N, 10)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print("  epoch %d: 损失 %.4f" % (epoch + 1, loss.item()))

    # 一次性评估整个验证集
    acc = accuracy(model, val_x, val_y)
    print("训练完成，验证准确率 %.1f%%（%d/%d）"
          % (100 * acc, int(round(acc * len(val_y))), len(val_y)))


if __name__ == "__main__":
    main()
