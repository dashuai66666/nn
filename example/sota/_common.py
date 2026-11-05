"""SOTA 示例共用的轻量工具：优化器与数组辅助函数。

同样只依赖 ``self`` 的嵌套 list 表示，不引入第三方库。
"""

import math

from NN.self.core.tensor import bcast_binary


def map_lists(fn, *lists):
    """对同形状的多个嵌套 list 逐元素调用 fn（fn 接收标量）。"""
    if isinstance(lists[0], (list, tuple)):
        return [map_lists(fn, *items) for items in zip(*lists)]
    return fn(*lists)


def zeros_like_list(x):
    if isinstance(x, (list, tuple)):
        return [zeros_like_list(item) for item in x]
    return 0.0


class SGD:
    """随机梯度下降。"""

    def __init__(self, params, lr=0.1):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = map_lists(lambda w, g: w - self.lr * g, p.data, p.grad)


class Adam:
    """Adam 优化器（Transformer 训练通常更稳定）。"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [zeros_like_list(p.data) for p in self.params]
        self.v = [zeros_like_list(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t          # 一阶/二阶矩的偏差校正
        bc2 = 1.0 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = map_lists(lambda m, x: self.b1 * m + (1 - self.b1) * x,
                                  self.m[i], g)
            self.v[i] = map_lists(lambda v, x: self.b2 * v + (1 - self.b2) * x * x,
                                  self.v[i], g)
            p.data = map_lists(
                lambda w, m, v: w - self.lr * (m / bc1) / (math.sqrt(v / bc2) + self.eps),
                p.data, self.m[i], self.v[i])


def argmax_last(x):
    """沿最后一维取 argmax，返回去掉最后一维的整数嵌套 list。"""
    if not isinstance(x, (list, tuple)):
        return x
    if not isinstance(x[0], (list, tuple)):
        best, best_i = x[0], 0
        for i in range(1, len(x)):
            if x[i] > best:
                best, best_i = x[i], i
        return best_i
    return [argmax_last(row) for row in x]
