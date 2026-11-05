"""神经网络基础设施：Module / Parameter 与可复用的基础层。

四个模型（RNN / CNN / LSTM / Transformer）都继承 ``Module`` 并组合这里的
``Linear`` / ``Embedding`` / ``LayerNorm`` / ``Dropout``，从而共享参数管理、
训练/评估状态切换与梯度清零等逻辑。
"""

import random

from .tensor import Tensor, zeros, ones
from . import init
from . import functional as F


class Parameter(Tensor):
    """可学习参数：本质是 ``requires_grad=True`` 的叶子张量。"""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """所有网络模块的基类，负责登记子模块与参数并递归管理它们。"""

    def __init__(self):
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        # 拦截赋值，自动登记 Parameter 与子 Module
        if isinstance(value, Parameter):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        for p in self._params.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Sequential(Module):
    """按顺序执行若干模块。"""

    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def forward(self, x):
        for i in range(len(self._modules)):
            x = self._modules[str(i)](x)
        return x


class ModuleList(Module):
    """以列表形式持有多个模块（用于 Transformer 的多层编码器）。"""

    def __init__(self, modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def __getitem__(self, i):
        return self._modules[str(i)]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        for i in range(len(self._modules)):
            yield self._modules[str(i)]


class Linear(Module):
    """全连接层：out = x @ W^T + b。权重形状 (out, in) 便于直接转置相乘。"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = Parameter(init.kaiming_uniform([out_features, in_features],
                                                     in_features))
        self.bias = Parameter(zeros([out_features])) if bias else None

    def forward(self, x):
        out = x.matmul(self.weight.transpose(0, 1))
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """词嵌入：按整数索引查表，梯度散射回对应行。"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter(init.kaiming_uniform([num_embeddings, embedding_dim],
                                                     embedding_dim))

    def forward(self, idx):
        return F.embedding(self.weight, idx)


class LayerNorm(Module):
    """层归一化：复用 functional.layer_norm。"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = Parameter(ones([normalized_shape]))
        self.bias = Parameter(zeros([normalized_shape]))
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class Dropout(Module):
    """随机失活：训练期按概率置零并缩放，评估期恒等。"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 / (1.0 - self.p)
        mask = Tensor(_dropout_mask(x.shape, self.p, keep))
        return F.dropout(x, self.p, mask)


def _dropout_mask(shape, p, keep):
    if not shape:
        return keep if random.random() >= p else 0.0
    return [_dropout_mask(shape[1:], p, keep) for _ in range(shape[0])]
