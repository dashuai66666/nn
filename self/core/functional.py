"""功能层：在自动求导 Tensor 之上实现的算子。

这里只放那些无法（或不便）由基础算子组合出的"融合算子"：
卷积、最大池化、层归一化、交叉熵、dropout。它们都直接复用
``tensor.Tensor`` 的自动求导机制，模型层调用它们即可共享同一套梯度传播。
"""

import math

from .tensor import (
    Tensor,
    zeros,
    zeros_like,
    _accum,
    softmax_last,
    bcast_binary,
)


def cross_entropy(logits, target):
    """多分类交叉熵（数值稳定，logits 为 (N, C)，target 为长度 N 的整数列表）。

    返回标量损失，反向传播直接给出 ``logits.grad = (softmax - onehot) / N``。
    """
    probs = softmax_last(logits.data)          # (N, C) 每行的概率
    n = len(target)
    loss = 0.0
    for i in range(n):
        loss += -math.log(probs[i][target[i]] + 1e-12)
    loss /= n

    out = Tensor(loss, shape=(), _children=(logits,), _op="cross_entropy")

    def _backward():
        grad = zeros_like(logits.data)
        for i in range(n):
            row = grad[i]
            t = target[i]
            for j in range(len(row)):
                row[j] = (probs[i][j] - (1.0 if j == t else 0.0)) / n
        _accum(logits, bcast_binary(grad, out.grad, lambda a, b: a * b))

    out._backward = _backward
    return out


def conv2d(x, weight, bias, stride=1, padding=0):
    """二维卷积（输入 (N, C, H, W)，权重 (C_out, C_in, kH, kW)）。

    采用直接滑窗 + 越界补零的方式，梯度同样按滑窗回填，无需 im2col。
    """
    xd, wd = x.data, weight.data
    N, C_in, H, W = x.shape
    C_out, C_in2, kH, kW = weight.shape
    assert C_in == C_in2, "通道数不一致"

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1
    bd = bias.data if bias is not None else None

    out = zeros([N, C_out, H_out, W_out])
    for n in range(N):
        for co in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    acc = bd[co] if bd is not None else 0.0
                    for ci in range(C_in):
                        for kh in range(kH):
                            ih = oh * stride - padding + kh
                            if 0 <= ih < H:
                                for kw in range(kW):
                                    iw = ow * stride - padding + kw
                                    if 0 <= iw < W:
                                        acc += xd[n][ci][ih][iw] * wd[co][ci][kh][kw]
                    out[n][co][oh][ow] = acc

    out_t = Tensor(out, _children=(x, weight) if bias is None else (x, weight, bias),
                   _op="conv2d")

    def _backward():
        g = out_t.grad
        dx = zeros(x.shape)
        dw = zeros(weight.shape)
        db = zeros([C_out]) if bias is not None else None
        for n in range(N):
            for co in range(C_out):
                for oh in range(H_out):
                    for ow in range(W_out):
                        gv = g[n][co][oh][ow]
                        if db is not None:
                            db[co] += gv
                        for ci in range(C_in):
                            for kh in range(kH):
                                ih = oh * stride - padding + kh
                                if 0 <= ih < H:
                                    for kw in range(kW):
                                        iw = ow * stride - padding + kw
                                        if 0 <= iw < W:
                                            dw[co][ci][kh][kw] += gv * xd[n][ci][ih][iw]
                                            dx[n][ci][ih][iw] += gv * wd[co][ci][kh][kw]
        _accum(x, dx)
        _accum(weight, dw)
        if db is not None:
            _accum(bias, db)

    out_t._backward = _backward
    return out_t


def max_pool2d(x, kernel, stride=None):
    """二维最大池化。反向传播只把梯度回传给窗口内的最大值位置。"""
    stride = stride or kernel
    xd = x.data
    N, C, H, W = x.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1

    out = zeros([N, C, H_out, W_out])
    argmax = zeros([N, C, H_out, W_out])  # 叶子存放 (ih, iw) 元组
    for n in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    best = -float("inf")
                    bi = bj = 0
                    for kh in range(kernel):
                        ih = oh * stride + kh
                        for kw in range(kernel):
                            iw = ow * stride + kw
                            v = xd[n][c][ih][iw]
                            if v > best:
                                best, bi, bj = v, ih, iw
                    out[n][c][oh][ow] = best
                    argmax[n][c][oh][ow] = (bi, bj)

    out_t = Tensor(out, _children=(x,), _op="max_pool2d")

    def _backward():
        g = out_t.grad
        dx = zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        ih, iw = argmax[n][c][oh][ow]
                        dx[n][c][ih][iw] += g[n][c][oh][ow]
        _accum(x, dx)

    out_t._backward = _backward
    return out_t


def avg_pool2d(x, kernel, stride=None):
    """二维平均池化。反向传播把梯度均分回窗口内的每个元素（LeNet 使用）。"""
    stride = stride or kernel
    xd = x.data
    N, C, H, W = x.shape
    H_out = (H - kernel) // stride + 1
    W_out = (W - kernel) // stride + 1
    area = kernel * kernel

    out = zeros([N, C, H_out, W_out])
    for n in range(N):
        for c in range(C):
            for oh in range(H_out):
                for ow in range(W_out):
                    s = 0.0
                    for kh in range(kernel):
                        ih = oh * stride + kh
                        for kw in range(kernel):
                            iw = ow * stride + kw
                            s += xd[n][c][ih][iw]
                    out[n][c][oh][ow] = s / area

    out_t = Tensor(out, _children=(x,), _op="avg_pool2d")

    def _backward():
        g = out_t.grad
        dx = zeros(x.shape)
        for n in range(N):
            for c in range(C):
                for oh in range(H_out):
                    for ow in range(W_out):
                        gv = g[n][c][oh][ow] / area
                        for kh in range(kernel):
                            ih = oh * stride + kh
                            for kw in range(kernel):
                                iw = ow * stride + kw
                                dx[n][c][ih][iw] += gv
        _accum(x, dx)

    out_t._backward = _backward
    return out_t


def layer_norm(x, gamma, beta, eps=1e-5):
    """层归一化（沿最后一维）。输入任意前导维，内部展平成 2D 处理再还原。

    融合实现以便稳定地一次性给出对输入、缩放、偏置三者的梯度。
    """
    D = x.shape[-1]
    original_shape = x.shape
    xf = x.reshape(-1, D)                    # 展平成 (N, D)，梯度经 reshape 算子回流
    xd = xf.data
    gd = gamma.data
    bd = beta.data
    n = len(xd)

    # 前向：逐行统计均值/方差
    out = zeros([n, D])
    xhat = zeros([n, D])
    inv_std = [0.0] * n
    for i in range(n):
        row = xd[i]
        mean = sum(row) / D
        var = sum((v - mean) ** 2 for v in row) / D
        inv = 1.0 / math.sqrt(var + eps)
        inv_std[i] = inv
        for j in range(D):
            xh = (row[j] - mean) * inv
            xhat[i][j] = xh
            out[i][j] = xh * gd[j] + bd[j]

    out_flat = Tensor(out, _children=(xf, gamma, beta), _op="layer_norm")

    def _backward():
        g = out_flat.grad
        dx = zeros([n, D])
        dgamma = zeros([D])
        dbeta = zeros([D])
        for i in range(n):
            gi = g[i]
            xh = xhat[i]
            inv = inv_std[i]
            for j in range(D):
                dgamma[j] += gi[j] * xh[j]
                dbeta[j] += gi[j]
            dxhat = [gi[j] * gd[j] for j in range(D)]
            s1 = sum(dxhat)
            s2 = sum(dxhat[j] * xh[j] for j in range(D))
            scale = inv / D
            for j in range(D):
                dx[i][j] = scale * (D * dxhat[j] - s1 - xh[j] * s2)
        _accum(xf, dx)
        _accum(gamma, dgamma)
        _accum(beta, dbeta)

    out_flat._backward = _backward
    return out_flat.reshape(original_shape)


def dropout(x, p, mask):
    """训练期随机失活：直接复用逐元素乘法，梯度经同一 mask 回传。"""
    return x * mask


def embedding(weight, idx):
    """查表嵌入：``idx`` 为任意嵌套的整数列表，输出形状为 idx 形状 + [D]。

    反向传播把梯度散射（累加）回被查到的权重行。
    """
    wd = weight.data
    out = _gather_rows(wd, idx)
    out_t = Tensor(out, _children=(weight,), _op="embedding")

    def _backward():
        g = out_t.grad
        dw = zeros(weight.shape)
        _scatter_embed(dw, g, idx)
        _accum(weight, dw)

    out_t._backward = _backward
    return out_t


def _gather_rows(wd, idx):
    if isinstance(idx, (list, tuple)):
        return [_gather_rows(wd, i) for i in idx]
    return wd[idx]


def _scatter_embed(dw, g, idx):
    if isinstance(idx, (list, tuple)):
        for gi, ii in zip(g, idx):
            _scatter_embed(dw, gi, ii)
    else:
        row = dw[idx]
        for j in range(len(row)):
            row[j] += g[j]
