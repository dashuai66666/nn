"""最小化的自动求导张量（纯 Python，仅用标准库）。

张量内部用嵌套 list 表示、同时记录 shape。所有基础算子都在这里实现，
RNN / CNN / LSTM / Transformer 四个模型共享同一套算子与同一套层，
从而实现最大程度的代码复用。
"""

import math
import random

# ---------------------------------------------------------------------------
# 形状 / 构造辅助
# ---------------------------------------------------------------------------

def shape_of(x):
    """返回嵌套 list 的形状（标量为 []）。"""
    s = []
    while isinstance(x, (list, tuple)):
        s.append(len(x))
        if len(x) == 0:
            break
        x = x[0]
    return s


def prod(seq):
    out = 1
    for v in seq:
        out *= v
    return out


def zeros(shape):
    if not shape:
        return 0.0
    return [zeros(shape[1:]) for _ in range(shape[0])]


def ones(shape):
    if not shape:
        return 1.0
    return [ones(shape[1:]) for _ in range(shape[0])]


def full(shape, value):
    if not shape:
        return value
    return [full(shape[1:], value) for _ in range(shape[0])]


def _gauss():
    # Box-Muller 变换产生标准正态随机数
    u1 = random.random()
    u2 = random.random()
    return math.sqrt(-2.0 * math.log(u1 + 1e-12)) * math.cos(2.0 * math.pi * u2)


def randn(shape):
    if not shape:
        return _gauss()
    return [randn(shape[1:]) for _ in range(shape[0])]


def uniform(shape, lo, hi):
    if not shape:
        return lo + (hi - lo) * random.random()
    return [uniform(shape[1:], lo, hi) for _ in range(shape[0])]


def randint(shape, high):
    if not shape:
        return random.randrange(high)
    return [randint(shape[1:], high) for _ in range(shape[0])]


def zeros_like(data):
    return map_elems(data, lambda _: 0.0)


def ones_like(data):
    return map_elems(data, lambda _: 1.0)


def map_elems(x, fn):
    if isinstance(x, (list, tuple)):
        return [map_elems(item, fn) for item in x]
    return fn(x)


def flatten_list(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    out = []
    for item in x:
        out.extend(flatten_list(item))
    return out


def reshape(x, new_shape):
    flat = flatten_list(x)
    assert prod(new_shape) == len(flat), "reshape 元素总数不一致"
    return _build(flat, list(new_shape))


def _build(flat, shape):
    if not shape:
        return flat[0]
    n = shape[0]
    inner = prod(shape[1:])
    return [_build(flat[i * inner:(i + 1) * inner], shape[1:]) for i in range(n)]


# ---------------------------------------------------------------------------
# 广播（numpy 风格，按尾维度对齐）
# ---------------------------------------------------------------------------

def broadcast_shape(sa, sb):
    rank = max(len(sa), len(sb))
    sa = [1] * (rank - len(sa)) + list(sa)
    sb = [1] * (rank - len(sb)) + list(sb)
    out = []
    for a, b in zip(sa, sb):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise ValueError("无法广播 %s 与 %s" % (sa, sb))
    return out


def _src_index(idx, shape):
    """把输出索引映射回某个操作数的索引（广播维恒取 0，标量返回空元组）。

    操作数按尾维度对齐，返回的索引长度等于操作数自身的秩。
    """
    if not shape:
        return ()
    rank = len(idx)
    spad = [1] * (rank - len(shape)) + list(shape)
    result = []
    for i in range(len(shape)):
        pos = rank - len(shape) + i
        result.append(0 if spad[pos] == 1 else idx[pos])
    return tuple(result)


def bcast_binary(a, b, fn):
    """按广播规则逐元素计算二元函数，基于显式索引遍历，避免嵌套递归的边界问题。"""
    sa = shape_of(a)
    sb = shape_of(b)
    if not sa and not sb:
        return fn(a, b)
    out_shape = broadcast_shape(sa, sb)
    out = zeros(out_shape)
    for idx in _iter_indices(out_shape):
        _set(out, idx, fn(_get(a, _src_index(idx, sa)), _get(b, _src_index(idx, sb))))
    return out


def unbroadcast(g, target_shape):
    """把广播后的梯度按 ``target_shape`` 归约回去（对广播轴求和）。"""
    g_shape = shape_of(g)
    target = list(target_shape)
    rank = len(g_shape)
    tpad = [1] * (rank - len(target)) + target
    sum_axes = [i for i in range(rank) if tpad[i] == 1 and g_shape[i] != 1]
    for axis in sorted(sum_axes, reverse=True):
        g = _sum_axis(g, axis)
    return g


def broadcast_to(x, target_shape):
    tshape = list(target_shape)
    out = zeros(tshape)
    xs = shape_of(x)
    for idx in _iter_indices(tshape):
        _set(out, idx, _get(x, _src_index(idx, xs)))
    return out


# ---------------------------------------------------------------------------
# 归约 / 索引
# ---------------------------------------------------------------------------

def _sum_all(x):
    if not isinstance(x, (list, tuple)):
        return x
    s = 0.0
    for item in x:
        s += _sum_all(item)
    return s


def _sum_axis(x, axis, depth=0):
    if depth == axis:
        result = x[0]
        for item in x[1:]:
            result = bcast_binary(result, item, lambda a, b: a + b)
        return result
    return [_sum_axis(item, axis, depth + 1) for item in x]


def _iter_indices(shape):
    if not shape:
        yield ()
        return
    for sub in _iter_indices(shape[1:]):
        for i in range(shape[0]):
            yield (i,) + sub


def _get(x, idx):
    for i in idx:
        x = x[i]
    return x


def _set(x, idx, val):
    for i in idx[:-1]:
        x = x[i]
    x[idx[-1]] = val


def transpose(x, d0, d1):
    shape = shape_of(x)
    n = len(shape)
    d0 %= n
    d1 %= n
    order = list(range(n))
    order[d0], order[d1] = order[d1], order[d0]
    out_shape = [shape[o] for o in order]
    out = zeros(out_shape)
    for out_idx in _iter_indices(out_shape):
        src_idx = [0] * n
        for oi in range(n):
            src_idx[order[oi]] = out_idx[oi]
        _set(out, out_idx, _get(x, src_idx))
    return out


def select_data(x, axis, idx, depth=0):
    if depth == axis:
        return x[idx]
    return [select_data(item, axis, idx, depth + 1) for item in x]


def scatter_add(target, src, axis, idx, depth=0):
    if depth == axis:
        target[idx] = bcast_binary(target[idx], src, lambda a, b: a + b)
        return target
    return [scatter_add(target[i], src[i], axis, idx, depth + 1)
            for i in range(len(target))]


def slice_data(x, axis, start, end, depth=0):
    if depth == axis:
        return x[start:end]
    return [slice_data(item, axis, start, end, depth + 1) for item in x]


def scatter_slice(target, src, axis, start, depth=0):
    if depth == axis:
        target[start:start + len(src)] = src
        return target
    return [scatter_slice(target[i], src[i], axis, start, depth + 1)
            for i in range(len(target))]


# ---------------------------------------------------------------------------
# 矩阵乘法
# ---------------------------------------------------------------------------

def matmul(a, b):
    sa = shape_of(a)
    sb = shape_of(b)
    if len(sa) == 2 and len(sb) == 2:
        return _matmul2d(a, b)
    if len(sa) == 3 and len(sb) == 2:
        return [_matmul2d(x, b) for x in a]
    if len(sa) == 3 and len(sb) == 3:
        return [_matmul2d(x, y) for x, y in zip(a, b)]
    if len(sa) == 2 and len(sb) == 3:
        # (M, K) @ (B, K, N) -> (B, M, N)
        return [_matmul2d(a, y) for y in b]
    raise ValueError("不支持的 matmul 形状: %s @ %s" % (sa, sb))


def _matmul2d(a, b):
    m, k = shape_of(a)
    k2, n = shape_of(b)
    assert k == k2, "内维不一致: %d vs %d" % (k, k2)
    out = zeros([m, n])
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += ai[t] * b[t][j]
            oi[j] = s
    return out


def _matmul_grads(a, b, g, sa, sb):
    """out = A @ B 时返回 (dA, dB)，覆盖 2D/3D 的批量组合。"""
    if len(sa) == 2 and len(sb) == 2:
        dA = _matmul2d(g, transpose(b, 0, 1))
        dB = _matmul2d(transpose(a, 0, 1), g)
    elif len(sa) == 3 and len(sb) == 2:
        bt = transpose(b, 0, 1)
        dA = [_matmul2d(gi, bt) for gi in g]
        dB = _sum_axis([_matmul2d(transpose(ai, 0, 1), gi) for ai, gi in zip(a, g)], 0)
    elif len(sa) == 3 and len(sb) == 3:
        dA = [_matmul2d(gi, transpose(bi, 0, 1)) for gi, bi in zip(g, b)]
        dB = [_matmul2d(transpose(ai, 0, 1), gi) for ai, gi in zip(a, g)]
    elif len(sa) == 2 and len(sb) == 3:
        dA = _sum_axis([_matmul2d(gi, transpose(bi, 0, 1)) for gi, bi in zip(g, b)], 0)
        dB = [_matmul2d(transpose(a, 0, 1), gi) for gi in g]
    else:
        raise ValueError("matmul 梯度形状不支持: %s @ %s" % (sa, sb))
    return dA, dB


# ---------------------------------------------------------------------------
# Tensor + 自动求导引擎
# ---------------------------------------------------------------------------

class Tensor:
    __slots__ = ("data", "shape", "requires_grad", "grad", "_prev", "_backward", "_op")

    def __init__(self, data, shape=None, requires_grad=False, _children=(), _op=""):
        self.data = data
        self.shape = tuple(shape) if shape is not None else tuple(shape_of(data))
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op

    # -- 前向图构建 --------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(bcast_binary(self.data, other.data, lambda a, b: a + b),
                     _children=(self, other), _op="+")

        def _backward():
            g = out.grad
            _accum(self, unbroadcast(g, self.shape))
            _accum(other, unbroadcast(g, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(bcast_binary(self.data, other.data, lambda a, b: a * b),
                     _children=(self, other), _op="*")

        def _backward():
            g = out.grad
            # dL/da = g * b，dL/db = g * a（两个操作数的偏导不同）
            _accum(self, unbroadcast(bcast_binary(g, other.data, lambda x, y: x * y),
                                     self.shape))
            _accum(other, unbroadcast(bcast_binary(g, self.data, lambda x, y: x * y),
                                      other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __neg__(self):
        return self * -1.0

    def __pow__(self, e):
        out = Tensor(map_elems(self.data, lambda x: x ** e),
                     _children=(self,), _op="**")

        def _backward():
            g = out.grad
            _accum(self, bcast_binary(g, map_elems(self.data, lambda x: e * x ** (e - 1)),
                                      lambda x, y: x * y))

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        sa, sb = self.shape, other.shape
        out = Tensor(matmul(self.data, other.data),
                     _children=(self, other), _op="@")

        def _backward():
            g = out.grad
            dA, dB = _matmul_grads(self.data, other.data, g, sa, sb)
            _accum(self, dA)
            _accum(other, dB)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None):
        if axis is None:
            data = _sum_all(self.data)
            shape = ()
        else:
            axis = axis % len(self.shape)
            data = _sum_axis(self.data, axis)
            shape = list(self.shape)
            del shape[axis]
        out = Tensor(data, shape=shape, _children=(self,), _op="sum")

        def _backward():
            g = out.grad
            if axis is None:
                _accum(self, broadcast_to(g, self.shape))
            else:
                _accum(self, broadcast_to(reshape(g, shape[:axis] + [1] + shape[axis:]),
                                          self.shape))

        out._backward = _backward
        return out

    def mean(self, axis=None):
        n = prod(self.shape) if axis is None else self.shape[axis]
        return self.sum(axis) / n

    def transpose(self, d0, d1):
        out = Tensor(transpose(self.data, d0, d1),
                     _children=(self,), _op="T")

        def _backward():
            _accum(self, transpose(out.grad, d0, d1))

        out._backward = _backward
        return out

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = new_shape[0]
        new_shape = list(new_shape)
        if -1 in new_shape:
            idx = new_shape.index(-1)
            known = prod([s for s in new_shape if s != -1])
            new_shape[idx] = prod(self.shape) // known
        out = Tensor(reshape(self.data, new_shape), shape=tuple(new_shape),
                     _children=(self,), _op="reshape")

        def _backward():
            _accum(self, reshape(out.grad, self.shape))

        out._backward = _backward
        return out

    def select(self, axis, idx):
        axis = axis % len(self.shape)
        data = select_data(self.data, axis, idx)
        shape = list(self.shape)
        del shape[axis]
        out = Tensor(data, shape=shape, _children=(self,), _op="select")

        def _backward():
            g_full = zeros(self.shape)
            scatter_add(g_full, out.grad, axis, idx)
            _accum(self, g_full)

        out._backward = _backward
        return out

    def slice(self, axis, start, end):
        """沿某一维取连续区间 [start, end)，梯度反向散射回原形状。"""
        axis = axis % len(self.shape)
        data = slice_data(self.data, axis, start, end)
        shape = list(self.shape)
        shape[axis] = end - start
        out = Tensor(data, shape=shape, _children=(self,), _op="slice")

        def _backward():
            g_full = zeros(self.shape)
            scatter_slice(g_full, out.grad, axis, start)
            _accum(self, g_full)

        out._backward = _backward
        return out

    # -- 逐元素非线性 -------------------------------------------------------
    def _unary(self, fn, dfn, op):
        out = Tensor(map_elems(self.data, fn), _children=(self,), _op=op)

        def _backward():
            g = out.grad
            _accum(self, dfn(g, out.data))

        out._backward = _backward
        return out

    def relu(self):
        return self._unary(lambda x: x if x > 0 else 0.0,
                           lambda g, o: bcast_binary(g, map_elems(o, lambda x: 1.0 if x > 0 else 0.0),
                                                     lambda x, y: x * y),
                           "relu")

    def tanh(self):
        return self._unary(math.tanh,
                           lambda g, o: bcast_binary(g, map_elems(o, lambda x: 1.0 - x * x),
                                                     lambda x, y: x * y),
                           "tanh")

    def sigmoid(self):
        return self._unary(lambda x: 1.0 / (1.0 + math.exp(-x)),
                           lambda g, o: bcast_binary(g, map_elems(o, lambda x: x * (1.0 - x)),
                                                     lambda x, y: x * y),
                           "sigmoid")

    def exp(self):
        return self._unary(math.exp,
                           lambda g, o: bcast_binary(g, o, lambda x, y: x * y), "exp")

    def log(self):
        return self._unary(math.log,
                           lambda g, o: bcast_binary(g, self.data, lambda x, y: x / (y + 1e-12)),
                           "log")

    def softmax(self, axis=-1):
        axis = axis % len(self.shape)
        data = softmax_last(self.data)
        out = Tensor(data, _children=(self,), _op="softmax")

        def _backward():
            _accum(self, softmax_last_backward(out.grad, out.data))

        out._backward = _backward
        return out

    # -- 其他 ---------------------------------------------------------------
    def item(self):
        if self.shape:
            raise ValueError("item() 仅用于标量张量")
        return self.data

    def tolist(self):
        return self.data

    def backward(self):
        topo = []
        visited = set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for c in t._prev:
                    build(c)
                topo.append(t)

        build(self)
        self.grad = 1.0 if not self.shape else ones(self.shape)
        for t in reversed(topo):
            t._backward()

    def __repr__(self):
        return "Tensor(shape=%s, requires_grad=%s)" % (self.shape, self.requires_grad)


def _accum(t, g):
    """把梯度累加到张量上（中间节点也需要梯度以便链式回传）。"""
    if t.grad is None:
        t.grad = g
    else:
        t.grad = bcast_binary(t.grad, g, lambda a, b: a + b)


# ---------------------------------------------------------------------------
# Softmax 辅助
# ---------------------------------------------------------------------------

def softmax_last(x):
    if not isinstance(x, (list, tuple)):
        return x
    if not isinstance(x[0], (list, tuple)):
        return _softmax_row(x)
    return [softmax_last(row) for row in x]


def _softmax_row(row):
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def softmax_last_backward(g, out):
    if not isinstance(out, (list, tuple)):
        return g
    if not isinstance(out[0], (list, tuple)):
        return _softmax_row_backward(g, out)
    return [softmax_last_backward(gi, oi) for gi, oi in zip(g, out)]


def _softmax_row_backward(g, out):
    dot = sum(gj * oj for gj, oj in zip(g, out))
    return [oi * (gi - dot) for oi, gi in zip(out, g)]


def set_seed(seed):
    random.seed(seed)
