"""self 目录的模型冒烟测试 + 梯度校验 + 小规模训练演示。

- 对 RNN / CNN / LSTM / Transformer 各跑一次前向 + 反向，确认所有参数都有梯度；
- 用数值微分（中心差分）抽查若干参数的解析梯度是否正确；
- 用随机梯度下降训练数步，确认损失下降，验证整个框架端到端可学习。

运行方式（在 NN 目录下）：
    python3 -m self.test_models
"""

import random

from .core import set_seed
from .core import functional as F
from .core.tensor import Tensor, randn, randint, _get, _set, bcast_binary
from .rnn import RNN
from .lstm import LSTM
from .cnn import CNN
from .transformer import Transformer


def cross_entropy_loss(model, inputs, target):
    """前向并计算交叉熵损失。"""
    return F.cross_entropy(model(*inputs), target)


def all_params_have_grad(model):
    for p in model.parameters():
        if p.grad is None:
            return False
    return True


def check_grad(model, inputs, target, samples=10, eps=1e-4, tol=1e-2):
    """数值梯度校验：先反向求解析梯度，再对若干参数元素做中心差分对比。"""
    model.eval()                                   # 关闭 dropout，保证确定性
    model.zero_grad()                              # 清空上一次反向的梯度，避免叠加
    loss = cross_entropy_loss(model, inputs, target)
    loss.backward()

    # 均匀地在各参数上抽样，收集 (参数, 索引, 解析梯度)
    params = list(model.parameters())
    picked = []
    rng = random.Random(123)
    per_param = max(1, samples // max(1, len(params)))
    for p in params:
        if p.grad is None or not p.shape:
            continue
        for _ in range(per_param):
            idx = tuple(rng.randrange(s) for s in p.shape)
            picked.append((p, idx, _get(p.grad, idx)))

    bad = 0
    for p, idx, ana in picked:
        orig = _get(p.data, idx)
        _set(p.data, idx, orig + eps)
        lp = cross_entropy_loss(model, inputs, target).item()
        _set(p.data, idx, orig - eps)
        lm = cross_entropy_loss(model, inputs, target).item()
        _set(p.data, idx, orig)                   # 还原
        num = (lp - lm) / (2 * eps)
        denom = abs(num) + abs(ana) + 1e-8
        if abs(num - ana) / denom > tol:
            bad += 1
            print("    梯度不符: shape=%s idx=%s 数值=%g 解析=%g" % (p.shape, idx, num, ana))
    return bad == 0


def train_steps(model, inputs, target, steps=6, lr=0.3):
    """确定性（eval 模式）地训练数步，返回 (初始损失, 最终损失)。"""
    model.eval()
    losses = []
    for step in range(steps):
        loss = cross_entropy_loss(model, inputs, target)
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.data = bcast_binary(p.data, p.grad, lambda w, g: w - lr * g)
    return losses[0], losses[-1]


def cases():
    """构造四个模型的 (名称, 模型, 输入, 标签)。"""
    cases_ = []
    cases_.append(("RNN", RNN(input_size=4, hidden_size=8, num_layers=2, num_classes=3),
                   (Tensor(randn([3, 5, 4])),), [0, 1, 2]))
    cases_.append(("LSTM", LSTM(input_size=4, hidden_size=8, num_layers=2, num_classes=3),
                   (Tensor(randn([3, 5, 4])),), [0, 1, 2]))
    cases_.append(("CNN", CNN(in_channels=1, num_classes=3, image_size=8),
                   (Tensor(randn([3, 1, 8, 8])),), [0, 1, 2]))
    cases_.append(("Transformer",
                   Transformer(vocab_size=10, d_model=8, n_heads=2, d_ff=16,
                               num_layers=2, num_classes=3, max_len=16),
                   (randint([3, 4], 10),), [0, 1, 2]))
    return cases_


def main():
    set_seed(0)
    all_ok = True
    for name, model, inputs, target in cases():
        print("== %s ==" % name)
        n_params = sum(1 for _ in model.parameters())

        # 1) 前向 + 反向，参数应全部获得梯度
        model.zero_grad()
        loss = cross_entropy_loss(model, inputs, target)
        loss.backward()
        assert all_params_have_grad(model), "%s 存在无梯度的参数" % name
        print("  前向/反向通过，参数数量=%d，损失=%g" % (n_params, loss.item()))

        # 2) 数值梯度校验
        ok = check_grad(model, inputs, target)
        print("  梯度校验:", "通过" if ok else "失败")
        all_ok = all_ok and ok

        # 3) 小规模训练，损失应下降
        init_loss, final_loss = train_steps(model, inputs, target)
        print("  训练 %d 步: 损失 %g -> %g" % (6, init_loss, final_loss))
        assert final_loss < init_loss, "%s 训练后损失未下降" % name

    print("\n全部通过" if all_ok else "\n存在失败项")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
