"""torch 目录的模型冒烟测试 + 小规模训练演示。

与 ``self.test_models`` 对应：对 RNN / CNN / LSTM / Transformer 各跑一次
前向 + 反向，确认参数获得梯度，并用 SGD 训练数步确认损失下降。

运行方式（需已安装 PyTorch，在 NN 目录下）：
    python3 -m torch.test_models
"""

import torch
import torch.nn.functional as F

from .rnn import RNN
from .lstm import LSTM
from .cnn import CNN
from .transformer import Transformer


def train_steps(model, inputs, target, steps=6, lr=0.3):
    """确定性训练数步，返回 (初始损失, 最终损失)。"""
    model.eval()                      # 关闭 dropout，便于观察损失单调变化
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(*inputs)
        loss = F.cross_entropy(logits, target)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses[0], losses[-1]


def cases():
    torch.manual_seed(0)
    cases_ = []
    cases_.append(("RNN", RNN(4, 8, 2, 3), (torch.randn(3, 5, 4),),
                   torch.tensor([0, 1, 2])))
    cases_.append(("LSTM", LSTM(4, 8, 2, 3), (torch.randn(3, 5, 4),),
                   torch.tensor([0, 1, 2])))
    cases_.append(("CNN", CNN(1, 3, image_size=8), (torch.randn(3, 1, 8, 8),),
                   torch.tensor([0, 1, 2])))
    cases_.append(("Transformer",
                   Transformer(vocab_size=10, d_model=8, n_heads=2, d_ff=16,
                               num_layers=2, num_classes=3, max_len=16),
                   (torch.randint(0, 10, (3, 4)),), torch.tensor([0, 1, 2])))
    return cases_


def main():
    for name, model, inputs, target in cases():
        print("== %s ==" % name)
        n_params = sum(p.numel() for p in model.parameters())
        model.zero_grad()
        logits = model(*inputs)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        grads_ok = all(p.grad is not None for p in model.parameters())
        assert grads_ok, "%s 存在无梯度的参数" % name
        print("  前向/反向通过，参数数量=%d，损失=%g" % (n_params, loss.item()))

        init_loss, final_loss = train_steps(model, inputs, target)
        print("  训练 6 步: 损失 %g -> %g" % (init_loss, final_loss))
        assert final_loss < init_loss, "%s 训练后损失未下降" % name

    print("\n全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
