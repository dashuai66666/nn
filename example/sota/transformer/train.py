"""Transformer 复现训练演示：在小型"复制任务"上训练编码器-解码器 Transformer。

任务：把源序列原样复制到目标序列（源 = [w1..wL] + EOS，目标 = SOS + [w1..wL] + EOS）。
这是检验 seq2seq 模型能否学会对齐的经典最小任务。训练用 teacher forcing，
推理用贪心解码并打印复制结果。

运行（在仓库根目录）：
    python3 -m NN.example.sota.transformer.train
"""

import random

from NN.self.core import set_seed, F
from NN.example.sota._common import Adam
from .model import Transformer, causal_mask

# 任务参数
SRC_LEN = 4         # 每个源序列的 token 数
VOCAB = 6           # 普通 token 取值 1..6
SOS = 0             # 目标序列起始符
EOS = VOCAB + 1     # 序列结束符
D_MODEL = 16
N_HEADS = 2
D_FF = 32
N_LAYERS = 1
STEPS = 100


def make_batch(batch):
    srcs, tgts = [], []
    for _ in range(batch):
        seq = [random.randrange(1, VOCAB + 1) for _ in range(SRC_LEN)]
        srcs.append(seq + [EOS])
        tgts.append([SOS] + seq + [EOS])
    return srcs, tgts


def compute_loss(model, srcs, tgts):
    B = len(srcs)
    tgt_in = [t[:-1] for t in tgts]          # 去掉最后的 EOS
    tgt_out = [t[1:] for t in tgts]          # 去掉开头的 SOS
    T = len(tgt_in[0])
    logits = model(srcs, tgt_in, causal_mask(T))   # (B, T, vocab)
    flat = logits.reshape(B * T, EOS + 1)          # (B*T, vocab)
    flat_targets = [tgt_out[b][t] for b in range(B) for t in range(T)]
    return F.cross_entropy(flat, flat_targets)


def greedy_decode(model, src, max_len=16):
    """贪心解码：从 SOS 开始逐 token 生成，直到 EOS 或达到最大长度。"""
    tgt = [SOS]
    for _ in range(max_len):
        logits = model([src], [tgt], causal_mask(len(tgt)))  # (1, T, vocab)
        last = logits.data[0][-1]                            # 最后一个位置的 logits
        nxt = max(range(len(last)), key=last.__getitem__)
        tgt.append(nxt)
        if nxt == EOS:
            break
    return tgt


def main():
    set_seed(0)
    random.seed(0)
    model = Transformer(src_vocab=EOS + 1, tgt_vocab=EOS + 1,
                        d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                        num_layers=N_LAYERS, max_len=32, dropout=0.0)
    opt = Adam(model.parameters(), lr=0.003)

    print("编码器-解码器 Transformer 训练（复制任务，序列长度 %d）" % SRC_LEN)
    for step in range(STEPS):
        srcs, tgts = make_batch(batch=6)
        loss = compute_loss(model, srcs, tgts)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 40 == 0:
            print("  step %3d/%d: 损失 %.4f" % (step + 1, STEPS, loss.item()))

    # 展示几个贪心解码结果
    print("\n贪心解码示例（源序列 -> 生成序列）：")
    for _ in range(4):
        src, tgt = make_batch(1)
        src = src[0]
        pred = greedy_decode(model, src)
        ok = "✓" if pred == tgt[0] else "✗"
        print("  %s -> %s  %s" % (src, pred, ok))


if __name__ == "__main__":
    main()
