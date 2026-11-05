# NN

神经网络模型集合，包含两套平行实现，用于对照「官方框架」与「从零手写」两种方式：

```
NN/
├── torch/                 # 基于 PyTorch 构建（需安装 torch）
│   ├── rnn.py             #   nn.RNN
│   ├── lstm.py            #   nn.LSTM
│   ├── cnn.py             #   nn.Conv2d / nn.MaxPool2d
│   ├── transformer.py     #   nn.TransformerEncoder
│   └── test_models.py     #   冒烟测试 + 小规模训练
└── self/                  # 纯 Python（仅标准库）从零实现
    ├── core/              # 共享核心：自动求导 + 基础层
    │   ├── tensor.py      #   最小自动求导张量（广播/矩阵乘/softmax/...）
    │   ├── functional.py  #   卷积/池化/层归一化/交叉熵等融合算子
    │   ├── nn.py          #   Module/Parameter/Linear/Embedding/LayerNorm/Dropout
    │   └── init.py        #   He / Xavier 初始化
    ├── rnn.py             # 手写 RNNCell + 多层 RNN
    ├── lstm.py            # 手写 LSTMCell + 多层 LSTM
    ├── cnn.py             # 手写 Conv2d / MaxPool2d
    ├── transformer.py     # 手写多头注意力 + 编码器
    └── test_models.py     # 梯度校验 + 训练演示
```

## 设计要点

**self 目录追求代码复用**：四个模型不是各自为政，而是全部建立在同一个
自研核心之上——

- `core.tensor` 提供一套自动求导算子（`matmul` / 广播加乘 / `softmax` /
  `transpose` / `reshape` / `slice` / `tanh` / `sigmoid` / `relu` 等），
  任何模型的前向都是这些算子的组合，反向传播自动完成；
- `core.nn` 提供 `Module` / `Parameter` 与 `Linear` / `Embedding` /
  `LayerNorm` / `Dropout` 等基础层，四个模型通过组合这些层实现；
- 卷积、池化、层归一化等「融合算子」封装在 `core.functional`，由 CNN /
  Transformer 直接调用，避免重复实现梯度。

因此各模型文件本身很薄，核心逻辑集中在 `core` 中，一处实现、四处复用。

## 运行

**self（无需任何第三方依赖，可直接运行）**：

```bash
cd NN
python3 -m self.test_models
```

测试会对四个模型各做一次前向 + 反向、用数值微分抽查解析梯度、再训练数步
验证损失下降。

**torch（需先安装 PyTorch）**：

```bash
pip install torch
cd NN
python3 -m torch.test_models
```

## 模型接口

两套实现的类接口保持一致，便于对照：

| 模型 | 构造参数 | 输入 | 输出 |
|------|----------|------|------|
| `RNN` | `input_size, hidden_size, num_layers, num_classes` | `(B, T, D)` | `(B, num_classes)` 或 `(B, H)` |
| `LSTM` | 同上 | `(B, T, D)` | 同上 |
| `CNN` | `in_channels, num_classes, image_size` | `(B, C, H, W)` | `(B, num_classes)` |
| `Transformer` | `vocab_size, d_model, n_heads, d_ff, num_layers, num_classes, ...` | `(B, T)` 整数索引 | `(B, num_classes)` |
