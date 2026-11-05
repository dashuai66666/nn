# SOTA 复现示例

用 ``self``（纯 Python 从零实现的框架）复现四种网络的"封神之作"：

| 目录 | 网络 | 复现的工作 | 论文/出处 |
|------|------|-----------|-----------|
| `lenet5/` | CNN | LeNet-5 手写数字识别 | LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998 |
| `char_rnn/` | RNN | 字符级语言模型与文本生成 | Karpathy, *The Unreasonable Effectiveness of Recurrent Neural Networks*, 2015 |
| `lstm/` | LSTM | 长时滞（long time lag）任务 | Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997 |
| `transformer/` | Transformer | 完整编码器-解码器 Transformer | Vaswani et al., *Attention Is All You Need*, 2017 |

所有示例只依赖 `NN/self`，不引入任何第三方库，可直接运行。

## 运行

在仓库根目录执行（纯 Python 实现较慢，单个示例约 20 秒 ~ 1 分钟）：

```bash
python3 -m NN.example.sota.lenet5.train        # LeNet-5 合成数字分类
python3 -m NN.example.sota.char_rnn.train      # 字符级 RNN 生成文本
python3 -m NN.example.sota.lstm.train          # LSTM 长时滞记忆对比 RNN
python3 -m NN.example.sota.transformer.train   # Transformer 复制任务（seq2seq）
```

## 各示例说明

### lenet5 —— LeNet-5（CNN）
忠实复现 LeNet-5 的五层结构：`C1(6,5x5)→S2(avg)→C3(16,5x5)→S4(avg)→
C5(120,5x5)→F6(84)→输出`，使用 tanh 与平均池化（为此在 ``self`` 中新增了
``AvgPool2d``）。因本机无 MNIST，用"每类一张原型图 + 高斯噪声"构造合成数字
做分类，演示端到端训练与评估。

### char_rnn —— 字符级 RNN（RNN）
复现 Karpathy 的字符级语言模型：嵌入 + 多层 ``RNNCell`` + 输出到词表，逐
时间步预测下一字符，训练后按温度采样续写文本。

### lstm —— 长短期记忆（LSTM）
复现 LSTM 门控结构（输入门/遗忘门/候选门/输出门 + 细胞状态），并在原论文
关心的"长时滞"任务上演示：回忆序列开头的关键 token。同时训练普通 RNN 作对
比。注意：纯 Python 速度有限，示例用时滞 T=20；更长的时滞下 LSTM 门控优势
更明显（可自行调大 `train.py` 中的 `T`）。

### transformer —— Attention Is All You Need（Transformer）
复现论文的完整编码器-解码器结构：缩放点积注意力、多头注意力、位置编码、
掩码自注意力（因果掩码）、编码器-解码器交叉注意力、残差 + 层归一化。
在"复制任务"上以 teacher forcing 训练，贪心解码展示结果。

## 与 self 核心的关系

四个示例都是薄封装——网络结构复用 ``NN/self`` 中的 ``Linear`` / ``Embedding`` /
``LayerNorm`` / ``Dropout`` / ``Conv2d`` / ``MultiHeadAttention`` / ``RNN`` /
``LSTM`` 等，训练复用 ``_common.py`` 的 ``SGD`` / ``Adam`` 优化器，体现了
"一处实现、多处复用"的设计。
