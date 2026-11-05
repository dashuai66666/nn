"""LeNet-5（LeCun et al., 1998）复现。

论文《Gradient-Based Learning Applied to Document Recognition》提出的卷积网络，
用于手写数字识别，是 CNN 的"封神之作"。这里用 ``self`` 逐层复现其结构：

    C1(6,5x5) -> tanh -> S2(avg 2x2) -> C3(16,5x5) -> tanh -> S4(avg 2x2)
    -> C5(120,5x5) -> tanh -> F6(84) -> tanh -> 输出(10)

注意：原论文 C5 为 5x5 卷积（等效全连接），这里保持一致；池化用平均池化。
"""

from NN.self.core.nn import Module, Linear
from NN.self.cnn import Conv2d, AvgPool2d


class LeNet5(Module):
    """32x32 单通道输入、10 类输出的 LeNet-5。"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2d(1, 6, 5)         # 32x32 -> 28x28
        self.pool1 = AvgPool2d(2)            # 28x28 -> 14x14
        self.conv2 = Conv2d(6, 16, 5)        # 14x14 -> 10x10
        self.pool2 = AvgPool2d(2)            # 10x10 -> 5x5
        self.conv3 = Conv2d(16, 120, 5)      # 5x5 -> 1x1（原论文的 C5）
        self.fc1 = Linear(120, 84)           # 原论文的 F6
        self.fc2 = Linear(84, num_classes)

    def forward(self, x):
        # x: (N, 1, 32, 32)
        x = self.pool1(self.conv1(x).tanh())
        x = self.pool2(self.conv2(x).tanh())
        x = self.conv3(x).tanh()             # (N, 120, 1, 1)
        x = x.reshape(x.shape[0], -1)        # (N, 120)
        x = self.fc1(x).tanh()
        return self.fc2(x)
