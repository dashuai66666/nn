"""卷积神经网络（CNN）：自实现的 Conv2d / MaxPool2d 组成的小型分类网络。

卷积与池化的前向、反向都封装在 ``core.functional`` 中，这里仅做模块化封装，
并与 ``Linear`` / ``ReLU`` 组合成完整网络，体现层次化复用。
"""

from .core import nn, init
from .core import functional as F
from .core.nn import Module, Linear, Parameter
from .core.tensor import zeros


class Conv2d(Module):
    """二维卷积层，权重形状 (out_channels, in_channels, kH, kW)。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = Parameter(init.kaiming_uniform(
            [out_channels, in_channels, kernel_size, kernel_size], fan_in))
        self.bias = Parameter(zeros([out_channels]))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


class MaxPool2d(Module):
    """二维最大池化层。"""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride)


class AvgPool2d(Module):
    """二维平均池化层（LeNet 等经典网络使用）。"""

    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride)


class CNN(Module):
    """两个卷积块 + 两个全连接层的小型 CNN（默认 28x28 输入，如 MNIST）。"""

    def __init__(self, in_channels=1, num_classes=10, image_size=28):
        super().__init__()
        self.conv1 = Conv2d(in_channels, 8, 3, padding=1)
        self.pool1 = MaxPool2d(2)
        self.conv2 = Conv2d(8, 16, 3, padding=1)
        self.pool2 = MaxPool2d(2)
        feat = image_size // 4                       # 两次 2x2 池化后的边长
        self.fc1 = Linear(16 * feat * feat, 64)
        self.fc2 = Linear(64, num_classes)

    def forward(self, x):
        # x: (N, C, H, W)
        x = self.pool1(self.conv1(x).relu())
        x = self.pool2(self.conv2(x).relu())
        x = x.reshape(x.shape[0], -1)                # 展平 (N, C*H*W)
        x = self.fc1(x).relu()
        return self.fc2(x)
