"""卷积神经网络（CNN）：基于 PyTorch ``nn.Conv2d`` / ``nn.MaxPool2d`` 构建。

结构上与 ``self.cnn`` 完全对应：两个「卷积 + ReLU + 池化」块，接展平与两个
全连接层，默认面向 28x28 的单通道图像（如 MNIST）。
"""

import torch.nn as nn


class CNN(nn.Module):
    """两个卷积块 + 两个全连接层的小型 CNN。"""

    def __init__(self, in_channels=1, num_classes=10, image_size=28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = image_size // 4                       # 两次 2x2 池化后的边长
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * feat * feat, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (N, C, H, W)
        return self.classifier(self.features(x))
