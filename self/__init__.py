"""self：纯 Python（仅标准库）从零实现的神经网络。

包含一个最小化的自动求导张量与基础层（``core``），以及在其上搭建的
RNN / CNN / LSTM / Transformer 四个模型，所有模型共享同一套底层算子。
"""

from .core import Tensor, nn, F, init, set_seed
from .rnn import RNN, RNNCell
from .lstm import LSTM, LSTMCell
from .cnn import CNN, Conv2d, MaxPool2d, AvgPool2d
from .transformer import (
    Transformer,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderBlock,
    PositionalEncoding,
)

__all__ = [
    "Tensor", "nn", "F", "init", "set_seed",
    "RNN", "RNNCell",
    "LSTM", "LSTMCell",
    "CNN", "Conv2d", "MaxPool2d", "AvgPool2d",
    "Transformer", "MultiHeadAttention", "FeedForward",
    "TransformerEncoderBlock", "PositionalEncoding",
]
