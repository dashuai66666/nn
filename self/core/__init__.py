"""self 目录的共享核心：自动求导张量 + 基础层。

对外暴露最常用的符号，模型文件只需 ``from .core import nn, F, Tensor`` 即可。
"""

from .tensor import Tensor, set_seed
from . import tensor
from . import functional as F
from . import nn
from . import init

__all__ = ["Tensor", "nn", "F", "init", "tensor", "set_seed"]
