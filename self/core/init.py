"""参数初始化：提供常见的 He / Xavier 均匀初始化与零初始化。

返回嵌套 list，供 ``Parameter`` 直接使用。
"""

import math

from .tensor import uniform, zeros


def kaiming_uniform(shape, fan_in, a=0.0):
    """He 初始化（适合 ReLU 族激活），a 为激活函数负半轴斜率。"""
    gain = math.sqrt(2.0 / (1.0 + a * a))
    bound = math.sqrt(3.0) * gain / math.sqrt(fan_in)
    return uniform(shape, -bound, bound)


def xavier_uniform(shape, fan_in, fan_out):
    """Xavier 初始化（适合 tanh / sigmoid 等对称激活）。"""
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, -bound, bound)


def zeros_(shape):
    return zeros(shape)
