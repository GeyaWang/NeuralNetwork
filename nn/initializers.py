from ._template import Initialize
from ._parameter import Parameter
import numpy as np


class Zeros(Initialize):
    @staticmethod
    def __call__(shape, dtype=None, **kwargs):
        param = np.zeros(shape, dtype).view(Parameter)
        return param


class GlorotUniform(Initialize):
    @staticmethod
    def __call__(shape, dtype=None, fan_in=None, fan_out=None):
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape).view(Parameter)
