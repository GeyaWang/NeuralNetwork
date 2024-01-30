import numpy as np

from .template import Loss


class MeanSquaredError(Loss):
    @staticmethod
    def func(y_true, y_pred):
        return float(np.mean(np.power(y_true - y_pred, 2)))

    @staticmethod
    def func_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    @staticmethod
    def func(y_true, y_pred):
        return float(np.sum(-y_true * np.log10(y_pred)))

    @staticmethod
    def func_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -(y_true / y_pred)
