import numpy as np
from ._template import ActivationLayer


class ReLU(ActivationLayer):
    def func(self, X):
        return np.maximum(0, X)

    def func_prime(self, X):
        X[X >= 0] = 1
        X[X < 0] = 0
        return X


class SoftMax(ActivationLayer):
    def __init__(self):
        super().__init__()
        self.cached_output = None

    def func(self, X):
        max_val = np.max(X, axis=1, keepdims=True)
        exp_x = np.exp(X - max_val)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def func_prime(self, Y):
        _, n = Y.shape
        tmp = np.repeat(Y[:, :, np.newaxis], n, axis=2)
        return tmp.transpose(0, 2, 1) * (np.identity(n) - tmp)

    def forward(self, X):
        self.cached_output = self.func(X)
        return self.cached_output

    def backward(self, dY):
        tmp = self.func_prime(self.cached_output)
        dX = np.einsum('ijk,ik->ij', tmp, dY)
        return dX
