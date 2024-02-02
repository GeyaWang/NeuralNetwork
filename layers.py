from .template import Layer, TrainableLayer, TrainingOnlyLayer, Initialise
from .initializers import GlorotUniform, Zeros
import numpy as np
from typing import Literal
from scipy.signal import correlate2d, convolve2d


class Activation(Layer):
    def __init__(self, activation: Layer):
        super().__init__()

        self.activation = activation
        self.display = activation.display

    def forward(self, X):
        return self.activation.forward(X)

    def backward(self, dY):
        return self.activation.backward(dY)


class Dense(TrainableLayer):
    def __init__(self, units: int, input_shape: tuple[int] = None, kernel_initializer: Initialise = GlorotUniform(), bias_initializer: Initialise = Zeros()):
        super().__init__(input_shape, (units,))

        self.weights = None
        self.bias = None

        self.optimiser = None

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.cached_X = None

    def forward(self, X):
        self.cached_X = X

        return np.dot(X, self.weights) + self.bias

    def backward(self, dY):
        dW = np.dot(self.cached_X.T, dY)  # dE/dW = X.T * dE/dY
        dB = np.sum(dY, axis=0)  # dE/dB = dE/dY
        dX = np.dot(dY, self.weights.T)  # dE/dX = dE/dY * W.T

        # update parameters
        self.optimiser.update_params(self.weights, dW)
        self.optimiser.update_params(self.bias, dB)

        return dX

    def init_params(self):
        kernel_shape = (self.input_shape[0], self.output_shape[0])
        bias_shape = self.output_shape[0]

        fan_in = np.prod(self.input_shape)
        fan_out = np.prod(self.output_shape)

        self.weights = self.kernel_initializer(kernel_shape, fan_in=fan_in, fan_out=fan_out)
        self.bias = self.bias_initializer(bias_shape, fan_in=fan_in, fan_out=fan_out)

    @property
    def params(self):
        return self.weights.size + self.bias.size


class Conv2D(TrainableLayer):
    def __init__(
            self,
            filters,
            kernel_size: int | tuple[int, int],
            input_shape: tuple[int, int, int] = None,
            strides=(1, 1),
            padding: Literal['valid', 'same'] = 'valid',
            kernel_initializer: Initialise = GlorotUniform(),
            bias_initializer: Initialise = Zeros()
    ):
        super().__init__(input_shape, None)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if padding == 'valid':
            self.pad = (0, 0)
        else:  # same
            self.pad = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.filters = filters
        self.strides = strides
        self.padding = padding

        self.weights = None
        self.bias = None

        self.optimiser = None

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.cached_X = None
        self.cached_X_pad = None

    def forward(self, X):
        self.cached_X = X

        N, _, _, C1, = X.shape
        H2, W2, C2 = self.output_shape

        # initiate output
        Y = np.zeros((N, H2, W2, C2))

        for n in range(N):
            for c2 in range(C2):
                for c1 in range(C1):
                    Y[n, :, :, c2] += correlate2d(X[n, :, :, c1], self.weights[:, :, c1, c2], mode=self.padding)
        return Y

    def backward(self, dY):
        dW = np.zeros(self.weights.shape)
        dB = np.zeros(self.bias.shape)
        dX = np.zeros(self.cached_X.shape)

        N, _, _, C1, = self.cached_X.shape
        k1, k2, _, C2 = self.weights.shape

        for n in range(N):
            for c2 in range(C2):
                dB[c2] = np.sum(dY[:, :, :, c2])

                for c1 in range(C1):
                    if self.padding == 'valid':
                        dX[n, :, :, c1] += convolve2d(dY[n, :, :, c2], self.weights[:, :, c1, c2], mode='full')
                        dW[:, :, c1, c2] += correlate2d(self.cached_X[n, :, :, c1], dY[n, :, :, c2], mode='valid')
                    elif self.padding == 'same':
                        dX[n, :, :, c1] += convolve2d(dY[n, :, :, c2], self.weights[:, :, c1, c2], mode='same')
                        dW[:, :, c1, c2] += correlate2d(self.cached_X[n, :, :, c1], dY[n, :, :, c2], mode='same')

        # update parameters
        self.optimiser.update_params(self.weights, dW)
        self.optimiser.update_params(self.bias, dB)

        return dX

    def init_params(self):
        kernel_shape = (*self.kernel_size, self.input_shape[2], self.filters)
        bias_shape = (self.filters,)

        fan_in = np.prod(self.input_shape)
        fan_out = np.prod(self.output_shape)

        self.weights = self.kernel_initializer(kernel_shape, fan_in=fan_in, fan_out=fan_out)
        self.bias = self.bias_initializer(bias_shape, fan_in=fan_in, fan_out=fan_out)

    @property
    def params(self):
        return self.weights.size + self.bias.size

    @property
    def output_shape(self):
        height = (self.input_shape[0] - self.kernel_size[0] + 2 * self.pad[0]) // self.strides[0] + 1
        width = (self.input_shape[1] - self.kernel_size[1] + 2 * self.pad[1]) // self.strides[1] + 1
        return height, width, self.filters


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.cached_X_shape = None

    def forward(self, X):
        self.cached_X_shape = X.shape

        n = X.shape[0]
        p = np.prod(X.shape[1:])

        return np.reshape(X, (n, p))

    def backward(self, dY):
        return np.reshape(dY, self.cached_X_shape)

    @property
    def output_shape(self):
        return (np.prod(self.input_shape),)


class Dropout(TrainingOnlyLayer):
    def __init__(self, p: float):
        super().__init__()

        self.p = p

        self.cached_X_mask = None

    def forward(self, X):
        self.cached_X_mask = np.random.rand(*X.shape) < (1 - self.p)
        return (X * self.cached_X_mask) / (1 - self.p)

    def backward(self, dY):
        return (dY * self.cached_X_mask) / (1 - self.p)
