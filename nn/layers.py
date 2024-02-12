from ._template import Layer, TrainableLayer, TrainingOnlyLayer
from .initializers import GlorotUniform, Zeros
from math import floor, ceil
import numpy as np
import conv_func


class Activation(Layer):
    def __init__(self, activation):
        super().__init__()

        self.activation = activation
        self.display = activation.display

    def forward(self, X):
        return self.activation.forward(X)

    def backward(self, dY):
        return self.activation.backward(dY)


class Dense(TrainableLayer):
    def __init__(self, units, input_shape=None, kernel_initializer=GlorotUniform(), bias_initializer=Zeros()):
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
        # calculate error gradients
        dW = np.dot(self.cached_X.T, dY)  # dE/dW = X.T * dE/dY
        dB = np.sum(dY, axis=0)  # dE/dB = dE/dY
        dX = np.dot(dY, self.weights.T)  # dE/dX = dE/dY * W.T

        # update parameters
        self.optimiser.update_params(self.weights, dW)
        self.optimiser.update_params(self.bias, dB)

        return dX

    def init(self):
        print(self.output_shape)

        # Init params
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
            kernel_size,
            input_shape=None,
            padding='valid',
            kernel_initializer=GlorotUniform(),
            bias_initializer=Zeros()
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
        self.padding = padding

        self.weights = None
        self.bias = None

        self.optimiser = None

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.cached_X = None

    def forward(self, X):
        self.cached_X = X

        return conv_func.forward(X, self.weights, self.bias, self.padding)

    def backward(self, dY):
        dX, dW, dB = conv_func.backward(self.cached_X, self.weights, dY, self.padding)

        # update parameters
        self.optimiser.update_params(self.weights, dW)
        self.optimiser.update_params(self.bias, dB)

        return dX

    def init(self):
        # Output shape
        height = self.input_shape[0] - self.kernel_size[0] + 2 * self.pad[0] + 1
        width = self.input_shape[1] - self.kernel_size[1] + 2 * self.pad[1] + 1
        self.output_shape = (height, width, self.filters)

        # Params
        kernel_shape = (*self.kernel_size, self.input_shape[2], self.filters)
        bias_shape = (self.filters,)

        fan_in = np.prod(self.input_shape)
        fan_out = np.prod(self.output_shape)

        self.weights = self.kernel_initializer(kernel_shape, fan_in=fan_in, fan_out=fan_out)
        self.bias = self.bias_initializer(bias_shape, fan_in=fan_in, fan_out=fan_out)

    @property
    def params(self):
        return self.weights.size + self.bias.size


class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__()

        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size

        if strides is None:
            self.strides = self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides

        self.padding = padding
        self.pad = None  # Initialised at self.init()

        self.cached_X = None

    def forward(self, X):
        self.cached_X = X

        N, H1, W1, C1 = X.shape
        H2, W2, _ = self.output_shape

        Y = np.zeros((N, H2, W2, C1))

        for n in range(N):
            for h in range(H2):
                for w in range(W2):
                    for c in range(C1):
                        max_ = -999

                        for i in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[0]):
                            for j in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[1]):
                                x_val = X[n, h * self.strides[0] + i - self.pad[0], w * self.strides[1] + j - self.pad[1], c]
                                if max_ < x_val:
                                    max_ = x_val

                        Y[n, h, w, c] = max_
        return Y

    def backward(self, dY):
        N, H1, W1, C1 = self.cached_X.shape
        H2, W2, _ = self.output_shape

        dX = np.zeros((N, H1, W1, C1))

        for n in range(N):
            for h in range(H2):
                for w in range(W2):
                    for c in range(C1):
                        rel_pos_x = None
                        rel_pos_y = None
                        max_ = -999

                        for i in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[0]):
                            for j in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[1]):
                                x_val = self.cached_X[n, h * self.strides[0] + i - self.pad[0], w * self.strides[1] + j - self.pad[1], c]
                                if x_val > max_:
                                    max_ = x_val
                                    rel_pos_x = i
                                    rel_pos_y = j

                        dX[n, h * self.strides[0] + rel_pos_x - self.pad[0], w * self.strides[1] + rel_pos_y - self.pad[1], c] = dY[n, h, w, c]
        return dX

    def init(self):
        if self.padding == 'same':
            height = ceil(self.input_shape[0] / self.strides[0])
            width = ceil(self.input_shape[1] / self.strides[1])

            # Calculate padding
            pad_x = int(((height - 1) * self.strides[0] + self.pool_size[0] - self.input_shape[0]) / 2)
            pad_y = int(((width - 1) * self.strides[1] + self.pool_size[1] - self.input_shape[1]) / 2)
            self.pad = (pad_x, pad_y)

        else:  # self.padding == 'valid'
            height = floor((self.input_shape[0] - self.pool_size[0]) / self.strides[0]) + 1
            width = floor((self.input_shape[1] - self.pool_size[1]) / self.strides[1]) + 1

            self.pad = (0, 0)

        self.output_shape = (height, width, self.input_shape[2])


class AveragePooling2D(MaxPooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__(pool_size, strides, padding)

    def forward(self, X):
        self.cached_X = X

        N, H1, W1, C1 = X.shape
        H2, W2, _ = self.output_shape

        Y = np.zeros((N, H2, W2, C1))

        for n in range(N):
            for h in range(H2):
                for w in range(W2):
                    for c in range(C1):
                        sum_ = 0

                        for i in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[0]):
                            for j in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[1]):
                                sum_ += X[n, h * self.strides[0] + i - self.pad[0], w * self.strides[1] + j - self.pad[1], c]

                        Y[n, h, w, c] = sum_ / (self.pool_size[0] * self.pool_size[1])
        return Y

    def backward(self, dY):
        N, H1, W1, C1 = self.cached_X.shape
        H2, W2, _ = self.output_shape

        dX = np.zeros((N, H1, W1, C1))

        for n in range(N):
            for h in range(H2):
                for w in range(W2):
                    for c in range(C1):
                        val = dY[n, h, w, c] / (self.pool_size[0] * self.pool_size[1])

                        for i in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[0]):
                            for j in range(max(0, self.pad[0] - h * self.strides[0]), self.pool_size[1]):
                                dX[n, h * self.strides[0] + i - self.pad[0], w * self.strides[1] + j - self.pad[1], c] = val
        return dX


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.cached_X_shape = None

    def forward(self, X):
        self.cached_X_shape = X.shape

        N = X.shape[0]
        P = np.prod(X.shape[1:])

        return np.reshape(X, (N, P))

    def backward(self, dY):
        return np.reshape(dY, self.cached_X_shape)

    def init(self):
        self.output_shape = (np.prod(self.input_shape),)


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
