from ._template import Layer, TrainableLayer, TrainingOnlyLayer, Initialize, Optimiser, ActivationLayer
from .initializers import GlorotUniform, Zeros
import numpy as np
from typing import Literal


class Activation(Layer):
    def __init__(self, activation: ActivationLayer) -> None:
        self.activation: Layer = activation

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...


class Dense(TrainableLayer):
    def __init__(
            self,
            units: int,
            input_shape: tuple[int] = None,
            kernel_initializer: Initialize = GlorotUniform(),
            bias_initializer: Initialize = Zeros()
    ):
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        self.optimiser: Optimiser = None
        self.kernel_initializer: Initialize = kernel_initializer
        self.bias_initializer: Initialize = bias_initializer
        self.cached_X: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...

    def init(self) -> None:
        ...

    @property
    def params(self) -> int:
        ...


class Conv2D(TrainableLayer):
    def __init__(
            self,
            filters: int,
            kernel_size: int | tuple[int, int],
            input_shape: tuple[int, int, int] = None,
            padding: Literal['valid', 'same'] = 'valid',
            kernel_initializer: Initialize = GlorotUniform(),
            bias_initializer: Initialize = Zeros()
    ) -> None:
        self.kernel_size: tuple[int, int] = kernel_size
        self.pad: tuple[int, int] = None
        self.filters: int = filters
        self.padding: Literal['valid', 'same']  = padding
        self.weights: np.ndarray = None
        self.bias: np.ndarray = None
        self.optimiser: Optimiser = None
        self.kernel_initializer: Initialize = kernel_initializer
        self.bias_initializer: Initialize = bias_initializer
        self.cached_X: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...

    def init(self) -> None:
        ...

    @property
    def params(self) -> int:
        ...


class MaxPooling2D(Layer):
    def __init__(
            self,
            pool_size: int | tuple[int, int] = (2, 2),
            strides: None | int | tuple[int, int] = None,
            padding: Literal['valid', 'same'] = 'valid'
    ):
        self.pool_size: tuple[int, int] = pool_size
        self.strides: tuple[int, int] = strides
        self.padding: Literal['valid', 'same'] = padding
        self.pad: tuple[int, int] = None
        self.cached_X: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...

    def init(self) -> None:
        pass


class AveragePooling2D(MaxPooling2D):
    def __init__(
            self,
            pool_size: int | tuple[int, int] = (2, 2),
            strides: None | int | tuple[int, int] = None,
            padding: Literal['valid', 'same'] = 'valid'
    ):
        self.pool_size: tuple[int, int] = pool_size
        self.strides: tuple[int, int] = strides
        self.padding: Literal['valid', 'same'] = padding
        self.cached_X: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...


class Reshape(Layer):
    def __init__(self, shape) -> None:
        self.shape: tuple[int, ...] = shape
        self.cached_X_shape: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...

    def init(self) -> None:
        pass


class Flatten(Layer):
    def __init__(self) -> None:
        self.cached_X_shape: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...

    def init(self) -> None:
        pass


class Dropout(TrainingOnlyLayer):
    def __init__(self, p: float) -> None:
        self.p: float = p
        self.cached_X_mask: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    def backward(self, dY: np.ndarray) -> np.ndarray:
        ...
