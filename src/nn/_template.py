from abc import ABC, abstractmethod
import numpy as np


class Initialise(ABC):
    @staticmethod
    @abstractmethod
    def __call__(shape, dtype=None, **kwargs) -> np.ndarray:
        pass


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def func(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def func_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class Optimiser(ABC):
    @abstractmethod
    def update_params(self, param: np.ndarray, err_grad: np.ndarray) -> None:
        pass


class Layer(ABC):
    display: str = ''
    params: int = 0
    input_shape = None
    output_shape = None

    def __init__(self) -> None:
        if self.display == '':
            self.display = self.__class__.__name__

    @abstractmethod
    def forward(self, X, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dY, **kwargs) -> np.ndarray:
        pass


class TrainableLayer(Layer, ABC):
    def __init__(self, input_shape: None | tuple[int, ...], output_shape: None | tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape
        self.output = output_shape

    @abstractmethod
    def init_params(self) -> None:
        pass

    @property
    @abstractmethod
    def params(self) -> int:
        pass

    @property
    def output_shape(self):
        return self.output


class TrainingOnlyLayer(Layer, ABC):
    pass


class ActivationLayer(Layer, ABC):
    def __init__(self):
        super().__init__()
        self.cached_input = None

    @abstractmethod
    def func(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def func_prime(self, X: np.ndarray) -> np.ndarray:
        pass

    def forward(self, X) -> np.ndarray:
        self.cached_input = X
        return self.func(X)

    def backward(self, dY) -> np.ndarray:
        return dY * self.func_prime(self.cached_input)
