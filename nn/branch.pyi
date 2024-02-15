import numpy as np
from ._template import Layer, Optimiser, Loss


class Branch:
    def __init__(self, n: int) -> None:
        self.branches: list[list[Layer]] = [[] for _ in range(n)]
        self.optimiser: Optimiser = None
        self.losses: list[Loss] = [None for _ in range(n)]
        self.prev_output_shape: tuple = None
        self.active_branch: int = None

    def add(self, layer: Layer, i: int) -> None:
        ...

    def summary(self, i: int) -> None:
        ...

    def compile(self, optimiser: Optimiser, loss: Loss, i: int) -> None:
        ...

    def predict(self, x: np.ndarray, i: int = None) -> np.ndarray | list[np.ndarray]:
        ...

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        ...

    def backward_propagation(self, y: np.ndarray) -> np.ndarray:
        ...
