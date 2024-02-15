from ._template import Layer, Optimiser, Loss
from .branch import Branch
from .agent import Agent
import numpy as np
from typing import Literal


class Sequential:
    def __init__(self, layers: list[Layer | Branch] = None) -> None:
        self.layers: list[Layer | Branch] = layers
        self.optimiser: Optimiser = None
        self.loss: Loss = None
        self.prev_output_shape: tuple = None
        self.is_branched: bool = False

    def add(self, item: Layer | Branch):
        """Add layer to list, init layer"""

        assert not self.is_branched, 'Cannot add further layers after branch'

        if isinstance(item, Layer):
            # set input shape
            if item.input_shape is None:
                assert self.prev_output_shape is not None, 'Input shape of first layer not specified'
                item.input_shape = self.prev_output_shape

            item.init()
            self.prev_output_shape = item.output_shape

        else:  # isinstance(item, Branch)
            self.is_branched = True

            for branch in item.branches:
                prev_output_shape = self.prev_output_shape

                for layer in branch:
                    if layer.input_shape is None:
                        layer.input_shape = prev_output_shape

                    layer.init()
                    prev_output_shape = layer.output_shape

        self.layers.append(item)

    def summary(self) -> None:
        ...

    def compile(self, optimiser: Optimiser, loss: Loss = None) -> None:
        ...

    def get_loss_func(self) -> Loss:
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        ...

    def forward_propagation(self, x: np.ndarray) -> np.ndarray:
        ...

    def backward_propagation(self, y: np.ndarray) -> np.ndarray:
        ...

    def train_step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

    def train_agent(self, agent: Agent, *args) -> None:
        ...

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1,
            epochs: int = 1,
            verbose: Literal[0, 1] = 1,
            running_mean_size: int = 100,
            save_filepath: str = None,
            metrics: list[Literal['accuracy']] = None
    ) -> None:
        ...

    def clone(self) -> Sequential:
        ...

    def save(self, filepath: str, verbose: Literal[0, 1] = 1) -> None:
        ...

    @classmethod
    def load(cls, filepath: str, verbose: Literal[0, 1] = 1) -> Sequential:
        ...
