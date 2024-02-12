from ._template import Layer, Optimiser, Loss, TrainableLayer, TrainingOnlyLayer
from .agent import TrainingAgent, Agent
import numpy as np
from typing import Literal
import os
import pickle
from copy import deepcopy

FILE_EXTENSION = 'ptd'


class Sequential:
    def __init__(self, layers: list[Layer] = None):
        if layers is None:
            self.layers: list[Layer] = []
        else:
            self.layers = layers

        self.optimiser = None
        self.loss = None

        self.prev_output_shape = None  # used in adding layers

    def add(self, layer: Layer):
        """Add layer to list, init layer"""

        # set input shape
        if layer.input_shape is None:
            assert self.prev_output_shape is not None, 'Input shape of first layer not specified'
            layer.input_shape = self.prev_output_shape

        layer.init()
        self.prev_output_shape = layer.output_shape

        self.layers.append(layer)

    def summary(self):
        """Prints tabular summary of model"""

        column_size = 30
        header = ["Layer (type)", "Output size", "Param #"]

        separator1 = '-' * column_size * len(header)
        separator2 = '=' * column_size * len(header)

        row_format = f"{{:<{column_size}.{column_size}}}" * (len(header))

        print(
            f'{separator1}\n'
            f'{row_format.format(*header)}\n'
            f'{separator2}'
        )

        total_params = 0
        for layer in self.layers:
            type_ = layer.display

            params = layer.params
            total_params += params

            print(f'{row_format.format(type_, str((None, *layer.output_shape)), str(params))}\n')

        print(f'{separator2}\n'
              f'Total params: {total_params}\n'
              f'{separator1}\n')

    def compile(self, optimiser: Optimiser, loss: Loss):
        self.optimiser = optimiser
        self.loss = loss

        for layer in self.layers:
            if isinstance(layer, TrainableLayer):
                layer.optimiser = self.optimiser

    def predict(self, x):
        for layer in self.layers:
            # ignore layer if training only
            if not isinstance(layer, TrainingOnlyLayer):
                x = layer.forward(x)
        return x

    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_propagation(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y

    def train_step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert self.optimiser is not None and self.loss is not None, 'Model must be compiled before training'

        # forward propagation
        y_pred = self.forward_propagation(x)

        # calculate error and error gradient
        dY = self.loss.func_prime(y, y_pred)

        # backward propagation
        self.backward_propagation(dY)
        return y_pred

    def train_agent(self, agent: Agent, *args):
        agent.model = self
        agent.train(*args)

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
    ):
        """Train over batch data using agent"""

        agent = TrainingAgent(batch_size, epochs, verbose, running_mean_size, save_filepath, metrics)
        self.train_agent(agent, x, y)

    def clone(self):
        """Returns deep copy of self"""
        return deepcopy(self)

    def save(self, filepath: str, verbose: Literal[0, 1] = 1):
        # handel file extension cases
        extension = os.path.splitext(filepath)[1]
        if extension == '':
            filepath += f'.{FILE_EXTENSION}'
        else:
            assert extension == f'.{FILE_EXTENSION}', f'Please save to a file with the extension "{FILE_EXTENSION}"'

        model_data = {
            'layers': self.layers
        }

        with open(filepath, 'wb') as file:
            pickle.dump(model_data, file)

        if verbose == 1:
            print(f'Successfully saved file to {os.path.abspath(filepath)}')

    @classmethod
    def load(cls, filepath: str, verbose: Literal[0, 1] = 1):
        # handel file extension cases
        extension = os.path.splitext(filepath)[1]
        if extension == '':
            filepath += f'.{FILE_EXTENSION}'
        else:
            assert extension == f'.{FILE_EXTENSION}', f'Please load a valid file with the extension "{FILE_EXTENSION}"'

        with open(filepath, 'rb') as file:
            model_data = pickle.load(file)

        layers = model_data["layers"]

        if verbose == 1:
            print(f'Successfully loaded file from {os.path.abspath(filepath)}')

        return Sequential(layers=layers)
