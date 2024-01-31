from copy import deepcopy
from .utils.progess_bar import ProgressBar
from .template import Layer, Optimiser, Loss, TrainableLayer, TrainingOnlyLayer
from matplotlib import pyplot as plt
import numpy as np
from typing import Literal
from collections import deque
import os
import pickle

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

        # set output shape
        if layer.output_shape is None:
            layer.output_shape = layer.input_shape
        self.prev_output_shape = layer.output_shape

        if isinstance(layer, TrainableLayer):
            layer.init_params()

        self.layers.append(layer)

    def summary(self):
        """Prints tabular summary of model"""

        column_size = 20
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

    def train_step(self, x: np.ndarray, y: np.ndarray):
        assert self.optimiser is not None and self.loss is not None, 'Model must be compiled before training'

        try:
            # forward propagation
            y_pred = self.forward_propagation(x)

            # calculate error and error gradient
            dY = self.loss.func_prime(y, y_pred)
            error = self.loss.func(y, y_pred)

            # backward propagation
            self.backward_propagation(dY)

        except KeyboardInterrupt:
            print('\n\nForcefully shut down by user')
            from sys import exit
            exit()

        return error

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 1, verbose: Literal[0, 1] = 1, graph: Literal[0, 1, 2] = 0, graph_filepath: str = 'graph', running_mean_err: int = 1):
        """Train over batch of input data"""

        assert x.shape[0] == y.shape[0], 'Input and output data must have the same batch size'
        total_batches = x.shape[0]

        # train batches
        if verbose == 1:
            # show progress bar
            progress_bar = ProgressBar()
            err_list = deque(maxlen=running_mean_err)
            prefix_len = len(str(total_batches)) * 2 + 2
            total_batches_round = (total_batches // batch_size) * batch_size
            progress_bar.prefix = f'0/{total_batches_round} '.rjust(prefix_len)

            iter_ = progress_bar(range(total_batches // batch_size))
        else:  # verbose == 0
            iter_ = range(total_batches // batch_size)

        if graph != 0:
            x_plot = []
            err_plot = []
            err_mean_plot = []

        for i in iter_:
            low = i * batch_size
            high = low + batch_size
            err = self.train_step(x[low: high], y[low: high])
            err_list.append(err)
            err_mean = np.mean(err_list)

            if graph != 0:
                x_plot.append(high)
                err_plot.append(err)
                err_mean_plot.append(err_mean)

            if verbose == 1:
                progress_bar.prefix = f'{high}/{total_batches_round} '.rjust(prefix_len)
                progress_bar.suffix = f' - loss: {err_mean:.4f}'

        if graph != 0:
            plt.plot(x_plot, err_plot, label='error')
            plt.plot(x_plot, err_mean_plot, label='mean error')
            plt.xlabel('Epoch')
            plt.legend()

            if graph == 1:
                plt.show()
            else:  # graph == 2
                plt.savefig(graph_filepath)

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
            print(f'Successfully saved file to {filepath}')

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

        if verbose == 0:
            print(f'Successfully loaded file from {filepath}')

        return Sequential(layers=layers)
