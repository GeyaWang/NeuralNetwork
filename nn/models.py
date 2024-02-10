import random
from copy import deepcopy
from .utils.progess_bar import ProgressBar
from ._template import Layer, Optimiser, Loss, TrainableLayer, TrainingOnlyLayer
import numpy as np
from typing import Literal
from collections import deque
import time
import os
import pickle
import sys

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

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 1,
            epochs: int = 1,
            verbose: Literal[0, 1] = 1,
            running_mean_err: int = 100,
            save_filepath: str = None,
            metrics: list[Literal['accuracy']] = None
    ):
        """Train over batch of input data"""

        assert x.shape[0] == y.shape[0], 'Input and output data must have the same batch size'
        total_steps = x.shape[0]

        # init variables
        err_list = deque(maxlen=running_mean_err)
        time_list = deque(maxlen=running_mean_err)

        acc_list = None
        progress_bar = None

        if 'accuracy' in metrics:
            acc_list = deque(maxlen=running_mean_err)

        total_steps_round = (total_steps // batch_size) * batch_size

        for epoch in range(epochs):
            # shuffle training_data
            zip_ = list(zip(x, y))
            random.shuffle(zip_)
            x_train, y_train = zip(*zip_)
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            epoch_start_time = time.perf_counter()

            iter_ = range(total_steps // batch_size)
            if verbose == 1:
                progress_bar = ProgressBar()
                progress_bar.prefix = f'Epoch: {epoch} - 0/{total_steps_round} '
                iter_ = progress_bar(iter_)
                print()

            for i in iter_:
                step_low = i * batch_size
                step_high = step_low + batch_size
                y_true = y_train[step_low: step_high]

                t1 = time.perf_counter()
                try:
                    y_pred = self.train_step(x_train[step_low: step_high], y_true)

                # catch keyboard interrupt
                except KeyboardInterrupt:
                    print('\n\nForcefully shut down by user')
                    sys.exit()
                t2 = time.perf_counter()

                # statistics
                time_list.append(t2 - t1)
                time_mean = np.mean(time_list)
                err = self.loss.func(y_true, y_pred)
                err_list.append(err)
                acc_list.append(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)))  # accuracy of batch

                # display data
                if verbose == 1:
                    # ETA and loss
                    prefix = f'Epoch: {epoch} - {step_high}/{total_steps_round} '
                    suffix = f' - ETA: {(total_steps_round - step_high) * time_mean / batch_size:.1f}s - loss: {np.mean(err_list):.4f}'

                    if 'accuracy' in metrics:
                        suffix += f' - accuracy: {np.mean(acc_list):.4f}'

                    progress_bar.prefix = prefix
                    progress_bar.suffix = suffix

            # display epoch data
            if verbose == 1:
                epoch_time = time.perf_counter() - epoch_start_time
                suffix = f' - {epoch_time:.1f}s {epoch_time / total_steps_round:.1f}s/step - loss: {np.mean(err_list):.4f}'

                if 'accuracy' in metrics:
                    suffix += f' - accuracy: {np.mean(acc_list):.4f}'

                progress_bar.set_end_txt(suffix=suffix)
                print()

            # save model
            if save_filepath is not None:
                self.save(save_filepath, verbose=verbose)

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
