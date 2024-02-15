from ._template import Layer, TrainableLayer, TrainingOnlyLayer
from .branch import Branch
from .agent import TrainingAgent, Agent
import os
import pickle
from copy import deepcopy

FILE_EXTENSION = 'ptd'


class Sequential:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.optimiser = None
        self.loss = None

        self.prev_output_shape = None  # used in adding layers
        self.is_branched = False

    def add(self, item):
        """Add layer to list and init layer"""

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
            if isinstance(layer, Branch):
                if layer.active_branch is not None:
                    for sub_layer in layer.branches[layer.active_branch]:
                        type_ = sub_layer.display
                        params = sub_layer.params
                        total_params += params
                        output_shape = (None, *sub_layer.output_shape)

                        print(f'{row_format.format(type_, str(output_shape), str(params))}\n')
                else:
                    print(f'{row_format.format("Branch", "None", "0")}\n')
            else:
                type_ = layer.display
                params = layer.params
                total_params += params
                output_shape = (None, *layer.output_shape)

                print(f'{row_format.format(type_, str(output_shape), str(params))}\n')

        print(f'{separator2}\n'
              f'Total params: {total_params}\n'
              f'{separator1}\n')

    def compile(self, optimiser, loss=None):
        self.optimiser = optimiser
        self.loss = loss

        for layer in self.layers:
            if isinstance(layer, TrainableLayer):
                layer.optimiser = self.optimiser

    def get_loss_func(self):
        if self.is_branched:
            branch = self.layers[-1]
            return branch.losses[branch.active_branch]
        else:
            return self.loss

    def predict(self, x):
        for item in self.layers:
            if isinstance(item, Branch):
                x = item.predict(x)

            # ignore layer if training only
            elif not isinstance(item, TrainingOnlyLayer):
                x = item.forward(x)

        return x

    def forward_propagation(self, x):
        for layer in self.layers:
            if isinstance(layer, Branch):
                x = layer.forward_propagation(x)
            else:
                x = layer.forward(x)
        return x

    def backward_propagation(self, y):
        for layer in reversed(self.layers):
            if isinstance(layer, Branch):
                y = layer.backward_propagation(y)
            else:
                y = layer.backward(y)
        return y

    def train_step(self, x, y):
        assert self.optimiser is not None, 'Model must be compiled before training'

        # forward propagation
        y_pred = self.forward_propagation(x)

        # calculate error and error gradient
        dY = self.get_loss_func().func_prime(y, y_pred)

        # backward propagation
        self.backward_propagation(dY)
        return y_pred

    def train_agent(self, agent: Agent, *args):
        agent.model = self
        agent.train(*args)

    def fit(
            self,
            x,
            y,
            batch_size=1,
            epochs=1,
            verbose=1,
            running_mean_size=100,
            save_filepath=None,
            metrics=None
    ):
        """Train over batch data using agent"""

        agent = TrainingAgent(batch_size, epochs, verbose, running_mean_size, save_filepath, metrics)
        self.train_agent(agent, x, y)

    def clone(self):
        """Returns deep copy of self"""
        return deepcopy(self)

    def save(self, filepath, verbose=1):
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
    def load(cls, filepath, verbose=1):
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
