from ._template import TrainingOnlyLayer, TrainableLayer


class Branch:
    def __init__(self, n):
        self.branches = [[] for _ in range(n)]

        self.optimiser = None
        self.losses = [None for _ in range(n)]
        self.prev_output_shape = None  # used in adding layers

        self.active_branch = None

    def add(self, layer, i):
        """Add layer to list"""

        self.branches[i].append(layer)

    def summary(self, i):
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
        for layer in self.branches[i]:
            type_ = layer.display
            params = layer.params
            total_params += params
            output_shape = (None, *layer.output_shape)

            print(f'{row_format.format(type_, str(output_shape), str(params))}\n')

        print(f'{separator2}\n'
              f'Total params: {total_params}\n'
              f'{separator1}\n')

    def compile(self, optimiser, loss, i):
        self.optimiser = optimiser
        self.losses[i] = loss

        for layer in self.branches[i]:
            if isinstance(layer, TrainableLayer):
                layer.optimiser = self.optimiser

    def predict(self, x, i=None):
        if i is None:
            output = []
            for branch in self.branches:
                layer_x = x
                for layer in branch:
                    # ignore layer if training only
                    if not isinstance(layer, TrainingOnlyLayer):
                        layer_x = layer.forward(layer_x)
                output.append(layer_x)
            return output

        else:
            for layer in self.branches[i]:
                # ignore layer if training only
                if not isinstance(layer, TrainingOnlyLayer):
                    x = layer.forward(x)
            return x

    def forward_propagation(self, x):
        assert self.active_branch is not None, "Please specify an active branch for training"

        for layer in self.branches[self.active_branch]:
            x = layer.forward(x)
        return x

    def backward_propagation(self, y):
        assert self.active_branch is not None, "Please specify an active branch for training"

        for layer in reversed(self.branches[self.active_branch]):
            y = layer.backward(y)
        return y
