from numpy import ndarray


class Parameter(ndarray):
    """Parameter class to store weights and biases"""

    def __array_finalize__(self, *args):
        # momentum for adam optimiser
        self.m = 0
        self.v = 0
