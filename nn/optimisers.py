from .template import Optimiser
import numpy as np


class Adam(Optimiser):
    display = 'Adam'

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-07):
        self.learning_rate = learning_rate
        self.beta1_0 = self.beta1 = beta1
        self.beta2_0 = self.beta2 = beta2
        self.epsilon = epsilon

        self.params = []

    def update_params(self, param, err_grad):
        # update moments
        param.m = self.beta1_0 * param.m + (1 - self.beta1_0) * err_grad
        param.v = self.beta2_0 * param.v + (1 - self.beta2_0) * np.square(err_grad)

        current_alpha = self.learning_rate * np.sqrt(1 - self.beta2) / (1 - self.beta1)

        # update parameter
        param -= current_alpha * param.m / (np.sqrt(param.v) + self.epsilon)

        # decay beta
        self.beta1 = self.beta1_0 * self.beta1
        self.beta2 = self.beta2_0 * self.beta2


class SGD(Optimiser):
    display = 'SGD'

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

        self.params = []

    def update_params(self, param, err_grad):
        param -= self.learning_rate * err_grad
