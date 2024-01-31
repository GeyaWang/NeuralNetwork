import mnist
from nn.models import Sequential
from nn.layers import Dense, Conv2D, Flatten, Activation, Dropout
from nn.activations import ReLU, SoftMax
from nn.losses import CrossEntropy
from nn.optimisers import Adam
import numpy as np


def main():
    x, y, x_test, y_test = mnist.mnist('MNIST')
    n, H, W = x.shape

    x_train = np.reshape(x, (n, H, W, 1))
    y_train = np.eye(10)[y]

    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=(28, 28, 1)))
    model.add(Activation(ReLU()))
    model.add(Conv2D(32, 3))
    model.add(Activation(ReLU()))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation(SoftMax()))
    model.summary()
    model.compile(Adam(), CrossEntropy())

    model.fit(x_train, y_train, batch_size=32, running_mean_err=100)

    model.save('mnist_training')


if __name__ == '__main__':
    main()
