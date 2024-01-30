from keras.datasets import mnist
from keras.utils import to_categorical
from nn.models import Sequential
from nn.layers import Dense, Conv2D, Flatten, Activation
from nn.activations import ReLU, SoftMax
from nn.losses import CrossEntropy
from nn.optimisers import Adam
import numpy as np


def main():
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    y_train = to_categorical(y_train)

    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=(28, 28, 1)))
    model.add(Activation(ReLU()))
    model.add(Conv2D(32, 3))
    model.add(Activation(ReLU()))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation(SoftMax()))
    model.summary()
    model.compile(Adam(), CrossEntropy())

    model.fit(x_train, y_train, batch_size=32, running_mean_err=1000)

    model.save('training')


if __name__ == '__main__':
    main()
