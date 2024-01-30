from nn.models import Sequential
from nn.layers import Dense, Activation
from nn.activations import ReLU
from nn.optimisers import Adam
from nn.losses import MeanSquaredError
import numpy as np

BATCHES = 1000000


def train(model):
    inputs = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    outputs = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    zipped_list = list(zip(inputs.repeat(BATCHES // 4, axis=0), outputs.repeat(BATCHES // 4, axis=0)))
    np.random.shuffle(zipped_list)
    x_train, y_train = zip(*zipped_list)

    model.fit(np.array(x_train), np.array(y_train), batch_size=32, running_mean_err=1000)


def main():
    model = Sequential()

    model.add(Dense(16, input_shape=(2,)))
    model.add(Activation(ReLU()))
    model.add(Dense(16))
    model.add(Activation(ReLU()))
    model.add(Dense(1))
    model.compile(Adam(), MeanSquaredError())
    model.summary()

    train(model)

    model.save('training')


if __name__ == '__main__':
    main()
