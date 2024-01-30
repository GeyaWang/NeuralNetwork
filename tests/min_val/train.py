from nn.models import Sequential
from nn.layers import Dense, Activation
from nn.activations import ReLU, SoftMax
from nn.optimisers import Adam, SGD
from nn.losses import CrossEntropy
import numpy as np


BATCH_SIZE = 100000
SAMPLES = 5


def main():
    model = Sequential()
    model.add(Dense(16, input_shape=(SAMPLES,)))
    model.add(Activation(ReLU()))
    model.add(Dense(16))
    model.add(Activation(ReLU()))
    model.add(Dense(SAMPLES))
    model.add(Activation(SoftMax()))
    model.compile(Adam(), CrossEntropy())
    model.summary()

    x_train = np.random.uniform(-1, 1, (BATCH_SIZE, SAMPLES))
    y_train = np.array(np.equal(np.repeat(np.min(x_train, axis=-1), SAMPLES).reshape(BATCH_SIZE, SAMPLES), x_train), dtype=np.float32)

    model.fit(x_train, y_train, batch_size=16, running_mean_err=5000)

    model.save('training.ptd')


if __name__ == '__main__':
    main()
