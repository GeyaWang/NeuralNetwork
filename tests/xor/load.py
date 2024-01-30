from nn.models import Sequential
import numpy as np


def get_input(model):
    while True:
        input_ = input('\nEnter an input:\n')
        output = model.predict(np.matrix(np.fromstring(input_, dtype=float, sep=',')))
        print(f'{input_} -> {output}')


def main():
    model = Sequential.load('training')
    get_input(model)


if __name__ == '__main__':
    main()
