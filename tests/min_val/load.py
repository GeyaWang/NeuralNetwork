from nn.models import Sequential
import numpy as np


def get_input(model):
    while True:
        input_ = np.array([np.fromstring(input('\nEnter an input:\n'), dtype=float, sep=',')])
        output = model.predict(input_)
        print(f'{input_} -> {output}')
        print(f'Predicted min: {input_[0, np.argmax(output)]}')


def main():
    model = Sequential.load('training.ptd')
    get_input(model)


if __name__ == '__main__':
    main()
