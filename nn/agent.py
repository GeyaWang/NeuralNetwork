from .utils.progess_bar import ProgressBar
import numpy as np
from typing import Literal
from collections import deque
from abc import ABC, abstractmethod
import random
import time
import sys


class Agent(ABC):
    def __init__(self):
        self.model = None  # model initiated upon training

    @abstractmethod
    def train(self, *args, **kwargs):
        pass


class TrainingAgent(Agent):
    def __init__(
            self,
            batch_size: int = 1,
            epochs: int = 1,
            verbose: Literal[0, 1] = 1,
            running_mean_size: int = 100,
            save_filepath: str = None,
            metrics: list[Literal['accuracy']] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.running_mean_size = running_mean_size
        self.save_filepath = save_filepath

        if metrics is None:
            self.metrics = []
        else:
            self.metrics = metrics

        # init variables
        self.err_list = deque(maxlen=self.running_mean_size)
        self.time_list = deque(maxlen=self.running_mean_size)

        self.acc_list = None
        self.progress_bar = None

        if 'accuracy' in self.metrics:
            self.acc_list = deque(maxlen=self.running_mean_size)

    def train(self, x, y):
        assert x.shape[0] == y.shape[0], 'Input and output data must have the same batch size'
        total_steps = x.shape[0]
        total_steps_round = (total_steps // self.batch_size) * self.batch_size

        for epoch in range(self.epochs):
            # shuffle training_data
            zip_ = list(zip(x, y))
            random.shuffle(zip_)
            x_train, y_train = zip(*zip_)
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            epoch_start_time = time.perf_counter()

            iter_ = range(total_steps // self.batch_size)
            if self.verbose == 1:
                self.progress_bar = ProgressBar()
                self.progress_bar.prefix = f'Epoch: {epoch} - 0/{total_steps_round} '
                iter_ = self.progress_bar(iter_)
                print()

            for i in iter_:
                step_low = i * self.batch_size
                step_high = step_low + self.batch_size
                y_true = y_train[step_low: step_high]

                t1 = time.perf_counter()
                try:
                    y_pred = self.model.train_step(x_train[step_low: step_high], y_true)

                # catch keyboard interrupt
                except KeyboardInterrupt:
                    print('\n\nForcefully shut down by user')
                    sys.exit()
                t2 = time.perf_counter()

                # statistics
                self.time_list.append(t2 - t1)
                time_mean = np.mean(self.time_list)
                err = self.model.get_loss_func().func(y_true, y_pred)
                self.err_list.append(err)

                if 'accuracy' in self.metrics:
                    self.acc_list.append(np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)))  # accuracy of batch

                # display data
                if self.verbose == 1:
                    # ETA and loss
                    prefix = f'Epoch: {epoch} - {step_high}/{total_steps_round} '
                    suffix = f' - ETA: {(total_steps_round - step_high) * time_mean / self.batch_size:.1f}s - loss: {np.mean(self.err_list):.4f}'

                    if 'accuracy' in self.metrics:
                        suffix += f' - accuracy: {np.mean(self.acc_list):.4f}'

                    self.progress_bar.prefix = prefix
                    self.progress_bar.suffix = suffix

            # display epoch data
            if self.verbose == 1:
                epoch_time = time.perf_counter() - epoch_start_time
                suffix = f' - {epoch_time:.1f}s {epoch_time / total_steps_round:.2f}s/step - loss: {np.mean(self.err_list):.4f}'

                if 'accuracy' in self.metrics:
                    suffix += f' - accuracy: {np.mean(self.acc_list):.4f}'

                self.progress_bar.set_end_txt(suffix=suffix)
                print()

            # save model
            if self.save_filepath is not None:
                self.model.save(self.save_filepath, verbose=self.verbose)
