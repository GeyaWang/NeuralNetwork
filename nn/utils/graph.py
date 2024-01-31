import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dataclasses import dataclass


@dataclass
class Plot:
    x: list
    y: list
    colour: str


class Graph:
    def __init__(self, plots: list[Plot], pause: float = 0.1):
        self.plots = plots
        self.pause = pause

        plt.close('all')
        matplotlib.use("TkAgg")
        plt.ion()
        plt.show()

    def update(self):
        for plot in self.plots:
            plt.plot(plot.x, plot.y, color=plot.colour)
        plt.show()
        plt.pause(self.pause)
