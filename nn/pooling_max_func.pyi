import numpy as np

def forward(
        X: np.ndarray,
        output_shape: tuple[int, ...],
        pool_size: tuple[int, int],
        strides: tuple[int, int],
        pad: tuple[int, int]
    ) -> np.ndarray:
    ...


def backward(
        X: np.ndarray,
        dY: np.ndarray,
        pool_size: tuple[int, int],
        strides: tuple[int, int],
        pad: tuple[int, int]
    ) -> np.ndarray:
    ...
