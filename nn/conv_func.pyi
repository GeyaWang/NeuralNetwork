import numpy as np
from typing import Literal

def forward(
        X: np.ndarray,
        K: np.ndarray,
        B: np.ndarray,
        padding: Literal['valid', 'same']
    ) -> np.ndarray:
    ...

def backward(
        X: np.ndarray,
        K: np.ndarray,
        dY: np.ndarray,
        padding: Literal['valid', 'same']
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
