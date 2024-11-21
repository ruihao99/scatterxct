import numpy as np
from numpy.typing import NDArray

def get_gamma(U0: float, alpha: float, ngrids: int) -> NDArray[np.float64]:
    return U0 / np.cosh(alpha * np.flip(np.arange(ngrids))) ** 2

def get_amplitude_reduction_term(gamma: NDArray[np.float64], dt: float) -> NDArray[np.complex128]:
    return (1 - gamma * dt)[:, np.newaxis]