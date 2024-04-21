import numpy as np
from numpy.typing import NDArray

def get_gamma(U0: float, alpha: float, ngrids: int) -> NDArray[np.float64]:
    return U0 / np.cosh(alpha * np.flip(np.arange(ngrids))) ** 2