# %%
import numpy as np
from numpy.typing import ArrayLike

from typing import Union

def morse(
    R: Union[float, ArrayLike],
    k: float,
    a: float,
    r_e: float,
    offset: float = 0,
) -> ArrayLike:
    return k * (1 - np.exp(-a * (R - r_e)))**2 + offset

def d_morse_dR(
    R: Union[float, ArrayLike],
    k: float,
    a: float,
    r_e: float,
    *args,
    **kwargs,
) -> ArrayLike:
    return 2 * k * a * (1 - np.exp(-a * (R - r_e))) * np.exp(-a * (R - r_e))

# %%
def _debug_test():
    import matplotlib.pyplot as plt
    r = np.linspace(0.3, 5, 1000)
    V = morse(r, 1, 5, 0.5)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(r, V)
    ax.set_xlabel('R')
    ax.set_ylabel('V')
    plt.show()
# %%
if __name__ == "__main__":
    _debug_test()

# %%
