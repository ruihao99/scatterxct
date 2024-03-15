import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt

def view_wavepacket(R: ArrayLike, psi: ArrayLike) -> plt.Figure:
    """View the wavepacket in the real space."""
    fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
    nstates = psi.shape[1]
    for ii in range(nstates):
        ax.plot(R, np.abs(psi[:, ii])**2, label=f"State {ii}")
    ax.set_xlabel("R")
    ax.set_ylabel("Nuclear Probability Density")
    ax.legend()
    return fig