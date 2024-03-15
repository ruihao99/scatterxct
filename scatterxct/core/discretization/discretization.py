import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import fftfreq

from scatterxct.core.discretization.utils import estimate_R_lims

from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Discretization:
    """ class for the real space and time discretization for time-independent Hamiltonian. """
    R: np.ndarray 
    k: np.ndarray
    mass: float
    dt: float 
    H: ArrayLike
    
    @classmethod
    def from_diabatic_potentials(cls, R0: float, k0: float, hamiltonian: Callable[[ArrayLike], ArrayLike], mass: float, dt: float, scatter_region_center: float=0.0) -> "Discretization":
        R_lims, n_real_space_grid = estimate_R_lims(
            R0=R0, k0=k0, scatter_region_center=scatter_region_center
        )
        
        n: int = n_real_space_grid
        R = np.linspace(R_lims[0], R_lims[1], n)
        delta_R = R[1] - R[0]
        k = fftfreq(n, d=delta_R) * 2 * np.pi
        # k = fftfreq(n, d=delta_R) 
        # H = hamiltonian(R)
        H = np.array(
            [hamiltonian(Ri) for Ri in R]
        )
        if H.ndim == 2:
            raise ValueError("The Hamiltonian should be a 3D array")
        elif H.ndim == 3:
            if H.shape[0] == n:
                H = H.swapaxes(0, 2) # Require H to be in the shape of (nstates, nstates, ngrid)
            elif H.shape[2] == n:
                pass
            else:
                raise ValueError(f"The Hamiltonian should be a 3D array of shape (nstates, nstates, ngrid). Got {H.shape}")
        return cls(R=R, k=k, H=H, mass=mass, dt=dt)
    
    @property
    def ngrid(self) -> int:
        return self.R.shape[0]
    
    @property
    def nstates(self) -> int:
        return self.H.shape[0]
    
    @property
    def dR(self) -> float:
        return self.R[1] - self.R[0]
    
    @property
    def dk(self) -> float:
        return self.k[1] - self.k[0]
    