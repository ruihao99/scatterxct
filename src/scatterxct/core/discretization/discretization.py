import numpy as np
from numpy.typing import NDArray
from scipy.fftpack import fftfreq

from scatterxct.core.discretization.utils import estimate_R_lims

from dataclasses import dataclass

@dataclass(frozen=True)
class Discretization:
    R: NDArray[np.float64]
    k: NDArray[np.float64]
    mass: float
    dt: float 
    
    @classmethod
    def from_diabatic_potentials(cls, R0: float, k0: float, mass: float, dt: float, scatter_region_center: float=0.0, scale: float=1) -> "Discretization":
        R_lims, n_real_space_grid = estimate_R_lims(
            R0=R0, k0=k0, scatter_region_center=scatter_region_center
        )
        
        n: int = int(n_real_space_grid * scale)
        R = np.linspace(R_lims[0]*scale, R_lims[1]*scale, n)
        delta_R = R[1] - R[0]
        k = fftfreq(n, d=delta_R) * 2 * np.pi
        # H = np.array(
        #     [hamiltonian(Ri) for Ri in R]
        # )
        # if H.ndim == 2:
        #     raise ValueError("The Hamiltonian should be a 3D array")
        # elif H.ndim == 3:
        #     if H.shape[0] == n:
        #         H = H.swapaxes(0, 2) # Require H to be in the shape of (nstates, nstates, ngrid)
        #     elif H.shape[2] == n:
        #         pass
        #     else:
        #         raise ValueError(f"The Hamiltonian should be a 3D array of shape (nstates, nstates, ngrid). Got {H.shape}")
        return cls(R=R, k=k, mass=mass, dt=dt)
    
    
    def get_R_grid(self) -> NDArray[np.float64]:
        return self.R
    
    def get_k_grid(self) -> NDArray[np.float64]:
        return self.k
    
    def get_dt(self) -> float:
        return self.dt
    
    @property
    def ngrid(self) -> int:
        return self.R.shape[0]
    
    @property
    def dR(self) -> float:
        return self.R[1] - self.R[0]
    
    @property
    def dk(self) -> float:
        return self.k[1] - self.k[0]
    