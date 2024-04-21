# %%
import numpy as np
import scipy.linalg as LA
from numpy.typing import NDArray

from scatterxct.core.discretization import Discretization
from scatterxct.models.nonadiabatic_hamiltonian import HamiltonianBase
from scatterxct.models.nonadiabatic_hamiltonian import adiabatic_to_diabatic

from .propagator_base import PropagatorBase
from .absorbing_boundary_condition import get_gamma

from dataclasses import dataclass
from typing import Optional
import warnings

@dataclass(frozen=True)
class Propagator(PropagatorBase):
    """Time independent propagator for the diabatic representation.

    Args:
        PropagatorBase (PropagatorBase): The base class for a generic wavefunction propagator

    Raises:
        TypeError: dt should be a real number
        ValueError: shapes of the T_propagator and half_T_propagator do not match
        ValueError: T_propagator should be a 1D array. (Since it is diagonal, and does not depend on the quantum states)
        ValueError: shapes of the V_propagator and half_V_propagator do not match
        ValueError: V_propagator should be a 3D array of shape (nstates, nstates, ngrid)
        ValueError: The V_propagator should be a 3D array of shape (nstates, nstates, ngrid). Got {shape_V}. Refused to initialize the DiabaticPropagator.

    Returns:
        Propagator(PropagatorBase): A time independent propagator for the diabatic representation
    """
    dt: float # A convenient time step stored in the propagator (value frozen)
    H: NDArray[np.complex128] # The total Hamiltonian (value reference frozen)
    E: NDArray[np.float64] # The eigenvalues of the Hamiltonian (value reference frozen)
    U: NDArray[np.complex128] # The eigenvectors of the Hamiltonian (value reference frozen)
    KE: NDArray[np.float64] # The kinetic energy of the Hamiltonian (value reference frozen)
    T_propagator: NDArray[np.complex128] # The kinetic energy propagator for dt (reference frozen)
    V_propagator: NDArray[np.complex128] # The potential energy propagator for dt (reference frozen)
    half_T_propagator: NDArray[np.complex128] # The kinetic energy propagator for dt/2 (reference frozen)
    half_V_propagator: NDArray[np.complex128] # The potential energy propagator for dt/2 (reference frozen)
    gamma: NDArray[np.float64]
    
    def __post_init__(self):
        # Type checks for the time independent propagator
        if not isinstance(self.dt, (int, float)):
            raise TypeError("The time step should be a real number.")
        
        # shape_T = self.T_propagator.shape
        # shape_half_T = self.half_T_propagator.shape
        
        # if shape_T != shape_half_T:
        #     raise ValueError("The T_propagator and half_T_propagator have incompatible shapes")
        # if self.T_propagator.ndim != 1:
        #     raise ValueError("The T_propagator should be a 1D array")
       
        shape_V = self.V_propagator.shape
        shape_half_V = self.half_V_propagator.shape
        if shape_V != shape_half_V:
            raise ValueError("The V_propagator and half_V_propagator have incompatible shapes")
        if self.V_propagator.ndim != 3:
            raise ValueError("The V_propagator should be a 3D array")
        nstates_1, nstates_2, ngrid = shape_V 
        if ((ngrid == nstates_1) or (ngrid == nstates_2)):
            if nstates_1 == nstates_2:
                warnings.warn(
                    f"What a coincidence! The V_propagator has shape {shape_V}, meaning the number of states equals the number of grid points. Please double check the shape of the V_propagator!"
                )
            else:
                raise ValueError(f"The V_propagator should be a 3D array of shape (nstates, nstates, ngrid). Got {shape_V}. Refused to initialize the DiabaticPropagator.")
    
    @classmethod
    def from_discretization(
        cls, 
        hamiltonian: HamiltonianBase, 
        discretization: Discretization,
        U0: float=1.0,
        alpha: float=0.2,
    ) -> "Propagator":
        ngrid = discretization.ngrid
        nstates = hamiltonian.dim
        mass = discretization.mass
        dt = discretization.dt
        
        # Create the kinetic energy propagator
        k = discretization.k
        KE = k**2/(2 * mass)
        T_propagator = np.exp(-1j * KE * dt)
        half_T_propagator = np.exp(-1j * KE * dt / 2)
        
        # Create the potential energy propagator
        R = discretization.R
        # H = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        dummy_t: float = 0.0 # dummy time for time-independent Hamiltonian
        H = hamiltonian.H(t=dummy_t, r=R, reduce_nuc=False)
        E = np.zeros((nstates, ngrid), dtype=np.float64)
        U = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        half_V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for ii in range(ngrid):
            H_ii = H[:, :, ii]
            E[:, ii], U[:, :, ii] = LA.eigh(H_ii)
            V_propagator[:, :, ii] = adiabatic_to_diabatic(np.diagflat(np.exp(-1j * E[:, ii] * dt)), U[:, :, ii])
            half_V_propagator[:, :, ii] = adiabatic_to_diabatic(np.diagflat(np.exp(-1j * E[:, ii] * dt / 2)), U[:, :, ii])
        return cls(
            dt=dt, 
            H=H, 
            E=E, 
            U=U, 
            KE=KE,
            T_propagator=T_propagator, 
            V_propagator=V_propagator, 
            half_T_propagator=half_T_propagator, 
            half_V_propagator=half_V_propagator,
            gamma=get_gamma(U0, alpha, ngrid)
        )        
    
    def get_T_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        return self.T_propagator
    
    def get_half_T_propagator(self, t: float) -> NDArray[np.complex128]:
        return self.half_T_propagator
    
    def get_V_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        return self.V_propagator
    
    def get_half_V_propagator(self, t: float) -> NDArray[np.complex128]:
        return self.half_V_propagator
    
    def get_amplitude_reduction(self, ) -> NDArray[np.float64]:
        return (1 - self.gamma * self.dt)[:, np.newaxis]
    
    @property
    def nstates(self) -> int:
        return self.V_propagator.shape[0]
    
    @property
    def ngrid(self) -> int:
        return self.V_propagator.shape[2]
    
# %%
