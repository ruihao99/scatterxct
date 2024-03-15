# %%
import numpy as np
import scipy.linalg as LA
from numpy.typing import ArrayLike

from scatterxct.core.discretization import Discretization
from scatterxct.models.nonadiabatic_hamiltonian import adiabatic_to_diabatic, diabatic_to_adiabatic

from .propagator_base import PropagatorBase

from dataclasses import dataclass
from typing import Optional
import warnings

@dataclass(frozen=True)
class DiabaticPropagator(PropagatorBase):
    dt: float
    T_propagator: ArrayLike
    V_propagator: ArrayLike
    half_T_propagator: ArrayLike 
    half_V_propagator: ArrayLike
    
    def __post_init__(self):
        # Type checks for the time independent propagator
        if not isinstance(self.dt, (int, float)):
            raise TypeError("The time step should be a real number.")
        
        shape_T = self.T_propagator.shape
        shape_half_T = self.half_T_propagator.shape
        
        if shape_T != shape_half_T:
            raise ValueError("The T_propagator and half_T_propagator have incompatible shapes")
        if self.T_propagator.ndim != 1:
            raise ValueError("The T_propagator should be a 1D array")
       
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
    def from_discretization(cls, descretization: Discretization) -> "DiabaticPropagator":
        ngrid = descretization.ngrid
        nstates = descretization.nstates
        mass = descretization.mass
        dt = descretization.dt
        
        # Create the kinetic energy propagator
        k = descretization.k
        T_propagator = np.exp(-1j * k**2 * dt / (2 * mass))
        half_T_propagator = np.exp(-1j * k**2 * dt / (4 * mass))
        
        # Create the potential energy propagator
        H = descretization.H
        V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        half_V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for ii in range(ngrid):
            Hi = H[:, :, ii]
            evals, evecs = LA.eigh(Hi)
            V_propagator[:, :, ii] = adiabatic_to_diabatic(np.diagflat(np.exp(-1j * evals * dt)), evecs)
            half_V_propagator[:, :, ii] = adiabatic_to_diabatic(np.diagflat(np.exp(-1j * evals * dt / 2)), evecs)
            
        return cls(dt, T_propagator, V_propagator, half_T_propagator, half_V_propagator)
    
    def get_T_propagator(self, t: Optional[float]=None) -> ArrayLike:
        return self.T_propagator
    
    def get_half_T_propagator(self, t: float) -> ArrayLike:
        return self.half_T_propagator
    
    def get_V_propagator(self, t: Optional[float]=None) -> ArrayLike:
        return self.V_propagator
    
    def get_half_V_propagator(self, t: float) -> ArrayLike:
        return self.half_V_propagator
    
# %%
