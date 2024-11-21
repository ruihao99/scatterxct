# %%
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from scatterxct.core.discretization import Discretization
from scatterxct.core.discretization.finite_difference import fddrv
from scatterxct.hamiltonian import HamiltonianBase, HamiData

from .propagator_base import PropagatorBase
from .absorbing_boundary_condition import get_gamma, get_amplitude_reduction_term
from .math_utils import diagonalization, get_nac

from dataclasses import dataclass
from typing import Optional
import warnings

@dataclass(frozen=True)
class PropagatorA(PropagatorBase):
    """Time independent propagator in the adiabatic representation.

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
    E: NDArray[np.float64]    # The eigenvalues of the H = H0(R) (value reference frozen)
    U: NDArray[np.complex128] # The eigenvectors of the H = H0(R) (value reference frozen)
    G: NDArray[np.float64]    # The generalized gradient of the H = H0(R) in the eigenbasis (value reference frozen)
    KE: NDArray[np.float64]   # The nuclear kinetic energy
    D: NDArray[np.float64]    # The nac 
    T_prop: NDArray[np.complex128] # The kinetic energy propagator for dt (reference frozen)
    half_T_prop: NDArray[np.complex128] # The kinetic energy propagator for dt/2 (reference frozen)
    V0_prop: NDArray[np.complex128] # The potential energy propagator for dt (reference frozen)
    half_V0_prop: NDArray[np.complex128] # The potential energy propagator for dt/2 (reference frozen)
    V_prop: NDArray[np.complex128] # The potential energy propagator for dt (reference frozen)
    half_V_prop: NDArray[np.complex128] # The potential energy propagator for dt/2 (reference frozen)
    gamma: NDArray[np.float64] # For the absorbing boundary condition
    
    def __post__init__(self):
        # Type checks for the time independent propagator
        if not isinstance(self.dt, (int, float)):
            raise TypeError("The time step should be a real number.")
        
        shape_V = self.V_prop.shape
        shape_half_V = self.half_V_prop.shape
        if shape_V != shape_half_V:
            raise ValueError("The V_propagator and half_V_propagator have incompatible shapes")
        if self.V_prop.ndim != 3:
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
    ) -> "PropagatorA":
        ngrid = discretization.ngrid
        nstates = hamiltonian.nquant
        mass = discretization.mass  
        dt = discretization.dt
        
        # create the kinetic energy propagator 
        # (in momentum space, the kinetic energy is diagonal)   
        k = discretization.k
        KE = k**2 / (2 * mass)
        T_prop = np.exp(-1j * KE * dt)  
        half_T_prop = np.exp(-1j * KE * dt / 2)
        
        # create the adiabatic potential energy propagator
        R = discretization.R
        U0_last = None
        E0_all = np.zeros((nstates, ngrid), dtype=np.float64)
        U0_all = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        G0_all = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for ii, rr in enumerate(R):
            hamdata = hamiltonian.eval_hami(rr, Uold=U0_last)
            E0_all[:, ii] = hamdata.E0
            U0_all[:, :, ii] = hamdata.U0
            G0_all[:, :, ii] = hamdata.G0[:,:]
            U0_last = U0_all[:, :, ii]
            
        # compute the V0 propagator
        V0_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        half_V0_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128) 
        for ii in range(ngrid):
            V0_prop[:,:,ii] = np.diagflat(np.exp(-1j * E0_all[:, ii] * dt))
            half_V0_prop[:,:,ii] = np.diagflat(np.exp(-1j * E0_all[:, ii] * dt / 2))
            
        # get the NAC
        D = get_nac(G0_all, E0_all)
        
        # compute half_V_propagator
        ACCURACY = 10
        vel = fddrv(length=ngrid, order=1, accuracy=ACCURACY) / (mass * discretization.dR * 1j)
        vD = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        half_V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for jj in range(nstates):
            for kk in range(nstates):
                vD[jj, kk, :] = np.dot(vel, D[jj, kk, :])
                
        for ii in range(ngrid):
            Heff = np.diagflat(E0_all[:, ii]) - 1j * vD[:, :, ii]
            V_prop[:,:,ii] = expm(-1j * Heff * dt)
            half_V_prop[:,:,ii] = expm(-1j * Heff * dt / 2)
            
        
        
            
        return cls(
            dt=dt,
            E=E0_all,   
            U=U0_all,
            G=G0_all,
            KE=KE,
            D=D,    
            T_prop=T_prop,
            half_T_prop=half_T_prop,
            V0_prop=V0_prop,
            half_V0_prop=half_V0_prop,
            V_prop=V_prop,
            half_V_prop=half_V_prop,
            gamma=get_gamma(U0, alpha, ngrid)
        )
        
    def get_T_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        raise NotImplementedError("Don not use expotential propagator for Adiabatic representation")
    
    def get_half_T_propagator(self, t: float) -> NDArray[np.complex128]:
        raise NotImplementedError("Don not use expotential propagator for Adiabatic representation")
    
    def get_V_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        raise NotImplementedError("Don not use expotential propagator for Adiabatic representation")
    
    def get_half_V_propagator(self, t: float) -> NDArray[np.complex128]:
        raise NotImplementedError("Don not use expotential propagator for Adiabatic representation")
    
    def get_absorbing_boundary_term(self) -> NDArray[np.complex128]:
        return get_amplitude_reduction_term(self.gamma, self.dt)
    
    @property
    def nstates(self) -> int:
        return self.E.shape[0]
    
    @property
    def ngrid(self) -> int:
        return self.E.shape[1]
        

         
            
            
            
        
        
        
    
        