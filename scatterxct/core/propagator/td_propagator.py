# %%
import numpy as np
from numpy.typing import NDArray

from scatterxct.core.discretization import Discretization
from scatterxct.models.nonadiabatic_hamiltonian import HamiltonianBase, TD_HamiltonianBase

from .absorbing_boundary_condition import get_gamma, get_amplitude_reduction_term
from .propagator_base import PropagatorBase
from .math_utils import get_diabatic_V_propagators, diagonalization, get_diabatic_V_propagators_expm

from dataclasses import dataclass
from typing import Optional, Union
import warnings

@dataclass(frozen=True)
class TD_Propagator(PropagatorBase):
    """Time-dependent propagator for the diabatic representation.

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
        TD_Propagator(PropagatorBase): A time independent propagator for the diabatic representation
        
    Comment annotations abbreviation for developers:
        - rf: reference frozen (the reference to the object is frozen, but you can modify the object. For instance, update the hamiltonian)
        - vf: value frozen (immutable object.)
        - rvf: both reference and value frozen (mutable object, but intended to be used as a constant. For instance, the time-independent part of the Hamiltonian.)
    """
    dt: float # A convenient time step stored in the propagator (vf)
    hamiltonian: TD_HamiltonianBase # The time-dependent Hamiltonian (rvf)
    H0: NDArray[np.complex128] # The time-independent part of Hamiltonian (rvf)
    H: NDArray[np.complex128]  # The total Hamiltonian (rf)
    R: NDArray[np.float64] # The real space grid (rvf)
    E: NDArray[np.float64] # The eigenvalues of the Hamiltonian (rvf)
    U: NDArray[np.complex128] # The eigenvectors of the Hamiltonian (rvf)
    KE: NDArray[np.float64]
    T_propagator: NDArray[np.complex128] # The kinetic energy propagator for dt (rf)
    V_propagator: NDArray[np.complex128] # The potential energy propagator for dt (rf)
    half_T_propagator: NDArray[np.complex128] # The kinetic energy propagator for dt/2 (rf)
    half_V_propagator: NDArray[np.complex128] # The potential energy propagator for dt/2 (rf)
    gamma: NDArray[np.float64]
    
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
    def from_discretization(
        cls, 
        hamiltonian: HamiltonianBase, 
        discretization: Discretization,
        U0: float=1.0,
        alpha: float=0.2,
    ) -> "TD_Propagator":
        # Make sure the Hamiltonian is time-dependent
        if not isinstance(hamiltonian, TD_HamiltonianBase):
            raise TypeError("The Hamiltonian should be time-dependent")
        
        ngrid = discretization.ngrid
        nstates = hamiltonian.dim
        mass = discretization.mass
        dt = discretization.dt
        
        # Create the kinetic energy propagator
        k = discretization.k
        KE = 0.5 * k**2 / mass
        T_propagator = np.exp(-1j * KE * dt)
        half_T_propagator = np.exp(-1j * KE * dt / 2)
        
        # Create the potential energy propagator
        R:NDArray[np.float64] = discretization.R.copy()
        H0: NDArray[np.complex128] = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        H: NDArray[np.complex128] = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        half_V_propagator = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for ii in range(ngrid):
            H0[:, :, ii] = hamiltonian.H0(R[ii])
            H[:, :, ii] = hamiltonian.H1(0, R[ii]) + H0[:, :, ii]
        E, U = diagonalization(H)
            
        return cls(
            dt=dt,
            hamiltonian=hamiltonian,
            H0=H0,
            H=H,
            R=R,
            E=E,
            U=U,
            KE=KE,
            T_propagator=T_propagator,
            V_propagator=V_propagator,
            half_T_propagator=half_T_propagator,
            half_V_propagator=half_V_propagator,
            gamma=get_gamma(U0, alpha, ngrid)
        )
    
    def get_T_propagator(self, t: float) -> NDArray[np.complex128]:
        return self.T_propagator
    
    def get_half_T_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        return self.half_T_propagator
    
    def get_V_propagator(self, t: float) -> NDArray[np.complex128]:
        self.update_hamiltonian(t)
        # get_diabatic_V_propagators(self.H, self.V_propagator, self.dt, self.E, self.U)
        get_diabatic_V_propagators_expm(self.H, self.half_V_propagator, self.dt)
        # get_diabatic_V_propagators(self.half_V_propagator, self.E, self.U, self.dt/2)
        return self.V_propagator
    
    def get_half_V_propagator(self, t: Optional[float]=None) -> NDArray[np.complex128]:
        self.update_hamiltonian(t)
        # get_diabatic_V_propagators(self.H, self.half_V_propagator, self.dt / 2, self.E, self.U)
        get_diabatic_V_propagators_expm(self.H, self.half_V_propagator, self.dt/2)
        # get_diabatic_V_propagators(self.half_V_propagator, self.E, self.U, self.dt/2)
        return self.half_V_propagator
    
    def update_hamiltonian(self, t: float) -> None:
        """Update the Hamiltonian at time t."""
        R: NDArray[np.float64] = self.R
        hamiltonian: TD_HamiltonianBase = self.hamiltonian
        self.H[:] = hamiltonian.H1(t, R, reduce_nuc=False) + self.H0
        # self.E[:], self.U[:] = diagonalization(self.H)
        # self.H[:] = self.H0
        
    def get_absorbing_boundary_term(self) -> NDArray[np.complex128]:
        return get_amplitude_reduction_term(self.gamma, self.dt)
            
    @property
    def ngrid(self) -> int:
        return self.R.shape[0]
    
    @property
    def nstates(self) -> int:
        return self.H.shape[0]
    
    
# %%
