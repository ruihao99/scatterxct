# %%
import numpy as np
from numpy.typing import NDArray

from scatterxct.core.discretization import Discretization
from scatterxct.hamiltonian import HamiltonianBase
from scatterxct.pulses import PulseBase

from scatterxct.core.propagator.absorbing_boundary_condition import get_gamma   
from scatterxct.core.propagator.absorbing_boundary_condition import get_amplitude_reduction_term
from scatterxct.core.propagator.math_utils import get_diabatic_V_propagators_expm
from scatterxct.core.propagator.math_utils import evaluate_propagator_laser

from dataclasses import dataclass
from typing import Optional

CArray = NDArray[np.complex128]
RArray = NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class Propagator:
    dt: float
    # Total Hamiltonian data
    H: CArray  # The total diabatic Hamiltonian (H_tot)
    E: RArray  # The eigenvalues of H_tot
    U: CArray  # The instantaneous eigenvectors of H_tot
    # molecular Hamiltonian data
    H0: CArray      # The diabatic molecular Hamiltonian (H_mol)
    gradH0: CArray  # The gradient of H_mol
    E0: RArray      # The eigenvalues of H_mol
    U0: CArray      # The eigenvectors of H_mol
    G0: CArray      # The generalized gradient of H_mol
    mu_ad: CArray   # The dipole moment (H_mol adiabatic representation)
    mu_diab: CArray # The dipole moment (H_mol diabatic representation)
    V_propagator: CArray      # The potential energy propagator
    half_V_propagator: CArray # The potential energy propagator for dt/2    
    # Kinetic energy data
    KE: CArray           # The kinetic energy operator (full)
    T_prop: CArray       # The kinetic energy propagator
    half_T_prop: CArray  # The kinetic energy propagator for dt/2
    # Laser data
    pulse: PulseBase
    # Absorbing boundary condition data
    gamma: RArray
    
    def __post__init__(self):
        # Type checks for the time independent propagator
        if not isinstance(self.dt, (int, float)):
            raise TypeError("The time step should be a real number.")
        
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
    def init(
        cls,
        hamiltonian: HamiltonianBase,
        discretization: Discretization,
        pulse: PulseBase,
        U0: float=1.0,
        alpha: float=0.2,
    ) -> 'Propagator':
        # unpack discretization data    
        ngrid = discretization.ngrid
        nstates = hamiltonian.nquant
        mass = discretization.mass
        dt = discretization.dt
        
        # Create the kinetic energy propagator
        k = discretization.k
        KE = k**2 / (2 * mass)
        T_prop = np.exp(-1j * KE * dt)  
        half_T_prop = np.exp(-1j * KE * dt / 2)
        
        
        # Create the adiabatic potential energy propagator
        R = discretization.R
        U0_last = None
        H = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        E = np.zeros((nstates, ngrid), dtype=np.float64)
        U = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        H0 = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        gradH0 = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        E0 = np.zeros((nstates, ngrid), dtype=np.float64)
        U0 = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        G0 = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        mu_ad = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        mu_diab = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        for ii, rr in enumerate(R):
            hamdata = hamiltonian.eval_hami(rr, Uold=U0_last)
            H0[:, :, ii] = hamdata.H0
            gradH0[:, :, ii] = hamdata.gradH0
            E0[:, ii] = hamdata.E0
            U0[:, :, ii] = hamdata.U0
            G0[:, :, ii] = hamdata.G0[:,:]
            mu_ad[:, :, ii] = hamdata.mu
            # transform the adiabatic dipole moment to diabatic representation
            mu_diab[:, :, ii] = np.dot(U0[:,:,ii], np.dot(mu_ad[:,:,ii], U0[:,:,ii].T))
            U0_last = U0[:, :, ii]
        
        # initialize the V propagators
        V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128) 
        half_V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        
        # evaluate the time-independent V propagators if the simulation
        # is free from laser field
        if pulse is None:
            get_diabatic_V_propagators_expm(H=H0, V=V_prop, dt=dt)
            get_diabatic_V_propagators_expm(H=H0, V=half_V_prop, dt=dt/2)
            H[:] = H0[:]
            E[:] = E0[:]
            U[:] = U0[:]
            
        return cls(
            dt=dt,
            H=H,
            E=E,
            U=U,
            H0=H0,
            gradH0=gradH0,
            E0=E0,
            U0=U0,
            G0=G0,
            mu_ad=mu_ad,
            mu_diab=mu_diab,
            V_propagator=V_prop,
            half_V_propagator=half_V_prop,
            KE=KE,
            T_prop=T_prop,
            half_T_prop=half_T_prop,
            pulse=pulse,
            gamma=get_gamma(U0, alpha, ngrid)
        )   
        
    
    def get_T_propagator(self,):
        return self.T_prop
    
    def get_half_T_propagator(self,):
        return self.half_T_prop
    
    def get_V_propagator(self, t: Optional[float]=None):
        if self.pulse is None:
            return self.V_propagator
        else:
            assert isinstance(t, (int, float)), "The time should be a real number."
            laser_signal = self.pulse.signal(t)
            H, E, U, V = evaluate_propagator_laser(
                H0=self.H0,
                mu=self.mu_diab,
                Et=laser_signal,
                U_old=self.U,
                dt=self.dt
            )
            # update the data
            self.H[:] = H
            self.E[:] = E
            self.U[:] = U
            self.V_propagator[:] = V
            print("V_propagator updated")
            return self.V_propagator
            
    
    def get_half_V_propagator(self, t: Optional[float]=None):   
        if self.pulse is None:
            return self.half_V_propagator
        else:
            assert isinstance(t, (int, float)), "The time should be a real number."
            laser_signal = self.pulse.signal(t)
            H, E, U, V = evaluate_propagator_laser(
                H0=self.H0,
                mu=self.mu_diab,
                Et=laser_signal,
                U_old=self.U,
                dt=self.dt/2
            )
            # update the data
            self.H[:] = H
            self.E[:] = E
            self.U[:] = U
            self.half_V_propagator[:] = V
            return self.half_V_propagator
        
    def get_absorbing_boundary_term(self):
        return get_amplitude_reduction_term(self.gamma, self.dt)
    
    @property
    def nstates(self) -> int:
        return self.V_propagator.shape[0]
    
    @property
    def ngrid(self) -> int:
        return self.V_propagator.shape[2]
    
    