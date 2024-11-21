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
    Hd: CArray # diabatic (d) hamiltonian (H0)
    Gd: CArray # gradient of (d) hamiltonian
    Md: CArray # (d) dipole operator 
    Dd: CArray # gradient of (d) dipole operator
    Ha: CArray # adiabatic (a) hamiltonian
    Ga: CArray # gradient of (a) hamiltonian
    Ma: CArray # (a) dipole operator
    Da: CArray # gradient of (a) dipole operator
    U0: CArray # Unitary transformation from diabatic to adiabatic 
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
        UU0: float=1.0,
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
        
        # grid points
        R = discretization.R
        
        ### Total Hamiltonian data 
        H = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        E = np.zeros((nstates, ngrid), dtype=np.float64)
        U = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        
        ### Molecular Hamiltonian data  
        U0_last = None
        Hd = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Gd = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Md = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Dd = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Ha = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Ga = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Ma = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        Da = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        U0 = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        
        for ii, rr in enumerate(R):
            hamdata = hamiltonian.eval_hami(rr, Uold=U0_last)
            hdata = hamiltonian.eval_hami([rr], Uold=U0_last)
            Hd[:, :, ii] = hdata.Hd
            Gd[:, :, ii] = hdata.Gd[:,:,0]
            Ha[:, :, ii] = hdata.Ha
            Ga[:, :, ii] = hdata.Ga[:,:,0]
            if hdata.Dd is not None:
                Md[:, :, ii] = hdata.Md
                Dd[:, :, ii] = hdata.Dd[:,:,0]
                Ma[:, :, ii] = hdata.Ma
                Da[:, :, ii] = hdata.Da[:,:,0]
            U0[:, :, ii] = hdata.U
            U0_last = hdata.U
        
        # initialize the V propagators
        V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128) 
        half_V_prop = np.zeros((nstates, nstates, ngrid), dtype=np.complex128)
        
        # evaluate the time-independent V propagators if the simulation
        # is free from laser field
        if pulse is None:
            get_diabatic_V_propagators_expm(H=Hd, V=V_prop, dt=dt)
            get_diabatic_V_propagators_expm(H=Hd, V=half_V_prop, dt=dt/2)
            H[:] = Hd[:]
            for ii in range(ngrid):
                E[:, ii] = np.real(np.diag(Ha[:, :, ii]))
                U[:, :, ii] = U0[:, :, ii]  
            
        return cls(
            dt=dt,
            H=H,
            E=E,
            U=U,
            Hd=Hd,
            Gd=Gd,
            Md=Md,
            Dd=Dd,
            Ha=Ha,
            Ga=Ga,
            Ma=Ma,
            Da=Da,
            U0=U0,
            V_propagator=V_prop,
            half_V_propagator=half_V_prop,
            KE=KE,
            T_prop=T_prop,
            half_T_prop=half_T_prop,
            pulse=pulse,
            gamma=get_gamma(UU0, alpha, ngrid)
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
            H, V = evaluate_propagator_laser(
                Hd=self.Hd,
                Md=self.Md,
                Et=laser_signal,
                dt=self.dt 
            )
            # update the data
            self.H[:] = H
            self.V_propagator[:] = V
            return self.V_propagator
            
    
    def get_half_V_propagator(self, t: Optional[float]=None):   
        if self.pulse is None:
            return self.half_V_propagator, self.half_V_propagator
        else:
            assert isinstance(t, (int, float)), "The time should be a real number."
            Et = self.pulse.signal(t+self.dt/2)
            H1, half_V_1 = evaluate_propagator_laser(
                Hd=self.Hd,
                Md=self.Md,
                Et=Et,
                dt=self.dt/2
            )
            
            Et_plus = self.pulse.signal(t + self.dt)
            H2, half_V_2 = evaluate_propagator_laser(
                Hd=self.Hd,
                Md=self.Md,
                Et=Et_plus,
                dt=self.dt/2
            )
           
            # update the data
            self.H[:] = H2
            self.half_V_propagator[:] = half_V_2
            return half_V_1, half_V_2
        
    def get_absorbing_boundary_term(self):
        return get_amplitude_reduction_term(self.gamma, self.dt)
    
    @property
    def nstates(self) -> int:
        return self.V_propagator.shape[0]
    
    @property
    def ngrid(self) -> int:
        return self.V_propagator.shape[2]
    
    