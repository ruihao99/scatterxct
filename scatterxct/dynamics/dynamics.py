# %%
import numpy as np

from scatterxct.core.discretization import Discretization 
from scatterxct.core.wavefunction import WaveFunctionData, gaussian_wavepacket
from scatterxct.core.propagator import PropagatorBase, DiabaticPropagator
from scatterxct.models.nonadiabatic_hamiltonian import HamiltonianBase, TD_HamiltonianBase
from scatterxct.dynamics.options import BasisRepresentation, TimeDependence

class ScatterXctDynamics:
    """ Class for organizing scattering dynamics using Kosloff's split operator method. """
    def __init__(
        self, 
        hamiltonian: HamiltonianBase, # the Hamiltonian class in pymddrive format
        k0: float, # initial momentum
        R0: float = -10.0, # initial position, defaults to that of the tully SAC model
        initial_state: int = 0, # initial state, defaults to the ground state
        dt: float = 0.1,   # time step
        mass: float = 2000.0, # mass of the particle, defaults to that of the tully models
        basis_representation: BasisRepresentation = BasisRepresentation.Diabatic
    )-> None:
        """The abstract method for initializing the dynamics."""
        self.hamiltonian = hamiltonian
        self.k0 = k0
        self.R0 = R0
        self.initial_state = initial_state
        self.dt = dt
        self.mass = mass
        
        # determine the time dependence for the problem
        if isinstance(hamiltonian, TD_HamiltonianBase):
            self.time_dependence = TimeDependence.TimeDependent
            raise NotImplementedError("Time-dependent Hamiltonian is not implemented yet.")
        else:
            self.time_dependence = TimeDependence.TimeIndependent
            
        # setting the basis representation
        if basis_representation == BasisRepresentation.Diabatic:
            self.basis_representation = BasisRepresentation.Diabatic
        else:
            self.basis_representation = BasisRepresentation.Adiabatic
            raise NotImplementedError("Adiabatic representation is not implemented yet.")
        
        # get the discretization 
        if self.time_dependence == TimeDependence.TimeIndependent:
            self.discretization = self._get_time_independent_descretization()
        else:
            raise NotImplementedError("Time-dependent Hamiltonian is not implemented yet.")
        
        # get the propagator
        if self.basis_representation == BasisRepresentation.Diabatic:
            self.propagator: PropagatorBase = self._get_diabatic_propagator(self.discretization)
        else:
            raise NotImplementedError("Adiabatic representation is not implemented yet.")
        
        # get the wavefunction data
        self.wavefunction_data = self._get_wavefunction_data()
        
    def __repr__(self) -> str:
        time_dependence = "Time Independent" if self.time_dependence == TimeDependence.TimeIndependent else "Time Dependent" 
        representation = "Diabatic" if self.basis_representation == BasisRepresentation.Diabatic else "Adiabatic"
        simulation_type = f"{time_dependence} {representation}"
        return f"<ScatterXctDynamics: ngrid={self.ngrid}, nstates={self.nstates}, dt={self.dt}, {simulation_type=}>"
        
    def _get_time_independent_descretization(self,) -> Discretization:
        dummy_time: float = 0.0
        def hamiltonian_wrapper(R: float) -> np.ndarray:
            return self.hamiltonian.H(t=dummy_time, r=R)
        
        discretization = Discretization.from_diabatic_potentials(
            R0=self.R0, k0=self.k0, hamiltonian=hamiltonian_wrapper, mass=self.mass, dt=self.dt
        )
        return discretization
    
    def _get_diabatic_propagator(self, discretization: Discretization) -> DiabaticPropagator:
        return DiabaticPropagator.from_discretization(discretization)
    
    def _get_wavefunction_data(self,) -> WaveFunctionData:
        ngrid: int = self.discretization.ngrid
        nstates: int = self.discretization.nstates
        
        R = self.discretization.R
        
        psi = np.zeros((ngrid, nstates), dtype=np.complex128)
        psi[:, self.initial_state] = gaussian_wavepacket(R, self.R0, self.k0)
        return WaveFunctionData.from_numpy_psi(psi_in=psi)
    
    @property
    def ngrid(self) -> int:
        return self.discretization.ngrid
    
    @property
    def nstates(self) -> int:
        return self.discretization.nstates
    

# %%
def _test_main(k0: float = 1):
    from scatterxct.models.tullyone import get_tullyone, TullyOnePulseTypes
    from scatterxct.core.wavefunction.view_wavepacket import view_wavepacket
    hamiltonian = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE
    )
    dynamics = ScatterXctDynamics(hamiltonian=hamiltonian, k0=k0)
    
    print(dynamics)
    
    
    R = dynamics.discretization.R
    psi = dynamics.wavefunction_data.psi
    
    fig = view_wavepacket(R, psi)
    
# %%
if __name__ == "__main__":
    _test_main(k0=30)
# %%
