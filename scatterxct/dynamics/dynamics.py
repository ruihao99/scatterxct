# %%
import numpy as np

from scatterxct.core.discretization import Discretization 
from scatterxct.core.wavefunction import WaveFunctionData, gaussian_wavepacket
from scatterxct.core.propagator import PropagatorBase, TD_Propagator, Propagator
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
        basis_representation: BasisRepresentation = BasisRepresentation.Diabatic,
        scale: float = 1.0, # scale the number of grid points
    )-> None:
        """The abstract method for initializing the dynamics."""
        self.hamiltonian = hamiltonian
        self.k0 = k0
        self.R0 = R0
        self.initial_state = initial_state
        self.dt = dt
        self.mass = mass
        # parse the options
        self._parse_options(hamiltonian, basis_representation)
        
        # get the discretization 
        self.discretization = self._get_descretization(scale=scale)
        
        # get the propagator
        if self.basis_representation == BasisRepresentation.Diabatic:
            self.propagator: PropagatorBase = self._get_propagator(self.discretization)
        else:
            raise NotImplementedError("Adiabatic representation is not implemented yet.")
        
        # get the wavefunction data
        self.wavefunction_data = self._get_wavefunction_data()
        
    def __repr__(self) -> str:
        time_dependence = "Time Independent" if self.time_dependence == TimeDependence.TimeIndependent else "Time Dependent" 
        representation = "Diabatic" if self.basis_representation == BasisRepresentation.Diabatic else "Adiabatic"
        simulation_type = f"{time_dependence} {representation}"
        return f"<ScatterXctDynamics: ngrid={self.ngrid}, nstates={self.nstates}, dt={self.dt}, {simulation_type=}>"
    
    def _parse_options(self, hamiltonian: HamiltonianBase, basis_representation: BasisRepresentation) -> None:
        # parse the time dependence for the problem
        time_dependence = None
        if isinstance(hamiltonian, TD_HamiltonianBase):
            time_dependence = TimeDependence.TimeDependent
        elif isinstance(hamiltonian, HamiltonianBase):
            time_dependence = TimeDependence.TimeIndependent
        else:
            raise TypeError(f"Got unexpected type for hamiltonian: {type(hamiltonian)}")    
        self.time_dependence = time_dependence    
        
        # parse the basis representation for the problem
        self.basis_representation = basis_representation
        
    def _get_descretization(self, scale: float = 1.0) -> Discretization:
        discretization = Discretization.from_diabatic_potentials(
            R0=self.R0, k0=self.k0, mass=self.mass, dt=self.dt, scale=scale
        )
        return discretization
    
    def _get_propagator(self, discretization: Discretization) -> PropagatorBase:
        if self.basis_representation == BasisRepresentation.Diabatic:
            if self.time_dependence == TimeDependence.TimeIndependent:
                return Propagator.from_discretization(hamiltonian=self.hamiltonian, discretization=discretization)
            elif self.time_dependence == TimeDependence.TimeDependent:
                return TD_Propagator.from_discretization(hamiltonian=self.hamiltonian, discretization=discretization)
            else:
                raise ValueError(f"Got unexpected time_dependence: {self.time_dependence}")
        else:
            raise NotImplementedError(f"Propagation in the {self.basis_representation} representation is not implemented yet.")
                
    
    def _get_wavefunction_data(self, ) -> WaveFunctionData:
        ngrid: int = self.discretization.ngrid
        nstates: int = self.propagator.nstates
        
        R = self.discretization.R
        
        psi = np.zeros((ngrid, nstates), dtype=np.complex128)
        psi[:, self.initial_state] = gaussian_wavepacket(R, self.R0, self.k0)
        
        diabatic_representation_flag = True if self.basis_representation == BasisRepresentation.Diabatic else False
        return WaveFunctionData.from_numpy_psi(psi, diabatic_representation_flag)
    
    @property
    def ngrid(self) -> int:
        return self.propagator.ngrid
    
    @property
    def nstates(self) -> int:
        return self.propagator.nstates
    

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
