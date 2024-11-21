# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector
from scatterxct.hamiltonian.phase_tracking.phase_tracking import PhaseTracking

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple, Union

HamiData = namedtuple(
    "HamiData",
    [
        "Hd", "Gd", "Md", "Dd", # diabatic H, gradient, dipole, dipole gradient
        "Ha", "Ga", "Ma", "Da", # adiabatic H, gradient, dipole, dipole gradient
        "U", "Vext", "Gext"     # eigenvectors, external potential/gradient
    ],   
)

@dataclass
class HamiltonianBase(ABC):
    nquant: int
    nclass: int
    mass: RealVector
    laser: bool
    phase_tracking: PhaseTracking = field(default_factory=lambda: PhaseTracking("none"))    
    
    @abstractmethod
    def eval_hami(self, R: RealVector, **kwargs) -> HamiData:
        """ evaluate all the Hamiltonian components at the given position """
        """ Returns a named tuple containing the Hamiltonian components """
        """ H0, G0, mu, gradmu, V, gradV """
        raise NotImplementedError
    
    def eval_hami_recursive(self, R: RealVector, Hold: ComplexOperator, Uold: ComplexVectorOperator, dt: float, **kwargs) -> HamiData:
        raise NotImplementedError
    
    def harmornic_params(self, ) -> Tuple[float, float, float]:
        """ return the harmonic frequency, xeq, and the mass """
        """ Used for initial classical trajectory generation """
        raise NotImplementedError
    
    def morse_params(self, ) -> Tuple[float, float, float, float]:
        """ return the Morse potential parameters """
        """ Used for initial classical trajectory generation """
        raise NotImplementedError   
    
    def get_max_dipole(self, ) -> Union[float, None]:
        """Return the maximum matrix element of the dipole operator
            Optional method to be implemented when the dipole operator
            is available. Used by the Floquet module to automatically 
            estimate the Floquet cutoff.

        Returns:
            Union[float, None]: The maximum matrix element of the 
                dipole operator. Defaults to None. 
                When None: use value 1.0 as the maximum dipole element,
                and output a [INFO] message. (in atomic units system, 
                1.0 is 2.5417 Debye, which corresponds to a strong
                dipole moment, should be sufficient for most cases)
        """
        return None
        

# %%
