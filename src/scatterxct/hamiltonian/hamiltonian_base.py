# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple

HamiData = namedtuple(
    "HamiData",
    ["H0", "gradH0", "E0", "G0", "U0", "mu", "gradmu", "V", "gradV"],   
)

@dataclass
class HamiltonianBase(ABC):
    nquant: int
    nclass: int
    mass: RealVector
    laser: bool
    
    @abstractmethod
    def eval_hami(self, R: RealVector, **kwargs) -> HamiData:
        """ evaluate all the Hamiltonian components at the given position """
        """ Returns a named tuple containing the Hamiltonian components """
        """ H0, G0, mu, gradmu, V, gradV """
        raise NotImplementedError
        

# %%
