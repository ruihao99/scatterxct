from numpy.typing import ArrayLike

from .hamiltonian_base import HamiltonianBase
from ...pulses import PulseBase as Pulse

from abc import abstractmethod
from typing import Union

class TD_HamiltonianBase(HamiltonianBase):
    def __init__(
        self,
        dim: int,
        pulse: Pulse,
    ) -> None:
        """ Time-dependent nonadiabatic Hamiltonian. """
        """ The time dependence is defined by a 'Pulse' object. """
        """ The pulse consists of a carrier frequency <Omega> and an envelope <E(t)>. """
        super().__init__(dim)
        self.pulse = pulse
        
    def H(self, t: float, r: Union[float, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        return self.H0(r, reduce_nuc) + self.H1(t, r, reduce_nuc)
    
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return self.dH0dR(r) + self.dH1dR(t, r)
    
    @abstractmethod 
    def H0(self, r: Union[float, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        pass
    
    @abstractmethod 
    def H1(self, t: float, r: Union[float, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        pass
    
    @abstractmethod
    def dH0dR(self, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod 
    def dH1dR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass