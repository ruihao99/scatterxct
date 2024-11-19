import numpy as np
from numpy.typing import ArrayLike

from .tullyone import TullyOne, _construct_2D_H
from ..nonadiabatic_hamiltonian import TD_HamiltonianBase
from ...pulses import PulseBase as Pulse
from ...pulses import ZeroPulse

from typing import Union
from numbers import Real


class TullyOneTD_type1(TD_HamiltonianBase):
    def __init__(
        self,
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        pulse: Pulse = ZeroPulse(),
    ) -> None:
        super().__init__(dim=2, pulse=pulse)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneTD_type1(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})" 
    
    def H0(self, r: Union[Real, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, V12, -V11, reduce_nuc=reduce_nuc)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        V12 = -self.pulse(t) * np.ones_like(r)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r), reduce_nuc=reduce_nuc)
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.dV11dR(r, self.A, self.B)
        dV12dR = TullyOne.dV12dR(r, self.C, self.D)  
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        return np.zeros((2, 2)) if isinstance(r, Real) else np.zeros((2, 2, len(r)))
