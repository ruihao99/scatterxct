# %% Use tully one to test the nonadiabatic hamiltonian abc class
import numpy as np
from numpy.typing import ArrayLike

from ..nonadiabatic_hamiltonian import HamiltonianBase

from typing import Union
from numbers import Real


class TullyOne(HamiltonianBase):
    def __init__(
        self,
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
    ) -> None:
        super().__init__(dim=2)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    @staticmethod
    def V11(
        r: Union[Real, ArrayLike],
        A: Real,
        B: Real,
    ) -> Union[Real, ArrayLike]:
        sign = np.sign(r)
        return sign * A * (1 - np.exp(-sign * B * r))
    
    @staticmethod
    def V12(
        r: Union[Real, ArrayLike],
        C: Real,
        D: Real,
    ) -> Union[Real, ArrayLike]:
        return C * np.exp(-D * r**2)
    
    @staticmethod
    def dV11dR(
        r: Union[Real, ArrayLike],
        A: Real,
        B: Real,
    ) -> Union[Real, ArrayLike]:
        return A * B * np.exp(-np.abs(r) * B) 
    
    @staticmethod
    def dV12dR(
        r: Union[Real, ArrayLike],
        C: Real,
        D: Real,
    ) -> Union[Real, ArrayLike]:
        return -2 * C * D * r * np.exp(-D * r**2)  

    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOne(A={self.A}, B={self.B}, C={self.C}, D={self.D})"
    
    def H(self, t: Real, r: Union[Real, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        V11 = self.V11(r, self.A, self.B)
        V12 = self.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, V12, -V11, reduce_nuc)
    
    def dHdR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = self.dV11dR(r, self.A, self.B)
        dV12dR = self.dV12dR(r, self.C, self.D)
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    

def _construct_2D_H(
    r: Union[Real, ArrayLike],
    V11: Union[Real, ArrayLike],
    V12: Union[Real, ArrayLike],
    V22: Union[Real, ArrayLike],
    reduce_nuc: bool = True,
) -> ArrayLike:
    if isinstance(r, Real):
        return np.array([[V11, V12], [np.conj(V12), V22]])
    elif isinstance(r, np.ndarray):
        H = np.array([[V11, V12], [V12.conj(), V22]]) 
        if reduce_nuc:
            try:
                return np.sum(H, axis=-1)
            except ValueError:
                raise ValueError(f"The input array 'r' must be either a number or a 1D array. 'r' input here has dimension of {r.ndim}.")
        else:
            return H
    else:
        raise NotImplemented