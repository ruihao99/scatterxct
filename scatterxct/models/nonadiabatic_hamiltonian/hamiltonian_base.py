from numpy.typing import ArrayLike

from abc import ABC, abstractmethod
from typing import Union, Optional

class HamiltonianBase(ABC):
    def __init__(
        self,
        dim: int,
    ) -> None:
        self.dim: int = dim
        self.last_evecs: Optional[ArrayLike] = None
        self.last_deriv_couplings: Optional[ArrayLike] = None
     
    @abstractmethod
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    def update_last_evecs(self, evecs: ArrayLike) -> None:
        self.last_evecs = evecs
        
    def update_last_deriv_couplings(self, deriv_couplings: ArrayLike) -> None:
        self.last_deriv_couplings = deriv_couplings