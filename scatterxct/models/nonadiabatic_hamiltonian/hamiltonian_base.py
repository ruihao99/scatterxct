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
    def H(self, t: float, r: Union[float, ArrayLike], reduce_nuc: bool=True) -> ArrayLike:
        """Generic evaluation of the Hamiltonian.

        Args:
            t (float): optional time, effective when the Hamiltonian is time-dependent
            r (Union[float, ArrayLike]): nuclear coordinate(s) 
            reduce_nuc (bool, optional): flag for reducing the nuclear dimensionality. Defaults to True. To elaborate,
                if True, the Hamiltonian is reduced to a 2x2 matrix (H = \sum_I \sum_i \sum_j C_{iI}^* C_{jI} H_{ij}(R_I)).
                if False, then the individual nuclear contributions to the Hamiltonian are returned, as a 3D array
                shapped as (n_states, n_states, nuclear_dimension).

        Returns:
            ArrayLike: the Hamiltonian shapped as (n_states, n_states) if `reduce_nuc` is True, or (n_states, n_states, nuclear_dimension) if `reduce_nuc` is False.
        """
        pass
    
    @abstractmethod
    def dHdR(self, t: float, r: Union[float, ArrayLike], ) -> ArrayLike:
        """Generic evaluation of the nuclear gradient of the Hamiltonian.

        Args:
            t (float): time, effective when the Hamiltonian is time-dependent
            r (Union[float, ArrayLike]): nuclear coordinate(s)

        Returns:
            ArrayLike: the nuclear gradient. The shape is (n_states, n_states, nuclear_dimension).
        """
        pass
    
    def update_last_evecs(self, evecs: ArrayLike) -> None:
        self.last_evecs = evecs
        
    def update_last_deriv_couplings(self, deriv_couplings: ArrayLike) -> None:
        self.last_deriv_couplings = deriv_couplings