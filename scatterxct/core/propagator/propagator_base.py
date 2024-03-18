import numpy as np
from numpy.typing import NDArray

from abc import ABC, abstractmethod
from typing import Optional

class PropagatorBase(ABC):  
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_T_propagator(self, t: Optional[float]) -> NDArray[np.complex128]:
        """get the kinetic energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Could be None for time-independent Hamiltonian..

        Returns:
            ArrayLike: the kinetic energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_V_propagator(self, t: Optional[float]) -> NDArray[np.complex128]:
        """get the potential energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Could be None for time-independent Hamiltonian..

        Returns:
            ArrayLike: the potential energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_half_T_propagator(self, t: Optional[float]) -> NDArray[np.complex128]:
        """get the half kinetic energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Could be None for time-independent Hamiltonian..

        Returns:
            ArrayLike: the half kinetic energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_half_V_propagator(self, t: Optional[float]) -> NDArray[np.complex128]:
        """get the half potential energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Could be None for time-independent Hamiltonian..

        Returns:
            ArrayLike: the half potential energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def nstates(self) -> int:
        """Get the number of quantum states in the system.

        Returns:
            int: the number of quantum states in the system.
        """
        pass
    
    @abstractmethod
    def ngrid(self) -> int:
        """Get the number of grid points in the system.

        Returns:
            int: the number of grid points in the system.
        """
        pass    