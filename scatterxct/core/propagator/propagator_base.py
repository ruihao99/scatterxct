from numpy.typing import ArrayLike

from abc import ABC, abstractmethod
from typing import Optional

class PropagatorBase(ABC):  
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def get_T_propagator(self, t: Optional[float]=None) -> ArrayLike:
        """get the kinetic energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Defaults to None.

        Returns:
            ArrayLike: the kinetic energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_V_propagator(self, t: Optional[float]=None) -> ArrayLike:
        """get the potential energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Defaults to None.

        Returns:
            ArrayLike: the potential energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_half_T_propagator(self, t: Optional[float]=None) -> ArrayLike:
        """get the half kinetic energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Defaults to None.

        Returns:
            ArrayLike: the half kinetic energy propagator at time t.
        """
        pass
    
    @abstractmethod
    def get_half_V_propagator(self, t: Optional[float]=None) -> ArrayLike:
        """get the half potential energy propagator at time t.

        Args:
            t (Optional[float], optional): the simulation time. Defaults to None.

        Returns:
            ArrayLike: the half potential energy propagator at time t.
        """
        pass