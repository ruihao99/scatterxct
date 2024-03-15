# %% 
"""
This module defines the Pulse and MultiPulse classes for handling pulse signals.
"""
from typing import Union, TypeAlias, Optional
from numbers import Real
from collections import OrderedDict
from abc import ABC, abstractmethod

# TypeOmega: TypeAlias = Union[int, float, None]

class PulseBase(ABC):
    def __init__(self, Omega: Optional[float]=None, cache_length: int=30):
        self.Omega = Omega  
        self._cache: OrderedDict = OrderedDict()
        self._cache_length = cache_length
        
    def __call__(self, time: float):
        if time in self._cache:
            return self._cache[time]
        else:
            return self._post_call(time)
        
    def _post_call(self, time: float):
        self._cache[time] = self._pulse_func(time)
        if len(self._cache) > self._cache_length:
            self._cache.popitem(last=False)
        return self._cache[time]
    
    @abstractmethod
    def _pulse_func(self, time: float):
        pass

    def set_Omega(self, Omega: float):
        """
        Set the carrier frequency of the pulse.

        Args:
            Omega (float): The carrier frequency.
        """
        if isinstance(Omega, Real):
            self.Omega = Omega
        else:
            raise ValueError(f"After the pulse has been initialized, you can only set the carrier frequency with a real number, not {Omega}")
    
def get_carrier_frequency(pulse: PulseBase) -> Optional[float]:
    return pulse.Omega        
# %%
