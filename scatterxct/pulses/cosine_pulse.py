# %%
import numpy as np

from .pulse_base import PulseBase

from typing import TypeAlias    
from numbers import Real


AnyNumber : TypeAlias = int | float | complex 
RealNumber : TypeAlias = int | float


class CosinePulse(PulseBase):
    def __init__(
        self,
        A: AnyNumber = 1,         # the amplitude of the cosine pulse
        Omega: RealNumber = 1,    # the carrier frequency of the pulse
        cache_length: int = 40
    ):
        super().__init__(Omega, cache_length)
        self.A = A
        
        if not isinstance(self.Omega, Real):
            raise ValueError(f"For CosinePulse, the carrier frequency {self.Omega=} should be a real number, not {type(self.Omega)}.")
    
    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        self.Omega: RealNumber
        return self.A * np.cos(self.Omega * time)
# %%
