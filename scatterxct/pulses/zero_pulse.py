from .pulse_base import PulseBase

from typing import TypeAlias

AnyNumber : TypeAlias = int | float | complex 
RealNumber : TypeAlias = int | float

class ZeroPulse(PulseBase):
    def __init__(self, cache_length: int = 40):
        super().__init__(None, cache_length)
        
    def _pulse_func(self, t: RealNumber) -> AnyNumber:
        return 0