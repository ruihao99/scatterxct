from .pulse_base import PulseBase
from .cosine_pulse import CosinePulse
from .sine_pulse import SinePulse

from typing import TypeAlias

AnyNumber : TypeAlias = int | float | complex 
RealNumber : TypeAlias = int | float

class UnitPulse(PulseBase):
    def __init__(self, A: AnyNumber=1.0, cache_length: int = 1000):
        super().__init__(None, cache_length)
        self.A = A
        
    def _pulse_func(self, t: RealNumber) -> AnyNumber:
        return self.A
    
    @classmethod
    def from_cosine_pulse(cls, cosine_pulse: "CosinePulse") -> "UnitPulse":
        return cls(A=cosine_pulse.A)
    
    @classmethod
    def from_sine_pulse(cls, sine_pulse: "SinePulse") -> "UnitPulse":
        return cls(A=sine_pulse.A)
    