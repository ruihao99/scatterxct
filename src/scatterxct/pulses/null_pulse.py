# %%
import numpy as np

from scatterxct.mytypes import RealArray
from scatterxct.pulses.pulse_base import PulseBase

from dataclasses import dataclass, field

@dataclass
class NullPulse(PulseBase):
    omega: float = float('nan')
    laserwidth: float = float('nan')
    cp: complex = complex('nan')
    params: dict = field(default_factory=dict)
    
    @staticmethod
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return np.zeros_like(time) 
    
    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return np.zeros_like(time)