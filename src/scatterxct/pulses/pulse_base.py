# %% 
"""
This module defines the Pulse for handling pulse signals.
"""
import numpy as np

from scatterxct.mytypes import RealArray

from dataclasses import dataclass, field
from typing import Dict
from collections import namedtuple


@dataclass
class PulseBase:
    omega: float = field(default=float('nan'))
    laserwidth: float = field(default=float('nan'))
    cp: complex = field(default=complex('nan'))
    params: Dict[str, float] = field(default_factory=dict)
    
    def signal(self, time: RealArray) -> np.ndarray:
        carrier = self.carrier_func(time, self.omega, **self.params)
        envelope = self.envelope_func(time, self.omega, **self.params)
        return carrier * envelope   
    
    def derivative(self, time: RealArray, h: float=1.0E-8) -> np.ndarray:
        s1 = self.signal(time - h / 2)
        s2 = self.signal(time + h / 2)
        return (s2 - s1) / h
    
    def get_Epsilon(self, time: RealArray) -> np.ndarray:
        return self.envelope_func(time, self.omega, **self.params) * self.cp
        
    def get_Epsilon_derivative(self, time: RealArray, h: float=1.0E-8) -> np.ndarray:
        e1 = self.envelope_func(time - h / 2, self.omega, **self.params)
        e2 = self.envelope_func(time + h / 2, self.omega, **self.params)
        return (e2 - e1) / h * self.cp
        
    
    @staticmethod    
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        raise NotImplementedError("Please implement the carrier function in the derived class.")
    
    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        raise NotImplementedError("Please implement the envelope function in the derived class.")
        
# %%
