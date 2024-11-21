# %% 
"""
This module defines the Pulse for handling pulse signals.
"""
# import numpy as np
import numpy as np

from scatterxct.mytypes import RealArray
from scatterxct.pulses.pulse_base import PulseBase    
from scatterxct.pulses.canonical_phase import cp_cosine

from dataclasses import dataclass

@dataclass
class MorletPulse(PulseBase):
    
    @classmethod    
    def init(cls, A: float, t0: float, tau: float, omega: float, phi: float, laserwidth: float=0.02) -> 'MorletPulse':
        cp = cp_cosine(phi - omega * t0)
        params = {'A': A, 't0': t0, 'tau': tau, 'phi': phi}
        return cls(
            omega=omega, laserwidth=laserwidth, cp=cp, params=params
        )
    
    @staticmethod
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        t0, phi = params['t0'], params['phi']
        return np.cos(omega * (time - t0) + phi)
    
    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        A, tau, t0 = params['A'], params['tau'], params['t0']
        return A * np.exp(-0.5 * (time - t0)**2 / tau**2)
    
# %%
    
# %%
def test_morlet_pulse():
    pulse = MorletPulse.init(1.0, 0.0, 1.0, 1.0, 0.0)
    assert pulse.signal(0.0) == 1.0
    assert pulse.signal(1.0) == np.cos(1.0) * np.exp(-0.5 * 1.0**2 / 1.0**2)
    assert pulse.signal(2.0) == np.cos(2.0) * np.exp(-0.5 * 2.0**2 / 1.0**2)
    assert pulse.signal(3.0) == np.cos(3.0) * np.exp(-0.5 * 3.0**2 / 1.0**2)
    
    print("All tests passed.")
    
# %%
if __name__ == "__main__":
    test_morlet_pulse() 
# %%
