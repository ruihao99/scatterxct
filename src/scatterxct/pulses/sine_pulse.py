# %%
"""
This module defines the Pulse for handling pulse signals.
"""
# import numpy as np
import numpy as np

from scatterxct.mytypes import RealArray
from scatterxct.pulses.pulse_base import PulseBase
from scatterxct.pulses.canonical_phase import cp_sine

from dataclasses import dataclass

@dataclass
class SinePulse(PulseBase):

    @classmethod
    def init(cls, A: float, omega: float, phi: float, laserwidth: float=0.02) -> 'SinePulse':
        cp = cp_sine(phi)
        params = {'A': A, 'phi': phi}
        return cls(omega=omega, laserwidth=laserwidth, cp=cp, params=params)

    @staticmethod
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return np.sin(omega * time + params['phi'])

    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return params['A'] * np.ones_like(time)

# %%
def test_sine_pulse():
    pulse = SinePulse.init(1.0, 1.0, np.pi/2)
    assert pulse.signal(0.0) == 1.0
    assert pulse.signal(1.0) == np.sin(1.0 + np.pi/2)
    assert pulse.signal(2.0) == np.sin(2.0 + np.pi/2)
    assert pulse.signal(3.0) == np.sin(3.0 + np.pi/2)
    print("All tests passed.")

# %%
if __name__ == "__main__":
    test_sine_pulse()
# %%
