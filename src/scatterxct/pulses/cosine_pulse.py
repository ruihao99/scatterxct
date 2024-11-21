# %%
# import numpy as np
import numpy as np

from scatterxct.mytypes import RealArray
from scatterxct.pulses.pulse_base import PulseBase
from scatterxct.pulses.canonical_phase import cp_cosine

from dataclasses import dataclass

@dataclass
class CosinePulse(PulseBase):

    @classmethod
    def init(cls, A: float, omega: float, phi: float, laserwidth: float=0.02) -> 'CosinePulse':
        params = {'A': A, 'phi': phi}
        cp = cp_cosine(phi)
        return cls(
            omega=omega, laserwidth=laserwidth, cp=cp, params=params
        )

    @staticmethod
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return np.cos(omega * time + params['phi'])

    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return params['A'] * np.ones_like(time)
    
# %%
def test_cosine_pulse():
    pulse = CosinePulse.init(1.0, 1.0, 0.0)
    assert pulse.signal(0.0) == 1.0
    assert pulse.signal(1.0) == np.cos(1.0)
    assert pulse.signal(2.0) == np.cos(2.0)
    assert pulse.signal(3.0) == np.cos(3.0)
    print("All tests passed.")

# %%
if __name__ == "__main__":
    test_cosine_pulse()
# %%
