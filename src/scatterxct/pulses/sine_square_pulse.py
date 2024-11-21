# %%
# import numpy as np
import numpy as np

from scatterxct.mytypes import RealArray
from scatterxct.pulses.pulse_base import PulseBase    
from scatterxct.pulses.canonical_phase import cp_sine


class SineSquarePulse(PulseBase):
    @classmethod
    def init(cls, A: float, N: int, omega: float, phi: float, laserwidth: float=0.02) -> 'SineSquarePulse':
        cp = cp_sine(phi)
        params = {'A': A, 'N': N, 'phi': phi}
        return cls(omega=omega, laserwidth=laserwidth, cp=cp, params=params)
    
    @staticmethod
    def carrier_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        return np.sin(omega * time + params['phi'])
    
    @staticmethod
    def envelope_func(time: RealArray, omega: float, **params: float) -> np.ndarray:
        A, N = params['A'], params['N']
        return A * np.square(np.sin(0.5*omega/N*time))
    
# %%
def test_sine_square_pulse():
    pulse = SineSquarePulse.init(1.0, 8, 10.0, np.pi)
    assert pulse.signal(0.0) == 0.0
    assert pulse.signal(1.0) == np.square(np.sin(0.5*10.0/8*1.0)) * np.sin(10.0 * 1.0 + np.pi)
    assert pulse.signal(2.0) == np.square(np.sin(0.5*10.0/8*2.0)) * np.sin(10.0 * 2.0 + np.pi)
    assert pulse.signal(3.0) == np.square(np.sin(0.5*10.0/8*3.0)) * np.sin(10.0 * 3.0 + np.pi)
    print("All tests passed.")
    
    
# %%
if __name__ == "__main__":
    test_sine_square_pulse() 
# %%
