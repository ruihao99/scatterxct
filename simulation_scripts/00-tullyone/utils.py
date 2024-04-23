import numpy as np
from scipy.interpolate import interp1d

from scatterxct.models.tullyone import TullyOnePulseTypes

import os

def get_tully_one_delay_time(R0: float, P0: float, ) -> float:
    import pymddrive
    
    if R0 != -10.0:
        raise ValueError("Only R0 = -10.0 is supported at the moment.")
    
    base = os.path.dirname(pymddrive.__path__[0])
    
    tully_one_delay_time_tabulate = os.path.join(
        base, 'tabulate', 'tully_one_delay_time.txt'
    )
    P0_tab, delay_tab = np.loadtxt(tully_one_delay_time_tabulate, unpack=True)
    interp_func = interp1d(P0_tab, delay_tab, kind='cubic')
    return interp_func(P0)
 
def linspace_log10(start, stop, num=50):
    return np.power(10, np.linspace(np.log10(start), np.log10(stop), num))

def sample_sigmoid(x_left: float, x_right: float, n: int) -> np.ndarray:
    x_center = (x_left + x_right) / 2
    p0_seg1 = np.sort(x_center + x_left - linspace_log10(x_left, x_center, n // 2))
    dp = p0_seg1[-1] - p0_seg1[-2]
    p0_seg2 = x_center - x_left + dp + np.sort(linspace_log10(x_left, x_center, n - n // 2))
    return np.concatenate((p0_seg1, p0_seg2))

def get_tullyone_p0_list(nsamples: int, pulse_type: TullyOnePulseTypes=TullyOnePulseTypes.NO_PULSE) -> np.ndarray:
    if pulse_type.value == TullyOnePulseTypes.NO_PULSE.value or pulse_type.value == TullyOnePulseTypes.PULSE_TYPE3.value:
        p0_bounds_0 = (5.0, 12.0); n_bounds_0 = nsamples // 2
        p0_bounds_1 = (13, 35); n_bounds_1 = nsamples - n_bounds_0
    elif pulse_type.value == TullyOnePulseTypes.PULSE_TYPE1.value or pulse_type.value == TullyOnePulseTypes.PULSE_TYPE2.value:
        p0_bounds_0 = (5.0, 19); n_bounds_0 = nsamples // 3 * 2
        p0_bounds_1 = (20, 35); n_bounds_1 = nsamples - n_bounds_0

    p0_segment_0 = sample_sigmoid(*p0_bounds_0, n_bounds_0)
    p0_segment_1 = np.linspace(*p0_bounds_1, n_bounds_1)
    return np.concatenate((p0_segment_0, p0_segment_1))

def estimate_dt(Omega: float, dt: float = 0.1) -> float:
    SAFTY_FACTOR: float = 10.0
    T: float = 2 * np.pi / Omega
    if dt > T / SAFTY_FACTOR:
        return T / SAFTY_FACTOR
    else:
        return dt