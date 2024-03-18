import numpy as np

from scatterxct.core.global_control import DR_MAX
from scatterxct.core.fft_utils import nearest_number_with_small_prime_factors
from scatterxct.core.wavefunction.wavepacket import estimate_a_from_k0

from typing import Tuple, Optional
from enum import Enum, unique

@unique
class ScatterDirection(Enum):
    FROM_LEFT = 1
    FROM_RIGHT = 2


def estimate_R_lims(
    R0: float, # the initial position of the nuclei wavepacket
    k0: float, # the initial momentum of the nuclei wavepacket
    ngrid: int=256, # the number of grid points for real space descretization
    scatter_region_center: float=0.0 # the center of the scattering region
) -> Tuple[Tuple[float, float], int]:
    """Estimate the real space boundaries and the number of grid points for an exact scattering calculation.

    Args:
        R0 (float): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[Tuple[float, float], int]: _description_
    """
    N_MAX: int = 1000 # break the loop if the number of iterations exceeds this value
    SAFTY_FACTOR_K: float = 1.5 # make sure the kgrid is large enough
    SAFTY_FACTOR_R: float = 6 # make sure the wavepacket is fully contained in the grid
    
    scatter_direction = ScatterDirection.FROM_LEFT if k0 > 0 else ScatterDirection.FROM_RIGHT
    
    # Estimate the width of the wavepacket
    a = estimate_a_from_k0(k0)
    n_stddev = 5
    R_lims: Optional[Tuple[float, float]] = None     
    
    
    while True and (n_stddev < N_MAX):
        R_lims = (scatter_region_center - n_stddev * a, scatter_region_center + n_stddev * a)
        dR: float = (R_lims[1] - R_lims[0]) / ngrid
        kgrid = np.fft.fftfreq(ngrid, dR) * 2 * np.pi
        
        flag_kgrid = np.abs(kgrid).max() > np.abs(SAFTY_FACTOR_K * k0)
        if scatter_direction == ScatterDirection.FROM_LEFT:
            flag_rgrid = R_lims[0] < R0 - SAFTY_FACTOR_R * a
        elif scatter_direction == ScatterDirection.FROM_RIGHT:
            flag_rgrid = R_lims[1] > R0 + SAFTY_FACTOR_R * a
        else:
            raise ValueError(f"Invalid scatter direction: {scatter_direction}")
        
        if flag_kgrid and flag_rgrid:
            break        
        else:
            n_stddev += 1
    dR = (R_lims[1] - R_lims[0]) / ngrid
    # if the while exhausts the number of iterations, 
    # or the R grid exceeds the global DR_MAX,
    # increase the ngrid and try again with ngrid + 32
    if (n_stddev >= N_MAX) or (dR > DR_MAX):
        return estimate_R_lims(R0, k0, ngrid + 16, scatter_region_center)
    # elif succesful break, return the R_lims and ngrid
    else:
        # find the nearest number with only small prime factors
        # to harness the power of the FFT!
        ngrid_nearest_prime_factored: int = nearest_number_with_small_prime_factors(ngrid)
        print(f"ngrid: {ngrid}", f"ngrid_nearest_prime_factored: {ngrid_nearest_prime_factored}")
        return R_lims, ngrid_nearest_prime_factored
      