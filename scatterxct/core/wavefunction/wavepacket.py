# %%
import numpy as np
from numpy.typing import ArrayLike

from typing import Optional
import warnings

def gaussian_wavepacket(R: ArrayLike,  R0: float, k0: float, a: Optional[float]=None) -> ArrayLike:
    """The wavefunction of a Gaussian wavepacket.

    Args:
        R (ArrayLike): real space grid
        a (float): the width of the nuclear wavepacket, usually taken as 20 * 1 / k0
        R0 (float): the initial expectation value of the position of the nuclear wavepacket
        k0 (float): the initial momentum of the nuclear wavepacket

    Returns:
        ArrayLike: the wavefunction in real space
    """
    if a is None:
        a = estimate_a_from_k0(k0)
    elif isinstance(a, float):
        a_estimated = estimate_a_from_k0(k0)
        if a != a_estimated:
            warnings.warn(f"The provided width of the wavepacket is different from the estimated value: {a_estimated}. You are not following the recommended practice. Please make sure you know what you are doing.")
        
    prefactor: float = 1.0 / np.sqrt(a * np.sqrt(np.pi))
    exp_re: ArrayLike = np.exp(-0.5 * (R - R0)**2 / a**2)
    exp_im: ArrayLike = np.exp(1j * k0 * R)
    return prefactor * exp_re * exp_im

def gaussian_wavepacket_kspace(k: ArrayLike, a: float, R0: float, k0: float) -> ArrayLike:
    """The analytical fourier transform of a Gaussian wavepacket.

    Args:
        k (ArrayLike): reciprocal space grid
        a (float): the width of the nuclear wavepacket, usually taken as 20 * 1 / k0
        R0 (float): the initial expectation value of the position of the nuclear wavepacket
        k0 (float): the initial momentum of the nuclear wavepacket

    Returns:
        ArrayLike: the wavefunction in reciprocal space
    """
    a *= 2 * np.sqrt(np.pi)
    # a *= np.pi
    prefactor: float = np.sqrt(a / np.sqrt(np.pi))
    exp_re: ArrayLike = np.exp(-0.5 * a**2 * (k - k0)**2)
    exp_im: ArrayLike = np.exp(-1j * (k - k0) * R0)
    return prefactor * exp_re * exp_im

def estimate_a_from_k0(k0: float) -> float:
    """Estimate the width of the wavepacket from the initial momentum.

    Args:
        k0 (float): The initial momentum

    Returns:
        float: the width of the wavepacket

    Reference:
    ---------
    [1] Tully, J. C. (1990). Molecular dynamics with electronic transitions. The Journal of Chemical Physics.
        "The width parameter was typically chosen to be 20 times the inverse of the initial momentum."
        (note that the tully paper uses gaussian wavepackets as exp(-ikx) * exp(-(x-x0)^2/a^2) )
        here we use exp(ikx) * exp(-(x-x0)^2/(2a^2)). Hence the factor of sqrt(2) in the denominator.
    """
    return 1.0 / k0 * 10.0 * np.sqrt(2.0)
# %%
