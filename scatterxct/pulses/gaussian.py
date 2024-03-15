# %%
import numpy as np

from .pulse_base import PulseBase
from .morlet import Morlet
from .morlet_real import MorletReal

from typing import Union
from numbers import Complex, Real

class Gaussian(PulseBase):
    def __init__(
        self, 
        A: float = 1,
        t0: float = 0,
        tau: float = 1, 
        cache_length: int = 30
    )->None:
        super().__init__(None, cache_length)
        self.A = A  
        self.tau = tau
        self.t0 = t0
        
    def __repr__(self) -> str:
        return f"Gaussian(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega})"
    
    def _pulse_func(self, time: float):
        return Gaussian.gaussian_pulse(self.A, self.t0, self.tau, time)
        
    @staticmethod
    def gaussian_pulse(
        A: Union[float, Complex],
        t0: float, 
        tau: float, 
        time: float
    )->Complex:
        return A * np.exp(-0.5 * (time - t0)**2 / tau**2) 
    
    @classmethod
    def from_quasi_floquet_morlet_real(cls, morlet: MorletReal)->"Gaussian":
        """Convert a <MorletReal> pulse to a <Gaussian> pulse by taking the quasi-Floquet transform.
        A <MorletReal> pulse is a real-valued Morlet wavelet, which is a real-valued Gaussian wavelet modulated by a cosine function.
        The cosine modulation can be removed by taking the quasi-Floquet transform, which results in a <Gaussian> pulse.
        Here, we use a convention that the returned <Gaussian> pulse modulates the upper right non-diagonal quadrant of the quasi-Floquet Hamiltonian.

        Args:
            morlet (MorletReal): A real-valued Morlet wavelet.

        Returns:
            Gaussian: A <Gaussian> pulse resides in the upper right non-diagonal quadrant of the quasi-Floquet Hamiltonian.
        """
        t0 = morlet.t0; Omega = morlet.Omega; phi = morlet.phi
        if isinstance(Omega, Real):
            # phase_factor = np.exp(-1.0j * (Omega * t0 - phi))
            # gaussian_complex_A: Complex = morlet.A * phase_factor
            # return cls(A=gaussian_complex_A, t0=t0, tau=morlet.tau)
            return cls(A=morlet.A, t0=t0, tau=morlet.tau)
        else:
            raise ValueError(f"The carrier frequency {Omega=} of the MorletReal pulse should be a real number, not {type(Omega)}.")
    
    @classmethod
    def from_quasi_floquet_morlet(cls, morlet: Morlet)->"Gaussian":
        raise NotImplementedError("The method <from_quasi_floquet_morlet> is not implemented yet.")
    
# %% the temperary testting/debugging code
def _debug_test_from_quasi_floquet_morlet_real():
    pulse_morletreal = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    pulse_gaussian = Gaussian.from_quasi_floquet_morlet_real(pulse_morletreal)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
    t = np.linspace(-0, 12, 3000)
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    dat_morlet = np.array([pulse_morletreal(tt) for tt in t])
    dat_gaussian = np.array([pulse_gaussian(tt) for tt in t])
    ax.plot(t, dat_morlet, lw=.5, label="MorletReal")
    ax.plot(t, np.abs(dat_gaussian), lw=.5, label=r"Abs Gaussian")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()
    
    fig = plt.figure(figsize=(3, 2), dpi=200)   
    ax = fig.add_subplot(111)
    ax.plot(t, dat_gaussian.real, lw=.5, label=r"$\Re$ Gaussian")
    ax.plot(t, dat_gaussian.imag, lw=.5, label=r"$\Im$ Gaussian")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()
    
# %% the __main__ test code
if __name__ == "__main__":
    _debug_test_from_quasi_floquet_morlet_real() 
# %%
