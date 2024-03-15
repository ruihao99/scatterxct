# %% The package
import numpy as np

from numbers import Real, Complex
from typing import Union    

from .pulse_base import PulseBase


class Morlet(PulseBase):
    def __init__(
        self,
        A: Union[complex, float] = 1.0,
        t0: float = 0.0,
        tau: float = 1.0,
        Omega: float = 1,
        phi: float = 0.0,
        cache_length: int = 40
    ) -> None:
        super().__init__(Omega=Omega, cache_length=cache_length)
        
        if not isinstance(self.Omega, Real):
            raise ValueError(f"For Morlet, the carrier frequency {self.Omega=} should be a real number, not {type(self.Omega)}.")

        self.A : Union[complex, float] = A
        self.t0 : float = t0
        self.tau : float = tau
        self.phi : float = phi

    def __repr__(self) -> str:
        return f"Morlet(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"

    def _pulse_func(self, time: float) -> Union[complex, float]:
        self.Omega: float
        return Morlet.morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)

    @staticmethod
    def morlet_pulse(
        A: Union[float, complex],
        t0: float,
        tau: float,
        Omega: float,
        phi: float,
        time: float 
    ) -> complex:
        return A * np.exp(-1j * (Omega * (time - t0) + phi)) * np.exp(-0.5 * (time - t0)**2 / tau**2)


# %% the temperary testting/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
    p = Morlet(A=1, t0=4, tau=1, Omega=10, phi=0)
    
    t = np.linspace(-0, 12, 3000)
    sig = np.array([p(tt) for tt in t])
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, np.abs(sig), lw=.5, label="Morlet ABS")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, np.real(sig), lw=.5, label="Morlet Real")
    ax.plot(t, np.imag(sig), lw=.5, label="Morlet Imag")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()

# %% the __main__ test code
if __name__ == "__main__":
    _debug_test()

# %%
