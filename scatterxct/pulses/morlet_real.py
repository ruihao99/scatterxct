# %% The package
import numpy as np

from .pulse_base import PulseBase

class MorletReal(PulseBase):
    def __init__(
        self,
        A: float = 1,
        t0: float = 0,
        tau: float = 1,
        Omega: float = 1,
        phi: float = 0,
        cache_length: int = 40
    ):
        super().__init__(Omega=Omega, cache_length=cache_length)
        self.A = A
        self.t0 = t0
        self.tau = tau
        self.phi = phi
        if not isinstance(self.Omega, float):
            raise ValueError(f"For MorletReal, the carrier frequency {self.Omega=} should be a real number, not {type(self.Omega)}.")

    def __repr__(self) -> str:
        return f"MorletReal(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"

    def __call__(self, time: float):
        return super().__call__(time)

    def _pulse_func(self, time: float) -> float:
        self.Omega: float
        return MorletReal.real_morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)

    @staticmethod
    def real_morlet_pulse(
        A: float,
        t0: float,
        tau: float,
        Omega: float,
        phi: float,
        time: float
    ):
        return A * np.cos(Omega * (time - t0) + phi) * np.exp(-0.5 * (time - t0)**2 / tau**2)
        # return A * np.cos(Omega * time) * np.exp(-0.5 * (time - t0)**2 / tau**2)
    
# %% The temperary testting/debugging code
def _test_debug_morlet_real():
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    p1 = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    t = np.linspace(-0, 12, 3000)
    sig = [p1(tt) for tt in t]
    fig = plt.figure(figsize=(3, 2), dpi=200)

    ax = fig.add_subplot(111)
    ax.plot(t, sig, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
# %% the __main__ testing/debugging code
if __name__ == "__main__":
    _test_debug_morlet_real()

# %%
