# %%
from .pulse_base import PulseBase

from collections import OrderedDict

class MultiPulse(PulseBase):
    def __init__(
        self,
        *pulses: PulseBase,
        cache_length: int = 40
    ):
        """
        Initialize a MultiPulse object.

        Args:
            *pulses (Pulse): Variable number of Pulse objects.
            cache_length (int): The maximum length of the cache.
        """
        self.pulses = pulses
        self._cache = OrderedDict()
        self._cache_length = cache_length
        
    def __call__(self, time: float):
        """
        Call the MultiPulse object with a time value.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        if time in self._cache:
            # print(f"From <class MultiPulse>: Retrieving value from the cache for time {time}")
            return self._cache[time]
        else:
            # print(f"From <class MultiPulse>: Calculating the value for time {time}")
            return self._post_call(time)
    
    def _post_call(self, time: float):
        """
        Perform post-call operations.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        self._cache[time] = self._pulse_func(time)
        if len(self._cache) > self._cache_length:
            self._cache.popitem(last=False)
        return self._cache[time]
    
    def _pulse_func(self, time: float):
        """
        Calculate the pulse value at the given time.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated pulse value.
        """
        return sum(p(time) for p in self.pulses)
# %% testing/debugging code
def _test_debug_multipulse():
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
    from pymddrive.pulses.gaussian import Gaussian
    from pymddrive.pulses.morlet_real import MorletReal
    
    p1 = Gaussian(A=1, t0=0, tau=1)
    p2 = MorletReal(A=1, t0=10, tau=1, Omega=10, phi=0)
    mp = MultiPulse(p1, p2)
    time = np.linspace(-10, 30, 1000)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, [mp(t) for t in time])
    plt.show()
    
# %%
if __name__ == "__main__":
    _test_debug_multipulse()
# %%
