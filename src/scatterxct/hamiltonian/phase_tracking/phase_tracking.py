# %%
import numpy as np

from dataclasses import dataclass

from scatterxct.hamiltonian.phase_tracking.exteded_pt import parallel_transport
from scatterxct.hamiltonian.phase_tracking.sharc_seb_mai import mai_projection

@dataclass
class PhaseTracking:
    phase_tracking_method: str

    def diag(self, Hdiab, U_last=None):
        # numerical diagonalization
        E, U = np.linalg.eigh(Hdiab)

        # if U_last is provided, proceed with phase tracking
        if U_last is not None:
            if self.phase_tracking_method == "none":
                # no phase tracking
                pass
            # elif self.phase_tracking_method == "parallel_transport":
            elif self.phase_tracking_method == "zeyu2019":
                # extended parallel transport
                U = parallel_transport(curevt=U_last, nextevt=U, )
            elif self.phase_tracking_method == "mai2015":
                # Mai et al. 2015, projection method used by SHARC
                U = mai_projection(E, U, U_last)
            else:
                raise ValueError(f"Unknown phase tracking method: {self.phase_tracking_method}")
        return E, U


# %%
