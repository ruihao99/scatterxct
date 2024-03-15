# %% Use tully one to test the nonadiabatic hamiltonian abc class
import numpy as np
from numpy.typing import ArrayLike
from numba import jit

# from pymddrive.models.morse.morse import morse, d_morse_dR
from .morse import morse, d_morse_dR
from ..nonadiabatic_hamiltonian import HamiltonianBase

from typing import Union

class FourLevelMorse(HamiltonianBase):
    def __init__(
        self,
        VV: float = 0.02,
    ) -> None:
        super().__init__(dim=4)
        self.params_1 = {
            'k': 3.0,
            'a': 1.0,
            'r_e': 0.3,
            'offset': 0.02,
        }
        self.params_2 = {
            'k': 4.0,
            'a': 1.0,
            'r_e': 0.35,
            'offset': 0.1,
        }
        self.params_3 = {
            'k': 2.0,
            'a': 1.0,
            'r_e': 0.5,
            'offset': 0.0,
        }
        self.params_4 = {
            'k': 4.0,
            'a': 1.0,
            'r_e': 0.4,
            'offset': 0.05,
        }
        self.VV = VV
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian FourLevelMorse(VV={self.VV})"
   
    def H(self, t: float, R: Union[float, ArrayLike]) -> ArrayLike:
        V11 = morse(R, **self.params_1)
        V22 = morse(R, **self.params_2)
        V33 = morse(R, **self.params_3)
        V44 = morse(R, **self.params_4)
        H = np.zeros((4, 4), dtype=np.float64)
        is_real = np.isreal(R)
        if not is_real.all():
            raise ValueError("R must be real")
        if (is_real.ndim == 0):
            np.fill_diagonal(H, [V11, V22, V33, V44])
        elif isinstance(R, np.ndarray):
            assert R.ndim == 1, "R must be 1D array of shape (n,)"
            np.fill_diagonal(H, np.sum([V11, V22, V33, V44], axis=-1))
        else:
            raise ValueError(f"R must be float or 1D array. Instead got {type(R)}")
        H = fill_four_level_nondiagonal(H, self.VV)
        return H
    
    def dHdR(self, t: float, R: Union[float, ArrayLike]) -> ArrayLike:
        dV11dR = d_morse_dR(R, **self.params_1)
        dV22dR = d_morse_dR(R, **self.params_2)
        dV33dR = d_morse_dR(R, **self.params_3)
        dV44dR = d_morse_dR(R, **self.params_4)
        is_real = np.isreal(R)
        if not is_real.all():
            raise ValueError("R must be real")
        if (is_real.ndim == 0):
            dHdR = np.zeros((4, 4), dtype=np.float64)
        elif isinstance(R, np.ndarray):
            assert R.ndim == 1, "R of ndarray type must be 1D array of shape (n,)"
            dHdR = np.zeros((4, 4, R.size), dtype=np.float64)
        else:
            raise ValueError(f"R must be float or 1D array. Instead got {type(R)}")
        dHdR[0, 0] = dV11dR
        dHdR[1, 1] = dV22dR
        dHdR[2, 2] = dV33dR
        dHdR[3, 3] = dV44dR
        return dHdR 
        

@jit(nopython=True) 
def fill_four_level_nondiagonal(H: ArrayLike, VV: float) -> ArrayLike:
    for i in range(4):
        for j in range(i+1, 4):
            H[i, j] = VV
            H[j, i] = VV
    return H
        
# %% 
def evaluate_hamiltonian(R: ArrayLike, hamiltonian: HamiltonianBase) -> ArrayLike:
    dim = hamiltonian.dim
    number_of_R: int = R.shape[0]
    V_diab = np.zeros((number_of_R, dim), dtype=np.float64)
    V_adiab = np.zeros((number_of_R, dim), dtype=np.float64)
    for ii, rr in enumerate(R):
        H = hamiltonian.H(0, rr)
        dHdR = hamiltonian.dHdR(0, rr)
        V_diab[ii, :] = np.diagonal(H)
        evals, evecs = np.linalg.eigh(H)
        V_adiab[ii, :] = evals
    return V_diab, V_adiab

def _debug_test():
    import matplotlib.pyplot as plt
    
    # Diabatic potential energy surfaces
    r = np.linspace(0, 1.2, 1000)
    hamiltonian = FourLevelMorse()
    V_diab, V_adiab = evaluate_hamiltonian(r, hamiltonian)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for i in range(4):
        ax.plot(r, V_diab[:, i], label=f'V{i+1}{i+1}')
    ax.legend()
    ax.set_xlabel('R (a.u.)')
    ax.set_ylabel('Diabatic Potential (a.u.)')
    plt.show()  
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for i in range(4):
        ax.plot(r, V_adiab[:, i], label=f'E{i+1}')
    ax.legend()
    ax.set_xlabel('R (a.u.)')
    ax.set_ylabel('Adiabatic Energy (a.u.)')
    plt.show()
    
        
    
# %%
if __name__ == "__main__":
    _debug_test()
# %%
