# %%
import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm
from numpy.typing import NDArray
from numba import jit

from typing import Tuple

def diagonalization(
    H: NDArray[np.complex128],
) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    E, U = LA.eigh(H.transpose(2, 0, 1))
    return E.T, U.transpose(1, 2, 0)

@jit(nopython=True)
def get_diabatic_V_propagators(
    V: NDArray[np.complex128], 
    E: NDArray[np.float64], 
    U: NDArray[np.complex128],
    dt: float, 
) -> NDArray[np.complex128]:
    """Get the diabatic propagators for the diabatic representation.

    Returns:
        ArrayLike: the diabatic propagators
    """
    _, _, ngrid = V.shape
    E_ii = np.zeros((E.shape[0],), dtype=np.complex128)
    U_ii = np.zeros((U.shape[0], U.shape[1]), dtype=np.complex128)
    V_ii = np.zeros((U.shape[0], U.shape[1]), dtype=np.complex128)
    for ii in range(ngrid):
        E_ii[:] = np.ascontiguousarray(E[:, ii])
        U_ii[:, :] = np.ascontiguousarray(U[:, :, ii])
        V_ii[:, :] = np.dot(U_ii, np.diagflat(np.exp(-1j * E_ii * dt)))
        V[:, :, ii] = np.dot(U_ii.conj().T, V_ii)
    return V 

def get_diabatic_V_propagators_expm(
    H: NDArray[np.complex128], 
    V: NDArray[np.complex128],
    dt: float
) -> NDArray[np.complex128]:
    V[:] = expm(-1.0j * H.transpose(2, 0, 1) * dt).transpose(1, 2, 0)
    return V