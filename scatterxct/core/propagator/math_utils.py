# %%
import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray
from numba import jit

@jit(nopython=True)
def get_diabatic_V_propagators(
    H: NDArray[np.complex128], 
    V: NDArray[np.complex128], 
    dt: float, 
    E: NDArray[np.float64], 
    U: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Get the diabatic propagators for the diabatic representation.

    Returns:
        ArrayLike: the diabatic propagators
    """
    _, _, ngrid = H.shape
    for ii in range(ngrid):
        Hii = H[:, :, ii]
        evals, evecs = LA.eigh(Hii)
        V[:, :, ii] = np.dot(np.diagflat(np.exp(-1j * evals * dt)), evecs.conj().T)
        V[:, :, ii] = np.dot(evecs, V[:, :, ii])
        E[:, ii], U[:, :, ii] = evals, evecs
    return V 
