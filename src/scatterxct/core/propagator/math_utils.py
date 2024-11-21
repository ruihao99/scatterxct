# %%
import numpy as np
import numpy.linalg as LA
from scipy.linalg import expm
from numpy.typing import NDArray
from numba import jit, njit
from scatterxct.core.discretization import fddrv
from scatterxct.hamiltonian.linalg import diagonalize_and_project

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
        V[:, :, ii] = np.dot(V_ii, U_ii.conj().T)
    return V

def get_diabatic_V_propagators_expm(
    H: NDArray[np.complex128],
    V: NDArray[np.complex128],
    dt: float
) -> NDArray[np.complex128]:
    V[:] = expm(-1.0j * H.transpose(2, 0, 1) * dt).transpose(1, 2, 0)
    return V

def get_nac(
    G: NDArray[np.float64],
    E: NDArray[np.float64],
):
    # evaluate the NACs from the generalized gradient G
    # using the Hellmann-Feynman theorem
    dim, ngrid = E.shape
    D = np.zeros((dim, dim, ngrid), dtype=np.complex128)
    for ii in range(ngrid):
        for jj in range(dim):
            for kk in range(jj+1, dim):
                D[jj, kk, ii] = G[jj, kk, ii] / (E[kk, ii] - E[jj, ii])
                D[kk, jj, ii] = -np.conj(D[jj, kk, ii])
    return D

def get_adiabatic_V_propagators_expm(
    E: NDArray[np.float64],
    U: NDArray[np.complex128],
    G: NDArray[np.float64],
    k: NDArray[np.complex128],
    dt: float,
    dx: float,
    mass: float,
    accuracy: int=2,
) -> NDArray[np.complex128]:
    dim, ngrid = E.shape
    # velocity matrix, v = hbar / im * d/dx / mass (shape: (ngrids, ngirds))
    # v_mat = fddrv(length=ngrid, order=1, accuracy=accuracy) / (mass * dx * 1j)

    # The non-adiabatic coupling vectors D (shape: (ngrids, dim, dim))
    D = get_nac(G, E)

    ###
    # The non-adiabatic coupling Matrix K = D v (shape: (ngrids, dim, dim))
    ###
    # allocate the memory for K
    K = np.zeros((dim, dim, ngrid), dtype=np.complex128)

    # evaluate the non-adiabatic coupling matrix K in momentum space
    # (here velocity is diagonal)
    K[:] = D[:] / mass


    # The adiabatic potential propagator V (shape: (ngrids, dim, dim))
    V = np.zeros((dim, dim, ngrid), dtype=np.complex128)
    for ii in range(ngrid):
        Heff = np.diagflat(E[:, ii]) - 1j * K[:, :, ii]
        V[:, :, ii] = expm(-1j * Heff * dt)
    return V


@njit
def expm_numba(H: np.ndarray, dt: float) -> np.ndarray:
    E, U = np.linalg.eigh(H)
    return np.dot(U, np.dot(np.diag(np.exp(-1j * E * dt)), U.T.conj()))

@njit
def get_propagators(
    H_arr: np.ndarray,
    dt: float,
) -> np.ndarray:
    dim, dim, ngrid = H_arr.shape
    propagators = np.zeros((dim, dim, ngrid), dtype=np.complex128)
    for ii in range(ngrid):
        H = np.ascontiguousarray(H_arr[:,:,ii])
        propagators[:,:,ii] = expm_numba(H, dt)
    return propagators

def evaluate_propagator_laser(
    Hd: NDArray[np.complex128],  # mol Hamiltonian in diabatic representation
    Md: NDArray[np.complex128],  # dipole moment in diabatic representation
    Et: float,     # time-dependent electric field
    dt: float,
) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128]]:
    nstates, _, ngrid = Hd.shape
    H = Hd - Md * Et
    return H, get_propagators(H, dt)

