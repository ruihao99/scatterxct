import numpy as np
from numpy.typing import ArrayLike
from numba import jit
import scipy.sparse as sp

from .hamiltonian_base import HamiltonianBase
from .math_utils import diagonalize_hamiltonian_history, diabatic_to_adiabatic

from typing import Tuple, Union, Optional

def evaluate_hamiltonian(
    t: float, R: Union[float, ArrayLike], hamiltonian: HamiltonianBase, enable_evec_following: bool=True
) -> Tuple[ArrayLike]:
    H = hamiltonian.H(t, R)
    dHdR = hamiltonian.dHdR(t, R)
    if enable_evec_following:
        evals, evecs = diagonalize_hamiltonian_history(H, hamiltonian.last_evecs) 
        # hamiltonian.update_last_evecs(evecs)
    else:
        evals, evecs = diagonalize_hamiltonian_history(H, None)
    return H, dHdR, evals, evecs

def evaluate_nonadiabatic_couplings(
    dHdR: ArrayLike,
    evals: ArrayLike,
    evecs: ArrayLike, 
    out_d: Optional[ArrayLike]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    dHdR = dHdR.todense() if sp.issparse(dHdR) else dHdR
    if dHdR.ndim == 2:
        d, F = _evaluate_nonadiabatic_couplings_scalar(dHdR, evals, evecs, out_d)
        # return _evaluate_nonadiabatic_couplings_scalar(dHdR, evals, evecs, out_d)
    elif dHdR.ndim == 3:
        # return _evaluate_nonadiabatic_couplings_vector(dHdR, evals, evecs, out_d)
        d, F = _evaluate_nonadiabatic_couplings_vector(dHdR, evals, evecs, out_d)
    else:
        raise ValueError(f"The number of dimensions of dHdR must be 2 or 3, but the input dHdR has {dHdR.ndim} dimensions.")
    return d, F

def _evaluate_nonadiabatic_couplings_scalar(
    dHdR: ArrayLike, 
    evals: ArrayLike,
    evecs: ArrayLike,
    out_d: Union[ArrayLike, None]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    # assert dHdR.ndim == 2
    if out_d is not None:
        assert out_d.shape == dHdR.shape
        diabatic_to_adiabatic(dHdR, evecs, out=out_d)
    else:
        out_d = diabatic_to_adiabatic(dHdR, evecs)
    F = -np.diagonal(out_d).real
    out_d = _post_process_d_scalar(out_d, evals)
    return out_d, F
    

def _evaluate_nonadiabatic_couplings_vector(
    dHdR: ArrayLike, 
    evals: ArrayLike,
    evecs: ArrayLike,
    out_d: Union[ArrayLike, None]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    if out_d is not None:
        assert out_d.shape[0] == dHdR.shape[-1]
        ndim_cl = dHdR.shape[-1]
        for ii in range(ndim_cl):
            diabatic_to_adiabatic(dHdR[:, :, ii], evecs, out=out_d[ii])
    else:
        out_d = np.array([diabatic_to_adiabatic(dHdR[:, :, ii], evecs) for ii in range(dHdR.shape[-1])])
    F = -np.diagonal(out_d, axis1=1, axis2=2).T.astype(np.float64)
    out_d = _post_process_d_vector(out_d, evals)
    return out_d, F
 
@jit(nopython=True)  
def _post_process_d_scalar(out_d: ArrayLike, evals: ArrayLike) -> ArrayLike:
    ndim = out_d.shape[0]
    for ii in range(ndim):
        out_d[ii, ii] = 0.0
        for jj in range(ii+1, ndim):
            dE = evals[jj] - evals[ii]
            out_d[ii, jj] /= dE
            out_d[jj, ii] /= -dE
    return out_d

@jit(nopython=True)
def _post_process_d_vector(out_d: ArrayLike, evals: ArrayLike) -> ArrayLike:
    ndim_qm = evals.shape[0]
    ndim_cl = out_d.shape[0]
    for ii in range(ndim_cl):
        for jj in range(ndim_qm):
            out_d[ii, jj, jj] = 0.0
            for kk in range(ndim_qm):
                dE = evals[kk] - evals[jj]
                out_d[ii, jj, kk] /= dE
                out_d[ii, kk, jj] /= -dE
    return out_d
