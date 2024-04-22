# %%
import numpy as np
from numba import njit
from numpy.typing import ArrayLike
import scipy.linalg as LA


from typing import Optional, Tuple

""" Wavefunction math utilities """
""" Convention: """
""" - wavefunction shape: (ngrid, nstates) """
""" - Operator shape: (nstates, nstates, ngrid) """

def get_nuclear_density(psi: ArrayLike, dR: float) -> ArrayLike:
    """The nuclear density for each state.

    Args:
        psi (ArrayLike): the wavefunction
        dR (float): the grid spacing

    Returns:
        ArrayLike: the nuclear density for each state [shape: (ngrid, nstates)]
    """
    return np.abs(psi)**2 * dR

def trace_nuclear(psi: ArrayLike, operator: ArrayLike, dR: float) -> ArrayLike:
    """The averaged operator after tracing over the nuclear density.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        operator (ArrayLike): an operator with shape: (nstates, nstates, ngrid)

    Returns:
        ArrayLike: the averaged operator after tracing over the nuclear density [shape: (nstates, nstates)]
    """
    prob: ArrayLike = get_nuclear_density(psi, dR)
    reduced_operator: ArrayLike = np.sum(prob.T * operator, axis=2)
    return reduced_operator

def expected_value(psi: ArrayLike, operator: ArrayLike, dR: float) -> ArrayLike:
    """The averaged operator after tracing over the electronic and nuclear density.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        operator (ArrayLike): an operator with shape: (nstates, nstates, ngrid)
        dR (float): the grid spacing

    Returns:
        ArrayLike: the averaged operator after tracing over the electronic and nuclear density [shape: (ngrid, ngrid)]
    """
    prob_nuclear: ArrayLike = get_nuclear_density(psi, dR)
    electronic_expectation: ArrayLike = np.zeros((psi.shape[0], ), dtype=np.float64)
    if operator.ndim == 3:
        _trace_electronic(psi, operator, electronic_expectation)
        return np.sum(electronic_expectation[:, np.newaxis] * prob_nuclear)
    elif operator.ndim == 1:
        return np.sum(prob_nuclear * operator[:, np.newaxis])
    else:
        raise ValueError(f"The operator should have either 1 or 3 dimensions. Got {operator.ndim} dimensions.")
        

# @jit(nopython=True)
def _trace_electronic(psi: ArrayLike, operator: ArrayLike, out: ArrayLike) -> ArrayLike:
    ngrid = psi.shape[0]
    for ii in range(ngrid):
        rho = np.outer(psi[ii:, ], psi[ii:, ].conj())
        out[ii] = np.trace(np.dot(rho, operator[:, :, ii])).real
    return out

@njit
def calculate_mean_R(psi_R: ArrayLike, R: ArrayLike, dR: float) -> float:
    psi_ii = np.zeros((psi_R.shape[1], ), dtype=np.complex128)
    mean_R: float = 0.0
    for ii in range(psi_R.shape[0]):
        psi_ii[:] = psi_R[ii, :]
        mean_R += np.real(np.dot(psi_ii.conj(), psi_ii)) * R[ii] * dR
    return mean_R

@njit
def calculate_state_dependent_R(psi_R: ArrayLike, R: ArrayLike, dR: float) -> ArrayLike:
    psi_jj = np.zeros((psi_R.shape[0], ), dtype=np.complex128)
    R_out = np.zeros((psi_R.shape[1], ), dtype=np.float64)
    for ii in range(psi_R.shape[1]):
        psi_jj[:] = np.ascontiguousarray(psi_R[:, ii])
        R_out[ii] = np.dot(psi_jj.conj(), R * psi_jj).real * dR
    return R_out
    

# def calculate_mean_R(psi_R: ArrayLike, R: ArrayLike, dR: float) -> ArrayLike:
#     """The mean position for each state.
# 
#     Args:
#         psi (ArrayLike): the real space wavefunction with shape: (ngrid, nstates)
#         R (ArrayLike): the real space grid
# 
#     Returns:
#         ArrayLike: the mean position for each state [shape: (nstates,)]
#     """
#     prob: ArrayLike = get_nuclear_density(psi_R, dR)
#     normalization: ArrayLike = np.sum(prob, axis=0)
#     expval_R: ArrayLike = np.tensordot(prob, R, axes=(0, 0))
#     THRESHOLD: float = 1e-10
#     mask: ArrayLike = normalization < THRESHOLD
#     expval_R[mask] = np.nan
#     expval_R[~mask] /= normalization[~mask]
#     return expval_R

@njit
def calculate_mean_k(psi_k: ArrayLike, k: ArrayLike, dR: float) -> float:
    psi_ii = np.zeros((psi_k.shape[1], ), dtype=np.complex128)
    mean_k: float = 0.0
    for ii in range(psi_k.shape[0]):
        psi_ii[:] = psi_k[ii, :]
        mean_k += np.real(np.dot(psi_ii.conj(), psi_ii)) * k[ii] * dR
    return mean_k


# def calculate_mean_k(psi_k: ArrayLike, k: ArrayLike, dR: float) -> ArrayLike:
#     """The mean momentum for each state.
# 
#     Args:
#         psi (ArrayLike): the k space wavefunction with shape: (ngrid, nstates)
#         k (ArrayLike): the k space grid
#         dR (float): the R space grid spacing
# 
#     Returns:
#         ArrayLike: the mean momentum for each state [shape: (nstates,)]
#     """
#     prob: ArrayLike = get_nuclear_density(psi_k, dR)
#     normalization: ArrayLike = np.sum(prob, axis=0)
#     expval_k: ArrayLike = np.tensordot(prob, k, axes=(0, 0))
#     THRESHOLD: float = 1e-10
#     mask: ArrayLike = normalization < THRESHOLD
#     expval_k[mask] = np.nan
#     expval_k[~mask] /= normalization[~mask]
#     return expval_k

def calculate_populations(psi: ArrayLike, dR: float) -> ArrayLike:
    """The population for each state.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        dR (float): the grid spacing

    Returns:
        ArrayLike: the population for each state [shape: (nstates,)]
    """
    return np.sum(get_nuclear_density(psi, dR), axis=0)

def calculate_other_populations(psi: ArrayLike, U: ArrayLike, dR: float) -> ArrayLike:
    """The population for each state.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        dR (float): the grid spacing

    Returns:
        ArrayLike: the population for each state [shape: (nstates,)]
    """
    psi_other_rep: ArrayLike = np.zeros_like(psi)
    _calculate_other_populations(psi, psi_other_rep, U)
    return np.sum(get_nuclear_density(psi_other_rep, dR), axis=0)

# @jit(nopython=True)
def _calculate_other_populations(psi: ArrayLike, psi_other_rep: ArrayLike, U: ArrayLike, ) -> ArrayLike:
    nstates, _, ngrid = U.shape
    psi_tmp = np.zeros((nstates, ), dtype=np.complex128)
    U_ii_dagger = np.zeros((nstates, nstates), dtype=np.complex128)
    for ii in range(ngrid):
        U_ii_dagger[:] = U[:, :, ii].conj().T
        psi_tmp[:] = np.dot(U_ii_dagger, psi[ii, :])
        psi_other_rep[ii, :] = psi_tmp[:]
    return psi_other_rep

@njit
def calculate_KE(
    psi_k: ArrayLike,
    KE: ArrayLike,
    dR: float,
) -> float:
    psi_ii = np.zeros((psi_k.shape[1], ), dtype=np.complex128)
    KE_out: float = 0.0
    for ii in range(psi_k.shape[0]):
        psi_ii[:] = psi_k[ii, :]
        KE_out += np.real(np.dot(psi_ii.conj(), psi_ii)) * KE[ii] * dR
    return KE_out
    

# def calculate_KE(psi_k: ArrayLike, k: ArrayLike, dR: float, mass: float) -> ArrayLike:
#     """The kinetic energy for each state.
# 
#     Args:
#         psi (ArrayLike): the k space wavefunction with shape: (ngrid, nstates)
#         k (ArrayLike): the k space grid
#         dR (float): the R space grid spacing
#         mass (float): the mass of the particle
# 
#     Returns:
#         ArrayLike: the kinetic energy for each state [shape: (nstates,)]
#     """
#     # return state_specific_expected_values(psi_k, k**2 / (2 * mass), dR)
#     prob: ArrayLike = get_nuclear_density(psi_k, dR)    
#     normalization: ArrayLike = np.sum(prob, axis=0)
#     expval_KE: ArrayLike = np.tensordot(prob, k**2 / (2 * mass), axes=(0, 0))
#     THRESHOLD: float = 1e-10
#     mask: ArrayLike = normalization < THRESHOLD
#     expval_KE[mask] = np.nan
#     expval_KE[~mask] /= normalization[~mask]
#     return expval_KE

@njit
def calculate_PE(
    psi_R: ArrayLike,
    H: ArrayLike,
    dR: float,
) -> float:
    psi_ii = np.zeros((psi_R.shape[1], ), dtype=np.complex128)
    H_ii = np.zeros((H.shape[0], H.shape[1]), dtype=np.complex128)
    PE: float = 0.0
    for ii in range(psi_R.shape[0]):
        H_ii[:] = np.ascontiguousarray(H[:, :, ii])
        psi_ii[:] = psi_R[ii, :]
        PE += np.dot(psi_ii.conj(), np.dot(H_ii, psi_ii)).real * dR
    return PE
    

# def calculate_PE(
#     psi_R: ArrayLike, 
#     E: ArrayLike,
#     U: ArrayLike,
#     dR: float, 
# ) -> ArrayLike:
#     """The potential energy for each state.
# 
#     Args:
#         psi (ArrayLike): the real space wavefunction with shape: (ngrid, nstates)
#         V (ArrayLike): the potential energy with shape: (ngrid, nstates)
#         dR (float): the real space grid spacing
# 
#     Returns:
#         ArrayLike: the potential energy for each state [shape: (nstates,)]
#     """
#     nuc_prob: ArrayLike = get_nuclear_density(psi_R, dR)
#     normalization: ArrayLike = np.sum(nuc_prob, axis=0)
#     mask: ArrayLike = normalization < 1e-10
#     
#     ngrid, nstates = psi_R.shape
#     
#     psi_ij: ArrayLike = np.zeros((nstates, ), dtype=np.complex128) 
#     expval_E: ArrayLike = np.zeros_like(E) 
#     for ii in range(ngrid):
#         evals, evecs = E[:, ii], U[:, :, ii]
#         for jj in range(nstates):
#             psi_ij[:] = 0
#             psi_ij[jj] = psi_R[ii, jj] 
#             expval_E[jj, ii] = np.dot(evals, np.abs(np.dot(evecs.conj().T, psi_ij))**2).real * dR
#     
#     expval_E = np.sum(expval_E, axis=1) 
#     expval_E[mask] = np.nan
#     expval_E[~mask] /= normalization[~mask] 
#     return expval_E

def psi_diabatic_to_adiabatic(
    psi_diabatic: ArrayLike,
    U: ArrayLike, 
    psi_adiabatic: Optional[ArrayLike] = None,
) -> ArrayLike:
    psi_adiabatic = _psi_psi_shape_checker(psi_diabatic, psi_adiabatic)
    np.einsum("jki,ij->ik", U, psi_diabatic, out=psi_adiabatic)
    return psi_adiabatic

def psi_adiabatic_to_diabatic(
    psi_adiabatic: ArrayLike,
    U: ArrayLike, 
    psi_diabatic: Optional[ArrayLike] = None,
) -> ArrayLike:
    psi_diabatic = _psi_psi_shape_checker(psi_adiabatic, psi_diabatic)
    np.einsum("jki,ik->ij", U, psi_adiabatic, out=psi_diabatic)
    return psi_diabatic
        
def _psi_psi_shape_checker(psi: ArrayLike, psi_other_rep: Optional[ArrayLike]=None) -> ArrayLike:
    if psi_other_rep is None:
        psi_other_rep = np.zeros_like(psi)
    else:
        assert psi.shape == psi_other_rep.shape, f"The shapes of psi and psi_other_rep do not match: {psi.shape} != {psi_other_rep.shape}"
    return psi_other_rep
        
    
# %%
import numpy as np
def _test_inplace_einsum_vs_plain_einsum(ndim: int):
    import time
    NGRID = 10
    U = np.random.rand(ndim, ndim, NGRID)
    PSI = np.random.rand(ndim, NGRID)
    
    # test the performance of the plain einsum
    start = time.perf_counter_ns()
    dummy = np.einsum("jki,ji->ki", U.conj(), PSI)
    end = time.perf_counter_ns()
    print(f"plain einsum: {end - start} ns")
    
    #test the performance of the inplace einsum
    start = time.perf_counter_ns()
    PSI_out = np.zeros_like(PSI)
    np.einsum("jki,ji->ki", U.conj(), PSI, out=PSI_out)
    end = time.perf_counter_ns()
    print(f"inplace einsum: {end - start} ns")
    
# %%
if __name__ == "__main__":
    _test_inplace_einsum_vs_plain_einsum(2)

# %%
