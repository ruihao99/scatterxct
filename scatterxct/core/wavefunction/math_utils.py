# %%
import numpy as np
from numba import jit
from numpy.typing import ArrayLike

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

def state_specific_expected_values(psi: ArrayLike, operator: ArrayLike, dR: float) -> ArrayLike:
    """The expected value of the operator for each state.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        operator (ArrayLike): an operator with shape: (nstates, nstates, ngrid)
        dR (float): the grid spacing

    Returns:
        ArrayLike: the expected value of the operator for each state [shape: (nstates,)]
    """
    nuc_prob: ArrayLike = get_nuclear_density(psi, dR)
    out: ArrayLike = np.zeros((psi.shape[0], psi.shape[1]), dtype=np.float64)
    if operator.ndim == 3:
        out = _state_specific_expected_values(psi, operator, out)
    elif operator.ndim == 1:
        out = _state_specific_expected_values_1d(nuc_prob, operator, out)
    return np.sum(out * nuc_prob, axis=0)

@jit(nopython=True)
def _state_specific_expected_values(psi: ArrayLike, operator: ArrayLike, out: ArrayLike) -> ArrayLike:
    ngrid, nstates = psi.shape
    O_i = np.zeros((nstates, nstates), dtype=np.complex128)
    psi_i = np.zeros((nstates, ), dtype=np.complex128)
    for nr in range(ngrid):
        for istate in range(nstates):
            O_i[:] = operator[:, :, nr]
            psi_i[:] = psi[nr, istate]
            out[nr, istate] = np.dot(psi_i.conj(), np.dot(O_i, psi_i)).real
    return out

def _state_specific_expected_values_1d(nuc_prob: ArrayLike, operator_1d: ArrayLike, out: ArrayLike) -> ArrayLike:
    out[:] = np.tensordot(nuc_prob, operator_1d, axes=(0, 0))
    return out
    
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
        

@jit(nopython=True)
def _trace_electronic(psi: ArrayLike, operator: ArrayLike, out: ArrayLike) -> ArrayLike:
    ngrid = psi.shape[0]
    for ii in range(ngrid):
        rho = np.outer(psi[ii:, ], psi[ii:, ].conj())
        out[ii] = np.trace(np.dot(rho, operator[:, :, ii])).real
    return out

def calculate_mean_R(psi_R: ArrayLike, R: ArrayLike, dR: float) -> ArrayLike:
    """The mean position for each state.

    Args:
        psi (ArrayLike): the real space wavefunction with shape: (ngrid, nstates)
        R (ArrayLike): the real space grid

    Returns:
        ArrayLike: the mean position for each state [shape: (nstates,)]
    """
    prob: ArrayLike = get_nuclear_density(psi_R, dR)
    return np.tensordot(prob, R, axes=(0, 0))

def calculate_mean_k(psi_k: ArrayLike, k: ArrayLike, dk: float) -> ArrayLike:
    """The mean momentum for each state.

    Args:
        psi (ArrayLike): the k space wavefunction with shape: (ngrid, nstates)
        k (ArrayLike): the k space grid

    Returns:
        ArrayLike: the mean momentum for each state [shape: (nstates,)]
    """
    prob: ArrayLike = get_nuclear_density(psi_k, dk)
    return np.tensordot(prob, k, axes=(0, 0))

def calculate_populations(psi: ArrayLike, dR: float) -> ArrayLike:
    """The population for each state.

    Args:
        psi (ArrayLike): the wavefunction with shape: (ngrid, nstates)
        dR (float): the grid spacing

    Returns:
        ArrayLike: the population for each state [shape: (nstates,)]
    """
    return np.sum(get_nuclear_density(psi, dR), axis=0)

def calculate_KE(psi_k: ArrayLike, k: ArrayLike, dk: float, mass: float) -> ArrayLike:
    """The kinetic energy for each state.

    Args:
        psi (ArrayLike): the k space wavefunction with shape: (ngrid, nstates)
        k (ArrayLike): the k space grid
        dk (float): the k space grid spacing
        mass (float): the mass of the particle

    Returns:
        ArrayLike: the kinetic energy for each state [shape: (nstates,)]
    """
    return np.sum(np.abs(psi_k)**2 * k[:, np.newaxis]**2 / (2 * mass) * dk, axis=0)

def calculate_PE(psi_R: ArrayLike, V: ArrayLike, dR: float) -> ArrayLike:
    """The potential energy for each state.

    Args:
        psi (ArrayLike): the real space wavefunction with shape: (ngrid, nstates)
        V (ArrayLike): the potential energy with shape: (ngrid, nstates)
        dR (float): the real space grid spacing

    Returns:
        ArrayLike: the potential energy for each state [shape: (nstates,)]
    """
    return state_specific_expected_values(psi_R, V, dR)
        
