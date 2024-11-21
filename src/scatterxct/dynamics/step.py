# %%
import numpy as np
from numpy.typing import ArrayLike
from numba import njit
import pyfftw
from pyfftw import FFTW

from scatterxct.core.wavefunction import WaveFunctionData
from scatterxct.core.propagator import Propagator
from scatterxct.core.discretization.finite_difference import fddrv

from enum import Enum, unique
from typing import Tuple

@unique
class SplitOperatorType(Enum):
    PLAIN = 1
    TVT = 2
    VTV = 3
    
@njit
def vectorized_multiply(
    V, psi,
):
    ngrid, nstates = psi.shape
    output = np.zeros((ngrid, nstates), dtype=np.complex128)
    for i in range(ngrid):
        for j in range(nstates):
            for k in range(nstates):
                output[i, j] += V[j, k, i] * psi[i, k]
    return output

def kinetic_propagate(
    psi_data: WaveFunctionData, # (ngrid, nstates)
    T_propagator: ArrayLike, # (ngrid, ) since T_propagator is diagonal 
) -> WaveFunctionData:
    # technical note: 
    # WavefunctionData is an immutable dataclass (with frozen=True)
    # hence psi_data.psi cannot be reassigned to another reference.
    # The only viable (and legit) way to update the psi_data.psi is through broadcasting,
    # since at the end of the day, the psi_data.psi itself is a numpy array, which is mutable.
    # This particular choice of design is to ensure the safety of the data,
    # and to avoid reallocation of memory.
    psi_data.real_space_to_k_space()
    psi_data.psi[:] *= T_propagator[:, np.newaxis]
    psi_data.k_space_to_real_space()
    return psi_data
    
def potential_propagate(
    psi_data: WaveFunctionData, # (ngrid, nstates)
    V_propagator: ArrayLike, # (nstates, nstates, ngrid, )
) -> WaveFunctionData:
    # technical note: WavefunctionData is an immutable dataclass
    # (more details explained in the kinetic_propagate function)
    output = vectorized_multiply(V_propagator, psi_data.psi)    
    psi_data.psi[:] = output
    return psi_data    
    
def plain_propagate(
    psi_data: WaveFunctionData, # (ngrid, nstates)
    T_prop: ArrayLike, # (ngrid, ) 
    V_prop: ArrayLike, # (nstates, nstates, ngrid)
) -> WaveFunctionData:
    psi_data = kinetic_propagate(psi_data, T_prop)
    psi_data = potential_propagate(psi_data, V_prop)
    return psi_data

def TVT_propagate(
    psi_data: WaveFunctionData, # (ngrid, nstates)
    half_T_prop: ArrayLike, # (ngrid, )  
    V_prop: ArrayLike, # (ngrid, nstates, nstates)
) -> WaveFunctionData:
    psi_data = kinetic_propagate(psi_data, half_T_prop)
    psi_data = potential_propagate(psi_data, V_prop)
    psi_data = kinetic_propagate(psi_data, half_T_prop)
    return psi_data

def VTV_propagate(
    psi_data: WaveFunctionData, # (ngrid, nstates)
    T_propagator: ArrayLike, # (ngrid, ) 
    half_V_prop1: ArrayLike, # (nstates, nstates, ngrid)
    half_V_prop2: ArrayLike, # (nstates, nstates, ngrid)
) -> WaveFunctionData:
    psi_data = potential_propagate(psi_data, half_V_prop1)
    psi_data = kinetic_propagate(psi_data, T_propagator)
    psi_data = potential_propagate(psi_data, half_V_prop2)
    return psi_data

def propagate(
    time: float,
    psi_data: WaveFunctionData, # (ngrid, nstates)
    propagator: Propagator,
    split_operator_type: SplitOperatorType = SplitOperatorType.PLAIN,
) -> Tuple[float, WaveFunctionData]:
    if split_operator_type == SplitOperatorType.PLAIN:
        T_prop = propagator.get_T_propagator()
        V_prop = propagator.get_V_propagator(time)
        time += propagator.dt
        return time, plain_propagate(psi_data, T_prop, V_prop)
    elif split_operator_type == SplitOperatorType.TVT:
        half_T_prop = propagator.get_half_T_propagator()
        V_prop = propagator.get_V_propagator(time)
        time += propagator.dt
        return time, TVT_propagate(psi_data, half_T_prop, V_prop)
    elif split_operator_type == SplitOperatorType.VTV:
        T_prop = propagator.get_T_propagator()
        half_V_prop1, half_V_prop2 = propagator.get_half_V_propagator(time)
        time += propagator.dt
        return time, VTV_propagate(psi_data, T_prop, half_V_prop1, half_V_prop2)
    else:
        raise ValueError(f"Unknown split operator type: {split_operator_type}")
