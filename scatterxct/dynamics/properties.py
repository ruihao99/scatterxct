import numpy as np
from numpy.typing import ArrayLike

from scatterxct.core.discretization import Discretization
from scatterxct.core.propagator import PropagatorBase, PropagatorD, PropagatorA
from scatterxct.core.wavefunction import (
    WaveFunctionData,
    calculate_mean_R,
    calculate_mean_k,
    calculate_populations,
    calculate_other_populations,
    calculate_KE,
    calculate_PE
)

from functools import namedtuple
from typing import NamedTuple, Optional, Dict

# output the following expected values:
# - R: (nstate, ), the wavepacket specific expected R values
# - k: (nstate, ), the wavepacket specific expected k values
# - populations: (nstate, ), the populations
# - KE: (nstate, ), the kinetic energy of each wavepacket
# - PE: (nstate, ), the potential energy of each wavepacket

ScatterXctProperties = namedtuple(
    "ScatterXctProperties",
    ["R", "k", "diab_populations", "adiab_populations", "KE", "PE"]
)

def evaluate_properties(
    discretization: Discretization,
    propagator: PropagatorBase,
    wavefunction_data: WaveFunctionData,
) -> NamedTuple:
    # retrieve the data
    R: ArrayLike = discretization.R
    dR: float = discretization.dR
    k: ArrayLike = discretization.k
    mass: float = discretization.mass
    # E: ArrayLike = propagator.E
    U: ArrayLike = propagator.U
    H: ArrayLike = propagator.H
    KE: ArrayLike = propagator.KE
    
    # calculate the expected values in real space
    psi_R: ArrayLike = wavefunction_data.psi
    expected_R = calculate_mean_R(psi_R, R, dR)
    expected_PE = calculate_PE(psi_R, H, dR)
    diab_populations = calculate_populations(psi_R, dR)
    adiab_populations = calculate_other_populations(psi_R, U, dR)
    
    # calculate the expected values in k space
    wavefunction_data.real_space_to_k_space()
    psi_k: ArrayLike = wavefunction_data.psi
    expected_k = calculate_mean_k(psi_k, k, dR)
    # expected_KE = calculate_KE(psi_k, k, dR, mass)
    
    expected_KE = calculate_KE(psi_k, KE, dR)
    
    # print(f"dK naive: {dK_naive}, dK yanze: {dK_yanze}")
    wavefunction_data.k_space_to_real_space()
    
    return ScatterXctProperties(
        expected_R, 
        expected_k, 
        diab_populations, 
        adiab_populations,
        expected_KE,
        expected_PE
    )

def append_properties(
    properties: NamedTuple, 
    output_dict: Optional[Dict[str, ArrayLike]]=None
) -> Dict[str, ArrayLike]:
    if output_dict is None:
        output_dict = {field: [] for field in properties._fields}
    
    for field in properties._fields:
        output_dict[field].append(getattr(properties, field))
    return output_dict

def parse_scatter(
    R: ArrayLike,
    nuclear_density: ArrayLike,
    is_diabatic_representation: bool,
    separate_R: float = 0.0
) -> ArrayLike:
    """Parse the final wavefunction to get the scattering results.
    The current implementation requires the last frame of the wavefunction.
    However, this particular approach has its issues, as when the packets of 
    different states are not well separated, the parsing can only give the 
    semi-quantitative results. For tully scattering model one, the momentum 
    range of (5, 10) can give schechy scattering results using this approach.

    Args:
        R (ArrayLike): the real space grid
        psi (ArrayLike): the wavefunction in real space

    Returns:
        ArrayLike: the scattering results returned as (4,) array
            (lower left, lower right, upper left, upper right)
    """
    mask_left = R <= separate_R
    mask_right = R >= separate_R 
    if is_diabatic_representation:
        return _parse_scatter_diabatic(nuclear_density, mask_left, mask_right)
    else:
        return _parse_scatter_adiabatic(nuclear_density, mask_left, mask_right) 
    
def _parse_scatter_diabatic(
    nuclear_density: ArrayLike,
    mask_left: ArrayLike,
    mask_right: ArrayLike,
) -> ArrayLike:
    out = np.zeros(4, dtype=np.float64)
    
    # lower left (RL)
    out[0] = np.sum(nuclear_density[mask_left, 0])
    
    # lower right (TL)
    out[1] = np.sum(nuclear_density[mask_right, 1])
    
    # upper left (RU)
    out[2] = np.sum(nuclear_density[mask_left, 1])
    
    # upper right (TU)
    out[3] = np.sum(nuclear_density[mask_right, 0])
    
    return out

def _parse_scatter_adiabatic(
    nuclear_density: ArrayLike,
    mask_left: ArrayLike,
    mask_right: ArrayLike,
) -> ArrayLike:
    out = np.zeros(4, dtype=np.float64)
    
    # lower left
    out[0] = np.sum(nuclear_density[mask_left, 0])
    
    # lower right
    out[1] = np.sum(nuclear_density[mask_right, 0])
    
    # upper left
    out[2] = np.sum(nuclear_density[mask_left, 1])
    
    # upper right
    out[3] = np.sum(nuclear_density[mask_right, 1])
    
    return out
    