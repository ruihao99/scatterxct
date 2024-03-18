import numpy as np
from numpy.typing import ArrayLike

from scatterxct.core.discretization import Discretization
from scatterxct.core.propagator import PropagatorBase
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
    E: ArrayLike = propagator.E
    U: ArrayLike = propagator.U
    
    # calculate the expected values in real space
    psi_R: ArrayLike = wavefunction_data.psi
    expected_R = calculate_mean_R(psi_R, R, dR)
    expected_PE = calculate_PE(psi_R, E, U, dR)
    diab_populations = calculate_populations(psi_R, dR)
    adiab_populations = calculate_other_populations(psi_R, U, dR)
    
    
    # calculate the expected values in k space
    wavefunction_data.real_space_to_k_space()
    psi_k: ArrayLike = wavefunction_data.psi
    expected_k = calculate_mean_k(psi_k, k, dR)
    expected_KE = calculate_KE(psi_k, k, dR, mass)
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
        