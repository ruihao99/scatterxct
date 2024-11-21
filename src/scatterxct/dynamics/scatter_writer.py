import numpy as np
from numpy.typing import ArrayLike
from numba import njit
import yaml

def get_1d_scatter_result(
    R: ArrayLike,
    nuclear_density: ArrayLike,
    target_fn: str,
    separate_R: float = 0.0
):
    # allocate the scattering results dictionary
    _1d_scatter_out = {
        'gs_reflect': 0.0,    # counter for ground state reflection
        'gs_transmit': 0.0,   # counter for ground state transmission
        'es_reflect': 0.0,    # counter for excited state reflection
        'es_transmit': 0.0,   # counter for excited state transmission
    }
    
    # mask the R
    mask_left = R <= separate_R
    mask_right = R > separate_R
    
    # data type conversion to avoid YAML serialization error
    _1d_scatter_out['gs_reflect'] += np.sum(nuclear_density[mask_left, 0])
    _1d_scatter_out['es_reflect'] += np.sum(nuclear_density[mask_left, 1])
    _1d_scatter_out['gs_transmit'] += np.sum(nuclear_density[mask_right, 0])
    _1d_scatter_out['es_transmit'] += np.sum(nuclear_density[mask_right, 1])

    # assert probability is conserved
    ptotal = 0
    for key in _1d_scatter_out:
        _1d_scatter_out[key] = float(_1d_scatter_out[key]) # to avoid YAML serialization error
        ptotal += _1d_scatter_out[key]
    assert np.isclose(ptotal, 1.0), f"Total probability is not conserved: {ptotal}"

    # write the scattering results to the target file
    with open(target_fn, 'w') as f:
        yaml.dump(_1d_scatter_out, f)