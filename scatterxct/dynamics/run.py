# %%
import numpy as np
from numpy.typing import ArrayLike

from scatterxct.core.wavefunction import state_specific_expected_values, expected_value
from scatterxct.core.wavefunction import ScatterMovie
from scatterxct.core.wavefunction import get_nuclear_density

from .dynamics import ScatterXctDynamics
from .options import BasisRepresentation
from .properties import evaluate_properties, append_properties
from .step import propagate, SplitOperatorType
from .utils import safe_boundary


from typing import Optional, Tuple

def outside_boundary(expected_R: float, R_lims: Tuple[float, float]) -> bool:
    """determine if the expected R value is outside the boundary.

    Args:
        expected_R (float): the expected R value of a particular state
        R_lims (Tuple[float, float]): the boundary of the scatter region

    Returns:
        bool: returns True of the all of the expected R values are outside the boundary
    """
    return expected_R < R_lims[0] or expected_R > R_lims[1]

def break_condition(psi: ArrayLike, R: ArrayLike, R_lims: Tuple[float, float]) -> bool:
    dR: float = R[1] - R[0]
    expected_R_values = state_specific_expected_values(psi, R, dR)
    expected_R = expected_value(psi, R, dR)
    vectorized_outside_boundary = np.vectorize(lambda expected_R: outside_boundary(expected_R, R_lims=R_lims))
    flag_each_packet_outside_boundary = np.any(vectorized_outside_boundary(expected_R_values))
    flag_average_packet_outside_boundary = outside_boundary(expected_R, R_lims=R_lims)
    return flag_each_packet_outside_boundary or flag_average_packet_outside_boundary

def run_time_independent_dynamics(
    hamiltonian, 
    R0: float, 
    k0: float, 
    initial_state: int=0, # defaults to the ground state
    dt: float=0.1,
    mass: float=2000.0,
    split_operator_type: SplitOperatorType=SplitOperatorType.TVT,
    basis_representation: BasisRepresentation=BasisRepresentation.Diabatic,
    max_iter: int=int(1e6),
    save_every: int=10,
    movie_every: int=1000,
    fname_movie: str="scatter_movie.gif",
):
    """Run the time-independent dynamics."""
    dynamics = ScatterXctDynamics(
        hamiltonian=hamiltonian,
        R0=R0,
        k0=k0,
        initial_state=initial_state,
        dt=dt,
        mass=mass,
        basis_representation=basis_representation,
    )
    scatter_R_lims: Optional[Tuple[float, float]] = None
    if k0 > 0: 
        # scatter from the right to the left
        scatter_R_lims = (R0, -R0)
    elif k0 < 0:
        scatter_R_lims = (-R0, R0)
    else:
        raise ValueError("The initial momentum should not be zero.")
    
    scatter_movie = ScatterMovie(R=dynamics.discretization.R, H=dynamics.discretization.H)
    
    # apply safe boundary conditions
    scatter_R_lims = safe_boundary(R_lims=scatter_R_lims)
    
    discretization = dynamics.discretization
    print(f"{discretization.dR=}, {discretization.ngrid=}, {discretization.nstates=} ")
    wavefunction_data = dynamics.wavefunction_data
    propagator = dynamics.propagator
    tlist = np.array([], dtype=float)
    output = None
    time = 0.0
    
    from scatterxct.core.wavefunction import view_wavepacket
    
    for istep in range(max_iter):
        if istep % save_every == 0:
            if break_condition(dynamics.wavefunction_data.psi, dynamics.discretization.R, scatter_R_lims):
                break
            tlist = np.append(tlist, time)
            properties = evaluate_properties(discretization, wavefunction_data)
            populations = properties.populations
            R = properties.R
            print(f"{time=}, {R=}, {populations=}")
            output = append_properties(properties, output) 
            # if istep % (save_every * 100) == 0:
            #     view_wavepacket(discretization.R, wavefunction_data.psi)
        if istep % movie_every == 0:
            nuclear_density: ArrayLike = get_nuclear_density(wavefunction_data.psi, discretization.dR)
            scatter_movie.append_frame(nuclear_density, time)
        time, wavefunction_data = propagate(
            time, wavefunction_data, propagator, split_operator_type
        )
        # finalize the the scatter movie
    scatter_movie.make_movie()
    scatter_movie.save_animation(fname_movie)
        
    return {'time': tlist, **output} 

# %%