# %%
import numpy as np
from numpy.typing import ArrayLike

from scatterxct.core.wavefunction import calculate_mean_R, expected_value
from scatterxct.core.wavefunction import ScatterMovie
from scatterxct.core.wavefunction import get_nuclear_density
from scatterxct.core.wavefunction.wavepacket import estimate_a_from_k0

from .dynamics import ScatterXctDynamics
from .options import BasisRepresentation
from .properties import evaluate_properties, append_properties, parse_scatter
from .step import propagate, SplitOperatorType
from .utils import safe_boundary

from typing import Optional, Tuple, NamedTuple
from pathlib import Path

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
    expected_R = expected_value(psi, R, dR)
    each_state_expected_R = calculate_mean_R(psi, R, dR)
    # return outside_boundary(expected_R, R_lims) or any(outside_boundary(expected_R, R_lims) for expected_R in each_state_expected_R)
    return outside_boundary(expected_R, R_lims) or outside_boundary(each_state_expected_R, R_lims) 

def run_time_independent_dynamics(
    hamiltonian,
    R0: float,
    k0: float,
    initial_state: int=0, # defaults to the ground state
    dt: float=0.1,
    mass: float=2000.0,
    split_operator_type: SplitOperatorType=SplitOperatorType.VTV,
    basis_representation: BasisRepresentation=BasisRepresentation.Diabatic,
    max_iter: int=int(1e6),
    save_every: int=10,
    movie_every: int=1000,
    movie_path: Optional[Path]=None,
    scale: float=1.0,
    apply_absorbing_boundary: bool=False,
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
        scale=scale
    )
    scatter_R_lims: Optional[Tuple[float, float]] = None
    a = estimate_a_from_k0(k0)
    if k0 > 0:
        # scatter from the right to the left
        scatter_R_lims = (R0*scale, -R0*scale)
    elif k0 < 0:
        scatter_R_lims = (-R0*scale, R0*scale)
    else:
        raise ValueError("The initial momentum should not be zero.")

    # apply safe boundary conditions
    scatter_R_lims = safe_boundary(R_lims=scatter_R_lims)

    discretization = dynamics.discretization
    propagator = dynamics.propagator
    wavefunction_data = dynamics.wavefunction_data
    tlist = np.array([], dtype=float)
    output = None
    time = 0.0
    
    def append_suffix(fpath: Path, suffix: str) -> Path:
        empty_path_str = fpath.with_suffix("").name
        return fpath.with_name(empty_path_str + suffix + fpath.suffix)
    
    # movie is named as *.gif, please change the name to *-diab.gif or *-adiab.gif 
    diab_movie_path: Optional[Path] = None if movie_path is None else append_suffix(movie_path, "-diab") 
    adiab_movie_path: Optional[Path] = None if movie_path is None else append_suffix(movie_path, "-adiab")
    
    # diab_fname_movie: Optional[str] = None if fname_movie is None else fname_movie.replace(".gif", "-diab.gif")
    # adiab_fname_movie: Optional[str] = None if fname_movie is None else fname_movie.replace(".gif", "-adiab.gif")
    
    diab_scatter_movie = None if diab_movie_path is None else ScatterMovie(
        R=discretization.R,
        state_representation=0
    )
    adiab_scatter_movie = None if adiab_movie_path is None else ScatterMovie(
        R=discretization.R,
        state_representation=1
    )
    
    if apply_absorbing_boundary:    
        abs_boundary = propagator.get_amplitude_reduction()
    
    for istep in range(max_iter):
        if istep % save_every == 0:
            if break_condition(dynamics.wavefunction_data.psi, dynamics.discretization.R, scatter_R_lims):
                break
            # if time >= 25000:
            #     break
            tlist = np.append(tlist, time)
            properties: NamedTuple = evaluate_properties(discretization, propagator, wavefunction_data)
            output = append_properties(properties, output)
        if (istep % movie_every == 0) and (diab_scatter_movie is not None):
            # we particularly want to save the wavepacket movie in the adiabatic representation
            # if dynamics.basis_representation == BasisRepresentation.Diabatic:
            #     nuclear_density: ArrayLike = get_nuclear_density(wavefunction_data.get_psi_of_the_other_representation(U=propagator.U), discretization.dR)
            # else:
            #     nuclear_density: ArrayLike = get_nuclear_density(wavefunction_data.psi, discretization.dR)
            nuclear_density_diab: ArrayLike = get_nuclear_density(wavefunction_data.psi, discretization.dR)
            nuclear_density_adiab: ArrayLike = get_nuclear_density(wavefunction_data.get_psi_of_the_other_representation(U=propagator.U), discretization.dR)
            adiab_scatter_movie.append_frame(time, nuclear_density_adiab, propagator.H)
            diab_scatter_movie.append_frame(time, nuclear_density_diab, propagator.H)
        time, wavefunction_data = propagate(
            time, wavefunction_data, propagator, split_operator_type
        )
        if apply_absorbing_boundary:
            wavefunction_data.psi[:] *= abs_boundary
            
    # parse_the scatter results
    nuclear_density_diab: ArrayLike = get_nuclear_density(wavefunction_data.psi, discretization.dR)
    nuclear_density_adiab: ArrayLike = get_nuclear_density(wavefunction_data.get_psi_of_the_other_representation(U=propagator.U), discretization.dR)
    scatter_out_diab = parse_scatter(discretization.R, nuclear_density_diab, is_diabatic_representation=True)
    scatter_out_adiab = parse_scatter(discretization.R, nuclear_density_adiab, is_diabatic_representation=False)
    
    # finalize the the scatter movie
    if diab_scatter_movie is not None:
        for fname, movie in zip([diab_movie_path, adiab_movie_path], [diab_scatter_movie, adiab_scatter_movie]):
            movie.make_movie()
            movie.save_animation(fname)

    return {'time': tlist, **output, 'scatter_out_diab': scatter_out_diab, 'scatter_out_adiab': scatter_out_adiab}

# %%
