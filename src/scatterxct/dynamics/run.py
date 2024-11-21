# %%
import numpy as np
from numpy.typing import ArrayLike

from scatterxct.core.wavefunction import calculate_mean_R, expected_value
from scatterxct.core.wavefunction import ScatterMovie
from scatterxct.core.wavefunction import get_nuclear_density
from scatterxct.core.wavefunction.wavepacket import estimate_a_from_k0
from scatterxct.core.wavefunction.math_utils import calculate_state_dependent_R

from scatterxct.dynamics.trajectory_writer import TrajectoryWriter
from scatterxct.dynamics.properties_writer import PropertiesWriter
from scatterxct.dynamics.scatter_writer import get_1d_scatter_result    
from .dynamics import ScatterXctDynamics
from .options import BasisRepresentation
from .properties import evaluate_properties, append_properties, ScatterXctProperties
from .step import propagate, SplitOperatorType
from .utils import safe_boundary

from typing import Optional, Tuple
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
    state_dependent_R = calculate_state_dependent_R(psi, R, dR)
    return outside_boundary(expected_R, R_lims) or any(outside_boundary(expected_R, R_lims) for expected_R in state_dependent_R)
    # return outside_boundary(expected_R, R_lims) or outside_boundary(each_state_expected_R, R_lims) 

def run_dynamics(
    hamiltonian,
    pulse,  
    R0: float,
    k0: float,
    trajectory_path: Path=Path("trajectory.nc"),
    properties_path: Path=Path("properties.dat"),
    scatter_path: Path=Path("scatter.dat"),
    initial_state: int=0, # defaults to the ground state
    dt: float=0.05,
    mass: float=2000.0,
    split_operator_type: SplitOperatorType=SplitOperatorType.VTV,
    max_iter: int=int(1e6),
    save_every: int=50,
    scale: float=1.0,
    apply_absorbing_boundary: bool=False,
    verbose: int=10
) -> None:
    """Run the time-independent dynamics."""
    dynamics = ScatterXctDynamics(
        hamiltonian=hamiltonian,
        R0=R0,
        k0=k0,
        pulse=pulse,
        initial_state=initial_state,
        dt=dt,
        mass=mass,
        scale=scale
    )
    scatter_R_lims: Optional[Tuple[float, float]] = None
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
    
    if apply_absorbing_boundary:    
        abs_boundary = propagator.get_absorbing_boundary_term()
        
    traj_writer = TrajectoryWriter(
        ngrid=discretization.ngrid,
        dim=wavefunction_data.nstates,
        R_grid=discretization.R,
        K_grid=discretization.k
    )
    
    prop_writer = PropertiesWriter(
        dim=wavefunction_data.nstates
    ) 
    
    for istep in range(max_iter):
        if istep % save_every == 0:
            if break_condition(dynamics.wavefunction_data.psi, dynamics.discretization.R, scatter_R_lims):
                traj_writer.save(trajectory_path)
                prop_writer.save(properties_path)
                break
            tlist = np.append(tlist, time)
            properties = evaluate_properties(discretization, propagator, wavefunction_data)
            output = append_properties(properties, output)
            traj_writer.write_frame(time, wavefunction_data.psi)
            prop_writer.write_frame(t=time, Ravg=properties.R, Pavg=properties.k, KE=properties.KE, PE=properties.PE, adiabatic_populations=properties.adiab_populations, diabatic_populations=properties.diab_populations)
            print(f"Step: {istep}, Time: {time}, R: {properties.R}, KE: {properties.KE}, PE: {properties.PE}")
                
        time, wavefunction_data = propagate(
            time, wavefunction_data, propagator, split_operator_type
        )
        
        if apply_absorbing_boundary:
            wavefunction_data.psi[:] *= abs_boundary
            
    # parse_the scatter results
    nuclear_density_diab: ArrayLike = get_nuclear_density(wavefunction_data.psi, discretization.dR)
    nuclear_density_adiab: ArrayLike = get_nuclear_density(wavefunction_data.get_psi_of_the_other_representation(U=propagator.U0), discretization.dR)
    # scatter_out_diab = parse_scatter(discretization.R, nuclear_density_diab, is_diabatic_representation=True)
    # scatter_out_adiab = parse_scatter(discretization.R, nuclear_density_adiab, is_diabatic_representation=False)
    
    # write the scatter results
    get_1d_scatter_result(
        R=discretization.R,
        nuclear_density=nuclear_density_adiab,
        target_fn=scatter_path,
    )
    

# %%
