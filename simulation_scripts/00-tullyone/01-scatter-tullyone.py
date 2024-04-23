import numpy as np
from joblib import Parallel, delayed

from scatterxct.dynamics.run import run_dynamics
from scatterxct.models.tullyone import get_tullyone, TullyOnePulseTypes
from pymddrive.utils import get_ncpus

from utils import get_tullyone_p0_list, get_tully_one_delay_time, estimate_dt

from typing import Optional
from pathlib import Path
import argparse

def check_if_has_ffmpeg():
    import shutil
    return shutil.which("ffmpeg") is not None

def run_single_tullyone(
    R0: float,
    k0: float,
    pulse_type: TullyOnePulseTypes,
    Omega: Optional[float] = None,
    tau: Optional[float] = None,
    phi: Optional[float] = None,
    project_dir: Path = Path("./"),
    save_movie: bool = False
) -> None:
    # estimate the delay time using MQC dynamics
    if pulse_type != TullyOnePulseTypes.NO_PULSE:
        delay_time = get_tully_one_delay_time(R0, k0)
        
    # prepare the output directory
    momentum_signature = f"k0-{k0:.6f}"
    output_dir = project_dir / momentum_signature
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path: Path = output_dir / f"trajectory.nc"
    properties_path: Path = output_dir / f"properties.dat"
    scatter_path: Path = output_dir / f"scatter.dat"
        
    # get the hamiltonian
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        hamiltonian = get_tullyone( 
            pulse_type=TullyOnePulseTypes.NO_PULSE
        )
    else:
        hamiltonian = get_tullyone(
            t0=delay_time, Omega=Omega, tau=tau, phi=phi,
            pulse_type=pulse_type
        )
    
    # estimate the time step based on the driving frequency
    dt = 0.05 if Omega is None else estimate_dt(Omega, dt=0.05)
    
    # prepare the file name for the movie
    # if pulse_type == TullyOnePulseTypes.NO_PULSE:
    #     # fname_movie: str = f"scatter_movie-k0_{k0}.gif"
    #     fname_movie: str = f"scatter_movie-k0_{k0}.mp4"
    # else:
    #     # fname_movie: str = f"scatter_movie-k0_{k0}-Omega_{Omega}-tau_{tau}.gif"
    #     fname_movie: str = f"scatter_movie-k0_{k0}.mp4"
    # if check_if_has_ffmpeg():
    #     fname_movie: str = f"scatter_movie-k0_{k0:0.4f}.mp4"
    # else:
    #     fname_movie: str = f"scatter_movie-k0_{k0:0.4f}.gif"
        
    
    # run the dynamics
    scale = 2.0
    run_dynamics(
        hamiltonian=hamiltonian,
        R0=R0,
        k0=k0,
        dt=dt,
        trajectory_path=trajectory_path,
        properties_path=properties_path,
        scatter_path=scatter_path,
        initial_state=0,
        save_every=50,
        scale=scale,
    )
    

def main(
    project_prefix: str,
    n_momentum_samples: int, 
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
    Omega: Optional[float] = None,  
    tau: Optional[float] = None,
    phi: Optional[float] = None
):
    
    # make the directory if it does not exist
    project_dir = Path(project_prefix)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # prepare the momentum list, and a fixed list of R0, Omega, and tau
    k0 = get_tullyone_p0_list(n_momentum_samples, pulse_type)
    R0 = np.array([-10.0] * n_momentum_samples)
    Omega = np.array([Omega] * n_momentum_samples)
    tau = np.array([tau] * n_momentum_samples)
    phi = np.array([phi] * n_momentum_samples)
    
    Parallel(n_jobs=get_ncpus(), verbose=5)(
        delayed(run_single_tullyone)(
            R0[i], k0[i], pulse_type, Omega[i], tau[i], phi[i], project_dir
        ) for i in range(n_momentum_samples)
    )
    
    # for i, out in enumerate(output):
    #     if pulse_type == TullyOnePulseTypes.NO_PULSE:
    #         trajdir = project_dir / f"traj_k0_{k0[i]}.npz"
    #     else:   
    #         trajdir = project_dir / f"traj_k0_{k0[i]}-Omega_{Omega[i]}-tau_{tau[i]}.npz"
    #     np.savez(
    #         trajdir,
    #         time=out['time'],
    #         R=out['R'],
    #         k=out['k'],
    #         diab_populations=np.array(out['diab_populations']),
    #         adiab_populations=np.array(out['adiab_populations']),
    #         KE=out['KE'],
    #         PE=out['PE'],
    #         scatter_out_diab=np.array(out['scatter_out_diab']),
    #         scatter_out_adiab=np.array(out['scatter_out_adiab'])
    #     )
    
# %%
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--Omega", type=float)
    # argparser.add_argument("--tau", type=float)
    # argparser.add_argument("--pulse_type", type=int)
    argparser.add_argument("--Omega", type=float)
    argparser.add_argument("--tau", type=float)
    # argparser.add_argument("phi", type=float)
    argparser.add_argument("--pulse_type", type=int)
    
    # if the user does not provide the Omega and tau, then complain and throw an error
    parsed_args = argparser.parse_args()

    # if len(parsed_args) != 3:
    #     raise ValueError("Please provide the Omega and tau.")
    
    
    Omega = parsed_args.Omega
    tau = parsed_args.tau
    # phi = parsed_args.phi
    phi = 0.0
    _pulse_type = parsed_args.pulse_type
    if _pulse_type == 1:
        pulse_type = TullyOnePulseTypes.PULSE_TYPE1
    elif _pulse_type == 2:
        pulse_type = TullyOnePulseTypes.PULSE_TYPE2
    elif _pulse_type == 3:
        pulse_type = TullyOnePulseTypes.PULSE_TYPE3
    elif _pulse_type == 0:
        pulse_type = TullyOnePulseTypes.NO_PULSE
    else:
        raise ValueError(f"Unknown pulse type {_pulse_type=}")

    NSAMPLES = 16
    
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        project_prefix = f"data_scatterxct"    
    else:
        project_prefix = f"data_scatterxct-Omega_{Omega}-tau_{tau}-pulse_{_pulse_type}"
    
    main(
        project_prefix=project_prefix,
        n_momentum_samples=NSAMPLES,
        Omega=Omega,
        tau=tau,
        pulse_type=pulse_type, 
    )
