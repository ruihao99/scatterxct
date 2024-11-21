# %%
import yaml
import numpy as np
import netCDF4 as nc

from scatterxct.io import parse_input
from scatterxct.dynamics.dynamics import ScatterXctDynamics
from scatterxct.core.wavefunction.scatter_movie import ScatterMovie

import os
import glob

TIME_TO_FRAME = 100 / 1

def find_input_file(dir_name: str):
    inpfile = os.path.join(dir_name, "input.yaml")

    if not os.path.exists(inpfile):
        # did not find the input file
        # try search for any yaml file in the directory
        yamls = glob.glob(os.path.join(dir_name, "*.yaml"))
        if len(yamls) == 1:
            inpfile = yamls[0]
        else:
            raise ValueError(
                f"Don't have a input.yaml file in {dir_name}."
                f"Found multiple yaml files: {yamls} but it is ambiguous which one is the input file."
            )
    return inpfile


def process_directory(dir_name: str):
    # message
    print(f"Processing {dir_name}")

    # parse the input file
    inpfile = find_input_file(dir_name)

    # load the input file, find the initial momentum
    yaml_data = yaml.safe_load(open(inpfile, 'r'))
    p0 = yaml_data["dynamics"]["initial"]["momentum"]
    data_dir = yaml_data["dynamics"]["output"]["data_dir"]

    # plot the scatter movie
    # parse the input file
    model, pulse, dynamics_cfg = parse_input(inpfile)

    # make the dynamics object
    dynamics = ScatterXctDynamics(
        hamiltonian=model,
        R0=dynamics_cfg['initial']['position'],
        k0=dynamics_cfg['initial']['momentum'],
        pulse=pulse,
        initial_state=dynamics_cfg['initial']['init_state'],
        dt=dynamics_cfg['dt'],
        mass=model.mass,
        scale=2.0
    )

    discretization = dynamics.discretization
    propagator = dynamics.propagator

    E0 = propagator.E0
    U0 = propagator.U0

    movie_writer = ScatterMovie(
        R=discretization.R,
        state_representation=1, # adiabatic representation
        E0=E0,
        U0=U0,
    )
    movie_path = os.path.join(dir_name, data_dir, f"scatter_movie.mp4")
    traj_path = os.path.join(dir_name, data_dir, f"traj.nc")
    
    dR = discretization.dR

    with nc.Dataset(traj_path, 'r') as f:
        t = f.variables['time'][:]
        re = f.variables['psi_diabatic_re'][:]
        im = f.variables['psi_diabatic_im'][:]
        psi_diabatic = np.array(re + 1j * im)

        nframes = t.size

        movie_frame = 0
        for i in range(nframes):
            time = t[i]
            if time >= movie_frame * TIME_TO_FRAME:
                movie_writer.append_frame(
                    time=time,
                    diabatic_wavefunction=psi_diabatic[i],
                )
                movie_frame += 1
            
    # make the movie
    movie_writer.make_movie()

    # save the movie
    movie_writer.save_animation(movie_path)

    # message
    print(f"Saved the movie to {movie_path}")


def main():
    scatter_dirs = glob.glob("p_init.*")

    for dir_name in scatter_dirs:
        try:
            process_directory(dir_name)
        except FileNotFoundError:
            #raise FileNotFoundError(f"Could not find the trajectory file in {dir_name}")
            print(f"Could not find the trajectory file in {dir_name}. Please wait for the simulation to finish.")




if __name__ == "__main__":
    main()
