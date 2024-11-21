# %%
import numpy as np

from scatterxct.dynamics.run import run_dynamics
from scatterxct.io import parse_input
from scatterxct.utils import create_path, estimate_dt

import argparse
import glob
import time
import os

    
def parse_args():
    parser = argparse.ArgumentParser(description="Floquet Molecular Dynamics Driver")
    # -- input, -i
    parser.add_argument("-i", "--input", type=str, help="Input file", default=None)
    args = parser.parse_args()

    input_file = args.input
    if input_file is None:
        # try to find any yaml file in the current directory
        yaml_files = glob.glob("*.yaml")

        # if there are more than one yaml file, raise an error
        if (len(yaml_files) == 0):
            raise ValueError(
                "There is no yaml file in the current directory. "
                "Please specify the input file using -i or --input option."
            )
        elif (len(yaml_files) > 1):
            raise ValueError(
                "There are more than one yaml file in the current directory. "
                "Please specify the input file using -i or --input option."
            )
        else:
            input_file = yaml_files[0]
    
    return input_file         

def prepare_output_dir(dynamics_cfg):
    name_base = dynamics_cfg["output"]["name"]
    data_dir = dynamics_cfg["output"]["data_dir"] 
    create_path(data_dir)   
    return name_base, data_dir
    

def main():
    # parse the arguments to get the input file
    input_file = parse_args()
    
    # parse the input file
    model, pulse, dynamics_cfg = parse_input(input_file)
    
    # prepare the output directory
    name_base, data_dir = prepare_output_dir(dynamics_cfg)
    
    # the output file names
    trajectory_path = os.path.join(data_dir, f"traj.nc") 
    properties_path = os.path.join(data_dir, f"properties.dat")
    scatter_path = os.path.join(data_dir, f"scatter.yaml")
    movie_path = os.path.join(data_dir, f"movie.gif") 
    
    # estimate the time step
    dt_inp = dynamics_cfg["dt"]
    Omega = None if pulse is None else pulse.omega
    dt = estimate_dt(Omega=Omega, dt=dt_inp)
    
    scale = 2.0
    
    # run the dynamics
    start_time = time.perf_counter()
    print(f"{pulse=}")
    
    output = run_dynamics(
        hamiltonian=model,
        pulse=pulse,    
        R0=dynamics_cfg['initial']['position'],
        k0=dynamics_cfg['initial']['momentum'],
        dt=dt,
        trajectory_path=trajectory_path,
        properties_path=properties_path,
        scatter_path=scatter_path,
        initial_state=0,
        scale=scale,
        apply_absorbing_boundary=False,
        save_every=dynamics_cfg['output']['nstep_write']
    ) 
    
    end_time = time.perf_counter()  
    
    time_elapsed = end_time - start_time    
    
    # post-processing the dynamics
    
    # print the time elapsed
    print(f"Time elapsed: {time_elapsed:.2f} s")
    
    



# %%
if __name__ == "__main__":
    main()