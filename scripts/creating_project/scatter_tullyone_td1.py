# %%
import yaml
import numpy as np

import os
import glob

def t_center_model(x, a=4.650588e-05, b=9.854187e-06):
    # WITH PRECOMPUTED NON-LASER TRAJECTORIES
    # THE ORIGNAL TULLY ONE MODEL PARAMETERS YIELD THE FOLLOWING FIT
    # FOR THE OPTIMAL LASER CENTER

    # THIS ADHOC MODEL ONLY WORKS WHEN:
    # - A = 0.01
    # - B = 1.6
    # - C = 0.005
    # - D = 1.0
    return 1/(a * x + b)

TEMPLATE = """
pulse:
  kind: morlet
  params:
    A: 0.005
    t0: 600.0
    tau: 300.0
    omega: 0.02
    phi: 0.0
    laserwidth: 0.003

model:
  kind: tully_one
  type: time_dependent_1 # dipole operator is sigma_x
  params:
    A: 0.01
    B: 1.6
    C: 0.005
    D: 1.0
    mass: 2000.0


dynamics:
  method: FSSH
  ntrajs: 1
  dt:     1                    # classical time step
  representation: adiabatic
  flag_frustrated: 1
  termination:
    tf: ~
    bounds: [-10.0, 10.0]
  initial:
    representation: adiabatic # the initial state is a diabatic state
    init_state: 0            # the initial state is the 0-th state
    position: -10.0          # the initial position
    momentum: 30.0            # the initial momentum
    distribution: delta      # the initial distribution is a delta function
                             # (options: delta, gaussian, boltzmann, wigner_t, wigner_nu)
                             # (gaussian requires sigma_x and sigma_p)
                             # (boltzmann and wigner_t require temperature)
                             # (wigner_nu requires the vibrational state)
  seed:
    seed1: 100869
    seed2: 231841
  output:
    nstep_write: 10
    name: fssh
    data_dir: ./data
"""


TEMPLATE_SLURM = """#!/bin/bash
#SBATCH --job-name="tully_one"
#SBATCH -p douwj
#SBATCH --qos huge
#SBATCH -c 1

# the temporary directory
# export TMPDIR="/data"

# in case the user don't run this script on the cluster
# if SLURM_NTASKS exists, we use it; otherwise,
# we use the max number of cores available
# make a variable call MPI_NP

# time stamp for the job
echo "Job started at" `date`

# loop over the initial momenta directories
for dir in PLACEHOLDER; do
    echo "===================================================="
    echo "Start running dynamics in ${dir}. Time", `date`
    start_time=`date +%s`
    cd ${dir}
        python -m scatterxct.main -i input.yaml
    cd ..
    end_time=`date +%s`
    echo "Dynamics in ${dir} is done. Time elapsed: $((end_time-start_time)) seconds."
    echo "===================================================="
    echo ""
done
"""

def linspace_log10(x_left: float, x_right: float, n: int) -> np.ndarray:
    return np.logspace(np.log10(x_left), np.log10(x_right), n)

def sample_sigmoid(x_left: float, x_right: float, n: int) -> np.ndarray:
    x_center = (x_left + x_right) / 2
    p0_seg1 = np.sort(x_center + x_left - linspace_log10(x_left, x_center, n // 2))
    dp = p0_seg1[-1] - p0_seg1[-2]
    p0_seg2 = x_center - x_left + dp + np.sort(linspace_log10(x_left, x_center, n - n // 2))
    return np.concatenate((p0_seg1, p0_seg2))

def get_tully_one_p0_list(nsamples: int, ttype: str) -> np.ndarray:
    if ttype == "time_independent":
        p0_bounds_0 = (2.0, 12.0); n_bounds_0 = nsamples // 2
        p0_bounds_1 = (13, 35); n_bounds_1 = nsamples - n_bounds_0
    elif ttype == "time_dependent_1":
        p0_bounds_0 = (0.5, 19); n_bounds_0 = nsamples // 3 * 2
        p0_bounds_1 = (20, 35); n_bounds_1 = nsamples - n_bounds_0
    else:
        raise ValueError(f"Unsupported TullyOne model type: {ttype}")

    p0_segment_0 = sample_sigmoid(*p0_bounds_0, n_bounds_0)
    p0_segment_1 = np.linspace(*p0_bounds_1, n_bounds_1)
    p0 = np.concatenate((p0_segment_0, p0_segment_1))
    return p0

def make_single_directory(p0: float, ntrajs, idx: int, t_center: float):
    dir_name = f"p_init.{idx:05d}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # load the template yaml file
    cfg = yaml.safe_load(TEMPLATE)

    # update the initial momentum and number of trajectories
    cfg["dynamics"]["ntrajs"] = ntrajs
    cfg["dynamics"]["initial"]["momentum"] = float(p0)

    # update the laser center
    cfg["pulse"]["params"]["t0"] = float(t_center)


    # write the updated yaml file
    yaml_fn = os.path.join(dir_name, "input.yaml")
    with open(yaml_fn, 'w') as f:
        yaml.dump(cfg, f)


def main(
    ntrajs: int = 64,
    n_momenta_sample: int = 10,
    njobs: int = 2
):
    # get the initial momenta samples
    p0_list = get_tully_one_p0_list(n_momenta_sample, "time_independent")

    # get the center of the laser
    t_center = t_center_model(p0_list)

    # make individual directories for each p0
    for idx, (p0, t_center) in enumerate(zip(p0_list, t_center)):
        make_single_directory(p0, ntrajs, idx, t_center)

    # generate the slurm scripts

    # 1. get all the directories
    all_dirs = glob.glob("p_init.*")

    # 2. randomly divide the directories into njobs
    np.random.shuffle(all_dirs)
    chunksize = len(all_dirs) // njobs
    chunks = [all_dirs[i:i+chunksize] for i in range(0, len(all_dirs), chunksize)]

    # 3. generate the slurm scripts
    for i, chunk in enumerate(chunks):
        slurm_fn = f"run_{i}.sh"
        with open(slurm_fn, 'w') as f:
            f.write(TEMPLATE_SLURM.replace("PLACEHOLDER", " ".join(chunk)))


if __name__ == "__main__":
    # testing parameters
    ntrajs = 8
    n_momenta_sample = 10
    njobs = 2


    # production parameters
    # ntrajs = 2048
    # n_momenta_sample = 40
    # njobs = 8

    main(ntrajs, n_momenta_sample, njobs)
