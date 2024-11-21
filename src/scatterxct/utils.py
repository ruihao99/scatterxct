# %%
import numpy as np
import importlib
from mpi4py import MPI
import os
import shutil

math = importlib.import_module('math')
globals().update({name: getattr(math, name) for name in dir(math) if not name.startswith('_')})


def evaluate_params(params: dict):
    # params is a dictionary containing floating point values
    # or strings that can be evaluated to floating point values
    return {
        key: eval(value) if isinstance(value, str) else value
        for key, value in params.items()
    }

def traverse_and_evaluate(d):
    for key, value in d.items():
        if isinstance(value, dict):
            if key == "params":
                d[key] = evaluate_params(value)
            else:
                traverse_and_evaluate(value)
    return d

def divide_job(jobsize):
    # get MPI variables
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # divide the job size
    njobs = jobsize // size
    remainder = jobsize % size
    if rank < remainder:
        njobs += 1
        start = rank * njobs
    else:
        start = rank * njobs + remainder
    end = start + njobs
    return start, end

def create_path(path, bk=True):
    """create 'path' directory. If 'path' already exists, then check 'bk':
       if 'bk' is True, backup original directory and create new directory naming 'path';
       if 'bk' is False, do nothing.

    Args:
        path ('str' or 'os.path'): The direcotry you are making.
        bk (bool, optional): If . Defaults to False.
    """
    path += '/'
    if os.path.isdir(path):
        if bk:
            dirname = os.path.dirname(path)
            counter = 0
            while True:
                bkdirname = dirname + ".bk{0:03d}".format(counter)
                if not os.path.isdir(bkdirname):
                    shutil.move(dirname, bkdirname)
                    break
                counter += 1
            os.makedirs(path)
            # print("Target path '{0}' exsists. Backup this path to '{1}'.".format(path, bkdirname))
        else:
            None
            # print("Target path '{0}' exsists. No backup for this path.".format(path))
    else:
        os.makedirs(path)

def output_file_name(data_dir, name_base, traj_idx, ext):
    return os.path.join(data_dir, f"{name_base}_{traj_idx:05d}.{ext}")

def estimate_dt(Omega: float, dt: float = 0.05) -> float:
    SAFTY_FACTOR: float = 10.0
    if (Omega is None) or (Omega == 0.0):
        return dt
    else:
        T: float = 2 * np.pi / Omega
        if dt > T / SAFTY_FACTOR:
            return T / SAFTY_FACTOR
        else:
            return dt
