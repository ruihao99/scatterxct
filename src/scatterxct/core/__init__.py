import warnings 
from . import global_control

def modify_globals(**kwargs):
    """ Modify the global parameters of the core package """
    for key, value in kwargs.items():
        if key in global_control.__all__:
            setattr(global_control, key, value)
        else:
            warnings.warn(f"Attempted to modify non-existent global parameter {key}")