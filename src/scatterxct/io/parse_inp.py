# %%
import yaml
from scatterxct.hamiltonian.tullyone import parse_tullyone
from scatterxct.hamiltonian import HamiltonianBase
from scatterxct.pulses import PulseBase

import importlib
from typing import Union, Tuple, Dict, Any

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

def parse_model(model_cfg: dict, pulse_cfg: Union[None, dict]) -> Tuple[HamiltonianBase, bool]:
    # parse the model hamiltonian
    if model_cfg["kind"] == "tully_one":
        is_scattering = True
        model, pulse = parse_tullyone(model_cfg, pulse_cfg, phase_tracking="mai2015")
        return model, pulse, is_scattering
    else:
        raise ValueError(
            f"Model {model_cfg['kind']} is not supported."
        )


def parse_input(input: str) -> Tuple[HamiltonianBase, Union[PulseBase, None], Dict[str, Any]]:
    # read the input file (yaml)
    with open(input, 'r') as f:
        inp = yaml.safe_load(f)
        
    # traverse the dictionary and evaluate the parameters
    cfg = traverse_and_evaluate(inp)
    
    # get the pulse config (optional)
    pulse_cfg = None if "pulse" not in cfg else cfg["pulse"]
    
    # parse the model hamiltonian
    try:
        model_cfg = cfg["model"]
    except KeyError:
        raise ValueError(
            "Model hamiltonian is not specified in the input file."
        )
    model, pulse, is_scattering = parse_model(model_cfg, pulse_cfg)
    
    # get the dynamics config
    try: 
        dynamics_cfg = cfg["dynamics"]  
    except KeyError:
        raise ValueError(
            "Dynamics configuration is not specified in the input file."
        ) 
        
    # add the scattering flag to the dynamics config
    dynamics_cfg["is_scattering"] = is_scattering
            
    return model, pulse, dynamics_cfg
