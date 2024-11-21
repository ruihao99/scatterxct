# %%
from scatterxct.hamiltonian import HamiltonianBase    
from scatterxct.pulses import parse_pulse, PulseBase
from scatterxct.hamiltonian.phase_tracking import PhaseTracking
from scatterxct.hamiltonian.tullyone.tullyone import TullyOne
from scatterxct.hamiltonian.tullyone.tullyone_td1 import TullyOneTD1
from scatterxct.hamiltonian.tullyone.tullyone_td2 import TullyOneTD2
from scatterxct.hamiltonian.tullyone.tullyone_td3 import TullyOneTD3

from typing import Union, Tuple

def parse_tullyone(
    config_model: dict,
    config_pulse: Union[dict, None],
    phase_tracking: PhaseTracking
) -> Tuple[HamiltonianBase, Union[PulseBase, None]]:
    # Assert the model is TullyOne
    assert config_model['kind'] == 'tully_one'   
    
    phase_tracking = PhaseTracking(phase_tracking)
    
    # determine if the model is time dependent
    if config_model['type'] == 'time_independent':
        params = config_model['params']
        return TullyOne.init(phase_tracking=phase_tracking, **params), None
    elif config_model['type'] == 'time_dependent_1':
        # we got type 1 time-dependent model
        assert config_pulse is not None
        pulse = parse_pulse(config_pulse)
        return TullyOneTD1.init(phase_tracking=phase_tracking, **config_model['params']), pulse
    elif config_model['type'] == 'time_dependent_2':
        # we got type 2 time-dependent model
        assert config_pulse is not None
        pulse = parse_pulse(config_pulse)
        return TullyOneTD2.init(phase_tracking=phase_tracking, **config_model['params']), pulse
    elif config_model['type'] == 'time_dependent_3':
        # we got type 3 time-dependent model
        assert config_pulse is not None
        pulse = parse_pulse(config_pulse)
        return TullyOneTD3.init(phase_tracking=phase_tracking, **config_model['params']), pulse
    else:
        raise ValueError(f"Unsupported TullyOne model type: {config_model['type']}")
        

def test_parse_tullyone():
    import yaml
    file_td0 = "../../../../example/tullyone/time_independent.yaml"
    file_td1 = "../../../../example/tullyone/time_dependent_1.yaml"
    file_td2 = "../../../../example/tullyone/time_dependent_2.yaml" 
    file_td3 = "../../../../example/tullyone/time_dependent_3.yaml"
    
    # load file_td0 and test the parsing
    with open(file_td0, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
        config_model = config['model']  
        config_pulse = None if 'pulse' not in config else config['pulse']
        hami, pulse = parse_tullyone(config_model, config_pulse)
        assert isinstance(hami, TullyOne)
        print(f"{hami=}")
        
    # load file_td1 and test the parsing
    with open(file_td1, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
        config_model = config['model']
        config_pulse = None if 'pulse' not in config else config['pulse']
        hami, pulse = parse_tullyone(config_model, config_pulse)
        assert isinstance(hami, TullyOneTD1)
        print(f"{hami=}")
        
    # load file_td2 and test the parsing
    with open(file_td2, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 
        config_model = config['model']
        config_pulse = None if 'pulse' not in config else config['pulse']
        hami, pulse = parse_tullyone(config_model, config_pulse)
        assert isinstance(hami, TullyOneTD2)
        print(f"{hami=}")
    
    
# %%
if __name__ == "__main__":
    test_parse_tullyone()

# %%
