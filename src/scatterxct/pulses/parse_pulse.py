# %%
from .pulse_base import PulseBase
from .cosine_pulse import CosinePulse
from .sine_pulse import SinePulse
from .morlet_pulse import MorletPulse
from .sine_square_pulse import SineSquarePulse

def parse_pulse(
    pulse_config: dict 
) -> PulseBase:
    if pulse_config['kind'] == 'cosine':
        return CosinePulse.init(**pulse_config['params'])
    
    elif pulse_config['kind'] == 'sine':
        return SinePulse.init(**pulse_config['params'])
    
    elif pulse_config['kind'] == 'morlet':
        return MorletPulse.init(**pulse_config['params'])
    
    elif pulse_config['kind'] == 'sine_square':
        return SineSquarePulse.init(**pulse_config['params'])
    
    else:
        raise ValueError(f"Unsupported pulse kind: {pulse_config['kind']}") 
        

def test_parse_pulse():
    import yaml
    from fmdriverpy.utils import traverse_and_evaluate
    
    # test cosine pulse
    with open("../../../example/pulses/cosine_pulse.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = traverse_and_evaluate(config)
        config_pulse = config['pulse']
        pulse = parse_pulse(config_pulse) 
        assert isinstance(pulse, CosinePulse)
        
    # test sine pulse
    with open("../../../example/pulses/sine_pulse.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = traverse_and_evaluate(config)
        config_pulse = config['pulse']
        pulse = parse_pulse(config_pulse) 
        assert isinstance(pulse, SinePulse)
        
    # test morlet pulse
    with open("../../../example/pulses/morlet_pulse.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = traverse_and_evaluate(config)
        config_pulse = config['pulse']
        pulse = parse_pulse(config_pulse) 
        assert isinstance(pulse, MorletPulse)
        
    # test sine square pulse
    with open("../../../example/pulses/sine_square_pulse.yaml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = traverse_and_evaluate(config)
        config_pulse = config['pulse']
        pulse = parse_pulse(config_pulse) 
        assert isinstance(pulse, SineSquarePulse)
        
# %%
if __name__ == "__main__":
    test_parse_pulse()

# %%
