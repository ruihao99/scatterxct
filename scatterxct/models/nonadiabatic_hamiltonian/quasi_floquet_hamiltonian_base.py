from numpy.typing import ArrayLike

from .td_hamiltonian_base import TD_HamiltonianBase
from ...pulses import PulseBase as Pulse
from ...pulses import get_carrier_frequency
from ..floquet import get_HF, FloquetType, _dim_to_dimF

from typing import Union
from enum import Enum, unique

@unique
class FloquetablePulses(Enum):
    MORLET = "Morlet"
    MORLET_REAL = "MorletReal"
    COSINE = "CosinePulse"
    SINE = "SinePulse"
    EXPONENTIAL = "ExponentialPulse"

@unique 
class ValidQuasiFloqeuetPulses(Enum):
    GAUSSIAN = "Gaussian"
    UNIT = "UnitPulse"
    
def get_floquet_type_from_pulsetype(pulsetype: FloquetablePulses) -> FloquetType:
    if pulsetype == FloquetablePulses.MORLET_REAL:
        return FloquetType.COSINE
    elif pulsetype == FloquetablePulses.MORLET:
        return FloquetType.EXPONENTIAL
    elif pulsetype == FloquetablePulses.COSINE:
        return FloquetType.COSINE
    elif pulsetype == FloquetablePulses.SINE:
        return FloquetType.SINE
    elif pulsetype == FloquetablePulses.EXPONENTIAL:
        return FloquetType.EXPONENTIAL
    else:
        raise NotImplementedError(f"The quasi-floquet model for pulse type {pulsetype} is not implemented yet.")
    
def check_original_pulse(pulse: Pulse) -> FloquetType:
    try:
        pulse_type = FloquetablePulses(pulse.__class__.__name__)
    except ValueError:
        raise ValueError(f"The pulse {pulse.__class__.__name__} is not a Floquet-able pulse.")
    return get_floquet_type_from_pulsetype(pulse_type)

def check_validity_of_floquet_pulse(pulse: Pulse) -> None:
    try:
        ValidQuasiFloqeuetPulses(pulse.__class__.__name__)
    except ValueError:
        raise ValueError(f"The pulse {pulse.__class__.__name__} is not a valid quasi-Floquet pulse.")


class QuasiFloquetHamiltonianBase(TD_HamiltonianBase):
    def __init__(
        self,
        dim: int,
        orig_pulse: Pulse,
        floq_pulse: Pulse,
        NF: int,
        Omega: Union[float, None]=None,
        floquet_type: Union[FloquetType, None]=None,
    ) -> None:
        """ Quasi-Floquet Hamiltonian for a time-dependent Hamiltonian """
        """ whose time dependence is definded by a 'Pulse' object. """
        if Omega is None:
            self.Omega = get_carrier_frequency(orig_pulse)
        else:
            assert Omega == get_carrier_frequency(orig_pulse)
            self.Omega = Omega
        assert self.Omega is not None 
        
        if floquet_type is None:
            self.floquet_type = check_original_pulse(orig_pulse)
        else:
            assert floquet_type == check_original_pulse(orig_pulse)
            self.floquet_type = floquet_type
            
        check_validity_of_floquet_pulse(floq_pulse)
            
        
        super().__init__(dim, floq_pulse)
        self.NF = NF
        self.floquet_type = floquet_type
        self.orig_pulse = orig_pulse
        self.floq_pulse = floq_pulse
        
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.H0(r), self.H1(t, r), self.Omega, self.NF, floquet_type=self.floquet_type) 
    
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.dH0dR(r), self.dH1dR(t, r), self.Omega, self.NF, floquet_type=self.floquet_type, is_gradient=True)
    
    def get_floquet_space_dim(self) -> int:
        return _dim_to_dimF(self.dim, self.NF)
    
    def set_NF(self, NF: int) -> None:
        if isinstance(NF, int) and NF > 0:
            self.NF = NF
        else:
            raise ValueError(f"The number of Floquet replicas must be a positive integer, but {NF} is given.")
        
    def get_carrier_frequency(self) -> float:
        return self.Omega