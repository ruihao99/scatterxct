import numpy as np
from numpy.typing import NDArray

from typing import List, Any
import tempfile
import shutil

class PropertiesWriter:
    def __init__(self, dim: int) -> None:
        # create a randomized filename
        self.fn = tempfile.NamedTemporaryFile().name
        
        # create a plain text file
        self.f = open(self.fn, 'w')
        
        # create header
        header_list = ['time', 'R', 'P', 'KE', 'PE'] + [f'apop_{i}' for i in range(dim)] + [f'dpop_{i}' for i in range(dim)]
        
        # write header
        self.write_line(header_list, is_header=True)
        
    def write_line(self, fields: List[Any], is_header: bool=False) -> None:
        if not is_header:
            line_str = self.get_line(fields) 
        else:
            line_str = '{:<s}'.format('# ') + f'{fields[0]:>10s}' + self.get_line(fields[1:])
        self.f.write(line_str)
            
        
    @staticmethod
    def get_line(fields: List[Any]) -> str:
        line_str = ''
        for field in fields:
            if isinstance(field, float):
                line_str += f'{field:>12.6f}'
            elif isinstance(field, int):
                line_str += f'{field:>12d}'
            elif isinstance(field, complex):
                raise ValueError('Complex numbers are not supported yet.')
            else:
                line_str += f'{str(field):>12s}'
        line_str += '\n'
        return line_str
        
    def write_frame(
        self, 
        t: float, 
        Ravg: float, 
        Pavg: float,
        KE: float,
        PE: float,
        adiabatic_populations: NDArray[np.float64],
        diabatic_populations: NDArray[np.float64]
    ) -> None:
        fields = [t, Ravg, Pavg, KE, PE] + list(adiabatic_populations) + list(diabatic_populations)
        self.write_line(fields)
        
    def close(self) -> None:
        self.f.close()
        
    def save(self, target_fn: str) -> None:
        self.close()
        shutil.copy(self.fn, target_fn)
        