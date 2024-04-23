import numpy as np
from numpy.typing import NDArray

from typing import List, Any
import tempfile
import shutil

class ScatterWriter:
    def __init__(self, scatter_out: NDArray[np.float64], scatter_out_path: str) -> None:
        # create a randomized filename
        self.fn = tempfile.NamedTemporaryFile().name
        
        # create a plain text file
        self.f = open(self.fn, 'w')
        
        # create header
        header_list = ["RL", "TL", "RU", "TU"]
        
        # write header
        self.write_line(header_list, is_header=True)
        
        # write the scatter_out
        self.write_line(list(scatter_out))
        
        # save the scatter_out
        self.save(scatter_out_path)
        
        
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
        
    def close(self) -> None:
        self.f.close()
        
    def save(self, target_fn: str) -> None:
        self.close()
        shutil.copy(self.fn, target_fn)
        