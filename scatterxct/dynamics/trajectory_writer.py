import numpy as np
from numpy.typing import NDArray
from netCDF4 import Dataset

import tempfile
import shutil

class TrajectoryWriter:
    def __init__(self, ngrid: int, dim: int, R_grid: NDArray[np.float64], K_grid: NDArray[np.float64]) -> None:
        # create a randomized filename
        self.fn = tempfile.NamedTemporaryFile().name
        
        # create a netCDF file in write mode
        self.nc = Dataset(self.fn, 'w', format='NETCDF3_64BIT_OFFSET')
        
        # create dimensions
        self.frame = self.nc.createDimension('frame', None)
        self.grid = self.nc.createDimension('grid', ngrid)
        self.dim = self.nc.createDimension('dim', dim)
        
        # create data variables
        self.time = self.nc.createVariable('time', 'f8', ('frame',))
        self.R_grid = self.nc.createVariable('R', 'f8', ('grid'))
        self.K_grid = self.nc.createVariable('K', 'f8', ('grid'))
        self.psi_diabatic_re = self.nc.createVariable('psi_diabatic_re', 'f8', ('frame', 'grid', 'dim'))
        self.psi_diabatic_im = self.nc.createVariable('psi_diabatic_im', 'f8', ('frame', 'grid', 'dim'))
        
        # write the grid
        self.R_grid[:] = R_grid
        self.K_grid[:] = K_grid
        
    def write_frame(self, t: float, psi: NDArray[np.complex128]) -> None:
        iframe = self.frame.size
        self.time[iframe] = t
        self.psi_diabatic_re[iframe] = psi.real
        self.psi_diabatic_im[iframe] = psi.imag
        
    def close(self) -> None:
        self.nc.close() 
        
    def save(self, target_fn: str) -> None:
        self.close()
        shutil.copy(self.fn, target_fn)