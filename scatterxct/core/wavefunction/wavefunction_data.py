# %%
import numpy as np
from numpy.typing import ArrayLike
import pyfftw
from pyfftw import FFTW

from dataclasses import dataclass
from enum import Enum, unique

@unique
class WaveFunctionType(Enum):
    REAL_SPACE = 1
    RECIPROCAL_SPACE = 2
    
@dataclass
class WaveFunctionStatus:
    type: WaveFunctionType


@dataclass(frozen=True)
class WaveFunctionData:
    """ The shape of the wavefunction is (ngrid, nstates). """
    # psi_R: np.ndarray  # The wavefunction in real space
    # psi_k: np.ndarray  # The wavefunction in reciprocal space
    psi: np.ndarray  # The wavefunction 
    fft_object: FFTW  # The FFTW object used to transform the wavefunction
    ifft_object: FFTW  # The FFTW object used to transform the wavefunction
    status: WaveFunctionStatus
    
    @classmethod
    def from_numpy_psi(cls, psi_in: ArrayLike) -> "WaveFunctionData":
        # create the real and reciprocal space arrays
        # psi_R = pyfftw.empty_aligned(psi_in.shape, dtype='complex128')
        # psi_k = pyfftw.empty_aligned(psi_in.shape, dtype='complex128')
        psi = pyfftw.empty_aligned(psi_in.shape, dtype='complex128')
        # Create the FFTW objects
        fft_object = FFTW(psi, psi, axes=(0,), direction='FFTW_FORWARD')
        ifft_object = FFTW(psi, psi, axes=(0,), direction='FFTW_BACKWARD')
        # Initialize the wavefunction in real space
        psi[:] = psi_in
        # wavefunction_type = WaveFunctionType.REAL_SPACE
        status = WaveFunctionStatus(type=WaveFunctionType.REAL_SPACE)
        # return the WaveFunctionData object
        return cls(psi, fft_object, ifft_object, status)
    
    def real_space_to_k_space(self) -> ArrayLike:
        if self.status.type == WaveFunctionType.REAL_SPACE:
            self.status.type = WaveFunctionType.RECIPROCAL_SPACE
            self.psi[:] = self.fft_object()
            return self.psi
        else:
            raise RuntimeError("Tried to convert a k-space wavefunction to k-space. Check the wavefunction type!")
        
    def k_space_to_real_space(self) -> ArrayLike: 
        if self.status.type == WaveFunctionType.RECIPROCAL_SPACE:
            self.status.type = WaveFunctionType.REAL_SPACE
            self.psi[:] = self.ifft_object()
            return self.psi
        else:
            raise RuntimeError("Tried to convert a real-space wavefunction to real-space. Check the wavefunction type!")
        
    
    @property
    def ngrid(self) -> int:
        return self.psi.shape[0]
    
    @property
    def nstates(self) -> int:
        return self.psi.shape[1]
# %%
def test_main():
    psi = np.zeros((5, 2), dtype=np.complex128)
    psi[0, 0] = 1.0
    wavefunction_data = WaveFunctionData.from_numpy_psi(psi)
    print(wavefunction_data.psi)
    print("initial wavefunction")
    print("=====")
    
    # modify the wavefunction in real space
    wavefunction_data.psi[:] = np.random.rand(5, 2) + 1j * np.random.rand(5, 2)
    psi_copy = wavefunction_data.psi.copy()
    print(wavefunction_data.psi)
    print("modified wavefunction in real space")
    print("=====")
    
    # convert the wavefunction to k-space
    wavefunction_data.real_space_to_k_space()
    print(np.fft.fftn(psi_copy[:, 0]))
    print(wavefunction_data.psi[:, 0])  
    # print(wavefunction_data.psi)
    print(np.fft.fftn(psi_copy[:, 1]))
    print(wavefunction_data.psi[:, 1])
    psi_copy_k = np.array([np.fft.fftn(psi_copy[:, 0]), np.fft.fftn(psi_copy[:, 1])]).T
    print(np.allclose(wavefunction_data.psi, psi_copy_k))   
    print("converted wavefunction to k-space")
    print("=====")
    
    # modify the wavefunction in k-space
    wavefunction_data.psi[:] /= 2
    psi_copy_k[:] /= 2
    print(wavefunction_data.psi) 
    print("modified wavefunction in k-space")
    print("=====")
    
    # convert the wavefunction to real space
    wavefunction_data.k_space_to_real_space()
    print(wavefunction_data.psi)
    psi_copy = np.array([np.fft.ifftn(psi_copy_k[:, 0]), np.fft.ifftn(psi_copy_k[:, 1])]).T
    print(np.allclose(wavefunction_data.psi, psi_copy))
    
    print("converted wavefunction to real space")
    print("=====")
    
    return wavefunction_data

# %%
if __name__ == "__main__":
    test_main()

# %%
