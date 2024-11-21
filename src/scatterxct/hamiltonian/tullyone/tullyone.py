# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector
from scatterxct.hamiltonian.hamiltonian_base import HamiltonianBase, HamiData
from scatterxct.hamiltonian.tullyone.common import H0_diab, gradH0_diab
from scatterxct.hamiltonian.linalg import (
    diagonalize_and_project,
    udagger_o_u
)

from dataclasses import dataclass, field

@dataclass
class TullyOne(HamiltonianBase):
    nquant: int = 2
    nclass: int = 1
    mass: RealVector = field(default_factory=lambda : np.array([2000.0], dtype=np.float64))
    laser: bool = False
    A: float = field(default=0.01) 
    B: float = field(default=1.6) 
    C: float = field(default=0.005) 
    D: float = field(default=1.0)  
    
    @classmethod    
    def init(
        cls, 
        A: float=0.01, 
        B: float=1.6, 
        C: float=0.005, 
        D: float=1.0,
        mass: float=2000.0
    ):
        return cls(A=A, B=B, C=C, D=D, mass=np.array([mass], dtype=np.float64))
    
    def eval_hami(self, R: RealVector, Uold: ComplexOperator=None) -> HamiData:
        # compute the diabatic Hamiltonian and its gradient
        H0 = H0_diab(R, self.A, self.B, self.C, self.D)
        gradH0 = gradH0_diab(R, self.A, self.B, self.C, self.D)
        
        # compute the adiabatic Hamiltonian and its gradient
        E0, U0 = diagonalize_and_project(H0, Uold)
        G0 = udagger_o_u(gradH0, U0)
        
        # in this time-independent case, don't need to evaluate dipole
        mu = None
        gradmu = None
        
        # in the tully model, there's no external classical potential 
        V = 0.0
        gradV = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(H0, gradH0, E0, G0, U0, mu, gradmu, V, gradV)
   
    
# %%
def test_tullyone():
    tullyone = TullyOne.init()
    
    nclass, nquant = tullyone.nclass, tullyone.nquant
    L = 4
    N = 100
    R = np.linspace(-L, L, N).reshape(-1, 1)
    Eout = np.zeros((N, nquant), dtype=np.complex128)
    Gout = np.zeros((nquant, nquant, 1, N), dtype=np.complex128)
    
    for ii, r in enumerate(R):
        U = None if ii == 0 else hdata.U0
        hdata = tullyone.eval_hami(r, Uold=U)
        Eout[ii, :] = hdata.E0
        Gout[:, :, :, ii] = hdata.G0
       
    from matplotlib import pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], Eout[:, 0].real, label="H11")
    ax.plot(R[:, 0], Eout[:, 1].real, label="H22")
    plt.show()
    
        
# %%
if __name__ == "__main__":
    test_tullyone()
# %%
