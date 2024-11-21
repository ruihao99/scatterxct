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
class TullyOneTD1(HamiltonianBase):
    nquant: int = 2
    nclass: int = 1
    laser: bool = True
    A: float = 0.01
    B: float = 1.6
    C: float = 0.005
    D: float = 1.0
    mass: RealVector = field(default_factory=lambda : np.array([2000.0], dtype=np.float64))
    
    @classmethod
    def init(
        cls,
        A: float=0.01, 
        B: float=1.6, 
        C: float=0.005,
        D: float=1.0, 
        mass: float=2000.0
    ):
        return cls(
            A=A, B=B, C=C, D=D, 
            mass=np.array([mass], dtype=np.float64), 
        )    
    
    def eval_hami(self, R: RealVector, Uold: ComplexOperator=None) -> HamiData:
        # compute the diabatic Hamiltonian and its gradient
        H0 = H0_diab(R, self.A, self.B, self.C, self.D)
        gradH0 = gradH0_diab(R, self.A, self.B, self.C, self.D)
        
        # compute the adiabatic Hamiltonian and its gradient
        E0, U0 = diagonalize_and_project(H0, Uold)
        G0 = udagger_o_u(gradH0, U0)
        
        # in this model, we assume the dipole operator is sigma_x 
        # i.e., only the transition dipole beteween the two states is non-zero
        mu = self.dipole()
        gradmu = self.grad_dipole()
        
        # in the tully model, there's no external classical potential
        V = 0.0
        gradV = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(H0, gradH0, E0, G0, U0, mu, gradmu, V, gradV)
       
        
    def dipole(self, ) -> ComplexOperator:
        """ Dipole moment operator to interact with the electric field """
        return np.array(
            [[0.0, 1.0],
             [1.0, 0.0]],
            dtype=np.complex128,
        )
        
    def grad_dipole(self,) -> ComplexVectorOperator:  
        """ Gradient of the dipole moment operator """
        return np.zeros((2, 2, self.nclass), dtype=np.complex128)
   
    
def test_tullyone_td1():
    hami = TullyOneTD1.init()
    
    N = 100 
    t = np.linspace(0, 4, N)
    R = np.linspace(-4, 4, N).reshape(-1, 1) 
    
    Eout = np.zeros((N, hami.nquant), dtype=np.complex128)
    Gout = np.zeros((hami.nquant, hami.nquant, 1, N), dtype=np.complex128)
    
   
    for ii, (tt, rr) in enumerate(zip(t, R)):
        U = None if ii == 0 else hdata.U0
        hdata = hami.eval_hami(rr)
        Eout[ii, :] = hdata.E0
        Gout[:, :, :, ii] = hdata.G0
        
        
    from matplotlib import pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], Eout[:, 0].real, label="H11")
    ax.plot(R[:, 0], Eout[:, 1].real, label="H22")
    ax.legend() 
    plt.show()

# %%
if __name__ == '__main__':
    test_tullyone_td1()
    
# %%
