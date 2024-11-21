# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector
from scatterxct.hamiltonian.hamiltonian_base import HamiltonianBase, HamiData
from scatterxct.hamiltonian.tullyone.common import H0_diab, gradH0_diab
from scatterxct.hamiltonian.linalg import udagger_o_u
from scatterxct.hamiltonian.phase_tracking import PhaseTracking

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
        mass: float=2000.0,
        phase_tracking: PhaseTracking=PhaseTracking("none")
    ):
        return cls(
            A=A, B=B, C=C, D=D, 
            mass=np.array([mass], dtype=np.float64), 
            phase_tracking=phase_tracking
        )    
    
    def eval_hami(self, R: RealVector, Uold: ComplexOperator=None) -> HamiData:
        # compute the diabatic Hamiltonian 
        Hd = H0_diab(R, self.A, self.B, self.C, self.D)
        Gd = gradH0_diab(R, self.A, self.B, self.C, self.D)
        Md = self.dipole()
        Dd = self.grad_dipole()
        
        # compute the adiabatic Hamiltonian 
        E, U = self.phase_tracking.diag(Hdiab=Hd, U_last=Uold)
        Ha = np.diagflat(E)
        Ga = udagger_o_u(Gd, U)
        Ma = udagger_o_u(Md, U)
        Da = udagger_o_u(Dd, U)
        
        # in the tully model, there's no external classical potential
        Vext = 0.0
        Gext = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(Hd, Gd, Md, Dd, Ha, Ga, Ma, Da, U, Vext, Gext)
    
        
    def dipole(self, ) -> ComplexOperator:
        """ Dipole moment operator to interact with the electric field """
        return np.array(
            [[0.0, 1.0],
             [1.0, 0.0]],
            dtype=np.complex128,
        )
        
    def grad_dipole(self,) -> ComplexVectorOperator:  
        """ Gradient of the dipole moment operator """
        return np.zeros((2, 2, self.nclass), dtype=np.complex128, order='F')
    
    def harmornic_params(self):
        raise ValueError("TullyOneTD1 does not have harmonic potential")
    
    def morse_params(self):
        raise ValueError("TullyOneTD1 does not have Morse potential") 
    
    def get_max_dipole(self):
        return 1.2
   
    
def test_tullyone_td1():
    hami = TullyOneTD1.init()
    
    N = 100 
    t = np.linspace(0, 4, N)
    R = np.linspace(-4, 4, N).reshape(-1, 1) 
    
    Eout = np.zeros((N, hami.nquant), dtype=np.complex128)
    Gout = np.zeros((hami.nquant, hami.nquant, 1, N), dtype=np.complex128)
    mu_diab_out = np.zeros((N, hami.nquant, hami.nquant), dtype=np.complex128)
    mu_adia_out = np.zeros((N, hami.nquant, hami.nquant), dtype=np.complex128)
    
    for ii, (tt, rr) in enumerate(zip(t, R)):
        U = None if ii == 0 else hdata.U
        hdata = hami.eval_hami(rr)
        Eout[ii, :] = hdata.Ha.diagonal()
        Gout[:, :, :, ii] = hdata.Ga
        mu_diab_out[ii, :, :] = hdata.Md
        mu_adia_out[ii, :, :] = hdata.Ma
        
    from matplotlib import pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], Eout[:, 0].real, label="H11")
    ax.plot(R[:, 0], Eout[:, 1].real, label="H22")
    ax.legend() 
    plt.show()
    
    # plot diabatic transition dipole moment
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], mu_diab_out[:, 0, 1].real, label="mu12")
    ax.legend()
    plt.show()
    
    # plot adiabatic transition dipole moment
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], mu_adia_out[:, 0, 1].real, label="mu12")
    ax.plot(R[:, 0], mu_adia_out[:, 1, 0].real, label="mu21")   
    ax.legend()
    plt.show()


# %%
if __name__ == '__main__':
    test_tullyone_td1()
    
# %%
