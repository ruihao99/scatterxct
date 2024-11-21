# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector
from scatterxct.hamiltonian.hamiltonian_base import HamiltonianBase, HamiData
from scatterxct.hamiltonian.tullyone.common import (
    H0_diab, gradH0_diab, V12, gradV12
) 
from scatterxct.hamiltonian.linalg import udagger_o_u
from scatterxct.hamiltonian.phase_tracking import PhaseTracking

from dataclasses import dataclass, field

@dataclass
class TullyOneTD3(HamiltonianBase):
    nquant: int = 2
    nclass: int = 1
    laser: bool = True
    A: float = 0.01
    B: float = 1.6
    C: float = 0.005
    D: float = 1.0 
    C1: float = 1.0
    D1: float = 1.0
    mass: RealVector = field(default_factory=lambda : np.array([2000.0], dtype=np.float64))
    
    @classmethod
    def init(
        cls,
        A: float=0.01, 
        B: float=1.6, 
        C: float=0.005,
        D: float=1.0, 
        C1: float=1.0,
        D1: float=1.0,
        mass: float=2000.0,
        phase_tracking: PhaseTracking=PhaseTracking("none")
    ) -> 'TullyOneTD2':
        return cls(
            A=A, B=B, C=C, D=D, C1=C1, D1=D1, 
            mass=np.array([mass], dtype=np.float64),
            phase_tracking=phase_tracking
        )
    
    def eval_hami(self, R: RealVector, Uold: ComplexOperator=None) -> HamiData:
        # compute the diabatic Hamiltonian 
        Hd = H0_diab(R, self.A, self.B, self.C, self.D)
        Gd = gradH0_diab(R, self.A, self.B, self.C, self.D)
        Md = self.dipole(R)
        Dd = self.grad_dipole(R)
        
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
    
    def eval_hami_recursive(self, R: RealVector, Hold: ComplexOperator, Uold: ComplexVectorOperator, dt: float) -> HamiData:
        # compute the diabatic Hamiltonian 
        Hd = H0_diab(R, self.A, self.B, self.C, self.D)
        Gd = gradH0_diab(R, self.A, self.B, self.C, self.D)
        Md = self.dipole(R)
        Dd = self.grad_dipole(R)
        
        # compute the adiabatic Hamiltonian 
        E, U = recursive_project(H, Hold. Uold, dt)
        Ha = np.diagflat(E)
        Ga = udagger_o_u(Gd, U)
        Ma = udagger_o_u(Md, U)
        Da = udagger_o_u(Dd, U)
        
        # in the tully model, there's no external classical potential
        Vext = 0.0
        Gext = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(Hd, Gd, Md, Dd, Ha, Ga, Ma, Da, U, Vext, Gext) 
         
       
    def dipole(self, R: RealVector) -> ComplexOperator:
        """ Dipole moment operator to interact with the electric field """
        v12 = np.sum(V12(R, self.C1, self.D1))
        return np.array(
            [[v12, 0.0],
             [0.0, -v12]],
            dtype=np.complex128,
        )
        
    def grad_dipole(self, R: RealVector) -> ComplexVectorOperator:
        """ Gradient of the dipole moment operator """
        gradv12 = gradV12(R, self.C1, self.D1)
        zeros = np.zeros(self.nclass, dtype=np.complex128)
        return np.array(
            [[gradv12, zeros],
             [zeros, -gradv12]],
            dtype=np.complex128,
        )
        
    def harmornic_params(self):
        raise ValueError("TullyOneTD3 does not have harmonic potential")    
    
    def morse_params(self):
        raise ValueError("TullyOneTD3 does not have Morse potential")
    
    def get_max_dipole(self):
        return 1.2 * self.C1
    
def test_tullyone_td3():
    hami = TullyOneTD3.init()
    
    N = 100 
    t = np.linspace(0, 3, N)
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
    L = 0.0075
    ax.axhline(y=L, color='r', linestyle='--', label="Laser")
    ax.axhline(y=-L, color='r', linestyle='--')
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
    ax.plot(R[:, 0], mu_adia_out[:, 0, 0].real, label="mu11")
    ax.plot(R[:, 0], mu_adia_out[:, 1, 1].real, label="mu22")
    ax.plot(R[:, 0], mu_adia_out[:, 0, 1].real, label="mu12")
    ax.plot(R[:, 0], mu_adia_out[:, 1, 0].real, label="mu21")
    ax.legend()
    plt.show()

# %%
if __name__ == '__main__':
    test_tullyone_td3()
    
# %%
        