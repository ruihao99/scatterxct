# %%
import numpy as np

from scatterxct.mytypes import ComplexOperator, ComplexVectorOperator, RealVector
from scatterxct.hamiltonian.hamiltonian_base import HamiltonianBase, HamiData
from scatterxct.hamiltonian.tullyone.common import H0_diab, gradH0_diab
from scatterxct.hamiltonian.linalg import udagger_o_u
from scatterxct.hamiltonian.phase_tracking import PhaseTracking

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
        mass: float=2000.0,
        phase_tracking: PhaseTracking=PhaseTracking("none")
    ):
        return cls(A=A, B=B, C=C, D=D, 
                   mass=np.array([mass], dtype=np.float64), 
                   phase_tracking=phase_tracking)
    
    def eval_hami(
        self, R: RealVector, 
        Uold: ComplexOperator=None, 
    ) -> HamiData:
        # compute the diabatic Hamiltonian 
        Hd = H0_diab(R, self.A, self.B, self.C, self.D)
        Gd = gradH0_diab(R, self.A, self.B, self.C, self.D)
        Md = None
        Dd = None
        
        # compute the adiabatic Hamiltonian 
        E, U = self.phase_tracking.diag(Hdiab=Hd, U_last=Uold)
        Ha = np.diagflat(E)
        Ga = udagger_o_u(Gd, U)
            
        
        Ma = None
        Da = None
        
        # in the tully model, there's no external classical potential 
        Vext = 0.0
        Gext = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(Hd, Gd, Md, Dd, Ha, Ga, Ma, Da, U, Vext, Gext)
    
    def eval_hami_recursive(
        self, R: RealVector,
        Hold: ComplexOperator,
        Uold: ComplexVectorOperator,
        dt: float,
    ) -> HamiData:
        # compute the diabatic Hamiltonian 
        Hd = H0_diab(R, self.A, self.B, self.C, self.D)
        Gd = gradH0_diab(R, self.A, self.B, self.C, self.D)
        Md = None
        Dd = None
        
        # compute the adiabatic Hamiltonian 
        E, U = recursive_project(Hd, Hold, Uold, dt)
        Ha = np.diagflat(E)
        Ga = udagger_o_u(Gd, U)
            
        
        Ma = None
        Da = None
        
        # in the tully model, there's no external classical potential 
        Vext = 0.0
        Gext = np.zeros((self.nclass,), dtype=np.float64)
        
        return HamiData(Hd, Gd, Md, Dd, Ha, Ga, Ma, Da, U, Vext, Gext)
    
    def harmornic_params(self, ):
        raise ValueError(
            """ The Tully model does not have a harmonic approximation """
        )
        
    def morse_params(self, ):
        raise ValueError(
            """ The Tully model does not have a Morse potential """
        )
            
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
        U = None if ii == 0 else hdata.U
        hdata = tullyone.eval_hami(r, Uold=U)
        Eout[ii, :] = hdata.Ha.diagonal()
        Gout[:, :, :, ii] = hdata.Gd
       
    from matplotlib import pyplot as plt 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R[:, 0], Eout[:, 0].real, label="H11")
    ax.plot(R[:, 0], Eout[:, 1].real, label="H22")
    w = 0.01
    # w = 0.02
    ax.axhline(-w/2, color="black", linestyle="--")
    ax.axhline(w/2, color="black", linestyle="--")
    plt.show()
    
        
# %%
if __name__ == "__main__":
    test_tullyone()
# %%
