import numpy as np
from numpy.typing import ArrayLike
from numba import njit
from matplotlib.animation import FuncAnimation

from scatterxct.core.wavefunction import get_nuclear_density

import matplotlib.pyplot as plt

from enum import Enum

@njit
def diabatic_to_adiabatic(psi_diabatic, U0):
    psi_adiabatic = np.zeros_like(psi_diabatic)
    ngrid, _ = psi_diabatic.shape
    for ii in range(ngrid):
        U0_ii = np.ascontiguousarray(U0[:, :, ii])
        psi_ii = np.ascontiguousarray(psi_diabatic[ii])
        # psi_adiabatic[ii, :] = np.dot(np.conj(U0[:,:, i]).T, np.ascontiguousarray(psi_diabatic[ii]))
        psi_adiabatic[ii, :] = np.dot(U0_ii.conj().T, psi_ii)
    return psi_adiabatic

class StateRepresentation(Enum):
    DIABATIC = 0
    ADIABATIC = 1

class ScatterMovie:
    def __init__(self, R: ArrayLike, E0, U0, state_representation: int = 1):
        if state_representation == 0:
            self.state_representation = StateRepresentation.DIABATIC
            self.ylabel = "Diabatic States Nuclear Probability Density"
        elif state_representation == 1:
            self.state_representation = StateRepresentation.ADIABATIC
            self.ylabel = "Adiabatic States Nuclear Probability Density"
        else:
            raise ValueError("state_representation must be 0 or 1")
            
        self.R = R
        self.E0 = E0
        self.U0 = U0
        self.fig, self.ax = plt.subplots(dpi=250,)
        self.axtwinx = self.ax.twinx()
        self.axtwinx.set_ylabel("Energy (a.u.)")
        self.frames = []
        self.times = []
        self.FPS = 8
        
    def append_frame(self, time: float, diabatic_wavefunction: ArrayLike) -> None:
        self.times.append(time)
        psia = diabatic_to_adiabatic(diabatic_wavefunction, self.U0)   
        n = get_nuclear_density(psia, dR=self.R[1] - self.R[0])
        self.frames.append(n)
        
    def add_labels(self, ) -> None: 
        self.ax.set_xlabel("R (a.u.)")
        self.ax.set_ylabel(self.ylabel) 
        self.fig.legend(
            handles=self.ax.lines, 
            labels=[f"State {ii}" for ii in range(self.frames[0].shape[1])], 
            loc = 'upper left',
            bbox_to_anchor=(0.15, 0.80)
        )
        
    def animate(self, i):
        self.ax.clear()
        self.axtwinx.clear() if self.axtwinx else None
        # axtwinx = self.ax.twinx()
        # self.axtwinx = axtwinx
        nstates = self.frames[i].shape[1]
        # E = self.get_adiabatic_energy_levels(self.H_frames[i])
        for ii in range(nstates):
            self.ax.plot(self.R, self.frames[i][:, ii], label=f"State {ii}")
            self.axtwinx.plot(self.R, self.E0[ii, :], ls="--", lw=2)
        self.ax.set_title(f"Time: {self.times[i]: .2f} a.u.")
        # use text to replace the buggy ylabel
        self.axtwinx.text(1.18, 0.5, "Energy (a.u.)", ha='center', va='center', rotation=90, transform=self.axtwinx.transAxes)
        
        self.add_labels()
        
    def make_movie(self,) -> None:
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.frames), interval=1000/self.FPS) 
        
    def save_animation(self, fname: str) -> None:
        self.animation.save(fname)