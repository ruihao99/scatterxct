import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as LA
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

from enum import Enum

class StateRepresentation(Enum):
    DIABATIC = 0
    ADIABATIC = 1

class ScatterMovie:
    def __init__(self, R: ArrayLike, state_representation: int = 1):
        if state_representation == 0:
            self.state_representation = StateRepresentation.DIABATIC
            self.ylabel = "Diabatic States Nuclear Probability Density"
        elif state_representation == 1:
            self.state_representation = StateRepresentation.ADIABATIC
            self.ylabel = "Adiabatic States Nuclear Probability Density"
        else:
            raise ValueError("state_representation must be 0 or 1")
            
        self.R = R
        self.fig, self.ax = plt.subplots(dpi=250,)
        self.axtwinx = self.ax.twinx()
        self.axtwinx.set_ylabel("Energy (a.u.)")
        self.frames = []
        self.H_frames = []
        self.times = []
        self.FPS = 8
        
    def get_adiabatic_energy_levels(self, H: ArrayLike) -> ArrayLike:
        nstates, _, ngrid = H.shape
        E = np.zeros((nstates, ngrid), dtype=np.float64)
        if self.state_representation == StateRepresentation.DIABATIC:
            E[:, :] = np.diagonal(H, axis1=0, axis2=1).T
        else:
            for ii in range(ngrid):
                H_ii = H[:, :, ii]
                evals, _ = LA.eigh(H_ii)
                E[:, ii] = evals
        return E
    
        
    def append_frame(self, time: float, probability_density: ArrayLike, H: ArrayLike) -> None:
        self.times.append(time)
        self.frames.append(probability_density.copy())
        self.H_frames.append(H.copy())
        
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
        E = self.get_adiabatic_energy_levels(self.H_frames[i])
        for ii in range(nstates):
            self.ax.plot(self.R, self.frames[i][:, ii], label=f"State {ii}")
            self.axtwinx.plot(self.R, E[ii, :], ls="--", lw=2)
        self.ax.set_title(f"Time: {self.times[i]: .2f} a.u.")
        # use text to replace the buggy ylabel
        self.axtwinx.text(1.18, 0.5, "Energy (a.u.)", ha='center', va='center', rotation=90, transform=self.axtwinx.transAxes)
        
        self.add_labels()
        
    def make_movie(self,) -> None:
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.frames), interval=1000/self.FPS) 
        
    def save_animation(self, fname: str) -> None:
        self.animation.save(fname)