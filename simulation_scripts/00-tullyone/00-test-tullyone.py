# %%
import numpy as np

from scatterxct.dynamics.run import run_time_independent_dynamics
from scatterxct.models.tullyone import get_tullyone, TullyOnePulseTypes
from utils import get_tully_one_delay_time

from pathlib import Path
from typing import Optional

def estimate_dt(Omega: float, dt: float = 0.1) -> float:
    SAFTY_FACTOR: float = 10.0
    T: float = 2 * np.pi / Omega
    if dt > T / SAFTY_FACTOR:
        return T / SAFTY_FACTOR
    else:
        return dt
    

def main(
    R0: float = -10.0,
    k0: float = 30.0, 
    Omega: Optional[float] = None, 
    tau: Optional[float] = None,
    phi: Optional[float] = None,
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
):
    # estimate the delay time using MQC dynamics 
    if pulse_type != TullyOnePulseTypes.NO_PULSE:
        delay_time = get_tully_one_delay_time(R0, k0)
        
    # get the hamiltonian
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        hamiltonian = get_tullyone(
            pulse_type=TullyOnePulseTypes.NO_PULSE
        )
    else:
        hamiltonian = get_tullyone(
            t0=delay_time, Omega=Omega, tau=tau, phi=phi,
            pulse_type=pulse_type
        )
        
    # estimate the time step
    dt = 0.05 if Omega is None else estimate_dt(Omega)
    scale = 1.0
    
    fname_movie: Path = Path(f"./scatter_movie-k0_{k0}.gif")
    
    output = run_time_independent_dynamics(
        hamiltonian=hamiltonian,
        R0=R0, 
        k0=k0,
        dt=dt,
        initial_state=0,
        save_every=10,
        # fname_movie=fname_movie
        # fname_movie=None
        movie_path=fname_movie,
        scale=scale,
        apply_absorbing_boundary=False
    )
    
    return output

# %%
if __name__ == "__main__":
    R0 = -10.0
    k0 = 10.0
    Omega = 0.3
    tau = 100.0
    phi = 0.0
    
    output = main(R0, k0, Omega, tau, phi, TullyOnePulseTypes.PULSE_TYPE3)
    # output = main(R0, k0, Omega, tau, TullyOnePulseTypes.NO_PULSE)
# %%
if __name__ == "__main__":
    time = output['time']
    R = output['R']
    k = output['k']
    diab_populations = np.array(output['diab_populations'])
    adiab_populations = np.array(output['adiab_populations'])
    import matplotlib.pyplot as plt 
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    # ax.plot(time, np.sum(R, axis=1), label="R")
    ax.plot(time, R, label="R")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("position (a.u.)")
    plt.show()
    
    # total momentum
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    # ax.plot(time, np.sum(k*populations, axis=1), label="total momentum")
    # ax.plot(time, np.sum(k, axis=1), label="momentum")
    ax.plot(time, k, label="momentum")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("Momentum (a.u.)")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    nstates = diab_populations.shape[1]
    for i in range(2):
        ax.plot(time, diab_populations[:, i], label=f"state {i}")
        
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("Population")
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for i in range(2):
        ax.plot(time, adiab_populations[:, i], label=f"state {i}")
    ax.axhline(0.72, ls='--', color='black', label="reference from subotnik group")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("Adiabatic Population")
    ax.legend()
    
    KE = np.array(output['KE'])
    PE = np.array(output['PE'])
    
    # print(KE.shape, diab_populations.shape, PE.shape)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, KE, label="KE")   
    # ax.plot(time, np.nansum(KE*diab_populations, axis=1), label="KE total")
    # ax.plot(time, np.sum(k, axis=1)**2/2000, label="KE alt")
    
    ax.set_xlabel("time (a.u.)")    
    ax.set_ylabel("KE")
    ax.legend()
    mass = 2000.0
    ax.axhline(k0**2/2/mass, ls='--', color='black', label="classical KE")
    plt.show()
    
    print(PE.shape, diab_populations.shape)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    # ax.plot(time, np.sum(PE*diab_populations, axis=1), label="PE")
    ax.plot(time, PE, label="PE")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("PE")
    # ax.set_xlim(600, 1500)
    plt.show()
    
    
    # TE = np.nansum((KE + PE) * diab_populations, axis=1)
    TE = KE + PE 
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, TE, label="TE")
    ax.axhline(TE[0], ls='--', color='black', label="initial TE")
    ax.set_xlabel("time (a.u.)")
    ax.set_ylabel("Energy (a.u.)")
    plt.show()

    
# %%
