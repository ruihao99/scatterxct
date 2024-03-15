# %%
from scatterxct.dynamics.run import run_time_independent_dynamics
from scatterxct.models.tullyone import get_tullyone, TullyOnePulseTypes

def main():
    
    # initial conditions
    R0: float = -10.0
    k0: float = 4.0
    
    fname_movie: str = f"./scatter_movie-k0_{k0}.gif" 
    
    hamiltonian = get_tullyone(pulse_type=TullyOnePulseTypes.NO_PULSE)
    output = run_time_independent_dynamics(
        hamiltonian=hamiltonian,
        R0=R0, 
        k0=k0,
        initial_state=0,
        save_every=10,
        fname_movie=fname_movie
    )
    
    return output

# %%
if __name__ == "__main__":
    main()
    
# %%
