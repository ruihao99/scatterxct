import numpy as np

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
from pymddrive.integrators.state import State
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 


def estimate_delay_time_tullyone(R0, P0, mass: float=2000.0):
    # model = TullyOne(A, B, C, D)
    hamiltonian = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE
    )
    print(f"{R0=}")
    print(f"{P0=}")
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=R0, P=P0, rho=rho0)
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=BasisRepresentation.Diabatic,
        qm_rep=QunatumRepresentation.DensityMatrix,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        dt=1,
        save_every=1
    )
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1]