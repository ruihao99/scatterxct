# %% the package
import numpy as np

from .tullyone import TullyOne
from .tullyone_td_type1 import TullyOneTD_type1
from .tullyone_td_type2 import TullyOneTD_type2
from .tullyone_floquet_type1 import TullyOneFloquet_type1
from .tullyone_floquet_type2 import TullyOneFloquet_type2
from ...pulses import PulseBase as Pulse
from ...pulses import MorletReal, Gaussian, UnitPulse

from typing import Union
from numbers import Real
from enum import Enum, unique

@unique
class TullyOnePulseTypes(Enum):
    NO_PULSE = "NoPulse"
    ZEROPULSE = "ZeroPulse" # for debugging
    UNITPULSE = "UnitPulse" # for debugging
    PULSE_TYPE1 = "PulseType1"
    PULSE_TYPE2 = "PulseType2"
    PULSE_TYPE3 = "PulseType3"

@unique
class TD_Methods(Enum):
    BRUTE_FORCE = "BruteForce"
    FLOQUET = "Floquet"

def get_tullyone(
    A: Real = 0.01, B: Real = 1.6, C: Real = 0.005, D: Real = 1.0, # Tully parameters
    t0: Union[Real, None] = None, Omega: Union[Real, None] = None,
    tau: Union[Real, None] = None, phi: Union[Real, None] = None, # pulse parameters
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
    td_method: TD_Methods = TD_Methods.BRUTE_FORCE,
    NF: Union[int, None] = None
):
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOne(A=A, B=B, C=C, D=D)
        else:
            raise ValueError(f"You are trying to get a floquet model for a time independent Hamiltonian.")
    else:
        if (t0 is None) or (Omega is None) or (tau is None):
            raise ValueError(f"You need to provide the pulse parameters t0, Omega, and tau for Time-dependent problems.")

    if td_method == TD_Methods.FLOQUET and NF is None:
        raise ValueError(f"You need to provide the number of Floquet replicas NF for Floquet models.")

    if pulse_type == TullyOnePulseTypes.PULSE_TYPE1:
        orig_pulse = MorletReal(A=C, t0=t0, tau=tau, Omega=Omega, phi=phi)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=0, D=0, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")

    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE2:
        orig_pulse = MorletReal(A=1.0, t0=t0, tau=tau, Omega=Omega, phi=phi)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type2(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type2(A=A, B=B, C=C, D=D, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")

    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE3:
        orig_pulse = MorletReal(A=C/2, t0=t0, tau=tau, Omega=Omega, phi=phi)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C/2, D=D, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=C/2, D=D, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
    elif pulse_type == TullyOnePulseTypes.ZEROPULSE:
        orig_pulse = Pulse()
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    elif pulse_type == TullyOnePulseTypes.UNITPULSE:
        orig_pulse = UnitPulse(A=C)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    else:
        raise ValueError(f"Invalid pulse type: {pulse_type}")

# %% testing/debugging code

def _evaluate_tullyone_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_hamiltonian, evaluate_nonadiabatic_couplings
    dim_elc = model.dim
    dim_cls = r.size
    E_out = np.zeros((dim_cls, dim_elc))
    F_out = np.zeros((dim_cls, dim_elc))
    d_out = np.zeros((dim_cls, dim_elc, dim_elc), dtype=np.complex128)
    for ii, rr in enumerate(r):
        _, dHdR, evals, evecs = evaluate_hamiltonian(t, rr, hamiltonian=model)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        E_out[ii, :] = evals
        F_out[ii, :] = F
        d_out[ii, :, :] = d
    return E_out, F_out, d_out

def _evaluate_tullyone_floquet_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_hamiltonian, evaluate_nonadiabatic_couplings, nac_phase_following
    dim_elc = model.dim
    NF = model.NF
    dim_F = dim_elc*(NF*2+1)
    E_out = np.zeros((len(r), dim_F))
    F_out = np.zeros((len(r), dim_F))
    d_out = np.zeros((len(r), dim_F, dim_F), dtype=np.complex128)
    prev_d = None
    for ii, rr in enumerate(r):
        _, dHdR, evals, evecs = evaluate_hamiltonian(t, rr, hamiltonian=model)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        d = d[np.newaxis, :, :]
        if prev_d is not None:
            d = nac_phase_following(prev_d, d)
            model.update_last_deriv_couplings(d)
        prev_d = d
        E_out[ii, :] = evals
        F_out[ii, :] = F
        d_out[ii, :, :] = d
    return E_out, F_out, d_out

def _plot_tullyone_hamiltonian(r, E, F, center_focus=False):
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')

    fig = plt.figure(figsize=(3*2, 2), dpi=300)
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots().flatten()

    _c = E.shape[1]//2
    center_indices = [_c-1, _c]

    # plot the eigen energies
    ax = axs[0]
    for ii in range(E.shape[1]):
        if center_focus and (ii not in center_indices):
            continue
        ax.plot(r, E[:, ii], label=f"E{ii}")
    ax.set_xlabel("R")
    ax.set_ylabel("Eigen Energies")

    # plot the adiabatic forces
    ax = axs[1]
    for ii in range(F.shape[1]):
        if center_focus and (ii not in center_indices):
            continue
        ax.plot(r, F[:, ii], label=f"F{ii}")
    ax.set_xlabel("R")
    ax.set_ylabel("Adiabatic Forces")
    for ax in axs.flatten():
        ax.legend()
        ax.set_xlim(-5, 5)

    fig.tight_layout()
    plt.show()

def _plot_tullyone_deriv_cpouplings(r, d):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3**2, 2), dpi=300)
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots().flatten()

    # abs value of the non-adiabatic couplings
    ax = axs[0]
    for ii in range(d.shape[1]):
        for jj in range(ii+1, d.shape[2]):
            dij_abs = np.abs(d[:, ii, jj])
            if np.max(np.abs(dij_abs)) < 1e-5:
                continue
            ax.plot(r, np.abs(d[:, ii, jj]), label=f"d{ii}{jj}")

    # real part of the non-adiabatic couplings
    ax = axs[1]
    for ii in range(d.shape[1]):
        for jj in range(ii+1, d.shape[2]):
            dij_re = np.real(d[:, ii, jj])
            if np.max(np.abs(dij_re)) < 1e-5:
                continue
            ax.plot(r, dij_re, label=f"Re(d{ii}{jj})")

    # imaginary part of the non-adiabatic couplings
    ax = axs[2]
    for ii in range(d.shape[1]):
        for jj in range(ii+1, d.shape[2]):
            dij_im = np.imag(d[:, ii, jj])
            if np.max(np.abs(dij_im)) < 1e-5:
                continue
            ax.plot(r, dij_im, label=f"Im(d{ii}{jj})")

    for ax in axs.flatten():
        ax.legend()
        ax.set_xlim(-5, 5)
        ax.set_xlabel("R")
        ax.set_ylabel("Non-adiabatic Couplings")
    plt.show()

def _test_tullyone():
    hamiltonian = TullyOne()
    r = np.linspace(-10, 10, 1000)
    E, F, d = _evaluate_tullyone_hamiltonian(0, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
    _plot_tullyone_deriv_cpouplings(r, d)

def _test_tullyone_pulsed(pulse_type: TullyOnePulseTypes):
    Omega = 0.3
    tau = 100
    t0 = 0
    hamiltonian = get_tullyone(
        t0=t0, Omega=Omega, tau=tau,
        pulse_type=pulse_type,
        td_method=TD_Methods.BRUTE_FORCE
    )
    r = np.linspace(-10, 10, 1000)
    t = 0
    E, F, d= _evaluate_tullyone_hamiltonian(t, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
    _plot_tullyone_deriv_cpouplings(r, d)

def _test_tullyone_floquet(pulse_type: TullyOnePulseTypes, Omega: float=0.3, tau: float=100, t0: float=10, NF: int=1):
    hamiltonian = get_tullyone(
        t0=t0, Omega=Omega, tau=tau,
        pulse_type=pulse_type,
        td_method=TD_Methods.FLOQUET,
        NF=NF
    )
    r = np.linspace(-10, 10, 1000)
    t = t0
    # t = 0.0
    E, F, d = _evaluate_tullyone_floquet_hamiltonian(t, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
    _plot_tullyone_deriv_cpouplings(r, d)
    # print(f"{hamiltonian.last_evecs=}")



# %%
if __name__ == "__main__":
    # test the time-independent TullyOne model
    # print("=====================================================")
    # print("Testing the time-independent TullyOne model")
    # _test_tullyone()
    # print("=====================================================")
    # test the time-dependent TullyOne model with different pulse types
    # print("=====================================================")
    # print("Testing the TullyOne with PulseType1")
    # _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE1)
    # print("=====================================================")
    # print("=====================================================")
    # print("Testing the TullyOne with PulseType2")
    # _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE2)
    # print("=====================================================")
    # print("=====================================================")
    # print("Testing the TullyOne with PulseType3")
    # _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE3)
    # print("=====================================================")

    # test the time-dependent Floquet TullyOne model with different pulse types
    Omega = 0.03
    tau = 100
    t0 = 10
    NF = 1
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType1")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE1, Omega=Omega, tau=tau, t0=t0, NF=NF)
    print("=====================================================")
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType2")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE2, Omega=Omega, tau=tau, t0=t0, NF=NF) 
    print("=====================================================")
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType3")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE3, Omega=Omega, tau=tau, t0=t0, NF=NF)
    print("=====================================================")

# %%
