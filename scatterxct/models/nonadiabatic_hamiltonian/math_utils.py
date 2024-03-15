# %%
import numpy as np
# import scipy.linalg as LA
import numpy.linalg as LA
from numba import jit
import scipy.sparse as sp
from numpy.typing import ArrayLike

from typing import Tuple, Union, Optional

def align_phase(prev_evecs: ArrayLike, curr_evecs: ArrayLike) -> ArrayLike:
    """Algorithm to align the phase of eigen vectors, originally proposed
    by Graham (Gaohan) Miao and Zeyu Zhou. (the Hamiltonian is changed adiabatically)

    Args:
        prev_evecs (ArrayLike): the eigen vectors of the previous iteration
        curr_evecs (ArrayLike): the eigen vectors of the current iteration

    Returns:
        ArrayLike: the phased-corrected eigen vectors

    Reference:
        [1]. `FSSHND` by Graham (Gaohan) Miao. git@github.com:Eplistical/FSSHND.git.
             See the 'Hamiltonian::cal_info()' function in 'src/hamiltonian.cpp'
    """
    tmp = np.dot(prev_evecs.conjugate().T, curr_evecs)
    diag_tmp = np.diag(tmp) 
    phase_factors = np.ones_like(diag_tmp) 
    mask = np.logical_not(np.isclose(diag_tmp, 0))
    phase_factors[mask] = diag_tmp[mask] / np.abs(diag_tmp[mask])
    aligned_evecs = curr_evecs / phase_factors
    return aligned_evecs

def nac_phase_following(d_prev: ArrayLike, d_curr: ArrayLike) -> ArrayLike:
    """A wrapper function for numba routine '_nac_phase_following' to calculate the
    phase-corrected non-adiabatic couplings (NACs).

    Args:
        d_prev (ArrayLike): NACs of the previous iteration
        d_curr (ArrayLike): NACs of the current iteration

    Returns:
        ArrayLike: the phase-corrected NACs
    """
    # print("nac_phase_following is called")
    dim_classical = d_curr.shape[0]
    dim_electrnic = d_curr.shape[1]
    d_corrected = np.zeros((dim_classical, dim_electrnic, dim_electrnic), dtype=d_curr.dtype)
    try:
        assert d_prev.shape == d_curr.shape == d_corrected.shape
    except AssertionError:
        raise ValueError(
            f"""Shape mismatch: {d_prev.shape=}, {d_curr.shape=}, {d_corrected.shape=}."""
            """This function REQUIRES the NACs array has the shape (N, M, M), where N  """
            """denotes the dimension of the classical variable; M denotes the dimension"""
            """of the quantum system."""
        )
    # return _nac_phase_following(d_prev, d_curr, d_corrected)
    return adjust_nac_phase(d_prev, d_curr, d_corrected)

def _nac_phase_following(d_prev: ArrayLike, d_curr: ArrayLike, d_corr: ArrayLike) -> ArrayLike:
    """The phase following algorithm for the non-adiabatic couplings (NAC) by
    the JADE-NAMD package developers.

    * Note in this implementation, the derivative coupling is a 3D array.
    The first dimension is the number of classical degrees of freedom, i.e.,
    the number of nuclear configurations (normal modes or Cartesian coordinates).
    The second and third dimensions are the number of electronic states.
    That is to say, the input d_prev and d_curr have the shape of
    (dim_classical, dim_electrnic, dim_electrnic).
    
    * Note that I have tried to use the `numba` to speed up the calculation.
    * However, numba yields a faulty result. Use pure numpy at the moment.


    Args:
        d_prev (ArrayLike): NACs of the previous iteration
        d_curr (ArrayLike): NACs of the current iteration
        d_corr (ArrayLike): the phase-corrected NACs

    Returns:
        ArrayLike: the phase-corrected NACs

    Reference:
        [1]. The JADE-NAMD pacakge. git@github.com:zglan/JADE-NAMD.git
        see the 'sub_nac_phase' routine in 'src/dynamics/tsh/sub_nac_phase.f90'.
    """
    ATOL = 1e-8
    RTOL = 1e-5
    for jj in range(d_corr.shape[1]):
        for kk in range(d_corr.shape[1]):
            if jj == kk:
                pass
            # calculate the norm of the NACs
            prev_nac_norm = LA.norm(d_prev[:, jj, kk])
            curr_nac_norm = LA.norm(d_curr[:, jj, kk])
            # calculate the dot product of the NACs
            nac_dot_product = np.dot(d_prev[:, jj, kk], d_curr[:, jj, kk])
            # if (prev_nac_norm == 0) or (curr_nac_norm == 0):
            if np.isclose(prev_nac_norm, 0, atol=ATOL, rtol=RTOL) or np.isclose(curr_nac_norm, 0, atol=ATOL, rtol=RTOL):
                # phase_factor = 1.0
                cos_angle: float = 1.0
            else:
                cos_angle: float = nac_dot_product / (prev_nac_norm * curr_nac_norm)
                # phase_factor = nac_dot_product / (prev_nac_norm * curr_nac_norm)
            # correct the phase of the NACs at (jj, kk) electronic indices using the phase factor
            # d_corr[:, jj, kk] = d_curr[:, jj, kk] / phase_factor
            d_corr[:, jj, kk] = d_curr[:, jj, kk] * np.sign(cos_angle)
    return d_corr

def adjust_nac_phase(d_prev: ArrayLike, d_curr: ArrayLike, d_corr: ArrayLike) -> ArrayLike:
    # print(f"{d_prev.shape=}", f"{d_curr.shape=}") 
    for ii in range(d_curr.shape[1]):
        for jj in range(ii+1, d_curr.shape[2]):
            ovlp: float = 0.0
            snac_old: float = 0.0
            snac: float = 0.0
            
            # snac_old = np.sqrt(np.sum(d_prev[:, ii, jj]**2))
            # snac = np.sqrt(np.sum(d_curr[:, ii, jj]**2))
            snac_old = LA.norm(d_prev[:, ii, jj])
            snac = LA.norm(d_curr[:, ii, jj])
            
            if (np.sqrt(snac_old * snac) < 1e-8):
                ovlp: float = 1.0
            else:
                dot_nac: float = 0.0
                dot_nac = np.sum(d_prev[:, ii, jj].conjugate() * d_curr[:, ii, jj])
                ovlp = dot_nac / (snac_old * snac)
                
            d_corr[:, ii, jj] = d_curr[:, ii, jj] / ovlp
            d_corr[:, jj, ii] = d_curr[:, jj, ii] / ovlp
    return d_corr
                
            
    

def diagonalize_hamiltonian_history(hamiltonian: ArrayLike, prev_evecs: Optional[ArrayLike]=None) -> Tuple[ArrayLike, ArrayLike]:
    evals, evecs = LA.eigh(hamiltonian.toarray()) if sp.isspmatrix(hamiltonian) else LA.eigh(hamiltonian)
    if prev_evecs is not None:
        evecs = align_phase(prev_evecs, evecs) #if prev_evecs is not None else evecs
    # else:
    #     phases = np.diagonal(evecs.conjugate().T @ evecs)
    #     phases = phases / np.abs(phases)
    #     over_all_sign = np.sign(LA.det(evecs))
    #     evecs = evecs * over_all_sign / phases
    # else:
        
        # signs = np.sign(np.diagonal(evecs))
        # print(f"{signs=}")
        # evecs = evecs * signs
            
    return evals, evecs

def diagonalize_2d_real_symmetric(H: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    a = H[0, 0]
    b = H[1, 1]
    c = H[0, 1]

    lambda1 = 0.5 * (a + b - np.sqrt((a - b)**2 + 4 * c**2))
    lambda2 = 0.5 * (a + b + np.sqrt((a - b)**2 + 4 * c**2))

    evals = np.array([lambda1, lambda2])
    theta = np.arctan2(2 * c, b - a) / 2
    evecs = np.array(
        [[ np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    )
    return evals, evecs

def diabatic_to_adiabatic(
    O: ArrayLike,
    U: ArrayLike,
    out: Union[ArrayLike, None]=None
)-> ArrayLike:
    if out is not None:
        np.dot(O, U, out=out)
        np.dot(U.conjugate().T, out, out=out)
    else:
        return np.dot(U.conjugate().T, np.dot(O, U))

def adiabatic_to_diabatic(
    O: ArrayLike,
    U: ArrayLike,
    out: Union[ArrayLike, None]=None
) -> ArrayLike:
    if out is not None:
        np.dot(U, O, out=out)
        np.dot(out, U.conjugate().T, out=out)
    else:
        return np.dot(U, np.dot(O, U.conjugate().T))

# %%
def _evaluate_tullyone_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
    dim_elc = model.dim
    dim_cls = r.size
    E_out = np.zeros((dim_cls, dim_elc))
    F_out = np.zeros((dim_cls, dim_elc))
    d_out = np.zeros((dim_cls, dim_elc, dim_elc), dtype=np.complex128)
    for ii, rr in enumerate(r):
        H = model.H(t, rr)
        dHdR = model.dHdR(t, rr)
        evals, evecs = diagonalize_hamiltonian_history(H, model.last_evecs)
        model.update_last_evecs(evecs)

        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        E_out[ii, :] = evals
        F_out[ii, :] = F
        d_out[ii, :, :] = d
    return E_out, F_out, d_out

def _evaluate_tullyone_floquet_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
    dim_elc = model.dim
    NF = model.NF
    dim_F = dim_elc*(NF*2+1)
    E_out = np.zeros((len(r), dim_F))
    F_out = np.zeros((len(r), dim_F))
    d_out = np.zeros((len(r), dim_F, dim_F), dtype=np.complex128)
    d_last = None
    for ii, rr in enumerate(r):
        # R = np.array([rr])
        H = model.H(t, rr)
        dHdR = model.dHdR(t, rr)
        evals, evecs = diagonalize_hamiltonian_history(H, model.last_evecs)
        model.update_last_evecs(evecs)
        # print(f"{np.linalg.det(evecs)=}")
        # evals, evecs = diagonalize_hamiltonian_history(H, model.last_evecs)
        # model.update_last_evecs(evecs)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        d = d[np.newaxis, :, :]
        d = nac_phase_following(d_last, d) if d_last is not None else d
        # d = adjust_nac_phase(d_last, d) if d_last is not None else d
        d_last = d
        # print(f"{d_last.dtype=}", f"{d.dtype=}")
        E_out[ii, :] = evals
        F_out[ii, :] = F
        d_out[ii, :, :] = d
    return E_out, F_out, d_out

def _plot_tullyone_hamiltonian(r, E, F, d_out, center_focus=True):
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

    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(3*3, 2), dpi=300)
    gs = fig.add_gridspec(1, 3)
    axs = gs.subplots().flatten()
    # plot the abs value of the nonadiabatic couplings
    ax = axs[0]
    for ii in range(d_out.shape[1]):
        for jj in range(ii+1, d_out.shape[2]):
            if np.abs(d_out[:, ii, jj]).max() < 1e-3:
                continue
            # ax.plot(r, np.abs(d_out[:, ii, jj]), label=f"abs(d{ii}{jj})")
            ax.plot(r, d_out[:, ii, jj].real, label=f"Re(d{ii}{jj})", alpha=.9)
            # ax.plot(r, d_out[:, ii, jj].imag, label=f"Im(d{ii}{jj})", alpha=.9)

    ax = axs[1]
    for ii in range(d_out.shape[1]):
        for jj in range(ii+1, d_out.shape[2]):
            if np.abs(d_out[:, ii, jj]).max() < 1e-3:
                continue
            ax.plot(r, d_out[:, ii, jj].imag, label=f"Im(d{ii}{jj})", alpha=.9)
    ax = axs[2]
    for ii in range(d_out.shape[1]):
        for jj in range(ii+1, d_out.shape[2]):
            if np.abs(d_out[:, ii, jj]).max() < 1e-3:
                continue
            ax.plot(r, np.abs(d_out[:, ii, jj]), label=f"abs(d{ii}{jj})")
    for ax in axs:
        ax.set_xlabel("R")
        ax.set_ylabel("Nonadiabatic Couplings")
        ax.legend()

    fig.tight_layout()

def _test_enforce_gauge_main(pulse_type):
    """ The original tullyone does not have the gauge enforcement issue """
    """ The pulsed tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE1 doesn't have the gauge enforcement issue"""
    """ The pulsed tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE2 doesn't have the gauge enforcement issue"""
    """ The pulsed tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE3 doesn't have the gauge enforcement issue"""
    """ The floquet tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE3 doesn't have the gauge enforcement issue"""
    """ *** Gauge enforcement fill cause force / to drop off after some point for *** """
    """ *** for floquet tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE2 *** """
    """ *** for floquet tullyone model with pulse_type=TullyOnePulseTypes.PULSE_TYPE1 *** """
    NF = 1
    Omega = 0.1

    t0 = 10.

    h_tullyone = get_tullyone(
        t0=t0, Omega=Omega, tau=120,
        pulse_type=pulse_type,
        td_method=TD_Methods.FLOQUET, NF=NF
    )
    # from negative to positive

    r = np.linspace(-5, 5, 1000)
    # E, F, d= _evaluate_tullyone_hamiltonian(0, r, h_tullyone)
    t = t0
    # E, F, d= _evaluate_tullyone_hamiltonian(0, r, h_tullyone)
    E, F, d = _evaluate_tullyone_floquet_hamiltonian(t, r, h_tullyone)
    _plot_tullyone_hamiltonian(r, E, F, d)

    h_tullyone = get_tullyone(
        t0=t0, Omega=Omega, tau=120,
        pulse_type=pulse_type,
        td_method=TD_Methods.FLOQUET, NF=NF
    )

    # from positive to negative
    r = np.flip(r)
    # E, F, d= _evaluate_tullyone_hamiltonian(0, r, h_tullyone)
    E, F, d = _evaluate_tullyone_floquet_hamiltonian(t, r, h_tullyone)
    _plot_tullyone_hamiltonian(r, E, F, d)


# %%