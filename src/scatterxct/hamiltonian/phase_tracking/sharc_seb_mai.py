# %%
import numpy as np
from numba import njit

import warnings 

from scatterxct.hamiltonian.linalg import lowdin

@njit
def reorder_S(S, evals, overlap_threshold=0.5, digeneracy_threshold=1e-12):
    S_reorder = np.copy(S)
    evals_reorder = np.copy(evals)
    dim = S.shape[0]
    
    
    for i in range(dim):
        for j in range(dim):
            if (abs(S_reorder[i, j])**2 > overlap_threshold):
                if (abs(evals_reorder[i] - evals_reorder[j]) < digeneracy_threshold):
                    # do not reorder the columns if the states are degenerate
                    break
                elif (i == j):
                    # do not reorder the columns if the states are the same
                    break
                elif (i < j):
                    # reorder the columns to the right
                    # (i-1, i, ..., j-1, j) -> (i-1, j, i, ..., j-1)
                    S_reorder[:, i] = S[:, j]
                    S_reorder[:, i+1] = S[:, i]
                    S_reorder[:, i+2:j+1] = S[:, i+1:j]
                    evals_reorder[i] = evals[j]
                    evals_reorder[i+1] = evals[i]
                    evals_reorder[i+2:j+1] = evals[i+1:j]
                elif (i > j):
                    # reorder the columns to the left
                    # (j-1, j, j+1, ..., i-1, i) -> (j-1, j+1, ..., i-1, i, j)
                    S_reorder[:,i] = S[:, j]
                    S_reorder[:,i-1] = S[:, i]
                    S_reorder[:,j:i-1] = S[:, j+1:i]
                    evals_reorder[i] = evals[j]
                    evals_reorder[i-1] = evals[i]
                    evals_reorder[j:i-1] = evals[j+1:i]
                break
            
    has_reorder = (evals_reorder != evals).any()
        
    return S_reorder, evals_reorder, has_reorder

@njit   
def compute_P(S, E, degen_threshold=1e-12, null_projection_threshold=1e-9):   
    mask = np.zeros_like(S, dtype=np.int8)
    P = np.zeros_like(S)
    dim = S.shape[0]
    
    # determine the degenerate states
    for i in range(dim):
        for j in range(dim):
            if (abs(E[i] - E[j]) < degen_threshold):
                mask[i, j] = 1
                P[i, j] = S[i, j]
                
    # non-degenerate states
    for i in range(dim):
        sums = 0.0
        for j in range(dim):
            sums += abs(P[j, i])**2
        
        if sums < null_projection_threshold:
            P[:, i] = 0.0
            for j in range(dim):
                if mask[j,i] == 1:
                    sums = 0.0
                    for k in range(dim):
                        sums += abs(P[j, k])**2
                    if sums < null_projection_threshold:
                        P[j, i] = 1.0
                        break   # break the j loop
                    
    return P

def reorder_eignstates(U, E_reorder):
    # E_reorder is a permutation of E
    inv_argsort = np.argsort(np.argsort(E_reorder))
    return U[:, inv_argsort]


def mai_projection(E, U, U_last):
    # calculate the overlap matrix
    # S = matmatmul(U, U_last, 'hn')
    S = np.dot(U.T.conj(), U_last)
     
    # reorder the overlap matrix and eigenvalues
    S_reorder, E_phase_track, has_reorder = reorder_S(S, E)
 
    # calculate the projection matrix
    P = compute_P(S_reorder, E_phase_track)
    
    # check for NaNs in P
    if np.isnan(P).any():
        print(f"NaNs in P before lowdin")
        print(f"P: {P}")
        print(f"H: {H}")
        print(f"U: {U}")
        print(f"U_last: {U_last}")
        
        
    # orthogonalize P
    P = lowdin(P)
    
    # check for NaNs in P
    if np.isnan(P).any():
        warnings.warn('NaNs in P. Resetting phase tracking.')   
        print('P: ', P)
        print('H: ', H) 
        print('U: ', U)
        print('U_last: ', U_last)   
        P = np.eye(P.shape[0], dtype=P.dtype)
            
    # project the eigenstates
    # U_projected = matmatmul(U, P, 'nn')
    U_projected = np.dot(U, P)  
    return U_projected
    

def recursive_project(H, H_old, U_old, dt):
    # parameters
    N_SUBSTEPS = 10
    diagthrs = 1E-3
    mindtsubstep = 1E-3
    dt_reduce = 0.1
    
    dt_substep = dt / N_SUBSTEPS 
    U_old_dummy = U_old.copy()
    
    dim = H.shape[0]
    
    for istep in range(N_SUBSTEPS):
        H_substep = H_old + (H - H_old) * (istep + 1) / N_SUBSTEPS
        E, U = np.linalg.eigh(H_substep)
        U = mai_projection(E, U, U_old_dummy)
        
        dU = (U - U_old) / dt_substep
        UdU = np.dot(U.T.conj(), dU)
        sub = False
        for i in range(dim):
            if (abs(UdU[i, i]) > diagthrs):
                sub = True
                break
            
        if sub and (dt_substep * dt_reduce > mindtsubstep):
            dt_substep *= dt_reduce
            H = H_old + (H - H_old) * (istep + 1) / N_SUBSTEPS
            H_old = H_old + (H - H_old) * istep / N_SUBSTEPS
            return recursive_project(H, H_old, U_old, dt_substep)
        else:
            U_old_dummy = U.copy()
            
    return E, U
