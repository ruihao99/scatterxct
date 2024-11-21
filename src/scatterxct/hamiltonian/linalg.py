# %%
import numpy as np
from numba import njit
import warnings 

# public functions
__all__ = [
    'matmatmul',
    'u_o_udagger',
    'udagger_o_u',
    'lowdin',
    'diagonalize_and_project',
    'hellmann_feynman',
    'compute_nac',
]

# private functions
private = [
    'reorder_S',
    'compute_P',
    'hellmann_feynman_d',
    'hellmann_feynman_z',
]

# ======
# Linear Algebra Wrappers 
# ======

def matmatmul(A, B, mode):
    if mode == 'nn':
        # normal-normal product
        return np.dot(A, B)
    elif mode == 'nh':
        # normal-hermitian_conj product
        return np.dot(A, B.T.conj())
    elif mode == 'hn':
        # hermitian_conj-normal product
        return np.dot(A.T.conj(), B)
    elif mode == 'hh':
        # hermitian_conj-hermitian_conj product
        return np.dot(A.T.conj(), B.T.conj())
    else:
        raise ValueError('Invalid mode: {}'.format(mode))
    
def u_o_udagger(O, U, inplace=False):
    if O.ndim == 2:
        return u_o_udagger_op(O, U, inplace)
    elif O.ndim == 3:
        return u_o_udagger_vecop(O, U, inplace)
    else:
        raise ValueError('Invalid dimension: {}'.format(O.ndim))
    
def udagger_o_u(O, U, inplace=False):
    if O.ndim == 2:
        return udagger_o_u_op(O, U, inplace)
    elif O.ndim == 3:
        return udagger_o_vecop(O, U, inplace)
    else:
        raise ValueError('Invalid dimension: {}'.format(O.ndim))
    
def u_o_udagger_op(O, U, inplace=False):
    """ operator transformation: U O U^dagger """
    if inplace:
        np.dot(O, U.T.conj(), out=O)
        np.dot(U, O, out=O)
    else:
        return np.dot(U, np.dot(O, U.T.conj()))

def udagger_o_u_op(O, U, inplace=False):
    """ operator transformation: U^dagger O U """
    if inplace:
        np.dot(U.T.conj(), O, out=O)
        np.dot(O, U, out=O)
    else:   
        return np.dot(U.T.conj(), np.dot(O, U))

def u_o_udagger_vecop(O, U, inplace=False):
    """ operator transformation: U O U^dagger  """
    """ for vector operators. The last axis is """
    """ the component to loop over             """
    if inplace:
        for i in range(O.shape[-1]):
            np.dot(O[:,:,i], U.T.conj(), out=O[:,:,i])
            np.dot(U, O[:,:,i], out=O[:,:,i])
    else:
        for i in range(O.shape[-1]):
            O[:,:,i] = np.dot(U, np.dot(O[:,:,i], U.T.conj()))
    return O

def udagger_o_vecop(O, U, inplace=False):
    """ operator transformation: U^dagger O U  """
    """ for vector operators. The last axis is """
    """ the component to loop over             """ 
    if inplace:
        for i in range(O.shape[-1]):
            np.dot(U.T.conj(), O[:,:,i], out=O[:,:,i])
            np.dot(O[:,:,i], U, out=O[:,:,i])
    else:
        for i in range(O.shape[-1]):
            O[:,:,i] = np.dot(U.T.conj(), np.dot(O[:,:,i], U))
    return O
    
    
def normalize_columns(A):
    return A / np.linalg.norm(A, axis=0)


def lowdin(A):
    # compute the overlap matrix
    S = matmatmul(A, A, 'hn')
    
    # diagonalize the overlap matrix
    s, U = np.linalg.eigh(S)
    
    # compute 1/sqrt(s) 
    S[:] = 0.0
    np.fill_diagonal(S, 1.0 / np.sqrt(s))
    u_o_udagger(U, S, inplace=True)
    
    # orthogonalize the matrix
    A_orth = matmatmul(A, S, 'nn')
    
    # normalize the columns
    return normalize_columns(A_orth)

@njit
def reorder_S(S, evals, overlap_threshold=0.5, digeneracy_threshold=1e-8):
    S_reorder = np.copy(S)
    evals_reorder = np.copy(evals)
    dim = S.shape[0]
    
    
    for i in range(dim):
        for j in range(dim):
            if (abs(S_reorder[i, j])**2 > overlap_threshold):
                if ((evals_reorder[i] - evals_reorder[j]) < digeneracy_threshold):
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
                
    return S_reorder, evals_reorder

@njit   
def compute_P(S, E, degen_threshold=1e-8, null_projection_threshold=1e-8):   
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
                        sums += abs(S[j, k])**2
                    if sums < null_projection_threshold:
                        P[j, i] = 1.0
                        break   # break the j loop
                    
    return P
                        

def diagonalize_and_project(H, U_last=None):
    # diagonalize the Hamiltonian
    E, U = np.linalg.eigh(H)
    
    if U_last is None:
        return E, U
    
    # calculate the overlap matrix
    S = matmatmul(U, U_last, 'hn')
    
    
    # reorder the overlap matrix and eigenvalues
    S, E = reorder_S(S, E)
    
    # calculate the projection matrix
    P = compute_P(S, E)
    
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
    U = matmatmul(U, P, 'nn')
    
    return E, U

@njit
def hellmann_feynman_d(gradH, U, E, d, gradH_adiab, nac_degen_threshold=1e-8, gradH_threshold=3e-6):  
    # compute the hellmann-feynman theorem
    # d and gradH_adiab are passed in as zeros
    dim, _, dim_nuc = gradH.shape
    tmp = np.zeros((dim, dim), dtype=np.float64)    
    tmp2 = np.zeros((dim, dim), dtype=np.float64)
    for k in range(dim_nuc):
        tmp2 = gradH[:,:,k]
        np.dot(tmp2, U, out=tmp)
        gradH_adiab[:,:,k] = np.dot(U.T, tmp)
        for j in range(dim):
            for i in range(j+1, dim):
                if (abs(E[j] - E[i]) < nac_degen_threshold):
                    if (abs(gradH_adiab[i, j, k]) < gradH_threshold):
                        # skip the degenerate states with very small NAC
                        continue
                    else:
                        continue # for now
                        # nac is not small enough to be considered 
                        # as numerical noise
                        # raise ZeroDivisionError(gradH_adiab[i, j, k])
                d[i, j, k] = gradH_adiab[i, j, k] / (E[j] - E[i])
                d[j, i, k] = -d[i, j, k]
    return d, gradH_adiab

@njit
def hellmann_feynman_z(gradH, U, E, d, gradH_adiab, nac_degen_threshold=1e-8):
    # compute the hellmann-feynman theorem
    # d and gradH_adiab are passed in as zeros
    dim, _, dim_nuc = gradH.shape
    tmp = np.zeros((dim, dim), dtype=np.complex128)
    tmp2 = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim_nuc):
        tmp2 = np.ascontiguousarray(gradH[:,:,k])
        np.dot(tmp2, U, out=tmp)
        gradH_adiab[:,:,k] = np.dot(U.T.conj(), tmp)
        for j in range(dim):
            for i in range(j+1, dim):
                d[i, j, k] = gradH_adiab[i, j, k] / (E[j] - E[i])
                d[j, i, k] = -np.conj(d[i, j, k])
    return d, gradH_adiab

def hellmann_feynman(gradH, U, E, d, gradH_adiab, nac_degen_threshold=1e-8):
    if np.iscomplexobj(gradH):
        return hellmann_feynman_z(gradH, U, E, d, gradH_adiab, nac_degen_threshold)
    else:
        return hellmann_feynman_d(gradH, U, E, d, gradH_adiab, nac_degen_threshold) 

def compute_nac(gradH, E, U, nac_degen_threshold=1e-8):
    # compute the NAC, assume the first two axis are the quantum numbers
    original_shape = gradH.shape
    dim = gradH.shape[0]
    
    # d is shapped as (dim, dim, ...)
    # ... could be any number of dimensions
    gradH_rs = gradH.reshape(dim, dim, -1)
    dim_nuc = gradH_rs.shape[2]
    
    d_rs = np.zeros_like(gradH_rs)
    gradH_adiab_rs = np.zeros_like(gradH_rs)
    
    d_rs, gradH_adiab_rs = hellmann_feynman(gradH_rs, U, E, d_rs, gradH_adiab_rs, nac_degen_threshold)
    
    d = d_rs.reshape(original_shape)
    gradH_adiab = gradH_adiab_rs.reshape(original_shape)
    return d, gradH_adiab

def psi_to_rho(psi):
    return np.outer(psi, np.conj(psi))

def compute_T_adiab(G, E, vel):
    # compute the non-adiabatic coupling matrix
    if np.iscomplexobj(G):
        return compute_T_adiab_z(G, E, vel)
    else:
        return compute_T_adiab_d(G, E, vel)

@njit
def compute_T_adiab_d(G, E, vel):
    # real-valued version of the NAC
    
    # assert nuclear dimensionality agreement
    dim, _, dim_nuc = G.shape
    assert dim_nuc == vel.shape[0]
    
    # initialize the NAC matrix
    T = np.zeros((dim, dim), dtype=np.float64)
    
    # compute the NAC via the Hellmann-Feynman theorem
    for k in range(dim_nuc):
        for j in range(dim):
            for i in range(j+1, dim):
                # if (abs(E[j] - E[i]) < 1e-8):
                #     # skip the degenerate states
                #     continue
                T[i, j] = G[i, j, k] * vel[k] / (E[j] - E[i])
                
    for j in range(dim):
        for i in range(j+1, dim):
            T[j, i] = -T[i, j]
    return T

@njit
def compute_T_adiab_z(G, E, vel):
    # complex-valued version of the NAC
    
    # assert nuclear dimensionality agreement
    dim, _, dim_nuc = G.shape
    assert dim_nuc == vel.shape[0]
    
    # initialize the NAC matrix
    T = np.zeros((dim, dim), dtype=np.complex128)
    
    # compute the NAC via the Hellmann-Feynman theorem
    for k in range(dim_nuc):
        for j in range(dim):
            for i in range(j+1, dim):
                T[i, j] += G[i, j, k] * vel[k] / (E[j] - E[i])
                
    for j in range(dim):
        for i in range(j+1, dim):
            T[j, i] = -np.conj(T[i, j])
    return T

# @njit
# def compute_T_diab_elem_d(Gdiab, U, E, vel, i, j):
#     # we can exploit that the generalized gradient is Hermitian
#     dim, _, dim_nuc = Gdiab.shape
    
#     # initialize the matrix element
#     Tij = complex(0.0, 0.0)
    
#     # compute the NAC via the Hellmann-Feynman theorem
#     for inuc in range(dim_nuc):
#         for k in range(dim):
#             for l in range(dim):
#                 Tij += 2.0 * U[k, i] * U[l, k] * Gdiab[k, l, inuc] * vel[inuc] / (E[i] - E[j])
    
#     return Tij

# @njit
# def compute_T_diab_elem_z(Gdiab, U, E, vel, i, j):
#     # we can exploit that the generalized gradient is Hermitian
#     dim, _, dim_nuc = Gdiab.shape
    
#     # initialize the matrix element
#     Tij = complex(0.0, 0.0)
    
#     # compute the NAC via the Hellmann-Feynman theorem
#     for inuc in range(dim_nuc):
#         for k in range(dim):
#             for l in range(dim):
#                 Tij += 2.0 * np.conj(U[k, i]) * U[l, k] * Gdiab[k, l, inuc] * vel[inuc] / (E[i] - E[j])
    
#     return Tij

# def compute_T_diab_elem(Gdiab, U, E, vel, i, j):
#     if np.iscomplexobj(Gdiab):
#         return compute_T_diab_elem_z(Gdiab, U, E, vel, i, j)
#     else:
#         return compute_T_diab_elem_d(Gdiab, U, E, vel, i, j)
    
    
@njit
def compute_T_diab_d(Gdiab, U, E, vel):
    # allocate the NAC matrix
    dim = Gdiab.shape[0]
    T = np.zeros((dim, dim), dtype=np.complex128)
    
    # we can exploit that the NAC matrix is anti-Hermitian
    # also T has zeros on the diagonal
    for i in range(dim):
        for j in range(1+i, dim):
            T[i, j] = compute_T_diab_elem_d(Gdiab, U, E, vel, i, j)
            T[j, i] = -T[i, j]
    return T


@njit
def compute_T_diab_z(Gdiab, U, E, vel):
    # allocate the NAC matrix
    dim = Gdiab.shape[0]
    T = np.zeros((dim, dim), dtype=np.complex128)
    
    # we can exploit that the NAC matrix is anti-Hermitian
    # also T has zeros on the diagonal
    for i in range(dim):
        for j in range(1+i, dim):
            T[i, j] = compute_T_diab_elem_z(Gdiab, U, E, vel, i, j)
            T[j, i] = -np.conj(T[i, j])
    return T    

def compute_T_diab(Gdiab, U, E, vel):
    if np.iscomplexobj(Gdiab):
        return compute_T_diab_z(Gdiab, U, E, vel)
    else:
        return compute_T_diab_d(Gdiab, U, E, vel)
    
    
    
def main():
    dim = 5
    
    H0 = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    H0 = H0+ H0.T.conj()
    
    hh = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    hh = hh + hh.T.conj()
    
    H1 = H0.copy() + hh * 0.05
    
    E0, U0= np.linalg.eigh(H0)
    # E1, U1 = np.linalg.eigh(H1) 
    E1, U1 = diagonalize_and_project(H1, U0)
    evals1, evecs1 = np.linalg.eigh(H1)
    
    print("evals: ", E0)
    print("E1: ", E1)
    
    
# %%
if __name__ == '__main__':
    main()
# %%
