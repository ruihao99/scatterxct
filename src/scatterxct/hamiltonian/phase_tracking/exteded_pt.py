# %%
"""Python implementation of the extended parallel transport algorithm 

Original author: Zeyu Zhou and Gaohan Miao

Paper:
    Zeyu Zhou, Zuxin Jin, Tian Qiu, Andrew M. Rappe, and Joseph Eli Subotnik
    JCTC 2020 16 (2), 835-846. DOI: 10.1021/acs.jctc.9b00952
    
Original c++ code:
    https://github.com/Eplistical/FSSHND/blob/master/src/adjust_evt_phase.cpp
"""

import numpy as np
from scipy.linalg import schur
from numba import njit
from warnings import warn

@njit
def clip(x, a, b):
    return np.maximum(a, np.minimum(b, x))


def rotation_angle(AA, BB, CC, DD):
    """Original c++ documentation:
        /*This is to minimize a cos(2 \theta) + b sin(2 \theta) + c cos(\theta) + d sin(\theta) 
         *	    by calculating the roots of the first derivative function, i.e.
         *	    -2a sin(2 \theta) + 2b cos(2 \theta) - c sin(\theta) + d cos(\theta) = 0
         *	  This is equivalent to solving a quartic equation (4th degree algebraic equation)
         *	    x^4 + u x^3 + v x^2 + p x + q = 0
         *	  by diagonalizing its companion matrix and the eigenvalues (lapack routine "zgees") are the roots.
         *	    		|   0   0   0   -q   |
         *	    		|   		     |
         *	    		|   1   0   0   -p   |
         *	          C  =  |		     |
         *	    		|   0   1   0   -v   |
         *	    		|   		     |
         *	    		|   0   0   1   -u   |
         *	    Note that |x| = cos(\theta), and it must be real. So we
         *	    1. Calculate {u, v, p, q} by {a, b, c, d}
         *	    2. Diagonalize the companion matrix C and obtain 4 roots.
         *	    3. Drop the complex roots  and save the real ones.
         *	    4. Calculate arccos(roots) and -arccos(roots)
         *	    5. See if they are actually the root of the first derivative function.
         *	    6. See which of them has the lowest value.
         *
         * */
    """
    # parameters
    TOL_X4 = 1e-9
     
    a = 2.0 * BB
    b = -2.0 * AA
    c = DD
    d = -CC
    x4 = 4 * (a**2 + b**2)
    
    # only one trivial crossing 
    if np.abs(x4) < TOL_X4:
        theta = np.arctan(-c / d)
        if np.isnan(theta):
            warn(
                """ Tricky event happened! """
                """ a = 0, b = 0, c = 0, d = 0 """
                """ Return a large phase factor as Zeyu did"""
            )
            return 40.0
            
        elif ((c * np.sin(theta) - d * np.cos(theta)) < 0):
            return theta
        else:
            return theta + np.pi
    
    # multiple trivial crossings
    u = 4.0 * (a * c + b * d) / x4
    v = (c * c - 4.0 * a * a + d * d - 4.0 * b * b) / x4
    p = -(4.0 *  b * d + 2*a * c) / x4
    q = (a * a - d * d) / x4
    
    # companion matrix
    C = np.array([
        [0, 0, 0, -q],
        [1, 0, 0, -p],
        [0, 1, 0, -v],
        [0, 0, 1, -u]
    ], dtype=np.complex128)
    
    # schur decomposition
    try:
        T, Z = schur(C, output="complex")
    except ValueError:
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"c = {c}")
        print(f"d = {d}")
        print(f"AA = {AA}")
        print(f"BB = {BB}")
        print(f"CC = {CC}")
        print(f"DD = {DD}")
        
        raise ValueError(
            """There is a problem!"""
        )
    
    # compute the rotation angle
    return _rotation_angle(a, b, c, d, AA, BB, CC, DD, T)
    
@njit
def _rotation_angle(a, b, c, d, AA, BB, CC, DD, T):
    # lapack routine "zgees" overwrites A,
    # so the scipy T is the same as the A in the original c++ code
    A = T
    flag_firstroot = True
    for i in range(4):
        if np.abs(A[i, i].imag) > 1e-5:
            continue
        else:
            if A[i, i].real > 1.0000010 or A[i, i].real < -1.000000100:
                continue
            prop_root = np.arccos(clip(A[i, i].real, -1.0, 1.0))
            func = a * np.cos(2.0 * prop_root) + b * np.sin(2.0 * prop_root) + c * np.cos(prop_root) + d * np.sin(prop_root)
            if np.abs(func) < 0.001:
                if flag_firstroot:
                    min_theta = prop_root
                    min_val = AA * np.cos(2.0 * prop_root) + BB * np.sin(2.0 * prop_root) + CC * np.cos(prop_root) + DD * np.sin(prop_root) 
                    flag_firstroot = False
                else:
                    func = AA * np.cos(2.0 * prop_root) + BB * np.sin(2.0 * prop_root) + CC * np.cos(prop_root) + DD * np.sin(prop_root)
                    if func < min_val:
                        min_theta = prop_root
                        min_val = func
            else:
                prop_root = -prop_root
                func = a * np.cos(2.0 * prop_root) + b * np.sin(2.0 * prop_root) + c * np.cos(prop_root) + d * np.sin(prop_root)
                if np.abs(func) < 0.001:
                    if flag_firstroot:
                        min_theta = prop_root
                        min_val = AA * np.cos(2.0 * prop_root) + BB * np.sin(2.0 * prop_root) + CC * np.cos(prop_root) + DD * np.sin(prop_root) 
                        flag_firstroot = False
                    else:
                        func = AA * np.cos(2.0 * prop_root) + BB * np.sin(2.0 * prop_root) + CC * np.cos(prop_root) + DD * np.sin(prop_root)
                        if func < min_val:
                            min_theta = prop_root
                            min_val = func
                else:
                    raise ValueError(
                        """There is a problem!"""
                    )
    if not flag_firstroot:
        return min_theta
    else:
        raise ValueError(""" There is a problem 2! """)
        return 40.0
        

@njit
def functiontominimize(Mat):
    """The original c++
        /* This is the function to minimize */
    """
    N = int(Mat.size**0.5)
    functional = 0.0
    for ii in range(N):
        for jj in range(N):
            # functional += std::real(Mat[ii + jj * N] * Mat[jj + ii * N]);
            functional += np.real(Mat[ii, jj] * Mat[jj, ii])    
        # functional += - 16.0 / 3.0 * std::real(Mat[ii * (N + 1)]);
        functional += - 16.0 / 3.0 * np.real(Mat[ii, ii])
    return functional

@njit
def check_pt_is_good_enough(U):
    N = int(U.size**0.5)
    if N <= 2:
        return True
    
    THRESHOLD = 1.0 - 2.0 / N
    for i in range(N):
        if np.abs(U[i, i]) < THRESHOLD:
            return False
    return True

@njit 
def PT_simple(curevt, nextevt):
    # get the dimension
    dim = curevt.shape[0]
    
    # compute the overlap matrix
    S = np.dot(curevt.T.conj(), nextevt)
    
    # compute the absolute value of the overlap matrix
    abs_S = np.abs(S)
    
    # allocate the corrected nextevt
    tempnextevt = np.zeros_like(nextevt)
    
    # compute the phase correction and apply it 
    # to the columns of the eigenvectors 
    for ii in range(dim):
        max_col = abs_S[ii, 0]
        max_row_idx = 0
        for jj in range(1, dim):
            if abs_S[ii, jj] > max_col:
                max_col = abs_S[ii, jj]
                max_row_idx = jj
        
        for jj in range(dim):
            tempnextevt[jj, ii] = nextevt[jj, ii] * abs_S[max_row_idx, ii] / S[max_row_idx, ii]
    
    # re-evaluate the overlap matrix
    S = np.dot(curevt.T.conj(), tempnextevt)
    
    return S, tempnextevt


def parallel_transport(
    curevt,
    nextevt,
    max_depth: int = 100
) -> np.ndarray:
    """Original c++ documentation:
        /* This function calculates the parallel transport matrix */
    """
    
    # Parameters
    maximumvalue_column = 0.0
    flagtrivialcrossing = False
    
    # // --- parallel transport --- //
    
    # dimension
    N = curevt.shape[0]
    
    # conventional version of parallel transport
    UUU, tempnextevt = PT_simple(curevt, nextevt)    
    
    # // --- make sure det(U) = 1 --- //
    detU = np.linalg.det(UUU)
    
    if (np.abs(np.abs(detU) - 1.0) > 0.0001):
        raise ValueError(
            """ Determinant of U is not 1!"""
        )
        
    # //change first column to make det(UUU) = +1!//
    tempnextevt[:, 0] /= detU
    UUU[:, 0] /= detU
    
    # // --- second order approximation --- //
    if not check_pt_is_good_enough(UUU):
        # // parallel transport is not good enough
        # /* Minimize f(theta) = a cos(theta) + b sin(theta), find the two coefficients {a, b} as denoted {coscoef, sincoef} */
        minimumvalue = functiontominimize(UUU)
        tempUUU = UUU.copy()     
        depth = 0
        while True:
            for inumtc in range(N):
                for jnumtc in range(inumtc + 1, N):
                    coscoef = 0.0
                    sincoef = 0.0
                    cos2coef = 0.0
                    sin2coef = 0.0
                    for ii in range(N):
                        #  coscoef += std::real(tempUUU[ii + inumtc * NNN] * tempUUU[ii * NNN + inumtc] + tempUUU[ii + jnumtc * NNN] * tempUUU[ii * NNN + jnumtc]);
                        # sincoef += -std::imag(tempUUU[ii + inumtc * NNN] * tempUUU[ii * NNN + inumtc] - tempUUU[ii + jnumtc * NNN] * tempUUU[ii * NNN + jnumtc]);
                        coscoef += np.real(tempUUU[inumtc, ii] * tempUUU[ii, inumtc] + tempUUU[jnumtc, ii] * tempUUU[ii, jnumtc])   
                        sincoef += -np.imag(tempUUU[inumtc, ii] * tempUUU[ii, inumtc] - tempUUU[jnumtc, ii] * tempUUU[ii, jnumtc])

                    # coscoef -= std::real(tempUUU[jnumtc + inumtc * NNN] * tempUUU[inumtc + jnumtc * NNN]) * 2 + std::real(tempUUU[inumtc * (NNN + 1)] + tempUUU[jnumtc * (NNN + 1)]) * 8.0 / 3.0 + std::real(tempUUU[inumtc * (NNN + 1)] * tempUUU[inumtc * (NNN + 1)] + tempUUU[jnumtc * (NNN + 1)] * tempUUU[jnumtc * (NNN + 1)]);
                    # sincoef -= 8.0 / 3.0 * std::imag(tempUUU[jnumtc * (NNN + 1)] - tempUUU[inumtc * (NNN + 1)]) + std::imag(tempUUU[jnumtc * (NNN + 1)] * tempUUU[jnumtc * (NNN + 1)] - tempUUU[inumtc * (NNN + 1)] * tempUUU[inumtc * (NNN + 1)]);
                    # cos2coef = std::real(tempUUU[inumtc * (NNN + 1)] * tempUUU[inumtc * (NNN + 1)] + tempUUU[jnumtc * (NNN + 1)] * tempUUU[jnumtc * (NNN + 1)]) * 0.50;
                    # sin2coef = std::imag(tempUUU[jnumtc * (NNN + 1)] * tempUUU[jnumtc * (NNN + 1)] - tempUUU[inumtc * (NNN + 1)] * tempUUU[inumtc * (NNN + 1)]) * 0.50;
                    # theta = rotation_angle(cos2coef, sin2coef, coscoef, sincoef);
                    coscoef -= np.real(tempUUU[inumtc, jnumtc] * tempUUU[jnumtc, inumtc]) * 2 + np.real(tempUUU[inumtc, inumtc] + tempUUU[jnumtc, jnumtc]) * 8.0 / 3.0 + np.real(tempUUU[inumtc, inumtc] * tempUUU[inumtc, inumtc] + tempUUU[jnumtc, jnumtc] * tempUUU[jnumtc, jnumtc])
                    sincoef -= 8.0 / 3.0 * np.imag(tempUUU[jnumtc, jnumtc] - tempUUU[inumtc, inumtc]) + np.imag(tempUUU[jnumtc, jnumtc] * tempUUU[jnumtc, jnumtc] - tempUUU[inumtc, inumtc] * tempUUU[inumtc, inumtc])
                    cos2coef = np.real(tempUUU[inumtc, inumtc] * tempUUU[inumtc, inumtc] + tempUUU[jnumtc, jnumtc] * tempUUU[jnumtc, jnumtc]) * 0.50
                    sin2coef = np.imag(tempUUU[jnumtc, jnumtc] * tempUUU[jnumtc, jnumtc] - tempUUU[inumtc, inumtc] * tempUUU[inumtc, inumtc]) * 0.50
                    theta = rotation_angle(cos2coef, sin2coef, coscoef, sincoef)

                    tempUUU[inumtc, :] *= np.exp(1j * theta)
                    tempUUU[jnumtc, :] *= np.exp(-1j * theta)
                    tempnextevt[inumtc, :] *= np.exp(1j * theta)
                    tempnextevt[jnumtc, :] *= np.exp(-1j * theta)
            tempminvalue = functiontominimize(tempUUU)
            if (abs(minimumvalue - tempminvalue) < 0.00001):
                break
            elif (depth > max_depth):
                warn(
                    """ Maximum depth reached! get of the parallel transport loop """
                )
                break
            else:
                minimumvalue = tempminvalue
                depth += 1
    # print(f"breaking out of the loop at depth {depth}")
            
    nextevt = tempnextevt
    return nextevt
    
     

# %%
