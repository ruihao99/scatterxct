# %%
import numpy as np
from numba import vectorize, float64

"""
Common functions for Tully 1 model Hamiltonian
"""
@vectorize([float64(float64, float64, float64)])
def V11(
    R: float,
    A: float,
    B: float,
) -> float:
    sign = np.sign(R)
    return sign * A * (1 - np.exp(-sign * B * R))

@vectorize([float64(float64, float64, float64)])
def V12(
    R: float,
    C: float,
    D: float,
) -> float:
    return C * np.exp(-D * R**2)

@vectorize([float64(float64, float64, float64)])
def gradV11(
    R: float,
    A: float,
    B: float,
) -> float:
    return A * B * np.exp(-np.abs(R) * B)

@vectorize([float64(float64, float64, float64)])
def gradV12(
    R: float,
    C: float,
    D: float,
) -> float:
    return -2 * C * D * R * np.exp(-D * R**2)

def H0_diab(R: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    v11 = np.sum(V11(R, A, B))
    v12 = np.sum(V12(R, C, D))
    
    return np.array(
        [[v11, v12],
        [v12, -v11]], 
        dtype=np.complex128,
    )
    
def gradH0_diab(R: np.ndarray, A: float, B: float, C: float, D: float) -> np.ndarray:
    gradv11 = gradV11(R, A, B)
    gradv12 = gradV12(R, C, D)
    
    return np.array(
        [[gradv11, gradv12],
        [gradv12, -gradv11]], 
        dtype=np.complex128,
    )
    
# %%
