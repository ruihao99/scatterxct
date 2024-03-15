# %%
import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp
from numpy.typing import ArrayLike

from enum import Enum, unique
from typing import List



@unique
class FloquetType(Enum):
    COSINE = "cosine"
    SINE = "sine"
    EXPONENTIAL = "exponential"

def _dim_to_dimF(dim: int, NF: int) -> int:
    return dim * (2 * NF + 1)

def _get_Floquet_offset(dim_sys: int, NF: int, Omega: float) -> List:
    return [np.identity(dim_sys) * ii * Omega for ii in range(-NF, NF+1)]

def _evaluate_sparsity(H: sp.spmatrix) -> float:
    return H.nnz / H.shape[0]**2

def get_HF_cos(
    H0: ArrayLike, # The time-independent part of the Hamiltonian,
    V: ArrayLike, # The time-dependent part of the Hamiltonian (times cosine function),
    Omega: float, # The frequency of the driving field,
    NF: int, # The number of floquet levels to consider,
    is_gradient: bool = False,
    to_csr: bool = True
) -> sp.bsr_matrix:
    """ Suppose the Hamiltonian is given by H(t) = H0 + V(t) * cos(Omega * t). """
    dim = H0.shape[0]
    dimF = _dim_to_dimF(dim, NF)
    dtype = np.complex128 if np.iscomplexobj(H0) or np.iscomplexobj(V) else np.float64  
    
    if NF == 0:
        return sp.bsr_matrix(H0, dtype=dtype)
    
    offsets = _get_Floquet_offset(dim, NF, Omega)
    offsets = np.zeros_like(offsets) if is_gradient else offsets
    V_upper = V
    V_lower = V.transpose().conj()
    # V_upper = V.transpose().conj()
    # V_lower = V 
    
    
    data_first_row = (H0 + offsets[0], V_upper)
    data_middle = ((V_lower, H0+offsets[ii+1], V_upper) for ii in range(2*NF-1))
    data_last_row = (V_lower, H0 + offsets[-1])
    
    data = np.concatenate((data_first_row, *data_middle, data_last_row))
    
    indptr = np.concatenate([(0, ), 2+3*np.arange(0, 2*NF, dtype=int), (6*NF+1, )])
    indices = np.concatenate([(0, 1), *(i+np.arange(0, 3) for i in range(2*NF-1)), (2*NF-1, 2*NF)])
    
    HF = sp.bsr_matrix((data, indices, indptr), shape=(dimF, dimF), dtype=dtype) 
    # print(f"{LA.ishermitian(HF.toarray())=}")
    return HF.tocsr() if to_csr else HF

def get_HF(
    H0: ArrayLike, # The time-independent part of the Hamiltonian,
    V: ArrayLike, # The time-dependent part of the Hamiltonian (times cosine function),
    Omega: float, # The frequency of the driving field,
    NF: int, # The number of floquet levels to consider,
    is_gradient: bool = False,
    to_csr: bool = True,
    floquet_type: FloquetType = FloquetType.COSINE
) -> sp.bsr_matrix:
    if floquet_type == FloquetType.COSINE:
        return get_HF_cos(H0, V, Omega, NF, is_gradient=is_gradient, to_csr=to_csr)
    elif floquet_type == FloquetType.SINE:
        raise NotImplementedError("The sine type of Floquet Hamiltonian is not implemented yet.")
    elif floquet_type == FloquetType.EXPONENTIAL:
        raise NotImplementedError("The exponential type of Floquet Hamiltonian is not implemented yet.")
    else:
        raise NotImplementedError("The Hamiltonian type is not implemented yet.")

# %%
if __name__ == "__main__":
    import numpy as np
    
    def benchmark_A_mul_B(A, B, n=1000):
        import time
        start = time.time()
        for _ in range(n):
            C = np.dot(A, B)
        end = time.time()
        return end - start 
    
    def benchmark_dense_mul_sparse(H, n=10000):
        B = np.zeros_like(H)
        
        t_sparse = benchmark_A_mul_B(H, B, n)
        print(f"Time for sparse-dense matrix multiplication: {t_sparse}")
        
        t_dense = benchmark_A_mul_B(H.toarray(), B, n)
        print(f"Time for dense-dense matrix multiplication: {t_dense}")
        
    def benchmark_bsr_to_csr(Mbsr, n=int(1e6)):
        import time
        start = time.time()
        for _ in range(n):
            Mcsr = Mbsr.tocsr()
        end = time.time()
        return (end - start) / n
     
    H = np.array([[1, 2], [2, 4]])
    V = np.array([[0, 0.3], [0.3, 0]])
    # V = np.array([[0., 1+1.j], [1+1.j, 0.0]])
    # V = np.zeros_like(H)
    
    
    Omega = 0.1
    NF = 2
    
    HF = get_HF_cos(H, V, Omega, NF, is_gradient=False)
    
    print(f"{HF.toarray()}=")
    
    print("Benchmark for BSR matrix multiplication:") 
    benchmark_dense_mul_sparse(HF, n=10000) 
    
    
    print("Benchmark for CSR matrix multiplication:") 
    benchmark_dense_mul_sparse(HF.tocsr(), n=10000) 
   
    print("The timescale for converting BSR to CSR: ", benchmark_bsr_to_csr(HF, n=10000))

# %%
