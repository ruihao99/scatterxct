# %%
"""
Finite Difference Code

This code implements finite difference methods for numerical simulations.

Source:
  - https://github.com/yanzewu/pywp
    The source code for this implementation is based on the pywp project by Yan Zewu.
  - https://en.wikipedia.org/wiki/Finite_difference_coefficient
    The finite difference coefficients `pywp` used was the central difference schemes.
    See the Wikipedia page for more information.

Author: Rui-Hao Bi
Date: 2024-09-11 

"""

import numpy as np
from scipy.special import factorial

def drv_kernel(order: int = 1, accuracy: int = 1): 
    if not np.any(order == np.array([1, 2])):
        raise ValueError(
            f"Derivative order must be 1 or 2. Got {order}."
        )    
    
    n = accuracy
    s = np.zeros(2*n+1) 
    p = np.arange(1, n+1)
    
    if order == 1:
        # s[n+1:] = (-1.0)**(p+1) * factorial(n)**2 / (p * factorial(n-p) * factorial(n+p))
        # s[:n] = -s[n+1:][::-1]  
        s[n+1:] = (-1)**(p+1)*factorial(n)**2/(p*factorial(n-p)*factorial(n+p))
        s[:n] = -s[-1:n:-1]
    else:
        # s[n+1:] = 2*(-1)**(p+1)*factorial(n)**2/(p**2*factorial(n-p)*factorial(n+p))
        # s[:n] = s[-1:n:-1]
        # s[n] = -2*np.sum(1/np.arange(1,n+1)**2)
        s[n+1:] = 2*(-1)**(p+1)*factorial(n)**2/(p**2*factorial(n-p)*factorial(n+p))
        s[:n] = s[-1:n:-1]
        s[n] = -2*np.sum(1/np.arange(1,n+1)**2)
    return s 

def fddrv(length: int, order: int, accuracy: int=1):
    m = np.zeros(length**2)
    s = drv_kernel(order, accuracy)
    
    # fill the diagonal
    m[::length+1] = s[accuracy] 
    
    # fill the off-diagonals
    for j in range(1, accuracy+1):
        m[j:(length-j)*length:length+1] = s[accuracy+j]
        m[length*j::length+1] = s[accuracy-j]
    return m.reshape(length, length)

def test():
    N = 10
    accuracy = 2
    
    D = fddrv(N, order=1, accuracy=accuracy)
    D2 = fddrv(N, order=2, accuracy=accuracy)
    
    print(np.round(D,3))    
    print(np.round(D2,3))
    
# %%
if __name__ == "__main__":
    test()
# %%
