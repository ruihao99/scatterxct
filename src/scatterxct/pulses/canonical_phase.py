# %%
import math

def cp_cosine(phi: float,) -> complex:
    """ exp(i * phi) = cos(phi) + i * sin(phi) """
    return math.cos(phi) + 1j * math.sin(phi)

def cp_sine(phi: float,) -> complex:
    """ -i * exp(i * phi) = cos(phi) - i * sin(phi) """ 
    return -1j * math.cos(phi) + math.sin(phi)