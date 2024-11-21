# %%
import numpy as np

def get_optimal_laser_center(
    t: np.ndarray, 
    R: np.ndarray, 
    P: np.ndarray
):
    """For a scattering problem, obtain the optimal laser onset to
    manipulate the electronic transition.
    
    You need to input a averaged trajectory obtained without the 
    laser interaction.

    Args:
        t (np.ndarray): time grid of the non-laser trajectory
        R (np.ndarray): nuclear position of the non-laser trajectory
        P (np.ndarray): electronic population of the non-laser trajectory
    """
    
    # P is reflected to the left
    if np.any(P < 0.0):
        # find the time between the first and the last negative population
        idx_pos = np.where(P >= 0.0)[0]
        idx_neg = np.where(P < 0.0)[0]
        t_pos = t[idx_pos]
        t_neg = t[idx_neg]
        t_neg_min = t_neg[0]
        t_neg_max = t_neg[-1]
        t_center = 0.5 * (t_neg_min + t_neg_max)
    else:
        # find the time between the nuclear position crossing 0
        idx_pos = np.where(R >= 0.0)[0]
        idx_neg = np.where(R < 0.0)[0]
        t_pos = t[idx_pos]
        t_neg = t[idx_neg]
        t_neg_min = t_neg[0]
        t_neg_max = t_neg[-1]
        t_center = 0.5 * (t_neg_min + t_neg_max)
        
    return t_center