from typing import Tuple

def safe_boundary(R_lims: Tuple[float, float], ) -> Tuple[float, float]:
    TOL_BOUNDARY = 1e-5
    DR = R_lims[1] - R_lims[0]
    return (R_lims[0] - TOL_BOUNDARY * DR, R_lims[1] + TOL_BOUNDARY * DR)
