import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_acceleration(G, r, m):
    
    N_body = r.shape[0]
    dim = r.shape[1]
    
    a = np.zeros((N_body, dim))  # Initialize acceleration array

    for i in prange(N_body):
        for j in range(N_body):
            if i != j:
                diff = r[i] - r[j]  # Vectorized difference computation
                dist_sq = np.dot(diff, diff)  # Compute squared distance
                dist = np.sqrt(dist_sq) + 1e-2  # Avoid division by zero
                
                a[i] += - G * m[j] * diff / (dist**3)  # Vectorized acceleration update

    return a