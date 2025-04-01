import numpy as np
from numba import njit, prange

@njit(parallel=True)
def compute_total_energy(G, r, v, m, softening=1e-2):

    N_body = r.shape[0]
    
    potential_arr = np.zeros(N_body)
    kinetic_arr = np.zeros(N_body)
    
    # Parallelize over the outer loop.
    for i in prange(N_body):

        kinetic_arr[i] = 0.5 * m[i] * np.sum(v[i] * v[i])

        tmp = 0.
        for j in range(i + 1, N_body):

            diff = r[i] - r[j]
            dist = np.linalg.norm(diff) + softening
            tmp += G * m[i] * m[j] / dist ** 3

        potential_arr[i] = tmp
        
    potential = np.sum(potential_arr)
    kinetic = np.sum(kinetic_arr)
    
    return kinetic + potential