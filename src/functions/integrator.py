from numba import njit
from functions import compute_acceleration, compute_total_energy
import numpy as np

@njit
def leapfrog_step(h, G, r, v, a, m, E, step): 

    # Update positions using r(t+h) = r(t) + h*v(t) + 0.5*h^2*a(t)
    r[:, :, step + 1] = r[:, :, step] + h * v[:, :, step] + 0.5 * h**2 * a[:, :, step]

    # Compute acceleration at new positions (time t+h)
    a[:, :, step + 1] = compute_acceleration(G, r[:, :, step + 1], m)

    # Update velocities using v(t+h) = v[:, :, step] + 0.5*h*(a[:, :, step] + a[:, :, step + 1])
    v[:, :, step + 1] = v[:, :, step] + 0.5 * h * (a[:, :, step] + a[:, :, step + 1])

    E[step + 1] = compute_total_energy(G, r[:, :, step + 1], v[:, :, step + 1], m, softening=0)

@njit
def rk4_step(h, G, r, v, m, E, step):

    N_body = r.shape[0]
    dim = r.shape[1]
    
    # Allocate temporary arrays for the RK4 increments.
    k1r = np.empty((N_body, dim))
    k2r = np.empty((N_body, dim))
    k3r = np.empty((N_body, dim))
    k4r = np.empty((N_body, dim))
    
    k1v = np.empty((N_body, dim))
    k2v = np.empty((N_body, dim))
    k3v = np.empty((N_body, dim))
    k4v = np.empty((N_body, dim))

    temp_r = np.empty((N_body, dim))
    
    # k1
    k1r[:, :] = v[:, :, step]

    # Compute acceleration at current positions.
    k1v[:, :] = compute_acceleration(G, r[:, :, step], m)
    
    # k2    
    temp_r[:, :] = r[:, :, step] + 0.5 * h * k1r[:, :]
    k2r[:, :] = v[:, :, step] + 0.5 * h * k1v[:, :]
    k2v[:, :] = compute_acceleration(G, temp_r, m)

    # k3
    temp_r[:, :] = r[:, :, step] + 0.5 * h * k2r[:, :]
    k3r[:, :] = v[:, :, step] + 0.5 * h * k2v[:, :]
    k3v[:, :] = compute_acceleration(G, temp_r, m)
    
    # k4
    temp_r[:, :] = r[:, :, step] + h * k3r[:, :]
    k4r[:, :] = v[:, :, step] + h * k3v[:, :]
    k4v[:, :] = compute_acceleration(G, temp_r, m)

    # Update positions and velocities for the next step.
    r[:, :, step+1] = r[:, :, step] + h * (k1r[:, :] + 2*k2r[:, :] + 2*k3r[:, :] + k4r[:, :]) / 6.0
    v[:, :, step+1] = v[:, :, step] + h * (k1v[:, :] + 2*k2v[:, :] + 2*k3v[:, :] + k4v[:, :]) / 6.0

    E[step + 1] = compute_total_energy(G, r[:, :, step + 1], v[:, :, step + 1], m, softening=0)