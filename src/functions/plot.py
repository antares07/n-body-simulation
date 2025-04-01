import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectories3D(config, r, m, mass_scale=100):
    """
    Plots the 3D trajectories of N bodies and marks the final positions.
    
    Parameters:
        r (np.ndarray): Position array of shape (N, 3, steps)
        m (np.ndarray): Mass array of shape (N,)
        mass_scale (float): Factor to scale the scatter marker size based on mass
    """
    N, dim, steps = r.shape
    if dim != 3:
        raise ValueError("The dimension of the positions must be 3 for 3D plotting.")
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory for each body
    for i in range(N):
        # Extract trajectory for body i
        traj = r[i, :, :]  # shape (3, steps)
        ax.plot(traj[0, :], traj[1, :], traj[2, :])
        
        # Scatter the final position, scaling the marker size by the mass.
        final_pos = traj[:, -1]
        ax.scatter(final_pos[0], final_pos[1], final_pos[2],
                   s=m[i] * mass_scale, c='k', marker='o')  # you can customize color
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.title("Trajectories and Final Positions of N Bodies")
    plt.savefig(config["config"]["output_path"]+"3Dplot"+config["config"]["method"]+".png")

def plot_trajectories2D(config, E, r, m, mass_scale=100):

    N, dim, _ = r.shape

    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(211)

    # Plot trajectory for each body
    for i in range(N):
        # Extract trajectory for body i
        traj = r[i, :, :]  # shape (3, steps)
        ax1.plot(traj[0, :], traj[1, :])
        
        # Scatter the final position, scaling the marker size by the mass.
        final_pos = traj[:, -1]
        ax1.scatter(final_pos[0], final_pos[1], s=m[i] * mass_scale, c='k', marker='o')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('N-Body Simulation Trajectories')

    ax2 = fig.add_subplot(212)
    ax2.plot(np.abs(1 - (E / E[0])))
    ax2.set_yscale('log')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Relative Energy Error')
    ax1.set_title('Accuracy')

    plt.grid(True)
    plt.savefig(config["config"]["output_path"]+"2Dplot"+config["config"]["method"]+".png")