from functions import compute_acceleration
from functions import compute_total_energy
from functions import leapfrog_step, rk4_step
from functions import load_initial_conditions, read_config, G, relaxation_time
from functions import plot_trajectories2D, plot_trajectories3D

import argparse
import numpy as np
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_simulation(initial_positions, initial_velocities, masses, G, h, n_steps, method):

        N_body, dim = initial_positions.shape[0], initial_positions.shape[1]

        # Initialize arrays to store the simulation history
        r = np.zeros((N_body, dim, n_steps + 1))
        v = np.zeros((N_body, dim, n_steps + 1))
        a = np.zeros((N_body, dim, n_steps + 1))
        E = np.zeros((n_steps + 1))

        # Set initial conditions
        r[:, :, 0] = initial_positions
        v[:, :, 0] = initial_velocities
        a[:, :, 0] = compute_acceleration(G, r[:, :, 0], masses)
        E[0] = compute_total_energy(G, r[:, :, 0], v[:, :, 0], masses)
        logging.info("Initial conditions set.")

        # Run the simulation
        logging.info(f"Running simulation with {method}.")
        step_bar = tqdm(range(n_steps), desc="Simulating")

        if method == 'leapfrog':
            for step in step_bar:
                step_bar.set_description(f'Accuracy = {np.abs(1 - E[step]/E[0]):.5f}')
                leapfrog_step(h, G, r, v, a, masses, E, step)

        elif method == 'rk4':
            for step in step_bar:
                step_bar.set_description(f'Accuracy = {np.abs(1 - (E[step]/E[0])):.5f}')
                rk4_step(h, G, r, v, masses, E, step)

        else:
            raise ValueError(f"{method} is a non existing method")

        logging.info("Simulation complete.")

        return r, v, a, E

def main():
    
    try:
        logging.info("Starting N-body simulation.")

        # Set up the argument parser
        parser = argparse.ArgumentParser(description="N-body simulation.")
        parser.add_argument("-i", type=str, help="Path to config file.", required=True)
        parser.add_argument("-p", help="Plot simulation.", required=False, action='store_true')
        args = parser.parse_args()

        # Get config file from command line
        logging.info("Reading configuration file.")
        config_file = read_config(args.i)
        plot = args.p

        # Load initial conditions
        logging.info("Loading initial conditions.")
        initial_positions, initial_velocities, masses = load_initial_conditions(config_file['config']['initial_conditions_file'])
        N_body = initial_positions.shape[0]

        # Relaxation time
        t_relax = relaxation_time(N_body, 0.8, masses)
        logging.info(f"Relaxation time computed: {t_relax:.5f}")

        # Total integration time
        t_tot = config_file['config']['integration_time'] * t_relax
        n_steps = config_file['config']['n_steps']
        h = t_tot / n_steps
        logging.info(f"Total integration time: {t_tot:.5f}, Number of steps: {n_steps}, Time step: {h:.5f}")

        r, v, a, E = run_simulation(initial_positions, initial_velocities, masses, G, h, n_steps,
                                    config_file['config']['method'])        

        # Plot the trajectories of the bodies
        if plot:
            logging.info("Generating plots.")
            
            plot_trajectories2D(config_file, E, r, masses)
            plot_trajectories3D(config_file, r, masses)

            logging.info(f"Plots saved to {config_file['config']['output_path']}")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == '__main__':
    main()
