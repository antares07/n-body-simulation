# Gravitational constant
G = 4.30091e-3

# Relaxation time
def relaxation_time(N_body, R, masses):

    import numpy as np

    t_relax = 0.1 * (N_body/np.log(N_body)) * R ** 1.5 * G * np.sum(masses) ** -0.5

    return t_relax

# Load initial conditions from text file
def load_initial_conditions(filename):

    import numpy as np

    data = np.loadtxt(filename, skiprows=0)
    positions = data[:, 1:4]
    velocities = data[:, 4:]
    masses = data[:, 0]

    return positions, velocities, masses

# Read config file
def read_config(filename):

    import yaml

    with open(filename, 'r') as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    return config_dict

