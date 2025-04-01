# N-Body Simulation

This program simulates the motion of multiple bodies under gravitational interaction using numerical integration methods.

## Features
- Supports **Leapfrog** and **Runge-Kutta 4 (RK4)** integration methods.
- Computes acceleration and total energy at each step.
- Configurable simulation parameters via a configuration file.
- Supports **2D and 3D trajectory plotting**.
- Optimized using **Numba** for performance.

## Installation

### Prerequisites
Make sure you have **Python 3.8+** installed. You also need the following dependencies:

- `numpy`
- `numba`
- `matplotlib`
- `tqdm`

### Install via `pip`

1. Clone this repository:
   ```bash
   git clone https://github.com/antares07/n-body-simulation.git
   cd n-body-simulation
   ```

2. Install the package:
   ```bash
   pip install .
   ```

This will install the program and make it available as a command-line tool.

## Usage

### Running the Simulation
Once installed, run the simulation using:
```bash
nbody -i path/to/config.yaml -p
```

#### Command-line Arguments:
- `-i <path>` : Specifies the path to the configuration file.
- `-p` : Enables plotting of the simulation results.

### Example Configuration File (`config.yaml`)
```yaml
config:
  initial_conditions_file: "initial_conditions.txt"
  integration_time: 10.0  # Scaled by relaxation time
  n_steps: 1000
  method: "leapfrog"
  output_path: "results/"
```


## Development
If you want to modify the code, install the package in **editable mode**:
```bash
pip install -e .
```

To run the program directly from source without installation:
```bash
python nbody.py -i path/to/config.yaml -p
```

## License
This project is licensed under the MIT License.

