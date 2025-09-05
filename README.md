# QEC-Sensing-Simulation
## Installation
To run these simulations, you need to install the required Python packages:
'''bash
python "pip install -r requirements.txt"
## How to Use

This repository contains both analytical models and numerical simulations.

### Analytical Models
Located in the `Analytical Results` folder:

- **`analytical_T2.py`**: This script analyzes how the discrete QEC time step `dt` affects the fitted T2 value.
  ```bash
  python "Analytical Results/analytical_T2.py"
### Numerical Simulations
Located in the `Numerical Simulations` folder:
- **`average_trajectory.py`**: Simulates the QEC process using the Monte Carlo quantum trajectory method.
- **'evolve_with_T2_noise.py'**: Solves the master equation for a simple two-level system with T2 noise.
- **'single_qec_round_simulation.py'**: A master equation simulation including the full QEC cycle.
