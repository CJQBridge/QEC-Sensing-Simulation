# This script compares the performance of a two-qubit Quantum Error Correction (QEC)
# protocol under different correction frequencies. It uses the Quantum Trajectory
# method to simulate the system's evolution.
#
# A key feature of this version is the use of a fixed, high-resolution time grid
# for all simulations. This decouples the plotting resolution from the simulation's
# physical parameters (like the QEC interval), ensuring that all curves are
# directly comparable and avoiding visual artifacts like aliasing.
#
# The simulation steps forward on this fine grid. The QEC protocol is triggered
# whenever the simulation time crosses a multiple of the specified QEC interval (dt_seg).

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

Omega = 2 * np.pi * 1e6  # Driving frequency: 1 MHz
T2 = 10e-6  # T2 decoherence time: 10 µs
total_time = 30e-6  # Total evolution time: 30 µs
n_points = 10001  # Fixed number of points for the time grid
n_traj = 1000  # Number of Monte Carlo trajectories to average
n_jobs = -1  # Use all available CPU cores for parallelization

# List of QEC intervals to simulate. A value >= total_time disables QEC.
dt_values = [0.01e-6, 0.1e-6, 0.2e-6, 30e-6]

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2, I4 = np.eye(2, dtype=complex), np.eye(4, dtype=complex)
u0 = np.sqrt(0.5) * np.array([[1, 1], [-1, 1]], dtype=complex)
U = np.kron(u0, u0)
U_dag = U.conj().T
ORDER = 'F'
rho_init = np.array([[.5, 0, 0, .5], [0, 0, 0, 0], [0, 0, 0, 0], [.5, 0, 0, .5]], complex).reshape(-1, order=ORDER)

H = (Omega / 2) * np.kron(I2, sigma_x)
L_B = np.sqrt(1 / (2 * T2)) * np.kron(I2, sigma_z)
K = L_B.conj().T @ L_B
Lc = -1j * (np.kron(I4, H) - np.kron(H.T, I4))
Ld = np.kron(L_B, L_B.conj()) - 0.5 * np.kron(I4, K) - 0.5 * np.kron(K.T, I4)
L_super = Lc + Ld

z_gate = np.kron(I2, sigma_z)
P_code = 0.5 * np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=complex)
P_error = 0.5 * np.array([[1, 0, 0, -1], [0, 1, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]], dtype=complex)


def simulate_one_trajectory(dt_seg, seed):
    """
    Simulates a single trajectory on a fixed high-resolution time grid.
    QEC is applied whenever the simulation time crosses a multiple of dt_seg.
    """
    rng = np.random.RandomState(seed)
    time_grid = np.linspace(0, total_time, n_points)
    dt_small = time_grid[1] - time_grid[0]
    P_small = sla.expm(L_super * dt_small)

    coherence_traj = np.empty(n_points)
    coherence_traj[0] = 0.5  # Initial coherence value

    rho = rho_init.copy()
    next_qec_time = dt_seg

    for i in range(1, n_points):
        # Evolve the system by one small time step
        rho = P_small @ rho

        # Check if it's time to perform QEC
        current_time = time_grid[i]
        if current_time >= next_qec_time:
            rhoM = rho.reshape(4, 4, order=ORDER)
            p_err = np.clip(float(np.real(np.trace(P_error @ rhoM))), 0.0, 1.0)

            if rng.random() < p_err:
                rhoM = z_gate @ (P_error @ rhoM @ P_error) @ z_gate
            else:
                rhoM = P_code @ rhoM @ P_code

            rhoM /= (np.real(np.trace(rhoM)) + 1e-16)
            rho = rhoM.reshape(-1, order=ORDER)

            # Schedule the next QEC event
            next_qec_time += dt_seg

        # Record the coherence at the current time step
        rhoM_final = rho.reshape(4, 4, order=ORDER)
        coherence_traj[i] = np.real((U @ rhoM_final @ U_dag)[0, 3])

    return coherence_traj


def run_and_average_trajectories(dt_seg):
    """Runs multiple trajectories in parallel and averages the results."""
    time_grid = np.linspace(0, total_time, n_points)

    trajectories = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one_trajectory)(dt_seg, s) for s in range(n_traj)
    )

    avg_coherence = np.mean(trajectories, axis=0)
    return time_grid, avg_coherence


plt.figure(figsize=(12, 8))

for dt in dt_values:
    if dt >= total_time:
        print("--- Running simulation for 'No Error Correction' case ---")
        label = 'No Error Correction'
    else:
        print(f"--- Running QEC simulation for dt_seg = {dt * 1e6:.2f} µs ---")
        label = f'QEC dt = {dt * 1e6:.2f} µs'

    t, Px_avg = run_and_average_trajectories(dt)
    plt.plot(t * 1e6, Px_avg, label=label, linewidth=2)

print("--- All simulations finished. Plotting results. ---")

plt.xlabel('Time (µs)', fontsize=14)
plt.ylabel('Average Coherence', fontsize=14)
plt.title(f'QEC Performance vs. Correction Interval (T2={T2 * 1e6:.0f}µs, {n_traj} trajectories)', fontsize=16)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()