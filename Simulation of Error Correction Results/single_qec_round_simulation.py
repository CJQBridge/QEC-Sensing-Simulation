# This script simulates the time evolution of a two-qubit system undergoing
# Quantum Error Correction (QEC). The system is subject to a driving field
# and a decoherence process (phase damping).
#
# The core of the simulation uses a matrix exponentiation method to solve the
# Lindblad master equation. The evolution is calculated in discrete time steps (dt_seg).
# At the end of each step, a QEC protocol is applied:
# 1. The probability of an error occurring is calculated.
# 2. A random number determines if the error is detected.
# 3. If an error is detected, a correction operation (Z-gate) is applied.
#    Otherwise, the state is projected back onto the code subspace.
#
# The script plots the populations of the |00> and |11> states, along with the
# probability of being in the error subspace, as a function of time.

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

Omega = 2 * np.pi * 1e6  # Driving frequency: 1 MHz
T2 = 10e-6  # T2 decoherence time: 10 µs
total_time = 30e-6  # Total evolution time: 30 µs
dt_seg = 2.5e-7  # Single segment duration: 2.5 µs
n_steps = int(total_time / dt_seg)  # Number of segments
n_t_seg = 100  # Plotting subdivisions within a segment

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

# Hamiltonian & Lindblad Operator
H = (Omega / 2) * np.kron(I2, sigma_x)
L_B = np.sqrt(1 / (2 * T2)) * np.kron(I2, sigma_z)

# Liouvillian Superoperator (16x16)
K = L_B.conj().T @ L_B
Lc = -1j * (np.kron(I4, H) - np.kron(H.T, I4))
Ld = (np.kron(L_B, L_B.conj())
      - 0.5 * np.kron(I4, K) - 0.5 * np.kron(K.T, I4))
L_super = Lc + Ld

# Evolution operator for a single segment: P = exp(L*dt)
P_seg = sla.expm(L_super * dt_seg)
z_gate = np.kron(I2, sigma_z)
P_code = 0.5 * np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=complex)
P_error = 0.5 * np.array([[1, 0, 0, -1], [0, 1, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]], dtype=complex)

# Initial state is (|00> + |11>)/sqrt(2)
rho = np.array([[0.5, 0, 0, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0.5]], dtype=complex).reshape(-1)  # Flatten to 16-dim vector

times, P00, P11, P_err_list = [0.0], [0.5], [0.5], [0.0]

for step in range(n_steps):
    # --- Fine-grained evolution for smooth plotting ---
    P_small = sla.expm(L_super * (dt_seg / n_t_seg))
    for _ in range(n_t_seg):
        rho = P_small @ rho
        rho_mat = rho.reshape(4, 4)
        t_now = times[-1] + dt_seg / n_t_seg
        times.append(t_now)
        P00.append(np.real(rho_mat[0, 0]))
        P11.append(np.real(rho_mat[3, 3]))
        P_err_list.append(np.real(np.trace(P_error @ rho_mat)))

    # --- QEC at the end of the segment ---
    rho_mat = rho.reshape(4, 4)
    error_prob = np.real(np.trace(P_error @ rho_mat))

    if np.random.rand() < error_prob:
        # Error detected: project to error space, apply Z-gate, and normalize
        rho_mat = P_error @ rho_mat @ P_error
        rho_mat /= np.trace(rho_mat)
        rho_mat = z_gate @ rho_mat @ z_gate.conj().T
    else:
        # No error detected: project back to code space and normalize
        rho_mat = P_code @ rho_mat @ P_code
        rho_mat /= np.trace(rho_mat)

    rho = rho_mat.reshape(-1)  # Flatten for the next evolution step

times_us = np.array(times) * 1e6  # Convert time to microseconds for plotting

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(times_us, P00, 'b-', label=r'$P_{00}$')
plt.ylabel(r'$P_{00}$', fontsize=12)
plt.grid(True)
plt.legend()
plt.title(
    f'QEC Simulation (Ω/2π={Omega / (2 * np.pi) * 1e-6:.1f} MHz, T2={T2 * 1e6:.0f}µs, QEC dt={dt_seg * 1e6:.0f}µs)',
    fontsize=14)

plt.subplot(3, 1, 2)
plt.plot(times_us, P11, 'r-', label=r'$P_{11}$')
plt.ylabel(r'$P_{11}$', fontsize=12)
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(times_us, P_err_list, 'g-', label=r'$P_{error}$')
plt.xlabel('Time (µs)', fontsize=12)
plt.ylabel(r'$P_{error}$', fontsize=12)
plt.grid(True)
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
