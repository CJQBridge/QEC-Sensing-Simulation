# This script calculates and plots the time evolution of the ground state
# population (rho_00) of a two-level quantum system. It solves the master
# equation for a system undergoing Rabi oscillations under a resonant driving
# field, considering a decoherence process characterized by the T2 time.
# The script compares the evolution with decoherence to the ideal, dissipationless case.

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# --- System Parameters ---
Omega = 2 * np.pi * 1e6  # Rabi frequency: 1 MHz
T2 = 10e-6              # T2 decoherence time: 10 µs
t_end = 30e-6           # Total evolution time: 30 µs
t_values = np.linspace(0, t_end, 500) # Time points for simulation

def equations(t, y, Omega, T2):
    rho00, rho01, rho10, rho11 = y
    drho00_dt = 1j * (Omega / 2) * (rho10 - rho01)
    drho01_dt = 1j * (Omega / 2) * (rho11 - rho00) - (1 / T2) * rho01
    drho10_dt = 1j * (Omega / 2) * (rho00 - rho11) - (1 / T2) * rho10
    drho11_dt = 1j * (Omega / 2) * (rho01 - rho10)
    return [drho00_dt, drho01_dt, drho10_dt, drho11_dt]

def equation_ideal(t, y, Omega):
    rho00, rho01, rho10, rho11 = y
    drho00_dt = 1j * (Omega / 2) * (rho10 - rho01)
    drho01_dt = 1j * (Omega / 2) * (rho11 - rho00)
    drho10_dt = 1j * (Omega / 2) * (rho00 - rho11)
    drho11_dt = 1j * (Omega / 2) * (rho01 - rho10)
    return [drho00_dt, drho01_dt, drho10_dt, drho11_dt]

# Initial state: |+> state, density matrix representation
y0 = [1/2, 1j/2, -1j/2, 1/2]

# --- Solve the Differential Equations ---
sol_numeric = solve_ivp(equations, [0, t_end], y0, args=(Omega, T2),
                        t_eval=t_values, method='RK45')
sol_ideal = solve_ivp(equation_ideal, [0, t_end], y0, args=(Omega,),
                      t_eval=t_values, method='RK45')

rho00_numeric = np.real(sol_numeric.y[0])
rho00_ideal = np.real(sol_ideal.y[0])


# --- Plotting the Results ---
plt.figure(figsize=(10, 6))
plt.plot(t_values * 1e6, rho00_numeric, label=f'With T2 = {T2 * 1e6:.0f} µs', linewidth=2, color='blue')
plt.plot(t_values * 1e6, rho00_ideal, '--', label='Ideal case (no T2)', color='gray')

plt.xlabel('Time t (µs)', fontsize=14)
plt.ylabel(r'Ground State Population $\rho_{00}(t)$', fontsize=14)
plt.title(f'Rabi Oscillation with Decoherence (Ω/2π = {Omega/(2*np.pi)*1e-6:.1f} MHz)', fontsize=16)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.show()