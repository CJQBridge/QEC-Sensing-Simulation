# This script analyzes the effect of the discrete time step (Delta t) of a
# Quantum Error Correction (QEC) cycle on the measurement of the effective
# coherence time (T2).
#
# It models the evolution of a specific coherence term (M_14) using an
# analytical solution derived from the QEC process applied at discrete
# intervals (dt). The evolution over a total time (t) is calculated as
# (Matrix_Element(dt))^(t/dt).
#
# The script scans a range of dt values. For each dt:
# 1. It generates the coherence decay curve over the total_time.
# 2. It fits this decay curve to an exponential function (A * e^(-gamma*t)).
# 3. The fitted decay constant gamma_fit is used to calculate an apparent
#    T2_fit = 1 / gamma_fit.
#
# Finally, it plots the fitted T2_fit as a function of the QEC interval dt,
# comparing it against the true T2 value used in the model. This reveals
# how the discreteness of the QEC protocol introduces systematic errors in
# the perceived decoherence time.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

total_time = 30.0e-6  # Total evolution time: 30 µs
T2_orig = 10.0e-6  # True T2 time: 10 µs
Omega = 2 * np.pi * 1e4  # Rabi frequency: 1 MHz


def exp_decay(t, A, gamma):
    """Exponential decay function: A * exp(-gamma * t)"""
    return A * np.exp(-gamma * t)


def compute_coherence_modulus(dt, T2, Omega, total_time):
    """
    Generates the coherence modulus |M_14|(t) data based on the discrete
    analytical model for a given dt.
    """
    times = np.arange(0.0, total_time, dt)

    exp_term = (0.5 * np.exp(-dt / T2) + 0.5) * np.exp(1j * Omega * dt)
    second_term = (0.5 - 0.5 * np.exp(-dt / T2)) * (
            np.exp(-dt / (2 * T2)) * np.sin(Omega * dt) /
            (T2 * Omega * (1 - np.exp(-dt / T2)))
    )
    m_elem = (exp_term + second_term) * 2 * np.exp(-dt / (2 * T2)) / (1 + np.exp(-dt / T2))

    n_steps = len(times)
    modulus = np.abs([m_elem ** n for n in range(n_steps)])

    return times, modulus

dt_vals = np.linspace(1.0e-6, 10.0e-6, 50)
T2_fit_vals = []

print("Starting simulation... This may take a moment.")

for dt in dt_vals:
    t_arr, mod_arr = compute_coherence_modulus(dt, T2_orig, Omega, total_time)

    try:
        popt, _ = curve_fit(exp_decay, t_arr, mod_arr, p0=[mod_arr[0], 1.0 / T2_orig])
        _, gamma_fit = popt
        T2_fit_vals.append(1.0 / gamma_fit)
    except RuntimeError:
        print(f"Fit failed for dt = {dt * 1e6:.2f} µs. Appending NaN.")
        T2_fit_vals.append(np.nan)  # Append NaN if fit fails

print("Simulation finished.")

T2_fit_us = np.array(T2_fit_vals) * 1e6  # Convert fitted T2 to µs
dt_vals_us = dt_vals * 1e6  # Convert dt axis to µs
T2_orig_us = T2_orig * 1e6  # Convert original T2 to µs

plt.figure(figsize=(10, 6))
plt.plot(dt_vals_us, T2_fit_us, 'o-', label=r'Fitted $T_2^{fit}$')
plt.axhline(T2_orig_us, color='gray', ls='--', label=f'Reference $T_2 = {T2_orig_us:.0f}$ µs')

plt.xlabel(r'QEC Time Step $\Delta t$ (µs)', fontsize=12)
plt.ylabel(r'Fitted Coherence Time $T_2^{fit}$ (µs)', fontsize=12)
plt.title(
    f'Fitted $T_2$ vs. QEC $\Delta t$ (True $T_2={T2_orig_us:.0f}$ µs, $\Omega/2\pi$={Omega / (2 * np.pi) * 1e-6:.1f} MHz)',
    fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()