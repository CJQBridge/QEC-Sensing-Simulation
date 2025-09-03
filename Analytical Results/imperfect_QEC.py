# This script generates two figures to analyze the performance and trade-offs
# of a discrete Quantum Error Correction (QEC) protocol.
#
# Figure 1: Illustrates the fundamental trade-off between two sources of error
# as a function of the QEC cycle time (dt) for a fixed total evolution time.
#   - Decoherence Error: Worsens as `dt` increases because the system is left
#     unprotected for longer intervals.
#   - Gate Error: Accumulates with each QEC cycle. This error worsens as `dt`
#     decreases, because more cycles are needed to cover the total time.
# The figure shows an optimal `dt` that maximizes the final signal amplitude.
#
# Figure 2: Calculates the gate-error threshold (p_L*) as a function of the
# total Ramsey time (T).
#   - The threshold `p_L*` represents the maximum tolerable error per QEC cycle
#     for the protocol to be beneficial. If the actual gate error exceeds this
#     threshold, the QEC protocol does more harm than good.
#   - The plot is generated for several fixed `dt` values, showing how the
#     threshold changes with the total experiment duration.

import numpy as np
import matplotlib.pyplot as plt

T2_us = 10.0  # Native T2 coherence time [µs]
T_total_us = 100.0  # Total free-evolution time for Figure 1 [µs]
pL_cycle = 0.03  # Per-cycle uncorrectable logical error rate for Figure 1

T_min_us = 20.0  # Minimum total time to plot [µs]
T_max_us = 200.0  # Maximum total time to plot [µs]
dt_list_us = [2.5, 5.0, 10.0]  # List of QEC cycle times to compare [µs]

T_ref_us = 100.0
Omega = 2 * np.pi / (T_ref_us * 1e-6)  # Rabi frequency [rad/s]


def r_single_step(dt, T2, Omega):
    """
    Calculates the coherence reduction factor r = |M_14(dt)| for a single QEC
    step of duration dt, using the full analytical model.
    """
    u = np.exp(-dt / T2)

    denominator = (T2 * Omega * (1 - u))
    if np.abs(denominator) < 1e-12:
        second_term_fraction = 1.0  # sin(x)/x -> 1 and (1-u)/dt -> 1/T2 cancel out
    else:
        second_term_fraction = np.sin(Omega * dt) / denominator

    exp_term = (0.5 * u + 0.5) * np.exp(1j * Omega * dt)
    second_term = (0.5 - 0.5 * u) * (np.exp(-dt / (2 * T2)) * second_term_fraction) * (T2 * Omega)
    m_elem = (exp_term + second_term) * 2 * np.exp(-dt / (2 * T2)) / (1 + u)
    return abs(m_elem)


T2 = T2_us * 1e-6
Ttot = T_total_us * 1e-6

dt_us = np.linspace(T_total_us / 400.0, T_total_us, 400)
dt = dt_us * 1e-6

A_qec_perfect = np.exp(-dt / (2 * T2))

n_cycles = np.maximum(1, np.floor(Ttot / dt).astype(int))
A_gate = (1.0 - pL_cycle) ** np.maximum(n_cycles - 1, 0)

A_combined = A_qec_perfect * A_gate

plt.figure(figsize=(8, 5))
plt.plot(dt_us, A_qec_perfect, label="a) Decoherence Term (Imperfect QEC)")
plt.plot(dt_us, A_gate, label="b) Gate Error Term")
plt.plot(dt_us, A_combined, label="c) Combined Amplitude A(dt)", linewidth=2.5)
plt.xlabel("QEC Cycle Time, dt (µs)", fontsize=12)
plt.ylabel("Final Signal Amplitude, A(dt)", fontsize=12)
plt.title(f"QEC Trade-off at T_total = {T_total_us} µs (T2={T2_us} µs, p_L/cycle={pL_cycle})", fontsize=14)
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

for dt_us in dt_list_us:
    dt = dt_us * 1e-6
    r = r_single_step(dt, T2, Omega)

    N_min = int(np.ceil(T_min_us / dt_us))
    N_max = int(np.floor(T_max_us / dt_us))

    Ts_us = []
    pLs = []
    for N in range(N_min, N_max + 1):
        T_us = N * dt_us
        T = T_us * 1e-6

        ratio = (1.0 / N) ** (1.0 / (2.0 * N))
        rhs = np.exp(-T / (N * T2)) / (r * ratio)
        pL_star = 1.0 - rhs

        Ts_us.append(T_us)
        pLs.append(pL_star)

    plt.plot(Ts_us, pLs, marker='o', linestyle='-', markersize=5, label=f"dt = {dt_us} µs")
plt.xlabel("Total Ramsey Time, T (µs)", fontsize=12)
plt.ylabel("Gate Error Threshold, p_L*", fontsize=12)
plt.title("QEC Gate-Error Threshold vs. Total Time", fontsize=14)
plt.grid(True, alpha=0.5)
plt.legend(title="Fixed QEC Cycle Time")
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()