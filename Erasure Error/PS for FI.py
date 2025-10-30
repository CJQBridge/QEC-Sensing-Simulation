# -*- coding: utf-8 -*-
"""
FI-rate ratio vs dt/T2 (PRL style): post-selection vs no post-selection.
x-axis: dt/T2 (dimensionless); 4 spines + inward ticks; clean legend; no grid.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== Parameters =====
T2_us     = 10.0          # µs
t_us      = 100.0         # µs
tau_m_us  = 1000.0        # µs
p_g_cycle = 0.02          # per cycle
p_e_cycle = 0.03          # per cycle

# ----- dt grid -----
dt_us = np.linspace(3.0, 60.0, 120)  # µs
dt    = dt_us * 1e-6                 # s
T2    = T2_us * 1e-6                 # s

# ===== Model (continuous cycle count) =====
Nc      = t_us / dt_us
A_env   = np.exp(-dt / (2.0 * T2))
A_gate  = (1.0 - p_g_cycle) ** Nc
S_surv  = (1.0 - p_e_cycle) ** Nc

A_core  = A_env * A_gate
A2_core = A_core**2
A2_bare = np.exp(-2.0 * t_us / T2_us) * np.ones_like(dt_us)

R_PS_over_bare   = (S_surv    * A2_core / A2_bare) / (1.0 + tau_m_us/dt_us)
R_noPS_over_bare = (S_surv**2 * A2_core / A2_bare) / (1.0 + tau_m_us/dt_us)

# ----- Light smoothing -----
def smooth(y, k=7):
    pad_left  = np.full(k//2, y[0])
    pad_right = np.full(k - k//2 - 1, y[-1])
    ypad = np.r_[pad_left, y, pad_right]
    return np.convolve(ypad, np.ones(k)/k, mode="valid")

R_PS_sm   = smooth(R_PS_over_bare,   k=7)
R_noPS_sm = smooth(R_noPS_over_bare, k=7)

# ----- Stability for log-y -----
y_floor = 1e-12
R_PS_sm   = np.clip(R_PS_sm,   y_floor, None)
R_noPS_sm = np.clip(R_noPS_sm, y_floor, None)

# ===== PRL style =====
SIZE_MODE = 'single'                      # 'single' or 'double'
WIDTH_IN  = 4 if SIZE_MODE=='single' else 7.0
HEIGHT    = WIDTH_IN * (0.62 if SIZE_MODE=='single' else 0.58)
LABEL_PT, TICK_PT = (8, 7) if SIZE_MODE=='single' else (9, 8)

mpl.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": LABEL_PT, "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ===== Colors =====
blue_edge = "#1F5AA6"   # PS
red_edge  = "#8C2D2D"   # no-PS

# ===== X-axis: dimensionless dt/T2 =====
x = dt_us / T2_us

# ===== Plot =====
fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT))
ax.plot(x, R_PS_sm,   lw=1.8, color=blue_edge, label='Post-selection')
ax.plot(x, R_noPS_sm, lw=1.8, color=red_edge,  label='No post-selection')

ax.set_yscale('log')
ax.set_xlabel(r"$dt/T_2$")
ax.set_ylabel(r'FI-rate ratio  $\mathcal{R}_{\rm QEC}/\mathcal{R}_{\rm bare}$')

leg = ax.legend(loc='best', frameon=True, fancybox=False)
f = leg.get_frame(); f.set_facecolor("white"); f.set_edgecolor("black"); f.set_linewidth(0.7)

plt.tight_layout()
plt.show()
