# -*- coding: utf-8 -*-
"""
Post-selection vs no post-selection — normalized time axis (t/T2).
Strict PRL styling: 4 spines (box), inward ticks, clean legends, no grid.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ====================== Physics / sampling ======================
T2_us    = 10.0          # µs
f_MHz    = 0.5           # MHz
t_max_us = 30.0          # µs
lambda_e = 1/40.0        # per µs
N0       = 400

t_us  = np.linspace(0.0, t_max_us, 1000)
omega = 2*np.pi*f_MHz    # rad/µs

P_model = 0.5*(1.0 - np.exp(-t_us/T2_us)*np.cos(omega*t_us))

S = np.exp(-lambda_e * t_us)

P_noPS = S * P_model
P_PS   = P_model

idx        = np.linspace(20, len(t_us)-1, 22, dtype=int)
t_ticks_us = t_us[idx]

def sem_binomial(p, N_eff):
    return np.sqrt(np.clip(p*(1-p), 0.0, 0.25) / np.maximum(N_eff, 1.0))

N_eff_noPS = N0 * np.ones_like(t_ticks_us)
N_eff_PS   = N0 * S[idx]

p_noPS   = P_noPS[idx]
p_PS     = P_PS[idx]
sem_noPS = sem_binomial(p_noPS, N_eff_noPS)
sem_PS   = sem_binomial(p_PS,   N_eff_PS)

# ====================== No Monte Carlo: points on theory ======================
y_noPS = p_noPS
y_PS   = p_PS

# ====================== PRL style ======================
SIZE_MODE   = 'double'                     # 'single' or 'double'
WIDTH_IN    = 7.0 if SIZE_MODE=='double' else 4
HEIGHT      = WIDTH_IN * (0.46 if SIZE_MODE=='double' else 0.78)
LABEL_PT    = 8
TICK_PT     = 7
LINE_W      = 1.3

mpl.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": LABEL_PT, "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.pad_inches": 0.01,
})

def prl_box(ax, lw=0.8):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True,
                   length=3.0, width=lw)

def legend_box(ax):
    leg = ax.legend(loc="upper right", frameon=True, fancybox=False)
    f = leg.get_frame()
    f.set_facecolor("white"); f.set_edgecolor("black"); f.set_linewidth(0.6)
    leg.set_zorder(10)

# ====================== Colors & markers ======================
blue_edge, blue_face = "#1F5AA6", "#8DB7FF"
red_edge,  red_face  = "#8C2D2D", "#F3A6A6"

m_size    = 18
m_edge_w  = 0.9
eb_w      = 1.1
cap_sz    = 2.5
cap_thk   = 1.1

# ====================== Normalize x-axis by T2 ======================
t_over_T2       = t_us / T2_us
t_ticks_over_T2 = t_ticks_us / T2_us
xmax_over_T2    = t_max_us / T2_us

# ====================== Plot ======================
fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT), sharey=True, constrained_layout=False)

# (a) No post-selection
ax = axes[0]
ax.plot(t_over_T2, P_noPS, color=red_edge, lw=LINE_W, label="Theory (no PS)")
ax.errorbar(t_ticks_over_T2, y_noPS, yerr=sem_noPS, fmt='none',
            ecolor=red_edge, elinewidth=eb_w, capsize=cap_sz, capthick=cap_thk, zorder=3)
ax.scatter(t_ticks_over_T2, y_noPS, s=m_size, facecolors=red_face, edgecolors=red_edge,
           linewidths=m_edge_w, zorder=4, label="Points (± s.e.m.)")
ax.set_title("No post-selection")
ax.set_xlabel(r"$dt/T_2$")
ax.set_ylabel(r"Excited-state population $P_e$")
ax.set_xlim(0, xmax_over_T2); ax.set_ylim(0, 1.0)
prl_box(ax); legend_box(ax)
ax.text(0.03, 0.95, "(a)", transform=ax.transAxes, ha="left", va="top")

# (b) Post-selection
ax = axes[1]
ax.plot(t_over_T2, P_PS, color=blue_edge, lw=LINE_W, label="Theory (PS)")
ax.errorbar(t_ticks_over_T2, y_PS, yerr=sem_PS, fmt='none',
            ecolor=blue_edge, elinewidth=eb_w, capsize=cap_sz, capthick=cap_thk, zorder=3)
ax.scatter(t_ticks_over_T2, y_PS, s=m_size, facecolors=blue_face, edgecolors=blue_edge,
           linewidths=m_edge_w, zorder=4, label="Points (± s.e.m.)")
ax.set_title("Post-selection")
ax.set_xlabel(r"$dt/T_2$")
ax.set_xlim(0, xmax_over_T2); ax.set_ylim(0, 1.0)
prl_box(ax); legend_box(ax)
ax.text(0.03, 0.95, "(b)", transform=ax.transAxes, ha="left", va="top")

plt.tight_layout()
plt.show()
