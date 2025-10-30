# -*- coding: utf-8 -*-
"""
PRL-style plot of final signal amplitude A versus dt/T2, combining
decoherence (T2) and per-cycle logical gate error p_L.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- PRL-like style (box + inward ticks) ----------
SIZE_MODE = "single"
FIG_W = 7.0 if SIZE_MODE == "double" else 4
FONT_SCALE = 1.6
LABEL_PT_BASE, TICK_PT_BASE = ((9, 8) if SIZE_MODE == "double" else (8, 7))
LABEL_PT = LABEL_PT_BASE * FONT_SCALE
TICK_PT  = TICK_PT_BASE  * FONT_SCALE

mpl.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT * 1.05,
    "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "axes.linewidth": 0.9,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
})

def prl_box(ax, lw=0.9):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True,
                   length=3.0, width=lw)

# ---------- Physics / inputs ----------
T2_us       = 10.0     # µs
T_total_us  = 100.0    # µs
pL_cycle    = 0.03     # per-cycle
IN_FIG_NOTE = False

T2   = T2_us * 1e-6
Ttot = T_total_us * 1e-6

dt_us = np.linspace(T_total_us/400.0, T_total_us, 400)
dt    = dt_us * 1e-6
x     = dt_us / T2_us

A_qec_perfect = np.exp(-dt/(2*T2))
n_cycles = np.maximum(1, np.floor(Ttot/dt).astype(int))
A_gate   = (1.0 - pL_cycle) ** np.maximum(n_cycles - 1, 0)
A_combined = A_qec_perfect * A_gate

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_W*0.62))

# Palette
COL = {
    "deph":  "#E69F00",
    "gate":  "#009E73",
    "combo": "#0072B2",
}

ax.plot(x, A_qec_perfect, lw=1.8, color=COL["deph"], ls="--",
        label="Decoherence term")
ax.plot(x, A_gate,        lw=1.8, color=COL["gate"], ls="-.",
        label="Gate-error term")
ax.plot(x, A_combined,    lw=2.6, color=COL["combo"], ls="-",
        label=r"Combined $A$")

ax.set_xlabel(r"$dt/T_2$")
ax.set_ylabel(r"Final signal amplitude, $A$")
ax.set_ylim(0, 1.05)

prl_box(ax)

leg = ax.legend(loc="center right", frameon=True, fancybox=False)
frame = leg.get_frame()
frame.set_facecolor("white"); frame.set_edgecolor("black"); frame.set_linewidth(0.8)
leg.set_zorder(10)

if IN_FIG_NOTE:
    txt = (rf"$T={T_total_us:.0f}\,\mu\mathrm{{s}},\; "
           rf"T_2={T2_us:.0f}\,\mu\mathrm{{s}},\; "
           rf"p_L={pL_cycle:.02f}$/cycle")
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.22", linewidth=0.8),
            zorder=10)

plt.tight_layout()
plt.show()
