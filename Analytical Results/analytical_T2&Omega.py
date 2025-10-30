# -*- coding: utf-8 -*-
"""
Exact vs. corrected approx (normalized by T2, r = dt/T2 <= 0.5)
Strict PRL style: single/double column, 4 spines (box), inward ticks, no grid.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===================== PRL config =====================
SAVE          = True
SIZE_MODE     = 'single'          # 'single' -> 3.375 in; 'double' -> 7.0 in
PRL_WIDTH_IN  = {'single': 4, 'double': 7.0}
HEIGHT_RATIO  = 0.62              # PRL-like aspect
LABEL_PT      = 8                 # base: axes/legend font size
TICK_PT       = 7                 # base: tick font size
LINE_W        = 1.3               # curve linewidth

# 全局字号放大系数（按需调 1.2~2.0）
FONT_SCALE    = 1.6

T2_LIST_US    = [10.0, 20.0, 50.0]   # compare several T2 values (µs)
Omega         = 2*np.pi*10e3         # rad/s (10 kHz)

# r = dt/T2 grid (avoid 0 to prevent division-by-zero); cutoff at 0.5
R_MIN, R_MAX, R_N = 0.005, 0.5, 600
r_grid = np.linspace(R_MIN, R_MAX, R_N)

# ===================== Strict PRL base style =====================
mpl.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": LABEL_PT * FONT_SCALE,
    "legend.fontsize": LABEL_PT * FONT_SCALE,
    "xtick.labelsize": TICK_PT * FONT_SCALE,
    "ytick.labelsize": TICK_PT * FONT_SCALE,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,          # ticks on 4 sides
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,           # editable vector fonts
    "savefig.pad_inches": 0.01,
    "figure.dpi": 150, "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
})

def fig_size():
    w = PRL_WIDTH_IN[SIZE_MODE]
    return (w, w*HEIGHT_RATIO)

def prl_box(ax, spine_lw=0.8):
    """4 spines visible + inward ticks on all sides."""
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(spine_lw)
        ax.spines[s].set_color("black")
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True,
                   length=3.0, width=spine_lw)

def legend_box(ax, handles, loc="upper right", **kwargs):
    """白底黑边图例；支持 bbox_to_anchor 等 kw 以便微调位置。"""
    leg = ax.legend(handles=handles, loc=loc,
                    frameon=True, fancybox=False, **kwargs)
    f = leg.get_frame()
    f.set_facecolor("white"); f.set_edgecolor("black"); f.set_linewidth(0.6)
    leg.set_zorder(10)
    return leg

# ===================== Physics helpers =====================
def A_of_dt(dt, T2, Omega):
    term1 = 0.5*(1.0 + np.exp(-dt/T2)) * np.exp(1j*Omega*dt)
    term2 = np.exp(-dt/(2*T2)) * np.sin(Omega*dt) / (2*T2*Omega)
    pref  = 2.0 * np.exp(-dt/(2*T2)) / (1.0 + np.exp(-dt/T2))
    return (term1 + term2) * pref

def curves_for_T2(T2, r_vec):
    """Return dimensionless curves for a given T2 (seconds)."""
    dt = r_vec * T2
    A = A_of_dt(dt, T2, Omega)
    absA = np.clip(np.abs(A), 1e-300, None)
    T2eff_exact = - dt / np.log(absA)
    phi = np.unwrap(np.angle(A))
    Omega_eff_exact = phi / dt

    invT2eff_approx = dt/(8*T2**2) + ((Omega**2)/(3*T2) + 1/(48*T2**3)) * (dt**2)
    T2eff_approx = 1.0 / invT2eff_approx
    Omega_eff_approx = Omega*(1 - dt/(2*T2)) + (Omega/(4*T2**2)) * (dt**2)

    # dimensionless
    y1_exact  = T2eff_exact / T2
    y1_approx = T2eff_approx / T2
    y2_exact  = Omega_eff_exact * T2
    y2_approx = Omega_eff_approx * T2
    return y1_exact, y1_approx, y2_exact, y2_approx

# Okabe–Ito palette (colorblind-safe)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
           "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

# ===================== Compute (and optional export) =====================
curves = {}
rows = []
for i, T2_us in enumerate(T2_LIST_US):
    T2 = T2_us * 1e-6
    y1e, y1a, y2e, y2a = curves_for_T2(T2, r_grid)
    curves[T2_us] = (y1e, y1a, y2e, y2a)
    if SAVE:
        rows.append(pd.DataFrame({
            "T2_us": T2_us,
            "r_dt_over_T2": r_grid,
            "T2eff_over_T2_exact": y1e,
            "T2eff_over_T2_approx": y1a,
            "Omegaeff_times_T2_exact": y2e,
            "Omegaeff_times_T2_approx": y2a,
        }))
if SAVE:
    pd.concat(rows, ignore_index=True).to_csv(
        f"exact_vs_approx_normT2_rmax0p5_{SIZE_MODE}.csv", index=False
    )

# ===================== Fig.1 — T2_eff/T2 (log) =====================
fig1, ax1 = plt.subplots(figsize=fig_size())
color_handles = []
style_handles = [Line2D([0],[0], color="k", lw=LINE_W, ls="-",  label="Exact"),
                 Line2D([0],[0], color="k", lw=LINE_W, ls="--", label="Approx")]

for i, T2_us in enumerate(T2_LIST_US):
    c = PALETTE[i % len(PALETTE)]
    y1e, y1a, _, _ = curves[T2_us]
    ax1.plot(r_grid, y1e, color=c, lw=LINE_W, ls="-")
    ax1.plot(r_grid, y1a, color=c, lw=LINE_W, ls="--")
    color_handles.append(Line2D([0],[0], color=c, lw=LINE_W,
                                label=rf"$T_2={T2_us:.0f}\,\mu\mathrm{{s}}$"))

ax1.set_xlabel(r"$dt/T_2$")
ax1.set_ylabel(r"$T_{2,\mathrm{eff}}/T_2$")
ax1.set_yscale("log")
prl_box(ax1)

# 颜色图例：右上向下挪一点，避免压住曲线
leg_colors = legend_box(ax1, color_handles, loc="upper right",
                        bbox_to_anchor=(0.98, 0.84))
ax1.add_artist(leg_colors)
# 样式图例：左下向上挪一点，避免压住曲线
_ = legend_box(ax1, style_handles, loc="lower left",
               bbox_to_anchor=(0.03, 0.5))

plt.tight_layout()

# ===================== Fig.2 — Omega_eff * T2 (linear) =====================
fig2, ax2 = plt.subplots(figsize=fig_size())

for i, T2_us in enumerate(T2_LIST_US):
    c = PALETTE[i % len(PALETTE)]
    _, _, y2e, y2a = curves[T2_us]
    ax2.plot(r_grid, y2e, color=c, lw=LINE_W, ls="-")
    ax2.plot(r_grid, y2a, color=c, lw=LINE_W, ls="--")

ax2.set_xlabel(r"$dt/T_2$")
ax2.set_ylabel(r"$\Omega_{\mathrm{eff}}\,T_2$")
prl_box(ax2)

# 同样微调两个图例的位置
leg_colors2 = legend_box(ax2, color_handles, loc="upper right",
                         bbox_to_anchor=(0.98, 0.84))
ax2.add_artist(leg_colors2)
_ = legend_box(ax2, style_handles, loc="lower left",
               bbox_to_anchor=(0.03, 0.26))

plt.tight_layout()

plt.show()
