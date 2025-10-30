# -*- coding: utf-8 -*-
"""
PRL-style plot of σΩ·√T versus total time T, comparing:
Heisenberg limit, No-QEC baseline, a QEC example (dt=2 µs), and the best QEC envelope over dt.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ================= PRL style =================
SIZE_MODE = 'single'                # 'single' (3.375 in) or 'double' (7.0 in)
W = 7.0 if SIZE_MODE=='double' else 4.8
H = W * 0.62

FONT_SCALE = 1.8
BASE_LABEL_PT = 9 if SIZE_MODE=='double' else 8
BASE_TICK_PT  = 8 if SIZE_MODE=='double' else 7
LABEL_PT = BASE_LABEL_PT * FONT_SCALE
TICK_PT  = BASE_TICK_PT  * FONT_SCALE

mpl.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT * 1.05,
    "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT,  "ytick.labelsize": TICK_PT,
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

# ================= Physics =================
def sigma_opt_T_given_T2eff(T, T2eff):
    T = np.asarray(T, dtype=float)
    out = np.empty_like(T, dtype=float)
    mask = T <= (T2eff / 2.0)
    out[mask]  = np.exp(T[mask] / T2eff) / (T[mask] + 1e-30)
    out[~mask] = np.sqrt(2.0 * np.e / (T2eff * T[~mask]))
    return out

# native and example T2,eff (example dt = 2 µs)
T2_native_us   = 10.0     # µs
dt2_example_us = 2.0      # µs
T2eff_dt2_us   = 80.0     # µs

# time axis
T_vals_us   = np.logspace(np.log10(0.05), np.log10(10*T2eff_dt2_us), 300)  # µs
T_vals_s    = T_vals_us * 1e-6                                             # s
T2_native_s = T2_native_us * 1e-6                                          # s

# curves: σ × √T
sigma_hl      = 1.0 / (T_vals_s + 1e-30)
sigma_noqec   = sigma_opt_T_given_T2eff(T_vals_s, T2_native_s)
sigma_qec_dt2 = sigma_opt_T_given_T2eff(T_vals_s, T2eff_dt2_us*1e-6)

# best-over-dt envelope (dt ∈ {2, 5, 10} µs)
dt_to_T2eff_us = {2.0: 80.0, 5.0: 120.0, 10.0: 80.0}  # µs
sigma_qec_all  = [sigma_opt_T_given_T2eff(T_vals_s, v*1e-6) for v in dt_to_T2eff_us.values()]
sigma_qec_best = np.min(np.vstack(sigma_qec_all), axis=0)

sqrtT   = np.sqrt(T_vals_s)
y_hl    = sigma_hl * sqrtT
y_noqec = sigma_noqec * sqrtT
y_dt2   = sigma_qec_dt2 * sqrtT
y_best  = sigma_qec_best * sqrtT

# ================= Colors =================
COL = {"best": "#1f77b4", "noqec": "#d62728", "dt2": "#2ca02c"}

# ================= Plot =================
fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02)

ax.loglog(T_vals_us, y_hl,    ls="--", color="black",      lw=1.8, label="Heisenberg limit")
ax.loglog(T_vals_us, y_noqec, ls="-.", color=COL["noqec"], lw=1.8, label="No QEC")
ax.loglog(T_vals_us, y_dt2,   color=COL["dt2"], lw=2.0, label=f"QEC, dt={dt2_example_us:.0f} μs")

# mark T = 0.5·T2,eff on the example curve (point only, not in legend)
T_half = 0.5 * T2eff_dt2_us  # µs
idx = np.argmin(np.abs(T_vals_us - T_half))
ax.plot(T_vals_us[idx], y_dt2[idx], 'o', ms=3.6, mfc='white', mec=COL["dt2"], mew=1.0, zorder=5)

# best envelope (over dt)
ax.loglog(T_vals_us, y_best, color=COL["best"], lw=2.8, label="QEC (best over $dt$)")

ax.set_xlabel(r"Total measurement time $T$ (μs)")
ax.set_ylabel(r"$\sigma_\Omega\sqrt{T}$ (rad·s$^{-1/2}$)")
prl_box(ax)
ax.grid(True, which="both", ls=":", color="#c8c8c8", alpha=0.55)

leg = ax.legend(loc="lower left", frameon=True, fancybox=False)
f = leg.get_frame()
f.set_facecolor("white"); f.set_edgecolor("black"); f.set_linewidth(0.8)
leg.set_zorder(10)

plt.show()
