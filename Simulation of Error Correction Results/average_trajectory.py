# -*- coding: utf-8 -*-
"""
QEC coherence vs time — PRL-ready single-column figure.
Legend sits just above the axes top spine, spans full width, and wraps to new lines.
"""

import numpy as np
import scipy.linalg as sla
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt

# ------------------ Simulation params ------------------
Omega      = 0.5 * np.pi * 1e6   # rad/s
T2         = 5e-6                # s
total_time = 30e-6               # s
n_points   = 10001
n_traj     = 1000
n_jobs     = -1
dt_values  = [0.01e-6, 0.50e-6, 30e-6]  # dt >= total_time -> No QEC

# ------------------ Operators (two-qubit) --------------
sigma_x = np.array([[0, 1],[1, 0]], complex)
sigma_z = np.array([[1, 0],[0,-1]], complex)
I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)

u0 = np.sqrt(0.5) * np.array([[1,1],[-1,1]], complex)
U = np.kron(u0, u0); U_dag = U.conj().T
ORDER = "F"

rho_init = np.array([[.5,0,0,.5],[0,0,0,0],[0,0,0,0],[.5,0,0,.5]], complex).reshape(-1, order=ORDER)

H   = (Omega/2) * np.kron(I2, sigma_x)
L_B = np.sqrt(1/(2*T2)) * np.kron(I2, sigma_z)
K   = L_B.conj().T @ L_B
Lc  = -1j*(np.kron(I4, H) - np.kron(H.T, I4))
Ld  = np.kron(L_B, L_B.conj()) - 0.5*np.kron(I4, K) - 0.5*np.kron(K.T, I4)
L_super = Lc + Ld

z_gate  = np.kron(I2, sigma_z)
P_code  = 0.5*np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], complex)
P_error = 0.5*np.array([[1,0,0,-1],[0,1,-1,0],[0,-1,1,0],[-1,0,0,1]], complex)

# ------------------ Simulator ------------------
def simulate_one_trajectory(dt_seg, seed):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, total_time, n_points)
    dt_small = t[1] - t[0]
    P_small = sla.expm(L_super * dt_small)

    coh = np.empty(n_points); coh[0] = 0.5
    rho = rho_init.copy()
    next_qec = np.inf if dt_seg >= total_time else dt_seg

    for i in range(1, n_points):
        rho = P_small @ rho
        if t[i] >= next_qec:
            rhoM = rho.reshape(4,4,order=ORDER)
            p_err = np.clip(float(np.real(np.trace(P_error @ rhoM))), 0.0, 1.0)
            if rng.random() < p_err:
                rhoM = z_gate @ (P_error @ rhoM @ P_error) @ z_gate
            else:
                rhoM = P_code @ rhoM @ P_code
            rhoM /= (np.real(np.trace(rhoM)) + 1e-16)
            rho = rhoM.reshape(-1, order=ORDER)
            next_qec += dt_seg

        rhoM_final = rho.reshape(4,4,order=ORDER)
        coh[i] = np.real((U @ rhoM_final @ U_dag)[0,3])

    return coh

def run_trajectories(dt_seg, errbar_mode="sem"):
    t = np.linspace(0, total_time, n_points)
    trajs = Parallel(n_jobs=n_jobs)(
        delayed(simulate_one_trajectory)(dt_seg, s) for s in range(n_traj)
    )
    trajs = np.asarray(trajs)
    mean = trajs.mean(axis=0)
    std  = trajs.std(axis=0, ddof=1 if n_traj>1 else 0)
    sem  = std/np.sqrt(n_traj)
    yerr = sem if errbar_mode=="sem" else std
    return t, mean, yerr

# ------------------ PRL style ------------------
WIDTH_IN, HEIGHT_RATIO = 4, 0.62
LABEL_PT_BASE, TICK_PT_BASE, LINE_W = 8, 7, 1.3
FONT_SCALE = 1.6
LABEL_PT = LABEL_PT_BASE * FONT_SCALE
TICK_PT  = TICK_PT_BASE  * FONT_SCALE
LEGEND_NCOL = 2

mpl.rcParams.update({
    "font.family": ["Arial","Helvetica","DejaVu Sans"],
    "axes.labelsize": LABEL_PT, "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "axes.linewidth": 0.8,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
})

def prl_box(ax, lw=0.8):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True,
                   length=3.0, width=lw)

# ------------------ Colors & styles ------------------
COL = {"fast":"#0072B2", "slow":"#009E73", "noqec":"#D55E00"}
LS  = {"fast":"-", "slow":"-", "noqec":"--"}

# ------------------ Error bars ------------------
N_ERRPTS    = 28
EB_LINE_W   = 1.1
MARKER_SIZE = 2.6
SHADE_ALPHA = 0.10

# ------------------ Plot ------------------
fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN*HEIGHT_RATIO), layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.18, hspace=0.02)

for dt in dt_values:
    t, mean, yerr = run_trajectories(dt, errbar_mode="sem")

    if dt >= total_time:
        label = "No QEC"; c, ls = COL["noqec"], LS["noqec"]
    else:
        r = dt / T2
        c, ls = (COL["fast"], LS["fast"]) if np.isclose(dt, dt_values[0]) else (COL["slow"], LS["slow"])
        label = rf"$dt/T_2={r:.3f}$"

    tx = t / T2
    ax.plot(tx, mean, lw=LINE_W, color=c, ls=ls, zorder=5, label=label)
    ax.fill_between(tx, mean-yerr, mean+yerr, color=c, alpha=SHADE_ALPHA, lw=0, zorder=3)

    idx = np.linspace(0, mean.size-1, N_ERRPTS, dtype=int)
    ax.errorbar(tx[idx], mean[idx], yerr=yerr[idx],
                fmt='none', ecolor=c, elinewidth=EB_LINE_W, capsize=0, zorder=7)
    ax.plot(tx[idx], mean[idx], 'o', ms=MARKER_SIZE, mfc='white', mec=c, mew=0.9, zorder=8)

ax.set_xlabel(r"$t/T_2$")
ax.set_ylabel(r"Amp (arb. units)")
prl_box(ax)

# ---- Legend above axes (full-width, wrapping) ----
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles, labels,
    loc="lower left",
    bbox_to_anchor=(0.0, 1.01, 1.0, 0.001),
    bbox_transform=ax.transAxes,
    mode="expand",
    ncol=min(LEGEND_NCOL, len(labels)),
    borderaxespad=0.0,
    frameon=True, fancybox=False,
    handlelength=1.0,
    handletextpad=0.35,
    columnspacing=0.9,
    labelspacing=0.20
)
leg.set_in_layout(True)
frame = leg.get_frame()
frame.set_facecolor("white"); frame.set_edgecolor("black")
frame.set_linewidth(0.6); frame.set_alpha(0.85)

plt.savefig("Fig_QEC_Coherence_PRL_border-top-legend_wrap.pdf",  bbox_inches="tight")
plt.savefig("Fig_QEC_Coherence_PRL_border-top-legend_wrap.png", dpi=600, bbox_inches="tight")
plt.savefig("Fig_QEC_Coherence_PRL_border-top-legend_wrap.eps",  bbox_inches="tight")
plt.show()
