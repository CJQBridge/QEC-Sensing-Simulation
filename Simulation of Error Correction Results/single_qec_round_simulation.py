# -*- coding: utf-8 -*-
"""
Simulate QEC-driven open-system time evolution of an encoded two-qubit state
and plot the |0⟩ population versus scaled time t/T2 using PRL-like aesthetics.
Forced QEC events are shown as vertical dashed lines with in-axis labels; legend spans the top.
"""

import numpy as np
import scipy.linalg as sla
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

# =================== PRL style / size & fonts ===================
SIZE_MODE  = 'single'
WIDTH_IN   = 4 if SIZE_MODE=='single' else 7.0
HEIGHT     = WIDTH_IN * 0.68

FONT_SCALE = 1.9
BASE_AX    = 8 if SIZE_MODE=='single' else 9
BASE_TK    = 7 if SIZE_MODE=='single' else 8
LEGEND_NCOL = 2

mpl.rcParams.update({
    "font.family": ["Arial","Helvetica","DejaVu Sans"],
    "axes.labelsize": BASE_AX * FONT_SCALE,
    "legend.fontsize": BASE_AX * FONT_SCALE,
    "xtick.labelsize": BASE_TK * FONT_SCALE,
    "ytick.labelsize": BASE_TK * FONT_SCALE,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.6, "ytick.major.size": 3.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

# =================== Physics / simulation ===================
Omega      = 1.0 * np.pi * 1e6   # rad/s
T2         = 10e-6               # s
total_time = 30e-6               # s
dt_seg     = 5e-7                # s
n_steps    = int(total_time / dt_seg)
n_t_seg    = 100
ORDER      = "F"

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex); I4 = np.eye(4, dtype=complex)

H   = (Omega / 2) * np.kron(I2, sigma_x)
L_B = np.sqrt(1/(2*T2)) * np.kron(I2, sigma_z)
K   = L_B.conj().T @ L_B
Lc  = -1j * (np.kron(I4, H) - np.kron(H.T, I4))
Ld  = np.kron(L_B, L_B.conj()) - 0.5*np.kron(I4, K) - 0.5*np.kron(K.T, I4)
L_super = Lc + Ld

z_gate  = np.kron(I2, sigma_z)
P_code  = 0.5*np.array([[1,0,0,1],[0,1,1,0],[0,1,1,0],[1,0,0,1]], dtype=complex)
P_error = 0.5*np.array([[1,0,0,-1],[0,1,-1,0],[0,-1,1,0],[-1,0,0,1]], dtype=complex)

def _apply_qec(rhoM):
    rhoM = P_error @ rhoM @ P_error
    rhoM /= (np.real(np.trace(rhoM)) + 1e-16)
    rhoM = z_gate @ rhoM @ z_gate.conj().T
    return rhoM

def run_one(seed=None, forced_qec_xt=None, forced_window=None, forced_count=0, stochastic=False):
    rng = np.random.default_rng(seed)
    rho = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype=complex).reshape(-1, order=ORDER)
    t_s, P00, qec_s = [0.0], [0.5], []
    P_small = sla.expm(L_super * (dt_seg / n_t_seg))

    targets_x = []
    if forced_qec_xt: targets_x = sorted(list(forced_qec_xt))
    elif forced_window is not None and forced_count > 0:
        x0, x1 = forced_window
        targets_x = list(np.linspace(x0, x1, forced_count + 2)[1:-1])
    targets_s = [x * T2 for x in targets_x]
    next_idx  = 0
    tol       = dt_seg * 0.6

    for _ in range(n_steps):
        for _ in range(n_t_seg):
            rho  = P_small @ rho
            rhoM = rho.reshape(4,4,order=ORDER)
            t_s.append(t_s[-1] + dt_seg/n_t_seg)
            P00.append(float(np.real(rhoM[0,0])))

        rhoM  = rho.reshape(4,4,order=ORDER)
        t_now = t_s[-1]
        forced_done = False

        if next_idx < len(targets_s):
            if abs(t_now - targets_s[next_idx]) <= tol or (t_now - targets_s[next_idx]) > 0:
                rhoM = _apply_qec(rhoM); qec_s.append(t_now)
                next_idx += 1; forced_done = True

        if not forced_done:
            if stochastic:
                p_err = float(np.clip(np.real(np.trace(P_error @ rhoM)), 0.0, 1.0))
                if rng.random() < p_err:
                    rhoM = _apply_qec(rhoM); qec_s.append(t_now)
                else:
                    rhoM = P_code @ rhoM @ P_code; rhoM /= (np.real(np.trace(rhoM)) + 1e-16)
            else:
                rhoM = P_code @ rhoM @ P_code; rhoM /= (np.real(np.trace(rhoM)) + 1e-16)

        rho = rhoM.reshape(-1, order=ORDER)

    return np.asarray(t_s), np.asarray(P00), np.asarray(qec_s)

def run_no_qec():
    rho = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype=complex).reshape(-1, order=ORDER)
    t_s, P00 = [0.0], [0.5]
    P_small = sla.expm(L_super * (dt_seg / n_t_seg))
    for _ in range(n_steps):
        for _ in range(n_t_seg):
            rho  = P_small @ rho
            rhoM = rho.reshape(4,4,order=ORDER)
            t_s.append(t_s[-1] + dt_seg/n_t_seg)
            P00.append(float(np.real(rhoM[0,0])))
    return np.asarray(t_s), np.asarray(P00)

# ===== simulate =====
times_s_qec, P00_qec, qec_s = run_one(forced_window=(1.0, 2.2), forced_count=2, stochastic=False)
times_s_noqec, P00_noqec    = run_no_qec()

x_qec    = times_s_qec  / T2
x_noqec  = times_s_noqec/ T2
x_qec_v  = qec_s / T2

# =================== plotting ===================
blue_edge   = "#1F5AA6"
noqec_edge  = "#B35C00"

fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT), layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.18, hspace=0.02)

ax.plot(x_noqec, P00_noqec, lw=1.7, alpha=0.95, color=noqec_edge, zorder=4, label="No QEC")
ax.plot(x_qec,    P00_qec,  lw=1.9, color=blue_edge, zorder=5, label="With QEC")

xmax = max(x_qec[-1], x_noqec[-1])
ax.set_xlim(0, xmax)
ax.set_ylim(0.0, 0.5)

for xv in x_qec_v:
    ax.axvline(xv, color="#6a6a6a", lw=1.05, ls="--", alpha=0.9, zorder=3)

TEXT_Y_FRAC = 0.06
x_data_to_y_axes = blended_transform_factory(ax.transData, ax.transAxes)

if x_qec_v.size:
    MIN_SEP, MAX_ANN = 0.20, 6
    sel = []
    for xv in x_qec_v:
        if not sel or (xv - sel[-1]) >= MIN_SEP:
            sel.append(xv)
        if len(sel) >= MAX_ANN:
            break

    y0 = ax.get_ylim()[0]
    for xv in sel:
        ax.annotate("QEC",
                    xy=(xv, y0 + 1e-5), xycoords="data",
                    xytext=(xv, TEXT_Y_FRAC), textcoords=x_data_to_y_axes,
                    ha="center", va="bottom",
                    fontsize=mpl.rcParams["axes.labelsize"],
                    bbox=dict(facecolor="white", edgecolor="#cfcfcf",
                              boxstyle="round,pad=0.24", linewidth=1.0),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#444444"),
                    annotation_clip=False, zorder=20)

ax.set_xlabel(r"$t/T_2$")
ax.set_ylabel("0-state population")

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(
    handles, labels,
    loc="lower left",
    bbox_to_anchor=(0.0, 1.01, 1.0, 0.001),
    bbox_transform=ax.transAxes,
    mode="expand",
    ncol=min(LEGEND_NCOL, len(labels)),
    frameon=True, fancybox=False,
    handlelength=1.6, handletextpad=0.45,
    columnspacing=0.9, labelspacing=0.22
)
leg.set_in_layout(True)
frame = leg.get_frame()
frame.set_facecolor("white"); frame.set_edgecolor("black")
frame.set_linewidth(0.7); frame.set_alpha(0.85)

# =================== CSV export (wide format) ===================
import os
import pandas as pd

out_dir = r"C:\Users\23600\Desktop\QEC阶段性总结\pythonProject\.venv"
out_name = "Fig2_SingleQEC.csv"

def _pad(arr, n):
    a = np.full(n, np.nan)
    a[:arr.size] = arr
    return a

max_len = max(x_qec.size, x_noqec.size, x_qec_v.size if x_qec_v.size else 0)
data = {
    "WithQEC__x":   _pad(x_qec, max_len),
    "WithQEC__y":   _pad(P00_qec, max_len),
    "WithQEC__yerr": np.full(max_len, np.nan),
    "NoQEC__x":     _pad(x_noqec, max_len),
    "NoQEC__y":     _pad(P00_noqec, max_len),
    "NoQEC__yerr":  np.full(max_len, np.nan),
    "QEC_events__x": _pad(x_qec_v, max_len),
}
df = pd.DataFrame(data)
os.makedirs(out_dir, exist_ok=True)
df.to_csv(os.path.join(out_dir, out_name), index=False, encoding="utf-8-sig")

plt.show()
