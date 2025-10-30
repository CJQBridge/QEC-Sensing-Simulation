# Final-fidelity ratio heatmap (PRL style), x-axis = dt/T2 (dimensionless); value = F_QEC / F_bare
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------- parameters ----------
T2_us    = 10.0     # µs
t_us     = 100.0    # µs
tau_m_us = 1000.0   # µs (kept for parity with previous scripts; unused here)
Tref_us  = 100.0    # µs
Omega    = 2*np.pi/(Tref_us*1e-6)  # rad/s (unused here)

# ---------- dt/T2 up to 1 ----------
dt_us_max = 1.0 * T2_us
dt_us = np.linspace(0.25, dt_us_max, 600)

# ---------- gate error p up to 50% ----------
p_vals = np.linspace(0.0, 0.5, 501)

DT, P = np.meshgrid(dt_us, p_vals)

# ---------- final-fidelity ratio: F_QEC / F_bare ----------
A_bare_sq = np.exp(-2.0*t_us/T2_us)
A_qec_sq  = np.exp(-DT/T2_us) * (1.0 - P)**(2.0*t_us/DT)
rho = A_qec_sq / A_bare_sq
rho = np.clip(rho, 1e-16, 1e16)
logrho = np.log10(rho)

# ---------- break-even curve (F_QEC = F_bare ⇒ ratio = 1) ----------
p_be = []
for d in dt_us:
    base = np.exp(-(2.0*t_us - d)/T2_us)
    if base <= 0.0:
        p_be.append(np.nan); continue
    if base >= 1.0:
        p_be.append(0.0); continue
    p = 1.0 - base**(d/(2.0*t_us))
    p_be.append(np.clip(p, 0.0, 1.0))
p_be = np.array(p_be)

# ---------- PRL styling ----------
SIZE_MODE = "single"                     # "single" (~4 in) or "double" (7.0 in)
W = 7.0 if SIZE_MODE=="double" else 4
H = W * (0.62 if SIZE_MODE=="double" else 0.78)

FONT_SCALE = 1.75
BASE_AX = 9 if SIZE_MODE=="double" else 8
BASE_TK = 8 if SIZE_MODE=="double" else 7
LABEL_PT = BASE_AX * FONT_SCALE
TICK_PT  = BASE_TK * FONT_SCALE

mpl.rcParams.update({
    "font.family": ["Arial","Helvetica","DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": LABEL_PT,
    "legend.fontsize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "axes.linewidth": 1.0,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.6, "ytick.major.size": 3.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150, "savefig.dpi": 300,
})

def prl_box(ax, lw=1.0):
    for s in ("top","right","bottom","left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color("black")
    ax.tick_params(which="both", direction="in",
                   top=True, right=True, left=True, bottom=True,
                   length=3.6, width=lw)

# ---------- colormap (diverging, centered at 0 = parity) ----------
cmap = LinearSegmentedColormap.from_list(
    "BlueWhiteRedWide",
    [
        (0.00, (8/255, 48/255, 107/255)),
        (0.20, (158/255, 202/255, 225/255)),
        (0.44, (0.98, 0.98, 0.98)),
        (0.56, (0.98, 0.98, 0.98)),
        (0.80, (252/255, 187/255, 161/255)),
        (1.00, (103/255, 0, 13/255)),
    ],
    N=256
)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-7.0, vmax=6.0)

# ---------- plot (x = dt/T2) ----------
X   = DT / T2_us
x_be = dt_us / T2_us

fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
pc = ax.pcolormesh(X, P, logrho, shading="auto", cmap=cmap, norm=norm)

ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 0.5)

cbar = fig.colorbar(pc, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}\!\left(F_{\rm QEC}/F_{\rm bare}\right)$", fontsize=LABEL_PT)
cbar.ax.tick_params(labelsize=TICK_PT, width=1.0, length=3.6)
cbar.outline.set_linewidth(1.0)

mask = np.isfinite(p_be) & (p_be <= 0.5)
ax.plot(x_be[mask], p_be[mask], "k--", lw=2.0, label=r"Break-even: $F_{\rm QEC}=F_{\rm bare}$")

p_ref = 0.10
ax.axhline(p_ref, color="0.4", lw=1.6, ls="--", zorder=9)
x0, x1 = ax.get_xlim()
ax.annotate("gate error = 10%",
            xy=(x1*0.98, p_ref), xycoords="data",
            xytext=(0, 6), textcoords="offset points",
            ha="right", va="bottom",
            fontsize=TICK_PT, color="0.25",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            clip_on=True)

ax.set_xlabel(r"$dt/T_2$")
ax.set_ylabel(r"Per-cycle gate error $p$")
prl_box(ax)

leg = ax.legend(loc="upper right", frameon=True, fancybox=False,
                handlelength=1.6, borderaxespad=0.6)
frame = leg.get_frame()
frame.set_facecolor("white"); frame.set_edgecolor("black"); frame.set_linewidth(0.9)
leg.set_zorder(10)

plt.show()
