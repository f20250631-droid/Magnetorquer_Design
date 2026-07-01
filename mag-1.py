"""
Magnetorquer Interactive Dashboard — CubeSat Edition (Combined Design & GUI)
==========================================================================
Integrates all rigorous physics and feasibility checks from the design file
into the interactive 6-plot dashboard GUI.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, RadioButtons

# ─────────────────────────────────────────────────────────
# CONSTANTS & DATABASES
# ─────────────────────────────────────────────────────────
WIRES = {
    "Copper":    (1.68e-8, 8960.0),  # (ρ [Ω·m], density [kg/m³])
    "Aluminium": (2.82e-8, 2700.0),
}

CORES = {
    "Air":       1.0,
    "Ferrite":   1000.0,
    "Si-Steel":  5000.0,
    "Permalloy": 8000.0,
}

SIGMA_SB      = 5.67e-8          # W/(m²·K⁴)
MU_0          = 4e-7 * math.pi   # H/m
B_LEO_REF_T   = 30e-6            # T

# CubeSat design limits
I_LIMIT_A     = 0.50             # 500 mA
P_LIMIT_W     = 2.00             # 2 W
T_LIMIT_C     = 60.0             # °C
MASS_LIMIT_G  = 50.0             # g
EMISSIVITY    = 0.85

# ─────────────────────────────────────────────────────────
# COLOUR THEME
# ─────────────────────────────────────────────────────────
C_BG          = "#0d1117"
C_PANEL       = "#161b22"
C_GRID        = "#21262d"
C_TEXT        = "#e6edf3"
C_DIM         = "#8b949e"
C_ACCENT1     = "#58a6ff"
C_ACCENT2     = "#3fb950"    # Green (Pass)
C_ACCENT3     = "#bc8cff"
C_ACCENT4     = "#ff7b72"    # Red (Fail)
C_ACCENT5     = "#ffa657"
C_DOT         = "#f85149"
C_LIMIT       = "#ff4500"

LINEWIDTH     = 2.0
DOT_SIZE      = 80

# ─────────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────
def awg_from_mm(d_mm):
    if d_mm <= 0: return None
    return round(36.0 - 39.0 * math.log(d_mm / 0.127) / math.log(92))

def is_practical_awg(awg):
    return awg is not None and 18 <= awg <= 46

def electrical_props(voltage, r_coil, r_wire, n_turns, rho):
    L = n_turns * 2.0 * math.pi * r_coil
    A = math.pi * r_wire**2
    R = rho * L / A
    I = voltage / R
    P = voltage * I
    return L, R, I, P

def magnetic_moment(voltage, r_coil, r_wire, rho, mu_r):
    # m = mu_r * N * I * A_coil; N cancels out when I = V/R
    return mu_r * (voltage * math.pi * r_coil * r_wire**2) / (2.0 * rho)

def inductance_properties(n_turns, r_coil, l_coil, mu_r):
    if l_coil <= 1e-12: return 0.0
    A_coil = math.pi * r_coil**2
    K_N = 1.0 / (1.0 + 0.9 * r_coil / l_coil) # Nagaoka correction
    return (MU_0 * mu_r * (n_turns**2) * A_coil / l_coil) * K_N

def wire_mass(L_wire, r_wire, density):
    return density * (math.pi * r_wire**2) * L_wire

def fill_factor(r_wire, r_coil, l_coil, n_turns):
    A_wire = math.pi * r_wire**2
    A_bobbin = l_coil * (0.5 * r_coil)
    return (n_turns * A_wire) / A_bobbin

def steady_state_temp(power_W, r_coil, l_coil, env_temp_c):
    A_surface = (2.0 * math.pi * r_coil * l_coil) + (2.0 * math.pi * r_coil**2)
    T_K4 = (power_W / (EMISSIVITY * SIGMA_SB * A_surface)) + (env_temp_c + 273.15)**4
    return T_K4**0.25 - 273.15

# ─────────────────────────────────────────────────────────
# FIGURE LAYOUT
# ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   C_BG,  "axes.facecolor":     C_PANEL,
    "axes.edgecolor":     C_GRID,"axes.labelcolor":    C_DIM,
    "axes.titlecolor":    C_TEXT,"xtick.color":        C_DIM,
    "ytick.color":        C_DIM, "grid.color":         C_GRID,
    "grid.linewidth":     0.8,   "text.color":         C_TEXT,
    "font.family":        "monospace", "axes.spines.top": False,
    "axes.spines.right":  False,
})

fig = plt.figure(figsize=(20, 12))
fig.canvas.manager.set_window_title("Magnetorquer Dashboard · Comprehensive")

outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4, 4.5], hspace=0.20)

# ── Top row: 6 plots ──
plot_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0], hspace=0.55, wspace=0.35)
ax_nR   = fig.add_subplot(plot_gs[0, 0])
ax_mRw  = fig.add_subplot(plot_gs[0, 1])
ax_IR   = fig.add_subplot(plot_gs[0, 2])
ax_pN   = fig.add_subplot(plot_gs[1, 0])
ax_TN   = fig.add_subplot(plot_gs[1, 1])
ax_ff   = fig.add_subplot(plot_gs[1, 2])
AXES = [ax_nR, ax_mRw, ax_IR, ax_pN, ax_TN, ax_ff]

# ── Bottom section ──
bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[3.5, 2.5], wspace=0.04)
ax_telem = fig.add_subplot(bottom_gs[0, 1])
ax_telem.set_facecolor(C_BG)
ax_telem.axis("off")

# ── Sliders & Radio Buttons ──
SL_L, SL_W, SL_H = 0.08, 0.35, 0.03
sl_defs = [
    ("Voltage (V)",          3.0, 7.0,   5.0,   None,  0.86),
    ("Wire Radius (mm)",     0.05, 1.0,  0.12,  None,  0.81), # Start at 0.12mm (AWG 31) for feasible default
    ("Number of Turns N",    50,  2000,  500,   1,     0.76),
    ("Coil Radius (cm)",     0.5, 5.0,   1.5,   None,  0.71),
    ("Coil Length (cm)",     1.0, 15.0,  4.0,   None,  0.66),
    ("Env Temp (°C)",       -30,  80,    20,    1,     0.61),
]

sliders = {}
for label, vmin, vmax, vinit, vstep, bottom in sl_defs:
    ax_sl = plt.axes([SL_L, bottom * 0.48, SL_W, SL_H], facecolor=C_GRID)
    sl = Slider(ax_sl, "", vmin, vmax, valinit=vinit, color=C_ACCENT1, track_color=C_GRID)
    if vstep: sl.valstep = vstep
    fig.text(SL_L - 0.01, bottom * 0.48 + SL_H / 2, label, ha="right", va="center", color=C_DIM, fontsize=9)
    sliders[label] = sl

ax_radio_w = plt.axes([0.48, 0.35, 0.08, 0.08], facecolor=C_PANEL)
radio_w = RadioButtons(ax_radio_w, list(WIRES.keys()), activecolor=C_ACCENT1)
ax_radio_w.set_title("Wire", color=C_DIM, fontsize=9, pad=2)

ax_radio_c = plt.axes([0.48, 0.20, 0.08, 0.12], facecolor=C_PANEL)
radio_c = RadioButtons(ax_radio_c, list(CORES.keys()), activecolor=C_ACCENT1)
ax_radio_c.set_title("Core", color=C_DIM, fontsize=9, pad=2)

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, pad=5, color=C_TEXT, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=C_DIM)
    ax.set_ylabel(ylabel, fontsize=9, color=C_DIM)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)

def hline(ax, y, color=C_LIMIT, label=None):
    ax.axhline(y, color=color, linewidth=1.2, linestyle="--", alpha=0.8, label=label)

# ─────────────────────────────────────────────────────────
# TELEMETRY PANEL RENDERER
# ─────────────────────────────────────────────────────────
def render_telemetry(v):
    ax_telem.cla()
    ax_telem.axis("off")

    # Feasibility Checks
    c_I    = v['I'] <= I_LIMIT_A
    c_P    = v['P'] <= P_LIMIT_W
    c_T    = v['T_c'] <= T_LIMIT_C
    c_Mass = (v['mass']*1000) <= MASS_LIMIT_G
    c_Fill = v['ff'] <= 1.0
    c_AWG  = is_practical_awg(v['awg'])
    c_1U   = v['r_coil'] <= 0.045
    all_ok = all([c_I, c_P, c_T, c_Mass, c_Fill, c_AWG, c_1U])

    def color(condition): return C_ACCENT2 if condition else C_ACCENT4
    def sym(condition):   return "[✓]" if condition else "[✗]"

    lines = [
        ("══  MAGNETORQUER DESIGN  ══", C_ACCENT1, True),
        ("", C_TEXT, False),
        ("── MAGNETIC & CORE ──────────", C_DIM, False),
        (f"  Core Material        {v['core']} (μᵣ = {v['mu_r']})", C_TEXT, False),
        (f"  Magnetic Moment      {v['moment']:.5f}  A·m²", C_ACCENT5, False),
        (f"  Peak Torque @ LEO    {v['moment']*B_LEO_REF_T*1e6:.4f}  µN·m", C_ACCENT5, False),
        (f"  Inductance (Nagaoka) {v['L_ind']*1000:.4f}  mH", C_TEXT, False),
        (f"  L/R Time Const (τ)   {v['tau']*1000:.4f}  ms", C_TEXT, False),
        ("", C_TEXT, False),
        ("── ELECTRICAL & WIRE ────────", C_DIM, False),
        (f"  Wire Material        {v['wire']} (AWG {v['awg']})", color(c_AWG), False),
        (f"  Wire Diameter        {v['rw']*2000:.3f}  mm", C_TEXT, False),
        (f"  Resistance           {v['R']:.3f}  Ω", C_TEXT, False),
        (f"  Current              {v['I']*1000:.2f}  mA", color(c_I), False),
        (f"  Power                {v['P']*1000:.2f}  mW", color(c_P), False),
        ("", C_TEXT, False),
        ("── THERMAL & GEOMETRY ───────", C_DIM, False),
        (f"  Coil Temp (vac)      {v['T_c']:.1f}  °C", color(c_T), False),
        (f"  Wire Length          {v['L']:.3f}  m", C_TEXT, False),
        (f"  Wire Mass            {v['mass']*1000:.2f}  g", color(c_Mass), False),
        (f"  Fill Factor          {v['ff']:.3f}", color(c_Fill), False),
        ("", C_TEXT, False),
        ("── FEASIBILITY CHECKS ───────", C_DIM, False),
        (f"  {sym(c_I)} Current  <= {I_LIMIT_A*1000:.0f} mA", color(c_I), True),
        (f"  {sym(c_P)} Power    <= {P_LIMIT_W*1000:.0f} mW", color(c_P), True),
        (f"  {sym(c_T)} Temp     <= {T_LIMIT_C:.0f} °C", color(c_T), True),
        (f"  {sym(c_Mass)} Mass     <= {MASS_LIMIT_G:.0f} g", color(c_Mass), True),
        (f"  {sym(c_Fill)} Fill Fct <= 1.0", color(c_Fill), True),
        (f"  {sym(c_AWG)} AWG in 18-46 (is {v['awg']})", color(c_AWG), True),
        (f"  {sym(c_1U)} Fits 1U  (r <= 4.5cm)", color(c_1U), True),
        ("", C_TEXT, False),
        (f"  OVERALL STATUS: {'PASS' if all_ok else 'FAIL'}", color(all_ok), True),
    ]

    y, dy = 1.0, 0.030
    for txt, col, bold in lines:
        if txt:
            ax_telem.text(0.02, y, txt, transform=ax_telem.transAxes, color=col,
                          fontsize=10.5, fontfamily="monospace",
                          fontweight="bold" if bold else "normal", va="top")
        y -= dy

# ─────────────────────────────────────────────────────────
# MAIN UPDATE FUNCTION
# ─────────────────────────────────────────────────────────
def update(_val=None):
    v       = sliders["Voltage (V)"].val
    rw      = sliders["Wire Radius (mm)"].val / 1000.0
    N       = int(sliders["Number of Turns N"].val)
    r_coil  = sliders["Coil Radius (cm)"].val / 100.0
    l_coil  = sliders["Coil Length (cm)"].val / 100.0
    T_env   = sliders["Env Temp (°C)"].val

    w_mat   = radio_w.value_selected
    c_mat   = radio_c.value_selected
    rho, density = WIRES[w_mat]
    mu_r    = CORES[c_mat]

    L, R, I, P = electrical_props(v, r_coil, rw, N, rho)
    m          = magnetic_moment(v, r_coil, rw, rho, mu_r)
    L_ind      = inductance_properties(N, r_coil, l_coil, mu_r)
    T_c        = steady_state_temp(P, r_coil, l_coil, T_env)

    vals = dict(
        wire=w_mat, core=c_mat, mu_r=mu_r, rw=rw, r_coil=r_coil, L=L, R=R, I=I, P=P,
        moment=m, L_ind=L_ind, tau=(L_ind/R if R>0 else 0), mass=wire_mass(L, rw, density),
        ff=fill_factor(rw, r_coil, l_coil, N), T_c=T_c, awg=awg_from_mm(rw*2000)
    )

    for ax in AXES: ax.cla()

    # 1. N vs Coil Radius (Constant Wire Length)
    rc_arr = np.linspace(0.005, 0.06, 100)
    ax_nR.plot(rc_arr * 100, L / (2.0 * math.pi * rc_arr), color=C_ACCENT3, lw=LINEWIDTH)
    ax_nR.scatter([r_coil * 100], [N], color=C_DOT, s=DOT_SIZE, zorder=5)
    style_ax(ax_nR, "Turns vs Coil Radius\n(Constant Wire Length)", "Coil Radius (cm)", "N (turns)")

    # 2. Moment vs Wire Radius
    rw_arr = np.linspace(0.05e-3, 1.0e-3, 100)
    ax_mRw.plot(rw_arr * 1e3, magnetic_moment(v, r_coil, rw_arr, rho, mu_r), color=C_ACCENT5, lw=LINEWIDTH)
    ax_mRw.scatter([rw * 1e3], [m], color=C_DOT, s=DOT_SIZE, zorder=5)
    style_ax(ax_mRw, "Moment vs Wire Radius", "Wire Radius (mm)", "Moment (A·m²)")

    # 3. Current vs Wire Radius
    I_arr = v / (rho * (N * 2.0 * math.pi * r_coil) / (math.pi * rw_arr**2))
    ax_IR.plot(rw_arr * 1e3, I_arr * 1e3, color=C_ACCENT1, lw=LINEWIDTH)
    ax_IR.scatter([rw * 1e3], [I * 1e3], color=C_DOT, s=DOT_SIZE, zorder=5)
    hline(ax_IR, I_LIMIT_A * 1e3, label="500 mA limit")
    style_ax(ax_IR, "Current vs Wire Radius", "Wire Radius (mm)", "Current (mA)")

    # 4. Power vs N
    N_arr = np.arange(50, 2001, 10)
    P_N = v * (v / (rho * (N_arr * 2.0 * math.pi * r_coil) / (math.pi * rw**2)))
    ax_pN.plot(N_arr, P_N * 1000, color=C_ACCENT1, lw=LINEWIDTH)
    ax_pN.scatter([N], [P * 1000], color=C_DOT, s=DOT_SIZE, zorder=5)
    hline(ax_pN, P_LIMIT_W * 1000, label="2000 mW limit")
    style_ax(ax_pN, "Power vs Turns", "N (turns)", "Power (mW)")

    # 5. Temp vs N
    T_N = [steady_state_temp(p, r_coil, l_coil, T_env) for p in P_N]
    ax_TN.plot(N_arr, T_N, color=C_ACCENT1, lw=LINEWIDTH)
    ax_TN.scatter([N], [T_c], color=C_DOT, s=DOT_SIZE, zorder=5)
    hline(ax_TN, T_LIMIT_C, label="60 °C limit")
    style_ax(ax_TN, "Temp vs Turns (vacuum)", "N (turns)", "Temp (°C)")

    # 6. Fill Factor vs N
    ff_arr = [fill_factor(rw, r_coil, l_coil, n) for n in N_arr]
    ax_ff.plot(N_arr, ff_arr, color=C_ACCENT1, lw=LINEWIDTH)
    ax_ff.scatter([N], [vals['ff']], color=C_DOT, s=DOT_SIZE, zorder=5)
    hline(ax_ff, 1.0, label="Fill = 1.0")
    style_ax(ax_ff, "Fill Factor vs Turns", "N (turns)", "Fill Ratio")

    render_telemetry(vals)
    fig.canvas.draw_idle()

for sl in sliders.values(): sl.on_changed(update)
radio_w.on_clicked(update)
radio_c.on_clicked(update)

fig.text(0.50, 0.96, "MAGNETORQUER DESIGN DASHBOARD — CubeSat ADCS", ha="center", va="top", color=C_ACCENT1, fontsize=15, fontweight="bold")
fig.text(0.50, 0.93, "Fully integrates core physics, wire mass, inductance (Nagaoka), and strict feasibility bounds.", ha="center", va="top", color=C_DIM, fontsize=10)
fig.text(SL_L - 0.01, 0.44, "PARAMETERS & MATERIALS", ha="right", va="top", color=C_ACCENT1, fontsize=11, fontweight="bold")

update()
plt.show()