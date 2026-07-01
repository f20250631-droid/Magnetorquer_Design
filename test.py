import math
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  MAGNETORQUER
# =============================================================================

# --- System Configurations ---
MATERIALS = {
    "air": {"mu_r": 1.0, "density_kg_m3": 0.0},
    "ferrite": {"mu_r": 2_000.0, "density_kg_m3": 4_800.0},
    "soft_iron": {"mu_r": 5_000.0, "density_kg_m3": 7_870.0},
    "permalloy": {"mu_r": 25_000.0, "density_kg_m3": 8_700.0},
    "mu_metal": {"mu_r": 50_000.0, "density_kg_m3": 8_740.0},
    "supermalloy": {"mu_r": 100_000.0, "density_kg_m3": 8_800.0}
}

WIRE_RADIUS_LIST = {
    "AWG20": {"radius_m": 0.000406},
    "AWG22": {"radius_m": 0.000322},
    "AWG24": {"radius_m": 0.000255},
    "AWG26": {"radius_m": 0.000202},
    "AWG28": {"radius_m": 0.000161},
    "AWG30": {"radius_m": 0.000127},
    "AWG32": {"radius_m": 0.000101},
    "AWG34": {"radius_m": 0.000080},
    "AWG36": {"radius_m": 0.000064},
    "AWG38": {"radius_m": 0.000051},
    "AWG40": {"radius_m": 0.000040},
}

# --- Design Constants ---
I_MAX = 0.5               # Maximum current (A)
m_target = 0.016          # Target magnetic moment (A·m^2)
rho_wire = 1.68e-8        # Copper resistivity (ohm·m)
wire_density = 8960       # Copper density (kg/m^3)
mu0 = 4 * math.pi * 1e-7  # Permeability of free space (H/m)

# --- Optimization Search Space ---
# previously: L_CORE_OPTS = [3..9 cm], R_CORE_OPTS = [2..6 mm], I_OPTS = [0.1..0.5]
L_CORE_OPTS = np.linspace(0.03, 0.09, 25)        # 3..9 cm, continuous
R_CORE_OPTS = np.linspace(0.002, 0.006, 21)      # 2..6 mm, continuous
I_OPTS      = np.linspace(0.05, I_MAX, 46)        # 0.05..0.5 A, continuous (kept name)


# =============================================================================
#  OPTIMIZER
# =============================================================================
def optimize_magnetorquer(is_air_core=False):
    feasible = []                                  # <-- collect everything valid

    material_keys = ["air"] if is_air_core else [k for k in MATERIALS.keys() if k != "air"]

    for mat_name in material_keys:
        mu_r = MATERIALS[mat_name]["mu_r"]
        core_density = MATERIALS[mat_name]["density_kg_m3"]

        for awg, wire_data in WIRE_RADIUS_LIST.items():
            r_wire = wire_data["radius_m"]
            d_wire = 2 * r_wire
            wire_area = math.pi * r_wire**2

            for l_core in L_CORE_OPTS:
                for r_core in R_CORE_OPTS:
                    A_core = math.pi * r_core**2
                    x = l_core / r_core

                    # Demagnetization Factor & Apparent Permeability
                    Nd = (4 * (math.log(x) - 1)) / (x**2 - 4 * math.log(x))
                    beta = 1.0 if is_air_core else (1 + (mu_r - 1) / (1 + Nd * (mu_r - 1)))

                    for I in I_OPTS:
                        # Turns needed to meet the target magnetic moment
                        N_turns = math.ceil(m_target / (I * A_core * beta))
                        if N_turns <= 0:
                            continue

                        # --- PACKING CONSTRAINT CHECK ---
                        turns_per_layer = int(l_core / d_wire)
                        if turns_per_layer == 0:
                            continue
                        layers = math.ceil(N_turns / turns_per_layer)
                        coil_thickness = layers * d_wire

                        # Geometric sanity guardrails
                        if coil_thickness > r_core or layers > 15:
                            continue

                        # --- Multi-layer Winding Geometry ---
                        wire_length = 0
                        remaining_turns = N_turns
                        for layer in range(1, layers + 1):
                            r_layer = r_core + (layer - 0.5) * d_wire
                            turns_in_layer = min(remaining_turns, turns_per_layer)
                            wire_length += turns_in_layer * (2 * math.pi * r_layer)
                            remaining_turns -= turns_in_layer

                        # --- Mass & Electrical Analysis ---
                        mass_coil = wire_length * wire_area * wire_density
                        core_volume = A_core * l_core
                        mass_core = core_volume * core_density
                        total_mass = mass_core + mass_coil

                        R_coil = rho_wire * wire_length / wire_area
                        power_loss = (I**2) * R_coil
                        V_drop = I * R_coil
                        if V_drop > 5.0:
                            continue

                        m_achieved = N_turns * I * A_core * beta
                        L_coil = mu0 * A_core * (N_turns**2) / (l_core * (1.0 / mu_r + Nd))

                        feasible.append({
                            "material": mat_name, "mu_r": mu_r, "awg": awg,
                            "l_core": l_core, "r_core": r_core,
                            "I": I, "N_turns": N_turns, "layers": layers,
                            "Nd": Nd, "beta": beta, "m_achieved": m_achieved,
                            "mass_core": mass_core, "mass_coil": mass_coil,
                            "total_mass": total_mass, "wire_length": wire_length,
                            "R_coil": R_coil, "V_drop": V_drop, "power_loss": power_loss,
                            "L_coil": L_coil, "coil_thickness": coil_thickness,
                        })
    return feasible


# =============================================================================
#  RANGE / PARETO EXTRACTION
# =============================================================================
def pareto_front(designs, x="power_loss", y="total_mass"):
    """Lower-left envelope: minimize both power and mass simultaneously."""
    pts = sorted(designs, key=lambda d: (d[x], d[y]))
    front, best_y = [], float("inf")
    for d in pts:
        if d[y] < best_y - 1e-18:
            front.append(d); best_y = d[y]
    return front


def metric_ranges(designs):
    keys = ["power_loss", "total_mass", "V_drop", "R_coil",
            "N_turns", "wire_length", "m_achieved"]
    return {k: (min(d[k] for d in designs), max(d[k] for d in designs)) for k in keys}


# =============================================================================
#  REPORTING
# =============================================================================
def print_design_results(title, design):
    if not design:
        print(f"\n--- {title}: no valid design met the constraints ---")
        return
    print(f"\n================ {title} ================")
    print(f"Material Chosen     : {design['material'].upper()} (mu_r = {design['mu_r']:.0f})")
    print(f"Wire Size           : {design['awg']}")
    print(f"Core Dimensions     : Length = {design['l_core']*100:.2f} cm, Radius = {design['r_core']*1000:.2f} mm")
    print(f"Operating Current   : {design['I']:.3f} A  (Max Allowable: {I_MAX} A)")
    print(f"Number of Turns     : {design['N_turns']} (wound in {design['layers']} layer(s))")
    print(f"Demag Factor (Nd)   : {design['Nd']:.6f}")
    print(f"Core Boost (beta)   : {design['beta']:.2f}")
    print(f"Magnetic Moment     : {design['m_achieved']:.6f} A·m^2  (Target: {m_target} A·m^2)")
    print(f"------------------- Physical Metrics -------------------")
    print(f"Core Mass           : {design['mass_core']:.6f} kg")
    print(f"Coil Winding Mass   : {design['mass_coil']:.6f} kg")
    print(f"Total System Mass   : {design['total_mass']:.6f} kg")
    print(f"Total Wire Length   : {design['wire_length']:.3f} m")
    print(f"------------------- Electrical Metrics -----------------")
    print(f"Coil Resistance     : {design['R_coil']:.4f} Ω")
    print(f"Coil Inductance     : {design['L_coil']*1e3:.4f} mH")
    print(f"Voltage Drop        : {design['V_drop']:.4f} V")
    print(f"Power Consumption   : {design['power_loss']:.4f} W")


def print_range_table(title, front, n_show=14):
    print(f"\n################ {title}: FEASIBLE DESIGN RANGE ################")
    print(f"(Pareto-optimal designs trading power vs. mass — pick a point along this front)\n")
    # thin to ~n_show representative points evenly along the front
    step = max(1, len(front) // n_show)
    shown = front[::step]
    if front[-1] not in shown:
        shown.append(front[-1])
    hdr = (f"{'mat':<11}{'awg':>6}{'L[cm]':>7}{'R[mm]':>7}{'I[A]':>7}{'N':>6}{'lay':>4}"
           f"{'P[mW]':>9}{'mass[g]':>9}{'V[V]':>7}")
    print(hdr); print("-" * len(hdr))
    for d in shown:
        print(f"{d['material']:<11}{d['awg']:>6}{d['l_core']*100:>7.2f}{d['r_core']*1000:>7.2f}"
              f"{d['I']:>7.3f}{d['N_turns']:>6}{d['layers']:>4}{d['power_loss']*1e3:>9.2f}"
              f"{d['total_mass']*1e3:>9.2f}{d['V_drop']:>7.3f}")
    rng = metric_ranges(front)
    print("\nRange across the Pareto front:")
    print(f"  Power        : {rng['power_loss'][0]*1e3:8.2f} – {rng['power_loss'][1]*1e3:8.2f} mW")
    print(f"  Total mass   : {rng['total_mass'][0]*1e3:8.2f} – {rng['total_mass'][1]*1e3:8.2f} g")
    print(f"  Voltage drop : {rng['V_drop'][0]:8.3f} – {rng['V_drop'][1]:8.3f} V")
    print(f"  Turns        : {rng['N_turns'][0]:8.0f} – {rng['N_turns'][1]:8.0f}")


# =============================================================================
#  EXECUTE
# =============================================================================
air_feasible   = optimize_magnetorquer(is_air_core=True)
ferro_feasible = optimize_magnetorquer(is_air_core=False)

air_front   = pareto_front(air_feasible)
ferro_front = pareto_front(ferro_feasible)

# endpoints of each front: lowest-power and lowest-mass designs
air_lowP    = min(air_feasible,   key=lambda d: d["power_loss"]) if air_feasible else None
ferro_lowP  = min(ferro_feasible, key=lambda d: d["power_loss"]) if ferro_feasible else None
air_lowM    = min(air_feasible,   key=lambda d: d["total_mass"]) if air_feasible else None
ferro_lowM  = min(ferro_feasible, key=lambda d: d["total_mass"]) if ferro_feasible else None

print_design_results("AIR-CORE  — lowest-power endpoint",  air_lowP)
print_design_results("FERRO-CORE — lowest-power endpoint", ferro_lowP)
print_design_results("FERRO-CORE — lowest-mass endpoint",  ferro_lowM)

print_range_table("AIR-CORE (Air-MT)",   air_front)
print_range_table("FERRO-CORE (Core-MT)", ferro_front)


# =============================================================================
#  PLOTS
# =============================================================================
import random
random.seed(0)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})
fig, ax = plt.subplots(figsize=(8, 6))

def subsample(feas, n=6000):
    return feas if len(feas) <= n else random.sample(feas, n)

def scatter_front(axis, feas, front, label, c):
    sub = subsample(feas)
    axis.scatter([d["total_mass"]*1e3 for d in sub],
                 [d["power_loss"]*1e3 for d in sub],
                 s=5, alpha=0.10, color=c, edgecolors="none", rasterized=True)
    axis.plot([d["total_mass"]*1e3 for d in front],
              [d["power_loss"]*1e3 for d in front],
              "-", color=c, lw=1.8, label=f"{label} Pareto front")

scatter_front(ax, air_feasible,   air_front,   "Air-core",   "tab:blue")
scatter_front(ax, ferro_feasible, ferro_front, "Ferro-core", "tab:red")
ax.set_xlabel("Total mass [g]"); ax.set_ylabel("Power dissipation [mW]")
ax.set_title("Feasible designs & Pareto front (power vs mass)")
ax.set_yscale("log"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("magnetorquer_analysis.png", bbox_inches="tight")
print("\nSaved plot -> magnetorquer_analysis.png")