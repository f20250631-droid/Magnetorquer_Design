import math
import numpy as np
import matplotlib.pyplot as plt


# --- System Configurations ---
MATERIALS = {
    "air":         {"mu_r": 1.0,       "density_kg_m3": 0.0,     "B_sat_T": None},
    "ferrite":     {"mu_r": 2_000.0,   "density_kg_m3": 4_800.0, "B_sat_T": 0.40},
    "soft_iron":   {"mu_r": 5_000.0,   "density_kg_m3": 7_870.0, "B_sat_T": 2.10},
    "permalloy":   {"mu_r": 25_000.0,  "density_kg_m3": 8_700.0, "B_sat_T": 0.75},
    "mu_metal":    {"mu_r": 50_000.0,  "density_kg_m3": 8_740.0, "B_sat_T": 0.75},
    "supermalloy": {"mu_r": 100_000.0, "density_kg_m3": 8_800.0, "B_sat_T": 0.80},
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
I_MAX        = 0.5               # Maximum current (A)
m_target     = 0.016             # Target magnetic moment (A·m^2)
rho_wire_20  = 1.68e-8           # Copper resistivity @ 20 C (ohm·m)
alpha_cu     = 0.00393           # Copper temp coeff of resistivity (1/K)
T_ref_c      = 20.0              # reference temp for rho_wire_20 (C)
wire_density = 8960              # Copper density (kg/m^3)
mu0          = 4 * math.pi * 1e-7

# thermal (radiative, vacuum) — verified from mag-1.py
SIGMA_SB     = 5.67e-8         # W/(m^2·K^4)
EMISSIVITY   = 0.85            # anodised/painted surface (0..1)

# feasibility limits
V_MAX        = 5.0            # supply headroom (V)
P_MAX        = 1.0            # per-rod power budget (W) — 3 axes must share bus
T_MAX_C      = 60.0           # coil hot limit (C)
SAT_MARGIN   = 0.80           # keep B_core below 80% of B_sat (linear model valid)

# 3U envelope (10 x 10 x 34 cm). Rod along the long axis.
MAX_LEN_M    = 0.25          # usable length along 3U long axis
MAX_OD_M     = 0.020         # max coil outer diameter we allow to reserve

B_LEO_REF_T  = 30e-6         # geomagnetic field for torque reporting

# --- Optimization Search Space ---
L_CORE_OPTS = np.linspace(0.03, 0.24, 30)        # 3..24 cm (3U long axis)
R_CORE_OPTS = np.linspace(0.002, 0.006, 21)      # 2..6 mm
I_OPTS      = np.linspace(0.05, I_MAX, 46)       # 0.05..0.5 A


# =============================================================================
#  PHYSICS HELPERS
# =============================================================================
def demag_factor(l_core, r_core):
    """Rod demagnetizing factor Nd (Fakhari 2010 eq.3, uses l/r).
    Returns None when the long-rod approximation is invalid (l/r <= e)."""
    x = l_core / r_core
    if x <= math.e:
        return None
    return 4.0 * (math.log(x) - 1.0) / (x**2 - 4.0 * math.log(x))


def apparent_perm_moment(mu_r, Nd):
    """Effective permeability for the MOMENT (paper eq.9 / Code1 / test.py).
    Includes the coil's own contribution: 1 + chi/(1+Nd*chi)."""
    if Nd is None:
        return mu_r
    return 1.0 + (mu_r - 1.0) / (1.0 + Nd * (mu_r - 1.0))


def apparent_perm_flux(mu_r, Nd):
    """Apparent permeability for core FLUX DENSITY (B = mu0*mu_app*H).
    Used for the saturation check.  mu_r / (1 + Nd*(mu_r-1))."""
    if Nd is None:
        return mu_r
    return mu_r / (1.0 + Nd * (mu_r - 1.0))


def resistivity_at(T_c):
    """Copper resistivity at temperature T_c (linear model)."""
    return rho_wire_20 * (1.0 + alpha_cu * (T_c - T_ref_c))


def steady_state_thermal(current, wire_length, wire_area, r_outer, l_core, env_temp_c):
    """Self-consistent radiative steady state in vacuum.
    Iterates because R (hence I^2R) rises with T via copper resistivity.
    Radiation only -> conservative UPPER bound (ignores conduction to structure).
    Returns (T_c, R_hot, power)."""
    A_surface = 2.0 * math.pi * r_outer * l_core + 2.0 * math.pi * r_outer**2
    T_c = env_temp_c
    R = resistivity_at(T_c) * wire_length / wire_area
    for _ in range(60):
        R = resistivity_at(T_c) * wire_length / wire_area
        P = current**2 * R
        T_new = ((P / (EMISSIVITY * SIGMA_SB * A_surface))
                 + (env_temp_c + 273.15)**4) ** 0.25 - 273.15
        if abs(T_new - T_c) < 1e-4:
            T_c = T_new
            break
        T_c = T_new
    R = resistivity_at(T_c) * wire_length / wire_area
    return T_c, R, current**2 * R


# =============================================================================
#  OPTIMIZER
# =============================================================================
def optimize_magnetorquer(is_air_core=False, env_temp_c=20.0):
    feasible = []

    material_keys = ["air"] if is_air_core else [k for k in MATERIALS if k != "air"]

    for mat_name in material_keys:
        mu_r         = MATERIALS[mat_name]["mu_r"]
        core_density = MATERIALS[mat_name]["density_kg_m3"]
        B_sat        = MATERIALS[mat_name]["B_sat_T"]

        for awg, wire_data in WIRE_RADIUS_LIST.items():
            r_wire    = wire_data["radius_m"]
            d_wire    = 2 * r_wire
            wire_area = math.pi * r_wire**2

            for l_core in L_CORE_OPTS:
                if l_core > MAX_LEN_M:
                    continue
                for r_core in R_CORE_OPTS:
                    A_core = math.pi * r_core**2

                    # --- MODEL VALIDITY: long-rod demag formula domain ---
                    Nd = demag_factor(l_core, r_core)      # None if l/r <= e -> invalid
                    if not is_air_core and Nd is None:
                        continue
                    beta   = 1.0 if is_air_core else apparent_perm_moment(mu_r, Nd)
                    mu_flx = 1.0 if is_air_core else apparent_perm_flux(mu_r, Nd)

                    for I in I_OPTS:
                        # Turns needed to meet the target magnetic moment
                        N_turns = math.ceil(m_target / (I * A_core * beta))
                        if N_turns <= 0:
                            continue

                        # --- PACKING CONSTRAINT CHECK ---
                        turns_per_layer = int(l_core / d_wire)
                        if turns_per_layer == 0:
                            continue
                        layers         = math.ceil(N_turns / turns_per_layer)
                        coil_thickness = layers * d_wire
                        r_outer        = r_core + coil_thickness

                        # Geometric guardrails + 3U outer-diameter envelope
                        if coil_thickness > r_core or layers > 15:
                            continue
                        if 2.0 * r_outer > MAX_OD_M:
                            continue

                        # --- Multi-layer Winding Geometry ---
                        wire_length = 0.0
                        remaining   = N_turns
                        for layer in range(1, layers + 1):
                            r_layer = r_core + (layer - 0.5) * d_wire
                            t_in    = min(remaining, turns_per_layer)
                            wire_length += t_in * (2 * math.pi * r_layer)
                            remaining  -= t_in

                        # --- Self-consistent THERMAL + ELECTRICAL ---
                        T_c, R_coil, power_loss = steady_state_thermal(
                            I, wire_length, wire_area, r_outer, l_core, env_temp_c)
                        V_drop = I * R_coil

                        # --- SATURATION guard (linear mu model must stay valid) ---
                        # B_core = mu0 * mu_app_flux * H_drive,  H_drive = N*I / l
                        H_drive = N_turns * I / l_core
                        B_core  = mu0 * mu_flx * H_drive
                        sat_ok  = (B_sat is None) or (B_core <= SAT_MARGIN * B_sat)

                        # --- FEASIBILITY GATES ---
                        if V_drop > V_MAX:      continue
                        if power_loss > P_MAX:  continue
                        if T_c > T_MAX_C:       continue
                        if not sat_ok:          continue

                        # --- Mass & remaining metrics ---
                        mass_coil   = wire_length * wire_area * wire_density
                        mass_core   = (A_core * l_core) * core_density
                        total_mass  = mass_core + mass_coil
                        m_achieved  = N_turns * I * A_core * beta
                        L_coil      = mu0 * A_core * (N_turns**2) / (
                                        l_core * (1.0 / mu_r + (Nd if Nd else 0.0)))
                        torque_leo  = m_achieved * B_LEO_REF_T

                        feasible.append({
                            "material": mat_name, "mu_r": mu_r, "awg": awg,
                            "l_core": l_core, "r_core": r_core, "r_outer": r_outer,
                            "I": I, "N_turns": N_turns, "layers": layers,
                            "Nd": Nd, "beta": beta, "m_achieved": m_achieved,
                            "torque_leo": torque_leo,
                            "mass_core": mass_core, "mass_coil": mass_coil,
                            "total_mass": total_mass, "wire_length": wire_length,
                            "R_coil": R_coil, "V_drop": V_drop, "power_loss": power_loss,
                            "L_coil": L_coil, "coil_thickness": coil_thickness,
                            "T_c": T_c, "B_core": B_core, "B_sat": B_sat,
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
            "N_turns", "wire_length", "m_achieved", "T_c"]
    return {k: (min(d[k] for d in designs), max(d[k] for d in designs)) for k in keys}


# =============================================================================
#  REPORTING
# =============================================================================
def print_design_results(title, design):
    if not design:
        print(f"\n--- {title}: no valid design met the constraints ---")
        return
    bsat = design['B_sat']
    bsat_s = "n/a" if bsat is None else f"{bsat:.2f} T"
    print(f"\n================ {title} ================")
    print(f"Material Chosen     : {design['material'].upper()} (mu_r = {design['mu_r']:.0f})")
    print(f"Wire Size           : {design['awg']}")
    print(f"Core Dimensions     : Length = {design['l_core']*100:.2f} cm, Radius = {design['r_core']*1000:.2f} mm")
    print(f"Coil Outer Dia      : {design['r_outer']*2000:.2f} mm")
    print(f"Operating Current   : {design['I']:.3f} A  (Max Allowable: {I_MAX} A)")
    print(f"Number of Turns     : {design['N_turns']} (wound in {design['layers']} layer(s))")
    print(f"Demag Factor (Nd)   : {design['Nd']:.6f}")
    print(f"Core Boost (beta)   : {design['beta']:.2f}")
    print(f"Magnetic Moment     : {design['m_achieved']:.6f} A·m^2  (Target: {m_target} A·m^2)")
    print(f"Torque @ LEO (30uT) : {design['torque_leo']*1e6:.4f} uN·m")
    print(f"------------------- Physical Metrics -------------------")
    print(f"Core Mass           : {design['mass_core']*1000:.3f} g")
    print(f"Coil Winding Mass   : {design['mass_coil']*1000:.3f} g")
    print(f"Total System Mass   : {design['total_mass']*1000:.3f} g")
    print(f"Total Wire Length   : {design['wire_length']:.3f} m")
    print(f"------------------- Electrical / Thermal ---------------")
    print(f"Coil Resistance(hot): {design['R_coil']:.4f} ohm")
    print(f"Coil Inductance     : {design['L_coil']*1e3:.4f} mH")
    print(f"Voltage Drop        : {design['V_drop']:.4f} V")
    print(f"Power Consumption   : {design['power_loss']:.4f} W")
    print(f"Steady-State Temp   : {design['T_c']:.1f} C  (limit {T_MAX_C:.0f} C)")
    print(f"------------------- Magnetic Validity ------------------")
    print(f"Core Flux B_core    : {design['B_core']*1e3:.1f} mT  (B_sat {bsat_s}, "
          f"margin {SAT_MARGIN*100:.0f}%)")


def print_range_table(title, front, n_show=14):
    if not front:
        print(f"\n################ {title}: no feasible designs ################")
        return
    print(f"\n################ {title}: FEASIBLE DESIGN RANGE ################")
    print(f"(Pareto-optimal designs trading power vs. mass — pick a point along this front)\n")
    step  = max(1, len(front) // n_show)
    shown = front[::step]
    if front[-1] not in shown:
        shown.append(front[-1])
    hdr = (f"{'mat':<11}{'awg':>6}{'L[cm]':>7}{'R[mm]':>7}{'I[A]':>7}{'N':>6}{'lay':>4}"
           f"{'P[mW]':>9}{'mass[g]':>9}{'T[C]':>7}{'V[V]':>7}")
    print(hdr); print("-" * len(hdr))
    for d in shown:
        print(f"{d['material']:<11}{d['awg']:>6}{d['l_core']*100:>7.2f}{d['r_core']*1000:>7.2f}"
              f"{d['I']:>7.3f}{d['N_turns']:>6}{d['layers']:>4}{d['power_loss']*1e3:>9.2f}"
              f"{d['total_mass']*1e3:>9.2f}{d['T_c']:>7.1f}{d['V_drop']:>7.3f}")
    rng = metric_ranges(front)
    print("\nRange across the Pareto front:")
    print(f"  Power        : {rng['power_loss'][0]*1e3:8.2f} – {rng['power_loss'][1]*1e3:8.2f} mW")
    print(f"  Total mass   : {rng['total_mass'][0]*1e3:8.2f} – {rng['total_mass'][1]*1e3:8.2f} g")
    print(f"  Temp         : {rng['T_c'][0]:8.1f} – {rng['T_c'][1]:8.1f} C")
    print(f"  Voltage drop : {rng['V_drop'][0]:8.3f} – {rng['V_drop'][1]:8.3f} V")
    print(f"  Turns        : {rng['N_turns'][0]:8.0f} – {rng['N_turns'][1]:8.0f}")


# =============================================================================
#  EXECUTE
# =============================================================================
if __name__ == "__main__":
    ENV_TEMP_C = 20.0

    air_feasible   = optimize_magnetorquer(is_air_core=True,  env_temp_c=ENV_TEMP_C)
    ferro_feasible = optimize_magnetorquer(is_air_core=False, env_temp_c=ENV_TEMP_C)

    air_front   = pareto_front(air_feasible)
    ferro_front = pareto_front(ferro_feasible)

    air_lowP   = min(air_feasible,   key=lambda d: d["power_loss"]) if air_feasible else None
    ferro_lowP = min(ferro_feasible, key=lambda d: d["power_loss"]) if ferro_feasible else None
    ferro_lowM = min(ferro_feasible, key=lambda d: d["total_mass"]) if ferro_feasible else None

    print_design_results("AIR-CORE  — lowest-power endpoint",  air_lowP)
    print_design_results("FERRO-CORE — lowest-power endpoint", ferro_lowP)
    print_design_results("FERRO-CORE — lowest-mass endpoint",  ferro_lowM)

    print_range_table("AIR-CORE (Air-MT)",    air_front)
    print_range_table("FERRO-CORE (Core-MT)", ferro_front)

    # -------------------------------------------------------------------------
    #  PLOTS
    # -------------------------------------------------------------------------
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
        if front:
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
