"""
=============================================================
 MAGNETORQUER COIL DESIGN TOOL (1U CubeSat Edition)
=============================================================
"""

import math

MU_0 = 4 * math.pi * 1e-7

CORES: dict[str, tuple[int, str]] = {
    "Air":          (1,      "Air core (PCB perimeter trace)"),
    "CK30":         (2000,   "Soft magnetic alloy"),
    "Ferrite":      (1000,   "Soft ferrite (MnZn / NiZn)"),
    "Mu_Metal":     (20000,  "Mu-metal — highest perm, brittle"),
}

WIRES: dict[str, tuple[float, str]] = {
    "Copper":   (1.68e-8, "Enameled Cu magnet wire"),
}

def awg_from_mm(d_mm: float) -> int | None:
    """Convert wire diameter [mm] to the nearest AWG gauge."""
    if d_mm <= 0: return None
    return round(36.0 - 39.0 * math.log(d_mm / 0.127) / math.log(92))

def is_practical_awg(awg: int) -> bool:
    """True if AWG falls within a physically windable range."""
    return awg is not None and 18 <= awg <= 40

def test_magnetorquer(
    core_material: str,
    coil_radius: float,
    core_length: float,
    target_moment: float = 0.016,
    supply_voltage: float = 5.0,
    max_current: float = 0.0500,
    wire_material: str = "Copper"
):
    """
    Modular function to test different magnetorquer configurations.
    """
    m_req = target_moment
    V = supply_voltage
    I_max = max_current
    r = coil_radius
    l_core = core_length
    mu_r, core_desc = CORES[core_material]
    rho, _ = WIRES[wire_material]

    A_coil = math.pi * (r ** 2)
    l_r = l_core / r

    # 1. Demagnetizing Factor & Effective Permeability
    if mu_r == 1:
        # Air core has no demagnetizing penalty
        N_d = 0
        mu_eff = 1
    else:
        if l_r <= math.e:
            return {"error": f"Core length must be >> radius (l/r > 2.71). Current: {l_r:.1f}"}
        
        N_d = (4 * (math.log(l_r) - 1)) / ((l_r ** 2) - 4 * math.log(l_r))
        mu_eff = 1 + ((mu_r - 1) / (1 + (mu_r - 1) * N_d))

    # 2. Number of Turns (N)
    N_exact = m_req / (mu_eff * I_max * A_coil)
    N = math.ceil(N_exact) if N_exact >= 1.0 else 1
    I_op = m_req / (mu_eff * N * A_coil) 

    # 3. Electrical Resistance & Wire Geometry
    R_op = V / I_op                           
    L_wire = N * 2.0 * math.pi * r            
    A_wire = rho * L_wire / R_op              
    d_wire = 2.0 * math.sqrt(A_wire / math.pi)
    d_mm = d_wire * 1e3
    awg = awg_from_mm(d_mm)
    
    # Power if supplying Max Current (As requested)
    P_max_I = V * I_max

    # 4. Inductance
    if mu_r == 1:
        # Nagaoka approximation for short air coils
        l_coil = max(N * d_wire, 1e-9)
        K_N = 1.0 / (1.0 + 0.9 * r / l_coil)
        L_H = MU_0 * (N**2) * A_coil / l_coil * K_N
    else:
        L_H = (MU_0 * math.pi * (r**2) * (N**2)) / (l_core * ((1/mu_r) + N_d))

    # Output Formatting
    print(f"\n{'='*50}")
    print(f" TEST RESULTS: {core_material.upper()} CORE")
    print(f" Radius: {r*1000:.1f} mm | Length: {l_core*1000:.1f} mm")
    print(f"{'='*50}")
    print(f" Target Moment    : {m_req} Am²")
    print(f" Effective Perm.  : {mu_eff:.2f} (Bulk: {mu_r})")
    print(f" Required Turns   : {N}")
    print(f" Total Wire Length: {L_wire:.3f} meters")
    print(f" Wire Gauge (AWG) : {awg} " + ("(WARNING: Too thin to make!)" if not is_practical_awg(awg) else "(Physically possible)"))
    print(f" Operating Current: {I_op*1000:.1f} mA (Limit: {I_max*1000:.1f} mA)")
    print(f" Max Power Draw   : {P_max_I:.2f} W")
    print(f" Inductance       : {L_H*1e6:.2f} µH")
    print(f"{'='*50}")

# ──────────────────────────────────────────────────────────
# RUNNING THE TESTS WITH "PERFECT" DIMENSIONS
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: AIR CORE (Large radius, short length)
    test_magnetorquer("Air", coil_radius=0.040, core_length=0.005)

    # Test 2: FERRITE ROD (CubeSat Form Factor: 7cm long, 3mm radius)
    test_magnetorquer("Ferrite", coil_radius=0.003, core_length=0.070)

    # Test 3: MU-METAL ROD (CubeSat Form Factor)
    test_magnetorquer("Mu_Metal", coil_radius=0.003, core_length=0.070)