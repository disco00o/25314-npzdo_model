import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



##############################################
# Parameters
##############################################

def make_params():
    p = {
        # ----------------------------
        # Grid / domain
        # ----------------------------
        "H": 20.0,      # depth (m)
        "n": 100,       # number of cells

        # ----------------------------
        # Diffusion profile
        # ----------------------------
        "kappa0": 10.0,     # upper-ocean diffusivity scale (m²/day)
        "kappa_min": 5.0,   # deep diffusivity (m²/day)
        "z0": 10.0,         # thermocline depth (m)
        "zeta": 2.0,        # transition thickness (m)

        # ----------------------------
        # Light (constant surface light, vertical attenuation)
        # ----------------------------
        "I0": 1400.0,   # surface light (µmol ph/m²/s)
        "kW": 0.15,     # light attenuation by sea water (1/m)
        "kP": 0.05,     # light attenuation by plankton (m²/mmol N)

        # Toggle: include phytoplankton self-shading in light profile
        "use_phyto_light_damping": True,

        # --- Seasonality ---
        "tMaxSpring": 75.0,    # day of formation of pycnocline
        "zMaxSteep": 4.0,      # exponent for sine function
        "zMixWinter": 18.0,    # winter mixed-layer depth (m)

        # Seasonal surface light
        "use_seasonal_light": True,
        "use_seasonal_diffusivity": True,
        "I0_winter": 0.0,      # winter minimum surface light (µmol ph/m^2/s)
        "tLightMax": 174.0,    # day of maximum light

        # ----------------------------
        # Oxygen
        # ----------------------------
        "yP": 9.0,          # (mmol 02/mmol N)
        "yN": 6.625,        # (mmol 02/mmol N)
        "KO2_surface": 20,  # (m/day)
        "KO2_bottom": 1,    # (m/day)
        "O2_atm": 260,      # (mmol O2/m^3)

        # ----------------------------
        # NPZD biology
        # ----------------------------
        "gP_max": 1.5,  # phytoplankton max growth (1/day)
        "kL": 50.0,     # light half-sat (µmol ph/m²/s)
        "kN": 0.4,      # nutrient half-sat (mmol/m^3)
        "mP": 0.25,     # phyto mortality (1/day)

        "gZ_max": 0.7,  # max grazing (1/day)
        "kZ": 3.0,      # phyto half-sat for grazing (mmol/m^3)
        "epsN": 0.3,    # zooplankton respiration fraction
        "epsD": 0.3,    # zooplankton egestion fraction
        "mZ": 0.2,      # zooplankton mortality (1/day/mmol/m^3)

        "r": 0.05,      # detritus remineralization (1/day)

        # ----------------------------
        # Vertical transport extras
        # ----------------------------
        "wD": 5.0,      # detritus sinking speed (m/day)
        "wP": 0.0,      # phytoplankton sinking speed (m/day)

        # ----------------------------
        # Boundary condition for nutrients at bottom (diffusive)
        # ----------------------------
        "N_bottom": 10.0,  # bottom nutrient concentration (mmol/m^3)

        # ----------------------------
        # Initial conditions (mmol/m^3)
        # ----------------------------
        "N_init": 0,
        "P_init": 10.0,
        "Z_init": 2.0,
        "D_init": 0,
        "O_init": 0,

        # ----------------------------
        # Time integration
        # ----------------------------
        "t0": 0.0,
        "t_end": 365.0 * 2,  # days
        "nt": 400,           # output points
        "rtol": 1e-3,
        "atol": 1e-3,
    }
    return p



##############################################
# Grid
##############################################

def build_grid(p):
    H = p["H"]
    n = int(p["n"])

    dz = H / n
    z = np.arange(dz/2, H + dz/2, dz)          # centers (length n)
    z_iface = np.arange(0.0, H + dz, dz)       # interfaces (length n+1)

    # Diffusion profile at centers
    D = p["kappa_min"] + (p["kappa0"] - p["kappa_min"]) * 0.5 * (1 - np.tanh((z - p["z0"]) / p["zeta"]))

    p["dz"] = dz
    p["z"] = z
    p["z_iface"] = z_iface
    p["D"] = D
    return p



##############################################
# State vector
##############################################

def pack_state(N, P, Z, D, O):
    return np.concatenate([N, P, Z, D, O])


def unpack_state(y, p):
    n = p["n"]
    N = y[0*n:1*n]
    P = y[1*n:2*n]
    Z = y[2*n:3*n]
    D = y[3*n:4*n]
    O = y[4*n:5*n]
    return N, P, Z, D, O



##############################################
# Light profile
##############################################

def surface_light(t, p):
    if p.get("use_seasonal_light", False):
        return 0.5 * (1.0 + np.sin(2.0 * np.pi * t / 365.0)) * (p["I0"] - p["I0_winter"]) + p["I0_winter"]
    else:
        return p["I0"]

def light_profile(p, P, t):
    z = p["z"]
    dz = p["dz"]
    I0t = surface_light(t, p)

    if p.get("use_phyto_light_damping", True):
        return I0t * np.exp(-p["kW"] * z - p["kP"] * np.cumsum(P) * dz)
    else:
        return I0t * np.exp(-p["kW"] * z)



##############################################
# Oxygen
##############################################

def sigmoid(O):
    return 1 / (1 + np.exp(-O))

def tau(O, p):
    return p["r"] * sigmoid(O)



##############################################
# Growth Limitation functions
##############################################

def gP(N, I, p):
    fL = I / (I + p["kL"])
    fN = N / (N + p["kN"])
    return p["gP_max"] * np.minimum(fL, fN)


def gZ(P, O, p):
    return p["gZ_max"] * P / (P + p["kZ"]) * sigmoid(O)



##############################################
# Model
##############################################

def npzd_reactions(N, P, Z, D, O, I, p):
    mu = gP(N, I, p)
    graz = gZ(P, O, p) * Z

    dN = (-mu * P
          + p["epsN"] * graz
          + tau(O, p) * D)

    dP = (mu * P
          - graz
          - p["mP"] * P)

    dZ = ((1.0 - p["epsN"] - p["epsD"]) * graz
          - p["mZ"] * Z**2)

    dD = (p["mP"] * P
          + p["mZ"] * Z
          + p["epsD"] * graz
          - tau(O, p) * D)

    dO = (p["yP"] * mu * P
          - p["yN"] * p["epsN"] * graz
          - p["yN"] * tau(O, p) * D)

    return dN, dP, dZ, dD, dO



##############################################
# Diffusion
##############################################

def z0_season(t, p):
    s = np.sin(2.0 * np.pi * (t - p["tMaxSpring"]) / 365.0)
    return 0.5 * (1.0 - s**p["zMaxSteep"]) * (p["zMixWinter"] - p["z0"]) + p["z0"]

def diffusivity_profile(p, t):
    z = p["z"]

    if p.get("use_seasonal_diffusivity", False):
        z0t = z0_season(t, p)
    else:
        z0t = p["z0"]

    return p["kappa_min"] + (p["kappa0"] - p["kappa_min"]) * 0.5 * (1.0 - np.tanh((z - z0t) / p["zeta"]))

def diffusive_flux(phi, p, t, bottom_value=None):
    n = p["n"]
    dz = p["dz"]

    D = diffusivity_profile(p, t)

    J = np.zeros(n + 1)

    # interior interfaces
    for i in range(1, n):
        J[i] = -D[i] * (phi[i] - phi[i - 1]) / dz

    # top boundary (no flux)
    J[0] = 0.0

    # bottom boundary
    if bottom_value is None:
        J[n] = 0.0
    else:
        J[n] = -D[-1] * (bottom_value - phi[-1]) / dz

    return J

def diffusion_tendency(phi, p, t, bottom_value=None):
    dz = p["dz"]
    J = diffusive_flux(phi, p, t, bottom_value=bottom_value)
    return -(J[1:] - J[:-1]) / dz


# Oxygen diffusion
def oxygen_diffusive_flux(O, p, t, O_bottom=0.0):
    n = p["n"]
    dz = p["dz"]
    kappa = diffusivity_profile(p, t)

    J = np.zeros(n + 1)

    # Interior interfaces
    for i in range(1, n):
        J[i] = -kappa[i] * (O[i] - O[i - 1]) / dz

    # Surface air–sea exchange
    J[0] = -(O[0] - p["O2_atm"]) * p["KO2_surface"]

    # Bottom sink / exchange
    J[n] = -(O_bottom - O[-1]) * p["KO2_bottom"]

    return J

def oxygen_diffusion_tendency(O, p, t, O_bottom=0.0):
    dz = p["dz"]
    J = oxygen_diffusive_flux(O, p, t, O_bottom=O_bottom)
    return -(J[1:] - J[:-1]) / dz



##############################################
# Advection
##############################################

def advective_flux_upwind(phi, w, p, top_inflow_value=0.0):
    n = p["n"]
    F = np.zeros(n + 1)

    if w >= 0.0:
        # Downward: upwind is "above" the interface
        F[0] = w * top_inflow_value         # inflow from above surface
        F[1:n] = w * phi[0:n-1]             # interior interfaces
        F[n] = w * phi[n-1]                 # bottom outflow
    else:
        # Upward: upwind is "below" the interface
        F[0] = w * phi[0]                   # top outflow
        F[1:n] = w * phi[1:n]               # interior
        F[n] = w * phi[n-1]                 # inflow from below not handled separately

    return F


def advection_tendency(phi, w, p, top_inflow_value=0.0):
    dz = p["dz"]
    F = advective_flux_upwind(phi, w, p, top_inflow_value=top_inflow_value)
    return -(F[1:] - F[:-1]) / dz



##############################################
# Full Model
##############################################

def rhs(t, y, p):
    N, P, Z, D, O = unpack_state(y, p)

    I = light_profile(p,P,t)

    # biology
    dN_bio, dP_bio, dZ_bio, dD_bio, dO_bio = npzd_reactions(N, P, Z, D, O, I, p)

    # diffusion
    dN_diff = diffusion_tendency(N, p, t, bottom_value=p["N_bottom"])
    dP_diff = diffusion_tendency(P, p, t, bottom_value=None)
    dZ_diff = diffusion_tendency(Z, p, t, bottom_value=None)
    dD_diff = diffusion_tendency(D, p, t, bottom_value=None)
    dO_diff = oxygen_diffusion_tendency(O, p, t, O_bottom=0.0)

    # sinking advection for detritus (downward wD > 0)
    dD_adv = advection_tendency(D, w=p["wD"], p=p, top_inflow_value=0.0)
    dP_adv = advection_tendency(P, w=p["wP"], p=p, top_inflow_value=0.0)

    dN = dN_bio + dN_diff
    dP = dP_bio + dP_diff + dP_adv
    dZ = dZ_bio + dZ_diff
    dD = dD_bio + dD_diff + dD_adv
    dO = dO_bio + dO_diff

    return pack_state(dN, dP, dZ, dD, dO)



##############################################
# Initial conditions
##############################################

def initial_state(p):
    n = p["n"]
    N0 = np.full(n, p["N_init"], dtype=float)
    P0 = np.full(n, p["P_init"], dtype=float)
    Z0 = np.full(n, p["Z_init"], dtype=float)
    D0 = np.full(n, p["D_init"], dtype=float)
    O0 = np.full(n, p["O_init"], dtype=float)
    return pack_state(N0, P0, Z0, D0, O0)



##############################################
# Solving
##############################################

def run_simulation(p):
    y0 = initial_state(p)
    t_eval = np.linspace(p["t0"], p["t_end"], p["nt"])

    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=(p["t0"], p["t_end"]),
        y0=y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=p["rtol"],
        atol=p["atol"],
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol



##############################################
# Vertical profiles
##############################################

def plot_final_profiles(p, y_final):
    z = p["z"]
    N, P, Zc, D, O = unpack_state(y_final, p)
    I = light_profile(p, P, p["t_end"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(I, z)
    ax.set_xlabel("Light I(z) (µmol ph/m$^2$/s)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    #ax.set_title("Final light profile")
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(N, z, label="N")
    ax.plot(P, z, label="P")
    ax.plot(Zc, z, label="Z")
    ax.plot(D, z, label="D")
    ax.plot(O/10, z, label="O/10")
    ax.set_xlabel("Concentration (mmol X m$^{-3}$)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    #ax.set_title("Final NPZDO profiles")
    ax.legend()
    plt.tight_layout()

    plt.show()



##############################################
# Convergence
##############################################

def plot_convergence_colormesh(p, sol):
    z = p["z"]
    T, Z = np.meshgrid(sol.t, z)

    # Unpack each time slice
    n = p["n"]
    N = sol.y[0*n:1*n, :]
    P = sol.y[1*n:2*n, :]
    Zc = sol.y[2*n:3*n, :]
    D = sol.y[3*n:4*n, :]
    O = sol.y[4*n:5*n, :]

    # Light depends on phytoplankton
    I = np.zeros_like(P)
    for k in range(sol.t.size):
        I[:, k] = light_profile(p, P[:, k], sol.t[k])

    fields = [
        ("Light I(z,t)", I, "Light"),
        ("N(z,t)", N, "Nutrients"),
        ("P(z,t)", P, "Phytoplankton"),
        ("Z(z,t)", Zc, "Zooplankton"),
        ("D(z,t)", D, "Detritus"),
        ("O(z,t)", O, "Oxygen")
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    plt.gca().invert_yaxis()

    for ax, (title, F, cblabel) in zip(axes, fields):
        pcm = ax.pcolormesh(T, Z, F, shading="auto")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Depth (m)")
        fig.colorbar(pcm, ax=ax, label=cblabel)

    plt.tight_layout()
    plt.show()



##############################################
# Limiting factor
##############################################

def plot_limiting_resource(p, y_state, colim_tol=1e-3):
    z = p["z"]
    N, P, Zc, D, O = unpack_state(y_state, p)
    I = light_profile(p, P, p["t_end"])

    fL = I / (I + p["kL"])
    fN = N / (N + p["kN"])

    plt.figure(figsize=(6, 6))
    plt.plot(fL, z, label="Light limitation fL = I/(I+kL)")
    plt.plot(fN, z, label="Nutrient limitation fN = N/(N+kN)")
    plt.gca().invert_yaxis()
    plt.xlabel("Limitation factor (0–1)")
    plt.ylabel("Depth (m)")
    #plt.title("Phytoplankton limiting factors vs depth")
    plt.legend()
    plt.tight_layout()
    plt.show()



##############################################
# High nutrient input
##############################################

def chl_from_phyto(P, N_to_Chl=8.0):
    # mmol N m^-3 -> g N m^-3
    gN = P * 0.014  # 1 mmol N = 1e-3 mol * 14 g/mol = 0.014 g

    # g N m^-3 -> g Chl m^-3
    gChl = gN / N_to_Chl

    # g Chl m^-3 -> mg Chl m^-3
    mgChl = gChl * 1000.0
    return mgChl

def run_baseline_vs_highN(N_high=100.0):
    # Baseline parameters
    p_base = make_params()
    p_base = build_grid(p_base)
    p_base["use_phyto_light_damping"] = True
    p_base["use_seasonal_light"] = False
    p_base["use_seasonal_diffusivity"] = False

    # High nutrient parameters
    p_highN = dict(p_base)
    p_highN["N_bottom"] = N_high

    # Run simulations
    sol_base = run_simulation(p_base)
    sol_highN = run_simulation(p_highN)

    return p_base, p_highN, sol_base, sol_highN

def extract_profiles(p_base, p_highN, sol_base, sol_highN):
    y_base = sol_base.y[:, -1]
    y_highN = sol_highN.y[:, -1]

    N0, P0, Z0, D0, O0 = unpack_state(y_base, p_base)
    N1, P1, Z1, D1, O1 = unpack_state(y_highN, p_highN)

    z = p_base["z"]

    # Chlorophyll
    Chl0 = chl_from_phyto(P0, N_to_Chl=8.0)
    Chl1 = chl_from_phyto(P1, N_to_Chl=8.0)

    return z, (N0, P0, Z0, D0, O0, Chl0), (N1, P1, Z1, D1, O1, Chl1)

def plot_npzdo_comparison(z, base, high):
    N0, P0, Z0, D0, O0, Chl0 = base
    N1, P1, Z1, D1, O1, Chl1 = high

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), sharey=True)
    axes = axes.flatten()
    plt.gca().invert_yaxis()

    plots = [
        (N0, N1, "Nutrient (mmol N m$^{-3}$)", "N baseline", "N high-N"),
        (P0, P1, "Phytoplankton (mmol P m$^{-3}$)", "P baseline", "P high-N"),
        (Z0, Z1, "Zooplankton (mmol Z m$^{-3}$)", "Z baseline", "Z high-N"),
        (D0, D1, "Detritus (mmol D m$^{-3}$)", "D baseline", "D high-N"),
        (O0, O1, "Oxygen (mmol O m$^{-3}$)", "O baseline", "O high-N"),
        (Chl0, Chl1, "Chlorophyll-a (mg m$^{-3}$), N:Chl = 8 gN:gChl", 
         "Chl-a baseline", "Chl-a high-N")
    ]

    for ax, (v0, v1, xlabel, lab0, lab1) in zip(axes, plots):
        ax.plot(v0, z, label=lab0)
        ax.plot(v1, z, label=lab1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()
        ax.legend()

    #plt.title("Baseline (N_bottom=10 mmol N m$^{-3}$) vs high nutrient input (N_bottom=100 mmol N m$^{-3}$)")
    plt.tight_layout()
    plt.show()



##############################################
# Sensitivity analysis
##############################################

def sensitivity_analysis(param_name, new_value, unit):
    # --- Default simulation ---
    p_default = make_params()
    p_default = build_grid(p_default)
    sol_default = run_simulation(p_default)

    # --- Modified simulation ---
    p_mod = make_params()
    p_mod[param_name] = new_value
    p_mod = build_grid(p_mod)
    sol_mod = run_simulation(p_mod)

    # --- Extract final states ---
    y_def = sol_default.y[:, -1]
    y_mod = sol_mod.y[:, -1]

    N0, P0, Z0, D0, O0 = unpack_state(y_def, p_default)
    N1, P1, Z1, D1, O1 = unpack_state(y_mod, p_mod)

    z = p_default["z"]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6,6))

    # Nutrients
    ax.plot(N0, z, '--', color='orange',
            label=f"N default ({param_name}={p_default[param_name]} {unit})")
    ax.plot(N1, z, '-', color='orange',
            label=f"N modified ({param_name}={new_value} {unit})")

    # Phytoplankton
    ax.plot(P0, z, '--', color='green', label="P default")
    ax.plot(P1, z, '-', color='green', label="P modified")

    # Zooplankton
    ax.plot(Z0, z, '--', color='violet', label="Z default")
    ax.plot(Z1, z, '-', color='violet', label="Z modified")

    # Detritus
    ax.plot(D0, z, '--', color='pink', label="D default")
    ax.plot(D1, z, '-', color='pink', label="D modified")

    # Oxygen
    ax.plot(O0/10, z, '--', color='blue', label="O/10 default")
    ax.plot(O1/10, z, '-', color='blue', label="O/10 modified")

    ax.set_xlabel("Concentration (mmol m$^{-3}$)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()

    ax.legend()
    plt.tight_layout()
    plt.show()



##############################################
# Output
##############################################

p = make_params()
p = build_grid(p)
p["use_phyto_light_damping"] = True
p["use_seasonal_light"] = False
p["use_seasonal_diffusivity"] = False
sol = run_simulation(p)

plot_convergence_colormesh(p, sol)
plot_final_profiles(p, sol.y[:, -1])
plot_limiting_resource(p, sol.y[:, -1])

# Low/high nutrient input
'''
p_base, p_highN, sol_base, sol_highN = run_baseline_vs_highN()
z, base_profiles, high_profiles = extract_profiles(p_base, p_highN, sol_base, sol_highN)
plot_final_profiles(p_base, sol_base.y[:, -1])
plot_final_profiles(p_highN, sol_highN.y[:, -1])
plot_npzdo_comparison(z, base_profiles, high_profiles)
'''

# Sensitivity analysis
'''
sensitivity_analysis("kL", 62.5, "µmol ph/m$^2$/s")
sensitivity_analysis("gP_max", 2.0, "day$^{-1}$")
'''
