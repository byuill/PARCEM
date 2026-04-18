"""
PARC (Profile And Reach Change) Model — Rio Coca Morphodynamic Simulator
=========================================================================
A 1-D physics-based morphodynamic model for simulating the longitudinal bed
evolution and lateral channel adjustment of the Rio Coca, Ecuador, following
the knickpoint-driven incision wave triggered by the 2020 collapse of the San
Rafael waterfall — Ecuador's largest waterfall — located approximately 19 km
downstream of the model inlet.

SCIENTIFIC FRAMEWORK
--------------------
The PARC model solves a coupled set of geomorphic process equations:
  (1)  Sediment Continuity (Exner Equation) — conservation of bed material mass
  (2)  Sediment Transport Capacity         — transport-limited power-law flux law
  (3)  Lateral Bank Erosion                — incision-coupled valley widening

MODELLED PROCESSES
------------------
  1. Fluvial Incision & Aggradation
       Net change in bed elevation is governed by the divergence of sediment
       flux (Exner Equation). Where flux diverges (more sediment leaving than
       entering), the bed incises. Where flux converges, the bed aggrades.
       This drives an upstream-migrating knickpoint and a downstream sediment
       deposition wave, consistent with field observations on the Rio Coca.

  2. Knickpoint (Nick-Zone) Migration
       The abrupt steepening at the waterfall / nick-zone propagates upstream
       as an incision wave at a rate set by the local stream power. This
       mechanism releases a large pulse of coarse and fine sediment that is
       subsequently routed and deposited downstream, filling the valley floor.

  3. Transport-Limited Sediment Flux
       The PARC model operates in the transport-limited regime (Tucker &
       Whipple, 2002), where sediment flux equals the local hydraulic transport
       capacity. This implies the channel bed is always erodible and net
       morphological change is controlled entirely by hydraulic capacity, not
       sediment supply. The resulting governing equation is diffusive:
           dz/dt ~ K * d²z/dx²
       where K depends on discharge, width, and the transport coefficient 'a'.

  4. Optional Blended (Hybrid) Transport Mode
       When TRANSPORT_MODE = 'Blended', an exponential adaptation length
       (L_ADAPT, metres) allows a gradual transition between transport-limited
       and a partly supply-limited response. This is implemented via a
       first-order linear relaxation of the actual flux (Qs) toward the
       transport capacity (Qc) over the adaptation distance:
           dQs/dx = (Qc - Qs) / L_ADAPT
       Longer L_ADAPT values produce more supply-limited (kinematic wave)
       behaviour; shorter values recover the fully diffusive, transport-limited
       end-member.

  5. Bank Erosion and Lateral Channel Widening
       Vertical bed incision destabilises cohesionless valley walls. In the
       PARC model, bank retreat is proportional to the local incision depth
       scaled by a spatially variable coefficient (bank_widening_factor).
       Bank collapse delivers additional sediment to the channel floor,
       acting as a source term in the Exner equation that partially buffers
       net incision. Near the waterfall, the widening factor is reduced to
       reflect the higher rock competence / slope resistance observed there.
       Over time, this process widens the active channel from its initial
       prescribed geometry toward a new equilibrium width.

  6. Flood Hydrograph Forcing
       Time-varying daily discharge (Q, m³/s) drives the transport capacity.
       Elevated flows during rainfall or ENSO-driven flood events concentrate
       the bulk of geomorphic work (consistent with the concept of the
       'dominant discharge' controlling long-term morphology; Wolman &
       Miller, 1960).

CORE ASSUMPTIONS
----------------
  A1. Transport-Limited System
        Sediment supply does not limit flux; the bed is always erodible and
        the system adjusts instantly to transport capacity. This is appropriate
        for the active incision front on the Rio Coca where abundant loose
        debris and unconsolidated valley fill are continuously mobilised.

  A2. 1-D Longitudinal Profile
        Lateral gradients, secondary flows, and bar migration are not
        explicitly resolved. Their aggregate effect on width adjustment is
        parameterised through the bank erosion coupling (bank_widening_factor).

  A3. Power-Law Transport Capacity
        Qs ∝ (Q/B)^n, analogous to a unit stream power or excess shear stress
        formulation (Howard, 1994; Whipple & Tucker, 1999). The exponent 'n'
        controls the non-linearity of transport with discharge; 'a' scales
        the absolute magnitude of flux for a given flow and slope.

  A4. Rectangular Cross-Section with Dynamic Width
        Channel width B is spatially prescribed at model initialisation (from
        hydraulic geometry data) and subsequently evolves through bank erosion.
        Flow depth and velocity are not explicitly resolved; unit discharge
        q = Q/B is the primary hydraulic variable.

  A5. Non-Cohesive Bed Material (No Threshold of Motion)
        No critical shear stress or entrainment threshold is applied. This
        simplification is justified for high-energy flood conditions that
        dominate long-term transport on the Rio Coca, where all sediment
        fractions are effectively mobile during major events.

  A6. Fixed Upstream Boundary Condition
        The upstream bed elevation is held constant at the initial surveyed
        value (e.g., at the waterfall lip or the model inlet). This represents
        a fixed base level for the upstream reach.

  A7. Open (Constant-Gradient) Downstream Boundary
        The downstream boundary uses a constant-slope (zero-second-derivative)
        condition, allowing sediment to freely exit the model domain without
        artificial reflection of the incision wave.

  A8. Simplified Valley Geometry
        Bank height at each cross-section is estimated from a prescribed
        hillslope angle (HILLSLOPE_SLOPE, rise/run) and an initial bank height
        offset (INITIAL_BANK_HEIGHT), yielding an approximated V-shaped or
        trapezoidal valley cross-section. True valley geometry from DEM surveys
        or field measurements can replace this parameterisation.

CALIBRATION STRATEGY
--------------------
  The PARC model is calibrated using a dual-objective Vol_Elev optimisation
  that simultaneously matches:
    (i)  Eroded Volume Trend  : Simulated cumulative mobilised sediment volume
         is matched to observed volumes derived from repeat bathymetric surveys,
         drone-based DEMs of difference (DoDs), or satellite imagery analysis.
    (ii) Bed Elevation Profile : Simulated longitudinal thalweg profiles are
         matched to observed profiles at multiple survey dates, using an optimal
         longitudinal x-shift to account for datum or registration offsets.
  Global parameter search uses Differential Evolution (scipy.optimize), followed
  by local Nelder–Mead polishing. Parameter sensitivity and cross-interaction
  matrices are computed using one-at-a-time (OAT) central finite differences
  evaluated at the optimal parameter set.

REFERENCES / THEORETICAL BASIS
-------------------------------
  - Exner (1925)           : Sediment continuity equation, dz/dt = -(1/B) dQs/dx
  - Howard (1994)          : Stream power incision model (detachment-limited end)
  - Tucker & Whipple (2002): Geomorphic transport laws and landscape evolution
  - Whipple & Tucker (1999): Dynamics of the stream-power river incision model
  - Parker (2004)          : 1-D morphodynamic models and Exner equation
  - Lane (1955)            : Hydraulic geometry and stable channel design
  - Wolman & Miller (1960) : Magnitude and frequency of geomorphic processes
  - Smith & Bretherton (1972): Stability of hillslope–channel systems

USAGE
-----
  Set CALIBRATION_MODE and OUTPUT_MODE flags in Section 1 (CONFIGURATION &
  CONTROL BOARD), then run:
      python PARC_PyVer_V2.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import os
import itertools
from datetime import datetime, timedelta
import gc
import glob
import scipy.signal as signal
import sys # Added for graceful exit on KeyboardInterrupt
from scipy.optimize import differential_evolution, minimize_scalar

# ==========================================
# 1. CONFIGURATION & CONTROL BOARD
# ==========================================
# GEOMORPHIC THEORY CONTEXT — PARC Model
# ----------------------------------------
# The PARC (Profile And Reach Change) model is a 1-D morphodynamic simulator
# designed for the Rio Coca, Ecuador, in the aftermath of the 2020 San Rafael
# waterfall collapse. It resolves the longitudinal evolution of the thalweg
# (channel bed profile) and the lateral expansion of the channel through
# bank erosion and valley widening.
#
# Governing Equation — Exner (Sediment Continuity):
#   (1 - λ_p) * ∂z/∂t = -(1/B) * ∂Qs/∂x
#   where λ_p = bed porosity (~0.35–0.40 for gravel beds), z = bed elevation,
#   B = channel width, and Qs = volumetric sediment flux (m³/s).
#   The PARC model absorbs porosity into the calibrated transport coefficient 'a'.
#
# Sediment Flux Law — Transport-Limited Power Law:
#   Qs = -B * K * (∂z/∂x)
#   K = (b * a / (3 * S_eq)) * (Q/B)^n
#   This is analogous to a unit stream power formulation where:
#     - 'a'  = transport efficiency coefficient (dimensional, absorbs grain size,
#               roughness, and bed material mobility factors)
#     - 'n'  = discharge non-linearity exponent (typically 1.5–3.0 for gravel beds;
#               higher values concentrate geomorphic work in large floods)
#     - 'b'  = slope-to-velocity scaling exponent (relates bed slope to local
#               flow velocity via a power law, consistent with normal-flow hydraulics)
#     - Q/B  = unit discharge (m²/s), the primary hydraulic driver of transport
#
# Regime Classification:
#   Transport-Limited (PARC default): Qs = transport capacity at all times.
#     The system behaves diffusively: d²z/dx² drives bed change.
#     Appropriate for reaches with abundant, unconsolidated bed material
#     (e.g., landslide-derived debris, alluvial fill) where supply exceeds capacity.
#   Blended Mode: Qs relaxes toward capacity over adaptation length L_ADAPT.
#     Captures supply-limited (kinematic wave / detachment-limited) behaviour
#     in bedrock or coarse-lag-armoured reaches where supply may limit flux.
#
# Lateral Coupling — Bank Erosion:
#   dB/dt = bank_widening_factor * max(0, -dz/dt)
#   Channel width grows when the bed incises. Collapsed bank material is added
#   back to the channel floor as a sediment source, partially offsetting incision
#   (consistent with field observations of debris-fan formation on the Rio Coca).
#
# Equilibrium Reference State:
#   An equilibrium longitudinal profile (zb_eq) is constructed from prescribed
#   reach-averaged slopes (US_slope_factor, DS_slope_factor). This reference
#   profile represents the graded condition toward which the system tends over
#   geological time (Mackin, 1948; Lane, 1955). Deviations from this profile
#   drive disequilibrium flux and net bed change.

CALIBRATION_MODE = False  # Options: False, True (grid sweep), 'Vol_Elev' (optimization)
OUTPUT_MODE = 'Advanced'    # Options: 'animation', 'plots', 'both', 'Advanced'
TRANSPORT_MODE = 'Transport_Limited' # Options: 'Transport_Limited', 'Blended'

# Paths
# Using relative paths based on the script location for local execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'Inputs')
BATHY_PATH = os.path.join(BASE_DIR, 'Initial_Bathy.csv')
DISCHARGE_PATH = os.path.join(BASE_DIR, 'Discharge_Timeseries.csv')
CAL_PATH = os.path.join(BASE_DIR, 'Calibration.csv')
CAL_VOL_PATH = os.path.join(BASE_DIR, 'Calibration_Vol.csv')
CAL_ELEV_PATH = os.path.join(BASE_DIR, 'Calibration_Elev.csv')
OUTPUT_BASE_DIR = os.path.join(SCRIPT_DIR, 'Calibration_Results')
CALIBRATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Calibration_Output')

# Vol_Elev Optimization Parameter Bounds
# These define the feasible search space for the PARC model calibration.
# Each bound pair (min, max) reflects physically plausible ranges for the
# Rio Coca based on published values for similar gravel-bed mountain rivers
# and exploratory sensitivity runs.
VOL_ELEV_PARAM_BOUNDS = {
    # 'a': Transport efficiency coefficient. Controls the overall magnitude of
    #      sediment flux for a given discharge and slope. Lower values produce
    #      sluggish transport; higher values produce a more responsive system.
    #      Units absorb grain size and roughness effects.
    'a': (0.0005, 0.01),
    # 'n': Discharge exponent in the transport capacity law Qs ∝ (Q/B)^n.
    #      Governs the non-linearity of flood effects: n > 1 concentrates
    #      geomorphic work in large, infrequent floods (flashy rivers).
    #      n ≈ 1.5–2 is typical for suspended load; n ≈ 2–4 for bedload.
    'n': (1.0, 4.0),
    # 'b': Slope-to-velocity scaling exponent. Links local bed slope to the
    #      effective flow velocity used in the transport capacity pre-factor.
    #      Consistent with Manning / Chezy normal-flow hydraulics.
    'b': (1.0, 5.0),
    # 'bank_widening_factor': Rate of lateral bank retreat per unit of vertical
    #      incision (m of widening per m of incision). Governs how quickly the
    #      channel widens in response to knickpoint-driven bed lowering.
    #      Higher values represent weaker, less resistant bank materials.
    'bank_widening_factor': (1.0, 10.0),
    # 'US_slope_factor': Multiplier applied to the prescribed upstream equilibrium
    #      slope. Values > 1 steepen the reference profile, increasing initial
    #      transport capacity headward of the nick-zone.
    'US_slope_factor': (0.5, 2.0),
    # 'DS_slope_factor': Multiplier applied to the prescribed downstream equilibrium
    #      slope. Adjusts the base-level gradient into the alluvial fan / lowland
    #      receiving zone.
    'DS_slope_factor': (0.5, 2.0),
    # 'width_factor': Global multiplier on the initial prescribed channel widths.
    #      Adjusts the hydraulic geometry (unit discharge) uniformly across
    #      the entire PARC model domain.
    'width_factor': (0.5, 2.0),
}

# Vol_Elev Calibration Parameter Priority
# When two parameter sets produce similar model fit, the PARC model optimizer
# prefers the one with lower values for parameters with a lower (higher-priority)
# number. Lower number = higher priority = stronger pull toward the minimum of that
# param's range. This encodes the parsimony principle: prefer simpler (lower-valued)
# parameter sets when model performance is statistically indistinguishable.
# Parameters absent from VOL_ELEV_PARAM_BOUNDS (e.g. 'volume_multiplier') are
# listed here for documentation purposes only and have no effect on optimization.
PARAM_PRIORITY = {
    'a': 1,
    'n': 1,
    'b': 1,
    'bank_widening_factor': 2,
    'US_slope_factor': 4,
    'DS_slope_factor': 4,
    'width_factor': 5,
    'volume_multiplier': 6,  # not directly optimized; listed for reference only
}

# Scaling factor for the priority penalty relative to the main objective.
# Increase to enforce priorities more aggressively; decrease to reduce influence.
# Typical main objective values are O(0.01–1.0), so 1e-3 acts as a tie-breaker
# without distorting real differences in PARC model fit quality.
PRIORITY_EPSILON = 1e-3

# -----------------------------------------------------------------------
# Physical Domain & Process Constants
# -----------------------------------------------------------------------
# WATERFALL_POS: Downstream distance (m) of the San Rafael waterfall / nick-
#   zone from the model inlet. This is the primary geomorphic discontinuity
#   that drives the knickpoint migration wave in the PARC model.
WATERFALL_POS = 19000

# HILLSLOPE_SLOPE: Valley wall gradient (rise/run) used in the PARC model's
#   bank erosion geometry. A value of 0.40 (≈ 22°) is representative of
#   steep, partially vegetated hillslopes flanking the Rio Coca gorge.
#   This controls how much bank material is delivered per unit of widening.
HILLSLOPE_SLOPE = 0.40

# RAMP_DISTANCE: Spatial extent (m) over which bank_widening_factor is
#   linearly reduced near the waterfall / nick-zone. Captures the higher
#   resistance of the competent volcanic / intrusive rock exposed at the
#   waterfall face, where lateral incision is slower than in alluvial reaches.
RAMP_DISTANCE = 3000.0

# INITIAL_BANK_HEIGHT: Approximate initial height of valley walls above the
#   channel thalweg (m) before incision begins. Combined with HILLSLOPE_SLOPE,
#   this defines the initial bank geometry for the PARC bank erosion module.
INITIAL_BANK_HEIGHT = 3.0

# USE_BANK_EROSION: Enables the lateral-vertical incision coupling. When True,
#   the PARC model tracks channel widening and the associated sediment flux
#   from bank collapse, which acts as a secondary source term in the Exner
#   equation. Set False to run a purely vertical, constant-width simulation.
USE_BANK_EROSION = True

# L_ADAPT: Adaptation length (m) for Blended transport mode only. Controls
#   the e-folding distance over which actual sediment flux Qs relaxes toward
#   the local transport capacity Qc. Shorter L_ADAPT → more transport-limited
#   (diffusive); longer L_ADAPT → more supply/detachment-limited (kinematic wave).
L_ADAPT = 1000.0

# -----------------------------------------------------------------------
# Parameter Ranges — Grid Calibration Mode
# -----------------------------------------------------------------------
# Used only when CALIBRATION_MODE = True (exhaustive grid sweep).
# These ranges span the physically plausible space for the PARC model
# transport and channel geometry parameters on the Rio Coca.
CALIBRATION_PARAMS = {
    'a': [0.002, 0.0025, 0.003], # Transport efficiency coefficient: higher → more flux per unit power
    'n': [2.0, 2.5, 3.0],        # Discharge exponent: higher → flood-dominated transport regime
    'b': [3],                    # Slope-velocity exponent: held fixed for grid sweep simplicity
    'bank_widening_factor': [4.0], # Bank retreat rate per unit incision; 4.0 represents moderate
                                   # alluvial bank erodibility on the Rio Coca
    'slope_factor': [0.8, 1.0, 1.2],  # Equilibrium profile slope multiplier (applied to both US/DS)
    'width_factor': [1.0]             # Channel width multiplier; held at 1 (survey-derived widths)
}

# -----------------------------------------------------------------------
# Default Parameters — Single-Run (CALIBRATION_MODE = False)
# -----------------------------------------------------------------------
# These are the Vol_Elev-calibrated PARC model parameters derived from the
# Calibration_23MAR2026 run using observed sediment volume time-series and
# repeated longitudinal bed profiles on the Rio Coca.
#
# *** PRE-RECALIBRATION values from Calibration_23MAR2026 ***
# The Vol_Elev objective now normalises simulated volume against the observed peak
# (not its own peak), so a new calibration run will drive bank_widening_factor up
# to close the absolute volume gap and volume_multiplier will converge to ~1.0.
# Update these values after running CALIBRATION_MODE = 'Vol_Elev'.
#best calcs:'bank_widening_factor': 1.0004209585,'volume_multiplier': 1.234002,
DEFAULT_PARAMS = {
    # Transport efficiency (dimensional): sets absolute magnitude of sediment flux
    'a': 0.0033572196,
    # Discharge non-linearity exponent (~2 indicates moderately flood-dominated transport)
    'n': 1.9065097658,
    # Slope-velocity scaling exponent (consistent with Manning-type normal flow relation)
    'b': 2.3706749607,
    # Bank retreat rate per unit incision: moderate erodibility in alluvial reaches
    'bank_widening_factor': 4.0,
    # Upstream equilibrium slope multiplier (>1 → steeper graded profile upstream of nick-zone)
    'US_slope_factor': 1.5839701898,
    # Downstream equilibrium slope multiplier (controls aggradation fan gradient)
    'DS_slope_factor': 1.5153937230,
    # Global channel width multiplier (close to 2 → wider channels reduce unit discharge,
    # consistent with post-collapse channel widening observed on the Rio Coca)
    'width_factor': 1.9995122367,
    # Post-hoc volume scaling multiplier; expected to converge to ~1.0 after re-calibration
    # with the absolute-normalisation Vol_Elev objective
    'volume_multiplier': 1.0,
}

# ==========================================
# 2. DATA LOADING (Global Scope)
# ==========================================
def load_data():
    # Load Initial Bathymetry (Longitudinal Profile z vs x)
    if os.path.exists(BATHY_PATH):
        df_bathy = pd.read_csv(BATHY_PATH)
        x = df_bathy.iloc[:, 0].values
        zb0 = df_bathy.iloc[:, 1].values
        nx = len(x)
        L = x[-1] - x[0]
        dx = L / (nx - 1)
        print(f"Loaded bathymetry: {nx} points over {L/1000:.2f} km.")
    else:
        print(f"Warning: {BATHY_PATH} not found. Using synthetic profile.")
        nx = 200
        L = 100000
        dx = L / (nx - 1)
        x = np.linspace(0, L, nx)
        base_elevation = 250.0
        waterfall_height = 100.0
        slope_initial = -0.0005
        zb0 = np.full(nx, base_elevation)
        zb0[x > WATERFALL_POS] = base_elevation - waterfall_height + slope_initial * (x[x > WATERFALL_POS] - WATERFALL_POS)
        zb0 += slope_initial * x

    # Load Discharge Time Series (Hydrologic Driver)
    if os.path.exists(DISCHARGE_PATH):
        df_q = pd.read_csv(DISCHARGE_PATH)
        date_col_q = df_q.columns[0]
        df_q[date_col_q] = pd.to_datetime(df_q[date_col_q], format='mixed')
        Q = df_q.iloc[:, 1].values
        nt = len(Q)
        start_date = df_q[date_col_q].iloc[0]
        duration = df_q[date_col_q].iloc[-1] - start_date
        total_time_days = duration.total_seconds() / (24 * 3600.0)
        dt_days = total_time_days / (nt - 1)
        dt_secs = dt_days * 24 * 3600
        print(f"Loaded discharge: {nt} steps over {total_time_days:.2f} days.")
    else:
        print(f"Warning: {DISCHARGE_PATH} not found. Using synthetic discharge.")
        nt = 2000
        total_time_days = 2000
        dt_days = total_time_days / nt
        dt_secs = dt_days * 24 * 3600
        Q = np.full(nt, 500.0)
        Q[500:550] = 1500
        Q[1200:1280] = 2000
        start_date = pd.Timestamp.now()

    # Load Calibration Data (Observed Sediment Volumes)
    df_cal = None
    cal_time_days = None
    cal_vol = None
    if os.path.exists(CAL_PATH):
        df_cal = pd.read_csv(CAL_PATH)
        date_col_cal = df_cal.columns[0]
        df_cal[date_col_cal] = pd.to_datetime(df_cal[date_col_cal], format='mixed')
        cal_vol = df_cal.iloc[:, 1].values
        cal_time_days = (df_cal[date_col_cal] - start_date).dt.total_seconds() / (24 * 3600.0)
        print(f"Loaded calibration data: {len(df_cal)} records.")

    return x, zb0, nx, dx, Q, nt, total_time_days, dt_days, dt_secs, df_cal, cal_time_days, cal_vol, start_date

# Load data once
X_GLOBAL, ZB0_GLOBAL, NX_GLOBAL, DX_GLOBAL, Q_GLOBAL, NT_GLOBAL, TOTAL_TIME_DAYS, DT_DAYS, DT_SECS, DF_CAL, CAL_TIME_DAYS, CAL_VOL, START_DATE = load_data()


def load_vol_elev_calibration_data(start_date):
    """
    Load calibration data for Vol_Elev mode:
    - Observed eroded sediment volumes over time
    - Observed bed elevation profiles at various dates
    """
    # --- Volume calibration data ---
    df_vol = pd.read_csv(CAL_VOL_PATH)
    vol_dates = pd.to_datetime(df_vol.iloc[:, 0], format='mixed')
    vol_values = df_vol.iloc[:, 1].values.astype(float)

    # First entry = model t=0; subsequent entries use actual dates vs model start
    vol_time_days = np.zeros(len(vol_dates))
    vol_time_days[0] = 0.0
    for i in range(1, len(vol_dates)):
        vol_time_days[i] = (vol_dates.iloc[i] - start_date).total_seconds() / 86400.0

    # --- Elevation profile calibration data ---
    df_elev = pd.read_csv(CAL_ELEV_PATH, header=None)
    date_row = df_elev.iloc[0]  # First row contains dates
    # Data starts from row 2 (row 1 is X/Z labels)

    elev_profiles = []
    n_cols = len(df_elev.columns)
    for j in range(0, n_cols, 2):  # Step through X,Z pairs
        date_str = str(date_row.iloc[j]).strip()
        profile_date = pd.to_datetime(date_str, format='mixed')

        x_raw = pd.to_numeric(df_elev.iloc[2:, j], errors='coerce')
        z_raw = pd.to_numeric(df_elev.iloc[2:, j + 1], errors='coerce')
        valid = x_raw.notna() & z_raw.notna()
        x_data = x_raw[valid].values
        z_data = z_raw[valid].values

        if len(x_data) == 0:
            continue

        # Map date to model time (first profile = t=0)
        if j == 0:
            t_days = 0.0
        else:
            t_days = (profile_date - start_date).total_seconds() / 86400.0

        elev_profiles.append({
            'date': profile_date,
            'time_days': t_days,
            'x': x_data,
            'z': z_data,
        })

    print(f"Loaded Vol_Elev calibration: {len(vol_values)} volume records, "
          f"{len(elev_profiles)} elevation profiles.")
    return vol_time_days, vol_values, elev_profiles


# ==========================================
# 3a. DIAGNOSTIC PLOT HELPERS (regular mode)
# ==========================================

def plot_vol_elev_comparison(run_id, t_arr, cum_eroded, zb, volume_multiplier=1.0):
    """
    Generate a Vol_Elev_Calibration_Result-style figure for a completed simulation:
      Panel 1 – Simulated vs observed eroded volume timeseries.
      Panel 2 – Simulated vs observed bed elevation profiles (all dates, colour-coded).

    The simulated volume (cum_eroded) is expected to already have the
    volume_multiplier applied if running in regular simulation mode.

    Saves to  Simulation_Vol_Elev_Result_{run_id}.png  alongside the script.
    """
    try:
        vol_time_days, vol_values, elev_profiles = load_vol_elev_calibration_data(START_DATE)
    except Exception as e:
        print(f"  [Vol-Elev plot] Could not load calibration data: {e}")
        return

    elev_cal_profiles = [p for p in elev_profiles if p['time_days'] > 0]

    fig, (ax_v, ax_e) = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35)

    # --- Panel 1: Volume timeseries ---
    label_text = 'Simulated eroded vol'
    if abs(volume_multiplier - 1.0) > 1e-6:
        label_text += f' (with \u00d7{volume_multiplier:.3f} multiplier)'

    ax_v.plot(t_arr, cum_eroded, 'b-', linewidth=1.5, label=label_text)
    ax_v.scatter(vol_time_days, vol_values, color='red', marker='x', s=60,
                 zorder=5, label='Observed eroded vol')
    ax_v.set_xlabel('Time (days)')
    ax_v.set_ylabel('Eroded Volume (m\u00b3)')
    ax_v.set_title(f'Simulation \u2014 Eroded Volume Trend  (run: {run_id})')
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    # --- Panel 2: Elevation profiles ---
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(elev_cal_profiles), 1)))
    for i_p, prof in enumerate(elev_cal_profiles):
        t_idx = np.argmin(np.abs(t_arr - prof['time_days']))
        z_sim = zb[t_idx, :]

        def _rmse_shift(xs, _zs=z_sim, _xo=prof['x'], _zo=prof['z']):
            zi = np.interp(_xo + xs, X_GLOBAL, _zs, left=np.nan, right=np.nan)
            v = ~np.isnan(zi)
            return float(np.sqrt(np.mean((_zo[v] - zi[v]) ** 2))) if np.sum(v) >= 3 else 1e6

        rs = minimize_scalar(_rmse_shift, bounds=(-X_GLOBAL[-1], X_GLOBAL[-1]),
                             method='bounded', options={'xatol': DX_GLOBAL})
        best_shift = rs.x
        lbl = prof['date'].strftime('%Y-%m-%d')
        ax_e.plot(prof['x'] + best_shift, prof['z'], 'o', color=colors[i_p],
                  markersize=3, label=f'Obs {lbl} (shift {best_shift:.0f} m)')
        ax_e.plot(X_GLOBAL, z_sim, '-', color=colors[i_p], alpha=0.6,
                  label=f'Sim {lbl}')

    ax_e.set_xlabel('Distance downstream (m)')
    ax_e.set_ylabel('Bed Elevation (m)')
    ax_e.set_title(f'Simulation \u2014 Bed Elevation Profiles  (run: {run_id})')
    ax_e.legend(fontsize=7, ncol=2)
    ax_e.grid(True, alpha=0.3)

    out_path = os.path.join(SCRIPT_DIR, f'Simulation_Vol_Elev_Result_{run_id}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Vol-Elev comparison saved \u2192 {out_path}")


def plot_profile_comparison(run_id, t_arr, zb):
    """
    Generate a multi-subplot figure comparing simulated vs observed bed elevation
    profiles, one subplot per calibration profile date.  Each subplot shows:
      • observed elevations (scatter) vs simulated profile (line)
      • RMSE (m) and Nash-Sutcliffe Efficiency (NSE) annotated in the subplot title.

    An optimal longitudinal x-shift is applied to minimise RMSE, matching the
    approach used in the Vol_Elev calibration post-processing.

    Saves to  Simulation_Profile_Comparison_{run_id}.png  alongside the script.
    """
    try:
        _, _, elev_profiles = load_vol_elev_calibration_data(START_DATE)
    except Exception as e:
        print(f"  [Profile comparison] Could not load calibration data: {e}")
        return

    elev_cal_profiles = [p for p in elev_profiles if p['time_days'] > 0]
    if not elev_cal_profiles:
        print("  [Profile comparison] No calibration elevation profiles found.")
        return

    n_profiles = len(elev_cal_profiles)
    ncols = min(3, n_profiles)
    nrows = int(np.ceil(n_profiles / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle(f'Simulated vs Observed Bed Elevation Profiles  (run: {run_id})',
                 fontsize=13, y=1.01)
    plt.subplots_adjust(hspace=0.50, wspace=0.35)

    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for i_p, prof in enumerate(elev_cal_profiles):
        row, col = divmod(i_p, ncols)
        ax = axes[row][col]

        t_idx = np.argmin(np.abs(t_arr - prof['time_days']))
        z_sim = zb[t_idx, :]

        # Optimal x-shift (minimise RMSE between observed and interpolated simulated elevation)
        def _rmse_shift(xs, _zs=z_sim, _xo=prof['x'], _zo=prof['z']):
            zi = np.interp(_xo + xs, X_GLOBAL, _zs, left=np.nan, right=np.nan)
            v = ~np.isnan(zi)
            return float(np.sqrt(np.mean((_zo[v] - zi[v]) ** 2))) if np.sum(v) >= 3 else 1e6

        rs = minimize_scalar(_rmse_shift, bounds=(-X_GLOBAL[-1], X_GLOBAL[-1]),
                             method='bounded', options={'xatol': DX_GLOBAL})
        best_shift = rs.x

        # Compute GoF metrics at shifted positions
        z_sim_at_obs = np.interp(prof['x'] + best_shift, X_GLOBAL, z_sim,
                                 left=np.nan, right=np.nan)
        valid = ~np.isnan(z_sim_at_obs)
        obs_v = prof['z'][valid]
        sim_v = z_sim_at_obs[valid]

        if np.sum(valid) >= 2:
            rmse_val = float(np.sqrt(np.mean((obs_v - sim_v) ** 2)))
            ss_res = float(np.sum((obs_v - sim_v) ** 2))
            ss_tot = float(np.sum((obs_v - np.mean(obs_v)) ** 2))
            nse_val = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        else:
            rmse_val = np.nan
            nse_val = np.nan

        # Plot
        ax.scatter(prof['x'] + best_shift, prof['z'], color=colors[i_p],
                   s=8, zorder=3, label='Observed')
        ax.plot(X_GLOBAL, z_sim, '-', color=colors[i_p], alpha=0.8, label='Simulated')

        date_str = prof['date'].strftime('%Y-%m-%d')
        rmse_str = f"{rmse_val:.2f} m" if not np.isnan(rmse_val) else "N/A"
        nse_str = f"{nse_val:.3f}" if not np.isnan(nse_val) else "N/A"
        shift_str = f"{best_shift:.0f} m"
        ax.set_title(f'{date_str}\nRMSE = {rmse_str}  |  NSE = {nse_str}  |  x-shift = {shift_str}',
                     fontsize=9)
        ax.set_xlabel('Distance downstream (m)', fontsize=8)
        ax.set_ylabel('Bed Elevation (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # Hide any unused subplot panels
    for i_empty in range(n_profiles, nrows * ncols):
        row, col = divmod(i_empty, ncols)
        axes[row][col].set_visible(False)

    out_path = os.path.join(SCRIPT_DIR, f'Simulation_Profile_Comparison_{run_id}.png')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Profile comparison saved \u2192 {out_path}")


# ==========================================
# 3. SIMULATION ENGINE
# ==========================================
def run_simulation(params, run_id="Single_Run", output_dir=None, silent=False, return_details=False, cal_t_indices=None):
    """
    Execute a single forward run of the PARC (Profile And Reach Change) model.

    The PARC model integrates the Exner equation (sediment continuity) forward
    in time using an adaptive sub-stepping scheme that satisfies the CFL
    stability criterion. At each timestep, the following sequence is computed:

      1. Transport Capacity  : K_local = (b*a / (3*S_eq)) * (Q/B)^n
                               This power-law formulation relates unit discharge
                               to sediment transport capacity, analogous to a
                               stream power or excess shear stress law.

      2. Sediment Flux (Qs)  : Qs = -B * K_local * (∂z/∂x)
                               In Transport-Limited mode, flux equals capacity.
                               In Blended mode, Qs relaxes toward capacity over
                               the adaptation length L_ADAPT.

      3. Exner Equation      : dz/dt = -(1/B) * (∂Qs/∂x)
                               Divergence of flux drives bed elevation change:
                               net erosion where flux diverges, aggradation where
                               it converges (Parker, 2004).

      4. Bank Erosion        : dB = incision * bank_widening_factor
                               Incision destabilises valley walls; bank material
                               collapsed into the channel adds a sediment source
                               term that partially offsets vertical incision and
                               progressively widens the active channel.

      5. Boundary Conditions : Upstream — fixed elevation (base level pinned at
                               the initial surveyed inlet / waterfall lip).
                               Downstream — constant slope (zero second-derivative),
                               allowing free export of sediment from the domain.

    Parameters
    ----------
    params : dict
        PARC model parameter set containing:
          'a'                   — transport efficiency coefficient
          'n'                   — discharge exponent (non-linearity)
          'b'                   — slope-velocity scaling exponent
          'bank_widening_factor' — lateral erosion rate per unit incision
          'US_slope_factor'     — upstream equilibrium slope multiplier
          'DS_slope_factor'     — downstream equilibrium slope multiplier
          'width_factor'        — global channel width multiplier
          'volume_multiplier'   — post-hoc volume scaling factor
    run_id : str
        Identifier string for output filenames and logging.
    output_dir : str, optional
        Directory for calibration-mode plot output. Ignored in single-run mode.
    silent : bool
        Suppress all terminal output and figure generation (for calibration sweeps).
    return_details : bool
        If True, return a compact results dict (no full zb history stored) for
        use in the Vol_Elev calibration objective function. Reduces RAM usage
        from O(NT × NX) to O(NX) per evaluation.
    cal_t_indices : list of int, optional
        Timestep indices at which to save zb snapshots (used in return_details mode).
        Only profiles at these indices are retained to minimise memory allocation.

    Returns
    -------
    dict
        Summary dictionary of key PARC model outputs (transport volumes, RMSE,
        sediment budget statistics). In return_details mode, also includes
        cumulative volume arrays and zb snapshots at calibration timesteps.
    """
    # Unpack parameters
    a = params['a']
    n = params['n']
    b = params['b']
    bank_widening_factor = params['bank_widening_factor']
    US_slope_factor = params.get('US_slope_factor', params.get('slope_factor', 1.0))
    DS_slope_factor = params.get('DS_slope_factor', params.get('slope_factor', 1.0))
    width_factor = params.get('width_factor', 1.0)
    volume_multiplier = params.get('volume_multiplier', 1.0)

    if not silent:
        print(f"Running Simulation {run_id} [{TRANSPORT_MODE}]: a={a}, n={n}, b={b}, bank_factor={bank_widening_factor}")

    # --- Setup IC/BC ---
    # Equilibrium (Graded) Longitudinal Profile
    # -----------------------------------------
    # The PARC model initialises a reach-averaged equilibrium slope for each
    # geomorphic zone (upstream gorge, nick-zone ramp, downstream alluvial fan).
    # This represents the theoretical graded condition (Mackin, 1948) toward which
    # the stream would tend given a constant discharge and sediment supply.
    # Deviations from this profile drive disequilibrium sediment flux and net
    # morphological change. The slope factors (US/DS) are calibrated parameters
    # that adjust the reference gradient to best match observed incision rates.
    ib = np.zeros(NX_GLOBAL)
    ib[X_GLOBAL < 42500] = 0.0091 * US_slope_factor           # Upstream gorge reach (steeper)
    ib[(X_GLOBAL >= 42500) & (X_GLOBAL < 47000)] = 0.012 * US_slope_factor  # Nick-zone transition
    ib[X_GLOBAL >= 47000] = 0.0065 * DS_slope_factor          # Downstream alluvial/fan reach (gentler)

    # Reconstruct the equilibrium profile by integrating slopes from the downstream end
    # (consistent with base-level control; Lane, 1955)
    zb_eq = np.zeros(NX_GLOBAL)
    zb_eq[-1] = ZB0_GLOBAL[-1]
    for i in range(NX_GLOBAL - 2, -1, -1):
        zb_eq[i] = zb_eq[i+1] + ib[i] * DX_GLOBAL

    # Channel Width — Hydraulic Geometry
    # ------------------------------------
    # Prescribed initial widths reflect reach-averaged hydraulic geometry derived
    # from field surveys and satellite imagery on the Rio Coca (pre-collapse).
    # Width directly controls the unit discharge (q = Q/B), which scales transport
    # capacity: narrower channels produce higher q, greater shear stress, and faster
    # incision per unit time — a key driver of post-waterfall channel adjustment.
    # The width_factor parameter globally scales these survey-derived widths to
    # account for uncertainty and post-collapse channel expansion not captured by
    # the initial survey.
    B = np.full(NX_GLOBAL, 100.0 * width_factor)
    B[(X_GLOBAL > 44000) & (X_GLOBAL < 47000)] = 50 * width_factor  # Confined nick-zone gorge
    B[X_GLOBAL > 65000] = 200 * width_factor  # Unconfined wide alluvial fan/delta reach
    B0 = B.copy()
    B_dynamic = B.copy()
    # B_history only allocated in full visualization mode (saves O(NT*NX) RAM during optimization)
    if not return_details:
        B_history = np.zeros((NT_GLOBAL, NX_GLOBAL))
        B_history[0, :] = B_dynamic

    # Bank Widening Coefficient Array
    # ----------------------------------
    # The bank_widening_factor is spatially variable across the PARC model domain
    # to reflect heterogeneous bank erodibility. In alluvial reaches, unconsolidated
    # glaciofluvial and colluvial fill yields rapid lateral retreat. Near the
    # waterfall / nick-zone, competent volcanic bedrock is more resistant, so the
    # factor is reduced. A smooth linear ramp over RAMP_DISTANCE avoids
    # discontinuities in the bank erosion source term.
    bank_widening_array = np.full(NX_GLOBAL, bank_widening_factor)
    mask_ramp = (X_GLOBAL >= WATERFALL_POS - RAMP_DISTANCE) & (X_GLOBAL <= WATERFALL_POS + RAMP_DISTANCE)
    reduced_factor = bank_widening_factor / 4.0
    dist_from_wf = np.abs(X_GLOBAL[mask_ramp] - WATERFALL_POS)
    bank_widening_array[mask_ramp] = reduced_factor + (bank_widening_factor - reduced_factor) * (dist_from_wf / RAMP_DISTANCE)

    # --- Simulation Variables ---
    # In optimization (return_details) mode use a rolling single-profile state to avoid
    # allocating the full NT*NX zb matrix, which grows unbounded across many evaluations.
    cumulative_bank_sediment = np.zeros(NT_GLOBAL)
    cumulative_eroded_vol = np.zeros(NT_GLOBAL)  # Bed erosion + bank erosion
    if return_details:
        _zb_state  = ZB0_GLOBAL.copy()          # rolling current bed profile
        zb         = None                        # full history not stored
        _cal_t_set = set(cal_t_indices) if cal_t_indices is not None else set()
        zb_cal_snapshots = {}                    # {t_idx: z_array} at requested times only
    else:
        zb = np.zeros((NT_GLOBAL, NX_GLOBAL))
        zb[0, :] = ZB0_GLOBAL
        _zb_state  = None
        _cal_t_set = set()
        zb_cal_snapshots = None

    # Flux Tracking Locations
    # Updated based on user request for Time Series plot
    target_xs = {
        'Flux @ Waterfall': WATERFALL_POS,
        'Flux @ +1 km': WATERFALL_POS + 1000,
        'Flux @ +5 km': WATERFALL_POS + 5000,
        'Flux @ +10 km': WATERFALL_POS + 10000,
        'Flux @ 52 km': 52000,
        'Flux @ 62 km': 62000,
        'Flux @ Outlet': X_GLOBAL[-1] # Added Flux leaving domain
    }
    target_indices = {k: np.argmin(np.abs(X_GLOBAL - val)) for k, val in target_xs.items()}
    flux_history = {k: np.zeros(NT_GLOBAL) for k in target_xs.keys()}

    # Binned Tracking (10km bins relative to WF)
    # Create bins aligned with WF position (19000). E.g., ... 9000, 19000, 29000 ...
    # Let's generate edges covering the whole domain.
    # Bins will be defined by lower edges.
    bin_size = 10000.0
    # Start from WF - int(WF/bin_size)*bin_size to cover 0
    offset = WATERFALL_POS % bin_size
    # Ensure we cover negative if needed (though domain starts at 0)
    bin_edges = np.arange(offset - bin_size, X_GLOBAL[-1] + bin_size, bin_size)
    # We align strictly to have a break at WF (19000).
    # If WF=19000, bin edges could be: -1000, 9000, 19000, 29000, 39000...
    bin_edges = np.arange(WATERFALL_POS - (np.ceil(WATERFALL_POS/bin_size)*bin_size),
                          X_GLOBAL[-1] + bin_size, bin_size)

    # Identify which bin each spatial cell belongs to
    bin_mapping = np.digitize(X_GLOBAL, bin_edges) - 1
    n_bins = len(bin_edges) - 1
    bin_eroded_vol_cum = np.zeros(n_bins)
    bin_aggraded_vol_cum = np.zeros(n_bins)

    # Bed Change Tracking for plots (Snapshots)
    bed_change_profiles = {}
    # Snapshots at 1/4, 1/2, 3/4, 4/4
    snapshot_times = [int(NT_GLOBAL * 0.25), int(NT_GLOBAL * 0.50), int(NT_GLOBAL * 0.75), NT_GLOBAL - 1]

    # Transport Capacity Pre-factor (K_prefactor)
    # ---------------------------------------------
    # K_prefactor = (b * a) / (3 * S_eq) is the spatially variable coefficient
    # that scales unit discharge to sediment flux. It consolidates the transport
    # efficiency parameter 'a', the slope-velocity exponent 'b', and the local
    # equilibrium slope S_eq (used here as a surrogate for the normal-flow depth-
    # discharge relationship, consistent with Manning/Chezy hydraulics).
    #
    # Physical interpretation: K_prefactor is highest where the equilibrium slope
    # is lowest (gentle reaches), meaning larger flux perturbations per unit of
    # discharge change. This reflects the well-known concavity-transport feedback:
    # lower gradient → deeper flow → proportionally larger transport capacity.
    # Using np.maximum(ib, 1e-6) prevents division by zero in flat-slope sections.
    K_prefactor = (b * a) / (3 * np.maximum(ib, 1e-6))

    if USE_BANK_EROSION:
        # Valley Wall Geometry — Terrain Constants for Bank Height Calculation
        # The PARC model approximates each cross-section as a trapezoidal valley
        # where the terrace / hillslope surface elevation at each bank edge is:
        #   z_terrace = z_bed_initial + INITIAL_BANK_HEIGHT + (B0/2) * HILLSLOPE_SLOPE
        # The bank height above the current bed (h_bank) is then:
        #   h_bank = z_terrace − z_bed_current
        # This geometry determines the volume of material delivered to the channel
        # per unit of lateral widening — higher/steeper banks supply more sediment
        # per metre of retreat, amplifying the Exner source term.
        terrain_const = ZB0_GLOBAL + INITIAL_BANK_HEIGHT - (B0 * HILLSLOPE_SLOPE / 2.0)
        terrain_slope_factor = HILLSLOPE_SLOPE / 2.0

    # Animation Setup (suppressed in silent/optimization mode to prevent frame accumulation)
    _do_anim = OUTPUT_MODE in ('animation', 'both', 'Advanced') and not silent
    anim_frames = []
    anim_dates = []
    anim_widths = []
    anim_cum_eroded = []
    anim_cum_bank = []
    anim_time_indices = []
    anim_fps = 30
    anim_duration = 20 # Seconds
    total_anim_frames = anim_fps * anim_duration
    anim_stride = max(1, int(NT_GLOBAL / total_anim_frames))

    # Helper Gradient Function (Central Difference)
    def fast_gradient(arr, dx):
        grad = np.empty_like(arr)
        grad[0] = (arr[1] - arr[0]) / dx
        grad[-1] = (arr[-1] - arr[-2]) / dx
        grad[1:-1] = (arr[2:] - arr[:-2]) / (2 * dx)
        return grad

    # --- Time Loop (Geomorphic Evolution) ---
    _next_pct_print = [10]  # mutable sentinel for % milestone tracking
    for t in range(NT_GLOBAL - 1):
        # Report simulation progress to terminal at 10% intervals (non-silent only)
        if not silent and NT_GLOBAL > 1:
            pct = int(100 * (t + 1) / (NT_GLOBAL - 1))
            if pct >= _next_pct_print[0]:
                print(f'  Simulation: {pct:3d}% complete '
                      f'(step {t + 1:,}/{NT_GLOBAL - 1:,})')
                _next_pct_print[0] = (pct // 10 + 1) * 10

        Qt = Q_GLOBAL[t]

        # Step 1 — Compute Local Transport Capacity (K_local)
        # ------------------------------------------------------
        # K_local represents the spatially distributed sediment transport capacity
        # per unit channel width at the current discharge Qt. The power-law form
        # Qs ∝ (Q/B)^n is consistent with stream power and excess shear stress
        # transport laws (Whipple & Tucker, 1999; Howard, 1994). The exponent 'n'
        # encodes the non-linearity of transport with flow: large n values mean
        # flood events disproportionately dominate long-term geomorphic work,
        # which is characteristic of flashy, high-gradient rivers like the Rio Coca.
        K_local = K_prefactor * (Qt / B_dynamic)**n

        # Adaptive Sub-step Time-Stepping (CFL Stability Condition)
        # -----------------------------------------------------------
        # The diffusive Exner equation has a maximum stable timestep governed by
        # the Courant–Friedrichs–Lewy (CFL) criterion:
        #   dt_crit = α * dx² / K_max   (α = 0.4, well below the stability limit of 0.5)
        # K_max is the maximum effective diffusivity anywhere in the domain at this
        # timestep. The PARC model uses adaptive sub-stepping to satisfy CFL
        # automatically: if the user-specified DT_SECS is too large for the current
        # K_local distribution, multiple sub-steps are taken within one hydrology
        # timestep. This ensures numerical stability without sacrificing temporal
        # resolution in the hydrograph forcing.
        K_max = np.max(K_local)
        dt_crit = 0.4 * (DX_GLOBAL**2) / K_max if K_max > 1e-12 else DT_SECS
        dt_sub = min(DT_SECS, max(0.1, dt_crit))

        time_remaining = DT_SECS
        # Use rolling state in optimization mode; read from full array otherwise.
        zb_current = (_zb_state if return_details else zb[t, :]).copy()
        step_bank_sed_vol = 0.0

        # Track accumulated change in this timestep for binning
        dz_step = np.zeros(NX_GLOBAL)

        # Temp flux storage for this step (average rate)
        qs_step_vals = {k: 0.0 for k in target_xs.keys()}
        qs_step_count = 0

        while time_remaining > 0:
            current_dt = min(time_remaining, dt_sub)

            # Step 2 — Compute Sediment Flux (Qs)
            # -------------------------------------
            # In Transport-Limited mode (PARC default), sediment flux equals the
            # local transport capacity, which is proportional to the bed slope
            # (negative of the elevation gradient):
            #   Qs = −B * K_local * (∂z/∂x)
            # A negative slope (∂z/∂x < 0, bed descending downstream) produces
            # positive (downstream) flux. Steep reaches carry more sediment.
            # This is the classic diffusive flux law; the resulting Exner equation
            # is a linear diffusion equation:  ∂z/∂t = K * ∂²z/∂x²
            # which smooths the longitudinal profile toward the graded condition.
            dzbdx = fast_gradient(zb_current, DX_GLOBAL)
            Qc = -B_dynamic * K_local * dzbdx  # Transport Capacity Flux (potential)

            if TRANSPORT_MODE == 'Blended':
                 # Blended (Hybrid) Mode — Exponential Flux Adaptation
                 # -------------------------------------------------------
                 # The actual flux Qs relaxes toward the transport capacity Qc
                 # exponentially over the adaptation length L_ADAPT. This is
                 # implemented as a causal first-order IIR filter along the
                 # downstream direction:
                 #   Qs[i] = α * Qs[i-1] + (1 − α) * Qc[i]
                 #   where α = exp(−dx / L_ADAPT)
                 # Short L_ADAPT → α ≈ 0 → Qs ≈ Qc  (transport-limited)
                 # Long  L_ADAPT → α ≈ 1 → Qs persists (supply/detachment-limited)
                 # The disequilibrium (Qc − Qs) / L_ADAPT then drives bed change,
                 # analogous to a kinematic wave in the detachment-limited regime.
                 alpha = np.exp(-DX_GLOBAL / L_ADAPT)
                 b_filt = [1.0 - alpha]
                 a_filt = [1.0, -alpha]
                 # Initial condition: assume equilibrium (Qs = Qc) at the inlet
                 zi = signal.lfilter_zi(b_filt, a_filt) * Qc[0]
                 Qs, _ = signal.lfilter(b_filt, a_filt, Qc, zi=zi)

                 # Bed change rate from disequilibrium relaxation
                 dQs_dx = (Qc - Qs) / L_ADAPT
            else:
                 # Transport-Limited Mode — Qs equals capacity everywhere
                 Qs = Qc
                 dQs_dx = fast_gradient(Qs, DX_GLOBAL)

            # Step 3 — Exner Equation (Sediment Mass Conservation)
            # ------------------------------------------------------
            # dz/dt = −(1/B) * dQs/dx
            # Where dQs/dx > 0 (flux increasing downstream, net sediment export),
            # the bed incises (dz < 0). Where dQs/dx < 0 (flux decreasing
            # downstream, net sediment import), the bed aggrades (dz > 0).
            # This is the central mass balance of the PARC model: every cubic
            # metre of sediment eroded from one node must appear either in the
            # flux leaving that node or as aggradation in an adjacent node.
            # The factor 1/B converts volumetric flux divergence to a bed
            # elevation change rate (m/s) for a rectangular cross-section.
            dz_dt_flow = - (1.0 / B_dynamic) * dQs_dx
            dz_flow = dz_dt_flow * current_dt

            # Save flux rates
            for name, idx in target_indices.items():
                qs_step_vals[name] += Qs[idx]

            # Step 4 — Bank Erosion (Lateral-Vertical Coupling)
            # ----------------------------------------------------
            # The PARC bank erosion module links vertical bed lowering to lateral
            # channel widening, consistent with field evidence of progressive
            # gorge widening on the Rio Coca following the 2020 waterfall collapse.
            #
            # Mechanism:
            #   (a) Incision depth = max(0, −dz_flow): only lowering, not aggradation,
            #       drives bank retreat (bank collapse is irreversible).
            #   (b) Channel widens by dB = incision * bank_widening_array (m).
            #   (c) The volume of bank material delivered to the channel floor is:
            #       V_bank = h_bank * dB * dx
            #       where h_bank is the current bank height above the thalweg.
            #   (d) This bank-derived sediment raises the channel bed by:
            #       dz_bank = V_bank / (B_dynamic + dB)
            #       acting as a partial negative feedback (Exner source term) that
            #       moderates net incision and supplies downstream sediment.
            #
            # This coupling is critical for matching observed sediment volumes:
            # bank erosion can contribute the same order of magnitude of mobilised
            # material as direct bed incision in confined gorge reaches.
            if USE_BANK_EROSION:
                # Only incision (negative dz) triggers bank retreat
                incision = np.maximum(0.0, -dz_flow)

                # Lateral widening proportional to incision depth x bank erodibility
                dB = incision * bank_widening_array

                # Bank height above current thalweg (trapezoidal geometry approximation)
                z_terrain = terrain_const + (B_dynamic * terrain_slope_factor)
                h_bank = np.maximum(0.0, z_terrain - zb_current)

                # Volumetric bank sediment contribution per unit length (m² per node)
                area_bank_sed = h_bank * dB
                step_bank_sed_vol += np.sum(area_bank_sed) * DX_GLOBAL

                # Translate bank volume to a bed elevation rise (Exner source term)
                # Distributes bank material across the widened channel floor
                dz_bank = area_bank_sed / (B_dynamic + dB)

                B_dynamic += dB
                net_change = dz_flow + dz_bank  # Net bed elevation change this sub-step
                zb_current += net_change
                dz_step += net_change
            else:
                zb_current += dz_flow
                dz_step += dz_flow

            # Step 5 — Boundary Conditions
            # ------------------------------
            # Upstream (x=0): Fixed elevation — pinned to the initial surveyed bed
            #   elevation at the model inlet (e.g., waterfall lip or catchment outlet).
            #   This represents a fixed base-level control from upstream and prevents
            #   the inlet from eroding upward into the supply domain.
            zb_current[0] = ZB0_GLOBAL[0]
            # Downstream (x=L): Constant slope — zero second-derivative (∂²z/∂x² = 0).
            #   Extrapolates the second-to-last gradient to the final node, allowing
            #   sediment to freely exit the domain without artificial reflection of
            #   the incision wave or aggradation front.
            zb_current[-1] = zb_current[-2] + (zb_current[-2] - zb_current[-3])

            time_remaining -= current_dt
            qs_step_count += 1

        # Store bed state — compact rolling update in optimization mode, full array otherwise.
        if return_details:
            _zb_state[:] = zb_current
            if (t + 1) in _cal_t_set:
                zb_cal_snapshots[t + 1] = zb_current.copy()
        else:
            zb[t+1, :] = zb_current
            B_history[t+1, :] = B_dynamic.copy()

        cumulative_bank_sediment[t+1] = cumulative_bank_sediment[t] + step_bank_sed_vol

        # Track cumulative eroded volume (bed erosion + bank/hillslope erosion)
        bed_erosion_vol = np.sum(np.maximum(0.0, ZB0_GLOBAL - zb_current) * B_dynamic) * DX_GLOBAL
        cumulative_eroded_vol[t+1] = bed_erosion_vol + cumulative_bank_sediment[t+1]

        # Capture animation frame (skipped entirely in silent/optimization mode)
        if _do_anim and (t % anim_stride == 0):
            anim_frames.append(zb_current.copy())
            anim_dates.append(START_DATE + timedelta(days=t * DT_DAYS))
            anim_widths.append(B_dynamic.copy())
            anim_cum_eroded.append(cumulative_eroded_vol[t+1])
            anim_cum_bank.append(cumulative_bank_sediment[t+1])
            anim_time_indices.append(t + 1)

        # Store avg flux for the timestep
        for name in target_xs.keys():
            flux_history[name][t] = (qs_step_vals[name] / qs_step_count) if qs_step_count > 0 else 0.0

        # Bin Statistics (Net Erosion/Aggradation per bin this step)
        # Volume change at each cell = dz_step * Width * dx
        dv_step = dz_step * B_dynamic * DX_GLOBAL

        # For each bin, sum positive dV (agg) and negative dV (ero)
        # Vectorized sum by bin index
        # Using np.bincount for speed (works for summing weights)
        # Separate positive and negative changes
        agg_step = np.where(dv_step > 0, dv_step, 0)
        ero_step = np.where(dv_step < 0, dv_step, 0)

        # Ensure bin_mapping is within range (handles edges if any)
        # Note: bin_mapping size is NX_GLOBAL
        bin_aggraded_vol_cum += np.bincount(bin_mapping, weights=agg_step, minlength=n_bins)
        bin_eroded_vol_cum += np.bincount(bin_mapping, weights=ero_step, minlength=n_bins)

        # Store plot snapshots (full mode only; uses zb_current which equals zb[t+1,:])
        if not return_details and (t+1) in snapshot_times:
            bed_change_profiles[t+1] = zb_current - ZB0_GLOBAL

    # --- Metrics & Analysis ---
    # Apply volume_multiplier to scale computed volumes (calibrated obs/sim ratio)
    if not CALIBRATION_MODE and volume_multiplier != 1.0:
        cumulative_eroded_vol *= volume_multiplier
        cumulative_bank_sediment *= volume_multiplier

    # Using waterfall location flux as reference for calibration
    cum_flux_wf = np.cumsum(flux_history['Flux @ Waterfall']) * DT_SECS
    if not CALIBRATION_MODE and volume_multiplier != 1.0:
        cum_flux_wf *= volume_multiplier
    total_flux_wf = cum_flux_wf[-1]

    # Flux at Outlet (leaving domain)
    cum_flux_outlet = np.cumsum(flux_history['Flux @ Outlet']) * DT_SECS
    if not CALIBRATION_MODE and volume_multiplier != 1.0:
        cum_flux_outlet *= volume_multiplier
    total_flux_outlet = cum_flux_outlet[-1]

    # AOI Storage
    flux_52 = np.sum(flux_history['Flux @ 52 km']) * DT_SECS
    flux_62 = np.sum(flux_history['Flux @ 62 km']) * DT_SECS
    aoi_storage_pct = ((flux_52 - flux_62) / flux_52) * 100 if flux_52 > 0 else 0.0

    # Total Sediment Storage %
    # (Eroded from WF - Leaving Domain) / Eroded from WF * 100
    total_storage_pct = 0.0
    if total_flux_wf > 0:
        total_storage_pct = ((total_flux_wf - total_flux_outlet) / total_flux_wf) * 100.0

    # RMSE
    rmse = np.nan
    if DF_CAL is not None:
        time_days_arr = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
        cal_indices = [np.argmin(np.abs(time_days_arr - t_cal)) for t_cal in CAL_TIME_DAYS]
        sim_vol_at_cal = cum_flux_wf[cal_indices]
        rmse = np.sqrt(np.mean((CAL_VOL - sim_vol_at_cal)**2))

    # Prepare Summary Data
    summary_data = {
        'Run_ID': run_id,
        'Transport_Mode': TRANSPORT_MODE,
        'a': a, 'n': n, 'b': b,
        'bank_widening_factor': bank_widening_factor,
        'US_slope_factor': US_slope_factor,
        'DS_slope_factor': DS_slope_factor,
        'width_factor': width_factor,
        'volume_multiplier': volume_multiplier,
        'RMSE': rmse,
        'Total_Flux_WF': total_flux_wf,
        'Flux_Leaving_Domain': total_flux_outlet,
        'Total_Storage_Pct': total_storage_pct,
        'AOI_Storage_Pct': aoi_storage_pct
    }

    # --- Return details for calibration if requested ---
    if return_details:
        time_days_arr = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
        return {
            'summary': summary_data,
            'cumulative_eroded_vol': cumulative_eroded_vol,
            'cumulative_bank_sediment': cumulative_bank_sediment,
            'zb_snapshots': zb_cal_snapshots,   # only at requested indices, not full history
            'time_days_arr': time_days_arr,
            'cum_flux_wf': cum_flux_wf,
        }

    if silent:
        return summary_data

    # --- Animation Generation ---
    if OUTPUT_MODE in ('animation', 'both', 'Advanced'):
        print("Generating Animation... (This may take a minute)")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Initial profile
        ax.plot(X_GLOBAL/1000, ZB0_GLOBAL, 'k--', label='Initial Bed')

        # Evolving profile (placeholder)
        line, = ax.plot([], [], 'b-', linewidth=2, label='Current Profile')

        # Text for date
        date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        ax.set_xlabel('Longitudinal distance (km)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'River Profile Evolution - Run {run_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set limits based on overall range
        ax.set_xlim(X_GLOBAL[0]/1000, X_GLOBAL[-1]/1000)
        min_z = np.min(zb)
        max_z = np.max(zb)
        range_z = max_z - min_z
        ax.set_ylim(min_z - 0.1*range_z, max_z + 0.1*range_z)

        def init():
            line.set_data([], [])
            date_text.set_text('')
            return line, date_text

        def animate(i):
            if i < len(anim_frames):
                z_data = anim_frames[i]
                d_val = anim_dates[i]
                line.set_data(X_GLOBAL/1000, z_data)
                date_text.set_text(f'Date: {d_val.strftime("%Y-%m-%d")}')
            return line, date_text

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(anim_frames), interval=1000/anim_fps, blit=True)

        save_path = os.path.join(os.getcwd(), f'River_Evolution_{run_id}.mp4')
        try:
            anim.save(save_path, writer='ffmpeg', fps=anim_fps)
            print(f"Animation saved to {save_path}")
        except:
             print("FFmpeg not found or error, saving as GIF.")
             save_path = save_path.replace('.mp4', '.gif')
             anim.save(save_path, writer='pillow', fps=anim_fps)
             print(f"Animation saved to {save_path}")

        plt.close(fig)

    # --- Static Plotting (Original) ---
    if OUTPUT_MODE in ('plots', 'both', 'Advanced'):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 20))
        plt.subplots_adjust(hspace=0.4, bottom=0.25) # Extra bottom space for table

        # --- Plot 1: Longitudinal Profile ---
        ax1.plot(X_GLOBAL/1000, ZB0_GLOBAL, 'k--', label='Initial Bed')
        ax1.plot(X_GLOBAL/1000, zb[-1, :], 'b-', linewidth=2, label='Final Bed')
        ax1.plot(X_GLOBAL/1000, zb_eq, 'g:', alpha=0.5, label='Equilibrium Profile')
        ax1.set_xlabel('Longitudinal distance (km)')
        ax1.set_ylabel('Elevation (m)')
        ax1.set_title(f'Longitudinal Profile - Run {run_id}')
        ax1.grid(True, alpha=0.3)

        # Second Y-axis for Bed Change
        ax1_right = ax1.twinx()
        ax1_right.set_ylabel('Bed Change (m)', color='gray')

        # Plot bed change lines in grayscale (getting darker)
        for i, t_snap in enumerate(snapshot_times):
            change = bed_change_profiles[t_snap]
            gray_val = 0.8 - (0.8 * (i / 3.0)) # 0.8 down to 0.0 (black)
            label_t = f'Bed Change {int((i+1)*25)}% Sim. Time'
            ax1_right.plot(X_GLOBAL/1000, change, color=str(gray_val), linestyle='-', alpha=0.7, label=label_t)

        ax1_right.tick_params(axis='y', labelcolor='gray')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_right.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # --- Plot 2: Time Series ---
        time_arr = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Cumulative Sediment Flux (m3)')
        ax2.set_title('Time Series')

        # Plot fluxes
        cum_flux_52 = np.cumsum(flux_history['Flux @ 52 km']) * DT_SECS
        cum_flux_62 = np.cumsum(flux_history['Flux @ 62 km']) * DT_SECS
        if not CALIBRATION_MODE and volume_multiplier != 1.0:
            cum_flux_52 *= volume_multiplier
            cum_flux_62 *= volume_multiplier

        # Plot AOI storage shading first so lines are on top
        ax2.fill_between(time_arr, cum_flux_52, cum_flux_62, color='orange', alpha=0.2, label='AOI Storage')

        # Plot lines
        for loc_name, flux_ts in flux_history.items():
            if loc_name != 'Flux @ Outlet': # Keep plot cleaner, outlet is just for calculation
                cum_flux = np.cumsum(flux_ts) * DT_SECS
                if not CALIBRATION_MODE and volume_multiplier != 1.0:
                    cum_flux = cum_flux * volume_multiplier
                ax2.plot(time_arr, cum_flux, label=loc_name)

        # Bank erosion
        ax2.plot(time_arr, cumulative_bank_sediment, 'k--', label='Bank/Hillslope Erosion')

        # Calibration data
        if DF_CAL is not None:
            ax2.scatter(CAL_TIME_DAYS, CAL_VOL, color='red', marker='x', s=50, label='Calibration Data', zorder=5)

        # Second Y-axis for Discharge
        ax2_right = ax2.twinx()
        ax2_right.plot(time_arr, Q_GLOBAL, color='gray', alpha=0.3, linewidth=1, label='Discharge')
        ax2_right.set_ylabel('Discharge (m3/s)', color='gray')
        ax2_right.tick_params(axis='y', labelcolor='gray')

        # Legends
        lines_l, labels_l = ax2.get_legend_handles_labels()
        lines_r, labels_r = ax2_right.get_legend_handles_labels()
        ax2.legend(lines_l + lines_r, labels_l + labels_r, loc='upper left')
        ax2.grid(True, alpha=0.3)

        # --- Plot 3: Sediment budget ---
        # Prepare X-axis labels relative to Waterfall
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Apply volume_multiplier to binned volumes
        if not CALIBRATION_MODE and volume_multiplier != 1.0:
            bin_eroded_vol_cum *= volume_multiplier
            bin_aggraded_vol_cum *= volume_multiplier

        # Generate labels (Distance relative to WF)
        bin_labels = []
        for edge in bin_edges[:-1]:
            rel_start_km = int((edge - WATERFALL_POS) / 1000)
            rel_end_km = int((edge + bin_size - WATERFALL_POS) / 1000)
            bin_labels.append(f"{rel_start_km} to {rel_end_km} km")

        x_pos = np.arange(len(bin_centers))
        width = 0.35

        ax3.bar(x_pos - width/2, np.abs(bin_eroded_vol_cum), width=width, color='red', alpha=0.7, label='Total Erosion')
        ax3.bar(x_pos + width/2, bin_aggraded_vol_cum, width=width, color='blue', alpha=0.7, label='Total Aggradation')

        # Net Change points (difference between bars)
        net_bin = bin_aggraded_vol_cum + bin_eroded_vol_cum
        ax3.plot(x_pos, np.abs(net_bin), 'ko', markerfacecolor='white', label='Net Change Magnitude')

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax3.set_title(f'Sediment budget (10km Bins relative to Waterfall)')
        ax3.set_xlabel('Distance Relative to Waterfall')
        ax3.set_ylabel('Total Volume (m3)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # --- Output Logic ---
        if not CALIBRATION_MODE:
            # Build summary table and embed it in the figure
            table_data = []
            for k, v in summary_data.items():
                if isinstance(v, float):
                    val_str = f"{v:.4e}" if (abs(v)>1e5 or abs(v)<1e-3) else f"{v:.4f}"
                else:
                    val_str = str(v)
                table_data.append([k, val_str])

            the_table = plt.table(cellText=table_data, colWidths=[0.4, 0.4], loc='bottom', bbox=[0.2, -0.75, 0.6, 0.55])
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)

            # Save the main 3-panel figure
            main_plot_path = os.path.join(SCRIPT_DIR, f'Simulation_Main_{run_id}.png')
            plt.savefig(main_plot_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"  Main simulation figure saved \u2192 {main_plot_path}")

            # --- Vol-Elev comparison (calibration-style) ---
            time_days_arr = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
            print("  Generating Vol-Elev comparison figure...")
            plot_vol_elev_comparison(run_id, time_days_arr, cumulative_eroded_vol, zb,
                                     volume_multiplier=volume_multiplier)

            # --- Per-profile observed vs simulated comparison ---
            print("  Generating profile comparison figure...")
            plot_profile_comparison(run_id, time_days_arr, zb)
        else:
            # Save plot silently (calibration mode)
            if output_dir:
                plot_path = os.path.join(output_dir, f"Simulation_{run_id}.png")
                plt.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)

    # --- Advanced Animation Mode ---
    if OUTPUT_MODE == 'Advanced' and not silent and len(anim_frames) > 0:
        n_anim = len(anim_frames)
        x_km = X_GLOBAL / 1000.0

        # Pre-compute slope and concavity for every saved frame
        anim_slopes = []
        anim_concavities = []
        for zf in anim_frames:
            sl = np.gradient(zf, DX_GLOBAL)
            anim_slopes.append(sl)
            anim_concavities.append(np.gradient(sl, DX_GLOBAL))

        # Pre-compute cumulative flux arrays evaluated at frame time-indices
        time_arr_full = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
        cum_fluxes = {}
        for loc_name, fts in flux_history.items():
            cum_fluxes[loc_name] = np.cumsum(fts) * DT_SECS
            if not CALIBRATION_MODE and volume_multiplier != 1.0:
                cum_fluxes[loc_name] = cum_fluxes[loc_name] * volume_multiplier

        cum_bank_full = cumulative_bank_sediment.copy()
        cum_eroded_full = cumulative_eroded_vol.copy()
        cum_flux_52_full = cum_fluxes.get('Flux @ 52 km', np.zeros(NT_GLOBAL))
        cum_flux_62_full = cum_fluxes.get('Flux @ 62 km', np.zeros(NT_GLOBAL))

        # ======================================================
        # ANIMATION 1 — Oblique 3-D cross-section + slope/concavity
        # ======================================================
        print("Generating Advanced Animation 1 (3-D oblique cross-section) ...")

        # Cross-section positions (evenly spaced along the profile)
        n_xsec = 8
        xsec_indices = np.linspace(0, NX_GLOBAL - 1, n_xsec + 2).astype(int)[1:-1]

        # Pre-compute z-range across all frames
        all_z = np.array(anim_frames)
        z_min_global = float(np.min(all_z))
        z_max_global = float(np.max(all_z))
        z_range = z_max_global - z_min_global
        z_pad = 0.1 * z_range

        # Half-width in the "y" (lateral) direction for cross-sections
        max_half_w = float(np.max([np.max(w) for w in anim_widths])) / 2.0
        # Hillslope visualization: cross-sections extend outward from each bank edge
        # at HILLSLOPE_SLOPE (rise/run = 0.40).  vis_hill_ht is the vertical height
        # drawn; hs_lat is the corresponding lateral extent per side (metres).
        vis_hill_ht = max(z_range * 0.55 + INITIAL_BANK_HEIGHT * 2.0, 12.0)
        hs_lat = vis_hill_ht / HILLSLOPE_SLOPE      # lateral extent per side (m)
        y_extent = (max_half_w + hs_lat) * 1.12    # total half-span (m) with padding

        slope_all = np.array(anim_slopes)
        conc_all = np.array(anim_concavities)
        # Percentile-based limits (2nd–98th) so gradient spikes near the waterfall
        # or domain edges do not collapse the useful variation in the bottom plot.
        sl_p2, sl_p98 = np.percentile(slope_all, [2, 98])
        sl_span = max(abs(sl_p98 - sl_p2), 1e-6)
        sl_pad = 0.12 * sl_span
        sl_min_lim, sl_max_lim = sl_p2 - sl_pad, sl_p98 + sl_pad

        cc_p2, cc_p98 = np.percentile(conc_all, [2, 98])
        cc_span = max(abs(cc_p98 - cc_p2), 1e-9)
        cc_pad = 0.12 * cc_span
        cc_min_lim, cc_max_lim = cc_p2 - cc_pad, cc_p98 + cc_pad

        fig1 = plt.figure(figsize=(16, 14))
        gs = fig1.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.28)

        # -- Top: 3-D oblique subplot --
        ax3d = fig1.add_subplot(gs[0], projection='3d')
        ax3d.view_init(elev=28, azim=-55)
        ax3d.set_xlim(x_km[0], x_km[-1])
        ax3d.set_ylim(-y_extent / 1000.0, y_extent / 1000.0)
        ax3d.set_zlim(z_min_global - z_pad, z_max_global + vis_hill_ht + z_pad)
        ax3d.set_xlabel('Distance (km)', labelpad=10)
        ax3d.set_ylabel('Lateral (km)', labelpad=10)
        ax3d.set_zlabel('Elevation (m)', labelpad=8)
        ax3d.set_title('Channel Profile with Cross-Sections', fontsize=13, pad=12)

        # Static: initial bed (center-line)
        ax3d.plot(x_km, np.zeros(NX_GLOBAL), ZB0_GLOBAL, 'k--', linewidth=0.8, alpha=0.5, label='Initial Bed')

        # Dynamic artists (updated each frame)
        line3d, = ax3d.plot([], [], [], 'b-', linewidth=1.8, label='Current Bed')
        date_txt_3d = ax3d.text2D(0.02, 0.95, '', transform=ax3d.transAxes, fontsize=11,
                                   fontweight='bold', color='navy')

        # Pre-create polygon collections for cross-sections (will be replaced each frame)
        _xsec_polys = []

        # -- Bottom: slope & concavity --
        ax_sl = fig1.add_subplot(gs[1])
        ax_sl.set_xlabel('Distance (km)')
        ax_sl.set_ylabel('Slope (m/m)', color='tab:blue')
        ax_sl.tick_params(axis='y', labelcolor='tab:blue')
        ax_sl.set_xlim(x_km[0], x_km[-1])
        ax_sl.set_ylim(sl_min_lim, sl_max_lim)
        ax_sl.grid(True, alpha=0.25)
        line_sl, = ax_sl.plot([], [], 'tab:blue', linewidth=1.2, label='Slope')

        ax_cc = ax_sl.twinx()
        ax_cc.set_ylabel('Concavity (1/m)', color='tab:red')
        ax_cc.tick_params(axis='y', labelcolor='tab:red')
        ax_cc.set_ylim(cc_min_lim, cc_max_lim)
        line_cc, = ax_cc.plot([], [], 'tab:red', linewidth=1.0, alpha=0.8, label='Concavity')
        ax_sl.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # Combined legend
        lns = [line_sl, line_cc]
        ax_sl.legend(lns, [l.get_label() for l in lns], loc='upper right', fontsize=9)

        def _build_xsec_poly(xi, zbed, half_w_m):
            """Return valley cross-section polygon vertices.

            Models a realistic valley profile: flat channel floor flanked by
            hillslope walls that rise from each bank edge at HILLSLOPE_SLOPE
            (rise/run = 0.40).  The height and steepness of the faces reflect
            the current incision and channel width at this location.
            """
            xk = x_km[xi]
            hw_km_i   = half_w_m / 1000.0
            hs_lat_km = hs_lat / 1000.0     # lateral run of hillslope (km)
            z_b  = zbed[xi]
            z_top = z_b + vis_hill_ht       # hilltop elevation for this XS
            # Polygon (counter-clockwise from +y view):
            #   hillslope rim left  →  channel edge left  →  channel edge right
            #   →  hillslope rim right  →  (close)
            verts = [
                (xk, -hw_km_i - hs_lat_km, z_top),   # hillslope rim, left
                (xk, -hw_km_i,             z_b),      # channel edge, left
                (xk,  hw_km_i,             z_b),      # channel edge, right
                (xk,  hw_km_i + hs_lat_km, z_top),   # hillslope rim, right
                (xk, -hw_km_i - hs_lat_km, z_top),   # close
            ]
            return verts

        def animate1(i):
            nonlocal _xsec_polys
            zf = anim_frames[i]
            wf = anim_widths[i]

            # Update center-line profile
            line3d.set_data_3d(x_km, np.zeros(NX_GLOBAL), zf)

            # Remove old cross-section polygons
            for p in _xsec_polys:
                p.remove()
            _xsec_polys.clear()

            # Draw cross-sections
            for xi in xsec_indices:
                hw = wf[xi] / 2.0
                verts = _build_xsec_poly(xi, zf, hw)
                poly = Poly3DCollection([verts], alpha=0.55,
                                         facecolor='sienna', edgecolor='saddlebrown',
                                         linewidth=0.9)
                ax3d.add_collection3d(poly)
                _xsec_polys.append(poly)

            date_txt_3d.set_text(f'Date: {anim_dates[i].strftime("%Y-%m-%d")}')

            # Slope / concavity
            line_sl.set_data(x_km, anim_slopes[i])
            line_cc.set_data(x_km, anim_concavities[i])

            return (line3d, date_txt_3d, line_sl, line_cc)

        anim1 = animation.FuncAnimation(fig1, animate1, frames=n_anim,
                                         interval=1000 / anim_fps, blit=False)

        save1 = os.path.join(SCRIPT_DIR, f'Advanced_3D_Profile_{run_id}.mp4')
        try:
            anim1.save(save1, writer='ffmpeg', fps=anim_fps)
            print(f"  Advanced 3-D animation saved \u2192 {save1}")
        except Exception:
            save1 = save1.replace('.mp4', '.gif')
            anim1.save(save1, writer='pillow', fps=anim_fps)
            print(f"  Advanced 3-D animation saved \u2192 {save1}")
        plt.close(fig1)

        # ======================================================
        # ANIMATION 2 — Dynamic 3-panel main plot with time bar
        # ======================================================
        print("Generating Advanced Animation 2 (dynamic main plots) ...")

        fig2, (ax_p1, ax_p2, ax_p3) = plt.subplots(3, 1, figsize=(15, 18))
        plt.subplots_adjust(hspace=0.38)

        # ---- Subplot 1: Longitudinal Profile ----
        ax_p1.plot(x_km, ZB0_GLOBAL, 'k--', linewidth=0.8, label='Initial Bed')
        ax_p1.plot(x_km, zb_eq, 'g:', alpha=0.5, label='Equilibrium Profile')
        line_bed, = ax_p1.plot([], [], 'b-', linewidth=2, label='Current Bed')

        ax_p1_r = ax_p1.twinx()
        ax_p1_r.set_ylabel('Bed Change (m)', color='gray')
        ax_p1_r.tick_params(axis='y', labelcolor='gray')
        line_dz, = ax_p1_r.plot([], [], color='0.3', linewidth=1.2, label='Bed Change')

        ax_p1.set_xlim(x_km[0], x_km[-1])
        z_min_bed = float(np.min(all_z))
        z_max_bed = float(np.max(all_z))
        z_r = z_max_bed - z_min_bed
        ax_p1.set_ylim(z_min_bed - 0.05 * z_r, z_max_bed + 0.05 * z_r)

        bed_changes = [f - ZB0_GLOBAL for f in anim_frames]
        dz_min = float(min(np.min(c) for c in bed_changes))
        dz_max = float(max(np.max(c) for c in bed_changes))
        dz_r = max(abs(dz_max - dz_min), 0.1)
        ax_p1_r.set_ylim(dz_min - 0.05 * dz_r, dz_max + 0.05 * dz_r)

        ax_p1.set_xlabel('Distance (km)')
        ax_p1.set_ylabel('Elevation (m)')
        ax_p1.set_title(f'Longitudinal Profile - Run {run_id}')
        ax_p1.grid(True, alpha=0.3)
        lns1 = [line_bed, line_dz]
        extra_static = ax_p1.get_legend_handles_labels()[0]
        ax_p1.legend(extra_static + lns1,
                     [h.get_label() for h in extra_static + lns1],
                     loc='upper right', fontsize=8)

        date_txt_2 = ax_p1.text(0.02, 0.94, '', transform=ax_p1.transAxes,
                                fontsize=11, fontweight='bold', color='navy')

        # ---- Subplot 2: Time Series (cumulative flux) ----
        ax_p2.set_xlabel('Time (days)')
        ax_p2.set_ylabel('Cumulative Sediment Flux (m\u00b3)')
        ax_p2.set_title('Time Series')
        ax_p2.grid(True, alpha=0.3)
        ax_p2.set_xlim(0, TOTAL_TIME_DAYS)

        # Pre-draw full-extent light-gray reference lines so axes auto-scale
        flux_ymin, flux_ymax = 0.0, 0.0
        flux_lines_data = {}
        for loc_name in flux_history:
            if loc_name != 'Flux @ Outlet':
                cf = cum_fluxes[loc_name]
                flux_ymin = min(flux_ymin, float(np.min(cf)))
                flux_ymax = max(flux_ymax, float(np.max(cf)))
                flux_lines_data[loc_name] = cf
        flux_ymax = max(flux_ymax, float(np.max(cum_bank_full)))
        flux_pad = 0.05 * max(abs(flux_ymax - flux_ymin), 1.0)
        ax_p2.set_ylim(flux_ymin - flux_pad, flux_ymax + flux_pad)

        # Dynamic growing lines
        flux_art = {}
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ci = 0
        for loc_name, cf_full in flux_lines_data.items():
            flux_art[loc_name], = ax_p2.plot([], [], color=color_cycle[ci % len(color_cycle)],
                                              linewidth=1.2, label=loc_name)
            ci += 1
        bank_line, = ax_p2.plot([], [], 'k--', linewidth=1.2, label='Bank/Hillslope Erosion')

        # AOI storage fill — references to the two bounding flux series
        _cf_52 = flux_lines_data.get('Flux @ 52 km', np.zeros(NT_GLOBAL))
        _cf_62 = flux_lines_data.get('Flux @ 62 km', np.zeros(NT_GLOBAL))
        # Seed with a single-point fill so the label appears in the legend
        _aoi_fill = [ax_p2.fill_between(time_arr_full[:1], _cf_52[:1], _cf_62[:1],
                                        color='orange', alpha=0.22, zorder=1,
                                        label='AOI Storage')]

        # Calibration markers (static)
        if DF_CAL is not None:
            ax_p2.scatter(CAL_TIME_DAYS, CAL_VOL, color='red', marker='x', s=50,
                          label='Calibration Data', zorder=5)

        # Discharge on right axis (full, static, faded)
        ax_p2_r = ax_p2.twinx()
        ax_p2_r.plot(time_arr_full, Q_GLOBAL, color='gray', alpha=0.2, linewidth=0.8)
        ax_p2_r.set_ylabel('Discharge (m\u00b3/s)', color='gray')
        ax_p2_r.tick_params(axis='y', labelcolor='gray')

        # Translucent time-bar — Rectangle with blended transform (x in data coords,
        # y in axes coords 0→1) so it always spans the full axis height regardless
        # of how y-limits change as flux lines grow.
        from matplotlib.transforms import blended_transform_factory as _btf
        _bar_hw_init = TOTAL_TIME_DAYS * 0.006
        _bar_transform = _btf(ax_p2.transData, ax_p2.transAxes)
        time_bar = plt.Rectangle((-_bar_hw_init, 0.0), _bar_hw_init * 2, 1.0,
                                 color='steelblue', alpha=0.30, zorder=3,
                                 transform=_bar_transform, clip_on=True)
        ax_p2.add_patch(time_bar)

        lines_l2, labels_l2 = ax_p2.get_legend_handles_labels()
        ax_p2.legend(lines_l2, labels_l2, loc='upper left', fontsize=7, ncol=2)

        # ---- Subplot 3: Sediment Budget Bars ----
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        n_bins_plot = len(bin_centers)
        x_pos_b = np.arange(n_bins_plot)
        bar_w = 0.35

        # Pre-compute per-frame binned erosion / aggradation from full zb history
        # (Use zb array available in non-details mode)
        frame_ero_bins = []
        frame_agg_bins = []
        for fi, tidx in enumerate(anim_time_indices):
            dz_tot = anim_frames[fi] - ZB0_GLOBAL
            dv = dz_tot * anim_widths[fi] * DX_GLOBAL
            ero = np.where(dv < 0, dv, 0)
            agg = np.where(dv > 0, dv, 0)
            ero_b = np.bincount(bin_mapping, weights=ero, minlength=n_bins_plot)[:n_bins_plot]
            agg_b = np.bincount(bin_mapping, weights=agg, minlength=n_bins_plot)[:n_bins_plot]
            if not CALIBRATION_MODE and volume_multiplier != 1.0:
                ero_b = ero_b * volume_multiplier
                agg_b = agg_b * volume_multiplier
            frame_ero_bins.append(np.abs(ero_b))
            frame_agg_bins.append(agg_b)

        max_bar_y = max(
            max(float(np.max(e)) for e in frame_ero_bins) if frame_ero_bins else 1,
            max(float(np.max(a)) for a in frame_agg_bins) if frame_agg_bins else 1,
        )
        ax_p3.set_ylim(0, max_bar_y * 1.15)
        ax_p3.set_xlim(x_pos_b[0] - 0.5, x_pos_b[-1] + 0.5)

        bin_labels_anim = []
        for edge in bin_edges[:n_bins_plot]:
            rs = int((edge - WATERFALL_POS) / 1000)
            re = int((edge + bin_size - WATERFALL_POS) / 1000)
            bin_labels_anim.append(f"{rs} to {re} km")
        ax_p3.set_xticks(x_pos_b)
        ax_p3.set_xticklabels(bin_labels_anim, rotation=45, ha='right', fontsize=7)
        ax_p3.set_title('Sediment Budget (10 km Bins relative to Waterfall)')
        ax_p3.set_xlabel('Distance Relative to Waterfall')
        ax_p3.set_ylabel('Total Volume (m\u00b3)')
        ax_p3.grid(True, alpha=0.3, axis='y')

        bars_ero = ax_p3.bar(x_pos_b - bar_w / 2, np.zeros(n_bins_plot), width=bar_w,
                             color='red', alpha=0.7, label='Erosion')
        bars_agg = ax_p3.bar(x_pos_b + bar_w / 2, np.zeros(n_bins_plot), width=bar_w,
                             color='blue', alpha=0.7, label='Aggradation')
        net_pts, = ax_p3.plot([], [], 'ko', markerfacecolor='white', markersize=4,
                              label='Net Change')
        ax_p3.legend(loc='upper right', fontsize=8)

        def animate2(i):
            tidx = anim_time_indices[i]
            t_days = time_arr_full[tidx]

            # Sub-1: bed profile + bed change
            line_bed.set_data(x_km, anim_frames[i])
            line_dz.set_data(x_km, bed_changes[i])
            date_txt_2.set_text(f'Date: {anim_dates[i].strftime("%Y-%m-%d")}')

            # Sub-2: grow flux lines up to current time
            for loc_name, art in flux_art.items():
                art.set_data(time_arr_full[:tidx + 1], flux_lines_data[loc_name][:tidx + 1])
            bank_line.set_data(time_arr_full[:tidx + 1], cum_bank_full[:tidx + 1])

            # AOI storage fill (remove old polygon, redraw grown version)
            _aoi_fill[0].remove()
            _aoi_fill[0] = ax_p2.fill_between(time_arr_full[:tidx + 1],
                                              _cf_52[:tidx + 1], _cf_62[:tidx + 1],
                                              color='orange', alpha=0.22, zorder=1)

            # Time bar (thick translucent vertical band)
            bar_hw = TOTAL_TIME_DAYS * 0.006
            time_bar.set_x(t_days - bar_hw)
            time_bar.set_width(bar_hw * 2)

            # Sub-3: bar chart
            for rect, val in zip(bars_ero, frame_ero_bins[i]):
                rect.set_height(val)
            for rect, val in zip(bars_agg, frame_agg_bins[i]):
                rect.set_height(val)
            net = frame_agg_bins[i] - frame_ero_bins[i]
            net_pts.set_data(x_pos_b, np.abs(net))

            return (line_bed, line_dz, date_txt_2, bank_line, time_bar, net_pts,
                    *flux_art.values(), *bars_ero, *bars_agg)

        anim2 = animation.FuncAnimation(fig2, animate2, frames=n_anim,
                                         interval=1000 / anim_fps, blit=False)

        save2 = os.path.join(SCRIPT_DIR, f'Advanced_Dynamic_Main_{run_id}.mp4')
        try:
            anim2.save(save2, writer='ffmpeg', fps=anim_fps)
            print(f"  Advanced dynamic-main animation saved \u2192 {save2}")
        except Exception:
            save2 = save2.replace('.mp4', '.gif')
            anim2.save(save2, writer='pillow', fps=anim_fps)
            print(f"  Advanced dynamic-main animation saved \u2192 {save2}")
        plt.close(fig2)

        print("Advanced animations complete.")

    return summary_data


# ==========================================
# 3b. Vol_Elev CALIBRATION HELPERS
# ==========================================

def load_calibration_logs(search_dir):
    """
    Scan search_dir for Calibration_log_*.txt files and extract prior evaluation
    records as a list of (param_array, objective_score) tuples.
    These are used to warm-start the optimizer with knowledge from previous runs.
    """
    log_files = sorted(glob.glob(os.path.join(search_dir, 'Calibration_log_*.txt')))
    prior_evals = []
    for fpath in log_files:
        try:
            with open(fpath, 'r') as f:
                lines = f.readlines()
            in_hist = False
            header_read = False
            n_loaded = 0
            for line in lines:
                line = line.rstrip('\n')
                if '=== EVALUATION HISTORY ===' in line:
                    in_hist = True
                    continue
                if in_hist:
                    if line.startswith('==='):
                        break
                    if not header_read:
                        # First line after section header = column names
                        header_read = True
                        continue
                    if line.startswith('-'):
                        continue  # separator row
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        try:
                            # Layout: eval | p1 | p2 | ... | vol_trend | elev_norm | combined
                            obj = float(parts[-1])
                            param_vals = [float(p) for p in parts[1:-3]]
                            if param_vals:
                                prior_evals.append((np.array(param_vals), obj))
                                n_loaded += 1
                        except (ValueError, IndexError):
                            pass
            if n_loaded:
                print(f"  Loaded {n_loaded} evaluations from {os.path.basename(fpath)}")
        except Exception as e:
            print(f"  Warning: could not parse {os.path.basename(fpath)}: {e}")
    return prior_evals


def compute_sensitivity(best_params, param_names, bounds, objective_fn, delta_frac=0.05):
    """
    One-at-a-time (OAT) sensitivity analysis and cross-sensitivity matrix via
    central finite differences around the calibrated PARC model parameter set.

    This analysis quantifies how sensitive the PARC model objective function is
    to perturbations in each parameter, and how pairs of parameters interact.

    Interpretation of results:
      - High |sens_indiv[i]|  → parameter i is a strong lever on model skill;
        small changes in this parameter produce large changes in fit quality.
        These parameters merit the tightest calibration effort.
      - Low  |sens_indiv[i]|  → parameter i is weakly constrained by the
        available observations; its value is uncertain (equifinality risk).
      - Large |sens_matrix[i,j]| (i≠j) → parameters i and j co-vary in their
        effect on the objective; jointly calibrating them is essential to avoid
        compensating errors (e.g., 'a' and 'bank_widening_factor' often interact,
        since both control net volumetric output in the PARC model).

    Returns:
        sens_indiv   : 1-D array — d(obj)/d(param_normalized) for each parameter
        sens_matrix  : 2-D array — mixed-partial / second-order interaction matrix
    """
    n = len(param_names)
    best_vec = np.array([best_params[p] for p in param_names])
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    ranges = hi - lo
    steps = ranges * delta_frac

    f0 = objective_fn(best_vec)

    f_plus  = np.zeros(n)
    f_minus = np.zeros(n)
    for i in range(n):
        vp = best_vec.copy(); vp[i] = min(best_vec[i] + steps[i], hi[i])
        vm = best_vec.copy(); vm[i] = max(best_vec[i] - steps[i], lo[i])
        f_plus[i]  = objective_fn(vp)
        f_minus[i] = objective_fn(vm)

    # Normalized first-order sensitivity (central difference, normalized by param range)
    sens_indiv = (f_plus - f_minus) / (2.0 * delta_frac)

    # Second-order matrix: diagonal = curvature, off-diagonal = mixed partials
    sens_matrix = np.zeros((n, n))
    for i in range(n):
        denom_ii = (steps[i] / ranges[i]) ** 2 if ranges[i] > 0 else 1.0
        sens_matrix[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / max(denom_ii, 1e-30)

    for i in range(n):
        for j in range(i + 1, n):
            vpp = best_vec.copy()
            vpp[i] = min(best_vec[i] + steps[i], hi[i])
            vpp[j] = min(best_vec[j] + steps[j], hi[j])
            f_pp = objective_fn(vpp)
            denom_ij = (steps[i] / ranges[i]) * (steps[j] / ranges[j]) if (ranges[i] > 0 and ranges[j] > 0) else 1.0
            cross = (f_pp - f_plus[i] - f_plus[j] + f0) / max(denom_ij, 1e-30)
            sens_matrix[i, j] = cross
            sens_matrix[j, i] = cross

    return sens_indiv, sens_matrix


def write_calibration_log(log_path, param_names, all_eval_records,
                           best_params, sens_indiv, sens_matrix,
                           multiplier, best_obj, early_termination=False): # Added early_termination flag
    """
    Write a structured calibration log:
      - Best parameters and objective value
      - Individual parameter sensitivities (ranked)
      - Parameter interaction matrix
      - Full evaluation history (all runs in this session)
    The log is machine-readable so future runs can load and reuse results.
    """
    n = len(param_names)
    col_w = 14

    with open(log_path, 'w') as f:
        f.write('=' * 72 + '\n')
        f.write('=== PARC MODEL Vol_Elev CALIBRATION LOG ===\n')
        ts_label = os.path.basename(log_path).replace('Calibration_log_', '').replace('.txt', '')
        f.write(f'Timestamp    : {ts_label}\n')
        f.write(f'Transport    : {TRANSPORT_MODE}\n')
        f.write(f'Parameters   : {", ".join(param_names)}\n')
        if early_termination:
            f.write('*** NOTE: Optimization terminated early by user. Results are partial. ***\n')
        f.write('=' * 72 + '\n\n')

        f.write('=== BEST PARAMETERS (from completed evaluations) ===\n')
        for k, v in best_params.items():
            f.write(f'  {k:>30s} = {v:.10f}\n')
        f.write(f'  {"Best combined objective":>30s} = {best_obj:.10f}\n')
        if not np.isnan(multiplier):
            f.write(f'  {"Volume multiplier":>30s} = {multiplier:.6f}\n')
        else:
            f.write(f'  {"Volume multiplier":>30s} = Not computed (requires full run)\n')
        f.write('\n')

        if sens_indiv is not None and len(sens_indiv) == n: # Check if sensitivity was computed
            f.write('=== PARAMETER SENSITIVITY (individual OAT, normalized) ===\n')
            f.write('  Interpretation: |Sensitivity| = rate of objective change per unit\n')
            f.write('  normalized parameter change.  Higher = stronger lever on model output.\n')
            f.write('  Sign: positive = increasing param worsens fit; negative = improves fit.\n\n')
            sorted_idx = np.argsort(np.abs(sens_indiv))[::-1]
            f.write(f'  {"Rank":<6} {"Parameter":<30} {"Sensitivity":>14}  Direction\n')
            f.write('  ' + '-' * 70 + '\n')
            for rank, i in enumerate(sorted_idx, 1):
                direction = 'increases obj' if sens_indiv[i] >= 0 else 'decreases obj'
                f.write(f'  {rank:<6} {param_names[i]:<30} {sens_indiv[i]:>14.6f}  (when increased: {direction})\n')
            f.write('\n')
        else:
            f.write('=== PARAMETER SENSITIVITY ===\n')
            f.write('  Sensitivity analysis not performed due to early termination or error.\n\n')


        if sens_matrix is not None and sens_matrix.shape == (n, n): # Check if sensitivity was computed
            f.write('=== PARAMETER INTERACTION MATRIX ===\n')
            f.write('  Matrix[i,j] = normalized cross-sensitivity d\u00b2(obj)/(dp_i_norm * dp_j_norm)\n')
            f.write('  Diagonal = parameter self-curvature.  Large off-diagonal = strong interaction.\n\n')
            header_line = ' ' * 32
            for pn in param_names:
                header_line += f'{pn[:col_w - 1]:>{col_w}}'
            f.write(header_line + '\n')
            f.write('  ' + '-' * (30 + n * col_w) + '\n')
            for i, pn_i in enumerate(param_names):
                row = f'  {pn_i:<30}'
                for j in range(n):
                    row += f'{sens_matrix[i, j]:>{col_w}.4f}'
                f.write(row + '\n')
            f.write('\n')
        else:
            f.write('=== PARAMETER INTERACTION MATRIX ===\n')
            f.write('  Interaction matrix not performed due to early termination or error.\n\n')

        f.write('=== EVALUATION HISTORY ===\n')
        if all_eval_records:
            metric_cols = ['vol_trend_rmse', 'elev_norm', 'combined']
            col_headers = ['eval'] + param_names + metric_cols
            sep = ' | '
            f.write(sep.join(f'{h:>{col_w}}' for h in col_headers) + '\n')
            f.write('-' * (col_w * len(col_headers) + 3 * (len(col_headers) - 1)) + '\n')
            for rec in all_eval_records:
                ev   = rec.get('eval', 0)
                pvec = rec.get('params', [])
                vt   = rec.get('vol_trend', float('nan'))
                en   = rec.get('elev_norm', float('nan'))
                cb   = rec.get('combined', float('nan'))
                vals = ([f'{ev:>{col_w}d}']
                        + [f'{p:>{col_w}.8f}' for p in pvec]
                        + [f'{vt:>{col_w}.6f}', f'{en:>{col_w}.6f}', f'{cb:>{col_w}.6f}'])
                f.write(sep.join(vals) + '\n')
        else:
            f.write('  No evaluations completed.\n')

        f.write('\n=== END OF LOG ===\n')

    print(f'  Calibration log written: {os.path.basename(log_path)}')


# ==========================================
# 3c. Vol_Elev CALIBRATION OPTIMIZATION
# ==========================================
def run_vol_elev_calibration():
    """
    Execute the PARC model Vol_Elev dual-objective calibration.

    The PARC model is calibrated against two independent observational datasets
    from the Rio Coca post-waterfall collapse:

      Objective 1 — Eroded Volume Trend (shape + magnitude):
        Simulated cumulative mobilised sediment volume is matched to observed
        volumes derived from repeat bathymetric surveys or satellite-based
        DEMs of difference (DoDs). Both the temporal shape (rate of incision)
        and absolute magnitude (total sediment mobilised) are matched by
        normalising both sim and obs by the same observed peak value. This
        ensures bank_widening_factor closes any absolute volume gap rather
        than relying on the post-hoc volume_multiplier.

      Objective 2 — Bed Elevation Profile (RMSE with x-shift):
        Simulated thalweg profiles are matched to observed longitudinal profiles
        at multiple survey dates. An optimal longitudinal x-shift is applied to
        minimise RMSE at each profile date, accounting for datum differences,
        thalweg digitisation offsets, or changes in the active channel alignment
        between surveys and model grid.

    Combined objective = vol_trend_RMSE + elev_RMSE_normalized
    Both components are dimensionless and equally weighted.

    Optimisation Method:
      Global search: Differential Evolution (scipy.optimize.differential_evolution)
        — robust to multi-modal objective surfaces and parameter interactions;
          population seeded from warm-start vectors drawn from prior calibration logs.
      Local polish: L-BFGS-B (via DE's 'polish' option) — refines the best
        DE solution within the feasible bounds.

    A priority penalty (PRIORITY_EPSILON) breaks ties between solutions with
    near-identical model skill by preferring physically simpler (lower-valued)
    parameter sets, implementing a Occam's Razor regularisation.

    Memory Efficiency:
      Only calibration-timestep zb snapshots are stored (not the full NT×NX
      history). Animation frames are suppressed. gc.collect() is called after
      every PARC model evaluation to prevent RAM accumulation across hundreds
      of evaluations in larger calibration runs.

    Outputs (written to Calibration_Output/ sub-folder):
      - Vol_Elev_Calibration_Result.png : volume trend + elevation profile plots
      - Vol_Elev_Sensitivity.png        : OAT individual and interaction matrix plots
      - Vol_Elev_Best_Params.csv        : calibrated PARC model parameter table
      - Vol_Elev_Eval_History.csv       : all evaluation records this session
    A Calibration_log_YYYYMMDDHHmm.txt is written to the working directory
    (updated with final multiplier value once the last simulation completes).
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    cal_out_dir = CALIBRATION_OUTPUT_DIR
    os.makedirs(cal_out_dir, exist_ok=True)

    print('\n=== LOADING Vol_Elev CALIBRATION DATA ===')
    vol_time_days, vol_values, elev_profiles = load_vol_elev_calibration_data(START_DATE)
    elev_cal_profiles = [p for p in elev_profiles if p['time_days'] > 0]
    print(f'  Volume obs points      : {len(vol_values)}')
    print(f'  Elevation cal profiles : {len(elev_cal_profiles)}')

    param_names = list(VOL_ELEV_PARAM_BOUNDS.keys())
    bounds      = [VOL_ELEV_PARAM_BOUNDS[p] for p in param_names]
    n_params    = len(param_names)

    # Precompute fixed mapping from observed profile → model timestep index.
    # This is deterministic (time axis is the same for every simulation).
    time_days_arr    = np.linspace(0, TOTAL_TIME_DAYS, NT_GLOBAL)
    profile_t_indices = [int(np.argmin(np.abs(time_days_arr - p['time_days'])))
                         for p in elev_cal_profiles]
    cal_t_indices = sorted(set(profile_t_indices))

    # Normalisation constants (computed once from observations)
    all_obs_z  = np.concatenate([p['z'] for p in elev_cal_profiles])
    elev_z_range = max(float(np.ptp(all_obs_z)), 1.0)
    obs_max      = max(float(np.max(np.abs(vol_values))), 1.0)
    obs_norm     = vol_values / obs_max

    # ------------------------------------------------------------------
    # Load prior calibration logs → warm-start seeds
    # ------------------------------------------------------------------
    print('\n=== CHECKING FOR PRIOR CALIBRATION LOGS ===')
    prior_evals = load_calibration_logs(SCRIPT_DIR)
    warm_seeds  = []
    if prior_evals:
        prior_sorted = sorted(prior_evals, key=lambda x: x[1])
        n_seeds = min(10, max(1, len(prior_sorted) // 5))
        lo_arr = np.array([b[0] for b in bounds])
        hi_arr = np.array([b[1] for b in bounds])
        for pvec, _ in prior_sorted[:n_seeds]:
            if len(pvec) == n_params:
                warm_seeds.append(np.clip(pvec, lo_arr, hi_arr))
        print(f'  Using {len(warm_seeds)} warm-start seeds '
              f'(best prior obj: {prior_sorted[0][1]:.4f})')
    else:
        print('  No prior logs found — starting cold.')

    # ------------------------------------------------------------------
    # Objective function + progress / checkpoint infrastructure
    # ------------------------------------------------------------------
    maxiter   = 50
    popsize   = 15
    pop_total = popsize * n_params
    # Estimate total evaluations: initial population + maxiter generations.
    # Actual count may differ (early convergence, polishing), but gives a
    # reliable denominator for % progress reporting.
    total_expected_evals = (maxiter + 1) * pop_total

    all_eval_records  = []
    eval_count        = [0]
    best_so_far       = [None, float('inf')]   # [param_vector, obj_value]
    last_ckpt_pct     = [-1]                   # last 10% band that triggered a checkpoint

    ts            = datetime.now().strftime('%Y%m%d%H%M')
    ckpt_log_path = os.path.join(SCRIPT_DIR, f'Calibration_log_{ts}.txt')

    # -- Priority penalty pre-computation (done once, shared by every objective call) --
    # For each optimized parameter, derive a weight = 1 / priority_number.
    # Parameters not listed in PARAM_PRIORITY (or with priority <= 0) get weight 0.
    _prio_lo      = np.array([b[0] for b in bounds])
    _prio_hi      = np.array([b[1] for b in bounds])
    _prio_ranges  = np.maximum(_prio_hi - _prio_lo, 1e-30)
    _prio_weights = np.array([
        1.0 / PARAM_PRIORITY[p]
        if (p in PARAM_PRIORITY and PARAM_PRIORITY[p] > 0) else 0.0
        for p in param_names
    ])
    _prio_weight_sum = float(np.sum(_prio_weights)) if np.sum(_prio_weights) > 0 else 1.0

    if PRIORITY_EPSILON > 0 and _prio_weight_sum > 0:
        print(f'  Priority penalty: PRIORITY_EPSILON={PRIORITY_EPSILON}  '
              f'Active params: {[p for p in param_names if p in PARAM_PRIORITY and PARAM_PRIORITY.get(p,0)>0]}')

    def _flush_checkpoint(label='checkpoint'):
        """Write current best params and all evaluations accumulated so far."""
        if not all_eval_records and best_so_far[0] is None:
            return
        bp = (dict(zip(param_names, best_so_far[0]))
              if best_so_far[0] is not None else {})
        write_calibration_log(
            ckpt_log_path, param_names, all_eval_records,
            bp,
            np.zeros(n_params),          # sensitivities computed only at end
            np.zeros((n_params, n_params)),
            multiplier=float('nan'),
            best_obj=best_so_far[1],
        )
        print(f'  [{label}] Checkpoint log written '
              f'({len(all_eval_records)} evals, '
              f'best obj={best_so_far[1]:.4f})')

    def objective(param_vector):
        params = dict(zip(param_names, param_vector))
        eval_count[0] += 1
        ev_num = eval_count[0]

        try:
            result = run_simulation(
                params,
                run_id=f'opt_{ev_num}',
                silent=True,
                return_details=True,
                cal_t_indices=cal_t_indices,
            )
        except Exception as exc:
            print(f'  [eval {ev_num:>4d}] FAILED: {exc}')
            all_eval_records.append({
                'eval': ev_num, 'params': param_vector.tolist(),
                'vol_trend': float('nan'), 'elev_norm': float('nan'), 'combined': 1e12,
            })
            return 1e12

        cum_eroded = result['cumulative_eroded_vol']
        zb_snaps   = result['zb_snapshots']
        t_arr      = result['time_days_arr']

        # -- Volume RMSE — Absolute Magnitude + Temporal Shape --
        # Both simulated and observed volumes are normalised by the SAME observed
        # peak (obs_max), so the PARC model optimizer sees both the temporal
        # shape (rate of sediment release) AND the absolute magnitude (total
        # volume mobilised). Normalising by the observed rather than the simulated
        # peak is critical: it prevents the optimizer from compensating for a
        # systematically under-powered transport coefficient by simply rescaling
        # the profile shape. This drives bank_widening_factor to close any absolute
        # volume gap, making the post-hoc volume_multiplier converge to ~1.0 after
        # a successful calibration.
        sim_vol_at_obs = np.interp(vol_time_days, t_arr, cum_eroded)
        sim_norm = sim_vol_at_obs / obs_max
        vol_trend_rmse = float(np.sqrt(np.mean((obs_norm - sim_norm) ** 2)))

        # -- Elevation Profile RMSE with Optimal x-Shift --
        # For each survey date, the PARC model profile is compared against the
        # observed thalweg by finding the longitudinal x-shift that minimises
        # RMSE between the interpolated simulated elevation and the observed points.
        # This accounts for datum offsets and digitisation errors between field
        # surveys (GPS-referenced) and the PARC model coordinate system derived
        # from the DEM/shapefile thalweg. The RMSE at the optimal shift is
        # normalised by the observed elevation range (elev_z_range) to make it
        # dimensionless and comparable to the volume RMSE component.
        elev_errors = []
        for i_p, prof in enumerate(elev_cal_profiles):
            t_idx = profile_t_indices[i_p]
            z_sim_full = zb_snaps.get(t_idx)
            if z_sim_full is None:
                continue
            x_obs = prof['x']
            z_obs = prof['z']

            # Closure captures z_sim_full, x_obs, z_obs by reference safely here.
            def _shift_rmse(xs, _z=z_sim_full, _xo=x_obs, _zo=z_obs):
                zi = np.interp(_xo + xs, X_GLOBAL, _z, left=np.nan, right=np.nan)
                valid = ~np.isnan(zi)
                return (float(np.sqrt(np.mean((_zo[valid] - zi[valid]) ** 2)))
                        if np.sum(valid) >= 3 else 1e6)

            rs = minimize_scalar(
                _shift_rmse,
                bounds=(-X_GLOBAL[-1], X_GLOBAL[-1]),
                method='bounded',
                options={'xatol': DX_GLOBAL},
            )
            elev_errors.append(rs.fun)

        mean_elev_rmse = float(np.mean(elev_errors)) if elev_errors else 0.0
        elev_err_norm  = mean_elev_rmse / elev_z_range
        combined       = vol_trend_rmse + elev_err_norm

        # Priority penalty: nudge the PARC model optimizer toward physically simpler
        # (lower-valued) parameter sets when model-fit differences are negligible.
        # Each parameter's normalised position within its bounds [0, 1] is weighted
        # by 1/priority_number (lower number = higher weight = stronger pull toward
        # the minimum of that parameter's range). The total penalty is scaled by
        # PRIORITY_EPSILON so it only breaks ties, never overrides real fit differences.
        # This implements a parsimony regularisation: among equally good solutions,
        # prefer the one closest to the simplest physical interpretation.
        norm_vals       = np.clip((param_vector - _prio_lo) / _prio_ranges, 0.0, 1.0)
        priority_penalty = (PRIORITY_EPSILON
                            * float(np.dot(_prio_weights, norm_vals))
                            / _prio_weight_sum)
        combined_penalized = combined + priority_penalty

        # Log the raw model-fit metric (without penalty) for interpretability.
        all_eval_records.append({
            'eval': ev_num, 'params': param_vector.tolist(),
            'vol_trend': vol_trend_rmse, 'elev_norm': elev_err_norm, 'combined': combined,
        })

        # Track global best using the penalized value (consistent with what optimizer sees).
        if combined_penalized < best_so_far[1]:
            best_so_far[0] = param_vector.copy()
            best_so_far[1] = combined_penalized

        # Per-evaluation terminal log
        if ev_num % 10 == 0:
            print(f'  [eval {ev_num:>4d}]  vol_trend={vol_trend_rmse:.4f}  '
                  f'elev_norm={elev_err_norm:.4f}  combined={combined:.4f}  '
                  f'penalty={priority_penalty:.5f}')

        # Progress % and checkpoint at every 10% band
        pct = min(int(100 * ev_num / total_expected_evals), 99)
        current_band = pct // 10
        if current_band > last_ckpt_pct[0]:
            last_ckpt_pct[0] = current_band
            bar = '#' * current_band + '-' * (10 - current_band)
            print(f'\n  OPTIMIZATION PROGRESS: {pct:3d}% [{bar}] '
                  f'{ev_num}/{total_expected_evals} evals'
                  f' | best obj: {best_so_far[1]:.4f}\n')
            _flush_checkpoint(label=f'{pct}%')

        # Release memory for this evaluation explicitly
        del result, cum_eroded, zb_snaps
        gc.collect()

        return combined_penalized

    # ------------------------------------------------------------------
    # Build initial population (warm seeds filled out with random draws)
    # ------------------------------------------------------------------
    init_pop  = None
    if warm_seeds:
        rng = np.random.default_rng(42)
        lo_arr = np.array([b[0] for b in bounds])
        hi_arr = np.array([b[1] for b in bounds])
        n_random = max(0, pop_total - len(warm_seeds))
        rand_pop = lo_arr + rng.random((n_random, n_params)) * (hi_arr - lo_arr)
        stacked  = np.vstack(warm_seeds + [rand_pop]) if n_random > 0 else np.vstack(warm_seeds)
        init_pop = stacked[:pop_total]

    # ------------------------------------------------------------------
    # Differential Evolution  (with crash / KeyboardInterrupt safety)
    # ------------------------------------------------------------------
    print('\n=== STARTING Vol_Elev OPTIMIZATION ===')
    print(f'  Parameters : {param_names}')
    print(f'  Population : {pop_total} members  |  Max iterations: {maxiter}')
    print(f'  Checkpoint log: {os.path.basename(ckpt_log_path)}')
    print(f'  (Interim logs written every ~10% of estimated {total_expected_evals} evals)\n')

    de_kwargs = dict(maxiter=maxiter, popsize=popsize, tol=1e-4,
                     seed=42, disp=True, polish=True)
    if init_pop is not None:
        de_kwargs['init'] = init_pop

    de_result   = None
    interrupted = False
    try:
        de_result = differential_evolution(objective, bounds, **de_kwargs)
    except KeyboardInterrupt:
        interrupted = True
        print('\n  KeyboardInterrupt received — saving current best and exiting cleanly.')
    except Exception as de_exc:
        interrupted = True
        print(f'\n  Optimization crashed: {de_exc} — saving current best.')
    finally:
        # Always flush whatever we have so the run is not lost
        _flush_checkpoint(label='final/interrupted' if interrupted else 'final')

    # Determine best parameters (from DE result or from tracked best)
    if de_result is not None:
        best_params = dict(zip(param_names, de_result.x))
        best_obj    = de_result.fun
    elif best_so_far[0] is not None:
        best_params = dict(zip(param_names, best_so_far[0]))
        best_obj    = best_so_far[1]
    else:
        print('  No evaluations completed — nothing to save.')
        return {}, float('nan')

    print('\n=== OPTIMIZATION COMPLETE ===')
    print(f'  Best objective : {best_obj:.6f}')
    print(f'  Evaluations    : {eval_count[0]}')
    if interrupted:
        print('  (Run was interrupted — results reflect best found so far.)')
    print('  Best parameters:')
    for k, v in best_params.items():
        print(f'    {k:>30s} = {v:.6f}')

    # ------------------------------------------------------------------
    # Sensitivity analysis around best point
    # ------------------------------------------------------------------
    print('\n--- Computing parameter sensitivity (OAT finite differences) ---')
    try:
        sens_indiv, sens_matrix = compute_sensitivity(
            best_params, param_names, bounds, objective, delta_frac=0.05
        )
    except Exception as e:
        print(f'  Sensitivity computation failed: {e}. Sensitivity data will not be logged.')
        sens_indiv  = np.zeros(n_params)
        sens_matrix = np.zeros((n_params, n_params))

    # ------------------------------------------------------------------
    # Write full calibration log with sensitivities (multiplier added after final run)
    # ------------------------------------------------------------------
    write_calibration_log(ckpt_log_path, param_names, all_eval_records,
                          best_params, sens_indiv, sens_matrix,
                          multiplier=float('nan'), best_obj=best_obj)

    # ------------------------------------------------------------------
    # Final diagnostic simulation with best parameters
    # ------------------------------------------------------------------
    print('\n--- Running final diagnostic simulation ---')
    final    = run_simulation(best_params, run_id='Vol_Elev_Best',
                              silent=True, return_details=True,
                              cal_t_indices=cal_t_indices)
    cum_eroded = final['cumulative_eroded_vol']
    t_arr      = final['time_days_arr']
    zb_snaps   = final['zb_snapshots']

    sim_vol_at_obs = np.interp(vol_time_days, t_arr, cum_eroded)
    mask = sim_vol_at_obs > 0
    multiplier = (float(np.nanmedian(vol_values[mask] / sim_vol_at_obs[mask]))
                  if np.any(mask) else float('nan'))
    print(f'\n  Volume Multiplier (obs / sim ratio): {multiplier:.4f}')

    # Re-write log with the correct multiplier and sensitivities
    write_calibration_log(ckpt_log_path, param_names, all_eval_records,
                          best_params, sens_indiv, sens_matrix,
                          multiplier=multiplier, best_obj=best_obj)

    # Per-profile elevation RMSE and optimal shifts
    print('\n  Elevation profile matching (post-optimization):')
    profile_best_shifts = []
    for i_p, prof in enumerate(elev_cal_profiles):
        t_idx      = profile_t_indices[i_p]
        z_sim_full = zb_snaps.get(t_idx)
        if z_sim_full is None:
            profile_best_shifts.append(0.0)
            continue

        def _rmse_final(xs, _z=z_sim_full, _xo=prof['x'], _zo=prof['z']):
            zi = np.interp(_xo + xs, X_GLOBAL, _z, left=np.nan, right=np.nan)
            v  = ~np.isnan(zi)
            return float(np.sqrt(np.mean((_zo[v] - zi[v]) ** 2))) if np.sum(v) >= 3 else 1e6

        rs = minimize_scalar(_rmse_final, bounds=(-X_GLOBAL[-1], X_GLOBAL[-1]),
                             method='bounded', options={'xatol': DX_GLOBAL})
        profile_best_shifts.append(rs.x)
        print(f'    {prof["date"].strftime("%Y-%m-%d")}: RMSE={rs.fun:.2f} m, '
              f'x_shift={rs.x:.1f} m')

    # ------------------------------------------------------------------
    # Diagnostic plots → Calibration_Output/
    # ------------------------------------------------------------------
    # --- Plot 1: Volume trend + elevation profiles ---
    fig, (ax_v, ax_e) = plt.subplots(2, 1, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.35)

    ax_v.plot(t_arr, cum_eroded, 'b-', linewidth=1.5, label='Simulated eroded vol')
    ax_v.scatter(vol_time_days, vol_values, color='red', marker='x', s=60,
                 zorder=5, label='Observed eroded vol')
    if not np.isnan(multiplier):
        ax_v.plot(t_arr, cum_eroded * multiplier, 'b--', alpha=0.5,
                  label=f'Simulated \u00d7{multiplier:.2f} (absolute match)')
    ax_v.set_xlabel('Time (days)')
    ax_v.set_ylabel('Eroded Volume (m\u00b3)')
    ax_v.set_title('Vol_Elev Calibration \u2014 Eroded Volume Trend')
    ax_v.legend()
    ax_v.grid(True, alpha=0.3)

    colors = plt.cm.viridis(np.linspace(0, 1, len(elev_cal_profiles)))
    for i_p, prof in enumerate(elev_cal_profiles):
        z_sim_full = zb_snaps.get(profile_t_indices[i_p], np.full(NX_GLOBAL, np.nan))
        best_shift = profile_best_shifts[i_p]
        lbl = prof['date'].strftime('%Y-%m-%d')
        ax_e.plot(prof['x'] + best_shift, prof['z'], 'o', color=colors[i_p],
                  markersize=3, label=f'Obs {lbl} (shift {best_shift:.0f}m)')
        ax_e.plot(X_GLOBAL, z_sim_full, '-', color=colors[i_p], alpha=0.6,
                  label=f'Sim {lbl}')
    ax_e.set_xlabel('Distance downstream (m)')
    ax_e.set_ylabel('Bed Elevation (m)')
    ax_e.set_title('Vol_Elev Calibration \u2014 Bed Elevation Profiles')
    ax_e.legend(fontsize=7, ncol=2)
    ax_e.grid(True, alpha=0.3)

    fig.savefig(os.path.join(cal_out_dir, 'Vol_Elev_Calibration_Result.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)

    # --- Plot 2: Sensitivity ---
    fig2, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(14, 5))
    sorted_idx = np.argsort(np.abs(sens_indiv))[::-1]
    ax_s1.barh(
        [param_names[i] for i in sorted_idx],
        [abs(float(sens_indiv[i])) for i in sorted_idx],
        color=['tomato' if sens_indiv[i] >= 0 else 'steelblue' for i in sorted_idx],
    )
    ax_s1.set_xlabel('|Sensitivity| (normalised)')
    ax_s1.set_title('Individual Parameter Sensitivity (OAT)')
    ax_s1.grid(True, alpha=0.3, axis='x')

    im = ax_s2.imshow(sens_matrix, cmap='RdBu_r', aspect='auto')
    ax_s2.set_xticks(range(n_params))
    ax_s2.set_xticklabels(param_names, rotation=45, ha='right', fontsize=8)
    ax_s2.set_yticks(range(n_params))
    ax_s2.set_yticklabels(param_names, fontsize=8)
    ax_s2.set_title('Parameter Interaction Matrix')
    plt.colorbar(im, ax=ax_s2)
    fig2.tight_layout()
    fig2.savefig(os.path.join(cal_out_dir, 'Vol_Elev_Sensitivity.png'),
                 bbox_inches='tight', dpi=150)
    plt.close(fig2)

    # --- Summary table: best parameters ---
    param_df = pd.DataFrame([best_params])
    param_df['objective']         = best_obj
    param_df['volume_multiplier'] = multiplier
    param_df.to_csv(os.path.join(cal_out_dir, 'Vol_Elev_Best_Params.csv'), index=False)

    # --- Evaluation history CSV ---
    if all_eval_records:
        eval_df    = pd.DataFrame(all_eval_records)
        param_cols = pd.DataFrame(eval_df['params'].tolist(), columns=param_names)
        eval_df    = pd.concat([eval_df.drop(columns=['params']), param_cols], axis=1)
        eval_df.to_csv(os.path.join(cal_out_dir, 'Vol_Elev_Eval_History.csv'), index=False)

    del final, cum_eroded, zb_snaps
    gc.collect()

    print(f'\n=== ALL OUTPUTS WRITTEN ===')
    print(f'  Calibration log  : {ckpt_log_path}')
    print(f'  Diagnostic plots : {cal_out_dir}')

    return best_params, multiplier


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
# Entry point for the PARC (Profile And Reach Change) model.
# Three execution modes are available (controlled by CALIBRATION_MODE):
#
#   CALIBRATION_MODE = 'Vol_Elev'
#     Runs the Differential Evolution global optimisation to calibrate PARC
#     model parameters against observed sediment volumes and bed elevation
#     profiles. Outputs calibrated parameter sets, diagnostic plots, and a
#     detailed calibration log to the Calibration_Output/ directory.
#
#   CALIBRATION_MODE = True
#     Exhaustive grid sweep across all combinations defined in CALIBRATION_PARAMS.
#     Useful for rapid exploration of parameter space before Vol_Elev optimisation.
#
#   CALIBRATION_MODE = False (default)
#     Single forward run using DEFAULT_PARAMS. Generates the full suite of
#     PARC model output figures (longitudinal profile animations, sediment budget
#     plots, Vol_Elev comparison, per-profile comparisons).
if __name__ == "__main__":
    if CALIBRATION_MODE == 'Vol_Elev':
        # --- Vol_Elev Optimization ---
        run_vol_elev_calibration()

    elif CALIBRATION_MODE:
        # --- Calibration Setup ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use OUTPUT_BASE_DIR directly
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

        print(f"--- STARTING CALIBRATION MODE ---")
        print(f"Output Directory: {OUTPUT_BASE_DIR}")

        # Generate Combinations
        keys, values = zip(*CALIBRATION_PARAMS.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print(f"Total Combinations to Run: {len(param_combinations)}")

        results_list = []

        for i, params in enumerate(param_combinations):
            run_num = i + 1 # Start from 1
            print(f"\n--- Running Combination {run_num}/{len(param_combinations)} ---")

            # Run Model
            res = run_simulation(params, run_id=run_num, output_dir=OUTPUT_BASE_DIR)
            results_list.append(res)

        # Save Aggregated Results to Excel
        df_results = pd.DataFrame(results_list)
        excel_filename = f"Summary_Values_{timestamp}.xlsx"
        excel_path = os.path.join(OUTPUT_BASE_DIR, excel_filename)
        df_results.to_excel(excel_path, index=False)
        print(f"\nCalibration Complete. Summary saved to {excel_path}")
        if 'RMSE' in df_results.columns:
            print("Top 5 Runs by RMSE:")
            print(df_results.sort_values('RMSE').head(5))

    else:
        # --- Single Run Mode ---
        # Executes one forward PARC model simulation using the calibrated
        # DEFAULT_PARAMS and generates all output visualisations.
        print("--- STARTING PARC MODEL — SINGLE RUN MODE ---")
        run_simulation(DEFAULT_PARAMS, run_id="Manual_Run")