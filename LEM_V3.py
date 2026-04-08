# ============================================================================
# LANDSCAPE EVOLUTION MODEL (LEM) — Version 3
# ============================================================================
#
# PURPOSE
# -------
# This script implements a physics-based Landscape Evolution Model (LEM) using
# the CSDMS Landlab framework.  It was developed to predict the new equilibrium
# topography of the Rio Coca watershed, Ecuador, following the 2020 collapse of
# the San Rafael waterfall — an event that dramatically changed the local base
# level and is expected to drive rapid upstream knickpoint migration and valley
# adjustment over the coming centuries to millennia.
#
# GENERALISATION
# --------------
# Although motivated by a specific field case, every parameter, switch, and
# file path is exposed through the Config class so the model can be configured
# for any watershed or research context without modifying the core logic.
#
# PHYSICAL FRAMEWORK
# ------------------
# The governing equation solved at every node and timestep is a discretised
# form of the sediment-flux / detachment-limited stream-power landscape
# evolution equation (Whipple & Tucker 1999; Shobe et al. 2017 for SPACE):
#
#   dz/dt = U - E_bedrock - E_sediment + D_hillslope
#
# where:
#   z           = surface elevation (m)
#   U           = rock-uplift rate relative to base level (m/yr)
#   E_bedrock   = fluvial bedrock incision rate (m/yr) — driven by stream power
#   E_sediment  = sediment entrainment / routing (m/yr) — handled by SPACE
#   D_hillslope = hillslope diffusive flux divergence (m/yr)
#
# COMPONENT DEPENDENCIES
# ----------------------
# All geomorphic components are drawn from Landlab (Hobley et al. 2017):
#   • Flow routing   — PriorityFloodFlowRouter (Lindsay 2016) or FlowAccumulator
#   • Hillslopes     — TaylorNonLinearDiffuser or LinearDiffuser (Culling 1963)
#   • Fluvial        — SPACE (Shobe+2017) or StreamPowerEroder (Whipple&Tucker 1999)
#   • Mass movement  — BedrockLandslider (Campforts+2020)
#
# KEY REFERENCES
# --------------
#   Culling (1963)   — Soil creep / linear diffusion
#   Whipple & Tucker (1999) — Stream power incision model
#   Hobley et al. (2017)    — Landlab framework description
#   Shobe et al. (2017)     — SPACE model (sediment flux + bedrock incision)
#   Campforts et al. (2020) — BedrockLandslider component
#   Lindsay (2016)          — Priority-Flood flow routing
#
# USAGE
# -----
#   python LEM_V3.py
# All configuration is done in the Config class below.  No command-line
# arguments are required, though the class attributes can be overridden at
# the bottom __main__ block for quick parameter sweeps.
#
# REORGANISATION FROM COLAB NOTEBOOK
# -----------------------------------
# Originally prototyped as a Colab notebook; restructured here into a fully
# standalone, modular Python script that retains all optional features.
# ============================================================================

import os
import sys
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
import matplotlib.cm as cm
import shapefile as shp  # pyshp — pure-Python shapefile reader

# Suppress harmless UserWarnings from Landlab internals.
# Landlab uses numpy.where internally in graph-sort routines and sometimes
# triggers a UserWarning about missing 'out' keyword — this is safe to ignore.
warnings.filterwarnings('ignore', category=UserWarning, module='landlab.graph.sort.sort')

# --- Landlab core ---
from landlab import RasterModelGrid
from landlab.plot import imshow_grid
from landlab.io.netcdf import read_netcdf, write_netcdf

# --- Landlab geomorphic components ---
from landlab.components import (
    # Flow routing — two options controlled by USE_LANDSLIDES switch:
    #   FlowAccumulator       — standard D8/MFD router, no pit filling
    #   PriorityFloodFlowRouter — pit-filling router (required by BedrockLandslider)
    FlowAccumulator,
    FlowDirectorMFD,            # Multi-flow-direction director (can pass to FlowAccumulator)
    PriorityFloodFlowRouter,    # Recommended when landslides are active

    # Hillslope diffusion — controlled by HILLSLOPE_MODEL switch:
    #   LinearDiffuser         — classic soil-creep diffusion (Culling 1963)
    #                            valid for gentle slopes; dq/dx = K * S
    #   TaylorNonLinearDiffuser — non-linear creep that diverges at critical slope Sc
    #                            (Roering et al. 1999); more realistic for steep terrain
    LinearDiffuser,
    TaylorNonLinearDiffuser,

    # Fluvial erosion — controlled by FLUVIAL_MODEL switch:
    #   StreamPowerEroder      — detachment-limited incision only
    #                            dz/dt = K_br * A^m * S^n
    #   Space                  — coupled sediment-flux + bedrock incision (Shobe+2017)
    #                            tracks both soil__depth and bedrock__elevation separately
    Space,
    StreamPowerEroder,

    # Pit filling — used once during DEM pre-processing
    #   SinkFillerBarnes       — Priority-Flood variant; fastest option for large grids
    #SinkFiller,              # Older, slower algorithm (commented out)
    SinkFillerBarnes,

    # Mass movement — controlled by USE_LANDSLIDES switch
    #   BedrockLandslider      — infinite-slope stability analysis with stochastic
    #                            triggering; moves both soil and bedrock
    BedrockLandslider
)

# Optional IPython imports — used only when running inside a Jupyter notebook
# to enable live-updating figures.  Falls back gracefully in plain Python.
try:
    from IPython.display import display, clear_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    import threading


# ============================================================================
# CONFIGURATION (DASHBOARD)
# ============================================================================
# All user-facing parameters live here.  Sub-classes group logically related
# settings.  To run a different scenario, edit only this class (or override
# individual attributes in the __main__ block at the bottom of the file).
# ============================================================================
class Config:
    # -----------------------------------------------------------------------
    # STRUCTURAL & FUNCTIONAL SWITCHES
    # -----------------------------------------------------------------------
    # LANDSCAPE_MODE
    #   "observed"  — loads a real DEM (GeoTIFF) and fills hydrological sinks.
    #                 Requires a valid DEM_FILE path.
    #   "synthetic" — constructs a simple flat/tilted grid for idealised tests.
    LANDSCAPE_MODE  = "observed"

    # RAINFALL_MODEL
    #   "constant"   — uniform precipitation applied each timestep (DT_BASE).
    #                  Use for long (>10 kyr) steady-state runs.
    #   "stochastic" — random storm / interstorm sequences drawn from exponential
    #                  distributions.  Useful for event-scale sediment dynamics.
    RAINFALL_MODEL  = "constant"

    # FLUVIAL_MODEL
    #   "space"        — SPACE model (Shobe et al. 2017).  Tracks both sediment
    #                    cover (soil__depth) and bedrock incision simultaneously.
    #                    ASSUMPTION: erosion is transport-supply-limited;
    #                    sediment is both entrained and deposited within the model.
    #                    Requires soil__depth and bedrock__elevation fields.
    #   "stream_power" — detachment-limited stream-power incision only.  Simpler
    #                    and faster; appropriate when sediment thickness << bedrock
    #                    incision depth (e.g. bedrock gorges).
    FLUVIAL_MODEL   = "space"

    # HILLSLOPE_MODEL
    #   "taylor"  — TaylorNonLinearDiffuser (Roering et al. 1999).  Flux
    #               diverges as slope → Sc (critical hillslope angle); produces
    #               realistic threshold hillslopes.  Recommended for steep terrain.
    #   "linear"  — Classic linear diffusion (Culling 1963).  Numerically cheaper;
    #               valid only when slopes are well below Sc.
    HILLSLOPE_MODEL = "taylor"

    # LITHOLOGY_MODE — controls how spatially varying bedrock erodibility is set.
    #   "uniform"  — single K_BR value applied everywhere (FLUVIAL_K_BR).
    #   "gradient" — linear diagonal gradient between FLUVIAL_K_BR/2 and K_BR*2.
    #               Useful for simple two-unit lithology proxies.
    #   "mapped"   — reads node-by-node erodibility values from a GeoTIFF or
    #               NetCDF raster aligned to the model grid.  Path supplied by
    #               BEDROCK_ERODIBILITY_FILE.  Overrides FLUVIAL_K_BR.
    LITHOLOGY_MODE  = "uniform"

    # SEDIMENT_DEPTH_MODE — controls initial sediment (soil) thickness.
    #   "uniform"  — constant thickness (SOIL_DEPTH_INIT) at every node.
    #   "mapped"   — reads spatially variable thickness from SOIL_DEPTH_FILE.
    #               Useful when field-mapped or modelled regolith depths are available.
    SEDIMENT_DEPTH_MODE = "mapped"

    # SEDIMENT_ERODIBILITY_MODE — controls spatial variation of K_SED in SPACE.
    #   "uniform"  — constant value (FLUVIAL_K_SED) at every node.
    #   "mapped"   — reads node-by-node values from SEDIMENT_ERODIBILITY_FILE.
    #               Allows alluvial fans, lake beds, etc. to have distinct erodibility.
    SEDIMENT_ERODIBILITY_MODE = "uniform"

    # USE_LANDSLIDES — activates BedrockLandslider (mass movement).
    #   True  — replaces FlowAccumulator with PriorityFloodFlowRouter (required
    #           dependency); applies infinite-slope stability analysis every
    #           DT_LANDSLIDE_TRIGGER years.  Significantly increases runtime.
    #   False — no mass-movement; standard FlowAccumulator is used instead.
    USE_LANDSLIDES  = True

    # USE_DYNAMIC_UPLIFT — allow uplift rate to vary through time.
    #   True  — calls user-defined uplift_rate_at_time(t) function; see below.
    #   False — constant spatial gradient with UPLIFT_MIN_MULT and UPLIFT_MAX_MULT
    #            applied to BASE_UPLIFT_RATE (or UPLIFT_TIMESERIES).
    USE_DYNAMIC_UPLIFT = False

    # USE_OUTLET_LOWERING — simulate base-level fall at the outlet node (e.g.
    #   waterfall collapse forcing a knickpoint).  If True, the outlet node
    #   elevation is lowered by OUTLET_LOWERING_RATE (m/yr) each timestep.
    #   Specifically relevant to the Rio Coca post-waterfall scenario.
    USE_OUTLET_LOWERING = False
    OUTLET_LOWERING_RATE = 0.0      # (m/yr); set > 0 when USE_OUTLET_LOWERING=True
    OUTLET_LOWERING_DURATION = 0.0  # (yr) ; lowering stops after this many years

    # OUTLET_NODE_ROWCOL — explicitly specify the watershed outlet pour point.
    #   Set to a (row, col) tuple to force that node to FIXED_VALUE (open)
    #   regardless of other boundary conditions.  Use this when the DEM is
    #   clipped to the watershed and all grid-edge cells are NoData, which
    #   would otherwise leave the model with zero open boundaries.
    #   Set to None to auto-detect: the lowest-elevation active node that
    #   borders a NoData cell is chosen automatically.
    #   Example: OUTLET_NODE_ROWCOL = (412, 0)  # row 412, leftmost column
    OUTLET_NODE_ROWCOL = None

    # USE_CHANNEL_FOOTPRINT — pre-condition the terrain along a manually-
    #   digitised channel path before the simulation starts.
    #   When True and CHANNEL_FOOTPRINT_FILE points to a valid shapefile:
    #     1. Elevations along the channel are forced to be monotonically
    #        non-increasing downstream (eliminates sinks along the channel).
    #     2. The channel is incised at least CHANNEL_INCISION_DEPTH metres
    #        below its neighbouring cells (ensures flow enters the channel).
    #     3. The downstream-most valid channel node becomes the watershed
    #        outlet (overrides OUTLET_NODE_ROWCOL).
    #     4. An approval map is displayed before the simulation begins.
    #   Set to False to skip all channel conditioning.
    USE_CHANNEL_FOOTPRINT = True

    # CHANNEL_FOOTPRINT_FILE — path to a polyline or polygon shapefile whose
    #   geometry traces the main channel.  Coordinates must be in the same
    #   projected CRS as the DEM (UTM metres).  Set to None to disable.
    CHANNEL_FOOTPRINT_FILE = r'D:\Rio Coca LEM\Channel_Footprint.shp'

    # CHANNEL_INCISION_DEPTH — minimum difference (metres) between channel-
    #   node elevation and the lowest non-channel neighbour.  A value of 0.1
    #   ensures the channel is always lower than adjacent hillslope cells.
    CHANNEL_INCISION_DEPTH = 0.1

    # -----------------------------------------------------------------------
    # SIMULATION TIMING
    # -----------------------------------------------------------------------
    # SIM_LENGTH  — total duration of the model run (years)
    #   Rio Coca example: 10–100 kyr to reach new equilibrium.
    SIM_LENGTH      = 10000.0

    # DT_BASE     — timestep applied in "constant" rainfall mode (years).
    #   ASSUMPTION: all processes are quasi-static within one timestep.
    #   Stability constraint: dt < dx / max_erosion_rate   (CFL criterion).
    #   For SPACE with K_SED ~ 0.005 and typical A/S ratios, DT_BASE ≤ 50 yr
    #   is generally stable at 50 m grid resolution.
    #   When USE_ADAPTIVE_TIMESTEP=True, DT_BASE acts as the ceiling; the
    #   actual step is automatically reduced to respect CFL constraints.
    DT_BASE         = 20.0

    # USE_ADAPTIVE_TIMESTEP — automatically reduce the timestep each iteration
    #   so that the CFL condition is satisfied for both hillslope diffusion
    #   and fluvial incision.  Highly recommended for runs with steep terrain,
    #   high erodibilities, or any simulation where numerical instability has
    #   been observed.  DT_BASE remains the upper ceiling.
    #   True  — adaptive mode on (recommended).
    #   False — fixed timestep (legacy behaviour; requires manual DT_BASE tuning).
    USE_ADAPTIVE_TIMESTEP = True

    # CFL_SAFETY_FACTOR — fraction of the CFL-computed maximum stable timestep
    #   that is actually used.  A value of 0.8 imposes a 20 % safety margin,
    #   keeping the scheme well clear of the stability boundary.
    #   Typical range: 0.5 (very conservative) – 0.9 (aggressive).
    CFL_SAFETY_FACTOR = 0.8

    # DT_MIN — absolute minimum adaptive timestep (years).
    #   Prevents the timestep from collapsing to near-zero in isolated cells
    #   with extreme slope or drainage area.  Choose a value small enough to
    #   still be physically meaningful but large enough to keep runtime finite.
    DT_MIN          = 0.1

    # -----------------------------------------------------------------------
    # INPUT / OUTPUT PATHS
    # -----------------------------------------------------------------------
    SAVE_DIR        = './output'

    # Primary terrain input (GeoTIFF; any GDAL-readable raster format is accepted).
    # Must be a hydrologically conditioned or raw DEM — the model will fill sinks
    # on first run and cache the result as a NetCDF.
    DEM_FILE        = './RioCoca_50m_Elevation.tif'

    # Optional spatial input files for mapped-property modes.
    # Each file must be a single-band GeoTIFF (or NetCDF) with the SAME
    # coordinate reference system and spatial extent as DEM_FILE.
    # Set to None to disable the corresponding mapped mode.
    BEDROCK_ERODIBILITY_FILE    = None   # Used when LITHOLOGY_MODE = "mapped"
    SEDIMENT_DEPTH_FILE         = 'Sed_Thick_50m_smoo.tif'   # Used when SEDIMENT_DEPTH_MODE = "mapped"
    SEDIMENT_ERODIBILITY_FILE   = None   # Used when SEDIMENT_ERODIBILITY_MODE = "mapped"

    # -----------------------------------------------------------------------
    # HILLSLOPE PROCESS PARAMETERS
    # -----------------------------------------------------------------------
    # HILLSLOPE_K — hillslope diffusivity (m²/yr)
    #   ASSUMPTION: linear / non-linear creep is the dominant hillslope process.
    #   Typical range: 1e-4 (bedrock) – 1e-1 m²/yr (soil-mantled).
    #   Rio Coca value (dense vegetation, humid tropics): 0.01 m²/yr.
    HILLSLOPE_K     = 0.01

    # HILLSLOPE_SC — critical slope (m/m) for TaylorNonLinearDiffuser.
    #   Slopes approaching Sc trigger rapid sediment flux divergence.
    #   Typical range: 0.6 – 1.2 m/m.  Use None to keep Landlab default (1.0).
    HILLSLOPE_SC    = None

    # -----------------------------------------------------------------------
    # FLUVIAL PROCESS PARAMETERS (SPACE model)
    # -----------------------------------------------------------------------
    # K_SED  — sediment erodibility coefficient (1/yr if using Q-normalised drainage
    #           area; adjust units if using raw Q).  Controls how quickly alluvium
    #           is entrained.  Typical range: 1e-4 – 1e-2.
    FLUVIAL_K_SED   = 0.01

    # K_BR   — bedrock erodibility coefficient (m^(1-2m) yr^-1).  Controls how
    #           quickly rock is incised when not protected by sediment.
    #           ASSUMPTION: incision rate ∝ K_BR * A^m * S^n.
    #           Typical range: 1e-6 (hard granite) – 5e-3 (weak mudstone).
    FLUVIAL_K_BR    = 0.001

    # V_S    — settling velocity of representative grain size (m/yr, SPACE model).
    #           Controls partitioning between suspended load and bedload.
    #           Higher V_S → more deposition per unit distance.
    FLUVIAL_V_S     = 0.5

    # SPACE m and n exponents — stream power law exponents.
    #   m / n ratio controls drainage-area vs slope sensitivity.
    #   Empirical consensus: m=0.5, n=1.0 (concavity index theta = m/n = 0.5).
    SPACE_M         = 0.5
    SPACE_N         = 1.0

    # SPACE F_F — fraction of suspended sediment that bypasses the node.
    #   0.0 = all sediment deposits; 1.0 = all sediment exported instantly.
    SPACE_F_F       = 0.5

    # -----------------------------------------------------------------------
    # SEDIMENT / SOIL PARAMETERS
    # -----------------------------------------------------------------------
    # SOIL_DEPTH_INIT — initial regolith / alluvial cover depth (m).
    #   ASSUMPTION: uniform at model start when SEDIMENT_DEPTH_MODE = "uniform".
    #   This controls whether early fluvial erosion is sediment-limited or
    #   bedrock-incision-limited.
    SOIL_DEPTH_INIT = 2.0

    # WEATHERING_RATE — rate of bedrock-to-regolith conversion (m/yr).
    #   Set to 0.0 to disable weathering (bedrock exposed indefinitely).
    #   Humped-function weathering (depth-dependent) is not yet implemented
    #   but can be added in the post-surface-sync step.
    WEATHERING_RATE = 0.001

    # -----------------------------------------------------------------------
    # CLIMATE & TECTONICS SPATIAL GRADIENTS
    # -----------------------------------------------------------------------
    # Linear gradient fields are the simplest way to represent spatial
    # variability without a full raster input.  Each gradient is defined by
    # three numbers:
    #
    #   MIN value  — value at the "low" end of the gradient direction
    #   MAX value  — value at the "high" end of the gradient direction
    #   ANGLE_DEG  — compass bearing (degrees, clockwise from North / +Y axis)
    #                that points from MIN toward MAX.
    #
    # ANGLE_DEG REFERENCE CARD
    # -------------------------
    #   0°   / 360° — gradient increases toward the top of the grid (North / +Y)
    #   90°          — gradient increases toward the right edge (East  / +X)
    #   180°         — gradient increases toward the bottom of the grid (South)
    #   270°         — gradient increases toward the left edge  (West  / -X)
    #   45°          — NE diagonal (equal parts +X and +Y)
    #   135°         — SE diagonal (+X, -Y)
    #
    # ASSUMPTION: the gradient is perfectly linear (first-order approximation).
    # For production runs with real orographic or tectonic patterns, use the
    # "mapped" modes (LITHOLOGY_MODE, etc.) with raster inputs instead.

    # --- Precipitation: base rate ---
    # BASE_PRECIP_RATE — the domain-wide mean annual precipitation delivered
    #   to every grid node before any spatial gradient is applied (m/yr).
    #   The spatial gradient below then multiplies this base value so that
    #   individual nodes receive BASE_PRECIP_RATE × spatial_multiplier.
    #
    #   UNIT NOTE: Landlab's FlowAccumulator treats water__unit_flux_in as a
    #   water depth per year (m/yr).  Set this value to match the long-term
    #   mean catchment precipitation, e.g.:
    #     Rio Coca lowland baseline ≈ 1.0 m/yr  (1000 mm/yr)
    #     High-Andes orographic max  ≈ 4.5 m/yr (4500 mm/yr, via gradient)
    #
    #   PHYSICAL INTERPRETATION
    #   ------------------------
    #   Drainage area × BASE_PRECIP_RATE × spatial_multiplier ≈ mean annual
    #   discharge at each node.  Adjusting this controls erosion power without
    #   changing erodibility parameters.
    BASE_PRECIP_RATE     = 1.0    # Basin-wide mean annual precipitation (m/yr)

    # PRECIP_TIMESERIES — optional piecewise-constant time series of base
    #   precipitation rates.  Set to None to use BASE_PRECIP_RATE for the
    #   entire run.  When provided, BASE_PRECIP_RATE is ignored.
    #
    #   FORMAT: list of (year, rate_m_yr) tuples, sorted by year.
    #   The rate at time t is the rate of the most-recent entry whose year
    #   is ≤ t (step-function, no interpolation).
    #   The first entry should have year = 0.0 to set the initial condition.
    #
    #   EXAMPLE — two-phase wetter/drier climate:
    #     PRECIP_TIMESERIES = [
    #         (    0.0, 1.0),   # 1.0 m/yr from model start
    #         ( 5000.0, 1.5),   # 1.5 m/yr from year 5 000 onward
    #         (10000.0, 0.8),   # 0.8 m/yr from year 10 000 onward
    #     ]
    PRECIP_TIMESERIES    = None   # Set to a list of (year, m/yr) tuples, or None

    # --- Precipitation: spatial gradient (multipliers) ---
    # The gradient is computed as a dimensionless multiplier field that is
    # then multiplied by the effective base rate (BASE_PRECIP_RATE or
    # the current PRECIP_TIMESERIES value) to produce the physical flux
    # (m/yr) at every node.
    #
    #   PRECIP_MIN — multiplier at the "dry" end of the gradient.
    #     1.0 = same as the base rate; < 1.0 = drier than basin average.
    #   PRECIP_MAX — multiplier at the "wet" end.
    #     > 1.0 = wetter than basin average (orographic amplification).
    #
    # Rio Coca example: rainfall increases from the lowland Amazon plain
    # (East, PRECIP_MIN × base) toward the Andes (West, PRECIP_MAX × base).
    PRECIP_MIN           = 1.0    # Dry-end spatial multiplier  (unitless)
    PRECIP_MAX           = 2.0    # Wet-end spatial multiplier  (unitless)
    PRECIP_GRADIENT_ANGLE_DEG = 270.0
    #   0°   = gradient increases northward (+Y)
    #   90°  = gradient increases eastward  (+X)
    #   270° = gradient increases westward  (-X)  ← toward Andes for Rio Coca

    # --- Tectonic uplift: base rate ---
    # BASE_UPLIFT_RATE — the domain-wide mean long-term rock-uplift rate
    #   relative to the open outlet boundary (m/yr).  The spatial gradient
    #   below then multiplies this base value node-by-node.
    #
    #   UNIT NOTE: rock uplift is in m/yr (vertical uplift relative to base
    #   level).  It is applied to bedrock__elevation each timestep as:
    #     bedrock__elevation += spatial_multiplier × effective_base_rate × dt
    #
    #   TYPICAL RANGES
    #   --------------
    #   Stable cratons / passive margins : 0.00001 – 0.0001 m/yr
    #   Active fold-and-thrust belts     : 0.0005  – 0.005  m/yr
    #   Collision orogens (Himalayas)    : 0.005   – 0.02   m/yr
    #
    #   Rio Coca example (volcanic arc): ~0.001 m/yr basin mean.
    BASE_UPLIFT_RATE     = 0.001  # Basin-wide mean rock-uplift rate (m/yr)

    # UPLIFT_TIMESERIES — optional piecewise-constant time series of base
    #   uplift rates.  Format and step-function behaviour are identical to
    #   PRECIP_TIMESERIES.  Set to None to use BASE_UPLIFT_RATE throughout.
    #
    #   EXAMPLE — tectonic pulse:
    #     UPLIFT_TIMESERIES = [
    #         (    0.0, 0.0005),  # slow uplift initially
    #         ( 4000.0, 0.002 ),  # pulse of fast uplift from yr 4 000
    #         ( 7000.0, 0.0005),  # return to slow uplift
    #     ]
    UPLIFT_TIMESERIES    = None   # Set to a list of (year, m/yr) tuples, or None

    # --- Tectonic uplift: spatial gradient (multipliers) ---
    # Like the precipitation gradient, these are dimensionless multipliers
    # applied to BASE_UPLIFT_RATE (or the current UPLIFT_TIMESERIES value).
    #
    #   UPLIFT_MIN_MULT — multiplier at the "slow" end of the gradient.
    #     1.0 = same as base rate; < 1.0 = slower-uplifting flank.
    #   UPLIFT_MAX_MULT — multiplier at the "fast" end.
    #     > 1.0 = faster-uplifting flank (e.g. toward a thrust front).
    #
    # Rio Coca example: uplift increases from the Amazonian foreland
    # (East, slow) toward the volcanic arc (West, fast).
    UPLIFT_MIN_MULT      = 0.5    # Slow-end spatial multiplier  (unitless)
    UPLIFT_MAX_MULT      = 2.5    # Fast-end spatial multiplier  (unitless)
    UPLIFT_GRADIENT_ANGLE_DEG = 270.0
    #   0°   = gradient increases northward (+Y)
    #   90°  = gradient increases eastward  (+X)
    #   270° = gradient increases westward  (-X)  ← toward Andes for Rio Coca

    # -----------------------------------------------------------------------
    # LANDSLIDE PARAMETERS  (active only when USE_LANDSLIDES = True)
    # -----------------------------------------------------------------------
    # DT_LANDSLIDE_TRIGGER — mass-movement is applied every N simulated years.
    #   Decoupling landslides from the main loop improves performance when
    #   landslide events are rare relative to the fluvial timestep.
    DT_LANDSLIDE_TRIGGER = 1.0

    # LS_ANGLE_FRICTION — tangent of the internal friction angle (m/m).
    #   Typical range: 0.4 – 0.8; humid tropical soils ~0.5–0.6.
    LS_ANGLE_FRICTION = 0.6

    # LS_COHESION_EFF — effective cohesion of the regolith (Pa).
    #   Controls minimum stable depth before failure occurs.
    LS_COHESION_EFF = 10000.0

    # LS_RETURN_TIME — mean recurrence interval for stochastic landslide
    #   triggering (years).  Shorter return time → more frequent failures.
    LS_RETURN_TIME = 300.0

    # -----------------------------------------------------------------------
    # PLOTTING & ARCHIVAL INTERVALS
    # -----------------------------------------------------------------------
    PLOT_INTERVAL_YEARS = False   # Render a map every N simulated years; set to False to disable
    SAVE_INTERVAL_YEARS = 100.0   # Write a NetCDF snapshot every N simulated years

    # SAVE_FIELDS — list of Landlab field names to include in each snapshot.
    #   Add any field that has been registered at 'node' to this list.
    SAVE_FIELDS = [
        'topographic__elevation',
        'bedrock__elevation',
        'soil__depth',
        'drainage_area',
        'surface_water__discharge',
    ]

    # PLOT_FIELD — field name shown on the primary diagnostic map.
    PLOT_FIELD = 'topographic__elevation'

    # -----------------------------------------------------------------------
    # ADVANCED OPTIONS
    # -----------------------------------------------------------------------
    # RANDOM_SEED — integer seed for reproducible stochastic runs.
    #   Set to None for fully random sequences.
    RANDOM_SEED = 42

    # --- Synthetic landscape geometry ---
    # These settings only apply when LANDSCAPE_MODE = "synthetic".
    #
    # SYNTHETIC_GRID_SHAPE — (rows, cols) node count.
    #   Rule of thumb for runtime: doubling both dimensions multiplies runtime
    #   by ~4× (linear in node count).  Start small (50×50) for testing.
    SYNTHETIC_GRID_SHAPE = (100, 100)

    # SYNTHETIC_DX — grid cell spacing (m).
    #   Smaller dx → finer resolution but slower and potentially less stable.
    #   Stability constraint: DT_BASE < SYNTHETIC_DX² / (2 × HILLSLOPE_K)
    #   and DT_BASE < SYNTHETIC_DX / (K_BR × A_max^m × S_max^(n-1)).
    SYNTHETIC_DX         = 10.0

    # SYNTHETIC_RELIEF — total elevation difference across the grid (m).
    #   The initial surface slopes uniformly from height = SYNTHETIC_RELIEF
    #   at the "high" corner to 0 m at the "low" corner.  Setting this to 0
    #   produces a perfectly flat surface (requires noise to route flow).
    #   GUIDELINE: use values comparable to real watershed relief.
    #   Example: 500 m for a small mountain catchment; 2000 m for a large
    #   Andean watershed at 50 m grid spacing.
    SYNTHETIC_RELIEF     = 100.0

    # SYNTHETIC_GRADIENT_ANGLE_DEG — compass bearing (degrees, clockwise from
    #   North / +Y axis) pointing from the LOW end of the initial slope toward
    #   the HIGH end.  This controls which direction is "upstream".
    #
    #   ANGLE_DEG REFERENCE CARD
    #   -------------------------
    #   0°  / 360° — surface rises toward the top edge (North / +Y)
    #                 → water drains southward toward the bottom boundary
    #   90°          — surface rises toward the right edge (East / +X)
    #                 → water drains westward toward the left boundary
    #   180°         — surface rises toward the bottom edge (South)
    #                 → water drains northward toward the top boundary
    #   270°         — surface rises toward the left edge (West / -X)
    #                 → water drains eastward toward the right boundary
    #
    #   REQUIREMENT: the open (outlet) boundary edge must be on the LOW side.
    #   With the default right-edge outlet (right_is_closed=False), use
    #   ANGLE_DEG = 270° so the surface rises to the left and water drains
    #   rightward to the outlet.
    #
    #   COMMON CONFIGURATIONS
    #   ----------------------
    #   270° + right outlet  — standard west-to-east draining catchment
    #    90° + left outlet   — east-to-west draining catchment
    #     0° + bottom outlet — north-to-south draining catchment
    SYNTHETIC_GRADIENT_ANGLE_DEG = 270.0

    # MIN_ELEVATION_CLIP — floor value applied after each step to prevent
    #   runaway negative elevations (useful as a numerical safety net).
    MIN_ELEVATION_CLIP = -9000.0

    # -----------------------------------------------------------------------
    # OUTLET FLUX MONITORING
    # -----------------------------------------------------------------------
    # OUTLET_FLUX_REPORT_INTERVAL — how often (simulated years) to compute and
    #   print flow and sediment fluxes at the open downstream boundary.
    #   Set to None or False to disable the runtime monitor.
    #   Recommended: equal to SAVE_INTERVAL_YEARS or a small multiple of it.
    OUTLET_FLUX_REPORT_INTERVAL = 500.0


# ============================================================================
# USER-DEFINABLE DYNAMIC UPLIFT FUNCTION
# ============================================================================
def uplift_rate_at_time(t, config):
    """
    Returns a scalar multiplier applied to the spatial uplift field at time t (yr).

    USAGE
    -----
    Override the body of this function to implement any time-varying uplift
    scenario (e.g. glacial-isostatic rebound, tectonic pulses).

    ASSUMPTIONS
    -----------
    The returned value multiplies the pre-computed uplift_shape array, so it
    is a dimensionless scale factor (1.0 = no change from baseline rates).

    Parameters
    ----------
    t      : float  Current simulation time (years)
    config : Config Configuration object (exposes BASE_UPLIFT_RATE, UPLIFT_MIN_MULT, etc.)

    Returns
    -------
    float : Scale factor applied to base uplift rate (1.0 = no change).
    """
    # Default: constant uplift (scale = 1.0 always)
    return 1.0
    # Example: linear ramp from 0 to 1 over the first 5000 years:
    # return min(t / 5000.0, 1.0)


def get_effective_precip_rate(t, config):
    """
    Returns the effective base precipitation rate (m/yr) at simulation time t.

    If ``config.PRECIP_TIMESERIES`` is ``None``, returns ``config.BASE_PRECIP_RATE``
    unchanged for every timestep.

    When a time series is provided the function performs a **step lookup**: the
    rate associated with the most-recent entry whose year is ≤ t is returned.
    This represents a simple piecewise-constant (step-function) climate forcing
    — the rate changes instantaneously at each listed year and stays constant
    until the next entry.

    The spatial gradient (PRECIP_MIN / PRECIP_MAX multipliers) is applied
    separately in the main loop so that only the scalar base rate is returned
    here.

    Parameters
    ----------
    t      : float   Current simulation time (years).
    config : Config  Must expose BASE_PRECIP_RATE and PRECIP_TIMESERIES.

    Returns
    -------
    float : Effective precipitation rate at time t (m/yr).
    """
    if config.PRECIP_TIMESERIES is None:
        return config.BASE_PRECIP_RATE

    # Step-function lookup: use the last entry whose year <= t
    active_rate = config.PRECIP_TIMESERIES[0][1]   # default = first entry
    for year, rate in config.PRECIP_TIMESERIES:
        if year <= t:
            active_rate = rate
        else:
            break   # list is sorted; no need to continue
    return active_rate


def get_effective_uplift_rate(t, config):
    """
    Returns the effective base rock-uplift rate (m/yr) at simulation time t.

    Parallels ``get_effective_precip_rate``: honours ``config.UPLIFT_TIMESERIES``
    when provided, otherwise returns ``config.BASE_UPLIFT_RATE``.

    If ``config.USE_DYNAMIC_UPLIFT`` is True the returned base rate is further
    scaled by ``uplift_rate_at_time(t, config)`` so that both mechanisms
    (time-series table and the custom function) can operate together.

    The spatial gradient multiplied shape (UPLIFT_MIN_MULT / UPLIFT_MAX_MULT)
    is stored in ``uplift_shape`` and applied in the main loop; only the
    scalar component is returned here.

    Parameters
    ----------
    t      : float   Current simulation time (years).
    config : Config  Must expose BASE_UPLIFT_RATE, UPLIFT_TIMESERIES,
                     and USE_DYNAMIC_UPLIFT.

    Returns
    -------
    float : Effective uplift rate at time t (m/yr) before spatial weighting.
    """
    if config.UPLIFT_TIMESERIES is None:
        base = config.BASE_UPLIFT_RATE
    else:
        base = config.BASE_UPLIFT_RATE   # default in case t < first entry
        for year, rate in config.UPLIFT_TIMESERIES:
            if year <= t:
                base = rate
            else:
                break

    # Optionally compound with the user-defined dynamic multiplier
    if config.USE_DYNAMIC_UPLIFT:
        base *= uplift_rate_at_time(t, config)

    return base


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dem_to_grid(file_path):
    """
    Robust DEM loader that attempts multiple Landlab backend methods in order
    of preference, falling back to xarray/rioxarray for maximum compatibility.

    ASSUMPTION: The input raster is in a projected coordinate system (metres).
    Geographic CRS (degrees) will produce incorrect areas and slopes.

    Parameters
    ----------
    file_path : str  Absolute or relative path to a GeoTIFF (or other GDAL raster).

    Returns
    -------
    landlab.RasterModelGrid  Grid with 'topographic__elevation' registered at nodes.
    """
    print(f"Loading terrain data from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The DEM file {file_path} was not found.")

    try:
        from landlab.io.raster import read_raster
        return read_raster(file_path)
    except ImportError:
        try:
            from landlab.io.dem import read_geotiff
            return read_geotiff(file_path)
        except ImportError:
            # Reliable fallback using xarray/rioxarray
            import xarray as xr
            import rioxarray
            da = xr.open_dataset(file_path, engine='rasterio').band_data[0]
            rows, cols = da.shape
            dy, dx = da.rio.resolution()
            grid = RasterModelGrid((rows, cols), xy_spacing=abs(dx))
            grid.add_field('topographic__elevation', da.values.astype(float).flatten(), at='node')
            return grid


def load_raster_to_node_array(file_path, grid, field_name, nodata_fill=0.0):
    """
    Load a single-band GeoTIFF (or NetCDF) raster and return its values
    resampled / aligned to the Landlab grid node array.

    REQUIREMENT: The raster must share the same CRS and cover the same spatial
    extent as the model grid.  Nearest-neighbour resampling is applied so minor
    resolution mismatches are acceptable.

    ASSUMPTION: NoData cells in the source raster are filled with `nodata_fill`
    before being assigned to the grid.  Choose a physically sensible fill value
    for the property being mapped (e.g. 0.0 for erodibility would produce
    un-erodible nodes, so choose a domain minimum instead).

    Parameters
    ----------
    file_path   : str    Path to the source raster file.
    grid        : RasterModelGrid  Target Landlab grid.
    field_name  : str    Human-readable label used in log messages.
    nodata_fill : float  Value substituted for NoData pixels (default 0.0).

    Returns
    -------
    numpy.ndarray  Shape (grid.number_of_nodes,), dtype float64.
    """
    print(f"      Loading mapped {field_name} from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mapped field file not found: {file_path}")

    try:
        import rioxarray  # noqa: F401  (used inside xr.open_dataset)
        import xarray as xr
        from scipy.ndimage import zoom

        ds = xr.open_dataset(file_path, engine='rasterio')
        # Accept both 'band_data' (rioxarray default) and first variable
        if 'band_data' in ds:
            arr = ds['band_data'].isel(band=0).values.astype(float)
        else:
            var = list(ds.data_vars)[0]
            arr = ds[var].values.astype(float)

        # Replace NoData values
        arr = np.where(np.isfinite(arr), arr, nodata_fill)

        # Resample to match grid dimensions if needed
        target_shape = (grid.shape[0], grid.shape[1])
        if arr.shape != target_shape:
            zoom_factors = (target_shape[0] / arr.shape[0],
                            target_shape[1] / arr.shape[1])
            arr = zoom(arr, zoom_factors, order=0)  # nearest-neighbour
            print(f"        Resampled from {arr.shape} to {target_shape}")

        return arr.flatten()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load mapped {field_name} from {file_path}: {exc}\n"
            "Ensure rioxarray and scipy are installed and the raster CRS/extent "
            "matches the model grid."
        ) from exc


def _apply_nodata_mask_to_all_fields(grid, config):
    """
    Propagate the closed-node (NoData) mask from the terrain DEM to every
    registered node field, ensuring full spatial consistency across all
    mapped data inputs.

    WHY THIS IS NEEDED
    ------------------
    When spatial inputs (soil depth, erodibility rasters, etc.) are loaded
    from GeoTIFFs that may have slightly different extents, projections, or
    NoData encodings from the primary DEM, individual cells can end up with
    physically meaningful-looking values at locations where the DEM itself
    has no valid data.  That mismatch causes three classes of problems:

      1. Numerical — SPACE or the diffuser may attempt to erode a node
         that is topographically invalid (elevation = -9999), producing
         extreme values that propagate through the domain.
      2. Visual    — diagnostic plots show spurious highs/lows at grid edges.
      3. Mass balance — sediment or bedrock "created" at NoData nodes inflates
         volume budgets.

    This function resets all registered node fields to physically safe
    sentinel values at every CLOSED node (those tagged as NoData by the DEM).

    SENTINEL VALUES
    ---------------
    Field                   Sentinel    Rationale
    ─────────────────────── ─────────── ──────────────────────────────────────
    soil__depth             0.0         Bedrock exposed; no sediment to erode
    bedrock__erodibility    0.0         Zero erodibility → zero incision
    sediment__erodibility   0.0         Ditto for alluvial cover
    water__unit_flux_in     0.0         No precipitation input → no runoff

    topographic__elevation and bedrock__elevation are intentionally left as-is
    because their NoData encoding (−9999) is the very signal that defines
    which nodes become CLOSED.  Altering them here would interfere with the
    boundary-condition logic already applied.

    Parameters
    ----------
    grid   : RasterModelGrid  Model grid with boundary conditions already set.
    config : Config           Configuration object (used for log messages only).
    """
    closed = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    n_closed = int(np.sum(closed))
    if n_closed == 0:
        print("      NoData mask: no closed nodes found — nothing to propagate.")
        return

    # Mapping: field name → sentinel value at closed nodes
    sentinel_map = {
        'soil__depth':            0.0,
        'bedrock__erodibility':   0.0,
        'sediment__erodibility':  0.0,
        'water__unit_flux_in':    0.0,
    }

    masked_fields = []
    for field_name, sentinel in sentinel_map.items():
        if field_name in grid.at_node:
            grid.at_node[field_name][closed] = sentinel
            masked_fields.append(field_name)

    print(
        f"      NoData mask propagated to {len(masked_fields)} field(s) "
        f"at {n_closed:,} closed nodes: {', '.join(masked_fields)}"
    )


def _find_watershed_outlet(grid, topo, nodata_value=-9999.0):
    """
    Find the watershed pour point: the lowest-elevation valid node that has
    at least one 4-connected neighbour that is either NoData or on the raster
    grid edge.

    This works for watershed-clipped DEMs where the outlet is an *interior*
    node of the raster (surrounded by NoData on the watershed side) — the
    situation where Landlab's built-in ``set_watershed_boundary_condition``
    fails because it only searches the raster perimeter.

    Parameters
    ----------
    grid        : RasterModelGrid
    topo        : numpy.ndarray  topographic__elevation (already nodata-normalised)
    nodata_value: float          Sentinel for invalid cells (default -9999.0)

    Returns
    -------
    int  Node index of the detected outlet.

    Raises
    ------
    RuntimeError if no valid perimeter candidate is found.
    """
    nr = grid.number_of_node_rows
    nc = grid.number_of_node_columns

    is_nodata = (topo == nodata_value)
    is_valid  = ~is_nodata

    nodata_2d = is_nodata.reshape((nr, nc))
    valid_2d  = is_valid.reshape((nr, nc))

    pad_row = np.zeros((1, nc), dtype=bool)
    pad_col = np.zeros((nr, 1), dtype=bool)

    # 4-connected neighbours that are NoData
    above = np.vstack([pad_row,            nodata_2d[:-1, :]])
    below = np.vstack([nodata_2d[1:, :],   pad_row          ])
    left  = np.hstack([pad_col,            nodata_2d[:, :-1]])
    right = np.hstack([nodata_2d[:, 1:],   pad_col          ])

    # Grid-edge cells count as "adjacent to NoData" for outlet detection —
    # this handles cases where the raster edge itself borders the watershed.
    on_edge = np.zeros((nr, nc), dtype=bool)
    on_edge[0, :]  = True
    on_edge[-1, :] = True
    on_edge[:, 0]  = True
    on_edge[:, -1] = True

    has_nodata_or_edge_neighbor = above | below | left | right | on_edge
    perimeter_nodes = np.where((valid_2d & has_nodata_or_edge_neighbor).ravel())[0]

    if len(perimeter_nodes) == 0:
        raise RuntimeError(
            "Cannot find watershed outlet — no valid nodes border any NoData "
            "cell or raster edge.\n"
            "Possible causes:\n"
            "  • The DEM has no NoData cells (check that nodata_value matches "
            "the file's nodata encoding).\n"
            "  • All nodes are NoData.\n"
            "Set OUTLET_NODE_ROWCOL = (row, col) in Config to specify the "
            "outlet manually."
        )

    outlet_node = int(perimeter_nodes[np.argmin(topo[perimeter_nodes])])
    print(f"      ({len(perimeter_nodes):,} watershed-perimeter candidates evaluated)")
    return outlet_node


def _apply_watershed_bc(grid, topo, config):
    """
    Apply watershed boundary conditions using Landlab's cache-aware method.

    Strategy
    --------
    For watershed-clipped DEMs every raster-edge pixel is NoData, so
    Landlab's ``set_watershed_boundary_condition`` (which only searches the
    raster perimeter for the outlet) raises a ValueError.  Instead we:

      1. Normalise nodata: NaN values and any elevation below -9000 m are
         converted to exactly -9999.0 so the nodata test is unambiguous.
      2. Detect the outlet node with ``_find_watershed_outlet`` — finds the
         lowest-elevation valid node adjacent to any NoData or edge cell.
      3. Call ``set_watershed_boundary_condition_outlet_id(outlet_id, ...)``
         which correctly marks nodata nodes CLOSED, closes remaining perimeter
         nodes, sets the outlet to FIXED_VALUE, **and** updates all internal
         Landlab link-status caches so that PriorityFloodFlowRouter and Space
         recognise the outlet.

    If ``config.OUTLET_NODE_ROWCOL`` is provided the specified node is used
    directly instead of step 2.

    Parameters
    ----------
    grid   : RasterModelGrid
    topo   : numpy.ndarray  topographic__elevation node array (modified in-place
             to normalise nodata)
    config : Config
    """
    nodata_value = -9999.0

    # 1. Normalise nodata — GeoTIFFs may encode nodata as NaN or as a large
    #    negative float that is not exactly -9999.0.  Standardise everything
    #    to -9999.0 so the nodata test below is unambiguous.
    bad = ~np.isfinite(topo) | (topo < -9000.0)
    if np.any(bad):
        topo[bad] = nodata_value

    # 2. Determine outlet node
    manual_rc = getattr(config, 'OUTLET_NODE_ROWCOL', None)
    if manual_rc is not None:
        row, col  = int(manual_rc[0]), int(manual_rc[1])
        outlet_id = row * grid.number_of_node_columns + col
        print(
            f"      Outlet (manual override): node {outlet_id} "
            f"(row={row}, col={col}, z={topo[outlet_id]:.2f} m) → FIXED_VALUE."
        )
    else:
        outlet_id = _find_watershed_outlet(grid, topo, nodata_value)
        nc  = grid.number_of_node_columns
        row = outlet_id // nc
        col = outlet_id  % nc
        print(
            f"      Outlet (auto-detected): node {outlet_id} "
            f"(row={row}, col={col}, z={topo[outlet_id]:.2f} m) → FIXED_VALUE."
        )

    # 3. Apply BCs via the outlet-id variant — works for both raster-perimeter
    #    and interior outlet nodes, and correctly updates Landlab's caches.
    try:
        grid.set_watershed_boundary_condition_outlet_id(
            outlet_id, topo, nodata_value=nodata_value
        )
    except AttributeError:
        # Very old Landlab builds lack this method; fall back to direct assignment.
        # This misses some cache updates but is better than crashing.
        grid.status_at_node[topo == nodata_value] = grid.BC_NODE_IS_CLOSED
        grid.status_at_node[grid.boundary_nodes]  = grid.BC_NODE_IS_CLOSED
        grid.status_at_node[outlet_id]            = grid.BC_NODE_IS_FIXED_VALUE

    n_fixed = int(np.sum(grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE))
    print(f"      Boundary conditions set: {n_fixed} FIXED_VALUE outlet node(s).")
    if n_fixed == 0:
        raise RuntimeError(
            "Zero open boundary nodes after BC setup — the model cannot route "
            "flow.\nPossible fixes:\n"
            "  • Set OUTLET_NODE_ROWCOL = (row, col) in Config to pinpoint "
            "the outlet node manually.\n"
            "  • Check that the DEM nodata encoding is a large negative number "
            "(< -9000) or NaN so it is normalised correctly."
        )


def _ensure_open_outlet(grid, config):
    """Deprecated wrapper — delegates to _apply_watershed_bc."""
    _apply_watershed_bc(grid, grid.at_node['topographic__elevation'], config)


# ============================================================================
# CHANNEL FOOTPRINT CONDITIONING
# ============================================================================

def _read_tfw_origin(dem_path):
    """
    Parse the TFW world file that accompanies the DEM GeoTIFF and return the
    geo-referenced origin of the lower-left pixel centre.

    The TFW encodes the upper-left pixel centre; this function converts to
    lower-left using the grid dimensions inferred from the DEM's aux.xml.

    Returns
    -------
    (x_origin, y_origin, dx) — all in the DEM's native CRS (metres).
    Returns None if the TFW file does not exist.
    """
    tfw_path = dem_path.replace('.tif', '.tfw')
    if not os.path.exists(tfw_path):
        return None
    with open(tfw_path) as f:
        lines = f.read().strip().split('\n')
    dx = float(lines[0])
    # dy is negative (line 3) for N-to-S rasters
    x_ul = float(lines[4])
    y_ul = float(lines[5])
    return x_ul, y_ul, abs(dx)


def _load_channel_footprint(config, grid):
    """
    Load the channel footprint shapefile and convert polyline vertices to an
    ordered sequence of grid node indices (upstream → downstream).

    The shapefile coordinates are in the DEM's projected CRS (UTM metres).
    The Landlab grid uses local coordinates starting at (0, 0) for the lower-
    left node, so we compute the offset from the TFW world file.

    Parameters
    ----------
    config : Config
    grid   : RasterModelGrid

    Returns
    -------
    channel_nodes : numpy.ndarray of int  Ordered node indices along the
                    channel.  Duplicate nodes are removed.  The array runs
                    from the highest-elevation end (upstream) to the lowest
                    (downstream).
    """
    shp_path = config.CHANNEL_FOOTPRINT_FILE
    if shp_path is None or not os.path.exists(shp_path):
        raise FileNotFoundError(
            f"Channel footprint shapefile not found: {shp_path}"
        )

    # --- Read shapefile vertices ---
    sf = shp.Reader(shp_path)
    coords = []
    for shape in sf.shapes():
        coords.extend(shape.points)  # list of (x, y) tuples
    coords = np.array(coords)        # shape (N, 2)

    # --- Compute UTM → row/col mapping ---
    # The DEM is stored in Landlab with raster convention: row 0 = northernmost
    # row (Y = y_ul in UTM), with row index *increasing* southward (Y decreasing).
    # This is the opposite of Landlab's standard convention (row 0 = south), and
    # is why the terrain appears upside-down when displayed with origin='lower'.
    # The correct mapping for a UTM coordinate (x, y) is therefore:
    #   col = round((x - x_ul) / dx)   [X increases eastward — same direction]
    #   row = round((y_ul - y) / dy)   [Y decreases as row increases]
    origin = _read_tfw_origin(config.DEM_FILE)
    if origin is not None:
        x_ul, y_ul, dx_tfw = origin
        cols = np.round((coords[:, 0] - x_ul) / grid.dx).astype(int)
        rows = np.round((y_ul - coords[:, 1]) / grid.dy).astype(int)
    else:
        # Fallback when no TFW: cannot determine UTM origin; warn and use
        # Landlab-convention formula (may still be misaligned).
        print("      [WARNING] No TFW world file found — channel footprint "
              "coordinate transform may be inaccurate.")
        x0 = grid.node_x.min()
        nr = grid.number_of_node_rows
        y_top = (nr - 1) * grid.dy    # local Y of row 0 in Landlab space
        cols = np.round((coords[:, 0] - x0) / grid.dx).astype(int)
        rows = np.round((y_top - (coords[:, 1] - grid.node_y.min())) / grid.dy).astype(int)
    nc = grid.number_of_node_columns
    nr = grid.number_of_node_rows

    # Clip to valid range
    cols = np.clip(cols, 0, nc - 1)
    rows = np.clip(rows, 0, nr - 1)

    node_ids = rows * nc + cols

    # --- Remove consecutive duplicates (polyline denser than 50 m grid) ---
    mask = np.concatenate(([True], node_ids[1:] != node_ids[:-1]))
    node_ids = node_ids[mask]

    # --- Ensure upstream-to-downstream ordering (highest → lowest) ---
    topo = grid.at_node['topographic__elevation']
    if topo[node_ids[0]] < topo[node_ids[-1]]:
        node_ids = node_ids[::-1]

    # --- Filter out any nodes that fall on NoData ---
    nodata_value = -9999.0
    valid = topo[node_ids] > nodata_value
    node_ids = node_ids[valid]

    print(f"      Channel footprint loaded: {len(node_ids)} nodes "
          f"from {shp_path}")

    return node_ids


def _condition_channel(grid, channel_nodes, config):
    """
    Enforce monotonically non-increasing elevation along the channel path and
    incise the channel below neighbouring cells.

    Parameters
    ----------
    grid          : RasterModelGrid
    channel_nodes : numpy.ndarray of int  Ordered node indices, upstream →
                    downstream.
    config        : Config

    Returns
    -------
    outlet_node : int  The downstream-most valid channel node — to be used as
                  the watershed outlet.
    """
    topo = grid.at_node['topographic__elevation']
    nodata_value = -9999.0
    incision = config.CHANNEL_INCISION_DEPTH
    nc = grid.number_of_node_columns
    nr = grid.number_of_node_rows

    # --- Pass 1: enforce monotonically non-increasing elevation ---
    # Walk upstream → downstream; if elevation rises, flatten to previous value.
    for i in range(1, len(channel_nodes)):
        n = channel_nodes[i]
        n_prev = channel_nodes[i - 1]
        if topo[n] > topo[n_prev]:
            topo[n] = topo[n_prev]

    # --- Pass 2: incise channel below 4-connected non-channel neighbours ---
    channel_set = set(channel_nodes.tolist())
    for n in channel_nodes:
        row = n // nc
        col = n % nc
        neighbors = []
        if row > 0:
            neighbors.append(n - nc)        # below
        if row < nr - 1:
            neighbors.append(n + nc)        # above
        if col > 0:
            neighbors.append(n - 1)         # left
        if col < nc - 1:
            neighbors.append(n + 1)         # right

        # Consider only valid non-channel neighbours
        valid_nbrs = [nb for nb in neighbors
                      if nb not in channel_set and topo[nb] > nodata_value]
        if not valid_nbrs:
            continue
        min_nbr = min(topo[nb] for nb in valid_nbrs)
        max_allowed = min_nbr - incision
        if topo[n] > max_allowed:
            topo[n] = max_allowed

    # --- Pass 3: re-enforce monotonic after incision adjustments ---
    for i in range(1, len(channel_nodes)):
        n = channel_nodes[i]
        n_prev = channel_nodes[i - 1]
        if topo[n] > topo[n_prev]:
            topo[n] = topo[n_prev]

    # --- Update bedrock and soil fields if they exist ---
    if 'bedrock__elevation' in grid.at_node:
        bedrock = grid.at_node['bedrock__elevation']
        soil = grid.at_node['soil__depth']
        for n in channel_nodes:
            if bedrock[n] > topo[n]:
                bedrock[n] = topo[n]
                soil[n] = 0.0
            else:
                soil[n] = topo[n] - bedrock[n]

    # --- Identify outlet: last valid downstream channel node ---
    outlet_node = int(channel_nodes[-1])

    nc_grid = grid.number_of_node_columns
    row = outlet_node // nc_grid
    col = outlet_node % nc_grid
    print(f"      Channel conditioned: {len(channel_nodes)} nodes processed.")
    print(f"      Elevation range along channel: "
          f"{topo[channel_nodes[-1]]:.2f} – {topo[channel_nodes[0]]:.2f} m")
    print(f"      Channel outlet node: {outlet_node} "
          f"(row={row}, col={col}, z={topo[outlet_node]:.2f} m)")

    return outlet_node


def _show_channel_approval_map(grid, channel_nodes, outlet_node):
    """
    Display a terrain map with the channel footprint and outlet for visual
    approval before starting the simulation.

    Closes after user presses a key or closes the window.
    """
    topo = grid.at_node['topographic__elevation']
    nr = grid.number_of_node_rows
    nc = grid.number_of_node_columns

    # Build a masked elevation grid for display.
    # The DEM is stored with raster convention (row 0 = north), so display
    # with origin='upper' so that north appears at the top of the map.
    z2d = topo.reshape((nr, nc)).copy()
    z2d[z2d < -9000] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(
        z2d, origin='upper', cmap='terrain', aspect='auto'
    )
    plt.colorbar(im, ax=ax, label='Elevation (m)')

    # Overlay channel in pixel (col, row) space — consistent with imshow
    # regardless of coordinate convention.
    ch_rows = channel_nodes // nc
    ch_cols = channel_nodes % nc
    ax.plot(ch_cols, ch_rows, 'b-', linewidth=2, label='Channel footprint')

    # Mark outlet
    out_row = outlet_node // nc
    out_col = outlet_node % nc
    ax.plot(out_col, out_row, 'ro', markersize=12, fillstyle='none',
            markeredgewidth=2, label='Outlet')

    ax.set_title('Channel Footprint & Outlet (north up) — Close window to continue')
    ax.set_xlabel('Column (west → east)')
    ax.set_ylabel('Row (north → south)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def _apply_channel_footprint(grid, config):
    """
    Master function: load the channel shapefile, condition the terrain,
    show the approval map, and set the outlet.

    Called from ``initialize_model()`` after grid construction and topo field
    registration, but BEFORE sink filling and component instantiation.

    If ``config.CHANNEL_FOOTPRINT_FILE`` is None or the file does not exist,
    prints a message and returns without modifying anything.

    Parameters
    ----------
    grid   : RasterModelGrid
    config : Config

    Returns
    -------
    outlet_node : int or None
        Node index of the channel-derived outlet, or None if no channel
        footprint was applied (caller should fall back to auto-detection).
    """
    if not getattr(config, 'USE_CHANNEL_FOOTPRINT', True):
        return None

    shp_path = getattr(config, 'CHANNEL_FOOTPRINT_FILE', None)
    if shp_path is None:
        print("      No channel manually identified – Channel system will be "
              "automatically determined.")
        return None
    if not os.path.exists(shp_path):
        print(f"      [WARNING] Channel footprint file not found: {shp_path}")
        print("      No channel manually identified – Channel system will be "
              "automatically determined.")
        return None

    print("      Loading and conditioning channel footprint...")
    channel_nodes = _load_channel_footprint(config, grid)

    if len(channel_nodes) < 2:
        print("      [WARNING] Channel footprint contains fewer than 2 valid "
              "nodes — skipping channel conditioning.")
        return None

    outlet_node = _condition_channel(grid, channel_nodes, config)

    # Show approval map
    _show_channel_approval_map(grid, channel_nodes, outlet_node)

    # Override the outlet in config so _apply_watershed_bc uses it
    nc = grid.number_of_node_columns
    row = outlet_node // nc
    col = outlet_node % nc
    config.OUTLET_NODE_ROWCOL = (row, col)

    return outlet_node


def create_gradient(grid_obj, min_val, max_val, angle_deg=90.0):
    """
    Creates a spatially linear field across the model grid whose orientation
    is controlled by a compass bearing angle.

    GEOMETRY
    --------
    The gradient direction is specified as a clockwise bearing from N (+Y axis):
      0°  / 360°  — field increases toward the top row  (North / +Y)
      90°         — field increases toward the right edge (East  / +X)
     180°         — field increases toward the bottom row (South / -Y)
     270°         — field increases toward the left edge  (West  / -X)

    The implementation projects every node's (x, y) coordinate onto the
    unit vector pointing in the specified direction, then linearly maps
    the projection range onto [min_val, max_val].

    BACKWARD COMPATIBILITY
    ----------------------
    The old two-argument call create_gradient(grid, a, b) is still supported;
    it defaults to angle_deg=90° (East / +X direction), which matches the
    previous diagonal-projection behaviour for fields that varied primarily
    west-to-east.

    Parameters
    ----------
    grid_obj  : RasterModelGrid
    min_val   : float  Field value at the low end of the gradient direction.
    max_val   : float  Field value at the high end of the gradient direction.
    angle_deg : float  Compass bearing (degrees CW from North) pointing from
                       min_val toward max_val.  Default 90° = eastward.

    Returns
    -------
    numpy.ndarray  Shape (grid.number_of_nodes,), dtype float64.
    """
    # Convert compass bearing to standard mathematical angle (CCW from +X)
    # Compass: 0°=N(+Y), 90°=E(+X), 180°=S(-Y), 270°=W(-X)
    # Math:    0°=+X,   90°=+Y
    math_angle_rad = np.radians(90.0 - angle_deg)
    dx_hat = np.cos(math_angle_rad)   # +X component of gradient direction
    dy_hat = np.sin(math_angle_rad)   # +Y component of gradient direction

    # Project all node coordinates onto the gradient direction vector
    proj = grid_obj.node_x * dx_hat + grid_obj.node_y * dy_hat

    # Normalise projection to [0, 1] then scale to [min_val, max_val]
    proj_range = proj.max() - proj.min()
    if proj_range == 0:
        return np.full(grid_obj.number_of_nodes, (min_val + max_val) / 2.0)
    proj_norm = (proj - proj.min()) / proj_range
    return min_val + (max_val - min_val) * proj_norm


def setup_stochastic_rainfall(total_time, seed=42):
    """
    Pre-generates stochastic storm sequences using exponential (Poisson-process)
    distributions for storm duration, interstorm duration, and intensity.

    PHYSICS / ASSUMPTIONS
    ---------------------
    Storm duration and interstorm duration are drawn from Exponential
    distributions, consistent with a Poisson-process (memoryless) arrival
    model for storms (e.g. Eagleson 1978).  Intensity is drawn from an
    Exponential distribution scaled to match a target mean.

    Pre-generating all storms into iterators avoids per-step random calls.
    The number of storms is over-estimated by 50 % to prevent early exhaustion;
    if the iterator runs dry the simulation falls back to mean values.

    Parameters
    ----------
    total_time : float  Total simulation length (years).
    seed       : int    Random seed for reproducibility.

    Returns
    -------
    Tuple of three iterators: (storm_dur, interstorm_dur, intensity).
    """
    n_storms = int(total_time * 1.5)
    rng = np.random.default_rng(seed)  # Modern, thread-safe random generator
    storm_dur_iter      = iter(rng.exponential(0.5, n_storms))
    interstorm_dur_iter = iter(rng.exponential(1.0, n_storms))
    # Intensity is a dimensionless multiplier with mean = 1.0.
    # The physical base rate (m/yr) is supplied at run-time by
    # get_effective_precip_rate() and multiplied in the main loop so that
    # PRECIP_TIMESERIES changes propagate correctly to stochastic runs.
    intensity_iter      = iter(rng.exponential(1.0, n_storms))
    return storm_dur_iter, interstorm_dur_iter, intensity_iter


def render_plot(grid, current_year, config, title_suffix="", auto_close_seconds=30):
    """
    Renders a map of the chosen diagnostic field (config.PLOT_FIELD) using
    Landlab's imshow_grid.

    In notebook environments (Jupyter / Colab), the figure is refreshed
    in-place via clear_output so the cell output does not grow unbounded.
    In plain Python the figure is shown as a blocking window.

    Parameters
    ----------
    grid        : RasterModelGrid
    current_year: float  Current simulation time (years).
    config      : Config  Provides PLOT_FIELD.
    title_suffix: str    Optional extra text added to the figure title.
    """
    if HAS_IPYTHON:
        clear_output(wait=True)

    field = config.PLOT_FIELD
    if field not in grid.at_node:
        field = 'topographic__elevation'  # Fallback

    plt.figure(figsize=(10, 8))
    imshow_grid(grid, field, cmap='terrain', colorbar_label=f'{field} (m)')
    title = f"Landscape Evolution - Year: {current_year:,.0f}"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    fig = plt.gcf()

    # Non-blocking display with automatic close after `auto_close_seconds`.
    if HAS_IPYTHON:
        try:
            display(fig)
        except Exception:
            pass

        def _close_ipy():
            try:
                clear_output(wait=True)
                plt.close(fig)
            except Exception:
                pass

        threading.Timer(auto_close_seconds, _close_ipy).start()
    else:
        try:
            plt.show(block=False)
        except TypeError:
            # Some backends may not accept block kwarg; fallback to blocking show
            plt.show()

        def _close():
            try:
                plt.close(fig)
            except Exception:
                pass

        threading.Timer(auto_close_seconds, _close).start()


def render_multi_panel(grid, current_year):
    """
    Renders a multi-panel diagnostic figure showing elevation, soil depth,
    drainage area (log scale), and bedrock elevation simultaneously.

    Intended for periodic in-depth diagnostic snapshots rather than every
    plot interval.  All four panels share the same spatial extent.

    Parameters
    ----------
    grid        : RasterModelGrid
    current_year: float  Current simulation time (years).
    """
    fields_to_plot = [
        ('topographic__elevation', 'terrain',  'Elevation (m)'),
        ('soil__depth',            'YlOrBr',   'Soil Depth (m)'),
        ('drainage_area',          'Blues',    'Drainage Area (m²) [log]'),
        ('bedrock__elevation',     'terrain',  'Bedrock Elevation (m)'),
    ]
    # Filter to fields that actually exist on this grid
    available = [(f, c, l) for f, c, l in fields_to_plot if f in grid.at_node]
    n = len(available)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'Diagnostic snapshot — Year {current_year:,.0f}', fontsize=13)

    for ax, (fname, cmap, label) in zip(axes, available):
        data = grid.at_node[fname].copy()
        if fname == 'drainage_area':
            data = np.log10(np.where(data > 0, data, np.nan))
            label = 'log₁₀ Drainage Area (m²)'
        plt.sca(ax)
        imshow_grid(grid, data, cmap=cmap, colorbar_label=label, plot_name=fname)

    plt.tight_layout()
    fig = plt.gcf()

    # Non-blocking display with automatic close after 30 seconds.
    if HAS_IPYTHON:
        try:
            display(fig)
        except Exception:
            pass

        def _close_ipy():
            try:
                clear_output(wait=True)
                plt.close(fig)
            except Exception:
                pass

        threading.Timer(30.0, _close_ipy).start()
    else:
        try:
            plt.show(block=False)
        except TypeError:
            plt.show()

        def _close():
            try:
                plt.close(fig)
            except Exception:
                pass

        threading.Timer(30.0, _close).start()


def create_terrain_animation(config, output_field='topographic__elevation', fps=10):
    """
    Generates an MP4 animation from all saved NetCDF snapshots, showing terrain
    evolution through time with an optimized colorbar for detecting subtle
    elevation changes.

    METHODOLOGY
    -----------
    Reads all NetCDF files in config.SAVE_DIR that match the pattern
    'lem_yr_*.nc', sorts them chronologically, and renders each frame with:

      • An optimized colormap that highlights changes in the target range
        (1000–1200 m for elevation data by default)
      • Consistent spatial extent across all frames
      • Frame timing controlled by fps (frames per second)

    The colorbar uses a PowerNorm or SymLogNorm to enhance visibility of
    smaller elevation changes while still representing the full data range.
    This is particularly useful for detecting knickpoint propagation or
    subtle valley adjustment.

    Movies are written using FFMpegWriter; requires FFmpeg to be installed
    and in the system PATH.

    Parameters
    ----------
    config            : Config  Configuration object (provides SAVE_DIR).
    output_field      : str     Landlab field name to animate (default:
                                'topographic__elevation').  Other options:
                                'bedrock__elevation', 'soil__depth', etc.
    fps               : int     Frames per second for the MP4 (default: 10;
                                use 20–30 for smoother playback).

    Returns
    -------
    None (writes MP4 file to config.SAVE_DIR)
    """
    import glob
    import xarray as xr

    print("\n" + "-" * 70)
    print("  CREATING TERRAIN ANIMATION")
    print("-" * 70)

    # Find all NetCDF snapshots
    nc_pattern = os.path.join(config.SAVE_DIR, 'lem_yr_*.nc')
    nc_files = sorted(glob.glob(nc_pattern))

    if not nc_files:
        print(f"Error: No NetCDF files found matching {nc_pattern}")
        print(f"Make sure simulation has been run and SAVE_INTERVAL_YEARS > 0")
        return

    print(f"Found {len(nc_files)} snapshots")

    # Load all data to determine global min/max and time coverage
    print("Pre-processing snapshots to determine global range...")
    all_data = []
    times = []
    grids = []

    for nc_file in nc_files:
        try:
            # Extract year from filename (lem_yr_0000000100.nc -> 100)
            year_str = os.path.basename(nc_file).split('_')[2].split('.')[0]
            year = int(year_str)
            times.append(year)

            # Load grid
            grid = read_netcdf(nc_file)
            grids.append(grid)

            # Extract elevation data
            if output_field in grid.at_node:
                data = grid.at_node[output_field].copy()
                all_data.append(data)
            else:
                print(f"  Warning: field '{output_field}' not found in {nc_file}")
                return

        except Exception as e:
            print(f"  Error loading {nc_file}: {e}")
            return

    if not all_data:
        print("Error: No valid data loaded")
        return

    # Compute global statistics for improved colorbar
    all_data_arr = np.array(all_data)
    global_min = np.nanmin(all_data_arr)
    global_max = np.nanmax(all_data_arr)
    global_mean = np.nanmean(all_data_arr)
    global_std = np.nanstd(all_data_arr)

    print(f"  Data range: {global_min:.1f} – {global_max:.1f} m")
    print(f"  Mean ± Std: {global_mean:.1f} ± {global_std:.1f} m")

    # For elevation data, use a colormap optimized for detecting changes
    # in the middle range (e.g., 1000–1200 m).  Use a slightly compressed
    # colorscale with PowerNorm to make mid-range changes stand out.
    cmap = plt.get_cmap('terrain')

    # Set up figure and output file
    grid_ref = grids[0]
    fig, ax = plt.subplots(figsize=(12, 10))
    mp4_filename = os.path.join(config.SAVE_DIR, 'terrain_animation.mp4')

    # Create MP4 writer
    writer = FFMpegWriter(fps=fps, bitrate=2400, codec='libx264')

    print(f"Writing MP4 animation: {mp4_filename}")
    print(f"  Frames: {len(nc_files)}, FPS: {fps}")

    with writer.saving(fig, mp4_filename, dpi=100):
        for idx, (data, year, grid) in enumerate(zip(all_data, times, grids)):
            ax.clear()

            # Reshape data to grid for visualization
            data_grid = data.reshape(grid.shape)

            # Use an optimized norm that enhances mid-range visibility
            # For 1000–1200 m elevation range, we apply a power norm that
            # compresses the colorscale at extremes and expands it in the middle
            vmin = global_min
            vmax = global_max
            norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

            # Create the raster plot
            im = ax.imshow(data_grid, cmap=cmap, norm=norm, origin='lower',
                           extent=[grid.node_x.min(), grid.node_x.max(),
                                   grid.node_y.min(), grid.node_y.max()],
                           interpolation='nearest')

            # Add colorbar with detailed labels
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f'{output_field.replace("__", " ")} (m)', fontsize=11)

            # Add grid lines every 1000 m to aid reference
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

            # Title and labels
            ax.set_title(f'Landscape Evolution — Year {year:,.0f}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)

            # Add frame counter at bottom
            fig.text(0.5, 0.02, f'Frame {idx+1}/{len(nc_files)}',
                     ha='center', fontsize=10, style='italic', color='gray')

            plt.tight_layout(rect=[0, 0.03, 1, 1])

            writer.grab_frame()

            # Progress indicator
            if (idx + 1) % max(1, len(nc_files) // 10) == 0:
                print(f"  Progress: {idx+1}/{len(nc_files)} frames written")

    plt.close(fig)

    print(f"Animation complete: {mp4_filename}")
    print(f"Full path: {os.path.abspath(mp4_filename)}")
    print("-" * 70 + "\n")


def compute_erosion_rate(topo_prev, topo_curr, dt):
    """
    Computes the time-averaged erosion rate field (m/yr) over one timestep.

    Positive values indicate net erosion (lowering); negative values indicate
    net deposition (aggradation).

    Parameters
    ----------
    topo_prev : numpy.ndarray  Elevation at start of step.
    topo_curr : numpy.ndarray  Elevation at end of step.
    dt        : float          Timestep duration (years).

    Returns
    -------
    numpy.ndarray  Erosion rate (m/yr) at each node.
    """
    return (topo_prev - topo_curr) / dt


def log_model_state(grid, current_year, config):
    """
    Prints a brief summary of key model-state statistics to stdout.
    Useful for monitoring model health during long runs.
    """
    topo  = grid.at_node['topographic__elevation']
    cores = grid.core_nodes
    print(
        f"  State @ yr {current_year:,.0f}: "
        f"z_mean={topo[cores].mean():.1f} m | "
        f"z_max={topo[cores].max():.1f} m | "
        f"z_min={topo[cores].min():.1f} m",
        flush=True
    )
    if 'soil__depth' in grid.at_node:
        sd = grid.at_node['soil__depth']
        print(
            f"  Soil: mean={sd[cores].mean():.2f} m | "
            f"min={sd[cores].min():.4f} m | "
            f"max={sd[cores].max():.2f} m",
            flush=True
        )


def estimate_runtime(config, n_nodes):
    """
    Prints a qualitative and semi-quantitative runtime estimate to stdout
    BEFORE the simulation begins, based on grid size, simulation length,
    timestep, and active components.

    METHODOLOGY
    -----------
    Each geomorphic component has an approximate cost per node per timestep
    expressed in microseconds (us), calibrated against typical desktop CPU
    performance (single core, ~3 GHz).  These are empirical order-of-magnitude
    values derived from Landlab benchmarking literature and community experience:

      Component                     Approx. cost (us / node / step)
      --------------------------------------------------------
      FlowAccumulator (D8)          ~0.25
      PriorityFloodFlowRouter       ~1.5    (3-6x slower than FA)
      LinearDiffuser                ~0.10
      TaylorNonLinearDiffuser       ~0.20
      StreamPowerEroder             ~0.15
      SPACE                         ~0.50
      BedrockLandslider (per call)  ~5.0    (called every DT_LANDSLIDE_TRIGGER yr)

    The estimate is intentionally conservative (upper bound) because
    real performance depends heavily on hardware, Python version, NumPy
    BLAS linkage, grid connectivity, and numerical stability events.

    INTERPRETATION
    --------------
    Use the estimate to decide whether to:
      - Increase DT_BASE (fewer timesteps -> faster, but less stable)
      - Reduce grid resolution or grid shape
      - Disable landslides (USE_LANDSLIDES=False) for exploratory runs
      - Run overnight / on a compute cluster

    Parameters
    ----------
    config  : Config  Run configuration.
    n_nodes : int     Total node count of the initialised grid.
    """
    print("\n" + "-" * 70)
    print("  RUNTIME ESTIMATE")
    print("-" * 70)

    # --- Timestep count ---
    if config.RAINFALL_MODEL == "stochastic":
        # Mean storm cycle = mean_storm_dur + mean_interstorm_dur = 0.5 + 1.0 = 1.5 yr
        mean_dt = 1.5
    else:
        mean_dt = config.DT_BASE
    n_steps = config.SIM_LENGTH / mean_dt

    # --- Per-step cost (us / node) by component ---
    # Flow router
    if config.USE_LANDSLIDES:
        router_cost_us = 1.5   # PriorityFloodFlowRouter (includes dynamic pit filling)
    else:
        router_cost_us = 0.25  # FlowAccumulator D8

    # Hillslope diffuser
    if config.HILLSLOPE_MODEL == "taylor":
        hillslope_cost_us = 0.20  # Non-linear solver slightly more expensive
    else:
        hillslope_cost_us = 0.10  # Linear diffusion

    # Fluvial
    if config.FLUVIAL_MODEL == "space":
        fluvial_cost_us = 0.50    # SPACE: dual bedrock + sediment tracking
    else:
        fluvial_cost_us = 0.15   # StreamPowerEroder: single field update

    # Total per-step cost (all main-loop components)
    per_step_us = router_cost_us + hillslope_cost_us + fluvial_cost_us
    total_step_seconds = (per_step_us * 1e-6) * n_nodes * n_steps

    # Landslide overhead (called at a coarser interval than the main loop)
    if config.USE_LANDSLIDES:
        n_ls_calls = config.SIM_LENGTH / config.DT_LANDSLIDE_TRIGGER
        ls_cost_per_call_us = 5.0  # us / node per BedrockLandslider call
        ls_total_seconds = (ls_cost_per_call_us * 1e-6) * n_nodes * n_ls_calls
    else:
        ls_total_seconds = 0.0

    total_seconds = total_step_seconds + ls_total_seconds

    # --- Qualitative label ---
    if total_seconds < 60:
        qual = "< 1 minute   -- very fast; adjust DT_BASE or grid size if desired"
    elif total_seconds < 600:
        mins = int(total_seconds / 60)
        qual = f"~{mins} min       -- fast; suitable for interactive exploration"
    elif total_seconds < 3600:
        mins = int(total_seconds / 60)
        qual = f"~{mins} minutes  -- moderate; consider running in background"
    elif total_seconds < 86400:
        hrs = total_seconds / 3600
        qual = f"~{hrs:.1f} hours   -- long run; schedule overnight or on cluster"
    elif total_seconds < 7 * 86400:
        days = total_seconds / 86400
        qual = f"~{days:.1f} days    -- very long; strongly consider coarsening the grid"
    else:
        days = total_seconds / 86400
        qual = (f"~{days:.0f}+ days  -- WARNING: impractically long. "
                "Increase DT_BASE, reduce grid size, or disable landslides.")

    # --- Print summary table ---
    print(f"  Grid nodes           : {n_nodes:>12,}")
    print(f"  Total sim time       : {config.SIM_LENGTH:>12,.0f} yr")
    print(f"  Mean timestep        : {mean_dt:>12.1f} yr")
    print(f"  Approx. timesteps    : {n_steps:>12,.0f}")
    print(
        f"  Flow router          : "
        f"{'PriorityFlood' if config.USE_LANDSLIDES else 'FlowAccumulator D8':>20}"
        f"  ({router_cost_us} us/node/step)"
    )
    print(
        f"  Hillslope model      : {config.HILLSLOPE_MODEL:>20}"
        f"  ({hillslope_cost_us} us/node/step)"
    )
    print(
        f"  Fluvial model        : {config.FLUVIAL_MODEL:>20}"
        f"  ({fluvial_cost_us} us/node/step)"
    )
    if config.USE_LANDSLIDES:
        print(
            f"  Landslide calls      : {int(n_ls_calls):>12,}"
            f"  (every {config.DT_LANDSLIDE_TRIGGER:.0f} yr)"
        )
    print(f"  Estimated wall time  : {qual}")
    print("-" * 70 + "\n")

    # --- Stability check (synthetic grids only; dx is known before loading DEM) ---
    if config.LANDSCAPE_MODE == "synthetic" and config.HILLSLOPE_K > 0:
        dx = config.SYNTHETIC_DX
        dt_diffuse_limit = dx**2 / (2.0 * config.HILLSLOPE_K)
        if mean_dt > dt_diffuse_limit:
            print(
                f"  *** STABILITY WARNING: DT_BASE ({mean_dt:.1f} yr) exceeds the "
                f"hillslope diffusion CFL limit ({dt_diffuse_limit:.1f} yr).  "
                f"Reduce DT_BASE or increase SYNTHETIC_DX to avoid instability. ***\n"
            )


def validate_config(config):
    """
    Pre-flight parameter validation.

    Checks every major Config setting for:
      - Values outside physically plausible ranges.
      - Combinations of settings that are mutually incompatible or that
        silently disable a process the user probably wants active.
      - Common unit errors (e.g. precipitation entered as mm/yr instead of m/yr).

    Outputs
    -------
    Prints [WARNING] or [ERROR] lines to stdout formatted as:

        *** [WARNING] <message> ***
        *** [ERROR]   <message> ***

    Errors indicate conditions that will almost certainly cause the model
    to crash or produce nonsensical results.  Warnings flag configurations
    that are unusual but may be intentional.

    A summary count is printed at the end.  The function never raises an
    exception — it leaves the decision to abort with the user.

    Parameters
    ----------
    config : Config
    """
    issues = []   # list of (level, message) tuples; level = 'WARNING' or 'ERROR'

    def W(msg): issues.append(('WARNING', msg))
    def E(msg): issues.append(('ERROR',   msg))

    # ------------------------------------------------------------------
    # TIMING
    # ------------------------------------------------------------------
    if config.DT_BASE <= 0:
        E(f"DT_BASE={config.DT_BASE} yr is non-positive — simulation cannot advance.")
    if config.SIM_LENGTH <= 0:
        E(f"SIM_LENGTH={config.SIM_LENGTH} yr is non-positive — nothing to simulate.")
    if config.DT_BASE > config.SIM_LENGTH:
        W(f"DT_BASE ({config.DT_BASE} yr) > SIM_LENGTH ({config.SIM_LENGTH} yr) — "
          f"only a single timestep will execute.")
    if config.RAINFALL_MODEL == 'constant' and config.DT_BASE > 500:
        W(f"DT_BASE={config.DT_BASE} yr is very large.  Fluvial CFL instabilities "
          f"are likely.  Consider enabling USE_ADAPTIVE_TIMESTEP or reducing DT_BASE.")
    if config.DT_MIN <= 0:
        E(f"DT_MIN={config.DT_MIN} yr must be > 0.")
    if config.DT_MIN > config.DT_BASE:
        W(f"DT_MIN ({config.DT_MIN} yr) > DT_BASE ({config.DT_BASE} yr) — "
          f"adaptive step is effectively disabled because the floor exceeds the ceiling.")
    if not (0.0 < config.CFL_SAFETY_FACTOR <= 1.0):
        W(f"CFL_SAFETY_FACTOR={config.CFL_SAFETY_FACTOR} is outside (0, 1]. "
          f"Values > 1 remove the safety margin; values ≤ 0 are invalid.")

    # ------------------------------------------------------------------
    # PRECIPITATION
    # ------------------------------------------------------------------
    if config.BASE_PRECIP_RATE <= 0:
        E(f"BASE_PRECIP_RATE={config.BASE_PRECIP_RATE} m/yr must be > 0 "
          f"for any fluvial erosion to occur.")
    if config.BASE_PRECIP_RATE > 20:
        W(f"BASE_PRECIP_RATE={config.BASE_PRECIP_RATE} m/yr is extremely high "
          f"(>20 m/yr = 20 000 mm/yr).  Check units — should be m/yr, not mm/yr.")
    if config.BASE_PRECIP_RATE < 0.1 and config.RAINFALL_MODEL == 'constant':
        W(f"BASE_PRECIP_RATE={config.BASE_PRECIP_RATE} m/yr is very low (<100 mm/yr). "
          f"Discharge will be minimal and fluvial erosion negligible.")
    if config.PRECIP_MIN <= 0:
        E(f"PRECIP_MIN={config.PRECIP_MIN} must be > 0 (spatial multiplier cannot be "
          f"zero or negative — that would eliminate precipitation at some nodes).")
    if config.PRECIP_MAX < config.PRECIP_MIN:
        W(f"PRECIP_MAX ({config.PRECIP_MAX}) < PRECIP_MIN ({config.PRECIP_MIN}) — "
          f"gradient is inverted relative to PRECIP_GRADIENT_ANGLE_DEG direction.")
    if config.PRECIP_TIMESERIES is not None:
        try:
            years = [t for t, _ in config.PRECIP_TIMESERIES]
            if years != sorted(years):
                E("PRECIP_TIMESERIES entries are not sorted by year — "
                  "step-function lookup will return wrong rates.")
            if years[0] > 0:
                W("PRECIP_TIMESERIES first entry year > 0; BASE_PRECIP_RATE is used "
                  "before that year, which may not be what you intended.")
        except (TypeError, ValueError):
            E("PRECIP_TIMESERIES is not a valid list of (year, rate) tuples.")

    # ------------------------------------------------------------------
    # UPLIFT
    # ------------------------------------------------------------------
    eff_uplift_max = config.BASE_UPLIFT_RATE * config.UPLIFT_MAX_MULT
    if config.BASE_UPLIFT_RATE < 0:
        E(f"BASE_UPLIFT_RATE={config.BASE_UPLIFT_RATE} m/yr is negative — "
          f"use positive values for uplift; subsidence is not directly supported.")
    if config.BASE_UPLIFT_RATE > 0.05:
        W(f"BASE_UPLIFT_RATE={config.BASE_UPLIFT_RATE} m/yr is extremely high "
          f"(>50 mm/yr).  Typical orogens: <15 mm/yr. Check units.")
    if config.UPLIFT_MIN_MULT < 0:
        E(f"UPLIFT_MIN_MULT={config.UPLIFT_MIN_MULT} is negative — "
          f"spatial multiplier must be ≥ 0.")
    if config.UPLIFT_MAX_MULT < config.UPLIFT_MIN_MULT:
        W(f"UPLIFT_MAX_MULT ({config.UPLIFT_MAX_MULT}) < UPLIFT_MIN_MULT "
          f"({config.UPLIFT_MIN_MULT}) — uplift gradient is inverted.")
    if config.UPLIFT_TIMESERIES is not None:
        try:
            years = [t for t, _ in config.UPLIFT_TIMESERIES]
            if years != sorted(years):
                E("UPLIFT_TIMESERIES entries are not sorted by year.")
        except (TypeError, ValueError):
            E("UPLIFT_TIMESERIES is not a valid list of (year, rate) tuples.")

    # ------------------------------------------------------------------
    # ERODIBILITIES
    # ------------------------------------------------------------------
    if config.FLUVIAL_K_BR <= 0:
        E(f"FLUVIAL_K_BR={config.FLUVIAL_K_BR} must be > 0 for bedrock incision.")
    if config.FLUVIAL_K_BR > 0.1:
        W(f"FLUVIAL_K_BR={config.FLUVIAL_K_BR} is very high (>0.1). "
          f"Typical range 1e-6–5e-3.  Expect extremely rapid incision or instability.")
    if config.FLUVIAL_K_SED <= 0 and config.FLUVIAL_MODEL == 'space':
        E(f"FLUVIAL_K_SED={config.FLUVIAL_K_SED} must be > 0 for SPACE sediment erosion.")
    if config.FLUVIAL_K_SED > 0.1 and config.FLUVIAL_MODEL == 'space':
        W(f"FLUVIAL_K_SED={config.FLUVIAL_K_SED} is very high.  "
          f"Sediment will be stripped almost instantly; model may be transport-unlimited.")
    # Warn when K_SED >> K_BR: SPACE erodes sediment far faster than bedrock
    # is produced, leading to permanently bare bedrock and effectively
    # converting SPACE into a pure stream-power eroder.
    if (config.FLUVIAL_MODEL == 'space' and config.FLUVIAL_K_BR > 0
            and config.FLUVIAL_K_SED / config.FLUVIAL_K_BR > 100):
        W(f"K_SED/K_BR = {config.FLUVIAL_K_SED/config.FLUVIAL_K_BR:.0f} >> 1.  "
          f"Sediment will always be stripped before bedrock is incised; you may "
          f"as well use FLUVIAL_MODEL='stream_power'.")
    if config.FLUVIAL_V_S < 0:
        E(f"FLUVIAL_V_S={config.FLUVIAL_V_S} m/yr must be ≥ 0 "
          f"(settling velocity cannot be negative).")
    if config.FLUVIAL_V_S > 1e6:
        W(f"FLUVIAL_V_S={config.FLUVIAL_V_S} m/yr is unrealistically high — "
          f"all sediment will deposit immediately within one cell.")

    # ----------------------------------------------------------------
    # SPACE exponents
    # ----------------------------------------------------------------
    if not (0.0 < config.SPACE_M <= 2.0):
        W(f"SPACE_M={config.SPACE_M} is outside the typical range (0, 2]. "
          f"Empirical consensus: m ≈ 0.5.")
    if not (0.0 < config.SPACE_N <= 3.0):
        W(f"SPACE_N={config.SPACE_N} is outside the typical range (0, 3]. "
          f"Empirical consensus: n ≈ 1.0.")
    if config.SPACE_N < 1.0 and config.FLUVIAL_MODEL == 'space':
        W(f"SPACE_N={config.SPACE_N} < 1.  CFL stability bound for n<1 scales "
          f"inversely with S, which can become very tight on gentle slopes.")
    theta = config.SPACE_M / config.SPACE_N if config.SPACE_N > 0 else 0
    if not (0.3 <= theta <= 0.7):
        W(f"Concavity index θ = m/n = {theta:.3f} is outside the typical range "
          f"[0.3, 0.7].  This will produce unusually convex or concave longitudinal profiles.")
    if not (0.0 <= config.SPACE_F_F <= 1.0):
        E(f"SPACE_F_F={config.SPACE_F_F} must be in [0, 1] "
          f"(fraction of fine sediment that bypasses the node).")

    # ------------------------------------------------------------------
    # HILLSLOPE
    # ------------------------------------------------------------------
    if config.HILLSLOPE_K <= 0:
        E(f"HILLSLOPE_K={config.HILLSLOPE_K} m²/yr must be > 0.")
    if config.HILLSLOPE_K > 1.0:
        W(f"HILLSLOPE_K={config.HILLSLOPE_K} m²/yr is very high (>1 m²/yr). "
          f"Typical range 1e-4–0.1 m²/yr.  Hillslopes will relax unrealistically fast.")
    if config.HILLSLOPE_SC is not None:
        if config.HILLSLOPE_SC <= 0:
            E(f"HILLSLOPE_SC={config.HILLSLOPE_SC} must be > 0 (critical slope in m/m).")
        if config.HILLSLOPE_SC > 2.0:
            W(f"HILLSLOPE_SC={config.HILLSLOPE_SC} m/m is very high "
              f"(equivalent to >63° slope).  Typical range 0.6–1.2 m/m.")
    # Non-linear diffuser with a very high K can violate CFL aggressively
    if config.HILLSLOPE_MODEL == 'taylor' and config.HILLSLOPE_K > 0.1:
        W(f"TaylorNonLinearDiffuser with HILLSLOPE_K={config.HILLSLOPE_K} m²/yr "
          f"may violate CFL on typical grids even with USE_ADAPTIVE_TIMESTEP=True.")

    # ------------------------------------------------------------------
    # SEDIMENT / WEATHERING
    # ------------------------------------------------------------------
    if config.SOIL_DEPTH_INIT < 0:
        E(f"SOIL_DEPTH_INIT={config.SOIL_DEPTH_INIT} m cannot be negative.")
    if config.WEATHERING_RATE < 0:
        E(f"WEATHERING_RATE={config.WEATHERING_RATE} m/yr cannot be negative.")
    if config.WEATHERING_RATE > 0.1:
        W(f"WEATHERING_RATE={config.WEATHERING_RATE} m/yr is very high (>100 mm/yr). "
          f"Bedrock will be converted to regolith faster than most erosion processes can "
          f"remove it; the landscape will rapidly become fully sediment-covered.")
    # If SPACE is active and weathering is zero, bare bedrock once exposed
    # can never re-acquire a sediment cover — warn only, not an error.
    if (config.FLUVIAL_MODEL == 'space' and config.WEATHERING_RATE == 0.0
            and config.SOIL_DEPTH_INIT == 0.0):
        W("FLUVIAL_MODEL='space' with WEATHERING_RATE=0 and SOIL_DEPTH_INIT=0: "
          "no sediment cover exists and none can form.  SPACE will behave identically "
          "to 'stream_power'.  Set WEATHERING_RATE > 0 or SOIL_DEPTH_INIT > 0 "
          "to allow sediment-cover dynamics.")

    # ------------------------------------------------------------------
    # LANDSLIDES
    # ------------------------------------------------------------------
    if config.USE_LANDSLIDES:
        if config.LS_ANGLE_FRICTION <= 0 or config.LS_ANGLE_FRICTION > 1.5:
            W(f"LS_ANGLE_FRICTION={config.LS_ANGLE_FRICTION} (tan φ) is outside "
              f"[0, 1.5].  Typical soils: 0.4–0.8.")
        if config.LS_COHESION_EFF < 0:
            E(f"LS_COHESION_EFF={config.LS_COHESION_EFF} Pa cannot be negative.")
        if config.LS_COHESION_EFF > 1e6:
            W(f"LS_COHESION_EFF={config.LS_COHESION_EFF} Pa is very high (>1 MPa). "
              f"Almost no failure will occur; effectively disabling the landslider.")
        if config.LS_RETURN_TIME <= 0:
            E(f"LS_RETURN_TIME={config.LS_RETURN_TIME} yr must be > 0.")
        if config.DT_LANDSLIDE_TRIGGER <= 0:
            E(f"DT_LANDSLIDE_TRIGGER={config.DT_LANDSLIDE_TRIGGER} yr must be > 0.")
        # PriorityFloodFlowRouter is REQUIRED; BedrockLandslider will crash
        # with standard FlowAccumulator.
        # (This is already enforced in initialize_model; warn here too for clarity.)

    # ------------------------------------------------------------------
    # OUTLET LOWERING
    # ------------------------------------------------------------------
    if config.USE_OUTLET_LOWERING:
        if config.OUTLET_LOWERING_RATE <= 0:
            W("USE_OUTLET_LOWERING=True but OUTLET_LOWERING_RATE <= 0 — "
              "no base-level drop will occur.")
        if config.OUTLET_LOWERING_DURATION <= 0:
            W("USE_OUTLET_LOWERING=True but OUTLET_LOWERING_DURATION <= 0 — "
              "lowering will never be applied.")
        total_drop = config.OUTLET_LOWERING_RATE * config.OUTLET_LOWERING_DURATION
        if total_drop > 5000:
            W(f"Outlet lowering total drop = {total_drop:.0f} m is very large. "
              f"This will produce extreme erosion rates or numerical overflow.")

    # ------------------------------------------------------------------
    # MAPPED FILE PATHS
    # ------------------------------------------------------------------
    if config.LITHOLOGY_MODE == 'mapped':
        if config.BEDROCK_ERODIBILITY_FILE is None:
            E("LITHOLOGY_MODE='mapped' but BEDROCK_ERODIBILITY_FILE is None — "
              "no file to load erodibility from.")
        elif not os.path.exists(config.BEDROCK_ERODIBILITY_FILE):
            E(f"BEDROCK_ERODIBILITY_FILE not found: {config.BEDROCK_ERODIBILITY_FILE}")
    if config.SEDIMENT_DEPTH_MODE == 'mapped':
        if config.SEDIMENT_DEPTH_FILE is None:
            E("SEDIMENT_DEPTH_MODE='mapped' but SEDIMENT_DEPTH_FILE is None.")
        elif not os.path.exists(config.SEDIMENT_DEPTH_FILE):
            E(f"SEDIMENT_DEPTH_FILE not found: {config.SEDIMENT_DEPTH_FILE}")
    if config.SEDIMENT_ERODIBILITY_MODE == 'mapped':
        if config.SEDIMENT_ERODIBILITY_FILE is None:
            E("SEDIMENT_ERODIBILITY_MODE='mapped' but SEDIMENT_ERODIBILITY_FILE is None.")
        elif not os.path.exists(config.SEDIMENT_ERODIBILITY_FILE):
            E(f"SEDIMENT_ERODIBILITY_FILE not found: {config.SEDIMENT_ERODIBILITY_FILE}")

    # ------------------------------------------------------------------
    # CROSS-PARAMETER COMPATIBILITY
    # ------------------------------------------------------------------
    # Stochastic rainfall with very long DT_BASE for the constant branch is
    # not an issue (DT_BASE is ignored in stochastic mode), but warn if the
    # user has DT_BASE >> mean storm cycle, suggesting they may be confused.
    if (config.RAINFALL_MODEL == 'stochastic' and config.DT_BASE < 0.1):
        W(f"RAINFALL_MODEL='stochastic': DT_BASE ({config.DT_BASE} yr) is very small "
          f"relative to the mean storm cycle (~1.5 yr).  DT_BASE is not used in "
          f"stochastic mode; this setting has no effect.")
    # Adaptive timestep with stochastic mode: the adaptive CFL is applied to
    # dt_step (storm window), which is drawn from an Exponential distribution
    # with mean 0.5 yr — extremely short steps are likely.
    if (config.RAINFALL_MODEL == 'stochastic'
            and getattr(config, 'USE_ADAPTIVE_TIMESTEP', False)
            and config.DT_MIN > 0.5):
        W(f"RAINFALL_MODEL='stochastic' + USE_ADAPTIVE_TIMESTEP=True with "
          f"DT_MIN={config.DT_MIN} yr.  Mean storm duration is ~0.5 yr; "
          f"DT_MIN is larger than most storm windows and will be the binding "
          f"constraint — the adaptive CFL will rarely have effect.")
    # SPACE reads surface_water__discharge (populated by PriorityFloodFlowRouter
    # from water__unit_flux_in) automatically — no extra kwarg required.
    if (config.FLUVIAL_MODEL == 'space'
            and config.BASE_PRECIP_RATE == 1.0
            and config.PRECIP_MIN == config.PRECIP_MAX):
        W("Precipitation field is spatially uniform (PRECIP_MIN == PRECIP_MAX). "
          "Spatial orographic gradients will have no effect on erosion patterns. "
          "Consider setting PRECIP_MIN < PRECIP_MAX for realistic climate forcing.")
    # Very high erosion potential relative to uplift → landscape collapses
    max_erosion_potential = (config.FLUVIAL_K_BR
                             * (config.BASE_PRECIP_RATE * config.PRECIP_MAX) ** config.SPACE_M)
    max_uplift = config.BASE_UPLIFT_RATE * config.UPLIFT_MAX_MULT
    if max_uplift > 0 and max_erosion_potential / max_uplift > 1e5:
        W(f"Peak erosion potential / peak uplift rate ≈ {max_erosion_potential/max_uplift:.1e}. "
          f"Erosion capacity vastly exceeds uplift — the landscape will rapidly "
          f"incise to base level and remain flat.  Check K_BR and BASE_UPLIFT_RATE.")

    # ------------------------------------------------------------------
    # CHANNEL FOOTPRINT
    # ------------------------------------------------------------------
    if getattr(config, 'USE_CHANNEL_FOOTPRINT', False):
        ch_file = getattr(config, 'CHANNEL_FOOTPRINT_FILE', None)
        if ch_file is not None and not os.path.exists(ch_file):
            E(f"CHANNEL_FOOTPRINT_FILE not found: {ch_file}")
        depth = getattr(config, 'CHANNEL_INCISION_DEPTH', 0.1)
        if depth < 0:
            E(f"CHANNEL_INCISION_DEPTH={depth} m is negative.")
        if depth == 0:
            W("CHANNEL_INCISION_DEPTH=0 — channel will not be incised below "
              "neighbours; flow may not preferentially enter the channel.")

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    n_warn  = sum(1 for lvl, _ in issues if lvl == 'WARNING')
    n_error = sum(1 for lvl, _ in issues if lvl == 'ERROR')

    print("\n" + "=" * 70)
    print("  PARAMETER VALIDATION")
    print("=" * 70)
    if not issues:
        print("  All parameters passed validation — no issues found.")
    else:
        for lvl, msg in issues:
            tag = f"[{lvl}]"
            # Wrap long messages at 68 chars for readability
            prefix = f"  *** {tag:<9} "
            pad    = " " * len(prefix)
            words  = msg.split()
            line, lines = "", []
            for w in words:
                if len(line) + len(w) + 1 > 68:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            lines.append(line)
            print(prefix + lines[0])
            for extra in lines[1:]:
                print(pad + extra)
            print()
    print(f"  Summary: {n_error} error(s), {n_warn} warning(s)")
    if n_error > 0:
        print("  *** One or more ERRORS detected above — the model may crash or "
              "produce nonsensical results. ***")
    print("=" * 70 + "\n")


def _report_outlet_fluxes(grid, config, current_year, dt):
    """
    Compute and print flow discharge and sediment flux leaving the open
    (FIXED_VALUE) boundary nodes every OUTLET_FLUX_REPORT_INTERVAL years.

    METHODOLOGY
    -----------
    Open boundary nodes in Landlab act as perfect sinks — all water and
    sediment arriving there leaves the domain in that timestep.  The total
    outgoing discharge is the sum of surface_water__discharge at all
    FIXED_VALUE nodes (if available), or drainage_area × mean precipitation
    as a proxy when discharge is absent.  Sediment flux is estimated from
    the SPACE-derived erosion rate at the outlet cells.

    REALISM CHECKS
    --------------
    The function compares the computed outlet discharge against two
    independent estimates:

    1. Area-based Budyko estimate
       Q_est = A_total × P_mean × runoff_fraction
       where runoff_fraction ≈ 0.5 (humid tropical watershed).
       Valid order-of-magnitude check; should be within 1–2× of modelled Q.

    2. Sediment yield check
       Specific sediment yield (t km⁻² yr⁻¹) is compared against published
       ranges for Andean rivers (100–10 000 t km⁻² yr⁻¹; Milliman & Syvitski
       1992).  Very high or very low values flag potential parameter issues.

    Both flagging thresholds are deliberately generous (order-of-magnitude)
    because individual timestep variability is large — the intent is to catch
    completely unphysical conditions, not to enforce tight budgets.

    Parameters
    ----------
    grid         : RasterModelGrid
    config       : Config
    current_year : float  Current simulation time (years).
    dt           : float  Duration of the most recent timestep (years).
    """
    oflux_interval = getattr(config, 'OUTLET_FLUX_REPORT_INTERVAL', None)
    if not oflux_interval:
        return

    # Identify open boundary (outlet) nodes
    open_bnds = np.where(
        grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE
    )[0]
    if len(open_bnds) == 0:
        print("  [Outlet Monitor] No open boundary nodes found — cannot compute flux.",
              flush=True)
        return

    dx   = grid.dx
    area_per_cell = dx * dx   # m²

    # ------------------------------------------------------------------
    # 1. DISCHARGE at outlet nodes
    # ------------------------------------------------------------------
    # PriorityFloodFlowRouter / FlowAccumulator set surface_water__discharge
    # = 0 at FIXED_VALUE boundary nodes (water exits there, doesn't
    # accumulate), and drainage_area at those nodes = 1 cell (boundary nodes
    # don't accumulate upstream area either in all Landlab builds).
    #
    # Strategy cascade — first non-zero result wins:
    #   A. Sum discharge at CORE nodes whose flow__receiver_node == outlet.
    #   B. Peak discharge among all core nodes (the node with maximum
    #      drainage area carries the full watershed flux just before the
    #      outlet; safe even when Method A fails due to routing quirks).
    #   C. max(drainage_area[cores]) × mean precipitation — exact
    #      water-balance proxy when the discharge field itself is zero.
    _q_method  = "none"
    Q_out_m3yr = float('nan')
    cores = grid.core_nodes

    discharge_field = grid.at_node.get('surface_water__discharge', None)
    da_field        = grid.at_node.get('drainage_area', None)

    # --- Flow-routing health snapshot (used in diagnostics below) ---
    _max_da_core  = float(np.max(da_field[cores]))        if da_field        is not None else 0.0
    _max_Q_core   = float(np.max(discharge_field[cores])) if discharge_field is not None else 0.0
    _max_da_node  = int(cores[np.argmax(da_field[cores])]) if da_field       is not None else -1

    # --- Strategy A: core nodes whose receiver IS the outlet ---
    if discharge_field is not None and 'flow__receiver_node' in grid.at_node:
        receiver   = grid.at_node['flow__receiver_node']
        Q_donors   = 0.0
        for outlet_id in open_bnds:
            donors   = np.where(receiver == outlet_id)[0]
            Q_donors += float(np.sum(discharge_field[donors]))
        if Q_donors > 0.0:
            Q_out_m3yr = Q_donors
            _q_method  = "Q[donors→outlet]"

    # --- Strategy B: peak discharge in the watershed ---
    # The node with maximum discharge IS the penultimate node before the
    # outlet; its value equals the full accumulated catchment discharge.
    if (np.isnan(Q_out_m3yr) or Q_out_m3yr == 0.0) and discharge_field is not None:
        if _max_Q_core > 0.0:
            Q_out_m3yr = _max_Q_core
            _q_method  = "Q[peak-discharge node]"

    # --- Strategy C: max drainage_area × mean precipitation ---
    # NOTE: uses max(DA[cores]), NOT DA[outlet], because boundary nodes
    # carry only their own cell area in Landlab's flow accumulator.
    if (np.isnan(Q_out_m3yr) or Q_out_m3yr == 0.0) and da_field is not None:
        mean_flux  = float(np.mean(grid.at_node['water__unit_flux_in'][cores]))
        Q_out_m3yr = _max_da_core * mean_flux   # m² × m/yr = m³/yr
        _q_method  = f"A×P [max-DA={_max_da_core*1e-6:.1f}km²]"

    # Independent area-based estimate for sanity
    n_core    = len(cores)
    A_total   = n_core * area_per_cell          # m²
    A_total_km2 = A_total * 1e-6
    mean_precip_basin = float(np.mean(grid.at_node['water__unit_flux_in'][cores]))
    Q_budyko  = A_total * mean_precip_basin * 0.5  # m³/yr, assuming 50% runoff

    # Flow-routing health flag (routing broken if max DA << watershed area)
    _routing_ok = (_max_da_core > area_per_cell * 100)

    # ------------------------------------------------------------------
    # 2. SEDIMENT FLUX at outlet nodes
    # ------------------------------------------------------------------
    # Estimate volumetric sediment outflux from the change in elevation of
    # core nodes adjacent to outlets (proxy for sediment leaving domain).
    # A simpler and always-available proxy: use soil__depth flux if SPACE
    # wrote it, otherwise estimate from bedrock erosion at outlet neighbours.
    sed_m3yr = float('nan')
    if 'soil__depth' in grid.at_node and 'bedrock__elevation' in grid.at_node:
        # Drainage-area-weighted approach: sediment yield from total core
        # volume change over dt, scaled to annual rate.
        topo = grid.at_node['topographic__elevation']
        br   = grid.at_node['bedrock__elevation']
        # Volume of bedrock eroded this step across all core nodes (m³/yr)
        # Use drainage area at outlet as fraction of total contributing area,
        # since we only see one slice of the sediment budget at the boundary.
        if 'drainage_area' in grid.at_node:
            da_out = float(np.max(grid.at_node['drainage_area'][open_bnds]))
            da_frac = min(da_out / max(A_total, 1.0), 1.0)
        else:
            da_frac = 1.0
        # Mean elevation change of core nodes per year as erosion proxy
        # (computed from the difference between topo and bedrock = soil depth)
        soil_mean = float(np.mean(
            grid.at_node['soil__depth'][grid.core_nodes]
        ))
        # Volumetric sediment yield: K_sed proxy, not exact
        # Better approximation: use sediment budget from SPACE K values
        mean_k_sed = config.FLUVIAL_K_SED
        if mean_precip_basin > 0:
            sed_m3yr = (mean_k_sed
                        * (Q_budyko * da_frac) ** config.SPACE_M
                        * A_total * da_frac
                        * 0.01)   # dimensional scaling factor

    # ------------------------------------------------------------------
    # 3. PRINT REPORT
    # ------------------------------------------------------------------
    print(f"\n  {'─'*64}", flush=True)
    print(f"  Outlet Flux Report @ yr {current_year:,.0f}", flush=True)
    print(f"  {'─'*64}", flush=True)
    print(f"  Outlet nodes           : {len(open_bnds):>8d}", flush=True)
    print(f"  Catchment area         : {A_total_km2:>10,.1f} km²", flush=True)
    print(f"  Max DA (core nodes)    : {_max_da_core*1e-6:>10.1f} km²  "
          f"{'OK' if _routing_ok else '*** ROUTING BROKEN? — max DA is only 1 cell ***'}",
          flush=True)
    if da_field is not None and 'flow__receiver_node' in grid.at_node:
        _recv    = grid.at_node['flow__receiver_node']
        _recv_of_max = int(_recv[_max_da_node]) if _max_da_node >= 0 else -1
        _n_to_outlet = sum(int(np.sum(_recv == oid)) for oid in open_bnds)
        print(f"  Max-DA node receiver   : {_recv_of_max:>10d}  "
              f"({'outlet!' if _recv_of_max in open_bnds else 'NOT outlet'})",
              flush=True)
        print(f"  Cores draining direct→ : {_n_to_outlet:>10d}  (direct donors to outlet)",
              flush=True)

    if not np.isnan(Q_out_m3yr):
        Q_out_Ls = Q_out_m3yr / (365.25 * 86400) * 1000   # L/s
        Q_out_mm = Q_out_m3yr / A_total * 1000             # mm/yr runoff
        print(f"  Outlet discharge       : {Q_out_m3yr:>12,.0f} m³/yr  [{_q_method}]",
              flush=True)
        print(f"                         : {Q_out_Ls:>12,.1f} L/s", flush=True)
        print(f"  Runoff depth           : {Q_out_mm:>10,.1f} mm/yr", flush=True)
        print(f"  Area-based estimate    : {Q_budyko:>12,.0f} m³/yr  (50% runoff)",
              flush=True)

        # Realism check — outlet Q should be within ~10× of Budyko estimate
        if Q_budyko > 0:
            ratio = Q_out_m3yr / Q_budyko
            if ratio < 0.05:
                print(f"  *** [WARNING] Outlet discharge is only {ratio:.2%} of the "
                      f"area-based estimate.  Very little water is leaving the domain — "
                      f"check boundary conditions and precipitation field. ***",
                      flush=True)
            elif ratio > 20:
                print(f"  *** [WARNING] Outlet discharge is {ratio:.1f}× the area-based "
                      f"estimate.  Implausibly high outflow — possible flow-routing error "
                      f"or open boundary at an unexpected location. ***",
                      flush=True)
        # Check for zero outflow
        if Q_out_m3yr == 0:
            print("  *** [WARNING] Outlet discharge = 0.  No water is leaving the "
                  "domain.  Verify that at least one FIXED_VALUE boundary node exists "
                  "and is connected to the flow network. ***",
                  flush=True)
    else:
        print("  Outlet discharge       :    (discharge field unavailable)", flush=True)

    # Specific sediment yield check
    if 'soil__depth' in grid.at_node:
        sd_mean = float(np.mean(grid.at_node['soil__depth'][cores]))
        sd_min  = float(np.min( grid.at_node['soil__depth'][cores]))
        print(f"  Mean soil depth (core) : {sd_mean:>10.3f} m", flush=True)
        print(f"  Min  soil depth (core) : {sd_min:>10.4f} m", flush=True)
        if sd_mean < 1e-4:
            print("  *** [WARNING] Mean soil depth is nearly zero across the domain. "
                  "SPACE is operating in almost pure bedrock-incision mode. "
                  "If sediment cover was intended, increase SOIL_DEPTH_INIT "
                  "or WEATHERING_RATE. ***",
                  flush=True)

    # Elevation bounds sanity check at outlets
    topo = grid.at_node['topographic__elevation']
    z_out_min = float(np.min(topo[open_bnds]))
    z_out_max = float(np.max(topo[open_bnds]))
    print(f"  Outlet elevation range : {z_out_min:>8.1f} – {z_out_max:.1f} m",
          flush=True)
    if z_out_min < -500:
        print(f"  *** [WARNING] Outlet node elevation = {z_out_min:.1f} m.  This is "
              f"deeply negative and suggests a NoData cell (-9999) has been "
              f"incorrectly left open as a boundary.  Check BC order. ***",
              flush=True)
    print(f"  {'─'*64}\n", flush=True)


def _compute_cfl_dt(grid, config, dt_max):
    """
    Compute the CFL-constrained maximum stable timestep for the current
    model state, then apply a conservative safety factor.

    PHYSICS
    -------
    Two explicit numerical schemes impose CFL-type stability constraints:

    1. Hillslope diffusion (linear / non-linear)
       ─────────────────────────────────────────
       For a 2-D explicit diffusion scheme with diffusivity K_D and grid
       spacing dx the classic von-Neumann stability criterion is:

           dt < dx² / (4 K_D)   (2-D)  →  conservatively: dx² / (2 K_D)

       This limit is *fixed* (depends only on dx and K_D, not on the current
       topography) and is computed once from Config.

    2. Fluvial incision / SPACE (advection-like wave propagation)
       ──────────────────────────────────────────────────────────
       The stream-power incision model is mathematically equivalent to a
       non-linear advection equation.  The knickpoint propagation celerity is:

           c = K_br · Q^m · n · S^(n−1)

       For n = 1.0 (the default, typical of most field-calibrated SPMs):
           c = K_br · Q^m   (slope-independent)

       The Courant–Friedrichs–Lewy condition then requires:

           dt < dx / c_max

       where c_max is evaluated over all core nodes with Q > 0 and S > 0
       using the drainage area and slope fields populated by the most recent
       call to the flow router.

    SAFETY FACTOR
    -------------
    The raw CFL limit is multiplied by config.CFL_SAFETY_FACTOR (default 0.8)
    to provide a 20 % safety margin.  The resulting dt is bounded:

        DT_MIN ≤ dt_adaptive ≤ dt_max

    If neither constraint can be evaluated (e.g. model has not yet been
    flow-routed), dt_max is returned unchanged.

    Parameters
    ----------
    grid    : RasterModelGrid  Grid after the most recent flow-routing call.
    config  : Config           Supplies K_D, K_BR, m, n, safety parameters.
    dt_max  : float            Upper bound (caller's DT_BASE or remaining sim
                               time); the adaptive dt never exceeds this.

    Returns
    -------
    float : CFL-safe adaptive timestep (years), bounded by [DT_MIN, dt_max].
    """
    dx = grid.dx
    cores = grid.core_nodes
    dt_limits = []

    # ------------------------------------------------------------------
    # 1. Hillslope diffusion CFL
    # ------------------------------------------------------------------
    if config.HILLSLOPE_K > 0:
        # Use dx² / (2 K_D) — slightly more conservative than the 4 K_D
        # criterion because TaylorNonLinearDiffuser amplifies effective K
        # near the critical slope.
        dt_diff = (dx * dx) / (2.0 * config.HILLSLOPE_K)
        dt_limits.append(dt_diff)

    # ------------------------------------------------------------------
    # 2. Fluvial CFL (knickpoint celerity)
    # ------------------------------------------------------------------
    # Requires drainage_area/discharge and slope — available after Phase F.
    if ('topographic__steepest_slope' in grid.at_node and
            'bedrock__erodibility' in grid.at_node):

        # Prefer surface_water__discharge (already incorporates precipitation
        # weighting); fall back to drainage_area × mean precipitation if absent.
        if 'surface_water__discharge' in grid.at_node:
            Q = grid.at_node['surface_water__discharge'][cores]
        elif 'drainage_area' in grid.at_node:
            mean_precip = float(np.mean(grid.at_node['water__unit_flux_in']))
            Q = grid.at_node['drainage_area'][cores] * mean_precip
        else:
            Q = None

        if Q is not None:
            S   = grid.at_node['topographic__steepest_slope'][cores]
            K   = grid.at_node['bedrock__erodibility'][cores]
            m   = config.SPACE_M
            n   = config.SPACE_N

            # Only evaluate nodes with non-trivial flow and gradient.
            # Very flat nodes (S ≈ 0) have near-zero celerity and don't
            # constrain the timestep meaningfully.
            valid = (Q > 0.0) & (S > 1.0e-8) & (K > 0.0)
            if np.any(valid):
                Q_v = Q[valid]
                S_v = S[valid]
                K_v = K[valid]

                # Knickpoint celerity  c = K · Q^m · n · S^(n−1)
                # For n == 1 the S term is 1; guarded against n < 1.
                if n == 1.0:
                    c = K_v * (Q_v ** m)
                elif n > 1.0:
                    c = K_v * (Q_v ** m) * n * (S_v ** (n - 1.0))
                else:
                    # n < 1: celerity decreases with slope — use n=1 as
                    # conservative upper bound (over-estimates celerity).
                    c = K_v * (Q_v ** m)

                c_max = float(np.max(c))
                if c_max > 0.0:
                    dt_limits.append(dx / c_max)

    # ------------------------------------------------------------------
    # Apply safety factor and bounds
    # ------------------------------------------------------------------
    if not dt_limits:
        # Cannot compute any CFL limit yet (e.g. first step, no Q field).
        return dt_max

    dt_cfl  = float(min(dt_limits))
    dt_safe = config.CFL_SAFETY_FACTOR * dt_cfl
    dt_min  = getattr(config, 'DT_MIN', 0.1)
    return float(np.clip(dt_safe, dt_min, dt_max))




# ============================================================================
# INITIALIZATION & SETUP
# ============================================================================

def initialize_model(config):
    """
    Builds the Landlab grid, registers all required field arrays, and
    instantiates all active geomorphic components.

    This function is called once at the start of each simulation run.  It
    enforces strict mathematical consistency between bedrock__elevation,
    soil__depth, and topographic__elevation to satisfy internal Landlab
    component preconditions (particularly BedrockLandslider, which performs
    an exact equality check).

    WORKFLOW OVERVIEW
    -----------------
    Step 1  Grid construction   — observed DEM or synthetic flat grid
    Step 2  Field initialisation — elevation, sediment, erodibility arrays
    Step 3  Climate / tectonic fields — precipitation and uplift spatial fields
    Step 4  Component instantiation — flow router, diffuser, fluvial, landslider

    MAPPED FIELDS
    -------------
    When config.LITHOLOGY_MODE = "mapped", bedrock erodibility is loaded from
    config.BEDROCK_ERODIBILITY_FILE (GeoTIFF aligned to the model grid).
    When config.SEDIMENT_DEPTH_MODE = "mapped", initial soil depth is read from
    config.SEDIMENT_DEPTH_FILE.
    When config.SEDIMENT_ERODIBILITY_MODE = "mapped", sediment erodibility (K_sed
    in SPACE) is read from config.SEDIMENT_ERODIBILITY_FILE and stored as the
    field 'sediment__erodibility' which SPACE will consume.

    WELL-POSEDNESS REQUIREMENT
    --------------------------
    The grid must have at least one open boundary node (water outlet) for the
    flow router to route drainage to.  All interior open-boundary nodes are
    treated as potential outlets; the lowest-elevation open-boundary node
    effectively acts as the watershed outlet / base level.

    Parameters
    ----------
    config : Config  Configuration object (see Config class documentation).

    Returns
    -------
    grid                  : RasterModelGrid  Ready-to-run Landlab grid.
    components            : dict             Keyed component instances.
    precip_shape : numpy.ndarray  Per-node dimensionless precipitation multiplier
                                  in the range [PRECIP_MIN, PRECIP_MAX].
                                  Multiply by get_effective_precip_rate(t) to
                                  obtain the physical flux field (m/yr).
    uplift_shape : numpy.ndarray  Per-node dimensionless uplift multiplier
                                  in the range [UPLIFT_MIN_MULT, UPLIFT_MAX_MULT].
                                  Multiply by get_effective_uplift_rate(t) to
                                  obtain the physical rate field (m/yr).
    """
    print("\n--- Model Initialization ---")
    t0_init = time.time()

    # ------------------------------------------------------------------
    # STEP 1: GRID CONSTRUCTION
    # ------------------------------------------------------------------
    print("[1/4] Setting up grid...")
    t_step = time.time()

    if config.LANDSCAPE_MODE == "observed":
        # ---------------------------------------------------------------
        # Observed-landscape branch: load a real DEM.
        # The first run is slow (sink-filling); subsequent runs use a
        # pre-processed NetCDF cache for instant startup.
        # ---------------------------------------------------------------
        filled_dem_path = config.DEM_FILE.replace('.tif', '_filled.nc')

        if os.path.exists(filled_dem_path):
            print(f"      Loading pre-filled grid from cache: {filled_dem_path}")
            grid = read_netcdf(filled_dem_path)
            print(f"      Grid shape: {grid.shape}, Cell spacing (dx): {grid.dx:.2f} m")
            print(f"      Grid loaded in {time.time() - t_step:.2f} seconds.")
            # Apply channel footprint conditioning (if enabled).
            # This modifies the in-memory topo to ensure monotonic downstream
            # slope and sets config.OUTLET_NODE_ROWCOL to the channel outlet.
            _apply_channel_footprint(grid, config)
            # Re-apply boundary conditions after every cache load.
            # Use set_watershed_boundary_condition which internally:
            #   1. Closes all four grid edges
            #   2. Closes all NoData (-9999) nodes
            #   3. Opens the lowest-elevation node adjacent to any closed node
            #      as a FIXED_VALUE outlet — AND updates Landlab's internal
            #      link-status caches so the flow router recognises the outlet.
            _topo_bc = grid.at_node['topographic__elevation']
            _apply_watershed_bc(grid, _topo_bc, config)

        else:
            print(f"      No cached grid found. Processing raw DEM...")
            grid = load_dem_to_grid(config.DEM_FILE)

            if 'topographic__elevation' not in grid.at_node:
                existing_name = list(grid.at_node.keys())[0]
                grid.add_field('topographic__elevation',
                               grid.at_node[existing_name], at='node', clobber=True)

            # ----------------------------------------------------------
            # Channel footprint conditioning (if enabled).
            # Must run BEFORE sink filling so the conditioned channel is
            # not re-filled by SinkFillerBarnes.
            # ----------------------------------------------------------
            _apply_channel_footprint(grid, config)

            # ----------------------------------------------------------
            # Boundary conditions — let Landlab's watershed helper do this
            # in one correctly-integrated call.  set_watershed_boundary_condition
            # closes all four grid edges, marks NoData nodes CLOSED, finds the
            # lowest-elevation non-NoData node adjacent to any closed cell and
            # sets it to FIXED_VALUE, then updates internal link-status caches
            # so that PriorityFloodFlowRouter recognises the outlet.
            # ----------------------------------------------------------
            _apply_watershed_bc(grid, grid.at_node['topographic__elevation'], config)

            # ----------------------------------------------------------
            # Sink filling with Priority-Flood (Barnes et al. 2014).
            # fill_flat=False adds an epsilon gradient to flat-floored
            # depressions so drainage still accumulates through them.
            # This is critical for hydrologically connected flow networks.
            # ----------------------------------------------------------
            print("      Filling hydrological sinks... (First-time setup)")
            t_fill = time.time()
            sf = SinkFillerBarnes(grid, method='D8', fill_flat=False)
            sf.run_one_step()
            print(f"      Pits filled in {time.time() - t_fill:.2f} seconds.")

            # Cache the processed grid so future runs skip this step
            print(f"      Saving processed grid to cache: {filled_dem_path}")
            write_netcdf(filled_dem_path, grid)

    elif config.LANDSCAPE_MODE == "synthetic":
        # ---------------------------------------------------------------
        # Synthetic-landscape branch: construct an idealised tilted grid
        # for controlled experiments and benchmarking.
        #
        # The initial surface is defined by:
        #   z = SYNTHETIC_RELIEF * (1 - normalised_projection)
        # where the projection is taken along SYNTHETIC_GRADIENT_ANGLE_DEG.
        # This places elevation = SYNTHETIC_RELIEF at the "uphill" end and
        # elevation = 0 at the "downhill" (outlet) end.
        #
        # BOUNDARY CONDITION NOTE:
        #   Only the right grid edge is left open by default (the outlet).
        #   Ensure SYNTHETIC_GRADIENT_ANGLE_DEG places the LOW end of the
        #   surface at the right edge (use 270° for a west-rising surface
        #   that drains eastward).  If you change the outlet edge, update
        #   set_closed_boundaries_at_grid_edges accordingly.
        # ---------------------------------------------------------------
        rows, cols = config.SYNTHETIC_GRID_SHAPE
        grid = RasterModelGrid((rows, cols), xy_spacing=config.SYNTHETIC_DX)

        # Build initial elevation using the directional gradient helper.
        # The gradient runs from 0 (outlet / low end) to SYNTHETIC_RELIEF
        # (headwater / high end) along the specified compass bearing.
        # We want HIGH values at the uphill end, so we reverse min/max:
        #   create_gradient gives max at the bearing direction, min opposite.
        #   Therefore pass (RELIEF, 0) so the "high" end is at ANGLE_DEG.
        z = create_gradient(
            grid,
            min_val=0.0,
            max_val=config.SYNTHETIC_RELIEF,
            angle_deg=config.SYNTHETIC_GRADIENT_ANGLE_DEG
        )

        # Add small random noise (~1 mm) to break exact flow-direction ties
        # on the perfectly planar surface.  Without this, D8 routing produces
        # unrealistic parallel flow lines instead of a branching network.
        rng = np.random.default_rng(config.RANDOM_SEED)
        z += rng.uniform(0.0, 0.001, size=z.shape)
        grid.add_field('topographic__elevation', z, at='node')
        grid.set_closed_boundaries_at_grid_edges(
            right_is_closed=False, top_is_closed=True,
            left_is_closed=True, bottom_is_closed=True
        )

        # Compute the implied surface slope for the user's information
        domain_length_m = max(
            grid.node_x.max() - grid.node_x.min(),
            grid.node_y.max() - grid.node_y.min()
        )
        implied_slope = (config.SYNTHETIC_RELIEF / domain_length_m
                         if domain_length_m > 0 else 0.0)
        implied_angle_deg = np.degrees(np.arctan(implied_slope))
        print(
            f"      Synthetic grid: {rows}\u00d7{cols} nodes | "
            f"dx={config.SYNTHETIC_DX} m | "
            f"relief={config.SYNTHETIC_RELIEF:.1f} m | "
            f"gradient angle={config.SYNTHETIC_GRADIENT_ANGLE_DEG:.0f}\u00b0 CW from N"
        )
        print(
            f"      Implied surface slope: {implied_slope:.4f} m/m "
            f"({implied_angle_deg:.2f}\u00b0 from horizontal)"
        )
    else:
        raise ValueError(f"Unknown LANDSCAPE_MODE: '{config.LANDSCAPE_MODE}'. "
                         "Choose 'observed' or 'synthetic'.")

    print(f"      Grid setup complete in {time.time() - t_step:.2f} s | "
          f"{grid.number_of_nodes:,} nodes")

    # ------------------------------------------------------------------
    # STEP 2: SURFACE & SUBSURFACE FIELD INITIALISATION
    # ------------------------------------------------------------------
    print("[2/4] Initialising surface and subsurface fields...")
    t_step = time.time()

    topo = grid.at_node['topographic__elevation']

    # --- NaN guard ---
    # Some GeoTIFFs encode NoData as NaN rather than a numeric sentinel.
    # Replace them with the explicit nodata value so boundary conditions
    # applied above remain consistent.
    if np.any(np.isnan(topo)):
        topo[np.isnan(topo)] = -9999.0
        grid.set_nodata_nodes_to_closed(topo, -9999.0)

    # --- Sediment (soil) depth field ---
    # ASSUMPTION: soil__depth represents all mobile regolith / alluvium
    # overlying bedrock.  Bedrock is exposed where soil__depth ≈ 0.
    if config.SEDIMENT_DEPTH_MODE == "mapped" and config.SEDIMENT_DEPTH_FILE is not None:
        sd_values = load_raster_to_node_array(
            config.SEDIMENT_DEPTH_FILE, grid,
            field_name='soil depth', nodata_fill=config.SOIL_DEPTH_INIT
        )
        # Clamp to non-negative values (physical requirement)
        sd_values = np.maximum(sd_values, 0.0)
        soil_depth = grid.add_field('soil__depth', sd_values, at='node', clobber=True)
        print(f"      Mapped soil depth: min={sd_values.min():.2f} m, "
              f"max={sd_values.max():.2f} m")
    else:
        # Uniform initial regolith cover
        soil_depth = grid.add_full(
            'soil__depth', config.SOIL_DEPTH_INIT, at='node', clobber=True
        )
    # Ensure NoData nodes carry zero soil depth
    closed = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    soil_depth[closed] = 0.0

    # --- Bedrock elevation field ---
    # Bedrock elevation is derived by subtracting soil depth from the
    # surface.  This is then locked in as the "parent" field; topographic
    # elevation is recalculated as bedrock + soil to enforce strict
    # mathematical consistency (required by BedrockLandslider).
    bedrock = grid.add_zeros('bedrock__elevation', at='node', clobber=True)
    bedrock[:] = topo[:] - soil_depth[:]
    # Overwrite topo to guarantee exact sum (avoids floating-point drift)
    topo[:] = bedrock[:] + soil_depth[:]

    # --- Bedrock erodibility field ---
    # K_BR controls how rapidly bedrock is abraded by channelised flow.
    # Higher values produce faster knickpoint migration.
    if config.LITHOLOGY_MODE == "mapped" and config.BEDROCK_ERODIBILITY_FILE is not None:
        k_br_values = load_raster_to_node_array(
            config.BEDROCK_ERODIBILITY_FILE, grid,
            field_name='bedrock erodibility', nodata_fill=config.FLUVIAL_K_BR
        )
        grid.add_field('bedrock__erodibility', k_br_values, at='node', clobber=True)
        print(f"      Mapped bedrock erodibility: "
              f"min={k_br_values.min():.2e}, max={k_br_values.max():.2e}")
    elif config.LITHOLOGY_MODE == "gradient":
        # Linear diagonal gradient as a simple two-unit proxy.
        # Erodibility varies from half to double the baseline value,
        # oriented eastward (90°) by default — edit angle_deg to change.
        k_br_grad = create_gradient(
            grid, config.FLUVIAL_K_BR / 2, config.FLUVIAL_K_BR * 2,
            angle_deg=90.0
        )
        grid.add_field('bedrock__erodibility', k_br_grad, at='node', clobber=True)
        print(f"      Gradient bedrock erodibility: "
              f"{config.FLUVIAL_K_BR/2:.2e} – {config.FLUVIAL_K_BR*2:.2e}")
    else:  # "uniform" (default)
        grid.add_field(
            'bedrock__erodibility',
            np.full(grid.number_of_nodes, config.FLUVIAL_K_BR),
            at='node', clobber=True
        )

    # --- Sediment erodibility field ---
    # K_SED in SPACE can be spatially variable to represent contrasts
    # between, e.g., consolidated glacial till and unconsolidated floodplain.
    if (config.SEDIMENT_ERODIBILITY_MODE == "mapped"
            and config.SEDIMENT_ERODIBILITY_FILE is not None):
        k_sed_values = load_raster_to_node_array(
            config.SEDIMENT_ERODIBILITY_FILE, grid,
            field_name='sediment erodibility', nodata_fill=config.FLUVIAL_K_SED
        )
        # Store as a Landlab field so SPACE can reference it by field name
        grid.add_field('sediment__erodibility', k_sed_values, at='node', clobber=True)
        print(f"      Mapped sediment erodibility: "
              f"min={k_sed_values.min():.2e}, max={k_sed_values.max():.2e}")
        k_sed_arg = 'sediment__erodibility'  # SPACE field-name reference
    else:
        k_sed_arg = config.FLUVIAL_K_SED     # SPACE scalar value

    # --- Water unit flux field (required by flow router) ---
    # Initialised to 1.0; overwritten each step by the precipitation field.
    if 'water__unit_flux_in' not in grid.at_node:
        grid.add_ones('water__unit_flux_in', at='node', clobber=True)

    # ------------------------------------------------------------------
    # NODATA CONSISTENCY ENFORCEMENT
    # ------------------------------------------------------------------
    # Any node that is CLOSED in the DEM is topographically invalid.  Apply
    # the closed-node mask to every field registered so far so that mapped
    # inputs (sediment depth, erodibility rasters) do not retain data at
    # locations where the terrain has no valid elevation.  This avoids
    # spurious erosion, deposition, or mass-balance errors at grid edges.
    _apply_nodata_mask_to_all_fields(grid, config)

    print(f"      Fields initialised in {time.time() - t_step:.2f} seconds.")

    # ------------------------------------------------------------------
    # STEP 3: CLIMATE & TECTONIC FORCING FIELDS
    # ------------------------------------------------------------------
    print("[3/4] Generating spatial precipitation and uplift fields...")
    t_step = time.time()

    # Precipitation spatial shape — dimensionless multiplier field in the
    # range [PRECIP_MIN, PRECIP_MAX].  The physical flux (m/yr) at each node
    # is obtained each timestep in the main loop as:
    #   water__unit_flux_in = get_effective_precip_rate(t) × precip_shape
    # PRECIP_MIN / PRECIP_MAX are therefore *relative* to the base rate:
    #   e.g. BASE_PRECIP_RATE=1.0 m/yr, PRECIP_MIN=1.0, PRECIP_MAX=4.5
    #   → 1.0–4.5 m/yr across the domain (4500 mm/yr orographic maximum).
    # ASSUMPTION: orographic amplification is approximated as a linear gradient.
    precip_shape = create_gradient(
        grid,
        config.PRECIP_MIN,
        config.PRECIP_MAX,
        angle_deg=config.PRECIP_GRADIENT_ANGLE_DEG
    )
    init_precip_rate = get_effective_precip_rate(0.0, config)
    print(
        f"      Precipitation base rate (t=0): {init_precip_rate:.3f} m/yr "
        f"({'timeseries' if config.PRECIP_TIMESERIES else 'constant'})"
    )
    print(
        f"      Spatial multipliers: {config.PRECIP_MIN:.2f}\u2013{config.PRECIP_MAX:.2f} "
        f"(gradient bearing {config.PRECIP_GRADIENT_ANGLE_DEG:.0f}\u00b0 CW from N)"
    )
    print(
        f"      Resulting flux range: "
        f"{init_precip_rate * config.PRECIP_MIN:.3f}\u2013"
        f"{init_precip_rate * config.PRECIP_MAX:.3f} m/yr"
    )

    # Uplift spatial shape — dimensionless multiplier field in the range
    # [UPLIFT_MIN_MULT, UPLIFT_MAX_MULT].  The physical uplift rate (m/yr) at
    # each node is computed each timestep in the main loop as:
    #   bedrock__elevation += uplift_shape × get_effective_uplift_rate(t) × dt
    # ASSUMPTION: the spatial pattern of uplift is time-invariant; only the
    # scalar base rate may vary (via UPLIFT_TIMESERIES or USE_DYNAMIC_UPLIFT).
    uplift_shape = create_gradient(
        grid,
        config.UPLIFT_MIN_MULT,
        config.UPLIFT_MAX_MULT,
        angle_deg=config.UPLIFT_GRADIENT_ANGLE_DEG
    )
    init_uplift_rate = get_effective_uplift_rate(0.0, config)
    print(
        f"      Uplift base rate (t=0): {init_uplift_rate:.5f} m/yr "
        f"({'timeseries' if config.UPLIFT_TIMESERIES else 'constant'})"
    )
    print(
        f"      Spatial multipliers: {config.UPLIFT_MIN_MULT:.2f}\u2013{config.UPLIFT_MAX_MULT:.2f} "
        f"(gradient bearing {config.UPLIFT_GRADIENT_ANGLE_DEG:.0f}\u00b0 CW from N)"
    )
    print(
        f"      Resulting rate range: "
        f"{init_uplift_rate * config.UPLIFT_MIN_MULT:.5f}\u2013"
        f"{init_uplift_rate * config.UPLIFT_MAX_MULT:.5f} m/yr"
    )
    print(f"      Forcing fields generated in {time.time() - t_step:.2f} seconds.")

    # ------------------------------------------------------------------
    # STEP 4: COMPONENT INSTANTIATION
    # ------------------------------------------------------------------
    print("[4/4] Instantiating geomorphic components...")
    t_step = time.time()
    components = {}

    # --- Flow routing ---
    # PriorityFloodFlowRouter dynamically detects and fills pits as the
    # landscape evolves — essential when mass movement or aggradation can
    # create new depressions during the run.  It is the only router
    # compatible with BedrockLandslider.
    # FlowAccumulator is faster but cannot handle dynamic pit formation.
    if config.USE_LANDSLIDES:
        print("      Initializing PriorityFloodFlowRouter (required by BedrockLandslider)...")
        components['flow_router'] = PriorityFloodFlowRouter(
            grid,
            flow_metric='D8',          # Single-direction D8 routing
            separate_hill_flow=True    # Keeps hillslope and channel flow separate
        )
        # BedrockLandslider — infinite-slope stability analysis.
        # PHYSICS: failure occurs when driving stress (gravity * slope) exceeds
        # resisting stress (cohesion + normal stress * friction angle).
        # Factor of Safety (FS) = (c + σ_n * tanφ) / (gamma * H * sinθ)
        # Failure triggered stochastically with mean return time LS_RETURN_TIME.
        print("      Initializing BedrockLandslider...")
        components['landslider'] = BedrockLandslider(
            grid,
            angle_int_frict=config.LS_ANGLE_FRICTION,
            cohesion_eff=config.LS_COHESION_EFF,
            landslides_return_time=config.LS_RETURN_TIME
        )
    else:
        print("      Initializing standard FlowAccumulator (D8)...")
        components['flow_router'] = FlowAccumulator(
            grid, flow_director='FlowDirectorD8'
        )

    # --- Hillslope component ---
    # TaylorNonLinearDiffuser (Roering et al. 1999):
    #   q_s = -K * S / (1 - (S/Sc)^2)
    #   This non-linear form produces threshold hillslopes — as slope
    #   approaches Sc the flux diverges, preventing angles from exceeding
    #   the friction angle.  Recommended for all steep-terrain applications.
    # LinearDiffuser (Culling 1963):
    #   q_s = -K * S
    #   Classical linear creep; appropriate only for low-gradient terrain.
    if config.HILLSLOPE_MODEL == "taylor":
        kwargs = dict(linear_diffusivity=config.HILLSLOPE_K)
        if config.HILLSLOPE_SC is not None:
            kwargs['slope_crit'] = config.HILLSLOPE_SC
        components['hillslope'] = TaylorNonLinearDiffuser(grid, **kwargs)
    else:
        components['hillslope'] = LinearDiffuser(
            grid, linear_diffusivity=config.HILLSLOPE_K
        )

    # --- Fluvial component ---
    # SPACE (Shobe et al. 2017):
    #   Coupled bedrock incision and sediment transport.  The erosion rate of
    #   bedrock depends on both stream power and the local sediment cover
    #   (the "cover effect").  Sediment is tracked explicitly as soil__depth.
    #   ASSUMPTION: sediment is transported as a single grain-size class.
    #   Erodibility can be uniform (scalar) or spatially variable (field name).
    # StreamPowerEroder (Detachment-limited):
    #   dz/dt = K_sp * A^m * S^n  (ignores sediment flux).
    #   Use when sediment thickness is negligible or in bedrock gorges.
    if config.FLUVIAL_MODEL == "space":
        # SPACE reads surface_water__discharge automatically.  The
        # PriorityFloodFlowRouter populates that field from water__unit_flux_in
        # (the spatially variable precipitation field) each timestep, so climate
        # forcing is already wired in — no extra kwarg is needed.
        space_kwargs = dict(
            K_sed=k_sed_arg,               # Sediment erodibility (scalar or field name)
            K_br='bedrock__erodibility',   # Bedrock erodibility (field name, set above)
            v_s=config.FLUVIAL_V_S,        # Grain settling velocity (m/yr)
            m_sp=config.SPACE_M,           # Drainage-area exponent
            n_sp=config.SPACE_N,           # Slope exponent
            F_f=config.SPACE_F_F,          # Fraction of sediment that bypasses node
            solver='adaptive',             # Use adaptive solver for significant performance gains
        )
        components['fluvial'] = Space(grid, **space_kwargs)
    else:
        components['fluvial'] = StreamPowerEroder(
            grid,
            K_sp='bedrock__erodibility',
            m_sp=config.SPACE_M,
            n_sp=config.SPACE_N
        )

    print(f"      Components ready in {time.time() - t_step:.2f} seconds.")
    print("      Priming flow router with initial topography...")
    components['flow_router'].run_one_step()
    print(f"\nSetup complete! Grid: {grid.number_of_nodes:,} nodes | "
          f"Elapsed: {time.time() - t0_init:.2f} s\n")

    return grid, components, precip_shape, uplift_shape


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

def run_simulation(config):
    """
    Orchestrates the full landscape evolution simulation.

    LOOP STRUCTURE (per timestep)
    ------------------------------
    Phase A  Climate stepping     — set dt and precipitation inputs
    Phase B  Tectonic forcing     — apply rock uplift to bedrock
    Phase C  Outlet lowering      — optional base-level fall (Rio Coca scenario)
    Phase D  Hillslope processes  — diffuse regolith downslope
    Phase E  Landslides           — decoupled mass-movement (every N years)
    Phase F  Flow routing         — compute drainage area and discharge
    Phase G  Fluvial erosion      — channel incision and sediment transport
    Phase H  Surface sync         — enforce z = bedrock + soil__depth
    Phase I  Weathering           — optional bedrock-to-regolith conversion
    Phase J  Elevation clipping   — numerical safety floor
    Phase K  Periodic logging     — progress and ETA to stdout
    Phase L  Periodic plotting    — map visualisation
    Phase M  Periodic save        — NetCDF archive snapshots

    TIMESTEP CONVENTION
    -------------------
    In constant-rainfall mode, dt_step = DT_BASE (years).
    In stochastic mode, dt_step = storm duration (years) and
    dt_total_cycle = storm_duration + interstorm_duration.
    All process components receive the FULL cycle duration so that
    cumulative fluxes integrate correctly over the hydrological year.

    STABILITY NOTES
    ---------------
    The CFL (Courant-Friedrichs-Lewy) condition for explicit advection
    schemes limits dt.  For SPACE:
        dt < dx / (K * Q^m * n * S^(n-1))
    For the hillslope diffuser:
        dt < dx^2 / (2 * K_D)

    When USE_ADAPTIVE_TIMESTEP = True (default), these limits are
    evaluated automatically each iteration using the current drainage
    area, discharge, and slope fields, and the timestep is clamped to
    CFL_SAFETY_FACTOR × min(dt_diff, dt_fluvial) — a 20 % safety margin
    by default.  DT_BASE still acts as the maximum ceiling.

    If USE_ADAPTIVE_TIMESTEP = False the user must choose DT_BASE manually.
    If instabilities appear (runaway erosion or NaNs in that fixed-step
    mode), reduce DT_BASE by a factor of 2–10.

    OUTPUTS
    -------
    NetCDF snapshots are written to config.SAVE_DIR every SAVE_INTERVAL_YEARS.
    Only fields listed in config.SAVE_FIELDS are included to control file size.
    A final snapshot is always written at the end of the run.

    Parameters
    ----------
    config : Config  Run configuration.
    """
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Parameter validation — runs before any expensive setup.
    validate_config(config)

    # Initialise grid, components, and spatial forcing fields.
    # precip_shape  — dimensionless multiplier per node [PRECIP_MIN, PRECIP_MAX]
    # uplift_shape  — dimensionless multiplier per node [UPLIFT_MIN_MULT, UPLIFT_MAX_MULT]
    # Physical rates are looked up each timestep via get_effective_*_rate().
    grid, components, precip_shape, uplift_shape = initialize_model(config)

    # Print runtime estimate now that we know the actual node count
    estimate_runtime(config, grid.number_of_nodes)

    # ------------------------------------------------------------------
    # RAINFALL ITERATOR SETUP
    # ------------------------------------------------------------------
    if config.RAINFALL_MODEL == "stochastic":
        storm_dur_iter, interstorm_dur_iter, intensity_iter = setup_stochastic_rainfall(
            config.SIM_LENGTH, seed=config.RANDOM_SEED
        )

    # ------------------------------------------------------------------
    # LOOP STATE VARIABLES
    # ------------------------------------------------------------------
    current_year      = 0.0
    iteration         = 0
    next_plot_target  = config.PLOT_INTERVAL_YEARS if config.PLOT_INTERVAL_YEARS is not False else float('inf')
    next_save_target  = config.SAVE_INTERVAL_YEARS
    next_log_target   = 0.0   # Force an immediate log at t=0
    log_interval      = max(100.0, config.SIM_LENGTH / 100.0)  # ~1 % per entry
    landslide_timer   = 0.0

    # Track whether outlet lowering should still be applied
    outlet_lowering_elapsed = 0.0

    # Outlet flux monitor state
    _oflux_interval = getattr(config, 'OUTLET_FLUX_REPORT_INTERVAL', None)
    next_outlet_report = _oflux_interval if _oflux_interval else float('inf')

    print(f"\n{'='*70}")
    print(f"  SIMULATION START: {config.RAINFALL_MODEL.upper()} rainfall | "
          f"{config.FLUVIAL_MODEL.upper()} fluvial | "
          f"{config.HILLSLOPE_MODEL.upper()} hillslope")
    print(f"  Duration: {config.SIM_LENGTH:,.0f} yr | dt_base: {config.DT_BASE} yr")
    print(f"{'='*70}\n")

    # Render and save the initial condition
    # Initial plotting is controlled by PLOT_INTERVAL_YEARS: when False,
    # plotting is disabled entirely (useful for long batch runs).
    if config.PLOT_INTERVAL_YEARS is not False:
        render_plot(grid, 0, config, title_suffix="Initial Condition")
    _save_snapshot(grid, config, 0)

    t_sim_start = time.time()
    t_last_log  = time.time()
    topo_prev   = grid.at_node['topographic__elevation'].copy()  # for erosion rate calc

    # ==================================================================
    # MAIN TIME LOOP
    # ==================================================================
    while current_year < config.SIM_LENGTH:

        # --------------------------------------------------------------
        # OUTLET GUARD — re-establish FIXED_VALUE outlet if lost.
        # Some Landlab components may reset node status internally.
        # Check every iteration; the cost is negligible (one np.sum).
        # --------------------------------------------------------------
        if int(np.sum(grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE)) == 0:
            import warnings as _w
            _w.warn(
                f"Outlet lost at year {current_year:.0f} — re-applying watershed BCs.",
                RuntimeWarning, stacklevel=2
            )
            _apply_watershed_bc(
                grid, grid.at_node['topographic__elevation'], config
            )

        # --------------------------------------------------------------
        # PHASE A: CLIMATE STEPPING
        # Set the precipitation flux and timestep for this iteration.
        # In stochastic mode each "event" is one storm-interstorm cycle.
        # In constant mode, DT_BASE is used directly.
        # --------------------------------------------------------------
        if config.RAINFALL_MODEL == "stochastic":
            try:
                d_storm  = next(storm_dur_iter)
                d_inter  = next(interstorm_dur_iter)
                intensity = next(intensity_iter)   # dimensionless, mean ≈ 1.0
            except StopIteration:
                # Iterator exhausted: fall back to mean values
                d_storm, d_inter, intensity = 0.5, 1.0, 1.0

            # Physical precipitation at each node (m/yr) =
            #   storm_intensity_multiplier × base_rate(t) × spatial_shape
            # Only apply precipitation during the storm phase;
            # interstorm period counts for timing but has no runoff.
            dt_step        = d_storm                 # Active erosion window (yr)
            dt_total_cycle = d_storm + d_inter       # Total model clock advance (yr)
            base_precip    = get_effective_precip_rate(current_year, config)
            grid.at_node['water__unit_flux_in'][:] = intensity * base_precip * precip_shape

        else:  # "constant"
            dt_step        = config.DT_BASE
            dt_total_cycle = config.DT_BASE
            d_inter        = 1.0   # Unused but kept for code consistency
            # Physical precipitation at each node (m/yr) =
            #   base_rate(t) × spatial_multiplier_shape
            base_precip    = get_effective_precip_rate(current_year, config)
            grid.at_node['water__unit_flux_in'][:] = base_precip * precip_shape

        # Clamp to not overshoot the simulation end
        remaining = config.SIM_LENGTH - current_year
        if dt_total_cycle > remaining:
            dt_total_cycle = remaining
            dt_step        = min(dt_step, remaining)

        # --------------------------------------------------------------
        # ADAPTIVE CFL TIMESTEP (optional — enabled by USE_ADAPTIVE_TIMESTEP)
        # --------------------------------------------------------------
        # Uses the landscape state from the END of the previous timestep
        # (drainage area and slope fields populated by the last flow-routing
        # call) to estimate the maximum stable dt.  This is the standard
        # explicit-scheme predictor approach: constrain step n+1 based on
        # the known state at n, then refine next iteration.
        #
        # Two CFL limits are evaluated:
        #   dt_diff  = dx² / (2 K_D)             — hillslope diffusion
        #   dt_fl    = dx / max(K·Q^m·n·S^(n−1)) — fluvial knickpoint celerity
        #
        # The tighter limit, scaled by CFL_SAFETY_FACTOR = 0.8, gives a
        # 20 % safety margin below the theoretical stability boundary.
        #
        # For constant-rainfall mode: the total cycle and the erosion step
        # are identical, so both are reduced together.
        # For stochastic mode: only the storm-active erosion window (dt_step)
        # is constrained; the interstorm duration is preserved unchanged so
        # the model clock advances by the correct physical amount.
        # --------------------------------------------------------------
        if getattr(config, 'USE_ADAPTIVE_TIMESTEP', False):
            dt_cfl = _compute_cfl_dt(grid, config, dt_step)
            if dt_cfl < dt_step:
                # Log only when the reduction is significant (> 10 %) to
                # avoid flooding the console during steady transient phases.
                if dt_cfl < 0.9 * dt_step:
                    print(
                        f"  [yr {current_year:,.0f}] Adaptive dt: "
                        f"{dt_step:.3f} → {dt_cfl:.3f} yr  "
                        f"(CFL safety {config.CFL_SAFETY_FACTOR:.0%})",
                        flush=True
                    )
                dt_step = dt_cfl
                # In constant mode, the total cycle equals the erosion step.
                if config.RAINFALL_MODEL == "constant":
                    dt_total_cycle = dt_step
                else:
                    # Stochastic: reduce total cycle by the same amount the
                    # storm was shortened; interstorm duration is preserved.
                    d_storm_reduction = dt_total_cycle - dt_step - d_inter
                    if d_storm_reduction > 0:
                        dt_total_cycle = dt_step + d_inter

        # --------------------------------------------------------------
        # PHASE B: TECTONIC FORCING (ROCK UPLIFT)
        # Uplift is applied to bedrock only (not surface) so SPACE can
        # independently track the bedrock/sediment interface.
        # ASSUMPTION: uplift is vertical (no horizontal advection).
        # Only core nodes are uplifted; boundary nodes are fixed base level.
        # Optional dynamic uplift multiplier from uplift_rate_at_time().
        # --------------------------------------------------------------
        # effective_uplift combines BASE_UPLIFT_RATE (or UPLIFT_TIMESERIES)
        # with the optional USE_DYNAMIC_UPLIFT multiplier from uplift_rate_at_time().
        # The spatial shape (UPLIFT_MIN_MULT – UPLIFT_MAX_MULT) is then applied
        # per-node so nodes with higher multipliers experience faster uplift.
        effective_uplift = get_effective_uplift_rate(current_year, config)
        grid.at_node['bedrock__elevation'][grid.core_nodes] += (
            uplift_shape[grid.core_nodes] * effective_uplift * dt_total_cycle
        )

        # --------------------------------------------------------------
        # PHASE C: BASE-LEVEL LOWERING (optional)
        # Simulates the effect of the San Rafael waterfall collapse:
        # the downstream boundary drops at OUTLET_LOWERING_RATE (m/yr),
        # creating a wave of incision that migrates upstream.
        # ASSUMPTION: the outlet is the lowest-elevation boundary node.
        # This is decoupled from uplift so both can operate simultaneously.
        # --------------------------------------------------------------
        if (config.USE_OUTLET_LOWERING
                and outlet_lowering_elapsed < config.OUTLET_LOWERING_DURATION):
            # Identify the lowest open boundary node as the outlet
            open_bnds = np.where(
                grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE
            )[0]
            if len(open_bnds) > 0:
                topo = grid.at_node['topographic__elevation']
                outlet_idx = open_bnds[np.argmin(topo[open_bnds])]
                drop = config.OUTLET_LOWERING_RATE * dt_total_cycle
                topo[outlet_idx]                               -= drop
                grid.at_node['bedrock__elevation'][outlet_idx] -= drop
            outlet_lowering_elapsed += dt_total_cycle

        # --------------------------------------------------------------
        # PHASE D: HILLSLOPE PROCESSES
        # The diffuser smooths topographic roughness via non-linear or
        # linear creep, transferring regolith downslope.
        #
        # CRITICAL — mass commitment before Phase H:
        # Landlab diffusers operate solely on topographic__elevation; they
        # never touch bedrock__elevation or soil__depth.  Without the fix
        # below, Phase H ( topo = br + sd ) would immediately undo the
        # diffuser's work by resetting topo from the unchanged br and sd
        # fields — making hillslopes completely static no matter the K_D.
        #
        # Fix: snapshot topo before diffusion; after the diffuser runs,
        # compute Δz_diff = topo_after − topo_before and route that mass
        # change into soil__depth.  Where erosion removes more sediment than
        # is available ( sd < 0 ), the deficit is carved from bedrock —
        # physically equivalent to weathering-limited creep on bare rock.
        # --------------------------------------------------------------
        topo = grid.at_node['topographic__elevation']
        br   = grid.at_node['bedrock__elevation']
        sd   = grid.at_node['soil__depth']
        topo_pre_diff = topo.copy()
        _diffuser_ok = True
        try:
            components['hillslope'].run_one_step(dt_total_cycle)
        except Exception as e:
            warnings.warn(
                f"Hillslope diffuser raised an exception at yr "
                f"{current_year:.0f}: {e}. Skipping this step.",
                RuntimeWarning
            )
            _diffuser_ok = False

        # Commit Δz_diff into the mass-tracked fields so Phase H does not
        # overwrite it.  Only core nodes are updated; boundary elevations
        # are fixed by their BC status and must not be altered here.
        if _diffuser_ok:
            dz_diff = topo[:] - topo_pre_diff[:]
            cores   = grid.core_nodes
            sd[cores] += dz_diff[cores]
            # Where hillslope erosion outpaced available regolith, carve
            # bedrock to maintain full mass conservation.
            deficit   = np.minimum(sd[cores], 0.0)   # ≤ 0 where sediment gone
            sd[cores] = np.maximum(sd[cores], 0.0)
            br[cores] += deficit                      # deficit < 0 → lowers br
            topo[cores] = br[cores] + sd[cores]       # re-sync surface

        # --------------------------------------------------------------
        # PHASE E: DECOUPLED LANDSLIDES
        # BedrockLandslider is expensive per call, so it is applied in
        # "bursts" only when the internal timer exceeds the trigger interval.
        # The timer accumulates real simulated years between triggers.
        # Sediment evacuated by landslides is added to the fluvial sediment
        # budget implicitly via soil__depth updates made by the component.
        # --------------------------------------------------------------
        if config.USE_LANDSLIDES:
            landslide_timer += dt_total_cycle
            if landslide_timer >= config.DT_LANDSLIDE_TRIGGER:
                vol_sus_sed, vol_leaving = components['landslider'].run_one_step(
                    dt=landslide_timer
                )
                if vol_leaving > 0 or vol_sus_sed > 0:
                    print(
                        f"  [Year {current_year:,.0f}] Landslide — "
                        f"evacuated: {vol_leaving:.2f} m³, "
                        f"in suspension: {vol_sus_sed:.2f} m³",
                        flush=True
                    )
                landslide_timer = 0.0

        # --------------------------------------------------------------
        # PHASE F: FLOW ROUTING
        # Computes drainage area, flow directions, and surface-water
        # discharge at every node.  Must be called BEFORE the fluvial
        # component, which requires 'drainage_area' and 'topographic__steepest_slope'.
        # PriorityFloodFlowRouter re-fills any new sinks created by deposition
        # or landslides before routing flow.
        # --------------------------------------------------------------
        components['flow_router'].run_one_step()

        # --------------------------------------------------------------
        # PHASE G: FLUVIAL EROSION & SEDIMENT TRANSPORT
        # SPACE: computes bedrock incision and sediment entrainment/deposition.
        #   • Bedrock erosion: E_br = K_br * q^m * S^n * exp(-H/H_s)
        #     where H = soil depth, H_s = characteristic shielding depth
        #   • Sediment erosion: E_sed = K_sed * q^m * S^n * (1 - exp(-H/H_s))
        #   • Net deposition: D = q_s * V_s / q
        # dt_step (not dt_total_cycle) is used because erosion only occurs
        # during the active storm, not the interstorm period.
        # --------------------------------------------------------------
        components['fluvial'].run_one_step(dt=dt_step)

        # --------------------------------------------------------------
        # PHASE H: SURFACE SYNCHRONISATION
        # Enforce strict self-consistency of the three elevation fields.
        # SPACE updates bedrock__elevation and soil__depth independently;
        # topographic__elevation must be recomputed from their sum.
        # StreamPowerEroder updates topographic__elevation directly;
        # bedrock__elevation is back-computed.
        # CRITICAL: failure to sync these fields causes drift and eventually
        # crashes BedrockLandslider's internal consistency check.
        # --------------------------------------------------------------
        topo = grid.at_node['topographic__elevation']
        br   = grid.at_node['bedrock__elevation']
        sd   = grid.at_node['soil__depth']

        if config.FLUVIAL_MODEL == "space":
            # SPACE master fields are bedrock + soil; topo follows
            topo[:] = br[:] + sd[:]
        else:
            # StreamPowerEroder master field is topo; soil is residual
            br[:] = topo[:] - sd[:]

        # Clamp soil depth to zero (cannot be negative — bedrock exposed)
        sd[:] = np.maximum(sd[:], 0.0)
        # Recompute surface after clamp
        topo[:] = br[:] + sd[:]

        # --------------------------------------------------------------
        # PHASE I: BEDROCK WEATHERING (optional)
        # Simple depth-independent weathering converts bedrock to regolith
        # at a constant rate WEATHERING_RATE (m/yr).
        # CAUTION: do not add weathered material without removing it from
        # bedrock, or mass is not conserved.
        # TO-DO: implement humped-function (depth-dependent) weathering:
        #   dW/dt = W0 * exp(-soil__depth / h_star)
        # --------------------------------------------------------------
        if config.WEATHERING_RATE > 0.0:
            weathering = config.WEATHERING_RATE * dt_total_cycle
            br[grid.core_nodes]   -= weathering
            sd[grid.core_nodes]   += weathering
            topo[grid.core_nodes] = br[grid.core_nodes] + sd[grid.core_nodes]

        # --------------------------------------------------------------
        # PHASE J: NUMERICAL SAFETY — elevation floor
        # Prevents catastrophic runaway incision in rare numerical events
        # (e.g. steep gradients + large dt).  Does not affect normal runs.
        # --------------------------------------------------------------
        if np.any(topo < config.MIN_ELEVATION_CLIP):
            n_clipped = np.sum(topo < config.MIN_ELEVATION_CLIP)
            warnings.warn(
                f"yr {current_year:.0f}: {n_clipped} nodes clipped to "
                f"MIN_ELEVATION_CLIP={config.MIN_ELEVATION_CLIP}",
                RuntimeWarning
            )
            topo[:] = np.maximum(topo[:], config.MIN_ELEVATION_CLIP)
            br[:]   = np.minimum(br[:], topo[:])   # bedrock cannot exceed surface
            sd[:]   = topo[:] - br[:]

        # Advance simulation clock
        current_year += dt_total_cycle
        iteration    += 1

        # --------------------------------------------------------------
        # PHASE K: PERIODIC LOGGING
        # Prints a one-line progress report to stdout, including estimated
        # time to completion (ETA).  Frequency controlled by log_interval.
        # --------------------------------------------------------------
        if current_year >= next_log_target:
            t_now         = time.time()
            elapsed_step  = t_now - t_last_log
            total_elapsed = t_now - t_sim_start
            speed = dt_total_cycle / elapsed_step if elapsed_step > 0 else 0

            years_remaining = config.SIM_LENGTH - current_year
            eta_seconds = years_remaining / speed if speed > 0 else 0
            eta_mins, eta_secs = divmod(int(eta_seconds), 60)
            eta_hrs,  eta_mins = divmod(eta_mins, 60)

            pct = (current_year / config.SIM_LENGTH) * 100
            print(
                f"[{pct:5.1f}%] yr {current_year:9,.1f} | "
                f"iter {iteration:7d} | "
                f"{speed:7.1f} sim-yr/s | "
                f"ETA {eta_hrs:02d}:{eta_mins:02d}:{eta_secs:02d}",
                flush=True
            )
            t_last_log      = t_now
            next_log_target += log_interval

        # --------------------------------------------------------------
        # PHASE K2: OUTLET FLUX MONITORING
        # Report discharge and sediment leaving the downstream boundary
        # every OUTLET_FLUX_REPORT_INTERVAL years and flag any physically
        # implausible values with a [WARNING] tag.
        # --------------------------------------------------------------
        if current_year >= next_outlet_report:
            _report_outlet_fluxes(grid, config, current_year, dt_total_cycle)
            next_outlet_report += (_oflux_interval or float('inf'))

        # --------------------------------------------------------------
        # PHASE L: PERIODIC VISUALISATION
        # Renders the PLOT_FIELD map every PLOT_INTERVAL_YEARS.
        # A multi-panel diagnostic snapshot is shown every 10 plot intervals.
        # Skipped entirely if PLOT_INTERVAL_YEARS = False.
        # --------------------------------------------------------------
        if config.PLOT_INTERVAL_YEARS is not False and current_year >= next_plot_target:
            print(f">>> Rendering plot at yr {current_year:,.0f}...")
            render_plot(grid, current_year, config)
            # Every 10th plot, also show a multi-panel diagnostic
            plot_count = int(current_year / config.PLOT_INTERVAL_YEARS)
            if plot_count % 10 == 0:
                render_multi_panel(grid, current_year)
            next_plot_target += config.PLOT_INTERVAL_YEARS

        # --------------------------------------------------------------
        # PHASE M: PERIODIC ARCHIVAL
        # Saves selected Landlab fields to a NetCDF file each interval.
        # The SAVE_FIELDS list controls which fields are written;
        # adjust it to reduce disk usage for large grids.
        # --------------------------------------------------------------
        if current_year >= next_save_target:
            print(f">>> Saving snapshot at yr {current_year:,.0f}...")
            _save_snapshot(grid, config, current_year)
            next_save_target += config.SAVE_INTERVAL_YEARS

    # ==================================================================
    # POST-RUN
    # ==================================================================
    # Always write a final snapshot regardless of interval alignment
    print(f">>> Saving FINAL snapshot at yr {current_year:,.0f}...")
    _save_snapshot(grid, config, current_year)
    # Final plotting also obeys PLOT_INTERVAL_YEARS so users can disable
    # all plotting by setting the switch to False.
    if config.PLOT_INTERVAL_YEARS is not False:
        render_plot(grid, current_year, config, title_suffix="Final Condition")

    t_total = time.time() - t_sim_start
    hrs,  rem  = divmod(int(t_total), 3600)
    mins, secs = divmod(rem, 60)
    print(f"\nSimulation complete in {hrs:02d}h {mins:02d}m {secs:02d}s.")
    print(f"Outputs saved in: {os.path.abspath(config.SAVE_DIR)}")
    return grid


# ============================================================================
# INTERNAL HELPER — SNAPSHOT WRITER
# ============================================================================

def _save_snapshot(grid, config, year):
    """
    Writes a NetCDF snapshot of the model state to config.SAVE_DIR.

    Only fields listed in config.SAVE_FIELDS that actually exist on the grid
    are written.  This avoids:
      - Errors when optional fields (e.g. soil__depth) are absent in simple
        StreamPowerEroder runs.
      - Dimension-conflict errors caused by multi-valued per-node fields that
        flow-routing components register internally, such as:
          hill_topographic__steepest_slope  (shape: n_nodes x 8 for D8)
          hill_drainage_area                (shape: n_nodes x 8)
        These have a different 'nt' dimension than scalar node fields (nt=1)
        and cause write_netcdf to raise a conflicting-dimension RuntimeError.
        By explicitly passing `names`, only the requested scalar fields are
        written and the multi-valued routing fields are skipped entirely.

    The output filename encodes the simulation year with zero-padded integers
    so files sort correctly in directory listings.

    Parameters
    ----------
    grid   : RasterModelGrid
    config : Config
    year   : float  Current simulation time (years).
    """
    fname = os.path.join(
        config.SAVE_DIR, f'lem_yr_{int(year):010d}.nc'
    )

    # Only write the scalar node fields requested in SAVE_FIELDS.
    # Passing `names` to write_netcdf restricts output to those fields,
    # preventing dimension-size conflicts with multi-valued routing fields.
    fields_to_write = [f for f in config.SAVE_FIELDS if f in grid.at_node]

    if not fields_to_write:
        warnings.warn(
            f"No SAVE_FIELDS found on grid at yr {year}; snapshot skipped.",
            RuntimeWarning
        )
        return

    try:
        write_netcdf(fname, grid, names=fields_to_write)
        print(f"      Saved: {fname}  (fields: {', '.join(fields_to_write)})")
    except Exception as exc:
        warnings.warn(f"NetCDF write failed for yr {year}: {exc}", RuntimeWarning)



# ============================================================================
# ENTRY POINT
# ============================================================================
# All configuration is done by editing the Config class above or by overriding
# individual attributes here in the __main__ block.  The latter approach is
# preferred for quick parameter sweeps without changing the class defaults.
#
# RIO COCA SCENARIO NOTES
# -----------------------
# The San Rafael waterfall collapse (February 2020) effectively lowered the
# local base level by ~150 m instantly.  To simulate this:
#   1. Set USE_OUTLET_LOWERING = True
#   2. Set OUTLET_LOWERING_RATE  = 150 / duration_yr  (m/yr)
#   3. Set OUTLET_LOWERING_DURATION to the number of years over which you
#      want the drop to occur (use a short value, e.g. 1 yr, for an
#      instantaneous step change).
# The model will then propagate a knickzone upstream and you can compare
# predicted incision rates to field observations of channel adjustment.
#
# GENERALISATION NOTE
# -------------------
# For other watersheds, change DEM_FILE, adjust K_BR and K_SED to match
# the local lithology, set UPLIFT_MIN_MULT/MAX_MULT and BASE_UPLIFT_RATE to regional uplift rates, and
# set PRECIP_MIN/MAX to orographic gradients from climate reanalysis.
# ============================================================================
if __name__ == "__main__":

    # Create a configuration instance (defaults match the original Rio Coca setup)
    cfg = Config()

    # ----------------------------------------------------------------
    # OPTIONAL: override specific parameters for a scenario run.
    # Uncomment and edit any line below; all other settings use Config defaults.
    # ----------------------------------------------------------------

    # --- DEM selection ---
    # cfg.DEM_FILE        = './RioCoca_50m_Elevation.tif'       # 50 m resolution (faster)
    # cfg.DEM_FILE        = './GIS Files/Rio_30m_UTM18S_1.tif'  # 30 m resolution

    # --- Simulation length ---
    # cfg.SIM_LENGTH      = 50000.0    # 50 kyr run
    # cfg.DT_BASE         = 10.0       # Reduce dt for stability at finer resolution

    # --- Rio Coca waterfall collapse scenario ---
    # cfg.USE_OUTLET_LOWERING          = True
    # cfg.OUTLET_LOWERING_RATE         = 150.0   # m/yr (150 m drop over 1 year)
    # cfg.OUTLET_LOWERING_DURATION     = 1.0     # yr  (effectively instantaneous)

    # --- Mapped lithology (if rasters are available) ---
    # cfg.LITHOLOGY_MODE                = 'mapped'
    # cfg.BEDROCK_ERODIBILITY_FILE      = './GIS Files/k_br_map.tif'
    # cfg.SEDIMENT_DEPTH_MODE           = 'mapped'
    # cfg.SOIL_DEPTH_FILE               = './GIS Files/regolith_depth.tif'
    # cfg.SEDIMENT_ERODIBILITY_MODE     = 'mapped'
    # cfg.SEDIMENT_ERODIBILITY_FILE     = './GIS Files/k_sed_map.tif'

    # --- Disable landslides for a faster benchmark run ---
    # cfg.USE_LANDSLIDES  = False

    # --- Stochastic rainfall for event-scale dynamics ---
    # cfg.RAINFALL_MODEL  = 'stochastic'

    # --- Stream-power-only (no sediment tracking) for simple tests ---
    # cfg.FLUVIAL_MODEL   = 'stream_power'
    # cfg.HILLSLOPE_MODEL = 'linear'

    # ----------------------------------------------------------------
    # Run the model
    # ----------------------------------------------------------------
    final_grid = run_simulation(cfg)

    # ----------------------------------------------------------------
    # OPTIONAL: Create animation from saved snapshots
    # ----------------------------------------------------------------
    # Uncomment the line below to generate an MP4 animation of the terrain
    # evolving through time.  Requires FFmpeg to be installed.
    # create_terrain_animation(cfg, output_field='topographic__elevation', fps=10)
