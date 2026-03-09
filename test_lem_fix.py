from LEM_V3 import Config, run_simulation

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False
cfg.OUTLET_FLUX_REPORT_INTERVAL = 1.0
cfg.PLOT_INTERVAL_YEARS = False

# The problem is that FlowAccumulator takes `runoff_rate='water__unit_flux_in'`
# If not passed, it uses a uniform default of 1.0, and since water__unit_flux_in
# is updated but not read, the discharge is wrong/missing.
# Actually let's check what FlowAccumulator expects.

import sys
sys.path.append(".")
import LEM_V3

def patched_init_model(config):
    # Just checking what happens if we patch components['flow_router'] initialization.
    pass
