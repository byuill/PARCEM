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

# We will modify LEM_V3 directly.
import re

with open("LEM_V3.py", "r") as f:
    code = f.read()

# Modify FlowAccumulator to explicitly accept runoff_rate='water__unit_flux_in'
code = code.replace(
    "components['flow_router'] = FlowAccumulator(\n            grid, flow_director='FlowDirectorD8'\n        )",
    "components['flow_router'] = FlowAccumulator(\n            grid, flow_director='FlowDirectorD8', runoff_rate='water__unit_flux_in'\n        )"
)

with open("LEM_V3_patched.py", "w") as f:
    f.write(code)

import LEM_V3_patched
from LEM_V3_patched import Config as ConfigPatched, run_simulation as run_simulation_patched

cfg2 = ConfigPatched()
cfg2.LANDSCAPE_MODE = "synthetic"
cfg2.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg2.SYNTHETIC_DX = 10.0
cfg2.SIM_LENGTH = 10.0
cfg2.DT_BASE = 1.0
cfg2.USE_CHANNEL_FOOTPRINT = False
cfg2.USE_LANDSLIDES = False
cfg2.OUTLET_FLUX_REPORT_INTERVAL = 1.0
cfg2.PLOT_INTERVAL_YEARS = False

try:
    final_grid = run_simulation_patched(cfg2)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
