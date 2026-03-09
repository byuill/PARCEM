import LEM_V3_patched
from LEM_V3_patched import Config, initialize_model
import numpy as np

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False

grid, components, precip, uplift = initialize_model(cfg)

from LEM_V3_patched import _report_outlet_fluxes

import inspect
lines = inspect.getsource(_report_outlet_fluxes)
print("Source of _report_outlet_fluxes:")
print(lines)
