import LEM_V3_patched
from LEM_V3_patched import Config, initialize_model, _report_outlet_fluxes
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

print("Before masking, discharge available:", 'surface_water__discharge' in grid.at_node)

from LEM_V3_patched import _apply_nodata_mask_to_all_fields

print("Before apply_nodata_mask_to_all_fields, max_Q =", np.max(grid.at_node['surface_water__discharge']))
# Actually let's just see where _report_outlet_fluxes fails...
_report_outlet_fluxes(grid, cfg, 0, 1)
