from LEM_V3_patched import Config, _report_outlet_fluxes, initialize_model
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
print("After initialization:")
print("max DA:", np.max(grid.at_node['drainage_area'][grid.core_nodes]))
print("max Q:", np.max(grid.at_node['surface_water__discharge'][grid.core_nodes]))

_report_outlet_fluxes(grid, cfg, 0, 1.0)
