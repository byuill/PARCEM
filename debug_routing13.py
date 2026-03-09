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
print(list(grid.at_node.keys()))
print("surface_water__discharge" in grid.at_node)
print("Wait, what happens if I use `grid.at_node['surface_water__discharge']` directly?")
print(grid.at_node['surface_water__discharge'])
