import LEM_V3_patched
from LEM_V3_patched import Config, initialize_model
from landlab.components import FlowAccumulator

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False

grid, components, precip, uplift = initialize_model(cfg)

print("Before router run:", 'surface_water__discharge' in grid.at_node)
# The flow router in components['flow_router'] was already run during initialize_model.
print("After initialize_model:", 'surface_water__discharge' in grid.at_node)

fa = FlowAccumulator(grid, flow_director='FlowDirectorD8', runoff_rate='water__unit_flux_in')
fa.run_one_step()

print("After second FA run:", 'surface_water__discharge' in grid.at_node)
