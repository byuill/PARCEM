import LEM_V3_patched
from LEM_V3_patched import Config, initialize_model

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False

grid, components, precip, uplift = initialize_model(cfg)

grid.at_node['water__unit_flux_in'][:] = 0.5
components['flow_router'].run_one_step()

print("Surface Water Discharge:", 'surface_water__discharge' in grid.at_node)
print("Drainage Area:", 'drainage_area' in grid.at_node)
print("Water Unit Flux In:", 'water__unit_flux_in' in grid.at_node)
