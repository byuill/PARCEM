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

final_grid = run_simulation(cfg)

# Print values
print("Discharge:", final_grid.at_node['surface_water__discharge'])
print("Water flux:", final_grid.at_node['water__unit_flux_in'])
print("Topographic elevation:", final_grid.at_node['topographic__elevation'])
