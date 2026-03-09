from LEM_V3_patched import Config, run_simulation

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
cfg.BASE_PRECIP_RATE = 1.0
cfg.RAINFALL_MODEL = "constant"

final_grid = run_simulation(cfg)
print("Success")
