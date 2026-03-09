from LEM_V3 import Config, run_simulation

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False

try:
    final_grid = run_simulation(cfg)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
