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

discharge_field = grid.at_node.get('surface_water__discharge', None)
da_field = grid.at_node.get('drainage_area', None)
cores = grid.core_nodes

print("Max Q array core nodes:", np.max(discharge_field[cores]))
print("Max DA array core nodes:", np.max(da_field[cores]))

_max_da_core  = float(np.max(da_field[cores]))        if da_field        is not None else 0.0
_max_Q_core   = float(np.max(discharge_field[cores])) if discharge_field is not None else 0.0

print("_max_da_core via line 966:", _max_da_core)
print("_max_Q_core via line 967:", _max_Q_core)

# Let's run _report_outlet_fluxes piece by piece.
open_bnds = np.where(grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE)[0]

_q_method  = "none"
Q_out_m3yr = float('nan')
cores = grid.core_nodes

if discharge_field is not None and 'flow__receiver_node' in grid.at_node:
    receiver   = grid.at_node['flow__receiver_node']
    Q_donors   = 0.0
    for outlet_id in open_bnds:
        donors   = np.where(receiver == outlet_id)[0]
        Q_donors += float(np.sum(discharge_field[donors]))
    if Q_donors > 0.0:
        Q_out_m3yr = Q_donors
        _q_method  = "Q[donors->outlet]"
        print("Strategy A:", Q_out_m3yr)
