import LEM_V3_patched
from LEM_V3_patched import Config, _report_outlet_fluxes, initialize_model
from landlab.components import FlowAccumulator

cfg = Config()
cfg.LANDSCAPE_MODE = "synthetic"
cfg.SYNTHETIC_GRID_SHAPE = (10, 10)
cfg.SYNTHETIC_DX = 10.0
cfg.SIM_LENGTH = 10.0
cfg.DT_BASE = 1.0
cfg.USE_CHANNEL_FOOTPRINT = False
cfg.USE_LANDSLIDES = False
cfg.OUTLET_FLUX_REPORT_INTERVAL = 1.0

grid, components, precip, uplift = initialize_model(cfg)

# Re-create FlowAccumulator WITHOUT runoff_rate string. Just pass runoff_rate as the field.
# But wait, we can just say runoff_rate='water__unit_flux_in' ... why didn't it create the field?
fa = FlowAccumulator(grid, flow_director='FlowDirectorD8', runoff_rate='water__unit_flux_in')
fa.run_one_step()

print("Q Field after our FA:", grid.at_node.get('surface_water__discharge'))
print("DA Field after our FA:", grid.at_node.get('drainage_area'))
