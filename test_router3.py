import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

grid = RasterModelGrid((5, 5), xy_spacing=10.0)
grid.add_zeros('topographic__elevation', at='node')
# Set right edge lowest so it drains right
z = np.array([
    10, 10, 10, 10, 10,
    9,  9,  9,  9,  0,
    8,  8,  8,  8,  0,
    7,  7,  7,  7,  0,
    6,  6,  6,  6,  6
])
grid.at_node['topographic__elevation'][:] = z
grid.set_closed_boundaries_at_grid_edges(True, True, False, True)
grid.add_ones('water__unit_flux_in', at='node')

fa = FlowAccumulator(grid, runoff_rate='water__unit_flux_in')
fa.run_one_step()
print("Max DA:", grid.at_node['drainage_area'].max())
if 'surface_water__discharge' in grid.at_node:
    print("Max Q:", grid.at_node['surface_water__discharge'].max())
else:
    print("No surface_water__discharge")
