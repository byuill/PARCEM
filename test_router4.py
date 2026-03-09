import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

grid = RasterModelGrid((5, 5), xy_spacing=10.0)
# Set right edge lowest so it drains right
z = np.array([
    10., 10., 10., 10., 10.,
    9.,  8.,  7.,  6.,  5.,
    9.,  8.,  7.,  6.,  5.,
    9.,  8.,  7.,  6.,  5.,
    10., 10., 10., 10., 10.
])
grid.add_field('topographic__elevation', z, at='node')
grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
grid.status_at_node[14] = grid.BC_NODE_IS_FIXED_VALUE

grid.add_ones('water__unit_flux_in', at='node')
fa = FlowAccumulator(grid, flow_director='FlowDirectorD8', runoff_rate='water__unit_flux_in')
fa.run_one_step()

print("Core DA:", grid.at_node['drainage_area'][grid.core_nodes])
print("Boundary DA:", grid.at_node['drainage_area'][14])
print("FA surface_water__discharge in grid:", 'surface_water__discharge' in grid.at_node)
if 'surface_water__discharge' in grid.at_node:
    print("Core Q:", grid.at_node['surface_water__discharge'][grid.core_nodes])
