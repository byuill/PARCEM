import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

grid = RasterModelGrid((5, 5), xy_spacing=10.0)
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
# Try without runoff_rate. It will just accumulate DA, not Q? Wait, let's see.
fa = FlowAccumulator(grid, flow_director='FlowDirectorD8')
fa.run_one_step()

print("FA surface_water__discharge in grid:", 'surface_water__discharge' in grid.at_node)
if 'surface_water__discharge' in grid.at_node:
    print("Core Q:", grid.at_node['surface_water__discharge'][grid.core_nodes])
else:
    print("No Q!")

# Try with runoff_rate=1.0 ?
