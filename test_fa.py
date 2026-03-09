import numpy as np
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

grid = RasterModelGrid((10, 10), xy_spacing=10.0)
grid.add_zeros('topographic__elevation', at='node')
grid.add_ones('water__unit_flux_in', at='node')
fa = FlowAccumulator(grid, runoff_rate='water__unit_flux_in')
fa.run_one_step()

print("Fields in grid:")
for key in grid.at_node.keys():
    print(key)

print("\nMax DA:", np.max(grid.at_node['drainage_area']))
if 'surface_water__discharge' in grid.at_node:
    print("Max Q:", np.max(grid.at_node['surface_water__discharge']))
else:
    print("No surface_water__discharge")
