from landlab import RasterModelGrid
from landlab.components import FlowAccumulator

grid = RasterModelGrid((10, 10), xy_spacing=10.0)
grid.add_zeros('topographic__elevation', at='node')
grid.add_ones('water__unit_flux_in', at='node')

fa = FlowAccumulator(grid)
fa.run_one_step()
print("FA surface_water__discharge in grid:", 'surface_water__discharge' in grid.at_node)
print("max FA discharge:", grid.at_node['surface_water__discharge'].max() if 'surface_water__discharge' in grid.at_node else "N/A")
