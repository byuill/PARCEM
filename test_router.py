from landlab import RasterModelGrid
from landlab.components import FlowAccumulator, PriorityFloodFlowRouter

grid = RasterModelGrid((10, 10), xy_spacing=10.0)
grid.add_zeros('topographic__elevation', at='node')
grid.add_ones('water__unit_flux_in', at='node')

fa = FlowAccumulator(grid, runoff_rate='water__unit_flux_in')
fa.run_one_step()
print("FA surface_water__discharge in grid:", 'surface_water__discharge' in grid.at_node)

grid2 = RasterModelGrid((10, 10), xy_spacing=10.0)
grid2.add_zeros('topographic__elevation', at='node')
grid2.add_ones('water__unit_flux_in', at='node')

pf = PriorityFloodFlowRouter(grid2, accumulate_flow=True, flow_metric='D8', runoff_rate='water__unit_flux_in')
pf.run_one_step()
print("PF surface_water__discharge in grid:", 'surface_water__discharge' in grid2.at_node)
