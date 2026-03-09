with open("LEM_V3.py", "r") as f:
    code = f.read()

# Fix 1: PriorityFloodFlowRouter -> replace FlowAccumulator args
code = code.replace(
    "components['flow_router'] = FlowAccumulator(\n            grid, flow_director='FlowDirectorD8'\n        )",
    "components['flow_router'] = FlowAccumulator(\n            grid, flow_director='FlowDirectorD8', runoff_rate='water__unit_flux_in'\n        )"
)

# Fix 2: grid.at_node.get('field', fallback) doesn't work correctly with FieldDataset in Landlab
# Let's replace grid.at_node.get('surface_water__discharge', None) with
# (grid.at_node['surface_water__discharge'] if 'surface_water__discharge' in grid.at_node else None)

import re

# Need to replace cases of grid.at_node.get('field', None)
code = re.sub(
    r"grid\.at_node\.get\('([^']+)',\s*None\)",
    r"(grid.at_node['\1'] if '\1' in grid.at_node else None)",
    code
)

# And one place uses da_field = grid.at_node.get('drainage_area', None)
# This will be fixed by the regex above!

# Let's also check for get('field') without second arg
code = re.sub(
    r"grid\.at_node\.get\('([^']+)'\)",
    r"(grid.at_node['\1'] if '\1' in grid.at_node else None)",
    code
)

with open("LEM_V3.py", "w") as f:
    f.write(code)
