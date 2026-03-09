from LEM_V3 import Config, run_simulation
from landlab.components import FlowAccumulator

# Modify run_simulation or config?
# The problem is that PriorityFloodFlowRouter uses `water__unit_flux_in` by default (or automatically).
# Oh wait, `PriorityFloodFlowRouter` *does* output `surface_water__discharge`.
# Let's check `PriorityFloodFlowRouter` behavior again.
# But it says "PriorityFloodFlowRouter requires richdem but richdem is not installed".
