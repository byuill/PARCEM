import argparse
import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.components import FlowAccumulator, Space, TaylorNonLinearDiffuser, BedrockLandslider
from landlab.plot import imshow_grid

def parse_args():
    parser = argparse.ArgumentParser(description="Landscape Evolution Model using Landlab")
    parser.add_argument("--rows", type=int, default=100, help="Number of rows in the elevation raster")
    parser.add_argument("--cols", type=int, default=100, help="Number of columns in the elevation raster")
    parser.add_argument("--dx", type=float, default=10.0, help="Cell resolution (m)")
    parser.add_argument("--time", type=float, default=100000.0, help="Total simulation time (years)")
    parser.add_argument("--dt", type=float, default=100.0, help="Time step (years)")
    parser.add_argument("--uplift-rate", type=float, default=0.001, help="Tectonic uplift rate (m/yr)")
    parser.add_argument("--k-br", type=float, default=0.001, help="Bedrock erodibility")
    parser.add_argument("--k-sed", type=float, default=0.005, help="Sediment erodibility")
    parser.add_argument("--diffusivity", type=float, default=0.01, help="Linear diffusivity for hillslope")
    parser.add_argument("--precip-rate", type=float, default=1.0, help="Effective continuous precipitation rate (m/yr)")
    parser.add_argument("--output-plot", type=str, default="landscape_evolution.png", help="Output plot filename")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Grid initialization
    grid = RasterModelGrid((args.rows, args.cols), xy_spacing=args.dx)

    # 2. Add required fields
    z = grid.add_zeros("topographic__elevation", at="node")
    # Add a slight slope and some random noise to encourage flow routing
    np.random.seed(42)
    z += np.random.rand(z.size) * 0.1
    z += grid.node_y * 0.01

    soil_depth = grid.add_zeros("soil__depth", at="node")
    soil_depth[:] = 1.0  # Initial uniform soil depth

    bedrock = grid.add_zeros("bedrock__elevation", at="node")
    bedrock[:] = z - soil_depth

    # 3. Boundary conditions
    # Set closed boundaries at the edges, except for the bottom edge (node_y = 0) which will be the outlet
    grid.set_closed_boundaries_at_grid_edges(True, True, True, False)

    # Precipitation config:
    # Set a continuous effective precipitation/runoff rate for FlowAccumulator
    # instead of explicitly simulating individual short-lived storms over long 100kyr+ timescales.
    grid.add_zeros('water__unit_flux_in', at='node')
    grid.at_node['water__unit_flux_in'][:] = args.precip_rate

    # 4. Instantiate Landlab components
    # Flow Accumulators
    # We will use D8 for channel routing
    fa_d8 = FlowAccumulator(grid, flow_director='FlowDirectorD8')
    # Run D8 first to initialize the D8-shaped flow fields so the grid has them
    fa_d8.run_one_step()

    # To isolate BedrockLandslider, we will create a temporary grid each time step for it.

    # SPACE for fluvial erosion, sediment transport routing and tools vs cover tradeoff
    # The Shobe et al. (2017) paper notes that SPACE natively incorporates the "cover effect"
    # where sediment thickness inhibits bedrock erosion, but standard parameters might not reflect
    # the "tools effect" (enhancement of bedrock erosion by mobile sediment).
    # Setting sp_crit_br > 0 can help approximate some transport thresholds, but the core equations
    # handle the transition automatically. We use standard parameters.
    sp = Space(
        grid,
        K_sed=args.k_sed,
        K_br=args.k_br,
        F_f=0.0,
        phi=0.0,
        H_star=1.0,
        v_s=0.001,
        m_sp=0.5,
        n_sp=1.0,
        sp_crit_sed=0,
        sp_crit_br=0
    )

    # TaylorNonLinearDiffuser for soil creep
    ld = TaylorNonLinearDiffuser(grid, linear_diffusivity=args.diffusivity, slope_crit=0.6)

    # 5. Main simulation loop
    dt = args.dt
    total_time = args.time
    uplift_rate = args.uplift_rate

    # Calculate number of iterations
    num_iterations = int(total_time / dt)

    print(f"Starting simulation for {total_time} years with dt={dt} years ({num_iterations} iterations)...")
    for i in range(num_iterations):
        # 5a. Apply tectonic uplift to core nodes
        grid.at_node['bedrock__elevation'][grid.core_nodes] += uplift_rate * dt
        grid.at_node['topographic__elevation'][grid.core_nodes] += uplift_rate * dt

        # 5b. Run hillslope diffusion
        ld.run_one_step(dt)

        # Ensure topography is strictly bed + soil before creating grid_mfd
        grid.at_node['topographic__elevation'][:] = grid.at_node['bedrock__elevation'][:] + grid.at_node['soil__depth'][:]

        # 5c. Run geotechnical landslide model on a dynamically isolated MFD grid

        # D8 routing on the main grid first to ensure `flow__` fields are updated for the timestep
        fa_d8.run_one_step()

        # Create a fresh grid each timestep for MFD routing to guarantee no field shape clashes
        grid_mfd = RasterModelGrid((args.rows, args.cols), xy_spacing=args.dx)
        grid_mfd.set_closed_boundaries_at_grid_edges(True, True, True, False)

        grid_mfd.add_field("topographic__elevation", grid.at_node["topographic__elevation"].copy(), at="node", clobber=True)
        grid_mfd.add_field("soil__depth", grid.at_node["soil__depth"].copy(), at="node", clobber=True)
        grid_mfd.add_field("bedrock__elevation", grid.at_node["bedrock__elevation"].copy(), at="node", clobber=True)
        grid_mfd.add_zeros('flood_status_code', at='node', dtype=int)

        # Run MFD to generate 2D fields
        from landlab.components import FlowDirectorMFD
        fd_mfd = FlowDirectorMFD(grid_mfd, diagonals=True)
        fd_mfd.run_one_step()

        # BedrockLandslider needs the 2D MFD fields explicitly prefixed with 'hill_'
        grid_mfd.add_field('hill_flow__receiver_node', grid_mfd.at_node['flow__receiver_node'].copy(), at='node', clobber=True)
        grid_mfd.add_field('hill_flow__receiver_proportions', grid_mfd.at_node['flow__receiver_proportions'].copy(), at='node', clobber=True)
        grid_mfd.add_field('hill_topographic__steepest_slope', grid_mfd.at_node['topographic__steepest_slope'].copy(), at='node', clobber=True)

        # BedrockLandslider ALSO requires the 1D D8 fields without a prefix for channel routing!
        # First, delete the newly generated 2D fields from grid_mfd
        for field in ['flow__receiver_node', 'flow__receiver_proportions', 'topographic__steepest_slope',
                      'flow__link_to_receiver_node', 'flow__sink_flag']:
            if field in grid_mfd.at_node:
                grid_mfd.delete_field('node', field)

        # Now copy the 1D fields from the main grid (D8)
        grid_mfd.add_field('flow__receiver_node', grid.at_node['flow__receiver_node'].copy(), at='node', clobber=True)
        grid_mfd.add_field('flow__upstream_node_order', grid.at_node['flow__upstream_node_order'].copy(), at='node', clobber=True)
        grid_mfd.add_field('topographic__steepest_slope', grid.at_node['topographic__steepest_slope'].copy(), at='node', clobber=True)

        # Instantiate and run landslider safely on this clean, mixed-field grid
        landslider = BedrockLandslider(grid_mfd, angle_int_frict=1.0, cohesion_eff=1e4)
        landslider.run_one_step(dt=dt)

        # Sync modified fields back to main grid
        grid.at_node['topographic__elevation'][:] = grid_mfd.at_node['topographic__elevation'][:]
        grid.at_node['soil__depth'][:] = grid_mfd.at_node['soil__depth'][:]
        grid.at_node['bedrock__elevation'][:] = grid_mfd.at_node['bedrock__elevation'][:]

        # Cleanup to free memory
        del grid_mfd
        del fd_mfd
        del landslider

        # 5d. Run D8 routing and fluvial erosion on main grid
        fa_d8.run_one_step()
        sp.run_one_step(dt=dt)

        # Maintain consistent bedrock and topographic elevation relation for BedrockLandslider
        grid.at_node['topographic__elevation'][:] = grid.at_node['bedrock__elevation'][:] + grid.at_node['soil__depth'][:]

        # Keep soil depth consistent if needed, though Space updates bedrock, soil, and topo

        if (i + 1) % max(1, num_iterations // 10) == 0:
            print(f"Completed {i + 1} iterations ({(i + 1) * dt} years)")

    print("Simulation complete.")

    # 6. Post-processing visualization
    plt.figure(figsize=(10, 8))
    imshow_grid(grid, "topographic__elevation", cmap="terrain", colorbar_label="Elevation (m)")
    plt.title(f"Landscape Elevation after {total_time} years")
    plt.savefig(args.output_plot)
    print(f"Saved landscape plot to {args.output_plot}")

if __name__ == "__main__":
    main()
