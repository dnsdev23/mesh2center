import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_dilation, binary_erosion
import networkx as nx
import argparse

def voxelize_mesh(mesh, density=None):
    """
    Voxelizes the mesh.
    If density is None, auto-calculates based on bounding box to have roughly 100^3 voxels.
    """
    if density is None:
        # Auto-calculate density
        bounds = mesh.bounds
        max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        # User requested smaller voxel size. 
        # Previous: max_dim / 150.0 (~2.23mm)
        # New: max_dim / 250.0 (~1.34mm) - Finer resolution
        density = max_dim / 250.0 
        print(f"Auto-calculated voxel density: {density:.4f} (Target ~250 voxels)")

    # voxels = mesh.voxelize(density=density, check_surface=False) # Removed as unused
    # (Unused code block removed to fix error and clean up)

    
    # Create a grid from these voxels
    # PyVista voxelize returns an UnstructuredGrid where cells are voxels.
    # We need to map this to a dense numpy array for skimage. 
    # This part can be tricky. A simpler way for 'robustness' with simply connected shapes 
    # is to create a dense grid and check points inside.
    
    # Let's use a simpler approach: Uniform Grid Resampling
    # Create the grid bounds
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    
    dims = (np.array([x_max-x_min, y_max-y_min, z_max-z_min]) / density).astype(int) + 2
    grid = pv.ImageData(
        dimensions=dims,
        spacing=(density, density, density),
        origin=(x_min, y_min, z_min)
    )
    
    # vvv ALTERNATIVE VOXELIZATION: SUBDIVISION + FILL vvv
    # Implicit distance is O(N_voxels * M_triangles) which is too slow (hours) for large grids/meshes.
    # Instead, we identify boundary voxels by mapping mesh points to the grid.
    
    print(f"Identifying surface voxels (Grid: {grid.dimensions})...")
    
    # 1. Ensure mesh has enough points to cover every intersecting voxel
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
        
    print("Subdividing mesh to densify surface points...")
    # Subdivide to ensuring point spacing < voxel size
    # For large meshes, level 3 might create too many points. Level 2 is usually safe for 100^3 grids.
    # Let's try 2 first to avoid OOM, but maybe check point count?
    if mesh.n_points < 1000000:
        dense_mesh = mesh.subdivide(2, subfilter='linear')
    else:
        dense_mesh = mesh # Already dense enough
    
    # 2. Map points to voxel indices
    points = dense_mesh.points
    # Indices = (points - origin) / spacing
    # origin is (x_min, y_min, z_min)
    indices = ((points - np.array([x_min, y_min, z_min])) / density).astype(int)
    
    # Filter out-of-bounds indices
    valid_filter = (indices[:, 0] >= 0) & (indices[:, 0] < dims[0]) & \
                   (indices[:, 1] >= 0) & (indices[:, 1] < dims[1]) & \
                   (indices[:, 2] >= 0) & (indices[:, 2] < dims[2])
    valid_indices = indices[valid_filter]
    
    # 3. Create the shell mask
    # Initialize empty grid z,y,x
    shell_mask = np.zeros(dims[::-1], dtype=bool) 
    
    # Set shell voxels to True
    # indices are (x, y, z), mask is (z, y, x)
    shell_mask[valid_indices[:, 2], valid_indices[:, 1], valid_indices[:, 0]] = True
    
    print("Dilating shell to close small gaps...")
    # 4. Dilate to ensure the shell is watertight (to allow filling)
    # This DOES grow outward, which connects gaps but also thickens the object.
    iterations = 2
    closed_shell = binary_dilation(shell_mask, iterations=iterations)
    
    print("Filling tubular structure (binary_fill_holes)...")
    # 5. Fill the interior to get a solid volume
    filled_mask = binary_fill_holes(closed_shell)
    
    # 6. Erode back the dilation amount to restore original volume size
    # We grew OUT by 'iterations', so now we shrink IN by same amount.
    # This effectively makes the operation "Closing" (Dilation -> Erosion), 
    # but with a Fill step in between to capture the interior.
    print(f"Eroding back {iterations} iterations to prevent surface sticking...")
    mask = binary_erosion(filled_mask, iterations=iterations)
    
    return mask, grid


def extract_centerline(mask):
    """
    Extracts centerline from binary mask using skeletonization.
    """
    print("Skeletonizing...")
    skeleton = skeletonize(mask)
    return skeleton


def skeleton_to_graph(skeleton, dt=None):
    """
    Converts a 3D skeleton (binary numpy array) to a NetworkX graph.
    If dt (Distance Transform) is provided, weighs edges to penalize proximity to walls.
    """
    print("Converting skeleton to graph...")
    # Get coordinates of skeleton points
    z, y, x = np.where(skeleton)
    nodes = list(zip(z, y, x))
    node_set = set(nodes)
    
    G = nx.Graph()
    
    for node in nodes:
        G.add_node(node)
        # Check 26-connectivity
        nz, ny, nx_coord = node
        
        # Get DT value for this node (measure of "centeredness")
        node_radius = 1.0
        if dt is not None:
            node_radius = dt[node]

        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    neighbor = (nz+dz, ny+dy, nx_coord+dx)
                    if neighbor in node_set:
                        # Euclidean distance
                        dist = np.sqrt(dz**2 + dy**2 + dx**2)
                        
                        # Weighting strategy:
                        # We want the path to stay in the center (high DT).
                        # Standard shortest path minimizes Sum(weights).
                        # If we set Weight = Length / DT, the algorithm will prefer 
                        # slightly longer paths that stay in high-DT regions (center) 
                        # over short paths that hug the walls (low DT).
                        
                        # Averaging DT of strictly adjacent helps smoothness
                        if dt is not None:
                            neighbor_radius = dt[neighbor]
                            avg_radius = (node_radius + neighbor_radius) / 2.0
                            # Weight inversely proportional to radius
                            weight = dist / (avg_radius + 1e-6)
                        else:
                            weight = dist
                            
                        G.add_edge(node, neighbor, weight=weight)
                        
    return G


def find_longest_path(G):
    """
    Finds the longest path in the skeleton graph (end-to-end extraction).
    Heuristic: Find the diameter of the graph (longest shortest path).
    """
    print("Finding longest path...")
    
    # Get connected components (there might be noise)
    components = list(nx.connected_components(G))
    if not components:
        return []
    
    # Assume the largest component is the colon
    largest_comp = max(components, key=len)
    subgraph = G.subgraph(largest_comp)
    
    # Finding the true diameter is O(V*(V+E)), which can be slow.
    # Approximation: Start from a random node, find farthest, then find farthest from there.
    # 1. Pick arbitrary node
    start_node = next(iter(subgraph.nodes))
    # 2. Find farthest from start_node
    lengths = nx.single_source_dijkstra_path_length(subgraph, start_node)
    farthest_node_1 = max(lengths, key=lengths.get)
    # 3. Find farthest from farthest_node_1
    lengths_2 = nx.single_source_dijkstra_path_length(subgraph, farthest_node_1)
    farthest_node_2 = max(lengths_2, key=lengths_2.get)
    
    # 4. Get path
    path = nx.shortest_path(subgraph, farthest_node_1, farthest_node_2)
    return path

def smooth_path(path, window_size=5):
    """
    Simple moving average smoothing.
    """
    if len(path) < window_size:
        return np.array(path)
    
    path_array = np.array(path)
    kernel = np.ones(window_size) / window_size
    
    smoothed_z = np.convolve(path_array[:, 0], kernel, mode='valid')
    smoothed_y = np.convolve(path_array[:, 1], kernel, mode='valid')
    smoothed_x = np.convolve(path_array[:, 2], kernel, mode='valid')
    
    return np.column_stack((smoothed_z, smoothed_y, smoothed_x))

def main():
    parser = argparse.ArgumentParser(description="Extract Centerline from Colon STL")
    parser.add_argument("input_file", help="Path to input STL file")
    parser.add_argument("--density", type=float, default=None, help="Voxel density (lower is coarser)")
    parser.add_argument("--output", default="centerline.vtk", help="Output filename")
    
    args = parser.parse_args()
    
    print(f"Loading {args.input_file}...")
    try:
        mesh = pv.read(args.input_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return

    # 1. Voxelize
    print("Voxelizing...")
    mask, grid = voxelize_mesh(mesh, args.density)
    
    # Visualization of Voxelized Colon
    print("Visualizing voxelized colon...")
    
    # Check if mask is empty
    if not np.any(mask):
        print("WARNING: Voxel mask is empty! Voxelization failed (threshold too strict or mesh has holes?).")
    else:
        print(f"Voxel mask has {np.sum(mask)} active voxels.")

    # Assign the mask to the grid point data
    grid.point_data["values"] = mask.flatten()
    
    try:
        # Show only points where value is 1 (inside)
        thresholded = grid.threshold(0.5)
        
        p_vox = pv.Plotter()
        p_vox.add_text("Voxelized Colon (Close window to continue)", position="upper_left", font_size=14)
        p_vox.add_mesh(thresholded, style='surface', color="tan", opacity=0.8, label="Filled Voxels")
        p_vox.add_mesh(mesh, style="wireframe", color="red", opacity=0.1, label="Original Mesh")
        p_vox.add_legend()
        p_vox.show()
    except Exception as e:
        print(f"Could not visualize voxels: {e}")
        import traceback
        traceback.print_exc()

    # 2. Skeletonize and DT
    print("Calculating Distance Transform...")
    dt = distance_transform_edt(mask)

    # 2b. Skeletonize
    skeleton = extract_centerline(mask)
    
    # 3. Graph & Path
    G = skeleton_to_graph(skeleton, dt=dt)
    path = find_longest_path(G)
    
    if not path:
        print("No path found!")
        return

    # Convert path back to world coordinates
    # Mask indices (z, y, x) -> Grid coordinates
    # Grid origin + index * spacing
    path_indices = np.array(path)
    # Remember our mask was (z, y, x) but spacing/origin are (x, y, z)
    # indices: z, y, x
    # we want: x, y, z world coords
    
    spacing = grid.spacing
    origin = grid.origin
    
    # x_world = origin_x + index_x * spacing_x
    world_path = []
    for z_idx, y_idx, x_idx in path_indices:
        x = origin[0] + x_idx * spacing[0]
        y = origin[1] + y_idx * spacing[1]
        z = origin[2] + z_idx * spacing[2]
        world_path.append([x, y, z])
    
    world_path = np.array(world_path)

    # 4. Smooth
    # Reduced window size to prevent cutting corners on sharp turns
    smoothed_path = smooth_path(world_path, window_size=5)
    
    # --- Interactive Editor ---
    print("Preparing interactive editor...")
    
    # Subsample points for Spline Widget handles (too many handles = slow/unusable)
    # User requested MORE control points.
    n_points = len(smoothed_path)
    n_handles = min(100, n_points) # Increased from 40 to 100 handles
    
    # Use linspace to pick indices uniformly
    idx = np.linspace(0, n_points - 1, n_handles).astype(int)
    control_points = smoothed_path[idx]
    
    # Container for the final result
    modified_spline_holder = [None]
    
    def on_spline_change(polydata):
        modified_spline_holder[0] = polydata

    # Visualization
    p = pv.Plotter()
    p.set_background('black')
    p.add_text("Interactive Mode: Drag spheres to adjust centerline.\nClose window to save.", position="upper_left", font_size=12, color='white')
    p.add_mesh(mesh, style='wireframe', opacity=0.1, color='tan', label='Colon Surface')
    
    # Add the spline widget
    widget_loaded = False
    try:
        # User requested handles to be VISIBLE.
        # We explicitly pass n_handles.
        widget = p.add_spline_widget(
            callback=on_spline_change, 
            initial_points=control_points,
            color="red",
            n_handles=n_handles 
        )
        widget_loaded = True
        print("Interactive widget added successfully.")
    except Exception as e:
        print(f"ERROR: add_spline_widget failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to static
        try:
            spline = pv.Spline(smoothed_path, 1000)
        except:
            spline = pv.lines_from_points(smoothed_path)
        p.add_mesh(spline, color='red', line_width=5, render_lines_as_tubes=True, label='Centerline (Static)')
        modified_spline_holder[0] = spline
    
    # Visual check: Show where the control points should be
    # ALWAYS show these now, to debug if the code is even reaching here.
    print(f"Adding debug spheres... Widget loaded: {widget_loaded}")
    
    # Use physical spheres to ensure visibility regardless of camera distance
    # User requested BIGGER spheres.
    radius = grid.spacing[0] * 5.0 
    
    # Create a single mesh for all debug spheres
    debug_cloud = pv.PolyData(control_points)
    # Use rendered points (screen space) - Normal visible size
    p.add_mesh(debug_cloud, color="yellow", point_size=10, render_points_as_spheres=True, label="Control Points", pickable=False)
    
    # Green spheres removed per user request.
    
    # Note: If the widget handles are too small to see, the user can try clicking/dragging
    # near the big green debug spheres, as they are coincident.

    p.add_legend()
    p.show()
    
    # Save Logic
    if modified_spline_holder[0] is not None:
        print("Saving current spline state.")
        raw_spline = modified_spline_holder[0]
        
        # User requested smoother/finer output.
        # The interactive widget might output a curve with limited resolution.
        # We resample it to a high-resolution spline (3000 points) to ensure high smoothness.
        try:
            # We assume raw_spline.points are ordered along the curve.
            final_spline = pv.Spline(raw_spline.points, 3000)
        except Exception as e:
            print(f"Warning: Could not resample spline ({e}), saving interactive result directly.")
            final_spline = raw_spline
    else:
        # If callback didn't fire (e.g. user just looked and closed), create default spline
        print("No changes detected. Saving original calculation.")
        try:
            final_spline = pv.Spline(smoothed_path, 3000)
        except:
            final_spline = pv.lines_from_points(smoothed_path)

    final_spline.save(args.output)
    print(f"Saved centerline to {args.output}")

    # Final Visualization of the Saved Result
    print("Displaying final saved centerline...")
    p_final = pv.Plotter()
    p_final.set_background('black')
    p_final.add_text("Final Centerline Result", position="upper_left", font_size=14, color='white')
    p_final.add_mesh(mesh, style='wireframe', opacity=0.1, color='tan', label='Colon Surface')
    p_final.add_mesh(final_spline, color='red', line_width=5, render_lines_as_tubes=True, label='Final Centerline')
    p_final.add_legend()
    p_final.show()

if __name__ == "__main__":
    main()
