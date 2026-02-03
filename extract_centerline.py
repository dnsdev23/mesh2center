import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
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
        density = max_dim / 150.0 # Target roughly 150 voxels along longest axis
        print(f"Auto-calculated voxel density: {density:.4f}")

    voxels = pv.voxelize(mesh, density=density, check_surface=False)
    
    # Create a grid from these voxels
    # PyVista voxelize returns an UnstructuredGrid where cells are voxels.
    # We need to map this to a dense numpy array for skimage. 
    # This part can be tricky. A simpler way for 'robustness' with simply connected shapes 
    # is to create a dense grid and check points inside.
    
    # Let's use a simpler approach: Uniform Grid Resampling
    # Create the grid bounds
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    
    dims = (np.array([x_max-x_min, y_max-y_min, z_max-z_min]) / density).astype(int) + 2
    grid = pv.UniformGrid(
        dimensions=dims,
        spacing=(density, density, density),
        origin=(x_min, y_min, z_min)
    )
    
    # Select points inside surface
    selection = grid.select_enclosed_points(mesh, check_surface=False)
    mask = selection['SelectedPoints'].reshape(dims[::-1]) # PyVista/VTK uses Fortran order? No, C order but check dims
    # Actually 'select_enclosed_points' adds 'SelectedPoints' array.
    # We need to ensure the reshape is correct.
    mask = selection['SelectedPoints'].view(bool).reshape(grid.dimensions[2], grid.dimensions[1], grid.dimensions[0])
    # Note: VTK usually uses (x,y,z) for dimensions but numpy uses (z,y,x) for indexing.
    # Verification needed on real data, but (z, y, x) is standard for 3D numpy arrays.
    
    return mask, grid

def extract_centerline(mask):
    """
    Extracts centerline from binary mask using skeletonization.
    """
    print("Skeletonizing...")
    skeleton = skeletonize(mask)
    return skeleton

def skeleton_to_graph(skeleton):
    """
    Converts a 3D skeleton (binary numpy array) to a NetworkX graph.
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
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    neighbor = (nz+dz, ny+dy, nx_coord+dx)
                    if neighbor in node_set:
                        G.add_edge(node, neighbor, weight=np.sqrt(dz**2 + dy**2 + dx**2))
                        
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
    
    # 2. Skeletonize
    skeleton = extract_centerline(mask)
    
    # 3. Graph & Path
    G = skeleton_to_graph(skeleton)
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
    smoothed_path = smooth_path(world_path, window_size=15)

    # Create Spline
    print("Creating visualization...")
    # pv.Spline fits a spline through the points
    try:
        spline = pv.Spline(smoothed_path, 1000)
    except:
        # Fallback if too few points
        spline = pv.lines_from_points(smoothed_path)

    # Visualization
    p = pv.Plotter()
    p.add_mesh(mesh, style='wireframe', opacity=0.3, color='tan', label='Colon Surface')
    p.add_mesh(spline, color='red', line_width=5, render_lines_as_tubes=True, label='Centerline')
    p.add_legend()
    p.show()
    
    # Save
    spline.save(args.output)
    print(f"Saved centerline to {args.output}")

if __name__ == "__main__":
    main()
