import numpy as np
import pyvista as pv
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, binary_fill_holes, binary_dilation, binary_erosion
import networkx as nx
import argparse
import os
import re
import tkinter as tk
from tkinter import filedialog

# Initialize tkinter root mostly to hide it
root = tk.Tk()
root.withdraw()

try:
    import vtk
except ImportError:
    pass # vtk is usually available if pyvista is installed

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
    
    # Create a grid from these voxels
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

def get_next_filename(base_path):
    """
    Helper to generate auto-incrementing filenames.
    Pattern: {original_name}_curve_{number}.vtp
    """
    # Prefer VTP (XML PolyData) for Omniverse/modern VTK support over legacy VTK
    # But user asked for "best curve format". USD is native to Omniverse, but not easily writable via PyVista without plugin.
    # VTP is standard VTK XML. OBJ is common but loses some data.
    # Let's stick to .vtp (XML PolyData) which is robust. 
    # Omniverse reads many formats.
    
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    # Check if name already follows pattern
    # We want: Name_curve_N.vtp
    
    pattern = re.compile(rf"{re.escape(name_no_ext)}_curve_(\d+)\.vtp")
    
    max_count = 0
    if os.path.exists(directory):
        for f in os.listdir(directory):
            match = pattern.match(f)
            if match:
                count = int(match.group(1))
                if count > max_count:
                    max_count = count
    
    new_count = max_count + 1
    new_name = f"{name_no_ext}_curve_{new_count}.vtp"
    return os.path.join(directory, new_name)

class CenterlineEditor:
    """
    Manages the interactive spline editor with Undo/Redo/Save support.
    """
    def __init__(self, plotter, initial_points, input_stl_path, voxel_grid=None, mesh=None):
        self.p = plotter
        self.input_stl_path = input_stl_path
        self.voxel_grid = voxel_grid
        self.mesh = mesh
        
        # State Management
        self.undo_stack = []
        self.redo_stack = []
        self.current_points = np.array(initial_points)
        
        # Add the widget
        try:
            self.widget = self.p.add_spline_widget(
                callback=self.on_spline_change,
                initial_points=initial_points,
                color="red",
                n_handles=len(initial_points)
            )
            self.widget.AddObserver("EndInteractionEvent", self.on_interaction_end)
            self.widget_loaded = True
        except Exception as e:
            print(f"Error initializing spline widget: {e}")
            self.widget_loaded = False
        
        # Disable Handle Highlighting (prevent size change on click)
        if self.widget_loaded:
            # The widget uses vtkProperty for handles. 
            # We want Selected property to match Normal property.
            prop = self.widget.GetHandleProperty()
            selected_prop = self.widget.GetSelectedHandleProperty()
            
            # FORCE POINTS REPRESENTATION (to avoid large spheres)
            # By default it might use sphere source.
            prop.SetPointSize(8)
            selected_prop.SetPointSize(8)
            
            # Shrink spheres if using 3D handles
            # Default is relative to scene? 
            self.widget.SetHandleSize(0.006)
            
            # Also try to access visual representation to force it
            # (Limitation: PyVista add_spline_widget abstracts the representation)
            
            # Copy basic style to selected
            selected_prop.DeepCopy(prop)
            pass
            
        # Bind Keys
        self.p.add_key_event('z', self.undo)
        self.p.add_key_event('y', self.redo)
        self.p.add_key_event('s', self.save_smart)
        self.p.add_key_event('a', self.add_points_action)
        
        # UI Buttons (2D actors)
        self.setup_ui()

    def setup_ui(self):
        # Helper to add a "Text Button"
        # We store the bounds for click detection
        # Normalized coordinates (0-1)
        # Position is Bottom-Left of the text
        self.btn_regions = []
        
        def add_btn(label, y_pos, callback):
            # Enforce Uniform Width by padding with spaces
            # Target width: 10 chars (Open Curve is exactly 10)
            target_len = 10
            padded_label = f"{label:^{target_len}}"
            
            # 1. Add Text
            actor = self.p.add_text(
                padded_label, 
                position=(0.02, y_pos), 
                viewport=True,
                color='white',
                font_size=10,
                font='courier',
                shadow=True
            )
            # 2. Style as "Button" (Background)
            prop = actor.GetTextProperty()
            prop.SetBackgroundColor(0.5, 0.5, 0.5) # Grey
            prop.SetBackgroundOpacity(1.0)
            prop.SetFrame(True)
            prop.SetFrameWidth(2)
            prop.SetFrameColor(0.8, 0.8, 0.8)
            
            # Store y_pos for hit testing
            # Note: We store the PADDED label so hit testing uses full width
            self.btn_regions.append({'actor': actor, 'callback': callback, 'label': padded_label, 'y_pos': y_pos})
            
        # Layout: Normalized Coordinates (0=Bottom, 1=Top)
        start_y = 0.90
        step = 0.07
        
        add_btn("Undo (Z)", start_y, self.undo)
        add_btn("Redo (Y)", start_y - step, self.redo)
        add_btn("Save As (S)", start_y - 2*step, self.save_smart)
        add_btn("Volume", start_y - 3*step, self.toggle_voxel_window_action)
        add_btn("Open Curve", start_y - 4*step, self.open_curve_window)
        add_btn("Add Pts", start_y - 5*step, self.add_points_action)

        # Bind Click Event with very high priority
        self.p.iren.add_observer("LeftButtonPressEvent", self.on_click, 10.0)
        
    def toggle_voxel_window_action(self):
        self.open_voxel_window(True)

    def on_click(self, obj, event):
        click_pos = self.p.iren.get_event_position() # (x, y) pixels
        
        # DEBUG: print(f"Click at: {click_pos}")
        
        matched = False
        for btn in self.btn_regions:
            actor = btn.get('actor')
            if not actor: continue
            
            renderer = self.p.renderer
            
            # Get bottom-left of text in pixels
            pos_px = actor.GetPositionCoordinate().GetComputedDisplayValue(renderer)
            
            # Robust bounds calculation
            # Text actors in VTK are anchored at Bottom-Left.
            # We'll use a wider hit box for reliability.
            char_w = 12 # Slightly wider for safety
            text_w = len(btn['label']) * char_w
            text_h = 40 # Taller hit box
            
            # Check bounds (with a small buffer)
            is_x_match = (pos_px[0] - 10 <= click_pos[0] <= pos_px[0] + text_w + 10)
            is_y_match = (pos_px[1] - 10 <= click_pos[1] <= pos_px[1] + text_h + 10)
            
            if is_x_match and is_y_match:
                print(f"Confirmed Button Click: {btn['label'].strip()}")
                
                # Ensure dialogs show up on top
                root.deiconify()
                root.lift()
                root.focus_force()
                root.withdraw() # Keep it hidden but active
                
                try:
                    btn['callback']()
                except Exception as e:
                    print(f"Error executing {btn['label']}: {e}")
                
                matched = True
                break
        
        if matched:
            obj.SetAbortEvent(1)

    # Removed old widget callbacks (undo_callback, etc are now direct)
    
    def get_handles(self):
        if not self.widget_loaded:
            return self.current_points
            
        n = self.widget.GetNumberOfHandles()
        points = []
        for i in range(n):
            pos = self.widget.GetHandlePosition(i)
            points.append(pos)
        return np.array(points)

    def set_handles(self, points):
        if not self.widget_loaded:
            return
            
        n_new = len(points)
        self.widget.SetNumberOfHandles(n_new)
        for i in range(n_new):
            self.widget.SetHandlePosition(i, points[i][0], points[i][1], points[i][2])
        
        # Trigger update of the spline visual and force redraw
        self.widget.InvokeEvent("InteractionEvent") 
        self.p.render()

    def on_spline_change(self, polydata):
        # Continuous drag - do not save state here
        pass

    def on_interaction_end(self, obj, event):
        # Called when user releases mouse after dragging handle
        
        # BRUTE FORCE: Reset handle size to prevent it from growing
        self.widget.SetHandleSize(0.006)
        
        new_points = self.get_handles()
        
        # Check if actually changed
        if np.array_equal(new_points, self.current_points):
            return
            
        # Push OLD state to Undo stack
        self.undo_stack.append(self.current_points)
        # Clear redo stack on new action
        self.redo_stack.clear()
        # Update current
        self.current_points = new_points
        
        print(f"Action recorded. Undo stack: {len(self.undo_stack)}")
        # User asked to "make change right away" - I assume this means visual update?
        # The widget updates itself visually.

    def undo(self):
        if not self.undo_stack:
            print("Nothing to undo.")
            return
            
        print("Undoing...")
        # Save current to Redo
        self.redo_stack.append(self.current_points)
        
        # Restore from Undo
        prev_points = self.undo_stack.pop()
        self.current_points = prev_points
        self.set_handles(prev_points)
        print("Undo complete.")

    def redo(self):
        if not self.redo_stack:
            print("Nothing to redo.")
            return

        print("Redoing...")
        # Save current to Undo
        self.undo_stack.append(self.current_points)
        
        # Restore from Redo
        next_points = self.redo_stack.pop()
        self.current_points = next_points
        self.set_handles(next_points)
        print("Redo complete.")

    def add_points_action(self):
        """
        Increases the number of control points by sampling the current spline.
        """
        print("Increasing control point density...")
        
        # Current handles
        old_points = self.get_handles()
        n_current = len(old_points)
        
        # Target: roughly 25% more points
        n_new = int(n_current * 1.25) + 2
            
        if n_new > 1000:
            print("Maximum handle limit reached (1000).")
            return

        # Use current handles to generate a smoother version for interpolation
        try:
            # We use a high res spline to sample points from
            temp_spline = pv.Spline(old_points, n_new)
            new_points = temp_spline.points
        except Exception as e:
            print(f"Error increasing points: {e}")
            return
            
        # Push to undo stack
        self.undo_stack.append(self.current_points)
        self.redo_stack.clear()
        
        # Update state and widget
        self.current_points = new_points
        self.set_handles(new_points)
        
        print(f"Points increased from {n_current} to {len(new_points)}")

    def generate_final_spline(self):
        # Reconstruct high-res spline from current handles
        try:
            return pv.Spline(self.current_points, 3000)
        except Exception:
            # Fallback
            return pv.lines_from_points(self.current_points)

    def save_smart(self):
        """
        Open file dialog with smart default name.
        """
        # 1. Generate default name
        default_path = get_next_filename(self.input_stl_path)
        default_dir = os.path.dirname(default_path)
        default_file = os.path.basename(default_path)
        
        # 2. Open Save As Dialog (using Tkinter hidden root)
        # Added CSV and OBJ support specifically as requested
        output_file = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_file,
            title="Save Centerline Curve",
            filetypes=[
                ("VTK XML PolyData", "*.vtp"), 
                ("CSV Points", "*.csv"),
                ("OBJ Wavefront", "*.obj"),
                ("VTK Legacy", "*.vtk")
            ]
        )
        
        if not output_file:
            print("Save cancelled.")
            return
            
        print(f"Saving to {output_file}...")
        spline = self.generate_final_spline()
        
        # Check extension
        ext = os.path.splitext(output_file)[1].lower()
        
        try:
            if ext == '.csv':
                # Save points as CSV: x, y, z
                # We use numpy.savetxt for efficiency and to avoid extra dependencies like pandas
                header = "x,y,z"
                np.savetxt(output_file, spline.points, delimiter=",", header=header, comments='')
                print(f"Saved {len(spline.points)} points to CSV.")
            elif ext == '.obj':
                # PyVista's save for .obj on PolyData containing lines works in recent versions
                # It writes 'v' (vertices) and 'l' (lines).
                spline.save(output_file)
                print(f"Saved to OBJ.")
            else:
                # Default VTK/VTP save
                spline.save(output_file)
                print(f"Saved to {ext.upper()}.")
            
            self.p.add_text(f"Saved: {os.path.basename(output_file)}", position="upper_right", font_size=16, color='green', name="save_message")
        except Exception as e:
            print(f"Error saving file: {e}")
            self.p.add_text(f"Error saving: {os.path.basename(output_file)}", position="upper_right", font_size=16, color='red', name="save_message")


    def open_voxel_window(self, state):
        if not state: return
        
        if self.voxel_grid is None or self.mesh is None:
            print("Voxel data not available.")
            return
            
        # Create a new Plotter for the voxel window
        # Note: This will block the main window content usually.
        pv_vox = pv.Plotter()
        pv_vox.set_background('black')
        pv_vox.add_text("Voxel Inspector", position="upper_left")
        
        # Actors
        # 1. Mesh (Wireframe)
        actor_mesh = pv_vox.add_mesh(self.mesh, style="wireframe", color="red", opacity=0.3, label="Mesh")
        
        # 2. Volume (Thresholded Grid)
        try:
            thresholded = self.voxel_grid.threshold(0.5)
            actor_vol = pv_vox.add_mesh(thresholded, style='surface', color="tan", opacity=0.8, label="Volume")
        except:
            actor_vol = None
            
        # Toggles
        def toggle_mesh(flag):
            if actor_mesh: actor_mesh.SetVisibility(flag)
            
        def toggle_vol(flag):
            if actor_vol: actor_vol.SetVisibility(flag)
            
        def toggle_both(flag):
            toggle_mesh(flag)
            toggle_vol(flag)
            
        # Add checkbox buttons for toggles
        size=40
        pv_vox.add_checkbox_button_widget(toggle_mesh, value=True, position=(10, 100), size=size, color_on='red', color_off='grey')
        pv_vox.add_text("Mesh", position=(60, 110), font_size=12)
        
        pv_vox.add_checkbox_button_widget(toggle_vol, value=True, position=(10, 50), size=size, color_on='tan', color_off='grey')
        pv_vox.add_text("Volume", position=(60, 60), font_size=12)

        pv_vox.show()

    def open_curve_window(self):
        # File Dialog
        filename = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.input_stl_path),
            title="Open Curve File",
            filetypes=[("VTK/VTP Files", "*.vtk *.vtp"), ("All Files", "*.*")]
        )
        if not filename:
            return

        print(f"Opening curve: {filename}")
        try:
            curve = pv.read(filename)
            
            p_curve = pv.Plotter()
            p_curve.set_background('black')
            p_curve.add_text(f"Curve: {os.path.basename(filename)}", position="upper_left")
            p_curve.add_mesh(self.mesh, style='wireframe', opacity=0.1, color='tan', label='Reference Mesh')
            p_curve.add_mesh(curve, color='cyan', line_width=4, render_lines_as_tubes=True, label='Loaded Curve')
            p_curve.add_legend()
            p_curve.show()
        except Exception as e:
            print(f"Error opening curve: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract Centerline from Colon STL")
    parser.add_argument("input_file", nargs='?', help="Path to input STL file")
    parser.add_argument("--density", type=float, default=None, help="Voxel density (lower is coarser)")
    parser.add_argument("--output", default="centerline.vtk", help="Output filename")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    
    # GUI Fallback: If no file provided, open picker
    if not input_file:
        print("No input file provided. Opening file picker...")
        input_file = filedialog.askopenfilename(
            title="Select Colon STL Mesh",
            filetypes=[("STL Files", "*.stl"), ("PLY Files", "*.ply"), ("OBJ Files", "*.obj"), ("All Files", "*.*")]
        )
        if not input_file:
            print("No file selected. Exiting.")
            return

    print(f"Loading {input_file}...")
    try:
        mesh = pv.read(input_file)
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
        
        # p_vox = pv.Plotter()
        # p_vox.add_text("Voxelized Colon (Close window to continue)", position="upper_left", font_size=14)
        # p_vox.add_mesh(thresholded, style='surface', color="tan", opacity=0.8, label="Filled Voxels")
        # p_vox.add_mesh(mesh, style="wireframe", color="red", opacity=0.1, label="Original Mesh")
        # p_vox.add_legend()
        # p_vox.show()
        pass
    except Exception as e:
        pass
        # print(f"Could not visualize voxels: {e}")
        # import traceback
        # traceback.print_exc()

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
    
    # --- Interactive Editor Refactored ---
    print("Preparing interactive editor...")
    
    n_points = len(smoothed_path)
    n_handles = min(80, n_points) 
    
    idx = np.linspace(0, n_points - 1, n_handles).astype(int)
    control_points = smoothed_path[idx]
    
    # Setup Plotter
    p = pv.Plotter()
    p.set_background('black')
    p.add_mesh(mesh, style='wireframe', opacity=0.1, color='tan', label='Colon Surface')
    
    # Initialize Editor Class
    # IMPORTANT: Use 'input_file' which we may have gotten from GUI picker
    editor = CenterlineEditor(p, control_points, input_file, voxel_grid=grid, mesh=mesh)
    
    # Add text instructions
    # (Removed legacy instructions)
    
    # Yellow spheres to show original calculation reference
    debug_cloud = pv.PolyData(control_points)
    p.add_mesh(debug_cloud, color="yellow", point_size=5, render_points_as_spheres=True, label="Original Reference", pickable=False)
    
    p.add_legend()
    p.show()
    
    # Simple autosave verification:
    # We always save "latest_autosave.vtp" on exit, just in case.
    print("Saving autosaves of final state (VTP, CSV, OBJ)...")
    try:
        final_spline = editor.generate_final_spline()
        # Save VTP
        final_spline.save("latest_autosave.vtp")
        
        # Save CSV (x,y,z points)
        np.savetxt("latest_autosave.csv", final_spline.points, delimiter=",", header="x,y,z", comments='')
        
        # Save OBJ (wavefront)
        final_spline.save("latest_autosave.obj")
        
        # Also save to the path specified in --output if provided
        if args.output and args.output != "centerline.vtk":
            ext = os.path.splitext(args.output)[1].lower()
            if ext == '.csv':
                np.savetxt(args.output, final_spline.points, delimiter=",", header="x,y,z", comments='')
            else:
                final_spline.save(args.output)
            print(f"Final result saved to {args.output}")
            
        print("Autosaves complete.")
    except Exception as e:
        print(f"Error saving autosave/output: {e}")

if __name__ == "__main__":
    main()
