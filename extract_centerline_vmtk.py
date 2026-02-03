import sys
import vtk
from vmtk import vmtkscripts
import pyvista as pv

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_centerline_vmtk.py <input_stl>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    print("Reading surface...")
    reader = vmtkscripts.vmtkSurfaceReader()
    reader.InputFileName = input_file
    reader.Execute()
    
    print("Calculating Centerlines...")
    print("Note: You may need to click 'Source' and 'Target' points in the interactive window if SeedSelector is 'pick'.")
    
    centerlines = vmtkscripts.vmtkCenterlines()
    centerlines.Surface = reader.Surface
    # "openprofiles" automates finding the ends for open tubes.
    # If the mesh is closed (caps at ends), use "pick" or "idlist".
    centerlines.SeedSelectorName = 'openprofiles' 
    centerlines.Execute()
    
    # Smoothing
    print("Smoothing centerline...")
    smoother = vmtkscripts.vmtkCenterlineSmoothing()
    smoother.Centerlines = centerlines.Centerlines
    smoother.Execute()
    
    # Save
    writer = vmtkscripts.vmtkSurfaceWriter()
    writer.Input = smoother.Centerlines
    writer.OutputFileName = "centerline_vmtk.vtp"
    writer.Execute()
    print("Saved to centerline_vmtk.vtp")
    
    # Visualize using PyVista
    print("Visualizing...")
    mesh = pv.read(input_file)
    cl = pv.read("centerline_vmtk.vtp")
    
    p = pv.Plotter()
    p.add_mesh(mesh, opacity=0.3, color='tan')
    p.add_mesh(cl, color='blue', line_width=5)
    p.show()

if __name__ == "__main__":
    main()
