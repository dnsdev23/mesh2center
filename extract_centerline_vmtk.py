
import sys
import os

def main():
    # Try importing VMTK
    try:
        import vmtk
        from vmtk import vmtkscripts
        HAS_VMTK = True
    except ImportError:
        HAS_VMTK = False
    
    if HAS_VMTK:
        # Original VMTK implementation
        import vtk
        import pyvista as pv
        
        if len(sys.argv) < 2:
            print("Usage: python extract_centerline_vmtk.py <input_stl>")
            sys.exit(1)
            
        input_file = sys.argv[1]
        
        print("Reading surface with VMTK...")
        reader = vmtkscripts.vmtkSurfaceReader()
        reader.InputFileName = input_file
        reader.Execute()
        
        print("Calculating Centerlines...")
        centerlines = vmtkscripts.vmtkCenterlines()
        centerlines.Surface = reader.Surface
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
        
    else:
        # Fallback to extract_centerline.py (scikit-image based)
        print("!"*60)
        print("VMTK module not found (likely incompatible with current Python environment).")
        print("Falling back to robust scikit-image skeletonization from extract_centerline.py")
        print("!"*60)
        
        try:
            # Manually load module from file path to avoid import caching issues or path issues
            import importlib.util
            spec = importlib.util.spec_from_file_location("extract_centerline", "extract_centerline.py")
            extract_centerline = importlib.util.module_from_spec(spec)
            sys.modules["extract_centerline"] = extract_centerline
            spec.loader.exec_module(extract_centerline)
        except Exception as e:
            print(f"Error: Could not import extract_centerline.py: {e}")
            sys.exit(1)
            
        # Adjust arguments for extract_centerline.main()
        # It expects: script.py input_file [--output OUTPUT]
        
        if len(sys.argv) < 2:
            print("Usage: python extract_centerline_vmtk.py <input_stl>")
            sys.exit(1)
            
        original_args = sys.argv
        input_file = original_args[1]
        
        # Override sys.argv to pass to the other script's argparse
        # We want to force output to centerline_vmtk.vtk (PyVista saves based on extension)
        sys.argv = [original_args[0], input_file, "--output", "centerline_vmtk.vtk"]
        
        print(f"Running fallback extraction on {input_file}...")
        extract_centerline.main()
        
if __name__ == "__main__":
    main()
